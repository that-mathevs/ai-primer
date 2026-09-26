"""Specification for primer.ml.efficient_architectures: making long context affordable."""

import numpy as np
import pytest

from primer.ml.attention import causal_mask, scaled_dot_product_attention
from primer.ml.efficient_architectures import (
    LatentKVAttention,
    cache_bytes_by_method,
    context_cost,
    discretize,
    feature_map,
    global_local_mask,
    hybrid_cache_bytes,
    latent_kv_bytes_per_token,
    latent_kv_worked_example,
    linear_attention_quadratic,
    linear_attention_recurrent,
    pairs_computed,
    parallel_scan,
    quantize_kv,
    quantized_cache_error,
    quantized_kv_bytes_per_token,
    reach,
    receptive_field,
    selective_recall,
    sliding_window_decode,
    sliding_window_mask,
    ssm_convolution,
    ssm_kernel,
    ssm_recurrent,
    ssm_state_bytes,
    strided_mask,
    time_invariant_recall,
)

GIB = 1e9


class TestTheCostOfLongContext:
    def test_given_a_llama_3_8b_shaped_model_the_kv_cache_for_128k_tokens_is_about_17_gigabytes(self):
        # 2 × 32 layers × 8 KV heads × 128 dims × 2 bytes = 131,072 bytes per token, times 131,072 tokens.
        assert context_cost(131_072)["kv_cache_bytes"] / GIB == pytest.approx(17.18, abs=0.01)

    def test_given_a_million_tokens_the_kv_cache_outgrows_an_80_gigabyte_gpu(self):
        assert context_cost(1_048_576)["kv_cache_bytes"] / GIB == pytest.approx(137.4, abs=0.1)

    def test_given_eight_times_the_context_the_pairs_scored_grow_sixty_four_fold(self):
        small, large = context_cost(8_192), context_cost(65_536)
        assert large["pairs_per_head_per_layer"] / small["pairs_per_head_per_layer"] == 64


class TestSlidingWindowAttention:
    def test_given_a_window_of_three_a_token_sees_itself_and_the_two_before_it(self):
        assert sliding_window_mask(8, 3)[5].tolist() == [False, False, False, True, True, True, False, False]

    def test_given_eight_tokens_and_a_window_of_three_21_of_the_36_causal_pairs_are_scored(self):
        # 1 + 2 + 3 × 6 = 21 pairs, against 8 × 9 / 2 = 36 for full causal attention.
        assert pairs_computed(sliding_window_mask(8, 3)) == 21
        assert pairs_computed(causal_mask(8)) == 36

    def test_given_a_window_as_long_as_the_sequence_it_is_ordinary_causal_attention(self):
        rng = np.random.default_rng(0)
        Q, K, V = (rng.standard_normal((6, 4)) for _ in range(3))
        windowed, _ = scaled_dot_product_attention(Q, K, V, mask=sliding_window_mask(6, 6))
        causal, _ = scaled_dot_product_attention(Q, K, V, mask=causal_mask(6))
        assert np.allclose(windowed, causal)

    def test_given_three_layers_of_window_three_the_last_token_hears_six_positions_back_but_not_seven(self):
        # Each layer carries information w − 1 = 2 positions further back: 3 layers reach 6.
        heard = receptive_field(8, w=3, layers=3)[7]
        assert heard[1] and not heard[0]

    def test_given_32_layers_and_a_4096_token_window_information_can_travel_about_131_thousand_tokens(self):
        # Mistral 7B's numbers: 32 × 4,095.
        assert reach(w=4096, layers=32) == 131_040

    def test_given_a_rolling_buffer_decoding_matches_the_masked_computation(self):
        rng = np.random.default_rng(1)
        Q, K, V = (rng.standard_normal((12, 4)) for _ in range(3))
        streamed, _ = sliding_window_decode(Q, K, V, w=4)
        masked, _ = scaled_dot_product_attention(Q, K, V, mask=sliding_window_mask(12, 4))
        assert np.allclose(streamed, masked)

    def test_given_a_rolling_buffer_it_never_holds_more_than_w_keys(self):
        rng = np.random.default_rng(1)
        Q, K, V = (rng.standard_normal((50, 4)) for _ in range(3))
        assert sliding_window_decode(Q, K, V, w=4)[1] == 4


class TestSparseAttentionPatterns:
    def test_given_a_global_first_token_every_later_token_can_attend_to_it(self):
        assert global_local_mask(16, w=4, global_tokens=(0,))[:, 0].all()

    def test_given_sixteen_tokens_a_window_of_four_and_one_global_token_70_of_136_pairs_are_scored(self):
        # The window gives 1 + 2 + 3 + 4 × 13 = 58; the 12 rows beyond it add the global column.
        assert pairs_computed(global_local_mask(16, w=4, global_tokens=(0,))) == 70

    def test_given_a_stride_of_four_a_token_sees_its_last_four_and_every_fourth_before_them(self):
        assert np.flatnonzero(strided_mask(16, stride=4)[13]).tolist() == [1, 5, 9, 10, 11, 12, 13]

    def test_given_sixteen_tokens_and_a_stride_of_four_82_pairs_are_scored(self):
        # 58 local pairs plus floor(i / 4) strided pairs per row: 0·4 + 1·4 + 2·4 + 3·4 = 24.
        assert pairs_computed(strided_mask(16, stride=4)) == 82

    def test_given_any_sparse_mask_the_pairs_it_skips_get_exactly_zero_attention(self):
        rng = np.random.default_rng(2)
        Q, K, V = (rng.standard_normal((16, 4)) for _ in range(3))
        mask = global_local_mask(16, w=4, global_tokens=(0,))
        _, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        assert np.all(weights[~mask] == 0.0)

    def test_given_a_fixed_window_doubling_the_sequence_doubles_the_pairs_instead_of_quadrupling_them(self):
        assert pairs_computed(sliding_window_mask(2048, 64)) / pairs_computed(sliding_window_mask(1024, 64)) == pytest.approx(2, rel=0.02)


# The worked example: φ(x) = x + 1 for positive x, so every number can be checked by hand.
Q_WORKED = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
K_WORKED = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
V_WORKED = np.array([[2.0], [4.0], [6.0]])


class TestLinearAttention:
    def test_given_any_input_the_feature_map_is_positive_so_no_weight_is_negative(self):
        assert (feature_map(np.linspace(-20, 20, 101)) > 0).all()

    def test_given_the_worked_example_the_third_token_reads_62_fifteenths(self):
        # φ(q) · S = (2, 1) · (20, 22) = 62; φ(q) · z = (2, 1) · (5, 5) = 15.
        out, _, _ = linear_attention_recurrent(Q_WORKED, K_WORKED, V_WORKED)
        assert out[2, 0] == pytest.approx(62 / 15)

    def test_given_the_worked_example_the_running_sums_are_20_22_and_5_5(self):
        _, S, z = linear_attention_recurrent(Q_WORKED, K_WORKED, V_WORKED)
        assert S[:, 0].tolist() == [20.0, 22.0] and z.tolist() == [5.0, 5.0]

    def test_given_the_same_inputs_the_running_sum_matches_the_quadratic_form_exactly(self):
        rng = np.random.default_rng(3)
        Q, K, V = (rng.standard_normal((20, 4)) for _ in range(3))
        assert np.allclose(linear_attention_recurrent(Q, K, V)[0], linear_attention_quadratic(Q, K, V), atol=1e-12)

    def test_given_the_same_inputs_linear_attention_gives_a_different_answer_from_softmax_attention(self):
        # Unscaled softmax gives weights (0.42, 0.16, 0.42) and exactly 4.0; the kernel gives 62/15 ≈ 4.13.
        softmax_out, _ = scaled_dot_product_attention(Q_WORKED, K_WORKED, V_WORKED, mask=causal_mask(3), scale=False)
        assert softmax_out[2, 0] == pytest.approx(4.0)
        assert linear_attention_quadratic(Q_WORKED, K_WORKED, V_WORKED)[2, 0] != pytest.approx(4.0)

    def test_given_any_sequence_length_the_state_it_carries_has_the_same_size(self):
        rng = np.random.default_rng(4)
        shapes = set()
        for n in (3, 30, 300):
            Q, K, V = (rng.standard_normal((n, 4)) for _ in range(3))
            _, S, z = linear_attention_recurrent(Q, K, V)
            shapes.add((S.shape, z.shape))
        assert shapes == {((4, 4), (4,))}


class TestStateSpaceModels:
    def test_given_a_half_decay_and_inputs_1_0_0_2_the_outputs_are_1_half_quarter_and_2_125(self):
        # h = 0.5·h + x: 1, then 0.5, then 0.25, then 0.125 + 2.
        assert ssm_recurrent([1, 0, 0, 2], A=0.5, B=1.0, C=1.0).tolist() == [1.0, 0.5, 0.25, 2.125]

    def test_given_a_half_decay_the_convolution_kernel_is_1_half_quarter_eighth(self):
        assert ssm_kernel(A=0.5, B=1.0, C=1.0, length=4).tolist() == [1.0, 0.5, 0.25, 0.125]

    def test_given_a_time_invariant_ssm_the_convolution_gives_the_same_outputs_as_the_recurrence(self):
        rng = np.random.default_rng(5)
        A = np.diag([0.9, 0.5, -0.3])
        B, C, x = rng.standard_normal(3), rng.standard_normal(3), rng.standard_normal(40)
        assert np.allclose(ssm_convolution(x, A, B, C), ssm_recurrent(x, A, B, C))

    def test_given_a_large_step_the_state_is_replaced_by_the_new_input(self):
        # Ā = e^(−5) ≈ 0.007 keeps almost nothing; B̄ = 1 − e^(−5) ≈ 0.993 writes almost everything.
        A_bar, B_bar = discretize(delta=5.0, a=-1.0, b=1.0)
        assert A_bar == pytest.approx(0.0067, abs=1e-4) and B_bar == pytest.approx(0.9933, abs=1e-4)

    def test_given_a_tiny_step_the_state_is_kept_and_the_input_ignored(self):
        A_bar, B_bar = discretize(delta=0.01, a=-1.0, b=1.0)
        assert A_bar == pytest.approx(0.990, abs=1e-3) and B_bar == pytest.approx(0.00995, abs=1e-5)

    def test_given_a_marked_token_followed_by_noise_a_selective_ssm_still_holds_it_at_the_end(self):
        assert selective_recall() == pytest.approx(7.0, rel=0.10)

    def test_given_the_same_task_no_fixed_step_size_holds_even_half_of_it(self):
        # A fixed step must both write the 7 strongly and keep it through nine later writes: it can't do both.
        best = max(time_invariant_recall(delta) for delta in np.geomspace(1e-3, 10, 60))
        assert best < 3.5

    def test_given_a_parallel_scan_it_gives_the_same_states_as_the_step_by_step_loop(self):
        rng = np.random.default_rng(6)
        a, b = rng.uniform(0, 1, 37), rng.standard_normal(37)
        h, loop = np.zeros(37), 0.0
        for t in range(37):
            loop = a[t] * loop + b[t]
            h[t] = loop
        assert np.allclose(parallel_scan(a, b)[0], h)

    def test_given_1024_steps_a_parallel_scan_needs_only_10_rounds(self):
        assert parallel_scan(np.ones(1024), np.ones(1024))[1] == 10

    def test_given_any_context_length_the_ssm_state_takes_the_same_memory(self):
        # 4,096 channels × 16 state numbers × 2 bytes, whether the conversation is 10 tokens or 10 million.
        assert ssm_state_bytes(channels=4096, state_size=16) == 131_072


class TestHybridModels:
    def test_given_one_attention_layer_in_eight_the_cache_is_about_an_eighth_of_an_all_attention_model(self):
        all_attention = context_cost(131_072)["kv_cache_bytes"]
        hybrid = hybrid_cache_bytes(131_072, layers=32, attention_layers=4)
        assert hybrid / all_attention == pytest.approx(1 / 8, rel=0.01)


class TestCompressingTheKVCache:
    def test_given_a_long_context_a_sliding_window_cache_stops_growing_at_the_window(self):
        assert cache_bytes_by_method(1_000_000)["sliding window 4k"] == cache_bytes_by_method(4_096)["sliding window 4k"]

    def test_given_the_worked_latent_example_a_token_is_stored_as_3_1_and_its_key_is_3_1_3_1(self):
        worked = latent_kv_worked_example()
        assert worked["latent"].tolist() == [3.0, 1.0] and worked["key"].tolist() == [3.0, 1.0, 3.0, 1.0]

    def test_given_a_512_number_latent_plus_a_64_number_position_key_the_cache_is_57_times_smaller_than_128_full_heads(self):
        # DeepSeek-V2's shape: 2 × 128 heads × 128 dims = 32,768 numbers per token per layer, against 576.
        full = 2 * 128 * 128 * 2
        assert full / latent_kv_bytes_per_token(layers=1, latent_dim=512, rope_dim=64) == pytest.approx(56.9, abs=0.05)

    def test_given_cached_latents_attention_matches_attention_over_the_up_projected_keys_and_values(self):
        layer = LatentKVAttention(d_model=32, n_heads=4, d_head=8, d_latent=6, seed=0)
        X = np.random.default_rng(7).standard_normal((10, 32))
        C = layer.compress(X)
        naive = layer.attend(X, C)
        # The same heads, written as ordinary multi-head attention with K = X·W_down·W_uk.
        Q = (X @ layer.W_q).reshape(10, 4, 8).transpose(1, 0, 2)
        K = (C @ layer.W_uk).reshape(10, 4, 8).transpose(1, 0, 2)
        V = (C @ layer.W_uv).reshape(10, 4, 8).transpose(1, 0, 2)
        heads, _ = scaled_dot_product_attention(Q, K, V, mask=causal_mask(10))
        assert np.allclose(naive, heads.transpose(1, 0, 2).reshape(10, 32))

    def test_given_the_up_projection_absorbed_into_the_query_the_output_is_unchanged(self):
        layer = LatentKVAttention(d_model=32, n_heads=4, d_head=8, d_latent=6, seed=0)
        X = np.random.default_rng(7).standard_normal((10, 32))
        C = layer.compress(X)
        assert np.allclose(layer.attend_absorbed(X, C), layer.attend(X, C))

    def test_given_a_latent_of_width_6_every_head_s_keys_together_have_rank_at_most_6(self):
        # 4 heads × 8 dims = 32 key numbers per token, but they are all made from 6: that is why storing 6 suffices.
        layer = LatentKVAttention(d_model=32, n_heads=4, d_head=8, d_latent=6, seed=0)
        assert np.linalg.matrix_rank(layer.W_down @ layer.W_uk) == 6

    def test_given_the_worked_key_4_bit_codes_are_7_minus_3_2_and_0(self):
        codes, scales = quantize_kv(np.array([[0.7, -0.3, 0.2, 0.04]]), bits=4)
        assert codes.tolist() == [[7, -3, 2, 0]] and scales[0, 0] == pytest.approx(0.1)

    def test_given_an_8_bit_cache_attention_output_moves_by_under_one_percent(self):
        assert quantized_cache_error(bits=8) < 0.01

    def test_given_a_4_bit_cache_attention_output_moves_more_than_with_8_bits(self):
        assert quantized_cache_error(bits=4) > 5 * quantized_cache_error(bits=8)

    def test_given_4_bit_codes_and_one_16_bit_scale_per_head_vector_the_cache_is_about_3_9_times_smaller(self):
        # Per head vector: 128 × 2 bytes = 256 at 16-bit, against 128 × 0.5 + 2 = 66 at 4-bit.
        full = quantized_kv_bytes_per_token(layers=32, kv_heads=8, head_dim=128, bits=16, scale_bits=0)
        small = quantized_kv_bytes_per_token(layers=32, kv_heads=8, head_dim=128, bits=4)
        assert full / small == pytest.approx(3.88, abs=0.01)
