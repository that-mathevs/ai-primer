"""Specification for primer.ml.transformer: the block that every modern LLM stacks."""

import numpy as np
import pytest

from primer.ml.transformer import (
    FeedForward,
    MixtureOfExperts,
    TinyGPT,
    TransformerBlock,
    gelu,
    gpt_param_count,
    inference_flops_per_token,
    layer_norm,
    load_balancing_loss,
    mask_patterns,
    rms_norm,
    top_k_gates,
    training_flops,
)


class TestNormalization:
    def test_layer_norm_turns_1_2_3_4_into_mean_zero_unit_spread(self):
        # mean 2.5, standard deviation √1.25 = 1.118; (1 − 2.5)/1.118 = −1.342
        assert layer_norm(np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx([-1.342, -0.447, 0.447, 1.342], abs=1e-3)

    def test_rms_norm_divides_1_2_3_4_by_their_root_mean_square_without_centring(self):
        # root mean square = √((1+4+9+16)/4) = √7.5 = 2.739
        assert rms_norm(np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx([0.365, 0.730, 1.095, 1.461], abs=1e-3)

    def test_layer_norm_ignores_how_large_the_input_is(self):
        # Scale-invariance is what keeps activations from drifting as layers stack.
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert layer_norm(100 * x) == pytest.approx(layer_norm(x), abs=1e-4)

    def test_each_token_is_normalized_on_its_own(self):
        rows = layer_norm(np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 0.0, 10.0, 0.0]]))
        assert rows[1] == pytest.approx([1.0, -1.0, 1.0, -1.0], abs=1e-4)


class TestGELU:
    def test_gelu_of_zero_is_zero(self):
        assert gelu(np.array(0.0)) == pytest.approx(0.0)

    def test_gelu_of_one_is_about_0_841(self):
        assert gelu(np.array(1.0)) == pytest.approx(0.841, abs=1e-3)

    def test_gelu_lets_large_positive_inputs_through_unchanged(self):
        assert gelu(np.array(10.0)) == pytest.approx(10.0)

    def test_gelu_squashes_large_negative_inputs_to_almost_zero(self):
        # Unlike ReLU it is smooth: a small negative value survives near zero.
        assert gelu(np.array(-3.0)) == pytest.approx(-0.004, abs=1e-3)


class TestFeedForward:
    def test_the_hidden_layer_is_four_times_wider_than_the_model(self):
        assert FeedForward(d_model=8).W1.shape == (8, 32)

    def test_the_output_has_the_same_shape_as_the_input(self):
        x = np.random.default_rng(0).standard_normal((5, 8))
        assert FeedForward(d_model=8)(x).shape == (5, 8)

    def test_with_d_model_8_it_has_552_parameters(self):
        # W1 8·32 + b1 32 + W2 32·8 + b2 8 = 256 + 32 + 256 + 8 = 552 (≈ 8·d² + 5·d)
        assert FeedForward(d_model=8).n_params() == 552

    def test_each_token_is_processed_without_looking_at_the_others(self):
        ffn = FeedForward(d_model=8)
        x = np.random.default_rng(1).standard_normal((3, 8))
        changed = x.copy()
        changed[2] += 5.0
        assert ffn(changed)[:2] == pytest.approx(ffn(x)[:2])


class TestTransformerBlock:
    def test_the_output_has_the_same_shape_as_the_input_so_blocks_can_stack(self):
        x = np.random.default_rng(0).standard_normal((6, 16))
        assert TransformerBlock(d_model=16, n_heads=4)(x).shape == (6, 16)

    def test_with_both_sublayers_switched_off_the_residual_path_passes_the_input_through_unchanged(self):
        # Pre-norm blocks add corrections to x; with zero corrections, x is untouched.
        block = TransformerBlock(d_model=16, n_heads=4)
        block.attn.W_o[:] = 0.0
        block.ffn.W2[:] = 0.0
        block.ffn.b2[:] = 0.0
        x = np.random.default_rng(0).standard_normal((6, 16))
        assert block(x) == pytest.approx(x)

    def test_in_a_decoder_block_a_later_token_never_changes_an_earlier_output(self):
        block = TransformerBlock(d_model=16, n_heads=4, causal=True)
        rng = np.random.default_rng(2)
        x = rng.standard_normal((5, 16))
        changed = x.copy()
        # A random change, not a constant: LayerNorm would erase a constant shift.
        changed[4] += rng.standard_normal(16)
        assert block(changed)[:4] == pytest.approx(block(x)[:4])

    def test_in_an_encoder_block_a_later_token_changes_the_first_output(self):
        # Bidirectional attention: every token sees every other (BERT-style).
        block = TransformerBlock(d_model=16, n_heads=4, causal=False)
        rng = np.random.default_rng(2)
        x = rng.standard_normal((5, 16))
        changed = x.copy()
        changed[4] += rng.standard_normal(16)
        assert not np.allclose(block(changed)[0], block(x)[0])

    def test_given_the_same_input_only_the_last_tokens_attention_row_matches_between_encoder_and_decoder(self):
        # The last token may see everything under either mask; every earlier row renormalises over fewer tokens.
        encoder, decoder = mask_patterns().values()
        rows_match = [bool(np.allclose(e, d)) for e, d in zip(encoder, decoder)]
        assert rows_match == [False, False, False, False, False, True]


class TestTinyGPT:
    def test_it_returns_one_score_per_vocabulary_entry_for_every_position(self):
        model = TinyGPT(vocab_size=50, d_model=32, n_layers=2, n_heads=4, max_len=16)
        assert model(np.array([3, 1, 4, 1, 5])).shape == (5, 50)

    def test_a_later_token_never_changes_the_scores_at_earlier_positions(self):
        model = TinyGPT(vocab_size=50, d_model=32, n_layers=2, n_heads=4, max_len=16)
        before = model(np.array([3, 1, 4, 1, 5]))
        after = model(np.array([3, 1, 4, 1, 9]))
        assert after[:4] == pytest.approx(before[:4])

    def test_the_output_layer_reuses_the_embedding_table(self):
        # Weight tying: the vector that reads token t in also scores token t out.
        model = TinyGPT(vocab_size=50, d_model=32, n_layers=2, n_heads=4, max_len=16)
        before = model(np.array([1, 2, 3]))[-1, 7]
        model.wte[7] *= 2.0
        assert model(np.array([1, 2, 3]))[-1, 7] != pytest.approx(before)

    def test_with_vocab_50_width_32_two_layers_and_16_positions_it_has_27328_parameters(self):
        # per layer 12·32² + 5·32 (FFN biases) + 4·32 (norms) = 12576; ×2 = 25152
        # + token table 50·32 + position table 16·32 + final norm 2·32 = 27328
        model = TinyGPT(vocab_size=50, d_model=32, n_layers=2, n_heads=4, max_len=16)
        assert model.n_params() == 27328


class TestParameterCounting:
    def test_gpt2_small_has_124439808_parameters(self):
        # The published size of GPT-2 small: vocab 50257, width 768, 12 layers, 1024 positions.
        assert gpt_param_count(vocab=50257, d_model=768, n_layers=12, max_len=1024) == 124_439_808

    def test_given_gpt2_small_without_attention_biases_it_has_124402944_parameters(self):
        # The tiny model's recipe: 12·(12·768² + 9·768) + 50257·768 + 1024·768 + 2·768 = 124,402,944,
        # which is 36,864 (12 blocks × 4·768 attention biases) short of the published size.
        assert gpt_param_count(vocab=50257, d_model=768, n_layers=12, max_len=1024, attn_bias=False) == 124_402_944

    def test_without_embeddings_a_wide_model_has_about_12_times_layers_times_width_squared(self):
        # 4·d² for attention + 8·d² for the 4×-wide feed-forward; biases and norms are tiny.
        body = gpt_param_count(vocab=0, d_model=4096, n_layers=32, max_len=0)
        assert body / (12 * 32 * 4096**2) == pytest.approx(1.0, abs=0.01)


class TestMixtureOfExperts:
    def test_router_scores_2_1_half_minus_1_send_the_token_to_experts_0_and_1_at_73_and_27_percent(self):
        # Keep the top 2 scores, softmax over just those: e²/(e²+e¹) = 0.731.
        gates = top_k_gates(np.array([[2.0, 1.0, 0.5, -1.0]]), k=2)
        assert gates[0] == pytest.approx([0.731, 0.269, 0.0, 0.0], abs=1e-3)

    def test_every_token_is_sent_to_exactly_k_experts(self):
        moe = MixtureOfExperts(d_model=16, n_experts=8, k=2)
        moe(np.random.default_rng(0).standard_normal((10, 16)))
        assert ((moe.last_gates > 0).sum(axis=1) == 2).all()

    def test_the_output_has_the_same_shape_as_the_input(self):
        moe = MixtureOfExperts(d_model=16, n_experts=8, k=2)
        assert moe(np.random.default_rng(0).standard_normal((10, 16))).shape == (10, 16)

    def test_eight_experts_of_width_16_hold_17152_parameters_in_total(self):
        # one expert: 16·64 + 64 + 64·16 + 16 = 2128; ×8 = 17024; router 16·8 = 128
        assert MixtureOfExperts(d_model=16, n_experts=8, k=2).n_params() == 17152

    def test_with_top_2_routing_each_token_uses_only_4384_of_those_parameters(self):
        # two experts 2·2128 + the router 128: about a quarter of the total.
        assert MixtureOfExperts(d_model=16, n_experts=8, k=2).active_params() == 4384

    def test_given_an_untrained_router_and_256_tokens_the_load_ranges_from_51_to_75_against_an_even_64(self):
        # The lesson's figure: random routing is already lopsided before training compounds it.
        moe = MixtureOfExperts(d_model=16, n_experts=8, k=2, seed=0)
        moe(np.random.default_rng(1).standard_normal((256, 16)))
        assert moe.tokens_per_expert().tolist() == [67, 56, 51, 75, 66, 75, 67, 55]


class TestLoadBalancing:
    def test_a_perfectly_balanced_router_scores_a_balancing_loss_of_1(self):
        # Token i strongly prefers expert i, so each of 4 experts gets a quarter of the traffic.
        assert load_balancing_loss(10 * np.eye(4), k=1) == pytest.approx(1.0, abs=1e-3)

    def test_a_router_sending_everything_to_one_of_4_experts_scores_about_4(self):
        # The Switch Transformer loss n·Σ fᵢ·Pᵢ peaks at n when one expert takes all.
        logits = np.tile([10.0, 0.0, 0.0, 0.0], (4, 1))
        assert load_balancing_loss(logits, k=1) == pytest.approx(4.0, abs=0.01)


class TestComputeArithmetic:
    def test_generating_one_token_with_a_7_billion_parameter_model_costs_about_14_gigaflops(self):
        # Each parameter is used once per token in a multiply and an add: 2 operations.
        assert inference_flops_per_token(7e9) == pytest.approx(1.4e10)

    def test_training_a_70_billion_parameter_model_on_1_trillion_tokens_costs_4_2e23_flops(self):
        # Forward (2 per parameter) + backward (about 4 per parameter) = 6 per parameter per token.
        assert training_flops(70e9, 1e12) == pytest.approx(4.2e23)

    def test_gpt3_training_on_300_billion_tokens_comes_to_about_3_1e23_flops(self):
        # The GPT-3 paper reports about 3.14e23 FLOPs for 175B parameters.
        assert training_flops(175e9, 300e9) == pytest.approx(3.14e23, rel=0.01)
