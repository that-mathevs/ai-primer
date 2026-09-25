"""Specification for primer.ml.inference: what actually happens, and what it costs, when a model generates."""

import numpy as np
import pytest

from primer.ml import inference as inf

GB = 1e9


class TestMemoryMath:
    def test_given_70_billion_parameters_at_16_bit_the_weights_need_140_gb(self):
        # 70e9 parameters × 2 bytes: more than one 80 GB GPU holds.
        assert inf.weight_bytes(70e9, bits=16) / GB == pytest.approx(140)

    def test_given_70_billion_parameters_at_4_bit_the_weights_need_35_gb(self):
        # Half a byte per parameter: now it fits on one GPU.
        assert inf.weight_bytes(70e9, bits=4) / GB == pytest.approx(35)

    def test_given_a_llama_3_8b_shaped_model_the_kv_cache_costs_131_072_bytes_per_token(self):
        # 2 (keys and values) × 32 layers × 8 KV heads × 128 dims × 2 bytes.
        assert inf.kv_cache_bytes_per_token(layers=32, kv_heads=8, head_dim=128, bits=16) == 131_072

    def test_given_a_32_000_token_context_one_request_holds_about_4_gb_of_kv_cache(self):
        per_request = inf.kv_cache_bytes(32_000, layers=32, kv_heads=8, head_dim=128, bits=16)
        assert per_request / GB == pytest.approx(4.19, abs=0.01)

    def test_given_an_80_gb_gpu_and_16_gb_of_weights_15_requests_at_32k_context_fit(self):
        # (80 - 16) GB / 4.19 GB per request = 15.3: long contexts cap concurrency.
        assert inf.max_concurrent_requests(gpu_gb=80, weights_gb=16, context=32_000, layers=32, kv_heads=8, head_dim=128) == 15

    def test_given_32_kv_heads_instead_of_8_the_kv_cache_is_four_times_larger(self):
        # Why grouped-query attention matters for serving cost.
        assert inf.kv_cache_bytes_per_token(layers=32, kv_heads=32, head_dim=128, bits=16) == 4 * 131_072


class TestPrefillVersusDecode:
    def test_given_one_token_at_a_time_at_16_bit_each_byte_of_weights_read_does_1_flop(self):
        # One multiply-add (2 FLOPs) per 2-byte weight: far too little work per byte to keep a GPU busy.
        assert inf.arithmetic_intensity(tokens_in_parallel=1, bits=16) == pytest.approx(1.0)

    def test_given_a_1000_token_prompt_prefill_does_1000_flops_per_byte_of_weights(self):
        assert inf.arithmetic_intensity(tokens_in_parallel=1000, bits=16) == pytest.approx(1000.0)

    def test_given_a_gpu_with_1_pflop_per_second_and_3_35_tb_per_second_the_break_even_is_about_300_flops_per_byte(self):
        assert inf.ridge_point(flops_per_second=1e15, bytes_per_second=3.35e12) == pytest.approx(298.5, abs=0.1)

    def test_given_single_token_decoding_the_gpu_is_memory_bound(self):
        assert inf.bottleneck(tokens_in_parallel=1) == "memory-bound"

    def test_given_a_1000_token_prefill_the_gpu_is_compute_bound(self):
        assert inf.bottleneck(tokens_in_parallel=1000) == "compute-bound"

    def test_given_an_8b_model_at_16_bit_each_decoded_token_takes_about_4_8_ms(self):
        # Every weight (16 GB) is read once per token at 3.35 TB/s: 16e9 / 3.35e12 s.
        assert inf.decode_seconds_per_token(8e9, bits=16) * 1000 == pytest.approx(4.78, abs=0.01)

    def test_given_an_8b_model_a_1000_token_prompt_prefills_in_about_16_ms(self):
        # 2 FLOPs × 8e9 parameters × 1000 tokens / 1e15 FLOP/s.
        assert inf.prefill_seconds(8e9, n_tokens=1000) * 1000 == pytest.approx(16.0, abs=0.01)


class TestKVCache:
    PROMPT = [3, 1, 4, 1, 5, 9, 2, 6]

    def test_given_a_cached_prefix_incremental_decoding_matches_full_recomputation(self):
        # Causal masking means earlier tokens' keys and values never change, so reusing them is exact.
        model = inf.TinyDecoder(seed=0)
        full = model.forward_full(np.array(self.PROMPT))
        cache = model.new_cache()
        stepped = np.stack([model.forward_step(tok, pos, cache) for pos, tok in enumerate(self.PROMPT)])
        assert np.allclose(full, stepped, atol=1e-10)

    def test_given_the_same_prompt_greedy_generation_with_and_without_the_cache_produces_the_same_tokens(self):
        model = inf.TinyDecoder(seed=0)
        with_cache = model.generate(self.PROMPT, n_new=16, use_cache=True)
        without = model.generate(self.PROMPT, n_new=16, use_cache=False)
        assert with_cache.tokens == without.tokens

    def test_given_an_8_token_prompt_and_16_new_tokens_full_recomputation_processes_248_token_positions(self):
        # Step k re-reads the prompt plus k generated tokens: 8 + 9 + ... + 23 = 248.
        assert inf.TinyDecoder(seed=0).generate(self.PROMPT, n_new=16, use_cache=False).positions_processed == 248

    def test_given_an_8_token_prompt_and_16_new_tokens_the_cache_processes_23_token_positions(self):
        # Prefill the 8 prompt tokens once, then one new position for each of the 15 later tokens.
        assert inf.TinyDecoder(seed=0).generate(self.PROMPT, n_new=16, use_cache=True).positions_processed == 23

    def test_when_a_token_is_decoded_every_layer_cache_grows_by_one_key_and_one_value(self):
        model = inf.TinyDecoder(seed=0, n_layers=2)
        cache = model.new_cache()
        for pos, tok in enumerate(self.PROMPT[:5]):
            model.forward_step(tok, pos, cache)
        assert [layer["k"].shape[1] for layer in cache] == [5, 5]
        assert [layer["v"].shape[1] for layer in cache] == [5, 5]


class TestSampling:
    def test_given_temperature_1_the_probabilities_are_the_plain_softmax(self):
        # softmax(2, 1, 0) = (0.665, 0.245, 0.090)
        assert inf.temperature_probs(np.array([2.0, 1.0, 0.0]), 1.0) == pytest.approx([0.665, 0.245, 0.090], abs=0.001)

    def test_given_temperature_one_half_the_distribution_sharpens_to_0_867_0_117_0_016(self):
        # Dividing by 0.5 doubles the logits: softmax(4, 2, 0).
        assert inf.temperature_probs(np.array([2.0, 1.0, 0.0]), 0.5) == pytest.approx([0.867, 0.117, 0.016], abs=0.001)

    def test_given_temperature_0_all_probability_goes_to_the_top_token(self):
        assert inf.temperature_probs(np.array([2.0, 1.0, 0.0]), 0.0).tolist() == [1.0, 0.0, 0.0]

    def test_given_top_k_of_2_only_the_two_likeliest_tokens_survive_renormalised(self):
        # 0.5 and 0.3 are kept and rescaled to sum to 1: 0.625 and 0.375.
        probs = np.array([0.5, 0.3, 0.15, 0.05])
        assert inf.top_k_filter(probs, 2) == pytest.approx([0.625, 0.375, 0.0, 0.0])

    def test_given_top_p_of_0_9_the_smallest_set_reaching_90_percent_survives(self):
        # 0.5 + 0.3 = 0.80 < 0.9, so 0.15 is also needed (0.95); the 0.05 tail is cut.
        probs = np.array([0.5, 0.3, 0.15, 0.05])
        assert inf.top_p_filter(probs, 0.9) == pytest.approx([0.5 / 0.95, 0.3 / 0.95, 0.15 / 0.95, 0.0])

    def test_given_a_confident_distribution_top_p_keeps_fewer_tokens_than_a_flat_one(self):
        # Top-p adapts: it keeps 1 token when the model is sure and many when it isn't.
        confident = inf.top_p_filter(np.array([0.95, 0.03, 0.01, 0.01]), 0.9)
        flat = inf.top_p_filter(np.array([0.3, 0.25, 0.25, 0.2]), 0.9)
        assert (np.count_nonzero(confident), np.count_nonzero(flat)) == (1, 4)

    def test_given_the_same_numbers_adding_them_in_a_different_order_gives_a_different_float(self):
        # Floating-point addition is not associative, one reason temperature 0 is not guaranteed deterministic.
        assert (inf.float32_sum([1e8, 1.0, -1e8]), inf.float32_sum([1e8, -1e8, 1.0])) == (0.0, 1.0)

    def test_given_two_nearly_tied_tokens_a_different_summation_order_flips_the_greedy_choice(self):
        # Token A's logit is the sum (1e8, 1, -1e8); token B's is 0.5. The order decides whether A reaches 1.
        picks = [inf.greedy_pick_with_summation_order(order, other_logit=0.5) for order in ([1e8, 1.0, -1e8], [1e8, -1e8, 1.0])]
        assert picks == ["B", "A"]


class TestSpeculativeDecoding:
    # A 3-token toy language: target (big model) and draft (small model) next-token tables.
    TARGET = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.3, 0.3, 0.4]])
    DRAFT = np.array([[0.3, 0.3, 0.4], [0.2, 0.5, 0.3], [0.6, 0.2, 0.2]])

    def test_given_target_0_5_0_3_0_2_and_draft_0_3_0_3_0_4_a_drafted_token_is_accepted_80_percent_of_the_time(self):
        # Acceptance rate = Σ min(p, q) = 0.3 + 0.3 + 0.2.
        assert inf.acceptance_rate(self.TARGET[0], self.DRAFT[0]) == pytest.approx(0.8)

    def test_given_an_80_percent_acceptance_rate_and_4_drafts_a_round_yields_3_36_tokens_on_average(self):
        # (1 - 0.8^5) / (1 - 0.8) = 3.36: up to 4 accepted drafts plus the target's own token.
        assert inf.expected_tokens_per_round(0.8, n_draft=4) == pytest.approx(3.3616)

    def test_given_a_draft_identical_to_the_target_every_draft_is_accepted_plus_one_bonus_token(self):
        rng = np.random.default_rng(0)
        tokens = inf.speculative_round(0, self.TARGET, self.TARGET, n_draft=4, rng=rng)
        assert len(tokens) == 5

    def test_given_a_different_draft_the_first_token_still_follows_the_target_distribution(self):
        # The accept/reject rule makes speculative output statistically identical to the big model's.
        rng = np.random.default_rng(1)
        firsts = [inf.speculative_round(0, self.TARGET, self.DRAFT, n_draft=3, rng=rng)[0] for _ in range(20_000)]
        assert np.bincount(firsts, minlength=3) / 20_000 == pytest.approx([0.5, 0.3, 0.2], abs=0.012)

    def test_given_a_different_draft_the_second_token_still_follows_the_target_distribution(self):
        # P(second = j) = Σ_i P(first = i) · TARGET[i, j] = (0.5·0.5 + 0.3·0.1 + 0.2·0.3, ...) = (0.34, 0.39, 0.27).
        rng = np.random.default_rng(2)
        # A round can stop after one token (an early rejection), so read the second token of the generated stream.
        seconds = [inf.speculative_generate(0, self.TARGET, self.DRAFT, n_tokens=2, n_draft=3, rng=rng)[1] for _ in range(20_000)]
        assert np.bincount(seconds, minlength=3) / 20_000 == pytest.approx([0.34, 0.39, 0.27], abs=0.012)


class TestQuantization:
    ROW = np.array([[0.5, -1.27, 0.02]])

    def test_given_a_row_whose_largest_weight_is_1_27_int8_codes_are_50_minus_127_and_2(self):
        # Scale = 1.27 / 127 = 0.01, so each weight is stored as round(w / 0.01).
        codes, _ = inf.quantize(self.ROW, bits=8)
        assert codes.tolist() == [[50, -127, 2]]

    def test_given_the_same_row_int4_codes_are_3_minus_7_and_0(self):
        # Scale = 1.27 / 7 = 0.181: 0.5 → 2.76 → 3, 0.02 → 0.11 → 0. Four bits keep only 15 levels.
        codes, _ = inf.quantize(self.ROW, bits=4)
        assert codes.tolist() == [[3, -7, 0]]

    def test_given_int4_codes_the_row_comes_back_as_0_544_minus_1_27_and_0(self):
        codes, scales = inf.quantize(self.ROW, bits=4)
        assert inf.dequantize(codes, scales)[0] == pytest.approx([0.544, -1.27, 0.0], abs=0.001)

    def test_given_one_outlier_weight_per_channel_scales_keep_the_other_rows_precise(self):
        # A single shared scale is stretched by the outlier; one scale per output row isolates it.
        rng = np.random.default_rng(0)
        W = rng.normal(0, 0.02, (64, 64))
        W[0, 0] = 2.0
        per_tensor = inf.quantization_error(W, bits=8, per_channel=False)
        per_channel = inf.quantization_error(W, bits=8, per_channel=True)
        assert per_channel < per_tensor / 5

    def test_given_random_weights_int8_error_is_far_below_int4_error(self):
        W = np.random.default_rng(1).normal(0, 0.02, (64, 64))
        assert inf.quantization_error(W, bits=8) < inf.quantization_error(W, bits=4) / 10


class TestContinuousBatching:
    # Four requests needing 4, 1, 1 and 1 decode steps, served two at a time.
    LENGTHS = [4, 1, 1, 1]

    def test_given_static_batches_the_short_request_waits_for_the_long_one_and_serving_takes_5_steps(self):
        # Batch {4, 1} runs 4 steps (one slot idle for 3), then batch {1, 1} runs 1 step.
        assert inf.simulate_static_batching(self.LENGTHS, slots=2).steps == 5

    def test_given_static_batches_the_gpu_slots_are_busy_70_percent_of_the_time(self):
        # 7 useful slot-steps out of 5 steps × 2 slots.
        assert inf.simulate_static_batching(self.LENGTHS, slots=2).utilization == pytest.approx(0.70)

    def test_given_continuous_batching_a_finished_slot_is_refilled_at_once_and_serving_takes_4_steps(self):
        assert inf.simulate_continuous_batching(self.LENGTHS, slots=2).steps == 4

    def test_given_continuous_batching_the_gpu_slots_are_busy_87_5_percent_of_the_time(self):
        # 7 useful slot-steps out of 4 steps × 2 slots.
        assert inf.simulate_continuous_batching(self.LENGTHS, slots=2).utilization == pytest.approx(0.875)


class TestPromptCaching:
    def test_given_two_prompts_sharing_the_first_1000_tokens_1000_tokens_can_be_reused(self):
        shared = list(range(1000))
        assert inf.reusable_prefix_tokens(shared + [7, 8], shared + [9]) == 1000

    def test_given_a_change_in_the_very_first_token_nothing_can_be_reused(self):
        # A cached prefix is only valid if every earlier token is identical: put volatile content last.
        assert inf.reusable_prefix_tokens([0] + list(range(1, 1000)), [99] + list(range(1, 1000))) == 0

    def test_given_100_requests_with_a_10k_token_shared_prefix_caching_cuts_the_input_bill_from_3_15_to_0_48_dollars(self):
        # At $3 per million input tokens, cache writes at 1.25× and cache reads at 0.1×:
        # uncached 100 × 10,500 × $3e-6 = $3.15; cached $0.039 once + 99 × $0.0045 = $0.4845.
        uncached, cached = inf.prompt_cache_cost(prefix=10_000, suffix=500, requests=100, price_per_million=3.0)
        assert (round(uncached, 4), round(cached, 4)) == (3.15, 0.4845)
