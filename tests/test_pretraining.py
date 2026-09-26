"""Specification for primer.ml.pretraining: cleaning a web's worth of text, and training on it across many GPUs."""

import numpy as np
import pytest

from primer.ml.pretraining import (
    BF16,
    CRAWL_SAMPLE,
    FP8_E4M3,
    FP16,
    FP32,
    DynamicLossScaler,
    MinHasher,
    QualityClassifier,
    accumulate_updates,
    activation_bytes,
    all_reduce_traffic,
    bubble_fraction,
    chinchilla_tokens,
    column_parallel_matmul,
    curate,
    detect_language,
    detect_spikes,
    epochs_per_source,
    estimate_jaccard,
    exact_dedup,
    jaccard,
    linear_regression_gradient,
    lsh_candidate_probability,
    near_duplicate_pairs,
    optimal_checkpoint_interval,
    pipeline_schedule,
    quality_failures,
    quantize,
    recursive_gaussian_fit,
    ring_all_reduce,
    row_parallel_matmul,
    scaled_gradient_roundtrip,
    sharded_global_norm,
    shingles,
    tensor_parallel_mlp,
    train_through_a_bad_batch,
    training_flops,
    training_memory,
    verbatim_probability,
    wasted_fraction,
    zero_memory_per_gpu,
)

GOOD_DOC = (
    "The river carries sand and small stones from the mountains to the sea. Over thousands of years "
    "that sand settles at the mouth of the river and builds a wide, flat delta. Farmers have grown rice "
    "on these deltas for centuries, because the soil is rich and the water is close. When the river "
    "floods, it brings new soil with it, which is why the land stays fertile."
)


class TestLanguageIdentification:
    def test_given_an_english_sentence_it_is_identified_as_english(self):
        assert detect_language("the cat is in the garden and it was happy") == "en"

    def test_given_a_french_sentence_it_is_identified_as_french(self):
        assert detect_language("le chat est dans le jardin et les enfants sont contents") == "fr"

    def test_given_a_string_of_product_codes_it_is_marked_unknown(self):
        assert detect_language("SKU-4431 X99 blk/wht 12pk") == "unknown"


class TestHeuristicQualityFilters:
    def test_given_a_well_formed_paragraph_it_passes_every_rule(self):
        assert quality_failures(GOOD_DOC) == []

    def test_given_a_short_snippet_it_fails_the_length_rule(self):
        assert "too_short" in quality_failures("Home | About | Contact")

    def test_given_a_page_full_of_hashtags_it_fails_the_symbol_rule(self):
        spam = " ".join(["#deal #sale #win the best price"] * 20)
        assert "too_many_symbols" in quality_failures(spam)

    def test_given_a_keyword_list_with_no_function_words_it_fails_the_stop_word_rule(self):
        # Real prose always uses little words like "the" and "of"; SEO keyword stuffing doesn't.
        keywords = " ".join(["cheap flights hotels deals booking discount travel"] * 10)
        assert "missing_stop_words" in quality_failures(keywords)


class TestClassifierQualityScores:
    def test_given_reference_text_style_the_classifier_scores_it_as_good(self):
        clf = QualityClassifier.trained_on_examples()
        assert clf.score("the delta is formed when the river deposits sand over many years") > 0

    def test_given_spam_style_the_classifier_scores_it_as_bad(self):
        clf = QualityClassifier.trained_on_examples()
        assert clf.score("click here buy now best price free free free") < 0


class TestExactDeduplication:
    def test_given_copies_that_differ_only_in_case_and_spacing_one_copy_is_kept(self):
        docs = ["Hello  world.", "hello world.", "Something else"]
        assert exact_dedup(docs) == ["Hello  world.", "Something else"]


class TestJaccardSimilarity:
    def test_given_two_sentences_one_word_apart_their_two_word_shingles_overlap_three_out_of_seven(self):
        a = shingles("the cat sat on the mat", k=2)
        b = shingles("the cat sat on a mat", k=2)
        assert jaccard(a, b) == pytest.approx(3 / 7)


class TestMinHash:
    def test_given_identical_documents_the_estimate_is_exactly_one(self):
        s = shingles(GOOD_DOC)
        h = MinHasher(num_perm=64, seed=0)
        assert estimate_jaccard(h.signature(s), h.signature(s)) == 1.0

    def test_given_sets_with_true_jaccard_one_third_256_hashes_estimate_it_within_point_zero_eight(self):
        a = {f"w{i}" for i in range(0, 100)}
        b = {f"w{i}" for i in range(50, 150)}  # 50 shared out of 150 in the union
        h = MinHasher(num_perm=256, seed=0)
        assert estimate_jaccard(h.signature(a), h.signature(b)) == pytest.approx(1 / 3, abs=0.08)

    def test_given_disjoint_documents_the_estimate_is_near_zero(self):
        h = MinHasher(num_perm=128, seed=0)
        a = {f"a{i}" for i in range(80)}
        b = {f"b{i}" for i in range(80)}
        assert estimate_jaccard(h.signature(a), h.signature(b)) < 0.05


class TestLocalitySensitiveHashing:
    def test_given_20_bands_of_5_rows_a_pair_at_similarity_08_becomes_a_candidate_almost_surely(self):
        # 1 - (1 - 0.8^5)^20, worked by hand: 0.8^5 = 0.32768, 0.67232^20 ≈ 0.00036.
        assert lsh_candidate_probability(0.8, bands=20, rows=5) == pytest.approx(0.99964, abs=1e-5)

    def test_given_20_bands_of_5_rows_a_pair_at_similarity_03_rarely_becomes_a_candidate(self):
        # 0.3^5 = 0.00243, 1 - 0.99757^20 ≈ 0.0475.
        assert lsh_candidate_probability(0.3, bands=20, rows=5) == pytest.approx(0.0475, abs=1e-3)

    def test_given_a_near_copy_among_different_documents_only_that_pair_is_flagged(self):
        near_copy = GOOD_DOC.replace("centuries", "hundreds of years")
        other = "Stars are born in clouds of gas that collapse under their own gravity until fusion begins."
        assert near_duplicate_pairs([GOOD_DOC, other, near_copy]) == [(0, 2)]


class TestWhyDuplicatesHurt:
    def test_given_a_boilerplate_line_repeated_a_hundred_times_a_bigram_model_nearly_memorises_it(self):
        assert verbatim_probability(copies=100) > 0.9

    def test_given_the_same_line_kept_once_the_model_is_far_less_sure_of_it(self):
        assert verbatim_probability(copies=1) < 0.2


class TestTheCurationPipeline:
    def test_given_a_small_crawl_only_the_clean_english_documents_survive_one_copy_each(self):
        assert curate(CRAWL_SAMPLE) == [
            "kept",
            "language: fr",
            "quality: too_short, too_few_alphabetic_words, missing_stop_words",
            "exact duplicate of 0",
            "quality: too_many_symbols, missing_stop_words",
            "near duplicate of 0",
            "kept",
        ]


class TestDataMixtures:
    def test_given_a_small_source_with_a_large_weight_it_is_seen_more_than_once(self):
        # 5% of a 1,000B-token budget is 50B tokens drawn from a 20B-token source: 2.5 passes.
        epochs = epochs_per_source({"web": 0.80, "wiki": 0.05, "code": 0.15},
                                   {"web": 900e9, "wiki": 20e9, "code": 150e9}, budget=1000e9)
        assert epochs["wiki"] == pytest.approx(2.5)

    def test_given_a_large_source_with_a_matching_weight_it_is_seen_less_than_once(self):
        epochs = epochs_per_source({"web": 0.80, "wiki": 0.05, "code": 0.15},
                                   {"web": 900e9, "wiki": 20e9, "code": 150e9}, budget=1000e9)
        assert epochs["web"] == pytest.approx(0.889, abs=1e-3)


class TestTokenBudgets:
    def test_given_a_7b_model_the_compute_optimal_budget_is_about_140b_tokens(self):
        # Hoffmann et al. (2022): roughly 20 training tokens per parameter.
        assert chinchilla_tokens(7e9) == pytest.approx(140e9)

    def test_given_7b_parameters_and_140b_tokens_training_costs_about_6_times_10_to_the_21_flops(self):
        # 6 · 7e9 · 140e9 = 5.88e21
        assert training_flops(7e9, 140e9) == pytest.approx(5.88e21)


class TestSyntheticDataAndModelCollapse:
    def test_given_each_generation_trained_only_on_the_last_ones_samples_the_spread_collapses(self):
        stds = recursive_gaussian_fit(generations=200, n=20, seed=0)
        assert stds[0] == pytest.approx(1.0, abs=0.35)
        assert stds[-1] < 0.5 * stds[0]


class TestTrainingMemory:
    def test_given_adam_in_mixed_precision_each_parameter_costs_16_bytes(self):
        # 2 (bf16 weights) + 2 (bf16 gradients) + 4 (fp32 master) + 4 + 4 (Adam's two moments).
        assert training_memory(1)["total"] == 16

    def test_given_a_7b_model_its_training_state_needs_112_gb_more_than_one_80_gb_gpu(self):
        assert training_memory(7e9)["total"] / 1e9 == pytest.approx(112.0)

    def test_given_a_7b_shape_at_4096_tokens_activations_take_about_104_gb(self):
        # Korthikanti et al. (2022): s·b·h·(34 + 5·a·s/h) bytes per layer; 4096·4096·194 · 32 layers.
        gb = activation_bytes(seq=4096, batch=1, hidden=4096, heads=32, layers=32) / 1e9
        assert gb == pytest.approx(104.2, abs=0.1)

    def test_given_attention_that_never_stores_its_score_matrix_activations_drop_to_about_18_gb(self):
        gb = activation_bytes(seq=4096, batch=1, hidden=4096, heads=32, layers=32, store_scores=False) / 1e9
        assert gb == pytest.approx(18.3, abs=0.1)


    def test_given_activation_checkpointing_only_each_layers_input_is_kept_about_1_gb(self):
        # 2 bytes · 4096 tokens · 4096 wide · 32 layers; everything else is recomputed during the backward pass.
        gb = activation_bytes(seq=4096, batch=1, hidden=4096, heads=32, layers=32, checkpointed=True) / 1e9
        assert gb == pytest.approx(1.07, abs=0.01)


class TestDataParallelism:
    def test_given_a_batch_split_across_four_workers_the_averaged_gradient_equals_the_full_batch_gradient(self):
        rng = np.random.default_rng(0)
        X, y, w = rng.standard_normal((32, 3)), rng.standard_normal(32), rng.standard_normal(3)
        shards = [linear_regression_gradient(Xs, ys, w) for Xs, ys in zip(np.split(X, 4), np.split(y, 4))]
        assert np.allclose(np.mean(shards, axis=0), linear_regression_gradient(X, y, w))

    def test_given_a_ring_all_reduce_every_worker_ends_with_the_sum_of_all_gradients(self):
        grads = [np.arange(8.0) * (i + 1) for i in range(4)]  # sum is arange(8) * 10
        results, _ = ring_all_reduce(grads)
        assert all(np.array_equal(r, np.arange(8.0) * 10) for r in results)

    def test_given_a_ring_of_four_each_worker_sends_one_and_a_half_times_its_gradient(self):
        # 2 · (N - 1) / N = 2 · 3 / 4 = 1.5 gradients' worth, whatever the ring size.
        _, sent = ring_all_reduce([np.ones(8) for _ in range(4)])
        assert sent == [12, 12, 12, 12]

    def test_given_a_thousand_workers_the_traffic_per_worker_stays_under_twice_the_gradient(self):
        assert all_reduce_traffic(14e9, n=1000) == pytest.approx(27.972e9)


class TestShardedOptimizerState:
    # Rajbhandari et al. (2019), Figure 1: a 7.5B model on 64 GPUs.
    def test_given_no_sharding_every_gpu_holds_all_120_gb(self):
        assert zero_memory_per_gpu(7.5e9, n_gpus=64, stage=0) / 1e9 == pytest.approx(120.0)

    def test_given_sharded_optimizer_state_each_gpu_holds_31_point_4_gb(self):
        assert zero_memory_per_gpu(7.5e9, n_gpus=64, stage=1) / 1e9 == pytest.approx(31.4, abs=0.05)

    def test_given_sharded_optimizer_state_and_gradients_each_gpu_holds_16_point_6_gb(self):
        assert zero_memory_per_gpu(7.5e9, n_gpus=64, stage=2) / 1e9 == pytest.approx(16.6, abs=0.05)

    def test_given_everything_sharded_each_gpu_holds_under_2_gb(self):
        assert zero_memory_per_gpu(7.5e9, n_gpus=64, stage=3) / 1e9 == pytest.approx(1.875)


class TestTensorParallelism:
    def test_given_a_weight_split_by_columns_across_four_devices_the_joined_result_equals_the_unsplit_matmul(self):
        rng = np.random.default_rng(1)
        X, W = rng.standard_normal((5, 8)), rng.standard_normal((8, 12))
        assert np.allclose(column_parallel_matmul(X, W, n=4), X @ W)

    def test_given_a_weight_split_by_rows_the_summed_partial_results_equal_the_unsplit_matmul(self):
        rng = np.random.default_rng(2)
        X, W = rng.standard_normal((5, 8)), rng.standard_normal((8, 12))
        assert np.allclose(row_parallel_matmul(X, W, n=4), X @ W)

    def test_given_a_feed_forward_layer_split_column_then_row_it_matches_the_single_device_layer(self):
        rng = np.random.default_rng(3)
        X, W1, W2 = rng.standard_normal((5, 8)), rng.standard_normal((8, 32)), rng.standard_normal((32, 8))
        assert np.allclose(tensor_parallel_mlp(X, W1, W2, n=4), np.maximum(X @ W1, 0) @ W2)


class TestPipelineParallelism:
    def test_given_4_stages_and_8_microbatches_idle_slots_in_the_schedule_are_3_elevenths(self):
        grid = pipeline_schedule(stages=4, microbatches=8)
        assert np.mean(grid == 0) == pytest.approx(3 / 11)

    def test_given_4_stages_and_one_microbatch_three_quarters_of_the_time_is_bubble(self):
        assert bubble_fraction(stages=4, microbatches=1) == pytest.approx(0.75)

    def test_given_32_microbatches_the_bubble_shrinks_below_nine_percent(self):
        assert bubble_fraction(stages=4, microbatches=32) == pytest.approx(3 / 35)


class TestNumberFormats:
    def test_given_random_values_fp16_rounding_matches_numpys_float16(self):
        x = np.random.default_rng(0).lognormal(mean=-5, sigma=6, size=2000) * np.sign(np.random.default_rng(1).standard_normal(2000))
        with np.errstate(over="ignore"):
            expected = x.astype(np.float16).astype(np.float64)
        assert np.array_equal(quantize(x, FP16), expected)

    def test_given_one_third_bf16_keeps_only_about_three_significant_digits(self):
        # 1/3 = 1.0101010101...₂ × 2⁻²; 7 fraction bits round up to 1.0101011₂ × 2⁻² = 0.333984375.
        assert quantize(1 / 3, BF16) == 0.333984375

    def test_given_70000_fp16_overflows_to_infinity(self):
        assert quantize(70000.0, FP16) == np.inf

    def test_given_70000_bf16_holds_it_because_it_has_fp32s_range(self):
        assert quantize(70000.0, BF16) == 70144.0

    def test_given_fp8_e4m3_its_largest_value_is_448(self):
        assert FP8_E4M3.max_value == 448.0
        assert quantize(448.0, FP8_E4M3) == 448.0


class TestLossScaling:
    def test_given_a_gradient_of_one_hundred_millionth_fp16_flushes_it_to_zero(self):
        # The smallest fp16 value is about 6e-8; 1e-8 is under half of it, so it rounds to zero.
        assert quantize(1e-8, FP16) == 0.0

    def test_given_a_loss_scale_of_65536_the_same_gradient_survives_fp16_and_unscales_correctly(self):
        assert scaled_gradient_roundtrip(1e-8, scale=65536, fmt=FP16) == pytest.approx(1e-8, rel=1e-3)

    def test_given_bf16_the_tiny_gradient_survives_without_any_scaling(self):
        assert quantize(1e-8, BF16) == pytest.approx(1e-8, rel=4e-3)

    def test_given_an_overflowing_step_the_scaler_skips_it_and_halves_the_scale(self):
        scaler = DynamicLossScaler(scale=65536.0, growth_interval=3)
        assert scaler.update(grads_finite=False) is False
        assert scaler.scale == 32768.0

    def test_given_enough_clean_steps_in_a_row_the_scaler_doubles_the_scale(self):
        scaler = DynamicLossScaler(scale=1024.0, growth_interval=3)
        for _ in range(3):
            scaler.update(grads_finite=True)
        assert scaler.scale == 2048.0


class TestMasterWeights:
    def test_given_a_thousand_tiny_updates_an_fp32_master_copy_accumulates_all_of_them(self):
        assert accumulate_updates(w0=1.0, update=1e-4, steps=1000, fmt=FP32) == pytest.approx(1.1, abs=1e-4)

    def test_given_the_same_updates_applied_to_bf16_weights_every_one_is_lost(self):
        # Next to 1.0 the bf16 grid spacing is 2⁻⁷ ≈ 0.0078; adding 0.0001 rounds straight back to 1.0.
        assert accumulate_updates(w0=1.0, update=1e-4, steps=1000, fmt=BF16) == 1.0


class TestGradientClippingAtScale:
    def test_given_gradients_sharded_across_gpus_the_combined_norm_equals_the_norm_of_the_whole(self):
        rng = np.random.default_rng(4)
        full = rng.standard_normal(100)
        assert sharded_global_norm(np.split(full, 4)) == pytest.approx(np.linalg.norm(full))

    def test_given_one_corrupted_batch_without_clipping_the_loss_spikes_above_a_thousand(self):
        assert max(train_through_a_bad_batch(clip=None)) > 1000

    def test_given_the_same_batch_with_clipping_the_loss_barely_moves(self):
        losses = train_through_a_bad_batch(clip=1.0)
        assert max(losses[30:]) < 0.1


class TestSpikeDetection:
    def test_given_a_loss_that_jumps_to_triple_its_recent_median_that_step_is_flagged(self):
        losses = [3.0, 2.9, 2.8, 2.8, 2.7, 8.5, 4.0, 2.7]
        assert detect_spikes(losses, window=4, factor=2.0) == [5]


class TestCheckpointing:
    def test_given_a_1_minute_save_and_a_failure_every_3_hours_the_best_interval_is_about_19_minutes(self):
        # Young (1974): sqrt(2 · 1 · 180) = sqrt(360) ≈ 18.97 minutes.
        assert optimal_checkpoint_interval(cost=1.0, mtbf=180.0) == pytest.approx(18.97, abs=0.01)

    def test_given_the_best_interval_saving_twice_as_often_or_half_as_often_wastes_more_time(self):
        best = optimal_checkpoint_interval(cost=1.0, mtbf=180.0)
        at_best = wasted_fraction(best, cost=1.0, mtbf=180.0)
        assert at_best < wasted_fraction(best / 2, cost=1.0, mtbf=180.0)
        assert at_best < wasted_fraction(best * 2, cost=1.0, mtbf=180.0)
