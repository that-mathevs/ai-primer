"""Specification for primer.ml.regularization: learning the pattern instead of memorising the examples."""

import numpy as np
import pytest

from primer.ml import regularization as reg


class TestMemorising:
    # Training points (0,0), (1,1), (2,0), (3,1): a zig-zag around 0.5.
    def test_a_cubic_through_the_four_zigzag_points_has_zero_training_error(self):
        assert reg.zigzag_fit_error(degree=3) == pytest.approx(0.0, abs=1e-12)

    def test_the_memorising_cubic_predicts_8_at_x_equals_4(self):
        # Finite differences 0,1,0,1 → 1,−1,1 → −2,2 → 4; extend the table: 6, 7, 8.
        assert reg.zigzag_predict(4.0, degree=3) == pytest.approx(8.0)

    def test_the_flat_line_at_0_5_has_training_error_0_25_but_predicts_sensibly(self):
        assert reg.zigzag_fit_error(degree=0) == pytest.approx(0.25)
        assert reg.zigzag_predict(4.0, degree=0) == pytest.approx(0.5)


class TestUnderfittingAndOverfitting:
    def test_given_12_noisy_points_a_degree_11_polynomial_fits_the_training_set_exactly(self):
        assert reg.polynomial_errors(11)["train"] < 1e-8

    def test_the_exact_fit_does_far_worse_on_new_data_than_a_cubic(self):
        assert reg.polynomial_errors(11)["val"] > 10 * reg.polynomial_errors(3)["val"]

    def test_a_straight_line_underfits_with_high_error_on_both_training_and_new_data(self):
        line, cubic = reg.polynomial_errors(1), reg.polynomial_errors(3)
        assert line["train"] > 3 * cubic["train"]
        assert line["val"] > 3 * cubic["val"]


class TestEarlyStopping:
    def test_given_patience_2_training_stops_two_epochs_after_the_best_and_keeps_the_best(self):
        # Validation loss bottoms out at epoch 3 (0.55); epochs 4 and 5 are worse, so stop at 5.
        best, stopped = reg.early_stopping([1.0, 0.8, 0.6, 0.55, 0.58, 0.65, 0.7], patience=2)
        assert (best, stopped) == (3, 5)

    def test_given_a_steadily_improving_run_early_stopping_never_triggers(self):
        best, stopped = reg.early_stopping([1.0, 0.9, 0.8, 0.7], patience=2)
        assert (best, stopped) == (3, 3)

    def test_when_a_flexible_model_trains_too_long_training_loss_keeps_falling_while_validation_loss_rises(self):
        run = reg.train_flexible_model()
        best = int(np.argmin(run["val"]))
        assert run["train"][-1] < run["train"][best]
        assert run["val"][-1] > 1.2 * run["val"][best]


class TestBiasVariance:
    def test_a_simple_model_is_systematically_off_so_its_bias_is_larger(self):
        assert reg.bias_variance(1)["bias_sq"] > 5 * reg.bias_variance(9)["bias_sq"]

    def test_a_flexible_model_swings_with_each_training_sample_so_its_variance_is_larger(self):
        assert reg.bias_variance(9)["variance"] > 5 * reg.bias_variance(1)["variance"]


class TestDropout:
    def test_given_a_drop_rate_of_one_half_every_surviving_activation_is_doubled(self):
        out = reg.dropout(np.ones(1000), p=0.5, training=True, seed=0)
        assert set(np.unique(out)) == {0.0, 2.0}

    def test_on_average_the_output_equals_the_input(self):
        # Scaling survivors by 1/(1 − p) is what keeps the expected value unchanged.
        out = reg.dropout(np.ones(200_000), p=0.5, training=True, seed=1)
        assert out.mean() == pytest.approx(1.0, abs=0.01)

    def test_at_evaluation_time_dropout_does_nothing(self):
        x = np.array([0.3, -1.2, 4.0])
        assert reg.dropout(x, p=0.5, training=False) == pytest.approx(x)


class TestL1AndL2Penalties:
    # With orthonormal features the penalised solutions have a closed form you can do by hand.
    def test_soft_thresholding_shrinks_3_to_2_and_snaps_0_5_to_exactly_zero(self):
        assert reg.soft_threshold(np.array([3.0, 0.5, -2.0]), 1.0) == pytest.approx([2.0, 0.0, -1.0])

    def test_given_plain_weights_3_0_5_and_minus_2_l1_with_strength_1_gives_2_0_and_minus_1(self):
        assert reg.penalised_weights(np.array([3.0, 0.5, -2.0]), strength=1.0, kind="l1") == pytest.approx([2.0, 0.0, -1.0])

    def test_given_the_same_weights_l2_with_strength_1_halves_each_one_and_zeroes_none(self):
        # w / (1 + λ) = w / 2.
        assert reg.penalised_weights(np.array([3.0, 0.5, -2.0]), strength=1.0, kind="l2") == pytest.approx([1.5, 0.25, -1.0])

    def test_given_ten_features_of_which_three_matter_l1_sets_most_useless_weights_to_exactly_zero(self):
        w = reg.fit_sparse_problem(kind="l1")
        assert np.sum(w[3:] == 0.0) >= 5

    def test_given_the_same_problem_l2_leaves_every_weight_nonzero(self):
        w = reg.fit_sparse_problem(kind="l2")
        assert np.all(w != 0.0)


class TestDataSplits:
    def test_given_100_examples_a_70_15_15_split_gives_70_15_and_15_with_no_overlap(self):
        tr, va, te = reg.train_val_test_split(100, val=0.15, test=0.15, seed=0)
        assert (len(tr), len(va), len(te)) == (70, 15, 15)
        assert len(set(tr) | set(va) | set(te)) == 100

    def test_in_5_fold_cross_validation_of_10_examples_every_example_is_held_out_exactly_once(self):
        held_out = np.concatenate([val for _, val in reg.k_fold(10, k=5, seed=0)])
        assert sorted(held_out.tolist()) == list(range(10))

    def test_in_5_fold_cross_validation_of_10_examples_each_model_trains_on_8(self):
        assert [len(train) for train, _ in reg.k_fold(10, k=5, seed=0)] == [8, 8, 8, 8, 8]


class TestDataLeakage:
    def test_given_duplicates_across_the_split_a_memorising_model_scores_far_higher_than_on_a_clean_split(self):
        scores = reg.duplicate_leakage_demo()
        assert scores["leaky"] > scores["clean"] + 0.2

    def test_given_a_feature_only_known_after_the_outcome_offline_accuracy_looks_near_perfect(self):
        assert reg.future_feature_demo()["offline"] > 0.95

    def test_given_the_same_model_in_production_without_that_feature_accuracy_collapses(self):
        assert reg.future_feature_demo()["production"] < 0.75
