"""Specification for primer.ml.losses: how "how wrong were we?" becomes one number."""

import numpy as np
import pytest

from primer.ml import losses


class TestCrossEntropy:
    @pytest.mark.parametrize(
        "p_correct, expected",
        [(0.9, 0.11), (0.5, 0.69), (0.1, 2.30), (0.01, 4.61)],
    )
    def test_given_probability_p_on_the_right_token_the_loss_is_minus_ln_p(self, p_correct, expected):
        assert losses.cross_entropy_from_prob(p_correct) == pytest.approx(expected, abs=0.005)

    def test_given_1_percent_versus_90_percent_on_the_right_token_the_loss_is_about_44_times_larger(self):
        # ln(100) / ln(10/9) = 4.605 / 0.1054 = 43.7: the steep penalty that pushes models toward calibrated probabilities.
        ratio = losses.cross_entropy_from_prob(0.01) / losses.cross_entropy_from_prob(0.9)
        assert ratio == pytest.approx(43.7, abs=0.05)

    def test_given_certainty_on_the_right_answer_the_loss_is_zero(self):
        assert losses.cross_entropy_from_prob(1.0) == pytest.approx(0.0)


class TestPerplexity:
    def test_given_an_average_loss_of_0_69_perplexity_is_2_like_a_coin_flip(self):
        assert losses.perplexity(np.array([0.5, 0.5, 0.5])) == pytest.approx(2.0)

    def test_given_uniform_guesses_over_10_tokens_perplexity_is_10(self):
        assert losses.perplexity(np.full(7, 0.1)) == pytest.approx(10.0)


class TestCrossEntropyFromLogits:
    def test_given_logits_2_1_and_0_1_with_the_first_correct_the_loss_is_0_417(self):
        # Hand calculation: ln(e^2 + e^1 + e^0.1) - 2 = ln(11.212) - 2 = 0.417.
        loss, _ = losses.softmax_cross_entropy(np.array([[2.0, 1.0, 0.1]]), np.array([0]))
        assert loss == pytest.approx(0.417, abs=1e-3)

    def test_given_huge_logits_the_loss_stays_finite(self):
        # Naive softmax would compute exp(1000) = inf; log-sum-exp gives exactly 1000.
        loss, _ = losses.softmax_cross_entropy(np.array([[1000.0, 0.0]]), np.array([1]))
        assert loss == pytest.approx(1000.0)

    def test_the_gradient_is_predicted_probabilities_minus_the_one_hot_target(self):
        # softmax([0, ln 3]) = [0.25, 0.75]; target class 1 -> [0.25, 0.75 - 1].
        _, grad = losses.softmax_cross_entropy(np.array([[0.0, np.log(3.0)]]), np.array([1]))
        assert grad == pytest.approx(np.array([[0.25, -0.25]]))


class TestRegressionLosses:
    residuals_with_one_outlier = (np.zeros(5), np.array([1.0, -1.0, 1.0, -1.0, 10.0]))

    def test_mse_of_four_unit_misses_and_one_miss_of_10_is_20_8(self):
        # (1 + 1 + 1 + 1 + 100) / 5.
        assert losses.mse(*self.residuals_with_one_outlier) == pytest.approx(20.8)

    def test_mae_of_the_same_misses_is_2_8(self):
        # (1 + 1 + 1 + 1 + 10) / 5.
        assert losses.mae(*self.residuals_with_one_outlier) == pytest.approx(2.8)

    def test_under_mse_one_outlier_dominates_the_loss(self):
        assert losses.outlier_share(*self.residuals_with_one_outlier, kind="mse") == pytest.approx(100 / 104)

    def test_under_mae_the_same_outlier_has_far_less_pull(self):
        assert losses.outlier_share(*self.residuals_with_one_outlier, kind="mae") == pytest.approx(10 / 14)


class TestContrastiveInfoNCE:
    def test_given_perfectly_matched_orthogonal_pairs_the_loss_is_ln_1_plus_e_minus_1(self):
        # Each row: -ln(e^1 / (e^1 + e^0)) = ln(1 + e^-1) = 0.3133.
        loss, _, _ = losses.info_nce(np.eye(2), np.eye(2), temperature=1.0)
        assert loss == pytest.approx(0.3133, abs=1e-4)

    def test_given_swapped_pairs_the_loss_is_ln_1_plus_e(self):
        # The positive now scores 0 and the in-batch negative scores 1: ln(1 + e) = 1.3133.
        loss, _, _ = losses.info_nce(np.eye(2), np.eye(2)[::-1], temperature=1.0)
        assert loss == pytest.approx(1.3133, abs=1e-4)

    def test_a_lower_temperature_sharpens_the_reward_for_correct_pairs(self):
        # Temperature 0.1: ln(1 + e^-10) ≈ 4.5e-5.
        loss, _, _ = losses.info_nce(np.eye(2), np.eye(2), temperature=0.1)
        assert loss == pytest.approx(4.54e-5, rel=1e-2)

    def test_a_hard_negative_close_to_the_query_raises_the_loss(self):
        q = np.array([[1.0, 0.0]])
        pos = np.array([[1.0, 0.0]])
        easy = np.vstack([pos, [[0.0, 1.0]]])
        hard = np.vstack([pos, [[0.9, 0.436]]])  # nearly the same direction as the query
        loss_easy, _, _ = losses.info_nce(q, easy, temperature=0.1)
        loss_hard, _, _ = losses.info_nce(q, hard, temperature=0.1)
        assert loss_hard > 100 * loss_easy

    def test_the_analytic_gradients_match_finite_differences(self):
        rng = np.random.default_rng(0)
        q, d = rng.standard_normal((4, 3)), rng.standard_normal((4, 3))
        assert losses.info_nce_gradient_check(q, d, temperature=0.5) < 1e-6


class TestLabelSmoothing:
    def test_given_smoothing_of_0_1_over_4_classes_the_target_keeps_0_925_on_the_right_class(self):
        # 0.9 on the right class plus 0.1/4 spread over every class, including the right one.
        assert losses.smoothed_targets(np.array([2]), n_classes=4, smoothing=0.1)[0] == pytest.approx(
            [0.025, 0.025, 0.925, 0.025]
        )

    def test_given_smoothing_a_perfectly_confident_prediction_no_longer_has_zero_loss(self):
        # That residual loss discourages infinitely large logits (over-confidence).
        loss = losses.smoothed_cross_entropy(np.array([[50.0, 0.0, 0.0, 0.0]]), np.array([0]), smoothing=0.1)
        assert loss > 1.0
