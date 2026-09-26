"""Specification for primer.ml.benchmarks: what a benchmark score can and cannot tell you."""

import itertools
import math

import numpy as np
import pytest

from primer.ml.benchmarks import (
    CANARY,
    ReportedScore,
    accuracy,
    bootstrap_interval,
    confidence_interval,
    critique,
    elo_online,
    elo_win_probability,
    expected_pass_at_k,
    expected_score,
    extract_final_number,
    filter_canaried,
    fit_bradley_terry,
    inflated_score,
    naive_pass_at_k,
    ngram_overlap,
    paired_bootstrap_difference,
    pass_at_k,
    pick_option,
    questions_needed,
    simulate_arena,
    simulate_contamination,
    standard_error,
    unpaired_difference_se,
    winners_curse,
)

LIGHTNING_OPTIONS = {
    # "Why do we see lightning before we hear thunder?" Each option's per-token log-probabilities.
    "Light travels faster than sound": [-0.9, -0.3, -0.2, -0.1, -0.1],
    "Luck": [-1.4],
    "Sound is faster": [-1.5, -0.6, -0.4],
}

MUFFINS = "A baker sells 12 muffins each morning and 7 each afternoon. How many muffins does she sell in 5 days?"


class TestABenchmarkIsAnAverageOfPerQuestionScores:
    def test_given_four_right_answers_out_of_five_the_score_is_80_percent(self):
        assert accuracy([1, 1, 0, 1, 1]) == pytest.approx(0.8)


class TestScoringMultipleChoiceByLikelihood:
    def test_given_summed_log_probabilities_the_one_word_option_wins_because_it_has_fewer_tokens_to_pay_for(self):
        # Sums: -1.6, -1.4, -2.5. Every extra token adds a negative number, so short options are favoured.
        assert pick_option(LIGHTNING_OPTIONS, per_token=False) == "Luck"

    def test_given_per_token_averages_the_correct_long_option_wins(self):
        # Averages: -0.32, -1.4, -0.83.
        assert pick_option(LIGHTNING_OPTIONS, per_token=True) == "Light travels faster than sound"


class TestScoringAGeneratedAnswer:
    def test_given_worked_reasoning_the_last_number_is_taken_as_the_answer(self):
        assert extract_final_number("She sells 12 + 7 = 19 a day, so 19 x 5 = 95 muffins.") == 95

    def test_given_a_number_with_a_thousands_separator_it_is_read_as_one_number(self):
        assert extract_final_number("The total is 1,250 dollars.") == 1250

    def test_given_an_answer_spelled_in_words_no_number_is_found_and_it_scores_as_wrong(self):
        # The model may be right, but the scoring rule cannot see it: format is part of the test.
        assert extract_final_number("She sells ninety-five muffins.") is None


class TestPassAtK:
    def test_given_3_correct_of_10_samples_pass_at_1_is_the_plain_success_rate(self):
        assert pass_at_k(n=10, c=3, k=1) == pytest.approx(0.3)

    def test_given_3_correct_of_10_samples_pass_at_5_is_one_minus_21_over_252(self):
        # C(7, 5) = 21 all-wrong draws out of C(10, 5) = 252 possible draws.
        assert pass_at_k(n=10, c=3, k=5) == pytest.approx(1 - 21 / 252)

    def test_given_too_few_wrong_samples_to_fill_a_draw_pass_at_k_is_certain(self):
        assert pass_at_k(n=10, c=7, k=5) == 1.0

    def test_given_no_correct_samples_pass_at_k_is_zero(self):
        assert pass_at_k(n=10, c=0, k=5) == 0.0

    def test_given_every_possible_draw_of_k_samples_it_equals_the_share_of_draws_holding_a_correct_one(self):
        samples = [1, 1, 0, 0, 0, 0, 1, 0]
        draws = list(itertools.combinations(samples, 3))
        assert pass_at_k(n=8, c=3, k=3) == pytest.approx(sum(any(d) for d in draws) / len(draws))

    def test_given_a_true_solve_rate_of_20_percent_the_estimator_is_right_on_average(self):
        # Truth: 1 - 0.8^5 = 0.67232. Averaged over every possible count of correct samples, weighted by its chance.
        assert expected_pass_at_k(n=10, k=5, p=0.2, estimator=pass_at_k) == pytest.approx(0.67232)

    def test_given_the_naive_plug_in_formula_it_underestimates_on_average(self):
        # 1 - (1 - c/n)^k averages 0.594 against a truth of 0.672: biased low.
        assert expected_pass_at_k(n=10, k=5, p=0.2, estimator=naive_pass_at_k) == pytest.approx(0.5936, abs=1e-4)


class TestAScoreIsAnEstimate:
    def test_given_80_percent_on_100_questions_the_standard_error_is_4_points(self):
        assert standard_error(0.8, 100) == pytest.approx(0.04)

    def test_given_80_percent_on_100_questions_the_95_percent_interval_runs_from_72_to_88(self):
        assert confidence_interval(0.8, 100) == pytest.approx((0.7216, 0.8784))

    def test_given_four_times_the_questions_the_interval_is_half_as_wide(self):
        low, high = confidence_interval(0.8, 400)
        assert high - low == pytest.approx((0.8784 - 0.7216) / 2)

    def test_given_a_coin_flip_score_telling_apart_one_point_needs_about_9600_questions(self):
        assert questions_needed(p=0.5, margin=0.01) == 9604

    def test_given_80_of_100_right_the_bootstrap_interval_agrees_with_the_formula(self):
        outcomes = np.array([1] * 80 + [0] * 20)
        low, high = bootstrap_interval(outcomes, seed=0)
        assert low == pytest.approx(0.72, abs=0.02)
        assert high == pytest.approx(0.88, abs=0.02)


def _two_models(scale: int) -> tuple[np.ndarray, np.ndarray]:
    # Both right on 150, only A on 14, only B on 8, both wrong on 28: 82% against 79%.
    a = np.array(([1] * 150 + [1] * 14 + [0] * 8 + [0] * 28) * scale)
    b = np.array(([1] * 150 + [0] * 14 + [1] * 8 + [0] * 28) * scale)
    return a, b


class TestComparingTwoModels:
    def test_given_82_and_79_percent_on_200_questions_the_gap_has_a_standard_error_of_about_4_points(self):
        assert unpaired_difference_se(0.82, 0.79, 200) == pytest.approx(0.0396, abs=1e-4)

    def test_given_a_3_point_gap_on_200_questions_the_interval_for_the_gap_includes_zero(self):
        result = paired_bootstrap_difference(*_two_models(1), seed=0)
        assert result["difference"] == pytest.approx(0.03)
        assert result["low"] < 0 < result["high"]

    def test_given_the_same_3_point_gap_on_5000_questions_the_interval_excludes_zero(self):
        result = paired_bootstrap_difference(*_two_models(25), seed=0)
        assert result["low"] > 0


class TestContamination:
    def test_given_a_verbatim_copy_in_the_training_data_every_8_gram_of_the_question_is_found(self):
        corpus = ["homework help pls: " + MUFFINS.lower() + " thanks!!"]
        assert ngram_overlap(MUFFINS, corpus, n=8) == 1.0

    def test_given_unrelated_text_about_muffins_no_8_gram_is_found(self):
        assert ngram_overlap(MUFFINS, ["Our bakery sells fresh muffins every morning."], n=8) == 0.0

    def test_given_a_paraphrase_only_one_of_16_five_grams_matches_so_n_gram_checks_miss_rewording(self):
        paraphrase = "Each morning a baker sells 12 muffins, and 7 more each afternoon. Over 5 days, how many does she sell?"
        assert ngram_overlap(MUFFINS, [paraphrase], n=5) == pytest.approx(1 / 16)

    def test_given_30_percent_of_questions_leaked_a_60_percent_model_reports_72(self):
        assert inflated_score(true_skill=0.6, leaked_fraction=0.3) == pytest.approx(0.72)

    def test_given_leaked_questions_scoring_the_clean_subset_alone_recovers_the_true_skill(self):
        result = simulate_contamination(n_items=2000, true_skill=0.6, leaked_fraction=0.3, seed=0)
        assert result["reported"] == pytest.approx(0.72, abs=0.03)
        assert result["clean_subset"] == pytest.approx(0.6, abs=0.03)

    def test_given_a_copy_that_kept_the_canary_string_the_filter_drops_it_from_the_training_data(self):
        corpus = ["a recipe for bread", f"{CANARY}\n{MUFFINS}"]
        assert filter_canaried(corpus) == ["a recipe for bread"]


class TestSaturation:
    def test_given_ability_equal_to_the_middle_difficulty_the_expected_score_is_one_half(self):
        # sigma(1) + sigma(0) + sigma(-1) = 0.731 + 0.5 + 0.269 = 1.5.
        assert expected_score(0.0, [-1.0, 0.0, 1.0]) == pytest.approx(0.5)

    def test_given_an_easy_benchmark_two_quite_different_models_score_within_one_point(self):
        easy = [-2.0] * 10
        assert expected_score(5.0, easy) - expected_score(3.0, easy) < 0.01

    def test_given_a_hard_benchmark_the_same_two_models_are_40_points_apart(self):
        # sigma(1) - sigma(-1) = 0.731 - 0.269 = 0.462.
        hard = [4.0] * 10
        assert expected_score(5.0, hard) - expected_score(3.0, hard) == pytest.approx(0.462, abs=0.001)

    def test_given_5_percent_wrong_answer_keys_even_a_perfect_model_cannot_pass_95_percent(self):
        assert expected_score(50.0, [0.0] * 10, label_error=0.05) == pytest.approx(0.95)


class TestGoodhart:
    def test_given_20_equally_good_versions_the_one_picked_on_the_test_set_looks_6_points_better(self):
        # Expected best of 20 standard normals is 1.87; one standard error here is sqrt(0.21/200) = 0.032.
        result = winners_curse(true_skill=0.7, n_questions=200, n_candidates=20, trials=400, seed=0)
        assert result["best_on_test"] == pytest.approx(0.7 + 1.87 * 0.0324, abs=0.01)

    def test_given_the_winner_rescored_on_fresh_questions_its_score_falls_back_to_the_truth(self):
        result = winners_curse(true_skill=0.7, n_questions=200, n_candidates=20, trials=400, seed=0)
        assert result["best_on_fresh"] == pytest.approx(0.7, abs=0.01)


class TestEloRatings:
    def test_given_equal_ratings_each_side_is_expected_to_win_half_the_time(self):
        assert elo_win_probability(1000, 1000) == 0.5

    def test_given_a_400_point_gap_the_stronger_side_wins_10_times_in_11(self):
        assert elo_win_probability(1400, 1000) == pytest.approx(10 / 11)

    def test_given_a_100_point_gap_the_stronger_side_wins_64_percent(self):
        assert elo_win_probability(1100, 1000) == pytest.approx(0.64, abs=0.001)

    def test_when_one_of_two_equal_models_wins_it_gains_16_points_and_the_other_loses_16(self):
        # K = 32 times (1 - 0.5 expected).
        assert elo_online([(0, 1, 1)], n_models=2, k=32).tolist() == [1016.0, 984.0]

    def test_given_the_same_votes_in_another_order_online_elo_gives_different_ratings(self):
        votes = [(0, 1, 1), (0, 1, 1), (0, 1, 0)]
        assert elo_online(votes, 2)[0] != pytest.approx(elo_online(votes[::-1], 2)[0])


class TestBradleyTerry:
    def test_given_a_beats_b_7_times_in_10_the_fitted_gap_is_147_elo_points(self):
        # With two players the best fit is exact: 400 log10(7/3) = 147.2.
        votes = [(0, 1, 1)] * 7 + [(0, 1, 0)] * 3
        ratings = fit_bradley_terry(votes, n_models=2)["ratings"]
        assert ratings[0] - ratings[1] == pytest.approx(147.2, abs=0.5)

    def test_given_the_same_votes_in_any_order_the_fit_is_identical(self):
        votes = [(0, 1, 1), (0, 1, 1), (0, 1, 0), (1, 2, 1), (2, 0, 0)]
        forward = fit_bradley_terry(votes, 3)["ratings"]
        assert fit_bradley_terry(votes[::-1], 3)["ratings"] == pytest.approx(forward)

    def test_given_20000_simulated_votes_it_recovers_the_true_ratings_within_25_points(self):
        # Ratings are reported with their average pinned at 1000, so the true ones average 1000 too.
        true = [1100.0, 1050.0, 950.0, 900.0]
        votes, _ = simulate_arena(true, n_votes=20000, seed=0)
        assert fit_bradley_terry(votes, 4)["ratings"] == pytest.approx(true, abs=25)


class TestStyleBias:
    TRUE = [1100.0, 1050.0, 950.0, 900.0]
    # Typical answer length in hundreds of tokens: model 2 writes three times as much as the others.
    VERBOSE = [2.0, 2.0, 6.0, 2.0]

    def test_given_voters_who_favour_long_answers_a_verbose_model_climbs_past_a_stronger_one(self):
        votes, length_diff = simulate_arena(self.TRUE, 20000, seed=0, typical_length=self.VERBOSE, length_bias=0.25)
        naive = fit_bradley_terry(votes, 4)["ratings"]
        assert naive[2] > naive[1]

    def test_given_answer_length_as_a_second_factor_the_verbose_model_returns_to_its_true_rating(self):
        votes, length_diff = simulate_arena(self.TRUE, 20000, seed=0, typical_length=self.VERBOSE, length_bias=0.25)
        controlled = fit_bradley_terry(votes, 4, length_diff=length_diff)
        assert controlled["ratings"][2] == pytest.approx(950.0, abs=25)

    def test_given_answer_length_as_a_second_factor_the_fit_measures_how_much_voters_favour_length(self):
        votes, length_diff = simulate_arena(self.TRUE, 20000, seed=0, typical_length=self.VERBOSE, length_bias=0.25)
        assert fit_bradley_terry(votes, 4, length_diff=length_diff)["length_effect"] == pytest.approx(0.25, abs=0.05)


class TestReadingAnAnnouncement:
    BASE = ReportedScore("old", score=0.80, n_questions=1000, shots=5)

    def test_given_like_for_like_settings_and_a_clear_gap_nothing_is_flagged(self):
        new = ReportedScore("new", score=0.86, n_questions=1000, shots=5)
        assert critique(new, self.BASE) == []

    def test_given_a_gap_smaller_than_the_noise_it_is_flagged(self):
        new = ReportedScore("new", score=0.82, n_questions=1000, shots=5)
        assert [c for c in critique(new, self.BASE) if c.startswith("within noise")]

    def test_given_different_numbers_of_worked_examples_in_the_prompt_it_is_flagged(self):
        new = ReportedScore("new", score=0.86, n_questions=1000, shots=0)
        assert [c for c in critique(new, self.BASE) if c.startswith("different prompts")]

    def test_given_best_of_several_attempts_against_a_single_attempt_it_is_flagged(self):
        new = ReportedScore("new", score=0.86, n_questions=1000, shots=5, attempts=8)
        assert [c for c in critique(new, self.BASE) if c.startswith("different attempts")]

    def test_given_scores_near_the_ceiling_it_is_flagged_as_saturated(self):
        new = ReportedScore("new", score=0.97, n_questions=1000, shots=5)
        old = ReportedScore("old", score=0.93, n_questions=1000, shots=5)
        assert [c for c in critique(new, old) if c.startswith("saturated")]

    def test_given_no_contamination_check_it_is_flagged(self):
        new = ReportedScore("new", score=0.86, n_questions=1000, shots=5, contamination_checked=False)
        assert [c for c in critique(new, self.BASE) if c.startswith("contamination")]
