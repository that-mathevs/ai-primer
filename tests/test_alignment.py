"""Specification for primer.ml.alignment: turning written values into training signal, and measuring where they fail."""

import pytest

from primer.ml.alignment import (
    CONSTITUTION,
    KeywordFilter,
    constitutional_preference_pairs,
    critique,
    HAND_WRITTEN_TESTS,
    attack_success_rate,
    goodhart_curve,
    patch_filter,
    red_team_attempts,
    refusal_tradeoff,
    release_gate,
    red_team_search,
    revise,
    sycophancy_flip_rate,
    sycophantic_preference,
)


class TestConstitutionalCritique:
    def test_given_a_response_that_breaks_no_principle_the_critique_finds_nothing(self):
        assert critique("Here is a clear, sourced answer.", CONSTITUTION) == []

    def test_given_a_response_that_overclaims_certainty_the_honesty_principle_flags_it(self):
        assert "honest" in critique("This is guaranteed to work, 100% certain.", CONSTITUTION)

    def test_given_two_candidates_the_one_breaking_fewer_principles_is_chosen(self):
        pairs = constitutional_preference_pairs(
            ["This is guaranteed to work, 100% certain.", "It usually works; test it first."]
        )
        assert pairs == [("It usually works; test it first.", "This is guaranteed to work, 100% certain.")]

    def test_given_two_candidates_that_tie_no_preference_pair_is_produced(self):
        # A tie carries no signal, so labelling it would only add noise to the reward model.
        assert constitutional_preference_pairs(["Answer A.", "Answer B."]) == []

    def test_given_an_overclaiming_uncited_answer_one_revision_leaves_no_principle_broken(self):
        assert critique(revise("This is guaranteed to work."), CONSTITUTION) == []


class TestRedTeaming:
    def test_given_the_exact_blocked_placeholder_the_filter_catches_it(self):
        assert KeywordFilter({"xyzzy"}).blocks("please do xyzzy")

    def test_given_a_seeded_search_the_attack_success_rate_is_between_zero_and_one(self):
        rate = red_team_search(KeywordFilter({"xyzzy"}), seed_prompt="please do xyzzy", tries=200, seed=0)
        assert 0.0 < rate < 1.0

    def test_given_the_same_seed_the_search_finds_the_same_rate(self):
        f = KeywordFilter({"xyzzy"})
        assert red_team_search(f, "please do xyzzy", 200, seed=3) == red_team_search(f, "please do xyzzy", 200, seed=3)

    def test_given_six_attempts_of_which_four_get_through_the_attack_success_rate_is_two_thirds(self):
        assert attack_success_rate([False, True, True, False, True, True]) == pytest.approx(4 / 6)

    def test_given_only_the_hand_written_tests_the_filter_looks_perfect(self):
        # Hand-written tests reuse the words the filter author already thought of.
        assert red_team_search(KeywordFilter({"xyzzy"}), HAND_WRITTEN_TESTS, tries=None, seed=0) == 0.0

    def test_given_the_filter_is_patched_with_what_the_search_found_the_same_search_no_longer_gets_through(self):
        f = KeywordFilter({"xyzzy"})
        patched = patch_filter(f, red_team_attempts(f, "please do xyzzy", 200, seed=0))
        assert red_team_search(patched, "please do xyzzy", 200, seed=1) == 0.0


class TestSycophancy:
    def test_given_a_model_that_never_defers_its_flip_rate_is_zero(self):
        assert sycophancy_flip_rate(deference=0.0, trials=500, seed=0) == 0.0

    def test_given_a_model_that_defers_half_the_time_about_half_its_answers_flip(self):
        assert sycophancy_flip_rate(deference=0.5, trials=2000, seed=0) == pytest.approx(0.5, abs=0.05)

    def test_given_raters_reward_agreement_by_2_and_correctness_by_1_the_agreeing_answer_wins_73_percent(self):
        # sigma(2 - 1) = 0.731: the Bradley-Terry model from primer.ml.training_stages.
        assert sycophantic_preference(agreement_bonus=2.0, correctness_gap=1.0) == pytest.approx(0.731, abs=0.001)


class TestRefusalTradeoff:
    def test_given_a_stricter_threshold_benign_refusals_rise_and_harmful_compliance_falls(self):
        loose, strict = refusal_tradeoff(thresholds=[0.9, 0.1], seed=0)
        assert strict["over_refusal"] > loose["over_refusal"]
        assert strict["harmful_compliance"] < loose["harmful_compliance"]

    def test_given_the_hand_checked_scores_a_threshold_of_half_gives_one_third_on_each_error(self):
        (row,) = refusal_tradeoff(thresholds=[0.5], benign=[0.1, 0.3, 0.6], off_limits=[0.4, 0.8, 0.9])
        assert (row["over_refusal"], row["harmful_compliance"]) == (pytest.approx(1 / 3), pytest.approx(1 / 3))


class TestGoodhart:
    def test_given_more_optimization_the_proxy_keeps_rising_while_the_true_goal_eventually_falls(self):
        steps = goodhart_curve(steps=50)
        assert steps[-1]["proxy"] > steps[0]["proxy"]
        assert steps[-1]["true"] < max(s["true"] for s in steps)

    def test_given_the_toy_curves_the_true_goal_peaks_near_step_16(self):
        # d/ds (1 - e^(-s/10)) = 1/50 at s = 10 ln 5 = 16.1.
        steps = goodhart_curve(steps=50)
        assert max(steps, key=lambda s: s["true"])["step"] == 16


class TestReleaseGate:
    def test_given_every_measurement_within_its_limit_the_release_passes(self):
        assert release_gate({"attack_success": 0.02}, {"attack_success": 0.05}) == (True, [])

    def test_given_one_measurement_over_its_limit_the_release_is_held_and_the_reason_is_named(self):
        ok, reasons = release_gate({"attack_success": 0.02, "sycophancy": 0.3}, {"attack_success": 0.05, "sycophancy": 0.2})
        assert (ok, reasons) == (False, ["sycophancy 0.3 > limit 0.2"])
