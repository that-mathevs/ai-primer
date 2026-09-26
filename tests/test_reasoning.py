"""Specification for primer.ml.reasoning: how writing steps, sampling, checking and RL make a model reason."""

import numpy as np
import pytest

from primer.ml.reasoning import (
    budget_accuracy,
    budget_sweep,
    check_step,
    correlated_vote_accuracy,
    first_bad_step,
    group_advantages,
    majority_accuracy,
    majority_vote,
    outcome_reward,
    pass_at_n,
    reasoning_cost,
    solve,
    steps_all_right,
    train_reasoner,
    verifier_experiment,
)


class TestThinkingOutLoud:
    # The toy model does one addition per forward pass, and every token it writes is one forward pass.

    def test_given_3_5_8_2_answered_in_one_token_the_model_runs_out_of_steps_and_guesses_17(self):
        # One pass adds 3 + 5 = 8; the two numbers left are estimated at 4.5 each: 8 + 9 = 17.
        attempt = solve([3, 5, 8, 2], max_tokens=1)
        assert attempt.answer == 17
        assert attempt.steps == []

    def test_given_room_to_write_steps_the_model_writes_running_totals_and_gets_18(self):
        attempt = solve([3, 5, 8, 2], max_tokens=10)
        assert attempt.steps == ["3+5=8", "8+8=16"]
        assert attempt.answer == 18

    def test_given_room_to_write_steps_it_stops_once_done_so_a_3_addition_sum_costs_3_tokens(self):
        assert solve([3, 5, 8, 2], max_tokens=10).tokens == 3

    def test_given_two_numbers_one_forward_pass_is_enough_to_answer_directly(self):
        assert solve([4, 7], max_tokens=1).answer == 11


class TestThinkingBudget:
    def test_given_six_difficulties_and_a_budget_of_8_tokens_the_formula_gives_70_percent(self):
        # 4 of 6 difficulties (1, 2, 4, 8 additions) fit; the other 2/6 are guessed right 10% of the time.
        assert budget_accuracy(8, needs=(1, 2, 4, 8, 16, 32), lucky=0.1) == pytest.approx(0.70)

    def test_given_difficulties_spaced_by_doubling_each_doubling_of_budget_adds_the_same_accuracy(self):
        gains = np.diff([budget_accuracy(b, needs=(1, 2, 4, 8, 16, 32), lucky=0.1) for b in (1, 2, 4, 8, 16, 32)])
        assert gains == pytest.approx([0.15] * 5)

    def test_given_a_budget_past_the_hardest_problem_more_tokens_buy_no_accuracy(self):
        assert budget_accuracy(64, needs=(1, 2, 4, 8, 16, 32), lucky=0.1) == budget_accuracy(32, needs=(1, 2, 4, 8, 16, 32), lucky=0.1)

    def test_given_real_sums_accuracy_climbs_from_under_a_third_to_every_problem_as_the_budget_grows(self):
        rows = budget_sweep(budgets=(1, 32), n_problems=600, seed=0)
        assert rows[0]["accuracy"] < 1 / 3
        assert rows[1]["accuracy"] == 1.0

    def test_given_real_sums_the_simulation_lands_on_the_formula_within_a_few_points(self):
        # The only gap is the lucky-guess rate, which the formula takes as a fixed 10%.
        for row in budget_sweep(budgets=(1, 4, 16), n_problems=1200, seed=1):
            assert row["accuracy"] == pytest.approx(budget_accuracy(row["budget"], lucky=0.1), abs=0.05)


class TestSelfConsistency:
    def test_given_answers_18_17_18_the_vote_picks_18(self):
        assert majority_vote([18, 17, 18]) == 18

    def test_given_a_solver_right_60_percent_of_the_time_three_votes_are_right_64_8_percent(self):
        # 3·0.6²·0.4 + 0.6³ = 0.432 + 0.216, by hand.
        assert majority_accuracy(0.6, 3) == pytest.approx(0.648)

    def test_given_a_solver_right_60_percent_of_the_time_five_votes_are_right_68_3_percent(self):
        assert majority_accuracy(0.6, 5) == pytest.approx(0.68256)

    def test_given_a_solver_right_less_than_half_the_time_on_a_yes_no_question_voting_makes_it_worse(self):
        assert majority_accuracy(0.4, 5) < 0.4

    def test_given_wrong_answers_that_scatter_a_40_percent_solver_still_wins_the_vote_most_of_the_time(self):
        # The right answer only needs to be the most common one, not a majority.
        assert correlated_vote_accuracy(p=0.4, n=15, rho=0.0, n_wrong=10, trials=1500, seed=0) > 0.8

    def test_given_independent_samples_31_votes_push_a_60_percent_solver_above_95_percent(self):
        assert correlated_vote_accuracy(p=0.6, n=31, rho=0.0, n_wrong=5, trials=1500, seed=0) > 0.95

    def test_given_samples_that_share_the_same_mistake_31_votes_barely_beat_one(self):
        # With rho = 0.9 almost every sample repeats one shared draw, so voting just repeats it.
        acc = correlated_vote_accuracy(p=0.6, n=31, rho=0.9, n_wrong=5, trials=1500, seed=0)
        assert acc == pytest.approx(0.6, abs=0.05)


class TestVerifiers:
    def test_given_a_step_whose_arithmetic_holds_the_process_verifier_accepts_it(self):
        assert check_step("8+8=16")

    def test_given_a_step_whose_arithmetic_is_off_the_process_verifier_rejects_it(self):
        assert not check_step("8+8=17")

    def test_given_a_chain_with_a_slip_in_step_two_the_process_verifier_points_at_step_two(self):
        assert first_bad_step(["3+5=8", "8+8=17", "17+2=19"]) == 1

    def test_given_a_clean_chain_the_process_verifier_finds_no_bad_step(self):
        assert first_bad_step(["3+5=8", "8+8=16", "16+2=18"]) is None

    def test_given_two_slips_that_cancel_the_outcome_reward_is_full_but_the_process_verifier_objects(self):
        # Right answer, wrong reasoning: an outcome check can't tell, a step check can.
        lucky = ["3+5=9", "9+8=16", "16+2=18"]
        assert outcome_reward(lucky, reference=18) == 1.0
        assert first_bad_step(lucky) == 0

    def test_given_samples_right_30_percent_of_the_time_one_of_five_is_right_83_percent_of_the_time(self):
        # 1 - 0.7⁵ = 1 - 0.16807.
        assert pass_at_n(0.3, 5) == pytest.approx(0.83193)

    def test_given_noisy_chains_best_of_n_with_a_step_checker_beats_voting_and_a_single_sample(self):
        row = verifier_experiment(ns=(16,), trials=400, seed=0)[0]
        assert row["verifier"] > row["vote"] > row["single"]

    def test_given_a_perfect_step_checker_best_of_n_gets_within_a_few_points_of_the_pass_at_n_ceiling(self):
        # The gap is chains whose slips cancelled: right by luck, so pass@n counts them and the checker refuses them.
        row = verifier_experiment(ns=(8,), trials=400, seed=0)[0]
        assert row["pass_at_n"] - 0.05 < row["verifier"] <= row["pass_at_n"]


class TestLearningToReasonWithRL:
    def test_given_rewards_1_0_0_1_group_advantages_are_plus_one_and_minus_one(self):
        assert group_advantages(np.array([1.0, 0.0, 0.0, 1.0])) == pytest.approx([1.0, -1.0, -1.0, 1.0])

    def test_given_a_group_that_all_succeeds_there_is_no_signal_to_learn_from(self):
        # Problems the model always solves (or always fails) teach it nothing.
        assert group_advantages(np.array([1.0, 1.0, 1.0, 1.0])).tolist() == [0.0, 0.0, 0.0, 0.0]

    def test_given_rewards_that_differ_by_a_hair_dividing_by_the_spread_blows_them_up_to_plus_and_minus_one(self):
        # 0.99 vs 0.98: a 0.01 difference becomes as loud as solved vs failed.
        assert group_advantages(np.array([0.99, 0.98])) == pytest.approx([1.0, -1.0])

    def test_given_rewards_that_differ_by_a_hair_without_dividing_by_the_spread_they_stay_a_hair_apart(self):
        assert group_advantages(np.array([0.99, 0.98]), divide_by_spread=False) == pytest.approx([0.005, -0.005])

    def test_given_only_a_correctness_reward_the_model_learns_to_write_longer_and_gets_more_right(self):
        history = train_reasoner(seed=0)
        assert history["mean_length"][-1] > 3 * history["mean_length"][0]
        assert history["accuracy"][0] < 0.5
        assert history["accuracy"][-1] > 0.9

    def test_given_a_hard_problem_the_trained_model_leaves_room_to_check_every_step(self):
        # 8 additions, each 10% likely to slip: only 16+ tokens leave room to recheck each step.
        history = train_reasoner(seed=0)
        assert history["preferred_length"][8] >= 16

    def test_given_a_length_penalty_the_model_thinks_less_on_easy_problems_than_on_hard_ones(self):
        history = train_reasoner(length_penalty=0.01, seed=0)
        assert history["preferred_length"][1] < history["preferred_length"][8]

    def test_given_a_length_penalty_and_division_by_the_groups_spread_easy_problems_shrink_to_one_token(self):
        # In a group that all succeeded, only the 0.01 penalty differs, and dividing by a tiny spread magnifies it.
        assert train_reasoner(length_penalty=0.01, seed=0)["preferred_length"][1] == 1

    def test_given_a_length_penalty_without_division_by_the_spread_easy_problems_settle_on_the_best_length_two(self):
        # Expected reward: 1 token 0.9 - 0.01 = 0.89; 2 tokens 0.99 - 0.02 = 0.97; 4 tokens 1 - 0.04 = 0.96.
        history = train_reasoner(length_penalty=0.01, divide_by_spread=False, seed=0)
        assert history["preferred_length"][1] == 2

    def test_given_a_length_penalty_the_model_spends_fewer_tokens_than_without_one(self):
        free = train_reasoner(seed=0)["mean_length"][-1]
        taxed = train_reasoner(length_penalty=0.01, seed=0)["mean_length"][-1]
        assert taxed < free


class TestWhatReasoningCosts:
    def test_given_a_2_percent_slip_per_step_a_50_step_chain_is_clean_only_36_percent_of_the_time(self):
        assert steps_all_right(0.02, 50) == pytest.approx(0.364, abs=0.001)

    def test_given_8_samples_of_2000_tokens_at_10_dollars_per_million_a_question_costs_16_cents(self):
        assert reasoning_cost(samples=8, tokens=2000, dollars_per_million=10.0, tokens_per_second=50)["dollars"] == pytest.approx(0.16)

    def test_given_parallel_samples_waiting_time_is_set_by_one_chain_not_by_how_many(self):
        # 2000 tokens at 50 per second, whether 1 or 8 chains run side by side.
        one = reasoning_cost(samples=1, tokens=2000, dollars_per_million=10.0, tokens_per_second=50)
        eight = reasoning_cost(samples=8, tokens=2000, dollars_per_million=10.0, tokens_per_second=50)
        assert one["seconds"] == eight["seconds"] == 40.0
