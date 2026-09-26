"""Specification for primer.ml.reinforcement: learning from reward, from bandits to GRPO and reward hacking."""

import math

import numpy as np
import pytest

from primer.ml.reinforcement import (
    ADDITION_PROMPTS,
    WIN_CHANCES,
    Bandit,
    clipped_policy_update,
    expected_reward,
    fit_reward_model,
    grad_log_prob,
    gradient_estimate_stats,
    group_advantages,
    kl_penalised_reward,
    kl_regularised_optimum,
    leash_sweep,
    optimise_lengths,
    ppo_clipped_objective,
    ppo_drift_experiment,
    proxy_reward,
    reinforce_step,
    sampled_gradient_variance,
    train_grpo,
    train_reinforce,
    true_quality,
    verify,
    worked_reinforce_step,
)


class TestTheBandit:
    def test_given_an_arm_that_wins_80_percent_of_the_time_about_800_of_1000_pulls_pay_out(self):
        bandit = Bandit(WIN_CHANCES, seed=0)
        assert sum(bandit.pull(2) for _ in range(1000)) == pytest.approx(800, abs=40)

    def test_given_an_offset_every_pull_pays_the_offset_plus_the_win(self):
        # A bandit whose rewards are all large and positive is what makes baselines matter.
        rewards = {Bandit(WIN_CHANCES, offset=5.0, seed=1).pull(0) for _ in range(50)}
        assert rewards <= {5.0, 6.0}

    def test_given_a_uniform_policy_over_arms_winning_20_50_and_80_percent_the_expected_reward_is_one_half(self):
        # (0.2 + 0.5 + 0.8) / 3 = 0.5
        assert expected_reward(np.full(3, 1 / 3), np.array(WIN_CHANCES)) == pytest.approx(0.5)

    def test_given_a_policy_that_always_picks_the_best_arm_the_expected_reward_is_its_win_chance(self):
        assert expected_reward(np.array([0.0, 0.0, 1.0]), np.array(WIN_CHANCES)) == pytest.approx(0.8)


class TestTheLogProbabilityTrick:
    def test_given_a_uniform_policy_over_three_actions_the_gradient_of_log_probability_of_the_third_is_one_hot_minus_a_third(self):
        assert grad_log_prob(np.zeros(3), 2) == pytest.approx([-1 / 3, -1 / 3, 2 / 3])

    def test_given_one_rewarded_step_at_learning_rate_one_half_the_chosen_action_rises_from_a_third_to_0_452(self):
        # Hand calculation: logits become (-1/6, -1/6, 1/3); softmax gives (0.274, 0.274, 0.452).
        assert worked_reinforce_step() == pytest.approx([0.274, 0.274, 0.452], abs=5e-4)

    def test_given_a_reward_of_zero_the_policy_does_not_move(self):
        logits = np.array([0.3, -0.2, 0.1])
        assert reinforce_step(logits, action=1, reward=0.0, lr=0.5) == pytest.approx(logits)

    def test_given_the_averaged_estimate_it_equals_the_slope_of_expected_reward_measured_by_nudging_each_logit(self):
        # The trick's whole promise: sampling actions and weighting ∇log π by reward gives the true gradient.
        logits, rewards = np.array([0.5, -0.3, 0.1]), np.array([0.2, 0.5, 0.8])

        def J(z):
            p = np.exp(z) / np.exp(z).sum()
            return float(p @ rewards)

        h = 1e-6
        numeric = [(J(logits + h * np.eye(3)[k]) - J(logits - h * np.eye(3)[k])) / (2 * h) for k in range(3)]
        assert gradient_estimate_stats(logits, rewards)["mean"] == pytest.approx(numeric, abs=1e-6)


class TestLearningFromReward:
    def test_given_500_pulls_the_policy_puts_over_90_percent_on_the_arm_that_pays_most(self):
        history = train_reinforce(Bandit(WIN_CHANCES, seed=0), steps=500, lr=0.1)
        assert history["probs"][-1][2] > 0.9

    def test_given_training_the_last_hundred_pulls_earn_more_than_the_first_hundred(self):
        rewards = train_reinforce(Bandit(WIN_CHANCES, seed=0), steps=500, lr=0.1)["rewards"]
        assert rewards[-100:].mean() > rewards[:100].mean() + 0.2


class TestBaselines:
    def test_given_rewards_10_and_12_and_no_baseline_the_gradient_estimate_swings_with_variance_30_25(self):
        # Picking arm 1 gives -5, arm 2 gives +6, each half the time: variance (25 + 36)/2 - 0.5² = 30.25.
        stats = gradient_estimate_stats(np.zeros(2), np.array([10.0, 12.0]))
        assert stats["variance"][1] == pytest.approx(30.25)

    def test_given_a_baseline_of_11_the_same_estimate_has_zero_variance(self):
        # Now both picks give +0.5: every sample is the right answer.
        stats = gradient_estimate_stats(np.zeros(2), np.array([10.0, 12.0]), baseline=11.0)
        assert stats["variance"][1] == pytest.approx(0.0)

    def test_given_a_baseline_the_average_gradient_is_unchanged(self):
        # Subtracting a constant removes noise, never signal: E[∇log π] = 0.
        without = gradient_estimate_stats(np.zeros(2), np.array([10.0, 12.0]))["mean"]
        with_baseline = gradient_estimate_stats(np.zeros(2), np.array([10.0, 12.0]), baseline=11.0)["mean"]
        assert with_baseline == pytest.approx(without) and with_baseline == pytest.approx([-0.5, 0.5])

    def test_given_rewards_offset_by_5_subtracting_the_average_reward_cuts_the_sampled_variance_over_fiftyfold(self):
        bandit = Bandit(WIN_CHANCES, offset=5.0, seed=0)
        plain = sampled_gradient_variance(bandit, baseline=0.0)
        centred = sampled_gradient_variance(Bandit(WIN_CHANCES, offset=5.0, seed=0), baseline=5.5)
        assert plain > 50 * centred

    def test_given_rewards_offset_by_5_a_running_average_baseline_finds_the_best_arm_in_every_run(self):
        finals = [train_reinforce(Bandit(WIN_CHANCES, offset=5.0, seed=s), 400, 0.1, baseline=True, seed=s)["probs"][-1][2] for s in range(20)]
        assert min(finals) > 0.5

    def test_given_rewards_offset_by_5_and_no_baseline_many_runs_lock_onto_a_worse_arm(self):
        finals = [train_reinforce(Bandit(WIN_CHANCES, offset=5.0, seed=s), 400, 0.1, baseline=False, seed=s)["probs"][-1][2] for s in range(20)]
        assert np.mean(np.array(finals) < 0.5) >= 0.3


class TestPPOClipping:
    def test_given_ratio_1_5_and_advantage_2_the_objective_is_capped_at_1_2_times_2(self):
        assert ppo_clipped_objective(1.5, 2.0, eps=0.2) == pytest.approx(2.4)

    def test_given_ratio_0_5_and_advantage_minus_1_the_objective_is_capped_at_minus_0_8(self):
        assert ppo_clipped_objective(0.5, -1.0, eps=0.2) == pytest.approx(-0.8)

    def test_given_ratio_1_5_and_advantage_minus_1_the_full_penalty_minus_1_5_applies(self):
        # The min is pessimistic: making a bad action more likely is never excused by the clip.
        assert ppo_clipped_objective(1.5, -1.0, eps=0.2) == pytest.approx(-1.5)

    def test_given_a_ratio_inside_the_band_the_objective_is_ratio_times_advantage(self):
        assert ppo_clipped_objective(1.1, 3.0, eps=0.2) == pytest.approx(3.3)

    def test_given_a_ratio_past_1_plus_epsilon_with_positive_advantage_raising_it_further_gains_nothing(self):
        h = 1e-6
        slope = (ppo_clipped_objective(1.3 + h, 2.0) - ppo_clipped_objective(1.3 - h, 2.0)) / (2 * h)
        assert slope == pytest.approx(0.0)

    def test_given_fifty_passes_over_one_batch_without_clipping_the_ratio_runs_past_two(self):
        assert ppo_drift_experiment()["unclipped"].max() > 2.0

    def test_given_fifty_passes_over_one_batch_with_clipping_the_ratio_stays_below_one_and_a_half(self):
        assert ppo_drift_experiment()["clipped"].max() < 1.5

    def test_given_a_single_pass_the_ratio_starts_at_exactly_one(self):
        # Before any update the new policy is the old one, so the clip cannot fire on the first pass.
        _, ratios = clipped_policy_update(np.zeros(3), np.array([0, 2]), np.array([-1.0, 1.0]), epochs=1, lr=0.3)
        assert ratios[0] == pytest.approx([1.0, 1.0])


class TestTheKLLeash:
    def test_given_policy_probability_0_6_against_reference_0_3_and_beta_0_1_a_reward_of_1_becomes_0_931(self):
        # 1 - 0.1·ln(0.6/0.3) = 1 - 0.0693
        assert kl_penalised_reward(1.0, math.log(0.6), math.log(0.3), beta=0.1) == pytest.approx(0.9307, abs=1e-4)

    def test_given_reference_one_half_each_rewards_1_and_0_and_beta_1_the_best_policy_is_0_731_and_0_269(self):
        # π* ∝ π_ref·e^(R/β): (0.5e, 0.5) normalised is e/(e+1) = 0.731.
        assert kl_regularised_optimum(np.array([0.5, 0.5]), np.array([1.0, 0.0]), beta=1.0) == pytest.approx([0.731, 0.269], abs=5e-4)

    def test_given_a_huge_beta_the_best_policy_is_the_reference(self):
        ref = np.array([0.6, 0.3, 0.1])
        assert kl_regularised_optimum(ref, np.array([0.0, 1.0, 5.0]), beta=1e6) == pytest.approx(ref, abs=1e-5)

    def test_given_a_tiny_beta_the_best_policy_puts_everything_on_the_highest_reward(self):
        best = kl_regularised_optimum(np.array([0.6, 0.3, 0.1]), np.array([0.0, 1.0, 5.0]), beta=1e-3)
        assert best == pytest.approx([0.0, 0.0, 1.0])


class TestGroupRelativeAdvantages:
    def test_given_group_rewards_1_0_0_1_the_advantages_are_plus_one_minus_one_minus_one_plus_one(self):
        assert group_advantages(np.array([1.0, 0.0, 0.0, 1.0])) == pytest.approx([1, -1, -1, 1], abs=1e-6)

    def test_given_one_right_answer_in_four_it_earns_1_73_and_each_wrong_one_minus_0_58(self):
        # mean 0.25, std √0.1875 = 0.433: (1 - 0.25)/0.433 and (0 - 0.25)/0.433.
        assert group_advantages(np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx([1.732, -0.577, -0.577, -0.577], abs=1e-3)

    def test_given_every_answer_in_a_group_is_right_every_advantage_is_zero(self):
        # No contrast, no signal: a prompt the model always solves teaches it nothing.
        assert group_advantages(np.ones(8)) == pytest.approx(np.zeros(8))

    def test_given_a_verifier_the_correct_sum_scores_one_and_anything_else_zero(self):
        assert (verify((3, 4), 7), verify((3, 4), 6)) == (1.0, 0.0)

    def test_given_grpo_training_accuracy_on_the_addition_prompts_climbs_from_under_30_to_over_90_percent(self):
        accuracy = train_grpo(steps=60, seed=0)["accuracy"]
        assert accuracy[0] < 0.3 and accuracy[-1] > 0.9

    def test_given_grpo_training_more_groups_stop_teaching_anything_as_the_prompts_are_mastered(self):
        no_signal = train_grpo(steps=60, seed=0)["no_signal"]
        assert no_signal[-10:].mean() > no_signal[:10].mean()

    def test_given_the_addition_prompts_every_answer_is_a_single_digit(self):
        assert all(0 <= a + b <= 9 for a, b in ADDITION_PROMPTS)


class TestRewardHacking:
    def test_given_ratings_of_one_to_four_sentence_answers_the_fitted_reward_model_is_0_3125_plus_0_1875_per_sentence(self):
        # Least squares through qualities 0.4375, 0.75, 0.9375, 1.0 at lengths 1..4.
        assert fit_reward_model() == pytest.approx((0.3125, 0.1875))

    def test_given_the_fitted_reward_model_it_scores_ten_sentences_highest_while_true_quality_peaks_at_four(self):
        lengths = np.arange(1, 11)
        assert (lengths[np.argmax(proxy_reward(lengths))], lengths[np.argmax(true_quality(lengths))]) == (10, 4)

    def test_given_optimising_the_flawed_reward_true_quality_first_rises(self):
        run = optimise_lengths(proxy_reward)
        assert run["true"].max() > run["true"][0] + 0.1

    def test_given_optimising_the_flawed_reward_for_longer_true_quality_ends_below_where_it_started(self):
        run = optimise_lengths(proxy_reward)
        assert run["true"][-1] < run["true"][0] and run["proxy"][-1] > run["proxy"][0]

    def test_given_a_kl_leash_true_quality_ends_higher_than_without_one(self):
        leashed, free = optimise_lengths(proxy_reward, beta=0.3), optimise_lengths(proxy_reward)
        assert leashed["true"][-1] > free["true"][-1] + 0.05

    def test_given_a_verifiable_reward_for_the_true_goal_quality_climbs_to_nearly_perfect(self):
        assert optimise_lengths(true_quality)["true"][-1] > 0.98

    def test_given_too_weak_a_leash_the_best_policy_is_worse_than_the_reference_and_a_moderate_one_is_better(self):
        rows = {r["beta"]: r for r in leash_sweep(betas=(0.1, 0.5, 1e6))}
        reference = rows[1e6]["true"]
        assert rows[0.1]["true"] < reference < rows[0.5]["true"]
