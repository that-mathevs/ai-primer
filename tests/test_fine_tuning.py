"""Specification for primer.ml.fine_tuning: preparing data, forgetting old skills, merging models."""

import numpy as np
import pytest

from primer.ml.fine_tuning import (
    TinyNet,
    base_model,
    break_even_requests,
    chat_example,
    cosine,
    deduplicate,
    fine_tune,
    fine_tuned,
    forgetting,
    general_skill_run,
    jaccard,
    label_agreement,
    make_task,
    margin_of_error,
    measured_accuracy,
    merge,
    merge_run,
    mixed_loss,
    overfitting_run,
    per_request_cost,
    remove_leaks,
    render_chat,
    replay_mix,
    sequential_run,
    split_before_training,
    task_vector,
    validate_chat,
)
from primer.ml.regularization import early_stopping


class TestTheCostOfFineTuning:
    def test_given_3000_prompt_tokens_at_2_dollars_per_million_a_request_costs_six_tenths_of_a_cent(self):
        assert per_request_cost(3000, 2.0) == pytest.approx(0.006)

    def test_given_600_dollars_up_front_and_0_48_cents_saved_per_request_it_pays_off_after_125000_requests(self):
        # 3,000 tokens at $2/M = $0.006 prompted; 300 tokens at $4/M = $0.0012 tuned; 600 / 0.0048 = 125,000.
        assert break_even_requests(600, per_request_cost(3000, 2.0), per_request_cost(300, 4.0)) == pytest.approx(125_000)

    def test_given_a_tuned_model_that_costs_more_per_request_it_never_pays_off(self):
        assert break_even_requests(600, 0.001, 0.002) == float("inf")


class TestFormattingChatExamples:
    def test_given_a_system_user_and_assistant_turn_only_the_assistant_text_is_trained_on(self):
        example = chat_example("Be brief.", "Capital of France?", "Paris.")
        trained = [text for text, is_trained in render_chat(example) if is_trained]
        assert trained == ["<|assistant|>Paris.<|end|>"]

    def test_given_a_well_formed_example_it_has_no_problems(self):
        assert validate_chat(chat_example("Be brief.", "Capital of France?", "Paris.")) == []

    def test_given_an_example_that_ends_on_the_users_turn_it_is_rejected(self):
        # Nothing to learn: the model is trained to produce the final assistant turn.
        example = {"messages": [{"role": "user", "content": "Hi"}]}
        assert validate_chat(example) == ["the last turn must be the assistant's"]

    def test_given_an_empty_assistant_reply_it_is_rejected(self):
        assert validate_chat(chat_example("Be brief.", "Capital of France?", "  ")) == ["turn 3 (assistant) is empty"]

    def test_given_an_unknown_role_it_is_rejected(self):
        example = {"messages": [{"role": "robot", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}
        assert validate_chat(example) == ["turn 1 has unknown role 'robot'"]


class TestDeduplication:
    def test_given_two_phrasings_one_word_apart_they_share_five_of_six_word_pairs(self):
        assert jaccard("How do I reset my password?", "how do I reset my password, please") == pytest.approx(5 / 6)

    def test_given_two_different_questions_with_the_same_opening_they_share_a_quarter_of_their_word_pairs(self):
        assert jaccard("How do I reset my password?", "How do I change my email?") == pytest.approx(0.25)

    def test_given_texts_that_differ_only_in_case_and_punctuation_they_count_as_identical(self):
        assert jaccard("Reset my PASSWORD!", "reset my password") == 1.0

    def test_given_a_near_duplicate_deduplication_keeps_only_the_first(self):
        texts = ["How do I reset my password?", "How do I change my email?", "how do I reset my password, please"]
        assert deduplicate(texts, threshold=0.7) == [0, 1]

    def test_given_a_training_example_that_nearly_copies_an_eval_example_it_is_removed_from_training(self):
        # A near-copy in training lets the model recite the eval answer, inflating the score.
        train = ["how do I reset my password, please", "Where is my order?"]
        assert remove_leaks(train, ["How do I reset my password?"], threshold=0.7) == [1]


class TestTheHeldOutSet:
    def test_given_100_eval_examples_at_80_percent_the_margin_of_error_is_about_8_points(self):
        # 1.96 * sqrt(0.8 * 0.2 / 100) = 1.96 * 0.04
        assert margin_of_error(0.8, 100) == pytest.approx(0.0784)

    def test_given_four_times_as_many_eval_examples_the_margin_of_error_halves(self):
        assert margin_of_error(0.8, 400) == pytest.approx(margin_of_error(0.8, 100) / 2)

    def test_given_a_split_the_eval_set_never_overlaps_training(self):
        texts = [f"question number {i} about topic {i % 7}" for i in range(50)]
        train, held_out = split_before_training(texts, eval_fraction=0.2, seed=0)
        assert (len(held_out), set(train) & set(held_out)) == (10, set())

    def test_given_the_same_seed_the_split_is_the_same_every_time(self):
        texts = [f"question number {i}" for i in range(50)]
        assert split_before_training(texts, 0.2, seed=3) == split_before_training(texts, 0.2, seed=3)


class TestLabelQuality:
    def test_given_a_90_percent_model_and_10_percent_wrong_labels_the_eval_reports_82_percent(self):
        # Right on a right label (0.9 * 0.9) plus wrong on a wrong label (0.1 * 0.1).
        assert measured_accuracy(0.9, 0.1) == pytest.approx(0.82)

    def test_given_a_perfect_model_the_eval_scores_no_higher_than_the_share_of_correct_labels(self):
        assert measured_accuracy(1.0, 0.1) == pytest.approx(0.9)

    def test_given_two_labelers_who_disagree_on_one_item_in_five_they_agree_80_percent(self):
        assert label_agreement(["yes", "no", "yes", "yes", "no"], ["yes", "no", "no", "yes", "no"]) == 0.8


class TestTheToyModel:
    def test_given_the_base_model_it_has_learned_its_general_skill(self):
        assert base_model().accuracy(*make_task("general", seed=100)) > 0.95

    def test_given_a_fine_tune_on_task_a_it_learns_task_a(self):
        assert fine_tuned("A").accuracy(*make_task("A", seed=100)) > 0.95

    def test_given_parameters_as_one_flat_vector_the_model_rebuilt_from_them_predicts_the_same(self):
        net = base_model()
        X, _ = make_task("A", seed=100)
        assert np.allclose(TinyNet(net.theta.copy()).predict(X), net.predict(X))


class TestForgettingTheGeneralSkill:
    def test_when_fine_tuning_runs_long_after_task_a_is_learned_the_general_skill_keeps_fading(self):
        run = general_skill_run(lr=0.5, steps=300)
        assert run["task"][9] > 0.95 and run["task"][-1] > 0.95
        assert run["general"][-1] < run["general"][9] - 0.1

    def test_given_a_lower_learning_rate_the_general_skill_fades_less_over_the_same_steps(self):
        assert general_skill_run(lr=0.02, steps=300)["general"][-1] > general_skill_run(lr=0.5, steps=300)["general"][-1] + 0.1

    def test_given_forgetting_it_grows_with_how_far_the_weights_move(self):
        run = general_skill_run(lr=0.5, steps=300)
        # Later steps are farther from the base and keep less of the general skill.
        assert np.corrcoef(run["distance"][10:], run["general"][10:])[0, 1] < -0.9


class TestCatastrophicForgetting:
    def test_given_forgetting_as_a_number_it_is_the_drop_in_accuracy_on_the_old_task(self):
        assert forgetting(0.98, 0.025) == pytest.approx(0.955)

    def test_given_a_model_fine_tuned_on_a_then_on_b_its_accuracy_on_a_collapses(self):
        run = sequential_run("B")
        assert run["A_before"] > 0.95 and run["A_after"] < 0.1

    def test_given_a_new_task_on_separate_inputs_the_old_task_is_mostly_kept(self):
        assert sequential_run("C")["A_after"] > 0.9

    def test_given_a_lower_learning_rate_task_a_is_still_lost_once_task_b_is_learned(self):
        # Slowing down only slides along the same trade-off: nothing in B's data says A still matters.
        run = sequential_run("B", lr=0.02)
        assert run["B_after"] > 0.95 and run["A_after"] < 0.5


class TestReplay:
    def test_given_200_new_and_10_old_examples_the_mix_has_210_examples(self):
        X, y = replay_mix(make_task("B"), make_task("A"), n_old=10)
        assert (X.shape, y.shape) == ((210, 4), (210,))

    def test_given_old_examples_that_are_being_forgotten_they_carry_most_of_the_loss(self):
        # 200 examples at loss 0.05 and 10 at loss 3.0: the 10 contribute 0.143 of the 0.19 total.
        assert mixed_loss(0.05, 200, 3.0, 10) == pytest.approx(0.19, abs=0.001)

    def test_given_5_percent_of_task_a_mixed_into_task_b_both_tasks_are_kept(self):
        run = sequential_run("B", n_replay=10)
        assert run["A_after"] > 0.9 and run["B_after"] > 0.95


class TestOverfittingASmallDataset:
    def test_given_16_examples_and_many_epochs_training_loss_keeps_falling(self):
        run = overfitting_run()
        assert run["train"][-1] < run["train"][run["best_epoch"]] / 10

    def test_given_16_examples_validation_loss_bottoms_out_early_then_rises(self):
        run = overfitting_run()
        assert run["best_epoch"] < 100 and run["val"][-1] > 2 * run["val"][run["best_epoch"]]

    def test_given_the_validation_curve_early_stopping_stops_soon_after_the_best_epoch(self):
        run = overfitting_run()
        best, stop = early_stopping(run["val"], patience=20)
        assert best == run["best_epoch"] and stop == best + 20


class TestTaskArithmetic:
    def test_given_a_fine_tune_its_task_vector_is_the_change_from_the_base(self):
        base, tuned = TinyNet(np.array([1.0, 0.0, 2.0])), TinyNet(np.array([1.5, 0.0, 2.0]))
        assert task_vector(tuned, base).tolist() == [0.5, 0.0, 0.0]

    def test_given_two_task_vectors_at_lambda_one_the_merge_adds_both_changes(self):
        base = TinyNet(np.array([1.0, 0.0, 2.0]))
        merged = merge(base, [np.array([0.5, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])], lam=1.0)
        assert merged.theta.tolist() == [1.5, -1.0, 2.0]

    def test_given_two_fine_tunes_their_average_is_task_arithmetic_at_lambda_one_half(self):
        base = TinyNet(np.array([1.0, 0.0, 2.0]))
        a, c = TinyNet(np.array([1.5, 0.0, 2.0])), TinyNet(np.array([1.0, -1.0, 2.0]))
        averaged = (a.theta + c.theta) / 2
        assert merge(base, [task_vector(a, base), task_vector(c, base)], lam=0.5).theta.tolist() == averaged.tolist()

    def test_given_task_vectors_for_separate_inputs_they_are_nearly_orthogonal(self):
        assert abs(merge_run("A", "C")["cosine"]) < 0.2

    def test_given_task_vectors_for_separate_inputs_their_sum_does_both_tasks(self):
        at_one = merge_run("A", "C")["by_lambda"][1.0]
        assert min(at_one) > 0.9

    def test_given_plain_averaging_each_skill_is_diluted_compared_with_the_full_sum(self):
        run = merge_run("A", "C")
        assert min(run["by_lambda"][0.5]) < min(run["by_lambda"][1.0]) - 0.1


class TestInterference:
    def test_given_the_worked_vectors_their_cosine_is_minus_eight_tenths(self):
        assert cosine(np.array([0.5, 0.0, 0.0]), np.array([-0.4, 0.0, 0.3])) == pytest.approx(-0.8)

    def test_given_tasks_that_rewrite_the_same_weights_their_task_vectors_point_apart(self):
        assert merge_run("A", "B")["cosine"] < -0.1

    def test_given_conflicting_task_vectors_no_scaling_makes_their_sum_do_both_tasks(self):
        assert max(min(accs) for accs in merge_run("A", "B")["by_lambda"].values()) < 0.7


class TestTraining:
    def test_given_a_step_of_gradient_descent_the_loss_goes_down(self):
        X, y = make_task("A")
        net = base_model()
        assert fine_tune(net, X, y, steps=1, lr=0.1).loss(X, y) < net.loss(X, y)

    def test_given_a_fine_tune_the_base_model_is_left_unchanged(self):
        before = base_model().theta.copy()
        fine_tune(base_model(), *make_task("A"), steps=5)
        assert np.array_equal(base_model().theta, before)
