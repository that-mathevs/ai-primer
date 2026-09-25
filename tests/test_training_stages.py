"""Specification for primer.ml.training_stages: how a text predictor becomes an assistant."""

import numpy as np
import pytest

from primer.ml import training_stages as ts


def _logits_giving(prob_on_correct: float, correct: int, vocab: int = 4) -> np.ndarray:
    """Logits whose softmax puts `prob_on_correct` on `correct`, the rest spread evenly."""
    p = np.full(vocab, (1 - prob_on_correct) / (vocab - 1))
    p[correct] = prob_on_correct
    return np.log(p)


class TestSupervisedFineTuningLoss:
    # Sequence of 5 tokens: a 3-token prompt then a 2-token reply. Position i predicts token i+1.
    TOKENS = np.array([0, 1, 2, 3, 1])
    PROMPT_LEN = 3
    # Positions 0 and 1 predict prompt tokens and are uniform over 4 tokens (loss ln 4 = 1.386 each);
    # positions 2 and 3 predict reply tokens with probability 0.5 (loss ln 2 = 0.693 each).
    LOGITS = np.stack([np.zeros(4), np.zeros(4), _logits_giving(0.5, 3), _logits_giving(0.5, 1), np.zeros(4)])

    def test_given_a_3_token_prompt_and_2_token_reply_only_the_2_positions_predicting_the_reply_are_trained(self):
        assert ts.response_mask(5, self.PROMPT_LEN).tolist() == [False, False, True, True]

    def test_given_the_reply_predicted_with_probability_one_half_the_sft_loss_is_ln_2(self):
        # The prompt's poor predictions are ignored: SFT teaches answering, not writing user prompts.
        assert ts.sft_loss(self.LOGITS, self.TOKENS, self.PROMPT_LEN) == pytest.approx(0.693, abs=0.001)

    def test_given_the_same_sequence_the_pretraining_loss_averages_every_position(self):
        # (1.386 + 1.386 + 0.693 + 0.693) / 4 = 1.040
        assert ts.next_token_loss(self.LOGITS, self.TOKENS) == pytest.approx(1.040, abs=0.001)


class TestRewardModel:
    def test_given_equal_rewards_the_preferred_answer_wins_half_the_time_and_the_loss_is_ln_2(self):
        assert ts.reward_model_loss(1.0, 1.0) == pytest.approx(0.693, abs=0.001)

    def test_given_the_chosen_answer_scores_2_higher_it_is_preferred_with_probability_0_88(self):
        # Bradley-Terry: P(chosen beats rejected) = sigmoid(2) = 0.881.
        assert ts.preference_probability(3.0, 1.0) == pytest.approx(0.881, abs=0.001)

    def test_given_the_chosen_answer_scores_2_higher_the_loss_is_0_127(self):
        # -ln(0.881) = 0.127
        assert ts.reward_model_loss(3.0, 1.0) == pytest.approx(0.127, abs=0.001)


class TestDirectPreferenceOptimization:
    def test_given_a_policy_identical_to_the_reference_the_dpo_loss_is_ln_2(self):
        # No preference has been learned yet: the implicit reward margin is 0 and sigmoid(0) = 1/2.
        assert ts.dpo_loss(-11.0, -11.0, -11.0, -11.0, beta=0.1) == pytest.approx(0.693, abs=0.001)

    def test_given_the_worked_example_the_dpo_loss_is_0_598(self):
        # Policy log-probs: chosen -10, rejected -12. Reference: both -11. beta = 0.1.
        # Margin = 0.1 * ((-10 - -11) - (-12 - -11)) = 0.2, loss = -ln sigmoid(0.2) = 0.598.
        assert ts.dpo_loss(-10.0, -12.0, -11.0, -11.0, beta=0.1) == pytest.approx(0.598, abs=0.001)

    def test_given_the_worked_example_the_update_strength_is_0_045(self):
        # Gradient scale beta * (1 - sigmoid(margin)) = 0.1 * 0.450: pairs already ranked right get small updates.
        assert ts.dpo_update_strength(-10.0, -12.0, -11.0, -11.0, beta=0.1) == pytest.approx(0.045, abs=0.001)

    def test_given_a_pair_the_policy_ranks_the_wrong_way_the_update_strength_is_larger(self):
        wrong_way = ts.dpo_update_strength(-12.0, -10.0, -11.0, -11.0, beta=0.1)
        right_way = ts.dpo_update_strength(-10.0, -12.0, -11.0, -11.0, beta=0.1)
        assert wrong_way > right_way

    def test_when_a_toy_policy_trains_on_preferences_the_chosen_answer_becomes_more_likely(self):
        before, after = ts.train_toy_dpo(steps=200)
        assert before[0] == pytest.approx(1 / 3)  # starts uniform over three candidate answers
        assert after[0] > 0.6

    def test_when_a_toy_policy_trains_for_50_steps_the_rambling_answer_falls_more_slowly_than_the_rude_one(self):
        # Rambling loses one pair and wins one, rude loses both: from 1/3 each, rambling is at 0.166, rude at 0.035.
        # Values from an independent pure-Python rerun of the same 50 steps.
        _, after = ts.train_toy_dpo(steps=50)
        assert after.tolist() == pytest.approx([0.799, 0.035, 0.166], abs=0.001)

    def test_given_a_larger_beta_the_same_departure_from_the_reference_already_gives_a_lower_loss(self):
        # beta is the leash: at beta = 0.5 the worked example's log-ratios (+1, -1) give margin 1.0,
        # loss -ln sigmoid(1) = 0.313 versus 0.598 at beta = 0.1, so less drift is needed to satisfy a pair.
        assert ts.dpo_loss(-10.0, -12.0, -11.0, -11.0, beta=0.5) == pytest.approx(0.313, abs=0.001)


class TestLoRA:
    def test_given_the_adapter_starts_with_B_at_zero_the_adapted_layer_behaves_exactly_like_the_frozen_one(self):
        # B = 0 makes B·A = 0, so fine-tuning starts from the pretrained model rather than from noise.
        W = np.random.default_rng(0).standard_normal((6, 4))
        x = np.random.default_rng(1).standard_normal((3, 4))
        assert np.array_equal(ts.LoRALinear(W, rank=2)(x), x @ W.T)

    def test_given_W_identity_B_1_0_and_A_0_1_input_1_2_becomes_3_2(self):
        # Hand calculation: W·x = (1, 2); B·A = [[0, 1], [0, 0]] so B·A·x = (2, 0); sum = (3, 2).
        layer = ts.LoRALinear(np.eye(2), rank=1, alpha=1.0)
        layer.B = np.array([[1.0], [0.0]])
        layer.A = np.array([[0.0, 1.0]])
        assert layer(np.array([[1.0, 2.0]])).tolist() == [[3.0, 2.0]]

    @pytest.mark.parametrize("rank, trainable", [(8, 65_536), (16, 131_072), (64, 524_288)])
    def test_given_a_4096_by_4096_layer_lora_trains_2_times_4096_times_rank_parameters(self, rank, trainable):
        # Versus 4096 × 4096 = 16,777,216 for full fine-tuning of the same layer.
        assert ts.lora_trainable_params(4096, 4096, rank) == trainable

    def test_given_a_4096_by_4096_layer_full_fine_tuning_trains_16_8_million_parameters(self):
        assert ts.full_trainable_params(4096, 4096) == 16_777_216

    def test_when_the_adapter_is_merged_into_W_the_outputs_are_unchanged(self):
        # Merging is why a LoRA fine-tune adds zero latency at inference time.
        rng = np.random.default_rng(2)
        layer = ts.LoRALinear(rng.standard_normal((5, 5)), rank=2)
        layer.B = rng.standard_normal((5, 2))
        x = rng.standard_normal((4, 5))
        assert np.allclose(x @ layer.merged_weight().T, layer(x))

    def test_when_trained_on_a_low_rank_correction_a_rank_2_adapter_removes_over_99_percent_of_the_error(self):
        run = ts.train_toy_lora(rank=2, steps=300)
        assert run["final_loss"] < 0.01 * run["initial_loss"]

    def test_when_the_adapter_trains_the_pretrained_weights_do_not_change(self):
        run = ts.train_toy_lora(rank=2, steps=50)
        assert run["frozen_weights_unchanged"]


class TestChoosingAnAdaptation:
    def test_given_missing_or_changing_facts_the_answer_is_retrieval_not_fine_tuning(self):
        # Fine-tuning teaches behaviour; RAG supplies knowledge that changes or must be cited.
        assert ts.choose_adaptation(needs_knowledge=True, wrong_behavior=False) == "RAG"

    def test_given_wrong_format_that_a_better_prompt_fixes_prompting_is_enough(self):
        assert ts.choose_adaptation(needs_knowledge=False, wrong_behavior=True, prompt_fixes_it=True) == "prompting"

    def test_given_behaviour_a_prompt_cannot_make_consistent_the_answer_is_a_lora_fine_tune(self):
        assert ts.choose_adaptation(needs_knowledge=False, wrong_behavior=True, prompt_fixes_it=False) == "LoRA"

    def test_given_a_major_domain_shift_with_lots_of_data_the_answer_is_a_full_fine_tune(self):
        choice = ts.choose_adaptation(
            needs_knowledge=False, wrong_behavior=True, prompt_fixes_it=False, major_shift_with_lots_of_data=True
        )
        assert choice == "full fine-tune"

    def test_given_neither_missing_knowledge_nor_wrong_behaviour_nothing_needs_changing(self):
        assert ts.choose_adaptation(needs_knowledge=False, wrong_behavior=False) == "no change"


class TestDistillation:
    def test_given_teacher_logits_2_1_0_at_temperature_1_the_targets_are_0_665_0_245_0_090(self):
        assert ts.soft_targets(np.array([2.0, 1.0, 0.0]), 1.0) == pytest.approx([0.665, 0.245, 0.090], abs=0.001)

    def test_given_the_same_logits_at_temperature_2_the_targets_flatten_to_0_506_0_307_0_186(self):
        # Dividing logits by T=2 gives softmax(1, 0.5, 0): the ranking of the wrong answers becomes visible.
        assert ts.soft_targets(np.array([2.0, 1.0, 0.0]), 2.0) == pytest.approx([0.506, 0.307, 0.186], abs=0.001)

    def test_given_teacher_50_50_and_student_90_10_the_kl_divergence_is_0_511(self):
        # 0.5·ln(0.5/0.9) + 0.5·ln(0.5/0.1) = -0.294 + 0.805 = 0.511 nats.
        assert ts.kl_divergence(np.array([0.5, 0.5]), np.array([0.9, 0.1])) == pytest.approx(0.511, abs=0.001)

    def test_given_a_student_that_already_matches_the_teacher_the_distillation_loss_is_zero(self):
        logits = np.array([2.0, 1.0, 0.0])
        assert ts.distillation_loss(logits, logits, temperature=2.0) == pytest.approx(0.0, abs=1e-12)
