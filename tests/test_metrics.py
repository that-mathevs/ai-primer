"""Specification for primer.ml.metrics: how you decide whether a model is any good."""

import numpy as np
import pytest

from primer.ml import metrics


class TestFraudWorkedExample:
    # 100 transactions, 10 fraud, the model flags 8 and 6 of those are truly fraud.

    def test_given_6_correct_of_8_flagged_precision_is_75_percent(self):
        assert metrics.confusion(*metrics.fraud_example()).precision == pytest.approx(0.75)

    def test_given_6_found_of_10_frauds_recall_is_60_percent(self):
        assert metrics.confusion(*metrics.fraud_example()).recall == pytest.approx(0.60)

    def test_given_precision_75_and_recall_60_f1_is_67_percent(self):
        assert metrics.confusion(*metrics.fraud_example()).f1 == pytest.approx(0.667, abs=0.001)

    def test_given_88_true_negatives_accuracy_looks_great_at_94_percent(self):
        assert metrics.confusion(*metrics.fraud_example()).accuracy == pytest.approx(0.94)

    def test_the_confusion_matrix_counts_6_hits_2_false_alarms_4_misses_and_88_correct_rejections(self):
        c = metrics.confusion(*metrics.fraud_example())
        assert (c.tp, c.fp, c.fn, c.tn) == (6, 2, 4, 88)


class TestAccuracyTrap:
    def test_given_1_percent_fraud_a_model_that_flags_nothing_is_99_percent_accurate(self):
        assert metrics.flag_nothing_trap().accuracy == pytest.approx(0.99)

    def test_given_1_percent_fraud_a_model_that_flags_nothing_finds_no_fraud(self):
        assert metrics.flag_nothing_trap().recall == 0.0


class TestRocAuc:
    # Hand example: positives score 0.9 and 0.4, negatives 0.6 and 0.1.
    # Of the 4 (positive, negative) pairs, 3 are ordered correctly (0.4 < 0.6 is the miss).
    Y = [1, 1, 0, 0]
    S = [0.9, 0.4, 0.6, 0.1]

    def test_given_3_of_4_pairs_ranked_correctly_the_area_under_the_roc_curve_is_0_75(self):
        assert metrics.roc_auc(self.Y, self.S) == pytest.approx(0.75)

    def test_given_3_of_4_pairs_ranked_correctly_the_pairwise_win_rate_is_0_75(self):
        assert metrics.roc_auc_rank(self.Y, self.S) == pytest.approx(0.75)

    def test_given_a_perfect_ranking_auc_is_1(self):
        assert metrics.roc_auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(1.0)

    def test_given_a_positive_and_negative_with_the_same_score_the_tie_earns_half_credit(self):
        assert metrics.roc_auc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)

    def test_given_many_tied_scores_the_curve_area_equals_the_pairwise_win_rate(self):
        # Rounding to one decimal creates lots of ties; the diagonal ROC segments must still give half credit.
        y, s = metrics.synthetic_scores()
        s = np.round(s, 1)
        assert metrics.roc_auc(y, s) == pytest.approx(metrics.roc_auc_rank(y, s), abs=1e-12)


class TestThresholdByErrorCost:
    Y = [1, 1, 0, 0]
    S = [0.9, 0.4, 0.6, 0.1]

    def test_given_misses_cost_ten_times_false_alarms_the_threshold_drops_to_catch_every_positive(self):
        # Flagging everything scored >= 0.4 catches both positives at the price of one false alarm (cost 1).
        threshold, cost, _ = metrics.best_threshold_by_cost(self.Y, self.S, cost_fn=10, cost_fp=1)
        assert (threshold, cost) == (0.4, 1.0)

    def test_given_false_alarms_cost_ten_times_misses_the_threshold_rises_to_flag_only_the_surest_case(self):
        # Flagging only 0.9 misses one positive (cost 1) but raises no false alarms.
        threshold, cost, _ = metrics.best_threshold_by_cost(self.Y, self.S, cost_fn=1, cost_fp=10)
        assert (threshold, cost) == (0.9, 1.0)


class TestRetrievalMetrics:
    # One query: results d7, d3, d9, d1, d4; relevant d3 (grade 3), d4 (grade 2), d8 (grade 1, never retrieved).
    RANKED = ["d7", "d3", "d9", "d1", "d4"]
    RELEVANT = {"d3", "d4", "d8"}

    def test_given_2_of_3_relevant_docs_in_the_top_5_recall_at_5_is_two_thirds(self):
        assert metrics.recall_at_k(self.RANKED, self.RELEVANT, 5) == pytest.approx(2 / 3)

    def test_given_2_relevant_docs_among_5_results_precision_at_5_is_0_4(self):
        assert metrics.precision_at_k(self.RANKED, self.RELEVANT, 5) == pytest.approx(0.4)

    def test_given_the_first_relevant_doc_at_rank_2_the_reciprocal_rank_is_one_half(self):
        assert metrics.reciprocal_rank(self.RANKED, self.RELEVANT) == pytest.approx(0.5)

    def test_given_first_hits_at_ranks_1_and_2_and_one_miss_mrr_is_one_half(self):
        runs = [(["a", "b"], {"a"}), (["a", "b"], {"b"}), (["a", "b"], {"z"})]
        assert metrics.mean_reciprocal_rank(runs) == pytest.approx(0.5)

    def test_given_graded_relevance_ndcg_at_5_is_0_56(self):
        # DCG = 3/log2(3) + 2/log2(6) = 2.6665; ideal = 3/1 + 2/log2(3) + 1/2 = 4.7619; ratio 0.560.
        grades = {"d3": 3, "d4": 2, "d8": 1}
        assert metrics.ndcg_at_k(self.RANKED, grades, 5) == pytest.approx(0.560, abs=0.001)

    def test_given_the_ideal_order_ndcg_is_1(self):
        assert metrics.ndcg_at_k(["d3", "d4", "d8"], {"d3": 3, "d4": 2, "d8": 1}, 3) == pytest.approx(1.0)


class TestWordOverlapMetrics:
    def test_given_an_exact_copy_of_the_reference_bleu_is_1(self):
        assert metrics.bleu(metrics.GEN_REFERENCE, metrics.GEN_REFERENCE) == pytest.approx(1.0)

    def test_given_a_repeated_word_bleu_credits_it_only_as_often_as_the_reference_has_it(self):
        # "the" appears once in the reference, so only 1 of the 4 candidate words counts.
        assert metrics.bleu("the the the the", "the cat is here", max_n=1) == pytest.approx(0.25)

    def test_given_candidate_abcd_and_reference_acde_rouge_l_is_0_75(self):
        # Longest common subsequence "a c d" has 3 words: precision 3/4, recall 3/4.
        assert metrics.rouge_l("a b c d", "a c d e") == pytest.approx(0.75)

    def test_a_wrong_answer_that_copies_the_wording_outscores_a_correct_paraphrase_on_bleu(self):
        # The reason teams use calibrated LLM judges instead of overlap metrics for generated text.
        wrong = metrics.bleu(metrics.GEN_CANDIDATES["wrong but copies wording"], metrics.GEN_REFERENCE)
        right = metrics.bleu(metrics.GEN_CANDIDATES["correct paraphrase"], metrics.GEN_REFERENCE)
        assert wrong > 0.7 > 0.2 > right

    def test_a_wrong_answer_that_copies_the_wording_outscores_a_correct_paraphrase_on_rouge_l(self):
        wrong = metrics.rouge_l(metrics.GEN_CANDIDATES["wrong but copies wording"], metrics.GEN_REFERENCE)
        right = metrics.rouge_l(metrics.GEN_CANDIDATES["correct paraphrase"], metrics.GEN_REFERENCE)
        assert wrong > 0.9 > 0.3 > right


class TestJudgeCalibration:
    def test_given_a_judge_that_always_says_pass_on_90_percent_pass_data_kappa_is_0(self):
        # Raw agreement is 90%, but that is exactly what chance predicts.
        human = ["pass"] * 18 + ["fail"] * 2
        assert metrics.cohens_kappa(human, ["pass"] * 20) == pytest.approx(0.0)

    def test_given_75_percent_agreement_where_chance_predicts_50_kappa_is_0_5(self):
        # p_o = 3/4; p_e = 0.5·0.25 + 0.5·0.75 = 0.5; kappa = (0.75 - 0.5) / 0.5.
        assert metrics.cohens_kappa(["y", "y", "n", "n"], ["y", "n", "n", "n"]) == pytest.approx(0.5)

    def test_given_perfect_agreement_kappa_is_1(self):
        assert metrics.cohens_kappa(["y", "n", "y"], ["y", "n", "y"]) == pytest.approx(1.0)
