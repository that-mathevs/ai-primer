"""Specification for primer.ml.embeddings.word2vec: how word vectors are learned from the company a word keeps."""

import numpy as np
import pytest

from primer.ml.embeddings.word2vec import (
    analogy,
    build_corpus,
    contextual_bank_demo,
    gradient_check,
    noise_distribution,
    nearest,
    ppmi,
    ppmi_svd_embeddings,
    sgns_loss,
    sgns_update,
    skipgram_pairs,
    train_sgns,
)


@pytest.fixture(scope="module")
def sgns():
    return train_sgns(build_corpus())


@pytest.fixture(scope="module")
def count_based():
    return ppmi_svd_embeddings(build_corpus())


class TestSkipGramTrainingPairs:
    def test_given_a_window_of_one_each_word_is_paired_with_its_immediate_neighbours(self):
        assert skipgram_pairs(["river", "bank", "fish"], window=1) == [
            ("river", "bank"),
            ("bank", "river"),
            ("bank", "fish"),
            ("fish", "bank"),
        ]


class TestNegativeSampling:
    def test_given_word_counts_of_1_and_16_noise_words_are_drawn_one_ninth_and_eight_ninths_of_the_time(self):
        # Counts are raised to the 3/4 power: 1 -> 1 and 16 -> 8, so 1/9 and 8/9.
        # The 3/4 power boosts rare words so they get sampled as negatives more than raw frequency would allow.
        assert noise_distribution(np.array([1, 16])) == pytest.approx([1 / 9, 8 / 9])


class TestSkipGramLoss:
    def test_given_orthogonal_vectors_each_true_and_noise_pair_costs_log_two(self):
        # sigma(0) = 0.5, and -log(0.5) = 0.693, so one positive plus two negatives cost 3 x 0.693.
        v = np.array([1.0, 0.0])
        loss = sgns_loss(v, np.array([0.0, 1.0]), np.array([[0.0, 1.0], [0.0, -1.0]]))
        assert loss == pytest.approx(2.079, abs=1e-3)

    def test_given_the_true_context_aligned_and_noise_opposed_the_loss_is_near_zero(self):
        v = np.array([10.0, 0.0])
        loss = sgns_loss(v, np.array([1.0, 0.0]), np.array([[-1.0, 0.0]]))
        assert loss < 1e-3

    def test_given_one_update_at_learning_rate_0_1_the_true_pair_scores_higher_and_the_loss_drops(self):
        # Worked by hand in the lesson: v=(1,0), true context (0,1), noise (0,-1).
        # Loss 2·ln2 = 1.386 -> both scores move 0.15 the right way -> 2·(-ln σ(0.15)) = 1.242.
        v, u, n = sgns_update(np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([[0.0, -1.0]]), lr=0.1)
        assert v @ u == pytest.approx(0.15)
        assert (n @ v)[0] == pytest.approx(-0.15)
        assert sgns_loss(v, u, n) == pytest.approx(1.242, abs=1e-3)

    def test_given_random_vectors_the_hand_derived_gradient_matches_a_numerical_estimate(self):
        assert gradient_check() < 1e-6


class TestLearnedGeometry:
    def test_given_the_trained_vectors_king_is_closest_to_a_word_sharing_two_of_its_three_attributes(self, sgns):
        # man (common), queen (female) and prince (child) each differ from king in exactly one attribute.
        assert nearest(sgns, "king", k=1)[0][0] in {"man", "queen", "prince"}

    def test_given_king_minus_man_plus_woman_the_nearest_word_is_queen(self, sgns):
        assert analogy(sgns, "king", "man", "woman") == "queen"

    def test_given_boy_minus_man_plus_woman_the_nearest_word_is_girl(self, sgns):
        assert analogy(sgns, "boy", "man", "woman") == "girl"

    def test_given_prince_minus_boy_plus_girl_the_nearest_word_is_princess(self, sgns):
        assert analogy(sgns, "prince", "boy", "girl") == "princess"


class TestCountBasedEmbeddings:
    def test_given_two_words_that_never_share_a_context_their_ppmi_is_zero(self):
        # P(w,c) = 0.5 on the diagonal, P(w) = P(c) = 0.5, so PMI = log(0.5 / 0.25) = log 2.
        m = ppmi(np.array([[2.0, 0.0], [0.0, 2.0]]))
        assert m == pytest.approx(np.array([[0.693, 0.0], [0.0, 0.693]]), abs=1e-3)

    def test_given_ppmi_factored_by_svd_king_minus_man_plus_woman_is_also_queen(self, count_based):
        # Counting and factorizing learns the same structure as prediction (Levy and Goldberg, 2014).
        assert analogy(count_based, "king", "man", "woman") == "queen"


class TestOneVectorPerWord:
    def test_given_static_vectors_bank_leans_to_neither_the_river_nor_the_money_sense(self, sgns):
        d = contextual_bank_demo(sgns)
        assert abs(d["static_to_river"] - d["static_to_money"]) < 0.25

    def test_given_attention_over_a_river_sentence_bank_moves_toward_river_words(self, sgns):
        d = contextual_bank_demo(sgns)
        assert d["river_bank_to_river"] > d["static_to_river"]
        assert d["river_bank_to_river"] > d["river_bank_to_money"]

    def test_given_attention_over_a_money_sentence_bank_moves_toward_money_words(self, sgns):
        d = contextual_bank_demo(sgns)
        assert d["money_bank_to_money"] > d["static_to_money"]
        assert d["money_bank_to_money"] > d["money_bank_to_river"]
