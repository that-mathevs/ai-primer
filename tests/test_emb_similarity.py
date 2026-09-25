"""Specification for primer.ml.embeddings.similarity: how "close" two embeddings are, and what a score means."""

import numpy as np
import pytest

from primer.ml.embeddings.similarity import (
    anisotropic_embeddings,
    best_threshold,
    cosine,
    curse_of_dimensionality,
    dot,
    euclidean,
    l2_normalize,
    mean_center,
    pair_cosines,
    popularity_in_the_norm,
    rank,
)

A, B, C = np.array([1.0, 2.0, 2.0]), np.array([2.0, 1.0, 2.0]), np.array([2.0, -1.0, -2.0])


class TestWorkedExample:
    def test_given_a_and_b_their_dot_product_is_eight(self):
        assert dot(A, B) == 8.0

    def test_given_a_and_b_their_cosine_is_eight_ninths(self):
        # Both vectors have length 3, so cosine = 8 / (3·3).
        assert cosine(A, B) == pytest.approx(0.89, abs=0.005)

    def test_given_a_and_c_their_cosine_is_minus_four_ninths(self):
        assert cosine(A, C) == pytest.approx(-0.44, abs=0.005)

    def test_given_a_and_c_they_are_farther_apart_than_a_and_b(self):
        # By hand: |a-b| = sqrt(1+1+0) = 1.414; |a-c| = sqrt(1+9+16) = 5.099.
        assert euclidean(A, B) == pytest.approx(1.414, abs=1e-3)
        assert euclidean(A, C) == pytest.approx(5.099, abs=1e-3)


class TestNormalization:
    def test_given_any_nonzero_vector_normalizing_gives_length_one(self):
        assert np.linalg.norm(l2_normalize(np.array([3.0, 4.0]))) == pytest.approx(1.0)

    def test_given_unit_vectors_squared_distance_equals_two_minus_twice_the_cosine(self):
        # a=(1,0), b=(0.6,0.8): |a-b|² = 0.16 + 0.64 = 0.8 and 2 - 2·0.6 = 0.8.
        a, b = np.array([1.0, 0.0]), np.array([0.6, 0.8])
        assert euclidean(a, b) ** 2 == pytest.approx(0.8)
        assert 2 - 2 * cosine(a, b) == pytest.approx(0.8)

    def test_given_unit_vectors_cosine_dot_and_euclidean_rank_neighbours_identically(self):
        rng = np.random.default_rng(0)
        docs, q = l2_normalize(rng.standard_normal((200, 32))), l2_normalize(rng.standard_normal(32))
        by_cosine = rank(q, docs, "cosine")
        assert np.array_equal(rank(q, docs, "dot"), by_cosine)
        assert np.array_equal(rank(q, docs, "l2"), by_cosine)

    def test_given_vectors_of_very_different_lengths_dot_product_ranks_differently_from_cosine(self):
        rng = np.random.default_rng(0)
        docs = rng.standard_normal((200, 32)) * rng.uniform(0.2, 5.0, (200, 1))
        q = rng.standard_normal(32)
        assert not np.array_equal(rank(q, docs, "dot"), rank(q, docs, "cosine"))


class TestSignalInTheVectorLength:
    def test_given_a_long_popular_vector_dot_product_prefers_it_while_cosine_prefers_the_better_aligned_one(self):
        (_, niche_cos, niche_dot), (_, popular_cos, popular_dot) = popularity_in_the_norm()
        assert niche_cos > popular_cos
        assert popular_dot > niche_dot


class TestCurseOfDimensionality:
    def test_given_two_dimensions_the_nearest_random_point_is_far_nearer_than_the_farthest(self):
        (row,) = curse_of_dimensionality(dims=(2,))
        assert row["ratio"] < 0.1

    def test_given_a_thousand_dimensions_the_nearest_and_farthest_random_points_are_almost_equally_far(self):
        (row,) = curse_of_dimensionality(dims=(1000,))
        assert row["ratio"] > 0.8


class TestAnisotropy:
    def test_given_a_model_with_a_shared_direction_unrelated_texts_still_score_about_three_quarters(self):
        # Built as sqrt(3)·m + n with orthogonal n parts, so cos = 3 / (3 + 1) = 0.75.
        left, right, related = anisotropic_embeddings()
        assert pair_cosines(left, right)[~related].mean() == pytest.approx(0.75, abs=0.01)

    def test_given_a_model_with_a_shared_direction_every_score_is_squeezed_between_0_7_and_0_9(self):
        left, right, _ = anisotropic_embeddings()
        scores = pair_cosines(left, right)
        assert scores.min() > 0.7 and scores.max() < 0.9

    def test_given_mean_centering_unrelated_texts_score_near_zero(self):
        left, right, related = anisotropic_embeddings()
        lc, rc = mean_center(left, right)
        assert abs(pair_cosines(lc, rc)[~related].mean()) < 0.05


class TestThresholdCalibration:
    def test_given_labeled_pairs_the_chosen_threshold_is_the_one_that_separates_them(self):
        # By hand: 0.9 and 0.8 are matches, 0.7 and 0.6 are not, so ">= 0.8" is perfect.
        best = best_threshold(np.array([0.9, 0.8, 0.7, 0.6]), np.array([True, True, False, False]))
        assert best["threshold"] == 0.8
        assert best["f1"] == 1.0

    def test_given_an_anisotropic_model_a_calibrated_threshold_beats_a_guessed_0_8(self):
        left, right, related = anisotropic_embeddings()
        scores = pair_cosines(left, right)
        guessed = scores >= 0.8
        guessed_f1 = 2 * np.sum(guessed & related) / (guessed.sum() + related.sum())
        assert best_threshold(scores, related)["f1"] > guessed_f1 + 0.05
