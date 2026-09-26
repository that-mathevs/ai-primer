"""Specification for primer.ml.embeddings.similarity: how "close" two embeddings are, and what a score means."""

import json
import re
import shutil
import subprocess
from pathlib import Path

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
    viz_data,
)

ROOT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")

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


def _node(script: str) -> str:
    return subprocess.run([NODE, "-e", script], capture_output=True, text=True, check=True, cwd=ROOT).stdout


def _widget(expression: str):
    """Evaluate one expression against the cosine widget's exported arithmetic, in Node."""
    return json.loads(_node(f"const m = require('./docs/assets/viz/cosine-similarity.js'); console.log(JSON.stringify({expression}));"))


# Pairs of 2-D vectors: the worked-example shapes, a scaled copy, right angles, opposites, a zero vector.
WIDGET_PAIRS = [
    ([1.0, 2.0], [2.0, 1.0]),
    ([1.0, 0.0], [0.6, 0.8]),
    ([0.5, 0.5], [1.5, 1.5]),
    ([2.0, 0.0], [0.0, 2.0]),
    ([1.2, -0.7], [-1.2, 0.7]),
    ([-2.5, 1.1], [0.3, -2.9]),
]
ZERO_PAIR = ([0.0, 0.0], [1.0, 2.0])


@pytest.mark.skipif(NODE is None, reason="needs Node to run the widget's arithmetic")
class TestTheCosineWidgetAgreesWithTheLesson:
    def test_given_pairs_of_arrows_the_widget_computes_the_lessons_dot_products(self):
        pairs = [*WIDGET_PAIRS, ZERO_PAIR]
        got = _widget(f"{json.dumps(pairs)}.map(([a, b]) => m.dot(a, b))")
        assert got == pytest.approx([dot(np.array(a), np.array(b)) for a, b in pairs], abs=1e-12)

    def test_given_pairs_of_arrows_the_widget_computes_the_lessons_cosines(self):
        got = _widget(f"{json.dumps(WIDGET_PAIRS)}.map(([a, b]) => m.cosine(a, b))")
        assert got == pytest.approx([cosine(np.array(a), np.array(b)) for a, b in WIDGET_PAIRS], abs=1e-12)

    def test_given_pairs_of_arrows_the_widget_computes_the_lessons_euclidean_distances(self):
        pairs = [*WIDGET_PAIRS, ZERO_PAIR]
        got = _widget(f"{json.dumps(pairs)}.map(([a, b]) => m.euclidean(a, b))")
        assert got == pytest.approx([euclidean(np.array(a), np.array(b)) for a, b in pairs], abs=1e-12)

    def test_given_an_arrow_normalizing_it_in_the_widget_matches_the_lessons_l2_normalize(self):
        arrows = [a for pair in WIDGET_PAIRS for a in pair]
        got = _widget(f"{json.dumps(arrows)}.map((a) => m.l2Normalize(a))")
        assert np.allclose(got, l2_normalize(np.array(arrows)), rtol=0, atol=1e-12)

    def test_given_a_zero_length_arrow_normalizing_it_leaves_it_at_zero_as_the_lesson_does(self):
        # The lesson's eps guard means an empty vector stays empty instead of becoming NaN.
        assert _widget("m.l2Normalize([0, 0])") == [0.0, 0.0]
        assert l2_normalize(np.array([0.0, 0.0])).tolist() == [0.0, 0.0]

    def test_given_an_angle_and_a_length_the_widget_places_the_arrow_tip_by_hand_trigonometry(self):
        # 90 degrees at length 2 points straight up; 180 degrees at length 0.5 points left.
        got = _widget("[m.fromPolar(90, 2), m.fromPolar(180, 0.5), m.fromPolar(0, 3)]")
        assert np.allclose(got, [[0.0, 2.0], [-0.5, 0.0], [3.0, 0.0]], rtol=0, atol=1e-12)


class TestTheCosineWidgetPresets:
    def _preset(self, name):
        (preset,) = [p for p in viz_data()["cosine-similarity"]["presets"] if p["name"] == name]
        return preset

    @staticmethod
    def _arrow(angle, length):
        return length * np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])

    def test_given_the_same_direction_preset_the_cosine_is_one_although_the_lengths_differ(self):
        p = self._preset("Same direction, different lengths")
        assert p["a"]["length"] != p["b"]["length"]
        assert cosine(self._arrow(**p["a"]), self._arrow(**p["b"])) == pytest.approx(1.0)

    def test_given_the_right_angles_preset_the_cosine_is_zero(self):
        p = self._preset("At right angles")
        assert cosine(self._arrow(**p["a"]), self._arrow(**p["b"])) == pytest.approx(0.0, abs=1e-12)

    def test_given_the_opposite_preset_the_cosine_is_minus_one(self):
        p = self._preset("Opposite")
        assert cosine(self._arrow(**p["a"]), self._arrow(**p["b"])) == pytest.approx(-1.0)

    def test_given_the_visualization_data_it_is_plain_json(self):
        json.dumps(viz_data())  # raises if anything isn't plain JSON


class TestTheLessonPlacesTheCosineWidget:
    def test_given_the_similarity_lesson_it_places_the_cosine_similarity_widget(self):
        import primer.ml.embeddings.similarity as lesson

        assert re.search(r'<div class="viz" data-viz="cosine-similarity" aria-label="[^"]+"></div>', lesson.__doc__)

    def test_given_the_cosine_similarity_widget_a_try_it_paragraph_introduces_it(self):
        import primer.ml.embeddings.similarity as lesson

        before = lesson.__doc__.split('<div class="viz" data-viz="cosine-similarity"')[0]
        assert "**Try it:**" in before.split("\n\n")[-2]
