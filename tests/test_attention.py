"""Specification for primer.ml.attention: how a token decides what to listen to."""

import importlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from primer.ml.attention import (
    SENTENCE,
    MultiHeadAttention,
    causal_mask,
    scaled_dot_product_attention,
    softmax,
    sqrt_dk_experiment,
    viz_data,
    worked_example_it,
    worked_example_sentence,
)

ROOT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")
IT, ANIMAL, TIRED, STREET = (SENTENCE.index(w) for w in ("it", "animal", "tired", "street"))


class TestSoftmax:
    def test_given_scores_2_1_and_half_the_weights_are_63_23_and_14_percent(self):
        weights = worked_example_it()
        assert weights["animal"] == pytest.approx(0.63, abs=0.005)
        assert weights["tired"] == pytest.approx(0.23, abs=0.005)
        assert weights["street"] == pytest.approx(0.14, abs=0.005)

    def test_given_any_scores_the_weights_sum_to_one(self):
        assert softmax(np.array([3.0, -1.0, 0.2, 7.5])).sum() == pytest.approx(1.0)

    def test_given_huge_scores_the_weights_stay_finite(self):
        # exp(1000) overflows; subtracting the max first is what keeps real models from producing NaN.
        weights = softmax(np.array([1000.0, 1001.0, 1002.0]))
        assert np.isfinite(weights).all()
        assert weights == pytest.approx([0.0900, 0.2447, 0.6652], abs=1e-4)


class TestCausalMasking:
    def test_given_a_causal_mask_future_tokens_receive_zero_attention(self):
        rng = np.random.default_rng(1)
        Q, K, V = (rng.standard_normal((5, 4)) for _ in range(3))
        _, weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask(5))
        assert np.all(np.triu(weights, k=1) == 0.0)

    def test_given_the_first_token_it_can_only_attend_to_itself(self):
        rng = np.random.default_rng(1)
        Q, K, V = (rng.standard_normal((3, 4)) for _ in range(3))
        _, weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask(3))
        assert weights[0].tolist() == [1.0, 0.0, 0.0]

    def test_when_a_later_token_changes_earlier_outputs_are_untouched(self):
        # This is what makes left-to-right generation and the KV cache valid.
        X = np.random.default_rng(2).standard_normal((5, 16))
        layer = MultiHeadAttention(16, n_heads=4)
        before, _ = layer(X)
        X_changed = X.copy()
        X_changed[4] += 10
        after, _ = layer(X_changed)
        assert np.allclose(before[:4], after[:4])
        assert not np.allclose(before[4], after[4])


class TestScalingBySqrtDk:
    def test_given_unscaled_512_dim_vectors_score_spread_grows_to_about_sqrt_512(self):
        row = next(r for r in sqrt_dk_experiment(d_ks=(512,), trials=500) if not r["scaled"])
        assert row["score_std"] == pytest.approx(22.6, rel=0.1)

    def test_given_scaling_score_spread_stays_near_one_at_any_width(self):
        rows = [r for r in sqrt_dk_experiment(d_ks=(4, 64, 512), trials=500) if r["scaled"]]
        assert [round(r["score_std"], 1) for r in rows] == [1.0, 1.0, 1.0]

    def test_given_unscaled_512_dim_vectors_softmax_saturates_and_gradients_vanish(self):
        rows = {r["scaled"]: r for r in sqrt_dk_experiment(d_ks=(512,), trials=500)}
        assert rows[False]["max_weight"] > 0.9
        assert rows[False]["grad_norm"] < rows[True]["grad_norm"] / 3


class TestGroupedQueryAttention:
    def test_given_two_kv_heads_instead_of_eight_key_value_parameters_shrink_fourfold(self):
        # d_model=64: MHA K+V = 2·64·64 = 8192; GQA with 2 KV heads of width 8 = 2·64·16 = 2048.
        assert MultiHeadAttention(64, 8).kv_params() == 8192
        assert MultiHeadAttention(64, 8, n_kv_heads=2).kv_params() == 2048

    def test_given_fewer_kv_heads_the_output_shape_is_unchanged(self):
        X = np.random.default_rng(3).standard_normal((6, 64))
        assert MultiHeadAttention(64, 8, n_kv_heads=1)(X)[0].shape == (6, 64)


class TestAgreementWithPyTorch:
    def test_given_the_same_inputs_the_output_matches_torch_scaled_dot_product_attention(self):
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(4)
        Q, K, V = (rng.standard_normal((2, 7, 8)) for _ in range(3))
        ours, _ = scaled_dot_product_attention(Q, K, V, mask=causal_mask(7))
        theirs = torch.nn.functional.scaled_dot_product_attention(
            *(torch.tensor(a) for a in (Q, K, V)), is_causal=True
        ).numpy()
        assert np.allclose(ours, theirs, atol=1e-6)


class TestTheWorkedSentence:
    def test_given_the_whole_sentence_it_still_scores_animal_tired_and_street_2_1_and_half(self):
        # The sentence reuses the worked example's query for "it" and keys for these three words.
        Q, K = worked_example_sentence()
        scores = Q[IT] @ K[[ANIMAL, TIRED, STREET]].T / np.sqrt(K.shape[1])
        assert scores.tolist() == [2.0, 1.0, 0.5]

    def test_given_no_mask_it_attends_most_to_animal(self):
        Q, K = worked_example_sentence()
        _, weights = scaled_dot_product_attention(Q, K, K)
        assert SENTENCE[int(weights[IT].argmax())] == "animal"

    def test_given_a_causal_mask_it_cannot_see_tired_because_tired_comes_later(self):
        Q, K = worked_example_sentence()
        _, weights = scaled_dot_product_attention(Q, K, K, mask=causal_mask(len(SENTENCE)))
        assert weights[IT, TIRED] == 0.0


def _widget_weights(Q, K, causal: bool, temperature: float = 1.0) -> np.ndarray:
    """Run the widget's own JavaScript on these queries and keys and return its weights."""
    options = json.dumps({"causal": causal, "temperature": temperature})
    script = (
        "const m = require('./docs/assets/viz/attention-matrix.js');"
        f"console.log(JSON.stringify(m.attentionWeights({json.dumps(Q)}, {json.dumps(K)}, {options})));"
    )
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, check=True, cwd=ROOT).stdout
    return np.array(json.loads(out))


@pytest.mark.skipif(NODE is None, reason="needs Node to run the widget's arithmetic")
class TestTheAttentionWidgetAgreesWithTheLesson:
    def test_given_the_sentence_without_a_mask_the_widget_computes_the_lessons_weights(self):
        data = viz_data()["attention-matrix"]
        Q, K = np.array(data["Q"]), np.array(data["K"])
        _, expected = scaled_dot_product_attention(Q, K, K)
        assert np.allclose(_widget_weights(data["Q"], data["K"], causal=False), expected, rtol=0, atol=1e-9)

    def test_given_the_sentence_with_a_causal_mask_the_widget_computes_the_lessons_weights(self):
        data = viz_data()["attention-matrix"]
        Q, K = np.array(data["Q"]), np.array(data["K"])
        _, expected = scaled_dot_product_attention(Q, K, K, mask=causal_mask(len(data["tokens"])))
        assert np.allclose(_widget_weights(data["Q"], data["K"], causal=True), expected, rtol=0, atol=1e-9)

    def test_given_temperature_one_half_the_widget_undoes_the_division_by_sqrt_dk(self):
        # The head width is 4, so √d_k = 2, and dividing by 2 · 0.5 is dividing by 1: no scaling at all.
        data = viz_data()["attention-matrix"]
        Q, K = np.array(data["Q"]), np.array(data["K"])
        _, unscaled = scaled_dot_product_attention(Q, K, K, scale=False)
        assert np.allclose(_widget_weights(data["Q"], data["K"], causal=False, temperature=0.5), unscaled, rtol=0, atol=1e-9)


class TestTheLessonPlacesTheAttentionWidget:
    def test_given_the_attention_lesson_it_places_the_widget_right_after_a_try_it_paragraph(self):
        doc = importlib.import_module("primer.ml.attention").__doc__
        placeholder = doc.index('<div class="viz" data-viz="attention-matrix" aria-label="')
        paragraph_before = doc[doc.rindex("\n\n", 0, placeholder - 2) : placeholder]
        assert paragraph_before.strip().startswith("**Try it:**")

    def test_given_the_widget_data_it_is_the_lessons_own_sentence_queries_and_keys(self):
        Q, K = worked_example_sentence()
        data = viz_data()["attention-matrix"]
        assert (data["tokens"], data["Q"], data["K"]) == (SENTENCE, Q.tolist(), K.tolist())
