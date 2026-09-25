"""Specification for primer.ml.big_picture: what happens when you send a prompt."""

import math

import numpy as np
import pytest

from primer.ml.big_picture import (
    build_pipeline,
    cross_entropy,
    generate,
    next_token_loss,
    next_token_probs,
    trace,
)


@pytest.fixture(scope="module")
def pipeline():
    return build_pipeline()


class TestTracingOnePrompt:
    def test_the_prompt_becomes_one_id_per_token(self, pipeline):
        # " password" is a single token in the toy tokenizer, "Reset" is two.
        stages = trace("Reset your password", *pipeline)
        assert len(stages["ids"]) == 5

    def test_each_id_becomes_one_vector_of_model_width(self, pipeline):
        stages = trace("Reset your password", *pipeline)
        assert stages["embeddings"].shape == (5, 32)

    def test_the_blocks_keep_one_vector_per_token(self, pipeline):
        stages = trace("Reset your password", *pipeline)
        assert stages["after_blocks"].shape == (5, 32)

    def test_the_output_layer_scores_every_one_of_the_400_vocabulary_entries(self, pipeline):
        stages = trace("Reset your password", *pipeline)
        assert stages["logits"].shape == (5, 400)

    def test_the_next_token_probabilities_add_up_to_one(self, pipeline):
        stages = trace("Reset your password", *pipeline)
        assert stages["next_token_probs"].sum() == pytest.approx(1.0)


class TestTemperature:
    def test_at_temperature_1_scores_2_1_and_half_become_63_23_and_14_percent(self):
        assert next_token_probs(np.array([2.0, 1.0, 0.5]), temperature=1.0) == pytest.approx([0.63, 0.23, 0.14], abs=0.005)

    def test_at_temperature_half_the_top_token_rises_to_84_percent(self):
        # Dividing by 0.5 doubles the scores: e⁴ / (e⁴ + e² + e¹) = 54.6 / 64.7.
        assert next_token_probs(np.array([2.0, 1.0, 0.5]), temperature=0.5)[0] == pytest.approx(0.844, abs=1e-3)

    def test_at_temperature_zero_the_top_token_gets_all_the_probability(self):
        assert next_token_probs(np.array([2.0, 1.0, 0.5]), temperature=0.0).tolist() == [1.0, 0.0, 0.0]


class TestTheGenerationLoop:
    def test_generating_5_tokens_appends_5_ids_to_the_prompt(self, pipeline):
        result = generate("The cat", 5, *pipeline, seed=0)
        prompt_len = len(pipeline[0].encode("The cat"))
        assert len(result.ids) == prompt_len + 5

    def test_the_same_seed_produces_the_same_continuation(self, pipeline):
        assert generate("The cat", 5, *pipeline, seed=1).ids == generate("The cat", 5, *pipeline, seed=1).ids

    def test_without_a_cache_each_step_reprocesses_the_whole_sequence(self, pipeline):
        # A 2-token prompt, 4 new tokens: passes over 2 + 3 + 4 + 5 = 14 positions.
        result = generate("The cat", 4, *pipeline, seed=0)
        assert result.positions_processed == 14


class TestTrainingIsTheSameForwardPassPlusALoss:
    def test_putting_half_the_probability_on_the_right_token_costs_0_693(self):
        # −ln(0.5) = 0.693
        assert cross_entropy(np.array([0.5, 0.3, 0.2]), target=0) == pytest.approx(0.693, abs=1e-3)

    def test_an_untrained_model_is_about_as_unsure_as_a_uniform_guess_over_400_tokens(self, pipeline):
        # Uniform over 400 tokens: loss = ln 400 = 5.99, i.e. perplexity ≈ 400.
        assert next_token_loss(TRAINING_SENTENCE, *pipeline) == pytest.approx(math.log(400), abs=0.1)


TRAINING_SENTENCE = "The model reads tokens, not words."
