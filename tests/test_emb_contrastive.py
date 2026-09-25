"""Specification for primer.ml.embeddings.contrastive: how embedding models are taught what "similar" means."""

import numpy as np
import pytest

from primer.ml.embeddings.contrastive import (
    clip_loss,
    encoder_gradient_check,
    info_nce_loss,
    look_alike_accuracy,
    positive_probability,
    train_bi_encoder,
    train_clip_toy,
)


@pytest.fixture(scope="module")
def in_batch_only():
    return train_bi_encoder(hard_negatives=False)


@pytest.fixture(scope="module")
def with_hard_negatives():
    return train_bi_encoder(hard_negatives=True)


class TestInfoNCELoss:
    def test_given_a_query_matching_its_passage_at_temperature_one_the_loss_is_0_313(self):
        # Scores (1, 0): -ln(e / (e + 1)) = ln(1 + 1/e) = 0.3133.
        loss = info_nce_loss(np.array([[1.0, 0.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]), tau=1.0)
        assert loss == pytest.approx(0.3133, abs=1e-4)

    def test_given_the_same_scores_at_temperature_0_1_the_loss_is_almost_zero(self):
        # Scores become (10, 0): ln(1 + e^-10) = 0.0000454.
        loss = info_nce_loss(np.array([[1.0, 0.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]), tau=0.1)
        assert loss == pytest.approx(4.54e-5, rel=1e-2)

    def test_given_random_inputs_the_hand_derived_encoder_gradient_matches_a_numerical_estimate(self):
        assert encoder_gradient_check() < 1e-6


class TestTemperature:
    def test_given_temperature_one_a_positive_at_0_8_against_three_negatives_at_0_6_gets_only_29_percent(self):
        # e^0.8 / (e^0.8 + 3·e^0.6) = 2.2255 / 7.6919 = 0.289.
        assert positive_probability(0.8, [0.6, 0.6, 0.6], tau=1.0) == pytest.approx(0.289, abs=1e-3)

    def test_given_temperature_0_05_the_same_scores_give_the_positive_95_percent(self):
        # Scores become 16 and 12: 1 / (1 + 3·e^-4) = 0.948.
        assert positive_probability(0.8, [0.6, 0.6, 0.6], tau=0.05) == pytest.approx(0.948, abs=1e-3)


class TestHardNegatives:
    def test_given_only_in_batch_negatives_the_encoder_often_confuses_look_alike_passages(self, in_batch_only):
        # In-batch negatives are nearly always a different topic, so matching the topic word is enough to win.
        assert look_alike_accuracy(in_batch_only) < 0.75

    def test_given_hard_negatives_the_encoder_tells_look_alike_passages_apart(self, with_hard_negatives):
        assert look_alike_accuracy(with_hard_negatives) >= 0.9

    def test_given_an_unseen_password_topic_the_hard_negative_encoder_ranks_reset_steps_above_the_policy(self, with_hard_negatives):
        q, right, wrong = with_hard_negatives.encode(
            ["my password is broken and i am stuck", "Password: restart, reinstall and follow the setup guide.", "Password: approval, compliance and usage limits for all staff."]
        )
        assert q @ right > q @ wrong


class TestCLIP:
    def test_given_two_correctly_paired_images_and_captions_the_symmetric_loss_is_0_313(self):
        # Each row and each column scores (1, 0) against (0, 1): ln(1 + 1/e) = 0.3133 both ways.
        eye = np.eye(2)
        assert clip_loss(eye, eye, tau=1.0) == pytest.approx(0.3133, abs=1e-4)

    def test_given_swapped_captions_the_symmetric_loss_rises_to_1_313(self):
        # Now the right caption scores 0 and the wrong one 1: ln(1 + e) = 1.3133.
        assert clip_loss(np.eye(2), np.eye(2)[::-1], tau=1.0) == pytest.approx(1.3133, abs=1e-4)

    def test_given_untrained_encoders_zero_shot_labels_are_near_chance(self):
        # Five classes, so guessing gets 20%.
        assert train_clip_toy(steps=0)["zero_shot_accuracy"] < 0.45

    def test_given_contrastive_training_images_are_labeled_by_their_nearest_caption_embedding(self):
        assert train_clip_toy()["zero_shot_accuracy"] >= 0.9

    def test_given_contrastive_training_4_of_the_40_pictured_test_images_are_closest_to_another_class_caption(self):
        # The lesson's heatmap is not a perfect staircase: 36 rows peak in their own column, 4 do not.
        result = train_clip_toy()
        misses = int(np.sum(result["similarity"].argmax(axis=1) != result["labels"]))
        assert misses == 4
