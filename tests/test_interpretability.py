"""Specification for primer.ml.interpretability: looking inside the model."""

import numpy as np
import pytest

from primer.ml.interpretability import (
    LENS_VOCAB,
    WORKED_DIRECTIONS,
    LandmarkModel,
    PlantedModel,
    SparseAutoencoder,
    compose_features,
    features_a_neuron_responds_to,
    flip_effect,
    fraction_restored,
    lens_map,
    logit_lens,
    match_features,
    finite_difference_gradient,
    patching_map,
    pentagon,
    probe_probability,
    probe_report,
    read_features,
    sae_objective,
    sample_sparse_features,
    superposition_loss_and_grads,
    superposition_readout,
    tinygpt_logit_lens,
    train_sae,
    train_superposition,
    worked_lens,
)
from primer.ml.transformer import TinyGPT


# Trained once per file: every scenario below reads the same trained toys, and the file stays under 5 seconds.
@pytest.fixture(scope="module")
def sparse_toy():
    return train_superposition(p_active=0.05)


@pytest.fixture(scope="module")
def dense_toy():
    return train_superposition(p_active=1.0)


@pytest.fixture(scope="module")
def superposed_activations():
    rng = np.random.default_rng(1)
    return sample_sparse_features(20_000, 5, 0.05, rng) @ pentagon().T


@pytest.fixture(scope="module")
def trained_sae(superposed_activations):
    return train_sae(superposed_activations, n_latents=5, lam=0.3)


@pytest.fixture(scope="module")
def probes():
    return probe_report()


class TestFeaturesAreDirections:
    def test_given_two_perpendicular_feature_directions_a_dot_product_reads_each_amount_back(self):
        h = compose_features({"positive": 0.9, "past tense": 0.4}, WORKED_DIRECTIONS)
        assert read_features(h, WORKED_DIRECTIONS) == pytest.approx({"positive": 0.9, "past tense": 0.4})

    def test_given_two_features_mixed_into_a_hidden_state_no_single_neuron_equals_either_amount(self):
        # (0.9)(0.6, 0.8) + (0.4)(0.8, -0.6) = (0.86, 0.48): each neuron is a blend of both features.
        h = compose_features({"positive": 0.9, "past tense": 0.4}, WORKED_DIRECTIONS)
        assert h == pytest.approx([0.86, 0.48])


class TestLinearProbes:
    def test_given_the_worked_probe_the_hidden_state_2_minus_1_reads_as_82_percent_positive(self):
        # z = 1·2 + 0.5·(−1) + 0 = 1.5, and σ(1.5) = 1 / (1 + e^−1.5) = 0.818.
        assert probe_probability([2.0, -1.0], [1.0, 0.5], 0.0) == pytest.approx(0.818, abs=1e-3)

    def test_given_hidden_states_that_encode_sentiment_a_probe_reads_it_on_examples_it_never_saw(self, probes):
        assert probes["sentiment"]["test"] > 0.95

    def test_given_random_labels_a_probe_scores_near_chance_on_examples_it_never_saw(self, probes):
        # The control task: nothing real to find, so held-out accuracy is a coin flip.
        assert 0.4 < probes["random labels"]["test"] < 0.6

    def test_given_random_labels_a_probe_still_fits_its_own_training_set_better_than_chance(self, probes):
        # This is why probe accuracy only counts on held-out examples: a probe can memorize noise.
        assert probes["random labels"]["train"] > 0.6


class TestDecodableIsNotTheSameAsUsed:
    def test_given_a_feature_the_models_output_ignores_a_probe_still_reads_it(self, probes):
        assert probes["tense"]["test"] > 0.95

    def test_when_the_ignored_feature_is_flipped_the_models_output_does_not_move(self):
        assert flip_effect(PlantedModel(), "tense") == pytest.approx(0.0, abs=1e-9)

    def test_when_the_used_feature_is_flipped_the_models_output_moves_by_two(self):
        # Sentiment is stored as ±1 along its direction and the readout gives it weight 1: a flip moves the output by 2.
        assert flip_effect(PlantedModel(), "sentiment") == pytest.approx(2.0)


class TestTheLogitLens:
    def test_given_the_worked_residual_stream_the_lens_reads_london_then_paris_then_paris(self):
        assert [layer["top"] for layer in worked_lens()] == ["London", "Paris", "Paris"]

    def test_given_the_worked_residual_stream_the_first_guess_is_split_36_40_24(self):
        # logits (0.2, 0.3, −0.2): e^0.2 = 1.22, e^0.3 = 1.35, e^−0.2 = 0.82, total 3.39.
        assert worked_lens()[0]["probs"] == pytest.approx([0.36, 0.40, 0.24], abs=0.005)

    def test_given_the_last_layer_the_lens_gives_paris_a_probability_of_087(self):
        # logits (2, 0, −2): 7.39 / (7.39 + 1 + 0.14) = 0.867.
        assert worked_lens()[-1]["probs"][LENS_VOCAB.index("Paris")] == pytest.approx(0.867, abs=0.001)

    def test_given_an_unembedding_the_lens_turns_any_residual_state_into_probabilities_that_sum_to_one(self):
        probs = logit_lens(np.array([[0.3, -1.2], [2.0, 0.5]]), np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))
        assert probs.sum(axis=-1) == pytest.approx([1.0, 1.0])

    def test_given_tinygpt_the_lens_at_the_last_layer_reproduces_the_models_own_logits(self):
        # The lens is the model's own output head applied early, so at the top it must agree exactly.
        model = TinyGPT(vocab_size=11, d_model=16, n_layers=3, n_heads=2, max_len=8, seed=4)
        ids = np.array([3, 1, 4, 1, 5])
        assert np.allclose(tinygpt_logit_lens(model, ids)[-1], model(ids))

    def test_given_tinygpt_the_lens_gives_one_reading_per_layer_plus_the_embedding(self):
        model = TinyGPT(vocab_size=11, d_model=16, n_layers=3, n_heads=2, max_len=8, seed=4)
        assert tinygpt_logit_lens(model, np.array([3, 1, 4])).shape == (4, 3, 11)

    def test_given_the_landmark_model_the_eiffel_position_reads_paris_one_layer_before_the_last_position_does(self):
        # Rows: after embedding, after the fact-lookup MLP, after the attention mover. Columns: Eiffel, is, in.
        p_paris = lens_map(LandmarkModel(), ["Eiffel", "is", "in"])
        assert p_paris[1, 0] > 0.9 and p_paris[1, 2] == pytest.approx(1 / 3)
        assert p_paris[2, 2] > 0.9


class TestActivationPatching:
    def test_given_the_clean_prompt_paris_beats_rome_by_three(self):
        run = LandmarkModel().run(["Eiffel", "is", "in"])
        assert run["logit_diff"] == pytest.approx(3.0, abs=1e-3)

    def test_given_the_corrupted_prompt_rome_beats_paris_by_three(self):
        run = LandmarkModel().run(["Colosseum", "is", "in"])
        assert run["logit_diff"] == pytest.approx(-3.0, abs=1e-3)

    def test_given_a_patched_difference_of_zero_between_plus_3_and_minus_3_half_the_answer_is_restored(self):
        # (0 − (−3)) / (3 − (−3)) = 3 / 6.
        assert fraction_restored(patched=0.0, clean=3.0, corrupt=-3.0) == 0.5

    def test_when_the_eiffel_positions_mlp_output_is_patched_in_the_answer_is_fully_restored(self):
        model = LandmarkModel()
        clean = model.run(["Eiffel", "is", "in"])
        patched = model.run(["Colosseum", "is", "in"], patch={("mlp_out", 0): clean["mlp_out"][0]})
        assert fraction_restored(patched["logit_diff"], 3.0, -3.0) == pytest.approx(1.0, abs=1e-3)

    def test_when_a_filler_words_state_is_patched_in_nothing_is_restored(self):
        # "is" is the same word in both prompts, so its state carries no information about the landmark.
        model = LandmarkModel()
        clean = model.run(["Eiffel", "is", "in"])
        patched = model.run(["Colosseum", "is", "in"], patch={("resid_mlp", 1): clean["resid"][1, 1]})
        assert fraction_restored(patched["logit_diff"], 3.0, -3.0) == pytest.approx(0.0, abs=1e-3)

    def test_given_the_whole_patching_map_the_answer_lives_at_the_subject_early_and_at_the_last_word_late(self):
        restored = patching_map(LandmarkModel(), ["Eiffel", "is", "in"], ["Colosseum", "is", "in"])
        expected = [[1, 0, 0], [1, 0, 0], [0, 0, 1]]
        assert np.round(restored, 2).tolist() == expected

    def test_given_the_eiffel_position_after_the_last_layer_the_lens_reads_paris_but_patching_it_restores_nothing(self):
        # Readable is not the same as used: nothing downstream reads that position any more.
        model = LandmarkModel()
        restored = patching_map(model, ["Eiffel", "is", "in"], ["Colosseum", "is", "in"])
        assert lens_map(model, ["Eiffel", "is", "in"])[2, 0] > 0.9
        assert restored[2, 0] == pytest.approx(0.0, abs=1e-3)


class TestSuperposition:
    def test_given_five_features_on_a_pentagon_one_active_feature_leaks_0309_into_each_neighbour(self):
        # Neighbours on a pentagon sit 72° apart and cos 72° = 0.309; the next ones sit 144° apart, cos 144° = −0.809.
        out = superposition_readout(pentagon(), np.zeros(5), np.array([1.0, 0, 0, 0, 0]), relu=False)
        assert out == pytest.approx([1.0, 0.309, -0.809, -0.809, 0.309], abs=1e-3)

    def test_given_a_bias_of_minus_031_the_relu_filters_the_leak_to_zero(self):
        out = superposition_readout(pentagon(), np.full(5, -0.31), np.array([1.0, 0, 0, 0, 0]))
        assert out == pytest.approx([0.69, 0, 0, 0, 0], abs=1e-3)

    def test_given_two_neighbouring_features_active_at_once_each_reads_back_too_high(self):
        # The rare collision sparsity bets against: 1 + 0.309 − 0.31 = 0.999 instead of 0.69.
        out = superposition_readout(pentagon(), np.full(5, -0.31), np.array([1.0, 1.0, 0, 0, 0]))
        assert out[:2] == pytest.approx([0.999, 0.999], abs=1e-3)

    def test_given_sparse_features_training_stores_all_five_in_two_dimensions(self, sparse_toy):
        assert np.all(np.linalg.norm(sparse_toy["W"], axis=0) > 0.9)

    def test_given_sparse_features_the_five_learned_directions_sit_72_degrees_apart(self, sparse_toy):
        angles = np.sort(np.degrees(np.arctan2(sparse_toy["W"][1], sparse_toy["W"][0])) % 360)
        gaps = np.diff(np.append(angles, angles[0] + 360))
        assert gaps == pytest.approx([72] * 5, abs=5)

    def test_given_sparse_features_training_learns_a_negative_bias_to_filter_interference(self, sparse_toy):
        assert np.all(sparse_toy["b"] < -0.1)

    def test_given_dense_features_training_keeps_only_the_two_most_important(self, dense_toy):
        norms = np.linalg.norm(dense_toy["W"], axis=0)
        assert norms[:2] == pytest.approx([1.0, 1.0], abs=0.05)
        assert np.all(norms[2:] < 0.1)

    def test_given_features_in_superposition_a_single_neuron_responds_to_three_of_them(self):
        # Neuron 0 of the pentagon carries cos 0° = 1, cos 72° = 0.309 and cos 288° = 0.309.
        assert features_a_neuron_responds_to(pentagon(), neuron=0, threshold=0.25) == [0, 1, 4]

    def test_given_the_hand_derived_gradients_they_match_finite_differences(self):
        rng = np.random.default_rng(0)
        W, b = rng.normal(0, 0.5, (2, 5)), rng.normal(0, 0.1, 5)
        X = sample_sparse_features(64, 5, 0.3, rng)
        _, grads = superposition_loss_and_grads(W, b, X)
        numeric = finite_difference_gradient(lambda W_: superposition_loss_and_grads(W_, b, X)[0], W)
        assert np.allclose(grads["W"], numeric, atol=1e-6)


class TestSparseAutoencoders:
    def test_given_lambda_01_the_one_latent_explanation_costs_less_than_the_two_latent_one(self):
        # Both rebuild h = (0.8, 0.6) exactly, so only the penalty differs: 0.1 × 1.0 against 0.1 × (0.8 + 0.6).
        decoder = np.array([[1.0, 0.0, 0.8], [0.0, 1.0, 0.6]])
        h = np.array([0.8, 0.6])
        assert sae_objective(h, np.array([0.0, 0.0, 1.0]), decoder, lam=0.1) == pytest.approx(0.10)
        assert sae_objective(h, np.array([0.8, 0.6, 0.0]), decoder, lam=0.1) == pytest.approx(0.14)

    def test_given_the_l1_penalty_the_best_activation_shrinks_below_the_true_amount(self):
        # Minimizing (1 − a)² + 0.1·a gives a = 0.95, not 1: the known "shrinkage" side effect of L1.
        decoder = np.array([[0.8], [0.6]])
        h = np.array([0.8, 0.6])
        assert sae_objective(h, np.array([0.95]), decoder, 0.1) < sae_objective(h, np.array([1.0]), decoder, 0.1)

    def test_given_superposed_activations_every_planted_feature_is_matched_by_a_learned_latent(self, trained_sae):
        assert np.all(match_features(pentagon(), trained_sae.W_d) > 0.99)

    def test_given_the_trained_autoencoder_an_active_example_lights_up_about_one_latent(self, trained_sae, superposed_activations):
        codes = trained_sae.encode(superposed_activations)
        active = np.abs(superposed_activations).sum(axis=1) > 0
        assert (codes[active] > 0).sum(axis=1).mean() < 1.3

    def test_given_a_tiny_penalty_the_autoencoder_rebuilds_the_data_but_misses_some_features(self, superposed_activations):
        # Any two directions rebuild a 2-D space perfectly; only the sparsity pressure finds the five real ones.
        weak = train_sae(superposed_activations, n_latents=5, lam=0.01)
        assert weak.reconstruction_error(superposed_activations) < 0.01
        assert match_features(pentagon(), weak.W_d).min() < 0.95

    def test_given_the_hand_derived_gradients_they_match_finite_differences(self):
        rng = np.random.default_rng(2)
        sae = SparseAutoencoder(d=2, n_latents=4, seed=3)
        H = rng.normal(0, 1, (32, 2))
        _, grads = sae.loss_and_grads(H, lam=0.2)

        def loss_with(W_e):
            sae_copy = SparseAutoencoder(d=2, n_latents=4, seed=3)
            sae_copy.W_e = W_e
            return sae_copy.loss_and_grads(H, lam=0.2)[0]

        assert np.allclose(grads["W_e"], finite_difference_gradient(loss_with, sae.W_e.copy()), atol=1e-6)
