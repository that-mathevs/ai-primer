"""Specification for primer.ml.generative.gans: a forger against a detective, and why the contest is unstable."""

import math

import numpy as np
import pytest

from primer.ml.generative.gans import (
    DISCRIMINATOR_SIZES,
    GENERATOR_SIZES,
    MLP,
    dirac_gan,
    fit_discriminator_table,
    generator_logit_gradient,
    head_start_experiment,
    js_divergence,
    largest_stretch,
    mode_coverage,
    optimal_discriminator,
    ring_modes,
    ring_of_gaussians,
    sigmoid,
    spectrally_normalize,
    train_gan,
    value_function,
    wasserstein_1d,
)

# Learning rates for the ring experiments: the forger always uses 1e-3.
EQUAL, BALANCED, DETECTIVE_TOO_FAST = 1e-3, 3e-3, 1e-2


@pytest.fixture(scope="module")
def runs():
    # One training run per detective speed, shared by every ring test (each takes well under a second).
    return {lr_d: train_gan(steps=2500, lr_d=lr_d) for lr_d in (EQUAL, BALANCED, DETECTIVE_TOO_FAST)}


class TestTheTwoPlayers:
    def test_given_a_batch_of_noise_the_forger_returns_one_2d_point_per_noise_vector(self):
        forger = MLP(GENERATOR_SIZES, seed=0)
        assert forger.forward(np.random.default_rng(0).standard_normal((5, 2))).shape == (5, 2)

    def test_given_any_points_the_detective_turns_its_score_into_a_probability_between_0_and_1(self):
        detective = MLP(DISCRIMINATOR_SIZES, seed=0)
        verdicts = sigmoid(detective.forward(np.array([[0.0, 0.0], [2.0, 0.0], [50.0, -50.0]])))
        assert np.all((verdicts > 0) & (verdicts < 1))

    def test_given_a_hand_written_backward_pass_it_matches_gradients_measured_by_nudging_each_number(self):
        # Training rests on this: the forger learns only through the detective's gradient with respect to its input.
        net = MLP((2, 4, 3, 1), seed=1)
        x = np.random.default_rng(2).standard_normal((3, 2))
        net.forward(x)
        param_grad, input_grad = net.backward(np.ones((3, 1)))
        eps, measured_params, measured_input = 1e-6, np.zeros_like(net.params), np.zeros_like(x)
        for i in range(net.params.size):
            net.params[i] += eps
            up = net.forward(x).sum()
            net.params[i] -= 2 * eps
            down = net.forward(x).sum()
            net.params[i] += eps
            measured_params[i] = (up - down) / (2 * eps)
        for idx in np.ndindex(x.shape):
            x[idx] += eps
            up = net.forward(x).sum()
            x[idx] -= 2 * eps
            down = net.forward(x).sum()
            x[idx] += eps
            measured_input[idx] = (up - down) / (2 * eps)
        assert np.allclose(param_grad, measured_params, atol=1e-6)
        assert np.allclose(input_grad, measured_input, atol=1e-6)

    def test_given_the_toy_data_every_sample_lies_near_one_of_eight_points_on_a_circle_of_radius_2(self):
        samples = ring_of_gaussians(500, seed=0)
        distance_to_nearest_mode = np.linalg.norm(samples[:, None] - ring_modes()[None], axis=2).min(axis=1)
        # The modes have a spread of 0.05, so 5 spreads out is already vanishingly rare.
        assert distance_to_nearest_mode.max() < 0.25
        assert ring_modes()[2] == pytest.approx([0.0, 2.0])


class TestTheMinimaxValue:
    def test_given_real_verdicts_0_9_and_0_8_and_fake_verdicts_0_2_and_0_4_the_value_is_minus_0_53(self):
        # By hand: (ln 0.9 + ln 0.8)/2 + (ln 0.8 + ln 0.6)/2 = -0.164 - 0.367.
        assert value_function([0.9, 0.8], [0.2, 0.4]) == pytest.approx(-0.531, abs=1e-3)

    def test_given_a_detective_that_is_never_wrong_the_value_reaches_its_ceiling_of_zero(self):
        assert value_function([1.0, 1.0], [0.0, 0.0]) == pytest.approx(0.0, abs=1e-9)

    def test_given_a_detective_that_always_says_one_half_the_value_is_minus_log_4(self):
        assert value_function([0.5, 0.5], [0.5, 0.5]) == pytest.approx(-1.386, abs=1e-3)


class TestTheBestPossibleDetective:
    def test_given_real_density_0_3_and_fake_density_0_1_at_a_point_the_best_verdict_is_0_75(self):
        assert optimal_discriminator(0.3, 0.1) == pytest.approx(0.75)

    def test_given_a_forger_that_matches_the_data_the_best_detective_says_one_half_everywhere(self):
        p = np.array([0.4, 0.3, 0.2, 0.1])
        assert optimal_discriminator(p, p) == pytest.approx([0.5] * 4)

    def test_given_a_fixed_forger_a_trained_detective_arrives_at_the_formula_for_the_best_verdict(self):
        # Real 0.4 vs fake 0.1 gives 0.8; 0.3 vs 0.1 gives 0.75; 0.2 vs 0.4 gives 1/3; 0.1 vs 0.4 gives 0.2.
        verdicts = fit_discriminator_table([0.4, 0.3, 0.2, 0.1], [0.1, 0.1, 0.4, 0.4])
        assert verdicts == pytest.approx([0.8, 0.75, 1 / 3, 0.2], abs=1e-3)


class TestWhyTheForgerNeedsANonSaturatingLoss:
    def test_given_a_detective_99_percent_sure_a_fake_is_fake_the_original_loss_gives_the_forger_a_gradient_of_only_0_01(self):
        assert abs(generator_logit_gradient(0.01, "saturating")) == pytest.approx(0.01)
        assert abs(generator_logit_gradient(0.01, "non_saturating")) == pytest.approx(0.99)

    def test_given_an_undecided_detective_both_losses_give_the_same_gradient(self):
        assert generator_logit_gradient(0.5, "saturating") == generator_logit_gradient(0.5, "non_saturating") == -0.5

    def test_given_a_detective_with_a_long_head_start_the_original_loss_starves_the_forger_while_the_non_saturating_one_does_not(self):
        start, late = head_start_experiment(head_starts=(0, 800))
        assert late["d_fake"] < 0.01
        # Early on the two gradients are about the same size; after the head start the original one
        # has shrunk to a fraction of where it began and is over 100 times smaller than the other.
        assert start["saturating"] == pytest.approx(start["non_saturating"], rel=0.5)
        assert late["saturating"] < start["saturating"] / 4
        assert late["saturating"] < late["non_saturating"] / 100


class TestOscillation:
    def test_given_theta_1_and_psi_0_one_step_of_size_1_moves_the_detective_to_minus_one_half_and_leaves_the_forger(self):
        # Real data sits at 0 and the fake at 1; the detective's slope psi turns negative to call x = 1 fake.
        path = dirac_gan(theta=1.0, psi=0.0, lr=1.0, steps=1)
        assert path[1] == pytest.approx([1.0, -0.5])

    def test_given_theta_1_and_psi_0_the_second_step_moves_the_forger_towards_the_data_at_0(self):
        # By hand: sigma(-0.5) = 0.3775, so theta = 1 - 0.3775 * 0.5 and psi = -0.5 - 0.3775.
        path = dirac_gan(theta=1.0, psi=0.0, lr=1.0, steps=2)
        assert path[2] == pytest.approx([0.811, -0.8775], abs=1e-3)

    def test_given_plain_simultaneous_steps_the_players_spiral_away_from_the_equilibrium(self):
        path = dirac_gan(theta=1.0, psi=0.0, lr=0.2, steps=300)
        assert np.linalg.norm(path[-1]) > 1.5 * np.linalg.norm(path[0])

    def test_given_alternating_steps_the_players_circle_the_equilibrium_without_ever_settling(self):
        distances = np.linalg.norm(dirac_gan(theta=1.0, psi=0.0, lr=0.2, steps=300, alternating=True), axis=1)
        assert distances[-100:].min() > 0.7


class TestModeCollapse:
    def test_given_samples_piled_on_three_modes_coverage_counts_three(self):
        modes = ring_modes()
        samples = np.repeat(modes[[0, 3, 5]], 100, axis=0)
        assert mode_coverage(samples)["covered"] == 3

    def test_given_samples_far_from_every_mode_none_count_as_high_quality(self):
        assert mode_coverage(np.zeros((50, 2)))["quality"] == 0.0

    def test_given_a_detective_that_learns_3_times_faster_than_the_forger_all_eight_modes_are_covered(self, runs):
        assert runs[BALANCED]["coverage"][-1]["covered"] == 8

    def test_given_equal_learning_rates_the_forger_collapses_onto_at_most_half_the_modes(self, runs):
        late = runs[EQUAL]["coverage"][-4:]
        assert max(c["covered"] for c in late) <= 4

    def test_given_equal_learning_rates_the_forger_hops_from_one_set_of_modes_to_another(self, runs):
        late = [c["which"] for c in runs[EQUAL]["coverage"][-6:]]
        assert len(set(late)) >= 3

    def test_given_a_detective_that_learns_10_times_faster_the_forger_gets_stuck_on_part_of_the_ring(self, runs):
        late = runs[DETECTIVE_TOO_FAST]["coverage"][-4:]
        assert len({c["which"] for c in late}) == 1
        assert late[-1]["covered"] < 8


class TestStabilizers:
    def test_given_an_r1_gradient_penalty_the_players_spiral_in_to_the_equilibrium(self):
        path = dirac_gan(theta=1.0, psi=0.0, lr=0.2, steps=300, gamma=1.0)
        assert np.linalg.norm(path[-1]) < 0.05

    def test_given_two_single_point_piles_js_divergence_is_log_2_however_far_apart_they_are(self):
        # Disjoint piles: the detective can always tell them apart perfectly, so distance stops mattering.
        grid = np.arange(11)
        data = (grid == 0).astype(float)
        near, far = (grid == 1).astype(float), (grid == 10).astype(float)
        assert js_divergence(data, near) == pytest.approx(math.log(2))
        assert js_divergence(data, far) == pytest.approx(math.log(2))

    def test_given_a_pile_shifted_3_to_the_right_the_wasserstein_distance_is_3(self):
        pile = np.random.default_rng(0).standard_normal(200)
        assert wasserstein_1d(pile, pile + 3.0) == pytest.approx(3.0)

    def test_given_a_matrix_that_stretches_one_direction_by_3_its_largest_stretch_is_3(self):
        assert largest_stretch(np.array([[3.0, 0.0], [0.0, 1.0]])) == pytest.approx(3.0, abs=1e-6)

    def test_given_spectral_normalization_the_largest_stretch_becomes_1(self):
        W = np.array([[2.0, 1.0], [1.0, 3.0], [0.0, 1.0]])
        assert largest_stretch(spectrally_normalize(W)) == pytest.approx(1.0, abs=1e-6)
