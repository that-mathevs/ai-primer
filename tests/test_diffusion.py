"""Specification for primer.ml.generative.diffusion: turning noise into data one small step at a time."""

import numpy as np
import pytest

from primer.ml.generative.diffusion import (
    add_noise,
    ddim_sample,
    ddim_step,
    ddpm_sample,
    ddpm_step,
    denoiser_gradient_check,
    euler_sample,
    euler_step,
    flow_point,
    flow_target,
    guidance_sweep,
    guided_noise,
    latent_shrink,
    linear_schedule,
    make_blobs,
    noise_step,
    noise_to_score,
    patch_tokens,
    trained_models,
    two_way_distance,
)


@pytest.fixture(scope="module")
def models():
    # Training three tiny networks takes about a second, so every scenario shares one set.
    return trained_models()


@pytest.fixture(scope="module")
def fresh_gap(models):
    # The yardstick: a second, independent draw of the real blobs is as close as samples can hope to be.
    fresh, _ = make_blobs(seed=99)
    return two_way_distance(fresh, models["data"])


class TestTheNoisingProcess:
    def test_given_alpha_bar_0_64_a_point_at_2_with_noise_0_5_lands_at_1_9(self):
        # √0.64 · 2 + √0.36 · 0.5 = 0.8 · 2 + 0.6 · 0.5 = 1.6 + 0.3
        assert add_noise(2.0, 0.64, 0.5) == pytest.approx(1.9)

    def test_given_step_sizes_0_1_and_0_2_the_signal_left_after_two_steps_is_0_72(self):
        # ᾱ multiplies the share each step keeps: 0.9 · 0.8.
        assert linear_schedule(T=2, beta_first=0.1, beta_last=0.2).alpha_bars.tolist() == pytest.approx([1.0, 0.9, 0.72])

    def test_when_noise_is_added_twice_in_small_steps_the_result_matches_the_one_jump_shortcut(self):
        # Two steps with β = 0.1 then 0.2 from x₀ = 1: the shortcut says mean √0.72 = 0.849, variance 1 − 0.72 = 0.28.
        rng = np.random.default_rng(0)
        x = np.ones(40_000)
        for beta in (0.1, 0.2):
            x = noise_step(x, beta, rng.standard_normal(x.shape))
        assert x.mean() == pytest.approx(0.849, abs=0.01)
        assert x.var() == pytest.approx(0.28, abs=0.01)

    def test_given_the_default_schedule_less_than_one_percent_of_the_signal_survives_the_last_step(self):
        assert linear_schedule().alpha_bars[-1] < 0.01

    def test_when_the_blobs_are_noised_to_the_last_step_they_look_like_plain_standard_noise(self):
        data, _ = make_blobs()
        schedule = linear_schedule()
        noisy = add_noise(data, schedule.alpha_bars[-1], np.random.default_rng(1).standard_normal(data.shape))
        assert np.abs(noisy.mean(axis=0)).max() < 0.1
        assert noisy.std(axis=0) == pytest.approx([1.0, 1.0], abs=0.1)


class TestLearningToDenoise:
    def test_given_the_hand_written_backward_pass_its_gradients_match_finite_differences(self):
        assert denoiser_gradient_check() < 1e-6

    def test_given_an_untrained_denoiser_its_first_score_is_what_guessing_zero_scores(self, models):
        # Its last layer starts near zero, and guessing ε = 0 scores E‖ε‖² = 2 in two dimensions.
        assert models["noise"].history[0] == pytest.approx(2.0, abs=0.3)

    def test_given_a_trained_denoiser_it_guesses_the_noise_far_better_than_guessing_zero(self, models):
        # Guessing ε = 0 scores E‖ε‖² = 2 in two dimensions.
        assert np.mean(models["noise"].history[-100:]) < 1.0

    def test_given_a_noise_guess_of_0_9_at_alpha_bar_0_64_the_score_is_minus_1_5(self):
        # −0.9 / √0.36: for standard normal data at x = 1.5 the true score is −x, and this is it.
        assert noise_to_score(0.9, 0.64) == pytest.approx(-1.5)


class TestSamplingBackwards:
    def test_given_one_denoising_step_with_no_fresh_noise_the_point_moves_from_1_to_0_9(self):
        # (1 − 0.19 / √0.25 · 0.5) / √0.81 = (1 − 0.19) / 0.9
        assert ddpm_step(1.0, 0.5, beta=0.19, alpha_bar=0.75, z=0.0) == pytest.approx(0.9)

    def test_given_one_denoising_step_with_fresh_noise_0_5_the_point_lands_at_1_118(self):
        # 0.9 + √0.19 · 0.5
        assert ddpm_step(1.0, 0.5, beta=0.19, alpha_bar=0.75, z=0.5) == pytest.approx(1.1179, abs=1e-4)

    def test_given_a_perfect_noise_guess_a_deterministic_jump_to_alpha_bar_1_recovers_the_clean_point(self):
        assert ddim_step(1.9, 0.5, 0.64, 1.0) == pytest.approx(2.0)

    def test_given_a_deterministic_jump_to_alpha_bar_0_96_the_point_lands_at_2_06(self):
        # Predict x̂₀ = (1.9 − 0.6 · 0.5) / 0.8 = 2, then re-noise lightly: √0.96 · 2 + √0.04 · 0.5.
        assert ddim_step(1.9, 0.5, 0.64, 0.96) == pytest.approx(2.0596, abs=1e-4)

    def test_given_pure_noise_a_hundred_small_steps_land_the_samples_on_the_blobs(self, models, fresh_gap):
        samples = ddpm_sample(models["noise"], seed=1)
        assert two_way_distance(samples, models["data"]) < 1.3 * fresh_gap

    def test_given_pure_noise_that_was_never_denoised_it_sits_far_from_the_blobs(self, models, fresh_gap):
        noise = np.random.default_rng(1).standard_normal((600, 2))
        assert two_way_distance(noise, models["data"]) > 2 * fresh_gap

    def test_given_only_twenty_deterministic_jumps_the_samples_still_land_on_the_blobs(self, models, fresh_gap):
        samples = ddim_sample(models["noise"], steps=20, seed=1)
        assert two_way_distance(samples, models["data"]) < 1.3 * fresh_gap

    def test_given_the_same_starting_noise_the_deterministic_sampler_draws_the_same_points(self, models):
        assert np.array_equal(ddim_sample(models["noise"], steps=10, seed=4), ddim_sample(models["noise"], steps=10, seed=4))


class TestFlowMatching:
    def test_given_noise_at_minus_1_and_data_at_2_a_quarter_of_the_way_along_the_point_is_at_minus_0_25(self):
        # 0.75 · (−1) + 0.25 · 2
        assert flow_point(-1.0, 2.0, 0.25) == pytest.approx(-0.25)

    def test_given_noise_at_minus_1_and_data_at_2_the_velocity_to_learn_is_3_everywhere_on_the_line(self):
        assert flow_target(-1.0, 2.0) == pytest.approx(3.0)

    def test_given_the_true_velocity_one_euler_step_of_0_75_arrives_exactly_at_the_data(self):
        # A straight path has nothing to cut corners on: −0.25 + 0.75 · 3 = 2.
        assert euler_step(-0.25, 3.0, 0.75) == pytest.approx(2.0)

    def test_given_ten_euler_steps_the_learned_flow_lands_the_samples_on_the_blobs(self, models, fresh_gap):
        samples = euler_sample(models["flow"], steps=10, seed=1)
        assert two_way_distance(samples, models["data"]) < 1.3 * fresh_gap

    def test_given_only_five_steps_the_flow_lands_closer_to_the_blobs_than_diffusion_does(self, models):
        flow = two_way_distance(euler_sample(models["flow"], steps=5, seed=1), models["data"])
        diffusion = two_way_distance(ddim_sample(models["noise"], steps=5, seed=1), models["data"])
        assert flow < diffusion

    def test_given_a_single_euler_step_every_sample_lands_near_the_empty_average_of_the_data(self, models):
        # One step follows the average direction, so it lands on the average point: the empty middle, not a blob.
        samples = euler_sample(models["flow"], steps=1, seed=1)
        assert np.linalg.norm(samples, axis=1).mean() < 0.3
        assert np.linalg.norm(models["data"], axis=1).mean() > 0.9


class TestGuidance:
    def test_given_guesses_0_2_without_and_0_5_with_the_label_a_weight_of_3_gives_1_1(self):
        # 0.2 + 3 · (0.5 − 0.2): step past the labelled guess, away from the unlabelled one.
        assert guided_noise(0.2, 0.5, 3.0) == pytest.approx(1.1)

    def test_given_a_weight_of_0_the_guided_guess_ignores_the_label(self):
        assert guided_noise(0.2, 0.5, 0.0) == pytest.approx(0.2)

    def test_given_a_weight_of_1_the_guided_guess_is_the_labelled_guess(self):
        assert guided_noise(0.2, 0.5, 1.0) == pytest.approx(0.5)

    def test_given_the_south_west_label_nearly_every_sample_lands_in_the_south_west_blob(self, models):
        row = guidance_sweep((1.0,), models=models)[0]
        assert row["on_target"] > 0.9

    def test_given_a_weight_of_0_samples_spread_over_all_four_blobs(self, models):
        row = guidance_sweep((0.0,), models=models)[0]
        assert row["on_target"] < 0.4

    def test_given_a_stronger_guidance_weight_the_samples_bunch_more_tightly(self, models):
        # The price of guidance: more on-label, less varied.
        rows = guidance_sweep((1.0, 3.0), models=models)
        assert rows[1]["spread"] < rows[0]["spread"]


class TestScalingUpToImages:
    def test_given_a_512_pixel_colour_image_its_latent_has_48_times_fewer_numbers(self):
        # 512 · 512 · 3 = 786,432 pixels' worth; 64 · 64 · 4 = 16,384 latent numbers.
        shrink = latent_shrink(512, 512)
        assert (shrink["pixels"], shrink["latents"], shrink["factor"]) == (786_432, 16_384, 48.0)

    def test_given_a_64_by_64_latent_cut_into_2_by_2_patches_the_transformer_reads_1024_tokens(self):
        assert patch_tokens(64, 64, patch=2) == 1024

    def test_given_16_frames_of_video_the_same_patches_make_16_times_the_tokens(self):
        assert patch_tokens(64, 64, patch=2, frames=16) == 16_384
