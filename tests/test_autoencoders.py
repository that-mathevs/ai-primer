"""Specification for primer.ml.generative.autoencoders: squeezing data into a code and back, and sampling new data from it."""

import math

import numpy as np
import pytest

from primer.ml.generative.autoencoders import (
    JUNK_DISTANCE,
    STROKE_KINDS,
    VAE,
    Autoencoder,
    codes_in_box,
    distance_to_data,
    expected_squared_error,
    generate,
    interpolate,
    kl_to_standard_normal,
    largest_step,
    make_strokes,
    pca_reconstruction,
    reconstruction_error,
    reparameterize,
    train,
    train_linear_autoencoder,
    trained_autoencoder,
    trained_vae,
    worked_example_line,
)


class TestTheStrokeImages:
    def test_given_the_toy_dataset_each_image_is_an_8_by_8_picture_of_one_of_four_stroke_kinds(self):
        X, kinds, offsets = make_strokes()
        assert X.shape == (200, 64)
        assert sorted(set(kinds.tolist())) == [0, 1, 2, 3]
        assert len(STROKE_KINDS) == 4

    def test_given_any_stroke_its_brightest_pixel_holds_at_least_0_707_of_full_ink(self):
        # Worst case: a line exactly between two pixel rows, 0.5 px from each: e^(-0.5² / (2 · 0.6²)) = 0.707.
        X, _, _ = make_strokes()
        assert 0.707 < X.max(axis=1).min() <= 1.0


class TestSqueezingIntoACode:
    def test_given_a_point_on_the_line_y_equals_2x_its_one_number_code_rebuilds_it_exactly(self):
        # (1, 2) lies on the line, so nothing is lost: its code is √5 and the rebuild is the point itself.
        ex = worked_example_line()
        assert ex["on_line_code"] == pytest.approx(math.sqrt(5))
        assert ex["on_line_rebuilt"] == pytest.approx([1.0, 2.0])

    def test_given_the_point_2_3_off_the_line_it_is_rebuilt_as_1_6_3_2_with_squared_error_0_2(self):
        # Hand calculation: code = (2·1 + 3·2)/√5 = 8/√5; rebuild = (8/√5)·(1, 2)/√5 = (1.6, 3.2); error 0.4² + 0.2².
        ex = worked_example_line()
        assert ex["off_line_code"] == pytest.approx(8 / math.sqrt(5))
        assert ex["off_line_rebuilt"] == pytest.approx([1.6, 3.2])
        assert ex["off_line_error"] == pytest.approx(0.2)

    def test_given_training_the_reconstruction_error_falls_to_under_a_tenth_of_where_it_started(self):
        X, _, _ = make_strokes()
        history = train(Autoencoder(seed=1), X, steps=400)
        assert history[-1]["reconstruction"] < history[0]["reconstruction"] / 10

    def test_given_stroke_images_a_two_number_code_rebuilds_them_far_better_than_two_principal_components(self):
        # PCA is a flat projection; strokes lie on a curved surface in pixel space, which a nonlinear code can follow.
        X, _, _ = make_strokes()
        pca_error = float(np.mean(np.sum((pca_reconstruction(X, 2) - X) ** 2, axis=1)))
        assert reconstruction_error(trained_autoencoder(), X) < pca_error / 10

    def test_given_trained_codes_each_strokes_nearest_neighbour_in_code_space_is_the_same_kind_of_stroke(self):
        X, kinds, _ = make_strokes()
        Z = trained_autoencoder().encode(X)
        d = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
        np.fill_diagonal(d, np.inf)
        assert np.mean(kinds[d.argmin(axis=1)] == kinds) > 0.9


class TestPCAIsTheLinearSpecialCase:
    @pytest.fixture(scope="class")
    def points_near_the_line(self):
        rng = np.random.default_rng(0)
        t = rng.uniform(-2, 2, 100)
        return np.c_[t, 2 * t] + rng.normal(0, 0.1, (100, 2))

    def test_given_points_near_y_equals_2x_a_linear_autoencoder_learns_the_lines_direction(self, points_near_the_line):
        _, W_dec = train_linear_autoencoder(points_near_the_line, n_code=1)
        direction = W_dec[0] / np.linalg.norm(W_dec[0])
        # The line's own direction, (1, 2)/√5 = (0.447, 0.894), up to sign.
        assert abs(direction @ np.array([0.4472, 0.8944])) > 0.999

    def test_given_the_same_points_a_linear_autoencoder_rebuilds_them_as_well_as_one_principal_component(self, points_near_the_line):
        X = points_near_the_line
        W_enc, W_dec = train_linear_autoencoder(X, n_code=1)
        mean = X.mean(axis=0)
        ours = np.mean(np.sum(((X - mean) @ W_enc @ W_dec + mean - X) ** 2, axis=1))
        pca = np.mean(np.sum((pca_reconstruction(X, 1) - X) ** 2, axis=1))
        assert ours == pytest.approx(pca, rel=0.02)


class TestHolesInAPlainAutoencodersCodeSpace:
    def test_given_codes_drawn_from_a_standard_normal_a_plain_autoencoder_decodes_junk(self):
        # Nothing in its training told it where codes should live, so a bell curve at 0 misses them.
        X, _, _ = make_strokes()
        assert np.median(distance_to_data(generate(trained_autoencoder(), 500, seed=5), X)) > 2 * JUNK_DISTANCE

    def test_given_codes_drawn_anywhere_inside_its_own_code_range_a_quarter_or_more_decode_to_junk(self):
        X, _, _ = make_strokes()
        ae = trained_autoencoder()
        images = ae.decode(codes_in_box(ae.encode(X), 500, seed=6))
        assert np.mean(distance_to_data(images, X) > JUNK_DISTANCE) > 0.25


class TestTheReparameterizationTrick:
    def test_given_mean_half_spread_half_and_noise_1_2_the_code_is_1_1(self):
        # z = μ + σ·ε = 0.5 + 0.5 · 1.2
        assert reparameterize(np.array([0.5]), np.log(np.array([0.25])), np.array([1.2])) == pytest.approx([1.1])

    def test_given_many_noise_draws_the_codes_average_to_the_mean_and_spread_by_the_spread(self):
        eps = np.random.default_rng(0).standard_normal(200_000)
        z = reparameterize(np.full_like(eps, 0.5), np.full_like(eps, np.log(0.25)), eps)
        assert z.mean() == pytest.approx(0.5, abs=0.01)
        assert z.std() == pytest.approx(0.5, abs=0.01)

    def test_given_the_noise_held_fixed_backprop_through_the_sampling_step_matches_finite_differences(self):
        # The trick makes sampling an ordinary sum, so the chain rule reaches the encoder; nudging each weight agrees.
        X, _, _ = make_strokes(n=6)
        model = VAE(n_hidden=5, seed=3)
        noise = np.random.default_rng(1).standard_normal((6, 2))
        _, grads = model.loss_and_gradients(X, noise)
        h = 1e-5
        for name in ("W1", "b2", "W3", "b4"):
            for idx in [(0,) * model.params[name].ndim, tuple(s - 1 for s in model.params[name].shape)]:
                old = model.params[name][idx]
                model.params[name][idx] = old + h
                up = model.loss_and_gradients(X, noise)[0]["loss"]
                model.params[name][idx] = old - h
                down = model.loss_and_gradients(X, noise)[0]["loss"]
                model.params[name][idx] = old
                assert grads[name][idx] == pytest.approx((up - down) / (2 * h), rel=1e-4, abs=1e-8)


class TestTheKLPenalty:
    def test_given_mean_0_and_spread_1_the_code_already_matches_the_standard_normal_and_pays_nothing(self):
        assert kl_to_standard_normal(np.array([0.0]), np.array([0.0])) == pytest.approx(0.0)

    def test_given_mean_half_and_spread_half_the_penalty_is_0_443(self):
        # ½(0.5² + 0.5² − ln 0.25 − 1) = ½(0.25 + 0.25 + 1.386 − 1) = 0.443
        assert kl_to_standard_normal(np.array([0.5]), np.log(np.array([0.25]))) == pytest.approx(0.443, abs=5e-4)

    def test_given_two_coordinates_their_penalties_add(self):
        # The second coordinate is already standard normal, so only the first one costs anything.
        assert kl_to_standard_normal(np.array([0.5, 0.0]), np.log(np.array([0.25, 1.0]))) == pytest.approx(0.443, abs=5e-4)

    def test_given_the_closed_form_it_agrees_with_an_average_over_samples(self):
        # KL is the average, over codes drawn from q, of log q(z) − log p(z); estimate it by brute force.
        mu, sigma = 0.8, 0.3
        z = np.random.default_rng(0).normal(mu, sigma, 400_000)
        log_q = -0.5 * ((z - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))
        log_p = -0.5 * z**2 - np.log(np.sqrt(2 * np.pi))
        estimate = float(np.mean(log_q - log_p))
        assert kl_to_standard_normal(np.array([mu]), np.log(np.array([sigma**2]))) == pytest.approx(estimate, abs=0.01)


class TestSamplingNewStrokes:
    def test_given_codes_drawn_from_a_standard_normal_a_vae_decodes_images_close_to_real_strokes(self):
        X, _, _ = make_strokes()
        vae_distance = np.median(distance_to_data(generate(trained_vae(), 500, seed=5), X))
        ae_distance = np.median(distance_to_data(generate(trained_autoencoder(), 500, seed=5), X))
        assert vae_distance < ae_distance / 4

    def test_given_a_trained_vae_its_codes_crowd_around_zero_with_spread_about_one(self):
        # The KL penalty pulls every code towards the standard normal we later sample from.
        X, _, _ = make_strokes()
        mu = trained_vae().encode(X)
        assert np.abs(mu.mean(axis=0)).max() < 0.5
        assert 0.5 < mu.std(axis=0).mean() < 1.5

    def test_given_a_walk_between_two_strokes_it_starts_and_ends_at_their_reconstructions(self):
        X, _, _ = make_strokes()
        vae = trained_vae()
        frames = interpolate(vae, X[0], X[1], steps=9)
        assert frames.shape == (9, 64)
        assert frames[0] == pytest.approx(vae.reconstruct(X[:1])[0])
        assert frames[-1] == pytest.approx(vae.reconstruct(X[1:2])[0])

    def test_given_walks_between_random_pairs_the_vaes_images_change_in_smaller_steps_than_a_plain_autoencoders(self):
        X, _, _ = make_strokes()
        pairs = np.random.default_rng(3).integers(0, len(X), (60, 2))

        def typical_largest_step(model):
            return np.mean([largest_step(interpolate(model, X[a], X[b], steps=9)) for a, b in pairs])

        assert typical_largest_step(trained_vae()) < 0.85 * typical_largest_step(trained_autoencoder())


class TestTheBetaTradeOff:
    def test_given_a_larger_beta_reconstructions_get_worse(self):
        X, _, _ = make_strokes()
        errors = [reconstruction_error(trained_vae(beta), X) for beta in (0.03, 0.3, 3.0)]
        assert errors == sorted(errors)

    def test_given_a_larger_beta_the_codes_pay_less_kl(self):
        X, _, _ = make_strokes()
        kls = [float(np.mean(kl_to_standard_normal(*trained_vae(beta).encode_distribution(X)))) for beta in (0.03, 0.3, 3.0)]
        assert kls == sorted(kls, reverse=True)

    def test_given_a_huge_beta_the_code_is_ignored_and_every_sample_is_the_same_grey_smear(self):
        # Posterior collapse: carrying information costs more KL than it saves in reconstruction.
        X, _, _ = make_strokes()
        samples = generate(trained_vae(3.0), 200, seed=5)
        assert samples.max(axis=1).mean() < 0.5  # real strokes peak above 0.8
        assert samples.std(axis=0).max() < 0.05  # every sample is nearly the same image

    def test_given_two_equally_likely_pixels_0_and_1_the_squared_error_best_guess_is_grey(self):
        # Squared error is minimised by the average: that is why VAE samples blur.
        assert expected_squared_error(0.5, [0.0, 1.0]) == pytest.approx(0.25)
        assert expected_squared_error(0.0, [0.0, 1.0]) == pytest.approx(0.5)
        assert expected_squared_error(1.0, [0.0, 1.0]) == pytest.approx(0.5)
