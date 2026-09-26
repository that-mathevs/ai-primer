"""Specification for primer.ml.generative.multimodal: images, audio and video into a language model."""

import numpy as np
import pytest

from primer.ml.generative.multimodal import (
    IMAGE,
    Projector,
    VisionEncoder,
    align_projector,
    audio_tokens,
    build_toy_vlm,
    caption_accuracy,
    chirp,
    count_image_tokens,
    dft_magnitudes,
    frame_count,
    hz_to_mel,
    image_to_patches,
    kmeans_codebook,
    learn_residual_codebooks,
    mel_filterbank,
    quantization_error,
    quantize,
    dequantize,
    residual_quantize,
    sample_frames,
    seconds_that_fit,
    stft,
    tone,
    toy_images,
    video_tokens,
    worked_example_patches,
    worked_example_projection,
)


class TestCuttingAnImageIntoPatches:
    def test_given_a_4_by_4_image_and_2_pixel_patches_it_becomes_4_tokens_of_4_numbers(self):
        patches = image_to_patches(np.arange(16).reshape(4, 4), patch=2)
        assert patches.shape == (4, 4)

    def test_given_the_worked_image_the_top_right_patch_reads_2_3_6_7(self):
        # Row by row inside the patch: pixels (0,2), (0,3), then (1,2), (1,3).
        assert image_to_patches(np.arange(16).reshape(4, 4), patch=2)[1].tolist() == [2, 3, 6, 7]

    def test_given_a_side_the_patch_does_not_divide_it_refuses(self):
        # A ragged edge would leave a patch with missing pixels; real pipelines resize or pad first.
        with pytest.raises(ValueError):
            image_to_patches(np.zeros((10, 10)), patch=4)


class TestCountingImageTokens:
    def test_given_224_pixels_and_16_pixel_patches_an_image_is_196_tokens(self):
        assert count_image_tokens(224, 224, patch=16) == 196

    def test_given_336_pixels_and_14_pixel_patches_an_image_is_576_tokens(self):
        assert count_image_tokens(336, 336, patch=14) == 576

    def test_given_neighbouring_patches_merged_2_by_2_the_count_falls_fourfold(self):
        assert count_image_tokens(336, 336, patch=14, merge=2) == 144

    def test_given_twice_the_side_length_the_token_count_quadruples(self):
        assert count_image_tokens(448, 448, patch=16) == 4 * count_image_tokens(224, 224, patch=16)


class TestPatchEmbedding:
    def test_given_the_worked_projection_and_positions_the_tokens_are_the_hand_computed_ones(self):
        # Column 1 sums a patch's top row, column 2 its bottom row; positions add (row, column).
        tokens = worked_example_patches()["tokens"]
        assert tokens.tolist() == [[1, 9], [5, 14], [18, 25], [22, 30]]

    def test_given_two_identical_patches_in_different_places_their_tokens_differ(self):
        # Without positions, attention sees a bag of patches and could not tell top from bottom.
        encoder = VisionEncoder(image_size=8, patch=4, channels=1, d_model=16, n_layers=1, n_heads=2)
        tokens = encoder.embed(np.ones((8, 8)))
        assert not np.allclose(tokens[0], tokens[3])


class TestTheVisionEncoder:
    def test_given_a_32_pixel_colour_image_and_8_pixel_patches_it_returns_16_vectors(self):
        encoder = VisionEncoder(image_size=32, patch=8, channels=3, d_model=24, n_layers=2, n_heads=4)
        assert encoder(np.zeros((32, 32, 3))).shape == (16, 24)

    def test_given_a_change_in_the_last_patch_the_first_patch_output_changes_too(self):
        # An image encoder is not causal: every patch may look at every other, in both directions.
        encoder = VisionEncoder(image_size=8, patch=4, channels=1, d_model=16, n_layers=1, n_heads=2)
        image = np.random.default_rng(0).random((8, 8))
        changed = image.copy()
        changed[4:, 4:] += 1.0
        assert not np.allclose(encoder(image)[0], encoder(changed)[0])


class TestTheProjector:
    def test_given_the_worked_example_vector_1_2_it_projects_to_1_2_3(self):
        assert worked_example_projection().tolist() == [1, 2, 3]

    def test_given_vision_width_32_and_language_width_48_every_image_token_comes_out_48_wide(self):
        projector = Projector(32, 48, hidden=64)
        assert projector(np.zeros((16, 32))).shape == (16, 48)


class TestInterleavingImageAndText:
    def test_given_one_image_placeholder_the_sequence_grows_by_the_image_tokens_minus_one(self):
        model = build_toy_vlm()
        ids = np.array([1, 2, IMAGE, 3])
        sequence, _ = model.splice(ids, toy_images(1)[0][0])
        assert len(sequence) == 3 + model.vision.n_patches

    def test_given_the_placeholder_the_projected_image_tokens_sit_exactly_where_it_was(self):
        model = build_toy_vlm()
        image = toy_images(1)[0][0]
        sequence, is_image = model.splice(np.array([1, IMAGE, 3]), image)
        n = model.vision.n_patches
        assert is_image.tolist() == [False] + [True] * n + [False]
        assert np.allclose(sequence[1 : 1 + n], model.projector(model.vision(image)))

    def test_given_an_image_and_a_prompt_the_model_scores_every_vocabulary_entry_at_every_position(self):
        model = build_toy_vlm()
        logits = model(np.array([1, IMAGE, 3]), toy_images(1)[0][0])
        assert logits.shape == (2 + model.vision.n_patches, model.lm.wte.shape[0])

    def test_given_a_different_image_the_text_before_it_is_scored_the_same(self):
        # The language model is causal, so tokens before the picture cannot see it.
        model = build_toy_vlm()
        images, _ = toy_images(2)
        ids = np.array([1, 2, IMAGE, 3])
        a, b = model(ids, images[0]), model(ids, images[-1])
        assert np.allclose(a[:2], b[:2])
        assert not np.allclose(a[-1], b[-1])


class TestAligningVisionToLanguage:
    def test_given_an_untrained_projector_captions_are_little_better_than_chance(self):
        model = build_toy_vlm()
        images, labels = toy_images(20)
        # Four classes, so a blind guess is right a quarter of the time.
        assert caption_accuracy(model, images, labels) <= 0.5

    def test_given_projector_training_on_image_caption_pairs_the_right_caption_word_wins(self):
        model = build_toy_vlm()
        images, labels = toy_images(20)
        align_projector(model, images, labels)
        assert caption_accuracy(model, *toy_images(20, seed=99)) >= 0.9

    def test_given_alignment_training_the_language_model_and_vision_encoder_stay_frozen(self):
        model = build_toy_vlm()
        wte, patch_w = model.lm.wte.copy(), model.vision.W_E.copy()
        align_projector(model, *toy_images(20))
        assert np.array_equal(model.lm.wte, wte) and np.array_equal(model.vision.W_E, patch_w)

    def test_given_alignment_training_the_caption_loss_falls(self):
        history = align_projector(build_toy_vlm(), *toy_images(20))
        assert history["loss"][-1] < history["loss"][0] / 2


class TestHowMuchOfEachPitch:
    def test_given_a_2_cycle_wave_over_8_samples_all_its_energy_lands_in_bin_2_and_its_mirror(self):
        wave = [1, 0, -1, 0, 1, 0, -1, 0]
        assert np.round(dft_magnitudes(wave), 6).tolist() == [0, 0, 4, 0, 0, 0, 4, 0]

    def test_given_any_signal_the_magnitudes_match_numpys_fast_fourier_transform(self):
        x = np.random.default_rng(1).standard_normal(32)
        assert np.allclose(dft_magnitudes(x), np.abs(np.fft.fft(x)))


class TestTheSpectrogram:
    def test_given_one_second_at_16_khz_with_25_ms_windows_every_10_ms_there_are_98_frames(self):
        assert frame_count(16_000, window=400, hop=160) == 98

    def test_given_a_steady_1000_hz_tone_every_frame_peaks_in_the_bin_nearest_1000_hz(self):
        sr, window = 8000, 256
        spec = stft(tone(1000, 0.25, sr), window=window, hop=64)
        # Bin k holds frequency k * sr / window, so 1000 Hz is bin 32.
        assert set(spec.argmax(axis=1).tolist()) == {32}

    def test_given_a_rising_chirp_the_loudest_frequency_climbs_frame_by_frame(self):
        spec = stft(chirp(200, 3000, 1.0, 8000), window=256, hop=64)
        peaks = spec.argmax(axis=1)
        assert peaks[-1] > peaks[len(peaks) // 2] > peaks[0]


class TestTheMelScale:
    def test_given_1000_hz_it_is_1000_mel_by_construction(self):
        assert hz_to_mel(1000) == pytest.approx(1000, abs=0.1)

    def test_given_700_hz_it_is_about_781_mel(self):
        # 2595 · log10(1 + 700/700) = 2595 · log10(2) = 781.17
        assert hz_to_mel(700) == pytest.approx(781.17, abs=0.01)

    def test_given_a_filterbank_the_high_pitch_filters_are_wider_than_the_low_ones(self):
        bank = mel_filterbank(n_mels=10, n_fft=512, sample_rate=16_000)
        widths = (bank > 0).sum(axis=1)
        assert widths[-1] > 3 * widths[0]


class TestCountingAudioTokens:
    def test_given_30_seconds_with_10_ms_frames_halved_once_the_encoder_sees_1500_tokens(self):
        assert audio_tokens(30) == 1500

    def test_given_one_second_the_encoder_sees_50_tokens(self):
        assert audio_tokens(1) == 50


class TestVectorQuantization:
    def test_given_the_point_0_9_0_2_its_nearest_codebook_entry_is_number_1(self):
        # Squared distances to (0,0), (1,0), (0,1) are 0.85, 0.05 and 1.45.
        codebook = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        assert quantize(np.array([[0.9, 0.2]]), codebook).tolist() == [1]

    def test_given_ids_decoding_returns_the_codebook_vectors_they_name(self):
        codebook = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        assert dequantize(np.array([2, 1]), codebook).tolist() == [[0.0, 1.0], [1.0, 0.0]]

    def test_given_a_larger_learned_codebook_the_reconstruction_error_falls(self):
        patches = np.concatenate([image_to_patches(im, 4) for im in toy_images(30)[0]])
        small, large = kmeans_codebook(patches, 4), kmeans_codebook(patches, 32)
        # Pixel noise (0.3² = 0.09 per pixel) is a floor no codebook of patterns removes, so the drop is a third, not all.
        assert quantization_error(patches, large) < 0.7 * quantization_error(patches, small)

    def test_given_a_learned_codebook_it_beats_a_codebook_of_random_points(self):
        patches = np.concatenate([image_to_patches(im, 4) for im in toy_images(30)[0]])
        random = np.random.default_rng(0).random((16, patches.shape[1]))
        assert quantization_error(patches, kmeans_codebook(patches, 16)) < quantization_error(patches, random)

    def test_given_a_second_codebook_for_the_leftover_error_the_reconstruction_improves(self):
        patches = np.concatenate([image_to_patches(im, 4) for im in toy_images(30)[0]])
        books = learn_residual_codebooks(patches, k=8, stages=2)
        _, one = residual_quantize(patches, books[:1])
        ids, two = residual_quantize(patches, books)
        assert ids.shape == (2, len(patches))
        assert np.mean((patches - two) ** 2) < np.mean((patches - one) ** 2)


class TestVideoTokens:
    def test_given_10_seconds_at_30_frames_per_second_and_256_tokens_a_frame_it_is_76800_tokens(self):
        assert video_tokens(10, sample_fps=30, tokens_per_frame=256) == 76_800

    def test_given_one_frame_a_second_the_same_clip_is_2560_tokens(self):
        assert video_tokens(10, sample_fps=1, tokens_per_frame=256) == 2_560

    def test_given_tubelets_two_frames_deep_the_count_halves(self):
        assert video_tokens(10, sample_fps=2, tokens_per_frame=256, tubelet=2) == 2_560

    def test_given_300_frames_at_30_fps_sampling_one_a_second_keeps_every_30th_frame(self):
        assert sample_frames(300, video_fps=30, target_fps=1).tolist() == list(range(0, 300, 30))


class TestTheContextBudget:
    def test_given_128k_tokens_and_256_tokens_a_second_500_seconds_of_video_fit(self):
        assert seconds_that_fit(128_000, tokens_per_second=256) == 500
