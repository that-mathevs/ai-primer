"""Specification for primer.ml.positional: how a transformer learns word order."""

import numpy as np
import pytest

from primer.ml.attention import scaled_dot_product_attention

from primer.ml.positional import (
    apply_rope,
    encode_sentence,
    interpolate_positions,
    ntk_scaled_base,
    rope_frequencies,
    sinusoidal_encoding,
)


class TestSinusoidalEncoding:
    def test_at_position_zero_sine_dimensions_are_0_and_cosine_dimensions_are_1(self):
        assert sinusoidal_encoding(1, 6)[0].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

    def test_at_position_one_with_four_dimensions_the_frequencies_are_1_and_one_hundredth(self):
        # d=4: pair 0 uses 1/10000^(0/4) = 1, pair 1 uses 1/10000^(2/4) = 1/100.
        assert sinusoidal_encoding(2, 4)[1] == pytest.approx([0.841471, 0.540302, 0.0099998, 0.99995], abs=1e-6)

    def test_every_value_stays_between_minus_one_and_one(self):
        pe = sinusoidal_encoding(2048, 64)
        assert pe.min() >= -1.0 and pe.max() <= 1.0

    def test_the_similarity_of_two_positions_depends_only_on_their_distance(self):
        # This is why the model can learn "3 tokens back" once and apply it anywhere.
        pe = sinusoidal_encoding(100, 32)
        assert pe[10] @ pe[13] == pytest.approx(pe[70] @ pe[73])


class TestRotaryPositionEmbedding:
    def test_at_position_zero_a_vector_is_left_unchanged(self):
        x = np.array([0.3, -1.2, 2.0, 0.5])
        assert apply_rope(x, 0) == pytest.approx(x)

    def test_a_2d_vector_at_position_one_turns_by_one_radian(self):
        # d=2 has a single pair with θ = 10000^0 = 1, so position 1 is a 1-radian turn.
        assert apply_rope(np.array([1.0, 0.0]), 1) == pytest.approx([0.540302, 0.841471], abs=1e-6)

    def test_rotation_never_changes_a_vectors_length(self):
        x = np.random.default_rng(0).standard_normal(64)
        assert np.linalg.norm(apply_rope(x, 1234)) == pytest.approx(np.linalg.norm(x))

    def test_the_attention_score_depends_only_on_the_distance_between_query_and_key(self):
        rng = np.random.default_rng(1)
        q, k = rng.standard_normal(64), rng.standard_normal(64)
        assert apply_rope(q, 3) @ apply_rope(k, 7) == pytest.approx(apply_rope(q, 103) @ apply_rope(k, 107))

    def test_the_attention_score_changes_when_the_distance_changes(self):
        rng = np.random.default_rng(1)
        q, k = rng.standard_normal(64), rng.standard_normal(64)
        assert apply_rope(q, 3) @ apply_rope(k, 7) != pytest.approx(apply_rope(q, 3) @ apply_rope(k, 8))

    def test_a_whole_sequence_rotates_each_row_by_its_own_position(self):
        x = np.tile([1.0, 0.0], (3, 1))
        # Rows are positions 0, 1, 2 with θ = 1: angles 0, 1 and 2 radians.
        expected = np.array([[1.0, 0.0], [0.540302, 0.841471], [-0.416147, 0.909297]])
        assert apply_rope(x) == pytest.approx(expected, abs=1e-6)


class TestOrderBlindnessWithoutPositions:
    # Worked example: dog=(1,0), bites=(0,1), man=(1,1); q = k = v = the word vector.
    DOG_BITES_MAN = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    def test_with_identity_projections_dog_first_outputs_0_802_0_599(self):
        X = self.DOG_BITES_MAN
        assert scaled_dot_product_attention(X, X, X)[0][0] == pytest.approx([0.802, 0.599], abs=1e-3)

    def test_with_identity_projections_dog_last_outputs_the_same_0_802_0_599(self):
        X = self.DOG_BITES_MAN[::-1]
        assert scaled_dot_product_attention(X, X, X)[0][2] == pytest.approx([0.802, 0.599], abs=1e-3)

    def test_without_positions_dog_bites_man_and_man_bites_dog_pool_to_the_same_vector(self):
        a = encode_sentence("dog bites man", scheme="none").mean(axis=0)
        b = encode_sentence("man bites dog", scheme="none").mean(axis=0)
        assert a == pytest.approx(b)

    def test_without_positions_each_word_gets_the_same_output_whatever_its_slot(self):
        # "dog" is word 0 in one sentence and word 2 in the other.
        a = encode_sentence("dog bites man", scheme="none")
        b = encode_sentence("man bites dog", scheme="none")
        assert a[0] == pytest.approx(b[2])

    @pytest.mark.parametrize("scheme", ["sinusoidal", "rope"])
    def test_with_positions_the_two_sentences_pool_to_different_vectors(self, scheme):
        a = encode_sentence("dog bites man", scheme=scheme).mean(axis=0)
        b = encode_sentence("man bites dog", scheme=scheme).mean(axis=0)
        assert not np.allclose(a, b)

    def test_a_causal_mask_alone_already_leaks_word_order(self):
        # Token i sees exactly i+1 tokens, so "who came first" is recoverable even with no encoding.
        a = encode_sentence("dog bites man", scheme="none", causal=True).mean(axis=0)
        b = encode_sentence("man bites dog", scheme="none", causal=True).mean(axis=0)
        assert not np.allclose(a, b)


class TestContextExtension:
    def test_position_interpolation_maps_position_32000_of_a_32k_context_to_4000_for_a_4k_model(self):
        assert interpolate_positions(np.array([32000]), trained_len=4096, new_len=32768)[0] == pytest.approx(4000)

    def test_position_interpolation_keeps_every_position_inside_the_trained_range(self):
        squeezed = interpolate_positions(np.arange(32768), trained_len=4096, new_len=32768)
        assert squeezed.max() < 4096

    def test_ntk_scaling_leaves_the_fastest_rotation_untouched_to_keep_local_order_sharp(self):
        assert rope_frequencies(128, ntk_scaled_base(10000, scale=8, d=128))[0] == pytest.approx(1.0)

    def test_ntk_scaling_slows_the_slowest_rotation_by_exactly_the_scale_factor(self):
        # Slowest pair: θ = base^(-(d-2)/d). Raising base to base·s^(d/(d-2)) divides it by s.
        slow_before = rope_frequencies(128, 10000)[-1]
        slow_after = rope_frequencies(128, ntk_scaled_base(10000, scale=8, d=128))[-1]
        assert slow_after == pytest.approx(slow_before / 8)
