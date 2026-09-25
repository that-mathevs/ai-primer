"""Specification for primer.ml.cnn_rnn: how networks see images and read sequences."""

import numpy as np
import pytest

from primer.ml import cnn_rnn as cr

# A 5×5 image: dark (0) on the left two columns, bright (1) on the right three, a vertical edge between.
EDGE_IMAGE = np.array([[0, 0, 1, 1, 1]] * 5, dtype=float)


class TestConvolution:
    def test_given_a_5x5_image_and_a_3x3_filter_the_feature_map_is_3x3(self):
        # (5 - 3) / 1 + 1 = 3 positions in each direction.
        assert cr.conv2d(EDGE_IMAGE, cr.VERTICAL_EDGE).shape == (3, 3)

    def test_given_a_vertical_edge_the_vertical_edge_filter_fires_3_where_the_edge_is_and_0_on_flat_bright_areas(self):
        # Top-left cell: rows (0, 0, 1) × weights (-1, 0, 1) = 1 per row, 3 rows = 3. Right column sees all-bright: -1 + 1 = 0.
        assert cr.conv2d(EDGE_IMAGE, cr.VERTICAL_EDGE).tolist() == [[3, 3, 0]] * 3

    def test_given_a_vertical_edge_the_horizontal_edge_filter_stays_silent(self):
        # Each filter detects one pattern; this image has no top-to-bottom change.
        assert cr.conv2d(EDGE_IMAGE, cr.HORIZONTAL_EDGE).tolist() == [[0, 0, 0]] * 3

    def test_given_a_straight_vertical_edge_the_diagonal_filter_still_fires_2_two_thirds_of_the_edge_filter(self):
        # A diagonal filter is not selective: its weights lean both ways, so a plain edge half-matches it.
        # Top-left patch rows (0, 0, 1) × rows of weights (0, 1, 1), (-1, 0, 1), (-1, -1, 0) = 1 + 1 + 0 = 2.
        assert cr.conv2d(EDGE_IMAGE, cr.DIAGONAL_EDGE).tolist() == [[2, 2, 0]] * 3

    def test_given_one_pixel_of_zero_padding_a_3x3_filter_keeps_the_5x5_size(self):
        assert cr.conv2d(EDGE_IMAGE, cr.VERTICAL_EDGE, padding=1).shape == (5, 5)

    def test_given_stride_2_the_filter_jumps_two_pixels_and_the_map_shrinks_to_2x2(self):
        # (5 - 3) / 2 + 1 = 2
        assert cr.conv2d(EDGE_IMAGE, cr.VERTICAL_EDGE, stride=2).shape == (2, 2)

    def test_given_a_224_pixel_image_a_7x7_filter_with_stride_2_and_padding_3_gives_112_positions(self):
        # (224 + 2·3 - 7) / 2 + 1 = 112, the first layer of ResNet.
        assert cr.conv_output_size(224, kernel=7, stride=2, padding=3) == 112

    def test_given_a_color_image_the_filter_sums_over_all_three_channels(self):
        # Three identical channels triple every response.
        rgb = np.stack([EDGE_IMAGE] * 3)
        kernel = np.stack([cr.VERTICAL_EDGE] * 3)
        assert cr.conv2d(rgb, kernel).tolist() == [[9, 9, 0]] * 3

    def test_given_the_same_inputs_the_result_matches_pytorch_conv2d(self):
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(0)
        image, kernel = rng.standard_normal((3, 9, 9)), rng.standard_normal((3, 3, 3))
        theirs = torch.nn.functional.conv2d(torch.tensor(image)[None], torch.tensor(kernel)[None], stride=2, padding=1)[0, 0].numpy()
        assert np.allclose(cr.conv2d(image, kernel, stride=2, padding=1), theirs)


class TestPooling:
    def test_given_a_4x4_map_2x2_max_pooling_keeps_the_strongest_value_in_each_quarter(self):
        fmap = np.array([[1, 3, 0, 0], [2, 4, 0, 1], [0, 0, 5, 2], [1, 0, 1, 6]], dtype=float)
        assert cr.max_pool2d(fmap, size=2).tolist() == [[4, 1], [1, 6]]


class TestWeightSharing:
    def test_given_a_224x224_color_image_64_filters_of_3x3_need_only_1792_parameters(self):
        # 3·3·3 weights per filter × 64 filters + 64 biases: the same filter is reused at every position.
        assert cr.conv_params(in_channels=3, out_channels=64, kernel=3) == 1_792

    def test_given_the_same_image_a_dense_layer_with_the_same_output_size_needs_483_billion_weights(self):
        # 150,528 inputs (224·224·3) × 3,211,264 outputs (224·224·64).
        assert cr.dense_params(inputs=224 * 224 * 3, outputs=224 * 224 * 64) == 483_385_147_392


class TestReceptiveField:
    @pytest.mark.parametrize("layers, seen", [(1, 3), (2, 5), (3, 7)])
    def test_given_stacked_3x3_filters_each_layer_widens_the_view_by_2_pixels(self, layers, seen):
        assert cr.receptive_field([(3, 1)] * layers) == seen

    def test_given_a_2x2_pool_between_two_3x3_convolutions_a_final_neuron_sees_8_pixels(self):
        # conv: 3; pool (stride 2): 3 + 1 = 4; conv after the pool steps 2 pixels at a time: 4 + 2·2 = 8.
        assert cr.receptive_field([(3, 1), (2, 2), (3, 1)]) == 8

    def test_given_two_3x3_filters_instead_of_one_5x5_the_same_view_costs_18_weights_instead_of_25(self):
        # The VGG insight: smaller stacked filters see as far with fewer weights and an extra nonlinearity.
        assert (cr.receptive_field([(3, 1), (3, 1)]), 2 * 3 * 3) == (cr.receptive_field([(5, 1)]), 18)


class TestImagePatchesAsTokens:
    def test_given_a_224x224_color_image_and_16_pixel_patches_a_vision_transformer_sees_196_tokens_of_768_numbers(self):
        # (224 / 16)² = 196 patches, each 16·16·3 = 768 values.
        image = np.zeros((224, 224, 3))
        assert cr.patchify(image, patch=16).shape == (196, 768)

    def test_given_a_4x4_image_and_2_pixel_patches_the_first_token_is_the_top_left_square(self):
        image = np.arange(16, dtype=float).reshape(4, 4, 1)
        assert cr.patchify(image, patch=2)[0].tolist() == [0, 1, 4, 5]


class TestRecurrentNetwork:
    # "not very good" with each word as one number (-1, 0.5, 1), recurrent weight 0.5, input weight 1.
    SENTENCE = [[-1.0], [0.5], [1.0]]

    def test_given_not_very_good_the_summary_after_each_word_is_minus_0_762_then_0_119_then_0_785(self):
        # h1 = tanh(0.5·0 + (-1)) = -0.762; h2 = tanh(0.5·(-0.762) + 0.5) = 0.119; h3 = tanh(0.5·0.119 + 1) = 0.785.
        states = cr.RNNCell.scalar(w_h=0.5, w_x=1.0).run(self.SENTENCE)
        assert [round(h[0], 3) for h in states] == [-0.762, 0.119, 0.785]

    def test_given_not_very_good_the_final_summary_has_nearly_forgotten_the_opening_not(self):
        # The negative start is washed out: the one-page summary is rewritten after every word.
        assert cr.RNNCell.scalar(w_h=0.5, w_x=1.0).run(self.SENTENCE)[-1][0] > 0.7

    def test_given_recurrent_weight_0_5_a_word_10_steps_back_has_influence_0_5_to_the_10th(self):
        # With zero inputs the state stays at 0, so each step multiplies the gradient by exactly 0.5.
        assert cr.RNNCell.scalar(w_h=0.5, w_x=1.0).influence_of_start([[0.0]] * 10) == pytest.approx(0.5**10)

    def test_given_recurrent_weight_1_5_the_influence_explodes_to_57_7_after_10_steps(self):
        # 1.5^10 = 57.7: the same multiplication, growing instead of shrinking.
        assert cr.RNNCell.scalar(w_h=1.5, w_x=1.0).influence_of_start([[0.0]] * 10) == pytest.approx(57.67, abs=0.01)


class TestLSTMGates:
    def test_given_the_eraser_off_and_the_pen_off_the_notebook_keeps_its_content_for_100_steps(self):
        cell = cr.LSTMCell.fixed_gates(size=1, forget=1.0, input=0.0, output=1.0, candidate=0.0)
        _, c = cell.run([[0.0]] * 100, c0=np.array([0.8]))
        assert c[0] == pytest.approx(0.8, abs=1e-6)

    def test_given_the_eraser_fully_on_the_notebook_is_wiped_in_one_step(self):
        cell = cr.LSTMCell.fixed_gates(size=1, forget=0.0, input=0.0, output=1.0, candidate=0.0)
        _, c = cell.run([[0.0]], c0=np.array([0.8]))
        assert c[0] == pytest.approx(0.0, abs=1e-6)

    def test_given_the_pen_writing_0_5_on_a_wiped_page_the_cell_holds_0_5_and_shows_tanh_0_5(self):
        # c = 0·old + 1·0.5 = 0.5; h = 1·tanh(0.5) = 0.462.
        cell = cr.LSTMCell.fixed_gates(size=1, forget=0.0, input=1.0, output=1.0, candidate=0.5)
        h, c = cell.run([[0.0]], c0=np.array([0.8]))
        assert (round(c[0], 3), round(h[0], 3)) == (0.5, 0.462)

    def test_given_the_highlighter_off_nothing_in_the_notebook_is_shown(self):
        cell = cr.LSTMCell.fixed_gates(size=1, forget=1.0, input=0.0, output=0.0, candidate=0.0)
        h, c = cell.run([[0.0]], c0=np.array([0.8]))
        assert (round(h[0], 6), round(c[0], 3)) == (0.0, 0.8)

    def test_given_50_steps_an_lstm_memory_carries_over_100_times_more_gradient_than_a_plain_rnn(self):
        # The cell state's additive path (c = f·c + i·g) avoids the repeated squashing that starves plain RNNs.
        rnn, lstm = cr.gradient_through_time(steps=50)
        assert lstm[-1] > 100 * rnn[-1]


class TestGRU:
    def test_given_the_update_gate_fully_on_the_hidden_state_is_carried_unchanged(self):
        cell = cr.GRUCell.fixed_gates(size=1, update=1.0, reset=1.0, candidate=0.3)
        assert cell.run([[0.0]] * 20, h0=np.array([0.6]))[0] == pytest.approx(0.6, abs=1e-6)

    def test_given_the_update_gate_off_the_hidden_state_becomes_the_candidate(self):
        cell = cr.GRUCell.fixed_gates(size=1, update=0.0, reset=1.0, candidate=0.3)
        assert cell.run([[0.0]], h0=np.array([0.6]))[0] == pytest.approx(0.3, abs=1e-6)


class TestWhyTransformersWon:
    def test_given_1000_tokens_an_rnn_needs_1000_steps_one_after_another(self):
        assert cr.sequential_steps("rnn", 1000) == 1000

    def test_given_1000_tokens_a_transformer_layer_processes_them_all_in_one_parallel_step(self):
        assert cr.sequential_steps("transformer", 1000) == 1

    def test_given_1000_tokens_information_from_the_first_reaches_the_last_in_999_rnn_hops_but_1_attention_hop(self):
        assert (cr.path_length("rnn", 1000), cr.path_length("transformer", 1000)) == (999, 1)
