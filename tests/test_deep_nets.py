"""Specification for primer.ml.deep_nets: why deep stacks fail to train, and the fixes that made them work."""

import numpy as np
import pytest

from primer.ml import deep_nets as dn


class TestGradientsThroughAChainOfUnits:
    # A chain of one-number "layers": the gradient is the product of every layer's slope.
    def test_given_ten_sigmoid_units_the_gradient_shrinks_to_a_quarter_to_the_tenth(self):
        # Sigmoid's slope at 0 is 0.25; 0.25^10 = 9.54e-7.
        assert dn.chain_gradient(depth=10, activation="sigmoid", weight=1.0) == pytest.approx(9.537e-7, rel=1e-3)

    def test_given_ten_linear_units_with_weight_1_5_the_gradient_explodes_to_57_7(self):
        # 1.5^10 = 57.665.
        assert dn.chain_gradient(depth=10, activation="linear", weight=1.5) == pytest.approx(57.665, rel=1e-3)

    def test_given_ten_linear_units_with_weight_1_the_gradient_passes_through_unchanged(self):
        assert dn.chain_gradient(depth=10, activation="linear", weight=1.0) == pytest.approx(1.0)


class TestGradientsThroughADeepNetwork:
    def test_given_relu_and_he_initialisation_the_gradient_keeps_its_size_across_30_layers(self):
        ratio = dn.first_to_last_gradient_ratio(depth=30, activation="relu", init="he")
        assert 0.1 < ratio < 10

    def test_given_relu_and_weights_that_are_too_small_the_gradient_vanishes(self):
        ratio = dn.first_to_last_gradient_ratio(depth=30, activation="relu", init="tiny")
        assert ratio < 1e-10

    def test_given_relu_and_weights_that_are_too_large_the_gradient_explodes(self):
        ratio = dn.first_to_last_gradient_ratio(depth=30, activation="relu", init="large")
        assert ratio > 1e10

    def test_given_sigmoid_the_gradient_vanishes_even_with_xavier_initialisation(self):
        ratio = dn.first_to_last_gradient_ratio(depth=30, activation="sigmoid", init="xavier")
        assert ratio < 1e-6

    def test_given_sigmoid_with_xavier_the_gradient_shrinks_about_18_orders_of_magnitude_over_30_layers(self):
        # Xavier makes σ_w·√64 ≈ 1, so each layer multiplies by sigmoid's slope, at most 0.25: 0.25^30 ≈ 8.7e-19.
        # That is half the ≈ 36 orders lost with too-small weights, not "almost as fast".
        ratio = dn.first_to_last_gradient_ratio(depth=30, activation="sigmoid", init="xavier")
        assert 1e-19 < ratio < 1e-17


class TestInitialisation:
    def test_xavier_scale_for_100_inputs_and_100_outputs_is_0_1(self):
        # √(2 / (100 + 100)) = 0.1.
        assert dn.init_std("xavier", fan_in=100, fan_out=100) == pytest.approx(0.1)

    def test_he_scale_for_50_inputs_is_0_2(self):
        # √(2 / 50) = 0.2: twice Xavier's variance, because ReLU zeroes half its inputs.
        assert dn.init_std("he", fan_in=50, fan_out=50) == pytest.approx(0.2)

    def test_given_he_initialisation_the_signal_keeps_its_size_through_30_relu_layers(self):
        rms = dn.forward_signal_rms(depth=30, activation="relu", init="he")
        assert 0.5 < rms[-1] / rms[0] < 2.0

    def test_given_sigmoid_with_xavier_the_forward_signal_holds_near_0_5_while_its_gradient_vanishes(self):
        # Sigmoid's outputs sit around 0.5 whatever comes in, so the forward signal cannot reveal the
        # vanishing gradient: the backward pass multiplies by sigmoid's slope, the forward pass does not.
        rms = dn.forward_signal_rms(depth=30, activation="sigmoid", init="xavier")
        assert np.all((rms[1:] > 0.45) & (rms[1:] < 0.6))


class TestResidualConnections:
    def test_given_ten_plain_blocks_of_slope_0_025_the_gradient_is_about_1e_minus_16(self):
        # 0.025^10 = 9.54e-17.
        assert dn.residual_chain_gradient(depth=10, block_slope=0.025, residual=False) == pytest.approx(9.537e-17, rel=1e-3)

    def test_given_the_same_blocks_with_skip_connections_the_gradient_is_1_28(self):
        # Each block contributes 1 + 0.025; 1.025^10 = 1.2801.
        assert dn.residual_chain_gradient(depth=10, block_slope=0.025, residual=True) == pytest.approx(1.2801, rel=1e-3)

    def test_given_skip_connections_a_30_layer_sigmoid_network_stops_vanishing(self):
        plain = dn.first_to_last_gradient_ratio(depth=30, activation="sigmoid", init="xavier")
        res = dn.first_to_last_gradient_ratio(depth=30, activation="sigmoid", init="xavier", residual=True)
        assert plain < 1e-6 and res > 0.1


class TestBatchNorm:
    batch = np.array([[1.0, 2.0], [3.0, 6.0]])

    def test_each_feature_is_rescaled_to_mean_0_and_spread_1_across_the_batch(self):
        # Feature means (2, 4), standard deviations (1, 2).
        assert dn.batch_norm(self.batch) == pytest.approx(np.array([[-1.0, -1.0], [1.0, 1.0]]), abs=1e-4)

    def test_the_same_example_normalises_differently_in_a_different_batch(self):
        # BatchNorm's output depends on the other examples, a problem for variable-length sequences.
        # With 1, 3, 5: mean 3, std √(8/3) = 1.633, so 1 → −1.2247. With 1, 3, 11: mean 5, std 4.320, so 1 → −0.9258.
        with_5 = dn.batch_norm(np.array([[1.0], [3.0], [5.0]]))
        with_11 = dn.batch_norm(np.array([[1.0], [3.0], [11.0]]))
        assert with_5[0, 0] == pytest.approx(-1.2247, abs=1e-3)
        assert with_11[0, 0] == pytest.approx(-0.9258, abs=1e-3)


class TestLayerNorm:
    def test_the_row_1_2_3_4_becomes_mean_0_spread_1(self):
        # Mean 2.5, standard deviation √1.25 = 1.118.
        out = dn.layer_norm(np.array([[1.0, 2.0, 3.0, 4.0]]))
        assert out[0] == pytest.approx([-1.3416, -0.4472, 0.4472, 1.3416], abs=1e-3)

    def test_an_example_normalises_the_same_whatever_else_is_in_the_batch(self):
        row = np.array([[1.0, 2.0, 3.0, 4.0]])
        batch = np.vstack([row, [[100.0, -5.0, 0.0, 7.0]]])
        assert dn.layer_norm(batch)[0] == pytest.approx(dn.layer_norm(row)[0])


class TestRMSNorm:
    def test_the_row_1_2_3_4_is_divided_by_its_root_mean_square_2_739(self):
        # √((1 + 4 + 9 + 16) / 4) = √7.5 = 2.7386; no mean is subtracted.
        out = dn.rms_norm(np.array([[1.0, 2.0, 3.0, 4.0]]))
        assert out[0] == pytest.approx([0.3651, 0.7303, 1.0954, 1.4606], abs=1e-3)


class TestClippingAnExplodingGradient:
    def test_given_an_exploding_network_clipping_caps_the_update_length_at_the_limit(self):
        grads = dn.layer_gradients(depth=30, activation="relu", init="large")
        clipped_norm = dn.global_norm_after_clipping(grads, max_norm=1.0)
        assert clipped_norm == pytest.approx(1.0)


class TestTheGradientFlowFigure:
    def test_given_the_gradient_flow_figure_every_line_it_describes_is_inside_the_visible_axis(self):
        # A line clipped off the plot can't show the dive or climb the text asks the reader to see.
        pytest.importorskip("matplotlib")
        ax = dn.figures()["gradient_flow"].axes[0]
        low, high = ax.get_ylim()
        heights = np.concatenate([line.get_ydata() for line in ax.get_lines()])
        assert low <= heights.min() and heights.max() <= high
