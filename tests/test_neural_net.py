"""Specification for primer.ml.neural_net: neurons, activations, backprop, the training loop."""

import numpy as np
import pytest

from primer.ml import neural_net as nn


class TestOneNeuronWorkedExample:
    def test_given_inputs_2_and_3_weights_half_and_1_and_bias_half_the_weighted_sum_is_4_5(self):
        assert nn.one_neuron_worked_example()["z"] == pytest.approx(4.5)

    def test_given_a_positive_sum_relu_passes_it_through_unchanged(self):
        assert nn.one_neuron_worked_example()["y"] == pytest.approx(4.5)

    def test_given_a_target_of_5_the_squared_error_is_0_25(self):
        assert nn.one_neuron_worked_example()["loss"] == pytest.approx(0.25)

    def test_the_weight_on_the_larger_input_receives_the_larger_gradient(self):
        # Hand calculation: dL/dy = 2(4.5 - 5) = -1, times inputs (2, 3).
        assert nn.one_neuron_worked_example()["dL_dw"] == pytest.approx([-2.0, -3.0])

    def test_one_sgd_step_raises_both_weights_the_second_one_more(self):
        # 0.5 + 0.01*2 = 0.52 and 1.0 + 0.01*3 = 1.03.
        assert nn.one_neuron_worked_example(lr=0.01)["w_new"] == pytest.approx([0.52, 1.03])

    def test_after_one_step_the_output_moves_toward_the_target_and_the_loss_drops(self):
        # New output 2*0.52 + 3*1.03 + 0.51 = 4.64; loss (5 - 4.64)^2 = 0.1296.
        r = nn.one_neuron_worked_example(lr=0.01)
        assert r["y_new"] == pytest.approx(4.64)
        assert r["loss_new"] == pytest.approx(0.1296)


class TestActivationFunctions:
    def test_relu_zeroes_negative_inputs(self):
        assert nn.relu(np.array([-2.0, 3.0])) == pytest.approx([0.0, 3.0])

    def test_given_a_negative_input_relu_passes_no_gradient_so_the_neuron_can_die(self):
        assert nn.relu_grad(np.array([-5.0])) == pytest.approx([0.0])

    def test_sigmoid_of_zero_is_one_half(self):
        assert nn.sigmoid(np.array([0.0])) == pytest.approx([0.5])

    def test_the_sigmoid_gradient_never_exceeds_one_quarter(self):
        z = np.linspace(-10, 10, 2001)
        assert nn.sigmoid_grad(z).max() == pytest.approx(0.25)

    def test_a_saturated_sigmoid_passes_almost_no_gradient(self):
        # σ'(10) = σ(10)(1 - σ(10)) ≈ 4.5e-5.
        assert nn.sigmoid_grad(np.array([10.0]))[0] < 1e-4

    def test_sigmoid_of_a_huge_negative_input_is_zero_without_overflow(self):
        with np.errstate(over="raise"):
            assert nn.sigmoid(np.array([-1000.0])) == pytest.approx([0.0])

    def test_the_tanh_gradient_at_zero_is_one(self):
        assert nn.tanh_grad(np.array([0.0])) == pytest.approx([1.0])

    def test_gelu_of_one_is_one_times_the_normal_cdf_at_one(self):
        # Φ(1) = 0.8413 from a standard normal table.
        assert nn.gelu(np.array([1.0])) == pytest.approx([0.8413], abs=1e-4)

    def test_gelu_leaks_a_small_negative_value_where_relu_gives_zero(self):
        # -1 * Φ(-1) = -0.1587.
        assert nn.gelu(np.array([-1.0])) == pytest.approx([-0.1587], abs=1e-4)

    def test_the_tanh_approximation_of_gelu_agrees_to_three_decimals(self):
        z = np.linspace(-4, 4, 81)
        assert nn.gelu_tanh(z) == pytest.approx(nn.gelu(z), abs=1e-3)

    def test_softmax_of_zero_and_ln3_gives_one_quarter_and_three_quarters(self):
        assert nn.softmax(np.array([0.0, np.log(3.0)])) == pytest.approx([0.25, 0.75])


class TestWhyNonlinearityMatters:
    def test_a_stack_of_linear_layers_equals_a_single_linear_layer(self):
        assert nn.linear_stack_collapses(n_layers=5) < 1e-12

    def test_a_linear_classifier_cannot_separate_the_two_moons(self):
        X, y = nn.make_moons()
        assert nn.train_logistic_regression(X, y) < 0.9

    def test_one_hidden_tanh_layer_separates_the_two_moons(self):
        X, y = nn.make_moons()
        model = nn.MLP(2, 16)
        nn.train(model, X, y, epochs=300)
        assert model.accuracy(X, y) > 0.95

    def test_a_hidden_layer_learns_xor(self):
        # XOR is the textbook function no single line can separate.
        X, y = nn.make_xor()
        model = nn.MLP(2, 4, seed=3)
        nn.train(model, X, y, epochs=2000, batch_size=4, lr=1.0)
        assert model.accuracy(X, y) == 1.0


class TestSingleWeightLearning:
    def test_given_one_weight_learning_y_equals_3x_each_step_halves_the_gap(self):
        # w starts at 0; lr 0.25 on loss (w·1 − 3)²: steps go 0 → 1.5 → 2.25 → 2.625.
        assert nn.learn_one_weight(steps=3) == pytest.approx([0.0, 1.5, 2.25, 2.625])


class TestTinyTwoLayerBackprop:
    # Network 1 → 1 → 1: x = 1, w1 = 0.5, tanh, w2 = 2, sigmoid, target 1. Values from a calculator.
    def test_the_forward_pass_predicts_0_716(self):
        assert nn.tiny_two_layer_example()["y_hat"] == pytest.approx(0.7159, abs=1e-4)

    def test_the_error_at_the_output_is_prediction_minus_target(self):
        assert nn.tiny_two_layer_example()["dL_dz2"] == pytest.approx(-0.2841, abs=1e-4)

    def test_the_output_weight_gradient_is_the_hidden_activation_times_the_error(self):
        # h = tanh(0.5) = 0.4621; 0.4621 × −0.2841 = −0.1313.
        assert nn.tiny_two_layer_example()["dL_dw2"] == pytest.approx(-0.1313, abs=1e-4)

    def test_the_blame_reaching_the_first_weight_is_shrunk_by_the_tanh_slope(self):
        # dh = −0.2841 × 2 = −0.5682; tanh slope 1 − 0.4621² = 0.7864; × x = 1 → −0.4469.
        assert nn.tiny_two_layer_example()["dL_dw1"] == pytest.approx(-0.4469, abs=1e-4)


class TestBackpropagation:
    def test_hand_written_gradients_match_finite_differences(self):
        X, y = nn.make_moons(n=20)
        assert nn.gradient_check(nn.MLP(2, 5, seed=1), X, y) < 1e-6

    def test_hand_written_gradients_match_pytorch_autograd(self):
        torch = pytest.importorskip("torch")
        X, y = nn.make_moons(n=16)
        model = nn.MLP(2, 5, seed=2)
        _, cache = model.forward(X)
        ours = model.backward(cache, y)

        t = {k: torch.tensor(v, requires_grad=True) for k, v in model.params.items()}
        h = torch.tanh(torch.tensor(X) @ t["W1"] + t["b1"])
        p = torch.sigmoid(h @ t["W2"] + t["b2"])[:, 0]
        torch.nn.functional.binary_cross_entropy(p, torch.tensor(y)).backward()
        for k in ours:
            assert ours[k] == pytest.approx(t[k].grad.numpy(), abs=1e-10)


class TestTrainingLoop:
    def test_given_400_examples_and_batches_of_32_an_epoch_has_13_steps(self):
        X, y = nn.make_moons(n=400)
        out = nn.train(nn.MLP(2, 4), X, y, epochs=1, batch_size=32)
        assert out["steps_per_epoch"] == 13

    def test_two_epochs_take_twice_the_steps_of_one(self):
        X, y = nn.make_moons(n=400)
        out = nn.train(nn.MLP(2, 4), X, y, epochs=2, batch_size=32)
        assert out["steps"] == 26

    def test_training_lowers_the_loss(self):
        X, y = nn.make_moons()
        hist = nn.train(nn.MLP(2, 16), X, y, epochs=50)["history"]
        assert hist[-1] < hist[0] / 2
