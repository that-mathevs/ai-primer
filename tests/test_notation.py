"""Specification for primer.notation: what each piece of math notation means, as code."""

import math

import numpy as np
import pytest

from primer.notation import (
    argmax,
    derivative,
    dot,
    gradient,
    matmul,
    mean,
    norm,
    product,
    softmax,
    std,
    summation,
    transpose,
    variance,
)


class TestSummation:
    def test_given_one_to_four_sigma_adds_them_to_ten(self):
        assert summation([1, 2, 3, 4]) == 10

    def test_given_an_empty_list_sigma_is_zero(self):
        # Adding nothing leaves you where you started, so an empty sum is 0.
        assert summation([]) == 0


class TestProduct:
    def test_given_one_to_four_capital_pi_multiplies_them_to_twenty_four(self):
        assert product([1, 2, 3, 4]) == 24

    def test_given_ten_steps_at_95_percent_the_chance_all_succeed_is_about_60_percent(self):
        assert product([0.95] * 10) == pytest.approx(0.599, abs=0.001)


class TestDotProduct:
    def test_given_1_2_and_3_half_the_dot_product_is_4(self):
        assert dot([1, 2], [3, 0.5]) == 4

    def test_given_vectors_at_right_angles_the_dot_product_is_zero(self):
        assert dot([1, 0], [0, 5]) == 0

    def test_given_lists_of_different_lengths_the_dot_product_is_refused(self):
        with pytest.raises(ValueError):
            dot([1, 2, 3], [1, 2])


class TestNorm:
    def test_given_a_3_4_triangle_the_length_is_5(self):
        assert norm([3, 4]) == 5

    def test_given_1_2_2_the_length_is_3(self):
        assert norm([1, 2, 2]) == 3


class TestMatrices:
    def test_given_a_2_by_3_matrix_transpose_turns_it_into_3_by_2(self):
        assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]

    def test_given_two_small_matrices_each_output_cell_is_a_row_dotted_with_a_column(self):
        # Row 1 of A · column 1 of B = 1·5 + 2·7 = 19, and so on.
        assert matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]]

    def test_given_inner_sizes_that_disagree_matrix_multiply_is_refused(self):
        with pytest.raises(ValueError):
            matmul([[1, 2, 3]], [[1, 2]])


class TestSoftmaxAndArgmax:
    def test_given_scores_2_1_and_half_softmax_gives_63_23_and_14_percent(self):
        assert softmax([2.0, 1.0, 0.5]) == pytest.approx([0.629, 0.231, 0.140], abs=0.001)

    def test_given_any_scores_the_shares_sum_to_one(self):
        assert sum(softmax([-3.0, 0.0, 12.5, 4.0])) == pytest.approx(1.0)

    def test_given_scores_argmax_names_the_position_of_the_largest_not_its_value(self):
        assert argmax([0.1, 7.0, 3.0]) == 1


class TestSpread:
    def test_given_2_4_4_4_5_5_7_9_the_mean_is_5(self):
        assert mean([2, 4, 4, 4, 5, 5, 7, 9]) == 5

    def test_given_2_4_4_4_5_5_7_9_the_variance_is_4_and_the_standard_deviation_2(self):
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        assert variance(data) == 4
        assert std(data) == 2


class TestDerivatives:
    def test_given_x_squared_at_3_the_slope_is_6(self):
        assert derivative(lambda x: x * x, 3.0) == pytest.approx(6.0, abs=1e-4)

    def test_given_a_bowl_x_squared_plus_y_squared_the_gradient_at_1_2_points_uphill_as_2_4(self):
        assert gradient(lambda v: v[0] ** 2 + v[1] ** 2, [1.0, 2.0]) == pytest.approx([2.0, 4.0], abs=1e-4)


class TestAgreementWithNumPy:
    def test_given_random_matrices_the_hand_written_matmul_matches_numpy(self):
        rng = np.random.default_rng(0)
        A, B = rng.standard_normal((3, 4)), rng.standard_normal((4, 2))
        assert np.allclose(matmul(A.tolist(), B.tolist()), A @ B)

    def test_given_a_vector_the_hand_written_norm_matches_numpy(self):
        v = [0.3, -1.2, 2.5, 4.0]
        assert norm(v) == pytest.approx(float(np.linalg.norm(v)))

    def test_given_e_the_natural_log_undoes_it(self):
        # ln is the inverse of e^x: it answers "e to what power gives this?"
        assert math.log(math.exp(2.5)) == pytest.approx(2.5)
