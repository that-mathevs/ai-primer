"""Specification for primer.ml.optimizers: how weights take their steps downhill."""

import numpy as np
import pytest

from primer.ml import optimizers as opt


class TestGradientDescentOnABowl:
    # The bowl f(w) = w² has slope 2w, so every value below is checkable by hand.
    def test_given_a_learning_rate_of_0_1_each_step_keeps_80_percent_of_w(self):
        assert opt.descend_bowl(w0=1.0, lr=0.1, steps=2) == pytest.approx([1.0, 0.8, 0.64])

    def test_given_a_learning_rate_of_0_5_one_step_lands_exactly_at_the_bottom(self):
        assert opt.descend_bowl(w0=1.0, lr=0.5, steps=1)[-1] == pytest.approx(0.0)

    def test_given_a_learning_rate_above_1_each_step_overshoots_further_and_training_diverges(self):
        # lr 1.1: w ← w − 1.1·2w = −1.2w, so |w| grows 20% per step.
        path = opt.descend_bowl(w0=1.0, lr=1.1, steps=3)
        assert path == pytest.approx([1.0, -1.2, 1.44, -1.728])

    def test_given_a_tiny_learning_rate_ten_steps_barely_move_the_weight(self):
        # (1 − 0.002)^10 = 0.980.
        assert opt.descend_bowl(w0=1.0, lr=0.001, steps=10)[-1] == pytest.approx(0.980, abs=1e-3)


class TestMomentum:
    def test_given_beta_0_9_the_velocity_accumulates_past_gradients(self):
        # Step 1: g = 2, v = 2, w = 0.8. Step 2: g = 1.6, v = 0.9·2 + 1.6 = 3.4, w = 0.8 − 0.34 = 0.46.
        assert opt.momentum_on_bowl(w0=1.0, lr=0.1, beta=0.9, steps=2) == pytest.approx([1.0, 0.8, 0.46])

    def test_in_a_long_narrow_valley_momentum_gets_far_lower_than_plain_sgd_in_the_same_steps(self):
        sgd = opt.run(opt.SGD(lr=0.015), opt.narrow_valley, np.array([-8.0, 1.0]), steps=100)
        mom = opt.run(opt.SGD(lr=0.015, momentum=0.9), opt.narrow_valley, np.array([-8.0, 1.0]), steps=100)
        assert mom["loss"][-1] < sgd["loss"][-1] / 100


class TestAdam:
    @pytest.mark.parametrize("gradient", [1000.0, 1.0, 0.001])
    def test_given_any_gradient_size_the_first_step_is_the_learning_rate(self, gradient):
        # Bias-corrected m̂ = g and v̂ = g², so the step is lr·g/|g| = lr.
        step = opt.adam_first_step(gradient, lr=0.01)
        assert step == pytest.approx(0.01, rel=1e-4)

    def test_in_a_long_narrow_valley_adam_reaches_the_bottom_where_sgd_stalls(self):
        sgd = opt.run(opt.SGD(lr=0.015), opt.narrow_valley, np.array([-8.0, 1.0]), steps=300)
        adam = opt.run(opt.Adam(lr=0.1), opt.narrow_valley, np.array([-8.0, 1.0]), steps=300)
        assert adam["loss"][-1] < 1e-3 < sgd["loss"][-1]


class TestWeightDecay:
    def test_given_no_gradient_adamw_shrinks_the_weight_by_lr_times_decay(self):
        # 1 − 0.1 × 0.1 = 0.99.
        assert opt.one_decay_step(w=1.0, lr=0.1, decay=0.1, kind="adamw") == pytest.approx(0.99)

    def test_given_l2_inside_adam_the_penalty_is_normalised_into_a_full_learning_rate_step(self):
        # g = λw = 0.1; Adam's first step is lr·g/|g| = 0.1 whatever λ is.
        assert opt.one_decay_step(w=1.0, lr=0.1, decay=0.1, kind="adam_l2") == pytest.approx(0.9, abs=1e-6)

    def test_given_l2_inside_adam_a_hundredfold_smaller_penalty_still_shrinks_the_weight_as_much(self):
        # The reason AdamW exists: with Adam, L2 strength stops meaning what it says.
        assert opt.one_decay_step(w=1.0, lr=0.1, decay=0.001, kind="adam_l2") == pytest.approx(0.9, abs=1e-6)

    def test_given_adamw_a_hundredfold_smaller_decay_shrinks_the_weight_a_hundredfold_less(self):
        assert opt.one_decay_step(w=1.0, lr=0.1, decay=0.001, kind="adamw") == pytest.approx(0.9999)


class TestWarmupCosineSchedule:
    schedule = dict(peak=1e-3, warmup=100, total=1000)

    def test_halfway_through_warmup_the_rate_is_half_the_peak(self):
        assert opt.warmup_cosine(50, **self.schedule) == pytest.approx(5e-4)

    def test_at_the_end_of_warmup_the_rate_is_the_peak(self):
        assert opt.warmup_cosine(100, **self.schedule) == pytest.approx(1e-3)

    def test_halfway_through_the_decay_the_cosine_has_fallen_to_half_the_peak(self):
        # Step 550 is halfway from 100 to 1000; ½(1 + cos(π/2)) = ½.
        assert opt.warmup_cosine(550, **self.schedule) == pytest.approx(5e-4)

    def test_at_the_final_step_the_rate_reaches_the_floor(self):
        assert opt.warmup_cosine(1000, **self.schedule, floor=1e-5) == pytest.approx(1e-5)


class TestGradientClipping:
    def test_given_a_gradient_of_length_5_and_a_limit_of_1_it_is_scaled_to_length_1(self):
        # (3, 4) has length 5; scaled by 1/5 it becomes (0.6, 0.8).
        clipped, norm = opt.clip_by_global_norm([np.array([3.0, 4.0])], max_norm=1.0)
        assert norm == pytest.approx(5.0)
        assert clipped[0] == pytest.approx([0.6, 0.8])

    def test_given_a_gradient_already_under_the_limit_it_is_unchanged(self):
        clipped, _ = opt.clip_by_global_norm([np.array([0.3, 0.4])], max_norm=1.0)
        assert clipped[0] == pytest.approx([0.3, 0.4])

    def test_the_length_is_measured_across_all_tensors_together_so_direction_is_kept(self):
        # Pieces (3) and (4) form one gradient of length 5; both shrink by the same factor.
        clipped, norm = opt.clip_by_global_norm([np.array([3.0]), np.array([4.0])], max_norm=1.0)
        assert norm == pytest.approx(5.0)
        assert [c[0] for c in clipped] == pytest.approx([0.6, 0.8])
