"""Specification for primer.ml.hardware: the chips, memories, links and number formats underneath a model."""

import math

import numpy as np
import pytest

from primer.ml.hardware import (
    BETWEEN_MACHINES_LINK,
    BF16,
    FP8_E4M3,
    FP8_E5M2,
    FP16,
    FP32,
    IN_MACHINE_LINK,
    LLAMA3_70B_SHAPE,
    MEMORY_HIERARCHY,
    all_reduce_seconds,
    bit_string,
    counted_matmul,
    decode,
    encode,
    matmul_flops,
    matmul_intensity,
    matmul_reads,
    max_cache_tokens,
    multiplier_cells,
    parallel_rounds,
    ring_all_reduce,
    round_to,
    tiled_matmul,
    training_step_seconds,
    transfer_seconds,
)


class TestCountingTheWorkInAMatrixMultiply:
    def test_given_a_2x3_times_3x2_multiply_it_takes_12_multiplies_and_12_adds(self):
        # Each of the 2·2 outputs is a dot product of length 3: three multiplies, three adds into the running total.
        _, multiplies, adds = counted_matmul([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]])
        assert (multiplies, adds) == (12, 12)

    def test_given_the_counted_multiply_its_answer_matches_the_product_worked_by_hand(self):
        C, _, _ = counted_matmul([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]])
        assert C == [[58, 64], [139, 154]]

    def test_given_two_4096_square_matrices_the_multiply_costs_2_m_n_k_flops(self):
        assert matmul_flops(4096, 4096, 4096) == 137_438_953_472


class TestSplittingAMultiplyAcrossCores:
    def test_given_one_core_a_64_cube_multiply_takes_one_round_per_multiply_add(self):
        assert parallel_rounds(64, 64, 64, cores=1) == 262_144

    def test_given_as_many_cores_as_outputs_only_one_dot_product_of_rounds_remains(self):
        # Every output cell is independent work, so 4,096 cores each finish one 64-step dot product.
        assert parallel_rounds(64, 64, 64, cores=4096) == 64

    def test_given_more_cores_than_outputs_the_extra_cores_sit_idle(self):
        assert parallel_rounds(64, 64, 64, cores=8192) == 64


class TestTheMemoryHierarchy:
    def test_given_the_levels_from_registers_outward_each_holds_more_than_the_one_before(self):
        capacities = [level.capacity_bytes for level in MEMORY_HIERARCHY]
        assert capacities == sorted(capacities) and len(set(capacities)) == len(capacities)

    def test_given_the_levels_from_registers_to_host_memory_each_is_slower_than_the_one_before(self):
        on_the_way_out = [level.bandwidth_bytes_per_s for level in MEMORY_HIERARCHY[:4]]
        assert on_the_way_out == sorted(on_the_way_out, reverse=True)

    def test_given_one_gigabyte_reading_it_from_hbm_takes_about_0_3_milliseconds(self):
        assert transfer_seconds(1e9, "HBM") * 1e3 == pytest.approx(0.30, abs=0.005)

    def test_given_one_gigabyte_fetching_it_from_host_memory_takes_67_times_longer_than_from_hbm(self):
        assert transfer_seconds(1e9, "host memory") / transfer_seconds(1e9, "HBM") == pytest.approx(67, abs=0.5)


class TestTilingAMatrixMultiply:
    @pytest.mark.parametrize("tile", [1, 2, 4, 8])
    def test_given_any_tile_size_the_tiled_product_equals_the_ordinary_product(self, tile):
        rng = np.random.default_rng(0)
        A, B = rng.standard_normal((8, 8)), rng.standard_normal((8, 8))
        C, _ = tiled_matmul(A, B, tile)
        assert np.allclose(C, A @ B)

    def test_given_4x4_matrices_and_no_reuse_128_values_are_read_from_slow_memory(self):
        # Each of 16 outputs reads a row of A and a column of B: 16 · (4 + 4) = 128.
        _, traffic = tiled_matmul(np.ones((4, 4)), np.ones((4, 4)), tile=1)
        assert traffic.reads == 128

    def test_given_4x4_matrices_and_2x2_tiles_the_reads_halve_to_64(self):
        _, traffic = tiled_matmul(np.ones((4, 4)), np.ones((4, 4)), tile=2)
        assert traffic.reads == 64

    def test_given_a_tile_as_big_as_the_matrix_each_input_is_read_exactly_once(self):
        _, traffic = tiled_matmul(np.ones((4, 4)), np.ones((4, 4)), tile=4)
        assert traffic.reads == 32

    @pytest.mark.parametrize("tile", [1, 2, 4])
    def test_given_any_tile_every_output_is_written_to_slow_memory_exactly_once(self, tile):
        _, traffic = tiled_matmul(np.ones((4, 4)), np.ones((4, 4)), tile)
        assert traffic.writes == 16

    def test_given_tiles_of_t_by_t_fast_memory_holds_three_of_them_at_once(self):
        # One tile of A, one of B, and the tile of C being accumulated.
        _, traffic = tiled_matmul(np.ones((8, 8)), np.ones((8, 8)), tile=4)
        assert traffic.fast_memory_peak == 48

    @pytest.mark.parametrize("tile", [1, 2, 4, 8, 16, 32])
    def test_given_the_simulation_its_counted_reads_match_the_formula_2_n_cubed_over_t(self, tile):
        _, traffic = tiled_matmul(np.ones((32, 32)), np.ones((32, 32)), tile)
        assert traffic.reads == matmul_reads(32, tile) == 2 * 32**3 // tile

    def test_given_4096_square_matrices_in_16_bit_128_tiles_do_about_63_flops_per_byte(self):
        assert matmul_intensity(4096, tile=128, bytes_per_element=2) == pytest.approx(63.02, abs=0.01)

    def test_given_8_bit_numbers_instead_of_16_bit_the_same_tiles_do_twice_the_flops_per_byte(self):
        assert matmul_intensity(4096, 128, 1) == pytest.approx(2 * matmul_intensity(4096, 128, 2))


class TestStoringANumberInAFloatFormat:
    def test_given_bf16_minus_6_5_is_stored_as_sign_1_exponent_129_mantissa_80(self):
        # -6.5 = -1.625 × 2²: exponent 2 + bias 127 = 129; .625 = .101 in binary, padded to 1010000 = 80.
        assert encode(-6.5, BF16) == (1, 129, 80)

    def test_given_the_stored_fields_of_minus_6_5_decoding_them_gives_back_minus_6_5(self):
        assert decode(1, 129, 80, BF16) == -6.5

    def test_given_bf16_the_bits_of_minus_6_5_read_sign_then_exponent_then_mantissa(self):
        assert bit_string(-6.5, BF16) == "1 10000001 1010000"

    def test_given_a_value_below_the_smallest_normal_fp16_stores_it_as_a_subnormal(self):
        # 2⁻²⁰ is below fp16's smallest normal 2⁻¹⁴, so it is 16 steps of the subnormal spacing 2⁻²⁴.
        assert encode(2.0**-20, FP16) == (0, 0, 16)

    @pytest.mark.parametrize(
        "fmt, largest",
        [(FP32, 3.4028234663852886e38), (BF16, 3.3895313892515355e38), (FP16, 65504.0), (FP8_E5M2, 57344.0), (FP8_E4M3, 448.0)],
        ids=lambda v: getattr(v, "name", ""),
    )
    def test_given_each_format_its_largest_finite_value_matches_its_specification(self, fmt, largest):
        assert fmt.max_value == largest

    @pytest.mark.parametrize(
        "fmt, gap",
        [(FP32, 2**-23), (BF16, 2**-7), (FP16, 2**-10), (FP8_E5M2, 2**-2), (FP8_E4M3, 2**-3)],
        ids=lambda v: getattr(v, "name", ""),
    )
    def test_given_each_format_the_gap_just_above_one_is_two_to_the_minus_mantissa_bits(self, fmt, gap):
        assert fmt.epsilon == gap


class TestRoundingAgreesWithRealHardwareFormats:
    def test_given_random_values_of_every_size_rounding_to_fp16_agrees_with_numpy_float16(self):
        rng = np.random.default_rng(1)
        xs = rng.standard_normal(2000) * 10.0 ** rng.uniform(-9, 5, 2000)
        # Some values pass 65504 on purpose: overflowing to infinity is part of what must agree.
        with np.errstate(over="ignore"):
            expected = [float(np.float16(x)) for x in xs]
        assert [round_to(float(x), FP16) for x in xs] == expected

    def test_given_random_values_of_every_size_rounding_to_fp32_agrees_with_numpy_float32(self):
        rng = np.random.default_rng(2)
        xs = rng.standard_normal(2000) * 10.0 ** rng.uniform(-40, 38, 2000)
        assert [round_to(float(x), FP32) for x in xs] == [float(np.float32(x)) for x in xs]

    def test_given_float32_values_rounding_to_bf16_agrees_with_rounding_away_the_low_16_bits(self):
        # bf16 is the top half of a float32, so an integer bit trick on the float32 pattern is an independent check.
        rng = np.random.default_rng(3)
        xs = (rng.standard_normal(2000) * 10.0 ** rng.uniform(-30, 30, 2000)).astype(np.float32)
        bits = xs.view(np.uint32).astype(np.uint64)
        rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16 << 16
        expected = rounded.astype(np.uint32).view(np.float32)
        assert [round_to(float(x), BF16) for x in xs] == [float(e) for e in expected]


class TestRangeAndPrecision:
    def test_given_a_value_past_fp16s_largest_it_overflows_to_infinity(self):
        assert round_to(70_000.0, FP16) == math.inf

    def test_given_fp8_e4m3_a_value_past_448_has_no_encoding(self):
        # E4M3 spends its top exponent on numbers instead of infinity, so there is nothing to overflow into.
        assert math.isnan(round_to(500.0, FP8_E4M3))

    def test_given_a_gradient_of_one_hundred_millionth_fp16_flushes_it_to_zero(self):
        assert round_to(1e-8, FP16) == 0.0

    def test_given_the_same_tiny_gradient_bf16_keeps_it_within_one_percent(self):
        # bf16 has fp32's 8 exponent bits, so its range reaches down to about 1e-38.
        assert round_to(1e-8, BF16) == pytest.approx(1e-8, rel=0.01)

    def test_given_bf16_adding_a_thousandth_to_one_changes_nothing(self):
        # The gap above 1 in bf16 is 1/128, so small weight updates vanish: why training keeps a float32 copy.
        assert round_to(1.0 + 1e-3, BF16) == 1.0

    def test_given_fp32_adding_a_thousandth_to_one_is_kept(self):
        assert round_to(1.0 + 1e-3, FP32) == pytest.approx(1.001, abs=1e-7)


class TestWhySmallerFormatsAreFaster:
    @pytest.mark.parametrize(
        "fmt, cells", [(FP32, 576), (FP16, 121), (BF16, 64), (FP8_E4M3, 16)], ids=lambda v: getattr(v, "name", "")
    )
    def test_given_a_format_its_multiplier_grows_with_the_square_of_its_significand_bits(self, fmt, cells):
        assert multiplier_cells(fmt) == cells


class TestAllReduceAcrossGpus:
    def test_given_four_gpus_ring_all_reduce_leaves_every_gpu_holding_the_sum(self):
        gradients = [np.arange(8.0) * (g + 1) for g in range(4)]
        results, _ = ring_all_reduce(gradients)
        assert all(r.tolist() == (np.arange(8.0) * 10).tolist() for r in results)

    def test_given_n_gpus_each_sends_2_times_n_minus_1_over_n_of_the_data(self):
        # 4 GPUs, 8 numbers each: 2 · 3/4 · 8 = 12 numbers sent per GPU, whatever the number of GPUs.
        _, sent = ring_all_reduce([np.ones(8) for _ in range(4)])
        assert sent == [12, 12, 12, 12]

    def test_given_7b_gradients_in_16_bit_on_8_gpus_in_one_machine_the_all_reduce_takes_about_49_ms(self):
        assert all_reduce_seconds(14e9, 8, IN_MACHINE_LINK) * 1e3 == pytest.approx(49.0, abs=0.1)

    def test_given_the_same_gradients_across_machines_the_all_reduce_takes_ten_times_longer(self):
        ratio = all_reduce_seconds(14e9, 8, BETWEEN_MACHINES_LINK) / all_reduce_seconds(14e9, 8, IN_MACHINE_LINK)
        assert ratio == pytest.approx(10)

    def test_given_7b_parameters_and_16k_tokens_one_training_step_of_arithmetic_takes_about_0_69_s(self):
        # 6 FLOPs per parameter per token (2 forward, 4 backward) at 10¹⁵ FLOP/s.
        assert training_step_seconds(7e9, 16_384) == pytest.approx(0.688, abs=0.001)


class TestWillAModelFitForServing:
    def test_given_a_70b_model_in_16_bit_its_weights_alone_leave_no_room_on_an_80_gb_gpu(self):
        assert max_cache_tokens(80e9, 70e9, weight_bits=16, kv_bits=16, **LLAMA3_70B_SHAPE) == 0

    def test_given_70b_weights_in_fp8_and_a_16_bit_cache_about_30_thousand_tokens_fit(self):
        # 10 GB left over, at 2·80·8·128·2 = 327,680 bytes per token.
        assert max_cache_tokens(80e9, 70e9, weight_bits=8, kv_bits=16, **LLAMA3_70B_SHAPE) == 30_517

    def test_given_the_cache_in_fp8_too_twice_as_many_tokens_fit(self):
        assert max_cache_tokens(80e9, 70e9, weight_bits=8, kv_bits=8, **LLAMA3_70B_SHAPE) == 61_035
