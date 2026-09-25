"""Specification for primer.ml.embeddings.compression: making vectors smaller without losing the neighbours."""

import numpy as np
import pytest

from primer.ml.embeddings.compression import (
    binary_quantize,
    dequantize_int8,
    hamming_distances,
    hnsw_link_bytes,
    make_corpus,
    matryoshka_order,
    random_order,
    recall_at_k,
    scalar_quantize_int8,
    search_binary,
    search_binary_then_rescore,
    search_int8,
    search_truncated,
    search_truncated_then_rescore,
    storage_bytes,
)


@pytest.fixture(scope="module")
def corpus():
    return make_corpus()


class TestStorageMath:
    def test_given_ten_million_1536_dim_float32_vectors_raw_storage_is_61_gigabytes(self):
        # 10,000,000 × 1,536 × 4 bytes = 61,440,000,000 bytes.
        assert storage_bytes(10_000_000, 1536, bits_per_value=32) == 61_440_000_000

    def test_given_int8_the_same_vectors_take_a_quarter_of_the_space(self):
        assert storage_bytes(10_000_000, 1536, bits_per_value=8) == 15_360_000_000

    def test_given_one_bit_per_value_the_same_vectors_take_a_thirty_second_of_the_space(self):
        # 1,536 bits = 192 bytes per vector.
        assert storage_bytes(10_000_000, 1536, bits_per_value=1) == 1_920_000_000

    def test_given_an_hnsw_graph_with_m_16_its_bottom_layer_links_add_1_28_gigabytes(self):
        # The bottom layer keeps 2·M = 32 neighbour ids per vector, 4 bytes each: 10M × 32 × 4.
        assert hnsw_link_bytes(10_000_000, M=16) == 1_280_000_000


class TestRecallAtK:
    def test_given_two_of_the_three_true_neighbours_found_recall_at_3_is_two_thirds(self):
        assert recall_at_k(np.array([[3, 4, 1]]), np.array([[1, 2, 3]])) == pytest.approx(2 / 3)


class TestMatryoshkaTruncation:
    def test_given_vectors_ordered_by_importance_the_first_eighth_of_dimensions_keeps_nine_in_ten_neighbours(self, corpus):
        docs, queries, truth = corpus
        assert search_truncated(matryoshka_order(docs), matryoshka_order(docs, queries), truth, dims=32) >= 0.9

    def test_given_vectors_without_that_order_the_first_eighth_of_dimensions_loses_over_half_the_neighbours(self, corpus):
        # Without an importance order every dimension carries a similar share, so cutting 7/8 of them cuts deep.
        docs, queries, truth = corpus
        assert search_truncated(random_order(docs), random_order(docs, queries), truth, dims=32) < 0.5

    def test_given_a_short_vector_shortlist_rescored_with_full_vectors_nearly_every_neighbour_is_recovered(self, corpus):
        docs, queries, truth = corpus
        d, q = matryoshka_order(docs), matryoshka_order(docs, queries)
        assert search_truncated_then_rescore(d, q, truth, dims=32, shortlist=100) >= 0.95


class TestScalarQuantization:
    def test_given_values_minus_one_zero_and_one_in_the_range_minus_one_to_one_the_codes_are_0_128_and_255(self):
        # 256 levels across a width of 2: code = round((x + 1) / 2 × 255), and 127.5 rounds to 128.
        codes = scalar_quantize_int8(np.array([[-1.0, 0.0, 1.0]]), lo=np.array([-1.0] * 3), hi=np.array([1.0] * 3))
        assert codes.tolist() == [[0, 128, 255]]

    def test_given_int8_codes_every_value_comes_back_within_half_a_step(self):
        # One step is 2/255, so rounding costs at most half a step: exactly 1/255 = 0.0039216.
        x = np.linspace(-1, 1, 101)[None, :]
        lo, hi = np.full(101, -1.0), np.full(101, 1.0)
        back = dequantize_int8(scalar_quantize_int8(x, lo, hi), lo, hi)
        assert np.max(np.abs(back - x)) <= 1 / 255 + 1e-12

    def test_given_int8_vectors_search_keeps_nearly_every_true_neighbour(self, corpus):
        docs, queries, truth = corpus
        assert search_int8(docs, queries, truth) >= 0.9


class TestBinaryQuantization:
    def test_given_eight_values_their_signs_pack_into_one_byte(self):
        # Positive -> 1, otherwise 0: 1 0 0 1 0 1 0 1 = 0b10010101 = 149.
        packed = binary_quantize(np.array([[0.3, -0.2, 0.0, 5.0, -1.0, 2.0, -3.0, 0.1]]))
        assert packed.tolist() == [[149]]

    def test_given_two_bytes_the_hamming_distance_counts_the_bits_that_differ(self):
        # 10010101 vs 00010100: they differ in the first and last bit.
        assert hamming_distances(np.array([[0b00010100]], dtype=np.uint8), np.array([0b10010101], dtype=np.uint8)).tolist() == [2]

    def test_given_binary_codes_alone_search_loses_more_neighbours_than_int8(self, corpus):
        docs, queries, truth = corpus
        assert search_binary(docs, queries, truth) < search_int8(docs, queries, truth)

    def test_given_a_binary_shortlist_rescored_with_full_vectors_nearly_every_neighbour_is_recovered(self, corpus):
        docs, queries, truth = corpus
        assert search_binary_then_rescore(docs, queries, truth, shortlist=100) >= 0.95
