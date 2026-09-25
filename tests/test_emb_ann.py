"""Specification for primer.ml.embeddings.ann: finding nearest vectors without scanning them all."""

import numpy as np
import pytest

from primer.ml.embeddings.ann import (
    FlatIndex,
    HNSWIndex,
    IVFIndex,
    IVFPQIndex,
    PQIndex,
    clustered_vectors,
    evaluate,
    ground_truth,
    storage_estimate,
)

DIM = 32


@pytest.fixture(scope="module")
def corpus():
    # 1,000 clustered vectors and 40 held-out queries: small enough that every index builds in well under a second.
    X = clustered_vectors(1000, DIM, n_clusters=20, seed=0)
    queries = clustered_vectors(40, DIM, n_clusters=20, seed=1)
    return X, queries, ground_truth(X, queries, k=10)


@pytest.fixture(scope="module")
def hnsw(corpus):
    index = HNSWIndex(DIM, M=8, ef_construction=64, ef_search=50)
    index.add(corpus[0])
    return index


class TestFlatSearch:
    def test_given_three_unit_vectors_the_closest_one_ranks_first(self):
        index = FlatIndex(2)
        index.add(np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]]))
        ids, scores = index.search(np.array([0.8, 0.6]), k=3)
        # Hand calculation: dot products are 0.8, 0.6 and 0.48 + 0.48 = 0.96.
        assert ids.tolist() == [2, 0, 1]
        assert scores == pytest.approx([0.96, 0.8, 0.6])

    def test_when_searching_it_compares_the_query_with_every_stored_vector(self, corpus):
        index = FlatIndex(DIM)
        index.add(corpus[0])
        index.search(corpus[1][0], k=10)
        assert index.ndist == 1000


class TestStorageMath:
    def test_given_ten_million_1536_dim_float32_vectors_raw_storage_is_about_61_gb(self):
        assert storage_estimate(10_000_000, 1536) == pytest.approx(61.44)


class TestIVF:
    def test_given_nprobe_equal_to_nlist_it_returns_exactly_what_flat_search_returns(self, corpus):
        X, queries, truth = corpus
        index = IVFIndex(DIM, nlist=16, nprobe=16)
        index.add(X)
        assert evaluate(index, queries, truth)["recall"] == 1.0

    def test_given_one_probe_it_scans_a_fraction_of_the_corpus(self, corpus):
        X, queries, truth = corpus
        index = IVFIndex(DIM, nlist=16, nprobe=1)
        index.add(X)
        # One of 16 clusters holds roughly 1/16 of the vectors; allow for uneven clusters.
        assert evaluate(index, queries, truth)["ndist"] < 250

    def test_when_nprobe_grows_recall_does_not_fall(self, corpus):
        X, queries, truth = corpus
        index = IVFIndex(DIM, nlist=16, nprobe=1)
        index.add(X)
        low = evaluate(index, queries, truth)["recall"]
        index.nprobe = 8
        assert evaluate(index, queries, truth)["recall"] >= low


class TestProductQuantization:
    def test_given_eight_one_byte_codes_per_vector_a_thousand_vectors_need_8000_bytes_of_codes(self, corpus):
        index = PQIndex(DIM, m=8)
        index.add(corpus[0])
        assert index.codes.nbytes == 8000

    def test_given_32_dim_float32_vectors_eight_byte_codes_are_16_times_smaller(self, corpus):
        index = PQIndex(DIM, m=8)
        index.add(corpus[0])
        # 32 dims × 4 bytes = 128 bytes raw versus 8 bytes of codes.
        assert corpus[0].astype(np.float32).nbytes / index.codes.nbytes == 16

    def test_when_a_shortlist_is_rescored_with_exact_vectors_recall_recovers(self, corpus):
        X, queries, truth = corpus
        approx, rescored = PQIndex(DIM, m=8), PQIndex(DIM, m=8, rerank=50)
        approx.add(X)
        rescored.add(X)
        assert evaluate(rescored, queries, truth)["recall"] > evaluate(approx, queries, truth)["recall"]

    def test_given_residual_codes_and_eight_probes_ivf_pq_with_rescoring_finds_most_true_neighbors(self, corpus):
        X, queries, truth = corpus
        index = IVFPQIndex(DIM, nlist=16, nprobe=8, m=8, rerank=50)
        index.add(X)
        assert evaluate(index, queries, truth)["recall"] >= 0.8


class TestHNSW:
    def test_given_a_thousand_vectors_it_finds_at_least_90_percent_of_the_true_top_10(self, hnsw, corpus):
        _, queries, truth = corpus
        assert evaluate(hnsw, queries, truth)["recall"] >= 0.9

    def test_when_searching_it_touches_under_half_the_corpus(self, hnsw, corpus):
        _, queries, truth = corpus
        assert evaluate(hnsw, queries, truth)["ndist"] < 500

    def test_given_m_of_8_each_layer_holds_roughly_one_eighth_of_the_layer_below(self, hnsw):
        sizes = hnsw.layer_sizes()
        # P(level >= 1) = 1/M = 0.125, so about 125 of 1,000 nodes reach layer 1.
        assert sizes[0] == 1000
        assert 80 <= sizes[1] <= 170

    def test_given_bidirectional_links_no_bottom_layer_node_exceeds_2m_neighbors(self, hnsw):
        # Layer 0 allows M0 = 2·M = 16 links; pruning keeps every node within it.
        assert max(len(node[0]) for node in hnsw.links) <= 16

    def test_when_ef_search_widens_recall_does_not_fall_and_more_vectors_are_compared(self, hnsw, corpus):
        _, queries, truth = corpus
        hnsw.ef_search = 10
        narrow = evaluate(hnsw, queries, truth)
        hnsw.ef_search = 100
        wide = evaluate(hnsw, queries, truth)
        hnsw.ef_search = 50
        assert wide["recall"] >= narrow["recall"]
        assert wide["ndist"] > narrow["ndist"]

    def test_given_an_empty_index_a_search_returns_nothing(self):
        ids, scores = HNSWIndex(DIM).search(np.ones(DIM) / np.sqrt(DIM), k=5)
        assert len(ids) == 0


@pytest.fixture(scope="module")
def trace():
    """One HNSW index and one search through it, built once for every search-path scenario."""
    from primer.ml.embeddings.ann import planar_vectors

    vectors, _ = planar_vectors(300, seed=0)
    index = HNSWIndex(3, M=6, ef_construction=40, ef_search=10)
    index.add(vectors)
    query, _ = planar_vectors(1, seed=5)
    return index, index.search_trace(query[0], k=5)


class TestHNSWSearchPath:

    def test_given_a_query_the_search_starts_on_the_top_layer(self, trace):
        index, steps = trace
        assert steps[0][0] == index.max_level

    def test_given_a_query_the_search_finishes_on_the_bottom_layer(self, trace):
        _, steps = trace
        assert steps[-1][0] == 0

    def test_when_descending_each_layer_starts_closer_to_the_query_than_the_one_above(self, trace):
        # Greedy hops on a coarse layer hand a better starting point to the finer layer below.
        _, steps = trace
        first_sim_per_layer = {}
        for layer, _, sim in steps:
            first_sim_per_layer.setdefault(layer, sim)
        sims_top_down = [first_sim_per_layer[l] for l in sorted(first_sim_per_layer, reverse=True)]
        assert sims_top_down == sorted(sims_top_down)

    def test_given_300_points_and_m_of_6_the_graph_has_four_layers_of_300_64_14_and_1_nodes(self, trace):
        # The figure draws only the lower three; the lone top node is the entry point, with nowhere to hop.
        index, _ = trace
        assert [sum(1 for lv in index.levels if lv >= layer) for layer in range(index.max_level + 1)] == [300, 64, 14, 1]

    def test_given_the_figures_query_the_search_compares_it_with_51_of_the_300_points(self, trace):
        # Counted from a fresh search: a few dozen comparisons instead of a 300-point scan.
        from primer.ml.embeddings.ann import planar_vectors

        index, _ = trace
        query, _ = planar_vectors(1, seed=5)
        index.ndist = 0
        index.search_trace(query[0], k=5)
        assert index.ndist == 51


class TestEightPointMap:
    """The hand-checkable toy: eight houses on a grid, query at (5, 4). Distances are squared."""

    def test_given_the_query_at_5_4_greedy_hnsw_takes_the_highway_to_e_then_a_side_street_to_h(self):
        from primer.ml.embeddings.ann import tiny_hnsw_search

        # By hand: A is 41 away; on layer 1 its neighbours E (5) and G (10) -> hop to E; nothing on
        # layer 1 beats E; on layer 0 E's neighbours include H (1) -> hop to H; H has no closer neighbour.
        assert tiny_hnsw_search((5, 4)) == [(1, "A", 41), (1, "E", 5), (0, "E", 5), (0, "H", 1)]

    def test_given_two_clusters_ivf_probes_only_the_right_hand_cluster_for_the_query_at_5_4(self):
        from primer.ml.embeddings.ann import tiny_ivf_search

        # Centroids (1.75, 1) and (3.75, 3.75): squared distances 19.5625 vs 1.625, so probe the right one.
        probed, scanned, best = tiny_ivf_search((5, 4), nprobe=1)
        assert probed == ["right"]
        assert scanned == ["E", "F", "G", "H"]
        assert best == "H"

    def test_given_two_sub_vectors_and_four_centroids_each_pq_stores_a_4d_vector_as_two_codes(self):
        from primer.ml.embeddings.ann import tiny_pq_example

        # (0.9, 0.1) is nearest centroid 0 = (1, 0); (-0.2, 0.8) is nearest centroid 1 = (0, 1).
        assert tiny_pq_example()["codes"] == [0, 1]

    def test_given_those_codes_the_table_lookup_score_is_2_while_the_exact_score_is_1_7(self):
        from primer.ml.embeddings.ann import tiny_pq_example

        # Query (1, 0, 0, 1): tables [1, 0, -1, 0] and [0, 1, 0, -1]; lookups 1 + 1 = 2. Exact 0.9 + 0.8 = 1.7.
        result = tiny_pq_example()
        assert result["approx_score"] == pytest.approx(2.0)
        assert result["exact_score"] == pytest.approx(1.7)
