"""Specification for primer.ml.embeddings.clustering: grouping, mapping and routing by meaning."""

import numpy as np
import pytest

from primer.ml.embeddings.clustering import (
    OUTLIERS,
    TICKETS,
    SemanticCache,
    anomaly_scores,
    best_k_by_silhouette,
    dbscan,
    kmeans,
    kmeans_history,
    near_duplicates,
    pca,
    silhouette,
    support_router,
    ticket_embeddings,
)

FOUR_POINTS = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 0.0], [10.0, 1.0]])


class TestKMeans:
    def test_given_two_obvious_pairs_of_points_k_means_puts_each_pair_in_its_own_cluster(self):
        labels, _, _ = kmeans(FOUR_POINTS, k=2)
        assert labels[0] == labels[1] and labels[2] == labels[3] and labels[0] != labels[2]

    def test_given_two_obvious_pairs_the_centres_land_halfway_between_each_pair(self):
        _, centres, _ = kmeans(FOUR_POINTS, k=2)
        assert sorted(map(tuple, centres.round(6))) == [(0.0, 0.5), (10.0, 0.5)]

    def test_given_two_obvious_pairs_each_round_never_increases_the_total_squared_distance(self):
        # Assigning to the nearest centre and moving centres to the mean can each only lower it.
        inertias = [h["inertia"] for h in kmeans_history(FOUR_POINTS, k=2)]
        assert all(later <= earlier + 1e-12 for earlier, later in zip(inertias, inertias[1:]))

    def test_given_two_obvious_pairs_the_total_squared_distance_to_centres_is_one(self):
        # Every point is 0.5 from its centre: 4 × 0.5² = 1.0.
        _, _, inertia = kmeans(FOUR_POINTS, k=2)
        assert inertia == pytest.approx(1.0)


class TestSilhouette:
    def test_given_two_tight_far_apart_pairs_the_silhouette_is_0_90(self):
        # Point (0,0): a = 1 (its partner), b = (10 + √101) / 2 = 10.025, s = (b - a) / b = 0.900. All four are alike.
        assert silhouette(FOUR_POINTS, np.array([0, 0, 1, 1])) == pytest.approx(0.900, abs=1e-3)

    def test_given_five_kinds_of_support_ticket_the_best_silhouette_picks_five_clusters(self):
        X, _ = ticket_embeddings(include_outliers=False)
        assert best_k_by_silhouette(X, range(2, 9)) == 5


class TestDensityClustering:
    def test_given_two_dense_runs_and_a_straggler_dbscan_finds_two_clusters_and_one_noise_point(self):
        # With eps 0.15, 0-0.1-0.2 chain together, so do 5.0-5.1-5.2; 20 has no neighbour within reach.
        X = np.array([[0.0], [0.1], [0.2], [5.0], [5.1], [5.2], [20.0]])
        assert dbscan(X, eps=0.15, min_samples=2, metric="euclidean").tolist() == [0, 0, 0, 1, 1, 1, -1]

    def test_given_support_tickets_dbscan_marks_exactly_the_off_topic_tickets_as_noise(self):
        X, texts = ticket_embeddings(include_outliers=True)
        labels = dbscan(X, eps=0.6, min_samples=3)
        assert {t for t, l in zip(texts, labels) if l == -1} == set(OUTLIERS)

    def test_given_too_small_a_reach_dbscan_strands_genuine_tickets_as_noise(self):
        X, _ = ticket_embeddings(include_outliers=True)
        assert int(np.sum(dbscan(X, eps=0.55, min_samples=3) == -1)) > len(OUTLIERS)

    def test_given_too_large_a_reach_dbscan_chains_different_kinds_into_one_cluster(self):
        # Density chaining: A near B near C puts A and C together even when A and C are unlike.
        X, _ = ticket_embeddings(include_outliers=True)
        labels = dbscan(X, eps=0.75, min_samples=3)
        assert len(set(labels.tolist()) - {-1}) < len(TICKETS)


class TestPCA:
    def test_given_points_on_a_straight_line_the_first_direction_explains_all_the_variation(self):
        _, ratio = pca(np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]), n=2)
        assert ratio == pytest.approx([1.0, 0.0], abs=1e-9)

    def test_given_points_on_a_straight_line_their_2d_coordinates_are_spaced_by_root_two(self):
        # Neighbouring points are √2 = 1.414 apart along the line, and PCA keeps that distance.
        coords, _ = pca(np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]), n=1)
        assert np.abs(np.diff(coords[:, 0])) == pytest.approx([1.414, 1.414], abs=1e-3)


class TestPracticalUses:
    def test_given_a_reworded_copy_of_a_ticket_near_duplicate_detection_pairs_them(self):
        texts = ["password reset link expired", "The password reset link has expired!", "printer out of toner"]
        assert near_duplicates(texts, threshold=0.95) == [(0, 1)]

    def test_given_a_vpn_complaint_the_router_sends_it_to_the_it_helpdesk(self):
        assert support_router().route("my vpn tunnel drops when I work remote") == "it_helpdesk"

    def test_given_a_receipt_question_the_router_sends_it_to_finance(self):
        assert support_router().route("who reimburses my hotel receipt") == "finance"

    def test_given_an_off_topic_question_the_router_falls_back_to_a_human(self):
        assert support_router().route("what is the capital of france") == "fallback"

    def test_given_ticket_clusters_the_off_topic_tickets_are_farthest_from_every_centre(self):
        X, texts = ticket_embeddings(include_outliers=True)
        _, centres, _ = kmeans(X[: -len(OUTLIERS)], k=5)
        worst_two = {texts[i] for i in np.argsort(-anomaly_scores(X, centres))[:2]}
        assert worst_two == set(OUTLIERS)


class TestSemanticCache:
    def test_given_a_reworded_question_in_the_same_context_the_cached_answer_is_reused(self):
        cache = SemanticCache(threshold=0.9)
        cache.put("How do I reset my password?", "Use the self-service portal.", context="everyone", now=0)
        assert cache.get("how can I reset my password", context="everyone", now=10) == "Use the self-service portal."

    def test_given_a_question_about_something_else_the_cache_misses(self):
        # "reset my VPN" shares two of three content words with "reset my password" but is a different question.
        cache = SemanticCache(threshold=0.9)
        cache.put("How do I reset my password?", "Use the self-service portal.", context="everyone", now=0)
        assert cache.get("How do I reset my VPN?", context="everyone", now=10) is None

    def test_given_the_same_personal_question_from_another_user_the_cache_misses(self):
        # Alice's PTO balance must never be served to Bob, however similar the words.
        cache = SemanticCache(threshold=0.9)
        cache.put("What is my PTO balance?", "Alice has 12 days.", context="user:alice", now=0)
        assert cache.get("What is my PTO balance?", context="user:bob", now=10) is None

    def test_given_an_entry_older_than_its_time_to_live_the_cache_misses(self):
        cache = SemanticCache(threshold=0.9, ttl_seconds=3600)
        cache.put("How do I reset my password?", "Use the self-service portal.", context="everyone", now=0)
        assert cache.get("How do I reset my password?", context="everyone", now=3601) is None
