"""Specification for primer.ml.embeddings.operations: running embeddings in production without surprises."""

import pytest

from primer.common.embedder import ConceptEmbedder
from primer.ml.embeddings.operations import (
    BETTER_MODEL,
    CURRENT_MODEL,
    JARGON_QUERIES,
    SAME_SIZE_NEW_MODEL,
    WORSE_MODEL,
    BlueGreenMigration,
    DomainTunedEmbedder,
    build_eval_set,
    golden_recall,
    reembed_estimate,
    triage,
    triage_golden_set,
)
from primer.common.corpus import Doc


class TestModelsHaveTheirOwnSpaces:
    def test_given_the_same_word_two_different_models_give_unrelated_vectors(self):
        # Each model defines its own coordinates, so even identical text lands somewhere unrelated.
        a = ConceptEmbedder(name="model-a").encode("vpn")
        b = ConceptEmbedder(name="model-b").encode("vpn")
        assert abs(float(a @ b)) < 0.3

    def test_given_queries_and_documents_from_the_same_model_retrieval_finds_nearly_every_answer(self):
        assert golden_recall(query_model=CURRENT_MODEL, doc_model=CURRENT_MODEL, k=3) >= 0.9

    def test_given_a_same_size_new_model_querying_old_document_vectors_silently_drops_to_chance(self):
        # 20 documents, top 3: a random ranking finds the answer 3/20 = 15% of the time. Nothing crashes.
        assert golden_recall(query_model=SAME_SIZE_NEW_MODEL, doc_model=CURRENT_MODEL, k=3) <= 0.25

    def test_given_v2_queries_over_v1_documents_exactly_3_of_the_12_golden_questions_find_their_answer(self):
        # The lesson's bar reads 0.25: 3 lucky hits out of 12, a little above the 0.15 of random ranking.
        assert golden_recall(query_model=SAME_SIZE_NEW_MODEL, doc_model=CURRENT_MODEL, k=3) == pytest.approx(3 / 12)

    def test_given_a_new_model_with_more_dimensions_querying_old_document_vectors_fails_loudly(self):
        # 128 numbers can't be dotted with 64: the lucky case, because you find out immediately.
        with pytest.raises(ValueError):
            golden_recall(query_model=BETTER_MODEL, doc_model=CURRENT_MODEL, k=3)


class TestBlueGreenMigration:
    def test_given_a_better_new_model_cutover_points_live_traffic_at_the_new_index(self):
        m = BlueGreenMigration(CURRENT_MODEL)
        m.start(BETTER_MODEL)
        m.backfill()
        m.shadow_compare(k=3)
        assert m.cutover() is True
        assert m.live_version == BETTER_MODEL.name

    def test_given_a_worse_new_model_cutover_is_refused_and_live_traffic_stays_put(self):
        m = BlueGreenMigration(CURRENT_MODEL)
        m.start(WORSE_MODEL)
        m.backfill()
        m.shadow_compare(k=3)
        assert m.cutover() is False
        assert m.live_version == CURRENT_MODEL.name

    def test_given_no_shadow_comparison_cutover_is_refused(self):
        # Never switch on faith: the evidence step can't be skipped.
        m = BlueGreenMigration(CURRENT_MODEL)
        m.start(BETTER_MODEL)
        m.backfill()
        assert m.cutover() is False

    def test_given_a_rollback_after_cutover_live_traffic_returns_to_the_old_index(self):
        m = BlueGreenMigration(CURRENT_MODEL)
        m.start(BETTER_MODEL)
        m.backfill()
        m.shadow_compare(k=3)
        m.cutover()
        m.rollback()
        assert m.live_version == CURRENT_MODEL.name

    def test_given_a_document_added_during_the_migration_both_indexes_can_find_it(self):
        m = BlueGreenMigration(CURRENT_MODEL)
        m.start(BETTER_MODEL)
        m.add(Doc("it-999", "Wi-Fi guest network", "The guest Wi-Fi password rotates every Monday.", "IT", "2026-09-01"))
        m.backfill()
        assert all("it-999" in idx.ids for idx in m.indexes.values())


class TestReembeddingCost:
    def test_given_fifty_million_documents_of_500_tokens_at_a_million_tokens_a_second_it_takes_about_seven_hours(self):
        # 50,000,000 × 500 = 25 billion tokens; 25e9 / 1e6 per second = 25,000 s = 6.94 h.
        est = reembed_estimate(50_000_000, avg_tokens=500, tokens_per_second=1_000_000, dollars_per_million_tokens=0.02)
        assert est["hours"] == pytest.approx(6.94, abs=0.01)

    def test_given_two_cents_per_million_tokens_re_embedding_fifty_million_documents_costs_500_dollars(self):
        # 25,000 million tokens × $0.02 = $500.
        est = reembed_estimate(50_000_000, avg_tokens=500, tokens_per_second=1_000_000, dollars_per_million_tokens=0.02)
        assert est["dollars"] == pytest.approx(500.0)


class TestDomainMismatch:
    def test_given_company_jargon_a_general_model_misses_most_of_the_answers(self):
        assert golden_recall(CURRENT_MODEL, CURRENT_MODEL, k=1, queries=JARGON_QUERIES) <= 0.5

    def test_given_a_model_tuned_on_the_jargon_it_finds_every_answer(self):
        tuned = DomainTunedEmbedder()
        assert golden_recall(tuned, tuned, k=1, queries=JARGON_QUERIES) == 1.0


class TestBuildingAnEvalSet:
    def test_given_a_search_log_only_resolved_sessions_become_eval_cases(self):
        log = [
            {"query": "vpn error", "clicked": "it-004", "resolved": True},
            {"query": "vpn broke", "clicked": "it-009", "resolved": False},
        ]
        assert build_eval_set(log) == [("vpn error", {"it-004"})]

    def test_given_the_same_question_resolved_by_two_documents_both_count_as_relevant(self):
        log = [
            {"query": "Travel policy?", "clicked": "fin-002", "resolved": True},
            {"query": "travel policy", "clicked": "fin-004", "resolved": True},
        ]
        assert build_eval_set(log) == [("travel policy", {"fin-002", "fin-004"})]


class TestFailureTriage:
    def test_given_no_relevant_document_in_the_top_k_the_failure_is_retrieval(self):
        assert triage(retrieved=["it-002", "fin-006", "fin-001"], relevant={"it-004"}, answer_used="it-002") == "retrieval failure"

    def test_given_the_relevant_document_retrieved_but_another_used_the_failure_is_generation(self):
        assert triage(retrieved=["hr-002", "hr-001", "hr-005"], relevant={"hr-001"}, answer_used="hr-002") == "generation failure"

    def test_given_the_relevant_document_retrieved_and_used_the_answer_is_correct(self):
        assert triage(retrieved=["it-001", "it-002"], relevant={"it-001"}, answer_used="it-001") == "correct"

    def test_given_the_golden_set_the_error_code_query_is_a_retrieval_failure_for_dense_search(self):
        # Dense vectors blur exact identifiers like ERR-4012; no prompt fix can help when the page wasn't retrieved.
        verdicts = dict(triage_golden_set(CURRENT_MODEL, k=3))
        assert verdicts["what does ERR-4012 mean"] == "retrieval failure"
