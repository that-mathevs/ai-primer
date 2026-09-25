"""Specification for primer.agents.rag: retrieval-augmented generation, end to end."""

import pytest

from primer.agents.rag import (
    MESSY_PDF,
    RAGIndex,
    agentic_answer,
    answer,
    build_graph,
    chunk_document,
    clean_extracted_text,
    debug_wrong_answers,
    extract_table,
    hyde_search,
    multi_query_search,
    post_generation_filter_answer,
    rewrite_query,
    table_rows_as_sentences,
    verify_citations,
)
from primer.common.corpus import DOCS_BY_ID

EVERYONE = {"everyone"}
FINANCE = {"everyone", "finance"}


@pytest.fixture(scope="module")
def index():
    return RAGIndex()


@pytest.fixture(scope="module")
def contextual_index():
    return RAGIndex(contextual=True)


def doc_ids(passages):
    return [p.doc_id for p in passages]


class TestParsingMessyDocuments:
    def test_given_a_header_repeated_on_every_page_it_is_removed(self):
        assert "CONFIDENTIAL" not in clean_extracted_text(MESSY_PDF)

    def test_given_page_number_footers_they_are_removed(self):
        assert "Page 1 of 2" not in clean_extracted_text(MESSY_PDF)

    def test_given_a_word_hyphenated_across_a_line_break_it_is_rejoined(self):
        cleaned = clean_extracted_text(MESSY_PDF)
        assert "employees travelling" in cleaned
        assert "em-" not in cleaned

    def test_given_a_table_each_row_becomes_a_self_contained_sentence(self):
        # A flattened "L1-L3 150 60 L4-L6 200 75" loses which number belongs to which grade.
        rows = extract_table(MESSY_PDF, header=["Grade", "Hotel cap", "Meal cap"])
        assert table_rows_as_sentences(rows)[1] == "Grade L4-L6: hotel cap 200, meal cap 75."


class TestChunking:
    def test_given_a_document_every_passage_carries_its_access_groups_and_date(self):
        passages = chunk_document(DOCS_BY_ID["fin-006"])
        assert all(p.acl == frozenset({"finance", "exec"}) and p.updated == "2026-07-01" for p in passages)

    def test_given_a_short_word_limit_the_error_document_splits_into_its_two_sentences(self):
        passages = chunk_document(DOCS_BY_ID["it-004"], max_words=15)
        assert [p.id for p in passages] == ["it-004#0", "it-004#1"]
        assert passages[1].text == "Update AnyConnect to version 5.1 or later and reboot."


class TestHybridRetrieval:
    def test_given_an_exact_error_code_dense_search_alone_misses_the_document(self, index):
        assert "it-004" not in doc_ids(index.retrieve("what does ERR-4012 mean", EVERYONE, k=3, mode="dense"))

    def test_given_an_exact_error_code_hybrid_search_brings_the_document_into_the_top_three(self, index):
        # Keyword search ranks it first; fusion keeps it near the top even though dense search missed it.
        assert "it-004" in doc_ids(index.retrieve("what does ERR-4012 mean", EVERYONE, k=3))

    def test_given_a_question_sharing_no_words_with_the_answer_keyword_search_alone_misses_it(self, index):
        assert "it-007" not in doc_ids(index.retrieve("scam message in my inbox", EVERYONE, k=3, mode="bm25"))

    def test_given_a_question_sharing_no_words_with_the_answer_hybrid_search_finds_it(self, index):
        assert "it-007" in doc_ids(index.retrieve("scam message in my inbox", EVERYONE, k=3))


class TestPermissionAwareRetrieval:
    def test_given_a_user_outside_finance_restricted_documents_never_reach_the_model(self, index):
        passages = index.retrieve("Q3 revenue forecast", EVERYONE, k=10)
        assert not {"fin-005", "fin-006"} & set(doc_ids(passages))

    def test_given_a_finance_user_the_restricted_forecast_ranks_first(self, index):
        assert doc_ids(index.retrieve("Q3 revenue forecast", FINANCE, k=3))[0] == "fin-006"

    def test_given_filtering_after_generation_the_restricted_figure_has_already_leaked_into_the_answer(self, index):
        # Removing the citation afterwards can't remove what the model already wrote.
        assert "41 million" in post_generation_filter_answer(index, "What is the Q3 revenue forecast?", EVERYONE)

    def test_given_filtering_before_generation_the_restricted_figure_never_appears(self, index):
        assert "41 million" not in answer(index, "What is the Q3 revenue forecast?", EVERYONE).answer


class TestGroundedGeneration:
    def test_given_a_covered_question_the_answer_cites_a_passage_from_the_context(self, index):
        result = answer(index, "What does ERR-4012 mean?", EVERYONE)
        assert "[it-004#0]" in result.answer
        assert result.citations["unknown_ids"] == []

    def test_given_a_correctly_cited_answer_no_claim_is_flagged_as_unsupported(self, index):
        assert answer(index, "What does ERR-4012 mean?", EVERYONE).citations["unsupported"] == []

    def test_given_a_citation_attached_to_a_claim_its_source_does_not_support_the_claim_is_flagged(self, index):
        passages = index.retrieve("What does ERR-4012 mean?", EVERYONE, k=3)
        report = verify_citations("Managers get unlimited sabbaticals [it-004#0].", passages)
        assert report["unsupported"] == ["Managers get unlimited sabbaticals"]

    def test_given_a_question_the_sources_do_not_cover_the_model_says_it_does_not_know(self, index):
        assert answer(index, "What is the office wifi password?", EVERYONE).answer.startswith("I don't know")

    def test_given_an_answer_citing_a_source_that_was_never_retrieved_verification_flags_it(self, index):
        passages = index.retrieve("What does ERR-4012 mean?", EVERYONE, k=3)
        report = verify_citations("It means the tunnel failed [it-999#0].", passages)
        assert report["unknown_ids"] == ["it-999#0"]


class TestMetadataFiltering:
    def test_given_no_date_filter_the_superseded_2023_travel_policy_ranks_first(self, index):
        assert doc_ids(index.retrieve("business class flights over four hours", EVERYONE, k=3))[0] == "fin-003"

    def test_given_a_filter_on_documents_updated_since_2024_the_superseded_policy_never_appears(self, index):
        passages = index.retrieve("business class flights over four hours", EVERYONE, k=5, updated_after="2024-01-01")
        assert "fin-003" not in doc_ids(passages)


class TestContextualRetrieval:
    FIX = "Update AnyConnect to version 5.1 or later and reboot."

    def test_given_plain_chunks_the_fix_sentence_is_not_found_for_how_do_i_fix_err_4012(self, index):
        assert self.FIX not in [p.text for p in index.retrieve("how do I fix ERR-4012", EVERYONE, k=3)]

    def test_given_chunks_prefixed_with_their_document_title_the_fix_sentence_is_found(self, contextual_index):
        assert self.FIX in [p.text for p in contextual_index.retrieve("how do I fix ERR-4012", EVERYONE, k=3)]

    def test_given_three_phrasings_of_the_question_the_titled_fix_sentence_ranks_second_second_and_fourth(self, contextual_index):
        # The figure's finding: in the top five every time, though not always in the top three.
        phrasings = ["how do I fix ERR-4012", "ERR-4012 what should I do", "resolve ERR-4012 on my laptop"]
        ranks = [[p.text for p in contextual_index.retrieve(q, EVERYONE, k=20)].index(self.FIX) + 1 for q in phrasings]
        assert ranks == [2, 2, 4]


class TestQueryRewriting:
    HISTORY = ["How do I set up the VPN?"]

    def test_given_a_follow_up_on_its_own_retrieval_goes_to_the_printer_guide(self, index):
        assert doc_ids(index.retrieve("what if it fails?", EVERYONE, k=1)) == ["it-009"]

    def test_given_the_follow_up_rewritten_with_the_earlier_topic_retrieval_finds_vpn_help(self, index):
        rewritten = rewrite_query("what if it fails?", self.HISTORY)
        assert {"it-003", "it-004"} & set(doc_ids(index.retrieve(rewritten, EVERYONE, k=3)))


class TestHyDE:
    QUESTION = "I got a weird message asking for my bank details"

    def test_given_a_question_in_the_users_own_words_plain_dense_search_misses_the_phishing_guide(self, index):
        assert "it-007" not in doc_ids(index.retrieve(self.QUESTION, EVERYONE, k=3, mode="dense"))

    def test_given_a_hypothetical_answer_to_search_with_the_phishing_guide_ranks_first(self, index):
        assert doc_ids(hyde_search(index, self.QUESTION, EVERYONE, k=3))[0] == "it-007"


class TestMultiQuery:
    def test_given_one_informal_phrasing_the_two_factor_guide_is_missed(self, index):
        assert "it-006" not in doc_ids(index.retrieve("two step login setup", EVERYONE, k=3))

    def test_given_several_phrasings_fused_the_two_factor_guide_is_found(self, index):
        assert "it-006" in doc_ids(multi_query_search(index, "two step login setup", EVERYONE, k=3))


class TestParentChildRetrieval:
    def test_given_a_match_on_one_sentence_the_whole_parent_document_is_returned(self, index):
        parents = index.retrieve_parents("reset link expires", EVERYONE, k=1)
        assert parents[0].text == DOCS_BY_ID["it-001"].text


class TestAgenticRAG:
    def test_given_a_two_part_question_the_agent_searches_twice_and_cites_both_sources(self, index):
        result = agentic_answer(index, "What is the per-diem for meals and how do I upload expense receipts?", EVERYONE)
        assert len(result["searches"]) == 2
        assert "fin-002" in result["answer"] and "fin-004" in result["answer"]

    def test_given_small_talk_the_agent_does_not_search_at_all(self, index):
        assert agentic_answer(index, "thanks, that's all!", EVERYONE)["searches"] == []


class TestGraphRAG:
    def test_given_the_knowledge_base_two_factor_connects_to_email_vpn_and_anyconnect(self):
        graph = build_graph()
        assert {"email", "VPN", "AnyConnect"} <= set(graph["two-factor"])

    def test_given_the_graph_each_connection_remembers_which_document_stated_it(self):
        assert "it-006" in build_graph()["two-factor"]["VPN"]


class TestDebuggingWrongAnswers:
    def test_given_dense_only_retrieval_the_error_code_question_is_diagnosed_as_a_retrieval_miss(self, index):
        report = debug_wrong_answers(index, mode="dense")
        assert report["by_query"]["what does ERR-4012 mean"] == "retrieval miss"

    def test_given_hybrid_retrieval_the_same_question_is_answered_correctly(self, index):
        assert debug_wrong_answers(index, mode="hybrid")["by_query"]["what does ERR-4012 mean"] == "ok"

    def test_given_both_modes_hybrid_has_higher_recall_at_three(self, index):
        assert debug_wrong_answers(index, mode="hybrid")["recall_at_k"] > debug_wrong_answers(index, mode="dense")["recall_at_k"]
