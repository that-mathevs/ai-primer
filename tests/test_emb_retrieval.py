"""Specification for primer.ml.embeddings.retrieval: keyword search, meaning search, and combining them."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from primer.common.corpus import DOCS
from primer.common.text import tokenize
from primer.ml.embeddings.retrieval import BM25, reciprocal_rank_fusion

# The hand-checkable toy corpus used throughout the lesson.
TOY = [["cat", "cat", "dog"], ["dog", "bird"], ["fish"]]


class TestBM25:
    def test_given_a_word_in_one_of_three_documents_its_rarity_weight_is_about_0_98(self):
        # IDF = ln(1 + (3 - 1 + 0.5) / (1 + 0.5)) = ln(2.667) = 0.981
        assert BM25(TOY).idf["cat"] == pytest.approx(0.981, abs=0.001)

    def test_given_a_word_in_two_of_three_documents_it_weighs_less_than_a_rarer_word(self):
        # IDF(dog) = ln(1 + 1.5 / 2.5) = ln(1.6) = 0.470, about half of cat's 0.981.
        assert BM25(TOY).idf["dog"] == pytest.approx(0.470, abs=0.001)

    def test_given_the_query_cat_the_document_mentioning_it_twice_scores_about_1_21(self):
        # f = 2, |d| = 3, avgdl = 2: denominator 2 + 1.5·(0.25 + 0.75·1.5) = 4.0625;
        # score = 0.981 · 2 · 2.5 / 4.0625 = 1.207
        assert BM25(TOY).scores(["cat"])[0] == pytest.approx(1.207, abs=0.001)

    def test_given_a_word_repeated_100_times_its_credit_stays_below_k1_plus_one(self):
        # Saturation: f·(k1+1)/(f+k1) approaches k1 + 1 = 2.5 but never passes it.
        one_doc = BM25([["spam"] * 100, ["ham"]], b=0.0)
        per_idf = one_doc.scores(["spam"])[0] / one_doc.idf["spam"]
        assert per_idf == pytest.approx(100 * 2.5 / 101.5)
        assert per_idf < 2.5

    def test_given_equal_mentions_the_shorter_document_scores_higher(self):
        index = BM25([["cat", "a", "b", "c", "d", "e"], ["cat", "a"], ["dog"]])
        long_doc, short_doc, _ = index.scores(["cat"])
        assert short_doc > long_doc

    def test_given_a_word_no_document_contains_every_score_is_zero(self):
        assert BM25(TOY).scores(["zebra"]).tolist() == [0.0, 0.0, 0.0]

    def test_when_searching_documents_with_zero_score_are_left_out(self):
        assert [i for i, _ in BM25(TOY).search(["dog"])] == [1, 0]

    def test_given_an_exact_error_code_the_matching_article_ranks_first(self):
        index = BM25([tokenize(d.title + ". " + d.text) for d in DOCS])
        best, _ = index.search(tokenize("what does ERR-4012 mean"), k=1)[0]
        assert DOCS[best].id == "it-004"


class TestReciprocalRankFusion:
    def test_given_ranks_1_and_3_the_fused_score_is_about_0_0323(self):
        fused = dict(reciprocal_rank_fusion([["doc", "x", "y"], ["a", "b", "doc"]]))
        # 1/61 + 1/63
        assert fused["doc"] == pytest.approx(0.0323, abs=0.0001)

    def test_given_rank_1_in_one_list_only_the_fused_score_is_about_0_0164(self):
        fused = dict(reciprocal_rank_fusion([["doc"], ["other"]]))
        # 1/61
        assert fused["doc"] == pytest.approx(0.0164, abs=0.0001)

    def test_given_two_lists_a_document_both_agree_on_outranks_one_that_tops_a_single_list(self):
        fused = reciprocal_rank_fusion([["solo", "both"], ["both", "other"]])
        # both: 1/62 + 1/61 = 0.0325; solo: 1/61 = 0.0164
        assert fused[0][0] == "both"


@pytest.fixture(scope="module")
def engine():
    from primer.ml.embeddings.retrieval import SearchEngine

    return SearchEngine(DOCS)


class TestHybridSearch:
    def test_given_an_exact_error_code_query_dense_search_leaves_the_answer_out_of_its_top_3(self, engine):
        assert "it-004" not in engine.dense("what does ERR-4012 mean", k=3)

    def test_given_an_exact_error_code_query_bm25_ranks_the_answer_first(self, engine):
        assert engine.bm25("what does ERR-4012 mean", k=3)[0] == "it-004"

    def test_given_a_synonym_query_with_no_shared_words_bm25_finds_nothing(self, engine):
        # "automobile reimbursement" vs. "car mileage expense ... reimbursed": zero words in common.
        assert engine.bm25("automobile reimbursement", k=3) == []

    def test_given_a_synonym_query_dense_search_ranks_the_car_expense_article_first(self, engine):
        assert engine.dense("automobile reimbursement", k=3)[0] == "fin-001"

    def test_given_the_error_code_query_hybrid_search_ranks_the_answer_first(self, engine):
        assert engine.hybrid("what does ERR-4012 mean", k=3)[0] == "it-004"

    def test_given_the_synonym_query_hybrid_search_ranks_the_answer_first(self, engine):
        assert engine.hybrid("automobile reimbursement", k=3)[0] == "fin-001"


class TestRetrievalQuality:
    """Recall@3 over the 20 evaluation queries: 12 everyday ones plus 5 paraphrases and 3 exact codes."""

    def test_given_the_evaluation_set_bm25_finds_15_of_20_answers_in_its_top_3(self, engine):
        from primer.ml.embeddings.retrieval import EVAL_QUERIES, evaluate

        # BM25 misses the five paraphrase queries, which share no words with their answers.
        assert evaluate(engine.bm25, EVAL_QUERIES, k=3)["recall"] == pytest.approx(0.75)

    def test_given_the_evaluation_set_dense_search_finds_18_of_20_answers_in_its_top_3(self, engine):
        from primer.ml.embeddings.retrieval import EVAL_QUERIES, evaluate

        # Dense search misses two error-code queries, where the code is one blurred word among many.
        assert evaluate(engine.dense, EVAL_QUERIES, k=3)["recall"] == pytest.approx(0.9)

    def test_given_the_evaluation_set_hybrid_search_finds_all_20_answers_in_its_top_3(self, engine):
        from primer.ml.embeddings.retrieval import EVAL_QUERIES, evaluate

        assert evaluate(engine.hybrid, EVAL_QUERIES, k=3)["recall"] == pytest.approx(1.0)

    def test_given_answers_at_ranks_1_2_and_missing_mean_reciprocal_rank_is_one_half(self):
        from primer.ml.embeddings.retrieval import evaluate

        rankings = {"q1": ["a", "x"], "q2": ["x", "b"], "q3": ["x", "y"]}
        queries = [("q1", {"a"}), ("q2", {"b"}), ("q3", {"c"})]
        # (1/1 + 1/2 + 0) / 3 = 0.5
        assert evaluate(lambda q, k: rankings[q][:k], queries, k=3)["mrr"] == pytest.approx(0.5)


class TestCrossEncoderReranking:
    def test_given_password_rules_hybrid_search_ranks_the_reset_how_to_above_the_policy(self, engine):
        # The hard negative: same topic (passwords), different question. First-stage search confuses them.
        ranking = engine.hybrid("what are the password rules", k=3)
        assert ranking.index("it-001") < ranking.index("it-002")

    def test_given_password_rules_the_policy_article_covers_both_ideas_in_the_query(self):
        from primer.ml.embeddings.retrieval import CrossEncoder

        # Query ideas: {password, rules}. The policy has both (password, policy); the how-to has one.
        judge = CrossEncoder()
        policy, reset = (next(d for d in DOCS if d.id == i) for i in ("it-002", "it-001"))
        assert judge.coverage("what are the password rules", policy) == 1.0
        assert judge.coverage("what are the password rules", reset) == 0.5

    def test_given_password_rules_the_policy_scores_about_1_78_and_the_reset_how_to_about_0_69(self):
        from primer.ml.embeddings.retrieval import CrossEncoder

        # coverage + 0.5·phrase + 0.25·exact + 0.25·cosine, by hand from the lesson's table:
        # policy 1.0 + 0.5·1.0 + 0.25·0.5 + 0.25·0.63 ≈ 1.78; how-to 0.5 + 0 + 0.25·0.5 + 0.25·0.27 ≈ 0.69.
        judge = CrossEncoder()
        policy, reset = (next(d for d in DOCS if d.id == i) for i in ("it-002", "it-001"))
        scores = (judge.score("what are the password rules", policy), judge.score("what are the password rules", reset))
        assert scores == (pytest.approx(1.78, abs=0.005), pytest.approx(0.69, abs=0.005))

    def test_when_reranked_the_password_policy_ranks_first(self, engine):
        from primer.ml.embeddings.retrieval import retrieve_then_rerank

        assert retrieve_then_rerank(engine, "what are the password rules", k=3)[0] == "it-002"

    def test_given_forgotten_sign_in_credentials_hybrid_search_ranks_the_vpn_guide_first(self, engine):
        # BM25 latches onto the literal word "credentials", which appears in the VPN guide.
        assert engine.hybrid("sign-in credentials forgotten", k=3)[0] == "it-003"

    def test_when_reranked_the_password_reset_how_to_ranks_first(self, engine):
        from primer.ml.embeddings.retrieval import retrieve_then_rerank

        assert retrieve_then_rerank(engine, "sign-in credentials forgotten", k=3)[0] == "it-001"

    def test_given_a_shortlist_of_10_the_cross_encoder_reads_exactly_10_query_document_pairs(self, engine):
        from primer.ml.embeddings.retrieval import CrossEncoder, retrieve_then_rerank

        # Its cost grows with the shortlist, not with the corpus: that's why it only sees the top N.
        judge = CrossEncoder()
        retrieve_then_rerank(engine, "enroll in MFA", k=3, shortlist=10, judge=judge)
        assert judge.pairs_scored == 10

    def test_given_how_many_vacation_days_the_literal_word_vacation_lifts_the_pto_article_above_sick_leave(self, engine):
        from primer.ml.embeddings.retrieval import retrieve_then_rerank

        # Both articles are about time off and mention days; only one says "vacation".
        assert retrieve_then_rerank(engine, "how many vacation days do I get", k=3)[0] == "hr-001"

    def test_given_refund_for_the_hotel_the_reranker_prefers_the_article_that_talks_about_reimbursing_trips(self, engine):
        from primer.ml.embeddings.retrieval import retrieve_then_rerank

        # A known limitation, kept on purpose: rerankers are models too, and must be evaluated like one.
        assert retrieve_then_rerank(engine, "refund for the hotel", k=3)[:2] == ["fin-001", "fin-002"]

    def test_given_the_evaluation_set_reranking_keeps_all_20_answers_in_the_top_3(self, engine):
        from primer.ml.embeddings.retrieval import EVAL_QUERIES, evaluate, retrieve_then_rerank

        search = lambda q, k: retrieve_then_rerank(engine, q, k=k)  # noqa: E731
        assert evaluate(search, EVAL_QUERIES, k=3)["recall"] == pytest.approx(1.0)


class TestLateInteraction:
    def test_given_two_query_tokens_each_takes_its_best_matching_document_token_and_the_best_scores_add_up(self):
        import numpy as np

        from primer.ml.embeddings.retrieval import maxsim

        query = np.array([[1.0, 0.0], [0.0, 1.0]])
        doc = np.array([[1.0, 0.0], [0.6, 0.8]])
        # token 1: max(1.0, 0.6) = 1.0; token 2: max(0.0, 0.8) = 0.8; total 1.8
        assert maxsim(query, doc) == pytest.approx(1.8)

    def test_given_a_two_word_title_late_interaction_stores_two_vectors_where_a_bi_encoder_stores_one(self):
        from primer.ml.embeddings.retrieval import LateInteraction

        assert LateInteraction().token_vectors("Password policy").shape[0] == 2

    def test_given_password_rules_late_interaction_ranks_the_policy_above_the_reset_how_to(self):
        from primer.ml.embeddings.retrieval import LateInteraction

        ranking = LateInteraction().search("what are the password rules", DOCS, k=5)
        assert ranking.index("it-002") < ranking.index("it-001")


class TestChunking:
    def test_given_130_words_50_word_chunks_with_10_words_overlap_make_3_chunks(self):
        from primer.ml.embeddings.retrieval import fixed_size_chunks

        words = " ".join(f"w{i}" for i in range(130))
        # Windows start every 50 - 10 = 40 words: at 0, 40 and 80; the third runs to the end.
        assert len(fixed_size_chunks(words, size=50, overlap=10)) == 3

    def test_given_10_words_overlap_consecutive_chunks_share_exactly_10_words(self):
        from primer.ml.embeddings.retrieval import fixed_size_chunks

        chunks = fixed_size_chunks(" ".join(f"w{i}" for i in range(130)), size=50, overlap=10)
        assert chunks[0].split()[-10:] == chunks[1].split()[:10]

    def test_given_the_handbook_no_structure_aware_chunk_spans_two_sections(self):
        from primer.ml.embeddings.retrieval import HANDBOOK, structure_aware_chunks

        sections = {c.section for c in structure_aware_chunks(HANDBOOK, source="remote-work-handbook")}
        # The handbook has five "##" sections; every chunk belongs to exactly one of them.
        assert len(sections) == 5

    def test_given_the_handbook_each_chunk_carries_its_source_section_date_and_access_list(self):
        from primer.ml.embeddings.retrieval import HANDBOOK, structure_aware_chunks

        chunk = structure_aware_chunks(HANDBOOK, source="remote-work-handbook", updated="2026-05-01")[0]
        assert (chunk.source, chunk.updated, chunk.acl) == ("remote-work-handbook", "2026-05-01", frozenset({"everyone"}))

    def test_given_the_stipend_question_the_best_fixed_size_chunk_loses_the_amount(self):
        from primer.ml.embeddings.retrieval import HANDBOOK, best_chunk, fixed_size_chunks

        best = best_chunk("home internet stipend", fixed_size_chunks(HANDBOOK, size=40, overlap=5))
        # The heading and the amount ended up in different windows.
        assert "50 dollars" not in best

    def test_given_the_stipend_question_the_best_structure_aware_chunk_holds_the_heading_and_the_amount(self):
        from primer.ml.embeddings.retrieval import HANDBOOK, best_chunk, structure_aware_chunks

        chunks = [c.text for c in structure_aware_chunks(HANDBOOK, source="remote-work-handbook")]
        best = best_chunk("home internet stipend", chunks)
        assert "Home internet stipend" in best and "50 dollars" in best

    def test_when_a_small_child_chunk_matches_the_whole_parent_section_is_returned(self):
        from primer.ml.embeddings.retrieval import HANDBOOK, parent_child_search

        # Children are single sentences (precise matches); the parent is the section (full context).
        section = parent_child_search("which laptop monitors are allowed", HANDBOOK)
        assert section.startswith("## Equipment")


class TestQueryAndPassagePrefixes:
    """A simulated embedding model trained with "query: " and "passage: " prefixes, like the E5 family."""

    def test_given_both_prefixes_used_correctly_11_of_the_12_everyday_questions_find_their_answer_in_the_top_3(self):
        from primer.common.corpus import LABELED_QUERIES
        from primer.ml.embeddings.retrieval import PrefixedEmbedder, SearchEngine, evaluate

        engine = SearchEngine(DOCS, embedder=PrefixedEmbedder(), passage_prefix="passage: ", query_prefix="query: ")
        # Same as the plain dense search: only the error-code question misses.
        assert evaluate(engine.dense, LABELED_QUERIES, k=3)["recall"] == pytest.approx(11 / 12)

    def test_when_the_passage_prefix_is_forgotten_at_indexing_time_recall_silently_drops(self):
        from primer.common.corpus import LABELED_QUERIES
        from primer.ml.embeddings.retrieval import PrefixedEmbedder, SearchEngine, evaluate

        # Nothing errors: vectors have the right shape and scores look plausible. Only an eval notices.
        engine = SearchEngine(DOCS, embedder=PrefixedEmbedder(), passage_prefix="", query_prefix="query: ")
        # With both prefixes, 11 of 12 questions find their answer (the scenario above).
        assert evaluate(engine.dense, LABELED_QUERIES, k=3)["recall"] < 11 / 12


NODE = shutil.which("node")
ROOT = Path(__file__).resolve().parent.parent
needs_node = pytest.mark.skipif(NODE is None, reason="needs Node to run the widget's JavaScript")


def _widget(expression: str, payload) -> object:
    """Evaluate `expression` against the rag-pipeline widget's exports, with `payload` as `input`."""
    script = (
        "const m = require('./docs/assets/viz/rag-pipeline.js');"
        "const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
        f"console.log(JSON.stringify({expression}));"
    )
    out = subprocess.run([NODE, "-e", script], input=json.dumps(payload), capture_output=True, text=True, check=True, cwd=ROOT)
    return json.loads(out.stdout)


@pytest.fixture(scope="module")
def widget_data():
    from primer.ml.embeddings.retrieval import viz_data

    return viz_data()["rag-pipeline"]


class TestTheRAGWidgetAgreesWithTheLesson:
    """The chunking, retrieval and reranking widget: its numbers must be the lesson's own."""

    @needs_node
    def test_given_several_sizes_and_overlaps_the_widgets_chunker_cuts_exactly_the_lessons_chunks(self):
        from primer.ml.embeddings.retrieval import HANDBOOK, fixed_size_chunks

        cases = [(HANDBOOK, s, o) for s in (20, 30, 40, 60) for o in (0, 5, 10)]
        # The 130-word worked example, a text shorter than one chunk, and an empty one.
        cases += [(" ".join(f"w{i}" for i in range(130)), 50, 10), ("just three words", 20, 5), ("", 10, 0)]
        js = _widget("input.map(([t, s, o]) => m.fixedSizeChunks(t, s, o))", cases)
        assert js == [fixed_size_chunks(t, s, o) for t, s, o in cases]

    @needs_node
    def test_given_each_question_and_chunking_the_widget_marks_exactly_the_chunks_that_hold_the_whole_answer(self, widget_data):
        from primer.ml.embeddings.retrieval import CHUNKING_QUESTIONS, HANDBOOK, fixed_size_chunks

        cases = [(s, o, a) for s in widget_data["sizes"] for o in widget_data["overlaps"] for _, a in CHUNKING_QUESTIONS]
        js = _widget("input.cases.map(([s, o, a]) => m.answerChunks(input.text, s, o, a).whole)", {"text": HANDBOOK, "cases": cases})
        assert js == [[i for i, c in enumerate(fixed_size_chunks(HANDBOOK, s, o)) if a in c] for s, o, a in cases]

    def test_given_each_question_and_chunking_the_widgets_first_stage_order_is_the_lessons_hybrid_search(self, widget_data):
        from primer.ml.embeddings.retrieval import HANDBOOK, chunk_engine, fixed_size_chunks

        for key, runs in widget_data["runs"].items():
            engine = chunk_engine(fixed_size_chunks(HANDBOOK, *map(int, key.split("/"))))
            for q, run in zip(widget_data["questions"], runs):
                shown = [engine.ids[c] for c, _ in run["retrieval"]]
                assert shown == engine.hybrid(q["question"], k=len(engine.ids)), (key, q["question"])

    def test_given_each_question_and_chunking_the_widgets_reranker_scores_are_the_lessons_cross_encoder(self, widget_data):
        from primer.ml.embeddings.retrieval import HANDBOOK, CrossEncoder, chunk_engine, fixed_size_chunks

        judge = CrossEncoder()
        for key, runs in widget_data["runs"].items():
            engine = chunk_engine(fixed_size_chunks(HANDBOOK, *map(int, key.split("/"))))
            for q, run in zip(widget_data["questions"], runs):
                expected = [judge.score(q["question"], d) for d in engine.docs]
                assert run["rerank"] == pytest.approx(expected, abs=1e-4), (key, q["question"])

    @needs_node
    def test_given_every_top_k_the_widgets_reranked_list_is_the_lessons_retrieve_then_rerank(self, widget_data):
        from primer.ml.embeddings.retrieval import HANDBOOK, chunk_engine, fixed_size_chunks, retrieve_then_rerank

        cases, expected = [], []
        for key, runs in widget_data["runs"].items():
            engine = chunk_engine(fixed_size_chunks(HANDBOOK, *map(int, key.split("/"))))
            for q, run in zip(widget_data["questions"], runs):
                for k in range(1, widget_data["max_k"] + 1):
                    cases.append((run, k))
                    ids = retrieve_then_rerank(engine, q["question"], k=k, shortlist=k)
                    expected.append([engine.ids.index(i) for i in ids])
        assert _widget("input.map(([run, k]) => m.rerank(run, k))", cases) == expected

    def test_given_30_word_chunks_without_overlap_no_chunk_holds_the_whole_vpn_answer(self):
        from primer.ml.embeddings.retrieval import CHUNKING_QUESTIONS, HANDBOOK, fixed_size_chunks

        vpn = next(a for q, a in CHUNKING_QUESTIONS if "VPN" in q)
        assert not any(vpn in c for c in fixed_size_chunks(HANDBOOK, size=30, overlap=0))

    def test_given_30_word_chunks_with_10_words_of_overlap_one_chunk_holds_the_whole_vpn_answer(self):
        from primer.ml.embeddings.retrieval import CHUNKING_QUESTIONS, HANDBOOK, fixed_size_chunks

        # The same size as above, but the overlap repeats the sentence's start at the head of the next chunk.
        vpn = next(a for q, a in CHUNKING_QUESTIONS if "VPN" in q)
        assert sum(vpn in c for c in fixed_size_chunks(HANDBOOK, size=30, overlap=10)) == 1

    def test_given_the_stipend_amount_question_hybrid_search_ranks_a_chunk_without_the_amount_first(self):
        from primer.ml.embeddings.retrieval import CHUNKING_QUESTIONS, HANDBOOK, chunk_engine, fixed_size_chunks

        engine = chunk_engine(fixed_size_chunks(HANDBOOK, size=40, overlap=0))
        question = next(q for q, _ in CHUNKING_QUESTIONS if "stipend" in q)
        top = engine.hybrid(question, k=1)[0]
        # The heading chunk repeats "home internet stipend"; the amount sits further on.
        assert "50 dollars" not in engine.docs[engine.ids.index(top)].text

    def test_when_the_stipend_shortlist_is_reranked_the_chunk_with_the_amount_ranks_first(self):
        from primer.ml.embeddings.retrieval import CHUNKING_QUESTIONS, HANDBOOK, chunk_engine, fixed_size_chunks, retrieve_then_rerank

        engine = chunk_engine(fixed_size_chunks(HANDBOOK, size=40, overlap=0))
        question = next(q for q, _ in CHUNKING_QUESTIONS if "stipend" in q)
        top = retrieve_then_rerank(engine, question, k=5, shortlist=5)[0]
        assert "50 dollars" in engine.docs[engine.ids.index(top)].text

    def test_given_every_setting_the_widgets_data_stays_small_enough_to_load_with_the_page(self, widget_data):
        # Every page with a widget loads the whole site's widget data, so each widget's share stays small.
        assert len(json.dumps(widget_data)) < 50_000

    def test_given_the_retrieval_lesson_it_places_the_rag_pipeline_widget(self):
        import primer.ml.embeddings.retrieval as lesson

        assert '<div class="viz" data-viz="rag-pipeline" aria-label="' in lesson.__doc__
