"""Specification for primer.agents.orchestration: the least autonomy that solves the problem."""

import threading

import pytest

from primer.agents.llm import ScriptedLLM
from primer.agents.orchestration import ChainStep, run_chain


def _has_three_points(text):
    return None if text.count("- ") >= 3 else "outline needs at least 3 points"


class TestPromptChaining:
    def test_given_an_outline_that_passes_the_gate_the_chain_writes_the_draft_from_it(self):
        llm = ScriptedLLM(["- scope\n- risks\n- timeline", "Draft covering scope, risks and timeline."])
        result = run_chain(llm, "Q3 plan", [ChainStep("outline", "Outline: {input}", _has_three_points), ChainStep("draft", "Draft from: {input}")])
        assert (result.stopped_at, result.outputs["draft"]) == (None, "Draft covering scope, risks and timeline.")

    def test_given_an_outline_that_fails_the_gate_the_chain_stops_before_drafting(self):
        # The gate is plain code between calls: a bad intermediate result never reaches the next step.
        llm = ScriptedLLM(["- scope", "never used"])
        result = run_chain(llm, "Q3 plan", [ChainStep("outline", "Outline: {input}", _has_three_points), ChainStep("draft", "Draft from: {input}")])
        assert (result.stopped_at, result.reason, llm.calls) == ("outline", "outline needs at least 3 points", 1)


HANDLERS = {
    "billing": lambda q: f"billing team: {q}",
    "technical": lambda q: f"tech support: {q}",
    "general": lambda q: f"front desk: {q}",
}


class TestRouting:
    def test_given_a_billing_question_the_classifier_sends_it_to_the_billing_handler(self):
        from primer.agents.orchestration import route

        label, answer = route(ScriptedLLM(["billing"]), HANDLERS, "Why was I charged twice?")
        assert (label, answer) == ("billing", "billing team: Why was I charged twice?")

    def test_given_a_classifier_answer_outside_the_allowed_labels_the_request_goes_to_the_fallback(self):
        # The label is free text from a model; code must never trust it to be one of the keys.
        from primer.agents.orchestration import route

        label, _ = route(ScriptedLLM(["Billing-ish, maybe refunds?"]), HANDLERS, "Why was I charged twice?")
        assert label == "general"

    def test_given_a_label_with_different_case_and_spacing_it_still_matches(self):
        from primer.agents.orchestration import route

        label, _ = route(ScriptedLLM(["  Technical\n"]), HANDLERS, "VPN keeps dropping")
        assert label == "technical"


class TestParallelization:
    def test_given_two_independent_sections_they_run_at_the_same_time(self):
        # Each call waits for the other at a barrier; run one after the other, the first would time out.
        from primer.agents.orchestration import run_sections

        barrier = threading.Barrier(2, timeout=2)

        def section(name):
            def run():
                barrier.wait()
                return f"{name} done"
            return run

        assert run_sections({"guardrail": section("guardrail"), "answer": section("answer")}) == {
            "guardrail": "guardrail done",
            "answer": "answer done",
        }

    def test_given_three_reviewers_where_two_say_vulnerable_the_majority_verdict_is_vulnerable(self):
        from primer.agents.orchestration import vote

        reviewers = [ScriptedLLM(["vulnerable"]), ScriptedLLM(["safe"]), ScriptedLLM(["vulnerable"])]
        assert vote(reviewers, "Review: query = 'SELECT * FROM users WHERE id=' + user_input") == ("vulnerable", {"vulnerable": 2, "safe": 1})

    def test_given_three_independent_voters_each_right_80_percent_of_the_time_the_majority_is_right_89_6_percent(self):
        # P(at least 2 of 3 right) = 0.8³ + 3 · 0.8² · 0.2 = 0.512 + 0.384
        from primer.agents.orchestration import majority_accuracy

        assert majority_accuracy(0.8, 3) == pytest.approx(0.896)


class TestOrchestratorWorkers:
    REQUEST = "I'm travelling for work next month: what's the meal per-diem, when are receipts due, and does unused PTO roll over?"

    def _run(self):
        from primer.agents.orchestration import demo_orchestrator, demo_synthesizer, orchestrate, policy_worker

        return orchestrate(demo_orchestrator(), policy_worker, demo_synthesizer(), self.REQUEST)

    def test_given_a_request_with_three_questions_the_orchestrator_splits_it_into_three_subtasks(self):
        assert self._run().subtasks == ["meal per-diem when travelling", "deadline for expense receipts", "PTO rollover"]

    def test_given_three_subtasks_each_worker_answer_is_grounded_in_its_own_policy_document(self):
        assert [answer.split("]")[0] + "]" for answer in self._run().worker_outputs.values()] == ["[fin-002]", "[fin-004]", "[hr-001]"]

    def test_given_the_worker_answers_the_final_reply_cites_all_three_sources(self):
        final = self._run().final
        assert all(doc in final for doc in ("fin-002", "fin-004", "hr-001"))


class TestEvaluatorOptimizer:
    def test_given_a_first_draft_without_a_citation_the_editor_sends_it_back_and_the_second_draft_passes(self):
        from primer.agents.orchestration import evaluate_optimize

        writer = ScriptedLLM(["PTO rolls over.", "Up to 5 unused PTO days roll over [hr-001]."])
        editor = ScriptedLLM(["Missing a citation: cite the policy id in brackets.", "PASS"])
        assert evaluate_optimize(writer, editor, "Does PTO roll over?") == ("Up to 5 unused PTO days roll over [hr-001].", 2, True)

    def test_given_an_editor_that_never_approves_the_loop_stops_after_three_rounds_and_says_so(self):
        from primer.agents.orchestration import evaluate_optimize

        writer, editor = ScriptedLLM(["draft"]), ScriptedLLM(["Not good enough."])
        draft, rounds, passed = evaluate_optimize(writer, editor, "Does PTO roll over?", max_rounds=3)
        assert (rounds, passed) == (3, False)


def _recording(reply, seen):
    """A specialist that records every prompt it receives and always gives the same reply."""

    def policy(system, messages, tools):
        seen.append(messages[-1]["content"])
        return reply

    return ScriptedLLM(policy)


SUPERVISOR_PLAN = '{"delegate": [{"to": "hr", "task": "How many PTO days does a full-time employee get?"}, {"to": "finance", "task": "What is the daily meal per-diem?"}]}'


class TestSupervisor:
    def test_given_a_question_needing_hr_and_finance_facts_the_supervisor_gives_one_task_to_each_specialist(self):
        from primer.agents.orchestration import supervise

        hr_seen, fin_seen = [], []
        result = supervise(
            ScriptedLLM([SUPERVISOR_PLAN, "20 PTO days; 75 dollars a day."]),
            {"hr": _recording("20 days [hr-001]", hr_seen), "finance": _recording("75 dollars [fin-002]", fin_seen)},
            "Planning a work trip: how much PTO do I get and what's the meal allowance?",
        )
        assert result.delegations == [("hr", "How many PTO days does a full-time employee get?"), ("finance", "What is the daily meal per-diem?")]

    def test_given_delegated_tasks_each_specialist_sees_only_its_own_task(self):
        # Focused context is the main benefit of multiple agents; leaking the other task would waste it.
        from primer.agents.orchestration import supervise

        hr_seen, fin_seen = [], []
        supervise(
            ScriptedLLM([SUPERVISOR_PLAN, "done"]),
            {"hr": _recording("20 days", hr_seen), "finance": _recording("75 dollars", fin_seen)},
            "Planning a work trip: how much PTO do I get and what's the meal allowance?",
        )
        assert (hr_seen, fin_seen) == (["How many PTO days does a full-time employee get?"], ["What is the daily meal per-diem?"])

    def test_given_a_delegation_to_a_specialist_that_does_not_exist_it_is_skipped_and_reported(self):
        from primer.agents.orchestration import supervise

        plan = '{"delegate": [{"to": "legal", "task": "Is this contract valid?"}]}'
        result = supervise(ScriptedLLM([plan, "I could not get a legal answer."]), {"hr": ScriptedLLM(["x"])}, "Is this contract valid?")
        assert result.answers == {"legal": "No specialist named 'legal'. Available: hr."}


SMALL_INVOICE = "INVOICE INV-2041 from Acme Supplies. Amount due: 1200.00 USD."
LARGE_INVOICE = "INVOICE INV-2042 from Globex. Amount due: 18000.00 USD."
NEWSLETTER = "Acme Supplies newsletter: our autumn catalogue is here!"


def _workflow(tmp_path, post=None):
    from primer.agents.orchestration import InvoiceWorkflow, demo_document_llm

    llm = demo_document_llm()
    posted = []
    wf = InvoiceWorkflow(llm, tmp_path / "checkpoint.json", post=post or posted.append)
    return wf, llm, posted


class TestStateMachine:
    def test_given_an_invoice_under_the_approval_limit_it_moves_through_every_state_to_done(self, tmp_path):
        wf, _, posted = _workflow(tmp_path)
        run = wf.run(SMALL_INVOICE)
        assert run.history == ["RECEIVED", "CLASSIFIED", "EXTRACTED", "VALIDATED", "POSTED", "DONE"]
        assert posted == [{"invoice_no": "INV-2041", "vendor": "Acme Supplies", "amount": 1200.0}]

    def test_given_an_invoice_over_the_approval_limit_the_run_pauses_for_a_human_and_posts_nothing(self, tmp_path):
        wf, _, posted = _workflow(tmp_path)
        assert (wf.run(LARGE_INVOICE).state, posted) == ("AWAITING_APPROVAL", [])

    def test_given_a_document_that_is_not_an_invoice_the_run_is_rejected_without_extracting_anything(self, tmp_path):
        wf, llm, _ = _workflow(tmp_path)
        run = wf.run(NEWSLETTER)
        assert (run.history, llm.calls) == (["RECEIVED", "REJECTED"], 1)

    def test_given_an_extracted_amount_that_is_not_positive_the_run_stops_for_review(self, tmp_path):
        # The model extracts; code validates. A model mistake here must not reach the ledger.
        wf, _, posted = _workflow(tmp_path)
        run = wf.run("INVOICE INV-2043 from Initech. Amount due: -50.00 USD.")
        assert (run.state, run.data["problem"], posted) == ("NEEDS_REVIEW", "amount must be positive, got -50.0", [])

    def test_given_a_crash_while_posting_resuming_finishes_without_asking_the_model_again(self, tmp_path):
        attempts = []

        def flaky_post(invoice):
            attempts.append(invoice)
            if len(attempts) == 1:
                raise ConnectionError("ledger unavailable")

        wf, llm, _ = _workflow(tmp_path, post=flaky_post)
        with pytest.raises(ConnectionError):
            wf.run(SMALL_INVOICE)
        calls_before = llm.calls
        run = wf.run(SMALL_INVOICE)  # a restarted process finds the checkpoint on disk
        assert (run.state, llm.calls, len(attempts)) == ("DONE", calls_before, 2)

    def test_given_a_paused_large_invoice_a_human_approval_posts_it_and_finishes(self, tmp_path):
        wf, _, posted = _workflow(tmp_path)
        wf.run(LARGE_INVOICE)
        run = wf.approve(approver="controller@acme")
        assert (run.state, run.data["approved_by"], len(posted)) == ("DONE", "controller@acme", 1)


class TestAutonomyCosts:
    def test_given_the_same_two_part_question_a_supervisor_needs_four_model_calls_where_simpler_designs_need_two(self):
        # workflow: rewrite + answer; router: classify + handler; agent loop: tool turn + answer;
        # supervisor: plan + one per specialist (2) + final answer.
        from primer.agents.orchestration import autonomy_costs

        assert [row["calls"] for row in autonomy_costs()] == [2, 2, 2, 4]


class TestFigures:
    def test_given_the_lesson_it_draws_autonomy_costs_voting_accuracy_and_the_price_of_resuming(self):
        pytest.importorskip("matplotlib")
        from primer.agents.orchestration import figures

        assert sorted(figures()) == ["autonomy_costs", "resume_cost", "voting"]
