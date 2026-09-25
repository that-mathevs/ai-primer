"""Specification for primer.agents.evals: turning "it seems better" into numbers you can gate a release on."""

import pytest

from primer.agents.evals import (
    GOLDEN_TASKS,
    Run,
    Signal,
    always_pass_judge,
    calibrate_judge,
    cohens_kappa,
    compare_versions,
    evaluate,
    exact_match,
    faithfulness,
    grade_run,
    implicit_dissatisfaction_rate,
    percentile,
    promote_to_golden,
    recall_at_k,
    rubric_judge,
    tool_selection_accuracy,
)

TASK_BY_ID = {t.id: t for t in GOLDEN_TASKS}


class TestCodeGraders:
    def test_given_answers_differing_only_in_case_and_spacing_exact_match_passes(self):
        assert exact_match("  Shipped ", "shipped")

    def test_given_different_answers_exact_match_fails(self):
        assert not exact_match("shipped", "delivered")


class TestGradingOutcomesNotPaths:
    def test_given_the_right_end_state_reached_by_a_different_tool_order_the_run_passes(self):
        # Refund task: the grader checks the database, not the exact sequence of calls.
        task = TASK_BY_ID["refund-small"]
        run = Run(answer="Refunded A200.", trajectory=[("refund", {"order_id": "A200", "amount": 40}), ("lookup_order", {"order_id": "A200"})],
                  end_state={"A200": "refunded"})
        assert grade_run(task, run).passed

    def test_given_a_correct_answer_but_a_forbidden_tool_call_the_run_fails(self):
        task = TASK_BY_ID["status"]
        run = Run(answer="Order A100 has shipped.", trajectory=[("lookup_order", {"order_id": "A100"}), ("refund", {"order_id": "A100", "amount": 25})],
                  end_state={"A100": "refunded"})
        assert not grade_run(task, run).passed

    def test_given_a_refund_above_the_limit_escalating_instead_of_refunding_passes(self):
        task = TASK_BY_ID["refund-over-limit"]
        run = Run(answer="I've escalated A300 to a supervisor.", trajectory=[("escalate", {"order_id": "A300", "reason": "over limit"})],
                  end_state={"A300": "escalated"})
        assert grade_run(task, run).passed

    def test_given_a_run_that_exceeds_the_step_limit_it_fails_even_if_the_outcome_is_right(self):
        task = TASK_BY_ID["status"]
        run = Run(answer="Order A100 has shipped.", trajectory=[("lookup_order", {"order_id": "A100"})] * 9, end_state={})
        assert not grade_run(task, run).passed


class TestTrajectoryMetrics:
    def test_given_two_required_tools_and_only_one_called_tool_selection_accuracy_is_one_half(self):
        assert tool_selection_accuracy(required={"lookup_order", "refund"}, called=["lookup_order"]) == 0.5

    def test_given_all_required_tools_called_tool_selection_accuracy_is_one(self):
        assert tool_selection_accuracy(required={"lookup_order"}, called=["lookup_order", "lookup_order"]) == 1.0


class TestJudgeCalibration:
    # Worked example: 20 answers. Both say pass on 10, both fail on 6, they disagree on 4.
    HUMAN = [1] * 10 + [0] * 6 + [1] * 2 + [0] * 2
    JUDGE = [1] * 10 + [0] * 6 + [0] * 2 + [1] * 2

    def test_given_the_worked_example_raw_agreement_is_eighty_percent(self):
        assert calibrate_judge(self.HUMAN, self.JUDGE)["agreement"] == pytest.approx(0.80)

    def test_given_the_worked_example_cohens_kappa_is_0_583(self):
        # p_o = 0.80, p_e = 0.6·0.6 + 0.4·0.4 = 0.52, kappa = 0.28 / 0.48.
        assert cohens_kappa(self.HUMAN, self.JUDGE) == pytest.approx(0.5833, abs=1e-4)

    def test_given_a_judge_that_always_says_pass_kappa_is_zero_despite_sixty_percent_agreement(self):
        # Agreement you'd get by chance is not skill.
        lazy = [1] * 20
        assert calibrate_judge(self.HUMAN, lazy)["agreement"] == pytest.approx(0.60)
        assert cohens_kappa(self.HUMAN, lazy) == pytest.approx(0.0)

    def test_given_kappa_below_0_6_the_judge_is_flagged_as_not_trustworthy(self):
        assert not calibrate_judge(self.HUMAN, self.JUDGE)["trusted"]

    def test_given_the_rubric_judge_an_answer_naming_the_order_and_its_status_passes(self):
        assert rubric_judge("Where is order A100?", "Order A100 has shipped and arrives Friday.")

    def test_given_the_rubric_judge_a_vague_answer_fails(self):
        assert not rubric_judge("Where is order A100?", "It's on its way, don't worry!")

    def test_given_the_always_pass_judge_even_a_vague_answer_passes(self):
        assert always_pass_judge("Where is order A100?", "It's on its way, don't worry!")


class TestRetrievalAndFaithfulness:
    def test_given_the_relevant_document_at_rank_three_recall_at_three_is_one(self):
        assert recall_at_k(["d9", "d4", "d1"], relevant={"d1"}, k=3) == 1.0

    def test_given_the_relevant_document_at_rank_three_recall_at_two_is_zero(self):
        assert recall_at_k(["d9", "d4", "d1"], relevant={"d1"}, k=2) == 0.0

    def test_given_one_invented_claim_out_of_two_faithfulness_is_one_half(self):
        sources = ["Full-time employees accrue 20 days of PTO per year."]
        assert faithfulness("Employees accrue 20 days of PTO per year. Managers get unlimited sabbaticals.", sources) == 0.5


class TestPercentiles:
    def test_given_latencies_one_to_one_hundred_the_p95_is_95(self):
        # Nearest-rank method: the 95th of 100 sorted values.
        assert percentile(list(range(1, 101)), 95) == 95


class TestRegressionGate:
    def test_given_the_same_version_twice_the_release_gate_passes(self):
        assert compare_versions(evaluate("v1"), evaluate("v1"))["ship"]

    def test_given_a_prompt_change_that_breaks_the_over_limit_rule_the_gate_blocks_the_release(self):
        decision = compare_versions(evaluate("v1"), evaluate("v2"))
        assert not decision["ship"]
        assert decision["regressed_tasks"] == ["refund-over-limit"]

    def test_given_version_one_every_golden_task_passes(self):
        assert evaluate("v1")["success_rate"] == 1.0

    def test_given_an_evaluation_cost_is_reported_per_successful_task(self):
        report = evaluate("v1")
        assert report["cost_per_success"] == pytest.approx(report["total_cost"] / report["successes"])


class TestOnlineSignals:
    def test_given_a_rephrase_a_retry_and_a_thumbs_up_two_of_three_sessions_look_unhappy(self):
        signals = [Signal("s1", "rephrase"), Signal("s2", "retry"), Signal("s3", "thumbs_up")]
        assert implicit_dissatisfaction_rate(signals) == pytest.approx(2 / 3)

    def test_given_a_flagged_production_failure_it_becomes_a_new_golden_task(self):
        golden = list(GOLDEN_TASKS)
        promote_to_golden(golden, trace_id="tr-42", user_input="Refund A300 please, it's broken", expected_outcome={"A300": "escalated"})
        assert golden[-1].id == "prod-tr-42"
        assert len(golden) == len(GOLDEN_TASKS) + 1
