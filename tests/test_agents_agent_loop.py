"""Specification for primer.agents.agent_loop: a loop that finishes, or says exactly why it stopped."""

import json
import pytest
import threading

from primer.agents.agent_loop import (
    DEMO_TOOL_DEFS,
    DEMO_TOOLS,
    HANDOFF_TOOL,
    AgentConfig,
    ToolError,
    happy_policy,
    recovering_policy,
    run_agent,
)
from primer.agents.llm import LLMResponse, ScriptedLLM, ToolCall, Usage


def _run(policy, task="task", tools=DEMO_TOOLS, config=None):
    return run_agent(ScriptedLLM(policy), task, tools, DEMO_TOOL_DEFS, config=config)


class TestCompletingATask:
    def test_given_a_model_that_answers_after_one_round_of_tools_the_run_completes_with_that_answer(self):
        result = _run(happy_policy)
        assert result.stop_cause == "completed"
        assert result.final_text == "Alice has 12 PTO days left. Up to 5 unused days roll over to next year [hr-001]."

    def test_given_two_lookups_requested_in_one_turn_both_results_return_in_a_single_user_message(self):
        # The API pairs tool_use and tool_result by id in the very next user turn.
        transcript = _run(happy_policy).transcript
        assert [b["type"] for b in transcript[2]["content"]] == ["tool_result", "tool_result"]

    def test_given_two_tools_that_each_wait_for_the_other_they_both_succeed_because_they_run_concurrently(self):
        # Run one after the other, the first would wait forever at the barrier and time out.
        barrier = threading.Barrier(2, timeout=2)

        def wait_for_partner(name):
            barrier.wait()
            return f"{name} ok"

        tools = {"a": wait_for_partner, "b": wait_for_partner}
        script = [[ToolCall("", "a", {"name": "a"}), ToolCall("", "b", {"name": "b"})], "done"]
        result = _run(script, tools=tools)
        assert [e.content for e in result.steps[0].tool_executions] == ["a ok", "b ok"]


class TestRunawayControls:
    def test_given_a_model_that_repeats_the_same_tool_call_the_loop_stops_it_as_a_loop(self):
        result = _run(lambda s, m, t: ToolCall("", "search_kb", {"query": "PTO rollover"}))
        assert result.stop_cause == "loop_detected"
        # Default threshold is 3: two searches run, the third identical request is refused.
        assert result.tool_call_count == 2

    def test_given_the_same_arguments_in_a_different_key_order_the_calls_count_as_one_repeated_call(self):
        script = [
            ToolCall("", "get_pto_balance", {"employee": "bob", "as_of": "2026-10-02"}),
            ToolCall("", "get_pto_balance", {"as_of": "2026-10-02", "employee": "bob"}),
            ToolCall("", "get_pto_balance", {"employee": "bob", "as_of": "2026-10-02"}),
        ]
        assert _run(script).stop_cause == "loop_detected"

    def test_given_a_model_that_never_finishes_the_run_stops_at_the_step_limit(self):
        policy = lambda s, m, t: ToolCall("", "search_kb", {"query": f"page {len(m)}"})  # noqa: E731
        result = _run(policy, config=AgentConfig(max_steps=4))
        assert (result.stop_cause, len(result.steps)) == ("max_steps", 4)

    def test_given_a_token_budget_a_run_that_exceeds_it_stops_as_budget_exceeded(self):
        policy = lambda s, m, t: ToolCall("", "search_kb", {"query": f"page {len(m)}"})  # noqa: E731
        result = _run(policy, config=AgentConfig(max_steps=50, max_total_tokens=3_000))
        assert result.stop_cause == "budget_exceeded"

    def test_given_a_dollar_budget_a_run_that_exceeds_it_stops_as_budget_exceeded(self):
        policy = lambda s, m, t: ToolCall("", "search_kb", {"query": f"page {len(m)}"})  # noqa: E731
        result = _run(policy, config=AgentConfig(max_steps=50, max_total_tokens=10**9, max_cost_usd=0.01))
        assert result.stop_cause == "budget_exceeded"
        assert result.cost_usd > 0.01

    def test_when_every_call_resends_the_history_each_step_sends_more_input_tokens_than_the_last(self):
        policy = lambda s, m, t: ToolCall("", "search_kb", {"query": f"page {len(m)}"})  # noqa: E731
        inputs = [s.usage.input_tokens for s in _run(policy, config=AgentConfig(max_steps=5)).steps]
        assert inputs == sorted(inputs) and len(set(inputs)) == 5


class TestToolErrors:
    def test_given_a_badly_formatted_date_the_model_receives_an_error_that_shows_the_valid_format(self):
        first = _run(recovering_policy).steps[0].tool_executions[0]
        assert first.is_error
        assert "YYYY-MM-DD" in first.content

    def test_given_an_actionable_error_the_model_corrects_its_call_and_completes(self):
        result = _run(recovering_policy)
        assert result.stop_cause == "completed"
        assert result.final_text == "Bob will have 3 PTO days left on 2026-10-02."

    def test_given_a_tool_name_the_model_invented_it_is_told_which_tools_exist(self):
        result = _run([ToolCall("", "lookup_everything", {}), "done"])
        execution = result.steps[0].tool_executions[0]
        assert execution.is_error
        assert "get_pto_balance" in execution.content and "search_kb" in execution.content

    def test_given_a_tool_that_crashes_the_run_continues_with_an_error_result(self):
        def broken():
            raise ConnectionError("upstream timed out")

        result = _run([ToolCall("", "broken", {}), "Sorry, the system is down."], tools={"broken": broken})
        assert result.stop_cause == "completed"
        assert result.steps[0].tool_executions[0].content == "broken failed (ConnectionError): upstream timed out"

    def test_given_a_tool_that_raises_a_tool_error_the_model_sees_its_message_verbatim(self):
        def strict():
            raise ToolError("amount must be positive, got -5")

        result = _run([ToolCall("", "strict", {}), "ok"], tools={"strict": strict})
        assert result.steps[0].tool_executions[0].content == "amount must be positive, got -5"


class TestStopReasons:
    def test_given_a_refusal_the_run_stops_as_a_refusal_without_running_tools(self):
        refusal = LLMResponse("", [], "refusal", Usage(10, 0), "m", [])
        assert _run([refusal]).stop_cause == "refusal"

    def test_given_a_truncated_response_its_half_written_tool_call_is_not_run(self):
        # A response cut off by max_tokens may hold incomplete arguments.
        truncated = LLMResponse("", [ToolCall("t1", "search_kb", {"query": "PT"})], "max_tokens", Usage(10, 4096), "m", [])
        result = _run([truncated])
        assert result.stop_cause == "max_tokens"
        assert result.tool_call_count == 0


class TestHandingOffToAHuman:
    def test_given_a_request_outside_the_agents_authority_the_run_ends_as_a_handoff_with_the_reason(self):
        result = _run([ToolCall("", HANDOFF_TOOL, {"reason": "Refunds over $500 need a finance approver."})])
        assert (result.stop_cause, result.final_text) == ("handoff", "Refunds over $500 need a finance approver.")


class TestCost:
    def test_given_one_million_input_and_100k_output_tokens_at_5_and_25_dollars_the_cost_is_7_50(self):
        # $5 per million input + $25 per million output: 5.00 + 2.50.
        assert AgentConfig(price_in_per_mtok=5, price_out_per_mtok=25).cost(Usage(1_000_000, 100_000)) == 7.5

    def test_given_a_tool_result_the_transcript_stores_it_as_json_the_model_can_read(self):
        content = _run(happy_policy).transcript[2]["content"][1]["content"]
        assert json.loads(content) == {"employee": "alice", "as_of": "2026-09-25", "days_left": 12}


class TestStopCauses:
    def test_given_the_controls_diagram_every_one_of_its_seven_exits_is_a_named_stop_cause(self):
        # The diagram's seven "stop:" boxes, in its order; a run never ends without saying which one.
        from typing import get_args

        from primer.agents.agent_loop import StopCause

        assert set(get_args(StopCause)) == {
            "refusal", "max_tokens", "completed", "budget_exceeded", "handoff", "loop_detected", "max_steps",
        }


class TestFigures:
    def test_given_the_lesson_it_draws_token_growth_quadratic_cost_and_parallel_latency(self):
        pytest.importorskip("matplotlib")
        from primer.agents.agent_loop import figures

        assert sorted(figures()) == ["parallel_latency", "quadratic_cost", "tokens_per_step"]


class TestQuadraticGrowth:
    def test_given_ten_steps_adding_500_tokens_each_the_run_sends_27500_input_tokens_not_5000(self):
        # 500 · (1 + 2 + ... + 10) = 500 · 55.
        from primer.agents.agent_loop import cumulative_input_tokens

        assert cumulative_input_tokens(10, 500) == 27_500
