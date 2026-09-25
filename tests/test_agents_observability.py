"""Specification for primer.agents.observability: recording every agent run so you can replay why it did what it did."""

import pytest

from primer.agents.observability import (
    FakeClock,
    Tracer,
    first_error,
    golden_case_from_trace,
    render,
    run_traced_agent,
    to_otel,
    trace_totals,
)


def successful_run():
    return run_traced_agent("Where is order A100?")


def failed_run():
    # The user typed the id with a hyphen; the lookup tool doesn't know it.
    return run_traced_agent("Where is order A-100?")


class TestSpanTree:
    def test_given_a_run_with_two_model_calls_and_one_tool_call_the_root_has_three_children_in_order(self):
        root = successful_run().root
        assert [c.name for c in root.children] == ["chat scripted-large", "execute_tool lookup_order", "chat scripted-large"]

    def test_given_a_child_span_its_parent_is_the_root(self):
        root = successful_run().root
        assert {c.parent_id for c in root.children} == {root.span_id}

    def test_given_code_that_raises_inside_a_span_the_span_is_marked_as_an_error_and_the_exception_still_propagates(self):
        tracer = Tracer(clock=FakeClock())
        with pytest.raises(ValueError):
            with tracer.span("invoke_agent demo", "agent"):
                raise ValueError("boom")
        assert (tracer.root.status, tracer.root.error) == ("error", "boom")


class TestGenAIAttributes:
    def test_given_a_model_call_the_span_records_the_model_and_token_usage(self):
        chat = successful_run().root.children[0]
        assert chat.attributes["gen_ai.request.model"] == "scripted-large"
        assert chat.attributes["gen_ai.usage.input_tokens"] > 0
        assert chat.attributes["gen_ai.usage.output_tokens"] > 0

    def test_given_a_tool_call_the_span_records_the_tool_name_and_call_id(self):
        tool = successful_run().root.children[1]
        assert tool.attributes["gen_ai.tool.name"] == "lookup_order"
        assert tool.attributes["gen_ai.tool.call.id"].startswith("toolu_")

    def test_given_a_model_call_the_span_records_which_prompt_version_produced_it(self):
        assert successful_run().root.children[0].attributes["app.prompt.version"] == "support-v3"


class TestTotals:
    def test_given_two_400ms_model_calls_and_one_100ms_tool_call_the_run_takes_900ms(self):
        assert trace_totals(successful_run().root)["duration_ms"] == 900

    def test_given_the_run_the_totals_count_two_model_calls_and_one_tool_call(self):
        totals = trace_totals(successful_run().root)
        assert (totals["llm_calls"], totals["tool_calls"]) == (2, 1)


class TestDebuggingFromATrace:
    def test_given_a_run_where_the_lookup_failed_the_first_error_is_the_tool_span(self):
        span = first_error(failed_run().root)
        assert span.name == "execute_tool lookup_order"
        assert span.error == "unknown order id A-100"

    def test_given_a_successful_run_there_is_no_error_span(self):
        assert first_error(successful_run().root) is None


class TestRendering:
    def test_given_a_trace_the_first_line_is_the_root_with_its_duration(self):
        assert render(successful_run().root).splitlines()[0].startswith("invoke_agent support-agent  900 ms  [ok]")

    def test_given_a_trace_children_are_drawn_as_branches_with_the_last_one_closing_the_tree(self):
        lines = render(successful_run().root).splitlines()
        assert lines[1].startswith("├─ chat")
        assert lines[-1].startswith("└─ chat")


class TestOpenTelemetryExport:
    def test_given_an_export_every_span_shares_the_trace_id(self):
        spans = to_otel(successful_run())
        assert len({s["traceId"] for s in spans}) == 1

    def test_given_an_export_children_point_at_the_root_span(self):
        spans = to_otel(successful_run())
        root = spans[0]
        assert all(s["parentSpanId"] == root["spanId"] for s in spans[1:])

    def test_given_an_export_timestamps_are_in_nanoseconds(self):
        # OTLP uses Unix nanoseconds; the fake clock starts at 0 ms, so the first model call ends at 400 ms.
        first_chat = to_otel(successful_run())[1]
        assert first_chat["endTimeUnixNano"] - first_chat["startTimeUnixNano"] == 400_000_000


class TestFeedbackToGoldenSet:
    def test_given_thumbs_down_on_a_failed_trace_the_new_golden_case_carries_the_question_and_the_error(self):
        tracer = failed_run()
        case = golden_case_from_trace(tracer, feedback="thumbs_down")
        assert case["id"] == f"prod-{tracer.trace_id}"
        assert case["input"] == "Where is order A-100?"
        assert case["observed_error"] == "unknown order id A-100"
