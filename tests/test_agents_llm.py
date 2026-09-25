"""Specification for primer.agents.llm: what a model call is, and what tool calling really is."""

import pytest

from primer.agents.llm import (
    ScriptedLLM,
    ToolCall,
    estimate_tokens,
    last_user_text,
    tool_result_block,
    tool_results,
    trace_tool_call_round_trip,
)


def ask(llm, text="What's the weather in Paris?", tools=None):
    return llm.complete(system="You are helpful.", messages=[{"role": "user", "content": text}], tools=tools)


class TestAModelReply:
    def test_given_a_policy_that_answers_the_reply_ends_the_turn_with_text(self):
        reply = ask(ScriptedLLM(lambda s, m, t: "Sunny."))
        assert (reply.stop_reason, reply.text, reply.tool_calls) == ("end_turn", "Sunny.", [])

    def test_given_a_policy_that_wants_a_tool_the_reply_is_a_request_not_a_result(self):
        # The model never runs anything: it hands back a filled-in request form.
        reply = ask(ScriptedLLM(lambda s, m, t: ToolCall("", "get_weather", {"city": "Paris"})))
        assert reply.stop_reason == "tool_use"
        assert reply.tool_calls[0].name == "get_weather" and reply.tool_calls[0].input == {"city": "Paris"}

    def test_given_a_tool_request_without_an_id_the_model_assigns_one_so_results_can_be_matched(self):
        reply = ask(ScriptedLLM(lambda s, m, t: ToolCall("", "get_weather", {})))
        assert reply.tool_calls[0].id.startswith("toolu_")

    def test_given_a_fixed_script_each_call_returns_the_next_line_then_repeats_the_last(self):
        llm = ScriptedLLM(["one", "two"])
        assert [ask(llm).text for _ in range(3)] == ["one", "two", "two"]

    def test_given_the_reply_the_assistant_content_holds_the_exact_blocks_to_send_back(self):
        reply = ask(ScriptedLLM(lambda s, m, t: ("Checking.", [ToolCall("t1", "get_weather", {"city": "Paris"})])))
        assert [b["type"] for b in reply.assistant_content] == ["text", "tool_use"]


class TestTokens:
    def test_given_4000_characters_of_prose_the_estimate_is_about_1000_tokens(self):
        assert estimate_tokens("x" * 4000) == 1000

    def test_given_1500_more_characters_of_history_the_next_call_bills_375_more_input_tokens(self):
        # Input tokens are billed per call on the whole history, which is why long agent loops get expensive.
        llm = ScriptedLLM(lambda s, m, t: "ok")
        short = llm.complete(system="", messages=[{"role": "user", "content": "hi"}]).usage.input_tokens
        long = llm.complete(system="", messages=[{"role": "user", "content": "hi" + "x" * 1500}]).usage.input_tokens
        assert long - short == 375

    def test_given_a_ten_call_loop_starting_at_2000_tokens_and_adding_500_per_step_it_bills_42500(self):
        from primer.agents.llm import loop_input_tokens

        assert sum(loop_input_tokens(10, 2000, 500)) == 42_500

    def test_given_the_same_loop_the_tenth_call_re_sends_6500_tokens_of_history(self):
        # 2,000 to start plus nine earlier steps of 500; the tenth step's own 500 then makes the 7,000-token conversation.
        from primer.agents.llm import loop_input_tokens

        assert loop_input_tokens(10, 2000, 500)[-1] == 6_500


class TestReadingAConversation:
    CONVERSATION = [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Paris"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "18°C, sunny"}]},
    ]

    def test_given_tool_results_in_the_latest_user_turn_the_last_user_text_is_still_the_question(self):
        assert last_user_text(self.CONVERSATION) == "Weather in Paris?"

    def test_given_a_conversation_its_tool_results_are_found_in_order(self):
        assert [r["content"] for r in tool_results(self.CONVERSATION)] == ["18°C, sunny"]

    def test_given_a_failed_tool_the_result_is_flagged_as_an_error_the_model_can_act_on(self):
        block = tool_result_block("t1", "city must be a name, got 42", is_error=True)
        assert block == {"type": "tool_result", "tool_use_id": "t1", "content": "city must be a name, got 42", "is_error": True}


class TestTheToolCallRoundTrip:
    def test_given_a_weather_question_the_round_trip_takes_two_model_calls_and_one_tool_run(self):
        transcript = trace_tool_call_round_trip()
        roles = [m["role"] for m in transcript]
        assert roles == ["user", "assistant", "user", "assistant"]
        assert transcript[2]["content"][0]["type"] == "tool_result"
        assert "18" in transcript[3]["content"][0]["text"]

    def test_given_the_round_trip_the_tool_result_answers_the_request_with_the_same_id(self):
        transcript = trace_tool_call_round_trip()
        assert transcript[1]["content"][-1]["id"] == transcript[2]["content"][0]["tool_use_id"]


class TestTheRealRequest:
    # Built by a pure function so the exact API request is testable without a network call.
    def test_given_low_effort_the_request_asks_the_model_to_think_briefly(self):
        from primer.agents.llm import claude_request

        req = claude_request(model="claude-opus-5", system="s", messages=[], tools=None, max_tokens=300, effort="low", fallbacks=False)
        assert req["extra_body"] == {"output_config": {"effort": "low"}}

    def test_given_no_effort_the_request_leaves_the_models_default(self):
        from primer.agents.llm import claude_request

        req = claude_request(model="claude-opus-5", system="s", messages=[], tools=None, max_tokens=300, effort=None, fallbacks=False)
        assert "extra_body" not in req

    def test_given_fallbacks_and_effort_the_request_carries_both(self):
        from primer.agents.llm import claude_request

        req = claude_request(model="claude-opus-5", system="s", messages=[], tools=None, max_tokens=300, effort="low", fallbacks=True)
        assert req["extra_body"] == {"fallbacks": "default", "output_config": {"effort": "low"}}
        assert req["extra_headers"] == {"anthropic-beta": "server-side-fallback-2026-07-01"}


class TestALocalModelThroughOllama:
    # Ollama serves open models (here Qwen) on your own machine; these specs check the
    # exact request and reply handling without needing Ollama to be running.
    def test_given_a_conversation_the_request_sends_the_system_prompt_first_and_turns_thinking_off(self):
        from primer.agents.llm import ollama_request

        req = ollama_request(model="qwen3:1.7b", system="Be brief.", messages=[{"role": "user", "content": "hi"}], max_tokens=160)
        assert req["messages"] == [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}]
        # A "thinking" model would otherwise spend its whole budget reasoning out loud.
        assert req["think"] is False and req["stream"] is False
        assert req["options"]["num_predict"] == 160

    def test_given_content_blocks_they_are_flattened_to_the_plain_text_ollama_expects(self):
        from primer.agents.llm import ollama_request

        msgs = [{"role": "user", "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]}]
        assert ollama_request(model="m", system="", messages=msgs, max_tokens=10)["messages"][-1] == {"role": "user", "content": "one\ntwo"}

    def test_given_ollamas_reply_the_text_and_token_counts_come_through(self):
        from primer.agents.llm import parse_ollama_reply

        reply = parse_ollama_reply({"model": "qwen3:1.7b", "message": {"content": " Two sentences. "}, "done_reason": "stop", "prompt_eval_count": 770, "eval_count": 54})
        assert (reply.text, reply.stop_reason, reply.usage.input_tokens, reply.usage.output_tokens) == ("Two sentences.", "end_turn", 770, 54)

    def test_given_the_reply_hit_the_length_limit_the_stop_reason_says_so(self):
        from primer.agents.llm import parse_ollama_reply

        assert parse_ollama_reply({"model": "m", "message": {"content": "cut"}, "done_reason": "length"}).stop_reason == "max_tokens"
