r"""
# Talking to a model: messages, and what tool calling really is

Run: `python -m primer.agents.llm`

Every applied-AI system in this primer, from a one-shot classifier to a
multi-agent research team, talks to the model the same way: it sends a list
of **messages** and gets one message back. This lesson shows that exchange
exactly, and then the one feature that turns a chatbot into an agent: **tool
calling**.

## The everyday picture

Think of the model as a brilliant consultant who works by post. You mail a
letter with your whole conversation so far (the consultant keeps no notes
between letters) and get one letter back. The reply is either an answer, or
a **filled-in request form**: "please look up the weather in Paris and send me
the result." The consultant never picks up the phone themselves. *You* decide
whether to run the request, run it, and mail back the result, together with
the whole conversation again.

Two facts fall straight out of this picture, and they explain most of the
behaviour of real systems:

1. **The model executes nothing.** A tool call is a request. Your code is
   the only thing that acts, so your code is where safety lives.
2. **Every call resends everything.** The model has no memory between calls,
   so each request carries the full history. Long conversations cost more on
   every single turn.

## A tiny worked example: one tool call, traced

A user asks "What's the weather in Paris?" and the application offers one
tool, `get_weather`. Four messages later, the user has an answer:

| # | Role | Content | Who produced it |
|---|---|---|---|
| 1 | user | "What's the weather in Paris?" | the user |
| 2 | assistant | `tool_use` id=`toolu_1`, name=`get_weather`, input=`{"city": "Paris"}` | the model (1st call) |
| 3 | user | `tool_result` for `toolu_1`: "18°C, sunny" | **your code**, after running the tool |
| 4 | assistant | "It's 18°C and sunny in Paris." | the model (2nd call) |

Message 3 has role `user` even though no human typed it: tool results always
travel back in the user's turn. The `id` in message 2 and the `tool_use_id`
in message 3 match, which is how the model knows which request a result
answers when it asked for several at once. `trace_tool_call_round_trip()`
produces exactly this transcript.

```mermaid
sequenceDiagram
  participant U as User
  participant A as Your code
  participant M as Model
  participant T as get_weather tool
  U->>A: What's the weather in Paris?
  A->>M: messages [1] + tool definitions
  M-->>A: tool_use get_weather {city: Paris}  (stop_reason: tool_use)
  A->>A: validate the arguments, check permissions
  A->>T: get_weather("Paris")
  T-->>A: 18°C, sunny
  A->>M: messages [1, 2, 3 = tool_result]
  M-->>A: It's 18°C and sunny in Paris.  (stop_reason: end_turn)
  A-->>U: It's 18°C and sunny in Paris.
```

**Reading it:** time runs downwards. Solid arrows are requests and dashed
arrows are replies. Notice that the model never talks to the tool: every
arrow into `get_weather` starts at *your code*, and the self-arrow on your
code ("validate, check permissions") is where production systems put their
guardrails. Notice too that the second call to the model carries messages
1 to 3, not just the new result: the model is stateless, so the history
travels every time.

**In code:** `ToolCall` holds the request in message 2, and
`tool_result_block` builds the reply in message 3, carrying the matching id.

## The message format

Messages use the Anthropic Messages API shape directly, so what you learn
here maps one-to-one onto production code:

```python
{"role": "user", "content": "What's our PTO policy?"}
{"role": "assistant", "content": [
    {"type": "text", "text": "Let me look that up."},
    {"type": "tool_use", "id": "toolu_1", "name": "search_kb", "input": {"query": "PTO"}},
]}
{"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "hr-001: ..."},
]}
```

| Field | Meaning |
|---|---|
| `role` | `user` (the human, or your code returning tool results) or `assistant` (the model) |
| `content` | a plain string, or a list of **content blocks** |
| `text` block | ordinary words |
| `tool_use` block | the model asking for a tool: `id`, `name`, and `input` (a JSON object matching the tool's schema) |
| `tool_result` block | your answer to one `tool_use`, matched by `tool_use_id`; set `is_error: true` when the tool failed |
| `stop_reason` | why the reply ended: `end_turn` (finished), `tool_use` (wants a tool), `max_tokens` (ran out of room), `refusal` (declined) |

A **tool definition** is a name, a description and a JSON Schema for the
input. The model chooses tools by reading those descriptions, so writing
them well is prompt engineering (see `primer.agents.tools`).

**In code:** `LLMResponse` is one reply, normalized: its text, its
`ToolCall` list, its stop reason, its `Usage` and the exact content blocks
to append as the assistant turn. `content_blocks`, `last_user_text`,
`tool_results` and `tool_calls_so_far` read a conversation in this format.

## Two implementations of one interface

Every agent lesson is written against one small interface, `LLM`, with one
method, `LLM.complete(system=..., messages=..., tools=...)`. Two classes
implement it:

```mermaid
flowchart LR
  L["LLM interface<br/>complete(system, messages, tools)"]
  L --> S["ScriptedLLM<br/>a Python function decides the reply<br/>offline, instant, deterministic"]
  L --> C["ClaudeLLM<br/>the real model via the anthropic SDK<br/>needs an API key"]
  S --> D["demos and tests<br/>reproduce any failure on purpose"]
  C --> P["production<br/>same agent code, real behaviour"]
```

**Reading it:** the agent code on the right-hand side never knows which
box it's talking to. `ScriptedLLM` lets every lesson run offline and
deterministically, and lets tests stage a model that loops, calls the wrong
tool or follows an injected instruction, which is hard to get a real model
to do on cue. Swapping in `ClaudeLLM` runs the identical loop against the
real thing.

**In code:** `ClaudeLLM.complete` builds its request with `claude_request`
and turns the API's reply into an `LLMResponse`. `OllamaLLM` is a third
implementation for a local open model (text only), using `ollama_request`
and `parse_ollama_reply`.

## Cost: why every call pays for the whole conversation

Because the model is stateless, input tokens are billed per call on the
entire history. In an agent loop with $n$ calls, where each call adds about
$t$ new tokens to a history that started at $h_0$ tokens, the total input
billed is:

$$
\text{input tokens} \approx \sum_{i=1}^{n} \left(h_0 + (i-1)\,t\right) = n\,h_0 + t\,\frac{n(n-1)}{2}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $n$ | number of model calls in the loop | 10 |
| $h_0$ | tokens in the first request (system prompt, tools, question) | 2,000 |
| $t$ | tokens each step adds (the model's tool call plus the tool's result) | 500 |
| $i$ | which call we're on, 1 to $n$ | |
| $\sum_{i=1}^{n}$ | add up the cost of every call | |
| $\frac{n(n-1)}{2}$ | 0 + 1 + … + (n − 1): how many "steps of history" pile up in total | 45 |

**In words:** "each call pays for the starting prompt plus everything added
so far, so the history term grows with the square of the number of steps."

**With the numbers:** 10 × 2,000 + 500 × 45 = 20,000 + 22,500 = **42,500 input
tokens** for a task whose final conversation is only 6,500 tokens long.

**In Python:**

```python
>>> n, h_0, t = 10, 2000, 500
>>> calls = [h_0 + (i - 1) * t for i in range(1, n + 1)]   # call i re-sends h_0 and i - 1 steps
>>> calls[0], calls[-1]
(2000, 6500)
>>> sum(calls)                                            # Σ over every call
42500
>>> n * h_0 + t * n * (n - 1) // 2                         # the shortcut on the right agrees
42500
```

![Input tokens per call and in total across an agent loop](figures/primer.agents.llm.loop_tokens.svg)

**Reading it:** the bars are the input tokens billed on each call. They grow
by the same amount every step, because each call re-sends the whole history.
The line is the running total, and it curves upwards (quadratic growth). The
dashed line is what you might naively expect: the size of the final
conversation, billed once. The gap between the dashed line and the curve is
why prompt caching, trimming tool outputs and keeping loops short are the
main cost levers (`primer.agents.cost`, `primer.agents.context`).

**In code:** `loop_input_tokens` lists the tokens billed on each call of
such a loop. `estimate_tokens` is the four-characters-per-token rule of
thumb, and `conversation_chars` measures everything a call resends, which is
how `ScriptedLLM` gives its fake replies realistic `Usage`.

## In 20 seconds

- A model call sends the full list of messages and returns one message; the
  model keeps no state between calls.
- A tool call is a structured *request* (`tool_use`). Your code validates it,
  runs it, and returns a `tool_result` with the matching id. The model
  executes nothing.
- `stop_reason` tells you what to do next: `tool_use` means run tools and
  call again, `end_turn` means done.
- Every call re-sends the whole history, so input cost grows quadratically
  with the number of steps in a loop.

## Self-test questions

**Walk through exactly what happens when a model "uses a tool".**
You send tool definitions (name, description, JSON Schema) with the
messages. The model replies with a `tool_use` block and `stop_reason:
tool_use`. Your code validates the arguments, runs the function, and sends a
new request with the history plus a `tool_result` block carrying the same
id. The model then answers or asks for another tool.

**Why is the model never a security boundary for tool use?**
It only produces requests. Everything that actually happens goes through
your code, so validation, permissions, approvals and rate limits all belong
there, and a manipulated model can only ever ask.

**The model asks for three tools in one reply. How do you send the results?**
Run them (concurrently if independent) and return all three `tool_result`
blocks in a single user message, each with its own `tool_use_id`. For one
that failed, return a `tool_result` with `is_error: true` and an actionable
message rather than dropping it.

**A 20-step agent loop costs far more than 20 times a single call. Why?**
Each call re-sends the full, growing history, so the input billed is a sum
that grows with the square of the number of steps. Caching the stable prefix
and trimming tool outputs attack exactly this.

**Why test agents against a scripted model at all?**
Real models are non-deterministic and rarely misbehave on cue. A scripted
model reproduces loops, bad arguments and injected instructions exactly, so
the code that must handle them can be tested every time.

## The papers behind this lesson

- **Schick et al., *Toolformer: Language Models Can Teach Themselves to Use
  Tools* (2023)**: https://arxiv.org/abs/2302.04761. Showed a model can learn
  when to call external tools and how to use their results.
  [Annotated companion](../../papers/toolformer.html)
- **Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*
  (2022)**: https://arxiv.org/abs/2210.03629. The pattern of interleaving
  reasoning with tool calls that modern agent loops descend from.
  [Annotated companion](../../papers/react.html)

## Further reading

- Tool use overview: https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview
- Implementing tool use: https://docs.claude.com/en/docs/agents-and-tools/tool-use/implement-tool-use
- Messages API reference: https://docs.claude.com/en/api/messages
- Python SDK: https://github.com/anthropics/anthropic-sdk-python
- Anthropic, *Building effective agents*: https://www.anthropic.com/engineering/building-effective-agents
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal"]

# Default model for the real adapter. Opus 5 is the current default Claude
# model for new code; pass `model=` to use another (e.g. "claude-haiku-4-5"
# for cheap routing/classification steps; see primer.agents.cost).
DEFAULT_MODEL = "claude-opus-5"


def estimate_tokens(text: str) -> int:
    """Rule-of-thumb token count: ~4 characters per token for English prose.

    Good enough for budgeting demos. For real numbers, use the provider's
    token-counting endpoint, since vocabularies differ between models.
    https://docs.claude.com/en/docs/build-with-claude/token-counting
    """
    return max(1, math.ceil(len(text) / 4))


@dataclass
class ToolCall:
    """One `tool_use` block: the model *asking* to run a tool."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __iadd__(self, other: "Usage") -> "Usage":
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        return self


@dataclass
class LLMResponse:
    """What an `LLM.complete` call returns, normalized across implementations.

    `assistant_content` is the exact list of content blocks to append to the
    conversation as the assistant turn. Always append *that* (not just the
    text): with the real API it can include thinking blocks that must be sent
    back unchanged.
    """

    text: str
    tool_calls: list[ToolCall]
    stop_reason: StopReason
    usage: Usage
    model: str
    assistant_content: list[dict[str, Any]] = field(default_factory=list)


class LLM(Protocol):
    """The one method agents need."""

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse: ...


# ----------------------------------------------------------------------------
# Helpers for reading a conversation (useful inside ScriptedLLM policies)
# ----------------------------------------------------------------------------


def content_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    """A message's content as a list of blocks (a bare string becomes one text block)."""
    c = message["content"]
    return [{"type": "text", "text": c}] if isinstance(c, str) else list(c)


def last_user_text(messages: list[dict[str, Any]]) -> str:
    """Text of the most recent user message that contains text (not just tool results)."""
    for m in reversed(messages):
        if m["role"] != "user":
            continue
        texts = [b["text"] for b in content_blocks(m) if b.get("type") == "text"]
        if texts:
            return "\n".join(texts)
    return ""


def tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """All `tool_result` blocks in the conversation, oldest first."""
    return [b for m in messages if m["role"] == "user" for b in content_blocks(m) if b.get("type") == "tool_result"]


def tool_calls_so_far(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """All `tool_use` blocks the assistant has emitted, oldest first."""
    return [b for m in messages if m["role"] == "assistant" for b in content_blocks(m) if b.get("type") == "tool_use"]


def conversation_chars(system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> int:
    """Rough size of everything sent to the model (for token estimates)."""
    return len(system) + len(json.dumps(messages, default=str)) + len(json.dumps(tools or []))


# ----------------------------------------------------------------------------
# ScriptedLLM: deterministic offline "model"
# ----------------------------------------------------------------------------

# A policy decides the next assistant turn. It returns either:
#   * a string                       -> final text answer (stop_reason end_turn)
#   * a ToolCall or list[ToolCall]   -> tool request(s)   (stop_reason tool_use)
#   * (text, [ToolCall, ...])        -> text plus tool requests
#   * an LLMResponse                 -> used as-is (e.g. to simulate a refusal)
PolicyResult = Any
Policy = Callable[[str, list[dict[str, Any]], list[dict[str, Any]] | None], PolicyResult]


class ScriptedLLM:
    """A fake LLM driven by a Python function, for offline demos and tests.

    Examples:
        >>> def policy(system, messages, tools):
        ...     if not tool_results(messages):
        ...         return ToolCall("t1", "get_time", {})
        ...     return "It is " + tool_results(messages)[-1]["content"]
        >>> llm = ScriptedLLM(policy)
        >>> r = llm.complete(system="", messages=[{"role": "user", "content": "time?"}])
        >>> r.stop_reason, r.tool_calls[0].name
        ('tool_use', 'get_time')

    Token usage is *estimated* from characters so cost demos have realistic
    shapes: input grows with the whole conversation each call, which is the
    single most important cost fact about agent loops.
    """

    def __init__(self, policy: Policy | list[PolicyResult], model: str = "scripted-large"):
        if isinstance(policy, list):
            # A fixed script: return each item in turn, repeating the last one.
            script = list(policy)
            counter = itertools.count()

            def policy_fn(system, messages, tools):  # noqa: ARG001
                return script[min(next(counter), len(script) - 1)]

            self.policy: Policy = policy_fn
        else:
            self.policy = policy
        self.model = model
        self.calls = 0
        self.total_usage = Usage()
        self._ids = itertools.count(1)

    def complete(self, *, system, messages, tools=None, max_tokens=4096) -> LLMResponse:  # noqa: ARG002
        self.calls += 1
        out = self.policy(system, messages, tools)
        if isinstance(out, LLMResponse):
            self.total_usage += out.usage
            return out

        text, calls = "", []
        if isinstance(out, str):
            text = out
        elif isinstance(out, ToolCall):
            calls = [out]
        elif isinstance(out, tuple):
            text, calls = out
        elif isinstance(out, list):
            calls = out
        else:
            raise TypeError(f"policy returned unsupported {type(out).__name__}")

        # Give tool calls unique ids if the policy didn't bother.
        calls = [c if c.id else ToolCall(f"toolu_{next(self._ids)}", c.name, c.input) for c in calls]

        blocks: list[dict[str, Any]] = []
        if text:
            blocks.append({"type": "text", "text": text})
        blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in calls]

        usage = Usage(
            input_tokens=estimate_tokens("x" * conversation_chars(system, messages, tools)),
            output_tokens=estimate_tokens(json.dumps(blocks)),
        )
        self.total_usage += usage
        return LLMResponse(
            text=text,
            tool_calls=calls,
            stop_reason="tool_use" if calls else "end_turn",
            usage=usage,
            model=self.model,
            assistant_content=blocks,
        )


# ----------------------------------------------------------------------------
# ClaudeLLM: the real model via the Anthropic SDK
# ----------------------------------------------------------------------------


class ClaudeLLM:
    """Adapter from the `LLM` interface to the Anthropic Messages API.

    Requires `pip install anthropic` and credentials (`ANTHROPIC_API_KEY`, or
    `ant auth login`). Nothing else in this repo needs it.

    Args:
        model: model id, default `claude-opus-5`.
        fallbacks: when True (default), opts into server-side refusal
            fallbacks, so if a safety classifier declines the request the API
            retries on a fallback model instead of returning
            `stop_reason == "refusal"`. Set False if your account or proxy
            rejects the beta header. We still check for `"refusal"` either way.

    Notes:
        * Adaptive thinking is on by default for Opus 5. The response may
          contain `thinking` blocks; `assistant_content` preserves them so the
          agent loop can send them back unchanged, as the API requires.
        * Tool inputs are parsed JSON dicts, never raw strings. Don't
          string-match serialized input.
    """

    def __init__(self, model: str = DEFAULT_MODEL, fallbacks: bool = True, effort: str | None = None):
        import anthropic  # imported lazily so the rest of the repo has no hard dependency

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.fallbacks = fallbacks
        self.effort = effort

    def complete(self, *, system, messages, tools=None, max_tokens=16000) -> LLMResponse:
        kwargs = claude_request(
            model=self.model, system=system, messages=messages, tools=tools,
            max_tokens=max_tokens, effort=self.effort, fallbacks=self.fallbacks,
        )
        resp = self.client.messages.create(**kwargs)

        blocks = [b.model_dump(exclude_none=True) for b in resp.content]
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        stop = resp.stop_reason if resp.stop_reason in ("end_turn", "tool_use", "max_tokens", "refusal") else "end_turn"
        return LLMResponse(
            text=text,
            tool_calls=calls,
            stop_reason=stop,  # type: ignore[arg-type]
            usage=Usage(resp.usage.input_tokens, resp.usage.output_tokens),
            model=resp.model,
            assistant_content=blocks,
        )


def claude_request(*, model, system, messages, tools, max_tokens, effort, fallbacks) -> dict[str, Any]:
    """The keyword arguments for `client.messages.create`, built without touching the network.

    `effort` ("low" to "max") trades thinking depth for speed and cost. Low suits
    short, grounded tasks such as a two-sentence explanation, where default-depth
    thinking could spend a small `max_tokens` cap before writing any text.
    """
    kwargs: dict[str, Any] = dict(model=model, max_tokens=max_tokens, system=system, messages=messages)
    if tools:
        kwargs["tools"] = tools
    extra_body: dict[str, Any] = {}
    if fallbacks:
        # Server-side refusal fallback (beta): route by refusal category.
        kwargs["extra_headers"] = {"anthropic-beta": "server-side-fallback-2026-07-01"}
        extra_body["fallbacks"] = "default"
    if effort:
        extra_body["output_config"] = {"effort": effort}
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


# ----------------------------------------------------------------------------
# OllamaLLM: an open model running on your own machine
# ----------------------------------------------------------------------------

OLLAMA_HOST = "http://127.0.0.1:11434"


def _plain_text(content: Any) -> str:
    """Ollama messages carry plain strings; join the text of any content blocks."""
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") or str(b.get("content", "")) for b in content if isinstance(b, dict))


def ollama_request(*, model: str, system: str, messages: list[dict[str, Any]], max_tokens: int,
                   temperature: float = 0.2, keep_alive: str = "30m") -> dict[str, Any]:
    """The JSON body for Ollama's /api/chat, built without touching the network.

    `think: False` matters: some small open models "think out loud" by default and
    would spend the whole token budget reasoning instead of answering.
    `keep_alive` keeps the weights in memory between calls, so only the first
    call pays to load them.
    """
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": m["role"], "content": _plain_text(m["content"])} for m in messages
    ]
    return {
        "model": model,
        "messages": msgs,
        "stream": False,
        "think": False,
        "keep_alive": keep_alive,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }


def parse_ollama_reply(data: dict[str, Any]) -> LLMResponse:
    """Normalize Ollama's /api/chat reply into an `LLMResponse`."""
    text = (data.get("message") or {}).get("content", "").strip()
    stop: StopReason = "max_tokens" if data.get("done_reason") == "length" else "end_turn"
    return LLMResponse(
        text=text,
        tool_calls=[],
        stop_reason=stop,
        usage=Usage(data.get("prompt_eval_count", 0), data.get("eval_count", 0)),
        model=data.get("model", ""),
        assistant_content=[{"type": "text", "text": text}] if text else [],
    )


class OllamaLLM:
    """Adapter from the `LLM` interface to a local model served by Ollama.

    Free, private and offline: nothing leaves your machine. Text only (no tool
    calling here). Start Ollama and download the model first
    (`ollama pull qwen3:1.7b`).
    """

    def __init__(self, model: str = "qwen3:1.7b", host: str = OLLAMA_HOST, temperature: float = 0.2):
        self.model, self.host, self.temperature = model, host, temperature

    def complete(self, *, system, messages, tools=None, max_tokens=300) -> LLMResponse:
        import urllib.request

        if tools:
            raise NotImplementedError("OllamaLLM here handles text only; use ClaudeLLM for tool calling")
        body = ollama_request(model=self.model, system=system, messages=messages,
                              max_tokens=max_tokens, temperature=self.temperature)
        req = urllib.request.Request(f"{self.host}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return parse_ollama_reply(json.load(r))


def tool_result_block(tool_use_id: str, content: str, is_error: bool = False) -> dict[str, Any]:
    """Build a `tool_result` block. On failure, set `is_error=True` with an
    actionable message the model can use to fix its next call."""
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        block["is_error"] = True
    return block


# ----------------------------------------------------------------------------
# The worked example, as code
# ----------------------------------------------------------------------------

WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Current weather for a city. Use for any question about today's weather.",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}


def _get_weather(city: str) -> str:
    return {"paris": "18°C, sunny"}.get(city.lower(), "no data")


def trace_tool_call_round_trip() -> list[dict[str, Any]]:
    """Run the lesson's worked example and return the full transcript (4 messages)."""

    def policy(system, messages, tools):
        results = tool_results(messages)
        if not results:  # first call: ask for the tool
            return ToolCall("", "get_weather", {"city": "Paris"})
        return f"It's {results[-1]['content'].replace(', ', ' and ')} in Paris."

    llm = ScriptedLLM(policy)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "What's the weather in Paris?"}]
    while True:
        reply = llm.complete(system="You are helpful.", messages=messages, tools=[WEATHER_TOOL])
        messages.append({"role": "assistant", "content": reply.assistant_content})
        if reply.stop_reason != "tool_use":
            return messages
        # Your code, not the model, runs the tool.
        messages.append(
            {"role": "user", "content": [tool_result_block(c.id, _get_weather(**c.input)) for c in reply.tool_calls]}
        )


def loop_input_tokens(n_calls: int, first: int, per_step: int) -> list[int]:
    """Input tokens billed on each call of a loop that re-sends its growing history."""
    return [first + i * per_step for i in range(n_calls)]


def figures() -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    per_call = loop_input_tokens(10, 2000, 500)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(1, 11)
    ax.bar(x, per_call, color="#93c5fd", label="input tokens billed on this call")
    ax.plot(x, np.cumsum(per_call), "o-", color="#2563eb", label="running total")
    ax.axhline(per_call[-1] + 500, ls="--", color="#9ca3af", label="final conversation size (billed once?)")
    ax.set_xlabel("model call in the loop")
    ax.set_ylabel("input tokens")
    ax.set_title("Every call re-sends the whole history")
    ax.legend(frameon=False)
    return {"loop_tokens": fig}


def demo() -> None:
    from primer._show import banner, say, table, takeaway

    banner("1. One tool call, traced message by message")
    for i, m in enumerate(trace_tool_call_round_trip(), 1):
        for b in content_blocks(m):
            detail = b.get("text") or (f"{b['name']}({b['input']}) id={b['id']}" if b["type"] == "tool_use" else f"result for {b['tool_use_id']}: {b['content']}")
            print(f"  {i}. {m['role']:9s} {b['type']:11s} {detail}")
    print()
    takeaway("The model asked; your code ran the tool; the model answered. The model executed nothing.")

    banner("2. Every call pays for the whole conversation")
    per_call = loop_input_tokens(10, 2000, 500)
    table(["call", "input tokens", "running total"], [(i + 1, t, sum(per_call[: i + 1])) for i, t in enumerate(per_call)])
    say("10 calls bill 42,500 input tokens for a conversation that ends at 7,000 tokens long.")


if __name__ == "__main__":
    demo()
