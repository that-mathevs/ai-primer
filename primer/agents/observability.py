r"""
# Observability: recording every agent run so you can replay why it did what it did

Run: `python -m primer.agents.observability`

## The everyday picture

An aeroplane's flight recorder writes down every control input, every
instrument reading and every radio call, each stamped with the time. When
something goes wrong, investigators don't guess; they replay the flight.

A **trace** is a flight recorder for one agent run. It records the whole
task and every step inside it (each model call, each tool call, each
search) with its inputs, outputs, timing, token counts, cost, and which
model and prompt version produced it. Without traces, "why did the agent
refund that order?" is guesswork. With them, it's a lookup.

## Spans: the steps of a run, as a tree

**Everyday picture.** A recipe card with sub-recipes: "make the lasagne"
contains "make the sauce" and "make the béchamel", each with its own start
and end time.

Each step is a **span**: a named, timed unit of work with key-value
**attributes** (model name, tokens, tool name…), a status (ok or error), and
a parent. The top span is the whole task; everything the agent does is a
child of it. All spans of one run share a **trace id**, so they can be
gathered back together.

**Worked example.** "Where is order A100?" produces this trace, on a clock
where a model call takes 400 ms and a tool call 100 ms:

```text
invoke_agent support-agent  900 ms  [ok]
├─ chat scripted-large  400 ms  [ok]  in=68 out=24 tokens
├─ execute_tool lookup_order  100 ms  [ok]
└─ chat scripted-large  400 ms  [ok]  in=127 out=14 tokens
```

The model asked for a tool (first `chat`), the tool ran, and the model
answered (second `chat`). Durations add up: 400 + 100 + 400 = 900 ms. The
second model call has more input tokens because the conversation, including
the tool result, is resent every step.

```mermaid
flowchart TD
  R["invoke_agent support-agent<br/>trace id tr-0001, 900 ms"] --> C1["chat: model call 1<br/>400 ms, 68 in / 24 out"]
  R --> T1["execute_tool lookup_order<br/>100 ms, args: A100"]
  R --> C2["chat: model call 2<br/>400 ms, 127 in / 14 out"]
```

**Reading it:** the root box is the whole task; the three children are the
steps, left to right in time. Every box carries its own timing and
attributes, and every box can be opened to see exactly what went in and
came out. Retrieval, sub-agents and guardrail checks become spans too, so
the tree shows the whole path.

**In code:** `Span` holds one step: its name, kind, start and end times,
attributes, status and children. `walk` visits a tree parents first, in time
order, and `render` draws it as the text tree above.

## Recording spans as the agent runs

```mermaid
sequenceDiagram
  participant U as User
  participant A as Agent loop
  participant T as Tracer
  participant M as Model
  participant K as Tool
  U->>A: Where is order A100?
  A->>T: open span invoke_agent
  A->>T: open span chat
  A->>M: messages + tools
  M-->>A: tool_use lookup_order(A100)
  A->>T: close span chat (tokens, model, finish reason)
  A->>T: open span execute_tool
  A->>K: lookup_order(A100)
  K-->>A: shipped
  A->>T: close span execute_tool
  A->>T: open span chat
  A->>M: messages + tool result
  M-->>A: "Order A100 has shipped."
  A->>T: close span chat
  A->>T: close span invoke_agent
  A-->>U: Order A100 has shipped.
```

**Reading it:** the tracer sits beside the agent loop and is told when each
step starts and ends; it never changes what the agent does. Opening a span
before a call and closing it after is all **instrumentation** means. In
Python it's a `with tracer.span(...):` block, and an exception inside the
block marks the span as an error automatically.

**In code:** `Tracer.span` opens a child of whatever span is open, and
closes and times it when the block ends; `Span.fail` marks an error that
didn't raise. `FakeClock` makes the timings deterministic, and
`run_traced_agent` is the agent loop in the diagram, recording every step.

## Speaking a common language: OpenTelemetry

**Everyday picture.** Shipping containers: because every container has the
same shape, any ship, crane or truck can carry any of them.

**OpenTelemetry** (OTel) is the open standard for traces, metrics and logs,
and its **GenAI semantic conventions** standardize the attribute names for
model and agent spans: `gen_ai.request.model`, `gen_ai.usage.input_tokens`,
`gen_ai.tool.name` and so on. Use them, and your traces flow into whatever
the organisation already runs (Datadog, Grafana, Honeycomb) or into
LLM-specific tools (Langfuse, LangSmith, Arize Phoenix, Braintrust) with no
custom glue. Attributes this module invents itself use an `app.` prefix,
such as `app.prompt.version` and `app.cost_usd`.

| Attribute | Meaning | Example |
|---|---|---|
| `gen_ai.operation.name` | what kind of step | `chat`, `execute_tool`, `invoke_agent` |
| `gen_ai.request.model` | model asked for | `claude-opus-5` |
| `gen_ai.usage.input_tokens` / `output_tokens` | tokens billed | 68 / 24 |
| `gen_ai.response.finish_reasons` | why the model stopped | `["tool_use"]` |
| `gen_ai.tool.name`, `gen_ai.tool.call.id` | which tool, which call | `lookup_order`, `toolu_1` |
| `app.prompt.version` | which prompt produced this call | `support-v3` |

```mermaid
flowchart LR
  A[Agent code<br/>+ tracer] -->|spans| X[Exporter]
  X -->|OTLP| C[Collector]
  C --> D[Existing monitoring<br/>Datadog, Grafana]
  C --> L[LLM tracing tools<br/>Langfuse, Phoenix]
  C --> W[Warehouse<br/>for evals and audits]
```

**Reading it:** the agent only knows how to hand spans to an **exporter**,
which ships them in the standard **OTLP** format (OpenTelemetry Protocol).
A **collector** fans them out to as many destinations as you like. Swapping
vendors means changing the collector's configuration, not the agent.
`to_otel` in this module produces OTLP-shaped span records.

## Debugging a failure from its trace

**Everyday picture.** Replaying the flight recorder to the moment the
warning light came on.

**Worked example.** A user asks "Where is order A-100?" (with a hyphen). The
agent's answer is an unhelpful "I couldn't find order A-100." From the
answer alone the order might simply not exist. The trace shows why in one
line:

```text
invoke_agent support-agent  900 ms  [ok]
├─ chat scripted-large  400 ms  [ok]  in=68 out=24 tokens
├─ execute_tool lookup_order  100 ms  [error]  unknown order id A-100
└─ chat scripted-large  400 ms  [ok]  in=135 out=15 tokens
```

The model extracted the id exactly as typed and the tool doesn't normalize
hyphens. `first_error` walks the tree and returns that span. The fix
belongs in the tool (normalize ids, or return an error the model can act
on, such as "ids look like A100"), not in the prompt, and it's the trace
that tells you so.

![In a three-order run, four 400 ms model calls alternate back to back with three 100 ms tool calls, so model calls fill most of the 1,900 ms](figures/primer.agents.observability.waterfall.svg)

**Reading it:** the classic trace view. Each row is a span, its bar placed
on a shared time axis; the top row is the whole task. Model calls (blue)
alternate with tool calls (orange), each one starting the moment the
previous one ends: that staircase is the loop's rhythm, think, act, think,
act. The blue bars are four times as long as the orange ones, so where the
time goes is visible at a glance. In a real trace, a gap between two bars
would be time spent outside any span (a queue, a retry wait, untraced
code), and worth a look.

![Input tokens climb with every model call, because each call resends the growing history](figures/primer.agents.observability.tokens.svg)

**Reading it:** each bar is one model call in a run that looks up six
orders one at a time. Input tokens climb with every step because each call
resends the growing history. Traces make this visible per call, which is
how you notice a verbose tool result or a runaway loop before the bill
does.

![In the six-order run, model calls take far more of the time than tool calls do](figures/primer.agents.observability.latency.svg)

**Reading it:** total milliseconds spent in model calls versus tool calls
for that six-order run. Model calls dominate, so the biggest latency wins
are fewer model calls (batch tool calls into one turn, run them in
parallel) and smaller contexts, not faster tools.

**In code:** `trace_totals` adds up a trace's duration, model and tool calls,
tokens, cost and errors, the numbers behind these figures.

## Closing the loop

Traces become most valuable when linked to feedback and evals. When a user
clicks thumbs-down, the feedback is stored against the trace id; an engineer
opens the exact trace, sees what went wrong, and turns it into a golden
test case (`golden_case_from_trace`, see `primer.agents.evals`). The fix is
made, the eval suite passes in CI (the automated checks run on every
change), and it ships.

```mermaid
flowchart LR
  P[Production traffic] --> T[Traces]
  T --> F[Failures flagged<br/>by users or monitors]
  F --> G[Add to golden set]
  G --> X[Fix prompt, tools<br/>or retrieval]
  X --> E[Evals pass in CI]
  E --> D[Deploy]
  D --> P
```

**Reading it:** follow the loop clockwise from production. Each lap turns a
real failure into a permanent test, so quality ratchets up and the same bug
can't come back unnoticed. Without traces the loop breaks at the second
box: you know something failed but not why.

## In 20 seconds
- A trace records one run as a tree of spans: the task, then each model
  call, tool call and retrieval, with inputs, outputs, tokens, latency,
  cost, and model and prompt versions.
- Use the OpenTelemetry GenAI attribute names so traces flow into existing
  monitoring or LLM tracing tools without custom glue.
- Debug from traces: find the first error span; the fix often belongs in a
  tool or in retrieval, not the prompt.
- Link traces to feedback and evals: production failure → trace → golden
  case → fix → CI → deploy.

## Self-test questions

**What would you record for every agent run?**
A tree of spans: the task at the root, and each model call, tool call and
retrieval as children, with inputs and outputs (redacted of personal data),
token counts, latency, cost, model id, prompt version, tool arguments and
results, errors, and the user and tenant it ran for. That's enough to
replay any decision and to aggregate cost and quality.

**A user says the agent gave a wrong answer yesterday. How do you find
out why?**
Look up the trace by conversation or user id and walk it: did retrieval
return the right document, did the model pick the right tool with the right
arguments, did a tool error, did a guardrail fire? The first error span or
the first surprising output shows where the chain broke, and the trace
becomes a new golden test case.

**Why use OpenTelemetry instead of a vendor's SDK directly?**
Portability and fit: the organisation's existing monitoring already speaks
OTel, the GenAI conventions give every tool the same attribute names, and
changing vendors becomes a collector configuration change rather than a
code change.

**What should you be careful about when logging prompts and outputs?**
They contain user data. Redact personal data before export, restrict who
can read traces, set retention limits, and keep tenants' traces separated.

## The papers behind this lesson

- Sigelman et al., *Dapper, a Large-Scale Distributed Systems Tracing
  Infrastructure* (Google technical report, 2010),
  https://research.google/pubs/dapper-a-large-scale-distributed-systems-tracing-infrastructure/.
  Introduced the trace-and-span tree model with propagated trace ids that
  OpenTelemetry, and therefore agent tracing, is built on.

## Further reading
- OpenTelemetry semantic conventions for generative AI: https://opentelemetry.io/docs/specs/semconv/gen-ai/
- OpenTelemetry concepts: traces and spans: https://opentelemetry.io/docs/concepts/signals/traces/
- Langfuse documentation: https://langfuse.com/docs
- Arize Phoenix documentation: https://arize.com/docs/phoenix
"""

from __future__ import annotations

import itertools
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from primer._show import banner, say, table, takeaway
from primer.agents.llm import ScriptedLLM, ToolCall, last_user_text, tool_result_block, tool_results

# ---------------------------------------------------------------------------
# 1. Clock, spans and the tracer
# ---------------------------------------------------------------------------


class FakeClock:
    """A clock you advance by hand, so traces are deterministic in tests.

    Real tracers read the system clock. Everything else is identical.
    """

    def __init__(self, start_ms: float = 0.0):
        self.now = start_ms

    def __call__(self) -> float:
        return self.now

    def advance(self, ms: float) -> None:
        self.now += ms


@dataclass
class Span:
    span_id: str
    name: str
    kind: str  # "agent" | "llm" | "tool" | "retrieval"
    parent_id: str | None
    start_ms: float
    end_ms: float | None = None
    status: str = "ok"
    error: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    children: list["Span"] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return (self.end_ms or self.start_ms) - self.start_ms

    def set(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def fail(self, message: str) -> None:
        """Mark this span as an error without raising (e.g. a tool returned an error)."""
        self.status, self.error = "error", message


_trace_ids = itertools.count(1)


class Tracer:
    """Records nested spans for one trace. Use `with tracer.span(...)`."""

    def __init__(self, clock: Any = None, trace_id: str | None = None):
        self.clock = clock or FakeClock()
        self.trace_id = trace_id or f"tr-{next(_trace_ids):04d}"
        self.root: Span | None = None
        self._stack: list[Span] = []
        self._span_ids = itertools.count(1)

    @contextmanager
    def span(self, name: str, kind: str, **attributes: Any) -> Iterator[Span]:
        parent = self._stack[-1] if self._stack else None
        sp = Span(f"{next(self._span_ids):016x}", name, kind, parent.span_id if parent else None, self.clock(), attributes=dict(attributes))
        if parent:
            parent.children.append(sp)
        else:
            self.root = sp
        self._stack.append(sp)
        try:
            yield sp
        except Exception as e:  # record, then let the caller see the failure
            sp.fail(str(e))
            raise
        finally:
            sp.end_ms = self.clock()
            self._stack.pop()


def walk(span: Span) -> Iterator[Span]:
    """Depth-first, parents before children, in time order."""
    yield span
    for c in span.children:
        yield from walk(c)


# ---------------------------------------------------------------------------
# 2. A traced agent run
# ---------------------------------------------------------------------------

ORDERS = {"A100": "shipped", "A200": "delivered", "A300": "processing", "A400": "shipped", "A500": "delivered", "A600": "processing"}
LLM_MS, TOOL_MS = 400, 100
# Illustrative prices (dollars per million tokens), not any vendor's list price.
PRICE_IN_PER_M, PRICE_OUT_PER_M = 3.0, 15.0

TOOLS = [{"name": "lookup_order", "description": "Get an order's status by id, e.g. A100.",
          "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}]


def _support_policy(system, messages, tools):  # noqa: ARG001
    """Looks up each order id in the question, one per turn, then answers."""
    ids = re.findall(r"\b[A-Z]-?\d{3}\b", last_user_text(messages))
    done = tool_results(messages)
    if len(done) < len(ids):
        return ToolCall("", "lookup_order", {"order_id": ids[len(done)]})
    parts = []
    for oid, r in zip(ids, done):
        parts.append(f"I couldn't find order {oid}." if r.get("is_error") else f"Order {oid} has {r['content']}.")
    return " ".join(parts)


def run_traced_agent(question: str, llm: Any = None, clock: FakeClock | None = None, prompt_version: str = "support-v3") -> Tracer:
    """Run a small support agent with every step recorded. Returns the tracer."""
    clock = clock or FakeClock()
    tracer = Tracer(clock)
    llm = llm or ScriptedLLM(_support_policy)
    messages: list[dict] = [{"role": "user", "content": question}]
    with tracer.span("invoke_agent support-agent", "agent", **{"gen_ai.operation.name": "invoke_agent",
                                                                "gen_ai.agent.name": "support-agent", "app.input": question}) as root:
        for _ in range(10):
            with tracer.span(f"chat {llm.model}", "llm", **{"gen_ai.operation.name": "chat", "gen_ai.request.model": llm.model,
                                                            "app.prompt.version": prompt_version}) as sp:
                reply = llm.complete(system="You are a support agent.", messages=messages, tools=TOOLS)
                clock.advance(LLM_MS)  # stands in for the model's response time
                sp.set("gen_ai.response.model", reply.model)
                sp.set("gen_ai.usage.input_tokens", reply.usage.input_tokens)
                sp.set("gen_ai.usage.output_tokens", reply.usage.output_tokens)
                sp.set("gen_ai.response.finish_reasons", [reply.stop_reason])
                sp.set("app.cost_usd", (reply.usage.input_tokens * PRICE_IN_PER_M + reply.usage.output_tokens * PRICE_OUT_PER_M) / 1e6)
            messages.append({"role": "assistant", "content": reply.assistant_content})
            if reply.stop_reason != "tool_use":
                root.set("app.output", reply.text)
                break
            results = []
            for call in reply.tool_calls:
                with tracer.span(f"execute_tool {call.name}", "tool", **{"gen_ai.operation.name": "execute_tool",
                                                                         "gen_ai.tool.name": call.name, "gen_ai.tool.call.id": call.id,
                                                                         "app.tool.arguments": call.input}) as sp:
                    clock.advance(TOOL_MS)
                    oid = call.input["order_id"]
                    if oid in ORDERS:
                        results.append(tool_result_block(call.id, ORDERS[oid]))
                    else:
                        sp.fail(f"unknown order id {oid}")
                        results.append(tool_result_block(call.id, f"unknown order id {oid}", is_error=True))
            messages.append({"role": "user", "content": results})
    return tracer


# ---------------------------------------------------------------------------
# 3. Reading traces
# ---------------------------------------------------------------------------


def trace_totals(root: Span) -> dict[str, Any]:
    spans = list(walk(root))
    llm = [s for s in spans if s.kind == "llm"]
    return {
        "duration_ms": root.duration_ms,
        "llm_calls": len(llm),
        "tool_calls": sum(s.kind == "tool" for s in spans),
        "input_tokens": sum(s.attributes.get("gen_ai.usage.input_tokens", 0) for s in llm),
        "output_tokens": sum(s.attributes.get("gen_ai.usage.output_tokens", 0) for s in llm),
        "cost_usd": sum(s.attributes.get("app.cost_usd", 0.0) for s in llm),
        "errors": sum(s.status == "error" for s in spans),
    }


def first_error(root: Span) -> Span | None:
    """The earliest span (in time order) that failed, or None."""
    return next((s for s in walk(root) if s.status == "error"), None)


def _line(s: Span) -> str:
    extra = ""
    if s.kind == "llm":
        extra = f"  in={s.attributes.get('gen_ai.usage.input_tokens')} out={s.attributes.get('gen_ai.usage.output_tokens')} tokens"
    if s.error:
        extra += f"  {s.error}"
    return f"{s.name}  {s.duration_ms:.0f} ms  [{s.status}]{extra}"


def render(root: Span, prefix: str = "") -> str:
    """Draw the span tree with box-drawing branches."""
    lines = [_line(root)] if not prefix else []

    def rec(span: Span, indent: str) -> None:
        for i, c in enumerate(span.children):
            last = i == len(span.children) - 1
            lines.append(f"{indent}{'└─' if last else '├─'} {_line(c)}")
            rec(c, indent + ("   " if last else "│  "))

    rec(root, prefix)
    return "\n".join(lines)


def to_otel(tracer: Tracer) -> list[dict[str, Any]]:
    """OTLP-shaped span records (times in Unix nanoseconds)."""
    out = []
    for s in walk(tracer.root):
        out.append({
            "traceId": tracer.trace_id,
            "spanId": s.span_id,
            "parentSpanId": s.parent_id or "",
            "name": s.name,
            "startTimeUnixNano": int(s.start_ms * 1_000_000),
            "endTimeUnixNano": int((s.end_ms or s.start_ms) * 1_000_000),
            "attributes": dict(s.attributes),
            "status": {"code": "ERROR" if s.status == "error" else "OK", "message": s.error or ""},
        })
    return out


def golden_case_from_trace(tracer: Tracer, feedback: str) -> dict[str, Any]:
    """Turn a flagged trace into a test case for the golden set."""
    err = first_error(tracer.root)
    return {
        "id": f"prod-{tracer.trace_id}",
        "input": tracer.root.attributes.get("app.input"),
        "observed_output": tracer.root.attributes.get("app.output"),
        "observed_error": err.error if err else None,
        "feedback": feedback,
    }


# ---------------------------------------------------------------------------
# Figures and demo
# ---------------------------------------------------------------------------


def figures() -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs: dict[str, Any] = {}
    colors = {"agent": "#8c8c8c", "llm": "#4c72b0", "tool": "#dd8452"}

    t = run_traced_agent("Where are orders A100, A200 and A300?")
    spans = list(walk(t.root))
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for row, s in enumerate(spans):
        ax.barh(row, s.duration_ms, left=s.start_ms, color=colors[s.kind])
        ax.text(s.start_ms + 5, row, s.name.replace("scripted-large", "model"), va="center", fontsize=7, color="white" if s.kind != "agent" else "black")
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xlabel("milliseconds since the task started")
    ax.set_title("Trace waterfall: model calls (blue) and tool calls (orange)")
    fig.tight_layout()
    figs["waterfall"] = fig

    t6 = run_traced_agent("Where are orders A100, A200, A300, A400, A500 and A600?")
    llm = [s for s in walk(t6.root) if s.kind == "llm"]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(range(1, len(llm) + 1), [s.attributes["gen_ai.usage.input_tokens"] for s in llm], color="#4c72b0", label="input")
    ax.bar(range(1, len(llm) + 1), [s.attributes["gen_ai.usage.output_tokens"] for s in llm],
           bottom=[s.attributes["gen_ai.usage.input_tokens"] for s in llm], color="#dd8452", label="output")
    ax.set_xlabel("model call number within one run")
    ax.set_ylabel("tokens")
    ax.set_title("Each call resends the growing conversation")
    ax.legend()
    fig.tight_layout()
    figs["tokens"] = fig

    by_kind = {"model calls": sum(s.duration_ms for s in walk(t6.root) if s.kind == "llm"),
               "tool calls": sum(s.duration_ms for s in walk(t6.root) if s.kind == "tool")}
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax.barh(list(by_kind), list(by_kind.values()), color=["#4c72b0", "#dd8452"])
    for i, v in enumerate(by_kind.values()):
        ax.text(v, i, f" {v:.0f} ms", va="center")
    ax.set_xlabel("total milliseconds in the six-order run")
    ax.set_xlim(0, max(by_kind.values()) * 1.2)
    ax.set_title("Where the time goes")
    fig.tight_layout()
    figs["latency"] = fig
    return figs


def demo() -> None:
    banner("1. A traced run")
    ok = run_traced_agent("Where is order A100?")
    print(render(ok.root))
    print()
    table(["total", "value"], list(trace_totals(ok.root).items()), floatfmt=".6f")

    banner("2. Debugging a failure from its trace")
    bad = run_traced_agent("Where is order A-100?")
    print(render(bad.root))
    print()
    err = first_error(bad.root)
    print(f"first error: {err.name}: {err.error}  (arguments {err.attributes['app.tool.arguments']})")
    print(f"agent said: {bad.root.attributes['app.output']!r}")
    print()
    say("""The model passed the id exactly as typed and the tool doesn't normalize
        hyphens. The fix belongs in the tool, and the trace is what tells you so.""")

    banner("3. OpenTelemetry-shaped export (first two spans)")
    for s in to_otel(ok)[:2]:
        print({k: s[k] for k in ("traceId", "spanId", "parentSpanId", "name", "startTimeUnixNano", "endTimeUnixNano")})
    print()

    banner("4. Feedback closes the loop")
    print(golden_case_from_trace(bad, feedback="thumbs_down"))
    print()
    takeaway("Record every step with standard attribute names; debug from the trace; turn flagged traces into golden tests.")


if __name__ == "__main__":
    demo()
