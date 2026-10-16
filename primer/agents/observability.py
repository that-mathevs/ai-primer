r"""
# Observability: recording every agent run so you can replay why it did what it did

Run: `python -m primer.agents.observability`

This lesson builds on the agent loop from `primer.agents.agent_loop` and on
the golden set from `primer.agents.evals`.

## Level 1: The practitioner's guide

**In one sentence.** Observability for an agent means recording every run
as a tree of timed steps (the task, then each model call, tool call and
retrieval, with what went in and what came out), so that "why did it do
that?" is a lookup and not a guess.

**When you need it.** From the first day real people use the system. The
tell: a user reports a wrong answer from yesterday and the only evidence is
the answer itself. This lesson's worked example is exactly that case. A user
asks "Where is order A-100?" (with a hyphen) and gets "I couldn't find order
A-100." From the answer alone the order might not exist; the trace shows a
tool error on one line, `unknown order id A-100`, and tells you the fix
belongs in the tool that doesn't normalise hyphens, not in the prompt. The
second tell is a bill you can't explain: in the lesson's six-order run,
input tokens climb with every model call because each call resends the
growing history, and only a per-call record makes that visible before the
invoice does. You don't need a tracing platform for a script you run by
hand, but even there, one structured record per model call (model, tokens,
latency, cost) pays for itself the first time something is slow.

**Your options.** Six ways to see inside a run, from the cheapest to the
most complete:

| Option | What it does | What it guarantees | What it costs | Where it lives |
|---|---|---|---|---|
| Print statements and plain logs | Lines of text as the agent runs | You can read them, once; nothing ties one run's steps together | Nothing up front; hours later, when you need to find one run among thousands | Your code |
| Structured per-call records | One record per model call with model, tokens, latency, cost, prompt version | Dashboards for cost and volume; the bill becomes explainable | A logging library and a few fields per call | Your code |
| Traces with spans | Each step is a timed span with attributes and a parent; a run is a tree under one trace id | Replay any run; find the first error; see where the time goes | Instrumenting each call (a `with` block), and somewhere to store the trees | Your agent loop |
| A vendor's own SDK | The tracing platform's client records the spans for you | Fastest start | Lock-in: changing vendors is a code change | Your code, tied to one service |
| OpenTelemetry with the GenAI conventions | The same spans in standard names (`gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.tool.name`) and the standard OTLP format | Flows into the monitoring the organisation already runs and into any LLM tool; changing vendors is a collector setting | Learning the conventions; running a collector | Exporter and collector |
| An LLM tracing platform on top | Traces, user feedback, prompt versions, datasets and evals in one place | The loop from a thumbs-down to a golden test is a few clicks | Hosting it yourself or sending traces (with user data in them) to a service | A service |

**How to choose.** Decide by what you'll need to answer, then by who else
needs to read it.

- You need to explain a bill: structured per-call records with tokens,
  model and cost; a sum over them is the invoice.
- You need to explain a decision: traces. The first error span, or the
  first surprising output, shows where the chain broke. Nothing less lets
  you replay a run.
- The organisation already runs Datadog, Grafana or Honeycomb: emit
  OpenTelemetry spans with the GenAI attribute names, and let the collector
  fan them out. Name your own attributes with a prefix (`app.prompt.version`)
  so they never collide with the standard's.
- You want traces linked to feedback and evals: put an LLM tracing platform
  behind the collector; most of them accept OpenTelemetry, so the agent code
  doesn't change.
- Whatever you pick, redact personal data before export, restrict who can
  read traces, set a retention limit, and keep tenants' traces apart. A
  trace holds the user's words.

**What it costs.** Instrumentation is cheap in code: a span is a `with`
block around a call, and an exception inside it marks the span as an error
by itself. It is not free in storage: a trace keeps every prompt and tool
result, so this lesson's three-step run (two model calls, one tool call,
900 ms on its toy clock, 195 input and 38 output tokens, about a tenth of
a cent) is a few kilobytes, and a million runs a day is gigabytes.
Distributed tracing at scale has always answered that by sampling: Google's
Dapper, the design OpenTelemetry descends from, kept overhead low by
tracing a sample of requests through shared libraries rather than every
request. Keep every trace while volume is small; sample when it isn't, but
always keep the ones a user flagged or a monitor caught. Latency cost is
negligible when export is asynchronous. The cost that matters is what you
learn from the trace itself: in the lesson's six-order run, model calls
take far more of the time than tool calls, so the latency wins are fewer
model calls and smaller contexts, not faster tools.

**What breaks.**

- **A log with no thread through it.** Thousands of lines from thousands of
  runs, and no way to gather one run back together. A trace id on every
  span is the fix.
- **Fixing the prompt when the tool was wrong.** Without the trace, the
  hyphen bug above looks like a model mistake. Read the first error span
  before editing anything.
- **Personal data in the trace store.** Prompts and outputs hold names,
  emails and card numbers. Redact before export, and limit retention.
- **Vendor-shaped traces.** Attributes named after one product's SDK don't
  flow anywhere else. Use the standard names and a collector.
- **Gaps between spans.** Time outside any span (a queue, a retry wait,
  untraced code) is invisible in the tree. Instrument the retries and the
  guardrail checks too.
- **Traces nobody reads.** Recording is half the job. Store user feedback
  against the trace id and route flagged traces to a person, so each one can
  become a golden case.

**In the wild.** OpenTelemetry is the open standard for traces, metrics and
logs; its concepts page defines a trace as the path of a request through an
application and a span as a unit of work with a name, a parent, start and
end times, attributes, events and a status, which is exactly what this
lesson's `Span` holds. Its GenAI semantic conventions, maintained in their
own repository, standardise the span, metric and event names for model
calls, tool calls, agents and MCP. Langfuse is an open-source, self-hostable
platform that captures model and non-model calls, versions prompts, runs
evaluations (LLM judge, code, human) and tracks cost per user, built on
OpenTelemetry. Arize Phoenix is open source too, with tracing built on
OpenTelemetry and OpenInference, evaluators and datasets for comparing
versions. LangSmith and Braintrust play the same role as services. All of
them descend from Dapper (Sigelman et al., 2010), which introduced the
trace-and-span tree with propagated ids.

**Go deeper.** Level 2 builds the tracer in plain Python: a span, a `with`
block that opens and closes it, a fake clock so timings are exact, the
agent loop instrumented step by step, the export to OpenTelemetry-shaped
records, the walk that finds the first error, and the path from a
thumbs-down to a golden case. If you only needed to decide what to record
and where to send it, you are done.

## Level 2: How it works, from scratch

An aeroplane's flight recorder writes down every control input, every
instrument reading and every radio call, each stamped with the time. When
something goes wrong, investigators replay the flight instead of guessing.
A **trace** does the same for one run of an agent. The agent traced here
is a small support agent built on the tool-calling loop (see
`primer.agents.agent_loop`): it answers questions about orders with one
tool, `lookup_order`. Its model is scripted and its clock is fake, with a
model call taking 400 ms and a tool call 100 ms, so every trace below is
the same on every run.

## Spans: the steps of a run, as a tree

Each step of a run is a **span**: a named, timed unit of work with
key-value **attributes** (model name, tokens, tool name and so on), a status
(ok or error) and a parent. The top span is the whole task, and everything
the agent does sits under it. All the spans of one run share a **trace
id**, so they can be gathered back together.

The question "Where is order A100?" produces this trace:

```text
invoke_agent support-agent  900 ms  [ok]
├─ chat scripted-large  400 ms  [ok]  in=68 out=24 tokens
├─ execute_tool lookup_order  100 ms  [ok]
└─ chat scripted-large  400 ms  [ok]  in=127 out=14 tokens
```

The model asked for a tool (the first `chat`), the tool ran, and the model
answered (the second `chat`). The durations add up: 400 + 100 + 400 = 900
ms. The second model call reads more input tokens than the first because
the whole conversation, now including the tool's result, is sent again at
every step.

```mermaid
flowchart TD
  R["invoke_agent support-agent<br/>trace id tr-0001, 900 ms"] --> C1["chat: model call 1<br/>400 ms, 68 in / 24 out"]
  R --> T1["execute_tool lookup_order<br/>100 ms, args: A100"]
  R --> C2["chat: model call 2<br/>400 ms, 127 in / 14 out"]
```

**Reading it:** the root box is the whole task, and its three children are
the steps, left to right in the order they ran. Each box keeps its own
timing and attributes, and in a tracing tool each one opens to show what
went in and what came out. In a bigger agent, retrieval, sub-agents and
guardrail checks become spans as well.

**In code:** `Span` holds one step, `walk` visits a tree parents first in
time order, and `render` prints the text tree above.

## Recording spans as the agent runs

A **tracer** builds the tree while the agent runs. Before each model or
tool call the loop opens a span; after it, the loop closes the span and
records what came back, such as the token counts, the model's name and why
it stopped. That is all **instrumentation** means. In Python it is a
`with tracer.span(...):` block, and an exception raised inside the block
marks the span as an error by itself.

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

**Reading it:** time runs down the page. Every call to the model or the
tool sits between an "open span" and a "close span" sent to the tracer, and
the `invoke_agent` span opens first and closes last, so it encloses the
other three. No arrow comes back from the tracer: it only records, and
never changes what the agent does.

**In code:** `Tracer.span` opens a child of whichever span is open and
closes and times it when the block ends, and `Span.fail` marks an error
that didn't raise. `FakeClock` makes the timings repeat exactly, and
`run_traced_agent` is the loop in the diagram.

## Speaking a common language: OpenTelemetry

The step names in these trees, `chat`, `execute_tool` and `invoke_agent`,
come from **OpenTelemetry** (OTel), the open standard for traces, metrics
and logs. Its **GenAI
semantic conventions** fix the names of the attributes on model and agent
spans: `gen_ai.request.model`, `gen_ai.usage.input_tokens`,
`gen_ai.tool.name` and the rest. With these names, the same traces can go
into whatever monitoring the organisation already runs (Datadog, Grafana,
Honeycomb) or into tools built for language model traces (Langfuse,
LangSmith, Arize Phoenix, Braintrust), without a converter for each.
Attributes the standard doesn't cover, such as the prompt version and the
cost, get a prefix of your own; here it is `app.`, as in
`app.prompt.version` and `app.cost_usd`.

| Attribute | Meaning | Example |
|---|---|---|
| `gen_ai.operation.name` | what kind of step | `chat`, `execute_tool`, `invoke_agent` |
| `gen_ai.request.model` | model asked for | `claude-opus-5` |
| `gen_ai.usage.input_tokens` / `output_tokens` | tokens billed | 68 / 24 |
| `gen_ai.response.finish_reasons` | why the model stopped | `["tool_use"]` |
| `gen_ai.tool.name`, `gen_ai.tool.call.id` | which tool, which call | `lookup_order`, `toolu_1` |
| `app.prompt.version` | which prompt produced this call | `support-v3` |

The spans reach those tools in two hops. The agent hands each finished span
to an **exporter**, which sends it in the standard's own format, **OTLP**
(the OpenTelemetry Protocol). A **collector**
receives the spans and forwards them to every destination it is configured
for. Changing vendors then means changing the collector's configuration,
and the agent's code stays as it is.

```mermaid
flowchart LR
  A[Agent code<br/>+ tracer] -->|spans| X[Exporter]
  X -->|OTLP| C[Collector]
  C --> D[Existing monitoring<br/>Datadog, Grafana]
  C --> L[LLM tracing tools<br/>Langfuse, Phoenix]
  C --> W[Warehouse<br/>for evals and audits]
```

**Reading it:** read left to right. The agent's side ends at the exporter.
Everything from the collector on is configuration, and each of the three
destinations receives the same spans.

**In code:** `to_otel` turns a trace into OTLP-shaped span records.

## Debugging a failure from its trace

Like a flight recorder, a trace is read after something goes wrong. A user asks "Where is order A-100?", with a hyphen, and the agent
answers "I couldn't find order A-100." From the answer alone, the order
might not exist. The trace shows what happened in one line:

```text
invoke_agent support-agent  900 ms  [ok]
├─ chat scripted-large  400 ms  [ok]  in=68 out=24 tokens
├─ execute_tool lookup_order  100 ms  [error]  unknown order id A-100
└─ chat scripted-large  400 ms  [ok]  in=135 out=15 tokens
```

The model passed the id on as the user typed it, and the tool doesn't strip
hyphens. To find this line in a bigger tree, walk the spans in time order
and stop at the first one marked as an error. The fix belongs in the tool:
normalise the id, or return an error the model can act on, such as "ids
look like A100". A rewritten prompt would leave the bug where it is.

**In code:** `first_error` walks the tree and returns the first span marked
as an error.

### Where the time and the tokens go

A trace also answers questions that arrive without a complaint: where a run
spends its time, and what it is billed for. The next three charts come from
the same agent asked about several orders at once.

![In a three-order run, four 400 ms model calls alternate back to back with three 100 ms tool calls, so model calls fill most of the 1,900 ms](figures/primer.agents.observability.waterfall.svg)

**Reading it:** this is the waterfall view that tracing tools draw, for a
run that looks up three orders. Each row is a span with its bar on a shared
time axis, and the top row is the whole task. Model calls (blue) and tool
calls (orange) take turns, each starting as the one before it ends, and
every blue bar is four times as long as an orange one. In a real trace, a
gap between two bars is time spent outside any span (a queue, a wait
before a retry, code with no span around it), and worth a look.

![Input tokens climb with every model call, because each call resends the growing history](figures/primer.agents.observability.tokens.svg)

**Reading it:** each bar is one model call, in order, from a run that looks
up six orders one at a time. The blue part is input and the orange part is
output. The input grows with every call, since the whole conversation so
far is sent again each time. A per-call record shows this
climb before the bill does, which is how a tool that returns too much text,
or a loop that keeps going, gets caught.

![In the six-order run, model calls take far more of the time than tool calls do](figures/primer.agents.observability.latency.svg)

**Reading it:** the total milliseconds the same six-order run spent in
model calls and in tool calls. The model calls take most of it, so the
biggest gains in speed come from making fewer model calls (asking for
several tools in one turn, running them in parallel) and from sending
shorter contexts, not from faster tools.

**In code:** `trace_totals` adds up a trace's duration, model and tool
calls, tokens, cost and errors, the numbers behind these charts.

## Closing the loop

Reading one trace fixes one failure. Keeping it fixed takes a link from the
trace to the evaluation suite. When a user clicks thumbs-down, the feedback
is stored against the trace id. An engineer opens that trace, finds what
went wrong, and adds the run to the **golden set**, the fixed list of tasks
with known good answers that every change is graded on (see
`primer.agents.evals`). Then the fix is made, the evaluation suite passes
in CI (the checks that run automatically on every change), and the fix
ships.

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

**Reading it:** start at production traffic and follow the arrows back to
it. Each lap adds a real failure to the golden set as a permanent test, so
the same bug cannot come back without an eval failing. Without traces the
loop stops at the second box, with a failure in hand and no cause.

**In code:** `golden_case_from_trace` turns a flagged trace into a new case
for the golden set.

## In 20 seconds
- A trace records one agent run as a tree of spans: the task at the root,
  then each model call, tool call and retrieval, with inputs, outputs,
  tokens, latency, cost, and model and prompt versions.
- Name the attributes with the OpenTelemetry GenAI conventions, and the
  traces flow into existing monitoring or LLM tracing tools without a
  converter for each.
- To debug a run, find its first error span. The fix may belong in a tool
  or in retrieval rather than the prompt.
- Store user feedback against the trace id, so a failure in production
  becomes a test: failure → trace → golden case → fix → CI → deploy.

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
