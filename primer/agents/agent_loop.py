r"""
# The agent loop, production-grade

Run: `python -m primer.agents.agent_loop`

## The idea

**Everyday picture.** A new assistant runs errands for you. They can't do
anything themselves. They come back after *every* errand and say: "here's
what I found; next I'd like to do X." You carry out X and hand them the
result. They decide the next step, and so on, until they say "done" (or you
say "that's enough, you've spent the budget"). The assistant is the model,
the errands are tool calls, and *you* are the loop in this file.

**Tiny worked example.** "How many PTO days does alice have left, and what
rolls over?" Here's the real transcript the happy-path demo produces:

```text
[user]       How many PTO days does alice have left, and what rolls over?
[assistant]  I'll check the PTO policy and alice's balance at the same time.
             tool_use toolu_1 search_kb       {"query": "PTO rollover"}
             tool_use toolu_2 get_pto_balance {"employee": "alice", "as_of": "2026-09-25"}
[user]       tool_result toolu_1  "[hr-001] Paid time off: ... Unused PTO up to 5 days rolls over. ..."
             tool_result toolu_2  {"employee": "alice", "as_of": "2026-09-25", "days_left": 12}
[assistant]  Alice has 12 PTO days left. Up to 5 unused days roll over to next year [hr-001].
             (stop_reason: end_turn)
```

Two model calls, two tools run in parallel, one answer. **ReAct** ("reason +
act") is the research name for this pattern of alternating a bit of
reasoning ("I'll check both at once") with actions (tool calls). The loop
itself is ten lines of code. Everything else in this file exists because
the ten-line version fails in production.

```mermaid
flowchart LR
  G[Goal] --> T[Think<br/>choose next step]
  T --> C[Call a tool]
  C --> O[Observe result]
  O --> D{Done, or budget<br/>or step limit hit?}
  D -->|No| T
  D -->|Yes| F[Final answer<br/>or hand to human]
```

**Reading it:** start at *Goal* on the left and follow the arrows around the
cycle. The model only ever does the *Think* box: it looks at everything so
far and picks the next action. *Call a tool* and *Observe result* are your
code. The diamond is the only exit, and it has three ways out: the model
says it's done, a limit trips, or the agent hands the task to a person. An
agent without that diamond is a loop you can't stop.

**In code:** `run_agent` is this loop, and it returns an `AgentResult`
holding the final text, the stop cause, every step, the usage and the cost.
`search_kb` and `get_pto_balance` are the two tools in the example.

## One round trip, message by message

```mermaid
sequenceDiagram
  participant U as User
  participant L as Your loop
  participant M as Model
  participant T as Tools
  U->>L: task
  L->>M: system + messages + tool definitions
  M-->>L: tool_use blocks (stop_reason = tool_use)
  L->>L: validate, check budget, check for loops
  L->>T: run the calls (concurrently)
  T-->>L: results or errors
  L->>M: ONE user message with every tool_result
  M-->>L: final text (stop_reason = end_turn)
  L-->>U: answer + stop cause
```

**Reading it:** time runs top to bottom. The model's reply is a *request*
(`tool_use`), never an action. Everything between that request and the next
call to the model happens in your loop, which is where every control in
this file lives: validation, budgets, loop detection, concurrency. Notice
the model is called twice for one question. Each call re-sends the whole
history, and that's where the cost comes from.

**In code:** each pass through `run_agent` appends the model's full
assistant content, runs the requested tools, and appends every result in one
user message; a `Step` records that one model call and the `ToolExecution`s
it triggered.

## The controls, in the order the loop checks them

**Everyday picture.** Lending your car to a new driver: a full tank but no
more fuel money (budget), "come back by six" (step limit), "if you drive past
the same petrol station three times, you're lost, so come home" (loop
detection), and "call me if anything feels off" (hand-off to a human).

```mermaid
flowchart TD
  R[Model response] --> S1{stop_reason?}
  S1 -->|refusal| X1[stop: refusal]
  S1 -->|max_tokens| X2[stop: max_tokens<br/>don't run half-written calls]
  S1 -->|end_turn| X3[stop: completed]
  S1 -->|tool_use| B{Over token or<br/>dollar budget?}
  B -->|yes| X4[stop: budget_exceeded]
  B -->|no| H{Called handoff_to_human?}
  H -->|yes| X5[stop: handoff]
  H -->|no| L{Same tool + same args<br/>seen N times?}
  L -->|yes| X6[stop: loop_detected]
  L -->|no| E[Run tools concurrently]
  E --> A[Append all results<br/>in one user message]
  A --> N{Step limit hit?}
  N -->|yes| X7[stop: max_steps]
  N -->|no| M[Call the model again]
```

**Reading it:** every red exit is a named `stop_cause` in `AgentResult`.
Read top to bottom: first trust the model's own stop reason, then protect
your wallet (budget), then honor an explicit escalation, then protect
against repetition, and only then spend time running tools. The order
matters. A truncated response (`max_tokens`) is checked before tools run,
because a call cut off mid-argument must never execute.

| Failure | What happens | Control in `run_agent` |
|---|---|---|
| Model never says "done" | Runs forever | `max_steps` |
| Each turn re-sends the whole conversation | Cost grows quadratically with steps | `max_total_tokens`, `max_cost_usd` |
| Model repeats the same call | Burns money, no progress | loop detection on (tool, canonical args) |
| Tool raises | Loop crashes, or the model never learns why | errors become `is_error` tool results with actionable text |
| Unknown tool / hallucinated name | `KeyError` | `is_error` result listing the real tools |
| Several independent calls in one turn | Slow if run one by one | run concurrently, return **all** results in **one** user message |
| Output cut off (`max_tokens`) | Half-written tool call | stop with cause `max_tokens` |
| Safety refusal (`refusal`) | Content is not an answer | stop with cause `refusal` |
| Task is outside the agent's authority | Agent guesses | a `handoff_to_human` tool that ends the run cleanly |

"It stopped" is not an outcome you can monitor; "it stopped because of
loop_detected at step 3" is.

**In code:** `AgentConfig` holds every limit (steps, tokens, dollars, the
loop threshold) and the prices `AgentConfig.cost` uses to turn `primer.agents.llm.Usage` into
dollars. A tool raises `ToolError` to send the model an actionable error
result, and `HANDOFF_TOOL_DEF` is the hand-off tool's definition.

## Parallel tool calls: fan out, fan in

**Everyday picture.** Three errands in three different shops. You can do them
one after another, or send three friends at once and be done when the
slowest one gets back.

```mermaid
flowchart LR
  M[Assistant turn<br/>tool_use A, tool_use B, tool_use C] --> A[run A]
  M --> B[run B]
  M --> C[run C]
  A --> J[One user message:<br/>tool_result A, B, C<br/>matched by tool_use_id]
  B --> J
  C --> J
```

**Reading it:** the model asked for A, B and C in the same turn, before
seeing any result, so they can't depend on each other and it's safe to run
them at once. Wall-clock time becomes the *slowest* call instead of the
*sum*. On the way back, all three results go in a single user message. The
API pairs each result with its request by id, and splitting them teaches the
model to stop asking for parallel calls.

**In code:** `execute_tools` runs one turn's calls on a thread pool and
returns their results in the order they were asked for, turning an unknown
tool name or a raised exception into an error result instead of a crash.

## Why cost grows quadratically

Each call sends the *entire* conversation so far. If every step adds about
$t$ tokens, call $k$ sends roughly $k\,t$ input tokens, so a run of $n$ steps
sends

$$
\text{total input} \approx \sum_{k=1}^{n} k\,t = \frac{t\,n(n+1)}{2} \approx \frac{t\,n^2}{2}.
$$

**Symbols**

| Symbol | Meaning |
|---|---|
| $t$ | new tokens each step adds to the conversation (tool call + result) |
| $k$ | the step number, 1, 2, 3, ... |
| $n$ | total steps in the run |

**In words:** step $k$ re-sends everything from the $k$ steps before it, so
the total is $t$ times $1 + 2 + \dots + n$, which is about half of $n$ squared times $t$.

**On the example:** with $t = 500$ and $n = 10$, the total is
$500 \times 55 = 27{,}500$ input tokens, not the $5{,}000$ you'd guess. Double
to $n = 20$ and it's $500 \times 210 = 105{,}000$, almost 4x.

**In Python:**

```python
t = 500
def total_input(n):
    # Σ_k k·t: step k re-sends k steps' worth
    return sum(k * t for k in range(1, n + 1))
total_input(10)  # → 27500
total_input(20)  # → 105000
# almost 4x
round(total_input(20) / total_input(10), 1)  # → 3.8
```

![Input tokens sent at each step of a runaway agent](figures/primer.agents.agent_loop.tokens_per_step.svg)

**Reading it:** each bar is one call to the model in the budget-burner demo
(a model that keeps searching). Bars grow step by step because the history
grows. The line is the running total, which is what you pay for, and it
bends upward. The dashed horizontal line is the token budget. The run stops
at the first step where the total crosses it, instead of carrying on.

![Cumulative input tokens: linear intuition vs. actual quadratic growth](figures/primer.agents.agent_loop.quadratic_cost.svg)

**Reading it:** the x-axis is the number of steps in a run, and the y-axis is
the total input tokens sent. The straight line is what people intuitively expect
("each step costs the same"). The curve is what actually happens when every
call re-sends the history. At 10 steps the gap is about 5x, and it keeps
widening. That gap is why step limits, trimming tool output, and prompt
caching (`primer.agents.cost`) matter.

![Wall-clock time for k tool calls, sequential vs. concurrent](figures/primer.agents.agent_loop.parallel_latency.svg)

**Reading it:** the x-axis is how many tools the model requested in one turn,
each with a realistic latency between 0.2 s and 1.2 s. Run one after another,
the time is the *sum* and climbs with every tool. Run concurrently, it's
the *max*, which flattens out near the slowest single call. Parallel tool
calls are among the cheapest latency wins in agent systems.

**In code:** `cumulative_input_tokens` evaluates the formula above, $t$
times $n(n+1)/2$, for any run length.

## Running it against a real model

The loop only depends on the `LLM` protocol, so the same code runs against
Claude:

```python
from primer.agents.llm import ClaudeLLM
from primer.agents.agent_loop import run_agent, AgentConfig

result = run_agent(
    ClaudeLLM(),                      # needs `pip install anthropic` + credentials
    task="How many PTO days does alice have left, and what's the rollover rule?",
    tools=HR_TOOLS,                   # name -> python function
    tool_defs=HR_TOOL_DEFS,           # Anthropic tool definitions (JSON Schema)
    system="You are an HR assistant. Use tools; cite doc ids.",
    config=AgentConfig(max_steps=8, max_cost_usd=0.50),
)
print(result.stop_cause, result.final_text)
```

The SDK also ships a *tool runner* (`client.beta.messages.tool_runner`)
that drives this loop for you. Owning the loop, as here, is what you do
when you need custom budgets, loop detection, approval gates or tracing in
exactly your shape.

## In 20 seconds
- The agent loop is: call model, run the tools it asks for, append results, repeat until `end_turn`.
- The model never executes anything. Your code validates and runs every tool call.
- Production loops need step limits, token/cost budgets, loop detection, error-as-result handling, and a human hand-off.
- Parallel tool calls: run them concurrently, return all results in a single user message.
- Input cost grows roughly with the square of the number of steps, because every call re-sends the history.

## Self-test questions

**Q: An agent in production suddenly costs 5x more per task. What do you check first?**
A: Steps per task and tokens per step, from traces. A jump usually means a loop
(the same call repeated), a tool that started returning huge payloads, or a
prompt change that stopped the model from recognizing "done". Step budgets and
loop detection cap the damage; alerts on tokens-per-task catch it early.

**Q: A tool throws an exception mid-run. What should the loop do?**
A: Catch it and return a `tool_result` with `is_error: true` and a message the
model can act on ("date must be YYYY-MM-DD, got 'next friday'"). The model
usually fixes its next call. Crashing the loop throws away the work so far;
swallowing the error silently makes the model guess.

**Q: Why return all parallel tool results in one message?**
A: The API pairs each `tool_use` with its `tool_result` by id in the next user
turn. Splitting results across several messages breaks that pairing and
teaches the model to stop making parallel calls, which slows every run.

**Q: When should an agent hand off to a human?**
A: When it lacks authority (refunds over a limit), lacks information after
reasonable search, detects conflicting sources, or hits a budget. Make hand-off a
tool, so it is an explicit, logged outcome rather than a vague final answer.

## The papers behind this lesson

- **Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2022).**
  https://arxiv.org/abs/2210.03629. It showed that interleaving short reasoning
  traces with tool actions, then observing the results, beats reasoning alone or
  acting alone, and this loop is the shape of nearly every agent built since.
  [annotated companion](../../papers/react.html)

## Further reading
- Anthropic, *Building effective agents*: https://www.anthropic.com/engineering/building-effective-agents
- Claude tool use, implementing the loop: https://docs.claude.com/en/docs/agents-and-tools/tool-use/implement-tool-use
- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2022): https://arxiv.org/abs/2210.03629
- Lilian Weng, *LLM Powered Autonomous Agents*: https://lilianweng.github.io/posts/2023-06-23-agent/
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from primer._show import banner, say, table, takeaway
from primer.agents.llm import (
    LLM,
    LLMResponse,
    ScriptedLLM,
    ToolCall,
    Usage,
    last_user_text,
    tool_result_block,
    tool_results,
)

StopCause = Literal[
    "completed",  # model said end_turn with a final answer
    "max_steps",  # step limit hit
    "budget_exceeded",  # token or dollar budget hit
    "loop_detected",  # same tool + same args repeated too often
    "max_tokens",  # a response was truncated
    "refusal",  # the model declined
    "handoff",  # the model handed the task to a human
]

HANDOFF_TOOL = "handoff_to_human"

# The hand-off tool definition. Add it to `tool_defs` to give the agent an
# explicit, auditable way to say "this needs a person".
HANDOFF_TOOL_DEF: dict[str, Any] = {
    "name": HANDOFF_TOOL,
    "description": (
        "Escalate to a human when the request is outside your authority (for example refunds, "
        "policy exceptions, legal questions), when sources conflict, or when you cannot find the "
        "answer after searching. Do not use for questions you can answer with the other tools."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"reason": {"type": "string", "description": "Why a human is needed, one sentence."}},
        "required": ["reason"],
    },
}


class ToolError(Exception):
    """Raise from a tool to send an actionable `is_error` result back to the model.

    The message is shown to the model verbatim, so write it for the model:
    say what was wrong and what a valid call looks like.
    """


@dataclass
class AgentConfig:
    max_steps: int = 10
    max_total_tokens: int = 50_000
    max_cost_usd: float | None = None
    # Stop when the same (tool, args) pair has been requested this many times.
    loop_threshold: int = 3
    max_parallel_tools: int = 4
    max_tokens_per_call: int = 4096
    # Dollar prices per million tokens. Defaults are Claude Opus 5 list prices
    # at the time of writing; check the provider's pricing page.
    price_in_per_mtok: float = 5.0
    price_out_per_mtok: float = 25.0

    def cost(self, usage: Usage) -> float:
        return (usage.input_tokens * self.price_in_per_mtok + usage.output_tokens * self.price_out_per_mtok) / 1e6


@dataclass
class ToolExecution:
    name: str
    input: dict[str, Any]
    content: str
    is_error: bool


@dataclass
class Step:
    """One model call plus the tools it triggered."""

    index: int
    text: str
    tool_executions: list[ToolExecution]
    usage: Usage
    stop_reason: str


@dataclass
class AgentResult:
    final_text: str
    stop_cause: StopCause
    steps: list[Step]
    usage: Usage
    cost_usd: float
    transcript: list[dict[str, Any]] = field(repr=False)
    detail: str = ""  # human-readable explanation of the stop cause

    @property
    def ok(self) -> bool:
        return self.stop_cause == "completed"

    @property
    def tool_call_count(self) -> int:
        return sum(len(s.tool_executions) for s in self.steps)


def _signature(call: ToolCall) -> str:
    # Canonical JSON (sorted keys) so {"a":1,"b":2} and {"b":2,"a":1} count as the same call.
    return f"{call.name}:{json.dumps(call.input, sort_keys=True, default=str)}"


def _execute_one(call: ToolCall, tools: dict[str, Callable[..., Any]]) -> ToolExecution:
    """Run one tool call, turning every failure into an actionable error result."""
    fn = tools.get(call.name)
    if fn is None:
        # Hallucinated tool names happen. Tell the model what exists.
        return ToolExecution(call.name, call.input, f"Unknown tool '{call.name}'. Available tools: {sorted(tools)}.", True)
    try:
        out = fn(**call.input)
        return ToolExecution(call.name, call.input, out if isinstance(out, str) else json.dumps(out, default=str), False)
    except ToolError as e:
        return ToolExecution(call.name, call.input, str(e), True)
    except TypeError as e:
        # Usually wrong/missing argument names. The message names the bad argument.
        return ToolExecution(call.name, call.input, f"Bad arguments for {call.name}: {e}", True)
    except Exception as e:  # noqa: BLE001
        # Don't leak stack traces or internals into the context; give the class and message.
        return ToolExecution(call.name, call.input, f"{call.name} failed ({type(e).__name__}): {e}", True)


def execute_tools(calls: list[ToolCall], tools: dict[str, Callable[..., Any]], max_workers: int = 4) -> list[ToolExecution]:
    """Run tool calls concurrently. Results come back in the same order as `calls`.

    Tools in one assistant turn are independent by construction (the model
    asked for them together, before seeing any result), so running them in
    parallel is safe and cuts wall-clock time to the slowest call.
    """
    if len(calls) <= 1:
        return [_execute_one(c, tools) for c in calls]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda c: _execute_one(c, tools), calls))


def run_agent(
    llm: LLM,
    task: str,
    tools: dict[str, Callable[..., Any]],
    tool_defs: list[dict[str, Any]],
    system: str = "You are a helpful assistant. Use the tools when needed.",
    config: AgentConfig | None = None,
) -> AgentResult:
    """Run the think -> act -> observe loop until done or a control trips.

    Args:
        llm: anything implementing `LLM.complete` (ScriptedLLM or ClaudeLLM).
        task: the user's request.
        tools: tool name -> Python callable taking the tool input as kwargs.
        tool_defs: Anthropic-format tool definitions sent to the model.
        system: system prompt.
        config: limits and prices.

    Returns:
        AgentResult with the final text, stop cause, per-step record, usage,
        cost and the full transcript (for tracing and replay).
    """
    config = config or AgentConfig()
    messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
    steps: list[Step] = []
    usage = Usage()
    seen: Counter[str] = Counter()

    def finish(cause: StopCause, text: str = "", detail: str = "") -> AgentResult:
        return AgentResult(text, cause, steps, usage, config.cost(usage), messages, detail)

    for i in range(config.max_steps):
        resp: LLMResponse = llm.complete(
            system=system, messages=messages, tools=tool_defs, max_tokens=config.max_tokens_per_call
        )
        usage += resp.usage
        # Always append the full assistant content (it may carry thinking blocks
        # that must be sent back unchanged with the real API).
        messages.append({"role": "assistant", "content": resp.assistant_content})
        step = Step(i, resp.text, [], resp.usage, resp.stop_reason)
        steps.append(step)

        # --- Stop reasons that aren't "use a tool" -------------------------
        if resp.stop_reason == "refusal":
            return finish("refusal", resp.text, "model declined the request")
        if resp.stop_reason == "max_tokens":
            # A truncated response may contain a half-formed tool call. Don't run it.
            return finish("max_tokens", resp.text, "response truncated; raise max_tokens or ask for less output")
        if resp.stop_reason == "end_turn" or not resp.tool_calls:
            return finish("completed", resp.text)

        # --- Budget: checked after every call, before spending more ---------
        total = usage.input_tokens + usage.output_tokens
        if total > config.max_total_tokens:
            return finish("budget_exceeded", resp.text, f"{total:,} tokens > budget {config.max_total_tokens:,}")
        if config.max_cost_usd is not None and config.cost(usage) > config.max_cost_usd:
            return finish("budget_exceeded", resp.text, f"${config.cost(usage):.4f} > budget ${config.max_cost_usd:.4f}")

        # --- Hand-off: an explicit, logged outcome --------------------------
        for call in resp.tool_calls:
            if call.name == HANDOFF_TOOL:
                return finish("handoff", call.input.get("reason", ""), "escalated to a human")

        # --- Loop detection: same call requested too many times -------------
        for call in resp.tool_calls:
            seen[_signature(call)] += 1
            if seen[_signature(call)] >= config.loop_threshold:
                return finish(
                    "loop_detected", resp.text, f"{call.name}({json.dumps(call.input)}) requested {seen[_signature(call)]} times"
                )

        # --- Act: run all requested tools concurrently ----------------------
        executions = execute_tools(resp.tool_calls, tools, config.max_parallel_tools)
        step.tool_executions = executions
        # All results go back in ONE user message, matched by tool_use_id.
        messages.append(
            {
                "role": "user",
                "content": [
                    tool_result_block(call.id, ex.content, ex.is_error) for call, ex in zip(resp.tool_calls, executions)
                ],
            }
        )

    return finish("max_steps", steps[-1].text if steps else "", f"hit max_steps={config.max_steps}")


# ---------------------------------------------------------------------------
# Demo tools and scripted "models"
# ---------------------------------------------------------------------------

PTO_BALANCES = {"alice": 12, "bob": 3}

SEARCH_KB_DEF = {
    "name": "search_kb",
    "description": "Search the company knowledge base (IT, HR, finance policies). Returns doc ids and text.",
    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
}
PTO_BALANCE_DEF = {
    "name": "get_pto_balance",
    "description": "Get an employee's remaining PTO days as of a date. Use for 'how many days do I have left'.",
    "input_schema": {
        "type": "object",
        "properties": {
            "employee": {"type": "string", "description": "lowercase username, e.g. 'alice'"},
            "as_of": {"type": "string", "description": "date as YYYY-MM-DD"},
        },
        "required": ["employee", "as_of"],
    },
}


def search_kb(query: str) -> str:
    """Tiny keyword search over the shared corpus (see primer.agents.rag for the real thing)."""
    from primer.common.corpus import DOCS
    from primer.common.text import tokenize

    q = set(tokenize(query))
    scored = sorted(DOCS, key=lambda d: -len(q & set(tokenize(d.title + " " + d.text))))
    return "\n".join(f"[{d.id}] {d.title}: {d.text}" for d in scored[:2])


def get_pto_balance(employee: str, as_of: str) -> str:
    import re

    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", as_of):
        # Actionable: says what's wrong AND what a valid value looks like.
        raise ToolError(f"as_of must be a date in YYYY-MM-DD format (e.g. 2026-10-02), got {as_of!r}.")
    if employee not in PTO_BALANCES:
        raise ToolError(f"Unknown employee {employee!r}. Known usernames: {sorted(PTO_BALANCES)}.")
    return json.dumps({"employee": employee, "as_of": as_of, "days_left": PTO_BALANCES[employee]})


DEMO_TOOLS = {"search_kb": search_kb, "get_pto_balance": get_pto_balance}
DEMO_TOOL_DEFS = [SEARCH_KB_DEF, PTO_BALANCE_DEF, HANDOFF_TOOL_DEF]


def happy_policy(system, messages, tools):
    """Turn 1: two independent lookups in parallel. Turn 2: answer from the results."""
    results = tool_results(messages)
    if not results:
        return (
            "I'll check the PTO policy and alice's balance at the same time.",
            [ToolCall("", "search_kb", {"query": "PTO rollover"}), ToolCall("", "get_pto_balance", {"employee": "alice", "as_of": "2026-09-25"})],
        )
    balance = json.loads(results[-1]["content"])["days_left"]
    return f"Alice has {balance} PTO days left. Up to 5 unused days roll over to next year [hr-001]."


def looping_policy(system, messages, tools):
    """A confused model that keeps re-running the same search."""
    return ToolCall("", "search_kb", {"query": "PTO rollover"})


def budget_burner_policy(system, messages, tools):
    """A model that keeps searching with new queries: no loop, just ever-growing context."""
    n = len(tool_results(messages))
    return ToolCall("", "search_kb", {"query": f"PTO policy details page {n}"})


def recovering_policy(system, messages, tools):
    """First call has a bad date; the model reads the actionable error and fixes it."""
    results = tool_results(messages)
    if not results:
        return ToolCall("", "get_pto_balance", {"employee": "bob", "as_of": "next friday"})
    last = results[-1]
    if last.get("is_error"):
        return ToolCall("", "get_pto_balance", {"employee": "bob", "as_of": "2026-10-02"})
    return f"Bob will have {json.loads(last['content'])['days_left']} PTO days left on 2026-10-02."


def handoff_policy(system, messages, tools):
    if "refund" in last_user_text(messages).lower():
        return ToolCall("", HANDOFF_TOOL, {"reason": "Refund requests over $500 need a finance approver."})
    return "OK."


def cumulative_input_tokens(n_steps: int, tokens_per_step: int) -> int:
    """Total input tokens for an n-step run when step k re-sends k·t tokens of history."""
    return tokens_per_step * n_steps * (n_steps + 1) // 2


def figures() -> dict:
    """Plots computed from this lesson's own code. matplotlib is imported here
    so the lesson itself needs only NumPy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figs = {}

    # 1. The budget-burner run, step by step.
    budget = 12_000
    run = run_agent(
        ScriptedLLM(budget_burner_policy), "Tell me everything about PTO.", DEMO_TOOLS, DEMO_TOOL_DEFS,
        config=AgentConfig(max_steps=50, max_total_tokens=budget),
    )
    per_step = [s.usage.input_tokens + s.usage.output_tokens for s in run.steps]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(per_step)), per_step, color="#7aa6d8", label="tokens sent this step")
    ax.plot(range(len(per_step)), np.cumsum(per_step), "o-", color="#c0392b", label="running total")
    ax.axhline(budget, ls="--", color="gray", label=f"budget ({budget:,})")
    ax.set(xlabel="step", ylabel="tokens", title="A runaway agent: each call re-sends the whole history")
    ax.legend()
    figs["tokens_per_step"] = fig

    # 2. Linear intuition vs. quadratic reality.
    n = np.arange(1, 31)
    t = 500
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n, t * n, label="if each step cost the same (linear)")
    ax.plot(n, [cumulative_input_tokens(int(k), t) for k in n], label="re-sending history (quadratic)")
    ax.set(xlabel="steps in the run", ylabel="total input tokens", title=f"Cumulative input tokens ({t} new tokens per step)")
    ax.legend()
    figs["quadratic_cost"] = fig

    # 3. Sequential vs. concurrent tool execution.
    latencies = np.random.default_rng(0).uniform(0.2, 1.2, 8)
    k = np.arange(1, 9)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k, [latencies[:i].sum() for i in k], "o-", label="sequential (sum of latencies)")
    ax.plot(k, [latencies[:i].max() for i in k], "o-", label="concurrent (max latency)")
    ax.set(xlabel="tool calls requested in one turn", ylabel="wall-clock seconds", title="Parallel tool calls: time is the slowest call, not the sum")
    ax.legend()
    figs["parallel_latency"] = fig
    return figs


def demo() -> None:
    banner("1. Happy path: parallel tool calls, then an answer")
    r = run_agent(ScriptedLLM(happy_policy), "How many PTO days does alice have left, and what rolls over?", DEMO_TOOLS, DEMO_TOOL_DEFS)
    say(f"stop_cause={r.stop_cause}, steps={len(r.steps)}, tool calls={r.tool_call_count}")
    say(f"Final answer: {r.final_text}")
    say(
        """
        Step 0 asked for two tools in one turn. They ran concurrently and both
        results went back in a single user message, which is what the API expects.
        """
    )

    banner("2. A model that loops: caught by loop detection")
    r = run_agent(ScriptedLLM(looping_policy), "What's the PTO rollover rule?", DEMO_TOOLS, DEMO_TOOL_DEFS)
    say(f"stop_cause={r.stop_cause}: {r.detail}")
    takeaway("Detect the same tool with the same canonical arguments; stop after N repeats instead of paying for N more.")

    banner("3. A model that burns budget: caught by the token budget")
    r = run_agent(
        ScriptedLLM(budget_burner_policy), "Tell me everything about PTO.", DEMO_TOOLS, DEMO_TOOL_DEFS,
        config=AgentConfig(max_steps=50, max_total_tokens=12_000),
    )
    table(
        ["step", "input tokens", "output tokens"],
        [(s.index, s.usage.input_tokens, s.usage.output_tokens) for s in r.steps],
    )
    say(f"stop_cause={r.stop_cause}: {r.detail}. Estimated cost ${r.cost_usd:.4f}.")
    say(
        """
        Input tokens per step climb steadily because every call re-sends the
        whole conversation. Total input grows with the square of the step count.
        """
    )

    banner("4. Recovering from a tool error")
    r = run_agent(ScriptedLLM(recovering_policy), "How many PTO days will bob have on Oct 2?", DEMO_TOOLS, DEMO_TOOL_DEFS)
    for s in r.steps:
        for ex in s.tool_executions:
            print(f"  step {s.index}: {ex.name}({ex.input}) -> {'ERROR: ' if ex.is_error else ''}{ex.content}")
    print()
    say(f"stop_cause={r.stop_cause}. Final: {r.final_text}")
    takeaway("Errors are results, not crashes. An actionable message lets the model fix its own call.")

    banner("5. Hand-off to a human")
    r = run_agent(ScriptedLLM(handoff_policy), "Please refund my $900 conference ticket.", DEMO_TOOLS, DEMO_TOOL_DEFS)
    say(f"stop_cause={r.stop_cause}. Reason: {r.final_text}")


if __name__ == "__main__":
    demo()
