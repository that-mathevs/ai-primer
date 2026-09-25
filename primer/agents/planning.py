r"""
# Multi-step planning: plans, checks, retries and why long tasks fail

Run: `python -m primer.agents.planning`

An agent that does one thing is easy. An agent that does twenty things in a
row mostly fails, for a reason that is pure arithmetic. This lesson measures
that arithmetic, then builds the three remedies: **decompose** the task into
steps you can check, **verify** each step with something outside the model,
and **checkpoint** so a failure costs one step instead of the whole run.

## 1. Compounding error: why long tasks fail

**Everyday picture.** A relay race with ten runners. Each runner drops the
baton only 1 time in 20. That sounds safe, but the team needs *all ten*
hand-offs to work, and it loses about 4 races in 10.

**Tiny worked example.** Each step of an agent succeeds 95% of the time.

| steps | chance all succeed |
|---|---|
| 1 | 0.95 |
| 2 | 0.95 × 0.95 = 0.9025 |
| 10 | 0.95¹⁰ ≈ 0.599 |
| 20 | 0.95²⁰ ≈ 0.358 |

$$
P(\text{task succeeds}) = p^{\,n}
$$

**Symbols**

| Symbol | Meaning |
|---|---|
| $p$ | chance a single step succeeds |
| $n$ | number of steps the task needs |

**In words:** multiply the per-step success rate by itself once per step.

**On the example:** $0.95^{10} \approx 0.60$ and $0.95^{20} \approx 0.36$.
A 95%-reliable step is a 60%-reliable 10-step task.

![End-to-end success vs. number of steps](figures/primer.agents.planning.compounding.svg)

**Reading it:** the x-axis is the number of steps in the task, and the y-axis is the
chance the whole task succeeds. Every curve starts near the top and slides
down, and the lower the per-step rate, the faster it falls. Find 10 steps on the
x-axis and read up to the 95% curve: about 0.6. That's why a demo of a
three-step task looks great and the same agent on a real twenty-step job
disappoints.

**In code:** `end_to_end_success` multiplies the per-step rate by itself once
per step, which is the whole formula above.

**Why it matters.** The remedies all attack $p$ or $n$: fewer steps
(higher-level tools, see `primer.agents.tools`), checks that catch failures,
retries of just the failed step, and human review at critical points.

## 2. Plan-and-execute, with replanning

**Everyday picture.** Cooking from a recipe. You read the whole recipe first
(the *plan*), then do the steps. Halfway through you discover you're out of
eggs. You don't start over, and you don't pretend you have eggs. You revise
the rest of the recipe with what you've got, keeping everything you've
already chopped.

**Tiny worked example.** Goal "Reconcile Q3 invoices". The planner's first
plan uses an old API. Here's the actual run:

```text
planner  -> {"steps": ["fetch_payments", "fetch_invoices_v1", "match", "draft_summary"]}
execute  fetch_payments      ok
execute  fetch_invoices_v1   failed: invoices API v1 was retired on 2026-07-01; use fetch_invoices_v2
planner  <- "Completed so far: fetch_payments. Step fetch_invoices_v1 failed: ...retired..."
planner  -> {"steps": ["fetch_invoices_v2", "match", "draft_summary"]}
execute  fetch_invoices_v2   ok
execute  match               ok
execute  draft_summary       ok     (fetch_payments was NOT run again)
```

```mermaid
flowchart TD
  G[Goal] --> P[Planner writes a plan]
  P --> X[Execute next step]
  X --> C{Step result<br/>meets expectations?}
  C -->|yes| M{More steps?}
  M -->|yes| X
  M -->|no| D[Done]
  C -->|no| R{Replans left?}
  R -->|yes| F[Tell the planner:<br/>what's done + why it failed]
  F --> P
  R -->|no| S[Stop: report the failure]
```

**Reading it:** the inner loop (execute → check → next) is the plan being
followed. The outer loop back to *Planner* only happens on a failed check,
and it carries two facts: what's already done (so work is kept) and why the
step failed (so the new plan can route around it). The *Replans left?*
diamond stops a planner that can't adapt from looping forever.

**The code.** `plan_and_execute(planner, goal)` asks for a JSON plan, runs
each action, records outputs in `state`, and on `StepFailed` asks for a
revised plan. Steps already in `state` are skipped.

**In code:** `PlanRun` is what `plan_and_execute` returns: every plan the
planner wrote, each executed step with "ok" or its failure, and the replan count.

**Why it matters.** Deciding one step at a time (a pure ReAct loop, see
`primer.agents.agent_loop`) drifts on long tasks. An upfront plan keeps the
agent on track and lets a person see its intent before it acts. The risk is
following a plan that reality has invalidated, which is why replanning exists.

## 3. Decomposition into verifiable subtasks

**Everyday picture.** Moving house. "Move house" isn't something you can check off,
but "every box labelled", "van booked for Saturday" and "keys handed back"
are. Each has a clear definition of done.

**Tiny worked example.** "Reconcile Q3 invoices" becomes five subtasks, each
with a check:

| subtask | output | check (definition of done) |
|---|---|---|
| fetch_invoices | 4 invoices | at least one invoice returned |
| fetch_payments | 3 payments | at least one payment returned |
| match | 4 rows: invoiced vs paid | each invoice appears exactly once |
| list_mismatches | INV-102 (1200 vs 1100), INV-104 (450 vs 0) | none |
| draft_summary | "Q3 reconciliation: 4 invoices, 3 payments, 2 mismatches (INV-102 short by 100, INV-104 unpaid)." | mentions the mismatches |

When the payments API times out once, **only `fetch_payments` runs again**:
attempts are `{fetch_invoices: 1, fetch_payments: 2, match: 1, ...}`.

```mermaid
flowchart LR
  A[fetch_invoices] --> C[match]
  B[fetch_payments] --> C
  C --> D[list_mismatches]
  D --> E[draft_summary]
  B -. timeout .-> B
```

**Reading it:** arrows show which outputs feed which step. The dotted
self-loop on `fetch_payments` is a retry. Because every step's output is
kept (a *checkpoint*), the retry doesn't redo `fetch_invoices`.

**In code:** `Subtask` pairs a step's work with its check (the definition of
done); `run_subtasks` runs them in order, retries only the step whose check
failed, and returns a `RunReport` of outputs and attempts. `reconcile_q3`
builds the five subtasks in the table.

### Checkpoints: how much work a failure costs

Without checkpoints, a failure anywhere means starting again from step one.
With them, you retry just the failed step. The expected number of step
executions to finish an $n$-step task:

$$
\text{with checkpoints: } \frac{n}{p}
\qquad
\text{restart from scratch: } \frac{1 - p^{n}}{(1-p)\,p^{n}}
$$

**Symbols**

| Symbol | Meaning |
|---|---|
| $p$ | chance a single step attempt succeeds (failures are noticed) |
| $n$ | number of steps |

**In words:** with checkpoints each step needs on average $1/p$ attempts, so
$n$ steps need $n/p$. Restarting from scratch needs $n$ successes *in a row*,
and the expected wait for a run of $n$ successes grows roughly like $1/p^n$.

**On the example:** $p = 0.95$, $n = 10$: with checkpoints
$10 / 0.95 \approx 10.5$ step runs; restarting from scratch
$(1 - 0.599) / (0.05 \times 0.599) \approx 13.4$. At $n = 50$ it's about
52.6 vs. 240.

![Expected step executions: checkpoints vs. restarting](figures/primer.agents.planning.checkpoints.svg)

**Reading it:** the x-axis is task length, and the y-axis (log scale) is how
many step executions you should expect to pay for. With checkpoints the line
is nearly straight, just slightly above $n$. Restarting from scratch curves
upward exponentially. Long tasks without checkpoints aren't just
unreliable; they're expensive. **Durable execution** is the engineering name
for this: a workflow engine that saves each step's result so a crashed run
resumes where it stopped.

**In code:** `expected_step_runs` evaluates both formulas: $n/p$ with
checkpoints, the restart formula without.

## 4. Reflection vs. external verification

**Everyday picture.** Proofreading your own essay versus having someone run the
numbers in it. You read what you *meant* to write, so your own blind spots
stay blind. A calculator doesn't share them.

**Tiny worked example.** The model writes a leap-year function:

```python
def is_leap(year):
    return year % 4 == 0        # forgets that 1900 was not a leap year
```

*Reflection* (asking a model to review it) replies "Looks correct: years
divisible by 4 are leap years." *External verification* (running it against
known answers) replies `is_leap(1900) returned True, expected False`. Fed
that failure, the model's second draft passes every case.

```mermaid
flowchart LR
  G[Model writes code] --> R1[Reflection:<br/>model reads its own code]
  R1 -->|same blind spot| A1[Approved, still wrong]
  G --> V[Verification:<br/>run tests / check schema / query the DB]
  V -->|1900 fails| F[Failure fed back]
  F --> G2[Second draft]
  G2 --> V2[Checks pass]
```

**Reading it:** two paths leave the same first draft. The top path stays inside
the model and ends at "approved, still wrong". The bottom path goes through
something outside the model (tests, a schema validator, a database query),
and that something produces a concrete, checkable failure the model can fix.

**In code:** `self_review` is the top path (a model reads the code and
approves or not); `run_checks` is the bottom path (it runs the code against
known answers); `generate_until_checks_pass` loops the bottom path, feeding
each failure back to the model until the checks pass.

With verification and retries, the per-step success rate rises:

$$
p_{\text{step}} = p \sum_{i=0}^{r} \big((1-p)\,d\big)^{i}
$$

**Symbols**

| Symbol | Meaning |
|---|---|
| $p$ | chance one attempt succeeds |
| $d$ | chance the check *detects* a failed attempt |
| $r$ | retries allowed after a detected failure |

**In words:** you succeed on the first try, or fail *and notice* and succeed
on the next try, and so on. Failures you don't notice get no retry.

**On the example:** $p = 0.95$, perfect check $d = 1$, one retry:
$0.95 + 0.05 \times 0.95 = 0.9975$ per step, so ten steps succeed
$0.9975^{10} \approx 97.5\%$ of the time instead of 60%. With a check that
catches only half the failures ($d = 0.5$): $0.97375^{10} \approx 77\%$.

![Ten-step success with and without verification](figures/primer.agents.planning.verification.svg)

**Reading it:** all curves use the same 95%-reliable step. The bottom curve has
no checks. The middle ones add a retry after a failure is caught, with a
check that catches half or all failures. The top curve allows two retries.
The gap between "half" and "all" is the lesson: retries are only as good
as the check that triggers them. Invest in the check.

**In code:** `step_success` computes $p_{\text{step}}$ from $p$, $d$ and $r$;
feed its result into `end_to_end_success` to get the ten-step curves.

## In 20 seconds
- Success compounds: $p^n$. 95% per step is 60% over ten steps.
- Plan first, execute step by step, replan from the point of failure, and keep finished work.
- Decompose into subtasks with explicit checks; retry only the failed step (checkpoints).
- Self-review shares the model's blind spots; tests, schemas and database checks don't.
- Retries help only for failures you detect, so the check matters more than the retry.

## Self-test questions

**Q: Each step of your agent is 95% reliable. How reliable is a 20-step task, and what do you do about it?**
A: $0.95^{20} \approx 36\%$. Cut the steps (higher-level tools), add
external checks after key steps with a retry of just that step, checkpoint
so failures don't restart the run, and put human review where a mistake is
expensive. With perfect checks and one retry, per-step reliability becomes
99.75%, and 20 steps succeed about 95% of the time.

**Q: Why isn't "ask the model to double-check its work" enough?**
A: The reviewer shares the author's blind spots, and self-critique can
confidently reinforce a wrong answer. Checks outside the model (tests,
schema validation, a query against the source of truth, a separate grader
model with a rubric) produce concrete failures the model can act on.

**Q: When is plan-and-execute better than deciding one step at a time?**
A: For long, multi-part tasks where staying on track matters and where
showing the plan to a person before acting is valuable. Always pair it with
replanning. A plan is a hypothesis, and the first failed step is evidence.

**Q: What makes a subtask "good"?**
A: It's small, concrete and verifiable. It has a definition of done that code
can check, and its output is saved so a later failure doesn't redo it.

## The papers behind this lesson

- **Wang et al., *Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models* (2023).**
  https://arxiv.org/abs/2305.04091. It showed that asking a model to first
  devise a plan and then carry it out step by step reduces missed steps
  compared with reasoning straight through.
- **Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning* (2023).**
  https://arxiv.org/abs/2303.11366. Agents that turn feedback from the
  environment (failed tests, wrong answers) into written lessons and retry
  improve markedly, and the gains come from *external* signals.
  [annotated companion](../../papers/reflexion.html)

## Further reading
- Anthropic, *Building effective agents*: https://www.anthropic.com/engineering/building-effective-agents
- Lilian Weng, *LLM Powered Autonomous Agents* (planning, reflection): https://lilianweng.github.io/posts/2023-06-23-agent/
- Temporal, durable execution: https://docs.temporal.io/
- LangGraph docs (persistence and checkpoints): https://langchain-ai.github.io/langgraph/
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from primer.agents.llm import LLM, ScriptedLLM, last_user_text


def end_to_end_success(p: float, n_steps: int) -> float:
    """Chance that n independent steps, each succeeding with probability p, all succeed."""
    return p**n_steps


def step_success(p: float, detect: float, retries: int) -> float:
    """Chance one step ends up correct when failures are caught with probability `detect`
    and each caught failure is retried, up to `retries` times.

    Attempt 1 succeeds with p. It fails *and is noticed* with (1-p)·detect, which
    buys another attempt. A failure nobody notices counts as a silent failure.

    ```text
    P = p · (1 + q + q² + ... + q^retries),  where q = (1-p)·detect
    ```
    """
    q = (1 - p) * detect
    return p * sum(q**i for i in range(retries + 1))


def expected_step_runs(p: float, n_steps: int, checkpoints: bool) -> float:
    """Expected number of step executions to get n steps done, retrying until success.

    With checkpoints, each step retries on its own: n / p (a geometric wait per step).
    Without them, any failure throws away all progress, so you need n successes
    *in a row*: (1 - p^n) / ((1 - p) · p^n).
    """
    if checkpoints:
        return n_steps / p
    return (1 - p**n_steps) / ((1 - p) * p**n_steps)


# ---------------------------------------------------------------------------
# Decomposition: small, verifiable subtasks with checks and checkpoints
# ---------------------------------------------------------------------------


@dataclass
class Subtask:
    """One step of a decomposed task.

    `run(outputs)` receives the outputs of earlier steps (by name) and returns
    this step's output. `check(output)` returns a problem description, or None
    when the output is acceptable. The check is the step's definition of done.
    """

    name: str
    run: Callable[[dict[str, Any]], Any]
    check: Callable[[Any], str | None] = lambda out: None


@dataclass
class RunReport:
    status: str  # "done" | "failed"
    outputs: dict[str, Any]
    attempts: dict[str, int]
    failed_step: str | None = None
    problem: str | None = None


def run_subtasks(subtasks: list[Subtask], max_retries: int = 2) -> RunReport:
    """Run subtasks in order, checking each one and retrying only the one that failed."""
    outputs: dict[str, Any] = {}
    attempts: dict[str, int] = {}
    for task in subtasks:
        problem = None
        for _ in range(1 + max_retries):
            attempts[task.name] = attempts.get(task.name, 0) + 1
            try:
                out = task.run(outputs)
                problem = task.check(out)
            except Exception as e:  # noqa: BLE001  (a crash is just another failed attempt)
                problem = f"{type(e).__name__}: {e}"
            if problem is None:
                outputs[task.name] = out
                break
        if problem is not None:
            return RunReport("failed", outputs, attempts, task.name, problem)
    return RunReport("done", outputs, attempts)


INVOICES = [
    {"invoice": "INV-101", "vendor": "Acme", "amount": 1500},
    {"invoice": "INV-102", "vendor": "Globex", "amount": 1200},
    {"invoice": "INV-103", "vendor": "Initech", "amount": 800},
    {"invoice": "INV-104", "vendor": "Umbrella", "amount": 450},
]
PAYMENTS = [
    {"payment": "PAY-9001", "invoice": "INV-101", "amount": 1500},
    {"payment": "PAY-9002", "invoice": "INV-102", "amount": 1100},
    {"payment": "PAY-9003", "invoice": "INV-103", "amount": 800},
]


def _match(outputs: dict[str, Any]) -> list[dict[str, Any]]:
    paid: dict[str, int] = {}
    for pay in outputs["fetch_payments"]:
        paid[pay["invoice"]] = paid.get(pay["invoice"], 0) + pay["amount"]
    return [
        {"invoice": inv["invoice"], "vendor": inv["vendor"], "invoiced": inv["amount"], "paid": paid.get(inv["invoice"], 0)}
        for inv in outputs["fetch_invoices"]
    ]


def _summary(outputs: dict[str, Any]) -> str:
    notes = [
        f"{m['invoice']} unpaid" if m["paid"] == 0 else f"{m['invoice']} short by {m['invoiced'] - m['paid']}"
        for m in outputs["list_mismatches"]
    ]
    return (
        f"Q3 reconciliation: {len(outputs['fetch_invoices'])} invoices, {len(outputs['fetch_payments'])} payments, "
        f"{len(notes)} mismatches ({', '.join(notes)})."
    )


def reconcile_q3(
    fetch_invoices: Callable[[], list[dict]] = lambda: list(INVOICES),
    fetch_payments: Callable[[], list[dict]] = lambda: list(PAYMENTS),
) -> list[Subtask]:
    """"Reconcile Q3 invoices" decomposed into five steps, each with its own definition of done."""
    return [
        Subtask("fetch_invoices", lambda o: fetch_invoices(), lambda out: None if out else "no invoices returned for Q3"),
        Subtask("fetch_payments", lambda o: fetch_payments(), lambda out: None if out else "no payments returned for Q3"),
        # Each invoice must appear exactly once in the match, or rows were duplicated.
        Subtask("match", _match, lambda out: None if len({m["invoice"] for m in out}) == len(out) else "duplicated invoices"),
        Subtask("list_mismatches", lambda o: [m for m in o["match"] if m["paid"] != m["invoiced"]]),
        Subtask("draft_summary", _summary, lambda out: None if "mismatch" in out else "summary does not report mismatches"),
    ]


# ---------------------------------------------------------------------------
# Plan-and-execute with replanning
# ---------------------------------------------------------------------------


class StepFailed(Exception):
    """A step's result didn't meet expectations. The message goes back to the planner."""


def _fetch_invoices_v1(state: dict[str, Any]) -> list[dict]:
    raise StepFailed("invoices API v1 was retired on 2026-07-01; use fetch_invoices_v2")


def _need(state: dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if k not in state]
    if missing:
        raise StepFailed(f"cannot run yet: missing {missing}; fetch them first")


def _match_step(state: dict[str, Any]) -> list[dict]:
    _need(state, "fetch_invoices_v2", "fetch_payments")
    return _match({"fetch_invoices": state["fetch_invoices_v2"], "fetch_payments": state["fetch_payments"]})


def _summary_step(state: dict[str, Any]) -> str:
    _need(state, "match")
    mismatches = [m for m in state["match"] if m["paid"] != m["invoiced"]]
    return _summary({"fetch_invoices": state["fetch_invoices_v2"], "fetch_payments": state["fetch_payments"], "list_mismatches": mismatches})


# action name -> function(state) -> output (stored in state under the action name)
PLAN_ACTIONS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "fetch_invoices_v1": _fetch_invoices_v1,
    "fetch_invoices_v2": lambda state: list(INVOICES),
    "fetch_payments": lambda state: list(PAYMENTS),
    "match": _match_step,
    "draft_summary": _summary_step,
}


@dataclass
class PlanRun:
    status: str  # "done" | "failed"
    replans: int
    executed: list[tuple[str, str]]  # (action, "ok" or "failed: why")
    plans: list[list[str]]
    state: dict[str, Any] = field(default_factory=dict)


def _ask_planner(planner: LLM, goal: str, actions: list[str], done: list[str], feedback: str) -> list[str]:
    prompt = (
        f"Goal: {goal}\n"
        f"Available actions: {', '.join(actions)}\n"
        f"Completed so far: {', '.join(done) or 'nothing'}\n"
        + (f"{feedback}\n" if feedback else "")
        + 'Reply with JSON {"steps": [...]} listing the remaining actions in order.'
    )
    resp = planner.complete(system="You are a planner. Output only JSON.", messages=[{"role": "user", "content": prompt}])
    return list(json.loads(resp.text)["steps"])


def plan_and_execute(
    planner: LLM,
    goal: str,
    actions: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
    max_replans: int = 2,
) -> PlanRun:
    """Ask for a plan, execute it step by step, and replan from where it broke.

    Completed steps are remembered in `state`, so a revised plan never redoes
    finished work. The planner only sees what it needs: the goal, the
    actions, what's done, and why the last step failed.
    """
    actions = actions or PLAN_ACTIONS
    state: dict[str, Any] = {}
    executed: list[tuple[str, str]] = []
    plans: list[list[str]] = []
    replans, feedback = 0, ""
    while True:
        plan = _ask_planner(planner, goal, list(actions), [a for a in state], feedback)
        plans.append(plan)
        failure = None
        for action in plan:
            if action in state:
                continue  # already done in an earlier plan: keep the work
            try:
                if action not in actions:
                    raise StepFailed(f"unknown action {action!r}")
                state[action] = actions[action](state)
                executed.append((action, "ok"))
            except StepFailed as e:
                executed.append((action, f"failed: {e}"))
                failure = f"Step {action} failed: {e}"
                break
        if failure is None:
            return PlanRun("done", replans, executed, plans, state)
        if replans >= max_replans:
            return PlanRun("failed", replans, executed, plans, state)
        replans += 1
        feedback = failure


def demo_planner() -> ScriptedLLM:
    """A scripted planner that starts with the old API and adapts when told it's retired."""

    def policy(system, messages, tools):
        if "retired" in last_user_text(messages):
            return '{"steps": ["fetch_invoices_v2", "match", "draft_summary"]}'
        return '{"steps": ["fetch_payments", "fetch_invoices_v1", "match", "draft_summary"]}'

    return ScriptedLLM(policy)


# ---------------------------------------------------------------------------
# Reflection vs. external verification
# ---------------------------------------------------------------------------

BUGGY_LEAP = "def is_leap(year):\n    return year % 4 == 0\n"
FIXED_LEAP = "def is_leap(year):\n    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)\n"

# (year, is it a leap year?) The century cases are the ones a quick read misses.
LEAP_CASES = [(2024, True), (2023, False), (1900, False), (2000, True)]


def self_review(critic: LLM, code: str) -> tuple[bool, str]:
    """Ask a model to review code by reading it. Returns (approved, its note)."""
    resp = critic.complete(
        system="Review the code. Start your reply with 'Looks correct' or 'Bug:'.",
        messages=[{"role": "user", "content": code}],
    )
    return resp.text.startswith("Looks correct"), resp.text


def run_checks(code: str) -> list[str]:
    """External verification: actually run the code against known answers."""
    namespace: dict[str, Any] = {}
    exec(code, namespace)  # our own generated snippet, run in an isolated namespace
    fn = namespace["is_leap"]
    return [f"is_leap({y}) returned {fn(y)}, expected {want}" for y, want in LEAP_CASES if fn(y) != want]


def confident_critic() -> ScriptedLLM:
    """A reviewer that shares the author's blind spot, which is the usual failure of self-critique."""
    return ScriptedLLM(["Looks correct: years divisible by 4 are leap years."])


def fixing_coder() -> ScriptedLLM:
    """Writes the buggy version first, then fixes it once shown a failing case."""

    def policy(system, messages, tools):
        return FIXED_LEAP if "1900" in last_user_text(messages) else BUGGY_LEAP

    return ScriptedLLM(policy)


def generate_until_checks_pass(coder: LLM, task: str, max_attempts: int = 3) -> tuple[str, int]:
    """Generate, verify externally, feed failures back, repeat. Returns (code, attempts)."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
    code = ""
    for attempt in range(1, max_attempts + 1):
        code = coder.complete(system="Reply with Python code only.", messages=messages).text
        failures = run_checks(code)
        if not failures:
            return code, attempt
        messages += [
            {"role": "assistant", "content": code},
            {"role": "user", "content": "These checks failed:\n" + "\n".join(failures)},
        ]
    return code, max_attempts


# ---------------------------------------------------------------------------
# Figures and walkthrough
# ---------------------------------------------------------------------------


def figures() -> dict:
    """Plots computed from this lesson's own formulas (matplotlib imported here)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = {}
    steps = list(range(1, 31))

    fig, ax = plt.subplots(figsize=(7, 4))
    for p in (0.90, 0.95, 0.99):
        ax.plot(steps, [end_to_end_success(p, n) for n in steps], "o-", ms=3, label=f"{p:.0%} per step")
    ax.axvline(10, ls=":", color="gray")
    ax.set(xlabel="steps in the task (n)", ylabel="P(task succeeds) = p^n", ylim=(0, 1.02),
           title="Compounding error: reliable steps, unreliable tasks")
    ax.legend()
    figs["compounding"] = fig

    fig, ax = plt.subplots(figsize=(7, 4))
    variants = [
        ("no checks", 0.95),
        ("check catches half, 1 retry", step_success(0.95, 0.5, 1)),
        ("check catches all, 1 retry", step_success(0.95, 1.0, 1)),
        ("check catches all, 2 retries", step_success(0.95, 1.0, 2)),
    ]
    for label, ps in variants:
        ax.plot(steps, [end_to_end_success(ps, n) for n in steps], label=f"{label} (step = {ps:.4f})")
    ax.set(xlabel="steps in the task (n)", ylabel="P(task succeeds)", ylim=(0, 1.02),
           title="Same 95% step, different verification")
    ax.legend(fontsize=8)
    figs["verification"] = fig

    fig, ax = plt.subplots(figsize=(7, 4))
    ns = list(range(1, 61))
    ax.plot(ns, [expected_step_runs(0.95, n, True) for n in ns], label="checkpoints: retry the failed step (n/p)")
    ax.plot(ns, [expected_step_runs(0.95, n, False) for n in ns], label="no checkpoints: restart from step 1")
    ax.set_yscale("log")
    ax.set(xlabel="steps in the task (n)", ylabel="expected step executions (log scale)",
           title="What a failure costs (p = 0.95 per step)")
    ax.legend()
    figs["checkpoints"] = fig
    return figs


def demo() -> None:
    from primer._show import banner, say, table, takeaway

    banner("1. Compounding error")
    table(["steps", "90%/step", "95%/step", "99%/step"],
          [(n, *(end_to_end_success(p, n) for p in (0.90, 0.95, 0.99))) for n in (1, 5, 10, 20, 50)], floatfmt=".3f")
    takeaway("A 95%-reliable step is a 60%-reliable ten-step task.")

    banner("2. Plan-and-execute with replanning")
    run = plan_and_execute(demo_planner(), "Reconcile Q3 invoices")
    for i, plan in enumerate(run.plans):
        print(f"  plan {i + 1}: {plan}")
    print()
    for action, result in run.executed:
        print(f"  {action:18} {result}")
    print()
    say(f"status={run.status}, replans={run.replans}. fetch_payments ran once: finished work is kept.")
    say(f"Summary: {run.state['draft_summary']}")

    banner("3. Decomposition: retry only the failed step")
    calls: list[int] = []

    def flaky_payments() -> list[dict]:
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("payments API timed out")
        return list(PAYMENTS)

    report = run_subtasks(reconcile_q3(fetch_payments=flaky_payments))
    table(["subtask", "attempts"], list(report.attempts.items()))
    for m in report.outputs["list_mismatches"]:
        print(f"  mismatch: {m}")
    print()
    table(["steps", "step runs with checkpoints", "step runs restarting"],
          [(n, expected_step_runs(0.95, n, True), expected_step_runs(0.95, n, False)) for n in (5, 10, 20, 50)], floatfmt=".1f")

    banner("4. Reflection vs. external verification")
    approved, note = self_review(confident_critic(), BUGGY_LEAP)
    print(BUGGY_LEAP)
    say(f"Self-review: approved={approved}: {note!r}")
    say(f"Running the test cases: {run_checks(BUGGY_LEAP)}")
    code, attempts = generate_until_checks_pass(fixing_coder(), "Write is_leap(year).")
    say(f"Feeding the failure back: passes on attempt {attempts}.")
    table(["verification", "per-step", "10-step task"],
          [(label, ps, end_to_end_success(ps, 10)) for label, ps in (
              ("none", 0.95), ("catches half, 1 retry", step_success(0.95, 0.5, 1)),
              ("catches all, 1 retry", step_success(0.95, 1.0, 1)))], floatfmt=".4f")
    takeaway("Retries are only as good as the check that triggers them. Check outside the model.")


if __name__ == "__main__":
    demo()
