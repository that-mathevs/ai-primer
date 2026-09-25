r"""
# Why the hard ones fail: a catalogue of agent failures, and the fix for each

Run: `python -m primer.agents.failures`

## The everyday picture

Pilots learn from a catalogue of accidents: each entry names what went
wrong, how it showed up in the cockpit, and the checklist item that now
prevents it. This lesson is that catalogue for agents: ten failures that
keep recurring in production systems, each with its symptom, its fix, and a
pointer to the lesson that builds the fix. Three of them get their fix
built right here: compounding error (arithmetic), brittle integrations
(retries with backoff, and circuit breakers) and runaway loops (loop
detection).

| Failure | Symptom | Fix | Lesson |
|---|---|---|---|
| Compounding error | Long tasks fail far more often than short ones | Fewer steps, checks after key steps, checkpoints | `primer.agents.planning` |
| Bad retrieval | Confident, wrong answers | Hybrid search, reranking, retrieval evals | `primer.agents.rag` |
| Ambiguous tools | Wrong tool, or bad arguments | Precise descriptions, fewer tools, validation | `primer.agents.tools` |
| Context rot | Quality drops as a session grows | Summarize, trim, restart with a state handoff | `primer.agents.context` |
| Loops and runaway | The same call repeated; cost spikes | Step budgets, loop detection, stop conditions | `primer.agents.agent_loop` |
| No evals | Regressions ship silently | Golden sets in CI, online monitoring | `primer.agents.evals` |
| Prompt injection | The agent obeys instructions found in data | Untrusted-content boundaries, privilege separation | `primer.agents.guardrails` |
| Messy enterprise data | Garbled tables, missing permissions | Invest in parsing; permission-aware retrieval | `primer.agents.rag` |
| Brittle integrations | Timeouts, expired credentials, API changes | Retries with backoff, circuit breakers, contract tests | `primer.agents.failures` |
| No adoption | It works, and nobody uses it | Build with users, show sources, easy human handoff | `primer.agents.deployment` |

**In code:** `FAILURES` is this table as data, one `Failure` record per row.

## Compounding error

**Everyday picture.** A relay race where each baton pass succeeds 95% of
the time. One pass is nearly safe; ten passes in a row drop the baton more
often than you'd think.

**Worked example.** If each step of an agent's task succeeds 95% of the
time, independently:

| Steps | Chance every step succeeds |
|---|---|
| 1 | 0.95 |
| 2 | 0.95 × 0.95 = 0.9025 |
| 10 | 0.95¹⁰ ≈ **0.599** |
| 20 | 0.95²⁰ ≈ **0.358** |

$$
P(\text{task succeeds}) = p^{\,n}
$$

**Symbols**

| Symbol | Meaning here | Worked example |
|---|---|---|
| $p$ | chance a single step succeeds | 0.95 |
| $n$ | number of steps that must all succeed | 10 |
| $p^{\,n}$ | $p$ multiplied by itself $n$ times | 0.599 |

**In words:** the chance that every step succeeds is the per-step chance
multiplied together once per step.

**On the worked example:** 0.95 multiplied by itself 10 times is 0.599, so a
ten-step task at 95% per step fails four times in ten.

**In Python:**

```python
>>> p, n = 0.95, 10
>>> round(p ** n, 3)                      # p multiplied by itself n times
0.599
>>> round(1 - p ** n, 1)                  # fails about four times in ten
0.4
```

```mermaid
flowchart LR
  S1[Step 1<br/>95%] --> S2[Step 2<br/>95%] --> S3[...] --> S10[Step 10<br/>95%] --> D[Done:<br/>60% of the time]
  S2 -.->|verify, retry<br/>just this step| S2
```

**Reading it:** every arrow is another chance to drop the baton, so success
shrinks with each step. The dotted loop is the remedy: check a step's
result where it happens and retry only that step, so an error doesn't
travel down the chain. Fewer steps, verification after key steps,
checkpoints to resume from the middle, and human review at critical points
all attack the same exponent.

![Chance of finishing a task vs. number of steps, for three per-step reliabilities](figures/primer.agents.failures.compounding.svg)

**Reading it:** the x-axis is the number of steps; each curve is a per-step
reliability. At 99% per step a 20-step task still succeeds 82% of the time;
at 95% it's 36%; at 90%, 12%. Small gains in per-step reliability are worth
far more than they look, and demos with three steps say little about tasks
with twenty.

**In code:** `chain_success` is $p^{\,n}$; `max_steps_for` runs it backwards,
returning the most steps you can chain and still reach a target success rate.

## Brittle integrations: retries with backoff

**Everyday picture.** Calling a busy phone line: you don't redial every
second. You wait a little, then longer, then longer still, and if everyone
redials on the same schedule the line stays jammed, so you add a random
pause (**jitter**).

**Worked example.** A service times out twice and then answers. With a base
delay of 0.1 s the waits are 0.1 s, then 0.2 s, and the third attempt
succeeds. With a cap of 10 s, the waits never exceed 10 s however many
attempts there are. A request that's *invalid* (a `ValueError` here) fails
the same way every time, so it's raised immediately: retrying only adds
load.

$$
d_k = \min\bigl(D_{\max},\; b \cdot 2^{k}\bigr), \qquad k = 0, 1, 2, \ldots
$$

**Symbols**

| Symbol | Meaning here | Worked example |
|---|---|---|
| $d_k$ | wait before retry number $k+1$ | $d_0$ = 0.1 s, $d_1$ = 0.2 s |
| $k$ | how many retries have already happened | 0, 1 |
| $b$ | base delay | 0.1 s |
| $2^{k}$ | doubles with every retry | 1, 2, 4, … |
| $D_{\max}$ | the cap on any single wait | 10 s |

**In words:** each wait doubles the previous one, starting from the base
delay, but never exceeds the cap.

**On the worked example:** $d_0$ = min(10, 0.1 × 1) = 0.1 s and
$d_1$ = min(10, 0.1 × 2) = 0.2 s.

**In Python:**

```python
>>> b, D_max = 0.1, 10
>>> [min(D_max, b * 2 ** k) for k in range(2)]   # d_0 and d_1
[0.1, 0.2]
```

```mermaid
sequenceDiagram
  participant A as Agent tool
  participant S as Upstream API
  A->>S: request
  S--xA: timeout
  Note over A: wait 0.1 s
  A->>S: retry 1
  S--xA: timeout
  Note over A: wait 0.2 s
  A->>S: retry 2
  S-->>A: 200 OK
```

**Reading it:** time runs downward. Each failure is followed by a longer
pause, which gives an overloaded service room to recover instead of being
hammered. Writes must be **idempotent** (safe to repeat: sending the same
request twice has the same effect as once, usually via an idempotency key)
before you retry them, or a retried "create order" makes two orders.

![Backoff delays per attempt, without and with jitter](figures/primer.agents.failures.backoff.svg)

**Reading it:** the x-axis is the attempt number; the black line is the
capped exponential delay. The dots are five clients using *full jitter*
(each waits a random time between zero and the capped delay): instead of
all retrying at the same moment, their retries spread out, which is what
lets a recovering service actually recover.

**In code:** `backoff_delays` lists the capped waits $d_k$;
`retry_with_backoff` calls a function, retries only the exceptions in
`RETRYABLE` with those waits (optionally with full jitter), and raises
anything else at once.

## Brittle integrations: circuit breakers

**Everyday picture.** The breaker in your home's fuse box. When a circuit
keeps shorting, it trips and cuts the power, instead of letting the wire
overheat. After a while you flip it back on to test; if it trips again, it
stays off.

A **circuit breaker** wraps calls to a dependency. After several
consecutive failures it **opens**: calls fail immediately without touching
the struggling service (fail fast), so the agent can fall back (use a cache,
tell the user, hand off to a person) instead of hanging on timeouts. After a
cool-down it goes **half-open** and lets one trial call through: success
closes it, failure opens it again.

**Worked example.** Threshold 3, cool-down 30 s. Three timeouts in a row:
open. A fourth call at 5 s fails instantly with `CircuitOpen` and the
service isn't called at all. At 31 s, one trial call goes through; it
succeeds, so the breaker closes.

```mermaid
stateDiagram-v2
  [*] --> Closed
  Closed --> Closed: success (reset the failure count)
  Closed --> Open: 3 failures in a row
  Open --> Open: calls rejected instantly
  Open --> HalfOpen: cool-down elapsed
  HalfOpen --> Closed: trial call succeeds
  HalfOpen --> Open: trial call fails
```

**Reading it:** in the normal state (Closed) calls flow through and failures
are counted. Open is the protective state: nothing reaches the dependency.
Half-open is a single, cautious test. The breaker turns a slow, cascading
failure (every request waiting 30 s on a dead service) into a fast,
contained one.

![Circuit breaker during a 50-second outage](figures/primer.agents.failures.breaker.svg)

**Reading it:** the shaded band is when the upstream service is down;
each marker is one request. Before the outage, calls succeed (green). The
first three failures (red) open the breaker; after that, requests fail fast
(grey) without reaching the service, apart from one trial call per
cool-down. Once the service is back, the next trial succeeds and traffic
resumes. The service received a handful of calls during the outage instead
of all of them.

**In code:** `CircuitBreaker.call` is the state diagram: it raises
`CircuitOpen` while open, lets one trial call through after the cool-down,
and closes on success. `simulate_outage` replays the outage in the figure.

**Contract tests** (automated checks that an external API still accepts
and returns what your tool expects) catch the third kind of brittleness,
upstream API changes, before users do.

## Loops and runaway

**Everyday picture.** A satnav that keeps rerouting you around the same
block. The fix isn't a better map; it's noticing that you've passed the same
corner three times.

**Worked example.** `search("vpn")`, `search("vpn")`, `search("vpn")`: the
same tool with the same arguments three times in a row. Nothing new can
come back, so the agent is looping. `search("vpn")`, `search("vpn error")`,
`search("ERR-4012")` is the same tool refining its query: progress.
Combine loop detection with hard step and token budgets
(`primer.agents.cost.TaskBudget`) so a confused agent stops and hands off
instead of burning money.

**In code:** `is_looping` reports whether the last few calls were the same
tool with identical arguments.

## In 20 seconds
- Long tasks fail multiplicatively: at 95% per step, 10 steps succeed 60%
  of the time. Shorten chains and verify at key steps.
- Retry transient failures with capped exponential backoff and jitter; never
  retry invalid requests; make writes idempotent first.
- Put circuit breakers around dependencies so outages fail fast instead of
  cascading.
- Detect loops (same call, same arguments) and enforce step and token
  budgets.
- Most failures are system failures (retrieval, tools, data, evals,
  adoption), not model failures, and each has a known fix.

## Self-test questions

**Your agent works in demos and fails half the time in production. Where do
you look first?**
At the length of real tasks versus demo tasks (compounding error), and at
traces for the first failing step: retrieval misses, wrong tools or
arguments, tool errors from integrations, context growth. Then fix
per-step reliability where it's lowest, add verification after key steps,
and put the failing cases into the eval set.

**Why add jitter to retries?**
Without it, every client that failed at the same moment retries at the same
moment, producing synchronized waves of load that keep an overloaded
service down. Random delays spread the retries out.

**When should you not retry?**
When the error can't go away on its own: invalid input, permission denied,
not found. And never retry a non-idempotent write without an idempotency
key, or you'll duplicate its effect.

**What's the difference between a retry and a circuit breaker?**
A retry handles a blip for one request. A breaker handles an outage across
many requests: it stops sending traffic to a failing dependency, fails fast
so callers can fall back, and probes for recovery.

## The papers behind this lesson

No single founding paper; these are engineering patterns. The canonical
write-ups are Michael Nygard's *Release It!* (which named the circuit
breaker) and the AWS Builders' Library article on backoff and jitter, both
linked below.

## Further reading
- Martin Fowler, *CircuitBreaker*: https://martinfowler.com/bliki/CircuitBreaker.html
- AWS Builders' Library, *Timeouts, retries, and backoff with jitter*: https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/
- Anthropic, *Building effective agents*: https://www.anthropic.com/engineering/building-effective-agents
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from primer._show import banner, say, table, takeaway

# ---------------------------------------------------------------------------
# 1. The catalogue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Failure:
    name: str
    symptom: str
    fix: str
    module: str


FAILURES: list[Failure] = [
    Failure("Compounding error", "Long tasks fail far more than short ones", "Fewer steps, verify key steps, checkpoints", "primer.agents.planning"),
    Failure("Bad retrieval", "Confident wrong answers", "Hybrid search, reranking, retrieval evals", "primer.agents.rag"),
    Failure("Ambiguous tools", "Wrong tool or bad arguments", "Precise descriptions, fewer tools, validation", "primer.agents.tools"),
    Failure("Context rot", "Quality drops as sessions grow", "Summarize, trim, restart with a state handoff", "primer.agents.context"),
    Failure("Loops and runaway", "Repeated calls, cost spikes", "Step budgets, loop detection, stop conditions", "primer.agents.agent_loop"),
    Failure("No evals", "Regressions ship silently", "Golden sets in CI, online monitoring", "primer.agents.evals"),
    Failure("Prompt injection", "Agent follows instructions found in data", "Untrusted-content boundaries, privilege separation", "primer.agents.guardrails"),
    Failure("Messy enterprise data", "Garbled parsing, permission gaps", "Invest in ingestion, permission-aware retrieval", "primer.agents.rag"),
    Failure("Brittle integrations", "Timeouts, auth expiry, API changes", "Retries with backoff, circuit breakers, contract tests", "primer.agents.failures"),
    Failure("No adoption", "Works, but nobody uses it", "Design with users, show sources, easy human handoff", "primer.agents.deployment"),
]

# ---------------------------------------------------------------------------
# 2. Compounding error
# ---------------------------------------------------------------------------


def chain_success(p: float, n: int) -> float:
    """Chance that n independent steps, each succeeding with probability p, all succeed."""
    return p**n


def max_steps_for(p: float, target: float) -> int:
    """The most steps you can chain at per-step reliability p and still reach `target`."""
    # p**n >= target  <=>  n <= log(target) / log(p)   (both logs are negative)
    return math.floor(math.log(target) / math.log(p) + 1e-12)


# ---------------------------------------------------------------------------
# 3. Retries with exponential backoff
# ---------------------------------------------------------------------------

RETRYABLE: tuple[type[Exception], ...] = (TimeoutError, ConnectionError)


def backoff_delays(attempts: int, base_delay: float, max_delay: float) -> list[float]:
    """The capped exponential wait before each retry (no jitter)."""
    return [min(max_delay, base_delay * 2**k) for k in range(attempts - 1)]


def retry_with_backoff(fn: Callable[[], Any], max_attempts: int = 5, base_delay: float = 0.1, max_delay: float = 10.0,
                       jitter: np.random.Generator | None = None, sleep: Callable[[float], None] = time.sleep,
                       retryable: tuple[type[Exception], ...] = RETRYABLE) -> Any:
    """Call fn, retrying transient failures with capped exponential backoff.

    Pass a seeded `np.random.default_rng()` as `jitter` for "full jitter":
    wait a uniform random time between 0 and the capped delay.
    """
    for attempt in range(max_attempts):
        try:
            return fn()
        except retryable:
            if attempt == max_attempts - 1:
                raise
            delay = min(max_delay, base_delay * 2**attempt)
            sleep(float(jitter.uniform(0, delay)) if jitter is not None else delay)
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# 4. Circuit breaker
# ---------------------------------------------------------------------------


class CircuitOpen(RuntimeError):
    """Raised instead of calling a dependency whose breaker is open."""


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, reset_timeout: float = 30.0, clock: Callable[[], float] = time.monotonic):
        self.threshold, self.reset_timeout, self.clock = failure_threshold, reset_timeout, clock
        self.state = "closed"
        self.failures = 0
        self.opened_at = 0.0

    def call(self, fn: Callable[[], Any]) -> Any:
        if self.state == "open":
            if self.clock() - self.opened_at < self.reset_timeout:
                raise CircuitOpen("dependency unavailable; failing fast")
            self.state = "half_open"  # cool-down over: allow one trial call
        try:
            result = fn()
        except Exception:
            self.failures += 1
            if self.state == "half_open" or self.failures >= self.threshold:
                self.state, self.opened_at = "open", self.clock()
            raise
        self.state, self.failures = "closed", 0
        return result


def simulate_outage(outage: tuple[float, float] = (10.0, 60.0), every: float = 2.0, until: float = 100.0) -> list[dict[str, Any]]:
    """Requests every `every` seconds through a breaker while the service is down during `outage`."""
    now = [0.0]
    breaker = CircuitBreaker(failure_threshold=3, reset_timeout=15.0, clock=lambda: now[0])
    events = []

    def service():
        if outage[0] <= now[0] < outage[1]:
            raise TimeoutError("down")
        return "ok"

    t = 0.0
    while t < until:
        now[0] = t
        try:
            breaker.call(service)
            outcome = "ok"
        except CircuitOpen:
            outcome = "fast-fail"
        except TimeoutError:
            outcome = "failed"
        events.append({"t": t, "outcome": outcome, "state": breaker.state})
        t += every
    return events


# ---------------------------------------------------------------------------
# 5. Loop detection
# ---------------------------------------------------------------------------


def is_looping(calls: list[tuple[str, dict[str, Any]]], repeats: int = 3) -> bool:
    """True if the last `repeats` calls are the same tool with identical arguments."""
    if len(calls) < repeats:
        return False
    keys = [(name, json.dumps(args, sort_keys=True)) for name, args in calls[-repeats:]]
    return len(set(keys)) == 1


# ---------------------------------------------------------------------------
# Figures and demo
# ---------------------------------------------------------------------------


def figures() -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs: dict[str, Any] = {}

    n = np.arange(1, 31)
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for p, color in [(0.99, "#55a868"), (0.95, "#4c72b0"), (0.90, "#c44e52")]:
        ax.plot(n, [chain_success(p, k) for k in n], color=color, label=f"{p:.0%} per step")
    for k in (10, 20):
        ax.plot(k, chain_success(0.95, k), "o", color="#4c72b0")
        ax.annotate(f"{chain_success(0.95, k):.0%}", (k, chain_success(0.95, k)), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("steps that must all succeed")
    ax.set_ylabel("chance the whole task succeeds")
    ax.set_ylim(0, 1.02)
    ax.set_title("Compounding error")
    ax.legend()
    fig.tight_layout()
    figs["compounding"] = fig

    attempts = 8
    base = backoff_delays(attempts, 0.5, 10.0)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.plot(range(1, attempts), base, "k-o", label="capped exponential (no jitter)")
    for c in range(5):
        ax.scatter(np.arange(1, attempts) + (c - 2) * 0.06, [rng.uniform(0, d) for d in base], s=18, alpha=0.8,
                   label="five clients with full jitter" if c == 0 else None, color="#4c72b0")
    ax.set_xlabel("retry number")
    ax.set_ylabel("seconds to wait")
    ax.set_title("Backoff: base 0.5 s, doubling, capped at 10 s")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figs["backoff"] = fig

    ev = simulate_outage()
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.axvspan(10, 60, color="#f2d0d0", label="service down")
    colors = {"ok": "#55a868", "failed": "#c44e52", "fast-fail": "#8c8c8c"}
    for kind in colors:
        pts = [e["t"] for e in ev if e["outcome"] == kind]
        ax.scatter(pts, [1] * len(pts), color=colors[kind], s=40, label=kind.replace("fast-fail", "failed fast (breaker open)"))
    ax.set_yticks([])
    ax.set_xlabel("seconds")
    ax.set_title("Circuit breaker (threshold 3, cool-down 15 s) during an outage")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=4, frameon=False)
    fig.tight_layout()
    figs["breaker"] = fig
    return figs


def demo() -> None:
    banner("1. The catalogue")
    table(["failure", "symptom", "fix", "lesson"], [(f.name, f.symptom, f.fix, f.module) for f in FAILURES])

    banner("2. Compounding error")
    table(["steps", "at 99%", "at 95%", "at 90%"], [(k, chain_success(0.99, k), chain_success(0.95, k), chain_success(0.90, k)) for k in (1, 5, 10, 20)], floatfmt=".3f")
    print(f"at 95% per step, the most steps that keep success at 90% or better: {max_steps_for(0.95, 0.90)}")
    print()

    banner("3. Retries with backoff")
    waits: list[float] = []
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] <= 3:
            raise TimeoutError("upstream timed out")
        return "ok"

    print("result:", retry_with_backoff(flaky, base_delay=0.1, sleep=waits.append), "after waits", waits)
    print()

    banner("4. Circuit breaker during an outage")
    ev = simulate_outage()
    reached = sum(e["outcome"] == "failed" for e in ev)
    fast = sum(e["outcome"] == "fast-fail" for e in ev)
    print(f"during the outage the service was hit {reached} times; {fast} calls failed fast instead")
    print()

    banner("5. Loop detection")
    print(is_looping([("search", {"q": "vpn"})] * 3), is_looping([("search", {"q": "vpn"}), ("search", {"q": "vpn error"}), ("search", {"q": "ERR-4012"})]))
    print()
    say("Same tool, same arguments, three times: nothing new can come back. Stop, and hand off.")
    takeaway("Most agent failures are system failures, and each one has a known fix. Build the fix, then test for it.")


if __name__ == "__main__":
    demo()
