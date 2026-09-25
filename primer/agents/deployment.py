r"""
# Safe deployment: letting an agent act in the real world, one earned step at a time

Run: `python -m primer.agents.deployment`

## The everyday picture

A new pilot doesn't get the captain's seat on day one. First they sit in the
right seat and call out every move they *would* make while the captain
flies. Then they fly with the captain's hands hovering over the controls.
Then they fly routine legs alone, while the captain still handles storms.
Trust is earned in steps, with evidence at each one, and it can be taken
back.

Deploying an agent that takes real actions (refunds, emails, changes to a
company's records) works the same way. This lesson builds each mechanism:
graduated autonomy, prompt versioning, canary releases, kill switches, rate
limits, and a tamper-evident audit log.

```mermaid
flowchart LR
  P[Proposed action] --> K{Kill switch<br/>on?}
  K -->|yes| X[Blocked]
  K -->|no| R{Rate limit<br/>allows it?}
  R -->|no| X
  R -->|yes| A{Autonomy level<br/>and risk allow<br/>acting alone?}
  A -->|no| H[Queue for<br/>human approval]
  A -->|yes| E[Execute]
  H -->|approved| E
  E --> L[Append to<br/>audit log]
```

**Reading it:** every action the agent proposes passes the same gates in
the same order, cheapest and most absolute first. The kill switch stops
everything instantly; the rate limit bounds how much damage even a
misbehaving agent can do per minute; the autonomy level decides whether a
person must approve. Whatever executes is written to the audit log.

**In code:** each gate is one call: `KillSwitch.allowed`, then
`TokenBucket.allow`, then `AutonomyController.may_act_alone`, and finally
`AuditLog.append` for whatever executes.

## Graduated autonomy: shadow mode, approval, then autonomy

**Everyday picture.** The trainee pilot again: call out moves (shadow), fly
with the captain ready to take over (approval), fly routine legs alone
(autonomy for low-risk actions), and hand back the controls after any
incident.

In **shadow mode** the agent runs on real inputs and records what it
*would* do, but takes no action; people keep doing the work. You compare its
decisions with theirs. When agreement is high enough, it **proposes**
actions and a person approves each one. When approvals are nearly always
"yes", it may act alone on low-risk actions, while high-risk ones keep
needing a person. Any incident or drop in quality sends it back a level.

**Worked example.** Ten support tickets in shadow mode:

| Agent proposed | Human did | Agree? |
|---|---|---|
| refund | refund | yes |
| refund | escalate | **no** |
| reply ×5 | reply ×5 | yes ×5 |
| refund | refund | yes |
| refund | escalate | **no** |
| refund | refund | yes |

Overall agreement is 8/10 = 80%. Broken down by what the agent proposed,
replies agree 5/5 = 100% but refunds only 3/5 = 60%. The breakdown tells
you *what* to let the agent do first: replies, not refunds.

$$
\text{agreement} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[a_i = h_i]
$$

**Symbols**

| Symbol | Meaning here | Worked example |
|---|---|---|
| $N$ | number of shadow decisions compared | 10 |
| $a_i$ | what the agent proposed for case $i$ | refund |
| $h_i$ | what the human actually did for case $i$ | escalate |
| $\mathbb{1}[\ldots]$ | 1 if the condition is true, 0 if not (an *indicator*) | 0 for case 2 |
| $\sum_{i=1}^{N}$ | add up over all cases | |

**In words:** agreement is the share of cases where the agent proposed
exactly what the human did.

**On the worked example:** eight of the ten indicators are 1, so agreement
= 8/10 = 0.8.

```mermaid
stateDiagram-v2
  state "Shadow mode: record, don't act" as Shadow
  state "Human approves each action" as Approve
  state "Autonomous for low-risk actions" as Auto
  [*] --> Shadow
  Shadow --> Approve: agreement ≥ 95% over the last 50 decisions
  Approve --> Auto: ≥ 98% of the last 50 approved, no incidents
  Auto --> Approve: incident or quality drift
  Approve --> Shadow: incident
```

**Reading it:** each box is a level of trust; each arrow is labelled with
the evidence needed to move. Promotions need a sustained record over a
window of recent decisions, not a lucky streak; demotions happen at once,
on a single incident. Even at the top level, high-risk actions (large
refunds, deletions) keep needing a person.

![Shadow-mode agreement accumulating until the agent earns promotion](figures/primer.agents.deployment.shadow.svg)

**Reading it:** the x-axis counts shadow decisions; the blue line is
agreement with humans over the last 50 of them. The dashed line is the 95%
bar. Early on the window is too short to judge (grey region); the marker
shows where the rolling agreement first clears the bar and the agent is
promoted to proposing actions for approval.

**In code:** `shadow_agreement` computes agreement overall and per proposed
action. `AutonomyController` is the state diagram: `AutonomyController.record_shadow`
and `AutonomyController.record_approval` promote once a full window clears
the bar, while `AutonomyController.record_incident` and
`AutonomyController.record_quality` demote at once.

## Prompts are code

**Everyday picture.** A restaurant's recipe binder where every recipe
change gets a new revision number, the old version stays in the binder, and
the kitchen can switch back in one step if customers complain.

A change to a prompt can break behaviour as badly as a code change, so treat
it the same way: version it, review it, run the evals before release
(`primer.agents.evals`), and keep the previous version one step away.
`PromptRegistry` is **content-addressed**: a version's name includes a
**hash** of its text. A hash (here SHA-256) is a fixed-length fingerprint:
the same text always gives the same fingerprint, and any change, even one
character, gives a completely different one.

**Worked example.** `"Be concise."` hashes to `ab018bdb…`, so its version is
`support@ab018bdb`. Registering the same text again returns the same
version; `"Be concise and friendly."` gets a new one. Every trace records
which version produced each model call (`primer.agents.observability`), so
a behaviour change can be tied to the exact prompt edit that caused it.

**In code:** `PromptRegistry.register` hashes the text into a version name
and makes it active; `PromptRegistry.rollback` steps back one version;
`PromptRegistry.active_text` returns the prompt currently in use.

## Canary releases: try it on a few users first

**Everyday picture.** Miners once carried a canary into the mine: if the air
turned bad, the canary showed it before the miners were harmed. A **canary
release** sends a small share of traffic to the new version, compares its
metrics with the current version (the *control*), and grows the share only
while the canary stays healthy.

**Worked example.** Steps 1% → 5% → 25% → 100%, allowed drop 2 points,
at least 100 canary tasks before judging:

| Control success | Canary success | Canary tasks | Decision |
|---|---|---|---|
| 95% | 95% | 100 | advance from 1% to 5% |
| 95% | 90% | 100 | 90 < 95 − 2: **roll back to 0%** |
| 95% | 90% | 10 | too few samples: stay at 1% |

$$
\text{roll back if } \hat{p}_{\text{canary}} < \hat{p}_{\text{control}} - \delta
$$

**Symbols**

| Symbol | Meaning here | Worked example |
|---|---|---|
| $\hat{p}_{\text{canary}}$ | measured success rate on canary traffic (the hat means "measured from samples") | 0.90 |
| $\hat{p}_{\text{control}}$ | measured success rate on the current version | 0.95 |
| $\delta$ | the largest drop you'll tolerate (Greek *delta*) | 0.02 |

**In words:** roll back when the new version's success rate falls more than
the allowed margin below the current version's.

**On the worked example:** 0.90 < 0.95 − 0.02 = 0.93, so roll back.

Users are assigned to the canary by hashing their id into one of 100
buckets: the same user always lands in the same bucket, so nobody flips
between versions mid-conversation, and growing from 5% to 25% only adds
users.

```mermaid
sequenceDiagram
  participant D as Deploy system
  participant R as Router
  participant M as Metrics
  D->>R: send 1% of users to v2
  R->>M: success rates, v1 vs v2
  M-->>D: v2 95%, v1 95%: healthy
  D->>R: grow to 5%
  R->>M: success rates, v1 vs v2
  M-->>D: v2 91%, v1 95%: worse by 4 points
  D->>R: roll back: 0% to v2
  D-->>D: alert the owner with traces
```

**Reading it:** the deploy system only ever acts on measurements. Each
growth step waits for enough traffic to judge, and a single bad reading
sends everyone back to the old version automatically. At worst, a few
percent of users saw the bad version for one step.

![Canary rollout of a good and a bad version, with the automatic rollback](figures/primer.agents.deployment.canary.svg)

**Reading it:** each panel is one rollout. The x-axis is the rollout step
(with the share of traffic on the canary); the lines are measured success
rates for control and canary. The good version (left) tracks the control
and reaches 100%. The bad version (right), truly 91% successful, gets lucky
on the 100 tasks of the 1% step and advances, then falls below the tolerance
band at 5% and is rolled back to 0%, marked by the red cross. Small steps
are cheap to get wrong; that's why there are several.

**In code:** `in_canary` hashes a user id into one of 100 buckets.
`CanaryController.observe` accumulates success counts, and
`CanaryController.step` applies the rollback rule above: advance, wait for
more samples, or roll back. `simulate_rollout` drives a whole rollout for the
figure.

A **feature flag** is the same idea as an on/off switch in configuration:
it turns a capability on for chosen tenants or users without redeploying,
and off again just as fast.

## Kill switches and rate limits: bound the blast radius

**Everyday picture.** The red emergency-stop button on a treadmill, and the
turnstile at a stadium that lets people through only so fast however hard
the crowd pushes.

A **kill switch** stops the agent instantly, for one **tenant** (one
customer organisation) or for everyone. A **rate limit** caps actions per
unit time, so even a malfunctioning agent can only send so many emails or
change so many records per minute. The **blast radius** is how much damage
a failure can do before someone notices; these two mechanisms bound it.

**Worked example: the token bucket.** Picture a jar that holds at most 5
tokens and gains 1 token per second. Each action takes a token; with no
token, the action is refused. Starting full: 5 actions in a burst succeed,
the 6th is refused. After 2 seconds the jar has 2 tokens again: 2 more
succeed, the next is refused.

$$
b(t) = \min\bigl(C,\; b(t_0) + r\,(t - t_0)\bigr)
$$

**Symbols**

| Symbol | Meaning here | Worked example |
|---|---|---|
| $b(t)$ | tokens in the bucket at time $t$ | 2 at t = 2 s |
| $t_0$ | when the bucket was last updated | 0 s |
| $b(t_0)$ | tokens left then | 0 after the burst |
| $r$ | refill rate, tokens per second | 1 |
| $C$ | capacity: the largest burst allowed | 5 |
| $\min$ | the smaller of the two: the jar can't overflow | |

**In words:** the tokens now are what was left plus what has dripped in
since, but never more than the jar holds.

**On the worked example:** min(5, 0 + 1 × (2 − 0)) = 2 tokens, so two
more actions are allowed.

![Tokens in the bucket during a burst of requests](figures/primer.agents.deployment.token_bucket.svg)

**Reading it:** the blue line is the number of tokens in the jar over time;
green dots are allowed actions and red crosses refused ones. The opening
burst drains the jar, refusals follow, and then actions trickle through at
the refill rate: bursts are allowed, sustained floods are not.

**In code:** `KillSwitch` holds the global and per-tenant off switches.
`TokenBucket.level` is the formula for $b(t)$, and `TokenBucket.allow`
spends a token or refuses.

## A tamper-evident audit log

**Everyday picture.** A ledger where every page ends with a wax seal
pressed from the previous page's seal plus this page's contents. Change one
number on page 12 and its seal no longer matches, and neither does any seal
after it.

An **audit log** records which agent took which action, for which user,
with what inputs. For regulated work, it must be **tamper-evident**: an
alteration can't go unnoticed. A **hash chain** does this: each entry
stores the hash of the previous entry, and its own hash covers its contents
and that previous hash.

$$
h_i = \operatorname{SHA256}(h_{i-1} \,\Vert\, \text{entry}_i), \qquad h_0 = 00\ldots0
$$

(Entries are numbered from 1 here; `verify()` reports Python list
positions, which start at 0, so "entry 2" is position 1.)

**Symbols**

| Symbol | Meaning here |
|---|---|
| $h_i$ | the fingerprint (hash) stored with entry $i$ |
| $h_{i-1}$ | the previous entry's fingerprint |
| $\operatorname{SHA256}$ | the hash function: any input to a 64-hex-digit fingerprint |
| $\Vert$ | "joined together with" (concatenation) |
| $h_0$ | a fixed starting value for the first entry |

**In words:** each entry's fingerprint is the fingerprint of the previous
fingerprint joined to this entry's contents.

**On the worked example:** three entries: refund A200 for 40, refund A201
for 15, escalate A300. Change entry 2's amount from 15 to 1,500 and
recomputing its fingerprint no longer gives the stored $h_2$, so `verify()`
reports position 1. Delete entry 2 instead, and entry 3 moves up to position
1; its stored previous fingerprint is $h_2$, the deleted entry's, which
doesn't match $h_1$, so the break is again reported at position 1.

```mermaid
flowchart LR
  G["h0 = 000…"] --> E1["entry 1: refund A200, 40<br/>prev = h0<br/>h1 = SHA256(h0 ‖ entry 1)"]
  E1 --> E2["entry 2: refund A201, 15<br/>prev = h1<br/>h2 = SHA256(h1 ‖ entry 2)"]
  E2 --> E3["entry 3: escalate A300<br/>prev = h2<br/>h3 = SHA256(h2 ‖ entry 3)"]
```

**Reading it:** each box carries the fingerprint of the box before it, so
the boxes are chained. Editing any box changes its fingerprint, which breaks
the link to the next box, so verification walks the chain and stops at the
first broken link. In production, also write the log to storage that can't
be modified afterwards (write-once storage) and link each entry to its full
trace.

**In code:** `AuditLog.append` stores the previous entry's hash in the new
entry and seals it with its own; `AuditLog.verify` walks the chain and
returns the position of the first broken link, or None.

## Change management: adoption is part of the job

A technically working agent still fails if people don't trust or use it.
Involve the people whose work it touches from shadow mode onwards (their
corrections are your best eval data), design around their actual workflow,
show sources and confidence so they can check answers, and make handing a
case to a person one click.

## In 20 seconds
- Earn autonomy in steps: shadow mode, then human approval of each action,
  then autonomy for low-risk actions only; demote on any incident or drift.
- Treat prompts as code: versioned (content hashes), reviewed, evaluated,
  and one step from rollback.
- Release to a small canary share first and grow it only while its metrics
  match the current version; roll back automatically otherwise.
- Bound the blast radius: kill switches per tenant and globally, and rate
  limits on actions.
- Keep a tamper-evident audit log (a hash chain) linking every action to its
  user, authority and trace.

## Self-test questions

**How would you roll out an agent that takes real actions in a company's
ERP system (its finance and operations system of record)?**
Start read-only in shadow mode on real transactions, comparing proposals
with what staff actually did, broken down by action type. Give the agent a
dedicated service identity with the narrowest permissions, a sandbox or
dry-run mode for writes, and idempotency keys so retries can't double-post.
Move to proposing actions that a person approves in their existing
workflow, starting with the low-value, reversible action types that scored
best in shadow. Grant autonomy only for those, with value thresholds that
still require approval, per-tenant kill switches, rate limits on writes, a
canary rollout for every prompt or model change, eval gates in CI, and a
tamper-evident audit log tied to traces. Demote automatically on incidents
or drift, and keep the finance team involved throughout.

**Why start in shadow mode instead of with human approval?**
Shadow mode costs the humans nothing extra and produces a clean comparison
against what they actually did, at full volume, before the agent can
influence anything. Approval mode adds review load and can bias reviewers
toward accepting the agent's suggestion.

**What does a canary protect against that offline evals don't?**
Real traffic: inputs, integrations and user behaviour the golden set didn't
anticipate. Evals gate the release; the canary limits the damage from
whatever the evals missed.

**Why a hash chain rather than an ordinary log table?**
Anyone with write access can quietly edit an ordinary row. With a hash
chain, any edit or deletion breaks every later link, so tampering is
detectable, and anchoring the latest hash somewhere separate (or using
write-once storage) makes it provable.

## The papers behind this lesson

- Haber & Stornetta, *How to Time-Stamp a Digital Document* (Journal of
  Cryptology, 1991), https://doi.org/10.1007/BF00196791. Introduced chaining
  each record to the hash of the one before, the idea behind tamper-evident
  logs (and, later, blockchains).

## Further reading
- Google SRE workbook, canarying releases: https://sre.google/workbook/canarying-releases/
- Martin Fowler, feature toggles: https://martinfowler.com/articles/feature-toggles.html
- Token bucket algorithm: https://en.wikipedia.org/wiki/Token_bucket
- NIST AI Risk Management Framework: https://www.nist.gov/itl/ai-risk-management-framework
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from primer._show import banner, say, table, takeaway

# ---------------------------------------------------------------------------
# 1. Shadow mode and graduated autonomy
# ---------------------------------------------------------------------------


def shadow_agreement(agent: list[str], human: list[str]) -> dict[str, Any]:
    """Compare what the agent would have done with what people actually did."""
    by_action: dict[str, list[bool]] = defaultdict(list)
    for a, h in zip(agent, human):
        by_action[a].append(a == h)
    overall = sum(a == h for a, h in zip(agent, human)) / len(agent)
    return {"overall": overall, "by_agent_action": {k: sum(v) / len(v) for k, v in by_action.items()}}


LEVELS = ("shadow", "approve_each", "auto_low_risk")


@dataclass
class AutonomyController:
    """Tracks evidence and moves an agent between autonomy levels.

    Promotions need a full window of recent decisions above a bar;
    demotions happen immediately.
    """

    level: str = "shadow"
    min_decisions: int = 50
    min_agreement: float = 0.95
    min_approval_rate: float = 0.98
    window: deque = field(default_factory=deque)
    incidents: list[str] = field(default_factory=list)

    def _rate(self) -> float | None:
        if len(self.window) < self.min_decisions:
            return None  # not enough evidence yet
        return sum(self.window) / len(self.window)

    def _push(self, ok: bool) -> float | None:
        self.window.append(ok)
        while len(self.window) > self.min_decisions:
            self.window.popleft()
        return self._rate()

    def _move(self, level: str) -> None:
        self.level = level
        self.window.clear()  # evidence at one level doesn't carry over to the next

    def record_shadow(self, agreed: bool) -> None:
        rate = self._push(agreed)
        if self.level == "shadow" and rate is not None and rate >= self.min_agreement:
            self._move("approve_each")

    def record_approval(self, approved: bool) -> None:
        rate = self._push(approved)
        if self.level == "approve_each" and rate is not None and rate >= self.min_approval_rate:
            self._move("auto_low_risk")

    def record_incident(self, description: str) -> None:
        self.incidents.append(description)
        self._move(LEVELS[max(0, LEVELS.index(self.level) - 1)])

    def record_quality(self, success_rate: float, baseline: float, max_drop: float) -> None:
        """Demote on drift: success falling more than `max_drop` below the baseline."""
        if self.level == "auto_low_risk" and success_rate < baseline - max_drop:
            self._move("approve_each")

    def may_act_alone(self, risk: str) -> bool:
        return self.level == "auto_low_risk" and risk == "low"


# ---------------------------------------------------------------------------
# 2. Prompts as code
# ---------------------------------------------------------------------------


class PromptRegistry:
    """Content-addressed prompt versions with one-step rollback."""

    def __init__(self) -> None:
        self.history: dict[str, list[tuple[str, str]]] = defaultdict(list)  # name -> [(version, text)]
        self.active: dict[str, int] = {}

    def register(self, name: str, text: str) -> str:
        version = f"{name}@{hashlib.sha256(text.encode()).hexdigest()[:8]}"
        versions = [v for v, _ in self.history[name]]
        if version in versions:
            self.active[name] = versions.index(version)
        else:
            self.history[name].append((version, text))
            self.active[name] = len(self.history[name]) - 1
        return version

    def rollback(self, name: str) -> str:
        self.active[name] = max(0, self.active[name] - 1)
        return self.history[name][self.active[name]][0]

    def active_text(self, name: str) -> str:
        return self.history[name][self.active[name]][1]


# ---------------------------------------------------------------------------
# 3. Canary releases
# ---------------------------------------------------------------------------


def in_canary(user_id: str, percent: float) -> bool:
    """Deterministic assignment: hash the user into one of 100 buckets."""
    bucket = int(hashlib.sha256(user_id.encode()).hexdigest(), 16) % 100
    return bucket < percent


@dataclass
class CanaryController:
    steps: list[float]
    max_drop: float = 0.02
    min_samples: int = 100
    index: int = 0
    rolled_back: bool = False
    counts: dict[str, int] = field(default_factory=lambda: {"cs": 0, "ct": 0, "ks": 0, "kt": 0})

    @property
    def percent(self) -> float:
        return 0 if self.rolled_back else self.steps[self.index]

    def observe(self, control_successes: int, control_total: int, canary_successes: int, canary_total: int) -> None:
        self.counts["cs"] += control_successes
        self.counts["ct"] += control_total
        self.counts["ks"] += canary_successes
        self.counts["kt"] += canary_total

    def step(self) -> float:
        """Decide: advance, wait, or roll back. Returns the new canary percent."""
        c = self.counts
        if self.rolled_back or c["kt"] < self.min_samples or c["ct"] == 0:
            return self.percent
        if c["ks"] / c["kt"] < c["cs"] / c["ct"] - self.max_drop:
            self.rolled_back = True
        elif self.index < len(self.steps) - 1:
            self.index += 1
            self.counts = {k: 0 for k in c}
        return self.percent


def simulate_rollout(canary_rate: float, control_rate: float = 0.95, users_per_step: int = 10_000, seed: int = 0) -> list[dict[str, Any]]:
    """Walk a rollout step by step with simulated task outcomes."""
    rng = np.random.default_rng(seed)
    ctl = CanaryController(steps=[1, 5, 25, 50, 100], max_drop=0.02, min_samples=100)
    history = []
    for _ in range(50):  # safety cap; a real rollout also has a wall-clock limit
        pct = ctl.percent
        n_canary = max(1, int(users_per_step * pct / 100))
        n_control = users_per_step - n_canary if pct < 100 else users_per_step
        ks, cs = int(rng.binomial(n_canary, canary_rate)), int(rng.binomial(n_control, control_rate))
        ctl.observe(cs, n_control, ks, n_canary)
        new = ctl.step()
        history.append({"percent": pct, "canary_rate": ks / n_canary, "control_rate": cs / n_control, "next": new, "rolled_back": ctl.rolled_back})
        if ctl.rolled_back or (pct == 100):
            break
    return history


# ---------------------------------------------------------------------------
# 4. Kill switch and rate limiting
# ---------------------------------------------------------------------------


class KillSwitch:
    def __init__(self) -> None:
        self.global_off = False
        self.tenants_off: set[str] = set()

    def disable_all(self) -> None:
        self.global_off = True

    def disable_tenant(self, tenant: str) -> None:
        self.tenants_off.add(tenant)

    def enable_tenant(self, tenant: str) -> None:
        self.tenants_off.discard(tenant)

    def allowed(self, tenant: str) -> bool:
        return not self.global_off and tenant not in self.tenants_off


class TokenBucket:
    """Allow bursts up to `capacity`, then `rate_per_second` actions per second."""

    def __init__(self, rate_per_second: float, capacity: float, clock: Callable[[], float] = time.monotonic):
        self.rate, self.capacity, self.clock = rate_per_second, capacity, clock
        self.tokens = float(capacity)
        self.last = clock()

    def level(self) -> float:
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + self.rate * (now - self.last))
        self.last = now
        return self.tokens

    def allow(self) -> bool:
        if self.level() >= 1:
            self.tokens -= 1
            return True
        return False


# ---------------------------------------------------------------------------
# 5. Tamper-evident audit log
# ---------------------------------------------------------------------------

GENESIS = "0" * 64


def _entry_hash(entry: dict[str, Any]) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    # sort_keys makes the serialization canonical: same content, same bytes, same hash.
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


class AuditLog:
    def __init__(self, clock: Callable[[], float] = time.time):
        self.entries: list[dict[str, Any]] = []
        self.clock = clock

    def append(self, actor: str, action: str, on_behalf_of: str, inputs: dict[str, Any], trace_id: str | None = None) -> dict[str, Any]:
        prev = self.entries[-1]["hash"] if self.entries else GENESIS
        entry = {"at": self.clock(), "actor": actor, "action": action, "on_behalf_of": on_behalf_of,
                 "inputs": inputs, "trace_id": trace_id, "prev_hash": prev}
        entry["hash"] = _entry_hash(entry)
        self.entries.append(entry)
        return entry

    def verify(self) -> int | None:
        """Index of the first entry whose link is broken, or None if intact."""
        prev = GENESIS
        for i, e in enumerate(self.entries):
            if e["prev_hash"] != prev or _entry_hash(e) != e["hash"]:
                return i
            prev = e["hash"]
        return None


# ---------------------------------------------------------------------------
# Figures and demo
# ---------------------------------------------------------------------------


def shadow_run(n: int = 200, accuracy: float = 0.95, seed: int = 4) -> dict[str, Any]:
    """Simulated shadow decisions; returns rolling agreement and the promotion point."""
    rng = np.random.default_rng(seed)
    agreed = rng.random(n) < accuracy
    c = AutonomyController(min_decisions=50, min_agreement=0.95)
    rolling, promoted_at = [], None
    for i, ok in enumerate(agreed):
        window = agreed[max(0, i - 49) : i + 1]
        rolling.append(window.mean())
        if promoted_at is None:
            c.record_shadow(bool(ok))
            if c.level == "approve_each":
                promoted_at = i + 1
    return {"rolling": rolling, "promoted_at": promoted_at}


def bucket_trace() -> dict[str, list]:
    """A burst of 8 requests at t=0, then one request every 0.5 s."""
    now = [0.0]
    b = TokenBucket(1.0, 5, clock=lambda: now[0])
    times = [0.0] * 8 + [0.5 * k for k in range(1, 13)]
    out = {"t": [], "allowed": [], "level_t": [], "level": []}
    for t in times:
        now[0] = t
        out["level_t"].append(t)
        out["level"].append(b.level())
        out["t"].append(t)
        out["allowed"].append(b.allow())
        out["level_t"].append(t)
        out["level"].append(b.tokens)
    return out


def figures() -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs: dict[str, Any] = {}

    s = shadow_run()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.axvspan(0, 50, color="#dddddd", label="window not yet full")
    ax.plot(range(1, len(s["rolling"]) + 1), s["rolling"], color="#4c72b0", label="agreement, last 50 decisions")
    ax.axhline(0.95, ls="--", color="k", lw=1, label="promotion bar (95%)")
    if s["promoted_at"]:
        ax.plot(s["promoted_at"], s["rolling"][s["promoted_at"] - 1], "o", color="#55a868", ms=9, label="promoted to approval mode")
    ax.set_ylim(0.85, 1.01)
    ax.set_xlabel("shadow decisions so far")
    ax.set_ylabel("agreement with humans")
    ax.set_title("Shadow mode: earning the first promotion")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    figs["shadow"] = fig

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, (title, rate, seed) in zip(axes, [("good version (95%)", 0.95, 0), ("bad version (91%)", 0.91, 3)]):
        h = simulate_rollout(rate, seed=seed)
        xs = range(len(h))
        ax.plot(xs, [r["control_rate"] for r in h], "o-", color="#8c8c8c", label="control")
        ax.plot(xs, [r["canary_rate"] for r in h], "o-", color="#4c72b0", label="canary")
        ax.fill_between(xs, [r["control_rate"] - 0.02 for r in h], [r["control_rate"] for r in h], color="#8c8c8c", alpha=0.2, label="tolerance (2 pts)")
        for i, r in enumerate(h):
            if r["rolled_back"]:
                ax.plot(i, r["canary_rate"], "x", color="#c44e52", ms=14, mew=3, label="rolled back to 0%")
        ax.set_xticks(list(xs), [f"{r['percent']}%" for r in h])
        ax.set_xlabel("share of traffic on the canary")
        ax.set_title(title)
    axes[0].set_ylabel("task success rate")
    axes[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    figs["canary"] = fig

    bt = bucket_trace()
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.step(bt["level_t"], bt["level"], where="post", color="#4c72b0", label="tokens in the bucket")
    ok = [(t, 5.3) for t, a in zip(bt["t"], bt["allowed"]) if a]
    no = [(t, 5.6) for t, a in zip(bt["t"], bt["allowed"]) if not a]
    ax.scatter([t for t, _ in ok], [y for _, y in ok], color="#55a868", marker="o", label="allowed")
    ax.scatter([t for t, _ in no], [y for _, y in no], color="#c44e52", marker="x", label="refused")
    ax.set_xlabel("seconds")
    ax.set_ylabel("tokens")
    ax.set_title("Token bucket: capacity 5, refill 1 per second")
    ax.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    figs["token_bucket"] = fig
    return figs


def demo() -> None:
    banner("1. Shadow mode: compare, don't act")
    agent = ["refund", "refund", "reply", "reply", "reply", "refund", "reply", "refund", "reply", "refund"]
    human = ["refund", "escalate", "reply", "reply", "reply", "refund", "reply", "escalate", "reply", "refund"]
    print(shadow_agreement(agent, human))
    print()
    say("Replies agree every time; refunds only 3 in 5. Let the agent draft replies first.")

    banner("2. Graduated autonomy")
    c = AutonomyController()
    s = shadow_run()
    print(f"simulated shadow run promoted after {s['promoted_at']} decisions")
    c.level = "auto_low_risk"
    print(f"autonomous: may act alone on low risk? {c.may_act_alone('low')}; on high risk? {c.may_act_alone('high')}")
    c.record_incident("refunded the wrong customer")
    print(f"after an incident: level = {c.level}")
    print()

    banner("3. Prompts as code")
    reg = PromptRegistry()
    v1 = reg.register("support", "Be concise.")
    v2 = reg.register("support", "Always be maximally helpful.")
    print(f"registered {v1}, then {v2}; rolled back to {reg.rollback('support')}")
    print()

    banner("4. Canary rollouts")
    for label, rate, seed in [("good", 0.95, 0), ("bad", 0.91, 3)]:
        h = simulate_rollout(rate, seed=seed)
        table(["canary %", "control", "canary", "next %"], [(r["percent"], r["control_rate"], r["canary_rate"], r["next"]) for r in h], floatfmt=".3f")
        print(f"{label} version: {'rolled back' if h[-1]['rolled_back'] else 'fully rolled out'}")
        print()

    banner("5. Kill switch and rate limit")
    ks = KillSwitch()
    ks.disable_tenant("acme")
    print(f"acme allowed? {ks.allowed('acme')}; globex allowed? {ks.allowed('globex')}")
    bt = bucket_trace()
    print("burst of 8 at t=0:", ["ok" if a else "refused" for a in bt["allowed"][:8]])
    print()

    banner("6. Tamper-evident audit log")
    log = AuditLog(clock=lambda: 0.0)
    log.append("refund-agent", "refund", "dana", {"order": "A200", "amount": 40})
    log.append("refund-agent", "refund", "lee", {"order": "A201", "amount": 15})
    log.append("refund-agent", "escalate", "sam", {"order": "A300"})
    print("intact log verifies:", log.verify() is None)
    log.entries[1]["inputs"]["amount"] = 1500
    print("after editing entry 1, first broken link at:", log.verify())
    print()
    takeaway("Earn autonomy with evidence, release to a canary first, bound the blast radius, and make every action auditable.")


if __name__ == "__main__":
    demo()
