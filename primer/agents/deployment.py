r"""
# Safe deployment: letting an agent act in the real world, one earned step at a time

Run: `python -m primer.agents.deployment`

This lesson builds on action guardrails from `primer.agents.guardrails`, on
the release gate from `primer.agents.evals` and on traces from
`primer.agents.observability`.

## Level 1: The practitioner's guide

**In one sentence.** Safe deployment means letting an agent act in the real
world one earned step at a time: it proves itself in shadow, then proposes,
then acts alone on low-risk work, while every release goes to a few users
first, every action can be stopped and rate-limited, and every action is
written to a log nobody can quietly edit.

**When you need it.** The moment an agent's output stops being a draft and
becomes an action: a refund issued, an email sent, a record changed in the
system a company runs on. Offline evals (`primer.agents.evals`) gate what
ships; this lesson is about limiting the damage from whatever the evals
missed, because real traffic always holds inputs the golden set didn't.
The tell: someone asks "what's the worst this could do in the next five
minutes?" and the answer is "we'd notice eventually". This lesson's worked
example shows why the first step is to watch rather than act: in shadow
mode over ten support tickets the agent agrees with staff 80% of the time
overall, but the split is 100% on replies and 60% on refunds, so the right
first grant of autonomy is replies, not refunds, and only the breakdown
shows it. You don't need canaries and hash chains for an internal tool
that drafts text a person edits; you need all of it for anything that
moves money or changes records, and regulated work needs the audit log
whatever the risk.

**Your options.** Seven mechanisms, from the ones that cost nothing to run
to the ones that hold up under audit. They stack rather than compete:

| Option | What it does | What it guarantees | What it costs | Where it lives |
|---|---|---|---|---|
| Shadow mode | The agent records what it would do on real inputs; people keep doing the work | Zero user impact and a clean comparison against what staff actually did, at full volume | Staff carry the whole load meanwhile; you need to log proposals and compare | Your agent loop, plus a comparison job |
| Human approval of each action | The agent proposes; a person approves before anything executes | Nothing executes unseen | Review load, and reviewers who start rubber-stamping | The reviewers' existing workflow |
| Graduated autonomy | Acts alone on low-risk actions once a full window of recent decisions clears a bar; high-risk actions still go to a person; any incident demotes at once | Trust is earned on evidence and taken back on one bad event | Classifying every action's risk; keeping the window's statistics | Your action gate |
| Prompts as code | Every prompt version is named by a hash of its text, reviewed, evaluated, and one step from rollback | Any behaviour change ties to the exact edit that caused it | A registry and the discipline to use it | Your code and configuration |
| Canary release | A new version goes to 1% of users, then 5, 25, 50 and 100, growing only while its success rate stays within a margin of the current version's | At worst a few percent of users saw the bad version for one step | Enough traffic per step to judge (100 canary tasks in this lesson), and metrics split by version | The router and the metrics store |
| Kill switch and rate limit | Stop the agent instantly, per tenant or for everyone; cap actions per minute with a token bucket | A bounded blast radius even when everything else has failed | Someone on call to flip the switch; a burst above the cap is refused | In front of every action |
| Tamper-evident audit log | Each entry stores the hash of the one before, so any edit or deletion breaks every later link | Tampering is detectable; with write-once storage, provable | Storage that can't be modified, and a link from each entry to its trace | Storage |

**How to choose.** Go by what the action can do if it's wrong, and how
fast you'd know.

- Reversible and low value (a reply, a draft, a label): shadow briefly,
  then approval, then autonomy, with the eval gate and a canary on every
  prompt or model change.
- Irreversible or high value (a payment, a deletion, an external email):
  approval stays, whatever the agent's record, above a value threshold you
  set on purpose. Autonomy applies below it.
- Many customers on one platform: kill switches and rate limits per
  tenant, so one customer's bad day never becomes everyone's.
- Regulated work: the audit log from day one, on write-once storage,
  linked to traces, with the agent acting under its own service identity
  and the narrowest permissions it can do the job with.
- Whatever you pick, promote on a sustained record over a window of
  decisions (this lesson uses the last 50), never a lucky streak, and demote
  on a single incident.

**What it costs.** Shadow mode costs the humans nothing extra and costs you
patience: in the lesson's simulation the rolling agreement first clears the
95% bar after 98 decisions, and a window of 50 is the least you can judge
on. Approval mode costs reviewer time on every action. A canary costs
traffic and time per step; the lesson's controller refuses to judge on
fewer than 100 canary tasks, and the SRE Workbook's chapter on canarying
makes the trade explicit: a 5% canary with a 20% error rate hurts 1% of
requests overall, which is the whole point, but a daily release cycle
can't afford a week-long canary. A rate limit costs refused actions during
a burst (a bucket of 5 refilling at one per second lets 5 of a burst of 8
through and refuses 3). The audit log costs a hash per entry and storage
you can't reclaim. Rollback of a prompt costs nothing, because the previous
version is already in the registry. Against all of that, the cost of not
having them is the incident: a refund issued on a bad prompt that no
switch could stop and no log can show.

**What breaks.**

- **Promoting on a streak.** Twenty good decisions in a row is luck at 95%.
  Require a full window.
- **A canary judged too early.** The lesson's bad version, truly 91%
  successful, passes the 1% step by luck on 100 tasks and is caught at 5%.
  Small steps are cheap to get wrong; that's why there are several.
- **Users flipping between versions.** Assign users by hashing their id
  into a bucket, so nobody changes version mid-conversation.
- **Reviewers who stop reading.** Approval mode can bias people toward
  accepting the agent's suggestion. Sample approvals for a second look, and
  keep incident demotion automatic.
- **Retried writes that double-post.** A canary or a retry that re-sends
  "create order" makes two orders. Use idempotency keys before granting any
  write.
- **An ordinary log table.** Anyone with write access can edit a row. Chain
  the hashes and anchor the latest one somewhere separate.
- **A working agent nobody uses.** Adoption is part of the job: build with
  the people whose work it touches, show sources, and make handing a case
  to a person one click.

**In the wild.** Canarying comes from site reliability engineering: the
Google SRE Workbook defines it as "a partial and time-limited deployment of
a change in a service and its evaluation", and its advice to rank metrics by
how well they show user-visible problems is why this lesson compares
success rates rather than latency alone. Feature flags are the mechanism
behind both canaries and kill switches; Martin Fowler's taxonomy separates
release, experiment, ops and permissioning toggles, with the kill switch as
a long-lived ops toggle, and the pattern is what feature-flag services and
OpenFeature, a vendor-neutral specification for feature flagging, exist to
implement. The token bucket is a textbook rate-limiting algorithm, linked
in Further reading. The hash chain is Haber and Stornetta's 1991
construction for time-stamping documents, the ancestor of every
tamper-evident log and of blockchains. The NIST AI Risk Management
Framework (released in January 2023) organises this whole lesson's
concerns into four functions, Govern, Map, Measure and Manage, and is the
vocabulary compliance teams will use for it.

**Go deeper.** Level 2 builds each gate in plain code: agreement counted
decision by decision, the autonomy state machine with its promotion window,
a content-addressed prompt registry, a canary controller with the rollback
rule worked in numbers, the token bucket formula, and a hash chain you can
break and watch `verify()` find the break. If you only needed to plan a
rollout, you are done.

## Level 2: How it works, from scratch

An agent that takes real actions (refunds, emails, changes to a company's
records) does real damage when it gets one wrong. So before anything
happens, every action it proposes passes the same gates, in the same order.

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

**Reading it:** follow a proposed action from the left. The first two gates,
the kill switch and the rate limit, are the cheapest to check and can only
block. The third decides whether the agent may act alone or must wait for a
person to approve. Whatever runs, alone or approved, is written to the audit
log on the right.

**In code:** the gates are `KillSwitch.allowed`, `TokenBucket.allow` and
`AutonomyController.may_act_alone`, and `AuditLog.append` records what runs.

## Graduated autonomy: from shadow mode to acting alone

The third gate is the one that takes judgement, and the agent earns its
setting. A new pilot does not start in the captain's seat. First they sit
beside the captain and call out every move they would make. Then they fly
with the captain's hands near the controls. Then they fly routine legs
alone while the captain still takes the storms, and after any incident the
captain takes over again. Letting an agent onto real work in steps like
these is called **graduated autonomy**.

In **shadow mode** the agent runs on real inputs and records what it
*would* do, but takes no action. People keep doing the work, and you compare
its decisions with theirs. When it agrees with them often enough, it
**proposes** actions and a person approves each one. When nearly every
proposal is approved, it may act alone on low-risk actions, while high-risk
ones, such as large refunds or deletions, still go to a person. Any
incident or drop in quality sends it back a level.

The shadow comparison says most when you break it down by action. Here are
ten support tickets:

| Agent proposed | Human did | Agree? |
|---|---|---|
| refund | refund | yes |
| refund | escalate | **no** |
| reply ×5 | reply ×5 | yes ×5 |
| refund | refund | yes |
| refund | escalate | **no** |
| refund | refund | yes |

Overall agreement is 8/10 = 80%. Broken down by what the agent proposed,
replies agree 5/5 = 100% but refunds only 3/5 = 60%. So the agent should be
trusted with replies first, and refunds later.

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

**In words:** agreement is the share of cases where the agent proposed what
the human did.

**On the worked example:** eight of the ten indicators are 1, so agreement
= 8/10 = 0.8.

**In Python:**

```python
a = ["refund", "refund"] + ["reply"] * 5 + ["refund", "refund", "refund"]
h = ["refund", "escalate"] + ["reply"] * 5 + ["refund", "escalate", "refund"]
# 𝟙[a_i = h_i]
indicators = [1 if a_i == h_i else 0 for a_i, h_i in zip(a, h)]
indicators  # → [1, 0, 1, 1, 1, 1, 1, 1, 0, 1]
# (1/N) Σ over i
sum(indicators) / len(indicators)  # → 0.8
```

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

**Reading it:** start at shadow mode and follow the arrows. Each promotion
names the record it needs over the last 50 decisions, so a short lucky
streak is not enough. Each arrow back names a single event: one incident
demotes the agent at once.

![Rolling agreement with humans must first clear the 95% bar over a full 50-decision window, and only then is the agent promoted](figures/primer.agents.deployment.shadow.svg)

**Reading it:** the x-axis counts shadow decisions, and the blue line is
agreement with humans over the last 50 of them. In the grey region fewer
than 50 decisions have been made, too few to judge. The green dot is where
agreement first clears the dashed 95% bar and the agent moves up to
proposing actions for approval.

**In code:** `shadow_agreement` computes agreement overall and per action,
and `AutonomyController` is the state diagram.

## Versioned prompts

Once an agent has earned its autonomy, it keeps changing: someone edits its
prompt, or the language model under it is replaced. A change to a prompt can break
behaviour as badly as a change to code, so it gets the same treatment. It is
versioned, reviewed and run through the evals before release
(`primer.agents.evals`), and the previous version stays one step away.

The version's name can come from the prompt itself. A **hash** (here
**SHA-256**) is a fixed-length fingerprint of a text: the same text always
gives the same fingerprint, and any change, even one character, gives a
completely different one. A prompt named by a hash of its own text is
**content-addressed**. `"Be concise."` hashes to `ab018bdb…`, so its version
is `support@ab018bdb`. Registering the same text again returns the same
version, and `"Be concise and friendly."` gets a new one. Every trace
records which version produced each model call (`primer.agents.observability`),
so a change in behaviour can be tied to the prompt edit that caused it.

**In code:** `PromptRegistry.register` turns a prompt into a version and
makes it active, and `PromptRegistry.rollback` steps back one version.

## Canary releases

A new version that passes the evals has passed a fixed set of cases, and
real users bring inputs that set never held. A **canary release** sends a small
share of traffic to the new version, compares its success rate with the
current version's (the *control*), and grows the share only while the new
version keeps up. The name comes from the canaries miners once carried
underground: when the air turned bad, the bird showed it before the miners
were harmed.

Take a rollout that grows from 1% of traffic to 5%, 25%, 50% and 100%. The
canary may fall at most 2 points below the control, and it is not judged
until it has handled 100 tasks:

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

**In Python:**

```python
p_canary, p_control, delta = 0.90, 0.95, 0.02
round(p_control - delta, 2)  # → 0.93
# roll back?
p_canary < p_control - delta  # → True
```

Users are assigned to the canary by hashing their id into one of 100
buckets. A user always lands in the same bucket, so no one switches versions
in the middle of a conversation, and growing from 5% to 25% only adds users.
A **feature flag** works the same way for a single capability: a switch in
configuration that turns it on for chosen customers or users without
redeploying, and off again as fast.

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

**Reading it:** read from the top. The deploy system grows the canary only
after the metrics come back healthy. At 5% they show the new version 4
points worse, more than the 2-point margin, so the deploy system sends every
user back to the old version on its own and alerts the owner with the
traces.

![The good version tracks the control through 1, 5, 25, 50 and 100%; the bad one passes the 1% step by luck and is rolled back at 5%](figures/primer.agents.deployment.canary.svg)

**Reading it:** each panel is one rollout, with the share of traffic on the
canary along the bottom. The grey line is the control, the blue line the
canary, and the grey band the 2-point margin. On the left, a good version
keeps up with the control all the way to 100%. On the right, a version that
truly succeeds 91% of the time gets lucky on the 100 tasks of the 1% step
and advances, then falls below the band at 5% and is rolled back to 0% (the
red cross). It reached one user in twenty, for one step.

**In code:** `in_canary` assigns users to buckets, `CanaryController.step`
applies the rollback rule, and `simulate_rollout` runs the two rollouts in
the chart.

## Kill switches and rate limits

A canary judges a new version by its average over many tasks. Once a
version is live, something else has to stop it when it goes wrong all at
once, say by sending the same email again and again. A **kill switch**
stops the agent at once, for one **tenant** (one customer organisation) or
for everyone. A **rate limit** caps actions per unit of time, so even a
malfunctioning agent can send only so many emails, or change only so many
records, per minute. Together they bound the **blast radius**: how much
damage a failure can do before someone notices.

One way to build a rate limit is a **token bucket**. Picture a jar that
holds at most 5 tokens and gains 1 token a second. Each action takes a
token, and an action that finds the jar empty is refused. Starting full, a
burst of 5 actions succeeds and the 6th is refused. After 2 seconds the jar
holds 2 tokens again, so 2 more succeed and the next is refused.

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

**In Python:**

```python
# capacity, tokens per second
C, r = 5, 1
def b(t, t_0, b_t0):
    # what was left plus the drip, capped at C
    return min(C, b_t0 + r * (t - t_0))
# 2 s after the burst emptied the jar
b(2, 0, 0)  # → 2
# a long wait refills only to capacity
b(60, 0, 0)  # → 5
```

![A burst of 8 requests gets 5 through before the bucket empties, then requests pass one per second at the refill rate](figures/primer.agents.deployment.token_bucket.svg)

**Reading it:** the blue line is the number of tokens in the jar over time.
Along the top, green dots are allowed actions and red crosses refused ones.
The opening burst empties the jar and the refusals start; after that,
actions get through one a second, at the refill rate. The jar lets a short
burst through but not a sustained flood.

**In code:** `KillSwitch` holds the global and per-tenant switches, and
`TokenBucket.allow` spends a token or refuses.

## A tamper-evident audit log

Once the kill switch has been pulled, someone has to find out what the agent
did. An **audit log** records which agent took which action, for which
user, with what inputs. For regulated work it must be **tamper-evident**,
so that no alteration goes unnoticed. A **hash chain** does this with the
fingerprints that named the prompt versions: each entry stores the hash of
the entry before it, and its own hash covers its contents together with
that previous hash.

$$
h_i = \operatorname{SHA256}(h_{i-1} \,\Vert\, \text{entry}_i), \qquad h_0 = 00\ldots0
$$

(Entries are numbered from 1 here, but verification reports positions
counted from 0, as Python lists are, so entry 2 is position 1.)

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

**On the worked example:** take three entries: refund A200 for 40, refund
A201 for 15, escalate A300. Change entry 2's amount from 15 to 1,500, and
recomputing its fingerprint no longer gives the stored $h_2$, so
verification reports position 1. Delete entry 2 instead, and entry 3 moves
up to position 1. The previous fingerprint it stores is $h_2$, the deleted
entry's, which doesn't match $h_1$, so the break is again reported at
position 1.

**In Python:**

```python
import hashlib
def seal(prev, entry):
    # SHA256(h_{i-1} ‖ entry_i)
    return hashlib.sha256((prev + entry).encode()).hexdigest()
# h_0
log, prev = [], "0" * 64
for entry in ["refund A200 40", "refund A201 15", "escalate A300"]:
    prev = seal(prev, entry)
    # each entry is stored with its h_i
    log.append((entry, prev))
def verify(log):
    prev = "0" * 64
    for position, (entry, h_i) in enumerate(log):
        if seal(prev, entry) != h_i:
            # the first broken link
            return position
        prev = h_i
verify(log) is None  # → True
# entry 2's amount changed
verify([log[0], ("refund A201 1500", log[1][1]), log[2]])  # → 1
# entry 2 deleted
verify([log[0], log[2]])  # → 1
```

```mermaid
flowchart LR
  G["h0 = 000…"] --> E1["entry 1: refund A200, 40<br/>prev = h0<br/>h1 = SHA256(h0 ‖ entry 1)"]
  E1 --> E2["entry 2: refund A201, 15<br/>prev = h1<br/>h2 = SHA256(h1 ‖ entry 2)"]
  E2 --> E3["entry 3: escalate A300<br/>prev = h2<br/>h3 = SHA256(h2 ‖ entry 3)"]
```

**Reading it:** each box carries the fingerprint of the box before it. Edit
any box and its own fingerprint changes, so it no longer matches what the
next box carries. Verification walks the chain from the left and stops at
the first link that doesn't match.

In production, the log also goes to storage that can't be modified
afterwards (write-once storage), and each entry links to its full trace.

**In code:** `AuditLog.append` seals each new entry with the previous
entry's hash, and `AuditLog.verify` returns the position of the first
broken link, or None.

## Change management

Every control so far keeps a working agent safe. The agent still fails if
the people whose work it touches don't trust it or don't use it. Involve
them from shadow mode on, since their corrections are your best eval data.
Design around the way they already work, show sources and confidence so
they can check an answer, and make handing a case to a person one click.

## In 20 seconds
- Earn autonomy in steps: shadow mode, then a person approves each action,
  then autonomy for low-risk actions only; demote on any incident or drift.
- Treat prompts as code: name each version by a hash of its text, review
  and evaluate it before release, and keep the previous version one step
  away.
- Send a new version to a small canary share of users first, grow the share
  only while its success rate keeps up with the current version's, and roll
  back automatically when it doesn't.
- Bound the blast radius with kill switches, per tenant and global, and rate
  limits on actions.
- Keep a tamper-evident audit log (a hash chain) linking every action to its
  user and its trace.

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
    ax.set_ylim(0.6, 1.01)  # low enough for the early, short-window dips (down to 2 of 3)
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
