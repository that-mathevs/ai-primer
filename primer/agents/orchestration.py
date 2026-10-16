r"""
# Orchestration: how much autonomy to give the model

Run: `python -m primer.agents.orchestration`

New to the notation? `primer.notation` explains every symbol used here from
zero. This lesson builds on the message exchange and tool calling in
`primer.agents.llm`.

## Level 1: The practitioner's guide

**In one sentence.** Orchestration is the decision of how much of a
system's control flow the model gets to decide, from none (code runs fixed
steps and asks the model to fill each one) to all of it (a team of models
steering each other), and the named patterns in between.

**When you need it.** You face this decision the moment a task needs more
than one model call: a document pipeline, a support desk, a research
assistant, anything that must look something up and then act. The tell that
you have chosen wrongly in one direction is a fixed pipeline that keeps
growing special cases for requests it didn't anticipate; the tell in the
other is an agent that spends a minute and a thousand tokens on a request a
two-step script answered correctly every time. Anthropic's *Building
effective agents*, the post these pattern names come from, draws the line
this way: workflows are model calls "orchestrated through predefined code
paths"; agents are systems where the model "dynamically directs its own
processes and tool usage". It also gives the rule: find the simplest
solution possible, and add complexity only when needed. You don't need
orchestration at all when one call with retrieval and a few examples in the
prompt answers the question; that post says many applications stop there.

This lesson measures the cost of each step up on one question ("How many PTO
days do I get, and what is the meal per-diem?"). A fixed workflow answers it
in 2 calls and 93 tokens; a router in 2 calls and 105 tokens; an agent loop
in 2 calls and about 990 tokens, ten times the workflow, because it re-sends
its tool definitions and results on every call; a supervisor with two
specialists in 4 calls and about 600 tokens. Same answer, four prices.

**Your options.** Seven designs, from the least autonomy to the most:

| Option | What it does | What it guarantees | What it costs | Where it lives |
|---|---|---|---|---|
| Fixed workflow (prompt chaining with gates) | Code runs steps in order; the model fills each one; a code check between steps stops a bad result | The same steps every time; a failed gate stops before the next call is paid for or misled | 2 calls and 93 tokens on the lesson's question; a gate per hand-off to write | Your code |
| Router | One cheap classifier call picks a handler; unknown labels go to a fallback | Exactly one path per request, and never a crash on a label the model invented | One extra call (105 tokens here) and a fallback handler | Your code, with one model decision |
| Parallelization (sectioning, voting) | Independent pieces run at once, or several judges answer the same prompt and the majority wins | Wall-clock time of the slowest branch; three 80%-accurate judges make an 89.6% panel if they err independently | n times the tokens for n branches | Your code (a thread pool) |
| Evaluator-optimizer | One call drafts, another critiques, until it passes or the round limit hits | A checked result, or a flagged best effort at the limit | Up to two calls per round (2 rounds in the lesson's example) | Your code |
| Agent loop | The model picks tools and decides when it is done | Handles requests you did not anticipate | About 990 tokens for the lesson's question, ten times the workflow; less predictability | The model decides; your loop enforces the budget (`primer.agents.agent_loop`) |
| Orchestrator-workers, multi-agent | A lead model splits the work, specialists each get only their brief, a synthesizer combines | Focused context per specialist and parallel work; subtasks decided at run time | 4 calls and about 600 tokens here; in Anthropic's production research system, about 15 times the tokens of a chat | Several model roles, coordinated by your code |
| Explicit state machine with checkpoints | Code owns every state and transition; the model is consulted at a few named nodes; the run is saved after each step | Auditable, testable transitions and a crash that costs one step, not the run | A state model to design and a store for checkpoints | Your code, or a durable-execution engine |

**How to choose.** Start from whether the *steps* vary between requests, or
only what happens inside each step.

- Steps known in advance, variation inside them (invoice intake: classify,
  extract, validate, approve, post): a fixed workflow or state machine, with
  the model at the judgement nodes only.
- A few kinds of request, each with its own handling: a router in front of
  the workflows, with a fallback that is safe rather than clever.
- One hard judgement call: voting, but vary the prompt, model or evidence,
  because copies of the same model tend to make the same mistake.
- Quality that a checker can state as criteria (a citation present, a test
  passing): evaluator-optimizer, with a round limit.
- The path depends on what is discovered along the way: an agent loop.
- Work that is separable and too big for one context (independent research
  threads, per-document processing): orchestrator-workers. Not for tasks
  where every part needs the same context or depends on the others; the
  same Anthropic report found most coding tasks fall on that side.
- Whatever you pick, code owns the transitions you need to audit, and the
  model decides only where judgement is needed.

**What it costs.** Tokens, first: each rung up the spectrum multiplies
them, from 93 to 105 to about 990 to about 600 on one toy question, and
Anthropic reports agents using about 4 times the tokens of a chat and
multi-agent systems about 15 times. Latency: a chain is the sum of its
calls, a parallel fan-out is its slowest branch, and every hand-off in a
multi-agent design adds a call. Quality: multi-agent can win (that report
measured a 90.2% improvement over a single agent on an internal research
evaluation) when the work parallelizes, and loses context at every hand-off
when it does not. Effort: every pattern in this lesson is under 100 lines
of plain Python; a framework adds tracing, persistence and approval hooks,
and hidden assumptions. Crashes: without checkpoints, a rerun after a
crash repeats every model call (4 instead of 2 in the lesson's invoice
run) and risks a different answer the second time.

**What breaks.**

- **A router that trusts the label.** The classifier returns free text
  (`"Billing-ish, maybe refunds?"`). Normalize it, accept only known
  labels, send the rest to a fallback, and log the misses.
- **A gate that isn't there.** Without a check between chain steps, a
  one-line outline becomes a paid-for, confident draft of nothing.
- **Correlated voters.** Five copies of one model with one prompt agree on
  the same error; the majority formula assumes independence.
- **An evaluator that is never satisfied.** No round limit, no exit. Put
  the limit in, and flag what came out at the limit.
- **Lost hand-offs.** A specialist knows only what the supervisor wrote in
  its brief. Whatever the supervisor forgot is gone. Test that each
  specialist sees only its own task, and that a delegation to a specialist
  that doesn't exist is reported, not crashed on.
- **No checkpoint.** A crash on the last step restarts from the first,
  doubling cost and, combined with a non-idempotent side effect, posting an
  invoice twice.

**In the wild.** The pattern names come from Anthropic's *Building
effective agents*, which also advises starting with the model API directly
before reaching for a framework. LangGraph models a workflow as a graph
with explicit state and checkpoints, which is the state-machine row above.
The Claude Agent SDK and the OpenAI Agents SDK ship vendor-built agent
loops, tools and hand-offs. CrewAI and AutoGen are built around multi-agent
teams. Temporal provides durable execution for any code: each completed
step's result is stored and a crashed run resumes from there. Anthropic's
own research feature is an orchestrator-workers system with a lead model
and parallel subagents, and its engineering report is the source of the
token multipliers above.

**Go deeper.** Level 2 builds every pattern in a few dozen lines each and
measures it: the four-rung cost table, a chain with a gate, a router with a
fallback, voting with the formula for why independent judges help, a
supervisor whose specialists see only their brief, and an invoice state
machine that survives a crash. If you only needed to choose, you are done.

## Level 2: How it works, from scratch

The examples come from one setting: an assistant that answers staff
questions from company policy documents, such as how many days of paid time
off (PTO) they get, or the meal allowance per day of travel (the
per-diem).

## 1. The autonomy spectrum

The simplest design is a **fixed workflow**: ordinary code that calls the
model at set points, here once to rewrite the question as a search and once
to answer from what the search found. A **router** gives the model one
decision: which of several fixed paths handles the request. An **agent
loop** hands over the rest: the model chooses which tools to call, reads
their results and decides when it is done. A **multi-agent** system puts
several agents on one job, one of them splitting the work among the others.

```mermaid
flowchart LR
  W[Fixed workflow<br/>code decides steps] --> R[Router<br/>LLM picks a path]
  R --> A[Agent loop<br/>LLM picks tools]
  A --> M[Multi-agent<br/>agents coordinate]
```

**Reading it:** each box to the right moves one more decision from your
code to the model. A box further right handles more requests you did not
plan for, and is harder to predict, dearer in tokens and harder to debug. Stop at the
leftmost box that handles the requests you actually get.

One question, "How many PTO days do I get, and what is the meal per-diem
when travelling?", sent to each design, shows the cost:

| design | model calls | why |
|---|---|---|
| fixed workflow | 2 | rewrite as a search, then answer |
| router | 2 | classify, then the chosen handler answers |
| agent loop | 2 | one turn with two parallel tool calls, one answer turn |
| multi-agent | 4 | plan, one call per specialist (2), final answer |

![Workflow, router and agent loop each make 2 calls and multi-agent 4, but the agent loop uses about 990 tokens, ten times the workflow's 93](figures/primer.agents.orchestration.autonomy_costs.svg)

**Reading it:** the left panel counts model calls and the right panel
tokens. Calls stay at 2 until multi-agent doubles them, but tokens climb
sooner. The agent loop sends about ten times the fixed workflow's, because
every call re-sends the tool definitions and tool results, and multi-agent
adds a planning call and the messages between agents. Those tokens pay for
handling requests the workflow was not written for, and are wasted on
requests it already answers.

## 2. The named patterns

Between the fixed workflow and the agent loop sit five designs common
enough to have names, from Anthropic's guide *Building effective agents*.
All five are workflows: code owns the flow and calls the model at set
points. They differ in what the model is trusted to decide.

**In code:** every pattern below is built from `ask`, one model call with
one user message that returns the reply's text.

### Prompt chaining

**Prompt chaining** splits a task into a fixed sequence of model calls,
each output feeding the next prompt. Between calls, a code check called a
**gate** can stop the chain. Here, step 1 writes an outline, and a gate checks that it has at
least three points. The outline `"- scope"` has one, so the chain stops after
one call and the draft step never runs.

```mermaid
flowchart LR
  I[Input] --> S1[LLM: outline] --> G{Gate: 3+ points?}
  G -->|yes| S2[LLM: draft] --> O[Output]
  G -->|no| X[Stop with the reason]
```

**Reading it:** two model calls run in a fixed order, with plain code (the
diamond) between them. An outline that fails takes the "no" arrow and ends
the run with the reason, before a draft is paid for or built on it.

**In code:** `run_chain` runs the steps in order and stops at the first
gate that reports a problem.

### Routing

A chain gives the model no say in which steps run. **Routing** gives it
one: a model call classifies the request, and code sends it to that class's
handler, such as billing or technical support. The label comes back as free
text, so code checks it. `"  Technical\n"` is normalized to `technical` and
routed. `"Billing-ish, maybe refunds?"` is not a known label, so the request
goes to the `general` fallback instead of crashing.

```mermaid
flowchart LR
  Q[Request] --> C[LLM classifier]
  C -->|billing| B[Billing handler]
  C -->|technical| T[Tech handler]
  C -->|anything else| F[Fallback]
```

**Reading it:** the classifier is the only model call that chooses
anything. The "anything else" arrow catches every label not on the list,
since a model's free text can be anything.

**In code:** `route` makes the classifier call, normalizes the label and
swaps anything unknown for the fallback.

### Parallelization: sectioning and voting

Chaining and routing make one call at a time. **Parallelization** makes
several at once. In **sectioning**, each call gets a different piece of the
work, such as a safety check running beside the answer. In **voting**,
every call gets the same prompt and code takes the majority. Three reviewers judge a SQL snippet: two
say "vulnerable" and one says "safe", so the verdict is `("vulnerable", {"vulnerable": 2, "safe": 1})`.

```mermaid
flowchart LR
  Q[Same prompt] --> R1[Reviewer 1]
  Q --> R2[Reviewer 2]
  Q --> R3[Reviewer 3]
  R1 --> V{Majority}
  R2 --> V
  R3 --> V
```

**Reading it:** the three calls run at once, so the wait is the slowest
one's, and plain code counts the answers. Sectioning has the same shape,
with a different sub-task on each branch.

A vote helps when the voters make their mistakes independently. The chance
that the majority is right is then:

$$
P(\text{majority right}) = \sum_{k=\lfloor n/2 \rfloor + 1}^{n} \binom{n}{k}\, p^{k} (1-p)^{n-k}
$$

**Symbols**

| Symbol | Meaning |
|---|---|
| $n$ | number of voters (odd, so there are no ties) |
| $p$ | chance a single voter is right |
| $k$ | number of voters who are right |
| $\binom{n}{k}$ | ways to choose which $k$ of the $n$ are right |

**In words:** add up the chances of every outcome where more than half the
voters are right.

**On the example:** $n = 3$, $p = 0.8$: $0.8^3 + 3 \times 0.8^2 \times 0.2 = 0.512 + 0.384 = 0.896$.
Three reviewers each right 80% of the time make a panel right 89.6% of the
time.

**In Python:**

```python
from math import comb
n, p = 3, 0.8
# the chance exactly k voters are right ...
P = sum(comb(n, k) * p ** k * (1 - p) ** (n - k)
        # ... for every k from ⌊n/2⌋ + 1 to n
        for k in range(n // 2 + 1, n + 1))
round(P, 3)  # → 0.896
```

![Majority accuracy rises with more independent voters: at 70% per voter, 5 voters reach 84% and 15 reach 95%; at 60% per voter the climb is slow](figures/primer.agents.orchestration.voting.svg)

**Reading it:** each curve is one voter's accuracy, from 60% to 90%, and
the x-axis adds voters. Every curve rises, slowly at 60%. All of them
assume independent mistakes. Five copies of one model given one prompt tend
to make the same mistake, and then a vote adds little, so vary the prompt,
the model or the evidence.

**In code:** `run_sections` runs independent pieces of work at once; `vote`
sends every reviewer the same prompt and returns the majority with its
tally; `majority_accuracy` computes the formula.

### Orchestrator-workers

In sectioning, you decide how to split the work when you write the code.
**Orchestrator-workers** lets the model split it, request by request. An
orchestrator call reads the request and lists the subtasks, a worker
handles each one, and a synthesizer call combines their results into one
answer. Asked "What's the meal per-diem, when are receipts due, and does
PTO roll over?", the orchestrator returns three subtasks. Each worker finds
one policy document (`fin-002`, `fin-004`, `hr-001`), and the synthesizer
writes one answer that cites all three.

```mermaid
flowchart TD
  Q[Request] --> O[Orchestrator LLM:<br/>decide the subtasks]
  O --> W1[Worker: per-diem]
  O --> W2[Worker: receipts]
  O --> W3[Worker: PTO rollover]
  W1 --> S[Synthesizer LLM:<br/>one cited answer]
  W2 --> S
  W3 --> S
```

**Reading it:** the shape is sectioning's, but the worker boxes did not
exist until the orchestrator call wrote them, so the split fits each
request, including ones you could not list in advance.

**In code:** `orchestrate` asks the orchestrator for the subtasks, runs a
worker on each in parallel and asks the synthesizer for one answer.

### Evaluator-optimizer

Each pattern so far runs once and stops. **Evaluator-optimizer** loops. A
generator call writes a draft, and an evaluator call checks it against
stated criteria and either passes it or sends back feedback for the next
draft. Draft 1, "PTO rolls over.", gets the feedback "Missing a citation".
Draft 2, "Up to 5 unused PTO days roll over [hr-001].", passes, after 2
rounds.

```mermaid
flowchart LR
  T[Task] --> G[Generator drafts]
  G --> E{Evaluator:<br/>PASS?}
  E -->|feedback| G
  E -->|PASS| D[Done]
  E -->|max rounds| B[Stop: best effort, flagged]
```

**Reading it:** the loop has two ways out, PASS and the round limit.
Without the limit, an evaluator that is never satisfied would keep it
running forever. The pattern works when the criteria can be checked, as
"cites a policy document" can.

**In code:** `evaluate_optimize` alternates drafts and verdicts, feeding
each critique back, and returns the last draft, the rounds used and whether
it passed.

## 3. Multi-agent: a supervisor and specialists

The agent loop, the third box on the spectrum, is the first design in
which the model owns the flow, and it has a lesson of its own (see
`primer.agents.agent_loop`). The last box puts several agents on one job.
In the form built here, a **supervisor** agent does none of the work
itself: it splits the job, writes each specialist agent a short brief, and
assembles their answers. The supervisor sends "PTO days per year?" to the
`hr` specialist and "Meal per-diem?" to `finance`, and neither specialist
sees the other's question. A task for a `legal` specialist that does not
exist is reported back instead of crashing the run.

```mermaid
sequenceDiagram
  participant U as User
  participant S as Supervisor
  participant H as HR agent
  participant F as Finance agent
  U->>S: PTO and meal allowance?
  S->>H: "PTO days per year?" (only this)
  S->>F: "Meal per-diem?" (only this)
  H-->>S: 20 days [hr-001]
  F-->>S: 75 dollars a day [fin-002]
  S-->>U: combined answer
```

**Reading it:** time runs downward. The two messages from the supervisor
are the **hand-offs**, and each is all its specialist knows. That keeps each
specialist's context small and focused, and anything the supervisor leaves
out is lost. Peer designs, where agents hand control to one another, lose
context at hand-offs the same way.

Several agents pay off when the subtasks are truly separate, such as
independent research threads, or documents that together would overflow one
context. They cost more tokens, lose context at every hand-off, and are much
harder to debug.

**In code:** `supervise` gets a delegation plan from the supervisor, sends
each specialist only its own task, then asks the supervisor for the final
answer.

## 4. An explicit state machine, with the model at specific nodes

A job whose steps are known in advance belongs at the other end of the
spectrum, in a fixed workflow. One way to build that workflow so every move
can be tested and audited is an explicit **state machine**: a fixed set of
states, a rule for each move between them, and the model asked only at the
states that need judgement. Take invoice intake. The model is asked twice:
"is this an invoice?" and "extract the number, vendor and amount". Code does
the rest. It validates the fields, holds any invoice over 5,000 for a person
to approve, and posts the others to the ledger. Three documents go through
it:

```text
INVOICE INV-2041 ... 1200.00   -> RECEIVED > CLASSIFIED > EXTRACTED > VALIDATED > POSTED > DONE
INVOICE INV-2042 ... 18000.00  -> RECEIVED > CLASSIFIED > EXTRACTED > VALIDATED > AWAITING_APPROVAL
Acme newsletter                -> RECEIVED > REJECTED
```

```mermaid
stateDiagram-v2
  [*] --> RECEIVED
  RECEIVED --> CLASSIFIED: LLM says invoice
  RECEIVED --> REJECTED: LLM says other
  CLASSIFIED --> EXTRACTED: LLM extracts fields
  EXTRACTED --> VALIDATED: code checks pass
  EXTRACTED --> NEEDS_REVIEW: code checks fail
  VALIDATED --> AWAITING_APPROVAL: amount > limit
  VALIDATED --> POSTED: code posts to ledger
  AWAITING_APPROVAL --> POSTED: human approves
  POSTED --> DONE
```

**Reading it:** every box is a state and every arrow a transition. Only two
arrows are labelled "LLM"; the rest are rules you can read, test and audit.
When a run goes wrong, its history shows which state it was in and which
rule moved it on.

**In code:** `InvoiceWorkflow.run` walks the states, calling the model only
at RECEIVED and CLASSIFIED, and `InvoiceWorkflow.approve` is the human step
that releases a held invoice.

### Checkpoints and durable execution

A state machine always knows which state a run is in, so after every
transition it can save the run to disk as a **checkpoint**. After a crash,
the rerun loads the checkpoint and resumes after the last completed step.
A workflow whose progress survives crashes this way has **durable
execution**. Engines such as Temporal provide it at scale.

![After one crash while posting, resuming from the checkpoint finishes in 2 model calls, while restarting from scratch takes 4](figures/primer.agents.orchestration.resume_cost.svg)

**Reading it:** both bars meet the same crash, while posting. The resumed
run finishes with the 2 model calls it had already made. The restarted run
asks the model both questions again, doubling the cost, and the model may
give a different answer the second time.

**In code:** `InvoiceWorkflow` saves a checkpoint after every transition
and loads it at the start of `InvoiceWorkflow.run`. `resume_comparison`
crashes the ledger once and counts the model calls each way.

## 5. Frameworks vs. plain code

Every design above can be written by hand or taken from a framework.
LangGraph models agents as graphs with explicit state and checkpoints. The
Claude Agent SDK and the OpenAI Agents SDK give you vendor-built agent
loops, tools and hand-offs. CrewAI and AutoGen focus on teams of agents.
Temporal provides durable execution for any code. A framework earns its
place when you want its standard patterns, built-in tracing, saved state
and hooks for a person to step in. Plain code suits simple flows, and
gives full control with fewer dependencies. Each design in this lesson is under 100
lines of plain Python, and having seen them you can judge what a framework
would add.

## In 20 seconds
- Designs run from a fixed workflow, where code decides every step, to several agents. Use the least autonomy that solves the problem.
- The named workflow patterns: prompt chaining with gates, routing with a fallback, sectioning and voting, orchestrator-workers, evaluator-optimizer.
- Several agents buy focused context and parallelism, and cost tokens, context lost at hand-offs and debuggability.
- Production shape: an explicit state machine that asks the model only at chosen states and saves a checkpoint after every step.

## Self-test questions

**Q: When would you use a fixed workflow instead of an agent? Give an example.**
A: When the steps are known in advance and the variation is *inside* each
step, not in which steps happen. Invoice intake is the example: classify,
extract, validate, approve, post. A workflow is cheaper, faster, testable
and auditable. The model is used where judgement is needed (classification,
extraction), and code owns the flow. Reach for an agent only when the path
genuinely depends on what's discovered along the way.

**Q: Single agent vs. multi-agent for a document-processing workflow: what does each side have going for it?**
A: *For one agent (or a workflow):* documents flow through the same steps,
hand-offs lose context, multi-agent multiplies tokens, and one trace is far
easier to debug. *For several agents:* documents are independent and can be
processed in parallel, each document or section may overflow a single
context, and specialists (tables, legal clauses, figures) benefit from
focused instructions and tools. A common answer is a workflow that fans out
per document to a focused worker, which is orchestrator-workers rather than free-form
agents talking to each other.

**Q: A router's classifier sometimes returns labels that aren't in its list. How should the code handle that?**
A: Normalize (case, whitespace), accept only known labels, and send
everything else to a safe fallback. Log the misses and add them to the
classifier's evaluation set. Consider structured output with an `enum` so
the model can only produce valid labels.

**Q: Why checkpoint after every step of a long workflow?**
A: So a crash costs one step, not the run. Resuming reuses completed work
(no repeated model calls, no double-posted invoices when combined with
idempotency keys) and avoids getting a different answer on the rerun.

## The papers behind this lesson

- **Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2022).**
  https://arxiv.org/abs/2210.03629. It established the reason-act-observe
  loop that the "agent loop" rung of the spectrum is built on.
  [annotated companion](../../papers/react.html)

## Further reading
- Anthropic, *Building effective agents* (the named patterns): https://www.anthropic.com/engineering/building-effective-agents
- LangGraph docs: https://langchain-ai.github.io/langgraph/
- OpenAI Agents SDK: https://openai.github.io/openai-agents-python/
- CrewAI docs: https://docs.crewai.com/
- AutoGen: https://microsoft.github.io/autogen/
- Temporal, durable execution: https://docs.temporal.io/
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from primer.agents.llm import LLM, ScriptedLLM, last_user_text


def ask(llm: LLM, prompt: str, system: str = "") -> str:
    """One model call with one user message; returns the text."""
    return llm.complete(system=system, messages=[{"role": "user", "content": prompt}]).text


# ---------------------------------------------------------------------------
# Prompt chaining
# ---------------------------------------------------------------------------


@dataclass
class ChainStep:
    name: str
    prompt: str  # template; "{input}" is replaced by the previous step's output
    gate: Callable[[str], str | None] | None = None  # returns a problem, or None to continue


@dataclass
class ChainResult:
    outputs: dict[str, str]
    stopped_at: str | None = None
    reason: str | None = None


def run_chain(llm: LLM, task: str, steps: list[ChainStep]) -> ChainResult:
    """Fixed sequence of calls; each output feeds the next, with a code check (gate) in between."""
    outputs: dict[str, str] = {}
    current = task
    for step in steps:
        current = ask(llm, step.prompt.format(input=current))
        outputs[step.name] = current
        problem = step.gate(current) if step.gate else None
        if problem:
            return ChainResult(outputs, step.name, problem)
    return ChainResult(outputs)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route(
    classifier: LLM, handlers: dict[str, Callable[[str], str]], request: str, fallback: str = "general"
) -> tuple[str, str]:
    """Classify the request with one model call, then hand it to that label's handler.

    The label comes back as free text, so it's normalized and checked against
    the known labels. Anything else goes to the fallback instead of crashing
    or guessing.
    """
    raw = ask(classifier, f"Classify into one of {sorted(handlers)}. Reply with the label only.\n\n{request}")
    label = raw.strip().lower()
    if label not in handlers:
        label = fallback
    return label, handlers[label](request)


# ---------------------------------------------------------------------------
# Parallelization: sectioning and voting
# ---------------------------------------------------------------------------


def run_sections(sections: dict[str, Callable[[], str]]) -> dict[str, str]:
    """Run independent pieces of work at the same time; wall-clock time is the slowest piece."""
    with ThreadPoolExecutor(max_workers=len(sections)) as pool:
        futures = {name: pool.submit(fn) for name, fn in sections.items()}
        return {name: f.result() for name, f in futures.items()}


def vote(reviewers: list[LLM], prompt: str) -> tuple[str, dict[str, int]]:
    """Ask several models (or one model several times) the same question; take the majority."""
    answers = run_sections({str(i): (lambda r=r: ask(r, prompt).strip().lower()) for i, r in enumerate(reviewers)})
    tally = Counter(answers.values())
    return tally.most_common(1)[0][0], dict(tally)


def majority_accuracy(p: float, n_voters: int) -> float:
    """Chance a majority of n independent voters, each right with probability p, is right."""
    return sum(math.comb(n_voters, k) * p**k * (1 - p) ** (n_voters - k) for k in range(n_voters // 2 + 1, n_voters + 1))


# ---------------------------------------------------------------------------
# Orchestrator-workers
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorResult:
    subtasks: list[str]
    worker_outputs: dict[str, str]
    final: str


def orchestrate(orchestrator: LLM, worker: Callable[[str], str], synthesizer: LLM, request: str) -> OrchestratorResult:
    """One model splits the request into subtasks, workers handle them in parallel, one model combines.

    Unlike sectioning, the subtasks aren't known in advance: the orchestrator
    decides them from the request.
    """
    plan = ask(orchestrator, f'Split into independent subtasks. Reply as JSON {{"subtasks": [...]}}.\n\n{request}')
    subtasks = list(json.loads(plan)["subtasks"])
    outputs = run_sections({s: (lambda s=s: worker(s)) for s in subtasks})
    notes = "\n".join(f"- {s}: {outputs[s]}" for s in subtasks)
    final = ask(synthesizer, f"Question: {request}\n\nFindings:\n{notes}\n\nWrite one answer that cites the [doc ids].")
    return OrchestratorResult(subtasks, outputs, final)


def policy_worker(subtask: str) -> str:
    """A worker: find the one current policy document that answers the subtask.

    Two things a dense-only lookup gets wrong here, and the fixes real systems use:
    it happily returns the superseded 2023 travel policy (fix: filter on
    metadata before ranking), and it confuses PTO with sick leave because both
    are "time off" (fix: add a keyword-overlap bonus, a tiny hybrid search;
    see primer.ml.embeddings.retrieval).
    """
    from primer.common.corpus import DOCS
    from primer.common.embedder import ConceptEmbedder
    from primer.common.text import tokenize

    current = [d for d in DOCS if "superseded" not in d.title.lower()]
    emb = ConceptEmbedder()
    dense = emb.encode([d.title + ". " + d.text for d in current]) @ emb.encode(subtask)
    words = set(tokenize(subtask))
    keyword = np.array([len(words & set(tokenize(d.title + " " + d.text))) for d in current])
    best = current[int(np.argmax(dense + 0.2 * keyword))]
    return f"[{best.id}] {best.text}"


def demo_orchestrator() -> ScriptedLLM:
    return ScriptedLLM(['{"subtasks": ["meal per-diem when travelling", "deadline for expense receipts", "PTO rollover"]}'])


def demo_synthesizer() -> ScriptedLLM:
    """Combines whatever findings it's given, keeping each source id."""

    def policy(system, messages, tools):
        findings = [line for line in last_user_text(messages).splitlines() if line.startswith("- ")]
        parts = [f"{line.split(': ', 1)[0][2:]}: see {re.search(r'\[[a-z]+-\d+\]', line).group(0)}" for line in findings]
        return "Here's what the policies say. " + "; ".join(parts) + "."

    return ScriptedLLM(policy)


# ---------------------------------------------------------------------------
# Evaluator-optimizer
# ---------------------------------------------------------------------------


def evaluate_optimize(generator: LLM, evaluator: LLM, task: str, max_rounds: int = 3) -> tuple[str, int, bool]:
    """Writer drafts, editor critiques, writer revises, until the editor says PASS or rounds run out.

    Returns (last draft, rounds used, passed). Stopping at max_rounds matters:
    a critic that is never satisfied would otherwise loop forever.
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
    draft = ""
    for rnd in range(1, max_rounds + 1):
        draft = generator.complete(system="Write the answer.", messages=messages).text
        verdict = ask(evaluator, f"Task: {task}\nDraft: {draft}\nReply PASS, or say what to fix.", system="You are a strict editor.")
        if verdict.strip().upper().startswith("PASS"):
            return draft, rnd, True
        messages += [{"role": "assistant", "content": draft}, {"role": "user", "content": f"Editor feedback: {verdict}"}]
    return draft, max_rounds, False


# ---------------------------------------------------------------------------
# Multi-agent: a supervisor delegating to specialists
# ---------------------------------------------------------------------------


@dataclass
class SupervisorResult:
    delegations: list[tuple[str, str]]
    answers: dict[str, str]
    final: str


def supervise(supervisor: LLM, specialists: dict[str, LLM], request: str) -> SupervisorResult:
    """A supervisor decides who does what; each specialist works in its own, fresh context.

    Each specialist receives only its task text, not the user's full message and not
    the other specialists' tasks. That focus is the main benefit of several agents.
    The cost is that anything the supervisor fails to write into the task is lost at
    the hand-off.
    """
    plan = json.loads(ask(supervisor, f'Delegate to {sorted(specialists)}. Reply as JSON {{"delegate": [{{"to": ..., "task": ...}}]}}.\n\n{request}'))
    delegations = [(d["to"], d["task"]) for d in plan["delegate"]]
    answers: dict[str, str] = {}
    for name, task in delegations:
        if name not in specialists:
            answers[name] = f"No specialist named {name!r}. Available: {', '.join(sorted(specialists))}."
            continue
        answers[name] = ask(specialists[name], task)
    findings = "\n".join(f"- {n}: {a}" for n, a in answers.items())
    final = ask(supervisor, f"Question: {request}\n\nSpecialist answers:\n{findings}\n\nWrite the final answer.")
    return SupervisorResult(delegations, answers, final)


# ---------------------------------------------------------------------------
# An explicit state machine with LLM decisions only at specific nodes
# ---------------------------------------------------------------------------

TERMINAL = ("DONE", "REJECTED", "NEEDS_REVIEW", "AWAITING_APPROVAL")


@dataclass
class WorkflowRun:
    state: str
    history: list[str]
    data: dict[str, Any] = field(default_factory=dict)


class InvoiceWorkflow:
    """Invoice intake as a state machine. Code owns every transition; the model is
    consulted at exactly two nodes (classify, extract).

    After every transition the run is saved to a checkpoint file, so a crash
    resumes from the last completed state instead of starting over, and the
    model is never asked the same question twice. This is durable execution in
    miniature.
    """

    def __init__(self, llm: LLM, checkpoint: str | Path, post: Callable[[dict[str, Any]], Any], approval_limit: float = 5000.0):
        self.llm = llm
        self.checkpoint = Path(checkpoint)
        self.post = post
        self.approval_limit = approval_limit

    def _save(self, run: WorkflowRun) -> None:
        # Write to a temp file then rename: a crash mid-write can't leave a half-written checkpoint.
        tmp = self.checkpoint.with_suffix(".tmp")
        tmp.write_text(json.dumps({"state": run.state, "history": run.history, "data": run.data}))
        os.replace(tmp, self.checkpoint)

    def _load(self) -> WorkflowRun | None:
        if not self.checkpoint.exists():
            return None
        saved = json.loads(self.checkpoint.read_text())
        return WorkflowRun(saved["state"], saved["history"], saved["data"])

    def _move(self, run: WorkflowRun, state: str) -> None:
        run.state = state
        run.history.append(state)
        self._save(run)

    def run(self, document: str) -> WorkflowRun:
        run = self._load() or WorkflowRun("RECEIVED", ["RECEIVED"], {"document": document})
        while run.state not in TERMINAL:
            if run.state == "RECEIVED":
                # LLM node 1: a judgement call code can't easily make.
                kind = ask(self.llm, f"Classify this document as 'invoice' or 'other':\n{run.data['document']}").strip().lower()
                self._move(run, "CLASSIFIED" if kind == "invoice" else "REJECTED")
            elif run.state == "CLASSIFIED":
                # LLM node 2: pull structured fields out of messy text.
                run.data["invoice"] = json.loads(ask(self.llm, f"Extract invoice_no, vendor, amount as JSON:\n{run.data['document']}"))
                self._move(run, "EXTRACTED")
            elif run.state == "EXTRACTED":
                problem = validate_invoice(run.data["invoice"])
                if problem:
                    run.data["problem"] = problem
                    self._move(run, "NEEDS_REVIEW")
                else:
                    self._move(run, "VALIDATED")
            elif run.state == "VALIDATED":
                if run.data["invoice"]["amount"] > self.approval_limit:
                    self._move(run, "AWAITING_APPROVAL")
                else:
                    # If this raises, the checkpoint still says VALIDATED and a rerun posts again.
                    # A real ledger call would carry invoice_no as an idempotency key.
                    self.post(run.data["invoice"])
                    self._move(run, "POSTED")
            elif run.state == "POSTED":
                self._move(run, "DONE")
        return run


    def approve(self, approver: str) -> WorkflowRun:
        """A person approves a paused invoice; the run continues from the checkpoint."""
        run = self._load()
        if run is None or run.state != "AWAITING_APPROVAL":
            raise ValueError("nothing is waiting for approval")
        run.data["approved_by"] = approver
        self.post(run.data["invoice"])
        self._move(run, "POSTED")
        self._move(run, "DONE")
        return run


def validate_invoice(invoice: dict[str, Any]) -> str | None:
    if not re.fullmatch(r"INV-\d+", str(invoice.get("invoice_no", ""))):
        return f"invoice_no must look like INV-1234, got {invoice.get('invoice_no')!r}"
    if not isinstance(invoice.get("amount"), (int, float)) or invoice["amount"] <= 0:
        return f"amount must be positive, got {invoice.get('amount')!r}"
    return None


def demo_document_llm() -> ScriptedLLM:
    """A stand-in model for the two LLM nodes: classification and extraction."""

    def policy(system, messages, tools):
        prompt = last_user_text(messages)
        document = prompt.split("\n", 1)[1]
        if prompt.startswith("Classify"):
            return "invoice" if document.startswith("INVOICE") else "other"
        m = re.search(r"INVOICE (\S+) from (.+?)\. Amount due: (-?[\d.]+)", document)
        return json.dumps({"invoice_no": m.group(1), "vendor": m.group(2), "amount": float(m.group(3))})

    return ScriptedLLM(policy)



# ---------------------------------------------------------------------------
# One question, four levels of autonomy
# ---------------------------------------------------------------------------

AUTONOMY_QUESTION = "How many PTO days do I get, and what is the meal per-diem when travelling?"


def autonomy_costs() -> list[dict[str, Any]]:
    """Answer the same two-part question at each rung of the autonomy ladder; count calls and tokens."""
    from primer.agents.agent_loop import DEMO_TOOL_DEFS, DEMO_TOOLS, run_agent, search_kb
    from primer.agents.llm import ToolCall, tool_results

    rows = []

    # 1. Fixed workflow: code decides the steps (rewrite the question as a search, then answer).
    llm = ScriptedLLM(["PTO days per year; meal per-diem", "20 PTO days a year [hr-001]; 75 dollars a day [fin-002]."])
    run_chain(llm, AUTONOMY_QUESTION, [ChainStep("query", "Rewrite as a search query: {input}"), ChainStep("answer", "Answer using: {input}")])
    rows.append({"design": "fixed workflow", "llm": llm})

    # 2. Router: the model picks a path; the chosen handler makes one more call.
    llm = ScriptedLLM(["policy", "20 PTO days a year [hr-001]; 75 dollars a day [fin-002]."])
    route(llm, {"policy": lambda q: ask(llm, q), "general": lambda q: ask(llm, q)}, AUTONOMY_QUESTION)
    rows.append({"design": "router", "llm": llm})

    # 3. Agent loop: the model chooses tools (two searches in parallel), then answers.
    def agent_policy(system, messages, tools):
        if not tool_results(messages):
            return [ToolCall("", "search_kb", {"query": "PTO days"}), ToolCall("", "search_kb", {"query": "meal per-diem"})]
        return "20 PTO days a year [hr-001]; 75 dollars a day [fin-002]."

    llm = ScriptedLLM(agent_policy)
    run_agent(llm, AUTONOMY_QUESTION, DEMO_TOOLS, DEMO_TOOL_DEFS)
    rows.append({"design": "agent loop", "llm": llm})

    # 4. Multi-agent: a supervisor plus two specialists, each with its own context.
    sup = ScriptedLLM([
        '{"delegate": [{"to": "hr", "task": "PTO days per year?"}, {"to": "finance", "task": "Meal per-diem?"}]}',
        "20 PTO days a year [hr-001]; 75 dollars a day [fin-002].",
    ])
    hr, fin = ScriptedLLM([search_kb("PTO days")]), ScriptedLLM([search_kb("meal per-diem")])
    supervise(sup, {"hr": hr, "finance": fin}, AUTONOMY_QUESTION)
    rows.append({"design": "multi-agent", "llm": sup, "extra": [hr, fin]})

    out = []
    for r in rows:
        models = [r["llm"], *r.get("extra", [])]
        out.append({
            "design": r["design"],
            "calls": sum(m.calls for m in models),
            "input_tokens": sum(m.total_usage.input_tokens for m in models),
            "output_tokens": sum(m.total_usage.output_tokens for m in models),
        })
    return out


def resume_comparison(tmp_dir: str | Path) -> dict[str, int]:
    """Model calls spent finishing one invoice when posting crashes once: resume vs. restart."""
    results = {}
    for mode in ("resume from checkpoint", "restart from scratch"):
        llm = demo_document_llm()
        attempts: list[dict] = []

        def flaky_post(invoice: dict[str, Any]) -> None:
            attempts.append(invoice)
            if len(attempts) == 1:
                raise ConnectionError("ledger unavailable")

        ckpt = Path(tmp_dir) / f"{mode.split()[0]}.json"
        wf = InvoiceWorkflow(llm, ckpt, post=flaky_post)
        try:
            wf.run("INVOICE INV-2041 from Acme Supplies. Amount due: 1200.00 USD.")
        except ConnectionError:
            if mode == "restart from scratch":
                ckpt.unlink()  # no durable state: the second attempt starts over
        wf.run("INVOICE INV-2041 from Acme Supplies. Amount due: 1200.00 USD.")
        results[mode] = llm.calls
    return results


# ---------------------------------------------------------------------------
# Figures and walkthrough
# ---------------------------------------------------------------------------


def figures() -> dict:
    """Plots computed from this lesson's own code (matplotlib imported here)."""
    import tempfile

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = {}

    rows = autonomy_costs()
    names = [r["design"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    ax1.bar(names, [r["calls"] for r in rows], color="#5b8fd6")
    ax1.set(ylabel="model calls", title="Calls for the same question")
    ax2.bar(names, [r["input_tokens"] + r["output_tokens"] for r in rows], color="#d98c3a")
    ax2.set(ylabel="tokens (input + output)", title="Tokens for the same question")
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    figs["autonomy_costs"] = fig

    voters = list(range(1, 16, 2))
    fig, ax = plt.subplots(figsize=(7, 4))
    for p in (0.6, 0.7, 0.8, 0.9):
        ax.plot(voters, [majority_accuracy(p, n) for n in voters], "o-", ms=4, label=f"each voter right {p:.0%}")
    ax.set(xlabel="number of independent voters (odd)", ylabel="P(majority is right)", ylim=(0.5, 1.01),
           title="Voting: many noisy judges beat one (if their errors are independent)")
    ax.legend()
    figs["voting"] = fig

    with tempfile.TemporaryDirectory() as d:
        costs = resume_comparison(d)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars = ax.bar(list(costs), list(costs.values()), color=["#6bb36b", "#c0392b"])
    ax.bar_label(bars)
    ax.set(ylabel="model calls to finish one invoice", title="Posting crashed once: what did it cost?")
    figs["resume_cost"] = fig
    return figs


def demo() -> None:
    import tempfile

    from primer._show import banner, say, table, takeaway

    banner("1. One question, four levels of autonomy")
    table(["design", "model calls", "input tokens", "output tokens"],
          [(r["design"], r["calls"], r["input_tokens"], r["output_tokens"]) for r in autonomy_costs()])
    takeaway("Use the least autonomy that solves the problem. Every rung up costs calls, tokens and predictability.")

    banner("2. Prompt chaining with a gate")
    gate = lambda text: None if text.count("- ") >= 3 else "outline needs at least 3 points"  # noqa: E731
    for outline in ("- scope\n- risks\n- timeline", "- scope"):
        llm = ScriptedLLM([outline, "A draft that follows the outline."])
        r = run_chain(llm, "Q3 plan", [ChainStep("outline", "Outline: {input}", gate), ChainStep("draft", "Draft from: {input}")])
        print(f"  outline {outline!r:32} -> stopped_at={r.stopped_at}, calls={llm.calls}, reason={r.reason}")
    print()

    banner("3. Routing (with a fallback for labels the model invents)")
    handlers = {"billing": lambda q: "billing team", "technical": lambda q: "tech support", "general": lambda q: "front desk"}
    for raw in ("billing", "  Technical\n", "Billing-ish, maybe refunds?"):
        print(f"  classifier said {raw!r:32} -> {route(ScriptedLLM([raw]), handlers, 'q')[0]}")
    print()

    banner("4. Parallelization: voting")
    reviewers = [ScriptedLLM(["vulnerable"]), ScriptedLLM(["safe"]), ScriptedLLM(["vulnerable"])]
    print(f"  three reviewers -> {vote(reviewers, 'Is this SQL safe?')}")
    table(["voters", "p=0.7", "p=0.8"], [(n, majority_accuracy(0.7, n), majority_accuracy(0.8, n)) for n in (1, 3, 5, 9)], floatfmt=".3f")

    banner("5. Orchestrator-workers")
    r = orchestrate(demo_orchestrator(), policy_worker, demo_synthesizer(),
                    "What's the meal per-diem, when are receipts due, and does PTO roll over?")
    for s in r.subtasks:
        print(f"  worker[{s}] -> {r.worker_outputs[s][:70]}...")
    say(f"Final: {r.final}")

    banner("6. Evaluator-optimizer")
    writer = ScriptedLLM(["PTO rolls over.", "Up to 5 unused PTO days roll over [hr-001]."])
    editor = ScriptedLLM(["Missing a citation: cite the policy id in brackets.", "PASS"])
    say(f"(draft, rounds, passed) = {evaluate_optimize(writer, editor, 'Does PTO roll over?')}")

    banner("7. Multi-agent: supervisor and specialists")
    res = supervise(
        ScriptedLLM(['{"delegate": [{"to": "hr", "task": "PTO days per year?"}, {"to": "finance", "task": "Meal per-diem?"}]}',
                     "20 PTO days a year [hr-001]; 75 dollars a day [fin-002]."]),
        {"hr": ScriptedLLM(["20 days [hr-001]"]), "finance": ScriptedLLM(["75 dollars a day [fin-002]"])},
        "PTO and per-diem?",
    )
    for (who, task), answer in zip(res.delegations, res.answers.values()):
        print(f"  {who:8} got only: {task!r:24} -> {answer}")
    say(f"Final: {res.final}")

    banner("8. A state machine with LLM decisions at two nodes, and checkpoints")
    with tempfile.TemporaryDirectory() as d:
        for i, doc in enumerate(["INVOICE INV-2041 from Acme Supplies. Amount due: 1200.00 USD.",
                                 "INVOICE INV-2042 from Globex. Amount due: 18000.00 USD.",
                                 "Acme Supplies newsletter: our autumn catalogue is here!"]):
            run = InvoiceWorkflow(demo_document_llm(), Path(d) / f"c{i}.json", post=lambda inv: None).run(doc)
            print(f"  {doc[:40]:42} -> {' > '.join(run.history)}")
        print()
        say(f"Posting crashes once. Model calls to finish: {resume_comparison(d)}")
    takeaway("Code owns the transitions; the model decides only where judgement is needed. Checkpoints make crashes cheap.")


if __name__ == "__main__":
    demo()
