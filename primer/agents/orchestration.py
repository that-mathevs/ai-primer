r"""
# Orchestration: how much autonomy to give the model

Run: `python -m primer.agents.orchestration`

"Agent" covers everything from one model call inside ordinary code to a
team of models steering each other. This lesson lays out that range, builds
each named pattern in a few dozen lines, and measures what each step up
costs. The rule that falls out is simple: **use the least autonomy that
solves the problem, and say so out loud.**

## 1. The autonomy spectrum

**Everyday picture.** Four ways to run an office. An **assembly line**
(fixed workflow): the steps never change and people do one task at each
station. A **receptionist** (router) decides which department you go to. A
**personal assistant** (agent loop) decides which errands to run and when
the job's done. A **team with a manager** (multi-agent) splits the work
among specialists.

```mermaid
flowchart LR
  W[Fixed workflow<br/>code decides steps] --> R[Router<br/>LLM picks a path]
  R --> A[Agent loop<br/>LLM picks tools]
  A --> M[Multi-agent<br/>agents coordinate]
```

**Reading it:** left to right, the model decides more and your code
decides less. Each arrow buys flexibility (the system can handle requests
you didn't anticipate) and costs predictability, tokens and debuggability.
The skill is stopping at the leftmost box that handles your real
requests.

**Tiny worked example.** One question, "How many PTO days do I get, and
what is the meal per-diem when travelling?", answered by each design
(`autonomy_costs()`):

| design | model calls | why |
|---|---|---|
| fixed workflow | 2 | rewrite as a search, then answer |
| router | 2 | classify, then the chosen handler answers |
| agent loop | 2 | one turn with two parallel tool calls, one answer turn |
| multi-agent | 4 | plan, one call per specialist (2), final answer |

![Calls and tokens for the same question at each level of autonomy](figures/primer.agents.orchestration.autonomy_costs.svg)

**Reading it:** the left panel counts model calls and the right counts tokens.
Calls barely move until multi-agent doubles them. Tokens tell a sharper story.
The agent loop re-sends its tool definitions and tool results on every call,
so it uses many times the tokens of the fixed workflow for the same answer. The
supervisor pays for planning and hand-offs. None of that is waste *if* you
need the flexibility, and all of it is waste if you don't.

## 2. The named patterns

These names (from Anthropic's *Building effective agents*) let you describe
a design in one word.

### Prompt chaining

**Everyday picture.** A relay race with an inspector at each hand-off: if
the baton is dropped, the next runner doesn't start.

**Tiny worked example.** Step 1 writes an outline, and a code *gate* checks it has at
least three points. The outline `"- scope"` fails the gate, so the chain stops after one
call and the draft step never runs.

```mermaid
flowchart LR
  I[Input] --> S1[LLM: outline] --> G{Gate: 3+ points?}
  G -->|yes| S2[LLM: draft] --> O[Output]
  G -->|no| X[Stop with the reason]
```

**Reading it:** two model calls in a fixed order, with plain code in the
middle. The gate is where you catch a bad intermediate result before
paying for, and being misled by, the next step.

### Routing

**Everyday picture.** The receptionist listens for five seconds and sends you
to billing, tech support or the front desk.

**Tiny worked example.** The classifier replies `"  Technical\n"`, which is
normalized to `technical` and routed. It replies `"Billing-ish, maybe refunds?"`,
which is not a known label, so the request goes to the `general` fallback
instead of crashing.

```mermaid
flowchart LR
  Q[Request] --> C[LLM classifier]
  C -->|billing| B[Billing handler]
  C -->|technical| T[Tech handler]
  C -->|anything else| F[Fallback]
```

**Reading it:** one cheap model call decides the path, then specialised
handling takes over. The "anything else" arrow is essential, because the label is free
text from a model and code must never assume it's one of the keys.

### Parallelization: sectioning and voting

**Everyday picture.** *Sectioning:* several cooks each make one dish at the
same time. *Voting:* a panel of judges, where the majority decides.

**Tiny worked example.** Three reviewers judge a SQL snippet: two say
"vulnerable" and one says "safe", so the verdict is `("vulnerable", {"vulnerable": 2, "safe": 1})`.

```mermaid
flowchart LR
  Q[Same prompt] --> R1[Reviewer 1]
  Q --> R2[Reviewer 2]
  Q --> R3[Reviewer 3]
  R1 --> V{Majority}
  R2 --> V
  R3 --> V
```

**Reading it:** the three calls run at the same time (wall-clock time is the
slowest one), and plain code counts the answers. Sectioning looks the same
except each branch gets a *different* sub-task, such as a guardrail check
running beside the answer.

Why voting helps, when voters err independently:

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
Three 80% judges make an 89.6% panel.

![Majority accuracy vs. number of voters](figures/primer.agents.orchestration.voting.svg)

**Reading it:** each curve is a per-voter accuracy, and the x-axis adds voters.
Every curve rises: more independent judges, better majority. The catch is
the word *independent*. Five copies of the same model with the same prompt
tend to make the *same* mistake, and then voting buys little. Vary the
prompt, the model or the evidence.

### Orchestrator-workers

**Everyday picture.** A project lead reads the brief, splits it into
sections, hands each to a writer, then edits the pieces into one report.

**Tiny worked example.** "What's the meal per-diem, when are receipts due,
and does PTO roll over?" The orchestrator returns three subtasks. Three workers each
find one document (`fin-002`, `fin-004`, `hr-001`), and the synthesizer
writes one answer citing all three.

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

**Reading it:** it looks like sectioning, but the subtasks aren't known in
advance. The orchestrator *decides* them from the request, which is what makes it
suited to open-ended requests.

### Evaluator-optimizer

**Everyday picture.** A writer and an editor. The draft goes back and forth
until the editor signs off, or the deadline hits.

**Tiny worked example.** Draft 1, "PTO rolls over.", gets the feedback "Missing a citation".
Draft 2, "Up to 5 unused PTO days roll over [hr-001].", passes. Result: 2 rounds,
passed.

```mermaid
flowchart LR
  T[Task] --> G[Generator drafts]
  G --> E{Evaluator:<br/>PASS?}
  E -->|feedback| G
  E -->|PASS| D[Done]
  E -->|max rounds| B[Stop: best effort, flagged]
```

**Reading it:** the loop only exits on PASS or on the round limit. Without
the limit, a critic that's never satisfied loops forever. Clear, checkable
criteria make this pattern work.

## 3. Multi-agent: a supervisor and specialists

**Everyday picture.** A manager who never does the work, only assigns it and
assembles the results. Each specialist gets a short, focused brief.

**Tiny worked example.** The supervisor sends "PTO days per year?" to `hr` and
"Meal per-diem?" to `finance`. The HR specialist never sees the finance question,
and vice versa (tested). A delegation to a non-existent `legal` specialist
is reported, not crashed on.

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

**Reading it:** the supervisor's messages to each specialist are the
**hand-off**, and they're *all* each specialist knows. That's both the
benefit (small, focused context) and the risk (whatever the supervisor
forgets to write down is lost). Peer designs, where agents hand control
directly to each other, have the same hand-off problem without a central
coordinator.

**When it's worth it.** Use several agents when subtasks are genuinely
separable, such as independent research threads or per-document work that would
overflow one context. The costs are real: more tokens, context lost at every
hand-off, and much harder debugging.

## 4. An explicit state machine, with the model at specific nodes

**Everyday picture.** A board game. The squares and the rules for moving
between them are fixed. On a few special squares you draw a card (ask the
model). Everywhere else, the rules decide.

**Tiny worked example.** Invoice intake. The model is asked twice: "is
this an invoice?" and "extract the number, vendor and amount". Code does
everything else: validation, the approval rule (over 5,000 waits for a person),
and posting. The runs:

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

**Reading it:** every box is a state and every arrow a transition, and only
two arrows are labelled "LLM". The rest are rules you can read, test and
audit. When something goes wrong, you know exactly which state it was in and
why it moved.

**Checkpoints and durable execution.** After every transition the run is
saved to disk. **Durable execution** means a workflow whose progress
survives crashes, because each completed step's result is stored and the
run resumes from there. Engines like Temporal do this at scale.

![Model calls to finish one invoice when posting crashes once](figures/primer.agents.orchestration.resume_cost.svg)

**Reading it:** the same crash happens in both bars. Resuming from the checkpoint
finishes with the 2 model calls already made. Restarting from scratch asks
the model both questions again, doubling cost, and, worse, risks a
*different* answer the second time.

## 5. Frameworks vs. plain code

**LangGraph** models agents as graphs with explicit state and checkpoints.
The **Claude Agent SDK** and **OpenAI Agents SDK** give you vendor-built agent
loops, tools and hand-offs. **CrewAI** and **AutoGen** focus on multi-agent
teams. **Temporal** provides durable execution for any code. A framework
earns its place with standard patterns, built-in tracing, persistence and
human-in-the-loop hooks. Plain code wins for simple flows, full control
and fewer dependencies. Everything in this lesson is under 100 lines of
plain Python, and knowing that is what lets you judge a framework.

## In 20 seconds
- Autonomy spectrum: fixed workflow → router → agent loop → multi-agent. Use the least that works.
- Named patterns: chaining (with gates), routing (with a fallback), parallelization (sectioning, voting), orchestrator-workers, evaluator-optimizer.
- Multi-agent buys focused context and parallelism, and costs tokens, hand-off losses and debuggability.
- Production shape: an explicit state machine, the model consulted only at specific nodes, checkpoints after each step.

## Self-test questions

**Q: When would you use a fixed workflow instead of an agent? Give an example.**
A: When the steps are known in advance and the variation is *inside* each
step, not in which steps happen. Invoice intake is the example: classify,
extract, validate, approve, post. A workflow is cheaper, faster, testable
and auditable. The model is used where judgement is needed (classification,
extraction), and code owns the flow. Reach for an agent only when the path
genuinely depends on what's discovered along the way.

**Q: Single agent vs. multi-agent for a document-processing workflow: argue both sides.**
A: *For one agent (or a workflow):* documents flow through the same steps,
hand-offs lose context, multi-agent multiplies tokens, and one trace is far
easier to debug. *For several agents:* documents are independent and can be
processed in parallel, each document or section may overflow a single
context, and specialists (tables, legal clauses, figures) benefit from
focused instructions and tools. A common answer is a workflow that fans out
per document to a focused worker, which is orchestrator-workers rather than free-form
agents talking to each other.

**Q: Your router's classifier sometimes returns labels that aren't in your list. What do you do?**
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
