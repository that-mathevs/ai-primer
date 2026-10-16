r"""
# Memory and state

Run: `python -m primer.agents.memory`

This lesson builds on the assembled prompt from `primer.agents.context`,
on similarity search from `primer.ml.embeddings.similarity`, and on the
agent loop from `primer.agents.agent_loop`.

## Level 1: The practitioner's guide

**In one sentence.** A model remembers nothing between calls, so "memory"
is the system you build around it: what you keep from a conversation, what
you store for months, what you put back into the prompt, who may see it,
and where the state of a long task lives so that a crash doesn't lose it.

**When you need it.** Any product where the second conversation should
know about the first, or where one conversation runs long enough to
outgrow the window: assistants that remember preferences, agents that
resume a multi-step job, support bots with long threads, anything
multi-tenant. You don't need long-term memory for a one-shot task, and you
don't need a database for a job that finishes in one call. The tells: users
repeating themselves every session; a chat that gets slower, dearer and
vaguer as it goes (this lesson's unmanaged history climbs without end,
while the managed one levels off just under its 300-token budget from about
turn 17); an agent that "forgot" a fact the user corrected an hour ago; or
a job that has to start over because the process died halfway.

**Your options.** From nothing to a full system, each layer added on top
of the previous one:

| Option | What it does | What it guarantees | What it costs | Where it lives |
|---|---|---|---|---|
| The whole history in the prompt | Resends every turn on every call | Nothing is ever forgotten inside one session | Tokens that grow every turn; quality that fades with length | Your prompt |
| Short-term memory with a budget | Keeps the last few turns word for word and folds older ones into a summary that only gets the room they leave | The history never exceeds the budget, however long the session | Older detail (in this lesson exchanges 1 and 2 vanish entirely); a summarizer, extractive or a cheap model call | Your code, per conversation |
| Notes the agent reads on demand | The model writes and reads files or notes outside the window through a tool | Facts and progress survive a reset of the window | Tool calls per read and write, and storage you control | A tool and a directory or table |
| Long-term memory, typed and recalled by similarity | Stores episodic (events), semantic (keyed facts) and procedural (how-to) records with embeddings, partitioned per tenant and user, and recalls the closest few | The right kind of record comes back for the right question; conflicts are superseded, not overwritten; one user can be exported or erased | An embedding per record, a write policy, a partitioned store, export and deletion paths | A store your code owns |
| Task state in a database | Keeps the steps of a job in SQLite (or any database); the model proposes updates, code validates and writes them in transactions | A run survives a crash and resumes at the first unfinished step; no step can be skipped | A schema and a validator per task type | A database |

**How to choose.** Decide separately for the conversation, for facts that
outlive it, and for the state of a job, because they fail differently.

- A chat that runs long: short-term memory with a budget, always. Keep the
  recent turns verbatim (the order number the user just typed) and
  summarize the rest.
- Anything the user would be annoyed to repeat (preferences, corrections,
  how they like a task done): long-term memory, written selectively. A good
  assistant's notebook holds "prefers morning meetings", not "said thanks
  at 3:02pm", and never a password.
- A job with steps that must happen in order and may outlive a process:
  task state in a database, with the model proposing and code holding the
  pen. In this lesson the model's attempt to mark `match` done while
  `fetch_payments` is still open is refused, and after a crash the new
  process resumes at `match` from the file, not from a conversation that
  no longer exists.
- Many customers on one system: partition the store by tenant and user,
  chosen by your authentication layer, never by the query. A query that
  quotes another tenant's secret word for word, with `OR 1=1` appended,
  returns nothing from outside the caller's partition here, because there
  is no path to search it.
- Whatever you pick: the model proposes, your code decides what is
  written. Arrows into the stores never come straight from the model.

**What it costs.** Short-term memory costs the summarizer (free and crude
if extractive; a small model call, with less lost, in production) and the
detail it drops. Long-term memory costs an embedding per record at write
time, a similarity search per recall, and the engineering around it: a
write policy, supersession with provenance, per-user export and deletion.
The last two are not optional where privacy law applies; the GDPR's Article
17 gives a person the right to erasure "without undue delay". Task state
costs a schema, a validator and a transaction per update, and buys runs
that people and monitoring can query. What none of this costs is model
quality: memory is retrieval over your own history, and it lives entirely
in your code.

**What breaks.**

- **Context rot.** Quality drops and cost climbs as a session grows.
  Budget the history, summarize the old turns, promote durable facts to
  long-term memory, or reset with a written hand-off.
- **Remembering everything.** Small talk crowds out facts, and a stored
  secret is read back into every future prompt and every export. Skip
  chatter, refuse anything that looks like a credential, and strip
  sensitive data before a note is written.
- **Silent overwrites.** The user said April, then corrected to July; a
  store that overwrites cannot explain why the agent ever said April.
  Supersede under the same key, recall only the current value, and keep
  the history with its source.
- **Isolation by filter.** A global index filtered after the search is one
  bug away from a leak; in this lesson's figure the other tenant's secret
  scores highest against the adversarial query. Partition first; search
  inside the partition only; test with adversarial queries.
- **Progress kept in the conversation.** It is lost on a crash, and it can
  be summarized, trimmed or misread. Keep it in a database and let code
  enforce the order of steps.
- **Paths that escape.** A memory tool that maps names to files must
  reject `../` and its encodings, or a request for `/memories/../secrets`
  reads outside the store.

**In the wild.** Park et al. (2023), *Generative Agents*, gave twenty-five
simulated agents a memory stream of natural-language observations,
retrieved by relevance, recency and importance and periodically reflected
into higher-level memories. Packer et al. (2023), *MemGPT*, treat the window
as fast memory and external storage as slow memory, paging between them
the way an operating system does. Claude's memory tool is the note-taking
option as a product: the model issues view, create, replace, insert,
delete and rename commands against a `/memories` directory that your
application maps onto storage it controls, its system instruction tells the
model to assume the window may be reset at any moment, and it pairs with
server-side compaction. LangGraph names the same split: thread-scoped
short-term memory held by a checkpointer, long-term memory in a store with
namespaces, and the semantic, episodic and procedural kinds. SQLite's
transactions are the all-or-nothing writes the task store relies on.

**Go deeper.** Level 2 builds each store in plain Python: a short-term
memory whose summary only gets the room the recent turns leave, a long-term
memory with three kinds of record recalled by cosine similarity, the write
policy, supersession and per-tenant partitions with the leak that a filter
would allow drawn as a figure, and a SQLite task store that refuses a
skipped step and resumes after a crash. If you only needed to choose, you
are done.

## Level 2: How it works, from scratch

Every call to a language model starts from a blank page plus the prompt you
send, so your code rebuilds that prompt on each call from stores it keeps in
between.

```mermaid
flowchart LR
  subgraph Prompt["What the model sees on this call"]
    S[System prompt<br/>+ summary of older turns]
    R[Recalled long-term memories]
    M[Recent messages, word for word]
  end
  STM[(Short-term memory<br/>this conversation)] --> S
  STM --> M
  LTM[(Long-term memory<br/>per tenant, per user)] -->|similarity search| R
  DB[(Task state<br/>SQLite)] -->|current step| S
  Model[Model] -->|proposes updates| Code[Your code]
  Code -->|validated writes| LTM
  Code -->|validated writes| DB
```

**Reading it:** the box titled "What the model sees on this call" is all the
model ever sees, rebuilt on every call from the three stores. Every arrow
into a store comes from your code, none from the model: the model proposes
what to remember, and your code decides what is written.

## 1. Short-term memory: the conversation, inside a budget

The first store is the conversation itself. Each call resends the history,
so a long chat costs more and runs slower with every turn, and the model
gets worse at using details buried in the middle, a decline called
**context rot**. **Short-term memory** keeps the history inside a fixed
token budget. The latest messages stay word for word, since they hold what
the user means right now, such as the order number they just typed. Older
ones are folded into a summary that gets only the room the recent messages
leave.

Take six question-and-answer exchanges about invoices, 17 or 18 tokens per
message (210 tokens in all), and a budget of 110 tokens. The last four
messages stay word for word: 18 + 17 + 18 + 17 = 70 tokens, which leaves
110 - 70 = 40 tokens for the summary. The summary keeps the first sentence
of each older message. All eight of those would take 65 tokens, so the
oldest drop out, one at a time, until the rest fit in 40:

```text
summary  : Earlier in this conversation: Question 3 is about invoices; Answer 3 lists the invoice;
           Question 4 is about invoices; Answer 4 lists the invoice              (36 tokens)
messages : Question 5 ..., Answer 5 ..., Question 6 ..., Answer 6 ...   (verbatim, 70 tokens)
```

The prompt carries 36 + 70 = 106 tokens of history, under the budget, and
exchanges 1 and 2 are gone from it. The summary goes into the system
prompt, not into a message of its own, so user and assistant turns still
alternate as the API requires. A production system often has a language
model write the summary instead: an extra call, but less is lost.

![Sending the whole history climbs without end, while the managed history levels off just under its 300-token budget](figures/primer.agents.memory.context_tokens.svg)

**Reading it:** the x-axis is the turn number and the y-axis the tokens of
history sent on that turn. The unmanaged line never stops climbing. The
managed line first passes the 300-token budget at turn 9, drops as older
turns fold into the summary, then climbs back and stays just under 300 from
about turn 17: each new turn folds one more exchange in, and the oldest
drops out of the summary to make room.

When a session has run too long even for that, reset it: write a short
hand-off note and start a fresh conversation from it. Either way something
is dropped, so a fact that must outlast the conversation needs a store of
its own.

**In code:** `ShortTermMemory.context` returns the summary and the recent
messages, and `first_sentences` is the default summarizer.

## 2. Long-term memory: three kinds of record

That store is **long-term memory**, kept outside any one conversation and
recalled into later ones. Its records come in three kinds: **episodic**, a
dated event ("on 18 Sept Alice rejected Globex"); **semantic**, a fact
stored under a key ("Alice's fiscal year starts in April", under
`fiscal_year_start`); and **procedural**, a way of doing something ("to
reconcile, match each invoice to its payment, then list mismatches"). Each
kind answers its own sort of question:

| kind | stored as | typical recall trigger |
|---|---|---|
| episodic | dated event | "what happened with...", "last time..." |
| semantic | keyed fact (`fiscal_year_start`) | any question the fact answers |
| procedural | how-to steps | "how do I...", before starting a known task |

Each record is stored with its embedding. Recall embeds the question and
returns the records closest to it by cosine similarity, optionally of some
kinds only: procedures, say, before the agent starts a task it knows.

![Each question recalls the right kind: the Globex question scores the event highest, the how-to question the procedure](figures/primer.agents.memory.recall_scores.svg)

**Reading it:** each group of bars is one question. Each bar is one of
Alice's memories, coloured by kind, and the tallest is the one
recalled. The Globex question picks the event, the only memory that names
Globex; the how-to question picks the procedure. This is RAG
(retrieval-augmented generation) over the agent's own history.

**In code:** `LongTermMemory.remember` stores a `MemoryRecord` with its
embedding, and `LongTermMemory.recall` returns the closest current records
in a scope.

## 3. The hard parts: what to write, conflicts, isolation, deletion

A store that lasts for months needs rules that a conversation never did.

**What to write.** Everything stored is read back into future prompts and
shown in exports, so a good store holds what the user would be annoyed to
repeat: "prefers morning meetings", not "said thanks at 3:02pm", and never a
bank PIN. The **write policy** skips small talk and refuses anything that
looks like a secret.

**Conflicts.** Alice said her fiscal year starts in April, then corrected it
to July. Both values are stored under the key `fiscal_year_start`. The April
record is marked as **superseded** by the July one, so recall returns only
July, while the key's history keeps both with their sources. That history
explains why the agent once said April.

**Isolation.** A **multi-tenant** system serves many customers (tenants)
whose memories must never mix. Think of one locked filing cabinet per
company, not a shared cabinet with a sign asking people to read only their
own folder. Every memory call names a scope (a tenant and a user) taken from
your authentication layer, never from the query. Storage is **partitioned**
by scope, so a lookup opens the caller's own cabinet and has no way into
another.

```mermaid
flowchart TD
  Q[recall scope=globex/carol, query=...] --> P[Open partition<br/>tenant=globex, user=carol]
  P --> S[Similarity search<br/>inside this partition only]
  S --> R[Results]
  A[(acme/alice partition)] -.-x S
```

**Reading it:** Carol's request opens her own partition and searches only
there. The dotted line from Acme's partition ends in a cross: no path leads
from her search to Acme's records, whatever the query says.

![Acme's secret scores a perfect match to Carol's quoted query, yet it sits outside her partition and is never searched](figures/primer.agents.memory.isolation.svg)

**Reading it:** Carol, at Globex, asks a question that quotes Acme's
confidential memory word for word. Each bar is that question's similarity
to one memory in the whole system. Acme's record scores highest, so a store
that searched everything and filtered the results afterwards would be one
bug away from leaking it. The hatched bars lie outside Carol's partition,
and the real search never scores them.

**Deletion.** Users may need to see, correct or delete what is remembered,
and privacy law such as the GDPR (the EU's General Data Protection
Regulation) can require it. An export of a user's scope shows everything,
superseded facts included, and deleting a user erases their records without
touching their colleagues'.

**In code:** `LongTermMemory.remember` applies the write policy and
supersedes an older record with the same key; `LongTermMemory.export` and
`LongTermMemory.delete_user` show and erase one user's records.

## 4. Task state outside the model

The stores so far hold what the agent knows. A job of several steps also
has a state: which steps are done and which comes next. Kept in the
conversation, that state is lost when the process crashes, and a summary
can trim or garble it.

Picture a checklist on a clipboard. The assistant may suggest ticking a
box, but only the supervisor holds the pen, and if the assistant goes home
sick, the next person starts at the first unticked box. Here the clipboard
is a database, the supervisor is your code and the assistant is the model.
The model proposes an update; your code checks it against the stored state
and writes it in a **transaction**, which completes entirely or not at all.
A job with three steps runs like this:

```text
steps                  : fetch_invoices, fetch_payments, match
model proposes         : {"step": "fetch_invoices", "status": "done", "output": "4 invoices"}   -> written
model proposes         : {"step": "match", "status": "done"}  -> refused: the current step is fetch_payments
process crashes after fetch_payments is done
new process, same file : next_step() -> "match"
```

```mermaid
sequenceDiagram
  participant M as Model
  participant C as Your code
  participant D as SQLite
  M->>C: propose {"step": "fetch_invoices", "status": "done"}
  C->>D: read current step
  D-->>C: fetch_invoices
  C->>D: write status=done (transaction)
  M->>C: propose {"step": "match", "status": "done"}
  C-->>M: refused: current step is fetch_payments
  Note over C,D: crash, restart
  C->>D: next_step?
  D-->>C: match
```

**Reading it:** time runs downward. No arrow joins the model to SQLite:
each proposal stops at your code, which reads the current step before it
writes. After the restart, the answer to "where do we resume?" comes from
the database, not from a conversation that no longer exists.

Because the state sits in a database, people and monitoring can query a run
while it is going, and code enforces rules, such as no skipped steps, that a
prompt could only ask for.

**In code:** `TaskStateStore.apply` validates a proposal before writing it,
raising `InvalidUpdate` for a bad status or a skipped step, and
`TaskStateStore.next_step` finds the first unfinished step.

## In 20 seconds
- A language model remembers nothing between calls; memory is what your code puts back into the prompt.
- Short-term memory keeps the recent turns word for word and folds older ones into a summary, inside a token budget.
- Long-term memory stores events (episodic), facts (semantic) and how-to steps (procedural), and recalls them by similarity.
- Write selectively, never store secrets, and supersede a changed fact instead of overwriting it, keeping each value's source.
- Partition storage by tenant and user, taken from authentication, never from the query.
- Keep a job's progress in a database: the model proposes updates, and code checks and writes them.

## Self-test questions

**Q: How would you design long-term memory for a multi-tenant agent platform?**
A: Every read and write takes a scope (tenant, user) from the authenticated
session, and storage is partitioned by it: separate namespaces, collections
or row-level security, never a global index filtered after the search.
Store typed records (episodic, semantic with keys, procedural) with
embeddings, source and time. Apply a write policy (durable, non-secret),
supersede conflicting facts instead of overwriting, retrieve the top few by
similarity within scope, and support export and deletion per user. Test
isolation with adversarial queries.

**Q: A long support chat gets worse and more expensive over time. Why, and what helps?**
A: Every call re-sends the whole history, so cost rises, and models use
details in the middle of long contexts less reliably. Keep a budget: recent
turns verbatim, older ones summarized, important facts promoted to
long-term memory, or reset with a written hand-off.

**Q: The user corrects a fact the agent remembered. What should happen?**
A: Store the new value under the same key, mark the old one as superseded
by the new (don't delete it silently), and recall only current records.
Keep the source of each so the change is explainable.

**Q: Why keep task state in a database when the model can "remember" progress in the conversation?**
A: The conversation is lost on a crash, and it can be summarized, trimmed or
misread. A database is authoritative, survives restarts, can be queried by
people and monitoring, and lets code enforce rules (no skipping steps)
that a prompt can only request.

## The papers behind this lesson

- **Park et al., *Generative Agents: Interactive Simulacra of Human Behavior* (2023).**
  https://arxiv.org/abs/2304.03442. It introduced a *memory stream* of
  observations retrieved by relevance, recency and importance, plus periodic
  reflection into higher-level memories, a template for long-term agent memory.
- **Packer et al., *MemGPT: Towards LLMs as Operating Systems* (2023).**
  https://arxiv.org/abs/2310.08560. It treats the context window like RAM and
  external storage like disk, with the agent paging information in and out,
  which is the short-term/long-term split made explicit.

## Further reading
- Lilian Weng, *LLM Powered Autonomous Agents* (memory section): https://lilianweng.github.io/posts/2023-06-23-agent/
- LangGraph docs (short- and long-term memory, persistence): https://langchain-ai.github.io/langgraph/
- SQLite, transactions: https://www.sqlite.org/lang_transaction.html
- GDPR, right to erasure (Art. 17): https://gdpr-info.eu/art-17-gdpr/
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from primer.agents.llm import estimate_tokens
from primer.common.embedder import ConceptEmbedder


@dataclass
class ShortTermMemory:
    """The conversation so far, kept inside a token budget."""

    budget_tokens: int
    keep_last: int = 4
    messages: list[dict[str, Any]] = field(default_factory=list)
    # How older messages are compressed. The default keeps the first sentence of
    # each (cheap and deterministic); in production this is often an LLM call.
    summarize: Callable[[list[dict[str, Any]]], str] = field(default=lambda msgs: first_sentences(msgs))

    def add(self, role: str, text: str) -> None:
        self.messages.append({"role": role, "content": text})

    def tokens(self) -> int:
        return sum(estimate_tokens(m["content"]) for m in self.messages)

    def context(self) -> tuple[str, list[dict[str, Any]]]:
        """(summary for the system prompt, messages to send).

        Under budget: everything, no summary. Over budget: fold all but the last
        `keep_last` messages into a summary. The summary goes in the system
        prompt rather than as a message, so user/assistant turns still alternate.

        The summary only gets the room the recent messages leave. Without that
        cap, a summary that gains a sentence per message would itself outgrow
        the budget in a long conversation. When it doesn't fit, the oldest
        folded messages drop out first. If even the recent messages overflow
        the budget, there is no room left and the summary is empty.
        """
        if self.tokens() <= self.budget_tokens:
            return "", list(self.messages)
        old, recent = self.messages[: -self.keep_last], self.messages[-self.keep_last :]
        room = self.budget_tokens - sum(estimate_tokens(m["content"]) for m in recent)
        while old:
            summary = "Earlier in this conversation: " + self.summarize(old)
            if estimate_tokens(summary) <= room:
                return summary, list(recent)
            old = old[1:]  # the oldest detail is the cheapest one to lose
        return "", list(recent)


def first_sentences(messages: list[dict[str, Any]]) -> str:
    """Extractive summary: the first sentence of each message, joined with '; '."""
    return "; ".join(m["content"].split(". ")[0].rstrip(".") for m in messages)


# ---------------------------------------------------------------------------
# Long-term memory
# ---------------------------------------------------------------------------

KINDS = ("episodic", "semantic", "procedural")

_SMALL_TALK = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|great|cool|bye)\b", re.I)
# Deliberately broad: a false alarm costs one unsaved note; a stored secret leaks
# into every future prompt and every data export.
_SECRET = re.compile(r"password|passcode|api[ _-]?key|secret|token|\b\d{3}-\d{2}-\d{4}\b", re.I)


@dataclass(frozen=True)
class Scope:
    """Who a memory belongs to. Every read and write names one; there is no global view."""

    tenant_id: str
    user_id: str

    def __post_init__(self) -> None:
        if not self.tenant_id or not self.user_id:
            raise ValueError("a memory scope needs both a tenant_id and a user_id")


@dataclass
class MemoryRecord:
    id: str
    scope: Scope
    kind: str
    content: str
    key: str | None
    source: str
    seq: int  # write order; stands in for a timestamp so examples are deterministic
    superseded_by: str | None = None


@dataclass
class WriteResult:
    stored: bool
    reason: str
    record: MemoryRecord | None = None


def worth_remembering(content: str) -> tuple[bool, str]:
    """The write policy: store durable, safe facts; skip chatter and refuse secrets."""
    if _SECRET.search(content):
        return False, "looks like a secret: credentials never go into memory"
    if _SMALL_TALK.match(content) and len(content.split()) <= 5:
        return False, "small talk: nothing durable to remember"
    return True, "stored"


class LongTermMemory:
    """Memories that outlive a conversation, partitioned by tenant then user."""

    def __init__(self) -> None:
        # tenant -> user -> records. Partitioning (not filtering) is the isolation:
        # a lookup starts from the caller's own partition and can't reach another.
        self._store: dict[str, dict[str, list[MemoryRecord]]] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._seq = 0
        self.embedder = ConceptEmbedder()

    def _records(self, scope: Scope) -> list[MemoryRecord]:
        return self._store.setdefault(scope.tenant_id, {}).setdefault(scope.user_id, [])

    def remember(self, scope: Scope, kind: str, content: str, key: str | None = None, source: str = "conversation") -> WriteResult:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        ok, reason = worth_remembering(content)
        if not ok:
            return WriteResult(False, reason)
        self._seq += 1
        record = MemoryRecord(f"m{self._seq}", scope, kind, content, key, source, self._seq)
        # A keyed fact replaces the older value for the same key, but the old
        # record stays (marked) so the change is explainable.
        if key is not None:
            for old in self._records(scope):
                if old.key == key and old.superseded_by is None:
                    old.superseded_by = record.id
        self._records(scope).append(record)
        self._vectors[record.id] = self.embedder.encode(content)
        return WriteResult(True, reason, record)

    def recall(self, scope: Scope, query: str, k: int = 3, kinds: tuple[str, ...] = KINDS) -> list[MemoryRecord]:
        """The k current memories in this scope most similar to the query."""
        candidates = [r for r in self._records(scope) if r.superseded_by is None and r.kind in kinds]
        q = self.embedder.encode(query)
        return sorted(candidates, key=lambda r: -float(self._vectors[r.id] @ q))[:k]

    def history(self, scope: Scope, key: str) -> list[MemoryRecord]:
        """Every value a keyed fact has had, oldest first, including superseded ones."""
        return [r for r in self._records(scope) if r.key == key]

    def forget(self, scope: Scope, record_id: str) -> None:
        """Delete one memory. Only ids inside the caller's own scope can be found at all."""
        records = self._records(scope)
        for i, r in enumerate(records):
            if r.id == record_id:
                del records[i]
                self._vectors.pop(record_id, None)
                return
        raise KeyError(f"no memory {record_id} in this scope")

    def export(self, scope: Scope) -> list[dict[str, Any]]:
        """Everything stored about this user, for them to see and correct (right of access)."""
        return [
            {"id": r.id, "kind": r.kind, "content": r.content, "key": r.key, "source": r.source, "current": r.superseded_by is None}
            for r in self._records(scope)
        ]

    def delete_user(self, scope: Scope) -> int:
        """Erase every memory of one user (right to erasure). Returns how many were deleted."""
        records = self._store.get(scope.tenant_id, {}).pop(scope.user_id, [])
        for r in records:
            self._vectors.pop(r.id, None)
        return len(records)


# ---------------------------------------------------------------------------
# Task state outside the model (SQLite)
# ---------------------------------------------------------------------------

STATUSES = ("pending", "running", "done", "failed")


class InvalidUpdate(ValueError):
    """A state change the model proposed that the code refuses to write."""


class TaskStateStore:
    """The authoritative record of a multi-step task, in a database, not in the prompt.

    The model reads it and *proposes* updates as JSON; `apply` checks each
    proposal against the rules and only then writes it. Because the state is on
    disk, a crashed run resumes from the first unfinished step.
    """

    def __init__(self, path: str | Path):
        self.db = sqlite3.connect(str(path))
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, goal TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS steps (
                task_id TEXT NOT NULL, idx INTEGER NOT NULL, name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending', output TEXT,
                PRIMARY KEY (task_id, idx)
            );
            """
        )

    def create_task(self, task_id: str, goal: str, steps: list[str]) -> None:
        with self.db:  # one transaction: the task and all its steps, or nothing
            self.db.execute("INSERT INTO tasks VALUES (?, ?)", (task_id, goal))
            self.db.executemany("INSERT INTO steps (task_id, idx, name) VALUES (?, ?, ?)", [(task_id, i, s) for i, s in enumerate(steps)])

    def steps(self, task_id: str) -> list[tuple[str, str, str | None]]:
        return self.db.execute("SELECT name, status, output FROM steps WHERE task_id = ? ORDER BY idx", (task_id,)).fetchall()

    def next_step(self, task_id: str) -> str | None:
        """The first step that isn't done, or None when the task is finished."""
        return next((name for name, status, _ in self.steps(task_id) if status != "done"), None)

    def apply(self, task_id: str, proposal: dict[str, Any]) -> None:
        """Validate a proposed update from the model, then write it."""
        step, status = proposal.get("step"), proposal.get("status")
        if status not in STATUSES:
            raise InvalidUpdate(f"status must be one of {STATUSES}, got {status!r}")
        current = self.next_step(task_id)
        if step != current:
            raise InvalidUpdate(f"{step} cannot change yet: the current step is {current}")
        with self.db:
            self.db.execute(
                "UPDATE steps SET status = ?, output = ? WHERE task_id = ? AND name = ?",
                (status, proposal.get("output"), task_id, step),
            )


# ---------------------------------------------------------------------------
# Figures and walkthrough
# ---------------------------------------------------------------------------


def _demo_store() -> tuple[LongTermMemory, Scope, Scope]:
    mem = LongTermMemory()
    alice, carol = Scope("acme", "alice"), Scope("globex", "carol")
    mem.remember(alice, "semantic", "Alice's fiscal year starts in April.", key="fiscal_year_start")
    mem.remember(alice, "episodic", "On 2026-09-18 Alice rejected the vendor Globex over late invoices.")
    mem.remember(alice, "procedural", "To reconcile invoices, match each vendor invoice to its payment, then list mismatches.")
    mem.remember(alice, "semantic", "Acme plans to acquire Initech in Q4.", key="acquisition")
    mem.remember(carol, "semantic", "Globex prefers invoices in euros.", key="currency")
    return mem, alice, carol


def figures() -> dict:
    """Plots computed from this lesson's own code (matplotlib imported here)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = {}

    managed = ShortTermMemory(budget_tokens=300, keep_last=6)
    raw, kept = [], []
    for i in range(1, 41):
        managed.add("user", f"Question {i} is about invoices. Please include the vendor name and amount.")
        managed.add("assistant", f"Answer {i} lists the invoice. It shows vendor, amount and due date.")
        summary, msgs = managed.context()
        raw.append(managed.tokens())
        kept.append((estimate_tokens(summary) if summary else 0) + sum(estimate_tokens(m["content"]) for m in msgs))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, 41), raw, label="send the whole history")
    ax.plot(range(1, 41), kept, label="recent turns + first-sentence summary")
    ax.axhline(300, ls="--", color="gray", label="budget before summarizing (300)")
    ax.set(xlabel="turn", ylabel="tokens of history sent", title="Short-term memory under a budget")
    ax.legend()
    figs["context_tokens"] = fig

    mem, alice, carol = _demo_store()
    records = [r for r in mem._records(alice) if r.key != "acquisition"]
    questions = ["what happened with Globex?", "how do I reconcile invoices?"]
    colors = {"episodic": "#e0a458", "semantic": "#5b8fd6", "procedural": "#6bb36b"}
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.25
    for qi, q in enumerate(questions):
        qv = mem.embedder.encode(q)
        for ri, r in enumerate(records):
            ax.bar(qi + (ri - 1) * width, float(mem._vectors[r.id] @ qv), width, color=colors[r.kind],
                   label=r.kind if qi == 0 else None)
    ax.set_xticks(range(len(questions)), questions)
    ax.set(ylabel="cosine similarity", title="Recall: which of Alice's memories each question finds")
    ax.legend(title="memory kind")
    figs["recall_scores"] = fig

    crafted = "Acme plans to acquire Initech in Q4."
    everything = [r for users in mem._store.values() for recs in users.values() for r in recs]
    qv = mem.embedder.encode(crafted)
    scores = [float(mem._vectors[r.id] @ qv) for r in everything]
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (r, sc) in enumerate(zip(everything, scores)):
        visible = r.scope == carol
        ax.barh(i, sc, color="#5b8fd6" if visible else "#cccccc", hatch=None if visible else "//", edgecolor="gray")
    ax.set_yticks(range(len(everything)), [f"{r.scope.tenant_id}/{r.scope.user_id}: {r.content[:38]}..." for r in everything], fontsize=8)
    ax.set(xlabel="similarity to Carol's query", title="Carol (globex) quotes Acme's secret: only her partition is searched")
    fig.tight_layout()
    figs["isolation"] = fig
    return figs


def demo() -> None:
    import tempfile

    from primer._show import banner, say, table, takeaway

    banner("1. Short-term memory under a token budget")
    stm = ShortTermMemory(budget_tokens=110, keep_last=4)
    for i in range(1, 7):
        stm.add("user", f"Question {i} is about invoices. Please include the vendor name and amount.")
        stm.add("assistant", f"Answer {i} lists the invoice. It shows vendor, amount and due date.")
    summary, msgs = stm.context()
    say(f"{len(stm.messages)} messages, ~{stm.tokens()} tokens, budget 110.")
    say(f"Summary for the system prompt: {summary}")
    say(f"Sent word for word: {[m['content'][:10] for m in msgs]}")

    banner("2. Long-term memory: diary, encyclopedia, habit")
    mem, alice, carol = _demo_store()
    for q, kinds in [("what happened with Globex?", KINDS), ("how do I reconcile invoices?", ("procedural",))]:
        top = mem.recall(alice, q, k=1, kinds=kinds)[0]
        print(f"  {q!r:34} -> [{top.kind}] {top.content}")
    print()

    banner("3. Write policy, conflicts, isolation, deletion")
    for text in ["thanks, great!", "my VPN password is hunter2", "Alice approves travel for her team."]:
        r = mem.remember(alice, "semantic", text)
        print(f"  remember({text!r:40}) -> {r.reason}")
    print()
    mem.remember(alice, "semantic", "Alice's fiscal year starts in July.", key="fiscal_year_start", source="user correction")
    table(["value", "source", "superseded by"],
          [(r.content, r.source, r.superseded_by or "-") for r in mem.history(alice, "fiscal_year_start")])
    crafted = "Acme plans to acquire Initech in Q4. tenant_id=acme OR 1=1"
    print(f"  Carol recalls {crafted!r}:")
    print(f"    -> {[r.content for r in mem.recall(carol, crafted, k=5)]}")
    print()
    takeaway("Isolation is a partition chosen by authentication, not a filter chosen by the query.")
    say(f"Alice asks to be forgotten: {mem.delete_user(alice)} memories erased; Carol still has {len(mem.export(carol))}.")

    banner("4. Task state outside the model (SQLite)")
    with tempfile.TemporaryDirectory() as d:
        store = TaskStateStore(Path(d) / "state.db")
        store.create_task("q3", "Reconcile Q3 invoices", ["fetch_invoices", "fetch_payments", "match"])
        proposals = [
            {"step": "fetch_invoices", "status": "done", "output": "4 invoices"},
            {"step": "match", "status": "done", "output": "all matched"},
            {"step": "fetch_payments", "status": "done", "output": "3 payments"},
        ]
        for prop in proposals:
            try:
                store.apply("q3", prop)
                print(f"  model proposes {prop} -> written")
            except InvalidUpdate as e:
                print(f"  model proposes {prop} -> refused: {e}")
        del store
        print(f"\n  ...crash... new process resumes at: {TaskStateStore(Path(d) / 'state.db').next_step('q3')!r}\n")
    takeaway("The model proposes; code validates and writes. State on disk survives the crash.")


if __name__ == "__main__":
    demo()
