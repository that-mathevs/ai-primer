r"""
# Operating embeddings: model changes, jargon, and finding what broke

Run: `python -m primer.ml.embeddings.operations`

New to vectors or Σ? `primer.notation` builds them from zero. This lesson
builds on `primer.ml.embeddings.similarity` and `primer.ml.embeddings.contrastive`.

## The everyday picture

Two mapmakers each draw a map of the same city with their own grid. On one
map, square (3, 7) is the train station; on the other, (3, 7) is a park. Both
maps are fine, but you can't read a position off one map and look it up on
the other.

Every embedding model is its own mapmaker. Upgrade the model and every
document has to be re-plotted on the new map, and until that's done, the old
map and the new one must never be mixed. That's the first of three
operational truths this lesson makes concrete:

1. **Changing models means re-embedding everything**, and doing it without
   downtime or a silent quality drop takes a plan.
2. **General models don't speak your company's jargon**, so measure on your
   own questions and adapt when needed.
3. **When a retrieval-augmented system answers wrongly, find out which half
   failed**: retrieval (the right page was never found) or generation (it was
   found and misused).

## Every model has its own space

**Tiny worked example:** the word "vpn" embedded by two versions of the toy
model (`primer.common.embedder`) with the same 64 dimensions: their cosine
is about 0.06, essentially unrelated, though it's the same word. Now take 12
real questions with known answers (the "golden set" in `primer.common.corpus`)
over 20 documents. Search v1 documents with v1 queries and the right answer
is in the top 3 for 11 of 12 questions (**0.92**). Search the *same* v1
documents with queries from the new v2 model and it drops to **0.25**, about
what random ranking would give (3 of 20 documents shown, so 15%), and
**nothing crashes**.

$$
\text{hit@}k = \frac{1}{|Q|} \sum_{q \in Q} \big[\, \text{rel}(q) \cap \text{top}_k(q) \neq \varnothing \,\big]
$$

**Symbols**

| Symbol | Meaning here | Example |
|---|---|---|
| Q | the golden set of questions | 12 questions |
| \|Q\| | how many questions | 12 |
| q ∈ Q | each question in turn | |
| rel(q) | the documents that truly answer q | {it-004} |
| top_k(q) | the k documents search returned | 3 documents |
| ∩, ≠ ∅ | "they share at least one document" | |
| [ … ] | 1 if true, 0 if false | |
| hit@k | the share of questions with a right answer in the top k | 0 to 1 |

**In words:** the share of golden questions for which at least one right
document appears among the top k results. (With one relevant document per
question, as here, this equals recall@k.)

**On the example:** 11 hits out of 12 = 0.92 with matching models; 3 out of 12 =
0.25 with mixed models.

![Recall@3 with matching and mixed models](figures/primer.ml.embeddings.operations.cross_model.svg)

**Reading it:** each bar is recall@3 on the golden set. The first two bars
use one model for both documents and queries, v1 then v2: both work. The
third bar searches v1 documents with v2 queries: recall falls to the dashed
line, which is what random ranking would score. A model with a *different*
number of dimensions would at least crash (you can't dot a 128-number vector
with a 64-number one); a same-size model fails silently, which is worse.

**In code:** `golden_recall` embeds the documents with one model and the
questions with another and returns hit@k. `VectorIndex` is a flat index tied
to one model version: `VectorIndex.search` embeds a text query, and
`VectorIndex.search_vector` takes a vector that is already made.

**Why it matters:** "we upgraded the embedding model and search got weird"
is a classic incident. The cause is almost always documents and queries
embedded by different models, typically because only new documents were
re-embedded.

## Migrating without a bad day: blue/green

**Everyday picture:** building a new bridge next to the old one. Traffic
keeps using the old bridge while the new one is built and inspected. When
the new bridge passes inspection, traffic is switched over, and the old
bridge stays standing for a while in case something turns up.

```mermaid
stateDiagram-v2
  [*] --> Live_v1
  Live_v1 --> DualWrite: start v2 (empty index, new docs go to both)
  DualWrite --> Backfilled: backfill (re-embed old docs in batches)
  Backfilled --> Compared: shadow compare (recall on golden set)
  Compared --> Live_v2: cutover, only if v2 at least as good
  Compared --> Live_v1: refused, v2 is worse
  Live_v2 --> Live_v1: rollback (v1 index kept)
  Live_v2 --> [*]: retire v1 after a soak period
```

**Reading it:** follow the states from the top. Users are served by v1 the
whole time until cutover. Dual-writing from the very first step means
documents added mid-migration land in both indexes, so the new one never
falls behind. The **shadow comparison** is the gate: the switch only happens
if v2 is at least as good on your own golden set (`BlueGreenMigration`
refuses to cut over without it). The old index is kept after cutover, so
rollback is flipping one pointer, not a multi-hour rebuild.

The "live alias" is a pointer: the search service asks for "the live
index", and cutover or rollback just repoints it. Many vector databases
support aliases for exactly this.

**In code:** each arrow of the diagram is one method: `BlueGreenMigration.start`,
`BlueGreenMigration.add` (the dual write), `BlueGreenMigration.backfill`,
`BlueGreenMigration.shadow_compare`, `BlueGreenMigration.cutover` and
`BlueGreenMigration.rollback`.

**Tiny worked example: what will re-embedding cost?** 50 million documents of
about 500 tokens each, an embedding throughput of 1 million tokens per second
across your workers, and a price of $0.02 per million tokens (all three are
inputs you replace with your own numbers):

$$
\text{hours} = \frac{n \cdot t}{r \cdot 3600}, \qquad
\text{dollars} = \frac{n \cdot t}{10^6} \cdot p
$$

**Symbols**

| Symbol | Meaning here | Example |
|---|---|---|
| n | number of documents (or chunks) | 50,000,000 |
| t | average tokens per document | 500 |
| n · t | total tokens to embed | 25,000,000,000 |
| r | throughput, tokens per second | 1,000,000 |
| 3600 | seconds per hour | |
| p | price per million tokens | $0.02 |

**In words:** total tokens divided by throughput gives the time; total
tokens in millions times the price gives the cost.

**On the example:** 25 × 10⁹ / (10⁶ × 3600) = **6.94 hours**; 25,000 × $0.02 =
**$500**.

![Time and cost to re-embed a corpus](figures/primer.ml.embeddings.operations.reembed.svg)

**Reading it:** the horizontal axis is corpus size (log scale); the left
panel shows hours at three throughputs, the right panel shows dollars at
three prices. Both grow in straight lines on these log axes: ten times the
documents, ten times the time and money. The money is usually modest; the
time, the rate limits and the double storage during the migration are what
need planning.

**In code:** `reembed_estimate` computes the tokens, hours and dollars for
your own n, t, r and p.

**Why it matters:** re-embedding is routine: new models, fine-tunes,
chunking changes and bug fixes all require it. Versioned indexes, dual
writes, a golden-set gate and a rollback path turn it from a risky event
into a boring one.

## Domain mismatch: fluent, but not in your jargon

**Everyday picture:** a new hire who speaks perfect English but doesn't yet
know that "AP" means accounts payable or that "T&E" means travel and
expenses. General embedding models are trained mostly on web text; your
acronyms and product names are new words to them.

**Tiny worked example:** four real-sounding employee questions in company
jargon: "ap aging report", "hotspot from a client site", "sso lockout",
"t-and-e submission deadline". The general toy model puts the right
document first for **1 of 4**. A version that has learned the four jargon
words (`DomainTunedEmbedder`, standing in for a model fine-tuned on
company pairs) gets **4 of 4**.

![Rank of the right document for each jargon question](figures/primer.ml.embeddings.operations.jargon.svg)

**Reading it:** each pair of bars is one jargon question; the height is where
the right document ranked (1 is best, shorter is better). The general model
buries three of the four answers; the tuned model puts every one first. The
one the general model gets right, "sso lockout", is saved by a word it does
know ("lockout").

**In code:** `rank_of_first_relevant` finds where the right document lands
for one question under one model: the height of each bar.

```mermaid
flowchart LR
  L["Search & support logs"] --> F["Keep resolved sessions<br/>(the clicked doc answered it)"]
  F --> N["Normalize and merge<br/>repeated questions"]
  N --> G["Golden set: question → relevant docs<br/>(a few hundred is plenty)"]
  G --> T["Test several models on YOUR set"]
  T --> D{Good enough?}
  D -->|no| FT["Fine-tune on domain pairs,<br/>add hybrid search"]
  D -->|yes| S[Ship, and keep the set for regressions]
  FT --> T
```

**Reading it:** the evaluation set comes from real usage, not invention.
Resolved sessions give you (question, answer document) pairs for free;
`build_eval_set` keeps only resolved ones and merges repeats. Test
candidate models on that set, and when none is good enough, fine-tune on
domain pairs (`primer.ml.embeddings.contrastive`) and add keyword search,
which matches jargon exactly (`primer.ml.embeddings.retrieval`).

**Why it matters:** public leaderboards measure public data. The only
number that predicts your system's quality is recall on your own
questions.

## Measure retrieval separately from generation

**Everyday picture:** an open-book exam. A wrong answer has one of two
causes: the right page wasn't in the book you brought (**retrieval
failure**), or it was, and you misread it (**generation failure**). Studying
harder fixes the second, not the first.

**Tiny worked example:** three answered questions, checked by hand.

| Retrieved (top 3) | Truly relevant | Answer used | Verdict |
|---|---|---|---|
| it-002, fin-006, fin-001 | it-004 | it-002 | retrieval failure: it-004 was never found |
| hr-002, hr-001, hr-005 | hr-001 | hr-002 | generation failure: found but not used |
| it-001, it-002 | it-001 | it-001 | correct |

```mermaid
flowchart TD
  A[Wrong answer] --> R{Was a relevant document<br/>in the retrieved top k?}
  R -->|no| RF[Retrieval failure:<br/>fix chunking, hybrid search,<br/>reranking, the embedding model]
  R -->|yes| G{Did the answer use it?}
  G -->|no| GF[Generation failure:<br/>fix the prompt, context order,<br/>fewer distracting chunks]
  G -->|yes| OK[Check the grader:<br/>the answer may be fine]
```

**Reading it:** one question splits every failure in two. If the right
document wasn't retrieved, no prompt change can help, so work on retrieval.
If it was retrieved and ignored, the problem is downstream, in the prompt or
the context. Only the retrieval half can be measured without a model, which
is why recall@k on a golden set is the first number to track.

![What went wrong, by question, as k changes](figures/primer.ml.embeddings.operations.triage.svg)

**Reading it:** each bar is the 12 golden questions, answered by a toy
generator that always uses the top document, with k documents retrieved.
Green is correct, red is a retrieval failure, orange is a generation
failure. Retrieving more (larger k) turns some retrieval failures into
generation failures: the right page is now in the book, but the reader still
opened the wrong one. Watch the error-code question ("what does ERR-4012
mean"): its answer ranks only fourth, so it's a retrieval failure until k = 5,
and even then the top document is the wrong one. Dense vectors blur exact
identifiers; keyword search would put that page first.

**In code:** `triage` gives the verdict for one answered question, following
the flowchart above; `triage_golden_set` runs every golden question through
retrieval and a toy generator that answers from the top document.

**Why it matters:** teams burn weeks tuning prompts for failures that were
retrieval all along. Triage first, then fix the half that's broken.

## In 20 seconds
- Every embedding model has its own vector space; never mix documents and
  queries from different models. Same-size models fail silently.
- Migrate blue/green: new index, dual-write, backfill, compare on a golden
  set, cut over only if better, keep the old index for rollback.
- Re-embedding cost is n × tokens: estimate hours and dollars before you start.
- General models miss company jargon; build an eval set from resolved logs,
  and fine-tune or add keyword search when needed.
- Triage RAG failures into retrieval vs. generation before fixing anything.

## Self-test questions

**Q: You need to switch embedding models on a 50-million-document index. Plan the migration.**
Create a versioned index for the new model and dual-write new documents to
both. Backfill by re-embedding existing documents in restartable batches
(estimate: 50M × 500 tokens = 25B tokens; at 1M tokens/s that's about 7 hours).
Compare recall on a golden set in shadow; cut over the live alias only if the
new model is at least as good; keep the old index for fast rollback, then
retire it.

**Q: Why can't you compare vectors from two different embedding models?**
Each model defines its own coordinate system. Even with the same number of
dimensions, the same text lands in unrelated places, so similarity between
the two spaces is meaningless.

**Q: A general embedding model performs poorly on a client's internal documents. What do you do?**
Build a small eval set from real queries and the documents that resolved
them, test several models on it, add hybrid (keyword + dense) search for
exact terms, and fine-tune an embedding model on domain pairs with hard
negatives if needed.

**Q: Your RAG system gives confident wrong answers. What's your first diagnostic step?**
Split retrieval from generation: for failing questions, check whether a
relevant document was in the retrieved top k. If not, it's retrieval (fix
search); if yes, it's generation (fix prompt and context). Track recall@k on
a labeled set continuously.

## The papers behind this lesson

- **Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models* (2021)**: https://arxiv.org/abs/2104.08663.
  Showed that retrieval models trained on one domain often lose badly on others, which is why you evaluate on your own data.
- **Muennighoff et al., *MTEB: Massive Text Embedding Benchmark* (2022)**: https://arxiv.org/abs/2210.07316.
  Compared embedding models across many tasks and found no single model wins everywhere.

## Further reading
- Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation* (2023): https://arxiv.org/abs/2309.15217
- Martin Fowler, *BlueGreenDeployment*: https://martinfowler.com/bliki/BlueGreenDeployment.html
- MTEB leaderboard: https://huggingface.co/spaces/mteb/leaderboard
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from primer._show import banner, say, table, takeaway
from primer.common.corpus import DOCS, LABELED_QUERIES, Doc
from primer.common.embedder import ConceptEmbedder, _unit_vector
from primer.common.text import tokenize

# Model versions. Each `name` is a different model, so each defines its own
# vector space, even when two have the same number of dimensions.
CURRENT_MODEL = ConceptEmbedder()  # the repo's standard toy model, "concept-v1", 64 dims
SAME_SIZE_NEW_MODEL = ConceptEmbedder(name="concept-v2")  # also 64 dims: mixing it with v1 fails *silently*
BETTER_MODEL = ConceptEmbedder(dim=128, name="concept-v2-large")
WORSE_MODEL = ConceptEmbedder(dim=8, name="concept-v2-mini")


def doc_text(d: Doc) -> str:
    return f"{d.title}. {d.text}"


# ---------------------------------------------------------------------------
# 1. A minimal versioned vector index, and golden-set recall
# ---------------------------------------------------------------------------


@dataclass
class VectorIndex:
    """Exact (flat) index for one model version. Vectors are only comparable within it."""

    model: ConceptEmbedder
    ids: list[str] = field(default_factory=list)
    vectors: list[np.ndarray] = field(default_factory=list)

    @property
    def version(self) -> str:
        return self.model.name

    def add(self, docs: list[Doc]) -> None:
        for d, v in zip(docs, self.model.encode([doc_text(d) for d in docs])):
            self.ids.append(d.id)
            self.vectors.append(v)

    def search_vector(self, q: np.ndarray, k: int) -> list[str]:
        scores = np.stack(self.vectors) @ q
        return [self.ids[i] for i in np.argsort(-scores, kind="stable")[:k]]

    def search(self, query: str, k: int) -> list[str]:
        return self.search_vector(self.model.encode(query), k)


def golden_recall(query_model: ConceptEmbedder, doc_model: ConceptEmbedder, k: int, queries=LABELED_QUERIES) -> float:
    """Share of golden queries with at least one relevant doc in the top k.

    Documents are embedded with `doc_model` and queries with `query_model`.
    Passing two different models reproduces the classic migration bug:
    querying an old index with a new model.
    """
    idx = VectorIndex(doc_model)
    idx.add(DOCS)
    hits = [bool(rel & set(idx.search_vector(query_model.encode(q), k))) for q, rel in queries]
    return float(np.mean(hits))


# ---------------------------------------------------------------------------
# 2. Blue/green migration to a new embedding model
# ---------------------------------------------------------------------------


class BlueGreenMigration:
    """Move search from one embedding model to another without a bad day.

    Steps: start (create the new index and begin dual-writing), backfill
    (re-embed every existing document), shadow_compare (measure both on the
    golden set), cutover (flip the live alias only if the new one is at least
    as good), rollback (flip it back; the old index is kept until retired).
    """

    def __init__(self, current: ConceptEmbedder, docs: list[Doc] = DOCS):
        self.docs = list(docs)
        self.indexes: dict[str, VectorIndex] = {current.name: VectorIndex(current)}
        self.indexes[current.name].add(self.docs)
        self.live_version = current.name
        self.previous_version: str | None = None
        self.candidate: str | None = None
        self.comparison: dict[str, float] = {}
        self.log: list[str] = [f"live index {current.name} serving {len(self.docs)} docs"]

    def start(self, new: ConceptEmbedder) -> None:
        self.indexes[new.name] = VectorIndex(new)
        self.candidate = new.name
        self.comparison = {}
        self.log.append(f"created empty index {new.name}; dual-writing new documents to both")

    def add(self, doc: Doc) -> None:
        """New documents go to every index, so the candidate never falls behind while backfilling."""
        self.docs.append(doc)
        for idx in self.indexes.values():
            idx.add([doc])
        self.log.append(f"dual-wrote {doc.id} to {sorted(self.indexes)}")

    def backfill(self, batch_size: int = 8) -> None:
        idx = self.indexes[self.candidate]
        missing = [d for d in self.docs if d.id not in set(idx.ids)]
        for start in range(0, len(missing), batch_size):  # batches: restartable, rate-limit friendly
            idx.add(missing[start : start + batch_size])
        self.log.append(f"backfilled {len(missing)} docs into {self.candidate}")

    def shadow_compare(self, k: int = 3) -> dict[str, float]:
        for name in (self.live_version, self.candidate):
            m = self.indexes[name].model
            self.comparison[name] = golden_recall(m, m, k)
        self.log.append("shadow recall@%d: %s" % (k, ", ".join(f"{n} {r:.3f}" for n, r in self.comparison.items())))
        return self.comparison

    def cutover(self, min_gain: float = 0.0) -> bool:
        if self.candidate not in self.comparison:
            self.log.append("cutover refused: no shadow comparison yet")
            return False
        old, new = self.comparison[self.live_version], self.comparison[self.candidate]
        if new < old + min_gain:
            self.log.append(f"cutover refused: {self.candidate} recall {new:.3f} < live {old:.3f}")
            return False
        self.previous_version, self.live_version = self.live_version, self.candidate
        self.log.append(f"cutover: live alias -> {self.live_version} (old index kept for rollback)")
        return True

    def rollback(self) -> None:
        if self.previous_version:
            self.live_version, self.previous_version = self.previous_version, self.live_version
            self.log.append(f"rollback: live alias -> {self.live_version}")


def reembed_estimate(n_docs: int, avg_tokens: int, tokens_per_second: float, dollars_per_million_tokens: float) -> dict[str, float]:
    """Back-of-envelope time and money to re-embed a corpus. Prices vary; pass your own."""
    tokens = n_docs * avg_tokens
    return {"tokens": tokens, "hours": tokens / tokens_per_second / 3600, "dollars": tokens / 1e6 * dollars_per_million_tokens}


# ---------------------------------------------------------------------------
# 3. Domain mismatch
# ---------------------------------------------------------------------------

# Jargon a general model has never seen, and the concept each word means here.
JARGON: dict[str, str] = {"ap": "invoice", "hotspot": "remote_access", "sso": "credential", "t-and-e": "money_back"}

JARGON_QUERIES: list[tuple[str, set[str]]] = [
    ("ap aging report", {"fin-005"}),  # accounts payable: vendor invoices
    ("hotspot from a client site", {"it-003"}),  # remote access
    ("sso lockout", {"it-001"}),  # single sign-on: your login
    ("t-and-e submission deadline", {"fin-004"}),  # travel and expenses
]


class DomainTunedEmbedder(ConceptEmbedder):
    """Stand-in for a model fine-tuned on company pairs: it has learned the jargon.

    In real life this comes from contrastive fine-tuning on (query, document)
    pairs from your own logs (`primer.ml.embeddings.contrastive`). Here we
    simply teach the toy embedder what four jargon words mean.
    """

    def __init__(self, jargon: dict[str, str] = JARGON, **kw):
        super().__init__(**{"name": "concept-v1-tuned", **kw})
        self.jargon = jargon

    def _token_vector(self, tok: str) -> np.ndarray:
        concept = self.jargon.get(tok)
        if concept is None:
            return super()._token_vector(tok)
        return _unit_vector(f"{self.name}/concept/{concept}", self.dim) + self.word_weight * _unit_vector(f"{self.name}/word/{tok}", self.dim)


def build_eval_set(log: list[dict]) -> list[tuple[str, set[str]]]:
    """Turn a search/support log into (query, relevant doc ids) cases.

    Keep only sessions where the user's problem was resolved (the clicked
    document really answered it); merge the same question asked twice.
    """
    cases: dict[str, set[str]] = {}
    for row in log:
        if not row["resolved"]:
            continue
        key = " ".join(tokenize(row["query"], drop_stopwords=False))  # "Travel policy?" == "travel policy"
        cases.setdefault(key, set()).add(row["clicked"])
    return list(cases.items())


# ---------------------------------------------------------------------------
# 4. Measure retrieval separately from generation
# ---------------------------------------------------------------------------


def triage(retrieved: list[str], relevant: set[str], answer_used: str) -> str:
    """Which half of a RAG system failed?

    If no relevant document was retrieved, no prompt can fix the answer: it's
    a retrieval failure. If one was retrieved but the answer relied on a
    different document, the generator misused good context.
    """
    if not relevant & set(retrieved):
        return "retrieval failure"
    if answer_used not in relevant:
        return "generation failure"
    return "correct"


def triage_golden_set(model: ConceptEmbedder, k: int = 3, queries=LABELED_QUERIES) -> list[tuple[str, str]]:
    """Run every golden query through retrieval and a toy generator that answers from the top document."""
    idx = VectorIndex(model)
    idx.add(DOCS)
    out = []
    for q, rel in queries:
        top = idx.search(q, k)
        out.append((q, triage(top, rel, answer_used=top[0])))
    return out



def rank_of_first_relevant(model: ConceptEmbedder, query: str, relevant: set[str]) -> int:
    """1-based position of the first relevant document in the full ranking."""
    idx = VectorIndex(model)
    idx.add(DOCS)
    ranking = idx.search(query, len(DOCS))
    return next(i + 1 for i, d in enumerate(ranking) if d in relevant)


# ---------------------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------------------


def figures() -> dict:
    """Plots computed from this module's own functions. Keys match the docstring's image names."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = {}

    # cross-model
    labels = ["v1 docs,\nv1 queries", "v2 docs,\nv2 queries", "v1 docs,\nv2 queries"]
    vals = [golden_recall(CURRENT_MODEL, CURRENT_MODEL, 3), golden_recall(SAME_SIZE_NEW_MODEL, SAME_SIZE_NEW_MODEL, 3), golden_recall(SAME_SIZE_NEW_MODEL, CURRENT_MODEL, 3)]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, vals, color=["C0", "C2", "C3"])
    ax.bar_label(bars, fmt="%.2f")
    ax.axhline(3 / len(DOCS), ls="--", color="0.5", label="random ranking (3 of 20)")
    ax.set(ylabel="recall@3 on the golden set", ylim=(0, 1.1), title="Mixing two models' vectors fails silently")
    ax.legend()
    figs["cross_model"] = fig

    # reembed
    ns = np.logspace(5, 9, 30)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for r in (1e5, 1e6, 1e7):
        a1.plot(ns, [reembed_estimate(int(n), 500, r, 0.02)["hours"] for n in ns], label=f"{r:,.0f} tokens/s")
    for p in (0.01, 0.02, 0.13):
        a2.plot(ns, [reembed_estimate(int(n), 500, 1e6, p)["dollars"] for n in ns], label=f"${p} per million tokens")
    for ax, ylab in ((a1, "hours"), (a2, "dollars")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(5e7, ls=":", color="0.5")
        ax.set(xlabel="documents (500 tokens each, log scale)", ylabel=f"{ylab} (log scale)")
        ax.legend(fontsize=8)
    a1.set_title("Time to re-embed")
    a2.set_title("Cost to re-embed")
    figs["reembed"] = fig

    # jargon
    tuned = DomainTunedEmbedder()
    qs = [q for q, _ in JARGON_QUERIES]
    x = np.arange(len(qs))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, [rank_of_first_relevant(CURRENT_MODEL, q, r) for q, r in JARGON_QUERIES], 0.4, label="general model")
    ax.bar(x + 0.2, [rank_of_first_relevant(tuned, q, r) for q, r in JARGON_QUERIES], 0.4, label="tuned on the jargon")
    ax.set_xticks(x, qs, fontsize=8)
    ax.set(ylabel="rank of the right document (1 is best)", title="Company jargon: general vs. tuned model")
    ax.legend()
    figs["jargon"] = fig

    # triage
    ks = [1, 3, 5]
    colours = {"correct": "C2", "generation failure": "C1", "retrieval failure": "C3"}
    fig, ax = plt.subplots(figsize=(6, 4))
    bottoms = np.zeros(len(ks))
    counts = {v: [] for v in colours}
    for k in ks:
        verdicts = [v for _, v in triage_golden_set(CURRENT_MODEL, k)]
        for v in colours:
            counts[v].append(verdicts.count(v))
    for v, c in colours.items():
        ax.bar([str(k) for k in ks], counts[v], bottom=bottoms, color=c, label=v)
        bottoms += np.array(counts[v])
    ax.set(xlabel="documents retrieved (k)", ylabel="golden questions", title="Which half failed? (generator uses the top document)")
    ax.legend(fontsize=8)
    figs["triage"] = fig

    for f in figs.values():
        f.tight_layout()
    return figs


# ---------------------------------------------------------------------------
# 6. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. Every model has its own space")
    a, b = ConceptEmbedder(name="concept-v1").encode("vpn"), ConceptEmbedder(name="concept-v2").encode("vpn")
    say(f"Cosine between 'vpn' from v1 and 'vpn' from v2 (both 64 dims): {float(a @ b):.3f}.")
    table(
        ["documents", "queries", "recall@3"],
        [
            ("v1", "v1", golden_recall(CURRENT_MODEL, CURRENT_MODEL, 3)),
            ("v2", "v2", golden_recall(SAME_SIZE_NEW_MODEL, SAME_SIZE_NEW_MODEL, 3)),
            ("v1", "v2 (the bug)", golden_recall(SAME_SIZE_NEW_MODEL, CURRENT_MODEL, 3)),
        ],
        floatfmt=".3f",
    )
    takeaway("Same number of dimensions, different space. Mixing them doesn't crash; it just quietly stops working.")

    banner("2. A blue/green migration, step by step")
    for candidate in (BETTER_MODEL, WORSE_MODEL):
        m = BlueGreenMigration(CURRENT_MODEL)
        m.start(candidate)
        m.add(Doc("it-999", "Wi-Fi guest network", "The guest Wi-Fi password rotates every Monday.", "IT", "2026-09-01"))
        m.backfill()
        m.shadow_compare(k=3)
        m.cutover()
        print(f"  candidate {candidate.name}:")
        for line in m.log:
            print("   -", line)
        print()
    est = reembed_estimate(50_000_000, 500, 1_000_000, 0.02)
    say(f"Re-embedding 50M docs × 500 tokens = {est['tokens']:,} tokens: {est['hours']:.2f} hours at 1M tokens/s, ${est['dollars']:,.0f} at $0.02 per million.")

    banner("3. Domain jargon")
    tuned = DomainTunedEmbedder()
    table(
        ["jargon question", "rank (general)", "rank (tuned)"],
        [(q, rank_of_first_relevant(CURRENT_MODEL, q, r), rank_of_first_relevant(tuned, q, r)) for q, r in JARGON_QUERIES],
    )
    log = [
        {"query": "VPN error 4012?", "clicked": "it-004", "resolved": True},
        {"query": "vpn error 4012", "clicked": "it-003", "resolved": True},
        {"query": "printer", "clicked": "it-009", "resolved": False},
    ]
    say(f"Eval set built from a 3-line log: {build_eval_set(log)}")

    banner("4. Triage: retrieval failure or generation failure?")
    table(["golden question", "verdict (k = 3)"], triage_golden_set(CURRENT_MODEL, 3))
    takeaway("If the right document wasn't retrieved, no prompt can fix it. Measure retrieval on its own first.")


if __name__ == "__main__":
    demo()
