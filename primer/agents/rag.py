r"""
# Retrieval-augmented generation (RAG), end to end

Run: `python -m primer.agents.rag`

## The everyday picture

An open-book exam with a librarian. Before you answer, the librarian fetches
the few pages most likely to contain the answer; you answer from those
pages and write down which page each fact came from. If the pages don't
contain the answer, you say so instead of guessing.

**Retrieval-augmented generation** is that arrangement for a language
model. The model's training knowledge is frozen and doesn't include your
company's documents, so before each answer the system *retrieves* relevant
passages from your documents and puts them in the prompt, and the model
*generates* an answer from them, with citations. It's how you give a model
knowledge that changes, or that must be cited, without retraining it.

```mermaid
flowchart LR
  subgraph ING["Ingestion, ahead of time"]
    D[Documents] --> P[Parse] --> C[Chunk + metadata] --> E[Embed + keyword index]
  end
  E --> IX[(Index)]
  subgraph QRY["Query time, per question"]
    Q[Question] --> F[Filter by permissions<br/>and date] --> R[Retrieve: hybrid] --> RR[Rerank] --> AC[Assemble context] --> G[Generate + cite] --> V[Verify citations]
  end
  IX --> F
```

**Reading it:** the top row runs once per document, whenever documents
change; the bottom row runs for every question. Everything the answer can
contain has to survive every box, left to right: a table mangled at *Parse*
or a passage missing at *Retrieve* can't be fixed by a better prompt at
*Generate*. That's why most RAG quality work happens in the first boxes, not
the last.

**In code:** `RAGIndex` is the top row: it chunks every document and builds
the keyword and dense indexes. `answer` is the bottom row, from filtered
retrieval to verified citations, and returns a `RAGResult`.

The rest of this lesson walks the boxes in order, then covers the upgrades
and how to debug a wrong answer.

## Parsing: where quality dies first

**Everyday picture.** Photocopy a newspaper page and read it straight
across: you get the first line of one column, then the first line of the
next story, and nothing makes sense. Text extracted from PDFs has the same
problems: running headers and page numbers mixed into the text, words
broken across lines, and tables flattened into a row of numbers with no
columns.

**Worked example.** `MESSY_PDF` is two pages of a travel policy as a naive
extractor returns it. `clean_extracted_text` repairs it in four steps:

| Damage | Before | After |
|---|---|---|
| Header repeated on every page | `Northwind Ltd  Travel Policy 2026 … CONFIDENTIAL` | removed (it appears on 2 of 2 pages) |
| Page-number footer | `Page 1 of 2` | removed |
| Word hyphenated across a line break | `all em-` / `ployees travelling` | `all employees travelling` |
| Table flattened | `L1-L3 150 60 L4-L6 200 75` | kept as rows, then one sentence per row |

The table is the dangerous one. Flattened, "200" floats free of its grade.
`table_rows_as_sentences` turns each row into a self-contained sentence,
`Grade L4-L6: hotel cap 200, meal cap 75.`, so any row can be retrieved on
its own and still says what its numbers mean.

```mermaid
flowchart LR
  RAW[Raw extracted text] --> H[Drop lines repeated<br/>on most pages]
  H --> N[Drop page numbers]
  N --> J[Rejoin hyphenated words,<br/>unwrap paragraphs]
  J --> T{Table rows?}
  T -->|yes| S[One sentence per row,<br/>headers repeated]
  T -->|no| P[Paragraphs]
  S --> OUT[Clean text for chunking]
  P --> OUT
```

**Reading it:** each box removes one kind of extraction damage. The diamond
is the important decision: prose and tables need different handling, because
a table's meaning lives in its columns. Real pipelines add layout-aware
parsers and **OCR** (optical character recognition: reading text from an
image of a page) for scanned documents, with a quality check on the output.

**Why it matters.** Enterprise documents are scanned, multi-column, full of
tables that span pages. If parsing garbles the numbers, no retriever or
model can recover them, and the failure looks like a *model* error.

**In code:** `extract_table` finds a table by its header row in the cleaned
text and returns each row as a dict, ready for `table_rows_as_sentences`.

## Chunking, with metadata that travels

**Everyday picture.** Index cards. Each card holds one idea, small enough to
match a question precisely, and in its corner a label: which document,
which date, who may read it.

**Worked example.** With a 15-word limit, `it-004` splits into two passages:

| Passage id | Text | Updated | Readers |
|---|---|---|---|
| `it-004#0` | ERR-4012 means the VPN tunnel could not be established, usually because the client is out of date. | 2026-04-10 | everyone |
| `it-004#1` | Update AnyConnect to version 5.1 or later and reboot. | 2026-04-10 | everyone |

(The limit is tiny because the toy documents are tiny; real systems use a few
hundred tokens.) Split on the document's structure (sentences, paragraphs,
sections), not at fixed character counts that cut a sentence in half.
`primer.ml.embeddings.retrieval` compares chunking strategies in depth.

**Why it matters.** Small chunks match precisely but can lose context (the
second card doesn't say *which* error it fixes; see contextual retrieval
below). The metadata is what makes permission and date filters possible
later.

**In code:** `chunk_document` splits a document on sentence boundaries into
`Passage`s, and each `Passage` carries its id, title, department, date and
readers.

## Retrieval: keyword and meaning, fused

**Everyday picture.** Two librarians: one searches the catalogue for your
exact words, the other understands what you *mean* even when you use
different words. You take the books both of them rank highly.

* **Keyword search (BM25)** scores passages by the question's words,
  giving more weight to rare words and less to long passages. It nails
  exact identifiers like `ERR-4012` and misses synonyms.
* **Dense search** compares **embeddings** (vectors where closeness means
  similar meaning; see `primer.ml.embeddings`). It finds "scam message in my
  inbox" → the phishing guide with no shared words, but blurs exact codes.
* **Hybrid search** runs both and merges the two rankings with **reciprocal
  rank fusion (RRF)**, which only needs the ranks, never the scores. BM25
  scores and cosine similarities live on different scales, so adding them
  would be meaningless.

$$
\text{RRF}(d) = \sum_{i} \frac{1}{k + \text{rank}_i(d)}
$$

**Symbols**

| Symbol | Meaning here | Worked example |
|---|---|---|
| $d$ | one passage | it-004#0 |
| $i$ | one of the rankings being fused (keyword, dense) | 2 rankings |
| $\text{rank}_i(d)$ | position of $d$ in ranking $i$ (1 = best); rankings that miss $d$ contribute nothing | 1st, 3rd |
| $k$ | a damping constant; 60 is the usual choice | 60 |
| $\sum_i$ | add up over the rankings | |

**In words:** each ranking gives a passage a vote worth one over sixty-plus
its position, and the votes are added.

**On the worked example:** a passage ranked 1st by dense search and 3rd by
keyword search scores 1/61 + 1/63 = 0.0323. A passage ranked 1st by one
list and absent from the other scores only 1/61 = 0.0164. Passages that both
methods agree on rise to the top.

**In Python:**

```python
k = 60
# 1st by dense, 3rd by keyword: Σ_i 1/(k + rank_i(d))
round(1 / (k + 1) + 1 / (k + 3), 4)  # → 0.0323
# 1st in one list, absent from the other
round(1 / (k + 1), 4)  # → 0.0164
```

![recall@k for keyword, dense, hybrid and hybrid plus reranking](figures/primer.agents.rag.recall.svg)

**Reading it:** the x-axis is how many passages you keep (k); the y-axis is
**recall@k**, the share of the 12 labelled questions whose correct document
is among those k. On this small set many questions reuse the documents'
own words, so keyword search is already strong. Dense search plateaus at
92% because it never finds `ERR-4012`. Hybrid reaches 100% by k = 2 but only
75% at k = 1. Reranking the hybrid shortlist lifts the top-1 result to 92%.
Run this measurement on *your* questions before choosing a method.

$$
\text{recall@}k = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{1}\bigl[\text{a relevant document is in the top } k \text{ for } q\bigr]
$$

**Symbols**

| Symbol | Meaning here |
|---|---|
| $Q$ | the labelled questions; $\lvert Q\rvert$ is how many (12 here) |
| $q$ | one question |
| $\mathbb{1}[\ldots]$ | 1 if true, 0 if not |

**In words:** the share of questions for which the right document made it
into the top k.

**On the worked example:** dense search at k = 3 finds the right document for
11 of 12 questions (all but `ERR-4012`): 11/12 = 0.92.

**In Python:**

```python
# 𝟙[...] for each of the 12 questions; ERR-4012 is the 0
found = [1] * 11 + [0]
# (1/|Q|) Σ over q in Q
round(sum(found) / len(found), 2)  # → 0.92
```

**In code:** `RAGIndex.retrieve` ranks the allowed passages with
`primer.ml.embeddings.retrieval.BM25`, with dense similarity, or with both
fused by `primer.ml.embeddings.retrieval.reciprocal_rank_fusion`, depending
on the mode you ask for. `recall_curve` measures recall@k for each method.

## Reranking: read the question and each passage together

**Everyday picture.** The librarians bring back twenty books quickly; an
expert then reads your question next to each book and picks the best three.

A **reranker** (usually a **cross-encoder**: a model that reads the question
and one passage *together* and outputs a relevance score) is far more
precise than comparing precomputed vectors, and far too slow to run over
every passage. So retrieval casts a wide, cheap net (here 8 candidates) and
the reranker keeps the best few (here 3). This module reuses the toy
cross-encoder from `primer.ml.embeddings.retrieval`.

**In code:** `RAGIndex.rerank` scores each (question, passage) pair with
`primer.ml.embeddings.retrieval.CrossEncoder` and keeps the best few.

## Assemble, generate, cite, verify

**Worked example.** "What does ERR-4012 mean?" becomes this context (each
source escaped and tagged with its id; the question last; see
`primer.agents.context`):

```text
<sources>
<source id="it-004#1" title="Error ERR-4012: VPN tunnel failed" updated="2026-04-10">Update AnyConnect to version 5.1 or later and reboot.</source>
<source id="it-004#0" title="Error ERR-4012: VPN tunnel failed" updated="2026-04-10">ERR-4012 means the VPN tunnel could not be established, usually because the client is out of date.</source>
<source id="it-002#2" title="Password policy" updated="2025-11-20">This policy applies to all employees and contractors.</source>
</sources>
<question>What does ERR-4012 mean?</question>
```

and the answer is *"ERR-4012 means the VPN tunnel could not be established,
usually because the client is out of date. [it-004#0]"*. `verify_citations`
then checks that every cited id was really in the context and that the
cited passage supports the claim before it; a question the sources don't
cover gets "I don't know based on the provided sources."

```mermaid
sequenceDiagram
  participant U as User
  participant A as RAG service
  participant I as Index
  participant M as Model
  U->>A: What does ERR-4012 mean?
  A->>I: hybrid search, filtered to what this user may read
  I-->>A: 8 candidate passages
  A->>A: rerank, keep 3
  A->>M: rules + tagged sources + question
  M-->>A: answer with [it-004#0]
  A->>A: verify: id was in context, passage supports claim
  A-->>U: answer + clickable citation
```

**Reading it:** time runs downward. Note the order: the permission filter
runs at the index, before any passage is fetched; the model only ever sees
three passages; and nothing reaches the user until the citations check out.
Citations are what let a person verify an answer in seconds, which is most
of what makes people trust the system.

**In code:** `assemble_context` builds the tagged sources and the question.
`generate` sends them to the model with the answer-from-sources rules, and
`grounded_policy` is the offline stand-in model that honours them. `answer`
chains retrieve, rerank, assemble, generate and `verify_citations`.

## Permission-aware retrieval: filter before, never after

**Everyday picture.** A librarian checks your library card *before*
fetching from the restricted archive. Checking it after you've read the
document is pointless.

Each passage carries its **access-control list** (ACL: the groups allowed to
read it), copied from the source system (SharePoint, Google Drive, a wiki).
`RAGIndex.retrieve` drops every passage the user can't read *before*
ranking, so restricted text can't reach the context, the model, or the
answer.

**Worked example.** "What is the Q3 revenue forecast?" (the forecast
document is readable only by `finance` and `exec`):

| Who asks | How it's filtered | Answer |
|---|---|---|
| finance user | before retrieval | "Q3 revenue is forecast at 41 million dollars… [fin-006#0]" |
| anyone else | before retrieval | "I don't know based on the provided sources." |
| anyone else | **after** generation (citation removed) | "Q3 revenue is forecast at **41 million dollars**…" |

```mermaid
sequenceDiagram
  participant U as Employee (not finance)
  participant A as RAG service
  participant M as Model
  rect rgb(250, 225, 225)
  Note over A,M: Wrong: filter after generation
  A->>M: sources include the restricted forecast
  M-->>A: "…41 million dollars [fin-006#0]"
  A->>A: strip the restricted citation
  A-->>U: "…41 million dollars" (leaked)
  end
  rect rgb(225, 240, 225)
  Note over A,M: Right: filter before retrieval
  A->>A: drop passages the user can't read
  A->>M: sources without the forecast
  M-->>A: "I don't know…"
  A-->>U: nothing restricted
  end
```

**Reading it:** in the red box the model saw the restricted passage, so the
number is already in its words, and deleting the citation marker afterwards
leaves the fact behind. In the green box the passage never left the index.
Permission changes in the source system must also sync to the index
quickly, or a revoked user keeps access until the next re-index.

**In code:** `post_generation_filter_answer` is the wrong way, kept so the
leak can be shown: it retrieves for every group, generates, then strips the
restricted citation markers.

## Retrieval upgrades

Each upgrade below is a working function, and each fixes a failure you can
reproduce in `demo()`:

| Upgrade | Everyday picture | Before → after (top 3) |
|---|---|---|
| **Query rewriting** (`rewrite_query`) | Repeating the earlier topic when you ask a follow-up | "what if it fails?" after a VPN question: printer guide → VPN setup and ERR-4012 |
| **HyDE** (`hyde_search`) | Telling the librarian "I'm looking for a page that says something like…" | "a weird message asking for my bank details": nothing relevant → phishing guide first |
| **Multi-query** (`multi_query_search`) | Asking three librarians in three different phrasings | "two step login setup": no MFA guide → MFA guide in top 3 |
| **Metadata filter** (`updated_after`) | Ignoring the out-of-date binder on the shelf | superseded 2023 travel policy first → gone |
| **Contextual retrieval** (`RAGIndex(contextual=True)`) | Writing the chapter title on every index card | "how do I fix ERR-4012": fix sentence missing → found |
| **Parent-child** (`retrieve_parents`) | Find the sentence, hand over the whole page | a matching sentence → its full document |

**HyDE** (Hypothetical Document Embeddings) deserves a picture, because it
sounds backwards: you ask a model to *invent* an answer, then search with
it.

```mermaid
flowchart LR
  Q["Question:<br/>'weird message asking<br/>for my bank details'"] --> M[Model drafts a<br/>plausible answer]
  M --> H["Draft: 'This sounds like a phishing<br/>email… report it in Outlook'"]
  H --> E[Embed the draft]
  E --> S[Dense search]
  S --> R[Phishing guide, rank 1]
```

**Reading it:** questions and answers are written differently. The user's
words ("weird", "bank details") appear nowhere in the phishing guide, but
the drafted answer uses the vocabulary answers use ("phishing", "report",
"suspicious"), so its embedding lands next to the real passage. The draft
may be wrong in its details; that's fine, because it's only used to search
and is never shown to the user.

**Contextual retrieval** fixes the "second index card" problem from the
chunking section: before indexing, each passage gets a short line saying
where it comes from. Here that's just the title and department; Anthropic's
version asks a model to write one sentence situating the chunk in its
document.

![Rank of the passage containing the fix, with and without a context line](figures/primer.agents.rag.contextual.svg)

**Reading it:** three ways of asking how to fix ERR-4012; each pair of bars
is the rank of the passage "Update AnyConnect to version 5.1 or later and
reboot." (shorter is better, and 21 means not in the top 20). Without
context the passage never mentions ERR-4012, so it ranks poorly or not at
all. With the title prefixed, it's in the top three every time.

## Agentic RAG: let the model decide when and what to search

**Everyday picture.** A researcher rather than a librarian: reads the
question, decides what to look up, reads the results, and searches again if
something's missing.

Plain RAG searches exactly once, with the user's words. In **agentic RAG**
search is a tool the model calls: zero times for small talk, several times
for a multi-part question, with its own reformulated queries.

**Worked example.** "What is the per-diem for meals and how do I upload
expense receipts?" produces two searches ("What is the per-diem for meals",
"how do I upload expense receipts") and an answer citing both `fin-002#2`
and `fin-004#0`. "thanks, that's all!" produces no search at all.

```mermaid
sequenceDiagram
  participant U as User
  participant M as Model
  participant S as search_kb tool
  U->>M: per-diem for meals AND how to upload receipts?
  M->>S: search_kb("per-diem for meals")
  S-->>M: fin-002 passages
  M->>S: search_kb("how do I upload expense receipts")
  S-->>M: fin-004 passages
  M-->>U: both answers, each with its citation
```

**Reading it:** the model, not a fixed pipeline, decides how many searches
to run and with which words. That handles multi-part and vague questions
better, at the cost of more model calls, more latency, and a loop that needs
the usual step budget (`primer.agents.agent_loop`). The search tool here also
applies the freshness filter itself, so the agent can't forget it.

**In code:** `agentic_answer` offers the model one search tool and runs the
loop: each search retrieves, reranks and returns tagged passages, until the
model answers or the step limit is hit.

## GraphRAG: questions about connections

**Everyday picture.** A detective's corkboard: photos of people and places
joined by string, each string labelled with the note that established it.

Some questions aren't answered by any single passage: "what depends on
two-factor authentication?" or "what are the main themes across these
contracts?". **GraphRAG** first extracts **entities** (named things:
systems, people, products) and the relationships between them into a graph,
then answers by walking it, or by summarizing groups of connected entities.
`build_graph` does the smallest honest version: two entities are linked when
one sentence mentions both, and each link remembers which document said so.

```mermaid
flowchart LR
  TF((two-factor)) ---|it-006| VPN((VPN))
  TF ---|it-006| EM((email))
  TF ---|it-006| AA((authenticator app))
  TF ---|it-003| AC((AnyConnect))
  TF ---|it-003| SC((software center))
  TF ---|hr-003| LT((laptop))
  AC ---|it-003| SC
```

**Reading it:** each circle is an entity found in the knowledge base; each
line is a sentence that mentioned both ends, labelled with its source
document. "What depends on two-factor?" is answered by reading the
neighbours of one circle, with citations for free. Real GraphRAG uses a
model to extract typed relations ("requires", "replaces") and to write a
summary for each cluster, which is what makes broad, whole-collection
questions answerable.

**In code:** `communities` groups the graph from `build_graph` into
connected clusters of entities, the "themes" a real GraphRAG system would
summarize.

## Debugging a confident wrong answer

**Everyday picture.** A student gets an exam question wrong. Either the
librarian brought the wrong book, or the student misread the right one. The
fixes are completely different, so find out which before changing anything.

**Worked example.** `debug_wrong_answers` runs the 12 labelled questions and
classifies each: **retrieval miss** (no relevant document in the top 3: no
prompt can fix it), **generation miss** (the right document was there but the
answer didn't use it), or ok. With dense-only retrieval, "what does ERR-4012
mean" is a retrieval miss and recall@3 is 0.92; switching to hybrid fixes it
(recall@3 = 1.00). What's left are generation misses. One of them,
"per-diem for meals when traveling", is answered from the superseded 2023
policy: a *data* problem that a date filter fixes.

```mermaid
flowchart TD
  W[Confident wrong answer] --> R{Right passage in<br/>the top k?}
  R -->|no| P{Is it in the<br/>index at all?}
  P -->|no| FIXI[Fix ingestion:<br/>parsing, chunking, permissions sync]
  P -->|yes| FIXR[Fix retrieval:<br/>hybrid, rerank, rewrite, HyDE, filters]
  R -->|yes| C{Was it in the<br/>final context?}
  C -->|no| FIXC[Fix assembly:<br/>reranker, budget, top_n]
  C -->|yes| FIXG[Fix generation:<br/>instructions, citations, stale sources]
```

**Reading it:** start at the top and answer each question with a
measurement, not a guess. Most teams jump straight to the bottom-right box
and edit the prompt; the diagram says to rule out the three earlier failure
points first, because they're more common and a prompt can't fix them.

![Where the wrong answers come from, by retrieval method](figures/primer.agents.rag.diagnosis.svg)

**Reading it:** each bar is one retrieval method over the 12 labelled
questions, split into ok (green), generation misses (orange) and retrieval
misses (red). Dense search has a retrieval miss (the error code); hybrid
removes it. The orange that remains is where to look next, and it's not
retrieval.

## In 20 seconds
- RAG retrieves relevant passages at question time and has the model answer
  from them with citations: knowledge that changes or must be cited, with no
  retraining.
- Quality is decided early: parsing (tables!), chunking with metadata, and
  hybrid retrieval with reranking.
- Filter by permissions before retrieval, never after generation.
- Upgrades: query rewriting, HyDE, multi-query, date filters, contextual
  retrieval, parent-child; agentic RAG for multi-part questions; GraphRAG
  for questions about connections.
- Debug by measuring retrieval (recall@k) first, then generation.

## Self-test questions

**Your RAG system gives confident wrong answers. Debug it step by step.**
Take the failing questions (and build a labelled set if you don't have one).
First, is the right passage in the index at all? If not, it's ingestion:
parsing, chunking or permission sync. Second, is it in the top k? Measure
recall@k; if not, fix retrieval: hybrid search, a reranker, query rewriting
or HyDE, metadata filters, contextual chunks, domain-tuned embeddings.
Third, did it survive into the final context? If not, fix reranking or the
context budget. Only then look at generation: instructions to answer only
from sources and to decline otherwise, citation checks, stale or
contradictory sources. Add every case to the golden set.

**Why does hybrid search beat pure vector search on enterprise data?**
Enterprise questions are full of exact identifiers (error codes, product
names, ticket numbers) that embeddings blur, and full of paraphrases that
keyword search misses. Hybrid gets both, and RRF merges the rankings without
having to reconcile incompatible score scales.

**How do you make RAG respect document permissions?**
Copy each document's access-control list onto every chunk at ingestion,
filter candidates by the user's groups before ranking, and sync permission
changes from the source system promptly. Never filter after generation: the
model has already used the restricted text.

**When would you use agentic RAG instead of a single retrieval step?**
For multi-part or vague questions, and when the model needs to judge whether
results are good enough and search again. It costs more calls and latency,
so keep single-shot retrieval for simple lookups.

**RAG or fine-tuning for a company knowledge assistant?**
RAG for knowledge: it updates by re-indexing, cites sources, and respects
permissions. Fine-tuning teaches behaviour and format, not facts that
change. Most systems are RAG plus a good prompt (see
`primer.ml.training_stages`).

## The papers behind this lesson

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks* (2020), https://arxiv.org/abs/2005.11401. Coined the term and
  showed that pairing a generator with a retriever over a document index
  beats a model relying on its weights alone for knowledge-heavy questions.
  [annotated companion](../../papers/rag.html)
- Gao, Ma, Lin & Callan, *Precise Zero-Shot Dense Retrieval without
  Relevance Labels* (2022), https://arxiv.org/abs/2212.10496. Introduced
  HyDE: search with an embedding of a model-written hypothetical answer.
  [annotated companion](../../papers/hyde.html)
- Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused
  Summarization* (2024), https://arxiv.org/abs/2404.16130. Builds an entity
  graph and community summaries so that whole-collection questions can be
  answered.

## Further reading
- Anthropic, *Introducing Contextual Retrieval*: https://www.anthropic.com/news/contextual-retrieval
- Anthropic citations docs: https://docs.claude.com/en/docs/build-with-claude/citations
- Microsoft GraphRAG documentation: https://microsoft.github.io/graphrag/
- RAGAS documentation (RAG evaluation): https://docs.ragas.io/
- The retrieval lesson in this repo, with BM25, RRF, cross-encoders and chunking in depth: `primer.ml.embeddings.retrieval`
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from primer._show import banner, say, table, takeaway
from primer.agents.context import xml_wrap
from primer.agents.guardrails import claim_support
from primer.agents.llm import ScriptedLLM, ToolCall, last_user_text, tool_result_block, tool_results
from primer.common.corpus import DOCS, DOCS_BY_ID, LABELED_QUERIES, Doc
from primer.common.embedder import ConceptEmbedder
from primer.common.text import tokenize
from primer.ml.embeddings.retrieval import BM25, CrossEncoder, ideas, reciprocal_rank_fusion

# ===========================================================================
# 1. INGESTION: parse, chunk, index
# ===========================================================================

# What a PDF looks like after naive text extraction: a header and footer on
# every page, words hyphenated across line breaks, and a table whose columns
# survive only as runs of spaces.
MESSY_PDF = (
    "Northwind Ltd  Travel Policy 2026                    CONFIDENTIAL\n"
    "Hotel and meal limits apply to all em-\n"
    "ployees travelling on company busi-\n"
    "ness. Book through the travel portal.\n"
    "Grade    Hotel cap    Meal cap\n"
    "L1-L3    150          60\n"
    "L4-L6    200          75\n"
    "Page 1 of 2\n"
    "\f"
    "Northwind Ltd  Travel Policy 2026                    CONFIDENTIAL\n"
    "Economy class is required for flights un-\n"
    "der six hours.\n"
    "Page 2 of 2\n"
)

_PAGE_NUMBER = re.compile(r"^page \d+ of \d+$", re.I)
_COLUMNS = re.compile(r"\s{2,}")  # two or more spaces separate table columns


def clean_extracted_text(raw: str) -> str:
    """Repair the usual damage of PDF text extraction.

    1. Drop lines repeated on two or more pages (running headers/footers).
    2. Drop "Page n of m" footers.
    3. Rejoin words hyphenated across line breaks ("em-" + "ployees").
    4. Unwrap prose lines into paragraphs; keep table rows on their own lines.
    """
    pages = [p.strip("\n").split("\n") for p in raw.split("\f")]
    seen_on: dict[str, int] = {}
    for lines in pages:
        for line in set(l.strip() for l in lines):
            seen_on[line] = seen_on.get(line, 0) + 1
    kept = [l.strip() for lines in pages for l in lines
            if l.strip() and seen_on[l.strip()] < 2 and not _PAGE_NUMBER.match(l.strip())]

    out: list[str] = []
    for line in kept:
        is_table_row = len(_COLUMNS.split(line)) > 1
        if out and not is_table_row and not out[-1].endswith("\n"):
            prev = out.pop()
            if prev.endswith("-") and line[:1].islower():
                out.append(prev[:-1] + line)  # hyphenated word: join with no space
            else:
                out.append(prev + " " + line)
        else:
            out.append(line + ("\n" if is_table_row else ""))
    return "\n".join(s.rstrip("\n") for s in out)


def extract_table(raw: str, header: list[str]) -> list[dict[str, str]]:
    """Find the table whose header row matches `header` and return its rows as dicts."""
    lines = [l.strip() for l in raw.split("\n")]
    for i, line in enumerate(lines):
        if _COLUMNS.split(line) == header:
            rows = []
            for row in lines[i + 1 :]:
                cells = _COLUMNS.split(row)
                if len(cells) != len(header):
                    break
                rows.append(dict(zip(header, cells)))
            return rows
    return []


def table_rows_as_sentences(rows: list[dict[str, str]]) -> list[str]:
    """One self-contained sentence per row, so each row can be retrieved on its own."""
    out = []
    for row in rows:
        key, *rest = list(row)
        out.append(f"{key} {row[key]}: " + ", ".join(f"{h.lower()} {row[h]}" for h in rest) + ".")
    return out


@dataclass(frozen=True)
class Passage:
    """A retrievable piece of a document, with the metadata filters need."""

    id: str  # "<doc id>#<position>"
    doc_id: str
    title: str
    text: str
    department: str
    updated: str
    acl: frozenset[str]
    index_text: str = ""  # what BM25 and the embedder actually see


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def chunk_document(doc: Doc, max_words: int = 15, contextual: bool = False) -> list[Passage]:
    """Split a document on sentence boundaries into passages of at most ~max_words words.

    With `contextual=True`, each passage is indexed with a short line saying
    where it comes from (title and department), so a sentence like "Update
    AnyConnect and reboot" still says which error it fixes.
    """
    groups: list[list[str]] = []
    for s in _sentences(doc.text):
        if groups and len(" ".join(groups[-1] + [s]).split()) <= max_words:
            groups[-1].append(s)
        else:
            groups.append([s])
    out = []
    for i, g in enumerate(groups):
        text = " ".join(g)
        prefix = f"{doc.title} ({doc.department}): " if contextual else ""
        out.append(Passage(f"{doc.id}#{i}", doc.id, doc.title, text, doc.department, doc.updated, doc.acl, prefix + text))
    return out


ALL_GROUPS = frozenset(g for d in DOCS for g in d.acl)


class RAGIndex:
    """Passages from every document, indexed for keyword (BM25) and dense search."""

    def __init__(self, docs: list[Doc] = DOCS, contextual: bool = False, embedder: ConceptEmbedder | None = None, max_words: int = 15):
        self.passages = [p for d in docs for p in chunk_document(d, max_words, contextual)]
        self.by_id = {p.id: p for p in self.passages}
        self.embedder = embedder or ConceptEmbedder()
        self.bm25 = BM25([tokenize(p.index_text) for p in self.passages])
        self.vectors = self.embedder.encode([p.index_text for p in self.passages])  # (n_passages, dim), unit rows
        self.reranker = CrossEncoder(self.embedder)

    def retrieve(self, query: str, user_groups: set[str], k: int = 5, mode: str = "hybrid",
                 updated_after: str | None = None, depth: int = 10) -> list[Passage]:
        """Top-k passages the user may see, ranked by BM25, dense, or both fused.

        The permission and date filters run FIRST, on the candidate set,
        before anything is ranked, so restricted text can't leak into the
        results, the context, or the answer.
        """
        allowed = [i for i, p in enumerate(self.passages)
                   if p.acl & set(user_groups) and (updated_after is None or p.updated >= updated_after)]
        rankings = []
        if mode in ("bm25", "hybrid"):
            s = self.bm25.scores(tokenize(query))
            rankings.append([self.passages[i].id for i in sorted(allowed, key=lambda i: -s[i]) if s[i] > 0][:depth])
        if mode in ("dense", "hybrid"):
            sims = self.vectors @ self.embedder.encode(query)
            rankings.append([self.passages[i].id for i in sorted(allowed, key=lambda i: -sims[i])][:depth])
        fused = rankings[0] if len(rankings) == 1 else [pid for pid, _ in reciprocal_rank_fusion(rankings)]
        return [self.by_id[pid] for pid in fused[:k]]

    def rerank(self, query: str, passages: list[Passage], top_n: int = 3) -> list[Passage]:
        """Second stage: score each (question, passage) pair together; keep the best few."""
        as_docs = {p.id: Doc(p.id, p.title, p.text, p.department, p.updated, p.acl) for p in passages}
        return sorted(passages, key=lambda p: -self.reranker.score(query, as_docs[p.id]))[:top_n]

    def retrieve_parents(self, query: str, user_groups: set[str], k: int = 3) -> list[Doc]:
        """Search small passages (precise matches), return their whole documents (full context)."""
        seen: list[str] = []
        for p in self.retrieve(query, user_groups, k=k * 4):
            if p.doc_id not in seen:
                seen.append(p.doc_id)
        return [DOCS_BY_ID[d] for d in seen[:k]]


# ===========================================================================
# 2. QUERY TIME: assemble context, generate, verify
# ===========================================================================

SYSTEM = ("Answer only from the sources. Cite the id of each source you use in square brackets. "
          "If the sources don't contain the answer, say you don't know.")
DONT_KNOW = "I don't know based on the provided sources."


def assemble_context(question: str, passages: list[Passage]) -> str:
    """Sources in escaped, id-tagged blocks; the question last (see primer.agents.context)."""
    sources = "\n".join(xml_wrap("source", p.text, id=p.id, title=p.title, updated=p.updated) for p in passages)
    return f"<sources>\n{sources}\n</sources>\n<question>{question}</question>"


def _overlap(question: str, sentence: str) -> int:
    """Shared ideas between question and sentence; identifiers (with digits) count double."""
    shared = set(ideas(question)) & set(ideas(sentence))
    return sum(2 if any(c.isdigit() for c in i) else 1 for i in shared)


def grounded_policy(system: str, messages: list[dict], tools: list[dict] | None) -> str:  # noqa: ARG001
    """Offline stand-in for a model that answers only from tagged sources, with citations.

    It picks the source sentence sharing the most ideas with the question
    and cites it; if nothing shares at least two, it declines. A real model
    writes more fluent answers, but the contract (answer from sources, cite,
    or decline) is the same one you'd put in its system prompt.
    """
    text = last_user_text(messages)
    question = re.search(r"<question>(.*?)</question>", text, re.S).group(1)
    best, best_score = None, 0
    for sid, body in re.findall(r'<source id="([^"]+)"[^>]*>(.*?)</source>', text, re.S):
        for sentence in _sentences(body):
            score = _overlap(question, sentence)
            if score > best_score:
                best, best_score = (sentence, sid), score
    if best is None or best_score < 2:
        return DONT_KNOW
    return f"{best[0]} [{best[1]}]"


def generate(question: str, passages: list[Passage], llm: Any = None) -> str:
    llm = llm or ScriptedLLM(grounded_policy)
    reply = llm.complete(system=SYSTEM, messages=[{"role": "user", "content": assemble_context(question, passages)}])
    return reply.text


_CITATION = re.compile(r"\[([\w-]+#\d+)\]")
_CLAIM_THEN_CITATION = re.compile(r"([^\[\]]+?)\s*\[([\w-]+#\d+)\]")


def verify_citations(answer_text: str, passages: list[Passage]) -> dict[str, list[str]]:
    """Check that every cited id was actually in the context and supports its sentence."""
    by_id = {p.id: p for p in passages}
    cited = _CITATION.findall(answer_text)
    unknown = [c for c in cited if c not in by_id]
    unsupported = []
    # Each citation vouches for the text between the previous citation and itself.
    for claim, c in _CLAIM_THEN_CITATION.findall(answer_text):
        claim = claim.strip(" .")
        if c in by_id and claim_support(claim, [by_id[c].text]) < 0.6:
            unsupported.append(claim)
    return {"cited": cited, "unknown_ids": unknown, "unsupported": unsupported}


@dataclass
class RAGResult:
    answer: str
    passages: list[Passage]
    context: str
    citations: dict[str, list[str]] = field(default_factory=dict)


def answer(index: RAGIndex, question: str, user_groups: set[str], k: int = 8, top_n: int = 3, llm: Any = None) -> RAGResult:
    """The whole query-time pipeline: retrieve (filtered) -> rerank -> assemble -> generate -> verify."""
    candidates = index.retrieve(question, user_groups, k=k)
    passages = index.rerank(question, candidates, top_n) if candidates else []
    text = generate(question, passages, llm)
    return RAGResult(text, passages, assemble_context(question, passages), verify_citations(text, passages))


def post_generation_filter_answer(index: RAGIndex, question: str, user_groups: set[str]) -> str:
    """The WRONG way: retrieve everything, generate, then strip restricted citations.

    The citation markers go; the restricted facts the model already wrote stay.
    """
    result = answer(index, question, set(ALL_GROUPS))
    allowed = {p.id for p in index.passages if p.acl & set(user_groups)}
    return _CITATION.sub(lambda m: m.group(0) if m.group(1) in allowed else "", result.answer).strip()


# ===========================================================================
# 3. RETRIEVAL UPGRADES
# ===========================================================================

_REFERRING = re.compile(r"\b(it|that|this|they|those|them|what about|and for)\b", re.I)


def rewrite_query(question: str, history: list[str], llm: Any = None) -> str:
    """Turn a follow-up into a standalone search query.

    Deterministic stand-in: if the follow-up leans on earlier context ("it",
    "that", "what about"), append the content words of the previous user
    turn. In production a small model does this with an instruction like
    "rewrite the last question so it can be understood without the chat".
    """
    if llm is not None:
        prompt = "Rewrite the last question as a standalone search query.\n" + "\n".join(history + [question])
        return llm.complete(system="", messages=[{"role": "user", "content": prompt}]).text
    if not history or not _REFERRING.search(question):
        return question
    extra = [t for t in tokenize(history[-1]) if t not in tokenize(question)]
    return f"{question} {' '.join(extra)}"


# What a model might write as a hypothetical answer. HyDE only uses the draft
# to *search*; its details may be wrong, and it's never shown to the user.
HYDE_DRAFTS = {
    "I got a weird message asking for my bank details":
        "This sounds like a phishing email. Don't click links; report the suspicious email with the Report phishing button in Outlook.",
    "why can't my computer reach the office network from home?":
        "Remote access to internal systems needs the VPN. Install AnyConnect, sign in, and connect from home.",
}


def _draft_policy(system, messages, tools):  # noqa: ARG001
    q = last_user_text(messages)
    return HYDE_DRAFTS.get(q, q)


def hyde_search(index: RAGIndex, question: str, user_groups: set[str], k: int = 5, llm: Any = None) -> list[Passage]:
    """Hypothetical Document Embeddings: search with a drafted answer instead of the question.

    Answers look like documents; questions don't. Embedding a plausible
    answer lands closer to the real passage than embedding the question.
    """
    llm = llm or ScriptedLLM(_draft_policy)
    draft = llm.complete(system="Write a short passage that answers the question.",
                         messages=[{"role": "user", "content": question}]).text
    return index.retrieve(draft, user_groups, k=k, mode="dense")


# Alternative phrasings a model might write for multi-query retrieval.
MULTI_QUERY_DRAFTS = {
    "two step login setup": ["two-factor authentication setup", "MFA enrollment authenticator app QR code"],
}


def _phrasings_policy(system, messages, tools):  # noqa: ARG001
    q = last_user_text(messages)
    return "\n".join(MULTI_QUERY_DRAFTS.get(q, []))


def multi_query_search(index: RAGIndex, question: str, user_groups: set[str], k: int = 5, llm: Any = None) -> list[Passage]:
    """Search several phrasings of the question and fuse the rankings with RRF."""
    llm = llm or ScriptedLLM(_phrasings_policy)
    reply = llm.complete(system="Write two alternative search queries, one per line.", messages=[{"role": "user", "content": question}])
    phrasings = [question] + [l for l in reply.text.split("\n") if l.strip()]
    rankings = [[p.id for p in index.retrieve(q, user_groups, k=10)] for q in phrasings]
    return [index.by_id[pid] for pid, _ in reciprocal_rank_fusion(rankings)[:k]]


# ===========================================================================
# 4. AGENTIC RAG: the model decides whether and what to search
# ===========================================================================

SEARCH_TOOL = {"name": "search_kb", "description": "Search the company knowledge base. Returns passages with ids.",
               "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}
_SMALL_TALK = re.compile(r"^\s*(thanks|thank you|hi|hello|ok|great|bye)\b", re.I)


def _agentic_policy(system, messages, tools):  # noqa: ARG001
    """Split multi-part questions, search once per part, then answer every part with citations."""
    question = last_user_text(messages)
    if _SMALL_TALK.match(question):
        return "You're welcome!"
    parts = [p.strip(" ?") for p in re.split(r"\band\b", question) if len(tokenize(p)) >= 2]
    done = tool_results(messages)
    if len(done) < len(parts):
        return ToolCall("", "search_kb", {"query": parts[len(done)]})
    answers = []
    for part, result in zip(parts, done):
        sub = grounded_policy("", [{"role": "user", "content": f"{result['content']}\n<question>{part}</question>"}], None)
        answers.append(sub)
    return " ".join(answers)


def agentic_answer(index: RAGIndex, question: str, user_groups: set[str], max_steps: int = 6, llm: Any = None) -> dict[str, Any]:
    llm = llm or ScriptedLLM(_agentic_policy)
    messages: list[dict] = [{"role": "user", "content": question}]
    searches: list[str] = []
    reply = None
    for _ in range(max_steps):
        reply = llm.complete(system=SYSTEM, messages=messages, tools=[SEARCH_TOOL])
        messages.append({"role": "assistant", "content": reply.assistant_content})
        if reply.stop_reason != "tool_use":
            break
        results = []
        for call in reply.tool_calls:
            q = call.input["query"]
            searches.append(q)
            # The tool applies the freshness filter, so superseded policies never come back.
            hits = index.rerank(q, index.retrieve(q, user_groups, k=8, updated_after="2024-01-01"), top_n=3)
            body = "<sources>" + "".join(xml_wrap("source", p.text, id=p.id) for p in hits) + "</sources>"
            results.append(tool_result_block(call.id, body))
        messages.append({"role": "user", "content": results})
    return {"answer": reply.text if reply else "", "searches": searches}


# ===========================================================================
# 5. GRAPHRAG, in miniature
# ===========================================================================

ENTITIES: dict[str, str] = {
    "two-factor": r"two-factor|\bmfa\b",
    "VPN": r"\bvpn\b",
    "AnyConnect": r"anyconnect",
    "software center": r"software center",
    "authenticator app": r"authenticator app",
    "email": r"\bemails?\b",
    "Outlook": r"outlook",
    "travel portal": r"travel portal",
    "HR portal": r"hr portal",
    "IT portal": r"it portal",
    "expense tool": r"expense tool",
    "self-service portal": r"self-service portal",
    "payroll": r"payroll",
    "laptop": r"\blaptops?\b",
}


def build_graph(docs: list[Doc] = DOCS) -> dict[str, dict[str, set[str]]]:
    """Entity graph: two entities are linked when a sentence mentions both.

    Each link remembers which documents stated it, so answers drawn from the
    graph can still cite sources. Real GraphRAG uses a model to extract
    entities and typed relations ("requires", "replaces") and then
    summarizes communities of related entities.
    """
    graph: dict[str, dict[str, set[str]]] = {}
    for d in docs:
        for sentence in _sentences(f"{d.title}. {d.text}"):
            found = sorted(name for name, pat in ENTITIES.items() if re.search(pat, sentence, re.I))
            for a in found:
                for b in found:
                    if a != b:
                        graph.setdefault(a, {}).setdefault(b, set()).add(d.id)
    return graph


def communities(graph: dict[str, dict[str, set[str]]]) -> list[set[str]]:
    """Connected groups of entities: the "themes" GraphRAG summarizes."""
    seen: set[str] = set()
    groups = []
    for start in graph:
        if start in seen:
            continue
        stack, comp = [start], set()
        while stack:
            n = stack.pop()
            if n not in comp:
                comp.add(n)
                stack.extend(graph.get(n, {}))
        seen |= comp
        groups.append(comp)
    return sorted(groups, key=len, reverse=True)


# ===========================================================================
# 6. DEBUGGING a confident wrong answer
# ===========================================================================


def debug_wrong_answers(index: RAGIndex, mode: str = "hybrid", k: int = 3, queries: list[tuple[str, set[str]]] = LABELED_QUERIES) -> dict[str, Any]:
    """Separate retrieval failures from generation failures on a labelled set.

    For each question: if no relevant document is in the top k, it's a
    retrieval miss (no prompt change can fix it). If one is there but the
    answer doesn't cite it, it's a generation miss.
    """
    by_query: dict[str, str] = {}
    for q, relevant in queries:
        passages = index.retrieve(q, set(ALL_GROUPS), k=k, mode=mode)
        if not relevant & {p.doc_id for p in passages}:
            by_query[q] = "retrieval miss"
            continue
        text = generate(q, passages)
        cited_docs = {c.split("#")[0] for c in _CITATION.findall(text)}
        by_query[q] = "ok" if cited_docs & relevant else "generation miss"
    misses = sum(v == "retrieval miss" for v in by_query.values())
    return {"by_query": by_query, "recall_at_k": 1 - misses / len(queries)}


def recall_curve(index: RAGIndex, mode: str, ks: list[int], rerank: bool = False,
                 queries: list[tuple[str, set[str]]] = LABELED_QUERIES) -> list[float]:
    """recall@k for each k: share of questions with a relevant document in the top k."""
    out = []
    for k in ks:
        hits = 0
        for q, relevant in queries:
            if rerank:
                passages = index.rerank(q, index.retrieve(q, set(ALL_GROUPS), k=10, mode=mode), top_n=k)
            else:
                passages = index.retrieve(q, set(ALL_GROUPS), k=k, mode=mode)
            hits += bool(relevant & {p.doc_id for p in passages})
        out.append(hits / len(queries))
    return out


# ===========================================================================
# Figures and demo
# ===========================================================================


def figures() -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs: dict[str, Any] = {}
    index = RAGIndex()
    ks = [1, 2, 3, 4, 5]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for label, mode, rr, color in [("keyword (BM25)", "bm25", False, "#dd8452"), ("dense", "dense", False, "#8172b2"),
                                   ("hybrid (RRF)", "hybrid", False, "#4c72b0"), ("hybrid + rerank", "hybrid", True, "#55a868")]:
        ax.plot(ks, recall_curve(index, mode, ks, rr), "o-", label=label, color=color)
    ax.set_xticks(ks)
    ax.set_xlabel("k (passages kept)")
    ax.set_ylabel("recall@k on the labelled questions")
    ax.set_ylim(0, 1.05)
    ax.set_title("Which retrieval finds the answer?")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    figs["recall"] = fig

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    kinds = ["ok", "generation miss", "retrieval miss"]
    colors = {"ok": "#55a868", "generation miss": "#dd8452", "retrieval miss": "#c44e52"}
    modes = ["bm25", "dense", "hybrid"]
    left = np.zeros(len(modes))
    for kind in kinds:
        counts = np.array([list(debug_wrong_answers(index, m)["by_query"].values()).count(kind) for m in modes])
        ax.barh(modes, counts, left=left, color=colors[kind], label=kind)
        left += counts
    ax.set_xlabel("labelled questions (top 3 passages)")
    ax.set_title("Where do wrong answers come from?")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    figs["diagnosis"] = fig

    plain, ctx = RAGIndex(), RAGIndex(contextual=True)
    fix = "Update AnyConnect to version 5.1 or later and reboot."
    qs = ["how do I fix ERR-4012", "ERR-4012 what should I do", "resolve ERR-4012 on my laptop"]

    def rank_of(idx, q):
        ids = [p.text for p in idx.retrieve(q, {"everyone"}, k=20)]
        return ids.index(fix) + 1 if fix in ids else 21

    fig, ax = plt.subplots(figsize=(6.5, 3))
    y = np.arange(len(qs))
    ax.barh(y - 0.2, [rank_of(plain, q) for q in qs], 0.4, color="#c44e52", label="plain chunks")
    ax.barh(y + 0.2, [rank_of(ctx, q) for q in qs], 0.4, color="#4c72b0", label="chunks with a context line")
    ax.set_yticks(y, qs)
    ax.invert_yaxis()
    ax.set_xlabel("rank of the passage containing the fix (lower is better; 21 = not in top 20)")
    ax.set_title("Contextual retrieval")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figs["contextual"] = fig
    return figs


def demo() -> None:
    index = RAGIndex()
    banner("1. Parsing: repair the extraction before anything else")
    print(clean_extracted_text(MESSY_PDF))
    print()
    for s in table_rows_as_sentences(extract_table(MESSY_PDF, ["Grade", "Hotel cap", "Meal cap"])):
        print("row ->", s)
    print()

    banner("2. The pipeline, end to end")
    r = answer(index, "What does ERR-4012 mean?", {"everyone"})
    print(r.context)
    print()
    print("answer:", r.answer)
    print("citations:", r.citations)
    print()

    banner("3. Permissions are filtered before retrieval, not after generation")
    print("finance user :", answer(index, "What is the Q3 revenue forecast?", {"everyone", "finance"}).answer)
    print("other user   :", answer(index, "What is the Q3 revenue forecast?", {"everyone"}).answer)
    print("post-filtered:", post_generation_filter_answer(index, "What is the Q3 revenue forecast?", {"everyone"}))
    print()
    say("Post-filtering removed the citation but the number had already been written. Filter first.")

    banner("4. Retrieval upgrades")
    ctx = RAGIndex(contextual=True)
    rows = [
        ("hybrid vs dense on 'what does ERR-4012 mean'", [p.doc_id for p in index.retrieve("what does ERR-4012 mean", {"everyone"}, 3, "dense")], [p.doc_id for p in index.retrieve("what does ERR-4012 mean", {"everyone"}, 3)]),
        ("date filter on the travel policy", [p.doc_id for p in index.retrieve("business class flights over four hours", {"everyone"}, 3)], [p.doc_id for p in index.retrieve("business class flights over four hours", {"everyone"}, 3, updated_after="2024-01-01")]),
        ("contextual chunks for 'how do I fix ERR-4012'", [p.id for p in index.retrieve("how do I fix ERR-4012", {"everyone"}, 3)], [p.id for p in ctx.retrieve("how do I fix ERR-4012", {"everyone"}, 3)]),
        ("rewrite 'what if it fails?' after a VPN question", [p.doc_id for p in index.retrieve("what if it fails?", {"everyone"}, 3)], [p.doc_id for p in index.retrieve(rewrite_query("what if it fails?", ["How do I set up the VPN?"]), {"everyone"}, 3)]),
        ("HyDE for 'weird message asking for my bank details'", [p.doc_id for p in index.retrieve("I got a weird message asking for my bank details", {"everyone"}, 3, "dense")], [p.doc_id for p in hyde_search(index, "I got a weird message asking for my bank details", {"everyone"}, 3)]),
        ("multi-query for 'two step login setup'", [p.doc_id for p in index.retrieve("two step login setup", {"everyone"}, 3)], [p.doc_id for p in multi_query_search(index, "two step login setup", {"everyone"}, 3)]),
    ]
    table(["upgrade", "before", "after"], rows)

    banner("5. Agentic RAG")
    print(agentic_answer(index, "What is the per-diem for meals and how do I upload expense receipts?", {"everyone"}))
    print(agentic_answer(index, "thanks, that's all!", {"everyone"}))
    print()

    banner("6. GraphRAG in miniature")
    g = build_graph()
    for n, docs in sorted(g["two-factor"].items()):
        print(f"two-factor -- {n}  (stated in {sorted(docs)})")
    print("communities:", communities(g))
    print()

    banner("7. Debugging confident wrong answers: retrieval or generation?")
    for mode in ("dense", "hybrid"):
        rep = debug_wrong_answers(index, mode)
        misses = {q: v for q, v in rep["by_query"].items() if v != "ok"}
        print(f"{mode:6s} recall@3 {rep['recall_at_k']:.2f}  problems: {misses}")
    print()
    takeaway("Measure retrieval first. If the right passage isn't in the top k, no prompt can fix the answer.")


if __name__ == "__main__":
    demo()
