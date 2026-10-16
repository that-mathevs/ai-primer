r"""
# Retrieval-augmented generation (RAG), end to end

Run: `python -m primer.agents.rag`

This lesson builds on keyword search, embeddings and reranking from
`primer.ml.embeddings.retrieval`, on prompt assembly from
`primer.agents.context`, and on the tool-calling loop from
`primer.agents.agent_loop`.

## Level 1: The practitioner's guide

**In one sentence.** Retrieval-augmented generation (RAG) fetches the few
passages of your own documents most likely to answer a question, puts them
in the prompt, and has the model answer from those passages with citations,
so the model can use knowledge it was never trained on and a reader can
check where each fact came from.

**When you need it.** When the answer lives in documents the model has not
seen (your policies, tickets, contracts, product manuals), when those
documents change faster than you could retrain, when every answer must name
its source, or when different users may read different documents. You don't
need it when the model already knows the subject (public, stable knowledge),
when you want to change *how* the model behaves rather than *what* it
knows (that is fine-tuning: it teaches format and style, not facts that
change), or when the whole collection fits in the prompt. Anthropic's
contextual retrieval post puts that last threshold at about 200,000 tokens,
roughly 500 pages: below it, put everything in the prompt and skip the
pipeline. The tell: your model answers a question about the 2023 travel
policy confidently and wrongly, because it never saw the 2026 one.

**Your options.** From the cheapest to the most capable, each one usually
added on top of the last:

| Option | What it does | What it guarantees | What it costs | Where it lives |
|---|---|---|---|---|
| Everything in the prompt | Sends the whole collection with every question | Nothing is ever missed by a search | Every token, every call; only up to a few hundred pages | Your prompt |
| Keyword search (BM25) | Ranks passages by the question's rarer words | Exact identifiers (`ERR-4012`, ticket numbers) are found | An inverted index; misses synonyms and paraphrase | A search engine |
| Dense search | Ranks passages by embedding similarity, meaning over words | Paraphrases are found ("scam message" finds the phishing guide) | An embedding model at ingest and per query, a vector index; blurs exact codes | A vector index |
| Hybrid search | Runs both and fuses the rankings by rank (RRF), never by score | Both kinds of question, and no score scales to reconcile | Two indexes, two searches per question | Both indexes plus a fusion step |
| Reranking | A cross-encoder reads the question next to each of the top candidates | Far better ordering of the top few | One model score per candidate, so only over a shortlist | A reranking model between search and the prompt |
| Retrieval upgrades | Contextual chunks, query rewriting, HyDE, multi-query, date filters, parent-child | Each fixes one named failure | Model calls at ingest (context lines) or per query (rewrites, drafts) | Ingestion or query time |
| Agentic RAG | The model decides when, how often and with what words to search | Multi-part and vague questions handled; no search for small talk | Several model calls and a step budget per question | An agent loop with a search tool |
| GraphRAG | Extracts entities and relations first, answers by walking the graph or summarizing clusters | Questions about connections and whole-collection themes | A model pass over every document at ingest, and a graph to maintain | Ingestion, plus a graph store |

**How to choose.** Start from the questions people actually ask, with a
labelled set of at least a dozen where you know the right document.

- Questions full of exact identifiers and paraphrases both, which is what
  enterprise data looks like: hybrid search with a reranker. In this
  lesson's toy set, dense search alone never finds `ERR-4012` and plateaus
  at 92% recall; hybrid reaches 100% by the second result; reranking the
  hybrid shortlist lifts the share of questions answered by the first
  result from 75% to 92%.
- Chunks that lose their meaning when cut from the document (a fix that
  never names the error it fixes): contextual chunks. Prefixing each chunk
  with where it comes from turns "not in the top 20" into "second" here;
  Anthropic measured a 49% drop in top-20 retrieval failures from
  contextual embeddings and contextual BM25, and 67% with a reranker added.
- Follow-up questions and vague phrasing: query rewriting and HyDE, which
  searches with a model-written draft answer.
- Two questions in one, or a question the first search doesn't settle:
  agentic RAG.
- "What depends on X?" and "what are the themes?": GraphRAG, and only then,
  because it is the most expensive to build and keep current.
- Whatever you pick: filter by permissions and date *before* ranking, and
  verify citations before the answer reaches anyone. Retrieval quality is
  decided in the first boxes (parsing, chunking, search), never by the
  prompt.

**What it costs.** Ingestion costs an embedding per chunk, a keyword index,
and for contextual chunks a model call per chunk: with prompt caching,
Anthropic's post prices that at \$1.02 per million document tokens. A query
costs one embedding, two searches, a reranker score for each of a handful
of candidates (8 here, kept to 3), and one generation whose input is the
question plus those passages; that prompt is what the answer's latency and
token bill mostly consist of. Agentic RAG multiplies the model calls by the
number of searches (two for the two-part question in this lesson, zero for
"thanks, that's all!") and needs a step limit. Quality costs come from the
front of the pipeline: a table flattened by a naive PDF extractor, so that
the hotel cap floats free of its grade, is a failure no later stage can
repair. Effort goes mostly into the labelled question set and the parsing,
not the model.

**What breaks.**

- **The right passage is not in the index.** Parsing garbled it, chunking
  split it from its context, or the permission sync dropped it. Check
  ingestion before touching anything downstream.
- **The right passage is in the index but not in the top k.** Measure
  recall@k on the labelled set. If it is low, change retrieval (hybrid,
  reranker, rewriting, HyDE, contextual chunks); a prompt cannot fix it.
- **Permission leaks.** Filtering after generation strips the citation and
  leaves the fact: in this lesson the restricted forecast's "41 million
  dollars" reaches an employee who may not read it. Copy each document's
  access list onto its chunks and filter before ranking; sync revocations
  promptly.
- **Stale sources.** A superseded 2023 policy outranks the current one.
  Carry the date on every chunk and filter or prefer by it.
- **Confident answers with no support.** Instruct the model to answer only
  from the sources and to decline otherwise, then check that every cited id
  was in the context and that its passage supports the claim.
- **Blaming the model.** With dense-only retrieval this lesson's twelve
  questions produce five wrong answers, of which only one is a retrieval
  miss; switching to hybrid removes that one, and the two that remain are
  generation and data problems. Each needs a different fix, so diagnose
  first.

**In the wild.** The name comes from Lewis et al. (2020), who paired a
retriever with a generator over a document index. Keyword search is BM25
in Lucene, Elasticsearch and OpenSearch; dense search runs on vector
indexes such as FAISS and hosted services such as Pinecone; Elasticsearch
documents RRF for fusing the two. HyDE is Gao et al. (2022). Contextual
retrieval and prompt caching are described in Anthropic's post, and
Claude's citations feature returns the exact passage behind each claim for
PDF, plain-text and custom documents. GraphRAG is Microsoft's open
implementation of Edge et al. (2024). RAGAS evaluates a RAG pipeline on
retrieval and generation separately, which is exactly the split the
debugging section relies on.

**Go deeper.** Level 2 builds the whole pipeline in plain Python: a PDF
extractor's damage repaired step by step, chunks that carry their
metadata, BM25 and dense search fused by a one-line formula, a reranker, a
prompt that tags every source, a citation check, the permission leak
reproduced, each upgrade as a working function with its before and after,
and the diagnosis figure you can rerun on your own questions. If you only
needed to choose, you are done.

## Level 2: How it works, from scratch

A language model knows only what was in its training data, and your
company's documents were not in it. **Retrieval-augmented generation** works
in two stages. Ahead of time, the documents are parsed, cut into passages
and indexed. Then, for every question, the system *retrieves* the passages
the asker may read that best match the question, keeps the best few, puts
them in the prompt, and the model *generates* an answer from them, citing
the passage behind each claim.

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
change; the bottom row runs for every question. An answer can contain only
what survives every box, left to right. A table mangled at *Parse*, or a
passage missing at *Retrieve*, cannot be fixed by a better prompt at
*Generate*, which is why most of the work on a RAG system's quality goes
into the early boxes.

**In code:** `RAGIndex` builds the top row's indexes, and `answer` runs the
bottom row and returns a `RAGResult`.

## Parsing: where quality dies first

Before a document can be searched, its text has to be pulled out of the
file, and text pulled out of a PDF comes out damaged. Here is what a naive
extractor does to two pages of a travel policy, and what four repairs make
of it:

| Damage | Before | After |
|---|---|---|
| Header repeated on every page | `Northwind Ltd  Travel Policy 2026 … CONFIDENTIAL` | removed (it appears on 2 of 2 pages) |
| Page-number footer | `Page 1 of 2` | removed |
| Word hyphenated across a line break | `all em-` / `ployees travelling` | `all employees travelling` |
| Table flattened | `L1-L3 150 60 L4-L6 200 75` | kept as rows, then one sentence per row |

The table is the dangerous one. Flattened, "200" no longer says which grade
it belongs to. The last repair writes each row as a sentence that stands on
its own, `Grade L4-L6: hotel cap 200, meal cap 75.`, so any row can be
retrieved by itself and still say what its numbers mean.

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
is the decision that matters: prose and tables need different handling,
because a table's meaning lives in its columns. Real pipelines add
layout-aware parsers and **OCR** (optical character recognition: reading
text from an image of a page) for scanned documents, with a quality check
on the output.

Company documents are scanned, laid out in columns and full of tables that
span pages. If parsing garbles a number, no retriever or model later in the
pipeline can recover it, and the failure looks like a *model* error.

**In code:** `clean_extracted_text` makes the first three repairs and keeps
table rows on their own lines; `extract_table` reads the rows and
`table_rows_as_sentences` writes each one as a sentence.

## Chunking, with metadata that travels

Clean text is still too long to search as one piece. A passage that holds
one idea can match a question precisely, so each document is cut into small
passages called **chunks**. Every chunk carries a label copied from its
document: the title, the date it was last updated and who may read it. With
a 15-word limit, the VPN error guide `it-004` splits into two chunks:

| Passage id | Text | Updated | Readers |
|---|---|---|---|
| `it-004#0` | ERR-4012 means the VPN tunnel could not be established, usually because the client is out of date. | 2026-04-10 | everyone |
| `it-004#1` | Update AnyConnect to version 5.1 or later and reboot. | 2026-04-10 | everyone |

The limit is tiny because these documents are tiny; real systems use chunks
of a few hundred tokens. Cut on the document's own structure (sentences,
paragraphs, sections), not every so many characters, which cuts sentences
in half. `primer.ml.embeddings.retrieval` compares chunking strategies in
depth.

Small chunks match precisely but can lose their context: the second chunk
never says *which* error it fixes, a problem contextual retrieval solves
under the retrieval upgrades below. The label is what makes the permission
and date filters possible later.

**In code:** `chunk_document` splits a document on sentence boundaries into
`Passage`s, each carrying its id, title, department, date and readers.

## Retrieval: keyword and meaning, fused

A question now has to find its chunks, and there are two ways to search
them.

* **Keyword search (BM25)** scores passages by the question's words,
  giving more weight to rare words and less to long passages. It finds
  exact identifiers like `ERR-4012` and misses synonyms.
* **Dense search** compares **embeddings** (vectors where closeness means
  similar meaning; see `primer.ml.embeddings`). It matches "scam message in
  my inbox" to the phishing guide with no word in common, but blurs exact
  codes.

**Hybrid search** runs both and merges the two rankings with **reciprocal
rank fusion (RRF)**, which uses only each passage's rank, never its score.
BM25 scores and cosine similarities are on different scales, so adding them
would mean nothing.

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

### Measuring retrieval

Which search to use is settled by measuring it. Take a set of questions
whose right document you know, here 12, and count how often that document
is among the top k passages a method returns. That share is
**recall@k**.

![Dense search plateaus at 92% recall because it never finds the error code, while hybrid reaches 100% by k = 2 and reranking lifts its top-1 from 75% to 92%](figures/primer.agents.rag.recall.svg)

**Reading it:** the x-axis is how many passages you keep (k), and the y-axis
is recall@k over the 12 labelled questions. On this small set many
questions reuse the documents' own words, so keyword search is already
strong. Dense search levels off at 92% because it never finds `ERR-4012`.
Hybrid reaches 100% by k = 2 but only 75% at k = 1, and reranking the
hybrid shortlist (the next step) lifts the top-1 result to 92%. Run this
measurement on *your* questions before choosing a method.

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

**In code:** `RAGIndex.retrieve` runs keyword, dense or hybrid search over
the passages the asker may read, and `recall_curve` measures recall@k for
each.

## Reranking: read the question and each passage together

Both searches are fast because their indexes are built ahead of time: a
question is compared with what is stored, never read alongside a passage.
A **reranker** does read them together. It is usually a **cross-encoder**,
a model that takes the question and one passage at once and outputs a
relevance score. That is far more precise than comparing stored vectors,
and far too slow to run over every passage. So retrieval casts a wide,
cheap net (8 candidates here) and the reranker keeps the best few (3 here),
using the small cross-encoder built in `primer.ml.embeddings.retrieval`.

**In code:** `RAGIndex.rerank` scores each (question, passage) pair with
`primer.ml.embeddings.retrieval.CrossEncoder` and keeps the best few.

## Assemble, generate, cite, verify

The passages the reranker keeps go into the prompt. Each sits in its own
tag with its id, title and date, its text escaped so it cannot break out of
the tag, and the question comes last (see `primer.agents.context`). The
instructions tell the model to answer only from the sources, to cite the id
of each source it uses, and to say it doesn't know when the sources don't
hold the answer. For "What does ERR-4012 mean?" the prompt reads:

```text
<sources>
<source id="it-004#1" title="Error ERR-4012: VPN tunnel failed" updated="2026-04-10">Update AnyConnect to version 5.1 or later and reboot.</source>
<source id="it-004#0" title="Error ERR-4012: VPN tunnel failed" updated="2026-04-10">ERR-4012 means the VPN tunnel could not be established, usually because the client is out of date.</source>
<source id="it-002#2" title="Password policy" updated="2025-11-20">This policy applies to all employees and contractors.</source>
</sources>
<question>What does ERR-4012 mean?</question>
```

The answer is *"ERR-4012 means the VPN tunnel could not be established,
usually because the client is out of date. [it-004#0]"*. A final check
then reads every citation: the cited id must have been in the context, and
the cited passage must support the claim before it. A question the sources
don't cover gets "I don't know based on the provided sources."

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

**Reading it:** time runs downward. The permission filter runs at the index,
before any passage is fetched; the model sees only three passages; and
nothing reaches the user until the citations check out. A checked citation
lets a person confirm the answer in seconds by reading the passage it
points to.

**In code:** `answer` chains `RAGIndex.retrieve`, `RAGIndex.rerank`,
`assemble_context`, `generate` and `verify_citations`; `grounded_policy` is
the offline model that stands in for a real one.

## Permission-aware retrieval: filter before, never after

The diagram above filters at the index, before anything is ranked. In a
company not everyone may read every document: the revenue forecast is for
finance and executives only. So each passage carries its document's **access-control list** (ACL:
the groups allowed to read it), copied from the system the document came
from (SharePoint, Google Drive, a wiki). The question is when to check it.
Retrieval checks it *before* ranking and drops every passage the asker
cannot read, so restricted text never reaches the context, the model or
the answer. The alternative, deleting restricted citations after the answer
is written, leaves the fact behind. Here is "What is the Q3 revenue
forecast?", where the forecast document is readable only by `finance` and
`exec`:

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
Permission changes in the source system must also reach the index quickly,
or a user whose access was revoked keeps it until the next re-index.

**In code:** `post_generation_filter_answer` is the wrong way, kept to show
the leak: it retrieves for every group, generates, then strips the
restricted citation markers.

## Retrieval upgrades

Even filtered hybrid search with a reranker fails in a few recognizable
ways: a follow-up question that says "it", a question worded nothing like
its answer, an out-of-date document that still matches, a passage that never
names its subject. Each upgrade fixes one failure. The last column shows the
top three passages without it and with it, on a question that has that
failure:

| Upgrade | What it does | Before → after (top 3) |
|---|---|---|
| **Query rewriting** | Fills in a follow-up's subject from earlier in the conversation | "what if it fails?" after a VPN question: printer guide → VPN setup and ERR-4012 |
| **HyDE** | Searches with a drafted answer instead of the question | "a weird message asking for my bank details": nothing relevant → phishing guide first |
| **Multi-query** | Searches several phrasings of the question and fuses the results | "two step login setup": no MFA guide → MFA guide in top 3 |
| **Metadata filter** | Drops documents updated before a given date | superseded 2023 travel policy first → gone |
| **Contextual retrieval** | Writes where each passage comes from into it before indexing | "how do I fix ERR-4012": fix sentence missing → found |
| **Parent-child** | Matches a small passage, hands the model its whole document | a matching sentence → its full document |

**In code:** `rewrite_query`, `hyde_search`, `multi_query_search` and
`RAGIndex.retrieve_parents` are four of the upgrades. The date filter is an
option of `RAGIndex.retrieve`, contextual chunks are an option of
`RAGIndex`, and `demo` runs each upgrade on its question.

### HyDE

**HyDE** (Hypothetical Document Embeddings) sounds backwards: it asks a
model to *invent* an answer, then searches with the invented answer instead
of the question.

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
the drafted answer uses the words answers use ("phishing", "report",
"suspicious"), so its embedding lands next to the real passage. The draft
may be wrong in its details. That does not matter, because it is used only
to search and is never shown to the user.

### Contextual retrieval

**Contextual retrieval** fixes the second `it-004` chunk from the chunking
step, the one that never says which error it fixes. Before indexing, each
passage gets a short line saying where it comes from. Here that line is
just the document's title and department; Anthropic's version asks a model
to write one sentence placing the chunk in its document.

![Without a context line the fix passage misses the top 20 for all three phrasings; with the title prefixed it ranks 2nd, 2nd and 4th](figures/primer.agents.rag.contextual.svg)

**Reading it:** three ways of asking how to fix ERR-4012. Each pair of bars
is the rank of the passage "Update AnyConnect to version 5.1 or later and
reboot." (shorter is better, and 21 means not in the top 20). Without a
context line the passage never mentions ERR-4012, and for all three
phrasings it misses the top 20. With the title prefixed it ranks 2nd, 2nd
and 4th. That is in the top five every time, though "resolve ERR-4012 on my
laptop" still ranks three passages from the laptop guide above it. The
context line gets the passage found; putting it first is the reranker's
job.

## Agentic RAG: let the model decide when and what to search

Every pipeline so far runs the same steps for every question, whatever it
asks. Some questions need several searches, and some need none. In
**agentic RAG** search is a tool the model calls inside the agent loop (see
`primer.agents.agent_loop`): the model reads the question, decides whether
to search and with which words, reads the results, and searches again if
something is missing. Asked "What is the per-diem for meals and how do I
upload expense receipts?", it runs two searches ("What is the per-diem for
meals", "how do I upload expense receipts") and answers citing both
`fin-002#2` and `fin-004#0`. Told "thanks, that's all!", it does not search
at all.

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

**Reading it:** time runs downward: one search for each part of the
question, then one answer with a citation for each part. Each search costs
a model call, so agentic RAG is slower and dearer than one search, and like
any agent loop it needs a limit on its steps. The search tool applies the
date filter itself, so the model cannot forget it.

**In code:** `agentic_answer` offers the model one search tool and runs the
loop: each search retrieves, reranks and returns tagged passages, until the
model answers or the step limit is hit.

## GraphRAG: questions about connections

Some questions are not answered by any passage, however many searches run:
"what depends on two-factor authentication?" or "what are the main themes
across these contracts?". The answer is spread across many documents.
**GraphRAG** first extracts **entities** (named things: systems, people,
products) and the relationships between them into a graph, then answers by
walking the graph or by summarizing groups of connected entities. The
version here is the smallest one that works: two entities are linked when
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
  VPN ---|it-006| EM
  AA ---|it-001| SSP((self-service portal))
```

**Reading it:** each circle is an entity found in the documents, and each
line is a sentence that mentions both ends, labelled with its source
document. "What depends on two-factor?" is answered by reading the
neighbours of one circle, and each neighbour comes with the document that
links it. Real GraphRAG uses a model to extract typed relations
("requires", "replaces") and to write a summary for each cluster, which is
what makes questions about the whole collection answerable.

**In code:** `build_graph` links the entities, and `communities` groups them
into connected clusters, the "themes" a real GraphRAG system would
summarize.

## Debugging a confident wrong answer

With this many steps between a question and its answer, a confident wrong
answer could start in any of them. Either retrieval brought the wrong
passages or the model misused the right ones. The fixes are different, so
find out which before changing anything. Run the 12 labelled questions and
sort each one: a **retrieval miss** when no passage from the right
document is in the top 3 (no prompt can fix that), a **generation miss**
when the right document was there but the answer didn't use it, or ok.

With dense-only retrieval, "what does ERR-4012 mean" is a retrieval miss and
recall@3 is 0.92; switching to hybrid fixes it (recall@3 = 1.00). What's left
are generation misses. One of them, "per-diem for meals when traveling", is
answered from the superseded 2023 policy: a *data* problem that a date
filter fixes. The other, "enroll in MFA", is a limit of the simple model
used here, not of RAG: it matches words literally, so "enroll" misses the
passage's "enrolling", and with only "MFA" in common it declines to answer.
A real model would read it-006 and answer.

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
measurement, not a guess. Editing the prompt is the bottom-right box. Rule
out the three boxes before it first, because no prompt can fix a failure in
any of them.

![Dense search has the only retrieval miss; switching to hybrid removes it, and the wrong answers left are generation misses](figures/primer.agents.rag.diagnosis.svg)

**Reading it:** each bar is one retrieval method over the 12 labelled
questions, split into ok (green), generation misses (orange) and retrieval
misses (red). Dense search has one retrieval miss, the error code, and
hybrid removes it. The orange that remains is generation, and that is where
to look next.

**In code:** `debug_wrong_answers` sorts the labelled questions into the
three kinds.

## In 20 seconds
- RAG searches your documents when a question arrives and has the model
  answer from the passages it finds, citing each one, so knowledge can
  change or be checked without retraining.
- Quality is decided before the model runs: in parsing (tables above all),
  in chunks that carry their document's metadata, and in hybrid search
  followed by reranking.
- Check permissions before retrieval, never after generation: once the
  model has read a restricted passage, deleting its citation leaves the fact
  in the answer.
- Each retrieval upgrade fixes one named failure; agentic RAG handles
  questions that need several searches, and GraphRAG handles questions about
  connections.
- Debug a wrong answer by measuring retrieval (recall@k) first, and look at
  generation only after.

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
