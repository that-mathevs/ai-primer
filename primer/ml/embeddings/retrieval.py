r"""
# Retrieval: keyword search, meaning search, and combining them

Run: `python -m primer.ml.embeddings.retrieval`

New to the notation (Σ, log, |d|)? `primer.notation` explains every symbol
used here from zero.

## The everyday picture

You walk into a library with a question. Two librarians are on duty.

- **The keyword librarian** takes your words literally. They count how often
  each word of your question appears in each book, give *rare* words far more
  weight than common ones, stop getting more excited after the tenth mention,
  and are a little suspicious of very long books that mention everything.
  Ask about "ERR-4012" and they find the one page that says "ERR-4012". Ask
  about "automobile reimbursement" and they find nothing, because the book
  says "car" and "reimbursed".
- **The meaning librarian** understands what you *mean*: "automobile" is a
  "car", "scam" is "phishing". But they remember the gist of each book, not
  its serial numbers, so "ERR-4012" and "ERR-4013" blur together.

Real systems ask **both** and trust the books both recommend. Then an
expert reads the top handful of candidates carefully, next to your question,
and puts them in final order. That's the whole modern retrieval pipeline,
and this lesson builds every piece of it.

Words used throughout, in plain terms:

- **Retrieval.** Finding the documents that answer a question, ranked best
  first. In a RAG system (`primer.agents.rag`) the language model can only use
  what retrieval hands it, so retrieval quality caps answer quality.
- **Token.** Here, one word after lowercasing and dropping filler words
  ("the", "is"): see `primer.common.text`.
- **Recall@k.** The share of questions whose answer appears in the top k
  results. **MRR** (mean reciprocal rank) also rewards putting the answer
  *first*: an answer at rank 1 scores 1, at rank 2 scores 1/2, missing scores
  0, averaged over questions.

The full pipeline, which the sections below build one box at a time:

```mermaid
flowchart LR
  Q[Question] --> K[Keyword search<br/>BM25]
  Q --> D[Meaning search<br/>embeddings]
  K --> F[Fuse the two rankings<br/>RRF]
  D --> F
  F --> S[Shortlist<br/>top 10 to 100]
  S --> R[Rerank each candidate<br/>with the question: cross-encoder]
  R --> T[Top 3 to 5 passages<br/>to the language model]
```

**Reading it:** two cheap, wide searches run side by side and see the whole
collection. Their rankings are merged into one shortlist. Only the shortlist
reaches the expensive, precise reranker on the right. Everything left of the
shortlist must be fast because it faces millions of documents; everything
right of it can be slow because it faces only a hundred. Each box below gets
its own section.

## 1. Keyword search: BM25

**The everyday picture.** The keyword librarian from above: rare words count
more, repetition helps less and less, and long books get a small penalty for
mentioning everything.

**A tiny worked example.** Three one-line "documents":

| Doc | Words | Length |
|---|---|---|
| d₁ | cat cat dog | 3 |
| d₂ | dog bird | 2 |
| d₃ | fish | 1 |

The average length is (3 + 2 + 1)/3 = 2. Search for **cat**. It appears in
one of the three documents, so it's rare and gets a high weight (0.981,
computed below); "dog" is in two, so it would get less (0.470). Only d₁
contains "cat", twice, and d₁ is a bit longer than average. Plugging in,
d₁ scores **1.207** and the others score 0.

```mermaid
flowchart LR
  Q[Query words] --> IDF["Rarity weight per word<br/>IDF: rarer = heavier"]
  Q --> TF["Count in this document<br/>tf, with saturation"]
  DL["Document length vs.<br/>the average length"] --> TF
  IDF --> M((×))
  TF --> M
  M --> SUM["Add up over<br/>the query's words"]
  SUM --> S[BM25 score]
```

**Reading it:** each query word contributes one product: *how rare the word
is* times *how much this document talks about it*. The length box feeds into
the count box because a mention in a short document is stronger evidence than
the same mention in a long one. Words the document doesn't contain
contribute zero, so a document sharing no words with the query scores exactly
0. That's the blind spot the meaning librarian covers.

$$
\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot
\frac{f(t, d)\,(k_1 + 1)}{f(t, d) + k_1 \left(1 - b + b\,\frac{|d|}{\text{avgdl}}\right)},
\qquad
\text{IDF}(t) = \ln\!\left(1 + \frac{N - n(t) + 0.5}{n(t) + 0.5}\right)
$$

**Symbols**

| Symbol | Meaning here | Range / typical |
|---|---|---|
| q | the query, as a list of words | |
| d | one document, as a list of words | |
| t ∈ q | "each word t in the query" | |
| Σ | add up the following over every query word | |
| f(t, d) | how many times word t appears in document d (term frequency) | 0, 1, 2, … |
| \|d\| | the document's length in words | |
| avgdl | the average document length in the collection | |
| k₁ | saturation: how quickly extra mentions stop helping | 1.2 to 2.0; 1.5 here |
| b | length normalization: 0 ignores length, 1 fully normalizes | 0.75 |
| N | number of documents in the collection | |
| n(t) | number of documents containing t (document frequency) | 1 … N |
| ln | natural logarithm: grows slowly, so ten times rarer is not ten times heavier | |
| IDF(t) | inverse document frequency: the word's rarity weight | ≥ 0 |

**In words:** for each word of the query, multiply how rare the word is by a
count of its mentions that saturates and is adjusted for document length,
then add those products up.

**On the example (query "cat", document d₁):** N = 3, n(cat) = 1, so
IDF = ln(1 + 2.5/1.5) = ln(2.667) = **0.981**. f = 2, |d| = 3, avgdl = 2:
the denominator is 2 + 1.5 · (1 − 0.75 + 0.75 · 3/2) = 2 + 1.5 · 1.375 =
4.0625. The count part is 2 · 2.5 / 4.0625 = 1.231. Score = 0.981 · 1.231 =
**1.207**.

**In Python:**

```python
>>> import math
>>> N, n_t = 3, 1                     # 3 documents; 1 of them contains "cat"
>>> IDF = math.log(1 + (N - n_t + 0.5) / (n_t + 0.5))   # ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))
>>> round(IDF, 3)
0.981
>>> f, d_len, avgdl = 2, 3, 2         # f(cat, d1), |d1|, average document length
>>> k_1, b = 1.5, 0.75
>>> count_part = f * (k_1 + 1) / (f + k_1 * (1 - b + b * d_len / avgdl))
>>> round(count_part, 3)
1.231
>>> round(IDF * count_part, 3)        # Σ over the query's only word
1.207
```

![BM25 saturation and length normalization](figures/primer.ml.embeddings.retrieval.bm25_curves.svg)

**Reading it:** on the left, the x-axis is how often a word appears in a
document and the y-axis is the credit BM25 gives it (before the rarity
weight). The dashed line is raw counting, where 20 mentions count 20 times
as much as one. The BM25 curves bend over and flatten toward k₁ + 1: the
tenth mention adds almost nothing, which stops keyword-stuffed pages from
winning. Smaller k₁ flattens sooner. On the right, one mention in documents
of different lengths: with b = 0 length is ignored; with b = 0.75 a mention
in a document twice the average length counts noticeably less.

**In code:** `BM25` precomputes each word's IDF and each document's length;
`BM25.scores` applies the formula to every document, and `BM25.search`
returns the top k with a non-zero score.

**Why it matters:** BM25 is decades old, needs no training, runs on any
search engine (Elasticsearch, OpenSearch, Lucene), and is still very hard to
beat on exact identifiers: product codes, error numbers, names, ticket IDs.
Those are everywhere in company data.

## 2. Meaning search: dense retrieval with a bi-encoder

**The everyday picture.** The meaning librarian writes a one-line summary
card for every book *before* anyone asks anything. When your question
arrives, they write a summary card for it too and pull the books whose cards
are most similar. "Automobile reimbursement" and "car mileage expense" get
similar cards even though they share no words.

**A tiny worked example.** With this repo's toy embedding model
(`primer.common.embedder`), the query "automobile reimbursement" and the
car-mileage article share zero words, yet their vectors have cosine
similarity of about 0.8, the highest in the collection. "What does ERR-4012
mean", by contrast, puts the right article only 4th: the code is one
blurred word among many, and "what" and "mean" pull the vector elsewhere.

```mermaid
flowchart LR
  subgraph AHEAD["Ahead of time"]
    D[Each document] --> E1[Encoder] --> V[(Document vectors)]
  end
  subgraph NOW["At question time"]
    Q[Question] --> E2[Same encoder] --> QV[Question vector]
  end
  QV --> NN["Nearest vectors<br/>(ann.py)"]
  V --> NN
  NN --> R[Ranked documents]
```

**Reading it:** the encoder runs twice, but never on the question and a
document *together*. That's why it's called a **bi-encoder**: two separate
encodings. Documents are encoded once, at ingestion. A question costs one
encoding plus a nearest-neighbor lookup (see `primer.ml.embeddings.ann`),
which takes milliseconds over millions of documents. The price: the model
never sees the question and the document side by side, so it can't check
fine details like "does this passage answer *this* question, or just share
its topic?".

**In code:** `SearchEngine` embeds every document once with
`primer.common.embedder.ConceptEmbedder` into a
`primer.ml.embeddings.ann.FlatIndex`; `SearchEngine.dense` embeds the
question and returns the nearest documents. `doc_text` decides what gets
indexed: the title plus the body.

**Why it matters:** dense retrieval handles paraphrases, synonyms and other
languages, but it blurs exact tokens. Measure both kinds of queries on your
own data before trusting either librarian alone.

## 3. Hybrid search: reciprocal rank fusion (RRF)

**The everyday picture.** Ask both librarians for their top-10 lists, then
hold a vote. A book earns points for each list it's on, more the higher it
sits. Books both librarians rank highly rise to the top. Crucially, you only
compare *positions*, never the librarians' private scoring systems, which
aren't on the same scale.

**A tiny worked example.** With the standard constant k = 60:

| Document | Dense rank | BM25 rank | RRF score |
|---|---|---|---|
| X | 1 | 3 | 1/61 + 1/63 = 0.0164 + 0.0159 = **0.0323** |
| Y | 1 | (absent) | 1/61 = **0.0164** |

X, which both methods like, beats Y, which only one method loves.

```mermaid
flowchart LR
  Q[Question] --> B[BM25 ranking]
  Q --> D[Dense ranking]
  B --> R["For every document:<br/>add 1/(60 + rank)<br/>from each list it's on"]
  D --> R
  R --> O[Sort by total:<br/>the fused ranking]
```

**Reading it:** the two rankings are the only inputs; their raw scores are
thrown away on purpose. BM25 scores run from 0 to about 20; cosine
similarities from −1 to 1. Adding them would let one system drown the other,
and fixing that needs tuning per collection. Ranks are always on the same
scale, so RRF works out of the box.

$$
\text{RRF}(d) = \sum_{i=1}^{m} \frac{1}{k + \text{rank}_i(d)}
$$

**Symbols**

| Symbol | Meaning here | Range / typical |
|---|---|---|
| d | a document | |
| m | number of rankings being fused | 2 here (BM25 and dense) |
| i | which ranking | 1 … m |
| rank_i(d) | d's position in ranking i (1 = top); lists d isn't on contribute nothing | 1, 2, 3, … |
| k | a damping constant: keeps rank 1 from dwarfing rank 2 | 60 (from the original paper) |
| RRF(d) | the fused score | small positive numbers |

**In words:** a document's fused score is the sum, over each ranking it
appears in, of one over sixty-plus-its-rank.

**On the example:** X: 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = **0.0323**.
Y: 1/(60+1) = **0.0164**.

**In Python:**

```python
>>> k = 60
>>> def RRF(ranks):                   # d's position in each ranking that contains it
...     return sum(1 / (k + rank_i) for rank_i in ranks)   # Σ_i 1/(k + rank_i(d))
>>> round(RRF([1, 3]), 4)             # X: 1st in one ranking, 3rd in the other
0.0323
>>> round(RRF([1]), 4)                # Y: on one ranking only
0.0164
```

![RRF on "what does ERR-4012 mean"](figures/primer.ml.embeddings.retrieval.rrf_fusion.svg)

**Reading it:** each bar is one document's fused score, split into the part
contributed by BM25 (orange) and by dense search (blue). The error-code
article it-004 is BM25's only hit and dense search's 4th, and the two
contributions stack up to put it clearly first. The password-policy article
that dense search wrongly ranked first gets only its blue half.

![Where each method ranks the right answer, for 20 questions](figures/primer.ml.embeddings.retrieval.method_ranks.svg)

**Reading it:** one row per question, one column per method; the number is
the rank at which the right answer appeared (✗ = not in the top 10). Look at
where the colors differ. BM25's failures (the paraphrase block) are where
dense search succeeds, and dense search's failures (the error-code block) are
where BM25 succeeds. Because their mistakes don't overlap, fusing them fixes
both: the hybrid column is all 1s.

| Method | recall@3 (20 questions) | MRR |
|---|---|---|
| BM25 alone | 0.75 | 0.75 |
| Dense alone | 0.90 | 0.85 |
| **Hybrid (RRF)** | **1.00** | **1.00** |

**In code:** `reciprocal_rank_fusion` adds up 1/(k + rank) across any number
of rankings; `SearchEngine.hybrid` fuses `SearchEngine.bm25` with
`SearchEngine.dense`, and `evaluate` measures recall@k and MRR for any search
method.

**Why it matters:** company data is full of both paraphrase-style questions
and exact identifiers, so hybrid search almost always beats either method
alone. It's cheap to add: most search engines and vector databases support
it built in.

## 4. Reranking: bi-encoder vs. cross-encoder

**The everyday picture.** A matchmaker has two ways to work. The fast way:
write an index card for each person once, then match cards, so thousands of
people can be pre-filed. The careful way: sit the two people down *together*
and watch them talk. That's far more accurate, but every pair needs its own
meeting. A **bi-encoder** is the index card. A **cross-encoder** is the
meeting. So you use cards to shortlist, then hold meetings only with the
shortlist.

**A tiny worked example: a hard negative.** "What are the password rules?"
has two candidate answers on the same topic. The password *reset* how-to
(it-001) mentions "password" three times; the password *policy* (it-002) is
the real answer. BM25 and the fused hybrid ranking both put the how-to first.
A **hard negative** is exactly this: a document that looks relevant (right
topic) but doesn't answer the question.

Our toy cross-encoder reads the question and the document together and
checks which of the question's *ideas* the document covers:

| | question ideas: {password, rules} | coverage | phrase | exact | score |
|---|---|---|---|---|---|
| it-002 policy | password ✓, policy/rules ✓ | 2/2 = **1.0** | 1.0 | 0.5 | ≈ 1.9 |
| it-001 reset how-to | password ✓, rules ✗ | 1/2 = **0.5** | 0.0 | 0.5 | ≈ 0.7 |

After reranking, the policy is first.

```mermaid
flowchart TB
  subgraph BI["Bi-encoder: two separate encodings"]
    direction LR
    Q1[Question] --> EQ[Encoder] --> VQ[vector]
    D1[Document] --> ED[Encoder] --> VD[vector]
    VQ --> COS[cosine]
    VD --> COS
  end
  subgraph CROSS["Cross-encoder: one joint reading"]
    direction LR
    QD["[Question] + [Document]<br/>as one input"] --> J["Encoder with attention<br/>across both texts"] --> SC[relevance score]
  end
```

**Reading it:** in the top box, the question and the document never meet
until their vectors are compared, so document vectors can be computed ahead
of time. In the bottom box they're one input, so attention (see
`primer.ml.attention`) can connect every question word to every document
word: "rules" can notice that the document says "policy" and "must". Nothing
can be precomputed, because the score depends on the pair.

**The cost argument.** Say a cross-encoder pass takes 10 ms on a GPU. Over a
1-million-document collection that's 10,000 seconds per question. Over a
shortlist of 50, it's 0.5 s, or much less when the pairs are batched. That
is why the standard pipeline is *retrieve wide and cheap, then rerank narrow
and precise*, and why you cap the shortlist to keep latency predictable.

**In code:** `CrossEncoder.score` reads one question-document pair and adds
`CrossEncoder.coverage`, `CrossEncoder.phrase` and `CrossEncoder.exact`, with
`ideas` grouping synonyms into ideas. `retrieve_then_rerank` shortlists with
hybrid search, then reorders the shortlist with the cross-encoder.

**Why it matters, and a warning:** rerankers are models too, and can be wrong.
On this repo's 20 questions, reranking fixes the hard negatives but demotes
one answer ("refund for the hotel" → it prefers the car-mileage article,
which talks about reimbursing trips). Measure recall@k and MRR with and
without the reranker before shipping it.

## 5. Late interaction: ColBERT

**The everyday picture.** Instead of one summary card per book, keep a card
for *every word* in the book. When your question arrives, each of your words
looks for its single best-matching card in the book, and the book's score is
how well your words found partners. That's more precise than one summary per
book, and unlike the matchmaker's meetings, the cards can still be written
ahead of time.

**A tiny worked example.** Two-number word vectors. The query has words
(1, 0) and (0, 1); the document has words (1, 0) and (0.6, 0.8).

| Query word | vs. doc word 1 | vs. doc word 2 | best |
|---|---|---|---|
| (1, 0) | 1.0 | 0.6 | **1.0** |
| (0, 1) | 0.0 | 0.8 | **0.8** |

Score = 1.0 + 0.8 = **1.8**.

```mermaid
flowchart LR
  QT["Query: one vector<br/>per word"] --> SIM["Similarity of every<br/>query word × doc word"]
  DT["Document: one vector<br/>per word (precomputed)"] --> SIM
  SIM --> MAX["Each query word keeps<br/>its best match (max)"]
  MAX --> SUM["Add the maxima"] --> S[Score]
```

**Reading it:** the grid in the middle is the question-by-document table
from the worked example; "max" picks the best cell in each row; "sum" adds
the row winners. The document side is precomputed, as with a bi-encoder, but
nothing is squeezed into one vector, so each question word gets matched on
its own.

$$
\text{score}(q, d) = \sum_{i=1}^{|q|} \max_{j=1}^{|d|} \; q_i \cdot d_j
$$

**Symbols**

| Symbol | Meaning here |
|---|---|
| \|q\|, \|d\| | number of word vectors in the query and the document |
| q_i | the vector of query word i |
| d_j | the vector of document word j |
| q_i · d_j | dot product: how similar those two words are |
| max over j | the best-matching document word for query word i |
| Σ over i | add those best matches up |

**In words:** for each query word, find its most similar word in the
document, and add up those best similarities.

**On the example:** max(1.0, 0.6) + max(0.0, 0.8) = 1.0 + 0.8 = **1.8**.

**In Python:**

```python
>>> q = [(1, 0), (0, 1)]              # one vector per query word
>>> d = [(1, 0), (0.6, 0.8)]          # one vector per document word
>>> def dot(a, b):
...     return sum(a_k * b_k for a_k, b_k in zip(a, b))
>>> [max(dot(q_i, d_j) for d_j in d) for q_i in q]   # each query word's best match
[1, 0.8]
>>> sum(max(dot(q_i, d_j) for d_j in d) for q_i in q)   # Σ_i max_j q_i · d_j
1.8
```

![ColBERT's MaxSim grid for "password rules" vs. the password policy](figures/primer.ml.embeddings.retrieval.maxsim_grid.svg)

**Reading it:** rows are the query's words, columns the document's words,
and color is the similarity of each pair. The outlined cell in each row is
that row's maximum, the only number that counts. "password" finds itself
(1.00), ignoring the near-identical "passwords"; "rules" finds "policy"
(0.85), a synonym it could never match by spelling. The score is the sum of
the outlined cells: 1.00 + 0.85 = 1.85.

**In code:** `maxsim` computes the score from two sets of word vectors;
`LateInteraction.token_vectors` gives a text one vector per word, and
`LateInteraction.search` ranks documents by MaxSim.

**Why it matters:** late interaction gets much of a cross-encoder's precision
while keeping precomputed documents. The cost is storage: one vector per
*word* instead of per document, often 50 to 200 times more. ColBERTv2
compresses those vectors to make it practical.

## 6. Chunking: how documents are cut before indexing

**The everyday picture.** You're turning a cookbook into index cards. Cut
every 50 words and a recipe's title lands on one card and its oven
temperature on the next. Nobody searching for that recipe finds the
temperature. Cut at each recipe instead, and every card is self-contained.

**A tiny worked example.** A 130-word text, 50-word chunks with 10 words of
overlap: windows start every 50 − 10 = 40 words, at 0, 40 and 80, giving
**3 chunks** (words 1–50, 41–90, 81–130). Each shares 10 words with the next,
so a sentence cut at a boundary survives whole in at least one chunk, if
it's short.

On this lesson's remote-work handbook, the question "home internet stipend"
shows the difference. With 40-word fixed chunks, the best-matching chunk
holds the heading "Home internet stipend" but the amount, "50 dollars per
month", fell into the next window. With structure-aware chunks, cut at
headings, the best chunk holds both.

```mermaid
flowchart LR
  DOC[Document] --> P[Parse: headings,<br/>paragraphs, tables]
  P --> SPLIT["Split at structure;<br/>merge small pieces<br/>up to a size limit"]
  SPLIT --> CTX["Prefix each chunk with<br/>its heading (context)"]
  CTX --> META["Attach metadata: source,<br/>section, date, access list"]
  META --> EMB[Embed + index]
```

**Reading it:** chunking is a small pipeline, not a single split. Parsing
decides what the structure *is* (and is where most quality is lost on messy
PDFs). Splitting respects that structure. Prefixing the heading means a
chunk that says "The stipend is 50 dollars" still says *which* stipend: the
idea behind *contextual retrieval*. Metadata rides along so later stages can
filter by date or by who's allowed to see it (see `primer.agents.rag`).

![Where the two chunkers cut the handbook](figures/primer.ml.embeddings.retrieval.chunk_boundaries.svg)

**Reading it:** the colored strip in the middle is the handbook, one color
per section, read left to right by word position. The brackets above it are
the fixed 40-word windows; those below are the structure-aware chunks. The
red marker is the "Home internet stipend" heading and the green marker the
"50 dollars" amount. Above the strip they sit in different windows; below
it, one chunk spans both. Notice also that the fixed windows cut straight
through section boundaries.

**Parent-child retrieval.** Small chunks match precisely; big chunks give the
model enough context. Get both: index small *children* (single sentences),
and when one matches, hand the model its *parent* (the whole section).

```mermaid
flowchart LR
  Q[Question] --> C["Search small children<br/>(sentences)"]
  C --> HIT[Best child]
  HIT --> PARENT["Look up its parent<br/>(the whole section)"]
  PARENT --> LLM[Give the parent<br/>to the model]
```

**Reading it:** the search happens on the left, at sentence level, where
matches are sharp. The answer handed on happens on the right, at section
level, where context is complete. The lookup in the middle is just a
dictionary from child to parent.

$$
\text{number of chunks} = \left\lceil \frac{n - o}{s - o} \right\rceil
$$

**Symbols**

| Symbol | Meaning here |
|---|---|
| n | words in the document |
| s | chunk size in words |
| o | overlap in words (smaller than s) |
| s − o | the step: how far each window moves |
| ⌈ ⌉ | "ceiling": round up to a whole number |

**In words:** the number of chunks is the words left after the first
overlap, divided by the step, rounded up.

**On the example:** ⌈(130 − 10)/(50 − 10)⌉ = ⌈120/40⌉ = **3**.

**In Python:**

```python
>>> import math
>>> n, s, o = 130, 50, 10             # words, chunk size, overlap
>>> math.ceil((n - o) / (s - o))      # ⌈(n - o) / (s - o)⌉: the step is s - o
3
```

**In code:** `fixed_size_chunks` cuts overlapping windows of words;
`structure_aware_chunks` cuts at headings and paragraphs and returns `Chunk`
records that carry their heading and metadata. `best_chunk` picks the chunk
BM25 ranks highest, and `parent_child_search` matches a sentence and returns
its whole section.

**Why it matters:** chunking choices often matter more than the choice of
embedding model. Split on structure, keep headings with their content, add
modest overlap, and attach metadata for filtering.

## 7. Asymmetric retrieval and the forgotten prefix

**The everyday picture.** A filing clerk was trained on sheets stamped
"QUESTION:" or "DOCUMENT:", and files each kind in a matching drawer. Hand
them an unstamped sheet and they don't complain: they file it anyway, in a
drawer nobody will look in. Nobody notices until people stop finding things.

Questions are short and phrased as asks; passages are long statements. Some
embedding models (the E5 family, for example) are trained to expect a
`"query: "` prefix on questions and a `"passage: "` prefix on documents, so
they can encode each side appropriately. Forget a prefix and every vector
still looks normal (right length, plausible scores) but sits in a slightly
wrong part of the space.

**A tiny worked example.** This lesson simulates such a model. With both
prefixes, 11 of the 12 everyday questions find their answer in the top 3.
Index the documents *without* "passage: " and the same questions fall to 9
of 12, and MRR drops from 0.875 to about 0.57. No error is raised anywhere.

```mermaid
flowchart LR
  subgraph OK["Correct"]
    Q1["'query: ' + question"] --> E1[Model] --> A1[aligned vectors]
    D1["'passage: ' + document"] --> E1
  end
  subgraph BUG["Prefix forgotten at indexing"]
    Q2["'query: ' + question"] --> E2[Model] --> A2[question vectors]
    D2["document (no prefix)"] --> E2 --> B2[drifted document vectors]
  end
```

**Reading it:** the top box is how the model was trained: both sides
stamped, vectors aligned. In the bottom box the documents were indexed
unstamped. The model still returns vectors, and the pipeline runs, but the
two sides no longer line up well. The only way to catch it is to measure
recall on labeled questions, which is the habit this whole lesson argues for.

**In code:** `PrefixedEmbedder` simulates such a model:
`PrefixedEmbedder.encode_one` strips a known prefix and encodes normally, but
partly rotates the vector of any text that lacks one.

**Why it matters:** always read the embedding model's card for required
prefixes or instructions, use the *same* model and settings at indexing and
query time, and keep a small labeled set so silent regressions show up.

## In 20 seconds
- **BM25** scores keyword matches: rare words weigh more (IDF), repeated
  mentions saturate (k₁), long documents are discounted (b). It's great at
  exact IDs and blind to synonyms.
- **Dense retrieval** (bi-encoder) finds meaning and paraphrases, but blurs
  exact tokens. Documents are embedded once, ahead of time.
- **Hybrid search** fuses both rankings with **RRF**, Σ 1/(60 + rank), using
  ranks not scores, and almost always beats either alone.
- **Retrieve, then rerank:** a bi-encoder or hybrid search shortlists 50 to
  100 cheaply; a cross-encoder reads each (question, candidate) pair together
  and reorders the shortlist precisely.
- **ColBERT** keeps a vector per word and scores with MaxSim: close to
  cross-encoder precision, still precomputable, much more storage.
- **Chunk on structure**, keep headings with content, attach metadata, and
  **measure recall@k** on labeled questions: it's how you catch silent bugs
  like a forgotten prefix.

## Self-test questions

**Documents "cat cat dog", "dog bird", "fish"; query "cat". What does BM25
give the first document?**
IDF(cat) = ln(1 + 2.5/1.5) = 0.981. With f = 2, |d| = 3, avgdl = 2, k₁ = 1.5
and b = 0.75 the denominator is 2 + 1.5 · 1.375 = 4.0625, so the score is
0.981 · 2 · 2.5 / 4.0625 = 1.207.

**A document is 1st in dense search and 3rd in BM25. What's its RRF score?**
1/61 + 1/63 ≈ 0.0323, versus 0.0164 for a document that is 1st in only one
list.

**Query words (1, 0) and (0, 1); document words (1, 0) and (0.6, 0.8). What's
the MaxSim score?**
max(1.0, 0.6) + max(0.0, 0.8) = 1.8.

**How many 50-word chunks with 10 words of overlap does a 130-word text make?**
⌈(130 − 10)/(50 − 10)⌉ = 3, starting at words 0, 40 and 80.

**What goes wrong if you forget an embedding model's "passage: " prefix when
indexing?**
Nothing errors: vectors look normal and scores look plausible, but documents
land where the model never aligned them with questions, and recall silently
drops (11 of 12 to 9 of 12 in the lesson's simulation). Only an evaluation
on labeled questions catches it.

**What do BM25's k₁ and b control?**
k₁ sets saturation: how fast extra mentions of a word stop adding score
(the credit per word can never exceed k₁ + 1). b sets length normalization:
0 ignores document length, 1 fully scales mentions by length relative to the
average. Typical values are k₁ ≈ 1.2 to 2 and b = 0.75.

**Bi-encoder vs. cross-encoder: how do you combine them in one pipeline?**
A bi-encoder embeds questions and documents separately, so documents are
embedded once and searched in milliseconds with an ANN index; it's fast but
never sees the pair together. A cross-encoder reads question and document as
one input, so it's accurate but costs one model pass per pair and can't
precompute anything. Combine them: retrieve the top 50 to 100 with the
bi-encoder (or hybrid search), rerank that shortlist with the cross-encoder,
and pass the top few to the model. Cap the shortlist to keep latency
predictable.

**Why does hybrid search beat pure vector search on company data?**
Company data is full of exact identifiers (error codes, product SKUs,
ticket numbers, names) that embeddings blur, and full of paraphrases and
jargon that keyword search misses. BM25 and dense search fail on *different*
questions, so fusing their rankings with RRF recovers the answers each one
misses.

**Why does RRF use ranks instead of scores?**
BM25 scores and cosine similarities live on unrelated scales, and those
scales vary by query and collection. Ranks are always comparable, so RRF
needs no tuning or score normalization. The constant k = 60 keeps the very
top rank from dominating.

**What's a hard negative, and how do you fix one in retrieval?**
A document on the right topic that doesn't answer the question, like the
password-reset how-to for "what are the password rules". Fix it with a
reranker that reads the question and document together, and when you
fine-tune an embedding model, train on hard negatives so it learns the
distinction (see `primer.ml.embeddings.contrastive`).

**What does ColBERT trade to get better precision than a bi-encoder?**
Storage and some query cost. It keeps one vector per word instead of one per
document, often 50 to 200 times more vectors, and scores with MaxSim over
word pairs, in exchange for word-level matching with precomputed documents.

**Fixed-size vs. structure-aware chunking?**
Fixed-size windows are simple but cut through headings, sentences and
tables, separating facts from their context. Structure-aware chunking splits
at headings and paragraphs, keeps each heading with its content, and attaches
metadata. Parent-child retrieval searches small pieces but returns their
larger parent for context.

**Your RAG system gives confident wrong answers. How do you tell whether
retrieval is the cause?**
Build a small labeled set of real questions with known answer passages and
measure recall@k of retrieval alone. If the right passage isn't in the top k,
no prompt change will fix the answer; fix retrieval (hybrid, reranking,
chunking, prefixes, domain fine-tuning). If it is there, the problem is in
generation.

## The papers behind this lesson

- **Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and
  Beyond* (Foundations and Trends in Information Retrieval, 2009).**
  https://www.nowpublishers.com/article/Details/INR-019. The authoritative
  account of where BM25 comes from: the probabilistic model behind IDF, and
  why term-frequency saturation and length normalization take the form they
  do.
- **Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion outperforms Condorcet
  and individual Rank Learning Methods* (SIGIR 2009).**
  https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf. Introduced RRF,
  Σ 1/(k + rank) with k = 60, and showed that this simple, tuning-free fusion
  beats more elaborate methods.
- **Reimers & Gurevych, *Sentence-BERT* (2019).**
  https://arxiv.org/abs/1908.10084. Made bi-encoders practical: a siamese
  BERT that produces one comparable vector per sentence, turning hours of
  pairwise cross-encoding into milliseconds of vector search.
  [annotated companion](../../../papers/sentence-bert.html)
- **Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question
  Answering* (2020).** https://arxiv.org/abs/2004.04906. Showed a dual
  encoder trained with in-batch and BM25-mined hard negatives can beat BM25
  on open-domain question answering, the template for modern dense
  retrievers. [annotated companion](../../../papers/dpr.html)
- **Khattab & Zaharia, *ColBERT: Efficient and Effective Passage Search via
  Contextualized Late Interaction over BERT* (2020).**
  https://arxiv.org/abs/2004.12832. Introduced late interaction: per-token
  vectors scored with MaxSim, getting near cross-encoder quality with
  precomputed documents. [annotated companion](../../../papers/colbert.html)

## Further reading
- Nogueira & Cho, *Passage Re-ranking with BERT* (2019), the cross-encoder reranker: https://arxiv.org/abs/1901.04085
- Santhanam et al., *ColBERTv2* (compressed late interaction): https://arxiv.org/abs/2112.01488
- Wang et al., *Text Embeddings by Weakly-Supervised Contrastive Pre-training* (E5, the query/passage prefixes): https://arxiv.org/abs/2212.03533
- Anthropic, *Introducing Contextual Retrieval* (contextual chunk prefixes + hybrid + reranking): https://www.anthropic.com/news/contextual-retrieval
- sentence-transformers documentation (bi-encoders and cross-encoders): https://www.sbert.net/
- Elasticsearch reciprocal rank fusion reference: https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from primer._show import banner, say, table, takeaway
from primer.common.corpus import DOCS, LABELED_QUERIES, Doc
from primer.common.embedder import WORD_TO_CONCEPT, ConceptEmbedder
from primer.common.text import tokenize
from primer.ml.embeddings.ann import FlatIndex

# ---------------------------------------------------------------------------
# 1. BM25
# ---------------------------------------------------------------------------


class BM25:
    """Okapi BM25 over pre-tokenized documents.

    Args:
        docs_tokens: one list of words per document.
        k1: term-frequency saturation (credit per word never exceeds k1 + 1).
        b: length normalization (0 = ignore length, 1 = fully normalize).
    """

    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs_tokens = [list(d) for d in docs_tokens]
        self.N = len(self.docs_tokens)
        self.doc_len = np.array([len(d) for d in self.docs_tokens], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if self.N else 0.0
        # Term frequencies per document: the f(t, d) in the formula.
        self.tf = [Counter(d) for d in self.docs_tokens]
        # Document frequency n(t): in how many documents does each word appear?
        df: Counter[str] = Counter()
        for d in self.docs_tokens:
            df.update(set(d))
        self.df = dict(df)
        # The "+1 inside the log" (Lucene's variant) keeps IDF positive even for
        # words in more than half the documents; the original could go negative.
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in self.df.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        """BM25 score of every document for the query, shape (N,)."""
        out = np.zeros(self.N)
        # The length-dependent part of the denominator, one value per document.
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / (self.avgdl or 1.0))
        for t in query_tokens:
            idf = self.idf.get(t)
            if idf is None:
                continue  # no document contains t: it contributes nothing
            f = np.array([tf.get(t, 0) for tf in self.tf], dtype=float)
            out += idf * f * (self.k1 + 1) / (f + norm)
        return out

    def search(self, query_tokens: list[str], k: int = 10) -> list[tuple[int, float]]:
        """[(doc index, score)] for the top k documents with a non-zero score."""
        s = self.scores(query_tokens)
        order = [i for i in np.argsort(-s, kind="stable") if s[i] > 0][:k]
        return [(int(i), float(s[i])) for i in order]


# ---------------------------------------------------------------------------
# 2. Reciprocal rank fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse several rankings: each document scores Σ 1/(k + rank). Sorted best first."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


# ---------------------------------------------------------------------------
# 3. One engine, three ways to search
# ---------------------------------------------------------------------------

# Paraphrases share no words with their answers (keyword search's blind spot);
# exact codes are one rare token among many (dense search's blind spot).
PARAPHRASE_QUERIES: list[tuple[str, set[str]]] = [
    ("automobile reimbursement", {"fin-001"}),
    ("notebook computer replacement", {"it-008"}),
    ("holiday allowance", {"hr-001"}),
    ("scam message in my inbox", {"it-007"}),
    ("refund for the hotel", {"fin-002"}),
]
EXACT_CODE_QUERIES: list[tuple[str, set[str]]] = [
    ("ERR-4012", {"it-004"}),
    ("fix ERR-4013", {"it-005"}),
    ("ERR-4013", {"it-005"}),
]
EVAL_QUERIES: list[tuple[str, set[str]]] = LABELED_QUERIES + PARAPHRASE_QUERIES + EXACT_CODE_QUERIES


def doc_text(doc: Doc) -> str:
    """What gets indexed: the title matters, so it's searched along with the body."""
    return f"{doc.title}. {doc.text}"


class SearchEngine:
    """Keyword, dense and hybrid search over the same documents.

    Every method takes a question and returns a ranked list of doc ids, so
    the methods can be compared, evaluated and fused on equal terms.
    """

    def __init__(
        self,
        docs: Sequence[Doc] = DOCS,
        embedder: ConceptEmbedder | None = None,
        depth: int = 10,
        passage_prefix: str = "",
        query_prefix: str = "",
    ):
        self.docs = list(docs)
        self.ids = [d.id for d in self.docs]
        self.depth = depth  # how deep each list goes before fusion
        # Some embedding models expect "passage: " on documents and "query: " on
        # questions. Forgetting either is a silent bug; see PrefixedEmbedder.
        self.passage_prefix, self.query_prefix = passage_prefix, query_prefix
        self.keyword = BM25([tokenize(doc_text(d)) for d in self.docs])
        self.embedder = embedder or ConceptEmbedder()
        self.vectors = FlatIndex(self.embedder.dim)  # 20 docs: exact search is the right index
        self.vectors.add(self.embedder.encode([passage_prefix + doc_text(d) for d in self.docs]))

    def bm25(self, query: str, k: int = 10) -> list[str]:
        return [self.ids[i] for i, _ in self.keyword.search(tokenize(query), k)]

    def dense(self, query: str, k: int = 10) -> list[str]:
        ids, _ = self.vectors.search(self.embedder.encode(self.query_prefix + query), k)
        return [self.ids[i] for i in ids]

    def hybrid(self, query: str, k: int = 10) -> list[str]:
        # Fuse ranks, not scores: BM25 scores are unbounded, cosines live in [-1, 1].
        fused = reciprocal_rank_fusion([self.bm25(query, self.depth), self.dense(query, self.depth)])
        return [doc_id for doc_id, _ in fused[:k]]


def evaluate(search: Callable[[str, int], list[str]], queries: Sequence[tuple[str, set[str]]], k: int = 3) -> dict[str, float]:
    """recall@k (share of questions with an answer in the top k) and MRR.

    Each question here has one relevant document, so recall@k per question
    is 0 or 1. MRR averages 1/rank of the first relevant hit (0 if none in
    the top k).
    """
    hits, rr = 0, 0.0
    for query, relevant in queries:
        ranking = search(query, k)
        first = next((rank for rank, doc_id in enumerate(ranking, 1) if doc_id in relevant), None)
        hits += first is not None
        rr += 1 / first if first else 0.0
    return dict(recall=hits / len(queries), mrr=rr / len(queries))


# ---------------------------------------------------------------------------
# 4. Reranking with a (toy) cross-encoder
# ---------------------------------------------------------------------------


def ideas(text: str) -> list[str]:
    """The distinct 'ideas' in a text: a synonym group if the word has one, else the word itself.

    "password rules" -> ['credential', 'policy']; "ERR-4012" -> ['err-4012'].
    """
    seen: dict[str, None] = {}  # a dict keeps first-seen order, unlike a set
    for tok in tokenize(text):
        seen.setdefault(WORD_TO_CONCEPT.get(tok, tok), None)
    return list(seen)


class CrossEncoder:
    """A toy cross-encoder: it reads the question and one document *together*.

    A real cross-encoder is a transformer that takes "[question] [SEP]
    [document]" as one input, so attention can compare every question word
    with every document word, and it outputs a single relevance score. Ours
    imitates that with features that can only be computed when both texts
    are in hand:

    * coverage: the share of the question's ideas that the document contains
      (does it answer *all* of the question, or just share its topic?)
    * phrase: the share of adjacent question-idea pairs that also sit next to
      each other in the document (word order, which a pooled vector loses)
    * exact: the share of question words that appear verbatim (synonym groups
      are coarse; the literal word "vacation" is stronger evidence than any
      time-off word)
    * cosine: the bi-encoder similarity, as a weak tie-breaker

    What it cannot do, like the real thing, is precompute anything per
    document: every (question, document) pair costs one full scoring pass,
    which `pairs_scored` counts.
    """

    def __init__(self, embedder: ConceptEmbedder | None = None):
        self.embedder = embedder or ConceptEmbedder()
        self.pairs_scored = 0

    def coverage(self, query: str, doc: Doc) -> float:
        q, d = ideas(query), set(ideas(doc_text(doc)))
        return sum(i in d for i in q) / len(q) if q else 0.0

    def phrase(self, query: str, doc: Doc) -> float:
        q = ideas(query)
        pairs = list(zip(q, q[1:]))
        if not pairs:
            return 0.0
        d = ideas(doc_text(doc))
        doc_pairs = set(zip(d, d[1:]))
        return sum(p in doc_pairs for p in pairs) / len(pairs)

    def exact(self, query: str, doc: Doc) -> float:
        q, d = tokenize(query), set(tokenize(doc_text(doc)))
        return sum(t in d for t in q) / len(q) if q else 0.0

    def score(self, query: str, doc: Doc) -> float:
        self.pairs_scored += 1
        cosine = float(self.embedder.encode(query) @ self.embedder.encode(doc_text(doc)))
        return self.coverage(query, doc) + 0.5 * self.phrase(query, doc) + 0.25 * self.exact(query, doc) + 0.25 * cosine


def retrieve_then_rerank(
    engine: SearchEngine, query: str, k: int = 5, shortlist: int = 10, judge: CrossEncoder | None = None
) -> list[str]:
    """Stage 1: cheap, wide hybrid search. Stage 2: expensive, precise reranking of the shortlist."""
    judge = judge or CrossEncoder(engine.embedder)
    by_id = {d.id: d for d in engine.docs}
    candidates = engine.hybrid(query, k=shortlist)
    scored = [(judge.score(query, by_id[c]), -rank, c) for rank, c in enumerate(candidates)]
    # Ties keep the first-stage order (-rank), so reranking never scrambles equals.
    return [c for _, _, c in sorted(scored, reverse=True)[:k]]


# ---------------------------------------------------------------------------
# 5. Late interaction (ColBERT)
# ---------------------------------------------------------------------------


def maxsim(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
    """ColBERT's late-interaction score: for each query word, its best match in the document, summed.

    query_vecs (n_q, d) @ doc_vecs.T (d, n_d) -> (n_q, n_d) word-vs-word
    similarities; max over the document axis; sum over the query axis.
    """
    return float((query_vecs @ doc_vecs.T).max(axis=1).sum())


class LateInteraction:
    """ColBERT-style retrieval: keep one vector per word instead of one per document.

    Documents can still be encoded ahead of time (unlike a cross-encoder),
    but nothing is squeezed into a single pooled vector, so each question
    word gets to find its own best partner in the document.
    """

    def __init__(self, embedder: ConceptEmbedder | None = None):
        self.embedder = embedder or ConceptEmbedder()

    def token_vectors(self, text: str) -> np.ndarray:
        toks = tokenize(text)
        return self.embedder.encode(toks) if toks else np.zeros((0, self.embedder.dim))

    def search(self, query: str, docs: Sequence[Doc], k: int = 10) -> list[str]:
        q = self.token_vectors(query)
        scores = [(maxsim(q, self.token_vectors(doc_text(d))), d.id) for d in docs]
        return [doc_id for _, doc_id in sorted(scores, key=lambda s: -s[0])[:k]]


# ---------------------------------------------------------------------------
# 6. Chunking
# ---------------------------------------------------------------------------

# A longer, multi-section document for the chunking lessons. Markdown
# headings mark the structure a good chunker should respect.
HANDBOOK = """# Remote work handbook

## Eligibility
Most roles can work remotely up to three days a week. Your manager approves the schedule, and roles that
handle physical equipment or visitors may need to be on site more often. Agree your remote days at the
start of each quarter.

## Home internet stipend
Remote staff need a reliable home internet connection. The home internet stipend helps cover the monthly
cost of broadband, fibre or cable service, and home internet upgrades needed for video calls are also
eligible when your current connection is too slow for daily work. Mobile phone plans and hotspot
devices are not eligible. Claim it with the monthly expense report. The stipend is 50 dollars per month.

## Equipment
Everyone receives a laptop, a headset and one external monitor. A second monitor can be requested for
design and engineering roles. Laptop monitors from home are allowed if they connect over USB-C or HDMI.
Return all equipment when you leave the company.

## Security at home
Lock your screen whenever you step away. Use the VPN on any network you do not control, including home
Wi-Fi shared with others. Never print confidential documents at home.

## Working hours
Core hours are 10:00 to 15:00 in your local time zone. Outside core hours, set your status so teammates
know when to expect a reply.
"""


@dataclass(frozen=True)
class Chunk:
    """A piece of a document, plus the metadata that travels with it into the index.

    Metadata is what makes filters possible later: by date, by department,
    and above all by who is allowed to see it (see primer.agents.rag).
    """

    text: str
    source: str
    section: str
    updated: str = ""
    acl: frozenset[str] = field(default_factory=lambda: frozenset({"everyone"}))


def fixed_size_chunks(text: str, size: int = 50, overlap: int = 10) -> list[str]:
    """Split every `size` words, each window overlapping the previous by `overlap` words.

    Simple and structure-blind: a heading can land at the end of one window
    and the fact it introduces at the start of the next. Overlap softens but
    doesn't fix that.
    """
    words = text.split()
    step = size - overlap
    # Stop once a window would start inside the previous window's overlap:
    # the previous window already reached the end.
    starts = range(0, max(1, len(words) - overlap), step)
    return [" ".join(words[s : s + size]) for s in starts]


def _sections(text: str) -> list[tuple[str, str]]:
    """[(heading, body)] for each '## ' section of a markdown document."""
    parts = re.split(r"^## +(.+)$", text, flags=re.M)
    # re.split with a capture group returns [preamble, heading1, body1, heading2, body2, ...]
    return [(parts[i].strip(), parts[i + 1].strip()) for i in range(1, len(parts), 2)]


def structure_aware_chunks(
    text: str, source: str, max_words: int = 120, updated: str = "", acl: frozenset[str] = frozenset({"everyone"})
) -> list[Chunk]:
    """Split on the document's own structure: sections, then paragraphs, never across a heading.

    Each chunk starts with its section heading, so a chunk that says "The
    stipend is 50 dollars per month" still says *which* stipend. (Prefixing
    extra context like this is the idea behind contextual retrieval.)
    """
    chunks = []
    for heading, body in _sections(text):
        paragraphs = [p.replace("\n", " ") for p in body.split("\n\n") if p.strip()]
        current: list[str] = []
        for p in paragraphs:
            if current and len(" ".join(current + [p]).split()) > max_words:
                chunks.append(Chunk(f"{heading}: " + " ".join(current), source, heading, updated, acl))
                current = []
            current.append(p)
        if current:
            chunks.append(Chunk(f"{heading}: " + " ".join(current), source, heading, updated, acl))
    return chunks


def best_chunk(query: str, chunks: Sequence[str]) -> str:
    """The single chunk BM25 ranks highest for the question."""
    index = BM25([tokenize(c) for c in chunks])
    return chunks[index.search(tokenize(query), k=1)[0][0]]


def parent_child_search(query: str, text: str) -> str:
    """Search small children (single sentences), return their parent (the whole section).

    Small chunks match precisely; big chunks give the model enough context
    to answer. Parent-child retrieval gets both.
    """
    children: list[tuple[str, str]] = []  # (sentence, parent section text)
    for heading, body in _sections(text):
        parent = f"## {heading}\n{body}"
        for sentence in re.split(r"(?<=\.)\s+", body.replace("\n", " ")):
            children.append((sentence, parent))
    return dict(children)[best_chunk(query, [s for s, _ in children])]


# ---------------------------------------------------------------------------
# 7. Asymmetric retrieval: query and passage prefixes
# ---------------------------------------------------------------------------


class PrefixedEmbedder(ConceptEmbedder):
    """Simulates an embedding model trained to see "query: " or "passage: " before every input.

    Models such as E5 are trained that way so one network can encode short
    questions and long passages differently. Give it text without the prefix
    and it still returns a perfectly normal-looking unit vector, but from a
    region of space it never learned to align. We simulate that drift by
    partly rotating un-prefixed vectors with a fixed random rotation R:

        v' = cos(θ)·v + sin(θ)·R·v, then rescaled to length 1

    With θ = 60°, half the signal survives, yet nothing errors.
    """

    PREFIXES = ("query: ", "passage: ")

    def __init__(self, drift_degrees: float = 60.0, **kwargs):
        super().__init__(**kwargs)
        theta = np.radians(drift_degrees)
        self._cos, self._sin = np.cos(theta), np.sin(theta)
        # A random orthogonal matrix (a rotation or reflection) from the QR decomposition of noise.
        self._R, _ = np.linalg.qr(np.random.default_rng(7).standard_normal((self.dim, self.dim)))

    def encode_one(self, text: str) -> np.ndarray:
        for prefix in self.PREFIXES:
            if text.startswith(prefix):
                return super().encode_one(text[len(prefix) :])
        v = super().encode_one(text)
        drifted = self._cos * v + self._sin * (self._R @ v)
        return drifted / np.linalg.norm(drifted)


# ---------------------------------------------------------------------------
# 8. Figures
# ---------------------------------------------------------------------------


def _answer_rank(ranking: list[str], relevant: set[str]) -> int | None:
    return next((r for r, d in enumerate(ranking, 1) if d in relevant), None)


def figures() -> dict:
    """Plot this lesson's data. matplotlib is imported here, and only here,
    so the lesson itself needs nothing beyond NumPy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    BM25_C, DENSE_C, HYB_C, MUTED, HOT, GOOD = "#d97706", "#2563eb", "#7c3aed", "#9ca3af", "#dc2626", "#059669"
    figs = {}

    # --- 1. BM25: saturation and length normalization ----------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    f = np.arange(0, 21)
    a1.plot(f, f, "--", color=MUTED, label="raw count")
    for k1, shade in [(0.5, 0.45), (1.2, 0.7), (1.5, 1.0), (3.0, 0.3)]:
        # With b = 0, BM25's per-word credit is f·(k1+1)/(f+k1): plot it straight from the class.
        index = BM25([["w"] * int(n) for n in f[1:]] + [["x"]], k1=k1, b=0.0)
        credit = [0.0] + [index.scores(["w"])[i] / index.idf["w"] for i in range(len(f) - 1)]
        a1.plot(f, credit, color=BM25_C, alpha=shade, lw=2, label=f"k₁ = {k1} (ceiling {k1 + 1})")
    a1.set_ylim(0, 6)
    a1.set_xlabel("mentions of the word in the document, f(t, d)")
    a1.set_ylabel("credit (before the rarity weight)")
    a1.set_title("Saturation: the 10th mention adds little")
    a1.legend(frameon=False, fontsize=8)
    lengths = np.linspace(0.25, 3, 60)  # |d| / avgdl
    for b, shade in [(0.0, 0.35), (0.5, 0.65), (0.75, 1.0), (1.0, 0.5)]:
        credit = 1 * 2.5 / (1 + 1.5 * (1 - b + b * lengths))  # one mention, k1 = 1.5
        a2.plot(lengths, credit, color=BM25_C, alpha=shade, lw=2, label=f"b = {b}")
    a2.axvline(1, color=MUTED, ls=":")
    a2.text(1.03, 1.55, "average length", color=MUTED, fontsize=8)
    a2.set_xlabel("document length ÷ average length, |d| / avgdl")
    a2.set_ylabel("credit for one mention")
    a2.set_title("Length normalization: long documents are discounted")
    a2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    figs["bm25_curves"] = fig

    engine = SearchEngine(DOCS)

    # --- 2. RRF on the error-code question ----------------------------------
    q = "what does ERR-4012 mean"
    b_rank, d_rank = engine.bm25(q, 10), engine.dense(q, 10)
    fused = reciprocal_rank_fusion([b_rank, d_rank])[:6]
    names = [d for d, _ in fused]
    from_b = [1 / (60 + b_rank.index(d) + 1) if d in b_rank else 0 for d in names]
    from_d = [1 / (60 + d_rank.index(d) + 1) if d in d_rank else 0 for d in names]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    x = np.arange(len(names))
    ax.bar(x, from_b, color=BM25_C, label="from BM25: 1/(60 + rank)")
    ax.bar(x, from_d, bottom=from_b, color=DENSE_C, label="from dense: 1/(60 + rank)")
    for xi, d in zip(x, names):
        tags = [f"B{b_rank.index(d) + 1}" if d in b_rank else "", f"D{d_rank.index(d) + 1}" if d in d_rank else ""]
        ax.text(xi, from_b[xi] + from_d[xi] + 0.0006, " ".join(t for t in tags if t), ha="center", fontsize=8)
    ax.set_ylim(0, max(a + b for a, b in zip(from_b, from_d)) * 1.2)  # room for the rank labels
    ax.set_xticks(x, names)
    ax.set_ylabel("fused RRF score")
    ax.set_title(f'RRF for "{q}" (B = BM25 rank, D = dense rank)')
    ax.legend(frameon=False, fontsize=8)
    figs["rrf_fusion"] = fig

    # --- 3. Where each method ranks the answer ------------------------------
    methods = {
        "BM25": engine.bm25,
        "dense": engine.dense,
        "hybrid": engine.hybrid,
        "hybrid + rerank": lambda qq, k: retrieve_then_rerank(engine, qq, k),
    }
    grid = np.array(
        [[_answer_rank(m(qq, 10), rel) or 11 for m in methods.values()] for qq, rel in EVAL_QUERIES], dtype=float
    )
    fig, ax = plt.subplots(figsize=(7.2, 8))
    ax.imshow(np.minimum(grid, 11), cmap="RdYlGn_r", vmin=1, vmax=11, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, "✗" if grid[i, j] > 10 else int(grid[i, j]), ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(methods)), list(methods))
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(EVAL_QUERIES)), [qq for qq, _ in EVAL_QUERIES], fontsize=8)
    n1, n2 = len(LABELED_QUERIES), len(LABELED_QUERIES) + len(PARAPHRASE_QUERIES)
    for y in (n1 - 0.5, n2 - 0.5):
        ax.axhline(y, color="black", lw=1.5)
    ax.text(3.6, (n1 + n2) / 2 - 0.5, "paraphrases", rotation=-90, va="center", fontsize=8)
    ax.text(3.6, (n2 + len(EVAL_QUERIES)) / 2 - 0.5, "exact codes", rotation=-90, va="center", fontsize=8)
    ax.set_title("Rank of the right answer (1 = top, ✗ = not in top 10)", pad=28)
    fig.tight_layout()
    figs["method_ranks"] = fig

    # --- 4. MaxSim grid ------------------------------------------------------
    li = LateInteraction()
    qtoks = tokenize("what are the password rules")
    policy = next(d for d in DOCS if d.id == "it-002")
    dtoks = tokenize(doc_text(policy))[:14]
    sims = li.token_vectors(" ".join(qtoks)) @ li.embedder.encode(dtoks).T
    fig, ax = plt.subplots(figsize=(8.4, 2.6))
    im = ax.imshow(sims, cmap="Blues", vmin=-0.2, vmax=1)
    for i, j in enumerate(sims.argmax(1)):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=HOT, lw=2.5))
        ax.text(j, i, f"{sims[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white")
    ax.set_xticks(range(len(dtoks)), dtoks, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(qtoks)), qtoks)
    ax.set_title(f"MaxSim: each query word keeps its best match (score = {sims.max(1).sum():.2f})")
    fig.colorbar(im, ax=ax, fraction=0.02, label="similarity")
    figs["maxsim_grid"] = fig

    # --- 5. Chunk boundaries on the handbook --------------------------------
    words = HANDBOOK.split()
    section_starts = [i for i, w in enumerate(words) if w == "##"]
    headings = [h for h, _ in _sections(HANDBOOK)]
    bounds = section_starts + [len(words)]
    heading_at = next(i for i in range(len(words)) if words[i : i + 3] == ["Home", "internet", "stipend"])
    amount_at = next(i for i in range(len(words)) if words[i : i + 2] == ["50", "dollars"])
    fig, ax = plt.subplots(figsize=(10, 3))
    cmap = plt.get_cmap("Pastel1")
    for n, (s, e, heading) in enumerate(zip(bounds, bounds[1:], headings)):
        ax.add_patch(Rectangle((s, -0.3), e - s, 0.6, color=cmap(n)))
        ax.text((s + e) / 2, 0, heading, ha="center", va="center", fontsize=7)
    size, overlap = 40, 5
    for n, s in enumerate(range(0, max(1, len(words) - overlap), size - overlap)):
        y = 0.55 + 0.18 * (n % 2)  # alternate heights so overlapping windows stay visible
        ax.plot([s, min(s + size, len(words))], [y, y], color=MUTED, lw=3)
    for n, (s, e) in enumerate(zip(bounds, bounds[1:])):
        y = -0.55 - 0.18 * (n % 2)
        ax.plot([s, e], [y, y], color=HYB_C, lw=3)
    ax.axvline(heading_at, color=HOT, lw=2)
    ax.axvline(amount_at, color=GOOD, lw=2)
    ax.text(heading_at, 1.02, "heading", color=HOT, ha="center", fontsize=8)
    ax.text(amount_at, 1.02, "50 dollars", color=GOOD, ha="center", fontsize=8)
    ax.text(-2, 0.63, "fixed\n40-word\nwindows", ha="right", va="center", fontsize=8, color="#4b5563")
    ax.text(-2, -0.63, "structure-\naware\nchunks", ha="right", va="center", fontsize=8, color=HYB_C)
    ax.set_xlim(-30, len(words) + 2)
    ax.set_ylim(-1.0, 1.15)
    ax.set_yticks([])
    ax.set_xlabel("word position in the handbook")
    ax.set_title("Where the chunkers cut: the stipend heading and its amount")
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    figs["chunk_boundaries"] = fig

    return figs


# ---------------------------------------------------------------------------
# 9. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. BM25 by hand: three tiny documents, query 'cat'")
    toy = [["cat", "cat", "dog"], ["dog", "bird"], ["fish"]]
    index = BM25(toy)
    table(
        ["doc", "words", "BM25('cat')"],
        [(f"d{i + 1}", " ".join(d), s) for i, (d, s) in enumerate(zip(toy, index.scores(["cat"])))],
        floatfmt=".3f",
    )
    say(f"IDF(cat) = ln(1 + 2.5/1.5) = {index.idf['cat']:.3f}; IDF(dog) = {index.idf['dog']:.3f}: rarer words weigh more.")

    engine = SearchEngine(DOCS)
    banner("2. The two librarians disagree")
    for q in ("what does ERR-4012 mean", "automobile reimbursement"):
        print(f"{q!r}")
        print(f"   BM25   : {engine.bm25(q, 3) or '(nothing: no shared words)'}")
        print(f"   dense  : {engine.dense(q, 3)}")
        print(f"   hybrid : {engine.hybrid(q, 3)}")
    print()

    banner("3. Reciprocal rank fusion, worked example (k = 60)")
    fused = dict(reciprocal_rank_fusion([["X", "a", "b"], ["c", "d", "X"], ["Y"]]))
    say(f"X at ranks 1 and 3: 1/61 + 1/63 = {fused['X']:.4f}. Y at rank 1 only: 1/61 = {fused['Y']:.4f}.")

    banner("4. Measured on 20 labeled questions")
    rows = []
    for name, fn in [
        ("BM25", engine.bm25),
        ("dense", engine.dense),
        ("hybrid (RRF)", engine.hybrid),
        ("hybrid + rerank", lambda q, k: retrieve_then_rerank(engine, q, k)),
        ("late interaction", lambda q, k: LateInteraction().search(q, DOCS, k)),
    ]:
        r = evaluate(fn, EVAL_QUERIES, k=3)
        rows.append((name, r["recall"], r["mrr"]))
    table(["method", "recall@3", "MRR"], rows, floatfmt=".3f")
    takeaway("BM25 and dense search fail on different questions; fusing them with RRF fixes both.")

    banner("5. A hard negative, fixed by reranking")
    q = "what are the password rules"
    say(f"{q!r}: hybrid ranks {engine.hybrid(q, 3)}, but the answer is it-002 (the policy).")
    judge = CrossEncoder()
    by_id = {d.id: d for d in DOCS}
    table(
        ["doc", "coverage", "phrase", "exact", "score"],
        [
            (d, judge.coverage(q, by_id[d]), judge.phrase(q, by_id[d]), judge.exact(q, by_id[d]), judge.score(q, by_id[d]))
            for d in ("it-002", "it-001")
        ],
        floatfmt=".2f",
    )
    say(f"After reranking: {retrieve_then_rerank(engine, q, 3)}.")

    banner("6. Chunking the remote-work handbook")
    fixed = best_chunk("home internet stipend", fixed_size_chunks(HANDBOOK, size=40, overlap=5))
    smart = best_chunk("home internet stipend", [c.text for c in structure_aware_chunks(HANDBOOK, "remote-work-handbook")])
    say(f"Best fixed-size chunk: ...{fixed[-110:]}  (amount present: {'50 dollars' in fixed})")
    say(f"Best structure-aware chunk: ...{smart[-110:]}  (amount present: {'50 dollars' in smart})")

    banner("7. The forgotten prefix: a silent bug")
    for label, pp in [("both prefixes", "passage: "), ("passage prefix forgotten", "")]:
        e = SearchEngine(DOCS, embedder=PrefixedEmbedder(), passage_prefix=pp, query_prefix="query: ")
        r = evaluate(e.dense, LABELED_QUERIES, k=3)
        print(f"{label:26s} recall@3 {r['recall']:.3f}   MRR {r['mrr']:.3f}")
    print()
    takeaway("Nothing errored. Only measuring recall on labeled questions reveals the bug.")


if __name__ == "__main__":
    demo()
