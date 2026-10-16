r"""
# Retrieval: keyword search, meaning search, and combining them

Run: `python -m primer.ml.embeddings.retrieval`

New to the notation (Σ, log, |d|)? `primer.notation` explains every symbol
used here from zero.

## Level 1: The practitioner's guide

**In one sentence.** Retrieval finds the few passages that answer a
question, ranked best first, by running a keyword search and a meaning
search side by side, fusing their rankings, and letting a slower, more
careful model reorder the short list that survives.

**When you need it.** You need retrieval whenever a model must answer from
material it was not trained on and that material is too big to paste into
the prompt: a knowledge base of tickets, manuals, contracts or code.
Anthropic's contextual retrieval post puts the line at about 200,000 tokens
(about 500 pages): below that, put the whole knowledge base in the prompt
and skip retrieval. Above it, retrieval quality caps answer quality, because
the language model can only use what retrieval hands it (`primer.agents.rag`).
The tell that one search method is not enough: on this lesson's 20 labeled
questions, keyword search alone puts the answer in the top 3 for 75% of them
and meaning search alone for 90%, and their misses never overlap. Keyword
search fails on paraphrases ("automobile reimbursement" when the document
says "car" and "reimbursed"); meaning search puts the article for "ERR-4012"
fourth. Fused, they reach 100%.

**Your options.** From the cheapest to the most precise; in practice each
stage feeds the next:

| Option | What it does | What it gives you | What it costs | Where it lives |
|---|---|---|---|---|
| Keyword search (BM25) | Weighs each query word by rarity, saturates repeated mentions, discounts long documents | Exact matches on identifiers, codes and names, with no training | Nothing for synonyms: a document sharing no words scores 0 | Lucene, Elasticsearch, OpenSearch, any search engine |
| Dense retrieval (bi-encoder) | Embeds every document once and the question at query time, then returns the nearest vectors | Paraphrases, synonyms, other languages | An embedding model and a vector index; exact tokens blur | An embedding model plus a vector index (`primer.ml.embeddings.ann`) |
| Hybrid search (RRF) | Runs both and fuses the two rankings by position, never by score | The strengths of both, with no tuning | Two searches per query and a merge | Built into Elasticsearch and most vector databases |
| Reranking (cross-encoder) | Reads the question and each shortlisted candidate together and reorders them | Catches hard negatives: right topic, wrong answer | One model pass per candidate, so it only ever sees a shortlist | sentence-transformers cross-encoders, Cohere Rerank |
| Late interaction (ColBERT) | Keeps one vector per word and matches each question word to its best document word | Near cross-encoder precision with precomputed documents | 50 to 200 times the vector storage | ColBERT and ColBERTv2 |

**How to choose.** Start from what your questions look like and how much
latency you can spend.

- Questions full of identifiers (error codes, product codes, ticket
  numbers, names): keyword search is hard to beat and must be in the
  pipeline. Company data is full of these.
- Questions phrased differently from the documents (paraphrases, jargon,
  other languages): dense retrieval. Read the model card for required
  prefixes (`query: ` and `passage: ` for the E5 family) and use the same
  model and settings at indexing and query time.
- Almost always: both, fused with reciprocal rank fusion at its standard
  constant of 60. It is cheap, needs no tuning, and on this lesson's
  questions lifts recall@3 from 0.75 and 0.90 to 1.00.
- When the top results are on the right topic but don't answer the question
  (a password *reset* guide above the password *policy*): add a reranker
  over a shortlist of 50 to 100, and measure recall and MRR with and without
  it. Rerankers can demote a right answer too: this lesson's fixes the hard
  negative and lowers MRR from 1.00 to 0.975 on the same 20 questions.
- When one vector per document is too coarse and you can afford the
  storage: late interaction.
- Whatever you pick, decide the chunking first: split on structure, keep
  headings with their content, add modest overlap, attach metadata. It often
  matters more than the choice of embedding model. Then keep a small labeled
  set of questions with known answer passages and measure recall@k on it.

**What it costs.** Keyword and dense search are the cheap, wide stages:
documents are indexed once, and a question costs one embedding plus a
lookup that takes milliseconds over millions of documents. Fusion is a merge
of two short lists. The reranker is where money and latency go: at 10 ms per
pair on a GPU, scoring a million documents takes 10,000 seconds, and a
shortlist of 50 takes 0.5 s (less when the pairs are batched), which is why
the shortlist is capped. Late interaction trades that latency for storage:
one vector per word instead of one per document. Preparing the chunks has a
price too: Anthropic reports \$1.02 per million document tokens, once, to
write a short context in front of every chunk with prompt caching, for a 49%
cut in top-20 retrieval failures with hybrid search and a 67% cut with
reranking added (from 5.7% to 1.9% on their evaluation). The cheapest item
of all, and the one most teams skip, is a labeled set of a few dozen
questions with known answer passages: it is the only instrument that shows
the failures below.

**What breaks.**

- **The right page is never found.** If the passage is not in the top k, no
  prompt change will fix the answer. Measure recall@k of retrieval alone
  before touching the prompt.
- **A forgotten prefix.** Index documents without the `passage: ` prefix a
  model was trained with and nothing errors: vectors look normal, scores
  look plausible, and in this lesson's simulation recall@3 falls from 0.92
  to 0.75 and MRR from 0.875 to 0.57. Only a labeled set catches it.
- **Scores added across systems.** BM25 scores run from 0 to about 20 and
  cosine similarities from −1 to 1; add them and one system drowns the
  other. Fuse ranks, not scores.
- **Hard negatives.** A document on the right topic that doesn't answer the
  question rises because it shares the words. A reranker reads the pair
  together; when you fine-tune an embedding model, train it on such pairs
  (`primer.ml.embeddings.contrastive`).
- **A heading cut from its fact.** Fixed 40-word windows split the heading
  "Home internet stipend" from "50 dollars per month" in this lesson's
  handbook; a structure-aware chunk holds both. Parent-child retrieval
  searches small children and returns their whole section.
- **A reranker trusted blindly.** It is a model and can be wrong: this
  lesson's demotes one correct answer while fixing another. Ship it only
  when the numbers say so.
- **A shortlist too short.** Give the reranker only the top result and it
  cannot help. Anthropic's evaluation retrieved 150 chunks, reranked them
  to 20, and found that passing 20 chunks to the model beat passing 10 or 5.

**In the wild.** BM25 runs in any search engine: Lucene, Elasticsearch,
OpenSearch. Elasticsearch's `rrf` retriever fuses a BM25 query with a kNN
query by the same one-over-sixty-plus-rank rule (`rank_constant` 60 by
default) with no weights to tune. Bi-encoders trace back to Sentence-BERT
and Dense Passage Retrieval; the sentence-transformers library ships both
bi-encoders and cross-encoder rerankers, and hosted rerankers such as Cohere
Rerank take a query and a list of documents and return them ordered by
relevance, cut to a `top_n`. Cross-encoder reranking with BERT is due to
Nogueira and Cho, late interaction to ColBERT and ColBERTv2, and the E5
models are the ones trained with the `query: ` and `passage: ` prefixes.
Anthropic's contextual retrieval puts a generated context in front of every
chunk and combines it with hybrid search and reranking. The papers behind
this lesson are listed at the end with their companions.

**Go deeper.** Level 2 builds every box of the pipeline by hand: BM25 on
three one-line documents, the bi-encoder over this repo's toy embedder,
reciprocal rank fusion on two ranks, a toy cross-encoder scoring a hard
negative, ColBERT's MaxSim on two-number word vectors, the chunking
arithmetic, and the forgotten-prefix bug measured. If you only needed to
design the pipeline, you are done.

## Level 2: How it works, from scratch

The examples search a small company knowledge base: twenty help articles
from IT, finance and HR, such as "Password policy", "Car mileage expense"
and "Error ERR-4012: VPN tunnel failed". Twenty test questions come with
their known answers, so every method can be scored the same way. Two terms
are used throughout:

- **Token.** Here, one word after lowercasing and dropping filler words
  ("the", "is").
- **Recall@k.** The share of questions whose answer appears in the top k
  results. **MRR** (mean reciprocal rank) also rewards putting the answer
  *first*: an answer at rank 1 scores 1, at rank 2 scores 1/2, missing scores
  0, averaged over questions.

The full pipeline, one box per section below:

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

**Reading it:** two cheap, wide searches run side by side over the whole
collection, and their rankings are merged into one shortlist. Only the
shortlist reaches the reranker on the right, which is slow but precise.
Everything left of the shortlist must be fast, because it faces millions of
documents; everything right of it can be slow, because it faces a hundred at
most.

## 1. Keyword search: BM25

Keyword search scores a document by the words it shares with the question.
**BM25**, the standard scoring rule, refines plain counting in three ways: a
rare word counts for more than a common one, each extra mention of a word
adds less than the one before, and a mention in a long document counts for a
little less than the same mention in a short one.

Take three one-line documents:

| Doc | Words | Length |
|---|---|---|
| d₁ | cat cat dog | 3 |
| d₂ | dog bird | 2 |
| d₃ | fish | 1 |

The average length is (3 + 2 + 1)/3 = 2. Search for **cat**. It appears in
one of the three documents, so it is rare and gets a high weight (0.981,
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
the same mention in a long one. A word the document doesn't contain
contributes zero, so a document that shares no words with the query scores
0, however close its meaning.

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
import math
# 3 documents; 1 of them contains "cat"
N, n_t = 3, 1
# ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))
IDF = math.log(1 + (N - n_t + 0.5) / (n_t + 0.5))
round(IDF, 3)  # → 0.981
# f(cat, d1), |d1|, average document length
f, d_len, avgdl = 2, 3, 2
k_1, b = 1.5, 0.75
count_part = f * (k_1 + 1) / (f + k_1 * (1 - b + b * d_len / avgdl))
round(count_part, 3)  # → 1.231
# Σ over the query's only word
round(IDF * count_part, 3)  # → 1.207
```

![BM25 credit flattens toward k1 + 1 however often a word repeats, and one mention counts less in a longer document once b is above 0](figures/primer.ml.embeddings.retrieval.bm25_curves.svg)

**Reading it:** on the left, the x-axis is how often a word appears in a
document and the y-axis is the credit BM25 gives it (before the rarity
weight). The dashed line is raw counting, where 20 mentions count 20 times
as much as one. The BM25 curves bend over and flatten toward k₁ + 1: the
tenth mention adds almost nothing, which stops keyword-stuffed pages from
winning. Smaller k₁ flattens sooner. On the right, one mention in documents
of different lengths: with b = 0 length is ignored; with b = 0.75 a mention
in a document twice the average length counts noticeably less.

**In code:** `BM25.scores` applies the formula to every document.

BM25 needs no training and runs in any search engine (Elasticsearch,
OpenSearch, Lucene). It is hard to beat on exact identifiers such as product
codes, error numbers, names and ticket IDs, and company data is full of
them. Its blind spot is the paraphrase. Ask the help articles about
"automobile reimbursement" and keyword search finds nothing, because the
article that answers it says "car" and "reimbursed".

## 2. Meaning search: dense retrieval with a bi-encoder

Meaning search finds that article. An embedding model turns every document
into a card of numbers, its embedding, before any question arrives. When a
question comes in, the same model makes a card for it, and the search
returns the documents whose cards are closest. Searching this way is called
**dense retrieval**.

With a small toy embedding model, the question "automobile reimbursement"
and the car-mileage article share no words, yet their cards have a cosine
similarity of about 0.8, the highest in the collection. Exact codes go the
other way. For "What does ERR-4012 mean" the right article comes only 4th:
"what" and "does" are dropped as filler, the code is one of the two words
left, and "mean" pulls the card elsewhere.

```mermaid
flowchart LR
  subgraph AHEAD["Ahead of time"]
    D[Each document] --> E1[Encoder] --> V[(Document vectors)]
  end
  subgraph NOW["At question time"]
    Q[Question] --> E2[Same encoder] --> QV[Question vector]
  end
  QV --> NN["Nearest vectors<br/>(a vector index)"]
  V --> NN
  NN --> R[Ranked documents]
```

**Reading it:** the encoder runs twice, but never on the question and a
document *together*, which is why this design is called a **bi-encoder**.
Documents are encoded once, when they are added. A question costs one
encoding plus a nearest-neighbor lookup (see `primer.ml.embeddings.ann`),
which takes milliseconds over millions of documents. Because the model never
sees the question and the document side by side, it cannot check whether a
passage answers *this* question or only shares its topic.

**In code:** `SearchEngine.dense` embeds the question and returns the
nearest documents.

Dense retrieval handles paraphrases, synonyms and other languages, and blurs
exact tokens; keyword search does the reverse. Measure both kinds of
question on your own data before trusting either search alone.

## 3. Hybrid search: reciprocal rank fusion (RRF)

Each search finds answers the other misses, so **hybrid search** runs both
and merges their two ranked lists. The usual merge is **reciprocal rank
fusion** (RRF). Each list votes for the documents on it, a higher place
earning a bigger vote, and a document's votes are added up. Only positions
are compared, never the two searches' scores, which are not on the same
scale.

With the standard constant k = 60:

| Document | Dense rank | BM25 rank | RRF score |
|---|---|---|---|
| X | 1 | 3 | 1/61 + 1/63 = 0.0164 + 0.0159 = **0.0323** |
| Y | 1 | (absent) | 1/61 = **0.0164** |

X, high on both lists, beats Y, first on only one.

```mermaid
flowchart LR
  Q[Question] --> B[BM25 ranking]
  Q --> D[Dense ranking]
  B --> R["For every document:<br/>add 1/(60 + rank)<br/>from each list it's on"]
  D --> R
  R --> O[Sort by total:<br/>the fused ranking]
```

**Reading it:** the two rankings are the only inputs; their raw scores are
thrown away on purpose. BM25 scores run from 0 to about 20, cosine
similarities from −1 to 1, so adding them would let one search drown the
other, and fixing that would need tuning for each collection. Ranks are on
the same scale for every search, so RRF needs no tuning.

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
k = 60
# d's position in each ranking that contains it
def RRF(ranks):
    # Σ_i 1/(k + rank_i(d))
    return sum(1 / (k + rank_i) for rank_i in ranks)
# X: 1st in one ranking, 3rd in the other
round(RRF([1, 3]), 4)  # → 0.0323
# Y: on one ranking only
round(RRF([1]), 4)  # → 0.0164
```

![For the query what does ERR-4012 mean, the error-code article collects credit from both BM25 and dense search and fuses to a clear first place](figures/primer.ml.embeddings.retrieval.rrf_fusion.svg)

**Reading it:** each bar is one document's fused score, split into the part
contributed by BM25 (orange) and by dense search (blue). The error-code
article it-004 is BM25's only hit and dense search's 4th, and the two
contributions stack up to put it clearly first. The password-policy article
that dense search wrongly ranked first gets only its blue half.

![BM25 misses the paraphrase questions and dense search misses the error codes, but their misses never overlap, so the hybrid ranks every answer first](figures/primer.ml.embeddings.retrieval.method_ranks.svg)

**Reading it:** one row per question, one column per method; the number is
the rank at which the right answer appeared (✗ = not in the top 10). Look at
where the colors differ. BM25's failures (the paraphrase block) are where
dense search succeeds, and dense search's failures (the error-code block) are
where BM25 succeeds. Their mistakes don't overlap, so the hybrid column is
all 1s.

| Method | recall@3 (20 questions) | MRR |
|---|---|---|
| BM25 alone | 0.75 | 0.75 |
| Dense alone | 0.90 | 0.85 |
| **Hybrid (RRF)** | **1.00** | **1.00** |

**In code:** `reciprocal_rank_fusion` fuses any number of rankings, and
`evaluate` measures recall@k and MRR for any search method.

Company data is full of both paraphrases and exact identifiers, so there
too hybrid search usually beats either method alone. Elasticsearch and most
vector databases offer it built in.

## 4. Reranking: bi-encoder vs. cross-encoder

Hybrid search ranks every test question's answer first, but a document on
the right topic can still fool it. Ask "What are the password rules?" Two
articles are about passwords. The password *reset* how-to (it-001) mentions
"password" three times; the password *policy* (it-002) is the real answer.
BM25 and the fused hybrid ranking both put the how-to first. A document like
this, which looks relevant but doesn't answer the question, is called a
**hard negative**.

Meaning search cannot settle it, because each document's card was made
without seeing the question. A **cross-encoder** reads the question and a
document together, as one input, and scores the pair. That is more accurate,
and more expensive: every pair needs its own pass through the model, so a
cross-encoder only reorders a shortlist that a cheaper search has already
found. This step is **reranking**.

A toy cross-encoder shows the idea. It checks which of the question's
*ideas* the document covers:

| | question ideas: {password, rules} | coverage | phrase | exact | score |
|---|---|---|---|---|---|
| it-002 policy | password ✓, policy/rules ✓ | 2/2 = **1.0** | 1.0 | 0.5 | ≈ 1.78 |
| it-001 reset how-to | password ✓, rules ✗ | 1/2 = **0.5** | 0.0 | 0.5 | ≈ 0.69 |

The score weighs the features: coverage + 0.5 × phrase + 0.25 × exact +
0.25 × cosine, where the cosine is the ordinary bi-encoder similarity (0.63
for the policy, 0.27 for the how-to). For the policy that's 1.0 + 0.5 +
0.125 + 0.16 ≈ 1.78; for the how-to, 0.5 + 0 + 0.125 + 0.07 ≈ 0.69.
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

The cost decides how the two kinds of model are combined. Say a
cross-encoder pass takes 10 ms on a GPU. Over a collection of a million
documents that is 10,000 seconds per question; over a shortlist of 50 it is
0.5 s, or much less when the pairs are batched. So the cheap searches
retrieve wide, the cross-encoder reranks narrow, and the shortlist is capped
to keep latency predictable.

**In code:** `retrieve_then_rerank` shortlists with hybrid search, then
reorders the shortlist with `CrossEncoder.score`.

A reranker is a model, and it can be wrong too. On the twenty test
questions, where hybrid search already ranks every answer first, it demotes
one: for "refund for the hotel" it prefers the car-mileage article, which
talks about reimbursing trips. Measure recall@k and MRR with and without the
reranker before shipping it.

## 5. Late interaction: ColBERT

A bi-encoder is fast because every document's card is made in advance, and
coarse because a whole document is squeezed into one card. A cross-encoder
is precise and can prepare nothing in advance. **Late interaction**,
introduced with the ColBERT model, sits between them. It keeps a card for *every word*
of a document, made in advance. When a question arrives, each of its words
looks for its single best-matching word in the document, and the document's
score adds up how well the question's words found partners.

Take two-number word vectors. The query has words (1, 0) and (0, 1); the
document has words (1, 0) and (0.6, 0.8).

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
# one vector per query word
q = [(1, 0), (0, 1)]
# one vector per document word
d = [(1, 0), (0.6, 0.8)]
def dot(a, b):
    return sum(a_k * b_k for a_k, b_k in zip(a, b))
# each query word's best match
[max(dot(q_i, d_j) for d_j in d) for q_i in q]  # → [1, 0.8]
# Σ_i max_j q_i · d_j
sum(max(dot(q_i, d_j) for d_j in d) for q_i in q)  # → 1.8
```

![In ColBERT's MaxSim grid for password rules, password best matches itself at 1.00 and rules best matches policy at 0.85, for a score of 1.85](figures/primer.ml.embeddings.retrieval.maxsim_grid.svg)

**Reading it:** rows are the query's words, columns the document's words,
and color is the similarity of each pair. The outlined cell in each row is
that row's maximum, the only number that counts. "password" finds itself
(1.00), ignoring the near-identical "passwords"; "rules" finds "policy"
(0.85), a synonym it could never match by spelling. The score is the sum of
the outlined cells: 1.00 + 0.85 = 1.85.

**In code:** `LateInteraction.search` ranks documents by `maxsim`.

Late interaction gets much of a cross-encoder's precision while keeping
documents precomputed. The cost is storage: one vector per *word* instead of
one per document, often 50 to 200 times more. ColBERTv2 compresses those
vectors to make that affordable.

## 6. Chunking: how documents are cut before indexing

Every search so far ranked whole help articles a few sentences long. A
longer document is cut into pieces called **chunks** before it is indexed,
and each chunk is embedded and searched on its own. Where the cuts fall
decides whether a fact stays next to the heading that explains it.

The simplest cut is a fixed-size window. Cut a 130-word text into 50-word
chunks with 10 words of overlap, and a window starts every 50 − 10 = 40
words, at 0, 40 and 80, giving **3 chunks** (words 1–50, 41–90, 81–130).
Each shares 10 words with the next, so a sentence no longer than the overlap
that is cut at one boundary survives whole in the next chunk.

Fixed windows ignore the document's structure. Take a short remote-work
handbook and the question "home internet stipend". With 40-word fixed
chunks, the best-matching chunk holds the heading "Home internet stipend",
but the amount, "50 dollars per month", fell into the next window. With
**structure-aware** chunks, cut at headings, the best chunk holds both.

```mermaid
flowchart LR
  DOC[Document] --> P[Parse: headings,<br/>paragraphs, tables]
  P --> SPLIT["Split at structure;<br/>merge small pieces<br/>up to a size limit"]
  SPLIT --> CTX["Prefix each chunk with<br/>its heading (context)"]
  CTX --> META["Attach metadata: source,<br/>section, date, access list"]
  META --> EMB[Embed + index]
```

**Reading it:** chunking takes several steps. Parsing decides what the
structure *is* (and is where most quality is lost on messy
PDFs). Splitting respects that structure. Prefixing the heading means a
chunk that says "The stipend is 50 dollars" still says *which* stipend: the
idea behind *contextual retrieval*. Metadata rides along so later stages can
filter by date or by who's allowed to see it (see `primer.agents.rag`).

![Fixed 40-word windows split the stipend heading from its amount and cut through sections, while one structure-aware chunk holds both](figures/primer.ml.embeddings.retrieval.chunk_boundaries.svg)

**Reading it:** the colored strip in the middle is the handbook, one color
per section, read left to right by word position. The brackets above it are
the fixed 40-word windows; those below are the structure-aware chunks. The
red marker is the "Home internet stipend" heading and the green marker the
"50 dollars" amount. Above the strip they sit in different windows; below
it, one chunk spans both. Notice also that the fixed windows cut straight
through section boundaries.

**Try it:** the handbook below is cut into fixed windows, searched with
hybrid search and reranked by the toy cross-encoder described above. With
the stipend question and 40-word chunks, retrieval ranks chunk 2 (the
heading) first, and reranking lifts chunk 3, which holds the amount. Now pick
the VPN question at 30 words with no overlap: the answer sentence is cut
across chunks 6 and 7; raise the overlap to 10 words and one chunk holds it
whole again. Drag top k down to 1 and watch the reranker lose its chance to
help.

<div class="viz" data-viz="rag-pipeline" aria-label="Chunking, retrieval and reranking explorer"></div>

**Parent-child retrieval.** Small chunks match precisely; big chunks give the
model enough context. Parent-child retrieval gets both. It indexes small
*children* (single sentences), and when one matches, it hands the model the
child's *parent* (the whole section).

```mermaid
flowchart LR
  Q[Question] --> C["Search small children<br/>(sentences)"]
  C --> HIT[Best child]
  HIT --> PARENT["Look up its parent<br/>(the whole section)"]
  PARENT --> LLM[Give the parent<br/>to the model]
```

**Reading it:** the search happens on the left, at sentence level, where
matches are sharp. The hand-over happens on the right, at section level,
where context is complete. The lookup in the middle maps each child to its
parent.

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
import math
# words, chunk size, overlap
n, s, o = 130, 50, 10
# ⌈(n - o) / (s - o)⌉: the step is s - o
math.ceil((n - o) / (s - o))  # → 3
```

**In code:** `fixed_size_chunks` and `structure_aware_chunks` cut the two
ways, and `parent_child_search` matches a sentence and returns its section.

Good chunking comes down to four rules: split on structure, keep headings
with their content, add modest overlap, and attach metadata for filtering.

## 7. Asymmetric retrieval and the forgotten prefix

A badly placed cut loses an answer without raising an error. So can an
embedding model that is called the wrong way. Retrieval is **asymmetric**:
questions are short and phrased as asks, while passages are long
statements. Some embedding models (the E5 family, for example) are trained
to expect a `"query: "` prefix on questions and a `"passage: "` prefix on
documents: a stamp that tells the model which side a text is on. Forget a
prefix and every vector still looks normal (right length, plausible scores)
but sits in a slightly wrong part of the space.

A simulated model shows the size of the effect. With both prefixes, 11 of 12
everyday questions find their answer in the top 3. Index the documents
*without* "passage: " and the same questions fall to 9 of 12, and MRR drops
from 0.875 to about 0.57. No error is raised anywhere.

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
recall on a set of questions with known answers.

**In code:** `PrefixedEmbedder` simulates the model: it partly rotates the
vector of any text that lacks a prefix.

Read the embedding model's documentation for required prefixes or
instructions, use the *same* model and settings at indexing and query time,
and keep a small set of questions with known answers, so that a drop in
recall shows up.

## In 20 seconds
- **BM25** scores keyword matches: rare words weigh more (IDF), repeated
  mentions saturate (k₁), long documents are discounted (b). It finds exact
  IDs and misses synonyms.
- **Dense retrieval** (bi-encoder) finds meaning and paraphrases but blurs
  exact tokens. Documents are embedded once, ahead of time.
- **Hybrid search** fuses both rankings with **RRF**, Σ 1/(60 + rank), using
  ranks, not scores, and usually beats either search alone.
- **Retrieve, then rerank:** a bi-encoder or hybrid search shortlists 50 to
  100 cheaply; a cross-encoder reads each (question, candidate) pair together
  and reorders the shortlist.
- **ColBERT** keeps a vector per word and scores with MaxSim: close to
  cross-encoder precision, still precomputable, much more storage.
- **Chunk on structure**, keep headings with their content and attach
  metadata. **Measure recall@k** on questions with known answers: a forgotten
  prefix raises no error, and only that measurement catches it.

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


# Questions about the handbook, each with the one sentence that answers it.
# Chosen so the interactive widget shows both failures: a boundary that cuts
# the answer sentence in two, and a first-stage ranking the reranker corrects.
CHUNKING_QUESTIONS: list[tuple[str, str]] = [
    ("What is the stipend amount in dollars?", "The stipend is 50 dollars per month."),
    ("Do I need the VPN on shared Wi-Fi?", "Use the VPN on any network you do not control, including home Wi-Fi shared with others."),
    ("What time are core hours?", "Core hours are 10:00 to 15:00 in your local time zone."),
    ("Can I get a second monitor?", "A second monitor can be requested for design and engineering roles."),
]


def chunk_engine(chunks: Sequence[str]) -> SearchEngine:
    """A search engine over one document's chunks, deep enough that every chunk gets a rank.

    Each chunk becomes a titleless `Doc` with id "chunk-<position>", so the
    whole retrieve-then-rerank pipeline runs on chunks exactly as it runs on
    articles.
    """
    docs = [Doc(f"chunk-{i}", "", c, "", "") for i, c in enumerate(chunks)]
    return SearchEngine(docs, depth=len(docs))


def viz_data() -> dict:
    """The numbers the site's chunking, retrieval and reranking widget shows.

    The widget cuts the chunks itself (its chunker mirrors
    `fixed_size_chunks`), but it can't embed text, so for every chunk size,
    overlap and question the slider can reach, this precomputes the hybrid
    ranking with its fused scores and the cross-encoder's score for every
    chunk. The widget reranks any top k from those.
    """
    sizes, overlaps = [20, 30, 40, 60], [0, 5, 10]
    judge = CrossEncoder()
    runs = {}
    for size in sizes:
        for overlap in overlaps:
            engine = chunk_engine(fixed_size_chunks(HANDBOOK, size, overlap))
            position = {doc_id: i for i, doc_id in enumerate(engine.ids)}
            runs[f"{size}/{overlap}"] = [
                {
                    # Exactly what engine.hybrid fuses, kept with its scores so the widget can show them.
                    "retrieval": [[position[d], round(s, 5)] for d, s in reciprocal_rank_fusion(
                        [engine.bm25(q, engine.depth), engine.dense(q, engine.depth)])],
                    "rerank": [round(judge.score(q, d), 4) for d in engine.docs],
                }
                for q, _ in CHUNKING_QUESTIONS
            ]
    return {
        "rag-pipeline": {
            "text": HANDBOOK,
            "sizes": sizes,
            "overlaps": overlaps,
            "max_k": 8,
            "questions": [{"question": q, "answer": a} for q, a in CHUNKING_QUESTIONS],
            "runs": runs,
        }
    }


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
    shown = f <= 6  # the raw count keeps climbing; draw it up to the chart's top, not through the title
    a1.plot(f[shown], f[shown], "--", color=MUTED, label="raw count")
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
    a2.set_title("Length: long documents are discounted")
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
    # Each label sits on its own line; an opaque box above the line keeps the words whole.
    on_line = dict(ha="center", fontsize=8, zorder=3, bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.text(heading_at, 1.02, "heading", color=HOT, **on_line)
    ax.text(amount_at, 1.02, "50 dollars", color=GOOD, **on_line)
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
