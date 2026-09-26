r"""
# Pretraining at scale: from a web crawl to a base model on thousands of GPUs

Run: `python -m primer.ml.pretraining`

New to the notation? `primer.notation` explains every symbol used here
(Σ, log, subscripts, ‖x‖ and so on) from zero. This lesson deepens section 1
of `primer.ml.training_stages`, which shows *what* pretraining optimizes: the
average next-token loss. Here we open the two boxes that make it hard in
practice: **the data** (where trillions of tokens come from and how they are
cleaned) and **the machine** (how one training run is spread across
thousands of GPUs without running out of memory, precision or patience).

## The everyday picture

Imagine writing an encyclopedia by reading everything ever printed. Two
problems appear at once.

First, most of what is printed is junk: flyers, receipts, the same cookie
notice on a million websites, spam. You need a sorting line that throws out
the junk and the photocopies before anyone reads a page.

Second, no single reader can do the reading. You hire a thousand readers,
and now you have a management problem: how do they split the work, share
what they learned, and keep going when one of them gets sick?

Pretraining is exactly those two problems. The first is data curation. The
second is distributed training.

```mermaid
flowchart LR
  W[Web crawl<br/>billions of pages] --> C[Curation<br/>language, quality,<br/>deduplication]
  C --> M[Mixture<br/>weights per source]
  M --> T[Tokenize<br/>trillions of tokens]
  T --> P[Parallel training<br/>data, tensor, pipeline]
  P --> MP[Mixed precision<br/>bf16 math, fp32 master]
  MP --> S[Stability<br/>clip, watch spikes,<br/>checkpoint]
  S --> B[Base model]
```

**Reading it:** the left half of the chain (crawl, curation, mixture,
tokenize) decides *what* the model learns; most of a model's knowledge and
many of its quirks are settled here, before any GPU is switched on. The right
half (parallel training, mixed precision, stability) decides whether the
learning can happen at all: a 7-billion-parameter model does not fit on one
GPU, and a months-long run on thousands of GPUs will see hardware fail many
times. Every box gets its own section below, in this order.

## 1. Where the data comes from, and how it is cleaned

**Everyday picture.** Think of a recycling plant. Trucks tip mixed rubbish
onto a conveyor belt. Magnets pull out the steel, blowers lift out the paper,
people pick out what the machines miss, and only a small fraction reaches the
bale at the end. Pretraining data goes down a belt like this, and just as in
a recycling plant, most of what goes in never comes out.

The raw material is usually **Common Crawl**, a nonprofit's public archive of
the web: regular snapshots, each of billions of pages. A page arrives as
HTML full of menus, adverts and scripts, so the first machine on the belt is
**text extraction**, which keeps the main body text. After that come the
filters this section builds: language identification, quality filters,
deduplication and mixing. FineWeb, an open dataset built this way from 96
Common Crawl snapshots, ends with about 15 trillion tokens.

```mermaid
flowchart LR
  H[HTML page] --> X[Extract<br/>main text]
  X --> L{Language ID<br/>is it English?}
  L -- no --> D1[drop, or route to<br/>that language]
  L -- yes --> Q{Quality rules<br/>and classifier}
  Q -- fail --> D2[drop]
  Q -- pass --> E{Exact duplicate?<br/>hash of the text}
  E -- yes --> D3[drop]
  E -- no --> N{Near duplicate?<br/>MinHash + LSH}
  N -- yes --> D4[drop]
  N -- no --> K[Keep:<br/>goes to the mixture]
```

**Reading it:** each diamond is a filter and each "drop" box is a way a page
leaves the belt. The order matters for cost: language ID and quality rules
look at one page at a time, so they are cheap and run first. Deduplication
compares pages *with each other*, which is the expensive part, so it runs
last, on what survived. The rest of this section builds every diamond in turn.

### 1a. Language identification

**Everyday picture.** Overhear two words of a phone call, "le" and "et", and
you already guess French. Every language has a handful of little words that
turn up in almost every sentence.

**Tiny worked example.** "the cat is in the garden and it was happy" has 10
words; 7 of them are on the English list of little words (the, is, in, the,
and, it, was) and none are on the French, German or Spanish lists, so the
guess is English with a share of 0.7. "SKU-4431 X99 blk/wht 12pk" matches no
list at all, so it is marked unknown and dropped.

Production pipelines use a trained classifier over character sequences
(fastText's language ID model covers 176 languages) and keep a page only if
the classifier is confident. The idea is the same: count the evidence for
each language and pick the strongest.

**In code:** `detect_language` counts each language's stop words (`STOP_WORDS`) and returns the language with the largest share, or "unknown" below 10%.

### 1b. Heuristic quality filters

**Everyday picture.** A librarian sorting donations does not read every
book. A book with no pages, a pamphlet that is all hashtags, a sheet of
keywords: each is rejected at a glance by a simple rule.

**Tiny worked example.** The Gopher paper (Rae et al., 2021) published a
set of such rules. Here is what they say about a navigation bar, "Home |
About us | Contact | Privacy policy | Log in":

| Rule | Keep only if | The navigation bar | Verdict |
|---|---|---|---|
| length | 50 to 100,000 words | 12 (each "\|" counts as a word) | fail |
| mean word length | 3 to 10 characters | 3.3 | pass |
| symbols (#, ...) per word | at most 0.1 | 0 | pass |
| words containing a letter | at least 80% | 8 of 12 = 67% | fail |
| common English words | at least 2 of: the, be, to, of, and, that, have, with | none | fail |

The last rule is the clever one. Real prose cannot avoid words like "the"
and "of"; keyword-stuffed pages and lists of product names avoid them
completely. Other rules in the set catch pages that are mostly bullet points
or mostly lines ending in "...".

**In code:** `quality_failures` applies each rule and returns the names of the ones a document breaks; an empty list means keep.

### 1c. Quality classifiers: scoring pages against a reference

Rules catch obvious junk. To prefer *good* text among the survivors,
pipelines train a classifier: examples of the text you want (encyclopedia
articles, books, pages that trusted sites link to) against random crawl, and
then keep pages that score like the reference. GPT-3 filtered Common Crawl
with a simple linear classifier of this kind; FineWeb-Edu asked a large
model to rate pages for educational value and trained a small classifier on
those ratings. We build the simplest version, **naive Bayes**: every word
casts a vote, and the votes add up.

**Tiny worked example.** Our reference examples contain the word "river" 3
times and the spam examples 0 times; "click" appears 0 times in the
reference and 4 times in spam. So "river" votes for good and "click" votes
for bad. A whole page's score is the sum of its words' votes.

$$
\text{score}(\text{doc}) = \sum_{w \in \text{doc}} \log \frac{P(w \mid \text{good})}{P(w \mid \text{bad})},
\qquad
P(w \mid c) = \frac{\text{count}_c(w) + 1}{N_c + V}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $w$ | one word of the document | "river" |
| $\sum_{w \in \text{doc}}$ | add up the term for every word in the document | |
| $c$ | a class: good (reference) or bad (spam) | |
| $\text{count}_c(w)$ | how often $w$ appears in class $c$'s examples | 3 in good, 0 in bad |
| $N_c$ | total words in class $c$'s examples | 77 good, 45 bad |
| $V$ | number of distinct words across both classes (the vocabulary) | 80 |
| $+1$ | add-one smoothing: pretend every word was seen once more, so an unseen word never gives probability 0 (and log 0 = −∞) | |
| $P(w \mid c)$ | "probability of $w$ given $c$": how likely a word drawn from class $c$ is $w$ | $P(\text{river} \mid \text{good}) = 4/157$ |
| $\log$ | natural logarithm: turns a ratio above 1 into a positive vote and below 1 into a negative vote | $\log 3.185 = 1.158$ |

**In words:** "for each word, ask how much more likely it is in good text
than in spam, take the log of that ratio as the word's vote, and add up the
votes."

**With the numbers:** P(river | good) = (3 + 1) / (77 + 80) = 4/157 and
P(river | bad) = (0 + 1) / (45 + 80) = 1/125. Their ratio is 3.185 and its
log is **+1.158**. For "click": (0 + 1)/157 over (4 + 1)/125 is 0.159, a vote
of **−1.837**. The sentence "the delta is formed when the river deposits
sand over many years" scores +8.52; "click here buy now best price free free
free" scores −16.49.

**In Python:**

```python
import math
N_good, N_bad, V = 77, 45, 80
def vote(count_good, count_bad):
    # log P(w | good) / P(w | bad), with add-one smoothing
    p_good = (count_good + 1) / (N_good + V)
    p_bad = (count_bad + 1) / (N_bad + V)
    return math.log(p_good / p_bad)
# "river": 3 times in good, 0 in bad
round(vote(3, 0), 3)  # → 1.158
# "click": 0 times in good, 4 in bad
round(vote(0, 4), 3)  # → -1.837
```

Why "naive"? It treats every word as independent evidence, which is false
(words come in phrases) but works well enough to sort billions of pages
cheaply. The practical danger is that a classifier keeps whatever resembles
its reference set, including its blind spots: pick only encyclopedia text
as "good" and you quietly filter out dialects, forums and whole topics.

**In code:** `QualityClassifier.trained_on_examples` counts words in `GOOD_EXAMPLES` and `BAD_EXAMPLES`; `QualityClassifier.word_vote` is the formula for one word and `QualityClassifier.score` adds the votes.

### 1d. Exact duplicates: one fingerprint per page

**Everyday picture.** A cloakroom attendant does not compare every coat
with every other coat. Each coat gets a numbered ticket, and two coats with
the same ticket are the same coat.

The web is full of copies: mirrored sites, syndicated news, the same terms
of service on a million shops. For exact copies the trick is a **hash**: a
function that turns any text into a short fingerprint, always the same for
the same text and almost never the same for different texts. Lowercase the
text and squash its spaces first, so that trivially reformatted copies get
the same fingerprint, then keep the first page with each fingerprint. One
pass, one lookup per page, no comparisons.

**In code:** `normalize_for_exact` lowercases and collapses whitespace, and `exact_dedup` keeps the first document with each SHA-1 fingerprint.

### 1e. Near duplicates: shingles and Jaccard similarity

A scraper site that copies an article and adds "Read more on our site"
defeats the exact hash: one changed character gives a completely different
fingerprint. We need a measure of *how much* two pages overlap.

**Everyday picture.** Cut each page into overlapping strips of a few words,
like roof shingles, and put each page's strips in a bag. Two pages are near
duplicates when their bags hold mostly the same strips.

**Tiny worked example.** With 2-word shingles:

| Sentence | Its shingles |
|---|---|
| "the cat sat on the mat" | the cat, cat sat, sat on, on the, the mat |
| "the cat sat on a mat" | the cat, cat sat, sat on, on a, a mat |

Three shingles are shared (the cat, cat sat, sat on) and seven are distinct
across both, so the overlap is 3/7 = 0.43.

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $A$, $B$ | the two sets of shingles | 5 shingles each |
| $A \cap B$ | the **intersection**: shingles in both sets | {the cat, cat sat, sat on} |
| $A \cup B$ | the **union**: shingles in either set, each counted once | 7 shingles |
| $\lvert \cdot \rvert$ | the number of items in a set | $\lvert A \cap B \rvert = 3$ |
| $J(A, B)$ | the **Jaccard similarity**, from 0 (nothing shared) to 1 (identical) | 0.43 |

**In words:** "the number of shingles the two pages share, divided by the
number of different shingles they have between them."

**With the numbers:** J = 3 / 7 = **0.43**. Real pipelines use 5-word
shingles on whole pages; the scraper's copy of our 68-word river paragraph
(one phrase changed, one sentence added) shares 78% of its shingles with
the original.

**In Python:**

```python
def shingles(text, k=2):
    ws = text.split()
    return {" ".join(ws[i:i + k]) for i in range(len(ws) - k + 1)}
A = shingles("the cat sat on the mat")
B = shingles("the cat sat on a mat")
# |A ∩ B| and |A ∪ B|
len(A & B), len(A | B)  # → (3, 7)
# J(A, B)
round(len(A & B) / len(A | B), 2)  # → 0.43
```

**In code:** `shingles` cuts a text into k-word windows (5 by default) and `jaccard` divides the intersection by the union.

### 1f. MinHash: estimating Jaccard from a few numbers

Jaccard needs both full shingle sets side by side. With billions of pages,
comparing every pair is impossible (a billion pages make about 5 × 10¹⁷
pairs). **MinHash** compresses each page into a short list of numbers, its
**signature**, such that comparing two signatures estimates their Jaccard.

**Everyday picture.** Shuffle a deck containing every shingle in the world
and deal from the top. Stop at the first card that belongs to page A, and
separately at the first card that belongs to page B. Those two "first cards"
are the same card exactly when the first card from A's-or-B's pile happens
to be one they share. The more they share, the likelier that is.

**Tiny worked example.** A = {a, b, c} and B = {b, c, d}, so J = 2/4 = 0.5.
Shuffle the four letters in all 24 possible orders. In 12 of them the first
letter from A ∪ B is b or c (a shared letter), and then A's first and B's
first are the same letter. 12 / 24 = 0.5, exactly J.

$$
P\big[\min h(A) = \min h(B)\big] = J(A, B),
\qquad
\hat{J} = \frac{1}{k} \sum_{i=1}^{k} \mathbf{1}\big[\min h_i(A) = \min h_i(B)\big]
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $h$ | a random hash function: gives every shingle a random-looking number, which acts as a random shuffle of all shingles | an order of a, b, c, d |
| $\min h(A)$ | the smallest number $h$ gives any shingle of $A$: A's "first card" | |
| $P[\ldots]$ | the probability that the statement in brackets is true | 12/24 |
| $k$ | how many independent hash functions (the signature length) | 128 |
| $h_i$ | the $i$-th hash function | |
| $\mathbf{1}[\ldots]$ | the **indicator**: 1 if the statement is true, 0 if not | |
| $\hat{J}$ | the estimate of $J$ (the hat means "estimated") | |

**In words:** "under a random shuffle, two sets have the same first element
with probability equal to their Jaccard similarity; so repeat with k
shuffles and report the fraction of times the first elements agreed."

**With the numbers:** all 24 orders of {a, b, c, d}: 12 agree, P = 0.5 = J.
With k = 128 hashes, each pair of pages is compared with 128 numbers
instead of hundreds of shingles, and the estimate's typical error is about
√(J(1 − J)/k) = √(0.25/128) ≈ 0.044.

**In Python:**

```python
import itertools, math
A, B = {"a", "b", "c"}, {"b", "c", "d"}
orders = list(itertools.permutations("abcd"))
def first(order, s):
    return next(x for x in order if x in s)
agree = sum(first(o, A) == first(o, B) for o in orders)
agree, len(orders)  # → (12, 24)
# P[min h(A) = min h(B)] equals J(A, B)
agree / len(orders), len(A & B) / len(A | B)  # → (0.5, 0.5)
# typical error of the estimate with k = 128 hashes
round(math.sqrt(0.5 * 0.5 / 128), 3)  # → 0.044
```

![Three pairs of sets with true Jaccard 0.2, 0.5 and 0.8: with a handful of hashes the estimates jump around; by 256 hashes each sits close to its dashed true value](figures/primer.ml.pretraining.minhash_convergence.svg)

**Reading it:** the horizontal axis is the number of hash functions k (log
scale) and the vertical axis the MinHash estimate. Each colour is one pair
of sets and its dashed line is that pair's true Jaccard. On the left, with
only a few hashes, the estimate can only be a coarse fraction and lurches
around. Moving right, each line settles onto its dashed line. That is the
whole promise of MinHash: a fixed, small signature per page, with error you
choose by choosing k.

**In code:** `MinHasher` draws k hash functions of the form (a·x + b) mod p and `MinHasher.signature` keeps the minimum under each; `estimate_jaccard` counts agreeing slots.

### 1g. Locality-sensitive hashing: finding candidate pairs without comparing all of them

Signatures make each comparison cheap, but a billion pages still make too
many pairs. **Locality-sensitive hashing (LSH)** avoids most comparisons:
cut each signature into $b$ **bands** of $r$ numbers, and file every page
into one bucket per band, keyed by that band's numbers. Only pages that
share a bucket in at least one band are ever compared.

```mermaid
flowchart LR
  D[Page] --> S[5-word shingles]
  S --> H[k hash functions<br/>keep each minimum]
  H --> G[Signature<br/>k numbers]
  G --> B1[band 1: r numbers] --> K1[bucket]
  G --> B2[band 2] --> K2[bucket]
  G --> BB[band b] --> KB[bucket]
  K1 & K2 & KB --> C[Candidate pairs:<br/>pages sharing any bucket]
  C --> V[Check estimated Jaccard<br/>against the threshold]
```

**Reading it:** a page flows left to right. Its shingles become a signature
of k = b × r numbers, and the signature is sliced into b bands. Each band is
a key into its own table of buckets. Two pages meet in the "candidate pairs"
box only if all r numbers of some band match exactly; for everything else
no comparison ever happens. The final box confirms each candidate with the
full signature.

$$
P(\text{candidate}) = 1 - \left(1 - s^{r}\right)^{b}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $s$ | the pair's Jaccard similarity | 0.8 or 0.3 |
| $r$ | rows: numbers per band | 5 |
| $b$ | number of bands | 20 |
| $s^r$ | chance that all $r$ numbers of one band agree (each agrees with chance $s$) | $0.8^5 = 0.328$ |
| $1 - s^r$ | chance that one band does *not* fully agree | 0.672 |
| $(1 - s^r)^b$ | chance that *no* band agrees | $0.672^{20} = 0.00036$ |

**In words:** "a pair becomes a candidate unless every one of its b bands
has at least one disagreeing number."

**With the numbers:** at s = 0.8: 1 − (1 − 0.328)²⁰ = **0.9996**, so near
duplicates are almost never missed. At s = 0.3: 0.3⁵ = 0.00243, and
1 − 0.99757²⁰ = **0.047**, so dissimilar pages are almost never compared.

**In Python:**

```python
def p_candidate(s, b=20, r=5):
    # 1 − (1 − s^r)^b
    return 1 - (1 - s ** r) ** b
round(p_candidate(0.8), 4)  # → 0.9996
round(p_candidate(0.3), 4)  # → 0.0475
# where the curve is steepest: about (1/b)^(1/r)
round((1 / 20) ** (1 / 5), 2)  # → 0.55
```

![Probability of becoming a candidate against true similarity, for three band layouts of about 100 hashes: each is an S-curve, steeper and further right as rows per band grow](figures/primer.ml.pretraining.lsh_s_curve.svg)

**Reading it:** the horizontal axis is the true Jaccard of a pair and the
vertical axis its chance of being compared. Every layout draws an S: pairs
on the left are almost never compared (that is the saving) and pairs on the
right almost always are (that is the recall). The dashed verticals mark
(1/b)^(1/r), where each curve is steepest. Choosing b and r is choosing where
the cliff sits: more rows per band push it right and make it sharper.

**In code:** `lsh_candidate_probability` is the formula, and `near_duplicate_pairs` runs the whole mechanism: signatures, band buckets, candidate pairs, and a final check against the threshold.

### 1h. Why duplicates hurt

**Everyday picture.** A student who reads the same paragraph a hundred
times can recite it, but has not learned a hundred paragraphs' worth. A
model that sees a page thousands of times does the same: it spends capacity
memorizing that page, and learns to recite it when prompted.

**Tiny worked example.** A **bigram model** predicts each word from the
word before it, by counting pairs. Train one on five short lines plus the
boilerplate "click here to subscribe for free updates", and ask for the
probability that it continues "click" into the whole boilerplate line:

| Next-word step | Kept once | Kept 100 times |
|---|---|---|
| click → here | 1/2 | 100/101 |
| here → to | 1/2 | 100/101 |
| to → subscribe | 1/3 | 100/102 |
| subscribe → for | 1 | 1 |
| for → free | 1/3 | 100/102 |
| free → updates | 1/2 | 100/101 |
| **whole line** | **1/72 = 0.014** | **0.93** |

Kept once, the model has many ways to continue "click". Repeated 100 times,
the line becomes the only road, and the model regurgitates it 93% of the
time. Lee et al. (2021) found a single 61-word sentence repeated more than
60,000 times in the C4 dataset, and that models trained on deduplicated
data emit memorized training text about ten times less often. Duplicates
also waste compute and leak into test sets: a benchmark question copied
across the web ends up in the training data, and the model's score on it
measures recall, not skill.

**In code:** `verbatim_probability` trains the bigram counts on `OTHER_LINES` plus the given number of copies of `BOILERPLATE` and multiplies the next-word probabilities along the line.

### 1i. The whole belt on a small crawl

`CRAWL_SAMPLE` holds seven pages. Running `curate` over it:

| Page | What it is | Verdict |
|---|---|---|
| 0 | a clean English paragraph about river deltas | kept |
| 1 | the same idea in French | language: fr |
| 2 | a navigation bar | quality: too_short, too_few_alphabetic_words, missing_stop_words |
| 3 | page 0 re-crawled with different capitals and spacing | exact duplicate of 0 |
| 4 | a hashtag spam page | quality: too_many_symbols, missing_stop_words |
| 5 | page 0 with a phrase changed and a link added | near duplicate of 0 |
| 6 | a clean English paragraph about stars | kept |

Two of seven survive. That is not unusual: large public pipelines keep only
a small fraction of the raw crawl they start from.

**In code:** `curate` runs language ID, the quality rules, the exact hash and a MinHash comparison in that order, and returns one verdict per page.

### 1j. Data mixtures: how much of each source

**Everyday picture.** A diet is not "all the food in the shop". You choose
proportions: mostly staples, some vegetables, a little of the rich stuff.
Pretraining data is mixed the same way from sources that differ in size and
value: web text, code, books, encyclopedias, scientific papers, maths.

Each source gets a **mixture weight**: its share of the tokens the model
will see. Weights do not follow size. A small, valuable source (an
encyclopedia) is often up-weighted, which means the model sees it more than
once; a huge, noisy one (the web) is sampled less than one full pass.

$$
\text{epochs}_i = \frac{w_i \, D}{N_i}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $i$ | one data source | the encyclopedia |
| $w_i$ | its mixture weight: share of all training tokens drawn from it | 0.05 |
| $D$ | the total training budget, in tokens | 1,000 billion |
| $N_i$ | the tokens that source actually has | 20 billion |
| $\text{epochs}_i$ | how many full passes over the source the mixture implies | 2.5 |

**In words:** "the tokens you plan to draw from a source, divided by the
tokens it has, is how many times the model reads it."

**With the numbers:** web 0.80 × 1,000B / 900B = **0.89** passes; wiki
0.05 × 1,000B / 20B = **2.5** passes; code 0.15 × 1,000B / 150B = **1.0**.
The LLaMA paper's mixture has the same shape: Common Crawl is two thirds of
the tokens at about 1.1 passes, while Wikipedia and books are 4.5% each but
are read more than twice (2.45 and 2.23 passes).

**In Python:**

```python
weights = {"web": 0.80, "wiki": 0.05, "code": 0.15}
available = {"web": 900e9, "wiki": 20e9, "code": 150e9}
D = 1000e9
# epochs_i = w_i D / N_i
{k: round(weights[k] * D / available[k], 2) for k in weights}  # → {'web': 0.89, 'wiki': 2.5, 'code': 1.0}
```

Repeating a small source a few times is fine; many more passes and the model
starts memorizing it (the duplicate problem again, on purpose). Mixture
weights are usually chosen by training small models on candidate mixtures
and comparing them, and many recipes change the mixture near the end of
training, up-weighting the cleanest sources.

**In code:** `epochs_per_source` applies the formula to every source.

### 1k. Token budgets: how much data for how big a model

**Everyday picture.** A bigger brain can learn more, but only if you give
it more to read. Given a fixed amount of study time (compute), there is a
best split between "bigger brain" and "more reading".

All budgets are counted in **tokens**, the pieces a tokenizer cuts text
into (see `primer.ml.tokenization`); in English a token is roughly three
quarters of a word. Two rules of thumb set the scale.

$$
D_{\text{opt}} \approx 20\,N,
\qquad
C \approx 6\,N\,D
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $N$ | the number of model parameters | 7 billion |
| $D$ | the number of training tokens | 140 billion |
| $D_{\text{opt}}$ | the compute-optimal token count for size $N$ (Hoffmann et al., 2022) | $20 \times 7 \times 10^9$ |
| $C$ | training compute in FLOPs (floating-point operations) | $5.88 \times 10^{21}$ |
| $6$ | 2 FLOPs per parameter per token in the forward pass (one multiply, one add), 4 in the backward pass | |
| $\approx$ | "approximately equal": a rule of thumb, not an exact law | |

**In words:** "for the best model per unit of compute, train on about 20
tokens per parameter; and training costs about six operations per
parameter per token."

**With the numbers:** a 7B model's compute-optimal budget is 20 × 7 × 10⁹ =
**140 billion tokens**, and training it costs 6 × 7 × 10⁹ × 1.4 × 10¹¹ =
**5.88 × 10²¹ FLOPs**. At a sustained 400 teraFLOP/s per GPU (about 40% of an
H100's bf16 peak) that is 5.88 × 10²¹ / 4 × 10¹⁴ ≈ 1.5 × 10⁷ GPU-seconds,
about 4,100 GPU-hours.

**In Python:**

```python
N = 7e9
# D_opt ≈ 20 N
D = 20 * N
D  # → 140000000000.0
# C ≈ 6 N D
C = 6 * N * D
C  # → 5.88e+21
# GPU-hours at 400 teraFLOP/s each
round(C / 400e12 / 3600)  # → 4083
```

In practice, models that will serve billions of requests are trained far
past this point, because a smaller model trained longer is cheaper to run
forever after. Llama 2 7B saw 2 trillion tokens (about 286 per parameter);
Llama 3 8B saw more than 15 trillion. That is why data, not compute, is now
often the binding limit, and why the next topic exists.

**In code:** `chinchilla_tokens` and `training_flops` are the two rules of thumb.

### 1l. Synthetic data, and model collapse

When good human text runs short, models generate more: rewritten web pages,
worked maths solutions checked by a program, question-and-answer pairs
drawn out of textbooks. Used carefully this works well: it can turn a
messy page into a clear one, and a checked answer is clean signal. Used
carelessly it has a known failure.

**Everyday picture.** Photocopy a photo, then photocopy the copy, and keep
going. Each copy is nearly right, but fine detail disappears first, and
after enough rounds you have a grey smudge. A model trained on its own
outputs loses the rare, surprising parts of the data (the tails) first.

**Tiny worked example.** Draw 20 numbers from a bell curve with spread 1.
Fit a bell curve to those 20 numbers, draw 20 new numbers from the fit, fit
again, and repeat. Each fit is a slightly narrower guess on average, and the
narrowing compounds.

```mermaid
flowchart LR
  R[Real data<br/>spread 1.0] --> F1[Fit model 1]
  F1 --> S1[Sample from model 1]
  S1 --> F2[Fit model 2]
  F2 --> S2[Sample from model 2]
  S2 --> FN[... model n]
  FN --> X[Spread shrinks:<br/>tails are forgotten first]
```

**Reading it:** follow the chain left to right. Only the first model ever
sees real data; every later model learns from the previous model's samples.
Nothing in the loop can put back a rare value once a sample happens to miss
it, so information only leaks out. The fix in practice is to keep real data
in every generation's mix, and to filter synthetic data with checks that do
not come from the same model.

$$
\mathbb{E}\big[\hat{\sigma}^2_{t+1}\big] = \frac{n - 1}{n}\,\hat{\sigma}^2_t
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $t$ | the generation number | 0, 1, 2, … |
| $\hat{\sigma}^2_t$ | the fitted **variance** (spread squared) of generation $t$'s model | 1.0 at the start |
| $n$ | samples each generation is fitted on | 20 |
| $\mathbb{E}[\ldots]$ | the **expected value**: the average over many repeats of the experiment | |
| $\frac{n-1}{n}$ | the shrink factor per generation of a maximum-likelihood fit | 0.95 |

**In words:** "on average, each refit on n samples keeps only (n − 1)/n of
the previous generation's variance."

**With the numbers:** 0.95 per generation; after 200 generations the
expected variance is 0.95²⁰⁰ ≈ 0.000035 of the original, a spread of about
0.006. The single run below does not follow the average exactly, but it
collapses all the same.

**In Python:**

```python
import math
n, generations = 20, 200
# (n − 1)/n per generation, compounded
shrink = ((n - 1) / n) ** generations
f"{shrink:.1e}"  # → '3.5e-05'
# the spread is the square root of the variance
round(math.sqrt(shrink), 4)  # → 0.0059
```

![Fitted spread over 200 generations of refitting on 20 samples, for three random seeds on a log scale: every run falls by four to five orders of magnitude, even faster than the dashed line from the formula](figures/primer.ml.pretraining.model_collapse.svg)

**Reading it:** the horizontal axis is the generation and the vertical axis
the fitted spread, on a log scale so that halving always looks the same
size. Each coloured line is one run with a different seed and the dashed
line is the formula's average. The runs wander, up some generations and
down others, but the trend is relentlessly downward, and a typical run
falls even faster than the dashed line: the average is held up by rare runs
that happen to stay wide. After 200 generations the "model" can only
produce values in a sliver of the original range.
Shumailov et al. (2023) showed the same effect in language models trained
recursively on their own text.

**In code:** `recursive_gaussian_fit` runs the fit-sample-refit loop and returns every generation's fitted spread.

## 2. Why one GPU is not enough: the memory bill

**Everyday picture.** To bake one cake you need the recipe, but to *learn*
to bake you also need your notes on every attempt: what went wrong, how
much to change, how confident you are in each change. Training a model is
the same. The weights are the recipe; training also keeps a gradient for
every weight (what to change) and the optimizer's running notes (how it has
been changing), and those notes are bigger than the recipe.

**Tiny worked example.** Take one parameter trained with Adam (see
`primer.ml.optimizers`) in mixed precision, the standard recipe (section 4
explains the number formats):

| What is stored per parameter | Format | Bytes |
|---|---|---|
| the weight used in the forward and backward pass | bf16 | 2 |
| its gradient | bf16 | 2 |
| a full-precision **master copy** of the weight | fp32 | 4 |
| Adam's momentum (running average of gradients) | fp32 | 4 |
| Adam's variance (running average of squared gradients) | fp32 | 4 |
| **total** | | **16** |

$$
M_{\text{state}} = (2 + 2 + 4 + 4 + 4)\,\Psi = 16\,\Psi \ \text{bytes}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $\Psi$ | the number of parameters (Psi, the letter the ZeRO paper uses) | $7 \times 10^9$ |
| $2 + 2$ | bytes for the bf16 weight and its bf16 gradient | |
| $4 + 4 + 4$ | bytes for the fp32 master weight and Adam's two fp32 averages | |
| $M_{\text{state}}$ | memory for the training state, before any activations | 112 GB |

**In words:** "training with Adam in mixed precision costs sixteen bytes
for every parameter, before you store a single activation."

**With the numbers:** 16 × 7 × 10⁹ = **112 GB** for a 7B model. A widely used
data-centre GPU, the H100, holds 80 GB (see `primer.ml.hardware`), so the
state alone does not fit. Serving the same model needs only the 2-byte weights,
14 GB: training costs eight times the memory of inference.

**In Python:**

```python
params = 7e9
bytes_per_param = 2 + 2 + 4 + 4 + 4
bytes_per_param  # → 16
# M_state in GB
bytes_per_param * params / 1e9  # → 112.0
# inference needs only the bf16 weights
2 * params / 1e9  # → 14.0
```

### Activations: the memory that grows with the batch

The backward pass needs the intermediate results of the forward pass (the
**activations**) to compute gradients, so they are kept until it runs. Their
size grows with the number of tokens in flight, not with the parameter count.
Korthikanti et al. (2022) counted them for one transformer layer, with 16-bit
activations:

$$
A = L \cdot s\,b\,h \left(34 + \frac{5\,a\,s}{h}\right) \ \text{bytes}
$$

**Symbols**

| Symbol | Meaning here | For a 7B shape |
|---|---|---|
| $L$ | number of layers | 32 |
| $s$ | sequence length in tokens | 4,096 |
| $b$ | sequences per GPU in the batch | 1 |
| $h$ | hidden width | 4,096 |
| $a$ | attention heads | 32 |
| $34\,s\,b\,h$ | the layer's ordinary tensors (inputs to each matrix multiply, norms, dropout masks) | |
| $5\,a\,s^2 b$ | attention's $s \times s$ score and weight matrices, one per head (written as $s b h \cdot 5as/h$) | |

**In words:** "each layer keeps about 34 bytes per token per hidden unit,
plus attention's square score matrices, which grow with the square of the
sequence length."

**With the numbers:** per layer, 4096 × 4096 × (34 + 5 × 32 × 4096 / 4096)
= 16.8 million × 194 bytes = 3.26 GB; over 32 layers, **104 GB**. Kernels
like FlashAttention never store the score matrices (they recompute them in
the backward pass), which removes the 5as/h term: 34 × 16.8 million × 32 =
**18.3 GB**.

**In Python:**

```python
L, s, b, h, a = 32, 4096, 1, 4096, 32
# A = L · s b h (34 + 5 a s / h)
round(L * s * b * h * (34 + 5 * a * s / h) / 1e9, 1)  # → 104.2
# without stored attention scores
round(L * s * b * h * 34 / 1e9, 1)  # → 18.3
# activation checkpointing keeps only each layer's 2-byte input
round(L * 2 * s * b * h / 1e9, 2)  # → 1.07
```

The last line is **activation checkpointing** (also called gradient
checkpointing, Chen et al., 2016): keep only each layer's input, and during
the backward pass re-run that layer's forward pass to rebuild what it needs.
It trades about one extra forward pass (roughly a third more compute) for
activation memory that no longer grows with depth.

![Stacked bars for a 7B model at 4,096 tokens: the 112 GB of training state alone passes the 80 GB line, and naive activations add another 104 GB](figures/primer.ml.pretraining.memory_7b.svg)

**Reading it:** each bar is the memory one GPU would need to train the 7B
model alone, split by what it holds; the dashed line is an 80 GB GPU. The
three bars differ only in how activations are handled: stored naively,
without attention scores, or checkpointed. Even the leanest bar is above
the line, because the 112 GB of weights, gradients and optimizer state is
there in every bar. Shrinking activations is not enough: the state itself
has to be split across GPUs. That is the next section.

**In code:** `training_memory` itemizes the 16 bytes per parameter, and `activation_bytes` is the activation formula, with `store_scores=False` for FlashAttention-style kernels and `checkpointed=True` for activation checkpointing.

## 3. Parallelism: many GPUs, one model

**Everyday picture.** A restaurant can serve more diners in three ways.
Open identical kitchens that each cook whole meals (**data parallelism**).
Split one dish across cooks working side by side on the same step, one
chopping the left half of the onions and one the right (**tensor
parallelism**). Or build an assembly line, with each cook doing one stage
and passing the plate on (**pipeline parallelism**). Real training runs use
all three at once.

### 3a. Data parallelism: identical copies, averaged gradients

Every GPU holds a full copy of the model and processes a different slice of
the batch. After the backward pass, the GPUs average their gradients so all
copies take the same step and stay identical. This works because the
gradient of an average loss is the average of the gradients.

**Tiny worked example.** A one-weight model predicts y = w·x, with loss the
mean of (w·x − y)². Batch: x = 1, 2, 3, 4 with y = 2, 4, 6, 8, and w = 1. On
one GPU the gradient is (2/4)·Σ x(wx − y) = (2/4)·(−1 − 4 − 9 − 16) = **−15**.
Split across two GPUs: GPU 1 gets x = 1, 2 and computes −5; GPU 2 gets
x = 3, 4 and computes −25. Their average is (−5 − 25)/2 = **−15**, the same.

$$
\nabla \mathcal{L} = \frac{1}{G} \sum_{k=1}^{G} \nabla \mathcal{L}_k
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $\mathcal{L}$ | the loss averaged over the whole batch | |
| $\nabla$ | "the gradient of": the slope of the loss for every weight | |
| $G$ | number of GPUs, each with an equal share of the batch | 2 |
| $\mathcal{L}_k$ | the loss averaged over GPU $k$'s share | |
| $\sum_{k=1}^{G}$ | add up over every GPU | |

**In words:** "the gradient for the whole batch is the average of the
gradients each GPU computed on its own equal share."

**With the numbers:** (−5 + −25) / 2 = −15, matching the single-GPU −15.

**In Python:**

```python
def grad(xs, ys, w=1.0):
    # d/dw of mean((w x − y)²)
    return 2 / len(xs) * sum(x * (w * x - y) for x, y in zip(xs, ys))
grad([1, 2, 3, 4], [2, 4, 6, 8])  # → -15.0
g1, g2 = grad([1, 2], [2, 4]), grad([3, 4], [6, 8])
g1, g2  # → (-5.0, -25.0)
(g1 + g2) / 2  # → -15.0
```

The averaging step is an **all-reduce**: every GPU contributes a vector and
every GPU receives the sum. Sending every gradient to one GPU would jam its
network link, so the standard algorithm is the **ring all-reduce**.

```mermaid
flowchart LR
  G0[GPU 0<br/>chunks A B C D] -->|one chunk per step| G1[GPU 1<br/>chunks A B C D]
  G1 -->|one chunk per step| G2[GPU 2<br/>chunks A B C D]
  G2 -->|one chunk per step| G3[GPU 3<br/>chunks A B C D]
  G3 -->|one chunk per step| G0
```

**Reading it:** the four GPUs sit in a ring and each only ever talks to its
right-hand neighbour. Each cuts its gradient into four chunks. Phase one
(**reduce-scatter**): for three steps, each GPU passes one chunk to the right,
where it is added to the neighbour's copy; afterwards each GPU owns the
complete sum for one chunk. Phase two (**all-gather**): for three more steps
the finished chunks travel round the ring, so everyone ends with all four
sums. Every link is busy at every step, and no GPU is a bottleneck.

$$
\text{sent per GPU} = \frac{2\,(G - 1)}{G}\,S
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $G$ | GPUs in the ring | 4, or 1,000 |
| $S$ | size of the gradient being summed | 14 GB (7B in bf16) |
| $G - 1$ | steps in each of the two phases | 3 |
| $\frac{1}{G}$ | each step moves one chunk, $1/G$ of the gradient | |
| $2$ | two phases: reduce-scatter, then all-gather | |

**In words:** "each GPU sends a little under twice its gradient, however
many GPUs are in the ring."

**With the numbers:** with 4 GPUs, 2 × 3/4 = 1.5 gradients' worth; for a
14 GB gradient on 8 GPUs, 2 × 7/8 × 14 = **24.5 GB** per GPU per step; on
1,000 GPUs, **27.97 GB**. The traffic per GPU barely grows, which is why
data parallelism scales to thousands of GPUs.

**In Python:**

```python
def sent(G, S):
    # 2 (G − 1) / G · S
    return 2 * (G - 1) / G * S
sent(4, 1.0)  # → 1.5
sent(8, 14e9) / 1e9  # → 24.5
round(sent(1000, 14e9) / 1e9, 2)  # → 27.97
```

Data parallelism alone does not solve the memory problem: every GPU still
holds all 16 bytes per parameter.

**In code:** `linear_regression_gradient` computes a batch's gradient, so the averaging identity can be checked on shards; `ring_all_reduce` simulates both phases and counts what each GPU sends; `all_reduce_traffic` is the formula.

### 3b. Sharding the state: ZeRO and FSDP

**Everyday picture.** A reading group with one expensive textbook does not
buy a copy each. They tear it into chapters, each keeps one, and whoever
needs a chapter borrows it for the evening and hands it back.

In plain data parallelism, every GPU stores an identical copy of the
optimizer state, the gradients and the weights: pure waste. **ZeRO** (the
Zero Redundancy Optimizer) keeps data parallelism's split of the batch but
gives each GPU only a 1/G shard of that state, in three stages. PyTorch's
**FSDP** (fully sharded data parallel) implements the third.

```mermaid
flowchart TB
  subgraph S0["Stage 0: plain data parallel"]
    A0["every GPU: weights + gradients + optimizer state"]
  end
  subgraph S1["Stage 1: shard the optimizer state"]
    A1["every GPU: weights + gradients<br/>its 1/G of the optimizer state"]
  end
  subgraph S2["Stage 2: also shard the gradients"]
    A2["every GPU: weights<br/>its 1/G of gradients and optimizer state"]
  end
  subgraph S3["Stage 3 = FSDP: shard everything"]
    A3["every GPU: its 1/G of everything<br/>borrow each layer's weights just in time"]
  end
  S0 --> S1 --> S2 --> S3
```

**Reading it:** read top to bottom; each stage moves one more kind of state
from "every GPU keeps all of it" to "every GPU keeps a 1/G slice". Stage 1
works because each GPU only needs to update its own slice of the weights.
Stage 2 works because a reduce-scatter (the first half of the ring) can
deliver each GPU just the gradient slice it updates. Stage 3 goes furthest:
before computing a layer, the GPUs all-gather that layer's weights, use
them, and throw them away again.

$$
M_{\text{stage 0}} = 16\Psi,
\quad
M_{1} = 4\Psi + \frac{12\Psi}{G},
\quad
M_{2} = 2\Psi + \frac{14\Psi}{G},
\quad
M_{3} = \frac{16\Psi}{G}
$$

**Symbols**

| Symbol | Meaning here | In the ZeRO paper's example |
|---|---|---|
| $\Psi$ | parameters | $7.5 \times 10^9$ |
| $G$ | GPUs sharing the state | 64 |
| $12\Psi$ | the fp32 master weights and Adam's two averages | 90 GB |
| $2\Psi$, $4\Psi$ | the bf16 weights; weights plus gradients | 15 GB; 30 GB |
| $M_{\text{stage}}$ | training state per GPU at that stage | |

**In words:** "whatever is sharded is divided by the number of GPUs; what
is not sharded stays whole on every GPU."

**With the numbers:** 7.5B parameters on 64 GPUs (the example in Figure 1
of the ZeRO paper): stage 0, **120 GB**; stage 1, 30 + 1.4 = **31.4 GB**;
stage 2, 15 + 1.6 = **16.6 GB**; stage 3, **1.9 GB**. The 7B model that could
not fit on one GPU now needs under 2 GB of state per GPU.

**In Python:**

```python
psi, G = 7.5e9, 64
stages = [16 * psi, 4 * psi + 12 * psi / G, 2 * psi + 14 * psi / G, 16 * psi / G]
[round(m / 1e9, 1) for m in stages]  # → [120.0, 31.4, 16.6, 1.9]
```

The price is communication. Stages 1 and 2 cost the same traffic as a plain
all-reduce; stage 3 adds an all-gather of the weights in the forward pass
and again in the backward pass, about 1.5 times the traffic. Frameworks hide
it by fetching the next layer's weights while the current layer computes.

![Training state per GPU for a 7B model as the number of GPUs grows from 1 to 1,024: stage 0 stays at 112 GB, stages 1 and 2 flatten at their unsharded floors, and stage 3 keeps falling](figures/primer.ml.pretraining.zero_stages.svg)

**Reading it:** the horizontal axis is the number of GPUs (doubling at each
tick) and the vertical axis the state each GPU holds, both on log scales. The
dashed line is an 80 GB GPU. Stage 0 is flat: adding GPUs never helps. Stages
1 and 2 fall at first and then level off at the part they do not shard (28
GB and 14 GB of weights and gradients). Only stage 3 keeps falling in a
straight line, because nothing is left unsharded.

**In code:** `zero_memory_per_gpu` gives the state per GPU for any stage and GPU count.

### 3c. Tensor parallelism: splitting one matrix multiply

**Everyday picture.** Two people fill in one large multiplication table:
one does the left half of the columns, the other the right half. Neither
needs the other's work until the end, when the halves are placed side by
side.

The biggest operations in a transformer are matrix multiplies, X·W. They
can be split across GPUs in two ways, and both give *exactly* the unsplit
answer.

**Tiny worked example.** X = [1, 2] and

W = [[1, 2, 3, 4], [5, 6, 7, 8]], so X·W = [11, 14, 17, 20].

- **By columns:** GPU 1 holds W's first two columns and computes [11, 14];
  GPU 2 holds the last two and computes [17, 20]. Place side by side:
  [11, 14, 17, 20].
- **By rows:** GPU 1 holds W's first row and X's first entry: 1 × [1, 2, 3, 4]
  = [1, 2, 3, 4]. GPU 2 holds the second: 2 × [5, 6, 7, 8] = [10, 12, 14, 16].
  Add: [11, 14, 17, 20].

$$
X W = \big[\,X W_1 \;\; X W_2\,\big]
\qquad\text{and}\qquad
X W = X_1 W_1 + X_2 W_2
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $X$ | the input activations, one row per token | [1, 2] |
| $W$ | a weight matrix | 2 × 4 |
| $[\,A \;\; B\,]$ | place two matrices side by side (**concatenate** columns) | |
| left: $W_1, W_2$ | $W$'s columns, split into two blocks | 2 × 2 each |
| right: $W_1, W_2$ | $W$'s rows, split into two blocks | 1 × 4 each |
| right: $X_1, X_2$ | $X$'s matching columns | [1] and [2] |

**In words:** "split the weight by columns and each GPU produces some of
the output columns; split it by rows and each GPU produces a partial sum of
every output, and the partial sums add up to the answer."

**With the numbers:** [11, 14] next to [17, 20], or [1, 2, 3, 4] + [10, 12,
14, 16]: both give **[11, 14, 17, 20]**.

**In Python:**

```python
X = [1, 2]
W = [[1, 2, 3, 4], [5, 6, 7, 8]]
def matmul(x, w):
    return [sum(x[i] * w[i][j] for i in range(len(x))) for j in range(len(w[0]))]
matmul(X, W)  # → [11, 14, 17, 20]
# by columns: each GPU computes half the output columns
matmul(X, [row[:2] for row in W]) + matmul(X, [row[2:] for row in W])  # → [11, 14, 17, 20]
# by rows: each GPU computes a partial sum of every output
p1, p2 = matmul(X[:1], W[:1]), matmul(X[1:], W[1:])
[a + b for a, b in zip(p1, p2)]  # → [11, 14, 17, 20]
```

Megatron-LM combines the two for a transformer's feed-forward layer, which
is Y = activation(X·W₁)·W₂: split W₁ by columns and W₂ by rows.

```mermaid
flowchart LR
  X[X, full copy<br/>on both GPUs] --> A1["GPU 1: X · W1 left columns"]
  X --> A2["GPU 2: X · W1 right columns"]
  A1 --> R1[ReLU, locally] --> B1["· W2 top rows<br/>partial sum"]
  A2 --> R2[ReLU, locally] --> B2["· W2 bottom rows<br/>partial sum"]
  B1 & B2 --> AR[All-reduce:<br/>add the partial sums] --> Y[Y, full copy<br/>on both GPUs]
```

**Reading it:** the input is copied to both GPUs. The column split of W₁
gives each GPU whole hidden units, so the activation function (applied
number by number) runs locally with no communication. The row split of W₂
then consumes exactly those hidden units and produces a partial sum of the
output. One all-reduce at the end adds the two partial sums. Needing only
one all-reduce per block (the attention block is split the same way) is
what makes this practical, but it still happens inside every layer, so
tensor parallelism needs the fastest links available and usually stays
within one server of 8 GPUs.

**In code:** `column_parallel_matmul` and `row_parallel_matmul` are the two splits, and `tensor_parallel_mlp` is the Megatron-LM feed-forward layer with its single all-reduce.

### 3d. Pipeline parallelism, and the bubble

**Everyday picture.** A car assembly line with four stations. The first
car takes four steps to roll off the end, and while it travels, stations
further down stand idle. Only when many cars are on the line at once is
every station busy.

Pipeline parallelism gives each GPU a consecutive block of layers (a
**stage**). To keep the stages busy, the batch is cut into
**micro-batches** that follow each other down the line. The idle time at
the start and end is the **pipeline bubble**.

```mermaid
flowchart LR
  MB[Micro-batches<br/>1, 2, 3, ...] --> S1[GPU 1<br/>layers 1-8]
  S1 -->|activations| S2[GPU 2<br/>layers 9-16]
  S2 -->|activations| S3[GPU 3<br/>layers 17-24]
  S3 -->|activations| S4[GPU 4<br/>layers 25-32]
  S4 -.->|gradients flow back| S1
```

**Reading it:** each GPU owns a quarter of the layers. Micro-batches enter
on the left one after another, and each GPU passes its output activations
to the next. When the forward passes are done, gradients flow back along
the dashed arrow in the reverse order. The only traffic is activations at
stage boundaries, far less than tensor parallelism's per-layer exchange,
which is why pipeline stages can sit on different servers.

$$
\text{bubble} = \frac{p - 1}{m + p - 1}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $p$ | pipeline stages (GPUs in the line) | 4 |
| $m$ | micro-batches per batch | 1, 8 or 32 |
| $m + p - 1$ | time steps to push $m$ micro-batches through $p$ stages: $p$ to fill the line, then one per extra micro-batch | 11 when $m = 8$ |
| $p - 1$ | steps each stage spends idle while the line fills (and again while it drains) | 3 |

**In words:** "the fraction of time each GPU sits idle is the fill time
divided by the total time; more micro-batches spread the same fill time
over more work."

**With the numbers:** with 4 stages and 1 micro-batch, 3/4 = **75%** of the
time is bubble; with 8 micro-batches, 3/11 = **27%**; with 32, 3/35 = **8.6%**.

**In Python:**

```python
def bubble(p, m):
    # (p − 1) / (m + p − 1)
    return (p - 1) / (m + p - 1)
[round(bubble(4, m), 3) for m in (1, 8, 32)]  # → [0.75, 0.273, 0.086]
```

![The GPipe schedule for 4 stages and 8 micro-batches as a grid of GPUs by time step: forward passes form a staircase down, backward passes a staircase back up, and the empty triangles in the corners are the bubble](figures/primer.ml.pretraining.pipeline_schedule.svg)

**Reading it:** each row is a GPU (stage) and each column one time step.
Blue cells are forward passes and green cells backward passes, numbered by
micro-batch. The forward staircase runs down and to the right as each
micro-batch moves to the next stage; the backward staircase runs back up.
The white triangles in the corners are the bubble: stage 4 waits three steps
for the first micro-batch to arrive, and stage 1 waits three steps at the
end. Count them: 24 of the 88 cells are empty, 3/11 of the grid.

![Bubble fraction against number of micro-batches for 2, 4, 8 and 16 stages: every curve starts high and falls towards zero, and deeper pipelines need more micro-batches for the same efficiency](figures/primer.ml.pretraining.bubble_fraction.svg)

**Reading it:** the horizontal axis is the number of micro-batches (log
scale) and the vertical axis the idle fraction. Each line is a pipeline
depth. To keep the bubble under about 10%, you need roughly ten times as
many micro-batches as stages, which pushes up the batch size. Smarter
schedules exist for exactly this reason: "one forward, one backward" (1F1B)
starts backward passes early to free activation memory, and interleaved
stages (Narayanan et al., 2021) give each GPU several smaller stages to
shrink the bubble further.

**In code:** `pipeline_schedule` builds the grid (positive numbers forward, negative backward, 0 idle) and `bubble_fraction` is the formula.

### 3e. All three at once

```mermaid
flowchart TB
  subgraph DP["Data parallel: replicas see different data, all-reduce gradients"]
    subgraph R1["Replica 1"]
      direction LR
      P1["Pipeline stage 1<br/>one server: 8 GPUs, tensor parallel"] --> P2["Pipeline stage 2<br/>one server: 8 GPUs, tensor parallel"]
    end
    subgraph R2["Replica 2"]
      direction LR
      Q1["Pipeline stage 1<br/>8 GPUs, tensor parallel"] --> Q2["Pipeline stage 2<br/>8 GPUs, tensor parallel"]
    end
  end
```

**Reading it:** the three kinds nest, and each is placed where its traffic
fits. Tensor parallelism, which talks inside every layer, stays inside one
server on its fastest links. Pipeline parallelism, which only passes
activations between stages, spans servers. Data parallelism (often sharded
with ZeRO or FSDP), which talks once per step, wraps the whole thing and
multiplies it across the cluster. Llama 3 405B was trained this way on up
to 16,384 GPUs, adding a fourth kind (context parallelism) that splits very
long sequences.

## 4. Mixed precision: doing the maths in fewer bits

**Everyday picture.** A carpenter measures a room with a tape measure and a
table leg with calipers. Using calipers for everything would be slow; using
the tape measure for everything would give wobbly tables. Mixed precision
does each job with the coarsest number format that is good enough: the
heavy matrix multiplies in 16 (or 8) bits, and the few places where tiny
differences matter in 32.

The payoff is large. Halving the bits halves the memory and the data moved,
and GPU tensor cores run 16-bit maths many times faster than 32-bit (see
`primer.ml.hardware`).

### 4a. What a floating-point number is

**Everyday picture.** Scientific notation, in binary. "6.02 × 10²³" has a
sign, a few significant digits and an exponent. A float is the same three
parts in bits: the exponent sets the **range** (how large or small a number
can be), and the fraction bits set the **precision** (how many significant
digits it keeps).

**Tiny worked example.** In bf16, 1/3 is stored as sign 0, exponent field
125 and fraction field 0101011 in binary (43). The value is
2^(125 − 127) × (1 + 43/128) = 0.25 × 1.3359375 = **0.333984375**. Seven
fraction bits keep only about three significant decimal digits, so bf16
cannot tell 1/3 from 0.33398.

$$
x = (-1)^{\text{sign}} \times 2^{\,E - \text{bias}} \times \left(1 + \frac{F}{2^{m}}\right)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| sign | 1 bit: 0 for positive, 1 for negative | 0 |
| $(-1)^{\text{sign}}$ | +1 or −1 | +1 |
| $E$ | the exponent field, read as a whole number | 125 |
| bias | a fixed offset, $2^{e-1} - 1$ for $e$ exponent bits, so negative exponents can be stored | 127 (bf16 and fp32) |
| $F$ | the fraction field, read as a whole number | 43 |
| $m$ | the number of fraction bits | 7 |
| $1 + F/2^m$ | the significant digits, between 1 and 2 (the leading 1 is implied, not stored) | 1.3359375 |

**In words:** "a float is plus or minus a number between 1 and 2, scaled by
a power of two; exponent bits choose the power, fraction bits choose the
number between 1 and 2."

**With the numbers:** 2^(−2) × (1 + 43/128) = **0.333984375**. The gap to the
next bf16 number above 1 is 2⁻⁷ = 0.0078; in fp32, with 23 fraction bits,
it is 2⁻²³ ≈ 0.00000012.

**In Python:**

```python
sign, E, bias, F, m = 0, 125, 127, 43, 7
# (−1)^sign × 2^(E − bias) × (1 + F / 2^m)
(-1) ** sign * 2.0 ** (E - bias) * (1 + F / 2 ** m)  # → 0.333984375
# the gap after 1 (the "epsilon") in bf16 and fp32
2.0 ** -7, 2.0 ** -23  # → (0.0078125, 1.1920928955078125e-07)
```

The formats that matter for training:

| Format | Exponent bits | Fraction bits | Largest | Smallest normal | Gap after 1 |
|---|---|---|---|---|---|
| fp32 | 8 | 23 | 3.4 × 10³⁸ | 1.2 × 10⁻³⁸ | 1.2 × 10⁻⁷ |
| bf16 | 8 | 7 | 3.4 × 10³⁸ | 1.2 × 10⁻³⁸ | 0.0078 |
| fp16 | 5 | 10 | 65,504 | 6.1 × 10⁻⁵ | 0.00098 |
| fp8 E5M2 | 5 | 2 | 57,344 | 6.1 × 10⁻⁵ | 0.25 |
| fp8 E4M3 | 4 | 3 | 448 | 0.016 | 0.125 |

Below the smallest normal number a format has a few **subnormal** values
that trade precision for extra range (fp16 reaches down to 6.0 × 10⁻⁸), and
below half of the smallest subnormal a number becomes 0: **underflow**.
Above the largest value is **overflow**, which becomes infinity.

![Horizontal bars on a log scale showing the range of each format from its smallest subnormal to its largest value: fp32 and bf16 span the same huge range, fp16 and fp8 are far narrower](figures/primer.ml.pretraining.float_ranges.svg)

**Reading it:** each bar covers the magnitudes one format can represent,
on a log scale where each tick is a factor of 10. The darker part is the
normal range and the lighter tail on the left is the subnormals. bf16's bar
is as long as fp32's: same 8 exponent bits, same range. fp16's is a small
fraction of it, and fp8's smaller still. Range is what decides whether a
gradient survives; precision (not shown) decides how finely it is recorded.

**In code:** `FloatFormat` describes a format by its bit counts (with `FP32`, `BF16`, `FP16`, `FP8_E5M2` and `FP8_E4M3` defined), and `quantize` rounds any number to the nearest value a format can hold, reproducing overflow, subnormals and underflow.

### 4b. Underflow, and loss scaling

**Everyday picture.** A kitchen scale that reads to the nearest gram
shows 0 for a pinch of saffron. Weigh the saffron together with a known
1 kg jar, subtract the jar afterwards, and the pinch shows up.

Gradients are often tiny. Many are smaller than fp16's smallest subnormal
(about 6 × 10⁻⁸), so a backward pass in fp16 silently turns them to 0 and
those weights stop learning. **Loss scaling** is the jar: multiply the loss
by a large number S before the backward pass, so every gradient is S times
larger (the chain rule passes the factor through unchanged); then divide
by S in fp32, before the update.

$$
g = \frac{\operatorname{fp16}\!\left(S \cdot \nabla \mathcal{L}\right)}{S}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $\nabla \mathcal{L}$ | the true gradient of one weight | 10⁻⁸ |
| $S$ | the loss scale, usually a power of two so multiplying is exact | 65,536 = 2¹⁶ |
| $\operatorname{fp16}(\ldots)$ | "rounded to the nearest fp16 value", which is where the backward pass stores it | |
| $g$ | the gradient the optimizer receives, after dividing in fp32 | ≈ 10⁻⁸ |

**In words:** "scale the loss up so the gradients are big enough for fp16,
compute them in fp16, and scale them back down in fp32."

**With the numbers:** unscaled, 10⁻⁸ is under half of fp16's smallest value
(5.96 × 10⁻⁸) and rounds to **0**. Scaled, 10⁻⁸ × 65,536 = 6.55 × 10⁻⁴, a
normal fp16 number; it is stored as 6.5517 × 10⁻⁴ and divided back to
**9.997 × 10⁻⁹**, within 0.03% of the truth.

**In Python:**

```python
import numpy as np
grad, S = 1e-8, 65536
# without scaling: fp16 flushes it to zero
float(np.float16(grad))  # → 0.0
# with scaling: fp16 holds S · grad, then divide in higher precision
f"{float(np.float16(S * grad)) / S:.4e}"  # → '9.9972e-09'
```

![Histogram of the size of a million simulated gradients on a log scale: unscaled, a large share falls left of fp16's smallest value and is lost; multiplied by 65,536 the whole histogram shifts right into fp16's range](figures/primer.ml.pretraining.loss_scaling.svg)

**Reading it:** the horizontal axis is a gradient's size in powers of two
and the height is how many gradients have that size. The shaded region on
the left is below fp16's smallest subnormal: whatever lands there becomes
0. The grey histogram is the unscaled gradients, with the lost share
printed; the blue one is the same gradients times 2¹⁶, shifted 16 steps to
the right, clear of the shaded region and still far from the overflow wall
on the right. Loss scaling does not change the shape, only where it sits.

A fixed scale is fragile: too small and gradients underflow, too large and
they overflow to infinity. **Dynamic loss scaling** adapts it: if any
gradient is infinite or not-a-number, skip the step and halve S; after a
long run of clean steps (2,000 is common), double S.

```mermaid
flowchart TB
  M[fp32 master weights] -->|cast| W16[16-bit weights]
  W16 --> F[Forward pass<br/>16-bit matmuls]
  F --> L[Loss, in fp32]
  L -->|times S| B[Backward pass<br/>16-bit gradients]
  B --> CK{Any inf or NaN?}
  CK -- yes --> SK[Skip the step<br/>halve S]
  CK -- no --> U[Divide by S in fp32<br/>clip, then Adam update]
  U --> M
  SK --> M
```

**Reading it:** one training step goes round the loop once. The weights live
in fp32 (top) but are cast to 16 bits for the expensive forward and backward
passes. The loss is multiplied by S before the backward pass. The diamond is
dynamic loss scaling's check: an overflow means S was too big, so the step
is thrown away and S halved; otherwise the gradients are unscaled, clipped
and applied to the fp32 master copy. With **bf16** the scale can usually be
dropped altogether, because bf16 has fp32's range: the same 10⁻⁸ gradient is
stored as 1.0012 × 10⁻⁸ without any help. That is why bf16 became the
default for training on hardware that supports it.

**In code:** `scaled_gradient_roundtrip` scales, stores and unscales one gradient, and `DynamicLossScaler.update` skips the step and halves the scale on overflow, or doubles it after a long enough run of clean steps.

### 4c. Why the master copy stays in fp32

**Everyday picture.** Pour a teaspoon of water into a full bathtub and
measure with a bucket: the level has not changed, as far as the bucket can
tell. Do it a thousand times and you have added four litres, but every
single measurement still reads "no change".

**Tiny worked example.** A weight of 1.0 receives an update of +0.0001 per
step. Next to 1.0, bf16 can only step in increments of 0.0078, so 1.0001
rounds straight back to 1.0. After 1,000 updates the bf16 weight is still
exactly 1.0; an fp32 weight has moved to 1.1, as it should.

That is why the optimizer keeps an fp32 **master copy** of the weights and
applies updates to it, even though the matrix multiplies use 16-bit copies.
The 16-bit copy is re-made from the master after every step.

**In Python:**

```python
import numpy as np
w16, w32 = np.float16(1.0), np.float32(1.0)
for _ in range(1000):
    w16 = np.float16(w16 + np.float16(1e-4))
    w32 = np.float32(w32 + np.float32(1e-4))
# every update lost in 16 bits, all kept in 32
float(w16), round(float(w32), 4)  # → (1.0, 1.1)
```

![A weight receiving 1,000 updates of 0.0001: in fp32 it climbs in a straight line from 1.0 to 1.1; stored in bf16 or fp16 it never leaves 1.0](figures/primer.ml.pretraining.master_weights.svg)

**Reading it:** the horizontal axis is the step and the vertical axis the
weight's value. The fp32 line rises steadily by 0.0001 per step. The bf16
and fp16 lines are flat at 1.0 for the whole run: each update is smaller
than half the gap to the next representable number, so it is rounded away
every time. The error is not noise that averages out; it is a systematic
loss of every small update, which is exactly what late training consists of.

**In code:** `accumulate_updates` adds the same update many times, rounding the weight to a chosen format after each step.

### 4d. fp8: the next halving

**Everyday picture.** A ruler with only eight marks is useless for
measuring a hair, unless you first slide it under a magnifying glass set to
the right zoom. fp8 is that short ruler, and a per-tensor scale is the
magnifying glass.

Recent GPUs multiply 8-bit floats at twice the 16-bit rate. With so few
bits, one format cannot cover everything, so training uses two: **E4M3**
(more precision, range to 448) for weights and activations, and **E5M2**
(more range, to 57,344) for gradients. Because the range is so short, every
tensor (or every small block of a tensor) gets its own scale factor,
chosen from its recent largest value, the same idea as loss scaling applied
everywhere. DeepSeek-V3 was trained largely in fp8 this way, keeping
sensitive parts (normalizations, the optimizer, the master weights) in
higher precision.

## 5. Stability at scale: loss spikes, clipping, warmup and checkpoints

**Everyday picture.** A ship on a months-long voyage does not assume the
sea stays calm. It trims the sails when gusts come (clipping), leaves port
slowly (warmup), keeps a log of its position (checkpoints), and has a
drill for when something goes wrong (rolling back).

A large run does see storms. The loss curve occasionally jumps upward, a
**loss spike**, sometimes recovering by itself and sometimes diverging for
good. Causes include a batch of bad data, a learning rate a little too high
for the model's current state, attention logits growing without bound, and
numerical overflow. And separately from the maths, hardware fails: in the
Llama 3 405B run, 419 unexpected interruptions occurred over 54 days,
about one every three hours.

```mermaid
flowchart LR
  T[Train step] --> W{Loss well above<br/>recent median?}
  W -- no --> CP{Checkpoint<br/>interval reached?}
  CP -- yes --> SV[Save weights,<br/>optimizer, data position]
  CP -- no --> T
  SV --> T
  W -- "yes, and it persists" --> RB[Roll back to a<br/>checkpoint before the spike]
  RB --> SKIP[Skip the batches<br/>around the spike]
  SKIP --> T
```

**Reading it:** the main loop is train, check, maybe save, repeat. The
upper diamond watches the loss. A brief blip is ignored, since clipping
usually absorbs it; a spike that persists sends the run back to an earlier
checkpoint, and the data batches that were in flight around the spike are
skipped. PaLM's authors did exactly this: restart about 100 steps before the
spike and skip roughly 200 to 500 batches, which removed the spikes. A
checkpoint must include the optimizer state and the position in the data,
or the resumed run is not the same run.

### 5a. Spotting a spike

**Tiny worked example.** Losses 3.0, 2.9, 2.8, 2.8, 2.7, then **8.5**. The
median of the previous four is 2.8, and 8.5 is more than twice that, so step
5 is flagged. The next step, 4.0, is compared with the median of 2.8, 2.8,
2.7, 8.5, which is still 2.8: one spike does not raise the bar, which is
why the rule uses the median and not the mean.

$$
\text{spike at } t \iff \mathcal{L}_t > f \cdot \operatorname{median}\big(\mathcal{L}_{t-w}, \ldots, \mathcal{L}_{t-1}\big)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $\mathcal{L}_t$ | the training loss at step $t$ | 8.5 at $t = 5$ |
| $w$ | how many recent steps to look back over (the window) | 4 |
| $\operatorname{median}$ | the middle value after sorting; one outlier cannot move it far | 2.8 |
| $f$ | how many times the median counts as a spike | 2 |
| $\iff$ | "exactly when" | |

**In words:** "flag a step when its loss is more than f times the median of
the last w losses."

**With the numbers:** median(2.9, 2.8, 2.8, 2.7) = 2.8, and 8.5 > 2 × 2.8 =
5.6, so step 5 is flagged; 4.0 < 5.6, so step 6 is not.

**In Python:**

```python
import statistics
losses = [3.0, 2.9, 2.8, 2.8, 2.7, 8.5, 4.0, 2.7]
w, f = 4, 2.0
# L_t > f · median(L_{t−w}, …, L_{t−1})
[t for t in range(w, len(losses)) if losses[t] > f * statistics.median(losses[t - w:t])]  # → [5]
```

**In code:** `detect_spikes` applies the median rule to a whole loss history.

### 5b. Gradient clipping, when the gradient is spread across GPUs

Clipping by global norm (built in `primer.ml.optimizers`) caps the length
of the whole gradient vector: if it is longer than c, shrink every entry by
the same factor. At scale there is a twist. With ZeRO or tensor parallelism
no GPU holds the whole gradient, so none can measure its length alone. Each
GPU sums the squares of its own shard, one all-reduce adds those sums (a
single number per GPU, so it is nearly free), and every GPU takes the
square root and applies the same factor.

$$
\lVert g \rVert = \sqrt{\sum_{k=1}^{G} \sum_{j} g_{kj}^2},
\qquad
g \leftarrow g \cdot \min\!\left(1, \frac{c}{\lVert g \rVert}\right)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $g$ | the whole gradient, all parameters | (1, 2, 2, 4) |
| $g_{kj}$ | entry $j$ of the shard on GPU $k$ | GPU 1: (1, 2); GPU 2: (2, 4) |
| $\sum_j g_{kj}^2$ | GPU $k$'s local sum of squares | 5 and 20 |
| $\lVert g \rVert$ | the length (**norm**) of the whole gradient | 5 |
| $c$ | the clipping threshold | 1 |
| $\min(1, \ldots)$ | never scale up, only down | 0.2 |
| $\leftarrow$ | "is replaced by" | |

**In words:** "add up every GPU's sum of squares, take the square root to
get the total length, and if it is over the limit, shrink every entry on
every GPU by the same factor."

**With the numbers:** 5 + 20 = 25, √25 = 5; with c = 1 the factor is 1/5, so
GPU 1's shard becomes (0.2, 0.4) and GPU 2's (0.4, 0.8).

**In Python:**

```python
import math
shards = [[1.0, 2.0], [2.0, 4.0]]
# each GPU's local sum of squares
local = [sum(x * x for x in s) for s in shards]
local  # → [5.0, 20.0]
# all-reduce the sums, then the square root
norm = math.sqrt(sum(local))
norm  # → 5.0
c = 1.0
factor = min(1.0, c / norm)
[[x * factor for x in s] for s in shards]  # → [[0.2, 0.4], [0.4, 0.8]]
```

![Clean held-out loss over 60 steps of training on y = 3x with one corrupted batch at step 30: without clipping the loss leaps above 6,000 and takes dozens of steps to come back; with clipping it barely moves](figures/primer.ml.pretraining.bad_batch.svg)

**Reading it:** the horizontal axis is the training step and the vertical
axis the loss on clean data, on a log scale. Both runs have converged by
step 30, when one batch arrives with corrupted labels. Without clipping
(red), that single gradient is hundreds of times too large, the weight is
thrown far off, and the loss jumps to over 6,000 before slowly recovering.
With clipping at 1 (blue), the same batch can only move the weight by one
learning-rate step, and the loss rises to about 0.01. Clipping cannot tell a
bad batch from a good one; it just limits how much damage any one batch can
do.

**In code:** `sharded_global_norm` computes the norm from per-GPU sums of squares, and `train_through_a_bad_batch` trains through a corrupted batch with or without `primer.ml.optimizers.clip_by_global_norm`.

### 5c. Warmup

**Everyday picture.** Nobody floors the accelerator in a car they have
never driven; they ease on until they know how it responds.

At the start of training the weights are random and Adam's running averages
have seen only a handful of gradients, so its step sizes are unreliable. A
full learning rate at step 1 can throw the model into a region it never
recovers from. **Warmup** ramps the learning rate linearly from 0 to its
peak over the first few hundred to few thousand steps; `primer.ml.optimizers`
derives the warmup-then-cosine schedule with worked numbers. At scale it
matters more, not less: bigger models and bigger batches tolerate smaller
peak learning rates, and a spike early in a months-long run wastes the most.

### 5d. Checkpoints: how often to save

**Everyday picture.** Saving a long document every few seconds wastes time
on saving; saving once an hour risks losing an hour of work to a crash.
Somewhere in between is the least total waste.

**Tiny worked example.** Suppose writing a checkpoint pauses training for 1
minute, and the cluster suffers a failure every 180 minutes on average.
Saving every 19 minutes spends 1/19 of the time saving, and each failure
loses on average half an interval, 9.5 minutes, every 180 minutes. Both
costs come to about 5.3%, and their total, 10.5%, is the smallest possible.

$$
\text{waste}(T) = \frac{C}{T} + \frac{T}{2M},
\qquad
T^{*} = \sqrt{2\,C\,M}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $T$ | time between checkpoints | 19 minutes |
| $C$ | time to write one checkpoint | 1 minute |
| $M$ | mean time between failures for the whole cluster | 180 minutes |
| $C/T$ | share of time spent saving | 1/19 |
| $T/(2M)$ | share of time redoing lost work: on average half an interval per failure | 9.5/180 |
| $T^{*}$ | the interval with the least total waste (Young, 1974) | 18.97 minutes |

**In words:** "saving often wastes time saving and saving rarely wastes
time redoing; the best interval is the square root of twice the save time
times the time between failures."

**With the numbers:** √(2 × 1 × 180) = √360 = **18.97 minutes**, and the
waste is 1/18.97 + 18.97/360 = 0.053 + 0.053 = **10.5%**. Cut the save time to
10 seconds (by writing asynchronously, in the background, from every GPU's
shard at once) and the best interval falls to 7.7 minutes with only 4.3%
waste.

**In Python:**

```python
import math
C, M = 1.0, 180.0
# T* = √(2 C M)
T = math.sqrt(2 * C * M)
round(T, 2)  # → 18.97
# waste(T) = C/T + T/(2M)
round(C / T + T / (2 * M), 3)  # → 0.105
# a 10-second save
C = 1 / 6
round(math.sqrt(2 * C * M), 1), round(C / math.sqrt(2 * C * M) + math.sqrt(2 * C * M) / (2 * M), 3)  # → (7.7, 0.043)
```

![Share of time wasted against checkpoint interval for a 1-minute and a 10-second save with a failure every 3 hours: each curve is a valley whose floor is marked at the square-root interval](figures/primer.ml.pretraining.checkpoint_interval.svg)

**Reading it:** the horizontal axis is the checkpoint interval in minutes
(log scale) and the vertical axis the share of time lost. Each curve is a
valley: on the left, saving too often; on the right, losing too much work
per failure. The dots mark √(2CM), the bottom of each valley. A faster save
moves the whole valley down and to the left, which is why large training
systems invest heavily in fast, asynchronous checkpointing: at thousands of
GPUs, failures are not an exception but the weather.

**In code:** `wasted_fraction` is the waste formula and `optimal_checkpoint_interval` is Young's square-root rule.

## In 20 seconds

- **Data:** pretraining data is mostly web crawl, pushed through language
  ID, quality rules, a quality classifier, and exact and near-duplicate
  removal (MinHash with LSH); most of the crawl is discarded. Duplicates
  cause memorization and waste compute.
- **Mixture and budget:** sources are mixed by weight, not size; a
  compute-optimal budget is about 20 tokens per parameter, compute is about
  6·N·D, and models meant for heavy use are trained far longer.
- **Memory:** Adam in mixed precision needs 16 bytes per parameter before
  activations (112 GB for 7B), so training needs many GPUs.
- **Parallelism:** data parallel (all-reduce gradients), ZeRO/FSDP (shard
  the state), tensor parallel (split each matmul, inside a server) and
  pipeline parallel (split the layers, mind the bubble (p − 1)/(m + p − 1)).
- **Precision:** compute in bf16 (fp32's range, less precision) or fp8 with
  scaling; fp16 needs loss scaling; the master weights stay in fp32 so small
  updates are not rounded away.
- **Stability:** warmup, global-norm clipping, spike detection with
  rollback, and checkpoints every √(2·C·M).

## Self-test questions

**Why does a pretraining pipeline run deduplication after the quality
filters, not before?**
Language ID and quality rules look at one page at a time, so they are cheap
per page. Near-duplicate detection compares pages with one another, which is
the expensive step. Running the cheap filters first means the expensive one
sees far fewer pages.

**How can MinHash estimate the overlap of two documents without comparing
their contents?**
Under a random ordering of all shingles, two sets have the same first
member with probability equal to their Jaccard similarity. A signature
records each document's first member under k random hash functions, so the
fraction of matching slots estimates the Jaccard. LSH then groups
signatures by bands so only likely pairs are ever compared.

**Why do duplicated documents hurt a model, when more data usually helps?**
A repeated document gets many times the training signal of any other, so
the model memorizes it and tends to regurgitate it; the repeats also spend
compute that would have taught something new, and copies of benchmark
questions contaminate evaluations.

**Where do the 16 bytes per parameter come from, and what do they mean for
a 7B model?**
2 bytes for the bf16 weight, 2 for its gradient, and 12 for fp32 state: the
master weight and Adam's two running averages. 16 × 7 × 10⁹ = 112 GB, more
than one 80 GB GPU holds, before any activations.

**What does each ZeRO stage shard, and what does it cost?**
Stage 1 shards the optimizer state, stage 2 also the gradients, stage 3
(FSDP) also the weights, dividing each by the number of GPUs. Stages 1 and 2
cost no more communication than plain data parallelism; stage 3 adds
all-gathers of each layer's weights in both passes, about 1.5 times the
traffic.

**Why is tensor parallelism kept inside one server while pipeline
parallelism spans servers?**
Tensor parallelism exchanges partial results inside every layer, so it
needs the fastest links, which exist only between GPUs in the same server.
Pipeline parallelism only passes activations at stage boundaries, a small
and infrequent exchange that slower links between servers can carry.

**What is the pipeline bubble, and how do you shrink it?**
The time stages sit idle while the pipeline fills and drains: (p − 1)/(m +
p − 1) of the schedule for p stages and m micro-batches. More micro-batches
shrink it (4 stages: 75% with 1, 8.6% with 32), as do schedules that
interleave forward and backward passes.

**Why does fp16 training need loss scaling while bf16 usually does not?**
fp16 has 5 exponent bits, so its smallest value is about 6 × 10⁻⁸ and many
gradients underflow to zero; multiplying the loss by a large scale lifts
them into range. bf16 keeps fp32's 8 exponent bits and therefore its range,
giving up precision instead.

**Why keep an fp32 copy of the weights if the maths runs in 16 bits?**
Late in training, updates are tiny compared with the weights. Next to 1.0
the bf16 grid spacing is about 0.008, so an update of 0.0001 rounds away
completely, every step. Applying updates to an fp32 master copy keeps them.

**How often should a large run write checkpoints?**
Roughly every √(2·C·M), where C is the time to save and M the mean time
between failures: saving more often wastes time saving, less often wastes
work redone after failures. Faster, asynchronous saves allow more frequent
checkpoints and less waste.

## The papers behind this lesson

- **Rae et al., *Scaling Language Models: Methods, Analysis & Insights from
  Training Gopher* (2021)**: https://arxiv.org/abs/2112.11446. Among much
  else, published the simple quality rules (length, symbols, stop words)
  that many open pipelines still apply.
- **Lee et al., *Deduplicating Training Data Makes Language Models Better*
  (2021)**: https://arxiv.org/abs/2107.06499. Showed that exact and MinHash
  near-duplicate removal cuts verbatim memorization about tenfold without
  hurting quality.
- **Penedo et al., *The FineWeb Datasets: Decanting the Web for the Finest
  Text Data at Scale* (2024)**: https://arxiv.org/abs/2406.17557. Documented
  and ablated a full open curation pipeline over 96 Common Crawl snapshots,
  including the classifier-filtered FineWeb-Edu.
- **Hoffmann et al., *Training Compute-Optimal Large Language Models*
  (2022)**: https://arxiv.org/abs/2203.15556. Found that parameters and
  training tokens should grow together, about 20 tokens per parameter.
- **Shumailov et al., *The Curse of Recursion: Training on Generated Data
  Makes Models Forget* (2023)**: https://arxiv.org/abs/2305.17493. Showed
  that models trained recursively on their own outputs lose the tails of the
  original distribution: model collapse.
- **Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training
  Trillion Parameter Models* (2019)**: https://arxiv.org/abs/1910.02054.
  Introduced the 16-bytes-per-parameter accounting and the three stages of
  sharding the training state across data-parallel GPUs.
- **Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language
  Models Using Model Parallelism* (2019)**: https://arxiv.org/abs/1909.08053.
  Split transformer layers across GPUs by columns and rows, with one
  all-reduce per block.
- **Huang et al., *GPipe: Efficient Training of Giant Neural Networks using
  Pipeline Parallelism* (2018)**: https://arxiv.org/abs/1811.06965. Split a
  model into stages fed by micro-batches, and analysed the resulting bubble.
- **Micikevicius et al., *Mixed Precision Training* (2017)**:
  https://arxiv.org/abs/1710.03740. Introduced the fp16 recipe: an fp32
  master copy of the weights, loss scaling, and fp32 accumulation.
- **Korthikanti et al., *Reducing Activation Recomputation in Large
  Transformer Models* (2022)**: https://arxiv.org/abs/2205.05198. Counted
  the activation memory of a transformer layer and showed how to recompute
  only the parts that are cheap to recompute.

## Further reading

- Micikevicius et al., *FP8 Formats for Deep Learning* (2022): https://arxiv.org/abs/2209.05433
- Kalamkar et al., *A Study of BFLOAT16 for Deep Learning Training* (2019): https://arxiv.org/abs/1905.12322
- Zhao et al., *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel* (2023): https://arxiv.org/abs/2304.11277
- Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM* (2021): https://arxiv.org/abs/2104.04473
- Chen et al., *Training Deep Nets with Sublinear Memory Cost* (activation checkpointing, 2016): https://arxiv.org/abs/1604.06174
- Chowdhery et al., *PaLM: Scaling Language Modeling with Pathways* (loss spikes and rollback, 2022): https://arxiv.org/abs/2204.02311
- Touvron et al., *LLaMA: Open and Efficient Foundation Language Models* (data mixture, 2023): https://arxiv.org/abs/2302.13971
- Llama Team, *The Llama 3 Herd of Models* (4D parallelism and failures at 16K GPUs, 2024): https://arxiv.org/abs/2407.21783
- DeepSeek-AI, *DeepSeek-V3 Technical Report* (fp8 training, 2024): https://arxiv.org/abs/2412.19437
- PyTorch automatic mixed precision: https://pytorch.org/docs/stable/amp.html
- PyTorch FullyShardedDataParallel: https://pytorch.org/docs/stable/fsdp.html
"""

from __future__ import annotations

import hashlib
import math
import re
import zlib
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from primer._show import banner, say, table, takeaway
from primer.ml.optimizers import clip_by_global_norm

# ---------------------------------------------------------------------------
# 1. Data: language ID, quality filters, deduplication, mixtures, budgets
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[a-zà-ÿ0-9']+")

# The ten commonest little words of each language. Real language ID (fastText's
# lid.176, CLD3) learns from character n-grams across 100+ languages; counting
# function words is the same idea at toy scale.
STOP_WORDS = {
    "en": {"the", "and", "of", "to", "is", "in", "that", "it", "was", "for"},
    "fr": {"le", "la", "les", "et", "est", "un", "une", "des", "du", "dans"},
    "de": {"der", "die", "das", "und", "ist", "nicht", "ein", "eine", "zu", "mit"},
    "es": {"el", "los", "las", "y", "es", "una", "que", "por", "con", "del"},
}


def words(text: str) -> list[str]:
    """Lowercased words: the unit every filter below counts."""
    return _WORD.findall(text.lower())


def detect_language(text: str, min_share: float = 0.1) -> str:
    """The language whose stop words make up the largest share of the text, or "unknown".

    A page of product codes or numbers matches no language's little words at
    all, so it falls under `min_share` and is reported as unknown.
    """
    ws = words(text)
    if not ws:
        return "unknown"
    shares = {lang: sum(w in sw for w in ws) / len(ws) for lang, sw in STOP_WORDS.items()}
    best = max(shares, key=shares.get)
    return best if shares[best] >= min_share else "unknown"


# Rae et al. (2021), the Gopher paper's "MassiveText" quality rules, Appendix A.
GOPHER_STOP_WORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}


def quality_failures(text: str) -> list[str]:
    """Names of the heuristic quality rules a document breaks (empty list = keep it).

    Each rule is cheap, obvious once stated, and removes a whole genre of junk:
    navigation snippets, hashtag spam, keyword stuffing, lists of links.
    """
    raw = text.split()
    n = len(raw)
    lines = [l for l in text.splitlines() if l.strip()] or [text]
    failures = []
    if n < 50:
        failures.append("too_short")
    if n > 100_000:
        failures.append("too_long")
    if n and not 3 <= sum(len(w) for w in raw) / n <= 10:
        failures.append("odd_word_length")
    symbols = text.count("#") + text.count("...") + text.count("…")
    if n and symbols / n > 0.1:
        failures.append("too_many_symbols")
    if sum(l.lstrip().startswith(("-", "*", "•")) for l in lines) / len(lines) > 0.9:
        failures.append("mostly_bullets")
    if sum(l.rstrip().endswith(("...", "…")) for l in lines) / len(lines) > 0.3:
        failures.append("mostly_ellipsis")
    if n and sum(any(c.isalpha() for c in w) for w in raw) / n < 0.8:
        failures.append("too_few_alphabetic_words")
    if len(GOPHER_STOP_WORDS & set(words(text))) < 2:
        failures.append("missing_stop_words")
    return failures


# Tiny labelled sets for the classifier: "reference" text in the style of an
# encyclopedia or textbook, and "raw crawl" text in the style of spam pages.
GOOD_EXAMPLES = [
    "the river deposits sand at its mouth and over many years builds a delta",
    "a delta is formed where a river meets the sea and slows down",
    "the cell is the basic unit of life and every organism is made of cells",
    "light travels faster than sound which is why we see lightning before thunder",
    "the history of the city begins with a small settlement on the river",
    "photosynthesis converts light into chemical energy stored in sugar",
]
BAD_EXAMPLES = [
    "click here buy now limited offer best price",
    "free free free win a prize click now",
    "best deals cheap price buy buy buy",
    "subscribe now click the link for free coupons",
    "hot singles best price click here now",
    "lose weight fast buy now free shipping",
]


@dataclass
class QualityClassifier:
    """A bag-of-words naive Bayes scorer: log-odds that a text looks like the reference set.

    GPT-3 filtered Common Crawl with a linear classifier of this kind (reference
    text vs. raw crawl); FineWeb-Edu used an LLM's educational-value ratings to
    train one. The score is a sum of per-word votes, so it is fast enough to
    run over billions of pages.
    """

    good: Counter = field(default_factory=Counter)
    bad: Counter = field(default_factory=Counter)

    @classmethod
    def trained_on_examples(cls, good=GOOD_EXAMPLES, bad=BAD_EXAMPLES) -> "QualityClassifier":
        clf = cls()
        for t in good:
            clf.good.update(words(t))
        for t in bad:
            clf.bad.update(words(t))
        return clf

    def word_vote(self, w: str) -> float:
        """log P(w | good) − log P(w | bad), with add-one smoothing so unseen words never give log 0."""
        vocab = len(set(self.good) | set(self.bad))
        p_good = (self.good[w] + 1) / (sum(self.good.values()) + vocab)
        p_bad = (self.bad[w] + 1) / (sum(self.bad.values()) + vocab)
        return math.log(p_good / p_bad)

    def score(self, text: str) -> float:
        """Sum of word votes over words the classifier has seen. Above 0 leans reference, below 0 leans spam."""
        known = set(self.good) | set(self.bad)
        return sum(self.word_vote(w) for w in words(text) if w in known)


def normalize_for_exact(text: str) -> str:
    """Lowercase and collapse whitespace, so trivially reformatted copies hash the same."""
    return " ".join(text.lower().split())


def exact_dedup(docs: list[str]) -> list[str]:
    """Keep the first copy of each document, comparing a hash of its normalized text."""
    seen, kept = set(), []
    for d in docs:
        key = hashlib.sha1(normalize_for_exact(d).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            kept.append(d)
    return kept


def shingles(text: str, k: int = 5) -> set[str]:
    """The set of overlapping k-word windows ("shingles") in a text."""
    ws = words(text)
    return {" ".join(ws[i : i + k]) for i in range(max(1, len(ws) - k + 1))}


def jaccard(a: set, b: set) -> float:
    """|A ∩ B| / |A ∪ B|: shared shingles over all distinct shingles."""
    return len(a & b) / len(a | b) if a | b else 1.0


_MERSENNE = (1 << 31) - 1  # a prime, and a·x + b stays below 2⁶³ so int64 never overflows


class MinHasher:
    """k random hash functions; a document's signature is its smallest hash under each.

    Two documents agree in one slot with probability exactly their Jaccard
    similarity, so the fraction of agreeing slots estimates it, from k numbers
    per document instead of the whole shingle set.
    """

    def __init__(self, num_perm: int = 128, seed: int = 0):
        rng = np.random.default_rng(seed)
        # h_i(x) = (a_i·x + b_i) mod p: a cheap family that behaves like random orderings.
        self.a = rng.integers(1, _MERSENNE, num_perm, dtype=np.int64)
        self.b = rng.integers(0, _MERSENNE, num_perm, dtype=np.int64)

    def signature(self, shingle_set: set[str]) -> np.ndarray:
        # crc32 rather than hash(): Python salts hash() per process, which would break determinism.
        x = np.array([zlib.crc32(s.encode()) % _MERSENNE for s in shingle_set], dtype=np.int64)
        if x.size == 0:
            return np.full(self.a.shape, _MERSENNE, dtype=np.int64)
        return ((self.a[:, None] * x[None, :] + self.b[:, None]) % _MERSENNE).min(axis=1)  # (num_perm,)


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Fraction of signature slots where the two minimums agree."""
    return float(np.mean(sig_a == sig_b))


def lsh_candidate_probability(s: float, bands: int, rows: int) -> float:
    """Chance that a pair with Jaccard s shares at least one whole band: 1 − (1 − sʳ)ᵇ."""
    return 1 - (1 - s**rows) ** bands


def near_duplicate_pairs(docs: list[str], bands: int = 20, rows: int = 5, threshold: float = 0.7, seed: int = 0) -> list[tuple[int, int]]:
    """Pairs (i, j), i < j, whose MinHash similarity is at least `threshold`.

    Instead of comparing all n² pairs, each signature is cut into `bands` bands
    of `rows` numbers and each band is hashed into a bucket. Only documents that
    land in the same bucket for some band are compared at all.
    """
    hasher = MinHasher(bands * rows, seed)
    sigs = [hasher.signature(shingles(d)) for d in docs]
    buckets: dict[tuple, list[int]] = {}
    for i, sig in enumerate(sigs):
        for band in range(bands):
            buckets.setdefault((band, *sig[band * rows : (band + 1) * rows].tolist()), []).append(i)
    candidates = {(i, j) for ids in buckets.values() for i in ids for j in ids if i < j}
    return sorted((i, j) for i, j in candidates if estimate_jaccard(sigs[i], sigs[j]) >= threshold)


CRAWL_SAMPLE = [
    # 0: a clean English paragraph
    "The river carries sand and small stones from the mountains to the sea. Over thousands of years "
    "that sand settles at the mouth of the river and builds a wide, flat delta. Farmers have grown rice "
    "on these deltas for centuries, because the soil is rich and the water is close. When the river "
    "floods, it brings new soil with it, which is why the land stays fertile.",
    # 1: a clean French paragraph (kept by a French pipeline, not by this English one)
    "Le fleuve transporte le sable et les pierres des montagnes vers la mer. Avec le temps, le sable se "
    "dépose à l'embouchure et forme un delta. Les paysans cultivent le riz dans les deltas depuis des "
    "siècles, car la terre est riche et l'eau est proche.",
    # 2: a navigation bar
    "Home | About us | Contact | Privacy policy | Log in",
    # 3: document 0 again, re-crawled with different capitalization and spacing
    "the river carries sand and small stones from the mountains to the sea.  Over thousands of years "
    "that sand settles at the mouth of the river and builds a wide, flat delta. Farmers have grown rice "
    "on these deltas for centuries, because the soil is rich and the water is close. When the river "
    "floods, it brings new soil with it, which is why the land stays fertile.",
    # 4: hashtag spam
    " ".join(["#deal #sale #win the best price today"] * 12),
    # 5: document 0, lightly edited by a scraper site
    "The river carries sand and small stones from the mountains to the sea. Over thousands of years "
    "that sand settles at the mouth of the river and builds a wide, flat delta. Farmers have grown rice "
    "on these deltas for hundreds of years, because the soil is rich and the water is close. When the "
    "river floods, it brings new soil with it, which is why the land stays fertile. Read more on our site.",
    # 6: a different clean English paragraph
    "Stars are born inside cold clouds of gas and dust. When part of a cloud becomes dense enough, gravity "
    "pulls it together faster than the gas can push back. The centre heats up as it shrinks, and once it "
    "reaches about ten million degrees, hydrogen begins to fuse into helium. That is the moment a star "
    "switches on, and it will shine with that fuel for millions or billions of years.",
]


def curate(docs: list[str], language: str = "en") -> list[str]:
    """Run the pipeline in order (language, quality, exact dedup, near dedup); one verdict per doc.

    Cheap checks run first so the expensive ones see fewer documents: language
    ID and the heuristics are a pass over each page, while near-duplicate
    detection compares pages with each other.
    """
    verdicts: list[str] = []
    kept_hashes: dict[str, int] = {}
    kept: list[int] = []
    hasher = MinHasher(100, seed=0)
    kept_sigs: dict[int, np.ndarray] = {}
    for i, d in enumerate(docs):
        lang = detect_language(d)
        if lang != language:
            verdicts.append(f"language: {lang}")
            continue
        failures = quality_failures(d)
        if failures:
            verdicts.append("quality: " + ", ".join(failures))
            continue
        key = hashlib.sha1(normalize_for_exact(d).encode()).hexdigest()
        if key in kept_hashes:
            verdicts.append(f"exact duplicate of {kept_hashes[key]}")
            continue
        sig = hasher.signature(shingles(d))
        twin = next((j for j in kept if estimate_jaccard(sig, kept_sigs[j]) >= 0.7), None)
        if twin is not None:
            verdicts.append(f"near duplicate of {twin}")
            continue
        kept_hashes[key], kept_sigs[i] = i, sig
        kept.append(i)
        verdicts.append("kept")
    return verdicts


BOILERPLATE = "click here to subscribe for free updates"
OTHER_LINES = [
    "click the map to see the river",
    "we walk here every day",
    "we went to the river",
    "subscribe for the weekly news",
    "free maps for every school",
]


def verbatim_probability(copies: int, line: str = BOILERPLATE) -> float:
    """Probability a bigram model, trained on the small corpus plus `copies` of the line, continues the line's
    first word into the whole line.

    A bigram model predicts each word from the one before it, by counting.
    Every duplicate adds the same counts again, so the line's path through the
    counts becomes the only road: that is memorization in its simplest form.
    """
    corpus = OTHER_LINES + [line] * copies
    pairs = Counter((a, b) for s in corpus for a, b in zip(s.split(), s.split()[1:]))
    firsts = Counter(a for (a, _), c in pairs.items() for _ in range(c))
    ws = line.split()
    return float(np.prod([pairs[(a, b)] / firsts[a] for a, b in zip(ws, ws[1:])]))


def epochs_per_source(weights: dict[str, float], sizes: dict[str, float], budget: float) -> dict[str, float]:
    """How many passes over each source a mixture implies: weight × budget / tokens available."""
    return {k: weights[k] * budget / sizes[k] for k in weights}


def chinchilla_tokens(params: float) -> float:
    """Compute-optimal training tokens for a model size: about 20 per parameter (Hoffmann et al., 2022)."""
    return 20 * params


def training_flops(params: float, tokens: float) -> float:
    """C ≈ 6·N·D: 2 FLOPs per parameter per token forward, 4 backward."""
    return 6 * params * tokens


def recursive_gaussian_fit(generations: int = 200, n: int = 20, seed: int = 0) -> list[float]:
    """Model collapse in miniature: fit a bell curve, sample from the fit, refit on the samples, repeat.

    Returns the fitted standard deviation of every generation. Each refit on a
    finite sample loses a little of the tails, and the losses compound.
    """
    rng = np.random.default_rng(seed)
    mu, sigma = 0.0, 1.0
    stds = []
    for _ in range(generations):
        sample = rng.normal(mu, sigma, n)
        mu, sigma = float(sample.mean()), float(sample.std())  # maximum-likelihood fit (divides by n)
        stds.append(sigma)
    return stds


# ---------------------------------------------------------------------------
# 2. Memory: why one GPU is not enough
# ---------------------------------------------------------------------------


def training_memory(params: float) -> dict[str, float]:
    """Bytes of training state for Adam in mixed precision, the "16 bytes per parameter" rule."""
    parts = {
        "weights (bf16)": 2 * params,
        "gradients (bf16)": 2 * params,
        "master weights (fp32)": 4 * params,
        "Adam momentum (fp32)": 4 * params,
        "Adam variance (fp32)": 4 * params,
    }
    return {**parts, "total": sum(parts.values())}


def activation_bytes(seq: int, batch: int, hidden: int, heads: int, layers: int,
                     store_scores: bool = True, checkpointed: bool = False) -> float:
    """Activations saved for the backward pass, in bytes (Korthikanti et al., 2022, no parallelism).

    Per layer: s·b·h·(34 + 5·a·s/h). The 34·s·b·h part is the layer's ordinary
    intermediate tensors; the 5·a·s² part is attention's score matrices, which
    FlashAttention-style kernels recompute instead of storing
    (`store_scores=False`). With activation checkpointing only each layer's
    16-bit input (2·s·b·h bytes) is kept, and the rest is recomputed.
    """
    sbh = seq * batch * hidden
    if checkpointed:
        return 2 * sbh * layers
    per_layer = sbh * (34 + (5 * heads * seq / hidden if store_scores else 0))
    return per_layer * layers


def zero_memory_per_gpu(params: float, n_gpus: int, stage: int, optimizer_bytes: int = 12) -> float:
    """Bytes of training state per GPU under ZeRO stage 0 (plain data parallel) to 3 (FSDP).

    Stage 1 shards the optimizer state, stage 2 also the gradients, stage 3 also
    the weights. Whatever is sharded is divided by the number of GPUs.
    """
    p, k, n = params, optimizer_bytes, n_gpus
    return {0: (2 + 2 + k) * p, 1: 2 * p + 2 * p + k * p / n, 2: 2 * p + (2 + k) * p / n, 3: (2 + 2 + k) * p / n}[stage]


# ---------------------------------------------------------------------------
# 3. Parallelism: data, tensor and pipeline
# ---------------------------------------------------------------------------


def linear_regression_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Gradient of the mean squared error mean((Xw − y)²) with respect to w: (2/n)·Xᵀ(Xw − y)."""
    return 2 / len(y) * X.T @ (X @ w - y)


def ring_all_reduce(arrays: list[np.ndarray]) -> tuple[list[np.ndarray], list[int]]:
    """Sum equal-length arrays across N simulated workers arranged in a ring.

    Each worker cuts its array into N chunks. Reduce-scatter: for N − 1 steps,
    every worker passes one chunk to its right-hand neighbour, which adds it to
    its own copy; afterwards worker i owns the complete sum of one chunk.
    All-gather: for N − 1 more steps the finished chunks travel round the ring
    so everyone ends with every sum. Returns each worker's result and how many
    numbers each worker sent.
    """
    n = len(arrays)
    chunks = [np.array_split(a.astype(float).copy(), n) for a in arrays]  # chunks[worker][chunk]
    sent = [0] * n
    for step in range(n - 1):  # reduce-scatter
        # Worker i sends chunk (i − step) to worker i + 1. Read everything first: all sends are simultaneous.
        outgoing = [(i, (i - step) % n, chunks[i][(i - step) % n].copy()) for i in range(n)]
        for i, c, data in outgoing:
            chunks[(i + 1) % n][c] += data
            sent[i] += data.size
    for step in range(n - 1):  # all-gather: worker i now owns the full sum of chunk (i + 1)
        outgoing = [(i, (i + 1 - step) % n, chunks[i][(i + 1 - step) % n].copy()) for i in range(n)]
        for i, c, data in outgoing:
            chunks[(i + 1) % n][c] = data
            sent[i] += data.size
    return [np.concatenate(c) for c in chunks], sent


def all_reduce_traffic(size: float, n: int) -> float:
    """Data each worker sends in a ring all-reduce of `size`: 2·(N − 1)/N · size, almost flat in N."""
    return 2 * (n - 1) / n * size


def column_parallel_matmul(X: np.ndarray, W: np.ndarray, n: int) -> np.ndarray:
    """X @ W with W's columns split across n devices; the pieces are placed side by side (an all-gather)."""
    return np.concatenate([X @ W_i for W_i in np.array_split(W, n, axis=1)], axis=1)


def row_parallel_matmul(X: np.ndarray, W: np.ndarray, n: int) -> np.ndarray:
    """X @ W with W's rows (and X's matching columns) split across n devices; partial results are summed
    (an all-reduce)."""
    parts = zip(np.array_split(X, n, axis=1), np.array_split(W, n, axis=0))
    return sum(X_i @ W_i for X_i, W_i in parts)


def tensor_parallel_mlp(X: np.ndarray, W1: np.ndarray, W2: np.ndarray, n: int) -> np.ndarray:
    """ReLU(X W1) W2 as Megatron-LM splits it: W1 by columns, W2 by rows, one all-reduce at the end.

    Splitting W1 by columns gives each device whole hidden units, so the
    elementwise ReLU can run locally with no communication at all.
    """
    W1s, W2s = np.array_split(W1, n, axis=1), np.array_split(W2, n, axis=0)
    partials = [np.maximum(X @ a, 0) @ b for a, b in zip(W1s, W2s)]  # each device: (tokens, d_model)
    return sum(partials)  # the one all-reduce


def pipeline_schedule(stages: int, microbatches: int) -> np.ndarray:
    """GPipe's fill-and-drain schedule as a (stages, time) grid.

    Cell value j > 0: forward pass of micro-batch j; −j: its backward pass;
    0: the stage is idle (the bubble). Forward passes flow down the stages,
    then backward passes flow back up.
    """
    p, m = stages, microbatches
    span = m + p - 1
    grid = np.zeros((p, 2 * span), dtype=int)
    for s in range(p):
        for j in range(m):
            grid[s, s + j] = j + 1  # stage s can start micro-batch j once stage s−1 has finished it
            grid[s, span + (p - 1 - s) + j] = -(j + 1)  # backward starts at the last stage
    return grid


def bubble_fraction(stages: int, microbatches: int) -> float:
    """Share of the schedule each stage spends idle: (p − 1) / (m + p − 1)."""
    return (stages - 1) / (microbatches + stages - 1)


# ---------------------------------------------------------------------------
# 4. Mixed precision: number formats, loss scaling, master weights
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FloatFormat:
    """A binary floating-point format: 1 sign bit, `exp_bits` exponent bits, `man_bits` fraction bits."""

    name: str
    exp_bits: int
    man_bits: int
    max_override: float | None = None  # E4M3 spends its top code on NaN only, so it reaches 448 not 240

    @property
    def bias(self) -> int:
        return 2 ** (self.exp_bits - 1) - 1

    @property
    def max_value(self) -> float:
        if self.max_override is not None:
            return self.max_override
        return (2 - 2.0**-self.man_bits) * 2.0**self.bias

    @property
    def min_normal(self) -> float:
        return 2.0 ** (1 - self.bias)

    @property
    def min_subnormal(self) -> float:
        return 2.0 ** (1 - self.bias - self.man_bits)

    @property
    def epsilon(self) -> float:
        """Gap between 1 and the next representable number: the format's relative precision."""
        return 2.0**-self.man_bits


FP32 = FloatFormat("fp32", 8, 23)
BF16 = FloatFormat("bf16", 8, 7)
FP16 = FloatFormat("fp16", 5, 10)
FP8_E4M3 = FloatFormat("fp8 E4M3", 4, 3, max_override=448.0)
FP8_E5M2 = FloatFormat("fp8 E5M2", 5, 2)
FORMATS = [FP32, BF16, FP16, FP8_E5M2, FP8_E4M3]


def quantize(x, fmt: FloatFormat):
    """Round x to the nearest value `fmt` can hold (ties to even), as the hardware does.

    Between 2ᵉ and 2ᵉ⁺¹ a format has 2^man_bits evenly spaced values, so the
    spacing is 2^(e − man_bits). Below the smallest normal number the spacing
    stops shrinking (subnormals), and anything under half the smallest
    subnormal rounds to 0: that is underflow. Above the largest value is
    overflow, shown here as infinity.
    """
    x = np.asarray(x, dtype=np.float64)
    mag = np.abs(x)
    e = np.floor(np.log2(np.where(mag > 0, mag, 1.0)))
    e = np.maximum(e, 1 - fmt.bias)  # subnormals share the smallest normal exponent's spacing
    step = 2.0 ** (e - fmt.man_bits)
    q = np.round(mag / step) * step  # np.round rounds halves to even, like IEEE hardware
    q = np.where(q > fmt.max_value, np.inf, q)
    out = np.sign(x) * q
    return float(out) if out.ndim == 0 else out


def scaled_gradient_roundtrip(grad: float, scale: float, fmt: FloatFormat) -> float:
    """Multiply by the loss scale, store in `fmt` (where the backward pass happens), divide back in fp32."""
    return quantize(grad * scale, fmt) / scale


@dataclass
class DynamicLossScaler:
    """Keep the loss scale as large as possible without overflowing.

    Overflow (an inf or NaN gradient) means the scale is too big: skip this
    step and halve it. A long run of clean steps means there may be headroom:
    double it.
    """

    scale: float = 2.0**16
    growth_interval: int = 2000
    clean_steps: int = 0

    def update(self, grads_finite: bool) -> bool:
        """Record one step's outcome; return True if the optimizer should apply this step."""
        if not grads_finite:
            self.scale /= 2
            self.clean_steps = 0
            return False
        self.clean_steps += 1
        if self.clean_steps == self.growth_interval:
            self.scale *= 2
            self.clean_steps = 0
        return True


def accumulate_updates(w0: float, update: float, steps: int, fmt: FloatFormat) -> float:
    """Add `update` to a weight `steps` times, rounding the weight to `fmt` after every step."""
    w = quantize(w0, fmt)
    for _ in range(steps):
        w = quantize(w + update, fmt)
    return w


# ---------------------------------------------------------------------------
# 5. Stability: clipping, spikes, checkpoints
# ---------------------------------------------------------------------------


def sharded_global_norm(shards: list[np.ndarray]) -> float:
    """‖g‖ when g is split across GPUs: each sums its own squares, one all-reduce adds the sums, then √."""
    local = [float(np.sum(s**2)) for s in shards]  # one number per GPU
    return math.sqrt(sum(local))  # the all-reduce of those numbers, then the square root


def train_through_a_bad_batch(clip: float | None, steps: int = 60, bad_step: int = 30, lr: float = 0.1,
                              seed: int = 0) -> list[float]:
    """SGD on y = 3x, with one corrupted batch (labels flipped and blown up 100×) at `bad_step`.

    Returns the loss on clean held-out data after every step. With `clip`, each
    gradient is scaled down to at most that length before the update.
    """
    rng = np.random.default_rng(seed)
    x_eval = rng.standard_normal(256)
    w = np.array([2.5])
    losses = []
    for t in range(steps):
        x = rng.standard_normal(32)
        y = -300 * x if t == bad_step else 3 * x
        g = linear_regression_gradient(x[:, None], y, w)
        if clip is not None:
            (g,), _ = clip_by_global_norm([g], clip)
        w = w - lr * g
        losses.append(float(np.mean((w[0] * x_eval - 3 * x_eval) ** 2)))
    return losses


def detect_spikes(losses: list[float], window: int = 50, factor: float = 2.0) -> list[int]:
    """Steps whose loss exceeds `factor` × the median of the previous `window` losses.

    The median, not the mean, so one spike does not raise the bar for the next.
    """
    return [i for i in range(window, len(losses)) if losses[i] > factor * float(np.median(losses[i - window : i]))]


def wasted_fraction(interval: float, cost: float, mtbf: float) -> float:
    """Share of time lost to checkpointing: saving (cost/interval) plus redoing lost work (interval/(2·mtbf))."""
    return cost / interval + interval / (2 * mtbf)


def optimal_checkpoint_interval(cost: float, mtbf: float) -> float:
    """Young's (1974) interval √(2·cost·mtbf), which minimises `wasted_fraction`."""
    return math.sqrt(2 * cost * mtbf)


# ---------------------------------------------------------------------------
# 6. Figures (rendered into the HTML docs by `make figures`)
# ---------------------------------------------------------------------------


def figures() -> dict:
    """Plot this lesson's data. matplotlib is imported here, and only here,
    so the lesson itself needs nothing beyond NumPy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, RED, GREEN, AMBER, MUTED = "#2563eb", "#dc2626", "#059669", "#d97706", "#9ca3af"
    figs = {}

    # --- 1. MinHash estimates converge on the true Jaccard -------------------
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ks = np.unique(np.logspace(0, np.log10(256), 40).astype(int))
    hasher = MinHasher(256, seed=1)
    for true_j, color in ((0.2, BLUE), (0.5, GREEN), (0.8, RED)):
        # |A| = |B| = 100 with overlap o gives J = o / (200 − o), so o = 200·J / (1 + J).
        o = round(200 * true_j / (1 + true_j))
        a = {f"s{i}" for i in range(100)}
        b = {f"s{i}" for i in range(100 - o, 200 - o)}
        sa, sb = hasher.signature(a), hasher.signature(b)
        ax.plot(ks, [estimate_jaccard(sa[:k], sb[:k]) for k in ks], color=color, label=f"true J = {jaccard(a, b):.2f}")
        ax.axhline(jaccard(a, b), color=color, ls="--", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of hash functions k")
    ax.set_ylabel("MinHash estimate of J")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("MinHash: more hashes, tighter estimate")
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.64))
    figs["minhash_convergence"] = fig

    # --- 2. LSH S-curves -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 3.4))
    s = np.linspace(0, 1, 201)
    for (b, r), color in (((50, 2), BLUE), ((20, 5), GREEN), ((10, 10), RED)):
        ax.plot(s, lsh_candidate_probability(s, b, r), color=color, label=f"{b} bands × {r} rows")
        ax.axvline((1 / b) ** (1 / r), color=color, ls="--", lw=1)
    ax.set_xlabel("true Jaccard similarity of a pair")
    ax.set_ylabel("chance the pair is compared")
    ax.set_title("LSH banding: an S-curve you can place")
    ax.legend(frameon=False, loc="lower right")
    figs["lsh_s_curve"] = fig

    # --- 3. Model collapse ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 3.4))
    gens = np.arange(1, 201)
    for seed, color in ((0, BLUE), (1, GREEN), (2, AMBER)):
        ax.semilogy(gens, recursive_gaussian_fit(200, 20, seed), color=color, lw=1.2, label=f"seed {seed}")
    ax.semilogy(gens, np.sqrt((19 / 20) ** gens), color="#4b5563", ls="--", label="average shrink √(0.95ᵗ)")
    ax.set_xlabel("generation (each fitted to 20 samples of the last)")
    ax.set_ylabel("fitted spread σ")
    ax.set_title("Training on your own samples: the spread collapses")
    ax.legend(frameon=False, loc="lower left")
    figs["model_collapse"] = fig

    # --- 4. The 7B memory bill ------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    state = training_memory(7e9)
    acts = {
        "naive": activation_bytes(4096, 1, 4096, 32, 32),
        "no stored scores": activation_bytes(4096, 1, 4096, 32, 32, store_scores=False),
        "checkpointed": activation_bytes(4096, 1, 4096, 32, 32, checkpointed=True),
    }
    colors = [BLUE, "#60a5fa", GREEN, "#34d399", "#a7f3d0"]
    x = np.arange(len(acts))
    bottom = np.zeros(len(acts))
    for (name, val), color in zip([(k, v) for k, v in state.items() if k != "total"], colors):
        ax.bar(x, val / 1e9, bottom=bottom, color=color, label=name, width=0.55)
        bottom += val / 1e9
    ax.bar(x, [a / 1e9 for a in acts.values()], bottom=bottom, color=AMBER, label="activations", width=0.55)
    for xi, a in zip(x, acts.values()):
        ax.text(xi, bottom[xi] + a / 1e9 + 3, f"{(state['total'] + a) / 1e9:.0f} GB", ha="center")
    ax.axhline(80, color=RED, ls="--", label="one 80 GB GPU")
    ax.set_xticks(x, list(acts))
    ax.set_ylabel("GB on one GPU")
    ax.set_ylim(0, 245)
    ax.set_title("Training a 7B model on one GPU: it does not fit")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    figs["memory_7b"] = fig

    # --- 5. ZeRO stages -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ns = 2 ** np.arange(0, 11)
    for stage, color in zip(range(4), (MUTED, BLUE, GREEN, RED)):
        ax.loglog(ns, [zero_memory_per_gpu(7e9, int(n), stage) / 1e9 for n in ns], "o-", color=color, ms=3,
                  label=["stage 0: plain data parallel", "stage 1: + shard optimizer", "stage 2: + shard gradients",
                         "stage 3 (FSDP): + shard weights"][stage])
    ax.axhline(80, color="#4b5563", ls="--")
    ax.text(2**7, 95, "80 GB GPU", color="#4b5563")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("GPUs sharing the state")
    ax.set_ylabel("training state per GPU (GB)")
    ax.set_title("ZeRO: sharding divides the 16 bytes per parameter")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    figs["zero_stages"] = fig

    # --- 6. The pipeline schedule --------------------------------------------
    from matplotlib.colors import ListedColormap

    p, m = 4, 8
    grid = pipeline_schedule(p, m)
    fig, ax = plt.subplots(figsize=(8, 2.6))
    kind = np.sign(grid) + 1  # 0 backward, 1 idle, 2 forward
    ax.imshow(kind, cmap=ListedColormap(["#86efac", "#ffffff", "#93c5fd"]), aspect="auto", vmin=0, vmax=2)
    for (row, col), v in np.ndenumerate(grid):
        if v:
            ax.text(col, row, str(abs(v)), ha="center", va="center", fontsize=8)
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, p, 1), minor=True)
    ax.grid(which="minor", color="#d1d5db", lw=0.6)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    ax.set_yticks(range(p), [f"GPU {i + 1}" for i in range(p)])
    ax.set_xticks(range(0, grid.shape[1], 2))
    ax.set_xlabel("time step (blue: forward of micro-batch n, green: backward, white: idle)")
    ax.set_title(f"GPipe schedule, {p} stages, {m} micro-batches: bubble = {bubble_fraction(p, m):.0%}")
    figs["pipeline_schedule"] = fig

    # --- 7. Bubble fraction ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ms = np.arange(1, 257)
    for p, color in zip((2, 4, 8, 16), (BLUE, GREEN, AMBER, RED)):
        ax.semilogx(ms, [bubble_fraction(p, int(m)) for m in ms], color=color, label=f"{p} stages")
    ax.axhline(0.1, color=MUTED, ls="--")
    ax.set_xlabel("micro-batches per batch")
    ax.set_ylabel("share of time idle")
    ax.set_title("The pipeline bubble: (p − 1) / (m + p − 1)")
    ax.legend(frameon=False)
    figs["bubble_fraction"] = fig

    # --- 8. Float ranges ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for i, fmt in enumerate(FORMATS):
        lo, mid, hi = np.log10(fmt.min_subnormal), np.log10(fmt.min_normal), np.log10(fmt.max_value)
        ax.barh(i, mid - lo, left=lo, color="#bfdbfe", height=0.6)
        ax.barh(i, hi - mid, left=mid, color=BLUE, height=0.6)
    ax.set_yticks(range(len(FORMATS)), [f.name for f in FORMATS])
    ax.invert_yaxis()
    ax.set_xlabel("log₁₀ of magnitude (light: subnormals, dark: normal range)")
    ax.set_title("Range of each format: bf16 keeps all of fp32's")
    figs["float_ranges"] = fig

    # --- 9. Loss scaling ------------------------------------------------------
    rng = np.random.default_rng(0)
    grads = np.exp(rng.normal(np.log(2.0**-22), 3.0, 1_000_000))  # sizes spread over many powers of two
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bins = np.arange(-50, 20, 0.5)
    lost = float(np.mean(quantize(grads, FP16) == 0))
    lost_scaled = float(np.mean(quantize(grads * 2.0**16, FP16) == 0))
    ax.hist(np.log2(grads), bins=bins, color=MUTED, alpha=0.8, label=f"unscaled: {lost:.0%} become 0 in fp16")
    ax.hist(np.log2(grads * 2.0**16), bins=bins, color=BLUE, alpha=0.6,
            label=f"× 2¹⁶: {lost_scaled:.2%} become 0")
    top = ax.get_ylim()[1] * 1.5  # headroom above the peaks for the labels and legend
    ax.set_ylim(0, top)
    ax.axvspan(-50, np.log2(FP16.min_subnormal / 2), color=RED, alpha=0.12)
    ax.axvline(np.log2(FP16.max_value), color=RED, ls="--")
    ax.text(np.log2(FP16.max_value) - 0.5, top * 0.93, "fp16 overflow", color=RED, ha="right")
    ax.text(-49, top * 0.93, "flushed to 0", color=RED)
    ax.set_xlabel("gradient size, log₂")
    ax.set_ylabel("number of gradients")
    ax.set_title("Loss scaling slides the gradients into fp16's range")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.88), fontsize=8)
    figs["loss_scaling"] = fig

    # --- 10. Master weights ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for fmt, color, ls in ((FP32, BLUE, "-"), (BF16, RED, "-"), (FP16, AMBER, ":")):
        w, path = quantize(1.0, fmt), []
        for _ in range(1000):
            w = quantize(w + 1e-4, fmt)
            path.append(w)
        ax.plot(range(1, 1001), path, color=color, ls=ls, lw=2, label=fmt.name)
    ax.set_xlabel("update step (each adds 0.0001)")
    ax.set_ylabel("weight value")
    ax.set_title("Tiny updates vanish in 16 bits; an fp32 master keeps them")
    ax.legend(frameon=False)
    figs["master_weights"] = fig

    # --- 11. One bad batch, with and without clipping ------------------------
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for clip, color, label in ((None, RED, "no clipping"), (1.0, BLUE, "clip global norm at 1")):
        losses = np.maximum(train_through_a_bad_batch(clip), 1e-12)
        ax.semilogy(losses, color=color, label=label)
    ax.axvline(30, color=MUTED, ls="--")
    ax.text(31, 1e-8, "corrupted batch", color="#4b5563")
    ax.set_xlabel("training step")
    ax.set_ylabel("clean held-out loss")
    ax.set_title("One bad batch: clipping limits the damage")
    ax.legend(frameon=False, loc="upper right")
    figs["bad_batch"] = fig

    # --- 12. Checkpoint interval ----------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 3.4))
    T = np.logspace(0, np.log10(300), 200)
    for cost, color, label in ((1.0, BLUE, "1-minute save"), (1 / 6, GREEN, "10-second save")):
        ax.semilogx(T, [wasted_fraction(t, cost, 180) for t in T], color=color, label=label)
        best = optimal_checkpoint_interval(cost, 180)
        ax.plot(best, wasted_fraction(best, cost, 180), "o", color=color)
        ax.annotate(f"{best:.1f} min, {wasted_fraction(best, cost, 180):.1%}", (best, wasted_fraction(best, cost, 180)),
                    textcoords="offset points", xytext=(6, -14), color=color)
    ax.set_xlabel("minutes between checkpoints (failure every 180 min)")
    ax.set_ylabel("share of time wasted")
    ax.set_ylim(0, 0.5)
    ax.set_title("Checkpoint interval: a valley at √(2·C·M)")
    ax.legend(frameon=False)
    figs["checkpoint_interval"] = fig

    return figs


# ---------------------------------------------------------------------------
# 7. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. Cleaning a crawl: language, quality, exact and near duplicates")
    say(
        """
        Seven pages arrive from a crawl. Cheap per-page checks run first
        (language ID, quality rules); the comparisons between pages
        (exact hash, then MinHash) run last, on what survived.
        """
    )
    labels = ["river paragraph", "French version", "navigation bar", "re-crawled copy", "hashtag spam",
              "scraper's edited copy", "stars paragraph"]
    table(["page", "what it is", "verdict"], [(i, l, v) for i, (l, v) in enumerate(zip(labels, curate(CRAWL_SAMPLE)))])
    clf = QualityClassifier.trained_on_examples()
    for text in ("the delta is formed when the river deposits sand over many years",
                 "click here buy now best price free free free"):
        print(f"  classifier score {clf.score(text):+7.2f}  {text!r}")
    print()
    takeaway("Most of a crawl never reaches the model: two of seven pages survive here.")

    banner("2. MinHash and LSH: near duplicates without comparing everything")
    a, b = CRAWL_SAMPLE[0], CRAWL_SAMPLE[5]
    true_j = jaccard(shingles(a), shingles(b))
    rows = []
    for k in (8, 32, 128, 512):
        h = MinHasher(k, seed=0)
        rows.append((k, true_j, estimate_jaccard(h.signature(shingles(a)), h.signature(shingles(b)))))
    table(["hashes k", "true Jaccard", "MinHash estimate"], rows, floatfmt=".3f")
    table(["similarity s", "P(candidate), 20 bands × 5 rows"],
          [(s, lsh_candidate_probability(s, 20, 5)) for s in (0.3, 0.5, 0.7, 0.8, 0.9)], floatfmt=".4f")
    pairs = near_duplicate_pairs([a, b, CRAWL_SAMPLE[6]])
    say(f"LSH over pages 0, 5 and 6 (listed as 0, 1, 2) flags {pairs}: pages 0 and 5, and nothing else.")

    banner("3. Why duplicates hurt: a bigram model memorizes what it sees often")
    table(["copies of the boilerplate line", "P(model continues 'click' into the whole line)"],
          [(c, verbatim_probability(c)) for c in (1, 10, 100, 1000)], floatfmt=".3f")
    takeaway("Repetition turns learning into recitation; deduplication is also privacy and eval hygiene.")

    banner("4. Mixtures, budgets and synthetic data")
    epochs = epochs_per_source({"web": 0.80, "wiki": 0.05, "code": 0.15},
                               {"web": 900e9, "wiki": 20e9, "code": 150e9}, budget=1000e9)
    table(["source", "weight", "passes over it"], [(k, w, epochs[k]) for k, w in (("web", 0.80), ("wiki", 0.05), ("code", 0.15))],
          floatfmt=".2f")
    say(f"A 7B model's compute-optimal budget: {chinchilla_tokens(7e9) / 1e9:.0f}B tokens, "
        f"costing {training_flops(7e9, chinchilla_tokens(7e9)):.2e} FLOPs.")
    stds = recursive_gaussian_fit()
    say(f"Refit on your own samples 200 times: spread 1.0 -> {stds[49]:.3f} after 50 generations, "
        f"{stds[-1]:.1e} after 200. Keep real data in the mix.")

    banner("5. The memory bill: why one GPU is not enough")
    mem = training_memory(7e9)
    table(["what", "GB for 7B"], [(k, v / 1e9) for k, v in mem.items()], floatfmt=".1f")
    table(["activations at 4,096 tokens", "GB"], [
        ("stored naively", activation_bytes(4096, 1, 4096, 32, 32) / 1e9),
        ("without attention scores", activation_bytes(4096, 1, 4096, 32, 32, store_scores=False) / 1e9),
        ("activation checkpointing", activation_bytes(4096, 1, 4096, 32, 32, checkpointed=True) / 1e9),
    ], floatfmt=".1f")
    table(["ZeRO stage", "GB per GPU, 7.5B on 64 GPUs"], [(s, zero_memory_per_gpu(7.5e9, 64, s) / 1e9) for s in range(4)],
          floatfmt=".2f")

    banner("6. Parallelism: all-reduce, tensor splits, the pipeline bubble")
    grads = [np.arange(8.0) * (i + 1) for i in range(4)]
    results, sent = ring_all_reduce(grads)
    say(f"Ring all-reduce over 4 workers: everyone ends with {results[0].tolist()}; "
        f"each sent {sent[0]} numbers, 1.5 times its own 8.")
    rng = np.random.default_rng(0)
    X, W1, W2 = rng.standard_normal((3, 8)), rng.standard_normal((8, 16)), rng.standard_normal((16, 8))
    err = np.abs(tensor_parallel_mlp(X, W1, W2, n=4) - np.maximum(X @ W1, 0) @ W2).max()
    say(f"Feed-forward layer split over 4 devices (columns, then rows): largest difference from one device = {err:.1e}.")
    print(pipeline_schedule(4, 4))
    print()
    table(["micro-batches (4 stages)", "bubble"], [(m, bubble_fraction(4, m)) for m in (1, 4, 8, 32)], floatfmt=".3f")

    banner("7. Mixed precision: range, precision, loss scaling, master weights")
    table(["format", "largest", "smallest normal", "gap after 1"],
          [(f.name, f"{f.max_value:.3g}", f"{f.min_normal:.3g}", f"{f.epsilon:.3g}") for f in FORMATS])
    say(f"A gradient of 1e-8 in fp16: {quantize(1e-8, FP16)}. Scaled by 65536, stored, unscaled: "
        f"{scaled_gradient_roundtrip(1e-8, 65536, FP16):.4e}. In bf16 with no scaling: {quantize(1e-8, BF16):.4e}.")
    scaler = DynamicLossScaler(scale=65536.0, growth_interval=3)
    events = [True, True, False, True, True, True]
    log = []
    for ok in events:
        applied = scaler.update(ok)
        log.append(("clean" if ok else "overflow", "applied" if applied else "skipped", scaler.scale))
    table(["gradients", "step", "scale after"], log, floatfmt=".0f")
    say(f"1,000 updates of 0.0001 to a weight of 1.0: fp32 ends at {accumulate_updates(1.0, 1e-4, 1000, FP32):.4f}, "
        f"bf16 at {accumulate_updates(1.0, 1e-4, 1000, BF16):.4f}.")
    takeaway("bf16 for the maths (fp32's range), fp32 for the master weights and optimizer state.")

    banner("8. Stability: clipping, spikes, checkpoints")
    no_clip, clipped = train_through_a_bad_batch(None), train_through_a_bad_batch(1.0)
    say(f"One corrupted batch at step 30: peak loss {max(no_clip):,.0f} without clipping, "
        f"{max(clipped[30:]):.4f} with clipping.")
    say(f"Spikes flagged in [3.0, 2.9, 2.8, 2.8, 2.7, 8.5, 4.0, 2.7]: steps "
        f"{detect_spikes([3.0, 2.9, 2.8, 2.8, 2.7, 8.5, 4.0, 2.7], window=4, factor=2.0)}.")
    table(["save time (min)", "best interval (min)", "time wasted"],
          [(c, optimal_checkpoint_interval(c, 180), wasted_fraction(optimal_checkpoint_interval(c, 180), c, 180))
           for c in (5.0, 1.0, 1 / 6)], floatfmt=".3f")
    takeaway("At thousands of GPUs, failures are the weather: clip, watch, roll back, and save often and fast.")


if __name__ == "__main__":
    demo()
