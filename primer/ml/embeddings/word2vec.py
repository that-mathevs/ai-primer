r"""
# word2vec: learning meaning from the company a word keeps

Run: `python -m primer.ml.embeddings.word2vec`

New to vectors, dot products or Σ? `primer.notation` builds them from zero.

## The everyday picture

You can learn a lot about a stranger from their friends. If two people keep
turning up with the same crowd, they probably have something in common.
Linguists put it the same way: *"You shall know a word by the company it
keeps"* (J. R. Firth, 1957). "Coffee" and "tea" both show up next to "cup",
"hot" and "morning", so they must be related.

word2vec turns that idea into a game. Every word gets a spot on a map of
meaning (a short list of numbers, its **vector**). The game: looking only at
a word's spot, guess which words sit near it in real sentences. Every wrong
guess nudges the spots a little. After millions of nudges, words that keep
the same company end up in the same neighbourhood, because that's the only
way to guess well for all of them.

## A tiny worked example: one sentence, one nudge

**Step 1, make guessing pairs.** Take the sentence "river bank fish" and a
**window** of 1 (look one word left and right). Each word, the **center**,
is paired with each neighbour, its **context**:

(river, bank), (bank, river), (bank, fish), (fish, bank)

No human labels anything: the text itself says which words appeared together.

**Step 2, score a pair.** Put everything in 2 dimensions. The center word has
vector v = (1, 0), its true context word has u = (0, 1), and one random
"noise" word (a word we pretend is a context, to have something to push away)
has uₙ = (0, −1). The score of a pair is their **dot product** (multiply
matching positions, add): v·u = 1·0 + 0·1 = 0, and v·uₙ = 0. Zero means "no
opinion".

**Step 3, turn scores into probabilities** with the **sigmoid** σ(x), which
squashes any number into 0 to 1 (σ(0) = 0.5, big positive → near 1, big
negative → near 0). The model says the true pair is 50% likely and the noise
pair is 50% likely. That's bad: we want ~100% and ~0%.

**Step 4, measure the mistake** as a **loss** (a single number that's big
when the guesses are bad): −ln σ(0) − ln σ(−0) = 0.693 + 0.693 = **1.386**.

**Step 5, nudge.** Move v a little toward u and away from uₙ, and move u and
uₙ a little toward or away from v. With a step size (**learning rate**) of
0.1 we get v = (1, 0.1), u = (0.05, 1), uₙ = (−0.05, −1). Now v·u = **0.15** and
v·uₙ = **−0.15**: the true pair scores higher, the noise pair lower, and the
loss drops to **1.242**. Word2vec is this nudge, repeated millions of times.

```mermaid
flowchart LR
  S["river bank fish"] --> W[Slide a window<br/>over each word]
  W --> P["(bank, river)<br/>(bank, fish) ..."]
  P --> T[Raise the score of<br/>each true pair]
  N[Draw k random<br/>noise words] --> T2[Lower the score of<br/>each noise pair]
  T --> U[Nudge the vectors<br/>in each pair]
  T2 --> U
  U -->|next pair| P
```

**Reading it:** the sentence becomes (center, context) pairs by a sliding
window. Each true pair is a positive example; alongside it we draw a few
random words as negatives ("bank" should *not* predict "throne"). The update
pulls true pairs together and pushes noise pairs apart, then moves on to the
next pair. The loop at the right is the whole training process.

**In code:** `skipgram_pairs` slides the window over a sentence and returns
every (center, context) pair; `build_corpus` writes the lesson's synthetic
sentences for it to slide over.

## The math: skip-gram with negative sampling (SGNS)

Every word has two vectors: vᵥ when it's the center and uᵥ when it's a
context. For a center c, its true context o and k noise words n₁ … nₖ:

$$
L = -\log \sigma(u_o \cdot v_c) - \sum_{i=1}^{k} \log \sigma(-u_{n_i} \cdot v_c),
\qquad \sigma(x) = \frac{1}{1 + e^{-x}}
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| L | the loss for one training pair: how wrong the guesses are | ≥ 0; 0 is perfect |
| c, o | the center word and its true context word | word ids |
| vc | the center word's vector | d numbers (d = 16 in this lesson) |
| uo | the true context word's vector | d numbers |
| nᵢ, u_{nᵢ} | the i-th noise word and its vector | k of them, d numbers each |
| k | how many noise words per true pair | 5 here; 5 to 20 in practice |
| · | dot product | one number |
| σ | sigmoid, squashes a score into a probability | 0 to 1 |
| e | Euler's number, ≈ 2.718 | |
| log | natural logarithm; −log p is large when p is small | |
| Σ | add up over i = 1 … k | |

**In words:** the loss is small when the model gives the true pair a high
probability and every noise pair a low one.

**On the example:** L = −log σ(0) − log σ(−0) = 0.693 + 0.693 = 1.386; after the
nudge, −log σ(0.15) − log σ(0.15) = 0.621 + 0.621 = 1.242.

**In Python:**

```python
import math
def sigma(x):
    # σ(x) = 1 / (1 + e^(-x))
    return 1 / (1 + math.exp(-x))
def dot(a, b):
    return sum(a_i * b_i for a_i, b_i in zip(a, b))
def L(v_c, u_o, noise):
    # -log σ(u_o · v_c)
    return (-math.log(sigma(dot(u_o, v_c)))
            # - Σ_i log σ(-u_ni · v_c)
            - sum(math.log(sigma(-dot(u_n, v_c))) for u_n in noise))
round(L((1, 0), (0, 1), [(0, -1)]), 3)  # → 1.386
# after the nudge
round(L((1, 0.1), (0.05, 1), [(-0.05, -1)]), 3)  # → 1.242
```

The nudge follows the **gradient**: for each vector, the direction in which
a small change would increase the loss fastest. We step the opposite way.
For this loss the gradients are short enough to derive by hand
(`sgns_grads`), using the fact that the slope of −log σ(x) is σ(x) − 1:

$$
g_o = \sigma(u_o \cdot v_c) - 1, \quad g_i = \sigma(u_{n_i} \cdot v_c), \quad
\frac{\partial L}{\partial v_c} = g_o u_o + \sum_i g_i u_{n_i}
$$

**Symbols**

| Symbol | Meaning here | Range |
|---|---|---|
| g_o | how far the true pair's probability is from 1, as a negative number | −1 to 0 |
| gᵢ | the i-th noise pair's probability, which should be 0 | 0 to 1 |
| ∂L/∂vc | gradient: how the loss changes as each number in vc changes | d numbers |

**In words:** move the center vector toward the true context in proportion
to how wrong that guess was, and away from each noise word in proportion to
how much it was wrongly believed.

**On the example:** g_o = 0.5 − 1 = −0.5 and g₁ = 0.5, so
∂L/∂vc = −0.5·(0, 1) + 0.5·(0, −1) = (0, −1). Stepping *against* it with learning
rate 0.1: vc = (1, 0) − 0.1·(0, −1) = (1, 0.1).

**In Python:**

```python
import math
def sigma(x):
    return 1 / (1 + math.exp(-x))
v_c, u_o, u_n = (1, 0), (0, 1), (0, -1)
# σ(u_o · v_c) - 1
g_o = sigma(sum(u * v for u, v in zip(u_o, v_c))) - 1
# σ(u_n1 · v_c)
g_1 = sigma(sum(u * v for u, v in zip(u_n, v_c)))
g_o, g_1  # → (-0.5, 0.5)
# ∂L/∂v_c = g_o u_o + Σ_i g_i u_ni
grad = [g_o * o + g_1 * n for o, n in zip(u_o, u_n)]
grad  # → [0.0, -1.0]
# step against it, learning rate 0.1
[v - 0.1 * g for v, g in zip(v_c, grad)]  # → [1.0, 0.1]
```

Noise words are drawn in proportion to their count raised to the 3/4 power:

$$
P(w) = \frac{\text{count}(w)^{0.75}}{\sum_{w'} \text{count}(w')^{0.75}}
$$

**Symbols**

| Symbol | Meaning here | Range |
|---|---|---|
| P(w) | chance that word w is drawn as a noise word | 0 to 1; all add to 1 |
| count(w) | how many times w appears in the training text | ≥ 1 |
| ^0.75 | raise to the power 3/4, which shrinks big counts more than small ones | |
| w′ | every word in the vocabulary, in turn | |
| Σ over w′ | add up over the whole vocabulary, so the shares sum to 1 | |

**In words:** common words are drawn as noise more often, but the 3/4 power
gives rare words a boost.

**On an example:** counts 1 and 16 become 1^0.75 = 1 and 16^0.75 = 8, so the
shares are 1/9 and 8/9 (0.111 and 0.889) instead of 1/17 and 16/17 (0.059
and 0.941).

**In Python:**

```python
counts = [1, 16]
# count(w)^0.75
weights = [count ** 0.75 for count in counts]
weights  # → [1.0, 8.0]
# divide by Σ over w′ so the shares add to 1
[round(w / sum(weights), 3) for w in weights]  # → [0.111, 0.889]
# without the 3/4 power
[round(c / sum(counts), 3) for c in counts]  # → [0.059, 0.941]
```

**In code:** `sgns_loss` computes L for one center, one context and k noise
words, `sgns_update` takes one step against the gradient, and
`noise_distribution` builds P(w). `train_sgns` runs the whole loop in
mini-batches and returns a `WordVectors` table; `gradient_check` confirms the
hand-derived gradients against a numerical estimate.

**Why it matters:** a full softmax over the vocabulary (`primer.ml.attention`
explains softmax) would score every word, 100,000 or more, for every
training pair. Negative sampling scores k + 1. That trick is what made
training on billions of words practical in 2013, and the same pull-together,
push-apart shape reappears in every modern embedding model
(`primer.ml.embeddings.contrastive`).

## The famous result: vector arithmetic

**Everyday picture:** on a street map, "walk two blocks east" is the same
walk wherever you start. word2vec's map works the same way: the walk from
"man" to "woman" is roughly the same walk as from "king" to "queen", because
in both cases the contexts change from he/his to she/her.

**Tiny example** with made-up 2-D vectors: man = (1, 0), woman = (1, 1),
king = (3, 0). Then king − man + woman = (3 − 1 + 1, 0 − 0 + 1) = (3, 1), exactly
where queen = (3, 1) would be.

$$
\hat{w} = \operatorname*{arg\,max}_{w \notin \{a, b, c\}} \cos(v_w,\; v_a - v_b + v_c)
$$

**Symbols**

| Symbol | Meaning here |
|---|---|
| a, b, c | the three given words, e.g. king, man, woman |
| v_a − v_b + v_c | start at king, remove the man direction, add the woman direction |
| cos | cosine similarity (`primer.ml.embeddings.similarity`) |
| arg max over w | the word w that makes the cosine largest |
| w ∉ {a, b, c} | skip the three input words themselves |
| ŵ | the answer |

**In words:** the answer is the word whose vector points most nearly the
same way as king minus man plus woman, not counting those three words.

**On the example:** v = (3, 1) and queen = (3, 1) point the same way, so the
cosine is 1, and queen wins.

**In Python:**

```python
import math
def cos(a, b):
    dot = sum(a_i * b_i for a_i, b_i in zip(a, b))
    return dot / (math.sqrt(sum(a_i ** 2 for a_i in a)) * math.sqrt(sum(b_i ** 2 for b_i in b)))
v = {"man": (1, 0), "woman": (1, 1), "king": (3, 0), "queen": (3, 1)}
a, b, c = "king", "man", "woman"
# v_a - v_b + v_c
target = [x_a - x_b + x_c for x_a, x_b, x_c in zip(v[a], v[b], v[c])]
target  # → [3, 1]
# w ∉ {a, b, c}
candidates = [w for w in v if w not in {a, b, c}]
# arg max of the cosine
w_hat = max(candidates, key=lambda w: cos(v[w], target))
w_hat, round(cos(v[w_hat], target), 2)  # → ('queen', 1.0)
```

This lesson trains on a small synthetic corpus built from three independent
attributes (male/female, royal/common, adult/child), so the effect appears
in seconds on a laptop.

![In a 2-D PCA view, the arrows king to queen, man to woman, prince to princess and boy to girl all point the same way with nearly equal length](figures/primer.ml.embeddings.word2vec.space.svg)

**Reading it:** these are the learned 16-dimensional vectors, squashed to 2-D
with **PCA** (a way to find the two directions along which the points are
most spread out, and draw the points along those; `primer.notation` explains
it). Arrows join male-female pairs: king→queen, man→woman, prince→princess,
boy→girl. The arrows are roughly parallel and about the same length. That
parallelism *is* the analogy: add the man→woman arrow to king and you land
near queen. The squashing loses some structure, so treat the picture as
intuition, not measurement.

**In code:** `analogy` returns the word nearest v_a − v_b + v_c, skipping the
three inputs; `nearest` lists a word's closest neighbours by cosine, and
`nearest_to_vector` does the same for any vector.

**Why it matters:** this was the first clear evidence that learned vectors
capture meaning as geometry. It's why "embedding" became the default way to
represent text, and why vector arithmetic (averaging, subtracting, finding
nearest neighbours) works on meaning.

## Counting instead of predicting: PPMI and GloVe

**Everyday picture:** instead of playing the guessing game, keep a tally
sheet: for every pair of words, count how often they appear near each other.
Then ask which pairs turn up together *more than chance would predict*.

**Tiny example:** two words that each appear only with themselves:
counts [[2, 0], [0, 2]]. Out of 4 sightings, each word appears half the time
(P = 0.5), and each diagonal pair also appears half the time. Chance would
predict 0.5 · 0.5 = 0.25, but we see 0.5, twice as often, so the score is
log(0.5 / 0.25) = log 2 = **0.693**. The off-diagonal pairs never co-occur, and
their score is clipped to **0**.

$$
\text{PMI}(w, c) = \log \frac{P(w, c)}{P(w)\,P(c)}, \qquad \text{PPMI} = \max(\text{PMI}, 0)
$$

**Symbols**

| Symbol | Meaning here | Range |
|---|---|---|
| w, c | a word and a context word | |
| P(w, c) | share of all co-occurrences that are this pair | 0 to 1 |
| P(w), P(c) | share of co-occurrences involving w (or c) at all | 0 to 1 |
| P(w)P(c) | how often the pair *would* appear if they were unrelated | |
| log | natural logarithm; log 1 = 0 means "exactly chance" | |
| PMI | pointwise mutual information: log of seen ÷ expected | any; > 0 means "together more than chance" |
| max(·, 0) | keep positive values, replace negatives with 0 | |

**In words:** PMI asks how many times more often two words appear together
than they would by chance, on a log scale; PPMI keeps only the "more often"
part.

**On the example:** PMI = log(0.5 / (0.5 · 0.5)) = log 2 = 0.693 on the
diagonal; the off-diagonal pairs have P(w, c) = 0, so PMI = −∞ and PPMI = 0.

**In Python:**

```python
import math
# co-occurrence counts
X = [[2, 0], [0, 2]]
# 4 sightings
total = sum(sum(row) for row in X)
# P(w): share of each row
P_w = [sum(row) / total for row in X]
# P(c): share of each column
P_c = [sum(col) / total for col in zip(*X)]
def ppmi(w, c):
    P_wc = X[w][c] / total
    # log 0 = -∞
    pmi = math.log(P_wc / (P_w[w] * P_c[c])) if P_wc else -math.inf
    # PPMI = max(PMI, 0)
    return max(pmi, 0.0)
[[round(ppmi(w, c), 3) for c in range(2)] for w in range(2)]  # → [[0.693, 0.0], [0.0, 0.693]]
```

The PPMI table has one row per word, as many columns as the vocabulary, and
is mostly zeros. **SVD** (singular value decomposition, see `primer.notation`)
compresses it into a few columns that keep its main patterns; those few
numbers per row are the word's embedding.

```mermaid
flowchart LR
  T[Text] --> C[Count co-occurrences<br/>in a window]
  C --> P[PPMI: keep pairs that meet<br/>more often than chance]
  P --> S[SVD: compress each row<br/>to d numbers]
  S --> E[Word vectors]
```

**Reading it:** the same text goes in as for word2vec, but instead of a
training loop there are three bulk steps: count, rescale by chance, compress.
Levy and Goldberg (2014) showed that word2vec's guessing game is secretly
doing the same thing: its vectors factorize a shifted PMI table.

**In code:** `cooccurrence` counts the pairs in a window, `ppmi` rescales the
table by chance and clips negatives to 0, and `ppmi_svd_embeddings` runs all
three steps and keeps d columns of the SVD.

**GloVe** (Pennington et al., 2014) is the best-known counting method. It
fits vectors so that each dot product predicts the log of the co-occurrence
count:

$$
w_i \cdot \tilde w_j + b_i + \tilde b_j \approx \log X_{ij}
$$

**Symbols**

| Symbol | Meaning here | Shape |
|---|---|---|
| i, j | two words in the vocabulary | |
| wᵢ | word i's vector | d numbers |
| w̃ⱼ (w-tilde) | word j's vector when it plays the context role | d numbers |
| bᵢ, b̃ⱼ | one extra adjustable number per word (an offset for very common words) | one number each |
| Xᵢⱼ | how many times words i and j appear together | a count |
| log | natural logarithm, which tames huge counts | |
| ≈ | "should be close to"; training shrinks the gap | |

**In words:** two words' vectors should have a dot product (plus offsets)
that tells you how often they meet, on a log scale.

**On an example:** if "ice" and "cold" appear together 20 times, training
nudges the vectors until w_ice · w̃_cold + b_ice + b̃_cold ≈ log 20 = 3.0. With
2-D vectors w_ice = (1, 1) and w̃_cold = (1, 0.5) and offsets 0.25 each, the
left side is 1.5 + 0.25 + 0.25 = 2.0, still 1.0 short, so training keeps
pushing the two vectors to agree more.

**In Python:**

```python
import math
# log X_ij: the target for 20 co-occurrences
round(math.log(20), 1)  # → 3.0
w_ice, w_cold_tilde = (1, 1), (1, 0.5)
b_ice, b_cold_tilde = 0.25, 0.25
# w_i · w̃_j + b_i + b̃_j
left = sum(a * b for a, b in zip(w_ice, w_cold_tilde)) + b_ice + b_cold_tilde
# the left side, and how far it still falls short
left, round(math.log(20) - left, 1)  # → (2.0, 1.0)
```

![A block pattern: male words light up under he and his, female under she and her, royals under crown and throne, children under young and school](figures/primer.ml.embeddings.word2vec.ppmi.svg)

**Reading it:** rows are the eight target words, columns are the context
words, and brighter cells mean higher PPMI (they meet more than chance). You
can read the attributes straight off the table: the royal words light up
under "crown" and "throne", the female words under "she" and "her", the
children under "young" and "school". SVD compresses exactly these block
patterns into a few directions, and `ppmi_svd_embeddings` then solves
king − man + woman = queen just like word2vec.

**Why it matters:** counting and predicting are two routes to the same
geometry. That's reassuring: embeddings aren't magic, they're compressed
co-occurrence statistics.

## The limit: one vector per word

**Everyday picture:** a phone book lists one address per name. If two
different people are called "Bank", the book can only print one address,
somewhere between the two. A static embedding table has the same problem:
"bank" gets one vector for both the riverbank and the bank account.

**Tiny example:** after training on this lesson's corpus, the single "bank"
vector has cosine about 0.52 with river words and 0.50 with money words: it
sits halfway between.

**The fix is context.** A transformer doesn't look words up once and stop.
It computes a fresh vector for every word *in every sentence* by attending
over the neighbours (`primer.ml.attention`): the new vector is a weighted
average of the sentence's vectors,

$$
\text{bank}' = \sum_{j} \alpha_j\, x_j, \qquad \sum_j \alpha_j = 1
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| j | which word of the sentence | 1 to the sentence length |
| xⱼ | the static vector of the j-th word | d numbers |
| αⱼ (alpha) | how much attention "bank" pays to word j (from softmax) | 0 to 1 |
| Σⱼ αⱼ = 1 | the attention weights are shares that add up to 1 | |
| bank′ | the new, context-aware vector for "bank" | d numbers |

**In words:** the new "bank" is a blend of the sentence's words, weighted by
how relevant each one is.

**On the example:** in "river bank fish", "bank" blends in "river" and "fish",
and its cosine with the river words climbs from 0.52 to about 0.83. The
same move by hand, in 2-D where the first axis is "river" and the second is
"money": river = (1, 0), bank = (1, 1), fish = (1, 0), and α = (0.25, 0.5, 0.25).
Then bank′ = 0.25·(1, 0) + 0.5·(1, 1) + 0.25·(1, 0) = (1, 0.5), and its cosine
with the river axis climbs from 0.71 to 0.89.

**In Python:**

```python
import math
# river, bank, fish: axis 1 is "river", axis 2 "money"
x = [(1, 0), (1, 1), (1, 0)]
# attention weights, Σ_j α_j = 1
alpha = [0.25, 0.5, 0.25]
# Σ_j α_j x_j
bank_new = [sum(a_j * x_j[i] for a_j, x_j in zip(alpha, x)) for i in range(2)]
bank_new  # → [1.0, 0.5]
def cos_with_river(v):
    # cosine with (1, 0)
    return v[0] / math.sqrt(v[0] ** 2 + v[1] ** 2)
round(cos_with_river(x[1]), 2), round(cos_with_river(bank_new), 2)  # → (0.71, 0.89)
```

```mermaid
flowchart LR
  subgraph Static["Static (word2vec, GloVe)"]
    B1["bank"] --> V1["one vector,<br/>both senses blended"]
  end
  subgraph Contextual["Contextual (transformer)"]
    R["river bank fish"] --> A1[attention mixes<br/>in neighbours] --> V2["bank ≈ riverside"]
    M["bank loan cash"] --> A2[attention mixes<br/>in neighbours] --> V3["bank ≈ finance"]
  end
```

**Reading it:** on the left a lookup table has one row for "bank", so every
sentence gets the same vector. On the right the same input goes through
attention, which blends in the surrounding words, so the output depends on
the sentence. `contextual_bank_demo` runs exactly this with the learned
vectors and one attention step.

![Static bank is about equally close to the river and money senses (0.52 vs 0.50); in context it tilts to 0.83 river or 0.85 money](figures/primer.ml.embeddings.word2vec.bank.svg)

**Reading it:** bars show cosine similarity to the river sense (probe words
water, shore, boat) and to the money sense (money, deposit, account). The
static vector is about equally similar to both. After one attention step
over "river bank fish" it tilts to the river sense, and over "bank loan
cash" it tilts to money. Same word, different vector, decided by context.

**Why it matters:** this is why BERT (2018) and every embedding model since
are built on transformers. Search, RAG and classification all need
"Python the language" and "python the snake" to be different points on the
map.

## In 20 seconds
- word2vec learns a vector per word by guessing nearby words; words that
  keep the same company get similar vectors.
- Negative sampling makes it cheap: raise the score of the true pair, lower
  the score of k random pairs, instead of scoring the whole vocabulary.
- Consistent differences in context become consistent directions, hence
  king − man + woman ≈ queen.
- GloVe and PPMI + SVD count co-occurrences and compress them, and reach the
  same geometry.
- Static vectors can't separate word senses; contextual (transformer)
  embeddings can, and they replaced static ones.

## Self-test questions

**Q: What does skip-gram predict, and where do its training labels come from?**
Each center word predicts the words within a window around it. The labels
come from the text itself (which words actually appeared nearby), so no
human labeling is needed.

**Q: Why negative sampling instead of a softmax over the vocabulary?**
A full softmax needs a score for every vocabulary word for every training
pair. Negative sampling turns it into k + 1 yes/no decisions (true context
vs. a few noise words), which is orders of magnitude cheaper and works as
well for learning vectors.

**Q: Why does king − man + woman land near queen?**
The contexts that separate king from queen (he/his vs. she/her) are the same
ones that separate man from woman, so training makes those differences the
same direction. Adding that direction to "king" moves it toward "queen".

**Q: What's the relationship between word2vec and GloVe?**
Both encode co-occurrence statistics. GloVe explicitly fits log
co-occurrence counts; skip-gram with negative sampling implicitly factorizes
a shifted PMI matrix. PPMI + SVD gives similar vectors.

**Q: What's the main limitation of static word embeddings and what fixed it?**
One vector per word regardless of meaning, so "bank" blends its senses.
Contextual embeddings from transformers compute a vector per word *in
context*, so each sense gets its own representation.

## The papers behind this lesson

- **Mikolov, Chen, Corrado and Dean, *Efficient Estimation of Word Representations in Vector Space* (2013)**, with **Mikolov et al., *Distributed Representations of Words and Phrases and their Compositionality* (2013)**: https://arxiv.org/abs/1301.3781 and https://arxiv.org/abs/1310.4546.
  Introduced skip-gram and negative sampling, and the vector-arithmetic analogies. [annotated companion](../../../papers/word2vec.html)
- **Pennington, Socher and Manning, *GloVe: Global Vectors for Word Representation* (2014)**: https://nlp.stanford.edu/projects/glove/.
  Fit word vectors directly to log co-occurrence counts, the counting route to the same geometry.
- **Levy and Goldberg, *Neural Word Embedding as Implicit Matrix Factorization* (2014)**: https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.
  Proved that skip-gram with negative sampling implicitly factorizes a shifted PMI matrix, uniting the predicting and counting families.
- **Devlin, Chang, Lee and Toutanova, *BERT* (2018)**: https://arxiv.org/abs/1810.04805.
  Made contextual embeddings mainstream: one vector per word *per sentence*, computed by a transformer.

## Further reading
- Mikolov et al., *Efficient Estimation of Word Representations in Vector Space* (2013): https://arxiv.org/abs/1301.3781
- Mikolov et al., *Distributed Representations of Words and Phrases and their Compositionality* (negative sampling, 2013): https://arxiv.org/abs/1310.4546
- Goldberg and Levy, *word2vec Explained* (the gradient derivation): https://arxiv.org/abs/1402.3722
- Levy and Goldberg, *Neural Word Embedding as Implicit Matrix Factorization* (2014): https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization
- GloVe project page (Stanford NLP): https://nlp.stanford.edu/projects/glove/
- Devlin et al., *BERT* (contextual embeddings, 2018): https://arxiv.org/abs/1810.04805
- Jay Alammar, *The Illustrated Word2vec*: https://jalammar.github.io/illustrated-word2vec/
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from primer._show import banner, say, table, takeaway
from primer.ml.attention import scaled_dot_product_attention

# ---------------------------------------------------------------------------
# 1. A corpus built so that analogies exist
# ---------------------------------------------------------------------------

# Eight target words, each a combination of three binary attributes.
# (gender, status, age)
TARGETS: dict[str, tuple[str, str, str]] = {
    "king": ("male", "royal", "adult"),
    "queen": ("female", "royal", "adult"),
    "prince": ("male", "royal", "child"),
    "princess": ("female", "royal", "child"),
    "man": ("male", "common", "adult"),
    "woman": ("female", "common", "adult"),
    "boy": ("male", "common", "child"),
    "girl": ("female", "common", "child"),
}

# Each attribute value has its own pool of context words. A target's
# sentences draw one word from each of its three pools, so its context
# distribution is the *product* of its attributes. That's the structure
# vector arithmetic can recover.
CONTEXT_POOLS: dict[str, list[str]] = {
    "male": ["he", "his", "him", "himself"],
    "female": ["she", "her", "hers", "herself"],
    "royal": ["crown", "throne", "palace", "castle"],
    "common": ["village", "farm", "market", "street"],
    "adult": ["adult", "married", "works", "old"],
    "child": ["young", "school", "plays", "toy"],
}

# Two senses of "bank", for the one-vector-per-word demo.
RIVER_WORDS = ["river", "water", "fish", "shore", "boat", "stream"]
MONEY_WORDS = ["money", "loan", "cash", "deposit", "account", "interest"]


def build_corpus(sentences_per_target: int = 150, seed: int = 0) -> list[list[str]]:
    """Generate short sentences. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    corpus: list[list[str]] = []
    for word, attrs in TARGETS.items():
        for _ in range(sentences_per_target):
            sent = [word] + [str(rng.choice(CONTEXT_POOLS[a])) for a in attrs]
            rng.shuffle(sent)
            corpus.append(sent)
    # "bank" appears equally often with each sense, so its static vector blends them.
    for pool in (RIVER_WORDS, MONEY_WORDS):
        for _ in range(sentences_per_target):
            corpus.append(["bank"] + [str(w) for w in rng.choice(pool, size=3, replace=False)])
            corpus.append([str(w) for w in rng.choice(pool, size=4, replace=False)])
    return corpus


# ---------------------------------------------------------------------------
# 2. Training data: (center, context) pairs and the noise distribution
# ---------------------------------------------------------------------------


def skipgram_pairs(sentence: list[str], window: int) -> list[tuple[str, str]]:
    """Every (center, context) pair with the context at most `window` positions away."""
    pairs = []
    for i, center in enumerate(sentence):
        for j in range(max(0, i - window), min(len(sentence), i + window + 1)):
            if j != i:
                pairs.append((center, sentence[j]))
    return pairs


def noise_distribution(counts: np.ndarray, power: float = 0.75) -> np.ndarray:
    """P(noise word) ∝ count^0.75. The power flattens the distribution so
    rare words are chosen as negatives more often than their raw frequency."""
    w = counts.astype(float) ** power
    return w / w.sum()


# ---------------------------------------------------------------------------
# 3. The SGNS loss and its hand-derived gradients
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sgns_loss(v_c: np.ndarray, u_o: np.ndarray, u_neg: np.ndarray) -> float:
    """-log σ(u_o·v_c) - Σ log σ(-u_n·v_c) for one center, one context, k noise words (rows of u_neg)."""
    return float(-np.log(_sigmoid(u_o @ v_c)) - np.sum(np.log(_sigmoid(-(u_neg @ v_c)))))


def sgns_grads(v_c: np.ndarray, u_o: np.ndarray, u_neg: np.ndarray):
    """Gradients of `sgns_loss` w.r.t. v_c, u_o and each noise vector.

    Derivation: d/dx[-log σ(x)] = σ(x) - 1, and d/dx[-log σ(-x)] = σ(x).
    Chain rule through x = u·v gives the "g times the other vector" forms.
    """
    g_o = _sigmoid(u_o @ v_c) - 1.0  # negative: pushes u_o·v_c up
    g_n = _sigmoid(u_neg @ v_c)  # positive: pushes u_n·v_c down, shape (k,)
    d_vc = g_o * u_o + g_n @ u_neg
    d_uo = g_o * v_c
    d_un = g_n[:, None] * v_c[None, :]
    return d_vc, d_uo, d_un


def sgns_update(v_c: np.ndarray, u_o: np.ndarray, u_neg: np.ndarray, lr: float):
    """One gradient-descent step on all three kinds of vector. Returns new copies.

    Every vector moves a small step (`lr`) *against* its gradient, the
    direction that lowers the loss.
    """
    d_vc, d_uo, d_un = sgns_grads(v_c, u_o, u_neg)
    return v_c - lr * d_vc, u_o - lr * d_uo, u_neg - lr * d_un


def gradient_check(seed: int = 0, eps: float = 1e-6) -> float:
    """Max |analytic - numerical| gradient w.r.t. v_c, via central differences."""
    rng = np.random.default_rng(seed)
    v, u, n = rng.standard_normal(5), rng.standard_normal(5), rng.standard_normal((3, 5))
    analytic = sgns_grads(v, u, n)[0]
    numeric = np.zeros_like(v)
    for i in range(len(v)):
        e = np.zeros_like(v)
        e[i] = eps
        numeric[i] = (sgns_loss(v + e, u, n) - sgns_loss(v - e, u, n)) / (2 * eps)
    return float(np.max(np.abs(analytic - numeric)))


# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------


@dataclass
class WordVectors:
    """A vocabulary and one L2-normalized vector per word."""

    vocab: list[str]
    vectors: np.ndarray  # (V, d), rows unit length

    def __post_init__(self):
        self.index = {w: i for i, w in enumerate(self.vocab)}

    def __getitem__(self, word: str) -> np.ndarray:
        return self.vectors[self.index[word]]


def train_sgns(
    corpus: list[list[str]],
    dim: int = 16,
    window: int = 4,
    negatives: int = 5,
    epochs: int = 8,
    lr: float = 0.05,
    batch: int = 256,
    seed: int = 0,
) -> WordVectors:
    """Skip-gram with negative sampling, mini-batched with NumPy.

    Returns the center vectors (W_in), each row L2-normalized. Why not add
    the context vectors too, as some implementations do? W_in + W_out mixes
    in "appears next to" (king ~ crown), while W_in alone captures "is used
    in similar contexts" (king ~ queen), which is the similarity we want.
    """
    rng = np.random.default_rng(seed)
    counts = Counter(w for s in corpus for w in s)
    vocab = sorted(counts)
    idx = {w: i for i, w in enumerate(vocab)}
    noise = noise_distribution(np.array([counts[w] for w in vocab]))

    pairs = np.array([(idx[c], idx[o]) for s in corpus for c, o in skipgram_pairs(s, window)])
    V = len(vocab)
    # Small random init for center vectors, zeros for context vectors (as in the original C code).
    W_in = rng.uniform(-0.5, 0.5, (V, dim)) / dim
    W_out = np.zeros((V, dim))

    for _ in range(epochs):
        rng.shuffle(pairs)
        for start in range(0, len(pairs), batch):
            b = pairs[start : start + batch]
            c, o = b[:, 0], b[:, 1]
            n = rng.choice(V, size=(len(b), negatives), p=noise)  # (B, k) noise word ids

            v_c, u_o, u_n = W_in[c], W_out[o], W_out[n]  # (B,d), (B,d), (B,k,d)
            g_o = _sigmoid(np.sum(u_o * v_c, axis=1)) - 1.0  # (B,)
            g_n = _sigmoid(np.einsum("bkd,bd->bk", u_n, v_c))  # (B,k)

            d_vc = g_o[:, None] * u_o + np.einsum("bk,bkd->bd", g_n, u_n)
            d_uo = g_o[:, None] * v_c
            d_un = g_n[:, :, None] * v_c[:, None, :]

            # np.add.at accumulates correctly when the same word appears twice in a batch
            # (plain fancy-index assignment would keep only the last update).
            np.add.at(W_in, c, -lr * d_vc)
            np.add.at(W_out, o, -lr * d_uo)
            np.add.at(W_out, n.ravel(), -lr * d_un.reshape(-1, dim))

    return WordVectors(vocab, W_in / np.linalg.norm(W_in, axis=1, keepdims=True))


# ---------------------------------------------------------------------------
# 5. Count-based alternative: PPMI + SVD
# ---------------------------------------------------------------------------


def cooccurrence(corpus: list[list[str]], window: int = 4) -> tuple[list[str], np.ndarray]:
    counts = Counter(w for s in corpus for w in s)
    vocab = sorted(counts)
    idx = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(vocab), len(vocab)))
    for s in corpus:
        for c, o in skipgram_pairs(s, window):
            X[idx[c], idx[o]] += 1
    return vocab, X


def ppmi(X: np.ndarray) -> np.ndarray:
    """Positive pointwise mutual information of a co-occurrence count matrix.

    PMI(w,c) = log P(w,c) / (P(w) P(c)). Pairs never seen together would be
    log 0 = -inf; PPMI clips all negatives to 0.
    """
    total = X.sum()
    p_wc = X / total
    p_w = X.sum(axis=1, keepdims=True) / total
    p_c = X.sum(axis=0, keepdims=True) / total
    with np.errstate(divide="ignore"):
        pmi = np.log(p_wc / (p_w * p_c))
    return np.maximum(pmi, 0.0)


def ppmi_svd_embeddings(corpus: list[list[str]], dim: int = 16) -> WordVectors:
    """Count, keep positive PMI, truncate the SVD. Word vectors = U·sqrt(S)."""
    vocab, X = cooccurrence(corpus)
    U, S, _ = np.linalg.svd(ppmi(X))
    E = U[:, :dim] * np.sqrt(S[:dim])
    return WordVectors(vocab, E / np.linalg.norm(E, axis=1, keepdims=True))


# ---------------------------------------------------------------------------
# 6. Using the vectors: neighbours, analogies, and the "bank" problem
# ---------------------------------------------------------------------------


def nearest(wv: WordVectors, word: str, k: int = 5, exclude: set[str] | None = None) -> list[tuple[str, float]]:
    """The k words with the highest cosine to `word` (vectors are unit length, so dot = cosine)."""
    return nearest_to_vector(wv, wv[word], k, (exclude or set()) | {word})


def nearest_to_vector(wv: WordVectors, v: np.ndarray, k: int, exclude: set[str]) -> list[tuple[str, float]]:
    sims = wv.vectors @ (v / np.linalg.norm(v))
    order = [i for i in np.argsort(-sims) if wv.vocab[i] not in exclude]
    return [(wv.vocab[i], float(sims[i])) for i in order[:k]]


def analogy(wv: WordVectors, a: str, b: str, c: str) -> str:
    """a is to b as ? is to c: returns the word nearest a - b + c, excluding a, b, c.

    king - man + woman: start at king, remove "male-ness", add "female-ness".
    """
    return nearest_to_vector(wv, wv[a] - wv[b] + wv[c], 1, {a, b, c})[0][0]


def contextual_bank_demo(wv: WordVectors) -> dict[str, float]:
    """One attention step turns the single static "bank" vector into two contextual ones.

    We use the learned vectors as queries, keys *and* values (identity
    projections) and attend over each short sentence. The output row for
    "bank" is a softmax-weighted blend of the sentence's vectors.
    Probes are sense words *not* in either sentence, so the comparison is fair.
    """
    river_probe = np.mean([wv[w] for w in ("water", "shore", "boat")], axis=0)
    money_probe = np.mean([wv[w] for w in ("money", "deposit", "account")], axis=0)

    def cos(x, y):
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))

    def bank_in(sentence):
        X = np.stack([wv[w] for w in sentence])
        out, _ = scaled_dot_product_attention(X, X, X)
        return out[sentence.index("bank")]

    river_bank = bank_in(["river", "bank", "fish"])
    money_bank = bank_in(["bank", "loan", "cash"])
    static = wv["bank"]
    return {
        "static_to_river": cos(static, river_probe),
        "static_to_money": cos(static, money_probe),
        "river_bank_to_river": cos(river_bank, river_probe),
        "river_bank_to_money": cos(river_bank, money_probe),
        "money_bank_to_river": cos(money_bank, river_probe),
        "money_bank_to_money": cos(money_bank, money_probe),
    }


# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------


def _pca_2d(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def figures() -> dict:
    """Plots computed from this module's own functions. Keys match the docstring's image names."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    corpus = build_corpus()
    wv = train_sgns(corpus)
    figs = {}

    # space: PCA of the eight target words, with male->female arrows
    words = list(TARGETS)
    P = _pca_2d(np.stack([wv[w] for w in words]))
    pos = dict(zip(words, P))
    fig, ax = plt.subplots(figsize=(6, 5))
    for m, f in (("king", "queen"), ("man", "woman"), ("prince", "princess"), ("boy", "girl")):
        ax.annotate("", xy=pos[f], xytext=pos[m], arrowprops=dict(arrowstyle="->", color="C3", lw=1.5))
    for w, (x, y) in pos.items():
        ax.scatter(x, y, color="C0")
        ax.annotate(w, (x, y), textcoords="offset points", xytext=(5, 5))
    ax.set(xlabel="principal component 1", ylabel="principal component 2", title="Learned word vectors: male→female arrows are parallel")
    figs["space"] = fig

    # ppmi: targets x context words heatmap
    vocab, X = cooccurrence(corpus)
    M = ppmi(X)
    ctx = [w for pool in ("male", "female", "royal", "common", "adult", "child") for w in CONTEXT_POOLS[pool]]
    sub = M[np.ix_([vocab.index(w) for w in words], [vocab.index(c) for c in ctx])]
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(sub, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(ctx)), ctx, rotation=60)
    ax.set_yticks(range(len(words)), words)
    ax.set(title="PPMI: target words (rows) vs. context words (columns)")
    fig.colorbar(im, ax=ax, label="PPMI")
    figs["ppmi"] = fig

    # bank: static vs contextual similarity to each sense
    d = contextual_bank_demo(wv)
    labels = ["static 'bank'", "'bank' in\nriver bank fish", "'bank' in\nbank loan cash"]
    river = [d["static_to_river"], d["river_bank_to_river"], d["money_bank_to_river"]]
    money = [d["static_to_money"], d["river_bank_to_money"], d["money_bank_to_money"]]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - 0.2, river, 0.4, label="similarity to river sense")
    ax.bar(x + 0.2, money, 0.4, label="similarity to money sense")
    ax.set_xticks(x, labels)
    ax.set(ylabel="cosine similarity", title="One attention step separates the senses of 'bank'")
    ax.legend()
    figs["bank"] = fig

    for f in figs.values():
        f.tight_layout()
    return figs


# ---------------------------------------------------------------------------
# 8. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. Training pairs come from the text itself")
    s = ["he", "king", "crown", "old"]
    say(f"Sentence {s} with window 1 yields pairs:")
    print("  ", skipgram_pairs(s, window=1), "\n")
    say("Noise words are drawn ∝ count^0.75. Counts 1 and 16 become weights 1 and 8:")
    table(["count", "raw share", "noise share (^0.75)"], [(1, 1 / 17, 1 / 9), (16, 16 / 17, 8 / 9)], floatfmt=".3f")

    banner("2. The loss, and a check that the hand-written gradient is right")
    say(
        f"""
        Orthogonal vectors score σ(0) = 0.5 for every pair, so one true pair
        plus two noise pairs cost 3·ln 2 = {3 * np.log(2):.3f}. Max difference
        between the hand-derived and numerical gradients: {gradient_check():.1e}.
        """
    )

    banner("3. Train skip-gram with negative sampling")
    corpus = build_corpus()
    wv = train_sgns(corpus)
    say(f"{len(corpus)} sentences, vocabulary of {len(wv.vocab)} words, 16-dimensional vectors.")
    table(["word", "nearest neighbours (cosine)"], [(w, ", ".join(f"{n} {s:.2f}" for n, s in nearest(wv, w, 3))) for w in ("king", "girl", "river")])

    banner("4. Vector arithmetic")
    rows = [(f"{a} - {b} + {c}", analogy(wv, a, b, c)) for a, b, c in [("king", "man", "woman"), ("boy", "man", "woman"), ("prince", "boy", "girl"), ("queen", "woman", "girl")]]
    table(["expression", "nearest word"], rows)
    takeaway("Consistent differences in context become consistent directions in vector space.")

    banner("5. Counting instead of predicting: PPMI + SVD")
    cb = ppmi_svd_embeddings(corpus)
    say(f"king - man + woman with PPMI+SVD vectors: {analogy(cb, 'king', 'man', 'woman')}.")

    banner("6. One vector per word, and how context fixes it")
    d = contextual_bank_demo(wv)
    table(
        ["vector for 'bank'", "cos to river sense", "cos to money sense"],
        [
            ("static (word2vec)", d["static_to_river"], d["static_to_money"]),
            ("in 'river bank fish'", d["river_bank_to_river"], d["river_bank_to_money"]),
            ("in 'bank loan cash'", d["money_bank_to_river"], d["money_bank_to_money"]),
        ],
        floatfmt=".3f",
    )
    takeaway(
        "A static table has one row for 'bank'. Attention blends in the neighbours, "
        "so the same word gets a different vector in each sentence: that's a contextual embedding."
    )


if __name__ == "__main__":
    demo()
