r"""
# Similarity: how close are two meanings?

Run: `python -m primer.ml.embeddings.similarity`

New to the notation (vectors, Σ, ‖x‖)? Start with `primer.notation`, which
builds every symbol used here from zero.

## The everyday picture

An **embedding** gives every piece of text a set of coordinates on a map of
meaning. Texts about similar things live on the same street; unrelated ones
are across town. Draw an arrow from the centre of the map to each text's
spot, and "how similar are these two texts?" becomes "how alike are these two
arrows?".

There are three natural ways to compare two arrows:

- **Cosine similarity** compares the *direction* they point and ignores how
  long they are, like two people pointing at the same mountain, one with a
  longer arm.
- **Euclidean distance** is the straight-line distance between the two arrow
  tips, like measuring with a ruler.
- **Dot product** mixes both: pointing the same way *and* being long both
  make it bigger.

## A tiny worked example

Three texts, each placed in a 3-dimensional space (real models use 384 to
3,072 **dimensions**, meaning numbers per vector; the arithmetic is the same):

a = (1, 2, 2), b = (2, 1, 2), c = (2, −1, −2)

1. **Dot product**: multiply position by position, then add.
   a·b = 1·2 + 2·1 + 2·2 = **8**. a·c = 1·2 + 2·(−1) + 2·(−2) = **−4**.
2. **Length** (the **norm**, written ‖a‖): square each number, add, and take
   the square root. ‖a‖ = √(1 + 4 + 4) = **3**, and b and c also have length 3.
3. **Cosine**: dot product divided by both lengths.
   cos(a, b) = 8 / (3·3) = **0.89**. cos(a, c) = −4 / 9 = **−0.44**.
4. **Euclidean distance**: subtract, square, add, square-root.
   ‖a − b‖ = √(1 + 1 + 0) = **1.41**. ‖a − c‖ = √(1 + 9 + 16) = **5.10**.

| Pair | Dot product | Lengths multiplied | Cosine | Distance | Reading |
|---|---|---|---|---|---|
| a, b | 8 | 9 | 0.89 | 1.41 | Very similar |
| a, c | −4 | 9 | −0.44 | 5.10 | Pointing away |

Cosine runs from −1 (opposite directions) through 0 (unrelated, at right
angles) to 1 (same direction).

```mermaid
flowchart LR
  A["a = (1, 2, 2)"] --> D["dot product<br/>a·b = 8"]
  B["b = (2, 1, 2)"] --> D
  A --> NA["length ‖a‖ = 3"]
  B --> NB["length ‖b‖ = 3"]
  D --> COS["cosine = 8 / (3 × 3) = 0.89"]
  NA --> COS
  NB --> COS
  A --> SUB["difference a − b = (−1, 1, 0)"]
  B --> SUB
  SUB --> L2["distance = √2 = 1.41"]
```

**Reading it:** the two input vectors feed three calculations. The top path
is the dot product. The middle paths measure each vector's length, and
cosine is simply the dot product divided by those lengths, which is how it
comes to "ignore length". The bottom path subtracts the vectors and measures
what's left: that's distance. Everything later in this lesson is these three
boxes.

**In code:** `worked_example` computes every number in the table above from
a, b and c, so you can check your hand arithmetic against it.

## The math

$$
\cos(a, b) = \frac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert}
\qquad
a \cdot b = \sum_{i=1}^{d} a_i b_i
\qquad
\lVert a - b \rVert = \sqrt{\sum_{i=1}^{d} (a_i - b_i)^2}
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| a, b | the two embedding vectors being compared | d numbers each |
| d | number of dimensions | 3 in the example; 384 to 3,072 in real models |
| i | which position (dimension) we're looking at | 1 to d |
| aᵢ | the i-th number in a | a real number |
| Σ | "add up", over i from 1 to d | |
| a · b | dot product: multiply matching positions and add | one number, any sign |
| ‖a‖ | length (norm) of a: √(Σ aᵢ²) | one number ≥ 0 |
| √ | square root | |
| cos(a, b) | cosine of the angle between a and b | −1 to 1 |
| ‖a − b‖ | Euclidean distance between the tips | ≥ 0; 0 means identical |

**In words:** the cosine of a and b is their dot product divided by the
product of their lengths; the dot product adds up the products of matching
positions; the distance is the square root of the summed squared
differences.

**On the example:** cos(a, b) = (1·2 + 2·1 + 2·2) / (3 · 3) = 8 / 9 = 0.89;
‖a − b‖ = √((1−2)² + (2−1)² + (2−2)²) = √2 = 1.41.

**In Python:**

```python
import math
a, b = [1, 2, 2], [2, 1, 2]
# a · b = Σ a_i b_i
dot = sum(a_i * b_i for a_i, b_i in zip(a, b))
dot  # → 8
# ‖a‖
norm_a = math.sqrt(sum(a_i ** 2 for a_i in a))
# ‖b‖
norm_b = math.sqrt(sum(b_i ** 2 for b_i in b))
norm_a, norm_b  # → (3.0, 3.0)
# cos(a, b)
round(dot / (norm_a * norm_b), 2)  # → 0.89
# ‖a - b‖
round(math.sqrt(sum((a_i - b_i) ** 2 for a_i, b_i in zip(a, b))), 2)  # → 1.41
```

The code is `cosine`, `dot` and `euclidean`, each a line of NumPy.

**In code:** `rank` sorts a whole collection of documents from best to worst
match under whichever of the three metrics you name.

**Why it matters:** every vector database, semantic search box and RAG
system does exactly this comparison, millions of times per second. Knowing
which of the three a system uses, and why, is the difference between
correct and silently wrong rankings.

## Normalize, and the three agree

**Everyday picture:** trim every arrow to the same length, one unit. Now the
only thing that can differ is direction, so "tips are close together" and
"arrows point the same way" become the same statement.

**Tiny example:** a = (1, 0) and b = (0.6, 0.8) both have length 1 (check:
0.6² + 0.8² = 0.36 + 0.64 = 1). Their cosine is 1·0.6 + 0·0.8 = 0.6. The squared
distance is (1 − 0.6)² + (0 − 0.8)² = 0.16 + 0.64 = 0.8, and 2 − 2·0.6 = 0.8.
Same number.

$$
\lVert a - b \rVert^2 = \lVert a \rVert^2 + \lVert b \rVert^2 - 2\,a\cdot b = 2 - 2\cos(a, b)
\quad \text{when } \lVert a \rVert = \lVert b \rVert = 1
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| ‖a − b‖² | squared distance between the tips | 0 to 4 for unit vectors |
| ‖a‖², ‖b‖² | squared lengths, both 1 after normalizing | 1 |
| a · b | dot product, equal to cosine for unit vectors | −1 to 1 |
| cos(a, b) | cosine similarity | −1 to 1 |

**In words:** for unit-length vectors, the squared distance is two minus
twice the cosine, so the higher the cosine, the smaller the distance, always.

**On the example:** ‖a − b‖² = 1 + 1 − 2·0.6 = 0.8 = 2 − 2·0.6.

**In Python:**

```python
# both length 1
a, b = [1, 0], [0.6, 0.8]
# for unit vectors, a · b is the cosine
cos = sum(a_i * b_i for a_i, b_i in zip(a, b))
# ‖a - b‖²
dist_sq = sum((a_i - b_i) ** 2 for a_i, b_i in zip(a, b))
# the two sides of the identity
round(dist_sq, 2), round(2 - 2 * cos, 2)  # → (0.8, 0.8)
```

**L2-normalizing** (dividing a vector by its own length) is what trims every
arrow to length 1. After it, the dot product *is* the cosine, and distance is
a decreasing function of the cosine. So **once vectors are L2-normalized,
cosine, dot product and Euclidean distance give the same ranking.**

**Try it:** pick "Same direction, different lengths", then drag B's length.
The dot product and the distance change, but the cosine stays at 1, because
it only sees the angle. Drag an angle instead and watch the cosine slide from
1 through 0 to −1. Then switch on "Normalize to length 1": both arrows shrink
to the unit circle, the dot product becomes the cosine, and the squared
distance becomes 2 − 2 × cosine.

<div class="viz" data-viz="cosine-similarity" aria-label="Cosine similarity of two arrows"></div>

![All 400 random unit-vector pairs fall exactly on the line squared distance = 2 - 2 x cosine, so higher cosine always means smaller distance](figures/primer.ml.embeddings.similarity.identity.svg)

**Reading it:** each dot is one pair of random unit vectors. The horizontal
axis is their cosine and the vertical axis is their squared Euclidean
distance. Every dot sits exactly on the line y = 2 − 2x. Because the line only
goes down, "higher cosine" and "smaller distance" always mean the same thing,
so the two metrics can never disagree about which neighbour is nearer.

```mermaid
flowchart TD
  S{What was the model<br/>trained with?} -->|cosine| N[L2-normalize every vector<br/>once, at write time]
  S -->|dot product| D{Does the vector length<br/>carry signal?}
  D -->|yes, e.g. popularity| RAW[Keep raw vectors,<br/>rank by dot product]
  D -->|no| N
  N --> FAST[Rank by dot product:<br/>same order as cosine and L2,<br/>and the cheapest to compute]
```

**Reading it:** start at the top with the question that matters, which is
how the model was trained, not which metric sounds best. Most text
embedding models are trained with cosine, so you normalize once when you
write vectors, then use the plain dot product at query time. It gives the
same order as cosine and costs less. Only when length is deliberate signal
do you skip normalizing.

**In code:** `l2_normalize` trims every row to length 1;
`rankings_agree_after_normalizing` ranks random documents by all three metrics
before and after normalizing and reports which rankings match, and
`l2_identity_check` returns both sides of the identity for two random vectors.

**Why it matters:** vector databases typically normalize on insert and
then use the dot product, the cheapest of the three. If you mix normalized
and unnormalized vectors, rankings quietly change. **Use the metric the
embedding model was trained with.**

## When length carries signal

**Everyday picture:** on the map of meaning, some shops are big and popular.
A recommender may deliberately make popular items' arrows longer, so they
win ties. Cosine throws that away; the dot product keeps it.

**Tiny example** (`popularity_in_the_norm`): the query points along
(1, 0). A niche item (0.99, 0.10) is almost perfectly aligned. A popular item
(0.95, 0.30) × 3 = (2.85, 0.90) is slightly off but three times longer.

| Item | Cosine with query | Dot with query |
|---|---|---|
| niche (0.99, 0.10) | 0.99 / 0.995 = 0.995 | 0.99 |
| popular (2.85, 0.90) | 2.85 / 2.99 = 0.954 | 2.85 |

Cosine prefers the niche item; the dot product prefers the popular one.

![The short item points almost along the query and wins on cosine (0.99 vs 0.95); the item three times longer wins on dot product (2.85 vs 0.99)](figures/primer.ml.embeddings.similarity.arrows.svg)

**Reading it:** the black arrow is the query. The blue arrow (niche item)
points almost exactly the same way but is short. The orange arrow (popular
item) leans a little away but is three times longer. Cosine only looks at
the angle between each arrow and the query, so blue wins. The dot product
also rewards length, so orange wins. Neither is wrong: it depends on what
the model was trained to encode.

**Why it matters:** some retrieval and recommender models are trained with
the raw dot product and put useful signal in the length. Normalizing their
vectors "to be safe" throws that signal away.

## The curse of dimensionality

**Everyday picture:** in a small village some neighbours live next door and
some live across the valley. Now imagine a city with thousands of streets
running in thousands of directions: pick random strangers and they all turn
out to live about equally far from you. "Nearest" stops meaning much.

**Tiny example:** scatter 1,000 random points in a cube and measure from a
random spot. In 2 dimensions the nearest point is about 1/100th as far away
as the farthest. In 1,000 dimensions it's about 90% as far: everyone is
nearly equidistant.

$$
\text{ratio} = \frac{\min_j \lVert q - x_j \rVert}{\max_j \lVert q - x_j \rVert}
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| q | the query point | d numbers |
| xⱼ | the j-th of the random points | d numbers each |
| j | which random point | 1 to 1,000 |
| min, max over j | the smallest / largest value across all the points | |
| ratio | nearest distance divided by farthest distance | 0 to 1; near 1 means "all equally far" |

**In words:** the ratio is the distance to the closest point divided by the
distance to the farthest point.

**On the example:** by hand first: a query at (0, 0) and three points at
(1, 0), (0, 2) and (3, 4) are 1, 2 and 5 away, so the ratio is 1 / 5 = 0.2.
With 1,000 random points the demo table shows about 0.01 at d = 2 and about
0.9 at d = 1,000. The Python below draws its own random points, so it lands
close by rather than on the same digits: 0.02 and 0.9.

**In Python:**

```python
import math, random
def ratio(q, xs):
    # ‖q - x_j‖ for every j
    dists = [math.dist(q, x_j) for x_j in xs]
    # min over j ÷ max over j
    return min(dists) / max(dists)
ratio((0, 0), [(1, 0), (0, 2), (3, 4)])  # → 0.2
random.seed(0)
for d in (2, 1000):
    q = [random.random() for _ in range(d)]
    xs = [[random.random() for _ in range(d)] for _ in range(1000)]
    # one significant figure
    print(d, f"{ratio(q, xs):.1g}")  # → 2 0.02 1000 0.9
```

![The nearest-to-farthest distance ratio rises from about 0 in 2 dimensions to 0.8 at 200 and 0.92 at 2,000: random points become almost equidistant](figures/primer.ml.embeddings.similarity.curse.svg)

**Reading it:** the horizontal axis is the number of dimensions (log scale);
the vertical axis is the nearest-to-farthest ratio above. In 2-D the ratio is
near 0: the nearest point really is much nearer. By a few hundred dimensions
the curve flattens toward 1, so every random point is roughly equally far
away, and "nearest" tells you almost nothing about random data.

**In code:** `curse_of_dimensionality` scatters the random points, measures
from a random query, and returns the nearest, farthest and ratio for each
dimension.

**Why it matters:** exact nearest-neighbour search in high dimensions gets
slow and fragile, which is one reason vector search uses approximate indexes
(`primer.ml.embeddings.ann`). Real embeddings aren't random: meaning puts
them on a much lower-dimensional structure, so nearest-neighbour search still
works well in practice.

## Anisotropy: why every score looks like 0.8

**Everyday picture:** a stadium crowd all facing the stage. Ask "are these
two people facing the same way?" and the answer is "roughly yes" for
everyone, so the question tells you little. Many embedding models behave
like that crowd: their vectors bunch in a narrow cone, so even unrelated
texts get cosines of 0.7 or more. This lopsidedness is called
**anisotropy** ("not the same in every direction").

**Tiny example:** build each vector as √3·m + n, where m is one direction
everybody shares (the stage) and n is a unit-length direction of each
text's own. For two unrelated texts the n parts are at right angles, so
their dot product is (√3)² = 3 and each length is √(3 + 1) = 2, and the
cosine is 3 / (2 · 2) = **0.75**. For paraphrases, whose own parts agree with
cosine ρ, the cosine becomes 0.75 + 0.25·ρ: about 0.79 to 0.86. All the signal
is squeezed into a few hundredths.

$$
v = \alpha\, m + \beta\, n, \qquad
\cos(v_1, v_2) = \frac{\alpha^2 + \beta^2 \rho}{\alpha^2 + \beta^2}
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| v, v₁, v₂ | an embedding vector | d numbers (768 here) |
| m | the shared direction every vector leans toward | unit vector |
| n | the text's own direction | unit vector |
| α (alpha) | how strongly every vector leans toward m | √3 here |
| β (beta) | how strongly each keeps its own direction | 1 here |
| ρ (rho) | cosine between the two texts' own directions | 0 unrelated; 0.15 to 0.45 for paraphrases |

**In words:** every vector is a big shared part plus a small personal part,
so the cosine is mostly the shared part's weight, nudged up a little by how
much the personal parts agree.

**On the example:** unrelated, ρ = 0: (3 + 0) / (3 + 1) = 0.75. A paraphrase with
ρ = 0.4: (3 + 0.4) / 4 = 0.85.

**In Python:**

```python
import math
alpha, beta = math.sqrt(3), 1
# shared direction; two unrelated own directions
m, n1, n2 = (1, 0, 0), (0, 1, 0), (0, 0, 1)
# v = α m + β n
v1 = [alpha * m_i + beta * n_i for m_i, n_i in zip(m, n1)]
v2 = [alpha * m_i + beta * n_i for m_i, n_i in zip(m, n2)]
dot = sum(a * b for a, b in zip(v1, v2))
# the cosine, measured directly
round(dot / (math.hypot(*v1) * math.hypot(*v2)), 2)  # → 0.75
def cos_formula(rho):
    return (alpha ** 2 + beta ** 2 * rho) / (alpha ** 2 + beta ** 2)
# unrelated, and a paraphrase
round(cos_formula(0), 2), round(cos_formula(0.4), 2)  # → (0.75, 0.85)
```

![Raw cosines crowd paraphrases (0.76 to 0.87) and unrelated pairs (0.72 to 0.78) together; mean-centred, unrelated sit near 0 and paraphrases near 0.3](figures/primer.ml.embeddings.similarity.anisotropy.svg)

**Reading it:** the left panel shows raw cosine scores for 300 paraphrase
pairs (one colour) and 300 unrelated pairs (the other). Both piles sit
between 0.7 and 0.9, so a score of "0.8" alone says little. The right panel
shows the same pairs after **mean-centering**, which means subtracting the
average vector of the whole collection (that removes the shared "stage"
direction). Unrelated pairs drop to about 0, paraphrases sit clearly above,
and the gap is easy to see. The information was there all along, squeezed
into a narrow band.

**In code:** `anisotropic_embeddings` builds the pairs as α·m + β·n,
`pair_cosines` scores each pair, and `mean_center` subtracts the average
vector to remove the shared direction.

### Picking a threshold from data, not folklore

Sometimes you need a yes/no cut-off: "are these duplicates?", "is this
cached answer close enough?". Pick it by measuring. Label some pairs,
score them, and choose the threshold with the best **F1**, the balance of
**precision** (of the pairs I called similar, how many were?) and **recall**
(of the truly similar pairs, how many did I catch?):

$$
F_1 = \frac{2PR}{P + R}
$$

**Symbols**

| Symbol | Meaning here | Range |
|---|---|---|
| P | precision: true matches ÷ everything called a match | 0 to 1 |
| R | recall: true matches found ÷ all true matches | 0 to 1 |
| F₁ | their harmonic mean; high only when both are high | 0 to 1 |

**In words:** F1 is twice the product of precision and recall, divided by
their sum.

**On a hand example:** scores 0.9, 0.8, 0.7, 0.6 where the first two are true
matches. Threshold 0.8 calls exactly those two matches: P = 2/2 = 1,
R = 2/2 = 1, F₁ = 2·1·1 / 2 = 1. `best_threshold` finds 0.8.

**In Python:**

```python
scores = [0.9, 0.8, 0.7, 0.6]
# the labels
is_match = [True, True, False, False]
# threshold 0.8
called = [s >= 0.8 for s in scores]
true_matches = sum(c and y for c, y in zip(called, is_match))
# precision
P = true_matches / sum(called)
# recall
R = true_matches / sum(is_match)
# F₁
2 * P * R / (P + R)  # → 1.0
```

```mermaid
flowchart LR
  L[Sample a few hundred pairs<br/>from your own data] --> H[Label each:<br/>same meaning or not]
  H --> E[Score every pair<br/>with THIS model]
  E --> W[Sweep thresholds,<br/>compute precision, recall, F1]
  W --> P[Pick the threshold that fits<br/>your cost of each error]
  P --> V[Store it with the<br/>model version]
  V -->|model upgraded| E
```

**Reading it:** a threshold is a property of one model on one kind of data,
so it is *measured*, not guessed. Label pairs, score them, sweep, and pick.
The loop back from "model upgraded" is the step people forget: a new model
has a different score distribution, so the old threshold is meaningless.

![F1 on raw scores swings within a few hundredths: the calibrated threshold 0.776 reaches F1 0.99, while the guessed 0.80 drops to 0.87](figures/primer.ml.embeddings.similarity.threshold.svg)

**Reading it:** the curve is F1 at each candidate threshold on the raw
anisotropic scores. The dashed line is the "obvious" guess of 0.8; the
solid line is the calibrated threshold. F1 changes sharply over a few
hundredths of cosine, which is exactly why a threshold carried over from a
different model (or a blog post) can be badly wrong.

**In code:** `f1_curve` computes F1 at every candidate threshold for this
plot; `best_threshold` returns the single best one with its precision and
recall.

**Why it matters:**

* Absolute scores mean different things for different models. "0.8" is
  not a universal "similar".
* **Rank, don't threshold, whenever you can.** A shared offset doesn't change
  which document is on top.
* When you *must* threshold (dedup, semantic caching, "no good answer
  found"), calibrate on labeled pairs for that model, and redo it when the
  model changes.

## In 20 seconds
- Cosine compares direction, the dot product compares direction and length,
  Euclidean distance measures the gap between the tips.
- On L2-normalized vectors all three give the same ranking, because
  ‖a − b‖² = 2 − 2cos.
- Use the metric the model was trained with.
- Scores aren't comparable across models (anisotropy). Rank when you can;
  calibrate thresholds on labeled pairs when you can't.

## Self-test questions

**Q: When do cosine, dot product and Euclidean distance give the same ranking?**
When all vectors are L2-normalized: the dot product equals the cosine, and
the squared distance equals 2 − 2cos, which always moves opposite to cosine.

**Q: When would you deliberately use the raw dot product on unnormalized vectors?**
When the model was trained that way and encodes useful signal in the length
(e.g. popularity or quality in some recommender and retrieval models).
Normalizing would throw that signal away.

**Q: Similarity scores all sit between 0.75 and 0.85. What's going on and how do you set a threshold?**
The model is anisotropic: its vectors share a dominant direction, so every
pair looks similar and the real signal lives in a narrow band. Ranking still
works. For a threshold, label a few hundred pairs from your own data, sweep
the threshold, and pick the one that maximizes F1 (or matches your
precision/recall needs). Optionally mean-center the vectors. Redo it for
every model version.

**Q: What is the curse of dimensionality for nearest-neighbour search?**
For random high-dimensional data, the nearest and farthest points are almost
the same distance away, so "nearest" carries little information and exact
search gets expensive. Real embeddings have structure, and approximate
indexes trade a little recall for large speedups.

## The papers behind this lesson

- **Beyer, Goldstein, Ramakrishnan and Shaft, *When Is "Nearest Neighbor" Meaningful?* (1999)**: https://doi.org/10.1007/3-540-49257-7_15.
  Proved that for broad families of random data, the nearest and farthest neighbours become equally far as dimension grows: the curse of dimensionality measured in this lesson.
- **Ethayarajh, *How Contextual are Contextualized Word Representations?* (2019)**: https://arxiv.org/abs/1909.00512.
  Showed that transformer embeddings occupy a narrow cone (anisotropy), which is why raw cosine scores bunch together.
- **Su, Cao, Liu and Ou, *Whitening Sentence Representations for Better Semantics and Faster Retrieval* (2021)**: https://arxiv.org/abs/2103.15316.
  Showed that centering and whitening sentence embeddings spreads them back out and improves similarity quality.

## Further reading
- Cosine similarity (Wikipedia): https://en.wikipedia.org/wiki/Cosine_similarity
- Beyer et al., *When Is "Nearest Neighbor" Meaningful?* (1999): https://doi.org/10.1007/3-540-49257-7_15
- Ethayarajh, *How Contextual are Contextualized Word Representations?* (anisotropy, 2019): https://arxiv.org/abs/1909.00512
- Su et al., *Whitening Sentence Representations* (2021): https://arxiv.org/abs/2103.15316
- sentence-transformers, semantic search and score functions: https://www.sbert.net/examples/applications/semantic-search/README.html
"""

from __future__ import annotations

import numpy as np

from primer._show import banner, say, table, takeaway

# ---------------------------------------------------------------------------
# 1. The three measures
# ---------------------------------------------------------------------------


def l2_normalize(X: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Scale each vector (row, by default) to unit length.

    `eps` avoids dividing by zero for an all-zero vector (e.g. empty text).
    """
    return X / np.maximum(np.linalg.norm(X, axis=axis, keepdims=True), eps)


def dot(a: np.ndarray, b: np.ndarray) -> float:
    """Sum of element-wise products. Sensitive to both angle and length."""
    return float(np.dot(a, b))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """cos(angle) = a·b / (|a||b|). In [-1, 1]; ignores length."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Straight-line distance. Smaller means closer."""
    return float(np.linalg.norm(a - b))


def rank(query: np.ndarray, docs: np.ndarray, metric: str) -> np.ndarray:
    """Doc indices from best to worst match under `metric` (cosine | dot | l2)."""
    if metric == "dot":
        scores = docs @ query
    elif metric == "cosine":
        scores = (docs @ query) / (np.linalg.norm(docs, axis=1) * np.linalg.norm(query))
    elif metric == "l2":
        scores = -np.linalg.norm(docs - query, axis=1)  # negate: closer = higher
    else:
        raise ValueError(metric)
    # Stable sort so ties break the same way for every metric.
    return np.argsort(-scores, kind="stable")


# ---------------------------------------------------------------------------
# 2. Worked example
# ---------------------------------------------------------------------------

A = np.array([1.0, 2.0, 2.0])
B = np.array([2.0, 1.0, 2.0])
C = np.array([2.0, -1.0, -2.0])


def worked_example() -> dict[str, float]:
    """a, b, c from the table above. Returns the numbers you should be able to compute by hand."""
    return {
        "len_a": float(np.linalg.norm(A)),
        "dot_ab": dot(A, B),
        "dot_ac": dot(A, C),
        "cos_ab": cosine(A, B),
        "cos_ac": cosine(A, C),
        "l2_ab": euclidean(A, B),
        "l2_ac": euclidean(A, C),
    }


# ---------------------------------------------------------------------------
# 3. Normalization makes the metrics agree; without it they can disagree
# ---------------------------------------------------------------------------


def rankings_agree_after_normalizing(n_docs: int = 200, dim: int = 32, seed: int = 0) -> dict[str, bool]:
    """Random docs with wildly different lengths: do the three rankings agree?

    Before normalizing: dot product favors long vectors, so it disagrees.
    After normalizing: all three rankings are identical, as the algebra says.
    """
    rng = np.random.default_rng(seed)
    docs = rng.standard_normal((n_docs, dim)) * rng.uniform(0.2, 5.0, (n_docs, 1))  # varied lengths
    q = rng.standard_normal(dim)

    raw = {m: rank(q, docs, m) for m in ("cosine", "dot", "l2")}
    dn, qn = l2_normalize(docs), l2_normalize(q)
    norm = {m: rank(qn, dn, m) for m in ("cosine", "dot", "l2")}
    return {
        "raw_dot_equals_cosine": bool(np.array_equal(raw["dot"], raw["cosine"])),
        "raw_l2_equals_cosine": bool(np.array_equal(raw["l2"], raw["cosine"])),
        "norm_dot_equals_cosine": bool(np.array_equal(norm["dot"], norm["cosine"])),
        "norm_l2_equals_cosine": bool(np.array_equal(norm["l2"], norm["cosine"])),
    }


def l2_identity_check(seed: int = 0) -> tuple[float, float]:
    """Return (||a-b||², 2 - 2cos(a,b)) for two random unit vectors. They're equal."""
    rng = np.random.default_rng(seed)
    a, b = l2_normalize(rng.standard_normal(16)), l2_normalize(rng.standard_normal(16))
    return float(np.sum((a - b) ** 2)), 2 - 2 * cosine(a, b)


def popularity_in_the_norm() -> list[tuple[str, float, float]]:
    """When magnitude carries signal, dot product and cosine rank differently.

    Two items point in almost the same direction as the query. The "popular"
    one has a long vector (a model trained with dot product can learn to do
    this). Cosine ignores that and prefers the slightly better-aligned niche
    item; dot product prefers the popular one.
    """
    q = np.array([1.0, 0.0])
    items = {
        "niche item (well aligned, short)": np.array([0.99, 0.10]) * 1.0,
        "popular item (a bit off, long)": np.array([0.95, 0.30]) * 3.0,
    }
    return [(name, cosine(q, v), dot(q, v)) for name, v in items.items()]


# ---------------------------------------------------------------------------
# 4. Curse of dimensionality
# ---------------------------------------------------------------------------


def curse_of_dimensionality(dims=(2, 10, 100, 1000), n_points: int = 1000, seed: int = 0) -> list[dict]:
    """For random points, how much nearer is the nearest than the farthest?

    ratio = nearest / farthest distance from a random query. Near 0 means
    "nearest" is meaningful; near 1 means every point is about equally far.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for d in dims:
        X = rng.uniform(0, 1, (n_points, d))
        q = rng.uniform(0, 1, d)
        dist = np.linalg.norm(X - q, axis=1)
        rows.append({"dim": d, "nearest": float(dist.min()), "farthest": float(dist.max()), "ratio": float(dist.min() / dist.max())})
    return rows


# ---------------------------------------------------------------------------
# 5. Anisotropy, and thresholds chosen from labeled pairs
# ---------------------------------------------------------------------------


def anisotropic_embeddings(n_pairs: int = 300, dim: int = 768, seed: int = 0):
    """Simulate a model whose vectors all live in a narrow cone.

    Each vector = sqrt(3)·m + 1·n, where m is one shared unit direction and n
    is a per-text unit direction. For unrelated texts the n parts are nearly
    orthogonal, so cos ≈ 3 / (3 + 1) = 0.75. For paraphrases we make the n parts
    correlated (rho in [0.15, 0.45]), so cos ≈ 0.75 + 0.25·rho, about 0.79 to 0.86.
    Everything lands in the 0.75-0.85 band even though the underlying
    signal (rho) is clear.

    Returns (left, right, is_paraphrase): left[i] and right[i] form pair i.
    """
    rng = np.random.default_rng(seed)
    m = l2_normalize(rng.standard_normal(dim))
    alpha, beta = np.sqrt(3.0), 1.0

    def unit(k):
        return l2_normalize(rng.standard_normal((k, dim)))

    n_left = unit(2 * n_pairs)
    labels = np.array([True] * n_pairs + [False] * n_pairs)
    rho = np.where(labels, rng.uniform(0.15, 0.45, 2 * n_pairs), 0.0)
    # Correlated partner: rho·n + sqrt(1 - rho²)·fresh, which is unit length with cos(n, partner) ≈ rho.
    n_right = l2_normalize(rho[:, None] * n_left + np.sqrt(1 - rho[:, None] ** 2) * unit(2 * n_pairs))
    left = alpha * m + beta * n_left
    right = alpha * m + beta * n_right
    return left, right, labels


def pair_cosines(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(l2_normalize(left) * l2_normalize(right), axis=1)


def mean_center(*arrays: np.ndarray) -> list[np.ndarray]:
    """Subtract the mean vector of all given vectors (the shared "cone" direction).

    In production you'd estimate the mean once from a sample of your corpus
    and store it with the index; queries get the same shift.
    """
    mu = np.concatenate(arrays).mean(axis=0)
    return [a - mu for a in arrays]


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Sweep every candidate threshold; return the one with the best F1.

    Predict "similar" when score >= threshold. This is how you pick a dedup
    or cache threshold for a specific model, from a few hundred labeled pairs,
    instead of guessing "0.8".
    """
    best = {"threshold": 0.0, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for t in np.unique(scores):
        pred = scores >= t
        tp = np.sum(pred & labels)
        if tp == 0:
            continue
        precision, recall = tp / pred.sum(), tp / labels.sum()
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": float(f1), "precision": float(precision), "recall": float(recall)}
    return best


def f1_curve(scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """F1 of the rule "score >= t" for each t in `thresholds` (for plotting)."""
    out = []
    for t in thresholds:
        pred = scores >= t
        tp = np.sum(pred & labels)
        out.append(0.0 if tp == 0 else 2 * tp / (pred.sum() + labels.sum()))
    return np.array(out)


# ---------------------------------------------------------------------------
# 6. Figures (rendered into docs/figures by `make figures`) and the widget's data
# ---------------------------------------------------------------------------


def viz_data() -> dict:
    """The starting arrows for the site's interactive cosine-similarity widget."""
    # Each arrow is an angle in degrees (0 points right, 90 up) and a length.
    # The widget recomputes dot, cosine and distance with the same formulas as
    # dot, cosine and euclidean above; tests check the two agree.
    return {
        "cosine-similarity": {
            "presets": [
                {"name": "Same direction, different lengths", "a": {"angle": 30, "length": 1.0}, "b": {"angle": 30, "length": 3.0}},
                {"name": "Close in direction", "a": {"angle": 20, "length": 2.0}, "b": {"angle": 50, "length": 1.5}},
                {"name": "At right angles", "a": {"angle": 0, "length": 2.0}, "b": {"angle": 90, "length": 2.0}},
                {"name": "Opposite", "a": {"angle": 30, "length": 1.5}, "b": {"angle": 210, "length": 1.5}},
            ],
        }
    }


def figures() -> dict:
    """Plots computed from this module's own functions. Keys match the docstring's image names."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = {}

    # identity: ||a-b||² = 2 - 2cos for unit vectors
    rng = np.random.default_rng(0)
    a, b = l2_normalize(rng.standard_normal((400, 8))), l2_normalize(rng.standard_normal((400, 8)))
    cos = np.sum(a * b, axis=1)
    sq = np.sum((a - b) ** 2, axis=1)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    xs = np.linspace(-1, 1, 50)
    ax.plot(xs, 2 - 2 * xs, color="0.6", lw=2, label="y = 2 - 2x")
    ax.scatter(cos, sq, s=10, alpha=0.7, label="random unit-vector pairs")
    ax.set(xlabel="cosine similarity", ylabel="squared Euclidean distance", title="On unit vectors, distance is a function of cosine")
    ax.legend()
    figs["identity"] = fig

    # arrows: niche vs popular item against a query
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    q = np.array([1.0, 0.0])
    niche, popular = np.array([0.99, 0.10]), np.array([0.95, 0.30]) * 3.0
    for v, color, label in ((q, "black", "query"), (niche, "C0", "niche item (aligned, short)"), (popular, "C1", "popular item (off-angle, long)")):
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=color, lw=2))
        ax.plot([], [], color=color, lw=2, label=label)
    ax.set(xlim=(-0.1, 3.1), ylim=(-0.1, 1.1), xlabel="dimension 1", ylabel="dimension 2", title="Cosine picks blue (angle); dot product picks orange (length)")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    figs["arrows"] = fig

    # curse: nearest/farthest ratio vs dimension
    rows = curse_of_dimensionality(dims=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000))
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot([r["dim"] for r in rows], [r["ratio"] for r in rows], marker="o")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set(xlabel="dimensions (log scale)", ylabel="nearest distance / farthest distance", title="Random points become equidistant as dimension grows")
    figs["curse"] = fig

    # anisotropy: raw vs centered score histograms
    left, right, related = anisotropic_embeddings()
    raw = pair_cosines(left, right)
    lc, rc = mean_center(left, right)
    centered = pair_cosines(lc, rc)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, s, title in ((axes[0], raw, "raw cosine: everything is 0.7 to 0.9"), (axes[1], centered, "after mean-centering: the gap opens")):
        bins = np.linspace(s.min(), s.max(), 40)
        ax.hist(s[related], bins=bins, alpha=0.7, label="paraphrase pairs")
        ax.hist(s[~related], bins=bins, alpha=0.7, label="unrelated pairs")
        ax.set(xlabel="cosine similarity", ylabel="number of pairs", title=title)
        ax.legend()
    figs["anisotropy"] = fig

    # threshold: F1 sweep, guessed vs calibrated
    ts = np.linspace(0.72, 0.88, 200)
    best = best_threshold(raw, related)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(ts, f1_curve(raw, related, ts))
    ax.axvline(0.8, ls="--", color="0.4", label="guessed 0.80")
    ax.axvline(best["threshold"], color="C3", label=f"calibrated {best['threshold']:.3f}")
    ax.set(xlabel="threshold on raw cosine", ylabel="F1", title="Threshold choice on an anisotropic model")
    ax.legend()
    figs["threshold"] = fig

    for f in figs.values():
        f.tight_layout()
    return figs


# ---------------------------------------------------------------------------
# 7. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. Worked example: a=(1,2,2), b=(2,1,2), c=(2,-1,-2)")
    w = worked_example()
    table(
        ["pair", "dot", "|x||y|", "cosine", "L2 distance"],
        [("a, b", w["dot_ab"], 9, w["cos_ab"], w["l2_ab"]), ("a, c", w["dot_ac"], 9, w["cos_ac"], w["l2_ac"])],
        floatfmt=".2f",
    )
    say("Every vector has length 3, so cosine = dot / 9: 8/9 = 0.89 and -4/9 = -0.44.")

    banner("2. Normalize and the three metrics agree")
    sq, formula = l2_identity_check()
    say(f"For two random unit vectors: ||a-b||² = {sq:.6f} and 2 - 2cos = {formula:.6f}.")
    agree = rankings_agree_after_normalizing()
    table(["check (200 docs, varied lengths)", "same ranking?"], [(k, v) for k, v in agree.items()])
    takeaway("On L2-normalized vectors, cosine, dot product and Euclidean distance rank identically.")

    banner("3. When they disagree: signal stored in the vector length")
    table(["item", "cosine", "dot"], popularity_in_the_norm(), floatfmt=".3f")
    say(
        """
        Cosine prefers the better-aligned niche item; dot product prefers the
        long "popular" vector. Neither is wrong. Use whatever the model was
        trained with.
        """
    )

    banner("4. Curse of dimensionality (random points)")
    table(["dim", "nearest", "farthest", "nearest / farthest"], [(r["dim"], r["nearest"], r["farthest"], r["ratio"]) for r in curse_of_dimensionality()], floatfmt=".3f")
    say("As dimension grows the ratio climbs toward 1: every random point is about equally far away.")

    banner("5. Anisotropy: every score sits in 0.75-0.85")
    left, right, labels = anisotropic_embeddings()
    raw = pair_cosines(left, right)
    table(
        ["pairs", "min cos", "mean cos", "max cos"],
        [
            ("paraphrases", raw[labels].min(), raw[labels].mean(), raw[labels].max()),
            ("unrelated", raw[~labels].min(), raw[~labels].mean(), raw[~labels].max()),
        ],
        floatfmt=".3f",
    )
    naive = raw >= 0.8
    naive_f1 = 2 * np.sum(naive & labels) / (naive.sum() + labels.sum())
    best = best_threshold(raw, labels)
    say(
        f"""
        A "magic" threshold of 0.80 gives F1 = {naive_f1:.3f}. Calibrating on
        labeled pairs picks {best['threshold']:.3f} with F1 = {best['f1']:.3f}.
        The signal is there, squeezed into a few hundredths.
        """
    )
    lc, rc = mean_center(left, right)
    centered = pair_cosines(lc, rc)
    table(
        ["after mean-centering", "mean cos"],
        [("paraphrases", centered[labels].mean()), ("unrelated", centered[~labels].mean())],
        floatfmt=".3f",
    )
    takeaway(
        "Raw scores aren't comparable across models. Rank when you can; when you "
        "must threshold, calibrate on labeled pairs from your own data, per model."
    )


if __name__ == "__main__":
    demo()
