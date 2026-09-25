r"""
# Attention: how a token decides what to listen to

Run: `python -m primer.ml.attention`

## The everyday picture

Imagine you're in a meeting and someone says "it's broken, can you fix it?"
To know what "it" means, you glance around the room: at the laptop on the
table, at the person who just walked in, at the whiteboard. You pay a lot of
attention to the laptop, a little to everything else, and your understanding
of "it" becomes mostly *laptop*.

That glance is attention. Every word in a sentence gets to look at every
other word, decide how relevant each one is, and rebuild its own meaning as
a mix of the relevant ones. "Bank" next to "river" becomes a riverbank; next
to "loan" it becomes a lender. The word is the same, but what it paid
attention to is different.

## A tiny worked example: what does "it" refer to?

Take "The animal didn't cross the street because it was tired" and follow
the word "it". To keep the numbers small, it compares itself with just three
other words. Each comparison gives a relevance **score** (how the score is
made comes next). Then we turn scores into shares of attention that add up
to 1:

1. Exponentiate each score (e^score), which makes every number positive and
   stretches the gaps between them.
2. Divide each by the total, 7.39 + 2.72 + 1.65 = 11.76.

| Word   | Score | e^score | Share of attention |
|--------|-------|---------|--------------------|
| animal | 2.0   | 7.39    | 7.39 / 11.76 = **0.63** |
| tired  | 1.0   | 2.72    | 2.72 / 11.76 = **0.23** |
| street | 0.5   | 1.65    | 1.65 / 11.76 = **0.14** |

Those two steps are called **softmax**. The new meaning of "it" is 63%
"animal", 23% "tired" and 14% "street". Without being told any grammar rule,
the model has worked out that "it" is the animal.

**In code:** `worked_example_it` runs these three scores through softmax and returns each word's share.

### Softmax, decoded

Softmax turns any list of numbers, positive or negative, into shares that
are all positive and add up to exactly 1. Think of it as a vote where louder
voices get disproportionately more say.

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $z$ | the list of scores going in | (2.0, 1.0, 0.5) |
| $n$ | how many scores there are | 3 |
| $i$ | the position we're computing a share for | 1 = "animal" |
| $z_i$ | the score at position $i$ | $z_1 = 2.0$ |
| $e$ | Euler's number, ≈ 2.718; $e^x$ is "2.718 multiplied by itself $x$ times" (works for fractions and negatives too) | $e^{2.0} = 7.39$ |
| $\sum_{j=1}^{n}$ | "add up the following, for $j$ = 1, 2, …, $n$" | $e^{2.0} + e^{1.0} + e^{0.5}$ |
| $j$ | a counter that walks over every position | 1, 2, 3 |
| $\text{softmax}(z)_i$ | the share of attention position $i$ gets | 0.63 |

**In words:** "the share for item *i* is *e* raised to its score, divided by
the sum of *e* raised to every score."

**With the numbers:** softmax(2.0, 1.0, 0.5)₁ = 7.39 / (7.39 + 2.72 + 1.65) =
7.39 / 11.76 = 0.63.

Why *e* to the power of the score, and not just the score divided by the
total? Three reasons you can check on the table above:

1. **Always positive.** Scores can be negative; *e* to any power is positive,
   so no word ever gets a negative share.
2. **Order is kept.** A higher score always gets a bigger share.
3. **Gaps are stretched.** Adding 1 to a score multiplies its *e*-value by
   2.72, so the winner pulls ahead decisively. (Plain division would give
   "animal" 2.0 / 3.5 = 0.57, a timid 57%, and would break on negative
   scores.)

One more property matters later: adding the same number to every score
changes nothing, because it multiplies the top and bottom of the fraction by
the same amount. The code uses this to subtract the largest score before
exponentiating, which stops *e*¹⁰⁰⁰ from overflowing to infinity. See
`primer.notation` for exponents and sums from scratch.

![Scores become weights for the word "it"](figures/primer.ml.attention.it_weights.svg)

**Reading it:** for each word, the grey bar is its share of the raw scores
and the blue bar is its share of attention after softmax. Softmax exaggerates
differences: "animal" scores only twice as high as "tired", yet it ends up
with almost three times the attention, because exponentiating stretches the
gaps. That is how attention commits to the most relevant word while keeping
a little of the others.

**In code:** `softmax` subtracts the largest score before exponentiating, and turns masked scores of −∞ into weights of exactly 0.

## Where the scores come from: queries, keys and values

Each word turns its vector into three new vectors by multiplying it by three
learned matrices:

- a **query**: what am I looking for? ("it" is looking for a noun that can
  be tired.)
- a **key**: what do I offer to others? ("animal" offers: a noun, alive.)
- a **value**: what I actually hand over if someone picks me.

A word's score for another word is the **dot product** of its query with the
other word's key: multiply them position by position and add.

$$
q \cdot k = \sum_{m=1}^{d_k} q_m \, k_m
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $q$ | the query vector: a list of $d_k$ numbers | (1, 2) |
| $k$ | the key vector: another list of $d_k$ numbers | (3, 0.5) |
| $d_k$ | how many numbers each list has (the head width) | 2 |
| $m$ | a counter walking over the positions | 1, 2 |
| $q_m, k_m$ | the $m$-th number in each list | $q_1 = 1$, $k_1 = 3$ |
| $\cdot$ | "dot product": multiply matching positions, then add | |

**In words:** "multiply the first numbers together, multiply the second
numbers together, and so on, then add up all the products."

**With the numbers:** (1, 2) · (3, 0.5) = 1·3 + 2·0.5 = 3 + 1 = **4**.

Vectors that point the same way give big positive scores, vectors at right
angles give zero, and opposite vectors give negative scores. That is why the
dot product works as a relevance score. The values are then blended with
the softmax shares, exactly as in the table above.

A library search is a good analogy. The query is what you type into the
search box, the keys are the catalogue entries, and the values are the books
on the shelf. You get back a *blend* of books, weighted by how well each
catalogue entry matched your search.

```mermaid
flowchart LR
  X[Token vectors] --> Q[Query] & K[Key] & V[Value]
  Q --> S[Scores<br/>Q times K]
  K --> S
  S --> SC[Scale by<br/>sqrt of d_k]
  SC --> M[Causal mask<br/>hide future tokens]
  M --> SM[Softmax<br/>weights sum to 1]
  SM --> W[Weighted sum<br/>of values]
  V --> W
  W --> O[Context-aware vectors]
```

**Reading it:** follow the arrows left to right. The token vectors split
three ways. Queries and keys meet in the Scores box: that is where relevance
is decided. The values take the lower path and wait, untouched, until the
Weighted sum box, where the relevance shares decide how much of each value
is mixed in. Keep the two roles apart: **Q and K decide *how much*; V carries
*what*.** The Scale and Mask boxes are explained in their own sections below.

Written as one formula, with every word's query stacked into a matrix Q (and
likewise K and V), the whole diagram is:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

**Symbols**

| Symbol | Meaning here | Shape |
|---|---|---|
| $n$ | number of tokens in the sequence | |
| $d_k$ | width of each query and key vector | |
| $d_v$ | width of each value vector | |
| $Q$ | every token's query, stacked as rows | $n \times d_k$ |
| $K$ | every token's key, stacked as rows | $n \times d_k$ |
| $V$ | every token's value, stacked as rows | $n \times d_v$ |
| $K^\top$ | $K$ **transposed**: rows become columns, so each key stands upright | $d_k \times n$ |
| $QK^\top$ | a **matrix multiply**: entry (row $i$, column $j$) is the dot product of token $i$'s query with token $j$'s key, so it is every score at once | $n \times n$ |
| $\sqrt{d_k}$ | square root of the head width; dividing by it keeps scores from growing with width (explained below) | a single number |
| softmax(…) | softmax applied to each row separately, so each token's shares add to 1 | $n \times n$ |
| $(\ldots)V$ | multiply the shares by the values: each output row is a share-weighted blend of value rows | $n \times d_v$ |

**In words:** "score every query against every key, shrink the scores by √d_k,
turn each row of scores into shares, and use the shares to blend the values."

**With the numbers:** the "it" row of $QK^\top/\sqrt{d_k}$ holds (2.0, 1.0,
0.5); softmax turns that row into (0.63, 0.23, 0.14); multiplying by $V$
gives 0.63·V(animal) + 0.23·V(tired) + 0.14·V(street).

In practice this one line runs in every layer of every modern language
model, for every word, many times per word generated. Nearly everything
about a model's speed and memory use traces back to it.

**In code:** `scaled_dot_product_attention` is the whole formula, one commented step per box of the diagram, and returns both the blended values and the attention weights.

## Shapes: the part that is easiest to get wrong

For one sequence of `n` tokens with model width `d_model`:

| Tensor | Shape | Meaning |
|---|---|---|
| X | (n, d_model) | input token vectors |
| W_q, W_k | (d_model, d_k) | learned projections |
| W_v | (d_model, d_v) | learned projection |
| Q, K | (n, d_k) | queries, keys |
| V | (n, d_v) | values |
| Q Kᵀ | **(n, n)** | every token vs. every token, hence O(n²) |
| weights | (n, n) | each row sums to 1 |
| output | (n, d_v) | one context-aware vector per token |

## Causal masking: no peeking at the answer

A decoder model (GPT, Claude, Llama) is trained to predict the *next* token.
If position i could attend to position i+1, it could simply read the answer.
So the scores for future positions are set to −∞ *before* softmax. Because
e^−∞ = 0, those weights come out as exactly zero.

```mermaid
flowchart LR
  subgraph S["Scores (n × n)"]
    direction TB
    r0["row 'The':    ✓ · · ·"]
    r1["row 'cat':    ✓ ✓ · ·"]
    r2["row 'sat':    ✓ ✓ ✓ ·"]
    r3["row 'down':   ✓ ✓ ✓ ✓"]
  end
  S --> MASK["set every · to −∞"] --> SOFT["softmax per row"] --> OUT["each row sums to 1<br/>using only the past"]
```

**Reading it:** each row is one token looking at the sequence. A ✓ is a
position it may read and a · is the future, which gets masked. The allowed
region is a lower triangle: the first token sees only itself and the last
sees everything before it. The same triangle appears as the dark region in
the heatmap below.

![Causal attention weights on a real sentence](figures/primer.ml.attention.causal_heatmap.svg)

**Reading it:** rows are the token doing the looking (the query), columns are
the tokens being looked at (the keys), and darker means more weight. The
upper-right triangle is exactly zero, so nothing reads the future. Each row
sums to 1, so a token's attention is a budget it spends across the past. The
weights come from random projections here, so the pattern itself means
nothing; the triangle is what matters. In a trained model, rows light up on
the tokens that actually help, such as "it" lighting up "animal".

Masking only the future is also what makes generation cheap: a token's
output never changes when later tokens arrive, so it can be computed once
and cached. See `primer.ml.inference` for the KV cache.

**In code:** `causal_mask` builds the lower triangle of allowed positions, and `scaled_dot_product_attention` sets every score outside it to −∞ before softmax.

## Why divide by √d_k?

**Everyday picture.** Roll one die and the result swings between 1 and 6.
Add up a hundred dice and the total swings far more, by dozens in either
direction. A dot product is exactly that kind of sum: one small random-ish
product per position. The wider the vectors, the more terms get added, and
the wilder the scores swing.

**Tiny example.** With d_k = 4, a query and a key of random ±1 entries give
four products of +1 or −1, so the score lands between −4 and +4 and is
usually within ±2. With d_k = 256 it's a sum of 256 such products: usually
within ±16. Softmax of scores like (16, 3, −9) gives about (0.999998, …, …).
All the attention lands on one word, whether or not that is right.

$$
\operatorname{Var}(q \cdot k) = d_k
\qquad\Longrightarrow\qquad
\operatorname{Var}\!\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = 1
$$

**Symbols**

| Symbol | Meaning here |
|---|---|
| $\operatorname{Var}(x)$ | **variance**: the average squared distance of $x$ from its average; a measure of how widely $x$ swings. Its square root is the **standard deviation**, the typical size of a swing |
| $q \cdot k$ | a raw attention score, for $q$ and $k$ whose entries are independent with average 0 and variance 1 |
| $d_k$ | the number of products added up in the dot product |
| $\Longrightarrow$ | "which implies" |
| $\sqrt{d_k}$ | dividing a quantity by $c$ divides its variance by $c^2$; here $c^2 = d_k$ |

**In words:** "the spread of a raw score grows with the head width, so we
divide the score by the square root of the width, which brings the spread
back to 1 at any width."

**With the numbers:** at d_k = 128, the standard deviation of a raw score is
√128 ≈ 11.3, so scores of ±20 are routine. After dividing by √128 the
standard deviation is 1, so scores of ±2 are typical.

Why "all attention on one word" is bad, beyond being a wrong answer: softmax
has stopped responding. Its **gradient** (the signal training uses to
adjust the weights) is `diag(p) − p pᵀ`, and when one share is 1 and the rest
are 0, every entry of that matrix is 0. The query and key weights receive no
signal and stop learning. This is called *saturation*.
`sqrt_dk_experiment` measures all of this.

![Unscaled attention saturates as width grows](figures/primer.ml.attention.sqrt_dk.svg)

**Reading it:** the x-axis is the head width d_k, on a log scale. On the left,
the y-axis is the average largest attention weight: 1.0 means one-hot, all
attention on one token. On the right, it is the size of the softmax gradient.
The unscaled lines (red) climb towards 1.0 on the left and fall towards zero
on the right as d_k grows: the wider the head, the more attention collapses
onto a single token and the less signal flows back. The scaled lines (blue)
stay flat at every width. That flatness is the whole reason for the √d_k.

In one sentence: *dot products grow with dimension, large scores saturate
softmax and kill gradients, and scaling by √d_k keeps scores in the range
where softmax is smooth and trainable.*

**In code:** `softmax_jacobian` builds the diag(p) − p pᵀ matrix, so you can watch every entry fall to 0 as one share approaches 1.

## Multi-head attention

Instead of one attention with d_k = d_model, run h heads in parallel, each
with d_k = d_model / h and its own W_q, W_k and W_v. The total compute is the
same, but each head can specialise: one tracks syntax, another which noun a
pronoun refers to, another the previous token.

```mermaid
flowchart LR
  X["X (n × d_model)"] --> P["project with W_q, W_k, W_v"]
  P --> H1["head 1<br/>attention on d_model/h dims"]
  P --> H2["head 2"]
  P --> H3["..."]
  P --> Hh["head h"]
  H1 & H2 & H3 & Hh --> C["concatenate<br/>(n × d_model)"]
  C --> WO["mix with W_o"] --> Y["output (n × d_model)"]
```

**Reading it:** the input is projected once, and the result is sliced into h
narrow slabs, one per head. Each head runs the full attention recipe from the
first diagram on its own slab, independently and in parallel. The heads'
outputs are glued back side by side, and W_o lets them exchange what they
found. The output has the same shape as the input, which is what lets blocks
stack.

**In code:** `MultiHeadAttention` holds the four projections W_q, W_k, W_v and W_o; calling it slices the projections into one slab per head, runs `scaled_dot_product_attention` on every head at once, and glues the results back together before W_o.

## Grouped-query attention: sharing keys and values

With **grouped-query attention (GQA)**, several query heads share one
key/value head. Llama 3 8B has 32 query heads but only 8 KV heads, so it
stores 4× fewer keys and values per token. Multi-query attention (MQA) is the
extreme: one KV head for everyone.

```mermaid
flowchart TB
  subgraph MHA["MHA: 4 query heads, 4 KV heads"]
    q1a[Q1]-->kv1a[KV1]
    q2a[Q2]-->kv2a[KV2]
    q3a[Q3]-->kv3a[KV3]
    q4a[Q4]-->kv4a[KV4]
  end
  subgraph GQA["GQA: 4 query heads, 2 KV heads"]
    q1b[Q1]-->kv1b[KV1]
    q2b[Q2]-->kv1b
    q3b[Q3]-->kv2b[KV2]
    q4b[Q4]-->kv2b
  end
  subgraph MQA["MQA: 4 query heads, 1 KV head"]
    q1c[Q1]-->kv1c[KV1]
    q2c[Q2]-->kv1c
    q3c[Q3]-->kv1c
    q4c[Q4]-->kv1c
  end
```

**Reading it:** count the KV boxes. Every query head still asks its own
question, but in GQA and MQA they read from a shared set of keys and values.
The KV boxes are what gets stored in GPU memory for every token of every
conversation during generation (the KV cache). Fewer boxes means more
conversations fit on one GPU, at a small cost in quality. See
`primer.ml.inference` for the memory arithmetic.

**In code:** `MultiHeadAttention` takes a number of KV heads: fewer than the query heads is GQA, one is MQA. `MultiHeadAttention.kv_params` counts the key and value weights that shrink.

## Cost: why long context is expensive

The score matrix is n × n per head per layer, so compute and memory grow
with **n²**. Doubling the context roughly quadruples attention's cost.

![Quadratic attention overtakes linear projections](figures/primer.ml.attention.quadratic_cost.svg)

**Reading it:** both axes are logarithmic, so a straight line is a power law
and a steeper line grows faster. The projections (X·W) cost O(n·d²), a line of
slope 1. The score-and-mix step costs O(n²·d), a line of slope 2. The dashed
marker is where they cross, at n = 2·d_model. Past that point, most of the
work is tokens comparing themselves to other tokens, and every doubling of
context costs 4× there.

The standard mitigations each attack this picture. FlashAttention produces
the exact same result but computes it in tiles sized for fast on-chip GPU
memory, so the n × n matrix is never written to slow memory. Sliding-window
and sparse attention let each token see only some others. GQA and MQA shrink
the KV cache. State-space models such as Mamba replace attention with a
linear-time recurrence.

**In code:** `attention_cost` counts the quadratic score-and-mix FLOPs and the linear projection FLOPs plotted above.

## In 20 seconds

- **Attention:** each token builds a query, key and value. Query-key
  similarity decides how much each token listens to each other token, and
  the output is a weighted blend of their values.
- **Why √d_k:** dot products grow with dimension, and large scores saturate
  softmax and kill gradients. Scaling keeps training stable.
- **Causal mask:** future scores are set to −∞ before softmax, so each token
  sees only the past. That is what makes next-token training honest and
  generation cacheable.
- **Long context cost:** attention compares every pair of tokens, so it is
  O(n²), and the KV cache grows with every token.

## Self-test questions

**Explain attention to a non-engineer in 30 seconds.**
When the model reads a word, it asks which other words here help it
understand this one. It scores every other word for relevance, then builds
its understanding of the word as a mix of the relevant ones. In "the animal
didn't cross the street because it was tired", "it" draws mostly from
"animal". It does this for every word, in parallel, dozens of times over.

**Now explain it to an ML engineer in two minutes.**
Project X into Q, K and V with learned matrices. Compute QKᵀ/√d_k, an n×n
matrix of scaled similarities; add a causal mask of −∞ above the diagonal
for a decoder; softmax each row; multiply by V. Run h heads in parallel on
d_model/h slices, concatenate them, and project with W_o. Wrap the whole
thing in a residual connection with pre-layer-norm and follow it with an
FFN. Cost is O(n²·d) in compute and O(n²) in memory for the scores, which is
why FlashAttention tiles it and GQA shrinks the KV cache.

**Why divide by √d_k? What breaks without it?**
The variance of q·k grows linearly with d_k. Without scaling, scores spread
out, softmax saturates towards one-hot, its Jacobian goes to about zero, and
the query/key projections stop receiving gradient. Training stalls or turns
unstable.

**Why is long context expensive? Name two techniques that reduce the cost.**
Scores are n×n per head per layer, so cost is quadratic, and the KV cache
grows linearly with every token held in memory. Two fixes: FlashAttention
(exact, memory-efficient tiling) and GQA/MQA (fewer KV heads). Others:
sliding-window or sparse attention, and prompt caching of stable prefixes.

**What does the causal mask buy you besides honest training?**
Earlier outputs never depend on later tokens, so during generation the
keys and values of past tokens can be computed once and cached. Each new
token then costs one row of attention instead of a full recomputation.

**What does GQA trade away, and for what?**
A little modelling capacity (query heads share keys and values) for a KV
cache that is several times smaller. That means more concurrent requests
and longer contexts per GPU.

## The papers behind this lesson

- **Vaswani et al., *Attention Is All You Need* (2017)**:
  https://arxiv.org/abs/1706.03762. Showed that attention alone, with no
  recurrence, is enough for state-of-the-art translation, and introduced
  scaled dot-product attention, multi-head attention and the transformer.
  [Annotated companion](../../papers/attention-is-all-you-need.html)
- **Ainslie et al., *GQA: Training Generalized Multi-Query Transformer
  Models from Multi-Head Checkpoints* (2023)**:
  https://arxiv.org/abs/2305.13245. Introduced grouped-query attention, the
  middle ground between multi-head and multi-query attention.
- **Dao et al., *FlashAttention* (2022)**: https://arxiv.org/abs/2205.14135.
  Computes exact attention in tiles sized for fast on-chip GPU memory,
  making long contexts practical.
  [Annotated companion](../../papers/flashattention.html)

## Further reading

- Vaswani et al., *Attention Is All You Need* (2017): https://arxiv.org/abs/1706.03762
- Jay Alammar, *The Illustrated Transformer*: https://jalammar.github.io/illustrated-transformer/
- Harvard NLP, *The Annotated Transformer* (the paper, line by line in code): https://nlp.seas.harvard.edu/annotated-transformer/
- Andrej Karpathy, *Let's build GPT: from scratch, in code, spelled out*: https://www.youtube.com/watch?v=kCc8FmEb1nY and nanoGPT: https://github.com/karpathy/nanoGPT
- 3Blue1Brown, *Attention in transformers, visually explained*: https://www.youtube.com/watch?v=eMlx5fFNoYc
- PyTorch `scaled_dot_product_attention`: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- Ainslie et al., *GQA* (2023): https://arxiv.org/abs/2305.13245
- Dao et al., *FlashAttention* (2022): https://arxiv.org/abs/2205.14135
"""

from __future__ import annotations

import numpy as np

from primer._show import banner, matrix, say, takeaway, table

# ---------------------------------------------------------------------------
# 1. Softmax: turns any vector of scores into probabilities that sum to 1
# ---------------------------------------------------------------------------


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along `axis`.

    softmax(x)_i = exp(x_i) / sum_j exp(x_j)

    Subtracting the max first doesn't change the result (it cancels in the
    ratio) but prevents overflow: exp(1000) is inf, exp(1000 - 1000) = 1.
    Entries equal to -inf (masked positions) come out as exactly 0.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    # If a whole row is -inf (fully masked), avoid inf - inf = nan.
    x_max = np.where(np.isfinite(x_max), x_max, 0.0)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def softmax_jacobian(p: np.ndarray) -> np.ndarray:
    """d softmax / d scores for one probability vector p: diag(p) - p pᵀ.

    When p is one-hot (saturated), every entry is ~0, so no gradient
    reaches the scores and the query/key weights stop learning. That is the
    failure the sqrt(d_k) scaling prevents.
    """
    return np.diag(p) - np.outer(p, p)


# ---------------------------------------------------------------------------
# 2. Scaled dot-product attention
# ---------------------------------------------------------------------------


def causal_mask(n: int) -> np.ndarray:
    """Boolean (n, n) mask, True where attention is ALLOWED.

    Row i may attend to columns 0..i (itself and the past), never the future:

        n=4 ->  [[1 0 0 0]
                 [1 1 0 0]
                 [1 1 1 0]
                 [1 1 1 1]]
    """
    return np.tril(np.ones((n, n), dtype=bool))


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
    scale: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """softmax(Q Kᵀ / sqrt(d_k)) V, the formula, one line per step.

    Args:
        Q: (..., n_q, d_k) queries. Leading dims (batch, heads) broadcast.
        K: (..., n_k, d_k) keys.
        V: (..., n_k, d_v) values.
        mask: boolean (n_q, n_k), True = may attend. None = attend everywhere.
        scale: divide by sqrt(d_k). Set False only to demonstrate why you shouldn't.

    Returns:
        (output, weights): output is (..., n_q, d_v); weights is
        (..., n_q, n_k) with rows summing to 1.
    """
    d_k = Q.shape[-1]

    # Step 1: scores. Every query dotted with every key: (n_q, d_k) @ (d_k, n_k).
    # High dot product = vectors point the same way = "this key is relevant".
    scores = Q @ np.swapaxes(K, -1, -2)

    # Step 2: scale so score variance is ~1 regardless of d_k.
    if scale:
        scores = scores / np.sqrt(d_k)

    # Step 3: mask. -inf before softmax -> weight exactly 0 after.
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)

    # Step 4: softmax each row -> attention weights (each row sums to 1).
    weights = softmax(scores, axis=-1)

    # Step 5: weighted sum of values: (n_q, n_k) @ (n_k, d_v).
    return weights @ V, weights


# ---------------------------------------------------------------------------
# 3. Multi-head attention (with optional grouped-query attention)
# ---------------------------------------------------------------------------


class MultiHeadAttention:
    """Multi-head self-attention with learned projections (forward pass only).

    `n_kv_heads < n_heads` gives grouped-query attention (GQA): each K/V head
    is shared by `n_heads // n_kv_heads` query heads. `n_kv_heads == 1` is
    multi-query attention (MQA). `n_kv_heads == n_heads` is classic MHA.

    Weights are random here (we're studying the mechanics, not training);
    `primer.ml.transformer` stacks this into a full block.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int | None = None, seed: int = 0):
        n_kv_heads = n_kv_heads or n_heads
        assert d_model % n_heads == 0, "d_model must split evenly across heads"
        assert n_heads % n_kv_heads == 0, "query heads must group evenly over KV heads"
        self.d_model, self.n_heads, self.n_kv_heads = d_model, n_heads, n_kv_heads
        self.d_head = d_model // n_heads
        rng = np.random.default_rng(seed)
        # Scaled init keeps activations ~unit variance (see primer.ml.deep_nets).
        s = 1 / np.sqrt(d_model)
        self.W_q = rng.normal(0, s, (d_model, n_heads * self.d_head))
        self.W_k = rng.normal(0, s, (d_model, n_kv_heads * self.d_head))  # smaller when GQA
        self.W_v = rng.normal(0, s, (d_model, n_kv_heads * self.d_head))
        self.W_o = rng.normal(0, s, (n_heads * self.d_head, d_model))

    def _split(self, x: np.ndarray, n: int) -> np.ndarray:
        # (seq, n*d_head) -> (n, seq, d_head): one slab per head.
        seq = x.shape[0]
        return x.reshape(seq, n, self.d_head).transpose(1, 0, 2)

    def __call__(self, X: np.ndarray, causal: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """X: (seq, d_model) -> (output (seq, d_model), weights (n_heads, seq, seq))."""
        seq = X.shape[0]
        Q = self._split(X @ self.W_q, self.n_heads)  # (H,   seq, d_head)
        K = self._split(X @ self.W_k, self.n_kv_heads)  # (Hkv, seq, d_head)
        V = self._split(X @ self.W_v, self.n_kv_heads)

        # GQA: repeat each KV head for the query heads in its group.
        # (Real kernels avoid materializing the copies; the math is the same.)
        group = self.n_heads // self.n_kv_heads
        K = np.repeat(K, group, axis=0)
        V = np.repeat(V, group, axis=0)

        mask = causal_mask(seq) if causal else None
        heads, weights = scaled_dot_product_attention(Q, K, V, mask)  # (H, seq, d_head)

        # Concatenate heads back to (seq, H*d_head), then mix with W_o.
        concat = heads.transpose(1, 0, 2).reshape(seq, self.n_heads * self.d_head)
        return concat @ self.W_o, weights

    def kv_params(self) -> int:
        """Parameters in the K and V projections. GQA shrinks this, and the KV cache."""
        return self.W_k.size + self.W_v.size


# ---------------------------------------------------------------------------
# 4. Worked examples
# ---------------------------------------------------------------------------

IT_EXAMPLE_TOKENS = ["animal", "tired", "street"]
IT_EXAMPLE_SCORES = np.array([2.0, 1.0, 0.5])  # already scaled


def worked_example_it() -> dict[str, float]:
    """Worked example: the token "it" attending over three keys.

    | Token  | Scaled score | e^score | Weight |
    |--------|--------------|---------|--------|
    | animal | 2.0          | 7.39    | 0.63   |
    | tired  | 1.0          | 2.72    | 0.23   |
    | street | 0.5          | 1.65    | 0.14   |
    """
    w = softmax(IT_EXAMPLE_SCORES)
    return dict(zip(IT_EXAMPLE_TOKENS, w.tolist()))


def sqrt_dk_experiment(d_ks=(4, 64, 512), n_keys: int = 16, trials: int = 2000, seed: int = 0) -> list[dict]:
    """Measure what happens to attention scores as d_k grows, with and without scaling.

    For each d_k we sample random unit-variance q and k, then record:
      * std of the raw dot product (theory: sqrt(d_k))
      * mean max softmax weight (≈1.0 means saturated / one-hot)
      * mean softmax-Jacobian norm (≈0 means no gradient flows back)
    """
    rng = np.random.default_rng(seed)
    rows = []
    for d in d_ks:
        q = rng.standard_normal((trials, 1, d))
        k = rng.standard_normal((trials, n_keys, d))
        raw = (q @ k.transpose(0, 2, 1))[:, 0, :]  # (trials, n_keys)
        for scaled in (False, True):
            s = raw / np.sqrt(d) if scaled else raw
            p = softmax(s)
            jac = np.mean([np.linalg.norm(softmax_jacobian(pi)) for pi in p[:200]])
            rows.append(
                dict(d_k=d, scaled=scaled, score_std=float(s.std()), max_weight=float(p.max(axis=1).mean()), grad_norm=float(jac))
            )
    return rows


def attention_cost(n: int, d_model: int, n_layers: int = 1) -> dict[str, float]:
    """Rough FLOPs of the attention *score* and *mix* steps for n tokens.

    QKᵀ is n·n·d multiply-adds, and weights·V is another n·n·d. Count 2 FLOPs
    per multiply-add. The projections (X·W) are O(n·d²), linear in n, so for
    long contexts the n² term dominates.
    """
    score_and_mix = 2 * (2 * n * n * d_model) * n_layers
    projections = 2 * (4 * n * d_model * d_model) * n_layers
    return dict(n=n, quadratic_flops=score_and_mix, linear_flops=projections, score_matrix_entries=n * n)


# ---------------------------------------------------------------------------
# 5. Figures (rendered into the HTML docs by `make figures`)
# ---------------------------------------------------------------------------


def figures() -> dict:
    """Plot this lesson's data. matplotlib is imported here, and only here,
    so the lesson itself needs nothing beyond NumPy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SCALED, UNSCALED, MUTED = "#2563eb", "#dc2626", "#9ca3af"
    figs = {}

    # --- 1. Scores -> softmax weights for "it" ------------------------------
    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(len(IT_EXAMPLE_TOKENS))
    weights = softmax(IT_EXAMPLE_SCORES)
    ax.bar(x - 0.2, IT_EXAMPLE_SCORES / IT_EXAMPLE_SCORES.sum(), 0.4, color=MUTED, label="score (share of total)")
    ax.bar(x + 0.2, weights, 0.4, color=SCALED, label="attention weight (softmax)")
    for xi, w in zip(x, weights):
        ax.text(xi + 0.2, w + 0.02, f"{w:.2f}", ha="center")
    ax.set_xticks(x, IT_EXAMPLE_TOKENS)
    ax.set_ylabel("share")
    ax.set_ylim(0, 0.78)
    ax.set_title('What "it" attends to: softmax sharpens the scores')
    ax.legend(frameon=False)
    figs["it_weights"] = fig

    # --- 2. Causal attention heatmap on a sentence --------------------------
    tokens = "The animal didn't cross the street because it was tired".split()
    rng = np.random.default_rng(7)
    X = rng.standard_normal((len(tokens), 32))
    _, w = MultiHeadAttention(32, n_heads=1, seed=3)(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(w[0], cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(tokens)), tokens, rotation=45, ha="right")
    ax.set_yticks(range(len(tokens)), tokens)
    ax.set_xlabel("key: the token being looked at")
    ax.set_ylabel("query: the token doing the looking")
    ax.set_title("Causal attention: each row sees only the past")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, label="attention weight")
    figs["causal_heatmap"] = fig

    # --- 3. Why sqrt(d_k): saturation and vanishing gradient ----------------
    dks = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    rows = sqrt_dk_experiment(d_ks=dks, trials=600)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    for scaled, color, label in ((False, UNSCALED, "unscaled  QKᵀ"), (True, SCALED, "scaled  QKᵀ/√d_k")):
        sel = [r for r in rows if r["scaled"] == scaled]
        a1.plot(dks, [r["max_weight"] for r in sel], "o-", color=color, label=label)
        a2.plot(dks, [r["grad_norm"] for r in sel], "o-", color=color, label=label)
    for a, title, ylabel in (
        (a1, "Softmax saturates (→ one-hot)", "mean largest weight"),
        (a2, "Gradient through softmax vanishes", "softmax Jacobian norm"),
    ):
        a.set_xscale("log", base=2)
        a.set_xlabel("head width d_k")
        a.set_ylabel(ylabel)
        a.set_title(title)
    a1.set_ylim(0, 1.05)
    a1.legend(frameon=False)
    fig.tight_layout()
    figs["sqrt_dk"] = fig

    # --- 4. Quadratic vs linear cost ----------------------------------------
    d_model = 4096
    ns = np.logspace(2, 6, 60)
    quad = [attention_cost(int(n), d_model)["quadratic_flops"] for n in ns]
    lin = [attention_cost(int(n), d_model)["linear_flops"] for n in ns]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.loglog(ns, quad, color=UNSCALED, label="scores + mix  O(n²·d)")
    ax.loglog(ns, lin, color=SCALED, label="projections  O(n·d²)")
    ax.axvline(2 * d_model, color=MUTED, ls="--")
    ax.text(2 * d_model * 1.15, min(quad) * 10, "crossover\nn = 2·d_model", color="#4b5563")
    ax.set_xlabel("context length n (tokens)")
    ax.set_ylabel("FLOPs per layer")
    ax.set_title(f"Why long context is expensive (d_model = {d_model})")
    ax.legend(frameon=False)
    figs["quadratic_cost"] = fig

    return figs


# ---------------------------------------------------------------------------
# 6. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. Worked example: what does 'it' attend to?")
    say(
        """
        "The animal didn't cross the street because it was tired." Suppose the
        query for "it", dotted with three keys and scaled, gives scores 2.0,
        1.0 and 0.5. Softmax exponentiates and normalizes:
        """
    )
    w = worked_example_it()
    e = np.exp(IT_EXAMPLE_SCORES)
    table(
        ["token", "score", "e^score", "weight"],
        [(t, s, ei, wi) for t, s, ei, wi in zip(IT_EXAMPLE_TOKENS, IT_EXAMPLE_SCORES, e, w.values())],
        floatfmt=".2f",
    )
    say(f"Total of e^score = {e.sum():.2f}. New 'it' = 0.63·V(animal) + 0.23·V(tired) + 0.14·V(street).")
    takeaway("The model resolved the pronoun: most of 'it' now comes from 'animal'.")

    banner("2. Full attention on a tiny sequence (shapes and causal mask)")
    rng = np.random.default_rng(0)
    n, d_k = 4, 8
    Q, K, V = (rng.standard_normal((n, d_k)) for _ in range(3))
    out, weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask(n))
    say(f"Q, K, V are each ({n}, {d_k}). Scores Q·Kᵀ are ({n}, {n}): every token vs. every token.")
    matrix("causal attention weights (row = query token, col = key token)", weights)
    say(
        """
        Upper triangle is exactly zero: token 0 sees only itself, token 3 sees
        all four. Every row sums to 1. Output shape is (4, 8): one new
        context-aware vector per token.
        """
    )

    banner("3. Why divide by sqrt(d_k)? Measure it.")
    rows = sqrt_dk_experiment()
    table(
        ["d_k", "scaled?", "score std", "mean max weight", "softmax grad norm"],
        [(r["d_k"], "yes" if r["scaled"] else "no", r["score_std"], r["max_weight"], r["grad_norm"]) for r in rows],
        floatfmt=".3f",
    )
    say(
        """
        Unscaled, the score std grows like sqrt(d_k) (2, 8, ~22.6). At d_k=512
        the softmax is essentially one-hot (max weight near 1) and the gradient
        norm collapses toward 0. Scaled, std stays ~1 and gradients stay
        healthy at every width.
        """
    )
    takeaway(
        "Dot products grow with dimension; big scores saturate softmax and kill "
        "gradients. Dividing by sqrt(d_k) keeps the variance at 1 so training stays stable."
    )

    banner("4. Multi-head vs. grouped-query attention")
    X = rng.standard_normal((6, 64))
    mha = MultiHeadAttention(d_model=64, n_heads=8)
    gqa = MultiHeadAttention(d_model=64, n_heads=8, n_kv_heads=2)
    mqa = MultiHeadAttention(d_model=64, n_heads=8, n_kv_heads=1)
    for name, layer in [("MHA (8 KV heads)", mha), ("GQA (2 KV heads)", gqa), ("MQA (1 KV head)", mqa)]:
        y, wts = layer(X)
        print(f"{name:18s} out {y.shape}, weights {wts.shape}, K+V params {layer.kv_params():5d}")
    print()
    say(
        """
        All three produce the same output shape. GQA and MQA keep 8 query heads
        but store 4x and 8x less K/V. The KV cache (what you keep in GPU memory
        per token while generating) shrinks by the same factor.
        """
    )

    banner("5. Why long context is expensive: O(n²)")
    table(
        ["tokens n", "score-matrix entries", "quadratic FLOPs", "linear FLOPs"],
        [
            (c["n"], f"{c['score_matrix_entries']:,}", f"{c['quadratic_flops']:.2e}", f"{c['linear_flops']:.2e}")
            for c in (attention_cost(n, 4096) for n in (1_000, 2_000, 8_000, 32_000, 128_000))
        ],
    )
    say(
        """
        Doubling n from 1k to 2k quadruples the score matrix. With d_model=4096
        the quadratic term overtakes the projections once n > 2·d_model.
        Mitigations: FlashAttention (tiles; never materializes n×n in slow
        memory), sliding-window/sparse attention, GQA for the KV cache, and
        prompt caching to avoid recomputing stable prefixes.
        """
    )


if __name__ == "__main__":
    demo()
