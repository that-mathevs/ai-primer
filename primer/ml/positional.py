r"""
# Positional information: how a transformer knows word order

Run: `python -m primer.ml.positional`

New to the notation (vectors, dot products, sine and cosine)? Every symbol
is decoded where it appears, and `primer.notation` teaches them all from zero.

## The problem: attention reads a bag, not a sentence

**Everyday picture.** Pour the words of a sentence into a bag of Scrabble
tiles. The bag for "dog bites man" and the bag for "man bites dog" hold
exactly the same tiles. Anyone who only gets the bag cannot tell a news
story from a routine dog attack. Attention, on its own, only ever gets the bag.

**Tiny worked example.** Give each word a 2-number embedding: dog = (1, 0),
bites = (0, 1), man = (1, 1). Average the words of each sentence, which is
the simplest possible "sentence vector":

| sentence | sum of vectors | average |
|---|---|---|
| dog bites man | (1,0)+(0,1)+(1,1) = (2, 2) | (0.67, 0.67) |
| man bites dog | (1,1)+(0,1)+(1,0) = (2, 2) | (0.67, 0.67) |

Addition doesn't care about order, and neither does attention: every token
compares itself with every other by dot product, and nothing in that
computation says who came first.

```mermaid
flowchart LR
  A["'dog bites man'"] --> E1[Embed each token]
  B["'man bites dog'"] --> E2[Embed each token]
  E1 --> S1["Set {dog, bites, man}"]
  E2 --> S2["Set {man, bites, dog}"]
  S1 --> ATT[Attention with no positions]
  S2 --> ATT
  ATT --> SAME[Same vectors, only reordered:<br/>the meaning difference is lost]
```

**Reading it:** follow both sentences left to right. Embedding looks up each
token independently, so both sentences produce the same three vectors.
Attention only compares vectors with each other; it has no notion of slot 1
versus slot 3. It sees the same *set* and produces the same output vectors,
listed in a different order. Pool them into one sentence vector and the two
sentences are identical.

**The math.** A **permutation** is a reshuffle of an ordered list. Written as
a matrix (a grid of numbers), a permutation matrix is all zeros except one 1
in each row, and multiplying by it just reorders rows. Attention without
positions is *permutation-equivariant*:

$$
\text{Attn}(PX) = P\,\text{Attn}(X)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $X$ | the sentence's token vectors, one row per word | rows dog (1,0), bites (0,1), man (1,1): a 3 × 2 grid |
| $P$ | a permutation matrix: reorders the rows of whatever it multiplies | swap rows 1 and 3 |
| $PX$ | the same rows in the new order | man (1,1), bites (0,1), dog (1,0) |
| $\text{Attn}(\cdot)$ | one attention layer with no position information (here with $W_q = W_k = W_v$ = identity, so q = k = v = the word vector) | 3 × 2 in, 3 × 2 out |
| $=$ | both sides are exactly the same grid of numbers | |

**In words:** "running attention on the shuffled sentence gives the same
answer as running it on the original sentence and shuffling afterwards."

**With the numbers:** in "dog bites man", dog's query (1,0) scores the three
keys (1,0), (0,1), (1,1) as 1, 0, 1; divided by √2 that is 0.707, 0, 0.707.
Softmax gives weights 0.401, 0.198, 0.401, so dog's output is
0.401·(1,0) + 0.198·(0,1) + 0.401·(1,1) = **(0.802, 0.599)**. In "man bites
dog" dog sits in row 3, but it scores the same three keys in a different
order, gets the same weights, and outputs the same **(0.802, 0.599)**.

Shuffle the input and you get the same outputs, shuffled the same way.
`encode_sentence(..., scheme="none")` runs a real attention layer and shows
the "dog" row is the same vector whether "dog" comes first or last.

![Pooled sentence similarity under each position scheme](figures/primer.ml.positional.order_blindness.svg)

**Reading it:** each bar compares the pooled attention output of "dog bites
man" with that of "man bites dog", as a cosine similarity (1.0 means
identical). With no positions the bar reaches exactly 1: the model literally
cannot tell the sentences apart. Adding sinusoidal codes or RoPE pulls the
bar below 1: the sentences now look different. The causal-mask bar is below 1
too, because a mask that lets token i see only i+1 tokens already leaks some
order.

**Why it matters.** Almost all meaning in language is carried by order:
negation scope, who did what to whom, code. So position must be **injected**.
There are three families, below.

## 1. Sinusoidal codes (the original 2017 transformer)

**Everyday picture.** A clock with many hands. The second hand moves fast
and tells nearby moments apart; the hour hand moves slowly and tells morning
from evening. Read all the hands together and every moment has a unique
fingerprint. Sinusoidal codes give every position such a set of "hands" and
add them to the token's embedding.

**Tiny worked example.** With 4 dimensions there are two hands. The fast one
turns 1 radian per position; the slow one turns 1/100 of a radian:

| position | sin(pos·1) | cos(pos·1) | sin(pos/100) | cos(pos/100) |
|---|---|---|---|---|
| 0 | 0.000 | 1.000 | 0.000 | 1.000 |
| 1 | 0.841 | 0.540 | 0.010 | 1.000 |
| 2 | 0.909 | −0.416 | 0.020 | 1.000 |

The fast pair already looks very different at positions 1 and 2; the slow
pair barely moves, and it will only distinguish positions hundreds apart.

```mermaid
flowchart LR
  T["token id"] --> E["embedding lookup<br/>(what the word means)"]
  P["position 0,1,2..."] --> S["sin/cos at many frequencies<br/>(where the word is)"]
  E --> ADD(("+"))
  S --> ADD
  ADD --> B["transformer blocks"]
```

**Reading it:** two independent lookups meet at the plus sign. The top path
says *what* the token is, the bottom path says *where* it is, and their sum
is a single vector carrying both. Everything downstream sees only that sum,
so position becomes part of the content attention compares.

**The math and the code.** Two functions do the work. Picture a point
walking around a circle of radius 1. The angle it has turned is measured in
**radians** (a full turn is 2π ≈ 6.28 radians). **Cosine** of the angle is
how far right the point is, and **sine** is how far up; both always lie
between −1 and +1, and both repeat every full turn.

$$
PE_{(pos,\,2i)} = \sin(pos\cdot\omega_i), \qquad PE_{(pos,\,2i+1)} = \cos(pos\cdot\omega_i),
\qquad \omega_i = \frac{1}{10000^{2i/d}}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $pos$ | the token's position, counting from 0 | 1 |
| $d$ | how many numbers each position code has | 4 |
| $i$ | which (sin, cos) pair, counting from 0 up to $d/2 - 1$ | 0 and 1 |
| $2i$, $2i+1$ | the two columns that pair $i$ fills: even column gets sine, odd gets cosine | pair 1 fills columns 2 and 3 |
| $\omega_i$ | pair $i$'s speed in radians per position (its **frequency**) | $\omega_0 = 1$, $\omega_1 = 1/100$ |
| $10000^{2i/d}$ | 10,000 raised to a fraction; makes each pair slower than the last | $10000^{2/4} = 100$ |
| $\sin$, $\cos$ | how far up / right a point is after turning by that many radians | $\sin 1 = 0.841$ |
| $PE_{(pos,\,c)}$ | the number in row $pos$, column $c$ of the code table | |

**In words:** "for position *pos*, each pair of columns stores where a hand
turning at that pair's speed points after *pos* steps: sine in the first
column, cosine in the second."

**With the numbers:** d = 4, pos = 1. Pair 0 turns 1 radian: sin 1 = 0.841,
cos 1 = 0.540. Pair 1 turns 1/100 radian: sin 0.01 = 0.010, cos 0.01 = 1.000.
So position 1's code is **(0.841, 0.540, 0.010, 1.000)**, the second row of
the table above.

`sinusoidal_encoding(n, d)` builds the whole (n × d) table in four lines. A
shift of k positions rotates every (sin, cos) pair by the same angle k·ω_i
wherever you start, so **the dot product of two position codes depends only
on their distance**.

![Sinusoidal encoding heatmap](figures/primer.ml.positional.sinusoidal_heatmap.svg)

**Reading it:** each row is a position (0 at the top), each column one
dimension, and colour is the value from −1 to +1. The left columns flip
colour every few rows (fast hands); the right columns change slowly over
hundreds of rows (slow hands). No two rows are identical, so every position
has a unique fingerprint, and neighbouring rows look alike, so nearby
positions get similar codes.

**Why it matters.** Fixed codes need no training and exist for any position,
and the distance property lets the model learn "look 3 tokens back" once and
use it everywhere. But the code is mixed into the content vector itself,
which later methods avoid.

## 2. Learned absolute positions (GPT-2, BERT)

**Everyday picture.** Numbered theatre seats. Every seat has its own brass
plate, and the theatre has a fixed number of seats.

**Tiny worked example.** GPT-2 has a table of 1,024 position vectors, each
768 numbers long: 786,432 extra parameters, learned like any other weight.
Position 5 looks up row 5. Position 1,024 has no row at all.

```mermaid
flowchart LR
  T["token id 318"] --> TE["token table<br/>(vocab × d)"]
  P["position 5"] --> PE["position table<br/>(max_len × d)"]
  TE --> ADD(("+"))
  PE --> ADD
  ADD --> B["transformer blocks"]
  P2["position ≥ max_len"] -.-> X["no row: cannot be encoded"]
```

**Reading it:** identical to the sinusoidal picture except that the bottom
path is a second trainable table instead of a formula. The dotted arrow is
the catch: the table ends at max_len, so a longer input simply has nowhere
to look.

**The math and the code.**

$$
x_{pos} = E_{token} + P_{pos}
$$

**Symbols**

| Symbol | Meaning here | In the example (GPT-2) |
|---|---|---|
| $E_{token}$ | the token's row of the token table (what the word means) | 768 numbers |
| $P_{pos}$ | row $pos$ of the position table, learned during training (where it is) | 768 numbers; $pos$ from 0 to 1,023 |
| $+$ | add the two lists number by number | |
| $x_{pos}$ | the vector the first transformer block receives | 768 numbers |

**In words:** "the input vector is the word's meaning plus its seat number's
learned vector."

**With the numbers:** with 2 dimensions, if $E_{dog}$ = (0.5, −0.2) and
$P_{3}$ = (0.1, 0.3), then $x_3$ = (0.6, 0.1). $P_{1024}$ does not exist.

`primer.ml.transformer.TinyGPT` uses exactly this.

**Why it matters.** It's simple and works well inside the trained length,
but it cannot extrapolate. That limitation pushed the field to RoPE.

## 3. Rotary position embeddings, RoPE (Llama, Mistral, Qwen and most modern LLMs)

**Everyday picture.** Two people point at a clock face, one at 1 o'clock and
one at 4 o'clock. The angle between their arms is 90°. Move both of them on
to 8 and 11 o'clock and the angle is still 90°. RoPE turns each token's
query and key like clock hands, by an amount proportional to its position.
When a query meets a key, only the **angle between them** (the distance
between the tokens) affects the score, not where on the clock they started.

**Tiny worked example.** Use 2 dimensions, so there is one pair turning 1
radian per position. Let q = k = (1, 0). Put the query at position 3 and the
key at position 7:

* q turns by 3 rad: (cos 3, sin 3) = (−0.990, 0.141)
* k turns by 7 rad: (cos 7, sin 7) = (0.754, 0.657)
* score = −0.990·0.754 + 0.141·0.657 = −0.654 = cos(4)

Now move both 100 positions later, to 103 and 107. The score is still
cos(107 − 103) = cos 4 = −0.654. Only the distance, 4, survived.

```mermaid
flowchart LR
  X[Token vector at position m] --> Q[q = x W_q]
  X --> K[k = x W_k]
  X --> V[v = x W_v]
  Q --> RQ["Rotate each pair of q<br/>by m·θ_i"]
  K --> RK["Rotate each pair of k<br/>by m·θ_i"]
  RQ --> S["Score = q'·k'<br/>depends only on distance"]
  RK --> S
  S --> SM[Softmax] --> W[Weighted sum of V<br/>V is NOT rotated]
  V --> W
```

**Reading it:** this is ordinary attention with one extra step on the query
and key branches. After projection, each (even, odd) pair of numbers in q
and k is spun like a clock hand by an angle that grows with the token's
position: fast for the first pairs, slow for the last. In the dot product
only the *difference* of the angles survives, so the score knows how far
apart the tokens are but not where they sit in the document. The value
branch is untouched, so what gets blended is plain content, and nothing is
ever added to the embedding.

**The math and the code.** A **rotation** turns a point around the centre
by some angle without changing its distance from the centre. Written as a
2 × 2 grid times a column of two numbers (a **matrix-vector multiply**:
each output number is one row of the grid multiplied position-by-position
with the input and added up):

$$
\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} =
\begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix},
\qquad \theta_i = 10000^{-2i/d}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $x_{2i}, x_{2i+1}$ | pair $i$ of a query or key vector, before rotating | (1, 0) |
| $x'_{2i}, x'_{2i+1}$ | the same pair after rotating | (−0.990, 0.141) |
| $m$ | the token's position | 3 |
| $i$ | which pair, from 0 to $d/2 - 1$ | 0 (the only pair when d = 2) |
| $\theta_i$ | pair $i$'s turning speed in radians per position; $10000^{-2i/d}$ means $1/10000^{2i/d}$ | $\theta_0 = 1$ |
| $m\theta_i$ | the total angle turned | 3 radians |
| the 2 × 2 grid | the rotation: row 1 gives the new first number, row 2 the new second | |

**In words:** "new first number = cos(angle) × old first − sin(angle) × old
second; new second number = sin(angle) × old first + cos(angle) × old second,
where the angle is position times the pair's speed."

**With the numbers:** x = (1, 0) at m = 3, θ = 1: x′ = (cos 3 · 1 − sin 3 · 0,
sin 3 · 1 + cos 3 · 0) = (−0.990, 0.141).

Why only the distance survives: turning q by angle a and k by angle b and
then taking their **dot product** (multiply matching numbers, add them up;
large when the vectors point the same way) gives the same answer as turning
k alone by b − a. So

$$
\langle R_m q,\; R_n k\rangle = \langle q,\; R_{n-m}\, k\rangle
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $q$, $k$ | a query and a key vector, before rotating | (1, 0) and (1, 0) |
| $m$, $n$ | the query's and the key's positions | 3 and 7 |
| $R_m$ | "rotate every pair by its speed times $m$" | turn by 3 radians |
| $R_{n-m}$ | the rotation for the *distance* only | turn by 4 radians |
| $\langle a, b \rangle$ | the dot product of $a$ and $b$ (same as $a \cdot b$) | |

**In words:** "the score between a query at *m* and a key at *n* equals the
score you'd get by leaving the query alone and turning the key by the gap
*n − m*."

**With the numbers:** left side (−0.990, 0.141) · (0.754, 0.657) = −0.654;
right side (1, 0) · (cos 4, sin 4) = cos 4 = −0.654. Same number, and the same
again at positions 103 and 107.

`apply_rope` does this for a vector or a whole sequence with three
element-wise lines, and no matrix multiply.

![RoPE score vs. distance](figures/primer.ml.positional.rope_relative.svg)

**Reading it:** the x-axis is the distance n − m between a query and a key;
each line uses the same q and k but places the pair at a different absolute
starting position (0, 100, 1,000). The lines lie exactly on top of each
other: the score depends only on distance. That is the relative-position
property in one picture.

![RoPE rotation angle per pair](figures/primer.ml.positional.rope_frequencies.svg)

**Reading it:** each line is one pair of dimensions; the y-axis is how far
that pair has turned (wrapped to one full turn, 2π) at each position on the
x-axis. Pair 0 spins fastest and wraps every ~6 tokens, giving fine-grained
local order. The last pairs barely turn across the whole range, giving
coarse long-range position. It's the many-handed clock again, now applied
inside attention.

**Why it matters.** Relative distance is what language actually uses ("the
word two back"), rotation keeps vector lengths unchanged, and because
position lives in angles, you can stretch it after training. That last
point is the next section.

## 4. Extending context after training

**Everyday picture.** A ruler marked from 0 to 4,000. To measure something
32,000 long you can shrink the object by 8× so it fits on the ruler
(interpolation), or re-draw only the long-distance marks while keeping the
fine millimetre marks (NTK-aware scaling).

**Tiny worked example.** A model trained on 4,096 tokens now reads 32,768.
Interpolation multiplies every position by 4096/32768 = 1/8, so position
32,000 is treated as 4,000: an angle the model has seen. Neighbours are now
1/8 of a position apart, which a short fine-tune teaches it to resolve.

```mermaid
flowchart LR
  P["positions 0 ... 32k"] --> C{method}
  C -->|interpolation| PI["pos × 4k/32k<br/>all hands slowed 8×"]
  C -->|NTK-aware| NTK["raise the RoPE base<br/>slow hands slowed, fast hands kept"]
  PI --> R["RoPE with angles inside<br/>the trained range"]
  NTK --> R
  R --> F["short fine-tune on long text"]
```

**Reading it:** both paths solve the same problem: angles beyond what the
model saw in training. Interpolation slows *every* hand equally, which also
blurs the fast hands that encode local word order. NTK-aware scaling slows
only the slow hands and leaves the fastest one untouched, so local order stays
sharp. Both usually end with a brief fine-tune.

**The math and the code.** Position interpolation (`interpolate_positions`):

$$
pos' = pos \cdot \frac{L_{train}}{L_{new}}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $pos$ | the real position in the long input | 32,000 |
| $L_{train}$ | the longest context seen in training | 4,096 |
| $L_{new}$ | the context we now want | 32,768 |
| $pos'$ | the position RoPE is actually given | 4,000 |

**In words:** "shrink every position by the ratio of old length to new length."

**With the numbers:** 32,000 × 4,096 / 32,768 = 32,000 / 8 = **4,000**.

NTK-aware scaling (`ntk_scaled_base`) changes the base instead:

$$
base' = base \cdot s^{\,d/(d-2)}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $base$ | the 10,000 inside $\theta_i = base^{-2i/d}$ | 10,000 |
| $s$ | how many times longer the new context is | 8 |
| $d$ | the head width | 128 |
| $s^{d/(d-2)}$ | $s$ raised to a power just above 1 | $8^{1.016} ≈ 8.27$ |
| $base'$ | the new, larger base | ≈ 82,685 |

**In words:** "multiply the base by slightly more than the stretch factor."

**With the numbers:** the fastest pair keeps $\theta_0 = base'^{0} = 1$
radian per position. The slowest pair goes from $10000^{-126/128} = 1.15
\times 10^{-4}$ to $82685^{-126/128} = 1.44 \times 10^{-5}$: exactly 8× slower.

YaRN refines this per frequency band and is
used by many long-context models. ALiBi skips position vectors entirely and
subtracts a penalty proportional to distance from each attention score.

![Slowest RoPE angle: trained, extended and interpolated](figures/primer.ml.positional.context_extension.svg)

**Reading it:** the x-axis is position up to 32k and the y-axis is the angle
of the slowest RoPE pair. The shaded band is the range of angles the model
saw in training (positions up to 4k). The raw extended line leaves the band
almost immediately after 4k: unfamiliar territory. The interpolated and
NTK-scaled lines stay inside the band all the way to 32k.

**Why it matters.** It's how models trained on a few thousand tokens are
cheaply stretched to 100k or more, instead of being pretrained again from
scratch.

## A subtlety: the causal mask leaks order

In a decoder, token i sees exactly i+1 tokens, so even with no position
encoding ("NoPE") the model can partly infer where it is. Bidirectional
attention (BERT-style) has no such crutch and is fully order-blind.

## In 20 seconds
- Attention is permutation-invariant: without position information, "dog
  bites man" equals "man bites dog".
- The original transformer adds fixed sine/cosine codes; GPT-2 and BERT
  learn a position table that ends at max_len.
- Modern LLMs use RoPE: rotate q and k by a position-dependent angle so the
  score depends only on relative distance. It also makes context extension
  (interpolation, NTK/YaRN) practical.

## Self-test questions

**Why does a transformer need positional information at all?**
Attention compares tokens by content only, so it's permutation-equivariant:
shuffling the input just shuffles the output. Without positions, word order
is invisible.

**How does RoPE encode position, and what property makes it attractive?**
It rotates each pair of dimensions in q and k by an angle proportional to
position. The q·k score then depends only on the relative distance, and
rotation preserves vector length.

**Compute a RoPE score by hand: 2 dimensions, θ = 1, q = k = (1, 0), query at 3, key at 7.**
cos(7 − 3) = cos 4 ≈ −0.654, and the same at positions 103 and 107.

**Why do sinusoidal codes use many frequencies?**
Fast frequencies distinguish neighbours; slow ones distinguish distant
positions. Together every position gets a unique, smoothly varying code, and
a shift by k is the same rotation everywhere.

**What goes wrong beyond the trained context length, and how do you fix it?**
The model meets position angles or indices it never saw, and attention
degrades. Position interpolation, NTK-aware scaling or YaRN rescale positions
or frequencies into the trained range, usually followed by a short fine-tune.

**Learned absolute position embeddings: pro and con?**
They are simple and effective within the trained length, but they cannot
represent any position beyond the table size.

## The papers behind this lesson

- **Vaswani et al. (2017), *Attention Is All You Need*.** https://arxiv.org/abs/1706.03762.
  Introduced the transformer and, in section 3.5, the sinusoidal position
  codes built here. [annotated companion](../../papers/attention-is-all-you-need.html)
- **Su et al. (2021), *RoFormer: Enhanced Transformer with Rotary Position Embedding*.**
  https://arxiv.org/abs/2104.09864. Introduced RoPE: rotate queries and keys
  so attention scores depend only on relative distance.
  [annotated companion](../../papers/roformer.html)
- **Chen et al. (2023), *Extending Context Window of Large Language Models via Positional Interpolation*.**
  https://arxiv.org/abs/2306.15595. Showed that scaling positions down
  plus a short fine-tune extends RoPE models to much longer contexts.
- **Peng et al. (2023), *YaRN*.** https://arxiv.org/abs/2309.00071. Refined
  RoPE context extension by treating fast and slow frequencies differently.
- **Press et al. (2021), *ALiBi*.** https://arxiv.org/abs/2108.12409.
  Replaced position vectors with a distance penalty on attention scores that
  extrapolates to longer inputs.

## Further reading
- Vaswani et al., *Attention Is All You Need* (sinusoidal codes, section 3.5): https://arxiv.org/abs/1706.03762
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021): https://arxiv.org/abs/2104.09864
- EleutherAI, *Rotary Embeddings: A Relative Revolution*: https://blog.eleuther.ai/rotary-embeddings/
- Chen et al., *Extending Context Window of Large Language Models via Positional Interpolation* (2023): https://arxiv.org/abs/2306.15595
- Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models* (2023): https://arxiv.org/abs/2309.00071
- Press et al., *Train Short, Test Long: Attention with Linear Biases (ALiBi)* (2021): https://arxiv.org/abs/2108.12409
- Kazemnejad et al., *The Impact of Positional Encoding on Length Generalization in Transformers* (NoPE, 2023): https://arxiv.org/abs/2305.19466
"""

from __future__ import annotations

import numpy as np

import hashlib

from primer._show import banner, matrix, say, table, takeaway
from primer.ml.attention import causal_mask, scaled_dot_product_attention

# ---------------------------------------------------------------------------
# 1. Sinusoidal encodings
# ---------------------------------------------------------------------------


def sinusoidal_encoding(n_positions: int, d: int, base: float = 10000.0) -> np.ndarray:
    """(n_positions, d) matrix of the original transformer's position codes.

    Column 2i holds sin(pos · ω_i), column 2i+1 holds cos(pos · ω_i), with
    ω_i = 1 / base^(2i/d). Frequencies fall geometrically from 1 (a full cycle
    every ~6 tokens) to 1/base (one cycle every ~60,000 tokens).
    """
    assert d % 2 == 0, "dimensions come in (sin, cos) pairs"
    pos = np.arange(n_positions)[:, None]  # (n, 1)
    omega = base ** (-np.arange(0, d, 2) / d)  # (d/2,) frequencies
    angles = pos * omega  # (n, d/2) via broadcasting
    pe = np.empty((n_positions, d))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe


# ---------------------------------------------------------------------------
# 2. RoPE: rotary position embeddings
# ---------------------------------------------------------------------------


def rope_frequencies(d: int, base: float = 10000.0) -> np.ndarray:
    """θ_i = base^(-2i/d) for each of the d/2 pairs: pair 0 turns fastest."""
    assert d % 2 == 0, "RoPE rotates dimensions in pairs"
    return base ** (-np.arange(0, d, 2) / d)


def apply_rope(x: np.ndarray, position: int | np.ndarray | None = None, base: float = 10000.0) -> np.ndarray:
    """Rotate each (even, odd) pair of `x` by position · θ_i.

    Args:
        x: one vector (d,) or a sequence (seq, d). In a real model this is
            applied to q and k (per head) right before the dot product.
        position: the token's position (int) or one per row. Defaults to
            0..seq-1 for a sequence.
        base: 10000 in the RoPE paper; raising it (NTK-aware scaling) slows
            every rotation, one way to stretch a model to longer contexts.
    """
    x = np.asarray(x, dtype=float)
    if position is None:
        position = np.arange(x.shape[0]) if x.ndim == 2 else 0
    pos = np.asarray(position, dtype=float)
    # angles: (d/2,) for one vector, or (seq, d/2) for a sequence.
    angles = pos[..., None] * rope_frequencies(x.shape[-1], base)
    cos, sin = np.cos(angles), np.sin(angles)
    even, odd = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    # The 2x2 rotation [[cos, -sin], [sin, cos]] applied to every pair at once.
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out


# ---------------------------------------------------------------------------
# 3. Order blindness: attention without positions sees a set, not a sequence
# ---------------------------------------------------------------------------

D_DEMO = 16
_rng = np.random.default_rng(7)
# One fixed single-head attention layer for the demos (random, untrained).
_W_Q, _W_K, _W_V = (_rng.normal(0, 1 / np.sqrt(D_DEMO), (D_DEMO, D_DEMO)) for _ in range(3))


def word_vector(word: str, d: int = D_DEMO) -> np.ndarray:
    """A fixed random embedding per word (a stand-in for a learned embedding table)."""
    seed = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)  # stable across runs, unlike hash()
    return np.random.default_rng(seed).standard_normal(d)


def encode_sentence(sentence: str, scheme: str = "none", causal: bool = False) -> np.ndarray:
    """Run one attention layer over `sentence` with a given position scheme.

    scheme:
        "none":       embeddings only; attention sees an unordered set.
        "sinusoidal": add the sinusoidal code to each embedding (2017 transformer).
        "rope":       rotate q and k by position inside attention (modern LLMs).
    Returns the (n_words, d) output vectors.
    """
    X = np.stack([word_vector(w) for w in sentence.split()])  # (n, d)
    n = X.shape[0]
    if scheme == "sinusoidal":
        X = X + sinusoidal_encoding(n, D_DEMO)
    Q, K, V = X @ _W_Q, X @ _W_K, X @ _W_V
    if scheme == "rope":
        Q, K = apply_rope(Q), apply_rope(K)  # only q and k are rotated; v carries plain content
    out, _ = scaled_dot_product_attention(Q, K, V, mask=causal_mask(n) if causal else None)
    return out


# ---------------------------------------------------------------------------
# 4. Context extension: squeeze long contexts back into the trained range
# ---------------------------------------------------------------------------


def interpolate_positions(positions: np.ndarray, trained_len: int, new_len: int) -> np.ndarray:
    """Position interpolation: scale positions by trained_len / new_len.

    A 4k-trained model run at 32k sees position 32,000 as 4,000: an angle it
    knows. The price is that neighbouring tokens are now only 1/8 of a
    position apart, which a short fine-tune teaches the model to resolve.
    """
    return np.asarray(positions, dtype=float) * (trained_len / new_len)


def ntk_scaled_base(base: float, scale: float, d: int) -> float:
    """NTK-aware RoPE scaling: raise the base instead of squeezing positions.

    base' = base · scale^(d/(d-2)). The fastest pair (θ_0 = 1) is unchanged,
    so local word order stays sharp, while the slowest pair slows by exactly
    `scale`, so distant positions fit into the trained range of angles.
    """
    return base * scale ** (d / (d - 2))


# ---------------------------------------------------------------------------
# 5. Figures (rendered to docs/figures by `make figures`)
# ---------------------------------------------------------------------------


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def order_similarity() -> dict[str, float]:
    """Cosine similarity of pooled 'dog bites man' vs 'man bites dog' per scheme."""
    out = {}
    for label, scheme, causal in [
        ("no positions", "none", False),
        ("sinusoidal", "sinusoidal", False),
        ("RoPE", "rope", False),
        ("causal mask only", "none", True),
    ]:
        a = encode_sentence("dog bites man", scheme, causal).mean(axis=0)
        b = encode_sentence("man bites dog", scheme, causal).mean(axis=0)
        out[label] = _cos(a, b)
    return out


def figures() -> dict:
    """Plot this lesson's data. matplotlib is imported here, and only here."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, RED, MUTED = "#2563eb", "#dc2626", "#9ca3af"
    figs = {}

    sims = order_similarity()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    colors = [RED] + [BLUE] * (len(sims) - 1)
    ax.bar(list(sims), list(sims.values()), color=colors)
    for i, v in enumerate(sims.values()):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    ax.set_ylim(min(sims.values()) - 0.1, 1.05)
    ax.set_ylabel("cosine similarity of pooled outputs")
    ax.set_title("'dog bites man' vs 'man bites dog'")
    fig.tight_layout()
    figs["order_blindness"] = fig

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(sinusoidal_encoding(200, 64), aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xlabel("dimension (left: fast hands, right: slow hands)")
    ax.set_ylabel("position")
    ax.set_title("Sinusoidal position codes")
    fig.colorbar(im, ax=ax, label="value")
    fig.tight_layout()
    figs["sinusoidal_heatmap"] = fig

    rng = np.random.default_rng(0)
    q, k = rng.standard_normal(64), rng.standard_normal(64)
    dist = np.arange(0, 60)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for start, style in ((0, "-"), (100, "--"), (1000, ":")):
        scores = [apply_rope(q, start) @ apply_rope(k, start + d) for d in dist]
        ax.plot(dist, scores, style, lw=2, label=f"query at position {start}")
    ax.set_xlabel("distance between key and query (n − m)")
    ax.set_ylabel("attention score q·k")
    ax.set_title("With RoPE the score depends only on distance")
    ax.legend(frameon=False)
    fig.tight_layout()
    figs["rope_relative"] = fig

    pos = np.arange(0, 200)
    theta = rope_frequencies(64)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i in (0, 4, 12, 31):
        ax.plot(pos, np.mod(pos * theta[i], 2 * np.pi), lw=1.5, label=f"pair {i} (θ={theta[i]:.2g})")
    ax.set_xlabel("position")
    ax.set_ylabel("rotation angle (mod 2π)")
    ax.set_title("RoPE: fast hands for local order, slow hands for long range")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    figs["rope_frequencies"] = fig

    d, trained, new = 128, 4096, 32768
    positions = np.arange(0, new, 64)
    slow = rope_frequencies(d)[-1]
    slow_ntk = rope_frequencies(d, ntk_scaled_base(10000, new / trained, d))[-1]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.axhspan(0, trained * slow, color=MUTED, alpha=0.3, label="angles seen in training")
    ax.plot(positions, positions * slow, color=RED, label="raw positions (extended)")
    ax.plot(positions, interpolate_positions(positions, trained, new) * slow, color=BLUE, label="position interpolation")
    ax.plot(positions, positions * slow_ntk, "--", color="black", label="NTK-aware base")
    ax.set_xlabel("position")
    ax.set_ylabel("angle of the slowest RoPE pair (rad)")
    ax.set_title("Keeping long contexts inside the trained range")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    figs["context_extension"] = fig
    return figs


# ---------------------------------------------------------------------------
# 6. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. Attention without positions reads a bag of words")
    a = encode_sentence("dog bites man", "none")
    b = encode_sentence("man bites dog", "none")
    say(
        f"""
        One attention layer, no position information. The output for 'dog' is
        the same vector whether it comes first or last (max difference
        {np.abs(a[0] - b[2]).max():.1e}), and the pooled sentence vectors are
        identical (cosine {_cos(a.mean(0), b.mean(0)):.3f}).
        """
    )
    table(["scheme", "cosine(dog bites man, man bites dog)"], list(order_similarity().items()), floatfmt=".3f")
    takeaway("Without positions, attention sees a set: shuffle the input and the outputs just shuffle.")

    banner("2. Sinusoidal codes: a clock with many hands")
    matrix("positions 0..2, d=4 (fast pair, then slow pair)", sinusoidal_encoding(3, 4))
    pe = sinusoidal_encoding(100, 32)
    say(
        f"""
        Similarity depends only on distance: PE(10)·PE(13) = {pe[10] @ pe[13]:.4f}
        and PE(70)·PE(73) = {pe[70] @ pe[73]:.4f}.
        """
    )

    banner("3. RoPE by hand: q = k = (1, 0), one pair turning 1 rad per position")
    q = k = np.array([1.0, 0.0])
    rows = [(m, n, apply_rope(q, m) @ apply_rope(k, n)) for m, n in ((3, 7), (103, 107), (3, 8))]
    table(["query pos", "key pos", "score"], rows, floatfmt=".3f")
    say("Positions 3/7 and 103/107 give the same score, cos 4 = -0.654; moving the key one step changes it.")
    takeaway("RoPE rotates q and k by position, so attention scores depend only on relative distance.")

    banner("4. Stretching a 4k model to 32k")
    say(
        f"""
        Position interpolation maps 32,000 to
        {interpolate_positions(np.array([32000]), 4096, 32768)[0]:.0f}. NTK-aware
        scaling raises the RoPE base from 10,000 to
        {ntk_scaled_base(10000, 8, 128):,.0f} for d=128: the fastest pair still turns
        1 rad per token, the slowest turns 8x slower.
        """
    )


if __name__ == "__main__":
    demo()
