r"""
# Positional information: how a transformer knows word order

Run: `python -m primer.ml.positional`

New to the notation (vectors, dot products, sine and cosine)? Every symbol
is decoded where it appears, and `primer.notation` teaches them all from zero.
This lesson builds on `primer.ml.attention`.

## Level 1: The practitioner's guide

**In one sentence.** Positional encoding is how a transformer learns where
each token sits, since attention on its own treats the input as an unordered
bag; the scheme a model uses decides how long a context it can read, whether
that length can be stretched after training, and what happens when you go
past it.

**When you need it.** You never add positions yourself; the model's authors
chose a scheme before training and it is baked into the weights. You meet it
the day a length matters: a model card says 128K tokens and you wonder how
far to trust it, a self-hosted model produces nonsense past a certain prompt
length, an embedding model rejects documents longer than its limit, or you
want to fine-tune a model to read documents longer than it was trained on.
The tell: quality that is fine at 4,000 tokens and falls off a cliff at
some longer length, with nothing in the logs. The number behind the whole
topic, from this lesson's `order_similarity`: with no positions, the pooled
attention outputs for "dog bites man" and "man bites dog" have cosine
similarity exactly 1.000. The model cannot tell them apart. Nothing about
this concerns you when prompts stay well inside the length the model was
trained at.

**Your options.** The first four are choices the model's authors made; you
choose among them by choosing a model. The last three are what you or the
vendor can do about length afterwards. From the simplest to the most
committed:

| Option | What it does | What it gives you | What it costs | Where it lives |
|---|---|---|---|---|
| A learned position table (GPT-2, BERT) | One learned vector per seat, added to the token's vector | Simple and effective inside the trained length | A hard ceiling: no row exists past the table's end (1,024 rows for GPT-2, 512 for BERT) | The architecture; `max_position_embeddings` on the card |
| Sinusoidal codes (the 2017 transformer) | A fixed sine and cosine fingerprint per position, added to the token | No parameters; a code exists for any position | Position is mixed into content; little used in new models | The architecture |
| RoPE (Llama, Mistral, Qwen) | Rotates each query and key by an angle that grows with position, so scores depend on the distance between tokens | Relative distance for free, and a context that can be stretched after training | Angles past the trained range are unfamiliar; extension needs scaling | The architecture; `rope_theta` and `rope_parameters` on the card |
| ALiBi | Subtracts a penalty proportional to distance from each score; no position vectors at all | Trained at 1,024 tokens, it extrapolates to 2,048; 11% faster and 11% less memory than sinusoidal in its paper | A built-in preference for nearby tokens; fewer models use it | The architecture |
| Inference-time scaling (linear or dynamic NTK) | Rescales positions or the RoPE base: linear scaling at every length, dynamic scaling only once a prompt exceeds the trained length | A modest stretch with no training at all | Quality drops as the stretch grows | The serving engine's config: `rope_type` and `factor` |
| Extension with a short fine-tune (PI, YaRN) | Scales positions into the trained range, then fine-tunes briefly on long text | 8× longer context: Llama to 32,768 tokens within 1,000 steps (PI); YaRN needs 10× fewer tokens than earlier methods | Long documents to train on, a training run, an evaluation at length | Your training stack |
| Staged long-context pretraining | The vendor grows the window during pretraining, checking a needle-in-a-haystack test at each stage | The genuine article: Llama 3 went from 8K to 128K in six stages | About 800B training tokens for Llama 3 405B; reaches you as a number on the card | The vendor |

**How to choose.** Start from the model in front of you.

- A hosted API: nothing to choose, but the window on the card is the length
  the vendor trained and tested, not a promise about your task. Measure at
  the lengths you will use.
- Picking an open model: read `max_position_embeddings` and
  `rope_parameters`. An entry with a `factor` means the model was trained
  shorter and stretched, so test it at the lengths you will use rather
  than trusting the stretched number.
- Serving a model beyond its trained length: do not raise the engine's
  context limit on its own. Set the scaling the checkpoint expects (or a
  dynamic scaling if none is given), and test before shipping.
- Documents longer than the model's window, and you can train: position
  interpolation or YaRN plus a fine-tune on long text, with a
  needle-in-a-haystack check and your own task at the target length.
- An encoder or embedding model with a learned table: the limit is hard.
  Chunk documents to fit (`primer.agents.rag`).
- Whatever you pick, evaluate at the length you will run, with the answer
  placed at the start, the middle and the end of the prompt.

**What it costs.** Positions are nearly free to compute; length is what
costs.

- Parameters and compute. GPT-2's table is 1,024 × 768 = 786,432 numbers,
  under 1% of its 124 million parameters. RoPE and ALiBi add none, and RoPE
  is a few element-wise multiplications per query and key, negligible next
  to attention itself.
- Stretching. Position interpolation for 4,096 to 32,768 tokens multiplies
  every position by 1/8 (32,000 becomes 4,000); NTK-aware scaling instead
  raises the RoPE base from 10,000 to 82,685 for a 128-wide head, leaving
  the fastest pair untouched and slowing the slowest 8×. Both are followed
  by a short fine-tune: within 1,000 steps in the PI paper. Llama 3's
  authors set the base to 500,000 and spent about 800B tokens taking the
  405B model from 8K to 128K.
- Quality. RoPE gives nearby tokens a head start (the RoFormer companion's
  long-term decay), and a long window is not used evenly: Liu et al. (*Lost
  in the Middle*, 2023) found models use information best at the start or
  the end of a long prompt and worst in the middle (`primer.agents.context`).

**What breaks.**

- **Past the table.** A learned-table model has no vector for position
  1,024 if its table has 1,024 rows: the request fails or the library
  truncates. Chunk the input.
- **Past the trained angles.** RoPE degrades silently rather than failing:
  this lesson's figure shows the slowest pair's angle leaving the trained
  band right after 4K tokens and reaching 8× its top by 32K. The PI paper
  reports that plain extrapolation can produce catastrophically large
  attention scores. Scale, then fine-tune.
- **A checkpoint served with the wrong scaling.** An extended model whose
  `rope_parameters` are dropped or mistyped reads positions it never
  learned. Copy the config with the weights.
- **Interpolating too far.** Interpolation slows every frequency equally,
  which also blurs the fast ones that carry local word order. NTK-aware
  scaling and YaRN keep the fast frequencies, which is why they are
  preferred for large stretches.
- **Pooling that forgets order.** Average the token vectors of an
  order-blind model and "dog bites man" equals "man bites dog" exactly.
  Order survives only if positions are injected, or an encoding step that
  is not permutation-blind comes first.
- **Trusting the causal mask for order.** In a decoder the mask alone lets
  the sentences differ (cosine 0.524 in this lesson's run), but it is a
  weak signal, not a position encoding.

**In the wild.** The original transformer (Vaswani et al., 2017) used
sinusoidal codes; GPT-2 and BERT switched to learned tables, which is why
each has a fixed maximum length. Llama, Mistral and Qwen use RoPE, and Llama
3 raised its base to 500,000 with 8 key-value heads beside it. Hugging Face
model configs carry the scheme as `rope_parameters` with a `rope_type` of
`linear`, `dynamic`, `yarn`, `longrope` or `llama3`, a `factor` and an
`original_max_position_embeddings`; vLLM's `--max-model-len` sets the
served length and `--hf-overrides` changes that config at load time. YaRN
(Peng et al., 2023) is the extension recipe behind many long-context
checkpoints, ALiBi (Press et al., 2021) the alternative that extrapolates
without vectors, and the NoPE study (Kazemnejad et al., 2023) found that a
decoder can generalize to longer inputs with no explicit positions at all.
The papers are linked at the end of the lesson.

**Go deeper.** Level 2 shows the bag-of-words problem in three numbers,
builds sinusoidal codes as a clock with many hands, checks RoPE's distance
property by hand at positions 3 and 7 and again at 103 and 107, and stretches
a 4K model to 32K by interpolation and by raising the base. If you only
needed to read a model card or serve a model safely, you are done.

## Level 2: How it works, from scratch

Attention scores each pair of words by comparing their vectors, and the
comparison comes out the same wherever the two words sit. So the order of the
words has to be put in from outside: added to each word's vector, or built
into the way attention compares them.

## The problem: attention reads a bag, not a sentence

Think of the words as Scrabble tiles poured into a bag. The bag for "dog
bites man" and the bag for "man bites dog" hold the same tiles, and anyone
handed only the bag cannot tell a news story from a routine dog attack.
Attention with no position information is handed only the bag.

Give each word an embedding of two numbers: dog = (1, 0), bites = (0, 1),
man = (1, 1). The simplest way to turn a sentence into one vector is to
average its words:

| sentence | sum of vectors | average |
|---|---|---|
| dog bites man | (1,0)+(0,1)+(1,1) = (2, 2) | (0.67, 0.67) |
| man bites dog | (1,1)+(0,1)+(1,0) = (2, 2) | (0.67, 0.67) |

The averages match because addition ignores order. Attention is an average
too, a weighted one, and its weights come from dot products between words,
which ignore order in the same way.

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

**Reading it:** follow both sentences from left to right. Embedding looks up
each word on its own, so both sentences give the same three vectors, and
attention receives the same *set*. Its outputs are the same vectors in a
different order, so pooled into one sentence vector the two sentences are
identical.

A **permutation** is a reshuffle of an ordered list. Written as a matrix (a
grid of numbers), a permutation matrix is all zeros except one 1 in each row,
and multiplying by it reorders rows. Attention without positions is
**permutation-equivariant**: shuffle its input and its output is shuffled the
same way.

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

**In Python:**

```python
import math
# q = k = v = the word vector, no positions
def Attn(X):
    out = []
    for q in X:
        scores = [sum(a * b for a, b in zip(q, k)) / math.sqrt(2) for k in X]
        exps = [math.exp(s) for s in scores]
        weights = [e / sum(exps) for e in exps]
        out.append(tuple(round(sum(w * v[c] for w, v in zip(weights, X)), 3) for c in range(2)))
    return out
# dog, bites, man
X = [(1, 0), (0, 1), (1, 1)]
# P swaps rows 1 and 3: man, bites, dog
PX = [X[2], X[1], X[0]]
Attn(X)  # → [(0.802, 0.599), (0.599, 0.802), (0.752, 0.752)]
# the same rows, shuffled the same way
Attn(PX)  # → [(0.752, 0.752), (0.599, 0.802), (0.802, 0.599)]
```

A larger layer, with 16 random numbers per word and random, untrained
weights, behaves the same way:

![Without positions the two sentences are identical (similarity 1.0); sinusoidal codes and RoPE pull it just below 1, and a causal mask alone drops it to 0.52](figures/primer.ml.positional.order_blindness.svg)

**Reading it:** each bar is the cosine similarity of the two sentences'
pooled outputs, where 1.0 means identical. With no positions the bar reaches
exactly 1, so the layer cannot tell the sentences apart. Sinusoidal codes and
RoPE pull it below 1. So does a causal mask, which lets each word see only
the words before it, with no positions added at all.

**In code:** `order_similarity` computes the bars, with `word_vector` giving each word its random numbers.

Order carries much of what a sentence means: who did what to whom, what a
"not" applies to, what a line of code does. So position has to be
**injected**, and the oldest way is to add a code for each position to the
word's vector.

## 1. Sinusoidal codes (the original 2017 transformer)

A clock's second hand moves fast and tells nearby moments apart; its hour
hand moves slowly and tells morning from evening. Read together, the hands
give every moment of the day its own reading. **Sinusoidal codes** give each
position a set of hands like these, each turning at its own speed, and add
the reading to the word's embedding.

Each hand is stored as two numbers. Picture the tip of a hand on a circle of
radius 1. The angle it has turned is measured in **radians** (a full turn is
2π ≈ 6.28 radians). The **cosine** of the angle is how far right the tip is,
and the **sine** is how far up; both lie between −1 and +1, and both repeat
every full turn. With 4 numbers per code there are two hands. The fast one
turns 1 radian per position, the slow one 1/100 of a radian:

| position | sin(pos·1) | cos(pos·1) | sin(pos/100) | cos(pos/100) |
|---|---|---|---|---|
| 0 | 0.000 | 1.000 | 0.000 | 1.000 |
| 1 | 0.841 | 0.540 | 0.010 | 1.000 |
| 2 | 0.909 | −0.416 | 0.020 | 1.000 |

From position 1 to position 2 the fast pair changes a lot and the slow pair
hardly at all. The slow pair is there to separate positions hundreds apart.

```mermaid
flowchart LR
  T["token id"] --> E["embedding lookup<br/>(what the word means)"]
  P["position 0,1,2..."] --> S["sin/cos at many frequencies<br/>(where the word is)"]
  E --> ADD(("+"))
  S --> ADD
  ADD --> B["transformer blocks"]
```

**Reading it:** two paths meet at the plus sign. The top path says *what*
the word is, the bottom path says *where* it is, and their sum is one vector
carrying both. Everything after it sees only that sum, so position becomes
part of what attention compares.

In general, a code of d numbers has d/2 hands, and each turns more slowly
than the one before:

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

**In Python:**

```python
import math
d, pos = 4, 1
PE = []
# one (sin, cos) pair per i
for i in range(d // 2):
    # ω_i: pair i's speed
    omega_i = 1 / 10000 ** (2 * i / d)
    # columns 2i and 2i+1
    PE += [math.sin(pos * omega_i), math.cos(pos * omega_i)]
[f"{v:.3f}" for v in PE]  # → ['0.841', '0.540', '0.010', '1.000']
```

Moving k positions on turns every hand by the same angle, k·ω_i, wherever
it starts. So **the dot product of two position codes depends only on how far
apart the positions are**.

**In code:** `sinusoidal_encoding(n, d)` builds the whole table, one row per position.

![Left columns flip colour every few positions and right columns change slowly, so every row is a unique fingerprint and neighbours look alike](figures/primer.ml.positional.sinusoidal_heatmap.svg)

**Reading it:** each row is a position (0 at the top), each column one of 64
numbers, and colour is the value from −1 to +1. The left columns flip colour
every few rows: the fast hands. The right columns change slowly over hundreds
of rows: the slow hands. No two rows are identical, so every position has its
own code, and neighbouring rows look alike, so nearby positions get similar
codes.

The codes need no training and exist for every position. Because their
similarity depends only on distance, a model can learn "look 3 tokens back"
once and use it everywhere. The cost is that position is mixed into the
word's vector itself.

## 2. Learned absolute positions (GPT-2, BERT)

GPT-2 and BERT keep the addition but learn the position vectors instead of
computing them. The model holds a table with one row per position, like
numbered seats in a theatre, and the number of seats is fixed when the model
is built. GPT-2's table has 1,024 rows of 768 numbers: 786,432 parameters,
learned like any other weight. Position 5 reads row 5. Position 1,024, the
1,025th token, has no row.

```mermaid
flowchart LR
  T["token id 318"] --> TE["token table<br/>(vocab × d)"]
  P["position 5"] --> PE["position table<br/>(max_len × d)"]
  TE --> ADD(("+"))
  PE --> ADD
  ADD --> B["transformer blocks"]
  P2["position ≥ max_len"] -.-> X["no row: cannot be encoded"]
```

**Reading it:** this is the sinusoidal picture with the formula on the
bottom path replaced by a second learned table. Then follow the dotted arrow:
the table ends at max_len, so a position past it has no row to look up.

Each word's input is a sum of two rows:

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

**In Python:**

```python
E_dog = [0.5, -0.2]
# a position table with rows 0 to 3
P = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.1, 0.3]]
# x_3 = E_dog + P_3
[round(e + p, 2) for e, p in zip(E_dog, P[3])]  # → [0.6, 0.1]
# no row 1024: it cannot be encoded
len(P) > 1024  # → False
```

**In code:** `primer.ml.transformer.TinyGPT` uses a table like this one.

A learned table is simple and works well up to its last row. Past it there is
nothing to look up, and the table cannot grow after training.

## 3. Rotary position embeddings, RoPE (Llama, Mistral, Qwen)

Both schemes so far add a code to the word's vector, one code per position.
RoPE adds nothing to the word. It turns the word's query and key, the two
vectors attention compares to score a pair of words, like clock hands, by an
angle that grows with the word's position. Two people pointing at a clock
face, one at 1 o'clock and one at 4, hold their arms 90° apart. Move them on
to 8 and 11 o'clock and the angle is still 90°. In the same way, when a
turned query meets a turned key, only the **angle between them** affects the
score, and that angle depends on how far apart the two words are, not on
where they sit.

Take a query and a key of two numbers each, q = k = (1, 0), so there is one
pair, turning 1 radian per position. Put the query at position 3 and the key
at position 7:

* q turns by 3 rad: (cos 3, sin 3) = (−0.990, 0.141)
* k turns by 7 rad: (cos 7, sin 7) = (0.754, 0.657)
* score = −0.990·0.754 + 0.141·0.657 = −0.654 = cos(4)

Move both 100 positions on, to 103 and 107, and the score is still
cos(107 − 103) = cos 4 = −0.654. Only the distance between them counts.

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
and key branches. After projection, each pair of numbers in q and k is
turned by an angle that grows with the word's position, fast for the first
pairs and slow for the last. The value branch is not turned, so what gets
blended is plain content.

A **rotation** turns a point around the centre by an angle without changing
its distance from the centre. For one pair it is a 2 × 2 grid times a column
of two numbers, a **matrix-vector multiply**: each output number is one row
of the grid multiplied entry by entry with the input and added up.

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

**In Python:**

```python
import math
def rotate(x, m, theta_i=1.0):
    # the angle m θ_i
    a = m * theta_i
    # row 1 of the grid
    return (math.cos(a) * x[0] - math.sin(a) * x[1],
            # row 2 of the grid
            math.sin(a) * x[0] + math.cos(a) * x[1])
[f"{v:.3f}" for v in rotate((1, 0), m=3)]  # → ['-0.990', '0.141']
d, i = 2, 0
# θ_0 = 10000^(-2i/d)
10000 ** (-2 * i / d)  # → 1.0
```

The score is a **dot product** (multiply matching numbers and add them up;
it is large when the vectors point the same way), and turning both vectors by
the same angle does not change it. So turning q by angle a and k by angle b
gives the same score as leaving q alone and turning k by b − a:

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

**In Python:**

```python
import math
# turn a pair by an angle (θ = 1, so the angle is the position)
def R(angle, x):
    return (math.cos(angle) * x[0] - math.sin(angle) * x[1],
            math.sin(angle) * x[0] + math.cos(angle) * x[1])
def dot(a, b): return sum(a_i * b_i for a_i, b_i in zip(a, b))
q = k = (1, 0)
# ⟨R_m q, R_n k⟩ with m = 3, n = 7
round(dot(R(3, q), R(7, k)), 3)  # → -0.654
# ⟨q, R_(n-m) k⟩: only the distance
round(dot(q, R(7 - 3, k)), 3)  # → -0.654
# 100 positions later, the same score
round(dot(R(103, q), R(107, k)), 3)  # → -0.654
```

The same holds for vectors of 64 numbers, 32 pairs each turning at its own
speed θ_i, starting anywhere in the text:

![The score curves for starting positions 0, 100 and 1,000 lie exactly on top of each other: only the distance matters](figures/primer.ml.positional.rope_relative.svg)

**Reading it:** the distance n − m between query and key runs along the
bottom. Each line uses the same q and k but starts the pair at a different
position: 0, 100 or 1,000. The lines lie on top of each other, so the score
depends only on distance.

The pairs turn at speeds that fall from the first pair to the last, as the
sinusoidal hands do:

![Pair 0 wraps a full turn about every 6 tokens while the last pairs barely turn across the whole range](figures/primer.ml.positional.rope_frequencies.svg)

**Reading it:** position runs along the bottom, and each line is one pair,
showing how far it has turned, wrapped at one full turn (2π). Pair 0 turns
fastest and wraps about every 6 tokens, so it tells neighbours apart. Pair
31, the last, barely turns across the whole range and carries long-range
position. These are the sinusoidal clock's hands, now inside attention.

**In code:** `rope_frequencies` gives each pair's speed θ_i, and `apply_rope` turns a vector or a whole sequence.

Language works in relative distances ("the word two back"), and RoPE builds
them into every score. Turning a vector never changes its length, so the
word's content is left as it was. And because position now lives in angles,
the angles can be rescaled after training.

## 4. Extending context after training

A RoPE model has only seen the angles produced by positions up to its
training length. Past that length its slow pairs turn to angles it never met,
and attention degrades. A ruler marked from 0 to 4,000 cannot measure
something 32,000 long either. You can shrink the object 8× so it fits the
ruler (**position interpolation**), or redraw the long-distance marks and
keep the fine millimetre marks (**NTK-aware scaling**).

Say a model trained on 4,096 tokens has to read 32,768.
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

**Reading it:** both paths bring the angles back inside the range seen in
training. Interpolation slows *every* hand by the same factor, which also
blurs the fast hands that carry local word order. NTK-aware scaling slows the
slow hands and leaves the fastest one as it was, so local order stays sharp.
Either is usually followed by a short fine-tune on long text.

Position interpolation is one multiplication:

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

**In Python:**

```python
pos, L_train, L_new = 32_000, 4_096, 32_768
# pos' = pos · L_train / L_new
pos * L_train / L_new  # → 4000.0
```

NTK-aware scaling leaves the positions alone and raises the base, the
10,000 in θ_i, instead:

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

**In Python:**

```python
base, s, d = 10_000, 8, 128
# s^(d/(d-2)): just above 8
round(s ** (d / (d - 2)), 2)  # → 8.27
# base' = base · s^(d/(d-2))
base_new = base * s ** (d / (d - 2))
round(base_new)  # → 82685
# θ_i for the last pair, 2i = 126
slowest = lambda b: b ** (-126 / 128)
f"{slowest(base):.2e}", f"{slowest(base_new):.2e}"  # → ('1.15e-04', '1.44e-05')
# exactly 8× slower
round(slowest(base) / slowest(base_new), 6)  # → 8.0
```

**In code:** `interpolate_positions` and `ntk_scaled_base`.

**YaRN** refines NTK-aware scaling by treating each band of frequencies
differently, and many long-context models use it. **ALiBi** takes another
route: it adds no position vectors at all and subtracts a penalty
proportional to distance from each attention score.

![Raw positions leave the trained angle band right after 4k and reach 8 times its top by 32k; interpolation and NTK scaling stay inside it](figures/primer.ml.positional.context_extension.svg)

**Reading it:** position up to 32k runs along the bottom, and the angle of
the slowest RoPE pair up the side. The shaded band is the angles the model
saw in training, at positions up to 4k. The unscaled line leaves the band
right after 4k. The interpolated and NTK-scaled lines stay inside it all the
way to 32k.

This is how a model trained on a few thousand tokens is stretched to 100k or
more without pretraining it again from scratch.

## 5. The causal mask leaks some order

Each scheme so far puts position in on purpose. A decoder, the kind of model
that writes text one word at a time, also gets some order without one. Its
causal mask lets each word see only itself and the words before it, so the
word at position i sees i + 1 words. Even with no position encoding at all
("NoPE"), the model can partly infer where each word sits from how many words
it sees. In bidirectional attention, where every word sees every other (as in
BERT), there is no such signal, and attention is blind to order.

**In code:** `encode_sentence` with `causal=True` applies `primer.ml.attention.causal_mask` and gives the bar chart's "causal mask only" bar.

## In 20 seconds
- Attention with no position information is blind to word order: shuffle
  the words and its outputs are shuffled the same way, so "dog bites man"
  pools to the same vector as "man bites dog".
- The original transformer adds a fixed code of sines and cosines to each
  word. GPT-2 and BERT learn a table of position vectors instead, and it has
  no row past its last position.
- Most current language models use RoPE, which turns each query and key by
  an angle that grows with position, so a score depends only on how far
  apart two words are. Because position lives in angles, a RoPE model can be
  stretched to longer prompts after training.

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

    Args:
        sentence: the words, separated by spaces.
        scheme: how position gets in. "none": embeddings only, so attention
            sees an unordered set. "sinusoidal": add the sinusoidal code to
            each embedding (the 2017 transformer). "rope": rotate q and k by
            position inside attention (modern LLMs).
        causal: when True, each word attends only to itself and earlier words.

    Returns:
        The (n_words, d) output vectors.
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
