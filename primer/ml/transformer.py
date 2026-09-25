r"""
# The transformer block: the unit every modern LLM stacks

Run: `python -m primer.ml.transformer`

New to the notation (vectors, matrix multiply, mean, square root)? Every
symbol is decoded where it appears, and `primer.notation` teaches them all
from zero. This lesson builds on `primer.ml.attention`.

Read alongside the annotated paper: [Attention Is All You Need, annotated](../../papers/attention-is-all-you-need.html).

## 1. The block: a meeting, then desk work

**Everyday picture.** A team works in rounds. Each round starts with a
**meeting**, where everyone listens to everyone else and takes notes on what's
relevant to them (that's attention). Then comes **desk work**: each person
goes back to their own desk and thinks through their notes alone, without
talking to anyone (that's the feed-forward network). Nobody throws away their
old notes; they only add to them (the *residual connection*). A large model
runs dozens of these rounds: GPT-2 small has 12, big models have 80 or more.

**Tiny worked example.** Take one token whose vector is x = (1, 2). The
meeting produces a correction (0.1, −0.3); adding it gives (1.1, 1.7). Desk
work then adds (−0.2, 0.4), giving (0.9, 2.1). The token's vector has been
*edited twice*, never replaced.

```mermaid
flowchart TD
  IN[Input vectors<br/>one per token] --> N1[Layer norm]
  N1 --> AT[Multi-head attention<br/>the meeting: mix across tokens]
  AT --> R1((Add))
  IN --> R1
  R1 --> N2[Layer norm]
  N2 --> FF[Feed-forward network<br/>desk work: each token alone]
  FF --> R2((Add))
  R1 --> R2
  R2 --> OUT[To the next block]
```

**Reading it:** follow the main line straight down the left. The input is
first normalized (rescaled, section 2) and fed to attention. Attention's
output does *not* replace the input: the "Add" circle adds it to the original,
which arrives by the side arrow that skips the whole step. The same pattern
repeats for the feed-forward network. Those two skip arrows are the
**residual connections**. Because every step only adds a correction, the
signal (and during training, the gradient) has an unobstructed highway
through a hundred blocks. This layout, with the norm *before* each sub-layer,
is called **pre-norm**; it trains more stably than the 2017 original, which
normalized after.

**The math and the code.**

$$
x \leftarrow x + \text{Attn}(\text{LN}(x)), \qquad x \leftarrow x + \text{FFN}(\text{LN}(x))
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $x$ | the token vectors, one row per token (the "residual stream") | (1, 2) for one token |
| $\leftarrow$ | "replace the left side with the right side", as in code: `x = x + ...` | |
| $\text{LN}(\cdot)$ | layer normalization (section 2) | |
| $\text{Attn}(\cdot)$ | multi-head attention from `primer.ml.attention`: the meeting | returns (0.1, −0.3) |
| $\text{FFN}(\cdot)$ | the feed-forward network (section 3): desk work | returns (−0.2, 0.4) |
| $+$ | add the correction to the vector, number by number | |

**In words:** "add what the meeting found to each token's notes, then add
what each token worked out alone."

**With the numbers:** (1, 2) + (0.1, −0.3) = (1.1, 1.7); then (1.1, 1.7) +
(−0.2, 0.4) = **(0.9, 2.1)**. `TransformerBlock.__call__` is exactly these
two lines.

**Why it matters.** The block's output has the same shape as its input, so
blocks stack like Lego. And because the input is never overwritten, switching
both sub-layers off gives back the input unchanged (a scenario in the spec),
which is why very deep stacks train at all.

## 2. Layer normalization: grading each token on its own curve

**Everyday picture.** A teacher grading on a curve: subtract the class
average from each score, then divide by how spread out the scores are. After
that, "2 above average" means the same thing in every class. LayerNorm does
this to the numbers *inside one token's vector*, so no token's numbers can
drift huge or tiny as they pass through dozens of blocks.

**Tiny worked example.** Normalize (1, 2, 3, 4). The **mean** (average) is
2.5. The **variance** (average squared distance from the mean) is
(2.25 + 0.25 + 0.25 + 2.25) / 4 = 1.25, so the spread (**standard
deviation**, its square root) is 1.118. Result: (−1.342, −0.447, 0.447,
1.342). Multiply the input by 100 and the result is identical.

```mermaid
flowchart LR
  X["one token: (1, 2, 3, 4)"] --> M["subtract the mean 2.5<br/>(−1.5, −0.5, 0.5, 1.5)"]
  M --> S["divide by the spread 1.118<br/>(−1.342, −0.447, 0.447, 1.342)"]
  S --> G["× learned gain, + learned bias<br/>(start as 1 and 0)"]
```

**Reading it:** two fixed steps, then one learned step. Centring and
dividing force every token's numbers to average 0 with spread 1; the learned
gain and bias then let the model pick whatever scale each feature actually
needs. RMSNorm (used by Llama and most recent models) skips the centring box
and divides by the root-mean-square instead: cheaper, and it works as well.

**The math and the code.**

$$
\text{LN}(x)_j = \gamma_j \,\frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_j,
\qquad \mu = \frac{1}{d}\sum_{j=1}^{d} x_j, \qquad \sigma^2 = \frac{1}{d}\sum_{j=1}^{d}(x_j - \mu)^2
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $x$ | one token's vector | (1, 2, 3, 4) |
| $d$ | how many numbers it has | 4 |
| $j$ | a counter over those numbers, 1 to $d$ | |
| $x_j$ | the $j$-th number | $x_1 = 1$ |
| $\sum_{j=1}^{d}$ | "add up the following for $j$ = 1, 2, ..., $d$" | |
| $\mu$ (mu) | the mean | 2.5 |
| $\sigma^2$ (sigma squared) | the variance | 1.25 |
| $\sqrt{\cdot}$ | square root; $\sqrt{\sigma^2}$ is the spread | 1.118 |
| $\epsilon$ (epsilon) | a tiny number (1e-5) so we never divide by zero | 0.00001 |
| $\gamma_j, \beta_j$ (gamma, beta) | learned gain and bias per feature | 1 and 0 at the start |

**In words:** "subtract the token's average from each of its numbers, divide
by the token's spread, then rescale and shift each feature by learned
amounts."

**With the numbers:** $x_1$: (1 − 2.5) / √(1.25 + 0.00001) = −1.5 / 1.118 =
**−1.342**, then × 1 + 0 = −1.342. `layer_norm` and `rms_norm` implement
both variants.

**Why it matters.** Without normalization, activations drift layer after
layer until training blows up or stalls. It normalizes per token (not per
batch, unlike the BatchNorm used in image networks), so it works for any
batch size and sequence length.

## 3. The feed-forward network: desk work

**Everyday picture.** Back at their desk, each person spreads their notes
out over a desk four times wider than the notebook, underlines what matters,
and writes a short summary back into the notebook. Nobody talks to anyone.

**Tiny worked example.** With model width 8, the network widens each token
to 32 numbers, applies GELU to each, and narrows back to 8. Parameters:
8·32 + 32 + 32·8 + 8 = **552**. GELU on single numbers: GELU(1) = 0.841,
GELU(10) = 10.0, GELU(−3) = −0.004, GELU(0) = 0.

```mermaid
flowchart LR
  X["token vector<br/>d numbers"] --> W1["× W1 + b1<br/>widen to 4·d"]
  W1 --> G["GELU on each number<br/>keep positives, squash negatives"]
  G --> W2["× W2 + b2<br/>back to d"]
  W2 --> Y["correction for this token"]
```

**Reading it:** a token's vector goes in on the left, alone. It is widened
by a matrix multiply, passed through a smooth on/off switch (GELU) number by
number, and narrowed back. The same weights are applied to every token
separately, so this step never mixes tokens; it transforms each one using
what attention already gathered. About two thirds of a large model's
parameters live here, and much of its factual knowledge is thought to be
stored in these weights.

**The math and the code.** A **matrix multiply** $xW$ turns a list of $d$
numbers into a list of $4d$ numbers: each output number is the dot product of
$x$ with one column of $W$ (multiply matching numbers, add them up).

$$
\text{FFN}(x) = \text{GELU}(x W_1 + b_1)\, W_2 + b_2,
\qquad \text{GELU}(z) \approx \tfrac{1}{2} z \left(1 + \tanh\!\left(\sqrt{2/\pi}\,(z + 0.044715\, z^3)\right)\right)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $x$ | one token's vector | 8 numbers |
| $W_1$ | first weight matrix, widens | 8 × 32 |
| $b_1$ | first bias, added after widening | 32 numbers |
| $W_2$ | second weight matrix, narrows | 32 × 8 |
| $b_2$ | second bias | 8 numbers |
| $z$ | any single number fed to GELU | 1 |
| $\tanh$ | "hyperbolic tangent": an S-shaped curve from −1 to +1 | $\tanh(0.8)$ = 0.664 |
| $\pi$ | pi, 3.14159... | |
| $\approx$ | "approximately equals": this tanh form is a fast stand-in for exact GELU | |

**In words:** "widen the token with a matrix multiply, softly switch off
negative numbers, then narrow it back with another matrix multiply."

**With the numbers:** GELU(1) = ½ · 1 · (1 + tanh(0.798 · 1.045)) = ½ · (1 +
tanh(0.834)) = ½ · (1 + 0.683) = **0.841**. `FeedForward` and `gelu` are the
code.

![GELU next to ReLU](figures/primer.ml.transformer.gelu_vs_relu.svg)

**Reading it:** the x-axis is the number going into the activation and the
y-axis is what comes out. ReLU (grey) is a hard hinge: zero for every
negative input, the identity for positives. GELU (blue) follows the same
shape far from zero but bends smoothly around it and dips slightly below
zero for small negatives. The smooth bend means the gradient never jumps,
which is part of why transformers train well with it. (Many newer models use
SwiGLU, a gated cousin of the same idea.)

**Why it matters.** Without a nonlinearity like GELU, the two matrix
multiplies would collapse into one, and stacking layers would add nothing.

## 4. A tiny GPT, end to end

**Everyday picture.** An assembly line. Token ids go in one end, get turned
into vectors, pass through a row of identical workstations (the blocks), get
a final polish (the last norm), and come out the far end as a score for every
word in the dictionary.

**Tiny worked example.** A model with a 50-token vocabulary, width 32, 2
blocks and 16 positions. The ids [3, 1, 4, 1, 5] become a 5 × 32 grid, pass
through both blocks as 5 × 32, and come out as a 5 × 50 grid of **logits**
(raw scores): row 5 scores every possible 6th token. It has **27,328**
parameters.

```mermaid
flowchart LR
  IDS["ids [3,1,4,1,5]"] --> TE["token table lookup<br/>(50 × 32)"]
  POS["positions 0..4"] --> PE["position table lookup<br/>(16 × 32)"]
  TE --> ADD(("+"))
  PE --> ADD
  ADD --> B1["block 1"] --> B2["block 2"] --> LN["final layer norm"]
  LN --> OUT["× token tableᵀ<br/>logits (5 × 50)"]
  TE -. "same table (tied)" .- OUT
```

**Reading it:** the top-left path says *what* each token is, the bottom-left
path *where* it is (see `primer.ml.positional`); their sum enters the blocks.
Every block keeps the 5 × 32 shape. At the end, each token's final vector is
dot-producted with every row of the token table, giving one score per
vocabulary entry. The dotted line marks **weight tying**: the table that
turned ids into vectors on the way in is reused to score them on the way out,
saving vocab × width parameters.

**The math and the code.**

$$
\text{logits} = \text{LN}(h)\, E^{\top}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $h$ | the vectors leaving the last block | 5 × 32 |
| $\text{LN}(h)$ | after the final layer norm | 5 × 32 |
| $E$ | the token embedding table, one row per vocabulary entry | 50 × 32 |
| $E^{\top}$ | $E$ **transposed**: rows become columns | 32 × 50 |
| logits | score of every vocabulary entry at every position | 5 × 50 |

**In words:** "score each candidate next token by how well its embedding
lines up with the model's final vector."

**With the numbers:** row 5 of the logits holds 50 scores, one per token id;
softmax turns them into next-token probabilities (see
`primer.ml.big_picture`). `TinyGPT.__call__` is this pipeline.

**Why it matters.** This is the whole forward pass of GPT-2, Llama and
Claude-style models in miniature. Real models differ in size, not in kind.

## 5. Three families: encoder, decoder, encoder-decoder

**Everyday picture.** An **editor** reads the whole page before commenting
on any sentence (encoder). A **storyteller** tells the story word by word
and can't peek at words they haven't said yet (decoder). A **translator**
reads the whole source sentence, then writes the translation word by word
(encoder-decoder).

**Tiny worked example.** With 3 tokens, an encoder's attention may use all 9
(query, key) pairs; a decoder only the 6 on or below the diagonal. In the
spec, changing token 5 changes token 1's output in an encoder block, and
leaves it untouched in a decoder block.

```mermaid
flowchart TB
  subgraph ENC["Encoder-only (BERT)"]
    e1[every token sees every token] --> e2[a vector per token<br/>classify, embed, tag]
  end
  subgraph DEC["Decoder-only (GPT, Claude, Llama)"]
    d1[each token sees only the past] --> d2[predict the next token<br/>generate, chat, agents]
  end
  subgraph ED["Encoder-decoder (T5, original transformer)"]
    x1[encoder reads the source<br/>bidirectionally] --> x2[decoder writes the output causally,<br/>attending to the encoder too]
  end
```

**Reading it:** the three boxes differ only in *who may attend to whom*.
The blocks inside are the same. That one choice of mask decides what the
model is good at: seeing everything suits understanding tasks, and seeing
only the past is what makes generation possible.

| Type            | Example                  | Attention     | Best for                        |
|-----------------|--------------------------|---------------|---------------------------------|
| Encoder-only    | BERT                     | Bidirectional | Classification, embeddings, NER |
| Decoder-only    | GPT, Claude, Llama       | Causal        | Generation, chat, agents        |
| Encoder-decoder | T5, original transformer | Both          | Translation, summarization      |

![Encoder vs. decoder attention patterns](figures/primer.ml.transformer.masks.svg)

**Reading it:** both panels are attention weights from one block on the
same 6 input vectors; rows are the token looking, columns the token looked
at, darker is more weight. On the left (encoder) weight is spread over the
whole grid. On the right (decoder) the upper-right triangle is empty, so
nothing reads the future. The bottom rows are identical in both panels: the
last token sees everything either way.

**Why it matters.** Embedding models (`primer.ml.embeddings`) are usually
encoders; chat models and agents are decoders.

## 6. Counting parameters

**Everyday picture.** Counting the bricks in a Lego model from its
blueprint, without opening the box.

**Tiny worked example.** For width d, one block holds: attention 4·d²
(W_q, W_k, W_v, W_o), feed-forward 8·d² (two d × 4d matrices), plus small
bias and norm terms. With d = 32: 12 · 1,024 = 12,288, plus 160 (biases) plus
128 (norms) = 12,576 per block. Two blocks, plus tables of 50 × 32 and 16 × 32
and a final norm of 64, gives **27,328**. The same recipe gives GPT-2 small
exactly its published **124,439,808**.

```mermaid
flowchart LR
  B["one block"] --> A["attention<br/>4·d²"]
  B --> F["feed-forward<br/>8·d²"]
  B --> N["norms and biases<br/>~13·d (tiny)"]
  A & F --> T["≈ 12·d² per block"]
  T --> ALL["× L blocks + embeddings<br/>vocab·d + positions·d"]
```

**Reading it:** almost all of a block's weight sits in two places,
attention's four square matrices and the feed-forward's two wide ones.
Everything else is a rounding error at large width, which is where the
famous 12·L·d² rule comes from. Embeddings are added once, not per block.

**The math.**

$$
N \approx 12\, L\, d^2 + V d
$$

**Symbols**

| Symbol | Meaning here | GPT-2 small |
|---|---|---|
| $N$ | total parameters | 124,439,808 |
| $L$ | number of blocks (layers) | 12 |
| $d$ | model width | 768 |
| $d^2$ | $d$ times $d$: the size of one square matrix | 589,824 |
| $V$ | vocabulary size | 50,257 |
| $Vd$ | the token embedding table | 38,597,376 |

**In words:** "about twelve square matrices per layer, plus the embedding
table."

**With the numbers:** 12 · 12 · 589,824 = 84,934,656, plus 38,597,376 =
123,532,032, about 1% under the exact 124,439,808 (which also counts positions,
biases and norms; see `gpt_param_count`).

![Where GPT-2's parameters live](figures/primer.ml.transformer.param_breakdown.svg)

**Reading it:** each bar is one GPT-2 size, split by where the parameters
live. Feed-forward (red) is always the largest block of the stack. In the
small model, embeddings (grey) are almost a third of everything, because a
50k-row table is big next to 12 narrow layers. As models grow, the
embeddings stay fixed while layers multiply, so their share shrinks to about
5% in XL, and the 12·L·d² term dominates.

**Why it matters.** Parameters × bytes per parameter is the memory a model
needs just to load (see `primer.ml.inference` for the arithmetic).

## 7. Mixture of Experts (MoE): a triage desk

**Everyday picture.** A hospital triage desk. Instead of one general
practitioner seeing every patient, the desk sends each patient to the two
most relevant specialists out of eight. The hospital employs eight doctors'
worth of expertise, but each patient only takes up two doctors' time.

**Tiny worked example.** A router scores one token against 4 experts: (2.0,
1.0, 0.5, −1.0). Keep the top 2 (experts 0 and 1) and softmax just those
two: e² / (e² + e¹) = 7.39 / 10.11 = **0.731**, and 0.269. The token's output is
0.731 × expert 0's output + 0.269 × expert 1's output; experts 2 and 3 never
run for this token.

```mermaid
flowchart LR
  X[token vector] --> R["router<br/>one score per expert"]
  R --> K["keep top 2<br/>softmax over them"]
  K -->|0.731| E0[expert 0: an FFN]
  K -->|0.269| E1[expert 1: an FFN]
  K -.->|skipped| E2[expert 2]
  K -.->|skipped| E3[expert 3]
  E0 --> S(("weighted sum"))
  E1 --> S
  S --> Y[output]
```

**Reading it:** the router is a tiny linear layer that scores the token
against every expert. Only the two best-scoring experts run, and their
outputs are blended by the renormalized scores (solid arrows); the others
are skipped entirely (dotted). An MoE layer replaces the feed-forward network
inside a block; attention is unchanged.

**The math and the code.**

$$
y = \sum_{i \in \text{TopK}(r)} g_i \, E_i(x), \qquad g = \text{softmax}\big(r_{\text{TopK}}\big), \qquad r = x W_r
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $x$ | one token's vector | |
| $W_r$ | the router's weights, width × number of experts | |
| $r$ | the router scores | (2.0, 1.0, 0.5, −1.0) |
| $\text{TopK}(r)$ | the positions of the $k$ largest scores | experts {0, 1} for k = 2 |
| $r_{\text{TopK}}$ | just those scores | (2.0, 1.0) |
| $g_i$ | expert $i$'s gate: softmax over the kept scores | 0.731, 0.269 |
| $E_i(x)$ | expert $i$ (a feed-forward network) applied to $x$ | |
| $\sum_{i \in \ldots}$ | add up over the chosen experts only | two terms |

**In words:** "score every expert, keep the best k, and blend those experts'
outputs by their softmaxed scores."

**With the numbers:** y = 0.731 · E₀(x) + 0.269 · E₁(x). With 8 experts of
width 16, the layer holds 8 × 2,128 + 128 = **17,152** parameters, but one token
touches only 2 × 2,128 + 128 = **4,384**. `MixtureOfExperts` and `top_k_gates`
are the code. Mixtral 8x7B works the same way: about 47B parameters in total,
about 13B active per token.

A router left alone tends to play favourites, overloading some experts
while others starve. Training adds a small **load-balancing loss** (Switch
Transformer):

$$
\mathcal{L}_{\text{balance}} = n \sum_{i=1}^{n} f_i \, P_i
$$

**Symbols**

| Symbol | Meaning here | Balanced example | Collapsed example |
|---|---|---|---|
| $n$ | number of experts | 4 | 4 |
| $f_i$ | share of routing slots that went to expert $i$ | ¼ each | (1, 0, 0, 0) |
| $P_i$ | expert $i$'s average router probability | ¼ each | ≈ (1, 0, 0, 0) |
| $\mathcal{L}$ | the penalty added to the training loss | 1.0 | ≈ 4.0 |

**In words:** "number of experts times the sum, over experts, of traffic
share times average router probability."

**With the numbers:** balanced: 4 · (4 · ¼ · ¼) = **1.0**, the minimum.
Collapsed onto one expert: 4 · (1 · 1) = **4.0**, the maximum
(`load_balancing_loss`).

![Tokens per expert with an untrained router](figures/primer.ml.transformer.moe_load.svg)

**Reading it:** each bar counts how many of 256 tokens an untrained router
sent to each of 8 experts (top-2, so 512 slots in all); the dashed line is the
perfectly even 64. Expert 3 gets 99 tokens while others get about 55: even
random routing is lopsided, and in training the imbalance compounds because
favoured experts improve and attract more traffic. The balancing loss pushes
the bars back towards the line.

**Why it matters.** MoE lets a model hold far more knowledge (parameters)
for the same compute per token, which is why many frontier models use it. The
price is memory: every expert must be loaded even though few run per token.

## 8. Compute arithmetic: 2N to generate, 6N to train

**Everyday picture.** Every parameter does one multiply and one add for
every token that passes through it, like a toll booth that charges two coins
per car.

**Tiny worked example.** A 7B-parameter model generating one token: 2 × 7e9
= **1.4e10** operations (14 GFLOPs). Training a 70B model on a trillion tokens:
6 × 70e9 × 1e12 = **4.2e23** operations.

```mermaid
flowchart LR
  F["forward pass<br/>2 FLOPs per parameter per token"] --> L[loss]
  L --> B["backward pass<br/>≈ 4 FLOPs per parameter per token"]
  B --> T["training total ≈ 6·N per token"]
  F --> I["inference total ≈ 2·N per token"]
```

**Reading it:** inference only runs the forward pass, so it pays 2 per
parameter per token. Training runs the same forward pass, then a backward
pass (`primer.ml.neural_net`) that costs about twice as much, because it
computes gradients both for the weights and for the activations. 2 + 4 = 6.

**The math.**

$$
C_{\text{infer}} \approx 2N \text{ per token}, \qquad C_{\text{train}} \approx 6ND
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $N$ | number of parameters | 7e9 / 70e9 / 175e9 |
| $D$ | number of training tokens | 1e12 / 300e9 |
| $C$ | compute in FLOPs (floating-point operations: one multiply or one add) | |
| $e$ in 7e9 | "times 10 to the power": 7e9 = 7,000,000,000 | |

**In words:** "generating costs two operations per parameter per token;
training costs six per parameter per training token."

**With the numbers:** GPT-3: 6 × 175e9 × 300e9 = **3.15e23** FLOPs, matching
the roughly 3.14e23 its paper reports (`training_flops`,
`inference_flops_per_token`).

**Why it matters.** These two lines let you estimate GPU-hours, serving cost
and training budgets on the back of an envelope.

## In 20 seconds
- A block is attention (tokens exchange information) then a feed-forward
  network (each token processes alone), each wrapped in a residual add and a
  layer norm (pre-norm).
- A GPT is: embed tokens and positions, run N blocks, normalize, and score
  every vocabulary entry with the (tied) embedding table.
- Parameters ≈ 12·L·d² + vocab·d; compute ≈ 2N per generated token and 6ND
  to train. MoE adds experts to grow parameters without growing per-token
  compute.

## Self-test questions

**What do attention and the feed-forward network each contribute?**
Attention mixes information across tokens; the feed-forward network
transforms each token independently and holds most of the parameters (and,
it's thought, much of the factual knowledge).

**Why residual connections?**
Each sub-layer adds a correction instead of replacing its input, so signal
and gradients have a direct path through very deep stacks.

**Pre-norm vs. post-norm?**
Pre-norm normalizes before each sub-layer and leaves the residual path
untouched; it trains more stably, so modern models use it.

**Why LayerNorm rather than BatchNorm in transformers?**
It normalizes within each token, so it doesn't depend on batch size or
sequence length.

**Roughly how many parameters does a model with 32 layers of width 4096 have, not counting embeddings?**
12 · 32 · 4096² ≈ 6.4 billion.

**Encoder-only vs. decoder-only?**
Encoders attend bidirectionally and suit classification and embeddings;
decoders attend causally and generate text.

**What does Mixture of Experts buy you, and what does it cost?**
More total parameters at the same compute per token, since each token runs
only k experts. It costs memory (all experts loaded), routing complexity and
load-balancing.

**Estimate the compute to train a 7B model on 2 trillion tokens.**
6 × 7e9 × 2e12 = 8.4e22 FLOPs.

## The papers behind this lesson

- **Vaswani et al. (2017), *Attention Is All You Need*.** https://arxiv.org/abs/1706.03762.
  Introduced the transformer block: attention plus feed-forward, residuals and
  layer norm, stacked. [annotated companion](../../papers/attention-is-all-you-need.html)
- **Ba, Kiros & Hinton (2016), *Layer Normalization*.** https://arxiv.org/abs/1607.06450.
  Normalized within each example instead of across the batch, the variant
  transformers use. [annotated companion](../../papers/layer-norm.html)
- **Zhang & Sennrich (2019), *Root Mean Square Layer Normalization*.**
  https://arxiv.org/abs/1910.07467. Dropped the mean-centring for a cheaper
  norm now used by most LLMs.
- **Hendrycks & Gimpel (2016), *Gaussian Error Linear Units (GELUs)*.**
  https://arxiv.org/abs/1606.08415. The smooth activation used in GPT-2 and
  BERT.
- **Xiong et al. (2020), *On Layer Normalization in the Transformer Architecture*.**
  https://arxiv.org/abs/2002.04745. Explained why pre-norm trains more
  stably than post-norm.
- **Devlin et al. (2018), *BERT*.** https://arxiv.org/abs/1810.04805. The
  canonical encoder-only model.
- **Fedus, Zoph & Shazeer (2021), *Switch Transformers*.** https://arxiv.org/abs/2101.03961.
  Scaled Mixture of Experts and introduced the load-balancing loss built here.
- **Jiang et al. (2024), *Mixtral of Experts*.** https://arxiv.org/abs/2401.04088.
  An open top-2-of-8 MoE model: about 47B parameters, about 13B active per token.
- **Kaplan et al. (2020), *Scaling Laws for Neural Language Models*.**
  https://arxiv.org/abs/2001.08361. Popularized the 6·N·D compute estimate.
  [annotated companion](../../papers/scaling-laws.html)

## Further reading
- The Annotated Transformer (Harvard NLP): https://nlp.seas.harvard.edu/annotated-transformer/
- The Illustrated Transformer (Jay Alammar): https://jalammar.github.io/illustrated-transformer/
- Andrej Karpathy, *Let's build GPT* (video): https://www.youtube.com/watch?v=kCc8FmEb1nY
- Karpathy's `nanoGPT`: https://github.com/karpathy/nanoGPT
- Hugging Face LLM course, *How do Transformers work?*: https://huggingface.co/learn/llm-course/chapter1/4
- Hugging Face blog, *Mixture of Experts Explained*: https://huggingface.co/blog/moe
"""

from __future__ import annotations

import numpy as np

from primer._show import banner, matrix, say, table, takeaway
from primer.ml.attention import MultiHeadAttention

# ---------------------------------------------------------------------------
# 1. Normalization and activation
# ---------------------------------------------------------------------------

EPS = 1e-5  # stops division by zero when every value in a row is equal


def layer_norm(x: np.ndarray, gain: np.ndarray | None = None, bias: np.ndarray | None = None) -> np.ndarray:
    """Normalize each row (token) to mean 0 and standard deviation 1, then rescale.

    Normalizing across a token's own features (not across the batch) is what
    makes it independent of batch size and sequence length.
    """
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    y = (x - mean) / np.sqrt(var + EPS)
    if gain is not None:
        y = y * gain
    if bias is not None:
        y = y + bias
    return y


def rms_norm(x: np.ndarray, gain: np.ndarray | None = None) -> np.ndarray:
    """Divide each row by its root mean square. No mean subtraction, no bias.

    Cheaper than LayerNorm and works as well in practice; Llama, Mistral and
    most recent LLMs use it.
    """
    y = x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + EPS)
    return y * gain if gain is not None else y


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit, tanh approximation (as in GPT-2 and BERT).

    Roughly: pass positive values, suppress negative ones, but smoothly, so
    the gradient never jumps the way ReLU's does at 0.
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


# ---------------------------------------------------------------------------
# 2. The feed-forward network: each token thinks on its own
# ---------------------------------------------------------------------------


class FeedForward:
    """Two linear layers with GELU between: d_model -> 4·d_model -> d_model.

    Applied to every token *separately* (the same weights for each row), so
    unlike attention it never mixes information between tokens. About two
    thirds of a transformer's parameters live here, and much of its factual
    knowledge is thought to be stored in these weights.
    """

    def __init__(self, d_model: int, expansion: int = 4, seed: int = 0):
        rng = np.random.default_rng(seed)
        hidden = expansion * d_model
        # Scaled init keeps activations near unit size (see primer.ml.deep_nets).
        self.W1 = rng.normal(0, 1 / np.sqrt(d_model), (d_model, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 1 / np.sqrt(hidden), (hidden, d_model))
        self.b2 = np.zeros(d_model)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # (seq, d) @ (d, 4d) -> (seq, 4d) -> GELU -> (seq, 4d) @ (4d, d) -> (seq, d)
        return gelu(x @ self.W1 + self.b1) @ self.W2 + self.b2

    def n_params(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size


# ---------------------------------------------------------------------------
# 3. The block: a meeting (attention), then desk work (feed-forward)
# ---------------------------------------------------------------------------


class TransformerBlock:
    """Pre-norm transformer block:

        x = x + Attention(LayerNorm(x))     # tokens exchange information
        x = x + FeedForward(LayerNorm(x))   # each token processes what it gathered

    `causal=True` is a decoder block (GPT, Claude, Llama): each token sees only
    the past. `causal=False` is an encoder block (BERT): every token sees all.
    """

    def __init__(self, d_model: int, n_heads: int, causal: bool = True, seed: int = 0):
        self.causal = causal
        self.attn = MultiHeadAttention(d_model, n_heads, seed=seed)
        self.ffn = FeedForward(d_model, seed=seed + 1)
        # LayerNorm's learned gain (start at 1) and bias (start at 0).
        self.ln1_g, self.ln1_b = np.ones(d_model), np.zeros(d_model)
        self.ln2_g, self.ln2_b = np.ones(d_model), np.zeros(d_model)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        attn_out, _ = self.attn(layer_norm(x, self.ln1_g, self.ln1_b), causal=self.causal)
        x = x + attn_out  # residual: add a correction, never replace the signal
        x = x + self.ffn(layer_norm(x, self.ln2_g, self.ln2_b))
        return x

    def n_params(self) -> int:
        a = self.attn
        return a.W_q.size + a.W_k.size + a.W_v.size + a.W_o.size + self.ffn.n_params() + 4 * self.ln1_g.size


# ---------------------------------------------------------------------------
# 4. A tiny GPT: embeddings -> N blocks -> final norm -> scores per token
# ---------------------------------------------------------------------------


class TinyGPT:
    """A complete decoder-only language model forward pass, GPT-2 style.

    token ids (seq,) -> token vectors + position vectors (seq, d)
                     -> n_layers × TransformerBlock          (seq, d)
                     -> final LayerNorm                      (seq, d)
                     -> dot with every token's embedding     (seq, vocab)  "logits"

    The output layer reuses the token-embedding table (weight tying): the
    score for token t is the dot product of the final vector with t's own
    embedding. It saves vocab·d parameters and works as well as a separate
    output matrix.

    Weights are random: this shows the machinery, not a trained model.
    """

    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, max_len: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # Small init (std 0.02, as in GPT-2) so the untrained model starts
        # out nearly uniform over the vocabulary.
        self.wte = rng.normal(0, 0.02, (vocab_size, d_model))  # token embeddings
        self.wpe = rng.normal(0, 0.02, (max_len, d_model))  # learned positions
        self.blocks = [TransformerBlock(d_model, n_heads, causal=True, seed=seed + 10 * i) for i in range(n_layers)]
        self.lnf_g, self.lnf_b = np.ones(d_model), np.zeros(d_model)
        self.max_len = max_len

    def hidden(self, ids: np.ndarray) -> np.ndarray:
        """The final (seq, d) vectors before the output layer."""
        ids = np.asarray(ids)
        assert len(ids) <= self.max_len, "learned positions stop at max_len (see primer.ml.positional)"
        x = self.wte[ids] + self.wpe[: len(ids)]  # what + where
        for block in self.blocks:
            x = block(x)
        return layer_norm(x, self.lnf_g, self.lnf_b)

    def __call__(self, ids: np.ndarray) -> np.ndarray:
        """Logits: (seq, vocab). Row i scores every possible token at position i+1."""
        return self.hidden(ids) @ self.wte.T

    def n_params(self) -> int:
        return self.wte.size + self.wpe.size + sum(b.n_params() for b in self.blocks) + 2 * self.lnf_g.size


def gpt_param_count(vocab: int, d_model: int, n_layers: int, max_len: int, attn_bias: bool = True) -> int:
    """Parameters of a GPT-2-shaped model, from the architecture alone.

    Per layer: attention 4·d² (+4·d biases), feed-forward 8·d² + 5·d, two
    LayerNorms 4·d. Plus token table vocab·d, position table max_len·d, and a
    final LayerNorm 2·d. The output layer is tied, so it adds nothing.
    """
    d = d_model
    per_layer = 12 * d * d + 5 * d + 4 * d + (4 * d if attn_bias else 0)
    return n_layers * per_layer + vocab * d + max_len * d + 2 * d


# ---------------------------------------------------------------------------
# 5. Mixture of Experts: many feed-forward networks, each token visits a few
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def top_k_gates(router_logits: np.ndarray, k: int) -> np.ndarray:
    """(tokens, experts) router scores -> gate weights, non-zero for the top k only.

    Softmax is taken over the k kept scores, so each token's gates sum to 1
    (the Mixtral recipe). Ties go to the lower-numbered expert.
    """
    # argsort ascending on the negated scores = descending; stable keeps ties in order.
    top = np.argsort(-router_logits, axis=-1, kind="stable")[:, :k]
    kept = np.take_along_axis(router_logits, top, axis=-1)
    gates = np.zeros_like(router_logits, dtype=float)
    np.put_along_axis(gates, top, _softmax(kept), axis=-1)
    return gates


class MixtureOfExperts:
    """Replaces one FeedForward with `n_experts` of them plus a router.

    Each token is scored against every expert by a tiny linear router, sent
    to its top `k`, and gets back the gate-weighted sum of those experts'
    outputs. Total parameters grow with n_experts; compute per token grows
    only with k.
    """

    def __init__(self, d_model: int, n_experts: int = 8, k: int = 2, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.k = k
        self.router = rng.normal(0, 1 / np.sqrt(d_model), (d_model, n_experts))
        self.experts = [FeedForward(d_model, seed=seed + 100 + i) for i in range(n_experts)]
        self.last_gates: np.ndarray | None = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        gates = top_k_gates(x @ self.router, self.k)  # (tokens, experts)
        self.last_gates = gates
        out = np.zeros_like(x)
        for e, expert in enumerate(self.experts):
            chosen = gates[:, e] > 0
            if chosen.any():  # an expert only runs on the tokens routed to it
                out[chosen] += gates[chosen, e : e + 1] * expert(x[chosen])
        return out

    def n_params(self) -> int:
        return self.router.size + sum(e.n_params() for e in self.experts)

    def active_params(self) -> int:
        """Parameters one token actually touches: the router plus k experts."""
        return self.router.size + self.k * self.experts[0].n_params()

    def tokens_per_expert(self) -> np.ndarray:
        assert self.last_gates is not None, "run the layer first"
        return (self.last_gates > 0).sum(axis=0)


def load_balancing_loss(router_logits: np.ndarray, k: int) -> float:
    """Switch Transformer auxiliary loss: n_experts · Σ_i f_i · P_i.

    f_i = share of routing slots that went to expert i (hard counts),
    P_i = average router probability for expert i (soft, differentiable).
    Equals 1.0 when traffic is perfectly even and approaches n_experts when
    one expert takes everything. Added (times a small weight) to the training
    loss so experts don't collapse onto a favourite few.
    """
    n_experts = router_logits.shape[-1]
    f = (top_k_gates(router_logits, k) > 0).mean(axis=0) / k
    P = _softmax(router_logits).mean(axis=0)
    return float(n_experts * np.sum(f * P))


# ---------------------------------------------------------------------------
# 6. Compute arithmetic
# ---------------------------------------------------------------------------


def inference_flops_per_token(n_params: float) -> float:
    """≈ 2·N: every weight does one multiply and one add per generated token.

    Ignores the attention-over-context term, which matters only for very long
    contexts (see primer.ml.attention.attention_cost).
    """
    return 2 * n_params


def training_flops(n_params: float, n_tokens: float) -> float:
    """≈ 6·N·D: forward pass 2·N per token, backward pass about twice that."""
    return 6 * n_params * n_tokens


# ---------------------------------------------------------------------------
# 7. Figures (rendered to docs/figures by `make figures`)
# ---------------------------------------------------------------------------

GPT2_SIZES = {"small": (768, 12), "medium": (1024, 24), "large": (1280, 36), "XL": (1600, 48)}


def param_breakdown(d_model: int, n_layers: int, vocab: int = 50257, max_len: int = 1024) -> dict[str, int]:
    """Where a GPT-2-shaped model's parameters live."""
    d = d_model
    return {
        "embeddings": vocab * d + max_len * d,
        "attention": n_layers * (4 * d * d + 4 * d),
        "feed-forward": n_layers * (8 * d * d + 5 * d),
        "norms": n_layers * 4 * d + 2 * d,
    }


def mask_patterns(n: int = 6, d_model: int = 16, seed: int = 3) -> dict[str, np.ndarray]:
    """Attention weights of one encoder block and one decoder block on the same input."""
    x = np.random.default_rng(seed).standard_normal((n, d_model))
    out = {}
    for name, causal in (("encoder (bidirectional)", False), ("decoder (causal)", True)):
        block = TransformerBlock(d_model, n_heads=1, causal=causal, seed=seed)
        _, w = block.attn(layer_norm(x), causal=causal)
        out[name] = w[0]
    return out


def figures() -> dict:
    """Plot this lesson's data. matplotlib is imported here, and only here."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, RED, MUTED = "#2563eb", "#dc2626", "#9ca3af"
    figs = {}

    x = np.linspace(-4, 4, 400)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(x, np.maximum(x, 0), color=MUTED, lw=2, label="ReLU: max(0, x)")
    ax.plot(x, gelu(x), color=BLUE, lw=2, label="GELU")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("input to the activation")
    ax.set_ylabel("output")
    ax.set_title("GELU: a smooth ReLU")
    ax.legend(frameon=False)
    fig.tight_layout()
    figs["gelu_vs_relu"] = fig

    masks = mask_patterns()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, (name, w) in zip(axes, masks.items()):
        im = ax.imshow(w, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(name)
        ax.set_xlabel("key (token looked at)")
        ax.set_ylabel("query (token looking)")
    fig.colorbar(im, ax=axes, fraction=0.03, label="attention weight")
    figs["masks"] = fig

    fig, ax = plt.subplots(figsize=(6.5, 4))
    names = list(GPT2_SIZES)
    parts = [param_breakdown(*GPT2_SIZES[n]) for n in names]
    bottom = np.zeros(len(names))
    for key, color in zip(["embeddings", "attention", "feed-forward", "norms"], [MUTED, BLUE, RED, "black"]):
        vals = np.array([p[key] for p in parts]) / 1e6
        ax.bar(names, vals, bottom=bottom, color=color, label=key)
        bottom += vals
    for i, total in enumerate(bottom):
        ax.text(i, total + 20, f"{total:,.0f}M", ha="center")
    ax.set_ylabel("parameters (millions)")
    ax.set_xlabel("GPT-2 size")
    ax.set_title("Where the parameters live")
    ax.legend(frameon=False)
    fig.tight_layout()
    figs["param_breakdown"] = fig

    moe = MixtureOfExperts(d_model=16, n_experts=8, k=2, seed=0)
    moe(np.random.default_rng(1).standard_normal((256, 16)))
    counts = moe.tokens_per_expert()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(range(8), counts, color=BLUE)
    ax.axhline(256 * 2 / 8, color=RED, ls="--", label="perfectly even (64 each)")
    ax.set_xlabel("expert")
    ax.set_ylabel("tokens routed to it (of 256, top-2)")
    ax.set_title("An untrained router plays favourites")
    ax.legend(frameon=False)
    fig.tight_layout()
    figs["moe_load"] = fig
    return figs


# ---------------------------------------------------------------------------
# 8. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. Normalization: grade every token on its own curve")
    x = np.array([1.0, 2.0, 3.0, 4.0])
    table(["input", "LayerNorm", "RMSNorm"], [(a, b, c) for a, b, c in zip(x, layer_norm(x), rms_norm(x))], floatfmt=".3f")
    say("LayerNorm subtracts the mean (2.5) and divides by the spread (1.118). RMSNorm skips the centring.")

    banner("2. The block: a meeting, then desk work")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 16))
    block = TransformerBlock(16, n_heads=4)
    Y = block(X)
    say(
        f"""
        Input (5 tokens × 16), output {Y.shape}: same shape, so blocks stack.
        The output is the input plus two added corrections, one from the
        meeting and one from the desk work; the input itself is never
        overwritten, which is what lets gradients flow through deep stacks.
        """
    )
    takeaway("Attention mixes information across tokens; the feed-forward network processes each token alone.")

    banner("3. A tiny GPT, end to end")
    model = TinyGPT(vocab_size=50, d_model=32, n_layers=2, n_heads=4, max_len=16)
    ids = np.array([3, 1, 4, 1, 5])
    logits = model(ids)
    say(
        f"""
        ids {ids.tolist()} -> vectors (5, 32) -> 2 blocks -> final norm ->
        logits {logits.shape}: one score for each of 50 vocabulary entries at
        each position. {model.n_params():,} parameters in total.
        """
    )

    banner("4. Counting parameters")
    table(
        ["model", "width", "layers", "parameters"],
        [(n, d, L, f"{gpt_param_count(50257, d, L, 1024):,}") for n, (d, L) in GPT2_SIZES.items()],
    )
    say("GPT-2 small comes out at exactly its published 124,439,808. Roughly 12·layers·width² plus the embeddings.")

    banner("5. Mixture of Experts: a triage desk")
    gates = top_k_gates(np.array([[2.0, 1.0, 0.5, -1.0]]), k=2)
    say(f"Router scores (2, 1, 0.5, -1) -> gates {np.round(gates[0], 3).tolist()}: experts 0 and 1, 73% / 27%.")
    moe = MixtureOfExperts(d_model=16, n_experts=8, k=2)
    moe(rng.standard_normal((256, 16)))
    say(
        f"""
        8 experts hold {moe.n_params():,} parameters but each token touches only
        {moe.active_params():,}. Tokens per expert for 256 tokens:
        {moe.tokens_per_expert().tolist()} (even would be 64 each).
        """
    )
    takeaway("MoE grows total parameters (knowledge) without growing compute per token.")

    banner("6. Compute arithmetic")
    say(
        f"""
        Generating one token with a 7B model: {inference_flops_per_token(7e9):.1e} FLOPs.
        Training 70B parameters on 1T tokens: {training_flops(70e9, 1e12):.1e} FLOPs.
        GPT-3 (175B, 300B tokens): {training_flops(175e9, 300e9):.2e}.
        """
    )


if __name__ == "__main__":
    demo()
