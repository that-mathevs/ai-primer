r"""
# Deep networks: why deep stacks fail to train, and the fixes that made them work

Run: `python -m primer.ml.deep_nets`

## The idea: a gradient is a product of slopes

Picture a game of telephone along a line of 30 people. Each person repeats
the message to the next, but everyone speaks at a quarter of the volume they
heard. By the end of the line the message is silence. If instead everyone
speaks 1.5× louder, the end of the line is a deafening roar. Training a deep
network has exactly this problem, run backwards: the learning signal (the
**gradient**, how much each weight should change; see `primer.ml.neural_net`)
starts at the output and is passed back layer by layer, and each layer
multiplies it by its own **slope** (derivative). Thirty multiplications by
something below 1 is almost zero: the **vanishing gradient**. Thirty by
something above 1 is enormous: the **exploding gradient**.

Worked example: a chain of ten one-number layers, each sitting at its
steepest point.

| chain | slope per layer | gradient after 10 layers |
|---|---|---|
| sigmoid units, weight 1 | 0.25 | 0.25¹⁰ = 0.00000095 |
| linear units, weight 1 | 1 | 1¹⁰ = 1 |
| linear units, weight 1.5 | 1.5 | 1.5¹⁰ = 57.7 |

```mermaid
flowchart RL
  L[Loss] -- "gradient 1" --> H10[layer 10]
  H10 -- "× 0.25" --> H9[layer 9]
  H9 -- "× 0.25" --> H8[layer 8]
  H8 -- "× 0.25 ... " --> H2[layer 2]
  H2 -- "× 0.25" --> H1["layer 1<br/>receives 0.25¹⁰ ≈ 1e-6"]
```

**Reading it:** read right to left, the direction backprop travels. The loss
hands the last layer a gradient of 1. Every hop multiplies by that layer's
slope (0.25 for a sigmoid at its steepest). After ten hops the first layer
receives about one millionth of the signal, so its weights barely change: it
effectively stops learning while the later layers carry on.

$$
\frac{\partial \mathcal{L}}{\partial h_0} = \frac{\partial \mathcal{L}}{\partial h_L}\prod_{l=1}^{L} w_l\,\phi'(z_l)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $h_0$ | the input to the first layer | |
| $h_L$ | the output of the last layer | |
| $L$ | the number of layers | 10 |
| $l$ | a counter over the layers | 1 to 10 |
| $\prod_{l=1}^{L}$ | "multiply together the following, for every layer" (like Σ, but multiplying) | ten factors |
| $w_l$ | layer $l$'s weight | 1 |
| $\phi'(z_l)$ | the activation's slope at layer $l$'s input | 0.25 |
| $\partial \mathcal{L}/\partial h$ | how much the loss changes when $h$ changes | 1 at the top |

**In words:** "the gradient reaching the first layer is the gradient at the
top times every layer's weight times every layer's slope."

**With the numbers:** $1 \times (1 \times 0.25)^{10} = 9.5 \times 10^{-7}$.

`chain_gradient` builds that chain and backprops through it.

**Why it matters:** this is why networks deeper than a handful of layers
were considered untrainable for decades. Every fix below (better
activations, careful initialization, residual connections, normalization)
is a way of keeping the per-layer factor close to 1.

## In a real network: watching the gradient layer by layer

In a real layer, each neuron sums 64 inputs, so the multiplier per layer
depends on three things together: the size of the weights, how many inputs
each neuron adds up, and the activation's slope. Same telephone game, but
now everyone in the line hears 64 people at once.

Worked example: a 30-layer network, 64 neurons per layer. The ratio of the
gradient at the first layer to the gradient at the last:

| setup | ratio first / last |
|---|---|
| ReLU, He initialization | ≈ 4 (healthy) |
| ReLU, weights too small (std 0.01) | ≈ 10⁻³⁶ (vanished) |
| ReLU, weights too large (std 1) | ≈ 10²² (exploded) |
| sigmoid, Xavier initialization | ≈ 10⁻¹⁸ (vanished) |

![Gradient size at every layer of a 30-layer network](figures/primer.ml.deep_nets.gradient_flow.svg)

**Reading it:** the horizontal axis is the layer (1 is next to the input, 30
next to the loss) and the vertical axis is the size of the gradient reaching
that layer, on a log scale where each gridline is 10× apart. Read each line
from right to left, following backprop. The healthy ReLU + He line stays
flat. The too-small line dives, and the sigmoid line dives almost as fast.
The too-large line climbs about 22 orders of magnitude. The dashed line is the same sigmoid
network with skip connections, and it stays flat too (see residual
connections below).

$$
\text{gain per layer} \approx \sigma_w \sqrt{n_{\text{in}}}\;\cdot\;\text{typical }\phi'
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $\sigma_w$ | the standard deviation (typical size) of the random starting weights | 0.01 (too small) |
| $n_{\text{in}}$ | "fan-in": how many inputs each neuron adds up | 64 |
| $\sqrt{n_{\text{in}}}$ | a sum of $n$ random terms grows like $\sqrt{n}$, not $n$ | 8 |
| typical $\phi'$ | the average slope the activation passes back | ≈ 0.7 for ReLU's "half on" |

**In words:** "each layer multiplies the gradient by roughly the weight
size, times the square root of how many inputs it sums, times the
activation's typical slope."

**With the numbers:** too small: $0.01 \times 8 \times 0.7 \approx 0.057$
per layer, and $0.057^{30} \approx 10^{-37}$. Too large:
$1 \times 8 \times 0.7 = 5.6$ per layer, and $5.6^{30} \approx 10^{22}$.

**Why it matters:** you can't see this from the loss curve alone. A network
whose early layers get no gradient still trains a little (the late layers
learn), just badly. Plotting per-layer gradient norms is a standard
diagnostic.

## Initialization: setting every amplifier's volume

Think of a chain of 30 audio amplifiers. If each is set a little too quiet,
the sound fades to nothing; a little too loud, and it distorts into noise.
Set each so that what comes out is exactly as loud as what went in, and the
music survives the whole chain. **Initialization** picks the random
starting weights' size so that each layer passes on a signal of the same
size, forward and backward.

Worked example:

| scheme | rule for the weight standard deviation | example |
|---|---|---|
| Xavier (Glorot), for tanh/sigmoid | √(2 / (fan-in + fan-out)) | 100 in, 100 out → √(2/200) = **0.1** |
| He (Kaiming), for ReLU | √(2 / fan-in) | 50 in → √(2/50) = **0.2** |

He uses twice Xavier's variance because ReLU zeroes about half its inputs,
throwing away half the signal's energy; the factor 2 puts it back.

```mermaid
flowchart LR
  A{Activation?} -->|ReLU / GELU| He["He: std = √(2 / fan_in)"]
  A -->|tanh / sigmoid / linear| X["Xavier: std = √(2 / (fan_in + fan_out))"]
  He & X --> S[Signal keeps its size<br/>layer after layer]
```

**Reading it:** the choice of starting weights follows the activation. Both
rules have the same goal, shown in the last box: a layer should neither
shrink nor grow what passes through it.

![Size of the forward signal at every layer, for four initializations](figures/primer.ml.deep_nets.signal.svg)

**Reading it:** this is the forward direction: the typical size of the
activations entering each layer, log scale. With He initialization the ReLU
network's signal stays near 1 for all 30 layers. Too small, it fades to
nothing within a few layers; too large, it grows by about 5.6× per layer.
The backward picture above mirrors this one, because the same weights
scale both directions.

$$
\operatorname{Var}(z) = n_{\text{in}}\,\operatorname{Var}(w)\,\mathbb{E}[h^2]
\quad\Rightarrow\quad
\operatorname{Var}(w) = \frac{2}{n_{\text{in}}}\ \text{for ReLU}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $z$ | one neuron's weighted sum | |
| $\operatorname{Var}(\cdot)$ | variance: the average squared distance from the mean (standard deviation squared) | $0.2^2 = 0.04$ |
| $w$ | one weight | |
| $h$ | one input to the neuron (the previous layer's output) | |
| $\mathbb{E}[h^2]$ | the average of $h^2$ ("E" for expected value, the long-run average) | half the pre-ReLU variance |
| $n_{\text{in}}$ | fan-in | 50 |
| $\Rightarrow$ | "therefore" | |

**In words:** "the variance of a neuron's sum is the number of inputs times
the weight variance times the average squared input; ReLU halves that
average, so to keep the variance steady the weights need variance 2 over
the fan-in."

**With the numbers:** $\operatorname{Var}(w) = 2/50 = 0.04$, so the standard
deviation is $\sqrt{0.04} = 0.2$.

**Why it matters:** every framework initializes this way by default
(PyTorch's `nn.Linear` uses a Kaiming-style uniform init). Custom layers or
deep stacks built without it can silently fail to train.

## Residual connections: an express lane for the gradient

Picture a building where messages go up by stairs, one floor at a time, and
at every landing someone might mumble. Add an express lift that runs the
whole height, and the message always arrives intact; each floor adds its
own notes to what the lift carries. A **residual** (or skip) connection is
that express lift: each block computes a correction and *adds* it to its
input, instead of replacing the input.

Worked example: ten blocks, each with slope 0.025 of its own.

| | per-block factor | after 10 blocks |
|---|---|---|
| plain: h ← f(h) | 0.025 | 0.025¹⁰ ≈ 9.5 × 10⁻¹⁷ |
| residual: h ← h + f(h) | 1 + 0.025 | 1.025¹⁰ = 1.28 |

```mermaid
flowchart TD
  IN[h] --> F["block F<br/>(layers, activation)"]
  F --> ADD((+))
  IN -- "skip: identity" --> ADD
  ADD --> OUT["h + F(h)"]
```

**Reading it:** the input splits. One copy goes through the block, the
other goes straight around it, and the two are added. Going backwards, the
gradient also splits: one part flows back through the block (and may
shrink), the other flows through the "+" untouched. That untouched path is
the "1" in 1 + 0.025, and it guarantees that some gradient reaches the
early layers no matter how deep the stack is.

$$
h_{l+1} = h_l + F(h_l), \qquad
\frac{\partial h_{l+1}}{\partial h_l} = I + \frac{\partial F}{\partial h_l}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $h_l$ | the signal entering block $l$ | |
| $F$ | what the block computes (its layers and activation) | slope 0.025 |
| $\partial h_{l+1}/\partial h_l$ | how the block's output changes with its input | $1.025$ |
| $I$ | the identity: "1" for a single number, the do-nothing matrix for vectors | 1 |
| $\partial F/\partial h_l$ | the block's own slope | 0.025 |

**In words:** "the output is the input plus a correction, so its slope is
one plus the correction's slope, never just the correction's slope."

**With the numbers:** $(1 + 0.025)^{10} = 1.28$ instead of
$0.025^{10} \approx 10^{-16}$.

![Gradient reaching the input as blocks are stacked, with and without skip connections](figures/primer.ml.deep_nets.residual.svg)

**Reading it:** the x-axis counts stacked blocks and the y-axis, on a log
scale, is how much gradient survives the trip back to the input (1 means
all of it). The plain line drops by a factor of 40 with every block, a
straight plunge on a log scale, and after 10 blocks it is at 10⁻¹⁶: early
layers receive essentially nothing. The residual line stays near 1 at every
depth, because each block's "1 +" passes the gradient through intact. That
flat line is why networks with hundreds of layers can be trained at all.

**Why it matters:** residual connections (ResNet, 2015) made 100+ layer
networks trainable, and every transformer wraps both its attention and its
feed-forward sub-layers in one (see `primer.ml.transformer`). One caveat
the code shows: adding corrections forever makes the signal grow (a 30-layer
residual ReLU stack here grows its gradient about 7 million×), which is why
residuals are always paired with normalization.

## Normalization: grading on a curve

A teacher can "grade on a curve" in two ways. **Batch normalization** curves
each question across the whole class: your score on question 3 is compared
with everyone else's score on question 3, so your grade depends on who else
sat the exam. **Layer normalization** curves each student across their own
answers: your scores are rescaled relative to your own average, whoever
else is in the room. Both re-centre and re-scale numbers into a steady range
so no layer is swamped by huge or tiny values.

Worked examples:

- **BatchNorm**, batch [[1, 2], [3, 6]]: column means (2, 4), standard
  deviations (1, 2), so the output is [[−1, −1], [1, 1]]. The value 1 becomes
  −1.22 in the batch (1, 3, 5) but −0.93 in the batch (1, 3, 11).
- **LayerNorm**, one row (1, 2, 3, 4): mean 2.5, standard deviation 1.118, so
  the output is (−1.342, −0.447, 0.447, 1.342), whatever else is in the batch.
- **RMSNorm**, the same row: root-mean-square √((1+4+9+16)/4) = 2.739, so the
  output is (0.365, 0.730, 1.095, 1.461). No mean is subtracted.

```mermaid
flowchart LR
  subgraph M["activations: rows = examples, columns = features"]
    direction TB
    r1["ex 1: a  b  c  d"]
    r2["ex 2: e  f  g  h"]
    r3["ex 3: i  j  k  l"]
  end
  M -->|"down each column<br/>(across the batch)"| BN[BatchNorm]
  M -->|"along each row<br/>(within one example)"| LN[LayerNorm / RMSNorm]
```

**Reading it:** the same grid of activations can be normalized in two
directions. BatchNorm takes statistics down each column, across the
examples, so every example's output depends on its batch-mates. LayerNorm
and RMSNorm take statistics along each row, within a single example, so an
example is normalized the same way alone or in any batch.

$$
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta,
\qquad
\text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\tfrac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $x$ | one example's activations (a row) | (1, 2, 3, 4) |
| $d$ | number of features in the row | 4 |
| $x_i$ | the $i$-th feature | $x_4 = 4$ |
| $\mu$ | "mu", the row's mean | 2.5 |
| $\sigma^2$ | "sigma squared", the row's variance | 1.25 |
| $\epsilon$ | a tiny number so we never divide by zero | $10^{-5}$ |
| $\gamma, \beta$ | "gamma, beta", a learned scale and shift per feature, so the network can undo the normalization if it helps | 1 and 0 at the start |
| $\odot$ | multiply feature by feature | |

**In words:** "LayerNorm subtracts the row's mean, divides by its standard
deviation, then applies a learned scale and shift; RMSNorm skips the mean
and just divides by the root-mean-square."

**With the numbers:** LayerNorm: $(4 - 2.5)/\sqrt{1.25} = 1.5/1.118 = 1.342$.
RMSNorm: $4 / \sqrt{7.5} = 4/2.739 = 1.461$.

BatchNorm is the column version of the same formula, with $\mu$ and
$\sigma^2$ computed across the batch for each feature.

**Why it matters:** transformers use LayerNorm or RMSNorm, never BatchNorm:
sequences have different lengths, batches at inference are often size 1, and
an example's output must not depend on its batch-mates. Modern LLMs
(Llama and others) use RMSNorm because it's cheaper and works as well, and
they place it *before* each sub-layer ("pre-norm"), which keeps the residual
path clean and trains more stably. BatchNorm remains common in CNNs.

## Gradient clipping: a circuit breaker

Even with all of the above, one unlucky batch can produce a gradient spike.
A circuit breaker doesn't stop the current; it caps it. Clipping by global
norm does the same to the update: if the gradient's total length exceeds a
limit, it's scaled down to the limit, direction unchanged.

Worked example: in the exploding (too-large) 30-layer network above, the
gradients' combined length is astronomically large. Clipped with a limit of
1, the update has length exactly 1 and points the same way.

```mermaid
flowchart LR
  G[Gradients of all layers] --> N["‖g‖: combined length"]
  N --> C{"above the limit?"}
  C -->|yes| S["scale every gradient by limit / ‖g‖"]
  C -->|no| K[leave unchanged]
```

**Reading it:** measure all layers' gradients together as one long list; if
that list is longer than the limit, shrink every entry by the same factor.

$$
g \leftarrow g \cdot \min\!\left(1, \frac{c}{\lVert g \rVert}\right)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $g$ | every weight gradient, as one long list | the 30 layers' gradients |
| $\lVert g \rVert$ | its length: square every entry, add them up, take the square root | enormous |
| $c$ | the limit | 1 |

**In words:** "if the gradient is longer than the limit, rescale it to the
limit."

**With the numbers:** in the too-large network $\lVert g \rVert \approx 8 \times 10^{46}$;
with $c = 1$ every entry is multiplied by about $10^{-47}$ and the new length
is exactly 1. (The
optimizers lesson traces (3, 4) → (0.6, 0.8); see
`primer.ml.optimizers.clip_by_global_norm`, which this lesson reuses.)

**Why it matters:** clipping treats the symptom, not the cause: it makes a
rare spike harmless, but a network that explodes on every step needs better
initialization or normalization. Almost every large training run clips at a
norm of about 1.

## In 20 seconds
- Backprop multiplies one slope per layer, so gradients shrink (vanish) or
  grow (explode) exponentially with depth.
- Initialization sets weight sizes so each layer preserves signal size:
  Xavier for tanh/sigmoid, He (2 / fan-in) for ReLU.
- Residual connections add the input back, giving the gradient a path
  multiplied by 1; they're why very deep nets and transformers train.
- Normalization keeps activations in a steady range: BatchNorm across the
  batch (CNNs), LayerNorm/RMSNorm within each example (transformers).
- Gradient clipping caps rare spikes.

## Self-test questions

**Why do gradients vanish in deep sigmoid networks?**
Backprop multiplies the gradient by each layer's slope, and sigmoid's slope
is at most 0.25. Thirty layers can shrink it by 0.25³⁰, so early layers
stop learning.

**What's the difference between Xavier and He initialization?**
Both choose the weight variance so signal size is preserved. Xavier uses
2 / (fan-in + fan-out), suited to symmetric activations like tanh. He uses
2 / fan-in, doubling the variance to compensate for ReLU zeroing half its
inputs.

**How do residual connections fix vanishing gradients?**
The block's output is input + F(input), so its derivative is 1 + F′. The
gradient always has an identity path back to early layers that isn't
multiplied by small slopes.

**Why do transformers use LayerNorm instead of BatchNorm?**
BatchNorm's statistics come from the batch, which breaks for variable-length
sequences, tiny or single-example batches at inference, and makes an
example's output depend on its batch-mates. LayerNorm normalizes each token
across its own features, independent of the batch.

**What is RMSNorm and why do modern LLMs use it?**
LayerNorm without the mean subtraction and shift: divide by the
root-mean-square and apply a learned scale. It's cheaper and trains as well.

**Gradient clipping or better initialization: which fixes exploding gradients?**
Initialization (and normalization) fix the cause, keeping per-layer gain
near 1. Clipping is a safety net for occasional spikes.

## The papers behind this lesson

- Bengio, Simard & Frasconi, *Learning long-term dependencies with gradient descent is difficult* (IEEE Trans. Neural Networks, 1994): https://doi.org/10.1109/72.279181
  Proved that gradients shrink or explode exponentially through many steps, the root of the problem.
- Glorot & Bengio, *Understanding the difficulty of training deep feedforward neural networks* (AISTATS 2010): https://proceedings.mlr.press/v9/glorot10a.html
  Diagnosed saturation and derived Xavier initialization to keep variance steady across layers.
- He, Zhang, Ren & Sun, *Delving Deep into Rectifiers* (2015): https://arxiv.org/abs/1502.01852
  Derived the 2 / fan-in (He) initialization for ReLU networks.
- He, Zhang, Ren & Sun, *Deep Residual Learning for Image Recognition* (2015): https://arxiv.org/abs/1512.03385
  Introduced residual connections and trained networks over 100 layers deep. [annotated companion](../../papers/resnet.html)
- Ioffe & Szegedy, *Batch Normalization* (2015): https://arxiv.org/abs/1502.03167
  Normalized activations across the batch, allowing much higher learning rates.
- Ba, Kiros & Hinton, *Layer Normalization* (2016): https://arxiv.org/abs/1607.06450
  Normalized within each example instead, independent of batch size; the version transformers use. [annotated companion](../../papers/layer-norm.html)
- Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019): https://arxiv.org/abs/1910.07467
  Dropped LayerNorm's mean subtraction for a cheaper normalization with the same benefit.
- Xiong et al., *On Layer Normalization in the Transformer Architecture* (2020): https://arxiv.org/abs/2002.04745
  Showed why putting the norm before each sub-layer (pre-norm) trains more stably.

## Further reading
- CS231n notes, *Neural Networks Part 2* (initialization, batch norm): https://cs231n.github.io/neural-networks-2/
- Michael Nielsen, *Why are deep neural networks hard to train?*: http://neuralnetworksanddeeplearning.com/chap5.html
- Goodfellow, Bengio & Courville, *Deep Learning*, ch. 8 (optimization for training deep models): https://www.deeplearningbook.org/contents/optimization.html
- PyTorch `nn.init` docs: https://pytorch.org/docs/stable/nn.init.html
- PyTorch `nn.LayerNorm`: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
"""

from __future__ import annotations

import numpy as np

from primer._show import banner, say, table, takeaway
from primer.ml.optimizers import clip_by_global_norm

WIDTH = 64
BATCH = 32

# ---------------------------------------------------------------------------
# 1. Activations and their slopes (just what this lesson needs)
# ---------------------------------------------------------------------------


def _act(name: str, z: np.ndarray) -> np.ndarray:
    if name == "relu":
        return np.maximum(0.0, z)
    if name == "sigmoid":
        return 1 / (1 + np.exp(-z))
    if name == "tanh":
        return np.tanh(z)
    if name == "linear":
        return z
    raise ValueError(name)


def _slope(name: str, z: np.ndarray) -> np.ndarray:
    if name == "relu":
        return (z > 0).astype(float)
    if name == "sigmoid":
        s = 1 / (1 + np.exp(-z))
        return s * (1 - s)
    if name == "tanh":
        return 1 - np.tanh(z) ** 2
    if name == "linear":
        return np.ones_like(z)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# 2. The one-number chain: gradients are products of slopes
# ---------------------------------------------------------------------------


def chain_gradient(depth: int = 10, activation: str = "sigmoid", weight: float = 1.0) -> float:
    """d(output)/d(input) for a chain of one-number layers h ← act(weight·h + bias).

    Biases are set so that every unit sits at z = 0, its steepest point: the
    best case for gradient flow. Backprop multiplies one local slope per
    layer, weight × act'(z), so the answer is a product of `depth` factors.
    """
    rest = float(_act(activation, np.array(0.0)))  # the value each unit outputs at z = 0
    h, zs = 0.0, []
    for layer in range(depth):
        bias = 0.0 if layer == 0 else -weight * rest  # keeps z at 0 for every unit after the first
        z = weight * h + bias
        zs.append(z)
        h = float(_act(activation, np.array(z)))
    grad = 1.0
    for z in reversed(zs):  # the backward pass: one multiplication per layer
        grad *= weight * float(_slope(activation, np.array(z)))
    return grad


def residual_chain_gradient(depth: int = 10, block_slope: float = 0.025, residual: bool = True) -> float:
    """Gradient through `depth` blocks whose own slope is `block_slope`.

    Plain block: h ← f(h), local slope s. Residual block: h ← h + f(h), local
    slope 1 + s. The "1" is the skip connection: a path the gradient can take
    without being multiplied by anything small.
    """
    grad = 1.0
    for _ in range(depth):
        grad *= (1.0 + block_slope) if residual else block_slope
    return grad


# ---------------------------------------------------------------------------
# 3. Initialization
# ---------------------------------------------------------------------------


def init_std(kind: str, fan_in: int, fan_out: int) -> float:
    """Standard deviation for a layer's random starting weights.

    xavier (Glorot): √(2 / (fan_in + fan_out)), keeps variance steady for tanh/sigmoid-like units.
    he (Kaiming):    √(2 / fan_in), doubles the variance because ReLU zeroes half its inputs.
    tiny / large:    deliberately bad choices, to watch gradients vanish or explode.
    """
    if kind == "xavier":
        return float(np.sqrt(2.0 / (fan_in + fan_out)))
    if kind == "he":
        return float(np.sqrt(2.0 / fan_in))
    if kind == "tiny":
        return 0.01
    if kind == "large":
        return 1.0
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# 4. A deep stack, forward and backward
# ---------------------------------------------------------------------------


def _deep_forward_backward(depth: int, activation: str, init: str, residual: bool = False, seed: int = 0) -> dict:
    """Run a `depth`-layer network of width 64 forward and backward on random data.

    Each layer: z = h·W, a = act(z), next h = a (plain) or h + a (residual).
    Loss = ½·mean over the batch of ‖h_last‖². Returns per-layer activations,
    the gradient arriving at each layer's input, and each weight's gradient.
    """
    rng = np.random.default_rng(seed)
    std = init_std(init, WIDTH, WIDTH)
    Ws = [rng.normal(0.0, std, (WIDTH, WIDTH)) for _ in range(depth)]
    h = rng.standard_normal((BATCH, WIDTH))
    hs, zs = [h], []
    with np.errstate(over="ignore", invalid="ignore"):
        for W in Ws:
            z = h @ W
            a = _act(activation, z)
            h = h + a if residual else a
            zs.append(z)
            hs.append(h)
        dh = hs[-1] / BATCH  # d/dh of ½·mean‖h‖²
        dh_norms = [np.linalg.norm(dh)]
        dWs = []
        for W, z, h_in in zip(reversed(Ws), reversed(zs), reversed(hs[:-1])):
            dz = dh * _slope(activation, z)  # through the activation
            dWs.append(h_in.T @ dz)  # this layer's weight gradient
            dh = dz @ W.T + (dh if residual else 0.0)  # back through W, plus the skip path
            dh_norms.append(np.linalg.norm(dh))
    return dict(
        signal_rms=np.array([np.sqrt(np.mean(x**2)) for x in hs]),
        grad_norms=np.array(dh_norms[::-1]),  # index 0 = the first layer's input
        weight_grads=dWs[::-1],
    )


def first_to_last_gradient_ratio(depth: int = 30, activation: str = "relu", init: str = "he", residual: bool = False) -> float:
    """How big the gradient reaching the first layer is, relative to the one at the last layer."""
    g = _deep_forward_backward(depth, activation, init, residual)["grad_norms"]
    return float(g[0] / g[-2])


def gradient_norms(depth: int = 30, activation: str = "relu", init: str = "he", residual: bool = False) -> np.ndarray:
    return _deep_forward_backward(depth, activation, init, residual)["grad_norms"]


def forward_signal_rms(depth: int = 30, activation: str = "relu", init: str = "he", residual: bool = False) -> np.ndarray:
    """Root-mean-square size of the activations entering each layer (index 0 = the input)."""
    return _deep_forward_backward(depth, activation, init, residual)["signal_rms"]


def layer_gradients(depth: int = 30, activation: str = "relu", init: str = "he") -> list[np.ndarray]:
    return _deep_forward_backward(depth, activation, init)["weight_grads"]


def global_norm_after_clipping(grads: list[np.ndarray], max_norm: float = 1.0) -> float:
    clipped, _ = clip_by_global_norm(grads, max_norm)
    return float(np.sqrt(sum(np.sum(g**2) for g in clipped)))


# ---------------------------------------------------------------------------
# 5. Normalization layers
# ---------------------------------------------------------------------------


def batch_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Normalize each *feature* (column) across the examples in the batch.

    Shape (batch, features). The statistics come from the other examples, so
    an example's output depends on who it's batched with. At inference,
    running averages from training replace the batch statistics.
    """
    mean = x.mean(axis=0, keepdims=True)
    var = x.var(axis=0, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def layer_norm(x: np.ndarray, eps: float = 1e-5, gamma: np.ndarray | None = None, beta: np.ndarray | None = None) -> np.ndarray:
    """Normalize each *example* (row) across its own features. Independent of the batch.

    `gamma` and `beta` are the learned per-feature scale and shift that let
    the network undo the normalization where that helps.
    """
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    out = (x - mean) / np.sqrt(var + eps)
    if gamma is not None:
        out = out * gamma
    if beta is not None:
        out = out + beta
    return out


def rms_norm(x: np.ndarray, eps: float = 1e-6, gamma: np.ndarray | None = None) -> np.ndarray:
    """Divide each row by its root-mean-square. No mean subtraction, no shift: cheaper than LayerNorm."""
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    out = x / rms
    return out * gamma if gamma is not None else out


# ---------------------------------------------------------------------------
# 6. Figures (rendered by `make figures`)
# ---------------------------------------------------------------------------

_SETUPS = [
    ("ReLU, He init", "relu", "he", False),
    ("ReLU, weights too small", "relu", "tiny", False),
    ("ReLU, weights too large", "relu", "large", False),
    ("sigmoid, Xavier init", "sigmoid", "xavier", False),
    ("sigmoid, Xavier, residual", "sigmoid", "xavier", True),
]


def figures() -> dict:
    """Plots computed from this module's own functions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = {}
    depth = 30
    layers = np.arange(1, depth + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, act, init, res in _SETUPS:
        g = gradient_norms(depth, act, init, res)[:depth]
        ax.plot(layers, g, ls="--" if res else "-", marker=".", ms=3, label=label)
    ax.set(yscale="log", ylim=(1e-40, 1e30), xlabel="layer (1 = next to the input)",
           ylabel="gradient size reaching the layer", title="Gradient flow through 30 layers")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figs["gradient_flow"] = fig

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for label, act, init, res in _SETUPS[:4]:
        ax.plot(np.arange(0, depth + 1), forward_signal_rms(depth, act, init, res), marker=".", ms=3, label=label)
    ax.set(yscale="log", ylim=(1e-40, 1e30), xlabel="layer (0 = the input)",
           ylabel="typical activation size (RMS)", title="Forward signal through 30 layers")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figs["signal"] = fig

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    depths = np.arange(1, 21)
    ax.plot(depths, [residual_chain_gradient(int(d), 0.025, False) for d in depths], marker="o", ms=3, label="plain: × 0.025 per block")
    ax.plot(depths, [residual_chain_gradient(int(d), 0.025, True) for d in depths], marker="o", ms=3, label="residual: × 1.025 per block")
    ax.set(yscale="log", xlabel="number of blocks", ylabel="gradient reaching the input",
           title="Skip connections keep the gradient alive")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figs["residual"] = fig
    return figs


# ---------------------------------------------------------------------------
# 7. Walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. A gradient is a product of slopes")
    table(
        ["chain of 10 units", "slope per unit", "gradient at the start"],
        [
            ("sigmoid, weight 1", 0.25, chain_gradient(10, "sigmoid", 1.0)),
            ("linear, weight 1", 1.0, chain_gradient(10, "linear", 1.0)),
            ("linear, weight 1.5", 1.5, chain_gradient(10, "linear", 1.5)),
        ],
        floatfmt=".3g",
    )
    takeaway("Multiply anything below 1 by itself enough times and it vanishes; anything above 1 explodes.")

    banner("2. Watching it happen in a 30-layer network")
    table(
        ["setup", "gradient first/last layer", "signal out/in"],
        [
            (label, first_to_last_gradient_ratio(30, act, init, res),
             forward_signal_rms(30, act, init, res)[-1] / forward_signal_rms(30, act, init, res)[0])
            for label, act, init, res in _SETUPS
        ],
        floatfmt=".2e",
    )
    say(
        """
        Only ReLU with He initialization, and the residual sigmoid network,
        keep the gradient within a factor of 10 across 30 layers.
        """
    )

    banner("3. Initialization scales")
    table(["scheme", "fan-in", "fan-out", "weight std"],
          [("xavier", 100, 100, init_std("xavier", 100, 100)), ("he", 50, 50, init_std("he", 50, 50))], floatfmt=".3f")

    banner("4. Residual connections")
    table(["blocks", "plain (0.025 each)", "residual (1.025 each)"],
          [(d, residual_chain_gradient(d, 0.025, False), residual_chain_gradient(d, 0.025, True)) for d in (1, 5, 10, 20)],
          floatfmt=".3g")
    say(
        f"""
        Caveat: a residual ReLU stack with He init and no normalization grows:
        its gradient ratio over 30 layers is
        {first_to_last_gradient_ratio(30, "relu", "he", True):.1e}. That's why
        every transformer pairs its residuals with LayerNorm or RMSNorm.
        """
    )

    banner("5. BatchNorm vs. LayerNorm vs. RMSNorm")
    row = np.array([[1.0, 2.0, 3.0, 4.0]])
    table(["norm", "input (1, 2, 3, 4)"],
          [("LayerNorm", str(np.round(layer_norm(row)[0], 3))), ("RMSNorm", str(np.round(rms_norm(row)[0], 3)))])
    say(
        f"""
        BatchNorm normalizes across the batch instead: the value 1 becomes
        {batch_norm(np.array([[1.0], [3.0], [5.0]]))[0, 0]:.3f} in the batch (1, 3, 5)
        but {batch_norm(np.array([[1.0], [3.0], [11.0]]))[0, 0]:.3f} in the batch (1, 3, 11).
        """
    )

    banner("6. Clipping an exploding gradient")
    grads = layer_gradients(30, "relu", "large")
    raw = float(np.sqrt(sum(np.sum(g**2) for g in grads)))
    say(f"Combined gradient length {raw:.2e}; after clipping at 1.0: {global_norm_after_clipping(grads, 1.0):.3f}.")


if __name__ == "__main__":
    demo()
