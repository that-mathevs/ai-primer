r"""
# Deep networks: why deep stacks fail to train, and the fixes that made them work

Run: `python -m primer.ml.deep_nets`

New to the notation (Π, ∂, σ)? `primer.notation` builds every symbol used
here from zero. This lesson builds on backpropagation from
`primer.ml.neural_net` and the optimizer step from `primer.ml.optimizers`.

## Level 1: The practitioner's guide

**In one sentence.** Backpropagation multiplies one factor per layer, so
in a deep stack the learning signal shrinks to nothing or grows without
bound unless every layer is built to pass it on at about its original
size, and the four standard fixes (a good activation, matched
initialization, residual connections, normalization) plus a safety net
(gradient clipping) are what make every modern architecture trainable.

**When you need it.** You need this when you read a model's config and
meet `rms_norm_eps`, `initializer_range` or `layer_norm_eps`, when a
network you built stops improving while its loss curve looks merely slow,
when a training run turns NaN in its first steps, or when a paper says
"pre-norm" and you have to decide whether it matters. The tell: a model
whose late layers learn while its early layers stay at their random start,
which no loss curve shows and a plot of gradient size per layer shows at
once. You don't need it to fine-tune a published transformer: the fixes
are baked into its architecture, and your job is to leave them alone. One
number from this lesson says why they are there: in a 30-layer network of
ReLU units, weights drawn a little too small shrink the gradient reaching
the first layer by 36 orders of magnitude, a little too large grow it by
22, and the right size keeps it within a factor of about 4.

**Your options.** The fixes, from the ones a framework applies for you to
the ones that shape an architecture:

| Option | What it does | What it guarantees | What it costs | Where it lives |
|---|---|---|---|---|
| An activation with slope near 1 (ReLU exactly; GELU and SiLU for large positive inputs) | Passes the gradient through unshrunk for positive inputs, where sigmoid passes at most 0.25 (GELU and SiLU pass 0.5 at zero, rising toward 1) | No 0.25-per-layer decay: ten sigmoid layers lose a factor of a million, ten ReLU layers lose nothing | ReLU units pushed negative pass nothing and can die; GELU and SiLU leak a little instead | `hidden_act` in a config |
| Initialization matched to the activation (Xavier, He) | Sets the starting weights' size so each layer passes signal on at the same size, forward and backward | The healthy line in this lesson's figure: a factor of about 4 over 30 layers, against 10⁻³⁶ or 10²² | Nothing at runtime; a per-layer rule you must apply to custom layers | The framework's default init; `initializer_range` in a config |
| Residual connections | Each block adds its correction to its input instead of replacing it | Some gradient always reaches the early layers: 1.28 after ten blocks against 10⁻¹⁶ without | The signal grows as corrections pile up (7 million times over 30 layers here) unless normalized; block input and output must share a shape | The architecture: every transformer block |
| Normalization (BatchNorm, LayerNorm, RMSNorm) | Re-centres and rescales activations, across the batch or within each example | Activations in a steady range at every depth; with residuals, a stable stack of any depth | A mean and a variance per layer per step (RMSNorm drops the mean); BatchNorm ties each example to its batch-mates | The architecture: `layer_norm_eps`, `rms_norm_eps` |
| Gradient clipping | Rescales the whole update when its length exceeds a limit | A rare spike cannot wreck the run: a gradient of length 8 × 10⁴⁶ becomes length 1, same direction | One norm per step; a network that explodes every step is hidden, not fixed | The training loop: `max_grad_norm` |

**How to choose.** Start from whether you are reading an architecture or
building one.

- Fine-tuning a published model: read the config and change nothing. A
  Llama model is built with RMSNorm placed before each sub-layer
  (pre-norm), residuals in every block and SiLU-based activations, and its
  released config records the numbers, such as Llama 2's `rms_norm_eps` of
  10⁻⁵; the trained weights assume every one of them.
- Building a network more than a few layers deep: ReLU or GELU, the
  framework's default initialization (PyTorch's `nn.Linear` scales its
  starting weights by the fan-in), a residual path around every block, and
  a normalization layer beside it.
- Sequences, or inference one example at a time: LayerNorm or RMSNorm,
  never BatchNorm, because an example's output must not depend on who else
  is in the batch. Convolutional networks with large batches: BatchNorm,
  which ResNet places after every convolution.
- A run that spikes: clip at 1.0, the limit GPT-3 and Llama 2 trained with.
  If the clip fires on every step, the fault is initialization or
  normalization, and clipping is masking it.
- A network that trains slowly for no visible reason: plot the gradient
  norm per layer. Vanishing shows up as a slope of many orders of
  magnitude from the last layer to the first.
- Whatever you pick, the goal is one number: a per-layer factor near 1 in
  both directions. Check the forward signal and the backward gradient
  separately, because a healthy one does not prove a healthy other.

**What it costs.** Initialization is free. Residual connections cost one
addition per value, nothing beside the block's matrix multiplies, but fix
the shape of every block's output to its input. A
normalization layer costs a mean and a variance per row per layer, which
is why RMSNorm, dropping the mean, is the cheaper choice modern language
models make. Clipping costs one norm over all parameters per step. What they buy is depth itself: ResNet trained networks over 100
layers deep with residuals, a 7-billion-parameter Llama config stacks 32
blocks, and Llama 3's largest model is a dense transformer with 405 billion
parameters, none of which could be trained if the per-layer factor drifted
from 1. Depth is also what you pay for at inference: every layer runs on
every token.

**What breaks.**

- **Early layers never learn.** The gradient vanished on the way back:
  sigmoid or tanh stacked deep (18 orders of magnitude lost over 30 layers
  even with Xavier initialization), or weights initialized too small.
- **NaN in the first steps.** Weights too large (22 orders of magnitude
  of growth), or residual blocks stacked without normalization.
- **A healthy forward pass with a dead backward pass.** The sigmoid
  network's signal holds steady near 0.5 through all 30 layers while its
  gradient collapses, because the forward pass sends values through the
  activation and the backward pass multiplies by its slope. Check both.
- **BatchNorm where the batch is not a population.** The value 1 becomes
  −1.22 in one batch and −0.93 in another; at batch size 1, or with
  variable-length sequences, the statistics are meaningless. Use LayerNorm.
- **A custom layer that silently fails to train.** It skipped the
  initialization rule the framework applies to its own layers.
- **Clipping that fires every step.** Not a spike: an explosion. Fix the
  cause.
- **Post-norm instability.** Placing the norm after the residual add trains
  less stably than before it (Xiong et al., 2020); pre-norm is what Llama
  and most recent models use.

**In the wild.** Llama 2's paper describes its blocks as pre-normalization
with RMSNorm, the SwiGLU activation and rotary position embeddings, and
Hugging Face's `LlamaConfig` exposes the settings (`initializer_range`
0.02, `num_hidden_layers` 32, `hidden_act` silu, and an `rms_norm_eps` that
defaults to 1e-6 while Meta's Llama 2 code and released config use 1e-5).
PyTorch's `nn.LayerNorm` takes the shape to normalize over with
`eps=1e-05` and a learned per-element scale and shift; its `nn.Linear`
initializes from a uniform range set by the fan-in; and
`torch.nn.utils.clip_grad_norm_` clips by the norm over all parameters
together, which Hugging Face's `TrainingArguments` calls with a default
`max_grad_norm` of 1.0, the same limit Llama 2 trained with. The fixes are
He et al. (ResNet and He initialization, 2015), Glorot and Bengio (Xavier,
2010), Ioffe and Szegedy (BatchNorm, 2015), Ba, Kiros and Hinton
(LayerNorm, 2016) and Zhang and Sennrich (RMSNorm, 2019), with the problem
itself diagnosed by Bengio, Simard and Frasconi (1994); all are linked at
the end of the lesson.

**Go deeper.** Level 2 multiplies the slopes of a ten-layer chain by hand,
watches the gradient at every layer of a 30-layer network under four
initializations, derives the Xavier and He rules from one variance
equation, shows the "1 +" that residual connections add, normalizes one
row three ways with the numbers shown, and clips an exploding gradient of
length 8 × 10⁴⁶ down to 1. If you only needed to read a config, you are
done.

## Level 2: How it works, from scratch

The problem shows up in the smallest deep network there is, a chain of
ten layers that each hold one number, so it is worked there first, by
hand. The fixes are then measured on a network of realistic shape: 30
layers of 64 neurons.

## The gradient is a product of slopes

Training sends a learning signal backward from the loss to every weight:
the **gradient**, how much each weight should change (see
`primer.ml.neural_net`). It travels one layer at a time, and each layer
multiplies it by a number of its own, the layer's weight times its
activation's **slope** (derivative). Think of a message passed down a line
of ten people, each repeating it at a quarter of the volume they heard.
After ten people it is down to about a millionth of its first volume. If
each speaks 1.5 times louder instead, it ends up 58 times louder. A factor
below 1 at every layer drives the gradient toward zero, the **vanishing
gradient**, and a factor above 1 drives it up without limit, the
**exploding gradient**.

Here are three chains of ten one-number layers, each unit at its steepest
point.

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

**Reading it:** read right to left, the direction the gradient travels. The
loss hands layer 10 a gradient of 1, and every arrow multiplies it by 0.25,
a sigmoid's slope at its steepest. Layer 1 receives about a millionth of it,
so its weights barely move: the first layers stop learning while the later
ones carry on.

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

**In Python:**

```python
def gradient_at_input(w_l, slope, L=10):
    # ∂L/∂h_L: the loss hands the top layer 1
    grad = 1.0
    # Π over the layers: a running product
    for l in range(L):
        # × w_l φ'(z_l)
        grad *= w_l * slope
    return grad
# sigmoid at its steepest
f"{gradient_at_input(1, 0.25):.1e}"  # → '9.5e-07'
# linear, weight 1
gradient_at_input(1, 1)  # → 1.0
# linear, weight 1.5
round(gradient_at_input(1.5, 1), 1)  # → 57.7
```

**In code:** `chain_gradient` builds the chain and passes the gradient back through it.

Every fix that follows works the same way: it holds the per-layer factor
near 1.

## The gradient in a 30-layer network

In the chain each layer had one weight. In a real layer each neuron adds up
many inputs, here 64, so the factor per layer depends on three things
together: the size of the weights, how many inputs each neuron adds up, and
the activation's slope. A 30-layer network with 64 neurons per layer shows
what that does. The table gives the gradient at the first layer divided by
the gradient at the last, for ReLU with weights of three sizes (the right
size is the He rule, which the next chapter derives) and for sigmoid:

| setup | ratio first / last |
|---|---|
| ReLU, He initialization | ≈ 4 (healthy) |
| ReLU, weights too small (std 0.01) | ≈ 10⁻³⁶ (vanished) |
| ReLU, weights too large (std 1) | ≈ 10²² (exploded) |
| sigmoid, Xavier initialization | ≈ 10⁻¹⁸ (vanished) |

![Gradient size at every layer, relative to layer 30: ReLU with He stays near 1, too-small weights dive 36 orders of magnitude, sigmoid dives 18, too-large weights climb 22, and sigmoid with skip connections stays flat](figures/primer.ml.deep_nets.gradient_flow.svg)

**Reading it:** the horizontal axis is the layer (1 is next to the input, 30
next to the loss). The vertical axis is the gradient reaching that layer
divided by the gradient at layer 30, on a log scale where each gridline is a
factor of 10⁶. Every line starts at 1 on the right; read it right to left,
the way the gradient travels, and its height at layer 1 is the ratio in the
table. The ReLU + He line stays within a factor of about 5 of 1. The
too-small line dives about 36 orders of magnitude and the too-large line
climbs about 22. The sigmoid line dives 18 orders: Xavier keeps the
weights' own gain near 1, so what shrinks the gradient is sigmoid's slope,
at most 0.25, about 4× per layer. The dashed line is the same sigmoid
network with skip connections, which stays flat; residual connections
explain why.

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

**With the numbers:** too small: $0.01 \times 8 \times 0.7 = 0.056$
per layer, and $0.056^{30} \approx 3 \times 10^{-38}$. Too large:
$1 \times 8 \times 0.7 = 5.6$ per layer, and $5.6^{30} \approx 3 \times 10^{22}$.

**In Python:**

```python
import math
n_in, typical_slope = 64, 0.7
for sigma_w in (0.01, 1.0):
    # σ_w √n_in · typical φ'
    gain = sigma_w * math.sqrt(n_in) * typical_slope
    # per layer, then over 30 layers
    print(round(gain, 3), f"{gain ** 30:.0e}")  # → 0.056 3e-38 5.6 3e+22
```

**In code:** `gradient_norms` measures the gradient at every layer, and `first_to_last_gradient_ratio` fills the table.

The loss curve does not show any of this. A network whose early layers get
no gradient still trains a little, because its late layers learn, so its
loss falls slowly instead of stopping. A plot of the gradient's size at
each layer shows the problem at once.

## Initialization

With ReLU and nothing else changed, the size of the starting weights alone
took the ratio in that table from 10⁻³⁶ to 10²². **Initialization** is the
rule that picks the size: the random starting weights are drawn at a scale
where each layer passes on a signal of the same size, forward and backward.
Two rules cover the common activations. Each works from the layer's fan-in
and, for Xavier, its fan-out (how many outputs the layer has):

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

**Reading it:** start from the activation on the left. ReLU and GELU take the
He rule; tanh, sigmoid and linear units take Xavier. Both arrows end in the
same box, because both rules aim at a layer that neither shrinks nor grows
what passes through it.

![Forward signal size at every layer: ReLU with He stays near 1, too-small weights fade to 10⁻³⁸, too-large weights grow to 10²², and sigmoid holds flat near 0.5 even though its gradient vanishes](figures/primer.ml.deep_nets.signal.svg)

**Reading it:** this chart shows the forward direction: the typical size of
the values entering each layer, on a log scale. With He initialization the
ReLU network's signal stays near 1 for all 30 layers. With weights too
small it fades to nothing within a few layers, and with weights too large
it grows by about 5.6× per layer. For the three ReLU lines the gradient
chart mirrors this one, because the same weights scale both directions.
The sigmoid line breaks the mirror. Its signal holds steady near 0.5 for
all 30 layers, since sigmoid's outputs sit around 0.5 whatever comes in,
yet its gradient lost 18 orders of magnitude. The forward pass only sends
values through sigmoid, while the backward pass multiplies by sigmoid's
slope, at most 0.25, at every layer. A healthy forward signal does not
prove a healthy gradient, so check both.

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

**In Python:**

```python
import math
n_in = 50
# Var(w) = 2 / n_in, for ReLU
var_w = 2 / n_in
# the variance, then the standard deviation
var_w, round(math.sqrt(var_w), 3)  # → (0.04, 0.2)
# Xavier for comparison: 100 in, 100 out
round(math.sqrt(2 / (100 + 100)), 3)  # → 0.1
```

**In code:** `init_std` gives each rule's standard deviation, and `forward_signal_rms` measures the forward signal.

Frameworks apply a rule like this to their own layers: PyTorch's
`nn.Linear` draws its starting weights from a range scaled by the fan-in (a
Kaiming-style uniform init). A layer you write yourself gets no such rule
unless you add it, and without it a deep stack can fail to train with no
error to say why.

## Residual connections

Initialization sets the weights' size, but it cannot change the
activation's slope: the sigmoid network with Xavier weights still lost 18
orders of magnitude. A **residual connection** (or skip connection) gets
around the slope. A block of layers computes a correction and *adds* it to
its input, instead of replacing the input. On the way back, the gradient
then has a path around every block that multiplies it by 1, whatever the
block's own slope. That path is what keeps the dashed sigmoid line in the
gradient chart flat.

Take ten blocks, each with a slope of 0.025 of its own:

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

**Reading it:** follow the input from the top. One copy goes through the
block, the other goes straight around it, and the two are added. Going
backwards, the gradient splits the same way: one part flows back through
the block and may shrink, the other passes through the "+" untouched. The
untouched part is the "1" in 1 + 0.025, and it means some gradient reaches
the early layers however deep the stack is.

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

**In Python:**

```python
# each block's own slope
dF_dh = 0.025
plain, residual = 1.0, 1.0
for l in range(10):
    # h ← F(h): the slope is just ∂F/∂h
    plain *= dF_dh
    # h ← h + F(h): the slope is I + ∂F/∂h
    residual *= 1 + dF_dh
f"{plain:.1e}", round(residual, 2)  # → ('9.5e-17', 1.28)
```

![Gradient reaching the input as blocks are stacked: without skip connections it falls 40× per block to 10⁻¹⁶ after 10 blocks, with them it stays near 1](figures/primer.ml.deep_nets.residual.svg)

**Reading it:** the horizontal axis counts stacked blocks, and the vertical
axis, on a log scale, is how much gradient survives the trip back to the
input (1 means all of it). The plain line drops by a factor of 40 with
every block, a straight line down on a log scale, and after 10 blocks it is
at 10⁻¹⁶, so the early layers receive almost nothing. The residual line
stays near 1 at every depth, because each block's "1 +" passes the gradient
through.

**In code:** `residual_chain_gradient` multiplies the per-block factors, with or without the skip path.

ResNet (2015) used residual connections to train networks over 100 layers
deep, and every transformer wraps both its attention and its feed-forward
sub-layers in one (see `primer.ml.transformer`). The skip path has a cost.
Each block adds its correction on top of what it received, so the signal
grows with depth: a 30-layer residual ReLU stack here grows its gradient
about 7 million times.

## Normalization

That growth is why residual connections come paired with
**normalization**. A normalization layer re-centres and rescales the
numbers passing through it into a steady range, so no layer is swamped by
huge or tiny values, however many corrections have piled up. The kinds
differ in which
numbers each value is compared with, the way a teacher can grade on a curve
in two ways. **Batch normalization** (BatchNorm) curves each question across
the whole class: your score on question 3 is compared with everyone else's
score on question 3, so your grade depends on who else sat the exam.
**Layer normalization** (LayerNorm) curves each student across their own
answers, so your grade depends only on you. **RMSNorm** is LayerNorm without
the step that subtracts the average.

All three, on numbers small enough to check:

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

**In Python:**

```python
import math
x = [1, 2, 3, 4]
d, eps, gamma, beta = len(x), 1e-5, 1.0, 0.0
# the row's mean
mu = sum(x) / d
# σ², the row's variance
var = sum((x_i - mu) ** 2 for x_i in x) / d
mu, var  # → (2.5, 1.25)
# LayerNorm
[round(gamma * (x_i - mu) / math.sqrt(var + eps) + beta, 3) for x_i in x]  # → [-1.342, -0.447, 0.447, 1.342]
# √((1/d) Σ x_i² + ε)
rms = math.sqrt(sum(x_i ** 2 for x_i in x) / d + eps)
round(rms, 3)  # → 2.739
# RMSNorm: no mean subtracted
[round(gamma * x_i / rms, 3) for x_i in x]  # → [0.365, 0.73, 1.095, 1.461]
```

**In code:** `batch_norm`, `layer_norm` and `rms_norm` compute the three examples.

Transformers use LayerNorm or RMSNorm rather than BatchNorm. Their
sequences have different lengths, a batch at inference is often a single
example, and an example's output must not depend on its batch-mates.
Llama and other recent language models use RMSNorm, which is cheaper and
works as well, and place it *before* each sub-layer ("pre-norm"), which
keeps the residual path clean and trains more stably. BatchNorm is used in
convolutional networks trained on large batches: ResNet puts one after
every convolution.

## Gradient clipping

Initialization, residual connections and normalization keep the per-layer
factor near 1, but one unusual batch can still produce a spike: a gradient
far larger than usual, and a weight update to match. **Gradient clipping**
caps the update. It measures all the gradients in the network together as
one length (their **global norm**), and when that length is over a set
limit it scales every gradient down by the same factor, so the update keeps
its direction and loses only its size.

The network with weights too large, from the 30-layer chart, has gradients
whose combined length is about 8 × 10⁴⁶. Clipped at a limit of 1, the
update has length 1 and points the same way.

```mermaid
flowchart LR
  G[Gradients of all layers] --> N["‖g‖: combined length"]
  N --> C{"above the limit?"}
  C -->|yes| S["scale every gradient by limit / ‖g‖"]
  C -->|no| K[leave unchanged]
```

**Reading it:** follow the arrows from the left. All the layers' gradients
are measured together as one long list; if the list is longer than the
limit, every entry is scaled by the same factor, and otherwise nothing
changes.

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
is exactly 1. For a case small enough to check by hand, see
`primer.ml.optimizers`, where (3, 4) is clipped to (0.6, 0.8).

**In Python:**

```python
import math
# a stand-in with the same enormous length
g = [4.8e46, 6.4e46]
# ‖g‖
norm = math.sqrt(sum(g_i ** 2 for g_i in g))
f"{norm:.0e}"  # → '8e+46'
c = 1
# min(1, c / ‖g‖)
scale = min(1, c / norm)
f"{scale:.0e}"  # → '1e-47'
# the new length
round(math.sqrt(sum((g_i * scale) ** 2 for g_i in g)), 6)  # → 1.0
```

**In code:** `layer_gradients` collects the too-large network's gradients, and `global_norm_after_clipping` clips them.

Clipping makes a rare spike harmless. It does not fix a network that
explodes on every step: that needs better initialization or normalization,
and clipping would only hide it. GPT-3 and Llama 2 both trained with a
clipping limit of 1.0.

## In 20 seconds
- Passing the gradient back multiplies it by one factor per layer, so with
  depth it shrinks (vanishes) or grows (explodes) exponentially.
- Initialization sizes the starting weights so each layer passes the signal
  on at the same size: Xavier for tanh and sigmoid, He (2 / fan-in) for ReLU.
- Residual connections add each block's input to its output, which gives
  the gradient a path back that is multiplied by 1.
- Normalization rescales each layer's values into a steady range: BatchNorm
  across the batch, in convolutional networks, and LayerNorm or RMSNorm
  within each example, in transformers.
- Gradient clipping scales a rare oversized gradient down to a set length
  without changing its direction.

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
        # Relative to layer 30, so every line starts at 1 on the right and its height at layer 1
        # is the first/last ratio in the lesson's table (raw sizes span 10⁻⁷⁵ to 10⁴⁶).
        ax.plot(layers, g / g[-1], ls="--" if res else "-", marker=".", ms=3, label=label)
    ax.set(yscale="log", ylim=(1e-38, 1e26), yticks=[10.0**k for k in range(-36, 25, 6)],
           xlabel="layer (1 = next to the input)",
           ylabel="gradient size ÷ size at layer 30", title="Gradient flow through 30 layers")
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
