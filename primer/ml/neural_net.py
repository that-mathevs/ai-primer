r"""
# Neural networks: neurons, activations and backprop by hand

Run: `python -m primer.ml.neural_net`

## The idea: learning is adjusting knobs to be less wrong

Picture learning to throw darts blindfolded, with a friend calling out "a bit
high, a bit left". You throw, hear how far off you were, and adjust your arm
a little. After enough throws your arm "knows" where the bullseye is. A
neural network learns the same way. Its "arm" is millions of numbers called
**weights**, and the friend is a formula (the **loss**) that says how far
off each guess was and which way to adjust.

Two words carry this whole lesson. The **derivative** (or **slope**) of the
loss with respect to a weight answers "if I nudge this weight up a tiny bit,
does the loss go up or down, and how fast?" The **gradient** is simply the
list of those slopes, one per weight. Stepping *against* the gradient is
walking downhill. (New to the notation? `primer.notation` builds every symbol
used here from zero.)

Worked example with a single knob: a model with one weight *w* predicts
y = w·x and should learn y = 3x from the example x = 1, y = 3. The loss is
(w·1 − 3)². Its slope with respect to w is 2(w − 3). Start at w = 0 and step
against the slope with step size 0.25:

| step | w | slope 2(w − 3) | new w = w − 0.25·slope |
|---|---|---|---|
| 0 | 0 | −6 | 1.5 |
| 1 | 1.5 | −3 | 2.25 |
| 2 | 2.25 | −1.5 | 2.625 |

Each step halves the distance to 3. That's all training is.

```mermaid
flowchart LR
  D[Batch of examples] --> F[Forward pass<br/>make predictions]
  F --> L[Loss<br/>how wrong were we?]
  L --> B[Backpropagation<br/>gradient per weight]
  B --> O[Optimizer step<br/>adjust weights]
  O -->|next batch| D
```

**Reading it:** follow the arrows clockwise. A batch of examples goes in; the
forward pass turns them into predictions; the loss scores how wrong those
predictions were as a single number; backpropagation works out, for every
weight, which direction would have made that number smaller; the optimizer
nudges each weight a small step that way. Then the next batch arrives. The
single-knob table above is one trip around this loop, four times.

The update rule, for every weight at once, with learning rate η:

$$
w \leftarrow w - \eta \, \frac{\partial \mathcal{L}}{\partial w}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $w$ | one weight (a knob the model can turn) | 0 at the start |
| $\leftarrow$ | "replace the old value with" (an update, not an equation) | |
| $\eta$ | "eta", the learning rate: how big a step to take | 0.25 |
| $\mathcal{L}$ | the loss: one number saying how wrong the model is | $(0 - 3)^2 = 9$ |
| $\partial \mathcal{L} / \partial w$ | the slope of the loss with respect to $w$ (the "partial derivative": how the loss changes when only $w$ moves) | $2(0 - 3) = -6$ |

**In words:** "the new weight is the old weight minus the learning rate times
the slope of the loss at the old weight."

**With the numbers:** $w \leftarrow 0 - 0.25 \times (-6) = 1.5$, the first row
of the table.

`learn_one_weight` runs the table above; `train` runs the loop for a real
network.

**Why it matters:** this loop is identical for a spam filter and for a
frontier language model. Only the size of the network, the data, and the
loss change. When a training run misbehaves, the fault is in one of these
four boxes.

## A neuron: a weighted vote followed by a decision

Think of a judge at a cooking contest. They score taste, presentation and
originality, but care about taste most, so they weight it more. They add a
baseline ("everyone starts at half a point"), then apply a rule to turn the
total into a verdict ("negative totals count as zero"). That is a neuron:
**weights** are how much each input matters, the **bias** is the baseline,
and the **activation function** is the rule.

Worked example: inputs (2, 3), weights (0.5, 1.0), bias 0.5.
Weighted sum: 2 × 0.5 + 3 × 1.0 + 0.5 = 4.5. The ReLU rule keeps positive
values, so the output is 4.5.

```mermaid
flowchart LR
  X1((x1 = 2)) -- "× w1 = 0.5" --> S["Sum + bias<br/>1.0 + 3.0 + 0.5 = 4.5"]
  X2((x2 = 3)) -- "× w2 = 1.0" --> S
  B((bias 0.5)) --> S
  S --> A["Activation φ<br/>ReLU(4.5) = 4.5"]
  A --> Y((y = 4.5))
```

**Reading it:** each input travels along an edge that multiplies it by that
edge's weight; the middle node adds the products and the bias; the
activation reshapes the sum into the output. A **layer** is this picture
repeated side by side, every input wired to every neuron, which is exactly a
matrix multiply. A **network** is layers stacked.

$$
y = \phi\left(\sum_i w_i x_i + b\right) = \phi(w \cdot x + b), \qquad
\text{a layer: } Y = \phi(XW + b)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $x_i$ | the $i$-th input | $x_1 = 2$, $x_2 = 3$ |
| $w_i$ | the weight on the $i$-th input | $w_1 = 0.5$, $w_2 = 1.0$ |
| $i$ | a counter over the inputs | 1, 2 |
| $\sum_i$ | "add up the following for every $i$" | $w_1x_1 + w_2x_2$ |
| $b$ | the bias (the baseline added to every sum) | 0.5 |
| $w \cdot x$ | the **dot product**: multiply matching entries of two lists, then add | $0.5·2 + 1.0·3 = 4.0$ |
| $\phi$ | "phi", the activation function | ReLU |
| $y$ | the neuron's output | 4.5 |
| $X$ | a batch of inputs, one example per row | shape (batch, inputs) |
| $W$ | a layer's weights, one column per neuron | shape (inputs, neurons) |
| $XW$ | **matrix multiply**: every row of $X$ dotted with every column of $W$ | shape (batch, neurons) |
| $Y$ | every neuron's output for every example | shape (batch, neurons) |

**In words:** "multiply each input by its weight, add them up, add the
bias, and pass the total through the activation function."

**With the numbers:** $y = \text{ReLU}(0.5 \cdot 2 + 1.0 \cdot 3 + 0.5) =
\text{ReLU}(4.5) = 4.5$.

**Why it matters:** every dense layer in every model is this, including the
feed-forward half of each transformer block, where most of a language
model's parameters live. `MLP.forward` is two of these layers in six lines.

## Backpropagation: passing the blame backwards

Picture an assembly line that produced a faulty product. The inspector at the
end measures how bad it is and passes the complaint back. Each station
works out how much of the fault was its own doing, based on what it received
and what it did to it, and passes the rest further back. Every station ends
up knowing exactly how to adjust its own machine. Backpropagation is that
blame-passing, done with derivatives.

Worked example, continuing the neuron: the target was 5, so the loss is
(5 − 4.5)² = 0.25.

| link | local derivative | running blame |
|---|---|---|
| loss → y | 2(y − 5) = −1 | −1 |
| y → z (ReLU, z > 0) | 1 | −1 |
| z → w1 | x1 = 2 | **−2** |
| z → w2 | x2 = 3 | **−3** |
| z → b | 1 | **−1** |

One step with learning rate 0.01 moves weights to (0.52, 1.03) and bias to
0.51. The new output is 4.64 and the loss drops from 0.25 to 0.1296. The
weight whose input was bigger (3) received the bigger share of blame and
moved more.

```mermaid
flowchart RL
  L["Loss (5 − y)² = 0.25"] -- "dL/dy = −1" --> Y[y = ReLU z]
  Y -- "dy/dz = 1" --> Z[z = w·x + b]
  Z -- "× x1 = 2 → dL/dw1 = −2" --> W1[w1]
  Z -- "× x2 = 3 → dL/dw2 = −3" --> W2[w2]
  Z -- "× 1 → dL/db = −1" --> Bb[b]
```

**Reading it:** this is the neuron read right to left. Start at the loss and
multiply the local derivatives along each path; that product is the chain
rule. All paths share the first two links (−1 × 1), then split. Each weight's
gradient is the blame arriving at the sum times the input that weight
multiplied. Negative gradient means "increase me to reduce the loss".

$$
\frac{\partial \mathcal{L}}{\partial w_i} =
\frac{\partial \mathcal{L}}{\partial y}\cdot\frac{\partial y}{\partial z}\cdot\frac{\partial z}{\partial w_i}
= 2(y - t)\cdot\phi'(z)\cdot x_i
$$

This is the **chain rule**: when a change passes through several steps, the
overall rate of change is the product of each step's rate of change. If
turning a knob moves a gear 3× as fast, and that gear moves a needle 2× as
fast, the knob moves the needle 6× as fast.

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $\mathcal{L}$ | the loss $(t - y)^2$ | 0.25 |
| $t$ | the target (the right answer) | 5 |
| $y$ | the neuron's output | 4.5 |
| $z$ | the weighted sum before the activation | 4.5 |
| $\partial \mathcal{L}/\partial y$ | how the loss changes as the output changes | $2(4.5 - 5) = -1$ |
| $\partial y/\partial z$ | how the output changes as the sum changes: the activation's slope $\phi'(z)$ | ReLU slope at 4.5 = 1 |
| $\partial z/\partial w_i$ | how the sum changes as weight $i$ changes: just its input $x_i$ | $x_2 = 3$ |
| $\phi'$ | the activation's derivative (slope) | 1 for ReLU when $z > 0$ |

**In words:** "the blame on a weight is how much the loss cares about the
output, times how much the output cares about the sum, times how much the
sum cares about that weight."

**With the numbers:** for $w_2$: $(-1) \times 1 \times 3 = -3$; for $w_1$:
$(-1) \times 1 \times 2 = -2$.

`one_neuron_worked_example` computes every row of the table.

**Why it matters:** PyTorch's `loss.backward()` does exactly this, for
billions of weights, by recording the forward computation and replaying it
backwards. Knowing it is just the chain rule is what lets you reason about
vanishing gradients, memory use, and why some architectures train and others
don't.

## Why the nonlinearity is essential

Stack three sheets of tinted glass and you get one darker sheet of glass: no
arrangement of flat panes can bend light around a corner. Linear layers are
flat panes. However many you stack, the result is one linear layer, which can
only separate classes with a straight line. The activation function is the
bend.

Worked example in one dimension: a layer that multiplies by 3, followed by a
layer that multiplies by 2, is the same as one layer that multiplies by 6.
With a ReLU between them, inputs −1 and 1 give 0 and 6. No single
multiplication does that.

```mermaid
flowchart LR
  subgraph NoAct["Without activation"]
    a1[X] --> a2[·W1] --> a3[·W2] --> a4[·W3] --> a5["= X·(W1W2W3)<br/>one matrix"]
  end
  subgraph WithAct["With activation"]
    b1[X] --> b2[·W1] --> b3[φ] --> b4[·W2] --> b5[φ] --> b6[·W3] --> b7[curved<br/>decision boundary]
  end
```

**Reading it:** in the top row the multiplications are chained with nothing
in between, and associativity lets you multiply the weights together first:
three layers equal one. In the bottom row a nonlinearity φ sits between the
multiplies, the product can no longer be pre-computed, and the network can
bend its decision boundary.

$$
(XW_1)W_2 = X(W_1W_2) \quad\text{but}\quad \phi(XW_1)W_2 \neq X W' \text{ for any } W'
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $X$ | the inputs | $x = -1$ or $x = 1$ |
| $W_1, W_2$ | two layers' weights | 3 and 2 |
| $W_1W_2$ | the two layers multiplied into one | 6 |
| $\phi$ | an activation between the layers | ReLU |
| $W'$ | any single weight you might try instead | none works |
| $\neq$ | "is not equal to" | |

**In words:** "two linear layers in a row are the same as one linear layer,
but put an activation between them and no single layer can copy them."

**With the numbers:** without ReLU, $-1 \to -3 \to -6$ and $1 \to 3 \to 6$,
exactly $6x$. With ReLU, $-1 \to -3 \to 0 \to 0$ and $1 \to 3 \to 3 \to 6$.
A single weight $W'$ would need $-W' = 0$ and $W' = 6$ at once: impossible.

![Decision boundaries: logistic regression vs. a one-hidden-layer MLP on two moons](figures/primer.ml.neural_net.decision_boundaries.svg)

**Reading it:** both panels show the same two interleaving half-moons,
coloured by class; the shaded background is what each model predicts at
every point. On the left, a single linear layer can only split the plane with
a straight line, so the tips of both moons land on the wrong side (about 88%
accuracy). On the right, one hidden tanh layer bends the boundary to follow
the gap between the moons (100%). Same data, same optimizer: the only
difference is a nonlinearity between two layers.

**In code:** `make_moons` builds the two interleaving half-moons (and `make_xor` the four-point XOR puzzle); `train_logistic_regression` fits the straight-line baseline in the left panel.

**Why it matters:** "depth" is only worth anything because of the
activations between layers. `linear_stack_collapses` shows five stacked
linear layers matching their single-matrix product to 1e-16.

## Activation functions: the rule after the sum

Think of different kinds of switch. **ReLU** is a one-way valve: water flows
freely forward, not at all backward. **Sigmoid** is a dimmer that squeezes
any input into 0–1 but barely moves at the extremes. **Tanh** is the same
dimmer centred on zero. **GELU** is a valve that leaks a little when nearly
closed.

Worked example, values you can check on a calculator (the ′ mark, as in
sigmoid′, means "the slope of"):

| z | ReLU | sigmoid | sigmoid' | tanh' | GELU |
|---|---|---|---|---|---|
| −1 | 0 | 0.269 | 0.197 | 0.420 | −0.159 |
| 0 | 0 | 0.5 | **0.25** (its maximum) | **1** (its maximum) | 0 |
| 1 | 1 | 0.731 | 0.197 | 0.420 | 0.841 |

![Activation functions and their derivatives](figures/primer.ml.neural_net.activations.svg)

**Reading it:** the left panel shows each activation's output, the right its
derivative, which is how much gradient passes back through it. Look at the
tails of the right panel first: sigmoid and tanh derivatives fall to almost
zero once |z| passes 3, so a *saturated* neuron barely learns. Sigmoid's
derivative never exceeds 0.25 even at its peak; stack ten such layers and the
gradient can shrink by 0.25¹⁰ ≈ 1e-6. ReLU's derivative is a step: exactly 1
for positive inputs (no shrinking) and exactly 0 for negatives. GELU's is a
smooth version of that step.

| Function | Formula | Range | Where it's used |
|---|---|---|---|
| ReLU | max(0, z) | [0, ∞) | Hidden layers of CNNs and MLPs. Fast. A neuron stuck negative gets zero gradient forever ("dying ReLU"). |
| GELU | z·Φ(z), Φ = normal CDF | ≈[−0.17, ∞) | Transformers (BERT, GPT-2 and most since). |
| Sigmoid | 1 / (1 + e^−z) | (0, 1) | Binary outputs, and the gates inside LSTMs. |
| Tanh | (e^z − e^−z)/(e^z + e^−z) | (−1, 1) | RNN hidden states; zero-centred. |
| Softmax | e^{z_i} / Σ e^{z_j} | probabilities | The output layer over classes or tokens, and inside attention. |

Reading the formulas: *e* is Euler's number, about 2.718, and e^z means
"2.718 raised to the power z", which is always positive and grows fast.
Φ(z) is the fraction of a bell curve (the standard normal distribution) that
lies below z: Φ(0) = 0.5, Φ(1) = 0.841. **Softmax** turns a list of scores
into shares that are all positive and add up to 1: raise *e* to each score,
then divide each by the total (Σ means "add them all up"). See
`primer.notation` for each symbol, and `primer.ml.attention` for softmax
worked through by hand.

**In code:** `relu`, `sigmoid`, `tanh` and `gelu` compute each rule, and `relu_grad`, `sigmoid_grad`, `tanh_grad` and `gelu_grad` compute its slope; `gelu_tanh` is the cheaper approximation GPT-2 uses, and `softmax` subtracts the largest score first so nothing overflows.

**Why it matters:** activation choice decides whether gradients survive a
deep stack. Sigmoid everywhere is why deep networks were hard to train
before 2010; ReLU and residual connections are why they aren't now (see
`primer.ml.deep_nets`).

## Backprop through two layers

Now the assembly line has two stations. The blame from the end first reaches
the output weights, then travels back through the hidden layer to the first
weights. On the way it passes through the hidden activation, which can
shrink it, just as a station that barely changed the product can take only a
little of the blame.

Worked example, one input, one hidden unit, one output: x = 1, w1 = 0.5,
tanh, w2 = 2, sigmoid, target 1.

| step | value |
|---|---|
| z1 = x·w1 | 0.5 |
| h = tanh(0.5) | 0.4621 |
| z2 = h·w2 | 0.9242 |
| ŷ = σ(0.9242) | 0.7159 |
| dL/dz2 = ŷ − y | −0.2841 |
| dL/dw2 = h · dL/dz2 | −0.1313 |
| dL/dh = dL/dz2 · w2 | −0.5682 |
| dL/dz1 = dL/dh · (1 − h²) | −0.5682 × 0.7864 = −0.4469 |
| dL/dw1 = x · dL/dz1 | −0.4469 |

```mermaid
flowchart LR
  subgraph Forward
    X[X] --> Z1["z1 = X·W1 + b1"] --> H["h = tanh z1"] --> Z2["z2 = h·W2 + b2"] --> P["ŷ = σ z2"] --> L["L = BCE ŷ, y"]
  end
  L -. "ŷ − y" .-> dZ2[dL/dz2]
  dZ2 -. "hᵀ · dz2" .-> dW2[dL/dW2]
  dZ2 -. "dz2 · W2ᵀ" .-> dH[dL/dh]
  dH -. "⊙ 1 − h²" .-> dZ1[dL/dz1]
  dZ1 -. "Xᵀ · dz1" .-> dW1[dL/dW1]
  H -. cached .-> dW2
  H -. cached .-> dZ1
  X -. cached .-> dW1
```

**Reading it:** the top row is the forward pass, left to right; the dotted
arrows are the backward pass, starting at the loss and walking back one box
at a time. Each label is one line of `MLP.backward`, and the worked table
above is that same path with numbers. Notice the three "cached" arrows: the
backward pass reuses h and X from the forward pass. That dependency is why
training must keep every layer's activations in memory, and inference does
not.

For a batch, with binary cross-entropy (the loss for yes/no predictions,
$-[y\ln\hat{y} + (1-y)\ln(1-\hat{y})]$; see `primer.ml.losses`), which
fuses with sigmoid into the simple error ŷ − y:

$$
\frac{\partial L}{\partial z_2} = \hat{y} - y,\;
\frac{\partial L}{\partial W_2} = h^\top \frac{\partial L}{\partial z_2},\;
\frac{\partial L}{\partial h} = \frac{\partial L}{\partial z_2} W_2^\top,\;
\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial h} \odot (1 - h^2),\;
\frac{\partial L}{\partial W_1} = X^\top \frac{\partial L}{\partial z_1}
$$

**Symbols**

| Symbol | Meaning here | In the example (one example, one unit) |
|---|---|---|
| $X$ | inputs, one row per example | $x = 1$ |
| $W_1, W_2$ | first and second layer weights | 0.5, 2 |
| $z_1, z_2$ | each layer's weighted sum, before its activation | 0.5, 0.9242 |
| $h$ | hidden activations, $\tanh(z_1)$ | 0.4621 |
| $\hat{y}$ | "y-hat", the prediction $\sigma(z_2)$ | 0.7159 |
| $y$ | the target | 1 |
| $\sigma$ | sigmoid, squashes any number into 0 to 1 | $\sigma(0.9242) = 0.7159$ |
| $\tanh$ | hyperbolic tangent, squashes into −1 to 1 | $\tanh(0.5) = 0.4621$ |
| $^\top$ | "transpose": flip a matrix so rows become columns, so the shapes line up for the multiply | a scalar is its own transpose |
| $\odot$ | multiply element by element (not a matrix multiply) | $-0.5682 \times 0.7864$ |
| $1 - h^2$ | the slope of tanh at $z_1$ | 0.7864 |

**In words:** "the error at the output is prediction minus target; each
layer's weight gradient is that layer's input times the error arriving at it;
to send the error one layer further back, multiply by the weights it came
through and by the activation's slope."

**With the numbers:** $\partial L/\partial z_2 = 0.7159 - 1 = -0.2841$;
$\partial L/\partial W_2 = 0.4621 \times -0.2841 = -0.1313$;
$\partial L/\partial h = -0.2841 \times 2 = -0.5682$;
$\partial L/\partial z_1 = -0.5682 \times 0.7864 = -0.4469$;
$\partial L/\partial W_1 = 1 \times -0.4469 = -0.4469$.

The hand-written gradients are checked two ways: a **numerical gradient
check** (nudge each weight by ±ε and measure the loss change; see
`gradient_check`) and, in the tests, PyTorch autograd.

**In code:** `tiny_two_layer_example` computes every row of the worked table; `MLP` holds both layers' weights and biases, `MLP.loss` scores a batch with `binary_cross_entropy`, and `numerical_gradient` is the slow nudge-every-weight answer that `gradient_check` compares against.

**Why it matters:** the factor (1 − h²) is where gradients shrink. Chain
fifty of them and the first layers hear almost nothing: the vanishing
gradient problem. The cached activations are why training a model needs
several times the memory of serving it.

## Batch, step, epoch

Imagine studying a deck of 400 flashcards. You work through a pile of 32,
then pause to update your notes; that pause is a **step**. The pile is a
**batch**. Going through the whole deck once is an **epoch**; then you
shuffle and start again.

Worked example: 400 examples in batches of 32 is 12 full batches plus one of
16, so ⌈400/32⌉ = 13 steps per epoch. Two epochs: 26 steps.

```mermaid
flowchart LR
  E[Epoch: one pass over all 400 examples] --> SH[Shuffle]
  SH --> B1[Batch 1<br/>32 examples] --> S1[Step 1<br/>update weights]
  S1 --> B2[Batch 2] --> S2[Step 2]
  S2 --> D[...] --> B13[Batch 13<br/>last 16 examples] --> S13[Step 13]
  S13 -->|next epoch| E
```

**Reading it:** an epoch reshuffles the data and slices it into batches;
every batch produces exactly one weight update. Shuffling each epoch means
batches differ every time, so the gradient noise doesn't repeat.

![Training loss per epoch for the MLP on two moons](figures/primer.ml.neural_net.training_curve.svg)

**Reading it:** the horizontal axis counts epochs on a log scale; the vertical
axis is the loss on the whole training set after each one. Loss drops slowly
at first while the weights are near their random start, falls steeply once
the hidden units find useful features, then flattens as the remaining errors
get harder. Plotting this curve is the first thing to do when training
anything.

$$
\text{steps per epoch} = \left\lceil \frac{N}{B} \right\rceil, \qquad
\text{total steps} = \text{epochs} \times \left\lceil \frac{N}{B} \right\rceil
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $N$ | number of training examples | 400 |
| $B$ | batch size | 32 |
| $\lceil \cdot \rceil$ | "ceiling": round up to the next whole number (the last, smaller batch still counts as a step) | $\lceil 12.5 \rceil = 13$ |

**In words:** "an epoch takes as many steps as it takes batches of size B to
cover N examples, rounding up."

**With the numbers:** $\lceil 400 / 32 \rceil = \lceil 12.5 \rceil = 13$ steps
per epoch; 2 epochs = 26 steps.

**In code:** `train` reshuffles the data every epoch, takes one plain gradient step per batch and records the loss after each epoch; `MLP.accuracy` reports the fraction of examples classified correctly.

**Why it matters:** larger batches give smoother gradient estimates and keep
GPUs busy, but need more memory. Pretraining a language model runs roughly
one epoch over trillions of tokens (it rarely sees the same text twice);
fine-tuning runs several epochs over a small dataset, which is why
fine-tunes can overfit.

## In 20 seconds
- A neuron is a weighted sum plus bias through a nonlinearity; a layer is a
  matrix multiply; a network is stacked layers.
- Without nonlinearity, depth is pointless: stacked linear layers equal one.
- Training loop: forward, loss, backprop (the chain rule gives a gradient per
  weight), optimizer step. Repeat.
- Backprop reuses forward activations, so training needs far more memory
  than inference.

## Self-test questions

**Why can't you just stack linear layers?**
Their composition is a single linear map (W1·W2 is just another matrix), so
extra layers add parameters but no expressive power.

**What does backprop actually compute?**
The gradient: for every weight, how much a tiny change would change the loss.
It applies the chain rule from the output backward, reusing values cached in
the forward pass.

**Why is ReLU preferred over sigmoid in hidden layers?**
Sigmoid's derivative is at most 0.25 and near 0 when saturated, so gradients
shrink multiplicatively with depth. ReLU's derivative is exactly 1 for
positive inputs. The cost: neurons whose input stays negative get zero
gradient and "die".

**Why does training need more memory than inference?**
The backward pass needs every layer's activations from the forward pass, so
they must be kept until the gradients are computed. Inference can discard
each activation as soon as the next layer has used it.

**What's the gradient of sigmoid + binary cross-entropy with respect to the logit?**
ŷ − y: prediction minus target. Clean, bounded and cheap, which is why the
two are always fused.

**Batch vs. step vs. epoch?**
A batch is the examples per update, a step is one update, an epoch is one
full pass over the data.

## The papers behind this lesson

- Rumelhart, Hinton & Williams, *Learning representations by back-propagating errors* (Nature, 1986): https://www.nature.com/articles/323533a0
  Showed that the chain rule, run backwards through a multi-layer network, trains hidden units to discover useful internal features.
- Hendrycks & Gimpel, *Gaussian Error Linear Units (GELUs)* (2016): https://arxiv.org/abs/1606.08415
  Introduced GELU, the smooth ReLU that became the default activation in transformers.

## Further reading
- CS231n notes, *Backpropagation, intuitions*: https://cs231n.github.io/optimization-2/
- CS231n notes, *Neural Networks Part 1*: https://cs231n.github.io/neural-networks-1/
- Andrej Karpathy, *micrograd* (autograd in about 100 lines): https://github.com/karpathy/micrograd
- Andrej Karpathy, *The spelled-out intro to neural networks and backpropagation*: https://www.youtube.com/watch?v=VMj-3S1tku0
- 3Blue1Brown, *Neural networks* series: https://www.3blue1brown.com/topics/neural-networks
- Michael Nielsen, *Neural Networks and Deep Learning*, ch. 2: http://neuralnetworksanddeeplearning.com/chap2.html
- Hendrycks & Gimpel, *GELU* (2016): https://arxiv.org/abs/1606.08415
- PyTorch autograd tutorial: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from primer._show import banner, say, table, takeaway

# ---------------------------------------------------------------------------
# 0. One knob: the whole training idea with a single weight
# ---------------------------------------------------------------------------


def learn_one_weight(steps: int = 3, lr: float = 0.25) -> list[float]:
    """Gradient descent on loss (w·1 − 3)², starting from w = 0. Returns w after each step.

    The slope is 2(w − 3); with lr 0.25 each step moves w halfway to 3.
    """
    w, history = 0.0, [0.0]
    for _ in range(steps):
        slope = 2 * (w * 1.0 - 3.0)
        w -= lr * slope
        history.append(w)
    return history


# ---------------------------------------------------------------------------
# 1. One neuron, by hand
# ---------------------------------------------------------------------------


def one_neuron_worked_example(lr: float = 0.01) -> dict:
    """Forward pass, squared-error loss, gradients, and one SGD step for one ReLU neuron.

    Inputs (2, 3), weights (0.5, 1.0), bias 0.5, target 5.
    """
    x = np.array([2.0, 3.0])
    w = np.array([0.5, 1.0])
    b = 0.5
    target = 5.0

    # Forward.
    z = w @ x + b  # weighted sum: 2*0.5 + 3*1.0 + 0.5 = 4.5
    y = relu(z)  # positive, so unchanged: 4.5
    loss = (target - y) ** 2  # 0.25

    # Backward, one chain-rule link at a time.
    dL_dy = 2 * (y - target)  # d/dy (t - y)^2 = -2 (t - y) = -1.0
    dy_dz = relu_grad(z)  # 1 because z > 0
    dL_dz = dL_dy * dy_dz
    dL_dw = dL_dz * x  # dz/dw_i = x_i, so the weight with the bigger input gets the bigger gradient
    dL_db = dL_dz  # dz/db = 1

    # SGD: step *against* the gradient.
    w_new = w - lr * dL_dw
    b_new = b - lr * dL_db
    y_new = relu(w_new @ x + b_new)
    return dict(
        z=float(z),
        y=float(y),
        loss=float(loss),
        dL_dw=dL_dw,
        dL_db=float(dL_db),
        w_new=w_new,
        b_new=float(b_new),
        y_new=float(y_new),
        loss_new=float((target - y_new) ** 2),
    )


# ---------------------------------------------------------------------------
# 2. Activation functions and their derivatives
# ---------------------------------------------------------------------------


def relu(z):
    return np.maximum(0.0, z)


def relu_grad(z):
    # The derivative at exactly 0 is undefined; frameworks use 0. It never matters in practice.
    return (np.asarray(z) > 0).astype(float)


def sigmoid(z):
    # Split by sign so exp never overflows: for z << 0 use e^z / (1 + e^z).
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1 + ez)
    return out


def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)  # peaks at 0.25 when z = 0


def tanh(z):
    return np.tanh(z)


def tanh_grad(z):
    return 1 - np.tanh(z) ** 2  # peaks at 1 when z = 0


_erf = np.vectorize(math.erf)


def gelu(z):
    """Exact GELU: z * Φ(z), where Φ is the standard normal CDF.

    Intuition: scale each input by the probability that a standard normal is
    below it. Large positives pass through, large negatives go to 0, and
    small negatives leak a little (unlike ReLU's hard cutoff).
    """
    z = np.asarray(z, dtype=float)
    return 0.5 * z * (1 + _erf(z / math.sqrt(2)))


def gelu_tanh(z):
    """The tanh approximation used by GPT-2 and many kernels."""
    z = np.asarray(z, dtype=float)
    return 0.5 * z * (1 + np.tanh(math.sqrt(2 / math.pi) * (z + 0.044715 * z**3)))


def gelu_grad(z):
    z = np.asarray(z, dtype=float)
    cdf = 0.5 * (1 + _erf(z / math.sqrt(2)))
    pdf = np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    return cdf + z * pdf


def softmax(z, axis: int = -1):
    """Stable softmax (see primer.ml.attention.softmax for the full story)."""
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


ACTIVATIONS = {
    "relu": (relu, relu_grad),
    "gelu": (gelu, gelu_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
    "tanh": (tanh, tanh_grad),
}


def tiny_two_layer_example() -> dict:
    """Backprop through x → (w1, tanh) → (w2, sigmoid) → BCE with scalars you can check by hand.

    x = 1, w1 = 0.5, w2 = 2, target 1, no biases.
    """
    x, w1, w2, y = 1.0, 0.5, 2.0, 1.0
    z1 = x * w1
    h = float(np.tanh(z1))
    z2 = h * w2
    y_hat = float(sigmoid(np.array([z2]))[0])
    dL_dz2 = y_hat - y  # sigmoid + BCE fuse to prediction − target
    dL_dw2 = h * dL_dz2  # the output weight's input was h
    dL_dh = dL_dz2 * w2  # blame sent back through w2
    dL_dz1 = dL_dh * (1 - h**2)  # shrunk by tanh's slope at z1
    dL_dw1 = x * dL_dz1
    return dict(z1=z1, h=h, z2=z2, y_hat=y_hat, dL_dz2=dL_dz2, dL_dw2=dL_dw2, dL_dh=dL_dh, dL_dz1=dL_dz1, dL_dw1=dL_dw1)


# ---------------------------------------------------------------------------
# 3. Why nonlinearity matters
# ---------------------------------------------------------------------------


def linear_stack_collapses(n_layers: int = 5, d: int = 4, seed: int = 0) -> float:
    """Return max |difference| between a deep linear stack and its single-matrix equivalent.

    It's ~1e-15 (floating-point noise): the deep stack *is* one linear layer.
    """
    rng = np.random.default_rng(seed)
    Ws = [rng.standard_normal((d, d)) / np.sqrt(d) for _ in range(n_layers)]
    X = rng.standard_normal((10, d))
    deep = X
    for W in Ws:
        deep = deep @ W
    collapsed = np.linalg.multi_dot(Ws)
    return float(np.abs(deep - X @ collapsed).max())


# ---------------------------------------------------------------------------
# 4. Toy dataset: two interleaving moons (not linearly separable)
# ---------------------------------------------------------------------------


def make_moons(n: int = 400, noise: float = 0.15, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaving half-circles. A straight line can't separate them."""
    rng = np.random.default_rng(seed)
    n1 = n // 2
    t1 = rng.uniform(0, np.pi, n1)
    t2 = rng.uniform(0, np.pi, n - n1)
    a = np.c_[np.cos(t1), np.sin(t1)]
    b = np.c_[1 - np.cos(t2), 0.5 - np.sin(t2)]
    X = np.vstack([a, b]) + rng.normal(0, noise, (n, 2))
    y = np.r_[np.zeros(n1), np.ones(n - n1)]
    return X, y


def make_xor() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0], dtype=float)
    return X, y


# ---------------------------------------------------------------------------
# 5. A two-layer MLP with hand-written backprop
# ---------------------------------------------------------------------------


def binary_cross_entropy(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Mean BCE: -[y log p + (1-y) log(1-p)]. Clip so log(0) can't happen."""
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


@dataclass
class MLP:
    """x -> tanh(x W1 + b1) -> sigmoid(h W2 + b2). Binary classifier.

    Every parameter lives in `params`; `forward` caches what `backward` needs.
    """

    n_in: int
    n_hidden: int
    seed: int = 0
    params: dict[str, np.ndarray] = field(init=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        # Xavier-style init for tanh: variance 1/fan_in keeps activations out
        # of the flat, saturated regions at the start (see primer.ml.deep_nets).
        self.params = {
            "W1": rng.normal(0, 1 / np.sqrt(self.n_in), (self.n_in, self.n_hidden)),
            "b1": np.zeros(self.n_hidden),
            "W2": rng.normal(0, 1 / np.sqrt(self.n_hidden), (self.n_hidden, 1)),
            "b2": np.zeros(1),
        }

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        p = self.params
        z1 = X @ p["W1"] + p["b1"]  # (n, hidden)
        h = np.tanh(z1)  # nonlinearity: without it this whole net is logistic regression
        z2 = h @ p["W2"] + p["b2"]  # (n, 1)
        yhat = sigmoid(z2)[:, 0]
        # Cache the intermediates: the backward pass needs X, h and yhat.
        return yhat, {"X": X, "h": h, "yhat": yhat}

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return binary_cross_entropy(self.forward(X)[0], y)

    def backward(self, cache: dict, y: np.ndarray) -> dict[str, np.ndarray]:
        """Chain rule, from the loss back to every parameter. Returns grads with the same keys as params."""
        X, h, yhat = cache["X"], cache["h"], cache["yhat"]
        n = X.shape[0]
        # Sigmoid + BCE fuse to a beautifully simple gradient: prediction minus target.
        # (Divide by n because the loss is a mean over the batch.)
        dz2 = ((yhat - y) / n)[:, None]  # (n, 1)
        dW2 = h.T @ dz2  # (hidden, 1): each weight's gradient = its input times the upstream error
        db2 = dz2.sum(axis=0)
        dh = dz2 @ self.params["W2"].T  # (n, hidden): send the error back through W2
        dz1 = dh * (1 - h**2)  # tanh'(z1) = 1 - tanh(z1)^2, reusing the cached h
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean((self.forward(X)[0] > 0.5) == y))


def numerical_gradient(model: MLP, X: np.ndarray, y: np.ndarray, eps: float = 1e-5) -> dict[str, np.ndarray]:
    """Central-difference gradient: (L(w+ε) - L(w-ε)) / 2ε for every weight.

    Slow (two forward passes per parameter) but obviously correct, which is
    what makes it the standard way to test a hand-written backward pass.
    """
    grads = {}
    for name, P in model.params.items():
        g = np.zeros_like(P)
        for idx in np.ndindex(P.shape):
            old = P[idx]
            P[idx] = old + eps
            lp = model.loss(X, y)
            P[idx] = old - eps
            lm = model.loss(X, y)
            P[idx] = old
            g[idx] = (lp - lm) / (2 * eps)
        grads[name] = g
    return grads


def gradient_check(model: MLP, X: np.ndarray, y: np.ndarray) -> float:
    """Max relative error between analytic and numerical gradients (expect < 1e-6)."""
    _, cache = model.forward(X)
    analytic = model.backward(cache, y)
    numeric = numerical_gradient(model, X, y)
    worst = 0.0
    for k in analytic:
        a, n = analytic[k], numeric[k]
        rel = np.abs(a - n) / np.maximum(1e-8, np.abs(a) + np.abs(n))
        worst = max(worst, float(rel.max()))
    return worst


# ---------------------------------------------------------------------------
# 6. The training loop: batches, steps, epochs
# ---------------------------------------------------------------------------


def train(
    model: MLP,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 0.5,
    seed: int = 0,
) -> dict:
    """Mini-batch SGD. Returns the loss history and step/epoch counts."""
    rng = np.random.default_rng(seed)
    n = len(X)
    steps = 0
    history = []
    for _epoch in range(epochs):  # one epoch = one full pass over the data
        order = rng.permutation(n)  # reshuffle each epoch so batches differ
        for start in range(0, n, batch_size):  # one batch -> one step
            idx = order[start : start + batch_size]
            _, cache = model.forward(X[idx])  # 1. forward
            grads = model.backward(cache, y[idx])  # 2-3. loss gradient via backprop
            for k in model.params:  # 4. optimizer step (plain SGD)
                model.params[k] -= lr * grads[k]
            steps += 1
        history.append(model.loss(X, y))
    return dict(history=history, steps=steps, steps_per_epoch=math.ceil(n / batch_size))


def train_logistic_regression(X: np.ndarray, y: np.ndarray, epochs: int = 300, lr: float = 0.5) -> float:
    """A single linear layer + sigmoid: the no-hidden-layer baseline. Returns accuracy."""
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        g = (p - y) / len(y)
        w -= lr * X.T @ g
        b -= lr * g.sum()
    return float(np.mean((sigmoid(X @ w + b) > 0.5) == y))


# ---------------------------------------------------------------------------
# 7. Figures (rendered by `make figures`)
# ---------------------------------------------------------------------------


def figures() -> dict:
    """Plots computed from this module's own functions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = {}

    z = np.linspace(-5, 5, 400)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, (f, g) in ACTIVATIONS.items():
        a1.plot(z, f(z), label=name)
        a2.plot(z, g(z), label=name)
    a1.set(xlabel="input z", ylabel="output φ(z)", title="Activation functions", ylim=(-1.5, 3))
    a2.set(xlabel="input z", ylabel="derivative φ'(z)", title="Their derivatives (gradient that flows back)")
    for a in (a1, a2):
        a.axhline(0, color="gray", lw=0.5)
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    figs["activations"] = fig

    X, y = make_moons()
    # Logistic regression, re-fit here so we can evaluate it on a grid.
    w, b = np.zeros(2), 0.0
    for _ in range(300):
        p = sigmoid(X @ w + b)
        g = (p - y) / len(y)
        w -= 0.5 * X.T @ g
        b -= 0.5 * g.sum()
    mlp = MLP(2, 16)
    train(mlp, X, y, epochs=300)
    xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 200), np.linspace(-1.0, 1.5, 150))
    grid = np.c_[xx.ravel(), yy.ravel()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, pred, title in (
        (axes[0], sigmoid(grid @ w + b), f"Linear (logistic regression): {train_logistic_regression(X, y):.0%}"),
        (axes[1], mlp.forward(grid)[0], f"MLP, 16 tanh hidden units: {mlp.accuracy(X, y):.0%}"),
    ):
        ax.contourf(xx, yy, pred.reshape(xx.shape), levels=[0, 0.5, 1], colors=["#cfe0f3", "#f6d5c8"])
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=8, color="C0", label="class 0")
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=8, color="C3", label="class 1")
        ax.set(title=title, xlabel="x1")
    axes[0].set_ylabel("x2")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    figs["decision_boundaries"] = fig

    hist = train(MLP(2, 16), X, y, epochs=300)["history"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(np.arange(1, len(hist) + 1), hist)
    ax.set(xscale="log", xlabel="epoch (log scale)", ylabel="training loss (BCE)", title="Training curve on two moons")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figs["training_curve"] = fig
    return figs


# ---------------------------------------------------------------------------
# 8. Walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("0. One knob: gradient descent on a single weight")
    ws = learn_one_weight(steps=5)
    table(["step", "w", "slope 2(w − 3)"], [(i, w, 2 * (w - 3)) for i, w in enumerate(ws)], floatfmt=".4f")
    say("Each step moves w halfway to 3, because the slope shrinks as the error shrinks.")
    takeaway("Training is: measure how wrong you are, find which way is downhill, take a small step. Repeat.")

    banner("1. One neuron, by hand")
    r = one_neuron_worked_example()
    say(
        f"""
        Inputs (2, 3), weights (0.5, 1.0), bias 0.5. Weighted sum
        2×0.5 + 3×1.0 + 0.5 = {r['z']}. ReLU keeps it: output {r['y']}.
        Target 5, so squared error = (5 − 4.5)² = {r['loss']}.
        """
    )
    table(
        ["quantity", "value", "why"],
        [
            ("dL/dy", 2 * (r["y"] - 5), "2(y − target): negative, so output is too low"),
            ("dL/dw1", r["dL_dw"][0], "error × input 2"),
            ("dL/dw2", r["dL_dw"][1], "error × input 3: the most influential weight"),
            ("dL/db", r["dL_db"], "error × 1"),
        ],
        floatfmt=".3f",
    )
    say(
        f"""
        One SGD step with learning rate 0.01 moves against the gradient:
        weights become ({r['w_new'][0]:.2f}, {r['w_new'][1]:.2f}), bias {r['b_new']:.2f}.
        New output {r['y_new']:.2f}, new loss {r['loss_new']:.4f} (was 0.25).
        """
    )
    takeaway("That's all training is, repeated across billions of weights and examples.")

    banner("2. Activation functions and their gradients")
    zs = np.array([-4.0, -1.0, 0.0, 1.0, 4.0])
    rows = []
    for name, (f, g) in ACTIVATIONS.items():
        rows.append((name, *[f"{v:+.2f}/{d:.2f}" for v, d in zip(f(zs), g(zs))]))
    table(["act (value/grad)"] + [f"z={z:+.0f}" for z in zs], rows)
    say(
        """
        Sigmoid's gradient never exceeds 0.25 and is ~0.02 at |z|=4:
        saturated. Tanh peaks at 1 but also flattens. ReLU's gradient is
        exactly 1 for positives and exactly 0 for negatives: a neuron pushed
        permanently negative gets no gradient and "dies". GELU is a smooth
        ReLU that leaks a little for small negatives.
        """
    )

    banner("3. Why nonlinearity is essential")
    err = linear_stack_collapses()
    say(f"Five stacked linear layers vs. their product as one matrix: max difference {err:.1e}.")
    X, y = make_moons()
    lin_acc = train_logistic_regression(X, y)
    model = MLP(2, 16)
    out = train(model, X, y, epochs=300)
    say(
        f"""
        On the two-moons data, a single linear layer (logistic regression)
        gets {lin_acc:.0%}: it can only draw a straight line. A 16-unit tanh
        hidden layer gets {model.accuracy(X, y):.0%}.
        """
    )
    takeaway("A chain of linear functions is one linear function. Nonlinearity is what makes depth useful.")

    banner("4. Backprop, verified")
    small = MLP(2, 5, seed=1)
    rel = gradient_check(small, X[:20], y[:20])
    say(
        f"""
        Hand-written chain-rule gradients vs. finite differences (nudge each
        weight ±1e-5 and re-measure the loss): max relative error {rel:.1e}.
        Anything below ~1e-6 means the backward pass is right.
        """
    )

    banner("5. Backprop through two layers, with numbers")
    t = tiny_two_layer_example()
    table(
        ["quantity", "value"],
        [("z1 = x·w1", t["z1"]), ("h = tanh(z1)", t["h"]), ("z2 = h·w2", t["z2"]), ("ŷ = σ(z2)", t["y_hat"]),
         ("dL/dz2 = ŷ − y", t["dL_dz2"]), ("dL/dw2 = h·dL/dz2", t["dL_dw2"]), ("dL/dh = dL/dz2·w2", t["dL_dh"]),
         ("dL/dz1 = dL/dh·(1 − h²)", t["dL_dz1"]), ("dL/dw1 = x·dL/dz1", t["dL_dw1"])],
    )
    say("The tanh slope (0.786) shrinks the blame on its way back. Fifty such layers and early weights hear almost nothing.")

    banner("6. Batch, step, epoch")
    say(
        f"""
        400 examples, batch size 32: {out['steps_per_epoch']} steps per epoch.
        300 epochs = {out['steps']} optimizer steps. Loss fell from
        {out['history'][0]:.3f} after epoch 1 to {out['history'][-1]:.3f}.
        """
    )
    table(["epoch", "train loss"], [(e + 1, out["history"][e]) for e in (0, 9, 49, 99, 299)])


if __name__ == "__main__":
    demo()
