r"""
# Regularization: learning the pattern instead of memorising the examples

Run: `python -m primer.ml.regularization`

## The idea: the student who memorised last year's exam

Two students prepare for an exam. One learns the ideas. The other memorises
last year's paper, answer by answer, and scores 100% on it in practice. On
the real exam, with new questions, the first student does fine and the
second falls apart. A model that memorises its training examples, noise
and all, instead of learning the pattern behind them is **overfitting**. It
looks brilliant on the data it has seen and fails on data it hasn't.
**Regularization** is everything we do to push a model toward the first
student.

Worked example: four training points (0, 0), (1, 1), (2, 0), (3, 1), a
zig-zag that is really "about 0.5, plus noise".

| model | error on the 4 training points | prediction at x = 4 |
|---|---|---|
| flat line at 0.5 (the pattern) | 0.25 | 0.5 |
| cubic through all four points (memorised) | **0** | **8** |

The cubic is perfect on the training data and absurd one step beyond it.
You can check the 8 by hand with a difference table: the values 0, 1, 0, 1
have differences 1, −1, 1, then −2, 2, then 4; a cubic keeps that last
difference constant, so extending the table gives 6, then 7, then 8.

```mermaid
flowchart LR
  D["Training data =<br/>pattern + noise"] --> M{Model capacity}
  M -->|too little| U["Underfits:<br/>misses the pattern"]
  M -->|about right| G["Generalizes:<br/>learns the pattern"]
  M -->|too much, unchecked| O["Overfits:<br/>memorises the noise"]
  G --> N[Good on new data]
  U & O --> B[Bad on new data]
```

**Reading it:** every dataset is a pattern plus noise. A model with too
little capacity can't even represent the pattern; one with too much, left
unchecked, fits the noise too. Only the middle path does well on new data.
Regularization techniques are ways of steering a high-capacity model onto
that middle path without giving up its capacity.

![Three polynomial fits to 12 noisy sine samples: the degree-1 line misses the curve, the cubic follows it, and the degree-11 polynomial hits every dot but swings wildly between them](figures/primer.ml.regularization.fits.svg)

**Reading it:** the dots are 12 noisy samples of the grey sine curve (the
true pattern). The straight line (degree 1) can't bend and misses the shape
entirely: underfitting. The cubic (degree 3) follows the sine closely. The
degree-11 polynomial passes through every dot exactly, and between and
beyond them it swings wildly: it has memorised the noise.

The two numbers that tell these apart are errors on two different sets of
data:

$$
\text{MSE}_{\text{train}} = \frac{1}{n}\sum_{i \in \text{train}} (y_i - \hat{y}_i)^2,
\qquad
\text{MSE}_{\text{val}} = \frac{1}{m}\sum_{j \in \text{val}} (y_j - \hat{y}_j)^2
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $n, m$ | how many training and validation examples | 4 training points |
| $i \in \text{train}$ | "for every example $i$ in the training set" | $x = 0, 1, 2, 3$ |
| $y_i$ | the true value | 0, 1, 0, 1 |
| $\hat{y}_i$ | the model's prediction | 0.5 everywhere for the flat line |
| $\text{MSE}$ | mean squared error (see `primer.ml.losses`) | 0.25 for the flat line |

**In words:** "training error is the average squared miss on the data the
model learned from; validation error is the same average on data it has
never seen."

**With the numbers:** flat line: $\frac{1}{4}(0.5^2 + 0.5^2 + 0.5^2 + 0.5^2) = 0.25$.
Cubic: every miss is 0, so training error is 0; its validation error on any
point beyond x = 3 is enormous.

**In Python:**

```python
# the true values at x = 0, 1, 2, 3
y = [0, 1, 0, 1]
# the flat line predicts 0.5 everywhere
y_hat = [0.5, 0.5, 0.5, 0.5]
n = len(y)
# (1/n) Σ (y_i - ŷ_i)²
sum((y_i - y_hat_i) ** 2 for y_i, y_hat_i in zip(y, y_hat)) / n  # → 0.25
```

**In code:** `zigzag_fit_error` and `zigzag_predict` fit a polynomial of any
degree to the four zig-zag points and report its training error and its
prediction; `true_function` and `make_curve_data` draw the noisy sine
samples in the figure.

**Why it matters:** training error alone always rewards memorisation. The
gap between training and validation error is the single most useful
diagnostic in machine learning.

## Underfitting vs. overfitting: finding the middle

Goldilocks tries three bowls of porridge: too cold, too hot, just right.
Model capacity works the same way, and you find "just right" by measuring,
not guessing: sweep the capacity and watch both errors.

Worked example: 12 noisy points from a sine wave, 300 fresh points for
validation, polynomials of increasing degree.

| degree | training error | validation error | diagnosis |
|---|---|---|---|
| 1 | 0.20 | 0.30 | underfit: both high |
| 3 | 0.021 | 0.065 | about right |
| 5 | 0.013 | 0.057 | best: lowest validation error |
| 9 | 0.004 | 0.71 | overfit: validation 12× worse than at degree 5 |
| 11 | 0.0000 | 7,631 | memorised |

![Error against polynomial degree on a log scale: training error only falls, while validation error is lowest at degree 5 and then climbs steeply](figures/primer.ml.regularization.degree_sweep.svg)

**Reading it:** the horizontal axis is model capacity (polynomial degree);
the vertical axis is error on a log scale. The training curve (blue) only
ever goes down: more capacity always fits the training data better. The
validation curve (orange) falls, is nearly flat from degree 3 to 5 (lowest
at 5), then climbs from degree 6 onward and shoots up. The left side of that valley is underfitting, the right side
overfitting; the bottom is the model you want.

**In code:** `polynomial_errors` fits by least squares (choosing the
coefficients that minimise training MSE) and reports both errors.

**Why it matters:** "both errors high" and "training low, validation high"
need opposite fixes. Underfitting wants more capacity, more features or more
training; overfitting wants more data, regularization or early stopping.
Diagnosing which one you have is the point.

## Early stopping: take the cake out before it burns

A baker checks the cake with a toothpick every few minutes and takes it out
when it comes out clean. Leave it longer and it burns, however good it was
a minute ago. Training a flexible model is similar: validation error falls
while the model learns the pattern, then rises once it starts memorising.
**Early stopping** watches validation error during training, stops once it
has failed to improve for a few checks (the **patience**), and keeps the
weights from the best check.

Worked example: validation losses per epoch 1.0, 0.8, 0.6, **0.55**, 0.58,
0.65, 0.7 with patience 2. Epoch 3 is the best. Epochs 4 and 5 are both
worse, so training stops at epoch 5 and the weights from epoch 3 are kept.

```mermaid
flowchart LR
  E[End of epoch] --> V[Measure validation loss]
  V --> Q{Better than best?}
  Q -->|yes| S[Save weights,<br/>reset counter]
  Q -->|no| C[Counter + 1]
  C --> P{Counter = patience?}
  P -->|no| E2[Next epoch]
  P -->|yes| R[Stop; restore<br/>best weights]
  S --> E2
```

**Reading it:** after every epoch the loop asks one question: is this the
best validation loss so far? If yes, snapshot the weights. If not, count a
strike. Reaching the patience limit ends training and restores the
snapshot, so you always ship the best model seen, not the last one.

![Loss curves for an over-sized network: training loss keeps falling while validation loss bottoms out near epoch 300 and then rises](figures/primer.ml.regularization.learning_curves.svg)

**Reading it:** a 64-unit network trained on only 30 noisy points. Both
curves fall at first. Around epoch 300 (the dashed line) the validation
curve bottoms out and turns upward while the training curve keeps falling:
the network has started fitting individual noisy points. Everything to the
right of the dashed line is wasted, or worse. Early stopping keeps the
weights at the dashed line.

$$
t^\star = \arg\min_{t} \; \mathcal{L}_{\text{val}}(t),
\qquad \text{stop at the first } t \text{ with } t - t^\star \ge \text{patience}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $t$ | the epoch number | 0 to 6 |
| $\mathcal{L}_{\text{val}}(t)$ | validation loss after epoch $t$ | 0.55 at $t = 3$ |
| $\arg\min_t$ | "the $t$ at which the following is smallest" (the position, not the value) | 3 |
| $t^\star$ | "t-star", the best epoch so far | 3 |
| patience | how many non-improving epochs to tolerate | 2 |

**In words:** "remember the epoch with the lowest validation loss, and stop
once you've gone `patience` epochs past it without doing better."

**With the numbers:** $t^\star = 3$; at $t = 5$, $5 - 3 = 2 \ge 2$, so stop.

**In Python:**

```python
# validation loss after epoch t = 0, 1, 2, ...
L_val = [1.0, 0.8, 0.6, 0.55, 0.58, 0.65, 0.7]
patience = 2
t_star = 0
for t, loss in enumerate(L_val):
    if loss < L_val[t_star]:
        # a new best epoch: remember it
        t_star = t
    if t - t_star >= patience:
        # patience used up: stop here
        break
t_star, t  # → (3, 5)
```

**In code:** `early_stopping` replays a run's validation losses and returns
the best epoch and the epoch where training stops; `train_flexible_model`
trains the over-sized network in the figure and records both losses.

**Why it matters:** it's the cheapest regularizer there is: no change to the
model, just a rule for when to stop. It's used almost everywhere a model is
trained for multiple epochs, including fine-tuning language models.

## Bias and variance: two ways to miss the target

Picture two archers. The first groups every arrow tightly, but a hand's
width to the left of the bullseye: consistently wrong. That's **bias**. The
second's arrows are centred on the bullseye on average but scattered all
over the target: inconsistent. That's **variance**. A model trained on a
different sample of data is like another volley of arrows. Simple models
behave like the first archer; very flexible ones like the second.

Worked example: fit polynomials to 300 different random training sets of 30
points each, and measure (averaged over the input range):

| degree | bias² (systematic error) | variance (swing between training sets) |
|---|---|---|
| 1 | 0.174 | 0.022 |
| 3 | 0.0035 | 0.015 |
| 9 | 0.0038 | 4.5 |

![Twenty fits from different training sets: degree-1 lines agree but all miss the sine (high bias), degree-9 curves average to the sine but scatter widely (high variance)](figures/primer.ml.regularization.bias_variance.svg)

**Reading it:** each thin line is the model fitted to one random training
set; the thick grey curve is the truth. On the left (degree 1), the lines
agree with each other but all miss the sine's shape in the same way: high
bias, low variance. On the right (degree 9), the lines follow the sine on
average, but each one wiggles differently, especially near the edges: low
bias, high variance.

$$
\mathbb{E}\big[(y - \hat{f}(x))^2\big] = \underbrace{\big(\mathbb{E}[\hat{f}(x)] - f(x)\big)^2}_{\text{bias}^2}
+ \underbrace{\mathbb{E}\big[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2\big]}_{\text{variance}}
+ \underbrace{\sigma^2}_{\text{noise}}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $f(x)$ | the true pattern | $\sin(\pi x)$ |
| $\hat{f}(x)$ | "f-hat", the model's prediction, which depends on which training set it saw | one thin line |
| $\mathbb{E}[\cdot]$ | expected value: the average over many random training sets (and noise) | average of the 300 fits |
| $y$ | a noisy observation, $f(x)$ plus noise | |
| $\sigma^2$ | the noise variance: error no model can remove | $0.3^2 = 0.09$ |

**In words:** "a model's expected squared error on new data splits into how
far its average prediction is from the truth (bias squared), plus how much
its predictions scatter around their own average (variance), plus noise
nobody can predict."

**With the numbers:** degree 1: $0.174 + 0.022 + 0.09 \approx 0.29$;
degree 3: $0.0035 + 0.015 + 0.09 \approx 0.11$; degree 9:
$0.0038 + 4.5 + 0.09 \approx 4.6$. Degree 3 has the best total.

**In Python:**

```python
sigma = 0.3
# σ²: the error no model can remove
noise = sigma ** 2
for degree, bias_sq, variance in [(1, 0.174, 0.022), (3, 0.0035, 0.015), (9, 0.0038, 4.5)]:
    # bias² + variance + σ²
    total = bias_sq + variance + noise
    # two significant figures
    print(degree, f"{total:.2g}")  # → 1 0.29 3 0.11 9 4.6
```

**In code:** `bias_variance` fits one polynomial to each of many random
training sets and splits their error into bias², variance and noise.

**Why it matters:** it names the two failure modes and explains why more
data helps flexible models (averaging tames variance) but not rigid ones
(more data doesn't fix bias). Very large neural networks complicate the
picture ("double descent": past a point, even bigger models generalize
better again), but the vocabulary is universal.

## Dropout: nobody gets to be indispensable

A coach who randomly sends half the team home before each practice forces
every player to learn every position; no single star can carry the team.
**Dropout** does this to neurons: during training, each activation is set
to zero with probability p on every step, so the network can't rely on any
single neuron or fragile combination of neurons.

Worked example: four activations (1, 1, 1, 1), drop rate p = 0.5. Suppose
the random mask keeps the last two: (0, 0, 1, 1). The survivors are scaled by
1 / (1 − 0.5) = 2, giving (0, 0, 2, 2). The average is still 1, so the next
layer sees the same size signal on average. At evaluation time, nothing is
dropped and nothing is scaled.

```mermaid
flowchart LR
  subgraph Train["Training step"]
    a1((1)) --> k1((0))
    a2((1)) --> k2((0))
    a3((1)) --> k3((2))
    a4((1)) --> k4((2))
  end
  subgraph Eval["Evaluation"]
    b1((1)) --> e1((1))
    b2((1)) --> e2((1))
    b3((1)) --> e3((1))
    b4((1)) --> e4((1))
  end
```

**Reading it:** on the left, one training step: two of the four units were
dropped to 0 and the two survivors were doubled, so the total (4) matches
what the full layer would have sent. A different random pair is dropped on
the next step. On the right, at evaluation time every unit passes through
unchanged. Scaling during training ("inverted dropout") is what lets
evaluation be a plain pass-through.

$$
\tilde{h} = \frac{m \odot h}{1 - p}, \qquad m_i \sim \text{Bernoulli}(1 - p)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $h$ | a layer's activations | (1, 1, 1, 1) |
| $\tilde{h}$ | "h-tilde", the activations after dropout | (0, 0, 2, 2) |
| $m$ | the random mask of 0s and 1s | (0, 0, 1, 1) |
| $m_i \sim \text{Bernoulli}(1-p)$ | each mask entry is independently 1 with probability $1 - p$, else 0 (a coin flip) | 50/50 |
| $p$ | the drop rate | 0.5 |
| $\odot$ | multiply entry by entry | |

**In words:** "flip a biased coin for each activation, zero the ones that
lose, and divide the survivors by the keep probability."

**With the numbers:** $(0\cdot1, 0\cdot1, 1\cdot1, 1\cdot1) / 0.5 = (0, 0, 2, 2)$.

**In Python:**

```python
import random
random.seed(0)
h = [1, 1, 1, 1]
p = 0.5
# m_i ~ Bernoulli(1 - p): a coin flip each
m = [1 if random.random() < 1 - p else 0 for _ in h]
m  # → [0, 0, 1, 1]
# (m ⊙ h) / (1 - p)
[m_i * h_i / (1 - p) for m_i, h_i in zip(m, h)]  # → [0.0, 0.0, 2.0, 2.0]
```

**In code:** `dropout` draws the mask and scales the survivors during
training, and passes activations through unchanged at evaluation time.

**Why it matters:** dropout was a key ingredient of the deep-learning
revival and is still used in many models (the original transformer used
p = 0.1). Forgetting to switch it off at evaluation (`model.eval()` in
PyTorch) is a classic bug that makes predictions noisy.

## L1 and L2 penalties: a tax on large weights

Add a tax on the size of the weights to the loss, and the model must decide
whether each weight earns its keep. **L2** (ridge, "weight decay") taxes the
*square* of each weight: big weights pay a lot, small ones almost nothing,
so everything shrinks a little but nothing is eliminated. **L1** (lasso)
charges a flat rate per unit of size: small weights can't justify the
fee, so they're wiped out entirely, to exactly zero.

Worked example: with independent (orthonormal) features, the penalized
weights have a closed form. Start from the unpenalized weights (3, 0.5, −2)
and use penalty strength 1:

| weight | L2: divide by 1 + 1 | L1: move 1 toward zero, stop at zero |
|---|---|---|
| 3 | 1.5 | 2 |
| 0.5 | 0.25 | **0** |
| −2 | −1 | −1 |

![Fitted weights on ten features where only three matter: L2 leaves small nonzero weights on every useless feature, L1 sets most of them to exactly zero](figures/primer.ml.regularization.l1_vs_l2.svg)

**Reading it:** ten features, but only the first three affect the target
(grey bars show the true weights: 3, −2, 1.5, then zeros). The L2 fit (blue)
shrinks everything a little and leaves small nonzero weights on all seven
useless features. The L1 fit (orange) gets the three real weights nearly
right and sets most of the useless ones to exactly zero: it has selected
features for you.

$$
\text{L2: } \min_w \tfrac{1}{2}\lVert y - Xw \rVert^2 + \tfrac{\lambda}{2}\lVert w \rVert_2^2,
\qquad
\text{L1: } \min_w \tfrac{1}{2}\lVert y - Xw \rVert^2 + \lambda \lVert w \rVert_1
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $\min_w$ | "choose the weights $w$ that make the following smallest" | |
| $X, y$ | the features (one row per example) and targets | |
| $\lVert y - Xw \rVert^2$ | the sum of squared prediction errors | |
| $\lambda$ | "lambda", the penalty strength | 1 |
| $\lVert w \rVert_2^2$ | the L2 norm squared: sum of squared weights | $3^2 + 0.5^2 + 2^2 = 13.25$ |
| $\lVert w \rVert_1$ | the L1 norm: sum of absolute weights | $3 + 0.5 + 2 = 5.5$ |

**In words:** "fit the data, but pay a penalty proportional to the sum of
squared weights (L2) or the sum of absolute weights (L1)."

**With the numbers:** with orthonormal features the solutions are
$w_{\text{L2}} = w / (1 + \lambda) = (1.5, 0.25, -1)$ and
$w_{\text{L1}} = \text{sign}(w)\max(\lvert w \rvert - \lambda, 0) = (2, 0, -1)$,
the "soft threshold" in `soft_threshold`.

**In Python:**

```python
import math
w = [3, 0.5, -2]
# λ ("lambda" is taken in Python)
lam = 1
# ‖w‖₂², ‖w‖₁
sum(w_i ** 2 for w_i in w), sum(abs(w_i) for w_i in w)  # → (13.25, 5.5)
# L2: shrink every weight
[w_i / (1 + lam) for w_i in w]  # → [1.5, 0.25, -1.0]
# L1: sign(w) max(|w| - λ, 0)
[math.copysign(max(abs(w_i) - lam, 0), w_i) for w_i in w]  # → [2.0, 0.0, -1.0]
```

**In code:** `penalised_weights` gives both closed forms from the table, and
`fit_sparse_problem` fits the ten-feature problem in the figure (L1 by
alternating gradient steps with `soft_threshold`).

**Why it matters:** L2 (as weight decay; see AdamW in
`primer.ml.optimizers`) is on by default when training transformers. L1
is the tool when you want a sparse, interpretable model that uses only a
few features.

## Train, validation and test: practice, mock and real exams

A student has practice questions (to learn from), a mock exam (to check
progress and decide what to revise), and the real exam, sat once. Data is
split the same way. **Training data** fits the weights. **Validation data**
tunes choices like model size, learning rate and when to stop. **Test data**
is touched once, at the end, to estimate real-world performance honestly.
If you keep tuning against the test set, it quietly becomes a second
validation set and stops being honest.

Worked example: 100 examples split 70 / 15 / 15, shuffled first, with no
example in two sets. With only 10 examples, **5-fold cross-validation** cuts
them into 5 folds of 2; each fold takes a turn as the validation set while
the model trains on the other 8, so every example is held out exactly once.

```mermaid
flowchart TB
  subgraph K["5-fold cross-validation"]
    f1["Run 1: [VAL] train train train train"]
    f2["Run 2: train [VAL] train train train"]
    f3["Run 3: train train [VAL] train train"]
    f4["Run 4: train train train [VAL] train"]
    f5["Run 5: train train train train [VAL]"]
  end
  K --> A[Average the 5 validation scores]
```

**Reading it:** the data is cut into five equal folds. In each run, one fold
is held out (VAL) and the model is trained from scratch on the other four.
Every example is validated on exactly once, and the average of the five
scores is a steadier estimate than any single split.

$$
\text{CV score} = \frac{1}{k}\sum_{i=1}^{k} \text{score}\big(\text{model trained without fold } i,\ \text{fold } i\big)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $k$ | number of folds | 5 |
| $i$ | which fold is held out | 1 to 5 |
| score(model, fold) | e.g. accuracy of that model on that fold | |

**In words:** "train k models, each with one fold held out, and average
their scores on the fold each one didn't see."

**With the numbers:** 10 examples, $k = 5$: five models, each trained on 8
and scored on 2; if they score 1.0, 0.5, 1.0, 1.0, 0.5, the CV score is 0.8.

**In Python:**

```python
examples = list(range(10))
k = 5
# 5 folds of 2
folds = [examples[2 * i: 2 * i + 2] for i in range(k)]
# each model trains on the other 8
[len(examples) - len(fold) for fold in folds]  # → [8, 8, 8, 8, 8]
# score of the model trained without fold i, on fold i
scores = [1.0, 0.5, 1.0, 1.0, 0.5]
# (1/k) Σ_i score_i
sum(scores) / k  # → 0.8
```

**In code:** `train_val_test_split` shuffles and cuts the indices into three
disjoint sets, and `k_fold` yields the training and validation indices for
each fold in turn.

**Why it matters:** with small datasets a single split is noisy, and
cross-validation gives a more reliable estimate. With large datasets (or
expensive models) one fixed validation set is enough.

## Data leakage: the model saw the answers

A student who glimpsed the answer key scores perfectly on the practice exam
and learns nothing. **Data leakage** is when information that won't be
available at prediction time, or information from the test set, sneaks into
training. Offline scores look superb; production scores collapse.

Worked examples:

- **Duplicates across the split.** Every row stored twice, labels 30% noisy
  (so no honest model can beat about 70%). Split *before* removing
  duplicates, and a model that just copies its nearest training row scores
  **87%**: it's reading the answers off each test row's twin. Deduplicate
  first and the same model scores about **50%**.
- **A feature from the future.** A fraud model is given `chargeback_filed`,
  a column filled in only after a customer disputes a charge, which is to
  say after the outcome. Offline accuracy: **98%**. In production that column
  is always empty at decision time, and accuracy falls to **49%**, worse
  than an honest model using only the legitimate features (**61%**).

```mermaid
flowchart LR
  subgraph T["Timeline of one transaction"]
    direction LR
    A[Transaction happens] --> P["Model must decide<br/>(prediction time)"]
    P --> O[Fraud confirmed?]
    O --> C[chargeback_filed recorded]
  end
  C -. "leaks backwards into<br/>the training table" .-> P
```

**Reading it:** read the timeline left to right. The model has to decide at
the second box. The chargeback is only recorded at the last box, after the
outcome is known. In a historical training table every column sits side by
side, so nothing stops the model from using a column that, in real time, it
could never have had. The dotted arrow is the leak.

**In code:** `duplicate_leakage_demo` and `future_feature_demo` run both
examples; the "model" in the first is a one-nearest-neighbour lookup (copy the label
of the closest training row), the purest memoriser there is.

**Why it matters:** leakage is the most common reason a model that looked
great in evaluation fails in production. For language-model benchmarks the
big version is **contamination**: test questions that appeared in the
pretraining data. The defences are procedural: deduplicate before
splitting, split by time or by user when the real task is predicting the
future or new users, and ask of every feature "would I actually know this
at prediction time?"

## In 20 seconds
- Overfitting: training error falls while validation error rises; the model
  memorised noise. Underfitting: both errors are high.
- Fixes for overfitting: more data, early stopping, dropout, weight decay
  (L2), L1 for sparsity, data augmentation.
- Bias is systematic error (too simple); variance is sensitivity to the
  training sample (too flexible).
- Train fits weights, validation tunes choices, test is touched once;
  cross-validation for small data.
- Leakage (duplicates across splits, features from the future) gives great
  offline scores that collapse in production.

## Self-test questions

**Training loss drops, validation loss rises. What's happening and what do you do?**
Overfitting: the model is memorising training-set noise. Stop early (keep
the best-validation weights), add regularization (dropout, weight decay),
get more or more varied data, or reduce capacity.

**How do you tell underfitting from overfitting?**
Underfitting: training and validation errors are both high. Overfitting:
training error is low and validation error is much higher. They need
opposite fixes.

**Why does dropout scale the surviving activations by 1/(1 − p)?**
So the expected value of each activation is the same during training as at
evaluation, where nothing is dropped. The evaluation-time network can then
be used as is.

**Why does L1 produce exact zeros but L2 doesn't?**
L1's penalty has a constant slope, so small weights feel a fixed pull toward
zero that the data can't outweigh; they're thresholded to zero. L2's pull is
proportional to the weight, so it fades as a weight shrinks and never quite
reaches zero.

**Why must the test set be touched only once?**
Every decision made by looking at test scores fits the model to the test
set a little. Do it repeatedly and the test score stops being an unbiased
estimate of performance on new data.

**Give two examples of data leakage.**
The same (or near-duplicate) documents in both the training and test
splits; and a feature that's only known after the outcome (a chargeback flag
in a fraud model). For LLM evaluation: benchmark questions present in the
pretraining data.

## The papers behind this lesson

- Srivastava, Hinton, Krizhevsky, Sutskever & Salakhutdinov, *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* (JMLR, 2014): https://jmlr.org/papers/v15/srivastava14a.html
  Introduced dropout and showed it as a cheap approximation to averaging many networks. [annotated companion](../../papers/dropout.html)
- Tibshirani, *Regression Shrinkage and Selection via the Lasso* (JRSS B, 1996): https://www.jstor.org/stable/2346178
  Introduced the L1 penalty and its property of setting coefficients exactly to zero.
- Geman, Bienenstock & Doursat, *Neural Networks and the Bias/Variance Dilemma* (Neural Computation, 1992): https://doi.org/10.1162/neco.1992.4.1.1
  Framed generalization in neural networks as the bias-variance trade-off.
- Belkin, Hsu, Ma & Mandal, *Reconciling modern machine learning practice and the bias-variance trade-off* (2018): https://arxiv.org/abs/1812.11118
  Documented "double descent", where very over-parameterized models generalize better again.
- Kapoor & Narayanan, *Leakage and the Reproducibility Crisis in ML-based Science* (2022): https://arxiv.org/abs/2207.07048
  Catalogued the kinds of data leakage and how widely they inflate published results.

## Further reading
- Goodfellow, Bengio & Courville, *Deep Learning*, ch. 7 (regularization): https://www.deeplearningbook.org/contents/regularization.html
- CS231n notes, *Neural Networks Part 2* (regularization and dropout): https://cs231n.github.io/neural-networks-2/
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (free PDF; ch. 3 and 7): https://hastie.su.domains/ElemStatLearn/
- scikit-learn user guide, *Cross-validation*: https://scikit-learn.org/stable/modules/cross_validation.html
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from primer._show import banner, say, table, takeaway
from primer.ml.neural_net import MLP, binary_cross_entropy, make_moons, sigmoid

# ---------------------------------------------------------------------------
# 1. Memorising a zig-zag: the smallest possible overfit
# ---------------------------------------------------------------------------

ZIGZAG_X = np.array([0.0, 1.0, 2.0, 3.0])
ZIGZAG_Y = np.array([0.0, 1.0, 0.0, 1.0])


def _poly_fit(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Least-squares polynomial coefficients (lowest power first).

    Builds the design matrix [1, x, x², …, x^degree] and solves for the
    coefficients that minimise squared error. With degree = n_points − 1 the
    polynomial passes through every point exactly.
    """
    V = np.vander(x, degree + 1, increasing=True)
    coef, *_ = np.linalg.lstsq(V, y, rcond=None)
    return coef


def _poly_eval(coef: np.ndarray, x: np.ndarray | float) -> np.ndarray:
    return np.vander(np.atleast_1d(np.asarray(x, dtype=float)), len(coef), increasing=True) @ coef


def zigzag_fit_error(degree: int) -> float:
    """Training MSE of a degree-`degree` polynomial fitted to the four zig-zag points."""
    coef = _poly_fit(ZIGZAG_X, ZIGZAG_Y, degree)
    return float(np.mean((_poly_eval(coef, ZIGZAG_X) - ZIGZAG_Y) ** 2))


def zigzag_predict(x: float, degree: int) -> float:
    coef = _poly_fit(ZIGZAG_X, ZIGZAG_Y, degree)
    return float(_poly_eval(coef, x)[0])


# ---------------------------------------------------------------------------
# 2. Underfitting and overfitting with polynomials
# ---------------------------------------------------------------------------


def true_function(x: np.ndarray) -> np.ndarray:
    """The pattern hidden in the data: one period of a sine wave on [−1, 1]."""
    return np.sin(np.pi * x)


def make_curve_data(n: int, noise: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # x in [−1, 1] keeps high powers of x well-behaved numerically.
    x = np.sort(rng.uniform(-1, 1, n))
    return x, true_function(x) + rng.normal(0, noise, n)


def polynomial_errors(degree: int, n_train: int = 12, seed: int = 0) -> dict[str, float]:
    """Train and validation MSE for a polynomial of `degree` fitted to `n_train` noisy points."""
    x_tr, y_tr = make_curve_data(n_train, seed=seed)
    x_va, y_va = make_curve_data(300, seed=seed + 100)
    coef = _poly_fit(x_tr, y_tr, degree)
    return dict(
        train=float(np.mean((_poly_eval(coef, x_tr) - y_tr) ** 2)),
        val=float(np.mean((_poly_eval(coef, x_va) - y_va) ** 2)),
    )


# ---------------------------------------------------------------------------
# 3. Early stopping
# ---------------------------------------------------------------------------


def early_stopping(val_losses: list[float], patience: int = 2) -> tuple[int, int]:
    """Replay a run's validation losses; return (best epoch, epoch at which training stops).

    Stop once `patience` epochs in a row fail to beat the best so far, and
    keep the weights from the best epoch. If that never happens, the run
    ends at its last epoch.
    """
    best, since_best = 0, 0
    for epoch in range(1, len(val_losses)):
        if val_losses[epoch] < val_losses[best]:
            best, since_best = epoch, 0
        else:
            since_best += 1
            if since_best >= patience:
                return best, epoch
    return best, len(val_losses) - 1


@lru_cache(maxsize=4)
def train_flexible_model(epochs: int = 2000, hidden: int = 64, lr: float = 0.5, seed: int = 0) -> dict:
    """A 64-unit MLP trained full-batch on 30 noisy two-moons points; records train/val loss per epoch.

    The model has far more capacity than 30 points need, so after it learns
    the moons it starts bending around individual noisy points.
    """
    X_tr, y_tr = make_moons(n=30, noise=0.4, seed=seed)
    X_va, y_va = make_moons(n=300, noise=0.4, seed=seed + 1)
    model = MLP(2, hidden, seed=seed)
    train_loss, val_loss = [], []
    for _ in range(epochs):
        # Record both losses at the current weights, then take one full-batch step.
        yhat, cache = model.forward(X_tr)
        train_loss.append(binary_cross_entropy(yhat, y_tr))
        val_loss.append(model.loss(X_va, y_va))
        grads = model.backward(cache, y_tr)
        for k in model.params:
            model.params[k] -= lr * grads[k]
    return dict(train=np.array(train_loss), val=np.array(val_loss))


# ---------------------------------------------------------------------------
# 4. Bias and variance, measured by resampling
# ---------------------------------------------------------------------------


def bias_variance(degree: int, n_datasets: int = 300, n_train: int = 30, noise: float = 0.3, seed: int = 0) -> dict:
    """Fit `degree` polynomials to many independent training sets; decompose their error.

    bias²    = how far the *average* prediction is from the truth (systematic error).
    variance = how much predictions swing from one training set to the next.
    """
    rng = np.random.default_rng(seed)
    x_test = np.linspace(-0.95, 0.95, 60)
    preds = np.empty((n_datasets, len(x_test)))
    for i in range(n_datasets):
        x = rng.uniform(-1, 1, n_train)
        y = true_function(x) + rng.normal(0, noise, n_train)
        preds[i] = _poly_eval(_poly_fit(x, y, degree), x_test)
    mean_pred = preds.mean(axis=0)
    return dict(
        bias_sq=float(np.mean((mean_pred - true_function(x_test)) ** 2)),
        variance=float(np.mean(preds.var(axis=0))),
        noise=noise**2,
        x_test=x_test,
        preds=preds,
    )


# ---------------------------------------------------------------------------
# 5. Dropout
# ---------------------------------------------------------------------------


def dropout(x: np.ndarray, p: float = 0.5, training: bool = True, seed: int | None = None) -> np.ndarray:
    """Inverted dropout: zero each activation with probability p and scale survivors by 1/(1 − p).

    Scaling during training keeps the expected value unchanged, so at
    evaluation time the layer is simply switched off (identity).
    """
    if not training or p == 0.0:
        return x
    keep = np.random.default_rng(seed).random(x.shape) >= p
    return x * keep / (1.0 - p)


# ---------------------------------------------------------------------------
# 6. L1 and L2 penalties
# ---------------------------------------------------------------------------


def soft_threshold(w: np.ndarray, t: float) -> np.ndarray:
    """Move every weight t toward zero, and snap any weight within t of zero to exactly zero."""
    return np.sign(w) * np.maximum(np.abs(w) - t, 0.0)


def penalised_weights(w_plain: np.ndarray, strength: float, kind: str) -> np.ndarray:
    """Closed-form penalised solution when the features are orthonormal.

    L1 (lasso): soft-threshold the unpenalised weights by λ.
    L2 (ridge): shrink every unpenalised weight by the factor 1 / (1 + λ).
    """
    if kind == "l1":
        return soft_threshold(w_plain, strength)
    if kind == "l2":
        return w_plain / (1.0 + strength)
    raise ValueError(kind)


TRUE_SPARSE_W = np.array([3.0, -2.0, 1.5, 0, 0, 0, 0, 0, 0, 0])


def fit_sparse_problem(kind: str, strength: float = 5.0, n: int = 50, seed: int = 0) -> np.ndarray:
    """Linear regression with 10 features of which only the first 3 matter, fitted with an L1 or L2 penalty.

    L1 is solved with ISTA (gradient step, then soft-threshold), which is the
    simplest solver that shows *why* L1 gives exact zeros: the threshold
    step sets small weights to 0, not merely near it. L2 has a closed form.
    Objective: ½‖y − Xw‖² + penalty.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 10))
    y = X @ TRUE_SPARSE_W + rng.normal(0, 1.0, n)
    if kind == "l2":
        return np.linalg.solve(X.T @ X + strength * np.eye(10), X.T @ y)
    if kind == "l1":
        step = 1.0 / np.linalg.norm(X, 2) ** 2  # 1 / largest eigenvalue of XᵀX keeps ISTA stable
        w = np.zeros(10)
        for _ in range(2000):
            w = soft_threshold(w - step * X.T @ (X @ w - y), step * strength)
        return w
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# 7. Splits and cross-validation
# ---------------------------------------------------------------------------


def train_val_test_split(n: int, val: float = 0.15, test: float = 0.15, seed: int = 0):
    """Shuffle indices 0..n−1 and cut them into disjoint train / validation / test sets."""
    idx = np.random.default_rng(seed).permutation(n)
    n_test, n_val = round(n * test), round(n * val)
    return idx[n_test + n_val :], idx[n_test : n_test + n_val], idx[:n_test]


def k_fold(n: int, k: int = 5, seed: int = 0):
    """Yield (train indices, validation indices) for each of k folds; every example is held out once."""
    folds = np.array_split(np.random.default_rng(seed).permutation(n), k)
    for i in range(k):
        yield np.concatenate([f for j, f in enumerate(folds) if j != i]), folds[i]


# ---------------------------------------------------------------------------
# 8. Data leakage
# ---------------------------------------------------------------------------


def _one_nearest_neighbour_accuracy(X_tr, y_tr, X_te, y_te) -> float:
    """Predict each test point's label by copying its closest training point: a pure memoriser."""
    d = ((X_te[:, None, :] - X_tr[None, :, :]) ** 2).sum(-1)
    return float(np.mean(y_tr[d.argmin(axis=1)] == y_te))


def duplicate_leakage_demo(n: int = 300, seed: int = 0) -> dict[str, float]:
    """Noisy labels, every row stored twice. Split before vs. after removing duplicates."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    y = (X[:, 0] > 0).astype(int)
    flip = rng.random(n) < 0.3  # 30% label noise: no model should beat ~70%
    y[flip] = 1 - y[flip]
    X_dup, y_dup = np.vstack([X, X]), np.concatenate([y, y])

    # Leaky: split the duplicated table, so many test rows have their twin in training.
    idx = rng.permutation(2 * n)
    te, tr = idx[: n // 2], idx[n // 2 :]
    leaky = _one_nearest_neighbour_accuracy(X_dup[tr], y_dup[tr], X_dup[te], y_dup[te])

    # Clean: deduplicate first, then split.
    idx = rng.permutation(n)
    te, tr = idx[: n // 4], idx[n // 4 :]
    clean = _one_nearest_neighbour_accuracy(X[tr], y[tr], X[te], y[te])
    return dict(leaky=leaky, clean=clean)


def _fit_logistic(X: np.ndarray, y: np.ndarray, epochs: int = 2000, lr: float = 0.5) -> tuple[np.ndarray, float]:
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(epochs):
        g = (sigmoid(X @ w + b) - y) / len(y)
        w -= lr * X.T @ g
        b -= lr * g.sum()
    return w, b


@lru_cache(maxsize=4)
def future_feature_demo(n: int = 2000, seed: int = 0) -> dict[str, float]:
    """A fraud model given a feature that's only recorded *after* fraud is confirmed.

    Features: two weak honest signals, plus `chargeback_filed`, which is set
    once the customer disputes the charge, i.e. after the outcome. Offline the
    column is filled in; at prediction time in production it's always 0.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    honest = rng.normal(0, 1, (n, 2)) + 0.4 * y[:, None]  # weakly informative
    chargeback = (y == 1) & (rng.random(n) < 0.97)  # the leak: almost equal to the label
    X = np.c_[honest, chargeback.astype(float)]
    tr, te = slice(0, n // 2), slice(n // 2, None)
    w, b = _fit_logistic(X[tr], y[tr])
    offline = float(np.mean((sigmoid(X[te] @ w + b) > 0.5) == y[te]))
    X_prod = X[te].copy()
    X_prod[:, 2] = 0.0  # not known yet when the decision has to be made
    production = float(np.mean((sigmoid(X_prod @ w + b) > 0.5) == y[te]))
    honest_only = _fit_logistic(X[tr, :2], y[tr])
    honest_acc = float(np.mean((sigmoid(X[te, :2] @ honest_only[0] + honest_only[1]) > 0.5) == y[te]))
    return dict(offline=offline, production=production, honest_model=honest_acc)



# ---------------------------------------------------------------------------
# 9. Figures (rendered by `make figures`)
# ---------------------------------------------------------------------------


def figures() -> dict:
    """Plots computed from this module's own functions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = {}
    grid = np.linspace(-1, 1, 400)
    x_tr, y_tr = make_curve_data(12, seed=0)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(grid, true_function(grid), color="gray", lw=3, alpha=0.5, label="true pattern")
    for degree, c in ((1, "C0"), (3, "C2"), (11, "C3")):
        ax.plot(grid, _poly_eval(_poly_fit(x_tr, y_tr, degree), grid), color=c, label=f"degree {degree}")
    ax.scatter(x_tr, y_tr, color="black", zorder=5, s=18, label="12 training points")
    ax.set(ylim=(-2, 2), xlabel="x", ylabel="y", title="Underfit, good fit, memorised")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figs["fits"] = fig

    degrees = np.arange(0, 12)
    errs = [polynomial_errors(int(d)) for d in degrees]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(degrees, [max(e["train"], 1e-6) for e in errs], "o-", label="training error")
    ax.plot(degrees, [e["val"] for e in errs], "o-", label="validation error")
    ax.set(yscale="log", xlabel="polynomial degree (model capacity)", ylabel="mean squared error (log)",
           title="Training error only falls; validation error is U-shaped")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figs["degree_sweep"] = fig

    run = train_flexible_model()
    best = int(np.argmin(run["val"]))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(run["train"], label="training loss")
    ax.plot(run["val"], label="validation loss")
    ax.axvline(best, color="gray", ls="--", lw=1)
    ax.text(best + 20, max(run["val"]) * 0.9, f"best validation\n(epoch {best})", fontsize=8)
    ax.set(xlabel="epoch", ylabel="cross-entropy", title="An over-sized network on 30 noisy points")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figs["learning_curves"] = fig

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, degree in zip(axes, (1, 9)):
        bv = bias_variance(degree)
        for p in bv["preds"][:20]:
            ax.plot(bv["x_test"], p, lw=0.8, alpha=0.6)
        ax.plot(bv["x_test"], true_function(bv["x_test"]), color="gray", lw=3, alpha=0.7)
        ax.set(ylim=(-2, 2), xlabel="x", title=f"degree {degree}: bias² {bv['bias_sq']:.3f}, variance {bv['variance']:.3f}")
    axes[0].set_ylabel("prediction")
    fig.tight_layout()
    figs["bias_variance"] = fig

    fig, ax = plt.subplots(figsize=(7, 3.4))
    idx = np.arange(10)
    ax.bar(idx - 0.27, TRUE_SPARSE_W, width=0.27, color="lightgray", label="true weights")
    ax.bar(idx, fit_sparse_problem("l2"), width=0.27, color="C0", label="L2 (ridge)")
    ax.bar(idx + 0.27, fit_sparse_problem("l1"), width=0.27, color="C1", label="L1 (lasso)")
    ax.axhline(0, color="black", lw=0.6)
    ax.set(xticks=idx, xlabel="feature (only 0, 1, 2 matter)", ylabel="fitted weight", title="L1 zeroes useless weights; L2 only shrinks them")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figs["l1_vs_l2"] = fig
    return figs


# ---------------------------------------------------------------------------
# 10. Walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. Memorising a zig-zag")
    table(["model", "training error", "prediction at x = 4"],
          [("flat line (degree 0)", zigzag_fit_error(0), zigzag_predict(4, 0)),
           ("cubic (degree 3)", zigzag_fit_error(3), zigzag_predict(4, 3))], floatfmt=".3f")
    takeaway("Perfect on the training data and absurd one step beyond it: that's overfitting.")

    banner("2. Sweeping model capacity")
    table(["degree", "train MSE", "val MSE"], [(d, *polynomial_errors(d).values()) for d in (0, 1, 3, 5, 9, 11)], floatfmt=".4g")
    say("Training error only falls. Validation error falls, bottoms out around degree 3 to 5, then explodes.")

    banner("3. Early stopping")
    losses = [1.0, 0.8, 0.6, 0.55, 0.58, 0.65, 0.7]
    best, stopped = early_stopping(losses, patience=2)
    say(f"Validation losses {losses}, patience 2: best epoch {best}, stop at epoch {stopped}, restore epoch {best}.")
    run = train_flexible_model()
    b = int(np.argmin(run["val"]))
    table(["epoch", "train loss", "val loss"], [(e, run["train"][e], run["val"][e]) for e in (0, 100, b, 1000, len(run["val"]) - 1)],
          floatfmt=".3f")

    banner("4. Bias and variance (300 training sets of 30 points)")
    table(["degree", "bias²", "variance", "bias² + variance + noise"],
          [(d, bv["bias_sq"], bv["variance"], bv["bias_sq"] + bv["variance"] + bv["noise"])
           for d, bv in ((d, bias_variance(d)) for d in (1, 3, 9))], floatfmt=".4f")

    banner("5. Dropout")
    say(f"p = 0.5 on (1, 1, 1, 1): {dropout(np.ones(4), 0.5, True, seed=3)}; at evaluation: {dropout(np.ones(4), 0.5, False)}.")

    banner("6. L1 vs. L2")
    w = np.array([3.0, 0.5, -2.0])
    table(["penalty", "weights (from 3, 0.5, −2; strength 1)"],
          [("L2", str(penalised_weights(w, 1.0, "l2"))), ("L1", str(penalised_weights(w, 1.0, "l1")))])
    say(f"On 10 features where 3 matter, L1 sets {int(np.sum(fit_sparse_problem('l1')[3:] == 0))} of the 7 useless weights to exactly 0; L2 sets none.")

    banner("7. Splits and cross-validation")
    tr, va, te = train_val_test_split(100)
    say(f"100 examples -> train {len(tr)}, validation {len(va)}, test {len(te)}.")
    for i, (train_idx, val_idx) in enumerate(k_fold(10, 5)):
        print(f"  fold {i + 1}: validate on {sorted(val_idx.tolist())}, train on {len(train_idx)}")
    print()

    banner("8. Data leakage")
    dup, fut = duplicate_leakage_demo(), future_feature_demo()
    table(["scenario", "accuracy"],
          [("duplicates across the split (1-NN)", dup["leaky"]), ("deduplicated first (1-NN)", dup["clean"]),
           ("future feature, offline", fut["offline"]), ("same model in production", fut["production"]),
           ("honest features only", fut["honest_model"])], floatfmt=".2f")
    takeaway("Ask of every feature: would I actually know this at prediction time?")


if __name__ == "__main__":
    demo()
