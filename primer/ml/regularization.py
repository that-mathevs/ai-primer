r"""
# Regularization: learning the pattern instead of memorising the examples

Run: `python -m primer.ml.regularization`

New to the notation? `primer.notation` explains every symbol used here from
zero. This lesson builds on the losses in `primer.ml.losses` and the
training loop in `primer.ml.neural_net`.

## Level 1: The practitioner's guide

**In one sentence.** Regularization is everything you do to make a model
learn the pattern in its training data rather than memorise the examples,
and the way you know it is working is the gap between the error on data
the model trained on and the error on data it has never seen.

**When you need it.** Whenever you train or fine-tune anything on less data
than the model could memorise, which in practice is always: a fine-tune on
a few thousand examples, a classifier on a table of ten thousand rows, a
forecasting model on a year of history. The tell is a training error that
keeps falling while the validation error turns and climbs. This lesson's
polynomial sweep shows the whole story in one table: at degree 5 the
training error is 0.013 and the validation error 0.057; at degree 9 the
training error has dropped to 0.004 and the validation error has risen
twelve-fold to 0.71; at degree 11 the training error is zero and the
validation error is 7,631. Its over-sized network bottoms out on
validation near epoch 300 and gets steadily worse for the next 1,700
epochs while its training loss keeps improving. The other tell is the
opposite one, and it hides in a good offline score: a model that looked
superb in evaluation and collapsed in production has usually seen the
answers, and that is leakage, which the same lesson covers. You do not need
more regularization when both errors are high; that is underfitting, and it
wants more capacity, more features or more training, the opposite fix.

**Your options.** From the cheapest to the most involved:

| Option | What it does | What it guarantees | What it costs | Where it lives |
|---|---|---|---|---|
| A held-out validation set (and a test set touched once) | Measures the gap that every other option is judged by | An honest estimate, as long as the test set is not used to choose | 15% to 30% of your data, and discipline | Your data pipeline |
| Early stopping | Watches validation loss, stops after `patience` bad checks, restores the best weights | You ship the best model seen, not the last | A validation pass per epoch; nothing in the model | The training loop: a callback in every framework |
| More data, or augmented data | Averages the noise away | Tames variance in flexible models; does nothing for a model too simple to fit the pattern | Collection or labelling time | Your data |
| L2 penalty (weight decay) | Taxes the square of every weight, so all of them shrink a little | Smaller, smoother weights; nothing is eliminated | One hyperparameter, λ | The optimiser: on by default in AdamW |
| Dropout | Zeroes each activation with probability p on every training step, scaled so evaluation is a plain pass-through | No neuron becomes indispensable | Slower convergence; a bug if left on at evaluation | The model, between layers |
| L1 penalty (lasso) | Charges a flat fee per unit of weight, so small weights go to exactly zero | A sparse model that names its features | Harder optimisation; one hyperparameter | Linear and tabular models, feature selection |
| Less capacity | A smaller model, a lower polynomial degree, fewer features | Less to memorise with | Possibly too little to learn the pattern | The architecture |
| Cross-validation | Trains k models, each with one fold held out, and averages their scores | A steadier estimate than one split when data is small | k training runs | Small tabular problems; scikit-learn's cross-validation guide |
| Leak-proof splitting | Deduplicates before splitting, splits by time or by user, drops features unknown at prediction time | An offline score that survives production | Thinking about the timeline of every feature | Your data pipeline, before any training |

**How to choose.** Diagnose first, because the two failures want opposite
cures.

- Both errors high: underfitting. Add capacity, features or training time.
  Regularizing further makes it worse.
- Training low, validation high: overfitting. Early stopping first, because
  it costs nothing; then weight decay (which you probably already have) and
  dropout; then more data if you can get it.
- A tabular model you must explain, or hundreds of features of which a few
  matter: L1, which in this lesson zeroes 5 of 7 useless weights while L2
  zeroes none.
- Fewer than a few thousand examples: cross-validate rather than trust one
  split.
- A model that was wonderful offline: before celebrating, check for twins
  across the split and for columns filled in after the outcome.
- Whatever you pick, the test set is touched once, at the end. Every
  decision made by looking at it turns it into a second validation set, and
  the benchmark version of this failure is Goodhart's law in
  `primer.ml.benchmarks`.

**What it costs.** Early stopping costs one validation pass per epoch and
stops training sooner, so it usually saves compute. Weight decay and L1 cost
nothing at inference and one hyperparameter each; PyTorch's AdamW defaults
weight decay to 0.01. Dropout slows training, because each step trains a
thinned network, and the original transformer used p = 0.1 rather than
anything heavier. Cross-validation multiplies training cost by the number
of folds. The dear cost is data: the validation and test sets are examples
you cannot train on, and with 100 examples a 70 / 15 / 15 split leaves 15
for each. Leakage costs the most of all, late: this lesson's fraud model
with a future feature scores 98% offline and 49% in production, worse than
the honest model's 61%.

**What breaks.**

- **Dropout left on at evaluation.** Predictions become noisy and change
  from call to call. Switch the model to evaluation mode; the lesson's
  `dropout` passes activations through unchanged there.
- **Early stopping with no restore.** Stopping is half the job; the weights
  at the stop are the ones that just failed twice. Keep the snapshot from
  the best check.
- **Patience too short.** A noisy validation curve stops training on a
  blip. Smooth it, or raise the patience.
- **Tuning against the test set.** Each look fits the model to it a little.
  Choose with validation, report with test.
- **Duplicates across the split.** A nearest-neighbour memoriser scores 87%
  on data whose labels are 30% noise, when no honest model can beat about
  70%; deduplicated, it scores 50%. Deduplicate before splitting.
- **A feature from the future.** `chargeback_filed` is recorded after the
  outcome. Ask of every column: would I know this at prediction time?
- **A random split on time-ordered data.** Predicting next month from a
  shuffle of all months is a leak. Split by time when the task is the
  future, by user when the task is new users.
- **Regularizing an underfit model.** It cannot get better by learning
  less. Look at both curves before reaching for the penalty.

**In the wild.** Weight decay is on by default in the optimisers that train
transformers: `torch.optim.AdamW` takes `weight_decay` with a default of
0.01, and `primer.ml.optimizers` shows why decoupling it from the gradient
step mattered. Early stopping is a callback everywhere it is offered;
Keras's `EarlyStopping` has `patience` and `restore_best_weights`, which
default to 0 and False, so a naive call gets neither. Dropout (Srivastava
et al., 2014) trains at p = 0.1 in the original transformer and is switched
off by `model.eval()` in PyTorch. The lasso is Tibshirani (1996), and
scikit-learn's cross-validation guide is the standard reference for folds
that respect groups and time. Kapoor and Narayanan (2022) catalogued the
kinds of leakage and how widely they inflate published results, and Belkin
et al. (2018) documented double descent, where very over-parameterised
models generalise well again, which is why the modern recipe is a large
model plus regularization rather than a small model.

**Go deeper.** Level 2 fits the memorising cubic by hand, sweeps polynomial
degree to draw the valley in validation error, replays early stopping on a
list of losses, splits error into bias and variance over 300 training sets,
builds dropout and the two penalties in a few lines each, cuts data into
folds, and reproduces both leaks with a nearest-neighbour memoriser. If you
only needed to diagnose a curve and pick a fix, you are done.

## Level 2: How it works, from scratch

Take four training points: (0, 0), (1, 1), (2, 0) and (3, 1). They zig-zag,
but the pattern behind them is a constant of about 0.5, and the zig-zag is
noise. Fit two models to them and ask each one for a prediction at x = 4,
one step past the data.

| model | error on the 4 training points | prediction at x = 4 |
|---|---|---|
| flat line at 0.5 (the pattern) | 0.25 | 0.5 |
| cubic through all four points (memorised) | **0** | **8** |

The cubic passes through every point, so its training error is zero, and it
predicts 8, far from anything the data suggests. You can check the 8 by hand
with a difference table: the values 0, 1, 0, 1 have differences 1, −1, 1,
then −2, 2, then 4; a cubic keeps that last difference constant, so extending
the table gives 6, then 7, then 8. The cubic has **overfit**: it learned the
noise along with the pattern.

How complicated a shape a model can fit is its **capacity**: a cubic has
more than a flat line. With too little capacity a model **underfits**,
because it cannot represent even the pattern. With too much, and nothing to
hold it back, it overfits. **Regularization** holds a high-capacity model
back without taking its capacity away.

```mermaid
flowchart LR
  D["Training data =<br/>pattern + noise"] --> M{Model capacity}
  M -->|too little| U["Underfits:<br/>misses the pattern"]
  M -->|about right| G["Generalizes:<br/>learns the pattern"]
  M -->|too much, unchecked| O["Overfits:<br/>memorises the noise"]
  G --> N[Good on new data]
  U & O --> B[Bad on new data]
```

**Reading it:** start from the data on the left, which is always pattern
plus noise, and follow the three arrows out of the capacity box. Too little
capacity misses the pattern and too much fits the noise as well, so only the
middle path does well on new data.

The same three outcomes appear with more data. Here 12 noisy samples of a
sine wave are fitted by polynomials of degree 1 (a straight line), 3 (a
cubic) and 11.

![Three polynomial fits to 12 noisy sine samples: the degree-1 line misses the curve, the cubic follows it, and the degree-11 polynomial hits every dot but swings wildly between them](figures/primer.ml.regularization.fits.svg)

**Reading it:** the dots are the 12 samples and the grey curve is the sine
they came from. The straight line cannot bend, so it misses the shape. The
cubic follows the sine closely. The degree-11 polynomial passes through every
dot and swings wildly between and beyond them, because it has fitted the
noise.

Training error alone always favours memorising: it ranks the zig-zag's
cubic and the degree-11 polynomial best. What shows the difference is the same error
measured on **validation** data, examples the model was not trained on:

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

**In code:** `zigzag_fit_error` and `zigzag_predict` fit the zig-zag, and
`make_curve_data` draws the sine samples.

## Underfitting and overfitting

To find the right capacity, fit the model at every capacity and measure
both errors. Here that means polynomials of degree 1 to 11, fitted to 12
noisy points from a sine wave and checked on 300 fresh points from the same
wave.

| degree | training error | validation error | diagnosis |
|---|---|---|---|
| 1 | 0.20 | 0.30 | underfit: both high |
| 3 | 0.021 | 0.065 | about right |
| 5 | 0.013 | 0.057 | best: lowest validation error |
| 9 | 0.004 | 0.71 | overfit: validation 12× worse than at degree 5 |
| 11 | 0.0000 | 7,631 | memorised |

![Error against polynomial degree on a log scale: training error only falls, while validation error is lowest at degree 5 and then climbs steeply](figures/primer.ml.regularization.degree_sweep.svg)

**Reading it:** capacity (polynomial degree) runs along the bottom and error
up the side, on a log scale. Follow the blue training curve first: it only
goes down, because more capacity always fits the training points better.
Then follow the orange validation curve. It falls, stays nearly flat from
degree 3 to 5 (lowest at 5), and climbs from degree 6 on. Left of the
valley's floor is underfitting, and right of it is overfitting.

The two sides of the valley need opposite fixes. An underfit model, with
both errors high, needs more capacity, more features or more training. An
overfit model, with training error low and validation error high, needs more
data, regularization or early stopping.

**In code:** `polynomial_errors` fits each degree by least squares and
reports both errors.

## Early stopping

The capacity sweep picked a model's size by watching validation error.
Training time needs the same watch. While a flexible model trains, its
validation error falls as it learns the pattern, then rises once it starts
fitting the noise. **Early stopping** measures validation error after every
epoch (one pass through the training data). It stops once the error has
failed to improve for a set number of epochs, the **patience**, and keeps
the weights from the best epoch. The model itself does not change, only when
its training ends.

Say the validation losses after each epoch are 1.0, 0.8, 0.6, 0.55, 0.58,
0.65 and 0.7, and the patience is 2. Epoch 3 is the best. Epochs 4 and 5 are
both worse, so training stops at epoch 5 and the weights from epoch 3 are
kept.

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

**Reading it:** start at the end of an epoch and follow the arrows. A new
best validation loss saves the weights and resets the counter; anything else
adds one to it. When the counter reaches the patience, training stops and the
saved weights come back, so you keep the best model seen, not the last.

A neural network with 64 hidden units, trained on only 30 noisy points,
shows the turn.

![Loss curves for an over-sized network: training loss keeps falling while validation loss bottoms out near epoch 300 and then rises](figures/primer.ml.regularization.learning_curves.svg)

**Reading it:** both curves fall at first. Near epoch 300, at the dashed
line, the validation curve bottoms out and turns up while the training curve
keeps falling: from there on the network is fitting individual noisy points.
To the right of the line the network gets worse on new data, and early
stopping keeps the weights from the line.

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

**In code:** `early_stopping` replays a list of validation losses, and
`train_flexible_model` trains the network in the chart.

## Bias and variance

In the capacity sweep and in the training run alike, validation error falls
and then rises. Bias and variance explain why. The same kind of model,
trained on a different sample of data, gives a different fit, and the fits
can miss the truth in two ways. **Bias** is the miss that
remains when you average the fits over many samples: a straight line misses a
sine wave the same way whichever points it gets. **Variance** is how much the
fit swings from one sample to the next: a degree-9 polynomial follows each
sample's noise, so every sample gives a different curve.

Fitting polynomials to 300 random training sets of 30 points each, and
averaging over the input range, gives:

| degree | bias² (systematic error) | variance (swing between training sets) |
|---|---|---|
| 1 | 0.174 | 0.022 |
| 3 | 0.0035 | 0.015 |
| 9 | 0.0038 | 4.5 |

Drawing twenty of the fits for each degree shows the two kinds of miss.

![Twenty fits from different training sets: degree-1 lines agree but all miss the sine (high bias), degree-9 curves average to the sine but scatter widely (high variance)](figures/primer.ml.regularization.bias_variance.svg)

**Reading it:** each thin line is the model fitted to one random training
set, and the thick grey curve is the truth. On the left, at degree 1, the
lines agree with each other but all miss the sine's shape in the same way:
high bias, low variance. On the right, at degree 9, the lines follow the sine
on average, but each one wiggles its own way, most near the edges: low bias,
high variance.

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
its predictions scatter around their own average (variance), plus noise that
no model can predict."

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

More data tames variance, so it helps a flexible model; it does nothing for
a model too simple to bend, whose bias stays. Very large neural networks
complicate the picture: past a point, even bigger ones do better on new data
again, an effect called **double descent** (Belkin et al., 2018).

**In code:** `bias_variance` fits one polynomial to each of many random
training sets and splits their error into bias², variance and noise.

## Dropout

An overfit model is a flexible one with too much variance. Early stopping
limits how long such a model trains. Dropout and weight penalties leave the
model and the training time alone and make the noise harder to fit instead.
**Dropout** works on the activations: on every training step, each activation is set to zero
with probability p, so the network cannot rely on any single neuron or on a
fragile combination of them.

Take four activations (1, 1, 1, 1) and a drop rate p = 0.5. Suppose the
random mask keeps the last two: (0, 0, 1, 1). The survivors are scaled up by
1 / (1 − 0.5) = 2, giving (0, 0, 2, 2). The average is still 1, so the next
layer sees a signal of the same size on average. At evaluation time nothing
is dropped and nothing is scaled.

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

**Reading it:** the left box is one training step. Two of the four units
drop to 0 and the two survivors double, so the total, 4, matches what the
full layer would send; the next step drops a different random pair. The
right box is evaluation, where every unit passes through unchanged. Scaling
during training (called inverted dropout) is what lets evaluation be a plain
pass-through.

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

The drop rate is a setting you choose; the original transformer trained with
p = 0.1. **Switch dropout off at
evaluation** (`model.eval()` in PyTorch): left on, it makes predictions
noisy, different from one call to the next.

**In code:** `dropout` draws the mask and scales the survivors during
training, and passes activations through unchanged at evaluation time.

## L1 and L2 penalties

Dropout works on the activations. A penalty works on the weights: add a tax
on the size of the weights to the loss, and each weight has to earn its keep
by reducing the error. **L2** (also called ridge, or **weight decay**) taxes
the *square* of each weight. Big weights pay a lot and small ones almost
nothing, so every weight shrinks a little and none is eliminated. **L1**
(the lasso) charges a flat rate per unit of size. A small weight cannot
justify the fee, so it goes to exactly zero.

With independent (orthonormal) features, the penalized weights have a closed
form. Starting from unpenalized weights (3, 0.5, −2) with penalty strength 1:

| weight | L2: divide by 1 + 1 | L1: move 1 toward zero, stop at zero |
|---|---|---|
| 3 | 1.5 | 2 |
| 0.5 | 0.25 | **0** |
| −2 | −1 | −1 |

The difference shows when most features are useless: here, ten features of
which only three matter.

![Fitted weights on ten features where only three matter: L2 leaves small nonzero weights on every useless feature, L1 sets most of them to exactly zero](figures/primer.ml.regularization.l1_vs_l2.svg)

**Reading it:** the grey bars are the true weights: 3, −2 and 1.5, then
seven zeros. The L2 fit (blue) shrinks everything a little and leaves small
nonzero weights on all seven useless features. The L1 fit (orange) gets the
three real weights nearly right and sets most of the useless ones to exactly
zero, so it has picked out the features that matter.

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
$w_{\text{L1}} = \text{sign}(w)\max(\lvert w \rvert - \lambda, 0) = (2, 0, -1)$;
the L1 rule is called a soft threshold.

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

L2 is on by default when transformers are trained, as the weight decay in
AdamW (`primer.ml.optimizers`). L1 is the choice when you want a model that
uses only a few features, so that you can see which ones it relies on.

**In code:** `penalised_weights` computes both rules, and
`fit_sparse_problem` fits the ten-feature problem in the chart.

## Train, validation and test sets

The degree and the stopping epoch were both chosen by watching validation
error. Every choice made that way fits the model to the validation data a
little, so its error is no longer an honest estimate for new data. Data is
therefore split three ways, like a student's practice questions, mock exam
and real exam. **Training data** fits the
weights. **Validation data** is the mock: it makes the choices, such as model
size, learning rate and when to stop. **Test data** is the real exam, used
once at the end to estimate performance on new data. Tuning against the test
set turns it into a second validation set.

Split 100 examples 70 / 15 / 15, shuffling first and putting no example in
two sets. With only 10 examples, one split leaves too few to check with, and
**cross-validation** uses every example instead. 5-fold cross-validation cuts
the 10 into 5 folds of 2. Each fold takes a turn as the validation set while
a model trains on the other 8, so every example is held out exactly once.

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

**Reading it:** read down the five runs. The VAL fold moves one place to the
right each run, and a fresh model trains on the other four folds. The
average of the five validation scores is a steadier estimate than any single
split.

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

Cross-validation costs k training runs instead of one. With a large
dataset, or a model that is expensive to train, one fixed validation set is
enough.

**In code:** `train_val_test_split` shuffles and cuts the indices into three
disjoint sets, and `k_fold` yields the training and validation indices for
each fold in turn.

## Data leakage

A split is only honest if nothing from the held-out data reaches training. A
student who has seen the answer key scores perfectly on the mock and learns
nothing. **Data leakage** is the same failure in a model: information from
the test set, or information that will not exist at prediction time, gets
into training. The model then scores well offline and badly in production.
Here is one example of each.

- **Duplicates across the split.** Every row is stored twice, and 30% of the
  labels are flipped at random, so no honest model can beat about 70%. Split
  *before* removing duplicates, and a model that copies the label of its
  nearest training row scores 87%, because it reads the answer off each test
  row's twin. Deduplicate first and the same model scores about 50%.
- **A feature from the future.** A fraud model is given `chargeback_filed`, a
  column filled in only after a customer disputes a charge, that is, after
  the outcome. Offline accuracy is 98%. In production that column is always
  empty at decision time, and accuracy falls to 49%, worse than an honest
  model using only the legitimate features (61%).

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

**Reading it:** read the timeline left to right. The model decides at the
second box, and the chargeback is recorded at the last, after the outcome is
known. A historical training table puts every column side by side, so
nothing stops the model from learning from a column it could never have had
at decision time. The dotted arrow is that leak.

Language-model benchmarks have their own version, **contamination**: test
questions that appeared in the pretraining data. The defences are habits.
Deduplicate before splitting. Split by time when the task is predicting the
future, and by user when it is serving new users. Ask of every column
whether you would know it at prediction time.

**In code:** `duplicate_leakage_demo` and `future_feature_demo` run both
examples.

## In 20 seconds
- Overfitting: training error keeps falling while validation error rises.
  Underfitting: both errors stay high, and it needs the opposite fix.
- Bias is error that stays whatever sample a model trains on (too simple);
  variance is how much the fit swings between samples (too flexible).
- Fixes for overfitting: more data, early stopping, dropout and weight
  penalties (L2 shrinks every weight; L1 sets the small ones to zero).
- Training data fits the weights, validation data makes the choices, and
  test data is used once, at the end; with little data, cross-validate.
- Leakage, such as duplicates across the split or a column filled in after
  the outcome, gives offline scores that collapse in production.

## Self-test questions

**Training loss drops while validation loss rises. What's happening, and what helps?**
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

    def on_chart(y, limit=2.0):
        # A wild fit swings far off the chart; break its line there instead of running it into the labels.
        return np.where(np.abs(y) <= limit, y, np.nan)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(grid, true_function(grid), color="gray", lw=3, alpha=0.5, label="true pattern")
    for degree, c in ((1, "C0"), (3, "C2"), (11, "C3")):
        ax.plot(grid, on_chart(_poly_eval(_poly_fit(x_tr, y_tr, degree), grid)), color=c, label=f"degree {degree}")
    ax.scatter(x_tr, y_tr, color="black", zorder=5, s=18, label="12 training points")
    ax.set(ylim=(-2, 2), xlabel="x", ylabel="y", title="Underfit, good fit, memorised")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)  # beside the plot: the wild fit crosses every corner
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
            ax.plot(bv["x_test"], on_chart(np.asarray(p)), lw=0.8, alpha=0.6)
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
