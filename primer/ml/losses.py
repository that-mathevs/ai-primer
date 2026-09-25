r"""
# Loss functions: turning "how wrong were we?" into one number

Run: `python -m primer.ml.losses`

## The idea

A loss is a scorecard with one number on it. Imagine a coach who, after every
practice, must summarise the whole team's performance in a single score so
the players know whether they're improving. Training a model does exactly
this: it computes the loss, then adjusts the weights to make that number
smaller (see `primer.ml.neural_net` for the loop).

What the scorecard rewards is what the model learns. Language models are
scored on how much probability they gave the real next word; price
predictors on how far off their numbers were; search models on whether they
ranked the right passage above the wrong ones.

```mermaid
flowchart LR
  P[Prediction] --> LF{Loss function}
  T[Right answer] --> LF
  LF -->|classes or tokens| CE[Cross-entropy]
  LF -->|numbers| REG[MSE or MAE]
  LF -->|matching pairs| CON[Contrastive / InfoNCE]
  CE & REG & CON --> N[One number to minimize]
```

**Reading it:** a prediction and the right answer go into the loss function
and one number comes out. Which branch you take depends on what the model
predicts: a choice among classes (or the next token) uses cross-entropy, a
number uses squared or absolute error, and "which of these belong together"
uses a contrastive loss. Each section below climbs one branch.

## Cross-entropy: how surprised were we by the truth?

Picture a weather forecaster scored every evening. If they said "90% chance
of rain" and it rained, they lose a point or two. If they said "1% chance of
rain" and it poured, they are humiliated. Cross-entropy is that humiliation
meter: it charges according to how little probability you gave to what
actually happened.

Worked example: the model assigned these probabilities to the correct next
token.

| Probability on the correct token | Loss (−ln p) |
|---|---|
| 0.9 | 0.11 |
| 0.5 | 0.69 |
| 0.1 | 2.30 |
| 0.01 | 4.61 |

Being confidently wrong (1%) costs about **40×** more than being mostly
right (90%): 4.61 / 0.105 ≈ 44.

![Cross-entropy loss as a function of the probability on the correct answer](figures/primer.ml.losses.cross_entropy.svg)

**Reading it:** the horizontal axis is how much probability the model put on
the right answer; the vertical axis is the loss it pays. Start at the right
edge: at p = 1 the loss is 0 and the curve is nearly flat, so going from 90%
to 99% buys little. Now slide left: the curve bends sharply upward, and at
p = 0.01 the loss is 4.61. The red dots are the table above. The shape is the
whole story: small, diminishing rewards for being more right, and an
unbounded bill for being confidently wrong.

The **natural logarithm** ln(p) answers "e to what power gives p?" For a
probability between 0 and 1 the answer is negative (ln 0.5 = −0.69,
because e^−0.69 = 0.5), so we flip the sign to get a positive cost. (Every symbol used in these
lessons is built from zero in `primer.notation`.)

$$
\mathcal{L} = -\ln p_{\text{correct}}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $\mathcal{L}$ | the loss for one prediction | 0.69 |
| $p_{\text{correct}}$ | probability the model gave to the right answer, between 0 and 1 | 0.5 |
| $\ln$ | natural logarithm: the power you'd raise $e \approx 2.718$ to, to get the input | $\ln 0.5 = -0.69$ |
| $-$ | flips the sign so the loss is positive | |

**In words:** "the loss is minus the natural log of the probability the
model gave the right answer."

**With the numbers:** $-\ln 0.5 = 0.69$; $-\ln 0.01 = 4.61$.

`cross_entropy_from_prob` is this one line.

**Why it matters:** a language model predicts the next token by choosing
among its whole vocabulary, so this *is* the pretraining loss of every LLM.
The steep penalty for confident mistakes is what pushes models toward
calibrated probabilities.

## From logits: the numerically safe way

A model doesn't output probabilities directly. It outputs raw scores called
**logits**, which softmax turns into probabilities (raise *e* to each score,
divide by the total; see `primer.ml.attention`). Computing that literally is
like measuring everyone's height in millimetres from the centre of the
Earth: the numbers get astronomically large and your calculator overflows.
Measure relative to the tallest person instead, and every number stays
small. That trick is **log-sum-exp**.

Worked example: logits (2, 1, 0.1), correct class 0.
e² + e¹ + e^0.1 = 7.389 + 2.718 + 1.105 = 11.212, ln 11.212 = 2.417, so the
loss is 2.417 − 2 = **0.417**. And with logits (1000, 0) and class 1 correct,
e^1000 overflows any computer, but log-sum-exp gives exactly 1000 − 0 = **1000**.

```mermaid
flowchart LR
  Z[Logits z<br/>one score per class] --> M[Subtract max m]
  M --> LSE[log-sum-exp<br/>m + ln sum e^z-m]
  LSE --> L[Loss = LSE minus z_correct]
  Z --> L
  L -.backward.-> G[Gradient = softmax z<br/>minus one-hot target]
  G -.-> U[Raise the right logit,<br/>lower the others by their probability]
```

**Reading it:** solid arrows are the forward pass, dotted arrows the backward
pass. The logits never go through an explicit softmax on the way forward:
the largest logit is subtracted first so no exponent can overflow, and the
loss is "log of the total" minus "the right class's score". Coming back, the
gradient for every class is its predicted probability, minus 1 for the
correct class. So the right logit is pushed up by (1 − p), and each wrong
logit is pushed down by exactly the probability it took.

$$
-\ln \text{softmax}(z)_y = \ln \sum_j e^{z_j} - z_y,
\qquad
\ln \sum_j e^{z_j} = m + \ln \sum_j e^{z_j - m},\; m = \max_j z_j
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $z$ | the logits, one raw score per class | (2, 1, 0.1) |
| $z_j$ | the score of class $j$ | $z_0 = 2$ |
| $y$ | the index of the correct class | 0 |
| $\text{softmax}(z)_y$ | the probability softmax gives the correct class | 7.389 / 11.212 = 0.659 |
| $\sum_j$ | "add up over every class $j$" | three terms |
| $e^{z_j}$ | *e* ≈ 2.718 raised to the score | $e^2 = 7.389$ |
| $m$ | the largest logit, subtracted for safety | 2 |
| $\max_j$ | "the biggest value over all $j$" | 2 |

**In words:** "the loss is the log of the sum of e-to-every-score, minus the
correct class's score; to compute that log safely, pull the biggest score
out front first."

**With the numbers:** $m = 2$; $\sum_j e^{z_j - 2} = e^0 + e^{-1} + e^{-1.9} =
1 + 0.368 + 0.150 = 1.518$; $\ln 1.518 = 0.417$; so log-sum-exp $= 2.417$ and the
loss is $2.417 - 2 = 0.417$ (and $-\ln 0.659 = 0.417$ too).

The gradient, which is how each logit should move:

$$
\frac{\partial \mathcal{L}}{\partial z} = \text{softmax}(z) - \text{onehot}(y)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $\partial \mathcal{L} / \partial z$ | the gradient: one slope per logit | (0.25, −0.25) |
| $\text{softmax}(z)$ | the predicted probabilities | (0.25, 0.75) for logits (0, ln 3) |
| $\text{onehot}(y)$ | 1 at the correct class, 0 elsewhere | (0, 1) |

**In words:** "each logit's slope is its predicted probability, minus one
if it's the right answer."

**With the numbers:** logits (0, ln 3) give softmax (1/4, 3/4); with class 1
correct, the gradient is (0.25 − 0, 0.75 − 1) = (0.25, −0.25).

**Why it matters:** this is why every framework fuses softmax and
cross-entropy into one operation that takes logits
(`torch.nn.functional.cross_entropy`). Computing softmax first and then the
log is the classic source of NaN losses.

## Perplexity: how many options is the model torn between?

Imagine a game show with doors, one hiding the prize. If the model is as
unsure as someone picking between two doors at random, its perplexity is 2;
between ten doors, 10. Perplexity converts an average loss back into that
"number of doors".

Worked example: a model that gives the right token 50% every time has loss
0.69 per token and perplexity e^0.69 = 2. One that gives 10% every time has
perplexity 10. Three confident tokens (0.9) and one disaster (0.001) give
perplexity about 6: one bad guess drags the whole average.

```mermaid
flowchart LR
  P["Per-token probabilities<br/>0.5, 0.5, 0.5"] --> NL["−ln each<br/>0.69, 0.69, 0.69"]
  NL --> AV["Average<br/>0.69"]
  AV --> EX["e to that power<br/>e^0.69 = 2"]
  EX --> D["≈ choosing between<br/>2 equally likely doors"]
```

**Reading it:** start with the probability the model gave each correct
token, turn each into a loss with −ln, average them, then undo the log with
e^x. The result is back in "number of options" units.

$$
\text{PPL} = \exp\left(\frac{1}{N}\sum_{i=1}^{N} -\ln p_i\right)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $\text{PPL}$ | perplexity | 2 |
| $N$ | number of tokens | 3 |
| $i$ | a counter over the tokens | 1, 2, 3 |
| $p_i$ | probability given to the correct token at position $i$ | 0.5 each |
| $\frac{1}{N}\sum_{i=1}^{N}$ | "the average over all $N$ tokens" | $(0.69 + 0.69 + 0.69)/3$ |
| $\exp(x)$ | another way to write $e^x$ | $e^{0.69}$ |

**In words:** "perplexity is e raised to the average cross-entropy per
token."

**With the numbers:** $\exp\left(\frac{1}{3}(0.69 \times 3)\right) = e^{0.69} = 2.0$.

**Why it matters:** perplexity is the standard training metric for language
models. It's comparable only between models that use the same tokenizer on
the same text, and it says nothing direct about whether answers are
helpful or correct.

## Regression losses: fines for being off by an amount

When the prediction is a number (a price, a temperature), think of fines for
arriving late. **MAE** (mean absolute error) is a flat rate: every minute late
costs the same. **MSE** (mean squared error) squares the minutes: 10 minutes
late costs 100, not 10, so one very late arrival outweighs many slightly late
ones.

Worked example: four predictions miss by 1 and one misses by 10.
MSE = (1 + 1 + 1 + 1 + 100) / 5 = **20.8**, and the outlier is 100/104 = 96% of
it. MAE = (1 + 1 + 1 + 1 + 10) / 5 = **2.8**, and the outlier is 10/14 = 71%.

![MSE vs. MAE penalty per error, and where each puts the best constant prediction](figures/primer.ml.losses.mse_vs_mae.svg)

**Reading it:** the left panel is the price of a single error. Near zero the
two curves are similar, but past an error of 1 the squared penalty climbs
away from the absolute one. The right panel asks: if the model could predict
only one constant for the data 1, 2, 2, 3, 3, 4 plus an outlier of 20, which
constant minimizes each loss? MSE's valley sits at the mean (5.0), dragged
right by the outlier; MAE's valley sits at the median (3), where most of the
data is.

$$
\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2, \qquad
\text{MAE} = \frac{1}{N}\sum_{i=1}^{N}\lvert y_i - \hat{y}_i\rvert
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $N$ | number of predictions | 5 |
| $y_i$ | the true value for example $i$ | 0 for all five |
| $\hat{y}_i$ | "y-hat", the predicted value | 1, −1, 1, −1, 10 |
| $y_i - \hat{y}_i$ | the error (residual) | −1, 1, −1, 1, −10 |
| $(\cdot)^2$ | square: multiply by itself (always positive) | $(-10)^2 = 100$ |
| $\lvert\cdot\rvert$ | absolute value: drop the sign | $\lvert -10\rvert = 10$ |

**In words:** "MSE is the average of the squared errors; MAE is the average
of the errors ignoring their sign."

**With the numbers:** MSE $= (1+1+1+1+100)/5 = 20.8$; MAE $= (1+1+1+1+10)/5 = 2.8$.

**Why it matters:** pick the loss whose valley is where you want your
predictions. MSE chases outliers (its best constant is the mean); MAE
shrugs them off (its best constant is the median). For noisy data with
occasional wild values, MAE or a blend (Huber loss) is safer.

## Contrastive loss (InfoNCE): find your partner in a crowd

Picture a party game. Everyone arrives in pairs, gets separated, and must
pick their partner out of the whole room. Everyone else in the room is a
decoy. A contrastive loss scores how confidently each person picks their own
partner over every decoy. The hardest decoy is your partner's lookalike
twin: same topic, wrong person. That's a **hard negative**.

Embedding models (and CLIP) learn this way: a query and the passage that
answers it are "partners"; the other passages in the same batch are free
decoys (**in-batch negatives**).

Worked example: two queries and two passages, as vectors, scored with the
**dot product** (multiply matching entries, add them up). Query 1 = (1, 0),
query 2 = (0, 1), passage 1 = (1, 0), passage 2 = (0, 1), temperature 1.
Query 1 scores 1 against its partner and 0 against the decoy. Its loss is
−ln(e¹ / (e¹ + e⁰)) = ln(1 + e^−1) = **0.3133**. Swap the passages and each
query now prefers the decoy: the loss rises to ln(1 + e¹) = **1.3133**.

```mermaid
flowchart LR
  Q[Batch of queries] --> S[Score matrix<br/>every query vs every passage]
  D[Their matching passages] --> S
  S --> CE[Cross-entropy per row<br/>right answer = the diagonal]
  CE --> L[Pull pairs together<br/>push the rest apart]
```

**Reading it:** a batch of queries and the passages that answer them are
turned into vectors, then every query is scored against every passage,
giving a square grid. Each row becomes a classification ("which of these
passages is mine?") whose right answer is on the diagonal. Cross-entropy on
those rows pulls each query toward its own passage and away from all the
others at once.

![InfoNCE probability matrix for a batch with one hard negative](figures/primer.ml.losses.infonce_matrix.svg)

**Reading it:** rows are queries, columns are passages, and each cell is the
probability that query i "picks" passage j. A well-trained model shows a
bright diagonal. Look at row 2: it puts most of its probability on column 4,
an extra mined hard negative built to resemble query 2, and only a few
percent on its own passage (column 2). The model is fooled, and that row is
exactly where the loss, and therefore the gradient, concentrates. Rows 0, 1
and 3 are already confident and contribute almost nothing to learning.

$$
\mathcal{L}_i = -\ln \frac{\exp(q_i \cdot d_i / \tau)}{\sum_{j=1}^{N} \exp(q_i \cdot d_j / \tau)}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $\mathcal{L}_i$ | the loss for query $i$ | 0.3133 |
| $q_i$ | query $i$'s vector | $q_1 = (1, 0)$ |
| $d_i$ | query $i$'s own passage (its positive) | $d_1 = (1, 0)$ |
| $d_j$ | passage $j$: every passage in the batch, the positive included | $d_1, d_2$ |
| $q_i \cdot d_j$ | dot product: how aligned query and passage are | $q_1 \cdot d_1 = 1$, $q_1 \cdot d_2 = 0$ |
| $\tau$ | "tau", the temperature: divides scores; small $\tau$ sharpens, large softens | 1 |
| $N$ | number of passages scored (batch plus any extra negatives) | 2 |
| $\exp$ | $e$ raised to the power | $e^1 = 2.718$ |

**In words:** "for each query, take e-to-the-score of its own passage,
divide by the sum of e-to-the-score over every passage in the batch, and
charge minus the log of that share."

**With the numbers:** $-\ln \frac{e^{1}}{e^{1} + e^{0}} = -\ln \frac{2.718}{3.718}
= -\ln 0.731 = 0.3133$. At temperature 0.1 the same vectors give
$-\ln \frac{e^{10}}{e^{10} + 1} \approx 0.000045$: sharper.

It's just cross-entropy where the "classes" are the passages in the batch,
so the gradient is the same softmax − onehot, pushed back through the dot
products into both sets of vectors.

**Why it matters:** this is how search and RAG embedding models are trained
(see `primer.ml.embeddings.contrastive`). Bigger batches mean more free
negatives. Easy negatives are already far away and contribute almost no
gradient; hard negatives produce most of the learning signal, which is why
mining them is the biggest lever on retrieval quality.

## Label smoothing: never say 100%

A good forecaster never says "100% chance of rain", because the one day
they're wrong would be infinitely embarrassing. Label smoothing teaches the
model the same humility: instead of "the answer is class 2, with total
certainty", the target says "class 2, with 92.5% certainty, and a sliver of
doubt spread over everything".

Worked example: 4 classes, correct class 2, smoothing ε = 0.1. Take 0.9 of
the one-hot target (0, 0, 0.9, 0) and add 0.1/4 = 0.025 to every class:
(0.025, 0.025, 0.925, 0.025). A model that puts logit 50 on the right class
has plain cross-entropy ≈ 0, but smoothed cross-entropy 3 × 0.025 × 50 = **3.75**.

```mermaid
flowchart LR
  O[One-hot target<br/>0, 0, 1, 0] --> MIX[Mix: 1 minus eps times one-hot<br/>plus eps/K everywhere]
  U[Uniform over K classes] --> MIX
  MIX --> T[Soft target<br/>0.025, 0.025, 0.925, 0.025]
  T --> CE[Cross-entropy against<br/>the soft target]
```

**Reading it:** the hard one-hot label and a uniform distribution are
blended with weight ε, giving a target that is still 92.5% sure but never
100%. Training against it means the loss can never reach zero, so the model
gains nothing by pushing its logits toward infinity.

$$
t = (1 - \varepsilon)\,\text{onehot}(y) + \frac{\varepsilon}{K}, \qquad
\mathcal{L} = -\sum_{k=1}^{K} t_k \ln \text{softmax}(z)_k
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $t$ | the smoothed target distribution | (0.025, 0.025, 0.925, 0.025) |
| $\varepsilon$ | "epsilon", how much certainty to give away | 0.1 |
| $K$ | number of classes | 4 |
| $k$ | a counter over the classes | 1 to 4 |
| $t_k$ | target weight on class $k$ | 0.925 on the right class |
| $\text{softmax}(z)_k$ | predicted probability of class $k$ | ≈1 on the right class |

**In words:** "the target keeps (1 − ε) on the right answer and spreads ε
evenly over all classes; the loss is cross-entropy against that softer
target."

**With the numbers:** $(1 - 0.1) \cdot 1 + 0.1/4 = 0.925$ on the right class
and $0.025$ elsewhere; with logits (50, 0, 0, 0) the three wrong classes each
have $\ln \text{softmax} \approx -50$, so $\mathcal{L} \approx 3 \times 0.025 \times 50 = 3.75$.

**Why it matters:** it curbs over-confidence and often improves calibration
(how well the model's stated confidence matches how often it's right). It
was used to train the original transformer.

## In 20 seconds
- Cross-entropy is −ln(probability on the right answer): small when
  confident and right, huge when confident and wrong.
- Perplexity = e^(average cross-entropy): the effective number of choices
  per token.
- Compute it from logits with log-sum-exp; its gradient is softmax − onehot.
- MSE punishes big misses (mean); MAE is robust to outliers (median).
- Contrastive (InfoNCE) loss is cross-entropy over "which passage in this
  batch is mine?"; hard negatives drive retrieval quality.

## Self-test questions

**The model puts 1% on the right token. What's the loss, and why so large?**
−ln(0.01) = 4.61, about 40× the loss at 90%. The log punishes confident
mistakes steeply, which pushes the model toward calibrated probabilities.

**Loss is 0.69 per token. What's the perplexity?**
e^0.69 = 2: as uncertain as a coin flip between two options.

**Why compute cross-entropy from logits instead of from softmax outputs?**
Softmax of large logits overflows, and the log of a tiny probability
underflows to −∞. Log-sum-exp subtracts the max first and never
exponentiates a large number. The fused gradient is simply softmax − onehot.

**When would you pick MAE over MSE?**
When outliers are noise you don't want to chase. MSE squares errors, so one
huge miss dominates; MAE weighs errors linearly.

**How does InfoNCE get negatives without labelling them?**
It uses the other items in the batch: for each query, every other query's
positive passage is a negative. Bigger batches mean more (and harder)
negatives for free.

**Why do hard negatives matter?**
Easy negatives are already far away and contribute almost no gradient. Hard
negatives score high and produce most of the learning signal, teaching the
model to separate "on topic" from "actually answers the question".

## The papers behind this lesson

- van den Oord, Li & Vinyals, *Representation Learning with Contrastive Predictive Coding* (2018): https://arxiv.org/abs/1807.03748
  Named and analysed InfoNCE, the "pick the positive out of N" contrastive loss.
- Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR, 2020): https://arxiv.org/abs/2002.05709
  Showed how much in-batch negatives, large batches and the temperature matter for contrastive training.
- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP, 2021): https://arxiv.org/abs/2103.00020
  Trained image and text encoders with a symmetric InfoNCE loss on 400 million pairs, putting both in one vector space. [annotated companion](../../papers/clip.html)
- Szegedy et al., *Rethinking the Inception Architecture for Computer Vision* (2015): https://arxiv.org/abs/1512.00567
  Introduced label smoothing as a regularizer against over-confident predictions.

## Further reading
- Goodfellow, Bengio & Courville, *Deep Learning*, ch. 6 (cost functions): https://www.deeplearningbook.org/contents/mlp.html
- CS231n notes, *Linear classification* (softmax and cross-entropy): https://cs231n.github.io/linear-classify/
- PyTorch `cross_entropy`: https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
- van den Oord et al., *Representation Learning with Contrastive Predictive Coding* (InfoNCE, 2018): https://arxiv.org/abs/1807.03748
- Chen et al., *SimCLR* (in-batch negatives and temperature, 2020): https://arxiv.org/abs/2002.05709
- Radford et al., *CLIP* (2021): https://arxiv.org/abs/2103.00020
- Szegedy et al., *Rethinking the Inception Architecture* (label smoothing, 2015): https://arxiv.org/abs/1512.00567
- Müller et al., *When Does Label Smoothing Help?* (2019): https://arxiv.org/abs/1906.02629
"""

from __future__ import annotations

import numpy as np

from primer._show import banner, say, table, takeaway

# ---------------------------------------------------------------------------
# 1. Cross-entropy and perplexity
# ---------------------------------------------------------------------------


def cross_entropy_from_prob(p_correct: float | np.ndarray) -> float | np.ndarray:
    """−ln(p) for the probability assigned to the correct answer."""
    return -np.log(p_correct)


def perplexity(p_correct: np.ndarray) -> float:
    """exp(mean cross-entropy) over a sequence of per-token probabilities on the right token.

    Averaging in log space then exponentiating is a *geometric* mean of 1/p,
    so one catastrophic token (p ≈ 0) hurts a lot.
    """
    return float(np.exp(np.mean(cross_entropy_from_prob(np.asarray(p_correct, dtype=float)))))


# ---------------------------------------------------------------------------
# 2. Cross-entropy from logits (what frameworks actually do)
# ---------------------------------------------------------------------------


def log_sum_exp(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """ln Σ e^z, computed without overflow by factoring out the max."""
    m = z.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def log_softmax(z: np.ndarray) -> np.ndarray:
    return z - log_sum_exp(z)[..., None]


def softmax_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean cross-entropy over a batch, and its gradient with respect to the logits.

    Args:
        logits: (batch, n_classes) raw scores.
        targets: (batch,) integer class ids.

    Returns:
        (loss, grad) where grad has the logits' shape.
    """
    n = logits.shape[0]
    rows = np.arange(n)
    # −ln softmax(z)_y = logsumexp(z) − z_y. Never forms a probability that could underflow.
    loss = float(np.mean(log_sum_exp(logits) - logits[rows, targets]))
    # Gradient: softmax(z) − onehot(y), divided by n because the loss is a mean.
    probs = np.exp(log_softmax(logits))
    grad = probs.copy()
    grad[rows, targets] -= 1.0
    return loss, grad / n


# ---------------------------------------------------------------------------
# 3. Regression losses
# ---------------------------------------------------------------------------


def mse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Mean squared error. Squaring makes one big miss outweigh many small ones."""
    return float(np.mean((np.asarray(y) - np.asarray(y_hat)) ** 2))


def mae(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Mean absolute error. Every unit of error costs the same, so outliers pull less."""
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(y_hat))))


def outlier_share(y: np.ndarray, y_hat: np.ndarray, kind: str = "mse") -> float:
    """Fraction of the total loss contributed by the single largest error."""
    err = np.abs(np.asarray(y) - np.asarray(y_hat))
    per_item = err**2 if kind == "mse" else err
    return float(per_item.max() / per_item.sum())


# ---------------------------------------------------------------------------
# 4. Contrastive loss (InfoNCE) with in-batch negatives
# ---------------------------------------------------------------------------


def info_nce(q: np.ndarray, d: np.ndarray, temperature: float = 0.05) -> tuple[float, np.ndarray, np.ndarray]:
    """InfoNCE loss and its gradients with respect to the query and document embeddings.

    Args:
        q: (B, dim) query embeddings. Query i's positive is d[i].
        d: (N, dim) document embeddings, N >= B. Rows B..N-1 are extra
            (e.g. mined hard) negatives; every other row is an in-batch negative.
        temperature: τ. Real embedding models use ~0.01 to 0.1 on unit vectors.

    Returns:
        (loss, dL/dq, dL/dd).
    """
    B = q.shape[0]
    # (B, N): row i is "how much does query i like each document".
    logits = q @ d.T / temperature
    # Each row is a classification whose right answer is column i (the diagonal).
    loss, dlogits = softmax_cross_entropy(logits, np.arange(B))
    # Chain rule through logits = q dᵀ / τ.
    dq = dlogits @ d / temperature
    dd = dlogits.T @ q / temperature
    return loss, dq, dd


def info_nce_gradient_check(q: np.ndarray, d: np.ndarray, temperature: float = 0.05, eps: float = 1e-6) -> float:
    """Max relative error between info_nce's analytic gradients and central differences."""
    _, dq, dd = info_nce(q, d, temperature)
    worst = 0.0
    for X, G in ((q, dq), (d, dd)):
        for idx in np.ndindex(X.shape):
            old = X[idx]
            X[idx] = old + eps
            lp = info_nce(q, d, temperature)[0]
            X[idx] = old - eps
            lm = info_nce(q, d, temperature)[0]
            X[idx] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - G[idx]) / max(1e-8, abs(num) + abs(G[idx])))
    return float(worst)


# ---------------------------------------------------------------------------
# 5. Label smoothing
# ---------------------------------------------------------------------------


def smoothed_targets(targets: np.ndarray, n_classes: int, smoothing: float = 0.1) -> np.ndarray:
    """(1 − ε)·onehot + ε/K: most mass on the right class, a little on every class."""
    t = np.full((len(targets), n_classes), smoothing / n_classes)
    t[np.arange(len(targets)), targets] += 1.0 - smoothing
    return t


def smoothed_cross_entropy(logits: np.ndarray, targets: np.ndarray, smoothing: float = 0.1) -> float:
    """Cross-entropy against the smoothed target distribution: −Σ t · log_softmax(z)."""
    t = smoothed_targets(targets, logits.shape[1], smoothing)
    return float(np.mean(-(t * log_softmax(logits)).sum(axis=1)))


# ---------------------------------------------------------------------------
# 6. Figures (rendered by `make figures`)
# ---------------------------------------------------------------------------


def figures() -> dict:
    """Plots computed from this module's own functions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = {}

    # Cross-entropy curve with the worked-example points.
    fig, ax = plt.subplots(figsize=(6, 3.6))
    p = np.linspace(0.005, 1, 400)
    ax.plot(p, cross_entropy_from_prob(p), color="C0")
    for pt in (0.9, 0.5, 0.1, 0.01):
        ax.scatter([pt], [cross_entropy_from_prob(pt)], color="C3", zorder=3)
        ax.annotate(f"p={pt}\nloss={cross_entropy_from_prob(pt):.2f}", (pt, cross_entropy_from_prob(pt)),
                    textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax.set(xlabel="probability the model gave the correct answer", ylabel="loss = −ln p",
           title="Cross-entropy punishes confident mistakes")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figs["cross_entropy"] = fig

    # MSE vs MAE: per-error penalty and the effect on a constant fit.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    r = np.linspace(-4, 4, 200)
    a1.plot(r, r**2, label="squared (MSE)")
    a1.plot(r, np.abs(r), label="absolute (MAE)")
    a1.set(xlabel="error y − ŷ", ylabel="penalty", title="Penalty per error")
    a1.legend()
    a1.grid(alpha=0.3)
    data = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 20.0])
    cs = np.linspace(0, 12, 300)
    a2.plot(cs, [mse(data, np.full_like(data, c)) for c in cs] / np.max([mse(data, np.full_like(data, c)) for c in cs]),
            label=f"MSE (min at mean {data.mean():.1f})")
    a2.plot(cs, [mae(data, np.full_like(data, c)) for c in cs] / np.max([mae(data, np.full_like(data, c)) for c in cs]),
            label=f"MAE (min at median {np.median(data):.0f})")
    a2.set(xlabel="constant prediction c", ylabel="loss (scaled to max 1)",
           title="Data 1,2,2,3,3,4 and outlier 20")
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    figs["mse_vs_mae"] = fig

    # InfoNCE: probability matrix for a batch where query 2's hard negative is document 3.
    q, d = _contrastive_batch()
    probs = np.exp(log_softmax(q @ d.T / 0.1))
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(probs, cmap="viridis", vmin=0, vmax=1)
    for i in range(probs.shape[0]):
        for j in range(probs.shape[1]):
            ax.text(j, i, f"{probs[i, j]:.2f}", ha="center", va="center",
                    color="white" if probs[i, j] < 0.5 else "black", fontsize=8)
    ax.set(xlabel="document j", ylabel="query i", title="InfoNCE: softmax over each row (τ = 0.1)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    figs["infonce_matrix"] = fig

    return figs


def _contrastive_batch() -> tuple[np.ndarray, np.ndarray]:
    """Four unit-length query/doc pairs plus doc 4, a mined hard negative for query 2."""
    rng = np.random.default_rng(0)
    q = rng.standard_normal((4, 16))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    positives = q + 0.3 * rng.standard_normal((4, 16))  # noisy copies of their queries
    hard_negative = q[2] + 0.3 * rng.standard_normal(16)  # same topic as query 2, not its answer
    d = np.vstack([positives, hard_negative])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return q, d


# ---------------------------------------------------------------------------
# 7. Walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. Cross-entropy: −ln(probability on the right answer)")
    ps = [0.9, 0.5, 0.1, 0.01]
    table(["p(correct)", "loss −ln p"], [(p, cross_entropy_from_prob(p)) for p in ps], floatfmt=".2f")
    ratio = cross_entropy_from_prob(0.01) / cross_entropy_from_prob(0.9)
    say(f"Confidently wrong (1%) costs {ratio:.0f}x more than mostly right (90%).")
    takeaway("Cross-entropy heavily punishes confident mistakes, which pushes models toward calibrated probabilities.")

    banner("2. Perplexity: the effective number of choices")
    table(
        ["per-token p on the right token", "mean loss", "perplexity"],
        [
            ("0.5 every time", float(np.mean(cross_entropy_from_prob(np.full(4, 0.5)))), perplexity(np.full(4, 0.5))),
            ("0.1 every time", float(np.mean(cross_entropy_from_prob(np.full(4, 0.1)))), perplexity(np.full(4, 0.1))),
            ("0.9, 0.9, 0.9, 0.001", float(np.mean(cross_entropy_from_prob(np.array([0.9, 0.9, 0.9, 0.001])))),
             perplexity(np.array([0.9, 0.9, 0.9, 0.001]))),
        ],
        floatfmt=".2f",
    )
    say("One catastrophic token (p = 0.001) drags perplexity from ~1.1 to ~6: it's a geometric mean.")

    banner("3. From logits: log-sum-exp and the softmax − onehot gradient")
    logits = np.array([[2.0, 1.0, 0.1]])
    loss, grad = softmax_cross_entropy(logits, np.array([0]))
    say(
        f"""
        Logits (2, 1, 0.1), correct class 0. Loss = ln(e²+e¹+e⁰·¹) − 2 =
        {loss:.3f}. Gradient = softmax − onehot = {np.round(grad[0], 3)}: push
        the right logit up, the others down, in proportion to their probability.
        """
    )
    big, _ = softmax_cross_entropy(np.array([[1000.0, 0.0]]), np.array([1]))
    say(f"Logits (1000, 0) with class 1 correct: naive softmax overflows; log-sum-exp gives exactly {big:.0f}.")

    banner("4. MSE vs. MAE: what one outlier does")
    y, y_hat = np.zeros(5), np.array([1.0, -1.0, 1.0, -1.0, 10.0])
    table(
        ["loss", "value", "share from the outlier"],
        [("MSE", mse(y, y_hat), outlier_share(y, y_hat, "mse")), ("MAE", mae(y, y_hat), outlier_share(y, y_hat, "mae"))],
        floatfmt=".2f",
    )
    takeaway("MSE chases outliers (its best constant is the mean); MAE shrugs them off (its best constant is the median).")

    banner("5. InfoNCE: every other passage in the batch is a free negative")
    q, d = _contrastive_batch()
    loss, _, _ = info_nce(q, d, temperature=0.1)
    per_row = -np.diag(log_softmax(q @ d.T / 0.1))
    table(["query", "loss for this row"], [(i, per_row[i]) for i in range(4)], floatfmt=".3f")
    say(
        f"""
        Mean loss {loss:.3f}. Query 2 dominates: document 4 is a mined hard
        negative that looks like it, so the model is fooled. Easy rows add almost nothing;
        hard negatives are where the learning happens.
        """
    )

    banner("6. Label smoothing")
    t = smoothed_targets(np.array([2]), 4, 0.1)[0]
    say(f"One-hot (0, 0, 1, 0) with ε = 0.1 over 4 classes becomes {np.round(t, 3)}.")
    confident = np.array([[50.0, 0.0, 0.0, 0.0]])
    say(
        f"""
        A prediction with logit 50 on the right class has plain cross-entropy
        {softmax_cross_entropy(confident, np.array([0]))[0]:.1e} but smoothed
        cross-entropy {smoothed_cross_entropy(confident, np.array([0])):.2f}:
        extreme confidence is now penalized.
        """
    )


if __name__ == "__main__":
    demo()
