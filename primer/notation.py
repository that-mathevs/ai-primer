r"""
# Math notation, from zero

Run: `python -m primer.notation`

Machine learning papers look impenetrable mostly because of **notation**:
Greek letters, big sigmas, little superscript Ts. Almost every symbol is
shorthand for a short loop you could write in a few lines of Python. This
lesson writes each one out as that loop, so that when a formula appears
elsewhere in this primer you can read it aloud.

Every function below is deliberately written the slow, obvious way, with
plain Python loops, so the code *is* the definition. The tests check each one
against NumPy, which does the same thing fast.

## Lists and tables of numbers: vectors and matrices

**Everyday picture.** A **vector** is a list of numbers, like a shopping
receipt: (apples 3, bread 1, milk 2). A **matrix** is a table of numbers, like
a spreadsheet with rows and columns.

**Tiny example.** $x = (3, 1, 2)$ is a vector with 3 entries. Its entries are
written with a small **subscript**: $x_1 = 3$, $x_2 = 1$, $x_3 = 2$. A matrix
$A$ with 2 rows and 3 columns has **shape** $2 \times 3$, and $A_{2,3}$ means
"row 2, column 3".

| You see | Say | Python |
|---|---|---|
| $x$ (lowercase, sometimes bold **x**) | "the vector x" | `x = [3, 1, 2]` |
| $x_i$ | "x sub i": the i-th entry | `x[i - 1]` (maths counts from 1, Python from 0) |
| $A$ (uppercase) | "the matrix A" | `A = [[1, 2, 3], [4, 5, 6]]` |
| $A_{ij}$ or $A_{i,j}$ | row i, column j | `A[i - 1][j - 1]` |
| $\mathbb{R}^d$ | "the set of all lists of d real numbers" | any list of d floats |
| $x \in \mathbb{R}^{768}$ | "x is a list of 768 numbers" | `len(x) == 768` |
| $n \times d$ | shape: n rows, d columns | `np.zeros((n, d))` |

In practice, a word's embedding is a vector, a batch of embeddings is a
matrix (one row per word), and a model's weights are mostly matrices.

## Σ (capital sigma): add them all up

**Everyday picture.** Totting up a receipt.

$$
\sum_{i=1}^{n} x_i = x_1 + x_2 + \cdots + x_n
$$

**Symbols**

| Symbol | Meaning |
|---|---|
| $\sum$ | "sum": add up everything that follows |
| $i = 1$ (below) | start a counter called $i$ at 1 |
| $n$ (above) | stop after the counter reaches $n$ |
| $x_i$ | the thing being added on each step |
| $\cdots$ | "and so on, following the same pattern" |

**In words:** "for i from 1 to n, add up x sub i."

**With the numbers:** $\sum_{i=1}^{4} i = 1 + 2 + 3 + 4 = 10$. In code, Σ is a
`for` loop with a running total (`summation` below).

**In Python:**

```python
x = [1, 2, 3, 4]
total = 0
# Σ: visit each x_i, from i = 1 to n ...
for x_i in x:
    # ... and add it to a running total
    total += x_i
total  # → 10
# Python's built-in sum is the same loop
sum(x)  # → 10
```

**Π (capital pi)** is the same idea with multiplication:
$\prod_{i=1}^{4} i = 1 \times 2 \times 3 \times 4 = 24$. It shows up whenever
independent chances combine. For example, ten steps that each succeed 95%
of the time all succeed with probability $\prod 0.95 = 0.95^{10} \approx 0.60$.
This single fact explains why long chains of AI agent steps fail so often
(`primer.agents.planning`).

**In Python:**

```python
x = [1, 2, 3, 4]
product = 1
# Π: the same loop, multiplying instead of adding
for x_i in x:
    product *= x_i
product  # → 24
# ten steps that each succeed 95% of the time
round(0.95 ** 10, 2)  # → 0.6
```

**In code:** `product` is Π written the same way: a loop with a running product that starts at 1.

## The dot product: how much two lists agree

**Everyday picture.** A recipe needs 2 eggs, 3 cups of flour and 1 cup of
sugar, and eggs cost \$1, flour \$0.50 and sugar \$2 per unit. The total
cost is 2·1 + 3·0.5 + 1·2 = \$5.50: multiply matching items, then add. That
is a dot product.

$$
a \cdot b = \sum_{i=1}^{d} a_i \, b_i
$$

**Symbols**

| Symbol | Meaning |
|---|---|
| $a, b$ | two vectors with the same number of entries, $d$ |
| $a_i b_i$ | the $i$-th entries multiplied together |
| $\cdot$ | "dot": the whole multiply-then-add operation |

**In words:** "multiply the lists position by position and add up the
products."

**With the numbers:** $(1, 2) \cdot (3, 0.5) = 1 \cdot 3 + 2 \cdot 0.5 = 4$.

**In Python:**

```python
a = [1, 2]
b = [3, 0.5]
# Σ over i of a_i times b_i
sum(a_i * b_i for a_i, b_i in zip(a, b))  # → 4.0
```

Geometrically, the dot product is large when two arrows point the same way,
zero when they are at right angles, and negative when they point apart.
That is why it is the standard similarity score for attention and for
embeddings.

```mermaid
flowchart LR
  A["a = (1, 2)"] --> M1["1 × 3 = 3"]
  B["b = (3, 0.5)"] --> M1
  A --> M2["2 × 0.5 = 1"]
  B --> M2
  M1 --> S["3 + 1 = 4"]
  M2 --> S
```

**Reading it:** each pair of matching positions meets in a multiply box,
first entries with first entries and second with second. Every product then
flows into one addition. A dot product is always this shape: many
multiplications feeding one sum, however long the lists get.

**In code:** `dot` pairs up matching entries, multiplies them and adds the products with `summation`, refusing lists of different lengths.

## ‖x‖: the length of a vector

**Everyday picture.** Walk 3 blocks east and 4 blocks north. As the crow
flies, you are 5 blocks from where you started (Pythagoras).

$$
\lVert x \rVert = \sqrt{\sum_{i=1}^{d} x_i^2} = \sqrt{x \cdot x}
$$

**Symbols**

| Symbol | Meaning |
|---|---|
| $\lVert x \rVert$ | the **norm** (length) of $x$, also written $\lVert x \rVert_2$ |
| $x_i^2$ | the $i$-th entry squared ($x_i \times x_i$) |
| $\sqrt{\ }$ | square root |

**In words:** "square every entry, add them up, and take the square root."

**With the numbers:** $\lVert (3, 4) \rVert = \sqrt{9 + 16} = 5$ and
$\lVert (1, 2, 2) \rVert = \sqrt{1 + 4 + 4} = 3$.

**In Python:**

```python
import math
def length(x):
    # √ of Σ x_i²
    return math.sqrt(sum(x_i ** 2 for x_i in x))
length([3, 4])  # → 5.0
length([1, 2, 2])  # → 3.0
```

Dividing a vector by its length gives a **unit vector** of length 1 that
points the same way. Embedding systems do this constantly, because then the
dot product measures only direction (see
`primer.ml.embeddings.similarity`).

**In code:** `norm` is the square root of a vector's dot product with itself.

## Matrix multiply and transpose

**Transpose**, written $A^\top$ ("A transpose"), flips a table on its
diagonal so rows become columns: a $2 \times 3$ matrix becomes $3 \times 2$.

**Matrix multiply**, written $AB$ with nothing in between, is a whole grid of
dot products. The cell in row $i$, column $j$ of the answer is row $i$ of $A$
dotted with column $j$ of $B$.

$$
(AB)_{ij} = \sum_{k=1}^{m} A_{ik} \, B_{kj}
$$

**Symbols**

| Symbol | Meaning |
|---|---|
| $A$ | an $n \times m$ matrix |
| $B$ | an $m \times p$ matrix; its row count must equal $A$'s column count, $m$ |
| $(AB)_{ij}$ | the answer's cell at row $i$, column $j$; the answer is $n \times p$ |
| $k$ | the counter that walks along row $i$ of $A$ and down column $j$ of $B$ together |

**In words:** "to fill cell (i, j), walk across row i of A and down column j
of B at the same pace, multiplying and adding."

**With the numbers:**
$\begin{pmatrix}1&2\\3&4\end{pmatrix}\begin{pmatrix}5&6\\7&8\end{pmatrix}
= \begin{pmatrix}1\cdot5+2\cdot7 & 1\cdot6+2\cdot8\\ 3\cdot5+4\cdot7 & 3\cdot6+4\cdot8\end{pmatrix}
= \begin{pmatrix}19&22\\43&50\end{pmatrix}$.

**In Python:**

```python
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
# A has m columns, B has m rows: they must match
m = len(B)
# (AB)_ij = Σ_k A_ik B_kj
AB = [[sum(A[i][k] * B[k][j] for k in range(m))
       for j in range(len(B[0]))]
      for i in range(len(A))]
AB  # → [[19, 22], [43, 50]]
# the transpose: rows become columns
[list(column) for column in zip(*A)]  # → [[1, 3], [2, 4]]
```

```mermaid
flowchart LR
  R["row 1 of A<br/>(1, 2)"] --> D["dot product<br/>1·5 + 2·7 = 19"]
  C["column 1 of B<br/>(5, 7)"] --> D
  D --> O["answer, row 1, column 1 = 19"]
```

**Reading it:** a matrix multiply is nothing but this box, repeated for every
(row, column) pair. The shape rule falls out: the row of A and the column of
B must be the same length, or there is nothing to pair up. Almost all the
computation in a neural network is this one operation, which is why GPUs,
built to do thousands of multiply-adds at once, are the hardware of AI.

**In code:** `transpose` turns columns into rows; `matmul` transposes B once, then fills each cell with one `dot` of a row of A and a column of B.

## e and log: growth, and its undo button

**Everyday picture.** Money in an account that compounds continuously at
100% a year grows by a factor of **e ≈ 2.718** in one year. $e^x$ is that
growth run for $x$ years. The **natural logarithm** $\ln y$ (often just
$\log y$ in ML papers) answers the reverse question: how many years of that
growth turn 1 into $y$?

| You see | Say | Example |
|---|---|---|
| $e^x$ or $\exp(x)$ | "e to the x" | $e^0 = 1$, $e^1 = 2.718$, $e^2 = 7.39$, $e^{-1} = 0.37$ |
| $\ln y$ or $\log y$ | "log of y" | $\ln 7.39 = 2$, $\ln 1 = 0$, $\ln 0.01 = -4.61$ |

Two facts carry most of machine learning:

1. $e^x$ is **always positive** and **grows fast**, which is why softmax
   uses it.
2. $\log(a \times b) = \log a + \log b$. Logs turn multiplying into adding.
   The probability of a whole sentence is a product of thousands of small
   numbers, which would underflow to 0 on a computer. Its log is a sum of
   manageable negative numbers. This is why models work with
   "log-probabilities", and why the standard training loss is
   $-\log p$ (see `primer.ml.losses`).

![e to the x stays above zero and climbs steeply; ln x mirrors it across y = x and plunges near 0, so ln 0.01 = -4.61 while ln 0.9 = -0.11](figures/primer.notation.exp_log.svg)

**Reading it:** the blue curve $e^x$ is always above zero and climbs steeply;
every step of 1 to the right multiplies its height by 2.718. The red curve
$\ln x$ is the same curve mirrored across the dashed diagonal, because it
undoes $e^x$. It only exists for positive inputs, and it dives towards −∞ as
its input approaches 0. That dive is the "confidently wrong" penalty in
cross-entropy: $-\ln(0.01) = 4.61$, far more than $-\ln(0.9) = 0.11$.

## softmax and argmax

**softmax** turns a list of scores into shares that are positive and sum to
1. It is covered step by step, with its symbols decoded, in
`primer.ml.attention`:

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
$$

**With the numbers:** softmax(2.0, 1.0, 0.5) = (0.63, 0.23, 0.14).

**In Python:**

```python
import math
z = [2.0, 1.0, 0.5]
# e^(z_i) for each score
exps = [math.exp(z_i) for z_i in z]
# Σ_j e^(z_j)
total = sum(exps)
# each share of the total
[round(e / total, 2) for e in exps]  # → [0.63, 0.23, 0.14]
# argmax: the position of the largest
max(range(len(z)), key=lambda i: z[i])  # → 0
```

**argmax** is simpler: it answers "*which position* holds the largest
value?", not "what is the largest value?". argmax(0.1, 7.0, 3.0) = position
2 (index 1 in Python). "Greedy decoding" in a language model is picking the
argmax token every step.

**In code:** `softmax` subtracts the largest score before exponentiating, so the exponentials cannot overflow; `argmax` walks the list and remembers the position of the biggest value.

## Mean, variance, standard deviation: the middle and the spread

**Everyday picture.** Two classes both average 70% on a test. In one,
everyone scored 68 to 72; in the other, scores ran from 30 to 100. The
**mean** is the same; the **spread** is not.

$$
\mu = \frac{1}{n}\sum_{i=1}^{n} x_i
\qquad
\sigma^2 = \frac{1}{n}\sum_{i=1}^{n} (x_i - \mu)^2
\qquad
\sigma = \sqrt{\sigma^2}
$$

**Symbols**

| Symbol | Meaning |
|---|---|
| $\mu$ (mu) | the **mean**: the average |
| $\sigma^2$ (sigma squared) | the **variance**: the average squared distance from the mean; also written $\operatorname{Var}(x)$ |
| $\sigma$ (sigma) | the **standard deviation**: the typical distance from the mean, in the data's own units |
| $\frac{1}{n}\sum$ | "add them up and divide by how many": an average |

**In words:** "the mean is the average; the variance is the average squared
distance from the mean; the standard deviation is its square root."

**With the numbers:** for (2, 4, 4, 4, 5, 5, 7, 9): μ = 40 / 8 = 5; the
squared distances are (9, 1, 1, 1, 0, 0, 4, 16), which sum to 32, so
σ² = 32 / 8 = 4 and σ = 2.

**In Python:**

```python
import math
x = [2, 4, 4, 4, 5, 5, 7, 9]
n = len(x)
# μ = (1/n) Σ x_i
mu = sum(x) / n
mu  # → 5.0
# the squared distances from μ
[(x_i - mu) ** 2 for x_i in x]  # → [9.0, 1.0, 1.0, 1.0, 0.0, 0.0, 4.0, 16.0]
# σ² = their average
variance = sum((x_i - mu) ** 2 for x_i in x) / n
# σ is its square root
variance, math.sqrt(variance)  # → (4.0, 2.0)
```

Spread matters constantly in neural networks. If numbers flowing through a
network spread out layer after layer, training blows up; normalization layers
and careful initialization exist to hold σ near 1 (see `primer.ml.deep_nets`
and the √d_k in `primer.ml.attention`).

![Two histograms share the mean 0, but the sigma 1 samples pile up tall and narrow while the sigma 3 samples spread about three times as wide](figures/primer.notation.spread.svg)

**Reading it:** both histograms are centred on the same mean (the dashed
line), but the blue one is tall and narrow (σ = 1) while the red one is low
and wide (σ = 3). The standard deviation is roughly how far from the dashed
line a typical sample lands. About two thirds of samples fall within one σ
of the mean.

**In code:** `mean`, `variance` and `std` are the three formulas above, each a short loop built on `summation`.

## Derivatives and gradients: which way is downhill?

**Everyday picture.** You're on a hillside in thick fog and want to reach
the valley. You can't see it, but you can feel the slope under your feet.
Take a small step in the steepest downhill direction, feel again, and
repeat. That is how every neural network is trained (**gradient descent**).

The **derivative** of a function at a point is its slope there: how much the
output changes per tiny nudge of the input.

$$
f'(x) = \frac{df}{dx} \approx \frac{f(x + h) - f(x - h)}{2h}
$$

**Symbols**

| Symbol | Meaning |
|---|---|
| $f(x)$ | a function: put $x$ in, get a number out |
| $f'(x)$ or $\frac{df}{dx}$ | the derivative: the slope of $f$ at $x$ ("dee f dee x") |
| $h$ | a tiny nudge, like 0.00001 |
| $\approx$ | "approximately equal": exact as $h$ shrinks to 0 |

**In words:** "nudge the input a hair up and a hair down, and see how much
the output changes per unit of nudge."

**With the numbers:** for $f(x) = x^2$ at $x = 3$: (3.00001² − 2.99999²) /
0.00002 = 6. The slope of $x^2$ at 3 is 6 (the rule is $2x$).

**In Python:**

```python
def f(x):
    return x ** 2
h = 0.00001
# rise over run across a tiny step
round((f(3 + h) - f(3 - h)) / (2 * h), 6)  # → 6.0
```

With many inputs, nudge each one separately. The slope in each direction is
a **partial derivative**, written $\frac{\partial f}{\partial x_i}$ (the curly
∂ just means "only this input moves, the others stay fixed"). Collect them
into a vector and you have the **gradient**, written $\nabla f$ ("nabla f" or
"grad f"). It points in the steepest *uphill* direction, so training steps
the opposite way:

$$
\theta_{\text{new}} = \theta - \eta \, \nabla L(\theta)
$$

**Symbols**

| Symbol | Meaning |
|---|---|
| $\theta$ (theta) | all the model's adjustable numbers (its **parameters** or **weights**) |
| $L(\theta)$ | the **loss**: one number measuring how wrong the model is |
| $\nabla L(\theta)$ | the gradient of the loss: for each weight, how much the loss rises if that weight is nudged up |
| $\eta$ (eta) | the **learning rate**: how big a step to take, e.g. 0.001 |

**In words:** "move every weight a small step in the direction that lowers
the loss the fastest."

**With the numbers:** for the bowl $L = x^2 + y^2$ at (1, 2), the gradient is
(2, 4), pointing uphill away from the bottom at (0, 0). With η = 0.1, the
step goes to (1 − 0.2, 2 − 0.4) = (0.8, 1.6), closer to the bottom.

**In Python:**

```python
# (x, y)
theta = [1.0, 2.0]
# ∇L for L = x² + y²: each slope is 2 times the value
gradient = [2 * theta[0], 2 * theta[1]]
gradient  # → [2.0, 4.0]
eta = 0.1
# θ_new = θ − η ∇L
[t - eta * g for t, g in zip(theta, gradient)]  # → [0.8, 1.6]
```

![Arrows on circular contours point straight at the centre, longest on the steep rim; descent from (1, 2) takes big steps first, then ever smaller ones](figures/primer.notation.gradient_descent.svg)

**Reading it:** the rings are contour lines, as on a hiking map: every point
on a ring has the same loss, and the bottom of the bowl is the centre. The
arrows show the negative gradient at several spots. They always cross the
rings at right angles, pointing straight downhill, and they are longer where
the slope is steeper. The dotted path is gradient descent from (1, 2): big
steps on the steep outer slope, shrinking steps as the ground flattens near
the bottom.

**The chain rule** says how slopes combine when functions are chained: if
$y = f(g(x))$, then $\frac{dy}{dx} = f'(g(x)) \cdot g'(x)$. Multiply the
slopes along the chain. **Backpropagation** is the chain rule applied
backwards through every layer of a network, reusing work as it goes (see
`primer.ml.neural_net`).

**In code:** `derivative` measures a slope by nudging the input a hair up and a hair down; `gradient` does that for each input in turn and collects the slopes into one list.

## Probability notation

| You see | Say | Meaning |
|---|---|---|
| $P(A)$ or $p(A)$ | "probability of A" | a number from 0 (never) to 1 (certain) |
| $P(A \mid B)$ | "probability of A given B" | the chance of A once you know B happened |
| $P(w_t \mid w_{<t})$ | "probability of word t given the words before it" | what a language model computes, one word at a time |
| $\mathbb{E}[x]$ | "expected value of x" | the average you'd get over many tries |
| $x \sim \mathcal{N}(0, 1)$ | "x is drawn from a normal distribution with mean 0 and variance 1" | `rng.standard_normal()` |

## Big-O: how cost grows

$O(n^2)$, "order n squared", describes how cost **grows** as the input
grows, ignoring constant factors. If $n$ doubles, an $O(n)$ cost doubles and
an $O(n^2)$ cost quadruples. Attention is $O(n^2)$ in sequence length, which
is why long context is expensive (`primer.ml.attention`).

## Greek letters you'll meet

| Letter | Name | Usually means |
|---|---|---|
| α | alpha | a mixing weight, or a learning rate |
| β | beta | momentum coefficients in optimizers (β₁, β₂); a strength knob in DPO |
| γ, β | gamma, beta | the learned scale and shift in normalization layers |
| δ | delta | a small change, or an error signal in backprop |
| ε | epsilon | a tiny number added to avoid dividing by zero |
| η | eta | the learning rate |
| θ | theta | all the model's parameters |
| λ | lambda | the strength of a penalty (weight decay) |
| μ | mu | a mean |
| σ | sigma | a standard deviation, or the sigmoid function $\sigma(x) = 1/(1+e^{-x})$ |
| τ | tau | a temperature (sharpness of softmax) |
| ∇ | nabla | the gradient |
| ∂ | "partial" | a partial derivative |

## In 20 seconds

- Σ is a loop that adds; Π is a loop that multiplies.
- A dot product multiplies matching entries and adds them up; it measures
  how much two vectors agree.
- A matrix multiply is a grid of dot products, and it is most of the work a
  neural network does.
- log undoes e, and it turns products into sums, which is why models work
  with log-probabilities.
- The gradient points uphill; training steps the other way.

## Self-test questions

**Read $\sum_{i=1}^{3} i^2$ aloud and evaluate it.**
"The sum, for i from 1 to 3, of i squared": 1 + 4 + 9 = 14.

**What is (2, −1, 3) · (1, 4, 0), and what does its sign tell you?**
2 − 4 + 0 = −2. It's negative, so the vectors point somewhat apart.

**A is 4 × 3 and B is 3 × 5. What shape is AB? What about BA?**
AB is 4 × 5. BA is not defined: B's 5 columns don't match A's 4 rows.

**Why do language models add log-probabilities instead of multiplying
probabilities?**
A product of thousands of numbers below 1 underflows to 0 in floating point.
log(a·b) = log a + log b, so the product becomes a sum of moderate
negative numbers.

**What does ∇L tell you, and which way does training move?**
For every weight, how fast the loss rises as that weight increases. Training
moves each weight the opposite way, scaled by the learning rate η.

## Further reading

- 3Blue1Brown, *Essence of Linear Algebra* (vectors, matrices, dot products, visually): https://www.3blue1brown.com/topics/linear-algebra
- 3Blue1Brown, *Essence of Calculus* (derivatives and the chain rule): https://www.3blue1brown.com/topics/calculus
- Khan Academy, *Linear algebra*: https://www.khanacademy.org/math/linear-algebra
- Deisenroth, Faisal and Ong, *Mathematics for Machine Learning* (free book): https://mml-book.github.io/
- NumPy, absolute basics for beginners: https://numpy.org/doc/stable/user/absolute_beginners.html
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

from primer._show import banner, say, table, takeaway

Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]


# ---------------------------------------------------------------------------
# Σ and Π
# ---------------------------------------------------------------------------


def summation(xs: Vector) -> float:
    """Σ xᵢ: a loop with a running total that starts at 0."""
    total = 0
    for x in xs:
        total += x
    return total


def product(xs: Vector) -> float:
    """Π xᵢ: a loop with a running product that starts at 1 (multiplying by 1 changes nothing)."""
    result = 1
    for x in xs:
        result *= x
    return result


# ---------------------------------------------------------------------------
# Vectors
# ---------------------------------------------------------------------------


def dot(a: Vector, b: Vector) -> float:
    """a · b = Σ aᵢ bᵢ. Lists must be the same length, or there's nothing to pair up."""
    if len(a) != len(b):
        raise ValueError(f"dot product needs equal lengths, got {len(a)} and {len(b)}")
    return summation([ai * bi for ai, bi in zip(a, b)])


def norm(x: Vector) -> float:
    """‖x‖ = √(x · x): Pythagoras in any number of dimensions."""
    return math.sqrt(dot(x, x))


# ---------------------------------------------------------------------------
# Matrices
# ---------------------------------------------------------------------------


def transpose(A: Matrix) -> list[list[float]]:
    """Aᵀ: row i of the answer is column i of A."""
    return [[row[j] for row in A] for j in range(len(A[0]))]


def matmul(A: Matrix, B: Matrix) -> list[list[float]]:
    """(AB)ᵢⱼ = row i of A · column j of B. A is n×m, B must be m×p, the answer is n×p."""
    if len(A[0]) != len(B):
        raise ValueError(f"inner sizes differ: A has {len(A[0])} columns, B has {len(B)} rows")
    columns_of_B = transpose(B)
    return [[dot(row, col) for col in columns_of_B] for row in A]


# ---------------------------------------------------------------------------
# softmax and argmax
# ---------------------------------------------------------------------------


def softmax(z: Vector) -> list[float]:
    """e^zᵢ / Σⱼ e^zⱼ. Subtracting max(z) first changes nothing mathematically
    (it cancels top and bottom) but keeps e^z from overflowing."""
    m = max(z)
    exps = [math.exp(zi - m) for zi in z]
    total = summation(exps)
    return [e / total for e in exps]


def argmax(xs: Vector) -> int:
    """The *position* of the largest value (0-based, as Python counts)."""
    best = 0
    for i in range(1, len(xs)):
        if xs[i] > xs[best]:
            best = i
    return best


# ---------------------------------------------------------------------------
# Mean and spread
# ---------------------------------------------------------------------------


def mean(xs: Vector) -> float:
    """μ = (1/n) Σ xᵢ."""
    return summation(xs) / len(xs)


def variance(xs: Vector) -> float:
    """σ² = (1/n) Σ (xᵢ − μ)²: the average squared distance from the mean."""
    mu = mean(xs)
    return mean([(x - mu) ** 2 for x in xs])


def std(xs: Vector) -> float:
    """σ = √σ²: the typical distance from the mean, in the data's own units."""
    return math.sqrt(variance(xs))


# ---------------------------------------------------------------------------
# Derivatives and gradients
# ---------------------------------------------------------------------------


def derivative(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """f'(x) ≈ (f(x+h) − f(x−h)) / 2h. Nudging both ways (a "central difference")
    is far more accurate than nudging one way."""
    return (f(x + h) - f(x - h)) / (2 * h)


def gradient(f: Callable[[list[float]], float], x: Vector, h: float = 1e-5) -> list[float]:
    """∇f: one partial derivative per input, each found by nudging only that input."""
    grads = []
    for i in range(len(x)):
        up, down = list(x), list(x)
        up[i] += h
        down[i] -= h
        grads.append((f(up) - f(down)) / (2 * h))
    return grads


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figures() -> dict:
    """Plot this lesson's data (matplotlib is imported here only)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figs = {}

    # e^x and ln x, mirrored across y = x
    fig, ax = plt.subplots(figsize=(5.5, 5))
    xs = np.linspace(-3, 2.2, 200)
    ax.plot(xs, np.exp(xs), label="$e^x$ (always > 0, grows fast)")
    pos = np.linspace(0.02, 9, 300)
    ax.plot(pos, np.log(pos), label=r"$\ln x$ (undoes $e^x$)")
    ax.plot([-3, 9], [-3, 9], ls="--", color="#9ca3af", lw=1, label="mirror line y = x")
    for p in (0.9, 0.01):
        ax.plot(p, np.log(p), "o", color="#dc2626")
        ax.annotate(f"ln {p} = {np.log(p):.2f}", (p, np.log(p)), xytext=(10, -4), textcoords="offset points")
    ax.set_xlim(-3, 9)
    ax.set_ylim(-5, 9)
    ax.axhline(0, color="#4b5563", lw=0.8)
    ax.axvline(0, color="#4b5563", lw=0.8)
    ax.set_title("e and its undo button, ln")
    ax.legend(frameon=False, loc="upper left")
    figs["exp_log"] = fig

    # same mean, different spread
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bins = np.linspace(-10, 10, 61)
    ax.hist(rng.normal(0, 1, 4000), bins=bins, alpha=0.7, label="σ = 1")
    ax.hist(rng.normal(0, 3, 4000), bins=bins, alpha=0.6, label="σ = 3")
    ax.axvline(0, ls="--", color="#4b5563")
    ax.set_xlabel("value")
    ax.set_ylabel("how many samples")
    ax.set_title("Same mean (μ = 0), different spread")
    ax.legend(frameon=False)
    figs["spread"] = fig

    # gradient descent on a bowl
    fig, ax = plt.subplots(figsize=(5.5, 5))
    g = np.linspace(-2.5, 2.5, 200)
    X, Y = np.meshgrid(g, g)
    ax.contour(X, Y, X**2 + Y**2, levels=10, colors="#9ca3af", linewidths=0.8)
    q = np.linspace(-2, 2, 7)
    QX, QY = np.meshgrid(q, q)
    ax.quiver(QX, QY, -2 * QX, -2 * QY, color="#2563eb", alpha=0.7, angles="xy", scale_units="xy", scale=8)
    path = [np.array([1.0, 2.0])]
    for _ in range(12):
        path.append(path[-1] - 0.15 * np.array(gradient(lambda v: v[0] ** 2 + v[1] ** 2, path[-1].tolist())))
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], "o:", color="#dc2626", ms=4, label="gradient descent from (1, 2)")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Loss $x^2+y^2$: arrows point downhill")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(False)
    figs["gradient_descent"] = fig

    return figs


# ---------------------------------------------------------------------------
# Walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. Σ is a loop that adds, Π is a loop that multiplies")
    say("Σ over 1..4 and Π over 1..4, then the chance that 10 steps at 95% all succeed:")
    table(["expression", "value"], [("Σ i, i=1..4", summation([1, 2, 3, 4])), ("Π i, i=1..4", product([1, 2, 3, 4])), ("0.95^10", product([0.95] * 10))], floatfmt=".3f")
    takeaway("A 95%-reliable step repeated 10 times succeeds only 60% of the time.")

    banner("2. Dot product and length")
    table(
        ["expression", "working", "value"],
        [("(1,2)·(3,0.5)", "1·3 + 2·0.5", dot([1, 2], [3, 0.5])), ("‖(3,4)‖", "√(9+16)", norm([3, 4])), ("‖(1,2,2)‖", "√(1+4+4)", norm([1, 2, 2]))],
        floatfmt=".1f",
    )

    banner("3. Matrix multiply is a grid of dot products")
    A, B = [[1, 2], [3, 4]], [[5, 6], [7, 8]]
    say(f"A = {A}, B = {B}. Cell (1,1) = row 1 of A · column 1 of B = 1·5 + 2·7 = 19.")
    say(f"AB = {matmul(A, B)}.  Aᵀ = {transpose(A)}.")

    banner("4. softmax and argmax")
    s = softmax([2.0, 1.0, 0.5])
    say(f"softmax(2.0, 1.0, 0.5) = ({s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f}), which sums to {sum(s):.2f}.")
    say(f"argmax(0.1, 7.0, 3.0) = {argmax([0.1, 7.0, 3.0])}: the position of the biggest value, not the value.")

    banner("5. Middle and spread")
    data = [2, 4, 4, 4, 5, 5, 7, 9]
    table(["data", "mean μ", "variance σ²", "std σ"], [(data, mean(data), variance(data), std(data))], floatfmt=".1f")

    banner("6. Slopes: derivative and gradient")
    say(f"Slope of x² at x=3: {derivative(lambda x: x * x, 3.0):.4f} (the rule 2x says 6).")
    gr = gradient(lambda v: v[0] ** 2 + v[1] ** 2, [1.0, 2.0])
    say(f"Gradient of x²+y² at (1,2): ({gr[0]:.2f}, {gr[1]:.2f}). It points uphill; training steps the other way.")
    takeaway("Every symbol in an ML paper is shorthand for a short loop. Read it aloud, then write the loop.")


if __name__ == "__main__":
    demo()
