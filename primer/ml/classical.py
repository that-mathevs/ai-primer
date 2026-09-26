r"""
# Trees and boosting: the other workhorse

Run: `python -m primer.ml.classical`

## Why a lesson on trees, in a primer on modern AI

Most of this primer is about neural networks, because language models are
neural networks. But open the models that decide whether a loan is approved,
whether a card payment looks like fraud, which ad to show, or how many
umbrellas a shop should stock, and very often you find no neural network at
all. You find hundreds of small **decision trees**, added together.

Those problems share a shape: a **table**. Each row is a customer, a payment
or a day; each column is a fact measured in its own units (age in years,
income in dollars, number of late payments, a region code). On tables of
modest size, ensembles of trees have stayed the model to beat, and they
train in seconds on a laptop. This lesson builds them from scratch, so you
can see why they work and, just as usefully, recognise when a neural network
is the wrong tool.

The plan: one tree (how it chooses its questions), why a single tree
overfits, how a **random forest** fixes that by averaging many trees, how
**gradient boosting** fixes it differently by adding trees one after another,
and finally a head-to-head against a small neural network from
`primer.ml.neural_net` on a loan table.

## A decision tree is a game of twenty questions

**Everyday picture.** A nurse at an emergency desk runs a flowchart: "Is the
patient breathing normally? No: resuscitation room. Yes: is there chest
pain? Yes: see a doctor now. No: take a seat." Each question looks at *one*
fact and sends the patient left or right, and the last box gives the answer.
A decision tree is exactly that flowchart, except the questions are chosen
by the computer from past examples.

**Tiny example.** Eight emails, each described by two facts: how many links
it contains, and whether it comes from a sender you have written to before.

| email | links | known sender | spam? |
|---|---|---|---|
| 1 | 0 | yes | no |
| 2 | 1 | yes | no |
| 3 | 4 | yes | no |
| 4 | 0 | no | no |
| 5 | 3 | no | yes |
| 6 | 5 | no | yes |
| 7 | 6 | no | yes |
| 8 | 1 | no | yes |

The tree this lesson grows from those eight rows:

```mermaid
flowchart TD
  Q1{"known sender?"} -->|yes| L1["not spam<br/>(emails 1, 2, 3)"]
  Q1 -->|no| Q2{"more than 0 links?"}
  Q2 -->|no| L2["not spam<br/>(email 4)"]
  Q2 -->|yes| L3["spam<br/>(emails 5, 6, 7, 8)"]
```

**Reading it:** start at the top diamond with a new email and answer each
question until you land in a box. Diamonds are questions (the tree's
**internal nodes**); boxes are answers (its **leaves**), and each leaf lists
the training emails that ended up there. Two questions sort all eight
correctly. Notice what the tree did *not* ask: "more than 2 links?" looks
reasonable, but it was worse. The next two sections show how the tree
decides that.

**In code:** `spam_emails` returns the table above; `DecisionTree` grows the tree, and `DecisionTree.rules` prints it as nested if/else lines.

## Measuring a mess: Gini impurity

**Everyday picture.** You are sorting laundry into piles. A pile of only
socks is *pure*: pull out any two items and they match. A pile that is half
socks, half shirts is as mixed-up as two kinds of item can be. A good
sorting question is one that leaves piles as pure as possible.

**Tiny example.** Before any question, the eight emails are 4 spam and 4
not: a 50/50 pile. Pull out two emails at random (putting the first one
back) and they disagree half the time. The pile of unknown senders is 1
"not spam" and 4 spam: purer, so two random picks disagree less often.

**Gini impurity** is exactly that chance of disagreement:

$$
G = 1 - \sum_{k=1}^{K} p_k^{2}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $G$ | the Gini impurity of one pile of examples: 0 when pure, larger when mixed | 0.5 for the eight emails |
| $K$ | how many different labels there are | 2 (spam, not spam) |
| $k$ | a counter that walks over the labels | 1 = not spam, 2 = spam |
| $p_k$ | the share of the pile carrying label $k$ | $p_1 = 4/8$, $p_2 = 4/8$ |
| $p_k^{2}$ | the chance two random picks *both* have label $k$ | $0.5^2 = 0.25$ |
| $\sum_{k=1}^{K}$ | "add up over every label": the chance two picks match | $0.25 + 0.25 = 0.5$ |
| $1 - \ldots$ | turns "chance they match" into "chance they disagree" | $1 - 0.5$ |

**In words:** "the impurity of a pile is one minus the sum of the squared
shares of each label: the chance that two examples drawn at random carry
different labels."

**With the numbers:** all eight emails: $1 - (0.5^2 + 0.5^2) = 1 - 0.5 =
0.5$. The unknown-sender pile, 1 of 5 not spam and 4 of 5 spam:
$1 - (0.2^2 + 0.8^2) = 1 - (0.04 + 0.64) = 0.32$. A pure pile: $1 - 1^2 = 0$.

**In Python:**

```python
# four spam (1) and four not spam (0)
labels = [1, 1, 1, 1, 0, 0, 0, 0]
# p_k: the share of each label
p = [labels.count(k) / len(labels) for k in (0, 1)]
p  # → [0.5, 0.5]
# G = 1 - Σ p_k²
1 - sum(p_k ** 2 for p_k in p)  # → 0.5
# the unknown-sender pile: 1 not spam, 4 spam
p = [1 / 5, 4 / 5]
round(1 - sum(p_k ** 2 for p_k in p), 2)  # → 0.32
```

**Why it matters:** the tree needs a single number to compare questions
with, and it must be cheap, because a real tree scores millions of candidate
questions. Gini needs only counts, one multiply per label, and no logarithm.

**In code:** `gini` computes $G$ for any list of labels.

### Entropy: the same idea, counted in yes/no questions

**Everyday picture.** Entropy asks: if I had to guess an email's label by
asking yes/no questions, how many would I need on average? A 50/50 pile
needs one full question. A pure pile needs none, because you already know.

**Tiny example.** For the eight emails, one question ("is it spam?")
settles it: entropy 1 bit. For the unknown-sender pile, 4 of 5 are spam, so
you are usually right before asking and the average cost is about 0.72 of a
question.

$$
H = -\sum_{k=1}^{K} p_k \log_2 p_k
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $H$ | the entropy of one pile, in **bits** (yes/no questions) | 1 for the eight emails |
| $p_k$ | the share of the pile carrying label $k$, as above | 0.2 and 0.8 |
| $\log_2 p_k$ | the **logarithm** base 2: the power you raise 2 to in order to get $p_k$. It is negative for shares below 1 | $\log_2 0.5 = -1$, $\log_2 0.2 = -2.32$ |
| $-\sum$ | add up over the labels, then flip the sign so the total is positive | |

**In words:** "entropy is the average, over the labels, of how surprising
each label is, weighted by how often it appears." A label with share 0.5
costs $-\log_2 0.5 = 1$ bit of surprise.

**With the numbers:** eight emails: $-(0.5 \cdot (-1) + 0.5 \cdot (-1)) = 1$
bit. Unknown-sender pile: $-(0.2 \cdot \log_2 0.2 + 0.8 \cdot \log_2 0.8) =
-(0.2 \cdot (-2.32) + 0.8 \cdot (-0.32)) = 0.722$ bits.

**In Python:**

```python
import math
# H = -Σ p_k log2 p_k
p = [0.5, 0.5]
-sum(p_k * math.log2(p_k) for p_k in p)  # → 1.0
p = [1 / 5, 4 / 5]
round(-sum(p_k * math.log2(p_k) for p_k in p), 3)  # → 0.722
```

Gini and entropy almost always pick the same question; Gini is a little
cheaper. The drop in entropy a question produces is called **information
gain**, the name the early ID3 algorithm used. See `primer.notation` for
logarithms from scratch and `primer.ml.losses` for entropy's close cousin,
cross-entropy.

**In code:** `entropy` computes $H$ in bits; `DecisionTree` takes `criterion="entropy"` to split by it instead of Gini.

## Choosing the best question

**Everyday picture.** A question splits one pile into two. Judge it by how
messy the two new piles are, but weigh each by its size: a tiny pure pile of
one email is worth less than a big pure pile of five.

**Tiny example.** "Known sender?" makes a pile of 3 (all not spam, Gini 0)
and a pile of 5 (1 not spam, 4 spam, Gini 0.32). "More than 2 links?" makes
a pile of 4 (3 not spam, 1 spam) and another pile of 4 (1 not spam, 3 spam),
Gini 0.375 each.

$$
G_{\text{split}} = \frac{n_L}{n}\, G(L) + \frac{n_R}{n}\, G(R)
$$

**Symbols**

| Symbol | Meaning here | For "known sender?" |
|---|---|---|
| $L$, $R$ | the left pile (answer yes) and the right pile (answer no) | known / unknown senders |
| $n_L$, $n_R$ | how many examples land in each pile | 3 and 5 |
| $n$ | how many examples reached this node, $n_L + n_R$ | 8 |
| $G(L)$, $G(R)$ | the Gini impurity of each pile (the formula above) | 0 and 0.32 |
| $\frac{n_L}{n}$ | the left pile's share of the examples: its weight | 3/8 |
| $G_{\text{split}}$ | the impurity left after asking: lower is better | 0.2 |

**In words:** "the mess left by a question is the average of the two piles'
impurities, each weighted by the share of examples in it."

**With the numbers:** known sender: $\frac{3}{8} \cdot 0 + \frac{5}{8} \cdot
0.32 = 0.2$. More than 2 links: $\frac{4}{8} \cdot 0.375 + \frac{4}{8}
\cdot 0.375 = 0.375$. The tree asks "known sender?" because 0.2 < 0.375.
The **impurity decrease** (the gain) is $0.5 - 0.2 = 0.3$.

**In Python:**

```python
# known senders: 3 emails, all not spam
n_left, g_left = 3, 0.0
# unknown senders: 1 not spam, 4 spam
n_right, g_right = 5, 0.32
n = n_left + n_right
# G_split = (n_L/n)·G(L) + (n_R/n)·G(R)
round(n_left / n * g_left + n_right / n * g_right, 3)  # → 0.2
# the impurity the question removed
round(0.5 - 0.2, 3)  # → 0.3
```

Where do candidate questions come from? For a number such as "links", only
thresholds *between* neighbouring values in the data can change which
emails go where. The links column holds 0, 1, 3, 4, 5 and 6, so there are
five thresholds worth trying (halfway points 0.5, 2, 3.5, 4.5, 5.5), plus
one for the yes/no column. The tree scores all six and keeps the lowest.

![Weighted Gini after each of six candidate first questions: known_sender scores 0.200, the best; the five links thresholds score between 0.333 and 0.467; all are below the 0.5 of the unsplit pile](figures/primer.ml.classical.spam_splits.svg)

**Reading it:** each bar is one question the tree could ask first, and its
length is the weighted Gini impurity left afterwards. The dashed line is the
impurity before asking anything (0.5). Every question helps a little, since
every bar ends left of the line, but "known_sender > 0.5" (blue) leaves the
least mess by a clear margin, so it becomes the root of the tree. Two links
thresholds tie at 0.333: the greedy search would take the first if known
sender did not exist.

**In code:** `split_impurity` is $G_{\text{split}}$; `candidate_splits` scores every question in the figure; `best_split` does the same search fast, by sorting each feature once and keeping running label counts, and returns a `Split`.

**Why it matters:** this greedy search is the whole learning algorithm. There
is no gradient and no learning rate: the tree tries every feature and every
threshold and takes the best. That is why trees handle columns in any units,
as the last section shows.

## Growing the tree: ask, split, repeat

**Everyday picture.** Sorting a big box of photos: first by year, then each
year's pile by who is in them, then each of those by place, until every pile
is tidy or too small to bother with.

**Tiny example.** After "known sender?", the left pile (emails 1, 2, 3) is
already pure: it becomes a leaf. The right pile (4, 5, 6, 7, 8) is not, so
the tree searches again *inside that pile only*. There, "more than 0 links?"
separates email 4 from the four spam emails perfectly. Both new piles are
pure, so growth stops at depth 2.

```mermaid
flowchart TD
  S["node: the examples that reached here"] --> P{"pure, too small,<br/>or at the depth limit?"}
  P -->|yes| LEAF["make a leaf:<br/>predict the majority label<br/>(or the average number)"]
  P -->|no| SEARCH["try every feature and every threshold;<br/>keep the lowest weighted impurity"]
  SEARCH --> SPLIT["send examples left (feature ≤ threshold)<br/>or right (feature > threshold)"]
  SPLIT --> LEFT["grow the left child"] & RIGHT["grow the right child"]
  LEFT -.-> S
  RIGHT -.-> S
```

**Reading it:** the same recipe runs at every node, which is why the code is
a function that calls itself (recursion). The dotted arrows are those
calls: each child is a fresh node holding only its share of the examples.
The diamond is where growth stops; everything else is the search from the
previous section. To predict, a new example simply follows the questions
from the root down to a leaf.

Each question looks at a single feature and compares it with a threshold,
so every question cuts the input space with a straight line parallel to an
axis. A tree therefore carves the plane into rectangles:

![A tree on two-moon data at depth 1, 3 and unlimited: one horizontal cut, then a few rectangles, then a patchwork of 51 boxes with thin strips around individual noisy points](figures/primer.ml.classical.regions.svg)

**Reading it:** the dots are 300 training points of two interleaved
half-moons with noise (`primer.ml.neural_net.make_moons`); the shaded
background is what the tree would predict at each spot. At depth 1 there is
one question, so one straight horizontal cut. At depth 3 there are 8 leaves, and the
boundary is a staircase of vertical and horizontal edges that roughly
follows the moons. With no limit, the tree keeps cutting until every leaf is
pure: 51 boxes, including thin strips that reach into the other class's
territory to capture a few noisy points.
Every edge is axis-aligned: trees cannot draw a diagonal, only approximate
one with steps.

A regression tree, predicting a number instead of a label, is the same
recipe with two swaps: the impurity is the **variance** of the numbers in a
pile (how spread out they are), and a leaf predicts their average. Gradient
boosting, below, is built from those.

**In code:** `DecisionTree` grows by the recipe in the diagram (a depth limit and a minimum leaf size are its stopping rules, and the "squared" criterion makes it a regression tree), and `DecisionTree.predict` routes each row down to its leaf.

## Overfitting: a question for every example

**Everyday picture.** Ask a friend to write rules for which of their emails
are spam, and forbid them from stopping until every past email is sorted
correctly. They will end up with rules like "an email from Pat at 9:14 on a
Tuesday with 3 links is spam": perfect on the past, useless for the future.

**Tiny example.** On the noisy moons, a tree with no depth limit grows 51
leaves and gets 100% of its 300 training points right. On 200 fresh points
it gets 82%. A tree stopped at depth 2 has 4 leaves, gets 88% on training
and 89.5% on the fresh points: it learned the shape rather than the noise.

![Training and validation accuracy against the depth limit: training rises from 78% to 100%; validation peaks at 89.5% for depth 2 and 3 and falls to 82% with no limit](figures/primer.ml.classical.depth_sweep.svg)

**Reading it:** the horizontal axis is the depth limit, from one question to
no limit at all. The blue training curve only ever rises, because every
extra level can only fit the training points better. The orange validation
curve peaks early, at depth 2 to 3, then slides down as the tree starts
building boxes around individual noisy points. The widening gap between
the two curves is overfitting, exactly as in `primer.ml.regularization`.

A single tree is the textbook **high-variance** model: change a few training
points and the top question can flip, and everything below it changes with
it. There are two ways out. Keep trees small (limit the depth, require a
minimum number of examples per leaf, or grow fully then **prune** back the
branches that do not help on validation data). Or keep trees big and combine
many of them, which is what the rest of this lesson does.

**In code:** `depth_sweep` grows one tree per depth limit on `moons_split` data and measures both accuracies.

**Why it matters:** depth is the tree's capacity knob. Every tree library
exposes it (along with a minimum leaf size), and choosing it on validation
data is the first thing to tune.

## Random forests: many noisy trees, one steady vote

**Everyday picture.** At a county fair, hundreds of people guess the weight
of an ox. Individual guesses are wildly off in both directions, but the
average of all of them lands remarkably close, because the errors point in
different directions and cancel. A **random forest** does this with trees:
grow many deep trees that each make *different* mistakes, then average
them. The trick is making the mistakes different, since averaging copies of
the same tree changes nothing.

Two sources of difference, both from Leo Breiman:

1. **Bagging** (bootstrap aggregating): each tree trains on its own
   **bootstrap sample**, a resample of the training rows drawn *with
   replacement*.
2. **Random feature subsets**: at every node, a tree may only consider a
   random handful of the features (commonly the square root of their
   number), so different trees are forced to find different questions.

```mermaid
flowchart LR
  D["training table<br/>(n rows)"] --> B1["bootstrap sample 1<br/>n rows drawn with replacement"] & B2["bootstrap sample 2"] & B3["..."] & BB["bootstrap sample B"]
  B1 --> T1["deep tree 1<br/>random features at each node"]
  B2 --> T2["deep tree 2"]
  B3 --> T3["..."]
  BB --> TB["deep tree B"]
  T1 & T2 & T3 & TB --> AVG["average their class shares<br/>(or their numbers)"]
  AVG --> OUT["prediction"]
```

**Reading it:** the table fans out into B different resamples, each resample
grows its own full-depth tree, and the trees' answers are averaged at the
end. Nothing flows between trees, so they can be grown in parallel on
separate machines. Diversity enters twice: once in which rows each tree
sees (the bootstrap boxes) and once in which features each node may use
(inside the tree boxes).

### Bagging: drawing names from a hat, and putting them back

**Tiny example.** Put the eight email numbers in a hat. Draw one, write it
down, *put it back*, and repeat eight times. One draw might give 3, 1, 3, 7,
5, 1, 8, 2: emails 1 and 3 twice, and emails 4 and 6 not at all. Each tree
sees a slightly different dataset, with some rows counted double and about
a third missing.

How big is "about a third"? A particular email is missed on one draw with
chance $1 - \frac{1}{n}$, and missed on all $n$ draws with:

$$
P(\text{left out}) = \left(1 - \frac{1}{n}\right)^{n} \;\longrightarrow\; \frac{1}{e} \approx 0.368
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $n$ | the number of rows, and also the number of draws | 8 |
| $\frac{1}{n}$ | the chance one draw picks this particular row | 1/8 |
| $1 - \frac{1}{n}$ | the chance one draw misses it | 7/8 |
| $(\ldots)^{n}$ | missing it on every one of the $n$ independent draws: multiply the chances | $(7/8)^8$ |
| $\longrightarrow$ | "gets closer and closer to, as $n$ grows" | |
| $e$ | Euler's number, ≈ 2.718 (see `primer.notation`) | $1/e = 0.368$ |

**In words:** "the chance a row never makes it into a bootstrap sample is
the chance of missing it once, multiplied by itself once per draw; for big
datasets that settles at about 37%."

**With the numbers:** $(7/8)^8 = 0.3436$ for the eight emails; for 100,000
rows, $(1 - 1/100000)^{100000} = 0.3679$, already equal to $1/e$ to four
places.

**In Python:**

```python
import math
n = 8
# (1 - 1/n)^n: missed on every one of the n draws
round((1 - 1 / n) ** n, 4)  # → 0.3436
round((1 - 1 / 100_000) ** 100_000, 4)  # → 0.3679
round(1 / math.e, 4)  # → 0.3679
```

The rows a tree never saw are its **out-of-bag** rows: a free validation set
for that tree, which random forests use to estimate their own accuracy
without holding data back.

**In code:** `bootstrap_sample` draws the n row indices; `out_of_bag_fraction` is the formula; `RandomForest` gives every tree its own sample and its own feature subsets.

### Why averaging helps, and what limits it

**Tiny example.** Three trees give a point spam probabilities of 0.9, 0.2
and 0.7. The forest says 0.6. If each tree's error is a coin flip that
cancels out on average, three trees already wobble less than one; a hundred
wobble far less. But if every tree makes the *same* mistake, averaging keeps
it.

$$
\operatorname{Var}\!\left(\frac{1}{B}\sum_{b=1}^{B} T_b\right) = \rho\,\sigma^{2} + \frac{1-\rho}{B}\,\sigma^{2}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $B$ | the number of trees | 10 |
| $T_b$ | tree number $b$'s prediction for one input | 0.9, 0.2, 0.7, … |
| $\frac{1}{B}\sum_{b=1}^{B} T_b$ | the forest's prediction: the average of all trees | 0.6 |
| $\operatorname{Var}$ | **variance**: how much a prediction would swing if the training data were redrawn | |
| $\sigma^{2}$ | the variance of one tree on its own | 1 (a unit, for comparison) |
| $\rho$ | the **correlation** between two trees' errors: 0 if they err independently, 1 if they always err together (Greek letter rho) | 0.3 |

**In words:** "the forest's wobble has two parts: a part shared by all trees,
which no amount of averaging removes, and a private part, which shrinks as
you add trees."

**With the numbers:** with $\rho = 0.3$ and $\sigma^2 = 1$: ten trees give
$0.3 + 0.7/10 = 0.37$; a thousand trees give $0.3 + 0.7/1000 = 0.3007$.
Going from 10 to 1,000 trees barely helps; lowering $\rho$ would. With fully
independent trees ($\rho = 0$), ten trees would cut the variance to 0.1.

**In Python:**

```python
sigma2, rho = 1.0, 0.3
# ρσ² + (1 - ρ)σ²/B
round(rho * sigma2 + (1 - rho) * sigma2 / 10, 4)  # → 0.37
round(rho * sigma2 + (1 - rho) * sigma2 / 1000, 4)  # → 0.3007
# independent trees: the variance falls as 1/B
round(0.0 * sigma2 + (1 - 0.0) * sigma2 / 10, 4)  # → 0.1
```

That is why a forest adds random feature subsets on top of bagging: bagging
alone leaves every tree free to pick the same strong feature at the top, so
the trees stay correlated. Forcing each node to choose from a random subset
pushes $\rho$ down, which is the only way to push the floor down. Each tree
gets a little worse; the average gets better.

![Left, one fully grown tree carves a jagged patchwork with small islands around noisy points and scores 82% on validation; right, a forest of 100 trees draws a smoother boundary and scores 88.5%](figures/primer.ml.classical.forest_regions.svg)

**Reading it:** both panels show the same 300 training points and the
predicted class at every spot. On the left, a single unlimited tree fences
off little islands around individual noisy points. On the right, a hundred
unlimited trees vote: an island appears only where most trees agree, so the
one-off islands mostly vanish and the boundary follows the two moons more
smoothly. Each tree in the forest is just as overfitted as the one on the
left; their average is not.

![Validation accuracy against the number of trees on a log axis: one tree scores 85.5%, two dip to 83%, then the forest climbs to 88 to 89% by 10 to 20 trees and stays level, all above a single deep tree's 82%](figures/primer.ml.classical.forest_curve.svg)

**Reading it:** the horizontal axis is how many trees are averaged, on a log
scale; the vertical axis is accuracy on the 200 validation points. The
dashed line is a single deep tree grown on all the data. The first points
are jumpy (two fully grown trees that disagree give a 50/50 tie, which is
why two trees dip below one), but the curve climbs over the first ten or
twenty trees and then flattens: the private
part of the variance is gone, and what remains is the shared part the
formula predicts. Unlike depth, adding trees does not overfit; past the
plateau it only costs time.

**In code:** `averaged_variance` is the formula; `forest_curve` grows one forest and scores its first k trees for each k; `RandomForest.predict_proba` averages the trees' class shares.

**Why it matters:** a random forest is the most forgiving model in machine
learning. Its defaults (a few hundred unpruned trees, square root of the
features per node) work well on most tables with no tuning at all, which is
why it is the standard first model on new tabular data.

## Gradient boosting: each tree fixes the last one's mistakes

**Everyday picture.** Golf. Your first shot from the tee gets you near the
green. Your second shot does not start over from the tee: it aims at what is
*left*, the gap between the ball and the hole. Each shot is small and
corrects what remains. **Gradient boosting** builds a model the same way:
start with a crude guess, then add small trees one at a time, each trained
only on what the model so far still gets wrong.

Where a forest grows big trees side by side and averages them, boosting
grows small trees in sequence and adds them up. The forest fights variance;
boosting fights **bias** (being systematically off), a little at a time.

```mermaid
flowchart LR
  F0["F₀: start with the average"] --> R["residuals:<br/>what is still wrong<br/>r = y − F"]
  R --> H["fit a small tree hₘ<br/>to the residuals"]
  H --> U["Fₘ = Fₘ₋₁ + η · hₘ<br/>take a small step"]
  U -->|"next round"| R
  U --> OUT["after M rounds:<br/>F₀ + η·(h₁ + h₂ + … + h_M)"]
```

**Reading it:** the loop in the middle is the whole algorithm. Each round
measures what is still wrong (the residuals), trains a small tree to predict
*those*, and adds a shrunken copy of that tree to the model. The model is
never retrained from scratch; it only grows by addition. The final model is
the starting guess plus every tree's contribution, which is why a boosted
model is literally a sum of trees.

**Tiny example.** Four houses of size 1, 2, 3 and 4 sold for 1, 2, 6 and 7
(hundreds of thousands, say). Boost with **stumps** (trees with a single
question) and learning rate $\eta = 0.5$:

| size | price $y$ | $F_0$ | residual $y - F_0$ | stump $h_1$ | $F_1 = F_0 + 0.5\,h_1$ | new residual |
|---|---|---|---|---|---|---|
| 1 | 1 | 4 | −3 | −2.5 | 2.75 | −1.75 |
| 2 | 2 | 4 | −2 | −2.5 | 2.75 | −0.75 |
| 3 | 6 | 4 | 2 | 2.5 | 5.25 | 0.75 |
| 4 | 7 | 4 | 3 | 2.5 | 5.25 | 1.75 |

$F_0 = 4$ is the average price, the best single guess. The stump's best
question on the residuals is "size ≤ 2.5?", and each leaf predicts its
average residual: −2.5 for the small houses, +2.5 for the big ones. Adding
half of that moves every guess toward its price. The mean squared error
falls from 6.5 to 1.8125 in one round, and a second round brings it to
0.64.

### Residuals are the negative gradient

Why is this called *gradient* boosting? Because "fit what is still wrong"
is a special case of gradient descent (`primer.ml.optimizers`), taken in the
space of predictions instead of weights.

$$
L(y, F) = \tfrac{1}{2}\,(y - F)^{2}
\qquad\Longrightarrow\qquad
r = -\frac{\partial L}{\partial F} = y - F
$$

**Symbols**

| Symbol | Meaning here | For house 4 |
|---|---|---|
| $y$ | the true value | 7 |
| $F$ | the model's current prediction | $F_0 = 4$ |
| $L(y, F)$ | the **loss**: half the squared miss (the ½ only tidies the slope) | $\frac{1}{2}(7 - 4)^2 = 4.5$ |
| $\frac{\partial L}{\partial F}$ | the **derivative**: how fast the loss changes as the prediction $F$ is nudged up | $4 - 7 = -3$ |
| $-\frac{\partial L}{\partial F}$ | the direction that lowers the loss fastest, called the **negative gradient** | 3 |
| $r$ | the **residual**: what each tree is trained to predict | 3 |
| $\Longrightarrow$ | "which gives" | |

**In words:** "for squared error, the direction that most quickly reduces
the loss at each example is simply the true value minus the prediction: the
residual."

**With the numbers:** house 4 has $y = 7$ and $F = 4$. Nudge $F$ up to 4.001
and the loss falls from 4.5 to about 4.497, a slope of −3; the negative
gradient is +3, which is exactly $7 - 4$.

**In Python:**

```python
y, F, eps = 7.0, 4.0, 1e-6
# ∂L/∂F, measured by nudging F both ways
slope = (0.5 * (y - (F + eps)) ** 2 - 0.5 * (y - (F - eps)) ** 2) / (2 * eps)
round(slope, 6)  # → -3.0
# r = -∂L/∂F is the plain residual y - F
round(-slope, 6), y - F  # → (3.0, 3.0)
```

The payoff of the gradient view: swap in any loss with a slope and the same
loop still works. For yes/no labels with the log loss (`primer.ml.losses`),
$F$ is a score in log-odds and the negative gradient is $y - \sigma(F)$, the
label minus the predicted probability ($\sigma$ is the **sigmoid**, which
squashes a score into 0 to 1). Each tree then fits "how surprised were we
by each example".

**In code:** `GradientBoosting.negative_gradient` returns $y - F$ for `loss="squared"` and $y - \sigma(F)$ for `loss="log"`.

### The update, and the learning rate

$$
F_m(x) = F_{m-1}(x) + \eta\, h_m(x)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $x$ | one input (a house's size) | size 4 |
| $m$ | the round number, 1, 2, …, $M$ | 1 |
| $F_{m-1}(x)$ | the model's prediction before this round | $F_0(4) = 4$ |
| $h_m(x)$ | this round's small tree, trained on the residuals | $h_1(4) = 2.5$ |
| $\eta$ | the **learning rate** (Greek letter eta), also called **shrinkage**: the fraction of each tree's correction actually applied | 0.5 |
| $F_m(x)$ | the prediction after this round | 5.25 |

**In words:** "after each round, the new prediction is the old prediction
plus a shrunken copy of the tree that was trained to fix it."

**With the numbers:** big houses: $4 + 0.5 \times 2.5 = 5.25$; small houses:
$4 + 0.5 \times (-2.5) = 2.75$. Mean squared error before:
$(9 + 4 + 4 + 9)/4 = 6.5$; after: $(1.75^2 + 0.75^2 + 0.75^2 +
1.75^2)/4 = 1.8125$.

**In Python:**

```python
prices = [1, 2, 6, 7]
eta = 0.5
# F_0: the average price
F0 = sum(prices) / len(prices)
F0  # → 4.0
residuals = [y - F0 for y in prices]
residuals  # → [-3.0, -2.0, 2.0, 3.0]
# h_1: the stump predicts each side's average residual
small, big = (residuals[0] + residuals[1]) / 2, (residuals[2] + residuals[3]) / 2
h1 = [small, small, big, big]
h1  # → [-2.5, -2.5, 2.5, 2.5]
# F_1 = F_0 + η h_1
F1 = [F0 + eta * h for h in h1]
F1  # → [2.75, 2.75, 5.25, 5.25]
# mean squared error before and after the round
sum((y - F0) ** 2 for y in prices) / 4  # → 6.5
sum((y - f) ** 2 for y, f in zip(prices, F1)) / 4  # → 1.8125
```

Why not take the whole correction ($\eta = 1$)? Each tree is fitted to the
training data, noise included. Taking the whole step lets the first few
trees commit hard to whatever they saw; taking small steps means many trees
must agree before the model moves far, which averages away some of the
noise. The price is more rounds.

![Boosted depth-2 trees fitting noisy points around a sine curve: after 1 round two flat steps, after 5 rounds a coarse staircase, after 50 rounds a jagged staircase that tracks the sine and starts chasing individual points](figures/primer.ml.classical.boosting_steps.svg)

**Reading it:** grey dots are 80 noisy samples of the dotted sine curve, and
the blue line is the boosted model's prediction everywhere between 0 and 6.
After 1 round the model is the average plus 0.3 of one small tree: a
line with just two flat steps. After 5 rounds it has the rough shape.
After 50 it follows the curve closely and has begun to jump for individual
noisy points, the first sign of the overfitting the next figure measures.
Every version is made of flat steps, because
every tree predicts a constant in each leaf; the curve is built from many
small staircases added together.

![Mean squared error per boosting round for learning rates 1, 0.3 and 0.1: dashed training curves fall toward zero; solid validation curves bottom out at round 4, 9 and 31 respectively, lowest for 0.1, and then climb](figures/primer.ml.classical.boosting_curves.svg)

**Reading it:** dashed lines are training error and solid lines are error on
400 fresh points, per round, for three learning rates. Every training curve
falls toward zero, since 200 rounds of trees can memorise 80 points. Each
validation curve has a best round (the dot) and then rises, because later
trees are fitting noise. With $\eta = 1$ the best comes at round 4 (0.122)
and is poor; with $\eta = 0.1$ it comes at round 31 and is the lowest
(0.110), close to the noise floor of 0.09 that no model can beat. Smaller
steps generalize better but need more rounds, and every learning rate needs
a stopping point chosen on validation data: early stopping, as in
`primer.ml.regularization`.

**In code:** `GradientBoosting` runs the loop and records the training loss after each round; `GradientBoosting.staged_loss` scores held-out data after each round; `boosting_worked_example` is the four-house table; `boosting_curves` sweeps the learning rate on `sine_data`.

**Why it matters:** gradient-boosted trees (through libraries such as
XGBoost, LightGBM and CatBoost) are the most common winning model on tabular
data in practice. They add refinements this lesson leaves out, such as a
second-order (Newton) step for each leaf's value, penalties on tree size,
fast histogram-based split search, and native handling of missing values,
but the loop is the one above. The two knobs that matter most are the ones
you have met: tree depth and the learning rate, with the number of rounds
chosen by early stopping.

## When trees win, and when neural networks do

**Everyday picture.** A spreadsheet and a photograph are both grids of
numbers, but they carry meaning differently. In a spreadsheet each column
means something on its own: "income is under 30,000" is a sentence a loan
officer would write. In a photo, "pixel (212, 40) is brighter than 0.7"
means nothing; the meaning is in how thousands of pixels are arranged. A
tree asks questions of one column at a time, which is perfect for the
spreadsheet and hopeless for the photo. A neural network builds its own
features out of combinations of inputs, which is what the photo needs.

**Tiny example.** A toy loan table: 800 training rows and 400 validation
rows, five columns in their natural units: age in years, income in dollars
(spanning a factor of 100), a count of late payments, a region code 0 to 5
(where region 4 is not "twice region 2"), and one column of pure random
noise. The label follows rules of the kind a lender writes (four or more
late payments; under 30,000 and under 30 years old; two late payments in
certain regions or on a very low income), with 5% of labels flipped at
random.

| model | what it was given | validation accuracy |
|---|---|---|
| always predict "fine" | nothing | 77.75% |
| single tree, depth 5 | the raw table | 92.25% |
| random forest, 50 trees | the raw table | 95.75% |
| gradient boosting, 100 depth-3 trees | the raw table | 96.00% |
| small neural net (32 hidden units) | the raw table | 77.75% |
| same net | every column rescaled to mean 0, spread 1 | 86.25% |
| same net | rescaled, plus log income and one column per region | 92.25% |

![Validation accuracy on the loan table: random forest 95.75% and gradient boosting 96.0% lead; the neural net scores 77.75% on raw inputs (the same as always saying fine), 86.25% scaled, and 92.25% with engineered features](figures/primer.ml.classical.showdown.svg)

**Reading it:** blue bars are tree models, orange bars are the same small
neural network given three versions of the same table, and the dashed line
is the score for ignoring the inputs and always predicting "fine". The
trees read the table exactly as it came. The net on raw inputs sits on the
dashed line: incomes around 50,000 swamp every other column and saturate
its neurons, so it learns nothing. Rescaling rescues it partly; hand-made
features (the logarithm of income, a yes/no column per region) rescue it
more. Even then it only ties a single depth-5 tree, and the ensembles stay
ahead.

```mermaid
flowchart TD
  START{"what does one input look like?"} -->|"a row of a table:<br/>each column means<br/>something alone"| SIZE{"how much data?"}
  START -->|"pixels, audio samples,<br/>words, sequences"| NN["neural network<br/>(often a pretrained one)"]
  SIZE -->|"thousands to millions of rows"| GBT["gradient-boosted trees<br/>or a random forest first"]
  SIZE -->|"huge, or mixed with<br/>text or images"| MIX["neural network, or trees<br/>on top of neural embeddings"]
  GBT --> CHECK["compare against a neural net<br/>only if the gap matters"]
```

**Reading it:** the first question is about the *shape* of one input, not
about fashion. Tables go down the left; perceptual and sequential data go
right. On the left, the size of the data decides: at typical business
scale, start with trees. The bottom-right branch is common in practice: a
neural network turns text or images into **embeddings**
(`primer.ml.embeddings.word2vec`), and those numbers become extra columns
for a boosted-tree model.

Why trees fit tables so well, each point visible in this lesson's code:

- **Units don't matter.** A tree only asks "is this bigger than that?", so
  any order-keeping change to a column (dollars to thousands, a logarithm,
  a square) leaves every prediction unchanged. A neural net multiplies and
  adds columns, so their scales collide.
- **Thresholds are native.** Rules like "four or more late payments" are
  one question to a tree; a net must approximate the sharp edge with smooth
  curves.
- **Irrelevant columns are mostly ignored**, because a column that never
  wins a split is never used.
- **Modest data is enough.** A few hundred rows can grow a useful tree; a
  net with thousands of weights needs far more to pin them down.

Where neural networks win, and trees cannot follow:

- **Perceptual and sequential data**: images, audio, text. The useful
  features are built from arrangements of raw values, which is what
  convolutions (`primer.ml.cnn_rnn`) and attention (`primer.ml.transformer`)
  learn.
- **Huge data and pretraining.** A net's capacity keeps paying off as data
  grows, and a pretrained model brings knowledge from billions of examples.
  A tree starts from nothing every time.
- **Smooth trends and extrapolation.** A tree's leaves hold averages of
  training values, so it can never predict outside the range it saw. The
  four-house model predicts the same price for a house of size 40 as for
  size 4.
- **End-to-end training** with other neural parts, since a tree has no
  gradient to pass along.

In a careful benchmark on 45 tabular datasets of about 10,000 rows each
(Grinsztajn, Oyallon and Varoquaux, 2022), tuned tree-based models remained
ahead of tuned deep learning models. Their analysis names three reasons that
match this lesson: trees shrug off uninformative columns, they respect each
column's own axis, and they fit irregular, step-like functions easily.
Research on neural networks
for tables is active and the gap may narrow; the practical rule is to start
with boosted trees on a table and make any other model beat them on your
validation data.

**In code:** `make_tabular` builds the loan table; `tabular_showdown` trains every model in the table above (the net is `primer.ml.neural_net.MLP`); `engineer_features` is the hand-made preparation the net needed.

### Feature importance, and its limits

**Everyday picture.** After a football season you want to know which
players mattered. One way: count how often each player touched the ball in
training. Another: bench each player for a real match and see how much the
score suffers. The first is quick but credits whoever is busy; the second
measures what the team actually depends on.

Trees offer the first kind for free. **Impurity importance** adds up, for
each feature, the impurity its splits removed (weighted by how many examples
reached each split), measured on the training data. **Permutation
importance** is the second kind: take held-out data, shuffle one column so
it no longer lines up with the labels, and measure how much accuracy drops.

**Tiny example.** The 30-tree forest scores 96.0% on the 400 validation
rows. Shuffle the late-payments column and it scores 77.5%: a drop of 18.5
points, so the model leans on that column heavily. Shuffle the noise column
and it scores 95.75%, a drop of 0.25 points (one row in 400).

$$
I_j = \text{acc}(X_{\text{val}}) - \text{acc}\big(X_{\text{val}} \text{ with column } j \text{ shuffled}\big)
$$

**Symbols**

| Symbol | Meaning here | For late payments |
|---|---|---|
| $j$ | the feature (column) being tested | late_payments |
| $X_{\text{val}}$ | held-out rows the model never trained on | 400 rows |
| $\text{acc}(\ldots)$ | the share of those rows the model gets right | 0.96 |
| shuffled | the column's values randomly reordered: same values, links to the labels broken | 0.775 |
| $I_j$ | the permutation importance of feature $j$: accuracy lost without it | 0.185 |

**In words:** "a feature's importance is how much accuracy the model loses
on fresh data when that feature is scrambled."

**With the numbers:** late payments $0.96 - 0.775 = 0.185$; noise
$0.96 - 0.9575 = 0.0025$.

**In Python:**

```python
# validation rows answered correctly, out of 400
correct_of_400 = {"intact": 384, "late_payments shuffled": 310, "noise shuffled": 383}
acc = {k: v / 400 for k, v in correct_of_400.items()}
# I_j = acc(X_val) - acc(X_val with column j shuffled)
round(acc["intact"] - acc["late_payments shuffled"], 4)  # → 0.185
round(acc["intact"] - acc["noise shuffled"], 4)  # → 0.0025
```

![Two bar charts over age, income, late payments, region and noise: impurity importance gives the noise column 11% of the credit; permutation importance gives it 0.25 points of accuracy, far below late payments at 18.5 points](figures/primer.ml.classical.importance.svg)

**Reading it:** the left panel is impurity importance from training, as
shares that add to 100%; the right is permutation importance on validation
data, in accuracy lost. Both agree the real columns matter. They disagree
about noise: on the left it gets 11% of the credit, because deep trees can
always find some threshold on a continuous random column that tidies up a
few training rows. On the right, scrambling it costs almost nothing,
because none of those splits help on new data.

The limits worth remembering:

- **Impurity importance favours columns with many distinct values**
  (continuous numbers, IDs), since they offer more thresholds to overfit
  with. Measure on held-out data instead.
- **Correlated columns split the credit.** If income appears twice (in
  dollars and in thousands), each copy looks half as important, and
  shuffling one does little because the other covers for it.
- **Importance is not cause.** It says what *this model* relies on to
  predict, not what would change the outcome in the world.

A single shallow tree is the one truly readable model here: print
`DecisionTree.rules` and you are reading the entire model. A forest of
hundreds of deep trees is not readable, and importances are a summary of it,
not an explanation.

**In code:** `permutation_importance` shuffles one column at a time on held-out data; `DecisionTree.feature_importances` and `RandomForest.feature_importances` are the impurity kind.

## In 20 seconds

- **A decision tree** asks yes/no questions about one column at a time,
  choosing each question greedily to leave the purest piles (lowest
  weighted Gini impurity or entropy), until a stopping rule says stop.
- **Deep trees overfit**: a question for every example gives 100% on
  training data and poor results on new data. Depth is the capacity knob.
- **Random forests** average many deep trees, each grown on a bootstrap
  sample with random feature subsets. Averaging removes the variance the
  trees don't share; decorrelating them lowers the part they do.
- **Gradient boosting** adds small trees one at a time, each fit to the
  residuals (the negative gradient of the loss), shrunk by a learning rate,
  with the number of rounds set by early stopping.
- **On tables, trees usually win**: no scaling, native thresholds, modest
  data. Neural networks win on images, audio, text and huge datasets.

## Self-test questions

**Why is Gini impurity 0.5 for a pile that is half spam and half not?**
It is the chance that two emails drawn at random (with replacement) carry
different labels: $1 - (0.5^2 + 0.5^2) = 0.5$. For two labels that is the
most mixed a pile can be.

**Why weight each child pile by its size when scoring a split?**
Otherwise a question that peels off one example into a tiny pure pile
would look as good as one that sorts half the data cleanly. Weighting by
size measures how much of the data the question actually tidied.

**Why does a tree need no feature scaling, when a neural network does?**
A tree only compares one column against a threshold, so any change that
keeps the order of values (rescaling, logarithms) produces the same splits
and the same predictions. A network multiplies and adds columns together,
so a column measured in tens of thousands drowns the others and saturates
its neurons.

**A tree scores 100% on training data and 80% on validation. What happened,
and what are two fixes?**
It overfitted: it kept splitting until each leaf fenced off individual
noisy examples. Limit its depth (or require a minimum number of examples per
leaf, or prune), or replace it with a random forest that averages many such
trees.

**Why does a random forest use random feature subsets, not just bootstrap
samples?**
The forest's variance is $\rho\sigma^2 + (1 - \rho)\sigma^2/B$. More trees
only shrink the second term; the floor is set by the correlation $\rho$
between trees. With bagging alone every tree tends to pick the same strong
feature first and they stay correlated. Random subsets force different
trees down different paths, lowering $\rho$.

**Does adding more trees overfit a random forest? Does adding more rounds
overfit gradient boosting?**
More trees in a forest does not: accuracy rises and then levels off, only
costing time. More rounds of boosting does: each round fits the training
residuals more closely, so validation error reaches a minimum and then
climbs. Boosting needs early stopping; forests don't.

**Why is the residual the right target for each boosting tree?**
For squared-error loss $\frac{1}{2}(y - F)^2$, the negative derivative with
respect to the prediction $F$ is $y - F$, the residual. Fitting a tree to it
and taking a step is gradient descent on the predictions. With another
loss, the tree fits that loss's negative gradient instead, such as $y - p$
for log loss.

**What does a smaller learning rate buy in gradient boosting, and what does
it cost?**
Each tree contributes only a fraction of its correction, so no single noisy
tree can move the model far and many trees must agree. That usually
generalizes better (0.110 against 0.122 in this lesson's sweep). It costs
more rounds, so more training and prediction time.

**Why can't a boosted-tree model predict a house price above the highest
price it trained on?**
Every leaf predicts an average of training values, and the model is a sum of
such leaves anchored at the training mean. Beyond the edge of the training
data every question gives the same answer as at the edge, so the prediction
stays flat.

**What is wrong with trusting impurity-based feature importance?**
It is measured on training data, where splits on noise still tidy up piles.
It favours columns with many distinct values and splits credit between
correlated columns. Permutation importance on held-out data is more honest,
and neither measures cause and effect.

**When would you choose a neural network over gradient-boosted trees?**
When one input is an image, audio clip, text or other sequence whose
meaning lies in the arrangement of raw values; when the data is huge or a
pretrained model can be reused; when predictions must extrapolate smoothly;
or when the model must be trained end to end with other neural parts.

## The papers behind this lesson

- **Breiman, Friedman, Olshen and Stone, *Classification and Regression
  Trees* (1984)**: https://doi.org/10.1201/9781315139470. The CART book:
  binary trees split by Gini impurity, regression trees with averaged
  leaves, and cost-complexity pruning, the recipe this lesson's tree
  follows.
- **Quinlan, *Induction of Decision Trees*, Machine Learning 1 (1986)**:
  https://doi.org/10.1007/BF00116251. ID3, which grows trees by choosing
  the split with the largest information gain (drop in entropy).
- **Breiman, *Bagging Predictors*, Machine Learning 24 (1996)**:
  https://doi.org/10.1007/BF00058655. Showed that averaging models trained
  on bootstrap samples reduces the error of unstable learners such as
  trees.
- **Breiman, *Random Forests*, Machine Learning 45 (2001)**:
  https://doi.org/10.1023/A:1010933404324. Added random feature subsets at
  each split, out-of-bag error estimates, and permutation importance.
- **Friedman, *Greedy Function Approximation: A Gradient Boosting Machine*,
  Annals of Statistics 29 (2001)**: https://doi.org/10.1214/aos/1013203451.
  Framed boosting as gradient descent in function space, fitting each tree
  to the negative gradient of any differentiable loss, with shrinkage.
- **Chen and Guestrin, *XGBoost: A Scalable Tree Boosting System* (2016)**:
  https://arxiv.org/abs/1603.02754. A regularized, second-order boosting
  objective with fast, sparsity-aware split finding, which made boosted
  trees the default on tabular problems.
- **Grinsztajn, Oyallon and Varoquaux, *Why do tree-based models still
  outperform deep learning on tabular data?* (2022)**:
  https://arxiv.org/abs/2207.08815. A benchmark on 45 medium-sized tabular
  datasets where tuned tree ensembles beat tuned neural networks, and an
  analysis of why.

## Further reading

- scikit-learn user guide, *Decision Trees*: https://scikit-learn.org/stable/modules/tree.html
- scikit-learn user guide, *Ensembles: gradient boosting, random forests, bagging*: https://scikit-learn.org/stable/modules/ensemble.html
- scikit-learn user guide, *Permutation feature importance*: https://scikit-learn.org/stable/modules/permutation_importance.html
- XGBoost documentation, *Introduction to Boosted Trees*: https://xgboost.readthedocs.io/en/stable/tutorials/model.html
- Hastie, Tibshirani and Friedman, *The Elements of Statistical Learning* (free PDF from the authors; chapters 9, 10 and 15 cover trees, boosting and random forests): https://hastie.su.domains/ElemStatLearn/
- Strobl et al., *Bias in random forest variable importance measures* (2007): https://doi.org/10.1186/1471-2105-8-25
- Grinsztajn et al., *Why do tree-based models still outperform deep learning on tabular data?* (2022): https://arxiv.org/abs/2207.08815
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np

from primer._show import banner, say, table, takeaway

# ---------------------------------------------------------------------------
# 1. Impurity: how mixed-up is a pile of labels?
# ---------------------------------------------------------------------------


def _class_shares(y: np.ndarray) -> np.ndarray:
    # Shares of each label present; absent labels contribute nothing to either measure.
    _, counts = np.unique(y, return_counts=True)
    return counts / counts.sum()


def gini(y: np.ndarray) -> float:
    """Gini impurity, 1 - Σ p_k²: the chance two labels drawn at random (with replacement) disagree."""
    if len(y) == 0:
        return 0.0
    p = _class_shares(y)
    return float(1.0 - np.sum(p**2))


def entropy(y: np.ndarray) -> float:
    """Entropy in bits, -Σ p_k log2 p_k: how many yes/no questions a label still costs on average."""
    if len(y) == 0:
        return 0.0
    p = _class_shares(y)
    return float(-np.sum(p * np.log2(p)))


def split_impurity(y_left: np.ndarray, y_right: np.ndarray, impurity=gini) -> float:
    """Impurity after a split: each side's impurity, weighted by its share of the examples."""
    n = len(y_left) + len(y_right)
    return len(y_left) / n * impurity(y_left) + len(y_right) / n * impurity(y_right)


# ---------------------------------------------------------------------------
# 2. Choosing a split: try every feature and every threshold
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Split:
    """The best question found for a node: "is feature <= threshold?", and the impurity it leaves."""

    feature: int
    threshold: float
    impurity: float


def _side_impurities(counts: np.ndarray, n_side: np.ndarray, criterion: str) -> np.ndarray:
    """Impurity of one side for every candidate threshold at once. counts: (candidates, classes)."""
    p = counts / n_side[:, None]
    if criterion == "gini":
        return 1.0 - np.sum(p**2, axis=1)
    # 0·log 0 counts as 0: an absent class adds no uncertainty.
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.sum(np.where(p > 0, p * np.log2(p), 0.0), axis=1)


def best_split(
    X: np.ndarray,
    y: np.ndarray,
    criterion: str = "gini",
    features: np.ndarray | None = None,
    min_samples_leaf: int = 1,
) -> Split | None:
    """The feature and threshold whose split leaves the lowest weighted impurity, or None.

    `criterion` is "gini" or "entropy" for class labels (integers 0..K-1), or
    "squared" for numbers (impurity = variance, as in a regression tree).
    Sorting each feature once turns "try every threshold" into running totals,
    so a node costs O(n log n) per feature instead of O(n²).
    """
    n = len(y)
    if n < 2 * min_samples_leaf:
        return None
    if criterion == "squared":
        parent = float(np.var(y))
    else:
        parent = gini(y) if criterion == "gini" else entropy(y)
    if parent <= 1e-12:  # already pure: nothing left to separate
        return None
    features = np.arange(X.shape[1]) if features is None else features
    best: Split | None = None
    n_left = np.arange(1, n)  # candidate i puts the first i sorted examples on the left
    n_right = n - n_left
    for f in features:
        order = np.argsort(X[:, f], kind="stable")
        xs, ys = X[order, f], y[order]
        # A threshold can only sit between two different values; equal values must stay together.
        valid = (xs[1:] > xs[:-1]) & (n_left >= min_samples_leaf) & (n_right >= min_samples_leaf)
        if not valid.any():
            continue
        if criterion == "squared":
            # Sum of squared errors from running sums: Σy² - (Σy)²/n on each side.
            s, s2 = np.cumsum(ys)[:-1], np.cumsum(ys**2)[:-1]
            tot, tot2 = ys.sum(), (ys**2).sum()
            sse = (s2 - s**2 / n_left) + ((tot2 - s2) - (tot - s) ** 2 / n_right)
            weighted = sse / n
        else:
            onehot = np.eye(int(y.max()) + 1)[ys]  # (n, classes)
            left = np.cumsum(onehot, axis=0)[:-1]  # class counts left of each candidate
            right = onehot.sum(axis=0) - left
            weighted = (n_left * _side_impurities(left, n_left, criterion) + n_right * _side_impurities(right, n_right, criterion)) / n
        weighted = np.where(valid, weighted, np.inf)
        i = int(np.argmin(weighted))
        # Strictly lower wins, so ties go to the earliest feature: the tree is reproducible.
        if best is None or weighted[i] < best.impurity - 1e-12:
            best = Split(int(f), float((xs[i] + xs[i + 1]) / 2), float(weighted[i]))
    return best


def candidate_splits(X: np.ndarray, y: np.ndarray) -> list[Split]:
    """Every question worth asking at one node, with the Gini impurity it would leave.

    Thresholds sit halfway between neighbouring distinct values of a
    feature; anywhere else in the gap gives the same split.
    """
    out = []
    for f in range(X.shape[1]):
        values = np.unique(X[:, f])
        for t in (values[1:] + values[:-1]) / 2:
            left = X[:, f] <= t
            out.append(Split(f, float(t), split_impurity(y[left], y[~left])))
    return out

# ---------------------------------------------------------------------------
# 3. The tree: ask, split, repeat
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """One box of the flowchart. A leaf has no question and carries the prediction in `value`."""

    value: np.ndarray  # class shares (classification) or [mean] (regression)
    n_samples: int
    impurity: float
    feature: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


@dataclass
class DecisionTree:
    """A CART-style decision tree, grown greedily from the top.

    criterion: "gini" or "entropy" to classify integer labels, "squared" to
    predict numbers. max_depth None grows until every leaf is pure.
    max_features limits how many randomly chosen features each node may try
    ("sqrt" or an int); that is the random forest's extra ingredient.
    """

    max_depth: int | None = None
    criterion: str = "gini"
    min_samples_leaf: int = 1
    max_features: int | str | None = None
    seed: int = 0
    root: Node | None = field(default=None, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int | None = None) -> "DecisionTree":
        X = np.asarray(X, dtype=float)
        self.regression = self.criterion == "squared"
        y = np.asarray(y, dtype=float if self.regression else int)
        # A bootstrap sample can miss a class; the forest passes the true count so shares line up.
        self.n_classes = None if self.regression else (n_classes or int(y.max()) + 1)
        self.n_features = X.shape[1]
        self._importance = np.zeros(self.n_features)
        self._rng = np.random.default_rng(self.seed)
        self._n_total = len(y)
        self.root = self._grow(X, y, depth=0)
        return self

    def _leaf_value(self, y: np.ndarray) -> np.ndarray:
        if self.regression:
            return np.array([y.mean()])
        return np.bincount(y, minlength=self.n_classes) / len(y)

    def _impurity(self, y: np.ndarray) -> float:
        if self.regression:
            return float(np.var(y))
        return gini(y) if self.criterion == "gini" else entropy(y)

    def _features_to_try(self) -> np.ndarray:
        d = self.n_features
        k = {None: d, "sqrt": max(1, int(math.sqrt(d)))}.get(self.max_features, self.max_features)
        if k >= d:
            return np.arange(d)
        # A fresh random subset at every node keeps the trees of a forest different from each other.
        return np.sort(self._rng.choice(d, size=k, replace=False))

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        node = Node(self._leaf_value(y), len(y), self._impurity(y))
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        split = best_split(X, y, self.criterion, self._features_to_try(), self.min_samples_leaf)
        if split is None:
            return node
        go_left = X[:, split.feature] <= split.threshold
        # Credit the feature with the impurity it removed, weighted by how many examples reached here.
        self._importance[split.feature] += len(y) / self._n_total * (node.impurity - split.impurity)
        node.feature, node.threshold = split.feature, split.threshold
        node.left = self._grow(X[go_left], y[go_left], depth + 1)
        node.right = self._grow(X[~go_left], y[~go_left], depth + 1)
        return node

    def _values(self, X: np.ndarray) -> np.ndarray:
        """Route every row to its leaf and return the leaf values, (n, classes) or (n, 1)."""
        X = np.asarray(X, dtype=float)
        out = np.empty((len(X), len(self.root.value)))

        def route(node: Node, rows: np.ndarray) -> None:
            if node.is_leaf:
                out[rows] = node.value
                return
            go_left = X[rows, node.feature] <= node.threshold
            route(node.left, rows[go_left])
            route(node.right, rows[~go_left])

        route(self.root, np.arange(len(X)))
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class shares of the leaf each row lands in, (n, classes)."""
        return self._values(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """The leaf's mean for regression, or the leaf's majority class."""
        values = self._values(X)
        return values[:, 0] if self.regression else values.argmax(axis=1)

    @property
    def depth(self) -> int:
        def d(node: Node) -> int:
            return 0 if node.is_leaf else 1 + max(d(node.left), d(node.right))

        return d(self.root)

    @property
    def n_leaves(self) -> int:
        def count(node: Node) -> int:
            return 1 if node.is_leaf else count(node.left) + count(node.right)

        return count(self.root)

    @property
    def feature_importances(self) -> np.ndarray:
        """Impurity removed by each feature's splits, as shares that sum to 1."""
        total = self._importance.sum()
        return self._importance / total if total > 0 else self._importance

    def rules(self, names: list[str] | None = None, classes: list[str] | None = None) -> list[str]:
        """The tree as nested if/else lines: the whole model, readable by a person."""
        names = names or [f"x{i}" for i in range(self.n_features)]
        lines: list[str] = []

        def leaf_text(node: Node) -> str:
            if self.regression:
                return f"predict {node.value[0]:.3g}"
            k = int(node.value.argmax())
            label = classes[k] if classes else str(k)
            noun = "example" if node.n_samples == 1 else "examples"
            return f"predict {label}  ({node.n_samples} {noun}, {node.value[k]:.0%} agree)"

        def walk(node: Node, indent: str) -> None:
            if node.is_leaf:
                lines.append(indent + leaf_text(node))
                return
            lines.append(f"{indent}if {names[node.feature]} <= {node.threshold:g}:")
            walk(node.left, indent + "    ")
            lines.append(f"{indent}else:  # {names[node.feature]} > {node.threshold:g}")
            walk(node.right, indent + "    ")

        walk(self.root, "")
        return lines


# ---------------------------------------------------------------------------
# 4. Worked examples and toy data
# ---------------------------------------------------------------------------

SPAM_FEATURES = ["links", "known_sender"]


def spam_emails() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The eight emails of the worked example: (links, known sender?) and 1 = spam.

    | email | links | known sender | spam? |
    |---|---|---|---|
    | 1 | 0 | yes | no |
    | 2 | 1 | yes | no |
    | 3 | 4 | yes | no |
    | 4 | 0 | no  | no |
    | 5 | 3 | no  | yes |
    | 6 | 5 | no  | yes |
    | 7 | 6 | no  | yes |
    | 8 | 1 | no  | yes |
    """
    X = np.array([[0, 1], [1, 1], [4, 1], [0, 0], [3, 0], [5, 0], [6, 0], [1, 0]], dtype=float)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y, list(SPAM_FEATURES)


def moons_split(n: int = 500, noise: float = 0.35, n_train: int = 300, seed: int = 1):
    """Noisy two-moons points (`primer.ml.neural_net.make_moons`), cut into training and validation sets."""
    from primer.ml.neural_net import make_moons

    X, y = make_moons(n=n, noise=noise, seed=seed)
    # make_moons lists one moon, then the other; shuffle so both sets hold both classes.
    order = np.random.default_rng(seed).permutation(n)
    X, y = X[order], y[order].astype(int)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def _accuracy(model, X: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(model.predict(X) == y))


def depth_sweep(depths=(1, 2, 3, 4, 5, 6, 8, 10, 12, None)) -> list[dict]:
    """Train and validation accuracy of one tree per depth limit on the noisy moons."""
    X, y, X_val, y_val = moons_split()
    rows = []
    for depth in depths:
        tree = DecisionTree(max_depth=depth).fit(X, y)
        rows.append(dict(depth=depth, train_accuracy=_accuracy(tree, X, y), val_accuracy=_accuracy(tree, X_val, y_val), leaves=tree.n_leaves))
    return rows


# ---------------------------------------------------------------------------
# 5. Random forests: bagging plus random feature subsets
# ---------------------------------------------------------------------------


def bootstrap_sample(n: int, rng: np.random.Generator) -> np.ndarray:
    """n row indices drawn with replacement: some rows twice or more, about a third not at all."""
    return rng.integers(0, n, size=n)


def out_of_bag_fraction(n: int) -> float:
    """(1 - 1/n)^n: the chance a given row is never drawn in n draws. Tends to 1/e ≈ 0.368."""
    return (1 - 1 / n) ** n


def averaged_variance(sigma2: float, rho: float, n_trees: int) -> float:
    """Variance of the average of n_trees predictions, each with variance sigma2 and pairwise correlation rho.

    ρσ² + (1 - ρ)σ²/B: more trees shrink only the second term, so the
    correlation between trees sets the floor. Random feature subsets exist
    to push ρ down.
    """
    return rho * sigma2 + (1 - rho) * sigma2 / n_trees


@dataclass
class RandomForest:
    """Many deep trees, each grown on its own bootstrap sample and trying only
    a random subset of features at every node; their class shares are averaged."""

    n_trees: int = 100
    max_depth: int | None = None
    max_features: int | str | None = "sqrt"
    seed: int = 0
    trees: list[DecisionTree] = field(default_factory=list, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        rng = np.random.default_rng(self.seed)
        n_classes = int(y.max()) + 1
        self.trees = []
        for _ in range(self.n_trees):
            rows = bootstrap_sample(len(y), rng)
            # Each tree gets its own seed, so its feature subsets differ from its neighbours'.
            tree = DecisionTree(max_depth=self.max_depth, max_features=self.max_features, seed=int(rng.integers(2**31)))
            self.trees.append(tree.fit(X[rows], y[rows], n_classes=n_classes))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.mean([t.predict_proba(X) for t in self.trees], axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    @property
    def feature_importances(self) -> np.ndarray:
        """Each tree's impurity importance, averaged over the forest."""
        return np.mean([t.feature_importances for t in self.trees], axis=0)


def forest_curve(n_trees=(1, 2, 3, 5, 10, 20, 50, 100), seed: int = 0) -> list[dict]:
    """Validation accuracy of a forest on the noisy moons as trees are added.

    One forest of max(n_trees) is grown; the first k trees' average gives the
    k-tree forest, exactly as if it had been grown alone.
    """
    X, y, X_val, y_val = moons_split()
    forest = RandomForest(n_trees=max(n_trees), seed=seed).fit(X, y)
    probas = np.cumsum([t.predict_proba(X_val) for t in forest.trees], axis=0)
    single = DecisionTree().fit(X, y)
    return [
        dict(n_trees=k, val_accuracy=float(np.mean(probas[k - 1].argmax(axis=1) == y_val)), single_tree=_accuracy(single, X_val, y_val))
        for k in n_trees
    ]


# ---------------------------------------------------------------------------
# 6. Gradient boosting: each small tree fixes what the others still get wrong
# ---------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


@dataclass
class GradientBoosting:
    """F_m = F_{m-1} + η·h_m, where each h_m is a small regression tree fit to
    the negative gradient of the loss (the residuals, for squared loss).

    loss="squared" predicts numbers; loss="log" classifies 0/1 labels, with F
    as log-odds. `train_loss` holds the training loss before any tree and
    after each round: mean squared error, or cross-entropy for "log".
    """

    n_rounds: int = 100
    learning_rate: float = 0.1
    max_depth: int = 2
    loss: str = "squared"
    trees: list[DecisionTree] = field(default_factory=list, init=False)
    train_loss: list[float] = field(default_factory=list, init=False)

    @staticmethod
    def negative_gradient(y: np.ndarray, F: np.ndarray, loss: str) -> np.ndarray:
        """-∂L/∂F for each example: the direction that most quickly reduces its loss.

        Squared loss ½(y - F)² gives y - F, the plain residual. Log loss
        gives y - sigmoid(F): the label minus the predicted probability.
        """
        return y - F if loss == "squared" else y - _sigmoid(F)

    def _loss(self, y: np.ndarray, F: np.ndarray) -> float:
        if self.loss == "squared":
            return float(np.mean((y - F) ** 2))
        p = np.clip(_sigmoid(F), 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoosting":
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        # F_0: the best single constant. The mean for squared loss; the log-odds of the base rate for log loss.
        mean = y.mean()
        self.f0 = mean if self.loss == "squared" else math.log(mean / (1 - mean))
        F = np.full(len(y), self.f0)
        self.trees, self.train_loss = [], [self._loss(y, F)]
        for _ in range(self.n_rounds):
            residual = self.negative_gradient(y, F, self.loss)
            tree = DecisionTree(max_depth=self.max_depth, criterion="squared").fit(X, residual)
            # Take only a fraction η of the step: many small corrections generalize better than a few big ones.
            F = F + self.learning_rate * tree.predict(X)
            self.trees.append(tree)
            self.train_loss.append(self._loss(y, F))
        return self

    def staged_scores(self, X: np.ndarray) -> np.ndarray:
        """F_0, F_1, ..., F_M for every row: (rounds + 1, n)."""
        steps = [np.full(len(X), self.f0)] + [self.learning_rate * t.predict(X) for t in self.trees]
        return np.cumsum(steps, axis=0)

    def staged_loss(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        """The loss on (X, y) before any tree and after each round: the validation curve."""
        return [self._loss(np.asarray(y, dtype=float), F) for F in self.staged_scores(X)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        F = self.staged_scores(X)[-1]
        return F if self.loss == "squared" else (F > 0).astype(int)


def boosting_worked_example(learning_rate: float = 0.5) -> dict:
    """Four houses (size 1..4, price 1, 2, 6, 7) and one round of boosting with a one-question tree."""
    X, y = np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([1.0, 2.0, 6.0, 7.0])
    model = GradientBoosting(n_rounds=1, learning_rate=learning_rate, max_depth=1).fit(X, y)
    F0 = np.full(4, model.f0)
    residuals = GradientBoosting.negative_gradient(y, F0, "squared")
    return dict(
        F0=model.f0,
        residuals=residuals.tolist(),
        stump=model.trees[0].predict(X).tolist(),
        F1=model.predict(X).tolist(),
        mse=model.train_loss,
    )


def sine_data(n: int, noise: float = 0.3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """n points of y = sin(x) + noise for x in [0, 6]: a curve for regression trees to approximate."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 6, n)
    return x[:, None], np.sin(x) + rng.normal(0, noise, n)


def boosting_curves(learning_rates=(1.0, 0.3, 0.1), n_rounds: int = 200, max_depth: int = 2) -> list[dict]:
    """Training and validation mean squared error per round on 80 noisy sine points, for each learning rate."""
    X, y = sine_data(80, seed=0)
    X_val, y_val = sine_data(400, seed=1)
    rows = []
    for lr in learning_rates:
        model = GradientBoosting(n_rounds=n_rounds, learning_rate=lr, max_depth=max_depth).fit(X, y)
        val = model.staged_loss(X_val, y_val)
        rows.append(dict(learning_rate=lr, train_loss=model.train_loss, val_loss=val, best_round=int(np.argmin(val))))
    return rows


# ---------------------------------------------------------------------------
# 7. When trees win: a tabular showdown
# ---------------------------------------------------------------------------

TABULAR_FEATURES = ["age", "income", "late_payments", "region", "noise"]


def make_tabular(n: int = 1200, n_train: int = 800, seed: int = 0):
    """A toy loan table with the mix real spreadsheets have.

    age in years, income in dollars (spanning 100×), a count of late
    payments, a region code 0..5 whose numbers mean nothing in order, and a
    column of pure noise. The label (1 = the loan went bad) follows
    threshold rules of the kind a lender writes, with 5% of labels flipped.
    Returns X_train, y_train, X_val, y_val, feature names.
    """
    rng = np.random.default_rng(seed)
    age = rng.uniform(18, 80, n)
    income = np.exp(rng.normal(np.log(50_000), 0.7, n))
    late = rng.poisson(1.2, n)
    region = rng.integers(0, 6, n)
    noise = rng.normal(0, 1, n)
    bad = (late >= 4) | ((income < 30_000) & (age < 30)) | (np.isin(region, [1, 4]) & (late >= 2)) | ((income < 20_000) & (late >= 2))
    flip = rng.random(n) < 0.05
    y = (bad ^ flip).astype(int)
    X = np.c_[age, income, late, region, noise]
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:], list(TABULAR_FEATURES)


def permutation_importance(model, X: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Accuracy lost on held-out data when one column is shuffled, per feature.

    Shuffling breaks the link between that column and the label while
    keeping its values, so the drop measures how much the model relies on it
    for data it did not train on.
    """
    rng = np.random.default_rng(seed)
    base = _accuracy(model, X, y)
    drops = []
    for f in range(X.shape[1]):
        shuffled = X.copy()
        shuffled[:, f] = rng.permutation(shuffled[:, f])
        drops.append(base - _accuracy(model, shuffled, y))
    return np.array(drops)


def engineer_features(X: np.ndarray) -> np.ndarray:
    """What a neural network needs done by hand before it can read the loan table.

    Income becomes its logarithm, so a raise from 20k to 40k counts as much
    as one from 100k to 200k. The region code becomes six yes/no columns,
    because region 4 is not "twice region 2". Columns: age, log income, late
    payments, region_0..region_5, noise.
    """
    region = np.eye(6)[X[:, 3].astype(int)]
    return np.c_[X[:, 0], np.log(X[:, 1]), X[:, 2], region, X[:, 4]]


def _standardize(X: np.ndarray, X_ref: np.ndarray) -> np.ndarray:
    # Scale with the training set's statistics only: using validation data here would leak it.
    return (X - X_ref.mean(axis=0)) / X_ref.std(axis=0)


@functools.lru_cache(maxsize=4)
def _showdown(seed: int) -> tuple[tuple[str, float], ...]:
    from primer.ml.neural_net import MLP, train

    X, y, X_val, y_val, _ = make_tabular(seed=seed)
    scores = {
        "single tree (depth 5)": _accuracy(DecisionTree(max_depth=5).fit(X, y), X_val, y_val),
        "random forest": _accuracy(RandomForest(n_trees=50, seed=seed).fit(X, y), X_val, y_val),
        "gradient boosting": _accuracy(GradientBoosting(n_rounds=100, learning_rate=0.3, max_depth=3, loss="log").fit(X, y), X_val, y_val),
    }
    E, E_val = engineer_features(X), engineer_features(X_val)
    inputs = {
        "neural net (raw inputs)": (X, X_val),
        "neural net (scaled inputs)": (_standardize(X, X), _standardize(X_val, X)),
        "neural net (scaled + engineered)": (_standardize(E, E), _standardize(E_val, E)),
    }
    for label, (A, A_val) in inputs.items():
        net = MLP(n_in=A.shape[1], n_hidden=32, seed=seed)
        with np.errstate(over="ignore"):  # raw incomes of 50,000 overflow exp inside the sigmoid; that is the point
            train(net, A, y.astype(float), epochs=150, batch_size=32, lr=0.1, seed=seed)
            scores[label] = net.accuracy(A_val, y_val.astype(float))
    return tuple(scores.items())


def tabular_showdown(seed: int = 0) -> dict[str, float]:
    """Validation accuracy on `make_tabular` for a tree, two tree ensembles and a small MLP
    (`primer.ml.neural_net.MLP`) fed raw, scaled, and scaled plus engineered inputs.

    Trees get the table exactly as it is. Every model is trained once per
    seed and remembered, because several callers ask for the same scores.
    """
    return dict(_showdown(seed))



# ---------------------------------------------------------------------------
# 8. Figures (rendered into the HTML docs by `make figures`)
# ---------------------------------------------------------------------------


def figures() -> dict:
    """Plot this lesson's data. matplotlib is imported here, and only here,
    so the lesson itself needs nothing beyond NumPy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    BLUE, ORANGE, MUTED, DARK = "#2563eb", "#ea580c", "#9ca3af", "#4b5563"
    REGIONS = ListedColormap(["#dbeafe", "#ffedd5"])
    figs = {}
    X, y, X_val, y_val = moons_split()
    xx, yy = np.meshgrid(np.linspace(-1.8, 2.8, 220), np.linspace(-1.5, 1.9, 170))
    grid = np.c_[xx.ravel(), yy.ravel()]

    def regions(ax, model, title):
        ax.contourf(xx, yy, model.predict(grid).reshape(xx.shape), levels=[-0.5, 0.5, 1.5], cmap=REGIONS)
        ax.scatter(*X[y == 0].T, s=9, color=BLUE, label="class 0")
        ax.scatter(*X[y == 1].T, s=9, color=ORANGE, label="class 1")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    # --- 1. Every candidate question on the eight emails -------------------
    Xs, ys, names = spam_emails()
    cands = candidate_splits(Xs, ys)
    labels = [f"{names[c.feature]} > {c.threshold:g}" for c in cands]
    scores = [c.impurity for c in cands]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    colors = [BLUE if s == min(scores) else MUTED for s in scores]
    ax.barh(labels, scores, color=colors)
    ax.axvline(gini(ys), color=DARK, ls="--")
    ax.text(gini(ys) + 0.01, 0.5, "before any\nsplit: 0.5", va="center", color=DARK)
    for i, s in enumerate(scores):
        ax.text(s + 0.008, i, f"{s:.3f}", va="center")
    ax.invert_yaxis()
    ax.set_xlim(0, 0.65)
    ax.set_xlabel("weighted Gini impurity after the split (lower is better)")
    ax.set_title("Six questions the tree could ask first")
    figs["spam_splits"] = fig

    # --- 2. Axis-aligned decision regions as depth grows --------------------
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    for ax, depth in zip(axes, (1, 3, None)):
        tree = DecisionTree(max_depth=depth).fit(X, y)
        regions(ax, tree, f"depth {depth or 'unlimited'}: {tree.n_leaves} leaves")
    axes[0].legend(frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()
    figs["regions"] = fig

    # --- 3. Depth vs train/validation accuracy ------------------------------
    rows = depth_sweep()
    xs = [r["depth"] or 16 for r in rows]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(xs, [r["train_accuracy"] for r in rows], "o-", color=BLUE, label="training")
    ax.plot(xs, [r["val_accuracy"] for r in rows], "o-", color=ORANGE, label="validation")
    ax.set_xticks(xs, [str(r["depth"] or "none") for r in rows])
    ax.set_xlabel("depth limit")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.75, 1.01)
    ax.set_title("Deeper trees memorise: training climbs, validation falls")
    ax.legend(frameon=False)
    figs["depth_sweep"] = fig

    # --- 4. One deep tree vs a forest ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    single = DecisionTree().fit(X, y)
    forest = RandomForest(n_trees=100).fit(X, y)
    regions(axes[0], single, f"one deep tree: {_accuracy(single, X_val, y_val):.1%} on validation")
    regions(axes[1], forest, f"forest of 100: {_accuracy(forest, X_val, y_val):.1%} on validation")
    fig.tight_layout()
    figs["forest_regions"] = fig

    # --- 5. Validation accuracy vs number of trees --------------------------
    curve = forest_curve(n_trees=(1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100))
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot([r["n_trees"] for r in curve], [r["val_accuracy"] for r in curve], "o-", color=BLUE, label="random forest")
    ax.axhline(curve[0]["single_tree"], color=ORANGE, ls="--", label="one deep tree, all the data")
    ax.set_xscale("log")
    ax.set_xlabel("number of trees (log scale)")
    ax.set_ylabel("validation accuracy")
    ax.set_title("Adding trees helps, then levels off")
    ax.legend(frameon=False, loc="lower right")
    figs["forest_curve"] = fig

    # --- 6. Boosting builds a curve from steps ------------------------------
    Xr, yr = sine_data(80, seed=0)
    line = np.linspace(0, 6, 400)[:, None]
    model = GradientBoosting(n_rounds=50, learning_rate=0.3, max_depth=2).fit(Xr, yr)
    staged = model.staged_scores(line)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, m in zip(axes, (1, 5, 50)):
        ax.scatter(Xr[:, 0], yr, s=8, color=MUTED)
        ax.plot(line[:, 0], np.sin(line[:, 0]), color=DARK, lw=1, ls=":", label="true curve sin(x)")
        ax.plot(line[:, 0], staged[m], color=BLUE, lw=2, label="boosted trees")
        ax.set_title(f"after {m} round{'s' if m > 1 else ''}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend(frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()
    figs["boosting_steps"] = fig

    # --- 7. Training and validation loss per round, by learning rate --------
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for row, color in zip(boosting_curves(), (ORANGE, "#7c3aed", BLUE)):
        lr = row["learning_rate"]
        ax.plot(row["train_loss"], color=color, lw=1, ls="--")
        ax.plot(row["val_loss"], color=color, lw=2, label=f"η = {lr:g} (best validation at round {row['best_round']})")
        ax.plot(row["best_round"], min(row["val_loss"]), "o", color=color)
    ax.axhline(0.09, color=MUTED, lw=1)
    ax.text(200, 0.095, "noise floor 0.3² = 0.09", ha="right", va="bottom", color=DARK, fontsize=8)
    ax.set_ylim(0, 0.7)
    ax.set_xlabel("boosting round")
    ax.set_ylabel("mean squared error")
    ax.set_title("Solid: validation. Dashed: training.")
    ax.legend(frameon=False)
    figs["boosting_curves"] = fig

    # --- 8. The tabular showdown --------------------------------------------
    scores = tabular_showdown()
    _, _, _, y_tab, _ = make_tabular()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    names_ = list(scores)
    colors = [BLUE if not k.startswith("neural") else ORANGE for k in names_]
    ax.barh(names_, [scores[k] for k in names_], color=colors)
    majority = max(y_tab.mean(), 1 - y_tab.mean())
    ax.axvline(majority, color=DARK, ls="--")
    ax.text(majority + 0.004, -0.75, f"always say 'fine': {majority:.2%}", va="bottom", color=DARK, fontsize=8)
    for i, k in enumerate(names_):
        ax.text(scores[k] + 0.004, i, f"{scores[k]:.2%}", va="center")
    ax.set_ylim(len(names_) - 0.5, -1.1)
    ax.set_xlim(0.7, 1.0)
    ax.set_xlabel("validation accuracy on the loan table")
    ax.set_title("Trees read the table as it is; the net needs help")
    figs["showdown"] = fig

    # --- 9. Impurity importance vs permutation importance -------------------
    Xt, yt, Xtv, ytv, tab_names = make_tabular()
    forest = RandomForest(n_trees=30).fit(Xt, yt)
    imp, perm = forest.feature_importances, permutation_importance(forest, Xtv, ytv)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    a1.barh(tab_names, imp, color=BLUE)
    a2.barh(tab_names, perm, color=ORANGE)
    for a, vals, fmt in ((a1, imp, "{:.0%}"), (a2, perm, "{:.2%}")):
        for i, v in enumerate(vals):
            a.text(v + 0.003, i, fmt.format(v), va="center")
    a1.invert_yaxis()
    a1.set_title("Impurity importance (training data)")
    a1.set_xlabel("share of impurity removed")
    a2.set_title("Permutation importance (validation data)")
    a2.set_xlabel("accuracy lost when the column is shuffled")
    fig.tight_layout()
    figs["importance"] = fig

    return figs


# ---------------------------------------------------------------------------
# 9. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. Sorting eight emails: which question first?")
    X, y, names = spam_emails()
    say(
        f"""
        Eight emails, four spam and four not. Before any question the pile has
        Gini impurity {gini(y):.2f}: two labels drawn at random disagree half
        the time. The tree tries every question it could ask:
        """
    )
    table(["question", "weighted Gini after"], [(f"{names[c.feature]} > {c.threshold:g}", c.impurity) for c in candidate_splits(X, y)], floatfmt=".3f")
    takeaway("'Known sender?' leaves the least mess (0.2), so it becomes the first question.")

    banner("2. The grown tree, as rules a person can read")
    tree = DecisionTree(max_depth=2).fit(X, y)
    print("\n".join(tree.rules(names, ["ham", "spam"])))
    print()
    say("Two questions sort all eight emails. The whole model is those five lines.")

    banner("3. Overfitting: depth against training and validation accuracy")
    table(
        ["depth limit", "leaves", "train acc", "val acc"],
        [(r["depth"] or "none", r["leaves"], r["train_accuracy"], r["val_accuracy"]) for r in depth_sweep()],
        floatfmt=".3f",
    )
    takeaway("With no limit the tree scores 100% on training data and worst on validation: it memorised the noise.")

    banner("4. Random forests: bootstrap samples, random features, one vote")
    say(
        f"""
        A bootstrap sample of n rows leaves each row out with chance
        (1 - 1/n)^n: {out_of_bag_fraction(8):.4f} for n = 8, approaching 1/e =
        {1 / math.e:.4f}. Averaging B trees with correlation 0.3 leaves variance
        {averaged_variance(1.0, 0.3, 10):.2f} at B = 10 and
        {averaged_variance(1.0, 0.3, 1000):.4f} at B = 1000: the correlation is the floor.
        """
    )
    table(["trees", "val acc", "one deep tree"], [(r["n_trees"], r["val_accuracy"], r["single_tree"]) for r in forest_curve()], floatfmt=".3f")
    takeaway("Many deep, decorrelated trees averaged together keep their low bias and lose much of their variance.")

    banner("5. Gradient boosting by hand: four houses, one round")
    ex = boosting_worked_example()
    table(
        ["size", "price y", "F0", "residual", "stump h1", "F1 = F0 + 0.5·h1"],
        [(s, p, ex["F0"], r, h, f) for s, p, r, h, f in zip((1, 2, 3, 4), (1, 2, 6, 7), ex["residuals"], ex["stump"], ex["F1"])],
        floatfmt=".2f",
    )
    say(f"Mean squared error falls from {ex['mse'][0]} to {ex['mse'][1]} in one round.")
    for row in boosting_curves():
        say(
            f"""
            Learning rate {row['learning_rate']:g}: best validation MSE
            {min(row['val_loss']):.3f} at round {row['best_round']}; by round 200 it
            is {row['val_loss'][-1]:.3f} while training MSE is {row['train_loss'][-1]:.3f}.
            """
        )
    takeaway("Each tree fits what is still wrong. Smaller steps take longer and generalize better; too many rounds memorise.")

    banner("6. When trees win: a loan table")
    scores = tabular_showdown()
    table(["model", "val accuracy"], list(scores.items()), floatfmt=".4f")
    say(
        """
        The trees took the table as it came: dollars, years, counts and region
        codes. The neural net learned nothing from raw inputs (it matches
        always predicting 'fine'), improved once scaled, and improved again
        once income was logged and regions one-hot encoded, and still trailed.
        """
    )
    Xt, yt, Xv, yv, tab_names = make_tabular()
    forest = RandomForest(n_trees=30).fit(Xt, yt)
    table(
        ["feature", "impurity importance", "permutation importance"],
        list(zip(tab_names, forest.feature_importances, permutation_importance(forest, Xv, yv))),
        floatfmt=".3f",
    )
    takeaway(
        "Impurity importance gives the pure-noise column real credit; shuffling it on held-out data shows the model "
        "barely relies on it. Check importances on data the model has not seen."
    )


if __name__ == "__main__":
    demo()
