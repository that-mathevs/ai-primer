r"""
# Reasoning models: thinking before answering

Run: `python -m primer.ml.reasoning`

## The everyday picture

Ask someone to multiply 37 by 48 in their head, instantly, and they will
probably guess. Hand them a pencil and a minute and they will get it right.
They didn't get smarter in that minute. They got room: somewhere to write
37 × 8 = 296 and 37 × 40 = 1,480, so each small step could lean on the last.

A language model is in the same position. It can do a fixed amount of work
for each token it writes: one pass through all of its layers. A **reasoning
model** is a model that has learned to use the pencil. Before it answers, it
writes out intermediate steps, checks them, backs up when one is wrong, and
only then commits to an answer. This lesson builds each piece from scratch:

1. why writing steps helps at all (**chain of thought**);
2. why more thinking can buy more accuracy (**test-time compute**);
3. sampling several attempts and voting on the answer (**self-consistency**);
4. checking attempts, either the final answer or every step (**verifiers**);
5. how reinforcement learning teaches a model to think this way;
6. what thinking costs, and where it still fails.

## 1. Why thinking out loud helps

**Everyday picture.** Picture a clerk who, each time they glance at a page,
can do exactly one addition and write down one number. Hand them a column of
four numbers and allow one glance, and they have to guess. Allow three
glances and a margin to write in, and they get it right every time. The
margin is the whole trick: what they write on one glance is there to read on
the next.

**Tiny example.** Our toy model is that clerk. Every token it writes costs
one **forward pass** (one run of the whole model to produce one token), and
each forward pass can do one addition. Ask it for 3 + 5 + 8 + 2, which needs
three additions.

| Pass | Answer at once (1 token) | Think first, then answer |
|---|---|---|
| 1 | adds 3 + 5 = 8, has no pass left for 8 and 2, estimates them at 4.5 each: 8 + 9 = **17** ✗ | writes `3+5=8` |
| 2 | | reads 8, writes `8+8=16` |
| 3 | | reads 16, answers 16 + 2 = **18** ✓ |

The weights are identical in both columns. The only difference is that the
second column wrote its running totals down, where the next pass could read
them. Writing intermediate steps before the answer is called **chain of
thought**.

```mermaid
flowchart LR
  subgraph D["Answer at once: 1 token"]
    direction LR
    q1["3+5+8+2 ="] --> p1["pass 1<br/>3+5 = 8<br/>no pass left"] --> a1["17, a guess"]
  end
  subgraph C["Think first: 3 tokens"]
    direction LR
    q2["3+5+8+2 ="] --> s1["pass 1<br/>writes 3+5=8"] --> s2["pass 2<br/>reads 8<br/>writes 8+8=16"] --> s3["pass 3<br/>reads 16<br/>answers 18"]
  end
```

**Reading it:** each box is one forward pass, and each arrow is text handed
to the next pass. In the top row there is only one box, so all the work has
to fit in it, and it doesn't. In the bottom row every box does one small
step and leaves its result in the text. Nothing inside the model carries a
running total from one token to the next except what has been written, so
the written steps *are* the model's working memory.

The rule behind the table: the number of steps a model can do one after
another grows with the number of tokens it writes.

$$
S = L \times T
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $S$ | serial steps: how many steps the model can chain, each using the result of the one before | 1 or 3 |
| $L$ | serial steps one forward pass can do; for a transformer, roughly its number of layers | 1 addition in the toy |
| $T$ | tokens generated, the answer included; each token is one forward pass | 1 at once, 3 thinking first |
| $\times$ | multiply | |

**In words:** "the length of the chain of steps a model can work through
equals the steps per token times the number of tokens it writes."

**With the numbers:** the toy has $L = 1$ and the sum needs 3 additions.
Answering at once, $T = 1$, so $S = 1$: two additions short, so it guesses.
Thinking first, $T = 3$, so $S = 3$: exactly enough. A 32-layer model
answering at once has at most 32 layers of one-after-another work; with a
1,000-token chain of thought it has up to 32,000.

**In Python:**

```python
# the toy: 1 addition per pass; 3 + 5 + 8 + 2 needs 3 additions
L = 1
# answering at once: T = 1, so S = 1, not enough
L * 1  # → 1
# the one pass adds 3 + 5, then estimates the unreached 8 and 2 at 4.5 each
(3 + 5) + round(4.5 * 2)  # → 17
# thinking first: T = 3, so S = 3, exactly enough
L * 3  # → 3
t1 = 3 + 5  # → 8
t2 = t1 + 8  # → 16
t2 + 2  # → 18
# a 32-layer model: answering at once, then after a 1,000-token chain of thought
32 * 1  # → 32
32 * 1000  # → 32000
```

A layer is not literally one addition, so $L$ is a rough count, but the
shape of the rule holds: a model with a fixed depth can only do a fixed
amount of one-after-another work per token, and problems that need more
must spread it over more tokens. Li et al. (2024) proved a version of this
for transformers: with enough chain-of-thought tokens they can solve
inherently serial problems that a fixed-depth transformer answering at once
cannot.

![Answering in one token, the toy is right only on one-addition sums and about 5% of the rest; writing steps, it is always right, at one token per addition](figures/primer.ml.reasoning.direct_vs_cot.svg)

**Reading it:** on the left, the x-axis is how many additions a sum needs
and the y-axis is how often the toy gets it right. Answering in one token
(red), it is perfect on one-addition sums and then collapses to the few
percent of lucky estimates. Writing steps (blue), it is right every time. On
the right is the price: the blue line climbs one token per addition, while
the red line stays at one. Chain of thought doesn't make the model smarter
per token; it lets the model buy more tokens.

That price is real: each token costs roughly $2N$ floating-point operations
for a model with $N$ parameters (see `primer.ml.inference`), so a
1,000-token chain costs a thousand answers' worth of compute. In practice
this is why simply asking a model to "think step by step" (Kojima et al.,
2022) or showing it worked examples with steps (Wei et al., 2022) improved
accuracy on arithmetic and logic puzzles, and why every reasoning model
writes a long chain before its answer.

**In code:** `solve` is the toy model: it spends one forward pass per written step, one on the answer, and estimates anything it never reached. It returns an `Attempt` holding the steps, the answer and the tokens used.

## 2. Test-time compute: buying accuracy with tokens

**Everyday picture.** An exam has questions of very different difficulty.
Give yourself one minute per question and you finish only the easiest. Two
minutes, and the next tier falls. Each doubling of time unlocks one more
tier, until you can finish everything and extra time buys nothing.

**Tiny example.** **Test-time compute** is computation spent while
answering, as opposed to while training. The simplest dial is a **thinking
budget**: the most tokens the model may write before it must answer. Take
six problems that need 1, 2, 4, 8, 16 and 32 additions, and a budget of
8 tokens. Four of them fit (1, 2, 4, 8). The other two must be guessed, and
say a guess is right 10% of the time. Accuracy is 4/6 + 2/6 × 0.1 = **0.70**.

There are two ways to spend test-time compute, and the rest of this lesson
uses both.

```mermaid
flowchart LR
  P[Problem] --> LONG["Think longer<br/>one chain, bigger budget"]
  P --> WIDE["Think wider<br/>n chains side by side"]
  LONG --> A1[Answer]
  WIDE --> PICK["Pick one:<br/>vote or verifier"] --> A2[Answer]
```

**Reading it:** the top path spends compute **sequentially**: one chain,
allowed to run longer. It helps when the problem needs many steps in a row.
The bottom path spends it **in parallel**: several independent chains, then
a rule that picks one answer. It helps when the model is right sometimes
but not reliably. The top path costs waiting time; the bottom path costs
money but, run side by side, no extra waiting. Sections 3 and 4 are about
the "Pick one" box.

$$
\text{acc}(B) = F(B) + \big(1 - F(B)\big)\, g
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $B$ | the thinking budget, in tokens | 8 |
| $F(B)$ | the share of problems that need at most $B$ steps, so fit in the budget | 4/6 = 0.667 |
| $1 - F(B)$ | the share that doesn't fit and must be guessed | 2/6 = 0.333 |
| $g$ | the chance a guess happens to be right | 0.1 |
| $\text{acc}(B)$ | the share of problems answered correctly with budget $B$ | 0.70 |

**In words:** "accuracy is the share of problems that fit in the budget,
plus a lucky share of the ones that don't."

**With the numbers:** $F(8) = 4/6 = 0.667$, so
$\text{acc}(8) = 0.667 + 0.333 \times 0.1 = 0.70$. Doubling to $B = 16$ lets a
fifth level fit: $F = 5/6$, $\text{acc} = 0.85$. Every doubling adds the
same $1/6 \times 0.9 = 0.15$, until $B = 32$ fits everything and $\text{acc} = 1$.

**In Python:**

```python
needs = [1, 2, 4, 8, 16, 32]
g = 0.1
B = 8
# F(B): the share of problems that fit
F = sum(d <= B for d in needs) / len(needs)
round(F, 3)  # → 0.667
# acc(B) = F(B) + (1 − F(B)) · g
round(F + (1 - F) * g, 3)  # → 0.7
# each doubling of B lets one more of the six levels fit
round((1 / len(needs)) * (1 - g), 3)  # → 0.15
```

![Accuracy climbs 0.15 per doubling of the budget, from 0.25 at 1 token to 1.0 at 32, and the simulated sums land on the formula; tokens actually spent level off at about 10](figures/primer.ml.reasoning.budget_scaling.svg)

**Reading it:** on the left, the x-axis is the budget on a doubling (log)
scale. The grey line is the formula; the blue dots are the toy model from
section 1 solving real random sums. On a log axis a straight line means
"each doubling adds the same amount", and that is what both show until
$B = 32$, where every problem fits and the line goes flat. The dots sit a
little below the line at small budgets because a guess about many
unreached numbers is right less often than 10%. On the right is what was
actually spent: always less than the budget, because easy problems stop
early, and nothing more past 32.

The straight line on a log axis is built into this toy, because its
difficulties are spaced by doubling. Real problem sets are also spread over
many scales of difficulty, and published reasoning models show the same
shape over a useful range: accuracy rising roughly in proportion to the
logarithm of thinking tokens, then leveling off. Snell et al. (2024) found
that spending test-time compute adaptively (more on harder prompts) beats
spending it evenly, and that on problems a small model can sometimes solve,
extra test-time compute can stand in for a much larger model. In practice,
model APIs expose this dial as a "reasoning effort" setting or a maximum
number of thinking tokens.

**In code:** `budget_accuracy` is the formula, and `budget_sweep` runs `solve` on random sums at each budget and reports accuracy and tokens actually spent.

## 3. Sampling many and voting: self-consistency

**Everyday picture.** Unsure of an answer, you ask five friends separately.
If four of them say the same thing, you trust it. Each friend can be wrong,
but it is unlikely that most of them are wrong *in the same way*, unless
they all read the same wrong article.

**Tiny example.** A model that samples its tokens (see `primer.ml.inference`)
writes a different chain each time. Ask the toy's noisy cousin for
3 + 5 + 8 + 2 three times and it answers 18, 17, 18. Keep only the final
answers and take the most common one: 18. Sampling several chains of thought
and returning the most common final answer is **self-consistency** (Wang et
al., 2022), also called **majority voting**.

```mermaid
flowchart LR
  Q["3+5+8+2 = ?"] --> S1["chain 1 ends in 18"] & S2["chain 2 ends in 17"] & S3["chain 3 ends in 18"]
  S1 & S2 & S3 --> V["count final answers<br/>18: two votes, 17: one"] --> A["answer 18"]
```

**Reading it:** the question fans out into independent chains, each free to
take a different route. The chains themselves are thrown away; only their
last lines meet in the counting box. That is why voting needs answers that
can be compared exactly (a number, a multiple-choice letter): two essays are
never identical, so there would be nothing to count.

How much does voting help? Take the simplest case, a yes/no question: every
wrong vote lands on the same wrong answer, and each vote is right with the
same chance $p$, independently of the others.

$$
P_{\text{vote}}(n) = \sum_{k=\lceil n/2 \rceil}^{n} \binom{n}{k}\, p^{k}\, (1-p)^{n-k}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $n$ | number of sampled votes (odd, so there are no ties) | 3 |
| $p$ | chance one vote is right | 0.6 |
| $k$ | how many of the $n$ votes are right | 2 or 3 |
| $\lceil n/2 \rceil$ | $n/2$ rounded up: the smallest number of votes that wins | 2 |
| $\binom{n}{k}$ | "$n$ choose $k$": how many ways to pick which $k$ of the $n$ votes are the right ones | $\binom{3}{2} = 3$ |
| $p^{k}(1-p)^{n-k}$ | the chance of one particular pattern: those $k$ right, the other $n-k$ wrong | $0.6^2 \times 0.4 = 0.144$ |
| $\sum_{k=\lceil n/2 \rceil}^{n}$ | add up over every winning count of right votes | $k = 2, 3$ |
| $P_{\text{vote}}(n)$ | the chance the majority is right | 0.648 |

**In words:** "the vote is right when at least half the votes are right, so
add up the chance of every such count: the number of ways to get that many
right, times the chance of each way."

**With the numbers:** $p = 0.6$, $n = 3$. Two right: $3 \times 0.6^2 \times 0.4 =
0.432$. Three right: $0.6^3 = 0.216$. Together, **0.648**, up from 0.6 for
one vote. Five votes give 0.683. But a solver right only 40% of the time
gets *worse* with five votes: 0.317. Voting amplifies whichever answer is
most common, right or wrong. (This is the Condorcet jury theorem, from 1785.)

**In Python:**

```python
import math
p, n = 0.6, 3
# one term per winning count k = 2, 3
terms = [math.comb(n, k) * p**k * (1 - p)**(n - k) for k in range(math.ceil(n / 2), n + 1)]
[round(t, 3) for t in terms]  # → [0.432, 0.216]
round(sum(terms), 3)  # → 0.648
# five votes
round(sum(math.comb(5, k) * 0.6**k * 0.4**(5 - k) for k in range(3, 6)), 3)  # → 0.683
# a 40% solver on a yes/no question: voting makes it worse
round(sum(math.comb(5, k) * 0.4**k * 0.6**(5 - k) for k in range(3, 6)), 3)  # → 0.317
```

![Votes of an 80% or 60% solver climb towards 1, a 40% solver on a yes/no question sinks towards 0, but the same 40% solver climbs past 0.9 when its wrong answers scatter over ten values](figures/primer.ml.reasoning.voting.svg)

**Reading it:** the x-axis is how many samples vote; the y-axis is how often
the vote is right. The solid lines are the formula. Above 0.5 (green, blue)
more votes push accuracy towards 1; below 0.5 (solid red) they push it
towards 0. The dashed red line is the same 40% solver on a question with a
numeric answer, where its mistakes scatter over ten different wrong values.
No wrong value gets more than a few percent of the votes, so 40% is the
biggest pile and the vote climbs past 0.9. That is the situation
self-consistency relies on in math problems: the right answer only needs to
be the most common one, not a majority.

**When voting fails: shared mistakes.** Everything above assumed the votes
were independent. Samples from one model are not: they share its training,
its blind spots and its reading of the question. Model that as a shared
draw: each sample copies one common answer with probability $\rho$ (rho),
and otherwise answers on its own. Each single sample is still right 60% of
the time, so only the correlation changes.

![With independent samples 31 votes reach nearly 1.0; with rho = 0.3 the vote levels off near 0.86; with rho = 0.6 or 0.9 it stays at about 0.6, no better than one sample](figures/primer.ml.reasoning.correlated_votes.svg)

**Reading it:** every line starts at 0.6 on the left, because one sample is
one sample. Independent samples (blue) climb to nearly 1. At $\rho = 0.6$
(amber) and $\rho = 0.9$ (red) the lines go flat at 0.6: whenever the shared
answer is wrong, the 60% or 90% of votes that copy it outnumber the at most
40% × 0.6 = 24% (or 6%) that can independently land on the right answer.
Thirty-one correlated votes are one opinion, repeated. At $\rho = 0.3$
(green) the right answer's share, 0.7 × 0.6 = 42%, still beats the shared
wrong answer's 0.3 + 0.7 × 0.4 / 5 = 35.6%, so the vote does win in the
long run, just slowly.

In practice, self-consistency gives a solid gain for a few samples and then
flattens, because a model's mistakes on a question are correlated. Diversity
helps (different prompts, different models, a tool that computes instead of
guessing), and voting only works where answers can be compared exactly.

**In code:** `majority_vote` picks the most common answer, `majority_accuracy` is the formula (with ties on even $n$ counted as a coin flip), and `correlated_vote_accuracy` simulates votes with scattered wrong answers and a shared-mistake rate.

## 4. Verifiers: checking the answer, or checking every step

**Everyday picture.** One teacher looks only at the boxed answer at the
bottom of the page. Another marks every line of working. The first can't
tell a lucky guess from understanding, and when the answer is wrong, can't
tell you where you went wrong. The second can do both.

**Tiny example.** A **verifier** is anything that scores a candidate
solution. Here are two chains the toy wrote for 3 + 5 + 8 + 2 (true answer
18), each with slips:

| Chain | Final answer | Outcome check: is it 18? | Step check: first line that is false |
|---|---|---|---|
| `3+5=8`, `8+8=17`, `17+2=19` | 19 | reward 0 | step 2: 8 + 8 is 16 |
| `3+5=9`, `9+8=16`, `16+2=18` | 18 | reward 1 | step 1: 3 + 5 is 8 |

An **outcome verifier** looks only at the final answer: either a learned
**outcome reward model (ORM)** that predicts whether it is right, or a check
against a known answer. It gives the second chain full marks, although two
slips just happened to cancel. A **process verifier**, or **process reward
model (PRM)**, scores every step. It catches the second chain's slip and
points at exactly where the first one went wrong.

```mermaid
flowchart LR
  subgraph O["Outcome verifier"]
    direction LR
    o1["3+5=8"] --> o2["8+8=17"] --> o3["17+2=19"] --> oc{"final 19<br/>equals 18?"} --> orr["reward 0<br/>but where did it go wrong?"]
  end
  subgraph P["Process verifier"]
    direction LR
    p1["3+5=8<br/>ok"] --> p2["8+8=17<br/>wrong"] --> p3["17+2=19<br/>ok"] --> pr["first bad step: 2"]
  end
```

**Reading it:** both rows read the same chain. The outcome verifier jumps
straight to the diamond at the end and returns one bit. The process
verifier stamps every box. Notice that step 3, 17 + 2 = 19, is marked ok: it
is correct arithmetic on a wrong input. A step checker judges each step on
its own terms, and the first bad step is where the chain left the rails.

Verifiers power **best-of-n**: sample $n$ chains, keep the one the verifier
scores highest. With a perfect verifier, best-of-n is right whenever *any*
of the $n$ samples is right, a number called **pass@n**.

$$
\text{pass@}n = 1 - (1 - p)^{n}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $p$ | chance one sample is right | 0.3 |
| $n$ | number of samples drawn | 5 |
| $1 - p$ | chance one sample is wrong | 0.7 |
| $(1 - p)^{n}$ | chance all $n$ are wrong, multiplying because samples are independent | $0.7^5 = 0.168$ |
| $\text{pass@}n$ | chance at least one of the $n$ is right | 0.832 |

**In words:** "the chance that at least one sample is right is one minus the
chance that every sample is wrong."

**With the numbers:** a model right 30% of the time is wrong on five tries
in a row with chance $0.7^5 = 0.168$, so pass@5 = **0.832**. A 30% model with
a perfect checker and five tries beats an 80% model on one try. In the
experiment below, each chain has 5 additions that each slip with chance 0.2,
so a chain is clean with chance $0.8^5 = 0.33$.

**In Python:**

```python
p, n = 0.3, 5
# (1 − p)^n: every one of the five is wrong
round((1 - p) ** n, 5)  # → 0.16807
round(1 - (1 - p) ** n, 5)  # → 0.83193
# the experiment's chains: 5 additions, each clean with chance 0.8
round(0.8 ** 5, 3)  # → 0.328
```

![With 8 chains, one sample is right 38% of the time, a majority vote 67%, best-of-8 with a step checker 96%, against a pass@8 ceiling of 99%](figures/primer.ml.reasoning.verifier.svg)

**Reading it:** the x-axis doubles the number of chains sampled; the y-axis
is accuracy. One sample (red) stays at 0.38 whatever $n$ is: about a third
of chains are clean, and a few more land on 18 by luck. Voting (blue) climbs
slowly, because wrong answers bunch on near misses one or two away from
the truth. Best-of-n with the step
checker (green) hugs the dotted pass@n ceiling: at 8 chains, 0.96 against
0.99. The small gap is the lucky chains: their slips cancelled, so pass@n
counts them as right, but the checker refuses them. That is the checker
doing its job.

In practice, the best verifiers are programs: unit tests for code, an exact
match for a math answer, a proof checker. Where no program can check, a
learned verifier stands in. Lightman et al. (2023) trained a process reward
model on 800,000 human labels of individual steps and found it picked
correct solutions far more reliably than an outcome reward model. A learned
verifier can be fooled, though, and more samples mean more chances to find
a wrong answer it likes: Cobbe et al. (2021) saw best-of-n accuracy start to
fall again after a few hundred samples.

**In code:** `check_step` is the toy's process verifier for one line, `first_bad_step` runs it over a chain, and `outcome_reward` compares only the final answer with a reference. `noisy_chain` writes chains whose slips carry forward, `pass_at_n` is the formula, and `verifier_experiment` compares one sample, voting, best-of-n with the step checker, and the pass@n ceiling.

## 5. Learning to reason with reinforcement learning

**Everyday picture.** A workbook with the answers printed in the back.
Nobody shows you how to solve anything. You try each problem several ways,
check the back, and do more of whatever worked. Over hundreds of problems
you discover good habits on your own: write things down, double-check the
tricky step, don't stop too early.

**Tiny example.** Prompting a model to "think step by step" gets it to
write steps; a reasoning model is *trained* to write good ones. The recipe
behind models such as DeepSeek-R1 is reinforcement learning (see
`primer.ml.reinforcement`) with two ingredients:

- **Verifiable rewards.** Train on problems whose answers a program can
  check: math with a known final answer, code with unit tests. The reward is
  1 if the final answer is right and 0 if not. Nobody grades the chain
  itself.
- **Group comparisons.** For each problem, sample a group of $G$ attempts
  and judge each one against its own group. With $G = 4$ and rewards
  1, 0, 0, 1, the group averages 0.5, so the two right attempts are above
  average (+1) and the two wrong ones below (−1). The model is nudged
  towards whatever the right ones did, including how they reasoned.
  This is the heart of **GRPO** (group relative policy optimization).

```mermaid
flowchart LR
  P["problem with a checkable answer"] --> G["sample G attempts,<br/>each with its own chain of thought"]
  G --> R["check each final answer:<br/>reward 1 or 0"]
  R --> A["advantage: reward minus group mean,<br/>divided by group spread"]
  A --> U["make above-average attempts more likely,<br/>below-average ones less"]
  U -->|next batch| P
```

**Reading it:** follow the loop clockwise. Nothing in it ever says "think
longer" or "check your work"; the only signal is whether the last line was
right. Whatever habits the chains of right attempts share get reinforced,
batch after batch. The group is what makes this work without a separate
model estimating how good an attempt "should" be: the other attempts at the
same problem are the baseline.

$$
A_i = \frac{r_i - \operatorname{mean}(r_1, \ldots, r_G)}{\operatorname{std}(r_1, \ldots, r_G)}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $G$ | how many attempts are sampled for one problem | 4 |
| $i$ | which attempt, from 1 to $G$ | 1, 2, 3, 4 |
| $r_i$ | attempt $i$'s reward: 1 if its final answer is right, 0 if not | 1, 0, 0, 1 |
| $\operatorname{mean}(\ldots)$ | the average reward in the group | 0.5 |
| $\operatorname{std}(\ldots)$ | the **standard deviation**: the typical distance of a reward from the mean (see `primer.notation`) | 0.5 |
| $A_i$ | attempt $i$'s **advantage**: how much better than its group it did, in units of the group's spread | +1, −1, −1, +1 |

**In words:** "an attempt's advantage is how far its reward sits above its
group's average, measured in units of how spread out the group's rewards
are."

**With the numbers:** rewards 1, 0, 0, 1 have mean 0.5 and standard
deviation 0.5, so the advantages are (1 − 0.5)/0.5 = +1 and
(0 − 0.5)/0.5 = −1. If all four attempts are right, every reward equals the
mean and the spread is 0: every advantage is 0, and the problem teaches
nothing. The same holds when all four are wrong. Training therefore needs
problems the model solves *sometimes*.

**In Python:**

```python
import statistics
r = [1, 0, 0, 1]
mu = statistics.mean(r)  # → 0.5
sigma = statistics.pstdev(r)  # → 0.5
[(r_i - mu) / sigma for r_i in r]  # → [1.0, -1.0, -1.0, 1.0]
# a group that all succeeded: spread 0, nothing to learn
statistics.pstdev([1, 1, 1, 1])  # → 0.0
```

**The toy: learning how long to think.** Our policy (the model's rule for
choosing what to do) makes one choice per problem: how many tokens to think
for, out of 1, 2, 4, 8, 16 or 32. It keeps a separate choice for each
difficulty (problems needing 1, 2, 4 or 8 steps), and it starts out
preferring short answers, like a model never rewarded for thinking: 63% of
the time it answers in 1 token. A chain shorter than the steps needed must
guess (right 10% of the time). Otherwise each step gets as many tries as the
length allows, and a try slips 10% of the time: one pass through the steps,
or two (the first attempt and a recheck that catches a slip), or more. The
reward is 1 for a right answer and 0 for a wrong one. That is all.

![Rewarded only for correct answers, mean thinking length rises from 2 to about 7.6 tokens and accuracy from 0.41 to 0.96; with a length penalty the length and accuracy settle a little lower](figures/primer.ml.reasoning.rl_training.svg)

**Reading it:** the x-axis is training time. On the left, the blue line
(reward for correctness only) shows the average thinking length rising from
2 tokens to about 7.6; on the right, accuracy rising from 0.41 to 0.96.
Nobody asked for longer answers: longer answers were simply right more
often, so they were reinforced. This is the toy version of what DeepSeek-R1
reported at scale: the length of its chains grew steadily through
reinforcement learning, and behaviours like re-checking and backing up
appeared without being taught. The red and green lines add a price per
token; they come next.

![Trained on correctness alone, the model spends about 2.5, 4, 8 and 16 tokens on problems needing 1, 2, 4 and 8 steps: at or just past twice the steps needed](figures/primer.ml.reasoning.rl_lengths.svg)

**Reading it:** each group of bars is one difficulty; bar height is how many
tokens the trained model spends on it. The black tick is the steps the
problem needs, the grey tick twice that. The blue bars (correctness only)
sit on the grey ticks: the model learned to spend *more on harder
problems*, and to leave room for one recheck of every step, which lifts the
hardest problems from 0.43 to 0.92. That extra room is self-correction in
miniature: the chain gets longer because checking pays.

**In code:** `group_advantages` is the formula, `success_probability` is the toy's chance of solving a problem at a given length, and `train_reasoner` runs the whole loop: sample a group of lengths per difficulty, reward, compute advantages, and nudge the policy.

### The price of thinking

Rewarded for correctness alone, nothing ever tells the model to *stop*:
any extra length that helps even slightly gets reinforced. Real reasoning
models show this as **overthinking**: hundreds of tokens spent on "what is
2 + 3?". The usual remedy is to charge for length: subtract a small penalty
per token from the reward. The expected reward for a chain of length $L$ on
a problem needing $d$ steps (with $L \ge d$) becomes:

$$
\mathbb{E}[r] = \left(1 - \varepsilon^{\lfloor L/d \rfloor}\right)^{d} - \lambda L
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $\mathbb{E}[r]$ | the **expected** (average) reward for this length | 0.763 |
| $d$ | steps the problem needs | 8 |
| $L$ | thinking length, in tokens | 16 |
| $\lfloor L/d \rfloor$ | $L/d$ rounded down: how many tries each step gets | 2 |
| $\varepsilon$ | chance one try at a step slips | 0.1 |
| $\varepsilon^{\lfloor L/d \rfloor}$ | chance *every* try at one step slips, so the step fails | 0.01 |
| $\left(1 - \ldots\right)^{d}$ | chance all $d$ steps come out right | 0.923 |
| $\lambda$ | lambda: the penalty per token of thinking | 0.01 |
| $\lambda L$ | the total charge for the chain | 0.16 |

**In words:** "the reward for a length is how likely it is to get every step
right, given the tries it allows, minus a small charge for every token."

**With the numbers:** for an 8-step problem, 8 tokens give
$0.9^8 - 0.08 = 0.43 - 0.08 = 0.35$; 16 tokens give
$0.99^8 - 0.16 = 0.923 - 0.16 = 0.763$; 32 tokens give
$0.999 - 0.32 = 0.679$. Sixteen wins: one recheck is worth paying for, a
third is not. For a 1-step problem, 1 token gives $0.9 - 0.01 = 0.89$, 2
tokens $0.99 - 0.02 = 0.97$, 4 tokens $0.96$. Two wins.

**In Python:**

```python
eps, lam = 0.1, 0.01
# an 8-step problem: chance of success at 8, 16 and 32 tokens
[round((1 - eps ** (L // 8)) ** 8, 3) for L in (8, 16, 32)]  # → [0.43, 0.923, 0.999]
# minus λL: 16 tokens wins
[round((1 - eps ** (L // 8)) ** 8 - lam * L, 3) for L in (8, 16, 32)]  # → [0.35, 0.763, 0.679]
# a 1-step problem at 1, 2 and 4 tokens: 2 tokens wins
[round((1 - eps ** (L // 1)) ** 1 - lam * L, 3) for L in (1, 2, 4)]  # → [0.89, 0.97, 0.96]
```

Now look back at the two figures. With the penalty and no division by the
spread (green), training finds exactly those best lengths, 2 tokens for the
easiest problems and 16 for the hardest. With GRPO's division by the spread
(red), the easiest problems shrink to **1** token, and accuracy settles at
0.92, below the 0.96 of correctness alone. Why? In a group where every attempt succeeded,
rewards differ only by the penalty: 0.99 for 1 token, 0.98 for 2. Their
spread is tiny, so dividing by it inflates that 0.01 difference into
advantages of +1 and −1, as loud as the difference between solved and
failed. The penalty ends up far stronger than its size. Liu et al. (2025)
analyse biases like this in GRPO; the lesson for anyone training or
budgeting a reasoning model is that the exact shape of the reward decides
how long the model thinks.

## 6. Where reasoning still fails, and how to budget it

**Everyday picture.** A long line of dominoes falls all the way only if
every single one is placed right. Add more dominoes and the chance that one
is misplaced grows, however careful you are with each.

### Slips compound

**Tiny example.** If each step of a chain slips 2% of the time, a 10-step
chain is clean 82% of the time, and a 50-step chain only 36% of the time.

$$
P(\text{no slip}) = (1 - \varepsilon)^{k}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $\varepsilon$ | chance one step slips | 0.02 |
| $1 - \varepsilon$ | chance one step is right | 0.98 |
| $k$ | steps in the chain | 50 |
| $(1-\varepsilon)^{k}$ | chance all $k$ are right, multiplying because each step is a separate chance to slip | 0.364 |

**In words:** "the chance a whole chain is clean is the chance one step is
right, multiplied by itself once per step."

**With the numbers:** $0.98^{10} = 0.817$, $0.98^{50} = 0.364$,
$0.98^{200} = 0.018$.

**In Python:**

```python
eps = 0.02
round((1 - eps) ** 10, 3)  # → 0.817
round((1 - eps) ** 50, 3)  # → 0.364
round((1 - eps) ** 200, 3)  # → 0.018
```

![At a 2% slip rate the chance of a clean chain falls to 0.36 by 50 steps and near zero by 200; at 0.5% it falls much more slowly](figures/primer.ml.reasoning.compounding.svg)

**Reading it:** the x-axis is the length of the chain; the y-axis is the
chance it contains no slip at all. Every curve falls, and the only thing
that flattens one is a lower slip rate per step. That is why length alone
is not reasoning: the trained model in section 5 got better by spending
its extra tokens on *rechecking*, which lowers the effective slip rate, and
why verifiers that check each step matter. The same law governs agents
taking many actions; see `primer.agents.planning`.

**In code:** `steps_all_right` is the formula.

### What thinking costs

**Tiny example.** Eight sampled chains of 2,000 tokens each, at an
illustrative \$10 per million output tokens, cost 16 cents per question. Run
side by side at 50 tokens per second, the user waits 40 seconds whether you
sample one chain or eight.

$$
\text{dollars} = \frac{n \cdot T \cdot c}{10^{6}}, \qquad \text{seconds} = \frac{T}{v}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $n$ | chains sampled for one question | 8 |
| $T$ | tokens in each chain | 2,000 |
| $c$ | price per million output tokens | \$10 |
| $10^{6}$ | one million: turns a per-million price into a per-token one | |
| $v$ | generation speed, tokens per second | 50 |

**In words:** "money grows with every token of every chain; waiting time,
with chains run side by side, grows only with the length of one chain."

**With the numbers:** 8 × 2,000 × 10 / 1,000,000 = **\$0.16**, and
2,000 / 50 = **40 seconds**. One chain instead of eight costs \$0.02 and
still takes 40 seconds.

**In Python:**

```python
n, T, c, v = 8, 2000, 10.0, 50
round(n * T * c / 10**6, 2)  # → 0.16
T / v  # → 40.0
# one chain instead of eight: an eighth of the money, the same wait
round(1 * T * c / 10**6, 2)  # → 0.02
```

Thinking wider costs money; thinking longer costs money *and* time. See
`primer.agents.cost` for pricing requests and measuring cost per successful
task.

**In code:** `reasoning_cost` returns both numbers for one question.

### Other ways reasoning fails

- **The chain is not a transcript.** The written reasoning need not be the
  real cause of the answer. Turpin et al. (2023) nudged models towards an
  answer with a hidden bias in the prompt; the answers followed the bias,
  and the written explanations never mentioned it. Treat a chain of thought
  as evidence about the model's reasoning, not a faithful record of it.
- **Overthinking.** Long chains on easy questions waste tokens and time,
  and a model can talk itself out of a right first answer.
- **No checker, less progress.** Reinforcement learning with verifiable
  rewards works where a program can check the answer. Open-ended writing,
  judgement calls and long projects have no cheap checker, and gains there
  are smaller.
- **Checkers get gamed.** A verifier with gaps is a target: code that
  special-cases the unit tests passes them. This is reward hacking; see
  `primer.ml.reinforcement`.

### Budgeting it

**Everyday picture.** You wouldn't convene a committee to decide what to
have for lunch, and you wouldn't let one person sign off a bridge design
alone. Match the effort to the stakes and to whether the result can be
checked.

```mermaid
flowchart TD
  Q[Incoming task] --> E{"Easy, or<br/>latency-critical?"}
  E -->|yes| N["No thinking,<br/>or a small budget"]
  E -->|no| C{"Can a program<br/>check the answer?"}
  C -->|"yes: tests, math"| BV["Think, sample n,<br/>keep what passes the check"]
  C -->|no| B["Think with a capped budget;<br/>vote if answers compare exactly"]
  N & BV & B --> M["Measure accuracy and cost per task;<br/>raise budgets only where it pays"]
```

**Reading it:** start at the top with each incoming task. The first
question routes easy or time-critical work away from thinking entirely,
because that is where overthinking wastes the most. The second asks whether
a program can check the answer: if so, sampling several chains and keeping
the one that passes turns spare compute into accuracy almost for free
(section 4). If not, cap the budget and vote only when answers can be
compared. Everything ends in the same box: measure, because the right
budget is an empirical question per task, not a constant. See
`primer.agents.planning` for decomposing long tasks into checkable steps.

## In 20 seconds

- **Chain of thought:** every written token is another forward pass, so
  writing steps gives a fixed-depth model more serial computation, and the
  text is its working memory.
- **Test-time compute:** spend more tokens at answer time, either one
  longer chain or many chains; accuracy often grows with the logarithm of
  the budget until the problems run out.
- **Self-consistency:** sample several chains and vote on the final
  answer; it helps when the right answer is the most common one and mistakes
  are independent, and stalls when they are shared.
- **Verifiers:** outcome checks score the answer, process checks score each
  step; best-of-n with a good checker approaches pass@n = 1 − (1 − p)ⁿ.
- **RL with verifiable rewards:** reward right final answers, compare each
  attempt with its group (GRPO), and longer, self-checking reasoning emerges
  because it pays; a length penalty keeps it from overthinking.
- **Cost and limits:** thinking is paid per token, slips compound over long
  chains, and the written chain is not a guaranteed account of why the
  model answered.

## Self-test questions

**How can writing its reasoning out make a model more accurate, when its
weights don't change?**
Each token is one forward pass with a fixed amount of serial computation.
A problem that needs more sequential steps than one pass can do can't be
solved in a single token. Writing intermediate results spreads the work over
many passes, and the written text carries each result to the next pass: it
is the model's working memory.

**What is test-time compute, and what are the two basic ways to spend it?**
Computation spent while answering rather than while training. Think longer
(one chain with a bigger thinking budget, which costs time and money) or
think wider (many chains in parallel, then vote or verify, which costs money
but no extra waiting when run side by side).

**Why does majority voting over samples help, and when does it stop
helping?**
If each sample is independently right more often than any single wrong
answer appears, the right answer becomes the biggest pile as samples grow.
It stops helping when samples share the same mistakes (they are one opinion
repeated), when the model is usually wrong in one consistent way, or when
answers can't be compared exactly.

**What is the difference between an outcome reward model and a process
reward model?**
An outcome reward model scores only the final answer, so it can't tell a
lucky answer from a sound one and can't say where a wrong chain went wrong.
A process reward model scores each step, catching errors that cancel out and
pointing at the first bad step.

**What is pass@n, and why is it a ceiling for best-of-n?**
The chance that at least one of n samples is right, 1 − (1 − p)ⁿ. A
best-of-n picker can only choose among the samples it has, so even a
perfect verifier can't do better than "some sample was right".

**How does reinforcement learning with verifiable rewards teach a model to
reason, if nobody grades the reasoning?**
Each problem gets a group of sampled attempts, each rewarded 1 or 0 by a
program that checks the final answer. Attempts better than their group's
average are made more likely, worse ones less likely. Whatever the right
attempts had in common, including longer chains and rechecking, is
reinforced. A group where every attempt scores the same teaches nothing, so
the useful problems are ones the model solves only sometimes.

**Why do reasoning models sometimes spend thousands of tokens on easy
questions, and what reins that in?**
A reward for correctness alone never says stop: any extra length that helps
slightly is reinforced. A per-token penalty in training, a thinking budget
at answer time, and routing easy tasks to little or no thinking all rein it
in. The exact shape of the reward matters: dividing by the group's spread,
as GRPO does, can magnify a tiny length penalty.

**Is a model's chain of thought an accurate account of why it answered?**
Not necessarily. Experiments that plant a hidden bias in a prompt show
answers following the bias while the written reasoning never mentions it.
The chain is useful evidence and often helpful, but it is not a guaranteed
record of the computation.

## The papers behind this lesson

- **Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large
  Language Models* (2022)**: https://arxiv.org/abs/2201.11903. Showed that
  prompting with a few worked examples that include intermediate steps makes
  large models far better at arithmetic, commonsense and symbolic
  reasoning.
- **Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in
  Language Models* (2022)**: https://arxiv.org/abs/2203.11171. Introduced
  sampling many chains of thought and taking a majority vote over their
  final answers.
- **Cobbe et al., *Training Verifiers to Solve Math Word Problems*
  (2021)**: https://arxiv.org/abs/2110.14168. Introduced the GSM8K dataset
  and showed that a trained verifier picking the best of many sampled
  solutions beats fine-tuning alone.
- **Lightman et al., *Let's Verify Step by Step* (2023)**:
  https://arxiv.org/abs/2305.20050. Showed that process supervision, a
  reward model trained on labels for each step, selects correct solutions
  more reliably than outcome supervision.
- **Snell et al., *Scaling LLM Test-Time Compute Optimally can be More
  Effective than Scaling Model Parameters* (2024)**:
  https://arxiv.org/abs/2408.03314. Measured how to spend test-time compute
  (longer revisions or verifier-guided search) and showed that adapting it
  to each prompt's difficulty pays.
- **Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical
  Reasoning in Open Language Models* (2024)**:
  https://arxiv.org/abs/2402.03300. Introduced GRPO, reinforcement
  learning that uses a group of sampled answers as its own baseline.
- **DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
  via Reinforcement Learning* (2025)**: https://arxiv.org/abs/2501.12948.
  Showed that reinforcement learning with verifiable rewards alone makes
  long, self-checking chains of thought emerge.

## Further reading

- Wei et al., *Chain-of-Thought Prompting* (2022): https://arxiv.org/abs/2201.11903
- Kojima et al., *Large Language Models are Zero-Shot Reasoners* ("let's think step by step", 2022): https://arxiv.org/abs/2205.11916
- Nye et al., *Show Your Work: Scratchpads for Intermediate Computation with Language Models* (2021): https://arxiv.org/abs/2112.00114
- Li et al., *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems* (2024): https://arxiv.org/abs/2402.12875
- Wang et al., *Self-Consistency* (2022): https://arxiv.org/abs/2203.11171
- Brown et al., *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling* (2024): https://arxiv.org/abs/2407.21787
- Cobbe et al., *Training Verifiers to Solve Math Word Problems* (2021): https://arxiv.org/abs/2110.14168
- Uesato et al., *Solving math word problems with process- and outcome-based feedback* (2022): https://arxiv.org/abs/2211.14275
- Lightman et al., *Let's Verify Step by Step* (2023): https://arxiv.org/abs/2305.20050
- Snell et al., *Scaling LLM Test-Time Compute Optimally* (2024): https://arxiv.org/abs/2408.03314
- Muennighoff et al., *s1: Simple test-time scaling* (2025): https://arxiv.org/abs/2501.19393
- Zelikman et al., *STaR: Bootstrapping Reasoning With Reasoning* (2022): https://arxiv.org/abs/2203.14465
- Shao et al., *DeepSeekMath* (GRPO, 2024): https://arxiv.org/abs/2402.03300
- DeepSeek-AI, *DeepSeek-R1* (2025): https://arxiv.org/abs/2501.12948
- Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective* (2025): https://arxiv.org/abs/2503.20783
- Chen et al., *Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs* (2024): https://arxiv.org/abs/2412.21187
- Turpin et al., *Language Models Don't Always Say What They Think* (2023): https://arxiv.org/abs/2305.04388
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from primer._show import banner, say, table, takeaway

# ---------------------------------------------------------------------------
# 1. Thinking out loud: a toy model that does one addition per forward pass
# ---------------------------------------------------------------------------

# The toy model's rough sense of "a digit I didn't have time to add": the average of 0..9.
DIGIT_MEAN = 4.5


@dataclass
class Attempt:
    """What the toy model wrote for one problem.

    steps:  the intermediate lines it wrote before answering, e.g. ["3+5=8", "8+8=16"]
    answer: the number it finally gave
    tokens: forward passes used, one per written step plus one for the answer
    """

    steps: list[str] = field(default_factory=list)
    answer: int = 0
    tokens: int = 1


def solve(numbers: list[int], max_tokens: int) -> Attempt:
    """Add up `numbers` with a model that can do ONE addition per forward pass.

    Every token the model writes is one forward pass. A step token ("8+8=16")
    spends its pass on one addition and leaves the running total in the text,
    where the next pass can read it. The answer token also gets one addition.
    If the budget runs out before the sum is done, the model estimates the
    numbers it never reached at `DIGIT_MEAN` each: a guess, right only by luck.
    """
    total = numbers[0]
    if len(numbers) == 1:
        return Attempt([], total, 1)
    steps: list[str] = []
    i = 1
    # Thinking tokens: all but the last addition, as long as the budget leaves a token for the answer.
    while i < len(numbers) - 1 and len(steps) < max_tokens - 1:
        new = total + numbers[i]
        steps.append(f"{total}+{numbers[i]}={new}")
        total, i = new, i + 1
    # The answer token's own forward pass: one more addition.
    total += numbers[i]
    unreached = len(numbers) - (i + 1)
    # Round half up, so one unreached number is estimated as 5 (4.5 rounded).
    answer = total + math.floor(DIGIT_MEAN * unreached + 0.5)
    return Attempt(steps, int(answer), len(steps) + 1)


# ---------------------------------------------------------------------------
# 2. Test-time compute: accuracy as a function of the thinking budget
# ---------------------------------------------------------------------------

# Problems need 1, 2, 4, ..., 32 additions: difficulty spaced by doubling, as real benchmarks roughly are.
NEEDS = (1, 2, 4, 8, 16, 32)


def budget_accuracy(budget: int, needs: tuple[int, ...] = NEEDS, lucky: float = 0.1) -> float:
    """acc(B) = F(B) + (1 - F(B)) * g.

    F(B) is the share of problems that fit in a budget of B tokens (need at
    most B additions); the rest are guessed, and a guess is right with
    probability g (`lucky`).
    """
    fits = sum(d <= budget for d in needs) / len(needs)
    return fits + (1 - fits) * lucky


def budget_sweep(budgets=(1, 2, 4, 8, 16, 32, 64), n_problems: int = 600, seed: int = 0) -> list[dict]:
    """Run `solve` on real random sums at each budget: accuracy and tokens actually spent.

    Each problem needs d additions, d drawn evenly from `NEEDS`, so it has d + 1
    random digits. The same problems are reused at every budget.
    """
    rng = np.random.default_rng(seed)
    problems = [rng.integers(0, 10, size=int(rng.choice(NEEDS)) + 1).tolist() for _ in range(n_problems)]
    rows = []
    for b in budgets:
        attempts = [solve(p, b) for p in problems]
        right = [a.answer == sum(p) for a, p in zip(attempts, problems)]
        rows.append(dict(budget=b, accuracy=float(np.mean(right)), mean_tokens=float(np.mean([a.tokens for a in attempts]))))
    return rows


# ---------------------------------------------------------------------------
# 3. Sampling many and voting: self-consistency
# ---------------------------------------------------------------------------


def majority_vote(answers: list) -> object:
    """The most common answer. Ties go to the answer seen first."""
    return Counter(answers).most_common(1)[0][0]


def majority_accuracy(p: float, n: int) -> float:
    """Chance that more than half of n independent votes are right, each right with probability p.

    A yes/no question: every wrong vote lands on the same wrong answer. With
    an even n, a tie is broken by a coin flip, so it counts half.
    """
    win = sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n // 2 + 1, n + 1))
    tie = math.comb(n, n // 2) * p ** (n // 2) * (1 - p) ** (n // 2) if n % 2 == 0 else 0.0
    return win + tie / 2


def correlated_vote_accuracy(
    p: float, n: int, rho: float = 0.0, n_wrong: int = 5, trials: int = 2000, seed: int = 0
) -> float:
    """Plurality-vote accuracy over n samples whose mistakes may be shared.

    Answer 0 is right; answers 1..n_wrong are the wrong answers. Each problem
    has one shared draw (right with probability p, else one "trap" answer).
    Each sample copies the shared draw with probability `rho`, otherwise draws
    independently (right with probability p, else a random wrong answer).
    Every single sample is right with probability p either way: only the
    correlation between samples changes.
    """
    rng = np.random.default_rng(seed)
    # (trials,): the shared draw per problem.
    shared = np.where(rng.random(trials) < p, 0, rng.integers(1, n_wrong + 1, trials))
    # (trials, n): each sample's own independent draw.
    own = np.where(rng.random((trials, n)) < p, 0, rng.integers(1, n_wrong + 1, (trials, n)))
    votes = np.where(rng.random((trials, n)) < rho, shared[:, None], own)
    counts = np.zeros((trials, n_wrong + 1))
    np.add.at(counts, (np.arange(trials)[:, None], votes), 1)
    # A random tiebreak below 1 vote, so ties don't systematically favour answer 0.
    winner = np.argmax(counts + rng.random(counts.shape) * 0.5, axis=1)
    return float(np.mean(winner == 0))


# ---------------------------------------------------------------------------
# 4. Verifiers: checking the answer, or checking every step
# ---------------------------------------------------------------------------

STEP = re.compile(r"^\s*(-?\d+)\s*\+\s*(-?\d+)\s*=\s*(-?\d+)\s*$")


def check_step(step: str) -> bool:
    """A process verifier for the toy: does this one line "a+b=c" hold?"""
    m = STEP.match(step)
    return bool(m) and int(m[1]) + int(m[2]) == int(m[3])


def first_bad_step(steps: list[str]) -> int | None:
    """Index of the first step the process verifier rejects, or None if every step checks out."""
    return next((i for i, s in enumerate(steps) if not check_step(s)), None)


def final_answer(steps: list[str]) -> int:
    """The number after the last "=": the chain's answer."""
    return int(steps[-1].rsplit("=", 1)[1])


def outcome_reward(steps: list[str], reference: int) -> float:
    """An outcome check: 1 if the final answer matches the reference, else 0. It never reads the steps."""
    return 1.0 if final_answer(steps) == reference else 0.0


def noisy_chain(numbers: list[int], step_error: float, rng: np.random.Generator) -> list[str]:
    """Write every addition as a step; each slips by ±1 or ±2 with probability `step_error`.

    A slip is carried forward: the next step adds to the wrong running total,
    which is how one early mistake poisons the final answer.
    """
    total, steps = numbers[0], []
    for x in numbers[1:]:
        new = total + x
        if rng.random() < step_error:
            new += int(rng.choice([-2, -1, 1, 2]))
        steps.append(f"{total}+{x}={new}")
        total = new
    return steps


def pass_at_n(p: float, n: int) -> float:
    """Chance that at least one of n independent samples is right: 1 - (1 - p)^n."""
    return 1 - (1 - p) ** n


def verifier_experiment(
    ns=(1, 2, 4, 8, 16, 32), n_terms: int = 6, step_error: float = 0.2, trials: int = 400, seed: int = 0
) -> list[dict]:
    """Compare four ways to turn n sampled chains into one answer.

    - single: the first sample, as if n were 1.
    - vote: the most common final answer (self-consistency).
    - verifier: the first chain whose every step checks out (best-of-n with a
      process verifier), falling back to the vote if none does.
    - pass_at_n: whether ANY chain's final answer is right, the ceiling a
      perfect answer-picker could reach.
    """
    rng = np.random.default_rng(seed)
    hits = {n: dict(single=0, vote=0, verifier=0, pass_at_n=0) for n in ns}
    for _ in range(trials):
        numbers = rng.integers(0, 10, n_terms).tolist()
        truth = sum(numbers)
        chains = [noisy_chain(numbers, step_error, rng) for _ in range(max(ns))]
        for n in ns:
            pool = chains[:n]
            answers = [final_answer(c) for c in pool]
            clean = [c for c in pool if first_bad_step(c) is None]
            picked = final_answer(clean[0]) if clean else majority_vote(answers)
            h = hits[n]
            h["single"] += answers[0] == truth
            h["vote"] += majority_vote(answers) == truth
            h["verifier"] += picked == truth
            h["pass_at_n"] += truth in answers
    return [dict(n=n, **{k: v / trials for k, v in hits[n].items()}) for n in ns]


# ---------------------------------------------------------------------------
# 5. Learning to reason with RL: verifiable rewards and group-relative advantages
# ---------------------------------------------------------------------------

DIFFICULTIES = (1, 2, 4, 8)  # additions a problem needs
LENGTHS = (1, 2, 4, 8, 16, 32)  # thinking lengths the policy can choose
INIT_TILT = 1.0  # how strongly the untrained policy prefers short answers


def group_advantages(rewards: np.ndarray, divide_by_spread: bool = True) -> np.ndarray:
    """GRPO's advantage: each reward compared with its own group, A_i = (r_i - mean) / std.

    A group where every sample scored the same carries no information about
    which choice was better, so every advantage is 0. With
    `divide_by_spread=False` the advantage is just r_i - mean, which keeps a
    tiny reward difference tiny instead of blowing it up to ±1.
    """
    rewards = np.asarray(rewards, dtype=float)
    centred = rewards - rewards.mean()
    spread = rewards.std()
    if not divide_by_spread:
        return centred
    if spread == 0:
        return np.zeros_like(rewards)
    return centred / spread


def success_probability(length: int, needs: int, step_error: float = 0.1, lucky: float = 0.1) -> float:
    """How often a chain of `length` tokens solves a problem needing `needs` steps.

    Too short (length < needs): it must guess, right with probability `lucky`.
    Otherwise every step gets `length // needs` tries: the first attempt plus
    rechecks that can catch and fix a slip. A step fails only if every try
    slips, so the chain succeeds with probability (1 - e^tries)^needs. More
    length always helps a little, which is why a reward for correctness alone
    never tells the model to stop.
    """
    if length < needs:
        return lucky
    tries = length // needs
    return (1 - step_error**tries) ** needs


def train_reasoner(
    iterations: int = 600,
    group_size: int = 16,
    lr: float = 0.2,
    length_penalty: float = 0.0,
    step_error: float = 0.1,
    lucky: float = 0.1,
    divide_by_spread: bool = True,
    seed: int = 0,
) -> dict:
    """Train a tiny policy to choose how long to think, from outcome rewards only.

    The policy is a table of logits: one row per difficulty, one column per
    thinking length in `LENGTHS`. It starts out preferring short answers, like
    a model that was never rewarded for thinking. Each iteration, for each
    difficulty, it samples a group of lengths, scores each one with a
    verifiable reward (1 if right, 0 if wrong, minus `length_penalty` per
    token), turns the rewards into group-relative advantages and nudges the
    logits along the policy gradient: logits += lr * mean(A_i * (onehot_i - pi)).

    Returns the expected mean length and accuracy after every iteration, and
    the length each difficulty ends up preferring.
    """
    rng = np.random.default_rng(seed)
    lengths = np.array(LENGTHS)
    # Short answers favoured at the start: pi(1 token) ≈ 0.63, pi(32 tokens) ≈ 0.004.
    logits = np.tile(-INIT_TILT * np.arange(len(LENGTHS)), (len(DIFFICULTIES), 1))
    # success[d, a]: how often length a solves difficulty d (d rows, a columns).
    success = np.array([[success_probability(L, d, step_error, lucky) for L in LENGTHS] for d in DIFFICULTIES])

    def policy() -> np.ndarray:
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def record(history: dict) -> None:
        pi = policy()
        # Expected values under the current policy, so the curves aren't sampling noise.
        history["mean_length"].append(float((pi @ lengths).mean()))
        history["accuracy"].append(float((pi * success).sum(axis=1).mean()))

    history: dict = dict(mean_length=[], accuracy=[])
    record(history)
    for _ in range(iterations):
        pi = policy()
        for row in range(len(DIFFICULTIES)):
            # A group of G attempts at the same problem, each thinking for a sampled length.
            actions = rng.choice(len(LENGTHS), size=group_size, p=pi[row])
            solved = rng.random(group_size) < success[row, actions]
            rewards = solved.astype(float) - length_penalty * lengths[actions]
            adv = group_advantages(rewards, divide_by_spread)
            # d log pi(a) / d logits = onehot(a) - pi: raise the chosen length in proportion to its advantage.
            onehot = np.eye(len(LENGTHS))[actions]
            logits[row] += lr * (adv[:, None] * (onehot - pi[row])).mean(axis=0)
        record(history)
    pi = policy()
    history["policy"] = pi
    history["preferred_length"] = {d: int(lengths[np.argmax(pi[i])]) for i, d in enumerate(DIFFICULTIES)}
    history["expected_length"] = {d: float(pi[i] @ lengths) for i, d in enumerate(DIFFICULTIES)}
    return history


# ---------------------------------------------------------------------------
# 6. What reasoning costs, and where it still fails
# ---------------------------------------------------------------------------


def steps_all_right(step_error: float, steps: int) -> float:
    """Chance a chain of `steps` steps has no slip at all: (1 - e)^k."""
    return (1 - step_error) ** steps


def reasoning_cost(samples: int, tokens: int, dollars_per_million: float, tokens_per_second: float) -> dict:
    """Money and waiting time for one question.

    Money grows with every token of every sample. Waiting time, when the
    samples run side by side, is set by the length of one chain.
    """
    return dict(
        dollars=samples * tokens * dollars_per_million / 1_000_000,
        seconds=tokens / tokens_per_second,
    )


# ---------------------------------------------------------------------------
# 7. Figures (rendered into the HTML docs by `make figures`)
# ---------------------------------------------------------------------------


def _direct_vs_cot(max_additions: int = 10, per_size: int = 300, seed: int = 0) -> list[dict]:
    """Accuracy and tokens for answering at once vs. writing every step, by problem size."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(1, max_additions + 1):
        problems = [rng.integers(0, 10, d + 1).tolist() for _ in range(per_size)]
        direct = [solve(p, 1) for p in problems]
        cot = [solve(p, 64) for p in problems]
        rows.append(dict(
            additions=d,
            direct=float(np.mean([a.answer == sum(p) for a, p in zip(direct, problems)])),
            cot=float(np.mean([a.answer == sum(p) for a, p in zip(cot, problems)])),
            cot_tokens=float(np.mean([a.tokens for a in cot])),
        ))
    return rows


def figures() -> dict:
    """Plot this lesson's data. matplotlib is imported here, and only here,
    so the lesson itself needs nothing beyond NumPy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, RED, GREEN, AMBER, PURPLE, MUTED = "#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#9ca3af"
    figs = {}

    # --- 1. Answering at once vs. thinking out loud --------------------------
    rows = _direct_vs_cot()
    ds = [r["additions"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    a1.plot(ds, [r["direct"] for r in rows], "o-", color=RED, label="answer in 1 token")
    a1.plot(ds, [r["cot"] for r in rows], "o-", color=BLUE, label="write steps, then answer")
    a1.set_xlabel("additions the problem needs")
    a1.set_ylabel("accuracy")
    a1.set_ylim(-0.03, 1.05)
    a1.set_title("One addition per pass: steps are the only way")
    a1.legend(frameon=False)
    a2.plot(ds, [1] * len(ds), "o-", color=RED, label="answer in 1 token")
    a2.plot(ds, [r["cot_tokens"] for r in rows], "o-", color=BLUE, label="write steps, then answer")
    a2.set_xlabel("additions the problem needs")
    a2.set_ylabel("tokens (forward passes) used")
    a2.set_title("The price: one token per step")
    fig.tight_layout()
    figs["direct_vs_cot"] = fig

    # --- 2. Accuracy vs. thinking budget --------------------------------------
    budgets = (1, 2, 4, 8, 16, 32, 64)
    sweep = budget_sweep(budgets, n_problems=600, seed=0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    a1.plot(budgets, [budget_accuracy(b) for b in budgets], "-", color=MUTED, label="formula, g = 0.1")
    a1.plot(budgets, [r["accuracy"] for r in sweep], "o", color=BLUE, label="simulated sums")
    a1.set_xscale("log", base=2)
    a1.set_xlabel("thinking budget B (tokens, log scale)")
    a1.set_ylabel("accuracy")
    a1.set_ylim(0, 1.05)
    a1.set_title("Each doubling of budget buys the same step")
    a1.legend(frameon=False)
    a2.plot(budgets, [r["mean_tokens"] for r in sweep], "o-", color=AMBER)
    a2.plot(budgets, budgets, ":", color=MUTED, label="budget allowed")
    a2.set_xscale("log", base=2)
    a2.set_yscale("log", base=2)
    a2.set_xlabel("thinking budget B (tokens, log scale)")
    a2.set_ylabel("tokens actually spent (mean)")
    a2.set_title("Easy problems stop early; past 32, nothing changes")
    a2.legend(frameon=False)
    fig.tight_layout()
    figs["budget_scaling"] = fig

    # --- 3. Voting: the binomial formula and scattered wrong answers ----------
    ns = [1, 3, 5, 7, 9, 15, 21, 31]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for p, color in ((0.8, GREEN), (0.6, BLUE), (0.4, RED)):
        ax.plot(ns, [majority_accuracy(p, n) for n in ns], "o-", color=color, label=f"yes/no question, p = {p}")
    ax.plot(ns, [correlated_vote_accuracy(0.4, n, rho=0.0, n_wrong=10, trials=1500) for n in ns], "s--", color=RED,
            label="p = 0.4, wrong answers scattered over 10 values")
    ax.set_xlabel("samples voting, n")
    ax.set_ylabel("accuracy of the vote")
    ax.set_ylim(0, 1.05)
    ax.set_title("Voting amplifies whatever answer is most common")
    ax.legend(frameon=False, fontsize=8)
    figs["voting"] = fig

    # --- 4. Correlated mistakes put a ceiling on voting -----------------------
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for rho, color in ((0.0, BLUE), (0.3, GREEN), (0.6, AMBER), (0.9, RED)):
        ax.plot(ns, [correlated_vote_accuracy(0.6, n, rho=rho, n_wrong=5, trials=1500) for n in ns], "o-", color=color,
                label=f"shared-mistake rate rho = {rho}")
    ax.axhline(0.6, color=MUTED, ls=":")
    ax.text(12, 0.56, "one sample alone: 0.6", color="#4b5563", fontsize=8)
    ax.set_xlabel("samples voting, n")
    ax.set_ylabel("accuracy of the vote")
    ax.set_ylim(0.2, 1.03)
    ax.set_title("Same per-sample accuracy, very different votes")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    figs["correlated_votes"] = fig

    # --- 5. Best-of-n with a verifier ------------------------------------------
    vns = (1, 2, 4, 8, 16, 32)
    vrows = verifier_experiment(vns, trials=400, seed=0)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(vns, [r["pass_at_n"] for r in vrows], ":", color=MUTED, lw=2, label="pass@n: any sample right (ceiling)")
    ax.plot(vns, [r["verifier"] for r in vrows], "o-", color=GREEN, label="best-of-n, step checker picks")
    ax.plot(vns, [r["vote"] for r in vrows], "o-", color=BLUE, label="majority vote")
    ax.plot(vns, [r["single"] for r in vrows], "o-", color=RED, label="one sample")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("chains sampled, n (log scale)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("A good checker turns many tries into a right answer")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    figs["verifier"] = fig

    # --- 6. RL: learning how long to think ------------------------------------
    runs = (
        ("correctness only", dict(), BLUE),
        ("penalty 0.01 per token, ÷ spread (GRPO)", dict(length_penalty=0.01), RED),
        ("penalty 0.01 per token, no ÷ spread", dict(length_penalty=0.01, divide_by_spread=False), GREEN),
    )
    histories = [(label, train_reasoner(seed=0, **kw), color) for label, kw, color in runs]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    for label, h, color in histories:
        a1.plot(h["mean_length"], color=color, label=label)
        a2.plot(h["accuracy"], color=color, label=label)
    a1.set_xlabel("training iteration")
    a1.set_ylabel("mean thinking length (tokens)")
    a1.set_title("It learns to write longer")
    a2.set_xlabel("training iteration")
    a2.set_ylabel("accuracy")
    a2.set_ylim(0.35, 1.0)
    a2.set_title("...and gets more right")
    a2.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    figs["rl_training"] = fig

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(DIFFICULTIES))
    width = 0.26
    for i, (label, h, color) in enumerate(histories):
        ax.bar(x + (i - 1) * width, [h["expected_length"][d] for d in DIFFICULTIES], width, color=color, label=label)
    ax.plot(x, DIFFICULTIES, "k_", ms=28, mew=2, label="steps needed d")
    ax.plot(x, [2 * d for d in DIFFICULTIES], "_", color="#4b5563", ms=28, mew=1.5, ls="none", label="2d: room for one recheck")
    ax.set_xticks(x, [f"d = {d}" for d in DIFFICULTIES])
    ax.set_ylim(0, 18)
    ax.set_xlabel("difficulty: additions the problem needs")
    ax.set_ylabel("tokens the trained model spends")
    ax.set_title("Hard problems get more thinking, with room to recheck")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    figs["rl_lengths"] = fig

    # --- 7. Compounding slips over long chains --------------------------------
    ks = np.arange(1, 201)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for e, color in ((0.005, GREEN), (0.02, BLUE), (0.05, RED)):
        ax.plot(ks, [steps_all_right(e, k) for k in ks], color=color, label=f"slip rate per step = {e}")
    ax.plot([50], [steps_all_right(0.02, 50)], "o", color=BLUE)
    ax.annotate("50 steps at 2%: 0.36", (50, steps_all_right(0.02, 50)), xytext=(70, 0.55), arrowprops=dict(arrowstyle="->", color="#4b5563"))
    ax.set_xlabel("steps in the chain, k")
    ax.set_ylabel("chance every step is right")
    ax.set_ylim(0, 1.03)
    ax.set_title("Long chains need checking, not just length")
    ax.legend(frameon=False)
    figs["compounding"] = fig

    return figs


# ---------------------------------------------------------------------------
# 8. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. Thinking out loud: one addition per forward pass")
    say(
        """
        Our toy model can do exactly one addition per forward pass, and every
        token it writes is one forward pass. Ask it for 3 + 5 + 8 + 2.
        """
    )
    direct = solve([3, 5, 8, 2], max_tokens=1)
    cot = solve([3, 5, 8, 2], max_tokens=10)
    table(
        ["how it answers", "what it wrote", "answer", "tokens"],
        [
            ("at once", "(nothing)", direct.answer, direct.tokens),
            ("thinking first", ", ".join(cot.steps), cot.answer, cot.tokens),
        ],
    )
    say(
        """
        At once, it adds 3 + 5 = 8 and has no pass left for 8 and 2, so it
        estimates them at 4.5 each: 17, wrong. Thinking first, it writes each
        running total where the next pass can read it: 18, right, for 3 tokens.
        """
    )
    takeaway("Each written token is another forward pass: writing steps buys serial computation the weights alone don't have.")

    banner("2. Test-time compute: accuracy vs. thinking budget")
    rows = budget_sweep()
    table(
        ["budget B", "formula acc(B)", "simulated accuracy", "tokens actually spent"],
        [(r["budget"], budget_accuracy(r["budget"]), r["accuracy"], r["mean_tokens"]) for r in rows],
        floatfmt=".3f",
    )
    say(
        """
        Problems need 1, 2, 4, 8, 16 or 32 additions. Each doubling of the
        budget lets one more difficulty level fit, so accuracy climbs by a
        steady step per doubling, then stops once the hardest problem fits.
        """
    )
    takeaway("With difficulty spread over many scales, accuracy grows with the log of the thinking budget, until it doesn't.")

    banner("3. Sample many, vote: self-consistency")
    say("Three sampled answers to 3 + 5 + 8 + 2: 18, 17, 18. The vote picks " + str(majority_vote([18, 17, 18])) + ".")
    table(
        ["samples n", "yes/no, p = 0.6", "yes/no, p = 0.4", "p = 0.6, rho = 0.9 (shared mistakes)"],
        [(n, majority_accuracy(0.6, n), majority_accuracy(0.4, n), correlated_vote_accuracy(0.6, n, rho=0.9)) for n in (1, 3, 5, 15, 31)],
        floatfmt=".3f",
    )
    say(
        """
        Independent 60% votes climb towards certainty; independent 40% votes on
        a yes/no question sink. When samples share their mistakes (rho = 0.9),
        31 votes are barely better than one: they are one opinion, repeated.
        """
    )
    takeaway("Voting helps only when the right answer is the most common one and the mistakes are independent.")

    banner("4. Verifiers: check the answer, or check every step")
    slipped = ["3+5=8", "8+8=17", "17+2=19"]
    lucky = ["3+5=9", "9+8=16", "16+2=18"]
    table(
        ["chain", "outcome reward (answer 18?)", "first bad step (process check)"],
        [(", ".join(c), int(outcome_reward(c, 18)), first_bad_step(c)) for c in (slipped, lucky)],
    )
    say(
        """
        The outcome check only sees the last number: it gives the lucky chain,
        whose two slips cancelled, full marks. The step check reads every line
        and points at the exact slip.
        """
    )
    table(
        ["chains n", "one sample", "vote", "best-of-n, step checker", "pass@n ceiling"],
        [(r["n"], r["single"], r["vote"], r["verifier"], r["pass_at_n"]) for r in verifier_experiment()],
        floatfmt=".3f",
    )
    takeaway("A reliable checker turns 'right sometimes' into 'right almost always' with enough samples; voting gets there far more slowly.")

    banner("5. Learning to reason with RL: verifiable rewards, group advantages")
    say(f"Group rewards [1, 0, 0, 1] give advantages {group_advantages(np.array([1.0, 0.0, 0.0, 1.0])).tolist()}.")
    for label, kw in (
        ("correctness only", dict()),
        ("penalty 0.01/token, GRPO", dict(length_penalty=0.01)),
        ("penalty 0.01/token, no ÷ spread", dict(length_penalty=0.01, divide_by_spread=False)),
    ):
        h = train_reasoner(seed=0, **kw)
        print(
            f"{label:32s} length {h['mean_length'][0]:.1f} -> {h['mean_length'][-1]:.1f} tokens, "
            f"accuracy {h['accuracy'][0]:.2f} -> {h['accuracy'][-1]:.2f}, "
            f"prefers {h['preferred_length']}"
        )
    print()
    say(
        """
        Rewarded only for right answers, the policy learns to think longer
        (about 2 tokens to about 7.6) and to leave room for a recheck on hard
        problems (16 tokens for 8 additions). A per-token penalty trims easy
        problems; dividing by the group's spread magnifies that penalty until
        1-addition problems get 1 token and accept an occasional slip.
        """
    )
    takeaway("Reward the outcome and let the model search: longer, self-checking reasoning emerges because it pays.")

    banner("6. What it costs, and where it fails")
    table(
        ["slip rate", "steps", "chance all right"],
        [(e, k, steps_all_right(e, k)) for e, k in ((0.02, 10), (0.02, 50), (0.02, 200), (0.005, 200))],
        floatfmt=".3f",
    )
    c1 = reasoning_cost(1, 2000, 10.0, 50)
    c8 = reasoning_cost(8, 2000, 10.0, 50)
    say(
        f"""
        One 2,000-token chain at $10 per million tokens costs ${c1['dollars']:.2f}
        and takes {c1['seconds']:.0f} s at 50 tokens per second. Eight in parallel
        cost ${c8['dollars']:.2f} and still take {c8['seconds']:.0f} s.
        """
    )
    takeaway("Thinking is paid per token: spend it where a check shows it helps, and cap it everywhere else.")


if __name__ == "__main__":
    demo()
