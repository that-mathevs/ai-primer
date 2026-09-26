r"""
# Reinforcement learning: learning from a score instead of an answer

Run: `python -m primer.ml.reinforcement`

## The everyday picture

Think of teaching a dog to sit. You can't show it the right answer: you can
only wait for it to try something and give it a treat when the something was
good. Over many tries the dog does more of what earned treats and less of
what didn't. Nobody ever told it what "sit" means; it worked it out from a
score.

That is **reinforcement learning (RL)**. An **agent** (the dog, or a
language model) takes an **action** (sits, or writes an answer), the world
hands back a **reward** (a treat, or a score from a grader), and the agent
adjusts itself so that rewarded actions become more likely. The agent's
current habits, written as a probability for every action, are called its
**policy**.

Compare this with ordinary supervised learning (`primer.ml.losses`), where
every example comes with the correct answer attached. In RL there is no
answer key, only a score after the fact. That is exactly the situation a
language model is in once pretraining is over: for "prove this theorem" or
"write a helpful reply" there is no single correct text to copy, but a
checker or a judge can say how good an attempt was. RL is how the model
learns from those judgements (`primer.ml.training_stages` places it in the
training pipeline).

```mermaid
flowchart LR
  P[Policy<br/>a probability for every action] -->|sample| A[Action]
  A --> E[Environment<br/>a slot machine, a grader, a user]
  E --> R[Reward<br/>one number]
  R -->|nudge the policy| P
```

**Reading it:** follow the loop clockwise. The policy is a set of
probabilities, and the action is *sampled* from it, so the agent sometimes
tries things it isn't sure about. The environment is whatever judges the
action; the agent can't see inside it. The only thing that comes back is one
number, the reward, and the only thing the agent can do with it is nudge its
own probabilities. Everything in this lesson is a better answer to one
question: how exactly should that nudge be computed?

## A tiny worked example: three slot machines

The simplest RL problem is a row of slot machines, called a **multi-armed
bandit** (a slot machine is a "one-armed bandit"). Machine A pays out 20% of
the time, B 50% and C 80%, but the agent isn't told that. Each pull pays 1
or 0. The agent must find C by pulling and seeing what happens.

A policy here is three probabilities, one per arm. A policy that picks each
arm a third of the time earns, on average, (0.2 + 0.5 + 0.8) / 3 = **0.5**
per pull. A policy that always picks C earns **0.8**. Learning means moving
from the first policy to the second using nothing but the 1s and 0s.

This one number, the average reward a policy expects, is what RL maximises:

$$
J(\theta) = \sum_{a} \pi_\theta(a)\, R(a)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $a$ | one action: which arm to pull | A, B or C |
| $\theta$ | the policy's adjustable numbers (here, one score per arm, called **logits**) | (0, 0, 0) |
| $\pi_\theta(a)$ | the policy: the probability of picking action $a$, given $\theta$. Here softmax of the logits | 1/3 each |
| $R(a)$ | the average reward action $a$ pays (unknown to the agent) | 0.2, 0.5, 0.8 |
| $\sum_{a}$ | add up over every action | three terms |
| $J(\theta)$ | the expected reward: what the policy earns per pull, on average | 0.5 |

**In words:** "the expected reward is each action's probability times its
average payout, added up over the actions."

**With the numbers:** J = ⅓·0.2 + ⅓·0.5 + ⅓·0.8 = 0.5 for the uniform
policy, and 1·0.8 = 0.8 for the policy that always pulls C.

**In Python:**

```python
policy = [1/3, 1/3, 1/3]
# average payout of arms A, B, C
R = [0.2, 0.5, 0.8]
# J = Σ_a π(a) R(a)
round(sum(p * r for p, r in zip(policy, R)), 3)  # → 0.5
round(sum(p * r for p, r in zip([0, 0, 1], R)), 3)  # → 0.8
```

The agent can't compute J, because it doesn't know R. It can only sample:
pull an arm, see a 1 or a 0. Every method below turns those samples into an
estimate of which way to move θ to make J bigger. Trying an arm you're
unsure of is called **exploration**; sticking with the best arm so far is
**exploitation**. A sampled policy does some of both automatically, as
long as no arm's probability has collapsed to zero.

**In code:** `Bandit` hides the win chances and pays out one pull at a time; `expected_reward` is the formula above.

## Policy gradients: do more of what worked (REINFORCE)

**Everyday picture.** A football coach reviews the tape after a match. For
every play that led to a goal, they tell the team "a bit more of that"; for
plays that went nowhere, nothing. They don't need to know *why* the play
worked. Repeat over hundreds of matches and the team drifts towards the
plays that score.

**Tiny worked example.** Start with logits (0, 0, 0), so each arm has
probability ⅓. The agent pulls C and wins: reward 1. The rule, explained
next, says: add to each logit *learning rate × reward × (1 if it's the
chosen arm, else 0, minus that arm's probability)*.

| Arm | Chosen? | 1[chosen] − π | × reward 1 × rate 0.5 | New logit | New probability |
|---|---|---|---|---|---|
| A | no | 0 − ⅓ = −0.333 | −0.167 | −0.167 | **0.274** |
| B | no | 0 − ⅓ = −0.333 | −0.167 | −0.167 | **0.274** |
| C | yes | 1 − ⅓ = +0.667 | +0.333 | +0.333 | **0.452** |

One lucky pull moved C from 33% to 45%. Had the pull paid 0, nothing would
have moved. Had the agent pulled A and won (A wins sometimes too), A would
have gone up instead. The rule is noisy, one pull at a time, but on average
the arm that wins most gets pushed up most.

```mermaid
flowchart LR
  L[Logits θ] --> S[softmax<br/>probabilities π]
  S -->|sample| A[Action a]
  A --> ENV[Pull the arm] --> R[Reward R]
  S --> G["∇ log π(a)<br/>= one-hot(a) − π"]
  A --> G
  G --> M["× R × learning rate"]
  R --> M
  M -->|add to| L
```

**Reading it:** the top path is acting: logits become probabilities, one
action is sampled, and the environment pays a reward. The lower path is
learning: from the action alone, work out which direction in logit space
makes that action more likely (the `∇ log π` box), then scale that direction
by how good the outcome was. A big reward is a big step towards repeating the
action; zero reward is no step.

### The log-probability trick, decoded

We want the **gradient** of J: for each logit, how much J rises if the
logit rises a little (see `primer.notation` for gradients from scratch).
The difficulty is that J is an average over actions we can only sample. The
trick rewrites the gradient as an average too, so a sample estimates it:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{a \sim \pi_\theta}\big[\, R(a)\, \nabla_\theta \log \pi_\theta(a) \,\big]
\qquad
\frac{\partial \log \pi_\theta(a)}{\partial z_k} = \mathbb{1}[k = a] - \pi_\theta(k)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $\nabla_\theta$ | "gradient with respect to θ": one slope per logit, collected into a vector | 3 slopes |
| $\mathbb{E}_{a \sim \pi_\theta}[\ldots]$ | **expected value**: the average of the bracket when $a$ is sampled from the policy | average over pulls |
| $\sim$ | "drawn from" | |
| $\log$ | the natural logarithm, the undo button for $e^x$ | $\log \tfrac13 = -1.10$ |
| $\nabla_\theta \log \pi_\theta(a)$ | the direction in logit space that makes action $a$ more likely, fastest | $(-\tfrac13, -\tfrac13, \tfrac23)$ for C |
| $z_k$ | the $k$-th logit (θ is the list of logits) | $z_3 = 0$ |
| $\partial$ | "partial derivative": the slope along one logit, holding the others still | |
| $\mathbb{1}[k = a]$ | 1 if $k$ is the chosen action, else 0 | (0, 0, 1) |
| $\pi_\theta(k)$ | the probability of action $k$ | ⅓ |

**In words:** "the direction that raises expected reward is, on average,
the direction that makes the sampled action more likely, weighted by the
reward it earned. For a softmax policy, that direction is 'one for the
chosen action, minus every action's probability'."

Why is this true? Because the slope of a probability equals the probability
times the slope of its log ($\nabla \pi = \pi \, \nabla \log \pi$, the chain
rule applied to log). So $\nabla J = \sum_a \nabla\pi(a) R(a) = \sum_a \pi(a)
\nabla\log\pi(a) R(a)$, and a sum weighted by $\pi(a)$ is an average over
samples from π. The update is then plain **gradient ascent**:
$\theta \leftarrow \theta + \alpha\, R\, \nabla_\theta \log \pi_\theta(a)$,
with learning rate $\alpha$ (`primer.ml.optimizers`).

**With the numbers:** at the uniform policy the true gradient is
(−0.1, 0, +0.1): push C up, A down, leave B (which pays exactly the average)
alone. One sample, "pulled C, got 1", estimates it as 1 × (−⅓, −⅓, ⅔). The
step with α = 0.5 gives logits (−0.167, −0.167, 0.333) and probabilities
(0.274, 0.274, 0.452), as in the table.

**In Python:**

```python
import math
z = [0.0, 0.0, 0.0]
pi = [math.exp(z_k) / sum(math.exp(v) for v in z) for z_k in z]
# the true gradient: Σ_a π(a) R(a) (1[k=a] − π(k)), for each logit k
R = [0.2, 0.5, 0.8]
true_grad = [sum(pi[a] * R[a] * ((k == a) - pi[k]) for a in range(3)) for k in range(3)]
[round(g, 3) for g in true_grad]  # → [-0.1, 0.0, 0.1]
# one sample: pulled C (a = 2), reward 1
a, reward, alpha = 2, 1.0, 0.5
grad_log_pi = [(k == a) - pi[k] for k in range(3)]
[round(g, 3) for g in grad_log_pi]  # → [-0.333, -0.333, 0.667]
z = [z_k + alpha * reward * g for z_k, g in zip(z, grad_log_pi)]
[round(z_k, 3) for z_k in z]  # → [-0.167, -0.167, 0.333]
[round(math.exp(z_k) / sum(math.exp(v) for v in z), 3) for z_k in z]  # → [0.274, 0.274, 0.452]
```

This algorithm is called **REINFORCE** (Williams, 1992). Run it for 500
pulls and the policy finds arm C:

![REINFORCE on the three-armed bandit: the probability of arm C climbs from a third to about 0.96 within 500 pulls, and the average reward rises from 0.5 towards 0.8](figures/primer.ml.reinforcement.bandit_learning.svg)

**Reading it:** on the left, each line is one arm's probability over 500
pulls. All three start at ⅓. C's line (the 80% arm) climbs towards 1 while A
and B sink; the wiggles are single lucky or unlucky pulls. On the right is
the reward, averaged over the last 50 pulls. It starts near 0.5 (random
pulling) and rises towards the dashed line at 0.8, the most any policy can
earn. Nobody told the agent which arm was best: the 1s and 0s were enough.

**Why it matters in practice.** A language model is exactly this kind of
policy, with a vocabulary of tokens as its arms, and one sampled answer is a
string of sampled tokens. REINFORCE applies unchanged: sum the log
probabilities of every token in the answer, and scale the gradient by the
answer's reward. Every method below (PPO, GRPO) is REINFORCE with repairs.

**In code:** `grad_log_prob` is one-hot minus the probabilities, `reinforce_step` is one update, `worked_reinforce_step` is the table above, and `train_reinforce` runs the whole loop against a `Bandit`.

## Variance and baselines: grade on a curve

**Everyday picture.** A teacher whose class all scores between 90 and 100
learns nothing by being told "you got a 92". What matters is whether 92 is
above or below the class average. Raw scores that are all large and
positive make every attempt look good; only the difference from typical
tells you which way to go.

**Tiny worked example.** Two arms, a 50/50 policy, and every pull pays a lot:
arm 1 always pays 10, arm 2 always pays 12. Arm 2 is better, so the logit of
arm 2 should rise. Look at the REINFORCE estimate for that logit:

| Pulled | Reward | 1[arm 2] − π(arm 2) | Estimate (no baseline) | Estimate (baseline 11) |
|---|---|---|---|---|
| arm 1 | 10 | 0 − 0.5 = −0.5 | 10 × −0.5 = **−5** | (10 − 11) × −0.5 = **+0.5** |
| arm 2 | 12 | 1 − 0.5 = +0.5 | 12 × +0.5 = **+6** | (12 − 11) × +0.5 = **+0.5** |

Without a baseline the estimate is −5 or +6 depending on the coin flip.
It averages to +0.5, the right answer, but any single sample points the
wrong way half the time, and violently. Subtract the average reward, 11,
first and *every* sample says +0.5. Same average, no noise at all.

$$
\nabla_\theta J(\theta) = \mathbb{E}_{a \sim \pi_\theta}\big[\, (R(a) - b)\, \nabla_\theta \log \pi_\theta(a) \,\big],
\qquad A(a) = R(a) - b
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $b$ | the **baseline**: any number that doesn't depend on which action was taken; usually the average reward | 11 |
| $A(a)$ | the **advantage**: how much better action $a$ did than typical | −1 for arm 1, +1 for arm 2 |
| everything else | as in the REINFORCE formula above | |

**In words:** "scale each step by how much better than typical the action
did, not by its raw reward."

Why is it allowed? Because the baseline's contribution averages to zero:
$\mathbb{E}[\, b\, \nabla \log \pi(a)] = b \sum_a \nabla \pi(a) = b\, \nabla
\sum_a \pi(a) = b\, \nabla 1 = 0$. Probabilities always add to 1, so pushing
all of them up is impossible; the baseline only removes noise, never
signal.

**With the numbers:** without a baseline the estimate's **variance** (the
average squared distance from its mean, see `primer.notation`) is
(25 + 36)/2 − 0.5² = **30.25**. With b = 11 it is **0**.

**In Python:**

```python
# arm 1 pays 10, arm 2 pays 12; the 1[arm 2] − π(arm 2) factor for each pull
pulls = [(10, -0.5), (12, +0.5)]
def mean_and_variance(b):
    estimates = [(reward - b) * direction for reward, direction in pulls]
    mean = sum(estimates) / 2
    return mean, sum((e - mean) ** 2 for e in estimates) / 2
mean_and_variance(b=0)  # → (0.5, 30.25)
mean_and_variance(b=11)  # → (0.5, 0.0)
```

```mermaid
flowchart LR
  R[Reward R] --> MINUS["R − b"]
  B["Baseline b<br/>average reward so far"] --> MINUS
  MINUS --> ADV{Advantage A}
  ADV -->|positive: better than typical| UP[make the action<br/>more likely]
  ADV -->|negative: worse than typical| DOWN[make the action<br/>less likely]
```

**Reading it:** the baseline sits between the reward and the update. Its
job is to turn "how good was this?" into "how much better than usual was
this?". The sign of the advantage now decides the direction of the step,
so a below-average action is actively pushed down, even though its raw
reward was positive.

To see it matter, give every arm of our bandit 5 extra points: rewards are
now 5 or 6 instead of 0 or 1, and nothing about which arm is best has
changed.

![With rewards offset by 5, the gradient estimate's variance falls from about 20 to 0.15 with a baseline, and 20 training runs all find arm C with it, while without it most runs lock onto the wrong arm](figures/primer.ml.reinforcement.baselines.svg)

**Reading it:** on the left, the variance of a one-pull gradient estimate
at the starting policy, on a log scale: about 20 without a baseline and
about 0.15 with one, over a hundred times smaller. On the right, each dot is
one of 20 training runs (400 pulls each), placed at its final probability
of picking C. With a running-average baseline (blue) every run ends near
0.95. Without one (red), the dots scatter to both ends: in most runs, early
pulls of a mediocre arm paid 5 and were pushed up hard, and the policy
committed before it ever learned C was better.

**Why it matters in practice.** Every practical policy-gradient method uses
a baseline. PPO learns one with a second network, the **value network** (or
**critic**), which predicts the expected reward from each state. GRPO, below,
gets one for free by comparing several answers to the same prompt.

**In code:** `gradient_estimate_stats` computes the exact mean and variance of the estimate, `sampled_gradient_variance` measures it from real pulls, and `train_reinforce` subtracts a running average when `baseline=True`.

## PPO: take several steps, but never too far

**Everyday picture.** A chef tests a new recipe on one evening's diners.
It would be wasteful to use their comments for just one small tweak, so the
chef makes several rounds of changes from the same comment cards. But the
further the recipe drifts from what the diners actually ate, the less their
comments apply, so the chef caps each change: never more than 20% more or
less of any ingredient per round.

For a language model, sampling answers is the expensive part (every answer
is a full generation), so **PPO (Proximal Policy Optimization)** reuses each
batch of answers for several gradient steps. It needs a way to tell how far
the policy has moved since the batch was sampled, and a brake.

**Tiny worked example.** The **probability ratio** compares the policy now
with the policy that generated the sample. A ratio of 1.5 means the current
policy is 50% more likely to produce that answer than when it was sampled.
With a clip range ε = 0.2 the ratio is allowed to count only between 0.8
and 1.2:

| Ratio ρ | Advantage A | ρ·A | clip(ρ, 0.8, 1.2)·A | min of the two | What happened |
|---|---|---|---|---|---|
| 1.1 | +3 | 3.3 | 3.3 | **3.3** | inside the band: plain REINFORCE |
| 1.5 | +2 | 3.0 | 1.2 × 2 = 2.4 | **2.4** | good action already boosted enough: gain capped |
| 0.5 | −1 | −0.5 | 0.8 × −1 = −0.8 | **−0.8** | bad action already cut enough: capped |
| 1.5 | −1 | −1.5 | 1.2 × −1 = −1.2 | **−1.5** | bad action made *more* likely: full penalty |

$$
\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}
\qquad
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(\rho_t A_t,\ \operatorname{clip}(\rho_t,\ 1-\varepsilon,\ 1+\varepsilon)\, A_t\big)\Big]
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $t$ | one sample in the batch; for a language model, one token of one answer | row of the table |
| $s_t$ | the **state**: what the policy saw before acting (the prompt plus the tokens so far) | a prompt |
| $a_t$ | the action taken (the token generated) | an answer |
| $\pi_{\text{old}}$ | the policy as it was when the batch was sampled, frozen | |
| $\pi_\theta$ | the policy now, after some steps on this batch | |
| $\rho_t$ | the probability ratio, new over old | 1.5 |
| $A_t$ | the advantage of that sample | +2 |
| $\varepsilon$ | the clip range, typically 0.1 to 0.3 | 0.2 |
| $\operatorname{clip}(x, lo, hi)$ | $x$, but pushed back to $lo$ or $hi$ if it falls outside | clip(1.5, 0.8, 1.2) = 1.2 |
| $\min$ | the smaller of the two | min(3.0, 2.4) = 2.4 |
| $\mathbb{E}_t$ | the average over all samples in the batch | |
| $L^{\text{CLIP}}$ | the objective PPO climbs | |

**In words:** "for each sample, take the ratio-weighted advantage, but once
the ratio has moved more than ε from 1 in the direction the advantage wants,
stop counting further movement; and always take the more pessimistic of the
clipped and unclipped versions."

The slope of ρ·A is exactly REINFORCE's gradient scaled by ρ (because
$\nabla\rho = \rho\,\nabla\log\pi_\theta$), so inside the band PPO is
REINFORCE with importance weighting. Outside it, the clipped term is flat:
that sample stops pushing.

**With the numbers:** the four rows of the table, computed:

**In Python:**

```python
def clip(x, lo, hi):
    return max(lo, min(x, hi))
def L_clip(rho, A, eps=0.2):
    return min(rho * A, clip(rho, 1 - eps, 1 + eps) * A)
[round(L_clip(rho, A), 2) for rho, A in [(1.1, 3), (1.5, 2), (0.5, -1), (1.5, -1)]]  # → [3.3, 2.4, -0.8, -1.5]
# past 1 + ε with a positive advantage, a higher ratio earns nothing more
L_clip(1.3, 2) == L_clip(1.6, 2)  # → True
```

```mermaid
flowchart TB
  OLD[Policy π_old] -->|generate a batch| B[Answers + rewards]
  B --> V[Value network<br/>predicts expected reward]
  V --> ADV[Advantages A_t]
  B --> ADV
  ADV --> LOOP
  subgraph LOOP["Several passes over the same batch"]
    RATIO["ratio ρ = π_θ / π_old"] --> CLIP["clip to 1 ± ε, take the min"]
    CLIP --> KL["subtract β × KL to the reference model"]
    KL --> STEP[gradient step on θ]
    STEP --> RATIO
  end
  LOOP -->|π_θ becomes the new π_old| OLD
```

**Reading it:** the outer loop is sampling, the expensive part: the frozen
old policy writes a batch of answers and a value network turns their rewards
into advantages. The inner loop reuses that batch for several passes. Each
pass recomputes how far the policy has moved (the ratio), stops counting
movement beyond the band (the clip), and applies the KL leash described
below. When the passes are done, the updated policy becomes the new sampler
and the cycle repeats.

To see the clip work, take one batch of 16 pulls from the bandit (C won all
four of its pulls, A lost all four) and make 50 passes over it:

![Two panels: the clipped objective is flat outside the 0.8 to 1.2 band on the side the advantage favours; over 50 passes the unclipped ratio climbs past 2.5 while the clipped one levels off near 1.24](figures/primer.ml.reinforcement.ppo_clip.svg)

**Reading it:** on the left is the objective for one sample as its ratio
changes. For a positive advantage (blue), the line rises with the ratio
until 1.2, then goes flat: no reward for pushing further. For a negative
advantage (red), it goes flat below 0.8. The dashed lines are what
REINFORCE would keep climbing. On the right, the largest ratio in the batch
after each of 50 passes. Unclipped (red), the policy chases the same 16
pulls further every pass until C's probability is 2.5 times what it was,
about 0.84 from sixteen pulls, which is wildly overconfident. Clipped
(blue), it levels off near 1.24: the brake is not a hard wall (other
samples can still nudge the policy), but it removes the incentive to
overfit one batch.

**In code:** `ppo_clipped_objective` is the formula, `clipped_policy_update` makes several passes over one batch with or without the clip, and `ppo_drift_experiment` is the 50-pass comparison.

### The KL leash: stay close to where you started

**Everyday picture.** A dog on a long leash can explore, but it can't run
off a cliff. In RL for language models, the leash ties the policy to a
frozen copy of the model it started from, the **reference model** (usually
the model after supervised fine-tuning).

**Tiny worked example.** An answer earns reward 1.0. The policy now gives
it probability 0.6; the reference gave it 0.3. With leash strength β = 0.1,
the reward actually used for training is 1.0 − 0.1 × ln(0.6 / 0.3) =
1.0 − 0.1 × 0.693 = **0.931**. The policy pays a small fee for having
doubled that answer's probability.

$$
R'(a) = R(a) - \beta \log\frac{\pi_\theta(a)}{\pi_{\text{ref}}(a)}
\qquad
\mathbb{E}_{a\sim\pi_\theta}\big[R'(a)\big] = \mathbb{E}_{a\sim\pi_\theta}\big[R(a)\big] - \beta\, \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $R(a)$ | the reward from the grader or reward model | 1.0 |
| $R'(a)$ | the reward after the leash's fee | 0.931 |
| $\beta$ | leash strength: how much a unit of drift costs | 0.1 |
| $\pi_{\text{ref}}(a)$ | the frozen reference model's probability for the answer | 0.3 |
| $\log\frac{\pi_\theta(a)}{\pi_{\text{ref}}(a)}$ | how much more (positive) or less (negative) likely the policy makes this answer than the reference | ln 2 = 0.693 |
| $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$ | **KL divergence**: the average of that log ratio over the policy's own answers; zero only when the two agree (decoded in `primer.ml.training_stages`) | |

**In words:** "each answer's reward is docked in proportion to how much
more likely the policy has made it than the reference did; on average, that
fee is β times the KL divergence between the two."

**With the numbers:** 1.0 − 0.1 × ln 2 = 0.931. Had the policy *halved* the
answer's probability instead (0.15), the log ratio would be −0.693 and the
reward would rise to 1.069: the leash pulls both ways.

**In Python:**

```python
import math
R, beta = 1.0, 0.1
pi, pi_ref = 0.6, 0.3
# R' = R − β log(π / π_ref)
round(R - beta * math.log(pi / pi_ref), 3)  # → 0.931
round(R - beta * math.log(0.15 / pi_ref), 3)  # → 1.069
```

**Why it matters in practice.** This is the "penalty for drifting" in the
RLHF loop of `primer.ml.training_stages`, and the same β appears in DPO. It
stops the policy from forgetting fluent language while it chases reward, and
it is the first line of defence against reward hacking, below.

**In code:** `kl_penalised_reward` is R′; `clipped_policy_update` adds the leash's gradient when given a reference policy and a β, and `primer.ml.training_stages.kl_divergence` computes the KL itself.

## GRPO: compare answers to the same question

**Everyday picture.** Instead of hiring an examiner to predict how hard
each exam question is, a teacher gives the same question to eight students
and marks each answer relative to the others on *that* question. On an easy
question, getting it right is expected and earns little credit; on a hard
one, the only right answer stands out.

PPO's baseline comes from the value network, a second model as large as the
policy that has to be trained alongside it. **GRPO (Group Relative Policy
Optimization)** throws the value network away. For each prompt it samples
a group of answers and uses the group's own average as the baseline.

**Tiny worked example.** The prompt is "3 + 4 =". The model samples four
answers and a checker scores them 1 if the answer is 7, else 0.

| Group rewards | Mean | Std | Advantages |
|---|---|---|---|
| 1, 0, 0, 1 | 0.5 | 0.5 | **+1, −1, −1, +1** |
| 1, 0, 0, 0 | 0.25 | 0.433 | **+1.73, −0.58, −0.58, −0.58** |
| 1, 1, 1, 1 | 1 | 0 | **0, 0, 0, 0** |

A lone right answer in a mostly wrong group earns a big advantage: it's
rare, so it's strong evidence. A group that is all right (or all wrong)
earns nothing: there is no contrast, so there is nothing to learn from.

$$
A_i = \frac{r_i - \operatorname{mean}(r_1, \ldots, r_G)}{\operatorname{std}(r_1, \ldots, r_G)}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $G$ | the group size: answers sampled per prompt | 4 |
| $i$ | which answer in the group | 1 … 4 |
| $r_i$ | the reward for answer $i$ | 1, 0, 0, 0 |
| $\operatorname{mean}(\ldots)$ | the group's average reward: the baseline | 0.25 |
| $\operatorname{std}(\ldots)$ | the group's **standard deviation**, the typical distance from the mean (square root of the variance) | 0.433 |
| $A_i$ | answer $i$'s advantage, shared by every token of that answer | +1.73 |

**In words:** "an answer's advantage is how far its reward sits above the
group's average, measured in units of the group's spread."

**With the numbers:** for (1, 0, 0, 0): mean 0.25, variance
(0.75² + 3 × 0.25²) / 4 = 0.1875, std √0.1875 = 0.433, so the right answer
gets 0.75 / 0.433 = 1.73 and each wrong one −0.25 / 0.433 = −0.58.

**In Python:**

```python
import statistics
def advantages(r):
    mu, sd = statistics.mean(r), statistics.pstdev(r)
    return [round((r_i - mu) / sd, 2) if sd else 0.0 for r_i in r]
advantages([1, 0, 0, 1])  # → [1.0, -1.0, -1.0, 1.0]
advantages([1, 0, 0, 0])  # → [1.73, -0.58, -0.58, -0.58]
advantages([1, 1, 1, 1])  # → [0.0, 0.0, 0.0, 0.0]
```

The rest of GRPO is PPO: the same ratio, the same clip, the same KL leash to
a reference model, averaged over the group. (Some implementations divide by
the sample standard deviation, with G − 1, rather than the population one;
the idea is identical.)

```mermaid
flowchart LR
  subgraph PPO["PPO"]
    P1[Prompt] --> A1[one answer]
    A1 --> RM1[reward]
    A1 --> VN[value network<br/>a second big model]
    RM1 --> AD1[advantage = reward − value]
    VN --> AD1
  end
  subgraph GRPO["GRPO"]
    P2[Prompt] --> G1[answer 1] & G2[answer 2] & G3[answer ...] & G4[answer G]
    G1 & G2 & G3 & G4 --> VER[verifier<br/>checks each answer]
    VER --> NORM[normalise within the group<br/>mean and std]
    NORM --> AD2[advantage per answer]
  end
```

**Reading it:** both pipelines end in an advantage per answer, which feeds
the same clipped update. PPO gets its baseline from a value network that
must be trained, stored and run, which is roughly a second copy of the model.
GRPO instead spends that compute on more answers per prompt and lets them
grade each other. The verifier box can be any scorer, but GRPO shines when
it is a program that checks the answer.

### Verifiable rewards, and why GRPO trains reasoning

A **verifiable reward** comes from a check that can't be argued with: does
the arithmetic equal 7, do the unit tests pass, does the proof check. No
learned reward model, so no learned blind spots (see reward hacking, next).
Here is GRPO on a toy "language model" that answers eight addition prompts
with a single digit token. It starts out about 23% accurate and leans
towards off-by-one mistakes.

![GRPO on eight addition prompts: accuracy climbs from 23% to 99% in 60 steps, while the share of prompts whose whole group agreed, and so taught nothing, rises from a few percent to over 80%](figures/primer.ml.reinforcement.grpo.svg)

**Reading it:** the blue line is the model's average chance of answering
correctly, which climbs from 0.23 to 0.99 in 60 steps of 8 answers per
prompt. The grey bars are the share of prompts whose group of 8 answers
were all right or all wrong: a few percent at the start, over 80% by the
end. They grow as the model masters the prompts:
once every answer is right, the advantages are all zero and that prompt has
nothing left to teach. Real GRPO training fights exactly this by filtering
for prompts at the edge of the model's ability.

**Why it matters in practice.** This recipe, a verifier on the final
answer plus GRPO, is how DeepSeek-R1-Zero learned to reason: rewarded only
for correct final answers (and a required format), the model learned by
itself to write longer chains of thought, to check its work and to back
up from mistakes, because those behaviours raised the chance of a correct
final answer. Each token in a long chain of thought shares its answer's
advantage, so the whole chain is reinforced or discouraged together. See
`primer.ml.reasoning` for what that training produces.

**In code:** `group_advantages` is the formula, `verify` is the checker, `pretrained_logits` is the weak starting model, and `train_grpo` runs the loop.

## Reward hacking: the score is not the goal

**Everyday picture.** A school pays tutors by the number of pages of
homework feedback they write. Feedback gets longer, not better. An RL agent
is the most literal-minded employee imaginable: it optimises the number you
wrote down, not the thing you meant. When those two differ, it finds the
difference. This is **reward hacking** (also called **specification
gaming**), and it is Goodhart's law in code: *when a measure becomes a
target, it ceases to be a good measure.*

**Tiny worked example.** The goal is a good answer, and good answers here
are about 4 sentences long. People rated answers of 1 to 4 sentences, and
on those, longer really was better. A reward model fitted to their ratings
learns a straight line: "each sentence is worth 0.19 points". Nobody ever
rated a 10-sentence answer, so nobody told the reward model it was bad:

| Sentences | 1 | 2 | 3 | **4** | 6 | 8 | **10** |
|---|---|---|---|---|---|---|---|
| True quality | 0.44 | 0.75 | 0.94 | **1.00** | 0.75 | 0.00 | −1.25 |
| Reward model | 0.50 | 0.69 | 0.88 | 1.06 | 1.44 | 1.81 | **2.19** |

The reward model's favourite answer, 10 sentences, is the worst one.

$$
\hat r(n) = w_0 + w_1 n,
\qquad
w_1 = \frac{\sum_i (n_i - \bar n)(q_i - \bar q)}{\sum_i (n_i - \bar n)^2},
\qquad
w_0 = \bar q - w_1 \bar n
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $n$ | the answer's length in sentences | 1 … 10 |
| $\hat r(n)$ | the reward model's score for a length-$n$ answer (the hat marks an estimate) | 2.19 at n = 10 |
| $n_i, q_i$ | the rated examples: a length and its true quality | (1, 0.44), …, (4, 1.0) |
| $\bar n, \bar q$ | their averages (the bar means "mean") | 2.5, 0.781 |
| $w_1$ | the fitted slope: points per extra sentence | 0.1875 |
| $w_0$ | the fitted intercept | 0.3125 |

**In words:** "the reward model is the straight line that best fits the
ratings it saw: its slope is how length and quality moved together in the
data, and it passes through the average point."

**With the numbers:** the deviations of length are (−1.5, −0.5, 0.5, 1.5)
and of quality (−0.344, −0.031, 0.156, 0.219). Their products add to 0.9375
and the squared length deviations to 5, so w₁ = 0.1875 and
w₀ = 0.781 − 0.1875 × 2.5 = 0.3125. At n = 10 it scores 2.19.

**In Python:**

```python
n = [1, 2, 3, 4]
q = [1 - ((n_i - 4) / 4) ** 2 for n_i in n]
q  # → [0.4375, 0.75, 0.9375, 1.0]
n_bar, q_bar = sum(n) / 4, sum(q) / 4
w1 = sum((a - n_bar) * (b - q_bar) for a, b in zip(n, q)) / sum((a - n_bar) ** 2 for a in n)
w0 = q_bar - w1 * n_bar
(w0, w1)  # → (0.3125, 0.1875)
# the reward model's score for a 10-sentence answer, and its true quality
(w0 + w1 * 10, 1 - ((10 - 4) / 4) ** 2)  # → (2.1875, -1.25)
```

![True quality peaks at 4 sentences and falls below zero past 8, while the reward model's straight line, fitted on lengths 1 to 4, keeps rising to 2.19 at 10 sentences](figures/primer.ml.reinforcement.length_rewards.svg)

**Reading it:** the shaded strip is the only region anyone rated. Inside
it, the reward model (red) and the truth (blue) agree on the direction:
longer is better. Outside it, the reward model is extrapolating a straight
line into territory it has never seen, while the truth turns over and
dives. Every learned reward model has regions like this, and an optimiser
is a machine for finding them.

```mermaid
flowchart LR
  GOAL[What we want<br/>helpful answers] -->|people rate a sample| DATA[Ratings]
  DATA -->|fit| RM[Reward model<br/>a proxy for the goal]
  RM -->|reward| OPT[RL optimiser]
  OPT --> POL[Policy]
  POL -->|drifts to where<br/>proxy and goal disagree| GAP[Gap:<br/>high reward, low quality]
  KL[KL leash] -.->|limits drift| POL
  VER[Verifiable reward] -.->|no learned gap| OPT
```

**Reading it:** the solid path is how a reward is usually made: the goal is
sampled by people, the ratings train a model, and that model, not the goal,
is what the optimiser sees. The optimiser pushes the policy wherever the
reward is highest, which, once the easy gains are taken, is wherever the
proxy is most wrong. The dotted arrows are the two main defences: a leash
that limits how far the policy can drift from where the ratings were
collected, and a reward that has no learned gap to exploit.

Now train the starting model (which writes 2 or 3 sentences, a little too
short) against each reward:

![Over 300 steps, optimising the flawed reward first raises true quality to 0.93 then drives it down to 0.75, below where it started, while a KL leash holds it near 0.84 and the verifiable reward climbs to 1.0; the right panel shows the best leashed policy's true quality peaking at a moderate KL and collapsing as the leash loosens](figures/primer.ml.reinforcement.reward_hacking.svg)

**Reading it:** on the left, true quality over 300 training steps. Against
the flawed reward with no leash (red), quality *rises* at first, from 0.80
to 0.93, because lengthening a too-short answer genuinely helps. It rests
on 5-sentence answers for a while, then the reward model's pull wins again:
around step 150 the policy jumps to 6 sentences and quality falls to 0.75,
below where it started, while the reward model's score keeps climbing.
That rise-then-fall is the signature of reward hacking, and it is why "the
reward went up" proves nothing. With a KL leash (β = 0.3, orange) the
policy also overshoots a little, but the leash holds it at 0.84. Against the true, verifiable quality (blue) it
reaches 1.0. On the right, the best policy for each leash strength, placed
by how far it strays from the reference (its KL). Near zero KL it barely
moves; at moderate KL true quality peaks; as the leash loosens further, the
proxy score keeps rising and the true quality collapses below zero.

The right-hand panel uses a closed form: the best policy under a KL leash
has an exact formula.

$$
\pi^*(a) = \frac{\pi_{\text{ref}}(a)\, e^{R(a)/\beta}}{Z},
\qquad
Z = \sum_{b} \pi_{\text{ref}}(b)\, e^{R(b)/\beta}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $\pi^*(a)$ | the policy that maximises $\mathbb{E}[R] - \beta\,\mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})$ | (0.731, 0.269) |
| $\pi_{\text{ref}}(a)$ | the reference model's probability for action $a$ | (0.5, 0.5) |
| $R(a)$ | the reward for action $a$ | (1, 0) |
| $\beta$ | leash strength | 1 |
| $e^{R(a)/\beta}$ | a boost that grows with reward; a small β makes it enormous | $e^1 = 2.718$ |
| $b$ | a counter over every action, so $Z$ adds up all of them | |
| $Z$ | the total, so the probabilities add to 1 | 1.859 |

**In words:** "the best leashed policy starts from the reference and
multiplies each action's probability by e to the power reward over β, then
rescales so everything adds to 1."

**With the numbers:** two actions, reference (0.5, 0.5), rewards (1, 0),
β = 1: weights 0.5 × 2.718 = 1.359 and 0.5 × 1 = 0.5, total 1.859, so
π* = (0.731, 0.269). At β = 0.1 the boost is e¹⁰ ≈ 22,026 and π* puts
99.995% on the rewarded action; as β grows, π* returns to the reference.

**In Python:**

```python
import math
ref, R = [0.5, 0.5], [1.0, 0.0]
def best_policy(beta):
    w = [p * math.exp(r / beta) for p, r in zip(ref, R)]
    return [round(w_a / sum(w), 5) for w_a in w]
best_policy(1.0)  # → [0.73106, 0.26894]
best_policy(0.1)  # → [0.99995, 5e-05]
best_policy(100.0)  # → [0.5025, 0.4975]
```

This is the same formula DPO starts from (`primer.ml.training_stages`): it
is why β means the same thing in RLHF and DPO.

**Why it matters in practice.** Reward hacking shows up wherever RL does.
A boat-racing game agent that learned to circle forever collecting bonus
targets instead of finishing the race. RLHF'd chat models that learned
long, confident, flattering answers score well with raters (length bias and
sycophancy). Coding models rewarded for passing tests that learned to edit
or special-case the tests. The defences, strongest first:

1. **Verifiable rewards** where the task allows: run the tests, check the
   answer. A check has no learned blind spot (though a buggy check does).
2. **Better reward models:** rate the policy's *current* outputs and
   retrain, so the reward model sees the regions the policy is exploring;
   use ensembles; penalise known exploits such as length directly.
3. **A KL leash** to keep the policy near the data the reward was fit on.
4. **Watch a held-out measure of the real goal** (human review, a separate
   evaluation set, see `primer.agents.evals`) and stop when it turns down,
   even if the reward is still climbing.

**In code:** `true_quality` is the goal, `fit_reward_model` and `proxy_reward` are the flawed reward, `optimise_lengths` trains against either with an optional leash, and `kl_regularised_optimum` with `leash_sweep` gives the closed-form best policy for each β.

## In 20 seconds

- **RL** learns from a score, not an answer: sample an action from the
  policy, get a reward, make rewarded actions more likely.
- **REINFORCE**: step along reward × ∇ log π(action); for softmax, ∇ log π
  is one-hot minus the probabilities. Unbiased but noisy.
- **Baselines** subtract the typical reward, turning rewards into
  advantages. Same average gradient, far less variance.
- **PPO** reuses each batch for several steps, clips the probability ratio
  to 1 ± ε so no step goes too far, and uses a value network as baseline
  plus a KL leash to a reference model.
- **GRPO** drops the value network: sample a group of answers per prompt
  and normalise rewards within the group. With a verifier as reward, it is
  how reasoning models are trained.
- **Reward hacking**: the policy optimises the reward you wrote, not the
  goal you meant. Defend with verifiable rewards, better reward models, a KL
  leash and a held-out check of the real goal.

## Self-test questions

**How does reinforcement learning differ from supervised learning?**
Supervised learning is given the correct output for every input and
learns to copy it. Reinforcement learning is given only a score for the
output it produced, so it must try things, see how they score and shift
probability towards what scored well. It fits tasks where judging an
answer is easy but writing the perfect one is not.

**What is the log-probability trick, and why is it needed?**
The gradient of expected reward, Σ ∇π(a) R(a), can't be computed without
knowing every action's reward. Rewriting ∇π = π ∇log π turns it into an
average over actions sampled from the policy, E[R ∇log π(a)], so each
sampled action and its reward give an unbiased estimate of the gradient.

**Why does subtracting a baseline not change the expected gradient?**
Because E[b ∇log π(a)] = b ∇ Σ π(a) = b ∇ 1 = 0: probabilities always add to
1, so the baseline's push averages to nothing. It only removes the noise
that comes from rewards being large or all the same sign.

**In PPO, what is the probability ratio, and what does clipping it do?**
The ratio is the current policy's probability of a sampled action divided
by the probability under the policy that sampled it; it measures how far
the policy has moved on that sample. Clipping stops counting movement
beyond 1 ± ε in the direction the advantage favours, so reusing a batch for
several steps can't push the policy far from where the data came from.

**Why does PPO's objective take the minimum of the clipped and unclipped terms?**
To stay pessimistic. Gains are capped once the ratio leaves the band, but
if a step made a bad action more likely, the full penalty still applies, so
the objective never rewards a harmful move.

**How does GRPO get a baseline without a value network?**
It samples several answers to the same prompt and uses their mean reward
as the baseline, dividing by their standard deviation to set the scale.
Each answer is judged against its siblings, which saves training and
serving a second model the size of the policy.

**What happens in GRPO when every answer in a group gets the same reward?**
Every advantage is zero, so that prompt contributes no gradient. Prompts
that are always solved or never solved teach nothing; learning comes from
prompts at the edge of the model's ability.

**What is reward hacking, and why does a KL penalty help against it?**
Reward hacking is the policy maximising the reward as written while the
real goal gets worse, usually by finding inputs where a learned reward
model is wrong. The KL penalty charges the policy for drifting from the
reference model, which keeps it near the kind of outputs the reward model
was trained on, where the reward is still trustworthy.

**Why are verifiable rewards attractive for training reasoning?**
A program that checks the final answer (or runs the tests) has no learned
blind spots to exploit and costs nothing to label, so RL can run for a long
time against it without the reward drifting away from correctness.

## The papers behind this lesson

- **Williams, *Simple statistical gradient-following algorithms for
  connectionist reinforcement learning* (Machine Learning, 1992)**:
  https://link.springer.com/article/10.1007/BF00992696. Introduced
  REINFORCE, the log-probability policy gradient with a baseline.
- **Schulman et al., *Proximal Policy Optimization Algorithms* (2017)**:
  https://arxiv.org/abs/1707.06347. Introduced the clipped probability-ratio
  objective that lets each batch be reused for several safe steps.
- **Ouyang et al., *Training language models to follow instructions with
  human feedback* (InstructGPT, 2022)**: https://arxiv.org/abs/2203.02155.
  Used PPO with a per-token KL penalty to a reference model to tune a
  language model against a learned reward model.
- **Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical
  Reasoning in Open Language Models* (2024)**:
  https://arxiv.org/abs/2402.03300. Introduced GRPO, replacing PPO's value
  network with group-relative advantages.
- **DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
  via Reinforcement Learning* (2025)**: https://arxiv.org/abs/2501.12948.
  Showed GRPO with rule-based, verifiable rewards alone can teach a model to
  produce long, self-checking chains of thought.
- **Gao, Schulman & Hilton, *Scaling Laws for Reward Model
  Overoptimization* (2022)**: https://arxiv.org/abs/2210.10760. Measured how
  true quality rises and then falls as a policy is optimised further against
  a learned reward model.

## Further reading

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd edition, free online): http://incompleteideas.net/book/the-book-2nd.html
- OpenAI, *Spinning Up in Deep RL*: https://spinningup.openai.com/
- Andrej Karpathy, *Deep Reinforcement Learning: Pong from Pixels*: http://karpathy.github.io/2016/05/31/rl/
- Lilian Weng, *Policy Gradient Algorithms*: https://lilianweng.github.io/posts/2018-04-08-policy-gradient/
- Hugging Face TRL, *GRPO Trainer*: https://huggingface.co/docs/trl/grpo_trainer
- Amodei et al., *Concrete Problems in AI Safety* (2016), section on reward hacking: https://arxiv.org/abs/1606.06565
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017): https://arxiv.org/abs/1707.06347
- Shao et al., *DeepSeekMath* (2024), which introduces GRPO: https://arxiv.org/abs/2402.03300
"""

from __future__ import annotations

import numpy as np

from primer._show import banner, say, table, takeaway
from primer.ml.attention import softmax
from primer.ml.training_stages import kl_divergence

# ---------------------------------------------------------------------------
# 1. The setting: a bandit, a policy, and the reward it expects
# ---------------------------------------------------------------------------

ARMS = ("A", "B", "C")
WIN_CHANCES = (0.2, 0.5, 0.8)  # chance each arm pays out 1; C is the one to find


class Bandit:
    """A row of slot machines. Pulling arm k pays `offset + 1` with chance
    `win_chances[k]`, else `offset`.

    The agent never sees `win_chances`: it only sees what each pull pays.
    That hidden-ness is the whole difficulty of reinforcement learning.
    """

    def __init__(self, win_chances, offset: float = 0.0, seed: int = 0):
        self.win_chances = np.asarray(win_chances, dtype=float)
        self.offset = float(offset)
        self.rng = np.random.default_rng(seed)

    @property
    def n_arms(self) -> int:
        return len(self.win_chances)

    def pull(self, arm: int) -> float:
        """Play one arm and return its reward: a noisy, delayed-free score."""
        return self.offset + float(self.rng.random() < self.win_chances[arm])


def expected_reward(probs: np.ndarray, arm_rewards: np.ndarray) -> float:
    """J = Σ_a π(a)·R(a): the average payout of a policy, if you knew every arm's average."""
    return float(np.dot(probs, arm_rewards))


# ---------------------------------------------------------------------------
# 2. Policy gradients: REINFORCE
# ---------------------------------------------------------------------------


def grad_log_prob(logits: np.ndarray, action: int) -> np.ndarray:
    """∇_z log softmax(z)[action] = one_hot(action) − softmax(z).

    Raise the chosen action's logit, lower every logit in proportion to how
    likely it already was. Shape: same as `logits`.
    """
    g = -softmax(logits)
    g[action] += 1.0
    return g


def reinforce_step(logits: np.ndarray, action: int, reward: float, lr: float, baseline: float = 0.0) -> np.ndarray:
    """One REINFORCE update: z ← z + α·(R − b)·∇ log π(action)."""
    return logits + lr * (reward - baseline) * grad_log_prob(logits, action)


def worked_reinforce_step() -> np.ndarray:
    """The lesson's worked example: a uniform policy over three arms picks C,
    earns reward 1, and takes one step at learning rate 0.5.

    Logits (0, 0, 0) → (−1/6, −1/6, 1/3); probabilities (1/3 each) → (0.274, 0.274, 0.452).
    """
    return softmax(reinforce_step(np.zeros(3), action=2, reward=1.0, lr=0.5))


def gradient_estimate_stats(logits: np.ndarray, arm_rewards: np.ndarray, baseline: float = 0.0) -> dict[str, np.ndarray]:
    """The exact mean and variance of the one-sample estimate (R(a) − b)·∇log π(a).

    Rewards here are fixed per arm, so we can enumerate every action instead
    of sampling: each action's estimate, weighted by how often the policy
    picks it. `mean` is the true gradient of expected reward; `variance` is
    how much a single sample swings around it, per logit.
    """
    probs = softmax(logits)
    # One row per action a: (R(a) − b)·∇log π(a). Shape (n_actions, n_logits).
    estimates = np.stack([(r - baseline) * grad_log_prob(logits, a) for a, r in enumerate(arm_rewards)])
    mean = probs @ estimates
    variance = probs @ (estimates - mean) ** 2
    return {"mean": mean, "variance": variance}


def sampled_gradient_variance(bandit: Bandit, n: int = 4000, baseline: float = 0.0, seed: int = 0) -> float:
    """Total variance (summed over logits) of n one-sample gradient estimates,
    taken at the uniform starting policy with real noisy pulls."""
    # The agent's dice get a stream of their own ([seed, 1]), so they never mirror the bandit's.
    rng = np.random.default_rng([seed, 1])
    logits = np.zeros(bandit.n_arms)
    estimates = []
    for _ in range(n):
        a = int(rng.choice(bandit.n_arms, p=softmax(logits)))
        estimates.append((bandit.pull(a) - baseline) * grad_log_prob(logits, a))
    return float(np.var(np.array(estimates), axis=0).sum())


def train_reinforce(bandit: Bandit, steps: int = 500, lr: float = 0.1, baseline: bool = False, seed: int = 0) -> dict:
    """REINFORCE on a bandit, one pull per step.

    With `baseline=True`, each reward is compared with the average of every
    reward seen before it (on the first pull there is no history, so the
    reward is its own baseline and the step is zero).

    Returns `probs` (steps + 1, n_arms), the policy after each step, and
    `rewards` (steps,), what each pull paid.
    """
    # The agent's dice get a stream of their own ([seed, 1]), so they never mirror the bandit's.
    rng = np.random.default_rng([seed, 1])
    logits = np.zeros(bandit.n_arms)
    probs, rewards = [softmax(logits)], []
    total = 0.0
    for t in range(steps):
        a = int(rng.choice(bandit.n_arms, p=softmax(logits)))
        r = bandit.pull(a)
        b = (total / t if t else r) if baseline else 0.0
        logits = reinforce_step(logits, a, r, lr, baseline=b)
        total += r
        rewards.append(r)
        probs.append(softmax(logits))
    return {"probs": np.array(probs), "rewards": np.array(rewards)}


# ---------------------------------------------------------------------------
# 3. PPO: the probability ratio, the clip, and the KL leash
# ---------------------------------------------------------------------------


def ppo_clipped_objective(ratio, advantage, eps: float = 0.2):
    """min(ratio·A, clip(ratio, 1 − ε, 1 + ε)·A), elementwise.

    The min takes the more pessimistic of the two: gains are capped once the
    ratio leaves the band, penalties never are.
    """
    ratio, advantage = np.asarray(ratio, dtype=float), np.asarray(advantage, dtype=float)
    out = np.minimum(ratio * advantage, np.clip(ratio, 1 - eps, 1 + eps) * advantage)
    return float(out) if out.ndim == 0 else out


def clipped_policy_update(
    logits: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    epochs: int = 1,
    lr: float = 0.3,
    eps: float | None = 0.2,
    ref_probs: np.ndarray | None = None,
    beta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Several passes of gradient ascent on the PPO objective over one batch.

    The batch (actions, advantages) was sampled from the policy as it was on
    entry, the "old" policy. Each pass recomputes every sample's ratio
    π_new(a)/π_old(a) and follows the gradient of the clipped objective,
    averaged over the batch. `eps=None` switches the clip off. With
    `ref_probs` and `beta`, the gradient of −β·KL(π ‖ π_ref) is added too.

    Returns (new logits, ratios of shape (epochs, batch) measured at the
    start of each pass).
    """
    logits = logits.astype(float).copy()
    old_logp = np.log(softmax(logits))[actions]  # frozen: what the sampler believed
    history = []
    for _ in range(epochs):
        probs = softmax(logits)
        ratios = np.exp(np.log(probs)[actions] - old_logp)
        history.append(ratios)
        grad = np.zeros_like(logits)
        for a, adv, ratio in zip(actions, advantages, ratios):
            # Outside the band on the side the advantage pushes towards, the clipped
            # term is the min and it is flat: no gradient, no further push.
            clipped = eps is not None and ((adv > 0 and ratio > 1 + eps) or (adv < 0 and ratio < 1 - eps))
            if not clipped:
                # d(ratio·A)/dz = A·ratio·∇log π(a), because d ratio = ratio·d log π.
                grad += adv * ratio * grad_log_prob(logits, a)
        grad /= len(actions)
        if ref_probs is not None and beta:
            # ∇_z KL(π ‖ π_ref) = π ⊙ (log(π/π_ref) − KL): push back towards the reference.
            log_ratio = np.log(probs / ref_probs)
            grad -= beta * probs * (log_ratio - probs @ log_ratio)
        logits = logits + lr * grad
    return logits, np.array(history)


def ppo_drift_experiment(epochs: int = 50, lr: float = 0.3, batch: int = 16, seed: int = 1) -> dict[str, np.ndarray]:
    """Reuse one batch of 16 pulls for 50 passes, with and without the clip.

    Advantages are rewards minus the batch average. Returns the ratios
    (epochs, batch) for each run, and each run's final probabilities.
    """
    rng = np.random.default_rng([seed, 1])  # the agent's own dice, separate from the bandit's
    bandit = Bandit(WIN_CHANCES, seed=seed)
    actions = rng.choice(3, size=batch)
    rewards = np.array([bandit.pull(int(a)) for a in actions])
    advantages = rewards - rewards.mean()
    out: dict[str, np.ndarray] = {"actions": actions, "rewards": rewards}
    for name, eps in (("clipped", 0.2), ("unclipped", None)):
        logits, ratios = clipped_policy_update(np.zeros(3), actions, advantages, epochs=epochs, lr=lr, eps=eps)
        out[name] = ratios
        out[name + "_probs"] = softmax(logits)
    return out


def kl_penalised_reward(reward: float, logp: float, logp_ref: float, beta: float) -> float:
    """R − β·(log π(a) − log π_ref(a)): the per-sample reward RLHF actually optimises."""
    return reward - beta * (logp - logp_ref)


def kl_regularised_optimum(ref_probs: np.ndarray, rewards: np.ndarray, beta: float) -> np.ndarray:
    """The policy that maximises E[R] − β·KL(π ‖ π_ref): π*(a) ∝ π_ref(a)·e^(R(a)/β).

    Computed in log space so a tiny β (a huge R/β) cannot overflow.
    """
    log_w = np.log(ref_probs) + np.asarray(rewards, dtype=float) / beta
    return softmax(log_w)


# ---------------------------------------------------------------------------
# 4. GRPO: group-relative advantages and a verifiable reward
# ---------------------------------------------------------------------------

# Toy "language model" task: one-token answers (the digits 0 to 9) to additions.
ADDITION_PROMPTS = ((1, 2), (2, 2), (3, 1), (4, 3), (2, 5), (3, 3), (1, 7), (4, 5))
DIGITS = np.arange(10)


def group_advantages(rewards: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """A_i = (r_i − mean(r)) / std(r), within one group of answers to the same prompt.

    `eps` keeps an all-equal group (std 0) at exactly zero advantage instead
    of dividing by zero. Population std, so hand calculations are exact.
    """
    rewards = np.asarray(rewards, dtype=float)
    return (rewards - rewards.mean()) / (rewards.std() + eps)


def verify(prompt: tuple[int, int], answer: int) -> float:
    """A verifiable reward: 1 if the answer is the correct sum, else 0. No model, no opinion."""
    a, b = prompt
    return 1.0 if answer == a + b else 0.0


def pretrained_logits(seed: int = 0) -> np.ndarray:
    """A weak starting model: (n_prompts, 10) logits over the digits.

    It leans a little towards the right answer (+1.0) and its neighbours
    (+0.5, the classic off-by-one), with some noise. About 23% accurate.
    """
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 0.3, (len(ADDITION_PROMPTS), len(DIGITS)))
    for i, (a, b) in enumerate(ADDITION_PROMPTS):
        logits[i, a + b] += 1.0
        for near in (a + b - 1, a + b + 1):
            if 0 <= near <= 9:
                logits[i, near] += 0.5
    return logits


def _accuracy(logits: np.ndarray) -> float:
    probs = softmax(logits)
    return float(np.mean([probs[i, a + b] for i, (a, b) in enumerate(ADDITION_PROMPTS)]))


def train_grpo(steps: int = 60, group_size: int = 8, lr: float = 0.5, beta: float = 0.0, epochs: int = 1, seed: int = 0) -> dict:
    """GRPO on the addition prompts.

    Each step, for every prompt: sample `group_size` answers, score them
    with `verify`, turn the scores into group-relative advantages, and apply
    the clipped update (with a KL leash to the starting model when beta > 0).
    No value network anywhere.

    Returns `accuracy` (steps + 1,), the average chance of a correct answer,
    and `no_signal` (steps,), the share of prompts whose group was all right
    or all wrong and so taught nothing.
    """
    rng = np.random.default_rng(seed)
    logits = pretrained_logits()
    ref = softmax(logits.copy())
    accuracy, no_signal = [_accuracy(logits)], []
    for _ in range(steps):
        silent = 0
        for i, prompt in enumerate(ADDITION_PROMPTS):
            answers = rng.choice(DIGITS, size=group_size, p=softmax(logits[i]))
            rewards = np.array([verify(prompt, int(a)) for a in answers])
            silent += rewards.std() == 0
            logits[i], _ = clipped_policy_update(
                logits[i], answers, group_advantages(rewards), epochs=epochs, lr=lr, ref_probs=ref[i], beta=beta
            )
        accuracy.append(_accuracy(logits))
        no_signal.append(silent / len(ADDITION_PROMPTS))
    return {"accuracy": np.array(accuracy), "no_signal": np.array(no_signal)}


# ---------------------------------------------------------------------------
# 5. Reward hacking: a flawed reward model for answer length
# ---------------------------------------------------------------------------

SENTENCES = np.arange(1, 11)  # the "action": how many sentences the answer runs to
RATED_LENGTHS = (1, 2, 3, 4)  # the only lengths people rated when the reward model was fit
# The starting (fine-tuned) model writes 2 or 3 sentences: a little too short.
REFERENCE_LOGITS = -((SENTENCES - 2.5) ** 2) / 6


def true_quality(n):
    """What we actually want: best at 4 sentences, worse either side, below zero past 8."""
    return 1 - ((np.asarray(n, dtype=float) - 4) / 4) ** 2


def fit_reward_model(lengths=RATED_LENGTHS) -> tuple[float, float]:
    """Least-squares line through the true quality at the rated lengths.

    Returns (intercept, slope). On lengths 1 to 4 quality really does rise
    with length, so the line says "longer is better", everywhere.
    """
    x = np.asarray(lengths, dtype=float)
    y = true_quality(x)
    slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
    return float(y.mean() - slope * x.mean()), float(slope)


def proxy_reward(n):
    """The learned reward model's score: a straight line in length, extrapolated far past its data."""
    intercept, slope = fit_reward_model()
    return intercept + slope * np.asarray(n, dtype=float)


def optimise_lengths(reward_fn=proxy_reward, steps: int = 300, lr: float = 2.0, beta: float = 0.0) -> dict:
    """Gradient ascent on E[reward] − β·KL(π ‖ π_ref) over answer lengths.

    Uses the exact expected gradient π ⊙ (R − J) rather than sampled pulls,
    so the curves show what the objective rewards, free of sampling noise.
    Returns per-step `proxy` and `true` (the policy's average proxy reward
    and true quality), `kl` from the reference, and the final `probs`.
    """
    rewards = reward_fn(SENTENCES)
    ref = softmax(REFERENCE_LOGITS)
    logits = REFERENCE_LOGITS.astype(float).copy()
    proxy, true, kl = [], [], []
    for t in range(steps + 1):
        probs = softmax(logits)
        proxy.append(expected_reward(probs, proxy_reward(SENTENCES)))
        true.append(expected_reward(probs, true_quality(SENTENCES)))
        kl.append(kl_divergence(probs, ref))
        if t == steps:
            break
        grad = probs * (rewards - probs @ rewards)
        log_ratio = np.log(probs / ref)
        grad -= beta * probs * (log_ratio - probs @ log_ratio)
        logits = logits + lr * grad
    return {"proxy": np.array(proxy), "true": np.array(true), "kl": np.array(kl), "probs": softmax(logits)}


def leash_sweep(betas=(0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0)) -> list[dict]:
    """For each β, the best policy under the flawed reward with a KL leash of strength β.

    Each row: beta, the policy's average proxy reward and true quality, and
    its KL from the reference. Small β lets the policy run to the reward
    model's blind spot; large β pins it to the reference.
    """
    ref = softmax(REFERENCE_LOGITS)
    rows = []
    for beta in betas:
        best = kl_regularised_optimum(ref, proxy_reward(SENTENCES), beta)
        rows.append(
            dict(
                beta=beta,
                proxy=expected_reward(best, proxy_reward(SENTENCES)),
                true=expected_reward(best, true_quality(SENTENCES)),
                kl=kl_divergence(best, ref),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# 6. Figures (rendered into the HTML docs by `make figures`)
# ---------------------------------------------------------------------------


def figures() -> dict:
    """Plot this lesson's data. matplotlib is imported here, and only here,
    so the lesson itself needs nothing beyond NumPy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, RED, GREEN, ORANGE, MUTED = "#2563eb", "#dc2626", "#059669", "#d97706", "#9ca3af"
    figs = {}

    # --- 1. REINFORCE on the bandit -----------------------------------------
    history = train_reinforce(Bandit(WIN_CHANCES, seed=0), steps=500, lr=0.1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    for k, (name, color) in enumerate(zip(ARMS, (MUTED, ORANGE, BLUE))):
        a1.plot(history["probs"][:, k], color=color, label=f"arm {name} (wins {WIN_CHANCES[k]:.0%})")
    a1.set(xlabel="pull", ylabel="probability of picking the arm", title="The policy finds the best arm", ylim=(0, 1))
    a1.legend(frameon=False)
    window = 50
    moving = np.convolve(history["rewards"], np.ones(window) / window, mode="valid")
    a2.plot(np.arange(window, len(history["rewards"]) + 1), moving, color=BLUE)
    a2.axhline(0.8, color=GREEN, ls="--", label="best possible (always C)")
    a2.axhline(0.5, color=MUTED, ls=":", label="random pulling")
    a2.set(xlabel="pull", ylabel=f"reward, average of last {window}", title="Reward rises as it learns", ylim=(0.3, 0.9))
    a2.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    figs["bandit_learning"] = fig

    # --- 2. Baselines: variance, and what it does to learning ----------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1, 1.4]})
    variances = [
        sampled_gradient_variance(Bandit(WIN_CHANCES, offset=5.0, seed=0), baseline=0.0),
        sampled_gradient_variance(Bandit(WIN_CHANCES, offset=5.0, seed=0), baseline=5.5),
    ]
    a1.bar(["no baseline", "baseline 5.5"], variances, color=[RED, BLUE])
    for x, v in enumerate(variances):
        a1.text(x, v * 1.15, f"{v:.2f}", ha="center")
    a1.set_yscale("log")
    a1.set(ylabel="variance of one-pull estimate", title="Rewards of 5 or 6: gradient noise", ylim=(0.05, 60))
    rng = np.random.default_rng(0)
    for row, (baseline, color, label) in enumerate(((False, RED, "no baseline"), (True, BLUE, "running-average baseline"))):
        finals = [train_reinforce(Bandit(WIN_CHANCES, offset=5.0, seed=s), 400, 0.1, baseline=baseline, seed=s)["probs"][-1][2] for s in range(20)]
        a2.scatter(finals, row + rng.uniform(-0.15, 0.15, len(finals)), color=color, alpha=0.8)
    a2.set_yticks([0, 1], ["no baseline", "baseline"])
    a2.set(xlabel="final probability of the best arm, C", title="20 runs each, 400 pulls", xlim=(-0.05, 1.05), ylim=(-0.6, 1.6))
    fig.tight_layout()
    figs["baselines"] = fig

    # --- 3. PPO: the clipped objective and ratio drift -----------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    ratios = np.linspace(0.4, 1.8, 300)
    for adv, color in ((1.0, BLUE), (-1.0, RED)):
        a1.plot(ratios, ratios * adv, color=color, ls="--", alpha=0.5)
        a1.plot(ratios, ppo_clipped_objective(ratios, adv), color=color, label=f"advantage {adv:+.0f}")
    a1.axvspan(0.8, 1.2, color=MUTED, alpha=0.2, label="band 1 ± 0.2")
    a1.set(xlabel="probability ratio  π_new / π_old", ylabel="objective for one sample", title="The clip: flat outside the band")
    a1.legend(frameon=False, loc="upper left")
    drift = ppo_drift_experiment()
    a2.plot(drift["unclipped"].max(axis=1), color=RED, label="no clip")
    a2.plot(drift["clipped"].max(axis=1), color=BLUE, label="clip ε = 0.2")
    a2.axhline(1.2, color=MUTED, ls="--")
    a2.set(xlabel="pass over the same 16 pulls", ylabel="largest ratio in the batch", title="Reusing one batch 50 times")
    a2.legend(frameon=False)
    fig.tight_layout()
    figs["ppo_clip"] = fig

    # --- 4. GRPO on the addition prompts ------------------------------------
    run = train_grpo(steps=60, seed=0)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(np.arange(1, 61), run["no_signal"], color=MUTED, alpha=0.6, label="prompts whose group all agreed (no signal)")
    ax.plot(np.arange(61), run["accuracy"], color=BLUE, lw=2, label="chance of a correct answer")
    ax.set(xlabel="GRPO step (8 answers per prompt)", ylabel="share", title="GRPO with a verifier: 8 addition prompts", ylim=(0, 1.35))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.legend(frameon=False, loc="upper left")
    figs["grpo"] = fig

    # --- 5. The flawed reward model -----------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    fine = np.linspace(1, 10, 200)
    ax.axvspan(min(RATED_LENGTHS), max(RATED_LENGTHS), color=MUTED, alpha=0.2, label="lengths people rated")
    ax.plot(fine, true_quality(fine), color=BLUE, label="true quality")
    ax.plot(fine, proxy_reward(fine), color=RED, label="reward model (a fitted line)")
    ax.plot(RATED_LENGTHS, true_quality(np.array(RATED_LENGTHS)), "o", color=BLUE)
    ax.axhline(0, color="#4b5563", lw=0.8)
    ax.set(xlabel="answer length (sentences)", ylabel="score", title="The reward model extrapolates; the truth turns over")
    ax.legend(frameon=False, loc="lower left")
    figs["length_rewards"] = fig

    # --- 6. Reward hacking during training, and the leash sweep --------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    for label, reward_fn, beta, color in (
        ("flawed reward, no leash", proxy_reward, 0.0, RED),
        ("flawed reward, KL leash β = 0.3", proxy_reward, 0.3, ORANGE),
        ("verifiable reward (the true goal)", true_quality, 0.0, BLUE),
    ):
        a1.plot(optimise_lengths(reward_fn, beta=beta)["true"], color=color, label=label)
    a1.set(xlabel="training step", ylabel="true quality of the policy", title="Rise, then fall: reward hacking", ylim=(0.6, 1.02))
    a1.legend(frameon=False, loc="lower left", fontsize=8)
    rows = leash_sweep(betas=np.geomspace(0.04, 20, 40))
    kls = [r["kl"] for r in rows]
    a2.plot(kls, [r["proxy"] for r in rows], color=RED, label="reward model's score")
    a2.plot(kls, [r["true"] for r in rows], color=BLUE, label="true quality")
    a2.set_xscale("log")
    a2.set(xlabel="KL from the reference (looser leash →)", ylabel="average score", title="Best policy for each leash strength")
    a2.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    figs["reward_hacking"] = fig

    return figs


# ---------------------------------------------------------------------------
# 7. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. A bandit: three slot machines, one hidden best")
    say(
        """
        Arms A, B and C pay 1 with chance 20%, 50% and 80%, else 0. The agent
        is not told this. A policy that picks uniformly earns 0.5 per pull on
        average; always picking C earns 0.8.
        """
    )
    table(
        ["policy", "expected reward J"],
        [("uniform", expected_reward(np.full(3, 1 / 3), np.array(WIN_CHANCES))), ("always C", expected_reward(np.array([0, 0, 1.0]), np.array(WIN_CHANCES)))],
        floatfmt=".2f",
    )

    banner("2. REINFORCE: one step, by hand")
    say("Uniform logits (0, 0, 0). The agent pulls C and wins: reward 1. Step = 0.5 × 1 × (one-hot − π).")
    table(["arm", "∇ log π(C)", "new probability"], zip(ARMS, grad_log_prob(np.zeros(3), 2), worked_reinforce_step()), floatfmt=".3f")
    history = train_reinforce(Bandit(WIN_CHANCES, seed=0), steps=500, lr=0.1)
    say(
        f"""
        Now 500 pulls. Reward over the first 100: {history['rewards'][:100].mean():.2f};
        over the last 100: {history['rewards'][-100:].mean():.2f}. Final policy:
        """
    )
    table(["arm", "win chance", "final probability"], zip(ARMS, WIN_CHANCES, history["probs"][-1]), floatfmt=".3f")
    takeaway("Step along reward × ∇ log π(action): rewarded actions become more likely, and on average the best arm wins.")

    banner("3. Baselines: same average gradient, far less noise")
    for b in (0.0, 11.0):
        stats = gradient_estimate_stats(np.zeros(2), np.array([10.0, 12.0]), baseline=b)
        say(f"Rewards 10 and 12, baseline {b:g}: mean gradient for arm 2 = {stats['mean'][1]:+.2f}, variance = {stats['variance'][1]:.2f}")
    wrong = {
        bl: np.mean([train_reinforce(Bandit(WIN_CHANCES, offset=5.0, seed=s), 400, 0.1, baseline=bl, seed=s)["probs"][-1][2] < 0.5 for s in range(20)])
        for bl in (False, True)
    }
    say(
        f"""
        Offset every reward by 5. Of 20 training runs, {wrong[False]:.0%} lock onto
        a worse arm without a baseline, and {wrong[True]:.0%} with a running-average baseline.
        """
    )
    takeaway("Subtract the typical reward: the advantage says 'better or worse than usual', which is the only thing that matters.")

    banner("4. PPO: the probability ratio and the clip")
    table(
        ["ratio", "advantage", "ratio × A", "clipped objective"],
        [(r, a, r * a, ppo_clipped_objective(r, a)) for r, a in ((1.1, 3.0), (1.5, 2.0), (0.5, -1.0), (1.5, -1.0))],
        floatfmt=".2f",
    )
    drift = ppo_drift_experiment()
    say(
        f"""
        One batch of 16 pulls, reused for 50 passes. Without the clip the largest
        ratio reaches {drift['unclipped'].max():.2f} and C's probability {drift['unclipped_probs'][2]:.2f}
        from sixteen pulls. With it, the ratio stops at {drift['clipped'].max():.2f} and C
        sits at {drift['clipped_probs'][2]:.2f}.
        """
    )
    say(f"KL leash: reward 1.0 for an answer at probability 0.6 vs reference 0.3, β = 0.1 → {kl_penalised_reward(1.0, np.log(0.6), np.log(0.3), 0.1):.3f}")
    takeaway("PPO reuses expensive samples for several steps, and the clip keeps each batch from pulling the policy too far.")

    banner("5. GRPO: advantages from a group of answers, no value network")
    table(
        ["group rewards", "advantages"],
        [(str(r), np.array2string(group_advantages(np.array(r, dtype=float)), precision=2)) for r in ([1, 0, 0, 1], [1, 0, 0, 0], [1, 1, 1, 1])],
    )
    run = train_grpo(steps=60, seed=0)
    say(
        f"""
        GRPO with a verifier on 8 addition prompts, 8 answers each: accuracy goes from
        {run['accuracy'][0]:.0%} to {run['accuracy'][-1]:.0%} in 60 steps. By the end,
        {run['no_signal'][-10:].mean():.0%} of groups are unanimous and teach nothing.
        """
    )
    takeaway("Compare each answer with its siblings: the group mean is the baseline, and a verifier is the reward.")

    banner("6. Reward hacking: a reward model that loves length")
    intercept, slope = fit_reward_model()
    say(f"Fitted on answers of 1 to 4 sentences, the reward model is {intercept:.4f} + {slope:.4f} × sentences.")
    table(
        ["sentences", "true quality", "reward model"],
        [(int(n), true_quality(n), proxy_reward(n)) for n in (1, 2, 3, 4, 6, 8, 10)],
        floatfmt=".2f",
    )
    free, leashed, true_run = optimise_lengths(), optimise_lengths(beta=0.3), optimise_lengths(true_quality)
    table(
        ["training against", "true quality: start", "peak", "end"],
        [
            (name, r["true"][0], r["true"].max(), r["true"][-1])
            for name, r in (("flawed reward", free), ("flawed reward + KL β=0.3", leashed), ("verifiable reward", true_run))
        ],
        floatfmt=".2f",
    )
    say(
        f"""
        Against the flawed reward, quality rises while lengthening helps, then falls below
        where it started as the policy settles on {SENTENCES[free['probs'].argmax()]}-sentence answers,
        even as the reward model's score climbs from {free['proxy'][0]:.2f} to {free['proxy'][-1]:.2f}.
        """
    )
    takeaway(
        "The policy optimises the reward you wrote, not the goal you meant. Prefer verifiable rewards, "
        "keep a KL leash, and watch a held-out measure of the real goal."
    )


if __name__ == "__main__":
    demo()
