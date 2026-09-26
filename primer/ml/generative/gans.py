r"""
# GANs: a forger against a detective

Run: `python -m primer.ml.generative.gans`

## The everyday picture

A forger wants to print banknotes that pass as real. A detective wants to
catch every fake. At first the forger is hopeless: smudged ink, the wrong
colour, and the detective spots every note. But every time the detective
rejects a note, the forger learns *what gave it away* and fixes that. Every
time the forger improves, the detective has to look more closely. Neither
is ever told what a real banknote should look like. They only push against
each other, and both keep getting better.

The contest ends when the forger's notes are so good that the detective can
do no better than guess: "real" half the time, "fake" the other half. At
that point the forger has learned to make banknotes, and nobody ever wrote
down a rule for what a banknote is.

That is a **generative adversarial network**, a **GAN**. The forger is a
neural network called the **generator**; the detective is a second network
called the **discriminator**. Train them against each other on photos of
faces and the generator learns to produce new faces that never existed.

This lesson builds both players from scratch in NumPy, trains them on a toy
picture you can see at a glance, and then shows the three ways the contest
goes wrong: the players chase each other in circles, the forger settles for
copying a few examples (**mode collapse**), and the detective wins so
completely that the forger stops learning. Each failure gets a fix you can
run. The same goal, making new samples, is reached differently by
`primer.ml.generative.autoencoders` (squeeze and rebuild) and
`primer.ml.generative.diffusion` (remove noise a little at a time).

## The two players

**Everyday picture.** The forger is a recipe that turns a few dice rolls
into a banknote: different rolls, a different note. The detective is a
machine you feed a note into, and out comes a single number, how sure it is
that the note is real.

**Tiny worked example.** Our "banknotes" are points on a flat page. The real
data is a **ring of eight little clouds**: eight centres evenly spaced on a
circle of radius 2, each real point scattered a tiny amount (a spread of
0.05) around one centre, chosen at random. This toy picture comes from the
research literature on GAN failures because you can *see* whether a forger
has found all eight clouds.

- The **generator** G takes two random numbers z (the dice rolls, drawn from
  a bell curve) and returns one point on the page. It is a small neural
  network: 2 numbers in, two layers of 32 tanh units, 2 numbers out.
- The **discriminator** D takes a point and returns a probability that it is
  real. It is another small network: 2 numbers in, 64 tanh units, one
  **score** out, then squashed into the range 0 to 1.

The squashing is the **sigmoid**, the same one `primer.ml.neural_net` uses to
turn a score into a probability:

$$
D(x) = \sigma\big(a(x)\big) = \frac{1}{1 + e^{-a(x)}}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $x$ | a point on the page, real or forged | $(2, 0)$ |
| $a(x)$ | the detective's raw **score** for $x$ (also called a **logit**): any number, large and positive for "surely real" | $a(x) = x_1 - 1$ |
| $x_1$ | the first coordinate of $x$ | 2 |
| $e$ | Euler's number, about 2.718 (see `primer.notation`) | |
| $\sigma$ | the sigmoid: squashes any score into a probability between 0 and 1 | $\sigma(1) = 0.731$ |
| $D(x)$ | the detective's verdict: the probability that $x$ is real | 0.731 |

**In words:** "the detective computes a score for the point, and the sigmoid
turns the score into a probability that the point is real."

**With the numbers:** take a detective with a single neuron whose score is
$a(x) = x_1 - 1$. The point $(2, 0)$, one of the eight real centres, scores
$2 - 1 = 1$, and $\sigma(1) = 1 / (1 + e^{-1}) = 0.731$: probably real. The
point $(0, 0)$ in the middle of the ring scores $-1$ and gets
$\sigma(-1) = 0.269$: probably fake.

**In Python:**

```python
import math
def sigma(a):
    return 1 / (1 + math.exp(-a))
# a one-neuron detective: score a(x) = x_1 - 1
def a(x):
    return x[0] - 1
round(sigma(a((2, 0))), 3)  # → 0.731
round(sigma(a((0, 0))), 3)  # → 0.269
```

```mermaid
flowchart LR
  Z["noise z<br/>2 random numbers"] --> G["Generator G<br/>the forger"]
  G --> F["fake point G(z)"]
  R["real point x<br/>from the ring"] --> D["Discriminator D<br/>the detective"]
  F --> D
  D --> P["D(point)<br/>probability it is real"]
  P -. "learns to call real real<br/>and fake fake" .-> D
  P -. "learns to make D say real<br/>about its fakes" .-> G
```

**Reading it:** start at the far left. Noise goes into the generator and
comes out as a fake point. Real points come in from the data. Both kinds of
point go through the same detective, which gives each one a probability.
The two dotted arrows are the learning signals, and they pull in opposite
directions: the detective adjusts itself to score real points high and fakes
low, and the generator adjusts itself to make the detective score *its*
points high. Notice that the generator never sees a real point. Everything
it learns about the data arrives through the detective's verdicts.

**Why it matters:** that last point is the whole trick. Nobody writes down
what a face, a voice or a banknote is. The detective discovers what separates
real from fake, and its gradient (the direction that would make it less
sure a fake is fake; see `primer.ml.neural_net` for gradients) tells the
forger how to improve.

**In code:** `ring_of_gaussians` draws real points around `ring_modes`, `MLP` is both players (its `MLP.backward` also returns the gradient with respect to the input, the channel through which the forger learns), with shapes `GENERATOR_SIZES` and `DISCRIMINATOR_SIZES`, and `sigmoid` turns the detective's score into a probability.

## The game: one number both players fight over

**Everyday picture.** Picture a scoreboard. The detective earns points for
confident, correct calls and loses points, heavily, for confident mistakes.
The detective wants the score as high as possible. The forger wants it as
low as possible. One number, two players pulling it in opposite directions:
that is a **minimax** game.

**Tiny worked example.** The detective looks at two real notes and two
fakes. The score uses the **logarithm** (log): log 1 = 0, and the log of a
small number is a large negative number, so a confident mistake costs far
more than a hesitant one (see `primer.notation` for logs from scratch).

| Note | Real? | Verdict D | What is scored | Value |
|---|---|---|---|---|
| 1 | real | 0.9 | log D = log 0.9 | −0.105 |
| 2 | real | 0.8 | log D = log 0.8 | −0.223 |
| 3 | fake | 0.2 | log (1 − D) = log 0.8 | −0.223 |
| 4 | fake | 0.4 | log (1 − D) = log 0.6 | −0.511 |

The average over the real notes is −0.164, over the fakes −0.367, and the
total is **−0.53**. A perfect detective (1 on every real note, 0 on every
fake) would score log 1 + log 1 = 0, the ceiling. A detective reduced to a
coin flip (0.5 on everything) scores log 0.5 + log 0.5 = −1.386.

$$
\min_G \, \max_D \; V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}\big[\log D(x)\big] + \mathbb{E}_{z \sim p_z}\big[\log\big(1 - D(G(z))\big)\big]
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $V(D, G)$ | the **value**: the scoreboard number | −0.53 |
| $\max_D$ | "the detective chooses its weights to make what follows as large as possible" | |
| $\min_G$ | "the forger chooses its weights to make that best-case value as small as possible" | |
| $\mathbb{E}$ | **expectation**: the average over many draws (see `primer.notation`, probability notation) | an average over 2 notes |
| $x \sim p_{\text{data}}$ | "$x$ drawn from the real data" | notes 1 and 2 |
| $z \sim p_z$ | "$z$ drawn from the noise the forger starts from" | the dice rolls behind notes 3 and 4 |
| $D(x)$ | the verdict on a real point | 0.9, 0.8 |
| $G(z)$ | a fake point made from noise $z$ | notes 3 and 4 |
| $D(G(z))$ | the verdict on a fake | 0.2, 0.4 |
| $\log$ | natural logarithm: 0 at 1, very negative near 0 | $\log 0.6 = -0.511$ |

**In words:** "average the log of the verdicts on real points, add the
average log of one minus the verdicts on fakes; the detective pushes this
number up, the forger pushes it down."

**With the numbers:** (log 0.9 + log 0.8) / 2 + (log 0.8 + log 0.6) / 2 =
−0.164 + (−0.367) = −0.53.

**In Python:**

```python
import math
# D(x) on two real notes, D(G(z)) on two fakes
d_real, d_fake = [0.9, 0.8], [0.2, 0.4]
# E over real x of log D(x): an average
real_term = sum(math.log(d) for d in d_real) / len(d_real)
round(real_term, 3)  # → -0.164
# E over noise z of log(1 - D(G(z)))
fake_term = sum(math.log(1 - d) for d in d_fake) / len(d_fake)
round(fake_term, 3)  # → -0.367
round(real_term + fake_term, 2)  # → -0.53
# a coin-flip detective, D = 1/2 on everything
round(math.log(0.5) + math.log(0.5), 3)  # → -1.386
```

If this looks familiar, it should: −V is exactly the **binary
cross-entropy** of a classifier that labels real points 1 and fakes 0 (see
`primer.ml.losses`). The detective is an ordinary classifier. What is new
is that its second class, the fakes, keeps changing underneath it.

Training alternates. Nobody can solve "min over G of max over D" directly,
so each round takes one small step for each player:

```mermaid
flowchart TB
  S["sample a batch of real points<br/>and a batch of noise"] --> F["forger makes fakes G(z)"]
  F --> DS["detective step:<br/>climb V, so D(real) rises<br/>and D(fake) falls"]
  DS --> GS["forger step:<br/>change G so the new detective<br/>scores its fakes higher"]
  GS --> CHK{"trained enough?"}
  CHK -- no --> S
  CHK -- yes --> OUT["keep G;<br/>throw D away"]
```

**Reading it:** follow one lap from the top. A fresh batch of real points
and noise comes in, the forger turns the noise into fakes, and then the two
players move in turn: first the detective takes one step of gradient
**ascent** on V (it wants V higher), then the forger takes one step of
gradient **descent** (it wants V lower), judged by the detective as it is
*after* its step. The loop repeats thousands of times. At the bottom, only
the generator is kept. The detective was scaffolding: its whole job was to
teach.

**Why it matters:** each player is trained with an ordinary optimizer on an
ordinary loss, but the loss surface moves every time the other player steps.
That is the root of everything that goes wrong later in this lesson.

**In code:** `value_function` computes V from a batch of verdicts, and `train_gan` runs the loop above on the ring, one detective step then one forger step, each with Adam (`primer.ml.optimizers.Adam`).

## The best possible detective, and where the game ends

**Everyday picture.** Suppose that, at one spot on the page, real points
turn up three times as often as fakes. However clever the detective is, it
cannot tell two identical-looking points apart, so the best it can do there
is to say "75% real". It should match the local mix, no more and no less.

**Tiny worked example.** At a point where the real data's **density** (how
thickly its points cover that spot) is 0.3 and the forger's is 0.1, the
best verdict is 0.3 / (0.3 + 0.1) = 0.75. Try its neighbours: the
detective's expected score at that spot is
0.3 · log D + 0.1 · log(1 − D), which is −0.2274 at D = 0.7, −0.2249 at
D = 0.75 and −0.2279 at D = 0.8. The middle one is the highest.

$$
D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $D^*(x)$ | the best possible verdict at $x$, for a forger that is held fixed | 0.75 |
| $p_{\text{data}}(x)$ | how densely the real data covers the spot $x$ | 0.3 |
| $p_g(x)$ | how densely the forger's fakes cover the spot $x$ | 0.1 |

**In words:** "the best verdict at each point is the share of the points
found there that are real."

**With the numbers:** 0.3 / (0.3 + 0.1) = 0.75. If the forger matched the
data exactly, $p_g = p_{\text{data}}$ everywhere, and every verdict would be
0.3 / (0.3 + 0.3) = 0.5.

**In Python:**

```python
import math
p_data, p_g = 0.3, 0.1
round(p_data / (p_data + p_g), 2)  # → 0.75
# the detective's expected score at this spot, for a few verdicts D
def score(D):
    return p_data * math.log(D) + p_g * math.log(1 - D)
[round(score(D), 4) for D in (0.7, 0.75, 0.8)]  # → [-0.2274, -0.2249, -0.2279]
# a forger that matches the data exactly
round(0.3 / (0.3 + 0.3), 2)  # → 0.5
```

Where does the formula come from? At each point the detective is choosing
one number D to maximise $p_{\text{data}} \log D + p_g \log(1 - D)$. The
slope of that with respect to D is $p_{\text{data}}/D - p_g/(1 - D)$, and
setting the slope to zero gives $D^*$. This is the central result of the
original GAN paper, and it tells you where the game ends. **At the
equilibrium the forger's distribution equals the data's, and the best
detective says 1/2 everywhere**, scoring V = −log 4 ≈ −1.386, the
coin-flip score from the table above. Plug $D^*$ back into V and what
remains is −log 4 plus twice the **Jensen-Shannon divergence** between the
real and forged distributions, a measure of how different two piles of
probability are that is zero only when they are identical. So a forger
facing a perfect detective is really minimising that divergence. Keep that
in mind: it returns as a problem in the fixes section.

![Top: the real data has two bumps and the forger one wide bump; bottom: the best detective's verdict rises to about 0.8 on the real bumps, drops towards 0 where only fakes live, and would be flat at one half if the forger matched](figures/primer.ml.generative.gans.optimal_detective.svg)

**Reading it:** the top panel shows two distributions on a line: the real
data (blue) has two narrow bumps at −2 and +2, and the forger (red) spreads
one wide bump over the middle. The bottom panel is the best verdict
$D^*$ at every position. Over the real bumps it rises to about 0.8,
because most points found there are real (the forger's wide bump still
reaches them). In the middle and at the far edges it falls towards 0,
because only fakes live there. The grey dashed line at 0.5 is what $D^*$
becomes once the forger matches the data: the detective has nothing left to
go on. The shape of the bottom curve is exactly the information the forger
needs, which way to move its mass.

![After training with balanced learning rates, the forger's green points sit on all eight clouds, and the detective's verdict on the ring is close to one half](figures/primer.ml.generative.gans.detective_view.svg)

**Reading it:** this is the real ring after 2,500 rounds of training. The
background colour is the trained detective's verdict at every spot: blue for
"real", red for "fake", white for 0.5. Black dots are real points, green
dots are the forger's. The green points sit on all eight clouds, and the
background there is pale, close to white: at the eight centres the verdict
is between 0.46 and 0.58. That is the equilibrium made visible, the detective
reduced to guessing where the forgery is good. Away from the ring, where
neither real nor fake points ever appear, the formula says nothing, and the
detective's colouring there is arbitrary (deep red in the empty middle).

**Why it matters:** the detective never needs to model what real data looks
like; it only estimates a *ratio* between two densities. That is why GANs
could learn sharp images long before anyone could write down a probability
for an image.

**In code:** `optimal_discriminator` is the formula, `fit_discriminator_table` trains the most flexible detective possible (one free score per point) against a fixed forger and arrives at the same verdicts without being told the formula, and `detective_verdicts` evaluates a trained detective anywhere on the page.

## Why the forger uses a different loss

**Everyday picture.** Early in training, the forger is terrible and the
detective is sure every fake is fake. Now picture a game of "hotter,
colder" where the helper stops talking once you are far enough away: the
further off you are, the quieter the hints. That is the original forger
loss. The fix is a helper who shouts louder the further off you are.

**Tiny worked example.** The detective looks at a fake and is 99% sure it
is fake: D(G(z)) = 0.01. How hard does each loss push the forger?

- The original loss, log(1 − D(G(z))), which the forger minimises: its slope
  with respect to the detective's score is −D = **−0.01**. Almost nothing.
- The **non-saturating** loss, −log D(G(z)), which the forger also
  minimises: its slope is −(1 − D) = **−0.99**. Ninety-nine times more.

When the detective is undecided, D = 0.5, both slopes are −0.5. The two
losses agree when it doesn't matter and differ exactly when it does.

$$
\frac{\partial}{\partial a}\,\log\big(1 - \sigma(a)\big) = -\sigma(a) = -D
\qquad\qquad
\frac{\partial}{\partial a}\,\big(-\log \sigma(a)\big) = -\big(1 - \sigma(a)\big) = -(1 - D)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $a$ | the detective's score for a fake, before the sigmoid | $\log(0.01/0.99) = -4.6$ |
| $\sigma(a)$ | the verdict D on that fake | 0.01 |
| $\frac{\partial}{\partial a}$ | the **derivative** with respect to $a$: how much the loss changes when the score is nudged up a tiny amount (see `primer.notation`) | |
| $\log(1 - \sigma(a))$ | the original, "saturating" forger loss | $\log 0.99$ |
| $-\log \sigma(a)$ | the non-saturating forger loss | $-\log 0.01 = 4.6$ |

**In words:** "the original loss changes by only D when the detective's
score moves, so it goes quiet when D is near 0; the non-saturating loss
changes by 1 − D, so it is loudest exactly when the forger is doing worst."

**With the numbers:** at D = 0.01, the slopes are −0.01 and −0.99.

**In Python:**

```python
import math
def sigma(a):
    return 1 / (1 + math.exp(-a))
# a score that makes the detective 99% sure the fake is fake
a = math.log(0.01 / 0.99)
round(sigma(a), 2)  # → 0.01
# measure each slope by nudging the score a tiny bit either way
h = 1e-6
def slope(loss):
    return (loss(a + h) - loss(a - h)) / (2 * h)
def saturating(a):
    return math.log(1 - sigma(a))
def non_saturating(a):
    return -math.log(sigma(a))
round(slope(saturating), 3)  # → -0.01
round(slope(non_saturating), 3)  # → -0.99
```

Both losses want the same thing, a higher verdict on the fakes, and they
share the same equilibrium. Only the strength of the push differs. The
original GAN paper already recommended the switch, and essentially every
GAN since uses the non-saturating loss or a Wasserstein-style loss (below).

![Left: as the detective grows sure a fake is fake, the original loss flattens out while the non-saturating loss keeps a steady slope; right: giving the detective an 800-step head start shrinks the original loss's gradient to 0.09 while the non-saturating one grows to 15](figures/primer.ml.generative.gans.forger_losses.svg)

**Reading it:** on the left, the x-axis is the detective's score for a
fake: far left means "certainly fake". The red curve, the original loss,
goes flat as you move left, and a flat curve has no slope, so no gradient.
The blue curve, the non-saturating loss, keeps a steady downhill slope all
the way. On the right is a measurement from our ring: the detective trains
alone against a frozen, untrained forger for more and more steps, and we
measure how big a gradient each loss would hand the forger (log scale).
At the start both are about 0.5 to 0.6. After 800 detective steps its
verdict on fakes is 0.003, the original loss's gradient has shrunk to 0.09,
and the non-saturating one has grown to 15: over a hundred times larger.

**Why it matters:** a forger that gets no gradient does not learn, and the
detective's lead only widens. Early in training the detective *always* has
the easy job, so this failure is the default, not an edge case.

**In code:** `generator_logit_gradient` returns both slopes, and `head_start_experiment` gives the detective a head start and measures the forger's gradient under each loss.

## Why training is unstable, part 1: the players chase each other in circles

**Everyday picture.** Two people share an office thermostat. One turns the
heat up whenever it feels cold; the other opens a window whenever it feels
hot. Each reacts to where the room *was*, so the temperature overshoots one
way, then the other, and never settles. A GAN's players do the same: the
forger moves to where the detective currently says "real", and by the time
it arrives, the detective has moved.

**Tiny worked example.** The smallest GAN there is, the **Dirac GAN**, has
one number per player. The real data is a single point at x = 0. The forger
has one number θ and always outputs x = θ. The detective has one number ψ
and scores a point as ψ · x, so D(x) = σ(ψx). Start with the forger at
θ = 1 and a detective with ψ = 0 (it cannot tell anything apart), and let
both take steps of size 1 at the same time:

| Step | σ(ψθ) | θ (forger) | ψ (detective) | Distance from (0, 0) |
|---|---|---|---|---|
| 0 | | 1 | 0 | 1 |
| 1 | σ(0) = 0.5 | 1 | −0.5 | 1.118 |
| 2 | σ(−0.5) = 0.3775 | 0.811 | −0.878 | 1.195 |

After step 1 the detective has learned "bigger x means fake" (ψ turned
negative), and after step 2 the forger has moved towards the data at 0.
Both moves were sensible, yet the pair got *further* from the equilibrium,
the point (0, 0) where the forger sits on the data and the detective is
indifferent.

$$
V(\theta, \psi) = \log \sigma(\psi \cdot 0) + \log\big(1 - \sigma(\psi\theta)\big)
\qquad
\frac{\partial V}{\partial \psi} = -\sigma(\psi\theta)\,\theta
\qquad
\frac{\partial V}{\partial \theta} = -\sigma(\psi\theta)\,\psi
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $\theta$ | the forger's only number: where it puts its fake | 1 |
| $\psi$ | the detective's only number: the slope of its score $\psi x$ | 0 |
| $\psi \cdot 0$ | the detective's score for the real data at $x = 0$: always 0, so $D(0) = 0.5$ | 0 |
| $\sigma(\psi\theta)$ | the verdict on the fake | 0.5 |
| $\frac{\partial V}{\partial \psi}$ | the slope the detective climbs: $\psi \leftarrow \psi + \eta\,\frac{\partial V}{\partial \psi}$ | $-0.5$ |
| $\frac{\partial V}{\partial \theta}$ | the slope the forger descends: $\theta \leftarrow \theta - \eta\,\frac{\partial V}{\partial \theta}$ | 0 |
| $\eta$ | the step size (learning rate) | 1 |
| $\leftarrow$ | "is replaced by" | |

**In words:** "the detective tilts its slope against wherever the fake sits;
the forger slides along whichever way the slope says is more real; each
reacts to the other's old position."

**With the numbers:** step 1: σ(0) = 0.5, so ψ becomes 0 − 1 · 0.5 · 1 =
−0.5 and θ stays 1 + 1 · 0.5 · 0 = 1. Step 2: σ(−0.5) = 0.3775, so θ
becomes 1 + 0.3775 · (−0.5) = 0.811 and ψ becomes −0.5 − 0.3775 · 1 = −0.878.

**In Python:**

```python
import math
def sigma(a):
    return 1 / (1 + math.exp(-a))
theta, psi, lr = 1.0, 0.0, 1.0
s = sigma(psi * theta)
s  # → 0.5
# both move at once, each using the other's old position
theta, psi = theta + lr * s * psi, psi - lr * s * theta
(theta, psi)  # → (1.0, -0.5)
s = sigma(psi * theta)
round(s, 4)  # → 0.3775
theta, psi = theta + lr * s * psi, psi - lr * s * theta
(round(theta, 3), round(psi, 3))  # → (0.811, -0.878)
# distance from the equilibrium (0, 0): 1, then 1.118, then 1.195
round(math.hypot(theta, psi), 3)  # → 1.195
```

```mermaid
flowchart LR
  A["fake sits to the right<br/>of the data"] --> B["detective tilts:<br/>right means fake"]
  B --> C["forger slides left,<br/>overshoots past the data"]
  C --> D["detective tilts back:<br/>left means fake"]
  D --> E["forger slides right,<br/>overshoots again"]
  E --> A
```

**Reading it:** the boxes form a loop because the game has no downhill
direction that both players share. Each arrow is a sensible move for the
player making it, but each player only reacts to the other's current
position, so the forger arrives just after the detective has changed its
mind. The loop is a rotation around the equilibrium, not a descent into it.

![In the plane of the forger's theta against the detective's psi, simultaneous steps spiral outward from the start at (1, 0), and alternating steps circle the equilibrium without ever reaching it](figures/primer.ml.generative.gans.dirac_oscillation.svg)

**Reading it:** the horizontal axis is the forger's θ, the vertical axis
the detective's ψ, and the black cross at (0, 0) is the equilibrium. Both
paths start at the dot, (1, 0), and take 300 steps of size 0.2. The red
path, where both players step at once, spirals *outwards*: after 300 steps
it is 2.6 away from the equilibrium, where it started 1 away. The blue path,
where the forger waits to see the detective's new ψ before it moves, stays
on a closed loop about 1 away forever. Neither ever arrives. Plain gradient
steps do not solve this game even in two dimensions.

**Why it matters:** a real GAN has millions of numbers per player, and this
rotation happens in many directions at once. It shows up as losses that
swing up and down without trending, and samples that keep changing
character instead of steadily improving.

**In code:** `dirac_gan` runs this two-number game with simultaneous or alternating steps.

## Why training is unstable, part 2: mode collapse

**Everyday picture.** The forger discovers that one particular note, say
the twenty, always fools the detective. So it prints nothing but twenties.
The detective eventually learns that twenties are suspicious, so the forger
switches to printing only fifties, and so on. Each forgery is good, but the
forger never produces the full *variety* of real money. Each distinct kind
of real example is a **mode**, and settling on a few of them is **mode
collapse**.

**Tiny worked example.** To measure variety on the ring, generate 1,000
points and count, for each of the eight clouds, how many land within 0.15
of its centre (3 spreads). A cloud is **covered** if it gets at least 20
of the 1,000 (2%; a perfect forger gives each about 125). A point that lands
near any centre counts as **high quality**. After 2,500 rounds with equal
learning rates for both players, the counts for the eight clouds are
(0, 68, 0, 20, 0, 51, 0, 17). Clouds 2, 4 and 6 are covered, cloud 8's 17
falls short, so **3 of 8** modes are covered, and only 156 of the 1,000
points are high quality: the rest are strung between clouds.

```mermaid
stateDiagram-v2
  direction LR
  EvenClouds: forger piles onto clouds 1, 3, 5, 7
  CatchEven: detective learns those clouds are suspicious
  OddClouds: forger jumps to clouds 2, 4, 6, 8
  CatchOdd: detective learns these are suspicious
  EvenClouds --> CatchEven
  CatchEven --> OddClouds
  OddClouds --> CatchOdd
  CatchOdd --> EvenClouds
```

**Reading it:** each box is a phase of training, and the arrows go round in
a circle. The forger does not have to cover every cloud to fool the
detective *today*; it only has to put its points where the detective
currently says "real". The detective then learns those spots are
over-supplied with fakes, the verdict there drops, and the forger's
gradient pushes the whole pile somewhere else. It is the chase from part 1,
now played out across the clouds of the data. The alternation between odd
and even clouds is what our run actually does, pictured next.

![Three rows of four snapshots between steps 1,750 and 2,500: with equal learning rates the green points hop between alternate clouds; with a detective three times faster they cover all eight; with a detective ten times faster they sit on six and never reach the other two](figures/primer.ml.generative.gans.ring_snapshots.svg)

**Reading it:** each panel is the page, grey circles mark the eight real
clouds, and green dots are 1,000 points from the forger at that step (always
made from the same noise, so the panels are comparable). The panel titles
spell the coverage with X for a covered cloud, counting anticlockwise from
the cloud on the right. Top row, equal learning rates: the green points hop
from clouds 1, 3 and 7 at step 1,750 to 2, 4, 6 and 8 at step 2,000, back
to 1, 3 and 5, then to 2, 4 and 6, never holding more than half the ring.
Middle row, a detective that learns three times faster than the forger: all
eight clouds, every snapshot. Bottom row, a detective ten times faster: six
clouds, and the same six at every snapshot. Two clouds stay empty for good.

![Modes covered over training: equal learning rates wander between one and four; a three-times-faster detective reaches all eight by step 1,000 and stays; a ten-times-faster detective locks onto six at step 500](figures/primer.ml.generative.gans.mode_coverage.svg)

**Reading it:** the x-axis is the training step, the y-axis the number of
clouds covered out of eight. The blue line (detective three times faster)
climbs to 8 by step 1,000 and stays there. The red line (equal speeds)
zigzags between 1 and 4, the hopping from the top row above. The amber line
(detective ten times faster) jumps to 6 by step 500 and never moves again.
The same code and the same data give three very different outcomes, and the
only thing that changed is how fast the detective learns.

These runs use one fixed seed. With other seeds the equal-speed run
sometimes finds all eight clouds late on, and the ten-times run sometimes
escapes after a while; the pattern, not the exact counts, is the lesson.

**Why it matters:** mode collapse is the classic GAN failure on real data. A
face generator that has collapsed produces beautiful faces that all look
like the same few people. Quality metrics that look at one sample at a time
can't see it; you have to measure *coverage*, as we did here.

**In code:** `mode_coverage` counts covered clouds and high-quality points, and `train_gan` records it every 250 steps.

## Why training is unstable, part 3: a detective that wins too fast

**Everyday picture.** A detective who learns much faster than the forger
soon rejects everything outside the few spots the forger has already
mastered, with total confidence. Every small experiment the forger tries in
a new direction comes back "more fake", so it retreats to what already works
and stops exploring.

**Tiny worked example.** The bottom row above: with the detective's learning
rate ten times the forger's, after 2,500 steps the counts per cloud are
(199, 181, 117, 0, 132, 160, 72, 0). The six covered clouds are served well:
86% of points are high quality, *more* than the balanced run's 81%. But
clouds 4 and 8 get nothing, from step 500 until the end. Better-looking
samples, less variety: a trade-off you will meet again with real image
generators.

```mermaid
flowchart LR
  A["detective learns<br/>much faster"] --> B["gaps between the forger's clouds<br/>become confident fake zones"]
  B --> C["every small step towards them<br/>lowers the verdict"]
  C --> D["forger polishes the clouds<br/>it already has"]
  D --> E["empty clouds stay empty"]
```

**Reading it:** read it as a chain of causes. A fast detective turns every
region the forger isn't already covering into a wall of confident "fake"
verdicts. To reach an empty cloud the forger would have to move points
*through* such a wall. But a gradient only sees the next small step, and
every small step into the wall makes the verdict worse, so the gradient
pushes the points back onto the clouds they already occupy. The forger
spends its effort sharpening those, and the missing ones are never
discovered.

**Why it matters:** "just train the detective harder" sounds like it should
give the forger a better teacher. Past a point it gives it a worse one. The
detective needs to be good enough to point the way, not so good that it only
says no.

**In code:** `train_gan` takes both learning rates as arguments; this run gives the detective ten times the forger's.

## Fixes that make the game trainable

**Everyday picture.** A good sparring partner matters more than a strong
one. Every stabiliser below is a way of keeping the detective useful: fast
enough to teach, smooth enough that its verdicts point somewhere.

```mermaid
flowchart LR
  P1["players out of step"] --> F1["careful learning rates<br/>two time-scales"]
  P2["detective builds<br/>steep cliffs"] --> F2["gradient penalty"]
  P2 --> F3["spectral normalization"]
  P3["piles don't overlap,<br/>so the verdict gives no direction"] --> F4["Wasserstein distance"]
  F4 --> F2
```

**Reading it:** failures are on the left and the fixes that target them on
the right. Players moving out of step are fixed by choosing their speeds.
A detective whose verdict jumps from 0 to 1 across a narrow cliff is fixed
by penalising or capping its steepness. A yardstick that can't tell "near
miss" from "far miss" is replaced by the Wasserstein distance, which in
practice needs one of the steepness fixes to compute, hence the last arrow.

### Careful learning rates

**Everyday picture.** Pace the two sparring partners so that neither runs
away with the match.

**Tiny worked example.** The three rows of the ring figure differ only in
the detective's learning rate, with the forger's fixed at 0.001:

| Detective's rate | Clouds covered at step 2,500 | High-quality points |
|---|---|---|
| 0.001 (equal) | 3, and hopping | 16% |
| 0.003 (three times) | **8**, steady | 81% |
| 0.01 (ten times) | 6, stuck | 86% |

A modestly faster detective keeps its verdicts close to the best-detective
formula for the forger it is facing *now*, so the forger follows a gradient
that points at the data rather than at yesterday's detective. Heusel and
colleagues called this the **two time-scale update rule** and proved it
reaches an equilibrium under reasonable conditions. The right ratio depends
on the problem; the lesson is that it is a dial worth turning.

**In code:** `train_gan` with a detective learning rate of 0.003 is the balanced run; the other two rows are the same call with 0.001 and 0.01.

### Gradient penalty: fine the detective for steep cliffs

**Everyday picture.** Charge the detective a fee for how sharply its score
changes right on top of the real data. A detective with gentle slopes still
ranks real above fake, but it can no longer swing wildly, and the swinging is
what fed the spiral in part 1.

**Tiny worked example.** In the Dirac GAN the detective's score is ψ · x,
whose slope with respect to x is just ψ. The **R1 gradient penalty** adds
(γ / 2) · ψ² to what the detective minimises. With γ = 1 and ψ = −0.5 the
fee is 0.125, and its gradient, γψ = −0.5, pulls ψ back towards 0 on every
step, like friction. With step size 0.2 that friction shrinks ψ by a factor
1 − 0.2 · 1 = 0.8 per step, on top of the game's own push.

$$
R_1 = \frac{\gamma}{2}\;\mathbb{E}_{x \sim p_{\text{data}}}\Big[\big\lVert \nabla_x\, a(x) \big\rVert^2\Big]
$$

**Symbols**

| Symbol | Meaning here | In the Dirac example |
|---|---|---|
| $R_1$ | the penalty added to the detective's loss | 0.125 |
| $\gamma$ | how heavily steepness is fined | 1 |
| $a(x)$ | the detective's score at $x$ | $\psi x$ |
| $\nabla_x\, a(x)$ | the **gradient** of the score with respect to the input point: which way, and how steeply, the score rises as you move $x$ | $\psi = -0.5$ |
| $\lVert \cdot \rVert^2$ | the squared length of that gradient (see `primer.notation`, the length of a vector) | 0.25 |
| $\mathbb{E}_{x \sim p_{\text{data}}}$ | averaged over real points only | the single real point $x = 0$ |

**In words:** "measure how steeply the detective's score changes at the real
points, square it, average it, and charge the detective γ/2 times that."

**With the numbers:** (1 / 2) · (−0.5)² = 0.125.

**In Python:**

```python
gamma, psi = 1.0, -0.5
# slope of the score psi * x with respect to x, at the real point
slope = psi
gamma / 2 * slope ** 2  # → 0.125
# the penalty's pull on psi: its derivative, gamma * psi
gamma * psi  # → -0.5
# one step of size 0.2 against that pull shrinks psi by a factor 0.8
round(psi - 0.2 * gamma * psi, 2)  # → -0.4
```

![Distance from the equilibrium over 300 steps: simultaneous steps grow from 1 to 2.6, alternating steps hold near 1, and with the R1 penalty the distance falls to 0.08 within 50 steps and to zero after](figures/primer.ml.generative.gans.dirac_r1.svg)

**Reading it:** the x-axis is the step, the y-axis the distance of (θ, ψ)
from the equilibrium (0, 0). The red line (plain simultaneous steps) climbs
in steps, one per lap: the outward spiral. The blue line (alternating steps) hovers near
1 forever: the endless orbit. The green line adds the R1 penalty with γ = 1
to the simultaneous steps and dives: 0.08 after 50 steps, essentially 0 by
100. The friction turns the rotation into a spiral *inwards*. That is the
Mescheder, Geiger and Nowozin result in two numbers, and it is why StyleGAN
and many later GANs train with an R1 penalty. The Wasserstein GAN with
gradient penalty (WGAN-GP) uses the same idea with a different target: it
keeps the slope near 1 at points between real and fake.

**In code:** `dirac_gan` takes the penalty weight γ as an argument and adds the R1 penalty's pull, −γψ, to the detective's step.

### Wasserstein distance: a yardstick that knows near from far

**Everyday picture.** Picture the real data and the fakes as two piles of
sand. The **Wasserstein distance**, also called the **earth mover's
distance**, is the least work needed to reshape one pile into the other:
the amount of sand moved times how far it travels. The Jensen-Shannon
divergence from the best-detective section only asks how much the piles
*overlap*.

**Tiny worked example.** Put all the real sand at x = 0 and all the fake sand
at x = θ. If θ = 1, the piles don't overlap, and the Jensen-Shannon
divergence is log 2 = 0.693. If θ = 5, they still don't overlap, and it is
still 0.693. It cannot tell a near miss from a far one, so it gives the
forger no direction. The Wasserstein distance is 1 for θ = 1 and 5 for
θ = 5: it shrinks as the forger approaches, so it always points home.

For two equally sized samples on a line, the cheapest way to move the sand
is to pair the smallest with the smallest, the next with the next, and so
on:

$$
W_1 = \frac{1}{n} \sum_{i=1}^{n} \big\lvert a_{(i)} - b_{(i)} \big\rvert
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $W_1$ | the Wasserstein (earth mover's) distance between two piles on a line | 3 |
| $n$ | how many grains (samples) each pile has | 3 |
| $a_{(i)}$ | the $i$-th smallest value in the first pile (the brackets mean "after sorting") | $a = (0, 1, 2)$ |
| $b_{(i)}$ | the $i$-th smallest value in the second pile | $b = (5, 3, 4)$ sorts to $(3, 4, 5)$ |
| $\lvert \cdot \rvert$ | the absolute value: the gap, ignoring direction | $\lvert 0 - 3 \rvert = 3$ |

**In words:** "sort both piles, pair them up in order, and average how far
each grain has to travel."

**With the numbers:** pairs (0, 3), (1, 4), (2, 5), gaps 3, 3, 3, average
**3**.

**In Python:**

```python
a = [0, 1, 2]
b = [5, 3, 4]
# pair the smallest with the smallest, and so on
pairs = list(zip(sorted(a), sorted(b)))
pairs  # → [(0, 3), (1, 4), (2, 5)]
sum(abs(x - y) for x, y in pairs) / len(pairs)  # → 3.0
```

![As the forger's pile slides from minus 4 to plus 4, the Jensen-Shannon divergence is flat at log 2 everywhere except exactly 0, while the Wasserstein distance is a V shape pointing at 0](figures/primer.ml.generative.gans.wasserstein_vs_js.svg)

**Reading it:** the x-axis is where the forger's pile sits; the real pile
is at 0. The red line (Jensen-Shannon) is flat at 0.693 everywhere except
the single point θ = 0, where it drops to 0: a flat line has no slope, so
nothing tells the forger which way to move. The blue V (Wasserstein) slopes
down towards 0 from both sides: wherever the forger is, the slope points at
the data. On images, where real photos and early fakes barely overlap, this
difference is the difference between learning and not.

The **Wasserstein GAN** replaces the detective with a **critic** that
outputs an unbounded score instead of a probability, and trains it so that
the gap between its average score on real and on fake points estimates this
distance. The catch: the estimate is only valid if the critic's slope is at
most 1 everywhere. The original WGAN enforced that crudely by clipping every
weight; WGAN-GP does it with a gradient penalty; spectral normalization,
next, does it by capping each layer.

**In code:** `wasserstein_1d` is the sort-and-pair formula, and `js_divergence` computes the Jensen-Shannon divergence between two histograms, flat at log 2 whenever they don't overlap.

### Spectral normalization: a speed limit on every layer

**Everyday picture.** A layer of a network is a matrix, and a matrix
stretches some directions more than others. If no layer can stretch
anything by more than 1, the whole detective can't change its score faster
than the input changes: its cliffs have a speed limit.

**Tiny worked example.** The matrix W = [[3, 0], [0, 1]] stretches the
horizontal direction by 3 and the vertical by 1. Its **largest stretch** is
3. Divide W by 3 and no direction is stretched by more than 1.

$$
\bar W = \frac{W}{\lVert W \rVert_2}
\qquad
\lVert W \rVert_2 = \max_{\lVert v \rVert = 1} \lVert W v \rVert
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $W$ | one layer's weight matrix (see `primer.notation` for matrices) | [[3, 0], [0, 1]] |
| $v$ | an input direction: an arrow of length 1 | $(1, 0)$ |
| $\lVert v \rVert$ | the length of $v$ | 1 |
| $Wv$ | the arrow after passing through the layer | $(3, 0)$ |
| $\max_{\lVert v \rVert = 1}$ | "the largest value over every arrow of length 1" | |
| $\lVert W \rVert_2$ | the **spectral norm**: the most W lengthens any arrow | 3 |
| $\bar W$ | the normalised layer used in the detective | [[1, 0], [0, 1/3]] |

**In words:** "find the direction the layer stretches most, and divide the
whole layer by that stretch."

**With the numbers:** $(1, 0)$ becomes $(3, 0)$, length 3; $(0, 1)$ stays
length 1; every other direction lands in between. So $\lVert W \rVert_2 = 3$,
and $W / 3$ stretches by at most 1.

**In Python:**

```python
import math
W = [[3, 0], [0, 1]]
def stretch(v):
    Wv = [sum(w * x for w, x in zip(row, v)) for row in W]
    return math.hypot(*Wv) / math.hypot(*v)
# try unit arrows all the way round the circle, one per degree
angles = [2 * math.pi * k / 360 for k in range(360)]
round(max(stretch((math.cos(t), math.sin(t))) for t in angles), 3)  # → 3.0
# divide W by its largest stretch
W = [[w / 3 for w in row] for row in W]
round(max(stretch((math.cos(t), math.sin(t))) for t in angles), 3)  # → 1.0
```

Trying every direction is fine for a 2 × 2 matrix but not for a layer with
a thousand inputs. **Power iteration** finds the most-stretched direction
cheaply:

```mermaid
flowchart LR
  V["start: any arrow v"] --> U["u = W v,<br/>rescaled to length 1"]
  U --> B["v = W transpose u,<br/>rescaled to length 1"]
  B --> Q{"repeat"}
  Q -- again --> U
  Q -- done --> S["largest stretch = length of W v"]
```

**Reading it:** push an arrow through the layer and back through its
transpose, over and over. Each round trip multiplies the arrow's
most-stretched component by more than any other, so the arrow swings round
to point along that direction, and the stretch is read off at the end.
Spectral normalization keeps one arrow per layer and runs a single round
trip per training step: the weights change slowly, so the arrow stays
nearly right, and the cost is a couple of extra matrix-vector products.

**Why it matters:** spectral normalization (Miyato and colleagues, 2018) is a
one-line change to the detective that made GAN training far less sensitive
to learning rates and architecture, and it became a default in large image
GANs such as BigGAN.

**In code:** `largest_stretch` is power iteration, and `spectrally_normalize` divides a matrix by it.

## Where GANs stand today

**Everyday picture.** The forger-and-detective idea started as the whole
machine. Today it is more often a picky critic *inside* another machine,
brought in for the one thing it does best: making outputs look sharp and
real.

**Tiny worked example.** A GAN makes an image in **one** pass through the
generator. A diffusion model typically runs its network tens of times per
image, removing a little noise each time (see
`primer.ml.generative.diffusion`). For a while, that speed plus very sharp
samples (StyleGAN's faces, from 2018, are still striking) made GANs the best
image generators. Then diffusion models overtook them on image quality and,
above all, on *coverage*: they don't mode-collapse and they train stably,
which is everything this lesson has been fighting. Since around 2021 most
new image and video generators are diffusion or flow models.

GANs didn't disappear. Their loss did the moving:

```mermaid
flowchart LR
  X["real image or audio"] --> E["encoder"]
  E --> C["compact code"]
  C --> DEC["decoder"]
  DEC --> Y["reconstruction"]
  Y --> L1["reconstruction loss<br/>pixel or perceptual"]
  X --> L1
  Y --> DISC["detective:<br/>real or reconstructed?"]
  X --> DISC
  L1 --> T["total loss for<br/>encoder and decoder"]
  DISC --> T
```

**Reading it:** this is an autoencoder (see
`primer.ml.generative.autoencoders`) with a detective bolted on. The top
path squeezes the input into a compact code and rebuilds it. A
reconstruction loss alone rewards *averages*, and the average of many
plausible textures is a blur. The detective on the lower path looks at the
reconstruction and the original and asks "which is real?", which punishes
blur and rewards crisp detail. The decoder is trained on both losses
together. This design is how the image autoencoders inside latent diffusion
models are trained, how VQGAN's image tokenizer is trained, and how neural
audio decoders such as HiFi-GAN produce clean waveforms. GAN-style losses
are also used to distil a slow many-step diffusion model into a fast one-
or few-step generator.

**Why it matters:** when you meet a modern image, audio or video system,
expect a GAN loss somewhere in its decoder or its fast sampling path, even
when the headline method is diffusion. Everything in this lesson, the
non-saturating loss, gradient penalties, spectral normalization and careful
learning rates, is still how those critics are kept stable.

## In 20 seconds

- **A GAN** trains a generator (forger) to turn noise into samples and a
  discriminator (detective) to tell real samples from fakes, against each
  other. The forger learns only through the detective's gradient.
- **The game:** min over G of max over D of E[log D(x)] + E[log(1 − D(G(z)))].
  The best detective says p_data / (p_data + p_g); at the equilibrium the
  forger matches the data and the detective says 1/2 everywhere.
- **The non-saturating loss** (maximise log D(G(z))) keeps the forger's
  gradient strong when the detective is confident, exactly when the
  original loss goes quiet.
- **Unstable because** each player's target moves: plain gradient steps
  circle or spiral around the equilibrium, the forger hops between a few
  modes (mode collapse), and a detective that wins too fast freezes it.
- **Fixes:** balanced learning rates, gradient penalties (R1, WGAN-GP),
  spectral normalization, and the Wasserstein distance, which still points
  the way when real and fake don't overlap.
- **Today:** diffusion has overtaken GANs for generating images, but
  adversarial losses remain inside image and audio decoders and fast
  distilled samplers.

## Self-test questions

**Explain a GAN to a non-engineer in 30 seconds.**
Two programs play a game. One makes fake pictures; the other looks at real
and fake pictures and guesses which is which. Every time the guesser catches
a fake, the faker learns what gave it away and improves. After enough
rounds the fakes are good enough that the guesser is reduced to guessing,
and the faker has learned to make realistic pictures without anyone ever
describing what a picture should look like.

**Why does the generator never need to see real data?**
It learns only from the discriminator's gradient with respect to its own
outputs: which way to move each fake so the discriminator finds it more
real. The discriminator has seen real data, so its verdicts carry that
information to the generator.

**What does the optimal discriminator compute, and what does it say at equilibrium?**
D*(x) = p_data(x) / (p_data(x) + p_g(x)): the share of points found at x
that are real. When the generator matches the data, p_g = p_data and D* is
1/2 everywhere, the value is −log 4, and the discriminator can do no better
than a coin flip.

**Why do almost all GANs use the non-saturating generator loss?**
The original loss log(1 − D(G(z))) has a slope of −D with respect to the
discriminator's score, which is near zero when the discriminator confidently
rejects fakes, as it does early in training. The non-saturating loss
−log D(G(z)) has slope −(1 − D), largest exactly then. Both have the same
equilibrium.

**What is mode collapse, and how would you detect it?**
The generator covers only some of the distinct kinds of data (modes), often
hopping between them as the discriminator catches up. Per-sample quality can
look excellent, so you detect it by measuring coverage: how many modes or
classes receive samples, or a distribution-level metric such as FID on real
data.

**Why doesn't plain gradient descent find the GAN equilibrium?**
The game has no shared downhill direction. Near the equilibrium the two
players' updates form a rotation, so simultaneous steps spiral outward and
alternating steps orbit forever, as the two-number Dirac GAN shows. Damping
the discriminator with a gradient penalty turns the spiral inward.

**Why is the Wasserstein distance a better training signal than Jensen-Shannon divergence?**
When the real and generated distributions don't overlap, Jensen-Shannon is
stuck at log 2 however far apart they are, so it gives no direction.
Wasserstein measures how far the mass must move, so it shrinks steadily as
the generator approaches the data. Estimating it needs a critic whose slope
is capped, which is what weight clipping, gradient penalties and spectral
normalization provide.

**If diffusion models won, why learn GANs?**
Adversarial losses are still how many image and audio decoders get sharp
output, and how some diffusion models are distilled into one-step
generators. And the instabilities here, moving targets and collapsing
variety, show up wherever two models are trained against each other.

## The papers behind this lesson

- **Goodfellow et al., *Generative Adversarial Nets* (2014)**:
  https://arxiv.org/abs/1406.2661. Introduced the generator-discriminator
  game, derived the optimal discriminator and the equilibrium where the
  generator matches the data, and suggested the non-saturating loss.
- **Metz, Poole, Pfau and Sohl-Dickstein, *Unrolled Generative Adversarial
  Networks* (2016)**: https://arxiv.org/abs/1611.02163. Used the ring of
  eight Gaussians to show a generator hopping between modes, and reduced it
  by letting the generator look ahead at several discriminator steps.
- **Heusel et al., *GANs Trained by a Two Time-Scale Update Rule Converge to
  a Local Nash Equilibrium* (2017)**: https://arxiv.org/abs/1706.08500.
  Showed that separate learning rates for the two players give provable
  convergence, and introduced the FID score for judging generated images.
- **Arjovsky, Chintala and Bottou, *Wasserstein GAN* (2017)**:
  https://arxiv.org/abs/1701.07875. Replaced the Jensen-Shannon objective
  with the earth mover's distance, which still gives a direction when real
  and generated data don't overlap.
- **Gulrajani et al., *Improved Training of Wasserstein GANs* (2017)**:
  https://arxiv.org/abs/1704.00028. Enforced the Wasserstein critic's
  slope limit with a gradient penalty instead of weight clipping.
- **Mescheder, Geiger and Nowozin, *Which Training Methods for GANs do
  actually Converge?* (2018)**: https://arxiv.org/abs/1801.04406.
  Introduced the two-number Dirac GAN to show why plain GAN training
  circles instead of converging, and the R1 penalty that makes it converge.
- **Miyato et al., *Spectral Normalization for Generative Adversarial
  Networks* (2018)**: https://arxiv.org/abs/1802.05957. Capped every
  discriminator layer's largest stretch at 1 using one step of power
  iteration per training step.

## Further reading

- Goodfellow, *NIPS 2016 Tutorial: Generative Adversarial Networks*: https://arxiv.org/abs/1701.00160
- Radford, Metz and Chintala, *Unsupervised Representation Learning with Deep Convolutional GANs* (DCGAN, 2015): https://arxiv.org/abs/1511.06434
- Karras, Laine and Aila, *A Style-Based Generator Architecture for Generative Adversarial Networks* (StyleGAN, 2018): https://arxiv.org/abs/1812.04948
- Dhariwal and Nichol, *Diffusion Models Beat GANs on Image Synthesis* (2021): https://arxiv.org/abs/2105.05233
- Esser, Rombach and Ommer, *Taming Transformers for High-Resolution Image Synthesis* (VQGAN, 2020): https://arxiv.org/abs/2012.09841
- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* (2021): https://arxiv.org/abs/2112.10752
- Kong, Kim and Bae, *HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis* (2020): https://arxiv.org/abs/2010.05646
- Sauer et al., *Adversarial Diffusion Distillation* (2023): https://arxiv.org/abs/2311.17042
- PyTorch, *DCGAN Tutorial*: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""

from __future__ import annotations

import functools
import math

import numpy as np

from primer._show import banner, say, table, takeaway
from primer.ml.optimizers import Adam

# ---------------------------------------------------------------------------
# 1. The toy data: a ring of eight little clouds
# ---------------------------------------------------------------------------

N_MODES, RADIUS, MODE_STD = 8, 2.0, 0.05


def ring_modes(n_modes: int = N_MODES, radius: float = RADIUS) -> np.ndarray:
    """The centres of the clouds: (n_modes, 2) points evenly spaced on a circle, the first on the x-axis."""
    angles = 2 * np.pi * np.arange(n_modes) / n_modes
    return radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def ring_of_gaussians(n: int, seed: int | np.random.Generator = 0, std: float = MODE_STD) -> np.ndarray:
    """n real samples: pick one of the eight modes at random, then add a little round noise. Shape (n, 2)."""
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    centres = ring_modes()[rng.integers(0, N_MODES, n)]
    return centres + std * rng.standard_normal((n, 2))


# ---------------------------------------------------------------------------
# 2. The two players: small multi-layer perceptrons, forward and backward by hand
# ---------------------------------------------------------------------------

GENERATOR_SIZES = (2, 32, 32, 2)  # 2 numbers of noise in, two tanh layers, one 2-D point out
DISCRIMINATOR_SIZES = (2, 64, 1)  # a 2-D point in, one tanh layer, one score (a logit) out


def sigmoid(a):
    """1 / (1 + e^-a), written with tanh so it never overflows for large |a|."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(a, dtype=float)))


class MLP:
    """A stack of layers `x W + b`, with tanh between them and nothing after the last.

    Every weight and bias lives in one flat vector, `params`; `W` and `b` are
    views into it. That lets one optimizer step update the whole network in
    one line, and lets a test nudge any single number. The same recipe as
    `primer.ml.neural_net.MLP`, generalised to any depth and to a backward
    pass that also returns the gradient with respect to the *input*, which is
    the only route by which the forger ever learns anything.
    """

    def __init__(self, sizes: tuple[int, ...], seed: int = 0):
        rng = np.random.default_rng(seed)
        shapes = [(a, b) for a, b in zip(sizes[:-1], sizes[1:])]
        self.params = np.zeros(sum(a * b + b for a, b in shapes))
        self.W, self.b = self._views(self.params, shapes)
        self._shapes = shapes
        for W in self.W:
            # Variance 1/fan_in keeps tanh out of its flat ends at the start (see primer.ml.deep_nets).
            W[...] = rng.normal(0.0, 1.0 / np.sqrt(W.shape[0]), W.shape)

    @staticmethod
    def _views(flat: np.ndarray, shapes) -> tuple[list[np.ndarray], list[np.ndarray]]:
        Ws, bs, i = [], [], 0
        for a, b in shapes:
            Ws.append(flat[i : i + a * b].reshape(a, b))
            i += a * b
            bs.append(flat[i : i + b])
            i += b
        return Ws, bs

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (n, sizes[0]) -> (n, sizes[-1]). Caches every layer's output for `backward`."""
        self._acts = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = self._acts[-1] @ W + b
            self._acts.append(np.tanh(z) if i < len(self.W) - 1 else z)
        return self._acts[-1]

    def backward(self, grad_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Chain rule from d(something)/d(output) back to (every parameter, the input).

        Returns (flat gradient laid out like `params`, gradient w.r.t. the input x).
        """
        grad = np.zeros_like(self.params)
        gW, gb = self._views(grad, self._shapes)
        g = grad_out
        for i in reversed(range(len(self.W))):
            gW[i][...] = self._acts[i].T @ g  # each weight's gradient: its input times the error arriving
            gb[i][...] = g.sum(axis=0)
            g = g @ self.W[i].T  # send the error back through the weights
            if i > 0:
                g = g * (1.0 - self._acts[i] ** 2)  # tanh'(z) = 1 - tanh(z)^2, reusing the cached output
        return grad, g


# ---------------------------------------------------------------------------
# 3. The game: the value function and the best possible detective
# ---------------------------------------------------------------------------


def value_function(d_real, d_fake, eps: float = 1e-12) -> float:
    """V = mean log D(x) over real samples + mean log(1 - D(G(z))) over fakes.

    The detective wants V high (ceiling 0); the forger wants it low. -V is
    exactly the detective's binary cross-entropy with real = 1 and fake = 0.
    """
    d_real = np.clip(np.asarray(d_real, dtype=float), eps, 1.0)
    d_fake = np.clip(np.asarray(d_fake, dtype=float), 0.0, 1.0 - eps)
    return float(np.mean(np.log(d_real)) + np.mean(np.log(1.0 - d_fake)))


def optimal_discriminator(p_data, p_g):
    """D*(x) = p_data(x) / (p_data(x) + p_g(x)): the best any detective can do against a fixed forger."""
    p_data, p_g = np.asarray(p_data, dtype=float), np.asarray(p_g, dtype=float)
    return p_data / (p_data + p_g)


def fit_discriminator_table(p_data, p_g, steps: int = 4000, lr: float = 5.0) -> np.ndarray:
    """Train the most flexible detective there is, one free score per point, and return its verdicts.

    The data and the forger live on the same few points with the given
    probabilities. Gradient ascent on the exact expected V: for a point with
    score a and verdict D = sigmoid(a), dV/da = p_data (1 - D) - p_g D. It is
    zero exactly when D = p_data / (p_data + p_g), so the trained table
    arrives at `optimal_discriminator` without ever being told the formula.
    """
    p_data, p_g = np.asarray(p_data, dtype=float), np.asarray(p_g, dtype=float)
    a = np.zeros_like(p_data)
    for _ in range(steps):
        d = sigmoid(a)
        a += lr * (p_data * (1.0 - d) - p_g * d)
    return sigmoid(a)


# ---------------------------------------------------------------------------
# 4. Saturation: the original forger loss versus the non-saturating one
# ---------------------------------------------------------------------------


def generator_logit_gradient(d_fake, loss: str = "non_saturating"):
    """The forger's loss gradient with respect to the detective's score a, where D(G(z)) = sigmoid(a).

    saturating (the original):  loss = log(1 - D)  ->  d loss / da = -D
    non_saturating:             loss = -log D      ->  d loss / da = -(1 - D)

    When the detective is sure a fake is fake (D near 0), the first is near 0
    and the second near -1: only the second still tells the forger which way to move.
    """
    d = np.asarray(d_fake, dtype=float)
    out = -d if loss == "saturating" else -(1.0 - d)
    return float(out) if out.ndim == 0 else out


def _forger_gradient(G: MLP, D: MLP, z: np.ndarray, loss: str) -> np.ndarray:
    """The gradient of the forger's loss with respect to every forger parameter (one flat vector)."""
    fakes = G.forward(z)
    a = D.forward(fakes)
    _, grad_fakes = D.backward(generator_logit_gradient(sigmoid(a), loss) / len(z))
    grad_params, _ = G.backward(grad_fakes)
    return grad_params


def head_start_experiment(head_starts=(0, 25, 50, 100, 200, 300), lr: float = 3e-3, batch: int = 128, seed: int = 0) -> list[dict]:
    """Train only the detective against a frozen, untrained forger, then measure what the forger would learn.

    For each head start (detective steps), returns the average verdict on
    fakes and the size (norm) of the forger's parameter gradient under each
    loss. The untrained forger's points sit near the centre, far from the
    ring, so the detective quickly becomes certain, and the original loss's
    gradient fades while the non-saturating one stays large.
    """
    rng = np.random.default_rng(seed)
    G, D = MLP(GENERATOR_SIZES, seed=seed), MLP(DISCRIMINATOR_SIZES, seed=seed + 1)
    opt = Adam(lr=lr, beta1=0.5)
    probe_z = np.random.default_rng(seed + 100).standard_normal((batch, 2))
    rows, done = [], 0
    for target in sorted(head_starts):
        while done < target:
            _detective_step(D, opt, ring_of_gaussians(batch, rng), G.forward(rng.standard_normal((batch, 2))))
            done += 1
        rows.append(
            dict(
                head_start=target,
                d_fake=float(sigmoid(D.forward(G.forward(probe_z))).mean()),
                saturating=float(np.linalg.norm(_forger_gradient(G, D, probe_z, "saturating"))),
                non_saturating=float(np.linalg.norm(_forger_gradient(G, D, probe_z, "non_saturating"))),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# 5. Training: alternate one detective step and one forger step
# ---------------------------------------------------------------------------


def _detective_step(D: MLP, opt: Adam, real: np.ndarray, fakes: np.ndarray) -> None:
    """One step up the value function: push D(real) towards 1 and D(fake) towards 0."""
    n = len(real)
    # d(-V)/da is D - 1 for a real point and D for a fake: binary cross-entropy's "prediction minus target".
    grad_real, _ = D.backward((sigmoid(D.forward(real)) - 1.0) / n)
    grad_fake, _ = D.backward(sigmoid(D.forward(fakes)) / n)
    D.params[:] = opt.step(D.params, grad_real + grad_fake)


def mode_coverage(samples: np.ndarray, min_share: float = 0.02, std: float = MODE_STD) -> dict:
    """How much of the ring the samples cover.

    A sample is *high quality* if it lands within 3 standard deviations of a
    mode's centre. A mode is *covered* if it holds at least `min_share` of
    all samples (2% by default: 20 of 1,000, where a perfect forger puts 125).
    Returns covered (a count), which (a string such as ".X...X.." marking the
    covered modes), counts per mode, and quality (the high-quality fraction).
    """
    distances = np.linalg.norm(samples[:, None, :] - ring_modes()[None], axis=2)
    nearest, close = distances.argmin(axis=1), distances.min(axis=1) < 3 * std
    counts = np.bincount(nearest[close], minlength=N_MODES)
    hit = counts >= min_share * len(samples)
    return dict(
        covered=int(hit.sum()),
        which="".join("X" if h else "." for h in hit),
        counts=counts,
        quality=float(close.mean()),
    )


@functools.lru_cache(maxsize=None)
def train_gan(
    steps: int = 2500,
    lr_g: float = 1e-3,
    lr_d: float = 3e-3,
    batch: int = 128,
    snapshot_every: int = 250,
    loss: str = "non_saturating",
    seed: int = 1,
) -> dict:
    """Train a forger and a detective on the ring, alternating one step each.

    Returns snapshots every `snapshot_every` steps: the step, 1,000 generated
    points (always from the same noise, so pictures are comparable), the
    detective's parameters, and the `mode_coverage` of the points. Cached,
    because the lesson, its figures and its tests all look at the same runs.
    """
    rng = np.random.default_rng(seed)
    G, D = MLP(GENERATOR_SIZES, seed=seed), MLP(DISCRIMINATOR_SIZES, seed=seed + 1)
    # beta1 = 0.5 rather than 0.9: less momentum, the usual choice for GANs since DCGAN, because the target keeps moving.
    opt_g, opt_d = Adam(lr=lr_g, beta1=0.5), Adam(lr=lr_d, beta1=0.5)
    probe_z = np.random.default_rng(seed + 100).standard_normal((1000, 2))
    snapshots = []
    for step in range(steps + 1):
        if step % snapshot_every == 0:
            points = G.forward(probe_z).copy()
            snapshots.append(dict(step=step, samples=points, detective=D.params.copy(), coverage=mode_coverage(points)))
        if step == steps:
            break
        z = rng.standard_normal((batch, 2))
        fakes = G.forward(z)
        # 1. Detective: one step towards telling this batch of real points from this batch of fakes.
        _detective_step(D, opt_d, ring_of_gaussians(batch, rng), fakes)
        # 2. Forger: one step towards fooling the detective as it is *now*.
        G.params[:] = opt_g.step(G.params, _forger_gradient(G, D, z, loss))
    return dict(snapshots=snapshots, coverage=[s["coverage"] for s in snapshots], lr_g=lr_g, lr_d=lr_d)


def detective_verdicts(params: np.ndarray, points: np.ndarray) -> np.ndarray:
    """D(x) for each point, using a detective's saved parameters."""
    D = MLP(DISCRIMINATOR_SIZES)
    D.params[:] = params
    return sigmoid(D.forward(points))[:, 0]


# ---------------------------------------------------------------------------
# 6. Oscillation in miniature: the Dirac GAN
# ---------------------------------------------------------------------------


def dirac_gan(theta: float = 1.0, psi: float = 0.0, lr: float = 0.2, steps: int = 300, gamma: float = 0.0, alternating: bool = False) -> np.ndarray:
    """The smallest GAN there is: two numbers, followed step by step. Returns the path, shape (steps + 1, 2).

    Real data is always the single point x = 0. The forger has one number,
    theta, and always outputs x = theta. The detective has one number, psi,
    and says D(x) = sigmoid(psi x). The value is
    V = log D(0) + log(1 - D(theta)) = log 0.5 + log(1 - sigmoid(psi theta)).

        dV/dpsi   = -sigmoid(psi theta) theta   (the detective climbs this)
        dV/dtheta = -sigmoid(psi theta) psi     (the forger descends this)

    `gamma` > 0 adds the R1 gradient penalty (gamma / 2)(dD-score/dx at the
    real data)^2 = (gamma / 2) psi^2 to what the detective minimises.
    `alternating` lets the forger see the detective's new psi before it moves.
    """
    path = [(theta, psi)]
    for _ in range(steps):
        s = sigmoid(psi * theta)
        new_psi = psi + lr * (-s * theta - gamma * psi)
        if alternating:
            psi = new_psi
            s = sigmoid(psi * theta)
        theta = theta + lr * s * psi  # descend dV/dtheta = -s psi
        psi = new_psi
        path.append((float(theta), float(psi)))
    return np.array(path)


# ---------------------------------------------------------------------------
# 7. Stabilizers: distances between piles, and capping the detective's steepness
# ---------------------------------------------------------------------------


def js_divergence(p, q) -> float:
    """Jensen-Shannon divergence between two histograms on the same bins (natural log).

    JS = 1/2 KL(p || m) + 1/2 KL(q || m), with m the average of p and q. It
    is 0 when p = q and log 2 when they share no bin at all, *however far
    apart* the bins are. That flatness is why a GAN's detective can stop
    giving useful directions.
    """
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0  # 0 log 0 counts as 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def wasserstein_1d(a, b) -> float:
    """Earth mover's distance between two equal-sized 1-D samples: sort both, pair them up, average the gaps."""
    a, b = np.sort(np.asarray(a, dtype=float)), np.sort(np.asarray(b, dtype=float))
    return float(np.mean(np.abs(a - b)))


def largest_stretch(W: np.ndarray, iters: int = 100, seed: int = 0) -> float:
    """The most W can lengthen any vector (its spectral norm), by power iteration.

    Push a vector through W and back through W transpose, again and again;
    it swings round to the direction W stretches most, and the stretch
    factor is read off at the end. Spectral normalization runs one step of
    this per training step instead of a full singular value decomposition.
    """
    v = np.random.default_rng(seed).standard_normal(W.shape[1])
    for _ in range(iters):
        u = W @ v
        u /= np.linalg.norm(u)
        v = W.T @ u
        v /= np.linalg.norm(v)
    return float(np.linalg.norm(W @ v))


def spectrally_normalize(W: np.ndarray) -> np.ndarray:
    """W divided by its largest stretch, so no input direction is stretched by more than 1."""
    return W / largest_stretch(W)


# ---------------------------------------------------------------------------
# 8. Figures (rendered into the HTML docs by `make figures`)
# ---------------------------------------------------------------------------

# The detective's learning rate for each of the three ring runs; the forger always uses 1e-3.
RING_RUNS = {"equal speeds (0.001)": 1e-3, "detective 3x faster (0.003)": 3e-3, "detective 10x faster (0.01)": 1e-2}


def _normal_pdf(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * math.sqrt(2 * math.pi))


def figures() -> dict:
    """Plot this lesson's data. matplotlib is imported here, and only here,
    so the lesson itself needs nothing beyond NumPy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    REAL, FAKE, GOOD, MUTED, THIRD = "#2563eb", "#dc2626", "#059669", "#9ca3af", "#d97706"
    run_colours = dict(zip(RING_RUNS, (FAKE, REAL, THIRD)))
    runs = {name: train_gan(lr_d=lr_d) for name, lr_d in RING_RUNS.items()}
    figs = {}

    # --- 1. The best detective on a line -------------------------------------
    xs = np.linspace(-5, 5, 400)
    p_data = 0.5 * _normal_pdf(xs, -2, 0.5) + 0.5 * _normal_pdf(xs, 2, 0.5)
    p_g = _normal_pdf(xs, 0, 1.5)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.4, 5), sharex=True)
    top.fill_between(xs, p_data, color=REAL, alpha=0.25)
    top.plot(xs, p_data, color=REAL, label="real data  p_data")
    top.fill_between(xs, p_g, color=FAKE, alpha=0.15)
    top.plot(xs, p_g, color=FAKE, label="forger  p_g")
    top.set_ylabel("density")
    top.set_title("Where each distribution puts its points")
    top.legend(frameon=False, loc="upper center")
    bottom.plot(xs, optimal_discriminator(p_data, p_g), color="#111827", label="best verdict  D*")
    bottom.axhline(0.5, color=MUTED, ls="--", label="D* once the forger matches the data")
    bottom.set_ylim(-0.03, 1.03)
    bottom.set_xlabel("position x")
    bottom.set_ylabel("D*(x)")
    bottom.set_title("The best detective: the share of points at x that are real")
    bottom.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    figs["optimal_detective"] = fig

    # --- 2. The trained detective's view of the ring -------------------------
    last = runs["detective 3x faster (0.003)"]["snapshots"][-1]
    grid = np.linspace(-3, 3, 161)
    gx, gy = np.meshgrid(grid, grid)
    verdict = detective_verdicts(last["detective"], np.stack([gx.ravel(), gy.ravel()], axis=1)).reshape(gx.shape)
    real = ring_of_gaussians(1000, seed=5)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(verdict, origin="lower", extent=(-3, 3, -3, 3), cmap="RdBu", vmin=0, vmax=1)
    ax.scatter(real[:, 0], real[:, 1], s=3, color="#111827", label="real")
    ax.scatter(last["samples"][:, 0], last["samples"][:, 1], s=3, color=GOOD, alpha=0.6, label="forger")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_title(f"The detective's verdict after {last['step']:,} steps")
    ax.legend(frameon=True, loc="upper right", markerscale=3)
    fig.colorbar(im, ax=ax, fraction=0.046, label="D(x): probability real")
    figs["detective_view"] = fig

    # --- 3. Saturating vs non-saturating forger loss -------------------------
    scores = np.linspace(-7, 3, 300)
    rows = head_start_experiment(head_starts=(0, 50, 100, 200, 300, 400, 600, 800))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(scores, np.log(1 - sigmoid(scores)), color=FAKE, label="original: log(1 - D)")
    a1.plot(scores, -np.log(sigmoid(scores)), color=REAL, label="non-saturating: -log D")
    a1.axvline(math.log(0.01 / 0.99), color=MUTED, ls="--")
    a1.text(math.log(0.01 / 0.99) + 0.15, 5.6, "D = 0.01", color="#4b5563")
    a1.set_xlabel("detective's score a for a fake  (left: sure it is fake)")
    a1.set_ylabel("forger's loss")
    a1.set_title("The original loss goes flat when the detective is sure")
    a1.legend(frameon=False, loc="upper right")
    steps = [r["head_start"] for r in rows]
    a2.semilogy(steps, [r["saturating"] for r in rows], "o-", color=FAKE, label="original loss")
    a2.semilogy(steps, [r["non_saturating"] for r in rows], "o-", color=REAL, label="non-saturating loss")
    for r in rows[::2]:
        a2.annotate(f"D={r['d_fake']:.3f}", (r["head_start"], r["non_saturating"]), textcoords="offset points", xytext=(-10, 8), fontsize=7, color="#4b5563")
    a2.set_xlabel("detective's head start (training steps)")
    a2.set_ylabel("size of the forger's gradient")
    a2.set_title("Measured on the ring: a head start starves one loss")
    a2.legend(frameon=False, loc="center right")
    fig.tight_layout()
    figs["forger_losses"] = fig

    # --- 4. The Dirac GAN: spiral out, or orbit ------------------------------
    simultaneous, alternating, penalised = dirac_gan(), dirac_gan(alternating=True), dirac_gan(gamma=1.0)
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.plot(simultaneous[:, 0], simultaneous[:, 1], color=FAKE, lw=1.2, label="simultaneous steps: spiral out")
    ax.plot(alternating[:, 0], alternating[:, 1], color=REAL, lw=1.2, label="alternating steps: endless orbit")
    ax.plot([1], [0], "o", color="#111827")
    ax.annotate("start (1, 0)", (1, 0), textcoords="offset points", xytext=(6, 6))
    ax.plot([0], [0], "x", color="#111827", ms=10, mew=2)
    ax.annotate("equilibrium", (0, 0), textcoords="offset points", xytext=(6, -12))
    ax.set_aspect("equal")
    ax.set_xlabel("forger's theta (where the fake sits)")
    ax.set_ylabel("detective's psi (its slope)")
    ax.set_title("The Dirac GAN never settles")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    figs["dirac_oscillation"] = fig

    # --- 5. Ring snapshots: hopping, covering, stuck -------------------------
    shown = (1750, 2000, 2250, 2500)
    modes = ring_modes()
    fig, axes = plt.subplots(3, 4, figsize=(10, 7.8), sharex=True, sharey=True)
    for row, (name, run) in zip(axes, runs.items()):
        by_step = {snap["step"]: snap for snap in run["snapshots"]}
        for ax, step in zip(row, shown):
            snap = by_step[step]
            for centre in modes:
                ax.add_patch(plt.Circle(centre, 3 * MODE_STD, color=MUTED, alpha=0.5))
            ax.scatter(snap["samples"][:, 0], snap["samples"][:, 1], s=2, color=GOOD, alpha=0.5)
            ax.set_title(f"step {step:,}  {snap['coverage']['which']}", fontsize=9, family="monospace")
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(name, fontsize=9)
    fig.suptitle("The forger's points (green) against the eight real clouds (grey)", fontweight="bold")
    fig.tight_layout()
    figs["ring_snapshots"] = fig

    # --- 6. Mode coverage over training ----------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for name, run in runs.items():
        ax.step([s["step"] for s in run["snapshots"]], [c["covered"] for c in run["coverage"]], where="post", color=run_colours[name], lw=2, label=name)
    ax.set_ylim(-0.3, 8.5)
    ax.set_yticks(range(0, 9))
    ax.set_xlabel("training step")
    ax.set_ylabel("clouds covered (of 8)")
    ax.set_title("How much of the ring the forger covers")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    figs["mode_coverage"] = fig

    # --- 7. R1 penalty: the spiral turns inward --------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for path, colour, label in (
        (simultaneous, FAKE, "simultaneous steps"),
        (alternating, REAL, "alternating steps"),
        (penalised, GOOD, "simultaneous + R1 penalty (gamma = 1)"),
    ):
        ax.plot(np.linalg.norm(path, axis=1), color=colour, lw=2, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("distance from equilibrium")
    ax.set_title("A gradient penalty damps the chase")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    figs["dirac_r1"] = fig

    # --- 8. Jensen-Shannon vs Wasserstein --------------------------------------
    bins = np.round(np.arange(-40, 41) / 10, 1)  # positions -4.0 ... 4.0 in steps of 0.1
    data = (bins == 0.0).astype(float)
    thetas = bins
    js = [js_divergence(data, (bins == t).astype(float)) for t in thetas]
    w = [wasserstein_1d([0.0], [t]) for t in thetas]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(thetas, js, ".", color=FAKE, ms=4, label="Jensen-Shannon divergence")
    ax.plot(thetas, w, color=REAL, lw=2, label="Wasserstein distance")
    ax.axhline(math.log(2), color=MUTED, ls="--")
    ax.text(-3.9, math.log(2) + 0.12, "log 2 = 0.693", color="#4b5563")
    ax.set_xlabel("where the forger's pile sits (the real pile is at 0)")
    ax.set_ylabel("distance between the piles")
    ax.set_title("Only one yardstick says which way to move")
    ax.legend(frameon=False, loc="upper center")
    figs["wasserstein_vs_js"] = fig

    return figs


# ---------------------------------------------------------------------------
# 9. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. The two players")
    G, D = MLP(GENERATOR_SIZES, seed=1), MLP(DISCRIMINATOR_SIZES, seed=2)
    fakes = G.forward(np.random.default_rng(0).standard_normal((1000, 2)))
    real = ring_of_gaussians(1000, seed=0)
    say(
        f"""
        The real data is a ring of eight little clouds of radius {RADIUS}. The
        forger maps 2 random numbers to a point through layers {GENERATOR_SIZES};
        the detective maps a point to a probability through {DISCRIMINATOR_SIZES}
        and a sigmoid. Untrained, the forger's points sit an average of
        {np.linalg.norm(fakes, axis=1).mean():.2f} from the centre, while real
        points sit {np.linalg.norm(real, axis=1).mean():.2f} away, and the
        untrained detective says {sigmoid(D.forward(real)).mean():.2f} on real
        points and {sigmoid(D.forward(fakes)).mean():.2f} on fakes: it knows nothing yet.
        """
    )

    banner("2. The game: one number both players fight over")
    table(
        ["detective", "D on real notes", "D on fakes", "V"],
        [
            ("the worked example", "0.9, 0.8", "0.2, 0.4", value_function([0.9, 0.8], [0.2, 0.4])),
            ("never wrong", "1, 1", "0, 0", value_function([1.0, 1.0], [0.0, 0.0])),
            ("coin flip", "0.5, 0.5", "0.5, 0.5", value_function([0.5, 0.5], [0.5, 0.5])),
        ],
        floatfmt=".3f",
    )
    takeaway("The detective pushes V up towards 0; the forger pushes it down. -V is the detective's binary cross-entropy.")

    banner("3. The best possible detective")
    p_data, p_g = [0.4, 0.3, 0.2, 0.1], [0.1, 0.1, 0.4, 0.4]
    trained = fit_discriminator_table(p_data, p_g)
    table(
        ["point", "p_data", "p_g", "formula D*", "trained detective"],
        [(i + 1, pd, pg, optimal_discriminator(pd, pg), t) for i, (pd, pg, t) in enumerate(zip(p_data, p_g, trained))],
        floatfmt=".3f",
    )
    say("A detective trained against a fixed forger lands on p_data / (p_data + p_g) without being told the formula.")
    takeaway("At equilibrium the forger matches the data, so the best detective says 1/2 everywhere and V = -log 4.")

    banner("4. Why the forger uses the non-saturating loss")
    table(
        ["D(G(z))", "original loss slope", "non-saturating slope"],
        [(d, generator_logit_gradient(d, "saturating"), generator_logit_gradient(d, "non_saturating")) for d in (0.5, 0.1, 0.01, 0.001)],
        floatfmt=".3f",
    )
    table(
        ["detective head start", "D on fakes", "forger gradient, original", "forger gradient, non-saturating"],
        [(r["head_start"], r["d_fake"], r["saturating"], r["non_saturating"]) for r in head_start_experiment(head_starts=(0, 200, 800))],
        floatfmt=".3f",
    )
    takeaway("When the detective is sure, the original loss goes quiet; the non-saturating one is loudest exactly then.")

    banner("5. Oscillation: the two-number Dirac GAN")
    table(
        ["steps of size 0.2", "distance after 50", "after 100", "after 300"],
        [
            (name, *(float(np.linalg.norm(path[i])) for i in (50, 100, 300)))
            for name, path in (
                ("simultaneous", dirac_gan()),
                ("alternating", dirac_gan(alternating=True)),
                ("simultaneous + R1 penalty", dirac_gan(gamma=1.0)),
            )
        ],
        floatfmt=".3f",
    )
    say("Start 1 away from the equilibrium. Plain steps spiral out or orbit; the R1 penalty's friction brings them home.")

    banner("6. Mode collapse on the ring")
    for name, lr_d in RING_RUNS.items():
        run = train_gan(lr_d=lr_d)
        print(f"{name:28s} " + " ".join(c["which"] for c in run["coverage"][5:]))
    print()
    table(
        ["detective learning rate", "clouds covered at the end", "high-quality points"],
        [(name, train_gan(lr_d=lr_d)["coverage"][-1]["covered"], train_gan(lr_d=lr_d)["coverage"][-1]["quality"]) for name, lr_d in RING_RUNS.items()],
        floatfmt=".2f",
    )
    say(
        """
        Each row shows coverage every 250 steps from step 1,250 (X = covered).
        Equal speeds: the forger hops between alternate clouds. A detective
        three times faster: all eight. Ten times faster: six, and stuck.
        """
    )
    takeaway("Mode collapse is invisible sample by sample; you have to measure coverage.")

    banner("7. Yardsticks and speed limits")
    grid = np.arange(11)
    table(
        ["forger's pile at", "Jensen-Shannon", "Wasserstein"],
        [(t, js_divergence(grid == 0, grid == t), wasserstein_1d([0.0], [float(t)])) for t in (1, 5, 10)],
        floatfmt=".3f",
    )
    W = np.array([[3.0, 0.0], [0.0, 1.0]])
    say(
        f"""
        Jensen-Shannon can't tell a near miss from a far one; Wasserstein can.
        Spectral normalization: W = [[3, 0], [0, 1]] has largest stretch
        {largest_stretch(W):.3f}; after dividing by it, {largest_stretch(spectrally_normalize(W)):.3f}.
        """
    )
    takeaway("Keep the detective useful: paced, smooth and never infinitely sure.")


if __name__ == "__main__":
    demo()
