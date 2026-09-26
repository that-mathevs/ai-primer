r"""
# Diffusion and flow matching: turning noise into data one small step at a time

Run: `python -m primer.ml.generative.diffusion`

New to the notation (vectors, sums, square roots, averages)? Every symbol is
decoded where it appears, and `primer.notation` teaches them all from zero.
This lesson trains small networks, so `primer.ml.neural_net` (how a network
learns) is the one to read first.

## The everyday picture: a sculptor who only ever improves things a little

Imagine a sculptor who cannot carve a statue in one go. Hand them a shapeless
block and ask for a horse, and they freeze. But hand them a *slightly rough*
horse, and they can always make it a little less rough: smooth this bump,
deepen that line. Now chain that one modest skill. Start from a shapeless
block, make it a bit more horse-like, then a bit more, a hundred times over,
and a horse comes out.

A diffusion model is that sculptor. It never learns "draw a picture from
nothing", which is very hard. It learns "here is a picture with a little too
much noise in it: take some of the noise away", which is much easier. And
the practice material is free: take any real picture, add noise yourself,
and you know exactly what the answer should be.

```mermaid
flowchart LR
  D["Real data<br/>(a photo)"] -- "add a little noise,<br/>T times (fixed, no learning)" --> N["Pure noise<br/>(TV static)"]
  N -- "remove a little noise,<br/>T times (learned)" --> S["A new sample<br/>(a photo no one took)"]
```

**Reading it:** the top arrow is the *forward process*: it destroys data by
adding noise a little at a time, and it involves no learning at all. The
bottom arrow runs the same road backwards, and that direction is what the
network learns. At generation time only the bottom arrow runs, starting
from fresh noise, so every run ends at a different, new sample.

**The tiny world this lesson works in.** A real image is a long list of
numbers, one per pixel colour. To keep everything small enough to see, our
"images" have just two numbers, so each one is a dot on a flat plane. The
real data is four round blobs of dots, one in each corner (north-east,
north-west, south-west, south-east), 600 dots in all. The job: learn to make
*new* dots that land in the blobs, starting from random noise. Every idea
below works unchanged on a 512 × 512 colour image; it just has 786,432
numbers per "dot" instead of 2.

Notice one thing about the blobs now, because it matters later: their
average is the empty middle. A model that plays safe and outputs "the
average answer" puts its dots where no real data lives. The same failure
makes blurry images: the average of many sharp faces is a blur.

## Step 1: the forward process, adding noise a little at a time

**Everyday picture.** Think of an old television slowly losing its signal.
Turn the static dial up one notch: the picture fades a little and a little
snow creeps in. Turn it a hundred notches and you see only snow, and nothing
in it tells you what the programme was.

**Tiny worked example.** Follow one pixel whose clean value is $x_0 = 2.0$.
Each step keeps most of the signal and mixes in a little fresh random noise.
The **noise** is a random number drawn from a **standard normal
distribution**: the bell curve centred on 0 whose typical size is 1 (values
like 0.5, −1.2 and 0.1 are common, 3 is rare). Say step 1 mixes in a share
$\beta = 0.1$ of noise, and the random draw is $\varepsilon = 0.5$.

$$
x_t = \sqrt{1 - \beta_t}\; x_{t-1} + \sqrt{\beta_t}\; \varepsilon
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $t$ | the step number, from 1 to $T$ (the last step) | 1 |
| $x_{t-1}$ | the value before this step ($x_0$ is the clean data) | 2.0 |
| $x_t$ | the value after this step | 2.06 |
| $\beta_t$ | "beta": how much noise this step mixes in, a number between 0 and 1 | 0.1 |
| $\sqrt{1-\beta_t}$ | how much of the old value survives: it shrinks a little | $\sqrt{0.9} = 0.949$ |
| $\varepsilon$ | "epsilon": fresh noise drawn from the standard normal bell curve | 0.5 |
| $\sqrt{\beta_t}$ | how much noise is mixed in | $\sqrt{0.1} = 0.316$ |

**In words:** "shrink the old value a little, then add a little random noise."

**With the numbers:** $x_1 = \sqrt{0.9} \cdot 2.0 + \sqrt{0.1} \cdot 0.5 =
1.897 + 0.158 = 2.06$.

**In Python:**

```python
import math
x_prev, beta, eps = 2.0, 0.1, 0.5
# √(1 − β) x_{t−1}: shrink the old value
round(math.sqrt(1 - beta) * x_prev, 3)  # → 1.897
# √β ε: the noise mixed in
round(math.sqrt(beta) * eps, 3)  # → 0.158
round(math.sqrt(1 - beta) * x_prev + math.sqrt(beta) * eps, 2)  # → 2.06
```

Why the square roots? A value's **variance** (its typical squared size) is
what matters, and variances of independent things add. If the old value has
variance 1, the new one has variance $(1 - \beta) \cdot 1 + \beta \cdot 1 = 1$:
exactly 1 again. The square roots keep every step at the same overall size,
so after many steps the value is noise of size 1, not a number that has
blown up or faded to zero.

### The shortcut: jump straight to any step

Running $t$ steps one by one is slow, and training needs noisy examples at
every step. Luckily, adding noise twice is the same as adding it once in a
bigger dose, so there is a formula that jumps straight to step $t$. First,
multiply up how much signal survives every step so far:

$$
\bar\alpha_t = \prod_{s=1}^{t} (1 - \beta_s)
\qquad\qquad
x_t = \sqrt{\bar\alpha_t}\; x_0 + \sqrt{1 - \bar\alpha_t}\; \varepsilon
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $1 - \beta_s$ | the share of variance step $s$ keeps (often written $\alpha_s$) | 0.9, then 0.8 |
| $\prod_{s=1}^{t}$ | "multiply together, for $s$ = 1, 2, …, $t$" (see `primer.notation`) | $0.9 \times 0.8$ |
| $\bar\alpha_t$ | "alpha bar": the share of the original signal's variance left after $t$ steps; 1 means clean, 0 means pure noise | 0.72; or 0.64 in the jump below |
| $x_0$ | the clean data | 2.0 |
| $\varepsilon$ | one fresh standard-normal draw, standing in for all the noise added so far | 0.5 |
| $\sqrt{\bar\alpha_t}$ | the **signal share**: how much of $x_0$ is still in $x_t$ | $\sqrt{0.64} = 0.8$ |
| $\sqrt{1-\bar\alpha_t}$ | the **noise share** | $\sqrt{0.36} = 0.6$ |

**In words:** "multiply together the survival share of every step so far;
then the noisy value is that much of the clean value plus the rest in noise."

**With the numbers:** two steps with $\beta = 0.1$ then $0.2$ leave
$\bar\alpha_2 = 0.9 \times 0.8 = 0.72$. In this lesson's schedule, step 30
leaves $\bar\alpha_{30} = 0.64$, so the pixel at 2.0 with noise 0.5 lands at
$x_{30} = 0.8 \cdot 2.0 + 0.6 \cdot 0.5 = 1.6 + 0.3 = 1.9$.

**In Python:**

```python
import math
betas = [0.1, 0.2]
alpha_bar = 1.0
# ∏ (1 − β_s): multiply up what each step keeps
for beta in betas:
    alpha_bar *= 1 - beta
round(alpha_bar, 2)  # → 0.72
# the jump to step 30, where ᾱ = 0.64
x0, eps, alpha_bar_30 = 2.0, 0.5, 0.64
round(math.sqrt(alpha_bar_30) * x0 + math.sqrt(1 - alpha_bar_30) * eps, 2)  # → 1.9
```

```mermaid
flowchart LR
  X0["x₀<br/>clean"] --> X1["x₁"] --> X2["x₂"] --> DOTS["…"] --> XT["x_T<br/>pure noise"]
  X0 -. "shortcut: √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε" .-> XM["x_t<br/>any step, in one jump"]
```

**Reading it:** the solid chain is the slow way: one small dose of noise per
arrow. The dotted arrow is the shortcut: pick any step $t$, look up
$\bar\alpha_t$, draw one noise sample, and land exactly where the chain
would have taken you (in distribution: the same spread of possible values).
Training uses only the shortcut.

The **noise schedule** is the list of $\beta_t$. This lesson uses $T = 100$
steps with $\beta$ rising in a straight line from 0.0001 to 0.1: tiny doses
first, while there is fine detail to lose, bigger ones once little is left.

![Five panels: the four coloured blobs at step 0 blur at step 30, merge at step 50, and are a single shapeless cloud by step 100](figures/primer.ml.generative.diffusion.noising.svg)

**Reading it:** each panel is the whole dataset, noised to the step in its
title, with every dot keeping its blob's colour. Step 10 barely differs from
the clean data (its signal share $\sqrt{\bar\alpha}$ is still 0.98). By step 30
the blobs have swollen into each other, though each colour still keeps to
its own corner. By step 50 the colours overlap. At step 100 only
7% of the signal is left and the colours are thoroughly mixed: a round cloud
of plain noise, the same cloud whatever data you started from. That last
fact is what makes generation possible: we know how to draw from that cloud.

![Two curves over 100 steps: the signal share falls from 1 to 0.07 while the noise share rises from 0 to nearly 1, crossing at step 37](figures/primer.ml.generative.diffusion.schedule.svg)

**Reading it:** the x-axis is the step. The blue curve is the signal share
$\sqrt{\bar\alpha_t}$ and the red curve is the noise share
$\sqrt{1 - \bar\alpha_t}$. The two dots at step 30 are the worked example's
0.8 and 0.6. The dashed line at step 37 is where signal and noise are equal.
Early steps change little (the curves are flat at the left), so the network
gets plenty of practice on nearly clean data, where the fine detail lives.

**In code:** `linear_schedule` builds the list of $\beta_t$ as a `NoiseSchedule`, whose `NoiseSchedule.alpha_bars` is the running product; `noise_step` is one step and `add_noise` is the shortcut.

**Why it matters in practice.** The forward process has no parameters and
needs no training. Its only job is to manufacture unlimited practice
material, and thanks to the shortcut, any clean example can be turned into a
practice question at any noise level in one line.

## Step 2: learning to denoise

**Everyday picture.** A photo restorer wants to practise removing specks of
dust. They take clean photos and sprinkle dust on them *themselves*, keeping
a note of exactly where every speck went. Now they can practise all day:
guess where the dust is, then check against the note. The homework is
unlimited and every answer is known.

**Tiny worked example.** Take the pixel from before: $x_0 = 2.0$, noise
$\varepsilon = 0.5$, step 30, so $x_{30} = 1.9$. The network is shown only
$x_{30} = 1.9$ and the step number 30, and asked: "what noise was added?"
The right answer is 0.5. If it guesses 0.4, it is off by 0.1 and scores
$(0.5 - 0.4)^2 = 0.01$. Lower is better.

Why guess the noise rather than the clean value? The two are equivalent:
given $x_t$, $t$ and the noise, the clean value follows by undoing the
shortcut, $x_0 = (1.9 - 0.6 \cdot 0.5) / 0.8 = 2.0$. Guessing the noise
simply works better in practice, because the target always has the same
size (a standard-normal draw), whatever the step.

```mermaid
flowchart LR
  D["pick a real point x₀"] --> MIX
  T["pick a random step t"] --> MIX
  E["draw noise ε"] --> MIX["mix with the shortcut<br/>x_t = √ᾱ_t x₀ + √(1 − ᾱ_t) ε"]
  MIX --> NET["network ε_θ<br/>sees x_t and t"]
  NET --> G["guess ε̂"]
  G --> L["loss ‖ε − ε̂‖²"]
  E --> L
  L --> U["nudge the weights<br/>(backprop + Adam)"]
```

**Reading it:** three random choices feed in from the left: which example,
which step, which noise. The shortcut mixes them into a noisy point. The
network sees only the noisy point and the step (not the noise, and not the
clean point). The noise takes the lower path straight to the loss, where the
guess is graded against it. Everything after the loss is ordinary training,
exactly as in `primer.ml.neural_net`.

$$
L(\theta) = \mathbb{E}_{x_0,\, t,\, \varepsilon}\Big[\, \big\| \varepsilon - \varepsilon_\theta(x_t, t) \big\|^2 \,\Big]
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $\theta$ | "theta": all of the network's weights, the knobs training turns | about 5,000 numbers here |
| $\varepsilon_\theta(x_t, t)$ | the network's guess of the noise, given the noisy point and the step; written $\hat\varepsilon$ ("epsilon hat") | (0.4, −0.8) |
| $\varepsilon$ | the noise that was really added (a 2-number vector in our world) | (0.5, −1.0) |
| $\varepsilon - \varepsilon_\theta$ | the error: how far off the guess is, per number | (0.1, −0.2) |
| $\lVert v \rVert^2$ | squared length: square every entry and add them up | $0.1^2 + 0.2^2 = 0.05$ |
| $\mathbb{E}_{x_0, t, \varepsilon}[\ldots]$ | **expectation**: the average over many random choices of example, step and noise; in code, the average over a batch | the batch average |
| $L(\theta)$ | the loss: one number saying how bad the guesses are with these weights | about 0.6 after training |

**In words:** "over many random examples, steps and noises, average how far
the network's noise guess is from the real noise, measured as squared
distance."

**With the numbers:** true noise (0.5, −1.0), guess (0.4, −0.8): error
(0.1, −0.2), squared length $0.01 + 0.04 = 0.05$. A network that always
guesses zero scores the average of $\varepsilon_1^2 + \varepsilon_2^2$, which
is 1 + 1 = 2 (each standard-normal number has variance 1). That is the score
to beat.

**In Python:**

```python
import random
eps = [0.5, -1.0]
guess = [0.4, -0.8]
# ‖ε − ε̂‖²: square each error and add
round(sum((e - g) ** 2 for e, g in zip(eps, guess)), 2)  # → 0.05
# the baseline: always guessing 0, averaged over many draws of 2-D noise
random.seed(0)
draws = [random.gauss(0, 1) ** 2 + random.gauss(0, 1) ** 2 for _ in range(100_000)]
round(sum(draws) / len(draws), 1)  # → 2.0
```

This is plain **regression** (predicting numbers, graded by squared error),
the most stable kind of training there is. There is no opponent to balance
and no trick: that is a big part of why diffusion displaced GANs.

**The network.** Our denoiser is a small multi-layer network: the noisy
point (2 numbers), plus the step turned into 8 numbers of sines and cosines
(a single number "30" is hard for a small network to use; a spread of waves
at different speeds is easy, the same trick as the sinusoidal positions in
`primer.ml.positional`), go through two hidden layers of 64 ReLU units
and come out as 2 numbers, the noise guess. It trains for 1,500 steps of
Adam (`primer.ml.optimizers`) on batches of 256, in about half a second.

![A training curve that starts at 2, the score of guessing zero, drops below 0.7 within a hundred steps and settles near 0.6](figures/primer.ml.generative.diffusion.training.svg)

**Reading it:** the x-axis is training steps and the y-axis is the batch's
average squared error. The dashed red line at 2 is the "always guess zero"
baseline, and the untrained network starts right on it (its last layer
starts near zero). The loss plunges within a hundred steps, then creeps
down. It never reaches 0, and cannot: at low noise levels the noise is
hidden inside the blob's own natural spread, so even a perfect network can
only guess its average. The floor near 0.6 is that honest uncertainty, not
a failure.

### What the noise guess really is: the score

Flip the noise guess around and it becomes an arrow pointing *toward the
data*. If the noise pushed the point up and to the right, the way back is
down and to the left. That arrow has a name, the **score**: at any point,
the direction in which the data gets more crowded, fastest. Learning to
guess noise is secretly learning the score at every noise level, which is
why diffusion models are also called *score-based models*, and why the
training trick is called **denoising score matching**.

$$
\nabla_{x} \log p_t(x_t) \;\approx\; -\,\frac{\varepsilon_\theta(x_t, t)}{\sqrt{1 - \bar\alpha_t}}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $p_t(x)$ | how crowded the noisy data is at point $x$ at step $t$ (its probability density) | the standard bell curve |
| $\log p_t$ | the **logarithm** of that density; crowdedness on a scale where multiplying becomes adding | $-x^2/2$ + a constant |
| $\nabla_x$ | "grad": the slope, in every direction, as $x$ moves; for one number, just the slope | slope of $-x^2/2$ is $-x$ |
| $\nabla_x \log p_t$ | the **score**: which way is uphill in crowdedness | $-1.5$ at $x = 1.5$ |
| $\varepsilon_\theta$ | the network's noise guess | 0.9 |
| $\sqrt{1-\bar\alpha_t}$ | the noise share, which converts "noise" units into "distance" units | 0.6 |
| $\approx$ | "approximately equal": exact for a perfect guesser | |

**In words:** "the direction toward the data is the noise guess, flipped and
divided by the noise share."

**With the numbers:** take data that is already standard-normal, so every
noisy version is the plain bell curve, whose score at $x$ is $-x$ (the slope
of $-x^2/2$). At $x_t = 1.5$ with $\bar\alpha_t = 0.64$, the best possible
noise guess is $0.6 \times 1.5 = 0.9$, and the formula gives
$-0.9 / 0.6 = -1.5$: exactly the score, pointing back toward the crowd at 0.

**In Python:**

```python
import math
x_t, alpha_bar = 1.5, 0.64
# the best noise guess for standard-normal data: √(1 − ᾱ) · x_t
eps_hat = math.sqrt(1 - alpha_bar) * x_t
round(eps_hat, 2)  # → 0.9
# −ε̂ / √(1 − ᾱ): the score, which for the bell curve is −x
round(-eps_hat / math.sqrt(1 - alpha_bar), 2)  # → -1.5
```

**In code:** `DenoiserMLP` is the network with its hand-written backward pass (`denoiser_gradient_check` compares it with finite differences), `time_features` turns the step into waves, `train_noise_predictor` is the training loop in the diagram and returns a `NoisePredictor`, and `noise_to_score` is the score formula.

**Why it matters in practice.** This loss, a squared error on guessed noise,
is the training loss of the original DDPM paper and of the first Stable
Diffusion models. The network is far bigger and the data is images, but the
training loop is the boxes above.

## Step 3: sampling, running the process backwards

**Everyday picture.** Back to the sculptor. Look at the block, guess which
bits are "not horse", chip away a small part of them, look again. Never
chip away everything you think is wrong at once: the guess is rough when the
block is rough, and it gets better as the shape emerges.

**Tiny worked example.** A point sits at $x_t = 1.0$ at a step with
$\beta_t = 0.19$ and $\bar\alpha_t = 0.75$. The network guesses the noise in
it is $\hat\varepsilon = 0.5$. One step back removes a scaled share of that
guess, undoes the shrinking, and (except on the very last step) adds a
little fresh noise.

$$
x_{t-1} = \frac{1}{\sqrt{1-\beta_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\, \hat\varepsilon \right) + \sqrt{\beta_t}\; z
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $x_t$ | the current, noisier point | 1.0 |
| $\hat\varepsilon$ | the network's noise guess, $\varepsilon_\theta(x_t, t)$ | 0.5 |
| $\beta_t$ | this step's noise dose, from the schedule | 0.19 |
| $\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}$ | how much of the guess belongs to *this one step*: only a sliver of all the noise was added here | $0.19 / 0.5 = 0.38$ |
| $\frac{1}{\sqrt{1-\beta_t}}$ | undo this step's shrink | $1/0.9$ |
| $z$ | fresh standard-normal noise; 0 on the final step | 0 or 0.5 |
| $\sqrt{\beta_t}\, z$ | a small random wobble | $0.436 \cdot 0.5 = 0.218$ |
| $x_{t-1}$ | the slightly cleaner point | 0.9, or 1.118 with the wobble |

**In words:** "subtract this step's share of the guessed noise, scale back
up, and add a small fresh wobble."

**With the numbers:** $(1.0 - 0.38 \times 0.5) / 0.9 = 0.81 / 0.9 = 0.9$;
with the wobble $z = 0.5$: $0.9 + 0.218 = 1.118$.

**In Python:**

```python
import math
x_t, eps_hat = 1.0, 0.5
beta, alpha_bar = 0.19, 0.75
# remove this step's share of the guessed noise, then undo the shrink
mean = (x_t - beta / math.sqrt(1 - alpha_bar) * eps_hat) / math.sqrt(1 - beta)
round(mean, 4)  # → 0.9
# add a small fresh wobble z
z = 0.5
round(mean + math.sqrt(beta) * z, 4)  # → 1.1179
```

Why add noise while trying to remove it? The network's guess is an average
over every clean point that could have produced $x_t$. Stepping to that
average would drift every sample toward the safe, blurry middle. The fresh
wobble keeps each sample committed to one specific possibility, so the
samples spread out over the whole data, as the real data does. This recipe
is **DDPM** (denoising diffusion probabilistic models).

```mermaid
flowchart LR
  N["x_T: draw pure noise"] --> G["network guesses<br/>the noise ε̂ at step t"]
  G --> R["remove this step's share,<br/>undo the shrink"]
  R --> W["add a small fresh wobble<br/>(not on the last step)"]
  W --> Q{"t = 0?"}
  Q -- "no: t ← t − 1" --> G
  Q -- "yes" --> OUT["x₀: a new sample"]
```

**Reading it:** the loop runs once per step, $T = 100$ times here and 1,000
in the original paper. Each lap costs one full run of the network, so the
number of laps is the price of a sample. Keep that in mind: the rest of the
lesson is largely about making that loop shorter.

![Five panels of purple dots over grey blobs: a round noise cloud at step 100 is still shapeless at 40 and 20, splits into four clusters by 10, and matches the blobs at 0](figures/primer.ml.generative.diffusion.denoising.svg)

**Reading it:** grey dots are the real data, purple dots are 600 samples
being generated, shown at five moments of the backwards run. For the first
half almost nothing seems to happen: the network makes coarse, whole-cloud
decisions while the noise is still loud. Between steps 20 and 10 the cloud
splits toward the four corners, and the last few steps tighten each group
onto its blob. Nobody told the network there were four blobs; it learned
that from guessing noise.

### Fewer steps: jump, don't crawl (DDIM)

**Everyday picture.** A sculptor in a hurry doesn't chip a sliver per look.
They glance at the block, picture the *finished* horse, then rough the block
out to a state only slightly less finished than that, and look again. Big
confident moves, a handful of looks.

**Tiny worked example.** From the Step 1 pixel: $x_t = 1.9$ at
$\bar\alpha_t = 0.64$, and suppose the network guesses the noise perfectly,
$\hat\varepsilon = 0.5$. First predict the finished value, then re-noise it
to a much quieter level, $\bar\alpha_s = 0.96$, reusing the same noise guess
instead of drawing new noise.

$$
\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\;\hat\varepsilon}{\sqrt{\bar\alpha_t}}
\qquad\qquad
x_s = \sqrt{\bar\alpha_s}\;\hat x_0 + \sqrt{1-\bar\alpha_s}\;\hat\varepsilon
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $x_t$ | the current point, at step $t$ | 1.9 |
| $\hat\varepsilon$ | the network's noise guess | 0.5 |
| $\hat x_0$ | "x-zero hat": the predicted clean point, from undoing the shortcut | 2.0 |
| $s$ | the earlier step to jump to; any step before $t$, not just $t - 1$ | 96% signal |
| $\bar\alpha_t, \bar\alpha_s$ | the signal left at steps $t$ and $s$ | 0.64 and 0.96 |
| $x_s$ | the point after the jump: the predicted clean point, re-noised lightly | 2.06 |

**In words:** "guess the finished point by undoing the shortcut, then walk
back to a quieter noise level along the same noise direction, with no fresh
randomness."

**With the numbers:** $\hat x_0 = (1.9 - 0.6 \cdot 0.5) / 0.8 = 1.6 / 0.8 =
2.0$, then $x_s = \sqrt{0.96} \cdot 2.0 + \sqrt{0.04} \cdot 0.5 = 1.960 +
0.1 = 2.06$. Jump to $\bar\alpha_s = 1$ and you land on $\hat x_0$ itself.

**In Python:**

```python
import math
x_t, eps_hat = 1.9, 0.5
alpha_bar_t, alpha_bar_s = 0.64, 0.96
# x̂₀: undo the shortcut to predict the clean point
x0_hat = (x_t - math.sqrt(1 - alpha_bar_t) * eps_hat) / math.sqrt(alpha_bar_t)
round(x0_hat, 4)  # → 2.0
# x_s: re-noise it lightly, along the same noise direction
round(math.sqrt(alpha_bar_s) * x0_hat + math.sqrt(1 - alpha_bar_s) * eps_hat, 4)  # → 2.0596
```

This is **DDIM** (denoising diffusion implicit models). It uses the very
same trained network; only the sampling loop changes. Because it adds no
fresh noise, the same starting noise always gives the same sample, and it
can take 20 jumps instead of 100 small steps. On the blobs, 20 DDIM jumps
land as close to the data as 100 DDPM steps (the spec checks both).

**In code:** `ddpm_step` and `ddpm_sample` are the small-step sampler (`ddpm_trajectory` keeps snapshots for the figure); `ddim_step` and `ddim_sample` are the jumping one; `two_way_distance` measures how close samples land to the data, in both directions, so piling every sample onto one blob can't score well.

**Why it matters in practice.** Sampling cost is the number of network
calls. The original DDPM used 1,000; samplers like DDIM brought that to
about 20 to 50, which is what made diffusion usable in products. Pushing
toward a handful of steps is the next idea.

## Step 4: flow matching, the straight road

**Everyday picture.** Diffusion's road from noise to data is a winding
mountain path set by the schedule. Flow matching asks: why not draw a
straight line from each noise point to a data point, and learn the *current*
that carries you along it? The network becomes a map of arrows, like a
weather map of wind: at every place and time, it says which way to drift,
and how fast. To generate, drop a leaf (a noise point) on the map and let it
drift.

**One warning about labels.** In flow-matching papers, and in this section,
$x_0$ is the **noise** and $x_1$ is the **data**, and time runs from 0
(noise) to 1 (data). That is the opposite of the diffusion sections above,
where $x_0$ was the clean data.

**Tiny worked example.** Noise point $x_0 = -1$, data point $x_1 = 2$. The
straight line between them passes, a quarter of the way along, through
$0.75 \cdot (-1) + 0.25 \cdot 2 = -0.25$. The speed along the line is
constant: it covers the distance $2 - (-1) = 3$ in one unit of time, so the
velocity is 3 everywhere on it. The network is trained to output 3 when
shown the point $-0.25$ at time 0.25.

$$
x_t = (1 - t)\, x_0 + t\, x_1
\qquad\qquad
L(\theta) = \mathbb{E}_{x_0,\, x_1,\, t}\Big[\, \big\| v_\theta(x_t, t) - (x_1 - x_0) \big\|^2 \,\Big]
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $x_0$ | a noise point (standard normal) | −1 |
| $x_1$ | a real data point, paired with the noise at random | 2 |
| $t$ | time along the line, from 0 (noise) to 1 (data), drawn at random in training | 0.25 |
| $x_t$ | the point on the straight line at time $t$ | −0.25 |
| $x_1 - x_0$ | the **velocity** of that line: where to go, and how fast; the same at every $t$ | 3 |
| $v_\theta(x_t, t)$ | the network's guessed velocity at that place and time | say 2.5 |
| $\lVert \ldots \rVert^2$, $\mathbb{E}$ | squared length and average over many random draws, as in Step 2 | $(2.5 - 3)^2 = 0.25$ |

**In words:** "pick a noise point, a data point and a time; find the point
that far along the straight line between them; train the network to output
the line's direction and speed there."

**With the numbers:** $x_{0.25} = 0.75 \cdot (-1) + 0.25 \cdot 2 = -0.25$;
target velocity $2 - (-1) = 3$; a guess of 2.5 scores $(2.5 - 3)^2 = 0.25$.

**In Python:**

```python
x0, x1, t = -1.0, 2.0, 0.25
# (1 − t) x₀ + t x₁: the point on the straight line
x_t = (1 - t) * x0 + t * x1
x_t  # → -0.25
# x₁ − x₀: the velocity to learn
x1 - x0  # → 3.0
# a guess of 2.5 is graded by squared error
(2.5 - (x1 - x0)) ** 2  # → 0.25
```

To generate, start from noise at $t = 0$ and follow the arrows in a few
**Euler steps** (the simplest way to follow a velocity: move in a straight
line for a short time, then look again):

$$
x_{t+h} = x_t + h\; v_\theta(x_t, t)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $h$ | the step size in time; $N$ equal steps means $h = 1/N$ | 0.75 |
| $v_\theta(x_t, t)$ | the velocity the network reports here and now | 3 |
| $x_{t+h}$ | where you are after moving for time $h$ | 2.0 |

**In words:** "move for a short time in the direction, and at the speed, the
network says."

**With the numbers:** from $x_{0.25} = -0.25$, one step of $h = 0.75$ at
velocity 3 gives $-0.25 + 0.75 \times 3 = 2.0$: exactly the data point. On a
straight line, one step of any size is exact, because there is no bend to
cut.

**In Python:**

```python
x_t, v, h = -0.25, 3.0, 0.75
# x + h v: one Euler step
x_t + h * v  # → 2.0
```

```mermaid
flowchart LR
  subgraph TRAIN["Training"]
    direction LR
    P["pair noise x₀<br/>with data x₁"] --> L1["point on the line<br/>(1 − t) x₀ + t x₁"]
    L1 --> V["network guesses v"]
    V --> LOSS["loss ‖v − (x₁ − x₀)‖²"]
  end
  subgraph SAMPLE["Sampling"]
    direction LR
    Z["draw noise, t = 0"] --> STEP["x ← x + h · v(x, t)"]
    STEP --> C{"t = 1?"}
    C -- no --> STEP
    C -- yes --> S["a new sample"]
  end
```

**Reading it:** training (top) is the same shape as diffusion's: random
choices in, a point in between, a squared-error guess. The difference is
the target: a velocity along a straight line rather than hidden noise.
Sampling (bottom) is a plain loop of Euler steps from $t = 0$ to $t = 1$,
with no schedule, no shrink factors and no fresh noise.

There is a catch, and it is instructive. Many different straight lines pass
through the same point (from different noise to different data), so the
network cannot know which one it is on. It learns the *average* of their
velocities. The learned paths are therefore only roughly straight: they
bend, because two paths can never cross (at any point the network gives one
velocity, so two paths that met would continue together).

![Left: 16 straight lines from noise to data that cross each other. Right: the learned flow's 16 paths from the same noise, which never cross and curve to reach the blobs](figures/primer.ml.generative.diffusion.flow_paths.svg)

**Reading it:** hollow circles are noise, filled circles are where each path
ends. On the left are training pairs: every line is straight, but they
criss-cross because noise and data were paired at random. On the right, the
learned flow starts from the same noise. Its paths never cross, so they
bend to share out the blobs, mostly near the end. The straighter the paths,
the fewer Euler steps you need. That is why *rectified flow* retrains on
the model's own (noise, sample) pairs, which no longer cross: each round
makes the paths straighter, until one or two steps are enough.

The average also explains the extreme case. With a single Euler step from
$t = 0$, every noise point moves by the average velocity toward "the data
in general" and lands on the data's average: the empty middle of the four
blobs (the spec checks it). Straight-line training does not make one step
free; straight *learned* paths do.

![Sample quality against the number of network calls, on log axes: at one or two calls both are about as far off as plain noise; flow matching nears the fresh-data floor by about 5 calls, diffusion by about 10](figures/primer.ml.generative.diffusion.steps.svg)

**Reading it:** the x-axis is network calls per sample and the y-axis is the
two-way distance between samples and data (lower is better). The green
floor is a fresh draw of the real blobs, as good as samples can get; the
grey line is plain noise. At one or two calls neither method does much
better than noise. Flow matching (teal) gets close to the floor within about
5 calls, while diffusion's DDIM jumps (purple) need about 10 to get as
close. Same network size, same training time: the straighter paths need
roughly half the steps.

**In code:** `flow_point` and `flow_target` build the training pairs, `train_velocity_predictor` trains a `VelocityPredictor`, `euler_step` and `euler_sample` follow it, and `step_sweep` measures both methods at each step count.

**Why it matters in practice.** Flow matching and rectified flow give a
simpler recipe (no noise schedule to tune, just straight lines) and need
fewer sampling steps. Stable Diffusion 3 is trained as a rectified flow, and
many recent image and video generators follow. Under the hood it is the same
family as diffusion: a network trained by squared error to point from noise
toward data, followed step by step.

## Step 5: conditioning and guidance, generating what you asked for

**Everyday picture.** A caricaturist draws a famous face by noticing what
makes it different from an average face (the big chin, the eyebrows) and
exaggerating exactly that difference. Guidance does the same: compare "what
I'd draw if told what to draw" with "what I'd draw anyway", and push further
in the direction of the difference.

**Conditioning** first: to ask for a specific corner, feed the label to the
network as an extra input. Our labels are four corners, given as a
**one-hot** vector (all zeros except a 1 in the chosen slot), plus a fifth
slot that means "no label". A text-to-image model does the same with a
sentence in place of a corner name.

**Classifier-free guidance** trains *one* network to answer both questions:
in training, the label is hidden (replaced by "no label") on a random 20% of
examples. At sampling time the network is asked twice per step, once with
the label and once without, and the two guesses are mixed.

**Tiny worked example.** At some point during sampling, the guess without
the label is $\hat\varepsilon_\varnothing = 0.2$ and the guess with the label
"south-west" is $\hat\varepsilon_c = 0.5$. The label moves the guess by
$0.5 - 0.2 = 0.3$. With guidance weight $w = 3$, move three times as far.

$$
\tilde\varepsilon = \hat\varepsilon_\varnothing + w \,\big(\hat\varepsilon_c - \hat\varepsilon_\varnothing\big)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $\hat\varepsilon_\varnothing$ | the noise guess with the label hidden ($\varnothing$, "empty set", stands for "no label") | 0.2 |
| $\hat\varepsilon_c$ | the noise guess given the label $c$ | 0.5 |
| $\hat\varepsilon_c - \hat\varepsilon_\varnothing$ | what the label changes: the direction "more like $c$" | 0.3 |
| $w$ | the **guidance weight** (or guidance scale): 0 ignores the label, 1 is the plain labelled guess, above 1 exaggerates | 3 |
| $\tilde\varepsilon$ | "epsilon tilde": the guided guess, used in place of $\hat\varepsilon$ by the sampler | 1.1 |

**In words:** "start from the unlabelled guess and move $w$ times as far as
the label would move it."

**With the numbers:** $0.2 + 3 \times (0.5 - 0.2) = 0.2 + 0.9 = 1.1$. With
$w = 0$ it is 0.2 (the label is ignored); with $w = 1$ it is 0.5 (exactly
the labelled guess).

**In Python:**

```python
eps_uncond, eps_cond = 0.2, 0.5
# ε̃ = ε̂_∅ + w (ε̂_c − ε̂_∅), for w = 0, 1 and 3
[round(eps_uncond + w * (eps_cond - eps_uncond), 2) for w in (0, 1, 3)]  # → [0.2, 0.5, 1.1]
```

```mermaid
flowchart LR
  X["noisy point x_t, step t"] --> A["network, label hidden"]
  X --> B["network, label = south-west"]
  A --> U["ε̂_∅"]
  B --> C["ε̂_c"]
  U --> MIX["ε̃ = ε̂_∅ + w (ε̂_c − ε̂_∅)"]
  C --> MIX
  MIX --> STEP["one sampler step<br/>(DDIM or DDPM)"]
```

**Reading it:** each step runs the *same* network twice, once with the
label hidden and once with it shown, which doubles the cost of a step. The
two guesses meet in the mixing box, and only the mixed guess reaches the
sampler, which is otherwise unchanged. Nothing new was trained for guidance:
it is a choice made at sampling time.

![Three panels of 400 green samples asked to be south-west: at w = 0 they spread over all four blobs (27% south-west); at w = 1 all land on the south-west blob with spread 0.27; at w = 3 they bunch into a tight knot at its outer edge](figures/primer.ml.generative.diffusion.guidance.svg)

**Reading it:** grey dots are the data, green dots are samples asked for
"south-west" at three guidance weights. At $w = 0$ the label is ignored, so
samples land in all four corners, only about a quarter in the south-west.
At $w = 1$ nearly all land in the right blob with roughly the real blob's
spread. At $w = 3$ they are all on target but bunched into a tight knot,
pushed to the side of the blob *farthest from the other blobs*: the most
unmistakably south-west spot. That is guidance in one picture: more
on-label, less varied, and exaggerated.

**In code:** `network_inputs` appends the one-hot label with its "no label" slot, `train_noise_predictor` hides labels at random when given them, `guided_noise` is the formula, `ddim_sample` applies it at every jump when given a label, and `guidance_sweep` measures the share on target and the spread at each weight.

**Why it matters in practice.** Text-to-image systems use guidance weights
well above 1, because unguided samples follow the prompt only loosely. Turn
it too high and images become oversaturated and samey, the image version of
the tight knot above. The "guidance scale" slider in image tools is this $w$.

## Step 6: scaling up to real images, video and audio

**Everyday picture.** An architect doesn't design a skyscraper by placing
every brick. They draw a floor plan, small enough to think about, and a
builder turns the plan into a building. Latent diffusion does the creative
work on a small "plan" of the image, and a separate decoder builds the
pixels.

**Tiny worked example.** A 512 × 512 colour image is $512 \times 512 \times 3
= 786{,}432$ numbers. An **autoencoder** (a network that squeezes an image
into a small code and rebuilds it; see `primer.ml.generative.autoencoders`)
shrinks each side by 8 and keeps 4 numbers per position: $64 \times 64 \times
4 = 16{,}384$ numbers. The denoiser now works on 48 times fewer numbers, and
every one of its many steps is that much cheaper.

$$
\text{shrink factor} = \frac{H \cdot W \cdot 3}{(H/f) \cdot (W/f) \cdot c}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $H, W$ | image height and width in pixels | 512, 512 |
| 3 | colour channels: red, green, blue | 3 |
| $f$ | how much the autoencoder shrinks each side | 8 |
| $c$ | numbers kept per latent position (latent channels) | 4 |
| shrink factor | how many times fewer numbers the denoiser handles | 48 |

**In words:** "count the numbers in the image, count the numbers in its
latent code, and divide."

**With the numbers:** $786{,}432 / 16{,}384 = 48$. The latent is then cut
into 2 × 2 patches, each patch becoming one token for a transformer:
$(64/2) \times (64/2) = 1{,}024$ tokens. A 16-frame video clip in the same
latent space is 16 times that, 16,384 tokens, and since attention compares
every token with every other (`primer.ml.attention`), it costs $16^2 = 256$
times as much.

**In Python:**

```python
H, W, f, c = 512, 512, 8, 4
pixels = H * W * 3
latents = (H // f) * (W // f) * c
pixels, latents  # → (786432, 16384)
# shrink factor
pixels / latents  # → 48.0
# 2 × 2 patches of the 64 × 64 latent: the transformer's tokens
tokens = (64 // 2) * (64 // 2)
tokens  # → 1024
# 16 video frames: 16× the tokens, 256× the attention pairs
(16 * tokens) ** 2 // tokens ** 2  # → 256
```

```mermaid
flowchart LR
  P["prompt: 'a red fox in snow'"] --> TE["text encoder<br/>(e.g. CLIP's text tower)"]
  TE --> TOK["text token vectors"]
  N["random latent noise<br/>64 × 64 × 4"] --> DEN
  TOK --> DEN["denoiser: a transformer<br/>over latent patches,<br/>attending to the text"]
  DEN -- "20 to 50 steps,<br/>with guidance" --> DEN
  DEN --> LAT["clean latent"]
  LAT --> DEC["autoencoder decoder"]
  DEC --> IMG["512 × 512 image"]
```

**Reading it:** follow the noise from the left. It is not an image but a
small latent, and all the sampling steps (the loop on the denoiser box)
happen in that cheap space. The prompt takes the upper path: a **text
encoder**, such as CLIP's text half (`primer.ml.embeddings.contrastive`),
turns it into token vectors that the denoiser reads through
**cross-attention** (attention whose queries come from the image patches and
whose keys and values come from the text tokens). Guidance's "no label"
answer is simply an empty prompt. Only at the very end does the decoder
turn the latent into pixels, once.

Three more scale-ups follow the same pattern:

- **The denoiser became a transformer.** Early systems used a U-Net (a
  convolutional network, `primer.ml.cnn_rnn`); the **diffusion transformer**
  (DiT) cuts the latent into patches and treats them as tokens, so the
  scaling lessons of language models carry over.
- **Video** adds a time axis: the latent is a stack of frames, and patches
  span space and time. It is the same denoising, with far more tokens.
- **Audio** is denoised as a spectrogram (a picture of sound: time across,
  pitch up, loudness as brightness) or as an audio autoencoder's latent.

**Compared with the other generators in this part:**

| | GAN (`primer.ml.generative.gans`) | VAE (`primer.ml.generative.autoencoders`) | Diffusion and flow matching |
|---|---|---|---|
| Training | a generator against a critic: unstable | reconstruct plus stay near a simple code: stable | squared error on noise or velocity: stable |
| Samples | sharp | often blurry | sharp |
| Covers all of the data? | can **mode collapse** (ignore whole regions) | yes | yes |
| Cost of one sample | one network call | one network call | 5 to 50 network calls |

**In code:** `latent_shrink` counts pixels against latent numbers and `patch_tokens` counts a diffusion transformer's tokens, per frame.

**Why it matters in practice.** Latent space made high-resolution diffusion
affordable on a single GPU, transformers made it scale, text encoders made
it follow prompts, and guidance made it follow them closely. The price that
remains is many network calls per sample, which is why step-reduction
(DDIM, flow matching, distillation into few-step students) is where so much
engineering effort goes.

## In 20 seconds

- **Forward process:** mix data with a little Gaussian noise per step until
  only noise is left; the shortcut $x_t = \sqrt{\bar\alpha_t}\,x_0 +
  \sqrt{1-\bar\alpha_t}\,\varepsilon$ jumps to any step in one go.
- **Training:** show the network a noised example and the step; it guesses
  the noise; grade by squared error. Plain, stable regression, and secretly
  learning the score (the direction toward the data).
- **Sampling:** start from pure noise and repeatedly remove a little guessed
  noise (DDPM, with a fresh wobble each step), or take a few big
  deterministic jumps (DDIM).
- **Flow matching:** learn the velocity along straight lines from noise to
  data and follow it with Euler steps; straighter paths need fewer steps.
- **Guidance:** ask the same network with and without the label and push
  past the labelled guess; more on-prompt, less varied.
- **At scale:** denoise an autoencoder's small latent with a transformer that
  reads a text encoder's tokens; video and audio are the same idea with more
  tokens.

## Self-test questions

**Why does the network learn to guess the noise, rather than the clean
picture?**
The two carry the same information: given the noisy point, the step and the
noise, the clean point follows by undoing the shortcut. Guessing the noise
works better in practice because the target is always the same size (a
standard-normal draw) at every step, which makes one network easy to train
across all noise levels. The noise guess, flipped and rescaled, is also the
score: the direction toward the data.

**Training only ever takes one jump from clean to noisy. Why does
generation need many steps back?**
The noise guess is an average over every clean point that could have
produced the noisy one. From heavy noise that average is vague, pointing at
the middle of the data, so one big step lands on a blur (or, in our blobs,
the empty middle). Small steps let the guess sharpen as the sample commits
to one specific region, and the network is asked again at every stage.

**Why does DDPM add fresh noise at each step, when the goal is to remove
noise?**
Without it, every step moves toward the network's averaged guess and
samples drift toward safe, typical, blurry results. The small fresh wobble
keeps each sample exploring one specific possibility, so the samples cover
all of the data. DDIM drops the wobble on purpose, trading that randomness
for determinism and big jumps.

**What does flow matching change, and why can it get away with fewer
steps?**
It replaces the noise schedule with straight lines from noise to data and
trains the network to output the velocity along them. Following a velocity
with Euler steps is only exact on straight paths; the learned paths are
straighter than diffusion's, so fewer steps cut fewer corners. They are not
perfectly straight, because paths can't cross, which is what rectified
flow's retraining fixes.

**What does the guidance weight do, and what goes wrong if it is too
large?**
It sets how far past the labelled guess to go, along the direction from the
unlabelled guess to the labelled one. 0 ignores the prompt, 1 follows it
plainly, above 1 exaggerates it, so samples match the prompt more reliably.
Too large and samples become samey, over-saturated caricatures of the
prompt, bunched into the most extreme examples.

**Why do real image generators denoise in a latent space instead of on
pixels?**
Most of an image's pixel values are fine texture that a decoder can fill in.
An autoencoder shrinks a 512 × 512 image 48-fold into a latent that keeps
the meaningful structure, and since sampling runs the denoiser dozens of
times, every step becomes that much cheaper. The decoder runs just once, at
the end.

**When would you choose a GAN over a diffusion model?**
When one-shot speed matters most: a GAN makes a sample in a single network
call, where diffusion needs several to dozens. Diffusion wins on training
stability and on covering all of the data (GANs can mode-collapse), which is
why it took over image generation; distillation is now closing its speed
gap.

## The papers behind this lesson

- **Sohl-Dickstein et al., *Deep Unsupervised Learning using Nonequilibrium
  Thermodynamics* (2015)**: https://arxiv.org/abs/1503.03585. The original
  idea: destroy data slowly with noise, and learn to reverse the
  destruction.
- **Song and Ermon, *Generative Modeling by Estimating Gradients of the Data
  Distribution* (2019)**: https://arxiv.org/abs/1907.05600. Generated
  samples by learning the score at many noise levels and following it.
- **Ho, Jain and Abbeel, *Denoising Diffusion Probabilistic Models*
  (2020)**: https://arxiv.org/abs/2006.11239. The simple "guess the noise"
  loss and the sampler of Step 3, with the first high-quality image
  results.
- **Song, Meng and Ermon, *Denoising Diffusion Implicit Models* (2020)**:
  https://arxiv.org/abs/2010.02502. Deterministic sampling with big jumps,
  using the same trained network.
- **Song et al., *Score-Based Generative Modeling through Stochastic
  Differential Equations* (2020)**: https://arxiv.org/abs/2011.13456. Showed
  diffusion and score-based models are one family, described as continuous
  time processes.
- **Ho and Salimans, *Classifier-Free Diffusion Guidance* (2022)**:
  https://arxiv.org/abs/2207.12598. Guidance from one network trained with
  and without the label, no separate classifier needed.
- **Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion
  Models* (2021)**: https://arxiv.org/abs/2112.10752. Denoising in an
  autoencoder's latent space, with text via cross-attention: the basis of
  Stable Diffusion.
- **Peebles and Xie, *Scalable Diffusion Models with Transformers*
  (2022)**: https://arxiv.org/abs/2212.09748. Replaced the U-Net with a
  transformer over latent patches, and showed it improves with scale.
- **Lipman et al., *Flow Matching for Generative Modeling* (2022)**:
  https://arxiv.org/abs/2210.02747. Trained continuous flows by regressing
  velocities along simple paths from noise to data.
- **Liu, Gong and Liu, *Flow Straight and Fast: Learning to Generate and
  Transfer Data with Rectified Flow* (2022)**:
  https://arxiv.org/abs/2209.03003. Straight-line paths, and retraining on
  the model's own pairs to straighten the learned flow for few-step
  sampling.
- **Esser et al., *Scaling Rectified Flow Transformers for High-Resolution
  Image Synthesis* (2024)**: https://arxiv.org/abs/2403.03206. Rectified
  flow with a transformer denoiser at scale: Stable Diffusion 3.

## Further reading

- Ho, Jain and Abbeel, *Denoising Diffusion Probabilistic Models* (2020): https://arxiv.org/abs/2006.11239
- Song, Meng and Ermon, *Denoising Diffusion Implicit Models* (2020): https://arxiv.org/abs/2010.02502
- Ho and Salimans, *Classifier-Free Diffusion Guidance* (2022): https://arxiv.org/abs/2207.12598
- Lipman et al., *Flow Matching for Generative Modeling* (2022): https://arxiv.org/abs/2210.02747
- Liu, Gong and Liu, *Rectified Flow* (2022): https://arxiv.org/abs/2209.03003
- Rombach et al., *Latent Diffusion Models* (2021): https://arxiv.org/abs/2112.10752
- Peebles and Xie, *Diffusion Transformers* (2022): https://arxiv.org/abs/2212.09748
- Lilian Weng, *What are Diffusion Models?*: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
- Hugging Face, *The Annotated Diffusion Model* (DDPM, line by line in code): https://huggingface.co/blog/annotated-diffusion
- Hugging Face Diffusers documentation: https://huggingface.co/docs/diffusers/index
"""

from __future__ import annotations

import os

# Training multiplies thousands of tiny matrices. Multi-threaded BLAS spends far longer coordinating
# threads than computing on matrices this small (the demo takes 10 s instead of 2 s), so use one
# thread, as tests/conftest.py and the Makefile do. This only works before NumPy is first imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import functools  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

import numpy as np  # noqa: E402

from primer._show import banner, say, table, takeaway  # noqa: E402
from primer.ml.optimizers import Adam  # noqa: E402

# ---------------------------------------------------------------------------
# 1. The toy data: four blobs, one per corner
# ---------------------------------------------------------------------------

CORNERS = ("north-east", "north-west", "south-west", "south-east")
SOUTH_WEST = CORNERS.index("south-west")


def make_blobs(n: int = 600, spread: float = 0.3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Four round blobs of 2-D points, one per corner, and each point's corner label.

    The blobs sit at distance 1 from the origin, so the data's average is the
    empty middle: a model that outputs "the average" lands where no data lives.
    Returns points (n, 2) and labels (n,) indexing `CORNERS`.
    """
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % len(CORNERS)
    angles = np.pi / 4 + labels * np.pi / 2  # 45°, 135°, 225°, 315°
    centres = np.c_[np.cos(angles), np.sin(angles)]
    return centres + spread * rng.standard_normal((n, 2)), labels


# ---------------------------------------------------------------------------
# 2. The forward (noising) process
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoiseSchedule:
    """How much noise each step adds.

    `betas[t]` is step t's noise share, for t = 1..T; `betas[0] = 0` stands for
    the clean data, so every array is indexed by the step number itself.
    """

    betas: np.ndarray

    @property
    def T(self) -> int:
        return len(self.betas) - 1

    @property
    def alphas(self) -> np.ndarray:
        """α_t = 1 − β_t: the share of the previous step's signal that survives step t."""
        return 1.0 - self.betas

    @property
    def alpha_bars(self) -> np.ndarray:
        """ᾱ_t = α_1 · α_2 · … · α_t: the share of the original data left after t steps (ᾱ_0 = 1)."""
        return np.cumprod(self.alphas)


def linear_schedule(T: int = 100, beta_first: float = 1e-4, beta_last: float = 0.1) -> NoiseSchedule:
    """β rising in a straight line from `beta_first` to `beta_last` over T steps.

    Small steps first, while the picture still has fine detail to lose; bigger
    ones later, when there is little left. With the defaults, ᾱ_T ≈ 0.006.
    """
    return NoiseSchedule(np.r_[0.0, np.linspace(beta_first, beta_last, T)])


def noise_step(x_prev, beta: float, eps):
    """One forward step: x_t = √(1 − β_t) · x_{t−1} + √β_t · ε.

    Shrinking the signal by √(1 − β) while adding √β of noise keeps the total
    variance at 1 when it starts at 1, so the numbers never blow up.
    """
    return np.sqrt(1.0 - beta) * x_prev + np.sqrt(beta) * eps


def add_noise(x0, alpha_bar, eps):
    """The shortcut: jump straight to step t, x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε.

    `alpha_bar` may be one number or one per row of `x0` (shape (n,) or (n, 1)).
    Training uses this to noise any example to any step in one line.
    """
    alpha_bar = np.asarray(alpha_bar, dtype=float)
    if alpha_bar.ndim == 1:
        alpha_bar = alpha_bar[:, None]  # one ᾱ per row, broadcast over the 2 coordinates
    return np.sqrt(alpha_bar) * x0 + np.sqrt(1.0 - alpha_bar) * eps


# ---------------------------------------------------------------------------
# 3. The denoiser: a small MLP with a hand-written backward pass
# ---------------------------------------------------------------------------


def time_features(t, n_features: int = 8) -> np.ndarray:
    """Turn a time in [0, 1] into sines and cosines at doubling frequencies.

    A single number t is hard for a small network to use; a spread of waves
    lets it tell t = 0.30 from t = 0.32 (the same trick as sinusoidal
    positions in `primer.ml.positional`). Shape (n, n_features).
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    freqs = np.pi * 2.0 ** np.arange(n_features // 2)
    angles = t[:, None] * freqs[None, :]
    return np.c_[np.sin(angles), np.cos(angles)]


def network_inputs(x: np.ndarray, t, labels=None, n_classes: int = 0) -> np.ndarray:
    """Glue together what the network sees: the noisy point, the time, and a label.

    Labels are one-hot over n_classes + 1 slots; the extra last slot means
    "no label", which is how one network learns both the conditional and the
    unconditional guess (see the guidance section).
    """
    t = np.broadcast_to(np.asarray(t, dtype=float), (len(x),))
    parts = [x, time_features(t)]
    if n_classes:
        onehot = np.zeros((len(x), n_classes + 1))
        onehot[np.arange(len(x)), np.broadcast_to(labels, (len(x),))] = 1.0
        parts.append(onehot)
    return np.concatenate(parts, axis=1)


@dataclass
class DenoiserMLP:
    """inputs → ReLU(· W1 + b1) → ReLU(· W2 + b2) → · W3 + b3: two hidden layers, 2 outputs.

    The same shape of network serves as a noise predictor (output = ε̂) and as
    a velocity predictor (output = v̂). `primer.ml.neural_net` builds this kind
    of network, and its backward pass, from scratch.
    """

    n_in: int
    hidden: int = 64
    n_out: int = 2
    seed: int = 0
    params: dict[str, np.ndarray] = field(init=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        # He-style init for ReLU (variance 2/fan_in); the last layer starts small
        # so the first guesses are near zero rather than wild.
        self.params = {
            "W1": rng.normal(0, np.sqrt(2 / self.n_in), (self.n_in, self.hidden)),
            "b1": np.zeros(self.hidden),
            "W2": rng.normal(0, np.sqrt(2 / self.hidden), (self.hidden, self.hidden)),
            "b2": np.zeros(self.hidden),
            "W3": rng.normal(0, 0.1 / np.sqrt(self.hidden), (self.hidden, self.n_out)),
            "b3": np.zeros(self.n_out),
        }

    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, dict]:
        p = self.params
        z1 = inputs @ p["W1"] + p["b1"]  # (n, hidden)
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ p["W2"] + p["b2"]  # (n, hidden)
        h2 = np.maximum(z2, 0.0)
        out = h2 @ p["W3"] + p["b3"]  # (n, 2): the guess
        return out, {"inputs": inputs, "z1": z1, "h1": h1, "z2": z2, "h2": h2}

    def backward(self, cache: dict, d_out: np.ndarray) -> dict[str, np.ndarray]:
        """Chain rule from d(loss)/d(output) back to every weight."""
        p = self.params
        grads = {"W3": cache["h2"].T @ d_out, "b3": d_out.sum(axis=0)}
        d_z2 = (d_out @ p["W3"].T) * (cache["z2"] > 0)  # ReLU passes gradient only where it was active
        grads["W2"], grads["b2"] = cache["h1"].T @ d_z2, d_z2.sum(axis=0)
        d_z1 = (d_z2 @ p["W2"].T) * (cache["z1"] > 0)
        grads["W1"], grads["b1"] = cache["inputs"].T @ d_z1, d_z1.sum(axis=0)
        return grads


def _squared_error(net: DenoiserMLP, inputs: np.ndarray, target: np.ndarray) -> tuple[float, dict, np.ndarray]:
    """Mean over the batch of ‖guess − target‖², plus the gradient with respect to the guess."""
    out, cache = net.forward(inputs)
    diff = out - target
    loss = float(np.mean(np.sum(diff**2, axis=1)))
    return loss, cache, 2.0 * diff / len(inputs)


def denoiser_gradient_check(seed: int = 0, eps: float = 1e-6) -> float:
    """Largest relative gap between the hand-written gradients and finite differences.

    Nudge every weight up and down by eps, measure the loss change, and compare
    with what `DenoiserMLP.backward` claims. Agreement to ~1e-8 means the
    backward pass is right.
    """
    rng = np.random.default_rng(seed)
    net = DenoiserMLP(n_in=5, hidden=6, seed=seed)
    net.params["W3"] *= 10  # the tiny default last layer would make every gradient near zero
    inputs, target = rng.standard_normal((7, 5)), rng.standard_normal((7, 2))
    _, cache, d_out = _squared_error(net, inputs, target)
    analytic = net.backward(cache, d_out)
    worst = 0.0
    for name, P in net.params.items():
        for idx in np.ndindex(P.shape):
            old = P[idx]
            P[idx] = old + eps
            up = _squared_error(net, inputs, target)[0]
            P[idx] = old - eps
            down = _squared_error(net, inputs, target)[0]
            P[idx] = old
            numeric = (up - down) / (2 * eps)
            gap = abs(analytic[name][idx] - numeric) / max(1e-8, abs(analytic[name][idx]) + abs(numeric))
            worst = max(worst, gap)
    return worst


def _fit(net: DenoiserMLP, make_batch, steps: int, lr: float, seed: int) -> list[float]:
    """Adam with a cosine-decaying learning rate; `make_batch(rng)` returns (inputs, target)."""
    rng = np.random.default_rng(seed)
    opts = {k: Adam(lr) for k in net.params}  # one Adam per weight array, as in primer.ml.optimizers
    history = []
    for step in range(steps):
        for opt in opts.values():
            # Big steps early, small ones late, so the guesses settle instead of jittering.
            opt.lr = lr * 0.5 * (1 + np.cos(np.pi * step / steps))
        inputs, target = make_batch(rng)
        loss, cache, d_out = _squared_error(net, inputs, target)
        grads = net.backward(cache, d_out)
        for k in net.params:
            net.params[k] = opts[k].step(net.params[k], grads[k])
        history.append(loss)
    return history


@dataclass
class NoisePredictor:
    """ε_θ(x_t, t, label): a trained network that guesses the noise hidden in x_t."""

    net: DenoiserMLP
    schedule: NoiseSchedule
    n_classes: int = 0
    history: list[float] = field(default_factory=list)

    def __call__(self, x: np.ndarray, t: int, label=None) -> np.ndarray:
        """Guess ε at integer step t (1..T). label=None asks the unconditional question."""
        label = self.n_classes if label is None else label  # the "no label" slot
        return self.net.forward(network_inputs(x, t / self.schedule.T, label, self.n_classes))[0]


def train_noise_predictor(
    data: np.ndarray,
    schedule: NoiseSchedule,
    labels: np.ndarray | None = None,
    n_classes: int = 0,
    p_uncond: float = 0.2,
    steps: int = 1500,
    batch: int = 256,
    hidden: int = 64,
    lr: float = 1e-2,
    seed: int = 0,
) -> NoisePredictor:
    """Teach a network to guess the noise: the whole of diffusion training.

    Every batch: pick clean points, pick a random step t for each, pick fresh
    noise ε, jump to x_t with `add_noise`, and score the guess by ‖ε − ε̂‖².
    With labels, each label is hidden (replaced by "no label") with
    probability p_uncond, so the same network also learns the unlabelled guess.
    """
    net = DenoiserMLP(2 + 8 + (n_classes + 1 if n_classes else 0), hidden, seed=seed)

    def make_batch(rng):
        idx = rng.integers(0, len(data), batch)
        x0 = data[idx]
        t = rng.integers(1, schedule.T + 1, batch)  # every step equally often
        eps = rng.standard_normal(x0.shape)
        x_t = add_noise(x0, schedule.alpha_bars[t], eps)
        lab = None
        if n_classes:
            lab = labels[idx].copy()
            lab[rng.uniform(size=batch) < p_uncond] = n_classes  # sometimes hide the label
        return network_inputs(x_t, t / schedule.T, lab, n_classes), eps

    history = _fit(net, make_batch, steps, lr, seed)
    return NoisePredictor(net, schedule, n_classes, history)


def noise_to_score(eps_hat, alpha_bar):
    """Score ≈ −ε̂ / √(1 − ᾱ_t): the noise guess, flipped and rescaled, points toward the data."""
    return -np.asarray(eps_hat) / np.sqrt(1.0 - alpha_bar)


# ---------------------------------------------------------------------------
# 4. Sampling: from pure noise, step backwards
# ---------------------------------------------------------------------------


def ddpm_step(x_t, eps_hat, beta: float, alpha_bar: float, z):
    """One DDPM step back: x_{t−1} = (x_t − β_t/√(1 − ᾱ_t) · ε̂) / √α_t + √β_t · z.

    Remove the scaled noise guess, undo the shrink, then add a little fresh
    noise z (the last step uses z = 0).
    """
    alpha = 1.0 - beta
    mean = (np.asarray(x_t) - beta / np.sqrt(1.0 - alpha_bar) * np.asarray(eps_hat)) / np.sqrt(alpha)
    return mean + np.sqrt(beta) * np.asarray(z)


def ddpm_trajectory(model: NoisePredictor, n: int = 600, seed: int = 1, keep=(0,)) -> dict[int, np.ndarray]:
    """Run all T DDPM steps from pure noise; return the points at each step listed in `keep`."""
    rng = np.random.default_rng(seed)
    s = model.schedule
    x = rng.standard_normal((n, 2))  # x_T: pure noise
    snapshots = {s.T: x.copy()} if s.T in keep else {}
    for t in range(s.T, 0, -1):
        z = rng.standard_normal(x.shape) if t > 1 else np.zeros_like(x)
        x = ddpm_step(x, model(x, t), s.betas[t], s.alpha_bars[t], z)
        if t - 1 in keep:
            snapshots[t - 1] = x.copy()
    return snapshots


def ddpm_sample(model: NoisePredictor, n: int = 600, seed: int = 1) -> np.ndarray:
    """T small stochastic steps from noise to data. Returns (n, 2) samples."""
    return ddpm_trajectory(model, n, seed, keep=(0,))[0]


def ddim_step(x_t, eps_hat, alpha_bar_t: float, alpha_bar_s: float):
    """One deterministic DDIM jump from step t to an earlier step s.

    Predict the clean point x̂_0 = (x_t − √(1 − ᾱ_t) · ε̂) / √ᾱ_t, then re-noise
    it to level s with the same ε̂: x_s = √ᾱ_s · x̂_0 + √(1 − ᾱ_s) · ε̂.
    """
    x0_hat = (np.asarray(x_t) - np.sqrt(1.0 - alpha_bar_t) * np.asarray(eps_hat)) / np.sqrt(alpha_bar_t)
    return np.sqrt(alpha_bar_s) * x0_hat + np.sqrt(1.0 - alpha_bar_s) * np.asarray(eps_hat)


def guided_noise(eps_uncond, eps_cond, w: float):
    """Classifier-free guidance: ε̃ = ε_∅ + w · (ε_c − ε_∅).

    w = 0 ignores the label, w = 1 is the plain labelled guess, w > 1 steps
    past it, away from the unlabelled guess.
    """
    return np.asarray(eps_uncond) + w * (np.asarray(eps_cond) - np.asarray(eps_uncond))


def ddim_path(model: NoisePredictor, n: int = 600, steps: int = 20, seed: int = 1, label=None, guidance: float = 1.0) -> list[np.ndarray]:
    """Deterministic DDIM from pure noise in `steps` jumps; returns the points after every jump.

    With a label, each guess is the guided mix of the labelled and unlabelled
    guesses (two network calls per jump, as in real systems).
    """
    s = model.schedule
    x = np.random.default_rng(seed).standard_normal((n, 2))
    times = np.linspace(s.T, 0, steps + 1).round().astype(int)  # e.g. 100, 80, 60, 40, 20, 0
    path = [x]
    for t, t_next in zip(times[:-1], times[1:]):
        eps_hat = model(x, t) if label is None else guided_noise(model(x, t), model(x, t, label), guidance)
        x = ddim_step(x, eps_hat, s.alpha_bars[t], s.alpha_bars[t_next])  # ᾱ_0 = 1 makes the last jump land on x̂_0
        path.append(x)
    return path


def ddim_sample(model: NoisePredictor, n: int = 600, steps: int = 20, seed: int = 1, label=None, guidance: float = 1.0) -> np.ndarray:
    """`steps` deterministic jumps from noise to data. Returns (n, 2) samples."""
    return ddim_path(model, n, steps, seed, label, guidance)[-1]


# ---------------------------------------------------------------------------
# 5. Flow matching: straight lines from noise to data
# ---------------------------------------------------------------------------


def flow_point(x0, x1, t):
    """The point a fraction t of the way along the straight line: x_t = (1 − t) · x_0 + t · x_1.

    Here x_0 is the noise and x_1 is the data (the flow-matching convention).
    """
    return (1.0 - np.asarray(t)) * np.asarray(x0) + np.asarray(t) * np.asarray(x1)


def flow_target(x0, x1):
    """The velocity along that line, x_1 − x_0: the same at every t, because the line is straight."""
    return np.asarray(x1) - np.asarray(x0)


def euler_step(x, v, h: float):
    """One Euler step: move for time h at velocity v, x ← x + h · v."""
    return np.asarray(x) + h * np.asarray(v)


@dataclass
class VelocityPredictor:
    """v_θ(x_t, t): a trained network that guesses which way, and how fast, x_t should move."""

    net: DenoiserMLP
    history: list[float] = field(default_factory=list)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        return self.net.forward(network_inputs(x, t))[0]


def train_velocity_predictor(data: np.ndarray, steps: int = 1500, batch: int = 256, hidden: int = 64, lr: float = 1e-2, seed: int = 0) -> VelocityPredictor:
    """Flow matching training: pair each data point with fresh noise, pick t, learn x_1 − x_0 at x_t."""
    net = DenoiserMLP(2 + 8, hidden, seed=seed)

    def make_batch(rng):
        x1 = data[rng.integers(0, len(data), batch)]
        x0 = rng.standard_normal(x1.shape)
        t = rng.uniform(0.0, 1.0, batch)
        x_t = flow_point(x0, x1, t[:, None])
        return network_inputs(x_t, t), flow_target(x0, x1)

    return VelocityPredictor(net, _fit(net, make_batch, steps, lr, seed))


def euler_path(model: VelocityPredictor, n: int = 600, steps: int = 10, seed: int = 1) -> list[np.ndarray]:
    """Integrate the learned velocity from t = 0 (noise) to t = 1 (data); return every stop."""
    x = np.random.default_rng(seed).standard_normal((n, 2))
    h = 1.0 / steps
    path = [x]
    for k in range(steps):
        x = euler_step(x, model(x, k * h), h)
        path.append(x)
    return path


def euler_sample(model: VelocityPredictor, n: int = 600, steps: int = 10, seed: int = 1) -> np.ndarray:
    """`steps` Euler steps along the learned flow. Returns (n, 2) samples."""
    return euler_path(model, n, steps, seed)[-1]


# ---------------------------------------------------------------------------
# 6. Measuring samples, and the trained toy models
# ---------------------------------------------------------------------------


def two_way_distance(samples: np.ndarray, data: np.ndarray) -> float:
    """How far samples sit from the data, checked in both directions, then averaged.

    One way: for each sample, the distance to the nearest real point (are the
    samples realistic?). The other: for each real point, the distance to the
    nearest sample (is every part of the data covered?). Piling every sample
    onto one blob passes the first check and fails the second.
    """
    d = np.sqrt(((samples[:, None, :] - data[None, :, :]) ** 2).sum(axis=-1))  # (n_samples, n_data)
    return float(0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean()))


@functools.lru_cache(maxsize=1)
def trained_models(seed: int = 0) -> dict:
    """The lesson's three trained networks on the four blobs (about a second to train).

    Keys: data, labels, schedule, noise (unconditional noise predictor),
    flow (velocity predictor), conditional (noise predictor that also takes
    a corner label).
    """
    data, labels = make_blobs(seed=seed)
    schedule = linear_schedule()
    return {
        "data": data,
        "labels": labels,
        "schedule": schedule,
        "noise": train_noise_predictor(data, schedule, seed=seed),
        "flow": train_velocity_predictor(data, seed=seed),
        "conditional": train_noise_predictor(data, schedule, labels, n_classes=len(CORNERS), seed=seed),
    }


def step_sweep(step_counts=(1, 2, 3, 5, 10, 20, 50), models: dict | None = None) -> list[dict]:
    """Sample quality (two-way distance, lower is better) against the number of network calls."""
    m = models or trained_models()
    return [
        dict(
            steps=k,
            diffusion=two_way_distance(ddim_sample(m["noise"], steps=k), m["data"]),
            flow=two_way_distance(euler_sample(m["flow"], steps=k), m["data"]),
        )
        for k in step_counts
    ]


def guidance_sweep(weights=(0.0, 1.0, 3.0), label: int = SOUTH_WEST, n: int = 400, steps: int = 25, models: dict | None = None) -> list[dict]:
    """For each guidance weight: the share of samples that land in the requested blob, and their spread.

    A sample "lands in" the blob whose centre is nearest. Spread is the
    samples' standard deviation, averaged over the two axes.
    """
    m = models or trained_models()
    centres = np.array([m["data"][m["labels"] == c].mean(axis=0) for c in range(len(CORNERS))])
    rows = []
    for w in weights:
        s = ddim_sample(m["conditional"], n=n, steps=steps, label=label, guidance=w)
        nearest = np.argmin(((s[:, None, :] - centres[None]) ** 2).sum(axis=-1), axis=1)
        rows.append(dict(weight=w, on_target=float(np.mean(nearest == label)), spread=float(s.std(axis=0).mean())))
    return rows


# ---------------------------------------------------------------------------
# 7. Scaling up: latents and patches
# ---------------------------------------------------------------------------


def latent_shrink(height: int, width: int, channels: int = 3, downsample: int = 8, latent_channels: int = 4) -> dict:
    """How many numbers the denoiser must handle in pixel space versus an autoencoder's latent space.

    Stable Diffusion's autoencoder shrinks each side by 8 and keeps 4 channels,
    so a 512 × 512 colour image becomes a 64 × 64 × 4 latent.
    """
    pixels = height * width * channels
    latents = (height // downsample) * (width // downsample) * latent_channels
    return dict(pixels=pixels, latents=latents, factor=pixels / latents)


def patch_tokens(latent_height: int, latent_width: int, patch: int = 2, frames: int = 1) -> int:
    """Tokens a diffusion transformer reads: one per patch × patch square, per frame."""
    return frames * (latent_height // patch) * (latent_width // patch)


# ---------------------------------------------------------------------------
# 8. Figures (rendered into the HTML docs by `make figures`)
# ---------------------------------------------------------------------------


def figures() -> dict:
    """Plot this lesson's data. matplotlib is imported here, and only here,
    so the lesson itself needs nothing beyond NumPy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CORNER_COLOURS = ["#2563eb", "#dc2626", "#059669", "#d97706"]
    DIFFUSION, FLOW, MUTED = "#7c3aed", "#0891b2", "#9ca3af"
    m = trained_models()
    data, labels, schedule = m["data"], m["labels"], m["schedule"]
    colours = np.array(CORNER_COLOURS)[labels]
    figs = {}

    def square(ax, lim=3.2):
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        ax.add_patch(plt.Rectangle((-lim, -lim), 2 * lim, 2 * lim, fill=False, color=MUTED, lw=0.6, clip_on=False))

    # --- 1. The blobs dissolving into noise --------------------------------
    steps_shown = (0, 10, 30, 50, 100)
    eps = np.random.default_rng(3).standard_normal(data.shape)
    fig, axes = plt.subplots(1, len(steps_shown), figsize=(11, 2.6))
    for ax, t in zip(axes, steps_shown):
        ab = schedule.alpha_bars[t]
        ax.scatter(*add_noise(data, ab, eps).T, s=3, c=colours, alpha=0.7)
        square(ax)
        ax.set_title(f"t = {t}\nsignal √ᾱ = {np.sqrt(ab):.2f}", fontsize=9)
    fig.suptitle("The forward process: four blobs dissolve into plain noise", y=1.04)
    figs["noising"] = fig

    # --- 2. The schedule: how signal and noise trade places ---------------
    t = np.arange(schedule.T + 1)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(t, np.sqrt(schedule.alpha_bars), color=CORNER_COLOURS[0], label="signal kept  √ᾱ_t")
    ax.plot(t, np.sqrt(1 - schedule.alpha_bars), color=CORNER_COLOURS[1], label="noise mixed in  √(1 − ᾱ_t)")
    ax.axvline(37, color=MUTED, ls="--")
    ax.text(39, 0.3, "t = 37: half\nand half", color="#4b5563")
    ax.plot([30, 30], [0.6, 0.8], "o", color="#4b5563", ms=4)
    ax.text(8, 0.68, "t = 30: 0.8 and 0.6", color="#4b5563", fontsize=8)
    ax.set_xlabel("step t")
    ax.set_ylabel("share")
    ax.set_title("The noise schedule (T = 100, β from 0.0001 to 0.1)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    figs["schedule"] = fig

    # --- 3. Training the noise guesser ------------------------------------
    hist = np.array(m["noise"].history)
    smooth = np.convolve(hist, np.ones(25) / 25, mode="valid")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(hist, color=MUTED, lw=0.6, label="each batch")
    ax.plot(np.arange(len(smooth)) + 12, smooth, color=DIFFUSION, label="average of 25 batches")
    ax.axhline(2.0, color=CORNER_COLOURS[1], ls="--")
    ax.text(60, 2.07, "guessing ε = 0 scores 2", color=CORNER_COLOURS[1])
    ax.set_ylim(0, 2.6)
    ax.set_xlabel("training step")
    ax.set_ylabel("‖ε − ε̂‖², batch average")
    ax.set_title("Learning to guess the noise")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.7))
    figs["training"] = fig

    # --- 4. Sampling: noise condensing into blobs --------------------------
    keep = (100, 40, 20, 10, 0)
    snaps = ddpm_trajectory(m["noise"], n=600, seed=1, keep=keep)
    fig, axes = plt.subplots(1, len(keep), figsize=(11, 2.6))
    for ax, t in zip(axes, keep):
        ax.scatter(*data.T, s=2, c=MUTED, alpha=0.25)
        ax.scatter(*snaps[t].T, s=3, c=DIFFUSION, alpha=0.8)
        square(ax)
        ax.set_title(f"t = {t}", fontsize=9)
    fig.suptitle("Sampling: 100 small denoising steps turn noise (purple) into the blobs (grey)", y=1.02)
    figs["denoising"] = fig

    # --- 5. Quality against the number of steps ---------------------------
    rows = step_sweep()
    ks = [r["steps"] for r in rows]
    fresh = two_way_distance(make_blobs(seed=99)[0], data)
    noise_level = two_way_distance(np.random.default_rng(1).standard_normal((600, 2)), data)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(ks, [r["diffusion"] for r in rows], "o-", color=DIFFUSION, label="diffusion (DDIM jumps)")
    ax.plot(ks, [r["flow"] for r in rows], "o-", color=FLOW, label="flow matching (Euler steps)")
    ax.axhline(fresh, color=CORNER_COLOURS[2], ls="--")
    ax.set_ylim(fresh * 0.8, None)
    ax.text(1.05, fresh * 0.87, "a fresh draw of the real blobs", color=CORNER_COLOURS[2])
    ax.axhline(noise_level, color=MUTED, ls="--")
    ax.text(22, noise_level * 1.05, "plain noise", color="#4b5563")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("network calls per sample")
    ax.set_ylabel("two-way distance to the data")
    ax.set_title("Straighter paths need fewer steps")
    ax.legend(frameon=False)
    figs["steps"] = fig

    # --- 6. Flow matching: straight training lines, non-crossing learned paths
    n_paths = 16
    rng = np.random.default_rng(7)
    starts = rng.standard_normal((n_paths, 2))
    ends = data[rng.integers(0, len(data), n_paths)]
    learned = np.stack(euler_path(m["flow"], n=n_paths, steps=40, seed=7))
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].scatter(*data.T, s=2, c=MUTED, alpha=0.25)
    for a, b in zip(starts, ends):
        axes[0].plot([a[0], b[0]], [a[1], b[1]], color=FLOW, lw=1)
    axes[0].scatter(*starts.T, s=18, facecolors="white", edgecolors=FLOW, zorder=3, label="noise x₀")
    axes[0].scatter(*ends.T, s=18, color=FLOW, zorder=3, label="data x₁")
    axes[0].set_title("Training: random pairs, straight lines\n(they cross)", fontsize=10)
    axes[1].scatter(*data.T, s=2, c=MUTED, alpha=0.25)
    for i in range(n_paths):
        axes[1].plot(learned[:, i, 0], learned[:, i, 1], color=FLOW, lw=1)
    axes[1].scatter(*learned[0].T, s=18, facecolors="white", edgecolors=FLOW, zorder=3)
    axes[1].scatter(*learned[-1].T, s=18, color=FLOW, zorder=3)
    axes[1].set_title("Sampling: the learned flow\n(paths never cross, so they bend)", fontsize=10)
    for ax in axes:
        square(ax, 2.8)
    axes[0].legend(frameon=False, loc="lower left", fontsize=8)
    figs["flow_paths"] = fig

    # --- 7. Guidance: on-label but less varied -----------------------------
    weights = (0.0, 1.0, 3.0)
    fig, axes = plt.subplots(1, len(weights), figsize=(9, 3.2))
    for ax, w in zip(axes, weights):
        s = ddim_sample(m["conditional"], n=400, steps=25, label=SOUTH_WEST, guidance=w)
        row = guidance_sweep((w,), models=m)[0]
        ax.scatter(*data.T, s=2, c=MUTED, alpha=0.25)
        ax.scatter(*s.T, s=3, color=CORNER_COLOURS[SOUTH_WEST], alpha=0.8)
        square(ax, 2.2)
        ax.set_title(f"w = {w:g}: {row['on_target']:.0%} south-west\nspread {row['spread']:.2f}", fontsize=9)
    fig.suptitle('Classifier-free guidance, asking for "south-west"', y=1.04)
    figs["guidance"] = fig

    return figs


# ---------------------------------------------------------------------------
# 9. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. The forward process: adding noise a little at a time")
    s = linear_schedule()
    say(
        """
        A pixel with clean value 2.0, mixed with noise 0.5 at step 30 (where
        the schedule has left ᾱ = 0.64 of the signal): √0.64·2 + √0.36·0.5.
        """
    )
    say(f"x_30 = {add_noise(2.0, 0.64, 0.5):.2f}. The schedule, step by step:")
    table(
        ["step t", "ᾱ_t", "signal √ᾱ_t", "noise √(1 − ᾱ_t)"],
        [(t, s.alpha_bars[t], np.sqrt(s.alpha_bars[t]), np.sqrt(1 - s.alpha_bars[t])) for t in (0, 10, 30, 37, 50, 100)],
        floatfmt=".3f",
    )
    takeaway("After 100 steps only 7% of the signal is left: any data ends as the same cloud of plain noise.")

    banner("2. Learning to denoise: guess the noise, graded by squared error")
    m = trained_models()
    hist = m["noise"].history
    say(
        f"""
        Four blobs of {len(m["data"])} dots. A 2-layer network sees a noised dot and
        its step, and guesses the noise. Guessing zero would score 2.
        First batch: {hist[0]:.2f}. Average of the last 100 batches:
        {np.mean(hist[-100:]):.2f}.
        """
    )
    say(f"Its hand-written backward pass agrees with finite differences to {denoiser_gradient_check():.1e}.")
    say(f"The noise guess is the score in disguise: a guess of 0.9 at ᾱ = 0.64 means score {noise_to_score(0.9, 0.64):.2f}.")
    takeaway("Diffusion training is plain regression: add known noise, learn to guess it back.")

    banner("3. Sampling: from pure noise, step backwards")
    fresh = two_way_distance(make_blobs(seed=99)[0], m["data"])
    noise = np.random.default_rng(1).standard_normal((600, 2))
    say(f"One DDPM step, worked: x_t = 1.0, guess 0.5, β = 0.19, ᾱ = 0.75 gives {ddpm_step(1.0, 0.5, 0.19, 0.75, 0.0):.2f}.")
    say(f"One DDIM jump, worked: 1.9 at ᾱ = 0.64 to ᾱ = 0.96 gives {ddim_step(1.9, 0.5, 0.64, 0.96):.4f}.")
    table(
        ["samples", "network calls", "two-way distance to the data"],
        [
            ("a fresh draw of the real blobs", "-", fresh),
            ("DDPM, small stochastic steps", 100, two_way_distance(ddpm_sample(m["noise"]), m["data"])),
            ("DDIM, deterministic jumps", 20, two_way_distance(ddim_sample(m["noise"], steps=20), m["data"])),
            ("plain noise, never denoised", 0, two_way_distance(noise, m["data"])),
        ],
        floatfmt=".3f",
    )
    takeaway("Twenty confident jumps land as close to the data as a hundred small steps.")

    banner("4. Flow matching: learn the velocity along straight lines")
    say(
        f"""
        Noise at -1, data at 2: a quarter of the way along, the point is at
        {flow_point(-1.0, 2.0, 0.25):.2f} and the velocity to learn is
        {flow_target(-1.0, 2.0):.0f}. One Euler step of 0.75 lands on
        {euler_step(-0.25, 3.0, 0.75):.1f}.
        """
    )
    table(["network calls", "diffusion (DDIM)", "flow matching (Euler)"], [(r["steps"], r["diffusion"], r["flow"]) for r in step_sweep(models=m)], floatfmt=".3f")
    one_step = euler_sample(m["flow"], steps=1)
    say(
        f"""
        Lower is better; {fresh:.3f} is as good as it gets. The flow gets close
        in about 5 calls, diffusion in about 10. One single step lands every
        sample near the data's average (mean distance from the centre
        {np.linalg.norm(one_step, axis=1).mean():.2f}), the empty middle
        between the blobs.
        """
    )
    takeaway("Straighter paths need fewer steps; retraining to straighten them further is rectified flow.")

    banner("5. Guidance: ask for 'south-west', and push past the labelled guess")
    say(f"Worked: unlabelled guess 0.2, labelled 0.5, weight 3 gives {guided_noise(0.2, 0.5, 3.0):.1f}.")
    table(
        ["weight w", "share landing south-west", "spread"],
        [(r["weight"], f"{r['on_target']:.0%}", r["spread"]) for r in guidance_sweep((0.0, 1.0, 2.0, 3.0, 5.0), models=m)],
        floatfmt=".2f",
    )
    takeaway("Guidance buys obedience with variety: more on-label, less varied.")

    banner("6. Scaling up: latents, patches and frames")
    shrink = latent_shrink(512, 512)
    table(
        ["what the denoiser handles", "numbers or tokens"],
        [
            ("512 x 512 colour image, in pixels", f"{shrink['pixels']:,}"),
            ("its 64 x 64 x 4 latent", f"{shrink['latents']:,}"),
            ("latent cut into 2 x 2 patches (tokens)", f"{patch_tokens(64, 64):,}"),
            ("16 video frames of those patches (tokens)", f"{patch_tokens(64, 64, frames=16):,}"),
        ],
    )
    say(
        f"""
        The latent is {shrink['factor']:.0f} times smaller than the image, and every one
        of the dozens of sampling steps gets that much cheaper. A transformer
        reads the patches as tokens and attends to a text encoder's tokens
        for the prompt.
        """
    )
    takeaway("Real image, video and audio generators are this lesson's loop, run on a latent by a transformer.")


if __name__ == "__main__":
    demo()
