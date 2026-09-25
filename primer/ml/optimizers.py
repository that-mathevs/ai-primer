r"""
# Optimizers: how weights take their steps downhill

Run: `python -m primer.ml.optimizers`

## The idea: walking downhill in fog

You're on a hillside in thick fog and want to reach the lowest point in the
valley. You can't see the valley; you can only feel the slope under your
feet. So you feel which way is downhill, take a step that way, and repeat.
That is **gradient descent**. The hillside is the loss (how wrong the model
is, for every possible setting of its weights), your position is the
current weights, and the slope under your feet is the **gradient**: the list
of slopes of the loss, one per weight. (See `primer.ml.neural_net` for how
backprop measures it, and `primer.notation` for the symbols.) An
**optimizer** is your rule for turning "the slope here" into "the step I
take".

Worked example on the simplest possible valley, the bowl f(w) = w², whose
slope at w is 2w. Start at w = 1 with step size 0.1:

| step | w | slope 2w | step taken 0.1 × slope | new w |
|---|---|---|---|---|
| 0 | 1.0 | 2.0 | 0.2 | 0.8 |
| 1 | 0.8 | 1.6 | 0.16 | 0.64 |

Each step keeps 80% of w, sliding smoothly toward the bottom at 0.

```mermaid
flowchart LR
  W[Current weights] --> G[Feel the slope<br/>compute gradient]
  G --> R{Optimizer rule}
  R --> S[Step]
  S --> W2[New weights]
  W2 -->|repeat| G
  R -.uses.-> H[Its own memory:<br/>velocity, averages]
```

**Reading it:** every optimizer runs this loop; they differ only in the
diamond. Plain gradient descent's rule looks only at the current slope.
Momentum and Adam also keep a small memory of past slopes (the dotted box)
and use it to choose a better step.

$$
w_{t+1} = w_t - \eta \, \nabla \mathcal{L}(w_t)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $w_t$ | the weights at step $t$ | $w_0 = 1.0$ |
| $t$ | the step counter | 0, 1, 2, … |
| $\eta$ | "eta", the learning rate (step size) | 0.1 |
| $\mathcal{L}$ | the loss | $w^2$ |
| $\nabla \mathcal{L}(w_t)$ | "nabla L", the gradient: the slope of the loss at the current weights, one number per weight | $2 \times 1.0 = 2.0$ |

**In words:** "the next weights are the current weights minus the learning
rate times the slope of the loss where we stand."

**With the numbers:** $w_1 = 1.0 - 0.1 \times 2.0 = 0.8$;
$w_2 = 0.8 - 0.1 \times 1.6 = 0.64$.

**In Python:**

```python
w, eta = 1.0, 0.1
# ∇L for the bowl L = w²
def grad_L(w): return 2 * w
for t in range(2):
    # w_(t+1) = w_t - η ∇L(w_t)
    w = w - eta * grad_L(w)
    print(round(w, 2))  # → 0.8 0.64
```

"Stochastic" gradient descent (SGD) means the slope is estimated from a
small random batch of examples instead of the whole dataset: noisier, but
thousands of times cheaper per step. `descend_bowl` runs the table above;
`SGD` is the general version.

**Why it matters:** every model you've heard of was trained by a descendant
of this one line. The variants below exist because real loss landscapes are
not round bowls.

## The learning rate: how long a stride?

In the fog, stride length is everything. Tiny shuffling steps are safe but
you'll be walking all night. Giant leaps overshoot the valley floor and land
you higher up the opposite slope; keep leaping and you climb out of the
valley altogether.

Worked example on the bowl w², where each step multiplies w by (1 − 2η):

| learning rate η | multiplier 1 − 2η | w after 1, 2, 3 steps | what happens |
|---|---|---|---|
| 0.001 | 0.998 | 0.998, 0.996, 0.994 | stalls: 10 steps only reach 0.980 |
| 0.1 | 0.8 | 0.8, 0.64, 0.512 | smooth progress |
| 0.5 | 0 | 0, 0, 0 | lands on the bottom in one step |
| 1.1 | −1.2 | −1.2, 1.44, −1.728 | overshoots further each time: diverges |

![Over 30 steps, rate 0.001 barely lowers the loss, 0.1 falls steadily, 0.45 plunges fastest, and 1.1 climbs off the chart as every step overshoots](figures/primer.ml.optimizers.learning_rates.svg)

**Reading it:** each line is the loss (log scale) over 30 steps for one
learning rate. The flat line near the top is 0.001: technically improving,
practically stuck. 0.1 falls steadily. 0.45 falls fastest. 1.1 climbs off
the top of the chart: every step makes things worse. On a real model you
can't compute the perfect rate, so you look for the fastest one that doesn't
blow up.

$$
w_{t+1} = w_t - \eta \cdot 2 w_t = (1 - 2\eta)\, w_t
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $2w_t$ | the slope of $w^2$ at $w_t$ | 2.0 at $w = 1$ |
| $1 - 2\eta$ | the factor each step multiplies $w$ by | −1.2 when $\eta = 1.1$ |

**In words:** "on this bowl, one step multiplies the weight by one minus
twice the learning rate; if that factor's size is above 1, the weight grows
instead of shrinking."

**With the numbers:** $\eta = 1.1$: $1 - 2.2 = -1.2$, so
$1 \to -1.2 \to 1.44 \to -1.728$.

**In Python:**

```python
def three_steps(eta, w=1.0):
    out = []
    for t in range(3):
        # w - η·2w = (1 - 2η) w
        w = (1 - 2 * eta) * w
        out.append(round(w, 3))
    return out
three_steps(1.1)  # → [-1.2, 1.44, -1.728]
three_steps(0.1)  # → [0.8, 0.64, 0.512]
# η = 0.001 after 10 steps: barely moved
round(0.998 ** 10, 3)  # → 0.98
```

**In code:** `descend_bowl` is the same loop with the learning rate as an argument; call it with each rate in the table to reproduce every row.

**Why it matters:** the learning rate is the single most important
hyperparameter. Too high and training diverges or bounces (loss spikes,
NaNs); too low and it takes forever or settles somewhere poor. The steepest
direction of the landscape sets the ceiling: on a bowl with slope 2w, any η
above 1 diverges.

## Momentum: a heavy ball instead of a cautious hiker

Now imagine a long, narrow valley: steep walls on both sides, a gentle slope
along the floor. A cautious hiker who only reads the local slope zig-zags
from wall to wall and barely moves along the floor. A heavy ball rolling
down the same valley behaves differently: its sideways bouncing cancels out,
while the gentle downhill pull along the floor keeps adding up, so it builds
speed exactly where you want it. That accumulated speed is **momentum**.

Worked example on the bowl w², learning rate 0.1, momentum β = 0.9:

| step | slope g = 2w | velocity v = 0.9·v + g | new w = w − 0.1·v |
|---|---|---|---|
| 1 | 2.0 | 2.0 | 1.0 − 0.2 = 0.8 |
| 2 | 1.6 | 0.9 × 2.0 + 1.6 = 3.4 | 0.8 − 0.34 = 0.46 |

After two steps plain descent is at 0.64; momentum is already at 0.46.

```mermaid
flowchart LR
  G[Slope now g_t] --> V["Velocity v_t = β·v_(t−1) + g_t<br/>(remember 90% of the old speed)"]
  VO[Old velocity v_(t−1)] --> V
  V --> S["Step: w − η·v_t"]
  S --> VO
```

**Reading it:** the velocity box mixes the new slope with 90% of the old
velocity, and that velocity (not the raw slope) sets the step. Follow the
loop back: next time, this velocity becomes the "old velocity". Slopes that
keep pointing the same way pile up; slopes that flip sign every step (the
walls) mostly cancel.

![After 100 steps from (-8, 1), plain SGD has only crept to x = -1.8 along the valley floor, while momentum and Adam reach the minimum after some overshoot](figures/primer.ml.optimizers.trajectories.svg)

**Reading it:** the ellipses are contour lines of the valley ½(x² + 100y²),
the start is at the left and the minimum is the star at the centre. Plain
SGD (its learning rate capped by the steep walls) creeps along the floor and
after 100 steps is still far from the star. Momentum swings hard: across
the valley to y = −1.1 (further out than it started) and past the star to
x ≈ 1.9, before the swings die down and it settles 0.003 from the minimum.
Adam, which rescales each direction separately, takes a steadier line: it
dips to y ≈ −0.5, overshoots to x ≈ 0.6, and ends 0.02 away. Both beat
plain SGD by far; momentum's overshoot is the price of the speed it builds.

$$
v_t = \beta\, v_{t-1} + g_t, \qquad w_{t+1} = w_t - \eta\, v_t
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $g_t$ | the gradient (slope) at step $t$ | 2.0, then 1.6 |
| $v_t$ | the velocity: a running, fading sum of past gradients | 2.0, then 3.4 |
| $\beta$ | "beta", how much old velocity is kept each step (0 to 1) | 0.9 |
| $\eta$ | learning rate | 0.1 |

**In words:** "the velocity is 90% of the previous velocity plus the new
slope, and the weights move by the learning rate times the velocity."

**With the numbers:** $v_2 = 0.9 \times 2.0 + 1.6 = 3.4$;
$w_2 = 0.8 - 0.1 \times 3.4 = 0.46$.

**In Python:**

```python
w, v, beta, eta = 1.0, 0.0, 0.9, 0.1
for t in range(2):
    # the slope of w² here
    g = 2 * w
    # v_t = β v_(t-1) + g_t
    v = beta * v + g
    # w_(t+1) = w_t - η v_t
    w = w - eta * v
    print(round(v, 2), round(w, 2))  # → 2.0 0.8 3.4 0.46
```

**In code:** `momentum_on_bowl` runs the two-row table; `SGD` with a nonzero momentum keeps its velocity between calls to `SGD.step`. `narrow_valley` is the valley in the figure (`rosenbrock` is a harder, banana-shaped one), and `run` walks any optimizer across a landscape and records its path.

**Why it matters:** real loss surfaces are full of narrow valleys. On the
valley above, 100 steps of momentum reach a loss over 10,000× lower than 100
steps of plain SGD at the same learning rate. Momentum is still the default
for training CNNs.

## Adam: a separate stride for every direction

Back in the narrow valley, what you'd really like is short steps across the
steep walls and long strides along the gentle floor. Adam does exactly that:
it keeps, for every single weight, a running average of the slope (the
direction, like momentum) and a running average of the *squared* slope (how
big that weight's slopes typically are), then divides the first by the
square root of the second. Every weight ends up taking steps of roughly the
same size, the learning rate, whatever the scale of its gradient.

Worked example: on its very first step, Adam moves a weight by exactly the
learning rate (0.01 here), no matter whether that weight's gradient is 1000,
1 or 0.001. (With bias correction, m̂ = g and v̂ = g², so the step is
0.01 × g / |g| = 0.01.)

```mermaid
flowchart LR
  G[Gradient g] --> M["m: average of g<br/>(direction)"]
  G --> V["v: average of g²<br/>(typical size)"]
  M --> MC["m̂ = m / (1 − β1^t)"]
  V --> VC["v̂ = v / (1 − β2^t)"]
  MC --> D["step = η · m̂ / (√v̂ + ε)"]
  VC --> D
  D --> W[w − step]
```

**Reading it:** the gradient feeds two running averages. The top path
remembers direction; the bottom path remembers magnitude. Both start at
zero, so for the first few steps they're too small, and the "hat" boxes
correct for that. The final box divides direction by typical size, which
turns every weight's step into roughly η in the right direction.

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t,\quad
v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2,\quad
\hat{m}_t = \frac{m_t}{1-\beta_1^t},\quad
\hat{v}_t = \frac{v_t}{1-\beta_2^t},\quad
w_{t+1} = w_t - \eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

**Symbols**

| Symbol | Meaning here | In the example (first step, g = 1000) |
|---|---|---|
| $g_t$ | this weight's gradient at step $t$ | 1000 |
| $m_t$ | running average of the gradient ("first moment") | $0.1 \times 1000 = 100$ |
| $v_t$ | running average of the squared gradient ("second moment") | $0.001 \times 10^6 = 1000$ |
| $\beta_1, \beta_2$ | how much of the old averages to keep | 0.9, 0.999 |
| $\beta_1^t$ | $\beta_1$ raised to the step number | $0.9^1 = 0.9$ |
| $\hat{m}_t, \hat{v}_t$ | "m-hat, v-hat": the bias-corrected averages | 1000, $10^6$ |
| $\sqrt{\cdot}$ | square root | $\sqrt{10^6} = 1000$ |
| $\epsilon$ | "epsilon", a tiny number to avoid dividing by zero | $10^{-8}$ |
| $\eta$ | learning rate | 0.01 |

**In words:** "keep a running average of the gradient and of its square,
correct both for starting at zero, then step by the learning rate times the
average gradient divided by its typical size."

**With the numbers:** $\hat{m}_1 = 100 / 0.1 = 1000$, $\hat{v}_1 = 1000 / 0.001 = 10^6$,
step $= 0.01 \times 1000 / (1000 + 10^{-8}) = 0.01$.

**In Python:**

```python
import math
beta_1, beta_2, eta, eps, t = 0.9, 0.999, 0.01, 1e-8, 1
def first_step(g):
    # m_1, starting from m_0 = 0
    m = beta_1 * 0 + (1 - beta_1) * g
    # v_1, starting from v_0 = 0
    v = beta_2 * 0 + (1 - beta_2) * g ** 2
    # undo the pull toward zero
    m_hat = m / (1 - beta_1 ** t)
    v_hat = v / (1 - beta_2 ** t)
    return eta * m_hat / (math.sqrt(v_hat) + eps)
# m_1, v_1
round((1 - beta_1) * 1000, 6), round((1 - beta_2) * 1000 ** 2, 6)  # → (100.0, 1000.0)
# the same step every time
[round(first_step(g), 6) for g in (1000, 1, 0.001)]  # → [0.01, 0.01, 0.01]
```

**In code:** `Adam` keeps the two running averages and the step count for every weight and applies the five formulas in `Adam.step`; `adam_first_step` shows the first step is always the learning rate.

**Why it matters:** Adam is forgiving: one learning rate works across
weights whose gradients differ by orders of magnitude, which is the norm in
transformers (embeddings, attention, layer norms all behave differently).
On the valley above it reaches a loss below 10⁻¹⁰ in 300 steps while plain
SGD is still around 10⁻³.

## Weight decay, and why AdamW exists

Weight decay is a gentle leash that pulls every weight a little toward zero
each step, so the model prefers small, smooth weights over large, spiky ones
(a form of regularization; see `primer.ml.regularization`). The classic way
to add it was an **L2 penalty**: add λw to the gradient. With plain SGD that
is the same as shrinking the weight. With Adam it isn't, because Adam
divides the whole gradient, penalty included, by its typical size. The
leash's strength gets rescaled away.

Worked example: one step with zero loss-gradient, so only the decay acts.
w = 1, learning rate 0.1.

| method | decay λ = 0.1 | decay λ = 0.001 |
|---|---|---|
| AdamW (decoupled) | 1 − 0.1 × 0.1 = **0.99** | 1 − 0.1 × 0.001 = **0.9999** |
| Adam with L2 in the gradient | **0.9** | **0.9** (the same!) |

With L2 inside Adam, a 100× weaker penalty shrinks the weight exactly as
much: λ has stopped meaning what it says.

```mermaid
flowchart TB
  subgraph L2["Adam + L2 penalty"]
    g1[loss gradient] --> add["+ λ·w"] --> ad1[Adam rescaling<br/>÷ √v̂] --> s1[step]
  end
  subgraph AW["AdamW (decoupled)"]
    g2[loss gradient] --> ad2[Adam rescaling<br/>÷ √v̂] --> s2[step]
    w2[weights] --> dec["shrink: w − η·λ·w"] --> s2
  end
```

**Reading it:** on the left, the decay term joins the gradient *before*
Adam's rescaling, so it gets divided by √v̂ like everything else and its
strength is distorted. On the right (AdamW), the decay bypasses the
rescaling and shrinks the weights directly, so λ means exactly "shrink by
η·λ per step".

$$
\text{AdamW:}\quad w_{t+1} = w_t - \eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} - \eta\,\lambda\, w_t
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $\lambda$ | "lambda", the weight-decay strength | 0.1 |
| $\eta\,\lambda\,w_t$ | how much the leash pulls this step | $0.1 \times 0.1 \times 1 = 0.01$ |
| $\hat{m}_t / (\sqrt{\hat{v}_t}+\epsilon)$ | Adam's usual step direction | 0 here (no loss gradient) |

**In words:** "take Adam's normal step, then separately shrink every
weight by learning rate times decay times the weight."

**With the numbers:** $1 - 0 - 0.1 \times 0.1 \times 1 = 0.99$.

**In Python:**

```python
# no loss gradient: Adam's step is 0
w, eta, adam_step = 1.0, 0.1, 0.0
for lam in (0.1, 0.001):
    # AdamW: shrink by η λ w
    print(round(w - eta * adam_step - eta * lam * w, 4))  # → 0.99 0.9999
for lam in (0.1, 0.001):
    g = lam * w
    print(round(w - eta * g / abs(g), 4))  # → 0.9 0.9
```

**In code:** `Adam` implements both recipes: with decoupled decay it is AdamW, otherwise it adds the L2 penalty to the gradient. `one_decay_step` runs the table.

**Why it matters:** AdamW is the default optimizer for transformers. The
fix was a one-line change that made weight decay behave predictably and
improved generalization.

## Warmup and cosine decay: easing on and off the gas

Think of driving an unfamiliar car. You ease onto the accelerator at first,
because you don't yet know how it responds. You cruise at speed for most of
the trip. Near the destination you slow gradually and glide into the
parking spot. Transformers are trained the same way: the learning rate
**warms up** linearly from 0 to its peak, then **decays** along a cosine
curve toward a small floor.

Worked example with peak 0.001, 100 warmup steps, 1,000 steps total:

| step | phase | learning rate |
|---|---|---|
| 50 | halfway through warmup | 0.0005 |
| 100 | end of warmup | 0.001 (peak) |
| 550 | halfway through decay | 0.0005 |
| 1000 | end | the floor (e.g. 0.00001) |

![The learning rate ramps straight up from 0 to 0.001 over 100 steps, then falls along a half cosine, passing 0.0005 at step 550 and ending near 0.00001](figures/primer.ml.optimizers.schedule.svg)

**Reading it:** the horizontal axis is the training step and the vertical
axis the learning rate. The short straight ramp on the left is warmup. Then
the curve rolls over the top, falls fastest in the middle and flattens as it
approaches the floor, like half a cosine wave. The dots are the table above.

$$
\eta(t) =
\begin{cases}
\eta_{\max}\,\dfrac{t}{T_w} & t < T_w \\[6pt]
\eta_{\min} + (\eta_{\max}-\eta_{\min})\,\dfrac{1 + \cos\!\left(\pi\,\dfrac{t - T_w}{T - T_w}\right)}{2} & t \ge T_w
\end{cases}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $\eta(t)$ | the learning rate at step $t$ | 0.0005 at $t = 550$ |
| $\eta_{\max}$ | the peak learning rate | 0.001 |
| $\eta_{\min}$ | the floor | 0 (or 0.00001) |
| $T_w$ | number of warmup steps | 100 |
| $T$ | total steps | 1000 |
| $\frac{t - T_w}{T - T_w}$ | progress through the decay, from 0 to 1 | $450/900 = 0.5$ |
| $\cos$ | cosine: 1 at 0, 0 at $\pi/2$, −1 at $\pi$ | $\cos(\pi/2) = 0$ |
| $\pi$ | pi, ≈ 3.1416 (half a turn, in radians) | |

**In words:** "during warmup the rate climbs in a straight line to the peak;
after that it follows half a cosine wave from the peak down to the floor."

**With the numbers:** step 550: progress 0.5, $\cos(0.5\pi) = 0$, so
$0 + 0.001 \times (1 + 0)/2 = 0.0005$.

**In Python:**

```python
import math
eta_max, eta_min, T_w, T = 0.001, 0.0, 100, 1000
def eta(t):
    if t < T_w:
        # the straight ramp
        return eta_max * t / T_w
    # 0 to 1 through the decay
    progress = (t - T_w) / (T - T_w)
    return eta_min + (eta_max - eta_min) * (1 + math.cos(math.pi * progress)) / 2
[round(eta(t), 6) for t in (50, 100, 550, 1000)]  # → [0.0005, 0.001, 0.0005, 0.0]
```

**In code:** `warmup_cosine` returns the learning rate for any step: the straight ramp during warmup, then the half cosine down to the floor.

**Why it matters:** at the very start, Adam's averages are unreliable and
the weights are random, so a full-size step can wreck them; warmup avoids
early divergence. The slow finish lets the model settle into a good minimum
instead of bouncing around it.

## Gradient clipping: a speed limiter

Occasionally one bad batch produces an enormous gradient, a sudden cliff in
the fog. Taking a full step along it could throw the weights far from
anywhere useful. Clipping is a speed limiter: if the step would be longer
than a set limit, shorten it to the limit, keeping its direction.

Worked example: the gradient (3, 4) has length √(3² + 4²) = 5. With a limit
of 1, scale it by 1/5 to get (0.6, 0.8), length 1, same direction. A gradient
of (0.3, 0.4) has length 0.5, under the limit, so it's left alone.

```mermaid
flowchart LR
  G[All gradients] --> N["Global length<br/>‖g‖ = √(sum of every squared entry)"]
  N --> C{"‖g‖ > limit?"}
  C -->|no| K[Use as is]
  C -->|yes| S["Multiply every gradient<br/>by limit / ‖g‖"]
  S --> K2[Same direction,<br/>length = limit]
```

**Reading it:** first measure the length of *all* the gradients together,
as if they were one long list. If it's within the limit, nothing happens.
If not, every gradient is multiplied by the same factor, which shortens the
step without changing its direction.

$$
g \leftarrow g \cdot \min\left(1, \frac{c}{\lVert g \rVert}\right), \qquad
\lVert g \rVert = \sqrt{\textstyle\sum_i g_i^2}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $g$ | every gradient, treated as one long list | (3, 4) |
| $g_i$ | one entry of that list | 3, 4 |
| $\lVert g \rVert$ | the **norm** (length) of $g$: square every entry, add, take the square root | 5 |
| $c$ | the clipping limit | 1 |
| $\min(a, b)$ | the smaller of the two | $\min(1, 0.2) = 0.2$ |

**In words:** "if the gradient's length exceeds the limit, scale it down so
its length equals the limit; otherwise leave it alone."

**With the numbers:** $(3, 4) \times \min(1, 1/5) = (0.6, 0.8)$.

**In Python:**

```python
import math
def clip(g, c):
    # ‖g‖ = √(Σ g_i²)
    norm = math.sqrt(sum(g_i ** 2 for g_i in g))
    # shrink only if too long
    scale = min(1, c / norm)
    return [round(g_i * scale, 2) for g_i in g]
clip([3, 4], c=1)  # → [0.6, 0.8]
# length 0.5: under the limit, left alone
clip([0.3, 0.4], c=1)  # → [0.3, 0.4]
```

**In code:** `clip_by_global_norm` measures the length of all gradients together and scales every one by the same factor when it exceeds the limit.

**Why it matters:** large-model training runs almost always clip (a limit of
1.0 is common). It turns rare loss spikes from run-ending disasters into
harmless blips. It's measured globally, across all layers together, so the
update's direction is preserved (see `primer.ml.deep_nets` for exploding
gradients).

## In 20 seconds
- Gradient descent: step against the slope, scaled by the learning rate.
- The learning rate is the most important knob: too high diverges, too low
  stalls.
- Momentum accumulates a velocity so consistent directions speed up and
  zig-zags cancel.
- Adam gives every weight its own step size (average gradient ÷ its typical
  size); AdamW decouples weight decay from that rescaling and is the
  transformer default.
- Transformers use warmup then cosine decay, and clip gradients by global norm.

## Self-test questions

**What's the single most important hyperparameter, and what happens at each extreme?**
The learning rate. Too high: steps overshoot and training diverges or
oscillates (loss spikes, NaNs). Too low: progress is so slow that training
stalls or settles in a poor spot.

**Why does momentum help in a narrow valley?**
Gradients across the valley flip sign every step and cancel in the velocity,
while the small, consistent gradient along the valley accumulates, so the
optimizer speeds up in the useful direction.

**What does Adam's division by √v̂ achieve?**
It normalizes each weight's step by the typical size of its gradient, so
every weight moves roughly η per step regardless of gradient scale. One
learning rate then works for all parameters.

**What's the difference between Adam with L2 and AdamW?**
With L2, the decay term is added to the gradient and then rescaled by Adam,
so its effective strength varies per weight and λ loses its meaning. AdamW
applies decay directly to the weights, outside the rescaling.

**Why warm up the learning rate?**
At the start, weights are random and Adam's moment estimates are based on a
handful of steps, so full-size updates can be wildly wrong and destabilize
training. Ramping up gives the statistics time to settle.

**What does gradient clipping protect against, and why clip the global norm?**
Rare huge gradients (exploding gradients, bad batches) that would throw the
weights far off. Clipping the combined norm scales every tensor by the same
factor, which preserves the update's direction.

## The papers behind this lesson

- Kingma & Ba, *Adam: A Method for Stochastic Optimization* (2014): https://arxiv.org/abs/1412.6980
  Combined momentum with per-weight step sizes and bias correction into the optimizer most networks are trained with. [annotated companion](../../papers/adam.html)
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW, 2017): https://arxiv.org/abs/1711.05101
  Showed that L2 regularization and weight decay differ under Adam, and fixed it by decoupling the decay.
- Sutskever, Martens, Dahl & Hinton, *On the importance of initialization and momentum in deep learning* (ICML 2013): https://proceedings.mlr.press/v28/sutskever13.html
  Demonstrated that well-tuned momentum makes plain SGD competitive on hard deep-network problems.
- Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts* (2016): https://arxiv.org/abs/1608.03983
  Introduced cosine learning-rate annealing, now the standard decay shape.
- Pascanu, Mikolov & Bengio, *On the difficulty of training recurrent neural networks* (2012): https://arxiv.org/abs/1211.5063
  Analysed exploding gradients and proposed clipping the gradient norm.

## Further reading
- Sebastian Ruder, *An overview of gradient descent optimization algorithms*: https://arxiv.org/abs/1609.04747
- Gabriel Goh, *Why Momentum Really Works* (Distill): https://distill.pub/2017/momentum/
- Kingma & Ba, *Adam: A Method for Stochastic Optimization* (2014): https://arxiv.org/abs/1412.6980
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW, 2017): https://arxiv.org/abs/1711.05101
- Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts* (cosine schedules, 2016): https://arxiv.org/abs/1608.03983
- Pascanu, Mikolov & Bengio, *On the difficulty of training recurrent neural networks* (gradient clipping, 2012): https://arxiv.org/abs/1211.5063
- CS231n notes, *Neural Networks Part 3* (parameter updates): https://cs231n.github.io/neural-networks-3/
- PyTorch `torch.optim` docs: https://pytorch.org/docs/stable/optim.html
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from primer._show import banner, say, table, takeaway

# A loss function here takes a parameter vector and returns (loss, gradient).
LossFn = Callable[[np.ndarray], tuple[float, np.ndarray]]

# ---------------------------------------------------------------------------
# 1. Plain gradient descent on the simplest bowl, f(w) = w²
# ---------------------------------------------------------------------------


def descend_bowl(w0: float = 1.0, lr: float = 0.1, steps: int = 10) -> list[float]:
    """Gradient descent on f(w) = w², whose slope is 2w. Returns w at every step.

    Each step is w ← w − lr·2w = (1 − 2·lr)·w, so:
      * 0 < lr < 0.5: w shrinks smoothly toward 0,
      * lr = 0.5: lands exactly on 0 in one step,
      * 0.5 < lr < 1: overshoots but still converges (oscillating),
      * lr > 1: |1 − 2·lr| > 1, every step overshoots further: divergence.
    """
    w, path = w0, [w0]
    for _ in range(steps):
        w = w - lr * 2 * w
        path.append(w)
    return path


def momentum_on_bowl(w0: float = 1.0, lr: float = 0.1, beta: float = 0.9, steps: int = 10) -> list[float]:
    """Heavy-ball momentum on f(w) = w²: v ← β·v + g, w ← w − lr·v."""
    w, v, path = w0, 0.0, [w0]
    for _ in range(steps):
        g = 2 * w
        v = beta * v + g
        w = w - lr * v
        path.append(w)
    return path


# ---------------------------------------------------------------------------
# 2. Test landscapes
# ---------------------------------------------------------------------------

VALLEY_STEEPNESS = 100.0


def narrow_valley(p: np.ndarray) -> tuple[float, np.ndarray]:
    """f(x, y) = ½(x² + 100·y²): a long, narrow valley.

    Along y the walls are 100× steeper than the floor slopes along x. The
    steep direction caps the stable learning rate (lr < 2/100), and at that
    rate progress along the gentle x direction is painfully slow. Real loss
    surfaces are full of such valleys, which is why momentum and Adam exist.
    """
    x, y = p
    loss = 0.5 * (x**2 + VALLEY_STEEPNESS * y**2)
    return float(loss), np.array([x, VALLEY_STEEPNESS * y])


def rosenbrock(p: np.ndarray) -> tuple[float, np.ndarray]:
    """The classic banana-shaped valley: f = (1 − x)² + 100(y − x²)², minimum at (1, 1)."""
    x, y = p
    loss = (1 - x) ** 2 + 100 * (y - x**2) ** 2
    grad = np.array([-2 * (1 - x) - 400 * x * (y - x**2), 200 * (y - x**2)])
    return float(loss), grad


# ---------------------------------------------------------------------------
# 3. Optimizers (each keeps its own state between steps)
# ---------------------------------------------------------------------------


class SGD:
    """Stochastic gradient descent, optionally with momentum.

    momentum = 0: w ← w − lr·g.
    momentum = β: v ← β·v + g; w ← w − lr·v. The velocity v is a running sum
    of past gradients: directions that agree step after step build speed,
    directions that flip sign (the valley walls) cancel out.
    """

    def __init__(self, lr: float, momentum: float = 0.0):
        self.lr, self.momentum = lr, momentum
        self.v: np.ndarray | None = None

    def step(self, w: np.ndarray, g: np.ndarray) -> np.ndarray:
        if self.momentum:
            self.v = g if self.v is None else self.momentum * self.v + g
            return w - self.lr * self.v
        return w - self.lr * g


class Adam:
    """Adam: momentum plus a per-weight step size, with bias correction.

    m tracks the average gradient (direction), v the average squared
    gradient (typical size). Dividing m by √v rescales every weight's step to
    roughly `lr`, so steep and shallow directions move at similar speeds.

    `weight_decay` with `decoupled=True` is AdamW: shrink the weights directly,
    outside the adaptive rescaling. With `decoupled=False` the decay is added
    to the gradient as an L2 penalty (the original, flawed recipe).
    """

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decoupled: bool = True,
    ):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.weight_decay, self.decoupled = weight_decay, decoupled
        self.m: np.ndarray | None = None
        self.v: np.ndarray | None = None
        self.t = 0

    def step(self, w: np.ndarray, g: np.ndarray) -> np.ndarray:
        if self.weight_decay and not self.decoupled:
            g = g + self.weight_decay * w  # L2 penalty's gradient, fed through the adaptive scaling
        if self.m is None:
            self.m, self.v = np.zeros_like(w), np.zeros_like(w)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * g**2
        # m and v start at 0, so early on they underestimate; dividing by
        # (1 − β^t) corrects that. At t = 1 it recovers m̂ = g and v̂ = g² exactly.
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        w = w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        if self.weight_decay and self.decoupled:
            w = w - self.lr * self.weight_decay * w  # AdamW: plain shrinkage, not rescaled by √v̂
        return w


def run(optimizer, loss_fn: LossFn, w0: np.ndarray, steps: int = 100) -> dict:
    """Run `optimizer` on `loss_fn` from `w0`. Returns the path and the loss at every step."""
    w = np.array(w0, dtype=float)
    path, losses = [w.copy()], [loss_fn(w)[0]]
    for _ in range(steps):
        _, g = loss_fn(w)
        w = optimizer.step(w, g)
        path.append(w.copy())
        losses.append(loss_fn(w)[0])
    return dict(path=np.array(path), loss=np.array(losses))


def adam_first_step(gradient: float, lr: float = 0.01) -> float:
    """How far Adam moves a weight on its very first step, for a given gradient."""
    w = np.array([0.0])
    w_new = Adam(lr=lr).step(w, np.array([gradient]))
    return float(abs(w_new[0] - w[0]))


def one_decay_step(w: float = 1.0, lr: float = 0.1, decay: float = 0.1, kind: str = "adamw") -> float:
    """One step with zero loss-gradient, so only weight decay acts. Returns the new weight.

    kind="adamw": decoupled decay (AdamW). kind="adam_l2": L2 penalty inside Adam.
    """
    optimizer = Adam(lr=lr, weight_decay=decay, decoupled=(kind == "adamw"))
    return float(optimizer.step(np.array([w]), np.array([0.0]))[0])


# ---------------------------------------------------------------------------
# 4. Learning-rate schedule and gradient clipping
# ---------------------------------------------------------------------------


def warmup_cosine(step: int, peak: float, warmup: int, total: int, floor: float = 0.0) -> float:
    """Linear warmup from 0 to `peak` over `warmup` steps, then cosine decay to `floor` at `total`."""
    if step < warmup:
        return peak * step / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))  # 0 → 1 across the decay phase
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def clip_by_global_norm(grads: list[np.ndarray], max_norm: float) -> tuple[list[np.ndarray], float]:
    """Scale all gradients by one common factor so their combined length is at most `max_norm`.

    Measuring the length across *all* tensors together (not each separately)
    keeps the overall direction of the update unchanged; only its size shrinks.
    Returns (clipped gradients, the norm before clipping).
    """
    norm = float(np.sqrt(sum(float(np.sum(g**2)) for g in grads)))
    scale = min(1.0, max_norm / (norm + 1e-12))
    return [g * scale for g in grads], norm


# ---------------------------------------------------------------------------
# 5. Figures (rendered by `make figures`)
# ---------------------------------------------------------------------------


def figures() -> dict:
    """Plots computed from this module's own functions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = {}

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for lr in (0.001, 0.1, 0.45, 1.1):
        path = np.array(descend_bowl(1.0, lr, steps=30))
        ax.plot(np.maximum(path**2, 1e-30), label=f"η = {lr}")
    ax.set(yscale="log", ylim=(1e-12, 1e3), xlabel="step", ylabel="loss w² (log scale)",
           title="Learning rate on the bowl f(w) = w²")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figs["learning_rates"] = fig

    start = np.array([-8.0, 1.0])
    runs = {
        "SGD (η = 0.015)": run(SGD(lr=0.015), narrow_valley, start, 100)["path"],
        "momentum (η = 0.015, β = 0.9)": run(SGD(lr=0.015, momentum=0.9), narrow_valley, start, 100)["path"],
        "Adam (η = 0.3)": run(Adam(lr=0.3), narrow_valley, start, 100)["path"],
    }
    xs, ys = np.meshgrid(np.linspace(-9, 3, 300), np.linspace(-1.3, 1.3, 200))
    zz = 0.5 * (xs**2 + VALLEY_STEEPNESS * ys**2)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.contour(xs, ys, zz, levels=np.geomspace(0.05, 200, 14), colors="lightgray", linewidths=0.8)
    for (name, path), c in zip(runs.items(), ("C0", "C1", "C2")):
        ax.plot(path[:, 0], path[:, 1], ".-", ms=2.5, lw=0.8, color=c, label=name)
    ax.scatter([0], [0], marker="*", s=180, color="black", zorder=5, label="minimum")
    ax.scatter([start[0]], [start[1]], s=30, color="black", zorder=5)
    ax.set(xlabel="x (gentle direction)", ylabel="y (steep direction)",
           title="100 steps on the narrow valley ½(x² + 100y²)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    figs["trajectories"] = fig

    steps = np.arange(0, 1001)
    lrs = [warmup_cosine(int(t), peak=1e-3, warmup=100, total=1000, floor=1e-5) for t in steps]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(steps, lrs)
    marks = [50, 100, 550, 1000]
    ax.scatter(marks, [warmup_cosine(t, 1e-3, 100, 1000, 1e-5) for t in marks], color="C3", zorder=3)
    ax.axvline(100, color="gray", ls="--", lw=0.8)
    ax.text(105, 2e-4, "end of warmup", fontsize=8)
    ax.set(xlabel="training step", ylabel="learning rate", title="Linear warmup, then cosine decay")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figs["schedule"] = fig
    return figs


# ---------------------------------------------------------------------------
# 6. Walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. Gradient descent on the bowl w²")
    path = descend_bowl(1.0, 0.1, steps=3)
    table(["step", "w", "slope 2w"], [(i, w, 2 * w) for i, w in enumerate(path)], floatfmt=".3f")
    takeaway("Feel the slope, step against it, repeat.")

    banner("2. The learning rate decides everything")
    rows = []
    for lr in (0.001, 0.1, 0.5, 1.1):
        p = descend_bowl(1.0, lr, steps=10)
        rows.append((lr, 1 - 2 * lr, p[1], p[3], p[10]))
    table(["η", "multiplier 1 − 2η", "w after 1", "w after 3", "w after 10"], rows, floatfmt=".4g")
    say("0.001 barely moves; 0.5 lands in one step; 1.1 overshoots further every step and diverges.")

    banner("3. Momentum, SGD and Adam in a narrow valley")
    start = np.array([-8.0, 1.0])
    results = [
        ("SGD η=0.015", run(SGD(lr=0.015), narrow_valley, start, 300)["loss"]),
        ("momentum η=0.015 β=0.9", run(SGD(lr=0.015, momentum=0.9), narrow_valley, start, 300)["loss"]),
        ("Adam η=0.1", run(Adam(lr=0.1), narrow_valley, start, 300)["loss"]),
    ]
    table(["optimizer", "loss @10", "loss @100", "loss @300"],
          [(n, l[10], l[100], l[300]) for n, l in results], floatfmt=".2e")
    say(
        """
        The walls are 100× steeper than the floor, which caps SGD's learning
        rate below 0.02, so it crawls along the floor. Momentum's velocity
        builds up along the floor; Adam rescales each direction separately.
        """
    )

    banner("4. Adam's first step is the learning rate, whatever the gradient")
    table(["gradient", "first step"], [(g, adam_first_step(g, 0.01)) for g in (1000.0, 1.0, 0.001)], floatfmt=".6f")

    banner("5. Weight decay: L2-in-Adam vs. AdamW")
    table(
        ["λ", "AdamW", "Adam + L2"],
        [(lam, one_decay_step(1.0, 0.1, lam, "adamw"), one_decay_step(1.0, 0.1, lam, "adam_l2")) for lam in (0.1, 0.001)],
        floatfmt=".4f",
    )
    takeaway("Inside Adam, an L2 penalty gets normalized away; AdamW applies decay directly, so λ means what it says.")

    banner("6. Warmup + cosine decay")
    table(["step", "learning rate"], [(t, warmup_cosine(t, 1e-3, 100, 1000, 1e-5)) for t in (0, 50, 100, 325, 550, 775, 1000)],
          floatfmt=".2e")

    banner("7. Clipping by global norm")
    clipped, norm = clip_by_global_norm([np.array([3.0]), np.array([4.0])], 1.0)
    say(f"Pieces (3) and (4): global norm {norm:.1f}; clipped to ({clipped[0][0]:.1f}) and ({clipped[1][0]:.1f}), length 1, same direction.")


if __name__ == "__main__":
    demo()
