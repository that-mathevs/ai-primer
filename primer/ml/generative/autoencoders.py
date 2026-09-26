r"""
# Autoencoders and VAEs

Run: `python -m primer.ml.generative.autoencoders`

## The everyday picture

You phone a friend and describe a picture so they can draw it, but you are
allowed to say only two numbers. You would agree on a system first: the first
number says which way the line runs, the second says how far it sits from the
middle. You squeeze the picture into two numbers, send them, and your friend
rebuilds it.

An **autoencoder** is a neural network that invents that system by itself. It
has two halves. The **encoder** squeezes a picture into a few numbers, called
the **code**. The **decoder** rebuilds the picture from the code. Nobody tells
it what the numbers should mean. The only instruction is "the rebuild must
match the original", and to obey it through such a narrow gap the network has
to discover what really varies in the data.

Then comes a second question. If your friend can draw from *any* two numbers,
can you invent a new picture by making up two numbers? With a plain
autoencoder, usually not: most made-up codes were never used for anything,
and the drawing comes out as a smudge. A **variational autoencoder (VAE)** is
trained so that a made-up code drawn from a known range draws something
sensible. That step, from compressing to generating, is what this lesson is
about, and it opens this part of the primer: GANs
(`primer.ml.generative.gans`) and diffusion (`primer.ml.generative.diffusion`)
are other answers to the same question.

## The toy data: pictures of pen strokes

**Tiny example.** Every picture in this lesson is 8 × 8 = 64 pixels holding
one soft pen stroke. The stroke runs in one of four directions (horizontal,
vertical, diagonal down, diagonal up) and is shifted up to 2.5 pixels from
the centre. The ink fades like a bell curve either side of the line: a pixel
on the line has brightness 1.0, a pixel one pixel away about 0.25, a pixel
two away almost nothing.

So each picture is 64 numbers, but only **two facts** change from picture to
picture: the direction (one of four) and the offset (any amount). A good
2-number code has to rediscover both from the pixels alone. The 200 pictures
are the top row of the first figure below.

**In code:** `make_strokes` draws the pictures and returns their hidden directions and offsets; `STROKE_KINDS` names the four directions.

## Squeeze and rebuild: the autoencoder

**Everyday picture.** A zip file squeezes a document and gets it back
exactly. An autoencoder is a *lossy*, *learned* zip: it keeps what matters
most for this kind of data and lets the rest go.

**Tiny example: a one-number code you can check by hand.** Take points that
lie on the line y = 2x, such as (1, 2) and (2, 4). Each takes two numbers to
write down, but one number is enough: how far along the line the point is.

- **Encoder:** code = u · x, where u = (1, 2)/√5 = (0.447, 0.894) is the
  line's direction scaled to length 1, and "·" is the dot product (multiply
  matching entries and add; see `primer.notation`).
- **Decoder:** rebuild = code × u.

For (1, 2): code = 0.447 + 1.789 = 2.236 (which is √5, its distance from the
origin), and rebuild = 2.236 × (0.447, 0.894) = (1, 2). Nothing lost.

For (2, 3), which is *off* the line: code = (2 + 6)/√5 = 3.578, and
rebuild = 3.578 × (0.447, 0.894) = (1.6, 3.2), the closest point on the line.
The part that pointed away from the line, (0.4, −0.2), is gone. That loss is
exactly what training measures and shrinks:

$$
L_{\text{rec}} = \frac{1}{n} \sum_{i=1}^{n} \bigl\lVert x_i - g\bigl(f(x_i)\bigr) \bigr\rVert^2
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $x_i$ | the $i$-th example: a point, or a picture's 64 pixels | $x = (2, 3)$ |
| $f$ | the encoder: squeezes an example into a code | $f(x) = u \cdot x$ |
| $f(x_i)$ | the code for example $i$, often written $z_i$ | 3.578 |
| $g$ | the decoder: rebuilds an example from a code | $g(z) = z\,u$ |
| $g(f(x_i))$ | the rebuild, often written $\hat{x}_i$ ("x hat") | (1.6, 3.2) |
| $x_i - \hat{x}_i$ | what the rebuild got wrong, entry by entry | (0.4, −0.2) |
| $\lVert v \rVert^2$ | squared length of $v$: square every entry and add | $0.4^2 + 0.2^2 = 0.2$ |
| $\sum_{i=1}^{n}$ | add up over every example | |
| $n$ | how many examples | 2: (1, 2) and (2, 3) |
| $L_{\text{rec}}$ | the reconstruction loss: average squared error of the rebuilds | 0.1 |

**In words:** "squeeze each example, rebuild it, measure the squared distance
between the rebuild and the original, and average over all the examples."

**With the numbers:** (1, 2) rebuilds perfectly, error 0. (2, 3) rebuilds as
(1.6, 3.2), error 0.4² + (−0.2)² = 0.16 + 0.04 = 0.2. Averaged over the two
points, $L_{\text{rec}}$ = (0 + 0.2)/2 = 0.1.

**In Python:**

```python
import math
# u: the line's direction, scaled to length 1
u = [1 / math.sqrt(5), 2 / math.sqrt(5)]
def f(x):
    return sum(u_m * x_m for u_m, x_m in zip(u, x))
def g(z):
    return [z * u_m for u_m in u]
def error(x):
    return sum((a - b) ** 2 for a, b in zip(x, g(f(x))))
# f(x): the encoder squeezes (2, 3) to one number
round(f([2, 3]), 3)  # → 3.578
# g(f(x)): the decoder rebuilds two numbers from it
[round(v, 2) for v in g(f([2, 3]))]  # → [1.6, 3.2]
# ‖x − x̂‖² for each point: (1, 2) is on the line, (2, 3) is not
[round(error(x), 2) for x in ([1, 2], [2, 3])]  # → [0.0, 0.2]
# L_rec: the average over n = 2 points
round((error([1, 2]) + error([2, 3])) / 2, 2)  # → 0.1
```

The real network has the same two halves, only bent. The encoder is a
two-layer network like the one built in `primer.ml.neural_net`: 64 pixels
into 32 hidden numbers (through tanh, which lets it bend), then out to a code
of 2 numbers. The decoder mirrors it: 2 numbers into 32 hidden, then out to
64 pixels squashed between 0 and 1 by a sigmoid so they are valid
brightnesses. Training is the loop from `primer.ml.neural_net` with the Adam
optimizer from `primer.ml.optimizers`: rebuild every picture, measure
$L_{\text{rec}}$, send the gradient back through both halves, adjust, 1,500
times.

```mermaid
flowchart LR
  X["picture x<br/>64 pixels"] --> E["encoder f<br/>64 → 32 → 2"]
  E --> Z["code z<br/>2 numbers"]
  Z --> D["decoder g<br/>2 → 32 → 64"]
  D --> XH["rebuild x̂<br/>64 pixels"]
  X --> L["loss<br/>squared error between x and x̂"]
  XH --> L
  L -. "gradients flow back<br/>through decoder, then encoder" .-> E
```

**Reading it:** follow the picture left to right. It is 64 numbers wide at
both ends and only 2 numbers wide in the middle: that narrow middle is the
**bottleneck**, and it is the whole point. Without it the network could copy
the pixels straight through and learn nothing. The loss box compares the two
ends, and the dotted arrow is backpropagation carrying the blame back through
the decoder and on into the encoder, so both halves learn together. The
encoder never sees a target code; it learns whatever code the decoder finds
most useful.

![Eight strokes, their autoencoder rebuilds from 2 numbers (error 0.07) and their PCA rebuilds from 2 numbers (error 4.42): the autoencoder's are near perfect, PCA's are grey smudges](figures/primer.ml.generative.autoencoders.reconstructions.svg)

**Reading it:** the top row is eight real strokes, two of each direction.
The middle row is what the autoencoder rebuilds from just 2 numbers per
picture: nearly identical. The average error is about 0.07, against about
8.8 units of squared ink in a whole stroke, so it keeps over 99% of the
picture. The bottom row squeezes the same pictures to 2 numbers with PCA and
rebuilds them: grey smudges, keeping only about half. Same budget of two
numbers, very different results. The next section explains why.

**In code:** `worked_example_line` is the hand example above; `Autoencoder` holds both halves and its hand-written backward pass in `Autoencoder.loss_and_gradients`; `train` runs Adam; `trained_autoencoder` is the lesson's trained network; `reconstruction_error` measures the rebuilds.

### PCA is the straight-line special case

**Everyday picture.** PCA (principal component analysis, built in
`primer.ml.embeddings.clustering`) fits a flat sheet through the data and
records where each point lands on the sheet. That is an autoencoder with no
bends: the encoder is one matrix multiply, and so is the decoder.

**Tiny example.** Scatter 100 points near the line y = 2x and train exactly
that no-bend autoencoder, with a 1-number code, by plain gradient descent.
The direction it learns is (0.447, 0.894), the line's own direction, and its
rebuild error matches PCA's with one component. Baldi and Hornik proved in
1989 that this always happens: a linear autoencoder trained on squared error
lands on the same flat sheet as PCA.

So why did PCA smudge the strokes? Because a stroke sliding across the image
does not travel in a straight line through pixel space. Take a horizontal
stroke at the top and another at the bottom. Their average, the point halfway
along the straight line between them, is *two faint lines*, not one stroke in
the middle. The real strokes lie on a curved surface in the 64-dimensional
space of pictures, and a flat sheet can only cut through it. The
autoencoder's tanh layers let its "sheet" bend to follow the curve.

**Why it matters in practice.** Before generation, autoencoders earned their
keep as learned compressors. They compress, they denoise (train on noisy
inputs, ask for clean outputs), and they detect anomalies: an input the
network rebuilds badly is unlike anything it trained on, which is a standard
way to flag fraud or a failing machine.

![The plain autoencoder's 200 codes form separate strands, one colour per stroke direction, spread from about -24 to 14, far from the small circle where a standard normal draw usually lands](figures/primer.ml.generative.autoencoders.ae_codes.svg)

**Reading it:** each dot is one picture, placed at its 2-number code and
coloured by its stroke direction, with bigger dots for larger offsets. The
codes form **strands**: pictures of one direction line up along a curve (now
and then broken into pieces), and sliding along a strand slides the stroke.
The network rediscovered both hidden facts without being told either. Now
look at the axes: the codes run from about −24 to 14. Nothing asked for that
range; it is an accident of training. The dashed circle near 0 is where a
"random" code from the bell curve would usually land, and it catches almost
none of the strands. Hold on to that for the next section.

**In code:** `train_linear_autoencoder` is the no-bend autoencoder; `pca_reconstruction` rebuilds from the top principal components.

## Why a plain autoencoder cannot generate: holes

**Everyday picture.** Imagine a town where houses were built only along four
winding roads. Pick a random spot inside the town limits and you will most
likely land in a field. Ask the decoder to draw the picture that "lives" in
that field, and it answers anyway, with whatever its weights happen to
produce, because nothing in training ever asked it about that spot.

**Tiny example.** The obvious way to make up a random code is to draw each
number from the **standard normal distribution**, the bell curve centred on
0 with a spread (standard deviation) of 1: most draws fall between −1 and 1,
almost all between −3 and 3. Written $\mathcal{N}(0, I)$ for several numbers
at once, it means "draw each number from that bell curve, independently".

Draw the code z = (1.2, −0.9) and decode it with the plain autoencoder. The
result is a bright smear with about three times the ink of any real stroke;
its squared distance to the nearest real stroke is about 7.5. Do that 500
times and the median distance is about 3, while three quarters of the draws
land further than 1.0 from every real stroke. That 1.0 is this lesson's line
for **junk**: about a ninth of a whole stroke's squared ink.

To be fair to the autoencoder, draw instead from the box that holds its own
codes, from −24 to 1.6 across and −17 to 14 up. Still about 44% of those
codes decode to junk. They fall in the **holes** between the strands.

![Left: the autoencoder's codes and 300 random codes from their bounding box, 44 percent marked as junk; right: the worst decoded ones are black blobs and the best are clean strokes](figures/primer.ml.generative.autoencoders.holes.svg)

**Reading it:** on the left, black dots are the codes of real strokes and the
dashed rectangle is the box around them. Every blue circle is a random code
that decoded to something close to a real stroke; every red cross decoded to
junk. The red crosses sit in the open spaces between strands, the blue
circles near them. On the right are the decoded pictures themselves: the 12
worst (top two rows) are black blobs no pen would draw, and the 12 best
(bottom) are clean strokes, because those random codes happened to land on
a strand.

**Why it matters in practice.** A compressor is not a generator. To sample
new data you need a code space with a **known shape** that you can draw from,
**no holes** inside that shape, and **smoothness**, so nearby codes decode to
similar pictures. A plain autoencoder promises none of these, because its
loss only ever looks at codes of real pictures.

**In code:** `generate` decodes codes drawn from the standard normal; `codes_in_box` draws from a model's own code range; `distance_to_data` measures how far each image is from the nearest real stroke, and `JUNK_DISTANCE` is the line between plausible and junk.

## The VAE: encode to a fuzzy region, not a point

**Everyday picture.** Instead of pinning each picture to one exact spot on
the map, the encoder draws a small fuzzy circle: "somewhere around here".
During training the decoder is handed a random point from inside that circle,
so it must draw the right picture from anywhere nearby. A whole neighbourhood
now decodes sensibly, not just one pin. A second rule stops the encoder from
cheating: every circle pays **rent**, more the further it sits from the
middle of the map and more if it shrinks towards a pin. Circles crowd
towards the centre and overlap, and the fields between the roads fill in.

**Tiny example.** One picture, one code number. The encoder says
μ = 0.5 and σ = 0.5: "about 0.5, give or take 0.5". This step, a bell-curve
roll gives ε = 1.2, so the decoder is handed z = 0.5 + 0.5 × 1.2 = 1.1. Next
step the roll is ε = −0.4 and the decoder gets z = 0.5 − 0.2 = 0.3. Both must
rebuild the same picture.

```mermaid
flowchart LR
  X["picture x"] --> E["encoder"]
  E --> MU["μ: centre of the region"]
  E --> LV["log σ²: size of the region"]
  EPS["ε drawn from N(0, I)"] --> Z["z = μ + σ·ε"]
  MU --> Z
  LV --> Z
  Z --> D["decoder"] --> XH["rebuild x̂"]
  XH --> R["rebuild error<br/>‖x − x̂‖²"]
  X --> R
  MU --> KL["KL rent<br/>pulls regions to N(0, I)"]
  LV --> KL
  R --> LOSS["loss = rebuild + β · KL"]
  KL --> LOSS
```

**Reading it:** compare it with the plain autoencoder's diagram. The encoder
now has two outputs per code number: a centre μ and a size, given as
log σ² (the logarithm of the variance, used because it can be any number
while σ itself must stay positive). The box z = μ + σ·ε is where the fuzzy
region becomes one concrete code: ε is fresh random noise every step. The
loss has two parts. The rebuild error, as before, wants each region small and
distinct so the decoder knows exactly which picture it came from. The KL
rent, fed straight from μ and log σ², wants every region to look like the
standard bell curve. Training settles on a compromise between them, and β
sets the exchange rate.

**In code:** `VAE` is the `Autoencoder` with a two-headed encoder; `VAE.encode_distribution` returns μ and log σ², and `VAE.encode` returns just the centre μ, the best single code for a picture.

### The reparameterization trick: moving the dice outside

**Everyday picture.** To pick a random seat in a row, you could close your
eyes and point. If someone then moves the row one seat to the left, you have
no idea how your pick would have changed. Or you could roll a die for "how
many seats from the middle" and count from wherever the middle is. Now if the
row moves one seat left, your seat moves exactly one seat left. Same kind of
random seat, but you can say how it responds to the row moving.

Training needs exactly that. Backpropagation asks of every step: "if I nudge
this number, how does the loss change?" (that rate of change is the
**gradient**, or **derivative**; see `primer.notation`). A raw dice roll has
no answer, so the gradient would stop at the sampling step and the encoder
would never learn. The **reparameterization trick** rolls the dice first, as
an ordinary input ε, and builds the code with arithmetic:

$$
z = \mu + \sigma \odot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I), \qquad \sigma = e^{\frac{1}{2} \log \sigma^2}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $\mu$ | the centre of the picture's region, from the encoder | 0.5 |
| $\log \sigma^2$ | the natural logarithm of the region's variance, from the encoder; "log of y" is the power you raise $e$ to in order to get y | $\log 0.25 = -1.386$ |
| $e$ | Euler's number, ≈ 2.718 | |
| $\sigma$ | the spread of the region (standard deviation); $e^{\frac{1}{2}\log\sigma^2}$ undoes the log and the square | 0.5 |
| $\varepsilon$ | "epsilon": random noise, one number per code number | 1.2 |
| $\sim$ | "is drawn from" | |
| $\mathcal{N}(0, I)$ | the standard normal: centre 0, spread 1, each number independent ($I$, the identity matrix, says "no links between numbers") | |
| $\odot$ | multiply matching entries (element by element) | |
| $z$ | the code the decoder receives this step | 1.1 |

**In words:** "roll a standard bell-curve number, stretch it by the region's
spread, and shift it to the region's centre."

**With the numbers:** $\sigma = e^{\frac{1}{2} \times (-1.386)} = e^{-0.693}
= 0.5$, so $z = 0.5 + 0.5 \times 1.2 = 1.1$. Now the gradients have a route:
nudge μ by a little and z moves by the same amount (∂z/∂μ = 1); nudge σ by a
little and z moves by ε times as much (∂z/∂σ = ε = 1.2). The symbol ∂ reads
"how much this changes when that is nudged".

**In Python:**

```python
import math
mu, log_var, eps = 0.5, math.log(0.25), 1.2
round(log_var, 3)  # → -1.386
# σ = e^(½ log σ²)
sigma = math.exp(0.5 * log_var)
round(sigma, 3)  # → 0.5
# z = μ + σ ⊙ ε
z = mu + sigma * eps
round(z, 3)  # → 1.1
# nudge μ, then σ, by a tiny h and watch z: ∂z/∂μ = 1 and ∂z/∂σ = ε
h = 1e-6
round((mu + h + sigma * eps - z) / h, 3)  # → 1.0
round((mu + (sigma + h) * eps - z) / h, 3)  # → 1.2
```

```mermaid
flowchart LR
  subgraph A["Sampling directly: the gradient stops"]
    direction LR
    m1["μ, σ"] --> s1["draw z from N(μ, σ²)<br/>a dice roll"] --> d1["decoder"] --> l1["loss"]
    l1 -. "no route back<br/>through a dice roll" .-> s1
  end
  subgraph B["Reparameterized: the gradient flows"]
    direction LR
    e2["ε from N(0, I)<br/>just another input"] --> s2["z = μ + σ·ε<br/>plain arithmetic"]
    m2["μ, σ"] --> s2 --> d2["decoder"] --> l2["loss"]
    l2 -. "∂z/∂μ = 1, ∂z/∂σ = ε" .-> m2
  end
```

**Reading it:** both rows produce the same kind of random code: a draw from a
bell curve centred on μ with spread σ. In the top row the randomness sits
*between* the encoder's outputs and the loss, and the dotted arrow of
backpropagation has nowhere to go. In the bottom row the randomness comes in
from the side as ε, an input like a pixel, and everything from μ and σ to the
loss is ordinary arithmetic the chain rule can pass through. The encoder
learns because of this one rearrangement.

**Why it matters in practice.** The same trick lets gradients pass through
random choices elsewhere too, for example in the noisy steps of diffusion
models (`primer.ml.generative.diffusion`) and in reinforcement learning with
continuous actions. It is a large part of why the VAE paper mattered.

**In code:** `reparameterize` is the formula; in `VAE.loss_and_gradients` the lines under "Through z = μ + σ·ε" are the two gradients above, and a test checks them against finite differences.

### The KL penalty: rent for every region

**Everyday picture.** The rent from the everyday picture has a formal name:
the **KL divergence** (Kullback-Leibler divergence) from the region to the
standard bell curve, a measure of how different two distributions are that
is 0 only when they match (`primer.ml.training_stages` uses it for
distillation). For a bell-curve region and the standard bell curve, it has a
short closed form:

$$
\mathrm{KL}\bigl(\mathcal{N}(\mu, \sigma^2) \,\big\Vert\, \mathcal{N}(0, 1)\bigr) = \frac{1}{2} \sum_{j=1}^{k} \bigl( \mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1 \bigr)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $\mathrm{KL}(q \,\Vert\, p)$ | how much distribution $q$ differs from distribution $p$; 0 when they match | |
| $\mathcal{N}(\mu, \sigma^2)$ | the picture's region: a bell curve with centre μ and variance σ² | $\mathcal{N}(0.5, 0.25)$ |
| $\mathcal{N}(0, 1)$ | the standard bell curve every region is pulled towards | |
| $k$ | how many numbers in the code | 1 (the lesson's network uses 2) |
| $j$ | a counter over the code numbers | |
| $\mu_j^2$ | rent for sitting away from the centre: 0 at μ = 0, growing on both sides | 0.25 |
| $\sigma_j^2 - \log \sigma_j^2$ | rent on size: smallest (exactly 1) when σ = 1, huge as σ shrinks to a pin, growing if it bloats | $0.25 + 1.386$ |
| $-1$ | shifts the total so a perfect match costs exactly 0 | |
| $\frac{1}{2}$ | a constant that falls out of the bell curve's formula | |

**In words:** "for each code number, add the squared distance of its centre
from 0, plus its variance, minus the log of its variance, minus 1; add those
up over the code numbers and halve."

**With the numbers:** for μ = 0.5 and σ = 0.5,
½ (0.25 + 0.25 − (−1.386) − 1) = ½ × 0.886 = **0.443**. It splits into two
rents: the centre alone costs ½ × 0.25 = 0.125, the shrunken size alone
½ (0.25 + 1.386 − 1) = 0.318, and 0.125 + 0.318 = 0.443. A second code
number that already has μ = 0 and σ = 1 adds ½ (0 + 1 − 0 − 1) = 0, so the
2-number total is still 0.443. Shrink σ to a pin of 0.05 and the rent jumps
to 2.62.

**In Python:**

```python
import math
def kl(mu, var):
    return 0.5 * (mu ** 2 + var - math.log(var) - 1)
# one code number with μ = 0.5, σ = 0.5 (σ² = 0.25)
round(kl(0.5, 0.25), 3)  # → 0.443
# the rent for the centre alone, then for the size alone
round(0.5 * 0.5 ** 2, 3)  # → 0.125
round(kl(0.0, 0.25), 3)  # → 0.318
# a code number that already is the standard normal pays nothing
kl(0.0, 1.0)  # → 0.0
# Σ over j: two code numbers, their rents add
round(kl(0.5, 0.25) + kl(0.0, 1.0), 3)  # → 0.443
# shrink σ to a pin (σ = 0.05) and the rent jumps
round(kl(0.5, 0.05 ** 2), 2)  # → 2.62
```

![Left: the KL penalty is a parabola in the centre mu, with mu = 0.5 costing 0.125; right: in the spread sigma it is zero at sigma = 1, rises steeply towards a pin, with sigma = 0.5 costing 0.318](figures/primer.ml.generative.autoencoders.kl_penalty.svg)

**Reading it:** the left curve holds the spread at σ = 1 and moves the
centre: a bowl with its bottom at μ = 0, so sitting at 0.5 costs 0.125 and
sitting at 3 costs 4.5. The right curve holds the centre at 0 and changes
the spread. It is 0 at σ = 1 (the dashed line), rises gently if the region
bloats, and shoots up as σ heads towards 0. That steep wall on the left is
what stops the encoder from shrinking every region to a pin and turning back
into a plain autoencoder with holes. The two red dots add up to the worked
example's 0.443.

**In code:** `kl_to_standard_normal` is the formula, summed over the code numbers; a test checks it against a brute-force average over 400,000 samples.

### The whole VAE loss

$$
L = \bigl\lVert x - \hat{x} \bigr\rVert^2 + \beta \cdot \mathrm{KL}\bigl(q(z \mid x) \,\big\Vert\, \mathcal{N}(0, I)\bigr)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $x$ | the picture | (2, 3) from the line example |
| $\hat{x}$ | the decoder's rebuild from the sampled code $z$ | (1.6, 3.2) |
| $\lVert x - \hat{x} \rVert^2$ | the rebuild error, as in the plain autoencoder | 0.2 |
| $q(z \mid x)$ | the region the encoder draws for picture $x$: read "q of z given x" | $\mathcal{N}(0.5, 0.25)$ |
| $\beta$ | "beta": how much one unit of KL rent costs, in units of rebuild error | 0.3 |
| KL(…) | the rent from the previous section | 0.443 |

**In words:** "rebuild well, but pay β for every unit by which your regions
differ from the standard bell curve."

**With the numbers:** 0.2 + 0.3 × 0.443 = 0.2 + 0.133 = 0.333.

**In Python:**

```python
rebuild, kl_value, beta = 0.2, 0.443, 0.3
# L = ‖x − x̂‖² + β · KL
round(rebuild + beta * kl_value, 3)  # → 0.333
```

Where does this come from? The VAE paper derives the loss (with β = 1) as the
negative of the **evidence lower bound (ELBO)**: a quantity that is
guaranteed never to exceed the log-probability the model gives to the real
pictures, so pushing the bound up pushes that probability up. Choosing
squared error amounts to assuming each pixel is the decoder's output plus
bell-curve noise of spread s, and working that through gives β = 2s². This
lesson uses β = 0.3, which assumes pixel noise of spread about 0.39.

![The VAE's 200 codes packed within about two units of the origin, one strand per stroke direction, each with a small shaded fuzzy region](figures/primer.ml.generative.autoencoders.vae_codes.svg)

**Reading it:** the same 200 pictures as the plain autoencoder's map, now
placed at their centres μ, each with its fuzzy region shaded around it
(regions are drawn one σ wide). The dashed circles are radius 1 and 2 of the
standard bell curve. Compare the axes with the plain autoencoder's: the codes
now sit around 0 with a spread of about 1.1 in each direction, instead of
wandering from −24 to 14. The strands are still there (the code still knows
direction and offset), but they are packed side by side and the regions
(σ about 0.14) overlap along each strand, so there is far less empty field
left.

**In code:** `VAE.loss_and_gradients` computes both terms and their gradients; `trained_vae` trains the lesson's VAE at a chosen β.

## Sampling new strokes

**Everyday picture.** Once the map is packed, throw a dart at the middle of
it and you land in a neighbourhood, not a field. Generating is exactly that:
throw away the encoder, draw a code from the standard bell curve, and decode.

**Tiny example.** Draw the same code as before, z = (1.2, −0.9), and decode
it with the VAE. Out comes a horizontal stroke through the centre of the
picture, with squared distance about 0.06 from a real stroke in the training
set. The plain autoencoder turned that same code into a smear 7.5 away.
Over 500 draws the VAE's median distance is about 0.56 against the plain
autoencoder's 3.

```mermaid
flowchart LR
  N["draw z from N(0, I)<br/>2 random numbers"] --> D["decoder g"] --> X["a new picture<br/>never seen in training"]
  ENC["encoder"] -. "not used when generating" .-> N
```

**Reading it:** generating uses only the right half of the network. The
encoder's job was done during training: it taught the decoder, through the
KL rent, that codes live where the standard normal puts them. That is why the
first box can draw from $\mathcal{N}(0, I)$ with confidence. The dotted arrow
is a reminder that nothing flows from the encoder here.

![Left: 24 codes from N(0, I) decoded by the plain autoencoder, many black blobs; right: the same codes decoded by the VAE, clean strokes and a few soft blends](figures/primer.ml.generative.autoencoders.samples.svg)

**Reading it:** both panels decode the *same* 24 random codes. The plain
autoencoder (left) gives blobs, broken lines and a few strokes by luck. The
VAE (right) gives strokes in all four directions at many offsets, plus a few
soft blends of two directions. Those blends are the VAE's honest weak spot:
with only 2 numbers to hold four directions, some codes sit on the border
between two strands, and the decoder hedges between them. About a third of
the VAE's samples still cross the junk line, most of them blends like these.

![An 11 by 11 grid of codes from -2.2 to 2.2, each decoded by the VAE: neighbouring tiles are similar strokes, and regions of the map hold each direction](figures/primer.ml.generative.autoencoders.latent_grid.svg)

**Reading it:** every tile is the VAE's decoding of one point on an even
grid over the middle of the code map. Read along any row or column: the
stroke changes a little at a time, sliding or turning, with no sudden jumps
to junk. Whole regions of the map belong to one direction, and the borders
between them are where the blends live. This is what a **latent space**
means: a space of codes where position means something. That smoothness is
what lets you edit or explore a picture by moving its code.

### Interpolation: walking between two codes

**Everyday picture.** A morph between two faces in a film: every in-between
frame should look like a face, not a double exposure.

**Tiny example.** Encode two pictures to codes $z_a$ and $z_b$, step along
the straight line between them, and decode every stop:

$$
z_t = (1 - t)\, z_a + t\, z_b, \qquad 0 \le t \le 1
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $z_a, z_b$ | the codes of the two pictures at the ends of the walk | $(-1, 0.5)$ and $(1, 1.5)$ |
| $t$ | how far along the walk: 0 at $z_a$, 1 at $z_b$ | 0.25 |
| $z_t$ | the code at that point of the walk | $(-0.5, 0.75)$ |

**In words:** "take a share $1 - t$ of the first code and a share $t$ of the
second, and add them."

**With the numbers:** at $t = 0.25$, $z_t = 0.75 \times (-1, 0.5) + 0.25
\times (1, 1.5) = (-0.75 + 0.25,\ 0.375 + 0.375) = (-0.5, 0.75)$. The walk in
the figure uses nine stops, $t = 0, 1/8, 2/8, \ldots, 1$.

**In Python:**

```python
z_a, z_b = [-1.0, 0.5], [1.0, 1.5]
# z_t = (1 − t) z_a + t z_b, a quarter of the way along
t = 0.25
[(1 - t) * a + t * b for a, b in zip(z_a, z_b)]  # → [-0.5, 0.75]
# the nine stops of a walk
[i / 8 for i in range(9)]  # → [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
```

![Two walks between the same two vertical strokes: the plain autoencoder's passes through a smeared cross 3.88 from any real stroke; the VAE's passes through diagonal strokes, never more than 0.70 from a real one](figures/primer.ml.generative.autoencoders.interpolation.svg)

**Reading it:** both rows walk between the same two vertical strokes, one
left of centre and one right of it. The plain autoencoder's walk (top)
leaves its strand and crosses a hole: the middle frames are a smeared
cross, 3.88 away from any real stroke. The VAE's walk (bottom) takes a
surprising route, through diagonal strokes, because in its 2-number map the
diagonal strand sits between those two points. But every stop is a
believable stroke, never more than 0.70 from a real one. Averaged over 60
random pairs, the VAE's biggest jump from one frame to the next is also
smaller (about 1.6 against 2.4).

**Why it matters in practice.** Smooth latent spaces are what made "move
this slider to add a smile" demos possible, and interpolation is still a
standard sanity check for any learned code: if the walk is full of junk, the
space has holes.

**In code:** `generate` draws and decodes; `interpolate` encodes two pictures and decodes the walk between them; `largest_step` measures its biggest jump.

## The trade-off: β, blur and collapse

**Everyday picture.** Go back to the rent. Set it too low and every region
shrinks to a pin far from the centre: rebuilds are sharp, but the holes come
back. Set it too high and every region moves to the centre and swells to
the full bell curve. Then every picture's region is the same region, the
code carries no information at all, and the decoder can do nothing better
than draw the average of all the strokes: a grey smudge. That failure is
called **posterior collapse**.

**Tiny example.** The lesson trains six VAEs, with β from 0.01 to 3:

| β | rebuild error | KL (information in the code) | median sample distance to a real stroke |
|---|---|---|---|
| 0.01 | 0.16 | 10.3 | 2.40 |
| 0.1 | 0.23 | 5.7 | 0.87 |
| 0.3 | 0.31 | 4.4 | 0.56 |
| 1 | 0.92 | 2.9 | 0.81 |
| 3 | 6.08 | 0.0 | 4.73 |

(`beta_sweep` produces this table, including β = 0.03, and `demo` prints it.)

The samples are best in the middle. At β = 3 the KL is zero: the regions are
the standard bell curve itself, and the rebuild error of 6.08 is what you get
by drawing the average stroke every time.

### Why VAE pictures are blurry

Even at a good β the regions overlap, so one code can stand for several
different pictures. What should the decoder draw for it? Take a single pixel
that is black (1) in one of those pictures and white (0) in the other, equally
likely. Squared error has a clear answer:

$$
\underset{c}{\arg\min}\ \mathbb{E}\bigl[(x - c)^2\bigr] = \mathbb{E}[x]
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $x$ | the pixel's true value, which could be either picture's | 0 or 1, equally likely |
| $c$ | the decoder's guess for that pixel | 0, 0.5 or 1 |
| $(x - c)^2$ | the squared error of the guess | |
| $\mathbb{E}[\ldots]$ | **expected value**: the average over the possibilities, each weighted by its chance | |
| $\arg\min_c$ | "the $c$ that makes what follows as small as possible" | |

**In words:** "the guess with the smallest average squared error is the
average of the possibilities."

**With the numbers:** guess black (1): error 1 half the time, 0 otherwise,
average 0.5. Guess white (0): also 0.5. Guess grey (0.5): error 0.25 either
way, average **0.25**. Grey wins, and grey is the average of 0 and 1.

**In Python:**

```python
outcomes = [0.0, 1.0]
def expected_error(c):
    return sum((x - c) ** 2 for x in outcomes) / len(outcomes)
# E[(x − c)²] for three guesses: white, grey, black
[expected_error(c) for c in (0.0, 0.5, 1.0)]  # → [0.5, 0.25, 0.5]
# E[x]: the average outcome, which is the winning guess
sum(outcomes) / len(outcomes)  # → 0.5
```

Across a whole picture, "average the pictures this code might mean" is a
blur. The more the regions overlap, the more pictures each code must stand
for, and the blurrier the output.

![Left: as beta grows from 0.01 to 3, rebuild error rises, KL falls to zero, and sample distance is lowest near beta 0.3; right: 8 samples per beta, sharp but broken at 0.01, clean at 0.3, softer at 1 and uniform grey smudges at 3](figures/primer.ml.generative.autoencoders.beta_tradeoff.svg)

**Reading it:** on the left, β grows along a logarithmic axis. The blue
rebuild error stays low, then climbs steeply past β = 1. The green KL, the
information the code carries, falls steadily and hits zero at β = 3. The red
line, how far random samples are from real strokes, is a U: high at small β
(holes), lowest near β = 0.3, high again at large β (blur and collapse).
On the right are eight samples at each β. The top rows have sharp ink in
junk shapes. The middle rows are clean strokes. The β = 1 row is softer.
The bottom row is eight copies of the same grey smudge: the collapse.

**Why it matters in practice.** β is a real dial, and the name β-VAE
(Higgins and colleagues, 2017) comes from turning it up to get codes whose
numbers line up with separate factors, like direction and offset here. The
blur is the VAE's best-known weakness. Modern systems fix it by adding a
second judge of realism to the loss (an adversarial loss, the trick behind
`primer.ml.generative.gans`, as in VQGAN), or by letting a stronger generator
do the generating while the autoencoder only compresses.

**In code:** `beta_sweep` trains and measures one VAE per β; `expected_squared_error` is the grey-pixel calculation.

## Where autoencoders live today

**Everyday picture.** The autoencoder became the zip format for images and
sound that other generative models work inside.

**Tiny example.** Stable Diffusion's autoencoder turns a 512 × 512 colour
image (512 × 512 × 3 = 786,432 numbers) into a 64 × 64 × 4 grid of codes
(16,384 numbers): 48 times fewer. The diffusion model does all its work on
that small grid, and the decoder paints full-size pixels only at the very
end. Its KL weight is tiny, so it is mostly a compressor, with just enough
rent to keep the codes well-behaved.

A **VQ-VAE** (vector-quantized VAE) goes one step further: it snaps each code
vector to the nearest entry of a learned **codebook**, so an image becomes a
grid of whole numbers, the entries' positions in the codebook. Those numbers
are **tokens**, and a transformer can read and write them exactly as it
reads and writes words. Neural audio codecs do the same for sound.

```mermaid
flowchart LR
  subgraph LD["Latent diffusion"]
    direction LR
    I1["image<br/>512 × 512 × 3"] --> E1["VAE encoder"] --> L1["latent<br/>64 × 64 × 4"]
    L1 --> DF["diffusion model<br/>works here"] --> D1["VAE decoder"] --> O1["image"]
  end
  subgraph TK["Tokenizer for a transformer"]
    direction LR
    I2["image or audio"] --> E2["encoder"] --> Q["snap each vector to<br/>nearest codebook entry"]
    Q --> T["grid of token ids"] --> TR["transformer"]
    TR --> D2["decoder"] --> O2["image or audio"]
  end
```

**Reading it:** in the top row the autoencoder is the outer shell: its
encoder shrinks the image 48-fold on the way in, its decoder restores it on
the way out, and the expensive generator in the middle never touches a
pixel. Training and sampling run far faster on 16,384 numbers than on
786,432, which is what made high-resolution diffusion affordable (see
`primer.ml.generative.diffusion`). In the bottom row the snapping step turns
continuous codes into token ids, which is how images and audio can enter
and leave a language-model-style transformer (see
`primer.ml.generative.multimodal`).

**Why it matters in practice.** When a model "generates an image", there is
very often an autoencoder at both ends of the pipeline. Its quality sets a
ceiling: whatever detail the decoder cannot rebuild, no generator working in
its latent space can produce.

## In 20 seconds

- **Autoencoder:** an encoder squeezes data through a narrow code (the
  bottleneck) and a decoder rebuilds it; training minimises the rebuild
  error. With no bends it learns exactly what PCA learns; with bends it can
  follow curved data far better.
- **No generation from a plain autoencoder:** its codes land wherever
  training put them, with holes in between, so random codes decode to junk.
- **VAE:** the encoder outputs a region (μ, σ); the reparameterization trick
  z = μ + σ·ε samples from it in a way gradients can pass through; a KL
  penalty pulls every region towards the standard normal, so drawing
  z ~ N(0, I) and decoding generates new data.
- **The trade-off:** β weighs the KL. Too small brings back holes, too large
  blurs and finally collapses; squared error averages over overlapping
  possibilities, which is why VAE output is blurry.
- **Today:** VAEs compress images for latent diffusion, and VQ-VAEs turn
  images and audio into tokens for transformers.

## Self-test questions

**What is the bottleneck for? What goes wrong if the code is as wide as the
input?**
It forces the network to keep only what matters: with 2 numbers for 64
pixels, it must find the few facts that actually vary. If the code is as
wide as the input, the network can learn to copy the pixels straight through,
rebuild perfectly and learn nothing useful, unless something else (noise on
the input, a penalty on the code) stops it.

**Why can't you generate new pictures by decoding random codes from a plain
autoencoder?**
Its loss only ever sees the codes of real pictures, so it says nothing about
where codes should live or what lies between them. The codes end up in an
arbitrary range with holes between clusters, and a random code usually lands
in a hole, where the decoder's output is junk.

**What problem does the reparameterization trick solve?**
Backpropagation needs every step between the weights and the loss to be
something you can differentiate, and a random draw is not. Writing the draw
as z = μ + σ·ε with the noise ε as a separate input turns sampling into
arithmetic, so gradients reach μ and σ, and through them the encoder.

**What do the two terms of the VAE loss each want, and why do you need
both?**
The rebuild term wants regions small and far apart so every picture is
decoded precisely. The KL term wants every region to be the standard bell
curve. Without KL you get a plain autoencoder with holes; without the
rebuild term every region collapses onto the bell curve and the code carries
nothing. The balance gives a packed, smooth code space that still tells
pictures apart.

**Work out the KL penalty for one code number with μ = 0 and σ = 2.**
½ (0 + 4 − log 4 − 1) = ½ (3 − 1.386) = 0.807. A region that is too big pays
rent too, though less steeply than one that is too small.

**Why are VAE samples blurry?**
Regions overlap, so one code can stand for several pictures, and under
squared error the best single answer is their average. Averaged pictures are
blurred pictures. Raising β increases the overlap and the blur.

**What is posterior collapse?**
When the KL rent outweighs what the code saves in rebuild error, the encoder
makes every region the standard bell curve, the code carries no information,
and the decoder outputs the same average picture for every code. In this
lesson that happens at β = 3.

**Why does latent diffusion run inside an autoencoder's code space instead of
on pixels?**
The code is many times smaller (48 times for Stable Diffusion's 512 × 512
images) and keeps what matters to the eye, so the expensive, many-step
generator is far cheaper to train and run. The decoder turns the result back
into full-size pixels once, at the end.

## The papers behind this lesson

- **Hinton & Salakhutdinov, *Reducing the Dimensionality of Data with Neural
  Networks* (Science, 2006)**: https://doi.org/10.1126/science.1127647.
  Showed that deep autoencoders, once they could be trained, compress data
  far better than PCA.
- **Kingma & Welling, *Auto-Encoding Variational Bayes* (2013)**:
  https://arxiv.org/abs/1312.6114. Introduced the variational autoencoder,
  the reparameterization trick and the ELBO loss with its closed-form
  Gaussian KL.
- **Rezende, Mohamed & Wierstra, *Stochastic Backpropagation and Approximate
  Inference in Deep Generative Models* (2014)**:
  https://arxiv.org/abs/1401.4082. Developed the same idea independently at
  the same time, showing how to backpropagate through random sampling.
- **Burgess et al., *Understanding disentangling in β-VAE* (2018)**:
  https://arxiv.org/abs/1804.03599. Explains why turning up β pushes the
  code's numbers to line up with separate factors of the data, and what it
  costs in rebuild quality.
- **van den Oord, Vinyals & Kavukcuoglu, *Neural Discrete Representation
  Learning* (2017)**: https://arxiv.org/abs/1711.00937. Introduced the
  VQ-VAE, which snaps codes to a learned codebook and so turns images and
  audio into tokens.
- **Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion
  Models* (2021)**: https://arxiv.org/abs/2112.10752. Ran diffusion inside a
  lightly regularized autoencoder's code space, the design behind Stable
  Diffusion.

## Further reading

- Kingma & Welling, *An Introduction to Variational Autoencoders* (2019): https://arxiv.org/abs/1906.02691
- Carl Doersch, *Tutorial on Variational Autoencoders* (2016): https://arxiv.org/abs/1606.05908
- Goodfellow, Bengio & Courville, *Deep Learning*, chapter 14, Autoencoders: https://www.deeplearningbook.org/contents/autoencoders.html
- Esser, Rombach & Ommer, *Taming Transformers for High-Resolution Image Synthesis* (VQGAN, 2020): https://arxiv.org/abs/2012.09841
- PyTorch's VAE example: https://github.com/pytorch/examples/tree/main/vae
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from primer._show import banner, say, table, takeaway
from primer.ml.optimizers import Adam

# ---------------------------------------------------------------------------
# 1. The toy data: 8 × 8 pictures of pen strokes
# ---------------------------------------------------------------------------

SIDE = 8  # images are SIDE × SIDE pixels, flattened to 64 numbers
STROKE_KINDS = ("horizontal", "vertical", "diagonal down", "diagonal up")
_STROKE_ANGLES = (0.0, np.pi / 2, np.pi / 4, -np.pi / 4)  # radians, one per kind above
STROKE_WIDTH = 0.6  # pixels: how quickly the ink fades either side of the line

# A real stroke carries about 8.8 units of squared ink (the sum of its squared
# pixels). An image whose squared distance to every training stroke is above
# 1.0 is more than about a ninth of a stroke away from anything real: junk.
JUNK_DISTANCE = 1.0


def make_strokes(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """n pictures of one soft pen stroke each: (images (n, 64), kinds (n,), offsets (n,)).

    Two hidden facts make each picture: which of the four directions the
    stroke runs (kinds cycle 0, 1, 2, 3) and how far it sits from the centre,
    a continuous offset in [-2.5, 2.5] pixels. A good 2-number code has to
    rediscover both from the 64 pixels alone.
    """
    rng = np.random.default_rng(seed)
    kinds = np.arange(n) % len(STROKE_KINDS)
    offsets = rng.uniform(-2.5, 2.5, n)
    # Pixel centres measured from the middle of the image, so offset 0 passes through the centre.
    rows, cols = np.mgrid[0:SIDE, 0:SIDE] - (SIDE - 1) / 2
    angle = np.array(_STROKE_ANGLES)[kinds]
    # Signed distance from each pixel to the line: project the pixel onto the
    # line's perpendicular direction, then subtract how far the line is shifted.
    across_x, across_y = -np.sin(angle), np.cos(angle)
    dist = across_x[:, None, None] * cols + across_y[:, None, None] * rows - offsets[:, None, None]
    # Ink fades like a bell curve with distance: 1.0 on the line, about 0.25 one pixel away.
    images = np.exp(-(dist**2) / (2 * STROKE_WIDTH**2))
    return images.reshape(n, SIDE * SIDE), kinds, offsets


# ---------------------------------------------------------------------------
# 2. The smallest autoencoder: points on a line, a 1-number code
# ---------------------------------------------------------------------------


def worked_example_line() -> dict:
    """Squeeze 2-D points to one number and back, for points near y = 2x.

    Encoder: code = u · x, with u = (1, 2)/√5, the line's direction scaled to
    length 1. Decoder: rebuilt = code · u. A point on the line survives the
    trip exactly; a point off it comes back as its closest point on the line.
    """
    u = np.array([1.0, 2.0]) / np.sqrt(5)
    on, off = np.array([1.0, 2.0]), np.array([2.0, 3.0])
    on_code, off_code = float(u @ on), float(u @ off)
    off_rebuilt = off_code * u
    return dict(
        direction=u,
        on_line_code=on_code,
        on_line_rebuilt=on_code * u,
        off_line_code=off_code,
        off_line_rebuilt=off_rebuilt,
        off_line_error=float(np.sum((off - off_rebuilt) ** 2)),
    )


def pca_reconstruction(X: np.ndarray, k: int) -> np.ndarray:
    """Project X onto its top-k principal directions and back (see `primer.ml.embeddings.clustering.pca`).

    PCA is the best any *flat* k-number code can do under squared error.
    """
    mean = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
    return (X - mean) @ Vt[:k].T @ Vt[:k] + mean


def train_linear_autoencoder(X: np.ndarray, n_code: int = 1, steps: int = 2000, lr: float = 0.02, seed: int = 0):
    """An autoencoder with no bends: code = x W_enc, rebuilt = code W_dec. Returns (W_enc, W_dec).

    Trained by plain gradient descent on the squared error of centred data.
    With nothing nonlinear in it, the best it can do is PCA's answer: it ends
    up spanning the same directions as the top principal components.
    """
    rng = np.random.default_rng(seed)
    Xc = X - X.mean(axis=0)
    n, d = Xc.shape
    W_enc = rng.normal(0, 0.1, (d, n_code))  # (d, k)
    W_dec = rng.normal(0, 0.1, (n_code, d))  # (k, d)
    for _ in range(steps):
        Z = Xc @ W_enc  # (n, k) codes
        R = Z @ W_dec - Xc  # (n, d) what the rebuild got wrong
        # Loss = mean over points of the summed squared error; these are its exact gradients.
        g_dec = 2 / n * Z.T @ R
        g_enc = 2 / n * Xc.T @ (R @ W_dec.T)
        W_enc -= lr * g_enc
        W_dec -= lr * g_dec
    return W_enc, W_dec


# ---------------------------------------------------------------------------
# 3. A nonlinear autoencoder, and its variational cousin, with hand-written backprop
# ---------------------------------------------------------------------------


def sigmoid(z: np.ndarray) -> np.ndarray:
    # Squashes any number into (0, 1): the decoder's pixels must be valid brightnesses.
    return 1 / (1 + np.exp(-z))


class Autoencoder:
    """64 pixels -> 32 hidden -> a code of `n_code` numbers -> 32 hidden -> 64 pixels.

    The encoder and the decoder are each a two-layer network like the one in
    `primer.ml.neural_net`. Training asks only one thing: rebuild the input.
    """

    def __init__(self, n_in: int = SIDE * SIDE, n_hidden: int = 32, n_code: int = 2, seed: int = 0):
        self.n_in, self.n_hidden, self.n_code = n_in, n_hidden, n_code
        rng = np.random.default_rng(seed)
        head = self._head_width()
        # Variance 1/fan_in keeps tanh out of its flat regions at the start (see primer.ml.deep_nets).
        self.params = {
            "W1": rng.normal(0, 1 / np.sqrt(n_in), (n_in, n_hidden)),  # encoder
            "b1": np.zeros(n_hidden),
            "W2": rng.normal(0, 1 / np.sqrt(n_hidden), (n_hidden, head)),
            "b2": np.zeros(head),
            "W3": rng.normal(0, 1 / np.sqrt(n_code), (n_code, n_hidden)),  # decoder
            "b3": np.zeros(n_hidden),
            "W4": rng.normal(0, 1 / np.sqrt(n_hidden), (n_hidden, n_in)),
            "b4": np.zeros(n_in),
        }

    def _head_width(self) -> int:
        return self.n_code  # the encoder's last layer outputs the code itself

    def _encoder(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self.params
        h = np.tanh(X @ p["W1"] + p["b1"])  # (n, hidden)
        return h, h @ p["W2"] + p["b2"]  # (n, head): no squashing, a code may be any number

    def _decoder(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self.params
        g = np.tanh(Z @ p["W3"] + p["b3"])  # (n, hidden)
        return g, sigmoid(g @ p["W4"] + p["b4"])  # (n, 64) pixels in (0, 1)

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Images (n, 64) -> codes (n, n_code)."""
        return self._encoder(X)[1]

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Codes (n, n_code) -> images (n, 64). Works on any code, seen in training or not."""
        return self._decoder(Z)[1]

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(X))

    def _backward_decoder(self, Z, g, X_hat, X) -> tuple[dict, np.ndarray]:
        """Gradients of the mean summed squared error for the decoder, and the error signal at the code."""
        p, n = self.params, len(X)
        d_out = 2 * (X_hat - X) / n * X_hat * (1 - X_hat)  # through the squared error, then the sigmoid
        d_g = d_out @ p["W4"].T * (1 - g**2)  # back through W4, then tanh' = 1 - tanh²
        grads = {"W4": g.T @ d_out, "b4": d_out.sum(axis=0), "W3": Z.T @ d_g, "b3": d_g.sum(axis=0)}
        return grads, d_g @ p["W3"].T  # (n, n_code): how the loss changes with each code number

    def _backward_encoder(self, X, h, d_head) -> dict:
        p = self.params
        d_h = d_head @ p["W2"].T * (1 - h**2)
        return {"W2": h.T @ d_head, "b2": d_head.sum(axis=0), "W1": X.T @ d_h, "b1": d_h.sum(axis=0)}

    def loss_and_gradients(self, X: np.ndarray, noise: np.ndarray | None = None) -> tuple[dict, dict]:
        """Reconstruction loss (squared error summed over pixels, averaged over images) and every gradient.

        `noise` is ignored; it is accepted so `train` can drive this and `VAE` the same way.
        """
        h, Z = self._encoder(X)
        g, X_hat = self._decoder(Z)
        rec = float(np.sum((X_hat - X) ** 2) / len(X))
        grads, d_Z = self._backward_decoder(Z, g, X_hat, X)
        grads |= self._backward_encoder(X, h, d_Z)
        return {"loss": rec, "reconstruction": rec, "kl": 0.0}, grads


class VAE(Autoencoder):
    """A variational autoencoder: the encoder outputs a fuzzy region, not a point.

    For each image the encoder gives a mean μ and a log-variance log σ² per
    code number. Training draws a code from that region with the
    reparameterization trick and adds `beta` times the KL penalty that pulls
    every region towards the standard normal distribution.
    """

    def __init__(self, n_in: int = SIDE * SIDE, n_hidden: int = 32, n_code: int = 2, beta: float = 0.3, seed: int = 0):
        self.beta = beta
        super().__init__(n_in, n_hidden, n_code, seed)

    def _head_width(self) -> int:
        return 2 * self.n_code  # μ and log σ² for every code number

    def encode_distribution(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Images -> (μ, log σ²), each (n, n_code)."""
        head = self._encoder(X)[1]
        return head[:, : self.n_code], head[:, self.n_code :]

    def encode(self, X: np.ndarray) -> np.ndarray:
        """The centre μ of each image's region: the single best code to decode it from."""
        return self.encode_distribution(X)[0]

    def loss_and_gradients(self, X: np.ndarray, noise: np.ndarray | None = None) -> tuple[dict, dict]:
        """Reconstruction + β·KL, with gradients through the sampling step.

        `noise` is ε, shape (n, n_code), drawn from the standard normal by the
        caller. Passing it in keeps the step deterministic, which is what lets
        the tests check every gradient against finite differences.
        """
        n = len(X)
        noise = np.zeros((n, self.n_code)) if noise is None else noise
        h, head = self._encoder(X)
        mu, logvar = head[:, : self.n_code], head[:, self.n_code :]
        sigma = np.exp(0.5 * logvar)
        Z = reparameterize(mu, logvar, noise)  # μ + σ·ε: randomness enters only through ε
        g, X_hat = self._decoder(Z)
        rec = float(np.sum((X_hat - X) ** 2) / n)
        kl = float(np.sum(kl_to_standard_normal(mu, logvar)) / n)
        grads, d_Z = self._backward_decoder(Z, g, X_hat, X)
        # Through z = μ + σ·ε: ∂z/∂μ = 1 and ∂z/∂σ = ε, with ∂σ/∂(log σ²) = σ/2.
        # The KL term adds its own pull: ∂KL/∂μ = μ and ∂KL/∂(log σ²) = (σ² - 1)/2.
        d_mu = d_Z + self.beta * mu / n
        d_logvar = d_Z * noise * sigma / 2 + self.beta * (sigma**2 - 1) / (2 * n)
        grads |= self._backward_encoder(X, h, np.hstack([d_mu, d_logvar]))
        return {"loss": rec + self.beta * kl, "reconstruction": rec, "kl": kl}, grads


def reparameterize(mu: np.ndarray, logvar: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """z = μ + σ·ε with σ = exp(½ log σ²): a draw from N(μ, σ²) written as a plain sum.

    The encoder predicts log σ² rather than σ because a log can be any number,
    positive or negative, while σ must stay positive; exp takes care of that.
    """
    return mu + np.exp(0.5 * logvar) * eps


def kl_to_standard_normal(mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
    """KL(N(μ, σ²) ‖ N(0, 1)) summed over the last axis: ½ Σ (μ² + σ² - log σ² - 1)."""
    return 0.5 * np.sum(mu**2 + np.exp(logvar) - logvar - 1, axis=-1)


def train(model: Autoencoder, X: np.ndarray, steps: int = 1500, lr: float = 0.01, seed: int = 0) -> list[dict]:
    """Full-batch training with Adam (see `primer.ml.optimizers`). Returns the losses at every step."""
    rng = np.random.default_rng(seed)
    optimizers = {name: Adam(lr=lr) for name in model.params}
    history = []
    for _ in range(steps):
        # Fresh ε every step, so each image's code lands somewhere new in its region.
        noise = rng.standard_normal((len(X), model.n_code))
        losses, grads = model.loss_and_gradients(X, noise)
        for name in model.params:
            model.params[name] = optimizers[name].step(model.params[name], grads[name])
        history.append(losses)
    return history


@lru_cache(maxsize=None)
def trained_autoencoder(steps: int = 1500) -> Autoencoder:
    """The lesson's plain autoencoder, trained once on `make_strokes` and reused."""
    model = Autoencoder()
    train(model, make_strokes()[0], steps=steps)
    return model


@lru_cache(maxsize=None)
def trained_vae(beta: float = 0.3, steps: int = 1500) -> VAE:
    """The lesson's VAE at a given β, trained once on `make_strokes` and reused."""
    model = VAE(beta=beta)
    train(model, make_strokes()[0], steps=steps)
    return model


# ---------------------------------------------------------------------------
# 4. Measuring: rebuilds, junk, samples and walks
# ---------------------------------------------------------------------------


def reconstruction_error(model: Autoencoder, X: np.ndarray) -> float:
    """Squared error summed over the 64 pixels, averaged over images."""
    return float(np.mean(np.sum((model.reconstruct(X) - X) ** 2, axis=1)))


def distance_to_data(images: np.ndarray, X: np.ndarray) -> np.ndarray:
    """For each image, the squared distance to the nearest training image. Large means junk."""
    # ‖a - b‖² = ‖a‖² + ‖b‖² - 2 a·b, for every pair at once: (n_images, n_train).
    d = np.sum(images**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None, :] - 2 * images @ X.T
    return np.maximum(d.min(axis=1), 0.0)


def generate(model: Autoencoder, n: int, seed: int = 0) -> np.ndarray:
    """Draw n codes from the standard normal and decode them: brand-new images."""
    Z = np.random.default_rng(seed).standard_normal((n, model.n_code))
    return model.decode(Z)


def codes_in_box(codes: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """n codes drawn evenly from the smallest box that holds every given code."""
    rng = np.random.default_rng(seed)
    return rng.uniform(codes.min(axis=0), codes.max(axis=0), (n, codes.shape[1]))


def interpolate(model: Autoencoder, x_a: np.ndarray, x_b: np.ndarray, steps: int = 9) -> np.ndarray:
    """Encode two images, walk a straight line between their codes, decode every stop: (steps, 64)."""
    z_a, z_b = model.encode(np.stack([x_a, x_b]))
    t = np.linspace(0, 1, steps)[:, None]  # 0 at image a, 1 at image b
    return model.decode((1 - t) * z_a + t * z_b)


def largest_step(frames: np.ndarray) -> float:
    """The biggest change (Euclidean distance in pixels) between consecutive frames of a walk."""
    return float(np.max(np.linalg.norm(np.diff(frames, axis=0), axis=1)))


def expected_squared_error(guess: float, outcomes) -> float:
    """Average squared error of one guess against equally likely outcomes."""
    return float(np.mean([(o - guess) ** 2 for o in outcomes]))


def beta_sweep(betas=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0)) -> list[dict]:
    """Train a VAE at each β and measure rebuild quality, KL, and how real its samples look."""
    X = make_strokes()[0]
    rows = []
    for beta in betas:
        model = trained_vae(beta)
        samples = generate(model, 500, seed=5)
        rows.append(
            dict(
                beta=beta,
                reconstruction=reconstruction_error(model, X),
                kl=float(np.mean(kl_to_standard_normal(*model.encode_distribution(X)))),
                sample_distance=float(np.median(distance_to_data(samples, X))),
                sample_peak=float(samples.max(axis=1).mean()),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# 5. Figures (rendered into the HTML docs by `make figures`)
# ---------------------------------------------------------------------------

_KIND_COLOURS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea")  # one per stroke kind
_JUNK, _REAL, _MUTED = "#dc2626", "#2563eb", "#9ca3af"
_WALK = (1, 45)  # two vertical strokes, 3.1 px apart, used for the walk figure and the demo


def _tile(ax, images: np.ndarray, cols: int, title: str | None = None) -> None:
    """Draw 64-pixel images as one mosaic of 8 × 8 tiles, dark ink on white, with thin gaps."""
    rows = int(np.ceil(len(images) / cols))
    mosaic = np.full((rows * (SIDE + 1) - 1, cols * (SIDE + 1) - 1), np.nan)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        mosaic[r * (SIDE + 1) : r * (SIDE + 1) + SIDE, c * (SIDE + 1) : c * (SIDE + 1) + SIDE] = img.reshape(SIDE, SIDE)
    ax.imshow(np.ma.masked_invalid(mosaic), cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    ax.set_facecolor("#e5e7eb")  # the gaps between tiles
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)


def _pick_examples(kinds: np.ndarray, offsets: np.ndarray, per_kind: int = 2) -> list[int]:
    """Indices of a few strokes of every kind, spread across offsets, for display."""
    picks = []
    for k in range(len(STROKE_KINDS)):
        idx = np.where(kinds == k)[0]
        idx = idx[np.argsort(offsets[idx])]
        picks += [int(idx[int(q * (len(idx) - 1))]) for q in np.linspace(0.15, 0.85, per_kind)]
    return picks


def figures() -> dict:
    """Plot this lesson's data. matplotlib is imported here, and only here,
    so the lesson itself needs nothing beyond NumPy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Ellipse, Rectangle

    X, kinds, offsets = make_strokes()
    ae, vae = trained_autoencoder(), trained_vae()
    figs = {}

    # --- 1. Rebuilds: autoencoder vs PCA, both with 2 numbers ----------------
    picks = _pick_examples(kinds, offsets, per_kind=2)
    fig, axes = plt.subplots(3, 1, figsize=(7, 3.6))
    _tile(axes[0], X[picks], 8, "the original pictures (64 pixels each)")
    _tile(axes[1], ae.reconstruct(X[picks]), 8, f"autoencoder rebuild from 2 numbers (error {reconstruction_error(ae, X):.2f})")
    pca_err = float(np.mean(np.sum((pca_reconstruction(X, 2) - X) ** 2, axis=1)))
    _tile(axes[2], pca_reconstruction(X, 2)[picks], 8, f"PCA rebuild from 2 numbers (error {pca_err:.2f})")
    fig.tight_layout()
    figs["reconstructions"] = fig

    # --- 2. The autoencoder's code space: four strands ----------------------
    Z = ae.encode(X)
    fig, ax = plt.subplots(figsize=(6, 4.4))
    for k, name in enumerate(STROKE_KINDS):
        sel = kinds == k
        ax.scatter(Z[sel, 0], Z[sel, 1], s=10 + 6 * (offsets[sel] + 2.5), color=_KIND_COLOURS[k], alpha=0.8, label=name)
    ax.add_patch(Circle((0, 0), 2, fill=False, ls="--", color=_MUTED))
    ax.annotate("where a standard normal\ndraw usually lands", (1.4, -1.4), (4, -12), color="#4b5563",
                arrowprops=dict(arrowstyle="->", color=_MUTED))
    ax.set_xlabel("code number 1")
    ax.set_ylabel("code number 2")
    ax.set_title("A plain autoencoder's codes: one strand per stroke direction")
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1), markerscale=0.7)
    figs["ae_codes"] = fig

    # --- 3. Holes: random codes inside the box decode to junk ---------------
    box = codes_in_box(Z, 300, seed=6)
    dist = distance_to_data(ae.decode(box), X)
    junk = dist > JUNK_DISTANCE
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    a1.scatter(Z[:, 0], Z[:, 1], s=8, color="#111827", label="codes of real strokes")
    a1.scatter(box[~junk, 0], box[~junk, 1], s=14, marker="o", facecolors="none", edgecolors=_REAL, label="random code, decodes to a stroke")
    a1.scatter(box[junk, 0], box[junk, 1], s=14, marker="x", color=_JUNK, label="random code, decodes to junk")
    lo, hi = Z.min(axis=0), Z.max(axis=0)
    a1.add_patch(Rectangle(lo, *(hi - lo), fill=False, color=_MUTED, ls="--"))
    a1.set_title(f"{junk.mean():.0%} of random codes in the box land in a hole")
    a1.set_xlabel("code number 1")
    a1.set_ylabel("code number 2")
    a1.set_aspect("equal")
    a1.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)
    order = np.argsort(-dist)[:12].tolist() + np.argsort(dist)[:12].tolist()
    _tile(a2, ae.decode(box[order]), 6, "decoded: the 12 worst (top) and 12 best (bottom)")
    fig.tight_layout()
    figs["holes"] = fig

    # --- 4. The KL penalty for one code number -------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    mus = np.linspace(-3, 3, 200)
    a1.plot(mus, kl_to_standard_normal(mus[:, None], np.zeros((200, 1))), color=_REAL)
    a1.plot([0.5], [0.125], "o", color=_JUNK)
    a1.annotate("μ = 0.5 costs 0.125", (0.5, 0.125), (0.9, 2.2), color=_JUNK, arrowprops=dict(arrowstyle="->", color=_JUNK))
    a1.set_xlabel("mean μ (spread held at σ = 1)")
    a1.set_ylabel("KL penalty")
    a1.set_title("Rent for sitting far from the centre")
    sigmas = np.linspace(0.05, 3, 200)
    a2.plot(sigmas, kl_to_standard_normal(np.zeros((200, 1)), np.log(sigmas[:, None] ** 2)), color=_REAL)
    a2.plot([0.5], [kl_to_standard_normal(np.zeros(1), np.log(np.array([0.25])))], "o", color=_JUNK)
    a2.annotate("σ = 0.5 costs 0.318", (0.5, 0.318), (1.1, 2.2), color=_JUNK, arrowprops=dict(arrowstyle="->", color=_JUNK))
    a2.axvline(1, color=_MUTED, ls="--")
    a2.set_xlabel("spread σ (mean held at μ = 0)")
    a2.set_title("Rent for shrinking to a pin (or bloating)")
    a2.set_ylim(0, 3.5)
    fig.tight_layout()
    figs["kl_penalty"] = fig

    # --- 5. The VAE's code space: fuzzy regions packed around 0 -------------
    mu, logvar = vae.encode_distribution(X)
    sigma = np.exp(0.5 * logvar)
    fig, ax = plt.subplots(figsize=(5.4, 5))
    for i in range(len(X)):
        ax.add_patch(Ellipse(mu[i], 2 * sigma[i, 0], 2 * sigma[i, 1], color=_KIND_COLOURS[kinds[i]], alpha=0.18, lw=0))
    for k, name in enumerate(STROKE_KINDS):
        sel = kinds == k
        ax.scatter(mu[sel, 0], mu[sel, 1], s=6, color=_KIND_COLOURS[k], label=name)
    for r in (1, 2):
        ax.add_patch(Circle((0, 0), r, fill=False, ls="--", color=_MUTED))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_xlabel("code number 1 (μ)")
    ax.set_ylabel("code number 2 (μ)")
    ax.set_title("A VAE's codes: fuzzy regions packed inside the bell curve")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    figs["vae_codes"] = fig

    # --- 6. Samples from the standard normal: plain AE vs VAE ---------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, model, name in ((a1, ae, "plain autoencoder"), (a2, vae, "VAE")):
        samples = generate(model, 24, seed=5)
        d = np.median(distance_to_data(generate(model, 500, seed=5), X))
        _tile(ax, samples, 8, f"{name}: 24 codes from N(0, I)\n(median distance to a real stroke {d:.2f})")
    fig.tight_layout()
    figs["samples"] = fig

    # --- 7. The VAE's latent space as a map -----------------------------------
    grid = np.linspace(-2.2, 2.2, 11)
    codes = np.array([[a, b] for b in grid[::-1] for a in grid])
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    _tile(ax, vae.decode(codes), len(grid), "Decoding every point of a grid from -2.2 to 2.2")
    ax.set_xlabel("code number 1 →")
    ax.set_ylabel("code number 2 →")
    figs["latent_grid"] = fig

    # --- 8. A walk between two strokes ---------------------------------------
    a, b = _WALK
    fig, axes = plt.subplots(2, 1, figsize=(7, 2.6))
    for ax, model, name in ((axes[0], ae, "plain autoencoder"), (axes[1], vae, "VAE")):
        frames = interpolate(model, X[a], X[b], steps=9)
        worst = distance_to_data(frames, X).max()
        _tile(ax, frames, 9, f"{name}: the worst stop is {worst:.2f} from any real stroke")
    fig.suptitle(f"Walking in a straight line between two vertical strokes (offsets {offsets[a]:.2f} and {offsets[b]:.2f})", fontsize=10)
    fig.tight_layout()
    figs["interpolation"] = fig

    # --- 9. The β trade-off ----------------------------------------------------
    rows = beta_sweep()
    betas = [r["beta"] for r in rows]
    fig = plt.figure(figsize=(10, 4.2))
    a1 = fig.add_subplot(1, 2, 1)
    a1.plot(betas, [r["reconstruction"] for r in rows], "o-", color=_REAL, label="rebuild error")
    a1.plot(betas, [r["kl"] for r in rows], "o-", color="#16a34a", label="KL (information in the code)")
    a1.plot(betas, [r["sample_distance"] for r in rows], "o-", color=_JUNK, label="samples' distance to a real stroke")
    a1.set_xscale("log")
    a1.set_xlabel("β (weight on the KL penalty)")
    a1.set_title("Too little β: holes. Too much: blur, then collapse")
    a1.legend(frameon=False, fontsize=8)
    a2 = fig.add_subplot(1, 2, 2)
    strip = np.vstack([generate(trained_vae(beta), 8, seed=5) for beta in betas])
    _tile(a2, strip, 8, "8 samples at each β (rows, top to bottom)")
    a2.set_yticks([(SIDE + 1) * i + SIDE / 2 - 0.5 for i in range(len(betas))], [f"β = {beta:g}" for beta in betas])
    fig.tight_layout()
    figs["beta_tradeoff"] = fig

    return figs


# ---------------------------------------------------------------------------
# 6. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. A one-number code, by hand: points near the line y = 2x")
    ex = worked_example_line()
    say(
        """
        Encoder: code = u · x with u = (1, 2)/√5, the line's direction.
        Decoder: rebuild = code × u. A point on the line survives the trip;
        a point off it comes back as its closest point on the line.
        """
    )
    table(
        ["point", "code", "rebuild", "squared error"],
        [
            ("(1, 2)", ex["on_line_code"], f"({ex['on_line_rebuilt'][0]:.2f}, {ex['on_line_rebuilt'][1]:.2f})", 0.0),
            ("(2, 3)", ex["off_line_code"], f"({ex['off_line_rebuilt'][0]:.2f}, {ex['off_line_rebuilt'][1]:.2f})", ex["off_line_error"]),
        ],
        floatfmt=".3f",
    )
    takeaway("An autoencoder keeps what varies most and loses the rest; training shrinks what it loses.")

    banner("2. Squeezing 8 × 8 stroke pictures into 2 numbers")
    X, kinds, offsets = make_strokes()
    ink = float(np.mean(np.sum(X**2, axis=1)))
    ae = trained_autoencoder()
    pca_err = float(np.mean(np.sum((pca_reconstruction(X, 2) - X) ** 2, axis=1)))
    ae_err = reconstruction_error(ae, X)
    say(f"200 pictures of 64 pixels, each one stroke in one of {len(STROKE_KINDS)} directions at some offset.")
    table(
        ["2-number code", "rebuild error", "share of the picture kept"],
        [("PCA (flat)", pca_err, f"{1 - pca_err / ink:.0%}"), ("autoencoder (bent)", ae_err, f"{1 - ae_err / ink:.0%}")],
        floatfmt=".3f",
    )
    Z = ae.encode(X)
    say(f"Its codes run from {Z.min(axis=0).round(1)} to {Z.max(axis=0).round(1)}: a range nothing asked for.")
    takeaway("Strokes lie on a curved surface in pixel space; a flat PCA sheet cannot follow it, a bent network can.")

    banner("3. Holes: decoding random codes from a plain autoencoder")
    from_normal = distance_to_data(generate(ae, 500, seed=5), X)
    from_box = distance_to_data(ae.decode(codes_in_box(Z, 500, seed=6)), X)
    table(
        ["where the random codes come from", "median distance to a real stroke", "share that is junk"],
        [
            ("standard normal N(0, I)", float(np.median(from_normal)), f"{np.mean(from_normal > JUNK_DISTANCE):.0%}"),
            ("the box around its own codes", float(np.median(from_box)), f"{np.mean(from_box > JUNK_DISTANCE):.0%}"),
        ],
        floatfmt=".2f",
    )
    takeaway("A plain autoencoder's loss never looks between its codes, so the gaps decode to junk.")

    banner("4. The VAE's two new pieces: the reparameterization trick and the KL rent")
    mu, logvar, eps = np.array([0.5]), np.log(np.array([0.25])), np.array([1.2])
    say(f"μ = 0.5, σ = 0.5, ε = 1.2  ->  z = μ + σ·ε = {reparameterize(mu, logvar, eps)[0]:.2f}")
    table(
        ["region", "KL rent"],
        [
            ("μ = 0,   σ = 1    (already standard)", float(kl_to_standard_normal(np.zeros(1), np.zeros(1)))),
            ("μ = 0.5, σ = 0.5", float(kl_to_standard_normal(mu, logvar))),
            ("μ = 0.5, σ = 0.05 (nearly a pin)", float(kl_to_standard_normal(mu, np.log(np.array([0.05**2]))))),
        ],
        floatfmt=".3f",
    )
    vae = trained_vae()
    mu_all, logvar_all = vae.encode_distribution(X)
    say(
        f"""
        Trained with β = {vae.beta}, the VAE's codes centre on
        ({mu_all.mean(axis=0)[0]:.2f}, {mu_all.mean(axis=0)[1]:.2f}) with spread about
        {mu_all.std(axis=0).mean():.2f}, and each region has σ about
        {np.exp(0.5 * logvar_all).mean():.2f}: packed inside the bell curve.
        """
    )
    takeaway("Sampling becomes arithmetic the chain rule can pass through, and the rent packs the codes where we will sample.")

    banner("5. Generating: draw z from N(0, I) and decode")
    rows = []
    for model, name in ((ae, "plain autoencoder"), (vae, "VAE")):
        d = distance_to_data(generate(model, 500, seed=5), X)
        rows.append((name, float(np.median(d)), f"{np.mean(d > JUNK_DISTANCE):.0%}"))
    table(["model", "median distance to a real stroke", "share that is junk"], rows, floatfmt=".2f")
    pairs = np.random.default_rng(3).integers(0, len(X), (60, 2))
    steps = {
        name: np.mean([largest_step(interpolate(model, X[a], X[b])) for a, b in pairs])
        for model, name in ((ae, "plain autoencoder"), (vae, "VAE"))
    }
    say(f"Walking between 60 random pairs, the biggest single jump averages {steps['plain autoencoder']:.2f} (plain) vs {steps['VAE']:.2f} (VAE).")
    takeaway("The same random codes that give a plain autoencoder junk give a VAE mostly real-looking strokes.")

    banner("6. The β trade-off: holes, then sharpness, then blur and collapse")
    table(
        ["β", "rebuild error", "KL", "sample distance", "sample peak ink"],
        [(r["beta"], r["reconstruction"], r["kl"], r["sample_distance"], r["sample_peak"]) for r in beta_sweep()],
        floatfmt=".3f",
    )
    say(
        f"""
        Why blur? For a pixel that is 0 or 1 with equal chance, guessing grey
        costs {expected_squared_error(0.5, [0.0, 1.0]):.2f} on average and guessing
        either extreme costs {expected_squared_error(0.0, [0.0, 1.0]):.2f}: squared error
        rewards the average. At β = 3 the code carries nothing and every sample
        is the same grey average stroke.
        """
    )
    takeaway("β trades rebuild sharpness for a smooth, sampleable code space; latent diffusion keeps it tiny and lets diffusion generate.")


if __name__ == "__main__":
    demo()
