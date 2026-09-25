r"""
# CNNs and RNNs: how networks see images and read sequences

Run: `python -m primer.ml.cnn_rnn`

New to the notation? `primer.notation` explains every symbol used here
(Σ, ⊙, subscripts, ∂, and so on) from zero.

## The idea

Before transformers, two designs dominated deep learning, and both still
matter. **Convolutional networks (CNNs)** see images by sliding small
pattern detectors across them. **Recurrent networks (RNNs)** read sequences
one item at a time, carrying a running memory. Each builds in an assumption
about its data (patterns in images are local; sequences unfold in order),
and each has a limit that the transformer removed. Knowing both stories
explains why modern models look the way they do.

# Part A: convolutional networks

## A1. A convolution is a flashlight looking for one pattern

**Everyday picture.** You are in a dark room with a large photograph and a
small flashlight. You're looking for one thing, say a place where dark turns
to bright from left to right. You sweep the flashlight across the photo, one
step at a time, and at every spot you jot down a score: high if the lit
patch matches what you're looking for, near zero if it doesn't. When you
finish, your notes form a new, smaller picture: a map of *where* the pattern
appears. That map is called a **feature map**, and the pattern you were
looking for, written as a small grid of numbers, is the **filter** (or
kernel).

**Tiny worked example.** A 5×5 image: two dark columns (0) then three bright
columns (1), so there is a vertical edge between columns 2 and 3. The filter
is a 3×3 "vertical edge" detector: −1 on the left column, 0 in the middle,
+1 on the right. It rewards "bright on the right, dark on the left".

```
image X                 filter K
0 0 1 1 1               -1 0 1
0 0 1 1 1               -1 0 1
0 0 1 1 1               -1 0 1
0 0 1 1 1
0 0 1 1 1
```

Put the filter on the top-left 3×3 patch of the image. Multiply each image
number by the filter number on top of it, and add all nine products:

```
patch      × filter    = products
0 0 1      -1 0 1        0 0 1
0 0 1      -1 0 1        0 0 1      sum = 1 + 1 + 1 = 3
0 0 1      -1 0 1        0 0 1
```

So the top-left cell of the feature map is **3**. Slide one step right: the
patch is `0 1 1` in every row; products `0 0 1` per row; sum again **3**
(the edge is still under the flashlight). One more step: the patch is
`1 1 1`, products `−1 0 1`, sum **0** (flat bright area, no edge). Doing the
same for every row gives the full 3×3 feature map:

```
3 3 0
3 3 0
3 3 0
```

The high numbers sit exactly where the edge is. Run a *horizontal* edge
filter over the same image and every cell is 0: this image has no
top-to-bottom change. Each filter answers one question.

```mermaid
flowchart LR
  W[Take the next k×k patch<br/>of the image] --> M[Multiply each pixel by the<br/>filter weight on top of it]
  M --> S[Add all the products<br/>into one number]
  S --> C[Write it into the<br/>feature map]
  C --> N{More positions?}
  N -->|slide by the stride| W
  N -->|no| F[Feature map complete]
```

**Reading it:** this loop *is* a convolution. The only arithmetic is
multiply-and-add, repeated at every position. The stride is how far the
flashlight jumps each time (1 pixel, or 2 to shrink the output faster).
Padding adds a border of zeros so the filter can also centre on edge pixels
and the output keeps the input's size.

$$
Y[i, j] = \sum_{c}\sum_{u=0}^{k-1}\sum_{v=0}^{k-1} K[c, u, v]\; X[c,\; i\,s + u,\; j\,s + v]
\qquad
\text{out} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| $X$ | the input image; $X[c, a, b]$ is channel c (red, green or blue) at row a, column b | (channels, height, width) |
| $K$ | the filter (kernel): a small grid of learned weights, one slice per channel | (channels, k, k) |
| $Y[i, j]$ | the feature-map value at output row i, column j | one number |
| $\sum_c \sum_u \sum_v$ | add up over every channel c and every filter row u and column v | |
| $k$ | filter size (3 for a 3×3 filter) | small integer |
| $s$ | stride: how many pixels the filter jumps between positions | 1 or 2 |
| $n$ | input size along one side | e.g. 5 or 224 |
| $p$ | padding: zeros added around each side | 0, 1, 2… |
| $\lfloor \cdot \rfloor$ | floor: round down to a whole number | |

**In words:** each output number is the sum, over the filter's footprint and
all colour channels, of filter weight times the pixel under it; the output
has one number per place the filter can stand.

**On the worked example:** one channel, k = 3, s = 1, p = 0. Y[0, 0] =
(−1)·0 + 0·0 + 1·1 for each of the 3 rows = 3. Output size
⌊(5 + 0 − 3)/1⌋ + 1 = 3. For a 224-pixel image, a 7×7 filter, stride 2 and
padding 3: ⌊(224 + 6 − 7)/2⌋ + 1 = 112.

**In code:** `conv2d` is the loop in the diagram, one multiply-and-add per
position, and `conv_output_size` is the output-size formula.

## A2. Pooling: summarise each neighbourhood by its loudest voice

**Everyday picture.** A manager asks each of four teams for one number: the
strongest signal anyone on the team saw. The report is four times shorter
and still says where something important happened.

**Tiny worked example.** 2×2 max pooling on a 4×4 map keeps the largest
value in each quarter:

```
1 3 | 0 0
2 4 | 0 1        ->   4 1
----+----             1 6
0 0 | 5 2
1 0 | 1 6
```

![Input, filter, feature map and pooled map, side by side](figures/primer.ml.cnn_rnn.cnn_pipeline.svg)

**Reading it:** left to right, one pass of a CNN layer. The input is an 8×8
image of a bright square on a dark background. The filter is the vertical
edge detector from the worked example. The feature map (with padding, so
still 8×8) is bright red down the square's left side (dark to bright, the
pattern it looks for) and blue down the right side (bright to dark, the
opposite pattern); everywhere flat is zero. After 2×2 max pooling the map
is 4×4: a quarter of the numbers, but the left edge is still clearly marked.
The right edge's negative responses become 0, because max pooling keeps the
*largest* value in each block. Real networks pass maps through ReLU (which
zeroes negatives) before pooling anyway, and detect the bright-to-dark edge
with a second, mirror-image filter. Numbers in the cells are the actual
values.

**In code:** `max_pool2d` keeps the largest value in each non-overlapping
block.

## A3. Weight sharing and the receptive field

**Everyday picture.** A rubber stamp: you carve the pattern once and use it
everywhere on the page. A CNN uses the *same* filter at every position,
so a cat detector works in any corner of the photo and costs the same
number of weights however large the photo is.

**Tiny worked example.** 64 filters of size 3×3 over a colour image need
3·3·3·64 + 64 = **1,792** parameters, for any image size. A fully connected
layer mapping a 224×224×3 image to an output of the same size as those 64
feature maps would need 150,528 × 3,211,264 ≈ **483 billion** weights.

As layers stack, each neuron sees more of the original image: its
**receptive field**. One 3×3 layer sees 3×3 pixels; two see 5×5; three see
7×7. Pooling between layers speeds this up: conv 3×3, pool 2×2, conv 3×3
already sees 8×8. That's why two stacked 3×3 filters (18 weights, and two
nonlinearities) replaced single 5×5 filters (25 weights) in VGG.

$$
r_{\ell} = r_{\ell-1} + (k_\ell - 1)\, j_{\ell-1} \qquad j_\ell = j_{\ell-1}\, s_\ell \qquad r_0 = j_0 = 1
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| $\ell$ | layer number, counting from the input | 1, 2, 3… |
| $r_\ell$ | receptive field after layer ℓ: input pixels (along one side) one neuron sees | ≥ 1 |
| $k_\ell$ | kernel (filter or pool window) size of layer ℓ | |
| $s_\ell$ | stride of layer ℓ | |
| $j_\ell$ | jump: input pixels between neighbouring neurons at layer ℓ | ≥ 1 |

**In words:** every layer widens the view by (kernel − 1) steps, and a step
at depth ℓ is as many input pixels as all earlier strides multiplied
together.

**On the worked example:** conv 3 (r = 1 + 2·1 = 3, j = 1), pool 2 stride 2
(r = 3 + 1·1 = 4, j = 2), conv 3 (r = 4 + 2·2 = 8).

![Receptive field grows with depth](figures/primer.ml.cnn_rnn.receptive_field.svg)

**Reading it:** the x-axis is the number of 3×3 convolution layers; the
y-axis is how many input pixels (along one side) one neuron at that depth
can see. Without pooling (lower line) the view grows by 2 pixels per layer,
slowly. With a 2×2 pool after every second layer (upper line) the jumps
double each time, so a dozen layers already see most of a 224-pixel image.
That growth is what lets deep layers recognise whole objects.

**In code:** `conv_params` and `dense_params` count the two layers'
parameters, and `receptive_field` applies the recurrence to a stack of
(kernel, stride) layers.

## A4. From edges to parts to objects

**Everyday picture.** Reading starts with strokes, then letters, then words,
then sentences. A CNN's first layer learns strokes (edges and colour
blobs), the next combines them into textures and corners, deeper layers
into parts (eyes, wheels), and the last into whole objects.

```mermaid
flowchart LR
  I[Image pixels] --> C1[Conv layer<br/>edges]
  C1 --> P1[Pool]
  P1 --> C2[Conv layer<br/>textures, parts]
  C2 --> P2[Pool]
  P2 --> C3[Conv layer<br/>whole objects]
  C3 --> FC[Dense layer]
  FC --> O[Prediction<br/>cat 94%]
```

**Reading it:** each conv layer applies many filters to the feature maps
below it, so its patterns are combinations of the patterns below. Each pool
halves the resolution, which widens what the next layer sees (section A3).
By the end, a small grid of very abstract features feeds an ordinary dense
layer that outputs class probabilities. Nobody hand-designs these filters;
training discovers them, and first-layer filters in trained networks look
remarkably like the edge detectors in this lesson.

![Hand-made first-layer filters and a second-layer corner detector](figures/primer.ml.cnn_rnn.filter_hierarchy.svg)

**Reading it:** the top row shows four 3×3 filters of the kind a first
layer learns (red = positive weight, blue = negative): vertical edge,
horizontal edge, diagonal, and a centre-surround "spot". The middle row
shows each one's response to the same small image of an L-shaped block:
each lights up only on its own pattern. The last panel is a second-layer
detector built from first-layer outputs: the vertical-edge strength times
the horizontal-edge strength, which is large only where a vertical *and* a
horizontal edge meet, at the L's corners. Combining simple detectors into
more specific ones is the whole hierarchy in miniature.

**In code:** the figure runs each first-layer filter over the L with
`conv2d`, and the corner detector multiplies two of those feature maps
entry by entry.

## A5. Landmarks, and patches as tokens

- **AlexNet (2012)** won the ImageNet competition by a wide margin by training
  a deep CNN on GPUs, which started the deep-learning boom.
- **VGG (2014)** showed that deep stacks of small 3×3 filters work well.
- **ResNet (2015)** added residual (skip) connections, letting gradients
  bypass layers, and made networks of 100+ layers trainable. The same idea
  sits inside every transformer (see `primer.ml.deep_nets`).
- **Vision Transformers (2020)** cut an image into patches and treat each
  patch as a token, then apply ordinary attention.

**Everyday picture for patches.** Cut a photo into a grid of jigsaw pieces,
lay them out in a row, and read them like the words of a sentence.

**Tiny worked example.** A 224×224 colour image cut into 16×16 patches gives
(224/16)² = **196** patches, each flattened to 16·16·3 = **768** numbers:
196 tokens of 768 values.

```mermaid
flowchart LR
  IMG[224×224×3 image] --> CUT[Cut into 16×16 patches<br/>14 × 14 = 196 pieces]
  CUT --> FLAT[Flatten each patch<br/>768 numbers]
  FLAT --> PROJ[Linear projection<br/>to model width]
  PROJ --> POS[Add position<br/>embeddings]
  POS --> TF[Transformer blocks<br/>attention across patches]
```

**Reading it:** after the cut-and-flatten step, an image is just a sequence
of 196 tokens, and everything downstream is the same transformer used for
text. Multimodal language models read images mostly this way.

**In code:** `patchify` does the cut-and-flatten step, turning an image into
one row of numbers per patch.

**Why it matters in practice.** CNNs remain efficient and strong for small
vision tasks, on-device models and limited data, because their built-in
assumptions (locality, weight sharing) mean they learn from less. At large
scale, Vision Transformers dominate.

# Part B: recurrent networks

## B1. An RNN reads with a one-page summary

**Everyday picture.** You read a book one word at a time, and you are
allowed to keep only a single page of notes. After every word you rewrite
the page: blend what the page said with the new word. At the end, the page
is all you have. That page is the **hidden state**.

**Tiny worked example.** The sentence "not very good", with each word turned
into one number: not = −1, very = 0.5, good = 1. The summary is a single
number h, starting at 0. The rule: new h = tanh(0.5 × old h + 1 × word).
(tanh squashes any number into the range −1 to 1: tanh(0) = 0,
tanh(1) = 0.76, tanh(−1) = −0.76.)

| step | word | 0.5 × old h + word | new h = tanh(…) |
|---|---|---|---|
| 1 | not (−1) | 0.5 × 0 + (−1) = −1 | **−0.762** |
| 2 | very (0.5) | 0.5 × (−0.762) + 0.5 = 0.119 | **0.119** |
| 3 | good (1) | 0.5 × 0.119 + 1 = 1.059 | **0.785** |

The final summary is strongly positive: the "not" at the start has been
almost washed out, because the page was rewritten twice since. This is the
central weakness of RNNs, in three lines of arithmetic.

```mermaid
flowchart LR
  H0[h0 = 0] --> C1[RNN cell]
  X1[x1: not] --> C1
  C1 --> H1[h1 = −0.762]
  H1 --> C2[RNN cell<br/>same weights]
  X2[x2: very] --> C2
  C2 --> H2[h2 = 0.119]
  H2 --> C3[RNN cell<br/>same weights]
  X3[x3: good] --> C3
  C3 --> H3[h3 = 0.785<br/>final summary]
```

**Reading it:** this is one RNN "unrolled in time": the three cells are the
*same* cell with the *same* weights, drawn once per step. Each step takes the
previous summary (arrow from the left) and the next word (arrow from below)
and produces the new summary. Everything the network knows about "not" must
survive two more rewrites to reach the end.

$$
h_t = \tanh\big(W_h\, h_{t-1} + W_x\, x_t + b\big)
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| $t$ | the time step (word position) | 1, 2, 3… |
| $x_t$ | the input at step t (a word's vector) | (inputs,) |
| $h_t$ | the hidden state (the summary page) after step t | (hidden,), each entry in −1…1 |
| $h_{t-1}$ | the summary before this word | (hidden,) |
| $W_h$ | recurrent weights: how the old summary feeds the new one | (hidden, hidden) |
| $W_x$ | input weights: how the word feeds the new summary | (hidden, inputs) |
| $b$ | bias | (hidden,) |
| $\tanh$ | hyperbolic tangent: squashes each number into −1…1 | |

**In words:** the new summary is the squashed sum of the old summary times
its weights, the new word times its weights, and a bias.

**On the worked example:** one-number summary, W_h = 0.5, W_x = 1, b = 0:
h₁ = tanh(0.5·0 − 1) = −0.762; h₂ = tanh(0.5·(−0.762) + 0.5) = 0.119;
h₃ = tanh(0.5·0.119 + 1) = 0.785.

**In code:** `RNNCell` holds W_h, W_x and b; `RNNCell.step` is the formula
once, `RNNCell.run` applies it along a sequence, and `RNNCell.scalar` builds
the one-number cell of the worked example.

## B2. Why RNNs forget: the vanishing gradient

**Everyday picture.** A photocopy of a photocopy of a photocopy. Each copy
loses a little; after twenty copies the original is unreadable. Training an
RNN sends a correction signal *backwards* through every step, and each step
multiplies it by a factor. Factors below 1 fade the signal to nothing
(**vanishing gradient**); factors above 1 blow it up (**exploding
gradient**).

**Tiny worked example.** With a recurrent weight of 0.5 and zero inputs, each
step back multiplies the signal by exactly 0.5. Ten steps back: 0.5¹⁰ =
**0.00098**, about a thousandth. With a weight of 1.5 instead: 1.5¹⁰ =
**57.7**, and training diverges.

$$
\frac{\partial h_T}{\partial h_0} = \prod_{t=1}^{T} \operatorname{diag}\!\big(1 - h_t^2\big)\, W_h
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| $\partial h_T / \partial h_0$ | "how much does the final summary change if the starting summary changes a little?" (a derivative, one per pair of entries) | (hidden, hidden) |
| $\prod_{t=1}^{T}$ | multiply the factors for every step from 1 to T | |
| $1 - h_t^2$ | the slope of tanh at step t: 1 near zero, near 0 when tanh saturates | 0…1 |
| $\operatorname{diag}(\cdot)$ | a matrix with these values on the diagonal and zeros elsewhere | (hidden, hidden) |
| $W_h$ | the recurrent weights, as above | (hidden, hidden) |

**In words:** the influence of the start on the end is the product, over
every step, of tanh's slope times the recurrent weights, so it shrinks or
grows geometrically with the number of steps.

**On the worked example:** h stays 0, so every slope is 1, and the product
is 0.5 × 0.5 × … (ten times) = 0.00098.

![Gradient reaching back through time: vanilla RNN vs. LSTM](figures/primer.ml.cnn_rnn.rnn_gradient.svg)

**Reading it:** the x-axis is how many steps separate the start of the
sequence from the current step; the y-axis (log scale) is how strongly the
current memory still responds to a nudge in the starting memory. The plain
RNN's line plunges: after 20 steps the start has almost no influence, and by
50 it's around 10⁻¹⁶, which means it cannot learn anything about the start.
The LSTM's line stays near 1 across all 50 steps. That flat line is the
reason LSTMs replaced plain RNNs.

**In code:** `RNNCell.influence_of_start` multiplies out the product above
for one sequence, and `gradient_through_time` measures both lines of the
figure.

## B3. LSTM: a notebook with an eraser, a pen and a highlighter

**Everyday picture.** Instead of rewriting the whole page after every word,
keep a notebook (the **cell state**) and three tools, each controlled by a
dial from 0 to 1 that the network sets for itself at every step:

- the **eraser** (forget gate f): how much of each line to keep;
- the **pen** (input gate i): how much of the new note to write in;
- the **highlighter** (output gate o): how much of the notebook to show the
  outside world right now.

Because the notebook is *edited* rather than rewritten, information can pass
through many steps untouched: eraser off, pen off, and the line survives.

**Tiny worked example.** One-line notebook holding 0.8.

| eraser keeps f | pen writes i | new note g | highlighter o | new notebook c = f·0.8 + i·g | shown h = o·tanh(c) |
|---|---|---|---|---|---|
| 1 | 0 | – | 1 | 0.8 (kept, even after 100 steps) | 0.664 |
| 0 | 0 | – | 1 | 0 (wiped) | 0 |
| 0 | 1 | 0.5 | 1 | 0.5 (overwritten) | tanh(0.5) = 0.462 |
| 1 | 0 | – | 0 | 0.8 (kept) | 0 (hidden) |

```mermaid
flowchart LR
  CP[notebook c_prev] --> FX((× f<br/>eraser))
  FX --> ADD((+))
  G[new note g] --> IX((× i<br/>pen))
  IX --> ADD
  ADD --> C[notebook c]
  C --> T[tanh]
  T --> OX((× o<br/>highlighter))
  OX --> H[shown h]
  XH[input x and previous h] -.-> FX & IX & OX & G
```

**Reading it:** follow the top line: the old notebook is multiplied by the
eraser setting, the pen's contribution is *added*, and the result is the new
notebook. There is no squashing on that line, only a multiply and an add,
so when the eraser is near 1 the notebook (and the training signal flowing
back along it) passes through almost unchanged. The dotted arrows show that
all four dials are computed from the current input and the previous shown
state, so the network decides at each step what to forget, write and show.

$$
\begin{aligned}
f_t &= \sigma(W_f [x_t, h_{t-1}] + b_f) &\quad i_t &= \sigma(W_i [x_t, h_{t-1}] + b_i) \\
o_t &= \sigma(W_o [x_t, h_{t-1}] + b_o) &\quad g_t &= \tanh(W_g [x_t, h_{t-1}] + b_g) \\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t &\quad h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| $f_t, i_t, o_t$ | forget (eraser), input (pen) and output (highlighter) gates | (hidden,), each 0…1 |
| $g_t$ | the candidate note to write | (hidden,), −1…1 |
| $c_t$ | the cell state: the notebook | (hidden,) |
| $h_t$ | the hidden state: what the cell shows | (hidden,) |
| $[x_t, h_{t-1}]$ | the input and previous shown state, stacked into one vector | (inputs + hidden,) |
| $W_f, W_i, W_o, W_g$ | learned weights for each gate | (hidden, inputs + hidden) |
| $\sigma$ | sigmoid, 1/(1 + e⁻ᶻ): squashes to 0…1, so it works as a dial | |
| $\odot$ | multiply element by element (entry 1 with entry 1, and so on) | |

**In words:** three sigmoid dials and one candidate are computed from the
input and the previous state; the notebook keeps f of itself and adds i of
the candidate; the cell shows o of the squashed notebook.

**On the worked example:** third row: f = 0, i = 1, g = 0.5, o = 1, so
c = 0·0.8 + 1·0.5 = 0.5 and h = 1·tanh(0.5) = 0.462.

**GRUs** simplify this to two dials and no separate notebook: an *update*
gate z chooses between keeping the old state (z = 1) and taking a new
candidate (z = 0), and a *reset* gate decides how much old state feeds that
candidate: h = (1 − z) ⊙ n + z ⊙ h_prev, where n is the candidate.
Similar performance, fewer parameters.

**In code:** `LSTMCell` holds the four gates' stacked weights;
`LSTMCell.step` computes the dials and updates the notebook,
`LSTMCell.run` carries it along a sequence, and
`LSTMCell.fixed_gates` pins the dials to replay the table above;
`GRUCell.step` is the two-dial version.

## B4. Why transformers won

**Everyday picture.** An RNN is a line of people passing a note: the last
person hears about the first only through everyone in between, and nobody
can start until the person before them finishes. A transformer is a meeting
where everyone can speak to everyone directly, all at once.

**Tiny worked example.** 1,000 tokens. An RNN needs **1,000** steps one after
another, and information from token 1 reaches token 1,000 through **999**
hand-offs. A transformer layer processes all 1,000 in **1** parallel step,
and any token reaches any other in **1** hop of attention.

```mermaid
flowchart LR
  subgraph RNN["RNN: one step at a time"]
    r1[The] --> r2[cat] --> r3[sat] --> r4[down]
  end
  subgraph TF["Transformer: all at once"]
    t1[The] & t2[cat] & t3[sat] & t4[down] --> A[Attention<br/>every pair compared]
  end
```

**Reading it:** in the RNN, information about "The" must survive three
hand-offs to reach "down", and the four steps cannot run at the same time. In
the transformer, all four positions go into attention together and every
pair is compared directly. Parallel training is what made it practical to
train on vastly more data, and that is what led to modern language models.
The price is attention's cost growing with the square of the sequence length
(see `primer.ml.attention`).

**In code:** `sequential_steps` and `path_length` return the two counts in
the worked example for an RNN or a transformer.

**State-space models** such as Mamba revisit recurrence with a design that
trains in parallel and runs in time linear in sequence length. They carry a
compressed state like an RNN but avoid its training bottleneck, and some
hybrid models mix them with attention.

![The "not very good" summary, step by step](figures/primer.ml.cnn_rnn.rnn_trace.svg)

**Reading it:** the worked example as a picture. Each bar is the one-number
summary after reading a word. "not" drives it negative; "very" pulls it back
near zero; "good" pushes it strongly positive. The final summary would read
as positive sentiment, which is wrong: the negation had to survive two
rewrites and didn't. Gates (LSTM) and direct connections (attention) are the
two historical fixes.

## In 20 seconds
- A convolution slides a small filter across an image, multiplying and adding
  at each position; the output map shows where the filter's pattern appears.
- Weight sharing (one filter everywhere) and local receptive fields make CNNs
  efficient; pooling downsamples; depth builds edges → parts → objects.
- An RNN reads one step at a time, carrying a hidden state; backpropagating
  through many steps multiplies gradients until they vanish or explode.
- LSTMs add a cell state edited by forget, input and output gates, so
  information and gradients survive many steps.
- Transformers won through parallel training and one-hop paths between any
  two tokens; Vision Transformers treat image patches as tokens.

## Self-test questions

**Q: What does one number in a feature map mean?**
A: How strongly the filter's pattern matches the image patch at that
position: the sum of filter weights times the pixels under them.

**Q: Why does a CNN need far fewer parameters than a dense layer on images?**
A: Weight sharing: one small filter is reused at every position, so the
parameter count depends on filter size and number of filters, not on image
size.

**Q: What does pooling buy you?**
A: Smaller maps (less computation), a faster-growing receptive field, and
tolerance to small shifts, since the strongest response in a neighbourhood
survives wherever exactly it was.

**Q: Why did VGG use stacks of 3×3 filters instead of larger ones?**
A: Two 3×3 layers see a 5×5 region with 18 weights instead of 25, and add an
extra nonlinearity between them.

**Q: How does a Vision Transformer turn an image into tokens?**
A: It cuts the image into fixed-size patches (e.g. 16×16), flattens each
into a vector, projects it to the model width and adds a position
embedding; the patches are then processed like words.

**Q: Why do plain RNNs struggle with long-range dependencies?**
A: Backpropagation through time multiplies the gradient by the recurrent
weights and tanh slopes at every step, so it shrinks geometrically (or
explodes) and early inputs stop influencing learning.

**Q: How do LSTM gates fix that?**
A: The cell state is updated additively, c = f ⊙ c_prev + i ⊙ g, so with the
forget gate near 1 information and gradients flow through many steps almost
unchanged.

**Q: Why did transformers replace RNNs?**
A: RNNs process tokens sequentially, which prevents parallel training, and
force all history through one fixed-size state. Attention connects any two
tokens in one step and trains all positions in parallel.

**Q: What did CNNs contribute that still matters?**
A: Residual connections (from ResNet), which make very deep networks
trainable and are in every transformer block, plus the general lesson that
built-in assumptions help when data is limited.

## The papers behind this lesson

- He, Zhang, Ren & Sun, *Deep Residual Learning for Image Recognition*
  (2015): https://arxiv.org/abs/1512.03385. Introduced residual connections,
  making networks of 100+ layers trainable.
  [annotated companion](../../papers/resnet.html)
- Vaswani et al., *Attention Is All You Need* (2017):
  https://arxiv.org/abs/1706.03762. Replaced recurrence with attention,
  enabling parallel training and one-hop paths between tokens.
  [annotated companion](../../papers/attention-is-all-you-need.html)
- Krizhevsky, Sutskever & Hinton, *ImageNet Classification with Deep
  Convolutional Neural Networks* (NeurIPS 2012). Trained a deep CNN on GPUs
  and won ImageNet by a wide margin, starting the deep-learning boom.
- Simonyan & Zisserman, *Very Deep Convolutional Networks for Large-Scale
  Image Recognition* (VGG, 2014): https://arxiv.org/abs/1409.1556. Showed
  deep stacks of 3×3 filters work well.
- Dosovitskiy et al., *An Image is Worth 16x16 Words* (ViT, 2020):
  https://arxiv.org/abs/2010.11929. Applied a plain transformer to image
  patches as tokens.
- Hochreiter & Schmidhuber, *Long Short-Term Memory* (1997):
  https://doi.org/10.1162/neco.1997.9.8.1735. Introduced the gated cell
  state that carries information across long sequences.
- Cho et al., *Learning Phrase Representations using RNN Encoder-Decoder*
  (2014): https://arxiv.org/abs/1406.1078. Introduced the GRU.
- Pascanu, Mikolov & Bengio, *On the difficulty of training Recurrent Neural
  Networks* (2012): https://arxiv.org/abs/1211.5063. Analysed vanishing and
  exploding gradients and proposed gradient clipping.
- Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces* (2023): https://arxiv.org/abs/2312.00752. A recurrent-style model
  that trains in parallel and scales linearly with length.

## Further reading
- Stanford CS231n, *Convolutional Neural Networks*: https://cs231n.github.io/convolutional-networks/
- Christopher Olah, *Understanding LSTM Networks*: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- Andrej Karpathy, *The Unreasonable Effectiveness of Recurrent Neural Networks*: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
- Olah, Mordvintsev & Schubert, *Feature Visualization* (Distill): https://distill.pub/2017/feature-visualization/
"""

from __future__ import annotations

import numpy as np

from primer._show import banner, matrix, say, table, takeaway

# ---------------------------------------------------------------------------
# 1. Convolution: slide a small pattern detector across the image
# ---------------------------------------------------------------------------

# Hand-made 3×3 filters. Learned filters in a trained CNN's first layer look
# very much like these.
VERTICAL_EDGE = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)  # dark-left, bright-right
HORIZONTAL_EDGE = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=float)  # dark-top, bright-bottom


def conv_output_size(size: int, kernel: int, stride: int = 1, padding: int = 0) -> int:
    """How many positions the filter visits along one side: (size + 2·padding − kernel) / stride + 1."""
    return (size + 2 * padding - kernel) // stride + 1


def conv2d(image: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """One filter swept over an image; returns the feature map.

    Shapes: `image` is (H, W) or (C, H, W); `kernel` is (k, k) or (C, k, k)
    with the same C. At each position we multiply the k×k (×C) window by the
    kernel element by element and add everything up: one number per
    position. (Deep-learning libraries call this "convolution" although,
    strictly, it is cross-correlation: the kernel is not flipped.)
    """
    if image.ndim == 2:
        image, kernel = image[None], kernel[None]  # treat as one channel
    c, h, w = image.shape
    k = kernel.shape[-1]
    padded = np.pad(image, ((0, 0), (padding, padding), (padding, padding)))  # zeros around the border
    out_h, out_w = conv_output_size(h, k, stride, padding), conv_output_size(w, k, stride, padding)
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            window = padded[:, i * stride : i * stride + k, j * stride : j * stride + k]
            out[i, j] = np.sum(window * kernel)  # multiply-and-add: the whole operation
    return out


def max_pool2d(fmap: np.ndarray, size: int = 2) -> np.ndarray:
    """Keep the largest value in each non-overlapping size×size block."""
    h, w = fmap.shape[0] // size, fmap.shape[1] // size
    return fmap[: h * size, : w * size].reshape(h, size, w, size).max(axis=(1, 3))


def conv_params(in_channels: int, out_channels: int, kernel: int) -> int:
    """Weights plus one bias per filter. Independent of image size: that's weight sharing."""
    return kernel * kernel * in_channels * out_channels + out_channels


def dense_params(inputs: int, outputs: int) -> int:
    """A fully connected layer has a separate weight for every input-output pair."""
    return inputs * outputs


def receptive_field(layers: list[tuple[int, int]]) -> int:
    """Pixels (along one side) that one output neuron can see after a stack of layers.

    `layers` is a list of (kernel size, stride). Each layer widens the view by
    (kernel − 1) × jump, where jump is how many input pixels separate
    neighbouring neurons at that depth (strides multiply it).
    """
    r, jump = 1, 1
    for k, s in layers:
        r += (k - 1) * jump
        jump *= s
    return r


def patchify(image: np.ndarray, patch: int) -> np.ndarray:
    """Cut an (H, W, C) image into non-overlapping patch×patch squares, each flattened.

    A Vision Transformer treats each flattened patch as one token.
    Returns (number of patches, patch·patch·C), row by row from the top left.
    """
    h, w, c = image.shape
    grid = image.reshape(h // patch, patch, w // patch, patch, c).transpose(0, 2, 1, 3, 4)
    return grid.reshape(-1, patch * patch * c)


# ---------------------------------------------------------------------------
# 2. Recurrent networks: a running summary rewritten after every word
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _logit(p: float) -> float:
    # The bias that makes a sigmoid gate output p when its weights are zero (p = 0 or 1 become ±30).
    p = min(max(p, 1e-13), 1 - 1e-13)
    return float(np.log(p / (1 - p)))


class RNNCell:
    """A vanilla recurrent cell: h_t = tanh(W_h h_{t-1} + W_x x_t + b).

    Shapes: h is (hidden,), x is (inputs,), W_h is (hidden, hidden),
    W_x is (hidden, inputs). The same weights are reused at every step.
    """

    def __init__(self, W_h: np.ndarray, W_x: np.ndarray, b: np.ndarray | None = None):
        self.W_h, self.W_x = W_h, W_x
        self.b = np.zeros(W_h.shape[0]) if b is None else b

    @classmethod
    def scalar(cls, w_h: float, w_x: float) -> "RNNCell":
        """A one-number summary, small enough to trace by hand."""
        return cls(np.array([[w_h]]), np.array([[w_x]]))

    @classmethod
    def random(cls, hidden: int, inputs: int, seed: int = 0, scale: float = 1.0) -> "RNNCell":
        rng = np.random.default_rng(seed)
        return cls(rng.normal(0, scale / np.sqrt(hidden), (hidden, hidden)), rng.normal(0, 1 / np.sqrt(inputs), (hidden, inputs)))

    def step(self, h: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.tanh(self.W_h @ h + self.W_x @ x + self.b)

    def run(self, xs, h0: np.ndarray | None = None) -> list[np.ndarray]:
        """All hidden states h_1..h_T for the input sequence `xs`."""
        h = np.zeros(self.W_h.shape[0]) if h0 is None else h0
        states = []
        for x in xs:
            h = self.step(h, np.asarray(x, dtype=float))
            states.append(h)
        return states

    def influence_of_start(self, xs) -> float:
        """Size of ∂h_T / ∂h_0: how much the final summary still depends on the start.

        By the chain rule it is the product over steps of diag(1 − h_t²) · W_h
        (tanh's slope times the recurrent weights). Many factors below 1
        shrink it towards zero (vanishing); above 1 blow it up (exploding).
        """
        J = np.eye(self.W_h.shape[0])
        for h in self.run(xs):
            J = np.diag(1 - h**2) @ self.W_h @ J
        return float(np.linalg.norm(J, 2))


class LSTMCell:
    """Long short-term memory: a notebook (cell state c) with three tools.

    * forget gate f (the eraser): how much of each line of the notebook to keep.
    * input gate i (the pen): how much of the new candidate g to write.
    * output gate o (the highlighter): how much of the notebook to show as h.

        f, i, o = sigmoid(...);  g = tanh(...)       each from [x, h_prev]
        c = f ⊙ c_prev + i ⊙ g                       the notebook update
        h = o ⊙ tanh(c)                              what the cell reveals

    ⊙ means multiply element by element. W stacks the four weight blocks and
    has shape (4·hidden, inputs + hidden).
    """

    def __init__(self, W: np.ndarray, b: np.ndarray):
        self.W, self.b = W, b
        self.hidden = b.shape[0] // 4

    @classmethod
    def random(cls, hidden: int, inputs: int, seed: int = 0, forget_bias: float = 1.0) -> "LSTMCell":
        rng = np.random.default_rng(seed)
        W = rng.normal(0, 1 / np.sqrt(inputs + hidden), (4 * hidden, inputs + hidden))
        b = np.zeros(4 * hidden)
        b[:hidden] = forget_bias  # a positive forget bias starts the eraser mostly off: remember by default
        return cls(W, b)

    @classmethod
    def fixed_gates(cls, size: int, forget: float, input: float, output: float, candidate: float) -> "LSTMCell":
        """A cell whose gates ignore the input and hold fixed values, to see each tool in isolation."""
        b = np.concatenate([np.full(size, _logit(forget)), np.full(size, _logit(input)),
                            np.full(size, _logit(output)), np.full(size, np.arctanh(candidate))])
        return cls(np.zeros((4 * size, 2 * size)), b)

    def step(self, x: np.ndarray, h: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = self.W @ np.concatenate([x, h]) + self.b
        H = self.hidden
        f, i, o = _sigmoid(z[:H]), _sigmoid(z[H : 2 * H]), _sigmoid(z[2 * H : 3 * H])
        g = np.tanh(z[3 * H :])
        c = f * c + i * g  # additive update: the gradient path through c is just multiplication by f
        return o * np.tanh(c), c

    def run(self, xs, h0: np.ndarray | None = None, c0: np.ndarray | None = None, keep_all: bool = False):
        h = np.zeros(self.hidden) if h0 is None else h0
        c = np.zeros(self.hidden) if c0 is None else c0
        cs = []
        for x in xs:
            h, c = self.step(np.asarray(x, dtype=float), h, c)
            cs.append(c)
        return (h, c, cs) if keep_all else (h, c)


class GRUCell:
    """Gated recurrent unit: an LSTM simplified to two gates and no separate notebook.

        z = sigmoid(...)  update gate: keep the old state (z → 1) or take the new one (z → 0)
        r = sigmoid(...)  reset gate: how much old state feeds the candidate
        n = tanh(W_n x + r ⊙ (U_n h) + b_n)
        h = (1 − z) ⊙ n + z ⊙ h_prev          (the PyTorch convention)
    """

    def __init__(self, W: np.ndarray, U: np.ndarray, b: np.ndarray):
        self.W, self.U, self.b = W, U, b
        self.hidden = b.shape[0] // 3

    @classmethod
    def fixed_gates(cls, size: int, update: float, reset: float, candidate: float) -> "GRUCell":
        b = np.concatenate([np.full(size, _logit(update)), np.full(size, _logit(reset)), np.full(size, np.arctanh(candidate))])
        return cls(np.zeros((3 * size, size)), np.zeros((3 * size, size)), b)

    def step(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        H = self.hidden
        wx, uh = self.W @ x, self.U @ h
        z = _sigmoid(wx[:H] + uh[:H] + self.b[:H])
        r = _sigmoid(wx[H : 2 * H] + uh[H : 2 * H] + self.b[H : 2 * H])
        n = np.tanh(wx[2 * H :] + r * uh[2 * H :] + self.b[2 * H :])
        return (1 - z) * n + z * h

    def run(self, xs, h0: np.ndarray | None = None) -> np.ndarray:
        h = np.zeros(self.hidden) if h0 is None else h0
        for x in xs:
            h = self.step(np.asarray(x, dtype=float), h)
        return h


def gradient_through_time(steps: int = 50, hidden: int = 8, seed: int = 0, eps: float = 1e-6) -> tuple[list[float], list[float]]:
    """How much the state after t steps still depends on the starting memory, for t = 1..steps.

    Measured by finite differences: nudge each coordinate of the starting
    memory (h_0 for the RNN, the cell state c_0 for the LSTM), rerun, and
    see how much the later memory moves. Returns (rnn_norms, lstm_norms).
    """
    rng = np.random.default_rng(seed)
    xs = rng.standard_normal((steps, 4))
    rnn = RNNCell.random(hidden, 4, seed=seed)
    lstm = LSTMCell.random(hidden, 4, seed=seed, forget_bias=3.0)

    start = rng.normal(0, 0.1, hidden)

    # RNN: exact chain rule, J_t = diag(1 − h_t²) · W_h · J_{t−1}. (Finite differences
    # can't resolve values this small: they bottom out around 1e-10.)
    rnn_norms, J = [], np.eye(hidden)
    for h in rnn.run(xs, h0=start):
        J = np.diag(1 - h**2) @ rnn.W_h @ J
        rnn_norms.append(float(np.linalg.norm(J, 2)))

    # LSTM: nudge each coordinate of the starting cell state and watch every later cell state move.
    base = np.array(lstm.run(xs, c0=start, keep_all=True)[2])  # (steps, hidden)
    cols = []
    for k in range(hidden):
        nudged = start.copy()
        nudged[k] += eps
        cols.append((np.array(lstm.run(xs, c0=nudged, keep_all=True)[2]) - base) / eps)
    Jc = np.stack(cols, axis=-1)  # (steps, hidden, hidden): ∂c_t/∂c_0 for every t
    lstm_norms = [float(np.linalg.norm(Jc[t], 2)) for t in range(steps)]
    return rnn_norms, lstm_norms


# ---------------------------------------------------------------------------
# 3. Why transformers won
# ---------------------------------------------------------------------------


def sequential_steps(model: str, n_tokens: int) -> int:
    """Steps that must happen one after another to process n tokens (per layer)."""
    return n_tokens if model == "rnn" else 1


def path_length(model: str, n_tokens: int) -> int:
    """Hops for information to travel from the first token to the last."""
    return n_tokens - 1 if model == "rnn" else 1


# ---------------------------------------------------------------------------
# 4. Figures (rendered into docs/figures by `make figures`)
# ---------------------------------------------------------------------------

# The worked-example image: dark left two columns, bright right three.
EDGE_IMAGE = np.array([[0, 0, 1, 1, 1]] * 5, dtype=float)

# An 8×8 bright square on a dark background, for the pipeline figure.
SQUARE_IMAGE = np.zeros((8, 8))
SQUARE_IMAGE[2:6, 2:6] = 1.0

DIAGONAL_EDGE = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]], dtype=float)
SPOT = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=float) / 8

# "not very good" as one number per word, for the RNN trace.
SENTENCE = [("not", -1.0), ("very", 0.5), ("good", 1.0)]


def _l_shape() -> np.ndarray:
    img = np.zeros((10, 10))
    img[2:8, 2:4] = 1.0  # vertical bar
    img[6:8, 2:8] = 1.0  # horizontal bar: together an L with corners
    return img


def figures() -> dict:
    """Data figures for this lesson, keyed by the name used in the docstring."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = {}

    def show(ax, m, title, cmap="gray", signed=False, annotate=True):
        lim = np.abs(m).max() or 1
        ax.imshow(m, cmap="RdBu_r" if signed else cmap, vmin=-lim if signed else 0, vmax=lim)
        if annotate:
            for (i, j), v in np.ndenumerate(m):
                ax.text(j, i, f"{v:g}", ha="center", va="center", fontsize=7,
                        color="black" if signed or v > lim / 2 else "white")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # Input, filter, feature map, pooled map.
    fmap = conv2d(SQUARE_IMAGE, VERTICAL_EDGE, padding=1)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
    show(axes[0], SQUARE_IMAGE, "input image (8×8)")
    show(axes[1], VERTICAL_EDGE, "filter: vertical edge (3×3)", signed=True)
    show(axes[2], fmap, "feature map (8×8, padding 1)", signed=True)
    show(axes[3], max_pool2d(fmap, 2), "after 2×2 max pooling (4×4)", signed=True)
    fig.suptitle("One CNN layer: sweep the filter, then pool")
    figs["cnn_pipeline"] = fig

    # Filters -> responses -> a corner detector.
    img = _l_shape()
    filters = [("vertical edge", VERTICAL_EDGE), ("horizontal edge", HORIZONTAL_EDGE), ("diagonal", DIAGONAL_EDGE), ("spot", SPOT)]
    fig = plt.figure(figsize=(10, 7.2))
    for k, (name, f) in enumerate(filters):
        show(fig.add_subplot(3, 4, k + 1), f, f"layer 1: {name}", signed=True, annotate=False)
        show(fig.add_subplot(3, 4, k + 5), conv2d(img, f, padding=1), f"response to the L", signed=True, annotate=False)
    v = np.abs(conv2d(img, VERTICAL_EDGE, padding=1))
    h = np.abs(conv2d(img, HORIZONTAL_EDGE, padding=1))
    corner = v * h  # a product is large only where BOTH edge detectors fire: an AND
    show(fig.add_subplot(3, 4, 9), img, "input: an L-shaped block", annotate=False)
    show(fig.add_subplot(3, 4, 10), corner, "layer 2: corner detector\n(vertical AND horizontal)", cmap="Reds", annotate=False)
    fig.suptitle("Edges first, then combinations of edges")
    figs["filter_hierarchy"] = fig

    # Receptive field growth.
    depth = np.arange(1, 13)
    plain = [receptive_field([(3, 1)] * d) for d in depth]
    pooled = []
    for d in depth:
        layers = []
        for n in range(1, d + 1):
            layers.append((3, 1))
            if n % 2 == 0:
                layers.append((2, 2))
        pooled.append(receptive_field(layers))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(depth, plain, "o-", label="3×3 convs only")
    ax.plot(depth, pooled, "s-", label="3×3 convs, 2×2 pool after every second")
    ax.set(xlabel="number of 3×3 convolution layers", ylabel="receptive field (input pixels per side)",
           title="How much of the image one neuron sees")
    ax.legend()
    figs["receptive_field"] = fig

    # RNN trace.
    states = RNNCell.scalar(0.5, 1.0).run([[x] for _, x in SENTENCE])
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    vals = [s[0] for s in states]
    ax.bar([w for w, _ in SENTENCE], vals, color=["C3" if v < 0 else "C0" for v in vals])
    for i, v in enumerate(vals):
        ax.text(i, v + (0.05 if v >= 0 else -0.1), f"{v:.3f}", ha="center")
    ax.axhline(0, color="black", lw=0.8)
    ax.set(ylim=(-1, 1), xlabel="word just read", ylabel="hidden state h (the summary)",
           title='Reading "not very good": the "not" fades')
    figs["rnn_trace"] = fig

    # Gradient through time.
    rnn, lstm = gradient_through_time(steps=50)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    t = np.arange(1, 51)
    ax.semilogy(t, rnn, lw=2, label="vanilla RNN: ∂h_t / ∂h_0")
    ax.semilogy(t, lstm, lw=2, label="LSTM: ∂c_t / ∂c_0 (forget bias 3)")
    ax.set(xlabel="steps between the start and now", ylabel="gradient size (log scale)",
           title="How much the start still matters", ylim=(1e-18, 10))
    ax.legend()
    figs["rnn_gradient"] = fig

    for f in figs.values():
        f.tight_layout()
    return figs


# ---------------------------------------------------------------------------
# 5. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. A convolution by hand: vertical edge filter on a 5×5 image")
    matrix("image (dark left, bright right)", EDGE_IMAGE, precision=0)
    matrix("filter", VERTICAL_EDGE, precision=0)
    patch = EDGE_IMAGE[:3, :3]
    say(f"Top-left patch rows are {patch[0].tolist()}; times the filter row [-1, 0, 1] gives 1 per row, 3 in total.")
    matrix("feature map", conv2d(EDGE_IMAGE, VERTICAL_EDGE), precision=0)
    matrix("horizontal-edge filter on the same image", conv2d(EDGE_IMAGE, HORIZONTAL_EDGE), precision=0)
    takeaway("A filter scores every position for one pattern; the map shows where the pattern is.")

    banner("2. Pooling, weight sharing, receptive field, patches")
    fmap = np.array([[1, 3, 0, 0], [2, 4, 0, 1], [0, 0, 5, 2], [1, 0, 1, 6]], dtype=float)
    matrix("2×2 max pool of a 4×4 map", max_pool2d(fmap), precision=0)
    table(
        ["layer", "parameters"],
        [("64 conv filters, 3×3, colour input", f"{conv_params(3, 64, 3):,}"),
         ("dense layer, same input and output size", f"{dense_params(224 * 224 * 3, 224 * 224 * 64):,}")],
    )
    say(f"Receptive field: 1 conv {receptive_field([(3, 1)])}, 2 convs {receptive_field([(3, 1)] * 2)}, conv-pool-conv {receptive_field([(3, 1), (2, 2), (3, 1)])} pixels.")
    say(f"A 224×224×3 image in 16-pixel patches becomes {patchify(np.zeros((224, 224, 3)), 16).shape} (tokens, values per token).")

    banner('3. An RNN reads "not very good"')
    states = RNNCell.scalar(0.5, 1.0).run([[x] for _, x in SENTENCE])
    h_prev = 0.0
    rows = []
    for (word, x), h in zip(SENTENCE, states):
        rows.append((word, x, f"0.5 × {h_prev:.3f} + {x}", 0.5 * h_prev + x, h[0]))
        h_prev = h[0]
    table(["word", "x", "calculation", "pre-tanh", "new h"], rows, floatfmt=".3f")
    takeaway("The summary is rewritten every word, so the opening 'not' is nearly gone by the end.")

    banner("4. Vanishing and exploding gradients")
    for w in (0.5, 1.0, 1.5):
        print(f"recurrent weight {w}: influence 10 steps back = {RNNCell.scalar(w, 1.0).influence_of_start([[0.0]] * 10):.5g}")
    print()
    rnn, lstm = gradient_through_time()
    table(["steps back", "vanilla RNN", "LSTM"], [(t, f"{rnn[t - 1]:.1e}", f"{lstm[t - 1]:.2f}") for t in (1, 10, 20, 30, 50)])

    banner("5. LSTM gates as notebook tools")
    rows = []
    for f, i, g, o, label in [(1, 0, 0.0, 1, "keep"), (0, 0, 0.0, 1, "erase"), (0, 1, 0.5, 1, "overwrite"), (1, 0, 0.0, 0, "hide")]:
        h, c = LSTMCell.fixed_gates(1, forget=f, input=i, output=o, candidate=g).run([[0.0]], c0=np.array([0.8]))
        rows.append((label, f, i, g, o, c[0], h[0]))
    table(["action", "eraser f", "pen i", "note g", "highlighter o", "notebook c", "shown h"], rows, floatfmt=".3f")
    takeaway("The notebook is edited, not rewritten, so information survives many steps.")

    banner("6. Why transformers won")
    table(
        ["", "sequential steps for 1,000 tokens", "hops from first to last token"],
        [("RNN", sequential_steps("rnn", 1000), path_length("rnn", 1000)),
         ("transformer layer", sequential_steps("transformer", 1000), path_length("transformer", 1000))],
    )
    takeaway("Parallel training and one-hop paths between any two tokens.")


if __name__ == "__main__":
    demo()
