r"""
# Multimodal models: images, audio and video into a language model

Run: `python -m primer.ml.generative.multimodal`

## The everyday picture

Imagine a brilliant reader who can take in only one thing: a long row of
index cards, each card holding a short list of numbers. That is a language
model. Every word it reads arrives as one card, the word's vector (see
`primer.ml.tokenization` and `primer.ml.transformer`). It has never seen a
photograph or heard a voice.

To tell this reader about a photo, you cut the photo into small squares and
write one card per square. To tell it about a voice recording, you write one
card for every fiftieth of a second of sound. If the cards are written in the
same "handwriting" as the word cards, the reader handles a photo the way it
handles a sentence: it pays attention across all the cards at once.

That is the whole idea of a **multimodal model**, a model that takes in more
than one **modality** (kind of input: text, images, audio, video):

> Turn every modality into a sequence of vectors ("tokens") that a transformer
> can attend over.

Everything in this lesson is a way of making those cards: for images, for
sound, for video, and, run backwards, for a model that *produces* pictures
and speech.

```mermaid
flowchart LR
  IMG["Image"] --> VE["Vision encoder<br/>(patches to vectors)"] --> P1["Projector"]
  AUD["Audio"] --> SP["Spectrogram"] --> AE["Audio encoder"] --> P2["Projector"]
  TXT["Text"] --> TOK["Tokenizer"] --> EMB["Embedding table"]
  P1 --> SEQ["One sequence of vectors,<br/>all the same width"]
  P2 --> SEQ
  EMB --> SEQ
  SEQ --> LM["Decoder language model"] --> OUT["Next token:<br/>a word, or a codebook id<br/>for a picture or a sound"]
```

**Reading it:** three roads, one destination. Each modality has its own
front end (top three rows), and every front end ends in the same place: a
list of vectors as wide as the language model's word vectors. From the box
"One sequence of vectors" onwards there is no difference between a word, a
patch of an image and a slice of sound; the language model attends over all
of them together. The last box shows the trick for *output*: if pictures and
sounds can be written as numbered tokens, the model can generate them the
way it generates words (see "Generating images and speech" below).

## A tiny worked example: a 4×4 picture becomes 4 tokens

Take a grayscale picture of 4×4 pixels, each pixel a brightness from 0 to 15:

```text
 0  1 |  2  3
 4  5 |  6  7
------+------
 8  9 | 10 11
12 13 | 14 15
```

1. **Cut** it into 2×2 **patches** (the lines above): 4 patches.
2. **Flatten** each patch into a list, reading row by row: the top-right
   patch becomes (2, 3, 6, 7).
3. **Project** each list to the model's width, here 2 numbers, by
   multiplying by a matrix $W_E$ whose first column adds up the patch's top
   row and whose second column adds up its bottom row.
4. **Add a position**: the patch's (row, column) in the grid, so the model
   knows where each patch came from.

| Patch        | Pixels          | Projected | + position | Token        |
|--------------|-----------------|-----------|------------|--------------|
| top left     | 0, 1, 4, 5      | (1, 9)    | (0, 0)     | **(1, 9)**   |
| top right    | 2, 3, 6, 7      | (5, 13)   | (0, 1)     | **(5, 14)**  |
| bottom left  | 8, 9, 12, 13    | (17, 25)  | (1, 0)     | **(18, 25)** |
| bottom right | 10, 11, 14, 15  | (21, 29)  | (1, 1)     | **(22, 30)** |

The picture is now four tokens of two numbers each: exactly the shape of
input a transformer reads. A real model does the same with bigger numbers:
patches of 14 or 16 pixels, and hundreds or thousands of numbers per token.

**In code:** `worked_example_patches` runs these four steps on this picture and returns the patches, the projections and the tokens.

## Images: a Vision Transformer from scratch

`primer.ml.cnn_rnn` introduced the idea of patches as tokens. Here we build
the whole front end, the **Vision Transformer** (ViT), and count what it
costs.

### Cutting and counting

**Everyday picture.** Lay a sheet of graph paper over a photo and cut along
every fourth line. You get a grid of small tiles, and you can hand them over
one by one, left to right and top to bottom, like the words of a sentence.

**Tiny example.** A 224×224 image in 16-pixel patches is a grid of
224 / 16 = 14 patches down and 14 across: 14 × 14 = **196** tokens. A
336×336 image in 14-pixel patches is 24 × 24 = **576** tokens.

$$
N = \frac{H}{P} \cdot \frac{W}{P}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $H, W$ | the image's height and width, in pixels | 224, 224 |
| $P$ | the side of one square patch, in pixels | 16 |
| $H/P$ | how many patches fit down the image (must be a whole number, so images are resized first) | 14 |
| $W/P$ | how many patches fit across | 14 |
| $\cdot$ | multiply | |
| $N$ | the number of tokens the image becomes | 196 |

Each token starts life as $P \cdot P \cdot C$ numbers, where $C$ is the
number of colour channels (3 for red, green, blue): 16 · 16 · 3 = 768.

**In words:** "the number of image tokens is the number of patches down
times the number of patches across."

**With the numbers:** (224 / 16) · (224 / 16) = 14 · 14 = 196; (336 / 14) ·
(336 / 14) = 24 · 24 = 576; the worked 4×4 picture in 2-pixel patches gives
2 · 2 = 4.

**In Python:**

```python
H, W, P = 224, 224, 16
# H/P patches down, times W/P across
(H // P) * (W // P)  # → 196
H, W, P = 336, 336, 14
(H // P) * (W // P)  # → 576
# the worked 4×4 picture in 2-pixel patches
(4 // 2) * (4 // 2)  # → 4
```

![A 32 by 32 picture of a sun over a striped field, cut into 16 numbered 8-pixel patches, and the 16-row matrix those patches become](figures/primer.ml.generative.multimodal.patches.svg)

**Reading it:** on the left, the orange lines cut a 32×32 picture into a
4 × 4 grid, numbered in reading order. On the right, each numbered patch has
become one row: its 64 pixels laid end to end. Rows 0 to 7 (the sky, with
the sun in patches 2, 3, 6 and 7) are mostly one grey; rows 8 to 15 (the
striped field) repeat the same light and dark pattern. Nothing about the
picture is lost, but it is now a list of 16 tokens, the same shape as a
16-word sentence.

![Tokens per image rise with the square of the side length: 576 at 336 pixels, over 9,000 at 1,344 pixels with 14-pixel patches, and a quarter of that when 2 by 2 neighbours are merged](figures/primer.ml.generative.multimodal.image_tokens.svg)

**Reading it:** the x-axis is the side length of a square image, the y-axis
the tokens it becomes. Each curve bends upward because the count grows with
the *square* of the side: double the side and the tokens quadruple. Bigger
patches (blue) give fewer tokens than smaller ones (red). The green curve
merges each 2×2 block of neighbouring patch vectors into one token before
the language model sees them, a common trick that cuts the count by four.

**In code:** `image_to_patches` cuts and flattens an image (the cutting is `primer.ml.cnn_rnn.patchify`), refusing sizes the patch does not divide, and `count_image_tokens` is the formula above, with an optional merge.

**Why it matters in practice.** Resolution is a cost dial. Small text in a
screenshot needs a high resolution to be legible, and every doubling of the
side costs four times the tokens (and attention cost grows faster still; see
`primer.ml.attention`). Systems resize images to a supported size, cut very
large ones into tiles, and merge neighbouring patches to keep the count
manageable.

### From patch to token: projection and position

**Everyday picture.** Every tile gets the same questionnaire: "how bright is
your top half? your bottom half? is there an edge?" The answers become the
tile's card. Then a sticker with the tile's grid address goes on the card,
so shuffling the cards loses nothing.

**Tiny example.** In the worked example, the questionnaire had two
questions (top-row total, bottom-row total), and the sticker was the patch's
(row, column).

$$
z_i = x_i W_E + p_i
$$

**Symbols**

| Symbol | Meaning here | Shape | In the example (top-right patch) |
|---|---|---|---|
| $i$ | which patch, counting in reading order from 0 | | 1 |
| $x_i$ | patch $i$ flattened into a row of pixels | $1 \times P^2 C$ | (2, 3, 6, 7) |
| $W_E$ | the learned **patch-embedding** matrix: one column per output number (a bias vector is usually added too; it is left out here) | $P^2 C \times d$ | the 4 × 2 matrix below |
| $x_i W_E$ | a **matrix multiply**: each output number is the dot product of the patch with one column of $W_E$ | $1 \times d$ | (5, 13) |
| $p_i$ | the learned position vector for slot $i$ | $1 \times d$ | (0, 1) |
| $z_i$ | the finished token for patch $i$ | $1 \times d$ | (5, 14) |
| $d$ | the model width: how many numbers per token | | 2 |

$W_E$ in the example is the rows (1, 0), (1, 0), (0, 1), (0, 1): the first
two pixels (the patch's top row) feed output 1, the last two (its bottom
row) feed output 2.

**In words:** "each token is its patch multiplied by a learned matrix, plus
a learned vector that marks where the patch sits."

**With the numbers:** (2, 3, 6, 7) · $W_E$ = (2 + 3, 6 + 7) = (5, 13); adding
the position (0, 1) gives (5, 14).

**In Python:**

```python
x_i = [2, 3, 6, 7]
W_E = [[1, 0], [1, 0], [0, 1], [0, 1]]
p_i = [0, 1]
# x_i W_E: dot the patch with each column of W_E
xW = [sum(x * W_E[r][c] for r, x in enumerate(x_i)) for c in range(2)]
xW  # → [5, 13]
# + p_i
[a + b for a, b in zip(xW, p_i)]  # → [5, 14]
```

Why the position vector? Attention compares every token with every other
and ignores their order (see `primer.ml.positional`). Without $p_i$, a sky
patch at the top and the same sky colour in a puddle at the bottom would be
the same token, and "the sun is above the field" could not be expressed.

```mermaid
flowchart LR
  IMG["Image<br/>H × W × C"] --> CUT["Cut into N patches<br/>N × P·P·C"]
  CUT --> PROJ["Multiply by W_E<br/>N × d"]
  PROJ --> POS["Add position vectors<br/>N × d"]
  POS --> ENC["Encoder blocks<br/>every patch attends to every patch<br/>(no causal mask)"]
  ENC --> OUT["N image vectors<br/>N × d"]
```

**Reading it:** follow the shapes under each box. Only the first two boxes
are new; from "Encoder blocks" on, this is the transformer block of
`primer.ml.transformer`, run with the causal mask switched off. A sentence
is read left to right, so a text decoder hides the future; a picture has no
future, so every patch may look at every other, above, below and to either
side. The output has one vector per patch, now informed by the whole image.

**In code:** `VisionEncoder` holds W_E, the position table and a stack of `primer.ml.transformer.TransformerBlock` with `causal=False`; `VisionEncoder.embed` is the formula above, and calling the encoder runs the blocks.

**Why it matters in practice.** The vision encoder inside most
vision-language models is a ViT that was first trained as the image half of
CLIP (see `primer.ml.embeddings.contrastive`). CLIP training pulled its
image vectors towards the vectors of matching captions, so its outputs
already carry meaning a language model can use. That is why builders start
from it instead of training vision from scratch.

## Connecting vision to a language model

### The projector: a plug adapter

**Everyday picture.** Your laptop charger has the right voltage but the
wrong plug for the wall socket abroad. A small adapter changes the shape,
not the power. The vision encoder speaks in vectors of its own width and
style; the language model expects vectors shaped like its word embeddings.
A **projector** is the adapter between them.

**Tiny example.** A vision vector with 2 numbers, (1, 2), must become a
language-model vector with 3 numbers.

$$
h = v \, W_P
$$

**Symbols**

| Symbol | Meaning here | Shape | In the example |
|---|---|---|---|
| $v$ | one image token from the vision encoder | $1 \times d_{\text{vision}}$ | (1, 2) |
| $W_P$ | the projector's learned matrix | $d_{\text{vision}} \times d_{\text{LM}}$ | rows (1, 0, 1) and (0, 1, 1) |
| $h$ | the same token, now shaped like a word embedding | $1 \times d_{\text{LM}}$ | (1, 2, 3) |
| $d_{\text{vision}}, d_{\text{LM}}$ | the widths of the two models | | 2 and 3 |

**In words:** "multiply each image vector by one learned matrix to turn it
into a vector the language model can read."

**With the numbers:** (1, 2) · $W_P$ = (1·1 + 2·0, 1·0 + 2·1, 1·1 + 2·1) =
(1, 2, 3).

**In Python:**

```python
v = [1, 2]
W_P = [[1, 0, 1], [0, 1, 1]]
# v W_P: dot v with each column of W_P
[sum(v[r] * W_P[r][c] for r in range(2)) for c in range(3)]  # → [1, 2, 3]
```

Many models use two such layers with a **GELU** between them (a smooth
"keep the positives" function; see `primer.ml.transformer`), which is a
small feed-forward network. Either way, the projector is tiny next to the
two models it joins: a few million numbers between models with billions.

**In code:** `Projector` is a single linear layer by default and a two-layer MLP when given a hidden width; `worked_example_projection` computes the (1, 2, 3) above.

### Splicing image tokens into the prompt

**Everyday picture.** Writing a letter and taping a strip of photos into the
middle of a sentence. The reader reads the words, then the photos, then the
rest of the words, in order.

**Tiny example.** The prompt "what is this `<image>`" has four text tokens,
one of them a placeholder. The placeholder is replaced by the image's 4
projected vectors, so the language model reads 3 + 4 = **7** rows.

| Step | Shape (toy model) | Meaning |
|---|---|---|
| image | (8, 8) | an 8×8 grayscale picture |
| patches | (4, 16) | four 4×4 patches, flattened |
| vision encoder output | (4, 16) | four image vectors of width 16 |
| projector output | (4, 24) | four image tokens, as wide as the language model |
| text token vectors | (3, 24) | "what", "is", "this" from the embedding table |
| spliced sequence | (7, 24) | text, then image, in prompt order |
| logits | (7, 12) | a score for each of the 12 vocabulary words, at every position |

```mermaid
flowchart LR
  IMG["Image"] --> VE["Vision encoder<br/>4 × 16"] --> PR["Projector<br/>4 × 24"]
  TXT["Prompt: what is this,<br/>then the image placeholder"] --> TE["Look up text tokens<br/>3 × 24"]
  PR --> SPL["Splice at the placeholder<br/>7 × 24"]
  TE --> SPL
  SPL --> POS["+ positions"] --> DEC["Decoder blocks<br/>(causal)"] --> LOG["Logits<br/>7 × 12"]
```

**Reading it:** two front ends meet in the "Splice" box, where the
placeholder's single row is replaced by the four projected image rows.
After that the language model runs completely unchanged: positions are
added (image tokens take up positions too), the causal decoder blocks run,
and every row produces scores over the vocabulary. The language model never
learns that some rows came from a picture.

Because the decoder is causal, a token can use the image only if it comes
*after* the image. Text before the picture is scored exactly as if there
were no picture. That is why prompts usually put the image first and the
question after it.

**In code:** `VisionLanguageModel.splice` builds the spliced sequence and a flag for each row saying whether it is an image token; calling `VisionLanguageModel` runs `primer.ml.transformer.TinyGPT`'s blocks on it; `build_toy_vlm` wires up the toy model in the table.

**Why it matters in practice.** This "encode, project, splice" design (as in
LLaVA) is the simplest and most common. Two other families exist. One adds
new cross-attention layers inside the language model that look at the image
vectors (as in Flamingo), leaving the text sequence short. Another first
compresses any image into a fixed small number of tokens, such as 32 or 64,
with a little attention module of learned queries (as in BLIP-2's
Q-Former). Both trade some detail for fewer tokens.

### How such a model is trained

**Everyday picture.** Two experts who don't share a language: a
photographer and a writer. You hire an interpreter. First, the interpreter
learns vocabulary while both experts carry on exactly as they are: the
photographer points at pictures, the writer names them. Then all three
practise real conversations together, and the writer is allowed to adapt a
little too.

That is the usual two-stage recipe:

1. **Alignment.** Freeze the vision encoder and the language model. Train
   only the projector on image-caption pairs, so the projected image tokens
   make the language model produce the caption.
2. **Visual instruction tuning.** Train the projector and the language model
   (fully, or with LoRA; see `primer.ml.training_stages`) on images paired
   with instructions and answers: "what is unusual about this picture?"
   followed by a good answer.

In both stages the loss is ordinary next-token **cross-entropy** (the
negative log of the probability given to the right token; see
`primer.ml.losses`), counted only on the answer tokens. The model is not
graded on predicting the image tokens or the question it was given.

**Tiny example.** The answer is "horizontal stripes", two tokens. The model
gives the right first token probability 0.5 and the right second token 0.25.

$$
L = -\frac{1}{|A|} \sum_{t \in A} \log p\,(y_t \mid \text{image}, y_{<t})
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $A$ | the positions of the answer tokens (the **loss mask** keeps only these) | 2 positions |
| $\lvert A \rvert$ | how many answer tokens there are | 2 |
| $t \in A$ | "for each position $t$ in the answer" | |
| $y_t$ | the correct token at position $t$ | "horizontal", then "stripes" |
| $y_{<t}$ | every token before position $t$ | |
| $p(y_t \mid \ldots)$ | the probability the model gives the correct token, given ($\mid$) the image and what came before | 0.5, 0.25 |
| $\log$ | the natural logarithm; $-\log p$ is 0 when $p = 1$ and grows as $p$ shrinks (see `primer.notation`) | $-\log 0.5 = 0.693$ |
| $L$ | the loss: the average surprise over the answer | 1.040 |

**In words:** "the loss is the average, over the answer tokens only, of how
surprised the model was by each correct token."

**With the numbers:** −(log 0.5 + log 0.25) / 2 = (0.693 + 1.386) / 2 =
**1.040**.

**In Python:**

```python
import math
# probabilities of the right answer tokens
p = [0.5, 0.25]
# −(1/|A|) Σ log p
round(-sum(math.log(q) for q in p) / len(p), 3)  # → 1.04
```

```mermaid
flowchart TB
  subgraph S1["Stage 1: alignment (image-caption pairs)"]
    direction LR
    V1["Vision encoder<br/>frozen"] --> P1["Projector<br/>TRAINED"] --> L1["Language model<br/>frozen"]
  end
  subgraph S2["Stage 2: visual instruction tuning (image, question, answer)"]
    direction LR
    V2["Vision encoder<br/>frozen"] --> P2["Projector<br/>trained"] --> L2["Language model<br/>trained or LoRA"]
  end
  S1 --> S2
```

**Reading it:** the same three boxes appear in both stages; what changes is
which ones learn. In stage 1 only the small middle box moves, so it is cheap
and it cannot damage what the two big models already know. In stage 2 the
language model joins in, so it learns to *use* the image tokens to follow
instructions, answer questions and describe details, not just name things.
The vision encoder often stays frozen throughout.

Our toy does stage 1. Its vision encoder and language model are random and
frozen; only the projector learns, from 80 small pictures of four patterns
(horizontal, vertical, diagonal, checkered), to make the language model's
output layer pick the right pattern word.

![Stage-1 alignment: the caption loss falls from 2.41 to 0.33, and on 80 new images the right word ranks first 99% of the time, up from 24%](figures/primer.ml.generative.multimodal.alignment.svg)

**Reading it:** on the left, the loss starts near the dashed line, the loss
of a blind guess among 12 words (log 12 = 2.48), and falls steadily. On the
right, the green line is the share of training images whose correct word
scores highest, and the red dots are the same measure on 80 images the
projector never saw: 24% before training (chance is 25%) and 99% after.
Nothing but the projector changed, which is the point of stage 1: a small
adapter is enough to make an existing vision encoder and an existing
language model understand each other.

**In code:** `align_projector` trains only the projector's weights with cross-entropy on the caption word and returns the loss and accuracy per step; `caption_accuracy` measures how often the right word wins. The toy scores a pooled image vector directly against the language model's token table (the last step of the real path) rather than back-propagating through every layer.

**Why it matters in practice.** Stage 1 is cheap enough to run in hours,
because almost every weight is frozen. Stage 2 decides the model's
behaviour: the quality and variety of the image-instruction data matter
more than its size. A common failure of a weakly aligned model is
describing objects that are not in the picture, a visual form of
hallucination.

## Audio: from a waveform to tokens

### Sound is a list of numbers

**Everyday picture.** A microphone is a tiny eardrum. It measures air
pressure many thousands of times a second, and the recording is just that
list of measurements: the **waveform**. Drawn on paper, it looks like a
seismograph trace.

**Tiny example.** Speech is usually recorded at a **sample rate** of 16,000
measurements per second, so one second is 16,000 numbers and a 30-second
clip is 480,000. Used directly as tokens that would be ruinous, and the
thing that matters, which pitches are sounding, is hidden in the wiggles
(top panel of the spectrogram figure below).

### How much of each pitch: the Fourier transform

**Everyday picture.** A prism splits white light into its colours. The
**Fourier transform** splits a sound into its pitches. It works by holding
a pure wave of each pitch up against the recording and asking "how well do
you line up?". A pitch that is present lines up again and again and scores
high; a pitch that is absent lines up as often as it clashes and scores
zero.

**Tiny example.** Eight samples of a wave that goes up and down twice:
(1, 0, −1, 0, 1, 0, −1, 0). Line it up against a wave that also cycles
twice, cos: (1, 0, −1, 0, 1, 0, −1, 0). Multiply matching positions and add:
1 + 0 + 1 + 0 + 1 + 0 + 1 + 0 = 4. A wave that cycles once agrees for half
its length and disagrees for the other half: it scores 0.

$$
\lvert X_k \rvert = \sqrt{\left(\sum_{n=0}^{N-1} x_n \cos\frac{2\pi k n}{N}\right)^2 + \left(\sum_{n=0}^{N-1} x_n \sin\frac{2\pi k n}{N}\right)^2}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $x_n$ | the $n$-th sample of the recording | (1, 0, −1, 0, 1, 0, −1, 0) |
| $N$ | how many samples | 8 |
| $n$ | a counter over the samples, from 0 to $N-1$ | 0, 1, …, 7 |
| $k$ | which pitch we are testing: a wave that completes $k$ cycles in the $N$ samples | 2 |
| $\cos, \sin$ | the cosine and sine waves (sine is the same wave shifted a quarter cycle, so a pitch that starts at a different moment is still caught) | |
| $2\pi$ | one full cycle, in the units cos and sin use | |
| $\sum_{n=0}^{N-1}$ | add up over every sample | |
| $\sqrt{a^2 + b^2}$ | the length of the pair (cosine score, sine score) | $\sqrt{4^2 + 0^2} = 4$ |
| $\lvert X_k \rvert$ | how much of pitch $k$ the recording holds | 4 |

Bin $k$ is the frequency $f_k = k \cdot f_s / N$ in **hertz** (cycles per
second), where $f_s$ is the sample rate. If the 8 samples were taken over
one second, $f_s = 8$ and bin 2 is 2 Hz. Textbooks write the same thing
with complex numbers, $X_k = \sum_n x_n e^{-2\pi i k n / N}$; the cosine
part and the sine part are exactly the two sums above.

**In words:** "for each pitch, correlate the recording with a cosine and a
sine of that pitch, and take the length of the two scores."

**With the numbers:** for $k = 2$ the cosine sum is 4 and the sine sum is 0,
so $\lvert X_2 \rvert = 4$. For $k = 1$ both sums are 0. Bin 6 also scores 4:
with only 8 samples, a wave cycling 6 times is indistinguishable from one
cycling 8 − 6 = 2 times, so the top half of the bins mirrors the bottom half
and only bins 0 to $N/2$ are kept.

**In Python:**

```python
import math
x = [1, 0, -1, 0, 1, 0, -1, 0]
N = len(x)
def magnitude(k):
    c = sum(x_n * math.cos(2 * math.pi * k * n / N) for n, x_n in enumerate(x))
    s = sum(x_n * math.sin(2 * math.pi * k * n / N) for n, x_n in enumerate(x))
    return math.sqrt(c ** 2 + s ** 2)
[round(magnitude(k), 6) for k in range(N)]  # → [0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0, 0.0]
# f_k = k · f_s / N, with 8 samples a second
2 * 8 / N  # → 2.0
```

**In code:** `dft_magnitudes` is this formula for every k at once, written from the definition (the tests check it against NumPy's fast Fourier transform, which computes the same numbers far faster).

### The spectrogram: one Fourier transform per slice

**Everyday picture.** A piano roll, or sheet music: time runs left to
right, pitch runs bottom to top, and a mark means "this note is sounding
now". To make one from a recording, listen through a short window, say
25 milliseconds, ask "which pitches are in here?", slide the window along a
little, and ask again. That is the **short-time Fourier transform** (STFT),
and its picture is a **spectrogram**.

Each slice is faded in and out with a **Hann window** (a smooth hump from 0
up to 1 and back) before measuring, because chopping a wave off abruptly
creates a click, and a click contains every pitch at once.

**Tiny example.** Speech systems commonly use 25 ms windows (400 samples at
16 kHz) that start every 10 ms (160 samples). How many windows fit in one
second?

$$
T = 1 + \left\lfloor \frac{L - N}{H} \right\rfloor
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $L$ | the recording's length, in samples | 16,000 |
| $N$ | the window length, in samples | 400 (25 ms) |
| $H$ | the **hop**: how far the window moves each step | 160 (10 ms) |
| $\lfloor \cdot \rfloor$ | **floor**: round down to a whole number (a part-window at the end is dropped) | $\lfloor 97.5 \rfloor = 97$ |
| $T$ | the number of frames (columns of the spectrogram) | 98 |

**In words:** "one window fits at the start; after that, count how many
whole hops still leave room for a full window."

**With the numbers:** 1 + ⌊(16,000 − 400) / 160⌋ = 1 + ⌊97.5⌋ = **98**
frames per second, which is why speech models talk about "100 frames a
second".

**In Python:**

```python
L, N, H = 16000, 400, 160
# 1 + ⌊(L − N) / H⌋
1 + (L - N) // H  # → 98
```

```mermaid
flowchart LR
  W["Waveform<br/>L samples"] --> F["Slice into frames<br/>T × N"]
  F --> HW["Fade each frame<br/>(Hann window)"]
  HW --> DFT["Fourier transform<br/>per frame<br/>T × (N/2 + 1)"]
  DFT --> MEL["Pool into mel bands<br/>T × n_mels"]
  MEL --> LOG["Take the log<br/>log-mel spectrogram"]
```

**Reading it:** a flat list of samples becomes a grid. The first three
boxes are the STFT; the shape under the Fourier box says each frame now
holds one number per frequency bin. The last two boxes, explained next,
shrink those bins into a few dozen bands spaced the way ears hear, and put
loudness on a log scale. The result, a **log-mel spectrogram**, is what
speech models actually read.

![A 20-millisecond waveform of wiggles, then a spectrogram of the same second of sound showing a flat line at 1000 Hz and a line rising from 200 to 3000 Hz, then the log-mel version where the rising line curves](figures/primer.ml.generative.multimodal.spectrogram.svg)

**Reading it:** the signal is a whistle sliding from 200 Hz to 3000 Hz over
a steady, quieter 1000 Hz hum, sampled 8,000 times a second. Top: the first
20 ms of the waveform, where both sounds are tangled into one wiggle.
Middle: the spectrogram, time across and frequency up, brighter meaning
louder. The two sounds separate cleanly: a flat line for the hum and a
straight rising line for the whistle. Bottom: the log-mel version with 40
bands. The rising line now *curves*, climbing fast through the low bands and
slowly through the high ones, because mel bands are narrow at low pitch and
wide at high pitch, which is also where ears are more and less sensitive.

**In code:** `stft` slices, windows (with `hann`) and measures every frame; `frame_count` is the formula above; `log_mel_spectrogram` adds the mel pooling and the log.

### The mel scale: spacing pitch the way ears do

**Everyday picture.** On a piano, every octave takes up the same width of
keyboard, yet each octave *doubles* the frequency: the A keys are at 110,
220, 440 and 880 Hz. Ears work the same way above about 1,000 Hz: a jump
from 1,000 to 2,000 Hz sounds about as big as a jump from 2,000 to 4,000.
The **mel scale** relabels frequencies so that equal steps in mel sound like
equal steps in pitch.

**Tiny example.** By construction 1,000 Hz is 1,000 mel. The 7,000 Hz from
1,000 to 8,000 Hz shrinks to just 1,840 mel.

$$
m = 2595 \, \log_{10}\!\left(1 + \frac{f}{700}\right)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $f$ | a frequency in hertz | 700 |
| $f / 700$ | frequency measured in units of 700 Hz; below about 700 Hz the scale is nearly straight, above it the log takes over | 1 |
| $\log_{10}$ | the base-10 **logarithm**: "10 to what power gives this?"; it turns ratios into equal steps (see `primer.notation`) | $\log_{10} 2 = 0.301$ |
| 2595 | a constant chosen so that 1,000 Hz comes out as 1,000 mel | |
| $m$ | the same frequency in mel | 781.2 |

**In words:** "mel is a log of the frequency, gently straightened below
700 Hz and scaled so that 1,000 Hz is 1,000 mel."

**With the numbers:** 2595 · log₁₀(1 + 700/700) = 2595 · 0.301 = **781.2**;
1,000 Hz gives 1,000.0; 4,000 Hz gives 2,146.1; 8,000 Hz gives 2,840.0.

**In Python:**

```python
import math
def mel(f):
    return 2595 * math.log10(1 + f / 700)
round(mel(700), 1)  # → 781.2
round(mel(1000), 1)  # → 1000.0
round(mel(4000), 1)  # → 2146.1
```

A **mel filterbank** turns the hundreds of frequency bins into a few dozen
bands: triangles evenly spaced in mel, so narrow at low frequencies and wide
at high ones. Each band adds up the energy under its triangle. Loudness is
also heard by ratio, so the last step takes the log.

![Left: the mel curve rises steeply to 1000 mel at 1000 Hz and then flattens, reaching only 2840 at 8000 Hz. Right: ten triangular filters, narrow below 1000 Hz and ever wider above](figures/primer.ml.generative.multimodal.mel.svg)

**Reading it:** on the left, the purple curve is the formula and the dashed
line is where it would be if mel were simply hertz. They agree near the
bottom (the red dot, 1,000 Hz = 1,000 mel) and part ways above: the top
7,000 Hz of the range is squeezed into under 2,000 mel. On the right are 10
mel filters for 16 kHz audio. Each peaks at 1 and overlaps its neighbours;
the low ones are a few bins wide and the top one spans over 3,000 Hz. Detail
is spent where ears, and speech, need it.

**In code:** `hz_to_mel` is the formula, and `mel_filterbank` builds the triangles, evenly spaced in mel.

### Speech recognition: an encoder over frames, a decoder for text

**Everyday picture.** A court stenographer listens to a whole sentence, then
types it out word by word, glancing back at what they heard as they go.

**Tiny example.** Take a 30-second clip. With 10 ms hops it is 3,000 frames
of 80 mel bands each. A small convolution that moves two frames at a time
halves that to **1,500** encoder positions: **50 audio tokens per second**.
A transformer encoder attends over all 1,500 at once, and a decoder writes
the transcript as text tokens, attending both to its own words so far and
to the encoder's output. This is the design of Whisper.

```mermaid
flowchart LR
  A["30 s of audio"] --> LM["Log-mel spectrogram<br/>3000 frames × 80 bands"]
  LM --> CV["Convolution, stride 2<br/>1500 positions"]
  CV --> ENC["Encoder blocks<br/>(no causal mask)"]
  ENC --> X["Cross-attention:<br/>decoder reads the audio"]
  DEC["Decoder blocks<br/>(causal)"] --> X
  X --> TXT["Text tokens,<br/>one at a time"]
  TXT -.-> DEC
```

**Reading it:** the left half is a front end like the image one: turn the
signal into a grid, shrink it, and run a non-causal encoder so every moment
of sound can inform every other. The right half is a text decoder with one
extra step, cross-attention, where its queries come from the text written
so far and its keys and values come from the audio (see
`primer.ml.transformer` for encoders and decoders). The dotted arrow is the
generation loop: each new word is fed back in.

**In code:** `audio_tokens` counts encoder positions from a clip's length, the hop and the downsampling.

**Why it matters in practice.** The same audio encoder can feed a general
language model instead of a dedicated decoder: add a projector, splice the
audio tokens into the prompt, and the model can answer questions about a
recording, exactly as with images. Fifty tokens a second is manageable for
a short voice message and expensive for an hour-long meeting.

## Generating images and speech: discrete tokens from a codebook

So far the model *reads* pictures and sound. To *write* them, a language
model needs them as something it can predict one at a time from a fixed
vocabulary, the way it predicts words.

**Everyday picture.** A paint-by-numbers kit comes with a palette of
numbered pots. Any small area of a picture is described by the number of
the pot closest to its colour. The whole picture becomes a grid of pot
numbers, and anyone with the same palette can repaint it. The palette is a
**codebook**; snapping to the nearest entry is **vector quantization**.

**Tiny example.** A codebook of three 2-number entries: $c_0 = (0, 0)$,
$c_1 = (1, 0)$, $c_2 = (0, 1)$. The vector $z = (0.9, 0.2)$ has squared
distances 0.85, 0.05 and 1.45 to them, so it becomes id **1**. Decoding id 1
gives back (1, 0): close, not exact.

$$
q(z) = \arg\min_{k \in \{0, \ldots, K-1\}} \lVert z - c_k \rVert^2
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $z$ | one vector to encode: an image patch's vector or a slice of sound | (0.9, 0.2) |
| $c_k$ | codebook entry number $k$ | $c_1 = (1, 0)$ |
| $K$ | the codebook size: how many entries | 3 |
| $k \in \{0, \ldots, K-1\}$ | "$k$ ranges over the entry numbers" | 0, 1, 2 |
| $\lVert z - c_k \rVert^2$ | squared distance: subtract, square each difference, add | $(0.9-1)^2 + (0.2-0)^2 = 0.05$ |
| $\arg\min_k$ | "the $k$ that gives the smallest value" (the position of the minimum, not the minimum itself) | 1 |
| $q(z)$ | the id that replaces $z$ | 1 |

**In words:** "replace each vector by the number of its nearest codebook
entry."

**With the numbers:** distances squared to $c_0, c_1, c_2$ are
0.81 + 0.04 = 0.85, 0.01 + 0.04 = 0.05 and 0.81 + 0.64 = 1.45; the
smallest is 0.05, at $k = 1$.

**In Python:**

```python
z = [0.9, 0.2]
codebook = [[0, 0], [1, 0], [0, 1]]
# ‖z − c_k‖² for each entry
d2 = [sum((a - b) ** 2 for a, b in zip(z, c)) for c in codebook]
[round(d, 2) for d in d2]  # → [0.85, 0.05, 1.45]
# arg min: the position of the smallest
d2.index(min(d2))  # → 1
```

```mermaid
flowchart LR
  IN["Picture or sound"] --> E["Encoder<br/>vectors"]
  E --> Q["Snap to nearest<br/>codebook entry"]
  Q --> IDS["Ids, like word tokens<br/>e.g. 17, 803, 4, ..."]
  IDS --> LM["Language model<br/>learns to predict ids<br/>after text"]
  LM --> GEN["Generated ids"]
  GEN --> LOOK["Look up codebook<br/>vectors"]
  LOOK --> D["Decoder"] --> OUTP["Pixels or waveform"]
```

**Reading it:** the top row is used during training: real pictures and
sounds are encoded, snapped to the codebook, and become sequences of ids
that sit in the training text next to their descriptions. The language
model learns to continue a caption with ids the same way it learns to
continue a sentence with words; its vocabulary is simply enlarged by $K$
entries. At generation time (bottom row) the model writes ids, the codebook
turns them back into vectors, and a decoder paints pixels or synthesises a
waveform. The encoder, codebook and decoder together are a VQ-VAE (see
`primer.ml.generative.autoencoders`).

How is the codebook chosen? It is learned so that the entries sit where the
data actually is, which amounts to k-means clustering (see
`primer.ml.embeddings.clustering`): snap every vector to its nearest entry,
move each entry to the average of the vectors that chose it, and repeat.
For audio, one codebook is rarely precise enough, so neural audio codecs
use **residual vector quantization**: a second codebook encodes what the
first one missed, a third what the second missed, and so on. Each moment
of sound becomes a small stack of ids.

![On new patches, a learned codebook's error falls from 0.32 to about 0.09 as it grows to 64 entries while random codebooks stay far above it; stacking residual codebooks of 8 entries keeps pushing the error down](figures/primer.ml.generative.multimodal.codebook.svg)

**Reading it:** the x-axis is bits per patch: a codebook of $K$ entries
costs log₂ $K$ bits per id, so 64 entries is 6 bits. The y-axis (log scale)
is the reconstruction error on patches of *new* pictures, not the ones the
codebook was learned from. Random codebooks (red) are useless at every
size: their entries sit where no data lives. Learned codebooks (blue) fall
fast and then flatten near the dotted line, which is the error you would get
by reproducing each clean pattern perfectly and dropping only the pixel
noise. The green squares stack 8-entry codebooks, each learned on the
previous one's leftovers: after four stages (12 bits) they reach below the
dotted line, because the extra codes start describing the noise of each
particular patch too.

**In code:** `quantize` is the arg min above and `dequantize` the lookup; `kmeans_codebook` learns a codebook; `learn_residual_codebooks` and `residual_quantize` stack codebooks on the leftovers.

**Why it matters in practice.** Discrete tokens let one model read and
write every modality with one mechanism, next-token prediction. The costs
are real: a 256×256 image compressed eight-fold per side is 32 × 32 =
1,024 ids (the scheme of the original DALL·E), and a neural audio codec
running at 75 frames a second with 8 codebooks writes 600 ids a second. The
other road to generating images is diffusion (see
`primer.ml.generative.diffusion`): the language model supplies text or
vectors that *condition* a diffusion model, which paints the pixels in many
small denoising steps. Discrete tokens are simpler to bolt onto a language
model; diffusion usually renders finer detail.

## Video: pictures over time

**Everyday picture.** A flipbook. Each page is a picture, and neighbouring
pages are almost identical. To follow the story you don't need to look at
every page; a glance at every tenth page shows what happens.

**Tiny example.** A 10-second clip at 30 frames a second, with each frame
cut into 256 tokens (224×224 in 14-pixel patches): 300 frames × 256 =
**76,800 tokens**, more than half of a 128,000-token context, for ten
seconds. Keeping one frame a second gives 10 × 256 = **2,560**.

$$
N_{\text{video}} = \frac{D \cdot r}{t} \cdot \frac{H}{P} \cdot \frac{W}{P}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $D$ | the clip's duration, in seconds | 10 |
| $r$ | frames kept per second (the **sampling rate**, often far below the video's own frame rate) | 30, then 1 |
| $t$ | the **tubelet** depth: how many consecutive frames share one token, when a patch is cut through time as well as space | 1 |
| $D \cdot r / t$ | how many frame slots become tokens | 300, then 10 |
| $\frac{H}{P} \cdot \frac{W}{P}$ | tokens per frame, from the image formula | 16 · 16 = 256 |
| $N_{\text{video}}$ | the clip's total tokens | 76,800, then 2,560 |

**In words:** "count the frames you keep, divide by how many frames each
token spans, and multiply by the tokens per frame."

**With the numbers:** (10 · 30 / 1) · 16 · 16 = 300 · 256 = 76,800; at one
frame a second, (10 · 1 / 1) · 256 = 2,560; sampling 2 frames a second in
tubelets 2 frames deep gives (10 · 2 / 2) · 256 = 2,560 again, while
covering twice as many moments.

**In Python:**

```python
H = W = 224
P = 14
tokens_per_frame = (H // P) * (W // P)
tokens_per_frame  # → 256
D, r, t = 10, 30, 1
D * r // t * tokens_per_frame  # → 76800
# one frame a second
D, r, t = 10, 1, 1
D * r // t * tokens_per_frame  # → 2560
```

```mermaid
flowchart LR
  V["Video<br/>30 frames a second"] --> S["Sample frames<br/>e.g. 1 a second"]
  S --> ENC["Vision encoder per frame<br/>(or tubelets across frames)"]
  ENC --> M["Merge neighbouring<br/>patch tokens"]
  M --> PR["Projector"]
  PR --> SEQ["Frame tokens,<br/>with timestamps as text"]
  SEQ --> LM["Language model"]
```

**Reading it:** each box from left to right cuts the token count. Frame
sampling throws away near-duplicate pages of the flipbook; tubelets and
merging squeeze each kept moment into fewer tokens. A short text timestamp
before each frame's tokens ("at 0:05") lets the model say *when* something
happened. What is left is the image pipeline, repeated once per kept frame.

**In code:** `video_tokens` is the formula, and `sample_frames` picks the evenly spaced frames to keep.

**Why it matters in practice.** Sampling is a bet that nothing important
happens between the frames you keep: one frame a second is fine for a
lecture and useless for a golf swing. Real systems adapt: more frames for
short or fast clips, fewer and smaller ones for long recordings, and a
transcript of the soundtrack alongside.

## What it costs: tokens per second and the context budget

**Everyday picture.** Packing a suitcase. Text is neatly folded shirts;
images are shoes; video is an inflatable boat. The suitcase (the context
window) is the same size whatever you pack.

**Tiny example.** A 128,000-token window holds 128,000 / 256 = **500
seconds**, about 8 minutes, of video sampled at one frame a second, before
a single word of the question. The same window holds about 11 hours of
speech written out as text.

| Input | Tokens per second | Minutes in 128k tokens |
|---|---|---|
| speech written as text (about 150 words a minute, about 1.3 tokens a word) | 3.2 | 656 |
| speech as audio-encoder tokens | 50 | 43 |
| video, 1 frame a second, 256 tokens a frame | 256 | 8.3 |
| audio as codec tokens (75 frames × 8 codebooks) | 600 | 3.6 |
| video, 30 frames a second, 256 tokens a frame | 7,680 | 0.3 |

![On log axes, five straight lines of tokens against recording length: speech-as-text crosses a 128k window only after hours, 30-frames-a-second video within seconds](figures/primer.ml.generative.multimodal.token_budget.svg)

**Reading it:** both axes are logarithmic, so each line is the same slope
shifted up or down by its tokens per second. Read across at a dashed line
(a context window) to see how long a recording of each kind fits. Text
(grey) is so dense that hours of speech fit easily. Audio tokens (green)
fill a 128k window in about 43 minutes, sampled video (blue) in about 8,
and full-rate video (red) in under 20 seconds. The gap between the grey
line and the others is the price of keeping the raw signal instead of
words.

**In code:** `seconds_that_fit` divides a window by a rate, and `TOKENS_PER_SECOND` holds the rates in the table.

**Why it matters in practice.** Every image and second of audio is billed
and computed like text: it takes room in the context window, lengthens the
prefill before the first word of the answer (see `primer.ml.inference`),
and competes with instructions and retrieved documents for attention (see
`primer.agents.context`). The practical levers follow from the formulas:
send the smallest resolution that still shows the detail you need, crop to
the region that matters, sample fewer frames, and transcribe audio to text
when the words matter and the tone does not.

## In 20 seconds

- **One idea:** every modality becomes a sequence of vectors as wide as the
  language model's word vectors; after that it is ordinary attention.
- **Images:** a Vision Transformer cuts a picture into patches, projects each
  one, adds a position, and runs non-causal encoder blocks. Tokens =
  (H/P)·(W/P), so they grow with the square of the resolution.
- **Connecting to a language model:** a small projector maps image vectors
  into the embedding space, and they are spliced into the prompt. Train the
  projector first with everything frozen, then fine-tune on image
  instructions.
- **Audio:** waveform, then a log-mel spectrogram (a Fourier transform per
  25 ms slice, pooled into ear-shaped bands), then an encoder: about 50
  tokens a second.
- **Generating pictures and sound:** snap vectors to a learned codebook so
  they become ids a language model can predict like words; or hand off to a
  diffusion model.
- **Video and cost:** frames multiply image tokens by time; sample frames,
  merge tokens, and remember that text is by far the densest input.

## Self-test questions

**How can a chatbot "see" a photo, explained without jargon?**
The photo is cut into a grid of small squares, and each square is described
by a list of numbers in the same format the model uses for words. The model
then reads the squares and the words of your question together, paying
attention to whichever squares help answer it. It never sees the picture
as a picture: it reads it as a few hundred extra "words".

**How many tokens is a 448×448 image with 14-pixel patches? And if each
2×2 block of neighbours is merged?**
448 / 14 = 32 patches a side, so 32 × 32 = 1,024 tokens. Merging 2×2
blocks divides by 4: 256 tokens. Doubling the side from 224 would have
quadrupled the count.

**Why does a Vision Transformer need position vectors, and why is its
attention not causal?**
Attention ignores order, so without positions the model sees an unordered
bag of patches and cannot tell the sky from the ground. It is not causal
because a picture has no "future": every patch may use every other, and
hiding half the image would only throw information away.

**What does the projector do, and why train it first with both big models
frozen?**
It maps each image vector into the language model's embedding space, so
image tokens look like something the language model can use. Training it
alone first is cheap (it is tiny) and safe (the frozen models keep all
they know). Only once the two sides understand each other is the language
model tuned on image instructions.

**In a prompt, why does it matter whether the image comes before or after
the question?**
The language model is causal: a token can only attend to tokens before it.
Question tokens placed after the image can look at it while being
processed; question tokens placed before it cannot. The answer comes after
both either way, but putting the image first lets the whole question be read
in light of the picture.

**Why does a speech model read a log-mel spectrogram rather than raw
samples?**
Raw audio is 16,000 numbers a second with pitch hidden in the wiggles. The
spectrogram makes pitch explicit (which frequencies sound at each moment),
the mel bands spend detail where ears and speech need it, the log matches
how loudness is heard, and 100 frames a second is far fewer positions to
attend over.

**How can a model that only predicts tokens produce an image or a voice?**
Encode pictures or sounds into vectors, snap each vector to its nearest
entry in a learned codebook, and use the entry numbers as extra vocabulary.
The model learns to predict those ids after a caption, and a decoder turns
generated ids back into pixels or a waveform. The alternative is to have
the model condition a diffusion model that paints the image.

**A 20-minute video goes to a model with a 128,000-token window. What goes
wrong, and what can be done?**
At one frame a second and 256 tokens a frame it is 1,200 × 256 = 307,200
tokens, more than twice the window, before any question. Options: sample
far fewer frames (one every 5 seconds gives 61,440), merge neighbouring
patch tokens, lower the resolution, transcribe the soundtrack to text, or
split the video and summarise the parts.

## The papers behind this lesson

- **Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for
  Image Recognition at Scale* (2020)**: https://arxiv.org/abs/2010.11929.
  Showed that a plain transformer over image patches matches convolutional
  networks when trained on enough data: the Vision Transformer.
- **Radford et al., *Learning Transferable Visual Models From Natural
  Language Supervision* (CLIP, 2021)**: https://arxiv.org/abs/2103.00020.
  Trained an image encoder and a text encoder into one shared space; its
  image encoder is the starting point of many vision-language models.
- **Alayrac et al., *Flamingo: a Visual Language Model for Few-Shot
  Learning* (2022)**: https://arxiv.org/abs/2204.14198. Connected a frozen
  vision encoder to a frozen language model through new cross-attention
  layers, handling images and video interleaved with text.
- **Liu et al., *Visual Instruction Tuning* (LLaVA, 2023)**:
  https://arxiv.org/abs/2304.08485. The encode, project and splice recipe,
  trained in two stages: align the projector, then tune on image
  instructions.
- **Radford et al., *Robust Speech Recognition via Large-Scale Weak
  Supervision* (Whisper, 2022)**: https://arxiv.org/abs/2212.04356. An
  encoder over log-mel spectrograms and a text decoder, trained on 680,000
  hours of transcribed audio.
- **van den Oord, Vinyals & Kavukcuoglu, *Neural Discrete Representation
  Learning* (VQ-VAE, 2017)**: https://arxiv.org/abs/1711.00937. Learned a
  codebook inside an autoencoder, turning images and audio into discrete
  tokens.
- **Zeghidour et al., *SoundStream: An End-to-End Neural Audio Codec*
  (2021)**: https://arxiv.org/abs/2107.03312. Residual vector quantization
  for audio: a stack of codebooks, each encoding the previous one's error.
- **Ramesh et al., *Zero-Shot Text-to-Image Generation* (DALL·E, 2021)**:
  https://arxiv.org/abs/2102.12092. Generated images as sequences of
  discrete codebook tokens after the text, with one transformer.
- **Arnab et al., *ViViT: A Video Vision Transformer* (2021)**:
  https://arxiv.org/abs/2103.15691. Extended patches through time as
  tubelets, and compared ways to factorise attention over space and time.

## Further reading

- Dosovitskiy et al., *An Image is Worth 16x16 Words* (ViT, 2020): https://arxiv.org/abs/2010.11929
- Liu et al., *Visual Instruction Tuning* (LLaVA, 2023): https://arxiv.org/abs/2304.08485
- Liu et al., *Improved Baselines with Visual Instruction Tuning* (LLaVA-1.5, 2023): https://arxiv.org/abs/2310.03744
- Li et al., *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models* (2023): https://arxiv.org/abs/2301.12597
- Alayrac et al., *Flamingo* (2022): https://arxiv.org/abs/2204.14198
- Radford et al., *Whisper* (2022): https://arxiv.org/abs/2212.04356 and its code: https://github.com/openai/whisper
- Défossez et al., *High Fidelity Neural Audio Compression* (EnCodec, 2022): https://arxiv.org/abs/2210.13438
- van den Oord et al., *Neural Discrete Representation Learning* (VQ-VAE, 2017): https://arxiv.org/abs/1711.00937
- Chameleon Team, *Chameleon: Mixed-Modal Early-Fusion Foundation Models* (2024): https://arxiv.org/abs/2405.09818
- Arnab et al., *ViViT: A Video Vision Transformer* (2021): https://arxiv.org/abs/2103.15691
"""

from __future__ import annotations

import numpy as np

from primer._show import banner, matrix, say, table, takeaway
from primer.ml.attention import softmax
from primer.ml.cnn_rnn import patchify
from primer.ml.transformer import TinyGPT, TransformerBlock, gelu, layer_norm

# ---------------------------------------------------------------------------
# 1. Images into tokens: patches, projection, positions
# ---------------------------------------------------------------------------


def image_to_patches(image: np.ndarray, patch: int) -> np.ndarray:
    """Cut an (H, W) or (H, W, C) image into square patches, one flattened row each.

    Returns (number of patches, patch·patch·C), row by row from the top left.
    The cutting itself is `primer.ml.cnn_rnn.patchify`; this adds the check
    that the patch size divides the image, because a ragged edge would give
    a last patch with pixels missing.
    """
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[:, :, None]  # grayscale: one colour channel
    h, w, _ = image.shape
    if h % patch or w % patch:
        raise ValueError(f"a {patch}-pixel patch does not divide a {h}×{w} image; resize or pad it first")
    return patchify(image, patch)


def count_image_tokens(height: int, width: int, patch: int, merge: int = 1) -> int:
    """Tokens for one image: (H/P)·(W/P) patches, divided by merge² if neighbours are pooled.

    Many vision-language models merge each merge×merge block of neighbouring
    patch vectors into one token before the language model sees them, which
    cuts the count by merge².
    """
    if height % patch or width % patch:
        raise ValueError("resize the image to a multiple of the patch size first")
    grid_h, grid_w = height // patch, width // patch
    return (grid_h * grid_w) // (merge * merge)


# The hand-checkable example from the lesson: pixel brightnesses 0 to 15.
WORKED_IMAGE = np.arange(16).reshape(4, 4)
# Column 1 adds up a patch's top row of pixels, column 2 its bottom row.
WORKED_W_E = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
# One position vector per patch: (row, column) of the patch in the grid.
WORKED_POSITIONS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])


def worked_example_patches() -> dict[str, np.ndarray]:
    """The 4×4 image cut into four 2×2 patches, projected, then given positions.

    | patch        | pixels       | x·W_E    | + position | token    |
    |--------------|--------------|----------|------------|----------|
    | top left     | 0, 1, 4, 5   | (1, 9)   | (0, 0)     | (1, 9)   |
    | top right    | 2, 3, 6, 7   | (5, 13)  | (0, 1)     | (5, 14)  |
    | bottom left  | 8, 9, 12, 13 | (17, 25) | (1, 0)     | (18, 25) |
    | bottom right | 10, 11, 14, 15 | (21, 29) | (1, 1)   | (22, 30) |
    """
    patches = image_to_patches(WORKED_IMAGE, patch=2)  # (4, 4): 4 patches of 4 pixels
    projected = patches @ WORKED_W_E  # (4, 2): each patch squeezed to 2 numbers
    return dict(patches=patches, projected=projected, tokens=projected + WORKED_POSITIONS)


class VisionEncoder:
    """A Vision Transformer, forward pass only: patches -> vectors -> encoder blocks.

    ```text
    image (H, W, C) -> patches (N, P·P·C) -> ·W_E + b_E (N, d) -> + positions (N, d)
                    -> n_layers × TransformerBlock(causal=False) -> LayerNorm (N, d)
    ```

    The blocks are `primer.ml.transformer.TransformerBlock` with the causal
    mask switched off: a picture has no "future", so every patch may attend
    to every other. Weights are random; this shows the machinery.
    """

    def __init__(self, image_size: int, patch: int, channels: int, d_model: int, n_layers: int, n_heads: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.patch = patch
        self.n_patches = count_image_tokens(image_size, image_size, patch)
        patch_dim = patch * patch * channels
        # Scaled init keeps each token's numbers near unit size whatever the patch size.
        self.W_E = rng.normal(0, 1 / np.sqrt(patch_dim), (patch_dim, d_model))
        self.b_E = np.zeros(d_model)
        # Learned positions, one vector per patch slot, as ViT does (std 0.02 as in GPT-2).
        self.pos = rng.normal(0, 0.02, (self.n_patches, d_model))
        self.blocks = [TransformerBlock(d_model, n_heads, causal=False, seed=seed + 10 * i + 1) for i in range(n_layers)]

    def embed(self, image: np.ndarray) -> np.ndarray:
        """(N, d): each patch projected to model width, plus its position vector."""
        return image_to_patches(image, self.patch) @ self.W_E + self.b_E + self.pos

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """(N, d): one context-aware vector per patch, the image's tokens."""
        x = self.embed(image)
        for block in self.blocks:
            x = block(x)
        return layer_norm(x)


# ---------------------------------------------------------------------------
# 2. The projector, and a toy vision-language model
# ---------------------------------------------------------------------------


class Projector:
    """Maps vision vectors into the language model's embedding space.

    `hidden=None` is a single linear layer, v·W + b. With `hidden` set it is
    a two-layer MLP, GELU(v·W1 + b1)·W2 + b2: the same shape as a
    transformer's feed-forward network (`primer.ml.transformer.FeedForward`),
    with a different width on each side.
    """

    def __init__(self, d_in: int, d_out: int, hidden: int | None = None, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.hidden = hidden
        if hidden is None:
            self.W = rng.normal(0, 1 / np.sqrt(d_in), (d_in, d_out))
            self.b = np.zeros(d_out)
        else:
            self.W1 = rng.normal(0, 1 / np.sqrt(d_in), (d_in, hidden))
            self.b1 = np.zeros(hidden)
            self.W2 = rng.normal(0, 1 / np.sqrt(hidden), (hidden, d_out))
            self.b2 = np.zeros(d_out)

    def __call__(self, v: np.ndarray) -> np.ndarray:
        # (N, d_in) -> (N, d_out): each image token translated on its own.
        if self.hidden is None:
            return v @ self.W + self.b
        return gelu(v @ self.W1 + self.b1) @ self.W2 + self.b2


# The worked linear projection: a 2-number vision vector into a 3-number text space.
WORKED_V = np.array([1, 2])
WORKED_W_P = np.array([[1, 0, 1], [0, 1, 1]])


def worked_example_projection() -> np.ndarray:
    """(1, 2)·W_P = (1·1 + 2·0, 1·0 + 2·1, 1·1 + 2·1) = (1, 2, 3)."""
    return WORKED_V @ WORKED_W_P


# A vocabulary small enough to print. Id 0 is the image placeholder.
TEXT_VOCAB = ["<image>", "a", "photo", "of", "horizontal", "vertical", "diagonal", "checkered", "stripes", "what", "is", "this"]
IMAGE = 0
CLASS_NAMES = ["horizontal", "vertical", "diagonal", "checkered"]
CLASS_WORD_IDS = np.array([TEXT_VOCAB.index(w) for w in CLASS_NAMES])


class VisionLanguageModel:
    """Vision encoder -> projector -> image tokens spliced into a decoder language model.

    The language model is `primer.ml.transformer.TinyGPT`. Instead of looking
    up every position in its token table, the image placeholder is replaced
    by the projected image tokens; everything after that is TinyGPT's own
    forward pass.
    """

    def __init__(self, vision: VisionEncoder, projector: Projector, lm: TinyGPT):
        self.vision, self.projector, self.lm = vision, projector, lm

    def splice(self, ids: np.ndarray, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(L, d_lm) input vectors with the image tokens where `IMAGE` was, and an is-image flag per row."""
        image_tokens = self.projector(self.vision(image))  # (N, d_lm)
        rows, flags = [], []
        for token_id in ids:
            if token_id == IMAGE:
                rows.append(image_tokens)
                flags += [True] * len(image_tokens)
            else:
                rows.append(self.lm.wte[token_id][None, :])  # a text token: its row of the table
                flags.append(False)
        return np.concatenate(rows), np.array(flags)

    def __call__(self, ids: np.ndarray, image: np.ndarray) -> np.ndarray:
        """(L, vocab) logits: row i scores every possible next token after position i."""
        x, _ = self.splice(ids, image)
        x = x + self.lm.wpe[: len(x)]  # positions count image tokens too
        for block in self.lm.blocks:
            x = block(x)
        return layer_norm(x, self.lm.lnf_g, self.lm.lnf_b) @ self.lm.wte.T


def build_toy_vlm(seed: int = 0) -> VisionLanguageModel:
    """An 8×8 grayscale ViT (4 patches, width 16), a linear projector, and a width-24 TinyGPT."""
    vision = VisionEncoder(image_size=8, patch=4, channels=1, d_model=16, n_layers=2, n_heads=2, seed=seed)
    projector = Projector(16, 24, seed=seed + 1)
    lm = TinyGPT(vocab_size=len(TEXT_VOCAB), d_model=24, n_layers=2, n_heads=2, max_len=32, seed=seed + 2)
    return VisionLanguageModel(vision, projector, lm)


def toy_images(n_per_class: int, size: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Four kinds of 8×8 grayscale picture (horizontal, vertical, diagonal, checkered), with noise.

    Each image gets its own contrast, brightness and pixel noise, so no two
    are identical. Returns (images (4n, size, size), labels (4n,)).
    """
    rng = np.random.default_rng(seed)
    r, c = np.mgrid[0:size, 0:size]
    # 2-pixel-wide bands, so every 4×4 patch holds a full light-and-dark cycle of its pattern.
    patterns = [(r // 2) % 2, (c // 2) % 2, ((r + c) // 2) % 2, (r // 2 + c // 2) % 2]
    images, labels = [], []
    for label, pattern in enumerate(patterns):
        for _ in range(n_per_class):
            contrast, brightness = rng.uniform(0.6, 1.4), rng.uniform(-0.3, 0.3)
            images.append(contrast * pattern + brightness + 0.3 * rng.standard_normal((size, size)))
            labels.append(label)
    return np.array(images, dtype=float), np.array(labels)


def _pooled_features(model: VisionLanguageModel, images: np.ndarray) -> np.ndarray:
    """(B, d_vision): the mean of each image's vision tokens. The encoder is frozen, so these can be computed once."""
    return np.array([model.vision(im).mean(axis=0) for im in images])


def caption_accuracy(model: VisionLanguageModel, images: np.ndarray, labels: np.ndarray) -> float:
    """Share of images whose projected tokens score their own class word highest in the whole vocabulary."""
    h = model.projector(_pooled_features(model, images))  # (B, d_lm)
    predicted = (h @ model.lm.wte.T).argmax(axis=1)  # scored against every word, as the output layer does
    return float(np.mean(predicted == CLASS_WORD_IDS[labels]))


def align_projector(model: VisionLanguageModel, images: np.ndarray, labels: np.ndarray, steps: int = 300, lr: float = 2.0) -> dict[str, list[float]]:
    """Stage 1 of training: fit only the (linear) projector so each image says its caption word.

    The loss is cross-entropy on the caption word, scored by the language
    model's own output layer (its tied token table). The vision encoder and
    the language model are frozen: only `projector.W` and `projector.b` move.
    A real model sends the image tokens through every language-model layer
    and back-propagates through them; this toy scores the pooled image
    vector against the token table directly, the last step of that same path.
    """
    v = _pooled_features(model, images)  # (B, d_vision), fixed while training
    wte = model.lm.wte  # (vocab, d_lm), frozen
    target = CLASS_WORD_IDS[labels]
    history: dict[str, list[float]] = {"loss": [], "accuracy": []}
    for _ in range(steps):
        h = model.projector(v)  # (B, d_lm)
        p = softmax(h @ wte.T)  # (B, vocab)
        history["loss"].append(float(-np.mean(np.log(p[np.arange(len(v)), target]))))
        history["accuracy"].append(float(np.mean(p.argmax(axis=1) == target)))
        # Gradient of mean cross-entropy: (p - one_hot) flows back through the frozen table.
        d_logits = p.copy()
        d_logits[np.arange(len(v)), target] -= 1
        d_h = d_logits @ wte / len(v)  # (B, d_lm)
        model.projector.W -= lr * v.T @ d_h
        model.projector.b -= lr * d_h.sum(axis=0)
    return history


# ---------------------------------------------------------------------------
# 3. Audio: waveform -> spectrogram -> tokens
# ---------------------------------------------------------------------------


def tone(freq: float, seconds: float, sample_rate: int) -> np.ndarray:
    """A pure sine wave: the waveform of a single steady pitch."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return np.sin(2 * np.pi * freq * t)


def chirp(f_start: float, f_end: float, seconds: float, sample_rate: int) -> np.ndarray:
    """A sine whose pitch rises steadily from f_start to f_end, like a whistle sliding up."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    # Phase is the running total of frequency: f(t) = f_start + (f_end - f_start)·t/T.
    return np.sin(2 * np.pi * (f_start * t + (f_end - f_start) * t**2 / (2 * seconds)))


def _dft_basis(n: int, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """(n, n_bins) cosine and sine waves: column k completes k cycles in n samples."""
    k = np.arange(n_bins)[None, :]
    t = np.arange(n)[:, None]
    angle = 2 * np.pi * k * t / n
    return np.cos(angle), np.sin(angle)


def dft_magnitudes(x) -> np.ndarray:
    """|X_k| for k = 0..n-1: how much of the wave that cycles k times the signal contains.

    Written as the definition, with no fast Fourier transform: correlate the
    signal with a cosine and a sine of each frequency, and take the length
    of the pair.
    """
    x = np.asarray(x, dtype=float)
    cos, sin = _dft_basis(len(x), len(x))
    return np.sqrt((x @ cos) ** 2 + (x @ sin) ** 2)


def frame_count(n_samples: int, window: int, hop: int) -> int:
    """Whole windows that fit: 1 + ⌊(L - N) / H⌋."""
    return 1 + (n_samples - window) // hop


def hann(n: int) -> np.ndarray:
    """A window that rises from 0 to 1 and back, so each slice fades in and out instead of starting with a click."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))


def stft(x: np.ndarray, window: int, hop: int) -> np.ndarray:
    """Short-time Fourier transform magnitudes: (frames, window//2 + 1).

    Slide a window of `window` samples along the signal in steps of `hop`,
    fade each slice with a Hann window, and measure every frequency in it.
    Only bins 0..window/2 are kept: for a real signal the rest mirror them.
    """
    t = frame_count(len(x), window, hop)
    starts = np.arange(t)[:, None] * hop
    frames = x[starts + np.arange(window)[None, :]] * hann(window)  # (T, window)
    cos, sin = _dft_basis(window, window // 2 + 1)
    return np.sqrt((frames @ cos) ** 2 + (frames @ sin) ** 2)


def hz_to_mel(f):
    """mel(f) = 2595·log10(1 + f/700): roughly even steps below 700 Hz, ratios above."""
    return 2595 * np.log10(1 + np.asarray(f, dtype=float) / 700)


def mel_to_hz(m):
    """The inverse of `hz_to_mel`."""
    return 700 * (10 ** (np.asarray(m, dtype=float) / 2595) - 1)


def mel_filterbank(n_mels: int, n_fft: int, sample_rate: int) -> np.ndarray:
    """(n_mels, n_fft//2 + 1) triangular filters, evenly spaced in mel, so wider in hertz as pitch rises.

    Filter m rises from 0 at mel point m to 1 at point m+1 and falls back to
    0 at point m+2. Multiplying a spectrogram by this matrix's transpose
    adds up each band's energy: many fine bins become a few bands.
    """
    edges = mel_to_hz(np.linspace(0, hz_to_mel(sample_rate / 2), n_mels + 2))  # n_mels + 2 edges in Hz
    freqs = np.arange(n_fft // 2 + 1) * sample_rate / n_fft  # the frequency of each spectrogram bin
    lo, centre, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rising = (freqs - lo) / (centre - lo)
    falling = (hi - freqs) / (hi - centre)
    return np.maximum(0, np.minimum(rising, falling))


def log_mel_spectrogram(x: np.ndarray, sample_rate: int, window: int, hop: int, n_mels: int) -> np.ndarray:
    """(frames, n_mels): power per mel band, on a log scale because loudness is heard by ratio too."""
    power = stft(x, window, hop) ** 2
    return np.log10(power @ mel_filterbank(n_mels, window, sample_rate).T + 1e-10)


def audio_tokens(seconds: float, hop_ms: float = 10, downsample: int = 2) -> int:
    """Encoder positions for a clip: one spectrogram frame per hop, halved by a stride-2 convolution."""
    return int(round(seconds * 1000 / hop_ms)) // downsample


# ---------------------------------------------------------------------------
# 4. Discrete tokens for outputs: vector quantization
# ---------------------------------------------------------------------------


def quantize(Z: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """(n,) ids: for each row of Z, the number of the nearest codebook entry."""
    # ‖z - c‖² for every pair, (n, K), without a Python loop.
    d2 = (Z**2).sum(axis=1)[:, None] - 2 * Z @ codebook.T + (codebook**2).sum(axis=1)[None, :]
    return d2.argmin(axis=1)


def dequantize(ids: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """(n, d): look each id up in the codebook. What a decoder receives back."""
    return codebook[ids]


def quantization_error(Z: np.ndarray, codebook: np.ndarray) -> float:
    """Mean squared difference between each vector and the codebook entry it snaps to."""
    return float(np.mean((Z - dequantize(quantize(Z, codebook), codebook)) ** 2))


def kmeans_codebook(Z: np.ndarray, k: int, iters: int = 20, seed: int = 0) -> np.ndarray:
    """Learn k codebook entries with k-means: snap every vector, move each entry to its vectors' mean.

    VQ-VAE learns its codebook inside a network, but the update it converges
    to is this one (see `primer.ml.embeddings.clustering` for k-means).
    """
    rng = np.random.default_rng(seed)
    codebook = Z[rng.choice(len(Z), size=k, replace=False)].copy()  # start from k real vectors
    for _ in range(iters):
        ids = quantize(Z, codebook)
        for j in range(k):
            members = Z[ids == j]
            if len(members):  # an entry nobody chose keeps its place rather than becoming NaN
                codebook[j] = members.mean(axis=0)
    return codebook


def learn_residual_codebooks(Z: np.ndarray, k: int, stages: int, seed: int = 0) -> list[np.ndarray]:
    """Residual vector quantization: each new codebook is learned on what the previous ones missed."""
    books, residual = [], Z.copy()
    for s in range(stages):
        book = kmeans_codebook(residual, k, seed=seed + s)
        residual = residual - dequantize(quantize(residual, book), book)
        books.append(book)
    return books


def residual_quantize(Z: np.ndarray, codebooks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(ids (stages, n), reconstruction (n, d)): one id per stage per vector; the sum of their entries rebuilds it."""
    ids, recon, residual = [], np.zeros_like(Z, dtype=float), Z.astype(float)
    for book in codebooks:
        stage_ids = quantize(residual, book)
        recon = recon + book[stage_ids]
        residual = residual - book[stage_ids]
        ids.append(stage_ids)
    return np.array(ids), recon


# ---------------------------------------------------------------------------
# 5. Video and the context budget
# ---------------------------------------------------------------------------


def video_tokens(seconds: float, sample_fps: float, tokens_per_frame: int, tubelet: int = 1) -> int:
    """(seconds · sampled frames per second / frames per tubelet) · tokens per frame."""
    frames = int(round(seconds * sample_fps))
    return (frames // tubelet) * tokens_per_frame


def sample_frames(n_frames: int, video_fps: float, target_fps: float) -> np.ndarray:
    """Indices of the frames kept when sampling a video at `target_fps`: evenly spaced, starting at 0."""
    step = video_fps / target_fps
    return np.floor(np.arange(0, n_frames, step)).astype(int)


def seconds_that_fit(context_tokens: int, tokens_per_second: float) -> float:
    """How many seconds of a stream fit in a context window, ignoring the prompt and the answer."""
    return context_tokens / tokens_per_second


# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------


# Rough tokens per second of input, for comparing modalities (see the lesson for where each comes from).
TOKENS_PER_SECOND = {
    "speech written as text (150 words a minute)": 150 * 1.3 / 60,
    "speech as audio-encoder tokens": audio_tokens(1),
    "video, 1 frame a second, 256 tokens a frame": video_tokens(1, 1, 256),
    "audio as codec tokens (75 frames × 8 codebooks)": 75 * 8,
    "video, 30 frames a second, 256 tokens a frame": video_tokens(1, 30, 256),
}

SAMPLE_RATE = 8000  # samples per second for the lesson's synthetic audio


def demo_signal(seconds: float = 1.0) -> np.ndarray:
    """A whistle sliding up from 200 Hz to 3000 Hz over a steady, quieter 1000 Hz hum."""
    return chirp(200, 3000, seconds, SAMPLE_RATE) + 0.5 * tone(1000, seconds, SAMPLE_RATE)


def toy_picture(size: int = 32) -> np.ndarray:
    """A 32×32 grayscale scene to cut into patches: a sun above a striped field."""
    r, c = np.mgrid[0:size, 0:size]
    sky = 0.25 + 0.3 * (r < size // 2)
    sun = ((r - 9) ** 2 + (c - 22) ** 2 < 25) * 0.6
    field = (r >= size // 2) * (0.3 + 0.4 * ((c // 3) % 2))
    return np.clip(sky * (r < size // 2) + sun + field, 0, 1)


def codebook_sweep(sizes=(1, 2, 4, 8, 16, 32, 64), seed: int = 0) -> list[dict]:
    """Quantization error on new toy-image patches, for learned and random codebooks of each size.

    The codebook is learned on one set of images and scored on another, so a
    large codebook cannot look good by memorising the noise it was fitted to.
    """
    patches = np.concatenate([image_to_patches(im, 4) for im in toy_images(30)[0]])
    new = np.concatenate([image_to_patches(im, 4) for im in toy_images(30, seed=99)[0]])
    rng = np.random.default_rng(seed)
    lo, hi = patches.min(), patches.max()
    rows = []
    for k in sizes:
        random = rng.uniform(lo, hi, (k, patches.shape[1]))
        rows.append(dict(k=k, bits=float(np.log2(k)), learned=quantization_error(new, kmeans_codebook(patches, k, seed=seed)),
                         random=quantization_error(new, random)))
    return rows


# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------


def figures() -> dict:
    """Plot this lesson's data. matplotlib is imported here, and only here,
    so the lesson itself needs nothing beyond NumPy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, RED, GREEN, AMBER, PURPLE, MUTED = "#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#9ca3af"
    figs = {}

    # --- 1. An image cut into patches, and the token matrix it becomes --------
    picture = toy_picture()
    patches = image_to_patches(picture, 8)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8), gridspec_kw=dict(width_ratios=[1, 1.6]))
    a1.imshow(picture, cmap="gray", vmin=0, vmax=1)
    for edge in range(0, 33, 8):
        a1.axhline(edge - 0.5, color=AMBER, lw=1.5)
        a1.axvline(edge - 0.5, color=AMBER, lw=1.5)
    for i in range(16):
        a1.text((i % 4) * 8 + 3.5, (i // 4) * 8 + 3.5, str(i), color="white", ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="#1f2937", ec="none", alpha=0.75))
    a1.set_xticks([])
    a1.set_yticks([])
    a1.grid(False)
    a1.set_title("32×32 image, 8-pixel patches")
    a2.imshow(patches, cmap="gray", vmin=0, vmax=1, aspect="auto")
    a2.set_yticks(range(16))
    a2.set_ylabel("token (patch number)")
    a2.set_xlabel("the patch's 64 pixels, read row by row")
    a2.grid(False)
    a2.set_title("…becomes 16 tokens of 64 numbers")
    fig.tight_layout()
    figs["patches"] = fig

    # --- 2. Tokens per image as resolution grows --------------------------------
    sides = np.arange(112, 1345, 112)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for patch, merge, color, label in ((14, 1, RED, "14-pixel patches"), (16, 1, BLUE, "16-pixel patches"),
                                       (14, 2, GREEN, "14-pixel patches, 2×2 merged")):
        ax.plot(sides, [count_image_tokens(s, s, patch, merge) for s in sides], "o-", color=color, label=label, ms=4)
    ax.annotate("336 px → 576 tokens", xy=(336, 576), xytext=(130, 5200), arrowprops=dict(arrowstyle="->", color="#4b5563"))
    ax.set_xlabel("image side length (pixels)")
    ax.set_ylabel("tokens per image")
    ax.set_title("Double the side, quadruple the tokens")
    ax.legend(frameon=False)
    figs["image_tokens"] = fig

    # --- 3. Stage-1 alignment: training only the projector ----------------------
    model = build_toy_vlm()
    images, labels = toy_images(20)
    held_out = toy_images(20, seed=99)
    before = caption_accuracy(model, *held_out)
    history = align_projector(model, images, labels)
    after = caption_accuracy(model, *held_out)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.3))
    a1.plot(history["loss"], color=BLUE)
    a1.axhline(np.log(len(TEXT_VOCAB)), color=MUTED, ls="--")
    a1.text(len(history["loss"]) * 0.4, np.log(len(TEXT_VOCAB)) - 0.2, "blind guess over 12 words", color="#4b5563")
    a1.set_xlabel("training step")
    a1.set_ylabel("caption cross-entropy")
    a1.set_title("Only the projector learns")
    a2.plot(history["accuracy"], color=GREEN, label="training images")
    a2.axhline(0.25, color=MUTED, ls="--", label="chance (4 classes)")
    a2.scatter([0, len(history["accuracy"]) - 1], [before, after], color=RED, zorder=3, label="new images")
    a2.set_ylim(0, 1.05)
    a2.set_xlabel("training step")
    a2.set_ylabel("right caption word ranked first")
    a2.set_title(f"Caption accuracy: {before:.0%} → {after:.0%} on new images")
    a2.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    figs["alignment"] = fig

    # --- 4. Waveform, spectrogram, log-mel spectrogram --------------------------
    x = demo_signal()
    window, hop = 256, 64
    spec = stft(x, window, hop)
    mel = log_mel_spectrogram(x, SAMPLE_RATE, window, hop, n_mels=40)
    seconds = len(x) / SAMPLE_RATE
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(7.5, 7.2))
    t_ms = np.arange(160) / SAMPLE_RATE * 1000
    a1.plot(t_ms, x[:160], color=BLUE, lw=1)
    a1.set_xlabel("time (milliseconds): the first 20 ms only")
    a1.set_ylabel("pressure")
    a1.set_title("Waveform: 8,000 numbers a second, pitch hidden in the wiggles")
    a2.imshow(20 * np.log10(spec.T + 1e-6), origin="lower", aspect="auto", cmap="magma",
              extent=[0, seconds, 0, SAMPLE_RATE / 2], vmin=-40)
    a2.set_ylabel("frequency (Hz)")
    a2.set_xlabel("time (seconds)")
    a2.set_title(f"Spectrogram: {spec.shape[0]} frames × {spec.shape[1]} frequency bins")
    a2.grid(False)
    a3.imshow(mel.T, origin="lower", aspect="auto", cmap="magma", extent=[0, seconds, 0, 40], vmin=mel.max() - 5)
    a3.set_ylabel("mel band")
    a3.set_xlabel("time (seconds)")
    a3.set_title(f"Log-mel spectrogram: {mel.shape[0]} frames × {mel.shape[1]} bands")
    a3.grid(False)
    fig.tight_layout()
    figs["spectrogram"] = fig

    # --- 5. The mel scale and its filters ----------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.3))
    hz = np.linspace(0, 8000, 400)
    a1.plot(hz, hz_to_mel(hz), color=PURPLE)
    a1.plot(hz, hz, color=MUTED, ls="--", label="if mel were hertz")
    a1.scatter([1000], [1000], color=RED, zorder=3)
    a1.annotate("1000 Hz = 1000 mel", xy=(1000, 1000), xytext=(2200, 400), arrowprops=dict(arrowstyle="->", color="#4b5563"))
    a1.set_xlabel("frequency (Hz)")
    a1.set_ylabel("mel")
    a1.set_title("Mel: even steps low, squeezed high")
    a1.legend(frameon=False)
    bank = mel_filterbank(10, 512, 16_000)
    freqs = np.arange(bank.shape[1]) * 16_000 / 512
    for row in bank:
        a2.plot(freqs, row)
    a2.set_xlabel("frequency (Hz)")
    a2.set_ylabel("weight")
    a2.set_title("10 mel filters: narrow low, wide high")
    fig.tight_layout()
    figs["mel"] = fig

    # --- 6. Codebook size vs reconstruction error, and what it looks like -----------
    rows = codebook_sweep()
    patches = np.concatenate([image_to_patches(im, 4) for im in toy_images(30)[0]])
    books = learn_residual_codebooks(patches, k=8, stages=4)
    new = np.concatenate([image_to_patches(im, 4) for im in toy_images(30, seed=99)[0]])
    stage_err = [float(np.mean((new - residual_quantize(new, books[:s])[1]) ** 2)) for s in range(1, 5)]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bits = [r["bits"] for r in rows]
    ax.plot(bits, [r["random"] for r in rows], "o-", color=RED, label="random codebook")
    ax.plot(bits, [r["learned"] for r in rows], "o-", color=BLUE, label="learned codebook (k-means)")
    ax.plot([3 * s for s in range(1, 5)], stage_err, "s--", color=GREEN, label="residual: 1 to 4 stages of 8 entries")
    ax.axhline(0.09, color=MUTED, ls=":")
    ax.text(0.1, 0.078, "clean pattern, noise dropped (0.3² = 0.09)", color="#4b5563")
    ax.set_xlabel("bits per patch  (log₂ of codebook size, summed over stages)")
    ax.set_ylabel("mean squared error per pixel")
    ax.set_title("A learned codebook, and a second one for the leftovers")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    figs["codebook"] = fig

    # --- 7. How fast each modality spends a context window ------------------------
    durations = np.logspace(0, np.log10(7200), 60)
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [MUTED, GREEN, BLUE, AMBER, RED]
    for (label, tps), color in zip(TOKENS_PER_SECOND.items(), colors):
        ax.loglog(durations, durations * tps, color=color, label=label)
    for budget, name in ((128_000, "128k-token window"), (1_000_000, "1M-token window")):
        ax.axhline(budget, color="#4b5563", ls="--", lw=1)
        ax.text(1.2, budget * 1.25, name, color="#4b5563")
    ax.set_xlabel("length of the recording (seconds; 60 = a minute, 3600 = an hour)")
    ax.set_ylabel("tokens")
    ax.set_title("How fast each kind of input fills a context window")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    figs["token_budget"] = fig

    return figs


# ---------------------------------------------------------------------------
# 7. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. The one idea: every modality becomes a row of vectors")
    worked = worked_example_patches()
    say("A 4×4 picture with brightnesses 0 to 15, cut into four 2×2 patches and flattened:")
    matrix("patches (one row per patch)", worked["patches"])
    say("Multiply by W_E (column 1 adds the top row of a patch, column 2 the bottom row), then add (row, column) positions:")
    table(["patch", "pixels", "x·W_E", "+ position", "token"],
          [(name, p.tolist(), q.tolist(), pos.tolist(), t.tolist()) for name, p, q, pos, t in
           zip(["top left", "top right", "bottom left", "bottom right"], worked["patches"], worked["projected"], WORKED_POSITIONS, worked["tokens"])])
    takeaway("A picture is now four tokens of two numbers each: exactly the kind of input a transformer reads.")

    banner("2. How many tokens is an image?")
    table(["image", "patch", "merge", "tokens"],
          [(f"{s}×{s}", p, m, count_image_tokens(s, s, p, m)) for s, p, m in ((224, 16, 1), (224, 14, 1), (336, 14, 1), (1008, 14, 2))])
    say("Tokens grow with the square of the side: twice the resolution, four times the tokens.")
    encoder = VisionEncoder(image_size=32, patch=8, channels=1, d_model=24, n_layers=2, n_heads=4)
    say(f"A from-scratch ViT reads the 32×32 toy picture and returns {encoder(toy_picture()).shape}: 16 patches, 24 numbers each.")

    banner("3. Plugging vision into a language model")
    model = build_toy_vlm()
    image = toy_images(1)[0][3]
    ids = np.array([TEXT_VOCAB.index(w) for w in ["what", "is", "this", "<image>"]])
    sequence, is_image = model.splice(ids, image)
    say(f"Prompt ids {ids.tolist()} ('what is this <image>'). The image placeholder expands into "
        f"{model.vision.n_patches} projected vectors, so the language model reads {len(sequence)} rows of width {sequence.shape[1]}.")
    say(f"Which rows are image tokens: {is_image.astype(int).tolist()}. Logits come out as {model(ids, image).shape}: one score per word per position.")
    held_out = toy_images(20, seed=99)
    before = caption_accuracy(model, *held_out)
    history = align_projector(model, *toy_images(20))
    after = caption_accuracy(model, *held_out)
    say(f"Stage 1 (alignment): train only the projector so each image names its pattern. Loss {history['loss'][0]:.2f} → "
        f"{history['loss'][-1]:.2f}; on 80 new images the right word ranks first {before:.0%} → {after:.0%} of the time.")
    takeaway("The vision encoder and the language model never changed: a small translator between them did all the learning.")

    banner("4. Audio: a waveform becomes a spectrogram becomes tokens")
    wave = [1, 0, -1, 0, 1, 0, -1, 0]
    say(f"8 samples of a wave that cycles twice: {wave}. How much of each frequency: {np.round(dft_magnitudes(wave), 3).tolist()}.")
    x = demo_signal()
    spec = stft(x, 256, 64)
    say(f"One second of a rising whistle over a 1000 Hz hum at {SAMPLE_RATE} samples a second is {len(x)} numbers. "
        f"Windows of 256 every 64 samples give a spectrogram of {spec.shape} (frames, frequencies).")
    peaks = spec.argmax(axis=1) * SAMPLE_RATE / 256
    say(f"Loudest frequency in the first, middle and last frames: {peaks[0]:.0f}, {peaks[len(peaks) // 2]:.0f}, {peaks[-1]:.0f} Hz: the whistle climbing.")
    table(["hertz", "mel"], [(f, float(hz_to_mel(f))) for f in (100, 700, 1000, 4000, 8000)], floatfmt=".1f")
    say(f"A speech encoder with 10 ms frames and one stride-2 layer sees {audio_tokens(1)} tokens a second, {audio_tokens(30)} for 30 seconds.")

    banner("5. Discrete tokens: snapping vectors to a codebook")
    codebook = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    z = np.array([[0.9, 0.2], [0.1, 0.8], [0.1, -0.1]])
    say(f"Codebook {codebook.tolist()}. Vectors {z.tolist()} snap to ids {quantize(z, codebook).tolist()}.")
    table(["codebook size", "bits", "learned error", "random error"], [(r["k"], r["bits"], r["learned"], r["random"]) for r in codebook_sweep()], floatfmt=".3f")
    takeaway("Once a patch or a slice of sound is a codebook number, a language model can predict it exactly as it predicts a word.")

    banner("6. Video, and what it costs")
    table(["clip", "sampled fps", "tubelet", "tokens"],
          [("10 s", fps, t, video_tokens(10, fps, 256, t)) for fps, t in ((30, 1), (2, 2), (1, 1))])
    say(f"Frames kept from 300 frames at 30 fps, sampled at 1 fps: {sample_frames(300, 30, 1).tolist()}.")
    table(["input", "tokens per second", "minutes in 128k tokens"],
          [(k, v, seconds_that_fit(128_000, v) / 60) for k, v in TOKENS_PER_SECOND.items()], floatfmt=".1f")
    takeaway("Text is by far the densest way to hold information in a context window; pictures, sound and video pay per pixel and per second.")


if __name__ == "__main__":
    demo()
