r"""
# Looking inside the model

Run: `python -m primer.ml.interpretability`

## Why look inside at all

**Everyday picture.** A car makes a strange noise. You can take it for a
test drive and note when the noise happens: that is testing the car's
*behaviour*. Or you can open the hood, put a stethoscope on the engine,
and swap parts until the noise stops: that is looking at the *mechanism*.
Test drives tell you *what* the car does. Only the open hood tells you
*why*.

Every other lesson in this primer treats a trained model as something to
build, train or call. Evaluations (`primer.agents.evals`) are test drives:
they measure what the model says. **Interpretability** opens the hood. It
asks what the model's billions of internal numbers represent, and which of
them cause the answer. Three reasons to care:

- **Debugging.** When a model gets something wrong, you want to know which
  step failed, the same way you read a stack trace rather than just the
  error message.
- **Trust.** A model can be right for the wrong reason. An image classifier
  that "detects" wolves by looking for snow in the background scores well
  until it meets a wolf on grass.
- **Safety.** The output is only part of what a model computes. Some
  questions (is it relying on a stereotype? does it know more than it
  says?) can only be answered by reading the computation itself.

**A tiny worked example.** This lesson asks four questions of one kind of
sentence, "the Eiffel Tower is in …" → "Paris", each with its own tool, on
toy models small enough to check by hand:

| Question | Tool | What our toy shows |
|---|---|---|
| Is a property stored in this hidden state? | a **probe** | sentiment, read at 99% on unseen examples |
| What would the model say if it stopped at layer ℓ? | the **logit lens** | "London", then "Paris", then "Paris" |
| Which activations *cause* the answer? | **activation patching** | the subject word early, the last word late |
| What are the model's units of meaning? | **superposition** and **sparse autoencoders** | five features packed into two neurons, then recovered |

```mermaid
flowchart LR
  M["A trained model<br/>(frozen)"] --> A["Hidden activations<br/>at every layer and word"]
  A --> P["Probe:<br/>is property X in here?"]
  A --> L["Logit lens:<br/>what would it predict now?"]
  A --> AP["Activation patching:<br/>does this activation cause the answer?"]
  A --> S["Sparse autoencoder:<br/>which features make up this state?"]
  P & L --> R["Reading<br/>(correlation)"]
  AP --> C["Intervening<br/>(causation)"]
  S --> U["Units of analysis<br/>(what to read and intervene on)"]
```

**Reading it:** every tool starts from the same place: the numbers a frozen
model computes inside itself while it runs. The top two tools only *read*
those numbers, so what they find is a correlation. Patching *changes* them
and watches the output, so what it finds is a cause. Sparse autoencoders
answer an earlier question: before you can read or change "a feature", you
need to know what the features are. Keep the reading/intervening split in
mind; it is the most important distinction in this lesson.

## A feature is a direction

**Everyday picture.** A band with two instruments plays through two
speakers. The sound engineer pans the guitar mostly to the right speaker
and the piano mostly to the left, but each speaker plays a *mix* of both.
If you unplug one speaker you don't lose "the guitar"; you lose part of
everything. The instruments are not the speakers. Each instrument is a
*setting across all speakers*, a direction, and the sound in the room is
the sum of every instrument times how loudly it plays.

A model's hidden state works the same way. The speakers are **neurons**:
the individual numbers in a layer's output vector. The instruments are
**features**: the things the model has learned to track, such as "this
review is positive" or "this verb is in the past tense". Each feature is
stored as a **direction** in the space of neurons, and the hidden state is
the sum of every active feature's direction, scaled by how strongly it is
present.

**A tiny worked example.** Take a hidden layer with two neurons and two
features whose directions are "positive" = (0.6, 0.8) and "past tense" =
(0.8, −0.6). A sentence that is quite positive (0.9) and a little
past-tense (0.4) has the hidden state

0.9 · (0.6, 0.8) + 0.4 · (0.8, −0.6) = (0.54 + 0.32, 0.72 − 0.24) = **(0.86, 0.48)**.

Neuron 1 reads 0.86 and neuron 2 reads 0.48. Neither number is "how
positive" or "how past-tense": each neuron is a blend of both features.
To get the features back, take the **dot product** (multiply matching
entries and add; see `primer.notation`) of the hidden state with each
direction:

- positive: 0.6 · 0.86 + 0.8 · 0.48 = 0.516 + 0.384 = **0.9**
- past tense: 0.8 · 0.86 − 0.6 · 0.48 = 0.688 − 0.288 = **0.4**

$$
h = \sum_{i=1}^{k} f_i \, d_i
\qquad\qquad
\hat{f}_i = d_i \cdot h
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $h$ | the hidden state: one number per neuron | (0.86, 0.48) |
| $k$ | how many features are present | 2 |
| $i$ | which feature | 1 = positive, 2 = past tense |
| $f_i$ | how strongly feature $i$ is present | $f_1 = 0.9$, $f_2 = 0.4$ |
| $d_i$ | feature $i$'s direction: a list with one number per neuron, of length 1 | $d_1 = (0.6, 0.8)$ |
| $\sum_{i=1}^{k}$ | add up one term per feature | two terms |
| $\hat{f}_i$ | the amount of feature $i$ we read back (the hat means "estimated") | 0.9 |
| $\cdot$ | dot product: multiply matching entries, then add | |

**In words:** "the hidden state is the sum of each feature's direction
times its strength; to read a feature back, dot the hidden state with that
feature's direction."

**With the numbers:** $h = 0.9 \cdot (0.6, 0.8) + 0.4 \cdot (0.8, -0.6) =
(0.86, 0.48)$, and $\hat{f}_1 = (0.6, 0.8) \cdot (0.86, 0.48) = 0.9$. The
read-back is exact here because the two directions are **perpendicular**
(their dot product is 0.6 · 0.8 + 0.8 · (−0.6) = 0) and each has length 1.
Hold on to that condition: the section on superposition is about what
happens when it fails.

**In Python:**

```python
positive = [0.6, 0.8]
past = [0.8, -0.6]
# h = Σ f_i d_i
h = [0.9 * p + 0.4 * q for p, q in zip(positive, past)]
[round(x, 2) for x in h]  # → [0.86, 0.48]
# f̂_i = d_i · h
round(sum(d * x for d, x in zip(positive, h)), 2)  # → 0.9
round(sum(d * x for d, x in zip(past, h)), 2)  # → 0.4
# the directions are perpendicular: their dot product is 0
round(sum(p * q for p, q in zip(positive, past)), 2)  # → 0.0
```

![Two perpendicular arrows for the positive and past-tense directions, and the hidden state (0.86, 0.48) as their weighted sum, with dotted lines dropping onto each arrow at 0.9 and 0.4](figures/primer.ml.interpretability.directions.svg)

**Reading it:** the axes are the two neurons. The blue and green arrows are
the two feature directions; the red arrow is the hidden state. The dotted
lines drop from the hidden state onto each feature arrow, and where they
land (0.9 of the way along blue, 0.4 along green) is the dot product: the
amount of that feature. Notice that the red arrow's coordinates on the
*neuron* axes, 0.86 and 0.48, are neither amount. To read a model you have
to look along the right directions, not along the neurons.

That features are directions, and that most of what a model tracks can be
read with a dot product, is called the **linear representation
hypothesis**. It is a hypothesis, not a law, but it holds often enough to
power everything below. You have met it before: word-vector arithmetic
such as king − man + woman ≈ queen (`primer.ml.embeddings.word2vec`) works
because "royalty" and "gender" are directions.

**In code:** `compose_features` builds a hidden state from feature amounts and `read_features` reads them back with dot products.

## Probes: can a straight line read it out?

**Everyday picture.** A doctor can't ask your liver how it is doing, but a
blood test can measure a marker that tells them. A **probe** is a blood
test for a hidden state: a tiny classifier, trained by you, that looks at a
layer's activations and answers one yes/no question, such as "is this
review positive?". The model itself is frozen; only the probe learns.

**A tiny worked example.** A probe for "positive" on a two-neuron layer
has weights w = (1, 0.5) and bias b = 0. On the hidden state h = (2, −1):

1. Score it: 1 · 2 + 0.5 · (−1) + 0 = 1.5.
2. Squash the score into a probability with the **sigmoid** σ(z) =
   1 / (1 + e^−z), which maps any number into the range 0 to 1:
   σ(1.5) = 1 / (1 + 0.223) = **0.818**.

The probe is 82% sure this hidden state belongs to a positive review. On
h = (−1, 1) the score is −1 + 0.5 = −0.5 and σ(−0.5) = 0.378: probably
negative.

$$
p = \sigma(w \cdot h + b) = \frac{1}{1 + e^{-(w \cdot h + b)}}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $h$ | the frozen model's hidden state for one example | (2, −1) |
| $w$ | the probe's weights: one per neuron, learned | (1, 0.5) |
| $b$ | the probe's bias: one number, learned | 0 |
| $w \cdot h + b$ | the probe's raw score (its **logit**) | 1.5 |
| $\sigma$ | the sigmoid: turns any score into a probability between 0 and 1 | |
| $e$ | Euler's number, about 2.718 | $e^{-1.5} = 0.223$ |
| $p$ | the probe's probability that the property is present | 0.818 |

**In words:** "dot the hidden state with the probe's weights, add the
bias, and squash the result into a probability."

**With the numbers:** $p = \sigma(1 \cdot 2 + 0.5 \cdot (-1) + 0) =
\sigma(1.5) = 1 / (1 + 0.223) = 0.818$.

**In Python:**

```python
import math
w = [1.0, 0.5]
b = 0.0
h = [2.0, -1.0]
# w · h + b
z = sum(wi * hi for wi, hi in zip(w, h)) + b
z  # → 1.5
# σ(z)
round(1 / (1 + math.exp(-z)), 3)  # → 0.818
# a second hidden state, (−1, 1), reads as probably negative
round(1 / (1 + math.exp(-(-1 * 1.0 + 1 * 0.5))), 3)  # → 0.378
```

This is exactly **logistic regression** (`primer.ml.neural_net`), trained
the usual way: show it examples whose answer you know, measure its
binary cross-entropy (`primer.ml.losses`), and nudge w and b downhill. The
gradient of that loss with respect to the score is simply (p − y), the gap
between the probe's probability and the true 0/1 label, which makes each
update one line of code.

```mermaid
flowchart LR
  T["Text with a known label<br/>(positive or negative)"] --> M["Frozen model<br/>(no weights change)"]
  M --> H["Hidden state h<br/>at the layer under study"]
  H --> P["Probe: σ(w · h + b)"]
  P --> Y["Probability the label is 'positive'"]
  Y --> G["Compare with the true label<br/>(cross-entropy)"]
  G -. "gradient updates w and b only" .-> P
```

**Reading it:** the solid arrows are one forward pass; the dotted arrow is
learning. The gradient stops at the probe: the model under study never
changes, so whatever the probe finds was already in the hidden state. That
is the point of a probe. It measures the model, not a new model trained on
top of it.

**On our toy model.** `PlantedModel` stores two known features in a
32-neuron hidden layer: sentiment and tense, each as ±1 along its own
random direction, plus random noise of size 0.4 on every neuron. No single
neuron holds either feature. A probe trained on just 100 examples reads
sentiment correctly on **99%** of 1,000 examples it never saw.

**Why keep probes simple, and check them against a control.** A probe can
succeed for the wrong reason. Give a probe **random labels**, a *control
task* with nothing real to find, and it still fits **73%** of its 100
training examples, because 33 adjustable numbers can memorize a lot of
noise. On unseen examples it scores **50%**, a coin flip. Two habits
follow: always score a probe on examples it never saw, and compare it with
a control task, so that "the probe works" means "the hidden state encodes
this" rather than "the probe is clever". This is also why probes are kept
*linear*: a powerful probe (a deep network) could compute the property
from raw ingredients by itself, and then its success would tell you about
the probe, not the model.

### Decodable is not the same as used

**Everyday picture.** A library holds a book nobody ever borrows. Finding
it on the shelf proves the library *has* it, not that it shaped anything
any reader did.

**A tiny worked example.** Our toy model's output reads sentiment and, by
construction, gives tense a weight of exactly zero. A probe still reads
tense at **98%** on unseen examples. Now intervene: take 200 examples, flip
only one feature's label, keep everything else fixed, and measure how much
the model's output moves.

| Flip | Probe accuracy for it | Output moves by |
|---|---|---|
| sentiment | 99% | **2.00** |
| tense | 98% | **0.00** |

![Left, probe accuracy: sentiment 0.99 and tense 0.98 on unseen examples, random labels 0.73 on training examples but about 0.5 on unseen ones. Right, flipping sentiment moves the output by 2 and flipping tense moves it by 0](figures/primer.ml.interpretability.probe_vs_use.svg)

**Reading it:** the left panel is what probes can read. Grey bars are
accuracy on the probe's own training examples, blue bars on unseen ones,
and the red dashed line is chance. Sentiment and tense are both clearly
readable; random labels look readable on the training set and collapse to
chance on unseen data, which is exactly what the control is for. The right
panel is what the model *uses*: the change in its output when one feature
is flipped. Tense is as readable as sentiment and has no effect at all.
The two panels answer different questions, and only the right one is about
cause.

A probe finding information does not mean the model uses it. To find out
what the model uses you have to change something and watch the output,
which is what the rest of this lesson does.

**In code:** `train_probe` fits a `Probe` by gradient descent on frozen hidden states from `PlantedModel`; `probe_report` trains the three probes, and `flip_effect` performs the intervention.

## The logit lens: reading the model's mind mid-thought

**Everyday picture.** A writer keeps every draft of an article. Draft 1
says the landmark is "in a European capital", draft 2 says "probably
Paris", the final says "Paris". Reading the drafts shows *when* the writer
made up their mind. The **logit lens** reads a language model's drafts.

It works because of how a transformer is built (`primer.ml.transformer`).
Each word has a running vector, the **residual stream**. Every layer reads
it and *adds* a correction to it, rather than replacing it. At the very
end, the output layer (the **unembedding**) turns the final vector into
one score per word in the vocabulary. Because every layer writes into the
same stream, you can apply that final step early, after any layer, and ask
"what would the model predict if it stopped here?"

**A tiny worked example.** A vocabulary of three words, Paris, London and
Rome, a residual stream of two numbers, and an unembedding that scores
Paris = first number, London = second number, Rome = minus the first:

| After | Residual $h$ | Scores (Paris, London, Rome) | Probabilities | Top guess |
|---|---|---|---|---|
| the embedding | (0.2, 0.3) | (0.2, 0.3, −0.2) | (0.36, 0.40, 0.24) | London |
| layer 1, which adds (0.6, 0) | (0.8, 0.3) | (0.8, 0.3, −0.8) | (0.55, 0.34, 0.11) | Paris |
| layer 2, which adds (1.2, −0.3) | (2.0, 0.0) | (2.0, 0.0, −2.0) | (0.87, 0.12, 0.02) | Paris |

The first draft is a vague "some capital" that happens to lean London;
layer 1 tips it to Paris; layer 2 commits.

$$
\text{lens}_\ell = \text{softmax}\big(W_U \, \text{LN}(h_\ell)\big)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $\ell$ | which layer we stop after (0 = straight after the embedding) | 2 |
| $h_\ell$ | the residual stream after layer $\ell$, for one word | (2.0, 0.0) |
| $\text{LN}$ | the model's final normalization (`primer.ml.transformer`); our 2-number toy has none, so here LN(h) = h | (2.0, 0.0) |
| $W_U$ | the unembedding: one row per vocabulary word, dotted with the state to give that word's score | rows (1, 0), (0, 1), (−1, 0) |
| $W_U \, \text{LN}(h_\ell)$ | the scores (**logits**), one per word | (2, 0, −2) |
| softmax | turns scores into probabilities that add up to 1 (`primer.ml.attention`) | (0.87, 0.12, 0.02) |
| $\text{lens}_\ell$ | what the model would predict if it stopped after layer $\ell$ | Paris at 0.87 |

**In words:** "take the residual stream partway up, push it through the
model's own final normalization and output layer, and read off the
probabilities."

**With the numbers:** after layer 2, $W_U h_2 = (1 \cdot 2 + 0 \cdot 0,\;
0 \cdot 2 + 1 \cdot 0,\; -1 \cdot 2 + 0 \cdot 0) = (2, 0, -2)$; then
$e^2 = 7.39$, $e^0 = 1$, $e^{-2} = 0.14$, total 8.52, so P(Paris) =
7.39 / 8.52 = 0.87.

**In Python:**

```python
import math
# rows: Paris, London, Rome
W_U = [[1, 0], [0, 1], [-1, 0]]
h = [0.2, 0.3]
writes = [[0.6, 0.0], [1.2, -0.3]]
# a residual layer adds its write to the stream
for w in writes:
    h = [a + b for a, b in zip(h, w)]
[round(x, 2) for x in h]  # → [2.0, 0.0]
# W_U h: one score per word
scores = [sum(u * x for u, x in zip(row, h)) for row in W_U]
[round(s, 2) for s in scores]  # → [2.0, 0.0, -2.0]
# softmax
exps = [math.exp(s) for s in scores]
[round(e / sum(exps), 2) for e in exps]  # → [0.87, 0.12, 0.02]
```

```mermaid
flowchart LR
  E["Embedding"] --> H0(("h₀")) --> L1["Layer 1<br/>adds its write"] --> H1(("h₁")) --> L2["Layer 2<br/>adds its write"] --> H2(("h₂")) --> OUT["Final norm + W_U<br/>(the real output)"]
  H0 -.-> LENS0["lens: norm + W_U<br/>London 0.40"]
  H1 -.-> LENS1["lens: norm + W_U<br/>Paris 0.55"]
  H2 -.-> LENS2["lens: norm + W_U<br/>Paris 0.87"]
```

**Reading it:** the solid line along the middle is the residual stream: it
flows from the embedding to the real output, and each layer adds to it.
The dotted taps hang the model's own output layer off the stream after
every layer. Nothing is trained; the lens borrows weights the model
already has. At the top layer the tap and the real output are the same
computation, so they must agree exactly, which is a good test that the
lens is wired correctly (on `primer.ml.transformer.TinyGPT`, they do).

**On a model where we know the answer.** `LandmarkModel` is a two-layer,
one-head transformer built by hand to complete "Eiffel is in" with
"Paris". Layer 1 is an MLP that looks up a fact at every word (landmark
in, city out); layer 2 is an attention head that lets the last word, "in",
find the landmark and copy its city.

```mermaid
flowchart LR
  T["Eiffel · is · in"] --> EMB["Embed each word"]
  EMB --> MLP["Layer 1: MLP at every word<br/>Eiffel → writes 'Paris'"]
  MLP --> ATT["Layer 2: attention head<br/>'in' looks for the landmark,<br/>copies its city"]
  ATT --> U["Output at 'in':<br/>Paris 3, Rome 0, London 0"]
```

**Reading it:** read left to right as the two steps of the answer. The
fact ("the Eiffel Tower is in Paris") is looked up at the landmark's own
position in layer 1. It is only moved to the last position, the one that
predicts the next word, in layer 2. We built it this way on purpose, so
every tool below can be checked against a known truth.

![Left, the worked example's probabilities for Paris, London and Rome after the embedding, layer 1 and layer 2. Right, a 3 by 3 grid of the lens's probability of Paris by layer and word: the Eiffel column turns dark after layer 1, the in column only after layer 2](figures/primer.ml.interpretability.logit_lens.svg)

**Reading it:** the left panel is the worked example: at each layer, three
bars for the three words, and the blue Paris bar grows from 0.36 to 0.87.
The right panel runs the lens over the landmark model: rows are layers,
columns are words, and each cell is the lens's probability of "Paris"
(1/3 = 0.33 means no opinion among three cities). The Eiffel column goes
dark after layer 1, when the MLP looks up the fact. The "in" column only
goes dark after layer 2, when attention copies the city over. The lens has
shown *where* and *when* the answer appears, without being told.

The logit lens was first described for GPT-2 in 2020, where middle layers
already "guess" the next word surprisingly well. In some models the early
layers read as nonsense through the final output layer, because they do
not yet speak its "language"; the **tuned lens** fixes this by training a
small translator for each layer before applying the output layer.

**In code:** `logit_lens` applies an output layer to any residual states, `worked_lens` runs the table above, `residual_stream` and `tinygpt_logit_lens` do the same for `primer.ml.transformer.TinyGPT`, and `lens_map` builds the grid for `LandmarkModel`.

## Activation patching: which activations cause the answer?

**Everyday picture.** Two cars of the same model sit side by side: one
starts, one doesn't. You move parts from the good car into the bad one,
one part at a time, and try the ignition after each swap. The part whose
swap makes the bad car start is the part that mattered. **Activation
patching** (also called **causal tracing**) does this with a model's
internal activations.

**A tiny worked example.** Run the landmark model twice:

- the **clean** prompt "Eiffel is in": scores Paris 3, Rome 0, so the
  **logit difference** Paris − Rome is **+3**;
- the **corrupted** prompt "Colosseum is in": Paris 0, Rome 3, so the
  difference is **−3**.

Now run the corrupted prompt again, but at one chosen place overwrite the
activation with the one from the clean run, and see how much of the gap
between −3 and +3 comes back:

- patch the MLP output at the landmark's position: the difference jumps
  back to **+3**, so 100% of the answer is restored;
- patch the state at "is": it stays at **−3**, 0% restored ("is" is the
  same word in both prompts, so its state carries nothing about the
  landmark).

$$
\text{restored} = \frac{LD_{\text{patched}} - LD_{\text{corrupt}}}{LD_{\text{clean}} - LD_{\text{corrupt}}}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $LD$ | **logit difference**: the correct answer's score minus the wrong answer's score (Paris − Rome) | |
| $LD_{\text{clean}}$ | on the clean prompt, nothing patched | +3 |
| $LD_{\text{corrupt}}$ | on the corrupted prompt, nothing patched | −3 |
| $LD_{\text{patched}}$ | on the corrupted prompt, with one clean activation copied in | +3, −3, or anything between |
| restored | the fraction of the gap that one patch closes: 0 = nothing, 1 = everything | 1, 0 |

**In words:** "how far the patch moved the answer, as a share of the full
distance from the corrupted answer to the clean one."

**With the numbers:** patching the landmark's MLP output gives
(3 − (−3)) / (3 − (−3)) = 6 / 6 = 1. Patching "is" gives
(−3 − (−3)) / 6 = 0. A patch that left the difference at 0 would give
(0 − (−3)) / 6 = 0.5: half the answer restored.

**In Python:**

```python
clean, corrupt = 3.0, -3.0
# restored = (LD_patched − LD_corrupt) / (LD_clean − LD_corrupt)
def restored(patched):
    return (patched - corrupt) / (clean - corrupt)
# the landmark's MLP output, patched in
restored(3.0)  # → 1.0
# the state at "is", patched in
restored(-3.0)  # → 0.0
# a patch that only reaches a tie
restored(0.0)  # → 0.5
```

Why a *difference* of two scores, rather than the probability of Paris?
Because the corrupted prompt differs from the clean one in exactly one
fact, the difference measures exactly that fact, and it moves smoothly,
where a probability can saturate near 0 or 1 and hide a change.

```mermaid
flowchart TB
  subgraph C["1. Clean run: 'Eiffel is in'"]
    c1["save every activation"] --> c2["Paris − Rome = +3"]
  end
  subgraph K["2. Corrupted run: 'Colosseum is in'"]
    k1["Paris − Rome = −3"]
  end
  subgraph P["3. Patched run: corrupted prompt,<br/>ONE activation taken from the clean run"]
    p1["Paris − Rome = ?"]
  end
  c1 -- "copy one activation" --> P
  P --> R["fraction restored =<br/>(? − (−3)) / (3 − (−3))"]
  K --> R
  C --> R
```

**Reading it:** there are three runs. The clean run is a donor: its
activations are saved. The corrupted run sets the baseline. The patched
run is the corrupted prompt with a single activation transplanted from
the donor. Repeat the patched run once per layer and position and you get
one "fraction restored" per site: a map of where the answer is carried.

![A 3 by 3 grid, layers by positions, of the fraction of the answer restored by one patch: 1 at the Eiffel column for the embedding and layer 1, 1 at the in column after layer 2, 0 everywhere else](figures/primer.ml.interpretability.patching.svg)

**Reading it:** rows are the residual stream after each layer, columns are
the three positions (clean word / corrupted word), and each cell is the
fraction of the answer restored by patching that one state. The answer
lives at the landmark early (the Eiffel column is 1 after the embedding
and after the MLP) and at the last word late (the "in" column is 1 after
attention). The diagonal hand-off between them is the two-step circuit we
built, found by intervention alone.

Compare with the lens grid above. After layer 2, the lens reads "Paris"
at the Eiffel position with probability 0.995, yet patching that state
restores **0.00**: after the last layer nothing reads that position any
more. The information is there, and it no longer matters. Readable is not
the same as used, again.

**Why it matters in practice.** This is how researchers located where
GPT-style models store facts: causal tracing on prompts like ours found
factual recall concentrated in middle-layer MLPs at the subject's last
token, and then used that to edit single facts. The same method, one site
at a time, has traced whole circuits, such as the one GPT-2 small uses to
fill in "When Mary and John went to the store, John gave a drink to" →
"Mary".

**In code:** `LandmarkModel` is the hand-built model and `LandmarkModel.run` accepts patches; `fraction_restored` is the formula and `patching_map` patches every layer and position in turn.

## Superposition: more features than neurons

**Everyday picture.** Back to the band, but now five instruments play
through two speakers. The engineer pans each instrument to its own
position around the room: hard left, front right, back left, and so on.
When one instrument plays alone, you can tell which one by where the sound
comes from. When two play at once, the positions blur together and you
might mistake the pair for a third instrument. The trick works because in
this band, most of the time, only one instrument is playing.

Models face the same squeeze. There are far more concepts in the world
than neurons in a layer, but in any one sentence almost all of them are
absent: features are **sparse**. So models store more features than they
have dimensions, as directions that are *nearly* perpendicular rather than
exactly. That is **superposition**.

**A tiny worked example.** Put five features in two neurons, spread 72°
apart like the points of a pentagon. Feature $k$'s direction is
(cos 72k°, sin 72k°): feature 0 is (1, 0), feature 1 is (0.309, 0.951),
and so on. Switch on feature 0 alone, at strength 1. The hidden state is
(1, 0). Reading every feature back with a dot product gives

| Feature | Angle from feature 0 | Read-back (cos of the angle) | After adding −0.31 and ReLU |
|---|---|---|---|
| 0 | 0° | **1.000** | 0.69 |
| 1 | 72° | 0.309 | 0 |
| 2 | 144° | −0.809 | 0 |
| 3 | 216° | −0.809 | 0 |
| 4 | 288° | 0.309 | 0 |

Five directions can't all be perpendicular in two dimensions, so reading
feature 0 leaks 0.309 into each neighbour: **interference**. The fix is a
small negative bias and a **ReLU** (which turns negatives into zero): the
leak of 0.309 falls below the bias of 0.31 and is filtered to 0, and the
real feature survives at 0.69. The cost comes when two neighbours are on
at once: each then reads 1 + 0.309 − 0.31 = 0.999 instead of 0.69, too
high. Sparsity is a bet that such collisions are rare.

$$
\hat{x} = \text{ReLU}\big(W^\top W x + b\big),
\qquad
\hat{x}_i = \text{ReLU}\Big(\|w_i\|^2 x_i + \sum_{j \neq i} (w_i \cdot w_j)\, x_j + b_i\Big)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $x$ | the true features: one strength per feature | (1, 0, 0, 0, 0) |
| $W$ | the squeeze: 2 rows (neurons) by 5 columns (features); column $i$ is feature $i$'s direction $w_i$ | the pentagon |
| $W x$ | the hidden state: 2 numbers holding 5 features | (1, 0) |
| $W^\top$ | $W$ **transposed** (rows and columns swapped): reads each feature back with a dot product | |
| $b$ | one bias per feature, learned; negative values filter small leaks | −0.31 each |
| ReLU | keep positives, turn negatives into 0 | |
| $\hat{x}$ | the features read back out | (0.69, 0, 0, 0, 0) |
| $\|w_i\|^2$ | feature $i$'s direction dotted with itself (its squared length) | 1 |
| $w_i \cdot w_j$ | how much feature $j$ leaks into feature $i$: 0 if perpendicular | 0.309 for neighbours |
| $\sum_{j \neq i}$ | add up over every *other* feature $j$ | |

**In words:** "squeeze the features into a few neurons, read each one back
by dotting with its direction, subtract a threshold and drop anything
negative. What you read for feature $i$ is its own strength, plus a leak
from every other active feature, minus the threshold."

**With the numbers:** with feature 0 alone on, $\hat{x}_1 =
\text{ReLU}(1 \cdot 0 + 0.309 \cdot 1 - 0.31) = \text{ReLU}(-0.001) = 0$
and $\hat{x}_0 = \text{ReLU}(1 \cdot 1 - 0.31) = 0.69$. With features 0
and 1 both on, $\hat{x}_0 = \text{ReLU}(1 + 0.309 - 0.31) = 0.999$.

**In Python:**

```python
import math
# feature k's direction: (cos 72k°, sin 72k°)
W = [[math.cos(math.radians(72 * k)), math.sin(math.radians(72 * k))] for k in range(5)]
def read_back(x, bias, relu=True):
    # h = W x: two numbers
    h = [sum(W[k][n] * x[k] for k in range(5)) for n in range(2)]
    # Wᵀ h + b: one number per feature
    z = [W[k][0] * h[0] + W[k][1] * h[1] + bias for k in range(5)]
    # ReLU keeps positives and turns negatives into 0
    return [round(max(0.0, v) if relu else v, 3) for v in z]
# the leak: feature 0 alone, no bias, no ReLU
read_back([1, 0, 0, 0, 0], bias=0, relu=False)  # → [1.0, 0.309, -0.809, -0.809, 0.309]
# the bias and ReLU filter it
read_back([1, 0, 0, 0, 0], bias=-0.31)  # → [0.69, 0.0, 0.0, 0.0, 0.0]
# two neighbours on: each reads too high
read_back([1, 1, 0, 0, 0], bias=-0.31)  # → [0.999, 0.999, 0.0, 0.0, 0.0]
```

```mermaid
flowchart LR
  X["5 features<br/>x₀ … x₄<br/>(mostly zero)"] --> W["squeeze: W<br/>(2 × 5)"]
  W --> H["hidden state<br/>2 neurons"]
  H --> WT["read back: Wᵀ<br/>(5 × 2)"]
  WT --> B["+ bias b<br/>(negative)"]
  B --> R["ReLU"]
  R --> XH["5 features<br/>read back"]
```

**Reading it:** follow a feature through the bottleneck. Five numbers go
in, are squeezed into two, and must be expanded back into five. There is
no way to fit five perpendicular directions into two neurons, so the
read-back always leaks; the bias and ReLU at the end are what make the
leak survivable, by throwing away small readings. This is the toy model
from Anthropic's *Toy Models of Superposition* (2022), and the next step
is to train it and see what it chooses.

**Training it.** Let the model learn $W$ and $b$ itself, by gradient
descent on the reconstruction error, with features that matter less and
less (feature $i$ is weighted $0.8^i$). Compare two worlds:

- **Dense**: every feature is on in every example. The model keeps the two
  most important features, perpendicular, at length 1.00, and gives the
  other three length 0: it simply drops them.
- **Sparse**: each feature is on only 5% of the time. The model keeps
  **all five**, 72° apart, at length about 1.1, and learns a negative bias
  of about −0.23 to filter the leaks.

![Three panels. Left, dense features: two perpendicular arrows and three near-zero ones. Middle, sparse features: five arrows spread evenly around a pentagon. Right, bars of how much each of five features moves neuron 1 and neuron 2](figures/primer.ml.interpretability.superposition.svg)

**Reading it:** in the two left panels each arrow is one feature's learned
direction in the two-neuron space, and its length is how well the model
stores it. Dense features (left) get the textbook answer: as many features
as neurons, perpendicular, the rest discarded. Sparse features (middle)
get a pentagon: five features in two dimensions, accepting a little
interference in exchange for storing everything. The right panel reads the
ideal pentagon by *neuron*: neuron 1 moves for feature 0 (by 1.0), feature
1 and feature 4 (by 0.309 each), and moves the other way for features 2
and 3. No neuron belongs to one feature.

That last point is why looking at single neurons so often fails. A neuron
that responds to several unrelated features is **polysemantic**. Vision
researchers found neurons that respond to cat faces and to the fronts of
cars; language models are full of neurons like that. Superposition
explains why: when features outnumber neurons, the features cannot line up
with the neurons, so every neuron is a mixture.

**In code:** `pentagon` builds the five directions, `superposition_readout` is the formula, `train_superposition` learns W and b with gradients from `superposition_loss_and_grads` (checked against `finite_difference_gradient`), and `features_a_neuron_responds_to` lists a neuron's features.

## Sparse autoencoders: getting the features back

**Everyday picture.** A sound engineer receives the two-speaker recording
of the five-instrument band, with no notes on who played when. She knows
one thing about this band: usually only one instrument plays at a time.
Many different scores could produce the same sound, but she writes down
the one that uses the *fewest instruments*. That preference is what lets
her recover the real parts rather than some arbitrary mixture.

A **sparse autoencoder** (SAE) does this for a model's hidden states. It
learns a **dictionary**: many more directions than the layer has neurons
(it is *overcomplete*), trained so that every hidden state can be rebuilt
from just a few of them. Each learned direction is a candidate feature,
and each one's strength on an input is its **latent** activation.

**A tiny worked example.** The hidden state is h = (0.8, 0.6) and the
dictionary has three directions, (1, 0), (0, 1) and (0.8, 0.6). Two codes
rebuild h perfectly:

| Code (strength of each direction) | Rebuilt | Error | Penalty with λ = 0.1 | Loss |
|---|---|---|---|---|
| A: (0.8, 0.6, 0), two directions | (0.8, 0.6) | 0 | 0.1 × (0.8 + 0.6) = 0.14 | 0.14 |
| B: (0, 0, 1), one direction | (0.8, 0.6) | 0 | 0.1 × 1.0 = 0.10 | **0.10** |

Rebuilding alone can't choose between them. The penalty on the total
strength, the **L1 penalty** (`primer.ml.regularization`), prefers the code
that uses one direction. One side effect: the best strength for direction
3 is not quite 1. With strength $a$, the loss is $(1 - a)^2 + 0.1a$, which
is lowest at $a = 0.95$ (loss 0.0975): the penalty always pulls strengths
a little toward zero, known as **shrinkage**.

$$
f = \text{ReLU}\big(W_e (h - b_d) + b_e\big),
\qquad
\hat{h} = W_d f + b_d,
\qquad
L = \|h - \hat{h}\|^2 + \lambda \sum_{i=1}^{m} |f_i|
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $h$ | a hidden state from the model under study | (0.8, 0.6) |
| $m$ | how many latents (dictionary directions) the SAE has, usually far more than neurons | 3 |
| $W_e$, $b_e$ | the encoder's weights and biases: turn a hidden state into latent strengths | |
| $f$ | the code: one strength per latent, ReLU keeps them ≥ 0 and mostly exactly 0 | (0, 0, 1) |
| $W_d$ | the decoder: column $i$ is latent $i$'s direction, kept at length 1 | (1, 0), (0, 1), (0.8, 0.6) |
| $b_d$ | the decoder bias: the typical hidden state, subtracted before encoding and added back after | (0, 0) |
| $\hat{h}$ | the rebuilt hidden state | (0.8, 0.6) |
| $\|h - \hat{h}\|^2$ | squared length of the rebuild error: add up the squares of its entries | 0 |
| $\lambda$ | the sparsity penalty's strength, chosen by you | 0.1 |
| $\lvert f_i \rvert$ | the size of latent $i$'s strength | |
| $L$ | the loss to minimize: rebuild error plus penalty | 0.10 |

**In words:** "encode the hidden state into many non-negative strengths,
decode it back as a weighted sum of dictionary directions, and pay for
both the rebuild error and the total strength used."

**With the numbers:** for code B, $\hat{h} = 1 \cdot (0.8, 0.6) = (0.8,
0.6)$, the error is 0, the penalty is $0.1 \times 1 = 0.10$, so $L = 0.10$.
For code A the penalty is $0.1 \times 1.4 = 0.14$.

**In Python:**

```python
dictionary = [[1, 0], [0, 1], [0.8, 0.6]]
h = [0.8, 0.6]
lam = 0.1
def loss(code):
    # ĥ = Σ f_i · direction_i
    rebuilt = [sum(f * d[n] for f, d in zip(code, dictionary)) for n in range(2)]
    # ‖h − ĥ‖² + λ Σ |f_i|
    return round(sum((a - b) ** 2 for a, b in zip(h, rebuilt)) + lam * sum(abs(f) for f in code), 4)
loss([0.8, 0.6, 0])  # → 0.14
loss([0, 0, 1.0])  # → 0.1
# shrinkage: a slightly weaker code is cheaper still
loss([0, 0, 0.95])  # → 0.0975
```

```mermaid
flowchart LR
  H["hidden state h<br/>(d numbers)"] --> ENC["encoder<br/>ReLU(W_e(h − b_d) + b_e)"]
  ENC --> F["code f<br/>(m ≫ d numbers,<br/>almost all exactly 0)"]
  F --> DEC["decoder<br/>W_d f + b_d"]
  DEC --> HH["rebuilt ĥ"]
  HH --> LOSS["loss = ‖h − ĥ‖² + λ Σ|f|"]
  H --> LOSS
  F --> LOSS
```

**Reading it:** the hidden state is blown up into a much wider code and
squeezed back down. The loss watches two things at once: the rebuild must
match the original (the arrow from h), and the code must be small (the
arrow from f). Without the second arrow the SAE could use any directions
at all. With it, each input is explained by a handful of latents, and each
latent's decoder column is a candidate feature you can inspect: read which
inputs make it fire, and add its direction to the model to see what it
does.

**On the superposition toy.** Take 20,000 hidden states from the pentagon
model (each feature on 5% of the time), train an SAE with 5 latents and
λ = 0.3, and compare the learned decoder columns with the five planted
directions. Every planted feature is matched by a latent with cosine
similarity above **0.99** (1 would be the same direction), and an example
with a feature on lights up about **1.07** latents on average. The price
is shrinkage: **11%** of the variance goes unexplained. With a tiny
penalty, λ = 0.01, the SAE rebuilds the data perfectly (0.0% unexplained),
yet its worst match to a real feature is only **0.80**: any two directions
can rebuild a two-dimensional space, so without sparsity pressure nothing
forces the dictionary to find the model's actual features.

![Left, grey dots of hidden states lying mostly along five rays, with thick grey planted directions and red learned latent arrows lying on top of them. Right, as the penalty grows from 0.003 to 1, the worst feature match peaks near 0.99 at a penalty of 0.3, while the unexplained variance climbs from 0 to above 0.5](figures/primer.ml.interpretability.sae.svg)

**Reading it:** on the left, each grey dot is one hidden state. Because
features are sparse, most dots lie along one of five rays: one feature on,
at some strength. The few dots off the rays are the rare examples with two
features on at once. The thick grey lines are the directions we planted;
the red arrows are what the SAE learned from the dots alone, and they land
on the rays. On the right, the x-axis is the penalty λ on a log scale. The
blue line is the worst match between a planted feature and its nearest
latent: it sits around 0.8 to 0.9 for small penalties and peaks near 0.99
at λ = 0.3. The red line, the variance left unexplained, is near 0 for
small penalties and climbs past 0.5 at λ = 1, where the penalty starts
crushing the codes and the match falls again. Choosing λ is a trade: too
small and the latents are not the features, too large and too much of the
model's activity is lost.

**Why it matters in practice.** This is the tool that turned
interpretability from "a few hand-picked neurons" into something that
scales. Anthropic's *Towards Monosemanticity* (2023) trained SAEs on a
small transformer and found thousands of features that each fire for one
recognizable thing. *Scaling Monosemanticity* (2024) did the same inside a
production model, Claude 3 Sonnet, and found features for concepts such as
the Golden Gate Bridge. Turning that one feature up made the model bring
up the bridge in almost every answer, which is the causal check that the
feature means what it seems to mean.

**In code:** `sae_objective` computes the loss for one proposed code, `SparseAutoencoder` holds the encoder and decoder with its hand-derived `SparseAutoencoder.loss_and_grads`, `train_sae` fits it, and `match_features` compares learned directions with planted ones.

## What these tools cannot (yet) show

**Everyday picture.** A brain scan shows which regions light up while a
person reads, but not the sentence they are thinking. Every tool in this
lesson is like that: a real instrument that measures something true and
partial.

**A tiny worked example.** Each of our own toys already showed a limit:

| What we saw | The limit it shows |
|---|---|
| A probe read tense at 98%, and the output ignored tense completely | Probes find information, not use |
| The lens read "Paris" at a position where patching restored 0% | Reading an activation doesn't mean anything downstream reads it |
| The weak SAE rebuilt everything perfectly and still found the wrong directions | A good rebuild score doesn't mean the features are real |
| The strong SAE left 11% of the variance unexplained | Some of the model's activity is in no feature the dictionary found |
| We knew the landmark model's circuit because we built it | In a real model, nobody has the answer key |

```mermaid
flowchart LR
  A["Correlation<br/>probes, the logit lens:<br/>'this information is here'"] --> B["Intervention<br/>patching, steering:<br/>'changing this changes the output'"]
  B --> C["Mechanism<br/>a circuit explained end to end:<br/>'this is how the output is computed'"]
```

**Reading it:** the arrow is the direction of stronger evidence. Most
claims about real models sit on the left or in the middle. A complete
mechanism, every step from input to output accounted for, exists only for
narrow behaviours in small or carefully chosen settings. When you read an
interpretability result, ask which box it reached.

The main open problems, in plain words:

- **Choices shape answers.** Patching results depend on how the corrupted
  prompt is built (swap one word? add noise?). SAE features depend on the
  dictionary's size and λ: a bigger dictionary can split one feature into
  several finer ones.
- **Redundancy hides importance.** Patching one site at a time can miss
  parts that back each other up. Models have been observed to partly
  repair themselves when one component is knocked out, so "patching this
  restores 0%" does not always mean "this plays no role".
- **Coverage.** The unexplained part of an SAE's rebuild is not noise to
  ignore: it is model behaviour no feature describes yet.
- **Scale and labour.** A circuit that explains one behaviour of a small
  model can take researchers weeks. No one has a complete account of how a
  frontier model produces any long answer.
- **Labels are human guesses.** Naming a feature "Golden Gate Bridge"
  summarizes the inputs that make it fire; checking that the name is right
  needs interventions like the one above.

**Why it matters in practice.** Treat these tools as evidence, stacked
next to behavioural evaluations (`primer.agents.evals`), never as a
certificate. A probe or a lens is a cheap first look; a patching or
steering experiment is the minimum for a causal claim; and a clean result
on a toy, like every result in this lesson, is where understanding starts,
not where it ends.

## In 20 seconds

- **Features are directions** in activation space, not individual neurons;
  a dot product with the right direction reads a feature out.
- **Probes** are small linear classifiers trained on frozen activations.
  They show information is present, not that the model uses it; score
  them on unseen data and against a control task.
- **The logit lens** applies the model's own output layer to intermediate
  residual states, showing what it would predict after each layer.
- **Activation patching** copies one activation from a clean run into a
  corrupted run and measures how much of the right answer returns: the
  basic causal experiment.
- **Superposition**: when features are sparse, models store more features
  than they have neurons, as nearly perpendicular directions, which makes
  neurons polysemantic.
- **Sparse autoencoders** learn an overcomplete dictionary with an L1
  penalty so each input uses a few latents, recovering features from
  superposition, at the cost of some unexplained variance.

## Self-test questions

**What does it mean to say a feature is a "direction" rather than a neuron?**
The feature's presence is stored as a pattern across many neurons: the
hidden state moves along a particular direction in proportion to how
strongly the feature is present. You read it with a dot product against
that direction. Any single neuron is typically a mix of several features.

**A linear probe reads a property from layer 12 at 95% accuracy. What can you conclude, and what can't you?**
You can conclude the property is linearly decodable from layer 12, as long
as the 95% was on held-out examples and clearly beats a control task with
random labels. You can't conclude the model uses it. For that you need an
intervention: remove or change that information and see whether the
output changes.

**How does the logit lens work, and why does it agree with the model at the last layer?**
It takes the residual stream after some layer and applies the model's own
final normalization and unembedding, turning it into next-token
probabilities. At the last layer that is exactly the computation the model
itself performs, so the two must match. Earlier layers are read "as if
the model stopped there".

**Describe an activation patching experiment, and what the "fraction restored" means.**
Run a clean prompt and save its activations; run a corrupted prompt that
changes one fact; then rerun the corrupted prompt with one activation
replaced by its clean value. The fraction restored is how much of the
clean-minus-corrupted logit difference the patch brings back: 1 means that
activation alone carries the fact, 0 means it carries none of it.

**Why do models use superposition, and why does it make neurons hard to interpret?**
There are more useful features than neurons. When features are rarely
active at the same time, the model can store them as nearly perpendicular
directions and filter the small interference with a bias and ReLU. The
directions can't line up with the neurons, so each neuron responds to
several unrelated features: it is polysemantic.

**Why does a sparse autoencoder need the L1 penalty? What goes wrong if λ is too small or too large?**
Many dictionaries rebuild the data equally well; the penalty picks the one
where each input uses few latents, which pushes latents onto the real
features. Too small and the SAE rebuilds perfectly with meaningless
directions; too large and it shrinks activations and leaves much of the
model's activity unexplained.

## The papers behind this lesson

- **Alain & Bengio, *Understanding intermediate layers using linear
  classifier probes* (2016)**: https://arxiv.org/abs/1610.01644.
  Introduced linear probes as a way to measure what each layer of a
  network makes linearly available.
- **Hewitt & Liang, *Designing and Interpreting Probes with Control Tasks*
  (2019)**: https://arxiv.org/abs/1909.03368. Showed that probes can
  succeed by memorizing, and introduced control tasks and selectivity to
  tell the two apart.
- **Belrose et al., *Eliciting Latent Predictions from Transformers with
  the Tuned Lens* (2023)**: https://arxiv.org/abs/2303.08112. Formalized
  the logit lens and fixed its failures on early layers by training a
  small translator for each layer.
- **Meng et al., *Locating and Editing Factual Associations in GPT*
  (2022)**: https://arxiv.org/abs/2202.05262. Introduced causal tracing,
  found factual recall in middle-layer MLPs at the subject's last token,
  and edited single facts there.
- **Wang et al., *Interpretability in the Wild: a Circuit for Indirect
  Object Identification in GPT-2 small* (2022)**:
  https://arxiv.org/abs/2211.00593. Used patching to reverse-engineer a
  complete circuit for one behaviour of a real language model.
- **Elhage et al., *Toy Models of Superposition* (2022)**:
  https://arxiv.org/abs/2209.10652. Showed with small ReLU models that
  sparse features are stored in superposition, and when and how the
  geometry changes.
- **Bricken et al., *Towards Monosemanticity: Decomposing Language Models
  With Dictionary Learning* (2023)**:
  https://transformer-circuits.pub/2023/monosemantic-features/index.html.
  Trained sparse autoencoders on a small transformer and found thousands
  of interpretable features hidden in polysemantic neurons.
- **Templeton et al., *Scaling Monosemanticity: Extracting Interpretable
  Features from Claude 3 Sonnet* (2024)**:
  https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html.
  Scaled sparse autoencoders to a production model and steered its
  behaviour through the features they found.

## Further reading

- Olah et al., *Zoom In: An Introduction to Circuits* (Distill, 2020): https://distill.pub/2020/circuits/zoom-in/
- Elhage et al., *A Mathematical Framework for Transformer Circuits* (2021): https://transformer-circuits.pub/2021/framework/index.html
- Elhage et al., *Toy Models of Superposition* (2022): https://transformer-circuits.pub/2022/toy_model/index.html
- Bricken et al., *Towards Monosemanticity* (2023): https://transformer-circuits.pub/2023/monosemantic-features/index.html
- Templeton et al., *Scaling Monosemanticity* (2024): https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
- Cunningham et al., *Sparse Autoencoders Find Highly Interpretable Features in Language Models* (2023): https://arxiv.org/abs/2309.08600
- Belinkov, *Probing Classifiers: Promises, Shortcomings, and Advances* (2021): https://arxiv.org/abs/2102.12452
- Meng et al., *Locating and Editing Factual Associations in GPT* (2022): https://arxiv.org/abs/2202.05262
- Belrose et al., *Eliciting Latent Predictions from Transformers with the Tuned Lens* (2023): https://arxiv.org/abs/2303.08112
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from primer._show import banner, matrix, say, table, takeaway
from primer.ml.attention import causal_mask, scaled_dot_product_attention, softmax
from primer.ml.neural_net import sigmoid
from primer.ml.optimizers import Adam
from primer.ml.transformer import TinyGPT, layer_norm

# ---------------------------------------------------------------------------
# 1. Features are directions
# ---------------------------------------------------------------------------

# Two perpendicular unit directions in a 2-neuron hidden state (0.6² + 0.8² = 1, and 0.6·0.8 − 0.8·0.6 = 0).
WORKED_DIRECTIONS: dict[str, np.ndarray] = {
    "positive": np.array([0.6, 0.8]),
    "past tense": np.array([0.8, -0.6]),
}


def compose_features(amounts: dict[str, float], directions: dict[str, np.ndarray]) -> np.ndarray:
    """Build a hidden state as Σ amount · direction, one term per feature."""
    return sum(amount * directions[name] for name, amount in amounts.items())


def read_features(h: np.ndarray, directions: dict[str, np.ndarray]) -> dict[str, float]:
    """Read each feature back with a dot product. Exact only when the directions are perpendicular unit vectors."""
    return {name: float(np.dot(d, h)) for name, d in directions.items()}


# ---------------------------------------------------------------------------
# 2. Probes: can a straight line read a property out of the hidden state?
# ---------------------------------------------------------------------------


def probe_probability(h, w, b: float) -> float:
    """One probe reading: σ(w · h + b), the probe's probability that the property is present."""
    return float(sigmoid(np.dot(w, h) + b))


@dataclass
class Probe:
    """A logistic-regression probe: a weight per neuron and one bias."""

    w: np.ndarray
    b: float

    def probability(self, H: np.ndarray) -> np.ndarray:
        return sigmoid(H @ self.w + self.b)

    def accuracy(self, H: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean((self.probability(H) > 0.5) == (y == 1)))


def train_probe(H: np.ndarray, y: np.ndarray, steps: int = 500, lr: float = 0.5) -> Probe:
    """Fit a probe by plain gradient descent on binary cross-entropy.

    The model under study is frozen: only the probe's d + 1 numbers change.
    The gradient of the mean cross-entropy with respect to the logit is
    simply (p − y), which is why the update is one line.
    """
    w, b = np.zeros(H.shape[1]), 0.0
    for _ in range(steps):
        error = sigmoid(H @ w + b) - y  # (n,): how far each prediction is from its label
        w -= lr * H.T @ error / len(y)
        b -= lr * float(error.mean())
    return Probe(w, b)


class PlantedModel:
    """A toy model whose hidden layer stores two known features as directions.

    Each example has a sentiment (0 = negative, 1 = positive) and a tense
    (0 = present, 1 = past). The hidden state is

        h = (±1) · sentiment_direction + (±1) · tense_direction + noise

    in `d` neurons, with both directions random, so no single neuron is
    either feature. The model's output reads sentiment and, by construction,
    gives tense a weight of exactly zero: the tense information is in the
    hidden state, but nothing downstream uses it.
    """

    FEATURES = ("sentiment", "tense")

    def __init__(self, d: int = 32, noise: float = 0.4, seed: int = 0):
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((d, 2))
        A /= np.linalg.norm(A, axis=0)  # unit directions, not perpendicular to each other
        self.directions = dict(zip(self.FEATURES, A.T))
        # The pseudo-inverse's first row is the readout with weight 1 on sentiment and 0 on tense.
        self.readout = np.linalg.pinv(A)[0]
        self.d, self.noise = d, noise

    def hidden(self, labels: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
        """(n, d) hidden states for 0/1 labels, with the noise passed in so a flip changes only the label."""
        signs = {name: 2 * np.asarray(labels[name]) - 1 for name in self.FEATURES}  # 0/1 -> −1/+1
        return sum(np.outer(signs[name], self.directions[name]) for name in self.FEATURES) + noise

    def output(self, H: np.ndarray) -> np.ndarray:
        """The model's own score for each example: positive means "positive review"."""
        return H @ self.readout

    def sample(self, n: int, seed: int = 0) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
        rng = np.random.default_rng(seed)
        labels = {name: rng.integers(0, 2, n) for name in self.FEATURES}
        noise = rng.normal(0, self.noise, (n, self.d))
        return self.hidden(labels, noise), labels, noise


def flip_effect(model: PlantedModel, feature: str, n: int = 200, seed: int = 0) -> float:
    """Average change in the model's output when only `feature` is flipped, noise held fixed.

    This is an intervention, not a reading: it asks whether the model *uses*
    the feature, which no probe can answer.
    """
    H, labels, noise = model.sample(n, seed)
    flipped = dict(labels, **{feature: 1 - labels[feature]})
    return float(np.mean(np.abs(model.output(model.hidden(flipped, noise)) - model.output(H))))


def probe_report(n_train: int = 100, n_test: int = 1000, seed: int = 0) -> dict[str, dict[str, float]]:
    """Train three probes on the same frozen hidden states and score each on its own training set and on unseen examples.

    * sentiment: stored and used by the model
    * tense: stored but never used by the model's output
    * random labels: the control task, with nothing real to find
    """
    model = PlantedModel()
    H_train, y_train, _ = model.sample(n_train, seed)
    H_test, y_test, _ = model.sample(n_test, seed + 1)
    rng = np.random.default_rng(seed + 2)
    y_train["random labels"] = rng.integers(0, 2, n_train)
    y_test["random labels"] = rng.integers(0, 2, n_test)
    report = {}
    for name in ("sentiment", "tense", "random labels"):
        probe = train_probe(H_train, y_train[name])
        report[name] = {"train": probe.accuracy(H_train, y_train[name]), "test": probe.accuracy(H_test, y_test[name])}
    return report


# ---------------------------------------------------------------------------
# 3. The logit lens: read the residual stream with the model's own output layer
# ---------------------------------------------------------------------------

LENS_VOCAB = ["Paris", "London", "Rome"]
# The output layer of the worked example: one row per word, dotted with the 2-number residual state.
LENS_UNEMBED = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
LENS_START = np.array([0.2, 0.3])  # the residual state straight after the embedding
LENS_WRITES = [np.array([0.6, 0.0]), np.array([1.2, -0.3])]  # what layers 1 and 2 add to it


def logit_lens(residuals: np.ndarray, W_U: np.ndarray, final_norm=None) -> np.ndarray:
    """Probabilities the model would give if it stopped here: softmax(W_U · norm(h)) for every state in `residuals`."""
    x = final_norm(residuals) if final_norm is not None else residuals
    return softmax(x @ W_U.T, axis=-1)


def worked_lens() -> list[dict]:
    """The worked example: the lens after the embedding, after layer 1 and after layer 2."""
    states = [LENS_START]
    for write in LENS_WRITES:
        states.append(states[-1] + write)  # a residual layer adds; it never overwrites
    probs = logit_lens(np.array(states), LENS_UNEMBED)
    return [
        dict(layer=i, residual=s, logits=LENS_UNEMBED @ s, probs=p.tolist(), top=LENS_VOCAB[int(np.argmax(p))])
        for i, (s, p) in enumerate(zip(states, probs))
    ]


def residual_stream(model: TinyGPT, ids: np.ndarray) -> np.ndarray:
    """(n_layers + 1, seq, d): the residual state after the embedding and after every block."""
    ids = np.asarray(ids)
    x = model.wte[ids] + model.wpe[: len(ids)]
    states = [x]
    for block in model.blocks:
        x = block(x)
        states.append(x)
    return np.stack(states)


def tinygpt_logit_lens(model: TinyGPT, ids: np.ndarray) -> np.ndarray:
    """(n_layers + 1, seq, vocab) logits: every layer's residual state read through the final norm and the output layer.

    Returns logits rather than probabilities so the top layer can be
    compared number for number with the model's own output.
    """
    R = residual_stream(model, ids)
    return layer_norm(R, model.lnf_g, model.lnf_b) @ model.wte.T


# ---------------------------------------------------------------------------
# 4. A hand-built model where we know where the answer flows
# ---------------------------------------------------------------------------


class LandmarkModel:
    """A two-layer, one-head transformer built by hand to answer "<landmark> is in" -> city.

    Residual dimensions (a readable basis, so you can check it by eye; no
    tool in this lesson relies on it):

        0-2  which landmark (Eiffel, Colosseum, Big Ben)
        3    "this token is a landmark"
        4-5  which filler word ("is", "in")
        6-8  which city (Paris, Rome, London)

    Layer 1 is an MLP that looks up a fact at every position: a landmark's
    identity in, its city out. Layer 2 is an attention head: the word "in"
    queries for the landmark flag, finds the landmark and copies its city
    dimensions into the last position. The output layer reads the city
    dimensions at the last position.
    """

    TOKENS = ["Eiffel", "Colosseum", "Big Ben", "is", "in"]
    CITIES = ["Paris", "Rome", "London"]
    STAGES = ("resid_embed", "resid_mlp", "resid_attn")
    D = 9

    def __init__(self, gain: float = 3.0, sharpness: float = 4.0):
        D = self.D
        self.E = np.zeros((len(self.TOKENS), D))
        for i in range(3):
            self.E[i, i] = 1.0  # landmark identity
            self.E[i, 3] = 1.0  # landmark flag
        self.E[3, 4] = self.E[4, 5] = 1.0  # "is", "in"
        # MLP: three hidden units, one per landmark; each writes `gain` onto its city.
        self.W_in = np.zeros((D, 3))
        self.W_in[:3, :3] = np.eye(3)
        self.W_out = np.zeros((3, D))
        self.W_out[:3, 6:9] = gain * np.eye(3)
        # Attention, one head of width 1: "in" asks (query), the landmark flag answers (key).
        self.W_q = np.zeros((D, 1))
        self.W_q[5, 0] = sharpness
        self.W_k = np.zeros((D, 1))
        self.W_k[3, 0] = sharpness
        self.W_v = np.zeros((D, D))
        self.W_v[6:9, 6:9] = np.eye(3)  # the value is the city part of the state, copied as is
        self.W_U = np.zeros((3, D))
        self.W_U[:, 6:9] = np.eye(3)

    def run(self, tokens: list[str], patch: dict | None = None, answer: tuple[str, str] = ("Paris", "Rome")) -> dict:
        """Run the model, optionally overwriting activations.

        `patch` maps (site, position) to a vector that replaces that activation
        during this run. Sites: "resid_embed", "mlp_out", "resid_mlp",
        "attn_out", "resid_attn". Returns every stage's residual state
        (3, seq, D), the component outputs, the last position's logits, and
        the logit difference between the two `answer` cities.
        """
        patch = patch or {}

        def apply(site: str, acts: np.ndarray) -> np.ndarray:
            acts = acts.copy()
            for (s, pos), vector in patch.items():
                if s == site:
                    acts[pos] = vector
            return acts

        ids = [self.TOKENS.index(t) for t in tokens]
        x = apply("resid_embed", self.E[ids])
        stages = [x]
        mlp_out = apply("mlp_out", np.maximum(x @ self.W_in, 0) @ self.W_out)
        x = apply("resid_mlp", x + mlp_out)
        stages.append(x)
        mixed, weights = scaled_dot_product_attention(x @ self.W_q, x @ self.W_k, x @ self.W_v, mask=causal_mask(len(ids)))
        attn_out = apply("attn_out", mixed)
        x = apply("resid_attn", x + attn_out)
        stages.append(x)
        logits = x[-1] @ self.W_U.T  # only the last position predicts the next word
        good, bad = (self.CITIES.index(c) for c in answer)
        return dict(
            resid=np.stack(stages), mlp_out=mlp_out, attn_out=attn_out, attn_weights=weights,
            logits=logits, logit_diff=float(logits[good] - logits[bad]),
        )


def fraction_restored(patched: float, clean: float, corrupt: float) -> float:
    """How much of the clean-vs-corrupt gap one patch closes: 0 = nothing, 1 = everything."""
    return (patched - corrupt) / (clean - corrupt)


def patching_map(model: LandmarkModel, clean: list[str], corrupt: list[str]) -> np.ndarray:
    """(stage, position) grid: patch each residual state from the clean run into the corrupted run, one at a time."""
    clean_run, corrupt_run = model.run(clean), model.run(corrupt)
    grid = np.zeros((len(model.STAGES), len(clean)))
    for s, stage in enumerate(model.STAGES):
        for pos in range(len(clean)):
            patched = model.run(corrupt, patch={(stage, pos): clean_run["resid"][s, pos]})
            grid[s, pos] = fraction_restored(patched["logit_diff"], clean_run["logit_diff"], corrupt_run["logit_diff"])
    return grid


def lens_map(model: LandmarkModel, tokens: list[str], answer: str = "Paris") -> np.ndarray:
    """(stage, position) grid of the logit lens's probability for `answer`."""
    resid = model.run(tokens)["resid"]
    return logit_lens(resid, model.W_U)[..., model.CITIES.index(answer)]


# ---------------------------------------------------------------------------
# 5. Superposition: more features than dimensions
# ---------------------------------------------------------------------------


def pentagon() -> np.ndarray:
    """(2, 5): five unit feature directions spread 72° apart in a 2-neuron space."""
    angles = np.radians(72 * np.arange(5))
    return np.stack([np.cos(angles), np.sin(angles)])


def superposition_readout(W: np.ndarray, b: np.ndarray, x: np.ndarray, relu: bool = True) -> np.ndarray:
    """Squeeze features x into h = W x, then read them back out: ReLU(Wᵀ h + b)."""
    z = W.T @ (W @ x) + b
    return np.maximum(z, 0) if relu else z


def sample_sparse_features(n: int, n_features: int, p_active: float, rng: np.random.Generator) -> np.ndarray:
    """(n, n_features): each feature is on with probability p_active, at a strength uniform in [0, 1)."""
    on = rng.random((n, n_features)) < p_active
    return on * rng.random((n, n_features))


def superposition_loss_and_grads(W: np.ndarray, b: np.ndarray, X: np.ndarray, importance: np.ndarray | None = None):
    """Importance-weighted reconstruction loss of the toy model, and its gradients, derived by hand.

    Forward:  H = X Wᵀ (batch, d),  Z = H W + b (batch, n),  Y = ReLU(Z)
    Loss:     mean over the batch of Σᵢ Iᵢ (Yᵢ − Xᵢ)²
    W appears twice (squeeze and unsqueeze), so its gradient has two terms.
    """
    importance = np.ones(X.shape[1]) if importance is None else importance
    H = X @ W.T
    Z = H @ W + b
    Y = np.maximum(Z, 0)
    loss = float(np.mean(np.sum(importance * (Y - X) ** 2, axis=1)))
    dZ = 2 * importance * (Y - X) / len(X) * (Z > 0)  # ReLU passes gradient only where it was on
    dH = dZ @ W.T
    grads = {"W": H.T @ dZ + dH.T @ X, "b": dZ.sum(axis=0)}
    return loss, grads


def train_superposition(
    p_active: float, n_features: int = 5, d: int = 2, importance_decay: float = 0.8,
    steps: int = 2000, batch: int = 1024, lr: float = 0.01, seed: int = 0,
) -> dict:
    """Train the toy model of Elhage et al. (2022) on features that are on with probability `p_active`.

    Feature i matters `importance_decay ** i` as much as feature 0, so when
    the model cannot keep everything it has a reason to choose.
    """
    rng = np.random.default_rng(seed)
    importance = importance_decay ** np.arange(n_features)
    params = {"W": rng.normal(0, 0.5, (d, n_features)), "b": np.zeros(n_features)}
    opt = {k: Adam(lr=lr) for k in params}
    losses = []
    for _ in range(steps):
        X = sample_sparse_features(batch, n_features, p_active, rng)
        loss, grads = superposition_loss_and_grads(params["W"], params["b"], X, importance)
        for k in params:
            params[k] = opt[k].step(params[k], grads[k])
        losses.append(loss)
    return dict(params, losses=np.array(losses))


def features_a_neuron_responds_to(W: np.ndarray, neuron: int, threshold: float = 0.25) -> list[int]:
    """Indices of the features that push this neuron up by more than `threshold` per unit of feature."""
    return [k for k in range(W.shape[1]) if W[neuron, k] > threshold]


def finite_difference_gradient(f, w: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Slow, obviously correct gradient: nudge each entry up and down and watch the loss."""
    grad = np.zeros_like(w)
    for idx in np.ndindex(w.shape):
        old = w[idx]
        w[idx] = old + eps
        up = f(w)
        w[idx] = old - eps
        down = f(w)
        w[idx] = old
        grad[idx] = (up - down) / (2 * eps)
    return grad


# ---------------------------------------------------------------------------
# 6. Sparse autoencoders: learn the features back
# ---------------------------------------------------------------------------


def sae_objective(h: np.ndarray, f: np.ndarray, decoder: np.ndarray, lam: float, b_d: np.ndarray | float = 0.0) -> float:
    """The sparse autoencoder's loss for one example and one proposed code f: ‖h − (D f + b_d)‖² + λ Σ|fᵢ|."""
    residual = h - (decoder @ f + b_d)
    return float(residual @ residual + lam * np.abs(f).sum())


class SparseAutoencoder:
    """encode: f = ReLU(W_e (h − b_d) + b_e);  decode: ĥ = W_d f + b_d.

    `n_latents` is usually much bigger than `d` (overcomplete). Each decoder
    column is kept at length 1, so the L1 penalty cannot be dodged by making
    codes tiny and decoder columns huge.
    """

    def __init__(self, d: int, n_latents: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        W_d = rng.standard_normal((d, n_latents))
        self.W_d = W_d / np.linalg.norm(W_d, axis=0)  # (d, m): one unit direction per latent
        self.W_e = self.W_d.T.copy()  # (m, d): start each latent listening for its own direction
        self.b_e = np.zeros(n_latents)
        self.b_d = np.zeros(d)

    def encode(self, H: np.ndarray) -> np.ndarray:
        return np.maximum((H - self.b_d) @ self.W_e.T + self.b_e, 0)

    def decode(self, F: np.ndarray) -> np.ndarray:
        return F @ self.W_d.T + self.b_d

    def reconstruction_error(self, H: np.ndarray) -> float:
        """Fraction of the data's variance the autoencoder fails to rebuild (0 = perfect)."""
        residual = self.decode(self.encode(H)) - H
        return float(np.sum(residual**2) / np.sum((H - H.mean(axis=0)) ** 2))

    def loss_and_grads(self, H: np.ndarray, lam: float) -> tuple[float, dict[str, np.ndarray]]:
        """Mean over the batch of ‖h − ĥ‖² + λ Σ fᵢ, and the gradient for every parameter, by hand."""
        n = len(H)
        centred = H - self.b_d
        pre = centred @ self.W_e.T + self.b_e  # (n, m)
        F = np.maximum(pre, 0)
        residual = F @ self.W_d.T + self.b_d - H  # (n, d)
        loss = float(np.sum(residual**2) / n + lam * F.sum() / n)  # F ≥ 0, so |F| = F
        d_hat = 2 * residual / n
        dF = d_hat @ self.W_d + lam / n
        d_pre = dF * (pre > 0)
        grads = {
            "W_d": d_hat.T @ F,
            "W_e": d_pre.T @ centred,
            "b_e": d_pre.sum(axis=0),
            # b_d is added back in the decoder and subtracted in the encoder.
            "b_d": d_hat.sum(axis=0) - (d_pre @ self.W_e).sum(axis=0),
        }
        return loss, grads


def train_sae(
    H: np.ndarray, n_latents: int = 5, lam: float = 0.3, steps: int = 2000, batch: int = 512, lr: float = 0.01, seed: int = 0
) -> SparseAutoencoder:
    """Fit a sparse autoencoder to activations H (n, d) with Adam, renormalizing decoder columns after every step."""
    rng = np.random.default_rng(seed)
    sae = SparseAutoencoder(H.shape[1], n_latents, seed)
    names = ("W_e", "b_e", "W_d", "b_d")
    opt = {k: Adam(lr=lr) for k in names}
    for _ in range(steps):
        _, grads = sae.loss_and_grads(H[rng.integers(0, len(H), batch)], lam)
        for k in names:
            setattr(sae, k, opt[k].step(getattr(sae, k), grads[k]))
        sae.W_d /= np.linalg.norm(sae.W_d, axis=0)
    return sae


def match_features(true_directions: np.ndarray, learned_directions: np.ndarray) -> np.ndarray:
    """For each true feature (a column), the cosine similarity of the learned column that points most nearly the same way."""
    t = true_directions / np.linalg.norm(true_directions, axis=0)
    ell = learned_directions / np.linalg.norm(learned_directions, axis=0)
    return (t.T @ ell).max(axis=1)


# ---------------------------------------------------------------------------
# 7. Figures (rendered into the HTML docs by `make figures`)
# ---------------------------------------------------------------------------

CLEAN = ["Eiffel", "is", "in"]
CORRUPT = ["Colosseum", "is", "in"]
STAGE_LABELS = ["after embedding", "after layer 1 (MLP)", "after layer 2 (attention)"]


def figures() -> dict:
    """Plot this lesson's data. matplotlib is imported here, and only here,
    so the lesson itself needs nothing beyond NumPy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, RED, GREEN, MUTED = "#2563eb", "#dc2626", "#059669", "#9ca3af"
    figs = {}

    # --- 1. Features as directions -----------------------------------------
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    amounts = {"positive": 0.9, "past tense": 0.4}
    h = compose_features(amounts, WORKED_DIRECTIONS)
    for (name, d), color in zip(WORKED_DIRECTIONS.items(), (BLUE, GREEN)):
        ax.annotate("", d, (0, 0), arrowprops=dict(arrowstyle="->", color=color, lw=2))
        # Labels sit beyond the arrow tip, above it for the upper arrow and below it for the lower one.
        ax.text(d[0] * 1.1, d[1] * 1.1, f'"{name}"\ndirection', color=color, ha="center", va="bottom" if d[1] > 0 else "top", fontsize=9)
        foot = amounts[name] * d
        ax.plot([h[0], foot[0]], [h[1], foot[1]], ls=":", color=color)
        ax.plot(*foot, "o", color=color, ms=4)
    ax.annotate("", h, (0, 0), arrowprops=dict(arrowstyle="->", color=RED, lw=2.5))
    ax.text(h[0] + 0.04, h[1] + 0.03, "h = (0.86, 0.48)", color=RED)
    ax.set_xlim(-0.2, 1.15)
    ax.set_ylim(-0.9, 1.1)
    ax.set_aspect("equal")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("neuron 1")
    ax.set_ylabel("neuron 2")
    ax.set_title("Two features, two directions, one hidden state")
    figs["directions"] = fig

    # --- 2. Probes: what they can read, and what the model uses ------------
    report = probe_report()
    model = PlantedModel()
    names = ["sentiment", "tense", "random labels"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    x = np.arange(len(names))
    a1.bar(x - 0.2, [report[n]["train"] for n in names], 0.4, color=MUTED, label="training examples")
    a1.bar(x + 0.2, [report[n]["test"] for n in names], 0.4, color=BLUE, label="unseen examples")
    for xi, n in zip(x, names):
        a1.text(xi + 0.2, report[n]["test"] + 0.02, f"{report[n]['test']:.2f}", ha="center")
    a1.axhline(0.5, color=RED, ls="--", lw=1, label="chance")
    a1.set_xticks(x, names)
    a1.set_ylim(0, 1.3)
    a1.set_ylabel("probe accuracy")
    a1.set_title("What a probe can read")
    a1.legend(frameon=False, loc="upper right", fontsize=8)
    effects = [flip_effect(model, f) for f in ("sentiment", "tense")]
    a2.bar([0, 1], effects, 0.5, color=[BLUE, MUTED])
    for xi, e in enumerate(effects):
        a2.text(xi, e + 0.05, f"{e:.2f}", ha="center")
    a2.set_xticks([0, 1], ["flip sentiment", "flip tense"])
    a2.set_ylim(0, 2.5)
    a2.set_ylabel("change in the model's output")
    a2.set_title("What the model actually uses")
    fig.tight_layout()
    figs["probe_vs_use"] = fig

    # --- 3. Logit lens: worked example and the landmark model -------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.6), gridspec_kw={"width_ratios": [1, 1.2]})
    lens = worked_lens()
    layers = np.arange(len(lens))
    for k, (word, color) in enumerate(zip(LENS_VOCAB, (BLUE, GREEN, MUTED))):
        a1.bar(layers + (k - 1) * 0.27, [row["probs"][k] for row in lens], 0.27, color=color, label=word)
    a1.set_xticks(layers, ["embedding", "layer 1", "layer 2"])
    a1.set_ylabel("lens probability")
    a1.set_ylim(0, 1)
    a1.set_title("Worked example: the guess firms up")
    a1.legend(frameon=False, fontsize=8)
    grid = lens_map(LandmarkModel(), CLEAN)
    im = a2.imshow(grid, cmap="Blues", vmin=0, vmax=1)
    for (r, c), v in np.ndenumerate(grid):
        a2.text(c, r, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.6 else "black")
    a2.set_xticks(range(3), CLEAN)
    a2.set_yticks(range(3), STAGE_LABELS)
    a2.set_title('Landmark model: P("Paris") by layer and word')
    a2.grid(False)
    fig.colorbar(im, ax=a2, fraction=0.046)
    fig.tight_layout()
    figs["logit_lens"] = fig

    # --- 4. Activation patching map ---------------------------------------
    grid = patching_map(LandmarkModel(), CLEAN, CORRUPT)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    im = ax.imshow(grid, cmap="Greens", vmin=0, vmax=1)
    for (r, c), v in np.ndenumerate(grid):
        ax.text(c, r, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.6 else "black")
    ax.set_xticks(range(3), [f"{a} / {b}" if a != b else a for a, b in zip(CLEAN, CORRUPT)])
    ax.set_yticks(range(3), STAGE_LABELS)
    ax.set_xlabel("position patched (clean / corrupted word)")
    ax.set_title("Fraction of the answer restored by one patch")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    figs["patching"] = fig

    # --- 5. Superposition: dense keeps two, sparse keeps five -------------
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    colors = [BLUE, GREEN, "#d97706", "#7c3aed", RED]
    for ax, p, title in ((axes[0], 1.0, "Dense features: keeps 2"), (axes[1], 0.05, "Sparse features: keeps all 5")):
        W = train_superposition(p_active=p)["W"]
        for k in range(W.shape[1]):
            if np.linalg.norm(W[:, k]) > 0.1:
                ax.annotate("", W[:, k], (0, 0), arrowprops=dict(arrowstyle="->", color=colors[k], lw=2))
                ax.text(*(W[:, k] * 1.18), f"f{k}", color=colors[k], ha="center", va="center")
            else:  # a dropped feature: an arrow this short would be all arrowhead
                ax.plot(*W[:, k], "o", color=colors[k], ms=4)
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.45, 1.45)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\n(each feature on {p:.0%} of the time)")
        ax.set_xlabel("neuron 1")
        ax.set_ylabel("neuron 2")
    W = pentagon()
    x = np.arange(5)
    axes[2].bar(x - 0.2, W[0], 0.4, color=BLUE, label="neuron 1")
    axes[2].bar(x + 0.2, W[1], 0.4, color=MUTED, label="neuron 2")
    axes[2].axhline(0, color="#4b5563", lw=0.8)
    axes[2].set_xticks(x, [f"f{k}" for k in x])
    axes[2].set_ylabel("how much the feature moves the neuron")
    axes[2].set_title("Each neuron answers to several features")
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    figs["superposition"] = fig

    # --- 6. Sparse autoencoder: recovering the planted features ------------
    rng = np.random.default_rng(1)
    H = sample_sparse_features(20_000, 5, 0.05, rng) @ pentagon().T
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.9))
    shown = H[np.abs(H).sum(axis=1) > 0][:600]
    a1.scatter(shown[:, 0], shown[:, 1], s=3, color=MUTED, alpha=0.5, label="hidden states")
    sae = train_sae(H, n_latents=5, lam=0.3)
    for k in range(5):
        t = pentagon()[:, k]
        a1.plot([0, 1.25 * t[0]], [0, 1.25 * t[1]], color=MUTED, lw=6, alpha=0.4, label="planted features" if k == 0 else None)
        d = sae.W_d[:, k]
        a1.annotate("", 1.1 * d, (0, 0), arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))
    a1.plot([], [], color=RED, label="learned latents")
    a1.set_aspect("equal")
    a1.set_xlim(-1.4, 1.4)
    a1.set_ylim(-1.4, 1.4)
    a1.set_title("λ = 0.3: each latent lands on a feature")
    a1.legend(frameon=False, fontsize=8, loc="lower left")
    lams = [0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.6, 1.0]
    worst, error = [], []
    for lam in lams:
        s = train_sae(H, n_latents=5, lam=lam)
        worst.append(match_features(pentagon(), s.W_d).min())
        error.append(s.reconstruction_error(H))
    a2.plot(lams, worst, "o-", color=BLUE, label="worst feature match (cosine)")
    a2.plot(lams, error, "s-", color=RED, label="variance left unexplained")
    a2.set_xscale("log")
    a2.set_xlabel("sparsity penalty λ")
    a2.set_ylim(0, 1.05)
    a2.set_title("The penalty trades rebuild for meaning")
    a2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    figs["sae"] = fig

    return figs


# ---------------------------------------------------------------------------
# 8. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. A feature is a direction")
    h = compose_features({"positive": 0.9, "past tense": 0.4}, WORKED_DIRECTIONS)
    say(
        f"""
        Mix 0.9 of the "positive" direction (0.6, 0.8) with 0.4 of the "past
        tense" direction (0.8, -0.6) and the hidden state is h = ({h[0]:.2f},
        {h[1]:.2f}). Neither neuron equals either feature. A dot product with
        each direction reads the amounts back:
        """
    )
    table(["feature", "direction · h"], [(k, v) for k, v in read_features(h, WORKED_DIRECTIONS).items()], floatfmt=".2f")
    takeaway("Features live in directions, not in individual neurons.")

    banner("2. Probes: reading a property out of the hidden state")
    say(
        f"""
        A probe is a tiny classifier trained on frozen hidden states. The
        worked probe w = (1, 0.5), b = 0 reads h = (2, -1) as
        σ(1.5) = {probe_probability([2.0, -1.0], [1.0, 0.5], 0.0):.3f}. Now three
        probes on a toy model with 32 neurons, trained on 100 examples and
        scored on 1,000 unseen ones:
        """
    )
    report = probe_report()
    table(["probe target", "train accuracy", "unseen accuracy"], [(k, v["train"], v["test"]) for k, v in report.items()], floatfmt=".3f")
    model = PlantedModel()
    say(
        f"""
        Random labels fit the training set to {report['random labels']['train']:.0%} and then
        score a coin flip on unseen examples: accuracy only counts on held-out
        data. Tense is read at {report['tense']['test']:.0%}, yet flipping it moves the
        model's output by {flip_effect(model, 'tense'):.2f}, while flipping sentiment moves
        it by {flip_effect(model, 'sentiment'):.2f}.
        """
    )
    takeaway("A probe shows information is present, not that the model uses it.")

    banner("3. The logit lens: what would the model say if it stopped here?")
    table(
        ["after", "residual", "P(Paris)", "P(London)", "P(Rome)", "top"],
        [(["embedding", "layer 1", "layer 2"][r["layer"]], str(np.round(r["residual"], 2)), *r["probs"], r["top"]) for r in worked_lens()],
        floatfmt=".2f",
    )
    tiny = TinyGPT(vocab_size=11, d_model=16, n_layers=3, n_heads=2, max_len=8, seed=4)
    ids = np.array([3, 1, 4, 1, 5])
    agree = np.allclose(tinygpt_logit_lens(tiny, ids)[-1], tiny(ids))
    say(f"On primer.ml.transformer.TinyGPT, the lens at the top layer equals the model's own logits: {agree}.")
    matrix('landmark model, lens P("Paris"): rows = layers, cols = Eiffel, is, in', lens_map(LandmarkModel(), CLEAN))
    takeaway("The Eiffel position knows Paris after layer 1; the last word only learns it in layer 2.")

    banner("4. Activation patching: which activations carry the answer?")
    lm = LandmarkModel()
    clean, corrupt = lm.run(CLEAN)["logit_diff"], lm.run(CORRUPT)["logit_diff"]
    say(
        f"""
        Logit difference Paris minus Rome: clean "Eiffel is in" gives {clean:+.2f},
        corrupted "Colosseum is in" gives {corrupt:+.2f}. Copy one clean
        residual state into the corrupted run at a time and measure the
        fraction of the gap that comes back:
        """
    )
    matrix("fraction restored: rows = layers, cols = positions", patching_map(lm, CLEAN, CORRUPT), precision=2)
    takeaway("The answer sits at the subject early and at the last word late: a two-step circuit, found by intervention.")

    banner("5. Superposition: five features in two neurons")
    table(
        ["feature", "readout when only f0 is on (no ReLU)", "with bias -0.31 and ReLU"],
        [
            (f"f{k}", a, c)
            for k, (a, c) in enumerate(
                zip(
                    superposition_readout(pentagon(), np.zeros(5), np.eye(5)[0], relu=False),
                    superposition_readout(pentagon(), np.full(5, -0.31), np.eye(5)[0]),
                )
            )
        ],
        floatfmt=".3f",
    )
    for p in (1.0, 0.05):
        W = train_superposition(p_active=p)["W"]
        print(f"trained with each feature on {p:>4.0%} of the time: feature lengths {np.round(np.linalg.norm(W, axis=0), 2)}")
    print()
    say(
        """
        Dense features: the model keeps the two most important and drops the
        rest. Sparse features: it keeps all five, 72 degrees apart, and a
        negative bias filters the small leaks. In the ideal pentagon, neuron 1 answers to
        features 0, 1 and 4: it is polysemantic.
        """
    )

    banner("6. Sparse autoencoders: getting the features back")
    rng = np.random.default_rng(1)
    H = sample_sparse_features(20_000, 5, 0.05, rng) @ pentagon().T
    rows = []
    for lam in (0.01, 0.3):
        sae = train_sae(H, n_latents=5, lam=lam)
        rows.append((lam, sae.reconstruction_error(H), match_features(pentagon(), sae.W_d).min()))
    table(["λ", "variance unexplained", "worst feature match (cosine)"], rows, floatfmt=".3f")
    takeaway(
        "Any two directions can rebuild a 2-D space; only the sparsity penalty makes the "
        "autoencoder find the five directions the model really used."
    )


if __name__ == "__main__":
    demo()
