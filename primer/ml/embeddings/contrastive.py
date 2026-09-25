r"""
# Contrastive training: how an embedding model learns what "similar" means

Run: `python -m primer.ml.embeddings.contrastive`

New to vectors, dot products, Σ or log? `primer.notation` builds them from zero.

## The everyday picture

A teacher has a stack of question cards and a stack of answer cards. She
lays out a few questions, deals out all the answers face up, and asks the
student to pair each question with its answer. Every wrong pairing, she
corrects. Over many rounds the student learns what makes an answer fit a
question.

Then she gets sneaky. Alongside each right answer she slips in a
**look-alike wrong card**: same subject, wrong answer. "How do I reset my
password?" now faces both *the reset steps* and *the password policy*. A
student who only ever saw easy wrong cards (answers about printers or
holidays) would happily pick the policy card because it says "password". The
look-alikes force the student to learn the difference that actually matters.

That's **contrastive training**, and it's how nearly every modern embedding
model is taught. The student is the model; "pairing" means making a
question's vector point the same way as its answer's vector (see
`primer.ml.embeddings.similarity`); the wrong cards are **negatives**; the
look-alikes are **hard negatives**.

## A tiny worked example: one question, two answer cards

A question's vector is q = (1, 0). The right answer is p₊ = (1, 0) and a wrong
one is p₋ = (0, 1). All three have length 1, so a dot product is a cosine.

1. **Score each card** with the dot product: q·p₊ = 1, q·p₋ = 0.
2. **Divide by the temperature** τ (tau), a sharpness dial explained
   below. With τ = 1 the scores stay (1, 0).
3. **Softmax** (from `primer.ml.attention`): exponentiate and share out.
   e¹ = 2.718 and e⁰ = 1, so the right card gets 2.718 / 3.718 = **0.731**.
4. **Loss** = −ln 0.731 = **0.313**. It would be 0 if the right card got 100%.

At τ = 0.1 the scores become (10, 0), the right card gets 99.995%, and the
loss is **0.0000454**. Same vectors, far more confident: that's what the
temperature does.

```mermaid
flowchart LR
  Q["Question: reset my password"] --> E1[Encoder]
  P["Right answer: reset steps"] --> E2[Encoder]
  N["Look-alike: password policy"] --> E3[Encoder]
  E1 --> PULL[Pull these two<br/>vectors together]
  E2 --> PULL
  E1 --> PUSH[Push these two<br/>vectors apart]
  E3 --> PUSH
```

**Reading it:** three texts go through the *same* encoder (the model that
turns text into a vector). The question and its right answer are pulled
together; the question and the look-alike are pushed apart. The look-alike
is about the same topic but doesn't answer the question. Learning to
separate it from the right answer is what makes a model good at *retrieval*
(finding the answer) rather than just *topic matching* (finding the subject).

## The math: InfoNCE, the loss behind it

In practice the teacher deals a whole batch. B questions sit in rows, all the
answer cards in the batch sit in columns, and every card that isn't a
question's own answer is a free negative for it: **in-batch negatives**.

```mermaid
flowchart LR
  subgraph Batch["Similarity table for a batch of 3"]
    direction TB
    r1["q₁:  ✔ ✗ ✗ | ✗"]
    r2["q₂:  ✗ ✔ ✗ | ✗"]
    r3["q₃:  ✗ ✗ ✔ | ✗"]
  end
  Batch --> SM["softmax along<br/>each row"] --> L["loss: −log of the<br/>✔ cell's share"]
```

**Reading it:** each row is one question scored against every answer card in
the batch. The ✔ on the diagonal is its own answer; every ✗ is a negative
that costs nothing extra because those answers are already in the batch.
Extra columns to the right of the bar (|) are hard negatives added on purpose,
which belong to no question. Softmax runs along each row, and the loss asks
each row to put its share on the ✔.

$$
L = -\frac{1}{B}\sum_{i=1}^{B} \log
\frac{\exp(q_i \cdot p_i / \tau)}{\sum_{j=1}^{M} \exp(q_i \cdot p_j / \tau)}
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| B | number of questions in the batch | 8 in this lesson |
| M | number of answer cards in the batch (B, plus any hard negatives) | 8 or 16 here |
| i | which question (row) | 1 to B |
| j | which answer card (column) | 1 to M |
| qᵢ | the i-th question's vector, unit length | d numbers (d = 16) |
| pᵢ | question i's own answer vector (the ✔) | d numbers |
| pⱼ | the j-th answer card in the batch | d numbers |
| · | dot product; equals cosine because the vectors are unit length | −1 to 1 |
| τ (tau) | temperature: divides every score; small τ sharpens the softmax | 0.01 to 1; 0.1 here |
| exp | e raised to the power | > 0 |
| Σⱼ | add up over all M answer cards | |
| the fraction | softmax share of the right card | 0 to 1 |
| log, −(1/B) Σᵢ | take −log of each row's share, then average over the rows | L ≥ 0 |

**In words:** for every question, measure what share of its softmax goes to
its own answer, take the negative log (big when the share is small), and
average over the batch.

**On the example:** B = 1, M = 2, τ = 1: L = −log(e¹ / (e¹ + e⁰)) = −log 0.731 = 0.313.

**In Python:**

```python
>>> import math
>>> def dot(a, b):
...     return sum(a_i * b_i for a_i, b_i in zip(a, b))
>>> def info_nce(q, p, tau):
...     B = len(q)
...     total = 0
...     for i in range(B):
...         exps = [math.exp(dot(q[i], p_j) / tau) for p_j in p]   # exp(q_i · p_j / τ) for every card j
...         total += math.log(exps[i] / sum(exps))                 # log of the ✔ card's share
...     return -total / B                                          # -(1/B) Σ_i
>>> q = [(1, 0)]                  # B = 1 question
>>> p = [(1, 0), (0, 1)]          # M = 2 cards; card 0 is the question's own answer
>>> round(info_nce(q, p, tau=1), 3)
0.313
>>> print(f"{info_nce(q, p, tau=0.1):.7f}")
0.0000454
```

The name InfoNCE comes from "noise-contrastive estimation": telling the true
pair apart from noise. It's the softmax cross-entropy loss from
`primer.ml.losses`, where the "classes" are the answer cards in the batch.

The model trained here is a deliberately tiny **bi-encoder** (the same
encoder applied separately to questions and answers): count the words in a
text (a **bag of words**), multiply by one learned matrix W, and scale the
result to length 1. `_contrastive_grads` and `_normalize_backward` derive the
**gradient** (the direction in which each number in W should move to lower
the loss) by hand, and `encoder_gradient_check` confirms it against a
brute-force numerical estimate.

**In code:** `info_nce_loss` computes L for a batch, treating any rows of P
beyond B as extra hard negatives. `BagOfWords` turns text into word counts,
`BiEncoder` holds the learned matrix W, and `BiEncoder.encode` maps texts to
unit vectors.

**Why it matters:** the embedding models behind search and RAG (Sentence-BERT,
DPR, E5 and their successors) are trained with exactly this loss on millions
of (question, answer) pairs. When a retrieval system confuses "reset my
password" with "password policy", this is the training signal that was
missing.

## Temperature: the sharpness dial

**Everyday picture:** grading on a curve. With a gentle curve (high τ), a
slightly better answer gets slightly more credit. With a steep curve (low τ),
the best answer takes nearly all the credit, so small differences in score
become big differences in outcome.

**Tiny example:** the right card has cosine 0.8 and three wrong cards have
0.6 each.

$$
\text{share}_+ = \frac{e^{0.8/\tau}}{e^{0.8/\tau} + 3\,e^{0.6/\tau}}
$$

**Symbols**

| Symbol | Meaning here | Range |
|---|---|---|
| share₊ | the softmax share that goes to the right card | 0 to 1 |
| 0.8, 0.6 | cosine of the right card and of each wrong card | −1 to 1 |
| 3 | the number of (identical) wrong cards | |
| τ (tau) | temperature: every cosine is divided by it | 0.01 to 1 in practice |
| e^x | e ≈ 2.718 raised to the power x | > 0 |

**In words:** the right card's share is its exponentiated, temperature-scaled
score divided by the total over all cards.

**On the example:** at τ = 1, 2.2255 / (2.2255 + 3 × 1.8221) = **0.289**; at τ = 0.05
the scores become 16 and 12, and 1 / (1 + 3e⁻⁴) = **0.948**.

**In Python:**

```python
>>> import math
>>> def share_plus(tau):
...     right = math.exp(0.8 / tau)          # e^(0.8/τ)
...     wrong = 3 * math.exp(0.6 / tau)      # three wrong cards, e^(0.6/τ) each
...     return right / (right + wrong)
>>> round(share_plus(1), 3), round(share_plus(0.05), 3)
(0.289, 0.948)
```

![Share of the right answer vs. temperature](figures/primer.ml.embeddings.contrastive.temperature.svg)

**Reading it:** the horizontal axis is τ (log scale), the vertical axis is
the right card's softmax share when it beats three wrong cards by 0.2 in
cosine. At τ = 1 a 0.2 lead earns under a third of the share, so the loss
keeps complaining about pairs that are already ranked correctly. Around
τ = 0.05 the same lead earns about 95%. Cosines live in a narrow range (−1 to
1), so embedding models use small temperatures (commonly 0.01 to 0.1) to
make those small differences count.

**In code:** `positive_probability` computes share₊ for given cosines and τ;
this curve is that function swept across τ.

**Why it matters:** too high a temperature and the model can't become
confident; too low and a few hard pairs dominate training and it becomes
unstable. It's one of the handful of settings that really matters when
fine-tuning an embedding model.

## Hard negatives: the experiment

**Everyday picture:** a driving test on an empty road teaches you to steer.
It teaches nothing about merging, because merging never came up. Negatives
only teach the distinctions they contain.

**The setup** (`train_bi_encoder`): questions come in two *intents* for each
topic: "how do I fix it?" (answered by a how-to card: "restart, reinstall
and follow the setup guide") and "what are the rules?" (answered by a policy
card: "approval, compliance and usage limits"). The question and answer
cards share almost no words, so matching intent must be *learned* ("fix"
goes with "restart"), not read off the overlap. Each batch holds one intent
across different topics, so its in-batch negatives differ from the right
card only by topic. We train two identical models; the only difference is
that one also gets each question's same-topic, other-intent card as a hard
negative. Then we test on four topics neither model has seen, including
"password".

```mermaid
flowchart TD
  D[Training pairs] --> A[Batches: one intent,<br/>8 different topics]
  A --> M1[Model 1: in-batch negatives only<br/>wrong cards differ by topic]
  A --> H[Add each question's<br/>look-alike card]
  H --> M2[Model 2: plus hard negatives<br/>wrong cards also differ by intent]
  M1 --> T[Test on unseen topics:<br/>right card vs. look-alike]
  M2 --> T
```

**Reading it:** both models see exactly the same questions and answers in
the same order. The branch on the right adds one extra wrong card per
question, a card that has the right topic but the wrong intent. Both models
then take the same exam: for a question about an unseen topic, does the
right card beat its look-alike?

![Look-alike test accuracy during training](figures/primer.ml.embeddings.contrastive.training.svg)

**Reading it:** the horizontal axis is training steps, the vertical axis is
the share of held-out questions whose right card beats the same-topic
look-alike (the dashed line is a coin flip). The in-batch-only model spends
most of its training *below* the coin flip, often preferring the wrong card,
and ends near 60%. Nothing in its training ever penalised "same topic, wrong
intent", so whatever it does on that distinction is an accident of what it
learned about topics. With hard negatives the model reaches 100% within ten
steps, *on topics it never trained on*, because it learned what "fix" and
"rules" mean.

![Where held-out questions and answers land](figures/primer.ml.embeddings.contrastive.space.svg)

**Reading it:** each panel squashes the embeddings of the four unseen topics
to 2-D with PCA (the two most spread-out directions; see `primer.notation`).
Circles are questions and squares are answer cards; blue means how-to and
orange means policy. With in-batch negatives only (left), the points group
by *topic*, and how-to and policy questions sit on top of each other. With
hard negatives (right), they split by *intent*, and each question sits next
to the kind of card that actually answers it.

![Question-by-card similarity for the unseen "password" topic](figures/primer.ml.embeddings.contrastive.heatmap.svg)

**Reading it:** rows are the ten "password" questions (five how-to, then five
policy), columns are the two password cards, and brighter means a higher
cosine. On the left the two columns are nearly the same colour: the model
sees "password" and can't tell the cards apart. On the right, the how-to
rows light up the reset card and the policy rows light up the policy card:
the diagonal blocks are exactly the right answers.

**In code:** `make_pairs` builds the (question, right card, look-alike)
triples; `look_alike_accuracy` runs the exam on unseen topics, and
`training_curve` records that score at checkpoints during training.

**Why it matters:** hard negatives are the biggest single driver of
retrieval quality. In practice you **mine** them: search your corpus with
BM25 or the current model, and take high-ranking results that are *not*
the right answer.

## Fine-tuning on your own domain

```mermaid
flowchart LR
  L[Logs: real questions +<br/>the document that resolved each] --> P[Positive pairs]
  P --> MINE[Mine hard negatives:<br/>top results that are wrong]
  MINE --> T[Fine-tune with InfoNCE,<br/>in-batch + hard negatives]
  T --> E[Evaluate recall@k<br/>on a held-out set]
  E -->|better than base model| D[Re-embed corpus, deploy]
  E -->|not better| P
```

**Reading it:** start from pairs you already have: search logs, support
tickets and the article that closed them, questions and their FAQ entries.
Mine hard negatives from your current search results. Train, then evaluate
recall@k (the share of questions whose right document is in the top k) on
held-out pairs, and only ship if it beats the base model. Shipping means
re-embedding the whole corpus, because a fine-tuned model has a new vector
space (`primer.ml.embeddings.operations`).

A few thousand good pairs is usually enough to adapt a general model to
legal, medical or internal jargon. Libraries such as sentence-transformers
implement this loss as `MultipleNegativesRankingLoss`.

## CLIP: two encoders, one shared space

**Everyday picture:** two people describing the same holiday, one through
photos and one through postcards. CLIP teaches a photo reader and a text
reader to put matching photos and captions at the same spot on one shared
map, so a photo of a dog and the words "a photo of a dog" land together.

**Tiny example:** two images and two captions, each a unit vector.
Correctly paired, image 1 scores (1, 0) against the captions and caption 1
scores (1, 0) against the images: each direction's loss is ln(1 + 1/e) =
**0.313**. Swap the captions and every right pair scores 0 against a wrong
pair's 1: the loss rises to ln(1 + e) = **1.313**.

$$
L_{\text{CLIP}} = \tfrac{1}{2}\left(L_{\text{image} \to \text{text}} + L_{\text{text} \to \text{image}}\right)
$$

**Symbols**

| Symbol | Meaning here |
|---|---|
| L_image→text | InfoNCE where each image (a row of the table) must pick its own caption from all captions in the batch |
| L_text→image | the same loss with the roles swapped: each caption (a column) must pick its own image |
| ½( … + … ) | the average of the two directions |
| L_CLIP | the loss both encoders are trained on together |

**In words:** each image must find its caption *and* each caption must find
its image.

**On the example:** ½(0.313 + 0.313) = 0.313 correctly paired; ½(1.313 + 1.313) =
1.313 with the captions swapped.

**In Python:**

```python
>>> import math
>>> def rows_pick_diagonal(S):           # InfoNCE with τ = 1: row i must pick column i
...     return -sum(math.log(math.exp(row[i]) / sum(math.exp(s) for s in row))
...                 for i, row in enumerate(S)) / len(S)
>>> def clip_loss(S):
...     columns = [list(col) for col in zip(*S)]              # captions picking images
...     return (rows_pick_diagonal(S) + rows_pick_diagonal(columns)) / 2   # ½(L_image→text + L_text→image)
>>> round(clip_loss([[1, 0], [0, 1]]), 3)                     # correctly paired
0.313
>>> round(clip_loss([[0, 1], [1, 0]]), 3)                     # captions swapped
1.313
```

**In code:** `clip_loss` computes L_CLIP by averaging `info_nce_loss` over the
rows (images pick captions) and the columns (captions pick images).

```mermaid
flowchart LR
  I[Images] --> IE[Image encoder] --> IV[Image vectors]
  T[Captions] --> TE[Text encoder] --> TV[Text vectors]
  IV --> S["Similarity table<br/>(images × captions)"]
  TV --> S
  S --> L[Symmetric InfoNCE:<br/>rows pick captions,<br/>columns pick images]
  Z["'a photo of a {label}'"] --> TE
```

**Reading it:** two different encoders, one per modality, each end in
vectors of the same length. The similarity table compares every image with
every caption in the batch, and the loss runs along the rows and down the
columns. The extra input at the bottom left is the zero-shot trick: to
classify an image with no extra training, write a caption for each label,
embed it, and pick the label whose caption is closest.

`train_clip_toy` trains two small linear encoders on toy "images" and
"captions" that describe the same hidden meaning through different, unrelated
feature spaces. Before training, zero-shot labeling is near chance (20% for
five classes); after training it's above 90%.

![Image-by-caption similarity after training](figures/primer.ml.embeddings.contrastive.clip.svg)

**Reading it:** rows are 40 test images sorted by their true class, columns
are the five label captions, and brighter means more similar. Each block of
rows lights up in exactly one column, its own label. That diagonal staircase
is zero-shot classification working: no classifier was trained, just two
encoders that agree on where meanings live.

**Why it matters:** the shared space enables search across modalities
(find images with text, or text with images), zero-shot classification,
and mixing scanned document images with text in one retrieval index.
Multimodal models use the same recipe.

## In 20 seconds
- Contrastive training pulls matching pairs together and pushes non-matches
  apart; the loss (InfoNCE) is softmax cross-entropy over the batch.
- In-batch negatives are free: every other answer in the batch.
- Negatives only teach the distinctions they contain. Hard negatives (same
  topic, wrong answer) are what teach retrieval rather than topic matching.
- Temperature sharpens the softmax so small cosine gaps count; typical τ is
  0.01 to 0.1.
- CLIP applies the same loss in both directions between an image encoder and
  a text encoder, which gives one shared space and zero-shot classification.

## Self-test questions

**Q: How are embedding models trained?**
Contrastively, on (query, relevant passage) pairs: an encoder embeds both,
and InfoNCE rewards the pair's similarity over the similarity to other
passages in the batch (in-batch negatives) and to deliberately chosen hard
negatives.

**Q: Why do hard negatives matter so much?**
Random negatives are usually about something else, so a model can beat them
by topic matching alone. Hard negatives share the topic but don't answer
the question, forcing the model to learn the fine distinction that
retrieval actually needs.

**Q: What does the temperature do in InfoNCE?**
It divides the similarities before softmax. A small temperature makes the
softmax sharp, so small cosine differences produce large differences in
probability and gradient. Too small makes training unstable; too large
makes the model unable to become confident.

**Q: How does CLIP put images and text in the same space?**
It trains an image encoder and a text encoder together on image-caption
pairs with a symmetric contrastive loss: each image must pick its caption
from the batch and each caption must pick its image. Matching pairs end up
close in one shared space.

**Q: How would you adapt an embedding model to a company's jargon?**
Collect real (query, correct document) pairs from logs, mine hard negatives
from current search results, fine-tune with in-batch plus hard negatives,
evaluate recall@k on a held-out set against the base model, then re-embed
the corpus with the new model.

## The papers behind this lesson

- **van den Oord, Li and Vinyals, *Representation Learning with Contrastive Predictive Coding* (2018)**: https://arxiv.org/abs/1807.03748.
  Named and popularised the InfoNCE loss: pick the true sample out of a set of negatives.
- **Reimers and Gurevych, *Sentence-BERT* (2019)**: https://arxiv.org/abs/1908.10084.
  Turned BERT into a bi-encoder that produces one comparable vector per sentence, making semantic search with transformers fast. [annotated companion](../../../papers/sentence-bert.html)
- **Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question Answering* (2020)**: https://arxiv.org/abs/2004.04906.
  Showed a question/passage bi-encoder trained with in-batch negatives plus BM25-mined hard negatives beats keyword search for open-domain QA. [annotated companion](../../../papers/dpr.html)
- **Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP, 2021)**: https://arxiv.org/abs/2103.00020.
  Trained image and text encoders with a symmetric contrastive loss on 400 million pairs, giving zero-shot image classification. [annotated companion](../../../papers/clip.html)
- **Gao, Yao and Chen, *SimCSE* (2021)**: https://arxiv.org/abs/2104.08821.
  Showed contrastive learning of sentence embeddings works even with dropout noise as the only "augmentation", and with NLI pairs as hard negatives.

## Further reading
- sentence-transformers, training overview: https://www.sbert.net/docs/sentence_transformer/training_overview.html
- Wang et al., *Text Embeddings by Weakly-Supervised Contrastive Pre-training* (E5, 2022): https://arxiv.org/abs/2212.03533
- Muennighoff et al., *MTEB: Massive Text Embedding Benchmark* (2022): https://arxiv.org/abs/2210.07316
- MTEB leaderboard: https://huggingface.co/spaces/mteb/leaderboard
- Lilian Weng, *Contrastive Representation Learning*: https://lilianweng.github.io/posts/2021-05-31-contrastive/
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from primer._show import banner, say, table, takeaway
from primer.common.text import tokenize

# ---------------------------------------------------------------------------
# 1. The InfoNCE loss, and what the temperature does
# ---------------------------------------------------------------------------


def _log_softmax(S: np.ndarray) -> np.ndarray:
    S = S - S.max(axis=1, keepdims=True)  # subtract the row max so exp() can't overflow
    return S - np.log(np.exp(S).sum(axis=1, keepdims=True))


def info_nce_loss(Q: np.ndarray, P: np.ndarray, tau: float) -> float:
    """Mean InfoNCE loss. Row i of Q (a query) should pick row i of P (its passage).

    Q: (B, d) query embeddings. P: (M, d) passage embeddings with M >= B;
    rows B..M-1 are extra negatives (e.g. hard negatives) that belong to no query.
    Every other row of P acts as a negative for query i: that's "in-batch negatives".
    """
    S = Q @ P.T / tau  # (B, M) scaled similarities; the diagonal holds the true pairs
    B = Q.shape[0]
    return float(-_log_softmax(S)[np.arange(B), np.arange(B)].mean())


def positive_probability(pos: float, negs: list[float], tau: float) -> float:
    """Softmax share the positive gets when its cosine is `pos` and the negatives' are `negs`."""
    s = np.array([pos, *negs]) / tau
    e = np.exp(s - s.max())
    return float(e[0] / e.sum())


# ---------------------------------------------------------------------------
# 2. A tiny bi-encoder: bag of words -> linear projection -> unit vector
# ---------------------------------------------------------------------------


class BagOfWords:
    """Turns text into a count vector over a fixed vocabulary (one slot per word)."""

    def __init__(self, texts: list[str]):
        self.vocab = sorted({t for text in texts for t in tokenize(text)})
        self.index = {w: i for i, w in enumerate(self.vocab)}

    def __call__(self, texts: list[str]) -> np.ndarray:
        X = np.zeros((len(texts), len(self.vocab)))
        for r, text in enumerate(texts):
            for t in tokenize(text):
                if t in self.index:
                    X[r, self.index[t]] += 1
        return X


def _normalize_rows(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    return A / norms, norms


def _normalize_backward(dU: np.ndarray, U: np.ndarray, norms: np.ndarray) -> np.ndarray:
    """Gradient through u = a / |a|: remove the part of dU along u, then divide by |a|.

    Changing a along its own direction doesn't change u (it's rescaled away),
    so only the sideways part of the gradient survives.
    """
    return (dU - U * np.sum(U * dU, axis=1, keepdims=True)) / norms


def _contrastive_grads(Q: np.ndarray, P: np.ndarray, tau: float, symmetric: bool = False):
    """Gradients of the (optionally symmetric) InfoNCE loss w.r.t. Q and P.

    Softmax cross-entropy has a famously simple gradient: probabilities minus
    the one-hot target. Everything else here is the chain rule through S = Q Pᵀ / τ.
    """
    B, M = Q.shape[0], P.shape[0]
    S = Q @ P.T / tau
    Y = np.zeros((B, M))
    Y[np.arange(B), np.arange(B)] = 1.0
    dS = (np.exp(_log_softmax(S)) - Y) / B
    if symmetric:  # CLIP: also each caption picks its image (columns); needs M == B
        dS = 0.5 * dS + 0.5 * ((np.exp(_log_softmax(S.T)) - Y.T) / B).T
    return dS @ P / tau, dS.T @ Q / tau


@dataclass
class BiEncoder:
    """One shared projection W for queries and passages (a "Siamese" bi-encoder)."""

    bow: BagOfWords
    W: np.ndarray  # (vocab, d)

    def encode(self, texts: list[str]) -> np.ndarray:
        return _normalize_rows(self.bow(texts) @ self.W)[0]


def _loss_and_grad(W: np.ndarray, Xq: np.ndarray, Xp: np.ndarray, tau: float) -> tuple[float, np.ndarray]:
    Aq, Ap = Xq @ W, Xp @ W
    Q, nq = _normalize_rows(Aq)
    P, np_ = _normalize_rows(Ap)
    loss = info_nce_loss(Q, P, tau)
    dQ, dP = _contrastive_grads(Q, P, tau)
    dW = Xq.T @ _normalize_backward(dQ, Q, nq) + Xp.T @ _normalize_backward(dP, P, np_)
    return loss, dW


def encoder_gradient_check(seed: int = 0, eps: float = 1e-6) -> float:
    """Max |analytic - numerical| gradient of the loss w.r.t. W on random data."""
    rng = np.random.default_rng(seed)
    Xq, Xp = rng.random((3, 6)), rng.random((5, 6))
    W = rng.standard_normal((6, 4))
    _, dW = _loss_and_grad(W, Xq, Xp, 0.5)
    worst = 0.0
    for idx in np.ndindex(W.shape):
        E = np.zeros_like(W)
        E[idx] = eps
        num = (_loss_and_grad(W + E, Xq, Xp, 0.5)[0] - _loss_and_grad(W - E, Xq, Xp, 0.5)[0]) / (2 * eps)
        worst = max(worst, abs(num - dW[idx]))
    return worst


# ---------------------------------------------------------------------------
# 3. Training data with look-alike passages
# ---------------------------------------------------------------------------

TRAIN_TOPICS = ["vpn", "laptop", "expense", "pto", "invoice", "printer", "email", "payroll", "badge", "wifi", "travel", "parking"]
HELDOUT_TOPICS = ["password", "monitor", "phone", "desk"]

# Queries and passages for the same intent share almost no words on purpose,
# so matching intent has to be *learned* (fix ~ restart), not read off overlap.
QUERY_TEMPLATES = {
    "howto": ["how do i fix my {t}", "my {t} is broken", "{t} not working help", "i am stuck with my {t}", "{t} keeps failing"],
    "policy": ["what are the {t} rules", "is there a limit on {t}", "who can get {t}", "{t} rules for my team", "am i permitted to have {t}"],
}
PASSAGE_TEMPLATES = {
    "howto": "{T}: restart, reinstall and follow the setup guide.",
    "policy": "{T}: approval, compliance and usage limits for all staff.",
}
OTHER = {"howto": "policy", "policy": "howto"}


def make_pairs(topics: list[str]) -> list[tuple[str, str, str]]:
    """(query, right passage, look-alike wrong passage) triples.

    The look-alike has the same topic and the other intent: the classic hard
    negative ("reset my password" vs. "password policy").
    """
    out = []
    for t in topics:
        for intent, templates in QUERY_TEMPLATES.items():
            for q in templates:
                out.append(
                    (q.format(t=t), PASSAGE_TEMPLATES[intent].format(T=t.capitalize()), PASSAGE_TEMPLATES[OTHER[intent]].format(T=t.capitalize()))
                )
    return out


def train_bi_encoder(
    hard_negatives: bool, dim: int = 16, tau: float = 0.1, steps: int = 400, batch: int = 8, lr: float = 0.5, seed: int = 0
) -> BiEncoder:
    """Train the shared projection with InfoNCE.

    A controlled experiment. Every batch holds one intent (all how-to or all
    policy questions) across different topics, so its in-batch negatives
    differ from the right passage *only by topic*. That mirrors real data,
    where random negatives are nearly always "about something else" and
    rarely "same subject, different answer". Negatives only teach the
    distinctions they contain, so this model can succeed by matching topics
    and never learns intent. With `hard_negatives=True` each query's
    same-topic, other-intent look-alike is appended to the batch as an extra
    negative; that is the only difference between the two runs.
    """
    rng = np.random.default_rng(seed)
    train = make_pairs(TRAIN_TOPICS)
    everything = train + make_pairs(HELDOUT_TOPICS)
    bow = BagOfWords([x for triple in everything for x in triple])
    W = rng.normal(0, 1 / np.sqrt(dim), (len(bow.vocab), dim))

    # (topic, intent) -> its triples. Intent is recovered from which passage is "right".
    groups: dict[tuple[str, str], list] = {}
    for t in TRAIN_TOPICS:
        for intent in QUERY_TEMPLATES:
            right = PASSAGE_TEMPLATES[intent].format(T=t.capitalize())
            groups[(t, intent)] = [p for p in train if p[1] == right]
    for _ in range(steps):
        intent = ("howto", "policy")[rng.integers(2)]
        topics = rng.choice(TRAIN_TOPICS, size=batch, replace=False)
        chosen = [groups[(t, intent)][rng.integers(len(groups[(t, intent)]))] for t in topics]
        queries = [c[0] for c in chosen]
        passages = [c[1] for c in chosen] + ([c[2] for c in chosen] if hard_negatives else [])
        _, dW = _loss_and_grad(W, bow(queries), bow(passages), tau)
        W -= lr * dW
    return BiEncoder(bow, W)


def look_alike_accuracy(enc: BiEncoder, topics: list[str] = HELDOUT_TOPICS) -> float:
    """On unseen topics: how often does the right passage beat its same-topic look-alike?"""
    triples = make_pairs(topics)
    Q = enc.encode([t[0] for t in triples])
    R = enc.encode([t[1] for t in triples])
    Wr = enc.encode([t[2] for t in triples])
    return float(np.mean(np.sum(Q * R, axis=1) > np.sum(Q * Wr, axis=1)))


# ---------------------------------------------------------------------------
# 4. CLIP: two encoders, one shared space
# ---------------------------------------------------------------------------


def clip_loss(I: np.ndarray, T: np.ndarray, tau: float) -> float:
    """Symmetric InfoNCE: each image picks its caption AND each caption picks its image."""
    I, T = _normalize_rows(I)[0], _normalize_rows(T)[0]
    return 0.5 * info_nce_loss(I, T, tau) + 0.5 * info_nce_loss(T, I, tau)


def train_clip_toy(n_classes: int = 5, dim: int = 16, steps: int = 300, batch: int = 32, tau: float = 0.1, lr: float = 0.5, seed: int = 0) -> dict:
    """Two linear encoders learn a shared space from (image, caption) pairs.

    Toy data: each example has a hidden meaning z (its class prototype plus
    noise). The "image" sees z through one random mixing matrix, the "caption"
    through another, in different sizes, so the raw features live in unrelated
    spaces. Training aligns them. Zero-shot test: embed one caption per class
    ("a photo of a <class>", i.e. the clean prototype) and label each held-out
    image by its nearest caption.
    """
    rng = np.random.default_rng(seed)
    latent, img_dim, txt_dim = 8, 32, 24
    protos = rng.standard_normal((n_classes, latent))
    A, Bm = rng.standard_normal((latent, img_dim)), rng.standard_normal((latent, txt_dim))

    def sample(n):
        y = rng.integers(n_classes, size=n)
        z = protos[y] + 0.5 * rng.standard_normal((n, latent))
        img = z @ A + 0.3 * rng.standard_normal((n, img_dim))
        txt = z @ Bm + 0.3 * rng.standard_normal((n, txt_dim))
        return img, txt, y

    Wi = rng.normal(0, 1 / np.sqrt(dim), (img_dim, dim))
    Wt = rng.normal(0, 1 / np.sqrt(dim), (txt_dim, dim))
    for _ in range(steps):
        img, txt, _ = sample(batch)
        Ai, At = img @ Wi, txt @ Wt
        I, ni = _normalize_rows(Ai)
        T, nt = _normalize_rows(At)
        dI, dT = _contrastive_grads(I, T, tau, symmetric=True)
        Wi -= lr * img.T @ _normalize_backward(dI, I, ni)
        Wt -= lr * txt.T @ _normalize_backward(dT, T, nt)

    test_img, _, test_y = sample(500)
    label_txt = protos @ Bm  # one clean "caption" per class
    I = _normalize_rows(test_img @ Wi)[0]
    L = _normalize_rows(label_txt @ Wt)[0]
    pred = np.argmax(I @ L.T, axis=1)
    return {"zero_shot_accuracy": float(np.mean(pred == test_y)), "Wi": Wi, "Wt": Wt, "similarity": I[:40] @ L.T, "labels": test_y[:40]}


# ---------------------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------------------


def training_curve(hard_negatives: bool, checkpoints=(0, 10, 25, 50, 100, 200, 400)) -> list[float]:
    """Held-out look-alike accuracy after each number of steps (each a fresh, seeded run)."""
    return [look_alike_accuracy(train_bi_encoder(hard_negatives, steps=s)) for s in checkpoints]


def _pca_2d(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def figures() -> dict:
    """Plots computed from this module's own functions. Keys match the docstring's image names."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = {}
    models = {"in-batch negatives only": train_bi_encoder(False), "plus hard negatives": train_bi_encoder(True)}

    # temperature
    taus = np.logspace(-2.3, 0.3, 60)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(taus, [positive_probability(0.8, [0.6] * 3, t) for t in taus])
    ax.set_xscale("log")
    ax.set(xlabel="temperature τ (log scale)", ylabel="softmax share of the right card", title="Right card at cos 0.8 vs. three wrong at 0.6")
    figs["temperature"] = fig

    # training curves
    cps = (0, 10, 25, 50, 100, 200, 400)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for hard, label in ((False, "in-batch negatives only"), (True, "plus hard negatives")):
        ax.plot(cps, training_curve(hard, cps), marker="o", label=label)
    ax.axhline(0.5, ls="--", color="0.6", label="coin flip")
    ax.set(xlabel="training steps", ylabel="right card beats look-alike (unseen topics)", ylim=(0, 1.05), title="Negatives only teach the distinctions they contain")
    ax.legend()
    figs["training"] = fig

    # space: PCA of held-out questions and cards, coloured by intent
    triples = make_pairs(HELDOUT_TOPICS)
    queries = [t[0] for t in triples]
    q_intent = ["howto" if t[1].endswith("setup guide.") else "policy" for t in triples]
    cards = sorted({t[1] for t in triples})
    c_intent = ["howto" if c.endswith("setup guide.") else "policy" for c in cards]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (title, enc) in zip(axes, models.items()):
        P = _pca_2d(enc.encode(queries + cards))
        for pts, intents, marker, size in ((P[: len(queries)], q_intent, "o", 25), (P[len(queries) :], c_intent, "s", 90)):
            for intent, color in (("howto", "C0"), ("policy", "C1")):
                m = np.array(intents) == intent
                ax.scatter(pts[m, 0], pts[m, 1], marker=marker, s=size, color=color, alpha=0.75, edgecolor="k" if marker == "s" else None)
        ax.set(title=title, xlabel="principal component 1", ylabel="principal component 2")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color="C0", label="how-to question"),
        plt.Line2D([], [], marker="o", ls="", color="C1", label="policy question"),
        plt.Line2D([], [], marker="s", ls="", color="C0", mec="k", label="how-to card"),
        plt.Line2D([], [], marker="s", ls="", color="C1", mec="k", label="policy card"),
    ]
    axes[1].legend(handles=handles, loc="best", fontsize=8)
    figs["space"] = fig

    # heatmap: password questions x password cards
    pw = make_pairs(["password"])
    pw_q = [t[0] for t in pw]
    pw_cards = [PASSAGE_TEMPLATES["howto"].format(T="Password"), PASSAGE_TEMPLATES["policy"].format(T="Password")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    for ax, (title, enc) in zip(axes, models.items()):
        S = enc.encode(pw_q) @ enc.encode(pw_cards).T
        im = ax.imshow(S, cmap="viridis", vmin=-0.2, vmax=1.0, aspect="auto")
        ax.set_xticks([0, 1], ["reset card", "policy card"])
        ax.set_yticks(range(len(pw_q)), pw_q, fontsize=7)
        ax.set(title=title)
    fig.colorbar(im, ax=axes, label="cosine similarity")
    figs["heatmap"] = fig

    # clip
    r = train_clip_toy()
    order = np.argsort(r["labels"], kind="stable")
    fig, ax = plt.subplots(figsize=(5, 6))
    im = ax.imshow(r["similarity"][order], cmap="viridis", aspect="auto")
    ax.set_xticks(range(5), [f"'a photo of\nclass {k}'" for k in range(5)], fontsize=7)
    ax.set(ylabel="test images, sorted by true class", title=f"Zero-shot accuracy {r['zero_shot_accuracy']:.0%}")
    fig.colorbar(im, ax=ax, label="cosine similarity")
    figs["clip"] = fig

    for name, f in figs.items():
        if name != "heatmap":  # heatmap uses a shared colorbar across axes, which tight_layout can't place
            f.tight_layout()
    return figs


# ---------------------------------------------------------------------------
# 6. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. Worked example: one question, two answer cards")
    q, cards = np.array([[1.0, 0.0]]), np.array([[1.0, 0.0], [0.0, 1.0]])
    table(["temperature τ", "scores", "share of right card", "InfoNCE loss"],
          [(t, f"({1 / t:g}, 0)", positive_probability(1.0, [0.0], t), info_nce_loss(q, cards, t)) for t in (1.0, 0.1)], floatfmt=".6f")
    takeaway("Same vectors, lower temperature: the softmax gets sharper and the loss drops.")

    banner("2. Temperature: right card at cos 0.8, three wrong cards at 0.6")
    table(["τ", "share of right card"], [(t, positive_probability(0.8, [0.6] * 3, t)) for t in (1.0, 0.5, 0.1, 0.05, 0.02)], floatfmt=".3f")

    banner("3. The hand-written gradient is right")
    say(f"Max difference from a numerical estimate: {encoder_gradient_check():.1e}.")

    banner("4. Hard negatives: the controlled experiment")
    say(
        """
        Two identical bi-encoders, same batches. Each batch holds one intent
        across 8 topics, so in-batch negatives differ only by topic. Model 2
        also gets each question's same-topic, other-intent card. The test:
        on 4 unseen topics, does the right card beat its look-alike?
        """
    )
    easy, hard = train_bi_encoder(False), train_bi_encoder(True)
    table(["model", "unseen topics", "training topics"],
          [("in-batch negatives only", look_alike_accuracy(easy), look_alike_accuracy(easy, TRAIN_TOPICS)),
           ("plus hard negatives", look_alike_accuracy(hard), look_alike_accuracy(hard, TRAIN_TOPICS))], floatfmt=".2f")
    texts = ["my password is broken and i am stuck", "Password: restart, reinstall and follow the setup guide.", "Password: approval, compliance and usage limits for all staff."]
    rows = []
    for name, enc in (("in-batch only", easy), ("plus hard negatives", hard)):
        v = enc.encode(texts)
        rows.append((name, v[0] @ v[1], v[0] @ v[2]))
    say(f"Query: '{texts[0]}'")
    table(["model", "cos to reset card", "cos to policy card"], rows, floatfmt=".3f")
    takeaway("Negatives only teach the distinctions they contain. Hard negatives teach retrieval, not topic matching.")

    banner("5. CLIP: two encoders, one shared space")
    say(f"Symmetric loss, 2 pairs, correctly paired: {clip_loss(np.eye(2), np.eye(2), 1.0):.4f}; captions swapped: {clip_loss(np.eye(2), np.eye(2)[::-1], 1.0):.4f}.")
    before, after = train_clip_toy(steps=0), train_clip_toy()
    table(["encoders", "zero-shot accuracy (5 classes)"], [("untrained", before["zero_shot_accuracy"]), ("after contrastive training", after["zero_shot_accuracy"])], floatfmt=".2f")
    takeaway("Label an image by embedding one caption per class and picking the nearest: no classifier training needed.")


if __name__ == "__main__":
    demo()
