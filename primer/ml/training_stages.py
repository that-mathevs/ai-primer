r"""
# Training stages: how a text predictor becomes an assistant

Run: `python -m primer.ml.training_stages`

New to the notation? `primer.notation` explains every symbol used here
(Σ, log, σ, subscripts, ᵀ, and so on) from zero.

## The idea

A chat assistant is built in layers, and each layer explains a different
behaviour you see:

```mermaid
flowchart LR
  P[Pretraining<br/>guess the next token<br/>over web-scale text] --> B[Base model<br/>knows a lot, rambles]
  B --> S[SFT<br/>study examples of<br/>good answers]
  S --> PT[Preference tuning<br/>RLHF or DPO]
  PT --> A[Assistant model]
  A --> D[Your adaptation<br/>prompts, RAG, LoRA]
```

**Reading it:** read left to right; each box is a training stage and each
arrow hands a model to the next stage. The model *vendor* runs the first
three stages: pretraining (months, millions of dollars) gives a base model
that knows facts but just continues text; supervised fine-tuning (SFT)
teaches it the *format* of being an assistant; preference tuning shapes tone,
helpfulness and safety. You work in the last box. When a model knows a fact
it learned that in pretraining; when it answers in tidy bullet points it
learned that in SFT and preference tuning.

This lesson builds a small, real version of every stage: the pretraining
and SFT losses, a reward model, DPO, LoRA, the "which adaptation?" decision,
and distillation.

## 1. Pretraining: guess the next word, a trillion times

**Everyday picture.** Imagine reading every book in a library with a card
covering the next word, guessing it, then sliding the card to check. Do
that trillions of times and you absorb grammar, facts, arithmetic habits and
coding conventions, because every one of them helps you guess the next word.

**Tiny worked example.** A 5-token sequence gives 4 guesses (token 1
predicts token 2, and so on; the last token has nothing after it to check).
Suppose the model gave the right next token probability 0.25, 0.25, 0.5 and
0.5. The penalty for each guess is −ln(probability): 1.386, 1.386, 0.693,
0.693. The pretraining loss is their average, **1.040**.

```mermaid
flowchart LR
  T1[t1] --> G1{guess t2}
  T2[t2] --> G2{guess t3}
  T3[t3] --> G3{guess t4}
  T4[t4] --> G4{guess t5}
  G1 & G2 & G3 & G4 --> AVG[average of −ln p<br/>= the loss]
```

**Reading it:** every position makes one guess about the token that follows
it, using only what came before. The model is scored on all guesses at
once and the average penalty is the number training pushes down. One text
sequence of n tokens therefore gives n − 1 training examples for free, which
is why unlabeled text is such a rich training signal.

$$
\mathcal{L}_{\text{pretrain}} = -\frac{1}{n-1}\sum_{i=1}^{n-1} \log p_\theta(t_{i+1} \mid t_{1..i})
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| $\mathcal{L}_{\text{pretrain}}$ | the loss: one number, lower is better | ≥ 0 |
| $n$ | number of tokens in the sequence | integer |
| $\Sigma_{i=1}^{n-1}$ | add up the term for every position i from 1 to n−1 | |
| $t_{i}$ | the i-th token (an integer id) | 0 … vocab−1 |
| $t_{1..i}$ | all tokens up to and including position i (the context) | |
| $p_\theta(t_{i+1} \mid t_{1..i})$ | probability the model with weights θ gives the true next token, given the context. The bar "∣" reads "given" | 0 … 1 |
| $\theta$ | all the model's weights | millions to trillions of numbers |
| $\log$ | natural logarithm. ln 1 = 0 and ln of a small number is very negative, so −log turns "probability of being right" into "penalty" | |

**In words:** the loss is the average, over every position, of minus the
log of the probability the model gave to the token that actually came next.

**On the worked example:** n = 5, the four probabilities are 0.25, 0.25,
0.5, 0.5, so the loss is −(ln 0.25 + ln 0.25 + ln 0.5 + ln 0.5) / 4 =
(1.386 + 1.386 + 0.693 + 0.693) / 4 = 1.040.

**Why it matters in practice.** Pretraining is where knowledge comes from,
and it is frozen at a date (the "knowledge cutoff"). A base model continues
text instead of answering questions: ask "What is the capital of France?"
and it may continue with "What is the capital of Spain?", because question
lists are common on the web.

## 2. Supervised fine-tuning (SFT): study the answers, not the questions

**Everyday picture.** An apprentice studies a binder of worked examples:
customer question, then the expert's reply. They read the question
carefully, but they are graded only on writing the reply. Nobody marks them
on reproducing the customer's typing.

**Tiny worked example.** Same 5 tokens, but the first 3 are the prompt and
the last 2 are the reply. Only the 2 guesses whose target is a reply token
count: penalties 0.693 and 0.693, average **0.693**. The prompt guesses
(1.386 each) are ignored.

```mermaid
flowchart LR
  subgraph Prompt["prompt: read, not graded"]
    P1[t1] --> P2[t2] --> P3[t3]
  end
  subgraph Reply["reply: graded"]
    R1[t4] --> R2[t5]
  end
  P3 --> R1
  R1 -.->|loss| L[average −ln p<br/>over reply tokens only]
  R2 -.->|loss| L
```

**Reading it:** the whole sequence flows through the model, so the reply is
conditioned on the prompt. The dotted arrows show which predictions reach
the loss: only those that produce reply tokens. That mask (a list of
true/false per position) is the whole difference between pretraining code
and SFT code.

$$
\mathcal{L}_{\text{SFT}} = -\frac{\sum_{i} m_i \,\log p_\theta(t_{i+1} \mid t_{1..i})}{\sum_i m_i}
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| $m_i$ | the mask: 1 if token i+1 belongs to the reply, else 0 | 0 or 1 |
| $\sum_i m_i$ | how many predictions are graded (the reply length) | integer |
| everything else | as in the pretraining formula | |

**In words:** average the next-token penalty over the reply tokens only.

**On the worked example:** m = (0, 0, 1, 1), so the loss is
(0.693 + 0.693) / 2 = 0.693.

![Per-position loss, with prompt positions masked out](figures/primer.ml.training_stages.sft_mask.svg)

**Reading it:** each bar is one next-token prediction from the worked
example, and its height is the penalty −ln p. Grey bars predict prompt
tokens and are masked out of the SFT loss; blue bars predict reply tokens
and are the only ones that count. The dashed lines mark the two averages:
1.040 over everything (pretraining) and 0.693 over the reply (SFT).

**Why it matters in practice.** SFT needs far less data than pretraining
(thousands to hundreds of thousands of examples) and quality beats quantity:
a few thousand excellent examples outperform a mountain of mediocre ones.
If you fine-tune your own model and forget the mask, it learns to write
user prompts too.

## 3. Preference tuning: a taste test instead of a recipe

**Everyday picture.** It is hard to write down the perfect answer, but easy
to taste two dishes and say which is better. Preference tuning collects
exactly those judgements: people (or a model following written principles)
compare two responses and pick one.

### 3a. RLHF: train a critic, then train the cook against it

A **reward model** learns to predict those judgements, giving each response
a score. Reinforcement learning then tunes the language model to earn high
scores. The link between scores and "which one wins" is the Bradley-Terry
model, built on the **sigmoid** function σ(z) = 1 / (1 + e^−z), which squashes
any number into a probability between 0 and 1 (σ(0) = 0.5, σ(2) = 0.881).

**Tiny worked example.** The reward model scores the chosen answer 3.0 and
the rejected one 1.0. The gap is 2, so it predicts the chosen answer wins
with probability σ(2) = **0.881**, and its loss on this pair is
−ln 0.881 = **0.127**. Had it scored both 1.0, it would predict a coin flip
(0.5) and pay ln 2 = 0.693.

```mermaid
flowchart LR
  PR[Prompt] --> LM[Language model]
  LM --> RA[Response A] & RB[Response B]
  RA & RB --> H[Human or AI labeler<br/>picks the better one]
  H --> RM[Reward model<br/>learns to score responses]
  RM --> RL[Reinforcement learning<br/>tune the LM to score high]
  RL --> LM
```

**Reading it:** the loop has two learners. First, labelers compare pairs and
the reward model learns to agree with them. Second, the language model
generates new responses, the reward model scores them, and reinforcement
learning (usually PPO) nudges the language model towards higher scores,
with a penalty for drifting far from where it started. Two models, two
training runs, lots of moving parts.

$$
P(y_w \succ y_l) = \sigma\big(r(y_w) - r(y_l)\big) \qquad
\mathcal{L}_{\text{RM}} = -\log \sigma\big(r(y_w) - r(y_l)\big)
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| $y_w, y_l$ | the preferred ("winning") and rejected ("losing") responses | text |
| $\succ$ | "is preferred to" | |
| $r(y)$ | the reward model's score for response y | any real number |
| $\sigma(z)$ | sigmoid, $1/(1+e^{-z})$: turns a score gap into a probability | 0 … 1 |
| $e$ | Euler's number, 2.718…; $e^{-z}$ is "e to the power −z" | |
| $\mathcal{L}_{\text{RM}}$ | the reward model's loss on this pair | ≥ 0 |

**In words:** the chance the preferred answer wins is the sigmoid of the
score gap, and the reward model is penalised by minus the log of the
probability it gave to the choice people actually made.

**On the worked example:** r(y_w) = 3, r(y_l) = 1, gap 2, σ(2) = 0.881,
loss −ln 0.881 = 0.127.

### 3b. DPO: skip the critic

**Everyday picture.** Instead of hiring a food critic and cooking to please
them, edit the recipe book directly from the taste-test results, while
keeping a copy of the original book so you don't drift too far from it.

**Tiny worked example.** A **log-probability** is the log of the
probability the model gives a whole response (a sum of per-token log
probabilities; more negative means less likely). The frozen reference
model gives both answers −11. After some training the policy gives the
chosen answer −10 (more likely than before) and the rejected one −12 (less
likely). With β = 0.1 the margin is 0.1 × ((−10 − −11) − (−12 − −11)) =
0.1 × (1 + 1) = **0.2**; the loss is −ln σ(0.2) = **0.598**, down from ln 2 =
0.693 when the policy still equalled the reference.

```mermaid
flowchart LR
  PAIR[Preference pair<br/>chosen, rejected] --> POL[Policy being trained<br/>log π of each]
  PAIR --> REF[Frozen reference<br/>log π_ref of each]
  POL --> M[Margin = β × how much more the policy<br/>boosted chosen than rejected]
  REF --> M
  M --> LOSS[−log σ margin]
  LOSS -->|gradient| POL
```

**Reading it:** each preference pair is scored twice: by the model being
trained and by a frozen copy of where it started. The margin measures how
much more the policy has boosted the chosen answer than the rejected one,
*relative to the reference*. The loss is the reward-model loss again, but
the "reward" is read straight off the policy's own probabilities, so there
is no separate reward model and no reinforcement-learning loop.

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\Big(\beta\big[(\log\pi_\theta(y_w) - \log\pi_{\text{ref}}(y_w)) - (\log\pi_\theta(y_l) - \log\pi_{\text{ref}}(y_l))\big]\Big)
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| $\pi_\theta(y)$ | the policy: the model being trained, with weights θ; π(y) is the probability it gives response y | 0 … 1 |
| $\pi_{\text{ref}}(y)$ | the frozen reference model (usually the SFT model) | 0 … 1 |
| $\log\pi_\theta(y) - \log\pi_{\text{ref}}(y)$ | the implicit reward: how much more likely training has made y | any real |
| $\beta$ | beta, the strength knob: larger means preferences push harder relative to staying near the reference | typically 0.1 to 0.5 |
| $\sigma$, $\log$ | sigmoid and natural log, as above | |

**In words:** raise the probability of the chosen answer and lower the
rejected one, measured relative to the frozen starting model, and penalise
the model by minus the log-sigmoid of that scaled gap.

**On the worked example:** β = 0.1, implicit rewards +1 (chosen) and −1
(rejected), margin 0.2, σ(0.2) = 0.550, loss 0.598. The gradient's size is
β × (1 − σ(margin)) = 0.1 × 0.450 = **0.045**: pairs the policy already
ranks correctly get gentle updates, and pairs it ranks the wrong way get
strong ones.

![Update strength falls as the policy learns the preference](figures/primer.ml.training_stages.dpo_strength.svg)

**Reading it:** the x-axis is the DPO margin (how strongly the policy
already prefers the chosen answer, relative to the reference); the y-axis is
how hard one gradient step pushes. Far left, the policy has the pair
backwards and gets the full push, β. At zero (no preference yet) it gets
half. Far right, the pair is learned and updates fade to nothing, so
training effort flows automatically to the pairs still wrong.

![DPO moves probability towards the preferred answer](figures/primer.ml.training_stages.dpo_training.svg)

**Reading it:** a one-prompt toy policy starts uniform over three answers
(1/3 each). The preference data says "helpful and correct" beats both
others and "correct but rambling" beats "rude". As training steps pass
(x-axis), probability (y-axis) flows to the top answer, the rude answer is
pushed down fastest, and the rambling answer settles in between: exactly
the ordering people expressed.

**Why it matters in practice.** DPO is simpler and more stable than RLHF,
which is why it is widely used in open-model fine-tuning. Constitutional
AI and AI-feedback methods scale the labelling by having a model judge
responses against written principles. This stage is where tone,
helpfulness, refusals and safety behaviour mostly come from.

## 4. LoRA: sticky notes instead of reprinting the textbook

**Everyday picture.** You want to adapt a 1,000-page textbook for your
class. Reprinting it is expensive. Instead you add a small stack of sticky
notes with corrections. The book stays untouched; the notes are cheap to
write, store and swap, and you can photocopy them into the book when you're
done.

**Tiny worked example.** Frozen weights W = the 2×2 identity (it copies its
input). Adapter B = (1, 0) as a column and A = (0, 1) as a row. For input
x = (1, 2): the frozen path gives W·x = (1, 2). The adapter first squeezes x
to one number, A·x = 2, then expands it back, B·2 = (2, 0). The output is
(1, 2) + (2, 0) = **(3, 2)**. Two small vectors changed the layer's
behaviour without touching W.

```mermaid
flowchart LR
  X[input x<br/>d numbers] --> W[W, frozen<br/>d × d]
  X --> A[A, trainable<br/>squeeze to r numbers]
  A --> B[B, trainable, starts at 0<br/>expand back to d]
  W --> ADD((+))
  B --> ADD
  ADD --> Y[output y]
```

**Reading it:** the input takes two paths. The top path is the pretrained
layer, which never changes. The bottom path is the adapter: A squeezes the
d-dimensional input down to a tiny rank r (say 8), and B expands it back.
Because B starts at zero, the bottom path adds nothing at first and the
model starts exactly where pretraining left it. Only A and B are trained.
Afterwards you can add B·A into W once ("merge") and serve the model at
exactly the original speed.

$$
y = xW^\top + \frac{\alpha}{r}\, x A^\top B^\top \qquad
\text{trainable parameters} = r\,d_{\text{in}} + d_{\text{out}}\,r
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| $x$ | input row vector | (d_in,) |
| $W$ | frozen pretrained weight matrix | (d_out, d_in) |
| $^\top$ | transpose: flip rows and columns so the shapes line up for multiplication | |
| $A$ | trainable "down" matrix, random at start | (r, d_in) |
| $B$ | trainable "up" matrix, zero at start | (d_out, r) |
| $r$ | the rank: the width of the bottleneck | usually 4 to 64 |
| $\alpha$ | alpha, a scale knob; α/r keeps update size steady when you change r | often r or 2r |

**In words:** the output is what the frozen layer produces plus a scaled
correction that passes through a narrow r-number bottleneck.

**On the worked example:** W = I, A = (0, 1), B = (1, 0)ᵀ, α/r = 1,
x = (1, 2): xWᵀ = (1, 2), xAᵀ = 2, 2·Bᵀ = (2, 0), y = (3, 2).

**Parameter savings for one 4096 × 4096 layer:**

| Method | Trainable parameters | Share of full |
|---|---|---|
| full fine-tune | 16,777,216 | 100% |
| LoRA r = 64 | 524,288 | 3.1% |
| LoRA r = 16 | 131,072 | 0.78% |
| LoRA r = 8 | 65,536 | 0.39% |

![Training a LoRA adapter at different ranks](figures/primer.ml.training_stages.lora_ranks.svg)

**Reading it:** the task needs a rank-2 change to a frozen 16 × 16 layer.
The y-axis (log scale) is the training error; the x-axis is the training
step. A rank-1 adapter plateaus: its bottleneck is too narrow to express
the change. Ranks 2 and 4 drive the error to essentially zero. This is the
LoRA bet: the *change* a fine-tune needs is low-rank even though the
weights themselves are not.

**QLoRA** goes further: it stores the frozen base weights in 4 bits (a
format called NF4) and trains LoRA adapters in 16-bit on top, which lets a
65-billion-parameter model be fine-tuned on a single 48 GB GPU.

**Why it matters in practice.** LoRA adapters are megabytes, not
gigabytes. You can keep one per customer or task and hot-swap them on one
shared base model, and training fits on far smaller hardware.

## 5. Which adaptation should you use?

**Everyday picture.** If a new employee doesn't know your product catalogue,
you hand them the catalogue (retrieval); you don't send them back to
school. If they know the facts but write emails in the wrong tone, you first
give clearer instructions, then coach them, and only for a whole new
profession do you retrain from scratch.

| Approach | Changes weights? | Use when |
|---|---|---|
| Prompting / few-shot | No | behaviour or format change |
| RAG | No | knowledge that changes or must be cited |
| LoRA / QLoRA | small adapters | style, domain vocabulary, consistent output format |
| Full fine-tune | yes, all | rarely: major domain shift with lots of data |

```mermaid
flowchart TD
  Q1{Missing knowledge,<br/>or facts that change?} -->|Yes| R[Use RAG]
  Q1 -->|No| Q2{Wrong behaviour,<br/>format or tone?}
  Q2 -->|No| N[No change needed]
  Q2 -->|Yes| P[Improve the prompt<br/>add examples]
  P -->|Still inconsistent<br/>or too costly| L[LoRA fine-tune]
  L -->|Major domain shift,<br/>lots of data| F[Full fine-tune]
```

**Reading it:** start at the top. Knowledge problems exit immediately to
retrieval. Behaviour problems climb a ladder of cost and only go as far as
they must: prompt first, a LoRA adapter if the prompt can't make the
behaviour consistent (or the prompt is too long and costly to send every
time), and a full fine-tune only for a genuine domain shift with lots of
data.

**The key line: fine-tuning teaches behaviour; RAG supplies knowledge.**
Most production systems end up as retrieval plus a well-built prompt.

## 6. Distillation: the apprentice learns how the master hesitates

**Everyday picture.** A master chef tastes a sauce and says "mostly thyme, a
bit of rosemary, definitely not mint". An apprentice who only hears "thyme"
learns less than one who hears the whole judgement. Distillation trains a
small, cheap *student* model to match a big *teacher's* full probability
spread, not just its top answer.

**Tiny worked example.** The teacher's raw scores (logits) for three
answers are 2, 1, 0. Softmax at temperature T = 1 gives 0.665, 0.245, 0.090.
Dividing the logits by T = 2 first gives softmax(1, 0.5, 0) = **0.506,
0.307, 0.186**: flatter, so the student clearly sees that answer 2 is a much
better runner-up than answer 3. That runner-up information is sometimes
called "dark knowledge".

```mermaid
flowchart LR
  X[Same input] --> T[Big teacher model]
  X --> S[Small student model]
  T --> TS[softmax of logits / T<br/>soft targets]
  S --> SS[softmax of logits / T]
  TS --> KL[KL divergence<br/>how different are they?]
  SS --> KL
  KL -->|gradient| S
```

**Reading it:** both models see the same input. Each one's scores are
softened by the same temperature, and the **KL divergence** measures how
far the student's spread is from the teacher's. Only the student learns; the
teacher is fixed. Typically this runs over large volumes of the exact
traffic your product sees, so the student becomes an expert at *your* task.

$$
p_i^{(T)} = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}} \qquad
\mathrm{KL}(p \,\|\, q) = \sum_i p_i \ln\frac{p_i}{q_i} \qquad
\mathcal{L} = \alpha\,T^2\,\mathrm{KL}\big(p^{(T)} \,\|\, q^{(T)}\big) + (1-\alpha)\,\mathrm{CE}
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| $z_i$ | the teacher's logit (raw score) for answer i | any real |
| $T$ | temperature: divide scores by T before softmax; T > 1 flattens | > 0 |
| $p^{(T)}, q^{(T)}$ | teacher and student probabilities at temperature T | each sums to 1 |
| $\mathrm{KL}(p\|q)$ | Kullback-Leibler divergence: the extra surprise from believing q when the truth is p. Zero only when they match | ≥ 0 |
| $\ln$ | natural logarithm | |
| $T^2$ | rescales the gradient, which softening shrinks by 1/T² | |
| $\alpha$ | mix between matching the teacher and matching the true label | 0 … 1 |
| CE | ordinary cross-entropy on the true label | ≥ 0 |

**In words:** the student is penalised by how far its softened spread is
from the teacher's, scaled by T², optionally mixed with the usual penalty on
the true answer.

**On the worked example:** with teacher (0.5, 0.5) and student (0.9, 0.1),
KL = 0.5·ln(0.5/0.9) + 0.5·ln(0.5/0.1) = −0.294 + 0.805 = **0.511**. When
the student matches the teacher exactly, KL = 0.

![Temperature softens the teacher's distribution](figures/primer.ml.training_stages.distill_temperature.svg)

**Reading it:** the same three teacher logits (2, 1, 0) shown at three
temperatures. At T = 1 (left group) the top answer dominates. At T = 2 and
T = 5 the bars even out and the *ranking* of the wrong answers becomes
visible to the student. Too high a temperature and everything flattens to
a uniform guess, so T is tuned (2 to 5 is common).

**Why it matters in practice.** Distillation is often the biggest cost
lever in production: a small student trained on a big model's outputs for
one narrow task can be many times cheaper and faster with little quality
loss on that task.

## In 20 seconds
- Pretraining predicts the next token over huge text and gives knowledge;
  SFT teaches the assistant format; preference tuning shapes tone,
  helpfulness and safety.
- SFT is next-token loss with the prompt masked out.
- RLHF trains a reward model on pairwise preferences, then optimises the LM
  against it; DPO gets the same effect directly from preference pairs.
- LoRA trains a tiny low-rank correction B·A beside frozen weights: under 1%
  of the parameters, mergeable for zero added latency.
- Fine-tuning teaches behaviour; RAG supplies knowledge. Start with
  prompting and RAG.
- Distillation trains a small student on a big teacher's soft targets; it's
  a major cost lever.

## Self-test questions

**Q: Why does a base model ramble instead of answering?**
A: Pretraining only teaches it to continue text. Answering questions in a
helpful format is learned later, in SFT and preference tuning.

**Q: What is the one code difference between pretraining loss and SFT loss?**
A: A mask. SFT computes the same next-token cross-entropy but averages it
only over the reply tokens, so the prompt is read but not trained on.

**Q: In RLHF, what does the reward model learn, and from what?**
A: A score for responses such that sigmoid(score gap) predicts which of two
responses a labeler preferred. It learns from pairwise comparisons, which
are far easier for people to give than perfect answers.

**Q: How does DPO avoid a reward model?**
A: It treats the log-probability ratio between the policy and a frozen
reference as an implicit reward, and applies the same pairwise loss
directly to the policy. One model, one supervised-style training loop.

**Q: What does β control in DPO?**
A: How strongly preferences push the policy relative to staying close to
the reference model. Larger β means sharper preference-following; smaller
means more conservative updates.

**Q: Why is B initialised to zero in LoRA?**
A: So B·A = 0 and the adapted model starts exactly equal to the pretrained
model. Training then learns only the change.

**Q: How many parameters does LoRA rank 8 train on a 4096 × 4096 layer?**
A: 2 × 4096 × 8 = 65,536, about 0.39% of the 16.8 million in the full
matrix.

**Q: Does LoRA slow down inference?**
A: Not if you merge: add B·A into W once and serve the result. Unmerged
adapters cost two thin extra matmuls, which is what lets you hot-swap
adapters on one base model.

**Q: A client wants the model to know their product catalogue, which
changes weekly. Fine-tune or RAG?**
A: RAG. The knowledge changes and answers should cite sources; fine-tuning
would be stale within a week and can't cite. Fine-tune only for behaviour
the prompt can't make consistent.

**Q: What does temperature do in distillation?**
A: It softens both distributions so the student learns the teacher's
relative preferences among wrong answers, not just its top pick; the T²
factor keeps gradient sizes comparable.

## The papers behind this lesson

- Brown et al., *Language Models are Few-Shot Learners* (GPT-3, 2020):
  https://arxiv.org/abs/2005.14165. Showed that scaling next-token
  pretraining produces a model that can follow instructions and examples
  given only in the prompt. [annotated companion](../../papers/gpt-3.html)
- Kaplan et al., *Scaling Laws for Neural Language Models* (2020):
  https://arxiv.org/abs/2001.08361, with Hoffmann et al., *Training
  Compute-Optimal Large Language Models* (2022):
  https://arxiv.org/abs/2203.15556. Measured how pretraining loss falls
  predictably with model size, data and compute, and how to balance them.
  [annotated companion](../../papers/scaling-laws.html)
- Ouyang et al., *Training language models to follow instructions with
  human feedback* (InstructGPT, 2022): https://arxiv.org/abs/2203.02155.
  Established the SFT, reward model and RLHF recipe behind chat assistants.
  [annotated companion](../../papers/instructgpt.html)
- Rafailov et al., *Direct Preference Optimization: Your Language Model is
  Secretly a Reward Model* (2023): https://arxiv.org/abs/2305.18290. Showed
  preference tuning can skip the reward model and RL loop entirely.
  [annotated companion](../../papers/dpo.html)
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021):
  https://arxiv.org/abs/2106.09685, with Dettmers et al., *QLoRA* (2023):
  https://arxiv.org/abs/2305.14314. Fine-tuned huge models by training small
  low-rank adapters beside frozen (and, in QLoRA, 4-bit) weights.
  [annotated companion](../../papers/lora.html)
- Hinton, Vinyals & Dean, *Distilling the Knowledge in a Neural Network*
  (2015): https://arxiv.org/abs/1503.02531. Trained small students on a
  large teacher's temperature-softened outputs.
  [annotated companion](../../papers/distillation.html)
- Bai et al., *Constitutional AI: Harmlessness from AI Feedback* (2022):
  https://arxiv.org/abs/2212.08073. Replaced much human preference labelling
  with a model judging responses against written principles.

## Further reading
- Hugging Face PEFT documentation: https://huggingface.co/docs/peft/index
- Hugging Face TRL documentation (SFT, DPO, reward modelling): https://huggingface.co/docs/trl/index
"""

from __future__ import annotations

import numpy as np

from primer._show import banner, say, table, takeaway

# ---------------------------------------------------------------------------
# 1. Pretraining and SFT share one loss; SFT just masks the prompt
# ---------------------------------------------------------------------------


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """log(softmax(x)) computed stably as x - logsumexp(x)."""
    m = logits.max(axis=axis, keepdims=True)
    return logits - m - np.log(np.exp(logits - m).sum(axis=axis, keepdims=True))


def per_position_losses(logits: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """-log p(next token) at every position.

    Shapes: `logits` is (n, vocab), where row i is the model's prediction for
    token i+1; `tokens` is (n,). There are n-1 predictions with a known
    answer (the last row predicts a token we don't have), so the result is
    (n-1,).
    """
    logp = log_softmax(logits[:-1])  # (n-1, vocab)
    targets = tokens[1:]  # (n-1,): shift left by one, the "next token"
    return -logp[np.arange(len(targets)), targets]


def next_token_loss(logits: np.ndarray, tokens: np.ndarray) -> float:
    """The pretraining objective: mean cross-entropy over every next-token prediction."""
    return float(per_position_losses(logits, tokens).mean())


def response_mask(n_tokens: int, prompt_len: int) -> np.ndarray:
    """Which of the n-1 predictions SFT trains on: those whose target is a reply token.

    Prediction i targets token i+1, which belongs to the reply when
    i + 1 >= prompt_len.
    """
    return np.arange(1, n_tokens) >= prompt_len


def sft_loss(logits: np.ndarray, tokens: np.ndarray, prompt_len: int) -> float:
    """Supervised fine-tuning loss: next-token cross-entropy on the reply only.

    The prompt is still fed in (the model must *read* it to answer), but its
    tokens contribute no loss, so the model learns to produce answers rather
    than to imitate users.
    """
    losses = per_position_losses(logits, tokens)
    mask = response_mask(len(tokens), prompt_len)
    return float(losses[mask].mean())


# ---------------------------------------------------------------------------
# 2. Preference tuning: reward models (RLHF) and DPO
# ---------------------------------------------------------------------------


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def preference_probability(reward_chosen: float, reward_rejected: float) -> float:
    """Bradley-Terry model: P(chosen is preferred) = sigmoid(r_chosen - r_rejected).

    Only the *difference* in reward matters, so a reward model's absolute
    scores are meaningless; only their ordering and gaps carry information.
    """
    return float(sigmoid(reward_chosen - reward_rejected))


def reward_model_loss(reward_chosen: float, reward_rejected: float) -> float:
    """-log P(chosen preferred): the loss a reward model minimizes on each labeled pair."""
    return float(-np.log(preference_probability(reward_chosen, reward_rejected)))


def dpo_margin(
    policy_chosen: float, policy_rejected: float, ref_chosen: float, ref_rejected: float, beta: float
) -> float:
    """beta × [(log π(y_w) - log π_ref(y_w)) - (log π(y_l) - log π_ref(y_l))].

    Each bracket is the policy's *implicit reward* for an answer: how much
    more likely the policy makes it than the frozen reference model did.
    Inputs are total log-probabilities of whole answers (sums over tokens).
    """
    return beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta: float = 0.1) -> float:
    """DPO loss for one preference pair: -log sigmoid(margin).

    Same shape as the reward-model loss, but the "reward" is read straight
    off the policy's own log-probabilities, so no separate reward model and
    no reinforcement-learning loop are needed.
    """
    z = dpo_margin(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
    return float(-np.log(sigmoid(z)))


def dpo_update_strength(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta: float = 0.1) -> float:
    """How hard one gradient step pushes on this pair: beta × (1 - sigmoid(margin)).

    d loss / d margin = -(1 - sigmoid(margin)). Pairs the policy already
    ranks correctly (large margin) get tiny updates; pairs it ranks the
    wrong way get strong ones. The push itself raises log π(chosen) and
    lowers log π(rejected).
    """
    z = dpo_margin(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
    return float(beta * (1 - sigmoid(z)))


# Three candidate answers to one prompt, and which one people preferred in each pair.
TOY_ANSWERS = ["helpful and correct", "rude", "correct but rambling"]
TOY_PREFERENCES = [(0, 1), (0, 2), (2, 1)]  # (chosen, rejected)


def train_toy_dpo(steps: int = 200, lr: float = 1.0, beta: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Train a one-prompt 'policy' (logits over the three TOY_ANSWERS) with DPO.

    Returns (probabilities before, probabilities after). With logits θ,
    log π(y) = θ_y - logsumexp(θ); the logsumexp cancels in every
    chosen-minus-rejected difference, so d margin / d θ = beta·(e_chosen - e_rejected).
    """
    theta = np.zeros(len(TOY_ANSWERS))  # start uniform
    ref_logp = log_softmax(theta.copy())  # the frozen reference = the starting policy
    before = np.exp(ref_logp)
    for _ in range(steps):
        logp = log_softmax(theta)
        grad = np.zeros_like(theta)
        for c, r in TOY_PREFERENCES:
            strength = dpo_update_strength(logp[c], logp[r], ref_logp[c], ref_logp[r], beta)
            # Gradient *descent* on the loss = raise chosen, lower rejected, scaled by strength.
            grad[c] -= strength
            grad[r] += strength
        theta -= lr * grad / len(TOY_PREFERENCES)
    return before, np.exp(log_softmax(theta))


# ---------------------------------------------------------------------------
# 3. LoRA: fine-tune a tiny low-rank correction instead of the whole matrix
# ---------------------------------------------------------------------------


class LoRALinear:
    """A frozen linear layer W plus a trainable low-rank correction B·A.

        y = x Wᵀ + (alpha / rank) · x Aᵀ Bᵀ

    Shapes: W is (d_out, d_in) and never changes; A is (rank, d_in); B is
    (d_out, rank). A starts random and B starts at zero, so B·A = 0 and the
    layer initially behaves exactly like the pretrained one. `alpha / rank`
    keeps the update's size roughly independent of the rank you choose.
    """

    def __init__(self, W: np.ndarray, rank: int, alpha: float | None = None, seed: int = 0):
        self.W = W  # frozen: only A and B receive gradients
        d_out, d_in = W.shape
        rng = np.random.default_rng(seed)
        self.A = rng.normal(0, 1 / np.sqrt(d_in), (rank, d_in))
        self.B = np.zeros((d_out, rank))
        self.scale = (alpha if alpha is not None else rank) / rank

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Two thin matmuls (d_in -> rank -> d_out) instead of forming B·A.
        return x @ self.W.T + self.scale * (x @ self.A.T) @ self.B.T

    def merged_weight(self) -> np.ndarray:
        """W + scale·B·A: fold the adapter in, so serving costs exactly what the base model did."""
        return self.W + self.scale * self.B @ self.A


def lora_trainable_params(d_out: int, d_in: int, rank: int) -> int:
    """A has rank·d_in entries and B has d_out·rank."""
    return rank * d_in + d_out * rank


def full_trainable_params(d_out: int, d_in: int) -> int:
    return d_out * d_in


def train_toy_lora(rank: int = 2, steps: int = 300, lr: float = 0.03, d: int = 16, seed: int = 0) -> dict:
    """Teach a frozen layer a new behaviour that is a rank-2 change of its weights.

    The "pretrained" layer is W0. The new task wants W0 + U·V with U·V of
    rank 2, which is the LoRA hypothesis in miniature: the change a
    fine-tune needs is low-rank even when the weights themselves are not.
    Trains A and B by plain gradient descent on mean squared error and
    returns the loss curve.
    """
    rng = np.random.default_rng(seed)
    W0 = rng.standard_normal((d, d))
    W_task = W0 + rng.standard_normal((d, 2)) @ rng.standard_normal((2, d)) / np.sqrt(d)
    x = rng.standard_normal((256, d))
    target = x @ W_task.T

    layer = LoRALinear(W0.copy(), rank=rank, seed=seed)
    losses = []
    for _ in range(steps):
        err = layer(x) - target  # (batch, d_out)
        losses.append(float(np.mean(err**2)))
        # Backprop through y = x Wᵀ + s·(x Aᵀ) Bᵀ with loss = mean(err²):
        dy = 2 * err / err.size  # d loss / d y
        h = x @ layer.A.T  # (batch, rank): the bottleneck activations
        grad_B = layer.scale * dy.T @ h  # (d_out, rank)
        grad_A = layer.scale * (dy @ layer.B).T @ x  # (rank, d_in)
        layer.A -= lr * grad_A * err.shape[1]  # rescale: the mean over outputs makes raw grads tiny
        layer.B -= lr * grad_B * err.shape[1]
    return dict(
        losses=losses,
        initial_loss=losses[0],
        final_loss=float(np.mean((layer(x) - target) ** 2)),
        frozen_weights_unchanged=bool(np.array_equal(layer.W, W0)),
        layer=layer,
    )


# ---------------------------------------------------------------------------
# 4. Choosing how to adapt a model
# ---------------------------------------------------------------------------


def choose_adaptation(
    needs_knowledge: bool,
    wrong_behavior: bool,
    prompt_fixes_it: bool = True,
    major_shift_with_lots_of_data: bool = False,
) -> str:
    """Walk the decision flowchart, cheapest option first.

    Knowledge problems go to retrieval; behaviour problems climb
    prompting -> LoRA -> full fine-tune only as far as they must.
    """
    if needs_knowledge:
        return "RAG"
    if not wrong_behavior:
        return "no change"
    if prompt_fixes_it:
        return "prompting"
    if major_shift_with_lots_of_data:
        return "full fine-tune"
    return "LoRA"


# ---------------------------------------------------------------------------
# 5. Distillation: a small student imitates a big teacher
# ---------------------------------------------------------------------------


def soft_targets(logits: np.ndarray, temperature: float) -> np.ndarray:
    """softmax(logits / T). T > 1 flattens the distribution, exposing how the
    teacher ranks the *wrong* answers ("dark knowledge")."""
    return np.exp(log_softmax(logits / temperature))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) = Σ p·ln(p/q): extra surprise from using q when the truth is p.

    Zero exactly when p == q, positive otherwise. Terms with p = 0 contribute
    nothing (0·ln 0 is taken as 0).
    """
    nz = p > 0
    return float(np.sum(p[nz] * np.log(p[nz] / q[nz])))


def distillation_loss(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    temperature: float = 2.0,
    label: int | None = None,
    alpha: float = 1.0,
) -> float:
    """alpha·T²·KL(teacher_T || student_T) + (1 - alpha)·cross-entropy(label).

    The T² factor keeps the soft-target gradients the same size as the
    hard-label ones when you change T (softening shrinks gradients by 1/T²).
    """
    p = soft_targets(teacher_logits, temperature)
    q = soft_targets(student_logits, temperature)
    loss = alpha * temperature**2 * kl_divergence(p, q)
    if label is not None and alpha < 1:
        loss += (1 - alpha) * float(-log_softmax(student_logits)[label])
    return loss


# ---------------------------------------------------------------------------
# 6. Figures (rendered into docs/figures by `make figures`)
# ---------------------------------------------------------------------------

# The worked example used in the lesson text and the tests: a 3-token prompt
# and a 2-token reply over a 4-token vocabulary.
EXAMPLE_TOKENS = np.array([0, 1, 2, 3, 1])
EXAMPLE_PROMPT_LEN = 3


def _example_logits() -> np.ndarray:
    def giving(p: float, correct: int) -> np.ndarray:
        probs = np.full(4, (1 - p) / 3)
        probs[correct] = p
        return np.log(probs)

    # Uniform guesses on the prompt, probability 0.5 on each reply token.
    return np.stack([np.zeros(4), np.zeros(4), giving(0.5, 3), giving(0.5, 1), np.zeros(4)])


def figures() -> dict:
    """Data figures for this lesson, keyed by the name used in the docstring."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = {}

    # SFT mask on the worked example.
    losses = per_position_losses(_example_logits(), EXAMPLE_TOKENS)
    mask = response_mask(len(EXAMPLE_TOKENS), EXAMPLE_PROMPT_LEN)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    labels = [f"t{i + 1}→t{i + 2}" for i in range(len(losses))]
    ax.bar(labels, losses, color=["#1f77b4" if m else "#bbbbbb" for m in mask])
    ax.axhline(losses.mean(), ls="--", color="black", label=f"pretraining loss {losses.mean():.3f} (all positions)")
    ax.axhline(losses[mask].mean(), ls="--", color="#1f77b4", label=f"SFT loss {losses[mask].mean():.3f} (reply only)")
    ax.set(ylabel="penalty  −ln p(true next token)", xlabel="prediction (grey = prompt, masked; blue = reply)",
           title="SFT grades only the reply", ylim=(0, 1.8))
    ax.legend(loc="upper right", fontsize=8)
    figs["sft_mask"] = fig

    # DPO update strength vs margin.
    z = np.linspace(-6, 6, 200)
    beta = 0.1
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(z, beta * (1 - sigmoid(z)), lw=2)
    ax.axvline(0, color="grey", lw=0.8)
    ax.plot([0.2], [beta * (1 - sigmoid(0.2))], "o", color="C3", label="worked example: margin 0.2 → 0.045")
    ax.set(xlabel="DPO margin (how strongly the policy already prefers the chosen answer)",
           ylabel="update strength  β(1 − σ(margin))", title="DPO pushes hardest on pairs it gets wrong (β = 0.1)")
    ax.legend()
    figs["dpo_strength"] = fig

    # DPO toy training trajectory.
    steps = np.arange(0, 201, 5)
    traj = np.array([train_toy_dpo(steps=int(s))[1] for s in steps])
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for i, name in enumerate(TOY_ANSWERS):
        ax.plot(steps, traj[:, i], lw=2, label=name)
    ax.set(xlabel="training step", ylabel="policy probability", title="DPO on three candidate answers", ylim=(0, 1))
    ax.legend()
    figs["dpo_training"] = fig

    # LoRA rank sweep on a rank-2 task.
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for r in (1, 2, 4):
        ax.semilogy(train_toy_lora(rank=r, steps=300)["losses"], lw=2, label=f"rank {r}")
    ax.set(xlabel="training step", ylabel="mean squared error (log scale)",
           title="The task needs a rank-2 change: rank 1 can't express it")
    ax.set_ylim(1e-8, 10)
    ax.legend()
    figs["lora_ranks"] = fig

    # Distillation temperature.
    logits = np.array([2.0, 1.0, 0.0])
    temps = [1.0, 2.0, 5.0]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    x = np.arange(len(temps))
    for i in range(3):
        ax.bar(x + (i - 1) * 0.25, [soft_targets(logits, t)[i] for t in temps], 0.25, label=f"answer {i + 1} (logit {logits[i]:.0f})")
    ax.set_xticks(x, [f"T = {t:g}" for t in temps])
    ax.set(ylabel="teacher probability", title="Higher temperature reveals the runner-up ranking", ylim=(0, 0.75))
    ax.legend()
    figs["distill_temperature"] = fig

    for f in figs.values():
        f.tight_layout()
    return figs


# ---------------------------------------------------------------------------
# 7. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. Pretraining vs. SFT: same loss, different mask")
    logits = _example_logits()
    losses = per_position_losses(logits, EXAMPLE_TOKENS)
    mask = response_mask(len(EXAMPLE_TOKENS), EXAMPLE_PROMPT_LEN)
    table(
        ["prediction", "part", "p(true next)", "penalty −ln p"],
        [(f"t{i + 1} → t{i + 2}", "reply" if m else "prompt", float(np.exp(-l)), l) for i, (l, m) in enumerate(zip(losses, mask))],
        floatfmt=".3f",
    )
    say(
        f"""
        Pretraining averages every penalty: {next_token_loss(logits, EXAMPLE_TOKENS):.3f}. SFT averages only
        the reply: {sft_loss(logits, EXAMPLE_TOKENS, EXAMPLE_PROMPT_LEN):.3f}. The model still reads the
        prompt; it just isn't graded on predicting it.
        """
    )
    takeaway("SFT is next-token prediction with the prompt masked out of the loss.")

    banner("2. Reward model: the Bradley-Terry taste test")
    table(
        ["reward chosen", "reward rejected", "P(chosen wins)", "loss"],
        [(rc, rr, preference_probability(rc, rr), reward_model_loss(rc, rr)) for rc, rr in [(1, 1), (3, 1), (1, 3)]],
        floatfmt=".3f",
    )
    say("Only the gap matters. Equal scores mean a coin flip (loss ln 2 = 0.693); a gap of 2 means 88%.")

    banner("3. DPO: the worked example")
    args = (-10.0, -12.0, -11.0, -11.0)
    say(
        f"""
        Policy log-probs: chosen −10, rejected −12. Reference: both −11. β = 0.1.
        Margin = 0.1 × ((−10 + 11) − (−12 + 11)) = {dpo_margin(*args, beta=0.1):.1f}.
        Loss = −ln σ(0.2) = {dpo_loss(*args):.3f} (it was ln 2 = 0.693 before any learning).
        Update strength = β(1 − σ(0.2)) = {dpo_update_strength(*args):.3f}.
        """
    )
    before, after = train_toy_dpo()
    table(["answer", "before", "after 200 DPO steps"], [(a, b, c) for a, b, c in zip(TOY_ANSWERS, before, after)], floatfmt=".3f")
    takeaway("DPO turns pairwise preferences into a supervised loss on the model itself: no reward model, no RL loop.")

    banner("4. LoRA: trainable parameters for one 4096 × 4096 layer")
    full = full_trainable_params(4096, 4096)
    rows = [("full fine-tune", f"{full:,}", "100%")]
    rows += [(f"LoRA r = {r}", f"{lora_trainable_params(4096, 4096, r):,}", f"{lora_trainable_params(4096, 4096, r) / full:.2%}") for r in (64, 16, 8)]
    table(["method", "trainable parameters", "share"], rows)
    for r in (1, 2, 4):
        run = train_toy_lora(rank=r)
        print(f"rank {r}: error {run['initial_loss']:.3f} → {run['final_loss']:.2e}   (frozen W unchanged: {run['frozen_weights_unchanged']})")
    print()
    say("The task needed a rank-2 change. Rank 1 plateaus; ranks 2 and 4 learn it while W never moves.")
    takeaway("LoRA learns a low-rank correction beside frozen weights, then merges it in for zero added latency.")

    banner("5. Which adaptation?")
    cases = [
        ("product catalogue changes weekly", dict(needs_knowledge=True, wrong_behavior=False)),
        ("answers too long; a clear instruction fixes it", dict(needs_knowledge=False, wrong_behavior=True, prompt_fixes_it=True)),
        ("house style inconsistent even with examples", dict(needs_knowledge=False, wrong_behavior=True, prompt_fixes_it=False)),
        ("new language + millions of examples", dict(needs_knowledge=False, wrong_behavior=True, prompt_fixes_it=False, major_shift_with_lots_of_data=True)),
    ]
    table(["situation", "recommendation"], [(s, choose_adaptation(**kw)) for s, kw in cases])
    takeaway("Fine-tuning teaches behaviour; RAG supplies knowledge.")

    banner("6. Distillation: soft targets")
    z = np.array([2.0, 1.0, 0.0])
    table(["temperature", "p(answer 1)", "p(answer 2)", "p(answer 3)"], [(t, *soft_targets(z, t)) for t in (1.0, 2.0, 5.0)], floatfmt=".3f")
    say(f"KL((0.5, 0.5) || (0.9, 0.1)) = {kl_divergence(np.array([0.5, 0.5]), np.array([0.9, 0.1])):.3f} nats; it is 0 when the student matches the teacher.")
    takeaway("A small student trained on a big teacher's soft targets is often the biggest cost win in production.")


if __name__ == "__main__":
    demo()
