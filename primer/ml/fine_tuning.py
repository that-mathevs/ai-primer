r"""
# Fine-tuning in practice: preparing data, forgetting old skills, merging models

Run: `python -m primer.ml.fine_tuning`

New to the notation? `primer.notation` explains every symbol used here
(Σ, ‖x‖, λ, subscripts, and so on) from zero.

This lesson builds on `primer.ml.training_stages`, which explains *what*
supervised fine-tuning (SFT), preference tuning and LoRA are. Here we deal
with what happens when you actually do it: deciding whether to fine-tune at
all, preparing the data, the old skills the model loses along the way, the
small dataset it memorises, and how two fine-tuned models can be merged into
one by plain arithmetic on their weights.

## The idea: an apprentice who already knows a lot

Picture a skilled cook who joins your restaurant. They already know how to
cook (that is the pretrained model). You want them to cook *your* menu, *your*
way. You can hand them a note with every order (a prompt), give them the
recipe binder to look things up in (retrieval, RAG), or send them on a course
about your kitchen (fine-tuning). The course is the only option that changes
the cook, and it comes with the risks every teacher knows: the lessons can
be badly written, the cook can cram the practice exam instead of learning,
and a month of drilling one cuisine can make them rusty at everything else.

Everything below is one of those risks, measured on a model small enough to
train in a fraction of a second.

## 1. Should you fine-tune at all?

**Everyday picture.** Giving the cook a note with every order costs a
little each time. The course costs a lot once, and it has to be repeated
whenever you hire a new cook (switch to a newer base model). The course pays
off only when the notes would otherwise be long, sent millions of times, or
simply not enough to make the cook consistent.

**Tiny worked example.** A support bot sends a 3,000-token prompt full of
instructions and examples on every request. A fine-tuned model has learned
that behaviour and needs only 300 tokens of prompt. Take these illustrative
prices (real ones vary by provider and change often): $2 per million input
tokens for the general model, $4 per million for the tuned one, because
hosting a custom model usually costs more per token.

| | Tokens per request | Price per million | Cost per request |
|---|---|---|---|
| prompted | 3,000 | $2 | 3,000 × 2 / 1,000,000 = **$0.0060** |
| fine-tuned | 300 | $4 | 300 × 4 / 1,000,000 = **$0.0012** |

Each request saves $0.0048. Writing and checking 1,000 training examples
plus the training run costs, say, $600 once. After 600 / 0.0048 = **125,000
requests** the course has paid for itself: 25 days at 5,000 requests a day.

```mermaid
flowchart TD
  E[Build the eval set first] --> P[Best prompt you can write]
  P --> M1{Good enough<br/>on the eval?}
  M1 -->|yes| SHIP[Ship the prompt]
  M1 -->|"no: missing facts"| R[Add retrieval, RAG]
  R --> M2{Good enough<br/>on the eval?}
  M2 -->|yes| SHIP
  M2 -->|"no: wrong behaviour, format or style"| FT[Fine-tune, often with LoRA]
  FT --> M3{Beats the prompt<br/>on the SAME eval?}
  M3 -->|yes, and pays off| SHIPFT[Ship the fine-tune]
  M3 -->|no| P
```

**Reading it:** start at the top, and notice that the first box is not a
model at all: it is the evaluation set, the fixed exam every option is
scored on. Each rung is cheaper and faster to change than the one below it,
so you climb down only when the eval proves the rung above falls short.
Missing knowledge goes to retrieval, because fine-tuning is a poor way to
add facts; wrong behaviour that no prompt can pin down goes to fine-tuning.
The last diamond matters most: a fine-tune ships only if it beats the prompt
on the same eval. `primer.ml.training_stages` has the same decision as a
flowchart of approaches; this one adds the measurement at every step.

$$
N^\star = \frac{C_{\text{once}}}{c_{\text{prompt}} - c_{\text{tuned}}},
\qquad c = \frac{t \cdot p}{10^6}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $N^\star$ | "N-star": the break-even number of requests | 125,000 |
| $C_{\text{once}}$ | one-off cost: writing and checking data, the training run | $600 |
| $c$ | cost of one request | |
| $c_{\text{prompt}}, c_{\text{tuned}}$ | cost of one request with the long prompt, and with the fine-tuned model | $0.0060, $0.0012 |
| $t$ | input tokens sent per request | 3,000 and 300 |
| $p$ | price in dollars per million input tokens | $2 and $4 |
| $10^6$ | one million: prices are quoted per million tokens | |

**In words:** "divide what the course costs once by what it saves on each
request; that is how many requests it takes to earn the cost back."

**With the numbers:** $c_{\text{prompt}}$ = 3,000 × 2 / 10⁶ = 0.006,
$c_{\text{tuned}}$ = 300 × 4 / 10⁶ = 0.0012, so $N^\star$ = 600 / 0.0048 =
125,000.

**In Python:**

```python
# c = t·p / 1,000,000 for each option
c_prompt = 3000 * 2.0 / 1_000_000
c_tuned = 300 * 4.0 / 1_000_000
c_prompt, c_tuned  # → (0.006, 0.0012)
C_once = 600
# N* = C_once / (c_prompt − c_tuned)
N_star = C_once / (c_prompt - c_tuned)
round(N_star)  # → 125000
# days to break even at 5,000 requests a day
round(N_star / 5000, 1)  # → 25.0
```

![The prompted line starts at zero and climbs steeply; the fine-tuned line starts at 600 dollars and climbs slowly; they cross at 125,000 requests](figures/primer.ml.fine_tuning.break_even.svg)

**Reading it:** the x-axis counts requests served and the y-axis is the
total money spent so far. The prompted line starts at zero but climbs
steeply, because every request pays for 3,000 tokens. The fine-tuned line
starts at $600 (the one-off cost) and climbs slowly. Left of the dashed
line, prompting is cheaper; right of it, the fine-tune is. If the tuned
model cost as much per request as the prompt, the two lines would never
cross, and the only reason left to fine-tune would be quality.

**In code:** `per_request_cost` prices one request and
`break_even_requests` returns $N^\star$, or infinity when the tuned model
saves nothing per request.

**Why it matters in practice.** Three costs hide outside this formula.
Every new base model means repeating the fine-tune, so $C_{\text{once}}$
recurs. Facts learned by fine-tuning go stale and are hard to update, which
is why knowledge belongs in retrieval (`primer.agents.rag`). And the
engineering time to build an eval is spent whichever way you go, which is
why it comes first.

## 2. Preparing the data

**Everyday picture.** Before the course, someone writes the course book.
Every recipe must be written in the same layout, no recipe may appear five
times, some recipes are locked in a drawer for the final exam *before* the
cook ever sees the book, and the recipes must actually be right. Most
fine-tuning failures are course-book failures.

```mermaid
flowchart LR
  RAW[Raw examples<br/>logs, experts, drafts] --> FMT[Format as chat<br/>messages]
  FMT --> VAL[Validate<br/>roles, empty turns]
  VAL --> DD[Deduplicate<br/>near-copies]
  DD --> SPLIT[Set aside the<br/>held-out set, frozen]
  SPLIT --> LEAK[Drop training examples<br/>that copy held-out ones]
  LEAK --> AUDIT[Audit a sample<br/>of the labels]
  AUDIT --> TRAIN[Training set]
  SPLIT --> EVAL[Held-out set]
```

**Reading it:** raw examples enter on the left and pass through five
filters. The branch at "Set aside the held-out set" is the important one:
the held-out examples leave the pipeline *before* anything is trained or
tuned, and the step after it removes any training example that nearly
copies one of them. The subsections below build each box.

### 2a. Formatting chat examples

**Everyday picture.** A play script: every line starts with who speaks it.
The model learns the play by reading scripts, and it is graded only on the
lines of the character it will play, the assistant.

**Tiny worked example.** One example as a list of role-tagged messages:

| Turn | Role | Content | Trained on? |
|---|---|---|---|
| 1 | system | Be brief. | no |
| 2 | user | Capital of France? | no |
| 3 | assistant | Paris. | **yes** |

Rendered into text, it becomes
`<|system|>Be brief.<|end|><|user|>Capital of France?<|end|><|assistant|>Paris.<|end|>`,
and only the last segment counts towards the loss.

```mermaid
flowchart LR
  S["system: Be brief."] --> U["user: Capital of France?"] --> A["assistant: Paris."]
  S -.->|"context only, no loss"| L[Loss]
  U -.->|"context only, no loss"| L
  A ==>|"every token scored"| L
```

**Reading it:** the solid arrows are the order the model reads the turns.
The dotted arrows carry no training signal: the system and user turns are
context the model conditions on. Only the thick arrow feeds the loss, so the
model learns to *answer*, not to write questions. This is the loss mask from
`primer.ml.training_stages`, section 2.

**The rules that matter.** Use the exact chat template the base model was
trained with; the role markers above are illustrative, and each model family
has its own. Train with the same system prompt you will deploy with. Reject
examples whose last turn is not the assistant's (nothing to learn), whose
turns are empty, or whose roles are unknown.

**In code:** `chat_example` builds the message list, `render_chat` flattens
it into (segment, trained?) pairs, and `validate_chat` lists every problem
with an example.

**Why it matters in practice.** A template mismatch is silent: training
succeeds, the loss falls, and the deployed model sees markers it never
learned, so it behaves like the base model or worse.

### 2b. Deduplication

**Everyday picture.** A flashcard deck with the same card in it five times.
You study that card five times as often, and you start answering every
question with it.

**Tiny worked example.** Compare questions by their **word pairs** (two
neighbouring words; a window of k words is called a **shingle**), after
lowercasing and dropping punctuation:

| Text | Word pairs |
|---|---|
| "How do I reset my password?" | how do, do i, i reset, reset my, my password |
| "how do I reset my password, please" | the same 5, plus password please |
| "How do I change my email?" | how do, do i, i change, change my, my email |

The first two share 5 of the 6 distinct pairs between them: a
near-duplicate. The first and third share 2 of 8: different questions.

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $A, B$ | the sets of word pairs of two texts | 5 pairs and 6 pairs |
| $\cap$ | intersection: pairs in both sets | 5 shared pairs |
| $\cup$ | union: pairs in either set, each counted once | 6 distinct pairs |
| $\lvert \cdot \rvert$ | the number of items in a set | |
| $J(A, B)$ | Jaccard similarity, from 0 (nothing shared) to 1 (identical) | 0.83 |

**In words:** "count the pairs the two texts share, and divide by the number
of distinct pairs they have between them."

**With the numbers:** J = 5 / 6 = 0.83 for the two password questions and
2 / 8 = 0.25 for password against email. A threshold of 0.7 keeps the first
password question, drops its rewording, and keeps the email question.

**In Python:**

```python
def pairs(text):
    words = text.lower().replace("?", "").replace(",", "").split()
    return {(a, b) for a, b in zip(words, words[1:])}
A = pairs("How do I reset my password?")
B = pairs("how do I reset my password, please")
C = pairs("How do I change my email?")
# |A ∩ B| and |A ∪ B|
len(A & B), len(A | B)  # → (5, 6)
# J(A, B)
round(len(A & B) / len(A | B), 2)  # → 0.83
# J(A, C)
len(A & C) / len(A | C)  # → 0.25
```

```mermaid
flowchart LR
  T[Next example] --> CMP{Jaccard with any<br/>kept example ≥ 0.7?}
  CMP -->|yes| DROP[Drop it:<br/>a near-copy]
  CMP -->|no| KEEP[Keep it]
  KEEP --> T
  DROP --> T
```

**Reading it:** examples arrive one at a time and are compared against
everything already kept. A match above the threshold is dropped; anything
else joins the kept list. The first copy of each group survives, so the
order of the data decides which wording you keep.

**In code:** `normalize` lowercases and strips punctuation, `shingles`
collects the word pairs, `jaccard` scores two texts and `deduplicate`
returns the indices worth keeping. Comparing every pair is fine for
thousands of examples; at web scale, MinHash estimates the same Jaccard
without comparing every pair.

**Why it matters in practice.** Duplicates overweight a few examples, so the
model parrots them. Worse, a near-copy of an eval question in the training
set lets the model recite the answer, and the eval score becomes a memory
test. Lee et al. found training sets with thousands of near-duplicates, and
removing them made models memorise less.

### 2c. The held-out set comes first

**Everyday picture.** A good teacher writes the final exam before teaching
the course and locks it in a drawer. If the exam were written afterwards,
it would drift towards what the class happened to practise.

**Tiny worked example.** Fifty support questions are deduplicated, shuffled
with a fixed seed, and 10 (20%) go in the drawer. Only then are the other 40
used, and any of the 40 that nearly copies one of the 10 is removed. Every
later choice (prompt wording, learning rate, which checkpoint to ship) is
scored on those 10. But how much can 10, or even 100, questions tell you?

$$
\text{margin} = z \sqrt{\frac{a\,(1 - a)}{n}}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $a$ | the accuracy measured on the held-out set | 0.8 |
| $n$ | the number of held-out examples | 100 |
| $\sqrt{a(1-a)/n}$ | the **standard error**: how much the measured accuracy would wobble if you drew a different held-out set of the same size | 0.04 |
| $z$ | how many standard errors to allow; 1.96 covers 95% of the wobble | 1.96 |
| margin | the true accuracy is probably within ± this of the measured one | 0.078 |

**In words:** "the measured accuracy is uncertain by about two standard
errors, and the standard error shrinks with the square root of the number of
examples."

**With the numbers:** with 100 examples at 80%, the margin is
1.96 × √(0.8 × 0.2 / 100) = 1.96 × 0.04 = 0.078, so "80%" means "somewhere
from about 72% to 88%". A fine-tune that scores 83% has not been shown to
beat a prompt that scores 80%. With 400 examples the margin halves to 0.039.

**In Python:**

```python
import math
a, n, z = 0.8, 100, 1.96
# the standard error sqrt(a(1 − a)/n)
round(math.sqrt(a * (1 - a) / n), 4)  # → 0.04
# the 95% margin
round(z * math.sqrt(a * (1 - a) / n), 4)  # → 0.0784
# four times as many examples halves it
round(z * math.sqrt(a * (1 - a) / 400), 4)  # → 0.0392
```

```mermaid
flowchart LR
  ALL[All examples,<br/>deduplicated] --> SH[Shuffle with<br/>a fixed seed]
  SH --> EV[20% held out<br/>frozen, never trained on]
  SH --> TR[80% training]
  EV --> CHK{Training example<br/>near-copies one?}
  TR --> CHK
  CHK -->|yes| X[Remove from training]
  CHK -->|no| OK[Keep for training]
```

**Reading it:** the fixed seed makes the split repeatable, so everyone on a
team scores against the same held-out set. The held-out branch is frozen the
moment it is made. Both branches then meet in the leak check, which only
ever removes examples from the *training* side.

![The margin of error falls from about 16 points at 25 examples to 1.4 points at 3,200; each fourfold increase halves it](figures/primer.ml.fine_tuning.eval_margin.svg)

**Reading it:** the x-axis is the number of held-out examples (log scale)
and the y-axis is the ± margin, in percentage points, around a measured 80%.
The marked points show the square-root law: 100 examples give ±7.8 points,
400 give ±3.9 and 1,600 give ±2.0. Each halving of the margin costs four
times the examples, which is why held-out sets of a few hundred are common
and a few dozen can only detect big differences.

**In code:** `split_before_training` deduplicates, shuffles with a seed,
freezes the held-out set and removes leaks with `remove_leaks`;
`margin_of_error` gives the ± for any accuracy and size.

**Why it matters in practice.** Every time you look at held-out scores and
change something, the held-out set leaks a little into your decisions. A
held-out set built first and used sparingly is the only honest measure of
whether the fine-tune helped. See `primer.ml.regularization` for train,
validation and test splits, and `primer.agents.evals` for building evals.

### 2d. Label quality

**Everyday picture.** An answer key with typos. A student who gets every
question right is marked wrong wherever the key is wrong, and a student who
makes the *same* mistake as the key is marked right.

**Tiny worked example.** A model is truly right 90% of the time, and 10% of
the reference labels are wrong. It scores a point when it is right on a
right label (0.9 × 0.9 = 0.81), or when it is wrong on a wrong label and the
two mistakes cancel (0.1 × 0.1 = 0.01). The eval reports **82%**, not 90%.

$$
a_{\text{measured}} = a\,(1 - \varepsilon) + (1 - a)\,\varepsilon
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $a$ | the model's true accuracy | 0.9 |
| $\varepsilon$ | epsilon: the share of reference labels that are wrong | 0.1 |
| $a(1-\varepsilon)$ | right answer, right label | 0.81 |
| $(1-a)\varepsilon$ | wrong answer on a wrong label (yes/no labels only, so the two errors agree) | 0.01 |
| $a_{\text{measured}}$ | what the eval reports | 0.82 |

**In words:** "the eval gives credit for being right on a correct label, and
by accident for being wrong on a wrong one."

**With the numbers:** 0.9 × 0.9 + 0.1 × 0.1 = 0.82. A perfect model
($a = 1$) scores only 1 − ε = 0.9: noisy labels put a ceiling on what the
eval can show.

**In Python:**

```python
a, eps = 0.9, 0.1
# right on a right label, plus wrong on a wrong label
round(a * (1 - eps) + (1 - a) * eps, 2)  # → 0.82
# even a perfect model scores only 1 − ε
round(1.0 * (1 - eps), 2)  # → 0.9
# two labelers, five items: how often do they agree?
labeler_1 = ["yes", "no", "yes", "yes", "no"]
labeler_2 = ["yes", "no", "no", "yes", "no"]
sum(p == q for p, q in zip(labeler_1, labeler_2)) / len(labeler_1)  # → 0.8
```

```mermaid
flowchart TD
  X[One eval item] --> R{Model right?}
  R -->|"yes, 0.9"| LR{Label right?}
  R -->|"no, 0.1"| LW{Label right?}
  LR -->|"yes, 0.9"| P1["scored right: 0.81"]
  LR -->|"no, 0.1"| P2["scored wrong: 0.09"]
  LW -->|"yes, 0.9"| P3["scored wrong: 0.09"]
  LW -->|"no, 0.1"| P4["scored right: 0.01"]
```

**Reading it:** each item takes two coin flips: is the model right, and is
the label right? Multiply along a path to get its share. Two paths are
scored as right, 0.81 and 0.01, which add up to the 0.82 the eval reports.
The 0.09 on the second path is a correct answer marked wrong by a bad label.

**How to check label quality.** Have two people label the same sample
independently and measure how often they agree. Above, they agree on 4 of
5 items, 80%: one item in five is ambiguous or mislabelled, and your eval
can't resolve differences smaller than that noise. Read every disagreement;
they are usually unclear instructions, not careless labelers.

**In code:** `measured_accuracy` applies the formula and `label_agreement`
compares two labelers.

**Why it matters in practice.** Bad training labels teach the model the
mistakes (section 4 shows a small model memorising three of them). Bad eval
labels hide real improvements. Zhou et al. (LIMA) fine-tuned a large model
on just 1,000 carefully chosen examples and got a strong assistant: quality
of examples beats quantity.

### 2e. How many examples?

**Everyday picture.** Teaching a house style is not teaching a language.
A new cook learns how you plate a dish from a few dozen good examples; they
don't need ten thousand.

**Tiny worked example.** Fine-tuning teaches a *behaviour* the base model
can almost do already, so the useful sizes are small: a few dozen examples
to show a format, hundreds for a consistent style or a narrow task,
thousands for a harder specialised skill. Section 4 fine-tunes on 16
examples and shows the other side: with so few, the model soon memorises
them, noise included.

```mermaid
flowchart LR
  S[Start with 50 to 100<br/>high-quality examples] --> T[Fine-tune]
  T --> E[Score on the<br/>held-out set]
  E --> Q{Gained more than<br/>the margin of error?}
  Q -->|yes| D[Double the data] --> T
  Q -->|no| STOP[Stop adding data:<br/>fix quality or approach]
```

**Reading it:** the loop grows the dataset only while each doubling still
buys a real improvement, meaning one larger than the held-out margin from
section 2c. When a doubling stops paying, more of the same data won't help;
the fix is better examples or a different approach.

**Why it matters in practice.** Labelled data is the expensive part of
fine-tuning. Doubling until the gains flatten spends it where it helps, and
the held-out set tells you when to stop.

## 3. Catastrophic forgetting

**Everyday picture.** Someone who learned to drive in Britain, keeping
left, moves to the United States and practises keeping right every day for
a month. The new habit wins, and on a trip home they drift to the wrong side
of the road. Nobody told them to forget the old rule; the new practice
simply rewrote the reflex both rules use. Neural networks do this to an
extreme: train on a new task alone and an old one can vanish. This is
**catastrophic forgetting**.

**The model we'll fine-tune.** To watch it happen, we need a model small
enough to train instantly. Each example is four numbers, and the answer is
yes or no:

| Task | Inputs it uses | Rule | Plays the role of |
|---|---|---|---|
| general | all four, anywhere in [−3, 3] | yes when x₁ + x₃ > 0 | the base model's pretraining |
| A | x₁ in [−3, −1], x₂ in [−2, 2] | yes when x₂ > 0 | keep left in Britain |
| B | x₁ in [1, 3], x₂ in [−2, 2] | yes when x₂ < 0 | keep right in the US |
| C | x₃, x₄ in [−2, 2] | yes when x₃ + x₄ > 0 | an unrelated skill |

A and B read the same two inputs and apply opposite rules in different
regions, so one model can learn both, but only by paying attention to the
region. C reads inputs that A and B never touch.

```mermaid
flowchart LR
  X["4 inputs<br/>x1 x2 x3 x4"] --> H["16 hidden units<br/>tanh"]
  H --> O["1 output<br/>probability of yes"]
  TH["θ: all 97 weights<br/>in one vector"] -.-> H
  TH -.-> O
```

**Reading it:** four numbers go in, 16 hidden units each mix all of them,
and one output unit turns the hidden units into a probability. Every weight
(64 in the first layer, 16 hidden biases, 16 output weights and one output
bias) sits in a single vector θ of 97 numbers. Every task flows through the
same hidden units, which is exactly why one task's training can damage
another. `primer.ml.neural_net` builds this kind of network from scratch.

The base model is this network trained on the general task (99% accuracy
on held-out examples). Every fine-tune below starts from it and runs plain
gradient descent on 200 examples.

**In code:** `TinyNet` holds θ and computes predictions, loss and
`TinyNet.gradient`; `make_task` draws examples for each of the `TASKS`;
`base_model` pretrains the base and `fine_tune` trains a copy, leaving the
starting model untouched.

### 3a. The general skill fades while you teach a new one

**Everyday picture.** A month of drilling one cuisine makes the cook a
little rusty at everything else, and a second month of drilling it makes
them rustier still, even though they had mastered the cuisine in the first
week.

**Tiny worked example.** Fine-tune the base on task A (learning rate 0.5)
and check both skills on held-out examples:

| After step | Accuracy on A | General skill | Distance from base |
|---|---|---|---|
| 0 (the base) | 0.53 | 0.99 | 0 |
| 5 | 0.975 | 0.825 | 2.60 |
| 10 | 0.995 | 0.79 | 2.83 |
| 300 | 0.98 | 0.655 | 5.19 |

Task A is learned in 5 steps. The next 295 steps teach nothing new about A
but keep eroding the general skill, from 0.825 to 0.655. The last column
explains why: the weights keep travelling away from the base.

$$
d_t = \lVert \theta_t - \theta_{\text{base}} \rVert = \sqrt{\sum_{i=1}^{P} \left(\theta_{t,i} - \theta_{\text{base},i}\right)^2}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $\theta_{\text{base}}$ | theta: every weight of the base model, as one vector | 97 numbers |
| $\theta_t$ | every weight after $t$ fine-tuning steps | |
| $P$ | the number of weights | 97 (3 in the hand example) |
| $i$ | a counter over the weights | 1 … P |
| $\theta_{t,i}$ | the $i$-th weight after $t$ steps | |
| $\lVert v \rVert$ | the length of vector $v$: square each entry, add, take the square root | |
| $d_t$ | how far fine-tuning has moved the model | 5.19 after 300 steps |

**In words:** "the distance travelled is the length of the change in the
weights: square every weight's change, add them up, take the square root."

**With the numbers:** for a three-weight model moving from (1, 0, 2) to
(1.5, −1, 2), the changes are (0.5, −1, 0), and
d = √(0.25 + 1 + 0) = √1.25 = 1.118.

**In Python:**

```python
import math
theta_base = [1.0, 0.0, 2.0]
theta_t = [1.5, -1.0, 2.0]
# each weight's change
[t - b for t, b in zip(theta_t, theta_base)]  # → [0.5, -1.0, 0.0]
# ‖θ_t − θ_base‖
round(math.sqrt(sum((t - b) ** 2 for t, b in zip(theta_t, theta_base))), 3)  # → 1.118
```

```mermaid
flowchart LR
  DA[Task A examples] -->|gradient| W["Shared weights θ"]
  W --> SA[Answers on task A]
  W --> SG[Answers on the<br/>general skill]
  SG -.->|"no examples,<br/>no gradient"| W
```

**Reading it:** the gradient from task A's examples moves the shared
weights, and both skills read those same weights. The dotted arrow is what
is missing: the general skill has no examples in this training run, so
nothing pushes back when a change that helps A hurts it. Forgetting is not
an event; it is the absence of a counterweight.

![Left: accuracy on A jumps to 1 within a few steps while the general skill slides from 0.99 to 0.66 at learning rate 0.5 and only to 0.79 at 0.02. Right: the general skill falls as distance from the base grows](figures/primer.ml.fine_tuning.general_skill.svg)

**Reading it:** on the left, the x-axis is the training step (log scale).
Blue lines are accuracy on task A, red lines the general skill; solid is
learning rate 0.5, dashed is 0.02. With the big rate, A is learned within a
few steps and the general skill then slides for the rest of the run. With
the small rate, A is learned later (around step 70) and the general skill
ends at 0.79 instead of 0.655, with A just as good. On the right, every
step of both runs is plotted as distance from the base against the general
skill: the points fall along one downward curve. **Forgetting tracks how
far the weights move.**

**Mitigations that shorten the trip.** Three knobs limit the distance, and
the toy measures two of them:

- **Fewer steps.** Stop once the held-out score on the new task stops
  improving. Here, stopping at step 10 keeps the general skill at 0.79
  instead of 0.655, with A at 0.995.
- **A lower learning rate.** Smaller steps travel less far for the same
  result: 0.79 instead of 0.655 after 300 steps.
- **A smaller update.** LoRA (`primer.ml.training_stages`, section 4)
  freezes the base weights and allows only a low-rank change. Biderman et
  al. measured this on real language models and summed it up in their
  title: LoRA learns less and forgets less.

**In code:** `general_skill_run` fine-tunes on A and records, after every
step, accuracy on A, the general skill and the distance from the base.

**Why it matters in practice.** A fine-tuned assistant that has become
worse at everything outside its narrow task is the most common
fine-tuning disappointment. Always score the general skills you care about
alongside the new task, and prefer the earliest checkpoint that has learned
the task.

### 3b. A new task that contradicts an old one

**Everyday picture.** Back to the driver. Practising "keep right" does not
merely add a skill; it pushes directly against "keep left", because both
use the same reflex.

**Tiny worked example.** Take the model fine-tuned on A (98% on A), then
fine-tune it on B alone for 300 steps:

| Second fine-tune | A before | A after | New task after |
|---|---|---|---|
| on B (contradicts A) | 0.98 | **0.025** | 1.00 |
| on C (separate inputs) | 0.98 | 0.97 | 0.98 |

After B, the model does not merely forget A; it answers A's questions
backwards, because it learned "yes when x₂ < 0" everywhere. After C, A is
barely touched: C's inputs never flowed through the weights A relies on
most.

$$
F = \text{acc}_{\text{old}}^{\text{before}} - \text{acc}_{\text{old}}^{\text{after}}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $\text{acc}_{\text{old}}^{\text{before}}$ | held-out accuracy on the old task before the new fine-tune | 0.98 |
| $\text{acc}_{\text{old}}^{\text{after}}$ | the same, after the new fine-tune | 0.025 |
| $F$ | forgetting: accuracy lost on the old task | 0.955 |

**In words:** "forgetting is how much accuracy the old task lost."

**With the numbers:** F = 0.98 − 0.025 = 0.955 after B, and
0.98 − 0.97 = 0.01 after C.

**In Python:**

```python
acc_before, acc_after_B, acc_after_C = 0.98, 0.025, 0.97
# F after the contradicting task B
round(acc_before - acc_after_B, 3)  # → 0.955
# F after task C, on separate inputs
round(acc_before - acc_after_C, 3)  # → 0.01
```

```mermaid
flowchart LR
  BASE[Base] -->|fine-tune on A| MA["Model knows A<br/>A: 0.98"]
  MA -->|fine-tune on B only| MB["Model knows B<br/>A: 0.025, B: 1.00"]
  MA -->|fine-tune on C only| MC["Model knows A and C<br/>A: 0.97, C: 0.98"]
```

**Reading it:** both second fine-tunes start from the same model. The only
difference is which weights the new task needs: B needs the very ones A
uses, in the opposite direction; C mostly needs others. How much a model
forgets depends less on how long you train than on how much the new task
overlaps and conflicts with the old one.

![Fine-tuning on B after A: accuracy on A falls from 0.98 to near 0 within a few steps while B rises to 1; with 10 replayed A examples, A dips and then recovers to 0.945](figures/primer.ml.fine_tuning.sequential.svg)

**Reading it:** the x-axis is the step of the second fine-tune (log scale),
the y-axis held-out accuracy. Solid lines are plain fine-tuning on B: B
(green) climbs to 1 while A (blue) falls to about half in the same few
steps, and A keeps sliding towards zero for as long as training continues.
Dashed lines add 10 replayed A examples (section 3c): A still dips
at first (to about 0.2 around step 40), then climbs back to 0.945 while B
reaches 0.99.

**Does a lower learning rate help here?** Only in the sense of slowing the
slide. At learning rate 0.02, or stopping after 10 steps, B reaches 0.975
and A still drops to 0.355.

![Accuracy on A against accuracy on B during the second fine-tune: every run on B alone traces the same curve whatever the learning rate and ends near zero on A, while the replay run climbs the right edge to the top-right corner](figures/primer.ml.fine_tuning.tradeoff.svg)

**Reading it:** each line traces one run: x is accuracy on B, y is accuracy
on A, and each run starts at the top left (knows A, not B). The three
learning rates (0.5, 0.1, 0.02) take very different numbers of steps but
trace the same curve, close to the dotted line where A + B = 1: every point
of B they gain costs about a point of A, and once B is learned, further
steps slide A down the right edge towards zero. The replay run (dashed)
follows the same curve at first, then climbs the right edge and ends in
the top-right corner, knowing both. When the new data
contradicts the old skill, slowing down can't help, because nothing in B's
data says A still matters.

**In code:** `sequential_run` starts from the A fine-tune, trains on a new
task (optionally with replay), and records both accuracies at every step;
`forgetting` computes F.

**Why it matters in practice.** Real fine-tunes contradict the base model
more often than you'd think: "always answer in JSON" contradicts "chat
naturally"; "be terse" contradicts "explain in detail". Expect the old
behaviour to vanish wherever your data overrides it, and test for it.

### 3c. Replay: keep practising the old skill

**Everyday picture.** A pianist learning a new piece plays one old piece
at the start of every practice session. It costs a few minutes and keeps
the old repertoire alive.

**Tiny worked example.** Add just 10 of task A's 200 training examples to
B's 200: under 5% of the mix. The result: A **0.945**, B 0.99 (up from A
0.025 without replay). Why can so few examples do so much? Look at the loss.
Suppose that early in training the model already scores B well (loss 0.05
per example) but has started forgetting A (loss 3.0 on each replayed
example):

$$
\mathcal{L}_{\text{mix}} = \frac{n_{\text{new}}\,\mathcal{L}_{\text{new}} + n_{\text{old}}\,\mathcal{L}_{\text{old}}}{n_{\text{new}} + n_{\text{old}}}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $n_{\text{new}}$ | examples of the new task | 200 |
| $n_{\text{old}}$ | replayed examples of the old task | 10 |
| $\mathcal{L}_{\text{new}}$ | average loss on the new examples | 0.05 |
| $\mathcal{L}_{\text{old}}$ | average loss on the replayed examples | 3.0 |
| $\mathcal{L}_{\text{mix}}$ | the loss training actually minimises: the average over every example in the mix | 0.19 |

**In words:** "the loss on a mixed dataset is the average over all its
examples, so each group counts in proportion to its size times its loss."

**With the numbers:** (200 × 0.05 + 10 × 3.0) / 210 = (10 + 30) / 210 =
0.19. The 10 replayed examples are under 5% of the data but contribute
0.143 of the 0.19: three quarters of the loss, and so most of the gradient.
The old examples shout loudest exactly when they are being forgotten.

**In Python:**

```python
n_new, L_new = 200, 0.05
n_old, L_old = 10, 3.0
# the replayed share of the data
round(n_old / (n_new + n_old), 3)  # → 0.048
# each group's contribution to the average loss
round(n_new * L_new / 210, 3), round(n_old * L_old / 210, 3)  # → (0.048, 0.143)
# L_mix
round((n_new * L_new + n_old * L_old) / (n_new + n_old), 2)  # → 0.19
```

```mermaid
flowchart LR
  NB["New task B<br/>200 examples"] --> MIX[Shuffle together<br/>210 examples]
  OA["Old task A<br/>10 kept examples"] --> MIX
  MIX --> FT[Fine-tune]
  FT --> BOTH["Knows B: 0.99<br/>and A: 0.945"]
```

**Reading it:** the old task's small sample joins the new data before
training, so every gradient step sees both. This supplies exactly the
counterweight that was missing in section 3a's diagram: when a change that
helps B starts hurting A, the replayed examples' loss rises and pushes back.

**In code:** `replay_mix` appends the old examples, `mixed_loss` is the
formula, and `sequential_run` with `n_replay=10` runs the experiment.

**Why it matters in practice.** When fine-tuning a language model, mix some
general instruction-following data into your task data, so the model keeps
being a good assistant while it learns your task. When you can't replay
(the old data is gone or private), methods such as elastic weight
consolidation (Kirkpatrick et al.) instead penalise changes to the weights
the old task relied on most.

## 4. Overfitting a small dataset

**Everyday picture.** A student with only 16 flashcards, three of which
have the wrong answer on the back. For a while, studying teaches the
pattern. Keep drilling and the student memorises every card word for word,
including the three wrong answers, and gets worse on new questions.
`primer.ml.regularization` builds this idea from scratch; here it is in a
fine-tune.

**Tiny worked example.** Fine-tune the base on only 16 examples of task A,
3 of them deliberately mislabelled, for 1,500 epochs. (With full-batch
training, one step is one pass over the data, one **epoch**.)

| | At the best epoch (70) | At the end (1,500 epochs) |
|---|---|---|
| training loss | 0.373 | 0.005 |
| validation loss | **0.319** | 1.277 |

At the best epoch, training loss is *higher* than validation loss: the
model is refusing to fit the three wrong labels, which is exactly right. By
the end, training loss is nearly zero, so the wrong labels have been
memorised, and validation loss has quadrupled.

$$
g(t) = \mathcal{L}_{\text{val}}(t) - \mathcal{L}_{\text{train}}(t)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $t$ | the epoch | 70, then 1,500 |
| $\mathcal{L}_{\text{train}}(t)$ | average loss on the 16 training examples | 0.373, then 0.005 |
| $\mathcal{L}_{\text{val}}(t)$ | average loss on 200 held-out examples | 0.319, then 1.277 |
| $g(t)$ | the **generalisation gap**: how much worse the model does on data it hasn't seen | −0.054, then 1.272 |

**In words:** "the gap is held-out loss minus training loss; a gap that
keeps growing means the model is memorising rather than learning."

**With the numbers:** 0.319 − 0.373 = −0.054 at epoch 70; 1.277 − 0.005 =
1.272 at the end.

**In Python:**

```python
L_train = {"best": 0.373, "end": 0.005}
L_val = {"best": 0.319, "end": 1.277}
# g = L_val − L_train at each point
{t: round(L_val[t] - L_train[t], 3) for t in L_val}  # → {'best': -0.054, 'end': 1.272}
```

![Training loss falls steadily to near zero while validation loss bottoms out at epoch 70 and then climbs to four times its best](figures/primer.ml.fine_tuning.overfitting.svg)

**Reading it:** the x-axis is the epoch (log scale), the y-axis the loss.
Both curves fall at first while the model learns the real rule. At epoch 70
(the dashed line) validation loss bottoms out; after that the training
curve keeps falling as the model memorises the three wrong labels, and the
validation curve climbs. The dotted line is where early stopping with a
patience of 20 epochs would end the run, keeping the weights from epoch 70.

```mermaid
flowchart LR
  EP[Train one epoch] --> SV[Save a checkpoint]
  SV --> SC[Score it on the<br/>held-out set]
  SC --> Q{Best so far?}
  Q -->|yes| MARK[Mark it best] --> EP
  Q -->|"no, patience used up"| SHIP[Ship the best checkpoint,<br/>not the last]
  Q -->|"no, patience left"| EP
```

**Reading it:** every epoch ends with a checkpoint and a held-out score.
The loop keeps going while scores improve or patience remains, and when it
stops, the checkpoint that ships is the marked best, not whatever the last
epoch left behind.

**In code:** `overfitting_run` fine-tunes on the 16 examples and records
both losses every epoch; `primer.ml.regularization.early_stopping` replays
the validation curve and returns the best epoch and the stopping epoch.

**Why it matters in practice.** Fine-tuning datasets are small compared to
pretraining, and big models memorise quickly, so fine-tunes typically run
for only a few epochs. Save checkpoints, score each on the held-out set,
and ship the best one.

## 5. Model merging

### 5a. Weight averaging and task arithmetic

**Everyday picture.** Two editors each take a copy of the same draft and
make tracked changes: one fixes the grammar, the other tightens the
argument. You can apply both sets of changes to the original. If instead
you "average" the two edited copies, each change is applied at half
strength: half the grammar fixed, half the argument tightened.

**Tiny worked example.** A three-weight base (1, 0, 2). One fine-tune
moves it to (1.5, 0, 2); another to (1, −1, 2).

| | Weights | Change from the base |
|---|---|---|
| base | (1, 0, 2) | |
| fine-tune on A | (1.5, 0, 2) | τ_A = (0.5, 0, 0) |
| fine-tune on C | (1, −1, 2) | τ_C = (0, −1, 0) |
| base + τ_A + τ_C | **(1.5, −1, 2)** | both changes in full |
| average of the two fine-tunes | (1.25, −0.5, 2) | both changes at half strength |

The change a fine-tune made, θ_ft − θ_base, is its **task vector**. Adding
task vectors to the base is **task arithmetic**.

$$
\tau_t = \theta_t - \theta_{\text{base}}, \qquad
\theta_{\text{merged}} = \theta_{\text{base}} + \lambda \sum_{t=1}^{T} \tau_t
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $t$ | which task: a counter over the fine-tunes | A, C |
| $T$ | how many fine-tunes are merged | 2 |
| $\theta_t$ | all the weights of the model fine-tuned on task $t$ | (1.5, 0, 2) |
| $\theta_{\text{base}}$ | the weights they all started from | (1, 0, 2) |
| $\tau_t$ | tau: task $t$'s task vector, everything its fine-tune changed | (0.5, 0, 0) |
| $\sum_{t=1}^{T}$ | add up the task vectors | τ_A + τ_C = (0.5, −1, 0) |
| $\lambda$ | lambda: how strongly to apply the combined changes | 1 |
| $\theta_{\text{merged}}$ | the merged model's weights | (1.5, −1, 2) |

**In words:** "each task vector is what its fine-tune changed; add the
changes up, scale them by λ, and apply them to the base."

**With the numbers:** θ_merged = (1, 0, 2) + 1 × (0.5, −1, 0) = (1.5, −1, 2).
With λ = 1/2: (1, 0, 2) + 0.5 × (0.5, −1, 0) = (1.25, −0.5, 2), which is
exactly the average of the two fine-tunes. **Averaging T fine-tunes is task
arithmetic with λ = 1/T.**

**In Python:**

```python
theta_base = [1.0, 0.0, 2.0]
theta_A = [1.5, 0.0, 2.0]
theta_C = [1.0, -1.0, 2.0]
# τ_t = θ_t − θ_base
tau_A = [a - b for a, b in zip(theta_A, theta_base)]
tau_C = [c - b for c, b in zip(theta_C, theta_base)]
tau_A, tau_C  # → ([0.5, 0.0, 0.0], [0.0, -1.0, 0.0])
# θ_merged at λ = 1
[b + 1.0 * (a + c) for b, a, c in zip(theta_base, tau_A, tau_C)]  # → [1.5, -1.0, 2.0]
# λ = 1/2 ...
[b + 0.5 * (a + c) for b, a, c in zip(theta_base, tau_A, tau_C)]  # → [1.25, -0.5, 2.0]
# ... is the plain average of the two fine-tunes
[(a + c) / 2 for a, c in zip(theta_A, theta_C)]  # → [1.25, -0.5, 2.0]
```

```mermaid
flowchart LR
  B[Base θ] -->|fine-tune on A| FA[θ_A]
  B -->|fine-tune on C| FC[θ_C]
  FA --> TA["τ_A = θ_A − θ_base"]
  FC --> TC["τ_C = θ_C − θ_base"]
  TA --> SUM["λ × (τ_A + τ_C)"]
  TC --> SUM
  B --> ADD((+))
  SUM --> ADD
  ADD --> M[Merged model:<br/>no extra training]
```

**Reading it:** both fine-tunes start from the same base; that shared start
is what makes their changes comparable. Subtracting the base turns each
fine-tuned model into a task vector, the arrows are added and scaled, and
the result is added back to the base. No data and no gradient steps are
involved: merging is arithmetic on weights, and the merged model is the same
size and speed as the base.

![Merging the fine-tunes on A and C: at lambda 1 the merged model scores 0.97 on both, while plain averaging (lambda one half) scores only 0.635 on A](figures/primer.ml.fine_tuning.merge_separate.svg)

**Reading it:** the x-axis is λ and the y-axis held-out accuracy of the
merged model on A (blue) and on C (green); the dotted line is 0.5, a coin
flip. At λ = 1/2, which is plain averaging, each skill is diluted: 0.635 on
A. Around λ = 1 the merged model does both tasks at 0.97, as well as either
specialist on its own task. Push λ further and the changes overshoot. Their
task vectors are nearly perpendicular (cosine −0.04), so adding one barely
disturbs the other.

**In code:** `task_vector` subtracts the base, `merge` adds scaled task
vectors back, and `merge_run` merges two fine-tunes at several λ and scores
the result.

**Why it matters in practice.** Merging combines skills trained separately,
by different teams or on data that can't be pooled, without any further
training and at no extra inference cost. Averaging several fine-tunes of the
same task ("model soups", Wortsman et al.) often beats the best single one.
Task vectors can also be *subtracted*: Ilharco et al. negated a task vector
learned from toxic text to make a model less toxic.

### 5b. Interference: when task vectors collide

**Everyday picture.** Two editors rewrote the *same* sentence in opposite
directions, one making it warmer and one making it colder. Applying both
sets of tracked changes gives a sentence neither intended.

**Tiny worked example.** Our three-weight changes again, plus a new one:
τ_A = (0.5, 0, 0), τ_C = (0, −1, 0) and τ_B = (−0.4, 0, 0.3). τ_A and τ_C
touch different weights: no conflict. τ_B pulls the first weight the other
way from τ_A. The **cosine** measures how aligned two changes are:

$$
\cos(\tau_A, \tau_B) = \frac{\tau_A \cdot \tau_B}{\lVert \tau_A \rVert \, \lVert \tau_B \rVert}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $\tau_A \cdot \tau_B$ | dot product: multiply matching entries, then add | 0.5 × (−0.4) = −0.2 |
| $\lVert \tau \rVert$ | a vector's length | 0.5 and 0.5 |
| $\cos$ | the cosine of the angle between the two changes: 1 same direction, 0 unrelated, −1 opposite | −0.8 |

**In words:** "multiply the changes weight by weight and add, then divide by
both lengths, so only the direction counts."

**With the numbers:** τ_A · τ_B = 0.5 × (−0.4) + 0 + 0 = −0.2; ‖τ_A‖ = 0.5,
‖τ_B‖ = √(0.16 + 0.09) = 0.5; cos = −0.2 / 0.25 = **−0.8**: strongly
opposed. cos(τ_A, τ_C) = 0: independent. See `primer.ml.embeddings.similarity`
for the cosine from scratch.

**In Python:**

```python
import math
tau_A = [0.5, 0.0, 0.0]
tau_B = [-0.4, 0.0, 0.3]
tau_C = [0.0, -1.0, 0.0]
def cos(u, v):
    dot = sum(a * b for a, b in zip(u, v))
    return dot / (math.sqrt(sum(a * a for a in u)) * math.sqrt(sum(b * b for b in v)))
# opposed changes
round(cos(tau_A, tau_B), 2)  # → -0.8
# changes to different weights
cos(tau_A, tau_C)  # → 0.0
```

In the toy, the fine-tunes on A and on B (which reverses A's rule) have task
vectors with cosine **−0.25**, against −0.04 for A and C. Each fine-tune
learned its rule everywhere, not just in its own region, so the two vectors
rewrite the same weights in opposite directions.

![Merging the fine-tunes on A and B: at every lambda at least one task stays at or below a coin flip, and the best the merge manages on both at once is 0.525](figures/primer.ml.fine_tuning.merge_conflict.svg)

**Reading it:** the same axes as the previous figure, now merging A with B.
No value of λ lifts both lines: wherever one task improves, the other
sits near or below a coin flip, and from λ = 1 on both stay there, because
the two changes cancel. A
model *can* do both tasks (replay in section 3c got 0.945 and 0.99), but
this merge can't reach it: that model needs to tell the regions apart, and
neither fine-tune learned to.

```mermaid
flowchart TD
  TV[Task vectors from<br/>the same base] --> COS{Cosine between them}
  COS -->|"near 0: separate weights"| ADD[Add them:<br/>task arithmetic]
  COS -->|"clearly negative: conflict"| FIX{Can you retrain?}
  FIX -->|yes| JOINT[Train one model on<br/>both datasets, or replay]
  FIX -->|no| TIES["Resolve conflicts:<br/>trim small changes,<br/>agree on a sign per weight"]
  ADD --> EV[Score the merge on<br/>every task's held-out set]
  TIES --> EV
  JOINT --> EV
```

**Reading it:** a cheap check before merging is the cosine between task
vectors. Near zero, the changes live in different weights and adding them
usually works. Clearly negative, they fight: train on both datasets
together if you can; if you can't, conflict-resolving merges such as
TIES-merging (Yadav et al.) drop each task's smallest changes and, for each
weight, keep only the changes that agree with the majority sign. Every path
ends at the same place: score the merge on every task's held-out set.

**In code:** `cosine` measures the angle between two task vectors, and
`merge_run` reports it alongside the merged accuracies.

**Why it matters in practice.** Merges are free to try, which makes them
tempting to trust. A merged model can quietly lose a skill both parents
had, so it is evaluated like any new model. The cosine tells you in advance
which merges to be nervous about.

## In 20 seconds

- **Decide with an eval:** build a held-out set first; try prompting, then
  retrieval, and fine-tune only for behaviour a prompt can't pin down. It
  pays off after C / (c_prompt − c_tuned) requests.
- **Data:** use the model's chat template and train only on assistant turns;
  deduplicate (Jaccard on word shingles); freeze the held-out set before
  training and remove near-copies of it; audit labels, because wrong labels
  cap what the eval can show. Quality beats quantity.
- **Forgetting:** every weight is shared, so learning a new task erodes old
  ones, in proportion to how far the weights move and how much the tasks
  conflict. Fewer steps, a lower learning rate and LoRA shorten the trip;
  replaying a little old data is what keeps a contradicted skill.
- **Overfitting:** on small data, validation loss bottoms out early; ship
  the best checkpoint, not the last.
- **Merging:** a task vector is θ_ft − θ_base. Adding task vectors combines
  skills without training; averaging is λ = 1/T and dilutes them; vectors
  that point in opposite directions interfere.

## Self-test questions

**When is fine-tuning the wrong tool, and what should you try first?**
When the model lacks facts, or the facts change: retrieval supplies them and
is easy to update. When a clearer prompt with a few examples fixes the
behaviour: that is cheaper and survives a base-model upgrade. Fine-tuning
earns its cost when behaviour stays inconsistent under the best prompt, or
when a long prompt sent millions of times costs more than the fine-tune.

**Why build the held-out set before training, and why check it against the training set?**
So that no choice (prompt, learning rate, checkpoint) is made by looking at
it, and it stays an honest measure. A training example that nearly copies a
held-out one lets the model recite the answer, turning the eval into a
memory test; deduplicating across the split prevents it.

**A held-out set has 100 examples and the fine-tune scores 83% against the prompt's 80%. Has it won?**
Not yet. At 80% on 100 examples the 95% margin is about ±7.8 points, so a
3-point difference is well inside the noise. You need a larger held-out set
(400 examples halve the margin) or a bigger difference.

**If 10% of the eval's labels are wrong, what is the best score a perfect model can get?**
90%, because it is marked wrong on every mislabelled item. A 90%-accurate
model would score 0.9 × 0.9 + 0.1 × 0.1 = 82%. Noisy labels shrink and blur
the differences you are trying to measure.

**What is catastrophic forgetting, and why does it happen?**
Training on a new task alone erodes, or wipes out, skills the model had.
Every weight is shared between tasks, and only the new task's examples
produce gradients, so nothing pushes back when a change that helps the new
task hurts an old one. It grows with how far the weights move and with how
much the new task conflicts with the old.

**Lowering the learning rate did not stop task A being forgotten. Why not, and what works?**
Task B contradicts A on the same inputs, so any progress on B costs A;
a lower rate only walks the same trade-off more slowly. Replay works: mixing
even 5% of A's examples into B's data gives the model a reason to keep A,
and those few examples carry most of the loss exactly when A is slipping.

**Training loss keeps falling, but validation loss has risen since epoch 70. What is happening, and which checkpoint do you ship?**
The model has stopped learning the general rule and is memorising the
training set, including its mislabelled examples. Ship the checkpoint from
epoch 70, the best on the held-out set; early stopping automates exactly
this.

**What is a task vector, and why is averaging two fine-tunes the same as task arithmetic with λ = 1/2?**
A task vector is everything a fine-tune changed: θ_ft − θ_base. The average
of two fine-tunes is (θ_base + τ_1 + θ_base + τ_2) / 2 = θ_base + ½(τ_1 + τ_2),
which is task arithmetic with λ = 1/2, so each skill arrives at half
strength.

**When does merging fail, and how can you see it coming?**
When the task vectors change the same weights in opposite directions, so
adding them cancels both skills. A clearly negative cosine between task
vectors is the warning; the remedy is joint training or replay, or a
conflict-resolving merge such as TIES, and a held-out check on every task
either way.

## The papers behind this lesson

- **Ilharco et al., *Editing Models with Task Arithmetic* (2022)**:
  https://arxiv.org/abs/2212.04089. Defined task vectors as fine-tuned
  minus pretrained weights and showed that adding them combines skills,
  and negating them removes a behaviour.
- **Wortsman et al., *Model soups: averaging weights of multiple fine-tuned
  models improves accuracy without increasing inference time* (2022)**:
  https://arxiv.org/abs/2203.05482. Showed that averaging the weights of
  several fine-tunes of one base often beats the best single fine-tune.
- **Yadav et al., *TIES-Merging: Resolving Interference When Merging
  Models* (2023)**: https://arxiv.org/abs/2306.01708. Traced failed merges
  to small redundant changes and sign conflicts, and fixed both by trimming
  and electing a sign per weight.
- **Kirkpatrick et al., *Overcoming catastrophic forgetting in neural
  networks* (2017)**: https://arxiv.org/abs/1612.00796. Introduced elastic
  weight consolidation, which slows learning on the weights most important
  to earlier tasks.
- **Biderman et al., *LoRA Learns Less and Forgets Less* (2024)**:
  https://arxiv.org/abs/2405.09673. Measured on real language models that
  LoRA keeps more of the base model's abilities than full fine-tuning, at
  the price of learning the new task less completely.
- **Zhou et al., *LIMA: Less Is More for Alignment* (2023)**:
  https://arxiv.org/abs/2305.11206. Fine-tuned a large base model on 1,000
  carefully curated examples and got a strong assistant, evidence that
  example quality matters more than quantity.
- **Lee et al., *Deduplicating Training Data Makes Language Models Better*
  (2021)**: https://arxiv.org/abs/2107.06499. Found widespread near-duplicates
  in standard datasets, including between training and test sets, and
  showed that removing them reduces memorisation.

## Further reading

- Goodfellow et al., *An Empirical Investigation of Catastrophic Forgetting in Gradient-Based Neural Networks* (2013): https://arxiv.org/abs/1312.6211
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021): https://arxiv.org/abs/2106.09685
- Ilharco et al., *Editing Models with Task Arithmetic* (2022): https://arxiv.org/abs/2212.04089
- Yadav et al., *TIES-Merging* (2023): https://arxiv.org/abs/2306.01708
- Hugging Face TRL, supervised fine-tuning trainer: https://huggingface.co/docs/trl/sft_trainer
- Hugging Face PEFT, parameter-efficient fine-tuning (LoRA and friends): https://huggingface.co/docs/peft/index
- mergekit, an open-source toolkit for merging models: https://github.com/arcee-ai/mergekit
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from primer._show import banner, say, table, takeaway

# ---------------------------------------------------------------------------
# 1. Should you fine-tune? The cost arithmetic
# ---------------------------------------------------------------------------


def per_request_cost(tokens: int, price_per_million: float) -> float:
    """Dollars for one request that sends `tokens` input tokens at `price_per_million` dollars per million."""
    return tokens * price_per_million / 1_000_000


def break_even_requests(fixed_cost: float, cost_prompted: float, cost_tuned: float) -> float:
    """How many requests before a fine-tune's one-off cost is paid back: C / (c_prompt - c_tuned).

    If the tuned model is not cheaper per request, the saving never
    arrives, so the answer is infinity: fine-tune for quality, not cost.
    """
    saving = cost_prompted - cost_tuned
    return fixed_cost / saving if saving > 0 else float("inf")


# ---------------------------------------------------------------------------
# 2. Preparing data: chat format, deduplication, held-out set, label quality
# ---------------------------------------------------------------------------

ROLES = ("system", "user", "assistant")


def chat_example(system: str, user: str, assistant: str) -> dict:
    """One training example in the common chat format: a list of role-tagged messages."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def validate_chat(example: dict) -> list[str]:
    """Every problem that would make this example teach the wrong thing (empty list = fine)."""
    messages = example.get("messages", [])
    problems = []
    for i, m in enumerate(messages, 1):
        if m.get("role") not in ROLES:
            problems.append(f"turn {i} has unknown role {m.get('role')!r}")
        elif not str(m.get("content", "")).strip():
            problems.append(f"turn {i} ({m['role']}) is empty")
    # Training teaches the model to produce the final assistant turn; without one there is no target.
    if not messages or messages[-1].get("role") != "assistant":
        problems.append("the last turn must be the assistant's")
    return problems


def render_chat(example: dict) -> list[tuple[str, bool]]:
    """Flatten messages into the text the model sees, as (segment, trained?) pairs.

    The role markers are illustrative; every model family has its own chat
    template, and you must use the one the base model was trained with.
    Only assistant segments are trained on: the loss mask of SFT (see
    `primer.ml.training_stages`).
    """
    return [(f"<|{m['role']}|>{m['content']}<|end|>", m["role"] == "assistant") for m in example["messages"]]


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse spaces: so trivial differences don't hide a duplicate."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def shingles(text: str, k: int = 2) -> set[tuple[str, ...]]:
    """The set of k-word windows ("shingles") in the normalized text."""
    words = normalize(text).split()
    return {tuple(words[i : i + k]) for i in range(max(len(words) - k + 1, 1))}


def jaccard(a: str, b: str, k: int = 2) -> float:
    """|A ∩ B| / |A ∪ B| over word k-grams: 1 = same text, 0 = nothing shared."""
    sa, sb = shingles(a, k), shingles(b, k)
    return len(sa & sb) / len(sa | sb)


def deduplicate(texts: list[str], threshold: float = 0.7) -> list[int]:
    """Indices of the texts to keep: the first of every group of near-duplicates.

    All pairs are compared here, which is fine for thousands of examples;
    at web scale MinHash estimates the same Jaccard without comparing every pair.
    """
    kept: list[int] = []
    for i, t in enumerate(texts):
        if all(jaccard(t, texts[j]) < threshold for j in kept):
            kept.append(i)
    return kept


def remove_leaks(train: list[str], held_out: list[str], threshold: float = 0.7) -> list[int]:
    """Indices of training texts that do NOT nearly copy any held-out text."""
    return [i for i, t in enumerate(train) if all(jaccard(t, e) < threshold for e in held_out)]


def split_before_training(texts: list[str], eval_fraction: float = 0.2, seed: int = 0) -> tuple[list[str], list[str]]:
    """Deduplicate, set aside a held-out set, then drop training texts that leak into it.

    The held-out set is chosen first and frozen, before any training or
    prompt tuning, so no decision is ever made by looking at it.
    """
    unique = [texts[i] for i in deduplicate(texts)]
    order = np.random.default_rng(seed).permutation(len(unique))
    n_eval = round(eval_fraction * len(unique))
    held_out = [unique[i] for i in order[:n_eval]]
    train = [unique[i] for i in order[n_eval:]]
    return [train[i] for i in remove_leaks(train, held_out)], held_out


def margin_of_error(accuracy: float, n: int, z: float = 1.96) -> float:
    """Half-width of the 95% interval around an accuracy measured on n examples: z·sqrt(a(1-a)/n)."""
    return z * np.sqrt(accuracy * (1 - accuracy) / n)


def measured_accuracy(true_accuracy: float, label_error: float) -> float:
    """What a yes/no eval reports when a fraction `label_error` of its reference labels are wrong.

    The model scores a point when it is right on a right label, or wrong on
    a wrong label (the two mistakes cancel): a(1 - ε) + (1 - a)ε.
    """
    a, e = true_accuracy, label_error
    return a * (1 - e) + (1 - a) * e


def label_agreement(labels_1: list, labels_2: list) -> float:
    """Share of items two labelers labelled the same way."""
    return float(np.mean([p == q for p, q in zip(labels_1, labels_2)]))


# ---------------------------------------------------------------------------
# 3. A tiny model to fine-tune: 4 inputs -> 16 tanh units -> 1 probability
# ---------------------------------------------------------------------------

N_IN, N_HIDDEN = 4, 16
# Where each weight lives in the flat parameter vector θ (task arithmetic needs one vector).
_SHAPES = (("W1", (N_IN, N_HIDDEN)), ("b1", (N_HIDDEN,)), ("w2", (N_HIDDEN,)), ("b2", (1,)))
N_PARAMS = sum(int(np.prod(s)) for _, s in _SHAPES)


def _unpack(theta: np.ndarray) -> dict[str, np.ndarray]:
    out, i = {}, 0
    for name, shape in _SHAPES:
        n = int(np.prod(shape))
        out[name] = theta[i : i + n].reshape(shape)
        i += n
    return out


class TinyNet:
    """A two-layer network whose every weight lives in one flat vector `theta`.

    Keeping θ flat is what makes fine-tuning arithmetic literal: a task
    vector is `theta_tuned - theta_base`, and merging is vector addition.
    """

    def __init__(self, theta: np.ndarray):
        self.theta = np.asarray(theta, dtype=float)

    @classmethod
    def random(cls, seed: int = 0) -> "TinyNet":
        rng = np.random.default_rng(seed)
        parts = {
            "W1": rng.normal(0, 1, (N_IN, N_HIDDEN)),
            "b1": np.zeros(N_HIDDEN),
            # 1/sqrt(fan-in) keeps the output's starting scale near 1 (see primer.ml.deep_nets).
            "w2": rng.normal(0, 1 / np.sqrt(N_HIDDEN), N_HIDDEN),
            "b2": np.zeros(1),
        }
        return cls(np.concatenate([parts[name].ravel() for name, _ in _SHAPES]))

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = _unpack(self.theta)
        h = np.tanh(X @ p["W1"] + p["b1"])  # (n, 16)
        z = h @ p["w2"] + p["b2"][0]  # (n,)
        return h, 1 / (1 + np.exp(-z))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Probability of "yes" for each row of X (shape (n, 4))."""
        return self._forward(X)[1]

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean((self.predict(X) > 0.5) == y))

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Average cross-entropy: -log of the probability given to the right answer."""
        q = np.clip(self.predict(X), 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    def gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """d loss / d θ by backpropagation, flattened like θ."""
        p = _unpack(self.theta)
        h, q = self._forward(X)
        dz = (q - y) / len(y)  # (n,): sigmoid + cross-entropy gives this simple error
        dh = np.outer(dz, p["w2"]) * (1 - h**2)  # (n, 16): back through tanh
        grads = {"W1": X.T @ dh, "b1": dh.sum(0), "w2": h.T @ dz, "b2": np.array([dz.sum()])}
        return np.concatenate([grads[name].ravel() for name, _ in _SHAPES])


def fine_tune(net: TinyNet, X: np.ndarray, y: np.ndarray, steps: int = 300, lr: float = 0.5, record=None):
    """Full-batch gradient descent from `net`, returning a NEW model (the start is untouched).

    With full-batch training one step is one pass over the data: one epoch.
    If `record` is given, it is called on the model after every step and
    the list of its results is returned alongside the model.
    """
    theta = net.theta.copy()
    history = []
    for _ in range(steps):
        theta = theta - lr * TinyNet(theta).gradient(X, y)
        if record is not None:
            history.append(record(TinyNet(theta)))
    tuned = TinyNet(theta)
    return (tuned, history) if record is not None else tuned


# The tasks. Each example is four numbers; the answer is yes (1) or no (0).
TASKS = {
    "general": "the base model's broad skill: all four numbers anywhere in [-3, 3]; yes when x1 + x3 > 0",
    "A": "first pair only, left region (x1 in [-3, -1]); yes when x2 > 0",
    "B": "first pair only, right region (x1 in [1, 3]); yes when x2 < 0 (A's rule, reversed)",
    "C": "second pair only (x3, x4 in [-2, 2]); yes when x3 + x4 > 0",
}
TRAIN_SEEDS = {"A": 1, "B": 2, "C": 3, "general": 4}
EVAL_SEEDS = {name: seed + 10 for name, seed in TRAIN_SEEDS.items()}


def make_task(name: str, n: int = 200, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Examples (X of shape (n, 4), y of 0/1) for one of `TASKS`. Default seed = its training set."""
    rng = np.random.default_rng(TRAIN_SEEDS[name] if seed is None else seed)
    X = np.zeros((n, N_IN))
    s = rng.uniform(-2, 2, n)
    if name == "A":
        X[:, 0], X[:, 1], y = rng.uniform(-3, -1, n), s, s > 0
    elif name == "B":
        X[:, 0], X[:, 1], y = rng.uniform(1, 3, n), s, s < 0
    elif name == "C":
        X[:, 2], X[:, 3] = rng.uniform(-2, 2, n), s
        y = X[:, 2] + s > 0
    elif name == "general":
        X = rng.uniform(-3, 3, (n, N_IN))
        y = X[:, 0] + X[:, 2] > 0
    else:
        raise ValueError(f"unknown task {name!r}; choose from {list(TASKS)}")
    return X, y.astype(float)


@lru_cache(maxsize=None)
def _eval_set(name: str) -> tuple[np.ndarray, np.ndarray]:
    # Held-out examples, drawn with their own seed and never trained on.
    return make_task(name, seed=EVAL_SEEDS[name])


def eval_accuracy(net: TinyNet, name: str) -> float:
    """Accuracy on a task's held-out set."""
    return net.accuracy(*_eval_set(name))


@lru_cache(maxsize=1)
def base_model() -> TinyNet:
    """The "pretrained" base: a random network trained on the general skill."""
    return fine_tune(TinyNet.random(seed=0), *make_task("general"), steps=300, lr=0.5)


@lru_cache(maxsize=None)
def fine_tuned(name: str, steps: int = 300, lr: float = 0.5) -> TinyNet:
    """The base model fine-tuned on one task's training set."""
    return fine_tune(base_model(), *make_task(name), steps=steps, lr=lr)


# ---------------------------------------------------------------------------
# 4. Catastrophic forgetting and its mitigations
# ---------------------------------------------------------------------------


def forgetting(acc_before: float, acc_after: float) -> float:
    """How much accuracy on the old task was lost: F = acc_before - acc_after."""
    return acc_before - acc_after


@lru_cache(maxsize=None)
def general_skill_run(lr: float = 0.5, steps: int = 300) -> dict[str, list[float]]:
    """Fine-tune the base on task A, recording after every step: accuracy on A,
    accuracy on the base's general skill, and distance travelled from the base."""
    base = base_model()

    def record(net: TinyNet):
        return eval_accuracy(net, "A"), eval_accuracy(net, "general"), float(np.linalg.norm(net.theta - base.theta))

    _, history = fine_tune(base, *make_task("A"), steps=steps, lr=lr, record=record)
    task, general, distance = (list(col) for col in zip(*history))
    return {"task": task, "general": general, "distance": distance}


def replay_mix(new: tuple, old: tuple, n_old: int) -> tuple[np.ndarray, np.ndarray]:
    """The new task's examples plus the first `n_old` examples of an old task."""
    (X_new, y_new), (X_old, y_old) = new, old
    return np.concatenate([X_new, X_old[:n_old]]), np.concatenate([y_new, y_old[:n_old]])


def mixed_loss(loss_new: float, n_new: int, loss_old: float, n_old: int) -> float:
    """Average loss over a mixed dataset: (n_new·L_new + n_old·L_old) / (n_new + n_old)."""
    return (n_new * loss_new + n_old * loss_old) / (n_new + n_old)


@lru_cache(maxsize=None)
def sequential_run(new: str, lr: float = 0.5, steps: int = 300, n_replay: int = 0) -> dict:
    """Start from the task-A fine-tune, then fine-tune on `new` (optionally replaying A).

    Returns accuracy on A before and after, accuracy on the new task after,
    and the (acc on A, acc on new) pair after every step.
    """
    start = fine_tuned("A")
    data = make_task(new)
    if n_replay:
        data = replay_mix(data, make_task("A"), n_replay)
    end, path = fine_tune(start, *data, steps=steps, lr=lr, record=lambda net: (eval_accuracy(net, "A"), eval_accuracy(net, new)))
    return {
        "A_before": eval_accuracy(start, "A"),
        "A_after": eval_accuracy(end, "A"),
        f"{new}_after": eval_accuracy(end, new),
        "path": path,
    }


# ---------------------------------------------------------------------------
# 5. Overfitting a small dataset
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def overfitting_run(n: int = 16, n_wrong: int = 3, epochs: int = 1500, lr: float = 0.5, seed: int = 2) -> dict:
    """Fine-tune on n task-A examples, `n_wrong` of them mislabelled, for many epochs.

    Records training loss and held-out (validation) loss after every epoch.
    """
    X, y = make_task("A", n=n, seed=seed)
    wrong = np.random.default_rng(seed).choice(n, n_wrong, replace=False)
    y = y.copy()
    y[wrong] = 1 - y[wrong]  # real datasets have some bad labels; a big enough model memorises them
    X_val, y_val = _eval_set("A")
    _, history = fine_tune(base_model(), X, y, steps=epochs, lr=lr, record=lambda net: (net.loss(X, y), net.loss(X_val, y_val)))
    train, val = (list(col) for col in zip(*history))
    return {"train": train, "val": val, "best_epoch": int(np.argmin(val))}


# ---------------------------------------------------------------------------
# 6. Model merging: weight averaging and task arithmetic
# ---------------------------------------------------------------------------


def task_vector(tuned: TinyNet, base: TinyNet) -> np.ndarray:
    """τ = θ_tuned - θ_base: everything the fine-tune changed, as one vector."""
    return tuned.theta - base.theta


def merge(base: TinyNet, task_vectors: list[np.ndarray], lam: float = 1.0) -> TinyNet:
    """θ_base + λ·Σ τ. With λ = 1/T this is exactly the average of the T fine-tunes."""
    return TinyNet(base.theta + lam * np.sum(task_vectors, axis=0))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """cos of the angle between two vectors: 1 same direction, 0 unrelated, -1 opposite."""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


LAMBDAS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)


@lru_cache(maxsize=None)
def merge_run(first: str, second: str, lambdas: tuple = LAMBDAS) -> dict:
    """Merge the fine-tunes on two tasks by task arithmetic at several λ.

    Returns the cosine between their task vectors and, for each λ, the
    merged model's accuracy on (first, second).
    """
    base = base_model()
    taus = [task_vector(fine_tuned(t), base) for t in (first, second)]
    by_lambda = {}
    for lam in lambdas:
        merged = merge(base, taus, lam)
        by_lambda[lam] = (eval_accuracy(merged, first), eval_accuracy(merged, second))
    return {"cosine": cosine(*taus), "by_lambda": by_lambda}


# ---------------------------------------------------------------------------
# 7. Figures (rendered into the HTML docs by `make figures`)
# ---------------------------------------------------------------------------


def figures() -> dict:
    """Plot this lesson's data. matplotlib is imported here, and only here,
    so the lesson itself needs nothing beyond NumPy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, RED, GREEN, AMBER, MUTED = "#2563eb", "#dc2626", "#059669", "#d97706", "#9ca3af"
    figs = {}

    # --- 1. Break-even: cumulative cost of prompting vs fine-tuning ---------
    c_prompt, c_tuned, fixed = per_request_cost(3000, 2.0), per_request_cost(300, 4.0), 600
    n = np.linspace(0, 250_000, 200)
    n_star = break_even_requests(fixed, c_prompt, c_tuned)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(n, c_prompt * n, color=RED, label=r"long prompt: \$0.0060 per request")
    ax.plot(n, fixed + c_tuned * n, color=BLUE, label=r"fine-tuned: \$600 once + \$0.0012 per request")
    ax.axvline(n_star, color=MUTED, ls="--")
    ax.text(n_star * 1.03, 150, f"break-even\n{n_star:,.0f} requests", color="#4b5563")
    ax.set_xlabel("requests served")
    ax.set_ylabel("total cost so far ($)")
    ax.set_title("When does a fine-tune pay for itself?")
    ax.legend(frameon=False, loc="upper left")
    figs["break_even"] = fig

    # --- 2. Margin of error vs held-out set size -----------------------------
    sizes = np.logspace(np.log10(25), np.log10(3200), 100)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(sizes, 100 * margin_of_error(0.8, sizes), color=BLUE)
    for size in (100, 400, 1600):
        m = 100 * margin_of_error(0.8, size)
        ax.plot(size, m, "o", color=BLUE)
        ax.annotate(f"n = {size}: ±{m:.1f}", (size, m), textcoords="offset points", xytext=(6, 6))
    ax.set_xscale("log")
    ax.set_xlabel("held-out examples n (log scale)")
    ax.set_ylabel("95% margin (percentage points)")
    ax.set_title("How precise is an 80% score? Four times the data halves the margin")
    figs["eval_margin"] = fig

    # --- 3. The general skill fades while task A is learned -----------------
    runs = {lr: general_skill_run(lr=lr, steps=300) for lr in (0.5, 0.02)}
    steps = np.arange(1, 301)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for lr, style in ((0.5, "-"), (0.02, "--")):
        a1.plot(steps, runs[lr]["task"], style, color=BLUE, label=f"task A, lr {lr}")
        a1.plot(steps, runs[lr]["general"], style, color=RED, label=f"general skill, lr {lr}")
        a2.plot(runs[lr]["distance"], runs[lr]["general"], style, color=RED, label=f"lr {lr}")
    a1.set_xscale("log")
    a1.set_xlabel("fine-tuning step (log scale)")
    a1.set_ylabel("held-out accuracy")
    a1.set_ylim(0.25, 1.02)
    a1.set_title("Learning A, forgetting the general skill")
    a1.legend(frameon=False, fontsize=8, loc="lower right")
    a2.set_xlabel("distance from the base  ‖θ − θ_base‖")
    a2.set_ylabel("general-skill accuracy")
    a2.set_title("Forgetting tracks how far the weights move")
    a2.legend(frameon=False)
    fig.tight_layout()
    figs["general_skill"] = fig

    # --- 4. Sequential fine-tuning: A then B, with and without replay -------
    plain, replay = sequential_run("B"), sequential_run("B", n_replay=10)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for run, style, tag in ((plain, "-", "B only"), (replay, "--", "B + 10 replayed A")):
        acc_a, acc_b = zip(*run["path"])
        ax.plot(steps, acc_a, style, color=BLUE, label=f"accuracy on A ({tag})")
        ax.plot(steps, acc_b, style, color=GREEN, label=f"accuracy on B ({tag})")
    ax.set_xscale("log")
    ax.set_xlabel("step of the second fine-tune (log scale)")
    ax.set_ylabel("held-out accuracy")
    ax.set_title("Fine-tuning on B after A: A collapses unless replayed")
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0.3, 0.68))
    figs["sequential"] = fig

    # --- 5. The trade-off: learning rate only walks the diagonal ------------
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [1, 0], color=MUTED, ls=":", label="A + B = 1")
    for lr, color in ((0.5, RED), (0.1, AMBER), (0.02, BLUE)):
        acc_a, acc_b = zip(*sequential_run("B", lr=lr)["path"])
        ax.plot(acc_b, acc_a, "-", color=color, label=f"B only, lr {lr}")
        ax.plot(acc_b[-1], acc_a[-1], "o", color=color)
    acc_a, acc_b = zip(*replay["path"])
    ax.plot(acc_b, acc_a, "--", color=GREEN, label="B + replay, lr 0.5")
    ax.plot(acc_b[-1], acc_a[-1], "o", color=GREEN)
    ax.set_xlabel("accuracy on the new task B")
    ax.set_ylabel("accuracy on the old task A")
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.03)
    ax.set_title("Slower is not safer; replay escapes the trade-off")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    figs["tradeoff"] = fig

    # --- 6. Overfitting 16 examples --------------------------------------------
    from primer.ml.regularization import early_stopping

    run = overfitting_run()
    epochs = np.arange(1, len(run["train"]) + 1)
    best, stop = early_stopping(run["val"], patience=20)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(epochs, run["train"], color=BLUE, label="training loss (16 examples, 3 mislabelled)")
    ax.plot(epochs, run["val"], color=RED, label="validation loss (200 held-out examples)")
    ax.axvline(best + 1, color=MUTED, ls="--")
    ax.axvline(stop + 1, color=MUTED, ls=":")
    ax.text((best + 1) * 1.08, 1.9, f"best epoch {best}", color="#4b5563")
    ax.set_xscale("log")
    ax.set_xlabel("epoch (log scale)")
    ax.set_ylabel("loss")
    ax.set_title("A small dataset: learning, then memorising")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    figs["overfitting"] = fig

    # --- 7 and 8. Merging by task arithmetic, a λ sweep -----------------------
    lams = tuple(float(x) for x in np.round(np.arange(0, 1.51, 0.125), 3))
    for key, second, color in (("merge_separate", "C", GREEN), ("merge_conflict", "B", AMBER)):
        merged = merge_run("A", second, lambdas=lams)
        acc_first, acc_second = zip(*merged["by_lambda"].values())
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.plot(lams, acc_first, "o-", color=BLUE, label="accuracy on A")
        ax.plot(lams, acc_second, "o-", color=color, label=f"accuracy on {second}")
        ax.axhline(0.5, color=MUTED, ls=":")
        ax.axvline(0.5, color=MUTED, ls="--")
        ax.text(0.52, 0.2, "plain average\n(λ = 1/2)", color="#4b5563")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("λ: strength of the summed task vectors")
        ax.set_ylabel("held-out accuracy of the merge")
        ax.set_title(f"base + λ(τ_A + τ_{second}):  cosine(τ_A, τ_{second}) = {merged['cosine']:.2f}")
        ax.legend(frameon=False, loc="lower right")
        figs[key] = fig

    return figs


# ---------------------------------------------------------------------------
# 8. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. Should you fine-tune? Put numbers on it")
    c_prompt, c_tuned = per_request_cost(3000, 2.0), per_request_cost(300, 4.0)
    table(
        ["option", "tokens", "$ per million", "$ per request"],
        [("long prompt", 3000, 2.0, c_prompt), ("fine-tuned", 300, 4.0, c_tuned)],
    )
    n_star = break_even_requests(600, c_prompt, c_tuned)
    say(f"A $600 fine-tune saves ${c_prompt - c_tuned:.4f} per request, so it pays off after {n_star:,.0f} requests.")
    takeaway("Build the eval first; prompt, then retrieve, and fine-tune only for behaviour a prompt can't pin down.")

    banner("2. Preparing the data")
    example = chat_example("Be brief.", "Capital of France?", "Paris.")
    table(["segment", "trained on?"], [(seg, "yes" if trained else "no") for seg, trained in render_chat(example)])
    say(f"Problems with an example that ends on the user's turn: {validate_chat({'messages': [{'role': 'user', 'content': 'Hi'}]})}")
    questions = ["How do I reset my password?", "how do I reset my password, please", "How do I change my email?"]
    table(
        ["pair", "Jaccard on word pairs"],
        [("password vs. its rewording", jaccard(questions[0], questions[1])), ("password vs. email", jaccard(questions[0], questions[2]))],
        floatfmt=".2f",
    )
    say(f"Deduplicating at 0.7 keeps indices {deduplicate(questions)}: the rewording is dropped.")
    table(
        ["held-out examples", "95% margin at 80%"],
        [(size, f"±{100 * margin_of_error(0.8, size):.1f} points") for size in (25, 100, 400, 1600)],
    )
    say(
        f"""
        With 10% wrong labels, a 90%-accurate model measures
        {measured_accuracy(0.9, 0.1):.0%}, and a perfect one only
        {measured_accuracy(1.0, 0.1):.0%}.
        """
    )
    takeaway("Freeze the held-out set first, keep its near-copies out of training, and check the labels.")

    banner("3. Catastrophic forgetting")
    say(f"The base model knows its general skill: {eval_accuracy(base_model(), 'general'):.2f} on held-out examples.")
    run = general_skill_run(lr=0.5, steps=300)
    slow = general_skill_run(lr=0.02, steps=300)
    table(
        ["fine-tune on A", "accuracy on A", "general skill", "distance from base"],
        [
            ("lr 0.5, after 5 steps", run["task"][4], run["general"][4], run["distance"][4]),
            ("lr 0.5, after 10 steps", run["task"][9], run["general"][9], run["distance"][9]),
            ("lr 0.5, after 300 steps", run["task"][-1], run["general"][-1], run["distance"][-1]),
            ("lr 0.02, after 300 steps", slow["task"][-1], slow["general"][-1], slow["distance"][-1]),
        ],
        floatfmt=".3f",
    )
    say("Task A is learned within a few steps; every step after that only moves the weights further and erodes the general skill.")
    rows = []
    for label, kwargs in (
        ("then B (contradicts A)", dict(new="B")),
        ("then B at lr 0.02", dict(new="B", lr=0.02)),
        ("then B, 10 steps only", dict(new="B", steps=10)),
        ("then B + 10 replayed A", dict(new="B", n_replay=10)),
        ("then C (separate inputs)", dict(new="C")),
    ):
        r = sequential_run(**kwargs)
        rows.append((label, r["A_before"], r["A_after"], r[f"{kwargs['new']}_after"], forgetting(r["A_before"], r["A_after"])))
    table(["after fine-tuning on A", "A before", "A after", "new task", "forgetting F"], rows, floatfmt=".3f")
    takeaway(
        "Forgetting grows with how far the weights move and how much the new task conflicts. "
        "Slowing down only walks the trade-off; replaying a little old data keeps both."
    )

    banner("4. Overfitting a small dataset")
    over = overfitting_run()
    best = over["best_epoch"]
    table(
        ["", "training loss", "validation loss"],
        [(f"best epoch ({best})", over["train"][best], over["val"][best]), ("last epoch", over["train"][-1], over["val"][-1])],
        floatfmt=".3f",
    )
    takeaway("On 16 examples the model learns the rule, then memorises the 3 wrong labels. Ship the best checkpoint.")

    banner("5. Merging models by task arithmetic")
    for second in ("C", "B"):
        merged = merge_run("A", second)
        say(f"A + {second}: cosine between task vectors = {merged['cosine']:.2f}")
        table(
            ["λ", "accuracy on A", f"accuracy on {second}"],
            [(lam, acc[0], acc[1]) for lam, acc in merged["by_lambda"].items()],
            floatfmt=".3f",
        )
    takeaway(
        "Nearly perpendicular task vectors add into a model that does both (0.97 and 0.97 at λ = 1); "
        "opposed ones cancel. Averaging is λ = 1/2 and dilutes each skill."
    )


if __name__ == "__main__":
    demo()
