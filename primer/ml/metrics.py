r"""
# Metrics: how you know whether a model is any good

Run: `python -m primer.ml.metrics`

New to the notation? `primer.notation` explains every symbol used here from
zero.

## The idea

A loss function is what the model *optimizes* during training; a metric is
what *you* use to decide whether it's working. Picking the wrong metric is
one of the most common ways a project looks great offline and fails in
production. This lesson builds every metric from scratch in three families:

1. **Classification:** precision, recall, F1, accuracy, the confusion
   matrix, ROC-AUC, precision-recall curves, and choosing a threshold by the
   *cost* of each kind of error.
2. **Retrieval:** recall@k, precision@k, MRR and nDCG, the metrics for search
   and retrieval-augmented generation (RAG).
3. **Generation:** BLEU and ROUGE-L (word overlap), why they fail on
   language-model output, and how to check an automated judge against people
   with Cohen's kappa.

## 1. Classification: precision, recall and the confusion matrix

**Everyday picture.** You're fishing for trout in a lake that also holds old
boots. **Precision** asks: of everything in your net, how much is trout?
**Recall** asks: of all the trout in the lake, how many did you catch? A tiny
net held in one good spot has high precision and low recall; dragging a huge
net across the whole lake has high recall and a lot of boots.

**Tiny worked example: fraud detection.** Of 100 transactions, 10 are fraud.
The model flags 8, and 6 of those are truly fraud. So 6 hits (true
positives, TP), 2 false alarms (false positives, FP), 4 misses (false
negatives, FN) and 88 correct passes (true negatives, TN).

| Metric    | Calculation                     | Result |
|-----------|---------------------------------|--------|
| Precision | 6 correct ÷ 8 flagged           | 75%    |
| Recall    | 6 found ÷ 10 actual fraud       | 60%    |
| F1        | 2 × 0.75 × 0.60 ÷ (0.75 + 0.60) | 67%    |
| Accuracy  | (6 + 88 correct) ÷ 100          | 94%    |

Accuracy looks great at 94% while the model misses 4 in 10 frauds.

```mermaid
flowchart LR
  X[Item] --> M[Model]
  M --> S[Score<br/>e.g. 0.73]
  S --> T{Score >= threshold?}
  T -->|yes| P[Flagged positive]
  T -->|no| N[Not flagged]
  P --> CM[Confusion matrix<br/>TP / FP / FN / TN]
  N --> CM
  L[True label] --> CM
  CM --> MET[Precision, recall,<br/>F1, accuracy]
```

**Reading it:** a classifier never outputs "fraud" directly; it outputs a
*score*, and a threshold that *you* choose turns scores into decisions.
Comparing each decision with the true label drops the item into one of four
cells of the confusion matrix, and every classification metric is just a
ratio of those four counts. Change the threshold and every metric changes,
which is why "what's the accuracy?" is incomplete without "at what
threshold?".

|                 | predicted positive | predicted negative |
|-----------------|--------------------|--------------------|
| actual positive | TP = 6             | FN = 4 (missed)    |
| actual negative | FP = 2 (false alarm) | TN = 88          |

$$
\text{precision} = \frac{TP}{TP + FP} \qquad
\text{recall} = \frac{TP}{TP + FN} \qquad
F_1 = \frac{2PR}{P + R} \qquad
\text{accuracy} = \frac{TP + TN}{N}
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| TP, FP, FN, TN | counts of hits, false alarms, misses and correct passes | whole numbers |
| $P$, $R$ | precision and recall | 0 … 1 |
| $F_1$ | the **harmonic mean** of P and R: an average that is dragged towards the smaller of the two | 0 … 1 |
| $N$ | total number of items, TP + FP + FN + TN | whole number |
| fraction bar | "divided by" | |

**In words:** precision is hits over everything flagged; recall is hits over
everything that should have been flagged; F1 is twice their product over
their sum; accuracy is everything right over everything.

**On the worked example:** P = 6/(6 + 2) = 0.75; R = 6/(6 + 4) = 0.60;
F1 = 2 × 0.75 × 0.60 / 1.35 = 0.667; accuracy = (6 + 88)/100 = 0.94. The
harmonic mean punishes imbalance: P = 1.0 with R = 0.1 gives F1 = 0.18, not
the ordinary average of 0.55.

**In Python:**

```python
TP, FP, FN, TN = 6, 2, 4, 88
N = TP + FP + FN + TN
# precision, recall
P, R = TP / (TP + FP), TP / (TP + FN)
P, R  # → (0.75, 0.6)
# F1, accuracy
round(2 * P * R / (P + R), 3), (TP + TN) / N  # → (0.667, 0.94)
# F1 when P = 1.0 and R = 0.1
round(2 * 1.0 * 0.1 / (1.0 + 0.1), 2)  # → 0.18
```

**In code:** `confusion` sorts labels and decisions into a `Confusion`,
whose `Confusion.precision`, `Confusion.recall`, `Confusion.f1` and
`Confusion.accuracy` are the four formulas. `fraud_example` rebuilds the
100 transactions above, and `flag_nothing_trap` builds the model below that
never flags anything.

**Why it matters in practice: the accuracy trap.** If 1% of transactions are
fraud, a model that flags *nothing* is 99% accurate and completely useless
(recall 0). On imbalanced data, never lead with accuracy.

## 2. ROC-AUC: how well does the model *rank*?

**Everyday picture.** A smoke alarm has a sensitivity dial. Turn it up and
it catches every fire but also shrieks at toast; turn it down and it stays
quiet but might miss a real fire. The ROC curve draws *every* dial setting
at once, and AUC scores the alarm across all of them.

**Tiny worked example.** Two frauds scored 0.9 and 0.4, two legitimate
transactions scored 0.6 and 0.1. Compare every (fraud, legitimate) pair:
0.9 > 0.6 ✓, 0.9 > 0.1 ✓, 0.4 > 0.6 ✗, 0.4 > 0.1 ✓. Three of four pairs are
ranked correctly, so **AUC = 0.75**.

```mermaid
flowchart TD
  A[Sort items by score, highest first] --> B[Start with threshold above every score<br/>nothing flagged: point 0,0]
  B --> C[Lower the threshold past the next distinct score]
  C --> D{Which items became flagged?}
  D -->|a positive| E[Step up: TPR rises]
  D -->|a negative| F[Step right: FPR rises]
  D -->|a tie of both| G[Diagonal step: half credit]
  E --> H{Everything flagged?}
  F --> H
  G --> H
  H -->|no| C
  H -->|yes| I[End at point 1,1<br/>AUC = area under the path]
```

**Reading it:** the ROC curve is a walk. Starting from "flag nothing"
(bottom-left) you lower the threshold one score at a time; each positive you
pick up moves you up, each negative moves you right. A perfect ranker goes
straight up then straight right (area 1.0); a random one wanders along the
diagonal (area 0.5). Because the walk moves *up* exactly when a positive
outranks the remaining negatives, the area counts correctly ordered
(positive, negative) pairs, which is why AUC equals the pairwise win rate.
The code computes AUC both ways (trapezoids under the curve, and counting
pairs) and they agree exactly.

$$
\text{TPR} = \frac{TP}{TP + FN} \qquad
\text{FPR} = \frac{FP}{FP + TN} \qquad
\text{AUC} = \frac{1}{|P|\,|N|}\sum_{p \in P}\sum_{n \in N}\Big([s_p > s_n] + \tfrac{1}{2}[s_p = s_n]\Big)
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| TPR | true-positive rate: recall, the share of positives flagged | 0 … 1 |
| FPR | false-positive rate: the share of negatives wrongly flagged | 0 … 1 |
| $P$, $N$ (in the AUC sum) | the set of positive items and the set of negative items | sets |
| $\lvert P \rvert$, $\lvert N \rvert$ | how many items each set holds | whole numbers |
| $s_p$, $s_n$ | the model's score for positive p and negative n | real |
| $[\ldots]$ | 1 if the statement inside is true, else 0 | 0 or 1 |
| $\sum_{p \in P}\sum_{n \in N}$ | add over every (positive, negative) pair | |

**In words:** AUC is the share of (positive, negative) pairs in which the
positive gets the higher score, counting ties as half.

**On the worked example:** 2 positives × 2 negatives = 4 pairs; 3 are
ordered correctly; AUC = 3/4 = 0.75. One point on the curve: at threshold
0.5 the model flags the 0.9 fraud and the 0.6 legitimate transaction, so
TPR = 1/(1 + 1) = 0.5 and FPR = 1/(1 + 1) = 0.5.

**In Python:**

```python
# scores of the positives (frauds)
s_P = [0.9, 0.4]
# scores of the negatives (legitimate)
s_N = [0.6, 0.1]
t = 0.5
TP, FN = sum(s >= t for s in s_P), sum(s < t for s in s_P)
FP, TN = sum(s >= t for s in s_N), sum(s < t for s in s_N)
# TPR, FPR
TP / (TP + FN), FP / (FP + TN)  # → (0.5, 0.5)
pairs = sum((s_p > s_n) + 0.5 * (s_p == s_n) for s_p in s_P for s_n in s_N)
# AUC: the share of pairs ranked correctly
pairs / (len(s_P) * len(s_N))  # → 0.75
```

![ROC curve of the synthetic classifier with its AUC](figures/primer.ml.metrics.roc.svg)

**Reading it:** the solid line is the ROC curve of `synthetic_scores()`
(positives shifted 1.5 standard deviations above negatives); the dashed
diagonal is a coin flip. The shaded area is the AUC, about 0.89: pick one
fraud and one legitimate transaction at random and the model scores the
fraud higher 89% of the time. Where the curve bends is where a sensible
threshold lives.

![Precision-recall curve of the same classifier](figures/primer.ml.metrics.pr.svg)

**Reading it:** the same scores, viewed as precision (y) against recall (x).
Moving right means lowering the threshold: you find more of the positives,
but precision falls as false alarms pile in. The dotted line is the base
rate (10% positives), i.e. what flagging at random achieves. On rare-event
problems this plot tells the truth that ROC's tiny false-positive *rates*
hide: at 80% recall only about a third of the flags are real.

**In code:** `roc_curve` takes the walk, `roc_auc` measures the area under
it with `auc_trapezoid`, and `roc_auc_rank` counts correctly ordered pairs
instead; `Confusion.fpr` is FPR, and `pr_curve` gives the precision-recall
points.

**Why it matters in practice.** AUC compares models independently of any
threshold. On heavily imbalanced data, look at the precision-recall curve
too, because FPR stays tiny even when false alarms swamp the true positives.

## 3. Picking the threshold: price your errors

**Everyday picture.** A hospital screening test and a spam filter want
opposite things. Missing a disease is terrible, so the screen flags
generously; losing an important email is annoying, so the spam filter flags
cautiously. Same maths, different prices.

**Tiny worked example.** With the four scores above (frauds 0.9 and 0.4,
legitimate 0.6 and 0.1): if a miss costs 10 and a false alarm costs 1, the
cheapest rule is "flag at 0.4 or above", catching both frauds and wrongly
flagging one legitimate transaction (total cost **1**). If a false alarm
costs 10 and a miss 1, the cheapest rule is "flag only 0.9", missing one
fraud but raising no false alarms (total cost **1**).

$$
\text{cost}(t) = c_{\text{FN}} \cdot FN(t) + c_{\text{FP}} \cdot FP(t) \qquad t^* = \arg\min_t \text{cost}(t)
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| $t$ | the threshold: flag items scoring at least t | real |
| $FN(t)$, $FP(t)$ | misses and false alarms at that threshold | whole numbers |
| $c_{\text{FN}}$, $c_{\text{FP}}$ | the price of one miss and of one false alarm | ≥ 0 |
| $\arg\min_t$ | "the value of t that makes this smallest" | |
| $t^*$ | the best threshold | real |

**In words:** the total cost at a threshold is misses times their price
plus false alarms times theirs; pick the threshold where that total is
lowest.

**On the worked example:** c_FN = 10, c_FP = 1: at t = 0.4, FN = 0 and
FP = 1, cost 1, the minimum. The other thresholds cost 10 (t = 0.9, one
miss), 11 (t = 0.6, one miss and one false alarm) and 2 (t = 0.1, two false
alarms).

**In Python:**

```python
frauds, legit = [0.9, 0.4], [0.6, 0.1]
c_FN, c_FP = 10, 1
def cost(t):
    # frauds below t are missed
    FN = sum(s < t for s in frauds)
    # legitimate ones at or above t are false alarms
    FP = sum(s >= t for s in legit)
    return c_FN * FN + c_FP * FP
[cost(t) for t in (0.9, 0.6, 0.4, 0.1)]  # → [10, 11, 1, 2]
# t* = argmin_t cost(t)
min((0.9, 0.6, 0.4, 0.1), key=cost)  # → 0.4
```

![Total error cost as the threshold sweeps, for three cost settings](figures/primer.ml.metrics.cost_vs_threshold.svg)

**Reading it:** each line is the total cost (misses × their price + false
alarms × their price) at every threshold, and each dot marks that line's
minimum. When a miss costs 10× a false alarm, the best threshold slides left
(flag more, higher recall); when a false alarm costs 10×, it slides right
(flag less, higher precision). The model is the same in all three lines;
only the business decides where to cut.

**In code:** `best_threshold_by_cost` tries every distinct score as t
(plus "flag nothing") and keeps the cheapest.

## 4. Retrieval metrics: judging a search result list

**Everyday picture.** You ask a librarian for books on a topic and get a
stack of five. Did the stack include the books that matter (recall)? How
much of the stack is useful (precision)? Is a good book on top, or do you
dig for it (reciprocal rank)? Are the *best* books nearest the top (nDCG)?

**Tiny worked example.** One query. The system returns d7, d3, d9, d1, d4 in
that order. The relevant documents are d3 (very relevant, grade 3), d4
(grade 2) and d8 (grade 1, never returned).

* recall@5 = 2 found of 3 relevant = **0.667**
* precision@5 = 2 relevant of 5 returned = **0.4**
* reciprocal rank = first relevant result at rank 2, so 1/2 = **0.5**
* nDCG@5 = **0.560** (worked below)

```mermaid
flowchart LR
  G[Golden set<br/>query + relevant doc ids] --> R[Retriever]
  R --> K[Ranked top-k list]
  K --> RK[recall@k<br/>did we find them?]
  K --> PK[precision@k<br/>how much noise?]
  K --> RR[MRR<br/>how high is the first hit?]
  K --> ND[nDCG@k<br/>are the best ones on top?]
  G --> RK & PK & RR & ND
```

**Reading it:** retrieval is evaluated *separately* from generation. A
golden set pairs real queries with the ids of the documents that answer
them; the retriever produces a ranked list for each query; four metrics ask
four different questions of the same list. Averaging them over the golden
set gives numbers you can track every time you change chunking, the
embedding model or the reranker.

$$
\text{recall@}k = \frac{|\text{relevant} \cap \text{top-}k|}{|\text{relevant}|} \qquad
\text{MRR} = \frac{1}{|Q|}\sum_{q \in Q} \frac{1}{\text{rank}_q} \qquad
\text{DCG@}k = \sum_{i=1}^{k} \frac{\text{rel}_i}{\log_2(i + 1)} \qquad
\text{nDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| $k$ | how many top results you look at | e.g. 5 |
| $\cap$ | "in both": documents that are relevant *and* in the top k | set |
| $\lvert\cdot\rvert$ | how many items a set holds | |
| $Q$, $q$ | the set of test queries, and one query | |
| $\text{rank}_q$ | position (1 = top) of the first relevant result for query q | 1, 2, … |
| $\text{rel}_i$ | the relevance grade of the result at position i (0 if irrelevant) | e.g. 0 … 3 |
| $\log_2(i + 1)$ | logarithm base 2: how many times you halve i + 1 to reach 1. It grows slowly, so it is a gentle position discount | 1 at rank 1, 1.58 at rank 2 |
| IDCG | "ideal DCG": the DCG of the same grades sorted best-first, so nDCG tops out at 1 | |

**In words:** recall@k is the share of relevant documents found in the top
k; MRR averages one-over-the-rank of the first hit; DCG adds each result's
grade, discounted by the log of its position; nDCG divides by the best
possible DCG.

**On the worked example:** recall@5 = 2/3 = 0.667; with one query whose
first hit is at rank 2, MRR = 1/2 = 0.5. DCG = 3/log₂(3) + 2/log₂(6) =
1.8928 + 0.7737 = 2.6665 (d3 at rank 2, d4 at rank 5). Ideal order d3, d4,
d8: IDCG = 3/1 + 2/1.585 + 1/2 = 4.7619. nDCG = 2.6665/4.7619 = 0.560.

**In Python:**

```python
import math
ranked = ["d7", "d3", "d9", "d1", "d4"]
# relevance grades; missing means 0
rel = {"d3": 3, "d4": 2, "d8": 1}
k = 5
# recall@k
round(len(set(rel) & set(ranked[:k])) / len(rel), 3)  # → 0.667
rank_q = next(i for i, doc in enumerate(ranked, start=1) if doc in rel)
# MRR over a single query
1 / rank_q  # → 0.5
DCG = sum(rel.get(doc, 0) / math.log2(i + 1) for i, doc in enumerate(ranked[:k], start=1))
# the same grades, best first
best = sorted(rel.values(), reverse=True)[:k]
IDCG = sum(g / math.log2(i + 1) for i, g in enumerate(best, start=1))
print(f"{DCG:.4f} {IDCG:.4f} {DCG / IDCG:.3f}")  # → 2.6665 4.7619 0.560
```

![How much each rank position counts in DCG](figures/primer.ml.metrics.ndcg_discount.svg)

**Reading it:** the bars are the weight 1/log2(rank + 1) that DCG gives a
result at each position. Rank 1 counts fully, rank 2 counts 63%, rank 10
only 29%. That gentle logarithmic decay says "position matters, but a good
result at rank 5 is still worth a lot", which matches how people and
language models actually use a result list.

**In code:** `recall_at_k` and `precision_at_k` score one ranked list,
`reciprocal_rank` finds its first hit and `mean_reciprocal_rank` averages
that over queries; `dcg_at_k` adds the discounted grades and `ndcg_at_k`
divides by the ideal ordering's DCG.

**Why it matters in practice.** For RAG, recall@k (with k = the number of
chunks you pass to the model) usually matters most: the model can't use a
passage that wasn't retrieved, and no prompt fixes that.

## 5. Generation metrics: why word overlap misleads

**Everyday picture.** Grading essays by counting how many words each one
shares with the answer key. A student who copies the key's phrasing but gets
the key fact wrong scores well; a student who explains it correctly in their
own words scores badly.

**Tiny worked example.** Reference: "The meeting was moved to Friday because
the manager is sick."

| candidate | BLEU | ROUGE-L |
|---|---|---|
| "Since the boss is ill, the team rescheduled the meeting for Friday." (correct) | 0.16 | 0.26 |
| "The meeting was moved to *Monday* because the manager is sick." (wrong) | 0.73 | 0.91 |

**BLEU** (from machine translation) multiplies together how many of the
candidate's 1-, 2-, 3- and 4-word sequences appear in the reference, with a
penalty for being too short. **ROUGE-L** (from summarisation) measures the
longest sequence of words the two share in the same order, gaps allowed.

$$
\text{BLEU} = \text{BP}\cdot\exp\!\Big(\tfrac{1}{4}\sum_{n=1}^{4}\ln p_n\Big) \qquad
\text{ROUGE-L} = \frac{2\,P_{\text{LCS}}\,R_{\text{LCS}}}{P_{\text{LCS}} + R_{\text{LCS}}}
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| $p_n$ | share of the candidate's n-word sequences also found in the reference, each counted at most as often as the reference has it | 0 … 1 |
| $\ln$, $\exp$ | natural log and its inverse; exp of the average log is the *geometric mean* of the four $p_n$ | |
| BP | brevity penalty: 1 if the candidate is longer than the reference, else e^(1 − ref length / candidate length) | 0 … 1 |
| LCS | longest common subsequence: the longest run of words appearing in both texts in the same order | |
| $P_{\text{LCS}}$, $R_{\text{LCS}}$ | LCS length over candidate length, and over reference length | 0 … 1 |

**In words:** BLEU is the geometric mean of the n-gram match rates, scaled
down for short answers; ROUGE-L is the F1 of the longest shared in-order
word sequence.

**On a hand example:** candidate "a b c d", reference "a c d e": the LCS is
"a c d", 3 words, so P = 3/4, R = 3/4, ROUGE-L = 0.75. And "the the the the"
against "the cat is here" scores unigram precision 1/4: "the" is credited
only as often as the reference contains it.

**With the numbers:** the *Monday* answer and the reference are both 11
words (full stop and capitals dropped), so BP = 1. It matches 10 of its 11
words, 8 of its 10 word pairs, 6 of its 9 triples and 4 of its 8 four-word
runs. `bleu` adds 1 to the top and bottom for n ≥ 2 (smoothing, so one
missing four-word run can't zero the score), giving p = 10/11, 9/11, 7/10,
5/9, and BLEU = exp(¼(ln 0.909 + ln 0.818 + ln 0.7 + ln 0.556)) = **0.73**.
Its longest shared in-order run is 10 words, so ROUGE-L = 2 · (10/11) ·
(10/11) / (20/11) = **0.91**.

**In Python:**

```python
import math
ref = "the meeting was moved to friday because the manager is sick".split()
cand = "the meeting was moved to monday because the manager is sick".split()
# every run of n words, in order
def grams(words, n):
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
def p(n):
    c, r = grams(cand, n), grams(ref, n)
    # clipped: at most as often as ref has it
    hits = sum(min(c.count(g), r.count(g)) for g in set(c))
    # add-one smoothing for n ≥ 2
    s = 1 if n > 1 else 0
    return (hits + s) / (len(c) + s)
[round(p(n), 3) for n in range(1, 5)]  # → [0.909, 0.818, 0.7, 0.556]
BP = 1.0 if len(cand) > len(ref) else math.exp(1 - len(ref) / len(cand))
# BLEU
round(BP * math.exp(sum(math.log(p(n)) for n in range(1, 5)) / 4), 2)  # → 0.73
# every word but "monday", in order
P_LCS, R_LCS = 10 / len(cand), 10 / len(ref)
# ROUGE-L
round(2 * P_LCS * R_LCS / (P_LCS + R_LCS), 2)  # → 0.91
# the hand example: LCS "a c d"
P_LCS, R_LCS = 3 / 4, 3 / 4
2 * P_LCS * R_LCS / (P_LCS + R_LCS)  # → 0.75
```

![BLEU and ROUGE-L for a paraphrase, a wrong answer and an exact copy](figures/primer.ml.metrics.overlap_scores.svg)

**Reading it:** three candidate answers to the same reference ("the meeting
was moved to Friday because the manager is sick"). The correct paraphrase
(left) scores lowest on both metrics; the answer that says *Monday* (middle)
scores nearly as high as an exact copy (right). Overlap metrics reward
wording, not truth.

**In code:** `bleu` clips each n-gram count, takes the geometric mean and
applies the brevity penalty; `rouge_l` finds the longest common subsequence
and returns its F1.

**Why it matters in practice.** Teams grade open-ended output with an **LLM
as a judge**, a model following an explicit rubric, and embedding-based
scores like BERTScore do somewhat better than overlap. But a judge must earn
trust first (next section).

## 6. Calibrating a judge: Cohen's kappa

**Everyday picture.** Two teachers grade the same 20 essays pass/fail and
agree on 18. Impressive? Not if 18 of the essays were obvious passes: two
teachers who stamped "pass" on everything without reading would also agree
on 18. **Cohen's kappa** subtracts the agreement you'd expect from luck.

**Tiny worked example.** Human labels: 18 pass, 2 fail. A lazy judge says
"pass" to everything. Raw agreement is 90%, and chance agreement (both say
pass at their own rates) is also 0.9 × 1.0 + 0.1 × 0 = 90%, so **kappa = 0**.
A second example: labels (y, y, n, n) vs. (y, n, n, n) agree 3/4 of the
time; chance predicts 0.5 × 0.25 + 0.5 × 0.75 = 0.5; **kappa = 0.5**.

$$
\kappa = \frac{p_o - p_e}{1 - p_e} \qquad p_e = \sum_{\ell} p_A(\ell)\, p_B(\ell)
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| $\kappa$ | kappa: agreement beyond chance | ≤ 1; 0 = chance, 1 = perfect |
| $p_o$ | observed agreement: share of items both raters labelled the same | 0 … 1 |
| $p_e$ | agreement expected by chance | 0 … 1 |
| $\ell$ | one possible label (pass, fail…) | |
| $p_A(\ell)$, $p_B(\ell)$ | how often rater A (and rater B) uses label ℓ | 0 … 1 |

**In words:** kappa is how far agreement beats chance, as a share of the most
it could beat chance by.

**On the worked example:** p_o = 0.75, p_e = 0.5, κ = (0.75 − 0.5)/(1 − 0.5)
= 0.5.

**In Python:**

```python
# rater A's labels
A = ["y", "y", "n", "n"]
# rater B's labels
B = ["y", "n", "n", "n"]
p_o = sum(a == b for a, b in zip(A, B)) / len(A)
# Σ_ℓ p_A(ℓ) p_B(ℓ)
p_e = sum((A.count(ell) / len(A)) * (B.count(ell) / len(B)) for ell in ("y", "n"))
p_o, p_e  # → (0.75, 0.5)
# κ
(p_o - p_e) / (1 - p_e)  # → 0.5
```

```mermaid
flowchart LR
  S[Sample of real outputs] --> H[Humans label<br/>pass / fail]
  S --> J[LLM judge + rubric<br/>labels pass / fail]
  H --> K[Cohen's kappa]
  J --> K
  K -->|kappa high enough| U[Use the judge at scale]
  K -->|too low| F[Fix rubric, examples<br/>or judge model]
  F --> J
```

**Reading it:** the judge earns trust the same way a new human reviewer
would: grade a sample that people have already graded and compare. Kappa,
not raw agreement, is the gate, because on a sample that is mostly "pass" a
lazy judge agrees by accident. Only once the judge clears the bar do you let
it grade thousands of outputs, and you repeat the check whenever the judge
model or rubric changes.

**In code:** `cohens_kappa` computes p_o and p_e from two lists of labels
and returns κ.

**Why it matters in practice.** Kappa above about 0.6 is usually considered
substantial agreement. Re-check it whenever the judge model or rubric
changes.

## In 20 seconds
- Precision: of what I flagged, how much was right. Recall: of what
  mattered, how much I found. F1 balances them.
- Accuracy lies on imbalanced data. Flagging nothing on 1% fraud scores 99%.
- ROC-AUC is the probability a random positive outranks a random negative.
- Choose the threshold by what each kind of error costs.
- For RAG, measure recall@k first. What isn't retrieved can't be used.
- BLEU/ROUGE punish correct paraphrases; use an LLM judge and calibrate it
  against humans with Cohen's kappa.

## Self-test questions

**Q: A fraud model has 94% accuracy. Is it good?**
A: You can't tell from accuracy. If fraud is 10% of traffic, check recall
and precision. In the worked example recall is only 60%, so 4 in 10 frauds
slip through.

**Q: When do you optimize for precision vs. recall?**
A: By the cost of each error. If a miss is expensive (fraud, cancer
screening, a relevant legal document), favor recall. If a false alarm is
expensive (blocking a good customer, paging an engineer at 3 a.m.), favor
precision. Set the threshold to minimize expected cost.

**Q: What does an AUC of 0.8 mean?**
A: A randomly chosen positive gets a higher score than a randomly chosen
negative 80% of the time. It measures ranking quality across all
thresholds; 0.5 is random.

**Q: Which retrieval metric matters most for RAG, and why?**
A: Recall@k, where k is the number of chunks you pass to the model. If the
right passage isn't in the top k, no prompt engineering can recover the
answer. Add MRR or nDCG when position within the context matters.

**Q: Why not grade LLM answers with BLEU or ROUGE?**
A: They measure surface overlap with one reference. A correct paraphrase
scores low and a wrong answer that reuses the reference's words scores
high. Use code-based checks where possible and a rubric-driven LLM judge
otherwise, validated against human labels.

**Q: Your LLM judge agrees with humans 90% of the time. Good enough?**
A: Not necessarily. If 90% of answers are "pass", a judge that always says
"pass" also agrees 90%. Compute Cohen's kappa to correct for chance
agreement.

## The papers behind this lesson

- Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*
  (2023): https://arxiv.org/abs/2306.05685. Measured how well strong
  language models agree with human preferences when used as judges, and
  catalogued their biases (position, verbosity, self-preference).
  [annotated companion](../../papers/llm-as-judge.html)
- Papineni et al., *BLEU: a Method for Automatic Evaluation of Machine
  Translation* (2002): https://aclanthology.org/P02-1040/. Introduced
  clipped n-gram precision with a brevity penalty.
- Lin, *ROUGE: A Package for Automatic Evaluation of Summaries* (2004):
  https://aclanthology.org/W04-1013/. Introduced recall-oriented overlap
  measures, including the LCS-based ROUGE-L.
- Zhang et al., *BERTScore: Evaluating Text Generation with BERT* (2019):
  https://arxiv.org/abs/1904.09675. Compared candidate and reference by
  embedding similarity rather than exact word overlap.

## Further reading
- scikit-learn, *Metrics and scoring*: https://scikit-learn.org/stable/modules/model_evaluation.html
- Receiver operating characteristic (Wikipedia): https://en.wikipedia.org/wiki/Receiver_operating_characteristic
- Discounted cumulative gain (Wikipedia): https://en.wikipedia.org/wiki/Discounted_cumulative_gain
- Cohen's kappa (Wikipedia): https://en.wikipedia.org/wiki/Cohen%27s_kappa
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence

import numpy as np

from primer._show import banner, say, table, takeaway

# ---------------------------------------------------------------------------
# 1. Classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Confusion:
    """Counts of the four outcomes of a binary classifier."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float:
        # Undefined when nothing was flagged; by convention report 0.
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        # Harmonic mean: dominated by the smaller of the two.
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n

    @property
    def fpr(self) -> float:
        """False-positive rate: share of negatives we wrongly flagged."""
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) else 0.0


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> Confusion:
    """Count TP/FP/FN/TN for binary labels (1 = positive)."""
    t = np.asarray(y_true).astype(bool)
    p = np.asarray(y_pred).astype(bool)
    return Confusion(
        tp=int(np.sum(t & p)),
        fp=int(np.sum(~t & p)),
        fn=int(np.sum(t & ~p)),
        tn=int(np.sum(~t & ~p)),
    )


def fraud_example() -> tuple[np.ndarray, np.ndarray]:
    """Worked example: 100 transactions, 10 fraud, 8 flagged, 6 correctly.

    Returns (y_true, y_pred) arrays that produce TP=6, FP=2, FN=4, TN=88.
    """
    y_true = np.array([1] * 10 + [0] * 90)
    y_pred = np.zeros(100, dtype=int)
    y_pred[:6] = 1  # 6 of the 10 frauds caught
    y_pred[10:12] = 1  # 2 legitimate transactions wrongly flagged
    return y_true, y_pred


def flag_nothing_trap(n: int = 10_000, fraud_rate: float = 0.01) -> Confusion:
    """A 'model' that never flags anything, on 1% fraud: 99% accurate, 0% recall."""
    n_fraud = int(n * fraud_rate)
    y_true = np.array([1] * n_fraud + [0] * (n - n_fraud))
    return confusion(y_true, np.zeros(n, dtype=int))


# ---------------------------------------------------------------------------
# 2. ROC and precision-recall curves, AUC two ways
# ---------------------------------------------------------------------------


def roc_curve(y_true: Sequence[int], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROC curve points (fpr, tpr, thresholds), sweeping the threshold high -> low.

    We place one threshold at each *distinct* score. Items with tied scores
    enter the "flagged" set together, which produces a diagonal segment.
    That diagonal is exactly how ties earn half credit in the AUC.
    """
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=float)
    P, N = y.sum(), (~y).sum()
    thresholds = np.unique(s)[::-1]  # distinct scores, descending
    tpr, fpr = [0.0], [0.0]  # threshold = +inf: nothing flagged
    for t in thresholds:
        flagged = s >= t
        tpr.append((flagged & y).sum() / P)
        fpr.append((flagged & ~y).sum() / N)
    return np.array(fpr), np.array(tpr), np.concatenate([[np.inf], thresholds])


def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    """Area under a piecewise-linear curve by the trapezoid rule."""
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))


def roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """ROC-AUC as the area under the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return auc_trapezoid(fpr, tpr)


def roc_auc_rank(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """ROC-AUC as P(random positive scores above random negative), ties = 1/2.

    This is the Mann-Whitney U statistic divided by (#pos × #neg). O(P·N)
    pairwise version for clarity; production code uses ranks for O(n log n).
    """
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=float)
    pos, neg = s[y], s[~y]
    diff = pos[:, None] - neg[None, :]  # every positive vs. every negative
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def pr_curve(y_true: Sequence[int], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision and recall at each distinct-score threshold (descending)."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=float)
    thresholds = np.unique(s)[::-1]
    precision, recall = [], []
    for t in thresholds:
        c = confusion(y, s >= t)
        precision.append(c.precision)
        recall.append(c.recall)
    return np.array(precision), np.array(recall), thresholds


def best_threshold_by_cost(
    y_true: Sequence[int], scores: Sequence[float], cost_fn: float, cost_fp: float
) -> tuple[float, float, Confusion]:
    """Pick the threshold minimizing cost_fn·FN + cost_fp·FP.

    Returns (threshold, total_cost, confusion_at_threshold). A high `cost_fn`
    (misses are expensive) pushes the threshold down (flag more, higher
    recall); a high `cost_fp` pushes it up (flag less, higher precision).
    """
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    candidates = np.concatenate([np.unique(s), [np.inf]])  # inf = flag nothing
    best = None
    for t in candidates:
        c = confusion(y, s >= t)
        cost = cost_fn * c.fn + cost_fp * c.fp
        if best is None or cost < best[1]:
            best = (float(t), float(cost), c)
    assert best is not None
    return best


def synthetic_scores(n_pos: int = 50, n_neg: int = 450, separation: float = 1.5, seed: int = 0):
    """Classifier scores where positives are shifted up by `separation` std devs."""
    rng = np.random.default_rng(seed)
    y = np.array([1] * n_pos + [0] * n_neg)
    s = np.concatenate([rng.normal(separation, 1, n_pos), rng.normal(0, 1, n_neg)])
    return y, s


# ---------------------------------------------------------------------------
# 3. Retrieval metrics
# ---------------------------------------------------------------------------


def recall_at_k(ranked: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    """Share of the relevant items that appear in the top k."""
    rel = set(relevant)
    return len(rel & set(ranked[:k])) / len(rel) if rel else 0.0


def precision_at_k(ranked: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    """Share of the top k that are relevant (divides by k, even if fewer results)."""
    rel = set(relevant)
    return sum(1 for d in ranked[:k] if d in rel) / k


def reciprocal_rank(ranked: Sequence[Hashable], relevant: Iterable[Hashable]) -> float:
    """1 / (1-based rank of the first relevant result), 0 if none is found."""
    rel = set(relevant)
    for i, d in enumerate(ranked, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(runs: Sequence[tuple[Sequence[Hashable], Iterable[Hashable]]]) -> float:
    """MRR over (ranked, relevant) pairs, one per query."""
    return float(np.mean([reciprocal_rank(r, rel) for r, rel in runs]))


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """DCG of relevance grades listed in ranked order: sum rel_i / log2(i + 1).

    Uses linear gain (rel_i), which matches scikit-learn's `ndcg_score`. Some
    search teams use exponential gain (2^rel - 1) to reward "perfect"
    results more strongly; the ranking logic is identical.
    """
    g = np.asarray(gains, dtype=float)[:k]
    positions = np.arange(1, len(g) + 1)
    return float(np.sum(g / np.log2(positions + 1)))


def ndcg_at_k(ranked: Sequence[Hashable], grades: dict[Hashable, float], k: int) -> float:
    """nDCG@k: DCG of this ranking divided by the DCG of the ideal ranking.

    Args:
        ranked: result ids in the order the system returned them.
        grades: graded relevance per id (missing ids count as 0).
    """
    gains = [grades.get(d, 0.0) for d in ranked]
    ideal = sorted(grades.values(), reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg_at_k(gains, k) / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# 4. Generation metrics: BLEU, ROUGE-L, and judge calibration
# ---------------------------------------------------------------------------


def _words(text: str) -> list[str]:
    return [w.strip(".,!?;:'\"").lower() for w in text.split() if w.strip(".,!?;:'\"")]


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu(candidate: str, reference: str, max_n: int = 4, smooth: bool = True) -> float:
    """Sentence-level BLEU against one reference.

    BLEU = brevity_penalty × geometric mean of modified n-gram precisions
    (n = 1..max_n). "Modified" means each candidate n-gram is credited at
    most as many times as it occurs in the reference, so "the the the the"
    can't game it. The brevity penalty stops very short candidates from
    getting high precision by saying almost nothing.

    `smooth=True` adds 1 to numerator and denominator for n > 1 (Lin & Och
    "add-one" smoothing), so one missing 4-gram doesn't zero the score.
    """
    c, r = _words(candidate), _words(reference)
    if not c:
        return 0.0
    log_precisions = []
    for n in range(1, max_n + 1):
        cand, ref = _ngrams(c, n), _ngrams(r, n)
        overlap = sum(min(cnt, ref[g]) for g, cnt in cand.items())
        total = max(sum(cand.values()), 0)
        if smooth and n > 1:
            overlap, total = overlap + 1, total + 1
        if overlap == 0 or total == 0:
            return 0.0
        log_precisions.append(math.log(overlap / total))
    bp = 1.0 if len(c) > len(r) else math.exp(1 - len(r) / len(c))
    return bp * math.exp(sum(log_precisions) / max_n)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Longest common subsequence length by dynamic programming, O(len(a)·len(b))."""
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(candidate: str, reference: str) -> float:
    """ROUGE-L F1: based on the longest common subsequence of words.

    A subsequence keeps word order but allows gaps, so it rewards getting the
    same content in the same order without requiring contiguous n-grams.
    """
    c, r = _words(candidate), _words(reference)
    lcs = _lcs_length(c, r)
    if lcs == 0:
        return 0.0
    p, rec = lcs / len(c), lcs / len(r)
    return 2 * p * rec / (p + rec)


def cohens_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Agreement between two raters corrected for chance: (p_o - p_e) / (1 - p_e).

    p_o is observed agreement. p_e is the agreement you'd get if both raters
    labeled independently at their own base rates:
    p_e = sum over labels of P(a says label) × P(b says label).
    """
    a, b = list(a), list(b)
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    p_e = sum((ca[lbl] / n) * (cb[lbl] / n) for lbl in set(ca) | set(cb))
    return (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0


# Reference answer and three candidates for the generation-metrics demo.
GEN_REFERENCE = "The meeting was moved to Friday because the manager is sick."
GEN_CANDIDATES = {
    "correct paraphrase": "Since the boss is ill, the team rescheduled the meeting for Friday.",
    "wrong but copies wording": "The meeting was moved to Monday because the manager is sick.",
    "exact copy": "The meeting was moved to Friday because the manager is sick.",
}


# ---------------------------------------------------------------------------
# 5. Figures (rendered into docs/figures by `make figures`)
# ---------------------------------------------------------------------------


def figures() -> dict:
    """Data figures for this lesson, keyed by the name used in the docstring."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = {}
    y, s = synthetic_scores()

    # ROC curve with the AUC shaded.
    fpr, tpr, _ = roc_curve(y, s)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, lw=2, label=f"model (AUC = {roc_auc(y, s):.2f})")
    ax.fill_between(fpr, tpr, step=None, alpha=0.15)
    ax.plot([0, 1], [0, 1], "--", color="grey", label="random (AUC = 0.50)")
    ax.set(xlabel="false-positive rate (share of negatives flagged)", ylabel="true-positive rate (recall)",
           title="ROC curve", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right")
    figs["roc"] = fig

    # Precision-recall curve against the base rate.
    prec, rec, _ = pr_curve(y, s)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(rec, prec, lw=2, label="model")
    ax.axhline(y.mean(), ls=":", color="grey", label=f"base rate ({y.mean():.0%} positive)")
    ax.set(xlabel="recall", ylabel="precision", title="Precision-recall curve", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="upper right")
    figs["pr"] = fig

    # Total cost vs. threshold for three pricings of errors.
    thresholds = np.linspace(s.min(), s.max(), 200)
    fig, ax = plt.subplots(figsize=(6, 4))
    for cfn, cfp in [(1, 1), (10, 1), (1, 10)]:
        costs = []
        for t in thresholds:
            c = confusion(y, s >= t)
            costs.append(cfn * c.fn + cfp * c.fp)
        costs = np.array(costs)
        line, = ax.plot(thresholds, costs, label=f"miss costs {cfn}, false alarm costs {cfp}")
        i = int(np.argmin(costs))
        ax.plot(thresholds[i], costs[i], "o", color=line.get_color())
    ax.set(xlabel="threshold (flag if score >= threshold)", ylabel="total cost", title="Where to cut depends on what errors cost")
    ax.set_ylim(0, 500)
    ax.legend()
    figs["cost_vs_threshold"] = fig

    # DCG position discount.
    ranks = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(ranks, 1 / np.log2(ranks + 1))
    ax.set(xlabel="rank position", ylabel="weight 1 / log2(rank + 1)", title="How much each position counts in DCG", xticks=ranks)
    figs["ndcg_discount"] = fig

    # BLEU and ROUGE-L for the three candidates.
    names = list(GEN_CANDIDATES)
    b = [bleu(GEN_CANDIDATES[n], GEN_REFERENCE) for n in names]
    r = [rouge_l(GEN_CANDIDATES[n], GEN_REFERENCE) for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - 0.18, b, 0.36, label="BLEU")
    ax.bar(x + 0.18, r, 0.36, label="ROUGE-L")
    ax.set_xticks(x, [n.replace(" but ", "\nbut ") for n in names])
    ax.set(ylabel="score", ylim=(0, 1.05), title="Overlap metrics reward wording, not truth")
    ax.legend()
    figs["overlap_scores"] = fig

    for f in figs.values():
        f.tight_layout()
    return figs


# ---------------------------------------------------------------------------
# 6. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. Worked example: fraud detection")
    c = confusion(*fraud_example())
    say(f"TP={c.tp}, FP={c.fp}, FN={c.fn}, TN={c.tn} (100 transactions, 10 fraud, 8 flagged, 6 correct).")
    table(
        ["metric", "calculation", "result"],
        [
            ("precision", "6 correct / 8 flagged", f"{c.precision:.0%}"),
            ("recall", "6 found / 10 actual fraud", f"{c.recall:.0%}"),
            ("F1", "2·0.75·0.60 / (0.75+0.60)", f"{c.f1:.0%}"),
            ("accuracy", "(6 + 88) / 100", f"{c.accuracy:.0%}"),
        ],
    )
    takeaway("Accuracy looks great at 94% while the model misses 4 in 10 frauds.")

    banner("2. The accuracy trap: flag nothing on 1% fraud")
    t = flag_nothing_trap()
    say(f"A model that never flags anything: accuracy {t.accuracy:.0%}, recall {t.recall:.0%}, F1 {t.f1:.0%}.")
    takeaway("On imbalanced data, never lead with accuracy.")

    banner("3. ROC-AUC, computed two ways")
    y, s = synthetic_scores()
    a1, a2 = roc_auc(y, s), roc_auc_rank(y, s)
    say(
        f"""
        500 items, 10% positive, positives scored ~1.5 std devs higher.
        Trapezoid area under the ROC curve: {a1:.4f}. Fraction of
        (positive, negative) pairs ranked correctly: {a2:.4f}. They're the same
        number, which is why AUC reads as "the chance a random positive
        outranks a random negative".
        """
    )

    banner("4. Choosing a threshold by the cost of errors")
    rows = []
    for cfn, cfp in [(1, 1), (10, 1), (1, 10)]:
        thr, cost, cc = best_threshold_by_cost(y, s, cfn, cfp)
        rows.append((f"miss={cfn}, false alarm={cfp}", thr, cc.precision, cc.recall, cost))
    table(["error costs", "threshold", "precision", "recall", "total cost"], rows, floatfmt=".2f")
    say("When misses are expensive the threshold drops and recall rises; when false alarms are expensive it rises.")

    banner("5. Retrieval metrics on one query")
    ranked = ["d7", "d3", "d9", "d1", "d4"]
    relevant = {"d3", "d4", "d8"}
    grades = {"d3": 3, "d4": 2, "d8": 1}
    table(
        ["metric", "value", "why"],
        [
            ("recall@5", recall_at_k(ranked, relevant, 5), "found d3, d4 of {d3, d4, d8}"),
            ("precision@5", precision_at_k(ranked, relevant, 5), "2 of 5 results relevant"),
            ("reciprocal rank", reciprocal_rank(ranked, relevant), "first hit (d3) at rank 2"),
            ("nDCG@5", ndcg_at_k(ranked, grades, 5), "graded: d3=3, d4=2, d8=1"),
        ],
        floatfmt=".3f",
    )
    takeaway("For RAG, recall@k comes first: the model can't use what wasn't retrieved.")

    banner("6. BLEU and ROUGE-L punish correct paraphrases")
    say(f'Reference: "{GEN_REFERENCE}"')
    table(
        ["candidate", "BLEU", "ROUGE-L", "text"],
        [(name, bleu(txt, GEN_REFERENCE), rouge_l(txt, GEN_REFERENCE), txt) for name, txt in GEN_CANDIDATES.items()],
        floatfmt=".2f",
    )
    say(
        """
        The factually wrong answer (Monday) scores almost perfectly because it
        copies the reference's words. The correct paraphrase scores near zero.
        Overlap metrics measure wording, not truth.
        """
    )

    banner("7. Calibrating an LLM judge: percent agreement vs. Cohen's kappa")
    human = ["pass"] * 18 + ["fail"] * 2
    lazy_judge = ["pass"] * 20
    good_judge = ["pass"] * 17 + ["fail"] + ["fail", "fail"]
    rows = []
    for name, j in [("always says pass", lazy_judge), ("careful judge", good_judge)]:
        agree = np.mean([x == y for x, y in zip(human, j)])
        rows.append((name, f"{agree:.0%}", cohens_kappa(human, j)))
    table(["judge", "raw agreement", "Cohen's kappa"], rows, floatfmt=".2f")
    takeaway(
        "A judge that always says 'pass' agrees 90% of the time and has kappa 0. "
        "Calibrate judges against human labels with a chance-corrected statistic."
    )


if __name__ == "__main__":
    demo()
