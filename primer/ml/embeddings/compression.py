r"""
# Compression: smaller vectors, same neighbours

Run: `python -m primer.ml.embeddings.compression`

New to vectors, dimensions or Σ? `primer.notation` builds them from zero.

## The everyday picture

An embedding describes a text with a long list of numbers, like describing
a person with hundreds of adjectives. More adjectives let you tell very
similar people apart, but every adjective takes shelf space, and a search
system has to keep millions of these descriptions in fast memory.

This lesson is about making the descriptions smaller without mixing people
up. There are three tricks, each with an everyday twin:

- **Keep fewer numbers** (Matryoshka truncation): a well-written news story
  puts the headline first and the details later, so you can cut from the
  bottom and still know what happened.
- **Store each number more roughly** (scalar quantization): round every
  price to the nearest dollar. Totals barely change, and the list gets much
  shorter to write down.
- **Keep only a yes/no per number** (binary quantization): instead of
  "4.7 out of 10", just record "above average: yes". Crude, but amazingly
  good for a first sort.

And one trick that rescues the crude ones: **shortlist, then check**. Skim a
pile of CVs fast to pick 100, then read those 100 carefully.

## A tiny worked example: counting bytes

A **dimension** is one number in the vector. A 32-bit float (the usual
format) takes 4 bytes. So one 1,536-dimension vector takes 1,536 × 4 = 6,144
bytes, and ten million of them take:

$$
\text{bytes} = n \times d \times \frac{b}{8}
$$

**Symbols**

| Symbol | Meaning here | Example value |
|---|---|---|
| n | number of vectors | 10,000,000 |
| d | dimensions per vector | 1,536 |
| b | bits per number | 32 (float32), 8 (int8), 1 (binary) |
| b / 8 | bytes per number (8 bits in a byte) | 4, 1, or 1/8 |

**In words:** storage is the number of vectors, times the numbers in each,
times the bytes per number.

**On the example:** 10,000,000 × 1,536 × 32/8 = **61,440,000,000 bytes ≈ 61 GB**.
As int8 it's **15.4 GB**; as bits, **1.9 GB**.

The index adds its own overhead. An HNSW graph (`primer.ml.embeddings.ann`)
stores about 2·M neighbour ids per vector at 4 bytes each: with M = 16 that's
10M × 32 × 4 = **1.28 GB** more.

![Raw storage for 10 million vectors](figures/primer.ml.embeddings.compression.storage.svg)

**Reading it:** each group of bars is one common embedding size, from 384 to
3,072 dimensions. Within a group, the three bars are float32, int8 and
binary storage for ten million vectors, on a log scale (each gridline is 10×).
A 3,072-dimension float32 index needs over 120 GB of memory; the same
vectors as bits fit in under 4 GB.

**Why it matters:** fast vector indexes like HNSW want every vector in RAM,
so storage *is* the server bill. Being able to do this sum in your head
tells you in seconds whether a design fits on one machine.

## Matryoshka: important numbers first, then cut

**Everyday picture:** Russian nesting dolls: a small doll inside a bigger one
inside a bigger one, each complete on its own. A **Matryoshka embedding** is
trained so its first 64 numbers are a decent embedding, its first 256 a
better one, and the full vector the best.

**Tiny example:** a vector (0.9, 0.4, 0.1, 0.05) where the numbers shrink in
importance. Keep the first two, (0.9, 0.4), and rescale it to length 1
(divide by √(0.81 + 0.16) = 0.985): (0.914, 0.406). The dropped numbers were
small, so the direction barely moves: its cosine with the full (normalized)
vector is 0.994.

$$
v_{:m} = \frac{(v_1, \dots, v_m)}{\lVert (v_1, \dots, v_m) \rVert}
$$

**Symbols**

| Symbol | Meaning here | Shape |
|---|---|---|
| v | the full embedding | d numbers |
| m | how many leading numbers we keep | 1 to d |
| v₁ … vₘ | the first m numbers | m numbers |
| ‖·‖ | length (norm) | one number |
| v₍:ₘ₎ | the truncated, re-normalized vector | m numbers, length 1 |

**In words:** keep the first m numbers and rescale them to length 1.

**On the example:** m = 2: (0.9, 0.4) / 0.985 = (0.914, 0.406).

This only works if the important numbers really come first. A real
Matryoshka model is *trained* that way: the same contrastive loss
(`primer.ml.embeddings.contrastive`) is applied to several prefixes at once
(the first 64, 128, 256, … numbers), so each prefix must work on its own.
Here we imitate it by rotating vectors onto their **principal directions**
(the directions along which the collection varies most, found with the SVD;
see `primer.notation`), which puts the most informative number first.

```mermaid
flowchart LR
  T[Text] --> E[Encoder] --> V["full vector (d numbers)"]
  V --> P64["first 64"] --> L64[loss]
  V --> P256["first 256"] --> L256[loss]
  V --> PD["all d"] --> LD[loss]
  L64 & L256 & LD --> S["sum: every prefix<br/>must work on its own"]
```

**Reading it:** one text, one encoder, one vector, scored several times. Each
prefix of the vector is judged by the usual contrastive loss, and the model
is trained on the sum. That's the whole trick: nothing about the model
changes, only how its output is graded.

![How much variation each dimension carries](figures/primer.ml.embeddings.compression.spectrum.svg)

**Reading it:** the horizontal axis is the dimension number and the vertical
axis is how much the collection varies along it (its **variance**: the
average squared distance from the mean, on a log scale). In importance
order, the first few dimensions carry most of the variation and it falls
steadily after that, so cutting from the end loses little. In a random
order every dimension carries a similar share, so cutting any of them costs
the same.

![Recall@10 vs. dimensions kept](figures/primer.ml.embeddings.compression.matryoshka.svg)

**Reading it:** the horizontal axis is how many leading dimensions we keep
(of 256); the vertical axis is **recall@10**, the share of each query's true
top-10 neighbours we still find. In importance order, 32 dimensions (one
eighth) still find about 93% of the neighbours; in random order they find
about 44%. The star is the two-stage design: search with the first 32
numbers to shortlist 100 candidates, then re-rank those 100 with the full
vectors. It finds all of them.

## Measuring what compression costs: recall@k

$$
\text{recall@}k = \frac{\lvert \text{found}_k \cap \text{true}_k \rvert}{k}
$$

**Symbols**

| Symbol | Meaning here | Example |
|---|---|---|
| k | how many results we look at | 10 in this lesson; 3 in the example |
| foundₖ | the k results the compressed search returned | (3, 4, 1) |
| trueₖ | the k results an exact, full-precision search returns | (1, 2, 3) |
| ∩ | "items in both" | {1, 3} |
| \|·\| | count the items | 2 |

**In words:** the share of the true top-k that the compressed search also
found.

**On an example:** true (1, 2, 3), found (3, 4, 1): two of the three appear,
so recall@3 = 2/3.

## Scalar quantization: 256 levels per number

**Everyday picture:** rounding prices to the nearest dollar. Here, every
number is rounded to one of 256 marks on a ruler that runs from the smallest
to the largest value seen in that dimension. 256 marks fit in one byte
(**int8**), a quarter of a float's 4 bytes.

**Tiny example:** a dimension whose values run from lo = −1 to hi = 1. The 256
marks are 2/255 = 0.00784 apart. The value 0 sits at (0 − (−1)) / 2 × 255 = 127.5
marks, rounds to mark **128**, and decodes back to −1 + 128/255 × 2 = **0.00392**.
It's off by 0.00392, half a mark, the worst case.

$$
\text{code} = \operatorname{round}\!\left(\frac{x - lo}{hi - lo} \times 255\right),
\qquad \hat{x} = lo + \frac{\text{code}}{255}\,(hi - lo)
$$

**Symbols**

| Symbol | Meaning here | Range |
|---|---|---|
| x | one number in a vector | between lo and hi (clipped if outside) |
| lo, hi | smallest and largest value of this dimension across the documents | calibrated once |
| (x − lo)/(hi − lo) | where x sits between lo and hi, as a fraction | 0 to 1 |
| × 255 | stretch to the 256 marks 0 … 255 | 0 to 255 |
| round | nearest whole number | |
| code | the stored byte | 0 to 255 |
| x̂ (x-hat) | the value decoded back | within half a mark of x |

**In words:** find where x sits between the dimension's minimum and maximum,
turn that into one of 256 whole-number marks, and store the mark; to decode,
walk back from the mark to the value.

**On the example:** x = 0, lo = −1, hi = 1: code = round(0.5 × 255) = round(127.5) = 128;
x̂ = −1 + (128/255)·2 = 0.00392.

![A vector before and after int8 rounding](figures/primer.ml.embeddings.compression.int8.svg)

**Reading it:** the line shows the first 40 numbers of one real vector from
this lesson's corpus; the steps show the same numbers after rounding to
256 levels and decoding. The two are almost indistinguishable: the rounding
error (the bottom panel) never exceeds half a mark. That's why int8 search
here still finds about 98% of the true neighbours.

## Binary quantization: one bit per number

**Everyday picture:** a yes/no questionnaire. For each number, record only
"positive: yes or no". Two texts are compared by counting how many answers
differ, the **Hamming distance**.

**Tiny example:** the eight values (0.3, −0.2, 0.0, 5, −1, 2, −3, 0.1) become the
bits 1 0 0 1 0 1 0 1, packed into one byte: 0b10010101 = **149**. Compare with
0b00010100: they differ in the first and last positions, so the Hamming
distance is **2**.

$$
\text{bit}_i = [\,x_i > 0\,], \qquad
\text{hamming}(a, b) = \sum_{i=1}^{d} [\,a_i \ne b_i\,]
$$

**Symbols**

| Symbol | Meaning here | Range |
|---|---|---|
| xᵢ | the i-th number of the vector | any real number |
| [ condition ] | 1 if the condition is true, 0 if not | 0 or 1 |
| bitᵢ | the stored bit for position i | 0 or 1 |
| a, b | two bit codes being compared | d bits each |
| aᵢ ≠ bᵢ | the two codes disagree at position i | |
| Σ | add up over all d positions | |
| hamming(a, b) | number of positions that disagree | 0 to d |

**In words:** keep one bit per number that says whether it was positive, and
measure distance as the number of positions where two codes disagree.

**On the example:** 149 = 10010101 vs 20 = 00010100: positions 1 and 8 differ,
so the distance is 2. Computers do this with one XOR (mark the differing
bits) and one popcount (count them), which is why binary search is extremely
fast.

```mermaid
flowchart LR
  Q[Query] --> B["Stage 1: bits + Hamming<br/>scan all 5,000 docs<br/>(32× smaller, very fast)"]
  B --> S[Shortlist of 100]
  S --> F["Stage 2: full float vectors<br/>exact dot product on 100 only"]
  F --> T[Top 10]
  STORE[("bits for every doc (RAM)<br/>floats for every doc (disk or RAM)")] --> B
  STORE --> F
```

**Reading it:** the cheap representation is used where the work is big (every
document), and the expensive one where the work is small (100 candidates).
The bits live in fast memory; the full vectors can live somewhere slower
because only a hundred are read per query.

![Recall@10 of each compression, with and without re-scoring](figures/primer.ml.embeddings.compression.quantization.svg)

**Reading it:** each bar is one way of storing the documents, measured by
recall@10 against exact float32 search. int8 alone keeps about 98%. Binary
alone keeps only about half: signs lose a lot. But binary as a *shortlist*,
re-scored with full vectors, climbs back to about 97%, and 32-dimension
Matryoshka shortlists to 100%. Crude-then-exact is the pattern to remember.

**Why it matters:** these knobs move real money. Many vector databases ship
int8 and binary quantization with re-scoring built in, and embedding
providers increasingly ship Matryoshka-trained models so you can pick your
dimension. The cost is always measured the same way: recall@k on your own
queries.

## In 20 seconds
- Raw storage is n × d × bytes per number: 10M × 1,536 float32 ≈ 61 GB, before
  index overhead.
- Matryoshka models put the important information in the leading numbers,
  so you can truncate and re-normalize.
- int8 stores 256 levels per number (4× smaller, little loss); binary keeps
  signs only (32× smaller, big loss alone).
- Shortlist with the crude form, re-score with full vectors: most of the
  quality at a fraction of the memory.
- Always measure the cost as recall@k against exact search.

## Self-test questions

**Q: How much memory do ten million 1,536-dimension float32 vectors need?**
10,000,000 × 1,536 × 4 bytes = 61.4 GB of raw vectors, plus index overhead
(e.g. ~1.3 GB of HNSW links at M = 16).

**Q: What makes Matryoshka embeddings truncatable, and how do you use that?**
They're trained with the loss applied to several prefixes at once, so the
first m numbers form a good embedding by themselves. Search with short
vectors for speed and memory, then re-score the top candidates with the
full vectors.

**Q: Scalar vs. binary quantization: what do you give up?**
int8 rounds each number to 256 levels, 4× smaller with a small recall loss.
Binary keeps only signs, 32× smaller, but recall drops a lot on its own; it
works as a first-stage shortlist followed by full-precision re-scoring.

**Q: Why not just use as many dimensions as possible?**
Gains flatten out while storage, memory and search time grow linearly.
A smaller model trained for your domain often beats a bigger generic one,
so benchmark on your own queries.

## The papers behind this lesson

- **Kusupati et al., *Matryoshka Representation Learning* (2022)**: https://arxiv.org/abs/2205.13147.
  Trained embeddings whose every prefix is a usable embedding, by summing the loss over nested prefix lengths. [annotated companion](../../../papers/matryoshka.html)

## Further reading
- Hugging Face, *Embedding Quantization* (binary and int8 with re-scoring): https://huggingface.co/blog/embedding-quantization
- Hugging Face, *Introduction to Matryoshka Embedding Models*: https://huggingface.co/blog/matryoshka
- Faiss wiki, *Guidelines to choose an index*: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
"""

from __future__ import annotations

import numpy as np

from primer._show import banner, say, table, takeaway

# ---------------------------------------------------------------------------
# 1. Storage math
# ---------------------------------------------------------------------------


def storage_bytes(n_vectors: int, dim: int, bits_per_value: int = 32) -> int:
    """Raw bytes to store n vectors of `dim` numbers at `bits_per_value` each.

    float32 = 32 bits, int8 = 8 bits, binary = 1 bit. Integer arithmetic so
    the answer is exact.
    """
    return n_vectors * dim * bits_per_value // 8


def hnsw_link_bytes(n_vectors: int, M: int = 16, bytes_per_link: int = 4) -> int:
    """Bytes for the bottom layer of an HNSW graph: 2·M neighbour ids per vector.

    Upper layers hold only a small fraction of the vectors, so they add a few
    percent more; this is the part that scales with every vector.
    (See `primer.ml.embeddings.ann` for how HNSW works.)
    """
    return n_vectors * 2 * M * bytes_per_link


# ---------------------------------------------------------------------------
# 2. A test corpus with realistic structure, and the recall metric
# ---------------------------------------------------------------------------


def _unit(X: np.ndarray) -> np.ndarray:
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def make_corpus(n_docs: int = 5000, n_queries: int = 100, dim: int = 256, k: int = 10, seed: int = 0):
    """Docs, queries and each query's true top-k neighbours (by exact cosine).

    Real embeddings aren't uniform noise: a few directions carry most of the
    variation. We build that in: 50 topic clusters, a spread along direction
    i that shrinks like 1/i, and then a random rotation so the raw coordinates
    show no order (like a real model's output). Queries are noisy copies of
    random docs.
    """
    rng = np.random.default_rng(seed)
    scales = 1 / np.arange(1, dim + 1)  # spread along direction i ∝ 1/i: a few directions dominate
    centers = rng.standard_normal((50, dim)) * scales
    docs = centers[rng.integers(50, size=n_docs)] + 0.6 * rng.standard_normal((n_docs, dim)) * scales
    R, _ = np.linalg.qr(rng.standard_normal((dim, dim)))  # random orthogonal matrix
    docs = _unit(docs @ R)
    queries = _unit(docs[rng.integers(n_docs, size=n_queries)] + 0.03 * rng.standard_normal((n_queries, dim)))
    truth = top_k(docs, queries, k)
    return docs, queries, truth


def top_k(docs: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest dot products per query, best first. Shape (n_queries, k)."""
    S = queries @ docs.T
    idx = np.argpartition(-S, k, axis=1)[:, :k]  # the k best, unordered (cheaper than a full sort)
    order = np.argsort(-np.take_along_axis(S, idx, axis=1), axis=1)
    return np.take_along_axis(idx, order, axis=1)


def recall_at_k(found: np.ndarray, truth: np.ndarray) -> float:
    """Average share of each query's true neighbours that appear in what was found."""
    return float(np.mean([len(set(f) & set(t)) / len(t) for f, t in zip(found, truth)]))


# ---------------------------------------------------------------------------
# 3. Matryoshka: put the important directions first, then truncate
# ---------------------------------------------------------------------------


def matryoshka_order(docs: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
    """Rotate so dimension 1 carries the most variation, dimension 2 the next, and so on.

    A real Matryoshka model is *trained* so every prefix is a good embedding.
    Rotating onto the principal directions (the SVD of the doc matrix, see
    `primer.notation`) is the closest do-it-yourself imitation, and it
    shows the same effect. A rotation changes no dot product, so full-length
    search is untouched; only truncation behaves differently.
    """
    _, _, Vt = np.linalg.svd(docs, full_matrices=False)
    return (docs if X is None else X) @ Vt.T


def random_order(docs: np.ndarray, X: np.ndarray | None = None, seed: int = 1) -> np.ndarray:
    """A random rotation: every dimension carries a similar share of the information."""
    R, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((docs.shape[1], docs.shape[1])))
    return (docs if X is None else X) @ R


def search_truncated(docs: np.ndarray, queries: np.ndarray, truth: np.ndarray, dims: int) -> float:
    """recall@k using only the first `dims` numbers of every vector (re-normalized)."""
    return recall_at_k(top_k(_unit(docs[:, :dims]), _unit(queries[:, :dims]), truth.shape[1]), truth)


def search_truncated_then_rescore(docs, queries, truth, dims: int, shortlist: int) -> float:
    """Two stages: shortlist with short vectors (fast, small), re-rank the shortlist with full vectors."""
    cand = top_k(_unit(docs[:, :dims]), _unit(queries[:, :dims]), shortlist)
    return recall_at_k(_rescore(docs, queries, cand, truth.shape[1]), truth)


def _rescore(docs: np.ndarray, queries: np.ndarray, cand: np.ndarray, k: int) -> np.ndarray:
    """Re-rank each query's candidate ids by exact dot product; keep the best k."""
    exact = np.einsum("qd,qcd->qc", queries, docs[cand])
    return np.take_along_axis(cand, np.argsort(-exact, axis=1)[:, :k], axis=1)


# ---------------------------------------------------------------------------
# 4. Scalar (int8) quantization: 256 levels per dimension
# ---------------------------------------------------------------------------


def scalar_quantize_int8(X: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Map each value in [lo, hi] (per dimension) onto the 256 codes 0..255.

    lo and hi are calibrated from the documents (their min and max per
    dimension). Values outside are clipped.
    """
    scaled = (np.clip(X, lo, hi) - lo) / (hi - lo) * 255
    return np.round(scaled).astype(np.uint8)


def dequantize_int8(codes: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return lo + codes.astype(float) / 255 * (hi - lo)


def search_int8(docs: np.ndarray, queries: np.ndarray, truth: np.ndarray) -> float:
    """Store docs as int8; compare float queries against the decoded docs."""
    lo, hi = docs.min(axis=0), docs.max(axis=0)
    approx = dequantize_int8(scalar_quantize_int8(docs, lo, hi), lo, hi)
    return recall_at_k(top_k(approx, queries, truth.shape[1]), truth)


# ---------------------------------------------------------------------------
# 5. Binary quantization: keep only the sign, compare with Hamming distance
# ---------------------------------------------------------------------------


def binary_quantize(X: np.ndarray) -> np.ndarray:
    """One bit per value (1 if positive), packed 8 per byte. 32x smaller than float32."""
    return np.packbits(X > 0, axis=-1)


# popcount lookup: how many 1-bits each byte value 0..255 has
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def hamming_distances(doc_codes: np.ndarray, query_code: np.ndarray) -> np.ndarray:
    """Number of differing bits between one query code and every doc code.

    XOR marks the differing bits; counting 1-bits ("popcount") adds them up.
    CPUs do both in a single instruction each, which is why binary search is fast.
    """
    return _POPCOUNT[np.bitwise_xor(doc_codes, query_code)].sum(axis=-1)


def _hamming_top(docs: np.ndarray, queries: np.ndarray, n: int) -> np.ndarray:
    dc, qc = binary_quantize(docs), binary_quantize(queries)
    return np.stack([np.argsort(hamming_distances(dc, q), kind="stable")[:n] for q in qc])


def search_binary(docs: np.ndarray, queries: np.ndarray, truth: np.ndarray) -> float:
    return recall_at_k(_hamming_top(docs, queries, truth.shape[1]), truth)


def search_binary_then_rescore(docs, queries, truth, shortlist: int) -> float:
    """Shortlist by Hamming distance on bits, then re-rank the shortlist with the full float vectors."""
    return recall_at_k(_rescore(docs, queries, _hamming_top(docs, queries, shortlist), truth.shape[1]), truth)


# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------


def figures() -> dict:
    """Plots computed from this module's own functions. Keys match the docstring's image names."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = {}
    docs, queries, truth = make_corpus()
    md, mq = matryoshka_order(docs), matryoshka_order(docs, queries)
    rd, rq = random_order(docs), random_order(docs, queries)

    # storage
    dims = [384, 768, 1024, 1536, 3072]
    x = np.arange(len(dims))
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (bits, label) in enumerate(((32, "float32"), (8, "int8"), (1, "binary"))):
        ax.bar(x + (i - 1) * 0.27, [storage_bytes(10_000_000, d, bits) / 1e9 for d in dims], 0.27, label=label)
    ax.set_yscale("log")
    ax.set_xticks(x, [str(d) for d in dims])
    ax.set(xlabel="dimensions per vector", ylabel="GB for 10 million vectors (log scale)", title="Raw vector storage")
    ax.legend()
    ax.grid(axis="y", which="major", alpha=0.3)
    figs["storage"] = fig

    # spectrum
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, 257), md.var(axis=0), label="importance order (principal directions)")
    ax.plot(np.arange(1, 257), rd.var(axis=0), label="random order")
    ax.set_yscale("log")
    ax.set(xlabel="dimension number", ylabel="variance along that dimension (log scale)", title="Where the information lives")
    ax.legend()
    figs["spectrum"] = fig

    # matryoshka
    ms = [4, 8, 16, 32, 64, 128, 256]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ms, [search_truncated(md, mq, truth, m) for m in ms], marker="o", label="importance order (Matryoshka-like)")
    ax.plot(ms, [search_truncated(rd, rq, truth, m) for m in ms], marker="o", label="random order")
    ax.scatter([32], [search_truncated_then_rescore(md, mq, truth, 32, 100)], marker="*", s=250, color="C3", zorder=5, label="32 dims → shortlist 100 → re-score")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ms, [str(m) for m in ms])
    ax.set(xlabel="dimensions kept (of 256)", ylabel="recall@10", ylim=(0, 1.05), title="Truncation only works when importance comes first")
    ax.legend(loc="lower right")
    figs["matryoshka"] = fig

    # int8
    lo, hi = docs.min(axis=0), docs.max(axis=0)
    v = docs[0, :40]
    back = dequantize_int8(scalar_quantize_int8(docs[:1], lo, hi), lo, hi)[0, :40]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    a1.plot(v, marker="o", ms=3, label="original float32")
    a1.step(np.arange(40), back, where="mid", label="int8, decoded")
    a1.set(ylabel="value", title="One vector, first 40 numbers")
    a1.legend()
    a2.bar(np.arange(40), back - v, color="C3")
    a2.set(xlabel="dimension", ylabel="rounding error")
    figs["int8"] = fig

    # quantization
    labels = ["float32\n(exact)", "int8", "binary", "binary →\nre-score 100", "32 dims →\nre-score 100"]
    vals = [1.0, search_int8(docs, queries, truth), search_binary(docs, queries, truth), search_binary_then_rescore(docs, queries, truth, 100), search_truncated_then_rescore(md, mq, truth, 32, 100)]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, vals, color=["0.6", "C0", "C1", "C2", "C3"])
    ax.bar_label(bars, fmt="%.2f")
    ax.set(ylabel="recall@10", ylim=(0, 1.1), title="Crude first pass, exact second pass")
    figs["quantization"] = fig

    for f in figs.values():
        f.tight_layout()
    return figs


# ---------------------------------------------------------------------------
# 7. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. Storage math: 10 million vectors")
    rows = [(d, storage_bytes(10_000_000, d, 32) / 1e9, storage_bytes(10_000_000, d, 8) / 1e9, storage_bytes(10_000_000, d, 1) / 1e9) for d in (384, 768, 1536, 3072)]
    table(["dims", "float32 GB", "int8 GB", "binary GB"], rows, floatfmt=".2f")
    say(f"Plus HNSW links at M=16: {hnsw_link_bytes(10_000_000) / 1e9:.2f} GB for the bottom layer alone.")
    takeaway("10M × 1,536 × 4 bytes ≈ 61 GB. Do this sum before you pick a machine.")

    docs, queries, truth = make_corpus()
    md, mq = matryoshka_order(docs), matryoshka_order(docs, queries)
    rd, rq = random_order(docs), random_order(docs, queries)

    banner("2. Matryoshka truncation: recall@10 vs. dimensions kept (of 256)")
    table(["dims kept", "importance order", "random order"], [(m, search_truncated(md, mq, truth, m), search_truncated(rd, rq, truth, m)) for m in (8, 16, 32, 64, 128, 256)], floatfmt=".3f")
    say(f"Two stages: 32 dims to shortlist 100, full vectors to re-rank: recall@10 = {search_truncated_then_rescore(md, mq, truth, 32, 100):.3f}.")

    banner("3. Quantization")
    x = np.array([[-1.0, 0.0, 1.0]])
    codes = scalar_quantize_int8(x, np.full(3, -1.0), np.full(3, 1.0))
    say(f"int8 codes for (-1, 0, 1) on the range [-1, 1]: {codes.tolist()[0]}. Signs of (0.3,-0.2,0,5,-1,2,-3,0.1) pack to {binary_quantize(np.array([[0.3, -0.2, 0.0, 5.0, -1.0, 2.0, -3.0, 0.1]])).tolist()[0]}.")
    table(
        ["storage", "bytes per vector", "recall@10"],
        [
            ("float32", 256 * 4, 1.0),
            ("int8", 256, search_int8(docs, queries, truth)),
            ("binary", 256 // 8, search_binary(docs, queries, truth)),
            ("binary, re-score top 100", "32 (+ floats for 100)", search_binary_then_rescore(docs, queries, truth, 100)),
        ],
        floatfmt=".3f",
    )
    takeaway("Crude representation for the big scan, exact vectors for the short list: most of the quality, a fraction of the memory.")


if __name__ == "__main__":
    demo()
