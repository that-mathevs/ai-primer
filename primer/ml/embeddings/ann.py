r"""
# Vector search: finding the nearest neighbors without checking everyone

Run: `python -m primer.ml.embeddings.ann`

New to the notation? `primer.notation` explains every symbol used here from
zero (vectors, Σ, logarithms, big-O).

## The everyday picture

You've just moved to a new country and want the house nearest to a landmark.
You could walk up to every house in the country and measure the distance.
That's always right, but it takes forever. Or you could do what everyone
actually does: take the **highway** to the right region, switch to **main
roads** to the right town, then walk the **side streets** to the house. You
check only a handful of places and still end up at (almost always) the right
door.

That is the whole problem of this lesson. A search engine that works on
meaning stores every document as a **vector** (a list of numbers, see
`primer.ml.embeddings.similarity`) and turns the question into a vector too.
The best documents are the vectors *nearest* the question. With ten million
documents, "walk to every house" is too slow, so we build a map with
highways. Structures like that are called **approximate nearest neighbor
(ANN) indexes**. *Approximate* because, like the traveller, they very
occasionally stop at the second-nearest house instead of the nearest.

Four ideas cover almost every vector database in use:

| Index | Everyday picture | What you give up |
|---|---|---|
| **Flat** | Visit every house | Nothing; it's exact. Just slow at scale |
| **IVF** | A library sorts books into sections; you search only the few nearest sections | Books shelved in the section next door |
| **PQ** | Describe a face by picking the closest nose, eyes and mouth from a small catalogue | Fine detail: the description is lossy |
| **HNSW** | Highway, then main roads, then side streets | Memory for all the roads, and the occasional near-miss |

A few words used throughout, in plain terms:

- **Similarity score.** How alike two vectors are. Here we use the **dot
  product** (multiply the two lists position by position and add up). All
  vectors are rescaled to length 1 first (**normalized**), so the dot
  product equals the **cosine similarity**, which runs from −1 (opposite) to
  1 (identical). Higher means closer.
- **Top k.** The k best matches, e.g. the top 10.
- **Recall@k.** Of the true top k (found by exhaustive search), the fraction
  the index actually returned. Recall@10 = 0.9 means it found 9 of the true
  10. It's the standard measure of an ANN index's accuracy.
- **Big-O, e.g. O(N).** How the work grows with the data. O(N) means doubling
  the number of vectors N doubles the work; O(log N) means doubling N adds
  only one more step.

## A tiny worked example: eight houses on a map

Every index in this lesson is shown first on the same eight points, drawn on
a flat map so you can check each step with a pencil. The question (the
**query**) sits at (5, 4). On a flat map "near" is ordinary distance; we keep
it **squared** (dx² + dy²) so every number stays whole. Squaring doesn't
change which point is nearest.

| Point | Position | Squared distance to the query (5, 4) |
|---|---|---|
| A | (0, 0) | 5² + 4² = **41** |
| B | (2, 1) | 3² + 3² = **18** |
| C | (4, 0) | 1² + 4² = **17** |
| D | (1, 3) | 4² + 1² = **17** |
| E | (3, 3) | 2² + 1² = **5** |
| F | (5, 2) | 0² + 2² = **4** |
| G | (2, 5) | 3² + 1² = **10** |
| H | (5, 5) | 0² + 1² = **1** ← the true nearest |

Checking all eight rows is **flat search**. It's exact, and it costs one
distance per stored point.

![The eight-point map, its two HNSW layers and the search path](figures/primer.ml.embeddings.ann.toy_map.svg)

**Reading it:** the eight dots are the stored points and the yellow star is
the query at (5, 4). Thin grey lines are the "streets" (layer 0: every point,
linked to its neighbors); the thick blue lines are the "highway" (layer 1:
only A, E and G). The red arrows are the HNSW search worked through below:
one highway hop from A to E, then one street hop from E to H. It checked a
few points near its route instead of reading the whole table.

## 1. Flat search: the exact baseline

```mermaid
flowchart LR
  Q[Query vector] --> D["Dot product with<br/>EVERY stored vector<br/>(N comparisons)"]
  D --> T[Keep the k highest]
  T --> R[Exact top k]
```

**Reading it:** there is only one path and no shortcuts. The middle box
touches every stored vector, which is why the cost grows in step with N. It
is always exactly right, so every other index is measured against it: its
answer is the "truth" in recall@k.

$$
s(q, x) = q \cdot x = \sum_{i=1}^{d} q_i \, x_i
$$

**Symbols**

| Symbol | Meaning here | Range |
|---|---|---|
| q | the query vector | d numbers, length 1 |
| x | one stored vector | d numbers, length 1 |
| d | number of dimensions (numbers per vector) | 2 in the toy; 384 to 3072 in practice |
| i | position in the vector, counted from 1 | 1 … d |
| q_i, x_i | the i-th number of q and of x | −1 … 1 |
| Σ | "add up the following for every i from 1 to d" | |
| s(q, x) | similarity score, the dot product | −1 … 1 for unit vectors |

**In words:** the score of a stored vector is what you get by multiplying it
with the query number by number and adding the results.

**On the example:** with unit vectors q = (0.8, 0.6) and x = (0.6, 0.8):
s = 0.8·0.6 + 0.6·0.8 = 0.48 + 0.48 = **0.96**, very similar.

**In Python:**

```python
q = [0.8, 0.6]
x = [0.6, 0.8]
# s(q, x) = Σ q_i x_i
round(sum(q_i * x_i for q_i, x_i in zip(q, x)), 2)  # → 0.96
points = {"A": (0, 0), "B": (2, 1), "C": (4, 0), "D": (1, 3),
          "E": (3, 3), "F": (5, 2), "G": (2, 5), "H": (5, 5)}
query = (5, 4)
def sq_dist(p):
    return (p[0] - query[0]) ** 2 + (p[1] - query[1]) ** 2
# flat search on the map: check all eight
min(points, key=lambda name: sq_dist(points[name]))  # → 'H'
```

**In code:** `FlatIndex.search` scores the query against every stored vector
and keeps the best k with `top_k`; `normalize` rescales vectors to length 1
first, so the dot product is the cosine.

**Why it matters:** flat search costs N·d multiply-adds per query. At 10
million vectors of 1,536 dimensions that's about 15 billion, far too many
for an interactive search, and the raw vectors alone take
10,000,000 × 1,536 × 4 bytes ≈ **61 GB** of memory. Below about a million
vectors, flat search is often the right answer: simple, exact, and fast
enough.

**In code:** `storage_estimate` does that memory sum for any n and d.

## 2. IVF: search only the nearest sections of the library

**The everyday picture.** A library doesn't search every shelf for a book
about sailing. It sorts books into sections ahead of time, and you go to
"Sports" and maybe "Travel", the one or two most promising sections, and
search only there. A sailing book shelved in the wrong section is missed
unless you check that section too.

**On the eight points.** Split them into two groups (**clusters**) ahead of
time: *left* = {A, B, C, D} and *right* = {E, F, G, H}. Each cluster is
summarized by its **centroid**, the average position of its members:
left = (1.75, 1), right = (3.75, 3.75). For the query (5, 4):

| Cluster | Centroid | Squared distance to (5, 4) |
|---|---|---|
| left | (1.75, 1) | 3.25² + 3² = 19.5625 |
| right | (3.75, 3.75) | 1.25² + 0.25² = **1.625** ← probe this one |

Scan only E, F, G, H: the nearest is H (distance 1). That's 2 centroid
checks + 4 point checks = 6 instead of 8. With a million points in 1,000
clusters the saving is enormous.

```mermaid
flowchart TB
  subgraph BUILD["Ahead of time"]
    V[All vectors] --> KM["k-means: find nlist centroids"]
    KM --> L["Put each vector on the list<br/>of its nearest centroid"]
  end
  subgraph QUERY["At query time"]
    Q[Query] --> C["Compare with the nlist centroids"]
    C --> P["Pick the nprobe nearest"]
    P --> S["Scan only those lists"]
    S --> R[Top k]
  end
  L -.-> S
```

**Reading it:** the top box runs once, when the index is built: **k-means**
(a simple clustering method: guess centroids, assign every vector to its
nearest, move each centroid to the average of its members, repeat) sorts the
vectors into `nlist` lists. The bottom box runs per query and never touches
lists it didn't pick. The only dial is `nprobe`: how many lists to open.

The set of all points closer to one centroid than to any other is called that
centroid's **Voronoi cell**: the "section" of the library. IVF's blind spot
is a true neighbor sitting just across a cell boundary.

![IVF cells and the two a query probes](figures/primer.ml.embeddings.ann.ivf_cells.svg)

**Reading it:** 600 points, colored by which of 12 centroids (black X) they
belong to; each color patch is a Voronoi cell. The query (yellow star)
compares itself with the 12 centroids, then scans only the two nearest cells
(red X, bright points). The faded points are never looked at. A true neighbor
sitting just across a boundary, in a faded cell, would be missed, which is
exactly why raising `nprobe` raises recall.

$$
\text{comparisons per query} \approx n_{\text{list}} + N \cdot \frac{n_{\text{probe}}}{n_{\text{list}}}
$$

**Symbols**

| Symbol | Meaning here | Typical range |
|---|---|---|
| N | number of stored vectors | thousands to billions |
| n_list | number of clusters (lists) | ≈ √N, e.g. 1,000 for 1M vectors |
| n_probe | clusters scanned per query, the recall dial | 1 … n_list |
| N · n_probe / n_list | vectors in the scanned lists, assuming equal-size clusters | |

**In words:** you pay once to compare with every centroid, plus the share of
the collection that lives in the clusters you open.

**On the example:** 2 + 8 · 1/2 = **6** comparisons (versus 8). At
N = 1,000,000, n_list = 1,000, n_probe = 10: 1,000 + 10,000 = **11,000**,
about 1% of a flat scan.

**In Python:**

```python
left = [(0, 0), (2, 1), (4, 0), (1, 3)]
right = [(3, 3), (5, 2), (2, 5), (5, 5)]
def centroid(members):
    return tuple(sum(p[i] for p in members) / len(members) for i in range(2))
centroid(left), centroid(right)  # → ((1.75, 1.0), (3.75, 3.75))
# probe the nearer
[(c[0] - 5) ** 2 + (c[1] - 4) ** 2 for c in (centroid(left), centroid(right))]  # → [19.5625, 1.625]
def comparisons(N, n_list, n_probe):
    # n_list centroids + the vectors in the opened lists
    return n_list + N * n_probe // n_list
comparisons(8, 2, 1)  # → 6
comparisons(1_000_000, 1_000, 10)  # → 11000
```

**In code:** `IVFIndex.train` finds the centroids with `kmeans`,
`IVFIndex.add` files each vector on its nearest centroid's list, and
`IVFIndex.search` scans only the nprobe nearest lists. `tiny_ivf_search`
replays the eight-point example.

**Why it matters:** IVF is cheap to build and light on memory, and
`nprobe = nlist` gives you back an exact flat search, which is a handy
sanity check. Its accuracy depends on how well the clusters fit the data, so
retrain the centroids when the data changes a lot.

## 3. PQ: store each vector as a few catalogue numbers

**The everyday picture.** A police sketch artist doesn't record every pixel
of a face. They pick the closest nose from a catalogue of 256 noses, the
closest eyes from 256 eyes, the closest mouth, and so on. The whole face
becomes a handful of catalogue numbers: tiny to store, close enough to
recognize someone. That's **product quantization (PQ)**. To **quantize**
means to round a value to the nearest entry of a fixed set.

**On a 4-number vector.** Split x = (0.9, 0.1, −0.2, 0.8) into two halves.
Each half is matched against a four-entry catalogue (a **codebook**), here
the four compass directions: 0 = (1, 0), 1 = (0, 1), 2 = (−1, 0),
3 = (0, −1).

| Half | Values | Nearest catalogue entry | Code |
|---|---|---|---|
| 1 | (0.9, 0.1) | (1, 0) | **0** |
| 2 | (−0.2, 0.8) | (0, 1) | **1** |

The vector is now stored as two codes, [0, 1]. To score the query
q = (1, 0, 0, 1) against *any* stored vector, first build one small table
per half: the query's half dotted with each catalogue entry.

| | entry 0 | entry 1 | entry 2 | entry 3 |
|---|---|---|---|---|
| T₁ (q's half 1 = (1, 0)) | 1 | 0 | −1 | 0 |
| T₂ (q's half 2 = (0, 1)) | 0 | 1 | 0 | −1 |

The approximate score of codes [0, 1] is T₁[0] + T₂[1] = 1 + 1 = **2**. The
exact score is 0.9 + 0.8 = **1.7**: close, not identical. That's the price of
compression.

```mermaid
flowchart LR
  subgraph ENC["Encode (once per vector)"]
    X["x: d numbers"] --> SPLIT["split into m pieces"]
    SPLIT --> NN["each piece → nearest of<br/>256 codebook entries"]
    NN --> CODE["m one-byte codes"]
  end
  subgraph ADC["Score (per query)"]
    Q["query q"] --> TAB["m small tables:<br/>q's piece · every entry"]
    TAB --> SUM["score = sum of m<br/>table lookups"]
  end
  CODE --> SUM
```

**Reading it:** the left box shrinks every stored vector to m bytes, once.
The right box is the trick: the query is *not* compressed. We precompute m
tables of 256 numbers, and then scoring any stored vector is just m lookups
and additions, with no multiplication at all. Keeping the query exact while
the stored side is compressed is called **asymmetric distance computation
(ADC)**.

$$
\hat{s}(q, x) = \sum_{j=1}^{m} T_j\big[c_j(x)\big], \qquad T_j[c] = q^{(j)} \cdot C_j[c]
$$

**Symbols**

| Symbol | Meaning here | Range |
|---|---|---|
| m | number of pieces each vector is split into | 2 in the toy; 8 to 96 in practice |
| j | which piece, counted from 1 | 1 … m |
| q^(j) | the j-th piece of the query (d/m numbers) | |
| C_j | the codebook for piece j: its catalogue of entries | 2^nbits entries, usually 256 |
| C_j[c] | entry number c of that catalogue | |
| c_j(x) | the code stored for piece j of x: which entry was nearest | 0 … 255 (one byte) |
| T_j[c] | precomputed table: q's piece j dotted with entry c | |
| ŝ(q, x) | approximate score ("s-hat": the hat means *estimate*) | |

**In words:** the estimated score is the sum, over the pieces, of the table
value for whichever catalogue entry that piece of the stored vector was
rounded to.

**On the example:** ŝ = T₁[c₁] + T₂[c₂] = T₁[0] + T₂[1] = 1 + 1 = **2**
(exact: 1.7).

**In Python:**

```python
# the compass codebook, used for both pieces
C = [(1, 0), (0, 1), (-1, 0), (0, -1)]
def dot(a, b):
    return sum(a_i * b_i for a_i, b_i in zip(a, b))
def sq_dist(a, b):
    return sum((a_i - b_i) ** 2 for a_i, b_i in zip(a, b))
x = [0.9, 0.1, -0.2, 0.8]
pieces = [x[0:2], x[2:4]]
# c_j(x)
codes = [min(range(4), key=lambda c: sq_dist(piece, C[c])) for piece in pieces]
codes  # → [0, 1]
q = [1, 0, 0, 1]
# T_j[c] = q^(j) · C_j[c]
T = [[dot(q_j, C[c]) for c in range(4)] for q_j in (q[0:2], q[2:4])]
T  # → [[1, 0, -1, 0], [0, 1, 0, -1]]
# ŝ = Σ_j T_j[c_j(x)]
sum(T[j][codes[j]] for j in range(2))  # → 2
# the exact score, for comparison
round(dot(q, x), 2)  # → 1.7
```

**In code:** `ProductQuantizer` learns the codebooks (`ProductQuantizer.train`),
rounds vectors to codes (`ProductQuantizer.encode`), builds the Tⱼ tables
(`ProductQuantizer.lookup_table`) and sums the lookups
(`ProductQuantizer.adc_scores`). `tiny_pq_example` replays the 4-number
example by hand-setting the compass codebooks.

**Why it matters:** a 768-dimension float32 vector is 3,072 bytes; with
m = 96 it's 96 bytes, 32× smaller, so a billion vectors fit in about 100 GB
instead of 3 TB. The lost accuracy is recovered by **re-scoring**: take the
top few hundred by PQ score and recompute their exact scores from the full
vectors kept on disk. Real systems combine PQ with IVF (**IVF-PQ**) and
encode each vector's *residual* (its offset from its cluster centroid),
which is smaller and so rounds more precisely.

![Bytes per vector vs. recall, with and without re-scoring](figures/primer.ml.embeddings.ann.pq_tradeoff.svg)

**Reading it:** left to right, each vector gets more bytes (less
compression). The orange line uses PQ scores alone: at 8 bytes per vector
(32× smaller than the raw 256 bytes) it finds under half of the true top 10.
The blue line re-scores PQ's top 100 with the exact vectors and is near
perfect from 8 bytes up. The lesson: PQ is excellent at *shortlisting* and
poor at *final ranking*, so production systems always re-score.

**In code:** `PQIndex` scans every PQ code and can re-score its top
candidates with the exact vectors; `IVFPQIndex` combines IVF lists with
PQ-encoded residuals.

## 4. HNSW: highway, main roads, side streets

**The everyday picture.** Back to the traveller: highways to get close fast,
main roads to get closer, side streets to find the door. **HNSW**
(*Hierarchical Navigable Small World*) builds exactly that as a **graph**: a
set of points (**nodes**) joined by links (**edges**). Every vector is a node
on the bottom layer (the side streets). A random few are also placed on
layer 1 (main roads), fewer still on layer 2 (highways), and so on.

**On the eight points.** Layer 1 holds only A, E and G. Layer 0 holds all
eight, linked to nearby points (the thin lines in the figure above). A
**greedy search** means "always move to whichever neighbor is closest to the
target; stop when none is closer". Start at A:

| Step | Layer | At | Neighbors checked (squared distance) | Move? |
|---|---|---|---|---|
| 1 | 1 | A (41) | E (5), G (10) | → E |
| 2 | 1 | E (5) | A (41), G (10) | no closer neighbor: drop a layer, staying at E |
| 3 | 0 | E (5) | B (18), D (17), F (4), G (10), H (1) | → H |
| 4 | 0 | H (1) | E (5), F (4), G (10) | no closer neighbor: done. **Answer: H** |

```mermaid
flowchart TD
  subgraph L2["Top layer: few nodes, long jumps"]
    A2[A] --- D2[D]
  end
  subgraph L1["Middle layer: more nodes"]
    A1[A] --- B1[B] --- D1[D] --- E1[E]
  end
  subgraph L0["Bottom layer: every vector"]
    A0[A] --- B0[B] --- C0[C] --- D0[D] --- E0[E] --- F0[F]
  end
  D2 -.-> D1
  E1 -.-> E0
```

**Reading it:** three layers of the same kind of map, fewer nodes the higher
you go. Solid lines are links within a layer; dotted arrows are "the same
node, one layer down". A search enters at the top, crosses a lot of ground
in a few long jumps (A to D), then follows a dotted arrow down and continues
from the same node with finer steps, until the bottom layer, where every
vector lives.

**In code:** `tiny_hnsw_search` replays the greedy walk from the table above
on the eight-point map and returns every stop; `HNSWIndex` is the full index
used on real vectors.

### The search, step by step

```mermaid
flowchart TD
  S[Start at the entry point<br/>on the top layer] --> G{Is any neighbor<br/>closer to the query?}
  G -->|yes| H[Hop to the closest neighbor] --> G
  G -->|no| B{On the bottom layer?}
  B -->|no| DOWN[Drop one layer,<br/>same node] --> G
  B -->|yes| BEAM["Beam search: keep the best efSearch<br/>candidates, expand the most promising<br/>until nothing better turns up"]
  BEAM --> K[Return the top k]
```

**Reading it:** the loop at the top is greedy descent: hop while something
closer exists, else drop a layer. On the upper layers it keeps a single best
node. On the bottom layer it switches to a **beam search**, which keeps a
shortlist of the best `efSearch` nodes found so far, not just one, and keeps
exploring from the most promising. That wider net is what protects against
getting stuck at a point that is only *locally* the best. `efSearch` is the
dial you tune at query time.

![One HNSW query on 300 points, layer by layer](figures/primer.ml.embeddings.ann.hnsw_search_path.svg)

**Reading it:** the same 300 points, drawn three times: the top layer on the
left (14 nodes), the middle (64) and the bottom (all 300). Grey lines are the
graph's links. The red path is one real query (yellow star) run by the code
below; the red circle marks where the search entered each layer. On the left
it covers most of the map in a couple of long hops. By the bottom layer it
is already next to the star, and it only explores a small neighborhood. Out
of 300 points, it looked at a few dozen.

**In code:** `HNSWIndex.search` runs the greedy descent and the bottom-layer
beam search; `HNSWIndex.search_trace` does the same and returns every node it
expanded, which is what this figure draws.

### How the layers are built

```mermaid
flowchart TD
  N[New vector] --> LV["Draw its top layer ℓ at random<br/>(most get 0, a few get more)"]
  LV --> DESC[Greedy descent from the entry point<br/>down to layer ℓ]
  DESC --> FIND["On each layer ≤ ℓ: beam search with<br/>efConstruction to find candidates"]
  FIND --> SEL["Keep up to M diverse neighbors<br/>(the heuristic below)"]
  SEL --> LINK[Link both ways]
  LINK --> PRUNE{Neighbor now has too many links?<br/>more than M, or 2·M on layer 0}
  PRUNE -->|yes| TRIM[Re-select its best-spread links]
  PRUNE -->|no| DONE[Next layer down]
  TRIM --> DONE
```

**Reading it:** inserting a vector is a search followed by wiring. The random
draw at the top decides how many layers the node lives on. The search part
is the same as a query, just with a wider beam (`efConstruction`) for a
better-quality result. The wiring part keeps each node's links to a fixed
budget, so the graph never gets too dense to walk quickly.

**The diversity heuristic.** When choosing a new node's M neighbors, go
through the candidates from nearest to farthest and keep one only if it is
closer to the new node than to every neighbor already kept. Otherwise a kept
neighbor already "covers" that direction. Without this rule, in clustered
data all M links would point into the same clump, and the graph could split
into islands a greedy walk can't cross. With it, links fan out in different
directions and long "bridges" between clusters survive.

The random layer draw is where the math comes in:

$$
\ell = \left\lfloor -\ln(U) \cdot m_L \right\rfloor, \qquad m_L = \frac{1}{\ln M}, \qquad P(\ell \ge l) = M^{-l}
$$

**Symbols**

| Symbol | Meaning here | Range |
|---|---|---|
| ℓ | the top layer the new node will live on | 0, 1, 2, … |
| U | a random number drawn uniformly | 0 < U ≤ 1 |
| ln | natural logarithm: the power you raise e ≈ 2.718 to in order to get the number. ln(1) = 0; ln of a number below 1 is negative, so −ln(U) is positive | |
| ⌊ ⌋ | "floor": round down to a whole number | |
| M | the link budget per node (also sets how fast layers thin out) | 4 to 64; 16 is common |
| m_L | the level multiplier, 1/ln M | ≈ 0.36 for M = 16 |
| P(ℓ ≥ l) | the probability a node reaches layer l or higher | |
| efConstruction | beam width while building | 100 to 400 |
| efSearch | beam width while searching, the recall dial | ≥ k; 50 to 500 |

**In words:** draw a random number, take minus its logarithm, scale it by
one over the log of M, and round down. That gives a node layer 1 or higher
with probability 1/M, layer 2 or higher with probability 1/M², and so on.

**On the example:** M = 16, so m_L = 1/ln 16 = 1/2.773 = 0.361. A draw of
U = 0.5 gives −ln 0.5 · 0.361 = 0.693 · 0.361 = 0.25 → floor → **layer 0**.
A draw of U = 0.05 gives 2.996 · 0.361 = 1.08 → **layer 1**. Only
1/16 = 6.25% of nodes reach layer 1, and 1/256 ≈ 0.4% reach layer 2: each
layer has about 1/M as many nodes as the one below, which is what makes the
upper layers "highways".

**In Python:**

```python
import math
M = 16
# m_L = 1 / ln M
m_L = 1 / math.log(M)
round(m_L, 3)  # → 0.361
for U in (0.5, 0.05):
    # -ln(U) · m_L
    scaled = -math.log(U) * m_L
    # ... then round down to get ℓ
    print(U, round(scaled, 2), math.floor(scaled))  # → 0.5 0.25 0 0.05 1.08 1
# P(ℓ ≥ l) = M^(-l): 6.25% and about 0.4%
[M ** -l for l in (1, 2)]  # → [0.0625, 0.00390625]
```

| Knob | Set when | Higher means |
|---|---|---|
| `M` | build | more links per node: better recall, more memory, slower build. 16 is a common default; 32 to 64 for high-dimensional data |
| `efConstruction` | build | a better-quality graph, a slower build |
| `efSearch` | **query time** | a wider beam: better recall, slower queries |

**In code:** `HNSWIndex.add` inserts each vector this way: draw its layer,
beam-search each layer with efConstruction, keep up to M diverse neighbors
and link both ways. `HNSWIndex.layer_sizes` counts the nodes on each layer,
and `HNSWIndex.memory_bytes` adds up the vectors plus their links.

**Why it matters:** HNSW is the default index in most vector databases
because it gives high recall at low latency with no training step. Its costs
are memory (every vector *plus* its links must sit in RAM) and slow builds
for very large collections. At billions of vectors, teams switch to IVF-PQ,
shard HNSW across machines, or use disk-based graphs (DiskANN).

## Turning the dial: recall vs. work

![Recall vs. share of the corpus compared, for HNSW and IVF](figures/primer.ml.embeddings.ann.recall_vs_work.svg)

**Reading it:** each point is one setting of the dial (the small labels are
efSearch for HNSW and nprobe for IVF). The x-axis is the share of the whole
collection compared per query, on a log scale (each tick is 10× more work),
and the dashed line at 100% is a flat scan. The y-axis is recall@10. Read it
as a menu: the further up and to the left, the better. HNSW reaches about 0.95
recall while comparing around a fifth of these 2,000 vectors. On real
collections of millions the share is far smaller, because the number of
hops grows only slowly with N. Both curves rise steeply and
then flatten, which is why the last few points of recall are the expensive
ones.

$$
\text{recall@}k = \frac{\lvert \text{returned top } k \;\cap\; \text{true top } k \rvert}{k}
$$

**Symbols**

| Symbol | Meaning here |
|---|---|
| k | how many results we ask for |
| true top k | the answer from an exact flat search |
| ∩ | "intersection": the items in both lists |
| \| \| | "the number of items in" |

**In words:** recall@k is the share of the true top k that the index
actually returned.

**On the example:** if the true top 10 is documents 1 to 10 and the index
returns 1 to 9 plus document 42, the overlap is 9, so recall@10 = 9/10 =
**0.9**.

**In Python:**

```python
# documents 1 to 10
true_top = set(range(1, 11))
# 1 to 9, plus document 42
returned = set(range(1, 10)) | {42}
k = 10
# |returned ∩ true| / k
len(returned & true_top) / k  # → 0.9
```

**In code:** `recall_at_k` computes this share; `ground_truth` runs the exact
flat search that supplies the true top k, and `evaluate` reports recall,
latency and distance computations per query for any index.

**Why it matters:** recall and latency are traded against each other on
every index. The only reliable way to pick a setting is to measure recall@k
against a flat index *on your own vectors* while you turn the dial, then
check the 95th-percentile latency.

## In 20 seconds
- **Flat** compares the query with everything: exact, O(N). It's the ground
  truth, and fine for small collections.
- **IVF** clusters ahead of time and searches the `nprobe` nearest clusters.
- **PQ** compresses each vector to a few bytes and scores by table lookup;
  re-score a shortlist with exact vectors.
- **HNSW** is a layered graph: long jumps on top, a careful beam search at
  the bottom. `efSearch` trades recall for latency; `M` and `efConstruction`
  set graph quality at build time. It's fast and accurate, but memory-hungry.
- Always measure **recall@k against a flat index** on your own data while you
  turn the dial.

## Self-test questions

**On the eight-point map, query at (5, 4): walk the HNSW search.**
Start at A on layer 1 (squared distance 41). Its layer-1 neighbors are E (5)
and G (10), so hop to E. No layer-1 neighbor of E is closer, so drop to
layer 0 at E. There, H (1) is closer, so hop to H. None of H's neighbors
beats 1, so the answer is H.

**PQ stores x = (0.9, 0.1, −0.2, 0.8) with the four compass directions as
each half's codebook. What are the codes, and what score does the query
(1, 0, 0, 1) get?**
Codes [0, 1]. The lookup tables are [1, 0, −1, 0] and [0, 1, 0, −1], so the
approximate score is 1 + 1 = 2, against an exact score of 0.9 + 0.8 = 1.7.

**Walk through an HNSW search and the knobs you'd tune.**
Enter at the top layer's entry point; greedily hop to whichever neighbor is
closest to the query until none is closer, then drop a layer at the same
node; at layer 0 run a beam search keeping `efSearch` candidates; return the
top k. Build-time knobs: M (links per node; memory and recall) and
efConstruction (graph quality; build time). Query-time knob: efSearch. Raise
it until recall@k against a flat index meets your target, then check p95
latency.

**Why does HNSW need a "diversity" rule when choosing neighbors?**
If a node linked to its M nearest points, in clustered data all M would sit
in one clump and the graph could split into islands that greedy search can't
cross. Keeping a candidate only if it's closer to the new node than to any
already-kept neighbor spreads links in different directions and keeps
bridges between clusters.

**Why do the upper layers of HNSW have so few nodes?**
Each node's top layer is drawn so that P(layer ≥ l) = M^−l. Each layer holds
about 1/M of the one below, like express stops on a subway line, so a few
long hops at the top cover the whole space.

**IVF: what happens as `nprobe` goes from 1 to `nlist`?**
Recall rises and the work grows roughly in proportion. At `nprobe = nlist`
it's an exact scan (plus the centroid comparisons).

**How does PQ score a vector without decompressing it?**
It precomputes, for each of the m pieces, the query piece's dot product with
all 256 codebook entries. A stored vector's score is then the sum of m table
lookups, one per code. The query stays exact; only the stored side is
approximate (asymmetric distance computation).

**You have 1 billion 768-dimension vectors. Why not HNSW over raw float32?**
The raw vectors alone are 10⁹ × 768 × 4 bytes ≈ 3 TB of RAM, before the
graph links. Use IVF-PQ (e.g. 64-byte codes ≈ 64 GB) with exact re-scoring
of a shortlist, shard the index across machines, or use a disk-based graph
index such as DiskANN.

**Why can an index that scores 0.99 recall on a benchmark do worse on your data?**
ANN performance depends on the data's structure. Uniformly random
high-dimensional vectors are the hardest case (all distances look alike),
while real embeddings are clustered. A benchmark with different structure,
dimension or size predicts little. Always measure on your own vectors.

## The papers behind this lesson

- **Malkov & Yashunin, *Efficient and robust approximate nearest neighbor
  search using Hierarchical Navigable Small World graphs* (2016).**
  https://arxiv.org/abs/1603.09320. Introduced HNSW: the layered graph, the
  exponential layer draw with m_L = 1/ln M, and the diversity heuristic for
  choosing neighbors, which together gave logarithmic-feeling search with
  state-of-the-art recall. [annotated companion](../../../papers/hnsw.html)
- **Jégou, Douze & Schmid, *Product Quantization for Nearest Neighbor
  Search* (IEEE TPAMI, 2011).** https://ieeexplore.ieee.org/document/5432202.
  Introduced product quantization, asymmetric distance computation and the
  IVF-ADC index (IVF with PQ-encoded residuals), the basis of billion-scale
  vector search.
- **Douze et al., *The Faiss library* (2024).** https://arxiv.org/abs/2401.08281.
  Describes the most widely used vector-search library and the design space
  of indexes (flat, IVF, PQ, HNSW and their combinations) this lesson walks
  through.

## Further reading
- FAISS wiki (index types, guidelines for choosing an index): https://github.com/facebookresearch/faiss/wiki
- Douze et al., *The Faiss library* (2024): https://arxiv.org/abs/2401.08281
- hnswlib, the reference HNSW implementation, and its parameter guide: https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
- ANN-Benchmarks (recall vs. queries per second across libraries): https://ann-benchmarks.com/
- Pinecone's illustrated guides to HNSW, IVF and PQ: https://www.pinecone.io/learn/series/faiss/
"""

from __future__ import annotations

import heapq
import math
import time

import numpy as np

from primer._show import banner, say, table, takeaway

# ---------------------------------------------------------------------------
# 0. Shared helpers
# ---------------------------------------------------------------------------


def normalize(X: np.ndarray) -> np.ndarray:
    """L2-normalize rows (or a single vector) so dot product = cosine similarity."""
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.where(norms == 0, 1, norms)


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores, best first.

    `argpartition` finds the top k in O(N) without fully sorting all N
    scores; then we sort only those k. Real libraries use a small heap.
    """
    k = min(k, len(scores))
    if k == 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx], kind="stable")]


def kmeans(X: np.ndarray, k: int, iters: int = 20, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Plain Lloyd's k-means. Returns (centroids (k, d), assignment (N,)).

    Used by IVF (coarse clusters) and PQ (per-sub-space codebooks).
    1. Start from k random data points.
    2. Assign every point to its nearest centroid.
    3. Move each centroid to the mean of its points. Repeat.
    Empty clusters are re-seeded from random points so k stays k.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    k = min(k, len(X))
    C = X[rng.choice(len(X), size=k, replace=False)].copy()
    assign = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        # Squared L2 distance via ||x||² - 2x·c + ||c||², all at once.
        d2 = (X**2).sum(1, keepdims=True) - 2 * X @ C.T + (C**2).sum(1)
        assign = d2.argmin(1)
        for j in range(k):
            members = X[assign == j]
            C[j] = members.mean(0) if len(members) else X[rng.integers(len(X))]
    return C, assign


class _Index:
    """Common bookkeeping: every index counts distance computations.

    'Distance computations per query' is the hardware-independent cost
    measure used in the ANN literature: it says how much of the database a
    query actually touched.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.ndist = 0  # running count of vector-vs-query comparisons

    def __len__(self) -> int:  # pragma: no cover - overridden
        raise NotImplementedError

    def memory_bytes(self) -> int:  # pragma: no cover - overridden
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 0b. The eight-point map: every index, small enough to check by hand
# ---------------------------------------------------------------------------
# Eight "houses" on a grid. On a flat map we use ordinary distance, and we
# keep it *squared* (dx² + dy²) so every number stays a whole number. Squaring
# doesn't change which house is nearest.

TOY_POINTS: dict[str, tuple[int, int]] = {
    "A": (0, 0), "B": (2, 1), "C": (4, 0), "D": (1, 3),
    "E": (3, 3), "F": (5, 2), "G": (2, 5), "H": (5, 5),
}

# A hand-built two-layer HNSW graph over the eight points.
# Layer 1 (the "highway"): only A, E and G, all linked to each other.
# Layer 0 (the "streets"): every point, linked to its nearby neighbours.
TOY_LAYERS: dict[int, dict[str, list[str]]] = {
    1: {"A": ["E", "G"], "E": ["A", "G"], "G": ["A", "E"]},
    0: {
        "A": ["B", "D"], "B": ["A", "C", "E"], "C": ["B", "F"], "D": ["A", "E", "G"],
        "E": ["B", "D", "F", "G", "H"], "F": ["C", "E", "H"], "G": ["D", "E", "H"], "H": ["E", "F", "G"],
    },
}


def _sqdist(p: tuple[float, float], q: tuple[float, float]) -> float:
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def tiny_hnsw_search(query: tuple[float, float], entry: str = "A") -> list[tuple[int, str, float]]:
    """Greedy HNSW search on the eight-point map. Returns every stop as (layer, point, squared distance).

    On each layer: look at the current point's neighbours; hop to the closest
    one if it beats where you stand; stop when none does; then drop to the
    layer below *at the same point*. (A real HNSW keeps a beam of `ef`
    candidates at layer 0 rather than a single one; with ef = 1 it is exactly
    this greedy walk.)
    """
    here = entry
    path: list[tuple[int, str, float]] = []
    for layer in sorted(TOY_LAYERS, reverse=True):
        path.append((layer, here, _sqdist(TOY_POINTS[here], query)))
        while True:
            best = min(TOY_LAYERS[layer][here], key=lambda p: _sqdist(TOY_POINTS[p], query))
            if _sqdist(TOY_POINTS[best], query) >= _sqdist(TOY_POINTS[here], query):
                break  # no neighbour is closer: a local best on this layer
            here = best
            path.append((layer, here, _sqdist(TOY_POINTS[here], query)))
    return path


# Two hand-picked IVF clusters: the four lower-left points and the four upper-right ones.
TOY_CLUSTERS: dict[str, list[str]] = {"left": ["A", "B", "C", "D"], "right": ["E", "F", "G", "H"]}


def tiny_ivf_search(query: tuple[float, float], nprobe: int = 1) -> tuple[list[str], list[str], str]:
    """IVF on the eight-point map: compare to the cluster centres, scan only the nearest `nprobe`.

    Returns (probed clusters, points scanned, nearest point found).
    """
    centroids = {
        name: tuple(np.mean([TOY_POINTS[p] for p in members], axis=0)) for name, members in TOY_CLUSTERS.items()
    }
    probed = sorted(centroids, key=lambda c: _sqdist(centroids[c], query))[:nprobe]
    scanned = [p for c in probed for p in TOY_CLUSTERS[c]]
    best = min(scanned, key=lambda p: _sqdist(TOY_POINTS[p], query))
    return probed, scanned, best


def tiny_pq_example() -> dict:
    """Product quantization on one 4-number vector, small enough to do by hand.

    Split x = (0.9, 0.1, -0.2, 0.8) into two halves. Each half is replaced by
    the nearest of four "catalogue" entries (the four compass directions),
    so the whole vector is stored as two small codes instead of four floats.
    """
    pq = ProductQuantizer(dim=4, m=2, nbits=2)
    compass = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
    pq.codebooks = np.stack([compass, compass])  # hand-set instead of learned by k-means
    x = np.array([[0.9, 0.1, -0.2, 0.8]], dtype=np.float32)
    q = np.array([1, 0, 0, 1], dtype=np.float32)
    codes = pq.encode(x)
    table = pq.lookup_table(q)
    return dict(
        codes=codes[0].tolist(),
        decoded=pq.decode(codes)[0].tolist(),
        table=table.tolist(),
        approx_score=float(pq.adc_scores(table, codes)[0]),
        exact_score=float(x[0] @ q),
    )


# ---------------------------------------------------------------------------
# 1. Flat: exact brute force (the ground truth)
# ---------------------------------------------------------------------------


class FlatIndex(_Index):
    """Exact search: score the query against every stored vector.

    O(N·d) per query. Always the reference when measuring ANN recall.
    """

    def __init__(self, dim: int):
        super().__init__(dim)
        self.X = np.zeros((0, dim), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        self.X = np.vstack([self.X, np.asarray(vectors, dtype=np.float32).reshape(-1, self.dim)])

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        scores = self.X @ np.asarray(query, dtype=np.float32)  # one dot product per stored vector
        self.ndist += len(self.X)
        ids = top_k(scores, k)
        return ids, scores[ids]

    def __len__(self) -> int:
        return len(self.X)

    def memory_bytes(self) -> int:
        return self.X.nbytes  # N · d · 4 bytes of float32


# ---------------------------------------------------------------------------
# 2. IVF: cluster, then search only the nearest clusters
# ---------------------------------------------------------------------------


class IVFIndex(_Index):
    """Inverted-file index: k-means coarse quantizer + per-cluster lists.

    Args:
        nlist: number of clusters. Rule of thumb ≈ sqrt(N).
        nprobe: clusters scanned per query, the recall/latency dial.
    """

    def __init__(self, dim: int, nlist: int = 64, nprobe: int = 4, seed: int = 0):
        super().__init__(dim)
        self.nlist, self.nprobe, self.seed = nlist, nprobe, seed
        self.centroids: np.ndarray | None = None
        self.lists: list[list[int]] = []
        self.X = np.zeros((0, dim), dtype=np.float32)

    def train(self, sample: np.ndarray) -> None:
        """Learn the clusters. Real systems train on a sample, then add everything."""
        self.centroids, _ = kmeans(sample, self.nlist, seed=self.seed)
        self.lists = [[] for _ in range(len(self.centroids))]

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32).reshape(-1, self.dim)
        if self.centroids is None:
            self.train(vectors)  # convenience: train on the first batch
        start = len(self.X)
        self.X = np.vstack([self.X, vectors])
        # Each vector goes on the list of its nearest centroid (by dot product,
        # since we search by dot product).
        for offset, c in enumerate((vectors @ self.centroids.T).argmax(1)):
            self.lists[c].append(start + offset)

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(query, dtype=np.float32)
        # Step 1: which clusters look closest? (nlist comparisons)
        centroid_scores = self.centroids @ q
        probe = top_k(centroid_scores, self.nprobe)
        # Step 2: scan only those clusters' members.
        cand = np.array([i for c in probe for i in self.lists[c]], dtype=int)
        self.ndist += len(self.centroids) + len(cand)
        if len(cand) == 0:
            return np.array([], dtype=int), np.array([])
        scores = self.X[cand] @ q
        order = top_k(scores, k)
        return cand[order], scores[order]

    def __len__(self) -> int:
        return len(self.X)

    def memory_bytes(self) -> int:
        return self.X.nbytes + self.centroids.nbytes + 8 * len(self.X)  # vectors + centroids + list ids


# ---------------------------------------------------------------------------
# 3. PQ: compress vectors to a few bytes, score by table lookup
# ---------------------------------------------------------------------------


class ProductQuantizer:
    """Splits d dims into m sub-spaces, each with a 2^nbits-entry codebook.

    - `ProductQuantizer.encode`: vector -> m small integer codes (1 byte each at nbits=8).
    - `ProductQuantizer.decode`: codes -> the concatenation of the chosen centroids (lossy).
    - `ProductQuantizer.lookup_table`: for a query, table[j, c] = q_j · centroid_j[c],
      so the approximate dot product with any code is sum_j table[j, code_j].
    """

    def __init__(self, dim: int, m: int = 8, nbits: int = 8, seed: int = 0):
        assert dim % m == 0, "dim must split evenly into m sub-vectors"
        self.dim, self.m, self.ksub, self.dsub, self.seed = dim, m, 2**nbits, dim // m, seed
        self.codebooks: np.ndarray | None = None  # (m, ksub, dsub)

    def _chunks(self, X: np.ndarray) -> np.ndarray:
        # (N, d) -> (N, m, dsub): the j-th slice is sub-vector j.
        return X.reshape(len(X), self.m, self.dsub)

    def train(self, X: np.ndarray) -> None:
        chunks = self._chunks(np.asarray(X, dtype=np.float32))
        books = []
        for j in range(self.m):
            C, _ = kmeans(chunks[:, j, :], self.ksub, iters=15, seed=self.seed + j)
            if len(C) < self.ksub:  # tiny training sets: pad by repeating
                C = np.vstack([C, C[np.arange(self.ksub - len(C)) % len(C)]])
            books.append(C)
        self.codebooks = np.stack(books)

    def encode(self, X: np.ndarray) -> np.ndarray:
        chunks = self._chunks(np.asarray(X, dtype=np.float32))
        codes = np.empty((len(X), self.m), dtype=np.uint8 if self.ksub <= 256 else np.uint16)
        for j in range(self.m):
            C = self.codebooks[j]
            d2 = (chunks[:, j, :] ** 2).sum(1, keepdims=True) - 2 * chunks[:, j, :] @ C.T + (C**2).sum(1)
            codes[:, j] = d2.argmin(1)  # nearest centroid in this sub-space
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        return np.concatenate([self.codebooks[j][codes[:, j]] for j in range(self.m)], axis=1)

    def lookup_table(self, q: np.ndarray) -> np.ndarray:
        """(m, ksub) table of partial dot products: the heart of ADC."""
        qc = np.asarray(q, dtype=np.float32).reshape(self.m, self.dsub)
        return np.einsum("jd,jkd->jk", qc, self.codebooks)

    def adc_scores(self, table: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Asymmetric distance computation: m lookups + adds per stored vector."""
        return table[np.arange(self.m), codes].sum(axis=1)


class PQIndex(_Index):
    """Flat scan over PQ codes. Tiny memory; approximate scores.

    `rerank`: re-score this many top candidates with the exact vectors
    (if you kept them, e.g. on disk). 0 = pure PQ.
    """

    def __init__(self, dim: int, m: int = 8, nbits: int = 8, rerank: int = 0, seed: int = 0):
        super().__init__(dim)
        self.pq = ProductQuantizer(dim, m, nbits, seed)
        self.rerank = rerank
        self.codes = np.zeros((0, m), dtype=np.uint8)
        self.X = np.zeros((0, dim), dtype=np.float32)  # only used for re-scoring

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32).reshape(-1, self.dim)
        if self.pq.codebooks is None:
            self.pq.train(vectors)
        self.codes = np.vstack([self.codes, self.pq.encode(vectors)])
        self.X = np.vstack([self.X, vectors])

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        table = self.pq.lookup_table(query)
        approx = self.pq.adc_scores(table, self.codes)
        self.ndist += len(self.codes)  # cheap lookups, but still touches all N
        if not self.rerank:
            ids = top_k(approx, k)
            return ids, approx[ids]
        shortlist = top_k(approx, max(k, self.rerank))
        exact = self.X[shortlist] @ np.asarray(query, dtype=np.float32)
        self.ndist += len(shortlist)
        order = top_k(exact, k)
        return shortlist[order], exact[order]

    def __len__(self) -> int:
        return len(self.codes)

    def memory_bytes(self) -> int:
        # Only the codes (and codebooks) must live in RAM; re-scoring vectors
        # can stay on disk. That is the entire point of PQ.
        return self.codes.nbytes + self.pq.codebooks.nbytes


class IVFPQIndex(_Index):
    """IVF coarse clustering + PQ-encoded *residuals* (vector minus its centroid).

    Score decomposes exactly as q·x = q·c + q·r, and PQ approximates q·r.
    Residuals are small and centered, so the same bytes buy more precision
    than encoding raw vectors. This is FAISS's `IndexIVFPQ`.
    """

    def __init__(self, dim: int, nlist: int = 64, nprobe: int = 4, m: int = 8, nbits: int = 8, rerank: int = 0, seed: int = 0):
        super().__init__(dim)
        self.nlist, self.nprobe, self.rerank, self.seed = nlist, nprobe, rerank, seed
        self.pq = ProductQuantizer(dim, m, nbits, seed)
        self.centroids: np.ndarray | None = None
        self.lists: list[list[int]] = []
        self.assign = np.zeros(0, dtype=int)
        self.codes = np.zeros((0, m), dtype=np.uint8)
        self.X = np.zeros((0, dim), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32).reshape(-1, self.dim)
        if self.centroids is None:
            self.centroids, _ = kmeans(vectors, self.nlist, seed=self.seed)
            self.lists = [[] for _ in range(len(self.centroids))]
            a = (vectors @ self.centroids.T).argmax(1)
            self.pq.train(vectors - self.centroids[a])
        a = (vectors @ self.centroids.T).argmax(1)
        start = len(self.X)
        for offset, c in enumerate(a):
            self.lists[c].append(start + offset)
        self.assign = np.concatenate([self.assign, a])
        self.codes = np.vstack([self.codes, self.pq.encode(vectors - self.centroids[a])])
        self.X = np.vstack([self.X, vectors])

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(query, dtype=np.float32)
        cscores = self.centroids @ q
        probe = top_k(cscores, self.nprobe)
        cand = np.array([i for c in probe for i in self.lists[c]], dtype=int)
        self.ndist += len(self.centroids) + len(cand)
        if len(cand) == 0:
            return np.array([], dtype=int), np.array([])
        table = self.pq.lookup_table(q)  # the residual table is query-only, shared by all lists
        approx = cscores[self.assign[cand]] + self.pq.adc_scores(table, self.codes[cand])
        if not self.rerank:
            order = top_k(approx, k)
            return cand[order], approx[order]
        short = cand[top_k(approx, max(k, self.rerank))]
        exact = self.X[short] @ q
        self.ndist += len(short)
        order = top_k(exact, k)
        return short[order], exact[order]

    def __len__(self) -> int:
        return len(self.codes)

    def memory_bytes(self) -> int:
        return self.codes.nbytes + self.pq.codebooks.nbytes + self.centroids.nbytes + 8 * len(self.codes)


# ---------------------------------------------------------------------------
# 4. HNSW: hierarchical navigable small-world graph
# ---------------------------------------------------------------------------


class HNSWIndex(_Index):
    """HNSW graph index (Malkov & Yashunin, 2016), written for readability.

    Similarity is the dot product (vectors assumed L2-normalized), so
    "closer" means "higher score" everywhere below.

    Args:
        M: links per node on upper layers; layer 0 allows M0 = 2·M.
        ef_construction: beam width while inserting (graph quality).
        ef_search: beam width while querying (the recall/latency dial).
    """

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 100, ef_search: int = 50, seed: int = 0):
        super().__init__(dim)
        self.M, self.M0 = M, 2 * M
        self.ef_construction, self.ef_search = ef_construction, ef_search
        # mL normalizes the level distribution: P(level >= l) = M^-l, so each
        # layer has ~1/M as many nodes as the one below (like a skip list).
        self.mL = 1.0 / math.log(M)
        self.rng = np.random.default_rng(seed)
        self._X = np.zeros((16, dim), dtype=np.float32)  # grows by doubling
        self.n = 0
        # links[node][layer] -> list of neighbor ids on that layer.
        self.links: list[list[list[int]]] = []
        self.levels: list[int] = []
        self.entry: int | None = None
        self.max_level = -1

    # -- small utilities --------------------------------------------------

    @property
    def X(self) -> np.ndarray:
        return self._X[: self.n]

    def _sims(self, q: np.ndarray, ids: list[int]) -> np.ndarray:
        """Similarity of q to several nodes at once (and count the work)."""
        self.ndist += len(ids)
        return self._X[ids] @ q

    def _random_level(self) -> int:
        # Exponentially decaying: floor(-ln U · mL). With M=16, ~94% of nodes
        # are level 0, ~6% reach level 1, ~0.4% level 2...
        return int(-math.log(1.0 - self.rng.random()) * self.mL)

    # -- the core routine: beam search on one layer -----------------------

    def _search_layer(
        self, q: np.ndarray, entry_points: list[int], ef: int, layer: int, trace: list | None = None
    ) -> list[tuple[float, int]]:
        """Best-first beam search on a single layer. Returns [(sim, id)], best first.

        Two heaps:
          * `candidates`: nodes still to expand, best first (a max-heap, stored
            as negated sims because heapq is a min-heap).
          * `results`: the best `ef` nodes seen, worst on top (a min-heap), so
            we can cheaply evict the worst when something better shows up.
        Stop when the best unexpanded candidate is worse than the worst
        result: nothing reachable from here can improve the beam.
        """
        visited = set(entry_points)
        sims = self._sims(q, entry_points)
        candidates = [(-s, e) for s, e in zip(sims.tolist(), entry_points)]
        heapq.heapify(candidates)
        results = [(s, e) for s, e in zip(sims.tolist(), entry_points)]
        heapq.heapify(results)
        while len(results) > ef:
            heapq.heappop(results)

        while candidates:
            neg_s, c = heapq.heappop(candidates)
            if -neg_s < results[0][0]:
                break  # closest remaining candidate is worse than our worst result
            if trace is not None:
                trace.append((layer, c, -neg_s))  # record each node we expand, for figures
            fresh = [n for n in self.links[c][layer] if n not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            for s, n in zip(self._sims(q, fresh).tolist(), fresh):
                if len(results) < ef or s > results[0][0]:
                    heapq.heappush(candidates, (-s, n))
                    heapq.heappush(results, (s, n))
                    if len(results) > ef:
                        heapq.heappop(results)
        return sorted(results, reverse=True)

    def _select_neighbors(self, candidates: list[tuple[float, int]], M: int) -> list[int]:
        """HNSW's diversity heuristic (Algorithm 4 in the paper).

        Walk candidates from closest to farthest. Keep candidate e only if it
        is closer to the base point than to every neighbor kept so far.
        Otherwise some kept neighbor already "covers" e's direction, and
        greedy search can reach e through it. The result: links that fan out
        in different directions, including long "bridges" between clusters.
        """
        selected: list[int] = []
        for s, e in candidates:
            if len(selected) >= M:
                break
            if not selected or float((self._X[selected] @ self._X[e]).max()) < s:
                selected.append(e)
        return selected

    # -- build ---------------------------------------------------------------

    def _insert(self, idx: int) -> None:
        q = self._X[idx]
        level = self._random_level()
        self.levels.append(level)
        self.links.append([[] for _ in range(level + 1)])

        if self.entry is None:  # the very first node
            self.entry, self.max_level = idx, level
            return

        # Phase 1: greedy descent (beam width 1) through layers above `level`.
        ep = [self.entry]
        for layer in range(self.max_level, level, -1):
            ep = [self._search_layer(q, ep, 1, layer)[0][1]]

        # Phase 2: on every layer this node lives on, find neighbors and link.
        for layer in range(min(level, self.max_level), -1, -1):
            found = self._search_layer(q, ep, self.ef_construction, layer)
            neighbors = self._select_neighbors(found, self.M)
            self.links[idx][layer] = neighbors
            max_degree = self.M0 if layer == 0 else self.M
            for nb in neighbors:
                nb_links = self.links[nb][layer]
                nb_links.append(idx)  # links are bidirectional
                if len(nb_links) > max_degree:
                    # Too many links: re-select the best-spread subset for nb.
                    sims = (self._X[nb_links] @ self._X[nb]).tolist()
                    ranked = sorted(zip(sims, nb_links), reverse=True)
                    self.links[nb][layer] = self._select_neighbors(ranked, max_degree)
            ep = [e for _, e in found]  # next layer starts from everything we found

        if level > self.max_level:  # new tallest node becomes the entry point
            self.entry, self.max_level = idx, level

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32).reshape(-1, self.dim)
        while self.n + len(vectors) > len(self._X):
            self._X = np.vstack([self._X, np.zeros_like(self._X)])
        for v in vectors:
            self._X[self.n] = v
            self.n += 1
            self._insert(self.n - 1)

    # -- query -----------------------------------------------------------------

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        if self.entry is None:
            return np.array([], dtype=int), np.array([])
        q = np.asarray(query, dtype=np.float32)
        ep = [self.entry]
        # Highways and main roads: greedy, one best node per layer.
        for layer in range(self.max_level, 0, -1):
            ep = [self._search_layer(q, ep, 1, layer)[0][1]]
        # Side streets: a wide beam at layer 0. ef must be at least k.
        found = self._search_layer(q, ep, max(self.ef_search, k), 0)[:k]
        return np.array([i for _, i in found], dtype=int), np.array([s for s, _ in found])

    def search_trace(self, query: np.ndarray, k: int = 10) -> list[tuple[int, int, float]]:
        """Same as `search`, but return every node expanded as (layer, id, sim).

        Plotting this shows the idea in one picture: a few long hops on the
        top layers, then a careful local search at the bottom.
        """
        q = np.asarray(query, dtype=np.float32)
        trace: list[tuple[int, int, float]] = []
        ep = [self.entry]
        for layer in range(self.max_level, 0, -1):
            ep = [self._search_layer(q, ep, 1, layer, trace)[0][1]]
        self._search_layer(q, ep, max(self.ef_search, k), 0, trace)
        return trace

    def __len__(self) -> int:
        return self.n

    def layer_sizes(self) -> list[int]:
        """How many nodes live on each layer (bottom first). Shrinks ~M× per layer."""
        return [sum(1 for lv in self.levels if lv >= layer) for layer in range(self.max_level + 1)]

    def memory_bytes(self) -> int:
        n_links = sum(len(nbrs) for node in self.links for nbrs in node)
        return self.X.nbytes + 4 * n_links  # vectors + 4-byte neighbor ids


# ---------------------------------------------------------------------------
# 5. Benchmarking: recall@k vs. flat, latency, work per query, memory
# ---------------------------------------------------------------------------


def clustered_vectors(n: int, dim: int, n_clusters: int = 50, spread: float = 0.35, seed: int = 0) -> np.ndarray:
    """Normalized vectors drawn around random cluster centers.

    Real embeddings are clustered (topics, languages, document types), not
    uniform. Uniform random high-dimensional data is the hardest case for
    ANN, because all distances look alike (the curse of dimensionality).
    """
    rng = np.random.default_rng(seed)
    centers = normalize(rng.standard_normal((n_clusters, dim)))
    which = rng.integers(n_clusters, size=n)
    return normalize(centers[which] + spread * rng.standard_normal((n, dim)) / np.sqrt(dim) * 3)


def planar_vectors(n: int, width: float = 0.6, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors in a small cap around the north pole, plus their 2-D (x, y).

    For points (x, y, 1) normalized with x and y small, ranking by dot product
    matches ranking by ordinary distance in the (x, y) plane. That lets us
    *draw* what an index does to high-dimensional data.
    """
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-width / 2, width / 2, size=(n, 2))
    return normalize(np.hstack([xy, np.ones((n, 1))])), xy


def recall_at_k(found: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of the true top-k that the index returned."""
    return len(set(found.tolist()) & set(truth.tolist())) / max(1, len(truth))


def evaluate(index: _Index, queries: np.ndarray, truth: list[np.ndarray], k: int = 10) -> dict[str, float]:
    """Mean recall@k, mean latency (ms) and distance computations per query."""
    index.ndist = 0
    t0 = time.perf_counter()
    recalls = [recall_at_k(index.search(q, k)[0], t) for q, t in zip(queries, truth)]
    ms = (time.perf_counter() - t0) * 1000 / len(queries)
    return dict(recall=float(np.mean(recalls)), ms=ms, ndist=index.ndist / len(queries))


def ground_truth(X: np.ndarray, queries: np.ndarray, k: int = 10) -> list[np.ndarray]:
    flat = FlatIndex(X.shape[1])
    flat.add(X)
    return [flat.search(q, k)[0] for q in queries]


def storage_estimate(n: int, dim: int, bytes_per_value: int = 4) -> float:
    """Raw vector storage in GB: n · dim · bytes. (10M × 1536 × 4 ≈ 61 GB.)"""
    return n * dim * bytes_per_value / 1e9


# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------


def figures() -> dict:
    """Plot this lesson's data. matplotlib is imported here, and only here,
    so the lesson itself needs nothing beyond NumPy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HNSW_C, IVF_C, PQ_C, MUTED, HOT = "#2563eb", "#059669", "#d97706", "#9ca3af", "#dc2626"
    figs = {}

    # --- 0. The eight-point map with both HNSW layers and the search path ----
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for layer, width, color in [(0, 1, MUTED), (1, 3.5, HNSW_C)]:
        for a, nbrs in TOY_LAYERS[layer].items():
            for b in nbrs:
                if a < b:
                    (xa, ya), (xb, yb) = TOY_POINTS[a], TOY_POINTS[b]
                    ax.plot([xa, xb], [ya, yb], color=color, lw=width, alpha=0.8, zorder=1,
                            label=None)
    ax.plot([], [], color=MUTED, lw=1, label="layer 0 links (every point)")
    ax.plot([], [], color=HNSW_C, lw=3.5, label="layer 1 links (A, E, G)")
    for name, (x, y) in TOY_POINTS.items():
        ax.scatter(x, y, s=160, color="white", edgecolors="#374151", lw=1.5, zorder=2)
        ax.text(x, y, name, ha="center", va="center", fontsize=9, weight="bold", zorder=3)
    query = (5, 4)
    path = tiny_hnsw_search(query)
    for (_, a, _), (_, b, _) in zip(path, path[1:]):
        if a != b:
            ax.annotate("", xy=TOY_POINTS[b], xytext=TOY_POINTS[a], zorder=4,
                        arrowprops=dict(arrowstyle="->", color=HOT, lw=2.5, shrinkA=9, shrinkB=9))
    ax.scatter(*query, marker="*", s=300, color="#facc15", edgecolors="black", zorder=5, label="query (5, 4)")
    ax.set_xlim(-0.7, 5.8)
    ax.set_ylim(-0.7, 5.8)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title("Eight points: highway hop A→E, street hop E→H")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    figs["toy_map"] = fig

    # --- 1. The recall/work trade-off for HNSW and IVF ----------------------
    n, dim = 2000, 32
    X = clustered_vectors(n, dim, seed=0)
    queries = clustered_vectors(50, dim, seed=1)
    truth = ground_truth(X, queries)
    hnsw = HNSWIndex(dim, M=12, ef_construction=60)
    hnsw.add(X)
    ivf = IVFIndex(dim, nlist=45)
    ivf.add(X)
    h_pts, i_pts = [], []
    for ef in (10, 15, 20, 30, 50, 80, 120, 200):
        hnsw.ef_search = ef
        r = evaluate(hnsw, queries, truth)
        h_pts.append((100 * r["ndist"] / n, r["recall"], ef))
    for nprobe in (1, 2, 3, 5, 8, 12, 20, 45):
        ivf.nprobe = nprobe
        r = evaluate(ivf, queries, truth)
        i_pts.append((100 * r["ndist"] / n, r["recall"], nprobe))
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for pts, color, label, knob in [(h_pts, HNSW_C, "HNSW", "ef"), (i_pts, IVF_C, "IVF", "nprobe")]:
        xs, ys, ks = zip(*pts)
        ax.plot(xs, ys, "o-", color=color, label=f"{label} (labels: {knob})")
        for x, y, kv in zip(xs, ys, ks):
            ax.annotate(str(kv), (x, y), textcoords="offset points", xytext=(4, -10), fontsize=7, color=color)
    ax.axvline(100, color=MUTED, ls="--")
    ax.text(95, 0.45, "flat scan:\nevery vector,\nrecall 1.0", ha="right", color=MUTED, fontsize=8)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("% of the corpus compared per query (log scale)")
    ax.set_ylabel("recall@10 vs. exact search")
    ax.set_title(f"Turning the dial: recall vs. work ({n:,} vectors, {dim}-d)")
    ax.legend(frameon=False, loc="lower right")
    figs["recall_vs_work"] = fig

    # --- 2. An HNSW search, layer by layer, on data we can draw -------------
    V, xy = planar_vectors(300, seed=0)
    g = HNSWIndex(3, M=6, ef_construction=40, ef_search=10, seed=0)
    g.add(V)
    qv, qxy = planar_vectors(1, seed=5)
    trace = g.search_trace(qv[0], k=5)
    shown = list(range(min(g.max_level, 2), -1, -1))
    fig, axes = plt.subplots(1, len(shown), figsize=(4 * len(shown), 4.2))
    for ax, layer in zip(np.atleast_1d(axes), shown):
        on = [i for i, lv in enumerate(g.levels) if lv >= layer]
        for i in on:
            for j in g.links[i][layer]:
                if i < j:
                    ax.plot(*xy[[i, j]].T, color=MUTED, lw=0.4, zorder=1)
        ax.scatter(*xy[on].T, s=12 if layer == 0 else 28, color="#374151", zorder=2)
        path = [node for lv, node, _ in trace if lv == layer]
        if path:
            ax.plot(*xy[path].T, "-o", color=HOT, lw=2, ms=5, zorder=3)
            ax.scatter(*xy[path[0]], s=90, facecolors="none", edgecolors=HOT, lw=2, zorder=4)
        ax.scatter(*qxy[0], marker="*", s=260, color="#facc15", edgecolors="black", zorder=5)
        ax.set_title(f"layer {layer}: {len(on)} nodes" + ("  (every vector)" if layer == 0 else ""))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    fig.suptitle("One HNSW query: long hops up top, a careful local search at the bottom (★ = query)")
    fig.tight_layout()
    figs["hnsw_search_path"] = fig

    # --- 3. IVF cells and which ones a query probes -------------------------
    V2, xy2 = planar_vectors(600, seed=2)
    ivf2 = IVFIndex(3, nlist=12, nprobe=2, seed=0)
    ivf2.add(V2)
    q2, qxy2 = planar_vectors(1, seed=9)
    cell = (V2 @ ivf2.centroids.T).argmax(1)
    probed = top_k(ivf2.centroids @ q2[0], 2)
    cxy = ivf2.centroids[:, :2] / ivf2.centroids[:, 2:3]  # back to plane coordinates
    fig, ax = plt.subplots(figsize=(5.6, 5))
    cmap = plt.get_cmap("tab20")
    for c in range(len(cxy)):
        members = xy2[cell == c]
        hit = c in probed
        ax.scatter(*members.T, s=14 if hit else 8, color=cmap(c % 20), alpha=1.0 if hit else 0.25, zorder=2)
    ax.scatter(*cxy.T, marker="X", s=80, color="black", zorder=3, label="centroids")
    ax.scatter(*cxy[probed].T, marker="X", s=160, color=HOT, zorder=4, label="probed (nprobe = 2)")
    ax.scatter(*qxy2[0], marker="*", s=260, color="#facc15", edgecolors="black", zorder=5, label="query")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title("IVF: 12 clusters, the query scans only the 2 nearest")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    figs["ivf_cells"] = fig

    # --- 4. PQ: bytes per vector vs. recall ---------------------------------
    n3, dim3 = 1200, 64
    X3 = clustered_vectors(n3, dim3, seed=3)
    q3 = clustered_vectors(40, dim3, seed=4)
    t3 = ground_truth(X3, q3)
    ms = (2, 4, 8, 16, 32)
    raw, rescored = [], []
    for m in ms:
        a, b = PQIndex(dim3, m=m), PQIndex(dim3, m=m, rerank=100)
        a.add(X3)
        b.pq = a.pq  # same codebooks: the only difference is the exact re-scoring step
        b.add(X3)
        raw.append(evaluate(a, q3, t3)["recall"])
        rescored.append(evaluate(b, q3, t3)["recall"])
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(ms, raw, "o-", color=PQ_C, label="PQ codes only")
    ax.plot(ms, rescored, "o-", color=HNSW_C, label="PQ shortlist of 100, re-scored exactly")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ms, [f"{m} B\n({dim3 * 4 // m}× smaller)" for m in ms])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(f"bytes per vector (raw float32 = {dim3 * 4} B)")
    ax.set_ylabel("recall@10")
    ax.set_title("Product quantization: memory vs. accuracy")
    ax.legend(frameon=False, loc="lower right")
    figs["pq_tradeoff"] = fig

    return figs


# ---------------------------------------------------------------------------
# 7. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo(n: int = 5000, dim: int = 64, n_queries: int = 100, k: int = 10) -> None:
    banner("0. Eight points on a map, query at (5, 4): every index by hand")
    table(["point", "position", "squared distance"],
          [(p, xy, _sqdist(xy, (5, 4))) for p, xy in TOY_POINTS.items()], floatfmt=".0f")
    say("Flat search reads all eight rows: the nearest is H (1). Now the shortcuts.")
    table(["layer", "at", "squared distance"], tiny_hnsw_search((5, 4)), floatfmt=".0f")
    say("HNSW: one highway hop (A to E), drop a layer, one street hop (E to H). Done.")
    probed, scanned, best = tiny_ivf_search((5, 4))
    say(f"IVF: nearest cluster is {probed[0]!r}; scan {', '.join(scanned)} only; best is {best}.")
    pq = tiny_pq_example()
    say(
        f"PQ: (0.9, 0.1, -0.2, 0.8) is stored as codes {pq['codes']} (decoded {pq['decoded']}). "
        f"Table-lookup score {pq['approx_score']:.1f} vs exact {pq['exact_score']:.1f}: close, not identical."
    )

    banner("0b. Why not just brute force? Storage and compute math")
    for N, d in [(1_000_000, 768), (10_000_000, 1536), (1_000_000_000, 768)]:
        print(f"{N:>13,} vectors × {d:4d} dims × 4 bytes = {storage_estimate(N, d):8.1f} GB raw; "
              f"flat query = {N * d / 1e9:6.1f} G multiply-adds")
    print()
    say("A flat scan touches every byte on every query. ANN indexes touch a tiny fraction.")

    X = clustered_vectors(n, dim, seed=0)
    queries = clustered_vectors(n_queries, dim, seed=1)
    truth = ground_truth(X, queries, k)

    banner(f"1. Build every index on {n:,} clustered {dim}-d vectors")
    builds = {}
    for name, idx in [
        ("Flat", FlatIndex(dim)),
        ("IVF (nlist=70)", IVFIndex(dim, nlist=70, nprobe=4)),
        ("PQ (m=16, 1 byte each)", PQIndex(dim, m=16)),
        ("IVF-PQ (+rerank 50)", IVFPQIndex(dim, nlist=70, nprobe=8, m=16, rerank=50)),
        ("HNSW (M=16, efC=100)", HNSWIndex(dim, M=16, ef_construction=100)),
    ]:
        t0 = time.perf_counter()
        idx.add(X)
        builds[name] = (idx, time.perf_counter() - t0)
    rows = []
    for name, (idx, secs) in builds.items():
        r = evaluate(idx, queries, truth, k)
        rows.append((name, f"{secs:.2f}s", r["recall"], f"{r['ms']:.2f}", f"{r['ndist']:.0f}", f"{idx.memory_bytes() / 1e6:.2f} MB"))
    table(["index", "build", f"recall@{k}", "ms/query", "dist/query", "memory"], rows, floatfmt=".3f")
    say(
        """
        Pure-Python timings are only relative (FAISS or hnswlib are ~100×
        faster), so watch "dist/query": the fraction of the database each
        query touched. HNSW reaches high recall while comparing against a
        small fraction of the vectors. PQ uses a fraction of the memory but
        its raw scores are approximate; re-scoring a shortlist with the exact
        vectors recovers most of the recall.
        """
    )

    hnsw = builds["HNSW (M=16, efC=100)"][0]
    banner("2. HNSW's layers (highway, main roads, side streets)")
    sizes = hnsw.layer_sizes()
    table(["layer", "nodes", "share"], [(i, s, s / sizes[0]) for i, s in enumerate(sizes)][::-1], floatfmt=".4f")
    say(f"mL = 1/ln(M) = {hnsw.mL:.3f}, so each layer holds ~1/M = {1 / hnsw.M:.3f} of the one below.")

    banner("3. The recall/latency dial: HNSW efSearch")
    rows = []
    for ef in (10, 20, 40, 80, 160):
        hnsw.ef_search = ef
        r = evaluate(hnsw, queries, truth, k)
        rows.append((ef, r["recall"], f"{r['ms']:.2f}", f"{r['ndist']:.0f}", f"{100 * r['ndist'] / n:.1f}%"))
    table(["efSearch", f"recall@{k}", "ms/query", "dist/query", "of corpus"], rows, floatfmt=".3f")

    banner("4. The same dial on IVF: nprobe")
    ivf = builds["IVF (nlist=70)"][0]
    rows = []
    for nprobe in (1, 2, 4, 8, 16, 70):
        ivf.nprobe = nprobe
        r = evaluate(ivf, queries, truth, k)
        rows.append((nprobe, r["recall"], f"{r['ms']:.2f}", f"{r['ndist']:.0f}"))
    table(["nprobe", f"recall@{k}", "ms/query", "dist/query"], rows, floatfmt=".3f")
    say("At nprobe = nlist, IVF scans everything and matches the flat index exactly.")

    banner("5. PQ: bytes per vector vs. recall (and why we re-score)")
    rows = []
    for m in (4, 8, 16, 32):
        for rerank in (0, 100):
            pq = PQIndex(dim, m=m, rerank=rerank)
            pq.add(X)
            r = evaluate(pq, queries, truth, k)
            rows.append((m, f"{dim * 4 // m}×", rerank or "-", r["recall"]))
    table(["m (bytes/vec)", "compression", "rerank top", f"recall@{k}"], rows, floatfmt=".3f")
    takeaway(
        "Every ANN index has a recall-vs-cost dial: efSearch for HNSW, nprobe for IVF, "
        "bytes-per-vector for PQ. Measure recall@k against a flat index on your own data while you turn it."
    )


if __name__ == "__main__":
    demo()
