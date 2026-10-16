r"""
# Clustering and everyday uses of embeddings: grouping, mapping, routing, caching

Run: `python -m primer.ml.embeddings.clustering`

New to vectors, distances or Σ? `primer.notation` builds them from zero.

## Level 1: The practitioner's guide

**In one sentence.** Clustering groups texts by meaning without being told
what the groups are, and the move behind it, one distance and one threshold,
also powers four everyday jobs: spotting near-duplicates, routing requests,
flagging anomalies and caching answers.

**When you need it.** You need clustering when you have a pile of unlabeled
text (support tickets, feedback, logs, documents) and want to know what
kinds of things are in it before anyone has named the kinds. You need the
everyday uses whenever the question is "have I seen something like this
before?": the same ticket filed twice, a request that belongs to another
team, an entry that looks like nothing else, a question already answered an
hour ago. You don't need clustering when the categories are already known
and labeled: that is classification, and a centroid per class or a trained
model does it directly. The tell: someone is reading tickets one by one to
find out what people are asking about, or a model is being called again for
a question it has already answered.

**Your options.** From the simplest to the most flexible:

| Option | What it does | What it gives you | What it costs | Where it lives |
|---|---|---|---|---|
| One distance and a threshold | Embeds the new item and compares it with what is already stored | Near-duplicate detection, routing, anomaly scores and a semantic cache, with no clustering at all | A threshold to calibrate on labeled pairs, plus the guards below | Your code over any embedding model; semantic-router, GPTCache |
| k-means | Alternates "assign each point to its nearest centre" and "move each centre to the mean of its points" | k crisp groups with a centre each; fast, and scikit-learn lists it for very large collections | You choose k, and it assumes round, similar-sized clusters | scikit-learn, FAISS (it is how IVF indexes cluster a corpus) |
| DBSCAN | Grows clusters through points with enough neighbours within a reach ε, leaving the rest as noise | The number of clusters found for you, and loners marked as noise | A reach no single value gets right; kinds chain together as it widens | scikit-learn |
| HDBSCAN | Runs DBSCAN at every reach at once and keeps the clusters that persist longest | Clusters of different densities, no ε to set, noise labelled | A minimum cluster size to pick; listed for large rather than very large collections | scikit-learn, the hdbscan library, BERTopic |
| Agglomerative (hierarchical) | Merges the closest pair of groups again and again into a tree | Many clusters, a cut at any level, any distance | You still choose where to cut the tree | scikit-learn |

**How to choose.** Start from whether you know how many groups there are
and whether every item belongs somewhere.

- You know the number of groups, or want a fixed number of buckets (topics
  for a dashboard, cells for a vector index): k-means, seeded with
  k-means++. Pick k by the silhouette, not the inertia: inertia falls at
  every k, while the silhouette peaks at the true number (k = 5 for this
  lesson's five ticket kinds).
- You don't know how many groups exist, some items belong nowhere, and the
  groups differ in shape and density (real tickets, logs): HDBSCAN. Plain
  DBSCAN is the same idea with one reach you have to guess, and on this
  lesson's tickets no reach gets everything right.
- You want a picture: PCA for an honest but lossy shadow (its title tells
  you what share of the variation you are seeing); UMAP or t-SNE for a
  prettier map of what is near what, never for measuring distances or
  cluster sizes.
- You want an action per item rather than groups: the threshold pattern.
  Flag near-duplicate pairs above a calibrated cosine, send a request to the
  route whose centroid clears the threshold, score an anomaly by its
  distance to the nearest centre, return a cached answer only above a strict
  threshold.
- Whatever you pick, calibrate every threshold on labeled pairs from your
  own data, check the clusters against a sample you have read, and keep the
  fallback: a router without one sends every unanswerable question
  somewhere.

**What it costs.** Everything here runs on embeddings you already have, so
the model cost is one embedding per item. k-means is the cheap one: a few
rounds of assign-and-move, restarted a few times, with the total squared
distance only ever falling. DBSCAN's naive form needs every pairwise
distance, which scikit-learn notes costs n² floats of memory when its tree
structures can't be used; HDBSCAN does more work than k-means for its freedom
from ε. The everyday uses cost one comparison against what is stored, and a
semantic cache pays for itself by skipping a model call on every hit, at the
risk of a wrong hit: GPTCache's documentation says plainly that a semantic
cache produces false positives on hits and false negatives on misses. The
real expense is labeling: a few dozen labeled pairs to calibrate a
threshold, and a sample of items read by a person to check that the
clusters mean what you think.

**What breaks.**

- **The wrong k.** Inertia keeps falling as k grows, so the lowest value is
  useless. Use the silhouette, or the elbow where inertia stops falling
  steeply.
- **Chaining.** Widen DBSCAN's reach until real tickets stop being noise and
  kinds merge through chains of near neighbours: at ε = 0.6 here the noise
  is exactly the two off-topic tickets, but VPN and printer tickets share a
  cluster. HDBSCAN keeps the clusters that persist across reaches instead.
- **Loners forced into a group.** k-means gives every point a cluster,
  including the coffee-machine and dog tickets that belong to none. Use a
  density method, or score anomalies by distance to the nearest centre.
- **Distances read off a UMAP or t-SNE plot.** Gaps between clusters and
  cluster sizes there don't reflect the real space. Use the map for
  intuition, measure in the original space.
- **A cache that answers a different question.** "How do I reset my VPN?"
  is about 0.46 similar to "How do I reset my password?" here: related, and
  wrong. Keep the threshold strict and calibrated.
- **A cache that leaks across users.** "What is my PTO balance?" means
  something different for each person. Filter by user, tenant and
  permission scope before the similarity search, so no threshold setting
  can leak an answer.
- **Stale hits.** Give every entry a time-to-live so answers age out when
  the facts change.
- **A router without a fallback.** Below the threshold, hand the request to
  a person or a general assistant rather than the least-bad route.

**In the wild.** scikit-learn ships k-means, DBSCAN, HDBSCAN and
agglomerative clustering with a guide to which fits which data (k-means for
even cluster sizes and flat geometry, DBSCAN and HDBSCAN for uneven sizes
and outlier removal, HDBSCAN when densities vary); the hdbscan library and
umap-learn are the reference implementations of their papers. FAISS runs
k-means to build the cells of an IVF index. BERTopic discovers topics by
chaining sentence-transformers embeddings, UMAP, HDBSCAN and a class-based
TF-IDF. semantic-router defines each route by example utterances, embeds
them, and returns no route at all when nothing matches, with thresholds you
can train. GPTCache embeds each query, searches a vector store and lets a
similarity evaluator decide the hit. The papers behind this lesson
(k-means++, DBSCAN, HDBSCAN and UMAP) are listed at the end, and Distill's
*How to Use t-SNE Effectively* shows the misreadings the map warning above
is about.

**Go deeper.** Level 2 builds k-means, the silhouette, DBSCAN, PCA and all
four everyday uses by hand, first on four points you can check with a
pencil and then on thirty support tickets, with every formula decoded and
every figure explained. If you only needed to choose a method and set its
thresholds, you are done.

## Level 2: How it works, from scratch

The examples throughout are 32 short support tickets. Thirty are about IT,
HR and finance, in five kinds with six tickets each: login, VPN, printer,
expenses and leave. The other two are off-topic, one about a coffee machine
and one about a dog. A small toy embedder (`primer.common.embedder`) turns
each ticket into a vector, and tickets that mean similar things get vectors
with a high cosine similarity (`primer.ml.embeddings.similarity`). Sort the
tickets by kind, compare every pair, and the groups are plain to see.

![Tickets of the same kind form five bright squares along the diagonal, and the two off-topic tickets are dark almost everywhere](figures/primer.ml.embeddings.clustering.similarity.svg)

**Reading it:** rows and columns are the 32 tickets in the same order,
grouped by kind, and a brighter cell means a higher cosine similarity. Each
bright square along the diagonal is one kind, whose six tickets resemble each
other and not the rest. The last two rows and columns, the off-topic
tickets, are dark almost everywhere.

A clustering method is not given that order. It gets the tickets shuffled,
with no kinds attached, and has to find the squares from the similarities
alone.

**In code:** `ticket_embeddings` returns the tickets' vectors, and
`ticket_kinds` the true kinds, which no clustering method sees.

## k-means: k meeting points

The simplest method decides the number of groups, k, in advance. Think of a
town that wants k post boxes placed so that everyone's walk to their nearest
box is as short as possible. Put the boxes anywhere to start. Everyone walks
to their nearest box, then each box moves to the middle of the people who
chose it, and the two steps repeat until nobody changes box. That is
**k-means**, with points for the people and centres for the boxes.

Take four points, (0, 0), (0, 1), (10, 0) and (10, 1), and k = 2. However
the boxes start, the two left points pick one box and the two right points
the other. Each box moves to the middle of its pair, (0, 0.5) and (10, 0.5).
Every point is now 0.5 from its box, so the total squared walk is
4 × 0.5² = **1.0**, and the next round changes nothing.

```mermaid
flowchart LR
  I["Place k centres<br/>(k-means++: spread out)"] --> A["Assign: each point<br/>joins its nearest centre"]
  A --> M["Move: each centre goes to<br/>the mean of its points"]
  M --> C{Did any centre move?}
  C -->|yes| A
  C -->|no| D[Done: labels + centres]
```

**Reading it:** follow the loop from assign to move and back, until no
centre moves. It always ends, because neither step can lengthen the total
squared walk: joining the nearest centre cannot make anyone's walk longer,
and a group's mean is the spot with the shortest total squared walk to its
points. Where it ends depends on the start. **k-means++** picks each new
starting centre far from the ones already chosen, so that two centres do not
begin in the same crowd.

The total squared walk is called **inertia**, and k-means makes it small:

$$
J = \sum_{i=1}^{N} \lVert x_i - \mu_{c(i)} \rVert^2
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| J | the total squared distance, called **inertia**; k-means makes it small | ≥ 0 |
| N | number of points | 4 in the example; 30 tickets |
| i | which point | 1 to N |
| xᵢ | the i-th point (an embedding) | d numbers |
| c(i) | the cluster point i is assigned to | 1 to k |
| μ_c (mu) | the **centroid** of cluster c: the mean of its points | d numbers |
| ‖·‖² | squared Euclidean distance | ≥ 0 |
| Σ | add up over all points | |

**In words:** add up, over every point, the squared distance from the point
to the centre of its cluster.

**On the example:** four points each 0.5 from their centre: J = 4 × 0.25 = 1.0.

**In Python:**

```python
x = [(0, 0), (0, 1), (10, 0), (10, 1)]
# the two centroids
mu = [(0, 0.5), (10, 0.5)]
# c(i): the cluster each point joined
c = [0, 0, 1, 1]
# ‖x_i - μ_c(i)‖²
J = sum((x_i[0] - mu[c_i][0]) ** 2 + (x_i[1] - mu[c_i][1]) ** 2
        # Σ over every point
        for x_i, c_i in zip(x, c))
J  # → 1.0
```

![k-means centres drift into the middle of their crowds round by round while the total squared distance only goes down](figures/primer.ml.embeddings.clustering.kmeans_steps.svg)

**Reading it:** three snapshots of k-means on 2-D points, read left to
right. Colours are the current assignments and black crosses are the
centres: the k-means++ starts, then the state after one assign-and-move
round, then the final state. The centres drift into the middle of their
crowds, and J, printed above each panel, falls from one panel to the next.

**In code:** `kmeans` runs the loop from k-means++ starts and keeps the best
of several restarts; `kmeans_history` records the rounds drawn here.

k-means is fast, and it reappears in vector search, where IVF indexes use it
to split a corpus into cells (`primer.ml.embeddings.ann`). It has two
weaknesses. You must choose k, and it assumes the clusters are round and of
similar size.

## Choosing k: the elbow and the silhouette

To choose k, run k-means for several values of k and score each result.
Inertia cannot be the score, because it falls every time k grows: more boxes
always mean shorter walks. What can be read from it is the **elbow**, the k
after which it stops falling steeply. A better score asks every point
whether it sits much closer to its own cluster than to the nearest other
one. That score is the **silhouette**.

Back to the four points, split into the two pairs. For (0, 0), its partner
is a = **1** away. The other pair is 10 and √101 = 10.05 away, on average
b = **10.025**. Its score is (b − a) / b = 9.025 / 10.025 = **0.900**. By
symmetry every point scores the same, so the silhouette is 0.900.

$$
s(i) = \frac{b(i) - a(i)}{\max\big(a(i), b(i)\big)}
$$

**Symbols**

| Symbol | Meaning here | Range |
|---|---|---|
| s(i) | silhouette score of point i | −1 to 1 |
| a(i) | average distance from i to the other points in its own cluster | ≥ 0 |
| b(i) | average distance from i to the points of the *nearest other* cluster | ≥ 0 |
| max(a, b) | the larger of the two, to scale the score into −1 … 1 | |

**In words:** how much farther the nearest other group is than your own,
as a share of the larger distance. The silhouette of a clustering is the
average over all points: near 1 is crisp, near 0 is overlapping, negative
means points sit in the wrong cluster.

**On the example:** (10.025 − 1) / 10.025 = 0.900.

**In Python:**

```python
import math
# a(i): distance to its partner
a = math.dist((0, 0), (0, 1))
# b(i): the other pair, averaged
b = (math.dist((0, 0), (10, 0)) + math.dist((0, 0), (10, 1))) / 2
a, round(b, 3)  # → (1.0, 10.025)
# s(i)
round((b - a) / max(a, b), 3)  # → 0.9
```

![Inertia falls at every k, so it cannot pick k, while the silhouette peaks clearly at the true k of 5](figures/primer.ml.embeddings.clustering.choose_k.svg)

**Reading it:** both panels run k-means on the 30 tickets for every k from 2
to 8. On the left, inertia falls at every k, steeply up to k = 5 and more
slowly after it. On the right, the silhouette peaks at k = 5,
the true number of ticket kinds. The elbow has to be judged by eye, while
the silhouette's peak is a single number to pick.

**In code:** `best_k_by_silhouette` runs `kmeans` for each candidate k and
keeps the one with the highest `silhouette`.

## Density clustering: DBSCAN and HDBSCAN

Even with k chosen well, k-means assumes round clusters, and it puts every
point in some cluster, so the coffee-machine ticket joins one of the five
kinds. Density clustering takes another view, the one you get of a festival from a drone. A crowd is
wherever people stand shoulder to shoulder, and a loner by the fence belongs
to no crowd. How many crowds there are is not decided in advance.
**DBSCAN** makes this a rule with two settings: a reach, ε, for how close
counts as shoulder to shoulder, and the number of points within reach that
make a crowd.

Take points on a line at 0, 0.1, 0.2, 5.0, 5.1, 5.2 and 20, with a reach of
ε = 0.15 and a crowd of at least 2. The points 0, 0.1 and 0.2 chain together,
each within 0.15 of the next, and so do 5.0, 5.1 and 5.2. Nothing is within
0.15 of 20. The result is two clusters and one noise point, labelled
**0 0 0 1 1 1 −1**.

$$
N_\varepsilon(x) = \{\, y : \text{dist}(x, y) \le \varepsilon \,\}, \qquad
x \text{ is a core point if } |N_\varepsilon(x)| \ge \text{minPts}
$$

**Symbols**

| Symbol | Meaning here |
|---|---|
| x, y | points |
| dist | distance: Euclidean, or 1 − cosine for embeddings |
| ε (epsilon) | the reach: how close counts as "shoulder to shoulder" |
| N_ε(x) | the ε-neighbourhood: every point within reach of x (x included) |
| {… : …} | "the set of … such that …" |
| \|·\| | how many points are in the set |
| minPts | how many points within reach make x a **core** point |

**In words:** a point's neighbourhood is everything within reach; a point
with enough neighbours is a core point; clusters grow outward from core
points, and anything no core point can reach is noise.

**On the example:** N₀.₁₅(0.1) = {0, 0.1, 0.2}, 3 ≥ 2, so 0.1 is core;
N₀.₁₅(20) = {20}, 1 < 2, so 20 is noise.

**In Python:**

```python
points = [0, 0.1, 0.2, 5.0, 5.1, 5.2, 20]
eps, minPts = 0.15, 2
# N_ε(x): every point within reach of x
def N(x):
    return [y for y in points if abs(x - y) <= eps]
# a core point
N(0.1), len(N(0.1)) >= minPts  # → ([0, 0.1, 0.2], True)
# nobody within reach: noise
N(20), len(N(20)) >= minPts  # → ([20], False)
```

```mermaid
flowchart TD
  P[Take an unvisited point] --> N{At least minPts<br/>within ε?}
  N -->|no| L[Leave it as noise for now]
  N -->|yes| S[Start a new cluster]
  S --> G[Add every neighbour;<br/>if a neighbour is also core,<br/>add its neighbours too]
  G --> G
  G --> P
  L --> P
```

**Reading it:** DBSCAN takes the points one at a time. A point with too few
neighbours is left as noise for now; a later crowd can still claim it as an
edge point. A core point starts a cluster, which floods outward through
other core points (the loop on the "Add" box) until it reaches the crowd's
edge.

**In code:** `dbscan` labels the clusters 0, 1, 2 and so on, and gives
noise the label −1.

![DBSCAN at reach 0.6 marks exactly the two off-topic tickets as noise but merges the VPN and printer tickets into one cluster](figures/primer.ml.embeddings.clustering.dbscan.svg)

**Reading it:** the tickets on a flat 2-D map, coloured by the clusters
DBSCAN found. Distance here is 1 − cosine, so ε = 0.6 means "cosine at least
0.4", and minPts = 3. The grey crosses are noise: the two off-topic tickets
and no others, found without telling DBSCAN how many clusters there are.
Now look at the VPN and printer tickets, which share one cluster. A
single pair joins them: "vpn client fails to start" and "printing fails with
an error" are within reach of each other.

![No reach gets DBSCAN right: small reaches strand real tickets as noise, and by the time noise is only the off-topic pair the kinds have begun chaining together](figures/primer.ml.embeddings.clustering.eps_sweep.svg)

**Reading it:** the horizontal axis is the reach ε. The blue line counts
clusters and the orange line counts noise points; the dashed line marks the
two off-topic tickets. At the smallest reach, 0.45, most tickets have too
few neighbours, and DBSCAN finds three clusters and calls 22 of the 32
tickets noise. At 0.5 and 0.55 it finds all five kinds, but calls 15, then
9, tickets noise, most of them genuine. At 0.6 the noise is down to the
off-topic pair, but the kinds have started **chaining**: A is near B and B
is near C, so A and C end up in one cluster. VPN and printer merge, leaving
four clusters. Wider still, the chains leave three clusters, then two, until
at 0.825 one cluster holds every kind and one off-topic ticket as well. No
single ε gets every ticket right, and real data has no labels to show where
the best one is.

**HDBSCAN** removes ε. In effect it runs DBSCAN at every reach, builds a
tree of how crowds merge as the reach grows, and keeps the crowds that last
over the widest range of reaches. That lets it find dense and loose clusters
in the same data, which is what messy text such as support tickets needs.

## Seeing the map: PCA (and why UMAP and t-SNE mislead on distance)

The DBSCAN figure drew the tickets, each a list of 128 numbers, on a flat
page. Any such map is a shadow. A 3-D object casts a 2-D shadow on a wall,
and turning the object changes the shadow. **PCA** (principal component analysis) turns the data so
that its shadow is as spread out as possible, which keeps as much of the
original variation as a flat picture can.

Take three points on a straight line, (1, 1), (2, 2) and (3, 3). All their
variation runs along the diagonal, so the first direction explains **100%**
of it and the second **0%**. Along that direction the points sit
√2 = **1.414** apart, as they do in the original. PCA scores each direction
by its share of the total spread:

$$
\text{explained}_j = \frac{\sigma_j^2}{\sum_k \sigma_k^2}
$$

**Symbols**

| Symbol | Meaning here | Range |
|---|---|---|
| j | which direction (principal component) | 1 to d |
| σⱼ (sigma) | the j-th singular value of the centred data, from the SVD (see `primer.notation`) | ≥ 0, largest first |
| σⱼ² | proportional to the **variance** (average squared spread) along direction j | ≥ 0 |
| Σₖ σₖ² | the total spread over all directions k | |
| explainedⱼ | the share of all the spread that direction j captures | 0 to 1 |

**In words:** square each direction's singular value and divide by the sum
of all the squares.

**On the example:** centred, the points are (−1, −1), (0, 0), (1, 1). Their
singular values are 2 along the diagonal and 0 across it, so the shares are
2² / (2² + 0²) = 1 and 0 / 4 = 0.

**In Python:**

```python
import math
centred = [(-1, -1), (0, 0), (1, 1)]
# along the diagonal
u = [(1 / math.sqrt(2), 1 / math.sqrt(2)),
     # across it (the SVD finds these; here we know them)
     (1 / math.sqrt(2), -1 / math.sqrt(2))]
# spread along u_j
sigma = [math.sqrt(sum((x * u_j[0] + y * u_j[1]) ** 2 for x, y in centred))
         for u_j in u]
[round(sigma_j, 3) for sigma_j in sigma]  # → [2.0, 0.0]
[round(sigma_j ** 2 / sum(sigma_k ** 2 for sigma_k in sigma), 3) for sigma_j in sigma]  # → [1.0, 0.0]
```

![Squashed to 2-D, each kind of ticket forms its own patch with its k-means centre inside it](figures/primer.ml.embeddings.clustering.map.svg)

**Reading it:** the 128-number ticket embeddings flattened to 2-D with PCA
and coloured by their true kind. Black crosses are the k-means centres,
projected the same way, and grey ✕ markers are the off-topic tickets. Each
kind forms its own patch, with a centre inside it. The title gives the share
of the variation the two axes keep. The rest is invisible, so two tickets
that look close on this map can be far apart in the full space.

**In code:** `pca` projects the centred data onto its top directions and
returns each direction's share.

UMAP and t-SNE draw maps by a different rule. They keep each point's
*neighbours* next to it and let every other distance stretch. Like a subway
map, the result shows what is near what, but not how far apart things are or
how big they are: the gaps between clusters and the sizes of clusters on
such a map do not match the real space. Read them for neighbours, and
measure in the full space.

## Everyday uses

Clustering looks at a whole collection at once. Most systems also have to
decide about each new item as it arrives, and the same closeness answers
that question: is this like something already seen? Four everyday jobs ask
it, spotting **near-duplicates**, **routing** requests to the right team,
flagging **anomalies** and **caching** answers.

```mermaid
flowchart LR
  T[New text] --> E[Embed]
  E --> D{Nearly identical to<br/>something stored?}
  D -->|yes| DUP[Near-duplicate:<br/>merge or skip]
  E --> R{Closest route centroid<br/>above threshold?}
  R -->|yes| ROUTE[Send to that team/tool]
  R -->|no| HUMAN[Fallback: human or<br/>general assistant]
  E --> A{Far from every<br/>cluster centre?}
  A -->|yes| ANOM[Flag as anomaly]
  E --> C{Cached question, same context,<br/>not expired, above threshold?}
  C -->|yes| HIT[Return cached answer]
```

**Reading it:** one embedding feeds four separate checks, each a distance
and a threshold. Read across the four branches: what changes is what the new
text is compared with (stored documents, route examples, cluster centres,
cached questions) and what happens on a hit.

**Near-duplicates.** The same ticket filed twice, with slightly different
wording, gets nearly identical vectors. "password reset link expired" and
"The password reset link has expired!" score a cosine of 1.0 with the toy
embedder. Flag pairs above a threshold set on pairs you have labelled by
hand (`primer.ml.embeddings.similarity`).

**In code:** `near_duplicates` returns every pair of texts whose cosine
reaches the threshold.

**Routing.** Each route (a team, a tool or an agent) is represented by the
centroid of a few example requests. A new request goes to the closest
centroid, or to a fallback when no centroid is close enough:

$$
\text{route}(q) =
\begin{cases}
\arg\max_r \cos(q, \mu_r) & \text{if } \max_r \cos(q, \mu_r) \ge \theta \\
\text{fallback} & \text{otherwise}
\end{cases}
$$

**Symbols**

| Symbol | Meaning here | Range |
|---|---|---|
| q | the incoming request's embedding | unit vector |
| r | a route: a team, tool or agent | it_helpdesk, finance, hr |
| μᵣ (mu) | route r's centroid: the mean of its example requests, rescaled to length 1 | unit vector |
| cos(q, μᵣ) | cosine similarity of the request with that route | −1 to 1 |
| arg maxᵣ | the route that gives the largest value | |
| θ (theta) | the confidence threshold | 0.3 here |

**In words:** send the request to the route it is most similar to, unless
even the best match is weak, in which case hand it to the fallback.

**On the example:** "my vpn tunnel drops when I work remote" scores highest
against the IT helpdesk; "what is the capital of france" is near 0 against
every route, below θ = 0.3, so it goes to the fallback.

**With the numbers:** a toy version with three routes in four dimensions:
μ_it = (1, 0, 0, 0), μ_finance = (0, 1, 0, 0), μ_hr = (0, 0, 1, 0). The request
q = (0.8, 0.6, 0, 0) has cosines 0.8, 0.6 and 0.0 with them; the best, 0.8,
clears θ = 0.3, so it goes to it_helpdesk. The request q = (0.1, 0.2, 0, 1)
points mostly where no route lies: its cosines are about 0.1, 0.2 and 0.0,
all below θ, so it goes to the fallback.

**In Python:**

```python
import math
def cos(a, b):
    dot = sum(a_k * b_k for a_k, b_k in zip(a, b))
    return dot / (math.sqrt(sum(a_k ** 2 for a_k in a)) * math.sqrt(sum(b_k ** 2 for b_k in b)))
mu = {"it_helpdesk": (1, 0, 0, 0), "finance": (0, 1, 0, 0), "hr": (0, 0, 1, 0)}
theta = 0.3
def route(q):
    scores = {r: cos(q, mu_r) for r, mu_r in mu.items()}
    # arg max_r cos(q, μ_r)
    best = max(scores, key=scores.get)
    return best if scores[best] >= theta else "fallback"
[round(cos((0.8, 0.6, 0, 0), mu_r), 2) for mu_r in mu.values()], route((0.8, 0.6, 0, 0))  # → ([0.8, 0.6, 0.0], 'it_helpdesk')
[round(cos((0.1, 0.2, 0, 1), mu_r), 2) for mu_r in mu.values()], route((0.1, 0.2, 0, 1))  # → ([0.1, 0.2, 0.0], 'fallback')
```

![The VPN complaint clears the threshold only for IT, the receipt question only for finance, and the off-topic question for no route, so it goes to a human](figures/primer.ml.embeddings.clustering.router.svg)

**Reading it:** each group of bars is one incoming request, each bar its
cosine with one route's centroid, and the dashed line the threshold θ. The
VPN complaint clears the line only for IT and the receipt question only for
finance. The off-topic question clears nothing, so it goes to a human.
Without the fallback it would go to whichever team scored highest, however
low that score.

**In code:** `Router.route` applies θ and the fallback; `support_router`
builds the three routes shown here.

**Anomaly detection.** An item far from every cluster centre is unlike
anything seen so far. Score each item by its distance to the nearest centre,
and the largest scores are the unusual items. Among the tickets, the coffee
machine and the dog score highest.

**In code:** `anomaly_scores` returns each point's distance to its nearest
cluster centre.

**Semantic caching.** If a new question means the same as one already
answered, return the stored answer and skip the call to the language model,
which saves its cost and its wait. Three guards stop the cache from
returning a wrong answer:

- **A strict, calibrated threshold.** "How do I reset my VPN?" is only about
  0.46 similar to "How do I reset my password?" here: related, but a
  different question.
- **A context key.** "What is my PTO balance?" means something different for
  Alice and Bob. Answers are only reused within the same user, tenant and
  permission scope.
- **Expiry.** A time-to-live, so answers age out when the facts change.

```mermaid
flowchart LR
  Q[Question + context] --> F[Keep entries with the<br/>same context, not expired]
  F --> S[Most similar entry]
  S --> T{cosine ≥ threshold?}
  T -->|yes| H[Hit: return stored answer]
  T -->|no| M[Miss: call the model,<br/>store the new answer]
```

**Reading it:** follow the arrows. The filter comes *before* the
similarity search, so an answer stored in another user's context is never a
candidate, and no threshold setting can leak it. Similarity decides only
among what is left.

**In code:** `SemanticCache.lookup` filters by context and expiry before it
compares, and `SemanticCache.get` returns an answer only above the
threshold.

## In 20 seconds
- k-means alternates two steps, "join the nearest centre" and "move each
  centre to the mean of its points"; you choose k, by the elbow or the
  silhouette.
- DBSCAN and HDBSCAN find clusters as dense crowds, decide how many there
  are themselves, and mark loners as noise, which suits messy text.
- A PCA map keeps the directions of widest spread and drops the rest; UMAP
  and t-SNE keep neighbours together but distort distances and sizes.
- Near-duplicates, routing, anomaly detection and semantic caching each
  embed a new item, compare it with stored ones and apply a threshold.
- A semantic cache needs a strict threshold, a context key and an expiry.

## Self-test questions

**Q: How does k-means work, and what are its limitations?**
Alternate assigning each point to its nearest centre and moving each centre
to the mean of its points until nothing changes. You must choose k, results
depend on the start (k-means++ helps), and it assumes round, similar-sized
clusters.

**Q: When would you choose HDBSCAN over k-means for text embeddings?**
When you don't know how many groups exist, clusters have different shapes
and densities, and some items belong nowhere (support tickets, logs). It
finds the number of clusters itself and labels outliers as noise.

**Q: Why can't you trust distances in a UMAP or t-SNE plot?**
They preserve local neighbourhoods and deliberately distort everything else,
so gaps between clusters and cluster sizes don't reflect the real space.

**Q: How would you route requests to the right agent or tool with embeddings?**
Represent each route by the centroid of example requests, send each new
request to the most similar centroid, and fall back to a human or general
assistant when the best similarity is below a calibrated threshold.

**Q: What can go wrong with a semantic cache?**
A loose threshold returns the answer to a *different* question; missing
context keys leak one user's or tenant's answer to another; missing expiry
serves stale facts. Filter by context first, then match strictly, and
expire entries.

## The papers behind this lesson

- **Arthur and Vassilvitskii, *k-means++: The Advantages of Careful Seeding* (2007)**: https://dl.acm.org/doi/10.5555/1283383.1283494.
  Showed that spreading out the starting centres makes k-means provably close to optimal and much faster to converge.
- **Ester, Kriegel, Sander and Xu, *A Density-Based Algorithm for Discovering Clusters* (DBSCAN, 1996)**: https://dl.acm.org/doi/10.5555/3001460.3001507.
  Defined clusters as dense regions reachable through core points, with everything else as noise.
- **Campello, Moulavi and Sander, *Density-Based Clustering Based on Hierarchical Density Estimates* (HDBSCAN, 2013)**: https://doi.org/10.1007/978-3-642-37456-2_14.
  Removed DBSCAN's single reach parameter by building a hierarchy over all reaches and keeping the most persistent clusters.
- **McInnes, Healy and Melville, *UMAP* (2018)**: https://arxiv.org/abs/1802.03426.
  A fast projection that keeps each point's neighbours near it, used to draw maps of embeddings.

## Further reading
- scikit-learn user guide, clustering: https://scikit-learn.org/stable/modules/clustering.html
- hdbscan documentation, *How HDBSCAN Works*: https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html
- UMAP documentation: https://umap-learn.readthedocs.io/
- Wattenberg, Viégas and Johnson, *How to Use t-SNE Effectively* (Distill): https://distill.pub/2016/misread-tsne/
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from primer._show import banner, say, table, takeaway
from primer.common.embedder import ConceptEmbedder

# ---------------------------------------------------------------------------
# 0. Data: short support tickets in five kinds, plus two off-topic ones
# ---------------------------------------------------------------------------

TICKETS: dict[str, list[str]] = {
    "login": [
        "I forgot my password and can't log in",
        "password reset link expired",
        "locked out of my account after too many tries",
        "can't sign in, password not accepted",
        "need to recover my login credentials",
        "account lockout, please unlock",
    ],
    "vpn": [
        "vpn keeps disconnecting",
        "can't connect to the vpn from home",
        "anyconnect tunnel error when working remote",
        "remote access not working",
        "vpn client fails to start",
        "vpn connection drops every hour",
    ],
    "expense": [
        "how do I get reimbursed for a hotel receipt",
        "expense report for my car mileage",
        "submit receipts for reimbursement",
        "refund for a business expense",
        "my expense claim was rejected",
        "reimburse travel expenses",
    ],
    "leave": [
        "how many vacation days do I have",
        "request pto for next month",
        "sick leave policy question",
        "book time-off over the holidays",
        "parental leave dates",
        "leave request not approved yet",
    ],
    "printer": [
        "printer on floor 3 is jammed",
        "printing fails with an error",
        "printer out of toner",
        "can't find the printer in settings",
        "print queue stuck",
        "printer offline again",
    ],
}
OUTLIERS = ["the coffee machine is making a strange noise", "can I bring my dog to the office on friday"]

EMBEDDER = ConceptEmbedder(dim=128)


def ticket_embeddings(include_outliers: bool = True) -> tuple[np.ndarray, list[str]]:
    """(unit vectors, texts) for every ticket, with the off-topic ones last if included."""
    texts = [t for group in TICKETS.values() for t in group] + (OUTLIERS if include_outliers else [])
    return EMBEDDER.encode(texts), texts


def ticket_kinds(include_outliers: bool = True) -> list[str]:
    return [k for k, group in TICKETS.items() for _ in group] + (["off-topic"] * len(OUTLIERS) if include_outliers else [])


# ---------------------------------------------------------------------------
# 1. k-means with k-means++ starting centres
# ---------------------------------------------------------------------------


def _sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance from every row of X to every row of C: shape (len(X), len(C))."""
    return np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=-1)


def _kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Pick the first centre at random, then each next one with probability ∝ squared
    distance to the nearest centre so far. Spread-out starts avoid bad local optima."""
    centres = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = _sq_dists(X, np.array(centres)).min(axis=1)
        centres.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    return np.array(centres)


def kmeans(X: np.ndarray, k: int, n_init: int = 5, iters: int = 100, seed: int = 0):
    """Lloyd's algorithm. Returns (labels, centres, inertia), best of `n_init` restarts.

    inertia = total squared distance from each point to its centre: the
    number k-means is trying to make small.
    """
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        C = _kmeans_pp_init(X, k, rng)
        for _ in range(iters):
            labels = _sq_dists(X, C).argmin(axis=1)  # step 1: assign each point to its nearest centre
            # step 2: move each centre to the mean of its points (keep it if it lost all of them)
            newC = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        labels = _sq_dists(X, C).argmin(axis=1)
        inertia = float(_sq_dists(X, C)[np.arange(len(X)), labels].sum())
        if best is None or inertia < best[2]:
            best = (labels, C, inertia)
    return best


def kmeans_history(X: np.ndarray, k: int, iters: int = 20, seed: int = 0) -> list[dict]:
    """One k-means run, recording labels, centres and inertia after the start and each round."""
    rng = np.random.default_rng(seed)
    C = _kmeans_pp_init(X, k, rng)
    history = []
    for _ in range(iters + 1):
        labels = _sq_dists(X, C).argmin(axis=1)
        history.append({"labels": labels, "centres": C.copy(), "inertia": float(_sq_dists(X, C)[np.arange(len(X)), labels].sum())})
        newC = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return history


# ---------------------------------------------------------------------------
# 2. Choosing k: silhouette
# ---------------------------------------------------------------------------


def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette: for each point, (b - a) / max(a, b), where
    a = mean distance to its own cluster, b = mean distance to the nearest other cluster.
    Near 1: tight, well-separated clusters. Near 0: overlapping. Negative: misassigned.
    """
    D = np.sqrt(_sq_dists(X, X))
    scores = []
    for i in range(len(X)):
        same = (labels == labels[i]) & (np.arange(len(X)) != i)
        if not same.any():  # a cluster of one has no "a"; convention: score 0
            scores.append(0.0)
            continue
        a = D[i, same].mean()
        b = min(D[i, labels == c].mean() for c in set(labels.tolist()) if c != labels[i])
        scores.append((b - a) / max(a, b))
    return float(np.mean(scores))


def best_k_by_silhouette(X: np.ndarray, ks) -> int:
    return max(ks, key=lambda k: silhouette(X, kmeans(X, k)[0]))


# ---------------------------------------------------------------------------
# 3. Density-based clustering: DBSCAN
# ---------------------------------------------------------------------------


def _distances(X: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        return np.sqrt(_sq_dists(X, X))
    if metric == "cosine":  # vectors assumed unit length
        return 1.0 - X @ X.T
    raise ValueError(metric)


def dbscan(X: np.ndarray, eps: float, min_samples: int, metric: str = "cosine") -> np.ndarray:
    """Labels 0, 1, 2, … for clusters and -1 for noise.

    A *core* point has at least `min_samples` points (itself included) within
    `eps`. Clusters grow outward from core points through their neighbours;
    anything no core point can reach is noise. The number of clusters is
    discovered, not chosen.
    """
    D = _distances(X, metric)
    neighbours = [np.flatnonzero(D[i] <= eps) for i in range(len(X))]
    core = np.array([len(n) >= min_samples for n in neighbours])
    labels = np.full(len(X), -1)
    cluster = 0
    for i in range(len(X)):
        if labels[i] != -1 or not core[i]:
            continue
        labels[i] = cluster
        frontier = list(neighbours[i])
        while frontier:  # breadth-first flood fill through core points
            j = frontier.pop()
            if labels[j] == -1:
                labels[j] = cluster
                if core[j]:
                    frontier.extend(neighbours[j])
        cluster += 1
    return labels


# ---------------------------------------------------------------------------
# 4. PCA: the best flat picture of high-dimensional points
# ---------------------------------------------------------------------------


def pca(X: np.ndarray, n: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Project onto the n directions of greatest variance.

    Returns (coordinates (N, n), share of total variance each direction explains).
    Computed with the SVD of the centred data (see `primer.notation`).
    """
    Xc = X - X.mean(axis=0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S**2
    return Xc @ Vt[:n].T, var[:n] / var.sum()


# ---------------------------------------------------------------------------
# 5. Practical uses
# ---------------------------------------------------------------------------


def near_duplicates(texts: list[str], threshold: float, embedder: ConceptEmbedder = EMBEDDER) -> list[tuple[int, int]]:
    """Index pairs (i < j) whose cosine is at least `threshold`."""
    X = embedder.encode(texts)
    S = X @ X.T
    i, j = np.triu_indices(len(texts), k=1)
    keep = S[i, j] >= threshold
    return list(zip(i[keep].tolist(), j[keep].tolist()))


@dataclass
class Router:
    """Send a request to the route whose example centroid it's closest to, or to a fallback."""

    routes: dict[str, list[str]]
    threshold: float = 0.3
    embedder: ConceptEmbedder = EMBEDDER

    def __post_init__(self):
        # One centroid per route: the mean of its examples, rescaled to length 1.
        self.names = list(self.routes)
        C = np.stack([self.embedder.encode(ex).mean(axis=0) for ex in self.routes.values()])
        self.centroids = C / np.linalg.norm(C, axis=1, keepdims=True)

    def scores(self, text: str) -> dict[str, float]:
        return dict(zip(self.names, (self.centroids @ self.embedder.encode(text)).tolist()))

    def route(self, text: str) -> str:
        s = self.scores(text)
        best = max(s, key=s.get)
        # Below the threshold nothing is a confident match: hand off rather than guess.
        return best if s[best] >= self.threshold else "fallback"


def support_router() -> Router:
    return Router(
        {
            "it_helpdesk": TICKETS["login"] + TICKETS["vpn"] + TICKETS["printer"],
            "finance": TICKETS["expense"],
            "hr": TICKETS["leave"],
        }
    )


def anomaly_scores(X: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Distance from each point to its nearest cluster centre. Large = unlike anything known."""
    return np.sqrt(_sq_dists(X, centres).min(axis=1))


@dataclass
class _Entry:
    vector: np.ndarray
    question: str
    answer: str
    context: str
    created: float


@dataclass
class SemanticCache:
    """Reuse a stored answer when a new question means the same thing.

    Three guards, each preventing a real failure:
    * `SemanticCache.threshold`: only near-identical meaning counts (calibrate it per model,
      see `primer.ml.embeddings.similarity`).
    * `context`: answers are only reused within the same context key (user,
      tenant, permissions), so one person's answer never reaches another.
    * `ttl_seconds`: entries expire, so stale answers age out.
    """

    threshold: float = 0.9
    ttl_seconds: float = 24 * 3600
    embedder: ConceptEmbedder = EMBEDDER
    entries: list[_Entry] = field(default_factory=list)

    def put(self, question: str, answer: str, context: str, now: float) -> None:
        self.entries.append(_Entry(self.embedder.encode(question), question, answer, context, now))

    def lookup(self, question: str, context: str, now: float) -> tuple[_Entry | None, float]:
        """Best eligible entry and its cosine (eligible = same context and not expired)."""
        q = self.embedder.encode(question)
        eligible = [e for e in self.entries if e.context == context and now - e.created <= self.ttl_seconds]
        if not eligible:
            return None, 0.0
        sims = [float(e.vector @ q) for e in eligible]
        i = int(np.argmax(sims))
        return eligible[i], sims[i]

    def get(self, question: str, context: str, now: float) -> str | None:
        entry, sim = self.lookup(question, context, now)
        return entry.answer if entry is not None and sim >= self.threshold else None


# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------

_KIND_COLOURS = {"login": "C0", "vpn": "C1", "expense": "C2", "leave": "C3", "printer": "C4"}


def figures() -> dict:
    """Plots computed from this module's own functions. Keys match the docstring's image names."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = {}
    X, texts = ticket_embeddings(include_outliers=True)
    kinds = ticket_kinds(include_outliers=True)
    n_in = len(texts) - len(OUTLIERS)
    coords, ratio = pca(X, 2)

    # similarity heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(X @ X.T, cmap="viridis", vmin=-0.2, vmax=1)
    ticks = [i * 6 + 2.5 for i in range(5)] + [n_in + 0.5]
    ax.set_xticks(ticks, list(TICKETS) + ["off-topic"], rotation=30)
    ax.set_yticks(ticks, list(TICKETS) + ["off-topic"])
    ax.set(title="Cosine similarity between every pair of tickets")
    fig.colorbar(im, ax=ax, label="cosine similarity")
    figs["similarity"] = fig

    # kmeans steps on toy 2-D blobs
    rng = np.random.default_rng(3)
    blobs = np.concatenate([rng.normal(c, 0.6, (25, 2)) for c in ((0, 0), (4, 1), (1.5, 4))])
    hist = kmeans_history(blobs, 3, seed=4)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for ax, (h, title) in zip(axes, ((hist[0], "start (k-means++)"), (hist[1], "after round 1"), (hist[-1], f"converged (round {len(hist) - 1})"))):
        ax.scatter(blobs[:, 0], blobs[:, 1], c=[f"C{l}" for l in h["labels"]], s=20, alpha=0.8)
        ax.scatter(h["centres"][:, 0], h["centres"][:, 1], marker="X", s=200, color="k")
        ax.set(title=f"{title}\ninertia {h['inertia']:.1f}", xlabel="x")
    axes[0].set_ylabel("y")
    figs["kmeans_steps"] = fig

    # choose k
    Xin = X[:n_in]
    ks = list(range(2, 9))
    runs = [kmeans(Xin, k) for k in ks]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(ks, [r[2] for r in runs], marker="o")
    a1.set(xlabel="k", ylabel="inertia (total squared distance)", title="Elbow: inertia always falls")
    a2.plot(ks, [silhouette(Xin, r[0]) for r in runs], marker="o", color="C1")
    a2.axvline(5, ls="--", color="0.5")
    a2.set(xlabel="k", ylabel="mean silhouette", title="Silhouette peaks at the true k = 5")
    figs["choose_k"] = fig

    # map: PCA coloured by kind with k-means centres
    _, centres, _ = kmeans(Xin, 5)
    cproj = (centres - X.mean(axis=0)) @ np.linalg.svd(X - X.mean(axis=0), full_matrices=False)[2][:2].T
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for kind, colour in _KIND_COLOURS.items():
        m = np.array(kinds) == kind
        ax.scatter(coords[m, 0], coords[m, 1], color=colour, label=kind, s=40, alpha=0.8)
    ax.scatter(coords[n_in:, 0], coords[n_in:, 1], marker="x", color="0.4", s=80, label="off-topic")
    ax.scatter(cproj[:, 0], cproj[:, 1], marker="+", color="k", s=250, linewidths=2.5, label="k-means centres")
    ax.set(xlabel="principal component 1", ylabel="principal component 2", title=f"Tickets on a 2-D map (keeps {ratio.sum():.0%} of the variation)")
    ax.legend(fontsize=8)
    figs["map"] = fig

    # eps sweep
    epss = np.round(np.arange(0.45, 0.851, 0.025), 3)
    runs_eps = [dbscan(X, eps=e, min_samples=3) for e in epss]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(epss, [len(set(l.tolist()) - {-1}) for l in runs_eps], marker="o", label="clusters found")
    ax.plot(epss, [int(np.sum(l == -1)) for l in runs_eps], marker="s", label="noise points")
    ax.axhline(len(OUTLIERS), ls="--", color="0.5", label="truly off-topic tickets")
    ax.set(xlabel="reach ε (cosine distance)", ylabel="count", title="DBSCAN: too small strands tickets, too large chains kinds")
    ax.legend()
    figs["eps_sweep"] = fig

    # dbscan
    labels = dbscan(X, eps=0.6, min_samples=3)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for l in sorted(set(labels.tolist())):
        m = labels == l
        if l == -1:
            ax.scatter(coords[m, 0], coords[m, 1], marker="x", color="0.5", s=70, label="noise")
        else:
            ax.scatter(coords[m, 0], coords[m, 1], s=40, alpha=0.8, label=f"cluster {l}")
    for i in range(n_in, len(texts)):
        ax.annotate(texts[i][:22] + "…", coords[i], fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set(xlabel="principal component 1", ylabel="principal component 2", title="DBSCAN (cosine, ε = 0.6, minPts = 3)")
    ax.legend(fontsize=8)
    figs["dbscan"] = fig

    # router
    router = support_router()
    reqs = ["my vpn tunnel drops when I work remote", "who reimburses my hotel receipt", "what is the capital of france"]
    x = np.arange(len(reqs))
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, name in enumerate(router.names):
        ax.bar(x + (j - 1) * 0.27, [router.scores(r)[name] for r in reqs], 0.27, label=name)
    ax.axhline(router.threshold, ls="--", color="0.3", label=f"threshold θ = {router.threshold}")
    ax.set_xticks(x, [r if len(r) < 30 else r[:28] + "…" for r in reqs], fontsize=8)
    ax.set(ylabel="cosine with route centroid", title="Route to the closest centroid, or fall back")
    ax.legend(fontsize=8)
    figs["router"] = fig

    for name, f in figs.items():
        f.tight_layout()
    return figs


# ---------------------------------------------------------------------------
# 7. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. k-means by hand: four points, k = 2")
    pts = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 0.0], [10.0, 1.0]])
    labels, centres, inertia = kmeans(pts, 2)
    say(f"Labels {labels.tolist()}, centres {sorted(map(tuple, centres.round(2).tolist()))}, inertia {inertia:.2f} (4 × 0.5²).")
    say(f"Silhouette of that split: {silhouette(pts, labels):.3f} (hand calculation: 9.025 / 10.025).")

    banner("2. Clustering 30 support tickets")
    X, texts = ticket_embeddings(include_outliers=False)
    table(["k", "inertia", "silhouette"], [(k, kmeans(X, k)[2], silhouette(X, kmeans(X, k)[0])) for k in range(2, 9)], floatfmt=".3f")
    say(f"Best k by silhouette: {best_k_by_silhouette(X, range(2, 9))} (there are 5 ticket kinds).")

    banner("3. DBSCAN finds the loners")
    Xo, texts_o = ticket_embeddings(include_outliers=True)
    lab = dbscan(Xo, eps=0.6, min_samples=3)
    say(f"ε = 0.6: {len(set(lab.tolist()) - {-1})} clusters and {int(np.sum(lab == -1))} noise points. Noise:")
    for t, l in zip(texts_o, lab):
        if l == -1:
            print("   -", t)
    print()
    table(["reach ε", "clusters", "noise points"], [(e, len(set(dbscan(Xo, e, 3).tolist()) - {-1}), int(np.sum(dbscan(Xo, e, 3) == -1))) for e in (0.55, 0.6, 0.65, 0.75)], floatfmt=".2f")
    say("Too small a reach strands real tickets; too large chains different kinds together. HDBSCAN avoids choosing.")

    banner("4. Near-duplicates, routing, anomalies")
    say(f"Near-duplicate pairs above 0.95: {near_duplicates(['password reset link expired', 'The password reset link has expired!', 'printer out of toner'], 0.95)}")
    router = support_router()
    table(["request", "route"], [(r, router.route(r)) for r in ("my vpn tunnel drops when I work remote", "who reimburses my hotel receipt", "how many sick days do I get", "what is the capital of france")])
    _, cen, _ = kmeans(X, 5)
    scores = anomaly_scores(Xo, cen)
    table(["most unusual tickets", "distance to nearest centre"], [(texts_o[i], scores[i]) for i in np.argsort(-scores)[:3]], floatfmt=".3f")

    banner("5. A semantic cache with its three guards")
    cache = SemanticCache(threshold=0.9, ttl_seconds=3600)
    cache.put("How do I reset my password?", "Use the self-service portal.", "everyone", now=0)
    cache.put("What is my PTO balance?", "Alice has 12 days.", "user:alice", now=0)
    rows = []
    for q, ctx, now in [
        ("how can I reset my password", "everyone", 10),
        ("How do I reset my VPN?", "everyone", 10),
        ("What is my PTO balance?", "user:bob", 10),
        ("How do I reset my password?", "everyone", 4000),
    ]:
        entry, sim = cache.lookup(q, ctx, now)
        rows.append((q, ctx, now, f"{sim:.2f}" if entry else "no eligible entry", cache.get(q, ctx, now) or "MISS"))
    table(["question", "context", "t (s)", "best cosine", "result"], rows)
    takeaway("Filter by context and age first, then match strictly. A cache that ignores context leaks answers between users.")


if __name__ == "__main__":
    demo()
