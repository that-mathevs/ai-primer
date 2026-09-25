r"""
# Clustering and everyday uses of embeddings: grouping, mapping, routing, caching

Run: `python -m primer.ml.embeddings.clustering`

New to vectors, distances or Σ? `primer.notation` builds them from zero.

## The everyday picture

Tip a sack of unlabeled mail onto a table and sort it into piles by what
each letter is about. Nobody gave you the pile names; you notice that some
letters are about passwords and others about holidays, and similar letters
end up together. That's **clustering**: finding groups in data without being
told what the groups are.

Embeddings make it possible for text. Every ticket, email or document gets
coordinates on a map of meaning (`primer.ml.embeddings.similarity`), so
"similar" becomes "close", and sorting mail becomes finding crowds of nearby
points. The same closeness also powers four everyday jobs: spotting
**near-duplicates**, **routing** requests to the right team, flagging
**anomalies**, and **caching** answers to questions already asked.

This lesson uses 30 short IT, HR and finance support tickets (five kinds, six
of each) and two off-topic ones, embedded with the repo's toy embedder
(`primer.common.embedder`).

![Ticket-by-ticket similarity](figures/primer.ml.embeddings.clustering.similarity.svg)

**Reading it:** rows and columns are the 32 tickets in the same order, grouped
by kind, and brighter cells mean a higher cosine similarity. The five bright
squares along the diagonal are the five kinds: tickets of one kind resemble
each other and not the rest. The last two rows and columns (the off-topic
tickets) are dark almost everywhere. Clustering is the job of finding those
squares *without* knowing the order.

**In code:** `ticket_embeddings` returns the tickets' unit vectors and texts,
and `ticket_kinds` their true kinds, which the clustering never sees.

## k-means: k meeting points

**Everyday picture:** a town wants k post boxes placed so that everyone's
walk to their nearest box is as short as possible. Start with the boxes
anywhere. Everyone walks to their nearest box; then each box moves to the
middle of the people who chose it. Repeat until nobody switches box.

**Tiny worked example:** four points, (0, 0), (0, 1), (10, 0), (10, 1), and
k = 2. However the boxes start, the two left points pick one box and the two
right points the other. Each box moves to the middle of its pair: (0, 0.5) and
(10, 0.5). Every point is now 0.5 away from its box, so the total squared
walk is 4 × 0.5² = **1.0**, and nothing changes on the next round.

```mermaid
flowchart LR
  I["Place k centres<br/>(k-means++: spread out)"] --> A["Assign: each point<br/>joins its nearest centre"]
  A --> M["Move: each centre goes to<br/>the mean of its points"]
  M --> C{Did any centre move?}
  C -->|yes| A
  C -->|no| D[Done: labels + centres]
```

**Reading it:** two steps alternate, assign and move, until the centres stop
moving. Each step can only shrink the total squared distance (assigning to
the nearest centre can't make anyone's walk longer; moving a centre to the
mean of its points is the spot that minimizes their squared walks), so the
loop always ends. The start matters: **k-means++** picks each new starting
centre far from the ones already chosen, which avoids two centres fighting
over one crowd.

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

![k-means, round by round](figures/primer.ml.embeddings.clustering.kmeans_steps.svg)

**Reading it:** three snapshots of k-means on 2-D points. Colours are the
current assignments and black crosses are the centres. On the left, the
k-means++ starting centres; in the middle, after the first assign-and-move
round; on the right, the final state. The centres drift into the middle of
their crowds and the total squared distance printed above each panel only
goes down.

**In code:** `kmeans` alternates assign and move from k-means++ starts and
returns the labels, centres and inertia J of the best of several restarts;
`kmeans_history` records one run round by round, which is what this figure
draws.

**Why it matters:** k-means is fast and simple, and it's inside things you
use: IVF vector indexes cluster the corpus with it (`primer.ml.embeddings.ann`),
and topic discovery over tickets or documents often starts with it. Its
weaknesses are that you must choose k and that it assumes round, similar-sized
clusters.

## Choosing k: the elbow and the silhouette

**Everyday picture:** at a party, you're in the right group if the people in
your group are much closer to you than the people in the next group over.

**Tiny worked example:** the four points again, split into the two pairs.
For (0, 0): its partner is a = **1** away; the other pair is 10 and √101 = 10.05
away, on average b = **10.025**. Its score is (b − a) / b = 9.025 / 10.025 = **0.900**,
and by symmetry every point scores the same, so the silhouette is 0.900.

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

![Choosing k: inertia and silhouette](figures/primer.ml.embeddings.clustering.choose_k.svg)

**Reading it:** both panels sweep k from 2 to 8 on the 30 tickets. On the
left, inertia always falls as k grows (more boxes, shorter walks), so the
lowest value is useless; you look for the **elbow** where it stops falling
steeply. On the right, the silhouette has a clear peak at k = 5, the true
number of ticket kinds. When the elbow is vague, the silhouette usually
isn't.

**In code:** `silhouette` averages s(i) over every point, and
`best_k_by_silhouette` runs `kmeans` for each candidate k and keeps the one
with the highest silhouette.

## Density clustering: DBSCAN and HDBSCAN

**Everyday picture:** a festival seen from a drone. A crowd is wherever
people stand shoulder to shoulder; a loner by the fence belongs to no crowd.
You don't decide in advance how many crowds there are.

**Tiny worked example:** points on a line at 0, 0.1, 0.2, 5.0, 5.1, 5.2 and 20,
with reach ε = 0.15 and "a crowd needs at least 2". 0, 0.1 and 0.2 chain
together (each within 0.15 of the next); so do 5.0, 5.1 and 5.2. 20 has nobody
within 0.15. Result: two clusters and one noise point, labels
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

**Reading it:** DBSCAN walks the points. A point with too few neighbours is
left as noise (it can still be claimed later as the edge of someone else's
crowd). A core point starts a cluster that floods outward through other core
points, the self-loop on the "Add" box, until the crowd's edge is reached.

**In code:** `dbscan` finds the core points, floods each cluster outward
through them, and labels everything unreached −1 (noise).

![DBSCAN on the tickets](figures/primer.ml.embeddings.clustering.dbscan.svg)

**Reading it:** the tickets drawn on the 2-D map from the next section,
coloured by the clusters DBSCAN found (cosine distance, so ε = 0.6 means
"cosine at least 0.4"; minPts = 3). Grey crosses are noise: exactly the two
off-topic tickets, found without anyone telling DBSCAN how many clusters
exist. Notice that VPN and printer tickets share a cluster: a chain of
tickets that are each close to the next ("can't connect", "not working",
"offline") bridges the two kinds.

![How the reach ε changes DBSCAN's answer](figures/primer.ml.embeddings.clustering.eps_sweep.svg)

**Reading it:** the horizontal axis is the reach ε. The blue line counts
clusters and the orange line counts noise points; the dashed line marks the
two truly off-topic tickets. With a small reach, DBSCAN finds all five kinds
but strands genuine tickets as noise. As the reach grows, noise falls to
just the off-topic pair, but kinds start **chaining** together (A is near B,
B is near C, so A and C end up in one cluster) until only two clusters are
left. No single ε gets everything right, and on real data you rarely know
where the sweet spot is.

That's the problem **HDBSCAN** solves. It effectively runs DBSCAN at every
reach at once, builds a tree of how crowds merge as the reach grows, and keeps
the crowds that persist over the widest range of reaches. It handles clusters
of different densities, needs no ε, and is the usual choice for exploring
messy real text such as support tickets.

## Seeing the map: PCA (and why UMAP and t-SNE mislead on distance)

**Everyday picture:** a shadow on a wall. A 3-D object casts a 2-D shadow;
turn the object and the shadow changes. **PCA** (principal component
analysis) turns the object so its shadow is as spread out as possible,
keeping as much of the original variation as a flat picture can.

**Tiny worked example:** three points on a straight line, (1, 1), (2, 2),
(3, 3). All their variation runs along the diagonal, so the first direction
explains **100%** of it and the second **0%**. Along that direction the points
sit √2 = **1.414** apart, exactly as in the original.

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

**In words:** the share of all the spread that direction j captures.

**On the example:** centred, the points are (−1, −1), (0, 0), (1, 1). Their
singular values are 2 along the diagonal and 0 across it, so the shares are
2² / (2² + 0²) = 1 and 0 / 4 = 0.

![The tickets on a 2-D map](figures/primer.ml.embeddings.clustering.map.svg)

**Reading it:** the 128-dimensional ticket embeddings squashed to 2-D with
PCA, coloured by their true kind; black crosses are the k-means centres,
projected the same way, and the grey ✕ markers are the off-topic tickets.
Kinds form separate patches and the centres sit inside them. The title says
what share of the variation the two axes keep: the rest is invisible here,
so tickets that look close on this map can be far apart in the real space.

**In code:** `pca` centres the data, projects it onto the top n directions
from the SVD, and returns each direction's explained share.

UMAP and t-SNE draw prettier maps by a different rule: keep each point's
*neighbours* next to it, and let distances elsewhere stretch. Like a subway
map, they're great for "what's near what" and misleading for "how far" or
"how big": gaps between clusters and cluster sizes in those plots don't
reflect the real space. Use them for intuition, never for measurement.

## Everyday uses

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

**Reading it:** one embedding feeds four independent checks, each a
comparison with things already known. All four have the same shape: a
distance and a threshold. What differs is what's being compared against
(stored documents, route examples, cluster centres, cached questions) and
what happens on a hit.

**Near-duplicates.** Photocopies with a sticky note on them: nearly
identical vectors. "password reset link expired" and "The password reset
link has expired!" score cosine 1.0 here; flag pairs above a threshold
calibrated on labeled pairs (`primer.ml.embeddings.similarity`).

**In code:** `near_duplicates` embeds a list of texts and returns every pair
whose cosine reaches the threshold.

**Routing.** A receptionist listening to a request and pointing to the right
desk. Each route (a team, a tool, an agent) is represented by the centroid
of a few example requests; a new request goes to the closest centroid, or to
a fallback when nothing is close:

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

**In words:** send the request to the route it's most similar to, unless even
the best match is weak, in which case hand it off.

**On the example:** "my vpn tunnel drops when I work remote" scores highest
against the IT helpdesk; "what is the capital of france" is near 0 against
every route, below θ = 0.3, so it goes to the fallback.

![Router scores for three requests](figures/primer.ml.embeddings.clustering.router.svg)

**Reading it:** each group of bars is one incoming request; each bar is its
cosine with one route's centroid; the dashed line is the threshold θ. The
VPN complaint clears the line only for IT, the receipt question only for
finance, and the off-topic question clears nothing, so it goes to a human.
The fallback is the important part: a router without one sends every
unanswerable question *somewhere*.

**In code:** `Router` holds one unit-length centroid per route;
`Router.scores` gives a request's cosine with each, and `Router.route`
applies θ and the fallback. `support_router` builds the lesson's three
routes.

**Anomaly detection.** A stranger at a party is far from every group. Score
each item by its distance to the nearest cluster centre; the largest scores
are the unusual items. Here the coffee machine and the dog come out on top.

**In code:** `anomaly_scores` returns each point's distance to its nearest
cluster centre.

**Semantic caching.** An FAQ desk that remembers answers. If a new question
means the same as one already answered, return the stored answer and skip
the model call, saving cost and latency. Three guards keep it safe:

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

**Reading it:** the filter comes *before* the similarity search, not after:
an answer from another user's context is never even a candidate, so no
threshold setting can leak it. Only then does similarity decide.

**In code:** `SemanticCache.put` stores a question, its answer and its
context; `SemanticCache.lookup` filters by context and expiry, then finds the
most similar entry, and `SemanticCache.get` returns its answer only above the
threshold.

## In 20 seconds
- k-means alternates "assign to nearest centre" and "move centre to the
  mean"; you choose k, using the silhouette or the elbow.
- DBSCAN/HDBSCAN find crowds by density, choose the number of clusters
  themselves, and mark loners as noise; better for messy text.
- PCA gives an honest but lossy 2-D shadow; UMAP/t-SNE keep neighbours but
  distort distances and sizes.
- Near-duplicates, routing, anomaly detection and semantic caching are all
  "embed, compare, threshold".
- Semantic caches need a strict threshold, a context key and an expiry.

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
  A fast neighbour-preserving projection now standard for visualizing embeddings.

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
    * `threshold`: only near-identical meaning counts (calibrate it per model,
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
