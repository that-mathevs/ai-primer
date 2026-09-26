r"""
# Long context and efficient architectures: reading a million tokens without a million-squared bill

Run: `python -m primer.ml.efficient_architectures`

## The idea

Attention (see `primer.ml.attention`) lets every token look at every earlier
token. That is where its power comes from, and it is also where its bill comes
from. The bill has two lines:

- **Compute.** Every token is scored against every other token, so the work
  grows with the *square* of the context length n.
- **Memory.** During generation the model keeps every earlier token's keys
  and values (the KV cache, see `primer.ml.inference`), so memory grows with
  n, per layer, per conversation.

Everything in this lesson attacks one of those two lines, in one of three ways:

1. **Look at fewer tokens**: sliding-window and sparse attention.
2. **Summarize the past into a fixed-size state**: linear attention and
   state-space models (Mamba), which run like a recurrent network.
3. **Store the past more compactly**: fewer key/value heads, a small latent
   vector per token, fewer bits per number.

```mermaid
flowchart TB
  P["Long context is expensive<br/>compute grows with n², memory with n"] --> F["Look at fewer tokens"]
  P --> S["Keep a fixed-size summary"]
  P --> C["Store the past compactly"]
  F --> F1["sliding window"] & F2["sparse: local + global, strided"]
  S --> S1["linear attention"] & S2["state-space models, Mamba"]
  C --> C1["GQA / MQA"] & C2["latent KV"] & C3["quantized KV"]
  S2 --> H["hybrids: a few attention layers<br/>among many SSM layers"]
  F1 --> H
```

**Reading it:** start at the top box, the problem. The three arrows below it
are the three strategies, and the leaves are the techniques this lesson
builds from scratch, section by section. The bottom box is where real
models often land: a mix, keeping a little full attention for exact lookups
and using cheap layers for everything else.

## 1. Why long context is expensive

**Everyday picture.** A dinner party where every new guest must shake hands
with every guest already there. Ten guests make 45 handshakes; a thousand
guests make half a million. On top of that, the coat check keeps one coat per
guest on every floor of the building, so the coat racks grow with every
arrival. Attention pays both bills: the handshakes are the query-key scores,
and the coats are the KV cache.

**Tiny worked example.** Take a Llama-3-8B-shaped model: 32 layers, 8
key/value heads, 128 numbers per head, 16-bit numbers (2 bytes). One token
costs 2 × 32 × 8 × 128 × 2 = 131,072 bytes of cache, about 128 KB.

| Context n | Pairs scored per head per layer (n²) | KV cache for one conversation |
|---|---|---|
| 8,192 (8k) | 67 million | 1.1 GB |
| 131,072 (128k) | 17 billion | 17.2 GB |
| 1,048,576 (1M) | 1.1 trillion | 137.4 GB |

Going from 8k to 1M is 128 times more tokens, 16,384 times more pairs, and a
cache that no longer fits on one 80 GB GPU.

```mermaid
flowchart LR
  T["new token n"] --> Q["its query"]
  Q -->|"scored against"| KC[("KV cache<br/>n − 1 keys and values<br/>per layer")]
  KC --> O["output for token n"]
  T -->|"its key and value<br/>are appended"| KC
```

**Reading it:** follow one new token. Its query is scored against every key
already in the cache, so the work for this one token grows with n, and the
work for all n tokens grows with n². Then its own key and value are appended,
so the cache grows by one entry, in every layer, for every token. The two
arrows into the cache are the two bills.

$$
\text{pairs} = n^2 \qquad\qquad \text{KV bytes} = 2 \, L \, H_{kv} \, d_h \, b \, n
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $n$ | context length: how many tokens are in play | 131,072 |
| $n^2$ | $n$ times $n$: every query paired with every key (causal masking halves it, which does not change how it grows) | 17,179,869,184 |
| $2$ | one key and one value per token | |
| $L$ | layers, each with its own cache | 32 |
| $H_{kv}$ | key/value heads per layer | 8 |
| $d_h$ | numbers per head | 128 |
| $b$ | bytes per stored number | 2 (16-bit) |

**In words:** "the score work is the context length squared, and the cache
holds a key and a value for every head, in every layer, for every token."

**With the numbers:** at n = 131,072 the pairs are 131,072² ≈ 1.7 × 10¹⁰,
and the cache is 2 × 32 × 8 × 128 × 2 × 131,072 = 17,179,869,184 bytes ≈
17.2 GB.

**In Python:**

```python
L, H_kv, d_h, b = 32, 8, 128, 2
# bytes per token: a key and a value, per layer, per KV head
per_token = 2 * L * H_kv * d_h * b
per_token  # → 131072
contexts = [8_192, 131_072, 1_048_576]
# pairs per head per layer
[f"{n * n:,}" for n in contexts]  # → ['67,108,864', '17,179,869,184', '1,099,511,627,776']
# GB of cache for one conversation
[round(n * per_token / 1e9, 1) for n in contexts]  # → [1.1, 17.2, 137.4]
```

![Pairs scored grow from 67 million at 8k tokens to 1.1 trillion at 1M, and the KV cache from 1.1 GB to 137 GB, past an 80 GB GPU](figures/primer.ml.efficient_architectures.context_cost.svg)

**Reading it:** the left panel counts query-key pairs per head per layer, on
a logarithmic axis where each gridline is ten times the last: every step to
the right multiplies the bar by 256 and then by 64. The right panel is the KV
cache for one conversation on an ordinary axis, with the dashed line at 80 GB,
the memory of one large GPU. The 1M bar crosses it before any weights are
loaded. The left panel is the compute bill; the right panel is the memory
bill.

**Why it matters in practice.** The compute bill is paid once per prompt
(prefill), and exact tricks such as FlashAttention make it faster without
changing it. The memory bill is paid for as long as a conversation is open,
and it decides how many users one GPU can serve. That is why most of the
techniques below target the cache.

**In code:** `context_cost` returns both lines of the bill for any context length, using `primer.ml.inference.kv_cache_bytes` for the cache.

## 2. Sliding-window attention: reading through a letterbox

**Everyday picture.** You read a long scroll through a letterbox slot that
shows only the last w words. On your own you would lose the beginning. But
suppose a row of readers sits one above the other, each reading the notes of
the reader below through a slot of the same width. Each reader passes things
along a little further, like a bucket brigade, so a stack of readers can
carry a fact much further back than any one slot shows.

**Tiny worked example.** Eight tokens, window w = 3: each token reads itself
and the two tokens before it.

```text
            key:  0 1 2 3 4 5 6 7
query 0           ✓ · · · · · · ·
query 1           ✓ ✓ · · · · · ·
query 2           ✓ ✓ ✓ · · · · ·
query 3           · ✓ ✓ ✓ · · · ·
query 4           · · ✓ ✓ ✓ · · ·
query 5           · · · ✓ ✓ ✓ · ·
query 6           · · · · ✓ ✓ ✓ ·
query 7           · · · · · ✓ ✓ ✓
```

That is 1 + 2 + 3 × 6 = **21** scores instead of the 36 a full causal mask
needs. Now stack three such layers. Token 7 reads token 5 in the top layer;
token 5 had read token 3 in the layer below; token 3 had read token 1 in the
layer below that. So after three layers, token 1 can influence token 7, six
positions back, but token 0 cannot.

```mermaid
flowchart RL
  subgraph L3["layer 3"]
    a7["token 7"]
  end
  subgraph L2["layer 2"]
    b5["token 5"]
  end
  subgraph L1["layer 1"]
    c3["token 3"]
  end
  subgraph IN["input"]
    d1["token 1"]
    d0["token 0: out of reach"]
  end
  a7 -->|"reads 2 back"| b5 -->|"reads 2 back"| c3 -->|"reads 2 back"| d1
```

**Reading it:** read right to left, from token 7 at the top layer down to the
input. Each arrow is one layer's window: it can reach at most w − 1 = 2
positions back. Three layers chain three arrows, so the furthest input token 7
can hear from is 3 × 2 = 6 positions back, token 1. Token 0 would need a
fourth hop. Information still travels far, but only in stages.

$$
M_{ij} = \begin{cases} 1 & \text{if } 0 \le i - j < w \\ 0 & \text{otherwise} \end{cases}
\qquad\qquad
\text{reach} = L\,(w - 1)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $M_{ij}$ | the mask: 1 if query $i$ may read key $j$, 0 if not | row 5: keys 3, 4, 5 |
| $i$, $j$ | the query's position and the key's position | $i = 5$ |
| $i - j$ | how far back the key is; negative means the future | 0, 1, 2 allowed |
| $w$ | the window: how many tokens each query reads, itself included | 3 |
| $\{$ | "cases": use the line whose condition holds | |
| $L$ | how many windowed layers are stacked | 3 |
| reach | the furthest back an input can influence an output | 6 |

**In words:** "a query may read a key if the key is not in the future and is
fewer than w positions back; stacking L layers lets information travel
L times w − 1 positions."

**With the numbers:** row 5 allows j = 3, 4, 5 (5 − 3 = 2 < 3). Three layers
of w = 3 reach 3 × 2 = 6. Mistral 7B uses w = 4,096 over 32 layers:
32 × 4,095 = 131,040 tokens, about 131k, from a window of 4k.

**In Python:**

```python
n, w = 8, 3
# M_ij for the row of token 5
[int(0 <= 5 - j < w) for j in range(n)]  # → [0, 0, 0, 1, 1, 1, 0, 0]
# pairs scored: row i keeps min(i + 1, w) keys
sum(min(i + 1, w) for i in range(n))  # → 21
# full causal attention, for comparison
n * (n + 1) // 2  # → 36
# reach after L layers
L = 3
L * (w - 1)  # → 6
# Mistral 7B: 32 layers, a 4,096-token window
32 * (4096 - 1)  # → 131040
```

![Left, one layer with a window of 4 is a thin diagonal band; right, after three layers each token can hear the last 10 positions, a band three times as wide](figures/primer.ml.efficient_architectures.window_reach.svg)

**Reading it:** both panels are 16 × 16 grids, with rows for the token doing
the looking and columns for the token being looked at, like the causal
heatmap in `primer.ml.attention`. Dark means "can reach". On the left, one
layer with w = 4 is a thin band along the diagonal: 4 cells per row at most.
On the right, the same mask applied three times: the band has widened to
3 × 3 + 1 = 10 cells, because each layer extends the reach by w − 1 = 3.
Neither panel ever touches the upper-right triangle: the window is still
causal.

**In code:** `sliding_window_mask` builds the band, `pairs_computed` counts its cells, `receptive_field` applies the mask layer after layer to find who can reach whom, and `reach` is the L(w − 1) formula.

### The rolling buffer: a cache that stops growing

**Everyday picture.** A whiteboard with room for exactly w notes. When it is
full, the next note goes over the oldest one. You never need a bigger board,
however long the meeting runs.

**Tiny worked example.** With w = 4, after token 9 the buffer holds the keys
and values of tokens 6, 7, 8 and 9. Token 10 overwrites token 6's slot. On
the running 8B example at 128k tokens, a 4,096-token window holds 4,096 × 128
KB = 0.54 GB instead of 17.2 GB: **32 times less**, and the same at 1M tokens.

```mermaid
flowchart LR
  subgraph B["buffer of w = 4 slots, after token 9"]
    s0["slot 0: token 8"]
    s1["slot 1: token 9"]
    s2["slot 2: token 6 (oldest)"]
    s3["slot 3: token 7"]
  end
  N["token 10 arrives"] -->|"overwrites the oldest"| s2
  B --> A["token 10 attends to<br/>tokens 7, 8, 9, 10"]
```

**Reading it:** the four slots are all the memory there is. Slot number is
token number modulo 4 (the remainder after dividing by 4), so token 10 lands
in slot 2, where token 6 sat. Token 6 was about to leave the window anyway,
so nothing the window needs is lost.

**Why it matters in practice.** Mistral 7B combined the window with this
rolling buffer. Several later model families interleave sliding-window layers
with a few full-attention layers: the local layers are cheap, and the full
layers keep long-range lookups exact. The weakness is plain from the diagram:
a fact beyond the reach is invisible, and a fact inside the reach has to
survive several hops.

**In code:** `sliding_window_decode` generates token by token with a buffer that never holds more than w keys, and returns the same outputs as masked attention with `sliding_window_mask`.

## 3. Sparse attention: a few long-distance lines

**Everyday picture.** An open-plan office. You mostly talk to the people at
the desks next to yours (local). Anyone can phone the front desk, and the
front desk hears from everyone (a *global* token): any two people are at
most two calls apart. Another design is the express train: you talk to your
neighbours and also to every fourth desk down the row (*strided*), so a
message can travel far in a few big jumps.

**Tiny worked example.** Sixteen tokens, window 4.

- The window alone: 1 + 2 + 3 + 4 × 13 = **58** pairs.
- Make token 0 global: the 12 rows past the window (tokens 4 to 15) add a
  pair each: **70** pairs, against **136** for full causal attention.
- Strided with stride 4: the last 4 tokens, plus every 4th token before them.
  Token 13 reads 10, 11, 12, 13 and then 9, 5, 1. Over all rows that is
  58 + 24 = **82** pairs.

```mermaid
flowchart LR
  G(("token 0<br/>global"))
  t3["token 3"] --- G
  t7["token 7"] --- G
  t11["token 11"] --- G
  t15["token 15"] --- G
  t14["token 14"] --- t15
  t13["token 13"] --- t14
```

**Reading it:** the circle is the global token. Every token has a line to
it, so token 15 can reach token 3 in two hops, through token 0, however long
the sequence. The short lines at the bottom are the ordinary local window.
The picture is sparse (few lines), yet no two tokens are far apart.

$$
M_{ij} = 1 \quad\text{when}\quad i - j \ge 0 \;\text{ and }\; \big(\, i - j < s \;\text{ or }\; (i - j) \bmod s = 0 \,\big)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $M_{ij}$ | 1 if query $i$ may read key $j$ | |
| $i - j$ | how far back key $j$ is | for $i = 13$: 0 to 13 |
| $s$ | the stride: the local width, and the jump between long-range keys | 4 |
| $\bmod$ | "modulo": the remainder after dividing; $(i - j) \bmod s = 0$ means "a whole number of strides back" | 8 mod 4 = 0 |
| and, or | both conditions must hold; at least one must hold | |

**In words:** "never read the future; read the last s tokens, and beyond
them every s-th token."

**With the numbers:** for i = 13, s = 4: distances 0 to 3 give keys 13, 12,
11, 10; distances 4, 8 and 12 give keys 9, 5, 1.

**In Python:**

```python
s = 4
# row 13: which keys does token 13 read?
[j for j in range(16) if 13 - j >= 0 and (13 - j < s or (13 - j) % s == 0)]  # → [1, 5, 9, 10, 11, 12, 13]
# strided pairs over all 16 rows
sum(1 for i in range(16) for j in range(i + 1) if i - j < s or (i - j) % s == 0)  # → 82
# a window of 4 plus global token 0
sum(1 for i in range(16) for j in range(i + 1) if i - j < 4 or j == 0)  # → 70
```

![Four 16 by 16 masks: full causal attention scores 136 pairs, the window 58, window plus a global first token 70, strided 82](figures/primer.ml.efficient_architectures.sparse_masks.svg)

**Reading it:** each panel is a mask, rows for queries and columns for keys,
dark where a score is computed; the title gives the count. The first panel
is full causal attention, a solid triangle. The window is a diagonal band.
Adding a global token fills in the first column: everyone reads token 0. The
strided pattern adds dotted diagonals every 4 columns: the express stops. At
16 tokens the savings look modest; with a fixed window they grow with n,
because the band stays w wide while the triangle keeps growing.

**Why it matters in practice.** The Sparse Transformer introduced strided
patterns; Longformer and BigBird combined windows with global tokens (BigBird
added a few random links too) to read documents of thousands of tokens.
StreamingLLM found that trained models park a lot of attention on the very
first tokens (*attention sinks*); a plain window drops them and falls apart,
while keeping the first few tokens plus a window lets a model stream
indefinitely. One caution: a sparse pattern only saves time if the GPU
kernel skips whole blocks of the score matrix, which is why real patterns
are built from blocks.

**In code:** `global_local_mask` adds global rows and columns to a window, `strided_mask` builds the express-stop pattern, and `pairs_computed` counts what each one scores. Any of them can be passed as the mask to `primer.ml.attention.scaled_dot_product_attention`, and every skipped pair gets a weight of exactly 0.

## 4. Linear attention: a pot instead of a guest list

**Everyday picture.** A potluck soup. In softmax attention every new guest
tastes every dish on the table, one by one, and then mixes a bowl: the more
dishes, the longer it takes. In linear attention every guest pours their dish
into one shared pot as they arrive, and a new guest takes a single ladle,
seasoned to their own taste. The pot never grows, and the ladle costs the
same for guest 3 as for guest 3 million.

The trick that makes the pot possible: softmax's score $e^{q \cdot k}$ cannot
be split into "a part that depends on q" times "a part that depends on k".
Replace it with $\phi(q) \cdot \phi(k)$, a dot product of transformed
vectors, and it can. Then the order of the matrix multiplies can be swapped:
$(\phi(Q)\phi(K)^\top)\,V = \phi(Q)\,(\phi(K)^\top V)$. The left side builds an
n × n matrix; the right side builds a small d × d one (the pot) and never
builds the big one. Matrix multiplication allows regrouping like this
(*associativity*), which `primer.notation` covers under matrix multiply.

**Tiny worked example.** Three tokens with 2-number keys (1, 0), (0, 1),
(1, 1) and one-number values 2, 4, 6. The third token's query is (1, 0). Use
the feature map φ(x) = elu(x) + 1, which for a positive number is simply
x + 1 and for zero or a negative number is $e^x$ (always above zero).

1. φ of the keys: (2, 1), (1, 2), (2, 2).
2. Pour into the pot: S = (2, 1)·2 + (1, 2)·4 + (2, 2)·6 = (20, 22), and the
   running total of keys z = (5, 5).
3. Ladle with φ(q) = (2, 1): (2·20 + 1·22) / (2·5 + 1·5) = 62 / 15 = **4.13**.

The slow way agrees: the weights φ(q)·φ(k) are 5, 4 and 6, so the output is
(5·2 + 4·6 + 6·6) / 15 = 62/15. Softmax attention on the same numbers gives
**4.00**: linear attention is a *different* attention, not a faster copy of
the same one.

```mermaid
flowchart LR
  subgraph T["each new token i"]
    K["φ(k_i)"]
    V["v_i"]
    Qi["φ(q_i)"]
  end
  K & V -->|"add φ(k_i) v_iᵀ"| S[("pot S<br/>d_k × d_v numbers")]
  K -->|"add φ(k_i)"| Z[("total z<br/>d_k numbers")]
  Qi --> R["output = φ(q_i)ᵀ S / φ(q_i)ᵀ z"]
  S --> R
  Z --> R
```

**Reading it:** each token does two things. Its key and value go into the
pot S (and its key into the running total z, used to normalize). Its query
reads the pot once. S and z have a fixed size set by the head width, not by
the number of tokens, so this is a recurrent network: a state updated once
per token, with no cache that grows.

$$
o_i = \frac{\sum_{j \le i} \big(\phi(q_i) \cdot \phi(k_j)\big)\, v_j}{\sum_{j \le i} \phi(q_i) \cdot \phi(k_j)}
= \frac{\phi(q_i)^\top S_i}{\phi(q_i)^\top z_i},
\qquad S_i = S_{i-1} + \phi(k_i)\, v_i^\top,
\qquad z_i = z_{i-1} + \phi(k_i)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $o_i$ | the output for token $i$ | $o_3 = 4.13$ |
| $q_i$, $k_j$, $v_j$ | query of token $i$; key and value of token $j$ | $q_3 = (1, 0)$ |
| $\phi$ | the feature map, applied to each number: $x + 1$ if $x > 0$, else $e^x$; always positive so no weight is negative | φ(1, 0) = (2, 1) |
| $\sum_{j \le i}$ | add up over every token $j$ up to and including $i$ (causal) | $j$ = 1, 2, 3 |
| $\phi(q_i) \cdot \phi(k_j)$ | the unnormalized weight, a dot product in place of $e^{q \cdot k}$ | 5, 4, 6 |
| $v_i^\top$ | the value laid on its side as a row | |
| $\phi(k_i)\, v_i^\top$ | an **outer product**: a column times a row, giving a small table whose entry (m, c) is $\phi(k_i)_m \, v_{i,c}$ | (2, 1) × 2 = (4, 2) |
| $S_i$ | the pot after token $i$: the sum of those tables | (20, 22) |
| $z_i$ | the running sum of $\phi(k_j)$, for the denominator | (5, 5) |
| $\phi(q_i)^\top S_i$ | the query's ladle: its dot product with each column of $S_i$ | 62 |

**In words:** "each output is a weighted average of the values so far, with
weights φ(q)·φ(k); because those weights split into a query part and a key
part, the key-and-value part can be kept as a running sum, and each query
reads that sum once."

**With the numbers:** S₃ = (20, 22), z₃ = (5, 5), φ(q₃) = (2, 1):
o₃ = (40 + 22) / (10 + 5) = 62/15 = 4.133.

**In Python:**

```python
import math
def phi(v):
    # elu(x) + 1: x + 1 above zero, e^x at or below it
    return [x + 1 if x > 0 else math.exp(x) for x in v]
def dot(a, b):
    return sum(a_m * b_m for a_m, b_m in zip(a, b))
keys = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
values = [2.0, 4.0, 6.0]
q3 = [1.0, 0.0]
[phi(k) for k in keys]  # → [[2.0, 1.0], [1.0, 2.0], [2.0, 2.0]]
# S: the running sum of φ(k_j) v_j; z: the running sum of φ(k_j)
S = [sum(phi(k)[m] * v for k, v in zip(keys, values)) for m in range(2)]
S  # → [20.0, 22.0]
z = [sum(phi(k)[m] for k in keys) for m in range(2)]
z  # → [5.0, 5.0]
# o_3 = φ(q)ᵀS / φ(q)ᵀz
round(dot(phi(q3), S) / dot(phi(q3), z), 3)  # → 4.133
# the quadratic form: the weights φ(q)·φ(k_j), then their weighted average
weights = [dot(phi(q3), phi(k)) for k in keys]
weights  # → [5.0, 4.0, 6.0]
round(dot(weights, values) / sum(weights), 3)  # → 4.133
# softmax attention on the same numbers (unscaled), for contrast
e = [math.exp(dot(q3, k)) for k in keys]
round(dot(e, values) / sum(e), 3)  # → 4.0
```

![Side by side on the same queries and keys: softmax attention puts almost half of each late row on one key, while linear attention spreads each row thinly, its largest weight about 0.2](figures/primer.ml.efficient_architectures.linear_vs_softmax.svg)

**Reading it:** the two heatmaps use the same ten queries and keys; rows are
queries, columns are keys, and darker means more weight. Both are causal
triangles and every row sums to 1. On the left, softmax picks favourites:
the exponential stretches gaps between scores (see `primer.ml.attention`), so
in the last five rows the largest weight averages 0.46. On the right, linear
attention spreads the same rows thinly: its largest weight averages 0.21. The
queries here are twice the usual size, and that is the telling part: at the
usual size the two numbers are 0.28 and 0.20, so making a query more decisive
sharpens softmax a lot and linear attention hardly at all, because φ(q)·φ(k)
never stretches a gap the way $e^x$ does. That bluntness is the price of the
pot: a fixed-size summary cannot pick out one exact token as sharply.

**Why it matters in practice.** The work drops from n² × d to n × d², and
generation needs only the pot, not a cache. The catch is recall: asked to
copy back a specific token from far away, a fixed-size pot does worse than a
full cache. Later work added forgetting (decay) to the pot, which turns out
to be exactly the state-space models of the next section; the Mamba-2 paper
shows the two views describe the same computation.

**In code:** `feature_map` is φ, `linear_attention_quadratic` builds all n × n weights (from `linear_attention_weights`) and blends the values, and `linear_attention_recurrent` gets the same outputs from the two running sums, returning the fixed-size S and z.

## 5. State-space models: a summary with a fixed update rule

### A recurrence that is also a convolution

**Everyday picture.** A cup of tea cooling on a desk. Every minute it keeps
some fraction of its heat and gains whatever hot water you pour in. Its
temperature right now is a summary of everything you ever poured, with older
pours counting less. That is a recurrent network's one-page summary (see
`primer.ml.cnn_rnn`), with one difference: the rewrite rule is *linear*
(multiply and add, nothing more), and linearity buys a second way to compute
the same thing.

**Tiny worked example.** The state keeps half of itself each step: A = 0.5,
B = 1, C = 1. Inputs x = (1, 0, 0, 2).

| step t | x_t | h_t = 0.5 · h_(t−1) + x_t | y_t |
|---|---|---|---|
| 1 | 1 | 0.5 · 0 + 1 = 1 | 1 |
| 2 | 0 | 0.5 · 1 + 0 = 0.5 | 0.5 |
| 3 | 0 | 0.5 · 0.5 + 0 = 0.25 | 0.25 |
| 4 | 2 | 0.5 · 0.25 + 2 = 2.125 | 2.125 |

Unroll the loop and each output is a weighted sum of all past inputs with
weights 1, 0.5, 0.25, 0.125 for "now, 1 step ago, 2 steps ago, 3 steps ago":
y₄ = 1·2 + 0.5·0 + 0.25·0 + 0.125·1 = 2.125. The same answer, with no loop.

```mermaid
flowchart LR
  X["inputs x_1 ... x_T"] --> R["recurrence<br/>h_t = A h_(t−1) + B x_t<br/>one step at a time"]
  X --> K["convolution<br/>slide the kernel (CB, CAB, CA²B, ...)<br/>all positions at once"]
  R --> Y["the same outputs y_1 ... y_T"]
  K --> Y
```

**Reading it:** two roads from the same inputs to the same outputs. The top
road is how the model *generates*: one token at a time, carrying a small
state h, with constant memory. The bottom road is how it *trains*: the
kernel is computed once, and every output is a weighted sum that can be done
in parallel, like a convolution in `primer.ml.cnn_rnn`. Classic RNNs only had
the top road, which is why they trained slowly.

$$
h_t = A\,h_{t-1} + B\,x_t, \qquad y_t = C\,h_t
\qquad\Longleftrightarrow\qquad
y_t = \sum_{k=0}^{t-1} \big(C A^{k} B\big)\, x_{t-k}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $x_t$ | the input at step $t$ (one channel) | (1, 0, 0, 2) |
| $h_t$ | the state after step $t$: $N$ numbers (here $N$ = 1); $h_0 = 0$ | 1, 0.5, 0.25, 2.125 |
| $A$ | $N \times N$ matrix: how the old state carries over (its size sets how fast things fade) | 0.5 |
| $B$ | how the input is written into the state | 1 |
| $C$ | how the state is read out | 1 |
| $y_t$ | the output at step $t$ | |
| $\Longleftrightarrow$ | "the same thing, written another way" | |
| $A^k$ | $A$ multiplied by itself $k$ times; $A^0$ is "do nothing" | 0.5³ = 0.125 |
| $C A^k B$ | the **kernel**: how much an input $k$ steps ago still counts now | 1, 0.5, 0.25, 0.125 |
| $\sum_{k=0}^{t-1}$ | add over every look-back distance $k$ from 0 to $t - 1$ | |

**In words:** "the state keeps a fraction of itself and adds the new input,
and the output reads the state; equivalently, the output is every past input
weighted by how much it has faded since."

**With the numbers:** y₄ = C A⁰ B x₄ + C A¹ B x₃ + C A² B x₂ + C A³ B x₁ =
1·2 + 0.5·0 + 0.25·0 + 0.125·1 = 2.125.

**In Python:**

```python
A, B, C = 0.5, 1.0, 1.0
x = [1.0, 0.0, 0.0, 2.0]
h, y = 0.0, []
for x_t in x:
    # keep half of the old state, add the new input
    h = A * h + B * x_t
    y.append(C * h)
y  # → [1.0, 0.5, 0.25, 2.125]
# the kernel C A^k B, for k = 0, 1, 2, 3
kernel = [C * A ** k * B for k in range(4)]
kernel  # → [1.0, 0.5, 0.25, 0.125]
# y_4 as a weighted sum of all four inputs, newest first
sum(kernel[k] * x[3 - k] for k in range(4))  # → 2.125
```

![Three kernels: with A = 0.5 an input is forgotten within about 5 steps, with 0.9 within about 40, with 0.99 it still counts about a fifth after 150 steps](figures/primer.ml.efficient_architectures.ssm_kernel.svg)

**Reading it:** the x-axis is how many steps ago an input arrived; the
y-axis is how much it still counts (the kernel C A^k B). Each curve is one
value of A. With A = 0.5 the curve collapses almost at once: short memory.
With A = 0.99 it is still well above zero after 150 steps: long memory. A
real SSM layer has thousands of channels, each with its own A, so it
remembers at many time scales at once. S4 was the first to make this work on
very long sequences, by choosing and computing these kernels carefully.

**In code:** `ssm_recurrent` runs the loop, `ssm_kernel` builds C A^k B, and `ssm_convolution` gets the same outputs from the kernel with a single NumPy convolution.

### Selective: letting each token decide what to keep (Mamba)

**Everyday picture.** A note-taker with a dial. For filler words ("um", "so",
"anyway") they barely touch their notes. For a name or a number they wipe the
relevant line and write the new fact. A fixed SSM uses the same dial setting
for every word, so it must either write everything (and forget quickly) or
write little (and never take in the important word properly). Mamba reads the
dial setting off each token itself: that is what *selective* means.

The dial is a step size Δ. The parameters of the state update are derived
from it at every step (this is called *discretization*: turning a continuous
rate of change into one step's keep and write amounts):

- keep factor $\bar{A} = e^{\Delta a}$, with $a$ negative, so a big Δ makes
  it nearly 0 (forget) and a tiny Δ nearly 1 (keep);
- write factor $\bar{B}$, which goes the opposite way.

**Tiny worked example.** One channel, a = −1, b = 1. A token that carries a
marker gets Δ ≈ 5; an ordinary token gets Δ ≈ 0.0067.

| token | Δ | keep Ā = e^(−Δ) | write B̄ = 1 − e^(−Δ) | effect |
|---|---|---|---|---|
| marked | 5.007 | 0.0067 | 0.9933 | replace the state with this token |
| ordinary | 0.0067 | 0.9933 | 0.0067 | leave the state almost untouched |

The recall task: a sequence of small noise values with one marked 7 in third
place, then nine more noise values. The selective SSM writes the 7 almost
fully and then keeps it: at the end its state is **6.55**. With one fixed Δ
for every token, the best any setting manages is **0.26**: a Δ big enough to
write the 7 also lets the nine later tokens overwrite it.

```mermaid
flowchart LR
  U["token u_t"] --> D["Δ_t = softplus(w · u_t + β)<br/>how much this token matters"]
  D --> AB["Ā_t = e^(Δ_t a): keep<br/>B̄_t: write"]
  U --> XV["x_t: what to write"]
  AB --> H["h_t = Ā_t h_(t−1) + B̄_t x_t"]
  XV --> H
  HP["h_(t−1)"] --> H
  H --> Y["y_t = C h_t"]
```

**Reading it:** the new part is the top path. The token itself feeds a small
function that produces its step size Δ, which sets how much of the old state
to keep and how much of this token to write. In Mamba the matrices B and C
are also computed from the token, by the same kind of path. Everything below
is the recurrence from before, except that Ā and B̄ now change at every step.

$$
\Delta_t = \operatorname{softplus}(w\,u_t + \beta), \qquad
\bar{A}_t = e^{\Delta_t a}, \qquad
\bar{B}_t = \frac{e^{\Delta_t a} - 1}{a}\, b, \qquad
h_t = \bar{A}_t\, h_{t-1} + \bar{B}_t\, x_t
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $u_t$ | what the model can see about token $t$ (here, just its marker, 0 or 1) | 1 for the 7 |
| $w$, $\beta$ | learned weight and offset that turn $u_t$ into a step size | 10, −5 |
| softplus | $\log(1 + e^x)$: a smooth ramp that is always positive, about $x$ for large $x$ and about 0 for very negative $x$ | softplus(5) = 5.007 |
| $\Delta_t$ | the step size for token $t$: how much this token matters | 5.007 or 0.0067 |
| $a$ | a negative learned rate; more negative means faster forgetting | −1 |
| $b$ | how strongly inputs are written | 1 |
| $\bar{A}_t$ | this step's keep factor ("A-bar") | 0.0067 or 0.9933 |
| $\bar{B}_t$ | this step's write factor ("B-bar") | 0.9933 or 0.0067 |
| $x_t$, $h_t$ | the value written, and the state after step $t$ | 7, then 6.95 |

**In words:** "each token computes how much it matters; that sets how much
of the old state survives and how much of the token gets written; then the
usual update runs with those per-token amounts."

**With the numbers:** the marked 7 has Δ = softplus(10·1 − 5) = 5.007, so
Ā = e^(−5.007) = 0.0067 and B̄ = (0.0067 − 1)/(−1) · 1 = 0.9933: the state
becomes about 0.9933 × 7 = 6.95. Each ordinary token after it keeps 0.9933 of
the state, and nine of them leave about 6.95 × 0.9933⁹ = 6.55.

**In Python:**

```python
import math
def softplus(x):
    return math.log(1 + math.exp(x))
a, b = -1.0, 1.0
# Δ for a marked token, then for an ordinary one
d_mark, d_plain = softplus(10 * 1 - 5), softplus(10 * 0 - 5)
round(d_mark, 3), round(d_plain, 4)  # → (5.007, 0.0067)
# Ā_t and B̄_t for the marked token: keep almost nothing, write almost everything
keep_mark, write_mark = math.exp(d_mark * a), (math.exp(d_mark * a) - 1) / a * b
round(keep_mark, 4), round(write_mark, 4)  # → (0.0067, 0.9933)
# and for an ordinary token: keep almost everything, write almost nothing
keep_plain = math.exp(d_plain * a)
round(keep_plain, 4)  # → 0.9933
# the 7 is written once, then kept through nine ordinary tokens
round(write_mark * 7 * keep_plain ** 9, 2)  # → 6.55
```

![The selective state jumps to about 7 at the marked token and holds near 6.5 to the end, while the best fixed step peaks near 0.6 and ends at 0.26, and a large fixed step just copies the latest noise](figures/primer.ml.efficient_architectures.selective_recall.svg)

**Reading it:** the x-axis is the position in the sequence; grey bars are
the input values, with the marked 7 at position 2. The blue line is the
selective SSM's state: it jumps to the 7 and then barely moves, because
every later token has a tiny Δ. The red line is the best fixed Δ: it can
only write a small fraction of each input, so the 7 lifts it to about 0.6 at
most and it ends at 0.26. The
orange line is a large fixed Δ: it writes every token fully, so it tracks
whatever arrived last and the 7 is gone by the next step. Only the selective
model can both write the important token and ignore the rest.

**In code:** `discretize` turns Δ into Ā and B̄, `softplus` and `step_sizes` compute each token's Δ, `selective_ssm` runs the per-token recurrence, and `selective_recall` and `time_invariant_recall` run the recall task.

### Training in parallel when the kernel keeps changing: the scan

**Everyday picture.** A relay race where each runner must know the total
time so far. Done one after another it takes as long as the whole race. But
"multiply by a, then add b" steps can be *merged*: two consecutive steps are
themselves one step of the same shape. So pairs of runners merge their legs,
then pairs of pairs, like a knockout tournament: log₂ n rounds instead of n (log₂ n is how many
times n can be halved before reaching 1: 10 for 1,024).

**Tiny worked example.** The four steps of the tea example are
(a, b) = (0.5, 1), (0.5, 0), (0.5, 0), (0.5, 2), meaning "h becomes a·h + b".
Round 1: every step merges with the one before it. Round 2: every step merges
with the result two places before it. After 2 rounds (log₂ 4 = 2), the b
parts are 1, 0.5, 0.25, 2.125: every state from the loop.

```mermaid
flowchart TB
  s1["(0.5, 1)"]
  s2["(0.5, 0)"]
  s3["(0.5, 0)"]
  s4["(0.5, 2)"]
  s1 --> r2["(0.25, 0.5)"]
  s2 --> r2
  s2 --> r3["(0.25, 0)"]
  s3 --> r3
  s3 --> r4["(0.25, 2)"]
  s4 --> r4
  s1 --> f3["(0.125, 0.25)"]
  r3 --> f3
  r2 --> f4["(0.0625, 2.125)"]
  r4 --> f4
```

**Reading it:** the top row holds the four steps. Each arrow pair is one
merge; every merge in a row happens at the same time. The middle row is
round 1, the bottom row round 2. Read the second number in each final box
(and in s1 and r2, which were already complete): 1, 0.5, 0.25, 2.125, the
states from the table above. A selective SSM cannot use the convolution road,
because its Ā changes every step, but it can use this one.

$$
(a_1, b_1) \circ (a_2, b_2) = \big(a_1 a_2,\; a_2 b_1 + b_2\big)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $(a, b)$ | one step: "multiply the state by $a$, then add $b$" | (0.5, 1) |
| $\circ$ | "do the first step, then the second": merging two steps into one | |
| $a_1 a_2$ | the combined multiplier | 0.25 |
| $a_2 b_1 + b_2$ | the first step's addition, shrunk by the second step, plus the second's own | 0.5·1 + 0 = 0.5 |

**In words:** "doing step 1 then step 2 is the same as one step that
multiplies by both and adds step 1's contribution, faded by step 2."

**With the numbers:** (0.25, 0.5) ∘ (0.25, 2) = (0.0625, 0.25·0.5 + 2) =
(0.0625, 2.125), the last box in the diagram.

**In Python:**

```python
def merge(first, then):
    a1, b1 = first
    a2, b2 = then
    return (a1 * a2, a2 * b1 + b2)
steps = [(0.5, 1.0), (0.5, 0.0), (0.5, 0.0), (0.5, 2.0)]
# round 1: each step absorbs the one just before it
r1 = [steps[0]] + [merge(steps[t - 1], steps[t]) for t in range(1, 4)]
r1  # → [(0.5, 1.0), (0.25, 0.5), (0.25, 0.0), (0.25, 2.0)]
# round 2: each absorbs the result two places before it
r2 = r1[:2] + [merge(r1[t - 2], r1[t]) for t in range(2, 4)]
[b for a, b in r2]  # → [1.0, 0.5, 0.25, 2.125]
```

**Why it matters in practice.** This is how Mamba trains on long sequences
at GPU speed despite being recurrent: a parallel scan, written so the state
stays in fast on-chip memory. At generation time it switches back to the
plain loop, one token at a time, with a state that never grows.

**In code:** `parallel_scan` runs the rounds for any length and returns the states and the number of rounds: 10 for 1,024 steps.

### Constant memory, and hybrids

**Everyday picture.** An SSM travels with a backpack of fixed size; the KV
cache is a suitcase that grows with every token. A *hybrid* model mostly uses
backpacks but brings one suitcase for every few layers, so it can still look
up an exact earlier token when it needs to.

**Tiny worked example.** One SSM layer with 4,096 channels and a 16-number
state per channel holds 4,096 × 16 × 2 bytes = **128 KB**, at 10 tokens or at
10 million. One attention layer of the running 8B example holds **537 MB** at
128k tokens. Build 32 layers as 4 attention layers and 28 SSM layers (one in
eight, the ratio Jamba uses) and the cache at 128k drops from 17.2 GB to
**2.15 GB**.

```mermaid
flowchart TB
  I["tokens in"] --> M1["SSM layer<br/>fixed state"] --> M2["SSM layer"] --> M3["..."] --> A1["attention layer<br/>KV cache grows"]
  A1 --> M4["SSM layer"] --> M5["..."] --> A2["attention layer"] --> O["next-token scores"]
```

**Reading it:** most boxes are SSM layers, each carrying a fixed-size state.
Every so often an attention layer sits in the stack; only those keep a KV
cache. The memory bill therefore scales with the number of attention layers,
not with the total depth, while the attention layers are still there for
exact copying and lookup.

$$
\text{bytes} = L_{\text{att}} \cdot n \cdot 2\,H_{kv}\,d_h\,b \;+\; (L - L_{\text{att}}) \cdot D \cdot N \cdot b
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $L$ | total layers | 32 |
| $L_{\text{att}}$ | how many of them are attention layers | 4 |
| $n$ | context length | 131,072 |
| $2\,H_{kv}\,d_h\,b$ | one attention layer's cache per token (from section 1) | 4,096 bytes |
| $D$ | channels in an SSM layer | 4,096 |
| $N$ | state numbers per channel | 16 |
| $b$ | bytes per number | 2 |

**In words:** "the attention layers pay per token as before; the SSM layers
pay a fixed amount that does not depend on n at all."

**With the numbers:** 4 × 131,072 × 4,096 = 2,147,483,648 bytes, plus
28 × 4,096 × 16 × 2 = 3,670,016 bytes: 2.15 GB, about one eighth of 17.2 GB.

**In Python:**

```python
n, H_kv, d_h, b = 131_072, 8, 128, 2
D, N = 4096, 16
# one SSM layer's state, in KB, at any context length
D * N * b / 1024  # → 128.0
# one attention layer's KV cache at 128k tokens, in MB
n * 2 * H_kv * d_h * b / 1e6  # → 536.870912
# 32 layers: all attention, then 4 attention + 28 SSM, in GB
round(32 * n * 2 * H_kv * d_h * b / 1e9, 2)  # → 17.18
round((4 * n * 2 * H_kv * d_h * b + 28 * D * N * b) / 1e9, 2)  # → 2.15
```

**Why it matters in practice.** Pure SSMs are strong at language modelling
but weaker than attention at copying and exact recall over long contexts,
for the same reason as linear attention: a fixed-size state is a summary.
Hybrids such as Jamba keep a few attention layers for those jobs and get most
of the SSM's memory savings.

**In code:** `ssm_state_bytes` is the fixed backpack, and `hybrid_cache_bytes` adds up a mixed stack.

## 6. Compressing the KV cache

### Fewer key/value heads: GQA and MQA, a recap

**Everyday picture.** Colleagues sharing one reference binder instead of each
keeping a personal copy: everyone still asks their own questions, but the
shelf holds fewer binders.

**Tiny worked example.** On the running example (32 layers, 128 numbers per
head, 16-bit), the cache per token for different numbers of KV heads:

| Design | KV heads | Cache per token |
|---|---|---|
| multi-head attention | 32 | 512 KB |
| grouped-query attention | 8 | 128 KB |
| multi-query attention | 1 | 16 KB |

The diagram of shared heads and the full memory arithmetic live in
`primer.ml.attention` and `primer.ml.inference`; everything below stacks on
top of whichever of these a model uses.

### Latent KV: cache the ingredients, cook on demand

**Everyday picture.** A restaurant that stores ingredients, not finished
dishes. Every head's keys and values can be cooked from a short list of
ingredients (the *latent* vector) with a fixed recipe shared by every token.
Store the ingredients, keep the recipe once, and cook when a query needs it.
Better still, the recipe can be folded into the query itself, so nothing is
ever cooked at all.

**Tiny worked example.** A token x = (1, 0, 2, 1), two heads of width 2.
Ordinary attention would cache 2 heads × 2 numbers × (key and value) = 8
numbers. Instead:

1. Squeeze: c = x W_down = (1 + 2, 0 + 1) = **(3, 1)**. Only these 2 numbers
   are cached: 4 times smaller.
2. When needed, expand: k = c W_uk = **(3, 1, 3, 1)**, both heads' keys.
3. Or fold the expansion into the query: for q = (1, 2, 0, 1), q · k = 6, and
   (q W_ukᵀ) · c = (1, 3) · (3, 1) = 6. Same score, and k was never built.

DeepSeek-V2 uses this at scale: its 128 heads of width 128 would cache 32,768
numbers per token per layer; its latent caches 512, plus 64 for a small key
that carries position information: **576, about 57 times less**.

```mermaid
flowchart LR
  X["token x<br/>(d_model numbers)"] -->|"W_down"| C[("cache: latent c<br/>d_c numbers")]
  C -->|"W_uk"| K["keys for every head"]
  C -->|"W_uv"| V["values for every head"]
  Qn["query q"] -->|"absorbed: q W_ukᵀ"| QL["query in latent space"]
  QL -->|"score directly against c"| C
```

**Reading it:** the cylinder is all that is stored per token. The two arrows
to the right show the plain way: rebuild keys and values from the latent.
The bottom path shows the absorbed way: translate the query into latent space
once, then score it against the cached latents directly, and mix latents
before expanding with W_uv. Both paths give identical outputs; the absorbed
one never materializes the big keys and values.

$$
c_t = x_t W^{D}, \qquad k_t = c_t W^{UK}, \qquad v_t = c_t W^{UV},
\qquad q \cdot k_t = \big(q\, {W^{UK}}^{\top}\big) \cdot c_t
$$

**Symbols**

| Symbol | Meaning here | Shape / example |
|---|---|---|
| $x_t$ | token $t$'s vector coming into the layer | $d_\text{model}$; (1, 0, 2, 1) |
| $W^{D}$ | the learned "down" projection that squeezes | $d_\text{model} \times d_c$ |
| $c_t$ | the latent: the only thing cached | $d_c$; (3, 1) |
| $W^{UK}$, $W^{UV}$ | learned "up" projections to every head's keys and values | $d_c \times H d_h$ |
| $k_t$, $v_t$ | token $t$'s keys and values for all heads, side by side | $H d_h$; $k$ = (3, 1, 3, 1) |
| $q$ | a query (one head's slice, or all heads side by side) | (1, 2, 0, 1) |
| ${W^{UK}}^{\top}$ | $W^{UK}$ transposed (rows become columns) | $H d_h \times d_c$ |
| $q\,{W^{UK}}^{\top}$ | the query translated into latent space | $d_c$; (1, 3) |

**In words:** "squeeze each token into a short latent and cache only that;
keys and values are the latent times fixed up-projections, so a query can
be moved into latent space once and scored against the cache directly."

**With the numbers:** c = (3, 1), k = (3, 1, 3, 1), q · k = 3 + 2 + 0 + 1 =
6, and (1, 3) · (3, 1) = 3 + 3 = 6.

**In Python:**

```python
x = [1, 0, 2, 1]
W_down = [[1, 0], [0, 1], [1, 0], [0, 1]]
W_uk = [[1, 0, 1, 0], [0, 1, 0, 1]]
def vecmat(v, M):
    # row vector times matrix: entry c is Σ_r v_r M[r][c]
    return [sum(v[r] * M[r][c] for r in range(len(v))) for c in range(len(M[0]))]
# c = x W_down: the only thing cached
c = vecmat(x, W_down)
c  # → [3, 1]
# k = c W_uk: both heads' keys, rebuilt on demand
k = vecmat(c, W_uk)
k  # → [3, 1, 3, 1]
q = [1, 2, 0, 1]
sum(q_m * k_m for q_m, k_m in zip(q, k))  # → 6
# absorbed: move q into latent space, then score against c
W_uk_T = [list(col) for col in zip(*W_uk)]
q_latent = vecmat(q, W_uk_T)
q_latent, sum(a * b for a, b in zip(q_latent, c))  # → ([1, 3], 6)
# DeepSeek-V2's shape: cached numbers per token per layer
2 * 128 * 128, 512 + 64, round(2 * 128 * 128 / (512 + 64), 1)  # → (32768, 576, 56.9)
```

Why can so few numbers stand in for so many? Because every head's keys and
values are built from the same token, they are highly redundant. In the
language of `primer.notation`, the key projection W^D W^UK has **low rank**:
however many numbers it outputs, they all vary along only d_c independent
directions. Storing those d_c coordinates loses nothing that projection can
produce. One wrinkle: rotary position embeddings (see `primer.ml.positional`)
rotate each key by its position, which breaks the absorption trick, so
DeepSeek-V2 carries position in that separate small 64-number key.

![Cache per token on the 32-layer example: 512 KB for 32 KV heads, 128 KB for 8, 36 KB for a 576-number latent, 33 KB for 8 heads at 4 bits, 16 KB for one head](figures/primer.ml.efficient_architectures.kv_per_token.svg)

**Reading it:** each bar is the cache one token costs across all 32 layers
of the running example, for one design. The top bar is classic multi-head
attention; every bar below it is a way of shrinking it. Sharing heads (GQA,
MQA), squeezing into a latent, and cutting bits land in the same range, tens
of kilobytes instead of hundreds, by different routes that can be combined.

**In code:** `LatentKVAttention` holds the four projections; `LatentKVAttention.compress` produces the cache, `LatentKVAttention.attend` expands it into keys and values, and `LatentKVAttention.attend_absorbed` gets the identical output without expanding. `latent_kv_worked_example` is the (3, 1) example and `latent_kv_bytes_per_token` the memory arithmetic.

### Quantizing the cache

**Everyday picture.** The same move as for weights in `primer.ml.inference`:
write each number to the nearest tenth instead of the nearest thousandth,
with one ruler per stored vector so one big number does not coarsen all the
others.

**Tiny worked example.** A cached key (0.7, −0.3, 0.2, 0.04) at 4 bits (codes
−7 to 7). The step size is 0.7 / 7 = 0.1, the codes are **(7, −3, 2, 0)**, and
reading back gives (0.7, −0.3, 0.2, 0): the tiny 0.04 is lost. A 128-number
head vector drops from 256 bytes to 64 bytes of codes plus a 2-byte scale:
66 bytes, **3.9 times smaller**.

```mermaid
flowchart LR
  KV["new key or value<br/>16-bit numbers"] -->|"scale = max / 7<br/>code = round(x / scale)"| ST[("cache: 4-bit codes<br/>+ one scale per vector")]
  ST -->|"code × scale"| R["approximate key or value"]
  R --> A["attention as usual"]
```

**Reading it:** writing to the cache quantizes once per token; reading from
it multiplies back. Attention itself is unchanged; it just sees slightly
rounded keys and values. The saving is on the cylinder, the part that grows
with every token.

$$
\text{bytes per token} = 2\,L\,H_{kv}\left(d_h \cdot \frac{\text{bits}}{8} + \frac{\text{scale bits}}{8}\right)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $2\,L\,H_{kv}$ | how many head vectors one token stores: a key and a value per KV head per layer | 2 × 32 × 8 = 512 |
| $d_h$ | numbers per head vector | 128 |
| bits / 8 | bytes per code | 4-bit: 0.5 |
| scale bits / 8 | bytes for the one scale each vector carries | 16-bit: 2 |

**In words:** "every stored head vector costs its codes plus one scale, and
a token stores a key vector and a value vector per head per layer."

**With the numbers:** 2 × 32 × 8 × (128 × 0.5 + 2) = 512 × 66 = 33,792
bytes, against 131,072 at 16 bits: 3.9 times smaller.

**In Python:**

```python
k = [0.7, -0.3, 0.2, 0.04]
# one scale per vector: the largest magnitude lands on code 7
s = max(abs(k_j) for k_j in k) / 7
round(s, 3)  # → 0.1
codes = [round(k_j / s) for k_j in k]
codes  # → [7, -3, 2, 0]
[round(s * c, 2) for c in codes]  # → [0.7, -0.3, 0.2, 0.0]
L, H_kv, d_h = 32, 8, 128
# bytes per token: 16-bit, then 4-bit codes plus a 16-bit scale per vector
2 * L * H_kv * d_h * 16 / 8, 2 * L * H_kv * (d_h * 4 / 8 + 16 / 8)  # → (131072.0, 33792.0)
```

**Why it matters in practice.** On random test data, an 8-bit cache moves
the attention output by under 1% and a 4-bit cache by about 12%; trained
models tolerate this far better than random data suggests, and careful
schemes go lower. Keys tend to have a few channels with consistently large
values, so KIVI quantizes keys per channel and values per token and reaches
2 bits. Quantization stacks with everything above: a GQA model with a sliding
window and a 4-bit cache enjoys all three savings.

**In code:** `quantize_kv` quantizes each cached vector with its own scale via `primer.ml.inference.quantize`, `quantized_kv_bytes_per_token` is the formula, and `quantized_cache_error` measures how far attention's output moves.

## Putting it together

![On log axes from 1k to 1M tokens, multi-head and grouped-query caches climb past 80 GB, latent and 4-bit caches climb 4 times lower, the hybrid 8 times lower, while the sliding window flattens at 0.5 GB and the SSM state stays at 4 MB](figures/primer.ml.efficient_architectures.memory_vs_context.svg)

**Reading it:** the x-axis is context length and the y-axis is cache memory
for one conversation, both logarithmic, with the dashed line at 80 GB. Lines
with the same slope grow the same way (in proportion to n); compression moves
a line down without changing its slope. The sliding window bends flat at
4,096 tokens, and the SSM is flat from the start. Flat lines are what make
million-token contexts affordable; the price is that they no longer keep
every token exactly.

| Technique | What it cuts | What it gives up |
|---|---|---|
| Sliding window | compute to n·w, cache to w tokens | direct access beyond the window |
| Sparse (global, strided) | compute | exact long-range pairs not in the pattern |
| Linear attention | compute to n·d², cache to a fixed state | sharp, exact recall |
| SSM / Mamba | the same | the same, softened by selectivity |
| Hybrid | most of the cache | a little of both |
| GQA / MQA | cache by the sharing factor | a little modelling capacity |
| Latent KV | cache by d_c / (2·H·d_h) | extra matrix work, care with positions |
| Quantized KV | cache by 16 / bits | a little precision |

**In code:** `cache_bytes_by_method` computes every line in the figure for a given context length.

## In 20 seconds

- Long context has two bills: attention scores grow with n², and the KV
  cache grows with n (per layer, per conversation). At 1M tokens the cache
  of an 8B-class model alone is about 137 GB.
- **Sliding window:** each token reads the last w; stacked layers still reach
  L·(w − 1) back, and a rolling buffer caps the cache at w tokens.
  **Sparse patterns** add a few global or strided links so any two tokens are
  a hop or two apart.
- **Linear attention** replaces e^(q·k) with φ(q)·φ(k), which turns attention
  into a running sum: O(n) work and a fixed-size state, but blurrier recall.
- **State-space models** update a fixed state linearly; fixed ones train as a
  convolution, **selective** ones (Mamba) let each token set how much to keep
  and write, and train with a parallel scan. **Hybrids** keep a few attention
  layers for exact lookup.
- **Compress the cache:** share KV heads (GQA/MQA), cache a small latent and
  expand on demand (latent KV), and store fewer bits per number.

## Self-test questions

**Why does doubling the context quadruple attention's compute but only
double its cache?**
Every token's query is scored against every key, so the scores form an
n × n table: doubling n quadruples it. The cache stores one key and one value
per token per layer, a list that grows by one entry per token, so doubling n
doubles it.

**A model uses a 4,096-token sliding window in all 32 layers. Can token
100,000 be influenced by token 1?**
Yes, in principle: information moves up to w − 1 = 4,095 positions per
layer, so 32 layers reach 131,040 positions back. In practice it must be
relayed through about 25 intermediate tokens and layers, so it arrives
weakened; direct, exact lookup only works within the window.

**What does a global token do in a sparse pattern, and why is it cheap?**
Every token reads it and it reads every token, so any two tokens are at most
two hops apart. It adds only about one column and one row of scores, a cost
that grows with n rather than n².

**Why can linear attention run as a recurrence but softmax attention cannot?**
Linear attention's weight φ(q)·φ(k) splits into a query part and a key part,
so the key-and-value parts can be summed ahead of time into a fixed-size
state that any later query can read. The softmax weight e^(q·k) does not
split that way, so each new query must revisit every stored key.

**What does "selective" mean in Mamba, and what problem does it fix?**
The step size Δ, and with it how much of the old state is kept and how much
of the new token is written, is computed from each token. A fixed SSM applies
the same keep and write amounts to every token, so it cannot both absorb one
important token and ignore the filler around it.

**If a selective SSM's parameters change every step, how does it train in
parallel?**
The update "multiply by a, add b" can be merged: two consecutive steps form
one step of the same kind. A parallel scan merges pairs, then pairs of
pairs, and finishes in about log₂ n rounds.

**How does latent KV caching save memory without changing the attention
outputs?**
Keys and values are computed as a small cached latent times fixed
up-projection matrices, so storing the latent is enough to rebuild them
exactly. The up-projection can even be folded into the query and output
side, so the full keys and values are never built.

**Why do hybrid models keep a few attention layers instead of going all-SSM?**
A fixed-size state is a lossy summary, which hurts copying and exact recall
over long contexts. A handful of attention layers restores exact lookup
while most layers keep constant memory, so the cache shrinks roughly by the
fraction of layers that are SSMs.

## The papers behind this lesson

- **Child, Gray, Radford & Sutskever, *Generating Long Sequences with Sparse
  Transformers* (2019)**: https://arxiv.org/abs/1904.10509. Introduced
  strided and fixed sparse attention patterns, cutting attention's cost to
  about n√n.
- **Beltagy, Peters & Cohan, *Longformer: The Long-Document Transformer*
  (2020)**: https://arxiv.org/abs/2004.05150. Combined a sliding window with
  task-chosen global tokens to read long documents in linear time.
- **Zaheer et al., *Big Bird: Transformers for Longer Sequences* (2020)**:
  https://arxiv.org/abs/2007.14062. Mixed window, global and random links, and
  proved such sparse attention keeps the expressive power of full attention.
- **Katharopoulos et al., *Transformers are RNNs: Fast Autoregressive
  Transformers with Linear Attention* (2020)**:
  https://arxiv.org/abs/2006.16236. Replaced softmax with a kernel feature
  map, turning attention into a running sum with O(n) cost.
- **Gu, Goel & Ré, *Efficiently Modeling Long Sequences with Structured State
  Spaces* (S4, 2021)**: https://arxiv.org/abs/2111.00396. Made linear
  state-space layers trainable on very long sequences through the
  convolution view.
- **Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces* (2023)**: https://arxiv.org/abs/2312.00752. Made the state-space
  parameters depend on the input and trained them with a hardware-aware
  parallel scan.
- **Dao & Gu, *Transformers are SSMs* (Mamba-2, 2024)**:
  https://arxiv.org/abs/2405.21060. Showed that selective SSMs and a form of
  linear attention are two views of one computation.
- **Jiang et al., *Mistral 7B* (2023)**: https://arxiv.org/abs/2310.06825.
  Used sliding-window attention with a rolling-buffer cache in a strong open
  model.
- **Xiao et al., *Efficient Streaming Language Models with Attention Sinks*
  (2023)**: https://arxiv.org/abs/2309.17453. Found that models lean on the
  first few tokens, and that keeping them plus a window allows streaming
  without limit.
- **Lieber et al., *Jamba: A Hybrid Transformer-Mamba Language Model*
  (2024)**: https://arxiv.org/abs/2403.19887. Interleaved one attention layer
  per seven Mamba layers to cut the KV cache while keeping recall.
- **Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*
  (2019)**: https://arxiv.org/abs/1911.02150. Introduced multi-query
  attention, one shared key/value head for all query heads.
- **DeepSeek-AI, *DeepSeek-V2* (2024)**: https://arxiv.org/abs/2405.04434.
  Introduced multi-head latent attention, caching a small latent per token
  instead of full keys and values.
- **Liu et al., *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV
  Cache* (2024)**: https://arxiv.org/abs/2402.02750. Quantized keys per
  channel and values per token, bringing the cache down to 2 bits.

## Further reading

- Child et al., *Sparse Transformers* (2019): https://arxiv.org/abs/1904.10509
- Beltagy et al., *Longformer* (2020): https://arxiv.org/abs/2004.05150
- Katharopoulos et al., *Transformers are RNNs* (2020): https://arxiv.org/abs/2006.16236
- Gu, Goel & Ré, *S4* (2021): https://arxiv.org/abs/2111.00396
- Sasha Rush et al., *The Annotated S4* (the S4 paper, line by line in code): https://srush.github.io/annotated-s4/
- Gu & Dao, *Mamba* (2023): https://arxiv.org/abs/2312.00752
- Dao & Gu, *Mamba-2* (2024): https://arxiv.org/abs/2405.21060
- Lieber et al., *Jamba* (2024): https://arxiv.org/abs/2403.19887
- DeepSeek-AI, *DeepSeek-V2* (2024): https://arxiv.org/abs/2405.04434
- Ainslie et al., *GQA* (2023): https://arxiv.org/abs/2305.13245
"""

from __future__ import annotations

from collections import deque

import numpy as np

from primer._show import banner, matrix, say, table, takeaway
from primer.ml.attention import causal_mask, scaled_dot_product_attention, softmax
from primer.ml.inference import dequantize, kv_cache_bytes, kv_cache_bytes_per_token, quantize

# The running example: a Llama-3-8B-shaped model, the same one `primer.ml.inference` sizes.
LAYERS, KV_HEADS, HEAD_DIM = 32, 8, 128

# ---------------------------------------------------------------------------
# 1. The bill: pairs scored and bytes cached
# ---------------------------------------------------------------------------


def context_cost(n: int, layers: int = LAYERS, kv_heads: int = KV_HEADS, head_dim: int = HEAD_DIM, bits: int = 16) -> dict:
    """The two costs of an n-token context: query-key pairs scored, and KV-cache bytes held.

    Pairs are counted per head per layer and without the causal halving, as
    `primer.ml.attention.attention_cost` does: the point is the n² growth.
    """
    return dict(n=n, pairs_per_head_per_layer=n * n, kv_cache_bytes=kv_cache_bytes(n, layers, kv_heads, head_dim, bits))


# ---------------------------------------------------------------------------
# 2. Sliding-window attention
# ---------------------------------------------------------------------------


def sliding_window_mask(n: int, w: int) -> np.ndarray:
    """Boolean (n, n) mask, True where query i may read key j: the last w tokens, itself included.

    For n = 6 and w = 3:

    ```text
    [[1 0 0 0 0 0]
     [1 1 0 0 0 0]
     [1 1 1 0 0 0]
     [0 1 1 1 0 0]
     [0 0 1 1 1 0]
     [0 0 0 1 1 1]]
    ```
    """
    back = np.arange(n)[:, None] - np.arange(n)[None, :]  # how far back key j is from query i
    return (back >= 0) & (back < w)  # never the future, never further back than w − 1


def pairs_computed(mask: np.ndarray) -> int:
    """How many query-key scores a mask asks for: the work attention actually does."""
    return int(np.count_nonzero(mask))


def receptive_field(n: int, w: int, layers: int) -> np.ndarray:
    """Boolean (n, n): True where token j can influence token i's output after `layers` windowed layers.

    One layer moves information along the mask's edges; stacking layers is
    following edges several times, which is repeated matrix multiplication
    of the 0/1 mask (clipped back to 0/1 so the counts stay small).
    """
    step = sliding_window_mask(n, w).astype(np.int64)
    heard = np.eye(n, dtype=np.int64)  # before any layer, each token knows only itself
    for _ in range(layers):
        heard = np.minimum(step @ heard, 1)
    return heard.astype(bool)


def reach(w: int, layers: int) -> int:
    """How many positions back information can travel: each layer adds w − 1."""
    return layers * (w - 1)


def sliding_window_decode(Q: np.ndarray, K: np.ndarray, V: np.ndarray, w: int) -> tuple[np.ndarray, int]:
    """Generate token by token with a rolling KV buffer that holds at most w entries.

    Returns (outputs (n, d_v), the largest the buffer ever got). The outputs
    equal masked attention with `sliding_window_mask`; the memory never grows past w.
    """
    keys: deque = deque(maxlen=w)  # appending the (w+1)-th entry silently drops the oldest
    values: deque = deque(maxlen=w)
    outputs, largest = [], 0
    for q, k, v in zip(Q, K, V):
        keys.append(k)
        values.append(v)
        largest = max(largest, len(keys))
        weights = softmax(np.array(keys) @ q / np.sqrt(len(q)))  # one row of attention, over the buffer only
        outputs.append(weights @ np.array(values))
    return np.array(outputs), largest


# ---------------------------------------------------------------------------
# 3. Sparse patterns: local + global, strided
# ---------------------------------------------------------------------------


def global_local_mask(n: int, w: int, global_tokens: tuple[int, ...] = (0,)) -> np.ndarray:
    """A sliding window plus a few global tokens that everyone reads and that read everyone (causally)."""
    mask = sliding_window_mask(n, w)
    g = list(global_tokens)
    mask[:, g] = True  # every token may read the global tokens...
    mask[g, :] = True  # ...and the global tokens may read every token...
    return mask & causal_mask(n)  # ...but nobody reads the future


def strided_mask(n: int, stride: int) -> np.ndarray:
    """The Sparse Transformer's strided pattern: the last `stride` tokens, plus every stride-th token before them."""
    back = np.arange(n)[:, None] - np.arange(n)[None, :]
    return (back >= 0) & ((back < stride) | (back % stride == 0))


# ---------------------------------------------------------------------------
# 4. Linear attention
# ---------------------------------------------------------------------------


def feature_map(x: np.ndarray) -> np.ndarray:
    """φ(x) = elu(x) + 1: x + 1 for positive x, e^x otherwise. Always positive, so weights never go negative."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, x + 1.0, np.exp(np.minimum(x, 0.0)))  # the minimum keeps exp from overflowing on the unused branch


def linear_attention_weights(Q: np.ndarray, K: np.ndarray) -> np.ndarray:
    """The (n, n) causal weights linear attention implies: φ(q_i)·φ(k_j), each row divided by its total."""
    scores = feature_map(Q) @ feature_map(K).T  # positive "similarities", no exponential
    scores = np.where(causal_mask(len(Q)), scores, 0.0)  # the future contributes nothing
    return scores / scores.sum(axis=1, keepdims=True)


def linear_attention_quadratic(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Linear attention written the slow way: build all n × n weights, then blend the values."""
    return linear_attention_weights(Q, K) @ V


def linear_attention_recurrent(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear attention the fast way: two running sums, updated once per token.

    S (d_k, d_v) accumulates φ(k_j) v_jᵀ and z (d_k,) accumulates φ(k_j).
    Token i reads φ(q_i)ᵀ S / φ(q_i)ᵀ z. Returns (outputs, final S, final z):
    the state is the same size at token 3 and at token 3 million.
    """
    S = np.zeros((K.shape[1], V.shape[1]))
    z = np.zeros(K.shape[1])
    out = np.zeros((len(Q), V.shape[1]))
    for i, (q, k, v) in enumerate(zip(Q, K, V)):
        phi_k = feature_map(k)
        S = S + np.outer(phi_k, v)  # pour this token's value into the pot, flavoured by its key
        z = z + phi_k  # and remember how much flavour went in, to normalise later
        phi_q = feature_map(q)
        out[i] = (phi_q @ S) / (phi_q @ z)
    return out, S, z


# ---------------------------------------------------------------------------
# 5. State-space models
# ---------------------------------------------------------------------------


def _ssm_params(A, B, C) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Accept plain numbers for the one-number worked example, matrices for the real thing.
    return np.atleast_2d(np.asarray(A, float)), np.atleast_1d(np.asarray(B, float)), np.atleast_1d(np.asarray(C, float))


def ssm_recurrent(x, A, B, C) -> np.ndarray:
    """h_t = A h_{t-1} + B x_t, y_t = C · h_t, one step at a time (how an SSM runs during generation).

    x: (T,) one input channel. A: (N, N) or a number; B, C: (N,) or numbers.
    """
    A, B, C = _ssm_params(A, B, C)
    h = np.zeros(len(B))  # the state: N numbers, whatever the sequence length
    y = []
    for x_t in np.asarray(x, float):
        h = A @ h + B * x_t
        y.append(float(C @ h))
    return np.array(y)


def ssm_kernel(A, B, C, length: int) -> np.ndarray:
    """The convolution kernel (C B, C A B, C A² B, ...): how much an input k steps ago still counts."""
    A, B, C = _ssm_params(A, B, C)
    kernel, A_power_B = [], B.copy()
    for _ in range(length):
        kernel.append(float(C @ A_power_B))
        A_power_B = A @ A_power_B
    return np.array(kernel)


def ssm_convolution(x, A, B, C) -> np.ndarray:
    """The same outputs as `ssm_recurrent`, computed as one convolution (how a fixed SSM trains in parallel)."""
    x = np.asarray(x, float)
    kernel = ssm_kernel(A, B, C, len(x))
    # y_t = Σ_k kernel_k · x_{t−k}; np.convolve does exactly that sum, and we keep the first T outputs.
    return np.convolve(x, kernel)[: len(x)]


def discretize(delta: float, a: float = -1.0, b: float = 1.0) -> tuple[float, float]:
    """Turn a step size Δ into this step's keep and write factors (zero-order hold, one channel).

    Ā = e^(Δa), B̄ = (e^(Δa) − 1) / a · b. With a = −1 and b = 1 they add up to 1:
    a big Δ overwrites the state, a tiny Δ leaves it alone.
    """
    A_bar = float(np.exp(delta * a))
    return A_bar, float((A_bar - 1.0) / a * b)


def softplus(x):
    """log(1 + e^x): a smooth ramp that is always positive, so a step size can never go negative."""
    return np.logaddexp(0.0, x)  # log(e^0 + e^x) without overflow


def step_sizes(markers, sharpness: float = 10.0, bias: float = -5.0) -> np.ndarray:
    """Selectivity in miniature: Δ_t = softplus(sharpness · marker_t + bias), read off each token itself.

    Marked tokens get Δ ≈ 5 (write), unmarked ones Δ ≈ 0.007 (ignore).
    Mamba learns this map; here it is set by hand so the numbers are checkable.
    """
    return softplus(sharpness * np.asarray(markers, float) + bias)


def selective_ssm(x, delta, a: float = -1.0, b: float = 1.0, c: float = 1.0) -> np.ndarray:
    """One-channel selective SSM: every step discretizes with its own Δ_t, so every step has its own Ā_t and B̄_t."""
    h, ys = 0.0, []
    for x_t, d_t in zip(np.asarray(x, float), np.asarray(delta, float)):
        A_bar, B_bar = discretize(d_t, a, b)
        h = A_bar * h + B_bar * x_t
        ys.append(c * h)
    return np.array(ys)


# The recall task: one marked token (the 7) among noise, then nine more noise tokens.
RECALL_VALUES = np.array([0.3, -0.5, 7.0, 0.8, -0.2, 0.6, -0.9, 0.4, 0.1, -0.3, 0.5, -0.6])
RECALL_MARKERS = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def selective_recall() -> float:
    """The selective SSM's state after the recall task: how much of the marked 7 survived."""
    return float(selective_ssm(RECALL_VALUES, step_sizes(RECALL_MARKERS))[-1])


def time_invariant_recall(delta: float) -> float:
    """The same task with one fixed Δ for every token (a time-invariant SSM, like S4)."""
    return float(selective_ssm(RECALL_VALUES, np.full(len(RECALL_VALUES), delta))[-1])


def parallel_scan(a, b) -> tuple[np.ndarray, int]:
    """Every state of h_t = a_t h_{t-1} + b_t (h before the start = 0) in ceil(log2 T) rounds.

    Each step is the map h -> a h + b. Two steps in a row are again such a map:
    first (a1, b1) then (a2, b2) is (a1·a2, a2·b1 + b2). In round r every
    position combines with the one 2^r places earlier, all positions at once,
    so on parallel hardware the loop over T steps becomes log2 T rounds.
    Returns (states (T,), rounds).
    """
    a, b = np.array(a, float), np.array(b, float)
    shift, rounds = 1, 0
    while shift < len(a):
        # The step `shift` places earlier; before the start it is the do-nothing map (1, 0).
        a_prev = np.concatenate([np.ones(shift), a[:-shift]])
        b_prev = np.concatenate([np.zeros(shift), b[:-shift]])
        a, b = a_prev * a, a * b_prev + b  # both right-hand sides use the old a
        shift, rounds = shift * 2, rounds + 1
    return b, rounds


def ssm_state_bytes(channels: int, state_size: int, bits: int = 16) -> int:
    """Memory an SSM layer carries between tokens: channels × state numbers, at any context length."""
    return channels * state_size * bits // 8


def hybrid_cache_bytes(
    context: int, layers: int = LAYERS, attention_layers: int = 4, kv_heads: int = KV_HEADS, head_dim: int = HEAD_DIM,
    channels: int = 4096, state_size: int = 16, bits: int = 16,
) -> int:
    """A hybrid stack: attention layers keep a growing KV cache, SSM layers a fixed state."""
    attention = attention_layers * context * kv_cache_bytes_per_token(1, kv_heads, head_dim, bits)
    ssm = (layers - attention_layers) * ssm_state_bytes(channels, state_size, bits)
    return attention + ssm


# ---------------------------------------------------------------------------
# 6. Compressing the KV cache: latents and quantization
# ---------------------------------------------------------------------------


def latent_kv_bytes_per_token(layers: int, latent_dim: int, rope_dim: int = 0, bits: int = 16) -> int:
    """Latent KV cache per token: one latent (plus a small position-carrying key) per layer."""
    return layers * (latent_dim + rope_dim) * bits // 8


# The worked example: a 4-number token, a 2-number latent, 2 heads of width 2.
WORKED_X = np.array([1.0, 0.0, 2.0, 1.0])
WORKED_W_DOWN = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
WORKED_W_UK = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])


def latent_kv_worked_example() -> dict[str, np.ndarray]:
    """x = (1, 0, 2, 1) is cached as the latent (3, 1), which expands to the key (3, 1, 3, 1) when needed."""
    latent = WORKED_X @ WORKED_W_DOWN
    return dict(latent=latent, key=latent @ WORKED_W_UK)


class LatentKVAttention:
    """Multi-head attention that caches one small latent per token instead of every head's keys and values.

    The latent c = x W_down (d_latent numbers) is all that is stored. Keys
    and values are rebuilt from it: K = C W_uk and V = C W_uv, for all heads.
    Weights are random: we are studying the mechanics, not training.
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int, d_latent: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_heads, self.d_head, self.d_latent = n_heads, d_head, d_latent
        s, s_latent = 1 / np.sqrt(d_model), 1 / np.sqrt(d_latent)  # keep activations near unit size
        self.W_q = rng.normal(0, s, (d_model, n_heads * d_head))
        self.W_down = rng.normal(0, s, (d_model, d_latent))  # squeeze: what gets cached
        self.W_uk = rng.normal(0, s_latent, (d_latent, n_heads * d_head))  # expand to every head's keys
        self.W_uv = rng.normal(0, s_latent, (d_latent, n_heads * d_head))  # and to every head's values

    def _heads(self, M: np.ndarray) -> np.ndarray:
        # (n, H·d_head) -> (H, n, d_head): one slab per head.
        return M.reshape(M.shape[0], self.n_heads, self.d_head).transpose(1, 0, 2)

    def _merge(self, M: np.ndarray) -> np.ndarray:
        # (H, n, d_head) -> (n, H·d_head): heads side by side again.
        return M.transpose(1, 0, 2).reshape(M.shape[1], self.n_heads * self.d_head)

    def compress(self, X: np.ndarray) -> np.ndarray:
        """(n, d_model) -> (n, d_latent): the whole KV cache for these tokens."""
        return X @ self.W_down

    def attend(self, X: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Causal attention the direct way: expand the cached latents into keys and values, then attend."""
        Q, K, V = self._heads(X @ self.W_q), self._heads(C @ self.W_uk), self._heads(C @ self.W_uv)
        out, _ = scaled_dot_product_attention(Q, K, V, mask=causal_mask(len(X)))
        return self._merge(out)

    def attend_absorbed(self, X: np.ndarray, C: np.ndarray) -> np.ndarray:
        """The same result without ever building K or V: fold W_uk into the query and W_uv into the output.

        q · (c W_uk) = (q W_ukᵀ) · c, and Σ w_j (c_j W_uv) = (Σ w_j c_j) W_uv.
        """
        Q = self._heads(X @ self.W_q)  # (H, n, d_head)
        W_uk = self.W_uk.reshape(self.d_latent, self.n_heads, self.d_head).transpose(1, 0, 2)  # (H, d_latent, d_head)
        W_uv = self.W_uv.reshape(self.d_latent, self.n_heads, self.d_head).transpose(1, 0, 2)
        Q_latent = Q @ W_uk.transpose(0, 2, 1)  # (H, n, d_latent): each query translated into latent space
        scores = Q_latent @ C.T / np.sqrt(self.d_head)  # scores against the latents directly
        scores = np.where(causal_mask(len(X)), scores, -np.inf)
        mixed = softmax(scores) @ C  # (H, n, d_latent): blend the latents first...
        return self._merge(mixed @ W_uv)  # ...then expand once per query, not once per cached token

    def cached_numbers_per_token(self) -> int:
        return self.d_latent

    def full_numbers_per_token(self) -> int:
        """What ordinary multi-head attention would cache: a key and a value for every head."""
        return 2 * self.n_heads * self.d_head


def quantize_kv(M: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Quantize cached keys or values with one scale per row (per token, per head): codes and scales."""
    return quantize(M, bits, per_channel=True)  # `primer.ml.inference.quantize`: per_channel means one scale per row


def quantized_kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int, bits: int, scale_bits: int = 16) -> float:
    """2 (key and value) × layers × KV heads × (head_dim codes of `bits` each + one scale)."""
    return 2 * layers * kv_heads * (head_dim * bits / 8 + scale_bits / 8)


def quantized_cache_error(bits: int, n: int = 64, d: int = 64, seed: int = 0) -> float:
    """Relative change in causal attention output when K and V are stored at `bits` bits instead of full precision."""
    rng = np.random.default_rng(seed)
    Q, K, V = (rng.standard_normal((n, d)) for _ in range(3))
    exact, _ = scaled_dot_product_attention(Q, K, V, mask=causal_mask(n))
    K_hat, V_hat = dequantize(*quantize_kv(K, bits)), dequantize(*quantize_kv(V, bits))
    approx, _ = scaled_dot_product_attention(Q, K_hat, V_hat, mask=causal_mask(n))
    return float(np.linalg.norm(approx - exact) / np.linalg.norm(exact))


def cache_bytes_by_method(context: int) -> dict[str, float]:
    """Memory to hold one conversation's past, for each design, on the 32-layer running example."""
    return {
        "multi-head (32 KV heads)": kv_cache_bytes(context, LAYERS, 32, HEAD_DIM),
        "grouped-query (8 KV heads)": kv_cache_bytes(context, LAYERS, KV_HEADS, HEAD_DIM),
        "grouped-query, 4-bit": context * quantized_kv_bytes_per_token(LAYERS, KV_HEADS, HEAD_DIM, bits=4),
        "latent (576 per layer)": context * latent_kv_bytes_per_token(LAYERS, 512, 64),
        "sliding window 4k": kv_cache_bytes(min(context, 4096), LAYERS, KV_HEADS, HEAD_DIM),
        "hybrid, 1 attention in 8": hybrid_cache_bytes(context),
        "state-space": LAYERS * ssm_state_bytes(4096, 16),
    }


# ---------------------------------------------------------------------------
# 7. Figures (rendered into the HTML docs by `make figures`)
# ---------------------------------------------------------------------------


def figures() -> dict:
    """Plot this lesson's data. matplotlib is imported here, and only here,
    so the lesson itself needs nothing beyond NumPy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, RED, ORANGE, GREEN, PURPLE, MUTED = "#2563eb", "#dc2626", "#d97706", "#059669", "#7c3aed", "#9ca3af"
    figs = {}

    # --- 1. The two bills at 8k, 128k and 1M tokens ------------------------------
    contexts = (8_192, 131_072, 1_048_576)
    labels = ("8k", "128k", "1M")
    costs = [context_cost(n) for n in contexts]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    a1.bar(labels, [c["pairs_per_head_per_layer"] for c in costs], color=RED)
    a1.set_yscale("log")
    a1.set_ylabel("pairs per head per layer")
    a1.set_title("Compute: pairs scored grow with n²")
    gb = [c["kv_cache_bytes"] / 1e9 for c in costs]
    a2.bar(labels, gb, color=BLUE)
    for i, g in enumerate(gb):
        a2.text(i, g + 2, f"{g:.1f} GB", ha="center")
    a2.axhline(80, color=MUTED, ls="--")
    a2.text(-0.4, 84, "one 80 GB GPU", color="#4b5563")
    a2.set_ylabel("KV cache, GB")
    a2.set_ylim(0, 160)
    a2.set_title("Memory: the KV cache grows with n")
    for a in (a1, a2):
        a.set_xlabel("context length")
    fig.tight_layout()
    figs["context_cost"] = fig

    # --- 2. Sliding window: one layer vs. what three layers can reach -----------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.8))
    a1.imshow(sliding_window_mask(16, 4), cmap="Blues", vmin=0, vmax=1.3)
    a1.set_title("One layer, window w = 4")
    a2.imshow(receptive_field(16, 4, layers=3), cmap="Blues", vmin=0, vmax=1.3)
    a2.set_title("Reach after 3 layers: 3·(4 − 1) = 9 back")
    for a in (a1, a2):
        a.set_xlabel("key position (being read)")
        a.set_ylabel("query position (reading)")
        a.set_xticks([0, 5, 10, 15])
        a.set_yticks([0, 5, 10, 15])
        a.grid(False)
    fig.tight_layout()
    figs["window_reach"] = fig

    # --- 3. Sparse masks and the pairs they score --------------------------------
    masks = [
        ("full causal", causal_mask(16)),
        ("window w = 4", sliding_window_mask(16, 4)),
        ("window + global token 0", global_local_mask(16, 4, (0,))),
        ("strided, stride 4", strided_mask(16, 4)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for a, (name, m) in zip(axes, masks):
        a.imshow(m, cmap="Blues", vmin=0, vmax=1.3)
        a.set_title(f"{name}\n{pairs_computed(m)} pairs")
        a.set_xticks([0, 5, 10, 15])
        a.set_yticks([0, 5, 10, 15])
        a.set_xlabel("key")
        a.grid(False)
    axes[0].set_ylabel("query")
    fig.tight_layout()
    figs["sparse_masks"] = fig

    # --- 4. Softmax vs. linear attention weights on the same inputs -------------
    rng = np.random.default_rng(11)
    # Queries twice the usual size: decisive, as trained queries often are. Softmax sharpens; linear barely moves.
    Q, K = 2 * rng.standard_normal((10, 8)), rng.standard_normal((10, 8))
    _, soft = scaled_dot_product_attention(Q, K, K, mask=causal_mask(10))
    lin = linear_attention_weights(Q, K)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for a, w, title in ((a1, soft, "softmax attention"), (a2, lin, "linear attention, φ = elu + 1")):
        im = a.imshow(w, cmap="Blues", vmin=0, vmax=1)
        a.set_title(title)
        a.set_xlabel("key")
        a.set_ylabel("query")
        a.grid(False)
    fig.colorbar(im, ax=[a1, a2], fraction=0.03, label="attention weight")
    figs["linear_vs_softmax"] = fig

    # --- 5. SSM kernels: A sets how long an input keeps counting ----------------
    steps = np.arange(151)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for A, color in ((0.5, RED), (0.9, ORANGE), (0.99, BLUE)):
        ax.plot(steps, ssm_kernel(A, 1.0, 1.0, len(steps)), color=color, label=f"A = {A}")
    ax.set_xlabel("steps since the input arrived (k)")
    ax.set_ylabel("how much it still counts, C·A^k·B")
    ax.set_title("The kernel: how fast a fixed SSM forgets")
    ax.legend(frameon=False)
    fig.tight_layout()
    figs["ssm_kernel"] = fig

    # --- 6. Selective recall ------------------------------------------------------
    grid = np.geomspace(1e-3, 10, 60)
    best = grid[int(np.argmax([time_invariant_recall(d) for d in grid]))]
    t = np.arange(len(RECALL_VALUES))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(t, RECALL_VALUES, color=MUTED, label="input (marked 7 at position 2)")
    ax.plot(t, selective_ssm(RECALL_VALUES, step_sizes(RECALL_MARKERS)), "o-", color=BLUE, label="selective: Δ set by each token")
    ax.plot(t, selective_ssm(RECALL_VALUES, np.full(len(t), best)), "o-", color=RED, label=f"best fixed Δ = {best:.2f}")
    ax.plot(t, selective_ssm(RECALL_VALUES, np.full(len(t), 5.0)), "o-", color=ORANGE, label="large fixed Δ = 5")
    ax.set_xlabel("position in the sequence")
    ax.set_ylabel("value / state")
    ax.set_title("Remembering one marked token through noise")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    figs["selective_recall"] = fig

    # --- 7. Cache per token, by design ------------------------------------------
    designs = [
        ("multi-head, 32 KV heads", kv_cache_bytes_per_token(LAYERS, 32, HEAD_DIM)),
        ("grouped-query, 8 KV heads", kv_cache_bytes_per_token(LAYERS, KV_HEADS, HEAD_DIM)),
        ("latent, 512 + 64 numbers", latent_kv_bytes_per_token(LAYERS, 512, 64)),
        ("grouped-query 8, 4-bit", quantized_kv_bytes_per_token(LAYERS, KV_HEADS, HEAD_DIM, bits=4)),
        ("multi-query, 1 KV head", kv_cache_bytes_per_token(LAYERS, 1, HEAD_DIM)),
    ]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    kb = [b / 1024 for _, b in designs]
    ax.barh([d for d, _ in designs][::-1], kb[::-1], color=[GREEN, ORANGE, PURPLE, BLUE, RED])
    for i, v in enumerate(kb[::-1]):
        ax.text(v + 5, i, f"{v:.0f} KB", va="center")
    ax.set_xlabel("KV cache per token, all 32 layers (KB)")
    ax.set_xlim(0, 600)
    ax.set_title("Shrinking the cache per token")
    fig.tight_layout()
    figs["kv_per_token"] = fig

    # --- 8. Memory vs. context, every design ------------------------------------
    ns = np.geomspace(1_000, 1_048_576, 80).astype(int)
    rows = [cache_bytes_by_method(int(n)) for n in ns]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name in rows[0]:
        ax.loglog(ns, [r[name] / 1e9 for r in rows], label=name)
    ax.axhline(80, color=MUTED, ls="--")
    ax.text(1_200, 95, "80 GB GPU", color="#4b5563")
    ax.set_xlabel("context length n (tokens)")
    ax.set_ylabel("cache for one conversation (GB)")
    ax.set_title("What each design must keep, as context grows")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    figs["memory_vs_context"] = fig

    return figs


# ---------------------------------------------------------------------------
# 8. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. Why long context is expensive")
    table(
        ["context", "pairs per head per layer", "KV cache (GB)"],
        [(f"{c['n']:,}", f"{c['pairs_per_head_per_layer']:,}", c["kv_cache_bytes"] / 1e9)
         for c in (context_cost(n) for n in (8_192, 131_072, 1_048_576))],
        floatfmt=".1f",
    )
    takeaway("Pairs grow with n², the cache with n. At 1M tokens the cache alone overflows an 80 GB GPU.")

    banner("2. Sliding-window attention")
    matrix("window w = 3 over 8 tokens (1 = score computed)", sliding_window_mask(8, 3).astype(int))
    say(
        f"""
        {pairs_computed(sliding_window_mask(8, 3))} pairs instead of {pairs_computed(causal_mask(8))}.
        After three layers, token 7 can hear token 1:
        {bool(receptive_field(8, 3, 3)[7, 1])}; token 0: {bool(receptive_field(8, 3, 3)[7, 0])}.
        Mistral 7B's 32 layers of w = 4,096 reach {reach(4096, 32):,} tokens back.
        """
    )
    rng = np.random.default_rng(0)
    Q, K, V = (rng.standard_normal((50, 8)) for _ in range(3))
    streamed, largest = sliding_window_decode(Q, K, V, w=4)
    masked, _ = scaled_dot_product_attention(Q, K, V, mask=sliding_window_mask(50, 4))
    say(f"Decoding 50 tokens with a rolling buffer: largest buffer {largest}, same outputs: {np.allclose(streamed, masked)}.")
    takeaway("A window caps both bills; depth still carries information L·(w − 1) positions back.")

    banner("3. Sparse patterns: count the pairs")
    table(
        ["pattern (16 tokens)", "pairs scored"],
        [
            ("full causal", pairs_computed(causal_mask(16))),
            ("window w = 4", pairs_computed(sliding_window_mask(16, 4))),
            ("window + global token 0", pairs_computed(global_local_mask(16, 4, (0,)))),
            ("strided, stride 4", pairs_computed(strided_mask(16, 4))),
        ],
    )
    table(
        ["tokens", "full causal", "window 64", "ratio"],
        [(n, pairs_computed(causal_mask(n)), pairs_computed(sliding_window_mask(n, 64)),
          pairs_computed(causal_mask(n)) / pairs_computed(sliding_window_mask(n, 64))) for n in (256, 1024, 4096)],
        floatfmt=".1f",
    )
    takeaway("With a fixed window the pairs grow with n, not n²; global tokens keep everyone two hops apart.")

    banner("4. Linear attention: the pot")
    Qw = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    Kw = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    Vw = np.array([[2.0], [4.0], [6.0]])
    out, S, z = linear_attention_recurrent(Qw, Kw, Vw)
    soft, _ = scaled_dot_product_attention(Qw, Kw, Vw, mask=causal_mask(3), scale=False)
    say(
        f"""
        Keys (1,0), (0,1), (1,1); values 2, 4, 6; the third query is (1,0).
        The pot S = {S[:, 0].tolist()}, the total z = {z.tolist()}, so the output is
        {out[2, 0]:.3f} (= 62/15). The same inputs through softmax attention give {soft[2, 0]:.3f}.
        """
    )
    Q, K, V = (rng.standard_normal((200, 8)) for _ in range(3))
    fast, S_big, z_big = linear_attention_recurrent(Q, K, V)
    same = np.allclose(fast, linear_attention_quadratic(Q, K, V))
    say(f"On 200 random tokens the running sum equals the n × n form: {same}. The state is {S_big.shape} plus {z_big.shape}, at any length.")
    takeaway("Swap e^(q·k) for φ(q)·φ(k) and attention becomes a running sum: O(n), fixed state, blurrier focus.")

    banner("5. State-space models")
    say(
        f"""
        A = 0.5, inputs (1, 0, 0, 2). Recurrence: {ssm_recurrent([1, 0, 0, 2], 0.5, 1.0, 1.0).tolist()}.
        Kernel: {ssm_kernel(0.5, 1.0, 1.0, 4).tolist()}. Convolution: {ssm_convolution([1, 0, 0, 2], 0.5, 1.0, 1.0).tolist()}.
        """
    )
    table(
        ["step size Δ", "keep Ā", "write B̄"],
        [(d, *discretize(d)) for d in (0.01, 0.1, 1.0, 5.0)],
    )
    grid = np.geomspace(1e-3, 10, 60)
    say(
        f"""
        Recall task (a marked 7, then nine noise tokens): the selective SSM ends at
        {selective_recall():.2f}; the best fixed step ends at
        {max(time_invariant_recall(d) for d in grid):.2f}.
        """
    )
    states, rounds = parallel_scan(np.full(1024, 0.99), rng.standard_normal(1024))
    say(f"A parallel scan over 1,024 steps finishes in {rounds} rounds.")
    table(
        ["context", "all-attention cache (GB)", "hybrid 4 of 32 (GB)", "pure SSM state (MB)"],
        [(f"{n:,}", context_cost(n)["kv_cache_bytes"] / 1e9, hybrid_cache_bytes(n) / 1e9, LAYERS * ssm_state_bytes(4096, 16) / 1e6)
         for n in (8_192, 131_072, 1_048_576)],
        floatfmt=".2f",
    )
    takeaway("A selective SSM decides per token what to keep; its memory never grows, and a few attention layers restore exact recall.")

    banner("6. Compressing the KV cache")
    worked = latent_kv_worked_example()
    say(f"Latent example: x = (1, 0, 2, 1) is cached as {worked['latent'].tolist()} and expands to the key {worked['key'].tolist()}.")
    layer = LatentKVAttention(d_model=64, n_heads=8, d_head=8, d_latent=16)
    X = rng.standard_normal((12, 64))
    C = layer.compress(X)
    say(
        f"""
        A latent layer with 8 heads of width 8 caches {layer.cached_numbers_per_token()} numbers per token
        instead of {layer.full_numbers_per_token()}. Expanded and absorbed attention agree:
        {np.allclose(layer.attend(X, C), layer.attend_absorbed(X, C))}.
        """
    )
    table(
        ["cache precision", "output error", "bytes per token (8B example)"],
        [(f"{b}-bit", quantized_cache_error(b), f"{quantized_kv_bytes_per_token(LAYERS, KV_HEADS, HEAD_DIM, b, scale_bits=0 if b == 16 else 16):,.0f}")
         for b in (16, 8, 4)],
    )
    table(
        ["design", "cache at 1M tokens (GB)"],
        [(name, b / 1e9) for name, b in cache_bytes_by_method(1_048_576).items()],
        floatfmt=".3f",
    )
    takeaway("Share heads, squeeze into a latent, cut bits, or bound the window: each moves the memory line down or flattens it.")


if __name__ == "__main__":
    demo()
