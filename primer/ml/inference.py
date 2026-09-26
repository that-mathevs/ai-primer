r"""
# Inference: what happens, and what it costs, when a model generates

Run: `python -m primer.ml.inference`

New to the notation? `primer.notation` explains every symbol used here from
zero.

## The idea

Training happens once; inference happens every time anyone uses the model,
so this is where the money goes. Generating text has two very different
phases, one memory trick that makes it affordable (the KV cache), a few
knobs that decide *which* token comes out (sampling), and a toolbox of
speedups: quantization, speculative decoding, continuous batching and
prompt caching. Every one of them follows from one fact:

> **Generating one token requires reading every weight of the model from
> memory, and memory is much slower than arithmetic.**

## 1. Two phases: prefill and decode

**Everyday picture.** You are handed a letter and asked to reply. Reading the
letter is fast: your eyes take in whole lines at once. Writing the reply is
slow: one word at a time, and before each word you must walk to a filing
cabinet and flip through an entire reference binder. The walk, not the
thinking, is what takes the time.

The model is the same. **Prefill** reads the whole prompt in one parallel
pass. **Decode** then produces the answer one token at a time, and every
single token requires streaming all the weights from GPU memory.

**Tiny worked example.** An 8-billion-parameter model stored at 16 bits is
16 GB of weights. On a GPU that moves 3.35 TB/s from memory and does about
10¹⁵ 16-bit operations per second:

* Prefill of a 1,000-token prompt: 2 × 8×10⁹ × 1,000 = 1.6×10¹³ operations,
  about **16 ms**.
* Decode: read 16 GB for every token, 16×10⁹ / 3.35×10¹² s = **4.78 ms per
  token**, at most about 209 tokens per second for a single user.

```mermaid
sequenceDiagram
  participant U as User
  participant M as Model
  participant C as KV cache
  U->>M: Prompt of 1,000 tokens
  M->>C: Prefill: store K and V for all 1,000
  M-->>U: First token
  loop Each new token
    M->>C: Read cached K and V, add one entry
    M-->>U: Next token
  end
```

**Reading it:** time runs downward. The first arrow is prefill: one big
parallel pass over the whole prompt, which also fills the KV cache (section
2). It decides the **time to first token**. Everything inside the loop is
decode: one small pass per token, each reading the cache and adding one
entry to it. It decides **tokens per second**. Long prompts slow the first
token; long answers slow the total.

The reason the phases behave so differently is **arithmetic intensity**:
how many operations you do for each byte you fetch from memory.

$$
I = \frac{2\,n}{b} \qquad
t_{\text{decode}} \approx \frac{P\,b}{\text{BW}} \qquad
t_{\text{prefill}} \approx \frac{2\,P\,n}{\text{FLOPS}}
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| $I$ | arithmetic intensity: operations per byte of weights read | FLOPs/byte |
| $n$ | tokens processed in one pass (1 when decoding, the prompt length when prefilling) | ≥ 1 |
| $2$ | one multiply plus one add per weight per token | |
| $b$ | bytes per weight (2 at 16-bit, 0.5 at 4-bit) | |
| $P$ | number of parameters (weights) | e.g. 8×10⁹ |
| BW | memory bandwidth: bytes the GPU can read per second | e.g. 3.35×10¹² |
| FLOPS | arithmetic throughput: operations per second | e.g. 10¹⁵ |
| $\approx$ | "roughly": these are lower bounds that ignore overheads | |

**In words:** decode does two operations per two-byte weight it reads, so
its speed is set by memory bandwidth; prefill reuses each weight for every
prompt token, so its speed is set by arithmetic.

**On the worked example:** decode I = 2 × 1 / 2 = 1 FLOP per byte; prefill
of 1,000 tokens I = 1,000. The GPU breaks even at 10¹⁵ / 3.35×10¹² ≈ **299**
FLOPs per byte, so decode sits far below it (**memory-bound**) and prefill
far above (**compute-bound**).

**In Python:**

```python
# 8 billion weights at 2 bytes each (16-bit)
P, b = 8e9, 2
# bytes read per second, operations per second
BW, FLOPS = 3.35e12, 1e15
def I(n):
    # operations per byte of weights read
    return 2 * n / b
# decode, then a 1,000-token prefill
I(1), I(1000)  # → (1.0, 1000.0)
# the break-even intensity
round(FLOPS / BW)  # → 299
# t_decode, in milliseconds per token
round(P * b / BW * 1000, 2)  # → 4.78
# t_prefill for 1,000 tokens, in milliseconds
round(2 * P * 1000 / FLOPS * 1000, 1)  # → 16.0
```

![Decode for 1 user uses 0.3% of the GPU's compute and a batch of 64 reaches 21%, while a 1,000-token prefill passes the 299 break-even to run at full speed](figures/primer.ml.inference.roofline.svg)

**Reading it:** the x-axis is arithmetic intensity (log scale); the y-axis
is the speed the GPU can actually reach. The sloped part of the roof is the
memory limit (bandwidth × intensity); the flat part is the arithmetic limit.
The corner is the break-even point, ~299. Decode at batch size 1 sits at
intensity 1, deep in the memory-bound region, using well under 1% of the
GPU's arithmetic. Batching many users together moves decode to the right,
because one read of the weights then serves every user in the batch. That is
the economic reason inference servers batch aggressively.

**In code:** `arithmetic_intensity` computes I, `ridge_point` finds the
break-even, and `bottleneck` says which side a pass falls on;
`decode_seconds_per_token` and `prefill_seconds` give the two time bounds.

**Why it matters in practice.** When a system feels slow, ask which phase
dominates. Slow first token: the prompt is long (trim it, or cache it).
Slow streaming: decode is memory-bound (quantize, batch, speculate).

## 2. The KV cache: take notes instead of rereading

**Everyday picture.** Reading a mystery novel, you don't reread the whole book
before each new sentence; you keep notes on every character and clue. For
each new sentence you glance at your notes and add one line.

Attention needs every earlier token's key and value (see `primer.ml.attention`).
Because of causal masking, an earlier token's keys and values never change
when later tokens arrive, so they can be computed once and kept.

**Tiny worked example.** An 8-token prompt, then 16 generated tokens.

* Without a cache, step k re-processes the whole sequence so far: 8 + 9 + … +
  23 = **248** token positions.
* With a cache: prefill the 8 prompt tokens once, then 1 new position for
  each of the next 15 tokens = **23** token positions.

On `TinyDecoder` that is about 11× fewer operations for this short run, and
the gap grows with length: without a cache the total work grows with the
*square* of the length.

```mermaid
flowchart LR
  subgraph NOCACHE["No cache: every step starts over"]
    A1[step 1: tokens 1..8] --> A2[step 2: tokens 1..9] --> A3[step 3: tokens 1..10]
  end
  subgraph CACHE["KV cache: every step adds one"]
    B1[prefill: tokens 1..8<br/>store K,V] --> B2[token 9 only<br/>read K,V 1..8] --> B3[token 10 only<br/>read K,V 1..9]
  end
```

**Reading it:** the top row recomputes a sequence that grows by one each
step. The bottom row computes each token exactly once: prefill stores keys
and values for the prompt, and each decode step computes only the newest
token's query, key and value, reading everything older from the cache. The
outputs are identical (`TinyDecoder` checks this to 10 decimal places); only
the cost differs.

![Without a cache, total work curves upward to about 20 times the cached total after 32 tokens; with the cache it grows in a straight line](figures/primer.ml.inference.kv_cache_work.svg)

**Reading it:** the x-axis is how many tokens have been generated after an
8-token prompt; the y-axis is total operations spent so far, counted inside
`TinyDecoder`. Without a cache the curve bends upward (each step costs more
than the last); with a cache it is a straight line (each step costs about the
same). The gap between them is pure waste the cache removes.

**In code:** `TinyDecoder.forward_full` processes a whole sequence (and
fills a cache during prefill), `TinyDecoder.forward_step` processes one new
token against the cache from `TinyDecoder.new_cache`, and
`TinyDecoder.generate` runs either way, returning a `Generation` that holds
the tokens and the work counted.

**Why it matters in practice.** The cache trades memory for speed, and that
memory is what limits how many users one GPU can serve. Section 3 does the
math.

## 3. Memory math: will it fit?

**Everyday picture.** Packing for a trip. The suitcase is GPU memory. The
model's weights are the big fixed items that always go in. Every active
conversation adds a bag of notes (its KV cache) whose size grows with the
conversation's length. Once the suitcase is full, the next customer waits.

**Tiny worked example.**

* Weights: 70×10⁹ parameters × 2 bytes (16-bit) = **140 GB**: more than one
  80 GB GPU. At 4 bits (half a byte) it is **35 GB** and fits on one.
* KV cache per token for a Llama-3-8B-shaped model (32 layers, 8 KV heads,
  128 dimensions per head, 16-bit): 2 × 32 × 8 × 128 × 2 = **131,072 bytes**,
  about 128 KB.
* A 32,000-token conversation: 32,000 × 131,072 ≈ **4.2 GB** of cache.
* An 80 GB GPU holding 16 GB of weights has 64 GB left: room for **15** such
  conversations at once.

$$
\text{weight bytes} = P \times \frac{\text{bits}}{8} \qquad
\text{KV bytes per token} = 2 \times L \times H_{kv} \times d_h \times b
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| $P$ | number of parameters | e.g. 7×10¹⁰ |
| bits / 8 | bytes per parameter (16 bits = 2 bytes) | |
| $2$ | one key and one value per token | |
| $L$ | number of transformer layers, each with its own cache | e.g. 32 |
| $H_{kv}$ | number of key/value heads (fewer than query heads with grouped-query attention) | e.g. 8 |
| $d_h$ | dimensions per head | e.g. 128 |
| $b$ | bytes per stored number | 2 at 16-bit |

**In words:** weights cost parameters times bytes each; the cache costs, for
every token, one key and one value per layer per KV head.

**On the worked example:** 7×10¹⁰ × 16/8 = 1.4×10¹¹ bytes = 140 GB; and
2 × 32 × 8 × 128 × 2 = 131,072 bytes per token.

**In Python:**

```python
P = 7e10
# weight GB at 16 bits, then at 4 bits
P * 16 / 8 / 1e9, P * 4 / 8 / 1e9  # → (140.0, 35.0)
L, H_kv, d_h, b = 32, 8, 128, 2
# KV bytes per token: a key and a value, per layer, per KV head
2 * L * H_kv * d_h * b  # → 131072
# GB of cache for a 32,000-token conversation
round(32_000 * 131_072 / 1e9, 1)  # → 4.2
```

![At 128k tokens, 32 KV heads need 67 GB, more than the 64 GB free, while 8 KV heads need 17 GB, so three such requests fit](figures/primer.ml.inference.kv_memory.svg)

**Reading it:** the x-axis is context length per request; the y-axis is KV
cache memory for one request. The steep line is a model with 32 KV heads
(classic multi-head attention); the shallow one has 8 (grouped-query
attention, like Llama 3). The dashed line is the 64 GB left on an 80 GB GPU
after 16 GB of weights. With 32 KV heads a single 128k-token request would
not fit; with 8 it fits three times over. Doing this arithmetic out
loud is the fastest way to size a deployment.

**Try it:** pick a model shape, then drag the context length and the number
of requests. The bar is one GPU's memory: the grey part is the weights, the
coloured part the KV cache. Watch how quickly a long context pushes the cache
past the weights, and how much further 8 KV heads go than 32.

<div class="viz" data-viz="kv-cache" aria-label="KV cache memory calculator"></div>

**In code:** `weight_bytes` and `kv_cache_bytes_per_token` are the two
formulas, `kv_cache_bytes` scales the cache to a context and batch, and
`max_concurrent_requests` counts how many conversations fit beside the
weights.

## 4. Sampling: from scores to one token

**Everyday picture.** Choosing where to eat. *Greedy* always picks the
top-rated place. *Sampling* holds a lottery weighted by rating. *Temperature*
is how adventurous you feel: low means you nearly always pick the favourite,
high means long shots get a real chance. *Top-k* says "only consider the top
3". *Top-p* says "consider just enough places to cover 90% of my
enthusiasm": one place if you have a clear favourite, several if you're torn.

**Tiny worked example.** Scores (logits) 2, 1, 0 for three tokens:

| temperature | probabilities |
|---|---|
| 0 (greedy) | 1, 0, 0 |
| 0.5 | 0.867, 0.117, 0.016 |
| 1 | 0.665, 0.245, 0.090 |
| 2 | 0.506, 0.307, 0.186 |

With probabilities 0.5, 0.3, 0.15, 0.05: top-k = 2 keeps 0.5 and 0.3,
renormalised to 0.625 and 0.375. Top-p = 0.9 keeps 0.5, 0.3 and 0.15 (the
first set whose total reaches 0.9) and cuts the 0.05 tail.

```mermaid
flowchart LR
  L[Logits, one per<br/>vocabulary token] --> T[Divide by<br/>temperature T]
  T --> S[Softmax<br/>probabilities]
  S --> K[Top-k: keep the<br/>k likeliest]
  K --> P[Top-p: keep the smallest set<br/>reaching probability p]
  P --> R[Renormalise<br/>and draw one token]
```

**Reading it:** the model only ever produces the scores on the left;
everything after that is a choice *you* make at request time. Temperature
reshapes the whole distribution; top-k and top-p then cut off the unreliable
tail before the draw, so a rare nonsense token can't be picked by bad luck.

$$
p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| $z_i$ | the model's score (logit) for vocabulary token i | any real |
| $T$ | temperature | > 0; T → 0 approaches greedy |
| $e^{x}$ | the exponential function; makes every score positive and stretches gaps | |
| $\sum_j$ | sum over every token j in the vocabulary, so the $p_i$ add to 1 | |
| $p_i$ | probability of drawing token i | 0 … 1 |

**In words:** divide every score by the temperature, exponentiate, and
divide by the total so the results sum to one.

**On the worked example:** T = 0.5 turns (2, 1, 0) into (4, 2, 0);
e⁴ = 54.6, e² = 7.39, e⁰ = 1, total 63.0; probabilities 0.867, 0.117, 0.016.

**In Python:**

```python
import math
z, T = [2.0, 1.0, 0.0], 0.5
# e^(z_i / T)
exps = [math.exp(z_i / T) for z_i in z]
# Σ_j e^(z_j / T)
round(sum(exps), 1)  # → 63.0
# p_i
[round(e / sum(exps), 3) for e in exps]  # → [0.867, 0.117, 0.016]
```

![Five tokens at three temperatures: at T = 0.5 the favourite takes 79% and top-p 0.9 cuts three tokens; at T = 2 it falls to 38% and only one is cut](figures/primer.ml.inference.sampling.svg)

**Reading it:** five candidate tokens with fixed scores, shown at three
temperatures. At T = 0.5 (left) the favourite takes 79%; at T = 2 (right) it
falls to 38% and the rest spread out, down to 7% for the least likely. The
hatched bars are the tokens top-p = 0.9 would cut: three at T = 0.5, two at
T = 1, one at T = 2. A confident model reaches 90% with fewer tokens, so
top-p cuts more of them; an unsure one needs more tokens to reach 90%, so it
cuts fewer. That's why top-p adapts where a fixed top-k can't.

**Temperature 0 is not a determinism guarantee.** Floating-point addition
isn't associative: in 32-bit floats, (10⁸ + 1) − 10⁸ = 0 but (10⁸ − 10⁸) + 1 = 1.
On a GPU the order of additions can depend on the kernel chosen and on what
else is in the batch, so two nearly tied tokens can swap places between
otherwise identical requests.

**In code:** `temperature_probs` applies the formula, `top_k_filter` and
`top_p_filter` cut the tail, and `sample_next` chains them into one draw.
`float32_sum` and `greedy_pick_with_summation_order` show a near tie flipping
with the order of additions.

**Why it matters in practice.** Use low temperature for extraction,
classification and tool calls; higher for brainstorming and creative text.

## 5. Speculative decoding: a junior drafts, a senior checks

**Everyday picture.** A junior writer drafts the next few sentences quickly.
A senior editor reads the whole draft *at once*, keeps every sentence they
would have written themselves, rewrites the first one they wouldn't, and
throws away the rest. The senior's reading is fast; their writing is slow.
So the team moves at the junior's speed but produces the senior's text.

This works because decode is memory-bound: checking 5 draft tokens in one
pass of the big model costs about the same as generating 1.

**Tiny worked example.** The big model's next-token probabilities are
(0.5, 0.3, 0.2); the small model's are (0.3, 0.3, 0.4). A draft token
survives with probability Σ min = 0.3 + 0.3 + 0.2 = **0.8**. Drafting 4
tokens per round yields on average (1 − 0.8⁵) / (1 − 0.8) = **3.36** tokens
per pass of the big model instead of 1.

```mermaid
flowchart TD
  D[Small model drafts γ tokens<br/>one by one, cheap] --> V[Big model scores all γ positions<br/>in ONE parallel pass]
  V --> C{For each draft x in order:<br/>keep with probability min 1, p/q}
  C -->|kept| N[Next draft]
  N --> C
  C -->|rejected| F[Replace x with a draw from<br/>max 0, p − q, renormalised. Stop.]
  C -->|all kept| B[Bonus: draw one more<br/>token from the big model]
```

**Reading it:** the loop in the middle walks the drafts left to right. A
draft the big model likes at least as much as the small one did
(p ≥ q) is always kept; one it likes less is kept only with probability
p/q. The first rejection is replaced by a token drawn from exactly the
probability the small model *under*-proposed, which is what makes the final
output statistically identical to sampling from the big model alone. The
test suite checks this empirically over 20,000 rounds.

$$
P(\text{keep } x) = \min\!\left(1, \frac{p(x)}{q(x)}\right) \qquad
\alpha = \sum_x \min\big(p(x), q(x)\big) \qquad
\mathbb{E}[\text{tokens per pass}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| $p(x)$ | the big (target) model's probability for token x | 0 … 1 |
| $q(x)$ | the small (draft) model's probability for token x | 0 … 1 |
| $\min(a, b)$ | the smaller of a and b | |
| $\alpha$ | alpha, the acceptance rate: how often a draft survives | 0 … 1 |
| $\gamma$ | gamma, how many tokens the small model drafts per round | e.g. 4 |
| $\mathbb{E}[\cdot]$ | expected value: the long-run average | |

**In words:** keep a draft with probability "how much the big model likes it
compared with the small one, capped at 1"; the average number of tokens per
big-model pass is a geometric series in the acceptance rate.

**On the worked example:** α = 0.8, γ = 4: (1 − 0.8⁵)/(1 − 0.8) =
(1 − 0.328)/0.2 = 3.36.

**In Python:**

```python
# big model
p = [0.5, 0.3, 0.2]
# small model
q = [0.3, 0.3, 0.4]
# P(keep x) for each token
[round(min(1.0, p_x / q_x), 2) for p_x, q_x in zip(p, q)]  # → [1.0, 1.0, 0.5]
alpha = sum(min(p_x, q_x) for p_x, q_x in zip(p, q))
round(alpha, 2)  # → 0.8
gamma = 4
# expected tokens per big-model pass
round((1 - alpha ** (gamma + 1)) / (1 - alpha), 2)  # → 3.36
```

**In code:** `acceptance_rate` computes α and `expected_tokens_per_round` the
geometric series; `speculative_round` runs the draft, verify and replace
loop once, and `speculative_generate` repeats it until enough tokens exist.

## 6. Quantization: fewer bits per weight

**Everyday picture.** Writing prices to the nearest dollar instead of the
nearest cent: shorter to store, slightly less exact. Using one ruler per
row of a spreadsheet, instead of one for the whole sheet, keeps a single
huge number in one row from making every other row coarse.

**Tiny worked example.** A row of weights (0.5, −1.27, 0.02).

* **int8**: the largest magnitude, 1.27, maps to 127, so the step size is
  0.01. Codes: **50, −127, 2**. Decoding gives back exactly
  (0.5, −1.27, 0.02).
* **int4**: only 15 levels (−7 … 7), step size 1.27/7 = 0.181. Codes:
  **3, −7, 0**. Decoding gives (0.544, −1.27, 0): the small weight vanished.

$$
s = \frac{\max_j |w_j|}{2^{\,\text{bits}-1} - 1} \qquad
c_j = \operatorname{round}\!\left(\frac{w_j}{s}\right) \qquad
\hat{w}_j = s \, c_j
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| $w_j$ | the j-th original weight in the row | real |
| $\max_j |w_j|$ | the largest absolute value in the row | ≥ 0 |
| $2^{\text{bits}-1} - 1$ | the largest code: 127 for 8 bits, 7 for 4 bits | |
| $s$ | the scale (step size), one per row | > 0 |
| $c_j$ | the stored integer code | −127…127 or −7…7 |
| $\hat{w}_j$ | the weight as reconstructed at inference time ("w-hat") | real |

**In words:** pick a step size so the largest weight lands on the largest
code, store each weight as the nearest whole number of steps, and multiply
back at run time.

**On the worked example:** int4: s = 1.27/7 = 0.181; 0.5/0.181 = 2.76 → 3;
3 × 0.181 = 0.544.

**In Python:**

```python
w = [0.5, -1.27, 0.02]
def quantize(w, bits):
    # the step size
    s = max(abs(w_j) for w_j in w) / (2 ** (bits - 1) - 1)
    # c_j: whole steps
    return s, [round(w_j / s) for w_j in w]
s, c = quantize(w, bits=8)
round(s, 3), c  # → (0.01, [50, -127, 2])
s, c = quantize(w, bits=4)
round(s, 3), c  # → (0.181, [3, -7, 0])
# ŵ_j = s · c_j: the 0.02 is gone
[round(s * c_j, 3) for c_j in c]  # → [0.544, -1.27, 0.0]
```

**In code:** `quantize` returns the integer codes and one scale per row,
`dequantize` multiplies them back, and `quantization_error` measures how far
the round trip lands from the original weights.

**Why it matters in practice.** 8-bit weights are nearly lossless; 4-bit
methods with smarter rounding (GPTQ, AWQ) keep most quality at a quarter
of the memory. Fewer bytes per weight also means faster memory-bound decode.

## 7. Continuous batching: seat the next party as soon as a table frees

**Everyday picture.** A restaurant with two tables. *Static batching* seats
two parties and won't seat anyone new until *both* have left, so a table
sits empty while one slow diner lingers. *Continuous batching* seats the next
party the moment any table frees.

**Tiny worked example.** Four requests needing 4, 1, 1 and 1 decode steps,
two slots. Static: {4, 1} runs 4 steps with one slot idle for 3, then
{1, 1} runs 1: **5 steps, 70%** of slot-steps busy. Continuous: the short
requests slide into the freed slot while the long one runs: **4 steps,
87.5%** busy.

![With 32 requests on 8 slots, static batching leaves idle gaps and needs 433 steps at 60% busy; continuous batching needs 318 steps at 82%](figures/primer.ml.inference.batching.svg)

**Reading it:** each row is a GPU batch slot and each column is one decode
step; colour identifies the request occupying the slot, and white is an idle
slot. On a realistic mix of 32 requests of varied length over 8 slots,
static batching (top) leaves white holes wherever a short request finished
early; continuous batching (bottom) keeps nearly every cell busy and
finishes the same work in fewer steps (about 82% vs. 60% utilisation).

$$
U = \frac{\text{useful slot-steps}}{\text{steps} \times \text{slots}}
$$

**Symbols**

| Symbol | Meaning here | Shape / range |
|---|---|---|
| $U$ | utilisation: the share of slot-steps doing real work | 0 … 1 |
| useful slot-steps | total decode steps all requests need | integer |
| steps × slots | the capacity the GPU offered while serving them | integer |

**In words:** utilisation is the work the requests needed divided by the
work capacity the server spent serving them.

**On the worked example:** 7 useful slot-steps; static 7/(5×2) = 0.70,
continuous 7/(4×2) = 0.875.

**In Python:**

```python
# decode steps the four requests need
useful = 4 + 1 + 1 + 1
slots = 2
# static takes 5 steps, continuous 4
useful / (5 * slots), useful / (4 * slots)  # → (0.7, 0.875)
```

**In code:** `simulate_static_batching` and `simulate_continuous_batching`
play out the two policies step by step, each returning a `ServingRun` that
holds the slot timeline and its utilisation U.

**Why it matters in practice.** Continuous batching (together with paged
KV-cache memory, as in vLLM) is a large part of why modern inference servers
reach high throughput.

## 8. Prompt caching: reuse the prefill of a shared prefix

**Everyday picture.** A kitchen that pre-chops the ingredients every order
uses, instead of chopping them again for each plate.

The KV cache normally lives for one request. Prompt caching keeps it
*across* requests: if many calls start with the same long system prompt,
tool definitions or reference document, the provider stores that prefix's
keys and values and skips its prefill next time. It is only valid up to the
first differing token, because every token's keys depend on everything
before it, so **put stable content first and volatile content last**.

**Tiny worked example.** 100 requests, each a 10,000-token shared prefix
plus a 500-token question, at $3 per million input tokens, with cache
writes at 1.25× and cache reads at 0.1× (typical of providers; check
current pricing):

* Without caching: 100 × 10,500 × $3/10⁶ = **$3.15**.
* With caching: the first call writes the cache ($0.039); each of the other
  99 costs $0.0045. Total **$0.48**, an 85% saving, and each cached call
  also skips 10,000 tokens of prefill, so its first token arrives sooner.

```mermaid
flowchart LR
  subgraph Prompt["One request's prompt, in order"]
    S[System prompt<br/>stable] --> T[Tool definitions<br/>stable] --> D[Reference docs<br/>stable] --> Q[User question<br/>changes every call]
  end
  S & T & D -.->|cached after the first call| C[(Prefix KV cache)]
```

**Reading it:** the prompt is read left to right, and the cache can cover
only an unbroken run from the very start. Stable parts go first so that
every request shares the longest possible prefix; the part that changes on
every call goes last. A timestamp or request ID placed at the top would
break the match on the first token and silently disable caching.

**In code:** `reusable_prefix_tokens` counts how many leading tokens two
prompts share, and `prompt_cache_cost` prices a run of requests with and
without the cache.

## In 20 seconds
- Prefill reads the prompt in parallel (compute-bound, sets time to first
  token); decode writes one token at a time (memory-bound, sets tokens per
  second).
- The KV cache stores every earlier token's keys and values so each step
  computes one token; it trades GPU memory for speed.
- Memory math: weights = parameters × bytes; KV per token = 2 × layers ×
  KV heads × head dim × bytes. 70B at 16-bit is 140 GB; 32k tokens of
  Llama-3-8B cache is about 4 GB.
- Temperature reshapes the distribution; top-k and top-p cut the tail.
  Temperature 0 reduces but does not guarantee determinism.
- Speculative decoding: a small model drafts, the big one verifies in one
  pass; output is identical in distribution.
- Quantization, continuous batching and prompt caching are the other big
  serving levers.

## Self-test questions

**Q: Why is decoding slow even on a huge GPU?**
A: Each token needs every weight read from memory but does only about one
operation per byte read, far below the GPU's break-even (~300 FLOPs/byte).
The arithmetic units mostly wait on memory.

**Q: What is the KV cache, and why does it matter for serving cost?**
A: Stored keys and values of all earlier tokens, so each new token is
computed once instead of recomputing the whole sequence. It grows with
context length and batch size, and that memory caps how many requests a GPU
serves at once.

**Q: How much memory does a 70B model need at 16-bit and at 4-bit?**
A: 140 GB and 35 GB for weights alone, plus KV cache and activations.

**Q: Estimate the KV cache for one 32k-token request on a Llama-3-8B-shaped
model.**
A: 2 × 32 × 8 × 128 × 2 = 131,072 bytes per token; × 32,000 ≈ 4.2 GB.

**Q: Why does grouped-query attention make serving cheaper?**
A: It shares each key/value head among several query heads, shrinking the
KV cache (4× for 32 → 8 KV heads), so more requests fit per GPU.

**Q: Why isn't temperature 0 perfectly deterministic?**
A: Floating-point addition isn't associative, and GPU reduction order can
change with kernels and batch composition, so nearly tied logits can flip.

**Q: How can speculative decoding be faster yet produce the same
distribution?**
A: Verifying several draft tokens costs one memory-bound pass of the big
model, about the same as generating one. The min(1, p/q) accept rule plus
resampling from max(0, p − q) on rejection makes the output exactly the big
model's distribution.

**Q: What does continuous batching fix?**
A: Idle slots: static batches wait for their longest request, while
continuous batching refills any freed slot immediately, raising utilisation.

**Q: How do you structure a prompt to benefit from prompt caching?**
A: Put stable content (system prompt, tool definitions, reference documents)
first and anything that varies (the question, timestamps, IDs) last, because
a cache is valid only up to the first differing token.

## The papers behind this lesson

- Leviathan, Kalman & Matias, *Fast Inference from Transformers via
  Speculative Decoding* (2022): https://arxiv.org/abs/2211.17192. Introduced
  the draft-then-verify scheme with an accept/reject rule that provably
  preserves the target model's output distribution.
  [annotated companion](../../papers/speculative-decoding.html)
- Kwon et al., *Efficient Memory Management for Large Language Model Serving
  with PagedAttention* (2023): https://arxiv.org/abs/2309.06180. Stored the KV
  cache in fixed-size pages like an operating system's virtual memory,
  eliminating fragmentation and enabling vLLM's high-throughput continuous
  batching. [annotated companion](../../papers/paged-attention.html)
- Dao et al., *FlashAttention* (2022): https://arxiv.org/abs/2205.14135.
  Computed exact attention in GPU-memory-sized tiles, cutting the slow
  memory traffic that dominates long-context inference.
  [annotated companion](../../papers/flashattention.html)
- Dettmers et al., *LLM.int8(): 8-bit Matrix Multiplication for Transformers
  at Scale* (2022): https://arxiv.org/abs/2208.07339. Showed that a few
  outlier features break naive 8-bit quantization and handled them
  separately.
- Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative
  Pre-trained Transformers* (2022): https://arxiv.org/abs/2210.17323. Made
  3-4-bit weight quantization practical with error-compensating rounding.
- Holtzman et al., *The Curious Case of Neural Text Degeneration* (2019):
  https://arxiv.org/abs/1904.09751. Introduced nucleus (top-p) sampling.

## Further reading
- vLLM documentation: https://docs.vllm.ai/
- Hugging Face, *Text generation strategies*: https://huggingface.co/docs/transformers/generation_strategies
- Anthropic, *Prompt caching*: https://docs.claude.com/en/docs/build-with-claude/prompt-caching
- Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference*: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- Grattafiori et al., *The Llama 3 Herd of Models* (2024): https://arxiv.org/abs/2407.21783
"""

from __future__ import annotations

import numpy as np

from primer._show import banner, matrix, say, table, takeaway

# Roughly a current datacenter GPU (H100-class): ~1 PFLOP/s of dense 16-bit
# math and ~3.35 TB/s of memory bandwidth. Change these to model other hardware.
PEAK_FLOPS = 1e15
HBM_BANDWIDTH = 3.35e12

# ---------------------------------------------------------------------------
# 1. Memory math: weights and the KV cache
# ---------------------------------------------------------------------------


def weight_bytes(params: float, bits: int = 16) -> float:
    """Memory for the weights alone: parameters × bytes per parameter."""
    return params * bits / 8


def kv_cache_bytes_per_token(layers: int, kv_heads: int, head_dim: int, bits: int = 16) -> int:
    """2 (a key and a value) × layers × KV heads × head dimension × bytes.

    Every layer stores one key vector and one value vector per KV head for
    every token in the context, so the cache grows by this much per token.
    """
    return int(2 * layers * kv_heads * head_dim * bits // 8)


def kv_cache_bytes(context: int, layers: int, kv_heads: int, head_dim: int, bits: int = 16, batch: int = 1) -> int:
    """KV cache for `batch` requests of `context` tokens each."""
    return batch * context * kv_cache_bytes_per_token(layers, kv_heads, head_dim, bits)


def max_concurrent_requests(gpu_gb: float, weights_gb: float, context: int, layers: int, kv_heads: int, head_dim: int, bits: int = 16) -> int:
    """How many full-context requests fit in the memory left after the weights."""
    free = (gpu_gb - weights_gb) * 1e9
    return int(free // kv_cache_bytes(context, layers, kv_heads, head_dim, bits))


# ---------------------------------------------------------------------------
# 2. Prefill vs. decode: compute-bound vs. memory-bound
# ---------------------------------------------------------------------------


def arithmetic_intensity(tokens_in_parallel: int, bits: int = 16) -> float:
    """FLOPs done per byte of weights read from memory.

    Each weight is read once per forward pass and used for one multiply-add
    (2 FLOPs) *per token processed in that pass*. Prefill processes the whole
    prompt in one pass; decode processes one new token per pass.
    """
    return 2 * tokens_in_parallel / (bits / 8)


def ridge_point(flops_per_second: float = PEAK_FLOPS, bytes_per_second: float = HBM_BANDWIDTH) -> float:
    """The intensity at which math and memory take equally long (the roofline's ridge)."""
    return flops_per_second / bytes_per_second


def bottleneck(tokens_in_parallel: int, bits: int = 16) -> str:
    """Below the ridge point the GPU waits on memory; above it, on arithmetic."""
    return "compute-bound" if arithmetic_intensity(tokens_in_parallel, bits) >= ridge_point() else "memory-bound"


def decode_seconds_per_token(params: float, bits: int = 16, bytes_per_second: float = HBM_BANDWIDTH) -> float:
    """Lower bound on time per generated token at batch size 1: read every weight once."""
    return weight_bytes(params, bits) / bytes_per_second


def prefill_seconds(params: float, n_tokens: int, flops_per_second: float = PEAK_FLOPS) -> float:
    """Lower bound on prefill time: 2 FLOPs per parameter per prompt token."""
    return 2 * params * n_tokens / flops_per_second


# ---------------------------------------------------------------------------
# 3. The KV cache, on a tiny but complete decoder
# ---------------------------------------------------------------------------


def _rms_norm(x: np.ndarray) -> np.ndarray:
    # Rescale each token vector to unit root-mean-square (no learned gain, for brevity).
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-6)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class Generation:
    """Result of `TinyDecoder.generate`: the new tokens and how much work they took."""

    def __init__(self, tokens: list[int], positions_processed: int, flops: int):
        self.tokens = tokens
        self.positions_processed = positions_processed
        self.flops = flops


class TinyDecoder:
    """A 2-layer decoder-only transformer with random weights, small enough to read.

    Each layer: RMS-norm -> multi-head causal self-attention -> residual add,
    then RMS-norm -> ReLU feed-forward -> residual add. Token embeddings plus
    learned position embeddings go in; the embedding matrix is reused to turn
    the final vectors into one score (logit) per vocabulary token.

    Two ways to run it:
      * `forward_full(tokens)`: process a whole sequence at once (prefill, or
        naive generation that recomputes everything every step).
      * `forward_step(token, pos, cache)`: process ONE new token, reading the
        keys and values of all earlier tokens from `cache` instead of
        recomputing them. That is the KV cache.
    """

    def __init__(self, vocab: int = 16, d_model: int = 16, n_heads: int = 2, n_layers: int = 2, max_len: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        s = 1 / np.sqrt(d_model)
        self.vocab, self.d, self.h, self.dh = vocab, d_model, n_heads, d_model // n_heads
        self.embed = rng.normal(0, 1, (vocab, d_model))
        self.pos = rng.normal(0, 0.5, (max_len, d_model))
        self.layers = [
            dict(
                Wq=rng.normal(0, s, (d_model, d_model)), Wk=rng.normal(0, s, (d_model, d_model)),
                Wv=rng.normal(0, s, (d_model, d_model)), Wo=rng.normal(0, s, (d_model, d_model)),
                W1=rng.normal(0, s, (d_model, 4 * d_model)), W2=rng.normal(0, s / 2, (4 * d_model, d_model)),
            )
            for _ in range(n_layers)
        ]
        self.flops = 0  # multiply-adds × 2, counted by _mm

    def _mm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Matrix multiply that also counts its cost: 2 FLOPs per multiply-add.
        out = a @ b
        self.flops += 2 * a.shape[-1] * out.size
        return out

    def _heads(self, x: np.ndarray) -> np.ndarray:
        # (seq, d_model) -> (heads, seq, d_head)
        return x.reshape(x.shape[0], self.h, self.dh).transpose(1, 0, 2)

    def _attend(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool) -> np.ndarray:
        # q: (heads, n_q, d_head); k, v: (heads, n_k, d_head). Returns (n_q, d_model).
        scores = self._mm(q, k.transpose(0, 2, 1)) / np.sqrt(self.dh)
        if causal:
            n_q, n_k = scores.shape[-2:]
            # Query i sits at absolute position n_k - n_q + i and may see keys up to there.
            allowed = np.arange(n_k)[None, :] <= (np.arange(n_q)[:, None] + n_k - n_q)
            scores = np.where(allowed, scores, -np.inf)
        out = self._mm(_softmax(scores), v)  # (heads, n_q, d_head)
        return out.transpose(1, 0, 2).reshape(q.shape[1], self.d)

    def _block(self, x: np.ndarray, layer: dict, k_all: np.ndarray, v_all: np.ndarray) -> np.ndarray:
        q = self._heads(self._mm(_rms_norm(x), layer["Wq"]))
        x = x + self._mm(self._attend(q, k_all, v_all, causal=True), layer["Wo"])
        return x + self._mm(np.maximum(0, self._mm(_rms_norm(x), layer["W1"])), layer["W2"])

    def _kv(self, x: np.ndarray, layer: dict) -> tuple[np.ndarray, np.ndarray]:
        n = _rms_norm(x)
        return self._heads(self._mm(n, layer["Wk"])), self._heads(self._mm(n, layer["Wv"]))

    def forward_full(self, tokens: np.ndarray, cache: list[dict] | None = None) -> np.ndarray:
        """Logits (seq, vocab) for every position, computing everything from scratch.

        Pass an empty `cache` to fill it in the same pass: that is *prefill*,
        processing the whole prompt in parallel and keeping every key and value.
        """
        x = self.embed[tokens] + self.pos[: len(tokens)]
        for i, layer in enumerate(self.layers):
            k, v = self._kv(x, layer)
            if cache is not None:
                cache[i]["k"], cache[i]["v"] = k, v
            x = self._block(x, layer, k, v)
        return self._mm(_rms_norm(x), self.embed.T)

    def new_cache(self) -> list[dict]:
        """One empty key store and value store per layer, each (heads, 0, d_head)."""
        empty = np.zeros((self.h, 0, self.dh))
        return [dict(k=empty, v=empty) for _ in self.layers]

    def forward_step(self, token: int, pos: int, cache: list[dict]) -> np.ndarray:
        """Logits (vocab,) for ONE new token, appending its key and value to `cache`."""
        x = self.embed[[token]] + self.pos[[pos]]  # (1, d_model)
        for layer, store in zip(self.layers, cache):
            k_new, v_new = self._kv(x, layer)  # only the new token's K and V are computed
            store["k"] = np.concatenate([store["k"], k_new], axis=1)
            store["v"] = np.concatenate([store["v"], v_new], axis=1)
            x = self._block(x, layer, store["k"], store["v"])
        return self._mm(_rms_norm(x), self.embed.T)[0]

    def generate(self, prompt: list[int], n_new: int, use_cache: bool = True) -> Generation:
        """Greedy generation of `n_new` tokens, with or without the KV cache."""
        self.flops = 0
        seq, out, processed = list(prompt), [], 0
        if use_cache:
            cache = self.new_cache()
            logits = self.forward_full(np.array(seq), cache)[-1]  # prefill: whole prompt, one pass
            processed += len(seq)
            for i in range(n_new):
                tok = int(np.argmax(logits))
                out.append(tok)
                seq.append(tok)
                if i < n_new - 1:  # decode: one new position per step, earlier K/V read from the cache
                    logits = self.forward_step(tok, len(seq) - 1, cache)
                    processed += 1
        else:
            for _ in range(n_new):
                logits = self.forward_full(np.array(seq))[-1]  # recompute the whole sequence each time
                processed += len(seq)
                tok = int(np.argmax(logits))
                out.append(tok)
                seq.append(tok)
        return Generation(out, processed, self.flops)


# ---------------------------------------------------------------------------
# 4. Sampling: turning scores into one next token
# ---------------------------------------------------------------------------


def temperature_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    """softmax(logits / T). T < 1 sharpens, T > 1 flattens, T = 0 means greedy (argmax)."""
    if temperature == 0:
        out = np.zeros_like(logits, dtype=float)
        out[int(np.argmax(logits))] = 1.0
        return out
    return _softmax(logits / temperature)


def top_k_filter(probs: np.ndarray, k: int) -> np.ndarray:
    """Keep the k most likely tokens, zero the rest, renormalise."""
    keep = np.argsort(probs)[::-1][:k]
    out = np.zeros_like(probs)
    out[keep] = probs[keep]
    return out / out.sum()


def top_p_filter(probs: np.ndarray, p: float) -> np.ndarray:
    """Nucleus sampling: keep the smallest set of top tokens whose total reaches p."""
    order = np.argsort(probs)[::-1]
    cumulative = np.cumsum(probs[order])
    # Include the token that crosses p: count how many cumulative sums are still below p, plus one.
    n_keep = int(np.searchsorted(cumulative, p - 1e-12) + 1)
    out = np.zeros_like(probs)
    out[order[:n_keep]] = probs[order[:n_keep]]
    return out / out.sum()


def sample_next(logits: np.ndarray, rng: np.random.Generator, temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None) -> int:
    """The usual pipeline: temperature, then top-k, then top-p, then draw one token."""
    probs = temperature_probs(logits, temperature)
    if top_k is not None:
        probs = top_k_filter(probs, top_k)
    if top_p is not None:
        probs = top_p_filter(probs, top_p)
    return int(rng.choice(len(probs), p=probs))


def float32_sum(values: list[float]) -> float:
    """Add values left to right in 32-bit floats, the way one GPU reduction order might."""
    total = np.float32(0.0)
    for v in values:
        total = np.float32(total + np.float32(v))
    return float(total)


def greedy_pick_with_summation_order(contributions: list[float], other_logit: float) -> str:
    """Token A's logit is a float32 sum of `contributions`; token B's is `other_logit`.

    On a GPU, the order of a reduction depends on kernel choice and on what
    else is in the batch. When two logits are nearly tied, that order alone
    can change which token wins, even at temperature 0.
    """
    return "A" if float32_sum(contributions) > other_logit else "B"


# ---------------------------------------------------------------------------
# 5. Speculative decoding: a small model drafts, the big model verifies
# ---------------------------------------------------------------------------


def _draw(p: np.ndarray, rng: np.random.Generator) -> int:
    # Inverse-CDF sampling: faster than rng.choice for many tiny draws.
    return int(min(np.searchsorted(np.cumsum(p), rng.random(), side="right"), len(p) - 1))


def acceptance_rate(p: np.ndarray, q: np.ndarray) -> float:
    """Probability a token drawn from the draft q survives verification against target p: Σ min(p, q)."""
    return float(np.minimum(p, q).sum())


def expected_tokens_per_round(alpha: float, n_draft: int) -> float:
    """Expected tokens per target pass when each draft is accepted with probability alpha.

    Accepting i drafts in a row has probability alpha^i; every round also
    yields one token from the target (a correction or a bonus). Summing gives
    1 + alpha + ... + alpha^n_draft = (1 - alpha^(n+1)) / (1 - alpha).
    """
    if alpha == 1:
        return n_draft + 1.0
    return (1 - alpha ** (n_draft + 1)) / (1 - alpha)


def speculative_round(last_token: int, target: np.ndarray, draft: np.ndarray, n_draft: int, rng: np.random.Generator) -> list[int]:
    """One round of speculative decoding for a toy bigram language.

    `target[i]` and `draft[i]` are next-token distributions after token i.
    1. The draft model proposes n_draft tokens, one after another (cheap).
    2. The target model scores every proposed position. In a real system
       this is ONE parallel forward pass, costing about the same as
       generating a single token, because decode is memory-bound.
    3. Walk the drafts left to right. Accept draft x with probability
       min(1, p(x) / q(x)). On the first rejection, draw a replacement from
       the leftover distribution max(0, p - q), renormalised, and stop.
    4. If every draft survives, draw one bonus token from the target.
    The accept/reject rule guarantees the output has exactly the target's
    distribution; the draft only changes *speed*.
    """
    drafts, qs, prev = [], [], last_token
    for _ in range(n_draft):
        q = draft[prev]
        x = _draw(q, rng)
        drafts.append(x)
        qs.append(q)
        prev = x
    ps = [target[last_token]] + [target[x] for x in drafts]  # the target's view of every position
    out = []
    for i, x in enumerate(drafts):
        if rng.random() < min(1.0, ps[i][x] / qs[i][x]):
            out.append(x)
            continue
        leftover = np.maximum(ps[i] - qs[i], 0)
        out.append(_draw(leftover / leftover.sum(), rng))
        return out
    out.append(_draw(ps[n_draft], rng))
    return out


def speculative_generate(start: int, target: np.ndarray, draft: np.ndarray, n_tokens: int, n_draft: int, rng: np.random.Generator) -> list[int]:
    """Run speculative rounds until at least `n_tokens` tokens exist; return the first n_tokens."""
    out: list[int] = []
    last = start
    while len(out) < n_tokens:
        new = speculative_round(last, target, draft, n_draft, rng)
        out += new
        last = new[-1]
    return out[:n_tokens]


# ---------------------------------------------------------------------------
# 6. Quantization: fewer bits per weight
# ---------------------------------------------------------------------------


def quantize(W: np.ndarray, bits: int = 8, per_channel: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric integer quantization: w ≈ scale × code, code in [-(2^(b-1)-1), 2^(b-1)-1].

    Per-channel means one scale per output row, so an outlier in one row
    doesn't coarsen every other row. Returns (integer codes, scales).
    """
    qmax = 2 ** (bits - 1) - 1  # 127 for int8, 7 for int4
    absmax = np.abs(W).max(axis=1, keepdims=True) if per_channel else np.abs(W).max(keepdims=True)
    scales = absmax / qmax
    codes = np.clip(np.round(W / scales), -qmax, qmax).astype(np.int32)
    return codes, scales


def dequantize(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return codes * scales


def quantization_error(W: np.ndarray, bits: int = 8, per_channel: bool = True) -> float:
    """Relative error ||W - dequant(quant(W))|| / ||W||."""
    codes, scales = quantize(W, bits, per_channel)
    return float(np.linalg.norm(W - dequantize(codes, scales)) / np.linalg.norm(W))


# ---------------------------------------------------------------------------
# 7. Serving many users: continuous batching and prompt caching
# ---------------------------------------------------------------------------


class ServingRun:
    """Result of a batching simulation."""

    def __init__(self, steps: int, useful_slot_steps: int, slots: int, timeline: list[list[int | None]]):
        self.steps = steps
        self.utilization = useful_slot_steps / (steps * slots)
        self.timeline = timeline  # timeline[step][slot] = request id, or None if idle


def simulate_static_batching(lengths: list[int], slots: int) -> ServingRun:
    """Fill every slot, run until the LONGEST request in the batch finishes, repeat.

    Finished slots sit idle until the whole batch is done.
    """
    timeline: list[list[int | None]] = []
    for start in range(0, len(lengths), slots):
        batch = list(range(start, min(start + slots, len(lengths))))
        for t in range(max(lengths[i] for i in batch)):
            row = [i if t < lengths[i] else None for i in batch]
            timeline.append(row + [None] * (slots - len(row)))
    return ServingRun(len(timeline), sum(lengths), slots, timeline)


def simulate_continuous_batching(lengths: list[int], slots: int) -> ServingRun:
    """After every decode step, hand any freed slot to the next waiting request."""
    queue = list(range(len(lengths)))
    remaining = list(lengths)
    active: list[int | None] = [None] * slots
    timeline: list[list[int | None]] = []
    while queue or any(a is not None for a in active):
        for s in range(slots):  # refill free slots before the step
            if active[s] is None and queue:
                active[s] = queue.pop(0)
        timeline.append(list(active))
        for s, r in enumerate(active):  # one decode step for every active request
            if r is not None:
                remaining[r] -= 1
                if remaining[r] == 0:
                    active[s] = None
    return ServingRun(len(timeline), sum(lengths), slots, timeline)


def reusable_prefix_tokens(previous: list[int], new: list[int]) -> int:
    """How many leading tokens two prompts share: the part whose KV cache can be reused.

    Attention makes every token's keys and values depend on ALL earlier
    tokens, so the cache is valid only up to the first difference.
    """
    n = 0
    for a, b in zip(previous, new):
        if a != b:
            break
        n += 1
    return n


def prompt_cache_cost(prefix: int, suffix: int, requests: int, price_per_million: float,
                      write_multiplier: float = 1.25, read_multiplier: float = 0.1) -> tuple[float, float]:
    """Input cost of `requests` calls sharing a `prefix`, without and with prompt caching.

    The multipliers are typical of providers that price cache writes at a
    small premium and cache reads at a steep discount; check your provider's
    current pricing. Returns (uncached dollars, cached dollars).
    """
    per_token = price_per_million / 1e6
    uncached = requests * (prefix + suffix) * per_token
    first = (prefix * write_multiplier + suffix) * per_token
    later = (prefix * read_multiplier + suffix) * per_token
    return uncached, first + (requests - 1) * later


# ---------------------------------------------------------------------------
# 8. Figures (rendered into docs/figures by `make figures`)
# ---------------------------------------------------------------------------

EXAMPLE_PROMPT = [3, 1, 4, 1, 5, 9, 2, 6]


def _cumulative_flops(use_cache: bool, n_new: int = 32) -> list[int]:
    return [TinyDecoder(seed=0).generate(EXAMPLE_PROMPT, n, use_cache).flops for n in range(1, n_new + 1)]


def _realistic_lengths(n: int = 32, seed: int = 0) -> list[int]:
    return [int(x) for x in np.random.default_rng(seed).integers(5, 120, n)]


def viz_data() -> dict:
    """The numbers the site's interactive KV-cache calculator starts from."""
    # Shapes of openly published models; the widget recomputes everything
    # else with the same formulas as kv_cache_bytes and weight_bytes.
    return {
        "kv-cache": {
            "gpu_bytes": 80e9,
            "presets": [
                {"name": "8B, 8 KV heads (Llama 3 8B shape)", "params": 8e9, "layers": 32, "kv_heads": 8, "head_dim": 128},
                {"name": "7B, 32 KV heads (no grouped-query attention)", "params": 7e9, "layers": 32, "kv_heads": 32, "head_dim": 128},
                {"name": "70B, 8 KV heads (Llama 3 70B shape)", "params": 70e9, "layers": 80, "kv_heads": 8, "head_dim": 128},
            ],
        }
    }


def figures() -> dict:
    """Data figures for this lesson, keyed by the name used in the docstring."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = {}

    # Roofline.
    x = np.logspace(-0.5, 4, 300)
    roof = np.minimum(PEAK_FLOPS, HBM_BANDWIDTH * x)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.loglog(x, roof, lw=2, color="black")
    for n, label in [(1, "decode, 1 user"), (64, "decode, batch of 64"), (1000, "prefill, 1,000 tokens")]:
        i = arithmetic_intensity(n)
        ax.plot(i, min(PEAK_FLOPS, HBM_BANDWIDTH * i), "o", ms=8, label=f"{label} (I = {i:g})")
    ax.axvline(ridge_point(), ls=":", color="grey")
    ax.text(ridge_point() * 1.1, 2e12, f"break-even ≈ {ridge_point():.0f} FLOPs/byte", fontsize=8)
    ax.set(xlabel="arithmetic intensity (FLOPs per byte of weights read)", ylabel="attainable FLOP/s",
           title="Decode is memory-bound; prefill is compute-bound")
    ax.legend(loc="lower right", fontsize=8)
    figs["roofline"] = fig

    # KV cache work.
    steps = np.arange(1, 33)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(steps, _cumulative_flops(False), lw=2, label="no cache: recompute the whole sequence")
    ax.plot(steps, _cumulative_flops(True), lw=2, label="KV cache: compute only the new token")
    ax.set(xlabel="tokens generated after an 8-token prompt", ylabel="total FLOPs so far (TinyDecoder)",
           title="The KV cache turns quadratic work into linear work")
    ax.legend()
    figs["kv_cache_work"] = fig

    # KV memory vs. context.
    ctx = np.linspace(0, 128_000, 200)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for heads, label in [(32, "32 KV heads (multi-head attention)"), (8, "8 KV heads (grouped-query, Llama-3-8B)")]:
        ax.plot(ctx / 1000, [kv_cache_bytes(int(c), 32, heads, 128) / 1e9 for c in ctx], lw=2, label=label)
    ax.axhline(64, ls="--", color="grey", label="free memory: 80 GB GPU − 16 GB weights")
    ax.set(xlabel="context length per request (thousands of tokens)", ylabel="KV cache for one request (GB)",
           title="Long contexts eat GPU memory; fewer KV heads help")
    ax.legend(fontsize=8)
    figs["kv_memory"] = fig

    # Sampling at three temperatures, with top-p cut-offs.
    logits = np.array([3.0, 2.2, 1.5, 0.5, -0.5])
    temps = [0.5, 1.0, 2.0]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4), sharey=True)
    for ax, t in zip(axes, temps):
        probs = temperature_probs(logits, t)
        kept = top_p_filter(probs, 0.9) > 0
        ax.bar(range(5), probs, color=["C0" if k else "white" for k in kept], edgecolor="C0",
               hatch=None, label="kept by top-p 0.9")
        for j in np.where(~kept)[0]:
            ax.bar(j, probs[j], color="white", edgecolor="C3", hatch="//")
        ax.set(title=f"T = {t}", xlabel="candidate token", xticks=range(5))
    axes[0].set_ylabel("probability")
    fig.suptitle("Temperature reshapes the distribution; top-p (hatched = cut) trims the tail")
    figs["sampling"] = fig

    # Batching timelines.
    lengths = _realistic_lengths()
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True)
    for ax, (name, run) in zip(axes, [("static", simulate_static_batching(lengths, 8)),
                                      ("continuous", simulate_continuous_batching(lengths, 8))]):
        grid = np.array([[np.nan if r is None else r for r in row] for row in run.timeline]).T
        ax.imshow(grid, aspect="auto", cmap="tab20", interpolation="nearest")
        ax.set(ylabel="slot", title=f"{name} batching: {run.steps} steps, {run.utilization:.0%} of slot-steps busy")
    axes[1].set_xlabel("decode step")
    figs["batching"] = fig

    for f in figs.values():
        f.tight_layout()
    return figs


# ---------------------------------------------------------------------------
# 9. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. Prefill vs. decode on an 8B model (16-bit, H100-class GPU)")
    table(
        ["phase", "tokens per pass", "FLOPs per byte", "bound by", "time"],
        [
            ("prefill 1,000 tokens", 1000, arithmetic_intensity(1000), bottleneck(1000), f"{prefill_seconds(8e9, 1000) * 1e3:.1f} ms total"),
            ("decode", 1, arithmetic_intensity(1), bottleneck(1), f"{decode_seconds_per_token(8e9) * 1e3:.2f} ms per token"),
        ],
        floatfmt=".0f",
    )
    say(f"The GPU breaks even at {ridge_point():.1f} FLOPs per byte. Decode does 1, so the arithmetic units mostly wait on memory.")
    takeaway("Prefill sets time to first token; decode sets tokens per second, and decode is memory-bound.")

    banner("2. The KV cache: identical output, far less work")
    model = TinyDecoder(seed=0)
    a = model.generate(EXAMPLE_PROMPT, 16, use_cache=False)
    b = model.generate(EXAMPLE_PROMPT, 16, use_cache=True)
    table(
        ["", "tokens generated", "positions processed", "FLOPs"],
        [("no cache", a.tokens[:6], a.positions_processed, f"{a.flops:,}"), ("KV cache", b.tokens[:6], b.positions_processed, f"{b.flops:,}")],
    )
    say(f"Same tokens (the weights are random, so the tokens themselves mean nothing), {a.flops / b.flops:.1f}× fewer operations. 248 = 8 + 9 + ... + 23; 23 = 8 prompt + 15 new.")
    takeaway("The KV cache trades GPU memory for speed: each new token is computed exactly once.")

    banner("3. Memory math")
    table(
        ["quantity", "calculation", "result"],
        [
            ("70B weights, 16-bit", "70e9 × 2 bytes", f"{weight_bytes(70e9, 16) / 1e9:.0f} GB"),
            ("70B weights, 4-bit", "70e9 × 0.5 bytes", f"{weight_bytes(70e9, 4) / 1e9:.0f} GB"),
            ("KV per token (Llama-3-8B shape)", "2 × 32 × 8 × 128 × 2", f"{kv_cache_bytes_per_token(32, 8, 128):,} bytes"),
            ("KV for 32,000 tokens", "32,000 × 131,072", f"{kv_cache_bytes(32_000, 32, 8, 128) / 1e9:.2f} GB"),
            ("32k requests per 80 GB GPU", "(80 − 16) GB / 4.19 GB", max_concurrent_requests(80, 16, 32_000, 32, 8, 128)),
        ],
    )

    banner("4. Sampling")
    z = np.array([2.0, 1.0, 0.0])
    table(["temperature", "p(token 1)", "p(token 2)", "p(token 3)"], [(t, *temperature_probs(z, t)) for t in (0.0, 0.5, 1.0, 2.0)], floatfmt=".3f")
    probs = np.array([0.5, 0.3, 0.15, 0.05])
    say(f"From (0.5, 0.3, 0.15, 0.05): top-k=2 gives {np.round(top_k_filter(probs, 2), 3)}; top-p=0.9 gives {np.round(top_p_filter(probs, 0.9), 3)}.")
    say(
        f"""
        Float32: (1e8 + 1) − 1e8 = {float32_sum([1e8, 1.0, -1e8])}, but (1e8 − 1e8) + 1 = {float32_sum([1e8, -1e8, 1.0])}.
        Reduction order alone can flip a near-tie, so temperature 0 is not a determinism guarantee.
        """
    )

    banner("5. Speculative decoding")
    target = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.3, 0.3, 0.4]])
    draft = np.array([[0.3, 0.3, 0.4], [0.2, 0.5, 0.3], [0.6, 0.2, 0.2]])
    rng = np.random.default_rng(0)
    firsts = np.bincount([speculative_round(0, target, draft, 3, rng)[0] for _ in range(20_000)], minlength=3) / 20_000
    table(["", "token 0", "token 1", "token 2"], [("big model alone", *target[0]), ("small model alone", *draft[0]), ("speculative output", *firsts)], floatfmt=".3f")
    say(f"Acceptance rate Σ min(p, q) = {acceptance_rate(target[0], draft[0]):.1f}; with 4 drafts that is {expected_tokens_per_round(0.8, 4):.2f} tokens per big-model pass.")
    takeaway("The draft model changes speed, never the output distribution.")

    banner("6. Quantization")
    row = np.array([[0.5, -1.27, 0.02]])
    for bits in (8, 4):
        codes, scales = quantize(row, bits)
        print(f"int{bits}: codes {codes[0].tolist()}, scale {scales[0, 0]:.4f}, back to {np.round(dequantize(codes, scales)[0], 3).tolist()}")
    print()

    banner("7. Continuous batching")
    for name, run in [("static", simulate_static_batching([4, 1, 1, 1], 2)), ("continuous", simulate_continuous_batching([4, 1, 1, 1], 2))]:
        print(f"{name:10s} {run.steps} steps, utilisation {run.utilization:.1%}, timeline {run.timeline}")
    print()

    banner("8. Prompt caching")
    uncached, cached = prompt_cache_cost(10_000, 500, 100, 3.0)
    say(f"100 calls with a 10,000-token shared prefix: ${uncached:.2f} uncached vs. ${cached:.2f} cached ({1 - cached / uncached:.0%} less).")
    takeaway("Stable content first, volatile content last: a cache is valid only up to the first differing token.")


if __name__ == "__main__":
    demo()
