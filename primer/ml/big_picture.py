r"""
# The big picture: what happens when you send a prompt

Run: `python -m primer.ml.big_picture`

New to the notation (vectors, sums, logarithms)? Every symbol is decoded
where it appears, and `primer.notation` teaches them all from zero.

## Level 1: The practitioner's guide

**In one sentence.** A language model is a next-token scorer run in a loop:
the prompt is cut into tokens, each token becomes a vector, the vectors
pass through a stack of transformer blocks, the last position is scored
against the whole vocabulary, one token is drawn, appended, and the loop
runs again until a stop token or a length limit.

**When you need it.** You need this picture whenever you set a parameter
you can't explain (`temperature`, `top_p`, `max_tokens`), read a bill that
counts input and output tokens differently, or debug an answer that was cut
off, repeats itself, or comes out as gibberish. The tell: you are adjusting
a sampling knob by trial and error, or estimating cost in words when the
meter counts tokens. This lesson's toy tokenizer turns "Reset your password"
into 5 tokens, and " password" is one of them while "Reset" is two, so a
word count is only an estimate. You don't need this lesson to write a good
prompt, and you don't need it to pick a model by its benchmark scores; you
need it the first time a model's behaviour has to be predicted rather than
observed.

**Your options.** The loop has a few knobs a caller can turn, from the
cheapest to the most certain:

| Option | What it does | What it guarantees | What it costs | Where it lives |
|---|---|---|---|---|
| Leave sampling at its defaults | Draws each token in proportion to the model's probabilities (temperature 1) | The variety the model was trained to produce | Run-to-run variation | The API's defaults |
| Lower the temperature, down to greedy | Divides the scores before softmax; at 0 it takes the single most likely token | The likely answer more often: at temperature 0.5 this lesson's top token rises from 0.63 to 0.84 | Blander text, and repetition at 0 (Holtzman et al., 2019); still not identical runs | One request parameter |
| Cut the tail (top-k, top-p) | Keeps only the k most likely tokens, or the smallest set whose probabilities reach p | No draws from the long tail of near-zero tokens | A knob some hosted APIs have withdrawn; open-model servers keep it | One request parameter |
| Bound the length (`max_tokens`, stop sequences) | Ends the loop at a token limit or at a string you name | A ceiling on cost and latency per call | Truncated answers if the limit is too low; check the stop reason | One request parameter |
| Reuse the prefix (KV cache, prompt caching) | Keeps the work done on tokens that haven't changed | Each new token costs one position instead of a re-read of everything: 219 positions against 23,900 for 200 tokens after a 20-token prompt | Server memory; caching rules and prices that differ by vendor | The serving stack |
| Constrain the output (schema, grammar) | Forbids, at every step, any token that cannot lead to a valid shape | A parseable answer by construction | A compiled schema and a small check per token | The model server (`primer.ml.structured_output`) |

**How to choose.** Start from who reads the answer and how long it is.

- Extraction, classification, tool calls: low temperature, a schema where
  the server offers one, and a `max_tokens` sized to the answer.
- Writing, brainstorming, dialogue: the default temperature, and a length
  bound that stops runaway output rather than shaping it.
- Long documents in the prompt: the input is read in one parallel pass, so
  it costs money more than time; cache the unchanged prefix when you send
  it repeatedly.
- Agents and multi-turn systems: every turn re-sends the whole history, so
  the prompt grows with the conversation; bound each answer and watch the
  stop reason, because a truncated tool call is a broken one.
- Whatever you pick, measure tokens in and out on your own traffic. Output
  tokens are produced one loop iteration at a time, so the cheapest way to
  make a call faster is to ask for less.

**What it costs.** Two meters run. Input tokens go through the model in
one parallel pass, so a long prompt costs money and memory more than
time. Output tokens are produced one per trip round the loop, so latency is
proportional to length: this lesson's naive loop reads the prompt plus
everything generated so far at every step (14 positions to produce 4 tokens
from a 2-token prompt), and a KV cache turns that into one position per
token. Context is a hard ceiling: the position table has a fixed number of
rows (64 in this toy; `max_position_embeddings` in a Hugging Face config,
where `LlamaConfig` defaults to 2048, and 4k in the Llama 2 paper), and
prompt plus answer must fit inside it. Quality is measured on the same loop: the training loss is
the average of −ln p for the real next token, and perplexity is *e* raised
to it. This lesson's untrained model scores 5.97 against 5.99 for a blind
guess over 400 tokens (perplexity 393), and training over trillions of
tokens (2.0T for Llama 2) is what pushes that number down.

**What breaks.**

- **Gibberish.** Random weights give a nearly flat guess over the
  vocabulary, and the demo's untrained model continues "The cat" with byte
  fragments. In practice the same symptom comes from a tokenizer that
  doesn't match the model, so ids fetch the wrong rows of the table. Check
  the pairing before the weights.
- **Cut-off answers.** The loop stopped at `max_tokens`, not at a stop
  token. Claude's API reports `end_turn` when the model finished on its own
  and another stop reason when your limit or stop sequence ended it; treat
  anything but a natural end as incomplete.
- **Temperature 0 that still varies.** Greedy picks the argmax, but Claude's
  reference says results are not fully deterministic even at 0.0, and the
  arithmetic on real serving hardware is why.
- **Repetition.** Always taking the most likely token produces loops of the
  same phrase; Holtzman et al. (2019) showed it and proposed nucleus
  sampling as the fix.
- **A bill that surprised you.** Cost is counted in tokens, not words, and
  a conversation re-sends its whole history every turn.
- **A context error.** Prompt plus requested output exceeded the position
  table. Shorten the prompt or the `max_tokens`, or retrieve less.

**In the wild.** The pipeline is the decoder-only recipe of GPT-2 (Radford
et al., 2019: byte-level BPE, learned positions, tied embeddings) that
GPT-3 (Brown et al., 2020) scaled until instructions in the prompt were
enough. Hugging Face's `generate()` exposes the loop's knobs in a
`GenerationConfig` (`do_sample`, otherwise greedy; `temperature` 1.0,
`top_k` 50, `top_p` 1.0, `max_new_tokens`, `repetition_penalty`,
`num_beams`), and its `LlamaForCausalLM` returns logits of shape (batch,
sequence, vocabulary) with a `logits_to_keep` option because, as its docs
put it, only the last token's logits are needed for generation. Claude's
Messages API offers `temperature` (0.0 to 1.0, default 1.0; models
released after Claude Opus 4.6 accept only 1.0), `max_tokens`,
`stop_sequences` and a `stop_reason` on every response, and lets you set
`max_tokens` to 0 to warm the prompt cache without generating. Karpathy's
nanoGPT is this lesson at full size, in a few hundred lines. The papers are
linked at the end of the lesson.

**Go deeper.** Level 2 traces "Reset your password" through every box with
its shapes, works temperature by hand on three scores, counts the loop's
positions, and measures the loss of a model that has learned nothing. If you
only needed to set the knobs, you are done.

## Level 2: How it works, from scratch

Follow one prompt, "Reset your password", through a small language model
whose weights are still random. It has learned nothing, so what it writes is
gibberish, but it takes the same steps as a trained model, with the same
shapes. This is the whole trip:

```mermaid
flowchart LR
  A[Prompt text] --> B[Tokenizer<br/>text to IDs]
  B --> C[Embedding lookup<br/>IDs to vectors]
  C --> D[Add position info]
  D --> E[Transformer blocks<br/>repeated N times]
  E --> F[Output layer<br/>score per vocab token]
  F --> G[Softmax + sampling]
  G --> H[Next token]
  H -->|append and repeat| E
```

**Reading it:** read left to right, then follow the arrow back. The prompt
is cut into tokens, and each token's number fetches a list of numbers from a
table, with its position added so that order counts. The lists pass through
the transformer blocks, where the tokens share information. Then the *last*
token's list becomes a score for every token the model knows, the scores
become probabilities, and one token is drawn. It is appended, and the loop
runs again until the model draws a stop token or reaches a length limit.
Training runs the same boxes and adds a loss at the end.

## 1. Text to ids: the coat check

A coat check takes your coat and hands you a numbered ticket. A
**tokenizer** does the same to text: it cuts the prompt into pieces called
**tokens** and hands back one number for each, its **token id**. From then
on the model only ever handles the tickets.

The toy tokenizer here has a kit of 400 pieces, and it turns "Reset your
password" into 5 tickets: `Re` `set` ` y` `our` ` password` →
**[346, 377, 309, 328, 291]**. It learned its kit from text, so a word it saw
often gets a ticket of its own: "password", with the space in front of it, is
one piece. Words it saw less often are split, and "Reset" and "your" take two
tickets each.

```mermaid
flowchart LR
  T["'Reset your password'"] --> TK["tokenizer<br/>(learned kit of 400 pieces)"]
  TK --> I["[346, 377, 309, 328, 291]"]
```

**Reading it:** one box: text goes in, whole numbers come out.

Prompt length, price and context limits are all counted in these tickets, not
in words. How a tokenizer learns its kit is the subject of the tokenization
lesson (`primer.ml.tokenization`).

**In code:** `tok.encode(prompt)` makes the tickets; `build_pipeline` trains the toy tokenizer and builds an untrained `primer.ml.transformer.TinyGPT` sized to its vocabulary.

## 2. Ids to vectors: a lookup, not a computation

A ticket number such as 291 is only a label: token 291 is not more of
anything than token 290. So each id fetches a row of numbers, a **vector**,
from the **token embedding table**. Here the table has 400 rows of 32
numbers, and the 5 ids fetch a 5 × 32 grid.

That grid says what each token is but not where it sits. A second table, the
**position table**, has a row for each seat in the prompt, and row i of it
(i = 0 to 4) is added to row i of the grid.

```mermaid
flowchart LR
  I["ids (5)"] --> E["token table<br/>400 × 32"]
  E --> X["5 × 32: what each token is"]
  P["positions 0..4"] --> PT["position table<br/>64 × 32"]
  PT --> Y["5 × 32: where each token is"]
  X --> ADD(("+"))
  Y --> ADD
  ADD --> OUT["5 × 32 input to the blocks"]
```

**Reading it:** two lookups and one add. Rows are fetched, not multiplied,
which is why this step costs almost nothing. Both tables are learned in
training, and similar tokens end up with similar rows
(`primer.ml.embeddings`).

For the token in seat *i*:

$$
x_i = E_{t_i} + P_i
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $i$ | a token's position in the prompt, from 0 | 4 (the last token) |
| $t_i$ | the token id at position $i$ | $t_4$ = 291 |
| $E$ | the token embedding table | 400 × 32 |
| $E_{t_i}$ | row $t_i$ of that table | row 291: 32 numbers |
| $P_i$ | row $i$ of the position table | row 4: 32 numbers |
| $x_i$ | what the first block receives for position $i$ | 32 numbers |

**In words:** "each token's input vector is its token's row plus its
position's row."

**With the numbers:** $x_4 = E_{291} + P_4$, one 32-number list plus
another. In miniature, with 3-number rows instead of 32: $E_{291}$ =
(0.2, −0.1, 0.5) and $P_4$ = (0.1, 0.3, −0.2) give $x_4$ = (0.3, 0.2, 0.3).

**In Python:**

```python
# row 291 of the token table (3 numbers, not 32)
E_291 = [0.2, -0.1, 0.5]
# row 4 of the position table
P_4 = [0.1, 0.3, -0.2]
# E_(t_i) + P_i, number by number
x_4 = [round(e + p, 2) for e, p in zip(E_291, P_4)]
x_4  # → [0.3, 0.2, 0.3]
```

**In code:** `trace` does both lookups and the add (`model.wte[ids] + model.wpe[:len(ids)]`), keeping every stage's result.

This is the only point where the tokens themselves enter the model. Every
step after it works on these vectors.

## 3. The transformer blocks: rounds of meeting and desk work

So far each row knows only its own token and its seat. In the **transformer
blocks** the rows learn from each other. Each block works like a meeting
followed by desk work. In the meeting, called **attention**, each token takes
in the tokens before it and adds what it hears to its own row. At the desk,
the **feed-forward** layer works on each row alone. The toy model has 2
blocks; large models stack dozens.

The grid leaves each block with the same shape, 5 × 32, and new contents. By
the end, the last row, " password", also carries something of "Re", "set",
" y" and "our", and that row will choose the next token.

```mermaid
flowchart LR
  X["5 × 32"] --> B1["block 1<br/>meeting + desk work"] --> B2["block 2"] --> LN["final norm"] --> H["5 × 32<br/>context-aware vectors"]
```

**Reading it:** the grid passes through two blocks, then a final norm that
rescales each row on its own.

$$
h = \text{LN}\big(\text{Block}_N(\cdots\text{Block}_2(\text{Block}_1(x))\cdots)\big)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $x$ | the 5 × 32 input grid from section 2 | |
| $\text{Block}_k$ | the $k$-th transformer block | $N$ = 2 |
| $\cdots$ | "and so on, for every block in between" | |
| $\text{LN}$ | the final layer norm | |
| $h$ | the context-aware vectors | 5 × 32 |

**In words:** "run the input through every block in turn, then normalize."

**With the numbers:** here N = 2, so h = LN(Block₂(Block₁(x))). In
miniature, with one 3-number vector and two stand-in blocks that each add an
edit: x = (1, 2, 3), Block₁ adds (0, 1, 0) and Block₂ adds (0, 0, 2), giving
(1, 3, 5). The final norm subtracts the mean (3) and divides by the spread
(1.63), so h = (−1.22, 0, 1.22).

**In Python:**

```python
import statistics
x = [1.0, 2.0, 3.0]
# a stand-in edit
def block_1(v): return [v_i + d for v_i, d in zip(v, [0, 1, 0])]
def block_2(v): return [v_i + d for v_i, d in zip(v, [0, 0, 2])]
def LN(v):
    mu, sigma = statistics.fmean(v), statistics.pstdev(v)
    # centre, then rescale
    return [round((v_i - mu) / sigma, 2) for v_i in v]
h = x
# Block_1 first, then Block_2 ... up to Block_N
for block in [block_1, block_2]:
    h = block(h)
h  # → [1.0, 3.0, 5.0]
LN(h)  # → [-1.22, 0.0, 1.22]
```

The transformer lesson (`primer.ml.transformer`) opens a block up.

## 4. Scores, softmax and temperature

Only the last row has taken in the whole prompt, so only it chooses the next
token. It is compared with every row of the token table, which gives a score
for each of the 400 tokens. **Softmax** turns the scores into probabilities
that add up to 1, and one token is drawn at random in proportion to them.
First, every score is divided by the **temperature**. Below 1 the gaps between
scores widen and the top token wins more often; above 1 they narrow; at 0 the
model always takes the top token (greedy decoding). For three candidates
scored 2.0, 1.0 and 0.5:

| temperature | p(A) | p(B) | p(C) |
|---|---|---|---|
| 0 (greedy) | 1.000 | 0.000 | 0.000 |
| 0.5 | 0.844 | 0.114 | 0.042 |
| 1 | 0.629 | 0.231 | 0.140 |
| 2 | 0.481 | 0.292 | 0.227 |

```mermaid
flowchart LR
  H["last row of h<br/>32 numbers"] --> S["× token tableᵀ<br/>400 scores"]
  S --> T["÷ temperature"]
  T --> SM["softmax<br/>400 probabilities"]
  SM --> PICK["draw one token"]
```

**Reading it:** follow one row of 32 numbers to one drawn token: 400 scores,
divided by the temperature, turned into 400 probabilities.

Softmax raises *e* (≈ 2.718) to the power of each score, so a bigger score
gets a disproportionately bigger share, then divides by the total so the
shares add up to 1:

$$
p_i = \frac{e^{z_i / T}}{\sum_{j=1}^{V} e^{z_j / T}}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $z_i$ | the score of candidate token $i$ | $z_A$ = 2.0 |
| $T$ | the temperature | 0.5 |
| $e^{\cdot}$ | *e* ≈ 2.718 raised to that power | $e^{4}$ = 54.6 |
| $V$ | number of candidates (the vocabulary size) | 3 here, 400 in the model |
| $\sum_{j=1}^{V}$ | add up over every candidate $j$ | |
| $p_i$ | probability that token $i$ comes next | 0.844 |

**In words:** "the chance of token *i* is *e* to the power of its score over
the temperature, divided by the same quantity summed over all tokens."

**With the numbers:** T = 0.5 doubles every score to (4, 2, 1): e⁴ = 54.6,
e² = 7.39, e¹ = 2.72, total 64.7, so p(A) = 54.6 / 64.7 = **0.844**.
Temperature 0 is the limit case: all probability on the top score.

**In Python:**

```python
import math
z = [2.0, 1.0, 0.5]
T = 0.5
# e^(z_i / T) for each candidate
exps = [math.exp(z_i / T) for z_i in z]
[round(e, 2) for e in exps]  # → [54.6, 7.39, 2.72]
# Σ_j e^(z_j / T)
total = sum(exps)
round(total, 1)  # → 64.7
# p_i: each share of the total
[round(e / total, 3) for e in exps]  # → [0.844, 0.114, 0.042]
```

The same three scores, across more temperatures:

![At T = 0.25 token A takes 98% of the probability; at T = 2 the three tokens share it 48%, 29% and 23%](figures/primer.ml.big_picture.temperature.svg)

**Reading it:** three groups of bars, one per candidate token; within each
group the bars run from low temperature (left) to high (right). For token A
the bars fall as temperature rises, and for tokens B and C they rise. At
T = 0.25, A takes almost everything; at T = 2 the three are much closer.

The untrained model's own scores show what random weights give:

![The untrained model's 12 favourites are random byte fragments, each only 1.25 to 1.4 times the uniform 1-in-400 share](figures/primer.ml.big_picture.untrained_next_token.svg)

**Reading it:** the bars are the untrained model's 12 likeliest next tokens
after "The cat sat on the"; the dashed line is a uniform guess, 1 in 400
(0.25%). The top one gets about 0.35%, 1.4 times uniform, and the twelfth
about 1.25 times; across all 400 tokens every probability stays between 0.74
and 1.39 times uniform. The favourites are random byte fragments: every step
works, but nothing has been learned yet.

For extraction and tool calls, where you want the most likely answer, use a
low temperature; for creative writing, a higher one. Temperature 0 makes the
output less random, but on real serving hardware it does not guarantee the
same output twice.

## 5. The loop: append and repeat

The drawn token is appended and the whole trip runs again, until the model
draws a stop token or the reply reaches a length limit. Done simply, each trip
re-reads the whole text so far. With a 2-token prompt and 4 new tokens, the
four steps read 2, 3, 4 and 5 tokens: **14** positions to produce 4 tokens.

```mermaid
flowchart LR
  S["ids so far"] --> M["full forward pass<br/>over ALL ids"]
  M --> P["probabilities for the next id"]
  P --> D["draw one id"]
  D --> A["append it"]
  A -->|"not done"| S
  A -->|"stop token or length limit"| OUT["decode ids to text"]
```

**Reading it:** the expensive arrow is "full forward pass over ALL ids":
every step re-reads text that has not changed.

That work is wasted. Each token takes in only the tokens before it (the
**causal mask**), so an earlier token's vectors cannot change when new tokens
arrive. A **KV cache**, named after the keys and values it keeps from
attention, stores that work and reuses it, so each later step runs only the
new token (`primer.ml.inference`).

Without a cache, the work adds up over the steps:

$$
W = \sum_{t=0}^{n-1} (p + t) = n\,p + \frac{n(n-1)}{2}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $p$ | prompt length in tokens | 2 |
| $n$ | tokens to generate | 4 |
| $t$ | the step counter, from 0 to $n-1$ | 0, 1, 2, 3 |
| $p + t$ | tokens re-read at step $t$ | 2, 3, 4, 5 |
| $W$ | total positions run through the model | 14 |

**In words:** "each step re-reads the prompt plus everything generated so
far; add that up over all steps."

**With the numbers:** 4 × 2 + (4 × 3) / 2 = 8 + 6 = **14**.

**In Python:**

```python
p, n = 2, 4
# Σ over t = 0 .. n-1 of (p + t): 2 + 3 + 4 + 5
sum(p + t for t in range(n))  # → 14
# the closed form gives the same count
n * p + n * (n - 1) // 2  # → 14
```

The square in that formula is what makes long replies expensive:

![Without a cache work grows with the square of output length, 23,900 positions at 200 tokens against 219 with a KV cache](figures/primer.ml.big_picture.loop_cost.svg)

**Reading it:** the x-axis is how many tokens have been generated after a
20-token prompt; the y-axis is total work in token positions. The red curve
bends upward: it grows with the square of the output length. The blue line
(with a KV cache) grows by one per token. At 200 tokens the gap is more than a
hundredfold.

**In code:** `generate` runs the loop without a cache, and the `Generation` it returns counts the positions read; `naive_vs_cached_work` counts both ways for the chart.

Even with a cache, each new token costs a trip through the model, while the
prompt is read in one pass, so the length of the reply drives both the wait
and the bill.

## 6. Training: the same forward pass, plus a loss

So far the untrained model has picked only random byte fragments, because
its weights are random. Training sets the weights, using the same pipeline.
Run a real sentence through it: at every position the model gives some
probability to the token that really comes next, and the lower that
probability, the more surprised it was. The **loss** is the average surprise,
and training nudges every weight to make it smaller, one batch of text after
another.

Surprise is −ln p, minus the **logarithm** of the probability. A logarithm
answers "to what power must I raise *e* to get this number?" For a
probability, between 0 and 1, it is negative, so the minus sign makes the
surprise positive: 0 when p = 1, and huge when p is tiny. A guess that gives
the right token 0.5 costs −ln 0.5 = **0.693**. The untrained model averages
**5.97** on a real sentence, close to ln 400 = 5.99, the loss of a blind guess
among 400 tokens.

```mermaid
flowchart LR
  T["training text"] --> F["forward pass<br/>(this whole lesson)"]
  F --> P["probabilities at every position"]
  P --> L["loss: −ln p(actual next token)<br/>averaged over positions"]
  L --> B["backpropagation<br/>gradient for every weight"]
  B --> U["optimizer nudges weights"]
  U -->|next batch| F
```

**Reading it:** the first two boxes are the trip a prompt takes. Training adds
the loss; backpropagation (`primer.ml.neural_net`) works out each weight's
share of it, and the optimizer (`primer.ml.optimizers`) adjusts the
weights. One pass over n tokens gives n − 1 guesses at once, in
parallel, which is a large part of why transformers train fast.

$$
\mathcal{L} = -\frac{1}{n-1}\sum_{i=1}^{n-1} \ln p\big(t_{i+1} \mid t_1, \ldots, t_i\big)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $n$ | tokens in the training text | 12 ("The model reads tokens, not words.") |
| $t_i$ | the token at position $i$ | |
| $p(t_{i+1} \mid t_1,\ldots,t_i)$ | probability the model gave the *actual* next token, having seen everything before it; the bar $\mid$ reads "given" | ≈ 1/400 when untrained |
| $\ln$ | natural logarithm | ln(1/400) = −5.99 |
| $\frac{1}{n-1}\sum$ | the average over all $n - 1$ guesses | |
| $\mathcal{L}$ | the loss training pushes down | 5.97 |

**In words:** "for every position, take the log of the probability the model
gave the true next token, average them, and flip the sign."

**With the numbers:** a uniform guess over 400 tokens gives every true token
p = 1/400, so each term is −ln(1/400) = ln 400 = **5.99**. The untrained model
scores 5.97, so it is still guessing. **Perplexity**, e raised to the loss, is
393: "as unsure as choosing among 393 equally likely tokens".

**In Python:**

```python
import math
# one guess that gave the true token p = 0.5
round(-math.log(0.5), 3)  # → 0.693
n = 12
# a uniform guess gives every true token 1/400
p = [1 / 400] * (n - 1)
# -(1/(n-1)) Σ ln p
L = -sum(math.log(p_i) for p_i in p) / (n - 1)
round(L, 2)  # → 5.99
```

**In code:** `next_token_loss` averages `cross_entropy` over every position of a text.

**Pretraining** is this, run over trillions of tokens. To predict the next
token well, a model has to pick up grammar, facts and patterns of reasoning,
and that is how the machinery that produced byte fragments here becomes a
useful model.

## In 20 seconds
- A language model reads your prompt as tokens, turns each token into a
  vector, and passes the vectors through a stack of transformer blocks.
- Only the last token's vector chooses the next token. It scores every token
  in the vocabulary, and temperature divides those scores before softmax: low
  is predictable, high is varied.
- The reply is written one token per trip round the loop. Without a KV cache,
  every trip re-reads the whole text so far.
- Training is the same forward pass plus a loss: how surprised the model was
  by each real next token.

## Self-test questions

**What happens, step by step, when you send a prompt?**
Tokenizer turns text into ids; each id looks up an embedding; position
information is added; the vectors pass through N transformer blocks; the last
position's vector is scored against the whole vocabulary; softmax and sampling
pick a token; it's appended and the loop repeats until a stop token.

**Why does only the last position matter when generating?**
Its vector has attended to every earlier token and is the one trained to
predict what comes next; earlier rows predict tokens we already have.

**What does temperature do to the scores (2, 1, 0.5) at T = 0.5?**
Doubles them to (4, 2, 1) before softmax, sharpening the distribution: the
top token rises from 0.63 to 0.84.

**How many token positions does the naive loop process for a 2-token prompt and 4 new tokens?**
2 + 3 + 4 + 5 = 14. A KV cache avoids re-reading the unchanged prefix.

**What loss does an untrained model get over a 400-token vocabulary, and why?**
About ln 400 ≈ 5.99 (perplexity about 400), because with random, tiny weights
its predictions are close to uniform.

**How is training different from inference?**
Same forward pass, plus a loss comparing predictions to the real next tokens,
then backpropagation and a weight update. Inference only runs the forward pass.

## The papers behind this lesson

- **Vaswani et al. (2017), *Attention Is All You Need*.** https://arxiv.org/abs/1706.03762.
  The architecture inside the "transformer blocks" box.
  [annotated companion](../../papers/attention-is-all-you-need.html)
- **Radford et al. (2019), *Language Models are Unsupervised Multitask Learners* (GPT-2).**
  https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf.
  The decoder-only, next-token-prediction recipe this pipeline follows,
  including learned positions, byte-level BPE and tied embeddings.
- **Brown et al. (2020), *Language Models are Few-Shot Learners* (GPT-3).**
  https://arxiv.org/abs/2005.14165. Showed that scaling this same loop up
  produces models that follow instructions from examples in the prompt.
  [annotated companion](../../papers/gpt-3.html)
- **Holtzman et al. (2019), *The Curious Case of Neural Text Degeneration*.**
  https://arxiv.org/abs/1904.09751. Introduced nucleus (top-p) sampling and
  explained why pure greedy decoding produces repetitive text.

## Further reading
- Andrej Karpathy, *Let's build GPT* (video): https://www.youtube.com/watch?v=kCc8FmEb1nY
- Karpathy's `nanoGPT`: https://github.com/karpathy/nanoGPT
- Jay Alammar, *The Illustrated GPT-2*: https://jalammar.github.io/illustrated-gpt2/
- 3Blue1Brown, *But what is a GPT?* (video): https://www.youtube.com/watch?v=wjZofJX0v4M
- Hugging Face, *How to generate text* (decoding strategies): https://huggingface.co/blog/how-to-generate
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from primer._show import banner, say, table, takeaway
from primer.ml.tokenization import ByteBPE, trained_tokenizer
from primer.ml.transformer import TinyGPT, layer_norm

# Model size for the walkthrough: small enough to run instantly in NumPy.
D_MODEL, N_LAYERS, N_HEADS, MAX_LEN = 32, 2, 4, 64


def build_pipeline(seed: int = 0) -> tuple[ByteBPE, TinyGPT]:
    """A trained toy tokenizer plus an *untrained* TinyGPT sized to its vocabulary."""
    tok = trained_tokenizer()
    model = TinyGPT(tok.vocab_size, D_MODEL, N_LAYERS, N_HEADS, MAX_LEN, seed=seed)
    return tok, model


# ---------------------------------------------------------------------------
# 1. One forward pass, stage by stage
# ---------------------------------------------------------------------------


def next_token_probs(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Turn one row of scores into next-token probabilities.

    Temperature divides the scores before softmax: below 1 sharpens the
    distribution (more predictable), above 1 flattens it (more varied).
    Temperature 0 is the limit: all probability on the top score (greedy).
    """
    if temperature == 0:
        p = np.zeros_like(logits, dtype=float)
        p[np.argmax(logits)] = 1.0
        return p
    z = logits / temperature
    e = np.exp(z - z.max())  # subtract the max so exp never overflows
    return e / e.sum()


def trace(prompt: str, tok: ByteBPE, model: TinyGPT) -> dict[str, np.ndarray]:
    """Run one prompt through every stage and keep each intermediate result."""
    ids = np.array(tok.encode(prompt))  # (n,)       text -> integers
    emb = model.wte[ids]  # (n, d)     integers -> vectors (a table lookup)
    x = emb + model.wpe[: len(ids)]  # (n, d)     add "where" to "what"
    for block in model.blocks:  # (n, d)     N rounds of meeting + desk work
        x = block(x)
    h = layer_norm(x, model.lnf_g, model.lnf_b)
    logits = h @ model.wte.T  # (n, vocab) a score for every possible next token
    return {
        "ids": ids,
        "embeddings": emb,
        "with_positions": emb + model.wpe[: len(ids)],
        "after_blocks": x,
        "logits": logits,
        "next_token_probs": next_token_probs(logits[-1]),  # only the last row predicts what comes next
    }


# ---------------------------------------------------------------------------
# 2. The generation loop: predict, sample, append, repeat
# ---------------------------------------------------------------------------


@dataclass
class Generation:
    ids: list[int]
    text: str
    positions_processed: int  # total token positions run through the model


def generate(
    prompt: str, n_new: int, tok: ByteBPE, model: TinyGPT, temperature: float = 1.0, seed: int = 0
) -> Generation:
    """The naive autoregressive loop: re-run the whole sequence for every new token.

    Step t processes all (prompt + t) tokens again, so total work grows with
    the square of the output length. The KV cache (primer.ml.inference)
    stores each token's keys and values so every step only processes the one
    new token. A real model also stops at an end-of-text token; this toy just
    stops after n_new.
    """
    rng = np.random.default_rng(seed)
    ids = tok.encode(prompt)
    processed = 0
    for _ in range(n_new):
        logits = model(np.array(ids))
        processed += len(ids)
        p = next_token_probs(logits[-1], temperature)
        ids.append(int(rng.choice(len(p), p=p)))
    return Generation(ids=ids, text=tok.decode(ids), positions_processed=processed)


# ---------------------------------------------------------------------------
# 3. Training: the same forward pass, plus a loss
# ---------------------------------------------------------------------------


def cross_entropy(probs: np.ndarray, target: int) -> float:
    """−ln(probability given to the right answer). Big when confidently wrong."""
    return float(-np.log(probs[target]))


def next_token_loss(text: str, tok: ByteBPE, model: TinyGPT) -> float:
    """Average next-token cross-entropy over `text`, the number training pushes down.

    Every position predicts the token after it, so one forward pass over n
    tokens gives n-1 training examples at once, in parallel. That is a big
    reason transformers train so much faster than RNNs.
    """
    ids = tok.encode(text)
    logits = model(np.array(ids))
    losses = [cross_entropy(next_token_probs(logits[i]), ids[i + 1]) for i in range(len(ids) - 1)]
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# 4. Figures (rendered to docs/figures by `make figures`)
# ---------------------------------------------------------------------------

TEMPERATURE_SCORES = np.array([2.0, 1.0, 0.5])


def naive_vs_cached_work(prompt_len: int, max_new: int) -> list[tuple[int, int, int]]:
    """(new tokens, positions processed without a cache, with a KV cache).

    Without a cache step t re-runs prompt_len + t positions. With a cache the
    first step runs the prompt once (prefill), and every later step runs just
    the one new token.
    """
    rows = []
    for n in range(1, max_new + 1):
        naive = sum(prompt_len + t for t in range(n))
        cached = prompt_len + (n - 1)
        rows.append((n, naive, cached))
    return rows


def figures() -> dict:
    """Plot this lesson's data. matplotlib is imported here, and only here."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, RED, MUTED = "#2563eb", "#dc2626", "#9ca3af"
    figs = {}
    tok, model = build_pipeline()

    p = trace("The cat sat on the", tok, model)["next_token_probs"]
    top = np.argsort(-p)[:12]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar([repr(tok.decode([int(i)])) for i in top], p[top], color=BLUE)
    ax.axhline(1 / len(p), color=RED, ls="--", label=f"uniform guess 1/{len(p)}")
    ax.set_ylabel("probability of being next")
    ax.set_title("Untrained model: 'The cat sat on the' → ?")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False)
    fig.tight_layout()
    figs["untrained_next_token"] = fig

    temps = (0.25, 0.5, 1.0, 2.0)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    width = 0.2
    for j, T in enumerate(temps):
        ax.bar(np.arange(3) + (j - 1.5) * width, next_token_probs(TEMPERATURE_SCORES, T), width, label=f"T = {T}")
    ax.set_xticks(range(3), ["token A (score 2.0)", "token B (1.0)", "token C (0.5)"])
    ax.set_ylabel("probability")
    ax.set_title("Temperature: low sharpens, high flattens")
    ax.legend(frameon=False)
    fig.tight_layout()
    figs["temperature"] = fig

    rows = naive_vs_cached_work(prompt_len=20, max_new=200)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot([r[0] for r in rows], [r[1] for r in rows], color=RED, label="no cache: re-read everything")
    ax.plot([r[0] for r in rows], [r[2] for r in rows], color=BLUE, label="KV cache: one new token per step")
    ax.set_xlabel("tokens generated (after a 20-token prompt)")
    ax.set_ylabel("token positions run through the model")
    ax.set_title("Why the naive loop is too slow")
    ax.legend(frameon=False)
    fig.tight_layout()
    figs["loop_cost"] = fig
    return figs


# ---------------------------------------------------------------------------
# 5. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    tok, model = build_pipeline()
    prompt = "Reset your password"

    banner("1. Text -> ids -> vectors -> blocks -> scores")
    st = trace(prompt, tok, model)
    table(
        ["stage", "shape", "what it holds"],
        [
            ("ids", st["ids"].shape, f"{st['ids'].tolist()} = {tok.tokens(prompt)}"),
            ("embeddings", st["embeddings"].shape, "one row of the token table per id"),
            ("with positions", st["with_positions"].shape, "plus one row of the position table"),
            ("after blocks", st["after_blocks"].shape, f"{N_LAYERS} rounds of attention + feed-forward"),
            ("logits", st["logits"].shape, f"a score for all {tok.vocab_size} tokens, per position"),
            ("next-token probs", st["next_token_probs"].shape, "softmax of the LAST row only"),
        ],
    )
    takeaway("Only the last position's scores decide the next token; the rest matter during training.")

    banner("2. Temperature on the scores (2.0, 1.0, 0.5)")
    table(
        ["temperature", "p(A)", "p(B)", "p(C)"],
        [(T, *next_token_probs(TEMPERATURE_SCORES, T)) for T in (0.0, 0.5, 1.0, 2.0)],
        floatfmt=".3f",
    )

    banner("3. The loop: predict, sample, append, repeat")
    g = generate("The cat", 12, tok, model, seed=0)
    say(
        f"""
        Continuation of 'The cat': {g.text!r}. It's gibberish, and it should be:
        the weights are random, so every next token is close to a uniform guess
        over {tok.vocab_size} tokens. The machinery is identical to a real model's;
        only the numbers inside the matrices are missing. They come from training.
        This loop ran {g.positions_processed} token positions to produce 12 tokens,
        because it re-reads the whole sequence every step (see primer.ml.inference
        for the KV cache).
        """
    )

    banner("4. Training = the same forward pass + a loss")
    loss = next_token_loss("The model reads tokens, not words.", tok, model)
    say(
        f"""
        Average next-token loss of the untrained model: {loss:.2f}. A uniform guess
        over {tok.vocab_size} tokens scores ln {tok.vocab_size} = {np.log(tok.vocab_size):.2f}.
        Perplexity e^loss = {np.exp(loss):.0f}: the model is as unsure as picking among
        ~{np.exp(loss):.0f} equally likely tokens. Training nudges every weight to push
        this number down, across trillions of tokens.
        """
    )
    takeaway("A language model is a next-token scorer run in a loop; training makes its scores good.")


if __name__ == "__main__":
    demo()
