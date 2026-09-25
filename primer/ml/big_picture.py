r"""
# The big picture: what happens when you send a prompt

Run: `python -m primer.ml.big_picture`

New to the notation (vectors, sums, logarithms)? Every symbol is decoded
where it appears, and `primer.notation` teaches them all from zero.

This lesson wires the real pieces from the other lessons into one working
pipeline: the tokenizer from `primer.ml.tokenization`, the transformer from
`primer.ml.transformer`, and a sampling loop. Keep this one picture in your
head; every other lesson zooms into one box of it.

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

**Reading it:** read left to right, then follow the loop back. The prompt is
cut into token ids, each id picks a vector from a table, and position
information is mixed in so order counts. The vectors pass through the
transformer blocks, where tokens exchange information. The output layer turns
the *last* token's final vector into a score for every token in the
vocabulary; softmax makes those scores probabilities and one token is
drawn. That token is appended and the loop runs again, until the model emits
a stop token or hits a length limit. Training uses the same boxes with one
addition, a loss, at the end (section 6).

## 1. Text to ids: the coat check

**Everyday picture.** A coat check. You hand over a coat (a piece of text)
and get back a numbered ticket (a token id). The model only ever handles the
tickets.

**Tiny worked example.** With this module's toy tokenizer, "Reset your
password" becomes 5 tickets: `Re` `set` ` y` `our` ` password` →
**[346, 377, 309, 328, 291]**. The common word " password" got a single
ticket; the rarer pieces got several.

```mermaid
flowchart LR
  T["'Reset your password'"] --> TK["tokenizer<br/>(learned kit of 400 pieces)"]
  TK --> I["[346, 377, 309, 328, 291]"]
```

**Reading it:** one box, text in and integers out. Everything to the right
of this box works only with integers and vectors.

**The code.** `tok.encode(prompt)`; how the kit is learned is the whole of
`primer.ml.tokenization`.

**In code:** `build_pipeline` trains the toy tokenizer (`primer.ml.tokenization.ByteBPE`) and builds an untrained `primer.ml.transformer.TinyGPT` sized to its vocabulary.

**Why it matters.** Prompt length, price and context limits are all counted
in these tickets.

## 2. Ids to vectors: a lookup, not a computation

**Everyday picture.** A dictionary where ticket number 291 opens to page 291,
and each page holds a list of numbers describing that token. A second
dictionary, indexed by seat number, describes *where* the token sits.

**Tiny worked example.** The table has 400 rows (one per token) of 32
numbers. Id 291 fetches row 291. The 5 ids fetch 5 rows: a 5 × 32 grid. Row
i of the position table (i = 0 to 4) is added to row i of that grid.

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

**Reading it:** two lookups, one add. Nothing is multiplied here, rows are
simply fetched, which is why this step is nearly free. The tables themselves
are learned during training, so similar tokens end up with similar rows (see
`primer.ml.embeddings`).

**The math and the code.**

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
another. The code is `model.wte[ids] + model.wpe[:len(ids)]`.

**In code:** `trace` does both lookups and the add, and keeps every stage's result so you can inspect the grid before and after positions are mixed in.

**Why it matters.** This is the only place a token's identity enters the
model; every later step works on these vectors.

## 3. The transformer blocks: rounds of meeting and desk work

**Everyday picture.** The team from `primer.ml.transformer`: each round is a
meeting where every token listens to the others (attention), then desk work
where each token thinks alone (feed-forward). This toy runs 2 rounds; large
models run dozens.

**Tiny worked example.** The 5 × 32 grid goes into block 1 and comes out
5 × 32; the same through block 2. By the end, the vector at the last position
("password") has absorbed information from "Re", "set", "y" and "our".

```mermaid
flowchart LR
  X["5 × 32"] --> B1["block 1<br/>meeting + desk work"] --> B2["block 2"] --> LN["final norm"] --> H["5 × 32<br/>context-aware vectors"]
```

**Reading it:** the shape never changes, only the contents. Each block
edits every token's vector by adding what it learned from the others.

**The math and the code.**

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

**With the numbers:** here N = 2, so h = LN(Block₂(Block₁(x))). The code is
the `for block in model.blocks` loop in `trace`.

**Why it matters.** This is where nearly all the compute and all the
"understanding" happen.

## 4. Scores, softmax and temperature

**Everyday picture.** A scoreboard with one line for every token in the
vocabulary. Softmax turns the scores into shares of a pie. **Temperature**
sets how adventurous the pick is: low temperature almost always takes the
biggest slice; high temperature gives the small slices a real chance.

**Tiny worked example.** Three candidate tokens scored 2.0, 1.0 and 0.5.

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

**Reading it:** only the last position's vector is used to choose the next
token. It is scored against every token's embedding, the scores are divided
by the temperature, softmax turns them into probabilities that sum to 1, and
one token is drawn at random in proportion to them.

**The math and the code.** **Softmax** raises *e* (≈ 2.718) to the power of
each score, so bigger scores get disproportionately bigger shares, then
divides by the total so the shares add up to 1:

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
e² = 7.39, e¹ = 2.72, total 64.7, so p(A) = 54.6 / 64.7 = **0.844**
(`next_token_probs`). Temperature 0 is the limit case: all probability on the
top score.

![Temperature reshapes the same three scores](figures/primer.ml.big_picture.temperature.svg)

**Reading it:** three groups of bars, one per candidate token; within each
group the bars run from low temperature (left) to high (right). For token A
the bars fall as temperature rises, and for tokens B and C they rise. At
T = 0.25, A takes almost everything; at T = 2 the three are much closer.

![What an untrained model predicts next](figures/primer.ml.big_picture.untrained_next_token.svg)

**Reading it:** the bars are the 12 most probable next tokens after "The cat
sat on the", according to our untrained model; the dashed line is a uniform
guess, 1 in 400. Every bar sits almost on that line and the "favourites" are
random byte fragments. That is exactly what random weights should give: the
machinery works, but nothing has been learned yet.

**Why it matters.** Use low temperature for extraction and tool calls,
where you want the most likely answer, and higher temperature for creative
writing. Temperature 0 reduces randomness but doesn't guarantee identical
outputs on real serving hardware.

## 5. The loop: append and repeat

**Everyday picture.** Writing a sentence one word at a time, and re-reading
everything you've written before choosing each new word.

**Tiny worked example.** A 2-token prompt, 4 new tokens. Step 1 reads 2
tokens, step 2 reads 3, step 3 reads 4, step 4 reads 5: **14** token
positions to produce 4 tokens. With a 20-token prompt and 200 new tokens, the
naive loop reads 23,900 positions; a KV cache reads 219.

```mermaid
flowchart LR
  S["ids so far"] --> M["full forward pass<br/>over ALL ids"]
  M --> P["probabilities for the next id"]
  P --> D["draw one id"]
  D --> A["append it"]
  A -->|"not done"| S
  A -->|"stop token or length limit"| OUT["decode ids to text"]
```

**Reading it:** the loop box is the whole of generation. The expensive
arrow is "full forward pass over ALL ids": every step re-reads text that
hasn't changed. Because of the causal mask, earlier tokens' internal vectors
can't change when new tokens arrive, so that repeated work is pure waste.
The KV cache (`primer.ml.inference`) stores it once.

**The math.** Total positions processed without a cache:

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

**With the numbers:** 4 × 2 + (4 × 3) / 2 = 8 + 6 = **14**, the count
`generate` reports as `positions_processed`.

![Positions processed with and without a cache](figures/primer.ml.big_picture.loop_cost.svg)

**Reading it:** the x-axis is how many tokens have been generated after a
20-token prompt; the y-axis is total work in token positions. The red curve
bends upward: it grows with the square of the output length. The blue line
(with a KV cache) grows by exactly one per token. At 200 tokens the gap is
more than a hundredfold.

**In code:** `Generation` holds the generated ids, the decoded text and the positions count; `naive_vs_cached_work` counts the positions processed with and without a KV cache for the figure.

**Why it matters.** Output length drives latency and cost; this is why
every serving system caches keys and values.

## 6. Training: the same forward pass, plus a loss

**Everyday picture.** A guessing game with instant feedback. Cover the next
word, guess it, uncover it, and note how surprised you were. Training nudges
every weight to make the surprise smaller next time.

**Tiny worked example.** If the model gives the right next token probability
0.5, the loss is −ln 0.5 = **0.693**. Our untrained model averages **5.97** on
a real sentence, close to ln 400 = 5.99, the score of a blind uniform guess.

```mermaid
flowchart LR
  T["training text"] --> F["forward pass<br/>(this whole lesson)"]
  F --> P["probabilities at every position"]
  P --> L["loss: −ln p(actual next token)<br/>averaged over positions"]
  L --> B["backpropagation<br/>gradient for every weight"]
  B --> U["optimizer nudges weights"]
  U -->|next batch| F
```

**Reading it:** the first two boxes are the forward pass you just traced.
Training adds the loss (how surprised the model was by the real next
tokens), then backpropagation (`primer.ml.neural_net`) works out how each
weight contributed, and the optimizer (`primer.ml.optimizers`) adjusts them.
One pass over n tokens gives n − 1 guesses at once, in parallel, which is a
big reason transformers train fast.

**The math and the code.** A **logarithm** answers "to what power must I
raise *e* to get this number?" For probabilities between 0 and 1 it is
negative, so we flip the sign; ln 1 = 0 (no surprise) and ln of a tiny
number is very negative (huge surprise).

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
p = 1/400, so each term is −ln(1/400) = ln 400 = **5.99**. Our untrained model
scores 5.97, so it is still essentially guessing. **Perplexity**, e raised to
the loss, is 393: "as unsure as choosing among 393 equally likely tokens"
(`next_token_loss`, `cross_entropy`).

**Why it matters.** Pretraining is exactly this, over trillions of tokens:
predicting the next token well forces the model to absorb grammar, facts and
reasoning patterns. Random weights produce gibberish, and training is what
turns the same machinery into a useful model.

## In 20 seconds
- Tokenize the prompt, look up a vector per token, add position, run the
  transformer blocks, score every vocabulary entry from the last position,
  softmax, sample, append, repeat.
- Temperature divides the scores before softmax: low is predictable, high is
  varied.
- The naive loop re-reads everything each step; the KV cache makes each step
  cost one token. Training is the same forward pass plus a next-token loss.

## Self-test questions

**Walk through what happens when you send a prompt.**
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
