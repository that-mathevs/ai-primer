r"""
# Tokenization: how text becomes the integers a model actually reads

Run: `python -m primer.ml.tokenization`

New to the notation (sums, subscripts)? Every symbol is decoded where it
appears, and `primer.notation` teaches them all from zero.

## 1. Tokens: building words from a fixed kit of bricks

**Everyday picture.** A box of Lego with a fixed set of bricks. Common
shapes have a ready-made brick; anything unusual gets assembled from smaller
bricks. You can build *anything*, but some builds take many more bricks than
others. A **tokenizer** is that kit for text: a fixed vocabulary of pieces
(typically 32,000 to 200,000 of them), each with a number, its **token id**.

**Tiny worked example.** With a kit that contains `the`, `un`, `believ` and
`able`, "the" takes 1 brick and "unbelievable" takes 3: `un` + `believ` +
`able`. The model never sees the letters; it sees a short list of ids such
as `[262, 403, 11009, 540]`.

| Approach | Kit size | "unbelievable" | Problem |
|---|---|---|---|
| Whole words | huge (every form of every word, every name) | 1 piece, if it's in the kit | words not in the kit become `<unk>`; typos break |
| Single bytes | 256 | 12 pieces | sequences 4-5× longer, and attention cost grows with length squared |
| **Subwords** | 32k to 200k | ~3 pieces | the middle ground every modern LLM uses |

```mermaid
flowchart LR
  T["'the unbelievable'"] --> TOK["tokenizer<br/>(fixed kit of pieces)"]
  TOK --> P["pieces: 'the' | ' un' | 'believ' | 'able'"]
  P --> IDS["ids: 262, 403, 11009, 540"]
  IDS --> EMB["embedding table<br/>one vector per id"]
  EMB --> M["the model"]
```

**Reading it:** text enters on the left and never reaches the model as
text. The tokenizer cuts it into pieces from its fixed kit and swaps each
piece for its number. Those numbers pick rows from the embedding table (see
`primer.ml.big_picture`), and only those vectors reach the model. Everything
the model knows about spelling it had to learn through this narrow window.
(The ids shown are illustrative.)

**In code:** `trained_tokenizer` builds this lesson's toy kit, and
`ByteBPE.encode` turns any text into its list of ids.

**Why it matters.** Everything downstream is counted in tokens: price,
speed, and how much fits in the context window.

## 2. Byte Pair Encoding (BPE): learning the kit from data

**Everyday picture.** A court stenographer inventing shorthand. Whatever
pair of letters they write most often gets its own squiggle. Then the most
common pair *including* that squiggle gets one too, and so on, until they
have as many squiggles as they can remember.

**Tiny worked example.** Training text "low lower lowest". Start from single
letters and repeatedly merge the most frequent neighbouring pair
(`train_char_bpe` reproduces this exactly):

| Step  | Most frequent pair | New token | Text becomes                    |
|-------|--------------------|-----------|---------------------------------|
| Start | none               | none      | l o w · l o w e r · l o w e s t |
| 1     | l + o (3 times)    | lo        | lo w · lo w e r · lo w e s t    |
| 2     | lo + w (3 times)   | low       | low · low e r · low e s t       |
| 3     | low + e (2 times)  | lowe      | low · lowe r · lowe s t         |

At step 1, `l+o` and `o+w` are tied at 3; ties go to the pair seen first.

```mermaid
flowchart LR
  T[Training text] --> S[Split into characters]
  S --> C[Count adjacent pairs]
  C --> M[Merge the most frequent pair<br/>record the merge rule]
  M -->|vocab not full yet| C
  M -->|vocab full| V[Vocabulary + ordered merge list]
```

**Reading it:** training is a loop around two boxes. "Count adjacent pairs"
looks at every neighbouring pair of symbols in the corpus; "Merge" glues the
winner into one new symbol everywhere and appends that rule to an ordered
list. The loop exits when the vocabulary reaches its target size. The output
is not just a vocabulary but the *ordered* merge list, and encoding depends
on that order.

**The math and the code.** Each round picks the pair with the largest count:

$$
\text{count}(a, b) = \sum_{w} f_w \cdot n_w(a, b)
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $a, b$ | two symbols that sit side by side | l and o |
| $w$ | one distinct word of the training text | "lower" |
| $\sum_{w}$ | "add up the following over every distinct word" | low, lower, lowest |
| $f_w$ | how many times word $w$ occurs in the text | 1 each |
| $n_w(a, b)$ | how many times $a$ is immediately followed by $b$ inside $w$ | 1 in each word |
| $\text{count}(a, b)$ | the pair's total, the number BPE ranks by | 3 |

**In words:** "for each distinct word, count how often the pair appears
inside it, multiply by how often the word occurs, and add everything up."

**With the numbers:** count(l, o) = 1·1 (low) + 1·1 (lower) + 1·1 (lowest) =
**3**; count(e, r) = 1·1 (lower) = **1**. `train_char_bpe` computes the same
totals with a `Counter`.

**In Python:**

```python
# f_w: how often each distinct word occurs
f = {"low": 1, "lower": 1, "lowest": 1}
# n_w(a, b): a immediately followed by b inside w
def n(w, a, b):
    return sum(1 for i in range(len(w) - 1) if w[i] == a and w[i + 1] == b)
def count(a, b):
    # Σ_w f_w · n_w(a, b)
    return sum(f_w * n(w, a, b) for w, f_w in f.items())
count("l", "o"), count("e", "r")  # → (3, 1)
```

**In code:** `train_char_bpe` runs the count-and-merge loop and returns one
`MergeStep` per round, holding the winning pair, its count, the new token
and the text after the merge (one row of the table above).

**Why it matters.** Frequent strings end up as single tokens and rare ones
stay in pieces, which is exactly why token counts differ from word counts,
and why a tokenizer trained mostly on English is cheap for English and
expensive for everything else (section 6).

## 3. Encoding new text: replay the merges in order

**Everyday picture.** Following a recipe card: do step 1 everywhere it
applies, then step 2, and so on. Skipping ahead would give a different dish.

**Tiny worked example.** Encode "lowest" with the three merges learned
above: `l o w e s t` → (merge 1) `lo w e s t` → (merge 2) `low e s t` →
(merge 3) `lowe s t`. Result: 3 tokens. A word never seen in training still
works: "slow" → `s low`; "newer" matches no merge and stays as letters.

```mermaid
flowchart LR
  W["'lowest' as letters<br/>l o w e s t"] --> R1["merge 1: l+o<br/>lo w e s t"]
  R1 --> R2["merge 2: lo+w<br/>low e s t"]
  R2 --> R3["merge 3: low+e<br/>lowe s t"]
  R3 --> OUT["no learned merge applies<br/>tokens: lowe | s | t"]
```

**Reading it:** each box applies one merge rule, in the order the rules
were learned, to every place it fits. The word shrinks from 6 symbols to 3.
It stops when none of the remaining neighbouring pairs is in the rule list.

**The code.** `encode_char_bpe` repeatedly finds, among the pairs present,
the one learned *earliest*, merges it, and loops. Training and encoding
agree because they apply rules in the same order.

**Why it matters.** Encoding is deterministic: the same text always gives
the same ids, which is what makes prompt caching and cost estimates possible.

## 4. Byte-level BPE: nothing is ever unknown

**Everyday picture.** Every file on a computer, whatever it holds, is a
sequence of **bytes**: numbers from 0 to 255. If the smallest bricks in the
kit are those 256 byte values, then no text can ever be unbuildable: emoji,
Chinese, source code or garbage all come apart into bytes.

**Tiny worked example.** In UTF-8 (the standard way to store text as bytes)
"a" is one byte, 97; "é" is two bytes, 195 169; "🙂" is four bytes,
240 159 153 130. An untrained byte-level tokenizer turns "é🙂" into 6 tokens.
After training on English, " password" is one token, id 291, because it was
frequent.

```mermaid
flowchart LR
  T["Text: 'Reset your password'"] --> P["Pre-tokenize (regex)<br/>'Reset' | ' your' | ' password'"]
  P --> B["UTF-8 bytes per chunk<br/>' your' = 32 121 111 117 114"]
  B --> R["Replay merges, earliest first<br/>inside each chunk only"]
  R --> I["Token ids<br/>346 377 309 328 291"]
  I --> D["Decode: look up bytes per id,<br/>concatenate, UTF-8 decode"]
```

**Reading it:** encoding runs left to right. A **regular expression** (a
text-matching pattern) first cuts the text into chunks, each keeping its
leading space; merges never cross a chunk boundary. Each chunk becomes raw
bytes (ids 0-255). The tokenizer then applies whichever learned merge came
earliest among the pairs present, until none applies. The ids shown are what
this module's toy tokenizer produces: "Reset" is two tokens (`Re` `set`),
" password" is one. Decoding is the reverse lookup: every id maps to a fixed
byte string, and concatenating them restores the exact original bytes. That
is why a byte-level tokenizer round-trips any text.

**The code.** `ByteBPE` is `train_char_bpe` with bytes instead of letters,
ids 0-255 for the bytes and 256, 257, ... for each merge. GPT-2, GPT-4's
`cl100k_base`, Llama 3 and most current LLMs work this way.

![Characters per token as the vocabulary grows](figures/primer.ml.tokenization.merge_curve.svg)

**Reading it:** the x-axis is vocabulary size: 256 raw bytes plus the
number of merges learned. The y-axis is compression, how many characters of
held-out English each token covers on average. At 256 (no merges) every byte
is a token, so the value is 1. The first few dozen merges capture the most
frequent pairs (" t", "he", " the") and the curve climbs steeply; later
merges capture rarer strings and the curve flattens. (Our toy corpus runs
out of repeated strings near 500; real corpora keep going.) Production
tokenizers sit far to the right, at 50k to 200k entries, and reach about 4
characters per token on English. Bigger kits mean shorter sequences but a
larger embedding table and output layer.

**In code:** `ByteBPE.train` learns the byte merges, `ByteBPE.encode` and
`ByteBPE.decode` make the round trip, and `compression_curve` trains
tokenizers of growing size and measures each one for the figure above.

**Why it matters.** No "unknown token" failures, ever, for any input. The
price is that unfamiliar scripts fall back to near-byte level and cost many
more tokens.

## 5. The cousins: WordPiece and Unigram

**Everyday picture.** BPE glues the *most common* pair. WordPiece glues the
most *inseparable* pair: two pieces that almost never appear apart, like
"Q" and "u" in English. Unigram works the other way round: start with a huge
kit and keep throwing out the bricks you'd miss least.

**Tiny worked example.** On "low lower lowest", BPE's first merge is `l+o`.
WordPiece's first merge is `s+t`: "s" and "t" each appear exactly once, and
always together.

```mermaid
flowchart TB
  subgraph BPE["BPE (GPT, Llama)"]
    b1[start small] --> b2[merge most frequent pair] --> b3[grow to target size]
  end
  subgraph WP["WordPiece (BERT)"]
    w1[start small] --> w2["merge pair with best<br/>count(ab) / count(a)·count(b)"] --> w3[grow to target size]
  end
  subgraph UG["Unigram (T5, SentencePiece default)"]
    u1[start huge] --> u2[drop pieces whose loss<br/>hurts likelihood least] --> u3[shrink to target size]
  end
```

**Reading it:** the two top rows grow a vocabulary bottom-up and differ only
in how they rank candidate pairs. The bottom row prunes top-down. All three
end with a fixed kit and a rule for cutting new text into pieces from it.

**The math.** WordPiece ranks pairs by

$$
\text{score}(a, b) = \frac{\text{count}(ab)}{\text{count}(a) \cdot \text{count}(b)}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $\text{count}(ab)$ | how often $a$ is immediately followed by $b$ | count(st) = 1 |
| $\text{count}(a)$ | how often $a$ appears at all | count(s) = 1 |
| $\text{count}(b)$ | how often $b$ appears at all | count(t) = 1 |
| fraction bar | divide: pairs are rewarded when their parts rarely appear apart | |

**In words:** "how often the two appear together, divided by how often each
appears at all."

**With the numbers:** score(s, t) = 1 / (1 · 1) = **1.0**; score(l, o) =
3 / (3 · 3) = **0.33**. WordPiece merges s+t first (`wordpiece_scores`).

**In Python:**

```python
f = {"low": 1, "lower": 1, "lowest": 1}
# how often a symbol, or a pair written together, appears
def count(piece):
    return sum(f_w * w.count(piece) for w, f_w in f.items())
def score(a, b):
    # count(ab) / (count(a) · count(b))
    return count(a + b) / (count(a) * count(b))
score("s", "t"), round(score("l", "o"), 2)  # → (1.0, 0.33)
```

BERT marks continuation pieces with `##` ("un", "##believ", "##able").
**SentencePiece** is a library that runs BPE or Unigram directly on raw text,
writing spaces as the visible symbol `▁`.

**Why it matters.** Different model families segment the same text
differently, so their token counts and costs differ. Always count with the
tokenizer of the model you'll actually call.

## 6. Tokens are the unit of cost, speed and memory

**Everyday picture.** A taxi meter that ticks per token, not per mile, with
a pricier meter for the return trip: output tokens usually cost several
times more than input tokens.

**Tiny worked example.** A prompt of 2,000 tokens with a 500-token answer, at
example prices of $3 per million input tokens and $15 per million output
tokens: 0.006 + 0.0075 = **$0.0135**. A million such calls cost $13,500.

```mermaid
flowchart LR
  P["prompt text"] --> TI["count input tokens<br/>meter A: $ per million in"]
  TI --> M[model]
  M --> TO["count output tokens<br/>meter B: $ per million out (pricier)"]
  TO --> BILL["bill = A + B"]
  TI -.-> CTX["also: must fit the<br/>context window"]
```

**Reading it:** two meters run on every call. Input tokens are counted once
on the way in; they also have to fit in the model's context window (the
dotted arrow). Output tokens are counted on the way out, and each one also
takes a full step of generation, so they drive latency as well as cost.

**The math and the code.**

$$
\text{cost} = \frac{T_{in}}{10^6}\,p_{in} + \frac{T_{out}}{10^6}\,p_{out}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $T_{in}$ | input tokens in the call | 2,000 |
| $T_{out}$ | output tokens generated | 500 |
| $10^6$ | one million, because prices are quoted per million tokens | 1,000,000 |
| $p_{in}$ | price per million input tokens, in dollars | 3 |
| $p_{out}$ | price per million output tokens, in dollars | 15 |

**In words:** "input tokens in millions times the input price, plus output
tokens in millions times the output price."

**With the numbers:** 2,000/1,000,000 × 3 + 500/1,000,000 × 15 = 0.006 +
0.0075 = **$0.0135** (`estimate_cost`).

**In Python:**

```python
T_in, T_out = 2_000, 500
# dollars per million tokens
p_in, p_out = 3, 15
round(T_in / 10**6 * p_in + T_out / 10**6 * p_out, 4)  # → 0.0135
```

**Rule of thumb:** English prose averages about **4 characters per token**,
or **0.75 words per token**, so 1,000 tokens ≈ 750 words
(`estimate_tokens`, `tokens_to_words`). Use it for quick estimates; count
with the real tokenizer for anything that matters.

![Characters per token by content type](figures/primer.ml.tokenization.content_types.svg)

**Reading it:** each bar is one kind of text encoded by the same
English-trained tokenizer from this module; taller bars mean each token
carries more text, so the content is cheaper. English prose scores highest
because the merges were learned from English. Code and JSON score lower
because symbols and unusual identifiers were rarely merged. German shares the
alphabet but not the words. Hindi and emoji fall below one character per
token: each character is 3-4 UTF-8 bytes and few of those byte pairs were
ever merged, the worst case of byte-level fallback.

**In code:** `chars_per_token` computes each bar: the length of the text
divided by the number of ids `ByteBPE.encode` returns for it.

**Why it matters.** The same request can differ several-fold in cost,
latency and context usage depending on language and content: dense JSON,
code, tables and non-English text all need more tokens than English prose.

## 7. Quirks the tokenizer explains

**Everyday picture.** Reading through frosted glass that only shows whole
bricks. You can tell which bricks are there, but not the letters printed on
them.

**Tiny worked example** (this module's toy tokenizer):

| Text | Tokens | What it explains |
|---|---|---|
| `" cat"` vs `"cat"` | `[303]` vs `[99, 268]` (`c` `at`) | a leading space makes a different token; the model must learn they mean the same |
| `" 1234"` vs `" 12345"` | `[' 1234']` vs `[' 1234', '5']` | digits split by length and context, so place values don't line up: one reason arithmetic is hard |
| `" strawberry"` | `[' straw', 'berry']` | 2 ids, not 10 letters: "how many r's?" is hard |

```mermaid
flowchart LR
  W["' strawberry'"] --> T["tokenizer"] --> I["ids 396, 311"]
  I --> M["model sees two ids<br/>no letters, no count of r"]
  Q["'how many r's?'"] --> M
```

**Reading it:** the question is about letters, but the letters were
discarded before the model saw anything. It has to have *memorized* how
` straw` and `berry` are spelled from seeing them in text, which is why
spelling and letter-counting questions trip up models that handle much
harder reasoning.

**The code.** `ByteBPE.tokens(text)` shows the pieces for any string; the
demo prints these cases.

**Why it matters.** These quirks explain surprising behaviour: miscounted
letters, shaky arithmetic, odd handling of rare names. Some newer tokenizers
split digits into fixed groups of up to 3 to reduce the arithmetic problem.

## In 20 seconds
- Text is split into subword pieces by BPE: start from characters or bytes
  and repeatedly merge the most frequent neighbouring pair.
- Byte-level BPE can encode anything with no unknown token.
- Cost, latency and context limits are counted in tokens; English is about
  4 characters (0.75 words) per token, while code and non-English text need
  more tokens.

## Self-test questions

**Why subwords instead of whole words or characters?**
Words give a huge vocabulary and unknown words; characters make sequences
several times longer, and attention cost grows with the square of length.
Subwords keep common strings short and still encode anything.

**Walk through BPE training on "low lower lowest".**
Split into letters, count neighbouring pairs, and merge the most frequent:
l+o (3), then lo+w (3), then low+e (2). Record each merge; to encode, replay
the merges in the order learned.

**Why can a byte-level BPE tokenizer never produce an unknown token?**
Its base vocabulary is all 256 byte values, and every string is a sequence of
UTF-8 bytes, so in the worst case text falls back to single bytes.

**What does WordPiece do differently from BPE?**
It merges the pair with the highest count(ab) / (count(a)·count(b)), which
favours pairs whose parts rarely appear apart, instead of the most frequent
pair.

**Your users write in Hindi. What changes in your estimates?**
Expect noticeably more tokens for the same content: higher cost and latency,
and less text per context window. Measure with the actual tokenizer.

**Why are LLMs bad at counting letters and at long arithmetic?**
They see token ids, not characters, and numbers split into chunks that don't
line up with place value.

**What does a call with 2,000 input and 500 output tokens cost at $3/$15 per million?**
0.006 + 0.0075 = $0.0135. And 1,000 tokens is about 750 English words.

## The papers behind this lesson

- **Sennrich, Haddow & Birch (2015), *Neural Machine Translation of Rare Words with Subword Units*.**
  https://arxiv.org/abs/1508.07909. Brought byte pair encoding, a 1990s
  compression trick, to language models as a way to build open-vocabulary
  subword units. [annotated companion](../../papers/bpe-subwords.html)
- **Radford et al. (2019), *Language Models are Unsupervised Multitask Learners* (GPT-2).**
  https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf.
  Introduced byte-level BPE with a pre-tokenization regex, the design most
  LLM tokenizers still follow.
- **Kudo & Richardson (2018), *SentencePiece*.** https://arxiv.org/abs/1808.06226.
  A language-independent tokenizer library that works on raw text and
  encodes spaces as the symbol ▁.
- **Kudo (2018), *Subword Regularization*.** https://arxiv.org/abs/1804.10959.
  Introduced the Unigram language-model tokenizer that prunes a large
  vocabulary down instead of merging up.

## Further reading
- Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units* (the BPE paper, 2015): https://arxiv.org/abs/1508.07909
- Andrej Karpathy, *Let's build the GPT Tokenizer* (video): https://www.youtube.com/watch?v=zduSFxRajkE
- Karpathy's `minbpe` (minimal, clean byte-level BPE): https://github.com/karpathy/minbpe
- OpenAI `tiktoken` (fast BPE used by GPT models): https://github.com/openai/tiktoken
- Hugging Face LLM course, BPE chapter: https://huggingface.co/learn/llm-course/chapter6/5
- Hugging Face LLM course, WordPiece chapter: https://huggingface.co/learn/llm-course/chapter6/6
- Hugging Face LLM course, Unigram chapter: https://huggingface.co/learn/llm-course/chapter6/7
- Kudo & Richardson, *SentencePiece* (2018): https://arxiv.org/abs/1808.06226
- Kudo, *Subword Regularization* (the Unigram LM, 2018): https://arxiv.org/abs/1804.10959
- Petrov et al., *Language Model Tokenizers Introduce Unfairness Between Languages* (2023): https://arxiv.org/abs/2305.15425
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from primer._show import banner, say, table, takeaway

# ---------------------------------------------------------------------------
# 1. Character-level BPE: the textbook algorithm, reproducing the worked example
# ---------------------------------------------------------------------------


@dataclass
class MergeStep:
    """One BPE training step: which pair merged, how often it occurred, and the result."""

    pair: tuple[str, str]
    count: int
    new_token: str
    text_after: str


def _count_pairs(words: dict[tuple[str, ...], int]) -> tuple[Counter, dict[tuple[str, str], int]]:
    """Count adjacent symbol pairs, weighted by word frequency.

    Also records the order in which each pair was *first seen*, used to break
    ties deterministically (earliest pair wins).
    """
    counts: Counter = Counter()
    first_seen: dict[tuple[str, str], int] = {}
    for symbols, freq in words.items():  # dicts preserve insertion (= corpus) order
        for pair in zip(symbols, symbols[1:]):
            counts[pair] += freq
            first_seen.setdefault(pair, len(first_seen))
    return counts, first_seen


def _merge_symbols(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    """Replace every occurrence of `pair` in `symbols` with the concatenated symbol."""
    out, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def _render(occurrences: list[str], words: dict[str, tuple[str, ...]]) -> str:
    """Show the corpus as symbols, `·` between words, like the worked-example table."""
    return " · ".join(" ".join(words[w]) for w in occurrences)


def train_char_bpe(text: str, num_merges: int) -> list[MergeStep]:
    """Learn `num_merges` BPE merges over the characters of whitespace-split words.

    >>> [s.new_token for s in train_char_bpe("low lower lowest", 3)]
    ['lo', 'low', 'lowe']
    """
    occurrences = text.split()
    # Each distinct word -> its current symbol sequence; frequencies -> weights.
    freq = Counter(occurrences)
    current: dict[str, tuple[str, ...]] = {w: tuple(w) for w in dict.fromkeys(occurrences)}
    steps: list[MergeStep] = []
    for _ in range(num_merges):
        counts, first_seen = _count_pairs({current[w]: freq[w] for w in current})
        if not counts:
            break  # every word is a single symbol; nothing left to merge
        # Most frequent pair; ties broken by earliest first occurrence.
        pair = max(counts, key=lambda p: (counts[p], -first_seen[p]))
        for w in current:
            current[w] = _merge_symbols(current[w], pair)
        steps.append(MergeStep(pair, counts[pair], pair[0] + pair[1], _render(occurrences, current)))
    return steps


def encode_char_bpe(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Tokenize one word by replaying learned merges in the order they were learned.

    Replaying by *rank* (earliest-learned merge first) is what makes encoding
    deterministic and consistent with training.

    >>> encode_char_bpe("lowest", [("l", "o"), ("lo", "w"), ("low", "e")])
    ['lowe', 's', 't']
    """
    rank = {p: i for i, p in enumerate(merges)}
    symbols = tuple(word)
    while len(symbols) > 1:
        # Among pairs present, find the one learned earliest.
        candidates = [p for p in zip(symbols, symbols[1:]) if p in rank]
        if not candidates:
            break
        symbols = _merge_symbols(symbols, min(candidates, key=rank.__getitem__))
    return list(symbols)


def wordpiece_scores(text: str) -> dict[tuple[str, str], float]:
    """WordPiece's merge score for every adjacent pair: count(ab) / (count(a)·count(b)).

    BPE picks the most *frequent* pair. WordPiece picks the pair whose parts
    are most strongly associated (they rarely occur apart), which is the
    merge that most increases likelihood under a unigram model.
    """
    occurrences = text.split()
    words = Counter(tuple(w) for w in occurrences)
    pair_counts, _ = _count_pairs(dict(words))
    sym_counts: Counter = Counter()
    for symbols, f in words.items():
        for s in symbols:
            sym_counts[s] += f
    return {p: c / (sym_counts[p[0]] * sym_counts[p[1]]) for p, c in pair_counts.items()}


# ---------------------------------------------------------------------------
# 2. Byte-level BPE: what GPT-2, GPT-4, Llama 3 etc. actually do
# ---------------------------------------------------------------------------

# Pre-tokenization regex, a simplified GPT-2 pattern. It splits text into
# chunks BEFORE merging, so merges never cross these boundaries:
#   * English contractions ('s, 't, ...)
#   * an optional leading space + a run of letters     -> " hello"
#   * an optional leading space + a run of digits      -> " 1234"
#   * an optional leading space + a run of other stuff -> " {}", "!!"
#   * whitespace runs
# The "optional leading space" is why " cat" (mid-sentence) and "cat"
# (start of text) become different tokens.
PRETOKENIZE = re.compile(r"""'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")


class ByteBPE:
    """A minimal byte-level BPE tokenizer (train, encode, decode).

    Token ids 0..255 are the raw bytes. Every learned merge adds one id:
    256, 257, ... `vocab[id]` holds the bytes each id stands for.
    """

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}  # (a, b) -> new id, in learned order
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, text: str, vocab_size: int) -> "ByteBPE":
        """Learn merges until the vocabulary has `vocab_size` entries (>= 256)."""
        assert vocab_size >= 256, "the 256 byte values are always in the vocabulary"
        # Distinct chunks with counts: merging per distinct chunk is much faster
        # than walking the whole text on every merge.
        chunks = Counter(tuple(c.encode("utf-8")) for c in PRETOKENIZE.findall(text))
        next_id = 256
        while next_id < vocab_size:
            pairs: Counter = Counter()
            for ids, f in chunks.items():
                for p in zip(ids, ids[1:]):
                    pairs[p] += f
            if not pairs:
                break
            # Most frequent; ties -> smallest pair (deterministic across runs).
            best = max(pairs, key=lambda p: (pairs[p], -p[0], -p[1]))
            self.merges[best] = next_id
            self.vocab[next_id] = self.vocab[best[0]] + self.vocab[best[1]]
            new_chunks: Counter = Counter()
            for ids, f in chunks.items():
                new_chunks[self._merge(ids, best, next_id)] += f
            chunks = new_chunks
            next_id += 1
        return self

    @staticmethod
    def _merge(ids: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
        out, i = [], 0
        while i < len(ids):
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return tuple(out)

    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = tuple(chunk.encode("utf-8"))
        while len(ids) > 1:
            # Apply the earliest-learned merge available (lowest new id).
            present = [p for p in zip(ids, ids[1:]) if p in self.merges]
            if not present:
                break
            pair = min(present, key=self.merges.__getitem__)
            ids = self._merge(ids, pair, self.merges[pair])
        return list(ids)

    def encode(self, text: str) -> list[int]:
        """Text -> token ids. Works for ANY string: worst case, one id per byte."""
        return [i for chunk in PRETOKENIZE.findall(text) for i in self._encode_chunk(chunk)]

    def decode(self, ids: list[int]) -> str:
        """Token ids -> text. Invalid UTF-8 (e.g. half an emoji) is replaced, never crashes."""
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def tokens(self, text: str) -> list[str]:
        """Human-readable pieces, for display. Partial multi-byte chars show as '\\x..'."""
        out = []
        for i in self.encode(text):
            b = self.vocab[i]
            try:
                out.append(b.decode("utf-8"))
            except UnicodeDecodeError:
                out.append("".join(f"\\x{x:02x}" for x in b))
        return out


# A small English training corpus. Repetition stands in for "web scale":
# frequent strings (" the", " password", " 1234") become single tokens.
TRAINING_TEXT = (
    """
The cat sat on the mat. The cat saw the other cat and the dog.
Reset your password in the portal. Your password must be long. The password
policy says the password expires every year. Error code 1234 means the network
is down; error code 1234 again means the network is still down. Call 1234.
The model reads tokens, not words. The tokenizer splits text into tokens and
the model predicts the next token. Tokens are counted for cost and for context.
We ate the strawberry and the blueberry. The berry was sweet. Straw hats and
straw men. The quick brown fox jumps over the lazy dog.
"""
    * 20
)


def trained_tokenizer(vocab_size: int = 400) -> ByteBPE:
    """The byte-level tokenizer used by the demos and by `primer.ml.big_picture`."""
    return ByteBPE().train(TRAINING_TEXT, vocab_size)


# ---------------------------------------------------------------------------
# 3. Rules of thumb for cost and context estimates
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN = 4.0  # English prose, typical modern tokenizer
WORDS_PER_TOKEN = 0.75


def estimate_tokens(text: str) -> int:
    """Quick estimate: ~4 characters per token for English prose."""
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def tokens_to_words(tokens: int) -> float:
    """1,000 tokens ≈ 750 words."""
    return tokens * WORDS_PER_TOKEN


def estimate_cost(
    input_tokens: int, output_tokens: int, usd_per_m_input: float, usd_per_m_output: float
) -> float:
    """Dollar cost of one call, given per-million-token prices.

    Output tokens are typically priced several times higher than input tokens,
    so long answers cost more than long prompts. Look up current prices on
    your provider's pricing page; they change often.
    """
    return input_tokens / 1e6 * usd_per_m_input + output_tokens / 1e6 * usd_per_m_output


def chars_per_token(tok: ByteBPE, text: str) -> float:
    return len(text) / len(tok.encode(text))


# ---------------------------------------------------------------------------
# 4. Figures (rendered to docs/figures by `make figures`)
# ---------------------------------------------------------------------------

HELD_OUT_ENGLISH = (
    "The user asked the model to reset the password and explain the error code. "
    "The tokenizer counts every token, and the model predicts the next token from the tokens before it."
)

CONTENT_SAMPLES = {
    "English prose": HELD_OUT_ENGLISH,
    "German": "Der Benutzer bat das Modell, das Passwort zurückzusetzen und den Fehlercode zu erklären.",
    "Python code": "def reset_password(user_id: int) -> bool:\n    return api.reset(user_id=user_id, force=True)",
    "JSON": '{"user_id": 81723, "action": "reset_password", "force": true, "ts": "2026-01-05T10:22:31Z"}',
    "Hindi": "उपयोगकर्ता ने मॉडल से पासवर्ड रीसेट करने और त्रुटि कोड समझाने को कहा।",
    "Emoji": "🙂🙃😀🚀🔥✅❌👍🎉💡",
}


def compression_curve(vocab_sizes=(256, 270, 300, 350, 400, 450, 500)) -> list[tuple[int, float]]:
    """(vocab size, characters per token on held-out English) for tokenizers of growing size."""
    return [(v, chars_per_token(ByteBPE().train(TRAINING_TEXT, v), HELD_OUT_ENGLISH)) for v in vocab_sizes]


def figures() -> dict:
    """Data figures for this lesson: {key: matplotlib Figure}."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = {}

    curve = compression_curve()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot([v for v, _ in curve], [c for _, c in curve], marker="o")
    ax.set_xlabel("vocabulary size (256 bytes + learned merges)")
    ax.set_ylabel("characters per token (held-out English)")
    ax.set_title("More merges compress text, with diminishing returns")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figs["merge_curve"] = fig

    tok = trained_tokenizer()
    names = list(CONTENT_SAMPLES)
    values = [chars_per_token(tok, CONTENT_SAMPLES[n]) for n in names]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(names, values)
    ax.set_ylabel("characters per token (higher = cheaper)")
    ax.set_title("Same tokenizer, very different cost by content")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    figs["content_types"] = fig
    return figs


# ---------------------------------------------------------------------------
# 5. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. BPE by hand: 'low lower lowest' in three merges")
    steps = train_char_bpe("low lower lowest", 3)
    rows = [("start", "none", "none", "l o w · l o w e r · l o w e s t")]
    rows += [(i + 1, f"{s.pair[0]} + {s.pair[1]} ({s.count} times)", s.new_token, s.text_after) for i, s in enumerate(steps)]
    table(["step", "most frequent pair", "new token", "text becomes"], rows)
    merges = [s.pair for s in steps]
    say(
        f"""
        Encoding replays the merges in learned order. A word never seen in
        training still works: 'lowest' -> {encode_char_bpe('lowest', merges)},
        'slow' -> {encode_char_bpe('slow', merges)}, 'newer' ->
        {encode_char_bpe('newer', merges)} (no merges apply, so characters).
        """
    )
    ws = wordpiece_scores("low lower lowest")
    best = max(ws, key=ws.get)
    say(
        f"""
        WordPiece scores pairs by count(ab)/(count(a)·count(b)) instead. Its first
        merge would be {best[0]}+{best[1]} (score {ws[best]:.2f}), not l+o
        (score {ws[('l', 'o')]:.2f}): 's' and 't' never appear apart.
        """
    )
    takeaway("BPE = repeatedly merge the most frequent adjacent pair; encoding replays the merges in order.")

    banner("2. Byte-level BPE: no unknown tokens, ever")
    tok = trained_tokenizer()
    say(f"Trained a byte-level BPE on a small English corpus: {tok.vocab_size} tokens (256 bytes + {len(tok.merges)} merges).")
    samples = ["The password expires.", "def f(x): return x**2", "naïve café 🙂", "東京"]
    table(
        ["text", "tokens", "pieces", "round-trips?"],
        [(repr(s), len(tok.encode(s)), tok.tokens(s)[:8], tok.decode(tok.encode(s)) == s) for s in samples],
    )
    say(
        """
        Frequent English words are single tokens. Code, accents, emoji and
        Chinese never seen in training still encode: they fall back toward raw
        bytes. That costs more tokens but never fails.
        """
    )

    banner("3. Quirks that explain odd model behavior")
    lead = tok.encode(" cat"), tok.encode("cat")
    say(f"Leading space: ' cat' -> ids {lead[0]}, 'cat' -> ids {lead[1]}. Different tokens, unrelated IDs.")
    say(
        f"""
        Numbers: ' 1234' -> {tok.tokens(' 1234')} but ' 12345' -> {tok.tokens(' 12345')},
        and '1234' at the start of a line -> {tok.tokens('1234')}. The same
        digits split differently depending on length and surroundings, one
        reason arithmetic is hard for LLMs.
        """
    )
    say(
        f"""
        Letters: 'strawberry' -> {tok.tokens(' strawberry')}. The model sees
        {len(tok.encode(' strawberry'))} ids, not 10 letters, so 'how many r's?'
        is harder than it looks.
        """
    )
    languages = {
        "English": "The weather is nice today and we will go for a walk.",
        "German": "Das Wetter ist heute schön und wir gehen spazieren.",
        "Hindi": "आज मौसम अच्छा है और हम टहलने जाएंगे।",
    }
    table(
        ["language", "chars", "tokens", "chars/token"],
        [(k, len(v), len(tok.encode(v)), chars_per_token(tok, v)) for k, v in languages.items()],
        floatfmt=".2f",
    )
    say(
        """
        Same meaning, very different token counts. A tokenizer trained mostly on
        English has few merges for other scripts; Hindi stays near byte level
        (3 bytes per character). More tokens means more cost, more latency, and
        less content per context window.
        """
    )

    banner("4. Rules of thumb for estimates")
    prose = "Tokens, not words, are the unit of cost, latency and context. " * 10
    say(
        f"""
        {len(prose)} characters of English ≈ {estimate_tokens(prose)} tokens at 4 chars/token.
        1,000 tokens ≈ {tokens_to_words(1000):.0f} words. A 2,000-token prompt with a
        500-token answer at example prices of $3 / $15 per million tokens costs
        ${estimate_cost(2000, 500, 3, 15):.4f}; a million such calls cost
        ${estimate_cost(2000, 500, 3, 15) * 1e6:,.0f}.
        """
    )
    takeaway(
        "Cost, latency and context limits are counted in tokens. English averages about 4 characters per token; "
        "code, JSON and non-English text need more. Measure with the real tokenizer."
    )


if __name__ == "__main__":
    demo()
