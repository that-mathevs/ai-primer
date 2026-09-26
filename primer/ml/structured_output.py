r"""
# Structured output: answers that fit a shape, every time

Run: `python -m primer.ml.structured_output`

New to the notation? `primer.notation` explains every symbol used here from
zero. This lesson builds on sampling from `primer.ml.inference` and on tool
arguments from `primer.agents.tools`.

## The idea

A language model writes one token at a time, and each token is a draw from a
probability distribution (see `primer.ml.inference`). Most of the time you
want free text. Sometimes a program is going to read the answer: a tool call's
arguments, a row for a database, a label from a fixed list. Then the answer
must have an exact **shape**, such as a JSON object with an integer field
called `age`, and one stray character breaks it.

**Structured output** is the family of techniques that make the shape
certain. The main one, **constrained decoding**, is surprisingly small: before
each token is drawn, find every token that could not possibly continue a valid
answer, and forbid it. This lesson builds that from scratch: first for simple
patterns, with a finite-state machine, then for nested JSON, with a stack.

## 1. Asking nicely is not enough

**Everyday picture.** You read a form aloud over the phone to a friend and ask
them to type it in exactly. They are careful and mostly right. But now and
then they add "Sure, here you go!" before the form, or they forget the last
bracket, or a finger slips. A person reading the result shrugs it off. A
program reading it stops dead: one wrong character and the whole thing is
rejected.

**Tiny worked example.** This lesson's toy model, `ToyModel`, has mostly
learned to answer `{"age": 42}`. It is a pretend model, a few lines of NumPy,
that copies that answer with some noise on every choice and a habit of
opening with "Sure". Asked 200 times, with nothing but the request to guide
it, it writes the exact answer 133 times (66.5%). The other 67 look like this:

| What it wrote | What went wrong |
|---|---|
| `Sure{"age": 42}` | a friendly word before the JSON (22 times) |
| `{"age": 42w` | a slip where the closing brace should be |
| `{"age": 428` | an extra digit, then it stopped |
| `{"age"6 42}` | a digit where the colon belongs |
| `Sure{"age": 42}EUR` | chatter at both ends |

Nothing here is a big mistake. Each one is a single bad token.

```mermaid
flowchart LR
  P["Prompt: reply with JSON"] --> M[Model draws one token]
  M --> T[Append it to the answer]
  T -->|not finished| M
  T -->|finished| J{Parser}
  J -->|every token right| OK[Valid JSON]
  J -->|one token wrong| ERR[Parse error:<br/>the whole answer is lost]
```

**Reading it:** follow the loop on the left: the model adds one token at a
time and nothing checks the answer until it is finished. Only then does the
parser look at it, and it has two exits. One bad token anywhere in the loop
sends the whole answer to the error exit, however good the rest was.

That is why failures pile up with length. If each token is right with chance
$p$, and the chances are roughly independent, the whole answer is right only
when every token is:

$$
P(\text{valid}) = p^{\,n}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $p$ | the chance that one token is right, between 0 and 1 | 0.98 |
| $n$ | how many tokens the answer has | 10 |
| $p^{\,n}$ | $p$ multiplied by itself $n$ times: the chance that all $n$ are right | $0.98^{10}$ |
| $P(\text{valid})$ | the chance that the whole answer parses | 0.817 |

**In words:** "the chance that a whole answer is valid is the chance that one
token is right, multiplied together once for every token."

**With the numbers:** a model that gets 98% of tokens right, writing a
10-token answer, produces valid output $0.98^{10} = 0.817$ of the time: nearly
one answer in five is broken. Make the answer 100 tokens long and it drops to
$0.98^{100} = 0.133$.

**In Python:**

```python
# p: chance one token is right; n: tokens in the answer
p, n = 0.98, 10
# p^n: all n tokens right
round(p ** n, 3)  # → 0.817
# a ten times longer answer
round(p ** 100, 3)  # → 0.133
```

![Valid-output rate falls as the answer grows: at 98% per token, 10 tokens are valid 82% of the time and 100 tokens only 13%](figures/primer.ml.structured_output.validity_vs_length.svg)

**Reading it:** the x-axis is the length of the answer in tokens, the y-axis
the share of answers that come out valid. Each curve is one per-token
accuracy. Even the top curve, 99.9% per token, sags over a long answer, and
the 95% curve is near zero by 100 tokens. The dot marks the worked example.
Better prompting moves you to a higher curve, but no curve stays at 100%.

**In code:** `ToyModel` is the pretend model, `sample_answer` draws one answer token by token, and `chance_all_valid` is the formula.

**Why it matters in practice.** A program that calls a model a million times
a day and fails 1% of the time fails ten thousand times a day. Long answers,
nested objects and small models make it worse. Prompting and examples help,
but they only move you to a better curve. To get to 100% you have to change
how tokens are chosen.

## 2. Constrained decoding: mask, then sample

**Everyday picture.** Picture a keyboard whose keys lock and unlock as you
type. You still decide what to write. But at every keystroke, any key that
would make the text break the form is locked for that one keystroke. You
cannot make the mistake, because the key isn't there.

**Tiny worked example.** At the very first step the model scores four tokens
(these scores are **logits**: raw preferences, before softmax turns them
into probabilities; see `primer.ml.attention` for softmax from zero). Only
`{` and `[` can start a JSON value, so the other two are locked:

| Token | Logit $z$ | Allowed? $m$ | Share before the mask | Share after the mask |
|---|---|---|---|---|
| `Sure` | 2.0 | 0 | 0.579 | **0** |
| ` Here` | 1.0 | 0 | 0.213 | **0** |
| `{` | 0.5 | 1 | 0.129 | **0.622** |
| `[` | 0.0 | 1 | 0.078 | **0.378** |

Before the mask the model put 79% of its belief on chatter. After the mask,
the chatter has a chance of exactly zero, and the two allowed tokens share
everything. They keep their odds against each other: `{` was 1.65 times as
likely as `[` before, and it still is.

```mermaid
flowchart LR
  A[Answer so far] --> M[Model: a logit<br/>for every token]
  A --> C[Constraint: which tokens<br/>can still lead to a valid answer?]
  M --> K[Set forbidden logits to minus infinity]
  C --> K
  K --> S[Softmax over what is left<br/>then sample]
  S --> N{End token?}
  N -->|no| A2[Append the token] --> A
  N -->|yes| D[Done: valid by construction]
```

**Reading it:** two arrows leave the answer so far. The top path is the
ordinary model, which scores every token as it always does. The bottom path
is the new part: a checker that knows the shape and says which tokens are
still possible. They meet in the mask box, and from there it is ordinary
sampling again. The end token is itself just a token, so it is allowed only
when the answer is complete.

The mask as a formula is softmax with a 0-or-1 switch on every term:

$$
p_i = \frac{m_i \, e^{z_i}}{\sum_{j=1}^{|V|} m_j \, e^{z_j}}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $V$ | the vocabulary: every token the model can write | the 4 tokens in the table |
| $\lvert V \rvert$ | how many tokens that is | 4 |
| $i$ | the token whose probability we are computing | 3, the `{` |
| $z_i$ | token $i$'s logit | $z_3 = 0.5$ |
| $m_i$ | the mask: 1 if token $i$ can continue a valid answer, 0 if not | $m = (0, 0, 1, 1)$ |
| $e^{z_i}$ | $e \approx 2.718$ raised to the logit: always positive | $e^{0.5} = 1.649$ |
| $\sum_{j=1}^{\lvert V \rvert}$ | add up the following for every token $j$ | $0 + 0 + 1.649 + 1$ |
| $p_i$ | the probability that token $i$ is drawn | 0.622 |

**In words:** "a token's probability is its usual softmax share, except that
forbidden tokens count as zero on top and bottom, so the allowed tokens share
all the probability between them."

**With the numbers:** $p_{\{} = \dfrac{1 \cdot e^{0.5}}{0 + 0 + 1 \cdot e^{0.5} + 1 \cdot e^{0}} = \dfrac{1.649}{2.649} = 0.622$.

**In Python:**

```python
import math
# Sure, " Here", "{", "["
z = [2.0, 1.0, 0.5, 0.0]
# m_i: 1 if the token may come next
m = [0, 0, 1, 1]
# m_i e^(z_i): forbidden tokens contribute nothing
kept = [m_i * math.exp(z_i) for m_i, z_i in zip(m, z)]
[round(k, 3) for k in kept]  # → [0.0, 0.0, 1.649, 1.0]
# Σ_j m_j e^(z_j)
total = sum(kept)
round(total, 3)  # → 2.649
[round(k / total, 3) for k in kept]  # → [0.0, 0.0, 0.622, 0.378]
# without the mask, most of the belief went to chatter
e = [math.exp(z_i) for z_i in z]
[round(x / sum(e), 3) for x in e]  # → [0.579, 0.213, 0.129, 0.078]
```

Setting a logit to minus infinity does the same thing, because
$e^{-\infty} = 0$: it is the trick the causal mask uses in
`primer.ml.attention`. After masking, temperature, top-k and top-p from
`primer.ml.inference` work exactly as before, on the tokens that are left.

![At the toy model's first step, 19% of its belief sits on Sure; after the mask, the brace gets all of it](figures/primer.ml.structured_output.masked_step.svg)

**Reading it:** these are the toy model's real first-step probabilities for
the `{"age": 42}` task, top four tokens only. Grey bars are what the model
wanted; blue bars are what it may choose from after the mask. "Sure" had
nearly a fifth of the belief and drops to zero. The brace, the only legal way
to begin, takes everything.

**Why 100% by construction.** The checker keeps one promise: *every answer so
far can still be finished validly*. It holds at the start (the empty answer
can be finished). Each step only allows a token that keeps it. The end token
is allowed only when the answer is already complete. So every answer that
ends is valid. No luck is involved.

![Unconstrained, the age answer is valid 66.5% of the time and the person object 18%; constrained, every finished answer is valid](figures/primer.ml.structured_output.validity_bars.svg)

**Reading it:** each bar is 200 answers from the toy model, split by what
happened. Green is valid. The unconstrained bars show all three failures:
chatter before the JSON, a broken character inside, and (rarely) running out
of token budget. The person object is longer, so it breaks far more often,
just as $p^n$ predicts. The constrained bars have no chatter and nothing
broken. The one sliver left, 5 of the 200 person answers, is **cut off**: the
40-token budget ran out mid-object. The guarantee covers every prefix, but
only finishing makes a whole answer, so leave room in the budget.

**In code:** `masked_softmax` applies the mask, `sample_answer` takes an optional constraint and masks every step, and `validity_experiment` produces the bars above.

**Why it matters in practice.** Constrained decoding changes nothing about the
model: no retraining, same weights. It only changes which tokens may be
drawn, which is why it can be added to any open model at serving time. The
hard part is the checker: answering "which of 100,000 tokens could still lead
to a valid answer?" fast, at every step. The next two sections build it.

## 3. From a pattern to a state machine

**Everyday picture.** A subway map. You stand at a station. Each line leaving
it is labelled with one character. To write a character, you ride the line
with that label; if no line from your station has it, that character is
impossible here. Some stations are marked "you may stop here". That map is a
**finite-state machine**: a fixed set of states (stations) and one move per
character. A **regular expression**, the pattern language behind
`[0-9]+` and `cat|car|dog`, can always be drawn as one.

**Tiny worked example.** The pattern `cat|car|dog` (one of three words)
becomes this machine:

```mermaid
stateDiagram-v2
  [*] --> S0
  S0 --> S1: c
  S0 --> S2: d
  S1 --> S3: a
  S2 --> S4: o
  S3 --> S5: r
  S3 --> S6: t
  S4 --> S7: g
  S5 --> [*]
  S6 --> [*]
  S7 --> [*]
```

**Reading it:** start at S0 and read a word one character at a time. "cat"
goes S0, S1, S3, S6, and S6 has an exit arrow, so "cat" is accepted. "cow"
gets stuck at S1, which has no "o" line. After "ca" (state S3) only "r" and
"t" are possible: the machine has turned "what may come next?" into "which
lines leave this station?".

**Tokens are not characters.** A model writes tokens, and one token can hold
several characters. The rule: **a token is allowed if the machine can swallow
all of its characters, one after another, without getting stuck.** With a
vocabulary of 13 tokens:

| Token | From S0 | From S3 (after "ca") |
|---|---|---|
| `c`, `d` | allowed | stuck |
| `ca`, `cat`, `do`, `dog` | allowed: every character has a line | stuck |
| `a`, `at`, `o`, `og`, `g` | stuck at the first character | stuck |
| `t`, `r` | stuck | allowed |

`cat` is allowed at the start even though no single line reads "cat": the
machine rides c, then a, then t. `at` is part of a real word and still never
allowed at the start, because S0 has no "a" line.

$$
\delta^*(s, t) = \delta\big(\cdots\delta(\delta(s, c_1), c_2)\cdots, c_k\big)
\qquad
A(s) = \{\, t \in V : \delta^*(s, t) \text{ is defined} \,\}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $s$ | a state of the machine | S0 |
| $\delta(s, c)$ | the **transition function**: the state you reach from $s$ by reading character $c$, or undefined if there is no such line | $\delta(0, \text{c}) = 1$ |
| $t$ | a token | `cat` |
| $c_1, \ldots, c_k$ | the characters of $t$, in order; $k$ is how many | c, a, t; $k = 3$ |
| $\delta^*(s, t)$ | read every character of $t$ in turn, starting from $s$ | $\delta^*(0, \texttt{cat}) = 6$ |
| $V$ | the vocabulary | the 13 tokens above |
| $\{\, t \in V : \ldots \,\}$ | "the set of every token $t$ in $V$ for which ... holds" | |
| $A(s)$ | the allowed tokens at state $s$ | $A(0) = \{$c, d, ca, cat, do, dog$\}$ |

The end token joins $A(s)$ only when $s$ is an **accepting** state (S5, S6
or S7 here), because only there is the text complete.

**In words:** "to see whether a token fits, walk its characters through the
machine one at a time; the allowed tokens are the ones that never get stuck."

**With the numbers:** $\delta^*(0, \texttt{cat}) = \delta(\delta(\delta(0, \text{c}), \text{a}), \text{t}) = \delta(\delta(1, \text{a}), \text{t}) = \delta(3, \text{t}) = 6$,
so `cat` is in $A(0)$. $\delta(0, \text{a})$ is undefined, so `at` is not.

**In Python:**

```python
# the cat|car|dog machine: state -> {character: next state}
delta = {0: {"c": 1, "d": 2}, 1: {"a": 3}, 2: {"o": 4}, 3: {"r": 5, "t": 6}, 4: {"g": 7}, 5: {}, 6: {}, 7: {}}
accepting = {5, 6, 7}
def walk(s, token):
    # δ*: one δ per character; None means stuck
    for ch in token:
        s = delta[s].get(ch) if s is not None else None
    return s
walk(0, "cat")  # → 6
walk(0, "at")  # → None
V = ["c", "a", "t", "r", "d", "o", "g", "ca", "cat", "do", "dog", "at", "og"]
# A(0): every token the machine can swallow whole from the start
[t for t in V if walk(0, t) is not None]  # → ['c', 'd', 'ca', 'cat', 'do', 'dog']
# A(3), after "ca"
[t for t in V if walk(3, t) is not None]  # → ['t', 'r']
# the end token only where the text is complete
walk(0, "cat") in accepting  # → True
```

**How a pattern becomes a machine.** Two classic steps, both in the code:

```mermaid
flowchart LR
  P["Pattern<br/>cat|car|dog"] --> T[Thompson's construction:<br/>one small machine per piece,<br/>glued with free jumps]
  T --> N[NFA: may be in<br/>several states at once]
  N --> D[Subset construction:<br/>each new state is a set<br/>of NFA states]
  D --> F[DFA: exactly one<br/>state at a time]
  F --> TB[Table: for every state,<br/>which tokens are allowed]
```

**Reading it:** left to right, the pattern gets more mechanical. Thompson's
construction reads the pattern like a sentence and builds a tiny machine for
each piece: one line for a character, a fork for `|`, a loop for `+`. Glued
together they make an **NFA** (a non-deterministic machine), which can be in
several states at once: after "c" it is both "inside cat" and "inside car".
The subset construction turns each *set* of NFA states into one state of a
**DFA** (a deterministic machine), which is always in exactly one state, so
following it is a dictionary lookup. The last box is the payoff: since the
DFA has a fixed number of states, the allowed tokens can be worked out for
every state before generation starts.

The pattern `-?[0-9]+` (an optional minus sign, then digits) compiles to just
three states: the start, "saw a minus", and "saw at least one digit", the only
accepting one. The lesson's running example, `\{"age": [0-9]+\}`, compiles to
11.

![Each row is one state of the age machine, each column a token; only a handful of cells are lit, and the digit states allow many tokens at once](figures/primer.ml.structured_output.mask_table.svg)

**Reading it:** rows are the 11 states, labelled by the text that reaches
them; columns are tokens; a dark cell means "allowed here". Most rows have one
or two dark cells: the structure is fixed, so only one next character is
legal, written alone or as the start of a longer token such as `"age"` or
`": `. The two digit rows are
where the model has real choice: any digit, a two-digit token such as `42`,
or, once one digit is down, the closing brace. The last row allows only the
end token. This whole table is computed once, before the first token.

**In code:** `compile_pattern` runs both constructions and returns a `DFA`; `DFA.walk` follows text through it; `allowed_tokens` is $A(s)$, and `mask_table` precomputes it for every state. `RegexConstraint` plugs the table into `sample_answer`.

**Why it matters in practice.** Integers, dates, enums, phone numbers and
fixed-key objects are all patterns. Reframing generation as moving between
the states of a machine, with the allowed tokens indexed per state, is the
idea of Willard and Louf (2023) behind the open-source Outlines library, and
it makes each step's mask a single lookup.

## 4. JSON Schema: nesting needs a stack

**Everyday picture.** A stack of plates. Each time you open something, a
bracket, a brace or a quote, you put a plate on the stack with a note: "an
array is open", "an object is open". To close something, you may only take the
**top** plate: close the most recent thing first. When the stack is empty,
you're done.

A subway map can't do this job. JSON can nest as deep as you like: `[[[[...]]]]`.
A machine with, say, 50 states can't tell 50 open brackets from 51, so it
can't know how many closers it still owes. Counting without limit needs
memory without limit. A finite-state machine plus a stack is called a
**pushdown automaton**, and it is exactly enough for nested formats such as
JSON, SQL and most programming languages.

**Tiny worked example.** Read `{"a": [1, {"b": ` one character at a time.

```mermaid
flowchart LR
  S1["after {<br/>stack: object"] --> S2["after [<br/>stack: object, array"]
  S2 --> S3["after the inner {<br/>stack: object, array, object"]
  S3 --> S4["after 2}<br/>stack: object, array"]
  S4 --> S5["after ]}<br/>stack: empty, done"]
```

**Reading it:** each box is a moment in the reading, with the stack written
bottom first. Every opener adds a frame on the right; every closer removes
the rightmost one. At the third box the innermost open thing is an object,
so `}` is the only closer allowed, and `]` would be refused even though an
array is open further down. The last box is empty: the value is complete, and
only now is the end token allowed.

The checker answers one question about any prefix: can it still be finished?

| Prefix | Status | Why |
|---|---|---|
| `{"a": [1, {"b": ` | open | a value for "b" may come next |
| `{"a": [1}` | dead | the array is on top, so `}` can't close it |
| `{"a": [1, {"b": 2}]}` | complete | the stack is empty |

**The schema steers every character.** A **JSON Schema** (see
`primer.agents.tools`) names the fields and their types. This lesson's person
schema allows `name` (a string), `age` (an integer) and `pets` (an array of
"cat" or "dog"); `name` and `age` are required and nothing else is allowed.
The checker enforces each rule at the first character that breaks it:

| Prefix | Status | The rule that decides |
|---|---|---|
| `{"na` | open | "na" can still become "name" |
| `{"nx` | dead | no field starts with "nx" |
| `{"name": "Ada"}` | dead | `age` is required, so the object may not close yet |
| `{"age": 3.` | dead | an integer has no decimal point |
| `{"age": 01` | dead | JSON numbers never start with a zero followed by digits |
| `{"pets": ["x` | dead | only "cat" and "dog" are listed |
| `{"name": "Ada", "age": 36}` | complete | every required field is present |

Inside one frame, a small state machine does the work. Here is an object's:

```mermaid
stateDiagram-v2
  [*] --> open: {
  open --> key: quote, if a field may be added
  key --> key: a letter that still spells an unused field
  key --> colon: closing quote, if the key is complete
  colon --> value: colon, then at most one space
  value --> after_value: the value's own frame is pushed and later popped
  after_value --> key: comma, space, quote
  after_value --> [*]: }, only if every required field is present
  open --> [*]: }, only if nothing is required
```

**Reading it:** these are the phases one object frame moves through, from
opening brace to closing brace. The labels carry the schema's rules: a key
letter is allowed only if it still spells a field not yet used, and the
closing brace only if every required field is in. The value arrow is where
nesting happens: the value gets its own frame pushed on top of the stack, and
this frame waits in "value" until that frame is popped.

The leaves of JSON (numbers, `true`, `false`, `null`) are regular patterns,
so the checker reads numbers with the machines from section 3. Real engines
split the work the same way: patterns for the small pieces, a stack for the
nesting.

One rule here is deliberate: at most one space after `:` and `,`. JSON allows
any amount of whitespace, and a constrained model that has lost its way can
pad with spaces until the budget runs out. Real engines limit whitespace for
the same reason.

**In code:** `SchemaChecker.step` reads one character and returns the new stack (a tuple of `Frame` entries), or nothing when the prefix is dead; `SchemaChecker.status` answers open, dead or complete for a prefix; `SchemaChecker.open_containers` shows the stack; `SchemaConstraint` tries every token against the stack at each step.

**Why it matters in practice.** This is how a schema becomes a guarantee:
compile the schema into grammar rules, run them as a pushdown automaton, and
mask every token that would kill it. Geng et al. (2023) showed the approach
works for structured tasks without any fine-tuning, and PICARD (Scholak et
al., 2021) did the same for SQL by parsing incrementally. The stack has a
cost: it can grow without limit, so the masks can't all be tabled in advance
as they were for a pattern. Section 5 comes back to that.

## 5. Costs and pitfalls

Constrained decoding guarantees the shape. It does not guarantee a good
answer, and it isn't free.

### The mask changes what the model says

**Everyday picture.** A satellite navigation system that only forbids illegal
turns, one junction at a time, and never looks ahead. It happily takes the
motorway because the motorway looks fastest right now, and only at the end
discovers that the one legal exit is a long detour. A driver who could see the
whole map would have taken the side road from the start.

**Tiny worked example 1: forced to invent.** The model is asked for a
person's age, but the text never says it. The model's honest belief for the
next token:

| Token | Model's belief | Allowed by `"type": "integer"`? | After the mask |
|---|---|---|---|
| `null` | 0.80 | no | 0 |
| `3` | 0.12 | yes | 0.12 / 0.20 = **0.60** |
| `5` | 0.08 | yes | 0.08 / 0.20 = **0.40** |

The mask throws away 80% of the model's belief, and the answer is a made-up
age, delivered as confidently as a real one. The toy model does this too:
told to write the person schema while it "wants" to write only a name, it is
forced to add an age, and invents numbers such as `9700`. The fix is in the
schema, not the decoder: allow `null` (`"type": ["integer", "null"]`), or add
a field for "not stated". A related finding (Tam et al., 2024): forcing a strict
format from the first token can hurt a model's reasoning, so let the model
reason in free text first, or put a reasoning field before the answer field.

**Tiny worked example 2: locally tempting, globally wrong.** A two-token
model. Valid answers must end in "y".

```mermaid
flowchart LR
  S[start] -->|A 0.9| A[A]
  S -->|B 0.1| B[B]
  A -->|x 0.99| AX["Ax: invalid"]
  A -->|y 0.01| AY["Ay: valid, 0.9 × 0.01 = 0.009"]
  B -->|x 0.1| BX["Bx: invalid"]
  B -->|y 0.9| BY["By: valid, 0.1 × 0.9 = 0.09"]
```

**Reading it:** each arrow is one token with the model's probability for it,
and each leaf is a whole answer with the product of the probabilities along
its path. Of the two valid answers, the model much prefers By (0.09 against
0.009). But masking decides one token at a time. At the first step both A
and B can still end in "y", so nothing is masked, and A wins 90% of the time.
At the second step the mask forces "y". The result: Ay, the answer the model
itself thought ten times less likely, comes out 90% of the time.

$$
P(y \mid \text{valid}) = \frac{P(y)}{\sum_{y' \in \mathcal{L}} P(y')}
\qquad
P_{\text{mask}}(y) = \prod_{i=1}^{n} p'(y_i \mid y_{<i})
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $y$ | one whole answer | Ay |
| $y_i$ | its $i$-th token; $y_{<i}$ is every token before it | $y_2 =$ y, $y_{<2} =$ A |
| $n$ | the number of tokens in the answer | 2 |
| $P(y)$ | the model's own chance of writing $y$: its token chances multiplied together | $0.9 \times 0.01 = 0.009$ |
| $\mathcal{L}$ | the set of valid answers (the "language" the grammar allows) | {Ay, By} |
| $\sum_{y' \in \mathcal{L}}$ | add up over every valid answer $y'$ | $0.009 + 0.09$ |
| $P(y \mid \text{valid})$ | "the chance of $y$ given that the answer is valid": the model's own odds, among valid answers only | 0.091 |
| $p'(y_i \mid y_{<i})$ | the masked, renormalised chance of token $y_i$ at that step | $p'(\text{y} \mid \text{A}) = 1$ |
| $\prod_{i=1}^{n}$ | multiply together over every step | $0.9 \times 1$ |
| $P_{\text{mask}}(y)$ | the chance that token-by-token masking produces $y$ | 0.9 |

**In words:** "what the model believes, restricted to valid answers, is each
valid answer's probability divided by the total of all valid ones; what
masking actually produces is the product of the step-by-step masked
probabilities, and the two need not agree."

**With the numbers:** $P(\text{Ay} \mid \text{valid}) = 0.009 / 0.099 = 0.091$,
but $P_{\text{mask}}(\text{Ay}) = 0.9 \times 1 = 0.9$: ten times the model's
own preference.

**In Python:**

```python
first = {"A": 0.9, "B": 0.1}
second = {"A": {"x": 0.99, "y": 0.01}, "B": {"x": 0.1, "y": 0.9}}
# P(y) for each valid answer: the model's token chances multiplied
P = {a + "y": first[a] * second[a]["y"] for a in first}
{k: round(v, 3) for k, v in P.items()}  # → {'Ay': 0.009, 'By': 0.09}
# P(y | valid): divide by the total over valid answers
total = sum(P.values())
{k: round(v / total, 3) for k, v in P.items()}  # → {'Ay': 0.091, 'By': 0.909}
# P_mask: step 1 keeps both, step 2 renormalises "y" to 1
{a + "y": first[a] * (second[a]["y"] / second[a]["y"]) for a in first}  # → {'Ay': 0.9, 'By': 0.1}
```

![The model's own odds among valid answers favour By 91 to 9; token-by-token masking produces Ay 90% of the time](figures/primer.ml.structured_output.distortion.svg)

**Reading it:** two answers, two bars each. Grey is the model's own
preference among valid answers; blue is what masked decoding actually
produces. The bars point in opposite directions. Nothing invalid comes out,
yet the distribution is badly bent. Park et al. (2024) name this problem and
propose a correction; in practice it is milder when the model already writes
the format well, because then the mask rarely has to overrule it.

The toy model shows the everyday version: once noise knocks it off its answer,
the mask keeps it legal but not sensible, and it writes digits until the
closing brace happens to win, as in `{"age": 4174209}`.

**In code:** `forced_choice` renormalises a belief over the allowed tokens and reports the share thrown away; `NULLABLE_AGE_SCHEMA` is the fix; `distortion_example` computes both distributions above.

### Token boundaries

**Everyday picture.** You can say "forty-two", or spell it "four, two". Both
arrive at the same text, but only one is how you would naturally say it.

**Tiny worked example.** The toy vocabulary has a `42` token and single
digits, so "42" can be written two ways: `42`, or `4` then `2`. The whole
answer `{"age": 42}` can be spelled 16 ways, and the person answer 1,536
ways. The mask allows every one of them, but a trained model has almost only
ever seen the first, its tokenizer's usual spelling (see
`primer.ml.tokenization`). When the mask forces it onto an unusual spelling,
it is in unfamiliar territory and its next choices get worse.

```mermaid
flowchart LR
  S["after the space"] -->|42, the usual token| E[before the brace]
  S -->|4| M[after the 4]
  M -->|2| E
```

**Reading it:** two paths lead from the same place to the same place and
write the same text. The top path is the one the model saw thousands of
times in training; the bottom one it rarely saw. The checker walks
characters, so it cannot tell them apart; only the model can.

Two more boundary effects follow from the same fact. A single token can cross
several structural boundaries at once, such as `"}` closing a string and an
object together; walking the token character by character, as `allowed_tokens`
does, handles that. And if the prompt ends in the middle of what would
normally be one token (a prompt ending in `{"age":` when the model would
usually write `": ` as one token), the natural token is no longer available.
Some engines back up one token and let the model rewrite it, which is called
**token healing**.

**In code:** `tokenizations` lists every way a vocabulary can spell a text.

### The speed of computing masks

**Everyday picture.** Before every keystroke, a proofreader checks every word
in the dictionary against the rules: slow. Or: a card for every station,
printed once, listing which words fit there: fast, once the cards exist.

**Tiny worked example.** A real vocabulary has around 128,000 tokens. Say
they average 4 characters, the answer is 200 tokens long, and the pattern's
machine has 50 states. Checking every token at every step walks
200 × 128,000 × 4 = 102.4 million characters for one answer. Building the
table walks 50 × 128,000 × 4 = 25.6 million characters once; after that,
each step only reads one row of 128,000 yes-or-no entries.

$$
W_{\text{naive}} = n \cdot \lvert V \rvert \cdot \bar{L}
\qquad
W_{\text{table}} = S \cdot \lvert V \rvert \cdot \bar{L} + n \cdot \lvert V \rvert
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $W$ | work: characters walked or table entries read | |
| $n$ | tokens generated | 200 |
| $\lvert V \rvert$ | vocabulary size | 128,000 |
| $\bar{L}$ | the average token length in characters (the bar means "average") | 4 |
| $S$ | states in the machine | 50 |
| $\cdot$ | multiply | |

**In words:** "the naive way walks every token at every step; the table walks
every token once per state, up front, then reads one row per step."

**With the numbers:** $W_{\text{naive}} = 200 \cdot 128{,}000 \cdot 4 = 102{,}400{,}000$
for every answer. $W_{\text{table}} = 50 \cdot 128{,}000 \cdot 4 + 200 \cdot 128{,}000 = 51{,}200{,}000$
for the first answer, and only the second term, 25.6 million cheap reads,
for each answer after that.

**In Python:**

```python
n, V, L_bar, S = 200, 128_000, 4, 50
# W_naive = n · |V| · L̄
n * V * L_bar  # → 102400000
# W_table = S · |V| · L̄ + n · |V|
S * V * L_bar + n * V  # → 51200000
# ten answers: the table's build cost is paid once
10 * n * V * L_bar, S * V * L_bar + 10 * n * V  # → (1024000000, 281600000)
```

![Checking every token at every step costs the same for every answer; the table pays once, then grows slowly](figures/primer.ml.structured_output.mask_cost.svg)

**Reading it:** the x-axis counts answers generated with one schema, the
y-axis the total work so far. The naive line climbs steeply and steadily. The
table line starts above zero (the build) and then climbs gently (one
row read per step). They cross partway through the very first answer. That is why hosted
APIs compile a schema once and cache it: the first request with a new schema
is slower, and the ones after are fast.

For a schema with nesting, the stack can grow without limit, so no table can
cover every situation. XGrammar (Dong et al., 2024) splits the vocabulary:
most tokens are **context-independent** (whether they fit depends only on
the current grammar position, not on what is deeper in the stack), and those
are prechecked into tables; only the few context-dependent ones are checked
against the stack at run time.

**In code:** `mask_cost` is the formula; `mask_table` builds the table once and `RegexConstraint` reads a row per step, while `SchemaConstraint` walks every token against the stack at every step, the naive way.

### When validating and retrying is enough

**Everyday picture.** Instead of a form that can't be filled in wrong, you
let people fill in a blank sheet, check it, and hand it back with a note when
it is wrong. Fine if most people get it right first time; miserable if most
don't.

**Tiny worked example.** The toy model writes a valid age answer 66.5% of the
time. Retrying until it succeeds takes 1 / 0.665 = 1.50 attempts on average,
and three tries in a row all fail only 0.335³ = 3.8% of the time. For the
longer person object, valid 18% of the time, it takes 5.56 attempts on
average: slow and expensive.

```mermaid
flowchart LR
  A[Ask the model] --> V{Validate}
  V -->|valid| D[Use it]
  V -->|invalid| E[Send back the<br/>validator's message] --> A
  E -.->|too many tries| F[Give up or<br/>fall back]
```

**Reading it:** the happy path goes straight through. Every failure costs a
full extra model call, round the loop. Sending back the validator's exact
message (as `primer.agents.tools.validate` produces) makes the second try
much more likely to succeed. The dotted exit is the budget: a loop needs a
limit.

$$
\mathbb{E}[\text{attempts}] = \frac{1}{p}
\qquad
P(\text{all } k \text{ tries fail}) = (1 - p)^k
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $p$ | the chance one try is valid | 0.665 |
| $\mathbb{E}[\ldots]$ | the **expected value**: the long-run average over many repeats | |
| $\frac{1}{p}$ | the average number of tries to the first success | 1 / 0.665 = 1.50 |
| $k$ | a number of tries | 3 |
| $1 - p$ | the chance one try fails | 0.335 |
| $(1 - p)^k$ | the chance that $k$ independent tries all fail | $0.335^3 = 0.0376$ |

**In words:** "if each try succeeds with chance p, you need one over p tries
on average, and the chance that k tries all fail is the chance of one
failure, multiplied together k times."

**With the numbers:** $1 / 0.665 = 1.50$; $(1 - 0.665)^3 = 0.0376$; for the
person object, $1 / 0.18 = 5.56$.

**In Python:**

```python
# the toy model's unconstrained rate on the age answer
p = 0.665
# E[attempts] = 1 / p
round(1 / p, 2)  # → 1.5
# (1 - p)^k: three tries, all invalid
round((1 - p) ** 3, 4)  # → 0.0376
# the longer person answer, valid 18% of the time
round(1 / 0.18, 2)  # → 5.56
```

![Expected attempts are close to 1 for a model that is usually right, and shoot up as the valid rate falls](figures/primer.ml.structured_output.retry.svg)

**Reading it:** the x-axis is the chance a single try is valid; the y-axis is
the average number of tries until one is. The curve is flat on the right
and steep on the left. The dots mark three cases: a model that is 95% right
barely notices retries (1.05 tries); the toy age answer needs half a try
extra; the toy person answer needs more than five calls.

Validate and retry when: you can't change the decoder (a hosted model
without a structured-output option), the model is already right nearly every
time, or the rule can't be expressed as a grammar. Constrain when answers are
long or nested, the model is small, or latency matters. Either way, keep the
validator: a grammar enforces shape, not rules such as `minimum`, and never
truth.

**In code:** `expected_attempts` and `chance_still_failing` are the two formulas.

## 6. JSON mode, strict schemas and tool calls

**Everyday picture.** Two paper forms. One only insists that you write in
block capitals: whatever you write is readable, but nothing says what goes
where. The other has a labelled box for each field, and tick boxes where
only certain answers are allowed. The first is **JSON mode**; the second is
a **strict schema**.

**Tiny worked example.** Give the toy model a reference answer that leaves
out the age, `{"name": "Ada"}`, and ask 50 times. With JSON mode (the empty
schema `{}`, which allows any JSON value) every finished answer parses: 44
copies of `{"name": "Ada"}` with the required age missing, one
`{"name": 42628}` whose name is a number, and one bare `false`. All valid
JSON; not one valid person. (The other 3 ran out of budget.) With the person
schema, every one of the 37 answers that finish has an age, because the
closing brace stays locked until one is written. Since the model had no age
to give, it invents one, such as `{"name": "Ada","age": 9700,"pets": []}`:
section 5's warning in action. The other 13 show a second cost of forcing a
model off its path: it loses its way and wanders, legally, until the token
budget runs out.

A tool call is the same thing with a name attached: the model's arguments are
a JSON object that must fit the tool's `input_schema` (see
`primer.agents.tools` and `primer.agents.llm`).

```mermaid
sequenceDiagram
  participant App as Your code
  participant API as Model server
  participant G as Grammar engine
  App->>API: messages + tool with a strict JSON Schema
  API->>G: compile the schema (cached for next time)
  loop every token
    G-->>API: mask of allowed tokens
    API->>API: mask, softmax, sample
  end
  API-->>App: tool_use with arguments that fit the schema
  App->>App: business rules: does customer C-999 exist? is the amount above the minimum?
```

**Reading it:** the grammar engine lives inside the model server, next to
sampling. It compiles the schema once (the slow first request from section
5) and then hands a mask to every step of the loop. What reaches your code is
guaranteed to have the right shape. The last arrow is the part no grammar
covers: whether the values are true and allowed. The payment tool in
`primer.agents.tools` makes this concrete: `{"amount": 0, "currency": "USD"}`
fits the schema's shape exactly, and `primer.agents.tools.validate` still
rejects it, because `"minimum": 0.01` is a rule about the value, not the
shape.

**In code:** `SchemaConstraint` with the schema `{}` is JSON mode, and with `PERSON_SCHEMA` it is a strict schema; `primer.agents.tools.ToolRegistry.definitions` shows how a strict tool definition is sent.

**Why it matters in practice.** You now have three tools and know what each
buys. JSON mode guarantees something parseable. A strict schema guarantees
the shape your code expects, so the parsing and type-checking code
disappears. Validation after the fact still catches what only your program
knows. Hosted APIs (for example Claude's structured outputs and strict tool
use) and open engines (Outlines, llama.cpp grammars, XGrammar) all run the
mechanism built in this lesson: a checker that masks the logits before every
draw.

## In 20 seconds

- **The problem:** every token is a chance to break the format, so an
  answer of n tokens is valid about $p^n$ of the time. Prompting raises p
  but never to 1.
- **Constrained decoding:** before each draw, set the logit of every token
  that can't lead to a valid answer to minus infinity, then sample as usual.
  Allow the end token only when the answer is complete. Valid by
  construction, if the answer is allowed to finish.
- **Patterns:** compile to a finite-state machine; a token is allowed if the
  machine can read all its characters; precompute the allowed tokens for
  every state, so each step is a lookup.
- **JSON Schema:** nesting needs a stack (a pushdown automaton); the schema
  decides which keys, types and closers are legal at each character.
- **Pitfalls:** the mask can force a made-up value or bend the model's
  preferences; tokens don't align with grammar pieces; nested grammars cost
  more to check. Retrying is fine when the model is usually right. Always
  validate business rules afterwards.

## Self-test questions

**Why does asking a model for JSON in the prompt fail some of the time, and why do longer outputs fail more?**
Each token is a separate draw with some small chance of being wrong, and a
single wrong token breaks the parse. The chance that all n tokens are right
is about $p^n$, which shrinks as n grows: 98% per token gives 82% at 10 tokens
and 13% at 100.

**What exactly does constrained decoding change in the model?**
Nothing in the weights. At each step it sets the logits of forbidden tokens to
minus infinity (probability zero) before sampling. The allowed tokens keep
their relative odds, and temperature and top-p still apply to them.

**Why is the output valid "by construction", and what can still go wrong with the shape?**
Every prefix is kept completable, and the end token is allowed only when the
answer is complete, so any answer that ends is valid. It can still be cut off
by the token budget, leaving a valid but unfinished prefix.

**How do you decide whether a multi-character token is allowed in a given state?**
Walk its characters through the state machine one at a time from the current
state. It is allowed if the walk never gets stuck. `cat` is allowed at the
start of `cat|car|dog`; `at` is not, because the start has no "a" line.

**Why can't a finite-state machine check arbitrary JSON?**
JSON nests without limit, and closing correctly requires remembering every
open bracket in order. A machine with a fixed number of states can't count
without limit. A stack, which a pushdown automaton adds, can.

**Why can masks be precomputed for a pattern but not fully for a JSON Schema?**
A pattern's machine has a fixed number of states, so the allowed tokens for
each can be tabled once. With nesting the stack can take unboundedly many
forms. Engines precompute the tokens whose fate depends only on the current
position and check the rest against the stack at run time.

**How can constrained decoding make answers worse?**
It forces the model's choices into the allowed set even when the model
believed something else: an integer field makes it invent a number when the
honest answer was null, and token-by-token masking can commit early to a
path the model thought unlikely overall. Allowing null, or letting the model
reason before the structured part, helps.

**When is validating and retrying good enough?**
When the model is valid almost every time (95% needs about 1.05 calls on
average), when you can't change the decoder, or when the rule can't be
written as a grammar. It gets expensive fast as the valid rate falls: 1/p
calls on average.

**What does a strict schema guarantee about tool arguments, and what doesn't it?**
It guarantees the shape: field names, types, required fields, enum values.
It does not guarantee the values are true or allowed: a customer ID can be
well formed and not exist, and an amount can fit the type while breaking a
minimum. Your code still validates business rules.

## The papers behind this lesson

- **Willard and Louf, *Efficient Guided Generation for Large Language
  Models* (2023)**: https://arxiv.org/abs/2307.09702. Recast constrained
  generation as moving between the states of a finite-state machine, with the
  allowed tokens indexed per state in advance, the design behind Outlines.
- **Geng, Josifoski, Peyrard and West, *Grammar-Constrained Decoding for
  Structured NLP Tasks without Finetuning* (2023)**:
  https://arxiv.org/abs/2305.13971. Showed that constraining decoding with a
  formal grammar lets an off-the-shelf model produce complex structured
  outputs reliably, with no task-specific training.
- **Scholak, Schucher and Bahdanau, *PICARD: Parsing Incrementally for
  Constrained Auto-Regressive Decoding from Language Models* (2021)**:
  https://arxiv.org/abs/2109.05093. Rejected tokens that an incremental
  parser could not accept, making generated SQL valid as it was written.
- **Dong et al., *XGrammar: Flexible and Efficient Structured Generation
  Engine for Large Language Models* (2024)**: https://arxiv.org/abs/2411.15100.
  Made grammar-constrained decoding fast by prechecking context-independent
  tokens and checking only the context-dependent ones against the stack.
- **Park et al., *Grammar-Aligned Decoding* (2024)**:
  https://arxiv.org/abs/2405.21047. Showed that token-by-token masking
  distorts the model's distribution over valid outputs, and proposed a way
  to sample closer to the model's own conditional odds.
- **Tam et al., *Let Me Speak Freely? A Study on the Impact of Format
  Restrictions on Performance of Large Language Models* (2024)**:
  https://arxiv.org/abs/2408.02442. Measured how strict output formats can
  lower a model's reasoning performance.

## Further reading

- Russ Cox, *Regular Expression Matching Can Be Simple And Fast* (Thompson's construction, explained): https://swtch.com/~rsc/regexp/regexp1.html
- *Understanding JSON Schema*: https://json-schema.org/understanding-json-schema
- RFC 8259, *The JavaScript Object Notation (JSON) Data Interchange Format*: https://www.rfc-editor.org/rfc/rfc8259
- Claude structured outputs (JSON outputs and strict tool use): https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- llama.cpp, *GBNF Guide* (grammars for local models): https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
- Outlines, structured generation library: https://github.com/dottxt-ai/outlines
- Willard and Louf (2023): https://arxiv.org/abs/2307.09702
- Dong et al., *XGrammar* (2024): https://arxiv.org/abs/2411.15100
- Park et al., *Grammar-Aligned Decoding* (2024): https://arxiv.org/abs/2405.21047
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np

from primer._show import banner, say, table, takeaway

# ---------------------------------------------------------------------------
# 1. The toy vocabulary and the toy model
# ---------------------------------------------------------------------------

EOS = "<eos>"  # the "I'm finished" token; it adds no text

# A toy tokenizer vocabulary: every single character the examples need, plus
# a few multi-character pieces, the way a real BPE vocabulary (see
# primer.ml.tokenization) holds both letters and common chunks.
_SINGLE = list('{}[]":, -.!\'\n') + list("0123456789") + list("abcdefghijklmnopqrstuvwxyz") + list("ABDEGPSU")
_MULTI = [
    "Sure", " Here", '"age"', "age", '": ', ": ", ", ", '"}', "42", "17", "20", "null", "true", "false",
    '"name"', "name", "Ada", '"amount"', "amount", '"currency"', "currency", '"USD"', "USD", "EUR", "GBP",
]
VOCAB: list[str] = _SINGLE + _MULTI + [EOS]

AGE_PATTERN = r'\{"age": [0-9]+\}'
AGE_REFERENCE = '{"age": 42}'


def chance_all_valid(p: float, n: int) -> float:
    """P(every one of n independent steps is right) = p ** n."""
    return p**n


class ToyModel:
    """A pretend language model that has *mostly* learned to write one answer.

    It copies `reference` (starting from its first "{"), preferring longer
    tokens the way a real model prefers its tokenizer's usual pieces. Random
    noise on every logit stands in for everything the model is unsure of, and
    `chatter` is its habit of opening with "Sure". Deterministic given the rng.
    """

    def __init__(self, reference: str, vocab: list[str] = VOCAB, skill: float = 8.0, noise: float = 1.0, chatter: float = 6.0):
        self.reference, self.vocab = reference, vocab
        self.skill, self.noise, self.chatter = skill, noise, chatter

    def logits(self, text: str, rng: np.random.Generator) -> np.ndarray:
        z = rng.normal(0.0, self.noise, len(self.vocab))
        brace = text.find("{")
        # Where the model thinks it is in its answer: characters written since the first brace.
        remaining = self.reference if brace < 0 else self.reference[len(text) - brace:]
        if text == "" and "Sure" in self.vocab:
            z[self.vocab.index("Sure")] += self.chatter
        for i, token in enumerate(self.vocab):
            if token != EOS and remaining.startswith(token):
                # Longer pieces get a small bonus: a trained model has seen "42" far more than "4" then "2".
                z[i] += self.skill + 0.5 * (len(token) - 1)
        if brace >= 0 and remaining == "":
            z[self.vocab.index(EOS)] += self.skill
        return z


# ---------------------------------------------------------------------------
# 2. Masked sampling
# ---------------------------------------------------------------------------


def masked_softmax(logits: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """softmax over the allowed tokens only; forbidden tokens get exactly 0."""
    z = np.where(allowed, logits, -np.inf)
    z = z - z[allowed].max()  # the largest allowed logit becomes 0, so exp never overflows
    e = np.exp(z)  # exp(-inf) = 0 for every forbidden token
    return e / e.sum()


@dataclass
class Decoded:
    """One generated answer: its text, the tokens that spelled it, and whether it ended by choice."""

    text: str
    tokens: list[str] = field(default_factory=list)
    finished: bool = False
    removed_mass: list[float] = field(default_factory=list)


def sample_answer(model: ToyModel, rng: np.random.Generator, constraint=None, max_tokens: int = 40) -> Decoded:
    """Sample one token at a time; with a constraint, mask every token that can't continue a valid output."""
    out = Decoded("")
    everything = np.ones(len(model.vocab), dtype=bool)
    for _ in range(max_tokens):
        logits = model.logits(out.text, rng)
        allowed = everything if constraint is None else constraint.mask(out.text)
        # How much of the model's own belief the mask throws away at this step.
        out.removed_mass.append(float(1 - masked_softmax(logits, everything)[allowed].sum()))
        token = model.vocab[int(rng.choice(len(logits), p=masked_softmax(logits, allowed)))]
        if token == EOS:
            out.finished = True
            break
        out.text += token
        out.tokens.append(token)
    return out


# ---------------------------------------------------------------------------
# 3. From a pattern to a finite-state machine
# ---------------------------------------------------------------------------

ENUM_PATTERN = "cat|car|dog"
# A vocabulary small enough to check by hand: letters, a few chunks, and two
# chunks ("at", "og") that are real pieces of the words but can never start one.
ENUM_VOCAB = ["c", "a", "t", "r", "d", "o", "g", "ca", "cat", "do", "dog", "at", "og", EOS]


class _PatternParser:
    """Thompson's construction: turn a pattern into an NFA, one small fragment per piece.

    Supported: literal characters, `\\x` escapes, classes like `[0-9]`,
    grouping `( )`, alternation `|`, and the repeats `*`, `+` and `?`.
    Each fragment is a (start, end) pair of state numbers; `edges[s]` lists
    (label, target) with label None for a free "epsilon" jump.
    """

    SPECIAL = set("()[]|*+?\\")

    def __init__(self, pattern: str):
        self.pattern, self.i = pattern, 0
        self.edges: list[list[tuple[frozenset[str] | None, int]]] = []

    def new_state(self) -> int:
        self.edges.append([])
        return len(self.edges) - 1

    def peek(self) -> str | None:
        return self.pattern[self.i] if self.i < len(self.pattern) else None

    def take(self) -> str:
        self.i += 1
        return self.pattern[self.i - 1]

    def alternation(self) -> tuple[int, int]:
        branches = [self.sequence()]
        while self.peek() == "|":
            self.take()
            branches.append(self.sequence())
        if len(branches) == 1:
            return branches[0]
        start, end = self.new_state(), self.new_state()
        for b_start, b_end in branches:
            self.edges[start].append((None, b_start))  # jump freely into any branch
            self.edges[b_end].append((None, end))
        return start, end

    def sequence(self) -> tuple[int, int]:
        parts = []
        while self.peek() is not None and self.peek() not in "|)":
            parts.append(self.repeat())
        if not parts:  # the empty pattern matches the empty string
            s = self.new_state()
            return s, s
        for (_, prev_end), (next_start, _) in zip(parts, parts[1:]):
            self.edges[prev_end].append((None, next_start))  # glue the pieces end to start
        return parts[0][0], parts[-1][1]

    def repeat(self) -> tuple[int, int]:
        inner_start, inner_end = self.atom()
        while self.peek() is not None and self.peek() in "*+?":
            op = self.take()
            start, end = self.new_state(), self.new_state()
            self.edges[start].append((None, inner_start))
            self.edges[inner_end].append((None, end))
            if op in "*+":
                self.edges[inner_end].append((None, inner_start))  # loop back for another round
            if op in "*?":
                self.edges[start].append((None, end))  # or skip the piece entirely
            inner_start, inner_end = start, end
        return inner_start, inner_end

    def atom(self) -> tuple[int, int]:
        ch = self.take()
        if ch == "(":
            fragment = self.alternation()
            if self.take() != ")":
                raise ValueError(f"unclosed group in {self.pattern!r}")
            return fragment
        if ch == "[":
            chars = self.char_class()
        elif ch == "\\":
            chars = frozenset(self.take())
        elif ch in self.SPECIAL:
            raise ValueError(f"unexpected {ch!r} at position {self.i - 1} of {self.pattern!r}")
        else:
            chars = frozenset(ch)
        start, end = self.new_state(), self.new_state()
        self.edges[start].append((chars, end))
        return start, end

    def char_class(self) -> frozenset[str]:
        chars: set[str] = set()
        while self.peek() != "]":
            lo = self.take()
            if self.peek() == "-" and self.pattern[self.i + 1] != "]":
                self.take()
                hi = self.take()
                chars.update(chr(c) for c in range(ord(lo), ord(hi) + 1))  # a range like 0-9
            else:
                chars.add(lo)
        self.take()
        return frozenset(chars)


@dataclass
class DFA:
    """A deterministic finite-state machine: one current state, one move per character.

    `transitions[s]` maps a character to the next state; a missing entry is
    the dead end (the text can no longer match). States are numbered from 0,
    the start.
    """

    transitions: list[dict[str, int]]
    accepting: set[int]
    start: int = 0

    @property
    def n_states(self) -> int:
        return len(self.transitions)

    def walk(self, state: int | None, text: str) -> int | None:
        """Follow `text` one character at a time; None means the machine is stuck."""
        for ch in text:
            if state is None:
                return None
            state = self.transitions[state].get(ch)
        return state

    def accepts(self, text: str) -> bool:
        return self.walk(self.start, text) in self.accepting


def compile_pattern(pattern: str) -> DFA:
    """Pattern -> NFA (Thompson) -> DFA (subset construction)."""
    parser = _PatternParser(pattern)
    nfa_start, nfa_accept = parser.alternation()
    if parser.i != len(pattern):
        raise ValueError(f"unexpected {pattern[parser.i]!r} in {pattern!r}")
    edges = parser.edges

    def closure(states: set[int]) -> frozenset[int]:
        # Everywhere you can reach from these states by free jumps alone.
        stack, seen = list(states), set(states)
        while stack:
            for label, target in edges[stack.pop()]:
                if label is None and target not in seen:
                    seen.add(target)
                    stack.append(target)
        return frozenset(seen)

    # Each DFA state is the *set* of NFA states you could be in at once.
    first = closure({nfa_start})
    ids, order, transitions = {first: 0}, [first], []
    for current in order:  # `order` grows as new sets are found: a breadth-first walk
        moves: dict[str, int] = {}
        chars = sorted({c for s in current for label, _ in edges[s] if label for c in label})
        for ch in chars:
            nxt = closure({t for s in current for label, t in edges[s] if label and ch in label})
            if nxt not in ids:
                ids[nxt] = len(order)
                order.append(nxt)
            moves[ch] = ids[nxt]
        transitions.append(moves)
    accepting = {ids[s] for s in order if nfa_accept in s}
    return DFA(transitions, accepting)


def allowed_tokens(dfa: DFA, state: int | None, vocab: list[str]) -> list[str]:
    """Every token the machine can consume *in full* from `state`; the end token only where the text is complete."""
    return [t for t in vocab if (state in dfa.accepting if t == EOS else dfa.walk(state, t) is not None)]


def mask_table(dfa: DFA, vocab: list[str]) -> np.ndarray:
    """Precompute the answer for every state once: row s is the allowed-token mask at state s."""
    table = np.zeros((dfa.n_states, len(vocab)), dtype=bool)
    for s in range(dfa.n_states):
        for j, t in enumerate(vocab):
            table[s, j] = s in dfa.accepting if t == EOS else dfa.walk(s, t) is not None
    return table


class RegexConstraint:
    """Constrain decoding to a pattern: compile once, build the table once, then each step is a lookup."""

    def __init__(self, pattern: str, vocab: list[str]):
        self.dfa = compile_pattern(pattern)
        self.table = mask_table(self.dfa, vocab)

    def mask(self, text: str) -> np.ndarray:
        state = self.dfa.walk(self.dfa.start, text)
        if state is None:
            raise ValueError(f"{text!r} already breaks the pattern")
        return self.table[state]


# ---------------------------------------------------------------------------
# 4. JSON Schema: nesting needs a stack
# ---------------------------------------------------------------------------

PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "pets": {"type": "array", "items": {"type": "string", "enum": ["cat", "dog"]}},
    },
    "required": ["name", "age"],
    # Strict: no fields beyond these. Constrained decoding needs to know every key it may spell.
    "additionalProperties": False,
}
PERSON_REFERENCE = '{"name": "Ada", "age": 36, "pets": ["cat"]}'

# The leaves of JSON are regular, so the machines from section 3 read them.
# (JSON also allows an exponent, as in 1e5; this subset leaves it out.)
_NUMBER = compile_pattern(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?")
_INTEGER = compile_pattern(r"-?(0|[1-9][0-9]*)")
_LITERALS = {"t": "true", "f": "false", "n": "null"}


@dataclass(frozen=True)
class Frame:
    """One entry on the stack: a value that has been opened but not yet finished.

    kind is "root", "object", "array", "string", "number" or "literal".
    phase says what the frame expects next; text holds the characters of a
    key, string, number or literal read so far; seen holds an object's keys.
    """

    kind: str
    schema: dict
    phase: str = ""
    text: str = ""
    seen: frozenset = frozenset()


Stack = tuple  # a tuple of Frames, innermost last; immutable, so every step returns a new one


def _types(schema: dict) -> set[str]:
    t = schema.get("type")
    if t is None:  # the empty schema {} allows any JSON value
        return {"string"} if "enum" in schema else {"object", "array", "string", "number", "integer", "boolean", "null"}
    return set(t) if isinstance(t, list) else {t}


class SchemaChecker:
    """A pushdown automaton for a JSON Schema subset: a state machine plus a stack.

    Covers objects (properties, required, additionalProperties), arrays
    (items), strings (with optional enum), integers, numbers, booleans and
    null, with at most one space after ":" and ",". Strings have no escapes.
    Every step either refuses the character or leaves a stack from which the
    value can still be finished, so "not dead" always means "completable".
    """

    def __init__(self, schema: dict):
        self.schema = schema

    def start(self) -> Stack:
        return (Frame("root", self.schema, "value"),)

    # --- helpers ----------------------------------------------------------

    @staticmethod
    def _strict(schema: dict) -> bool:
        return schema.get("additionalProperties") is False

    def _unseen_keys(self, frame: Frame) -> list[str]:
        return [k for k in frame.schema.get("properties", {}) if k not in frame.seen]

    def _may_add_key(self, frame: Frame) -> bool:
        return not self._strict(frame.schema) or bool(self._unseen_keys(frame))

    def _may_close(self, frame: Frame) -> bool:
        return set(frame.schema.get("required", [])) <= frame.seen

    def _open_value(self, stack: Stack, schema: dict, ch: str) -> Stack | None:
        """The first character of a value decides which kind of frame to push."""
        types = _types(schema)
        if ch == "{" and "object" in types:
            return stack + (Frame("object", schema, "open"),)
        if ch == "[" and "array" in types:
            return stack + (Frame("array", schema, "open"),)
        if ch == '"' and "string" in types:
            return stack + (Frame("string", schema),)
        if (ch == "-" or ch.isdigit()) and types & {"number", "integer"}:
            return stack + (Frame("number", schema, text=ch),)
        if ch in _LITERALS and ("boolean" if ch in "tf" else "null") in types:
            return stack + (Frame("literal", schema, _LITERALS[ch], ch),)
        return None

    def _close(self, stack: Stack) -> Stack:
        """The innermost value is finished: pop it and tell its parent."""
        stack = stack[:-1]
        parent = stack[-1]
        if parent.kind == "object":
            return stack[:-1] + (replace(parent, phase="after_value", seen=parent.seen | {parent.text}),)
        return stack[:-1] + (replace(parent, phase="done" if parent.kind == "root" else "after_value"),)

    def _child_schema(self, frame: Frame) -> dict:
        if frame.kind == "array":
            return frame.schema.get("items", {})
        extra = frame.schema.get("additionalProperties", {})
        return frame.schema.get("properties", {}).get(frame.text, extra if isinstance(extra, dict) else {})

    # --- one character ----------------------------------------------------

    def step(self, stack: Stack | None, ch: str) -> Stack | None:
        """Read one character. Returns the new stack, or None if no valid value starts this way."""
        if stack is None:
            return None
        top = stack[-1]
        rest = stack[:-1]
        k, phase = top.kind, top.phase

        if k == "root":
            return self._open_value(stack, top.schema, ch) if phase == "value" else None

        if k == "string":
            if ch == '"':
                enum = top.schema.get("enum")
                return self._close(stack) if enum is None or top.text in enum else None
            if ch == "\\" or ord(ch) < 32:  # no escapes in this subset; raw control characters are never legal JSON
                return None
            text = top.text + ch
            enum = top.schema.get("enum")
            if enum is not None and not any(e.startswith(text) for e in enum):
                return None
            return rest + (replace(top, text=text),)

        if k == "number":
            machine = _INTEGER if _types(top.schema) == {"integer"} else _NUMBER
            if machine.walk(machine.start, top.text + ch) is not None:
                return rest + (replace(top, text=top.text + ch),)
            # The number can't grow: if it is finished, the character belongs to whatever comes after it.
            if machine.accepts(top.text):
                return self.step(self._close(stack), ch)
            return None

        if k == "literal":
            text = top.text + ch
            if not top.phase.startswith(text):  # phase holds the target word here
                return None
            return self._close(stack) if text == top.phase else rest + (replace(top, text=text),)

        if k == "object":
            if phase in ("open", "after_comma", "need_key") and ch == '"' and self._may_add_key(top):
                return rest + (replace(top, phase="key", text=""),)
            if phase == "after_comma" and ch == " ":
                return rest + (replace(top, phase="need_key"),)
            if phase in ("open", "after_value") and ch == "}" and self._may_close(top):
                return self._close(stack)
            if phase == "after_value" and ch == "," and self._may_add_key(top):
                return rest + (replace(top, phase="after_comma"),)
            if phase == "key":
                if ch == '"':
                    ok = not self._strict(top.schema) or top.text in self._unseen_keys(top)
                    return rest + (replace(top, phase="colon"),) if ok else None
                if ch == "\\" or ord(ch) < 32:
                    return None
                text = top.text + ch
                if self._strict(top.schema) and not any(key.startswith(text) for key in self._unseen_keys(top)):
                    return None
                return rest + (replace(top, text=text),)
            if phase == "colon":
                return rest + (replace(top, phase="space_or_value"),) if ch == ":" else None
            if phase == "space_or_value" and ch == " ":
                return rest + (replace(top, phase="value"),)
            if phase in ("space_or_value", "value"):
                return self._open_value(rest + (replace(top, phase="value"),), self._child_schema(top), ch)
            return None

        if k == "array":
            if phase in ("open", "after_value") and ch == "]":
                return self._close(stack)
            if phase == "after_value":
                return rest + (replace(top, phase="after_comma"),) if ch == "," else None
            if phase == "after_comma" and ch == " ":
                return rest + (replace(top, phase="need_value"),)
            if phase in ("open", "after_comma", "need_value"):
                return self._open_value(rest + (replace(top, phase="value"),), self._child_schema(top), ch)
            return None

        raise AssertionError(f"unknown frame kind {k!r}")

    # --- whole prefixes ---------------------------------------------------

    def feed(self, stack: Stack | None, text: str) -> Stack | None:
        for ch in text:
            stack = self.step(stack, ch)
            if stack is None:
                return None
        return stack

    def can_end(self, stack: Stack | None) -> bool:
        """May the output stop here? Yes once the top-level value is finished."""
        if stack is None:
            return False
        if stack[-1].kind == "root":
            return stack[-1].phase == "done"
        # A top-level number has no closing character: "42" is finished when the output stops.
        top = stack[-1]
        machine = _INTEGER if _types(top.schema) == {"integer"} else _NUMBER
        return len(stack) == 2 and top.kind == "number" and machine.accepts(top.text)

    def status(self, prefix: str) -> str:
        """"dead" (no continuation can fix it), "complete" (may stop here) or "open" (valid so far)."""
        stack = self.feed(self.start(), prefix)
        return "dead" if stack is None else "complete" if self.can_end(stack) else "open"

    def open_containers(self, prefix: str) -> list[str]:
        """The objects and arrays opened but not yet closed, outermost first: the visible part of the stack."""
        stack = self.feed(self.start(), prefix)
        if stack is None:
            raise ValueError(f"{prefix!r} can't be completed")
        return [f.kind for f in stack if f.kind in ("object", "array")]


class SchemaConstraint:
    """Constrain decoding to a JSON Schema: at each step, try every token against the stack."""

    def __init__(self, schema: dict, vocab: list[str]):
        self.checker, self.vocab = SchemaChecker(schema), vocab

    def mask(self, text: str) -> np.ndarray:
        stack = self.checker.feed(self.checker.start(), text)
        # Unlike a pattern's table, this can't be precomputed for every state: the stack can grow without limit.
        return np.array([self.checker.can_end(stack) if t == EOS else self.checker.feed(stack, t) is not None for t in self.vocab])


# ---------------------------------------------------------------------------
# 5. Costs and pitfalls
# ---------------------------------------------------------------------------

NULLABLE_AGE_SCHEMA = {
    "type": "object",
    "properties": {"age": {"type": ["integer", "null"]}},  # "we don't know" is now a valid answer
    "required": ["age"],
    "additionalProperties": False,
}


def forced_choice(probs: dict[str, float], allowed: set[str]) -> tuple[dict[str, float], float]:
    """Renormalise a model's next-token belief over the allowed tokens; also return the share thrown away."""
    kept_mass = sum(p for t, p in probs.items() if t in allowed)
    return {t: p / kept_mass for t, p in probs.items() if t in allowed}, 1 - kept_mass


# A two-step model small enough to enumerate. Valid answers must end in "y".
TWO_STEP_FIRST = {"A": 0.9, "B": 0.1}
TWO_STEP_SECOND = {"A": {"x": 0.99, "y": 0.01}, "B": {"x": 0.1, "y": 0.9}}


def distortion_example() -> dict[str, dict[str, float]]:
    """Token-by-token masking versus the model's own odds among valid answers.

    masked: at step 1 both A and B can still end in "y", so nothing is
    masked and A wins 90% of the time; at step 2 the mask forces "y".
    conditional: P(answer) / P(any valid answer), the distribution you would
    get by sampling whole answers and throwing away the invalid ones.
    """
    valid = {first + "y": TWO_STEP_FIRST[first] * TWO_STEP_SECOND[first]["y"] for first in TWO_STEP_FIRST}
    total = sum(valid.values())
    masked = {first + "y": TWO_STEP_FIRST[first] * 1.0 for first in TWO_STEP_FIRST}  # step 2 renormalises to y = 1
    return {"masked": masked, "conditional": {a: p / total for a, p in valid.items()}}


def tokenizations(text: str, vocab: list[str]) -> list[list[str]]:
    """Every way to spell `text` as a sequence of vocabulary tokens."""
    pieces = [t for t in vocab if t != EOS]
    ways: dict[int, list[list[str]]] = {len(text): [[]]}
    for i in range(len(text) - 1, -1, -1):  # from the end backwards, so each suffix is solved once
        ways[i] = [[t] + rest for t in pieces if text.startswith(t, i) for rest in ways[i + len(t)]]
    return ways[0]


def mask_cost(n_steps: int, vocab_size: int, avg_len: float, n_states: int) -> dict[str, float]:
    """Character steps spent building masks: walk every token at every step, or once per state up front."""
    naive = n_steps * vocab_size * avg_len
    table = n_states * vocab_size * avg_len + n_steps * vocab_size  # build once, then read one row per step
    return {"naive": naive, "table": table}


def expected_attempts(p: float) -> float:
    """Average number of tries until the first valid answer, when each try is valid with chance p."""
    return 1 / p


def chance_still_failing(p: float, k: int) -> float:
    """Chance that k independent tries are all invalid."""
    return (1 - p) ** k


NO_AGE_REFERENCE = '{"name": "Ada"}'  # a model that "forgot" the required age


def _age_ok(text: str) -> bool:
    import re

    return re.fullmatch(AGE_PATTERN, text) is not None  # Python's own regex engine is the referee


def _person_ok(text: str) -> bool:
    import json

    from primer.agents.tools import validate  # the tools lesson's validator is the referee, not our checker

    try:
        return validate(json.loads(text), PERSON_SCHEMA) == []
    except ValueError:
        return False


def validity_experiment(n: int = 200, seed: int = 0) -> list[dict]:
    """Sample n answers per task, with and without constraints, and sort the failures by kind."""
    tasks = [
        ("age pattern", AGE_REFERENCE, _age_ok, lambda: RegexConstraint(AGE_PATTERN, VOCAB)),
        ("person schema", PERSON_REFERENCE, _person_ok, lambda: SchemaConstraint(PERSON_SCHEMA, VOCAB)),
    ]
    rows = []
    for task, reference, ok, make_constraint in tasks:
        for constrained in (False, True):
            rng = np.random.default_rng(seed)
            model, constraint = ToyModel(reference), make_constraint() if constrained else None
            counts = {"valid": 0, "chatter": 0, "cut_off": 0, "broken": 0}
            for _ in range(n):
                answer = sample_answer(model, rng, constraint)
                if ok(answer.text) and answer.finished:
                    counts["valid"] += 1
                elif not answer.text.startswith("{"):
                    counts["chatter"] += 1  # words before the JSON, such as "Sure"
                elif not answer.finished:
                    counts["cut_off"] += 1  # ran out of token budget mid-value
                else:
                    counts["broken"] += 1  # finished, but a wrong character somewhere inside
            rows.append({"task": task, "constrained": constrained, **counts})
    return rows


# ---------------------------------------------------------------------------
# 6. Figures (rendered into the HTML docs by `make figures`)
# ---------------------------------------------------------------------------


def _state_labels(dfa: DFA, reference: str) -> dict[int, str]:
    """Name each state by the shortest prefix of `reference` that reaches it."""
    labels: dict[int, str] = {}
    for i in range(len(reference) + 1):
        labels.setdefault(dfa.walk(dfa.start, reference[:i]), reference[:i] or "(start)")
    return labels


def figures() -> dict:
    """Plot this lesson's data. matplotlib is imported here, and only here,
    so the lesson itself needs nothing beyond NumPy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, RED, GREEN, AMBER, MUTED = "#2563eb", "#dc2626", "#059669", "#d97706", "#9ca3af"
    figs = {}

    # --- 1. p ** n: longer answers break more often -------------------------
    fig, ax = plt.subplots(figsize=(6, 3.4))
    n = np.arange(1, 201)
    for p, color in ((0.999, GREEN), (0.99, BLUE), (0.98, AMBER), (0.95, RED)):
        ax.plot(n, chance_all_valid(p, n), color=color, label=f"{p:.1%} per token")
    ax.plot([10], [chance_all_valid(0.98, 10)], "o", color="#111827")
    ax.annotate("10 tokens at 98%: 0.82", (10, 0.817), (60, 0.72), arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_xlabel("answer length n (tokens)")
    ax.set_ylabel("share of answers valid, pⁿ")
    ax.set_ylim(0, 1.05)
    ax.set_title("Every token is a chance to break the format")
    ax.legend(frameon=False, loc="lower left")
    figs["validity_vs_length"] = fig

    # --- 2. One step of masking --------------------------------------------
    model, rng = ToyModel(AGE_REFERENCE), np.random.default_rng(0)
    logits = model.logits("", rng)
    before = masked_softmax(logits, np.ones(len(VOCAB), dtype=bool))
    after = masked_softmax(logits, RegexConstraint(AGE_PATTERN, VOCAB).mask(""))
    top = np.argsort(before)[::-1][:4]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(len(top))
    ax.bar(x - 0.2, before[top], 0.4, color=MUTED, label="what the model wanted")
    ax.bar(x + 0.2, after[top], 0.4, color=BLUE, label="after the mask")
    for xi, (b, a) in enumerate(zip(before[top], after[top])):
        ax.text(xi - 0.2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)
        ax.text(xi + 0.2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8, color=BLUE)
    ax.set_xticks(x, [repr(VOCAB[i]) for i in top])
    ax.set_ylabel("probability")
    ax.set_ylim(0, 1.15)
    ax.set_title('First token of {"age": 42}: the mask removes the chatter')
    ax.legend(frameon=False)
    figs["masked_step"] = fig

    # --- 3. Validity with and without constraints ----------------------------
    rows = validity_experiment(n=200)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    names = [f"{r['task']}\n{'constrained' if r['constrained'] else 'unconstrained'}" for r in rows]
    bottom = np.zeros(len(rows))
    for key, color, label in (("valid", GREEN, "valid"), ("chatter", AMBER, "chatter before the JSON"),
                              ("broken", RED, "broken inside"), ("cut_off", MUTED, "cut off by the token budget")):
        share = np.array([r[key] / 200 for r in rows])
        ax.bar(names, share, bottom=bottom, color=color, label=label)
        bottom += share
    for i, r in enumerate(rows):
        ax.text(i, r["valid"] / 200 / 2, f"{r['valid'] / 200:.1%}", ha="center", color="white", fontweight="bold")
    ax.set_ylabel("share of 200 answers")
    ax.set_title("Constrained decoding: every finished answer is valid")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    figs["validity_bars"] = fig

    # --- 4. The precomputed mask table for the age pattern -------------------
    dfa = compile_pattern(AGE_PATTERN)
    table = mask_table(dfa, VOCAB)
    shown = [j for j in range(len(VOCAB)) if table[:, j].any()] + [VOCAB.index("Sure"), VOCAB.index("x")]
    labels = _state_labels(dfa, AGE_REFERENCE)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.imshow(table[:, shown], cmap="Blues", vmin=0, vmax=1.3, aspect="auto")
    ax.set_xticks(range(len(shown)), [repr(VOCAB[j]) if VOCAB[j] != EOS else "end" for j in shown], rotation=90, fontsize=8)
    ax.set_yticks(range(dfa.n_states), [labels[s] for s in range(dfa.n_states)], fontsize=8)
    ax.set_xlabel("token (every token allowed somewhere, plus two that never are)")
    ax.set_ylabel("state, named by the text that reaches it")
    ax.set_title("The age pattern's mask table, computed once before generation")
    ax.grid(False)
    figs["mask_table"] = fig

    # --- 5. Distortion: masked decoding vs the model's own odds --------------
    dist = distortion_example()
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(2)
    answers = ["Ay", "By"]
    cond = [dist["conditional"][a] for a in answers]
    masked = [dist["masked"][a] for a in answers]
    ax.bar(x - 0.2, cond, 0.4, color=MUTED, label="model's own odds among valid answers")
    ax.bar(x + 0.2, masked, 0.4, color=BLUE, label="what token-by-token masking produces")
    for xi, (c, m) in enumerate(zip(cond, masked)):
        ax.text(xi - 0.2, c + 0.02, f"{c:.2f}", ha="center")
        ax.text(xi + 0.2, m + 0.02, f"{m:.2f}", ha="center")
    ax.set_xticks(x, answers)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("probability")
    ax.set_title("Valid, but not what the model preferred")
    ax.legend(frameon=False, fontsize=8)
    figs["distortion"] = fig

    # --- 6. Mask cost: every step vs a table built once ----------------------
    answers_made = np.arange(0, 11)
    per_answer = mask_cost(n_steps=200, vocab_size=128_000, avg_len=4, n_states=50)
    build = 50 * 128_000 * 4
    naive = answers_made * per_answer["naive"]
    tabled = build + answers_made * (per_answer["table"] - build)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(answers_made, naive / 1e6, "o-", color=RED, label="check every token at every step")
    ax.plot(answers_made, tabled / 1e6, "o-", color=BLUE, label="build a table once, read a row per step")
    ax.set_xlabel("answers generated with one schema")
    ax.set_ylabel("work (millions of steps)")
    ax.set_title("|V| = 128,000, 200 tokens per answer, 50 states")
    ax.legend(frameon=False)
    figs["mask_cost"] = fig

    # --- 7. Retrying: expected attempts --------------------------------------
    p = np.linspace(0.05, 1.0, 200)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(p, expected_attempts(p), color=BLUE)
    for value, label, dx, dy in ((0.95, "95% valid", -0.12, 1.4), (0.665, "toy age answer", -0.3, 3.2), (0.18, "toy person answer", 0.04, 1.6)):
        ax.plot([value], [expected_attempts(value)], "o", color=RED)
        ax.annotate(f"{label}: {expected_attempts(value):.2f}", (value, expected_attempts(value)),
                    (value + dx, expected_attempts(value) + dy), fontsize=8, arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_xlabel("chance a single try is valid, p")
    ax.set_ylabel("average tries until valid, 1/p")
    ax.set_ylim(0, 12)
    ax.set_title("Retrying is cheap only when the model is usually right")
    figs["retry"] = fig

    return figs


# ---------------------------------------------------------------------------
# 7. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. Asking nicely: the toy model, unconstrained")
    rng = np.random.default_rng(0)
    model = ToyModel(AGE_REFERENCE)
    answers = [sample_answer(model, rng).text for _ in range(8)]
    say(f"The toy model has mostly learned to write {AGE_REFERENCE}. Eight free answers:")
    for a in answers:
        print(f"  {'ok ' if _age_ok(a) else 'BAD'}  {a!r}")
    print()
    say(f"Per-token accuracy compounds: 0.98 ** 10 = {chance_all_valid(0.98, 10):.3f}, 0.98 ** 100 = {chance_all_valid(0.98, 100):.3f}.")
    takeaway("One bad token breaks the whole answer, so long outputs fail far more often than short ones.")

    banner("2. Mask, then sample")
    z, m = np.array([2.0, 1.0, 0.5, 0.0]), np.array([False, False, True, True])
    table(["token", "logit", "allowed", "before", "after"],
          [(t, zi, bool(mi), b, a) for t, zi, mi, b, a in
           zip(["Sure", " Here", "{", "["], z, m, masked_softmax(z, np.ones(4, dtype=bool)), masked_softmax(z, m))],
          floatfmt=".3f")
    rows = validity_experiment(n=200)
    table(["task", "constrained", "valid", "chatter", "broken", "cut off"],
          [(r["task"], "yes" if r["constrained"] else "no", r["valid"], r["chatter"], r["broken"], r["cut_off"]) for r in rows])
    takeaway("Forbidden tokens get probability zero, so every answer that finishes is valid by construction.")

    banner("3. A pattern becomes a state machine")
    dfa = compile_pattern(ENUM_PATTERN)
    say(f"'{ENUM_PATTERN}' compiles to {dfa.n_states} states. Allowed tokens, walking every character:")
    for prefix in ("", "ca", "dog"):
        print(f"  after {prefix!r:6}: {allowed_tokens(dfa, dfa.walk(dfa.start, prefix), ENUM_VOCAB)}")
    print()
    age = compile_pattern(AGE_PATTERN)
    say(f"The age pattern {AGE_PATTERN} has {age.n_states} states; its whole mask table is "
        f"{age.n_states} x {len(VOCAB)} entries, built once before the first token.")
    takeaway("A token is allowed if the machine can swallow all of its characters; precompute that per state.")

    banner("4. JSON Schema needs a stack")
    anything = SchemaChecker({})
    say(f"Stack after '{{\"a\": [1, {{\"b\": ': {anything.open_containers('{\"a\": [1, {\"b\": ')}")
    person = SchemaChecker(PERSON_SCHEMA)
    table(["prefix", "status"], [(p, person.status(p)) for p in
          ('{"na', '{"nx', '{"name": "Ada"}', '{"age": 3.', '{"age": 01', '{"name": "Ada", "age": 36}')])
    rng = np.random.default_rng(0)
    constraint = SchemaConstraint(PERSON_SCHEMA, VOCAB)
    for _ in range(3):
        print("  constrained:", sample_answer(ToyModel(PERSON_REFERENCE), rng, constraint).text)
    print()
    takeaway("Nesting needs memory without limit; the schema decides which keys, types and closers are legal.")

    banner("5. Costs and pitfalls")
    kept, removed = forced_choice({"null": 0.80, "3": 0.12, "5": 0.08}, {"3", "5"})
    say(f"Forcing an integer when the model wanted null throws away {removed:.0%} of its belief; "
        f"it now answers 3 with {kept['3']:.0%} and 5 with {kept['5']:.0%}: an invented age.")
    d = distortion_example()
    say(f"Two-step model: its own odds for Ay among valid answers are {d['conditional']['Ay']:.3f}, "
        f"but masking one token at a time produces Ay {d['masked']['Ay']:.0%} of the time.")
    say(f"'42' can be spelled {tokenizations('42', VOCAB)}; the whole age answer has "
        f"{len(tokenizations(AGE_REFERENCE, VOCAB))} spellings, and the mask allows them all.")
    c = mask_cost(n_steps=200, vocab_size=128_000, avg_len=4, n_states=50)
    say(f"Mask work for one 200-token answer: {c['naive']:,.0f} steps checking every token, "
        f"{c['table']:,.0f} with a table (including building it).")
    say(f"Retrying instead: {expected_attempts(0.665):.2f} tries on average at 66.5% valid, "
        f"{expected_attempts(0.18):.2f} at 18%.")
    takeaway("The mask guarantees shape, not good answers: design schemas that allow the honest answer.")

    banner("6. JSON mode versus a strict schema")
    import json

    from primer.agents.tools import PAYMENT_SCHEMA, validate

    forgetful = ToyModel(NO_AGE_REFERENCE)
    say(f"A model whose answer leaves out the required age, {NO_AGE_REFERENCE}, asked 50 times:")
    for name, schema in (("JSON mode, schema {}", {}), ("strict person schema", PERSON_SCHEMA)):
        rng = np.random.default_rng(0)
        constraint = SchemaConstraint(schema, VOCAB)
        answers = [sample_answer(forgetful, rng, constraint) for _ in range(50)]
        finished = Counter(a.text for a in answers if a.finished)
        print(f"  {name}: {sum(finished.values())} finished, {50 - sum(finished.values())} cut off. Most common:")
        for text, count in finished.most_common(3):
            print(f"    {count:2d} x {text}")
    print()
    say("JSON mode parses but misses the age; the schema forces an age, which the model has to invent.")
    arguments = '{"amount": 0, "currency": "USD"}'
    say(f"Tool arguments {arguments}: the schema checker says '{SchemaChecker(PAYMENT_SCHEMA).status(arguments)}', "
        f"yet validation says {validate(json.loads(arguments), PAYMENT_SCHEMA)}.")
    takeaway("JSON mode guarantees parseable; a strict schema guarantees the shape; only your code checks the truth.")


if __name__ == "__main__":
    demo()
