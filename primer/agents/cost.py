r"""
# Cost and latency: paying less per successful task without getting worse

Run: `python -m primer.agents.cost`

This lesson builds on tokens and pricing from `primer.ml.inference`, on the
context window from `primer.agents.context` and on the release gate from
`primer.agents.evals`.

## Level 1: The practitioner's guide

**In one sentence.** Cost engineering for a model-backed system means paying
less per *successful* task, not per call, by taking the levers that can't
hurt quality first and checking the ones that can against an eval.

**When you need it.** When the bill grows faster than the value, when a
latency budget is missed, or before the first big customer arrives and the
price per task stops being a rounding error. The tell: you know your spend
per month but not your cost per successful task, or you know cost per call
but have never counted retries and human clean-up. The lesson's worked
example shows why that number, and only that number, decides things: a
small model at \$0.002 a call with a 60% success rate beats a large one at
\$0.010 and 95% if failures can simply be retried (\$0.0033 against
\$0.0105 per success), and loses by 7x if a person has to fix each failure
(\$0.802 against \$0.110 per task, at an illustrative \$2 per fix). You don't
need any of this for a prototype with ten users a day, and you shouldn't
touch the levers that change behaviour (routing, semantic caching) until
you have an eval to watch them with. All prices in this lesson are
illustrative constants chosen for round arithmetic (a large model at \$5
per million input tokens and \$25 per million output, a small one at \$1
and \$5); check a provider's current price list for real numbers.

**Your options.** Eight levers, from the ones that can't hurt quality to the
ones that can:

| Option | What it does | What it guarantees | What it costs | Where it lives |
|---|---|---|---|---|
| Prompt caching | Puts the stable part of the prompt first so the provider reuses its work | No behaviour change; the worked 10,000-token call falls from \$0.0625 to \$0.0265 with 8,000 tokens cached | A small write premium the first time; reordering the prompt | The provider's API |
| Trim tokens | Compress tool results, send fewer retrieved chunks, ask for concise output | No change if you only cut what the next step never read; output is the expensive direction | Engineering, and care not to cut what was needed | Your prompt and code |
| Parallel tool calls | Run independent tool calls at the same time | Wall-clock time falls to the slowest call (300 ms to 100 ms for three lookups); tokens unchanged | Concurrency in the agent loop | Your agent loop |
| Exact response cache | The same question, ignoring case and spaces, returns the stored answer | Never wrong until the facts change | A time-to-live to manage | Your code |
| Batch API | Non-interactive work submitted in bulk at about half price | Half price, results within hours | Waiting; only for work nobody is waiting on | The provider's API |
| Budgets and alerts | Caps steps and tokens per task and spend per tenant; alerts on a sudden jump | A confused agent stops instead of looping all night | Choosing the limits; a false alarm now and then | Your code |
| Routing | Sends easy tasks (classify, extract, format) to a small model and hard ones to a large one | 53% saved on this lesson's ten-task workload, if the router is right | The first lever that can lose quality; needs an eval before and after | Your code, in front of every request |
| Semantic cache | Answers a *similar* question from memory, using embeddings to judge similarity | The cheapest possible hit | Confidently wrong answers: on this lesson's pairs, 40% of different-intent questions are served from cache at a 0.95 threshold | Your code, plus an embedding model |

A ninth, for later: train or distil a small model on the large model's
answers so it can take more of the routed traffic (Hinton et al., 2015).

**How to choose.** Measure first, then go down the table in order.

- No measurement yet: instrument cost per task by step, by model, input
  against output and cached against uncached, and put an eval set in place
  to hold quality fixed.
- A long, stable system prompt or tool list: prompt caching, today. It is
  the only lever that halves a bill without changing a single answer.
- Big tool results or many retrieved chunks: trim. Compress to the fields
  the next step needs and send the top few reranked chunks.
- Several independent lookups per turn: run them concurrently, and stream
  the answer so users see progress.
- Anything nobody is waiting on (nightly evals, backfills, bulk
  classification): batch it.
- Most traffic is simple: route, with the eval watching. A router that sends
  hard tasks to the small model saves money and quietly loses quality.
- Narrow, curated, FAQ-style traffic and nothing else: a semantic cache
  with guards (identifiers, numbers and negation must match), a
  time-to-live and a measured wrong-hit rate. On everyday questions no
  threshold makes it safe.
- Whatever you pick, judge it by cost per successful task, with retries and
  human clean-up counted in.

**What it costs.** The levers stack. On this lesson's ten-task workload the
baseline is \$0.671; caching takes it to \$0.311 (2.2x), trimming tool output
to \$0.186 (3.6x), concise output to \$0.171 (3.9x), routing to \$0.086 (7.8x)
and batching the non-interactive share to \$0.080 (8.4x). The first three
are free wins that don't touch quality; routing is the big one and the
first that can. Caching is not free on the first request: Anthropic's
prompt caching documentation, for one, prices a cache write at 1.25 times
the base input price for a five-minute cache (2 times for an hour), reads at
a tenth of it or less, and only caches prompts above a per-model minimum of
a few hundred to a few thousand tokens. Batching costs time: the same
provider quotes a 50% discount with most batches finishing within an hour.
Parallel calls and streaming cost no tokens at all; they buy time and
perceived speed. Budgets cost the occasional false alarm: this lesson's
anomaly rule flags a task using more than the mean plus three standard
deviations of recent usage, so a 10,000-token task against a history around
1,000 trips it at once.

**What breaks.**

- **Cost per call replacing cost per success.** The cheap model looks
  cheaper until failures are priced. Count retries and fixes.
- **Routing that loses quality quietly.** Nothing errors; the answers just
  get worse. Gate the router with the eval, and re-check when the small
  model changes.
- **The semantic cache that answers the wrong question.** "How many sick
  days do I get?" scores 0.97 against the vacation question and gets the
  vacation answer. Near-misses score higher than real paraphrases, so no
  threshold separates them.
- **Stale cache hits.** The policy changed; the cache didn't. Give every
  entry a time-to-live.
- **A cache that never hits.** Something volatile (a timestamp, the user's
  name) sits at the top of the prompt, so the prefix differs every call.
  Stable content first.
- **Trimming what was needed.** A tool result cut to 500 tokens loses the
  field the next step reads. Compress by field, not by length alone.
- **The runaway loop.** An agent calls the same tool with the same
  arguments all night. Cap steps and tokens per task; alert on tokens per
  task jumping.
- **One tenant's surprise invoice.** A shared platform with no per-customer
  cap turns one customer's bug into everyone's bill.

**In the wild.** Prompt caching and batch processing are standard features
of hosted APIs; Anthropic's documentation for both is linked in Further
reading and gives the multipliers quoted above. FrugalGPT (Chen, Zaharia and
Zou, 2023) named the three families, prompt adaptation, model approximation
and cascades that try a cheap model first and escalate, and reported
matching the best single model with up to 98% less cost on its benchmarks.
Distillation (Hinton, Vinyals and Dean, 2015) is how the small model in the
routing table gets good enough to take more traffic. Tooling: LiteLLM is an
open-source gateway that puts one interface in front of many providers and
adds routing with fallbacks, per-key and per-team budgets, spend tracking
and caching; GPTCache is an open-source semantic cache built on embeddings
and a vector store with a pluggable similarity evaluator, the design this
lesson's `SemanticCache` reproduces in miniature, guards and all. Parallel
tool use is part of the tool-calling protocol of the major APIs: the model
asks for several tools in one turn and you return all the results in one
message.

**Go deeper.** Level 2 prices a request symbol by symbol, builds the router,
both caches and the parallel loop in plain Python, derives the anomaly
alert from a mean and a standard deviation, works the unit economics with
every number, and stacks the levers to draw the 8.4x figure. If you only
needed to know which lever to pull first, you are done.

## Level 2: How it works, from scratch

A language model's bill is the sum of its calls, and each call is priced by
its tokens. Lowering the bill takes three steps: work out what one call
costs, work out what a finished task costs once failures are counted, and
then pull the levers that bring it down, starting with the ones that
cannot change the answers.

All prices here are illustrative, chosen for round arithmetic: a "large"
model at \$5 per million input tokens and \$25 per million output tokens,
and a "small" one at \$1 and \$5. They show the shape of the trade-offs.
For real numbers, check your provider's current price list.

## How a request is priced

A model charges per **token** (a word piece, about 4 characters of
English), at one rate for the tokens you send (**input**) and a higher rate
for the tokens it writes (**output**). Output costs more because the model
writes it one token at a time (see `primer.ml.inference`).

Take a large-model call with 10,000 input tokens and 500 output tokens.
Input costs 10,000 × \$5 / 1,000,000 = \$0.05 and output costs
500 × \$25 / 1,000,000 = \$0.0125, so the call costs \$0.0625. Now suppose
8,000 of those input tokens are a stable prefix, the same system prompt and
tool definitions sent on every call. The **prompt cache** reuses the work
already done on a prefix it has seen, and bills it here at 10% of the input
price. Input becomes 8,000 × \$0.50/M + 2,000 × \$5/M = \$0.014, and the
call costs \$0.0265, 58% less.

$$
\text{cost} = \frac{c \cdot \rho\, p_{\text{in}} + (n_{\text{in}} - c)\, p_{\text{in}} + n_{\text{out}}\, p_{\text{out}}}{10^6}
$$

**Symbols**

| Symbol | Meaning here | Worked example |
|---|---|---|
| $n_{\text{in}}$ | input tokens sent | 10,000 |
| $c$ | input tokens served from the prompt cache | 8,000 |
| $n_{\text{out}}$ | output tokens generated | 500 |
| $p_{\text{in}}, p_{\text{out}}$ | price per million input / output tokens | \$5, \$25 |
| $\rho$ | cached-read price as a fraction of normal input price (Greek *rho*) | 0.1 |
| $10^6$ | prices are per million tokens | |

**In words:** cached input at its discounted rate, plus the rest of the
input at full rate, plus output at the output rate, all per million.

**On the worked example:** (8,000 × 0.1 × 5 + 2,000 × 5 + 500 × 25) / 10⁶ =
(4,000 + 10,000 + 12,500) / 10⁶ = \$0.0265.

**In Python:**

```python
n_in, c, n_out = 10_000, 8_000, 500
p_in, p_out, rho = 5, 25, 0.1
cost = (c * rho * p_in + (n_in - c) * p_in + n_out * p_out) / 10**6
round(cost, 4)  # → 0.0265
```

Two things follow. An output token costs five times as much as an input
token here, so asking for concise answers pays. A cached prefix is nearly
free, so put the content that never changes at the start of the prompt
(see `primer.agents.context`). Some providers charge a small premium the
first time a prefix is written to the cache; the arithmetic here leaves it
out.

**In code:** `request_cost` is the formula above.

## Cost per successful task

A call's price is not what a task costs: a failed attempt is paid for,
and so is the retry or the person who fixes the result. Take a small model
at \$0.002 an attempt that succeeds 60% of the time and a large one at
\$0.010 that succeeds 95% of the time.

$$
\text{cost per success (retry until it works)} = \frac{c}{p}
\qquad
\text{cost per task (a person fixes failures)} = c + (1 - p)\,h
$$

**Symbols**

| Symbol | Meaning here | Small model | Large model |
|---|---|---|---|
| $c$ | model cost of one attempt | \$0.002 | \$0.010 |
| $p$ | chance an attempt succeeds | 0.60 | 0.95 |
| $1/p$ | expected number of attempts until one succeeds | 1.67 | 1.05 |
| $h$ | cost of a person fixing one failure (illustrative: a few minutes of staff time) | \$2.00 | \$2.00 |

**In words:** if you can simply retry, you pay one attempt's cost for every
expected attempt; if failures need a person, every task pays its attempt
plus the chance of failure times the cost of the fix.

**On the worked example:** retrying, the small model costs 0.002 / 0.60 =
\$0.0033 per success and the large one 0.010 / 0.95 = \$0.0105, so the small
model is cheaper. With a person fixing failures, the small model costs
0.002 + 0.40 × 2.00 = \$0.802 per task and the large one
0.010 + 0.05 × 2.00 = \$0.110: the cheap model costs seven times as much.

**In Python:**

```python
def per_success(c, p):
    # retry until it works
    return c / p
def per_task(c, p, h=2.00):
    # a person fixes each failure
    return c + (1 - p) * h
round(per_success(0.002, 0.60), 4), round(per_success(0.010, 0.95), 4)  # → (0.0033, 0.0105)
round(per_task(0.002, 0.60), 3), round(per_task(0.010, 0.95), 3)  # → (0.802, 0.11)
# small vs. large, with cleanup
round(per_task(0.002, 0.60) / per_task(0.010, 0.95))  # → 7
```

![With automatic retries the small model is cheaper per success (0.3 vs 1.1 cents); if a person fixes each failure the large model wins, 11 cents vs 80](figures/primer.agents.cost.unit_economics.svg)

**Reading it:** the dollar axis is a log scale. On the left, failures are
retried automatically and the small model's bar is the lower one. On the
right, a person fixes each failure: the small model's bar rises to 80 cents
and the large model's to 11. Once failures cost a person's time, the
failure rate sets the cost, not the price per token.

Every lever below is judged by this number. One that cuts the price of a
call but raises the failure rate can make tasks dearer, so the levers that
can change answers are checked against an evaluation set (see
`primer.agents.evals`).

**In code:** `cost_per_success_with_retries` is $c/p$ and
`cost_per_success_with_cleanup` is $c + (1-p)\,h$.

## Route each task to the cheapest model that can do it

The first lever is which model runs the task. Many requests are simple:
classify a ticket, pull a date out of an email. A small model handles
those at a fifth of the large model's price per token. A **router** sits in
front of every request and sends the simple ones to the small model and the
hard ones (planning, analysis, reasoning over several steps) to the large
one.

The sample workload has 10 tasks, 7 simple and 3 complex (planning a
migration). On the large model it costs \$0.671. Routing the 7 simple tasks to the small model brings it
to \$0.314, 53% less, with the hard tasks still on the large model.

```mermaid
flowchart LR
  T[Incoming task] --> R{Router<br/>rules or a small classifier}
  R -->|classify, extract,<br/>format, short| S[Small, fast model]
  R -->|plan, analyze,<br/>multi-step reasoning| L[Large model]
  S --> O[Result]
  L --> O
```

**Reading it:** every task passes through the router, which here is a few
keyword rules; a production router is often a small classifier. All of the
saving comes from the branch to the small model, and so does the risk: a
hard task sent down it costs less and fails. Check the router against the
evaluation set before and after you turn it on.

**In code:** `route` is the rule-based router, and `workload_cost` prices
the sample workload with any combination of levers switched on.

## Response caching and semantic caching

Routing makes a call cheaper; a cache skips it. An **exact response
cache** stores each answer under its question and returns it when the same
question comes again (ignoring case and spacing). It costs nothing and is
never wrong, but it misses every rewording. A **semantic cache** also
answers questions *similar* to a stored one. It turns each question into an
embedding (a vector in which closeness means similar meaning, see
`primer.ml.embeddings.similarity`) and returns the stored answer when the nearest
stored question is close enough. But similar is not the same: "How many
sick days do I get?" sits very close to "How many vacation days do I get?"
and needs a different answer.

Here are four pairs scored by a small demonstration embedder:

| Stored question | New question | Similarity | Same intent? |
|---|---|---|---|
| How do I reset my password? | I forgot my password, how do I recover it? | 0.88 | yes |
| How many vacation days do I get? | How many sick days do I get? | **0.97** | **no** |
| Request a new laptop | Request a new monitor | **0.96** | **no** |
| What does ERR-4012 mean? | What does ERR-4013 mean? | 0.56 | no |

The two pairs that ask different things score higher than the genuine
rewording, so a threshold high enough to reject them also rejects it. Two
cheap **guards** catch some wrong matches before the threshold is tried:
identifiers and numbers must match exactly (ERR-4012 is not ERR-4013), and
negation must match ("cancel my order" is not "don't cancel my order").
Neither guard can tell sick days from vacation days, which share a topic
and differ by one word.

```mermaid
flowchart TD
  Q[New question] --> E{Exact match<br/>in response cache?}
  E -->|yes| A1[Return stored answer]
  E -->|no| V[Embed and find the<br/>nearest stored question]
  V --> T{Similarity above<br/>threshold?}
  T -->|no| M[Call the model]
  T -->|yes| G{IDs, numbers and<br/>negation identical?<br/>entry not expired?}
  G -->|no| M
  G -->|yes| A2[Return stored answer]
  M --> S[Store the new answer]
```

**Reading it:** follow a question from the top. The exact cache comes
first because it costs nothing and is never wrong. On a miss, the semantic
path has two gates before it answers: the similarity threshold, then the
guards. The guards also turn away an entry older than its **TTL** (time to
live), so an answer about a fact that changes does not stay in use for
ever.

![On twelve labelled pairs, wrong hits stay at 40 to 60% from 0.4 through 0.95, above the paraphrase hit rate, and reach 0 only where paraphrase hits do too](figures/primer.agents.cost.semantic_cache.svg)

**Reading it:** the similarity threshold runs along the bottom. The blue
line is the share of seven genuine rewordings answered from the cache; the
red line is the share of five different-intent questions answered from it,
each a confident wrong answer. The guards stop two of the five, so red tops
out at 60%. From 0.4 to 0.95 red sits above blue: on these pairs, every
threshold in that range gets a larger share of the different questions
wrong than of the rewordings right. Above 0.95 the cache answers nothing.

Twelve hand-written pairs and a toy embedder are a demonstration, not a
measurement of semantic caching in general. What they show is the check to
run first: label pairs from your own traffic and measure the wrong-hit rate
at the threshold you plan to use. Keep the cache to narrow traffic, such as
a curated FAQ, with the guards and a TTL.

**In code:** `ResponseCache` is the exact cache and `SemanticCache` the
semantic one, guards and TTL included. `semantic_cache_sweep` draws the
figure from the labelled pairs in `CACHE_PAIRS`.

## Trim tokens

Caching and routing lower the price of tokens; trimming sends fewer of
them. Cut tool results down to the fields the next step needs (see
`primer.agents.context`), send the top few reranked
chunks instead of dozens, remove repeated boilerplate from the system
prompt, and ask for concise answers, since output is the expensive
direction.

**In code:** `workload_cost` models trimming as cutting each task's tool
output to 500 tokens, and concise output as cutting long answers by a third.

## Run independent tool calls at the same time

Latency, the wait for an answer, has a lever of its own. An agent that
calls three tools one after another waits for each in turn. When the calls
do not depend on each other, start them together, and the wait is the time
of the slowest: three lookups of 100 ms each take about 300 ms in sequence
and about 100 ms together.

```mermaid
sequenceDiagram
  participant A as Agent
  participant T1 as Weather API
  participant T2 as Calendar API
  participant T3 as CRM API
  Note over A,T3: Sequential: about 300 ms
  A->>T1: call
  T1-->>A: result
  A->>T2: call
  T2-->>A: result
  A->>T3: call
  T3-->>A: result
  Note over A,T3: Parallel: about 100 ms
  par
    A->>T1: call
  and
    A->>T2: call
  and
    A->>T3: call
  end
  T1-->>A: result
  T2-->>A: result
  T3-->>A: result
```

**Reading it:** in the top half each call waits for the result of the one
before; in the bottom half the three requests leave together and the agent
waits once, for the slowest.

A model can ask for several tools in one turn (**parallel tool use**). Run
them concurrently and send all the results back in one message. This
shortens the wait, not the bill: the tokens are the same.

![Three 100 ms tool calls take 300 ms end to end when run one after another, but only 100 ms when started together](figures/primer.agents.cost.parallel.svg)

**Reading it:** the same three calls on a time axis in milliseconds. The
sequential bars stack end to end and finish at 300; the parallel bars
start together and finish at 100. The chart is drawn from the planned
delays; running the module times real calls, which land within a few
milliseconds of them.

**In code:** `run_sequential` awaits each call before starting the next;
`run_parallel` starts them all with Python's asyncio gather and waits once.

**Streaming**, showing the answer as it is generated, does not shorten the
total time either, but the user sees progress at once, so the system feels
faster.

## Batch what isn't interactive

Some work has no one waiting on it: nightly evaluation runs, reprocessing a
backlog of documents, classifying a pile of old tickets. Providers offer
**batch APIs** for it. You submit many requests at once and collect the
results later, within a day, at about half the normal price.

**In code:** `batch_cost` prices a list of requests at half price.

## Budgets and alerts

The levers so far lower the normal bill. Budgets guard against an abnormal
one, from an agent stuck in a loop or a customer whose usage runs away.

A **task budget** caps the steps and tokens one task may use, so a confused
agent stops instead of looping all night. A **tenant spend cap** (a tenant
is one customer organisation on a shared platform) stops one customer's
runaway usage from turning into a surprise invoice. An **anomaly alert**
fires when a task uses far more tokens than usual:

$$
\text{alert if } x > \mu + z\,\sigma
$$

**Symbols**

| Symbol | Meaning here | Worked example |
|---|---|---|
| $x$ | tokens used by the task just finished | 10,000 |
| $\mu$ | mean tokens per task over this tenant's recent history (Greek *mu*) | 1,000 |
| $\sigma$ | standard deviation of that history: the typical distance from the mean (Greek *sigma*) | ≈ 71 |
| $z$ | how many standard deviations count as unusual | 3 |

**In words:** alert when a task uses more tokens than the usual amount plus
three times the usual spread.

**On the worked example:** history 1000, 1100, 900, 1050, 950, 1000 has mean
1,000 and standard deviation ≈ 71 (the squared distances from the mean add
up to 25,000; divided by 6 − 1 = 5 that's 5,000, whose square root is 70.7),
so the line is 1,000 + 3 × 70.7 ≈ 1,212.
A 10,000-token task is far above it, so the alert fires. A jump like that
points to a loop or a bad deploy.

**In Python:**

```python
import statistics
history = [1000, 1100, 900, 1050, 950, 1000]
mu = statistics.mean(history)
# divides by 6 - 1, as above
sigma = statistics.stdev(history)
mu, round(sigma, 1)  # → (1000, 70.7)
z = 3
# the alert line
round(mu + z * sigma)  # → 1212
# alert?
10_000 > mu + z * sigma  # → True
```

**In code:** `TaskBudget` stops a task at its step or token limit, and
`TenantSpend` keeps each tenant's total and raises the cap and anomaly
alerts.

## The levers in order

Take the levers that cannot hurt quality first:

1. **Prompt caching**: reorder the prompt so the stable prefix is reused.
   The answers do not change.
2. **Trim tokens**: compress tool results and send fewer chunks.
3. **Concise output**: ask for shorter answers where length adds nothing.
4. **Route easy tasks to the small model**: the first lever that can change
   the answers, so check it against the evaluation set before and after.
5. **Batch** the work no one is waiting on.

![Workload cost falls from 67 to 8 cents as levers stack: caching 2.2x, trimming 3.6x, concise output 3.9x, routing 7.8x, batching 8.4x](figures/primer.agents.cost.five_x.svg)

**Reading it:** each bar is the cost of the same 10-task workload with
that lever and every lever to its left switched on; the label is the total
reduction so far. Caching alone more than halves the cost. Trimming and
concise output take it to 3.9x, routing to 7.8x and batching to 8.4x, from
67 cents to 8. The three blue bars are the levers that cannot change
quality.

**In code:** `five_x_plan` switches the levers on one at a time, in this
order, and reports the cost and cumulative reduction after each.

## In 20 seconds
- Measure cost per successful task, not per call. A cheap model that fails
  often can cost more per task than a dear one.
- First take the levers that cannot change quality: prompt caching,
  trimming tool output and chunks, and concise answers.
- Then route easy tasks to a small model, checked against an evaluation
  set.
- Batch work no one waits on, at about half price; run independent tool
  calls together; stream answers so the wait feels shorter.
- Cap steps and tokens per task and spend per customer; alert on sudden
  jumps in tokens per task.
- A semantic cache can answer a different question with confidence;
  measure its wrong-hit rate before using it.

## Self-test questions

**How would you cut the cost per task by 5x without hurting quality?**
First measure: cost per successful task, broken down by step, model, input
vs. output, and cached vs. uncached tokens, with an eval set to hold
quality fixed. Then, in order: restructure prompts for prompt caching;
compress tool results and send only the top reranked chunks; ask for
concise output; route simple steps (classification, extraction,
formatting) to a small model, checking the eval before and after; move
anything non-interactive to a batch API; cap steps and tokens per task.
On the sample workload here that's about 8x, and every step after the
first three is checked against the eval set.

**Why can a cheaper model cost more?**
Because failures cost money too: retries, human cleanup, lost customers.
At 60% success with a \$2 human fix, a \$0.002 call costs \$0.80 per task;
a \$0.010 call at 95% costs \$0.11.

**What are the risks of a semantic cache?**
Returning a confident, wrong answer to a question that's similar but not
the same ("sick days" vs "vacation days"), and serving stale answers after
facts change. Mitigate with high thresholds, exact-match guards on IDs,
numbers and negation, per-intent scoping, a TTL, and a measured wrong-hit
rate on labelled pairs before enabling it.

**Your average tokens per task doubled overnight. What do you check?**
Whether a deploy changed a prompt or a tool (a bigger tool output, a new
retrieval setting), whether an agent is looping (the same tool called with
the same arguments), and whether the prompt cache hit rate dropped
(something volatile moved to the top). Traces make this a lookup rather
than a guess (`primer.agents.observability`).

## The papers behind this lesson

- Hinton, Vinyals & Dean, *Distilling the Knowledge in a Neural Network*
  (2015), https://arxiv.org/abs/1503.02531. Showed how to train a small
  model to imitate a large one, the technique behind making the cheap model
  good enough to take more of the routed traffic.
  [annotated companion](../../papers/distillation.html)
- Chen, Zaharia & Zou, *FrugalGPT: How to Use Large Language Models While
  Reducing Cost and Improving Performance* (2023),
  https://arxiv.org/abs/2305.05176. Studied prompt adaptation, caching and
  cascades that try cheap models first and escalate to expensive ones only
  when needed.

## Further reading
- Anthropic prompt caching docs: https://docs.claude.com/en/docs/build-with-claude/prompt-caching
- Anthropic Message Batches docs: https://docs.claude.com/en/docs/build-with-claude/batch-processing
- Anthropic, parallel tool use: https://docs.claude.com/en/docs/agents-and-tools/tool-use/implement-tool-use
- Python `asyncio.gather`: https://docs.python.org/3/library/asyncio-task.html#asyncio.gather
"""

from __future__ import annotations

import asyncio
import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from primer._show import banner, say, table, takeaway
from primer.common.embedder import ConceptEmbedder

# ---------------------------------------------------------------------------
# 1. Pricing (ILLUSTRATIVE constants, not any vendor's price list)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Price:
    input_per_m: float  # dollars per million input tokens
    output_per_m: float  # dollars per million output tokens


PRICES: dict[str, Price] = {"small": Price(1.0, 5.0), "large": Price(5.0, 25.0)}
CACHE_READ_MULTIPLIER = 0.1  # cached input billed at 10% of the input price
BATCH_MULTIPLIER = 0.5  # batch interface at half price


def request_cost(model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
    """Dollar cost of one call. `cached_tokens` of the input are billed at the cached rate."""
    p = PRICES[model]
    uncached = input_tokens - cached_tokens
    return (cached_tokens * CACHE_READ_MULTIPLIER * p.input_per_m + uncached * p.input_per_m + output_tokens * p.output_per_m) / 1e6


def batch_cost(requests: list[tuple[str, int, int]]) -> float:
    """Cost of (model, input, output) requests submitted through a batch interface."""
    return BATCH_MULTIPLIER * sum(request_cost(m, i, o) for m, i, o in requests)


# ---------------------------------------------------------------------------
# 2. Routing
# ---------------------------------------------------------------------------

# Words that suggest multi-step reasoning. A real router is often a small
# classifier trained on labelled traffic; rules are the transparent start.
HARD_SIGNALS = ("plan", "analy", "design", "strategy", "compare", "debug", "reason", "multi-step", "trade-off", "why")


def route(task_text: str, max_simple_chars: int = 400) -> str:
    """Return "small" or "large" for a task."""
    text = task_text.lower()
    if len(text) > max_simple_chars or any(s in text for s in HARD_SIGNALS):
        return "large"
    return "small"


@dataclass
class WorkItem:
    text: str
    output_tokens: int
    interactive: bool = True
    # Input is split so the levers have something to act on.
    stable_prefix: int = 8_000  # system prompt + tool definitions, identical every call
    tool_output: int = 3_000  # verbose tool results that could be compressed
    rest: int = 1_000  # the question and its specific context


SAMPLE_WORKLOAD: list[WorkItem] = (
    [WorkItem("Classify this ticket as billing, technical or other", 150, interactive=False) for _ in range(4)]
    + [WorkItem("Extract the invoice date and total from this email", 150) for _ in range(3)]
    + [WorkItem("Plan the migration of our billing system and analyze the risks of each step", 600) for _ in range(3)]
)


def workload_cost(work: list[WorkItem], cache: bool = False, trim: bool = False, concise: bool = False,
                  routed: bool = False, batch: bool = False) -> float:
    """Cost of a workload with any combination of levers switched on."""
    total = 0.0
    for w in work:
        tool = 500 if trim else w.tool_output  # compress tool output to the needed fields
        out = int(w.output_tokens * 2 / 3) if concise and w.output_tokens > 300 else w.output_tokens
        n_in = w.stable_prefix + tool + w.rest
        model = route(w.text) if routed else "large"
        c = request_cost(model, n_in, out, cached_tokens=w.stable_prefix if cache else 0)
        if batch and not w.interactive:
            c *= BATCH_MULTIPLIER
        total += c
    return total


def routing_savings(work: list[WorkItem] | None = None) -> dict[str, float]:
    work = work or SAMPLE_WORKLOAD
    all_large = workload_cost(work)
    routed = workload_cost(work, routed=True)
    return {"all_large": all_large, "routed": routed, "savings": 1 - routed / all_large}


# ---------------------------------------------------------------------------
# 3. Response cache and semantic cache
# ---------------------------------------------------------------------------


def _normalize(q: str) -> str:
    return " ".join(q.lower().split())


class ResponseCache:
    """Exact-match cache: free and never wrong, but misses every paraphrase."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def put(self, question: str, answer: str) -> None:
        self._store[_normalize(question)] = answer

    def get(self, question: str) -> str | None:
        return self._store.get(_normalize(question))


_NEGATION = re.compile(r"\b(not|no|never|don't|dont|doesn't|isn't|can't|won't|cannot)\b")


def _identifiers(text: str) -> set[str]:
    """Tokens containing a digit: order ids, error codes, amounts, dates."""
    return {t for t in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()) if any(c.isdigit() for c in t)}


def _negated(text: str) -> bool:
    return bool(_NEGATION.search(text.lower()))


@dataclass
class _Entry:
    vector: np.ndarray
    question: str
    answer: str
    stored_at: float


class SemanticCache:
    """Answer similar questions from memory, with guards against the obvious wrong hits.

    Guards: identifiers and numbers must match exactly, negation must match,
    and entries expire after `ttl_seconds`. None of this can separate
    "sick days" from "vacation days"; only narrow scoping and a measured
    wrong-hit rate can.
    """

    def __init__(self, embedder: Any = None, threshold: float = 0.9, ttl_seconds: float | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.embedder = embedder or ConceptEmbedder()
        self.threshold = threshold
        self.ttl = ttl_seconds
        self.clock = clock
        self.entries: list[_Entry] = []

    def put(self, question: str, answer: str) -> None:
        self.entries.append(_Entry(self.embedder.encode(question), question, answer, self.clock()))

    def lookup(self, question: str) -> tuple[_Entry | None, float]:
        """Nearest stored entry that passes the guards, and its similarity."""
        q = self.embedder.encode(question)
        best, best_sim = None, -1.0
        for e in self.entries:
            if self.ttl is not None and self.clock() - e.stored_at > self.ttl:
                continue
            if _identifiers(e.question) != _identifiers(question) or _negated(e.question) != _negated(question):
                continue
            sim = float(e.vector @ q)  # vectors are unit length, so dot product = cosine similarity
            if sim > best_sim:
                best, best_sim = e, sim
        return best, best_sim

    def get(self, question: str) -> str | None:
        entry, sim = self.lookup(question)
        return entry.answer if entry is not None and sim >= self.threshold else None


# (stored question, new question, same intent?)
CACHE_PAIRS: list[tuple[str, str, bool]] = [
    ("How do I reset my password?", "I forgot my password, how do I recover it?", True),
    ("How do I reset my password?", "How do I reset my VPN password?", False),
    ("How many vacation days do I get?", "How much PTO do I get per year?", True),
    ("How many vacation days do I get?", "How many sick days do I get?", False),
    ("How do I get reimbursed for car mileage?", "automobile expense reimbursement", True),
    ("Cancel my order", "Don't cancel my order", False),
    ("What does ERR-4012 mean?", "What does ERR-4013 mean?", False),
    ("How do I set up the VPN?", "VPN setup steps", True),
    ("Report a phishing email", "How do I report a suspicious email?", True),
    ("Request a new laptop", "Request a new monitor", False),
    ("What is the travel per-diem?", "How much is the daily meal per-diem when traveling?", True),
    ("Printer not working", "printing fails on floor printer", True),
]


def semantic_cache_sweep(thresholds: list[float]) -> list[dict[str, float]]:
    """For each threshold: share of paraphrases served from cache, share of near-misses served wrongly."""
    out = []
    for th in thresholds:
        useful = wrong = 0
        for stored, new, same in CACHE_PAIRS:
            cache = SemanticCache(threshold=th)
            cache.put(stored, "cached answer")
            hit = cache.get(new) is not None
            useful += hit and same
            wrong += hit and not same
        n_same = sum(1 for p in CACHE_PAIRS if p[2])
        out.append({"threshold": th, "useful_hit_rate": useful / n_same, "wrong_hit_rate": wrong / (len(CACHE_PAIRS) - n_same)})
    return out


# ---------------------------------------------------------------------------
# 4. Parallel tool calls
# ---------------------------------------------------------------------------


async def _fake_tool(i: int, delay: float, t0: float, log: list) -> str:
    start = time.perf_counter() - t0
    await asyncio.sleep(delay)  # stands in for a network call
    log.append((i, start, time.perf_counter() - t0))
    return f"tool-{i}"


async def run_sequential(delays: list[float]) -> dict[str, Any]:
    t0, log = time.perf_counter(), []
    results = [await _fake_tool(i, d, t0, log) for i, d in enumerate(delays)]
    return {"seconds": time.perf_counter() - t0, "results": results, "spans": sorted(log)}


async def run_parallel(delays: list[float]) -> dict[str, Any]:
    t0, log = time.perf_counter(), []
    # gather starts every call before awaiting any, and returns results in call order.
    results = await asyncio.gather(*(_fake_tool(i, d, t0, log) for i, d in enumerate(delays)))
    return {"seconds": time.perf_counter() - t0, "results": list(results), "spans": sorted(log)}


# ---------------------------------------------------------------------------
# 5. Budgets and spend alerts
# ---------------------------------------------------------------------------


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class TaskBudget:
    """Hard per-task limits. The agent loop calls charge() before every model call."""

    max_steps: int
    max_tokens: int
    steps: int = 0
    tokens: int = 0

    def charge(self, tokens: int) -> None:
        if self.steps + 1 > self.max_steps:
            raise BudgetExceeded(f"step limit {self.max_steps} reached")
        if self.tokens + tokens > self.max_tokens:
            raise BudgetExceeded(f"token limit {self.max_tokens} would be exceeded ({self.tokens} + {tokens})")
        self.steps += 1
        self.tokens += tokens


@dataclass
class TenantSpend:
    """Per-tenant monthly spend cap plus an anomaly alert on tokens per task."""

    monthly_cap: float
    z: float = 3.0
    min_history: int = 5
    spent: dict[str, float] = field(default_factory=dict)
    history: dict[str, list[int]] = field(default_factory=dict)

    def record(self, tenant: str, cost: float, tokens: int) -> dict[str, str]:
        """Record one finished task; return alerts keyed by kind ("cap", "anomaly")."""
        alerts: dict[str, str] = {}
        self.spent[tenant] = self.spent.get(tenant, 0.0) + cost
        if self.spent[tenant] > self.monthly_cap:
            alerts["cap"] = f"{tenant} spent {self.spent[tenant]:.2f} of a {self.monthly_cap:.2f} cap"
        hist = self.history.setdefault(tenant, [])
        if len(hist) >= self.min_history:
            mu, sigma = statistics.mean(hist), statistics.stdev(hist)
            if tokens > mu + self.z * max(sigma, 1.0):
                alerts["anomaly"] = f"{tokens} tokens vs usual {mu:.0f} ± {sigma:.0f}"
        hist.append(tokens)
        return alerts


# ---------------------------------------------------------------------------
# 6. Unit economics and the plan
# ---------------------------------------------------------------------------


def cost_per_success_with_retries(cost_per_attempt: float, success_rate: float) -> float:
    """Retry until success: expected attempts are 1/p."""
    return cost_per_attempt / success_rate


def cost_per_success_with_cleanup(cost_per_attempt: float, success_rate: float, human_fix_cost: float) -> float:
    """One attempt per task; a person fixes each failure."""
    return cost_per_attempt + (1 - success_rate) * human_fix_cost


def five_x_plan(work: list[WorkItem] | None = None) -> list[dict[str, Any]]:
    """Apply the levers cumulatively, free ones first. Returns cost after each."""
    work = work or SAMPLE_WORKLOAD
    levers = [
        ("baseline: everything on the large model", {}),
        ("prompt caching", {"cache": True}),
        ("trim tool output", {"cache": True, "trim": True}),
        ("concise output", {"cache": True, "trim": True, "concise": True}),
        ("route easy tasks to the small model", {"cache": True, "trim": True, "concise": True, "routed": True}),
        ("batch non-interactive work", {"cache": True, "trim": True, "concise": True, "routed": True, "batch": True}),
    ]
    base = workload_cost(work)
    steps = []
    for name, kw in levers:
        c = workload_cost(work, **kw)
        steps.append({"lever": name, "cost": c, "factor": base / c})
    return steps


# ---------------------------------------------------------------------------
# Figures and demo
# ---------------------------------------------------------------------------


def figures() -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs: dict[str, Any] = {}

    # 1. Unit economics.
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    labels = ["retry until success", "person fixes failures"]
    small = [cost_per_success_with_retries(0.002, 0.60), cost_per_success_with_cleanup(0.002, 0.60, 2.0)]
    large = [cost_per_success_with_retries(0.010, 0.95), cost_per_success_with_cleanup(0.010, 0.95, 2.0)]
    x = np.arange(2)
    ax.bar(x - 0.2, small, 0.4, label="small: $0.002/attempt, 60% success", color="#c44e52")
    ax.bar(x + 0.2, large, 0.4, label="large: $0.010/attempt, 95% success", color="#4c72b0")
    for xi, (s, lg) in enumerate(zip(small, large)):
        ax.text(xi - 0.2, s, f"${s:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + 0.2, lg, f"${lg:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("dollars per successful task (log scale)")
    ax.set_title("The cheap model is only cheap if failures are free")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    figs["unit_economics"] = fig

    # 2. The 5x plan.
    steps = five_x_plan()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(range(len(steps)), [s["cost"] for s in steps], color=["#8c8c8c"] + ["#4c72b0"] * 3 + ["#dd8452"] * 2)
    for i, s in enumerate(steps):
        ax.text(i, s["cost"], f"{s['factor']:.1f}x", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(steps)), ["baseline", "+ caching", "+ trim", "+ concise", "+ routing", "+ batch"])
    ax.set_ylabel("workload cost ($)")
    ax.set_title("Levers applied in order (blue: can't change quality; orange: validate with evals)")
    fig.tight_layout()
    figs["five_x"] = fig

    # 3. Semantic cache threshold sweep.
    ths = list(np.round(np.arange(0.3, 1.001, 0.025), 3))
    sweep = semantic_cache_sweep(ths)
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.plot(ths, [s["useful_hit_rate"] for s in sweep], color="#4c72b0", label="paraphrases served from cache (good)")
    ax.plot(ths, [s["wrong_hit_rate"] for s in sweep], color="#c44e52", label="different questions served from cache (wrong)")
    ax.set_xlabel("similarity threshold")
    ax.set_ylabel("share of pairs")
    ax.set_title("Semantic cache: no threshold separates them cleanly")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figs["semantic_cache"] = fig

    # 4. Parallel vs sequential timeline: the schedule each approach follows. Drawn from the delays, not
    # a stopwatch, so the site doesn't change with a millisecond of jitter; the demo measures the real thing.
    delays_ms = [100, 100, 100]
    sequential = [(i, sum(delays_ms[:i]), sum(delays_ms[: i + 1])) for i in range(len(delays_ms))]
    parallel = [(i, 0, d) for i, d in enumerate(delays_ms)]
    fig, ax = plt.subplots(figsize=(6.5, 3))
    for row, (name, spans, color) in enumerate([("sequential", sequential, "#c44e52"), ("parallel", parallel, "#4c72b0")]):
        for i, start, end in spans:
            ax.barh(row * 4 + i, end - start, left=start, color=color)
            ax.text(start + 2, row * 4 + i, f"tool {i}", va="center", fontsize=8, color="white")
    ax.set_yticks([1, 5], ["sequential", "parallel"])
    ax.invert_yaxis()
    ax.set_xlabel("milliseconds since the agent started the calls")
    ax.set_title("Three independent 100 ms tool calls")
    fig.tight_layout()
    figs["parallel"] = fig
    return figs


def demo() -> None:
    banner("1. How a call is priced (illustrative prices)")
    table(["call", "cost"], [
        ("large, 10k in / 500 out", f"${request_cost('large', 10_000, 500):.4f}"),
        ("same, 8k of input cached", f"${request_cost('large', 10_000, 500, cached_tokens=8_000):.4f}"),
        ("same, via batch interface", f"${batch_cost([('large', 10_000, 500)]):.4f}"),
        ("small, 10k in / 500 out", f"${request_cost('small', 10_000, 500):.4f}"),
    ])

    banner("2. Routing")
    r = routing_savings()
    print(f"all large ${r['all_large']:.3f} -> routed ${r['routed']:.3f}: {r['savings']:.0%} saved")
    print()

    banner("3. Semantic caching: useful and dangerous")
    cache = SemanticCache(threshold=0.85)
    cache.put("How do I reset my password?", "Use the self-service portal.")
    cache.put("How many vacation days do I get?", "20 days of PTO per year.")
    for q in ["I forgot my password, how do I recover it?", "How many sick days do I get?", "How do I reset my VPN password?"]:
        entry, sim = cache.lookup(q)
        print(f"{q!r:48} sim {sim:.2f} -> {cache.get(q)!r}")
    print()
    say("The sick-days question gets the vacation answer. That's the risk: similar is not the same.")

    banner("4. Parallel tool calls")
    seq, par = asyncio.run(run_sequential([0.1] * 3)), asyncio.run(run_parallel([0.1] * 3))
    print(f"sequential {seq['seconds'] * 1000:.0f} ms, parallel {par['seconds'] * 1000:.0f} ms")
    print()

    banner("5. Budgets and alerts")
    spend = TenantSpend(monthly_cap=1000)
    for t in [1000, 1100, 900, 1050, 950, 1000, 10_000]:
        alerts = spend.record("acme", 0.01, t)
        if alerts:
            print(f"task with {t} tokens -> {alerts}")
    print()

    banner("6. Unit economics")
    table(["model", "per success (retry)", "per task (human fixes)"], [
        ("small $0.002 @ 60%", cost_per_success_with_retries(0.002, 0.6), cost_per_success_with_cleanup(0.002, 0.6, 2.0)),
        ("large $0.010 @ 95%", cost_per_success_with_retries(0.010, 0.95), cost_per_success_with_cleanup(0.010, 0.95, 2.0)),
    ])

    banner("7. The plan, free levers first")
    table(["lever", "workload cost", "cumulative"], [(s["lever"], f"${s['cost']:.4f}", f"{s['factor']:.1f}x") for s in five_x_plan()])
    takeaway("Measure cost per successful task, take the free wins first, then route with evals watching quality.")


if __name__ == "__main__":
    demo()
