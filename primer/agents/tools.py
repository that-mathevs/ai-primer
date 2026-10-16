r"""
# Tools: design, validation and safety

Run: `python -m primer.agents.tools`

New to the notation? `primer.notation` explains every symbol used here from
zero. This lesson builds on the tool-call exchange in `primer.agents.llm`
and on the loop that runs it in `primer.agents.agent_loop`.

## Level 1: The practitioner's guide

**In one sentence.** A tool is a function you let the model ask for, and
tool design is everything that decides whether the request it fills in is
the right one, has correct arguments, and is safe to carry out: the
definition the model reads, the checks your code runs, and the gates in
front of the real system.

**When you need it.** The moment a model's output does something rather
than says something: looks up a record, refunds a payment, sends an email.
A chatbot that only answers from its weights has no tools and needs none of
this. A one-off script with one read-only tool needs the definition and
little else. Everything in this lesson becomes necessary as soon as a tool
writes, costs money or can be called by a model that was misled. The tell:
if a wrong tool call would be visible to a customer, an auditor or a bank,
you need the checks. Two of this lesson's measurements show how much the
design decides. The same three tools, described vaguely ("search stuff"),
get the right tool for 2 of 6 requests; described precisely (what it does,
when to use it, when not to), 6 of 6. And a job that takes five tool calls
in a row, each 97% likely to be right, succeeds 85.9% of the time, so
about one run in seven fails; as one high-level call it succeeds 97% of
the time.

**Your options.** Six layers of protection between the model's request and
the real system, from the cheapest to the most certain. They stack; a
production registry has all of them.

| Option | What it does | What it guarantees | What it costs | Where it lives |
|---|---|---|---|---|
| A schema in the definition | A JSON Schema tells the model which fields exist, their types and which are required | Nothing by itself; the model does its best to match | A few tokens per tool per call | Your tool definition |
| Strict mode | The provider constrains the model's output, token by token, to your schema | Arguments in exactly the schema's shape and a tool name that exists (Claude's strict tool use; OpenAI's strict mode) | A schema written to the provider's rules (`additionalProperties: false`, required fields); the compiled grammar is cached | The model server |
| Validation in your registry | Checks shape and then business rules, and returns every problem with what a correct value looks like | A wrong call never reaches the system, and the model can fix all its mistakes in one retry (four at once in the lesson's example) | A validator and the rules; error messages worth writing | Your code |
| Descriptions and consolidation | Precise descriptions, few parameters, `enum`s for closed sets, and one high-level tool in place of a chain | The right tool chosen far more often (6 of 6 against 2 of 6 here), and fewer chances to fail in a row | Writing, and an evaluation set to measure selection | Your tool definitions |
| Loading only the relevant tools | Retrieval over the catalogue picks the few tools a request needs (5 of 30 in the lesson), or the provider searches it on demand | Selection accuracy holds as the catalogue grows; far fewer definition tokens per call | One embedding or search per request; the top few most-used tools kept always loaded | Your code, or the model server (Claude's tool search) |
| Safety gates | Scopes per agent, a dry run that describes instead of acting, human approval over a threshold, an idempotency key on every write | Least privilege, rehearsals, a person's sign-off, and a retry that cannot repeat a refund | A credential model, an approver in the loop, a store of keys and results | Your code, directly in front of the real system |

**How to choose.** Decide by what the tool can do, not by how clever the
model is.

- Read-only lookups: a schema, a good description, and validation that
  returns actionable errors. Strict mode if the provider offers it; it
  removes a class of retries for free.
- Anything that writes: all of the above, plus an idempotency key on the
  call and business-rule checks before it runs. `C-999` fits the pattern
  `^C-\d+$` and is still a customer that does not exist.
- Anything irreversible or over a money threshold: an approval rule, and a
  declined message that tells the model not to retry.
- Any agent that reads untrusted text (email, web pages, documents): the
  narrowest scopes you can give it, so a tricked model *cannot* refund, not
  merely should not.
- A catalogue past a few dozen tools: load per request, or route to a
  sub-agent that holds only its own tools. Claude's docs put the accuracy
  drop past 30 to 50 tools and recommend search from 10 tools up.
- Whatever you pick, prefer fewer, higher-level tools. A chain of five
  calls compounds five chances to go wrong; the sequencing belongs in
  tested code, not in the model.

**What it costs.** Tokens: every definition is re-sent on every call, so a
long catalogue is a standing charge. Claude's tool-search docs put a
typical multi-server setup (GitHub, Slack, Sentry, Grafana, Splunk) at about
55,000 tokens of definitions before any work is done, and on-demand loading
cuts that by over 85%. Tool results cost tokens too: Anthropic's tool-writing
guide reports that a concise response format used about a third of the
tokens of a detailed one, and that Claude Code caps a tool response at
25,000 tokens by default. Latency: validation is microseconds; an approval
gate is however long a person takes, so the call parks as
`awaiting_approval` rather than blocking. Quality: descriptions are the
cheapest lever, and the lesson's 2-of-6 to 6-of-6 swing cost only words.
Effort: the registry in this lesson, with every gate, is a few hundred lines,
and the idempotency store can be a dictionary until it needs to survive a
restart.

**What breaks.**

- **Schema-valid, false.** Strict mode and schemas prove shape, not truth.
  Keep the business-rule check, and return an error that says how to find
  the right id.
- **"Invalid input."** A bare error makes the model guess. List every
  problem with the expected format ("date as YYYY-MM-DD"); the model fixes
  them all in one retry.
- **Vague descriptions.** A model that cannot tell the tools apart picks
  the first one every time; in the lesson it got exactly the two requests
  that happened to belong to that tool.
- **Too many tools.** Selection accuracy falls and definition tokens climb
  together. Load per request.
- **A retry that pays twice.** A refund that times out after the bank
  processed it gets retried. Without a key the customer gets \$40; with the
  same key the registry returns the stored result and refunds once.
  Stripe's API keeps a key's first result and returns it on every repeat.
- **A persistent model after a decline.** "Declined" alone invites another
  attempt. Say "do not retry, tell the user".
- **One admin credential.** If the agent is tricked, it can do anything
  the credential allows. Scope per tool and per agent.

**In the wild.** Claude's Messages API accepts `strict: true` on a tool and
guarantees the arguments match the schema and the name is valid; its tool
search tool defers tool definitions and loads them when the model searches
for them; OpenAI's function calling has a strict mode its guide recommends
always enabling. Anthropic's *Writing effective tools for agents* is the
design reference behind sections 3 and 4 here: a few thoughtful tools over
one per endpoint, consistent namespacing (`asana_search`, `jira_search`),
responses that return meaning rather than identifiers, and evaluations
built from dozens of real prompt and response pairs. Stripe's idempotent
requests are the canonical form of the idempotency key: a client-chosen
key, the first result saved and replayed, keys pruned after 24 hours.
Toolformer (Schick et al., 2023) is the paper that showed a model can learn
when and how to call an API, which is why today's models emit tool calls
at all.

**Go deeper.** Level 2 writes the tool call out message by message, walks
a call through every diamond of the registry in the order they are checked,
measures the description experiment and the compounding formula, builds
tool retrieval from cosine similarity, and replays the lost-reply refund
with and without a key. If you only needed to choose, you are done.

## Level 2: How it works, from scratch

A language model can only produce text, so anything it does in the world
goes through code you own. Here that code is a **tool registry**: it holds
the tools, shows them to the model, and checks every request the model
makes before carrying it out.

## 1. What a tool call is

Picture a new assistant who may not touch anything. When they need
something done, they fill in a request form ("refund customer C-100, $20")
and hand it to you. You decide whether to carry it out, and hand back a note
saying what happened. A language model works with tools the same way. You
send it a list of tools, each with a name, a description and the shape of
its arguments. When it wants one, it stops and writes a **tool call**: the
tool's name with the arguments filled in, like the form. Here is the whole
exchange for "what's 2 + 3?" with one tool, written as the actual messages
in the format of `primer.agents.llm`:

```text
you -> model   tools: [{"name": "add", "description": "Add two integers.",
                        "input_schema": {"type": "object",
                          "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                          "required": ["a", "b"]}}]
               user: "what's 2 + 3?"
model -> you   tool_use {"id": "toolu_1", "name": "add", "input": {"a": 2, "b": 3}}
               stop_reason: "tool_use"          <- "please run this for me"
you            (validate the input, run add(2, 3) -> 5)
you -> model   user: tool_result {"tool_use_id": "toolu_1", "content": "5"}
model -> you   "2 + 3 = 5."   stop_reason: "end_turn"
```

The `input_schema` is a **JSON Schema**, a JSON document that describes the
shape other JSON must have: which fields exist, what type each is, and
which are required. It is the blank form the model fills in.

```mermaid
sequenceDiagram
  participant M as Model
  participant C as Your code
  participant T as The real system
  C->>M: tool definitions (name, description, JSON Schema) + user message
  M-->>C: tool_use: name + arguments (a filled-in form)
  C->>C: check the form (schema, business rules, permissions, approval)
  C->>T: carry it out
  T-->>C: result
  C->>M: tool_result (matched by tool_use_id)
  M-->>C: answer in words
```

**Reading it:** follow the arrows from top to bottom. The model never talks
to the real system; everything it writes goes to your code first. The arrow
that loops back on your code, "check the form", is where every safeguard
runs. That split is what keeps
the system safe: a model can be wrong or tricked, and your code is where you
enforce what is allowed.

**In code:** `ToolRegistry.register` adds a tool, `ToolRegistry.definitions`
lists them for the model, and `ToolRegistry.call` runs a call through the
checks.

## 2. Validate every call before running it

The first thing your code does with a call is check it, the way a bank
clerk checks a withdrawal slip twice. First, is it filled in properly:
every box, a real date, a positive amount? Then, does it make sense: does
the account exist, and does it have the money? For a tool call, the first
check is the JSON Schema and the second is your **business rules**.

Take a payment tool with an `amount`, a `currency` (EUR, GBP or USD) and an
optional `pay_on` date. The model sends
`{"ammount": 5, "currency": "usd", "pay_on": "next friday"}`. The validator
returns *every* problem at once, each saying what a correct value looks
like:

```text
$: missing required field 'amount'
$: unexpected field 'ammount' (allowed: amount, currency, pay_on)
$.currency: must be one of ['EUR', 'GBP', 'USD'], got 'usd'
$.pay_on: 'next friday' does not match the required format. Expected: date as YYYY-MM-DD
```

Given these four lines, the model can fix all four mistakes in one retry.
Given only `invalid input`, it has to guess.

Passing the schema is not enough. `"C-999"` matches the pattern `^C-\d+$`
and still names a customer that doesn't exist. With the Claude API you can
mark a tool definition `"strict": true`, and the API then guarantees that
the model's arguments match the schema (`primer.ml.structured_output`
explains how such guarantees work). That settles the shape, not the truth,
so the business-rule check stays.

A full registry runs these two checks among others, in the order below. The
last three (dry run, approval and the idempotency key) guard calls that
change something, and are explained with the other safeguards at the end.

```mermaid
flowchart TD
  A[tool_use from the model] --> U{Tool exists?}
  U -->|no| E1[unknown_tool:<br/>list the real tools]
  U -->|yes| P{Caller has the<br/>required scope?}
  P -->|no| E2[denied]
  P -->|yes| S{Matches the<br/>JSON Schema?}
  S -->|no| E3[invalid:<br/>every problem, with the fix]
  S -->|yes| B{Business rules pass?<br/>e.g. customer exists}
  B -->|no| E4[rejected]
  B -->|yes| D{Dry run?}
  D -->|yes| E5[dry_run:<br/>describe, change nothing]
  D -->|no| H{Needs human<br/>approval?}
  H -->|yes, none given| E6[awaiting_approval]
  H -->|declined| E7[declined]
  H -->|no, or approved| I{Idempotency key<br/>seen before?}
  I -->|yes| E8[duplicate:<br/>return first result]
  I -->|no| X[run the tool: ok]
```

**Reading it:** a call enters at the top and must pass every diamond to
reach *run the tool* at the bottom. Each exit from a diamond ends in a named
status, and each returns text the model can act on. Permissions come before
anything that reveals how the tool works, and cheap checks come before
expensive ones. Approval and idempotency sit last, next to the action they
protect.

**In code:** `validate` checks a value against a subset of JSON Schema and
returns every problem with its path. `ToolRegistry.call` runs the checks in
the order shown and returns a `ToolOutcome`, and `definitions(strict=True)`
adds the strict flag.

## 3. Descriptions are prompts

A call can pass every check and still be to the wrong tool. The model
chooses a tool by reading its name and description, so the description is a
prompt. Label a wall of drawers "stuff", "things" and "items", and even a
careful person opens the wrong one; label them "Policies (PTO, travel, VPN).
Not customers", and nobody hesitates.

To measure this without a real model, a stand-in picks the tool whose name
and description share the most words with the request. For "billing status
for customer 1042", the vague descriptions share no words with any tool, a
three-way tie, so the stand-in takes the first. The precise description of
`lookup_record` shares "billing", "status" and "customer", and wins.

![Vague descriptions pick the right tool for 2 of 6 requests; precise descriptions of the same tools get all 6 right](figures/primer.agents.tools.selection_accuracy.svg)

**Reading it:** two bars, one per set of descriptions, each scored on the
same six labelled requests. The vague set gets 2 of 6, and those two are
the requests that belong to the first tool, which the stand-in picks
whenever it can't tell the tools apart. The precise set says *what* each
tool is for, *when* to use it and *when not to*, and gets 6 of 6. Only the
descriptions changed.

A real model reads far better than a word count, but it chooses from the
same text, so the same fixes help it: precise descriptions, few parameters,
an `enum` wherever the values form a closed set, and error messages that say
how to fix the call.

**In code:** `pick_tool` is the stand-in, and `selection_accuracy` scores it
on `VAGUE_TOOLS` and `PRECISE_TOOLS`.

## 4. Fewer, higher-level tools

Every choice the model makes is a chance to choose wrong, so the number of
calls a job takes matters too. Asking an assistant
to "book my trip to Denver" is safer than dictating five separate forms
(find a flight, hold a seat, find a hotel, reserve a room, add it to the
calendar), because every hand-off is another chance to drop something.

```mermaid
flowchart LR
  subgraph Low["Five low-level calls"]
    a1[get_customer] --> a2[get_price_list] --> a3[compute_tax] --> a4[render_pdf] --> a5[send_email]
  end
  subgraph High["One high-level call"]
    b1[create_invoice]
  end
```

**Reading it:** with five low-level tools, the model has to pick each one in
the right order and carry each output into the next input itself. With one
high-level tool, the same work is a single call, and the sequencing lives in
ordinary tested code.

If each call is right with probability $p$, independently of the others, a
chain of $n$ calls all succeed with probability:

$$
P(\text{all succeed}) = p^{\,n}
$$

**Symbols**

| Symbol | Meaning |
|---|---|
| $p$ | probability that one call is chosen and filled in correctly |
| $n$ | number of calls in the chain |

**In words:** multiply the per-call success rate by itself once per call.

**On the example:** with $p = 0.97$ and $n = 5$, $0.97^5 = 0.859$, so about
1 run in 7 fails. With one high-level call it's $0.97^1 = 0.97$.

**In Python:**

```python
p = 0.97
# five calls that must all succeed
round(p ** 5, 3)  # → 0.859
# about 1 run in 7 fails
round(1 - p ** 5, 2)  # → 0.14
# one high-level call
p ** 1  # → 0.97
```

![Success falls with every extra call: at 99% per call twenty calls succeed about 82% of the time, at 90% ten calls barely a third](figures/primer.agents.tools.compounding.svg)

**Reading it:** the x-axis is how many calls the job takes, and the y-axis
is the chance the whole job succeeds. Each curve is one per-call
reliability. Even at 99% per call, twenty calls succeed only about 82% of
the time; at 90% per call, ten calls succeed barely a third of the time.

**In code:** `chain_success` evaluates $p^{\,n}$.

## 5. Too many tools: load only the relevant ones

Merging tools shortens the chains, but a real product still collects many
tools, and a long list makes each choice harder. Selection accuracy drops as
the list grows past a few dozen tools, and every definition costs input
tokens on every call. So send only the tools a request needs, the way you
would hand a plumber the five tools for today's job rather than send them
into a warehouse of 500.

Take a catalogue of 30 tools and the request "I forgot my password and I'm
locked out". The request is embedded, compared with every tool description,
and only the top 5 tools go to the model, with `reset_password` first.

```mermaid
flowchart LR
  R[User request] --> E[Embed request]
  C[(Tool catalogue<br/>30 descriptions,<br/>embedded once)] --> S
  E --> S[Cosine similarity<br/>to every tool]
  S --> K[Top 5 tools]
  K --> M[Model sees<br/>only these 5]
```

**Reading it:** this is the same search as in
`primer.ml.embeddings.retrieval`, with tool descriptions as the documents. The catalogue is embedded once, ahead
of time; each request costs one embedding and a lookup of the top matches.

![Each request is similar to only a small cluster of related tools and dim against the rest of the catalogue](figures/primer.agents.tools.tool_similarity.svg)

**Reading it:** each row is a user request and each column a tool from the
catalogue, and brighter cells are more similar. Each request lights up a
small cluster of related tools and stays dark everywhere else. Sending only
that cluster keeps the model's choice small however large the catalogue
grows.

Two other designs do the same job: route the request to a sub-agent that
holds only the relevant tools, or use a provider's built-in tool search.

**In code:** `select_tools` ranks the catalogue against the request and
returns the top matches.

## 6. Safety: idempotency, dry runs, approval, least privilege

With the right tool and valid arguments, a call can still do harm. It can
run twice, it can be impossible to undo, or it can come from a model that
was tricked. Four safeguards deal with these.

**Idempotency.** An operation is **idempotent** when doing it twice has the
same effect as doing it once, like pressing a lift button twice. A refund
is not, and agents retry. Suppose a refund call times out *after* the bank
has processed it: the agent cannot tell, so it retries. The fix is an
**idempotency key**, a unique id the caller sends with each write. The
registry stores the result under that key, and a call that repeats the key
gets the stored result back.

```mermaid
sequenceDiagram
  participant A as Agent
  participant R as Registry
  participant B as Bank
  A->>R: refund C-100 $20 (key req-1)
  R->>B: refund
  B-->>R: done
  R--xA: network timeout (reply lost)
  A->>R: retry: refund C-100 $20 (key req-1)
  R-->>A: duplicate: "refunded 20.00 to C-100" (no second refund)
```

**Reading it:** the first reply is lost on its way back (the crossed
arrow), so the agent cannot know the refund happened, and it retries. The
retry carries the same key, so the registry returns the stored result
instead of refunding again. Without the key, the customer gets $40.

**Dry run.** A dry run passes every check, then describes the action
instead of doing it. It serves previews and shadow-mode rollouts
(`primer.agents.deployment`).

**Human approval.** Some actions should wait for a person, as spending over
a limit waits for a manager's signature. Irreversible actions, or ones over
a threshold such as refunds over $500, park as `awaiting_approval` until a
person decides.

```mermaid
sequenceDiagram
  participant M as Model
  participant R as Registry
  participant H as Human approver
  M->>R: refund C-100 $900
  R->>H: approve refund_payment {amount: 900}?
  alt approved
    H-->>R: yes
    R-->>M: ok: refunded
  else declined
    H-->>R: no
    R-->>M: declined: do not retry, tell the user
  end
```

**Reading it:** the approval gate sits between the model's request and the
action, and both branches return text the model can act on. The declined
message says "do not retry" because otherwise a persistent model asks
again.

**Least privilege.** Give each agent credentials for its own job only, the
way a valet key starts the car but won't open the boot. Credentials come as
**scopes**, named permissions such as `payments:write`. An agent that reads
tickets holds `tickets:read`, so even if a malicious email tricks it, it
*cannot* issue refunds.

**In code:** a `Tool` carries the scopes it needs, whether it reads, writes
or acts irreversibly, and an optional approval rule. `ToolRegistry.call`
enforces them, given the caller's credentials, an idempotency key, a
dry-run flag and an approver.

## In 20 seconds
- The model only writes a request (`tool_use`). Your code checks it, runs it and returns a `tool_result`.
- Check both shape (JSON Schema) and meaning (business rules), and return every problem with its fix.
- The model picks a tool by its description: say what the tool does, when to use it, and when not to.
- A job of $n$ calls, each right with probability $p$, succeeds with probability $p^n$, so fewer, higher-level tools are more reliable.
- With a large catalogue, send the model only the tools relevant to each request.
- Writes carry idempotency keys, irreversible or high-value actions wait for human approval, and each agent's credentials allow only its job.

## Self-test questions

**Q: How do you design tools so the model picks the right one with correct arguments?**
A: Give each tool one clear job and a description that says what it does,
when to use it and when not to. Keep parameters few, use `enum`s for closed
sets, and put the format in the schema description ("date as YYYY-MM-DD").
Prefer one high-level tool over several low-level ones. Validate every call
and return actionable errors so the model can self-correct. Measure tool
selection accuracy in evals, and load tools dynamically when there are many.

**Q: The schema says `customer_id` must match `^C-\d+$`. Is that enough validation?**
A: No. The schema checks shape, not truth. `C-999` is well-formed and may not
exist. Add a business-rule check before acting, and return an error that
tells the model how to find the right id.

**Q: A refund call timed out. Should the agent retry?**
A: Only if the call is idempotent. Send an idempotency key with every write,
and the retry returns the original result instead of refunding twice.

**Q: Why not give the agent one admin credential for everything?**
A: Blast radius. If the agent is tricked (for example by prompt injection in
a document it reads), it can do anything the credential allows. Scope
credentials per tool and per agent, and put irreversible actions behind human
approval.

## The papers behind this lesson

- **Schick et al., *Toolformer: Language Models Can Teach Themselves to Use Tools* (2023).**
  https://arxiv.org/abs/2302.04761. It showed a language model can learn *when*
  to call an API, *which* one and *with what arguments*, by keeping only the
  self-generated calls that made its predictions better, which is the idea behind
  tool calling being trained into today's models.
  [annotated companion](../../papers/toolformer.html)

## Further reading
- Anthropic, *Writing effective tools for agents*: https://www.anthropic.com/engineering/writing-tools-for-agents
- Anthropic, *Building effective agents* (appendix on prompt-engineering tools): https://www.anthropic.com/engineering/building-effective-agents
- Claude tool use overview: https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview
- Implementing tool use: https://docs.claude.com/en/docs/agents-and-tools/tool-use/implement-tool-use
- *Understanding JSON Schema*: https://json-schema.org/understanding-json-schema
- Stripe, idempotent requests: https://docs.stripe.com/api/idempotent_requests
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable


def _json_type(value: Any) -> str:
    # bool is a subclass of int in Python, so test it first: True must not pass as an integer.
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "null" if value is None else type(value).__name__


def _type_ok(actual: str, expected: str) -> bool:
    # JSON Schema's "number" accepts integers too.
    return actual == expected or (expected == "number" and actual == "integer")


def validate(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Check `value` against a JSON Schema subset. Returns every problem found."""
    expected = schema.get("type")
    if expected and not _type_ok(_json_type(value), expected):
        # Stop here: the other checks assume the right type.
        return [f"{path}: expected {expected}, got {_json_type(value)} ({value!r})"]

    errors: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        # Listing the allowed values lets the model fix the call in one retry.
        errors.append(f"{path}: must be one of {schema['enum']}, got {value!r}")
    if isinstance(value, str) and "pattern" in schema and not re.search(schema["pattern"], value):
        hint = schema.get("description", f"a string matching {schema['pattern']}")
        errors.append(f"{path}: {value!r} does not match the required format. Expected: {hint}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: must be >= {schema['minimum']}, got {value}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: must be <= {schema['maximum']}, got {value}")
    if isinstance(value, dict):
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}: missing required field '{name}'")
        props = schema.get("properties", {})
        for name, sub in value.items():
            if name in props:
                errors += validate(sub, props[name], f"{path}.{name}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected field '{name}' (allowed: {', '.join(props)})")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors += validate(item, schema["items"], f"{path}[{i}]")
    return errors


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., Any]
    # Business-rule check run after the schema passes: "does this customer exist?"
    # Returns a list of actionable problems; empty means OK.
    check: Callable[[dict[str, Any]], list[str]] | None = None
    # Permissions the caller's credentials must include to run this tool.
    scopes: frozenset[str] = frozenset()
    # read | write | irreversible. Writes get idempotency; irreversible ones need approval.
    side_effect: str = "read"
    # Extra rule for when a human must approve, e.g. lambda args: args["amount"] > 500.
    needs_approval: Callable[[dict[str, Any]], bool] | None = None

    def requires_approval(self, args: dict[str, Any]) -> bool:
        return self.side_effect == "irreversible" or bool(self.needs_approval and self.needs_approval(args))


@dataclass
class ToolOutcome:
    """What a tool call produced. `content` is what the model will read."""

    status: str  # ok | invalid | unknown_tool | denied | rejected | awaiting_approval | declined | dry_run | duplicate
    content: str

    @property
    def is_error(self) -> bool:
        return self.status not in ("ok", "dry_run", "duplicate")


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)
    # idempotency key -> the outcome of the call that first used it.
    _completed: dict[str, ToolOutcome] = field(default_factory=dict)

    def register(
        self, name: str, description: str, input_schema: dict[str, Any], fn: Callable[..., Any], **options: Any
    ) -> Tool:
        tool = Tool(name, description, input_schema, fn, **options)
        self.tools[name] = tool
        return tool

    def definitions(self, strict: bool = False) -> list[dict[str, Any]]:
        """Tool definitions in the Anthropic Messages API shape.

        `strict=True` adds `"strict": true`, which makes the API guarantee the
        model's arguments validate against the schema. Strict schemas must
        forbid unknown fields (`additionalProperties: false`).
        """
        defs = []
        for t in self.tools.values():
            d: dict[str, Any] = {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            if strict:
                d["input_schema"] = {**t.input_schema, "additionalProperties": False}
                d["strict"] = True
            defs.append(d)
        return defs

    def call(
        self,
        name: str,
        args: dict[str, Any],
        credentials: frozenset[str] = frozenset(),
        idempotency_key: str | None = None,
        dry_run: bool = False,
        approver: Callable[[str, dict[str, Any]], bool] | None = None,
    ) -> ToolOutcome:
        """Run a tool call from the model through every check, in order."""
        tool = self.tools.get(name)
        if tool is None:
            return ToolOutcome("unknown_tool", f"Unknown tool '{name}'. Available: {', '.join(sorted(self.tools))}.")
        # Permissions first: a caller without the scope learns nothing more about the tool.
        missing = sorted(tool.scopes - credentials)
        if missing:
            return ToolOutcome(
                "denied",
                f"Permission denied: {name} needs scope {missing[0]!r}; this agent has {sorted(credentials)}.",
            )
        problems = validate(args, tool.input_schema)
        if problems:
            return ToolOutcome("invalid", f"Invalid arguments for {name}:\n" + "\n".join(f"- {p}" for p in problems))
        # Well-formed is not the same as valid: "C-999" matches the pattern but may not exist.
        problems = tool.check(args) if tool.check else []
        if problems:
            return ToolOutcome("rejected", "\n".join(problems))
        # Dry run: every check has passed, so describe exactly what would happen, then stop.
        if dry_run:
            planned = json.dumps(args, sort_keys=True)
            return ToolOutcome("dry_run", f"Dry run: would call {name} with {planned}. Nothing was changed.")
        # Human approval for irreversible or high-value actions. With no approver
        # attached, the call parks instead of running; a person can approve it later.
        if tool.requires_approval(args):
            if approver is None:
                return ToolOutcome("awaiting_approval", f"{name} needs human approval. It has been queued; tell the user.")
            if not approver(name, args):
                return ToolOutcome("declined", f"A human declined {name}. Do not retry; tell the user it was not approved.")
        # Idempotency: a retry carrying the same key returns the first result instead of acting again.
        if idempotency_key is not None and idempotency_key in self._completed:
            return ToolOutcome("duplicate", self._completed[idempotency_key].content)
        result = tool.fn(**args)
        outcome = ToolOutcome("ok", result if isinstance(result, str) else json.dumps(result, default=str))
        if idempotency_key is not None:
            self._completed[idempotency_key] = outcome
        return outcome


# ---------------------------------------------------------------------------
# Tool descriptions are prompts: the model chooses from their words alone
# ---------------------------------------------------------------------------

# The payment tool from the worked example in section 2. `additionalProperties:
# False` is what turns a typo like "ammount" into an error instead of a field
# that is silently ignored.
PAYMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "amount": {"type": "number", "minimum": 0.01},
        "currency": {"type": "string", "enum": ["EUR", "GBP", "USD"]},
        "pay_on": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$", "description": "date as YYYY-MM-DD"},
    },
    "required": ["amount", "currency"],
    "additionalProperties": False,
}

_OBJ = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}

VAGUE_TOOLS = [
    {"name": "search_docs", "description": "Searches for information.", "input_schema": _OBJ},
    {"name": "lookup_record", "description": "Looks up information about things.", "input_schema": _OBJ},
    {"name": "find_item", "description": "Finds information in the system.", "input_schema": _OBJ},
]

PRECISE_TOOLS = [
    {
        "name": "search_docs",
        "description": (
            "Search the company policy knowledge base: how-to guides and rules for PTO, vacation, travel, "
            "flights, expenses and VPN. Use for questions about a policy. Not for customers or tickets."
        ),
        "input_schema": _OBJ,
    },
    {
        "name": "lookup_record",
        "description": (
            "Get one customer record by customer id or name: plan, billing status and balance. Use for "
            "questions about a specific customer. Not for policies or tickets."
        ),
        "input_schema": _OBJ,
    },
    {
        "name": "find_item",
        "description": (
            "Find support tickets by ticket number or status (open, closed, escalated). Use for questions "
            "about tickets. Not for policies or customers."
        ),
        "input_schema": _OBJ,
    },
]

# (user request, the tool that should handle it)
LABELED_REQUESTS = [
    ("what is the travel policy for flights", "search_docs"),
    ("how many PTO days do I get", "search_docs"),
    ("billing status for customer 1042", "lookup_record"),
    ("balance for customer acme", "lookup_record"),
    ("is ticket 553 still open", "find_item"),
    ("escalated tickets this week", "find_item"),
]


def pick_tool(request: str, tool_defs: list[dict[str, Any]]) -> str:
    """A stand-in model: choose the tool whose name + description shares the most words with the request.

    Ties go to the first tool listed, which is how an arbitrary guess looks.
    """
    from primer.common.text import tokenize

    words = set(tokenize(request))

    def overlap(d: dict[str, Any]) -> int:
        return len(words & set(tokenize(d["name"].replace("_", " ") + " " + d["description"])))

    return max(tool_defs, key=overlap)["name"]


def selection_accuracy(tool_defs: list[dict[str, Any]], labeled: list[tuple[str, str]]) -> tuple[int, int]:
    """(correct picks, total requests)."""
    return sum(pick_tool(q, tool_defs) == want for q, want in labeled), len(labeled)


def chain_success(p: float, n_calls: int) -> float:
    """Probability that n independent calls, each succeeding with probability p, all succeed: p**n."""
    return p**n_calls


# ---------------------------------------------------------------------------
# Dynamic tool loading: send only the tools that fit the request
# ---------------------------------------------------------------------------

_CATALOG_SPEC = [
    ("reset_password", "Reset a forgotten password or unlock a locked account."),
    ("enroll_mfa", "Enroll a user in two-factor authentication with an authenticator app."),
    ("check_vpn_status", "Check whether the VPN tunnel for remote access is up."),
    ("renew_device_certificate", "Renew an expired device certificate on a laptop."),
    ("request_laptop", "Request new laptop hardware or a monitor."),
    ("report_phishing", "Report a suspicious phishing email to security."),
    ("fix_printer", "Troubleshoot a broken printer or order toner."),
    ("submit_expense", "Submit an expense with receipts for reimbursement, including car mileage."),
    ("book_travel", "Book flights and hotels for a business trip."),
    ("get_per_diem", "Get the daily per-diem rate for travel meals."),
    ("request_pto", "Request vacation or PTO days off."),
    ("get_pto_balance", "Get the remaining PTO and vacation balance."),
    ("report_sick_day", "Report a sick day or sick leave."),
    ("request_parental_leave", "Request parental leave for a birth or adoption."),
    ("get_payslip", "Get a payslip or payroll statement."),
    ("get_salary_band", "Get the salary band and bonus target for a role."),
    ("start_onboarding", "Start onboarding for a new hire and schedule orientation."),
    ("list_invoices", "List vendor invoices for a quarter."),
    ("list_payments", "List payments sent to vendors."),
    ("reconcile_invoices", "Reconcile vendor invoices against payments and list mismatches."),
    ("create_invoice", "Create and send an invoice to a customer."),
    ("send_email", "Send an email from the shared inbox."),
    ("search_policies", "Search company policies and guidelines."),
    ("open_ticket", "Open an IT support ticket for an issue."),
    ("close_ticket", "Close a resolved IT support ticket."),
    ("get_org_chart", "Look up who reports to whom in the org chart."),
    ("book_meeting_room", "Book a meeting room in an office building."),
    ("order_office_supplies", "Order office supplies such as pens and notebooks."),
    ("translate_text", "Translate text between languages."),
    ("summarize_document", "Summarize a long document."),
]

TOOL_CATALOG: list[dict[str, Any]] = [
    {"name": n, "description": d, "input_schema": {"type": "object", "properties": {}}} for n, d in _CATALOG_SPEC
]


def select_tools(request: str, catalog: list[dict[str, Any]], k: int = 5) -> list[dict[str, Any]]:
    """Return the k tool definitions whose descriptions are most similar to the request.

    Same machinery as retrieval (`primer.agents.rag`): embed the request, embed
    each description once, rank by cosine similarity. Here it retrieves tools
    instead of documents.
    """
    from primer.common.embedder import ConceptEmbedder

    emb = ConceptEmbedder()
    tool_vecs = emb.encode([d["name"].replace("_", " ") + ". " + d["description"] for d in catalog])
    scores = tool_vecs @ emb.encode(request)  # unit vectors, so dot product = cosine similarity
    order = sorted(range(len(catalog)), key=lambda i: -scores[i])
    return [catalog[i] for i in order[:k]]


# ---------------------------------------------------------------------------
# Figures and walkthrough
# ---------------------------------------------------------------------------

FIGURE_REQUESTS = [
    "I forgot my password and I'm locked out",
    "submit my car mileage expense",
    "how many vacation days are left",
    "suspicious email asking for my login",
    "reconcile vendor invoices for Q3",
]


def figures() -> dict:
    """Plots computed from this lesson's own code (matplotlib imported here, not at module level)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from primer.common.embedder import ConceptEmbedder

    figs = {}

    n = np.arange(1, 21)
    fig, ax = plt.subplots(figsize=(7, 4))
    for p in (0.90, 0.95, 0.97, 0.99):
        ax.plot(n, [chain_success(p, int(k)) for k in n], "o-", ms=3, label=f"p = {p:.2f} per call")
    ax.set(xlabel="calls in the chain (n)", ylabel="P(every call succeeds) = p^n", ylim=(0, 1.02),
           title="Reliability compounds: fewer, higher-level tools")
    ax.legend()
    figs["compounding"] = fig

    vague, total = selection_accuracy(VAGUE_TOOLS, LABELED_REQUESTS)
    precise, _ = selection_accuracy(PRECISE_TOOLS, LABELED_REQUESTS)
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["vague descriptions", "precise descriptions"], [vague, precise], color=["#d98c7a", "#7ab87a"])
    ax.bar_label(bars, labels=[f"{vague}/{total}", f"{precise}/{total}"])
    ax.set(ylabel="requests routed to the right tool", ylim=(0, total + 0.8), title="Same model, same requests, new descriptions")
    figs["selection_accuracy"] = fig

    emb = ConceptEmbedder()
    tool_vecs = emb.encode([d["name"].replace("_", " ") + ". " + d["description"] for d in TOOL_CATALOG])
    sims = emb.encode(FIGURE_REQUESTS) @ tool_vecs.T  # (requests, tools)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    im = ax.imshow(sims, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(TOOL_CATALOG)), [d["name"] for d in TOOL_CATALOG], rotation=75, ha="right", fontsize=7)
    ax.set_yticks(range(len(FIGURE_REQUESTS)), FIGURE_REQUESTS, fontsize=8)
    ax.set_title("Cosine similarity: each request lights up a few tools")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    figs["tool_similarity"] = fig
    return figs


def demo() -> None:
    from primer._show import banner, say, table, takeaway

    banner("1. A tool definition is a form template (JSON Schema)")
    reg = ToolRegistry()
    reg.register("add", "Add two integers.", {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}, lambda a, b: a + b)
    print(json.dumps(reg.definitions()[0], indent=2))
    print()
    say("The model fills the form in; your code decides whether to act on it.")
    for args in ({"a": 2, "b": 3}, {"a": "two"}):
        o = reg.call("add", args)
        print(f"  add({args}) -> [{o.status}] {o.content}")
    print()

    banner("2. Validation: every problem at once, each with its fix")
    bad = {"ammount": 5, "currency": "usd", "pay_on": "next friday"}
    print(f"  arguments: {bad}")
    for e in validate(bad, PAYMENT_SCHEMA):
        print(f"  - {e}")
    print()
    takeaway("'invalid input' makes the model guess. A list of fixes gets it right in one retry.")

    banner("3. Descriptions are prompts")
    table(
        ["request", "expected", "vague picks", "precise picks"],
        [(q, want, pick_tool(q, VAGUE_TOOLS), pick_tool(q, PRECISE_TOOLS)) for q, want in LABELED_REQUESTS],
    )
    v, t = selection_accuracy(VAGUE_TOOLS, LABELED_REQUESTS)
    p, _ = selection_accuracy(PRECISE_TOOLS, LABELED_REQUESTS)
    say(f"Vague: {v}/{t} correct. Precise: {p}/{t}. Only the descriptions changed.")

    banner("4. Fewer, higher-level tools: p^n")
    table(["per-call p", "1 call", "5 calls", "10 calls", "20 calls"],
          [(pp, *(chain_success(pp, n) for n in (1, 5, 10, 20))) for pp in (0.90, 0.95, 0.97, 0.99)], floatfmt=".3f")

    banner("5. Dynamic tool loading: 30 tools in the catalogue, 5 sent")
    for q in FIGURE_REQUESTS[:3]:
        print(f"  {q!r:45} -> {[d['name'] for d in select_tools(q, TOOL_CATALOG, k=3)]}")
    print()

    banner("6. Safety: scopes, dry run, approval, idempotency")
    ledger: list = []

    def refund(customer_id: str, amount: float) -> str:
        ledger.append((customer_id, amount))
        return f"refunded {amount:.2f} to {customer_id}"

    pay = ToolRegistry()
    pay.register(
        "refund_payment", "Refund a customer payment.",
        {"type": "object", "properties": {"customer_id": {"type": "string"}, "amount": {"type": "number"}}, "required": ["customer_id", "amount"]},
        refund, scopes=frozenset({"payments:write"}), side_effect="write", needs_approval=lambda a: a["amount"] > 500,
    )
    ok = frozenset({"payments:write"})
    steps = [
        ("ticket-reading agent", dict(credentials=frozenset({"tickets:read"}))),
        ("dry run", dict(credentials=ok, dry_run=True)),
        ("$900, no approver", dict(credentials=ok, args={"customer_id": "C-100", "amount": 900})),
        ("$20, key req-1", dict(credentials=ok, idempotency_key="req-1")),
        ("retry $20, key req-1", dict(credentials=ok, idempotency_key="req-1")),
    ]
    for label, kw in steps:
        args = kw.pop("args", {"customer_id": "C-100", "amount": 20})
        o = pay.call("refund_payment", args, **kw)
        print(f"  {label:22} -> [{o.status}] {o.content}")
    print(f"\n  ledger after all of that: {ledger}  (one refund)\n")
    takeaway("Five attempts, one refund: least privilege, rehearsal, approval and idempotency each did their job.")


if __name__ == "__main__":
    demo()
