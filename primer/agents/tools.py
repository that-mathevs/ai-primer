r"""
# Tools: design, validation and safety

Run: `python -m primer.agents.tools`

A language model can only produce text. A **tool** is how that text turns
into an action: looking up a record, refunding a payment, sending an email.
This lesson builds a tool registry from scratch and shows the six things
that decide whether tools work in production: how a call really happens,
validation, descriptions, granularity, how many tools you expose, and safety.

## 1. What a tool call really is

**Everyday picture.** You hire a new assistant who is brilliant but may not
touch anything. When they need something done, they fill in a *request
form* ("refund customer C-100, $20") and hand it to you. **You** decide
whether to carry it out, and you hand back a note saying what happened. The
assistant is the model. The form is a *tool call*. You are your code.

**Tiny worked example.** Here's the whole exchange for "what's 2 + 3?" with
one tool, written as the actual messages:

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

`input_schema` is a **JSON Schema**, a JSON document that describes the
shape other JSON must have: which fields exist, what type each is, and
which are required. It's the "form template" the model fills in.

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

**Reading it:** follow the arrows top to bottom. The model's only arrow
towards the real system goes *through your code*. Nothing the model writes
can touch the real system unless your code decides to act on it. That
middle box, "check the form", is where the rest of this lesson lives.

**The code.** `ToolRegistry.register` stores a Python function with its
name, description and schema; `ToolRegistry.definitions()` produces the
list you send to the model; `ToolRegistry.call()` runs a call through every
check.

**Why it matters.** The separation is the security model. A model can be
wrong or tricked. Your code is where you enforce what's allowed.

## 2. Validate every call before running it

**Everyday picture.** A bank clerk checks a withdrawal slip twice. Is it
filled in properly: every box, a real date, a positive amount? Then does it
make sense: does this account exist, does it have the money? Only then do
they open the drawer.

**Tiny worked example.** The model sends `{"ammount": 5, "currency": "usd"}`
to a payment tool. The validator returns *every* problem at once, each one
saying what a correct value looks like:

```text
$: missing required field 'amount'
$: unexpected field 'ammount' (allowed: amount, currency, pay_on, quantity, tags)
$.currency: must be one of ['EUR', 'GBP', 'USD'], got 'usd'
```

Compare that to `invalid input`. With the first, the model fixes all three
mistakes in one retry. With the second, it guesses.

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

**Reading it:** a call enters at the top and must pass every diamond to reach
*run the tool* at the bottom. Each exit on the side is a named `status` in
`ToolOutcome`, and each returns text the model can act on. The order is
deliberate: permissions come before anything that reveals how the tool works, and
cheap checks come before expensive ones. Approval and idempotency sit last,
right next to the action they protect.

**Why it matters.** "Structured output" and strict mode guarantee the
*shape* of the arguments, not that they're *true*. `"C-999"` matches the
pattern `^C-\d+$` perfectly and is still a customer that doesn't exist.
Schema checks and business-rule checks are both required.

With the Claude API you can add `"strict": true` to a tool definition
(`definitions(strict=True)` here). The API then guarantees the model's
arguments validate against the schema. You still need the business-rule
checks.

## 3. Descriptions are prompts

**Everyday picture.** A wall of drawers labelled "stuff", "things" and
"items". Even a careful person opens the wrong one. Relabel them "Policies
(PTO, travel, VPN). Not customers", and nobody hesitates.

**Tiny worked example.** A stand-in model (`pick_tool`) chooses the tool
whose name and description share the most words with the request. For
"billing status for customer 1042", the vague descriptions share zero words
with every tool (a three-way tie, so it guesses the first). The precise
`lookup_record` description shares "billing", "status" and "customer" and
wins.

![Correct tool picks: vague vs. precise descriptions](figures/primer.agents.tools.selection_accuracy.svg)

**Reading it:** two bars, one per set of descriptions, each out of the same six
labelled requests. With vague descriptions the model gets 2 of 6, and only
the two that happen to belong to the first tool, which it picks every time it
can't tell them apart. With descriptions that say *what* each tool is for,
*when* to use it and *when not to*, it gets 6 of 6. Only the descriptions
changed.

**Why it matters.** Real models are far better readers than word overlap,
but they choose from exactly the same text. Precise descriptions, a few
parameters, `enum`s instead of free text, and error messages that explain
the fix remove a large share of agent errors.

## 4. Fewer, higher-level tools

**Everyday picture.** Asking an assistant to "book my trip to Denver" versus
dictating five separate forms (find flight, hold seat, find hotel, reserve
room, add to calendar). Every hand-off is another chance to drop something.

```mermaid
flowchart LR
  subgraph Low["Five low-level calls"]
    a1[get_customer] --> a2[get_price_list] --> a3[compute_tax] --> a4[render_pdf] --> a5[send_email]
  end
  subgraph High["One high-level call"]
    b1[create_invoice]
  end
```

**Reading it:** on the left the model has to pick five tools in the right
order and carry each output into the next input by hand. On the right the
same work is one call, and the sequencing lives in ordinary tested code.

The chance that a chain of calls all succeed:

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

![Chain success vs. number of calls](figures/primer.agents.tools.compounding.svg)

**Reading it:** the x-axis is how many calls the job takes, and the y-axis is the chance
the whole job succeeds. Each curve is a different per-call reliability.
Even at 99% per call, twenty calls succeed only about 82% of the time, and at
90% per call, ten calls succeed barely a third of the time. Fewer calls is
the cheapest reliability you can buy.

## 5. Too many tools: load only the relevant ones

**Everyday picture.** A warehouse versus a toolbox. You don't send a
plumber into a warehouse of 500 tools; you hand them the five they need for
today's job.

**Tiny worked example.** A catalogue of 30 tools. For "I forgot my password and I'm
locked out", `select_tools` embeds the request, compares it to every tool
description, and sends the model only the top 5, with `reset_password`
first.

```mermaid
flowchart LR
  R[User request] --> E[Embed request]
  C[(Tool catalogue<br/>30 descriptions,<br/>embedded once)] --> S
  E --> S[Cosine similarity<br/>to every tool]
  S --> K[Top 5 tools]
  K --> M[Model sees<br/>only these 5]
```

**Reading it:** this is retrieval, the same machinery as RAG, but the
"documents" are tool descriptions. The catalogue is embedded ahead of time;
per request you embed one string and take the top matches.

![Similarity of requests to tools](figures/primer.agents.tools.tool_similarity.svg)

**Reading it:** each row is a user request and each column a tool from the
catalogue. Brighter cells mean more similar. Each request lights up a small
cluster of related tools and stays dark everywhere else. Sending only
that cluster keeps the model's choice small, which is why accuracy holds
up as the catalogue grows.

**Why it matters.** Selection accuracy drops as the tool list grows past a
few dozen, and every definition costs input tokens on every call. The
alternatives are routing the request to a sub-agent that holds only the
relevant tools, or using a provider's built-in tool search.

## 6. Safety: idempotency, dry runs, approval, least privilege

**Idempotency.** *Everyday picture:* pressing a lift button twice still
takes you to the floor once. An operation is **idempotent** when doing it
twice has the same effect as doing it once. *Worked example:* a refund call
times out *after* the bank processed it, so the agent retries.

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

**Reading it:** the first reply is lost on the way back (the crossed
arrow), so the agent can't know the refund happened and sensibly retries.
Because the retry carries the same key, the registry returns the stored
result instead of refunding again. Without the key, the customer gets $40.

**Dry run.** A rehearsal: run every check, then *describe* the action
instead of doing it. Useful for previews and for shadow-mode rollouts
(`primer.agents.deployment`).

**Human approval.** *Everyday picture:* a manager signs off on spending over
a limit. Irreversible actions, or ones over a threshold such as refunds over
$500, park as `awaiting_approval` until a person decides.

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
action. Both branches return text the model can act on. "Do not retry" in
the declined message matters, because otherwise a persistent model asks again.

**Least privilege.** *Everyday picture:* a valet key starts the car but
won't open the boot. Give each agent credentials (**scopes**, named
permissions such as `payments:write`) for its job only. An agent that reads
tickets holds `tickets:read`, so even if a malicious email tricks it, it
*cannot* issue refunds.

## In 20 seconds
- The model writes a request (`tool_use`). Your code validates it, runs it, and returns a `tool_result`.
- Validate shape (JSON Schema) *and* meaning (business rules). Return every problem, each with its fix.
- Descriptions are prompts: say what the tool does, when to use it, and when not to.
- Prefer fewer, higher-level tools: $p^n$ punishes long chains of calls.
- Past a few dozen tools, load only the relevant ones per request.
- Writes get idempotency keys; irreversible or high-value actions get human approval; credentials get the narrowest scopes.

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
- Stripe, idempotent requests (the canonical explanation): https://docs.stripe.com/api/idempotent_requests
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
    schema = {
        "type": "object",
        "properties": {
            "amount": {"type": "number", "minimum": 0.01},
            "currency": {"type": "string", "enum": ["EUR", "GBP", "USD"]},
            "pay_on": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$", "description": "date as YYYY-MM-DD"},
        },
        "required": ["amount", "currency"],
        "additionalProperties": False,
    }
    bad = {"ammount": 5, "currency": "usd", "pay_on": "next friday"}
    print(f"  arguments: {bad}")
    for e in validate(bad, schema):
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
