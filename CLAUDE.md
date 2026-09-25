# AI Primer: rules for every agent working here

## What this project is

A public, open-source **primer on how modern AI works, built from scratch in
Python**, written as a gift to engineers who want to understand the machinery
rather than call it. Two parts:

- [`primer/ml/`](primer/ml): ML foundations. Neural nets, attention, transformers,
  tokenization, training and inference, losses and metrics, and the
  centerpiece, [`primer/ml/embeddings/`](primer/ml/embeddings).
- [`primer/agents/`](primer/agents): applied AI. Agent loops, tools, MCP, RAG, memory,
  planning, evals, guardrails, cost, observability, deployment.

[`primer/common/`](primer/common) holds the shared toy corpus and the deterministic
[`ConceptEmbedder`](primer/common/embedder.py); [`primer/agents/llm.py`](primer/agents/llm.py) holds the [`LLM`](primer/agents/llm.py) interface with the
offline [`ScriptedLLM`](primer/agents/llm.py) and the real [`ClaudeLLM`](primer/agents/llm.py). The reading order lives in
[`README.md`](README.md) and each package's `__init__.py`.

## The purpose: demystify everything

**We are destroying the priesthood.** Nothing here may depend on already
knowing the jargon, the notation or the folklore. Nothing stays a black box. Every mechanism is built in plain code, **drawn**,
and **explained**, until a reader could rebuild it from memory. When choosing
between one more paragraph and one more diagram, draw the diagram, then
explain it. See [Diagrams](#diagrams).

**Start from zero.** Write every concept as a **ladder**, climbed in this order:

1. **Everyday picture**: an analogy or scene a curious non-engineer follows
   with no jargon. A convolution is a flashlight sweeping a photo, looking
   for one pattern. An RNN is reading a book while keeping a one-page summary
   you rewrite after every word.
2. **Tiny worked example**: real numbers small enough to check by hand (a 5×5
   image, a 3-word sentence), traced step by step.
3. **Diagram**, with its `**Reading it:**` walkthrough.
4. **The math and the code**, now that the reader already knows what the
   symbols mean.
5. **Why it matters in practice**: where it shows up, what breaks without it.

Every section of every lesson climbs the full ladder. [`primer/ml/attention.py`](primer/ml/attention.py)
is the gold standard; every module meets it.

**Every formula is decoded.** Directly under each `$$…$$` block comes:

- a **Symbols** table: every symbol, subscript and operator in the formula
  (`e`, `Σ`, `ᵀ`, `‖x‖`, `log`, `∂`, `i`, `j`, …) with its meaning *in this
  formula* and its shape or range;
- an **In words** line that reads the whole formula aloud as a sentence;
- the formula evaluated on the lesson's tiny worked example, so every symbol
  gets a real number.
- an **In Python** block: the same worked example computed in plain Python
  (standard library only), written to mirror the formula symbol by symbol
  (`sum(...)` for Σ, a loop for Π, variables named after the symbols). It is
  ordinary code a reader can copy and run: each explanation sits on its own
  comment line above the code it explains, and every line that produces a
  number the lesson shows ends in `# → value`.
  [`tests/test_math_in_python.py`](tests/test_math_in_python.py) runs every
  block and checks every `# →` claim ([`tools/inpython.py`](tools/inpython.py)), and fails on a
  formula without a block.

  ````markdown
  **In Python:**

  ```python
  import math
  z = [2.0, 1.0, 0.5]
  # e^(z_i) for each score
  exps = [math.exp(z_i) for z_i in z]
  # each share of the total Σ_j e^(z_j)
  [round(e / sum(exps), 2) for e in exps]  # → [0.63, 0.23, 0.14]
  ```
  ````

Every math term (softmax, dot product, logarithm, gradient, matrix
multiply, …) is explained in plain words the first time a lesson uses it,
with a link to [`primer.notation`](primer/notation.py) for the full treatment. Unexplained
notation is a bug.

## Hard rules

1. **Voice.** Write as a teacher addressing a curious engineer: general,
   timeless, vendor-neutral where the idea is. The repo is public. It never
   refers to hiring, job preparation or any company's evaluation process.
2. **No em dashes, anywhere**: code, comments, docstrings, markdown, HTML.
   Use a colon, a comma, parentheses or a new sentence.
   [`tests/test_house_style.py`](tests/test_house_style.py) enforces it.
3. **Every module is a lesson.** It has a markdown docstring (pdoc renders it)
   with, in order: `# Title`, `Run: python -m <module>`, the idea in plain
   English, formulas in `$$…$$`, diagrams and figures (see
   [Diagrams](#diagrams)), `## In 20 seconds`, `## Self-test questions` (each followed by its answer),
   and `## Further reading`. It ends with a `demo()` that narrates the lesson
   through [`primer._show`](primer/_show.py) and an `if __name__ == "__main__": demo()`.
   [`primer/ml/attention.py`](primer/ml/attention.py) is the reference lesson; match it.
4. **Links are real.** Every URL in Further reading points at a primary
   source you are certain exists: the paper (arXiv), the official docs, the
   canonical post. Omit a link rather than guess one.
5. **Behaviour-driven, test-driven.** Work red → green in vertical slices:
   one scenario, watch it fail, write only the code that passes it, repeat.
   The suite is the specification; see [Tests are the spec](#tests-are-the-spec).
6. **From scratch.** Implementations use NumPy and the standard library, so
   the reader sees every step. `torch`, `sklearn`, `tiktoken` and `anthropic`
   appear only as optional cross-checks (`pytest.importorskip`) or lazy
   imports; the whole repo runs offline with `numpy` alone.
7. **Deterministic and tight.** Seed every random generator. Each test file
   runs in under 5 seconds. No network in tests or demos.
8. **Comments explain why.** Explain the reason a line exists, the shape of
   a tensor, the failure it prevents. Let the code say what it does.
9. **One source of truth.** Shared vocabulary lives in [`primer/common`](primer/common) and
   [`primer/agents/llm.py`](primer/agents/llm.py); import it rather than redefine it. Change a shared
   file only when the change serves every caller, and run the full suite
   after.

## Primary sources come in

Where an idea has a founding paper, the paper comes into the repo. Each
lesson ends its docstring (before Further reading) with
`## The papers behind this lesson`: for each paper, the citation, its
primary-source link, one sentence on what it contributed, and a link to its
annotated companion `docs/papers/<slug>.html`. Companions are interactive
HTML walkthroughs with hover guidance on every term and equation symbol,
built on the shared assets in [`docs/papers/assets/`](docs/papers/assets). The slugs live in
[`docs/papers/CATALOG.md`](docs/papers/CATALOG.md), and how to write a companion is in
[`docs/papers/README.md`](docs/papers/README.md). Companions quote briefly and link to the original.
They never republish a paper's full text.

## Navigable, and always current

A reader can reach anything from anywhere in two clicks, and no map is ever
stale. Navigation is **generated**, never hand-maintained, from three
sources of truth:

| To add… | Edit only | Generated from it |
|---|---|---|
| a lesson | [`primer/curriculum.py`](primer/curriculum.py) | README reading order and [`docs/SELF_TEST.md`](docs/SELF_TEST.md) (`make readme`), package reading lists, site home page, breadcrumbs, previous/next links |
| a big question | `BIG_QUESTIONS` in [`primer/curriculum.py`](primer/curriculum.py) | README's big-questions map and [`docs/BIG_QUESTIONS.md`](docs/BIG_QUESTIONS.md) (`make readme`), the home page's Big questions |
| a term | [`primer/glossary.py`](primer/glossary.py) | the Glossary page, hover definitions on every page (`glossary.js`) |
| a paper | [`docs/papers/CATALOG.md`](docs/papers/CATALOG.md) | papers index, `catalog.js`, companion nav, the home page's paper list |
| a link from prose to code | an `**In code:**` line naming the function or class in backticks (fully dotted if it lives in another module) | a link to its entry on the page, with View Source and its exact lines on GitHub |
| the site's address | `SITE_URL` in [`primer/curriculum.py`](primer/curriculum.py) | the README's links to the live site (absolute, because the built HTML isn't committed) |

Text between `<!-- BEGIN … -->` and `<!-- END … -->` markers is generated.
Edit its source, never the text itself. [`tests/test_navigation.py`](tests/test_navigation.py) fails
when anything drifts: a lesson on disk that isn't in the curriculum, a stale
README, a link to an unknown paper, a lesson missing a required section.
Every new term a lesson introduces gets a glossary entry in the same change.

Every mechanism section ends its code rung with an `**In code:**` line
naming what implements it, so a reader clicks from the explanation straight
to the code. `make sitecheck` fails if a named thing didn't become a link, or
if any link into the repository points at a missing file or line.

Nothing that names code is left unlinked. On the site, the builder links
every code mention it can resolve: public members to their entry, dunder
methods and private classes to their lines, a lesson's own name (its title,
its `python -m` line) to its code. In Markdown, GitHub renders the page but not
docstrings, so every code span naming a module, a committed file, a lesson's
run command or a class defined in one module is a relative link
(`link_code_references`, which the generators use). Write a name from
another module fully dotted. Tests and `make sitecheck` fail on any leftover,
and on any link to a missing file, line or anchor.

Code links are never hardcoded. A site built locally links to the files in
the reader's own checkout (relative paths). The published site, built by
GitHub Actions, links to GitHub at the exact commit it was built from, so
line numbers always match. The repository's address comes from the build
([`tools/docsite.py`](tools/docsite.py): `repo_url`, `code_link`), so a fork's site links to the fork.

## Diagrams

Every lesson is diagram-first. Two kinds, both rendered in the HTML site:

- **Mermaid diagrams** (```` ```mermaid ```` blocks in the docstring) for
  every mechanism, data flow, architecture, state machine and pipeline in
  the lesson: one per mechanism, not one per module.
- **Data figures** plotted from the lesson's own code: curves, heatmaps,
  vector spaces, trade-off sweeps. The module defines
  `figures() -> dict[str, matplotlib.figure.Figure]`, importing matplotlib
  inside the function so the lesson itself still needs only NumPy. Each
  figure is embedded in the docstring as
  `![What it shows](figures/<dotted.module>.<key>.svg)`, where `<key>` is its
  dict key. `make figures` renders them into `docs/figures/`.

**Every diagram and figure is followed by a paragraph that starts with
`**Reading it:**`** and walks the reader through it: what the axes or boxes
are, where to look first, and what the picture proves. A picture without its
explanation is unfinished.

## Tests are the spec

Test files mirror modules: `tests/test_<module>.py`,
`tests/test_emb_<module>.py`, `tests/test_agents_<module>.py`.

- **One scenario per test**: given a context, when one action happens,
  then one observable outcome. Two actions make two tests.
- **Names read as sentences in the domain's words.** A class per behaviour
  area, methods phrased as the scenario:

  ```python
  class TestCausalMasking:
      def test_given_a_causal_mask_future_tokens_receive_zero_attention(self): ...
  ```

  Present tense, no "should". Name the behaviour, never the call graph.
- **Expected values come from an independent source**: the worked example,
  the paper, a hand calculation written as a literal. An assertion that
  recomputes the answer the way the code does proves nothing.
- **Write the reason into the test** with a one-line comment when a rule
  would otherwise look arbitrary.
- **Done means the spec reads well.** Run `make spec`; a stranger reading
  only those lines learns what the module teaches.

## Commands

```bash
make test     # full suite
make spec     # the suite printed as a readable specification
make demos    # run every module's walkthrough
make readme   # regenerate README's reading order from primer/curriculum.py
make figures  # render every figures() into docs/figures/*.svg
make docs     # readme + figures + the HTML site (nav, hover glossary, papers) into docs/html
make sitecheck  # crawl the built site; fails on any broken internal link
make links    # check every external URL (needs network; run before publishing)
python -m primer.ml.attention        # any single lesson
```

## Definition of done for a module

- It has its entry in [`primer/curriculum.py`](primer/curriculum.py), and every new term it
  introduces has an entry in [`primer/glossary.py`](primer/glossary.py).

- Its scenarios were written first and watched fail.
- `make test` passes and `make spec` reads as a clear description.
- `python -m <module>` runs cleanly and teaches something on its own.
- Every mechanism has a mermaid diagram, the lesson has data figures, and
  each one is followed by its `**Reading it:**` explanation.
- Every mechanism section has an `**In code:**` line, and `make sitecheck` passes.
- Every section climbs the ladder, and every formula has its Symbols table,
  In words line and worked numbers.
- The docstring has every section from rule 3. Its self-test questions reach
  [`docs/SELF_TEST.md`](docs/SELF_TEST.md) through `make readme`; that file is generated.
