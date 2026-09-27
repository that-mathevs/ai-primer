---
name: writing-a-paper-companion
description: Write or revise an annotated paper companion, docs/papers/<slug>.html, for a paper in docs/papers/CATALOG.md, and link it from the lessons that cite it. Use when a catalog paper has no companion yet, or an existing companion needs fixing.
---

# Writing a paper companion

A companion is a **lesson about one paper**: a reader who finishes it could
rebuild the paper's core idea from memory, not just summarise it. How a page
is built lives in [`docs/papers/README.md`](../../../docs/papers/README.md);
voice, the ladder and formula decoding live in
[`CLAUDE.md`](../../../CLAUDE.md). This skill is the order of work and the
bar for done. Work each step to its criterion before starting the next.

## 1. Claim the paper

Find its row in [`docs/papers/CATALOG.md`](../../../docs/papers/CATALOG.md): slug, title, primary source,
lessons.

**Done when** you hold the slug, the arXiv id or DOI, and the list of lessons.

## 2. Read the lessons that cite it

For each lesson in the row, read its `## The papers behind this lesson` entry
and every section that teaches the paper's idea. Note the functions and
classes that implement it: the companion links to them
(`<a data-lesson="primer/ml/gans.html#train_gan"><code>train_gan</code></a>`),
and every name it links must exist.

**Done when**, for each of the paper's main ideas, you can name the lesson
section and the code that builds it, or know that nothing does.

## 3. Read the paper, end to end

Fetch the full text: `https://ar5iv.labs.arxiv.org/html/<id>` for arXiv
papers, else the publisher's page or PDF. Every number, claim and equation on
the page comes from the paper or from your own worked example, never from
memory. A detail you cannot confirm in the source stays off the page.

**Done when** you have the paper's section outline with its ar5iv anchors
(`#S3.SS2`), every equation you will decode, the one or two figures worth
redrawing, and each headline result with the section it comes from.

## 4. Copy the reference

Read [`docs/papers/README.md`](../../../docs/papers/README.md) in full, then the reference companion,
[`docs/papers/attention-is-all-you-need.html`](../../../docs/papers/attention-is-all-you-need.html). Start your page from its
skeleton: `<head>`, hero with toolbar and the **About this page** box, table of
contents, the empty glossary section, footer, and `window.PAPER`.

## 5. Write the page

Follow the paper's own section order. Every section climbs the ladder:
everyday picture, tiny example, diagram, the math, why it matters today.

- **Every equation** has `\htmlData` symbols, a `.symtable`, an **In words**
  line, a **With the numbers** line and an **In Python** block (standard
  library, each shown number on a line ending `# → value`).
- **Every figure** is redrawn by you: an explorable SVG (`figure.ix` with
  `data-block` parts) or a plot from `onReady`, followed by its
  `<p class="reading"><strong>Reading it:</strong>`. At least one per page.
- **Every term** a reader might not know is a `data-t` key. A general term
  missing from [`primer/glossary.py`](../../../primer/glossary.py) gets an entry there, in the section it
  belongs to, pointing at the lesson that teaches it; a term only this paper
  uses gets an inline `data-tip`.
- **Quotes** are one or two sentences per section at most, in
  `blockquote.quote` with a `<cite>`. Numbers you made up are labelled
  illustrative on the page.
- House style: colons, commas and parentheses where an em dash would go;
  prices as `\$2`; the teacher's voice from [`CLAUDE.md`](../../../CLAUDE.md).

**Done when** every section of the paper has its ladder, and every equation,
figure and term meets the list above.

## 6. Link it from the lessons

In each lesson whose papers section cites this paper, end that entry with
a link climbing from the built lesson page to the site root:
`[Annotated companion](../../papers/<slug>.html)` for a module directly under
[`primer/ml/`](../../../primer/ml) or [`primer/agents/`](../../../primer/agents), and
`[Annotated companion](../../../papers/<slug>.html)` one package deeper
([`primer/ml/embeddings/`](../../../primer/ml/embeddings), [`primer/ml/generative/`](../../../primer/ml/generative)). A lesson the page links
must be in the paper's catalog row, because the row builds the page's
"Lessons that build this" box.

## 7. Check it

```bash
python -m pytest -q tests/test_navigation.py tests/test_math_in_python.py tests/test_house_style.py tests/test_glossary.py
make docs && make sitecheck && make browsercheck
```

`make browsercheck` loads every page at phone width and fails on a script
error, an unlabelled control, and any term, symbol or diagram part the page
uses without explaining. Then look at each figure yourself, at phone width,
in both themes, and read the PNGs it prints:

```bash
python tools/screenshot.py papers/<slug>.html#fig1 --theme light
python tools/screenshot.py papers/<slug>.html#fig1 --theme dark
```

Fix whatever a reader would stumble on: text over a picture, a label too
small to read, a table cut off, a colour that vanishes in one theme.

**Done when** the tests and all three `make` targets pass, and you have looked
at every figure on the page at phone width in both themes.

## 8. Commit

One commit per companion, on your own branch. The message says what a reader
can now do, in the repository's style:
`GAN companion: the two-player game, the optimal discriminator derived, and the ring of eight Gaussians collapsing to one mode`.

## Working as one of several agents

Companions are often written by several agents at once, each in its own
worktree, merged later with `shipping-the-primer`. When you are one of them:

- Branch from the latest `main`, one branch per job: `git checkout -b companion/<name> main`.
  Commit, but leave pushing and merging to whoever merges.
- Grep [`primer/glossary.py`](../../../primer/glossary.py) before adding a term: another page may have
  added it already, and a term defined twice fails the suite.
- The scratchpad is shared: keep scratch files in a subfolder named after
  your branch.
- A bug outside your pages goes in your report, not in your commits.
- Report, briefly: branch, one line per companion (sections covered, figures
  drawn), glossary terms added, lessons linked, check results, and anything
  you could not confirm in the source, including places the paper
  contradicts itself.
