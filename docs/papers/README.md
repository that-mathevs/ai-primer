# Writing an annotated paper companion

A companion is one self-contained HTML page, `docs/papers/<slug>.html`,
that walks a landmark paper section by section. Every term and equation
symbol explains itself on hover, the paper's figures are redrawn as diagrams
you can explore, and each idea links to the lesson that builds it in code.
The rules in [`CLAUDE.md`](../../CLAUDE.md) apply in full.

**The reference companion is
[`attention-is-all-you-need.html`](attention-is-all-you-need.html).** Copy
its structure; this guide explains the parts.

## Before you start

1. Find the slug in [`CATALOG.md`](CATALOG.md). It is the single source of
   truth for slugs, titles, primary sources and related lessons. The site nav,
   the "Lessons that build this" box, previous/next links and the papers index
   are all generated from it (via [`assets/catalog.js`](assets/catalog.js), written by
   `make docs`), so you never hand-write them.
2. Read the paper itself, end to end.

## The copyright rule

Companions explain; they never republish.

- **Quotes:** at most one or two short sentences per section, in a
  `blockquote.quote` with a `<cite>` naming the section.
- **Figures:** redraw them yourself as SVG. Never embed the original images.
- **Tables:** reproduce only when the paper grants permission (many arXiv
  papers print a notice) or as a few selected rows, always with attribution
  in a `<caption>`.
- **Equations** can be reproduced (math is not copyrightable), with every
  symbol decoded.
- Link the original at the top (abstract page and PDF), and on every section
  heading (`ar5iv.labs.arxiv.org/html/<id>#S3.SS2` style anchors for arXiv
  papers).
- Say this in an "About this page" box at the top.

## How each section is written

Follow the paper's own section order. Inside each section, climb the ladder
from [`CLAUDE.md`](../../CLAUDE.md) using labels one level below the heading
they sit under, so the outline never skips a level: `<h4>` under a subsection
(`<h3>`), and `<h3 class="rung">` (styled the same) directly under a section
(`<h2>`). `make sitecheck` fails on a skipped level.

1. **Everyday picture**: an analogy with no jargon.
2. **Tiny example**: numbers small enough to check by hand.
3. **Diagram or interactive figure**, immediately followed by
   `<p class="reading"><strong>Reading it:</strong> …</p>`.
4. **The math**: each equation, then its Symbols table, an **In words**
   line, a **With the numbers** line, and an **In Python** block: the same
   numbers computed in plain Python (standard library only), mirroring the
   equation symbol by symbol. It is ordinary code a reader can copy and run:
   explanations on their own comment line above the code, and every line that
   produces a number the page shows ending in `# → value`, which a test checks
   ([`tests/test_math_in_python.py`](../../tests/test_math_in_python.py)):

   ```html
   <p class="inwords"><strong>In Python:</strong></p>
   <pre class="in-python"><code># m_t = β1 m_(t−1) + (1 − β1) g_t
   m = 0.9 * 0.2 + 0.1 * 1
   round(m, 2)  # → 0.28</code></pre>
   ```

   Escape `<`, `>` and `&` as `&lt;`, `&gt;`, `&amp;`; the test reads the
   code the way a browser shows it.
5. **Why it matters today**: what survived, what changed, and a link to the
   lesson.

## Page skeleton

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paper Title, Annotated</title>
<meta name="description" content="One sentence.">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" crossorigin="anonymous">
<link rel="stylesheet" href="../assets/theme.css">   <!-- the site-wide light/dark theme -->
<script src="../assets/theme.js"></script>          <!-- in <head>, so dark never flashes light -->
<link rel="stylesheet" href="assets/papers.css">
<script src="assets/glossary.js"></script>   <!-- shared term definitions -->
<script src="assets/catalog.js"></script>    <!-- nav, lessons box, prev/next -->
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js" crossorigin="anonymous"></script>
<script defer src="assets/papers.js"></script>
</head>
<body data-paper="SLUG">
<div class="layout">
<nav class="toc" aria-label="Contents"><details open><summary>Contents</summary>
  <ol><li><a href="#s1">1 Introduction</a></li> … <li><a href="#glossary">Glossary</a></li></ol>
</details></nav>
<main>
<header class="hero">
  <div class="kicker">An annotated companion · primer</div>
  <h1>Paper Title, annotated</h1>
  <p class="byline">A plain-English, hover-to-explain guide to Authors (Year), …</p>
  <div class="toolbar ui">
    <a class="btn" href="https://arxiv.org/abs/ID">Paper on arXiv</a>
    <a class="btn" href="https://arxiv.org/pdf/ID">PDF</a>
    <button type="button" id="themeBtn" data-theme-toggle>Theme: auto</button>
  </div>
  <div class="permission"><strong>About this page.</strong> …</div>
</header>

<section id="s1">
<h2>1 Introduction <span class="orig">· <a href="https://ar5iv.labs.arxiv.org/html/ID#S1">original</a></span></h2>
<blockquote class="quote">“One short sentence.”<cite>Authors (Year), §1</cite></blockquote>
<h4>Everyday picture</h4> …
</section>

<section id="glossary"><h2>Glossary</h2><dl class="glossary" id="glossary-list"></dl></section>
<footer>Citation, attribution note.</footer>
</main>
</div>
<script>
window.PAPER = { symbols: { … }, blocks: { … }, onReady(P) { … } };
</script>
</body>
</html>
```

`papers.js` adds the site nav bar, the "Lessons that build this" box, the
previous/next pager, the theme toggle, the table-of-contents highlighting and
the glossary list. You write none of them.

## Marking a term

Terms that exist in the shared glossary ([`primer/glossary.py`](../../primer/glossary.py), the single
source of truth, exported to [`assets/glossary.js`](assets/glossary.js)) take a key:

```html
<span class="t" data-t="softmax">softmax</span>
<span class="t" data-t="residual connection">residual connections</span>
```

Keys are the lowercase glossary terms. For a term that only matters to this
paper, write the definition inline. `data-tip-title` also adds it to the
page's glossary:

```html
<span class="t" data-tip-title="Beam search"
      data-tip="Keep the best few partial outputs at each step instead of just one.">beam search</span>
```

If a general term is missing from the shared glossary, add it to
[`primer/glossary.py`](../../primer/glossary.py) rather than defining it inline. In the browser console,
`Papers.missingTerms()` lists any `data-t` key that has no definition.

## Hoverable equations

Write an equation as a `.eq` block (or `.m` for inline math) holding TeX.
Wrap each symbol in `\htmlData{sym=KEY}{…}`, and give plain text in
`data-fallback` for readers without JavaScript:

```html
<div class="eq"
     data-tex="\htmlData{sym=Q}{Q}\htmlData{sym=KT}{K^\top}"
     data-fallback="Q Kᵀ"></div>
<div class="symtable" data-syms="Q,KT"></div>
<p class="inwords"><strong>In words:</strong> “…”</p>
<p class="inwords"><strong>With the numbers:</strong> …</p>
```

Every `KEY` is defined once in `window.PAPER.symbols`, and the same entry
feeds both the hover tooltip and the generated Symbols table:

```js
symbols: {
  Q: { tex: "Q", name: "Queries", meaning: "Every word's query vector stacked as rows…",
       shape: "n × d_k (10 × 64)", example: "the row for “it”…" },
}
```

`\htmlData` is the only command KaTeX is allowed to trust. Keys are plain
identifiers (letters and digits), because `\htmlData` values cannot contain
commas.

## An interactive diagram

Redraw the figure as inline SVG inside `figure.ix`, next to a `.panel`. Each
explorable part is a `<g class="blk KIND" data-block="KEY">`. Parts that share
a key light up together.

```html
<figure class="ix" id="fig1">          <!-- add class "wide" for a large diagram -->
  <div class="ixgrid">
    <svg viewBox="0 0 300 200" role="img" aria-label="What the diagram shows">
      <g class="blk attn" data-block="mha" aria-label="Multi-head attention">
        <rect x="20" y="20" width="140" height="40" rx="7"/>
        <text x="90" y="45" text-anchor="middle">Multi-Head Attention</text>
      </g>
    </svg>
    <div class="panel" aria-live="polite"><p class="hint">Hover or tap a block.</p></div>
  </div>
  <figcaption>Figure N of the paper, redrawn. Based on Authors (Year), Figure N.</figcaption>
</figure>
<p class="reading"><strong>Reading it:</strong> …</p>
```

```js
blocks: {
  mha: { title: "Multi-head attention", body: "<p>…</p>",
         shape: "n × 512", lesson: "primer/ml/attention.html" },
}
```

Block colours come from the class: `attn`, `ff`, `norm`, `emb`, `lin`,
`soft`, `plain`, `pe` (for a circle), and `frame` (a dashed container). Draw
wires with `<path class="wire" … marker-end="url(#arr)">`, defining the
arrow marker in the SVG's `<defs>` as the reference page does. Hover, focus,
tap and the Enter key all work automatically. `papers.js` makes each part a
keyboard-focusable button named after its note (its `aria-label`, else the
block's `title`), and turns the SVG's `role="img"` into a group so screen
readers can reach the parts; the `.panel` announces the note.

Equations need no extra markup for the keyboard either: an equation with
`\htmlData` symbols is one tab stop, and the arrow keys step through its
symbols, showing each tooltip. (KaTeX's visual layer is hidden from screen
readers, which read its MathML copy and the Symbols table instead.) A term's
tooltip link ("Build it in code →") is reached with Tab from the term.

## Plots

`onReady(P)` receives the helpers once the page is wired up.

```js
// Line chart with a hover crosshair; call .update(series) to redraw (e.g. from a slider).
// It is also one tab stop: the arrow keys move the crosshair and write the same readout.
// With two or more series it draws its own legend and gives each series a line pattern
// (solid, dashed, dotted, …) as well as a colour; pass legend: false only if the page draws one.
const chart = P.lineChart(document.getElementById("lr-chart"), {
  series: [{ name: "learning rate", points: [[1, 1e-7], [4000, 7e-4], …] }],   // sorted by x
  xLabel: "training step", yLabel: "learning rate",
  xFmt: v => `${v / 1000}k`, yFmt: v => v.toExponential(1),
  logX: false, readout: document.getElementById("lr-readout"),
});

// Heatmap on a canvas with a diverging colour scale and a hover callback. Give the canvas an
// aria-label saying what it shows; the arrow keys move a cell cursor through the same onHover.
P.heatmap(document.getElementById("pe-canvas"), {
  rows: 100, cols: 128, min: -1, max: 1, value: (r, c) => …,
  onHover(r, c, v) { if (r !== null) readout.textContent = `…`; },
});

// A point the reader places by clicking (a start point, a query) must move from the keyboard too.
// x and y are fractions of the picture (0 to 1, from the left and from the top).
P.keyPoint(canvas, { what: "the starting point", get: () => [x, y], set(x, y) { … } });
```

Wrap plots in `<div class="plot">` (with `.controls` for sliders and a
`p.readout` for the hover text) and follow each with its **Reading it**
paragraph. For small one-off interactions (a sentence playground, a slider
demo), write plain JS in `onReady` using the `.play`, `.sliders`,
`.share-row`, `.share-bar` and `.tok` classes from `papers.css`. Label any
illustrative (non-computed) numbers as illustrative, on the page.

Never tell series apart by colour alone. Line charts get patterns
automatically; for a second series of bars use
`background: var(--hatch) var(--accent)` (stripes over the colour) and say
"striped" in its legend and Reading it paragraph. A `.readout` whose text
changes is announced to screen readers once it settles, so write results
into readouts rather than only into a picture.

## Links

- To a lesson: `<a data-lesson="primer/ml/attention.html">the attention lesson</a>`.
  `papers.js` resolves it for both the built site (`docs/html/papers/`) and
  the source tree ([`docs/papers/`](.)).
- To another companion: `<a href="roformer.html">RoFormer companion</a>`.
- To code in a lesson: `<a data-lesson="primer/agents/rag.html#hyde_search"><code>hyde_search</code></a>`.
  Every function the page names must exist; `make sitecheck` fails on a call
  to one that doesn't.
- A function the page invents as an example (not code in this repository):
  `<code class="example">is_palindrome(s)</code>`, which the check skips.
- Every external URL must be a primary source you are certain exists.

## Done means

- Every section of the paper is covered and climbs the ladder.
- Every equation symbol is hoverable and appears in a Symbols table; every
  equation has In words and With the numbers lines.
- Every figure and diagram is followed by its **Reading it** paragraph.
- The browser console shows no errors, and `Papers.missingTerms()` returns `[]`.
- The page works at phone width (360px) with no horizontal scroll, in light
  and dark themes, and with the keyboard alone (Tab to every term and block).
- Quotes are brief, attributed and marked; figures are redrawn; the original
  is linked throughout.
