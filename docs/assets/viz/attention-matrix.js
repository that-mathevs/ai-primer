/*
 * attention-matrix.js: the lesson's sentence as a live attention heatmap.
 *
 * Rows are the word doing the looking (its query), columns the words looked
 * at (their keys). The weights are softmax(Q Kᵀ / (√d_k · temperature)) with
 * an optional causal mask, computed here from the Q and K that
 * primer/ml/attention.py exports; tests/test_attention.py runs attentionWeights
 * against scaled_dot_product_attention, so the widget can't drift from the lesson.
 */
(function () {
  // One row of weights per query. Temperature 1 is the lesson's formula exactly.
  function attentionWeights(Q, K, { causal = false, temperature = 1 } = {}) {
    const divisor = Math.sqrt(K[0].length) * temperature;
    return Q.map((q, i) => {
      // A future word scores -Infinity, so e^score is exactly 0 after softmax.
      const scores = K.map((k, j) =>
        causal && j > i ? -Infinity : q.reduce((sum, qm, m) => sum + qm * k[m], 0) / divisor
      );
      // Subtracting the largest score first keeps e^score from overflowing, as the lesson's softmax does.
      const top = Math.max(...scores);
      const exps = scores.map((s) => Math.exp(s - top));
      const total = exps.reduce((a, b) => a + b, 0);
      return exps.map((e) => e / total);
    });
  }

  if (typeof module !== "undefined") module.exports = { attentionWeights };
  if (typeof window === "undefined" || !window.Viz) return;

  const NAME = "attention-matrix";
  const SCOPE = `.viz[data-viz="${NAME}"]`;
  const STYLE = `
    ${SCOPE} .am-grid { display: grid; gap: 2px; margin-top: 0.7rem; max-width: 36rem; font-size: 12px; line-height: 1.2; cursor: pointer; }
    ${SCOPE} .am-col { writing-mode: vertical-rl; transform: rotate(180deg); justify-self: center; align-self: end;
      white-space: nowrap; color: var(--p-muted); padding-top: 0.2rem; }
    ${SCOPE} .am-row { align-self: center; justify-self: end; white-space: nowrap; color: var(--p-muted); padding-right: 0.35rem; }
    ${SCOPE} .am-row.sel, ${SCOPE} .am-col.sel { color: var(--p-fg); font-weight: 600; }
    ${SCOPE} .am-cell { text-align: center; padding: 0.4rem 0; border-radius: 3px; font-variant-numeric: tabular-nums; font-size: 11px;
      color: var(--p-fg); min-width: 0; overflow: hidden; }
    ${SCOPE} .am-cell.hi { color: var(--p-bg); }
    ${SCOPE} .am-cell.mid { color: var(--am-mid-text, var(--p-fg)); }
    @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) ${SCOPE} { --am-mid-text: var(--p-bg); } }
    :root[data-theme="dark"] ${SCOPE} { --am-mid-text: var(--p-bg); }
    ${SCOPE} .am-cell.masked { background: var(--p-card-2); color: var(--p-muted); opacity: 0.55; }
    ${SCOPE} .am-cell.sel { box-shadow: inset 0 0 0 2px var(--p-fg); }
    ${SCOPE} .am-cell.sel.masked { opacity: 0.8; }
    @media (max-width: 560px) {
      ${SCOPE} .am-grid { gap: 1px; font-size: 11px; }
      ${SCOPE} .am-cell { font-size: 9.5px; padding: 0.35rem 0; letter-spacing: -0.02em; }
      ${SCOPE} .am-row { padding-right: 0.2rem; }
    }`;

  window.Viz.register(NAME, (el, data) => {
    if (!document.getElementById("viz-style-" + NAME)) {
      const style = document.createElement("style");
      style.id = "viz-style-" + NAME;
      style.textContent = STYLE;
      document.head.appendChild(style);
    }
    const { tokens, Q, K } = data;
    const n = tokens.length;
    const state = { causal: false, temperature: 1, row: Math.max(0, tokens.indexOf("it")) };

    window.Viz.select(el, {
      label: "Causal mask",
      options: [["off", "Off: every word sees the whole sentence"], ["on", "On: each word sees only the past"]],
      value: "off",
      onChange: (v) => { state.causal = v === "on"; draw(); },
    });
    window.Viz.slider(el, {
      label: "Temperature", min: 0.25, max: 3, step: 0.05, value: state.temperature,
      format: (v) => v.toFixed(2),
      onInput: (v) => { state.temperature = v; draw(); },
    });
    const rowSelect = window.Viz.select(el, {
      label: "Word doing the looking",
      options: tokens.map((t, i) => [i, `${i + 1}. ${t}`]),
      value: state.row,
      onChange: (v) => { state.row = Number(v); draw(); },
    });

    // The grid repeats what the readout says in words, so screen readers skip it.
    const grid = document.createElement("div");
    grid.className = "am-grid";
    grid.setAttribute("aria-hidden", "true");
    grid.style.gridTemplateColumns = `max-content repeat(${n}, minmax(0, 1fr))`;
    el.appendChild(grid);
    const readout = window.Viz.readout(el);
    const note = document.createElement("p");
    note.className = "viz-note";
    note.textContent =
      "Each row is one word's attention and adds up to 1; darker means more. " +
      `Temperature divides every score before softmax: 1 is exactly softmax(QKᵀ / √${K[0].length}), ` +
      `and ${(1 / Math.sqrt(K[0].length)).toFixed(2)} cancels the √dₖ scaling. Click a row to pick it.`;
    el.appendChild(note);

    // A mouse shortcut for the row select above; keyboard users have the select itself.
    grid.addEventListener("click", (event) => {
      const row = event.target.closest("[data-row]");
      if (!row) return;
      state.row = Number(row.dataset.row);
      rowSelect.value = String(state.row);
      draw();
    });

    function cell(cls, text, row) {
      const div = document.createElement("div");
      div.className = cls;
      div.textContent = text;
      if (row !== undefined) div.dataset.row = row;
      return div;
    }

    function draw() {
      const weights = attentionWeights(Q, K, state);
      const parts = [cell("am-corner", "")];
      tokens.forEach((t, j) => parts.push(cell(`am-col${weights[state.row][j] === maxOf(weights[state.row]) ? " sel" : ""}`, t)));
      weights.forEach((row, i) => {
        const selected = i === state.row;
        parts.push(cell(`am-row${selected ? " sel" : ""}`, tokens[i], i));
        row.forEach((w, j) => {
          const masked = state.causal && j > i;
          const c = cell(`am-cell${masked ? " masked" : ""}${shade(w)}${selected ? " sel" : ""}`, w.toFixed(2), i);
          if (!masked) c.style.background = `color-mix(in srgb, var(--p-accent) ${Math.round(w * 100)}%, var(--p-card))`;
          parts.push(c);
        });
      });
      grid.replaceChildren(...parts);
      readout.textContent = describe(weights[state.row]);
    }

    // The selected row, in words: where its attention goes most, and why.
    function describe(row) {
      const word = `"${tokens[state.row]}"`;
      const top = maxOf(row);
      const winners = tokens.filter((_, j) => Math.abs(row[j] - top) < 1e-9).map((t) => `"${t}"`);
      const rest = row.filter((w) => top - w > 1e-9);
      const runnerUp = rest.length ? Math.max(...rest) : 0;
      const seen = state.causal ? state.row + 1 : n;
      let text =
        `${word} gives its biggest share of attention to ${winners.join(" and ")}: ` +
        `${top.toFixed(2)}${winners.length > 1 ? " each" : ""}. `;
      if (runnerUp > 0) {
        const next = tokens.filter((_, j) => Math.abs(row[j] - runnerUp) < 1e-9).map((t) => `"${t}"`);
        text += `Next comes ${next.join(" and ")} at ${runnerUp.toFixed(2)}. `;
      } else if (seen > 1) {
        text += "Every other word gets 0. ";
      }
      text += state.causal
        ? `With the causal mask on it sees ${seen === 1 ? "only itself" : `only the ${seen} words up to itself`}; later words get exactly 0. `
        : `With no mask it sees all ${n} words, including those after it. `;
      const t = state.temperature;
      text +=
        t === 1
          ? "Temperature 1 is the lesson's formula."
          : `Temperature ${t.toFixed(2)} makes every row ${t < 1 ? "sharper" : "flatter"} than the lesson's formula.`;
      return text;
    }

    draw();
  });

  // Which text colour stays readable on a cell this dark. The light theme's
  // accent is a deep blue and the dark theme's a pale one, so the cells flip
  // from the page's text colour to its background colour at different weights:
  // from 0.8 in light ("hi"), already from 0.6 in dark ("mid", set in STYLE).
  function shade(w) {
    if (w >= 0.8) return " hi";
    return w >= 0.6 ? " mid" : "";
  }

  function maxOf(row) {
    return Math.max(...row);
  }
})();
