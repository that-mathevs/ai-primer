/*
 * context-budget.js: one context window, split into the parts an agent sends,
 * and what gets cut when they don't all fit.
 *
 * fitToWindow mirrors fit_to_window in primer/agents/context.py exactly;
 * tests/test_agents_context.py runs both and compares them, so the widget
 * can't drift from the lesson.
 */
(function () {
  // Cut the least important piece, then the next, until the rest fits.
  // documents are ranked best first; turns run oldest first.
  function fitToWindow(window, { system, tools, message, reserve, documents, turns, keepRecent = 2 }) {
    const must = system + tools + message + reserve;
    const t = turns.length;
    const split = Math.max(t - keepRecent, 0);
    // Most important first, so cutting from the end removes the least important:
    // the last turns, then documents best first, then older turns newest first.
    const pieces = [];
    for (let i = t - 1; i >= split; i -= 1) pieces.push(["turn", i, turns[i]]);
    documents.forEach((d, i) => pieces.push(["doc", i, d]));
    for (let i = split - 1; i >= 0; i -= 1) pieces.push(["turn", i, turns[i]]);
    let used = must + pieces.reduce((sum, p) => sum + p[2], 0);
    while (used > window && pieces.length) used -= pieces.pop()[2];
    const kept = (kind) => pieces.filter((p) => p[0] === kind).map((p) => p[1]).sort((a, b) => a - b);
    return { keptDocuments: kept("doc"), keptTurns: kept("turn"), used, fits: used <= window };
  }

  if (typeof module !== "undefined") module.exports = { fitToWindow };
  if (typeof window === "undefined" || !window.Viz) return;

  const SCOPE = '.viz[data-viz="context-budget"]';
  const STYLE = `
${SCOPE} .cb-track { position: relative; height: 1.6rem; margin: 0.8rem 0 0.4rem; border-radius: 0.3rem; background: var(--p-card-2); }
${SCOPE} .cb-seg { position: absolute; top: 0; bottom: 0; box-sizing: border-box; box-shadow: inset -2px 0 0 var(--p-card); }
${SCOPE} .cb-limit { position: absolute; top: -0.35rem; bottom: -0.35rem; width: 2px; margin-left: -1px; background: var(--p-fg); }
${SCOPE} .cb-system { background: var(--p-muted); }
${SCOPE} .cb-tools { background: color-mix(in srgb, var(--p-muted) 50%, var(--p-card-2)); }
${SCOPE} .cb-docs { background: var(--p-accent); }
${SCOPE} .cb-turns { background: color-mix(in srgb, var(--p-accent) 45%, var(--p-card-2)); }
${SCOPE} .cb-message { background: var(--p-fg); }
${SCOPE} .cb-answer { background: var(--p-accent-soft); border: 2px dashed var(--p-muted); }
${SCOPE} .cb-cut { background: repeating-linear-gradient(135deg, var(--p-muted) 0 2px, transparent 2px 6px); opacity: 0.75; }
${SCOPE} .cb-legend { list-style: none; margin: 0.6rem 0 0; padding: 0; display: grid; grid-template-columns: repeat(auto-fill, minmax(13rem, 1fr)); gap: 0.2rem 1rem; }
${SCOPE} .cb-legend li { display: flex; gap: 0.45rem; align-items: baseline; min-width: 0; margin: 0; font-variant-numeric: tabular-nums; }
${SCOPE} .cb-legend .cb-swatch { position: relative; flex: none; width: 0.9rem; height: 0.9rem; top: 0.12rem; border-radius: 0.15rem; box-shadow: none; }
${SCOPE} .cb-legend .cb-cut { outline: 1px solid var(--p-muted); outline-offset: -1px; }
${SCOPE} .cb-dim { color: var(--p-muted); }
`;

  function addStyle() {
    if (document.getElementById("cb-style")) return;
    const style = document.createElement("style");
    style.id = "cb-style";
    style.textContent = STYLE;
    document.head.appendChild(style);
  }

  const n = (x) => x.toLocaleString("en-US");
  const plural = (count, word) => `${count} ${word}${count === 1 ? "" : "s"}`;
  // "14 turns and 2 docs"
  const cutSummary = (turns, docs) => [turns && plural(turns, "turn"), docs && plural(docs, "doc")].filter(Boolean).join(" and ");

  window.Viz.register("context-budget", (el, data) => {
    addStyle();
    const state = { window: data.window, turns: data.turns, documents: data.documents, reserve: data.reserve };

    window.Viz.select(el, {
      label: "Context window",
      options: data.windows.map((w) => [w, `${n(w)} tokens`]),
      value: data.window,
      onChange: (v) => { state.window = Number(v); draw(); },
    });
    window.Viz.slider(el, {
      label: "Conversation turns", min: 0, max: data.max_turns, value: state.turns,
      format: (v) => plural(v, "turn"),
      onInput: (v) => { state.turns = v; draw(); },
    });
    window.Viz.slider(el, {
      label: "Retrieved documents", min: 0, max: data.max_documents, value: state.documents,
      format: (v) => plural(v, "doc"),
      onInput: (v) => { state.documents = v; draw(); },
    });
    window.Viz.slider(el, {
      label: "Room for the answer", min: 0, max: data.max_reserve, step: 250, value: state.reserve,
      format: (v) => `${n(v)} tokens`,
      onInput: (v) => { state.reserve = v; draw(); },
    });

    // The bar repeats the legend and readout as a picture, so screen readers skip it.
    const track = document.createElement("div");
    track.className = "cb-track";
    track.setAttribute("aria-hidden", "true");
    el.appendChild(track);
    const legend = document.createElement("ul");
    legend.className = "cb-legend";
    el.appendChild(legend);
    const readout = window.Viz.readout(el);
    const note = document.createElement("p");
    note.className = "viz-note";
    note.textContent =
      "The bar is everything the agent would like to send, in prompt order; the vertical line is the window. " +
      "Hatched pieces past it were cut and are not sent.";
    el.appendChild(note);

    function draw() {
      const documents = Array(state.documents).fill(data.document);
      const turns = Array(state.turns).fill(data.turn);
      const fit = fitToWindow(state.window, {
        system: data.system, tools: data.tools, message: data.message, reserve: state.reserve,
        documents, turns, keepRecent: data.keep_recent,
      });
      const keptDocs = fit.keptDocuments.length;
      const keptTurns = fit.keptTurns.length;
      const cutDocs = state.documents - keptDocs;
      const cutTurns = state.turns - keptTurns;
      const cutTokens = cutDocs * data.document + cutTurns * data.turn;
      const parts = [
        ["cb-system", data.system, "System prompt", n(data.system)],
        ["cb-tools", data.tools, "Tool definitions", n(data.tools)],
        ["cb-docs", keptDocs * data.document, "Documents", `${keptDocs} of ${state.documents}, ${n(keptDocs * data.document)}`],
        ["cb-turns", keptTurns * data.turn, "Turns", `${keptTurns} of ${state.turns}, ${n(keptTurns * data.turn)}`],
        ["cb-message", data.message, "User's message", n(data.message)],
        ["cb-answer", state.reserve, "Room for the answer", n(state.reserve)],
        ["cb-cut", cutTokens, "Cut", cutTokens ? `${cutSummary(cutTurns, cutDocs)}, ${n(cutTokens)}` : "nothing"],
      ];
      const wanted = fit.used + cutTokens;
      // Keep the window line visible even when everything fits comfortably.
      const scale = Math.max(wanted, state.window) * 1.04;
      const pct = (x) => `${(x / scale) * 100}%`;

      const segs = [];
      let left = 0;
      parts.forEach(([cls, tokens]) => {
        if (tokens <= 0) return;
        const seg = document.createElement("div");
        seg.className = `cb-seg ${cls}`;
        seg.style.left = pct(left);
        seg.style.width = pct(tokens);
        segs.push(seg);
        left += tokens;
      });
      const limit = document.createElement("div");
      limit.className = "cb-limit";
      limit.style.left = pct(state.window);
      track.replaceChildren(...segs, limit);

      legend.replaceChildren(...parts.map(([cls, , label, value]) => {
        const item = document.createElement("li");
        const swatch = document.createElement("span");
        swatch.className = `cb-seg cb-swatch ${cls}`;
        swatch.setAttribute("aria-hidden", "true");
        const text = document.createElement("span");
        text.textContent = `${label}: `;
        const amount = document.createElement("span");
        amount.className = "cb-dim";
        amount.textContent = value;
        text.appendChild(amount);
        item.append(swatch, text);
        return item;
      }));

      const cuts = [];
      if (cutTurns) cuts.push(cutTurns === state.turns ? `all ${plural(cutTurns, "turn")}` : `${cutTurns} oldest ${cutTurns === 1 ? "turn" : "turns"}`);
      if (cutDocs) cuts.push(cutDocs === state.documents ? `all ${plural(cutDocs, "document")}` : `${cutDocs} lowest-ranked ${cutDocs === 1 ? "document" : "documents"}`);
      const split =
        `System prompt ${n(data.system)}, tools ${n(data.tools)}, ${keptDocs} of ${plural(state.documents, "document")} ` +
        `${n(keptDocs * data.document)}, ${keptTurns} of ${plural(state.turns, "turn")} ${n(keptTurns * data.turn)}, ` +
        `message ${n(data.message)}, room for the answer ${n(state.reserve)}.`;
      if (!fit.fits) {
        readout.textContent =
          `Does not fit: the parts that are never cut (system prompt, tools, message and room for the answer) ` +
          `need ${n(fit.used)} tokens, more than the ${n(state.window)}-token window, even with every document and turn cut. ` +
          "Shrink them or choose a bigger window.";
      } else if (cuts.length) {
        readout.textContent = `${n(fit.used)} of ${n(state.window)} tokens used; ${cuts.join(" and ")} cut to make room. ${split}`;
      } else {
        readout.textContent = `${n(fit.used)} of ${n(state.window)} tokens used, ${n(state.window - fit.used)} to spare; nothing cut. ${split}`;
      }
    }

    draw();
  });
})();
