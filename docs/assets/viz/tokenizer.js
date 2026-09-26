/*
 * tokenizer.js: type any text and watch byte-level BPE cut it into tokens,
 * one learned merge at a time.
 *
 * The encoder mirrors primer/ml/tokenization.py exactly: the same
 * pre-tokenization pattern (PRETOKENIZE), UTF-8 bytes as ids 0-255, merge i
 * making id 256 + i, and inside each chunk the earliest-learned merge applied
 * first (ByteBPE._encode_chunk). Pieces that are only part of a character
 * show as \xNN bytes, like ByteBPE.tokens. tests/test_tokenization.py runs
 * this file in Node and compares it with the Python, so the two can't drift.
 */
(function () {
  // Python's \s, spelled out: JavaScript's \s adds U+FEFF and lacks
  // U+001C-U+001F and U+0085, which would chunk some text differently.
  const WS = "\\t-\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
  const PRETOKENIZE = new RegExp(
    `'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]+| ?[^${WS}A-Za-z0-9]+|[${WS}]+(?![^${WS}])|[${WS}]+`,
    "gu",
  );
  const encoder = new TextEncoder();
  // ignoreBOM keeps a leading U+FEFF as a character, as Python's "utf-8" codec does.
  const strict = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });

  function pretokenize(text) {
    return text.match(PRETOKENIZE) || [];
  }

  // Every id's bytes: 0-255 stand for themselves, 256 + i for merge i's two halves.
  function vocabulary(merges) {
    const vocab = [];
    for (let b = 0; b < 256; b++) vocab.push([b]);
    merges.forEach(([a, b]) => vocab.push(vocab[a].concat(vocab[b])));
    return vocab;
  }

  // Replace every non-overlapping (a, b), left to right, with newId.
  function merge(ids, a, b, newId) {
    const out = [];
    for (let i = 0; i < ids.length; i++) {
      if (i + 1 < ids.length && ids[i] === a && ids[i + 1] === b) {
        out.push(newId);
        i += 1;
      } else out.push(ids[i]);
    }
    return out;
  }

  function encode(text, merges) {
    const rank = new Map(merges.map(([a, b], i) => [`${a},${b}`, i]));
    const out = [];
    for (const chunk of pretokenize(text)) {
      let ids = Array.from(encoder.encode(chunk));
      while (ids.length > 1) {
        // The earliest-learned merge among the pairs present, as in training.
        let best = -1;
        for (let i = 0; i + 1 < ids.length; i++) {
          const r = rank.get(`${ids[i]},${ids[i + 1]}`);
          if (r !== undefined && (best < 0 || r < best)) best = r;
        }
        if (best < 0) break;
        ids = merge(ids, merges[best][0], merges[best][1], 256 + best);
      }
      out.push(...ids);
    }
    return out;
  }

  function piece(bytes) {
    try {
      return strict.decode(new Uint8Array(bytes));
    } catch (e) {
      // Half a character: show its bytes rather than a replacement mark.
      return bytes.map((x) => `\\x${x.toString(16).padStart(2, "0")}`).join("");
    }
  }

  function tokenize(text, merges) {
    const vocab = vocabulary(merges);
    return encode(text, merges).map((id) => piece(vocab[id]));
  }

  // Characters, not UTF-16 units, so an emoji counts once as in Python's len().
  function charsPerToken(text, merges) {
    const n = encode(text, merges).length;
    return n ? Array.from(text).length / n : 0;
  }

  if (typeof module !== "undefined") module.exports = { pretokenize, encode, tokenize, charsPerToken };
  if (typeof window === "undefined" || !window.Viz) return;

  const STYLE = `
.viz[data-viz="tokenizer"] { position: relative; }
.viz[data-viz="tokenizer"] .viz-row input[type="text"] {
  grid-column: 2 / 4; min-width: 0; width: 100%; box-sizing: border-box;
  font: inherit; color: var(--p-fg); background: var(--p-bg);
  border: 1px solid var(--p-border); border-radius: 0.35rem; padding: 0.25rem 0.45rem;
}
.viz[data-viz="tokenizer"] .tok-chips { display: flex; flex-wrap: wrap; gap: 0.25rem 0.2rem; margin-top: 0.7rem; }
.viz[data-viz="tokenizer"] .tok-chip {
  font: 0.95em/1.35 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  white-space: pre-wrap; overflow-wrap: anywhere; min-width: 0.6em; text-align: center;
  padding: 0.05rem 0.3rem; border-radius: 0.3rem; color: var(--p-fg);
  border: 1px solid var(--p-border);
  background: color-mix(in srgb, var(--p-accent) 22%, var(--p-card));
}
.viz[data-viz="tokenizer"] .tok-chip.odd { background: var(--p-mark); }
.viz[data-viz="tokenizer"] .tok-chip.newest { outline: 2px solid var(--p-accent); outline-offset: 1px; }
.viz[data-viz="tokenizer"] .tok-last { margin-top: 0.2rem; }
.viz[data-viz="tokenizer"] .tok-code {
  font: 0.95em ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre;
  padding: 0 0.25rem; border-radius: 0.25rem; background: var(--p-card-2); border: 1px solid var(--p-border);
}
.viz[data-viz="tokenizer"] .tok-said {
  position: absolute; width: 1px; height: 1px; overflow: hidden;
  clip: rect(0 0 0 0); clip-path: inset(50%); white-space: nowrap;
}
@media (max-width: 560px) {
  .viz[data-viz="tokenizer"] .viz-row input[type="text"] { grid-column: 1 / 3; }
}`;

  // Spaces are part of tokens (" the"), so draw them where the eye can see them.
  function visible(s) {
    return s.replace(/ /g, "·").replace(/\t/g, "⇥").replace(/\n/g, "↵");
  }

  window.Viz.register("tokenizer", (el, data) => {
    const merges = data.merges;
    const vocab = vocabulary(merges);
    const state = { text: data.example, n: merges.length };

    if (!document.getElementById("viz-tokenizer-style")) {
      const style = document.createElement("style");
      style.id = "viz-tokenizer-style";
      style.textContent = STYLE;
      document.head.appendChild(style);
    }

    const row = document.createElement("label");
    row.className = "viz-row";
    const name = document.createElement("span");
    name.textContent = "Text to tokenize";
    const input = document.createElement("input");
    Object.assign(input, { type: "text", value: state.text, maxLength: 300, spellcheck: false });
    input.addEventListener("input", () => { state.text = input.value; draw(); });
    row.append(name, input);
    el.appendChild(row);

    window.Viz.slider(el, {
      label: "Merges applied", min: 0, max: merges.length, value: state.n,
      format: (v) => `${v} of ${merges.length}`,
      onInput: (v) => { state.n = v; draw(); },
    });

    const chips = document.createElement("div");
    chips.className = "tok-chips";
    chips.setAttribute("aria-hidden", "true");
    el.appendChild(chips);
    const readout = window.Viz.readout(el);
    const note = document.createElement("p");
    note.className = "viz-note";
    note.textContent =
      "Each box is one token; the colours alternate so you can see where one ends. " +
      "A dot (·) is a space, and an outlined box is the newest merge at work.";
    el.appendChild(note);

    function draw() {
      const active = merges.slice(0, state.n);
      const ids = encode(state.text, active);
      const pieces = ids.map((id) => visible(piece(vocab[id])));
      const newest = state.n ? 256 + state.n - 1 : -1;
      chips.replaceChildren(...pieces.map((p, i) => {
        const chip = document.createElement("span");
        chip.className = `tok-chip${i % 2 ? " odd" : ""}${ids[i] === newest ? " newest" : ""}`;
        chip.textContent = p;
        return chip;
      }));

      const chars = Array.from(state.text).length;
      const count = ids.length
        ? `${ids.length} token${ids.length === 1 ? "" : "s"} for ${chars} character${chars === 1 ? "" : "s"}: ` +
          `${charsPerToken(state.text, active).toFixed(2)} characters per token.`
        : "No text, so no tokens.";
      const said = document.createElement("span");
      said.className = "tok-said";
      said.textContent = ids.length ? ` Tokens: ${pieces.join(", ")}.` : "";
      // The newest merge on a line of its own, its pieces in code type so a
      // piece like "·the" isn't mistaken for punctuation in the sentence.
      const last = document.createElement("div");
      last.className = "tok-last";
      if (state.n) {
        const [a, b] = merges[state.n - 1];
        last.append(`Newest merge, number ${state.n}: `, code(a), " + ", code(b), " → ", code(newest));
      } else last.textContent = "No merges yet, so every byte is its own token.";
      readout.replaceChildren(count, last, said);
    }

    function code(id) {
      const c = document.createElement("code");
      c.className = "tok-code";
      c.textContent = visible(piece(vocab[id]));
      return c;
    }

    draw();
  });
})();
