/*
 * rag-pipeline.js: chunk a document, retrieve chunks, rerank them.
 *
 * The chunker mirrors fixed_size_chunks in primer/ml/embeddings/retrieval.py
 * exactly, and rerank mirrors retrieve_then_rerank. The widget can't embed
 * text, so the lesson's viz_data() precomputes the hybrid ranking and the
 * cross-encoder's scores for every setting the sliders reach.
 * tests/test_emb_retrieval.py runs these functions in Node and compares them
 * with the Python, so the widget can't drift from the lesson.
 */
(function () {
  // Python's str.split(): runs of whitespace separate words, ends are ignored.
  function words(text) {
    return text.split(/\s+/).filter(Boolean);
  }

  // [start, end) word positions of each window, as fixed_size_chunks cuts them.
  function chunkSpans(n, size, overlap) {
    const step = size - overlap;
    const spans = [];
    // Stop once a window would start inside the previous window's overlap.
    for (let s = 0; s < Math.max(1, n - overlap); s += step) spans.push([s, Math.min(s + size, n)]);
    return spans;
  }

  function fixedSizeChunks(text, size = 50, overlap = 10) {
    const w = words(text);
    return chunkSpans(w.length, size, overlap).map(([s, e]) => w.slice(s, e).join(" "));
  }

  // Which chunks hold the whole answer sentence, and which hold only part of it.
  function answerChunks(text, size, overlap, answer) {
    const w = words(text);
    const a = words(answer);
    let start = -1;
    for (let i = 0; i + a.length <= w.length && start < 0; i++) {
      if (a.every((word, j) => w[i + j] === word)) start = i;
    }
    const whole = [];
    const partial = [];
    if (start < 0) return { whole, partial, start, end: start };
    const end = start + a.length;
    chunkSpans(w.length, size, overlap).forEach(([s, e], i) => {
      if (s <= start && end <= e) whole.push(i);
      else if (s < end && start < e) partial.push(i);
    });
    return { whole, partial, start, end };
  }

  // Stage 2 on the top k of stage 1: highest cross-encoder score first, ties
  // keep their first-stage order, exactly as retrieve_then_rerank sorts.
  function rerank(run, k) {
    return run.retrieval
      .slice(0, k)
      .map(([chunk], rank) => ({ chunk, rank, score: run.rerank[chunk] }))
      .sort((x, y) => y.score - x.score || x.rank - y.rank)
      .map((c) => c.chunk);
  }

  // The readout in words: where the answer landed, what retrieval ranked
  // first, and what reranking the top k changed. `run` is one entry of
  // viz_data()'s runs; `ans` comes from answerChunks.
  function describe(run, ans, size, overlap, k) {
    const setting = `With ${size}-word chunks and ${overlap ? `${overlap} words of overlap` : "no overlap"}`;
    let where;
    if (ans.whole.length === 1) where = `${setting}, the answer sentence sits whole in chunk ${ans.whole[0] + 1}.`;
    else if (ans.whole.length) where = `${setting}, the overlap copies the whole answer sentence into chunks ${names(ans.whole)}.`;
    else where = `${setting}, the answer sentence is split across chunks ${names(ans.partial)}: no chunk holds all of it.`;

    // When no chunk holds the whole sentence, the best a pipeline can do is a chunk with part of it.
    const hits = ans.whole.length ? ans.whole : ans.partial;
    const noun = ans.whole.length ? "the answer" : "part of the answer";
    const firstRank = new Map(run.retrieval.map(([c], i) => [c, i + 1]));
    const top = run.retrieval[0][0];
    const best = hits.reduce((a, b) => (firstRank.get(a) <= firstRank.get(b) ? a : b));
    const r1 = firstRank.get(best);
    let retrieval;
    if (hits.includes(top)) retrieval = `Retrieval ranks chunk ${top + 1} first, which holds ${noun}.`;
    else {
      retrieval = `Retrieval ranks chunk ${top + 1} first; chunk ${best + 1}, with ${noun}, comes ${ordinal(r1)}`;
      retrieval += r1 > k ? `, outside the top ${k}, so the reranker never sees it.` : ".";
    }

    const second = rerank(run, k);
    const now = second[0];
    let reranking;
    if (r1 > k) reranking = "Reranking can only reorder what retrieval found.";
    else if (k === 1) reranking = "With a top k of 1 there is nothing for the reranker to reorder.";
    else if (!hits.includes(top) && hits.includes(now)) reranking = `Reranking moves chunk ${now + 1} to the top`;
    else if (hits.includes(top) && now === top) reranking = "Reranking keeps it on top";
    else if (hits.includes(top) && hits.includes(now)) reranking = `Reranking puts chunk ${now + 1}, which also holds ${noun}, on top`;
    else if (hits.includes(top)) reranking = `Reranking puts chunk ${now + 1} above it, a mistake: rerankers are models too.`;
    else {
      const r2 = second.findIndex((c) => hits.includes(c)) + 1;
      reranking = `Reranking puts chunk ${now + 1} first and chunk ${second[r2 - 1] + 1} ${ordinal(r2)}: neither stage puts ${ans.whole.length ? "the answer" : "any part of the answer"} first.`;
    }
    if (!reranking.endsWith(".")) {
      const rest = ans.partial.filter((c) => c !== now);
      // A split answer: even the right chunk hands the model half a sentence.
      reranking += ans.whole.length ? "." : `, but the rest of the sentence is in chunk ${names(rest)}.`;
    }
    return `${where} ${retrieval} ${reranking}`;
  }

  // [2, 3, 4] -> "3, 4 and 5" (chunks are numbered from 1 on screen)
  function names(cs) {
    const n = cs.map((c) => c + 1);
    return n.length > 1 ? `${n.slice(0, -1).join(", ")} and ${n[n.length - 1]}` : String(n[0]);
  }

  // 2 -> "2nd"
  function ordinal(n) {
    const tail = n % 100 >= 11 && n % 100 <= 13 ? "th" : { 1: "st", 2: "nd", 3: "rd" }[n % 10] || "th";
    return `${n}${tail}`;
  }

  if (typeof module !== "undefined") module.exports = { fixedSizeChunks, chunkSpans, answerChunks, rerank, describe };
  if (typeof window === "undefined" || !window.Viz) return;

  const STYLE = `
.viz[data-viz="rag-pipeline"] .rag-doc { margin: 0.7rem 0 0.3rem; padding: 0.6rem 0.7rem; border: 1px solid var(--p-border);
  border-radius: 0.4rem; background: var(--p-bg); line-height: 1.9; font-size: 0.9em; overflow-wrap: anywhere; }
.viz[data-viz="rag-pipeline"] .rag-seg { padding: 0.15em 0; -webkit-box-decoration-break: clone; box-decoration-break: clone; }
.viz[data-viz="rag-pipeline"] .rag-a { background: color-mix(in srgb, var(--p-fg) 9%, var(--p-bg)); }
.viz[data-viz="rag-pipeline"] .rag-b { background: color-mix(in srgb, var(--p-accent) 24%, var(--p-bg)); }
.viz[data-viz="rag-pipeline"] .rag-both { background: repeating-linear-gradient(135deg,
  color-mix(in srgb, var(--p-accent) 42%, transparent) 0 3px, transparent 3px 7px), color-mix(in srgb, var(--p-fg) 9%, var(--p-bg)); }
.viz[data-viz="rag-pipeline"] .rag-ans { font-weight: 600; text-decoration: underline 2px; text-underline-offset: 3px; }
.viz[data-viz="rag-pipeline"] .rag-badge { display: inline-block; min-width: 1.3em; margin-right: 0.25em; padding: 0 0.3em;
  border: 1px solid var(--p-border); border-radius: 0.3em; background: var(--p-card); color: var(--p-muted);
  font-size: 0.75em; line-height: 1.5; text-align: center; font-weight: 600; font-variant-numeric: tabular-nums; }
.viz[data-viz="rag-pipeline"] .rag-badge.hit { background: var(--p-accent); border-color: var(--p-accent); color: var(--p-bg); }
.viz[data-viz="rag-pipeline"] .rag-legend { display: flex; flex-wrap: wrap; gap: 0.2rem 1rem; color: var(--p-muted); font-size: 0.85em; }
.viz[data-viz="rag-pipeline"] .rag-legend i { display: inline-block; width: 1.4em; height: 0.9em; margin-right: 0.35em;
  vertical-align: -0.1em; border-radius: 0.2em; }
.viz[data-viz="rag-pipeline"] .rag-lists { display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; margin-top: 0.8rem; }
.viz[data-viz="rag-pipeline"] .rag-col { min-width: 0; }
.viz[data-viz="rag-pipeline"] .rag-title { margin: 0 0 0.3rem; font-weight: 600; }
.viz[data-viz="rag-pipeline"] .rag-sub { display: block; font-weight: 400; color: var(--p-muted); font-size: 0.85em; }
.viz[data-viz="rag-pipeline"] ol { list-style: none; margin: 0; padding: 0; }
.viz[data-viz="rag-pipeline"] li { display: grid; grid-template-columns: 1.6rem 1fr auto; gap: 0 0.4rem; align-items: baseline;
  margin: 0.25rem 0; padding: 0.3rem 0.45rem; border: 1px solid var(--p-border); border-radius: 0.35rem; background: var(--p-card); }
.viz[data-viz="rag-pipeline"] li > * { min-width: 0; }
.viz[data-viz="rag-pipeline"] li.whole { border-color: var(--p-accent); box-shadow: inset 3px 0 0 var(--p-accent); }
.viz[data-viz="rag-pipeline"] li.part { border-color: var(--p-accent); border-style: dashed; }
.viz[data-viz="rag-pipeline"] .rank { color: var(--p-muted); font-variant-numeric: tabular-nums; }
.viz[data-viz="rag-pipeline"] .score { color: var(--p-muted); font-variant-numeric: tabular-nums; font-size: 0.9em; text-align: right; }
.viz[data-viz="rag-pipeline"] .tag { color: var(--p-accent); font-size: 0.85em; font-weight: 600; }
.viz[data-viz="rag-pipeline"] .snip { grid-column: 2 / 4; color: var(--p-muted); font-size: 0.82em;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
@media (max-width: 560px) { .viz[data-viz="rag-pipeline"] .rag-lists { grid-template-columns: 1fr; } }
`;

  window.Viz.register("rag-pipeline", (el, data) => {
    if (!document.getElementById("rag-pipeline-style")) {
      const style = document.createElement("style");
      style.id = "rag-pipeline-style";
      style.textContent = STYLE;
      document.head.appendChild(style);
    }
    const w = words(data.text);
    const state = { q: 0, size: data.sizes.indexOf(40), overlap: 0, k: 5 };

    window.Viz.select(el, {
      label: "Question",
      options: data.questions.map((q, i) => [i, q.question]),
      value: 0,
      onChange: (v) => { state.q = Number(v); draw(); },
    });
    window.Viz.slider(el, {
      label: "Chunk size", min: 0, max: data.sizes.length - 1, value: state.size,
      format: (v) => `${data.sizes[v]} words`,
      onInput: (v) => { state.size = v; draw(); },
    });
    window.Viz.slider(el, {
      label: "Overlap", min: 0, max: data.overlaps.length - 1, value: state.overlap,
      format: (v) => `${data.overlaps[v]} words`,
      onInput: (v) => { state.overlap = v; draw(); },
    });
    window.Viz.slider(el, {
      label: "Top k to rerank", min: 1, max: data.max_k, value: state.k,
      format: (v) => `${v} chunk${v === 1 ? "" : "s"}`,
      onInput: (v) => { state.k = v; draw(); },
    });

    const doc = document.createElement("div");
    doc.className = "rag-doc";
    const legend = document.createElement("div");
    legend.className = "rag-legend";
    legend.setAttribute("aria-hidden", "true");
    legend.innerHTML =
      '<span><i class="rag-a"></i><i class="rag-b"></i>one chunk each</span>' +
      '<span><i class="rag-both"></i>overlap: in two chunks</span>' +
      '<span><b class="rag-ans">underlined</b>: the answer sentence</span>';
    const lists = document.createElement("div");
    lists.className = "rag-lists";
    el.append(doc, legend, lists);
    const readout = window.Viz.readout(el);

    function draw() {
      const size = data.sizes[state.size];
      const overlap = data.overlaps[state.overlap];
      const k = state.k;
      const q = data.questions[state.q];
      const run = data.runs[`${size}/${overlap}`][state.q];
      const spans = chunkSpans(w.length, size, overlap);
      const ans = answerChunks(data.text, size, overlap, q.answer);
      const hits = ans.whole.length ? ans.whole : ans.partial;

      drawDocument(spans, ans, hits);
      const first = run.retrieval.slice(0, k);
      const second = rerank(run, k);
      const firstRank = new Map(run.retrieval.map(([c], i) => [c, i + 1]));
      lists.replaceChildren(
        column("1. Retrieve", "hybrid search, top k by fused score",
          first.map(([c, s]) => ({ c, score: s.toFixed(4) })), spans, ans),
        column("2. Rerank", "the top k, reordered by the cross-encoder",
          second.map((c) => ({ c, score: run.rerank[c].toFixed(2), was: firstRank.get(c) })), spans, ans),
      );
      readout.textContent = describe(run, ans, size, overlap, k);
    }

    // The document once, word by word, grouped into runs that belong to the same chunks.
    function drawDocument(spans, ans, hits) {
      const owners = w.map((_, i) => spans.flatMap(([s, e], c) => (s <= i && i < e ? [c] : [])));
      doc.replaceChildren();
      let i = 0;
      while (i < w.length) {
        let j = i + 1;
        const inAnswer = (x) => x >= ans.start && x < ans.end;
        while (j < w.length && owners[j].join() === owners[i].join() && inAnswer(j) === inAnswer(i)) j++;
        const starts = spans.findIndex(([s]) => s === i);
        if (starts >= 0) {
          const badge = document.createElement("span");
          badge.className = `rag-badge${hits.includes(starts) ? " hit" : ""}`;
          badge.textContent = starts + 1;
          badge.title = `chunk ${starts + 1} starts here`;
          doc.appendChild(badge);
        }
        const seg = document.createElement("span");
        const o = owners[i];
        seg.className = `rag-seg ${o.length > 1 ? "rag-both" : o[0] % 2 ? "rag-b" : "rag-a"}${inAnswer(i) ? " rag-ans" : ""}`;
        seg.textContent = w.slice(i, j).join(" ");
        doc.append(seg, " ");
        i = j;
      }
    }

    function column(title, sub, rows, spans, ans) {
      const col = document.createElement("div");
      col.className = "rag-col";
      const head = document.createElement("p");
      head.className = "rag-title";
      head.textContent = title;
      const small = document.createElement("span");
      small.className = "rag-sub";
      small.textContent = sub;
      head.appendChild(small);
      const ol = document.createElement("ol");
      rows.forEach(({ c, score, was }, n) => {
        const li = document.createElement("li");
        const whole = ans.whole.includes(c);
        const part = ans.partial.includes(c);
        li.className = whole ? "whole" : part ? "part" : "";
        const rank = document.createElement("span");
        rank.className = "rank";
        rank.textContent = `${n + 1}.`;
        const name = document.createElement("span");
        name.textContent = `Chunk ${c + 1} `;
        if (whole || part) {
          const tag = document.createElement("span");
          tag.className = "tag";
          tag.textContent = whole ? "answer" : "part of answer";
          name.appendChild(tag);
        }
        const sc = document.createElement("span");
        sc.className = "score";
        sc.textContent = was && was !== n + 1 ? `${score} (was ${was})` : score;
        const snip = document.createElement("span");
        snip.className = "snip";
        snip.textContent = w.slice(spans[c][0], spans[c][1]).join(" ");
        li.append(rank, name, sc, snip);
        ol.appendChild(li);
      });
      col.append(head, ol);
      return col;
    }

    draw();
  });
})();
