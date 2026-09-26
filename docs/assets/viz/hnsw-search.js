/*
 * hnsw-search.js: step through one HNSW search on a 60-point map.
 *
 * The search mirrors primer/ml/embeddings/ann.py (HNSWIndex.search,
 * HNSWIndex._search_layer) exactly: the same layer order, the same beam
 * rule, the same tie-breaking as Python's heapq, the same count of distance
 * computations. tests/test_emb_ann.py runs both on the exported graph and
 * checks they expand the same nodes in the same order.
 */
(function () {
  // Similarity is the dot product of unit vectors, as in the lesson: higher is closer.
  function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i += 1) s += a[i] * b[i];
    return s;
  }

  // Python's heapq pops the smallest tuple. Candidates are stored as (-sim, id),
  // so the next one out has the highest sim, and on a tie the smaller id.
  function takeBest(candidates) {
    let at = 0;
    for (let i = 1; i < candidates.length; i += 1) {
      const c = candidates[i], b = candidates[at];
      if (c.s > b.s || (c.s === b.s && c.id < b.id)) at = i;
    }
    return candidates.splice(at, 1)[0];
  }

  // Results are stored as (sim, id), so the worst is the lowest sim, then the smaller id.
  function worstIndex(results) {
    let at = 0;
    for (let i = 1; i < results.length; i += 1) {
      const r = results[i], w = results[at];
      if (r.s < w.s || (r.s === w.s && r.id < w.id)) at = i;
    }
    return at;
  }

  // sorted(results, reverse=True): highest sim first, and on a tie the larger id.
  function bestFirst(results) {
    return results.slice().sort((a, b) => b.s - a.s || b.id - a.id);
  }

  // One layer of best-first beam search, recording every node it expands.
  function searchLayer(graph, q, entryPoints, ef, layer, log) {
    const visited = new Set(entryPoints);
    const start = entryPoints.map((id) => ({ s: dot(graph.vectors[id], q), id }));
    log.ndist += entryPoints.length;
    start.forEach((m) => log.seen.add(m.id));
    const candidates = start.slice();
    const results = start.slice();
    while (results.length > ef) results.splice(worstIndex(results), 1);
    const parent = {};
    let first = true;

    while (candidates.length) {
      const c = takeBest(candidates);
      // The closest node still to expand is worse than the beam's worst: nothing can improve it.
      if (c.s < results[worstIndex(results)].s) break;
      const fresh = graph.links[c.id][layer].filter((n) => !visited.has(n));
      fresh.forEach((n) => visited.add(n));
      const measured = fresh.map((id) => ({ s: dot(graph.vectors[id], q), id }));
      log.ndist += fresh.length;
      measured.forEach((m) => {
        log.seen.add(m.id);
        parent[m.id] = c.id;
        if (results.length < ef || m.s > results[worstIndex(results)].s) {
          candidates.push(m);
          results.push(m);
          if (results.length > ef) results.splice(worstIndex(results), 1);
        }
      });
      log.events.push({
        layer,
        node: c.id,
        sim: c.s,
        parent: c.id in parent ? parent[c.id] : null,
        firstOnLayer: first,
        measured: measured.map((m) => m.id),
        beam: bestFirst(results).map((r) => ({ id: r.id, s: r.s })),
        ndist: log.ndist,
        seen: Array.from(log.seen),
      });
      first = false;
    }
    return bestFirst(results);
  }

  // Greedy descent (a beam of one) through the upper layers, then a beam of ef at the bottom.
  function hnswSearch(graph, q, ef, k = 1) {
    const log = { events: [], ndist: 0, seen: new Set() };
    let ep = [graph.entry];
    for (let layer = graph.max_level; layer > 0; layer -= 1) {
      ep = [searchLayer(graph, q, ep, 1, layer, log)[0].id];
    }
    const found = searchLayer(graph, q, ep, Math.max(ef, k), 0, log).slice(0, k);
    return { events: log.events, found: found.map((r) => r.id), ndist: log.ndist };
  }

  // The exact answer: measure every point and keep the closest.
  function bruteForce(vectors, q) {
    let id = 0, s = -Infinity;
    vectors.forEach((v, i) => {
      const d = dot(v, q);
      if (d > s) { s = d; id = i; }
    });
    return { id, s, ndist: vectors.length };
  }

  if (typeof module !== "undefined") module.exports = { hnswSearch, bruteForce };
  if (typeof window === "undefined" || !window.Viz) return;

  const SVG = "http://www.w3.org/2000/svg";
  const STYLE = `
.viz[data-viz="hnsw-search"] .hnsw-layers { display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.6rem 0 0.4rem; }
.viz[data-viz="hnsw-search"] .hnsw-layers span { padding: 0.05rem 0.5rem; border: 1px solid var(--p-border); border-radius: 999px; color: var(--p-muted); font-size: 0.9em; }
.viz[data-viz="hnsw-search"] .hnsw-layers span.on { border-color: var(--p-accent); background: var(--p-accent-soft); color: var(--p-fg); }
.viz[data-viz="hnsw-search"] svg { display: block; width: 100%; max-width: 25rem; height: auto; margin: 0 auto; background: var(--p-bg); border: 1px solid var(--p-border); border-radius: 0.4rem; }
.viz[data-viz="hnsw-search"] .e-base { stroke: var(--p-border); stroke-width: 0.8; }
.viz[data-viz="hnsw-search"] .e-street { stroke: var(--p-muted); stroke-width: 0.8; opacity: 0.55; }
.viz[data-viz="hnsw-search"] .e-layer { stroke: var(--p-muted); stroke-width: 1.3; }
.viz[data-viz="hnsw-search"] .e-path { stroke: var(--p-accent); stroke-width: 3; stroke-linecap: round; }
.viz[data-viz="hnsw-search"] .e-next { stroke: var(--p-accent); stroke-width: 2; stroke-dasharray: 4 3; }
.viz[data-viz="hnsw-search"] .n-off { fill: var(--p-muted); opacity: 0.35; }
.viz[data-viz="hnsw-search"] .n-on { fill: var(--p-muted); }
.viz[data-viz="hnsw-search"] .n-seen { fill: var(--p-accent-soft); stroke: var(--p-accent); stroke-width: 1.3; }
.viz[data-viz="hnsw-search"] .n-done { fill: var(--p-accent); }
.viz[data-viz="hnsw-search"] .r-beam { fill: none; stroke: var(--p-accent); stroke-width: 1; }
.viz[data-viz="hnsw-search"] .r-best { fill: none; stroke: var(--p-accent-strong); stroke-width: 2.4; }
.viz[data-viz="hnsw-search"] .r-true { fill: none; stroke: var(--p-fg); stroke-width: 1.6; stroke-dasharray: 3 2; }
.viz[data-viz="hnsw-search"] .query { fill: var(--p-mark); stroke: var(--p-fg); stroke-width: 1.2; }
.viz[data-viz="hnsw-search"] .label { fill: var(--p-fg); font: 600 10px system-ui, -apple-system, sans-serif; paint-order: stroke; stroke: var(--p-bg); stroke-width: 3; }
`;

  window.Viz.register("hnsw-search", (el, data) => {
    if (!document.getElementById("viz-style-hnsw-search")) {
      const style = document.createElement("style");
      style.id = "viz-style-hnsw-search";
      style.textContent = STYLE;
      document.head.appendChild(style);
    }
    const N = data.points.length;
    const levels = data.links.map((perLayer) => perLayer.length - 1);
    const layerSize = (l) => levels.filter((lv) => lv >= l).length;
    const state = { query: 0, ef: data.ef_options[0], step: 0 };
    let run = null;
    let truth = null;

    window.Viz.select(el, {
      label: "Query",
      options: data.queries.map((q, i) => [i, `${q.name} at (${q.xy.map((v) => v.toFixed(2)).join(", ")})`]),
      value: 0,
      onChange: (v) => { state.query = Number(v); rerun(); },
    });
    window.Viz.select(el, {
      label: "Beam width (efSearch)",
      options: data.ef_options.map((ef) => [ef, ef === 1 ? "1 (pure greedy)" : String(ef)]),
      value: state.ef,
      onChange: (v) => { state.ef = Number(v); rerun(); },
    });
    const stepper = window.Viz.slider(el, {
      label: "Step",
      min: 0, max: 1, value: 0,
      format: (v) => `${v} of ${run ? run.events.length : 0}`,
      onInput: (v) => { state.step = v; draw(); },
    });

    const chips = document.createElement("div");
    chips.className = "hnsw-layers";
    chips.setAttribute("aria-hidden", "true");
    const chipFor = [];
    for (let l = data.max_level; l >= 0; l -= 1) {
      const chip = document.createElement("span");
      chip.textContent = `Layer ${l}: ${layerSize(l)} points`;
      chips.appendChild(chip);
      chipFor[l] = chip;
    }
    el.appendChild(chips);

    const svg = document.createElementNS(SVG, "svg");
    svg.setAttribute("viewBox", "0 0 300 300");
    svg.setAttribute("aria-hidden", "true");
    el.appendChild(svg);
    const readout = window.Viz.readout(el);
    const note = document.createElement("p");
    note.className = "viz-note";
    note.textContent =
      "Star: the query. Solid blue: points the search stood on, and the links it walked. " +
      "Hollow blue: points it only measured. Thick ring: the best so far. Dashed ring: the true nearest, " +
      "when the search misses it. On an upper layer, that layer's points and links stand out.";
    el.appendChild(note);

    // Map the plane ([-0.3, 0.3] on both axes) onto the square, y pointing up.
    const PAD = 14, SPAN = 0.6;
    const px = (x) => PAD + ((x + SPAN / 2) / SPAN) * (300 - 2 * PAD);
    const py = (y) => 300 - px(y);
    const at = (i) => [px(data.points[i][0]), py(data.points[i][1])];

    // Each layer's links, once per pair, for drawing.
    const edges = [];
    for (let l = 0; l <= data.max_level; l += 1) {
      const pairs = new Set();
      data.links.forEach((perLayer, i) => {
        (perLayer[l] || []).forEach((j) => pairs.add(i < j ? `${i}-${j}` : `${j}-${i}`));
      });
      edges[l] = Array.from(pairs, (p) => p.split("-").map(Number));
    }

    function add(tag, attrs) {
      const node = document.createElementNS(SVG, tag);
      Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
      svg.appendChild(node);
      return node;
    }
    const line = (i, j, cls) => {
      const [x1, y1] = at(i), [x2, y2] = at(j);
      add("line", { x1, y1, x2, y2, class: cls });
    };
    const ring = (i, r, cls) => add("circle", { cx: at(i)[0], cy: at(i)[1], r, class: cls });
    function label(i, text) {
      const [x, y] = at(i);
      const t = add("text", { x: x + 7, y: y - 6, class: "label" });
      if (x > 250) { t.setAttribute("x", x - 7); t.setAttribute("text-anchor", "end"); }
      if (y < 20) t.setAttribute("y", y + 15);
      t.textContent = text;
    }
    function star(x, y) {
      const pts = [];
      for (let i = 0; i < 10; i += 1) {
        const r = i % 2 ? 3.8 : 9.5, a = -Math.PI / 2 + (i * Math.PI) / 5;
        pts.push(`${(x + r * Math.cos(a)).toFixed(1)},${(y + r * Math.sin(a)).toFixed(1)}`);
      }
      add("polygon", { points: pts.join(" "), class: "query" });
    }

    const distance = (s) => Math.sqrt(Math.max(0, 2 - 2 * s));
    const fmt = (s) => distance(s).toFixed(3);
    const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

    function rerun() {
      const q = data.queries[state.query].vector;
      run = hnswSearch(data, q, state.ef, 1);
      truth = bruteForce(data.vectors, q);
      stepper.max = run.events.length;
      stepper.value = 0;
      stepper.dispatchEvent(new Event("input"));
    }

    function draw() {
      const k = state.step;
      const events = run.events.slice(0, k);
      const last = events[events.length - 1];
      const layer = last ? last.layer : data.max_level;
      const beam = last ? last.beam : [{ id: data.entry, s: dot(data.vectors[data.entry], data.queries[state.query].vector) }];
      const best = beam[0];
      const seen = new Set(last ? last.seen : [data.entry]);
      const done = new Set(events.map((e) => e.node));
      const finished = k === run.events.length;

      chipFor.forEach((chip, l) => chip.classList.toggle("on", l === layer));
      svg.replaceChildren();
      // The bottom layer's streets, always, for context; the current upper layer's roads on top.
      edges[0].forEach(([i, j]) => line(i, j, layer === 0 ? "e-street" : "e-base"));
      if (layer > 0) edges[layer].forEach(([i, j]) => line(i, j, "e-layer"));
      events.forEach((e) => { if (e.parent !== null) line(e.parent, e.node, "e-path"); });
      if (last && best.id !== last.node && !done.has(best.id)) line(last.node, best.id, "e-next");
      for (let i = 0; i < N; i += 1) {
        const cls = done.has(i) ? "n-done" : seen.has(i) ? "n-seen" : levels[i] >= layer ? "n-on" : "n-off";
        const r = done.has(i) || seen.has(i) ? 4 : levels[i] >= layer && layer > 0 ? 4 : 2.6;
        add("circle", { cx: at(i)[0], cy: at(i)[1], r, class: cls });
      }
      beam.slice(1).forEach((b) => ring(b.id, 6.5, "r-beam"));
      ring(best.id, 8.5, "r-best");
      const missed = finished && run.found[0] !== truth.id;
      if (missed) ring(truth.id, 8.5, "r-true");
      const q = data.queries[state.query].xy;
      star(px(q[0]), py(q[1]));
      if (last && last.node !== best.id) label(last.node, String(last.node));
      label(best.id, String(best.id));
      if (missed) label(truth.id, `${truth.id} (true nearest)`);

      readout.textContent = describe(k, last, best, seen.size, finished);
    }

    function describe(k, e, best, seen, finished) {
      const visited = ` Visited ${seen} of ${N} points so far.`;
      if (!e) {
        return `Start on the top layer, layer ${data.max_level}, at the entry point, node ${data.entry} ` +
          `(distance ${fmt(best.s)} from the query). Only ${layerSize(data.max_level)} points live up here. ` +
          `Move the Step slider to follow the search.` + visited;
      }
      const L = e.layer;
      const where = L === 0 ? "Layer 0 (every point)" : `Layer ${L}`;
      const dropped = e.firstOnLayer && k > 1 ? `Dropped to ${where.toLowerCase()}, still at node ${e.node}. ` : `${where}: `;
      const measured = e.measured.length
        ? `measured ${plural(e.measured.length, "unvisited neighbour")} of node ${e.node} (distance ${fmt(e.sim)}).`
        : `node ${e.node} (distance ${fmt(e.sim)}) has no unvisited neighbours.`;
      let text = dropped + measured.charAt(0).toUpperCase() + measured.slice(1) + " ";
      const beamWidth = L === 0 ? Math.max(state.ef, 1) : 1;
      if (beamWidth > 1) {
        text += `The beam keeps the ${beamWidth} closest found so far (nodes ${e.beam.map((b) => b.id).join(", ")}); ` +
          `the best is node ${best.id} (distance ${fmt(best.s)}).`;
      } else if (best.id !== e.node) {
        text += `Node ${best.id} is closer to the query (distance ${fmt(best.s)}), so the walk moves there next.`;
      } else if (L > 0) {
        text += `None is closer, so the walk drops to layer ${L - 1}, staying at node ${e.node}.`;
      } else {
        text += "None is closer.";
      }
      text += visited;
      if (finished) {
        const found = run.found[0];
        // The lesson's count re-measures the starting point of every layer, so it can exceed the points visited.
        const work = `${run.ndist} distance computations (${seen} points, some measured on more than one layer)`;
        text += beamWidth > 1
          ? ` Nothing left to expand can beat the beam, so the search stops and returns node ${found}.`
          : ` The search stops and returns node ${found}.`;
        text += ` Brute force measures all ${N} points and finds node ${truth.id}: ` +
          (found === truth.id
            ? `the search found the true nearest neighbour with ${work} instead of ${N}.`
            : `the search missed it (node ${truth.id} is at distance ${fmt(truth.s)}), after ${work}. Try a wider beam.`);
      }
      return text;
    }

    rerun();
  });
})();
