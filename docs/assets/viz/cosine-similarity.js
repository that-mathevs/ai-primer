/*
 * cosine-similarity.js: two arrows from the origin, and the three ways to
 * compare them (dot product, cosine, Euclidean distance).
 *
 * The arithmetic mirrors primer/ml/embeddings/similarity.py (dot, cosine,
 * euclidean, l2_normalize) exactly; tests/test_emb_similarity.py runs both
 * and compares them, so the widget can't drift from the lesson.
 */
(function () {
  function dot(a, b) {
    return a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  }

  function norm(a) {
    return Math.sqrt(dot(a, a));
  }

  // Not clamped to [-1, 1], to match the lesson's cosine digit for digit.
  function cosine(a, b) {
    return dot(a, b) / (norm(a) * norm(b));
  }

  function euclidean(a, b) {
    return norm(a.map((ai, i) => ai - b[i]));
  }

  // The same eps guard as l2_normalize: an all-zero arrow stays zero instead of becoming NaN.
  function l2Normalize(a, eps = 1e-12) {
    const n = Math.max(norm(a), eps);
    return a.map((ai) => ai / n);
  }

  // Degrees counterclockwise from "pointing right", and a length, to (x, y).
  function fromPolar(angle, length) {
    const r = (angle * Math.PI) / 180;
    return [length * Math.cos(r), length * Math.sin(r)];
  }

  if (typeof module !== "undefined") module.exports = { dot, norm, cosine, euclidean, l2Normalize, fromPolar };
  if (typeof window === "undefined" || !window.Viz) return;

  const NAME = "cosine-similarity";
  const SVG = "http://www.w3.org/2000/svg";
  const SCALE = 40; // SVG units per unit of length, so length 3 reaches 120
  const SCOPE = `.viz[data-viz="${NAME}"]`;

  // Arrow B needs a second colour the theme doesn't name; it gets a light and a dark shade like the theme's own tokens.
  function injectStyle() {
    if (document.getElementById("viz-style-" + NAME)) return;
    const dark = "--cs-b: #f0a35e;";
    const style = document.createElement("style");
    style.id = "viz-style-" + NAME;
    style.textContent = `
${SCOPE} { --cs-a: var(--p-accent); --cs-b: #b8561b; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) ${SCOPE} { ${dark} } }
:root[data-theme="dark"] ${SCOPE} { ${dark} }
${SCOPE} .cs-stage { display: flex; flex-wrap: wrap; gap: 0.4rem 1.2rem; align-items: center; margin-top: 0.5rem; }
${SCOPE} .cs-plot { flex: 0 1 17rem; width: 100%; max-width: 17rem; margin: 0 auto; }
${SCOPE} .cs-plot svg { display: block; width: 100%; height: auto; overflow: visible; }
${SCOPE} .cs-side { flex: 1 1 15rem; min-width: 0; }
${SCOPE} .viz-readout { margin-top: 0; }
${SCOPE} .viz-readout p { margin: 0 0 0.35rem; }
${SCOPE} .cs-a { color: var(--cs-a); font-weight: 600; }
${SCOPE} .cs-b { color: var(--cs-b); font-weight: 600; }
${SCOPE} .cs-verdict { font-weight: 600; }
${SCOPE} .viz-row input[type="range"].cs-slide-b { accent-color: var(--cs-b); }
`;
    document.head.appendChild(style);
  }

  function el(tag, attrs = {}, parent) {
    const node = document.createElementNS(SVG, tag);
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
    if (parent) parent.appendChild(node);
    return node;
  }

  // 0.4999 -> "0.50", -0.001 -> "0.00" (never "-0.00"), negatives with a true minus sign.
  function num(x) {
    const s = (Math.abs(x) < 0.005 ? 0 : x).toFixed(2);
    return s.replace("-", "−");
  }

  function verdict(c) {
    if (c > 0.95) return "Same direction: as similar as two arrows get.";
    if (c > 0.3) return "Similar direction.";
    if (c > -0.3) return "Close to a right angle: unrelated.";
    if (c > -0.95) return "Pointing away from each other.";
    return "Opposite directions.";
  }

  window.Viz.register(NAME, (root, data) => {
    injectStyle();
    const presets = data.presets;
    const state = {
      a: { ...presets[0].a },
      b: { ...presets[0].b },
      normalize: false,
    };
    // True while a preset moves the sliders, so those moves don't count as the reader's own.
    let applying = false;

    const presetSelect = window.Viz.select(root, {
      label: "Starting arrows",
      options: [...presets.map((p, i) => [i, p.name]), ["custom", "Your own arrows"]],
      value: 0,
      onChange: (v) => { if (v !== "custom") applyPreset(presets[Number(v)]); },
    });

    const deg = (v) => `${v}°`;
    const len = (v) => v.toFixed(1);
    function slider(label, arrow, key, opts, cls) {
      const input = window.Viz.slider(root, {
        label, ...opts, value: state[arrow][key],
        onInput: (v) => {
          state[arrow][key] = v;
          if (!applying) presetSelect.value = "custom";
          draw();
        },
      });
      if (cls) input.classList.add(cls);
      return input;
    }
    const angleOpts = { min: 0, max: 359, step: 1, format: deg };
    const lengthOpts = { min: 0.5, max: 3, step: 0.1, format: len };
    const inputs = {
      aAngle: slider("A: angle", "a", "angle", angleOpts),
      aLength: slider("A: length", "a", "length", lengthOpts),
      bAngle: slider("B: angle", "b", "angle", angleOpts, "cs-slide-b"),
      bLength: slider("B: length", "b", "length", lengthOpts, "cs-slide-b"),
    };

    window.Viz.select(root, {
      label: "Normalize to length 1",
      options: [["off", "No: keep each arrow's own length"], ["on", "Yes: trim both arrows to length 1"]],
      value: "off",
      onChange: (v) => { state.normalize = v === "on"; draw(); },
    });

    function set(input, value) {
      input.value = value;
      // Let the helper refresh the value shown beside the slider, and redraw.
      input.dispatchEvent(new Event("input"));
    }

    function applyPreset(p) {
      applying = true;
      set(inputs.aAngle, p.a.angle);
      set(inputs.aLength, p.a.length);
      set(inputs.bAngle, p.b.angle);
      set(inputs.bLength, p.b.length);
      applying = false;
    }

    const stage = document.createElement("div");
    stage.className = "cs-stage";
    root.appendChild(stage);
    const plot = document.createElement("div");
    plot.className = "cs-plot";
    // The picture repeats what the readout says in words, so screen readers skip it.
    plot.setAttribute("aria-hidden", "true");
    stage.appendChild(plot);
    const side = document.createElement("div");
    side.className = "cs-side";
    stage.appendChild(side);
    const readout = window.Viz.readout(side);

    const svg = el("svg", { viewBox: "-140 -140 280 280", focusable: "false" }, plot);
    const defs = el("defs", {}, svg);
    ["a", "b"].forEach((k) => {
      const marker = el("marker", {
        id: `cs-head-${k}`, viewBox: "0 0 10 10", refX: "8", refY: "5",
        markerWidth: "4", markerHeight: "4", orient: "auto-start-reverse",
      }, defs);
      el("path", { d: "M0,0 L10,5 L0,10 z", style: `fill: var(--cs-${k})` }, marker);
    });
    // Grid: axes, the unit circle every normalized arrow lands on, and the longest reach.
    el("line", { x1: -130, y1: 0, x2: 130, y2: 0, style: "stroke: var(--p-border); stroke-width: 1" }, svg);
    el("line", { x1: 0, y1: -130, x2: 0, y2: 130, style: "stroke: var(--p-border); stroke-width: 1" }, svg);
    el("circle", { r: 3 * SCALE, style: "fill: none; stroke: var(--p-border); stroke-width: 1" }, svg);
    const unit = el("circle", { r: SCALE, style: "fill: none; stroke: var(--p-muted); stroke-width: 1; stroke-dasharray: 3 3" }, svg);
    // Below the circle, where the arrows and the dashed gap rarely reach.
    const unitLabel = el("text", { x: 0, y: SCALE + 13, "text-anchor": "middle", style: "fill: var(--p-muted); font-size: 10px" }, svg);
    unitLabel.textContent = "length 1";
    const layer = el("g", {}, svg);

    const pt = ([x, y]) => [x * SCALE, -y * SCALE]; // SVG's y axis points down

    function arrow(parent, v, k, faint) {
      const [x, y] = pt(v);
      el("line", {
        x1: 0, y1: 0, x2: x, y2: y, "marker-end": `url(#cs-head-${k})`,
        style: `stroke: var(--cs-${k}); stroke-width: ${faint ? 1.5 : 3}; stroke-linecap: round; opacity: ${faint ? 0.3 : 1}`,
      }, parent);
    }

    function label(parent, v, text, k, side) {
      const n = norm(v) || 1;
      const [ux, uy] = [v[0] / n, v[1] / n];
      // Just past the tip, nudged sideways (side +1 counterclockwise, -1
      // clockwise) so the two labels stay apart even when the arrows are close.
      const [x, y] = pt([v[0] + ux * 0.3 - side * uy * 0.25, v[1] + uy * 0.3 + side * ux * 0.25]);
      const t = el("text", {
        x, y: y + 5, "text-anchor": "middle",
        style: `fill: var(--cs-${k}); font-size: 15px; font-weight: 700`,
      }, parent);
      t.textContent = text;
    }

    // The signed turn from A to B the short way round, in degrees, in (-180, 180].
    function turnFromAToB() {
      const turn = (((state.b.angle - state.a.angle) % 360) + 540) % 360 - 180;
      return turn === -180 ? 180 : turn;
    }

    function angleArc(parent, turn, shortest) {
      const from = state.a.angle;
      if (Math.abs(turn) < 1) return;
      const r = 0.45 * SCALE;
      const [x1, y1] = pt(fromPolar(from, 0.45));
      const [x2, y2] = pt(fromPolar(from + turn, 0.45));
      // Counterclockwise on screen is SVG's negative sweep, because y is flipped.
      el("path", {
        d: `M${x1},${y1} A${r},${r} 0 0 ${turn > 0 ? 0 : 1} ${x2},${y2}`,
        style: "fill: none; stroke: var(--p-fg); stroke-width: 1.5",
      }, parent);
      // Out along the bisector until the wedge is about 0.7 units wide. If that
      // is past the shorter arrow's tip the label would sit among the arrow
      // labels, so the arc goes unlabelled; the readout always gives the angle.
      const reach = Math.max(0.8, 0.7 / (2 * Math.sin((Math.abs(turn) * Math.PI) / 360)));
      if (reach + 0.15 > shortest) return;
      const [lx, ly] = pt(fromPolar(from + turn / 2, reach));
      const t = el("text", {
        x: lx, y: ly + 4, "text-anchor": "middle",
        style: "fill: var(--p-fg); font-size: 12px",
      }, parent);
      t.textContent = `${Math.round(Math.abs(turn))}°`;
    }

    function draw() {
      const rawA = fromPolar(state.a.angle, state.a.length);
      const rawB = fromPolar(state.b.angle, state.b.length);
      const a = state.normalize ? l2Normalize(rawA) : rawA;
      const b = state.normalize ? l2Normalize(rawB) : rawB;
      const d = dot(a, b);
      const na = norm(a);
      const nb = norm(b);
      const c = cosine(a, b);
      const dist = euclidean(a, b);

      layer.replaceChildren();
      unit.style.strokeWidth = state.normalize ? "1.5" : "1";
      // After normalizing, the raw arrows stay as faint ghosts so the trim is visible.
      if (state.normalize) {
        arrow(layer, rawA, "a", true);
        arrow(layer, rawB, "b", true);
      }
      // The gap between the tips is exactly what Euclidean distance measures.
      const [ax, ay] = pt(a);
      const [bx, by] = pt(b);
      el("line", {
        x1: ax, y1: ay, x2: bx, y2: by,
        style: "stroke: var(--p-muted); stroke-width: 1.5; stroke-dasharray: 5 4",
      }, layer);
      const turn = turnFromAToB();
      angleArc(layer, turn, Math.min(na, nb));
      // Longer arrow first, so a shorter one lying along it stays visible on top.
      const order = na >= nb ? [[a, "a"], [b, "b"]] : [[b, "b"], [a, "a"]];
      order.forEach(([v, k]) => arrow(layer, v, k, false));
      // Each label leans away from the other arrow.
      const sideA = turn > 0 ? -1 : 1;
      label(layer, a, "A", "a", sideA);
      label(layer, b, "B", "b", -sideA);

      const lines = [
        `<p><span class="cs-a">A</span> = (${num(a[0])}, ${num(a[1])}), length ${num(na)}.<br>` +
          `<span class="cs-b">B</span> = (${num(b[0])}, ${num(b[1])}), length ${num(nb)}.</p>`,
        `<p>Angle between them: ${Math.round(Math.abs(turn))}°.</p>`,
        `<p>Dot product A·B = ${num(d)}.</p>`,
        `<p>Cosine = ${num(d)} / (${num(na)} × ${num(nb)}) = <strong>${num(c)}</strong>.</p>`,
        `<p>Euclidean distance (the dashed gap) = ${num(dist)}.</p>`,
        state.normalize
          ? `<p>Both lengths are 1, so the dot product is the cosine, and distance² = ${num(dist * dist)} = 2 − 2 × ${c < -0.005 ? `(${num(c)})` : num(c)}.</p>`
          : "",
        `<p class="cs-verdict">${verdict(c)}</p>`,
      ];
      readout.innerHTML = lines.join("");
    }

    draw();
  });
})();
