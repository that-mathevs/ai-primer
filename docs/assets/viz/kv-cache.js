/*
 * kv-cache.js: how much memory a model's KV cache takes, against one GPU.
 *
 * The arithmetic mirrors primer/ml/inference.py (kv_cache_bytes_per_token,
 * kv_cache_bytes, weight_bytes) exactly; tests/test_viz.py runs both and
 * compares them, so the widget can't drift from the lesson.
 */
(function () {
  // A key and a value, per layer, per KV head, per dimension, at bits/8 bytes each.
  function kvCacheBytesPerToken(layers, kvHeads, headDim, bits = 16) {
    return Math.floor((2 * layers * kvHeads * headDim * bits) / 8);
  }

  function kvCacheBytes(context, layers, kvHeads, headDim, bits = 16, batch = 1) {
    return batch * context * kvCacheBytesPerToken(layers, kvHeads, headDim, bits);
  }

  function weightBytes(params, bits = 16) {
    return (params * bits) / 8;
  }

  if (typeof module !== "undefined") module.exports = { kvCacheBytesPerToken, kvCacheBytes, weightBytes };
  if (typeof window === "undefined" || !window.Viz) return;

  window.Viz.register("kv-cache", (el, data) => {
    const presets = data.presets;
    const gpu = data.gpu_bytes;
    const state = { preset: 0, logContext: 15, batch: 1, bits: 16 };

    window.Viz.select(el, {
      label: "Model shape",
      options: presets.map((p, i) => [i, p.name]),
      value: 0,
      onChange: (v) => { state.preset = Number(v); draw(); },
    });
    // Powers of two, because context windows are quoted that way (8k, 128k, 1M).
    window.Viz.slider(el, {
      label: "Context per request", min: 9, max: 20, value: state.logContext,
      format: (v) => tokens(2 ** v),
      onInput: (v) => { state.logContext = v; draw(); },
    });
    window.Viz.slider(el, {
      label: "Requests at once", min: 1, max: 64, value: state.batch,
      format: String,
      onInput: (v) => { state.batch = v; draw(); },
    });
    window.Viz.select(el, {
      label: "Precision",
      options: [[16, "16-bit (2 bytes)"], [8, "8-bit (1 byte)"], [4, "4-bit (half a byte)"]],
      value: 16,
      onChange: (v) => { state.bits = Number(v); draw(); },
    });

    const bars = document.createElement("div");
    bars.className = "viz-bars";
    bars.setAttribute("aria-hidden", "true");
    el.appendChild(bars);
    const readout = window.Viz.readout(el);
    const note = document.createElement("p");
    note.className = "viz-note";
    note.textContent = `The vertical line is one ${window.Viz.bytes(gpu)} GPU. Weights and cache share it.`;
    el.appendChild(note);

    function bar(label, parts, scale) {
      const row = document.createElement("div");
      row.className = "viz-bar";
      const name = document.createElement("span");
      name.textContent = label;
      const track = document.createElement("div");
      track.className = "track";
      let left = 0;
      parts.forEach(([bytes, cls]) => {
        const fill = document.createElement("div");
        fill.className = `fill ${cls}`;
        fill.style.left = `${Math.min(100, (left / scale) * 100)}%`;
        fill.style.width = `${Math.max(0, Math.min(100, ((left + bytes) / scale) * 100) - Math.min(100, (left / scale) * 100))}%`;
        track.appendChild(fill);
        left += bytes;
      });
      const limit = document.createElement("div");
      limit.className = "limit";
      limit.style.left = `${Math.min(100, (gpu / scale) * 100)}%`;
      track.appendChild(limit);
      row.append(name, track);
      return row;
    }

    function draw() {
      const p = presets[state.preset];
      const context = 2 ** state.logContext;
      const perToken = kvCacheBytesPerToken(p.layers, p.kv_heads, p.head_dim, state.bits);
      const cache = kvCacheBytes(context, p.layers, p.kv_heads, p.head_dim, state.bits, state.batch);
      const weights = weightBytes(p.params, state.bits);
      const total = weights + cache;
      // Keep the GPU line visible even when everything fits comfortably.
      const scale = Math.max(total, gpu) * 1.05;
      bars.replaceChildren(bar("Weights + cache", [[weights, "second"], [cache, ""]], scale));
      const fits = total <= gpu;
      const free = gpu - weights;
      const perRequest = cache / state.batch;
      const room = free > 0 ? Math.floor(free / perRequest) : 0;
      readout.textContent =
        `${window.Viz.bytes(perToken)} per token \u00d7 ${tokens(context)} \u00d7 ${state.batch} ` +
        `= ${window.Viz.bytes(cache)} of cache, plus ${window.Viz.bytes(weights)} of weights: ` +
        `${window.Viz.bytes(total)} total. ` +
        (fits ? "It fits on one GPU. " : "It does not fit on one GPU. ") +
        (free > 0
          ? `Beside the weights there is room for ${room} request${room === 1 ? "" : "s"} of this length.`
          : "The weights alone are larger than the GPU.");
    }

    draw();
  });

  // 131072 -> "131,072 tokens"
  function tokens(n) {
    return `${n.toLocaleString("en-US")} tokens`;
  }
})();
