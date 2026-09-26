/*
 * viz.js: shared helpers for the lessons' interactive visualizations.
 *
 * A lesson places <div class="viz" data-viz="NAME" aria-label="..."></div> in
 * its docstring; the site loads this file, the lesson's exported numbers
 * (window.PRIMER_VIZ_DATA, written from the lesson's viz_data() at build
 * time) and docs/assets/viz/NAME.js, which calls Viz.register(NAME, draw).
 * Every control is a native <input> or <select>, so it works from the
 * keyboard, and every readout is a polite live region, so screen readers
 * hear the result.
 */
(function () {
  const registry = {};

  function start() {
    document.querySelectorAll(".viz[data-viz]").forEach((el) => {
      const draw = registry[el.dataset.viz];
      if (!draw || el.dataset.ready) return;
      el.dataset.ready = "1";
      // aria-label names a group; on a plain div screen readers ignore it.
      el.setAttribute("role", "group");
      draw(el, (window.PRIMER_VIZ_DATA || {})[el.dataset.viz]);
    });
  }

  const Viz = {
    register(name, draw) {
      registry[name] = draw;
      if (document.readyState !== "loading") start();
    },

    // A labelled slider with its current value shown beside it.
    slider(parent, { label, min, max, step = 1, value, format = String, onInput }) {
      const row = document.createElement("label");
      row.className = "viz-row";
      const name = document.createElement("span");
      name.textContent = label;
      const input = document.createElement("input");
      Object.assign(input, { type: "range", min, max, step, value });
      const shown = document.createElement("output");
      const update = () => {
        shown.textContent = format(Number(input.value));
        input.setAttribute("aria-valuetext", shown.textContent);
        onInput(Number(input.value));
      };
      input.addEventListener("input", update);
      row.append(name, input, shown);
      parent.appendChild(row);
      shown.textContent = format(Number(value));
      input.setAttribute("aria-valuetext", shown.textContent);
      return input;
    },

    // A labelled choice from a short list.
    select(parent, { label, options, value, onChange }) {
      const row = document.createElement("label");
      row.className = "viz-row";
      const name = document.createElement("span");
      name.textContent = label;
      const select = document.createElement("select");
      options.forEach(([val, text]) => select.add(new Option(text, val, false, String(val) === String(value))));
      select.addEventListener("change", () => onChange(select.value));
      row.append(name, select);
      parent.appendChild(row);
      return select;
    },

    // The result, announced politely to screen readers when it changes.
    readout(parent) {
      const out = document.createElement("div");
      out.className = "viz-readout";
      out.setAttribute("role", "status");
      out.setAttribute("aria-live", "polite");
      parent.appendChild(out);
      return out;
    },

    // 17_200_000_000 -> "17.2 GB"
    bytes(n) {
      const units = ["B", "KB", "MB", "GB", "TB"];
      let i = 0;
      while (n >= 1000 && i < units.length - 1) {
        n /= 1000;
        i += 1;
      }
      return `${n >= 100 || i === 0 ? Math.round(n) : n.toFixed(1)} ${units[i]}`;
    },
  };

  window.Viz = Viz;
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
})();
