/*
 * papers.js: the shared engine behind every annotated paper companion.
 *
 * A page loads, in this order:
 *   <script src="assets/glossary.js"></script>        window.PRIMER_GLOSSARY (shared terms)
 *   <script> window.PAPER = { symbols, blocks, terms, onReady } </script>   page-specific data
 *   <script defer src=".../katex.min.js"></script>
 *   <script defer src="assets/papers.js"></script>
 *
 * On DOMContentLoaded it wires up, with no build step:
 *   - hover / focus / tap tooltips for  [data-t]   (a glossary key),
 *                                       [data-tip] (inline HTML, optional data-tip-title),
 *                                       [data-sym] (an equation symbol from PAPER.symbols);
 *   - KaTeX for .eq[data-tex] (display) and .m[data-tex] (inline), with
 *     \htmlData{sym=KEY}{...} marking hoverable symbols;
 *   - .symtable[data-syms="a,b,c"] tables generated from PAPER.symbols;
 *   - figure.ix interactive SVG diagrams: .blk[data-block] parts explained in the .panel;
 *   - a[data-lesson] links resolved against the site root;
 *   - the theme button (#themeBtn), the sticky table of contents, and the
 *     #glossary-list generated from every term the page uses.
 * It also exposes window.Papers.lineChart and window.Papers.heatmap for plots.
 * Authoring guide: docs/papers/README.md.
 */
(function () {
  "use strict";
  const P = (window.Papers = window.Papers || {});
  const page = { terms: {}, symbols: {}, blocks: {} };

  // ------------------------------------------------------------------ helpers
  const esc = (s) =>
    String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  // Companions are served from <site>/papers/<slug>.html (docs/html/papers/ after `make docs`).
  // Opened straight from the source tree (docs/papers/), the built site is at ../html/.
  function siteRoot() {
    const path = decodeURIComponent(location.pathname);
    return /\/docs\/papers\/[^/]*$/.test(path) ? "../html/" : "../";
  }
  P.lessonHref = (path) => siteRoot() + String(path).replace(/^\/+/, "");

  function termEntry(key) {
    const k = String(key).toLowerCase();
    const shared = window.PRIMER_GLOSSARY || {};
    return shared[k] || page.terms[k] || null;
  }

  function texToHtml(tex, display) {
    if (!window.katex) return esc(tex);
    return window.katex.renderToString(tex, katexOptions(display));
  }

  function katexOptions(display) {
    return {
      displayMode: !!display,
      throwOnError: false,
      strict: "ignore",
      // \htmlData is the only "trusted" command: it tags symbols with data-sym.
      trust: (ctx) => ctx.command === "\\htmlData",
    };
  }

  // ----------------------------------------------------------------- tooltips
  let tip = null;
  let owner = null;
  let hideTimer = null;
  let lastPointer = "mouse";

  function tipHtml(el) {
    if (el.dataset.tip) {
      const title = el.dataset.tipTitle || el.textContent.trim();
      return `<b class="tt">${esc(title)}</b>${el.dataset.tip}`;
    }
    if (el.dataset.t) {
      const e = termEntry(el.dataset.t);
      if (!e) return `<b class="tt">${esc(el.dataset.t)}</b><em>No definition yet.</em>`;
      const title = e.term || el.textContent.trim() || el.dataset.t;
      const lesson = e.lesson ? `<span class="meta"><a href="${P.lessonHref(e.lesson)}">Build it in code →</a></span>` : "";
      return `<b class="tt">${esc(title)}</b>${e.def}${lesson}`;
    }
    if (el.dataset.sym) {
      const s = page.symbols[el.dataset.sym];
      if (!s) return `<b class="tt">${esc(el.dataset.sym)}</b><em>No note for this symbol yet.</em>`;
      return (
        `<b class="tt">${s.name ? esc(s.name) : texToHtml(s.tex || el.dataset.sym)}</b>${s.meaning}` +
        (s.shape ? `<span class="meta"><strong>Shape:</strong> ${s.shape}</span>` : "") +
        (s.example ? `<span class="meta"><strong>Example:</strong> ${s.example}</span>` : "")
      );
    }
    return "";
  }

  function position(el) {
    const r = el.getBoundingClientRect();
    const t = tip.getBoundingClientRect();
    let left = r.left + r.width / 2 - t.width / 2;
    left = Math.max(12, Math.min(left, window.innerWidth - t.width - 12));
    let top = r.bottom + 8;
    if (top + t.height > window.innerHeight - 8 && r.top - t.height - 8 > 0) top = r.top - t.height - 8;
    tip.style.left = `${left}px`;
    tip.style.top = `${top}px`;
  }

  function show(el) {
    clearTimeout(hideTimer);
    const html = tipHtml(el);
    if (!html) return;
    if (owner && owner !== el) {
      owner.classList.remove("on");
      owner.removeAttribute("aria-describedby");
    }
    owner = el;
    el.classList.add("on");
    tip.innerHTML = html;
    // A tooltip may not hold anything focusable, so a tip with a link in it is a plain
    // described region instead; Tab from the term moves into it (see initTooltips).
    if (tip.querySelector("a[href]")) tip.removeAttribute("role");
    else tip.setAttribute("role", "tooltip");
    // Symbols live in KaTeX's aria-hidden layer: describe their keyboard host instead.
    if (!el.closest('[aria-hidden="true"]')) el.setAttribute("aria-describedby", "papers-tip");
    tip.removeAttribute("aria-hidden");
    tip.classList.add("show");
    position(el);
  }

  function hide() {
    clearTimeout(hideTimer);
    tip.classList.remove("show");
    // Hidden from assistive tech too, so a stale definition is never read out.
    tip.setAttribute("aria-hidden", "true");
    if (owner) {
      owner.classList.remove("on");
      owner.removeAttribute("aria-describedby");
    }
    owner = null;
  }

  const hideSoon = () => {
    clearTimeout(hideTimer);
    hideTimer = setTimeout(hide, 180);
  };

  const TIP_SEL = "[data-t],[data-tip],[data-sym]";

  const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select, textarea, summary, [tabindex]:not([tabindex="-1"])';

  // The next place Tab would land after `el`, skipping the tip itself and anything hidden.
  function focusableAfter(el) {
    return [...document.querySelectorAll(FOCUSABLE)].find(
      (n) =>
        el.compareDocumentPosition(n) & Node.DOCUMENT_POSITION_FOLLOWING &&
        !el.contains(n) &&
        !tip.contains(n) &&
        !n.closest('[aria-hidden="true"]') &&
        n.getClientRects().length
    );
  }

  function initTooltips() {
    tip = document.createElement("div");
    tip.className = "tip";
    tip.id = "papers-tip";
    tip.setAttribute("role", "tooltip");
    tip.setAttribute("aria-hidden", "true");
    document.body.appendChild(tip);
    tip.addEventListener("pointerenter", () => clearTimeout(hideTimer));
    tip.addEventListener("pointerleave", hideSoon);
    const hint = document.createElement("p");
    hint.id = "papers-sym-hint";
    hint.hidden = true;
    hint.textContent = "Use the left and right arrow keys to step through the symbols; each one is explained as you reach it.";
    document.body.appendChild(hint);

    // Focus coming back from the tip (Shift+Tab or Escape) should not reopen what was just closed.
    let skipShow = null;
    document.addEventListener("pointerdown", (e) => (lastPointer = e.pointerType || "mouse"), true);
    document.addEventListener("pointerover", (e) => {
      if (e.pointerType === "touch") return;
      const el = e.target.closest && e.target.closest(TIP_SEL);
      if (el) show(el);
    });
    document.addEventListener("pointerout", (e) => {
      const el = e.target.closest && e.target.closest(TIP_SEL);
      if (el && !(e.relatedTarget && el.contains(e.relatedTarget))) hideSoon();
    });
    document.addEventListener("focusin", (e) => {
      if (tip.contains(e.target)) return clearTimeout(hideTimer);
      const el = e.target.closest && e.target.closest(TIP_SEL);
      if (el && el === skipShow) skipShow = null;
      else if (el) show(el);
    });
    document.addEventListener("focusout", (e) => {
      // Moving from a term into its own tip (to reach "Build it in code") keeps the tip open.
      const into = e.relatedTarget && (tip.contains(e.relatedTarget) || e.relatedTarget === owner);
      if (into) return;
      if (tip.contains(e.target) || (e.target.closest && e.target.closest(TIP_SEL))) hideSoon();
    });
    document.addEventListener("click", (e) => {
      const el = e.target.closest && e.target.closest(TIP_SEL);
      if (el) {
        // Tap toggles; a mouse click keeps it open.
        if (owner === el && tip.classList.contains("show") && lastPointer === "touch") hide();
        else show(el);
      } else if (!tip.contains(e.target)) hide();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        const back = tip.contains(document.activeElement) ? owner : null;
        hide();
        if (back) {
          skipShow = back;
          back.focus();
        }
        return;
      }
      // The tip sits at the end of the page, so Tab would never reach its links on its own:
      // Tab from the term steps into the tip, and Tab off its last link carries on after the term.
      if (e.key !== "Tab" || e.altKey || e.ctrlKey || e.metaKey || !owner || !tip.classList.contains("show")) return;
      const links = [...tip.querySelectorAll("a[href]")];
      if (!links.length) return;
      const at = links.indexOf(document.activeElement);
      if (document.activeElement === owner && !e.shiftKey) {
        e.preventDefault();
        links[0].focus();
      } else if (at === 0 && e.shiftKey) {
        e.preventDefault();
        skipShow = owner;
        owner.focus();
      } else if (at === links.length - 1 && !e.shiftKey) {
        e.preventDefault();
        const next = focusableAfter(owner);
        hide();
        if (next) next.focus();
      }
    });
    window.addEventListener("scroll", () => owner && position(owner), { passive: true });
    window.addEventListener("resize", () => owner && position(owner));
  }

  /**
   * Equation symbols are tagged inside KaTeX's visual layer, which is aria-hidden (screen
   * readers get the MathML copy and the Symbols table instead). A tab stop in there would
   * be silent, so the equation itself takes the one tab stop and the arrow keys walk its
   * symbols, showing each one's tooltip exactly as hovering does.
   */
  function makeSymbolHost(host) {
    if (host.dataset.symHost) return;
    host.dataset.symHost = "1";
    if (!host.hasAttribute("tabindex")) host.tabIndex = 0;
    host.setAttribute("aria-describedby", "papers-sym-hint");
    let at = 0;
    const syms = () => [...host.querySelectorAll("[data-sym]")];
    const pick = (i, speak) => {
      const list = syms();
      if (!list.length) return;
      at = (i + list.length) % list.length;
      show(list[at]);
      if (speak) P.announce(tip.innerText, 0);
    };
    // Keyboard focus shows the current symbol; a click focuses the host too, but the
    // clicked symbol is the one to show, so it only moves the place.
    host.addEventListener("focus", () => host.matches(":focus-visible") && pick(at, false));
    host.addEventListener("click", (e) => {
      const i = syms().indexOf(e.target.closest && e.target.closest("[data-sym]"));
      if (i >= 0) at = i;
    });
    host.addEventListener("blur", hideSoon);
    host.addEventListener("keydown", (e) => {
      const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
      if (step) pick(at + step, true);
      else if (e.key === "Home") pick(0, true);
      else if (e.key === "End") pick(-1, true);
      else return;
      e.preventDefault();
    });
  }

  function makeFocusable(root) {
    root.querySelectorAll(TIP_SEL).forEach((el) => {
      const hidden = el.closest('[aria-hidden="true"]');
      if (!hidden) {
        if (!el.hasAttribute("tabindex")) el.tabIndex = 0;
        return;
      }
      const host = hidden.parentElement && (hidden.parentElement.closest(".eq, .m") || hidden.parentElement.closest(".katex"));
      if (host) makeSymbolHost(host);
    });
  }

  // ------------------------------------------------------------ announcements
  // One polite live region for the whole page. Readouts change on every pointer move and
  // every animation frame, so each is announced only once it has settled.
  let live = null;
  let liveTimer = null;
  let userActed = false;
  P.announce = function (text, delay = 350) {
    if (!live) return;
    clearTimeout(liveTimer);
    liveTimer = setTimeout(() => {
      live.textContent = String(text || "").replace(/\s+/g, " ").trim();
    }, delay);
  };

  function initLiveReadouts() {
    live = document.createElement("div");
    live.className = "sr-only";
    live.id = "papers-live";
    live.setAttribute("aria-live", "polite");
    document.body.appendChild(live);
    // Only what the reader causes is announced, not the page drawing itself on load.
    ["pointerdown", "keydown"].forEach((t) => document.addEventListener(t, () => (userActed = true), true));
    const watched = new Set();
    const obs = new MutationObserver((records) => {
      if (!userActed) return;
      const changed = new Set();
      records.forEach((r) => {
        const node = r.target.nodeType === 1 ? r.target : r.target.parentElement;
        const ro = node && node.closest(".readout");
        if (ro && watched.has(ro)) changed.add(ro);
      });
      changed.forEach((ro) => P.announce(ro.textContent));
    });
    const watch = () =>
      document.querySelectorAll("main .readout").forEach((ro) => {
        // Readouts the page already made live announce themselves.
        if (watched.has(ro) || ro.closest("[aria-live]")) return;
        watched.add(ro);
        obs.observe(ro, { childList: true, characterData: true, subtree: true });
      });
    watch();
    return watch;
  }

  // A role="img" is one picture to assistive tech, so anything focusable inside it would be a
  // silent tab stop. A picture with controls in it becomes a named group instead.
  function exposeControlsInPictures(root = document) {
    root.querySelectorAll('[role="img"]').forEach((el) => {
      if (el.querySelector(FOCUSABLE)) el.setAttribute("role", "group");
    });
  }

  // --------------------------------------------------------------------- math
  P.renderMath = function (root = document) {
    root.querySelectorAll(".eq[data-tex], .m[data-tex]").forEach((el) => {
      const display = el.classList.contains("eq");
      if (window.katex) {
        try {
          window.katex.render(el.dataset.tex, el, katexOptions(display));
          return;
        } catch (err) {
          console.warn("KaTeX failed on", el.dataset.tex, err);
        }
      }
      el.textContent = el.dataset.fallback || el.dataset.tex;
      el.classList.add("eq-fallback");
    });
    makeFocusable(root);
  };

  P.renderSymbolTables = function (root = document) {
    root.querySelectorAll(".symtable[data-syms]").forEach((el) => {
      const rows = el.dataset.syms
        .split(",")
        .map((k) => k.trim())
        .filter(Boolean)
        .map((k) => {
          const s = page.symbols[k];
          if (!s) {
            console.warn(`symtable: no symbol "${k}" in PAPER.symbols`);
            return "";
          }
          return `<tr><td>${texToHtml(s.tex || k)}</td><td>${s.name ? `<strong>${esc(s.name)}.</strong> ` : ""}${s.meaning}</td><td>${s.shape || ""}</td><td>${s.example || ""}</td></tr>`;
        })
        .join("");
      el.innerHTML =
        `<div class="tablewrap"><table><caption>Symbols</caption><thead><tr><th>Symbol</th><th>Meaning here</th><th>Shape or range</th><th>In the worked example</th></tr></thead><tbody>${rows}</tbody></table></div>`;
    });
  };

  // ----------------------------------------------------------------- diagrams
  // Plain text of an HTML snippet (a block title may hold markup).
  const plain = (html) => {
    const d = document.createElement("div");
    d.innerHTML = html;
    return d.textContent.replace(/\s+/g, " ").trim();
  };

  let panelCount = 0;
  P.wireDiagrams = function (root = document) {
    root.querySelectorAll("figure.ix").forEach((fig) => {
      const panel = fig.querySelector(".panel");
      const parts = [...fig.querySelectorAll(".blk[data-block]")];
      if (panel) {
        if (!panel.id) panel.id = `papers-panel-${++panelCount}`;
        if (!panel.hasAttribute("aria-live")) panel.setAttribute("aria-live", "polite");
      }
      const activate = (key) => {
        parts.forEach((b) => b.classList.toggle("on", b.dataset.block === key));
        if (!panel) return;
        const info = page.blocks[key];
        if (!info) {
          panel.innerHTML = `<p class="hint">No note for “${esc(key)}” yet.</p>`;
          return;
        }
        panel.innerHTML =
          `<h5>${info.title}</h5>${info.body}` +
          (info.shape ? `<div class="shape">${info.shape}</div>` : "") +
          (info.lesson ? `<p><a href="${P.lessonHref(info.lesson)}">Build it in code →</a></p>` : "");
        P.renderMath(panel);
      };
      // The diagram's parts are buttons a screen reader can reach, each named after the note
      // it opens; the panel beside them is where that note appears. So the picture is a group
      // of named parts, not a single image with its parts hidden inside.
      if (parts.length) fig.querySelectorAll('svg[role="img"]').forEach((svg) => svg.setAttribute("role", "group"));
      parts.forEach((b) => {
        if (!b.hasAttribute("tabindex")) b.setAttribute("tabindex", "0");
        if (!b.hasAttribute("role")) b.setAttribute("role", "button");
        if (!b.hasAttribute("aria-label")) {
          const info = page.blocks[b.dataset.block];
          const text = [...b.querySelectorAll("text")].map((t) => t.textContent.trim()).filter(Boolean).join(" ");
          b.setAttribute("aria-label", (info && info.title ? plain(info.title) : "") || text || b.dataset.block);
        }
        if (panel) b.setAttribute("aria-controls", panel.id);
        b.addEventListener("pointerenter", () => activate(b.dataset.block));
        b.addEventListener("focus", () => activate(b.dataset.block));
        b.addEventListener("click", () => activate(b.dataset.block));
        b.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            activate(b.dataset.block);
          }
        });
      });
    });
  };

  // ------------------------------------------------------------ lesson links
  function wireLessonLinks(root = document) {
    root.querySelectorAll("a[data-lesson]").forEach((a) => (a.href = P.lessonHref(a.dataset.lesson)));
  }

  // -------------------------------------------------------------------- theme
  function initTheme() {
    // docs/assets/theme.js owns the switcher and the saved choice; diagrams
    // drawn on canvas just need to hear about changes so they can redraw.
    document.addEventListener("primer:theme", () => document.dispatchEvent(new CustomEvent("papers:theme")));
  }

  // ---------------------------------------------------------------------- TOC
  function initToc() {
    const details = document.querySelector("nav.toc details");
    if (details && window.matchMedia("(max-width: 900px)").matches) details.open = false;
    const links = [...document.querySelectorAll('nav.toc a[href^="#"]')];
    const byId = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
    const targets = [...byId.keys()].map((id) => document.getElementById(id)).filter(Boolean);
    if (!("IntersectionObserver" in window) || !targets.length) return;
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((en) => {
          if (!en.isIntersecting) return;
          links.forEach((a) => a.classList.remove("current"));
          const a = byId.get(en.target.id);
          if (a) a.classList.add("current");
        });
      },
      { rootMargin: "0px 0px -70% 0px" }
    );
    targets.forEach((t) => io.observe(t));
  }

  // ----------------------------------------------------------------- glossary
  function buildGlossary() {
    const list = document.getElementById("glossary-list");
    if (!list) return;
    const items = new Map();
    document.querySelectorAll("main [data-t]").forEach((el) => {
      const key = el.dataset.t.toLowerCase();
      const e = termEntry(key);
      if (e && !items.has(key)) items.set(key, { label: e.term || key, def: e.def, lesson: e.lesson });
    });
    document.querySelectorAll("main [data-tip][data-tip-title]").forEach((el) => {
      const key = el.dataset.tipTitle.toLowerCase();
      if (!items.has(key)) items.set(key, { label: el.dataset.tipTitle, def: el.dataset.tip });
    });
    const sorted = [...items.values()].sort((a, b) => a.label.localeCompare(b.label));
    list.innerHTML = sorted
      .map(
        (i) =>
          `<dt>${esc(i.label)}</dt><dd>${i.def}${i.lesson ? ` <a href="${P.lessonHref(i.lesson)}">Lesson →</a>` : ""}</dd>`
      )
      .join("");
  }

  /** Glossary keys used on the page with no definition anywhere (handy while authoring). */
  P.missingTerms = () =>
    [...new Set([...document.querySelectorAll("[data-t]")].map((el) => el.dataset.t.toLowerCase()))].filter(
      (k) => !termEntry(k)
    );

  // -------------------------------------------------------------------- plots
  function niceTicks(min, max, count = 5) {
    const range = max - min || 1;
    let step = Math.pow(10, Math.floor(Math.log10(range / count)));
    const err = (count / range) * step;
    if (err <= 0.15) step *= 10;
    else if (err <= 0.35) step *= 5;
    else if (err <= 0.75) step *= 2;
    const ticks = [];
    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) ticks.push(+v.toPrecision(12));
    return ticks;
  }

  const SVGNS = "http://www.w3.org/2000/svg";
  const svgEl = (tag, attrs = {}) => {
    const n = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    return n;
  };
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  // Series are told apart by line pattern as well as colour, so the chart reads the same
  // in greyscale or to a reader who cannot separate the hues. Its legend draws both.
  const DASHES = ["", "8 4", "2 3", "10 3 2 3", "5 5"];
  const SERIES_COLOURS = ["--accent", "--active", "--soft-stroke", "--attn-stroke", "--lin-stroke"];

  /**
   * Line chart with a hover crosshair, also driven from the keyboard.
   *   Papers.lineChart(container, {
   *     series: [{ name, color, points: [[x, y], ...] }],   // points sorted by x
   *     xLabel, yLabel, xFmt, yFmt, logX, yMin, yMax, width, height,
   *     readout: element that receives the hover text (optional),
   *     legend: false to leave out the legend drawn under a chart of two or more series
   *   })  ->  { update(series) }
   * The chart is one tab stop: the arrow keys (Page Up/Down, Home, End) move the crosshair
   * along the data and write the same readout the pointer does.
   */
  P.lineChart = function (container, opts) {
    const W = opts.width || 640;
    const H = opts.height || 280;
    const m = { l: 62, r: 18, t: 12, b: 42 };
    const fx = opts.xFmt || ((v) => String(+v.toPrecision(4)));
    const fy = opts.yFmt || ((v) => String(+v.toPrecision(4)));
    // A slider is the closest standard widget: one value (a place along x) picked with the arrow keys.
    const svg = svgEl("svg", {
      viewBox: `0 0 ${W} ${H}`,
      role: "slider",
      tabindex: "0",
      "aria-label": `${opts.ariaLabel || opts.yLabel || "chart"}. Chart: use the arrow keys to read values.`,
      "aria-orientation": "horizontal",
      "aria-valuemin": "0",
      "aria-valuemax": "0",
      "aria-valuenow": "0",
      "aria-valuetext": "Use the arrow keys to read values along the x axis.",
    });
    container.appendChild(svg);
    const legend = document.createElement("div");
    legend.className = "legend chart-legend";
    if (opts.legend !== false) container.appendChild(legend);
    let series = opts.series;
    let view = null; // what the current drawing needs to place the crosshair
    let at = null; // keyboard position: an index into view.grid

    function draw() {
      svg.textContent = "";
      const xs = series.flatMap((s) => s.points.map((p) => p[0]));
      const ys = series.flatMap((s) => s.points.map((p) => p[1]));
      const tx = (v) => (opts.logX ? Math.log10(v) : v);
      const x0 = tx(Math.min(...xs));
      const x1 = tx(Math.max(...xs));
      const y0 = opts.yMin ?? Math.min(0, ...ys);
      const y1 = opts.yMax ?? Math.max(...ys) * 1.05;
      const X = (v) => m.l + ((tx(v) - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
      const Y = (v) => H - m.b - ((v - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
      const grid = svgEl("g", { class: "grid" });
      const axis = svgEl("g", { class: "axis" });
      const yt = niceTicks(y0, y1, 5);
      yt.forEach((v) => {
        grid.appendChild(svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }));
        const t = svgEl("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end", class: "lbl" });
        t.textContent = fy(v);
        axis.appendChild(t);
      });
      const xt = opts.logX
        ? Array.from({ length: Math.floor(x1) - Math.ceil(x0) + 1 }, (_, i) => Math.pow(10, Math.ceil(x0) + i))
        : niceTicks(Math.min(...xs), Math.max(...xs), 6);
      xt.forEach((v) => {
        const t = svgEl("text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle", class: "lbl" });
        t.textContent = fx(v);
        axis.appendChild(t);
      });
      axis.appendChild(svgEl("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
      axis.appendChild(svgEl("line", { x1: m.l, x2: m.l, y1: m.t, y2: H - m.b }));
      const xl = svgEl("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", class: "lbl" });
      xl.textContent = opts.xLabel || "";
      const yl = svgEl("text", { x: 14, y: (m.t + H - m.b) / 2, "text-anchor": "middle", class: "lbl", transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})` });
      yl.textContent = opts.yLabel || "";
      axis.append(xl, yl);
      svg.append(grid, axis);
      const look = series.map((s, i) => ({
        stroke: s.color || cssVar(SERIES_COLOURS[i % SERIES_COLOURS.length]),
        dash: series.length > 1 ? DASHES[i % DASHES.length] : "",
      }));
      series.forEach((s, i) => {
        const d = s.points.map((p, j) => `${j ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
        // As a style, not an attribute, so a colour given as var(--name) works too.
        const path = svgEl("path", { d, class: "series", style: `stroke:${look[i].stroke}` });
        if (look[i].dash) path.setAttribute("stroke-dasharray", look[i].dash);
        svg.appendChild(path);
      });
      legend.innerHTML =
        series.length > 1
          ? series
              .map(
                (s, i) =>
                  `<span><svg class="swatch" viewBox="0 0 28 6" width="28" height="6" aria-hidden="true"><line x1="0" y1="3" x2="28" y2="3" style="stroke:${esc(look[i].stroke)}" stroke-width="2.4"${look[i].dash ? ` stroke-dasharray="${look[i].dash}"` : ""}/></svg>${esc(s.name)}</span>`
              )
              .join("")
          : "";
      legend.hidden = series.length < 2;
      const cross = svgEl("line", { class: "cross", y1: m.t, y2: H - m.b, visibility: "hidden" });
      const dots = series.map(() => svgEl("circle", { r: 4, class: "dot", visibility: "hidden" }));
      svg.append(cross, ...dots);
      const hit = svgEl("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" });
      svg.appendChild(hit);
      // Every x that any series has a point at: the stops the arrow keys move between.
      const gridXs = [...new Set(xs)].sort((a, b) => a - b);
      const readAt = (xv) => {
        const parts = [];
        series.forEach((s, i) => {
          let lo = 0;
          let hi = s.points.length - 1;
          while (hi - lo > 1) {
            const mid = (lo + hi) >> 1;
            if (s.points[mid][0] < xv) lo = mid;
            else hi = mid;
          }
          const p = Math.abs(s.points[lo][0] - xv) < Math.abs(s.points[hi][0] - xv) ? s.points[lo] : s.points[hi];
          dots[i].setAttribute("cx", X(p[0]));
          dots[i].setAttribute("cy", Y(p[1]));
          dots[i].setAttribute("visibility", "visible");
          cross.setAttribute("x1", X(p[0]));
          cross.setAttribute("x2", X(p[0]));
          parts.push({ name: s.name, x: p[0], y: p[1] });
        });
        cross.setAttribute("visibility", "visible");
        if (opts.readout)
          opts.readout.innerHTML =
            `<strong>${esc(opts.xLabel || "x")}</strong> ${fx(parts[0].x)} · ` +
            parts.map((p) => `${esc(p.name)}: <strong>${fy(p.y)}</strong>`).join(" · ");
        return parts;
      };
      const clear = () => {
        cross.setAttribute("visibility", "hidden");
        dots.forEach((d) => d.setAttribute("visibility", "hidden"));
      };
      view = { grid: gridXs, readAt, clear };
      svg.setAttribute("aria-valuemax", String(Math.max(0, gridXs.length - 1)));
      if (at !== null) at = Math.min(at, gridXs.length - 1);
      const move = (ev) => {
        const pt = svg.createSVGPoint();
        pt.x = ev.clientX;
        pt.y = ev.clientY;
        const loc = pt.matrixTransform(svg.getScreenCTM().inverse());
        const frac = (loc.x - m.l) / (W - m.l - m.r);
        readAt(opts.logX ? Math.pow(10, x0 + frac * (x1 - x0)) : x0 + frac * (x1 - x0));
      };
      hit.addEventListener("pointermove", move);
      hit.addEventListener("pointerdown", move);
      hit.addEventListener("pointerleave", clear);
    }

    function keyTo(i) {
      if (!view || !view.grid.length) return;
      at = Math.max(0, Math.min(view.grid.length - 1, i));
      const parts = view.readAt(view.grid[at]);
      const where = `${opts.xLabel || "x"} ${fx(parts[0].x)}`;
      svg.setAttribute("aria-valuenow", String(at));
      // With a readout, its (announced) text carries the values; without one, the slider does.
      svg.setAttribute(
        "aria-valuetext",
        opts.readout ? where : `${where}: ` + parts.map((p) => `${p.name} ${fy(p.y)}`).join(", ")
      );
    }
    svg.addEventListener("keydown", (e) => {
      if (!view) return;
      const n = view.grid.length;
      const big = Math.max(1, Math.round(n / 10));
      const cur = at === null ? -1 : at;
      const to = {
        ArrowRight: cur + 1, ArrowUp: cur + 1, ArrowLeft: cur - 1, ArrowDown: cur - 1,
        PageUp: cur + big, PageDown: cur - big, Home: 0, End: n - 1,
      }[e.key];
      if (to === undefined) return;
      e.preventDefault();
      keyTo(at === null && to < 0 ? 0 : to);
    });
    svg.addEventListener("focus", () => svg.matches(":focus-visible") && keyTo(at === null ? 0 : at));
    svg.addEventListener("blur", () => view && view.clear());

    draw();
    document.addEventListener("papers:theme", draw);
    return {
      update(next) {
        series = next;
        draw();
      },
    };
  };

  /**
   * Let the keyboard move a point that the pointer places by clicking (a start point, a query).
   *   Papers.keyPoint(el, { get: () => [x, y], set(x, y), what: "the query", step, announce })
   * x and y are fractions of the picture, 0 to 1 from the left and from the top. The arrow
   * keys move the point by `step` (default 0.02), five times as far with Shift held.
   */
  P.keyPoint = function (el, o) {
    if (!el.dataset.keypointLabel) el.dataset.keypointLabel = el.getAttribute("aria-label") || "";
    el.dataset.keypoint = "1";
    el.tabIndex = 0;
    // "application" hands the arrow keys to the page, which a two-dimensional control needs.
    el.setAttribute("role", "application");
    el.setAttribute(
      "aria-label",
      `${el.dataset.keypointLabel} Arrow keys move ${o.what || "the point"}; hold Shift to move it further.`.trim()
    );
    el.addEventListener("keydown", (e) => {
      const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
      if (!d) return;
      e.preventDefault();
      const step = (o.step || 0.02) * (e.shiftKey ? 5 : 1);
      const clamp = (v) => Math.max(0, Math.min(1, v));
      const [x, y] = o.get();
      const nx = clamp(x + d[0] * step);
      const ny = clamp(y + d[1] * step);
      o.set(nx, ny);
      if (o.announce !== false)
        P.announce(`${o.what || "point"} at ${Math.round(nx * 100)}% across, ${Math.round(ny * 100)}% down`, 0);
    });
  };

  /**
   * Heatmap on a <canvas> with a hover readout; the arrow keys move a cell cursor too.
   *   Papers.heatmap(canvas, { rows, cols, value(r, c), min, max, cell, onHover(r, c, v) })
   * Values are coloured on a diverging scale: blue below the midpoint, red above.
   */
  P.heatmap = function (canvas, opts) {
    const cell = opts.cell || Math.max(1, Math.floor(720 / opts.cols));
    canvas.width = opts.cols * cell;
    canvas.height = opts.rows * cell;
    const ctx = canvas.getContext("2d");
    const lo = opts.min ?? -1;
    const hi = opts.max ?? 1;
    const mid = (lo + hi) / 2;
    const color = (v) => {
      const t = Math.max(-1, Math.min(1, (v - mid) / ((hi - lo) / 2 || 1)));
      const base = [247, 247, 247];
      const end = t < 0 ? [37, 99, 235] : [220, 38, 38];
      const a = Math.abs(t);
      return `rgb(${base.map((b, i) => Math.round(b + (end[i] - b) * a)).join(",")})`;
    };
    const paint = (r, c) => {
      ctx.fillStyle = color(opts.value(r, c));
      ctx.fillRect(c * cell, r * cell, cell, cell);
    };
    for (let r = 0; r < opts.rows; r++) for (let c = 0; c < opts.cols; c++) paint(r, c);
    const where = (ev) => {
      const rect = canvas.getBoundingClientRect();
      const c = Math.floor(((ev.clientX - rect.left) / rect.width) * opts.cols);
      const r = Math.floor(((ev.clientY - rect.top) / rect.height) * opts.rows);
      if (r < 0 || c < 0 || r >= opts.rows || c >= opts.cols) return;
      opts.onHover && opts.onHover(r, c, opts.value(r, c));
    };
    canvas.addEventListener("pointermove", where);
    canvas.addEventListener("pointerdown", where);
    canvas.addEventListener("pointerleave", () => opts.onHover && opts.onHover(null));

    // Keyboard: a cell cursor, outlined, that reports through the same onHover as the pointer.
    // The canvas's own label (kept once, since a page may clone the canvas) is its text alternative.
    if (!canvas.dataset.heatLabel) canvas.dataset.heatLabel = canvas.getAttribute("aria-label") || "Heatmap";
    canvas.tabIndex = 0;
    canvas.setAttribute("role", "application");
    canvas.setAttribute(
      "aria-label",
      `${canvas.dataset.heatLabel}, ${opts.rows} rows by ${opts.cols} columns. Arrow keys move between cells and read each value.`
    );
    let cur = null;
    let last = [0, 0];
    const mark = (next) => {
      if (cur) {
        // Repaint the neighbourhood the old outline touched.
        for (let r = cur[0] - 1; r <= cur[0] + 1; r++)
          for (let c = cur[1] - 1; c <= cur[1] + 1; c++)
            if (r >= 0 && c >= 0 && r < opts.rows && c < opts.cols) paint(r, c);
      }
      cur = next;
      if (!cur) return;
      last = cur;
      const lw = Math.max(1.5, cell / 5);
      ctx.lineWidth = lw;
      ctx.strokeStyle = cssVar("--fg") || "#000";
      ctx.strokeRect(cur[1] * cell + lw / 2, cur[0] * cell + lw / 2, cell - lw, cell - lw);
      opts.onHover && opts.onHover(cur[0], cur[1], opts.value(cur[0], cur[1]));
    };
    canvas.addEventListener("keydown", (e) => {
      const d = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] }[e.key];
      if (!d) return;
      e.preventDefault();
      if (!cur) return mark(last);
      const k = e.shiftKey ? 10 : 1;
      const r = Math.max(0, Math.min(opts.rows - 1, cur[0] + d[0] * k));
      const c = Math.max(0, Math.min(opts.cols - 1, cur[1] + d[1] * k));
      mark([r, c]);
    });
    canvas.addEventListener("focus", () => canvas.matches(":focus-visible") && mark(last));
    canvas.addEventListener("blur", () => {
      mark(null);
      opts.onHover && opts.onHover(null);
    });
  };

  // ------------------------------------------------ site nav and paper context
  // Everything here is derived from window.PRIMER_PAPERS (assets/catalog.js, generated
  // from CATALOG.md by the docs build), so links never go stale.
  // Accepts either {title, authors_year} or a raw CATALOG.md cell as `title`
  // ("Vaswani et al., *Attention Is All You Need* (2017)") and normalizes it.
  const papers = () =>
    (window.PRIMER_PAPERS || []).map((p) => {
      if (p.authors_year || !/\*/.test(p.title || "")) return p;
      const titles = [...p.title.matchAll(/\*([^*]+)\*/g)].map((m) => m[1]);
      const authors = p.title.replace(/,?\s*\*[^*]+\*/g, "").replace(/\s+/g, " ").trim();
      return { ...p, title: titles.join(" + "), authors_year: authors };
    });
  const lessonPage = (dotted) => {
    const mod = dotted.startsWith("primer.") ? dotted : `primer.${dotted}`;
    return P.lessonHref(mod.replace(/\./g, "/") + ".html");
  };
  // A lesson's title ("Attention"), as the home page names it; its module path only if the title is unknown.
  const lessonName = (dotted) => (window.PRIMER_LESSONS || {})[dotted] || dotted.replace(/^primer\./, "");
  const paperSlug = () =>
    document.body.dataset.paper || decodeURIComponent(location.pathname).split("/").pop().replace(/\.html$/, "");
  const paperHref = (p) => (p.exists ? `${p.slug}.html` : `index.html#${p.slug}`);

  function initNav() {
    const root = siteRoot();
    const nav = document.createElement("nav");
    nav.className = "topbar";
    nav.setAttribute("aria-label", "Site");
    nav.innerHTML =
      `<a class="home" href="${root}index.html">primer</a>` +
      `<a href="${root}index.html#lessons">Lessons</a>` +
      `<a href="index.html">Papers</a>` +
      `<a href="${root}primer/glossary.html">Glossary</a>` +
      `<a href="${root}primer/notation.html">Notation</a>` +
      (window.PRIMER_REPO ? `<a href="${window.PRIMER_REPO}">Code on GitHub</a>` : "");
    document.body.prepend(nav);

    const list = papers();
    const idx = list.findIndex((p) => p.slug === paperSlug());
    if (idx < 0) return;
    const me = list[idx];
    const box = document.createElement("aside");
    box.className = "paper-context";
    box.innerHTML =
      `<strong>Lessons that build this in code:</strong> ` +
      ((me.lessons || []).map((m) => `<a href="${lessonPage(m)}">${esc(lessonName(m))}</a>`).join(" · ") || "none yet") +
      (me.sources && me.sources.length
        ? `<br><strong>Primary source${me.sources.length > 1 ? "s" : ""}:</strong> ` +
          me.sources.map((u) => `<a href="${esc(u)}">${esc(u.replace(/^https?:\/\//, ""))}</a>`).join(" · ")
        : "");
    const hero = document.querySelector("header.hero");
    if (hero) hero.after(box);
    else document.querySelector("main")?.prepend(box);

    const prev = list[idx - 1];
    const next = list[idx + 1];
    const pager = document.createElement("nav");
    pager.className = "pager";
    pager.setAttribute("aria-label", "Previous and next paper");
    pager.innerHTML =
      (prev ? `<a class="prev" href="${paperHref(prev)}">← ${esc(prev.title)}${prev.exists ? "" : " (coming soon)"}</a>` : "<span></span>") +
      (next ? `<a class="next" href="${paperHref(next)}">${esc(next.title)}${next.exists ? "" : " (coming soon)"} →</a>` : "<span></span>");
    const footer = document.querySelector("main footer");
    if (footer) footer.before(pager);
    else document.querySelector("main")?.append(pager);
  }

  /** Render the catalogue of companions into `el` (used by docs/papers/index.html). */
  P.renderIndex = function (el) {
    const list = papers();
    const ready = list.filter((p) => p.exists).length;
    el.innerHTML =
      `<p class="readout">${ready} of ${list.length} companions written so far. Titles without a link are on the way.</p>` +
      `<div class="cards">` +
      list
        .map(
          (p) => `<article class="card${p.exists ? "" : " soon"}" id="${esc(p.slug)}">
        <span class="status ${p.exists ? "ready" : "soon"}">${p.exists ? "Annotated companion" : "Coming soon"}</span>
        <h3>${p.exists ? `<a href="${p.slug}.html">${esc(p.title)}</a>` : esc(p.title)}</h3>
        <div>${esc(p.authors_year || "")}</div>
        <div style="margin-top:.4rem">${(p.sources || []).map((u) => `<a href="${esc(u)}">original</a>`).join(" · ")}</div>
        <div style="margin-top:.4rem">Lessons: ${(p.lessons || []).map((m) => `<a href="${lessonPage(m)}">${esc(lessonName(m))}</a>`).join(" · ")}</div>
      </article>`
        )
        .join("") +
      `</div>`;
  };

  // --------------------------------------------------------------------- boot
  document.addEventListener("DOMContentLoaded", () => {
    const data = window.PAPER || {};
    Object.assign(page.symbols, data.symbols || {});
    Object.assign(page.blocks, data.blocks || {});
    for (const [k, v] of Object.entries(data.terms || {})) page.terms[k.toLowerCase()] = v;
    initTheme();
    initNav();
    const index = document.getElementById("paper-index");
    if (index) P.renderIndex(index);
    initTooltips();
    P.renderSymbolTables();
    P.renderMath();
    P.wireDiagrams();
    wireLessonLinks();
    makeFocusable(document);
    initToc();
    buildGlossary();
    if (typeof data.onReady === "function") data.onReady(P);
    const watchReadouts = initLiveReadouts();
    exposeControlsInPictures();
    // Pages redraw their interactive pictures (a new layer, a rebuilt tree): keep them honest.
    let pending = false;
    new MutationObserver(() => {
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        exposeControlsInPictures();
        watchReadouts();
      });
    }).observe(document.querySelector("main") || document.body, { childList: true, subtree: true });
    const missing = P.missingTerms();
    if (missing.length) console.info("papers.js: glossary keys with no definition:", missing);
  });
})();
