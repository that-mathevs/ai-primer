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
    if (owner && owner !== el) owner.classList.remove("on");
    owner = el;
    el.classList.add("on");
    el.setAttribute("aria-describedby", "papers-tip");
    tip.innerHTML = html;
    tip.classList.add("show");
    position(el);
  }

  function hide() {
    clearTimeout(hideTimer);
    tip.classList.remove("show");
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

  function initTooltips() {
    tip = document.createElement("div");
    tip.className = "tip";
    tip.id = "papers-tip";
    tip.setAttribute("role", "tooltip");
    document.body.appendChild(tip);
    tip.addEventListener("pointerenter", () => clearTimeout(hideTimer));
    tip.addEventListener("pointerleave", hideSoon);

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
      const el = e.target.closest && e.target.closest(TIP_SEL);
      if (el) show(el);
    });
    document.addEventListener("focusout", (e) => {
      if (e.target.closest && e.target.closest(TIP_SEL)) hideSoon();
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
      if (e.key === "Escape") hide();
    });
    window.addEventListener("scroll", () => owner && position(owner), { passive: true });
    window.addEventListener("resize", () => owner && position(owner));
  }

  function makeFocusable(root) {
    root.querySelectorAll(TIP_SEL).forEach((el) => {
      if (!el.hasAttribute("tabindex")) el.tabIndex = 0;
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
  P.wireDiagrams = function (root = document) {
    root.querySelectorAll("figure.ix").forEach((fig) => {
      const panel = fig.querySelector(".panel");
      const parts = [...fig.querySelectorAll(".blk[data-block]")];
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
      parts.forEach((b) => {
        if (!b.hasAttribute("tabindex")) b.setAttribute("tabindex", "0");
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

  /**
   * Line chart with a hover crosshair.
   *   Papers.lineChart(container, {
   *     series: [{ name, color, points: [[x, y], ...] }],   // points sorted by x
   *     xLabel, yLabel, xFmt, yFmt, logX, yMin, yMax, width, height,
   *     readout: element that receives the hover text (optional)
   *   })  ->  { update(series) }
   */
  P.lineChart = function (container, opts) {
    const W = opts.width || 640;
    const H = opts.height || 280;
    const m = { l: 62, r: 18, t: 12, b: 42 };
    const fx = opts.xFmt || ((v) => String(+v.toPrecision(4)));
    const fy = opts.yFmt || ((v) => String(+v.toPrecision(4)));
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.ariaLabel || opts.yLabel || "chart" });
    container.appendChild(svg);
    let series = opts.series;

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
      const palette = ["--accent", "--active", "--soft-stroke", "--attn-stroke", "--lin-stroke"];
      series.forEach((s, i) => {
        const d = s.points.map((p, j) => `${j ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
        svg.appendChild(svgEl("path", { d, class: "series", stroke: s.color || cssVar(palette[i % palette.length]) }));
      });
      const cross = svgEl("line", { class: "cross", y1: m.t, y2: H - m.b, visibility: "hidden" });
      const dots = series.map(() => svgEl("circle", { r: 4, class: "dot", visibility: "hidden" }));
      svg.append(cross, ...dots);
      const hit = svgEl("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" });
      svg.appendChild(hit);
      const move = (ev) => {
        const pt = svg.createSVGPoint();
        pt.x = ev.clientX;
        pt.y = ev.clientY;
        const loc = pt.matrixTransform(svg.getScreenCTM().inverse());
        const frac = (loc.x - m.l) / (W - m.l - m.r);
        const xv = opts.logX ? Math.pow(10, x0 + frac * (x1 - x0)) : x0 + frac * (x1 - x0);
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
      };
      hit.addEventListener("pointermove", move);
      hit.addEventListener("pointerdown", move);
      hit.addEventListener("pointerleave", () => {
        cross.setAttribute("visibility", "hidden");
        dots.forEach((d) => d.setAttribute("visibility", "hidden"));
      });
    }
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
   * Heatmap on a <canvas> with a hover readout.
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
    for (let r = 0; r < opts.rows; r++)
      for (let c = 0; c < opts.cols; c++) {
        ctx.fillStyle = color(opts.value(r, c));
        ctx.fillRect(c * cell, r * cell, cell, cell);
      }
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
  const lessonName = (dotted) => dotted.replace(/^primer\./, "");
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
    const missing = P.missingTerms();
    if (missing.length) console.info("papers.js: glossary keys with no definition:", missing);
  });
})();
