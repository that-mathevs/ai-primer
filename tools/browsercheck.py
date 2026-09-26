"""
Check every built page in a real browser: on a phone, and with a keyboard or screen reader.

    python tools/browsercheck.py    # or: make browsercheck (after make docs)

Serves the built site (docs/html) and loads every page in a 390-pixel-wide
frame in headless Chrome, after its scripts have run, then checks:

1. Nothing sticks out past the screen, so a phone reader never scrolls
   sideways. (Content inside a box that scrolls on purpose, a code block or
   a wide table, is fine.)
2. Nothing focusable sits inside content hidden from screen readers
   (aria-hidden): a keyboard user would land on something that says nothing.
3. Every image has alt text.
4. Every diagram is named, and described by its "Reading it" paragraph.
5. A focused glossary term is described by its definition, and the
   definition sits right after it, so Tab reaches its "Learn it" link.
6. Every interactive visualization drew itself, is a named group, and every
   control in it has a label, so it works from the keyboard and is announced.

Needs Chrome or Chromium. Without one it says so and exits cleanly, since the
rest of the checks run without a browser.
"""

from __future__ import annotations

import functools
import html
import http.server
import json
import re
import shutil
import socketserver
import subprocess
import sys
import threading
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent / "docs" / "html"
WIDTH = 390  # a typical phone, in CSS pixels
PROBE = "_browsercheck.html"

PROBE_PAGE = """<!doctype html><html><body><pre id="out">running</pre><script>
const pages = %s, width = %d, results = [];
let i = 0;
function next() {
  if (i >= pages.length) { document.getElementById("out").textContent = JSON.stringify(results); return; }
  const frame = document.createElement("iframe");
  frame.style.width = width + "px"; frame.style.height = "800px"; frame.src = pages[i];
  frame.onload = () => setTimeout(() => {
    const doc = frame.contentDocument, screen = doc.documentElement.clientWidth;
    const scrollsOnPurpose = (el) => {
      for (let p = el.parentElement; p && p !== doc.body; p = p.parentElement) {
        if (getComputedStyle(p).overflowX !== "visible") return true;
      }
      return false;
    };
    const wide = [...doc.querySelectorAll("body *")]
      .filter((el) => { const r = el.getBoundingClientRect(); return r.width > 0 && r.right > screen + 1; })
      .filter((el) => !scrollsOnPurpose(el))
      .slice(0, 5)
      .map((el) => el.tagName.toLowerCase() + ": " + (el.textContent || "").trim().slice(0, 60));
    const a11y = [];
    const focusable = 'a[href], button, input, select, textarea, [tabindex]:not([tabindex="-1"])';
    doc.querySelectorAll('[aria-hidden="true"]').forEach((hidden) => {
      hidden.querySelectorAll(focusable).forEach((el) => a11y.push("focusable inside aria-hidden: " + el.tagName.toLowerCase()));
    });
    doc.querySelectorAll("img").forEach((img) => {
      if (!(img.getAttribute("alt") || "").trim()) a11y.push("image without alt text: " + img.getAttribute("src"));
    });
    doc.querySelectorAll("div.mermaid svg").forEach((svg, n) => {
      const described = svg.getAttribute("aria-describedby");
      if (svg.getAttribute("role") !== "img" || !svg.getAttribute("aria-label")) a11y.push("diagram " + (n + 1) + " has no name");
      else if (!described || !doc.getElementById(described)) a11y.push("diagram " + (n + 1) + " has no description");
    });
    doc.querySelectorAll(".viz[data-viz]").forEach((viz) => {
      const name = viz.dataset.viz;
      if (!viz.dataset.ready || !viz.children.length) a11y.push("visualization " + name + " did not draw");
      if (viz.getAttribute("role") !== "group" || !viz.getAttribute("aria-label")) a11y.push("visualization " + name + " is not a named group");
      viz.querySelectorAll("input, select, button").forEach((control) => {
        if (!control.labels?.length && !control.getAttribute("aria-label")) a11y.push("visualization " + name + " has an unlabelled " + control.tagName.toLowerCase());
      });
      if (!viz.querySelector('[role="status"]')) a11y.push("visualization " + name + " never announces its result");
    });
    const term = doc.querySelector(".gl-term[data-lesson]");
    if (term && doc.getElementById("gl-tip")) {
      term.focus();
      const tip = doc.getElementById("gl-tip");
      if (term.getAttribute("aria-describedby") !== "gl-tip") a11y.push("a focused glossary term isn't described by its definition");
      if (tip.previousElementSibling !== term) a11y.push("a glossary definition doesn't follow its term, so Tab can't reach its link");
      term.blur();
    }
    results.push({ page: pages[i], width: doc.documentElement.scrollWidth, screen, wide, a11y: [...new Set(a11y)] });
    frame.remove(); i++; next();
  }, 700);
  document.body.appendChild(frame);
}
next();
</script></body></html>"""


def chrome() -> str | None:
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        if path := shutil.which(name):
            return path
    mac = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    return str(mac) if mac.exists() else None


def pages_to_check(site: Path = SITE) -> list[str]:
    """Every page a reader can land on, except the forwarding stubs."""
    return sorted(
        p.relative_to(site).as_posix()
        for p in site.rglob("*.html")
        if p.name != PROBE and 'http-equiv="refresh"' not in p.read_text(errors="ignore")
    )


def too_wide(results: list[dict]) -> list[dict]:
    return [r for r in results if r["width"] > r["screen"]]


def accessibility_problems(results: list[dict]) -> list[dict]:
    return [r for r in results if r.get("a11y")]


def main() -> int:
    browser = chrome()
    if browser is None:
        print("browsercheck: no Chrome or Chromium found; skipped")
        return 0
    pages = pages_to_check()
    (SITE / PROBE).write_text(PROBE_PAGE % (json.dumps(pages), WIDTH), encoding="utf-8")
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):  # one line per request would bury the result
            pass

    handler = functools.partial(Quiet, directory=str(SITE))
    try:
        with socketserver.TCPServer(("127.0.0.1", 0), handler) as server:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            url = f"http://127.0.0.1:{server.server_address[1]}/{PROBE}"
            dom = subprocess.run(
                [browser, "--headless=new", "--disable-gpu", "--no-sandbox",
                 f"--virtual-time-budget={len(pages) * 2500}", "--dump-dom", url],
                capture_output=True, text=True, timeout=600,
            ).stdout
            server.shutdown()
    finally:
        (SITE / PROBE).unlink(missing_ok=True)
    found = re.search(r'<pre id="out">(.*?)</pre>', dom, re.S)
    if not found or found.group(1) == "running":
        print("browsercheck: the probe didn't finish; no result")
        return 1
    results = json.loads(html.unescape(found.group(1)))
    bad = too_wide(results)
    print(f"checked {len(results)} pages at {WIDTH}px: {len(bad)} wider than the screen")
    for r in bad:
        print(f"  ✗ {r['page']}: {r['width']}px wide")
        for item in r["wide"]:
            print(f"      {item}")
    barriers = accessibility_problems(results)
    print(f"accessibility barriers: {sum(len(r['a11y']) for r in barriers)}")
    for r in barriers:
        print(f"  ✗ {r['page']}:")
        for item in r["a11y"][:6]:
            print(f"      {item}")
    return 1 if bad or barriers or len(results) != len(pages) else 0


if __name__ == "__main__":
    sys.exit(main())
