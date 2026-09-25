"""
Check that no page is wider than a phone screen.

    python tools/phonecheck.py      # or: make phonecheck (after make docs)

Serves the built site (docs/html), loads every page in a 390-pixel-wide frame
in headless Chrome, and lists anything that sticks out past the screen: a
reader on a phone would have to scroll sideways to see it. Content inside a
box that scrolls on purpose (a code block, a wide table) is fine.

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
PROBE = "_phonecheck.html"

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
    results.push({ page: pages[i], width: doc.documentElement.scrollWidth, screen, wide });
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


def pages_to_check() -> list[str]:
    """Every page a reader can land on, except the forwarding stubs."""
    return sorted(
        p.relative_to(SITE).as_posix()
        for p in SITE.rglob("*.html")
        if p.name != PROBE and 'http-equiv="refresh"' not in p.read_text(errors="ignore")
    )


def too_wide(results: list[dict]) -> list[dict]:
    return [r for r in results if r["width"] > r["screen"]]


def main() -> int:
    browser = chrome()
    if browser is None:
        print("phonecheck: no Chrome or Chromium found; skipped")
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
        print("phonecheck: the probe didn't finish; no result")
        return 1
    results = json.loads(html.unescape(found.group(1)))
    bad = too_wide(results)
    print(f"checked {len(results)} pages at {WIDTH}px: {len(bad)} wider than the screen")
    for r in bad:
        print(f"  ✗ {r['page']}: {r['width']}px wide")
        for item in r["wide"]:
            print(f"      {item}")
    return 1 if bad or len(results) != len(pages) else 0


if __name__ == "__main__":
    sys.exit(main())
