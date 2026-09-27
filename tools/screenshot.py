"""
See a built page the way a phone shows it, in either theme.

    python tools/screenshot.py papers/hyde.html                  # light, top of the page
    python tools/screenshot.py papers/hyde.html#fig1 --theme dark --out /tmp/fig1.png

Serves the built site (docs/html, after make docs) and photographs the page
inside a 390-pixel-wide frame. Headless Chrome won't make its window narrower
than about 500 pixels, so a plain screenshot lays the page out too wide and
crops it; the frame gives the true phone layout. The theme is set before the
page loads, so the picture doesn't depend on this machine's light or dark mode.
An anchor opens the frame scrolled to that part of the page.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import socketserver
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.browsercheck import SITE, WIDTH, chrome  # noqa: E402

FRAME = "_screenshot.html"
HEIGHT = 1400  # about two phone screens: tall enough for a figure and its Reading it paragraph


def frame_page(page: str, theme: str, width: int = WIDTH, height: int = HEIGHT) -> str:
    return (
        '<!doctype html><html><body style="margin:0">'
        f'<iframe id="f" style="width:{width}px;height:{height}px;border:0"></iframe><script>'
        # theme.js reads this key, and the frame shares this page's storage.
        f'try {{ localStorage.setItem("primer-theme", {json.dumps(theme)}); }} catch (_) {{}}\n'
        f"document.getElementById(\"f\").src = {json.dumps(page)};"
        "</script></body></html>"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("page", help="path under docs/html, optionally with #anchor")
    parser.add_argument("--theme", choices=("light", "dark"), default="light")
    parser.add_argument("--out", default=None, help="PNG to write (default: /tmp/<page>-<theme>.png)")
    args = parser.parse_args()
    browser = chrome()
    if browser is None:
        print("screenshot: no Chrome or Chromium found")
        return 1
    out = args.out or f"/tmp/{Path(args.page.split('#')[0]).stem}-{args.theme}.png"
    (SITE / FRAME).write_text(frame_page(args.page, args.theme), encoding="utf-8")

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    try:
        with socketserver.TCPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(SITE))) as server:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            url = f"http://127.0.0.1:{server.server_address[1]}/{FRAME}"
            subprocess.run(
                [browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={WIDTH + 200},{HEIGHT}",
                 "--virtual-time-budget=4000", f"--screenshot={out}", url],
                capture_output=True, timeout=120,
            )
            server.shutdown()
    finally:
        (SITE / FRAME).unlink(missing_ok=True)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
