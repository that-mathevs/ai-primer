"""
Check that every URL in the lessons, the paper catalog, the README and the
paper companions is alive.

    python tools/links.py      # or: make links

Uses the network, so it is not part of `make test`. Run it before publishing.
Exits non-zero if any link is dead, listing each with the lesson that cites it.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
URL = re.compile(r"https?://[^\s)>\]\"'`*]+")


def collect() -> dict[str, set[str]]:
    import primer

    found: dict[str, set[str]] = {}
    for info in pkgutil.walk_packages(primer.__path__, "primer."):
        doc = importlib.import_module(info.name).__doc__ or ""
        for url in URL.findall(doc):
            found.setdefault(url.rstrip(".,;:"), set()).add(info.name)
    for md in [ROOT / "docs" / "papers" / "CATALOG.md", ROOT / "README.md"]:
        for url in URL.findall(md.read_text()):
            found.setdefault(url.rstrip(".,;:"), set()).add(md.name)
    # The paper companions cite the most sources: every external link they hold.
    for page in sorted((ROOT / "docs" / "papers").glob("*.html")):
        for url in re.findall(r'href="(https?://[^"]+)"', page.read_text(encoding="utf-8")):
            found.setdefault(url.split("#")[0], set()).add(page.name)
    return found


def status(url: str) -> tuple[str, int | str]:
    headers = {"User-Agent": "Mozilla/5.0 (primer link check)"}
    for method in ("HEAD", "GET"):  # some servers reject HEAD
        try:
            req = urllib.request.Request(url, method=method, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as r:
                return url, r.status
        except urllib.error.HTTPError as e:
            if method == "GET" or e.code not in (403, 405, 400):
                return url, e.code
        except Exception as e:  # DNS, TLS, timeout
            if method == "GET":
                return url, type(e).__name__
    return url, "unknown"


def main() -> int:
    found = collect()
    with ThreadPoolExecutor(16) as pool:
        results = list(pool.map(status, sorted(found)))
    # 403/429 usually mean "bot blocked", not "dead"; report them separately.
    dead = [(u, s) for u, s in results if not (isinstance(s, int) and (s < 400 or s in (403, 429)))]
    blocked = [(u, s) for u, s in results if s in (403, 429)]
    print(f"checked {len(results)} links: {len(dead)} dead, {len(blocked)} blocked by the server (check by hand)")
    for u, s in dead:
        print(f"  ✗ {s}  {u}   <- {', '.join(sorted(found[u]))}")
    for u, s in blocked:
        print(f"  ? {s}  {u}   <- {', '.join(sorted(found[u]))}")
    return 1 if dead else 0


if __name__ == "__main__":
    sys.exit(main())
