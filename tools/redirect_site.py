"""The published site is a set of redirects to rumblr.com.

The lessons and the annotated papers are read at rumblr.com, which is built from this
repository. GitHub Pages used to serve a second copy of every lesson; it now serves one
redirect per page it used to have, so every old link still lands on the lesson or paper
it named. `make redirects` writes them into docs/html, which the pages workflow publishes.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # run as a script from anywhere
SITE = ROOT / "docs" / "html"
RUMBLR = "https://rumblr.com/"


def redirect_page(target: str) -> str:
    """A page that sends the reader straight on to `target`, with a plain link as a fallback."""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta http-equiv="refresh" content="0; url={target}"><link rel="canonical" href="{target}">'
        f'<title>AI Primer</title></head><body><p>The AI Primer is read at <a href="{target}">{target}</a>.</p></body></html>\n'
    )


def redirects() -> dict[str, str]:
    """Every page the old site had, mapped to where it is read now."""
    from primer.curriculum import CURRICULUM, lesson_url

    pages = {"index.html": RUMBLR, "404.html": RUMBLR, "spec.html": RUMBLR, "papers/index.html": f"{RUMBLR}papers/"}
    for lesson in CURRICULUM:
        pages[lesson.module.replace(".", "/") + ".html"] = lesson_url(lesson.module)
    for companion in sorted((ROOT / "docs" / "papers").glob("*.html")):
        if companion.stem != "index":
            pages[f"papers/{companion.name}"] = f"{RUMBLR}papers/primer/{companion.stem}"
    return pages


def main() -> int:
    if SITE.exists():
        shutil.rmtree(SITE)
    for page, target in redirects().items():
        out = SITE / page
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(redirect_page(target), encoding="utf-8")
    print(f"wrote {len(redirects())} redirects to rumblr.com into {SITE.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
