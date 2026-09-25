"""
Crawl the built site (docs/html) and fail if any internal link or image is broken.

    python tools/sitecheck.py      # or: make sitecheck (after make docs)

Links built by JavaScript at runtime are checked by the pages themselves;
this covers every href and src written into the HTML.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent / "docs" / "html"
EXTERNAL = re.compile(r"^(https?:|mailto:|data:|javascript:|//)")


def broken_links() -> tuple[int, dict[str, set[str]]]:
    bad: dict[str, set[str]] = {}
    checked = 0
    for page in SITE.rglob("*.html"):
        # Script bodies hold template strings like `${url}`, not links.
        html = re.sub(r"<script\b.*?</script>", "", page.read_text(errors="ignore"), flags=re.S | re.I)
        for ref in re.findall(r'(?:href|src)="([^"#?]*)(?:[#?][^"]*)?"', html):
            if not ref.strip() or EXTERNAL.match(ref):
                continue
            checked += 1
            if not (page.parent / ref).resolve().exists():
                bad.setdefault(page.relative_to(SITE).as_posix(), set()).add(ref)
    return checked, bad


def main() -> int:
    if not SITE.exists():
        print("docs/html doesn't exist yet: run `make docs` first")
        return 1
    checked, bad = broken_links()
    print(f"checked {checked} internal links across {len(list(SITE.rglob('*.html')))} pages: {sum(map(len, bad.values()))} broken")
    for page, refs in sorted(bad.items()):
        print(f"  ✗ {page} -> {', '.join(sorted(refs))}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
