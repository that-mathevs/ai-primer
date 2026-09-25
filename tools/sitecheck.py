"""
Crawl the built site (docs/html) and fail if any link is broken.

    python tools/sitecheck.py      # or: make sitecheck (after make docs)

Three checks, all offline:

1. Every internal href and src written into the HTML resolves to a file.
2. Every link into this repository on GitHub (and every relative link in
   README.md) names a committed file, and a line range that file has.
3. Every name in an **In code:** line became a link to its code.

Links built by JavaScript at runtime are checked by the pages themselves.
"""

from __future__ import annotations

import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "docs" / "html"
sys.path.insert(0, str(ROOT))
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


@lru_cache(maxsize=1)
def _committed() -> frozenset[str] | None:
    """Files git tracks. None outside a git checkout, where only existence can be checked."""
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    return frozenset(out.stdout.split("\n")) if out.returncode == 0 else None


def own_repo_problem(url: str) -> str | None:
    """What is wrong with a link into this repository on GitHub, or None if nothing.

    GitHub serves the pushed tree, so a file that exists here but isn't committed
    is still a broken link for readers.
    """
    from primer.curriculum import BRANCH, REPO_URL

    m = re.match(rf"{re.escape(REPO_URL)}/(?:blob|tree)/{re.escape(BRANCH)}/([^#?]+)(?:#L(\d+)(?:-L(\d+))?)?$", url)
    if not m:
        return None if url.rstrip("/") == REPO_URL else f"not a file link on {BRANCH}: {url}"
    path, first, last = m.group(1), m.group(2), m.group(3) or m.group(2)
    if not (ROOT / path).exists():
        return f"no such file: {path}"
    committed = _committed()
    if committed is not None and (ROOT / path).is_file() and path not in committed:
        return f"not committed: {path}"
    if first:
        n = len((ROOT / path).read_text(encoding="utf-8").splitlines())
        if int(last) > n:
            return f"{path} has {n} lines, link asks for L{first}-L{last}"
    return None


def unlinked_code_names(page_html: str) -> list[str]:
    """Names in the page's **In code:** lines that pdoc did not turn into links."""
    names = []
    for para in re.findall(r"<p><strong>In code:</strong>(.*?)</p>", page_html, re.S):
        bare = re.sub(r"<code>\s*<a\b[^>]*>.*?</a>\s*</code>|<a\b[^>]*>\s*<code>.*?</code>\s*</a>", "", para, flags=re.S)
        names += re.findall(r"<code>(.*?)</code>", bare, re.S)
    return names


def repo_link_problems() -> dict[str, set[str]]:
    """Every broken link into this repository, from the site's pages and from README.md."""
    from primer.curriculum import REPO_URL

    bad: dict[str, set[str]] = {}
    pages = [(p.relative_to(SITE).as_posix(), p.read_text(errors="ignore")) for p in SITE.rglob("*.html")]
    for name, html in pages:
        for url in set(re.findall(rf'href="({re.escape(REPO_URL)}[^"]*)"', html)):
            if problem := own_repo_problem(url):
                bad.setdefault(name, set()).add(problem)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)\s]+)\)", readme):
        if EXTERNAL.match(target) or target.startswith("#"):
            continue
        if problem := own_repo_problem(f"{REPO_URL}/blob/main/{target.split('#')[0]}"):
            bad.setdefault("README.md", set()).add(problem)
    return bad


def main() -> int:
    if not SITE.exists():
        print("docs/html doesn't exist yet: run `make docs` first")
        return 1
    checked, bad = broken_links()
    print(f"checked {checked} internal links across {len(list(SITE.rglob('*.html')))} pages: {sum(map(len, bad.values()))} broken")
    for page, refs in sorted(bad.items()):
        print(f"  ✗ {page} -> {', '.join(sorted(refs))}")

    repo_bad = repo_link_problems()
    print(f"links into the repository: {sum(map(len, repo_bad.values()))} broken")
    for page, problems in sorted(repo_bad.items()):
        print(f"  ✗ {page}: {'; '.join(sorted(problems))}")

    unlinked = {p.relative_to(SITE).as_posix(): names for p in SITE.rglob("*.html")
                if (names := unlinked_code_names(p.read_text(errors="ignore")))}
    print(f"In code lines: {sum(map(len, unlinked.values()))} names that did not become links")
    for page, names in sorted(unlinked.items()):
        print(f"  ✗ {page}: {', '.join(names)}")
    return 1 if bad or repo_bad or unlinked else 0


if __name__ == "__main__":
    sys.exit(main())
