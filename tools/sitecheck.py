"""
Crawl the built site (docs/html) and fail if any link is broken.

    python tools/sitecheck.py      # or: make sitecheck (after make docs)

Three checks, all offline:

1. Every internal href and src written into the HTML resolves to a file.
2. Every link into this repository on GitHub (and every relative link in
   a committed Markdown file) names a committed file, and a line range that file has.
3. Every name in an **In code:** line became a link to its code.
4. No link detours through a package page that only forwards to the home page.
5. No mention of code is left unlinked (the same scan the site builder links with).
6. Every #anchor a link points at exists on its page.
7. Every lesson a paper companion's script links to exists.
8. No link has an empty address (it would only reload the page).
9. The specification page lists the whole suite, and every count the site states matches it.
10. Nothing renders broken: no indented lines collapsed into one paragraph, no raw Markdown,
    no heading that skips a level, no two prices typeset as one formula.
11. Every page can be reached: some other page links to it.
12. Every page with diagrams draws each one once, under its own name.

Links built by JavaScript at runtime are checked by the pages themselves.
"""

from __future__ import annotations

import html as htmllib
import posixpath
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

    Links name the main branch or the commit the site was built from; either way
    the working tree is what they show (a published build checks out that commit).
    GitHub serves the pushed tree, so a file that exists here but isn't committed
    is still a broken link for readers.
    """
    from primer.curriculum import BRANCH
    from tools.docsite import repo_url

    repo = repo_url()
    m = re.match(rf"{re.escape(repo)}/(?:blob|tree)/(?:{re.escape(BRANCH)}|[0-9a-f]{{40}})/([^#?]+)(?:#L(\d+)(?:-L(\d+))?)?$", url)
    if not m:
        return None if url.rstrip("/") == repo else f"not a link to a file in this repository: {url}"
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
        # Only names can link; a keyword argument or a literal shows how to call something.
        names += [c for c in re.findall(r"<code>(.*?)</code>", bare, re.S) if re.fullmatch(r"[A-Za-z_][\w.]*(\(\))?", c)]
    return names


def repo_link_problems() -> dict[str, set[str]]:
    """Every broken link into this repository, from the site's pages and every committed Markdown file."""
    from primer.curriculum import BRANCH
    from tools.docsite import repo_url

    repo = repo_url()
    bad: dict[str, set[str]] = {}
    pages = [(p.relative_to(SITE).as_posix(), p.read_text(errors="ignore")) for p in SITE.rglob("*.html")]
    for name, html in pages:
        for url in set(re.findall(rf'href="({re.escape(repo)}[^"]*)"', html)):
            if problem := own_repo_problem(url):
                bad.setdefault(name, set()).add(problem)
    # Relative links in every committed Markdown file, as GitHub resolves them: from the file's own folder.
    for md in sorted(f for f in (_committed() or ()) if f.endswith(".md")):
        # Code spans and fenced blocks show link syntax as an example; they aren't links.
        prose = re.sub(r"```.*?```|`[^`\n]*`", "", (ROOT / md).read_text(encoding="utf-8"), flags=re.S)
        for target in re.findall(r"\]\(([^)\s]+)\)", prose):
            if EXTERNAL.match(target) or target.startswith("#"):
                continue
            path = posixpath.normpath(posixpath.join(posixpath.dirname(md), target.split("#")[0]))
            problem = (f"leaves the repository: {target}" if path.startswith("..")
                       else own_repo_problem(f"{repo}/blob/{BRANCH}/{path}"))
            if problem:
                bad.setdefault(md, set()).add(problem)
    return bad


def links_through_forwards(site: Path = SITE) -> dict[str, set[str]]:
    """Links that land on a forwarding package page instead of going straight to the home page.

    Covers ordinary links and the lesson links companions write as data (data-lesson="...", from the site root).
    """
    from tools.docsite import package_forwards

    forwarded = {(site / f).resolve() for f in package_forwards()}
    found: dict[str, set[str]] = {}
    for page in site.rglob("*.html"):
        if page.resolve() in forwarded:
            continue
        html = re.sub(r"<script\b.*?</script>", "", page.read_text(errors="ignore"), flags=re.S | re.I)
        for ref in re.findall(r'href="([^"#?]+)', html):
            if not EXTERNAL.match(ref) and (page.parent / ref).resolve() in forwarded:
                found.setdefault(page.relative_to(site).as_posix(), set()).add(ref)
        for ref in re.findall(r'data-lesson="([^"#?]+)', html):
            if (site / ref).resolve() in forwarded:
                found.setdefault(page.relative_to(site).as_posix(), set()).add(ref)
    return found


def unlinked_code_mentions(page_html: str, module: str, page: str | None = None) -> list[str]:
    """Names on a page that mean code in this repository but aren't links."""
    from tools.docsite import _page, _scan_code_mentions

    return _scan_code_mentions(page_html, module, page or _page(module), fix=False)[1]


def missing_anchors(site: Path = SITE) -> dict[str, set[str]]:
    """Links to an #anchor that the target page doesn't have."""
    ids: dict[Path, set[str]] = {}

    def anchors_of(path: Path) -> set[str]:
        if path not in ids:
            ids[path] = set(re.findall(r'\b(?:id|name)="([^"]+)"', path.read_text(errors="ignore")))
        return ids[path]

    missing: dict[str, set[str]] = {}
    for page in site.rglob("*.html"):
        html = re.sub(r"<script\b.*?</script>", "", page.read_text(errors="ignore"), flags=re.S | re.I)
        for ref in re.findall(r'href="([^"]*#[^"]+)"', html):
            if EXTERNAL.match(ref):
                continue
            path, anchor = ref.split("#", 1)
            target = (page.parent / path).resolve() if path else page.resolve()
            if target.suffix == ".html" and target.exists() and anchor not in anchors_of(target):
                missing.setdefault(page.relative_to(site).as_posix(), set()).add(ref)
    return missing


def broken_companion_lessons(site: Path = SITE) -> dict[str, set[str]]:
    """Lesson links that paper companions write as data (data-lesson, lesson: "...") for their script to build."""
    broken: dict[str, set[str]] = {}
    for page in sorted((site / "papers").glob("*.html")):
        for a, b in re.findall(r'data-lesson="([^"]+)"|\blesson:\s*"([^"]+)"', page.read_text(errors="ignore")):
            path = a or b
            if not (site / path.split("#")[0]).exists():
                broken.setdefault(page.name, set()).add(path)
    return broken


def nonexistent_code(page_html: str, module: str) -> list[str]:
    """Calls written as code, like `hyde()`, naming a function that exists nowhere in the package."""
    import builtins

    from tools.docsite import _package_names, _resolve_code_name

    html = re.sub(r"<(script|style|pre|a)\b.*?</\1>", " ", page_html, flags=re.S | re.I)
    found = []
    for code in re.findall(r"<code>([^<]*)</code>", html):
        call = re.fullmatch(r"\s*([a-z_][\w.]*)\(.*\)\s*", htmllib.unescape(code), re.S)
        if not call:
            continue
        name = call.group(1)
        root = name.split(".")[0]
        if "." in name and root != "primer":
            continue  # another library's function, like np.exp
        if hasattr(builtins, name) or _resolve_code_name(name, module) is not None:
            continue
        last = name.rsplit(".", 1)[-1]
        # A tool's public name is often served by a private function of the same name (get_ticket, _get_ticket).
        if last not in _package_names() and f"_{last}" not in _package_names():
            found.append(code.strip())
    return found


def orphan_pages(site: Path = SITE) -> list[str]:
    """Pages no other page links to (the home page excepted), counting companions' data-lesson links."""
    pages = {p.resolve() for p in site.rglob("*.html")}
    reached = {(site / "index.html").resolve()}
    for page in pages:
        html = page.read_text(errors="ignore")
        for ref in re.findall(r'href="([^"#?]+)', html):
            if not EXTERNAL.match(ref):
                target = (page.parent / ref).resolve()
                if target != page:
                    reached.add(target)
        for ref in re.findall(r'data-lesson="([^"#?]+)', html):
            reached.add((site / ref).resolve())
    # Forwarding pages exist only for old bookmarks and pdoc's search; nothing should link to them.
    forwards = {p for p in pages if 'http-equiv="refresh"' in p.read_text(errors="ignore")}
    return sorted(p.relative_to(site.resolve()).as_posix() for p in pages - reached - forwards)


def diagram_problems(site: Path = SITE) -> list[str]:
    """Pages whose diagrams use mermaid's automatic start, which lets diagrams collide and overlap."""
    found = []
    for page in sorted(site.rglob("*.html")):
        html = page.read_text(errors="ignore")
        if 'class="mermaid"' in html and "startOnLoad: false" not in html:
            found.append(page.relative_to(site).as_posix())
    return found


def empty_links(site: Path = SITE) -> dict[str, list[str]]:
    """Links whose address is empty: clicking one just reloads the page."""
    found: dict[str, list[str]] = {}
    for page in sorted(site.rglob("*.html")):
        html = re.sub(r"<script\b.*?</script>", "", page.read_text(errors="ignore"), flags=re.S | re.I)
        texts = [re.sub(r"<[^>]+>", "", t).strip() for t in re.findall(r'<a\b[^>]*\bhref=""[^>]*>(.*?)</a>', html, re.S)]
        if texts:
            found[page.relative_to(site).as_posix()] = texts
    return found


def spec_count_problems(site: Path = SITE) -> list[str]:
    """Every place the site states a test count that isn't what the suite actually holds."""
    from tools.docsite import collect_tests

    total = len(collect_tests())
    problems = []
    if total == 0:
        problems.append("the suite collected no tests")
    stated = {
        "spec.html": re.findall(r"The specification: ([\d,]+) tests", (site / "spec.html").read_text(errors="ignore")),
        "index.html": re.findall(r"a suite of ([\d,]+) tests", (site / "index.html").read_text(errors="ignore")),
    }
    for page, found in stated.items():
        if [int(n.replace(",", "")) for n in found] != [total]:
            problems.append(f"{page} states {found or 'no count'}, the suite has {total:,}")
    return problems


def _visible_text(page_html: str) -> str:
    """The page's text as a reader sees it, without code, source listings, scripts, styles or math."""
    html = re.sub(r"<(script|style|pre|code|textarea)\b.*?</\1>", " ", page_html, flags=re.S | re.I)
    text = htmllib.unescape(re.sub(r"<[^>]+>", " ", html))
    # Math is typeset in the browser from TeX, where * and _ mean something else.
    return re.sub(r"\$\$.*?\$\$|\$[^$\n]+\$|\\\(.*?\\\)|\\\[.*?\\\]", " ", text, flags=re.S)


def collapsed_blocks(page_html: str) -> list[str]:
    """Paragraphs whose text has indented lines: a list or diagram that rendered as one run-on paragraph."""
    found = []
    for para in re.findall(r"<p\b[^>]*>(.*?)</p>", page_html, re.S):
        text = htmllib.unescape(re.sub(r"<[^>]+>", "", para))
        if re.search(r"\n {4,}\S", text):
            found.append(" ".join(text.split()))
    return found


def heading_skips(page_html: str) -> list[str]:
    """Headings that jump more than one level down (h2 straight to h4)."""
    found, last = [], None
    html = re.sub(r"<(script|style)\b.*?</\1>", "", page_html, flags=re.S | re.I)
    for level, text in re.findall(r"<h([1-6])\b[^>]*>(.*?)</h\1>", html, re.S):
        level = int(level)
        if last is not None and level > last + 1:
            found.append(f"h{last} -> h{level}: {' '.join(re.sub(r'<[^>]+>', ' ', text).split())}")
        last = level
    return found


def raw_markdown(page_html: str) -> list[str]:
    """Markdown that reached the reader unconverted: *emphasis*, **bold**, `code`, [text](link)."""
    text = _visible_text(page_html)
    patterns = [
        r"\*\*[^*\n]+\*\*",
        r"(?<![\w*])\*(?!\s)[^*\n]*[A-Za-z][^*\n]*?(?<!\s)\*(?![\w*])",
        r"`[^`\n]+`",
        r"\[[^\]\n]+\]\([^)\s]+\)",
    ]
    return [m for p in patterns for m in re.findall(p, text)]


def prices_typeset_as_math(page_html: str) -> list[str]:
    """Dollar signs MathJax pairs into a formula that is really prose: "$2 per million ... $4".

    A price is written \\$2. When two unescaped prices share a paragraph, the closing sign is
    really the second price's, which follows a space; a real formula never ends in one.
    """
    html = re.sub(r"<(script|style|pre|code|textarea)\b.*?</\1>", " ", page_html, flags=re.S | re.I)
    found = []
    for block in re.findall(r"<(p|li|td|th|dd|h[1-6])\b[^>]*>(.*?)</\1>", html, re.S):
        text = htmllib.unescape(re.sub(r"<[^>]+>", "", block[1])).replace("\\$", " ")
        text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)
        found += [f"${span}$" for span in re.findall(r"\$([^$]+)\$", text) if span[0].isspace() or span[-1].isspace()]
    return found


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

    detours = links_through_forwards()
    print(f"links through forwarding pages: {sum(map(len, detours.values()))}")
    for page, refs in sorted(detours.items()):
        print(f"  ✗ {page} -> {', '.join(sorted(refs))}")

    leftovers: dict[str, list[str]] = {}
    for page in sorted(SITE.rglob("*.html")):
        name = page.relative_to(SITE).as_posix()
        # Lesson pages resolve names in their own module; the home page and companions in the package.
        module = name.removesuffix(".html").replace("/", ".") if name.startswith("primer/") else "primer"
        html = page.read_text(errors="ignore")
        if 'http-equiv="refresh"' in html:
            continue
        if names := unlinked_code_mentions(html, module, name):
            leftovers[name] = names
    print(f"code mentions left unlinked: {sum(map(len, leftovers.values()))}")
    for page, names in leftovers.items():
        print(f"  ✗ {page}: {', '.join(names)}")

    missing = missing_anchors()
    print(f"links to anchors that don't exist: {sum(map(len, missing.values()))}")
    for page, refs in sorted(missing.items()):
        print(f"  ✗ {page} -> {', '.join(sorted(refs)[:8])}{' ...' if len(refs) > 8 else ''}")

    companions = broken_companion_lessons()
    print(f"companion links to lessons that don't exist: {sum(map(len, companions.values()))}")
    for page, refs in sorted(companions.items()):
        print(f"  ✗ papers/{page} -> {', '.join(sorted(refs))}")

    ghosts: dict[str, list[str]] = {}
    for page in sorted(SITE.rglob("*.html")):
        name = page.relative_to(SITE).as_posix()
        module = name.removesuffix(".html").replace("/", ".") if name.startswith("primer/") else "primer"
        if found := nonexistent_code(page.read_text(errors="ignore"), module):
            ghosts[name] = found
    print(f"code that names a function that doesn't exist: {sum(map(len, ghosts.values()))}")
    for page, names in sorted(ghosts.items()):
        print(f"  ✗ {page}: {', '.join(names)}")

    orphans = orphan_pages()
    print(f"pages nothing links to: {len(orphans)}")
    for page in orphans:
        print(f"  ✗ {page}")

    diagrams = diagram_problems()
    print(f"pages whose diagrams can collide: {len(diagrams)}")
    for page in diagrams:
        print(f"  ✗ {page}")

    empties = empty_links()
    print(f"links with an empty address: {sum(map(len, empties.values()))}")
    for page, texts in sorted(empties.items()):
        print(f"  ✗ {page}: {', '.join(texts)}")

    counts = spec_count_problems()
    print(f"test counts that don't match the suite: {len(counts)}")
    for problem in counts:
        print(f"  ✗ {problem}")

    rendering: dict[str, list[str]] = {}
    for page in sorted(SITE.rglob("*.html")):
        html = page.read_text(errors="ignore")
        if problems := collapsed_blocks(html) + raw_markdown(html) + heading_skips(html) + prices_typeset_as_math(html):
            rendering[page.relative_to(SITE).as_posix()] = problems
    print(f"broken rendering (collapsed lists or diagrams, raw Markdown, prices as math): {sum(map(len, rendering.values()))}")
    for page, problems in rendering.items():
        for problem in problems[:5]:
            print(f"  ✗ {page}: {problem[:110]}")
    return 1 if (bad or repo_bad or unlinked or detours or leftovers or missing or companions or empties
                 or counts or rendering or ghosts or orphans or diagrams) else 0


if __name__ == "__main__":
    sys.exit(main())
