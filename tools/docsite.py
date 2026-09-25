"""
Build the browsable HTML site into docs/html.

    python tools/docsite.py        # or: make docs

Steps:
1. Render every lesson's figures (tools/figures.py) into docs/figures/.
2. Run pdoc over the `primer` package (markdown docstrings, KaTeX math,
   mermaid diagrams).
3. Copy docs/figures and docs/papers into the site, and write the shared
   glossary as glossary.js.
4. Post-process every lesson page: point figure images at the right relative
   path, and give the first mention of each glossary term a hover definition
   (`annotate_html`), plus the small script and styles that show it.
"""

from __future__ import annotations

import os

# Single-threaded BLAS: much faster on the lessons' tiny matrices (set before NumPy loads).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import html as htmllib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "docs" / "html"

# Hover definitions never go inside these: code would be corrupted, links and
# headings are navigation, and math/diagrams are rendered by their own scripts.
SKIP_TAGS = {"a", "code", "pre", "h1", "h2", "h3", "h4", "h5", "h6", "script", "style", "title", "button", "nav", "svg", "math", "summary", "label", "textarea"}
SKIP_CLASSES = ("gl-term", "katex", "math", "mermaid", "pdoc-code", "docstring-signature")

_ATOMIC = re.compile(r"(<!--.*?-->|<script\b.*?</script>|<style\b.*?</style>)", re.S | re.I)
_TAG = re.compile(r"(<[^>]+>)")
_TAG_NAME = re.compile(r"<\s*(/?)\s*([a-zA-Z][a-zA-Z0-9-]*)")
# Math is left as raw TeX for the browser's typesetter: $$display$$, $inline$, \\( \\), \\[ \\].
_INLINE_MATH = re.compile(r"(\$[^$\n]+?\$|\\\(.*?\\\)|\\\[.*?\\\])", re.S)
_VOID = {"br", "img", "hr", "input", "meta", "link", "source", "wbr", "col", "area", "base", "embed", "param", "track"}


def annotate_html(page_html: str, terms: dict[str, tuple], page: str) -> str:
    """Wrap the first prose mention of each term in a hover-definition span.

    Args:
        page_html: the page's HTML.
        terms: lowercase term -> (definition, lesson path or None[, scope prefixes]).
            A term with scope prefixes is only annotated on pages under one of them.
        page: this page's path relative to the site root, for relative lesson links.

    Everything outside the changed spans is returned byte for byte.
    """
    terms = {t: v for t, v in terms.items() if len(v) < 3 or not v[2] or page.startswith(tuple(v[2]))}
    if not terms:
        return page_html
    # Longest terms first, so "multi-head attention" wins over "attention".
    alternation = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    term_re = re.compile(rf"(?<![\w-])({alternation})(?![\w-])", re.I)
    used: set[str] = set()
    page_dir = os.path.dirname(page)

    def wrap(match: re.Match) -> str:
        key = match.group(1).lower()
        if key in used:
            return match.group(0)
        used.add(key)
        definition, lesson = terms[key][:2]
        attrs = f'class="gl-term" tabindex="0" data-tip="{htmllib.escape(definition, quote=True)}"'
        if lesson and lesson != page:
            attrs += f' data-lesson="{os.path.relpath(lesson, page_dir or ".")}"'
        return f"<span {attrs}>{match.group(0)}</span>"

    def annotate_prose(text: str) -> str:
        # Only annotate outside inline math; display math is handled by the caller.
        parts = _INLINE_MATH.split(text)
        return "".join(part if i % 2 else term_re.sub(wrap, part) for i, part in enumerate(parts))

    out: list[str] = []
    stack: list[tuple[str, bool]] = []  # (tag, is_skip)
    skipping = 0
    in_display_math = False  # $$ ... $$ can span several tags
    for chunk in _ATOMIC.split(page_html):
        if _ATOMIC.fullmatch(chunk or ""):
            out.append(chunk)
            continue
        for piece in _TAG.split(chunk):
            if not piece:
                continue
            if piece.startswith("<"):
                out.append(piece)
                m = _TAG_NAME.match(piece)
                if not m or piece.startswith("<!"):
                    continue
                closing, name = m.group(1) == "/", m.group(2).lower()
                if name in _VOID or piece.endswith("/>"):
                    continue
                if closing:
                    # Pop to the matching open tag (tolerates sloppy nesting).
                    for i in range(len(stack) - 1, -1, -1):
                        if stack[i][0] == name:
                            for _, was_skip in stack[i:]:
                                skipping -= was_skip
                            del stack[i:]
                            break
                else:
                    cls = re.search(r'class\s*=\s*"([^"]*)"', piece)
                    is_skip = name in SKIP_TAGS or bool(cls and any(c in cls.group(1).split() for c in SKIP_CLASSES))
                    stack.append((name, is_skip))
                    skipping += is_skip
            elif skipping:
                out.append(piece)
            else:
                # Split on $$ and toggle: even chunks are outside display math at the current state.
                chunks = piece.split("$$")
                rendered = []
                for i, chunk in enumerate(chunks):
                    if i:
                        in_display_math = not in_display_math
                    rendered.append(chunk if in_display_math else annotate_prose(chunk))
                out.append("$$".join(rendered))
    return "".join(out)


TOOLTIP_ASSETS = """
<style>
.gl-term{border-bottom:1px dotted currentColor;cursor:help}
.gl-term:focus{outline:2px solid var(--p-accent);outline-offset:2px}
#gl-tip{position:absolute;z-index:1000;max-width:22rem;padding:.6rem .75rem;border-radius:.5rem;
 background:var(--p-tip-bg);color:var(--p-tip-fg);font:14px/1.45 system-ui,sans-serif;box-shadow:0 6px 24px rgba(0,0,0,.25);display:none}
#gl-tip a{color:var(--p-tip-link)}
</style>
<div id="gl-tip" role="tooltip"></div>
<script>
(function(){
  var tip=document.getElementById('gl-tip'), current=null, hideTimer=null;
  function show(el){
    clearTimeout(hideTimer); current=el;
    var html=el.dataset.tip.replace(/&/g,'&amp;').replace(/</g,'&lt;');
    if(el.dataset.lesson){html+=' <a href="'+el.dataset.lesson+'">Learn it &rarr;</a>';}
    tip.innerHTML=html; tip.style.display='block';
    var r=el.getBoundingClientRect(), w=tip.offsetWidth;
    var left=Math.max(8,Math.min(window.scrollX+r.left, window.scrollX+document.documentElement.clientWidth-w-8));
    tip.style.left=left+'px'; tip.style.top=(window.scrollY+r.bottom+6)+'px';
  }
  function hide(){hideTimer=setTimeout(function(){tip.style.display='none';current=null;},150);}
  document.querySelectorAll('.gl-term').forEach(function(el){
    el.addEventListener('mouseenter',function(){show(el);});
    el.addEventListener('mouseleave',hide);
    el.addEventListener('focus',function(){show(el);});
    el.addEventListener('blur',hide);
    el.addEventListener('click',function(e){ if(current===el && tip.style.display==='block'){hide();} else {show(el);} e.stopPropagation();});
  });
  tip.addEventListener('mouseenter',function(){clearTimeout(hideTimer);});
  tip.addEventListener('mouseleave',hide);
  document.addEventListener('click',function(e){ if(!tip.contains(e.target)){tip.style.display='none';current=null;} });
  document.addEventListener('keydown',function(e){ if(e.key==='Escape'){tip.style.display='none';} });
})();
</script>
"""


# ---------------------------------------------------------------------------
# Where code links point. The published site holds only HTML, so its links go
# to GitHub, pinned to the exact commit the site was built from: the lines they
# name always match the page. A site built on your own machine links to the
# files in your own checkout instead, so it always shows the code you have.
# ---------------------------------------------------------------------------


def web_url_of_remote(remote: str) -> str:
    """A git remote's web page: git@github.com:o/r.git and friends become https://github.com/o/r."""
    url = remote.strip()
    ssh = re.match(r"^(?:ssh://)?git@([^:/]+)[:/](.+)$", url)
    if ssh:
        url = f"https://{ssh.group(1)}/{ssh.group(2)}"
    return re.sub(r"\.git$", "", url.rstrip("/"))


def repo_url() -> str:
    """This repository's web page: the Actions run's own repository when publishing,
    else the checkout's origin remote (so a fork links to itself), else the project's home."""
    if os.environ.get("GITHUB_REPOSITORY"):
        return f'{os.environ.get("GITHUB_SERVER_URL", "https://github.com")}/{os.environ["GITHUB_REPOSITORY"]}'
    out = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode == 0 and out.stdout.strip():
        return web_url_of_remote(out.stdout)
    from primer.curriculum import REPO_URL

    return REPO_URL


def published_commit() -> str | None:
    """The commit a published build comes from, or None for a build on a reader's machine."""
    return os.environ.get("GITHUB_SHA") if os.environ.get("GITHUB_ACTIONS") == "true" else None


def code_link(path: str, page: str, first: int | None = None, last: int | None = None) -> str:
    """A link from `page` (relative to the site root) to a file in the repository, optionally to lines."""
    sha = published_commit()
    if sha is None:
        # Browsers don't jump to line numbers in a plain file, so a local link names the file only.
        return os.path.relpath(ROOT / path, (SITE / page).parent).replace(os.sep, "/")
    url = f"{repo_url()}/blob/{sha}/{path}"
    if first is not None:
        url += f"#L{first}" + (f"-L{last}" if last is not None and last != first else "")
    return url


_SOURCE_BUTTON = re.compile(r'(<label class="view-source-button" for="([^"]+)-view-source">.*?</label>)', re.S)


def _member_lines(module, ident: str) -> tuple[int, int] | None:
    """First and last line of `ident` (like "run_chain" or "Tool.call") in the module's own file.

    None when the member is defined in another file, such as a method inherited
    from the standard library, since a line range in this file would be wrong.
    """
    import inspect

    obj = module
    for part in ident.split("."):
        obj = getattr(obj, part, None)
    obj = getattr(obj, "fget", obj)  # a property's lines are its getter's
    obj = inspect.unwrap(obj) if callable(obj) else obj
    try:
        if Path(inspect.getsourcefile(obj) or "").resolve() != Path(module.__file__).resolve():
            return None
        lines, first = inspect.getsourcelines(obj)
    except (TypeError, OSError):
        return None
    return first, first + len(lines) - 1


def link_members_to_source(page_html: str, module_name: str, page: str) -> str:
    """Put a link to each member's source beside pdoc's View Source button (see `code_link`).

    Args:
        page_html: the page pdoc rendered for one module.
        module_name: that module's dotted name, such as "primer.agents.orchestration".
        page: the page's path relative to the site root.
    """
    import importlib

    module = importlib.import_module(module_name)
    source = Path(module.__file__).resolve().relative_to(ROOT).as_posix()
    label = "on GitHub" if published_commit() else "open file"

    def add(m: re.Match) -> str:
        ident = m.group(2)
        # The module's own button covers the whole file.
        span = () if ident.startswith("mod-") else _member_lines(module, ident)
        if span is None:
            return m.group(1)
        return m.group(1) + f'<a class="code-source" href="{code_link(source, page, *span)}">{label}</a>'

    return _SOURCE_BUTTON.sub(add, page_html)


# ---------------------------------------------------------------------------
# Navigation: generated from primer/curriculum.py and docs/papers/CATALOG.md,
# so it can never drift from the lessons and papers that actually exist.
# ---------------------------------------------------------------------------


def _page(module: str) -> str:
    return module.replace(".", "/") + ".html"


def _rel(target: str, from_page: str) -> str:
    return os.path.relpath(target, os.path.dirname(from_page) or ".").replace(os.sep, "/")


def catalog() -> list[dict]:
    """Parse docs/papers/CATALOG.md into one dict per paper, marking which companions exist."""
    papers = []
    for line in (ROOT / "docs" / "papers" / "CATALOG.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*`([a-z0-9-]+)`\s*\|(.*?)\|(.*?)\|(.*?)\|\s*$", line)
        if not m:
            continue
        slug, paper, sources, lessons = (g.strip() for g in m.groups())
        papers.append(
            {
                "slug": slug,
                "title": paper,
                "sources": re.findall(r"https?://\S+", sources),
                "lessons": ["primer." + l.strip() for l in lessons.split(",") if l.strip()],
                "exists": (ROOT / "docs" / "papers" / f"{slug}.html").exists(),
            }
        )
    return papers


def lesson_nav(module: str) -> str:
    """Breadcrumb plus previous/next links for one lesson page."""
    from primer.curriculum import CURRICULUM, PARTS, neighbours, source_path, tests_for

    page = _page(module)
    index = next(i for i, l in enumerate(CURRICULUM) if l.module == module)
    lesson = CURRICULUM[index]
    part = next(p for p in PARTS if p.key == lesson.part)
    before, after = neighbours(module)
    home = _rel("index.html", page)

    def link(l, arrow_first: bool) -> str:
        if l is None:
            return "<span></span>"
        n = next(i for i, x in enumerate(CURRICULUM) if x is l)
        label = f"{n}. {htmllib.escape(l.title)}"
        text = f"&larr; {label}" if arrow_first else f"{label} &rarr;"
        return f'<a class="pn-{"prev" if arrow_first else "next"}" href="{_rel(_page(l.module), page)}">{text}</a>'

    return (
        # A div, not <nav>: pdoc's stylesheet positions every <nav> as its sidebar.
        '<div class="primer-nav" role="navigation" aria-label="Lesson navigation">'
        f'<div class="pn-crumbs"><a href="{home}">primer</a> &rsaquo; '
        f'<a href="{home}#{part.key}">{htmllib.escape(part.title)}</a> &rsaquo; '
        f"<span>Lesson {index} of {len(CURRICULUM) - 1}: {htmllib.escape(lesson.title)}</span>"
        f'<span class="pn-links"><a href="{home}#lessons">All lessons</a> · '
        f'<a href="{_rel("papers/index.html", page)}">Papers</a> · '
        f'<a href="{_rel("primer/glossary.html", page)}">Glossary</a> · '
        f'<a href="{_rel("primer/notation.html", page)}">Notation</a> · '
        f'<a href="{repo_url()}">GitHub</a><button type="button" class="theme-toggle" data-theme-toggle>Theme</button></span></div>'
        f'<div class="pn-code">Code: <a href="{code_link(source_path(module), page)}">{source_path(module)}</a> · '
        f'Specified by: <a href="{code_link(tests_for(module), page)}">{tests_for(module)}</a> · '
        f"Run: <code>python -m {module}</code></div>"
        f'<div class="pn-steps">{link(before, True)}{link(after, False)}</div>'
        "</div>"
    )


def add_theme(page_html: str, page: str) -> str:
    """Load the shared theme at the end of <head>: after pdoc's styles so it wins,
    and before the body so a dark choice never flashes light."""
    root = _rel("assets", page)
    return page_html.replace(
        "</head>", f'<link rel="stylesheet" href="{root}/theme.css"><script src="{root}/theme.js"></script>\n</head>', 1
    )


def site_nav(page: str) -> str:
    """The bar at the top of pages that aren't lessons: glossary, packages, shared code."""
    home = _rel("index.html", page)
    return (
        '<div class="primer-nav" role="navigation" aria-label="Site navigation"><div class="pn-crumbs">'
        f'<a href="{home}">primer</a><span class="pn-links"><a href="{home}#lessons">All lessons</a> · '
        f'<a href="{_rel("papers/index.html", page)}">Papers</a> · '
        f'<a href="{_rel("primer/glossary.html", page)}">Glossary</a> · '
        f'<a href="{_rel("primer/notation.html", page)}">Notation</a> · '
        f'<a href="{repo_url()}">GitHub</a><button type="button" class="theme-toggle" data-theme-toggle>Theme</button></span></div></div>'
    )


NAV_CSS = """
<style>
.primer-nav{font:14px/1.5 system-ui,sans-serif;margin:0 0 1.5rem;padding:.6rem .8rem;border:1px solid var(--p-border);border-radius:.5rem;background:var(--p-card)}
.primer-nav a{text-decoration:none;color:var(--p-accent)}
.pn-crumbs{display:flex;flex-wrap:wrap;gap:.25rem .4rem;align-items:baseline;color:var(--p-muted)}
.pn-links{margin-left:auto}
.pn-code{color:var(--p-muted);margin-top:.25rem;font-size:13px}
.pn-code code{font-size:12px}
.code-source{float:right;font-size:.75rem;line-height:1.5rem;padding:0 .6rem}
.primer-nav .theme-toggle{margin-left:.5rem}
.pn-steps{display:flex;justify-content:space-between;gap:1rem;margin-top:.4rem;font-weight:600}
.pn-next{text-align:right;margin-left:auto}
.primer-nav.pn-bottom{margin:2.5rem 0 0}
.pdoc img{max-width:100%;height:auto}
</style>
"""


def render_home() -> str:
    """The site's front page: every part, every lesson, every paper, generated."""
    from primer.curriculum import BIG_QUESTIONS, CURRICULUM, PARTS, lessons_in

    def card(i, l):
        return (
            f'<li><a href="{_page(l.module)}"><span class="num">{i}</span>'
            f'<span class="t">{htmllib.escape(l.title)}</span>'
            f'<span class="o">{htmllib.escape(l.outcome)}</span></a></li>'
        )

    parts = "".join(
        f'<section id="{p.key}"><h2>{htmllib.escape(p.title)}</h2><p class="blurb">{htmllib.escape(p.blurb)}</p>'
        f'<ol class="cards">{"".join(card(i, l) for i, l in lessons_in(p.key))}</ol></section>'
        for p in PARTS
    )
    lesson_titles = {l.module: l.title for l in CURRICULUM}
    big = "".join(
        f'<details><summary>{htmllib.escape(q.question)}</summary>'
        f'<p class="route">Route: ' + " &rarr; ".join(f'<a href="{_page(m)}">{htmllib.escape(lesson_titles[m])}</a>' for m in q.route) + "</p>"
        "<ol>" + "".join(f"<li>{htmllib.escape(pt)}</li>" for pt in q.spine) + "</ol></details>"
        for q in BIG_QUESTIONS
    )
    paper_rows = "".join(
        "<li>"
        + (f'<a href="papers/{p["slug"]}.html">{p["title"]}</a> <span class="tag">annotated</span>' if p["exists"]
           else f'<a href="{p["sources"][0]}">{p["title"]}</a> <span class="tag soon">companion coming</span>')
        + '<span class="o">Built in: '
        + ", ".join(f'<a href="{_page(m)}">{htmllib.escape(lesson_titles.get(m, m))}</a>' for m in p["lessons"])
        + "</span></li>"
        for p in catalog()
    )
    repo = repo_url()
    return HOME_TEMPLATE.replace("{{REPO}}", repo).replace("{{REPO_NAME}}", repo.rsplit("/", 1)[-1]).replace(
        "{{REPO_LABEL}}", repo.split("://", 1)[-1]).replace("{{LICENSE}}", code_link("LICENSE", "index.html")).replace("{{BIG}}", big).replace("{{PARTS}}", parts).replace("{{PAPERS}}", paper_rows).replace(
        "{{COUNT}}", str(len(CURRICULUM))
    )


HOME_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>primer: how modern AI works</title>
<link rel="stylesheet" href="assets/theme.css"><script src="assets/theme.js"></script>
<style>
:root{--bg:var(--p-bg);--fg:var(--p-fg);--muted:var(--p-muted);--line:var(--p-border);--card:var(--p-card);--accent:var(--p-accent)}
.top{display:flex;align-items:flex-start;justify-content:space-between;gap:1rem}.top .theme-toggle{margin-top:.9rem}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,-apple-system,sans-serif}
main{max-width:980px;margin:0 auto;padding:2rem 16px 4rem}
a{color:var(--accent)}header h1{font-size:2.2rem;margin:.2rem 0}header p{color:var(--muted);max-width:44rem}
.jump{display:flex;flex-wrap:wrap;gap:.5rem;margin:1rem 0 2rem}.jump a{padding:.35rem .7rem;border:1px solid var(--line);border-radius:999px;text-decoration:none}
h2{margin-top:2.5rem;border-bottom:1px solid var(--line);padding-bottom:.3rem}.blurb{color:var(--muted)}
.cards{list-style:none;padding:0;display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:.75rem}
.cards a{display:grid;grid-template-columns:2rem 1fr;gap:0 .5rem;padding:.8rem;border:1px solid var(--line);border-radius:.6rem;background:var(--card);text-decoration:none;color:inherit;height:100%}
.cards a:hover,.cards a:focus{border-color:var(--accent)}
.num{grid-row:span 2;font-weight:700;color:var(--accent)}.t{font-weight:600}.o{color:var(--muted);font-size:.9rem;display:block}
.papers{padding-left:1.1rem}.papers li{margin:.5rem 0}.tag{font-size:.75rem;padding:.05rem .45rem;border-radius:999px;background:var(--accent);color:var(--bg)}
.tag.soon{background:var(--line);color:var(--muted)}
details{border:1px solid var(--line);border-radius:.6rem;padding:.6rem .9rem;margin:.5rem 0;background:var(--card)}
summary{cursor:pointer;font-weight:600}details[open] summary{margin-bottom:.4rem}
footer{margin-top:3rem;border-top:1px solid var(--line);color:var(--muted)}
pre{background:var(--card);border:1px solid var(--line);border-radius:.5rem;padding:.8rem;overflow-x:auto}
.route{color:var(--muted);font-size:.92rem;margin:.3rem 0}details ol{margin:.3rem 0 .2rem;padding-left:1.3rem}
</style></head>
<body><main>
<header><div class="top"><h1>primer: how modern AI works, built from scratch</h1><button type="button" class="theme-toggle" data-theme-toggle>Theme</button></div>
<p>{{COUNT}} lessons. Every idea is built in plain Python, drawn, decoded symbol by symbol, and checked by a test.
Hover over any underlined term for a plain-English definition.</p></header>
<nav class="jump" aria-label="Jump to">
<a href="#big">Big questions</a><a href="#lessons">Lessons</a><a href="primer/notation.html">Math notation</a><a href="primer/glossary.html">Glossary</a>
<a href="#papers">Annotated papers</a><a href="primer.html">Browse the code</a>
<a href="{{REPO}}">Source on GitHub</a></nav>
<section id="big"><h2>Big questions</h2>
<p class="blurb">The lessons build the field from the bottom up. Start here for the top-down view: open a question to see the
points a complete answer covers, and the lessons that teach them, in order.</p>{{BIG}}</section>
<div id="lessons">{{PARTS}}</div>
<section id="papers"><h2>The papers behind the lessons</h2>
<p class="blurb">Annotated, interactive companions: hover over any term or equation symbol. <a href="papers/index.html">All papers</a>.</p>
<ul class="papers">{{PAPERS}}</ul></section>
<footer><p>Every lesson is one Python file: the explanation is its docstring, the code follows it, and a test file
specifies it. Clone it and run any lesson offline with only NumPy:</p>
<pre><code>git clone {{REPO}}
cd {{REPO_NAME}} &amp;&amp; python -m pip install -e ".[dev]"
python -m primer.ml.attention</code></pre>
<p><a href="{{REPO}}">{{REPO_LABEL}}</a> · <a href="{{LICENSE}}">License: free for noncommercial use</a></p></footer>
</main></body></html>
"""


def catalog_js() -> str:
    import json

    return (
        "window.PRIMER_PAPERS = " + json.dumps(catalog(), ensure_ascii=False, indent=1) + ";\n"
        + f"window.PRIMER_REPO = {json.dumps(repo_url())};\n"
    )


def _postprocess(path: Path, terms: dict[str, tuple]) -> None:
    page = path.relative_to(SITE).as_posix()
    depth = os.path.relpath(SITE, path.parent).replace(os.sep, "/")
    text = path.read_text(encoding="utf-8")
    # Docstrings reference figures as "figures/<name>.svg"; make that relative to this page.
    text = text.replace('src="figures/', f'src="{depth}/figures/')
    # Lesson links to companions are written relative to the lesson page already.
    body_start = text.find("<main")
    if body_start != -1:
        text = text[:body_start] + annotate_html(text[body_start:], terms, page)
    # Lessons get a breadcrumb and previous/next bar at the top and bottom.
    from primer.curriculum import CURRICULUM

    module = page.removesuffix(".html").replace("/", ".")
    if any(l.module == module for l in CURRICULUM):
        nav = lesson_nav(module)
        text = re.sub(r"(<main[^>]*>)", lambda m: m.group(1) + nav, text, count=1)
        text = text.replace("</main>", nav.replace('class="primer-nav"', 'class="primer-nav pn-bottom"') + "</main>", 1)
    else:
        text = re.sub(r"(<main[^>]*>)", lambda m: m.group(1) + site_nav(page), text, count=1)
    text = link_members_to_source(text, module, page)
    text = text.replace("</head>", NAV_CSS + "</head>", 1)
    text = add_theme(text, page)
    text = text.replace("</body>", TOOLTIP_ASSETS + "</body>", 1)
    path.write_text(text, encoding="utf-8")


def build() -> int:
    sys.path.insert(0, str(ROOT))
    from primer.glossary import GLOSSARY, glossary_js, lesson_path

    rc = subprocess.run([sys.executable, str(ROOT / "tools" / "figures.py")], cwd=ROOT).returncode
    if rc:
        print("figures: problems found (see above); building the site anyway")

    if SITE.exists():
        shutil.rmtree(SITE)
    pdoc = [sys.executable, "-m", "pdoc"]
    if subprocess.run(pdoc + ["--version"], capture_output=True).returncode != 0:
        pdoc = ["uvx", "--with", "numpy", "--with", "matplotlib", "pdoc"]
    subprocess.run(
        pdoc
        + ["primer", "--docformat", "markdown", "--math", "--mermaid",
           "--footer-text", "primer: how modern AI works, built from scratch", "-o", str(SITE)],
        cwd=ROOT, check=True, env={**os.environ, "PYTHONPATH": str(ROOT)},
    )

    for sub in ("assets", "figures", "papers"):
        if (ROOT / "docs" / sub).exists():
            shutil.copytree(ROOT / "docs" / sub, SITE / sub)
    for name, js in (("glossary.js", glossary_js()), ("catalog.js", catalog_js())):
        for target in (ROOT / "docs" / "papers" / "assets" / name, SITE / "papers" / "assets" / name):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(js, encoding="utf-8")
    (SITE / "index.html").write_text(render_home(), encoding="utf-8")

    terms = {t: (e.definition, lesson_path(e.lesson) if e.lesson else None, e.scope) for t, e in GLOSSARY.items()}
    pages = sorted((SITE / "primer").rglob("*.html"))
    for p in pages:
        _postprocess(p, terms)
    print(f"built {len(pages)} lesson pages into {SITE.relative_to(ROOT)}/  (open docs/html/index.html)")
    return 0


if __name__ == "__main__":
    sys.exit(build())
