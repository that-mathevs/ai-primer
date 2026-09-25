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
#gl-tip{position:fixed;z-index:1000;max-width:22rem;padding:.6rem .75rem;border-radius:.5rem;
 background:var(--p-tip-bg);color:var(--p-tip-fg);font:14px/1.45 system-ui,sans-serif;box-shadow:0 6px 24px rgba(0,0,0,.25);display:none}
#gl-tip a{color:var(--p-tip-link)}
</style>
<span id="gl-tip" role="tooltip"></span>
<script>
/* Glossary definitions on hover, tap and keyboard focus.
   A focused term is described by the tip (aria-describedby), so screen readers read the definition.
   The tip is moved to sit right after its term, so Tab goes from the term into its "Learn it" link,
   and it stays open while focus is inside it. Escape closes it and returns focus to the term. */
(function(){
  var tip=document.getElementById('gl-tip'), current=null, hideTimer=null;
  function place(el){
    var r=el.getBoundingClientRect(), w=tip.offsetWidth, vw=document.documentElement.clientWidth;
    tip.style.left=Math.max(8,Math.min(r.left, vw-w-8))+'px'; tip.style.top=(r.bottom+6)+'px';
  }
  function show(el){
    clearTimeout(hideTimer);
    if(current&&current!==el){current.removeAttribute('aria-describedby');}
    current=el;
    var html=el.dataset.tip.replace(/&/g,'&amp;').replace(/</g,'&lt;');
    if(el.dataset.lesson){html+=' <a href="'+el.dataset.lesson+'">Learn it &rarr;</a>';}
    tip.innerHTML=html;
    el.after(tip);
    el.setAttribute('aria-describedby','gl-tip');
    tip.style.display='block'; place(el);
  }
  function hideNow(){
    tip.style.display='none';
    if(current){current.removeAttribute('aria-describedby');}
    current=null;
  }
  function hideSoon(){ clearTimeout(hideTimer); hideTimer=setTimeout(function(){
    if(!tip.contains(document.activeElement)){hideNow();} },150); }
  document.querySelectorAll('.gl-term').forEach(function(el){
    el.addEventListener('mouseenter',function(){show(el);});
    el.addEventListener('mouseleave',hideSoon);
    el.addEventListener('focus',function(){show(el);});
    el.addEventListener('blur',hideSoon);
    el.addEventListener('click',function(e){ if(current===el && tip.style.display==='block'){hideNow();} else {show(el);} e.stopPropagation();});
  });
  tip.addEventListener('mouseenter',function(){clearTimeout(hideTimer);});
  tip.addEventListener('mouseleave',hideSoon);
  tip.addEventListener('focusin',function(){clearTimeout(hideTimer);});
  tip.addEventListener('focusout',function(e){ if(e.relatedTarget!==current){hideSoon();} });
  document.addEventListener('click',function(e){ if(!tip.contains(e.target)){hideNow();} });
  document.addEventListener('keydown',function(e){
    if(e.key==='Escape'&&current){var el=current; hideNow(); el.focus({preventScroll:true});}
  });
  window.addEventListener('scroll',function(){ if(current&&!tip.contains(document.activeElement)){place(current);} },{passive:true});
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
# No code mention left unlinked. pdoc links the names it can resolve on a page;
# this pass links the rest: names imported from another module, dunder methods,
# private classes in signatures, and a lesson's own name (its title and its
# "python -m" line), which links to the lesson's code.
# ---------------------------------------------------------------------------

_PASS_THROUGH = {"script", "style", "title", "head", "pre", "textarea"}
_DOTTED = re.compile(r"(?<![\w./#-])primer(?:\.\w+)+")
_CODE_NAME = re.compile(r"[A-Za-z_]\w*(?:\.\w+)*")
_PROSE_NAME = re.compile(
    r"(?<![\w.#/$\\-])(?:primer(?:\.\w+)+|[a-z][a-z0-9]*(?:_[a-z0-9]+)+|[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+)(?![\w])"
)
AMBIGUOUS = object()  # a name more than one thing could mean: reported, never guessed


def _resolve_code_name(name: str, module: str):
    """The object `name` means on `module`'s page: a dotted primer.* path, or a name in the module's namespace."""
    import importlib

    if name.startswith("primer."):
        parts = name.split(".")
        for i in range(len(parts), 0, -1):
            try:
                obj = importlib.import_module(".".join(parts[:i]))
            except ImportError:
                continue
            for part in parts[i:]:
                obj = getattr(obj, part, None)
            return obj
        return None
    obj = importlib.import_module(module)
    for part in name.split("."):
        obj = getattr(obj, part, None)
    return obj


def _code_target(name: str, module: str, page: str, page_html: str) -> str | None:
    """Where a code mention on `page` should link, or None if it names nothing in this repository."""
    import inspect

    obj = _resolve_code_name(name, module)
    if obj is None:
        return None
    if inspect.ismodule(obj):
        if not obj.__name__.startswith("primer"):
            return None
        if obj.__name__ == module:
            # A lesson's own name: the page is already here, so the name leads to the code.
            return code_link(Path(obj.__file__).resolve().relative_to(ROOT).as_posix(), page)
        target = _page(obj.__name__)
        forwards = package_forwards()
        if target in forwards:
            import posixpath

            target = posixpath.normpath(posixpath.join(posixpath.dirname(target), forwards[target]))
        return _rel(target, page)
    obj = getattr(obj, "fget", obj)
    owner = getattr(obj, "__module__", None)
    if owner is None:
        # Plain data (a dict, a list): link only to an entry this page actually has.
        return f"#{name}" if "." not in name and f'id="{name}"' in page_html else None
    if not str(owner).startswith("primer."):
        return None
    qual = getattr(obj, "__qualname__", "")
    if qual and "<locals>" not in qual and not any(part.startswith("_") for part in qual.split(".")):
        # A public member has its own entry on its module's page.
        return f"#{qual}" if owner == module else f"{_rel(_page(owner), page)}#{qual}"
    # pdoc documents neither dunder methods nor private names, so link their lines instead.
    try:
        lines, first = inspect.getsourcelines(inspect.unwrap(obj))
        source = Path(inspect.getsourcefile(obj)).resolve().relative_to(ROOT).as_posix()
    except (TypeError, OSError, ValueError):
        return None
    return code_link(source, page, first, first + len(lines) - 1)


_PACKAGE_NAMES: frozenset[str] | None = None


def _package_names() -> frozenset[str]:
    """Every function, class and method name defined anywhere in the package."""
    import importlib
    import inspect
    import pkgutil

    import primer

    global _PACKAGE_NAMES
    if _PACKAGE_NAMES is None:
        names: set[str] = set()
        for info in pkgutil.walk_packages(primer.__path__, "primer."):
            mod = importlib.import_module(info.name)
            for n, o in vars(mod).items():
                if getattr(o, "__module__", None) == info.name and (inspect.isfunction(o) or inspect.isclass(o)):
                    names.add(n)
                    if inspect.isclass(o):
                        names |= set(vars(o))
        _PACKAGE_NAMES = frozenset(names)
    return _PACKAGE_NAMES


def _scan_code_mentions(page_html: str, module: str, page: str, fix: bool) -> tuple[str, list[str]]:
    """Find (and with fix=True, link) every unlinked mention of code on a page."""
    found: list[str] = []

    def link(text: str, name: str) -> str:
        # The lesson's own title: resolved on its own page, never ambiguous.
        target = _code_target(name, module, page, page_html)
        if target is None:
            return text
        found.append(name)
        return f'<a href="{target}">{text}</a>' if fix else text

    from primer.curriculum import _class_homes, _is_committed

    ids = set(re.findall(r'\bid="([^"]+)"', page_html))
    current_member = [""]  # the member whose documentation is being read

    def _parameters(member: str) -> set[str]:
        import inspect

        obj = _resolve_code_name(member, module) if member else None
        try:
            return set(inspect.signature(obj).parameters) if callable(obj) else set()
        except (TypeError, ValueError):
            return set()

    def target_of(name: str) -> str | None:
        """Where `name` should link, AMBIGUOUS if only the author can say, or None if it names nothing here."""
        target = _code_target(name, module, page, page_html)
        if target or _resolve_code_name(name, module) is not None:
            return target
        # A class from another module, named without it: link it if there's only one it can be.
        homes = _class_homes().get(name.split(".")[0], [])
        if len(homes) == 1:
            return _code_target(f"{homes[0]}.{name}", module, page, page_html)
        if homes:
            return AMBIGUOUS
        # Inside a function's own documentation, its parameters are just parameters.
        if "." not in name and name in _parameters(current_member[0]):
            return None
        # A field or method named without its class: link it if one class on this page has it.
        if "." not in name:
            owners = sorted(i for i in ids if i.count(".") == 1 and i.endswith("." + name) and "-" not in i)
            if len(owners) == 1:
                return f"#{owners[0]}"
            if owners:
                return AMBIGUOUS
        return None

    def link_name(text: str, name: str) -> str:
        target = target_of(name)
        if target is None:
            return text
        found.append(name)
        if target is AMBIGUOUS or not fix:
            return text
        return f'<a href="{target}">{text}</a>'

    def in_code(text: str) -> str:
        stripped = text.strip()
        call = re.fullmatch(r"([A-Za-z_][\w.]*)\((.*)\)", stripped, re.S)
        if call and call.group(2):
            # A call written with its arguments: the function's name is the link.
            name = call.group(1)
            return text.replace(name, link_name(name, name), 1)
        bare = stripped.removesuffix("()")
        if _CODE_NAME.fullmatch(bare):
            return text.replace(bare, link_name(bare, bare), 1)
        if re.fullmatch(r"[\w.-]+(?:/[\w.-]+)+/?", bare) and _is_committed(ROOT / bare):
            found.append(bare)
            return text.replace(bare, f'<a href="{code_link(bare, page)}">{bare}</a>', 1) if fix else text
        return _DOTTED.sub(lambda m: link_name(m.group(0), m.group(0)), text)

    def in_prose(text: str) -> str:
        # Dotted names, and identifiers no English word looks like: snake_case and CamelCase.
        def one(m: re.Match) -> str:
            name = m.group(0)
            target = target_of(name)
            if target is None or target is AMBIGUOUS:
                return name
            found.append(name)
            return f'<a href="{target}">{name}</a>' if fix else name

        return _PROSE_NAME.sub(one, text)

    # pdoc links a module named on its own page with href="", which only reloads the page.
    def empty(m: re.Match) -> str:
        before, after, inner = m.group(1), m.group(2), m.group(3)
        name = re.sub(r"<[^>]+>", "", inner).strip()
        target = _code_target(name, module, page, page_html) if _CODE_NAME.fullmatch(name) else None
        if not target:
            return m.group(0)
        found.append(name)
        return f'{before}href="{target}"{after}{inner}</a>' if fix else m.group(0)

    page_html = re.sub(r'(<a\b[^>]*?)href=""([^>]*>)(.*?)</a>', empty, page_html, flags=re.S)

    out: list[str] = []
    stack: list[str] = []
    signature_flags: list[bool] = []  # parallel to stack: is that element a signature or default value?
    in_display_math = False
    for piece in re.split(r"(<[^>]+>)", page_html):
        if piece.startswith("<"):
            out.append(piece)
            # pdoc opens each member's documentation with a tag whose id is the member's name.
            member = re.match(r'<(?:section|div)\b[^>]*\bid="([A-Za-z_][\w.]*)"', piece)
            if member:
                current_member[0] = member.group(1)
            m = re.match(r"<\s*(/?)\s*([a-zA-Z][\w-]*)", piece)
            if not m or piece.startswith("<!") or piece.endswith("/>") or m.group(2).lower() in _VOID:
                continue
            name = m.group(2).lower()
            if m.group(1):
                if name in stack:
                    at = len(stack) - 1 - stack[::-1].index(name)
                    del stack[at:], signature_flags[at:]
            else:
                stack.append(name)
                # Signatures and default values display code: only fully dotted names link there.
                signature_flags.append(bool(re.search(r'class="[^"]*\b(signature|default_value)\b', piece)))
            continue
        if not piece or "a" in stack or any(t in stack for t in _PASS_THROUGH):
            out.append(piece)
        elif "code" in stack:
            out.append(in_code(piece))
        elif any(signature_flags):
            out.append(_DOTTED.sub(lambda m: link_name(m.group(0), m.group(0)), piece))
        else:
            # Math is typeset from TeX in the browser; a link inside it would break the formula.
            chunks = piece.split("$$")
            for i, chunk in enumerate(chunks):
                if i:
                    in_display_math = not in_display_math
                if not in_display_math:
                    chunks[i] = "".join(part if j % 2 else in_prose(part) for j, part in enumerate(_INLINE_MATH.split(chunk)))
            out.append("$$".join(chunks))
    text = "".join(out)

    # The title's last part is the lesson's own name, left as plain text by pdoc.
    def title(m: re.Match) -> str:
        return m.group(1) + link(m.group(2), module) + m.group(3)

    text = re.sub(r'(<h1 class="modulename">.*<wbr>\.)(\w+)(\s*</h1>)', title, text, count=1, flags=re.S)

    # pdoc also links to entries it never writes (a private base class's members), so those
    # links would land nowhere: send them to the code, or to the nearest enclosing code.
    ids = set(re.findall(r'\bid="([^"]+)"', text))

    def dead(m: re.Match) -> str:
        name = m.group(1)
        if name in ids or not _CODE_NAME.fullmatch(name):
            return m.group(0)
        parts = name.split(".")
        for i in range(len(parts), 0, -1):
            target = _code_target(".".join(parts[:i]), module, page, text)
            if target and not (target.startswith("#") and target[1:] not in ids):
                found.append(name)
                return f'href="{target}"' if fix else m.group(0)
        return m.group(0)

    text = re.sub(r'href="#([^"]+)"', dead, text)
    return text, found


def link_code_mentions(page_html: str, module: str, page: str) -> str:
    """Link every mention of code on a lesson page that pdoc left as plain text."""
    return _scan_code_mentions(page_html, module, page, fix=True)[0]


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
                # Each lesson is written as a link to its file: [ml.attention](../../primer/ml/attention.py).
                "lessons": ["primer." + name for name in re.findall(r"\[?((?:ml|agents|common)(?:\.\w+)+)\]?", lessons)],
                "exists": (ROOT / "docs" / "papers" / f"{slug}.html").exists(),
            }
        )
    return papers


def lesson_nav(module: str, tests: int | None = None) -> str:
    """Breadcrumb plus previous/next links for one lesson page, and how many tests specify it."""
    from primer.curriculum import CURRICULUM, PARTS, neighbours, source_path, tests_for

    page = _page(module)
    index = next(i for i, l in enumerate(CURRICULUM) if l.module == module)
    lesson = CURRICULUM[index]
    part = next(p for p in PARTS if p.key == lesson.part)
    before, after = neighbours(module)
    home = _rel("index.html", page)
    count = f' (<a href="{_rel("spec.html", page)}#{module}">{tests} tests</a>)' if tests else ""

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
        f"<span>Lesson {index} of 0 to {len(CURRICULUM) - 1}: {htmllib.escape(lesson.title)}</span>"
        f'<span class="pn-links"><a href="{home}#lessons">All lessons</a> · '
        f'<a href="{_rel("papers/index.html", page)}">Papers</a> · '
        f'<a href="{_rel("primer/glossary.html", page)}">Glossary</a> · '
        f'<a href="{_rel("primer/notation.html", page)}">Notation</a> · '
        f'<a href="{repo_url()}">Code on GitHub</a><button type="button" class="theme-toggle" data-theme-toggle>Theme</button></span></div>'
        f'<div class="pn-code">Code: <a href="{code_link(source_path(module), page)}">{source_path(module)}</a> · '
        f'Specified by: <a href="{code_link(tests_for(module), page)}">{tests_for(module)}</a>{count} · '
        f"Run: <code>python -m {module}</code></div>"
        f'<div class="pn-steps">{link(before, True)}{link(after, False)}</div>'
        "</div>"
    )


# ---------------------------------------------------------------------------
# The specification. The suite is behaviour-driven and written test-first:
# every test names one behaviour as a sentence. Published as a page, it is a
# precise statement of what each lesson's code does, one click from the test.
# ---------------------------------------------------------------------------


def collect_tests() -> list[str]:
    """Every test in the suite, as pytest names it: "tests/test_x.py::TestArea::test_name[param]"."""
    out = subprocess.run(
        # No -q here: the project's pytest settings already add one, and -qq would print only per-file totals.
        [sys.executable, "-m", "pytest", "--collect-only", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout
    return [line.strip() for line in out.splitlines() if "::" in line]


def tests_per_file(collected: list[str]) -> dict[str, int]:
    """How many tests each test file holds, counting every parametrised case."""
    counts: dict[str, int] = {}
    for test in collected:
        path = test.split("::")[0]
        counts[path] = counts.get(path, 0) + 1
    return counts


def _test_lines(path: str) -> dict[tuple[str | None, str], tuple[int, int]]:
    """(class, test) -> (first line, last line) of each test function in a test file."""
    import ast

    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    lines = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            lines[(None, node.name)] = (node.lineno, node.end_lineno)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    lines[(node.name, item.name)] = (item.lineno, item.end_lineno)
    return lines


def spec_page(collected: list[str]) -> str:
    """The whole suite as a readable specification: one sentence per behaviour, grouped by lesson."""
    from primer.curriculum import CURRICULUM, tests_for
    from tools.spec import humanize_class, humanize_test

    page = "spec.html"
    by_file: dict[str, dict[str | None, list[str]]] = {}
    for test in collected:
        parts = test.split("::")
        path, klass, name = parts[0], (parts[1] if len(parts) == 3 else None), parts[-1].split("[")[0]
        areas = by_file.setdefault(path, {})
        # A parametrised test is one behaviour checked on several inputs: list it once.
        if name not in areas.setdefault(klass, []):
            areas[klass].append(name)
    counts = tests_per_file(collected)

    def behaviours(path: str) -> str:
        lines = _test_lines(path) if (ROOT / path).exists() else {}
        html = []
        for klass, names in by_file.get(path, {}).items():
            if klass:
                html.append(f"<h3>{htmllib.escape(humanize_class(klass))}</h3>")
            items = "".join(
                f'<li><a href="{code_link(path, page, *lines.get((klass, n), ()))}">{htmllib.escape(humanize_test(n))}</a></li>'
                for n in names
            )
            html.append(f"<ul>{items}</ul>")
        return "".join(html)

    sections, toc = [], []
    lesson_files = set()
    for i, lesson in enumerate(CURRICULUM):
        path = tests_for(lesson.module)
        lesson_files.add(path)
        if path not in by_file:
            continue
        n = counts[path]
        toc.append(f'<li><a href="#{lesson.module}">{i}. {htmllib.escape(lesson.title)}</a> <span class="n">{n}</span></li>')
        sections.append(
            f'<section id="{lesson.module}"><h2><a href="{_page(lesson.module)}">{i}. {htmllib.escape(lesson.title)}</a></h2>'
            f'<p class="file">{n} tests in <a href="{code_link(path, page)}">{path}</a></p>{behaviours(path)}</section>'
        )
    others = sorted(p for p in by_file if p not in lesson_files)
    for path in others:
        anchor_id = Path(path).stem
        label = humanize_test(Path(path).stem.removeprefix("test_"))
        toc.append(f'<li><a href="#{anchor_id}">{htmllib.escape(label)}</a> <span class="n">{counts[path]}</span></li>')
        sections.append(
            f'<section id="{anchor_id}"><h2>{htmllib.escape(label)}</h2>'
            f'<p class="file">{counts[path]} tests in <a href="{code_link(path, page)}">{path}</a></p>{behaviours(path)}</section>'
        )
    total = len(collected)
    return (SPEC_TEMPLATE.replace("{{NAV}}", site_nav(page)).replace("{{TOTAL}}", f"{total:,}")
            .replace("{{TOC}}", "".join(toc)).replace("{{SECTIONS}}", "".join(sections)).replace("{{NAV_CSS}}", NAV_CSS.replace("<style>", "").replace("</style>", "")))


SPEC_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>primer: the specification</title>
<link rel="stylesheet" href="assets/theme.css"><script src="assets/theme.js"></script>
<style>
*{box-sizing:border-box}body{margin:0;font:16px/1.6 system-ui,-apple-system,sans-serif}
main{max-width:980px;margin:0 auto;padding:1.5rem 16px 4rem}a{color:var(--p-accent)}
h1{font-size:2rem;margin:.5rem 0}h2{margin-top:2.5rem;border-bottom:1px solid var(--p-border);padding-bottom:.3rem}
h2 a{color:inherit;text-decoration:none}h2 a:hover{color:var(--p-accent)}h3{font-size:1rem;margin:1.2rem 0 .3rem;color:var(--p-muted)}
p{max-width:46rem}.file{color:var(--p-muted);font-size:.92rem}ul{margin:.2rem 0;padding-left:1.2rem}li{margin:.15rem 0}
ul a{color:var(--p-fg);text-decoration:none}ul a:hover{color:var(--p-accent);text-decoration:underline}
.toc{columns:2 18rem;list-style:none;padding:0}.toc a{text-decoration:none}.n{color:var(--p-muted);font-size:.85rem}
{{NAV_CSS}}
</style></head>
<body><main>
{{NAV}}
<h1>The specification: {{TOTAL}} tests</h1>
<p>A test is a small program that runs the lessons' code and checks that it does what the lesson claims. Every test
here was written <em>before</em> the code it checks: first the behaviour is stated and seen to fail, then only
enough code is written to make it pass (this is called test-driven development). Each test is named as a plain
sentence, usually in the form <em>given</em> a situation, <em>when</em> something happens, <em>then</em> a result
(behaviour-driven development), so the whole suite reads as a specification.</p>
<p>Every sentence below is one behaviour; click it to read the test that checks it. A behaviour checked on several
inputs counts as several tests but appears once. The whole suite runs in a few seconds with <code>make test</code>,
offline, and <code>make spec</code> prints this page in a terminal.</p>
<ul class="toc">{{TOC}}</ul>
{{SECTIONS}}
</main></body></html>
"""


DIAGRAM_SCRIPT = """<script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
// Draw each diagram once, in order, under its own name. Mermaid's automatic start names
// diagrams after the clock, so two drawn in the same millisecond collide and overlap.
mermaid.initialize({ startOnLoad: false });
const diagrams = [...document.querySelectorAll("div.mermaid")];
for (const [i, el] of diagrams.entries()) {
  const { svg } = await mermaid.render(`diagram-${i + 1}`, el.textContent);
  el.innerHTML = svg;
  // Name each diagram, and let its "Reading it" paragraph describe it to screen readers.
  const drawing = el.querySelector("svg");
  drawing.setAttribute("role", "img");
  drawing.setAttribute("aria-label", "Diagram");
  let next = el.closest("pre") ? el.closest("pre").nextElementSibling : el.nextElementSibling;
  while (next && !/^Reading it/.test(next.textContent.trim()) && next.tagName !== "H2" && next.tagName !== "H3") next = next.nextElementSibling;
  if (next && /^Reading it/.test(next.textContent.trim())) {
    next.id = next.id || `diagram-${i + 1}-reading`;
    drawing.setAttribute("aria-describedby", next.id);
  }
}
</script>"""


def draw_diagrams_once(page_html: str) -> str:
    """Swap pdoc's mermaid script (automatic start, redraw on every page change) for one that draws each diagram once."""
    return re.sub(r'<script type="module" defer>\s*import mermaid.*?</script>', lambda m: DIAGRAM_SCRIPT, page_html, count=1, flags=re.S)


def label_sections(page_html: str) -> str:
    """pdoc's section labels ("Arguments:", "Returns:", "Inherited Members") as bold labels.

    pdoc writes them as h5/h6 wherever they fall, so a page's outline would jump from h2 to h6;
    they label a block rather than start a section of the page.
    """
    return re.sub(r"<h([56])\b([^>]*)>(.*?)</h\1>", r'<p class="doc-label"\2><strong>\3</strong></p>', page_html, flags=re.S)


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
        f'<a href="{repo_url()}">Code on GitHub</a><button type="button" class="theme-toggle" data-theme-toggle>Theme</button></span></div></div>'
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
.doc-label{margin:.9rem 0 .2rem}
</style>
"""


def package_forwards() -> dict[str, str]:
    """Package pages pdoc would render, each mapped to the home-page section listing the same lessons.

    A package's own page holds little but a reading list the home page already shows,
    so readers who land on one (from pdoc's sidebar or title links) go there instead.
    """
    from primer.curriculum import CURRICULUM

    forwards = {"primer.html": "index.html#lessons"}
    for lesson in CURRICULUM:
        package = lesson.module.rsplit(".", 1)[0]
        if package != "primer":
            page = _page(package)
            forwards.setdefault(page, _rel(f"index.html#{lesson.part}", page))
    return forwards


def skip_forwarded_pages(page_html: str, page: str) -> str:
    """Point links that lead to a forwarded package page straight at its home-page section."""
    import posixpath

    forwards = package_forwards()
    targets = {forwarded: target_from_root for forwarded, target_from_root in
               ((f, posixpath.normpath(posixpath.join(posixpath.dirname(f), t))) for f, t in forwards.items())}
    page_dir = posixpath.dirname(page)

    def fix(m: re.Match) -> str:
        href = m.group(1)
        if EXTERNAL_HREF.match(href) or href.startswith("#"):
            return m.group(0)
        path = posixpath.normpath(posixpath.join(page_dir, href.split("#")[0]))
        if path not in targets:
            return m.group(0)
        return f'href="{_rel(targets[path], page)}"'

    return re.sub(r'href="([^"]*)"', fix, page_html)


EXTERNAL_HREF = re.compile(r"^(https?:|mailto:|data:|javascript:|//)")


def forward_page(target: str) -> str:
    """A page that sends the reader straight on to `target`, with a plain link as a fallback."""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta http-equiv="refresh" content="0; url={target}"><title>primer</title></head>'
        f'<body><p>This page moved to <a href="{target}">the home page</a>.</p></body></html>\n'
    )


def part_intro(part_key: str) -> str:
    """A part's introduction for the home page, from its package's docstring (the text above its reading list)."""
    import importlib

    from primer.curriculum import CURRICULUM

    packages = {l.module.rsplit(".", 1)[0] for l in CURRICULUM if l.part == part_key} - {"primer"}
    if len(packages) != 1:
        return ""
    package = packages.pop()
    doc = importlib.import_module(package).__doc__ or ""
    intro = doc.split("## Reading order")[0].strip()
    intro = re.sub(r"^# .*\n", "", intro).strip()

    # The part's one-line blurb is shown just above; don't say it again.
    from primer.curriculum import PARTS

    blurb = next(p.blurb for p in PARTS if p.key == part_key)
    paragraphs = [" ".join(para.split()) for para in re.split(r"\n\s*\n", intro) if para.strip()]
    paragraphs = [para.removeprefix(blurb).strip() for para in paragraphs]
    return "".join(f'<p class="blurb">{inline_markdown(para, package)}</p>' for para in paragraphs if para)


def inline_markdown(text: str, module: str = "primer") -> str:
    """Markdown emphasis, bold and code to HTML, for text the home page takes from Markdown sources.

    Code that names something in this repository becomes a link, as it does on every page.
    """
    text = htmllib.escape(" ".join(text.split()), quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)([^*]+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", text)

    def code(m: re.Match) -> str:
        target = _code_target(m.group(1), module, "index.html", "") if _CODE_NAME.fullmatch(m.group(1)) else None
        return f'<code><a href="{target}">{m.group(1)}</a></code>' if target else f"<code>{m.group(1)}</code>"

    return re.sub(r"`([^`]+)`", code, text)


def render_home(tests: int | None = None) -> str:
    """The site's front page: every part, every lesson, every paper, generated."""
    from primer.curriculum import BIG_QUESTIONS, CURRICULUM, PARTS, lessons_in

    def card(i, l):
        return (
            f'<li><a href="{_page(l.module)}"><span class="num">{i}</span>'
            f'<span class="t">{htmllib.escape(l.title)}</span>'
            f'<span class="o">{htmllib.escape(l.outcome)}</span></a></li>'
        )

    parts = "".join(
        f'<section id="{p.key}"><h2>{htmllib.escape(p.title)}</h2><p class="blurb">{htmllib.escape(p.blurb)}</p>{part_intro(p.key)}'
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
        + (f'<a href="papers/{p["slug"]}.html">{inline_markdown(p["title"])}</a> <span class="tag">annotated</span>' if p["exists"]
           else f'<a href="{p["sources"][0]}">{inline_markdown(p["title"])}</a> <span class="tag soon">companion coming</span>')
        + '<span class="o">Built in: '
        + ", ".join(f'<a href="{_page(m)}">{htmllib.escape(lesson_titles.get(m, m))}</a>' for m in p["lessons"])
        + "</span></li>"
        for p in catalog()
    )
    from primer.curriculum import REPO_URL

    repo = repo_url()
    return HOME_TEMPLATE.replace("{{EXAMPLE_TESTS}}", code_link("tests/test_attention.py", "index.html")).replace(
        "{{EXAMPLE_CODE}}", code_link("primer/ml/attention.py", "index.html")).replace(
        "{{HOME_REPO}}", REPO_URL).replace("{{REPO}}", repo).replace("{{REPO_NAME}}", repo.rsplit("/", 1)[-1]).replace(
        "{{REPO_LABEL}}", repo.split("://", 1)[-1]).replace("{{LICENSE}}", code_link("LICENSE", "index.html")).replace("{{BIG}}", big).replace("{{PARTS}}", parts).replace("{{PAPERS}}", paper_rows).replace(
        "{{COUNT}}", str(len(CURRICULUM))).replace("{{LAST}}", str(len(CURRICULUM) - 1)).replace(
        "{{TESTS}}", f"a suite of {tests:,} tests" if tests else "a suite of tests")


HOME_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>primer: how modern AI works</title>
<link rel="stylesheet" href="assets/theme.css"><script src="assets/theme.js"></script>
<style>
:root{--bg:var(--p-bg);--fg:var(--p-fg);--muted:var(--p-muted);--line:var(--p-border);--card:var(--p-card);--accent:var(--p-accent)}
.repo{font-weight:600}.top{display:flex;align-items:flex-start;justify-content:space-between;gap:1rem}.top .theme-toggle{margin-top:.9rem}
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
<p>{{COUNT}} lessons, numbered 0 to {{LAST}}. Every idea is built in plain Python, drawn, and decoded symbol by symbol.
Hover over any underlined term for a plain-English definition.</p>
<p>All of it is pinned down by {{TESTS}}: small programs that run the lessons' code and check it does what the
lessons claim. They were written before the code (test-driven), and each is named as a plain sentence in the
form <em>given</em> a situation, <em>then</em> a result (behaviour-driven), such as
<em>given a causal mask, future tokens receive zero attention</em> (<a href="{{EXAMPLE_TESTS}}">the attention lesson's tests</a>). Read together,
they are a precise specification of what every lesson teaches: <a href="spec.html">read the specification</a>.</p>
<p>Every lesson also runs on its own in a terminal as a narrated walkthrough (<code>python -m <a href="{{EXAMPLE_CODE}}">primer.ml.attention</a></code>),
and ends with links to the primary sources. The shared toy data and stand-in embedder the lessons use live in
<a href="primer/common.html"><code>primer.common</code></a>.</p>
<p class="repo">The code: <a href="{{HOME_REPO}}">{{HOME_REPO}}</a></p>
<p>Every map on this site (this page, the reading order, each lesson's previous and next) is generated from one file,
<a href="primer/curriculum.html"><code>primer.curriculum</code></a>, so none of them can go stale.</p></header>
<nav class="jump" aria-label="Jump to">
<a href="#lessons">Lessons</a><a href="#big">Big questions</a><a href="spec.html">Specification</a><a href="primer/notation.html">Math notation</a><a href="primer/glossary.html">Glossary</a>
<a href="#papers">Annotated papers</a>
<a href="{{REPO}}">Code on GitHub</a></nav>
<div id="lessons">{{PARTS}}</div>
<section id="big"><h2>Big questions</h2>
<p class="blurb">The lessons build the field from the bottom up. These questions give the top-down view: open one to see the
points a complete answer covers, and the lessons that teach them, in order.</p>{{BIG}}</section>
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

    from primer.curriculum import CURRICULUM

    return (
        "window.PRIMER_PAPERS = " + json.dumps(catalog(), ensure_ascii=False, indent=1) + ";\n"
        + f"window.PRIMER_REPO = {json.dumps(repo_url())};\n"
        # Companions name each lesson by its title, as the home page does.
        + "window.PRIMER_LESSONS = " + json.dumps({l.module: l.title for l in CURRICULUM}, ensure_ascii=False) + ";\n"
    )


def _postprocess(path: Path, terms: dict[str, tuple], counts: dict[str, int] | None = None) -> None:
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
        from primer.curriculum import tests_for

        nav = lesson_nav(module, tests=(counts or {}).get(tests_for(module)))
        text = re.sub(r"(<main[^>]*>)", lambda m: m.group(1) + nav, text, count=1)
        text = text.replace("</main>", nav.replace('class="primer-nav"', 'class="primer-nav pn-bottom"') + "</main>", 1)
    else:
        text = re.sub(r"(<main[^>]*>)", lambda m: m.group(1) + site_nav(page), text, count=1)
    text = draw_diagrams_once(text)
    text = label_sections(text)
    text = link_code_mentions(text, module, page)
    text = link_members_to_source(text, module, page)
    text = skip_forwarded_pages(text, page)
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
        # primer._show is a private helper for the terminal walkthroughs, not something a reader studies.
        + ["primer", "!primer._show", "--docformat", "google", "--math", "--mermaid",
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
    collected = collect_tests()
    counts = tests_per_file(collected)
    (SITE / "index.html").write_text(render_home(tests=len(collected)), encoding="utf-8")
    (SITE / "spec.html").write_text(spec_page(collected), encoding="utf-8")

    terms = {t: (e.definition, lesson_path(e.lesson) if e.lesson else None, e.scope) for t, e in GLOSSARY.items()}
    pages = sorted((SITE / "primer").rglob("*.html"))
    for p in pages:
        _postprocess(p, terms, counts)
    for page, target in package_forwards().items():
        (SITE / page).write_text(forward_page(target), encoding="utf-8")
    print(f"built {len(pages)} lesson pages into {SITE.relative_to(ROOT)}/  (open docs/html/index.html)")
    return 0


if __name__ == "__main__":
    sys.exit(build())
