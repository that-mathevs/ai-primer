"""Specification for house style rules that apply to every file in the repository."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT_SUFFIXES = {".py", ".md", ".html", ".js", ".css", ".toml", ".yml", ".yaml", ".txt"}
GENERATED = ("docs/html/",)  # the built site mirrors the sources checked here
EM_DASH = chr(0x2014)  # built from its code point so this file never contains one


def source_files():
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT).as_posix()
        if path.is_file() and (path.suffix in TEXT_SUFFIXES or path.name == "Makefile") and not rel.startswith(GENERATED) and "/." not in "/" + rel:
            yield rel, path


class TestPunctuation:
    def test_given_any_source_file_it_contains_no_em_dashes(self):
        # House rule: use a colon, a comma, parentheses or a new sentence instead.
        offenders = {}
        for rel, path in source_files():
            for n, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                if EM_DASH in line:
                    offenders.setdefault(rel, []).append(n)
        assert offenders == {}, f"em dashes found (file: line numbers): {offenders}"


# Coaching language: this is a primer, not a guide to performing answers. Built so this file doesn't trip itself.
COACHING = [
    "strong " + "answer", "points to " + "hit", "spine of " + "the answer",
    "spine of a", "practise answering", "practice answering", "answering each question out " + "loud",
    "explaining a topic out " + "loud", "walk me " + "through", "walk " + "through what", "walk " + "through exactly",
    "walk " + "through bpe", "walk " + "through an", "argue both " + "sides", "what would you " + "say",
    "how would you " + "answer", "say " + "this", "talking " + "points", "in an " + "interview", "interview" + "er",
    "what do you " + "do?",
]
READER_FACING = ("primer/", "docs/", "README.md")


class TestTone:
    def test_given_anything_a_reader_sees_it_teaches_rather_than_coaches_answers(self):
        offenders = {}
        for rel, path in source_files():
            if not rel.startswith(READER_FACING) or rel.endswith((".js", ".css")):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            found = [phrase for phrase in COACHING if phrase in text]
            if found:
                offenders[rel] = found
        assert offenders == {}, f"coaching language found: {offenders}"


def _inline_math_spans(doc: str) -> list[str]:
    """What MathJax would typeset between single $ signs, paragraph by paragraph and table cell by table cell."""
    import re

    doc = re.sub(r"```[\s\S]*?```", "\n\n", doc)  # code blocks are never typeset
    doc = re.sub(r"\$\$[\s\S]*?\$\$", " ", doc)  # display math is fine as it is
    doc = re.sub(r"`[^`\n]*`", " ", doc)
    doc = doc.replace("\\$", " ")  # an escaped dollar is a plain dollar sign
    doc = doc.replace("**", "")  # bold becomes a tag, so "$0.02 = **$500**" pairs as "0.02 = "
    spans = []
    for block in re.split(r"\n\s*\n", doc):
        for unit in block.split("|"):
            spans += re.findall(r"\$([^$]+)\$", unit)
    return spans


def _reads_as_prose(tex: str) -> bool:
    import re

    # When two prices pair up, the closing "$" is really the next price's sign, so it follows a space.
    if tex[0].isspace() or tex[-1].isspace():
        return True
    tex = re.sub(r"\\(?:text|mathrm|operatorname|textbf|mathit)\{[^}]*\}", " ", tex)
    tex = re.sub(r"\\[A-Za-z]+", " ", tex)
    # Two ordinary words in a row don't happen in a formula.
    return re.search(r"[A-Za-z]{3,}\s+[A-Za-z]{3,}", tex) is not None


class TestDollarSigns:
    def test_given_prices_in_a_lesson_they_are_never_typeset_as_math(self):
        # MathJax reads "$2 per million ... $4" as one formula and runs the words together.
        # Write a price as \$2 so it stays a dollar sign.
        import importlib

        from primer.curriculum import CURRICULUM

        modules = ["primer", "primer.notation", *(lesson.module for lesson in CURRICULUM)]
        offenders = {}
        for module in modules:
            doc = importlib.import_module(module).__doc__ or ""
            prose = [span[:60] for span in _inline_math_spans(doc) if _reads_as_prose(span)]
            if prose:
                offenders[module] = prose
        assert offenders == {}
