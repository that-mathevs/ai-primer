"""Specification: every formula comes with its Python, and that Python runs.

Each formula is decoded symbol by symbol, worked through with small numbers,
and then written as plain Python that computes those same numbers. The Python
is a `>>>` example, so running it proves the formula, the worked numbers and
the code agree.
"""

import doctest
import html
import importlib
import re
from pathlib import Path

import pytest

from primer.curriculum import CURRICULUM

ROOT = Path(__file__).resolve().parent.parent
COMPANIONS = sorted((ROOT / "docs" / "papers").glob("*.html"))

# **In Python:** followed by a fenced python block holding >>> examples.
LESSON_BLOCK = re.compile(r"\*\*In Python:\*\*\s*\n\s*```python\n(.*?)\n\s*```", re.S)
# In a companion: <pre class="in-python"><code>...</code></pre>.
COMPANION_BLOCK = re.compile(r'<pre class="in-python"><code>(.*?)</code></pre>', re.S)


def _run(examples: str, name: str) -> list[str]:
    """Run a block of >>> examples; return a description of each one whose output differs."""
    test = doctest.DocTestParser().get_doctest(examples, {}, name, name, 0)
    assert test.examples, f"{name}: the In Python block has no >>> examples"
    failures: list[str] = []

    class Collect(doctest.DocTestRunner):
        def report_failure(self, out, test, example, got):
            failures.append(f"{example.source.strip()} -> expected {example.want.strip()!r}, got {got.strip()!r}")

        def report_unexpected_exception(self, out, test, example, exc_info):
            failures.append(f"{example.source.strip()} -> raised {exc_info[1]!r}")

    Collect(optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE).run(test, out=lambda _: None)
    return failures


def _dedent(block: str) -> str:
    lines = block.split("\n")
    indent = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
    return "\n".join(l[indent:] for l in lines)


def _sections(doc: str) -> list[str]:
    """A docstring cut at each heading, so a formula's Python must sit in its own section."""
    return re.split(r"\n(?=#{1,6} )", doc)


class TestEveryFormulaInALessonHasItsPython:
    @pytest.mark.parametrize("lesson", [l.module for l in CURRICULUM])
    def test_given_a_formula_the_same_section_shows_it_in_python_after_it(self, lesson):
        doc = importlib.import_module(lesson).__doc__
        missing = []
        for section in _sections(doc):
            for formula in re.finditer(r"\$\$(.+?)\$\$", section, re.S):
                # Formulas worked through together may share one block, after the last of them.
                if not LESSON_BLOCK.search(section, formula.end()):
                    missing.append(" ".join(formula.group(1).split())[:60])
        assert missing == []

    @pytest.mark.parametrize("lesson", [l.module for l in CURRICULUM])
    def test_given_its_python_it_runs_and_prints_exactly_what_the_lesson_shows(self, lesson):
        doc = importlib.import_module(lesson).__doc__
        failures = []
        for i, block in enumerate(LESSON_BLOCK.findall(doc), 1):
            failures += [f"block {i}: {f}" for f in _run(_dedent(block), f"{lesson} block {i}")]
        assert failures == []


class TestEveryEquationInACompanionHasItsPython:
    @pytest.mark.parametrize("companion", [p.name for p in COMPANIONS])
    def test_given_an_equation_the_same_section_shows_it_in_python_after_it(self, companion):
        page = (ROOT / "docs" / "papers" / companion).read_text()
        missing = []
        for section in re.split(r"(?=<h2\b)", page):
            for eq in re.finditer(r'<div class="eq"', section):
                if not COMPANION_BLOCK.search(section, eq.end()):
                    missing.append(re.sub(r"<[^>]+>|\s+", " ", section[eq.start(): eq.start() + 200]).strip()[:60])
        assert missing == []

    @pytest.mark.parametrize("companion", [p.name for p in COMPANIONS])
    def test_given_its_python_it_runs_and_prints_exactly_what_the_page_shows(self, companion):
        page = (ROOT / "docs" / "papers" / companion).read_text()
        failures = []
        for i, block in enumerate(COMPANION_BLOCK.findall(page), 1):
            # The page escapes < > & for HTML; the code itself is plain Python.
            code = html.unescape(re.sub(r"<[^>]+>", "", block))
            failures += [f"block {i}: {f}" for f in _run(code, f"{companion} block {i}")]
        assert failures == []
