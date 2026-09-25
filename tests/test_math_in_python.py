"""Specification: every formula comes with its Python, and that Python runs.

Each formula is decoded symbol by symbol, worked through with small numbers,
and then written as plain Python that computes those same numbers. Every line
that produces a number ends in `# → value`, and running the block checks each
one, so the formula, the worked numbers and the code can't disagree.
"""

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


def _dedent(block: str) -> str:
    lines = block.split("\n")
    indent = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
    return "\n".join(l[indent:] for l in lines)


def _sections(doc: str) -> list[str]:
    """A docstring cut at each heading, so a formula's Python must sit in its own section.

    Only Markdown headings count: a "# comment" inside a fenced code block is code.
    """
    sections, current, in_code = [], [], False
    for line in doc.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
        if not in_code and re.match(r"#{1,6} ", line) and current:
            sections.append("\n".join(current))
            current = []
        current.append(line)
    return sections + ["\n".join(current)]


class TestCheckingAnInPythonBlock:
    # A block is plain Python; a line ending in "# → value" claims what that line produces, and the check proves it.

    def test_given_a_line_whose_result_matches_its_comment_the_block_passes(self):
        from tools.inpython import check

        assert check("x = [1, 2, 3, 4]\nsum(x)  # → 10") == []

    def test_given_a_line_whose_result_differs_from_its_comment_it_is_reported(self):
        from tools.inpython import check

        assert check("sum([1, 2, 3, 4])  # → 11") == ["line 1: sum([1, 2, 3, 4]) gives 10, the comment says 11"]

    def test_given_an_assignment_with_a_result_comment_the_assigned_value_is_checked(self):
        from tools.inpython import check

        assert check("mu = 40 / 8  # → 5.0") == []

    def test_given_a_print_its_printed_text_is_checked(self):
        from tools.inpython import check

        assert check('print("ice", 20)  # → ice 20') == []

    def test_given_a_statement_over_several_lines_the_result_comment_on_its_last_line_is_checked(self):
        from tools.inpython import check

        assert check("[i * i\n for i in range(3)]  # → [0, 1, 4]") == []

    def test_given_a_block_that_shows_no_result_it_is_reported(self):
        from tools.inpython import check

        # A block with nothing to check proves nothing about the page's numbers.
        assert check("x = 1") == ["the block shows no result: end a line with # → value"]

    def test_given_a_block_that_raises_it_is_reported(self):
        from tools.inpython import check

        assert check("1 / 0  # → 0") == ["line 1: 1 / 0 raised ZeroDivisionError('division by zero')"]


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
    def test_given_its_python_every_result_it_shows_is_what_the_code_produces(self, lesson):
        from tools.inpython import check

        doc = importlib.import_module(lesson).__doc__
        failures = [f"block {i}: {p}" for i, block in enumerate(LESSON_BLOCK.findall(doc), 1) for p in check(_dedent(block))]
        assert failures == []

    @pytest.mark.parametrize("lesson", [l.module for l in CURRICULUM])
    def test_given_its_python_it_reads_as_code_you_can_run_not_a_session_transcript(self, lesson):
        doc = importlib.import_module(lesson).__doc__
        assert [b for b in LESSON_BLOCK.findall(doc) if re.search(r"^\s*(>>>|\.\.\.) ", b, re.M)] == []


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
    def test_given_its_python_every_result_it_shows_is_what_the_code_produces(self, companion):
        from tools.inpython import check

        page = (ROOT / "docs" / "papers" / companion).read_text()
        # The page escapes < > & for HTML; the code itself is plain Python.
        blocks = [html.unescape(re.sub(r"<[^>]+>", "", b)) for b in COMPANION_BLOCK.findall(page)]
        assert [f"block {i}: {p}" for i, b in enumerate(blocks, 1) for p in check(b)] == []

    @pytest.mark.parametrize("companion", [p.name for p in COMPANIONS])
    def test_given_its_python_it_reads_as_code_you_can_run_not_a_session_transcript(self, companion):
        page = (ROOT / "docs" / "papers" / companion).read_text()
        assert [b for b in COMPANION_BLOCK.findall(page) if re.search(r"^\s*(&gt;&gt;&gt;|\.\.\.) ", b, re.M)] == []
