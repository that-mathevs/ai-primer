"""
Check an In Python block: plain Python whose results are written as comments.

    x = [1, 2, 3, 4]
    # Σ: add every x_i
    sum(x)  # → 10

A line ending in `# → value` claims what that line produces: the value of an
expression, the value an assignment stores, or the text a `print` shows.
`check` runs the block top to bottom in a fresh namespace and returns one
message per claim that doesn't hold, so the page's numbers are proven by the
code that sits beside them.
"""

from __future__ import annotations

import ast
import contextlib
import doctest
import io
import re

RESULT = re.compile(r"#\s*→\s*(.*?)\s*$")


def _same(expected: str, actual: str) -> bool:
    """Equal up to whitespace; "..." in the comment stands for any text, as in doctest."""
    expected, actual = " ".join(expected.split()), " ".join(actual.split())
    return expected == actual or ("..." in expected and doctest._ellipsis_match(expected, actual))


def check(code: str) -> list[str]:
    lines = code.split("\n")
    tree = ast.parse(code)
    namespace: dict = {}
    problems: list[str] = []
    claims = 0
    for statement in tree.body:
        claim = RESULT.search(lines[statement.end_lineno - 1])
        source = ast.get_source_segment(code, statement) or ""
        printed = io.StringIO()
        try:
            with contextlib.redirect_stdout(printed):
                if isinstance(statement, ast.Expr):
                    value = eval(compile(ast.Expression(statement.value), "<in python>", "eval"), namespace)
                else:
                    exec(compile(ast.Module([statement], []), "<in python>", "exec"), namespace)
                    value = None
                    if claim and isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                        # The value the assignment stored: read the target back by its source text.
                        value = eval(ast.unparse(statement.targets[0]), namespace)
        except Exception as error:  # the claim can't hold if the line doesn't run
            problems.append(f"line {statement.lineno}: {source} raised {error!r}")
            continue
        if not claim:
            continue
        claims += 1
        actual = printed.getvalue() if printed.getvalue() else repr(value)
        if not _same(claim.group(1), actual):
            shown = " ".join(actual.split())
            problems.append(f"line {statement.lineno}: {source.split(chr(10))[-1].split('#')[0].strip()} gives {shown}, "
                            f"the comment says {claim.group(1)}")
    if not claims and not problems:
        problems.append("the block shows no result: end a line with # → value")
    return problems
