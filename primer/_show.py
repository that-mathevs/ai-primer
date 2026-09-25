"""
Tiny printing helpers used by every module's `demo()`.

They keep the walkthrough output readable in a terminal: a banner per
section, aligned tables, and short "key takeaway" callouts. None
of this is part of the material; it just keeps the demos tidy.
"""

from __future__ import annotations

import textwrap
from typing import Iterable, Sequence

import numpy as np

WIDTH = 78


def banner(title: str) -> None:
    """Print a section banner, e.g. `== 1. Scaled dot-product attention ==`."""
    print()
    print("=" * WIDTH)
    print(f" {title}")
    print("=" * WIDTH)


def say(text: str) -> None:
    """Print a paragraph of narration, wrapped to the terminal width."""
    for para in textwrap.dedent(text).strip().split("\n\n"):
        print(textwrap.fill(" ".join(para.split()), WIDTH))
        print()


def takeaway(text: str) -> None:
    """Print the one-sentence takeaway of a section."""
    body = textwrap.fill(" ".join(text.split()), WIDTH - 4)
    print("  >> " + body.replace("\n", "\n     "))
    print()


def table(headers: Sequence[str], rows: Iterable[Sequence[object]], floatfmt: str = ".4f") -> None:
    """Print rows as an aligned plain-text table. Floats use `floatfmt`."""

    def fmt(v: object) -> str:
        if isinstance(v, (float, np.floating)):
            return format(float(v), floatfmt)
        return str(v)

    str_rows = [[fmt(v) for v in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        widths = [max(w, len(c)) for w, c in zip(widths, row)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in str_rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))
    print()


def matrix(name: str, m: np.ndarray, precision: int = 3) -> None:
    """Print a small matrix with a label."""
    with np.printoptions(precision=precision, suppress=True, linewidth=WIDTH):
        print(f"{name} (shape {m.shape}):")
        print(m)
        print()
