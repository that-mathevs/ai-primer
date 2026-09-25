"""
Render every lesson's `figures()` into docs/figures/<dotted.module>.<key>.svg,
then check that docstrings and figures agree:

* every `![...](figures/X.svg)` in a docstring has a rendered figure, and
* every rendered figure is shown somewhere, followed by a "**Reading it:**"
  paragraph (see CLAUDE.md, "Diagrams").

    python tools/figures.py            # all modules
    python tools/figures.py attention  # modules whose name contains "attention"

Exits non-zero when a figure is missing, unused, or unexplained.
"""

from __future__ import annotations

import os

# Single-threaded BLAS: much faster on the lessons' tiny matrices (set before NumPy loads).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import importlib
import pkgutil
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# One house style for every figure, so the site reads as one book.
plt.rcParams.update(
    {
        "figure.dpi": 110,
        # Fixed ids instead of random ones, so re-rendering an unchanged figure leaves its file unchanged.
        "svg.hashsalt": "primer",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.prop_cycle": matplotlib.cycler(
            color=["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#4b5563"]
        ),
        "svg.fonttype": "none",  # keep text as text: selectable, searchable, small
    }
)

SLOW_SECONDS = 20  # a lesson's figures should render fast enough to rebuild the site casually

IMG_RE = re.compile(r"!\[[^\]]*\]\(figures/([\w.]+)\.svg\)")


def lesson_modules(filter_text: str = ""):
    import primer

    for info in pkgutil.walk_packages(primer.__path__, "primer."):
        name = info.name
        if filter_text in name and not name.rsplit(".", 1)[-1].startswith("_"):
            yield name


def main(filter_text: str = "") -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    rendered = 0
    for name in lesson_modules(filter_text):
        mod = importlib.import_module(name)
        doc = mod.__doc__ or ""
        referenced = set(IMG_RE.findall(doc))
        produced: set[str] = set()
        if hasattr(mod, "figures"):
            started = time.perf_counter()
            try:
                figs = mod.figures()
            except Exception as exc:  # report every broken module, not just the first
                problems.append(f"{name}.figures() raised {exc!r}")
                continue
            for key, fig in figs.items():
                stem = f"{name}.{key}"
                fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", metadata={"Date": None})
                plt.close(fig)
                produced.add(stem)
                rendered += 1
            took = time.perf_counter() - started
            print(f"  {took:6.1f}s  {name}", flush=True)
            if took > SLOW_SECONDS:
                problems.append(f"{name}.figures() took {took:.0f}s; keep each lesson's figures under {SLOW_SECONDS}s")
        for stem in sorted(referenced - produced):
            problems.append(f"{name}: docstring shows figures/{stem}.svg but figures() doesn't produce it")
        for stem in sorted(produced - referenced):
            problems.append(f"{name}: figure '{stem}' is rendered but never shown in the docstring")
        for m in IMG_RE.finditer(doc):
            following = doc[m.end() : m.end() + 400].lstrip()
            if not following.startswith("**Reading it:**"):
                problems.append(f"{name}: figure '{m.group(1)}' is not followed by a **Reading it:** paragraph")
        for m in re.finditer(r"```mermaid.*?```", doc, re.S):
            if not doc[m.end() : m.end() + 400].lstrip().startswith("**Reading it:**"):
                first = m.group(0).splitlines()[1].strip() if "\n" in m.group(0) else ""
                problems.append(f"{name}: mermaid diagram ({first}...) is not followed by a **Reading it:** paragraph")
    print(f"rendered {rendered} figures into {OUT.relative_to(ROOT)}/")
    for p in problems:
        print("  ✗", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
