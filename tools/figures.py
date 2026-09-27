"""
Render every lesson's `figures()` into docs/figures/<dotted.module>.<key>.svg,
then check that docstrings and figures agree:

* every `![...](figures/X.svg)` in a docstring has a rendered figure, and
* every rendered figure is shown somewhere, followed by a "**Reading it:**"
  paragraph (see CLAUDE.md, "Diagrams"), and
* no line or neighbouring panel runs through a figure's words (text_collisions).

    python tools/figures.py            # all modules
    python tools/figures.py attention  # modules whose name contains "attention"

Exits non-zero when a figure is missing, unused, unexplained, or has words a line runs through.
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


def _clipped(p, q, box):
    """The part of segment p→q inside box (Liang-Barsky), or None: lines are drawn clipped to their plot."""
    (x0, y0), (x1, y1) = p, q
    dx, dy = x1 - x0, y1 - y0
    lo, hi = 0.0, 1.0
    for d, dist in ((-dx, x0 - box.x0), (dx, box.x1 - x0), (-dy, y0 - box.y0), (dy, box.y1 - y0)):
        if d == 0:
            if dist < 0:
                return None
        else:
            t = dist / d
            if d < 0:
                lo = max(lo, t)
            else:
                hi = min(hi, t)
    if lo > hi:
        return None
    return (x0 + lo * dx, y0 + lo * dy), (x0 + hi * dx, y0 + hi * dy)


def _crosses(p, q, box) -> bool:
    return _clipped(p, q, box) is not None


def _hides(text, zorder: float) -> bool:
    """Whether the label sits on an opaque box drawn above a line of this zorder, hiding the line."""
    from matplotlib.colors import to_rgba

    patch = text.get_bbox_patch()
    return patch is not None and to_rgba(patch.get_facecolor())[3] > 0.9 and text.get_zorder() > zorder


def text_collisions(fig) -> list[str]:
    """Words in the figure that a drawn line runs through, or that sit on another panel.

    Words just outside a plot own the edge they face: a line that leaves the plot under the
    title runs into the title, one that leaves through the bottom runs into the x-axis
    labels, and one that leaves through the left runs into the y-axis labels (a guide that
    merely stops at an edge leaves nothing). Inside the plot, a line runs through a label
    when it crosses it; a plotted line that ends at the label (an edge to its node) points
    at it, but an annotation arrow never gets that pass: its head covers what it lands on.
    A note is measured by its words alone (an annotation's own extent includes its arrow).
    """
    import math

    from matplotlib.text import Text
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panels = [ax for ax in fig.axes if ax.get_visible()]
    problems: list[str] = []
    for ax in panels:
        plot = ax.get_window_extent(renderer)
        titles = [t for t in (ax.title, ax._left_title, ax._right_title) if t.get_text()]
        words = [(f"axis label '{t.get_text()}'", t) for t in (ax.xaxis.label, ax.yaxis.label) if t.get_text()]
        # Only ticks inside the plot's range are drawn; matplotlib keeps labels for the rest.
        drawn = [tick.label1 for axis, (lo, hi) in ((ax.xaxis, sorted(ax.get_xbound())), (ax.yaxis, sorted(ax.get_ybound())))
                 for tick in axis.get_major_ticks() if lo - 1e-9 <= tick.get_loc() <= hi + 1e-9]
        words += [(f"tick label '{t.get_text()}'", t) for t in drawn if t.get_visible() and t.get_text()]
        words += [(f"label '{t.get_text()}'", t) for t in ax.texts if t.get_visible() and t.get_text().strip()]
        # (segment as drawn, endpoints that end a plotted line, the annotation it belongs to, its zorder)
        segments: list[tuple] = []
        exits: dict[str, list[float]] = {"top": [], "bottom": [], "left": []}
        for line in ax.lines:
            if not line.get_visible() or line.get_linestyle() in ("None", "none", "", " "):
                continue
            points = line.get_transform().transform(line.get_path().vertices)
            last = len(points) - 1
            for i, (p, q) in enumerate(zip(points[:-1], points[1:])):
                if not all(math.isfinite(v) for v in (*p, *q)):
                    continue
                seg = _clipped(p, q, plot) if line.get_clip_on() else (p, q)
                if not seg:
                    continue
                ends = [pt for pt, j in ((seg[0], i), (seg[1], i + 1)) if j in (0, last) and pt in (tuple(p), tuple(q))]
                segments.append((seg, ends, None, line.get_zorder()))
                if line.get_clip_on():
                    for pt, orig in ((seg[0], p), (seg[1], q)):
                        if orig[1] > plot.y1 + 1e-6 and abs(pt[1] - plot.y1) < 0.5:
                            exits["top"].append(pt[0])
                        elif orig[1] < plot.y0 - 1e-6 and abs(pt[1] - plot.y0) < 0.5:
                            exits["bottom"].append(pt[0])
                        elif orig[0] < plot.x0 - 1e-6 and abs(pt[0] - plot.x0) < 0.5:
                            exits["left"].append(pt[1])
        for note in ax.texts:
            arrow = getattr(note, "arrow_patch", None)
            if note.get_visible() and arrow is not None:
                points = arrow.get_transform().transform(arrow.get_path().vertices)
                segments += [((p, q), [], note, arrow.get_zorder()) for p, q in zip(points[:-1], points[1:])]
        for title in titles:
            box = title.get_window_extent(renderer)
            if any(box.x0 <= x <= box.x1 for x in exits["top"]) or any(
                _crosses(*seg, box.shrunk(0.98, 0.9)) for seg, _, _, _ in segments
            ):
                problems.append(f"a line runs into the title '{title.get_text()}'")
        if exits["bottom"] and (ax.xaxis.label.get_text() or any(t.get_text() for t in ax.get_xticklabels())):
            problems.append("a line runs off the bottom of the plot into the x-axis labels")
        if exits["left"] and any(t.get_text() for t in ax.get_yticklabels()):
            problems.append("a line runs off the left of the plot into the y-axis labels")
        for name, text in words:
            # Text's own extent: an annotation's get_window_extent also spans its arrow.
            box = Text.get_window_extent(text, renderer).shrunk(0.98, 0.9)
            if any(_crosses(*seg, box) and not any(box.contains(*pt) for pt in ends)
                   and not _hides(text, zorder) for seg, ends, owner, zorder in segments):
                problems.append(f"a line runs into the {name}")
            for other in panels:
                if other is ax:
                    continue
                overlap = Bbox.intersection(box, other.get_window_extent(renderer))
                if overlap is not None and overlap.width > 2 and overlap.height > 2:
                    problems.append(f"the {name} sits on another panel")
                    break
    return sorted(set(problems))


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
                produced.add(stem)
                rendered += 1
            took = time.perf_counter() - started
            print(f"  {took:6.1f}s  {name}", flush=True)
            if took > SLOW_SECONDS:
                problems.append(f"{name}.figures() took {took:.0f}s; keep each lesson's figures under {SLOW_SECONDS}s")
            # Checked outside the timing: words a line or another panel runs through can't be read.
            for key, fig in figs.items():
                problems += [f"{name}.{key}: {p}" for p in text_collisions(fig)]
                plt.close(fig)
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
