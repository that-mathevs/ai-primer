"""Specification: interactive visualizations in the lessons.

A lesson places <div class="viz" data-viz="NAME"> in its docstring; the site
loads docs/assets/viz/NAME.js and the numbers the lesson exports from
viz_data(). A widget is only trustworthy if it agrees with the lesson's own
Python, so where a widget computes something, these specs run its JavaScript
and compare the result with the Python function it mirrors.
"""

import importlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from primer.curriculum import CURRICULUM

ROOT = Path(__file__).resolve().parent.parent
VIZ = ROOT / "docs" / "assets" / "viz"
NODE = shutil.which("node")
PLACEHOLDER = re.compile(r'<div class="viz" data-viz="([a-z0-9-]+)"[^>]*aria-label="[^"]+"')


def _lesson_vizzes() -> dict[str, list[str]]:
    found = {}
    for lesson in CURRICULUM:
        names = PLACEHOLDER.findall(importlib.import_module(lesson.module).__doc__ or "")
        if names:
            found[lesson.module] = names
    return found


def _node(script: str) -> str:
    return subprocess.run([NODE, "-e", script], capture_output=True, text=True, check=True, cwd=ROOT).stdout


class TestEveryVisualizationIsWired:
    def test_given_a_placeholder_in_a_lesson_its_script_exists(self):
        missing = [name for names in _lesson_vizzes().values() for name in names if not (VIZ / f"{name}.js").exists()]
        assert missing == []

    def test_given_any_placeholder_it_names_the_widget_for_screen_readers(self):
        # A widget with no accessible name is announced as an unlabelled group.
        for lesson in CURRICULUM:
            doc = importlib.import_module(lesson.module).__doc__ or ""
            unnamed = [m for m in re.findall(r'<div class="viz"[^>]*>', doc) if "aria-label=" not in m]
            assert unnamed == [], lesson.module

    @pytest.mark.skipif(NODE is None, reason="needs Node to parse JavaScript")
    def test_given_every_visualization_script_it_is_valid_javascript(self):
        broken = []
        for script in sorted(VIZ.glob("*.js")):
            result = subprocess.run([NODE, "--check", str(script)], capture_output=True, text=True)
            if result.returncode:
                broken.append(f"{script.name}: {result.stderr.strip().splitlines()[-1]}")
        assert broken == []

    def test_given_a_lesson_that_exports_visualization_data_it_is_json_for_its_own_widgets(self):
        for module, names in _lesson_vizzes().items():
            lesson = importlib.import_module(module)
            if hasattr(lesson, "viz_data"):
                data = lesson.viz_data()
                json.dumps(data)  # raises if anything isn't plain JSON
                assert set(data) <= set(names), module


class TestTheSiteLoadsVisualizations:
    def test_given_a_page_with_a_widget_it_loads_the_styles_data_helpers_and_that_widget(self):
        from tools.docsite import add_viz_assets

        page = '<html><head></head><body><div class="viz" data-viz="kv-cache" aria-label="x"></div></body></html>'
        html = add_viz_assets(page, "primer/ml/inference.html")
        assert '<link rel="stylesheet" href="../../assets/viz/viz.css">' in html
        order = [html.index(f'src="../../assets/viz/{f}"') for f in ("data.js", "viz.js", "kv-cache.js")]
        # The data and the helpers must load before the widget that uses them.
        assert order == sorted(order)

    def test_given_a_page_without_a_widget_nothing_is_added(self):
        from tools.docsite import add_viz_assets

        page = "<html><head></head><body><p>no widgets</p></body></html>"
        assert add_viz_assets(page, "primer/ml/attention.html") == page


@pytest.mark.skipif(NODE is None, reason="needs Node to run the widget's arithmetic")
class TestTheKVCacheWidgetAgreesWithTheLesson:
    def test_given_model_shapes_and_contexts_the_widget_computes_the_lessons_kv_cache_bytes(self):
        from primer.ml.inference import kv_cache_bytes

        cases = [(8192, 32, 8, 128, 16, 1), (131072, 32, 8, 128, 16, 1), (1048576, 80, 8, 128, 8, 4), (4096, 32, 32, 128, 4, 16)]
        js = _node(
            "const m = require('./docs/assets/viz/kv-cache.js');"
            f"console.log(JSON.stringify({json.dumps(cases)}.map(c => m.kvCacheBytes(...c))));"
        )
        assert json.loads(js) == [kv_cache_bytes(*c) for c in cases]

    def test_given_the_inference_lesson_it_places_the_kv_cache_widget(self):
        doc = importlib.import_module("primer.ml.inference").__doc__
        assert "kv-cache" in PLACEHOLDER.findall(doc)
