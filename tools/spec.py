"""
Print the test suite as a readable specification.

    python tools/spec.py            # every module
    python tools/spec.py attention  # files whose name contains "attention"

Each test file becomes a heading, each test class a behaviour area, and each
test name a sentence. If a line teaches a reader nothing, rename the test.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def humanize_class(name: str) -> str:
    # TestCausalMasking -> "causal masking"; keep acronyms like KV or RRF intact.
    words = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|\d+", name.removeprefix("Test"))
    return " ".join(w if w.isupper() and len(w) > 1 else w.lower() for w in words)


def humanize_test(name: str) -> str:
    return name.removeprefix("test_").replace("_", " ")


def main(filter_text: str = "") -> int:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout
    last_file = last_class = None
    for line in out.splitlines():
        if "::" not in line or filter_text not in line.split("::")[0]:
            continue
        parts = line.split("::")
        path, test = parts[0], parts[-1].split("[")[0]
        klass = parts[1] if len(parts) == 3 else None
        if path != last_file:
            module = Path(path).stem.removeprefix("test_")
            print(f"\n{module}")
            last_file, last_class = path, None
        if klass and klass != last_class:
            print(f"  {humanize_class(klass)}")
            last_class = klass
        print(f"    {'  ' if klass else ''}- {humanize_test(test)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
