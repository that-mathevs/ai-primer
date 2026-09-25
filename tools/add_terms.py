"""
Merge `term | definition | dotted.module` lines into primer/glossary.py,
skipping terms it already defines. Handy when a lesson introduces several terms.

    python tools/add_terms.py new_terms.txt
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from primer.glossary import GLOSSARY  # noqa: E402

lines, skipped = [], []
for row in Path(sys.argv[1]).read_text().strip().splitlines():
    term, definition, module = (x.strip() for x in row.split("|"))
    term = term.lower()
    if term in GLOSSARY:
        skipped.append(term)
        continue
    lines.append(f"    {term!r}: _E({definition!r}, {module!r}),")
p = ROOT / "primer" / "glossary.py"
s = p.read_text()
anchor = "}\n\n\ndef lesson_path"
p.write_text(s.replace(anchor, "\n".join(lines) + "\n" + anchor, 1))
print(f"{len(lines)} added; already defined: {', '.join(skipped) or 'none'}")
