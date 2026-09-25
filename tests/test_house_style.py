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
