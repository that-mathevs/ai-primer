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


# Coaching language: this is a primer, not a guide to performing answers. Built so this file doesn't trip itself.
COACHING = [
    "strong " + "answer", "points to " + "hit", "spine of " + "the answer",
    "spine of a", "practise answering", "practice answering", "answering each question out " + "loud",
    "explaining a topic out " + "loud", "walk me " + "through", "walk " + "through what", "walk " + "through exactly",
    "walk " + "through bpe", "walk " + "through an", "argue both " + "sides", "what would you " + "say",
    "how would you " + "answer", "say " + "this", "talking " + "points", "in an " + "interview", "interview" + "er",
    "what do you " + "do?",
]
READER_FACING = ("primer/", "docs/", "README.md")


class TestTone:
    def test_given_anything_a_reader_sees_it_teaches_rather_than_coaches_answers(self):
        offenders = {}
        for rel, path in source_files():
            if not rel.startswith(READER_FACING) or rel.endswith((".js", ".css")):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            found = [phrase for phrase in COACHING if phrase in text]
            if found:
                offenders[rel] = found
        assert offenders == {}, f"coaching language found: {offenders}"
