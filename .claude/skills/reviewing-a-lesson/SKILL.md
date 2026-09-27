---
name: reviewing-a-lesson
description: Hunt bugs in one lesson or paper companion (prose the code contradicts, wrong numbers, factual slips, broken ladder rungs) and fix each one red to green. Use when asked to review, audit or bug-hunt a lesson or companion, or after one changes a lot.
---

# Reviewing a lesson

You are a **skeptical reader**: every sentence is a claim until you have
checked it. The bar is the one in [`CLAUDE.md`](../../../CLAUDE.md), and
[`primer/ml/attention.py`](../../../primer/ml/attention.py) is the lesson
every other one matches. A companion (`docs/papers/<slug>.html`) is reviewed
the same way, against its paper.

## 1. Run it

```bash
python -m <module>                       # the demo narrates the lesson
python -m pytest -q tests/test_<module>.py
```

Read the demo's output as a newcomer would. Numbers it prints are the ground
truth the prose must match.

## 2. Check every claim

Read the docstring (or page) top to bottom, and put each claim into one
bucket and check it the way that bucket demands:

- **A number** (a worked example, a shape, a count, a result): recompute it,
  by running the lesson's code or by hand. Every `# →` in an In Python block
  is already checked by the tests; the prose around it is not.
- **A fact about the world** (what a paper showed, how a system works, a date,
  a price): confirm it in the primary source, fetched now. Memory is not a
  source.
- **A claim about the code** (`**In code:**` names, "this function returns",
  the defaults the prose states): open the code and confirm it.
- **A picture** (a mermaid diagram, a figure, its `**Reading it:**`): does the
  picture show what the paragraph says? Look at it: `make docs`, then
  `python tools/screenshot.py primer/<path>.html#<anchor>`.
- **Teaching** (a rung missing from the ladder, a term used before it is
  explained, a formula without its Symbols table, In words line, numbers or
  In Python block, a glossary term with no entry): note it.

**Done when** every claim in the lesson has been put in a bucket and checked,
and you have a findings list: line, what is wrong, what is right, and the
source that says so.

## 3. Fix, red to green

- **Code behaves wrongly**: write the failing scenario first
  (`test_given_..._then...` style, expected value from an independent
  source), watch it fail, then fix the code.
- **Prose is wrong**: fix the prose. If a checker could have caught this kind
  of mistake everywhere (a banned phrase, a broken link shape, a missing
  section), add the check to `tests/` or `tools/sitecheck.py` so the whole
  class stays fixed.
- **Teaching gap**: climb the missing rung, following the reference lesson.

## 4. Check the whole

```bash
make test && make docs && make sitecheck && make browsercheck
```

**Done when** all four pass and every finding is either fixed or listed with
the reason it was left.
