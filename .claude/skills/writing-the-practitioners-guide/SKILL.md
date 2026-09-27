---
name: writing-the-practitioners-guide
description: Give a lesson its Level 1, the practitioner's guide, and mark where Level 2 begins. Use when a lesson is in LEVELS_PENDING in primer/curriculum.py, or when an existing guide needs revising.
---

# Writing the practitioner's guide

The guide is the level most readers stop at: the level of a good professional
book on building with AI, not of a paper. A reader who finishes it can **use
the idea well and choose between the options**, having seen no formula and no
code. The rules for it are in [`CLAUDE.md`](../../../CLAUDE.md) ("Start at
the top"); the reference guide is the one in
[`primer/ml/structured_output.py`](../../../primer/ml/structured_output.py).
Match its level, its shape and its length.

## 1. Read the lesson as a practitioner

Read the whole docstring and run `python -m <module>`. As you read, collect
the answers a practitioner needs, with the number or code that backs each:

- what the thing is, in one plain sentence;
- the situations that call for it, and the ones that don't;
- every approach or option the lesson covers or that practice offers, with
  what each guarantees, what it costs (money, latency, effort, quality) and
  where it runs (your prompt, your code, the model server, training);
- how the choice is made in practice;
- the failure modes, each as a thing that goes wrong for a real system;
- the real tools, APIs, systems and papers that do this, from the lesson's
  Further reading and papers section.

**Done when** you have a fact list where every entry points at a line of the
lesson, its code, or a primary source. A claim with no source is not written.

## 2. Write Level 1

Insert `## Level 1: The practitioner's guide` as the **first `##` section** of
the docstring, right after the title, the `Run:` line and the line on what the
lesson builds on. Write these labels, in this order, each opening its own
paragraph:

1. `**In one sentence.**` What it is and what it is for.
2. `**When you need it.**` The situations, and the ones that don't need it.
   Give the reader a tell they will recognise in their own work. If the
   lesson has a number that shows how badly the naive approach fails, use it.
3. `**Your options.**` A markdown table of the approaches, cheapest to most
   certain: option, what it does, what it guarantees, what it costs, where it
   lives. Every row must be a real choice a practitioner could make.
4. `**How to choose.**` Decision guidance, by situation, as a short list.
   End with the rule that holds whatever they pick.
5. `**What it costs.**` Money, latency, quality, effort, in that spirit; with
   numbers where the lesson has them.
6. `**What breaks.**` A bold-led list of failure modes, each with what to do.
7. `**In the wild.**` Named tools, APIs, systems and papers. Naming a product
   is fine here; the idea stays vendor-neutral, the examples are real.
8. `**Go deeper.**` Two or three sentences on what Level 2 will build, ending
   with permission to stop.

The bar, checked by [`tests/test_navigation.py`](../../../tests/test_navigation.py):
600 to 1,600 words; no `$$`, no Symbols table, no In Python block, no Python
code block; at most one mermaid diagram (a decision flow or the landscape, if
a picture beats the table). House style: teacher's voice, no em dashes,
prices as `\$2`, terms explained on first use. Every number comes from the
lesson's own code or a primary source, and says which.

## 3. Mark Level 2

Right after the guide, insert `## Level 2: How it works, from scratch`,
followed by the lesson's opening explanation of the idea (the paragraphs that
used to sit under `## The idea` or before the first numbered section). Then
the existing sections follow unchanged. Remove the module from
`LEVELS_PENDING` in [`primer/curriculum.py`](../../../primer/curriculum.py).

## 4. Check it

```bash
python -m pytest -q tests/test_navigation.py tests/test_house_style.py tests/test_<module>.py
make docs && python tools/screenshot.py primer/<path>.html?depth=1 --theme light
```

Read the screenshot as the reader who will stop there. Then `make gate`.

**Done when** the tests pass, the guide reads as advice a practitioner could
act on tomorrow, and nothing in it needs a formula to understand.
