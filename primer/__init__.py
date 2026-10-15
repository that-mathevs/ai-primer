"""
# AI Primer

Two subpackages, six parts (the parts are in `primer.curriculum`):

* `primer.ml`: **how the model works, how it is trained and judged, how it is run** (Parts 1 to 3),
  the embeddings and search lessons (Part 4, `primer.ml.embeddings`) and the other kinds of
  model and the generators (Part 5, with `primer.ml.generative`).
* `primer.agents`: **Part 6, building systems people rely on.** Agent loops, tool
  calling, RAG, memory, planning, evals, guardrails, cost, observability,
  and safe deployment.

Shared toy data and a deterministic "semantic" embedder live in
`primer.common`.

Every module is runnable: `python -m primer.ml.attention` prints a narrated
walkthrough. Every module docstring ends with **Further reading** links to
the primary sources (papers, official docs).

The reading order is in `primer.curriculum`, and on the site's home page.
"""
