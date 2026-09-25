"""
# AI Primer

Two parts, each a subpackage:

* `primer.ml`: **Part 1, ML foundations: how the model works inside.** Neural nets, backprop,
  optimizers, attention, transformers, tokenization, training vs. inference,
  losses and metrics, regularization, CNNs/RNNs, and (the centerpiece)
  `primer.ml.embeddings`.
* `primer.agents`: **Part 2, Applied AI: building agents and RAG systems that people rely on.** Agent loops, tool
  calling, RAG, memory, planning, evals, guardrails, cost, observability,
  and safe deployment.

Shared toy data and a deterministic "semantic" embedder live in
`primer.common`.

Every module is runnable: `python -m primer.ml.attention` prints a narrated
walkthrough. Every module docstring ends with **Further reading** links to
the primary sources (papers, official docs).

The reading order is in `primer.curriculum`, and on the site's home page.
"""
