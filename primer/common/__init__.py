"""
Shared building blocks used by both parts.

* `primer.common.text`: a tiny tokenizer (lowercase words, stopwords removed).
* `primer.common.embedder`: `ConceptEmbedder`, a deterministic stand-in for a
  real embedding model. It knows a small synonym lexicon, so "car" and
  "automobile" land close together (like a dense model) while exact IDs such
  as "ERR-4012" only match themselves (the classic weakness BM25 fixes).
* `primer.common.corpus`: a small enterprise knowledge base (IT, HR, finance)
  with metadata, access-control groups, and labeled queries for retrieval
  evaluation.

Why a fake embedder? So every example runs offline, instantly, and
deterministically, and so tests can assert exact rankings. The *mechanics*
(normalize, dot product, index, rank) are identical to production. To try a
real model, swap in e.g. `sentence-transformers`:
https://www.sbert.net/docs/quickstart.html
"""
