"""
# Embeddings, the centerpiece

An embedding is a learned mapping from an input (a word, sentence, image or
piece of code) to a list of numbers, a vector, arranged so that **closeness in
space means similarity in meaning**. Search, retrieval-augmented generation,
routing, deduplication, clustering and semantic caching all run on them.
"""

from primer.curriculum import reading_list as _reading_list

__doc__ += _reading_list("primer.ml.embeddings")
