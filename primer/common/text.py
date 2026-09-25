"""
Minimal text normalization shared by BM25, the toy embedder, and RAG.

Real systems use a proper analyzer (Lucene/Elasticsearch: lowercasing,
stemming, stopwords) for keyword search, and a subword tokenizer (BPE, see
`primer.ml.tokenization`) for neural models. This is intentionally simple
so the retrieval code stays readable.

Further reading:
- Elasticsearch analyzers: https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis.html
"""

from __future__ import annotations

import re

# Small English stopword list. Stopwords carry little meaning for retrieval,
# and dropping them keeps BM25 and the toy embedder focused on content words.
STOPWORDS = frozenset(
    """
    a an and are as at be by can do does for from has have how i if in is it
    its me my of on or our so that the their them this to was we what when
    where which who why will with you your
    """.split()
)

# Keep tokens like "err-4012", "q3", "vpn", "2fa": letters/digits joined by
# single hyphens. Everything else (punctuation, whitespace) is a separator.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    """Lowercase and split `text` into word tokens.

    >>> tokenize("How do I reset my password? Error ERR-4012!")
    ['reset', 'password', 'error', 'err-4012']
    """
    toks = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks
