"""
`ConceptEmbedder`: a deterministic, dependency-free stand-in for a real
sentence-embedding model.

How it works (and why it's a fair teaching model):

1. Tokenize the text (`primer.common.text.tokenize`).
2. For each token, look it up in a tiny synonym lexicon (`CONCEPTS`). If the
   word belongs to a concept (e.g. "car" and "automobile" both belong to
   `vehicle`), add that concept's fixed random vector. Also add a smaller
   word-specific vector so synonyms are close but not identical.
3. Words outside the lexicon (including IDs like "err-4012") get only their
   own random vector. They match themselves and nothing else.
4. Mean-pool (sum) the token vectors and L2-normalize, exactly like a real
   bi-encoder does with mean pooling.

That reproduces the two behaviors that matter most in practice:

* **Dense retrieval finds meaning.** "automobile reimbursement" is close to
  "car expense" even with zero shared words.
* **Dense retrieval blurs exact tokens.** An error code is just one of many
  pooled token vectors, so it's a weak signal next to the concept words.
  That's why hybrid search (BM25 + dense) wins on enterprise data. See
  `primer.ml.embeddings.retrieval`.

Random high-dimensional vectors are nearly orthogonal (their cosine is
about 0 plus or minus 1/sqrt(dim)), which is what makes "unrelated" words
unrelated here.

Further reading:
- Sentence-BERT (mean pooling, bi-encoders): https://arxiv.org/abs/1908.10084
- sentence-transformers docs: https://www.sbert.net/
- Random projections / near-orthogonality: https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np

from primer.common.text import tokenize

# concept name -> words that express it. Keep it small and obvious.
CONCEPTS: dict[str, list[str]] = {
    "credential": ["password", "passwords", "passcode", "credentials", "login", "sign-in", "signin", "logon"],
    "reset": ["reset", "recover", "recovery", "forgot", "forgotten", "unlock", "locked", "lockout"],
    "mfa": ["2fa", "mfa", "authenticator", "two-factor", "otp", "verification"],
    "remote_access": ["vpn", "remote", "tunnel", "anyconnect"],
    "vehicle": ["car", "automobile", "vehicle", "auto", "mileage", "driving"],
    "money_back": ["reimburse", "reimbursement", "reimbursed", "expense", "expenses", "refund", "receipt", "receipts"],
    "time_off": ["pto", "vacation", "leave", "holiday", "holidays", "time-off", "sick"],
    "computer": ["laptop", "computer", "notebook", "macbook", "device", "hardware", "monitor"],
    "invoice": ["invoice", "invoices", "bill", "billing", "payment", "payments", "vendor", "vendors"],
    "travel": ["travel", "trip", "flight", "flights", "hotel", "airfare", "per-diem"],
    "error": ["error", "errors", "failure", "fail", "failed", "fails", "crash", "broken", "issue"],
    "email": ["email", "emails", "mail", "inbox", "outlook"],
    "threat": ["phishing", "suspicious", "scam", "malware", "attack"],
    "policy": ["policy", "policies", "rule", "rules", "requirement", "requirements", "guideline", "guidelines"],
    "new_hire": ["onboarding", "onboard", "hire", "hires", "orientation"],
    "reconcile": ["reconcile", "reconciliation", "match", "matching", "mismatch", "mismatches"],
    "salary": ["salary", "pay", "payroll", "paycheck", "compensation", "bonus"],
    "printer": ["printer", "printers", "print", "printing", "toner"],
}

# Reverse index: word -> concept.
WORD_TO_CONCEPT: dict[str, str] = {w: c for c, ws in CONCEPTS.items() for w in ws}


def _seed(key: str) -> int:
    # Python's built-in hash() is salted per process; md5 is stable across runs.
    return int(hashlib.md5(key.encode()).hexdigest()[:16], 16)


@lru_cache(maxsize=None)
def _unit_vector(key: str, dim: int) -> np.ndarray:
    """A fixed pseudo-random unit vector for `key`. Same key -> same vector."""
    v = np.random.default_rng(_seed(key)).standard_normal(dim)
    v /= np.linalg.norm(v)
    v.setflags(write=False)  # cached arrays are shared; make them read-only
    return v


class ConceptEmbedder:
    """Deterministic toy bi-encoder. `encode(texts)` returns L2-normalized rows.

    Args:
        dim: output dimensionality (real models use 384 to 3072).
        word_weight: how much each word's own identity counts relative to its
            concept. Lower means synonyms are closer together.
        name: a "model version" string. Two embedders with different names
            produce incompatible vector spaces, which lets
            `primer.ml.embeddings.operations` demonstrate why switching
            models means re-embedding everything.
    """

    def __init__(self, dim: int = 64, word_weight: float = 0.35, name: str = "concept-v1"):
        self.dim = dim
        self.word_weight = word_weight
        self.name = name

    def _token_vector(self, tok: str) -> np.ndarray:
        concept = WORD_TO_CONCEPT.get(tok)
        word_vec = _unit_vector(f"{self.name}/word/{tok}", self.dim)
        if concept is None:
            return word_vec
        concept_vec = _unit_vector(f"{self.name}/concept/{concept}", self.dim)
        return concept_vec + self.word_weight * word_vec

    def encode_one(self, text: str) -> np.ndarray:
        toks = tokenize(text)
        if not toks:
            return np.zeros(self.dim)
        v = np.sum([self._token_vector(t) for t in toks], axis=0)
        return v / np.linalg.norm(v)

    def encode(self, texts: str | Sequence[str] | Iterable[str]) -> np.ndarray:
        """Embed one string (returns shape `(dim,)`) or many (returns `(n, dim)`)."""
        if isinstance(texts, str):
            return self.encode_one(texts)
        return np.stack([self.encode_one(t) for t in texts])
