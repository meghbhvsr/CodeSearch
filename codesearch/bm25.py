"""BM25 baseline with a code-aware tokenizer, implemented on sparse matrices.

rank_bm25 scores one query at a time with a Python loop over every document. On the
~22k-function test set that means roughly 22k x 22k Python-level operations per query term,
which takes hours. Precomputing the BM25 weight of every (doc, term) once turns scoring
into one sparse matrix product per batch of queries, and it finishes in seconds. The
scoring formula is standard Okapi BM25 with Lucene's (non-negative) IDF.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import scipy.sparse as sp

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

STOPWORDS = frozenset(
    "a an and are as at be by for from has have if in into is it its of on or that the "
    "this to was were will with which when where whether given".split()
)


def tokenize(text: str) -> list[str]:
    """Split identifiers on snake_case/camelCase, lowercase, drop stopwords.

    `getHTTPResponse_code` -> ["get", "http", "response", "code", "gethttpresponse_code"].
    The whole identifier is kept too, so exact-name queries still match strongly.
    """
    out = []
    for ident in _IDENT.findall(text):
        parts = [p.lower() for piece in ident.split("_") for p in _CAMEL.findall(piece)]
        out.extend(parts)
        if len(parts) > 1:
            out.append(ident.lower())
    return [t for t in out if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        for d, text in enumerate(docs):
            for term, tf in Counter(tokenize(text)).items():
                rows.append(d)
                cols.append(self.vocab.setdefault(term, len(self.vocab)))
                vals.append(tf)
        n_docs = len(docs)
        tf = sp.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(n_docs, len(self.vocab)))

        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        df = np.bincount(tf.indices, minlength=len(self.vocab))
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        # w(d, t) = idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
        norm = k1 * (1 - b + b * doc_len / max(doc_len.mean(), 1e-9))
        row_norm = np.repeat(norm, np.diff(tf.indptr))
        w = tf.copy()
        w.data = self.idf[w.indices] * w.data * (k1 + 1) / (w.data + row_norm)
        self.weights_t = w.T.tocsr()  # (vocab, docs)

    def encode_queries(self, queries: list[str]) -> sp.csr_matrix:
        """Bag-of-terms count vectors; terms absent from the corpus are dropped."""
        rows, cols, vals = [], [], []
        for q, text in enumerate(queries):
            for term, c in Counter(tokenize(text)).items():
                if term in self.vocab:
                    rows.append(q)
                    cols.append(self.vocab[term])
                    vals.append(c)
        return sp.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(len(queries), len(self.vocab)))

    def score(self, query_vecs: sp.csr_matrix, doc_ids: np.ndarray | None = None) -> np.ndarray:
        """Dense (n_queries, n_docs) BM25 scores, optionally restricted to doc_ids."""
        w = self.weights_t if doc_ids is None else self.weights_t[:, doc_ids]
        return (query_vecs @ w).toarray()
