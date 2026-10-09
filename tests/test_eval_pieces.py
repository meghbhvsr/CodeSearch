import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codesearch.bm25 import BM25, tokenize  # noqa: E402
from codesearch.data import docstring_to_query, strip_docstring  # noqa: E402
from codesearch.metrics import gold_ranks, summarize  # noqa: E402


# ---------------------------------------------------------------- data


def test_strip_docstring_method():
    code = 'def f(self, x):\n        """Add one.\n\n        :param x: n\n        """\n        return x + 1'
    out = strip_docstring(code)
    assert "Add one" not in out and "return x + 1" in out


def test_strip_docstring_python2_fallback():
    code = "def f(x):\n    '''Print it.'''\n    print x\n"
    out = strip_docstring(code)
    assert "Print it" not in out and "print x" in out


def test_strip_docstring_leaves_other_strings():
    code = 'def f():\n    x = 1\n    return """not a docstring"""\n'
    assert strip_docstring(code) == code


def test_docstring_to_query_first_paragraph():
    doc = "Fetch a URL,\n    retrying on failure.\n\n    Args:\n        url: where"
    assert docstring_to_query(doc) == "Fetch a URL, retrying on failure."
    assert docstring_to_query("Do it.\n:param a: thing") == "Do it."


# ---------------------------------------------------------------- bm25


def test_tokenize_splits_identifiers():
    toks = tokenize("def getHTTPResponse_code(self):")
    assert {"get", "http", "response", "code", "gethttpresponse_code", "def", "self"} <= set(toks)


def test_bm25_matches_brute_force():
    docs = ["retry failed http request", "parse json file", "http client retry retry backoff", "write file"]
    bm = BM25(docs, k1=1.2, b=0.75)
    scores = bm.score(bm.encode_queries(["retry http"]))[0]

    toks = [tokenize(d) for d in docs]
    avgdl = sum(map(len, toks)) / len(toks)
    expected = []
    for t in toks:
        tf, s = Counter(t), 0.0
        for q in ["retry", "http"]:
            df = sum(q in d for d in toks)
            idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
            s += idf * tf[q] * 2.2 / (tf[q] + 1.2 * (0.25 + 0.75 * len(t) / avgdl))
        expected.append(s)
    np.testing.assert_allclose(scores, expected, rtol=1e-5)


# ---------------------------------------------------------------- metrics


def test_gold_ranks_and_ties():
    # rows = queries, cols = docs; gold is the diagonal
    s = np.array([[0.9, 0.1, 0.2], [0.8, 0.5, 0.1], [0.0, 0.0, 0.0]])
    ranks = gold_ranks(lambda q, d: s[np.ix_(q, d)], 3, chunk_size=2)
    assert ranks.tolist() == [1, 2, 3]  # all-zero row is a 3-way tie -> worst rank
    m = summarize(ranks)
    assert math.isclose(m["MRR@10"], (1 + 1 / 2 + 1 / 3) / 3)
    assert m["R@1"] == 1 / 3


def test_gold_ranks_pools_stay_within_pool():
    n = 10
    s = np.eye(n)  # perfect retriever
    ranks = gold_ranks(lambda q, d: s[np.ix_(q, d)], n, pool_size=4)
    assert (ranks == 1).all()
