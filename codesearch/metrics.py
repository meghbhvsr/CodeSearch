"""Ranking metrics for one-relevant-document-per-query retrieval.

Query i's only relevant document is document i. Everything below works from the 1-based
rank of that document, so the per-query ranks can be saved and compared across models.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

# score_fn(query_ids, doc_ids) -> (len(query_ids), len(doc_ids)) scores, higher = better
ScoreFn = Callable[[np.ndarray, np.ndarray], "torch.Tensor | np.ndarray"]


def gold_ranks(
    score_fn: ScoreFn,
    n: int,
    pool_size: int | None = None,
    chunk_size: int = 1024,
    seed: int = 0,
) -> np.ndarray:
    """1-based rank of each query's gold document.

    pool_size=None: rank against the full corpus (all n documents).
    pool_size=1000: the CodeSearchNet protocol, where the shuffled set is cut into pools
      of 1000 and each query is ranked against the 999 other functions in its pool.

    Ties count against the query: rank = number of docs scoring >= the gold doc (including
    itself). Otherwise a query whose BM25 scores are all 0 would come out at rank 1.
    """
    if pool_size is None:
        pools = [np.arange(n)]
    else:
        perm = np.random.default_rng(seed).permutation(n)
        pools = [perm[i : i + pool_size] for i in range(0, n, pool_size)]

    ranks = np.empty(n, dtype=np.int64)
    for docs in pools:
        for start in range(0, len(docs), chunk_size):
            qids = docs[start : start + chunk_size]
            scores = torch.as_tensor(score_fn(qids, docs)).float()
            pos = torch.arange(start, start + len(qids), device=scores.device)
            gold = scores[torch.arange(len(qids), device=scores.device), pos]
            ranks[qids] = (scores >= gold[:, None]).sum(dim=1).cpu().numpy()
    return ranks


def summarize(ranks: np.ndarray, ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    rr = np.where(ranks <= 10, 1.0 / ranks, 0.0)
    out = {"MRR@10": float(rr.mean())}
    for k in ks:
        out[f"R@{k}"] = float((ranks <= k).mean())
    out["median_rank"] = float(np.median(ranks))
    return out
