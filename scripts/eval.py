"""Evaluate a retriever on CodeSearchNet Python (docstring -> function).

Examples:
  python scripts/eval.py bm25
  python scripts/eval.py dense                      # zero-shot models/unixcoder-base
  python scripts/eval.py dense --model checkpoints/unixcoder-mnrl --name finetuned
  python scripts/eval.py dense --protocol csn1000
  python scripts/eval.py bm25 --limit 2000          # quick smoke test

Each run writes results/<name>__<split>__<protocol>.json (metrics, config) and a matching
.ranks.npy (per-query gold rank, aligned with data/csn_python_<split>.jsonl), which you
can use later to find the queries where one model beats another.
Run scripts/report.py to turn all saved runs into a markdown table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("USE_TF", "0")  # a broken TensorFlow install otherwise breaks transformers imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from codesearch.data import load_pairs  # noqa: E402
from codesearch.metrics import gold_ranks, summarize  # noqa: E402

RESULTS_DIR = ROOT / "results"
PROTOCOLS = {"full": None, "csn1000": 1000}


def build_bm25(queries, codes, args):
    from codesearch.bm25 import BM25

    index = BM25(codes, k1=args.k1, b=args.b)
    qvecs = index.encode_queries(queries)
    return lambda qids, dids: index.score(qvecs[qids], None if len(dids) == len(codes) else dids)


def build_dense(queries, codes, args):
    from codesearch.dense import DenseEncoder

    enc = DenseEncoder(args.model, fp16=not args.no_fp16)
    q = enc.encode(queries, max_len=args.query_max_len, batch_size=args.batch_size)
    d = enc.encode(codes, max_len=args.code_max_len, batch_size=args.batch_size)
    return lambda qids, dids: q[qids] @ d[dids].T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("method", choices=["bm25", "dense"])
    ap.add_argument("--model", default="models/unixcoder-base",
                    help="local sentence-transformers dir or HF id (see scripts/convert_base_model.py)")
    ap.add_argument("--name", help="run name for results/ (default: derived from method/model)")
    ap.add_argument("--split", default="test", choices=["test", "validation"])
    ap.add_argument("--protocol", default="full", choices=list(PROTOCOLS),
                    help="full: rank against the whole split; csn1000: 999 distractors per query")
    ap.add_argument("--limit", type=int, help="evaluate only the first N pairs (for smoke tests)")
    ap.add_argument("--query-max-len", type=int, default=64)
    ap.add_argument("--code-max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--k1", type=float, default=1.2)
    ap.add_argument("--b", type=float, default=0.75)
    args = ap.parse_args()

    pairs = load_pairs(args.split)
    if args.limit:
        pairs = pairs[: args.limit]
    queries = [p.query for p in pairs]
    codes = [p.code for p in pairs]

    name = args.name or ("bm25" if args.method == "bm25" else Path(args.model.rstrip("/\\")).name)
    if args.limit:
        name += f"_limit{args.limit}"
    print(f"[eval] {name}: {len(pairs)} pairs from {args.split}, protocol={args.protocol}")

    t0 = time.time()
    score_fn = (build_bm25 if args.method == "bm25" else build_dense)(queries, codes, args)
    t_index = time.time() - t0
    ranks = gold_ranks(score_fn, len(pairs), pool_size=PROTOCOLS[args.protocol])
    t_total = time.time() - t0

    metrics = summarize(ranks)
    result = {
        "name": name,
        "method": args.method,
        "model": args.model if args.method == "dense" else None,
        "split": args.split,
        "protocol": args.protocol,
        "n_queries": len(pairs),
        "metrics": metrics,
        "seconds": {"index": round(t_index, 1), "total": round(t_total, 1)},
        "config": {k: v for k, v in vars(args).items() if k not in ("method", "name")},
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    stem = RESULTS_DIR / f"{name}__{args.split}__{args.protocol}"
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2))
    np.save(stem.with_suffix(".ranks.npy"), ranks)

    print(f"\n{name} ({args.protocol}, n={len(pairs)}, {t_total:.0f}s)")
    for k, v in metrics.items():
        print(f"  {k:<12} {v:.4f}" if k != "median_rank" else f"  {k:<12} {v:.0f}")
    print(f"saved -> {stem}.json")


if __name__ == "__main__":
    main()
