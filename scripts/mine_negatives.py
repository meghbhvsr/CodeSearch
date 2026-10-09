"""Mine one hard negative per training query with a fine-tuned model.

A hard negative is a function the model ranks highly for a query but that isn't the
query's own function. Training on (query, positive, hard negative) triples teaches
finer distinctions than random in-batch negatives, most of which are trivially unrelated.

The risk is false negatives: CodeSearchNet contains near-copies of the same function
under different docstrings, and treating one as "wrong" teaches the model something
false. Two filters guard against that:
  * score cap: skip candidates scoring >= `--max-pos-ratio` x the positive's score
    (the "TopK-PercPos" filter from NV-Retriever; 0.95 is their default)
  * same docstring: skip candidates whose own query text is identical to this query
From the candidates that survive, one is picked at random from the top `--sample-from`.
That gives some variety instead of always taking the single hardest one, which is the
most likely to be a missed duplicate.

  python scripts/mine_negatives.py --limit 5000          # quick check
  python scripts/mine_negatives.py                       # all 401k train pairs, ~15-20 min

Output: data/hardneg_<name>.jsonl, one line per train pair (same order as
data/csn_python_train.jsonl), giving the negative's index into the train set.
Then: python scripts/train.py --run-name mnrl-hn --hard-negatives data/hardneg_mnrl.jsonl
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
sys.modules["tensorflow"] = None  # see train.py
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from codesearch.data import load_pairs  # noqa: E402
from codesearch.dense import DenseEncoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="checkpoints/mnrl/final", help="model that does the mining")
    ap.add_argument("--name", help="output name (default: the model's run directory name)")
    ap.add_argument("--top-k", type=int, default=200,
                    help="candidates retrieved per query. At 50, 16%% of queries had every candidate filtered out")
    ap.add_argument("--sample-from", type=int, default=10, help="pick randomly among the top N surviving candidates")
    ap.add_argument("--max-pos-ratio", type=float, default=0.95, help="drop candidates scoring >= ratio x positive")
    ap.add_argument("--query-max-len", type=int, default=64)
    ap.add_argument("--code-max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--limit", type=int, help="mine within the first N train pairs only (smoke test)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model_path = Path(args.model)
    name = args.name or (model_path.parent.name if model_path.name == "final" else model_path.name)
    if args.limit:
        name += f"_limit{args.limit}"
    out_path = ROOT / "data" / f"hardneg_{name}.jsonl"

    pairs = load_pairs("train")
    if args.limit:
        pairs = pairs[: args.limit]
    n = len(pairs)
    print(f"[mine] {n:,} train pairs, mining with {args.model}")

    t0 = time.time()
    enc = DenseEncoder(args.model)
    q = enc.encode([p.query for p in pairs], max_len=args.query_max_len, batch_size=args.batch_size).half()
    d = enc.encode([p.code for p in pairs], max_len=args.code_max_len, batch_size=args.batch_size).half()
    t_embed = time.time() - t0

    # Group queries by exact docstring text, for the same-docstring filter.
    query_group = {}
    group_of = np.array([query_group.setdefault(p.query, len(query_group)) for p in pairs])
    group_of_t = torch.as_tensor(group_of, device=d.device)

    rng = np.random.default_rng(args.seed)
    k = min(args.top_k + 1, n)  # +1 because the positive itself is usually in the top k
    neg_idx = np.full(n, -1, dtype=np.int64)
    neg_rank = np.zeros(n, dtype=np.int64)  # rank of the chosen negative among all candidates
    neg_score = np.zeros(n, dtype=np.float32)
    pos_score = np.zeros(n, dtype=np.float32)
    n_filtered = 0
    chunk = 1024
    for start in range(0, n, chunk):
        qi = torch.arange(start, min(start + chunk, n), device=d.device)
        scores = q[qi] @ d.T  # (chunk, n)
        pos = scores[torch.arange(len(qi), device=d.device), qi].float()
        top_s, top_i = scores.topk(k, dim=1)
        top_s = top_s.float()

        ok = top_i != qi[:, None]  # not the positive itself
        ok &= top_s < args.max_pos_ratio * pos[:, None]  # not suspiciously close to the positive
        ok &= group_of_t[top_i] != group_of_t[qi][:, None]  # not a function with the same docstring
        n_filtered += int((~ok).sum() - len(qi))  # don't count the positive itself

        ok, top_s, top_i, pos = ok.cpu().numpy(), top_s.cpu().numpy(), top_i.cpu().numpy(), pos.cpu().numpy()
        for row, i in enumerate(range(start, start + len(qi))):
            survivors = np.flatnonzero(ok[row])[: args.sample_from]
            pos_score[i] = pos[row]
            if len(survivors) == 0:
                continue
            pick = rng.choice(survivors)
            neg_idx[i], neg_rank[i], neg_score[i] = top_i[row, pick], pick + 1, top_s[row, pick]

    # Rare: every top-k candidate was filtered out. Fall back to a random function, so
    # every training row still has a negative.
    missing = np.flatnonzero(neg_idx < 0)
    for i in missing:
        j = int(rng.integers(n - 1))
        neg_idx[i] = j + (j >= i)
        neg_rank[i] = -1
        neg_score[i] = float(q[i].float() @ d[neg_idx[i]].float())

    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"idx": i, "neg_idx": int(neg_idx[i]), "neg_rank": int(neg_rank[i]),
                                "neg_score": round(float(neg_score[i]), 4),
                                "pos_score": round(float(pos_score[i]), 4)}) + "\n")

    mined = neg_rank > 0
    stats = {
        "model": args.model,
        "n_pairs": n,
        "embed_seconds": round(t_embed),
        "total_seconds": round(time.time() - t0),
        "random_fallbacks": int(len(missing)),
        "candidates_filtered_per_query": round(n_filtered / n, 2),
        "median_neg_rank": float(np.median(neg_rank[mined])),
        "median_pos_score": float(np.median(pos_score)),
        "median_neg_score": float(np.median(neg_score[mined])),
        "args": vars(args),
    }
    out_path.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "args"}, indent=2))
    print(f"saved -> {out_path}")

    print("\nexamples:")
    for i in rng.choice(np.flatnonzero(mined), size=min(3, mined.sum()), replace=False):
        first = lambda code: code.strip().splitlines()[0][:90]  # noqa: E731
        print(f"  query:    {pairs[i].query[:90]}")
        print(f"  positive: {first(pairs[i].code)}   (score {pos_score[i]:.3f})")
        print(f"  negative: {first(pairs[neg_idx[i]].code)}   (score {neg_score[i]:.3f}, rank {neg_rank[i]})\n")


if __name__ == "__main__":
    main()
