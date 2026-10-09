"""Fine-tune UniXcoder for docstring -> code retrieval on the CodeSearchNet Python train split.

Loss: CachedMultipleNegativesRankingLoss. Each query's own function is its positive, and
every other function in the batch is a negative. More negatives per batch generally means
a better model, but a batch of 256 code snippets at 256 tokens doesn't fit in 8 GB.
Gradient caching (GradCache) gets around this: it embeds the batch in mini-batches of
`--mini-batch-size` without gradients, computes the loss over the full batch, then
re-runs each mini-batch with gradients. The result is the same as one big batch, using
memory for only one mini-batch, at the cost of about one extra forward pass.

  python scripts/train.py --run-name smoke --limit 2048 --eval-steps 4    # 2-minute check
  python scripts/train.py --run-name mnrl                                 # full run, 1 epoch
  python scripts/train.py --run-name mnrl --resume                        # continue after a crash
  python scripts/train.py --run-name mnrl-hn --hard-negatives data/hardneg_mnrl.jsonl

With --hard-negatives, each row is (query, function, mined near-miss). Every query is then
scored against 2 x batch - 1 candidates, so each step embeds twice as much code and takes
correspondingly longer.

Progress: python scripts/view_logs.py   (then open http://localhost:6006)
Afterwards, score the best model with the same eval as the baselines:
  python scripts/eval.py dense --model checkpoints/mnrl/final --name mnrl --split validation
  python scripts/eval.py dense --model checkpoints/mnrl/final --name mnrl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("USE_TF", "0")  # a broken TensorFlow install otherwise breaks transformers imports
# TensorBoard ignores USE_TF and imports TensorFlow whenever it's installed; the broken install
# then fails with "module 'tensorflow' has no attribute 'io'". Hiding it makes TensorBoard use
# its built-in TensorFlow-free writer. Nothing in this project needs TensorFlow.
sys.modules["tensorflow"] = None
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from sentence_transformers import (  # noqa: E402
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator  # noqa: E402
from sentence_transformers.training_args import BatchSamplers  # noqa: E402

from codesearch.data import load_pairs  # noqa: E402

BEST_METRIC = "eval_val_cosine_mrr@10"


def truncate_queries(tokenizer, queries: list[str], max_len: int) -> list[str]:
    """Cut each query's text to `max_len` tokens (including special tokens).

    The model has one max_seq_length (sized for code), so queries are trimmed as text up
    front to match the 64-token limit eval.py uses for queries.
    """
    enc = tokenizer(queries, truncation=True, max_length=max_len, return_offsets_mapping=True)
    out = []
    for text, offsets in zip(queries, enc["offset_mapping"]):
        end = max((e for _, e in offsets), default=0)  # special tokens have (0, 0)
        out.append(text[:end])
    return out


def build_val_evaluator(tokenizer, size: int, query_max_len: int, seed: int) -> InformationRetrievalEvaluator:
    """MRR@10 / Recall@k over a fixed random validation subset, used to pick checkpoints.

    The pool is only `size` functions instead of the 21k in eval.py's full protocol, so these
    numbers come out higher than the final ones. Use them to compare checkpoints with each
    other, not with the test results.
    """
    pairs = load_pairs("validation")
    pairs = random.Random(seed).sample(pairs, min(size, len(pairs)))
    queries = truncate_queries(tokenizer, [p.query for p in pairs], query_max_len)
    return InformationRetrievalEvaluator(
        queries={str(i): q for i, q in enumerate(queries)},
        corpus={str(i): p.code for i, p in enumerate(pairs)},
        relevant_docs={str(i): {str(i)} for i in range(len(pairs))},
        name="val",
        mrr_at_k=[10],
        accuracy_at_k=[1, 5, 10],
        precision_recall_at_k=[10],
        ndcg_at_k=[10],
        map_at_k=[10],
        batch_size=128,
        show_progress_bar=False,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True, help="output goes to checkpoints/<run-name>")
    ap.add_argument("--base-model", default="models/unixcoder-base")
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=256, help="contrastive batch: 1 positive + batch-1 negatives")
    ap.add_argument("--mini-batch-size", type=int, default=32, help="GradCache chunk. 32 peaks at ~7.7 GB on an 8 GB card; 64 overflows into shared system RAM on Windows and runs 9x slower")
    ap.add_argument("--query-max-len", type=int, default=64)
    ap.add_argument("--code-max-len", type=int, default=256)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--eval-steps", type=int, default=250)
    ap.add_argument("--val-size", type=int, default=5000)
    ap.add_argument("--hard-negatives", help="data/hardneg_<name>.jsonl from mine_negatives.py; adds a third column")
    ap.add_argument("--limit", type=int, help="train on only the first N pairs (smoke test)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-to", default="tensorboard", help="tensorboard | wandb | none")
    ap.add_argument("--resume", action="store_true", help="resume from the last checkpoint in the run dir")
    args = ap.parse_args()

    out_dir = ROOT / "checkpoints" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2))

    model = SentenceTransformer(args.base_model, device="cuda" if torch.cuda.is_available() else "cpu")
    model.max_seq_length = args.code_max_len
    tokenizer = model.tokenizer

    all_pairs = load_pairs("train")
    pairs = all_pairs[: args.limit] if args.limit else all_pairs
    print(f"[train] {len(pairs):,} training pairs, batch {args.batch_size} "
          f"(GradCache mini-batch {args.mini_batch_size}), {args.epochs} epoch(s)")
    columns = {
        # Column order matters: (anchor, positive[, negative]). Every other code/negative
        # in the batch is also used as a negative for each query.
        "query": truncate_queries(tokenizer, [p.query for p in pairs], args.query_max_len),
        "code": [p.code for p in pairs],
    }
    if args.hard_negatives:
        with open(args.hard_negatives, encoding="utf-8") as f:
            mined = [json.loads(line) for line in f]
        if len(mined) != len(all_pairs) or any(m["idx"] != i for i, m in enumerate(mined)):
            sys.exit(f"{args.hard_negatives} doesn't line up with the train split; re-run mine_negatives.py")
        columns["negative"] = [all_pairs[m["neg_idx"]].code for m in mined[: len(pairs)]]
        print(f"[train] + 1 mined hard negative per query from {args.hard_negatives}")
    train_ds = Dataset.from_dict(columns)

    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch_size)
    evaluator = build_val_evaluator(tokenizer, args.val_size, args.query_max_len, args.seed)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir),
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        fp16=torch.cuda.is_available(),
        # A batch with two copies of the same docstring or function would mark the second
        # as a "negative" for the first, which teaches the model something false.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        dataloader_drop_last=True,  # keep every batch at full size, i.e. a fixed number of negatives
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=BEST_METRIC,
        greater_is_better=True,
        logging_steps=max(1, min(25, args.eval_steps)),
        logging_dir=str(out_dir / "tb"),
        report_to=[] if args.report_to == "none" else [args.report_to],
        seed=args.seed,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        loss=loss,
        evaluator=evaluator,
    )

    if not args.resume:
        start = evaluator(model)  # step-0 score: the zero-shot model on the same val subset
        print(f"[train] before training: val MRR@10 = {start['val_cosine_mrr@10']:.4f}")
    trainer.train(resume_from_checkpoint=True if args.resume else None)

    final_dir = out_dir / "final"
    model.save(str(final_dir))  # best checkpoint by val MRR@10 (load_best_model_at_end)
    final = evaluator(model)
    print(f"[train] best checkpoint: val MRR@10 = {final['val_cosine_mrr@10']:.4f} -> {final_dir}")

    # The saved tokenizer must still add UniXcoder's <encoder-only> mode token.
    ids = SentenceTransformer(str(final_dir), device="cpu").tokenizer("x")["input_ids"]
    assert ids[:3] == [0, 6, 2], f"saved tokenizer lost the <encoder-only> prefix: {ids}"


if __name__ == "__main__":
    main()
