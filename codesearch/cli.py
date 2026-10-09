"""codesearch: natural-language search over a Python repository.

  codesearch "where do we retry failed HTTP requests" ./requests
  codesearch retry failed http requests ./requests -k 5     # quotes optional
  codesearch ./requests                                     # interactive: one model load, many queries

The model defaults to the fine-tuned checkpoints/mnrl/final; override it with --model or
the CODESEARCH_MODEL environment variable.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("USE_TF", "0")  # a broken TensorFlow install otherwise breaks transformers imports
sys.modules.setdefault("tensorflow", None)  # see scripts/train.py

import argparse  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

from rich.console import Console  # noqa: E402
from rich.syntax import Syntax  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "checkpoints" / "mnrl" / "final"
console = Console(highlight=False)


def print_results(results, lines: int):
    for rank, (chunk, score) in enumerate(results, 1):
        console.print(f"\n[bold]{rank:>2}.[/] [green]{score:.3f}[/]  [cyan]{chunk.path}:{chunk.start_line}[/]  "
                      f"[bold yellow]{chunk.name}[/]")
        if lines > 0:
            src = chunk.code.splitlines()
            snippet = "\n".join(src[:lines]) + ("\n    ..." if len(src) > lines else "")
            first_def = chunk.end_line - len(src) + 1  # start_line is the first decorator, if any
            console.print(Syntax(snippet, "python", line_numbers=True, start_line=first_def,
                                 theme="ansi_dark", background_color="default", code_width=110))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="codesearch", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("terms", nargs="+", metavar="[QUERY ...] REPO", help="search terms followed by the repo directory")
    ap.add_argument("-k", type=int, default=10, help="number of results (default 10)")
    ap.add_argument("--lines", type=int, default=8, help="lines of code to show per result, 0 for none")
    ap.add_argument("--model", default=os.environ.get("CODESEARCH_MODEL", str(DEFAULT_MODEL)))
    ap.add_argument("--tests", action="store_true", help="include test code (hidden by default)")
    ap.add_argument("--reindex", action="store_true", help="ignore the cache and re-embed everything")
    ap.add_argument("--cpu", action="store_true", help="run the model on CPU")
    args = ap.parse_args(argv)

    repo = Path(args.terms[-1])
    if not repo.is_dir():
        ap.error(f"last argument must be a repo directory, got {repo!s}")
    query = " ".join(args.terms[:-1]).strip()
    if not Path(args.model).exists() and args.model == str(DEFAULT_MODEL):
        console.print(f"[yellow]{DEFAULT_MODEL} not found; falling back to the zero-shot base model[/]")
        args.model = str(ROOT / "models" / "unixcoder-base")

    # Library chatter (deprecations, "no sentence-transformers model found" etc.) would bury results.
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    os.environ["TQDM_DISABLE"] = "1"

    with console.status("loading model ..."):
        from codesearch.dense import DenseEncoder
        from codesearch.index import QUERY_MAX_LEN, RepoIndexError, RepoIndex

        encoder = DenseEncoder(args.model, device="cpu" if args.cpu else None)
        encoder.show_progress = False
    try:
        index = RepoIndex.build(repo, encoder, args.model, force=args.reindex,
                                log=lambda msg: console.print(f"[dim]{msg}[/]"))
    except RepoIndexError as e:
        sys.exit(f"codesearch: {e}")

    def run(q: str):
        t0 = time.time()
        q_emb = encoder.encode([q], max_len=QUERY_MAX_LEN).cpu().numpy()
        results = index.search(q_emb, args.k, tests=args.tests)
        print_results(results, args.lines)
        n_tests = int(index.is_test.sum())
        hidden = f" ({n_tests:,} in test code hidden; --tests to include)" if n_tests and not args.tests else ""
        console.print(f"\n[dim]{len(index.chunks):,} functions searched in {(time.time() - t0) * 1000:.0f} ms{hidden}[/]")

    if query:
        run(query)
        return
    console.print(f"[dim]{len(index.chunks):,} functions indexed in {index.repo}. Empty line or Ctrl+C to quit.[/]")
    while True:
        try:
            q = console.input("\n[bold magenta]search>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        run(q)


if __name__ == "__main__":
    main()
