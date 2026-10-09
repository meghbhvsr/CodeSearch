"""Download CodeSearchNet Python and write the cleaned splits to data/.

eval.py does this automatically for the split it needs; run this to build all three up front
(train is ~400k functions and is needed for Phase 2).

  python scripts/prepare_data.py              # all splits
  python scripts/prepare_data.py test --refresh
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("USE_TF", "0")  # a broken TensorFlow install otherwise breaks transformers imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codesearch.data import SPLIT_FILES, load_pairs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("splits", nargs="*", default=list(SPLIT_FILES))
ap.add_argument("--refresh", action="store_true", help="rebuild even if cached")
args = ap.parse_args()

for split in args.splits:
    pairs = load_pairs(split, refresh=args.refresh)
    ex = pairs[0]
    print(f"{split}: {len(pairs):,} pairs   e.g. {ex.query!r}")
