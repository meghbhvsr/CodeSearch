"""Print every saved eval run in results/ as a markdown table (paste into the README).

  python scripts/report.py
  python scripts/report.py --protocol csn1000
"""

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
COLS = ["MRR@10", "R@1", "R@5", "R@10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="full")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    runs = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))]
    runs = [r for r in runs if r["protocol"] == args.protocol and r["split"] == args.split]
    if not runs:
        print(f"no runs for split={args.split} protocol={args.protocol} in {RESULTS_DIR}")
        return
    runs.sort(key=lambda r: r["metrics"]["MRR@10"])

    print(f"CodeSearchNet Python {args.split}, protocol={args.protocol}\n")
    print("| Model | Queries | " + " | ".join(COLS) + " |")
    print("|---|---:|" + "---:|" * len(COLS))
    for r in runs:
        cells = " | ".join(f"{r['metrics'][c]:.3f}" for c in COLS)
        print(f"| {r['name']} | {r['n_queries']:,} | {cells} |")


if __name__ == "__main__":
    main()
