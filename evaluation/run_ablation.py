"""Run ablation studies on tier-3 (complex) queries.

Supports two ablation modes:
  - no-occasion: Disable occasion scoring, rank purely by relevance (α=1.0, β=0.0)
  - no-pipeline: Raw embedding similarity only — no parsing, filtering, or occasion

Usage (from project root, with myenv activated):
    python -m evaluation.run_ablation                          # no-occasion (default)
    python -m evaluation.run_ablation --mode no-pipeline       # raw baseline
    python -m evaluation.run_ablation --start 0 --end 3        # first 3 tier-3 queries
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.run_eval import run

ABLATION_DIRS = {
    "no-occasion": ROOT / "evaluation" / "results" / "ablation_no_occasion",
    "no-pipeline": ROOT / "evaluation" / "results" / "ablation_no_pipeline",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation study: run tier-3 queries under degraded conditions"
    )
    parser.add_argument(
        "--mode",
        choices=["no-occasion", "no-pipeline"],
        default="no-occasion",
        help="Ablation mode (default: no-occasion)",
    )
    parser.add_argument(
        "--queries", type=Path,
        default=ROOT / "evaluation" / "test_queries.json",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    out_dir = args.out or ABLATION_DIRS[args.mode]

    # Filter to tier-3 only, write a temporary queries file
    all_queries = json.loads(args.queries.read_text(encoding="utf-8"))
    tier3 = [q for q in all_queries if q.get("tier") == 3]
    tier3 = tier3[args.start : args.end]

    tmp_queries = out_dir / "tier3_queries.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_queries.write_text(json.dumps(tier3, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.mode == "no-occasion":
        print(f"Running {len(tier3)} tier-3 queries with occasion scoring DISABLED")
        run(queries_path=tmp_queries, out_dir=out_dir, disable_occasion=True)
    elif args.mode == "no-pipeline":
        print(f"Running {len(tier3)} tier-3 queries with NO PIPELINE (raw embedding baseline)")
        run(queries_path=tmp_queries, out_dir=out_dir, no_pipeline=True)


if __name__ == "__main__":
    main()
