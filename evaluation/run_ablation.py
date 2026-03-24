"""Run the occasion ablation study on tier-3 (complex) queries.

Re-runs all tier-3 queries with occasion scoring disabled (α=1.0, β=0.0)
so results can be compared against the existing occasion-enabled runs.

Usage (from project root, with myenv activated):
    python -m evaluation.run_ablation
    python -m evaluation.run_ablation --start 0 --end 25   # first 25 tier-3 queries
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.run_eval import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Occasion ablation: run tier-3 queries with relevance-only ranking"
    )
    parser.add_argument(
        "--queries", type=Path,
        default=ROOT / "evaluation" / "test_queries.json",
    )
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "evaluation" / "results" / "ablation_no_occasion",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    # Filter to tier-3 only, write a temporary queries file
    all_queries = json.loads(args.queries.read_text(encoding="utf-8"))
    tier3 = [q for q in all_queries if q.get("tier") == 3]
    tier3 = tier3[args.start : args.end]

    tmp_queries = args.out / "tier3_queries.json"
    args.out.mkdir(parents=True, exist_ok=True)
    tmp_queries.write_text(json.dumps(tier3, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Running {len(tier3)} tier-3 queries with occasion scoring DISABLED")
    run(
        queries_path=tmp_queries,
        out_dir=args.out,
        disable_occasion=True,
    )


if __name__ == "__main__":
    main()
