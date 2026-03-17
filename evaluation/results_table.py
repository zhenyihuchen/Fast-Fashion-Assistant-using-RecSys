"""Generate the per-tier × per-model comparison table from evaluation results.

python -m evaluation.results_table              # latest single file (same as before)
python -m evaluation.results_table --all        # merge all eval_*.json files
python -m evaluation.results_table --results eval_day1.json eval_day2.json  # specific files

Reads a single eval JSON (with tier info + random baseline) and produces:
  1. Formatted tables printed to stdout
  2. CSVs saved next to the JSON
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"

ITEM_RUBRICS = ["item_relevance", "occasion_appropriateness", "explanation_quality"]
SET_RUBRICS = ["set_answer_relevance"]
PARSER_RUBRICS = ["parser_completeness", "parser_no_hallucination", "parser_occasion_detection"]

TIER_LABELS = {1: "Basic", 2: "Medium", 3: "Complex"}

# Display names for models
MODEL_DISPLAY = {
    "clip": "CLIP",
    "fashion_clip": "FashionCLIP",
    "random": "Random",
}


def _safe_mean(values: list) -> float | None:
    valid = [v for v in values if isinstance(v, (int, float)) and v >= 0]
    return round(statistics.mean(valid), 2) if valid else None


def _safe_std(values: list) -> float | None:
    valid = [v for v in values if isinstance(v, (int, float)) and v >= 0]
    return round(statistics.stdev(valid), 2) if len(valid) >= 2 else None


def _extract_item_scores(query_eval: dict, model: str, rubric: str) -> list[float]:
    """Get all item-level scores for a model+rubric from one query's eval."""
    items = query_eval.get(model, {}).get("items", [])
    scores = []
    for item in items:
        if item is None:
            continue
        s = item.get(rubric, {}).get("score")
        if s is not None and s >= 0:
            scores.append(s)
    return scores


def _extract_set_score(query_eval: dict, model: str, rubric: str) -> float | None:
    s = query_eval.get(model, {}).get("set", {}).get(rubric, {}).get("score")
    return s if s is not None and s >= 0 else None


def _extract_parser_score(query_eval: dict, rubric: str) -> float | None:
    s = query_eval.get("parser", {}).get(rubric, {}).get("score")
    return s if s is not None and s >= 0 else None


def _detect_models(per_query: list[dict]) -> list[str]:
    """Detect all model keys from the eval results (clip, fashion_clip, random, ...)."""
    models = set()
    for q in per_query:
        for key in q.get("eval", {}):
            if key not in ("parser", "cross_model"):
                models.add(key)
    # Sort with a preferred order: random first (baseline), then clip, fashion_clip
    order = {"random": 0, "clip": 1, "fashion_clip": 2}
    return sorted(models, key=lambda m: order.get(m, 99))


def build_tier_table(data: dict) -> pd.DataFrame:
    """Build the full per-tier × per-model comparison DataFrame."""
    per_query = data.get("per_query", [])
    models = _detect_models(per_query)

    # Detect tiers — fall back to single group if no tier info
    tiers = sorted(set(q.get("tier") for q in per_query if q.get("tier") is not None))
    if not tiers:
        tiers = [None]

    rows = []

    for tier in tiers:
        tier_queries = [q for q in per_query if q.get("tier") == tier]
        tier_label = TIER_LABELS.get(tier, f"Tier {tier}") if tier else "All"
        n_queries = len(tier_queries)

        # Parser rubrics (model-independent)
        for rubric in PARSER_RUBRICS:
            scores = []
            for q in tier_queries:
                s = _extract_parser_score(q.get("eval", {}), rubric)
                if s is not None:
                    scores.append(s)
            rows.append({
                "Tier": tier_label,
                "N": n_queries,
                "Evaluated": len(scores),
                "Scope": "Parser",
                "Rubric": rubric,
                "Model": "—",
                "Mean": _safe_mean(scores),
                "Std": _safe_std(scores),
            })

        # Item + Set rubrics per model (including random)
        for model in models:
            model_label = MODEL_DISPLAY.get(model, model)
            for rubric in ITEM_RUBRICS:
                scores = []
                for q in tier_queries:
                    scores.extend(_extract_item_scores(q.get("eval", {}), model, rubric))
                rows.append({
                    "Tier": tier_label,
                    "N": n_queries,
                    "Evaluated": len(scores),
                    "Scope": "Item",
                    "Rubric": rubric,
                    "Model": model_label,
                    "Mean": _safe_mean(scores),
                    "Std": _safe_std(scores),
                })

            for rubric in SET_RUBRICS:
                scores = []
                for q in tier_queries:
                    s = _extract_set_score(q.get("eval", {}), model, rubric)
                    if s is not None:
                        scores.append(s)
                rows.append({
                    "Tier": tier_label,
                    "N": n_queries,
                    "Evaluated": len(scores),
                    "Scope": "Set",
                    "Rubric": rubric,
                    "Model": model_label,
                    "Mean": _safe_mean(scores),
                    "Std": _safe_std(scores),
                })

        # Cross-model win rates (CLIP vs FashionCLIP only — random not included)
        clip_wins = fc_wins = ties = 0
        for q in tier_queries:
            winner = q.get("eval", {}).get("cross_model", {}).get("winner")
            if winner == "clip":
                clip_wins += 1
            elif winner == "fashion_clip":
                fc_wins += 1
            elif winner == "tie":
                ties += 1
        total_cm = clip_wins + fc_wins + ties
        if total_cm > 0:
            rows.append({
                "Tier": tier_label, "N": n_queries, "Evaluated": total_cm, "Scope": "Cross-Model",
                "Rubric": "clip_win_rate", "Model": "CLIP",
                "Mean": round(clip_wins / total_cm * 100, 1), "Std": None,
            })
            rows.append({
                "Tier": tier_label, "N": n_queries, "Evaluated": total_cm, "Scope": "Cross-Model",
                "Rubric": "fashionclip_win_rate", "Model": "FashionCLIP",
                "Mean": round(fc_wins / total_cm * 100, 1), "Std": None,
            })
            rows.append({
                "Tier": tier_label, "N": n_queries, "Evaluated": total_cm, "Scope": "Cross-Model",
                "Rubric": "tie_rate", "Model": "—",
                "Mean": round(ties / total_cm * 100, 1), "Std": None,
            })

    return pd.DataFrame(rows)


def build_compact_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot into the compact comparison table with gain-over-random columns.

    Output columns: Tier | Rubric | Random | CLIP | FashionCLIP | CLIP_gain | FC_gain
    """
    item_set = df[df["Scope"].isin(["Item", "Set"])].copy()
    if item_set.empty:
        return pd.DataFrame()

    pivot = item_set.pivot_table(
        index=["Tier", "Rubric"],
        columns="Model",
        values="Mean",
        aggfunc="first",
    ).reset_index()

    # Flatten column names
    pivot.columns.name = None

    # Add gain-over-random columns
    if "Random" in pivot.columns:
        for model, gain_col in [("CLIP", "CLIP_gain"), ("FashionCLIP", "FC_gain")]:
            if model in pivot.columns:
                pivot[gain_col] = (pivot[model] - pivot["Random"]).round(2)

    # Reorder columns
    desired_order = ["Tier", "Rubric", "Random", "CLIP", "FashionCLIP", "CLIP_gain", "FC_gain"]
    cols = [c for c in desired_order if c in pivot.columns]
    # Add any extra model columns not in desired_order
    for c in pivot.columns:
        if c not in cols:
            cols.append(c)
    pivot = pivot[cols]

    return pivot


def print_tables(df_long: pd.DataFrame, df_compact: pd.DataFrame) -> None:
    """Pretty-print tables to stdout."""

    # ── Compact table (the main one) ──
    if not df_compact.empty:
        print("\n" + "=" * 90)
        print("  COMPARISON TABLE — Item & Set rubrics (mean score, 1-5 scale)")
        print("  Gain = model score − random score")
        print("=" * 90)
        print(df_compact.to_string(index=False))

    # ── Parser ──
    parser_rows = df_long[df_long["Scope"] == "Parser"]
    if not parser_rows.empty:
        print("\n" + "=" * 90)
        print("  PARSER SCORES (model-independent, 1-5 scale)")
        print("=" * 90)
        print(parser_rows[["Tier", "Rubric", "Evaluated", "Mean", "Std"]].to_string(index=False))

    # ── Cross-model ──
    cm_rows = df_long[df_long["Scope"] == "Cross-Model"]
    if not cm_rows.empty:
        print("\n" + "=" * 90)
        print("  CROSS-MODEL WIN RATES (%, CLIP vs FashionCLIP)")
        print("=" * 90)
        cm_pivot = cm_rows.pivot_table(
            index=["Tier", "Evaluated"], columns="Rubric", values="Mean", aggfunc="first",
        ).reset_index()
        cm_pivot.columns.name = None
        print(cm_pivot.to_string(index=False))


def _merge_results(paths: list[Path]) -> dict:
    """Merge multiple batch eval JSONs into a single data dict."""
    all_per_query: list[dict] = []
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        all_per_query.extend(data.get("per_query", []))
    return {
        "run_timestamp": "merged",
        "total_queries": len(all_per_query),
        "per_query": all_per_query,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate per-tier comparison table from eval results")
    parser.add_argument("--results", type=Path, nargs="+", default=None,
                        help="Path(s) to eval JSON file(s). Multiple files are merged.")
    parser.add_argument("--latest", action="store_true", help="Use the most recent result file")
    parser.add_argument("--all", action="store_true", dest="merge_all",
                        help="Merge all eval_*.json files in the results directory")
    args = parser.parse_args()

    if args.results:
        paths = args.results
    elif args.merge_all:
        paths = sorted(RESULTS_DIR.glob("eval_*.json"))
        if not paths:
            print("No result files found in", RESULTS_DIR)
            return
    else:
        # Default: use the most recent file
        files = sorted(RESULTS_DIR.glob("eval_*.json"), reverse=True)
        if not files:
            print("No result files found in", RESULTS_DIR)
            return
        paths = [files[0]]

    if len(paths) == 1:
        results_path = paths[0]
        print(f"Reading: {results_path}")
        data = json.loads(results_path.read_text(encoding="utf-8"))
        stem = results_path.stem.replace("eval_", "")
    else:
        print(f"Merging {len(paths)} result files:")
        for p in paths:
            print(f"  {p.name}")
        data = _merge_results(paths)
        stem = "merged"

    print(f"Queries: {data.get('total_queries', '?')} | Timestamp: {data.get('run_timestamp', '?')}")

    df_long = build_tier_table(data)
    df_compact = build_compact_table(df_long)

    print_tables(df_long, df_compact)

    # Save CSVs
    out_dir = paths[0].parent if paths else RESULTS_DIR
    long_csv = out_dir / f"tier_table_{stem}.csv"
    df_long.to_csv(long_csv, index=False)
    print(f"\nDetailed CSV: {long_csv}")

    if not df_compact.empty:
        compact_csv = out_dir / f"tier_compact_{stem}.csv"
        df_compact.to_csv(compact_csv, index=False)
        print(f"Compact CSV:  {compact_csv}")


if __name__ == "__main__":
    main()
