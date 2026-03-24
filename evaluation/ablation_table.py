"""Compare occasion-enabled vs relevance-only results for tier-3 queries.

Produces two outputs:
  1. Complete table across all rubrics for both conditions
  2. Focused "Relevance-only vs Relevance+Occasion" across item_relevance
     and occasion_appropriateness, per model

Usage:
    python -m evaluation.ablation_table
    python -m evaluation.ablation_table --occasion-results eval_a.json eval_b.json \
                                        --no-occasion-results ablation.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "evaluation" / "results"
ABLATION_DIR = RESULTS_DIR / "ablation_no_occasion"

MODEL_DISPLAY = {"clip": "CLIP", "fashion_clip": "FashionCLIP", "random": "Random"}
ITEM_RUBRICS = ["item_relevance", "occasion_appropriateness", "explanation_quality"]
SET_RUBRICS = ["set_answer_relevance"]
PARSER_RUBRICS = ["parser_completeness", "parser_no_hallucination", "parser_occasion_detection"]


def _safe_mean(values: list) -> float | None:
    valid = [v for v in values if isinstance(v, (int, float)) and v >= 0]
    return round(statistics.mean(valid), 2) if valid else None


def _safe_std(values: list) -> float | None:
    valid = [v for v in values if isinstance(v, (int, float)) and v >= 0]
    return round(statistics.stdev(valid), 2) if len(valid) >= 2 else None


def _extract_item_scores(query_eval: dict, model: str, rubric: str) -> list[float]:
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
    models = set()
    for q in per_query:
        for key in q.get("eval", {}):
            if key not in ("parser", "cross_model"):
                models.add(key)
    order = {"random": 0, "clip": 1, "fashion_clip": 2}
    return sorted(models, key=lambda m: order.get(m, 99))


def _merge_jsons(paths: list[Path]) -> list[dict]:
    all_pq: list[dict] = []
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        all_pq.extend(data.get("per_query", []))
    return all_pq


def _filter_tier3(per_query: list[dict]) -> list[dict]:
    return [q for q in per_query if q.get("tier") == 3]


# ── Output 1: Full table for both conditions ─────────────────────────────────

def build_full_comparison(occ_pq: list[dict], no_occ_pq: list[dict]) -> pd.DataFrame:
    """Build a complete rubric table for both conditions side-by-side."""
    rows = []
    for condition_label, pq in [
        ("Relevance + Occasion", occ_pq),
        ("Relevance Only", no_occ_pq),
    ]:
        models = _detect_models(pq)
        n_queries = len(pq)

        # Parser rubrics
        for rubric in PARSER_RUBRICS:
            scores = []
            for q in pq:
                s = _extract_parser_score(q.get("eval", {}), rubric)
                if s is not None:
                    scores.append(s)
            rows.append({
                "Condition": condition_label,
                "N": n_queries,
                "Evaluated": len(scores),
                "Scope": "Parser",
                "Rubric": rubric,
                "Model": "—",
                "Mean": _safe_mean(scores),
                "Std": _safe_std(scores),
            })

        # Item + Set rubrics per model
        for model in models:
            model_label = MODEL_DISPLAY.get(model, model)
            for rubric in ITEM_RUBRICS:
                scores = []
                for q in pq:
                    scores.extend(_extract_item_scores(q.get("eval", {}), model, rubric))
                rows.append({
                    "Condition": condition_label,
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
                for q in pq:
                    s = _extract_set_score(q.get("eval", {}), model, rubric)
                    if s is not None:
                        scores.append(s)
                rows.append({
                    "Condition": condition_label,
                    "N": n_queries,
                    "Evaluated": len(scores),
                    "Scope": "Set",
                    "Rubric": rubric,
                    "Model": model_label,
                    "Mean": _safe_mean(scores),
                    "Std": _safe_std(scores),
                })

        # Cross-model win rates
        clip_wins = fc_wins = ties = 0
        for q in pq:
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
                "Condition": condition_label, "N": n_queries, "Evaluated": total_cm,
                "Scope": "Cross-Model", "Rubric": "clip_win_rate", "Model": "CLIP",
                "Mean": round(clip_wins / total_cm * 100, 1), "Std": None,
            })
            rows.append({
                "Condition": condition_label, "N": n_queries, "Evaluated": total_cm,
                "Scope": "Cross-Model", "Rubric": "fashionclip_win_rate", "Model": "FashionCLIP",
                "Mean": round(fc_wins / total_cm * 100, 1), "Std": None,
            })
            rows.append({
                "Condition": condition_label, "N": n_queries, "Evaluated": total_cm,
                "Scope": "Cross-Model", "Rubric": "tie_rate", "Model": "—",
                "Mean": round(ties / total_cm * 100, 1), "Std": None,
            })

    return pd.DataFrame(rows)


# ── Output 2: Focused comparison on 2 rubrics ────────────────────────────────

def build_focused_comparison(occ_pq: list[dict], no_occ_pq: list[dict]) -> pd.DataFrame:
    """Relevance-only vs Relevance+Occasion across item_relevance and
    occasion_appropriateness, per model."""
    focus_rubrics = ["item_relevance", "occasion_appropriateness"]
    rows = []

    for condition_label, pq in [
        ("Relevance Only", no_occ_pq),
        ("Relevance + Occasion", occ_pq),
    ]:
        models = _detect_models(pq)
        for model in models:
            model_label = MODEL_DISPLAY.get(model, model)
            for rubric in focus_rubrics:
                scores = []
                for q in pq:
                    scores.extend(_extract_item_scores(q.get("eval", {}), model, rubric))
                rows.append({
                    "Rubric": rubric,
                    "Model": model_label,
                    "Condition": condition_label,
                    "Mean": _safe_mean(scores),
                    "Std": _safe_std(scores),
                    "N": len(scores),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Pivot to get conditions as columns
    pivot = df.pivot_table(
        index=["Model", "Rubric"],
        columns="Condition",
        values="Mean",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None

    # Add delta column
    if "Relevance Only" in pivot.columns and "Relevance + Occasion" in pivot.columns:
        pivot["Δ"] = (pivot["Relevance + Occasion"] - pivot["Relevance Only"]).round(2)

    # Reorder columns
    desired = ["Model", "Rubric", "Relevance Only", "Relevance + Occasion", "Δ"]
    cols = [c for c in desired if c in pivot.columns]
    return pivot[cols]


def print_and_save(
    df_full: pd.DataFrame,
    df_focused: pd.DataFrame,
    out_dir: Path,
) -> None:
    print("\n" + "=" * 100)
    print("  OUTPUT 1: COMPLETE TABLE — Complex queries (Tier 3)")
    print("  Relevance + Occasion (α=0.6, β=0.4) vs Relevance Only (α=1.0, β=0.0)")
    print("=" * 100)
    print(df_full.to_string(index=False))

    print("\n" + "=" * 100)
    print("  OUTPUT 2: FOCUSED COMPARISON — item_relevance & occasion_appropriateness")
    print("  Δ = (Relevance + Occasion) − (Relevance Only)")
    print("=" * 100)
    print(df_focused.to_string(index=False))

    out_dir.mkdir(parents=True, exist_ok=True)
    full_csv = out_dir / "ablation_full_table.csv"
    focused_csv = out_dir / "ablation_focused_comparison.csv"
    df_full.to_csv(full_csv, index=False)
    df_focused.to_csv(focused_csv, index=False)
    print(f"\nSaved: {full_csv}")
    print(f"Saved: {focused_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Occasion ablation comparison tables")
    parser.add_argument(
        "--occasion-results", type=Path, nargs="+", default=None,
        help="Eval JSON(s) with occasion scoring enabled. Default: all eval_*.json in results/",
    )
    parser.add_argument(
        "--no-occasion-results", type=Path, nargs="+", default=None,
        help="Eval JSON(s) from ablation (no occasion). Default: all in results/ablation_no_occasion/",
    )
    parser.add_argument(
        "--out", type=Path, default=RESULTS_DIR,
    )
    args = parser.parse_args()

    # Load occasion-enabled results (existing runs, filtered to tier 3)
    if args.occasion_results:
        occ_paths = args.occasion_results
    else:
        occ_paths = sorted(RESULTS_DIR.glob("eval_*.json"))
    if not occ_paths:
        print("No occasion-enabled result files found.")
        return
    occ_pq = _filter_tier3(_merge_jsons(occ_paths))
    print(f"Occasion-enabled: {len(occ_pq)} tier-3 queries from {len(occ_paths)} file(s)")

    # Load relevance-only ablation results
    if args.no_occasion_results:
        no_occ_paths = args.no_occasion_results
    else:
        no_occ_paths = sorted(ABLATION_DIR.glob("eval_*.json"))
    if not no_occ_paths:
        print(f"No ablation result files found in {ABLATION_DIR}")
        print("Run `python -m evaluation.run_ablation` first.")
        return
    no_occ_pq = _merge_jsons(no_occ_paths)  # already tier-3 only
    print(f"Relevance-only:   {len(no_occ_pq)} tier-3 queries from {len(no_occ_paths)} file(s)")

    df_full = build_full_comparison(occ_pq, no_occ_pq)
    df_focused = build_focused_comparison(occ_pq, no_occ_pq)
    print_and_save(df_full, df_focused, args.out)


if __name__ == "__main__":
    main()
