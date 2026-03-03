"""Offline batch evaluation runner.

Usage (from project root, with myenv activated):
    python -m evaluation.run_eval
    python -m evaluation.run_eval --queries evaluation/test_queries.json --out evaluation/results/

For each query in the test set the script:
  1. Parses the query via parse_query_llm
  2. Retrieves and ranks candidates (CLIP + FashionCLIP)
  3. Generates explanations
  4. Runs all evaluation rubrics (parser, item, set, cross-model)
  5. Saves per-query results + aggregate statistics to JSON + CSV
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from online.candidate_retrieval import MODEL_PATHS, PARQUET_PATH, retrieve_candidates
from online.explanation_generation_groq import generate_explanations
from online.final_ranking import rank_candidates
from online.occasion_suitability_scores import (
    MODEL_OCCASION_EMBEDDINGS_PATHS,
    compute_occasion_scores,
)
from online.query_processing_llm import parse_query_llm

from evaluation.evaluators import evaluate_query_result

IMAGES_DIR = ROOT / "offline" / "data" / "images"

# ── Catalog cache ────────────────────────────────────────────────────────────

_catalog: pd.DataFrame | None = None


def _get_catalog() -> pd.DataFrame:
    global _catalog
    if _catalog is None:
        df = pd.read_parquet(PARQUET_PATH)
        df["row_id"] = df["row_id"].astype(str)
        _catalog = df.set_index("row_id")
    return _catalog


# ── Pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline(query: str) -> tuple[dict, dict[str, list[dict]]]:
    """Run the full recommendation pipeline for one query.

    Returns:
        (parsed, rows_by_model)
        where rows_by_model = {"clip": [...], "fashion_clip": [...]}
    """
    parsed = parse_query_llm(query)
    occ = parsed.get("occasion") or {}
    if occ.get("mode") != "on" or not occ.get("target"):
        # Return empty results if no occasion detected — parser eval still runs
        return parsed, {}

    candidates_by_model = retrieve_candidates(
        query,
        parsed=parsed,
        topk=30,
        filter_first=True,
        use_faiss=True,
        embedding_model="both",
    )
    if not any(candidates_by_model.values()):
        return parsed, {}

    df = _get_catalog()
    rows_by_model: dict[str, list[dict]] = {}

    for model_name, candidates in candidates_by_model.items():
        if not candidates:
            continue

        paths = MODEL_PATHS[model_name]
        product_ids = np.load(paths["image_ids"], allow_pickle=True).astype(str)
        embeddings = np.load(paths["image_embeddings"]).astype("float32", copy=False)

        occasion_scores = compute_occasion_scores(
            candidates,
            parsed=parsed,
            product_ids=product_ids,
            embeddings=embeddings,
            model_name=model_name,
            occasion_embeddings_path=MODEL_OCCASION_EMBEDDINGS_PATHS.get(model_name),
        )
        ranked = rank_candidates(candidates, occasion_scores)

        rows: list[dict] = []
        for row_id, relevance, occ_score, final_score in ranked:
            if row_id not in df.index:
                continue
            row = df.loc[row_id]
            rows.append({
                "row_id": row_id,
                "relevance_score": float(relevance),
                "occasion_score": float(occ_score),
                "final_score": float(final_score),
                "product_name": str(row.get("product_name", "") or ""),
                "product_description": str(row.get("product_description", "") or ""),
                "product_url": str(row.get("product_url", "") or ""),
                "price": str(row.get("price", "") or ""),
                "color": str(row.get("color", "") or ""),
                "product_category": str(row.get("product_category", "") or ""),
                "image_url": str(row.get("image_url", "") or ""),
                "local_image_path": str(
                    IMAGES_DIR / (str(row.get("reference_number", "") or "").replace("/", "_") + ".jpg")
                ),
            })

        explanations = generate_explanations(
            rows,
            candidates=candidates,
            parsed=parsed,
            product_ids=product_ids,
            embeddings=embeddings,
            occasion_scores=occasion_scores,
            model_name=model_name,
            occasion_embeddings_path=MODEL_OCCASION_EMBEDDINGS_PATHS.get(model_name),
        )
        for row in rows:
            row["explanation"] = explanations.get(str(row["row_id"]), "")

        rows_by_model[model_name] = rows[:5]

    return parsed, rows_by_model


# ── Aggregation ───────────────────────────────────────────────────────────────

def _safe_mean(values: list) -> float | None:
    valid = [v for v in values if isinstance(v, (int, float)) and v >= 0]
    return round(statistics.mean(valid), 4) if valid else None


def _safe_stdev(values: list) -> float | None:
    valid = [v for v in values if isinstance(v, (int, float)) and v >= 0]
    return round(statistics.stdev(valid), 4) if len(valid) >= 2 else None


def aggregate_results(per_query: list[dict]) -> dict:
    """Compute mean ± std for every rubric across queries."""

    def collect_item_scores(model_key: str, rubric_name: str) -> list:
        scores = []
        for q in per_query:
            model_data = q.get("eval", {}).get(model_key, {})
            for item_result in model_data.get("items", []):
                s = item_result.get(rubric_name, {}).get("score")
                if s is not None:
                    scores.append(s)
        return scores

    def collect_set_scores(model_key: str, rubric_name: str) -> list:
        return [
            q["eval"][model_key]["set"].get(rubric_name, {}).get("score")
            for q in per_query
            if model_key in q.get("eval", {}) and "set" in q["eval"][model_key]
        ]

    def collect_parser_scores(rubric_name: str) -> list:
        return [
            q["eval"]["parser"].get(rubric_name, {}).get("score")
            for q in per_query
            if "parser" in q.get("eval", {})
        ]

    item_rubric_names = [
        "item_relevance", "occasion_appropriateness",
        "explanation_groundedness", "explanation_quality", "constraint_adherence",
    ]
    set_rubric_names = ["set_answer_relevance"]
    parser_rubric_names = ["parser_completeness", "parser_no_hallucination", "parser_occasion_detection"]

    agg: dict = {"clip": {}, "fashion_clip": {}, "parser": {}, "cross_model": {}}

    for model_key in ("clip", "fashion_clip"):
        for rubric_name in item_rubric_names:
            vals = collect_item_scores(model_key, rubric_name)
            agg[model_key][rubric_name] = {"mean": _safe_mean(vals), "std": _safe_stdev(vals), "n": len(vals)}
        for rubric_name in set_rubric_names:
            vals = collect_set_scores(model_key, rubric_name)
            agg[model_key][rubric_name] = {"mean": _safe_mean(vals), "std": _safe_stdev(vals), "n": len(vals)}

    for rubric_name in parser_rubric_names:
        vals = collect_parser_scores(rubric_name)
        agg["parser"][rubric_name] = {"mean": _safe_mean(vals), "std": _safe_stdev(vals), "n": len(vals)}

    # Cross-model win rates
    winners = [q["eval"].get("cross_model", {}).get("winner") for q in per_query]
    agg["cross_model"] = {
        "clip_wins": winners.count("clip"),
        "fashionclip_wins": winners.count("fashion_clip"),
        "ties": winners.count("tie"),
        "total": len([w for w in winners if w in {"clip", "fashion_clip", "tie"}]),
    }

    return agg


# ── Main runner ──────────────────────────────────────────────────────────────

def run(queries_path: Path, out_dir: Path, skip_pipeline: bool = False) -> None:
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    queries = queries[:1]  # TODO: remove to run full test set
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    per_query_results: list[dict] = []
    total = len(queries)

    for i, q in enumerate(queries, 1):
        qid = q.get("id", f"q{i:03d}")
        query_text = q["query"]
        print(f"[{i}/{total}] {qid}: {query_text[:60]}")

        try:
            parsed, rows_by_model = run_pipeline(query_text)
            print(f"  → parsed occasion: {(parsed.get('occasion') or {}).get('target')} | "
                  f"clip={len(rows_by_model.get('clip', []))} fc={len(rows_by_model.get('fashion_clip', []))}")

            eval_result = evaluate_query_result(query_text, parsed, rows_by_model)

            # Log parser scores
            parser_scores = {k: v.get("score") for k, v in eval_result.get("parser", {}).items()}
            print(f"  → parser scores: {parser_scores}")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            parsed, rows_by_model, eval_result = {}, {}, {"error": str(exc)}

        per_query_results.append({
            "id": qid,
            "query": query_text,
            "type": q.get("type", []),
            "parsed": parsed,
            "rows_by_model": {
                model: [
                    {
                        "product_name": r.get("product_name"),
                        "product_category": r.get("product_category"),
                        "color": r.get("color"),
                        "price": r.get("price"),
                        "explanation": r.get("explanation", ""),
                        "image_url": r.get("image_url"),
                        "product_url": r.get("product_url"),
                        "final_score": r.get("final_score"),
                        "relevance_score": r.get("relevance_score"),
                        "occasion_score": r.get("occasion_score"),
                    }
                    for r in rows
                ]
                for model, rows in rows_by_model.items()
            },
            "eval": eval_result,
        })

    aggregate = aggregate_results(per_query_results)

    output = {
        "run_timestamp": timestamp,
        "queries_file": str(queries_path),
        "total_queries": total,
        "aggregate": aggregate,
        "per_query": per_query_results,
    }

    out_file = out_dir / f"eval_{timestamp}.json"
    out_file.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to: {out_file}")

    # Also save a flat CSV for quick inspection
    _save_aggregate_csv(aggregate, out_dir / f"aggregate_{timestamp}.csv")
    print(f"Aggregate CSV: {out_dir / f'aggregate_{timestamp}.csv'}")


def _save_aggregate_csv(aggregate: dict, path: Path) -> None:
    rows = []
    for scope, metrics in aggregate.items():
        if scope == "cross_model":
            rows.append({"scope": "cross_model", "metric": "clip_wins", "mean": metrics.get("clip_wins"), "std": None, "n": metrics.get("total")})
            rows.append({"scope": "cross_model", "metric": "fashionclip_wins", "mean": metrics.get("fashionclip_wins"), "std": None, "n": metrics.get("total")})
            rows.append({"scope": "cross_model", "metric": "ties", "mean": metrics.get("ties"), "std": None, "n": metrics.get("total")})
        else:
            for metric_name, stats in metrics.items():
                rows.append({"scope": scope, "metric": metric_name, **stats})
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fashion RecSys LLM-as-a-Judge batch evaluator")
    parser.add_argument("--queries", type=Path, default=ROOT / "evaluation" / "test_queries.json")
    parser.add_argument("--out", type=Path, default=ROOT / "evaluation" / "results")
    args = parser.parse_args()
    run(args.queries, args.out)
