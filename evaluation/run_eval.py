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
  
(done) python -m evaluation.run_eval --start 0 --end 30     # day 1 (queries 0-29)
(done) python -m evaluation.run_eval --start 30 --end 60    # day 2
(done) python -m evaluation.run_eval --start 60 --end 90    # day 3
(done) python -m evaluation.run_eval --start 90 --end 120   # day 4
python -m evaluation.run_eval --start 120 --end 150  # day 5

"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from online.candidate_retrieval import MODEL_PATHS, PARQUET_PATH, retrieve_candidates
from online.explanation_generation_llm import generate_explanations
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


def _random_products(n: int = 5) -> list[dict]:
    """Sample n random products from the catalog (random baseline)."""
    df = _get_catalog()
    sample = df.sample(n=n)
    rows: list[dict] = []
    for row_id, row in sample.iterrows():
        rows.append({
            "row_id": str(row_id),
            "relevance_score": 0.0,
            "occasion_score": 0.0,
            "final_score": 0.0,
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
            "explanation": "",
        })
    return rows


# ── Pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline(
    query: str,
    disable_occasion: bool = False,
    no_pipeline: bool = False,
    alpha: float = 0.6,
    beta: float = 0.4,
) -> tuple[dict, dict[str, list[dict]]]:
    """Run the full recommendation pipeline for one query.

    Args:
        query: The user query string.
        disable_occasion: If True, skip occasion scoring and rank purely by
            relevance (α=1.0, β=0.0) even when the parser detects an occasion.
        no_pipeline: If True, skip query parsing, metadata filtering, occasion
            scoring, and explanation generation. Returns raw embedding similarity
            results only (baseline for ablation).
        alpha: Weight for relevance score in final ranking.
        beta: Weight for occasion score in final ranking.

    Returns:
        (parsed, rows_by_model)
        where rows_by_model = {"clip": [...], "fashion_clip": [...]}
    """
    if no_pipeline:
        return _run_no_pipeline(query)

    parsed = parse_query_llm(query)
    occ = parsed.get("occasion") or {}
    has_occasion = (
        not disable_occasion
        and occ.get("mode") == "on"
        and bool(occ.get("target"))
    )

    candidates_by_model = retrieve_candidates(
        query,
        parsed=parsed,
        topk=30,
        filter_first=True,
        use_faiss=True,
        embedding_model="both",
        return_metadata=True,
    )
    if not any(result.get("candidates") for result in candidates_by_model.values()):
        return parsed, {}

    df = _get_catalog()
    rows_by_model: dict[str, list[dict]] = {}

    for model_name, result in candidates_by_model.items():
        candidates = result.get("candidates", [])
        if not candidates:
            continue
        match_stage = result.get("match_stage", "strict")
        match_message = result.get("match_message", "")

        paths = MODEL_PATHS[model_name]
        product_ids = np.load(paths["image_ids"], allow_pickle=True).astype(str)
        embeddings = np.load(paths["image_embeddings"]).astype("float32", copy=False)

        if has_occasion:
            occasion_scores = compute_occasion_scores(
                candidates,
                parsed=parsed,
                product_ids=product_ids,
                embeddings=embeddings,
                model_name=model_name,
                occasion_embeddings_path=MODEL_OCCASION_EMBEDDINGS_PATHS.get(model_name),
            )
            ranked = rank_candidates(candidates, occasion_scores, alpha=alpha, beta=beta)
        else:
            # No occasion: skip occasion scoring, rank purely by retrieval relevance
            occasion_scores = {}
            ranked = rank_candidates(candidates, occasion_scores, alpha=1.0, beta=0.0)

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
                "match_stage": match_stage,
                "match_message": match_message,
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


def _run_no_pipeline(query: str) -> tuple[dict, dict[str, list[dict]]]:
    """Baseline: raw embedding similarity only — no filtering, no occasion
    scoring, no explanations.  The query is still parsed so the judge can
    evaluate occasion appropriateness against the detected occasion."""
    parsed = parse_query_llm(query)

    candidates_by_model = retrieve_candidates(
        query,
        parsed={},           # empty → no constraint filtering
        topk=30,
        filter_first=False,  # no metadata pre-filtering
        use_faiss=True,
        embedding_model="both",
        return_metadata=True,
    )
    if not any(result.get("candidates") for result in candidates_by_model.values()):
        return parsed, {}

    df = _get_catalog()
    rows_by_model: dict[str, list[dict]] = {}

    for model_name, result in candidates_by_model.items():
        candidates = result.get("candidates", [])
        if not candidates:
            continue

        # Rank by raw similarity only (alpha=1, beta=0, no occasion scores)
        ranked = rank_candidates(candidates, {}, alpha=1.0, beta=0.0)

        rows: list[dict] = []
        for row_id, relevance, occ_score, final_score in ranked:
            if row_id not in df.index:
                continue
            row = df.loc[row_id]
            rows.append({
                "row_id": row_id,
                "relevance_score": float(relevance),
                "occasion_score": 0.0,
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
                "match_stage": "no_pipeline",
                "match_message": "Raw embedding similarity — no filtering or reranking",
                "explanation": "",
            })

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

    def _eval_ok(q: dict) -> dict:
        """Return the eval dict only if it is a proper result (not an error stub)."""
        ev = q.get("eval", {})
        if not isinstance(ev, dict) or "error" in ev:
            return {}
        return ev

    def collect_item_scores(model_key: str, rubric_name: str) -> list:
        scores = []
        for q in per_query:
            model_data = _eval_ok(q).get(model_key, {})
            if not isinstance(model_data, dict):
                continue
            for item_result in model_data.get("items", []):
                if not isinstance(item_result, dict):
                    continue
                s = item_result.get(rubric_name, {}).get("score")
                if s is not None:
                    scores.append(s)
        return scores

    def collect_set_scores(model_key: str, rubric_name: str) -> list:
        scores = []
        for q in per_query:
            ev = _eval_ok(q)
            model_data = ev.get(model_key, {})
            if not isinstance(model_data, dict):
                continue
            set_data = model_data.get("set", {})
            if not isinstance(set_data, dict):
                continue
            s = set_data.get(rubric_name, {}).get("score")
            if s is not None:
                scores.append(s)
        return scores

    def collect_parser_scores(rubric_name: str) -> list:
        scores = []
        for q in per_query:
            parser_data = _eval_ok(q).get("parser", {})
            if not isinstance(parser_data, dict):
                continue
            s = parser_data.get(rubric_name, {}).get("score")
            if s is not None:
                scores.append(s)
        return scores

    item_rubric_names = [
        "item_relevance", "occasion_appropriateness", "explanation_quality",
    ]
    set_rubric_names = ["set_answer_relevance"]
    parser_rubric_names = ["parser_completeness", "parser_no_hallucination", "parser_occasion_detection"]

    # Collect all model keys dynamically (clip, fashion_clip, random, ...)
    model_keys = set()
    for q in per_query:
        ev = _eval_ok(q)
        for key in ev:
            if key not in ("parser", "cross_model", "error"):
                model_keys.add(key)
    model_keys = sorted(model_keys)

    agg: dict = {k: {} for k in model_keys}
    agg["parser"] = {}
    agg["cross_model"] = {}

    for model_key in model_keys:
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
    winners = [_eval_ok(q).get("cross_model", {}).get("winner") for q in per_query]
    agg["cross_model"] = {
        "clip_wins": winners.count("clip"),
        "fashionclip_wins": winners.count("fashion_clip"),
        "ties": winners.count("tie"),
        "total": len([w for w in winners if w in {"clip", "fashion_clip", "tie"}]),
    }

    return agg


# ── Main runner ──────────────────────────────────────────────────────────────

def _load_query_ids(ids_file: Path) -> list[str]:
    return [
        line.strip()
        for line in ids_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def run(
    queries_path: Path,
    out_dir: Path,
    start: int = 0,
    end: int | None = None,
    ids_file: Path | None = None,
    disable_occasion: bool = False,
    no_pipeline: bool = False,
    alpha: float = 0.6,
    beta: float = 0.4,
) -> None:
    all_queries = json.loads(queries_path.read_text(encoding="utf-8"))
    if ids_file is not None:
        requested_ids = _load_query_ids(ids_file)
        requested_set = set(requested_ids)
        queries = [q for q in all_queries if q.get("id") in requested_set]
        found_ids = {q.get("id") for q in queries}
        missing_ids = [qid for qid in requested_ids if qid not in found_ids]
        print(
            f"Running {len(queries)} queries from ids file {ids_file} "
            f"({len(requested_ids)} requested, {len(all_queries)} total available)"
        )
        if missing_ids:
            print(f"Warning: {len(missing_ids)} query ids were not found: {', '.join(missing_ids)}")
    else:
        queries = all_queries[start:end]
        print(f"Running queries [{start}:{end or len(all_queries)}] "
              f"({len(queries)} of {len(all_queries)} total)")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    per_query_results: list[dict] = []
    total = len(queries)
    elapsed_times: list[float] = []
    run_start = time.monotonic()

    for i, q in enumerate(queries, 1):
        qid = q.get("id", f"q{i:03d}")
        query_text = q["query"]
        query_start = time.monotonic()
        print(f"\n[{i}/{total}] {qid}: {query_text[:60]}")

        try:
            t0 = time.monotonic()
            parsed, rows_by_model = run_pipeline(
                query_text,
                disable_occasion=disable_occasion,
                no_pipeline=no_pipeline,
                alpha=alpha,
                beta=beta,
            )
            pipeline_s = time.monotonic() - t0

            # Add random baseline (no pipeline needed)
            rows_by_model["random"] = _random_products(5)

            print(f"  → pipeline: {pipeline_s:.1f}s | "
                  f"occasion: {(parsed.get('occasion') or {}).get('target')} | "
                  f"clip={len(rows_by_model.get('clip', []))} "
                  f"fc={len(rows_by_model.get('fashion_clip', []))} "
                  f"random={len(rows_by_model.get('random', []))}")

            t0 = time.monotonic()
            eval_result = evaluate_query_result(query_text, parsed, rows_by_model)
            eval_s = time.monotonic() - t0

            parser_scores = {k: v.get("score") for k, v in eval_result.get("parser", {}).items()}
            print(f"  → eval: {eval_s:.1f}s | parser scores: {parser_scores}")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            parsed, rows_by_model, eval_result = {}, {}, {"error": str(exc)}

        query_elapsed = time.monotonic() - query_start
        elapsed_times.append(query_elapsed)
        avg_s = sum(elapsed_times) / len(elapsed_times)
        remaining = total - i
        eta_s = avg_s * remaining
        total_elapsed = time.monotonic() - run_start
        print(f"  → query done in {query_elapsed:.1f}s | "
              f"total elapsed: {timedelta(seconds=int(total_elapsed))} | "
              f"ETA: {timedelta(seconds=int(eta_s))} ({remaining} left)")

        per_query_results.append({
            "id": qid,
            "query": query_text,
            "tier": q.get("tier"),
            "tags": q.get("tags", []),
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
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive) for query slice")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive) for query slice")
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=None,
        help="Path to a text file containing one query id per line; overrides --start/--end slicing",
    )
    parser.add_argument(
        "--no-occasion",
        action="store_true",
        default=False,
        help="Disable occasion scoring; rank purely by relevance (alpha=1.0, beta=0.0)",
    )
    parser.add_argument(
        "--no-pipeline",
        action="store_true",
        default=False,
        help="Baseline: raw embedding similarity only — no parsing, filtering, occasion, or explanations",
    )
    parser.add_argument("--alpha", type=float, default=0.6, help="Relevance weight (default 0.6)")
    parser.add_argument("--beta", type=float, default=0.4, help="Occasion weight (default 0.4)")
    args = parser.parse_args()
    run(
        args.queries, args.out,
        start=args.start, end=args.end,
        ids_file=args.ids_file,
        disable_occasion=args.no_occasion,
        no_pipeline=args.no_pipeline,
        alpha=args.alpha, beta=args.beta,
    )
