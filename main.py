import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from online.candidate_retrieval import (
    MODEL_PATHS,
    PARQUET_PATH,
    retrieve_candidates,
)
from online.explanation_generation_llm import generate_explanations
from online.final_ranking import rank_candidates
from online.occasion_suitability_scores import (
    MODEL_OCCASION_EMBEDDINGS_PATHS,
    compute_occasion_scores,
)
from online.query_processing_llm import parse_query_llm

IMAGES_DIR = Path("offline/data/images")


def _enrich_row(row_id: str, df: pd.DataFrame) -> dict:
    """Build a product metadata dict from the catalog."""
    row = df.loc[row_id]
    return {
        "row_id": row_id,
        "product_name": str(row.get("product_name", "") or ""),
        "product_category": str(row.get("product_category", "") or ""),
        "product_description": str(row.get("product_description", "") or ""),
        "color": str(row.get("color", "") or ""),
        "price": str(row.get("price", "") or ""),
        "product_url": str(row.get("product_url", "") or ""),
        "image_url": str(row.get("image_url", "") or ""),
        "local_image_path": str(
            IMAGES_DIR / (str(row.get("reference_number", "") or "").replace("/", "_") + ".jpg")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Online retrieval pipeline")
    parser.add_argument("--query")
    parser.add_argument("--test-query", action="store_true")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--filter-first", action="store_true")
    parser.add_argument(
        "--embedding-model",
        choices=["clip", "fashion_clip", "both"],
        default="both",
    )
    parser.add_argument("--no-faiss", action="store_true")
    args = parser.parse_args()

    if args.test_query:
        args.query = "I am looking for a night out short black dress but that costs less than 50 euros"
            # I want something to wear for a party night out with friends, I like dresses and skirts, I want to look elegant but also feel comfortable, I want to wear something red or black" 
            # "I want a pair of dark blue color jeans for a chill and casual day in the park, I like the cut to be wide-leg"
            # "I want a pair of dark blue color jeans for a chill and casual day in the park, I like the cut to be wide-leg"
            # #"I am looking for a night short red dress but that costs less than 50 euros"
            # " "I want something to wear for a party night out with friends, I like dresses and skirts, I want to look elegant but also feel comfortable, I want to wear something red or black""
            # "I want something to wear for an Ibicenca party this summer"
            # "I want a pair of black jeans for a chill and casual day in the park, I like the cut to be straight"
            # "I am looking for a night short red dress but that costs less than 50 euros"
            #"I want a pair of dark blue color jeans for a chill and casual day in the park, I like the cut to be straight"
            #"I want a pair of wine color jeans for a chill and casual day in the park, I like the cut to be straight",
            # "I want a pair of dark blue color straight jeans for a chill and casual day in the park, high-rise" (para mejorar)
            
    if not args.query:
        raise SystemExit("Provide --query or use --test-query")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results_test") / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Parse query ──────────────────────────────────────────────────
    parsed = parse_query_llm(args.query)
    print("Parsed query:")
    print(json.dumps(parsed, indent=2, ensure_ascii=False))

    # ── Step 2: Retrieve candidates ──────────────────────────────────────────
    candidates_by_model = retrieve_candidates(
        args.query,
        parsed=parsed,
        topk=args.topk,
        filter_first=args.filter_first,
        use_faiss=not args.no_faiss,
        embedding_model=args.embedding_model,
        return_metadata=True,
    )

    total = sum(len(result.get("candidates", [])) for result in candidates_by_model.values())
    print(f"\nFound {total} candidates across {len(candidates_by_model)} model(s)")
    for model_name, result in candidates_by_model.items():
        print(f"  - {model_name}: {len(result.get('candidates', []))} [{result.get('match_stage', 'strict')}]")

    if total == 0:
        return

    df = pd.read_parquet(PARQUET_PATH)
    df["row_id"] = df["row_id"].astype(str)
    df = df.set_index("row_id")

    # ── Per-model processing ─────────────────────────────────────────────────
    output: dict = {
        "query": args.query,
        "timestamp": timestamp,
        "parsed": parsed,
        "models": {},
    }

    for model_name, result in candidates_by_model.items():
        candidates = result.get("candidates", [])
        if not candidates:
            continue
        match_stage = result.get("match_stage", "strict")
        match_message = result.get("match_message", "")

        paths = MODEL_PATHS[model_name]
        product_ids = np.load(paths["image_ids"], allow_pickle=True).astype(str)
        embeddings = np.load(paths["image_embeddings"]).astype("float32", copy=False)

        # ── Step 3: Relevance-only ranking (top-k by embedding similarity) ───
        relevance_ranked = rank_candidates(candidates, {}, alpha=1.0, beta=0.0)
        retrieval_items = []
        for row_id, relevance, _, _ in relevance_ranked:
            if row_id not in df.index:
                continue
            item = _enrich_row(row_id, df)
            item["relevance_score"] = float(relevance)
            item["match_stage"] = match_stage
            item["match_message"] = match_message
            retrieval_items.append(item)

        # ── Step 4: Occasion scoring + final reranking ───────────────────────
        occasion_scores = compute_occasion_scores(
            candidates,
            parsed=parsed,
            product_ids=product_ids,
            embeddings=embeddings,
            model_name=model_name,
            occasion_embeddings_path=MODEL_OCCASION_EMBEDDINGS_PATHS.get(model_name),
        )
        final_ranked = rank_candidates(candidates, occasion_scores)
        reranked_items = []
        for row_id, relevance, occ_score, final_score in final_ranked:
            if row_id not in df.index:
                continue
            item = _enrich_row(row_id, df)
            item["relevance_score"] = float(relevance)
            item["occasion_score"] = float(occ_score)
            item["final_score"] = float(final_score)
            item["match_stage"] = match_stage
            item["match_message"] = match_message
            reranked_items.append(item)

        # ── Step 5: Explanations for top 5 ───────────────────────────────────
        top5 = reranked_items[:5]
        explanations = generate_explanations(
            top5,
            candidates=candidates,
            parsed=parsed,
            product_ids=product_ids,
            embeddings=embeddings,
            occasion_scores=occasion_scores,
            model_name=model_name,
            occasion_embeddings_path=MODEL_OCCASION_EMBEDDINGS_PATHS.get(
                model_name,
                MODEL_OCCASION_EMBEDDINGS_PATHS["clip"],
            ),
        )
        for item in top5:
            item["explanation"] = explanations.get(str(item["row_id"]), "")

        print(f"\n[{model_name}] retrieval: {len(retrieval_items)} | "
              f"reranked: {len(reranked_items)} | top5 with explanations")

        output["models"][model_name] = {
            "retrieval_relevance_only": retrieval_items,
            "reranked_with_occasion": reranked_items,
            "top5": top5,
        }

    # ── Save everything ──────────────────────────────────────────────────────
    out_path = results_dir / "pipeline_results.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull results saved to: {out_path}")

    # Also save top-5 CSVs for quick inspection
    for model_name, model_data in output["models"].items():
        csv_path = results_dir / f"top5_{model_name}.csv"
        pd.DataFrame(model_data["top5"]).to_csv(csv_path, index=False)
        print(f"Top-5 CSV: {csv_path}")


if __name__ == "__main__":
    main()
