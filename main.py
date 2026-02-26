import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from online.candidate_retrieval import (
    MODEL_PATHS,
    PARQUET_PATH,
    retrieve_candidates,
)
from online.explanation_generation_groq import generate_explanations
from online.final_ranking import rank_candidates
from online.occasion_suitability_scores import (
    MODEL_OCCASION_EMBEDDINGS_PATHS,
    compute_occasion_scores,
)
from online.query_processing_llm import parse_query_llm


def main() -> None:
    parser = argparse.ArgumentParser(description="Online retrieval pipeline")
    parser.add_argument("--query")
    parser.add_argument("--test-query", action="store_true")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--filter-first", action="store_true")
    parser.add_argument(
        "--embedding-model",
        choices=["clip", "fashion_clip", "both"],
        default="both",
    )
    parser.add_argument("--out", default="online_results.csv")
    parser.add_argument("--no-faiss", action="store_true")
    args = parser.parse_args()

    if args.test_query:
        args.query = "I want something to wear for a party night out with friends, I like dresses and skirts, I want to look elegant but also feel comfortable, I want to wear something red or black" 
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

    parsed = parse_query_llm(args.query)
    candidates_by_model = retrieve_candidates(
        args.query,
        parsed=parsed,
        topk=args.topk,
        filter_first=args.filter_first,
        use_faiss=not args.no_faiss,
        embedding_model=args.embedding_model,
    )

    print("Parsed query:")
    print(parsed)
    total = sum(len(cands) for cands in candidates_by_model.values())
    print(f"\nFound {total} candidates across {len(candidates_by_model)} model(s)")
    for model_name, cands in candidates_by_model.items():
        print(f"  - {model_name}: {len(cands)}")

    if total == 0:
        return

    df = pd.read_parquet(PARQUET_PATH)
    df["row_id"] = df["row_id"].astype(str)
    df = df.set_index("row_id")

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

        print(f"\n[{model_name}] ranked candidates: {len(ranked)}")
        rows: list[dict] = []
        for row_id, relevance, occ_score, final_score in ranked:
            print(f"{model_name} :: {row_id} -> occasion_score={occ_score:.4f}")
            if row_id not in df.index:
                continue
            row = df.loc[row_id]
            rows.append(
                {
                    "model_name": model_name,
                    "row_id": row_id,
                    "relevance_score": relevance,
                    "occasion_score": occ_score,
                    "final_score": final_score,
                    "product_name": row.get("product_name", ""),
                    "product_description": row.get("product_description", ""),
                    "product_url": row.get("product_url", ""),
                    "price": row.get("price", ""),
                    "color": row.get("color", ""),
                    "product_category": row.get("product_category", ""),
                    "image_url": row.get("image_url", ""),
                }
            )

        explanations = generate_explanations(
            rows,
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
        for row in rows:
            row["explanation"] = explanations.get(str(row["row_id"]), "")

        rows_by_model[model_name] = rows

    if not rows_by_model:
        print("No rows could be materialized from ranked candidates.")
        return

    base_name = args.out
    if base_name == "online_results.csv":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"online_results_{stamp}.csv"
    stem = Path(base_name).stem
    suffix = Path(base_name).suffix or ".csv"
    results_dir = Path("results_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    for model_name, rows in rows_by_model.items():
        out_path = results_dir / f"{stem}_{model_name}{suffix}"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
