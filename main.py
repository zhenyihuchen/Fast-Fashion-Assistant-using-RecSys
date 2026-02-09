import argparse

import pandas as pd
import numpy as np

from online.candidate_retrieval import (
    EMBEDDINGS_PATH,
    IDS_PATH,
    PARQUET_PATH,
    retrieve_candidates,
)
from online.explanation_generation import generate_explanations
from online.final_ranking import rank_candidates
from online.occasion_suitability_scores import compute_occasion_scores
from online.query_processing_llm import parse_query_llm


def main() -> None:
    parser = argparse.ArgumentParser(description="Online retrieval pipeline")
    parser.add_argument("--query")
    parser.add_argument("--test-query", action="store_true")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--filter-first", action="store_true")
    parser.add_argument("--out", default="online_results.csv")
    parser.add_argument("--no-faiss", action="store_true")
    args = parser.parse_args()

    if args.test_query:
        args.query =  "I want something to wear for a party night out with friends, I like dresses and skirts, I want to look elegant but also feel comfortable, I want to wear something red or black"
            # "I want something to wear for an Ibicenca party this summer"
            # "I want a pair of black jeans for a chill and casual day in the park, I like the cut to be straight"
            # "I am looking for a night short red dress but that costs less than 50 euros"
            #"I want a pair of dark blue color jeans for a chill and casual day in the park, I like the cut to be straight"
            #"I want a pair of wine color jeans for a chill and casual day in the park, I like the cut to be straight",
            
    if not args.query:
        raise SystemExit("Provide --query or use --test-query")

    parsed = parse_query_llm(args.query)
    candidates = retrieve_candidates(
        args.query,
        parsed=parsed,
        topk=args.topk,
        filter_first=args.filter_first,
        use_faiss=not args.no_faiss,
    )

    print("Parsed query:")
    print(parsed)
    print(f"\nFound {len(candidates)} candidates")

    if not candidates:
        return

    product_ids = np.load(IDS_PATH, allow_pickle=True).astype(str)
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32", copy=False)
    occasion_scores = compute_occasion_scores(
        candidates,
        parsed=parsed,
        product_ids=product_ids,
        embeddings=embeddings,
    )
    ranked = rank_candidates(candidates, occasion_scores)

    df = pd.read_parquet(PARQUET_PATH)
    df["row_id"] = df["row_id"].astype(str)
    df = df.set_index("row_id")

    rows = []
    for row_id, relevance, occ_score, final_score in ranked:
        print(f"{row_id} -> occasion_score={occ_score:.4f}")
        if row_id in df.index:
            row = df.loc[row_id]
            rows.append(
                {
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
    )
    for row in rows:
        row["explanation"] = explanations.get(str(row["row_id"]), "")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
