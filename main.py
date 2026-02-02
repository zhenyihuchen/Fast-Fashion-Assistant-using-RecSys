import argparse

import pandas as pd

from online.candidate_retrieval import PARQUET_PATH, retrieve_candidates
from online.query_processing_llm import parse_query_llm


def main() -> None:
    parser = argparse.ArgumentParser(description="Online retrieval pipeline")
    parser.add_argument("--query")
    parser.add_argument("--test-query", action="store_true")
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--filter-first", action="store_true")
    parser.add_argument("--out", default="online_results.csv")
    parser.add_argument("--no-faiss", action="store_true")
    args = parser.parse_args()

    if args.test_query:
        args.query = "I am looking for a night short red dress but that costs less than 50 euros"
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

    df = pd.read_parquet(PARQUET_PATH)
    df["row_id"] = df["row_id"].astype(str)
    df = df.set_index("row_id")

    rows = []
    for row_id, score in candidates:
        if row_id in df.index:
            row = df.loc[row_id]
            rows.append(
                {
                    "row_id": row_id,
                    "score": score,
                    "product_name": row.get("product_name", ""),
                    "product_url": row.get("product_url", ""),
                    "price": row.get("price", ""),
                    "color": row.get("color", ""),
                    "product_category": row.get("product_category", ""),
                    "image_url": row.get("image_url", ""),
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
