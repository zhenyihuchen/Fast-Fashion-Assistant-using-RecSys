import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
EMBED_DIR = DATA_DIR / "image_embeddings"

TEXT_NPZ = BASE_DIR / "occasion_prompt_embeddings.npz"
IMAGE_EMB = EMBED_DIR / "image_embeddings.npy"
IMAGE_IDS = EMBED_DIR / "product_ids.npy"
PARQUET = DATA_DIR / "women_data.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Find top images for each prompt in an occasion.")
    parser.add_argument("--occasion", default="party_night_out")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    text_data = np.load(TEXT_NPZ)
    if args.occasion not in text_data.files:
        raise SystemExit(f"Occasion not found: {args.occasion}")
    prompt_emb = text_data[args.occasion]

    image_emb = np.load(IMAGE_EMB).astype("float32", copy=False)
    image_ids = np.load(IMAGE_IDS, allow_pickle=True).astype(str)

    if image_emb.shape[0] != len(image_ids):
        raise SystemExit("Embeddings and product_ids length mismatch")

    df = pd.read_parquet(PARQUET)
    if "row_id" not in df.columns:
        raise SystemExit("row_id column not found in parquet")
    df["row_id"] = df["row_id"].astype(str)
    df = df.set_index("row_id")

    rows = []
    for p_idx, p_vec in enumerate(prompt_emb):
        scores = image_emb @ p_vec
        top_idx = np.argsort(scores)[-args.topk :][::-1]
        for rank, idx in enumerate(top_idx, 1):
            row_id = image_ids[idx]
            if row_id in df.index:
                row = df.loc[row_id]
                rows.append(
                    {
                        "occasion": args.occasion,
                        "prompt_index": p_idx,
                        "rank": rank,
                        "score": float(scores[idx]),
                        "row_id": row_id,
                        "product_url": row.get("product_url", ""),
                        "product_name": row.get("product_name", ""),
                        "image_url": row.get("image_url", ""),
                        "product_category": row.get("product_category", ""),
                        "color": row.get("color", ""),
                    }
                )

    out_path = BASE_DIR / f"{args.occasion}_recs.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
