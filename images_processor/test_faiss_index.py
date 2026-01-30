from pathlib import Path
import argparse
import random

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
EMBED_DIR = DATA_DIR / "image_embeddings"

EMBEDDINGS_PATH = EMBED_DIR / "image_embeddings.npy"
IDS_PATH = EMBED_DIR / "product_ids.npy"
INDEX_PATH = EMBED_DIR / "faiss.index"


def main() -> None:
    parser = argparse.ArgumentParser(description="Test FAISS index with a sample query.")
    parser.add_argument("--row-id", help="Row id to query; defaults to random.")
    parser.add_argument("--topk", type=int, default=5, help="Number of neighbors to return.")
    args = parser.parse_args()

    try:
        import faiss  # type: ignore
    except Exception:
        raise SystemExit("faiss not installed; run `pip install faiss-cpu`")

    embeddings = np.load(EMBEDDINGS_PATH).astype("float32", copy=False)
    product_ids = np.load(IDS_PATH, allow_pickle=True).astype(str)

    if embeddings.shape[0] != len(product_ids):
        raise SystemExit("Embeddings and product_ids length mismatch")

    if not INDEX_PATH.exists():
        raise SystemExit(f"Index not found at {INDEX_PATH}")

    index = faiss.read_index(str(INDEX_PATH))

    if args.row_id:
        try:
            query_idx = int(np.where(product_ids == args.row_id)[0][0])
        except Exception:
            raise SystemExit(f"row_id {args.row_id} not found in product_ids")
    else:
        query_idx = random.randrange(len(product_ids))

    query_vec = embeddings[query_idx : query_idx + 1]
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, args.topk)
    print(f"Query row_id: {product_ids[query_idx]}")
    print("Top neighbors:")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        print(f"{rank}. row_id={product_ids[idx]} score={score:.4f}")


if __name__ == "__main__":
    main()
