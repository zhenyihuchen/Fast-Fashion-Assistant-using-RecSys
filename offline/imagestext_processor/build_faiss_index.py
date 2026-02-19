from pathlib import Path

import numpy as np


EMBEDDINGS_PATH = Path(__file__).resolve().parent.parent / "data" / "image_embeddings" / "image_embeddings.npy"
INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "image_embeddings" / "faiss.index"


def main() -> None:
    try:
        import faiss  # type: ignore
    except Exception:
        raise SystemExit("faiss not installed; run `pip install faiss-cpu`")

    embeddings = np.load(EMBEDDINGS_PATH)
    if embeddings.ndim != 2:
        raise SystemExit(f"Expected 2D embeddings, got shape {embeddings.shape}")

    embeddings = embeddings.astype("float32", copy=False)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    print(f"Saved FAISS index to {INDEX_PATH}")
    print(f"Index size: {index.ntotal} vectors of dim {dim}")


if __name__ == "__main__":
    main()
