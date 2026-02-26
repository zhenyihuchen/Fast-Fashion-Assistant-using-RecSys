from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "image_embeddings"

MODELS = {
    "clip": BASE_DIR / "clip",
    "fashion_clip": BASE_DIR / "fashion_clip",
}


def build_index(embeddings_dir: Path) -> None:
    try:
        import faiss  # type: ignore
    except Exception:
        raise SystemExit("faiss not installed; run `pip install faiss-cpu`")

    embeddings_path = embeddings_dir / "image_embeddings.npy"
    index_path = embeddings_dir / "faiss.index"

    if not embeddings_path.exists():
        print(f"  Skipping {embeddings_dir.name}: {embeddings_path} not found")
        return

    embeddings = np.load(embeddings_path)
    if embeddings.ndim != 2:
        raise SystemExit(f"Expected 2D embeddings, got shape {embeddings.shape}")

    embeddings = embeddings.astype("float32", copy=False)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, str(index_path))
    print(f"  Saved {index_path}  ({index.ntotal} vectors, dim={dim})")


def main() -> None:
    for name, directory in MODELS.items():
        print(f"[{name}] Building FAISS index …")
        build_index(directory)


if __name__ == "__main__":
    main()
