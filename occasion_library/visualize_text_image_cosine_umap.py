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
    parser = argparse.ArgumentParser(description="UMAP/t-SNE with cosine distance for text+image embeddings.")
    parser.add_argument("--method", default="umap", choices=["umap", "tsne"])
    parser.add_argument("--html", default="text_image_cosine.html")
    parser.add_argument("--label", default="product_category", choices=["product_category", "color"])
    args = parser.parse_args()

    try:
        import plotly.express as px
    except Exception:
        raise SystemExit("plotly not installed; run `pip install plotly`")

    text_data = np.load(TEXT_NPZ)
    text_vectors = []
    text_labels = []
    for occasion in text_data.files:
        emb = text_data[occasion]
        text_vectors.append(emb)
        text_labels.extend([occasion] * emb.shape[0])
    if not text_vectors:
        raise SystemExit("No text embeddings found")
    text_vectors = np.vstack(text_vectors)

    image_vectors = np.load(IMAGE_EMB)
    image_ids = np.load(IMAGE_IDS, allow_pickle=True).astype(str)

    df = pd.read_parquet(PARQUET)
    if "row_id" not in df.columns:
        raise SystemExit("row_id column not found in parquet")
    df["row_id"] = df["row_id"].astype(str)
    df = df.set_index("row_id")

    image_labels = []
    keep_idx = []
    for i, pid in enumerate(image_ids):
        if pid in df.index:
            image_labels.append(str(df.loc[pid, args.label]))
            keep_idx.append(i)
    if not keep_idx:
        raise SystemExit("No matching ids found between embeddings and parquet")
    image_vectors = image_vectors[keep_idx]
    image_ids = image_ids[keep_idx]

    X = np.vstack([image_vectors, text_vectors]).astype("float32", copy=False)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    if args.method == "umap":
        try:
            import umap  # type: ignore
        except Exception:
            raise SystemExit("umap-learn not installed; run `pip install umap-learn`")
        reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42)
        X2 = reducer.fit_transform(X)
    else:
        from sklearn.manifold import TSNE

        X2 = TSNE(n_components=2, metric="cosine", random_state=42).fit_transform(X)

    n_img = image_vectors.shape[0]
    img_xy = X2[:n_img]
    txt_xy = X2[n_img:]

    img_df = pd.DataFrame(
        {
            "x": img_xy[:, 0],
            "y": img_xy[:, 1],
            "type": "image",
            "label": image_labels,
            "row_id": image_ids,
        }
    )
    txt_df = pd.DataFrame(
        {
            "x": txt_xy[:, 0],
            "y": txt_xy[:, 1],
            "type": "text",
            "label": text_labels,
            "row_id": "",
        }
    )

    plot_df = pd.concat([img_df, txt_df], ignore_index=True)
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="label",
        symbol="type",
        hover_data=["type", "label", "row_id"],
        title=f"Text + Image Embeddings ({args.method.upper()}, cosine)",
    )
    fig.write_html(args.html)
    print(f"Saved {args.html}")


if __name__ == "__main__":
    main()
