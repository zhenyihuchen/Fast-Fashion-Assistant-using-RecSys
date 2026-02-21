import argparse
from pathlib import Path
import webbrowser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load_umap():
    try:
        import umap  # type: ignore
    except Exception:
        return None
    return umap


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / "data"
    embed_dir = data_dir / "image_embeddings"

    parser = argparse.ArgumentParser(description="Visualize CLIP embeddings in 2D.")
    parser.add_argument("--embeddings", default=str(embed_dir / "image_embeddings.npy"))
    parser.add_argument("--ids", default=str(embed_dir / "product_ids.npy"))
    parser.add_argument("--parquet", default=str(data_dir / "women_data.parquet"))
    parser.add_argument("--label", default="product_category", choices=["product_category", "color"])
    parser.add_argument("--method", default="pca", choices=["pca", "umap"])
    parser.add_argument("--out", default="embeddings_2d.png")
    parser.add_argument("--show", action="store_true", help="Show interactive plot window")
    parser.add_argument("--legend", action="store_true", help="Include a legend (may be large)")
    parser.add_argument("--hover", action="store_true", help="Enable hover tooltips (requires mplcursors)")
    parser.add_argument("--plotly", action="store_true", help="Generate interactive Plotly HTML")
    parser.add_argument("--html", default="embeddings_2d.html")
    parser.add_argument("--open", action="store_true", help="Open the HTML in the default browser")
    args = parser.parse_args()

    embeddings = np.load(args.embeddings)
    product_ids = np.load(args.ids, allow_pickle=True).astype(str)

    df = pd.read_parquet(args.parquet)
    if "row_id" not in df.columns:
        raise SystemExit("row_id column not found in parquet")

    df["row_id"] = df["row_id"].astype(str)
    df = df.set_index("row_id")

    labels = []
    keep_idx = []
    for i, pid in enumerate(product_ids):
        if pid in df.index:
            labels.append(str(df.loc[pid, args.label]))
            keep_idx.append(i)

    if not keep_idx:
        raise SystemExit("No matching ids found between embeddings and parquet")

    X = embeddings[keep_idx]
    labels = np.array(labels)

    if args.method == "umap":
        umap = _load_umap()
        if umap is None:
            raise SystemExit("umap-learn not installed; use --method pca or install umap-learn")
        reducer = umap.UMAP(n_components=2, random_state=42)
        X2 = reducer.fit_transform(X)
    else:
        from sklearn.decomposition import PCA

        X2 = PCA(n_components=2, random_state=42).fit_transform(X)

    # Map labels to colors
    unique = sorted(set(labels))
    cmap = plt.get_cmap("tab20")
    color_map = {lab: cmap(i % 20) for i, lab in enumerate(unique)}
    colors = [color_map[l] for l in labels]

    if args.plotly:
        try:
            import plotly.express as px
        except Exception:
            raise SystemExit("plotly not installed; run `pip install plotly`")
        plot_df = pd.DataFrame(
            {
                "x": X2[:, 0],
                "y": X2[:, 1],
                "label": labels,
                "row_id": product_ids[keep_idx],
            }
        )
        fig = px.scatter(
            plot_df,
            x="x",
            y="y",
            color="label",
            hover_data=["row_id", "label"],
            title=f"Embeddings ({args.method.upper()}) colored by {args.label}",
        )
        fig.write_html(args.html)
        print(f"Saved interactive plot to {args.html}")
        if args.open:
            webbrowser.open_new_tab(str(Path(args.html).resolve()))
        return

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X2[:, 0], X2[:, 1], c=colors, s=10, alpha=0.7)
    plt.title(f"Embeddings ({args.method.upper()}) colored by {args.label}")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    if args.legend:
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", label=lab, markerfacecolor=color_map[lab], markersize=6)
            for lab in unique
        ]
        plt.legend(handles=handles, title=args.label, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved plot to {args.out}")
    if args.show:
        if args.hover:
            try:
                import mplcursors
            except Exception:
                raise SystemExit("mplcursors not installed; run `pip install mplcursors`")
            cursor = mplcursors.cursor(scatter, hover=True)
            cursor.connect(
                "add",
                lambda sel: sel.annotation.set_text(
                    f"id={product_ids[keep_idx[sel.index]]}\n{args.label}={labels[sel.index]}"
                ),
            )
        plt.show()


if __name__ == "__main__":
    main()
