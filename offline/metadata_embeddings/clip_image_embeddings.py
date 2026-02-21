from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image
import torch
import open_clip


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
EMBED_DIR = DATA_DIR / "image_embeddings"

INPUT_PARQUET = DATA_DIR / "women_data.parquet"
IMAGE_DIR = DATA_DIR / "images"
EMBEDDINGS_OUT = EMBED_DIR / "image_embeddings.npy"
IDS_OUT = EMBED_DIR / "product_ids.npy"
MODEL_INFO_OUT = EMBED_DIR / "embedding_model.txt"
STATUS_OUT = EMBED_DIR / "embedding_status.csv"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
BATCH_SIZE = 32


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "unknown"


def _product_id(row: pd.Series) -> str:
    if "row_id" in row and pd.notna(row["row_id"]) and str(row["row_id"]).strip():
        return str(row["row_id"]).strip()
    return "unknown"


def main() -> None:
    df = pd.read_parquet(INPUT_PARQUET)
    if "reference_number" not in df.columns or "row_id" not in df.columns:
        raise SystemExit("Missing reference_number or row_id column")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model = model.to(device)
    model.eval()

    embeddings = []
    product_ids = []

    status_rows = []
    batch_images = []
    batch_ids = []

    def flush_batch() -> None: #  inner helper to run inference on a batch
        if not batch_images:
            return
        with torch.no_grad(): #  inference without gradients
            batch = torch.stack(batch_images).to(device) # pack images into a tensor
            feats = model.encode_image(batch) # compute image embeddings
            feats = feats / feats.norm(dim=-1, keepdim=True) # normalize embeddings
            embeddings.append(feats.cpu().numpy())
            product_ids.extend(batch_ids) # align ids with embeddings
        batch_images.clear()
        batch_ids.clear()

    for _, row in df.iterrows():
        ref = str(row["reference_number"]).strip()
        pid = _product_id(row)
        filename = f"{_safe_filename(ref)}.jpg"
        image_path = IMAGE_DIR / filename

        if not image_path.exists():
            status_rows.append(
                {
                    "product_id": pid,
                    "reference_number": ref,
                    "image_path": str(image_path),
                    "status": "missing_file",
                    "error": "",
                }
            )
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            batch_images.append(preprocess(image))
            batch_ids.append(pid)
            status_rows.append(
                {
                    "product_id": pid,
                    "reference_number": ref,
                    "image_path": str(image_path),
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as e:
            status_rows.append(
                {
                    "product_id": pid,
                    "reference_number": ref,
                    "image_path": str(image_path),
                    "status": "failed",
                    "error": str(e),
                }
            )
            continue

        if len(batch_images) >= BATCH_SIZE:
            flush_batch()

    flush_batch()

    if embeddings:
        embeddings_array = np.concatenate(embeddings, axis=0)
    else:
        embeddings_array = np.empty((0, 0), dtype=np.float32)

    np.save(EMBEDDINGS_OUT, embeddings_array)
    np.save(IDS_OUT, np.array(product_ids, dtype=object))

    with open(MODEL_INFO_OUT, "w", encoding="utf-8") as f:
        f.write(f"model={MODEL_NAME}\n")
        f.write(f"pretrained={PRETRAINED}\n")
        f.write(f"device={device}\n")
        f.write(f"generated_at={datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")

    with open(STATUS_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["product_id", "reference_number", "image_path", "status", "error"],
        )
        writer.writeheader()
        writer.writerows(status_rows)

    print(f"Saved {EMBEDDINGS_OUT} with shape {embeddings_array.shape}")
    print(f"Saved {IDS_OUT} with {len(product_ids)} ids")
    print(f"Saved {MODEL_INFO_OUT} and {STATUS_OUT}")


if __name__ == "__main__":
    main()
