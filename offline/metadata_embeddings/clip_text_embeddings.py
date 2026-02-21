from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import open_clip


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
EMBED_DIR = DATA_DIR / "text_embeddings"

INPUT_PARQUET = DATA_DIR / "women_data.parquet"
EMBEDDINGS_OUT = EMBED_DIR / "text_embeddings.npy"
IDS_OUT = EMBED_DIR / "product_ids.npy"
MODEL_INFO_OUT = EMBED_DIR / "embedding_model.txt"
STATUS_OUT = EMBED_DIR / "embedding_status.csv"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
BATCH_SIZE = 64


def _product_id(row: pd.Series) -> str:
    if "row_id" in row and pd.notna(row["row_id"]) and str(row["row_id"]).strip():
        return str(row["row_id"]).strip()
    return "unknown"


def _build_text(row: pd.Series) -> str:
    title = str(row.get("product_name", "") or "").strip()
    desc = str(row.get("product_description", "") or "").strip()
    category = str(row.get("product_category", "") or "").strip()
    color = str(row.get("color", "") or "").strip()
    parts = [title, desc, category, color]
    return ". ".join([p for p in parts if p])


def main() -> None:
    df = pd.read_parquet(INPUT_PARQUET)
    if "row_id" not in df.columns:
        raise SystemExit("Missing row_id column in women_data.parquet")

    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.to(device)
    model.eval()

    embeddings = []
    product_ids = []
    status_rows = []

    batch_texts = []
    batch_ids = []

    def flush_batch() -> None:
        if not batch_texts:
            return
        tokens = tokenizer(batch_texts).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu().numpy())
            product_ids.extend(batch_ids)
        batch_texts.clear()
        batch_ids.clear()

    for _, row in df.iterrows():
        pid = _product_id(row)
        text = _build_text(row)
        if not text:
            status_rows.append(
                {
                    "product_id": pid,
                    "status": "missing_text",
                    "error": "",
                }
            )
            continue

        batch_texts.append(text)
        batch_ids.append(pid)
        status_rows.append(
            {
                "product_id": pid,
                "status": "ok",
                "error": "",
            }
        )

        if len(batch_texts) >= BATCH_SIZE:
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
            fieldnames=["product_id", "status", "error"],
        )
        writer.writeheader()
        writer.writerows(status_rows)

    print(f"Saved {EMBEDDINGS_OUT} with shape {embeddings_array.shape}")
    print(f"Saved {IDS_OUT} with {len(product_ids)} ids")
    print(f"Saved {MODEL_INFO_OUT} and {STATUS_OUT}")


if __name__ == "__main__":
    main()
