from __future__ import annotations

import csv
from datetime import datetime, timezone
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import numpy as np
import pandas as pd
import torch
import open_clip
from fashion_clip.fashion_clip import FashionCLIP


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
EMBED_DIR = DATA_DIR / "text_embeddings"

CLIP_DIR = EMBED_DIR / "clip"
FCLIP_DIR = EMBED_DIR / "fashion_clip"

INPUT_PARQUET = DATA_DIR / "women_data.parquet"

# open_clip / CLIP settings
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
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
    return ". ".join([p for p in [title, desc, category, color] if p])


def _save_outputs(
    out_dir: Path,
    embeddings_array: np.ndarray,
    product_ids: list[str],
    status_rows: list[dict],
    model_info_lines: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "text_embeddings.npy", embeddings_array)
    np.save(out_dir / "product_ids.npy", np.array(product_ids, dtype=object))

    with open(out_dir / "embedding_model.txt", "w", encoding="utf-8") as f:
        f.writelines(model_info_lines)

    with open(out_dir / "embedding_status.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["product_id", "status", "error"])
        writer.writeheader()
        writer.writerows(status_rows)

    print(f"  embeddings : {out_dir / 'text_embeddings.npy'}  shape={embeddings_array.shape}")
    print(f"  product_ids: {out_dir / 'product_ids.npy'}  count={len(product_ids)}")


def _run_clip(
    texts: list[str],
    valid_ids: list[str],
    status_rows: list[dict],
    device: str,
) -> None:
    print("\n[CLIP] Loading model …")
    model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    model = model.to(device)
    model.eval()

    embeddings: list[np.ndarray] = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu().numpy())

    embeddings_array = (
        np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 0), dtype=np.float32)
    )

    model_info = [
        f"model={CLIP_MODEL_NAME}\n",
        f"pretrained={CLIP_PRETRAINED}\n",
        f"device={device}\n",
        f"generated_at={datetime.now(timezone.utc).isoformat(timespec='seconds')}\n",
    ]

    print("[CLIP] Saving outputs …")
    _save_outputs(CLIP_DIR, embeddings_array, valid_ids, status_rows, model_info)


def _run_fashion_clip(
    texts: list[str],
    valid_ids: list[str],
    status_rows: list[dict],
) -> None:
    print("\n[FashionCLIP] Loading model …")
    fclip = FashionCLIP("fashion-clip")

    text_embeddings = fclip.encode_text(texts, batch_size=BATCH_SIZE)
    text_embeddings = text_embeddings / np.linalg.norm(
        text_embeddings, ord=2, axis=-1, keepdims=True
    )

    model_info = [
        "model=fashion-clip\n",
        f"generated_at={datetime.now(timezone.utc).isoformat(timespec='seconds')}\n",
    ]

    print("[FashionCLIP] Saving outputs …")
    _save_outputs(FCLIP_DIR, text_embeddings, valid_ids, status_rows, model_info)


def main() -> None:
    df = pd.read_parquet(INPUT_PARQUET)
    if "row_id" not in df.columns:
        raise SystemExit("Missing row_id column in women_data.parquet")

    # --- single text-building pass ---
    texts: list[str] = []
    valid_ids: list[str] = []
    status_rows: list[dict] = []

    for _, row in df.iterrows():
        pid = _product_id(row)
        text = _build_text(row)

        if not text:
            status_rows.append({"product_id": pid, "status": "missing_text", "error": ""})
            continue

        texts.append(text)
        valid_ids.append(pid)
        status_rows.append({"product_id": pid, "status": "ok", "error": ""})

    print(f"Built {len(texts)} text descriptions ({len(status_rows) - len(texts)} skipped)")

    _run_clip(texts, valid_ids, status_rows, device="cuda" if torch.cuda.is_available() else "cpu")
    _run_fashion_clip(texts, valid_ids, status_rows)

    print("\nDone.")


if __name__ == "__main__":
    main()
