from __future__ import annotations

import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import re

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import numpy as np
import pandas as pd
from PIL import Image
import torch
import open_clip
from fashion_clip.fashion_clip import FashionCLIP


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
EMBED_DIR = DATA_DIR / "image_embeddings"

CLIP_DIR = EMBED_DIR / "clip"
FCLIP_DIR = EMBED_DIR / "fashion_clip"

INPUT_PARQUET = DATA_DIR / "women_data.parquet"
IMAGE_DIR = DATA_DIR / "images"

# open_clip / CLIP settings
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
BATCH_SIZE = 32


def _safe_l2_normalize(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = arr.astype("float32", copy=False)
    norms = np.linalg.norm(arr, ord=2, axis=-1, keepdims=True)
    norms = np.where(np.isfinite(norms) & (norms > eps), norms, 1.0)
    out = arr / norms
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "unknown"


def _product_id(row: pd.Series) -> str:
    if "row_id" in row and pd.notna(row["row_id"]) and str(row["row_id"]).strip():
        return str(row["row_id"]).strip()
    return "unknown"


def _save_outputs(
    out_dir: Path,
    embeddings_array: np.ndarray,
    product_ids: list[str],
    status_rows: list[dict],
    model_info_lines: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "image_embeddings.npy", embeddings_array)
    np.save(out_dir / "product_ids.npy", np.array(product_ids, dtype=object))

    with open(out_dir / "embedding_model.txt", "w", encoding="utf-8") as f:
        f.writelines(model_info_lines)

    with open(out_dir / "embedding_status.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["product_id", "reference_number", "image_path", "status", "error"],
        )
        writer.writeheader()
        writer.writerows(status_rows)

    print(f"  embeddings : {out_dir / 'image_embeddings.npy'}  shape={embeddings_array.shape}")
    print(f"  product_ids: {out_dir / 'product_ids.npy'}  count={len(product_ids)}")


def _run_clip(
    pil_images: list[Image.Image],
    valid_ids: list[str],
    status_rows: list[dict],
    device: str,
) -> None:
    print("\n[CLIP] Loading model …")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    model = model.to(device)
    model.eval()

    embeddings: list[np.ndarray] = []

    for start in range(0, len(pil_images), BATCH_SIZE):
        batch_pil = pil_images[start : start + BATCH_SIZE]
        batch_tensor = torch.stack([preprocess(img) for img in batch_pil]).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor)
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
    pil_images: list[Image.Image],
    valid_ids: list[str],
    status_rows: list[dict],
) -> None:
    print("\n[FashionCLIP] Loading model …")
    fclip = FashionCLIP("fashion-clip")

    image_embeddings = fclip.encode_images(pil_images, batch_size=BATCH_SIZE)
    image_embeddings = _safe_l2_normalize(image_embeddings)

    model_info = [
        "model=fashion-clip\n",
        f"generated_at={datetime.now(timezone.utc).isoformat(timespec='seconds')}\n",
    ]

    print("[FashionCLIP] Saving outputs …")
    _save_outputs(FCLIP_DIR, image_embeddings, valid_ids, status_rows, model_info)


def main() -> None:
    df = pd.read_parquet(INPUT_PARQUET)
    if "reference_number" not in df.columns or "row_id" not in df.columns:
        raise SystemExit("Missing reference_number or row_id column")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- single image-loading pass ---
    pil_images: list[Image.Image] = []
    valid_ids: list[str] = []
    status_rows: list[dict] = []

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
            pil_images.append(Image.open(image_path).convert("RGB"))
            valid_ids.append(pid)
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

    print(f"Loaded {len(pil_images)} images ({len(status_rows) - len(pil_images)} skipped)")

    # only the status rows for valid images are shared; skipped rows are included in both CSVs
    _run_clip(pil_images, valid_ids, status_rows, device)
    _run_fashion_clip(pil_images, valid_ids, status_rows)

    print("\nDone.")


if __name__ == "__main__":
    main()
