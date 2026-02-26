from __future__ import annotations

import json
from datetime import datetime, timezone
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import numpy as np
import open_clip
import torch
from fashion_clip.fashion_clip import FashionCLIP


BASE_DIR = Path(__file__).resolve().parent
PROMPTS_PATH = BASE_DIR / "occasion_prompts_clavix.json"

DATA_DIR = BASE_DIR.parent / "data" / "prompt_embeddings"
CLIP_DIR = DATA_DIR / "clip"
FCLIP_DIR = DATA_DIR / "fashion_clip"

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
BATCH_SIZE = 32


def _load_prompts(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("occasion_prompts.json must be a JSON object")
    if "occasions" in data and isinstance(data["occasions"], dict):
        data = data["occasions"]
    cleaned: dict[str, list[str]] = {}
    for key, value in data.items():
        if not isinstance(value, list):
            continue
        prompts = [str(v).strip() for v in value if str(v).strip()]
        if prompts:
            cleaned[str(key)] = prompts
    return cleaned


def _save_outputs(
    out_dir: Path,
    embeddings: dict[str, np.ndarray],
    prompt_texts: dict[str, list[str]],
    meta_extra: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "occasion_embeddings.npz"
    meta_path = out_dir / "occasion_embeddings_meta.json"

    np.savez(npz_path, **embeddings)

    meta = {
        **meta_extra,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompts": prompt_texts,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"  embeddings : {npz_path}")
    print(f"  metadata   : {meta_path}")


def _run_clip(prompt_map: dict[str, list[str]], device: str) -> None:
    print("\n[CLIP] Loading model …")
    model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    model = model.to(device)
    model.eval()

    out_data: dict[str, np.ndarray] = {}
    for occasion, prompts in prompt_map.items():
        embeddings = []
        for i in range(0, len(prompts), BATCH_SIZE):
            tokens = tokenizer(prompts[i : i + BATCH_SIZE]).to(device)
            with torch.no_grad():
                feats = model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu().numpy())
        emb = np.concatenate(embeddings, axis=0)
        out_data[occasion] = emb
        print(f"  {occasion}: {emb.shape}")

    print("[CLIP] Saving outputs …")
    _save_outputs(
        CLIP_DIR,
        out_data,
        prompt_map,
        {"model": CLIP_MODEL_NAME, "pretrained": CLIP_PRETRAINED, "device": device},
    )


def _run_fashion_clip(prompt_map: dict[str, list[str]]) -> None:
    print("\n[FashionCLIP] Loading model …")
    fclip = FashionCLIP("fashion-clip")

    out_data: dict[str, np.ndarray] = {}
    for occasion, prompts in prompt_map.items():
        emb = fclip.encode_text(prompts, batch_size=BATCH_SIZE)
        emb = emb / np.linalg.norm(emb, ord=2, axis=-1, keepdims=True)
        out_data[occasion] = emb
        print(f"  {occasion}: {emb.shape}")

    print("[FashionCLIP] Saving outputs …")
    _save_outputs(
        FCLIP_DIR,
        out_data,
        prompt_map,
        {"model": "fashion-clip"},
    )


def main() -> None:
    prompt_map = _load_prompts(PROMPTS_PATH)
    if not prompt_map:
        raise SystemExit("No prompts found in occasion_prompts.json")

    print(f"Loaded {len(prompt_map)} occasions from {PROMPTS_PATH.name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _run_clip(prompt_map, device)
    _run_fashion_clip(prompt_map)

    print("\nDone.")


if __name__ == "__main__":
    main()
