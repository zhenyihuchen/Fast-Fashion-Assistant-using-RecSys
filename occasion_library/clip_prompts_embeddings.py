from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open_clip
import torch


BASE_DIR = Path(__file__).resolve().parent
PROMPTS_PATH = BASE_DIR / "occasion_prompts_clavix.json"
OUT_PATH = BASE_DIR / "occasion_clavix_embeddings.npz"
META_PATH = BASE_DIR / "occasion_clavix_embeddings_meta.json"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
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


def _encode_prompts(model, tokenizer, prompts: list[str], device: str) -> np.ndarray:
    embeddings = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch_prompts = prompts[i : i + BATCH_SIZE]
        tokens = tokenizer(batch_prompts).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings.append(feats.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def main() -> None:
    prompt_map = _load_prompts(PROMPTS_PATH)
    if not prompt_map:
        raise SystemExit("No prompts found in occasion_prompts.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.to(device)
    model.eval()

    out_data: dict[str, np.ndarray] = {}
    prompt_texts: dict[str, list[str]] = {}

    for occasion, prompts in prompt_map.items():
        emb = _encode_prompts(model, tokenizer, prompts, device)
        out_data[occasion] = emb
        prompt_texts[occasion] = prompts
        print(f"{occasion}: {emb.shape}")

    np.savez(OUT_PATH, **out_data)

    meta = {
        "model": MODEL_NAME,
        "pretrained": PRETRAINED,
        "device": device,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompts": prompt_texts,
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved embeddings to {OUT_PATH}")
    print(f"Saved metadata to {META_PATH}")


if __name__ == "__main__":
    main()
