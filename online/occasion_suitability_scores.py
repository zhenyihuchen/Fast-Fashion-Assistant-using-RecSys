from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_EMBEDDINGS_DIR = BASE_DIR / "offline" / "data" / "prompt_embeddings"
MODEL_OCCASION_EMBEDDINGS_PATHS: dict[str, Path] = {
    "clip": PROMPT_EMBEDDINGS_DIR / "clip" / "occasion_embeddings.npz",
    "fashion_clip": PROMPT_EMBEDDINGS_DIR / "fashion_clip" / "occasion_embeddings.npz",
}


def _get_occasion_target(parsed: dict) -> str | None:
    if not parsed:
        return None
    occasion = parsed.get("occasion")
    if not occasion:
        return None
    if hasattr(occasion, "mode"):
        mode = getattr(occasion, "mode")
        target = getattr(occasion, "target", None)
    else:
        mode = occasion.get("mode")
        target = occasion.get("target")
    if mode != "on":
        return None
    return target


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def compute_occasion_scores(
    candidates: Iterable[tuple[str, float]],
    parsed: dict,
    product_ids: np.ndarray,
    embeddings: np.ndarray,
    model_name: str = "clip",
    occasion_embeddings_path: Path | None = None,
) -> dict[str, float]:
    target = _get_occasion_target(parsed)
    if not target:
        return {}

    if occasion_embeddings_path is None:
        occasion_embeddings_path = MODEL_OCCASION_EMBEDDINGS_PATHS.get(
            model_name,
            MODEL_OCCASION_EMBEDDINGS_PATHS["clip"],
        )
    if not occasion_embeddings_path.exists():
        return {}

    occasion_npz = np.load(occasion_embeddings_path)
    if target not in occasion_npz:
        return {}
    prompt_embeddings = occasion_npz[target].astype("float32", copy=False)
    if not np.isfinite(prompt_embeddings).all():
        return {}
    prompt_embeddings = _normalize_rows(prompt_embeddings)

    id_to_index = {str(pid): idx for idx, pid in enumerate(product_ids)}
    cand_ids = [str(row_id) for row_id, _ in candidates if str(row_id) in id_to_index]
    if not cand_ids:
        return {}

    cand_indices = [id_to_index[row_id] for row_id in cand_ids]
    cand_embeddings = embeddings[cand_indices].astype("float32", copy=False)
    cand_embeddings = _normalize_rows(cand_embeddings)

    # cosine similarity to each prompt, then mean across prompts
    sims = cand_embeddings @ prompt_embeddings.T
    mean_scores = sims.mean(axis=1)
    return {row_id: float(score) for row_id, score in zip(cand_ids, mean_scores)}
