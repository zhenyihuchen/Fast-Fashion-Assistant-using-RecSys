from __future__ import annotations

import os
# Must be set before any HuggingFace import — prevents the Rust tokenizer from
# spawning semaphores that cause SIGBUS on macOS ARM64.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
from pathlib import Path
import re

import open_clip
import torch  # import before numpy to avoid MKL issues
import numpy as np
import pandas as pd
import faiss

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "offline" / "data"
PARQUET_PATH = DATA_DIR / "women_data.parquet"

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

# Per-model paths for image embeddings, product id arrays, pre-built FAISS
# indices, and (optional) text embeddings.
MODEL_PATHS: dict[str, dict[str, Path]] = {
    "clip": {
        "image_embeddings": DATA_DIR / "image_embeddings" / "clip" / "image_embeddings.npy",
        "image_ids":        DATA_DIR / "image_embeddings" / "clip" / "product_ids.npy",
        "faiss_index":      DATA_DIR / "image_embeddings" / "clip" / "faiss.index",
        "text_embeddings":  DATA_DIR / "text_embeddings"  / "clip" / "text_embeddings.npy",
        "text_ids":         DATA_DIR / "text_embeddings"  / "clip" / "product_ids.npy",
    },
    "fashion_clip": {
        "image_embeddings": DATA_DIR / "image_embeddings" / "fashion_clip" / "image_embeddings.npy",
        "image_ids":        DATA_DIR / "image_embeddings" / "fashion_clip" / "product_ids.npy",
        "faiss_index":      DATA_DIR / "image_embeddings" / "fashion_clip" / "faiss.index",
        "text_embeddings":  DATA_DIR / "text_embeddings"  / "fashion_clip" / "text_embeddings.npy",
        "text_ids":         DATA_DIR / "text_embeddings"  / "fashion_clip" / "product_ids.npy",
    },
}

# Module-level lazy caches — models are loaded once and reused across queries.
_clip_cache: tuple | None = None   # (model, tokenizer, device)
_fclip_cache: tuple | None = None  # (CLIPModel, CLIPTokenizer, device)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _get_clip_model() -> tuple:
    global _clip_cache
    if _clip_cache is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, _ = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
        )
        tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        model = model.to(device).eval()
        _clip_cache = (model, tokenizer, device)
    return _clip_cache


def _get_fclip() -> tuple:
    """Load patrickjohncyh/fashion-clip directly via transformers.

    low_cpu_mem_usage=False  — skips meta-tensor / memory-mapped loading,
                               which triggers SIGBUS on macOS ARM64 (16 KB vs
                               4 KB page alignment mismatch).
    CLIPTokenizer (slow, pure-Python) — no Rust extension, no semaphores.
    """
    global _fclip_cache
    if _fclip_cache is None:
        from transformers import CLIPModel, CLIPTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained(
            "patrickjohncyh/fashion-clip", low_cpu_mem_usage=False
        )
        tokenizer = CLIPTokenizer.from_pretrained("patrickjohncyh/fashion-clip")
        model = model.to(device).eval()
        _fclip_cache = (model, tokenizer, device)
    return _fclip_cache


# ---------------------------------------------------------------------------
# Query encoding
# ---------------------------------------------------------------------------

def _encode_query_clip(text: str) -> np.ndarray:
    model, tokenizer, device = _get_clip_model()
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0].astype("float32")


def _encode_query_fashion_clip(text: str) -> np.ndarray:
    model, tokenizer, device = _get_fclip()
    inputs = tokenizer(
        [text], return_tensors="pt",
        max_length=77, padding="max_length", truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    emb = features.cpu().numpy()
    emb = emb / np.linalg.norm(emb, ord=2, axis=-1, keepdims=True)
    return emb[0].astype("float32")


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def _parse_price(value: str) -> float | None:
    if not value:
        return None
    m = re.search(r"(\d+(\.\d+)?)", value)
    return float(m.group(1)) if m else None


def _normalize_color(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _category_to_dataset(cat: str) -> str:
    return cat.strip().upper()


def _filter_rows(df: pd.DataFrame, parsed: dict) -> pd.DataFrame:
    filtered = df
    constraints = parsed.get("constraints")
    if not constraints:
        return filtered

    categories = constraints.categories if hasattr(constraints, "categories") else constraints.get("categories", [])
    colors     = constraints.colors     if hasattr(constraints, "colors")      else constraints.get("colors", [])
    price_min  = constraints.price_min  if hasattr(constraints, "price_min")   else constraints.get("price_min")
    price_max  = constraints.price_max  if hasattr(constraints, "price_max")   else constraints.get("price_max")

    if categories:
        mapped = {_category_to_dataset(c) for c in categories}
        filtered = filtered[filtered["product_category"].isin(mapped)]

    if colors:
        color_norm = filtered["color"].astype(str).map(_normalize_color)
        color_set  = {_normalize_color(c) for c in colors}
        mask = color_norm.apply(lambda c: any(k in c for k in color_set))
        filtered = filtered[mask]

    if price_min is not None or price_max is not None:
        prices = filtered["price"].astype(str).map(_parse_price)
        if price_min is not None:
            filtered = filtered[prices >= float(price_min)]
        if price_max is not None:
            filtered = filtered[prices <= float(price_max)]

    return filtered


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1.0, norms)


def _search_embeddings(
    q_vec: np.ndarray,
    embeddings: np.ndarray,
    ids: np.ndarray,
    topk: int,
    index_path: Path | None = None,
) -> list[tuple[str, float]]:
    """Search with a pre-built FAISS index when available, or build one on-the-fly."""
    if index_path and index_path.exists():
        index = faiss.read_index(str(index_path))
    else:
        emb = embeddings.copy().astype("float32")
        faiss.normalize_L2(emb)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)

    scores, indices = index.search(q_vec.reshape(1, -1), min(topk, len(ids)))
    return [(str(ids[i]), float(scores[0][j])) for j, i in enumerate(indices[0]) if i >= 0]


def _rerank_with_text(
    candidates: list[tuple[str, float]],
    q_vec: np.ndarray,
    text_ids: np.ndarray,
    text_embeddings: np.ndarray,
    image_weight: float,
    text_weight: float,
) -> list[tuple[str, float]]:
    text_embeddings = _normalize_rows(text_embeddings.astype("float32", copy=False))
    q_vec = q_vec / np.linalg.norm(q_vec)
    id_to_index = {str(pid): idx for idx, pid in enumerate(text_ids)}

    rescored: list[tuple[str, float]] = []
    for row_id, image_score in candidates:
        t_idx = id_to_index.get(row_id)
        text_score = float(text_embeddings[t_idx] @ q_vec) if t_idx is not None else 0.0
        rescored.append((row_id, image_weight * image_score + text_weight * text_score))

    rescored.sort(key=lambda x: x[1], reverse=True)
    return rescored


# ---------------------------------------------------------------------------
# Per-model retrieval
# ---------------------------------------------------------------------------

def _retrieve_for_model(
    model_name: str,
    query: str,
    parsed: dict,
    df: pd.DataFrame,
    topk: int,
    filter_first: bool,
    use_text_embeddings: bool,
    image_weight: float,
    text_weight: float,
) -> list[tuple[str, float]]:
    paths = MODEL_PATHS[model_name]

    q_vec = (
        _encode_query_clip(query)
        if model_name == "clip"
        else _encode_query_fashion_clip(query)
    )

    product_ids = np.load(paths["image_ids"], allow_pickle=True).astype(str)
    embeddings  = np.load(paths["image_embeddings"]).astype("float32", copy=False)

    text_ids = text_emb = None
    if use_text_embeddings and paths["text_embeddings"].exists():
        text_ids = np.load(paths["text_ids"], allow_pickle=True).astype(str)
        text_emb = np.load(paths["text_embeddings"]).astype("float32", copy=False)

    if filter_first:
        filtered    = _filter_rows(df, parsed)
        allowed_ids = set(filtered["row_id"].astype(str))
        mask        = np.array([pid in allowed_ids for pid in product_ids])
        if not mask.any():
            return []
        results = _search_embeddings(q_vec, embeddings[mask], product_ids[mask], topk)
        if use_text_embeddings and text_ids is not None and text_emb is not None:
            results = _rerank_with_text(results, q_vec, text_ids, text_emb, image_weight, text_weight)
        return results

    # search-first, filter-after
    results = _search_embeddings(q_vec, embeddings, product_ids, topk, paths["faiss_index"])
    if use_text_embeddings and text_ids is not None and text_emb is not None:
        results = _rerank_with_text(results, q_vec, text_ids, text_emb, image_weight, text_weight)
    if not parsed:
        return results
    allowed_ids = set(_filter_rows(df, parsed)["row_id"].astype(str))
    return [(pid, score) for pid, score in results if pid in allowed_ids]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_candidates(
    query: str,
    parsed: dict,
    topk: int = 200,
    filter_first: bool = True,
    use_faiss: bool = True,          # kept for backward compat; FAISS is always used
    use_text_embeddings: bool = False,
    image_weight: float = 0.7,
    text_weight: float = 0.3,
    embedding_model: str = "both",   # "clip" | "fashion_clip" | "both"
) -> dict[str, list[tuple[str, float]]]:
    """Return candidates for each embedding model separately.

    Returns ``{"clip": [...], "fashion_clip": [...]}`` (or a single-key dict
    when *embedding_model* is not ``"both"``).  Results are kept separate so
    every downstream step (occasion scoring, ranking, explanations) runs
    independently per model.
    """
    df = pd.read_parquet(PARQUET_PATH)
    df["row_id"] = df["row_id"].astype(str)

    models_to_run = ["clip", "fashion_clip"] if embedding_model == "both" else [embedding_model]

    results: dict[str, list[tuple[str, float]]] = {}
    for model_name in models_to_run:
        results[model_name] = _retrieve_for_model(
            model_name, query, parsed, df,
            topk, filter_first,
            use_text_embeddings, image_weight, text_weight,
        )
    return results


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    query_text = (
        "I am looking for a night short red dress but that costs less than 50 euros"
    )
    parsed_path = BASE_DIR / "online" / "test_query_outputs" / "example1.json"
    with parsed_path.open("r", encoding="utf-8") as handle:
        parsed_query = json.load(handle)

    print(f"Query : {query_text}")
    print(f"Parsed: {parsed_path}\n")

    results_by_model = retrieve_candidates(
        query_text, parsed_query, topk=10, filter_first=True
    )

    for model_name, candidates in results_by_model.items():
        print(f"── {model_name.upper()} ── {len(candidates)} candidates")
        for rank, (product_id, score) in enumerate(candidates, start=1):
            print(f"  {rank:02d}. {product_id}  score={score:.4f}")
        print()
