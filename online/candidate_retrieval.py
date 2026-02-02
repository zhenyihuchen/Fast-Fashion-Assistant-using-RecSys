from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import open_clip
import torch # !!! import this before numpy to avoid MKL issues
import numpy as np
import pandas as pd
import faiss
# import os

# os.environ['KMP_DUPLICATE_LIB_OK']='True'
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EMBED_DIR = DATA_DIR / "image_embeddings"

EMBEDDINGS_PATH = EMBED_DIR / "image_embeddings.npy"
IDS_PATH = EMBED_DIR / "product_ids.npy"
INDEX_PATH = EMBED_DIR / "faiss.index"
PARQUET_PATH = DATA_DIR / "women_data.parquet"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"


def _parse_price(value: str) -> float | None:
    if not value:
        return None
    m = re.search(r"(\d+(\.\d+)?)", value)
    if not m:
        return None
    return float(m.group(1))


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
    colors = constraints.colors if hasattr(constraints, "colors") else constraints.get("colors", [])
    price_min = constraints.price_min if hasattr(constraints, "price_min") else constraints.get("price_min")
    price_max = constraints.price_max if hasattr(constraints, "price_max") else constraints.get("price_max")

    if categories:
        mapped = {_category_to_dataset(c) for c in categories}
        filtered = filtered[filtered["product_category"].isin(mapped)]

    if colors:
        color_norm = filtered["color"].astype(str).map(_normalize_color)
        color_set = {_normalize_color(c) for c in colors}
        mask = color_norm.apply(lambda c: any(k in c for k in color_set)) # match by substring
        filtered = filtered[mask]

    if price_min is not None or price_max is not None:
        prices = filtered["price"].astype(str).map(_parse_price)
        if price_min is not None:
            filtered = filtered[prices >= float(price_min)]
        if price_max is not None:
            filtered = filtered[prices <= float(price_max)]

    return filtered


def _encode_query(text: str) -> np.ndarray:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.to(device)
    model.eval()

    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0].astype("float32")


def _search_vectors(
    q_vec: np.ndarray,
    embeddings: np.ndarray,
    topk: int,
    use_faiss: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not use_faiss:
        raise ValueError("FAISS search disabled; enable use_faiss to search vectors")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    scores, indices = index.search(q_vec.reshape(1, -1), topk)
    return scores[0], indices[0]


def retrieve_candidates(
    query: str,
    parsed: dict,
    topk: int = 200,
    filter_first: bool = True,
    use_faiss: bool = True,
) -> list[tuple[str, float]]:
    df = pd.read_parquet(PARQUET_PATH)
    df["row_id"] = df["row_id"].astype(str)

    product_ids = np.load(IDS_PATH, allow_pickle=True).astype(str)
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32", copy=False)

    q_vec = _encode_query(query)

    if filter_first:
        filtered = _filter_rows(df, parsed)
        allowed_ids = set(filtered["row_id"].astype(str))
        mask = np.array([pid in allowed_ids for pid in product_ids])
        if not mask.any():
            return []
        sub_embeddings = embeddings[mask]
        sub_ids = product_ids[mask]
        scores, idx = _search_vectors(
            q_vec,
            sub_embeddings,
            min(topk, len(sub_ids)),
            use_faiss=use_faiss,
        )
        return [(sub_ids[i], float(scores[j])) for j, i in enumerate(idx)]

    # FAISS first, then filter.
    scores, idx = _search_vectors(
        q_vec,
        embeddings,
        min(topk, len(product_ids)),
        use_faiss=use_faiss,
    )
    pairs = [(product_ids[i], float(scores[j])) for j, i in enumerate(idx)]
    if not parsed:
        return pairs

    filtered = _filter_rows(df, parsed)
    allowed_ids = set(filtered["row_id"].astype(str))
    return [(pid, score) for pid, score in pairs if pid in allowed_ids]


# if __name__ == "__main__":
#     query_text = (
#         "I am looking for a night short dress red but that costs red that costs less than 50 euros"
#     )
#     parsed_path = BASE_DIR / "online" / "test_query_outputs" / "example1.json"
#     with parsed_path.open("r", encoding="utf-8") as handle:
#         parsed_query = json.load(handle)
#     results = retrieve_candidates(query_text, parsed_query, topk=10, filter_first=True)
#     print(f"Query: {query_text}")
#     print(f"Parsed JSON: {parsed_path}")
#     for rank, (product_id, score) in enumerate(results, start=1):
#         print(f"{rank:02d}. {product_id} -> {score:.4f}")
