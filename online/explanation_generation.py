from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from dotenv import load_dotenv
from groq import Groq


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

OCCASION_EMBEDDINGS_PATH = BASE_DIR / "occasion_library" / "occasion_prompt_embeddings.npz"
OCCASION_PROMPTS_META = BASE_DIR / "occasion_library" / "occasion_prompt_embeddings_meta.json"

TIMEOUT_SECONDS = 60


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


def _extract_query_keywords(parsed: dict) -> list[str]:
    if not parsed:
        return []
    constraints = parsed.get("constraints") or {}
    if hasattr(constraints, "categories"):
        categories = getattr(constraints, "categories", []) or []
        colors = getattr(constraints, "colors", []) or []
        fit = getattr(constraints, "fit", []) or []
    else:
        categories = constraints.get("categories", []) or []
        colors = constraints.get("colors", []) or []
        fit = constraints.get("fit", []) or []
    keywords = [*map(str, categories), *map(str, colors), *map(str, fit)]
    return [k for k in (kw.strip() for kw in keywords) if k]


def compute_top_prompt_matches(
    candidates: Iterable[tuple[str, float]],
    parsed: dict,
    product_ids: np.ndarray,
    embeddings: np.ndarray,
    occasion_embeddings_path: Path = OCCASION_EMBEDDINGS_PATH,
    prompts_meta_path: Path = OCCASION_PROMPTS_META,
) -> dict[str, dict[str, float | str]]:
    target = _get_occasion_target(parsed)
    if not target:
        return {}

    occasion_npz = np.load(occasion_embeddings_path)
    if target not in occasion_npz:
        return {}
    prompt_embeddings = occasion_npz[target].astype("float32", copy=False)
    prompt_embeddings = _normalize_rows(prompt_embeddings)

    meta = json.loads(prompts_meta_path.read_text(encoding="utf-8"))
    prompt_texts = meta.get("prompts", {}).get(target)
    if not prompt_texts:
        return {}

    id_to_index = {str(pid): idx for idx, pid in enumerate(product_ids)}
    cand_ids = [str(row_id) for row_id, _ in candidates if str(row_id) in id_to_index]
    if not cand_ids:
        return {}

    cand_indices = [id_to_index[row_id] for row_id in cand_ids]
    cand_embeddings = embeddings[cand_indices].astype("float32", copy=False)
    cand_embeddings = _normalize_rows(cand_embeddings)

    sims = cand_embeddings @ prompt_embeddings.T
    best_idx = np.argmax(sims, axis=1)
    best_scores = sims[np.arange(sims.shape[0]), best_idx]
    return {
        row_id: {
            "prompt": str(prompt_texts[int(i)]),
            "score": float(score),
        }
        for row_id, i, score in zip(cand_ids, best_idx, best_scores)
    }


def _build_messages(evidence: dict) -> list[dict[str, str]]:
    system = (
        "You generate concise recommendation explanations for a fashion assistant. "
        "Use only the provided evidence; do not invent attributes. "
        "Write 1-2 sentences. If a field is missing or empty, omit it. "
        "If description is provided, incorporate a short phrase from it (verbatim or lightly paraphrased). "
        "Do not mention that you are an AI, and do not add extra claims."
    )
    user = {
        "evidence": evidence,
        "style_rules": [
            "Ground every statement in evidence fields only.",
            "If occasion fields are missing, do not mention an occasion.",
            "If top_occasion_prompt is missing, do not mention prompt similarity.",
            "If price is missing, do not mention price.",
            "If description is present, mention a concrete attribute from it.",
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user)},
    ]


def _call_groq(messages: list[dict[str, str]]) -> str:
    if not GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY is not set")
    client = Groq(api_key=GROQ_API_KEY)
    chat = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        timeout=TIMEOUT_SECONDS,
    )
    return chat.choices[0].message.content.strip()


def generate_explanations(
    rows: list[dict],
    candidates: Iterable[tuple[str, float]],
    parsed: dict,
    product_ids: np.ndarray,
    embeddings: np.ndarray,
    occasion_scores: dict[str, float],
) -> dict[str, str]:
    target = _get_occasion_target(parsed)
    query_keywords = _extract_query_keywords(parsed)
    top_prompt_matches = compute_top_prompt_matches(
        candidates,
        parsed=parsed,
        product_ids=product_ids,
        embeddings=embeddings,
    )

    explanations: dict[str, str] = {}
    for row in rows:
        row_id = str(row.get("row_id", ""))
        if not row_id:
            continue
        prompt_info = top_prompt_matches.get(row_id, {})
        evidence = {
            "product_title": row.get("product_name", ""),
            "description": row.get("product_description", ""),
            "color": row.get("color", ""),
            "category": row.get("product_category", ""),
            "price": row.get("price", ""),
            "occasion_target": target,
            "occasion_score": occasion_scores.get(row_id),
            "top_occasion_prompt": prompt_info.get("prompt"),
            "top_occasion_prompt_score": prompt_info.get("score"),
            "query_keywords": query_keywords,
        }
        messages = _build_messages(evidence)
        explanations[row_id] = _call_groq(messages)
    return explanations
