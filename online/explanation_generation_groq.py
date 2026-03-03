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
# MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# Explanations are short (2-3 sentences) with structured evidence, so a fast
# small model is sufficient and avoids rate-limit errors from parallel calls.
# Override with GROQ_EXPLANATION_MODEL in .env if needed.
EXPLANATION_MODEL = os.getenv("GROQ_EXPLANATION_MODEL", "llama-3.1-8b-instant")
# EXPLANATION_MODEL = os.getenv("GROQ_EXPLANATION_MODEL", "groq/compound-mini")


TIMEOUT_SECONDS = 60

PROMPT_EMBEDDINGS_DIR = BASE_DIR / "offline" / "data" / "prompt_embeddings"
MODEL_PROMPT_EMBEDDINGS_PATHS: dict[str, Path] = {
    "clip": PROMPT_EMBEDDINGS_DIR / "clip" / "occasion_embeddings.npz",
    "fashion_clip": PROMPT_EMBEDDINGS_DIR / "fashion_clip" / "occasion_embeddings.npz",
}
SHARED_PROMPTS_META_PATH = PROMPT_EMBEDDINGS_DIR / "occasion_prompt_embeddings_meta.json"
MODEL_PROMPTS_META_PATHS: dict[str, Path] = {
    "clip": PROMPT_EMBEDDINGS_DIR / "clip" / "occasion_embeddings_meta.json",
    "fashion_clip": PROMPT_EMBEDDINGS_DIR / "fashion_clip" / "occasion_embeddings_meta.json",
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
    model_name: str = "clip",
    occasion_embeddings_path: Path | None = None,
    prompts_meta_path: Path | None = None,
) -> dict[str, dict[str, float | str]]:
    target = _get_occasion_target(parsed)
    if not target:
        return {}

    if occasion_embeddings_path is None:
        occasion_embeddings_path = MODEL_PROMPT_EMBEDDINGS_PATHS.get(
            model_name,
            MODEL_PROMPT_EMBEDDINGS_PATHS["clip"],
        )
    if prompts_meta_path is None:
        prompts_meta_path = MODEL_PROMPTS_META_PATHS.get(
            model_name,
            SHARED_PROMPTS_META_PATH,
        )
    if not occasion_embeddings_path.exists() or not prompts_meta_path.exists():
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
    n_prompts = min(prompt_embeddings.shape[0], len(prompt_texts))
    if n_prompts <= 0:
        return {}
    prompt_embeddings = prompt_embeddings[:n_prompts]
    prompt_texts = prompt_texts[:n_prompts]

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


def _build_prompt(evidence: dict) -> tuple[str, str]:
    system = (
        "You generate detailed yet concise recommendation explanations for a fashion assistant. "
        "Use only the provided evidence; do not invent attributes. "
        "Write 2-3 sentences. If a field is missing or empty, omit it. "
        "If description is provided, incorporate a short phrase from it (verbatim or lightly paraphrased). "
        "Do not mention that you are an AI, and do not add extra claims. "
        "Do not mention numeric scores or technical details."
    )
    user = {
        "evidence": evidence,
        "style_rules": [
            "Ground every statement in evidence fields only.",
            "If occasion fields are missing, do not mention an occasion.",
            "If top_occasion_prompt is missing, do not mention prompt similarity.",
            "If price is missing, do not mention price.",
            "If description is present, mention a concrete attribute from it.",
            "Explain why the item fits the occasion and the user query keywords.",
            "Use specific metadata like color, category, and price when available.",
            "Do not mention scores or numeric values from the evidence.",
        ],
    }
    return system, json.dumps(user)


def _call_groq_langchain(system: str, evidence_payload: str) -> str:
    if not GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY is not set")
    client = Groq(api_key=GROQ_API_KEY)
    chat = client.chat.completions.create(
        model=EXPLANATION_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": evidence_payload},
        ],
        temperature=0.2,
        timeout=TIMEOUT_SECONDS,
    )
    return chat.choices[0].message.content.strip()


def _generate_one_explanation(
    row: dict,
    target: str | None,
    query_keywords: list[str],
    top_prompt_matches: dict[str, dict[str, float | str]],
    occasion_scores: dict[str, float],
) -> tuple[str, str]:
    row_id = str(row.get("row_id", ""))
    if not row_id:
        return "", ""
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
    system, payload = _build_prompt(evidence)
    return row_id, _call_groq_langchain(system, payload)


def generate_explanations(
    rows: list[dict],
    candidates: Iterable[tuple[str, float]],
    parsed: dict,
    product_ids: np.ndarray,
    embeddings: np.ndarray,
    occasion_scores: dict[str, float],
    model_name: str = "clip",
    occasion_embeddings_path: Path | None = None,
    prompts_meta_path: Path | None = None,
) -> dict[str, str]:
    target = _get_occasion_target(parsed)
    query_keywords = _extract_query_keywords(parsed)
    top_prompt_matches = compute_top_prompt_matches(
        candidates,
        parsed=parsed,
        product_ids=product_ids,
        embeddings=embeddings,
        model_name=model_name,
        occasion_embeddings_path=occasion_embeddings_path,
        prompts_meta_path=prompts_meta_path,
    )

    explanations: dict[str, str] = {}
    work_rows = [row for row in rows if str(row.get("row_id", ""))]
    if not work_rows:
        return explanations

    # Sequential to avoid hitting Groq free-tier TPM limits (6 000 TPM on llama-3.1-8b-instant)
    for row in work_rows:
        row_id = str(row.get("row_id", ""))
        try:
            out_row_id, text = _generate_one_explanation(
                row, target, query_keywords, top_prompt_matches, occasion_scores
            )
            if out_row_id:
                explanations[out_row_id] = text
        except Exception as exc:
            print(f"[explanation] failed for row_id={row_id}: {exc}")
            explanations[row_id] = "Explanation unavailable."
    return explanations
