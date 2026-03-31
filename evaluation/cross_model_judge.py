"""Cross-model pairwise judge: compares CLIP vs FashionCLIP in a single multimodal call.

All 10 product images are attached (5 CLIP first, then 5 FashionCLIP) so the
judge can visually compare both sets independently — no pre-computed scores used.

Rubric prompts are loaded from a YAML file (default: prompts/cross_model_judge.yaml).

Returns:
    {
        "winner":            str,  # "clip" | "fashion_clip" | "tie"
        "clip_score":        int,  # 1-5 overall quality score for CLIP
        "fashionclip_score": int,  # 1-5 overall quality score for FashionCLIP
        "reasoning":         str,
    }
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import yaml

from evaluation._client import (
    MULTIMODAL_MODEL,
    create_json_response,
    load_image_b64,
    multimodal_semaphore,
)

DEFAULT_PROMPTS_PATH = Path(__file__).parent / "prompts" / "cross_model_judge.yaml"

_prompts_cache: dict[str, dict] = {}


def _get_prompts(path: Path = DEFAULT_PROMPTS_PATH) -> dict:
    key = str(path)
    if key not in _prompts_cache:
        _prompts_cache[key] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _prompts_cache[key]


def _build_cross_model_prompt(prompts: dict) -> str:
    rubric = prompts["cross_model_preference"]
    model_scale_lines = "\n".join(
        f"            {k} = {v}" for k, v in rubric["model_scale"].items()
    )
    winner_scale_lines = "\n".join(
        f"            {k} = {v}" for k, v in rubric["winner_scale"].items()
    )
    return (
        "You are an expert evaluator for a fashion recommendation system.\n\n"
        "10 product images are attached: the first 5 are CLIP's recommendations (Set A,\n"
        "in list order), the next 5 are FashionCLIP's recommendations (Set B, in list order).\n\n"
        "=== CROSS-MODEL PREFERENCE ===\n"
        f"Criteria:\n    {rubric['criteria'].strip()}\n\n"
        f"Steps:\n  {rubric['steps'].strip()}\n\n"
        f"Model quality scores (1-5):\n{model_scale_lines}\n\n"
        f"Winner decision:\n{winner_scale_lines}\n\n"
        "Evidence:\n{evidence}\n\n"
        "Think step by step through each evaluation step, then provide your final judgment.\n\n"
        "Respond ONLY with valid JSON in this exact format:\n"
        '{{\n'
        '  "reasoning":         "<step-by-step evaluation>",\n'
        '  "winner":            "<clip | fashion_clip | tie>",\n'
        '  "clip_score":        <integer 1-5>,\n'
        '  "fashionclip_score": <integer 1-5>\n'
        '}}'
    )


CROSS_MODEL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "reasoning": {"type": "string"},
        "winner": {"type": "string", "enum": ["clip", "fashion_clip", "tie"]},
        "clip_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "fashionclip_score": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["reasoning", "winner", "clip_score", "fashionclip_score"],
}

def _product_summary(products: list[dict]) -> list[dict]:
    return [
        {
            "product_number": i + 1,
            "name":     p.get("product_name", ""),
            "category": p.get("product_category", ""),
            "color":    p.get("color", ""),
            "price":    p.get("price", ""),
            "explanation": p.get("explanation", ""),
        }
        for i, p in enumerate(products)
    ]


def run_cross_model_judge(
    query: str,
    parsed: dict,
    clip_products: list[dict],
    fc_products: list[dict],
    prompts_path: Path = DEFAULT_PROMPTS_PATH,
) -> dict:
    """Pairwise CLIP vs FashionCLIP evaluation from raw product evidence.

    All 10 images are embedded as base64 (CLIP first, FashionCLIP second).
    No pre-computed item or set scores are used — independent assessment only.

    Args:
        query:         Original user query string.
        parsed:        Output of parse_query_llm(query).
        clip_products: CLIP model top-5 products (must include local_image_path).
        fc_products:   FashionCLIP model top-5 products (must include local_image_path).

    Returns:
        {"winner": str, "clip_score": int, "fashionclip_score": int, "reasoning": str}
    """
    occ = parsed.get("occasion") or {}
    occasion_target = occ.get("target") if occ.get("mode") == "on" else None

    evidence = {
        "query":              query,
        "parsed_constraints": parsed.get("constraints") or {},
        "occasion_target":    occasion_target,
        "set_a_clip":         _product_summary(clip_products),
        "set_b_fashionclip":  _product_summary(fc_products),
    }

    prompts = _get_prompts(prompts_path)
    prompt_template = _build_cross_model_prompt(prompts)
    prompt = prompt_template.format(
        evidence=json.dumps(evidence, indent=2, ensure_ascii=False)
    )

    # Build multimodal content: CLIP images first (1-5), then FashionCLIP (6-10)
    images: list[dict[str, str]] = []
    for product in clip_products + fc_products:
        b64 = load_image_b64(product.get("local_image_path", ""))
        if b64:
            b64_data, media_type = b64
            images.append({"url": f"data:{media_type};base64,{b64_data}", "detail": "low"})

    try:
        with multimodal_semaphore:
            data = create_json_response(
                model=MULTIMODAL_MODEL,
                instructions=prompts.get("system", "You are an expert evaluator for a fashion recommendation system."),
                user_text=prompt,
                schema_name="cross_model_evaluation",
                schema=CROSS_MODEL_SCHEMA,
                images=images,
            )
        winner = data.get("winner", "tie")
        if winner not in {"clip", "fashion_clip", "tie"}:
            winner = "tie"
        clip_score = int(data.get("clip_score", -1))
        fc_score   = int(data.get("fashionclip_score", -1))
        return {
            "winner":            winner,
            "clip_score":        clip_score if 1 <= clip_score <= 5 else -1,
            "fashionclip_score": fc_score   if 1 <= fc_score   <= 5 else -1,
            "reasoning":         data.get("reasoning", ""),
        }
    except Exception as exc:
        traceback.print_exc()
        return {
            "winner": "tie",
            "clip_score": -1,
            "fashionclip_score": -1,
            "reasoning": f"Judge error: {exc}",
        }
