"""Set-level judge: evaluates set_answer_relevance in a single multimodal API call.

All 5 product images are attached so the judge can visually assess diversity,
style coherence, and occasion appropriateness across the full set — independently
of any item-level scores.

Rubric prompts are loaded from a YAML file (default: prompts/set_judge.yaml).

Returns:
    {"set_answer_relevance": {"score": int, "reasoning": str}}
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

DEFAULT_PROMPTS_PATH = Path(__file__).parent / "prompts" / "set_judge.yaml"

SET_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "set_answer_relevance": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "score": {"type": "integer", "minimum": 1, "maximum": 5},
                "reasoning": {"type": "string"},
            },
            "required": ["score", "reasoning"],
        }
    },
    "required": ["set_answer_relevance"],
}

_prompts_cache: dict[str, dict] = {}


def _get_prompts(path: Path = DEFAULT_PROMPTS_PATH) -> dict:
    key = str(path)
    if key not in _prompts_cache:
        _prompts_cache[key] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _prompts_cache[key]


def _build_set_prompt(prompts: dict) -> str:
    rubric = prompts["set_answer_relevance"]
    scale_lines = "\n".join(
        f"            {k} = {v}" for k, v in rubric["scale"].items()
    )
    return (
        "You are an expert evaluator for a fashion recommendation system.\n\n"
        "Product images for all items in the set are attached in order (Product 1 first,\n"
        "Product 5 last). Evaluate whether the complete set collectively addresses\n"
        "the user's overall shopping intent.\n\n"
        "=== SET ANSWER RELEVANCE ===\n"
        f"Criteria:\n    {rubric['criteria'].strip()}\n\n"
        f"Steps:\n  {rubric['steps'].strip()}\n"
        f"{scale_lines}\n\n"
        "Evidence:\n{evidence}\n\n"
        "Respond ONLY with valid JSON in this exact format:\n"
        '{{\"set_answer_relevance\": {{\"score\": <integer 1-5>, \"reasoning\": \"<step-by-step evaluation>\"}}}}'
    )


def run_set_judge(
    query: str,
    products: list[dict],
    parsed: dict,
    prompts_path: Path = DEFAULT_PROMPTS_PATH,
) -> dict:
    """Evaluate the full recommendation set on set_answer_relevance.

    All product images are embedded as base64 and sent alongside the prompt
    for independent visual assessment (no item-level scores used).

    Args:
        query:    Original user query string.
        products: List of up to 5 product dicts (must include local_image_path).
        parsed:   Output of parse_query_llm(query).

    Returns:
        {"set_answer_relevance": {"score": int, "reasoning": str}}
    """
    occ = parsed.get("occasion") or {}
    occasion_target = occ.get("target") if occ.get("mode") == "on" else None

    products_summary = [
        {
            "product_number": i + 1,
            "name":     p.get("product_name", ""),
            "category": p.get("product_category", ""),
            "color":    p.get("color", ""),
            "price":    p.get("price", ""),
        }
        for i, p in enumerate(products)
    ]

    evidence = {
        "query":           query,
        "occasion_target": occasion_target,
        "products":        products_summary,
    }

    prompts = _get_prompts(prompts_path)
    prompt_template = _build_set_prompt(prompts)
    prompt = prompt_template.format(
        evidence=json.dumps(evidence, indent=2, ensure_ascii=False)
    )

    # Build multimodal content: images first (in product order), then the prompt
    images: list[dict[str, str]] = []
    for product in products:
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
                schema_name="set_evaluation",
                schema=SET_SCHEMA,
                images=images,
            )
        entry = data.get("set_answer_relevance", {})
        score = int(entry.get("score", -1))
        if not (1 <= score <= 5):
            score = -1
        return {"set_answer_relevance": {"score": score, "reasoning": entry.get("reasoning", "")}}
    except Exception as exc:
        traceback.print_exc()
        return {"set_answer_relevance": {"score": -1, "reasoning": f"Judge error: {exc}"}}
