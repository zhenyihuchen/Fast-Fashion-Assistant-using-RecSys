"""Item-level judge: evaluates all 3 item rubrics in a single multimodal API call.

A base64-encoded product image is attached alongside the prompt so the model
can visually assess the item for occasion appropriateness and relevance.

Rubric prompts are loaded from a YAML file (default: prompts/item_judge.yaml).
To A/B test, create a variant YAML and pass its path via the ``prompts_path``
argument to :func:`run_item_judge`.

Returns:
    {
        "item_relevance":          {"score": int,       "reasoning": str},
        "occasion_appropriateness":{"score": int|None,  "reasoning": str},
        "explanation_quality":     {"score": int,       "reasoning": str},
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

DEFAULT_PROMPTS_PATH = Path(__file__).parent / "prompts" / "item_judge.yaml"


def _load_prompts(path: Path = DEFAULT_PROMPTS_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _build_rubric_block(index: int, name: str, rubric: dict) -> str:
    title = name.upper().replace("_", " ")
    scale_lines = "\n".join(
        f"            {k} = {v}" for k, v in rubric["scale"].items()
    )
    return (
        f"=== {index}. {title} ===\n"
        f"Criteria:\n    {rubric['criteria'].strip()}\n\n"
        f"Steps:\n  {rubric['steps'].strip()}\n"
        f"{scale_lines}\n"
    )


def _build_item_prompt(prompts: dict) -> str:
    rubric_order = ["item_relevance", "occasion_appropriateness", "explanation_quality"]
    blocks = []
    for i, name in enumerate(rubric_order, 1):
        blocks.append(_build_rubric_block(i, name, prompts[name]))

    return (
        "You are an expert evaluator for a fashion recommendation system.\n\n"
        "A product image is attached. Evaluate the recommended product on THREE dimensions\n"
        "simultaneously. Think through each dimension carefully, then respond with a single\n"
        "JSON object.\n\n"
        + "\n".join(blocks)
        + "\nEvidence:\n{evidence}\n\n"
        "Respond ONLY with valid JSON in this exact format:\n"
        '{{\n'
        '  "item_relevance":           {{"score": <integer 1-5>, "reasoning": "<step-by-step evaluation>"}},\n'
        '  "occasion_appropriateness": {{"score": <integer 1-5 or null>, "reasoning": "<step-by-step evaluation>"}},\n'
        '  "explanation_quality":      {{"score": <integer 1-5>, "reasoning": "<step-by-step evaluation>"}}\n'
        '}}'
    )

ITEM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "item_relevance": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "score": {"type": "integer", "minimum": 1, "maximum": 5},
                "reasoning": {"type": "string"},
            },
            "required": ["score", "reasoning"],
        },
        "occasion_appropriateness": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "score": {
                    "anyOf": [
                        {"type": "integer", "minimum": 1, "maximum": 5},
                        {"type": "null"},
                    ]
                },
                "reasoning": {"type": "string"},
            },
            "required": ["score", "reasoning"],
        },
        "explanation_quality": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "score": {"type": "integer", "minimum": 1, "maximum": 5},
                "reasoning": {"type": "string"},
            },
            "required": ["score", "reasoning"],
        },
    },
    "required": [
        "item_relevance",
        "occasion_appropriateness",
        "explanation_quality",
    ],
}

_prompts_cache: dict[str, dict] = {}


def _get_prompts(path: Path = DEFAULT_PROMPTS_PATH) -> dict:
    key = str(path)
    if key not in _prompts_cache:
        _prompts_cache[key] = _load_prompts(path)
    return _prompts_cache[key]


def _product_metadata(product: dict) -> dict:
    return {
        "name":        product.get("product_name", ""),
        "category":    product.get("product_category", ""),
        "color":       product.get("color", ""),
        "price":       product.get("price", ""),
        "description": product.get("product_description", ""),
    }


def _occasion_target(parsed: dict) -> str | None:
    occ = parsed.get("occasion") or {}
    if isinstance(occ, dict) and occ.get("mode") == "on":
        return occ.get("target")
    return None


def run_item_judge(
    query: str,
    product: dict,
    parsed: dict,
    prompts_path: Path = DEFAULT_PROMPTS_PATH,
) -> dict:
    """Evaluate a single product on 3 rubrics in one multimodal call.

    Args:
        query:   Original user query string.
        product: Product dict with keys: product_name, product_category, color,
                 price, product_description, explanation, local_image_path.
        parsed:  Output of parse_query_llm(query).
        prompts_path: Path to rubric YAML file (for A/B testing variants).

    Returns:
        {rubric_name: {"score": int|None, "reasoning": str}, ...}
    """
    prompts = _get_prompts(prompts_path)

    occasion = _occasion_target(parsed)
    meta = _product_metadata(product)
    constraints = parsed.get("constraints") or {}

    evidence = {
        "query":             query,
        "parsed_constraints": constraints,
        "occasion_target":   occasion,
        "product_metadata":  meta,
        "explanation":       product.get("explanation", ""),
    }

    item_prompt_template = _build_item_prompt(prompts)
    prompt = item_prompt_template.format(
        evidence=json.dumps(evidence, indent=2, ensure_ascii=False)
    )

    # Build multimodal content: [image (if available), text]
    images: list[dict[str, str]] = []
    b64_result = load_image_b64(product.get("local_image_path", ""))
    if b64_result:
        b64_data, media_type = b64_result
        images.append({"url": f"data:{media_type};base64,{b64_data}", "detail": "low"})

    try:
        with multimodal_semaphore:
            data = create_json_response(
                model=MULTIMODAL_MODEL,
                instructions=prompts.get("system", "You are an expert evaluator for a fashion recommendation system."),
                user_text=prompt,
                schema_name="item_evaluation",
                schema=ITEM_SCHEMA,
                images=images,
            )

        result: dict = {}
        for rubric in ("item_relevance", "explanation_quality"):
            entry = data.get(rubric, {})
            score = int(entry.get("score", -1))
            if not (1 <= score <= 5):
                score = -1
            result[rubric] = {"score": score, "reasoning": entry.get("reasoning", "")}

        # occasion_appropriateness: may be null
        occ_entry = data.get("occasion_appropriateness", {})
        occ_score = occ_entry.get("score")
        if occ_score is None:
            result["occasion_appropriateness"] = {
                "score": None,
                "reasoning": occ_entry.get("reasoning", "No occasion specified."),
            }
        else:
            s = int(occ_score)
            result["occasion_appropriateness"] = {
                "score": s if 1 <= s <= 5 else -1,
                "reasoning": occ_entry.get("reasoning", ""),
            }

        return result

    except Exception as exc:
        traceback.print_exc()
        return {
            rubric: {"score": -1, "reasoning": f"Judge error: {exc}"}
            for rubric in (
                "item_relevance",
                "occasion_appropriateness",
                "explanation_quality",
            )
        }


# run_item_judge(query, product, parsed, prompts_path=Path("evaluation/prompts/item_judge_v2.yaml"))
