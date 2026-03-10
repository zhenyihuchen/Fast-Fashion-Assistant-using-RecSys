"""Item-level judge: evaluates all 3 item rubrics in a single multimodal API call.

A base64-encoded product image is attached alongside the prompt so the model
can visually assess the item for occasion appropriateness and relevance.

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

from evaluation._client import (
    MULTIMODAL_MODEL,
    create_json_response,
    load_image_b64,
    multimodal_semaphore,
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

ITEM_PROMPT = """\
You are an expert evaluator for a fashion recommendation system.

A product image is attached. Evaluate the recommended product on THREE dimensions
simultaneously. Think through each dimension carefully, then respond with a single
JSON object.

=== 1. ITEM RELEVANCE ===
Criteria:
    Evaluate whether the recommended fashion product is relevant to the user's
    search query. Adapted from the RAG metric 'Context Relevance': checks if the
    retrieved item aligns with what the user was looking for.

Steps:
  Step 1: Read the user query and identify what the user is looking for:
          item type, color, fit, price range.
  Step 2: Check if the product category matches what the user requested.
  Step 3: Check if the product color aligns with any color specified in the query.
  Step 4: Check if the product fit or style matches any fit preference mentioned.
  Step 5: Check if the price is within any stated price range.
  Step 6: Consider the overall semantic alignment between the query intent and
          the product name and description. Also examine the attached image.
  Step 7: Assign a score from 1 to 5:
            1 = completely irrelevant — wrong category, color, and style
            2 = mostly irrelevant — matches at most one constraint
            3 = partially relevant — matches some constraints but misses key ones
            4 = mostly relevant — matches most constraints with one minor misalignment
            5 = highly relevant — matches all stated constraints and overall intent

=== 2. OCCASION APPROPRIATENESS ===
Criteria:
    Evaluate whether the recommended product is visually and contextually appropriate
    for the stated occasion, using both the attached product image and its metadata.
    If no occasion is specified in the evidence, output null for this dimension.

Steps:
  Step 1: Check the evidence for an occasion_target. If absent or null, skip and
          return null score with reasoning "No occasion specified".
  Step 2: Identify the stated occasion (e.g. wedding, job interview, date night,
          festival, New Year's Eve).
  Step 3: Consider what clothing styles, formality levels, and aesthetics are
          typical and expected for this occasion.
  Step 4: Examine the attached product image — assess its visual style, formality
          level, silhouette, and overall aesthetic.
  Step 5: Review the product metadata (category, name, color) to further understand
          the item beyond what the image shows.
  Step 6: Determine whether a person attending this occasion would realistically
          and comfortably wear this item.
  Step 7: Assign a score from 1 to 5:
            1 = clearly inappropriate — e.g. swimwear for a job interview
            2 = unlikely to be appropriate — wrong formality level or aesthetic
            3 = could work in some contexts — borderline, depends on interpretation
            4 = appropriate for the occasion — fits expected style and formality
            5 = clearly and ideally suited — exemplary, stands out as a great choice

=== 3. EXPLANATION QUALITY ===
Criteria:
    Evaluate whether the generated product explanation is both grounded in the
    actual product metadata (no hallucinated attributes) and genuinely useful for
    a shopper making a purchase decision. Adapted from the RAG metric 'Groundedness'.

Steps:
  Step 1: Read the product metadata: name, category, color, price, and description.
  Step 2: Read the user query and the stated occasion (if any).
  Step 3: Read the generated explanation.
  Step 4: For each claim in the explanation, verify it can be directly confirmed
          from the metadata — flag any attributes NOT in the metadata as hallucinations.
  Step 5: Assess whether the explanation is specific — does it mention concrete
          product attributes, or is it generic praise that could apply to any item?
  Step 6: Assess whether the explanation connects the product to the user's query
          and/or occasion with clear, logical reasoning.
  Step 7: Penalise both hallucinated claims and generic filler statements.
  Step 8: Assign a score from 1 to 5:
            1 = hallucinated or completely generic — could describe any product
            2 = multiple issues: hallucinations AND/OR very generic statements
            3 = mostly grounded and somewhat useful — one or two minor issues
            4 = well-grounded and specific, clearly connects product to query/occasion
            5 = fully grounded, specific, and genuinely helpful for the purchase decision

Evidence:
{evidence}

Respond ONLY with valid JSON in this exact format:
{{
  "item_relevance":           {{"score": <integer 1-5>, "reasoning": "<step-by-step evaluation>"}},
  "occasion_appropriateness": {{"score": <integer 1-5 or null>, "reasoning": "<step-by-step evaluation>"}},
  "explanation_quality":      {{"score": <integer 1-5>, "reasoning": "<step-by-step evaluation>"}}
}}"""


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


def run_item_judge(query: str, product: dict, parsed: dict) -> dict:
    """Evaluate a single product on 3 rubrics in one multimodal call.

    Args:
        query:   Original user query string.
        product: Product dict with keys: product_name, product_category, color,
                 price, product_description, explanation, local_image_path.
        parsed:  Output of parse_query_llm(query).

    Returns:
        {rubric_name: {"score": int|None, "reasoning": str}, ...}
    """
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

    prompt = ITEM_PROMPT.format(
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
                instructions="You are an expert evaluator for a fashion recommendation system.",
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
