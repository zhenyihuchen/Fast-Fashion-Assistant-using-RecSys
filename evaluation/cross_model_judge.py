"""Cross-model pairwise judge: compares CLIP vs FashionCLIP in a single multimodal call.

All 10 product images are attached (5 CLIP first, then 5 FashionCLIP) so the
judge can visually compare both sets independently — no pre-computed scores used.

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

from evaluation._client import (
    MULTIMODAL_MODEL,
    TIMEOUT,
    client,
    load_image_b64,
    multimodal_semaphore,
    parse_json,
)

CROSS_MODEL_PROMPT = """\
You are an expert evaluator for a fashion recommendation system.

10 product images are attached: the first 5 are CLIP's recommendations (Set A,
in list order), the next 5 are FashionCLIP's recommendations (Set B, in list order).

=== CROSS-MODEL PREFERENCE ===
Criteria:
    Given two sets of fashion product recommendations for the same query — one
    from CLIP (Set A) and one from FashionCLIP (Set B) — determine which model's
    set better serves the user's needs. Use both the product metadata and the
    attached images for your assessment.

Steps:
  Step 1: Read the user query, parsed constraints, and stated occasion.
  Step 2: Review Set A (CLIP) — examine each product's metadata and its attached
          image (images 1-5) to understand category, style, and visual quality.
  Step 3: Review Set B (FashionCLIP) — examine each product's metadata and its
          attached image (images 6-10).
  Step 4: Compare both sets on relevance to the query and hard constraints
          (category, color, fit, price range) — which set better matches?
  Step 5: Compare both sets on visual occasion appropriateness (if applicable) —
          which model's images better convey the right formality and aesthetic?
  Step 6: Compare both sets on visual diversity — which set offers more
          meaningfully different styles and options for the user?
  Step 7: Consider which set better reflects the user's explicit intent overall,
          accounting for both text metadata and visual appearance.
  Step 8: Assign individual overall quality scores (1-5) to each model:
            1 = very poor — most items irrelevant or visually inappropriate
            2 = below average — minority of items meet the user's needs
            3 = average — roughly half the items are suitable
            4 = above average — majority of items are suitable and visually appropriate
            5 = excellent — nearly all items highly relevant and visually appropriate
  Step 9: Declare the winner or a tie:
            clip         = CLIP is clearly better across most dimensions
            fashion_clip = FashionCLIP is clearly better across most dimensions
            tie          = Both sets are comparably good or no clear winner emerges

Evidence:
{evidence}

Think step by step through each evaluation step, then provide your final judgment.

Respond ONLY with valid JSON in this exact format:
{{
  "reasoning":         "<step-by-step evaluation>",
  "winner":            "<clip | fashion_clip | tie>",
  "clip_score":        <integer 1-5>,
  "fashionclip_score": <integer 1-5>
}}"""


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

    prompt = CROSS_MODEL_PROMPT.format(
        evidence=json.dumps(evidence, indent=2, ensure_ascii=False)
    )

    # Build multimodal content: CLIP images first (1-5), then FashionCLIP (6-10)
    content: list[dict] = []
    for product in clip_products + fc_products:
        b64 = load_image_b64(product.get("local_image_path", ""))
        if b64:
            b64_data, media_type = b64
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{b64_data}"},
            })
    content.append({"type": "text", "text": prompt})

    try:
        with multimodal_semaphore:
            resp = client.chat.completions.create(
                model=MULTIMODAL_MODEL,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                timeout=TIMEOUT,
            )
        data = parse_json(resp.choices[0].message.content)
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
