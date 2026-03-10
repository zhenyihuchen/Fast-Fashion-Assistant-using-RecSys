"""Set-level judge: evaluates set_answer_relevance in a single multimodal API call.

All 5 product images are attached so the judge can visually assess diversity,
style coherence, and occasion appropriateness across the full set — independently
of any item-level scores.

Returns:
    {"set_answer_relevance": {"score": int, "reasoning": str}}
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

SET_PROMPT = """\
You are an expert evaluator for a fashion recommendation system.

Product images for all items in the set are attached in order (Product 1 first,
Product 5 last). Evaluate whether the complete set collectively addresses
the user's overall shopping intent.

=== SET ANSWER RELEVANCE ===
Criteria:
    Evaluate whether the complete set of recommended products collectively addresses
    the user's overall shopping intent, using both the product metadata and the
    attached images. Adapted from the RAG metric 'Answer Relevance'.

Steps:
  Step 1: Read the user query and identify the overall shopping intent — what the
          user wants, in what context, and for what occasion (if any).
  Step 2: Review each product's metadata (name, category, color, price) and visually
          examine its attached image to understand its style and formality.
  Step 3: Assess whether the set collectively covers what the user asked for —
          do the products span the right categories, styles, and price points?
  Step 4: Assess visual and stylistic diversity across the 5 images — are they
          meaningfully different from each other, or nearly identical in style?
  Step 5: Evaluate visual coherence with the stated occasion (if any) — do the
          images convey the right formality level and aesthetic as a group?
  Step 6: Identify glaring omissions — obvious product types the user mentioned
          that are absent from the set.
  Step 7: Synthesise your observations into a holistic set-level judgment.
  Step 8: Assign a score from 1 to 5:
            1 = set completely misses the intent — most items are irrelevant or visually inappropriate
            2 = set partially relevant — fewer than half the items fit the intent
            3 = set addresses intent moderately — majority fit but notable gaps or poor visual diversity
            4 = set addresses intent well — strong majority fit with only minor gaps
            5 = set fully and comprehensively addresses the intent with good visual diversity

Evidence:
{evidence}

Respond ONLY with valid JSON in this exact format:
{{"set_answer_relevance": {{"score": <integer 1-5>, "reasoning": "<step-by-step evaluation>"}}}}"""


def run_set_judge(
    query: str,
    products: list[dict],
    parsed: dict,
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

    prompt = SET_PROMPT.format(
        evidence=json.dumps(evidence, indent=2, ensure_ascii=False)
    )

    # Build multimodal content: images first (in product order), then the prompt
    content: list[dict] = []
    for product in products:
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
        entry = data.get("set_answer_relevance", {})
        score = int(entry.get("score", -1))
        if not (1 <= score <= 5):
            score = -1
        return {"set_answer_relevance": {"score": score, "reasoning": entry.get("reasoning", "")}}
    except Exception as exc:
        traceback.print_exc()
        return {"set_answer_relevance": {"score": -1, "reasoning": f"Judge error: {exc}"}}
