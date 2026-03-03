"""High-level evaluation functions.

Three evaluation scopes:
  evaluate_parser()      — Query parsing quality (3 rubrics, text-only)
  evaluate_item()        — Single product quality (5 rubrics: 4 text + 1 multimodal)
  evaluate_set()         — Full model result set quality (1 rubric, text-only)
  evaluate_cross_model() — CLIP vs FashionCLIP pairwise preference (1 rubric, text-only)

All functions return a dict keyed by rubric name with {"score", "reasoning"} values,
except evaluate_cross_model() which returns {"winner", "clip_score", "fashionclip_score", "reasoning"}.

Parallelism strategy (OpenAI — no Groq rate-limit constraints):
  evaluate_parser()       — 3 rubrics in parallel
  evaluate_item()         — 5 rubrics in parallel per product
  evaluate_query_result() — products evaluated in parallel per model;
                            both models evaluated in parallel;
                            parser + models + cross-model all in parallel
"""
from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from evaluation.judge import run_judge
from evaluation.rubrics import (
    CROSS_MODEL_PREFERENCE,
    ITEM_RUBRICS,
    PARSER_RUBRICS,
    SET_RUBRICS,
    Rubric,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _product_metadata(product: dict) -> dict:
    return {
        "name": product.get("product_name", ""),
        "category": product.get("product_category", ""),
        "color": product.get("color", ""),
        "price": product.get("price", ""),
        "description": product.get("product_description", ""),
    }


def _occasion_target(parsed: dict) -> str | None:
    occ = parsed.get("occasion") or {}
    if isinstance(occ, dict) and occ.get("mode") == "on":
        return occ.get("target")
    return None


def _run_rubric_safe(rubric: Rubric, inputs: dict) -> tuple[str, dict]:
    """Run a single rubric and catch any exception so one failure doesn't abort the batch."""
    try:
        result = run_judge(rubric, inputs)
    except Exception as exc:
        traceback.print_exc()
        result = {"score": -1, "reasoning": f"Judge error: {exc}"}
    return rubric.name, result


# ── Parser evaluation ────────────────────────────────────────────────────────

def evaluate_parser(query: str, parsed: dict) -> dict[str, dict]:
    """Evaluate query parsing quality on 3 rubrics — all 3 run in parallel.

    Args:
        query:  Original user query string.
        parsed: Output of parse_query_llm(query).

    Returns:
        {rubric_name: {"score": int, "reasoning": str}, ...}
    """
    inputs = {
        "query": query,
        "parsed_output": {
            "categories": (parsed.get("constraints") or {}).get("categories", []),
            "colors": (parsed.get("constraints") or {}).get("colors", []),
            "fit": (parsed.get("constraints") or {}).get("fit", []),
            "price_min": (parsed.get("constraints") or {}).get("price_min"),
            "price_max": (parsed.get("constraints") or {}).get("price_max"),
            "occasion_mode": (parsed.get("occasion") or {}).get("mode"),
            "occasion_target": (parsed.get("occasion") or {}).get("target"),
        },
    }
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=len(PARSER_RUBRICS)) as executor:
        futures = {executor.submit(_run_rubric_safe, rubric, inputs): rubric for rubric in PARSER_RUBRICS}
        for future in as_completed(futures):
            name, result = future.result()
            results[name] = result
    return results


# ── Item-level evaluation ────────────────────────────────────────────────────

def evaluate_item(product: dict, query: str, parsed: dict) -> dict[str, dict]:
    """Evaluate a single product recommendation on 5 rubrics — all 5 run in parallel.

    Args:
        product: Product dict with keys: product_name, product_category, color,
                 price, product_description, explanation, image_url, local_image_path.
        query:   Original user query string.
        parsed:  Output of parse_query_llm(query).

    Returns:
        {rubric_name: {"score": int, "reasoning": str}, ...}
    """
    occasion = _occasion_target(parsed)
    meta = _product_metadata(product)
    constraints = (parsed.get("constraints") or {})

    base_inputs = {
        "query": query,
        "parsed_constraints": constraints,
        "occasion_target": occasion,
        "product_metadata": meta,
        "explanation": product.get("explanation", ""),
        "image_url": product.get("image_url", ""),
        "local_image_path": product.get("local_image_path", ""),
    }

    results: dict[str, dict] = {}

    # Separate rubrics that can run from those that must be skipped
    runnable: list[Rubric] = []
    for rubric in ITEM_RUBRICS:
        if rubric.judge_type == "multimodal" and not occasion:
            results[rubric.name] = {"score": None, "reasoning": "No occasion specified — skipped."}
        else:
            runnable.append(rubric)

    with ThreadPoolExecutor(max_workers=len(runnable)) as executor:
        futures = {executor.submit(_run_rubric_safe, rubric, base_inputs): rubric for rubric in runnable}
        for future in as_completed(futures):
            name, result = future.result()
            results[name] = result

    return results


# ── Set-level evaluation ─────────────────────────────────────────────────────

def evaluate_set(products: list[dict], query: str, parsed: dict) -> dict[str, dict]:
    """Evaluate the full recommendation set (top-5) of one model on set-level rubrics.

    Args:
        products: List of up to 5 product dicts (same structure as evaluate_item).
        query:    Original user query string.
        parsed:   Output of parse_query_llm(query).

    Returns:
        {rubric_name: {"score": int, "reasoning": str}, ...}
    """
    occasion = _occasion_target(parsed)
    products_summary = [
        {
            "name": p.get("product_name", ""),
            "category": p.get("product_category", ""),
            "color": p.get("color", ""),
            "price": p.get("price", ""),
        }
        for p in products
    ]
    inputs = {
        "query": query,
        "occasion_target": occasion,
        "products_summary": products_summary,
    }
    results: dict[str, dict] = {}
    for rubric in SET_RUBRICS:
        name, result = _run_rubric_safe(rubric, inputs)
        results[name] = result
    return results


# ── Cross-model pairwise evaluation ──────────────────────────────────────────

def evaluate_cross_model(
    query: str,
    parsed: dict,
    clip_products: list[dict],
    fashionclip_products: list[dict],
) -> dict:
    """Pairwise CLIP vs FashionCLIP preference evaluation.

    Args:
        query:               Original user query string.
        parsed:              Output of parse_query_llm(query).
        clip_products:       CLIP model top-5 products.
        fashionclip_products: FashionCLIP model top-5 products.

    Returns:
        {"winner": str, "clip_score": int, "fashionclip_score": int, "reasoning": str}
    """
    def summarise(products: list[dict]) -> list[dict]:
        return [
            {
                "name": p.get("product_name", ""),
                "category": p.get("product_category", ""),
                "color": p.get("color", ""),
                "price": p.get("price", ""),
                "explanation": p.get("explanation", ""),
            }
            for p in products
        ]

    inputs = {
        "query": query,
        "parsed_constraints": (parsed.get("constraints") or {}),
        "occasion_target": _occasion_target(parsed),
        "clip_products": summarise(clip_products),
        "fashionclip_products": summarise(fashionclip_products),
    }
    _, result = _run_rubric_safe(CROSS_MODEL_PREFERENCE, inputs)
    return result


# ── Convenience: evaluate all scopes for one query result ───────────────────

def evaluate_query_result(
    query: str,
    parsed: dict,
    rows_by_model: dict[str, list[dict]],
) -> dict:
    """Run all evaluations for a single query pipeline result — fully parallelised.

    Parallelism layout:
      - Parser rubrics run in parallel with model evaluations
      - Both models (clip, fashion_clip) evaluate their items in parallel
      - Within each model, all products evaluate in parallel
      - Within each product, all rubrics run in parallel

    Args:
        query:          Original user query.
        parsed:         Parsed query dict from parse_query_llm.
        rows_by_model:  {"clip": [product, ...], "fashion_clip": [product, ...]}

    Returns:
        {
          "parser": {rubric: result, ...},
          "clip": {
            "items": [{rubric: result, ...}, ...],   # one dict per product
            "set": {rubric: result, ...},
          },
          "fashion_clip": { same structure },
          "cross_model": {winner, clip_score, fashionclip_score, reasoning},
        }
    """
    out: dict = {}

    def _eval_model(model_name: str, products: list[dict]) -> tuple[str, dict]:
        """Evaluate all products for one model — products run in parallel."""
        with ThreadPoolExecutor(max_workers=len(products)) as ex:
            item_futures = {
                ex.submit(evaluate_item, product, query, parsed): i
                for i, product in enumerate(products)
            }
            item_scores = [None] * len(products)
            for future in as_completed(item_futures):
                i = item_futures[future]
                item_scores[i] = future.result()

        set_score = evaluate_set(products, query, parsed)
        return model_name, {"items": item_scores, "set": set_score}

    # Run parser + all models in parallel
    with ThreadPoolExecutor(max_workers=2 + len(rows_by_model)) as executor:
        futures: dict = {}

        # Parser
        futures[executor.submit(evaluate_parser, query, parsed)] = "parser"

        # Per-model evaluations
        for model_name, products in rows_by_model.items():
            futures[executor.submit(_eval_model, model_name, products)] = ("model", model_name)

        for future in as_completed(futures):
            key = futures[future]
            if key == "parser":
                out["parser"] = future.result()
            else:
                _, model_name = key
                model_name, model_result = future.result()
                out[model_name] = model_result

    # Cross-model pairwise (needs both models to be done first — runs after the block above)
    clip_products = rows_by_model.get("clip", [])
    fc_products = rows_by_model.get("fashion_clip", [])
    if clip_products and fc_products:
        out["cross_model"] = evaluate_cross_model(query, parsed, clip_products, fc_products)
    else:
        out["cross_model"] = {"winner": "n/a", "reasoning": "One or both models returned no results."}

    return out
