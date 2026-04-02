"""High-level evaluation functions.

Four evaluation scopes — each independent, each a single API call:
  evaluate_parser()      — 3 parser rubrics, 1 text call      (gpt-4o-mini)
  evaluate_item()        — 3 item rubrics,   1 multimodal call (gpt-4o + image)
  evaluate_set()         — 1 set rubric,     1 multimodal call (gpt-4o + 5 images)
  evaluate_cross_model() — 1 pairwise rubric, 1 multimodal call (gpt-4o + 10 images)

All levels evaluate from raw evidence independently — no score chaining.

Parallelism (evaluate_query_result):
  Parser + all item evals + both set evals + cross-model all run in parallel.
  Only the multimodal_semaphore (cap=3) limits concurrent vision calls.
"""
from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from evaluation.cross_model_judge import run_cross_model_judge
from evaluation.item_judge import run_item_judge
from evaluation.parser_judge import run_parser_judge
from evaluation.set_judge import run_set_judge

# ── V2 prompt paths (loosened scales for better score spread) ────────────────
_PROMPTS_DIR = Path(__file__).parent / "prompts"
ITEM_PROMPTS_PATH    = _PROMPTS_DIR / "item_judge_v2.yaml"
SET_PROMPTS_PATH     = _PROMPTS_DIR / "set_judge_v2.yaml"
PARSER_PROMPTS_PATH  = _PROMPTS_DIR / "parser_judge_v2.yaml"
CROSS_PROMPTS_PATH   = _PROMPTS_DIR / "cross_model_judge_v2.yaml"

_ITEM_RUBRICS = ("item_relevance", "occasion_appropriateness", "explanation_quality")


# ── Public convenience wrappers ──────────────────────────────────────────────

def evaluate_parser(query: str, parsed: dict) -> dict[str, dict]:
    """Evaluate query parsing quality — single text call, 3 rubrics at once.

    Returns:
        {rubric_name: {"score": int, "reasoning": str}, ...}
    """
    return run_parser_judge(query, parsed, prompts_path=PARSER_PROMPTS_PATH)


def evaluate_item(product: dict, query: str, parsed: dict) -> dict[str, dict]:
    """Evaluate a single product — single multimodal call, 3 rubrics at once.

    Returns:
        {rubric_name: {"score": int|None, "reasoning": str}, ...}
    """
    return run_item_judge(query, product, parsed, prompts_path=ITEM_PROMPTS_PATH)


def evaluate_set(products: list[dict], query: str, parsed: dict) -> dict[str, dict]:
    """Evaluate the recommendation set — single multimodal call with 5 images.

    Returns:
        {"set_answer_relevance": {"score": int, "reasoning": str}}
    """
    return run_set_judge(query, products, parsed, prompts_path=SET_PROMPTS_PATH)


def evaluate_cross_model(
    query: str,
    parsed: dict,
    clip_products: list[dict],
    fc_products: list[dict],
) -> dict:
    """Pairwise CLIP vs FashionCLIP — single multimodal call with 10 images.

    Returns:
        {"winner": str, "clip_score": int, "fashionclip_score": int, "reasoning": str}
    """
    return run_cross_model_judge(query, parsed, clip_products, fc_products, prompts_path=CROSS_PROMPTS_PATH)


# ── Full pipeline evaluation ─────────────────────────────────────────────────

def evaluate_query_result(
    query: str,
    parsed: dict,
    rows_by_model: dict[str, list[dict]],
) -> dict:
    """Run all evaluations for a single query result — fully parallel.

    All levels are independent (no score chaining), so everything runs
    concurrently. The multimodal_semaphore in _client.py caps vision calls at 3.

    Args:
        query:          Original user query.
        parsed:         Parsed query dict from parse_query_llm.
        rows_by_model:  {"clip": [product, ...], "fashion_clip": [product, ...]}

    Returns:
        {
          "parser": {rubric: result, ...},
          "clip": {
            "items": [{rubric: result, ...}, ...],
            "set":   {rubric: result, ...},
          },
          "fashion_clip": { same structure },
          "cross_model": {winner, clip_score, fashionclip_score, reasoning},
        }
    """
    items_by_model: dict[str, list[dict | None]] = {
        m: [None] * len(prods) for m, prods in rows_by_model.items()
    }
    set_results:  dict[str, dict] = {}
    parser_result: dict = {}
    cross: dict = {"winner": "n/a", "reasoning": "One or both models returned no results."}

    clip_products = rows_by_model.get("clip", [])
    fc_products   = rows_by_model.get("fashion_clip", [])

    with ThreadPoolExecutor() as ex:
        futures: dict = {}

        # Parser
        futures[ex.submit(run_parser_judge, query, parsed, PARSER_PROMPTS_PATH)] = ("parser",)

        # Item evals — all products across all models
        for model_name, products in rows_by_model.items():
            for i, product in enumerate(products):
                f = ex.submit(run_item_judge, query, product, parsed, ITEM_PROMPTS_PATH)
                futures[f] = ("item", model_name, i)

        # Set evals — one per model
        for model_name, products in rows_by_model.items():
            f = ex.submit(run_set_judge, query, products, parsed, SET_PROMPTS_PATH)
            futures[f] = ("set", model_name)

        # Cross-model
        if clip_products and fc_products:
            f = ex.submit(run_cross_model_judge, query, parsed, clip_products, fc_products, CROSS_PROMPTS_PATH)
            futures[f] = ("cross_model",)

        for f in as_completed(futures):
            tag = futures[f]
            try:
                result = f.result()
            except Exception:
                traceback.print_exc()
                result = None

            if tag[0] == "parser":
                parser_result = result or {
                    r: {"score": -1, "reasoning": "Judge error."}
                    for r in ("parser_completeness", "parser_no_hallucination", "parser_occasion_detection")
                }
            elif tag[0] == "item":
                _, model_name, i = tag
                items_by_model[model_name][i] = result or {
                    r: {"score": -1, "reasoning": "Judge error."}
                    for r in _ITEM_RUBRICS
                }
            elif tag[0] == "set":
                _, model_name = tag
                set_results[model_name] = result or {
                    "set_answer_relevance": {"score": -1, "reasoning": "Judge error."}
                }
            elif tag[0] == "cross_model":
                cross = result or {
                    "winner": "tie", "clip_score": -1,
                    "fashionclip_score": -1, "reasoning": "Judge error.",
                }

    result = {"parser": parser_result, "cross_model": cross}
    for model_name in rows_by_model:
        result[model_name] = {
            "items": items_by_model.get(model_name, []),
            "set":   set_results.get(model_name, {}),
        }
    return result
