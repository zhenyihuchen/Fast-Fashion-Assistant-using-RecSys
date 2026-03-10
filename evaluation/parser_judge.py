"""Parser-level judge: evaluates all 3 parser rubrics in a single text API call.

Returns:
    {
        "parser_completeness":      {"score": int, "reasoning": str},
        "parser_no_hallucination":  {"score": int, "reasoning": str},
        "parser_occasion_detection":{"score": int, "reasoning": str},
    }
"""
from __future__ import annotations

import json
import traceback

from evaluation._client import TEXT_MODEL, create_json_response

PARSER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "parser_completeness": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "score": {"type": "integer", "minimum": 1, "maximum": 5},
                "reasoning": {"type": "string"},
            },
            "required": ["score", "reasoning"],
        },
        "parser_no_hallucination": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "score": {"type": "integer", "minimum": 1, "maximum": 5},
                "reasoning": {"type": "string"},
            },
            "required": ["score", "reasoning"],
        },
        "parser_occasion_detection": {
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
        "parser_completeness",
        "parser_no_hallucination",
        "parser_occasion_detection",
    ],
}

PARSER_PROMPT = """\
You are an expert evaluator for a fashion recommendation system.

Evaluate the output of a query parser on THREE dimensions simultaneously.
Think through each dimension carefully, then respond with a single JSON object.

=== 1. PARSER COMPLETENESS ===
Criteria:
    Evaluate whether the query parser extracted ALL constraints that the user
    explicitly mentioned in their natural-language query.

Steps:
  Step 1: Read the user's query carefully.
  Step 2: Identify every constraint the user explicitly stated: item categories,
          colors, fit preferences, price limits, and occasion.
  Step 3: Compare these with the parsed output fields (categories, colors, fit,
          price_min, price_max, occasion_mode, occasion_target).
  Step 4: Note any constraint mentioned by the user that was NOT extracted.
  Step 5: Assign a score from 1 to 5:
            1 = missed most constraints (3 or more missed)
            2 = missed several constraints (2 missed)
            3 = missed exactly one constraint
            4 = extracted nearly all — at most a minor ambiguous gap
            5 = perfectly complete — every stated constraint was captured

=== 2. PARSER NO HALLUCINATION ===
Criteria:
    Evaluate whether the query parser avoided extracting constraints that the
    user did NOT explicitly mention or clearly imply in their query.

Steps:
  Step 1: Read the user's query carefully.
  Step 2: Read the parsed output: categories, colors, fit, price range, occasion.
  Step 3: For each extracted constraint, verify it was explicitly stated or
          clearly implied by the user — not just a plausible guess.
  Step 4: Identify any constraints in the parsed output NOT mentioned in the query.
  Step 5: Assign a score from 1 to 5:
            1 = many hallucinated constraints (3 or more invented)
            2 = several hallucinated constraints (2 invented)
            3 = exactly one hallucinated constraint
            4 = no hallucinations, but one constraint is a borderline over-inference
            5 = perfectly clean — no invented or over-inferred constraints

=== 3. PARSER OCCASION DETECTION ===
Criteria:
    Evaluate whether the parser correctly identified whether an occasion was
    mentioned in the query and mapped it to the correct occasion label.

Steps:
  Step 1: Read the user's query.
  Step 2: Determine whether an occasion is clearly mentioned.
  Step 3: Check the parsed output's occasion field: mode ('on'/'off') and target label.
  Step 4: If NO occasion was mentioned: mode should be 'off' — penalise a false positive.
  Step 5: If an occasion WAS mentioned: mode should be 'on' and target should match
          the correct occasion type — penalise if wrong or missing.
  Step 6: Assign a score from 1 to 5:
            1 = completely wrong (e.g. occasion present but mode='off', or opposite error)
            2 = wrong direction — clear false positive or false negative
            3 = detected presence correctly but mapped to wrong occasion label
            4 = correct mode and close label match, minor wording/capitalisation difference
            5 = perfectly correct — right mode and exact occasion label

Evidence:
{evidence}

Respond ONLY with valid JSON in this exact format:
{{
  "parser_completeness":       {{"score": <integer 1-5>, "reasoning": "<step-by-step evaluation>"}},
  "parser_no_hallucination":   {{"score": <integer 1-5>, "reasoning": "<step-by-step evaluation>"}},
  "parser_occasion_detection": {{"score": <integer 1-5>, "reasoning": "<step-by-step evaluation>"}}
}}"""


def run_parser_judge(query: str, parsed: dict) -> dict:
    """Evaluate query parsing quality on 3 rubrics in a single text call.

    Args:
        query:  Original user query string.
        parsed: Output of parse_query_llm(query).

    Returns:
        {rubric_name: {"score": int, "reasoning": str}, ...}
    """
    constraints = parsed.get("constraints") or {}
    occ = parsed.get("occasion") or {}

    evidence = {
        "query": query,
        "parsed_output": {
            "categories":     constraints.get("categories", []),
            "colors":         constraints.get("colors", []),
            "fit":            constraints.get("fit", []),
            "price_min":      constraints.get("price_min"),
            "price_max":      constraints.get("price_max"),
            "occasion_mode":  occ.get("mode"),
            "occasion_target": occ.get("target"),
        },
    }

    prompt = PARSER_PROMPT.format(
        evidence=json.dumps(evidence, indent=2, ensure_ascii=False)
    )

    try:
        data = create_json_response(
            model=TEXT_MODEL,
            instructions="You are an expert evaluator for a fashion recommendation system.",
            user_text=prompt,
            schema_name="parser_evaluation",
            schema=PARSER_SCHEMA,
        )
        result = {}
        for rubric in ("parser_completeness", "parser_no_hallucination", "parser_occasion_detection"):
            entry = data.get(rubric, {})
            score = int(entry.get("score", -1))
            if not (1 <= score <= 5):
                score = -1
            result[rubric] = {"score": score, "reasoning": entry.get("reasoning", "")}
        return result
    except Exception as exc:
        traceback.print_exc()
        return {
            rubric: {"score": -1, "reasoning": f"Judge error: {exc}"}
            for rubric in (
                "parser_completeness",
                "parser_no_hallucination",
                "parser_occasion_detection",
            )
        }
