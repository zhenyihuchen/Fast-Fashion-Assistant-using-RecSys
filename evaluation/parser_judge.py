"""Parser-level judge: evaluates all 3 parser rubrics in a single text API call.

Rubric prompts are loaded from a YAML file (default: prompts/parser_judge.yaml).

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
from pathlib import Path

import yaml

from evaluation._client import TEXT_MODEL, create_json_response

DEFAULT_PROMPTS_PATH = Path(__file__).parent / "prompts" / "parser_judge.yaml"

_prompts_cache: dict[str, dict] = {}


def _get_prompts(path: Path = DEFAULT_PROMPTS_PATH) -> dict:
    key = str(path)
    if key not in _prompts_cache:
        _prompts_cache[key] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _prompts_cache[key]


def _build_parser_prompt(prompts: dict) -> str:
    rubric_order = ["parser_completeness", "parser_no_hallucination", "parser_occasion_detection"]
    blocks = []
    for i, name in enumerate(rubric_order, 1):
        rubric = prompts[name]
        title = name.upper().replace("_", " ")
        scale_lines = "\n".join(
            f"            {k} = {v}" for k, v in rubric["scale"].items()
        )
        blocks.append(
            f"=== {i}. {title} ===\n"
            f"Criteria:\n    {rubric['criteria'].strip()}\n\n"
            f"Steps:\n  {rubric['steps'].strip()}\n"
            f"{scale_lines}\n"
        )

    return (
        "You are an expert evaluator for a fashion recommendation system.\n\n"
        "Evaluate the output of a query parser on THREE dimensions simultaneously.\n"
        "Think through each dimension carefully, then respond with a single JSON object.\n\n"
        + "\n".join(blocks)
        + "\nEvidence:\n{evidence}\n\n"
        "Respond ONLY with valid JSON in this exact format:\n"
        '{{\n'
        '  "parser_completeness":       {{"score": <integer 1-5>, "reasoning": "<step-by-step evaluation>"}},\n'
        '  "parser_no_hallucination":   {{"score": <integer 1-5>, "reasoning": "<step-by-step evaluation>"}},\n'
        '  "parser_occasion_detection": {{"score": <integer 1-5>, "reasoning": "<step-by-step evaluation>"}}\n'
        '}}'
    )


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

def run_parser_judge(query: str, parsed: dict, prompts_path: Path = DEFAULT_PROMPTS_PATH) -> dict:
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

    prompts = _get_prompts(prompts_path)
    prompt_template = _build_parser_prompt(prompts)
    prompt = prompt_template.format(
        evidence=json.dumps(evidence, indent=2, ensure_ascii=False)
    )

    try:
        data = create_json_response(
            model=TEXT_MODEL,
            instructions=prompts.get("system", "You are an expert evaluator for a fashion recommendation system."),
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
