"""Static-prompt LLM-as-a-Judge — alternative to judge.py.

Each rubric's prompt is written as a plain triple-quoted string instead of
being assembled by _build_geval_prompt(). The evidence dict is JSON-serialised
and injected via a single {evidence} placeholder.

Drop-in compatible: run_judge_static(rubric, inputs) has the same signature
and return value as run_judge(rubric, inputs) in judge.py.

Models used (OpenAI):
  Text + pairwise : gpt-4o-mini
  Multimodal      : gpt-4o-mini (vision)
"""
from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from evaluation.rubrics import Rubric

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TEXT_JUDGE_MODEL = os.getenv("OPENAI_TEXT_JUDGE_MODEL", "gpt-4o-mini")
MULTIMODAL_JUDGE_MODEL = os.getenv("OPENAI_MULTIMODAL_JUDGE_MODEL", "gpt-4o-mini")
TIMEOUT = 60


# ── Static prompt templates ──────────────────────────────────────────────────
# Each template has exactly one placeholder: {evidence}
# Literal braces in the JSON response format are escaped as {{ and }}.

ITEM_RELEVANCE_PROMPT = """\
You are an expert evaluator for a fashion recommendation system.

Task: item_relevance
Criteria: Evaluate whether the recommended fashion product is relevant to the
user's search query. Adapted from the RAG metric 'Context Relevance': checks
if the retrieved item aligns with what the user was looking for.

Evaluation steps (follow in order):
  Step 1: Read the user query and identify what the user is looking for: item type, color, fit, price range.
  Step 2: Check if the product category matches what the user requested.
  Step 3: Check if the product color aligns with any color specified in the query.
  Step 4: Check if the product fit or style matches any fit preference mentioned.
  Step 5: Check if the price is within any stated price range.
  Step 6: Consider the overall semantic alignment between the query intent and the product name and description.
  Step 7: Assign a score from 1 to 5: 1=completely irrelevant, 2=mostly irrelevant,
          3=partially relevant, 4=mostly relevant, 5=highly relevant.

Evidence:
{evidence}

Think step by step through each evaluation step, then provide your final score.

Respond ONLY with valid JSON in this exact format:
{{"reasoning": "<your step-by-step evaluation>", "score": <integer from 1 to 5>}}"""


OCCASION_APPROPRIATENESS_PROMPT = """\
You are an expert evaluator for a fashion recommendation system.

Task: occasion_appropriateness
Criteria: Evaluate whether the recommended product is visually and contextually
appropriate for the stated occasion, considering both the product image and its metadata.

Evaluation steps (follow in order):
  Step 1: Identify the stated occasion (e.g. wedding, job interview, date night, festival, New Year's Eve).
  Step 2: Consider what clothing styles, formality levels, and aesthetics are typical for this occasion.
  Step 3: Examine the product image (provided above) and assess its visual style, formality level, and overall aesthetic.
  Step 4: Review the product metadata (category, name, color) to further understand the item.
  Step 5: Determine if a person attending this occasion would realistically wear this item.
  Step 6: Consider both visual cues from the image and textual information from the metadata.
  Step 7: Assign a score from 1 to 5: 1=clearly inappropriate, 2=unlikely to be appropriate,
          3=could work in some contexts, 4=appropriate for the occasion,
          5=clearly and ideally suited for the occasion.

Evidence:
{evidence}

Think step by step through each evaluation step, then provide your final score.

Respond ONLY with valid JSON in this exact format:
{{"reasoning": "<your step-by-step evaluation>", "score": <integer from 1 to 5>}}"""


EXPLANATION_GROUNDEDNESS_PROMPT = """\
You are an expert evaluator for a fashion recommendation system.

Task: explanation_groundedness
Criteria: Evaluate whether the generated product explanation is grounded in the
actual product metadata, with no hallucinated or invented attributes.
Adapted from the RAG metric 'Groundedness'.

Evaluation steps (follow in order):
  Step 1: Read the product metadata: name, category, color, price, and description.
  Step 2: Read the generated explanation.
  Step 3: For each claim in the explanation, check whether it can be directly verified from the metadata.
  Step 4: Identify any attributes mentioned in the explanation that are NOT present in the metadata (hallucinations).
  Step 5: Identify any metadata that was incorrectly described or distorted in the explanation.
  Step 6: Assign a score from 1 to 5: 1=heavily hallucinated, 2=several unsupported claims,
          3=mostly grounded with minor additions, 4=well-grounded with very minor embellishment,
          5=fully grounded with no invented attributes.

Evidence:
{evidence}

Think step by step through each evaluation step, then provide your final score.

Respond ONLY with valid JSON in this exact format:
{{"reasoning": "<your step-by-step evaluation>", "score": <integer from 1 to 5>}}"""


EXPLANATION_QUALITY_PROMPT = """\
You are an expert evaluator for a fashion recommendation system.

Task: explanation_quality
Criteria: Evaluate whether the product explanation is specific, useful, and
genuinely helpful for a shopper making a purchase decision.

Evaluation steps (follow in order):
  Step 1: Read the user query and the stated occasion.
  Step 2: Read the product explanation.
  Step 3: Assess whether the explanation is specific — does it mention concrete product attributes
          rather than generic praise that could apply to any item?
  Step 4: Assess whether the explanation connects the product to the user's query and/or occasion
          with clear, logical reasoning.
  Step 5: Determine if the explanation would help a real shopper decide whether to consider this product.
  Step 6: Penalise generic statements such as 'this is a great item for any occasion'.
  Step 7: Assign a score from 1 to 5: 1=generic and unhelpful, 2=mostly generic,
          3=moderately useful, 4=specific and useful, 5=specific, concrete, and genuinely helpful.

Evidence:
{evidence}

Think step by step through each evaluation step, then provide your final score.

Respond ONLY with valid JSON in this exact format:
{{"reasoning": "<your step-by-step evaluation>", "score": <integer from 1 to 5>}}"""


CONSTRAINT_ADHERENCE_PROMPT = """\
You are an expert evaluator for a fashion recommendation system.

Task: constraint_adherence
Criteria: Evaluate whether the recommended product respects the hard constraints
explicitly stated in the user's query (color, category, fit, price range).

Evaluation steps (follow in order):
  Step 1: Identify the explicit constraints from the user query: color, category, fit, and price range.
  Step 2: If no hard constraints were stated, assign a score of 5 (vacuously satisfied).
  Step 3: Check whether the product color matches the requested color.
  Step 4: Check whether the product category matches the requested category.
  Step 5: Check whether the product fit or style matches the requested fit.
  Step 6: Check whether the product price is within the stated price range.
  Step 7: Assign a score from 1 to 5: 1=violates multiple constraints, 2=violates one key constraint,
          3=satisfies some but not all constraints, 4=satisfies all constraints with minor ambiguity,
          5=fully satisfies all stated constraints.

Evidence:
{evidence}

Think step by step through each evaluation step, then provide your final score.

Respond ONLY with valid JSON in this exact format:
{{"reasoning": "<your step-by-step evaluation>", "score": <integer from 1 to 5>}}"""


SET_ANSWER_RELEVANCE_PROMPT = """\
You are an expert evaluator for a fashion recommendation system.

Task: set_answer_relevance
Criteria: Evaluate whether the complete set of recommended products collectively
addresses the user's overall shopping intent. Adapted from the RAG metric 'Answer Relevance'.

Evaluation steps (follow in order):
  Step 1: Read the user query and identify the overall shopping intent.
  Step 2: Review all recommended products as a set (names, categories, colors).
  Step 3: Assess whether the set collectively covers what the user asked for.
  Step 4: Check whether there is reasonable variety and diversity across the 5 recommendations.
  Step 5: Assess whether the set is coherent with the user's stated occasion (if any).
  Step 6: Consider whether there are glaring omissions or mismatches in the set.
  Step 7: Assign a score from 1 to 5: 1=set completely misses the intent, 2=partially relevant,
          3=addresses intent moderately well, 4=addresses intent well with minor gaps,
          5=set fully and comprehensively addresses the user's intent.

Evidence:
{evidence}

Think step by step through each evaluation step, then provide your final score.

Respond ONLY with valid JSON in this exact format:
{{"reasoning": "<your step-by-step evaluation>", "score": <integer from 1 to 5>}}"""


PARSER_COMPLETENESS_PROMPT = """\
You are an expert evaluator for a fashion recommendation system.

Task: parser_completeness
Criteria: Evaluate whether the query parser extracted all constraints that the
user explicitly mentioned in their natural-language query.

Evaluation steps (follow in order):
  Step 1: Read the user's query carefully.
  Step 2: Identify all constraints the user explicitly stated: item categories, colors,
          fit preferences, price limits, and occasion.
  Step 3: Compare these with the parsed output fields (categories, colors, fit, price_min, price_max, occasion).
  Step 4: Note any constraints mentioned by the user that were NOT extracted in the parsed output.
  Step 5: Assign a score from 1 to 5: 1=missed most constraints, 2=missed several,
          3=missed one constraint, 4=extracted nearly all with very minor gaps,
          5=perfectly complete extraction.

Evidence:
{evidence}

Think step by step through each evaluation step, then provide your final score.

Respond ONLY with valid JSON in this exact format:
{{"reasoning": "<your step-by-step evaluation>", "score": <integer from 1 to 5>}}"""


PARSER_NO_HALLUCINATION_PROMPT = """\
You are an expert evaluator for a fashion recommendation system.

Task: parser_no_hallucination
Criteria: Evaluate whether the query parser avoided extracting constraints that
the user did NOT explicitly mention in their query.

Evaluation steps (follow in order):
  Step 1: Read the user's query carefully.
  Step 2: Read the parsed output: categories, colors, fit, price range, occasion.
  Step 3: For each extracted constraint, verify that it was explicitly stated or clearly implied by the user.
  Step 4: Identify any constraints present in the parsed output that were NOT mentioned in the query.
  Step 5: Assign a score from 1 to 5: 1=many hallucinated constraints, 2=several hallucinated,
          3=one hallucinated constraint, 4=no hallucinations but minor ambiguity,
          5=perfectly clean with no invented constraints.

Evidence:
{evidence}

Think step by step through each evaluation step, then provide your final score.

Respond ONLY with valid JSON in this exact format:
{{"reasoning": "<your step-by-step evaluation>", "score": <integer from 1 to 5>}}"""


PARSER_OCCASION_DETECTION_PROMPT = """\
You are an expert evaluator for a fashion recommendation system.

Task: parser_occasion_detection
Criteria: Evaluate whether the parser correctly identified whether an occasion was
mentioned in the query and mapped it to the correct occasion label.

Evaluation steps (follow in order):
  Step 1: Read the user's query.
  Step 2: Determine whether an occasion is clearly mentioned in the query.
  Step 3: Check the parsed output's occasion field: mode ('on'/'off') and the target label.
  Step 4: If no occasion was mentioned: mode should be 'off' — penalise a false positive.
  Step 5: If an occasion was mentioned: mode should be 'on' and target should match the correct
          occasion type — penalise if wrong or missing.
  Step 6: Assign a score from 1 to 5: 1=completely wrong, 2=wrong direction (false positive/negative),
          3=detected presence but mapped to wrong occasion, 4=correct with minor label mismatch,
          5=perfectly correct detection and mapping.

Evidence:
{evidence}

Think step by step through each evaluation step, then provide your final score.

Respond ONLY with valid JSON in this exact format:
{{"reasoning": "<your step-by-step evaluation>", "score": <integer from 1 to 5>}}"""


CROSS_MODEL_PREFERENCE_PROMPT = """\
You are an expert evaluator for a fashion recommendation system.

Task: cross_model_preference
Criteria: Given two sets of fashion product recommendations for the same query from
two retrieval models (CLIP and FashionCLIP), determine which set better serves
the user's needs. This is a pairwise comparison, not a single score.

Evaluation steps (follow in order):
  Step 1: Read the user query, parsed constraints, and stated occasion.
  Step 2: Review Set A (CLIP model): the 5 recommended products with their metadata.
  Step 3: Review Set B (FashionCLIP model): the 5 recommended products with their metadata.
  Step 4: Compare both sets on relevance to the user's query and hard constraints.
  Step 5: Compare both sets on occasion appropriateness.
  Step 6: Compare both sets on diversity (do they offer meaningfully different options?).
  Step 7: Consider which set better reflects the user's explicit constraints and overall intent.
  Step 8: Output the winner ('clip', 'fashion_clip', or 'tie') with individual scores for each model.

Evidence:
{evidence}

Think step by step, then provide your final judgment.

Respond ONLY with valid JSON in this exact format:
{{"reasoning": "<brief reasoning>", "winner": "<clip | fashion_clip | tie>", "clip_score": <integer 1-5>, "fashionclip_score": <integer 1-5>}}"""


# ── Prompt lookup ────────────────────────────────────────────────────────────

PROMPTS: dict[str, str] = {
    "item_relevance":             ITEM_RELEVANCE_PROMPT,
    "occasion_appropriateness":   OCCASION_APPROPRIATENESS_PROMPT,
    "explanation_groundedness":   EXPLANATION_GROUNDEDNESS_PROMPT,
    "explanation_quality":        EXPLANATION_QUALITY_PROMPT,
    "constraint_adherence":       CONSTRAINT_ADHERENCE_PROMPT,
    "set_answer_relevance":       SET_ANSWER_RELEVANCE_PROMPT,
    "parser_completeness":        PARSER_COMPLETENESS_PROMPT,
    "parser_no_hallucination":    PARSER_NO_HALLUCINATION_PROMPT,
    "parser_occasion_detection":  PARSER_OCCASION_DETECTION_PROMPT,
    "cross_model_preference":     CROSS_MODEL_PREFERENCE_PROMPT,
}


# ── OpenAI callers ───────────────────────────────────────────────────────────

def _openai_text(prompt: str) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=TEXT_JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
        timeout=TIMEOUT,
    )
    return resp.choices[0].message.content


def _openai_multimodal(prompt: str, local_image_path: str) -> str:
    content: list[dict] = []

    path = Path(local_image_path)
    if path.exists():
        suffix = path.suffix.lower().lstrip(".")
        media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                      "png": "image/png", "webp": "image/webp"}.get(suffix, "image/jpeg")
        b64 = base64.b64encode(path.read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}})

    content.append({"type": "text", "text": prompt})

    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=MULTIMODAL_JUDGE_MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        response_format={"type": "json_object"},
        timeout=TIMEOUT,
    )
    return resp.choices[0].message.content


# ── Response parsers ─────────────────────────────────────────────────────────

def _parse_score(content: str, scale: tuple[int, int]) -> dict:
    try:
        cleaned = re.sub(r"```(?:json)?|```", "", content).strip()
        data = json.loads(cleaned)
        score = int(data.get("score", -1))
        if not (scale[0] <= score <= scale[1]):
            score = -1
        return {"score": score, "reasoning": data.get("reasoning", "")}
    except Exception:
        match = re.search(r'"score"\s*:\s*(\d)', content)
        score = int(match.group(1)) if match else -1
        return {"score": score, "reasoning": content}


def _parse_pairwise(content: str) -> dict:
    try:
        cleaned = re.sub(r"```(?:json)?|```", "", content).strip()
        data = json.loads(cleaned)
        winner = data.get("winner", "tie")
        if winner not in {"clip", "fashion_clip", "tie"}:
            winner = "tie"
        return {
            "winner": winner,
            "clip_score": int(data.get("clip_score", -1)),
            "fashionclip_score": int(data.get("fashionclip_score", -1)),
            "reasoning": data.get("reasoning", ""),
        }
    except Exception:
        return {"winner": "tie", "clip_score": -1, "fashionclip_score": -1, "reasoning": content}


# ── Main entry point ─────────────────────────────────────────────────────────

def run_judge_static(rubric: Rubric, inputs: dict) -> dict:
    """Drop-in replacement for run_judge() in judge.py.

    Looks up the static prompt template by rubric.name, injects the evidence,
    and dispatches to the correct Groq backend.
    """
    if not GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY not set")

    template = PROMPTS.get(rubric.name)
    if template is None:
        raise ValueError(f"No static prompt defined for rubric '{rubric.name}'")

    # Strip image_url from evidence passed as text (image is sent separately for multimodal)
    evidence = {k: v for k, v in inputs.items() if k not in ("image_url", "local_image_path")}
    prompt = template.format(evidence=json.dumps(evidence, indent=2, ensure_ascii=False))

    if rubric.level == "pairwise":
        raw = _openai_text(prompt)
        return _parse_pairwise(raw)

    if rubric.judge_type == "multimodal":
        local_path = inputs.get("local_image_path", "")
        raw = _openai_multimodal(prompt, local_path)
        return _parse_score(raw, rubric.scale)

    raw = _openai_text(prompt)
    return _parse_score(raw, rubric.scale)
