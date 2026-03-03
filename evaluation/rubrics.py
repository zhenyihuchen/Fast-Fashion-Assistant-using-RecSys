"""Evaluation rubric definitions for LLM-as-a-Judge (G-Eval pattern).

Each rubric specifies:
  - name:       metric identifier used in result dicts
  - criteria:   what is being evaluated (shown to the judge as task description)
  - steps:      chain-of-thought evaluation steps (the G-Eval "evaluation_steps")
  - params:     which input fields the judge requires
  - judge_type: "text" (Groq) or "multimodal" (Claude with image)
  - level:      "item" | "set" | "parser" | "pairwise"
  - scale:      (min, max) integer score range

RAG metric mapping
------------------
  Context Relevance  → ITEM_RELEVANCE
  Groundedness       → EXPLANATION_QUALITY (merged with quality)
  Answer Relevance   → SET_ANSWER_RELEVANCE
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Rubric:
    name: str
    criteria: str
    steps: list[str]
    params: list[str]
    judge_type: str           # "text" | "multimodal"
    level: str = "item"       # "item" | "set" | "parser" | "pairwise"
    scale: tuple[int, int] = (1, 5)


# ── Item-level rubrics ──────────────────────────────────────────────────────

ITEM_RELEVANCE = Rubric(
    name="item_relevance",
    criteria=(
        "Evaluate whether the recommended fashion product is relevant to the "
        "user's search query. Adapted from the RAG metric 'Context Relevance': "
        "checks if the retrieved item aligns with what the user was looking for."
    ),
    steps=[
        "Read the user query and identify what the user is looking for: item type, color, fit, price range.",
        "Check if the product category matches what the user requested.",
        "Check if the product color aligns with any color specified in the query.",
        "Check if the product fit or style matches any fit preference mentioned.",
        "Check if the price is within any stated price range.",
        "Consider the overall semantic alignment between the query intent and the product name and description.",
        "Assign a score from 1 to 5: 1=completely irrelevant, 2=mostly irrelevant, "
        "3=partially relevant, 4=mostly relevant, 5=highly relevant.",
    ],
    params=["query", "parsed_constraints", "product_metadata"],
    judge_type="text",
    level="item",
)

OCCASION_APPROPRIATENESS = Rubric(
    name="occasion_appropriateness",
    criteria=(
        "Evaluate whether the recommended product is visually and contextually "
        "appropriate for the stated occasion, considering both the product image "
        "and its metadata."
    ),
    steps=[
        "Identify the stated occasion (e.g. wedding, job interview, date night, festival, New Year's Eve).",
        "Consider what clothing styles, formality levels, and aesthetics are typical for this occasion.",
        "Examine the product image and assess its visual style, formality level, and overall aesthetic.",
        "Review the product metadata (category, name, color) to further understand the item.",
        "Determine if a person attending this occasion would realistically wear this item.",
        "Consider both visual cues from the image and textual information from the metadata.",
        "Assign a score from 1 to 5: 1=clearly inappropriate, 2=unlikely to be appropriate, "
        "3=could work in some contexts, 4=appropriate for the occasion, "
        "5=clearly and ideally suited for the occasion.",
    ],
    params=["query", "occasion_target", "product_metadata", "image_url"],
    judge_type="multimodal",
    level="item",
)

EXPLANATION_QUALITY = Rubric(
    name="explanation_quality",
    criteria=(
        "Evaluate whether the generated product explanation is both grounded in the "
        "actual product metadata (no hallucinated attributes) and genuinely useful "
        "for a shopper making a purchase decision."
    ),
    steps=[
        "Read the product metadata: name, category, color, price, and description.",
        "Read the user query and the stated occasion (if any).",
        "Read the generated explanation.",
        "For each claim in the explanation, verify it can be directly confirmed from the metadata; "
        "flag any attributes that are NOT in the metadata as hallucinations.",
        "Assess whether the explanation is specific — does it mention concrete product attributes "
        "rather than generic praise that could apply to any item?",
        "Assess whether the explanation connects the product to the user's query and/or occasion "
        "with clear, logical reasoning.",
        "Penalise both hallucinated claims and generic filler statements.",
        "Assign a score from 1 to 5: 1=hallucinated or generic and unhelpful, "
        "2=multiple issues (hallucinations or very generic), "
        "3=mostly grounded and somewhat useful, "
        "4=well-grounded and specific, "
        "5=fully grounded, specific, and genuinely helpful.",
    ],
    params=["query", "occasion_target", "product_metadata", "explanation"],
    judge_type="text",
    level="item",
)

# ── Set-level rubrics ───────────────────────────────────────────────────────

SET_ANSWER_RELEVANCE = Rubric(
    name="set_answer_relevance",
    criteria=(
        "Evaluate whether the complete set of recommended products collectively "
        "addresses the user's overall shopping intent. "
        "Adapted from the RAG metric 'Answer Relevance'."
    ),
    steps=[
        "Read the user query and identify the overall shopping intent.",
        "Review all recommended products as a set (names, categories, colors).",
        "Assess whether the set collectively covers what the user asked for.",
        "Check whether there is reasonable variety and diversity across the 5 recommendations.",
        "Assess whether the set is coherent with the user's stated occasion (if any).",
        "Consider whether there are glaring omissions or mismatches in the set.",
        "Assign a score from 1 to 5: 1=set completely misses the intent, 2=partially relevant, "
        "3=addresses intent moderately well, 4=addresses intent well with minor gaps, "
        "5=set fully and comprehensively addresses the user's intent.",
    ],
    params=["query", "occasion_target", "products_summary"],
    judge_type="text",
    level="set",
)

# ── Parser evaluation rubrics ───────────────────────────────────────────────

PARSER_COMPLETENESS = Rubric(
    name="parser_completeness",
    criteria=(
        "Evaluate whether the query parser extracted all constraints that the "
        "user explicitly mentioned in their natural-language query."
    ),
    steps=[
        "Read the user's query carefully.",
        "Identify all constraints the user explicitly stated: item categories, colors, "
        "fit preferences, price limits, and occasion.",
        "Compare these with the parsed output fields (categories, colors, fit, price_min, price_max, occasion).",
        "Note any constraints mentioned by the user that were NOT extracted in the parsed output.",
        "Assign a score from 1 to 5: 1=missed most constraints, 2=missed several, "
        "3=missed one constraint, 4=extracted nearly all with very minor gaps, "
        "5=perfectly complete extraction.",
    ],
    params=["query", "parsed_output"],
    judge_type="text",
    level="parser",
)

PARSER_NO_HALLUCINATION = Rubric(
    name="parser_no_hallucination",
    criteria=(
        "Evaluate whether the query parser avoided extracting constraints that "
        "the user did NOT explicitly mention in their query."
    ),
    steps=[
        "Read the user's query carefully.",
        "Read the parsed output: categories, colors, fit, price range, occasion.",
        "For each extracted constraint, verify that it was explicitly stated or clearly implied by the user.",
        "Identify any constraints present in the parsed output that were NOT mentioned in the query.",
        "Assign a score from 1 to 5: 1=many hallucinated constraints, 2=several hallucinated, "
        "3=one hallucinated constraint, 4=no hallucinations but minor ambiguity, "
        "5=perfectly clean with no invented constraints.",
    ],
    params=["query", "parsed_output"],
    judge_type="text",
    level="parser",
)

PARSER_OCCASION_DETECTION = Rubric(
    name="parser_occasion_detection",
    criteria=(
        "Evaluate whether the parser correctly identified whether an occasion was "
        "mentioned in the query and mapped it to the correct occasion label."
    ),
    steps=[
        "Read the user's query.",
        "Determine whether an occasion is clearly mentioned in the query.",
        "Check the parsed output's occasion field: mode ('on'/'off') and the target label.",
        "If no occasion was mentioned: mode should be 'off' — penalise a false positive.",
        "If an occasion was mentioned: mode should be 'on' and target should match the correct "
        "occasion type — penalise if wrong or missing.",
        "Assign a score from 1 to 5: 1=completely wrong, 2=wrong direction (false positive/negative), "
        "3=detected presence but mapped to wrong occasion, 4=correct with minor label mismatch, "
        "5=perfectly correct detection and mapping.",
    ],
    params=["query", "parsed_output"],
    judge_type="text",
    level="parser",
)

# ── Cross-model pairwise rubric ─────────────────────────────────────────────

CROSS_MODEL_PREFERENCE = Rubric(
    name="cross_model_preference",
    criteria=(
        "Given two sets of fashion product recommendations for the same query from "
        "two retrieval models (CLIP and FashionCLIP), determine which set better "
        "serves the user's needs. This is a pairwise comparison, not a single score."
    ),
    steps=[
        "Read the user query, parsed constraints, and stated occasion.",
        "Review Set A (CLIP model): the 5 recommended products with their metadata.",
        "Review Set B (FashionCLIP model): the 5 recommended products with their metadata.",
        "Compare both sets on relevance to the user's query and hard constraints.",
        "Compare both sets on occasion appropriateness.",
        "Compare both sets on diversity (do they offer meaningfully different options?).",
        "Consider which set better reflects the user's explicit constraints and overall intent.",
        "Output JSON with: winner ('clip', 'fashion_clip', or 'tie'), "
        "clip_score (1-5), fashionclip_score (1-5), and a brief reasoning.",
    ],
    params=["query", "parsed_constraints", "clip_products", "fashionclip_products"],
    judge_type="text",
    level="pairwise",
)

# ── Registries ──────────────────────────────────────────────────────────────

ITEM_RUBRICS: list[Rubric] = [
    ITEM_RELEVANCE,
    OCCASION_APPROPRIATENESS,
    EXPLANATION_QUALITY,
]

SET_RUBRICS: list[Rubric] = [SET_ANSWER_RELEVANCE]

PARSER_RUBRICS: list[Rubric] = [
    PARSER_COMPLETENESS,
    PARSER_NO_HALLUCINATION,
    PARSER_OCCASION_DETECTION,
]
