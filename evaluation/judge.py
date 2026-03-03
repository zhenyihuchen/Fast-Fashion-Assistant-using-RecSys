"""LLM judge implementations following the G-Eval methodology.

Three judge backends (all OpenAI):
  - run_text_judge()       — gpt-4o-mini: fast, text-only metrics
  - run_multimodal_judge() — gpt-4o-mini: occasion_appropriateness
                             (base64 product image + metadata + query via vision API)
  - run_pairwise_judge()   — gpt-4o-mini: cross-model preference (returns winner dict)

G-Eval pattern (DeepEval-inspired, implemented directly):
  1. Build a prompt from the rubric's criteria + evaluation steps
  2. Ask the judge to reason step by step through each step
  3. Ask for a final integer score
  4. Parse and return {"score": int, "reasoning": str}

References:
  Liu et al. (2023) "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"
  DeepEval G-Eval metric: https://docs.confident-ai.com/docs/metrics-llm-evals
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

TEXT_JUDGE_MODEL = os.getenv("OPENAI_TEXT_JUDGE_MODEL", "gpt-4o-mini")
# MULTIMODAL_JUDGE_MODEL = os.getenv("OPENAI_MULTIMODAL_JUDGE_MODEL", "gpt-4o-mini")
MULTIMODAL_JUDGE_MODEL = os.getenv("OPENAI_MULTIMODAL_JUDGE_MODEL", "gpt-4o")
TIMEOUT = 60

# max_retries=5: SDK automatically retries 429 rate-limit errors with exponential
# backoff (waits scale from ~500ms up). Parallel vision calls with large base64
# images can momentarily saturate the per-minute token window; retries resolve this
# without any scores being recorded as -1.
_client = OpenAI(api_key=OPENAI_API_KEY, max_retries=5)

# Limit concurrent multimodal (vision) calls to 3. Each base64 product image costs
# ~5 000 tokens; firing 10 at once = ~50 000 token burst that saturates the 200 K
# TPM window. Capping at 3 keeps the burst well under the limit.
_multimodal_semaphore = threading.Semaphore(3)

from evaluation.rubrics import Rubric


# ── Prompt builders ─────────────────────────────────────────────────────────

def _build_geval_prompt(rubric: Rubric, inputs: dict) -> str:
    steps_text = "\n".join(
        f"  Step {i + 1}: {step}" for i, step in enumerate(rubric.steps)
    )
    evidence = {k: v for k, v in inputs.items() if k != "image_url"}
    return (
        f"You are an expert evaluator for a fashion recommendation system.\n\n"
        f"Task: {rubric.name}\n"
        f"Criteria: {rubric.criteria}\n\n"
        f"Evaluation steps (follow in order):\n{steps_text}\n\n"
        f"Evidence:\n{json.dumps(evidence, indent=2, ensure_ascii=False)}\n\n"
        f"Think step by step through each evaluation step, then provide your final score.\n\n"
        f"Respond ONLY with valid JSON in this exact format:\n"
        f'{{"reasoning": "<your step-by-step evaluation>", '
        f'"score": <integer from {rubric.scale[0]} to {rubric.scale[1]}> }}'
    )


def _build_pairwise_prompt(rubric: Rubric, inputs: dict) -> str:
    steps_text = "\n".join(
        f"  Step {i + 1}: {step}" for i, step in enumerate(rubric.steps)
    )
    return (
        f"You are an expert evaluator for a fashion recommendation system.\n\n"
        f"Task: {rubric.name}\n"
        f"Criteria: {rubric.criteria}\n\n"
        f"Evaluation steps (follow in order):\n{steps_text}\n\n"
        f"Evidence:\n{json.dumps(inputs, indent=2, ensure_ascii=False)}\n\n"
        f"Think step by step, then provide your final judgment.\n\n"
        f"Respond ONLY with valid JSON in this exact format:\n"
        f'{{"reasoning": "<brief reasoning>", '
        f'"winner": "<clip | fashion_clip | tie>", '
        f'"clip_score": <integer 1-5>, '
        f'"fashionclip_score": <integer 1-5> }}'
    )


# ── Score parsers ────────────────────────────────────────────────────────────

def _parse_score_response(content: str, scale: tuple[int, int]) -> dict:
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


def _parse_pairwise_response(content: str) -> dict:
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


# ── Text judge (OpenAI) ──────────────────────────────────────────────────────

def run_text_judge(rubric: Rubric, inputs: dict) -> dict:
    """Runs a G-Eval style evaluation using OpenAI. Returns {"score": int, "reasoning": str}."""
    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY not set")

    prompt = _build_geval_prompt(rubric, inputs)
    resp = _client.chat.completions.create(
        model=TEXT_JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
        timeout=TIMEOUT,
    )
    return _parse_score_response(resp.choices[0].message.content, rubric.scale)


def run_pairwise_judge(rubric: Rubric, inputs: dict) -> dict:
    """Runs a pairwise cross-model comparison using OpenAI.
    Returns {"winner": str, "clip_score": int, "fashionclip_score": int, "reasoning": str}.
    """
    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY not set")

    prompt = _build_pairwise_prompt(rubric, inputs)
    resp = _client.chat.completions.create(
        model=TEXT_JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
        timeout=TIMEOUT,
    )
    return _parse_pairwise_response(resp.choices[0].message.content)


# ── Multimodal judge (OpenAI vision) ────────────────────────────────────────

def _local_image_as_b64(local_path: str) -> tuple[str, str] | None:
    """Read a local image file and return (base64_data, media_type).
    Returns None if the file doesn't exist or can't be read.
    """
    import base64
    path = Path(local_path)
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(suffix.lstrip("."), "image/jpeg")
    return base64.b64encode(path.read_bytes()).decode(), media_type


def run_multimodal_judge(rubric: Rubric, inputs: dict) -> dict:
    """Runs occasion_appropriateness evaluation using OpenAI's vision API (gpt-4o-mini).

    Uses the locally stored image (offline/data/images/) to avoid Zara CDN
    403 errors that occur when the API tries to fetch the image URL server-side.
    Returns {"score": int, "reasoning": str}.
    """
    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY not set")

    prompt_text = _build_geval_prompt(rubric, inputs)
    content: list[dict] = []

    local_path = inputs.get("local_image_path", "")
    b64_result = _local_image_as_b64(local_path) if local_path else None
    if b64_result:
        b64_data, media_type = b64_result
        content.append({"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64_data}"}})

    content.append({"type": "text", "text": prompt_text})

    with _multimodal_semaphore:
        resp = _client.chat.completions.create(
            model=MULTIMODAL_JUDGE_MODEL,
            messages=[{"role": "user", "content": content}],
            # temperature=0,  # omitted: reasoning models (o1/o3/o4-mini) only support default (1)
            response_format={"type": "json_object"},
            timeout=TIMEOUT,
        )
    return _parse_score_response(resp.choices[0].message.content, rubric.scale)


# ── Dispatcher ───────────────────────────────────────────────────────────────

def run_judge(rubric: Rubric, inputs: dict) -> dict:
    """Route to the correct judge based on rubric.judge_type and rubric.level."""
    if rubric.level == "pairwise":
        return run_pairwise_judge(rubric, inputs)
    if rubric.judge_type == "multimodal":
        return run_multimodal_judge(rubric, inputs)
    return run_text_judge(rubric, inputs)
