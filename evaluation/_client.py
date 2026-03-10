"""Shared API client, constants, and utilities for all per-level judges."""
from __future__ import annotations

import base64
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

TIMEOUT = 60

# ── OpenAI (active) ──────────────────────────────────────────────────────────
TEXT_MODEL       = os.getenv("OPENAI_TEXT_JUDGE_MODEL",       "gpt-5-mini")
MULTIMODAL_MODEL = os.getenv("OPENAI_MULTIMODAL_JUDGE_MODEL", "gpt-5-mini")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    max_retries=5,
)

# ── Z.AI (commented out) ─────────────────────────────────────────────────────
# TEXT_MODEL       = os.getenv("ZAI_TEXT_MODEL",       "glm-4.7-flash")
# MULTIMODAL_MODEL = os.getenv("ZAI_MULTIMODAL_MODEL", "glm-4.6v-flash")
#
# client = OpenAI(
#     api_key=os.getenv("ZAI_API_KEY"),
#     base_url="https://api.z.ai/api/paas/v4/",
#     max_retries=5,
# )
# ─────────────────────────────────────────────────────────────────────────────

# Cap concurrent vision calls to 3 — each base64 image is ~5K tokens.
multimodal_semaphore = threading.Semaphore(3)


def load_image_b64(local_path: str) -> tuple[str, str] | None:
    """Read a local image and return (base64_data, media_type), or None if missing."""
    path = Path(local_path)
    if not path.exists():
        return None
    ext = path.suffix.lower().lstrip(".")
    media = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png",  "webp": "image/webp",
    }.get(ext, "image/jpeg")
    return base64.b64encode(path.read_bytes()).decode(), media


def parse_json(content: str) -> dict:
    """Strip markdown fences and parse JSON; raises json.JSONDecodeError on failure."""
    cleaned = re.sub(r"```(?:json)?|```", "", content).strip()
    return json.loads(cleaned)


def create_json_response(
    *,
    model: str,
    instructions: str | None,
    user_text: str,
    schema_name: str,
    schema: dict[str, Any],
    images: list[dict[str, str]] | None = None,
) -> dict:
    """Call the Responses API with low-cost GPT-5 defaults and JSON Schema output."""
    content: list[dict[str, Any]] = []
    if user_text:
        content.append({"type": "input_text", "text": user_text})
    for image in images or []:
        content.append(
            {
                "type": "input_image",
                "image_url": image["url"],
                "detail": image.get("detail", "low"),
            }
        )

    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=[{"role": "user", "content": content}],
        reasoning={"effort": "low"},
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema,
                "strict": True,
            },
            "verbosity": "low",
        },
        timeout=TIMEOUT,
    )
    return parse_json(response.output_text)
