"""Shared API client, constants, and utilities for all per-level judges."""
from __future__ import annotations

import base64
import json
import os
import re
import threading
from pathlib import Path

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
