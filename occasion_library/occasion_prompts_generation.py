from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

OUTPUT_PATH = BASE_DIR / "occasion_library" / "occasion_prompts_v2.json"

OCCASIONS = {
    "wedding_guest": "wedding guest attire (formal or semi-formal)",
    "party_night_out": "party night-out outfits (going-out, club, cocktail, glam)",
    "work_office": "office-appropriate outfits (workwear, business casual)",
    "formal_event": "formal event outfits (evening wear, cocktail, black-tie inspired)",
    "casual_everyday": "casual everyday outfits (relaxed, comfortable, street style)",
    "date_night": "date night outfits (romantic, smart casual, evening)",
    "winter_outerwear": "winter outerwear outfits (coats, layered cold-weather looks)",
    "smart_casual": "smart casual outfits (polished yet relaxed)",
}


def _call_groq(messages: list[dict[str, str]]) -> dict:
    if not GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY is not set")
    client = Groq(api_key=GROQ_API_KEY)
    chat = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    return json.loads(chat.choices[0].message.content)


def _build_messages(occasion_key: str, occasion_desc: str) -> list[dict[str, str]]:
    system = (
        "You generate high-quality, semantically diverse short prompts for a fashion "
        "recommendation system. Output strictly valid JSON."
    )
    user = {
        "task": "Generate 10 prompts for the given occasion.",
        "occasion_key": occasion_key,
        "occasion_description": occasion_desc,
        "requirements": [
            "Each prompt should be 3-8 words.",
            "No brand names.",
            "No duplicate phrasing.",
            "Cover diverse styles within the occasion.",
            "Keep prompts concise and natural.",
        ],
        "output_schema": {
            "occasion": "string",
            "prompts": ["string"],
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user)},
    ]


def generate_all() -> dict:
    out: dict[str, list[str]] = {}
    for key, desc in OCCASIONS.items():
        messages = _build_messages(key, desc)
        resp = _call_groq(messages)
        prompts = resp.get("prompts", [])
        cleaned = [str(p).strip() for p in prompts if str(p).strip()]
        out[key] = cleaned
    return out


def main() -> None:
    prompts = generate_all()
    payload = {
        "meta": {
            "language": "en",
            "version": "v2",
            "notes": "Occasion prompt library for Zara CRS",
        },
        "occasions": prompts,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
