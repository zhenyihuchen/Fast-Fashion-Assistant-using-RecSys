from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq


BASE_DIR = Path(__file__).resolve().parent.parent
OCCASION_PROMPTS = BASE_DIR / "occasion_library" / "occasion_prompts.json"
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

TIMEOUT_SECONDS = 60

CATEGORY_KEYWORDS = [
    "jackets",
    "coats",
    "dresses",
    "tops",
    "shirts",
    "bodies",
    "jeans",
    "trousers",
    "t-shirts",
    "cardigans",
    "skirts",
    "sweatshirts",
    "shoes",
    "bags",
]

COLOR_KEYWORDS = [
    "089",
    "758",
    "831",
    "anthracite grey",
    "apple green",
    "aubergine",
    "beige",
    "beige marl",
    "beige-pink",
    "biscuit",
    "black",
    "black / brown",
    "black / ecru",
    "black / white",
    "black/orange",
    "blue",
    "blue / grey",
    "blue / navy",
    "blue grey",
    "blue/black",
    "blue/white",
    "bluish",
    "bordeaux/ecru",
    "bottle green",
    "bronze",
    "brown",
    "brown / ecru",
    "brown / green",
    "brown / taupe",
    "brown marl",
    "brown stripes",
    "brown vigore",
    "brown-blue",
    "brown/white",
    "burgundy",
    "burgundy red",
    "burnt orange",
    "camel",
    "caramel",
    "cava",
    "chalk pink",
    "charcoal",
    "chocolate",
    "chocolate brown",
    "cobalt",
    "cognac brown",
    "cream",
    "dark anthracite",
    "dark beige",
    "dark bottle green",
    "dark brown",
    "dark green",
    "dark grey",
    "dark grey marl",
    "dark indigo",
    "dark khaki",
    "dark mink",
    "dark navy",
    "dark red",
    "dark russet",
    "dark tan",
    "denim blue",
    "ecru",
    "ecru / beige",
    "ecru / black",
    "ecru / blue",
    "ecru / brown",
    "ecru / marl",
    "ecru / maroon",
    "ecru / red",
    "ecru white",
    "ecru/khaki",
    "faded pink",
    "garnet",
    "green",
    "green / blue",
    "green / ecru",
    "green marl",
    "green stripe",
    "grey",
    "grey / beige",
    "grey / blue",
    "grey / natural",
    "grey / tan",
    "grey green",
    "grey marl",
    "greys",
    "indigo",
    "ink blue",
    "khaki",
    "khaki green",
    "leopard",
    "light beige",
    "light blue",
    "light brown",
    "light camel",
    "light green",
    "light grey",
    "light mink",
    "light tan",
    "lilac",
    "marsala",
    "mauve",
    "mid khaki",
    "mid-blue",
    "mid-camel",
    "mid-ecru",
    "mid-grey",
    "mid-mink",
    "mid-pink",
    "mink",
    "mink marl",
    "moss green",
    "multicoloured",
    "mustard",
    "navy / white",
    "navy blue",
    "olive green",
    "only one",
    "orange",
    "orange-green",
    "oyster-white",
    "pink",
    "pink / white",
    "purple",
    "purples",
    "red",
    "red / coral",
    "sand",
    "sand / blue",
    "sand / marl",
    "sand brown",
    "sand/brown",
    "sea green",
    "silver",
    "sky blue",
    "stone",
    "striped",
    "taupe grey",
    "tobacco",
    "vanilla",
    "violet",
    "washed green",
    "white",
    "white / green",
    "white / grey",
    "white / navy",
    "white / red",
    "wine",
    "yellow",
]

FIT_KEYWORDS = [
    "oversized",
    "slim",
    "skinny",
    "loose",
    "wide leg",
    "straight leg",
    "straight",
    "cropped",
    "midi",
    "maxi",
    "mini",
    "short",
    "long sleeve",
    "short sleeve",
]


def normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


@dataclass
class QueryConstraints:
    categories: list[str]
    colors: list[str]
    price_min: float | None
    price_max: float | None
    fit: list[str]


@dataclass
class OccasionDecision:
    mode: str
    target: str | None
    score: float | None


def _load_occasions() -> list[str]:
    data = json.loads(OCCASION_PROMPTS.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("occasions"), dict):
        return sorted(list(data["occasions"].keys()))
    return []


def _build_prompt(query: str) -> list[dict[str, str]]:
    occasions = _load_occasions()
    system = (
        "You are a strict JSON generator for fashion query parsing. "
        "Return ONLY valid JSON matching the schema below. "
        "Do not include extra keys or explanations."
    )
    schema = {
        "normalized": "string",
        "constraints": {
            "categories": "list of strings (from allowed categories)",
            "colors": "list of strings (from allowed colors)",
            "price_min": "number or null",
            "price_max": "number or null",
            "fit": "list of strings (from allowed fit keywords)",
        },
        "occasion": {
            "mode": "\"on\" or \"off\"",
            "target": "string from allowed occasions or null",
            "score": "number 0-1 or null",
        },
        "confidence": {
            "overall": "number 0-1",
            "occasion": "number 0-1",
        },
    }
    user = {
        "query": query,
        "allowed_categories": CATEGORY_KEYWORDS,
        "allowed_colors": COLOR_KEYWORDS,
        "allowed_fit": FIT_KEYWORDS,
        "allowed_occasions": occasions,
        "output_schema": schema,
        "notes": [
            "Use allowed values exactly; lowercase strings.",
            "If no price is found, use nulls.",
            "If no occasion is clear, set mode=off, target=null, score to your best guess.",
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user)},
    ]


def _call_groq(messages: list[dict[str, str]]) -> dict[str, Any]:
    if not GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY is not set")
    client = Groq(api_key=GROQ_API_KEY)
    chat = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
        timeout=TIMEOUT_SECONDS,
    )
    return {"content": chat.choices[0].message.content}


def _parse_llm_json(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {e}") from e


def parse_query_llm(query: str) -> dict[str, Any]:
    normalized = normalize_query(query)
    messages = _build_prompt(normalized)
    resp = _call_groq(messages)
    content = resp["content"]
    result = _parse_llm_json(content)
    result["normalized"] = normalized
    return result


if __name__ == "__main__":
    test_queries = [
        # "I want a pair of wine color jeans for a chill and casual day in the park, I like the cut to be straight",
        "I am looking for a night short dress red but that costs red that costs less than 50 euros",
    ]
    for q in test_queries:
        print("\nQuery:", q)
        print(json.dumps(parse_query_llm(q), indent=2))
        #Save JSON response to a file for inspection
        output_path = BASE_DIR / "online" / "test_query_outputs"
        output_path.mkdir(parents=True, exist_ok=True)
        file_name = re.sub(r"\W+", "_", q.strip().lower())[:50] + ".json"
        (output_path / file_name).write_text(json.dumps(parse_query_llm(q), indent=2), encoding="utf-8")
