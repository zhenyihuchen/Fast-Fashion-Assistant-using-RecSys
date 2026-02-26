from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
OCCASION_PROMPTS = BASE_DIR / "occasion_library" / "occasion_prompts.json"
OCCASION_EMB = BASE_DIR / "occasion_library" / "occasion_prompt_embeddings.npz"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

OCCASION_THRESHOLD = 0.30

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

CATEGORY_SYNONYMS = {
    "dress": "dresses",
    "jacket": "jackets",
    "coat": "coats",
    "top": "tops",
    "shirt": "shirts",
    "body": "bodies",
    "jean": "jeans",
    "trouser": "trousers",
    "tshirt": "t-shirts",
    "t shirt": "t-shirts",
    "cardigan": "cardigans",
    "skirt": "skirts",
    "sweatshirt": "sweatshirts",
    "shoe": "shoes",
    "bag": "bags",
}


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

COLOR_SYNONYMS = {
    "navy": "navy blue",
    "dark navy": "navy blue",
    "gray": "grey",
    "off white": "ecru",
    "cream": "ecru",
    "tan": "sand",
    "burgundy": "wine",
    "maroon": "burgundy",
    "olive": "olive green",
    "stone": "ecru",
    "camel": "light camel",
}
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


def normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _match_keywords(text: str, keywords: list[str]) -> list[str]:
    tokens = set(re.findall(r"[a-z0-9]+", text))
    matches = []
    for kw in keywords:
        kw_tokens = set(re.findall(r"[a-z0-9]+", kw))
        if kw in text or (kw_tokens and kw_tokens.issubset(tokens)):
            matches.append(kw)
    return matches


def extract_constraints(text: str) -> QueryConstraints:
    categories = _match_keywords(text, CATEGORY_KEYWORDS)
    for syn, target in CATEGORY_SYNONYMS.items():
        if syn in text:
            categories.append(target)
    categories = sorted(set(categories))
    colors = _match_keywords(text, COLOR_KEYWORDS)
    for syn, target in COLOR_SYNONYMS.items():
        if syn in text:
            colors.append(target)
    colors = sorted(set(colors))
    fit = _match_keywords(text, FIT_KEYWORDS)

    price_min = None
    price_max = None

    m = re.search(r"(under|max|less than|below)\s*\$?(\d+)", text)
    if m:
        price_max = float(m.group(2))

    m = re.search(r"(between)\s*\$?(\d+)\s*(and|-)\s*\$?(\d+)", text)
    if m:
        price_min = float(m.group(2))
        price_max = float(m.group(4))

    return QueryConstraints(
        categories=categories,
        colors=colors,
        price_min=price_min,
        price_max=price_max,
        fit=fit,
    )


def _load_occasions() -> dict[str, list[str]]:
    data = json.loads(OCCASION_PROMPTS.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("occasions"), dict):
        return {k: [str(v) for v in vals] for k, vals in data["occasions"].items()}
    return {}


def _explicit_occasion(text: str) -> str | None:
    occasions = _load_occasions()
    for name in occasions.keys():
        if name in text or name.replace("_", " ") in text:
            return name
    return None


def _embed_text(text: str) -> np.ndarray:
    try:
        import open_clip
        import torch
    except Exception:
        raise SystemExit("open_clip not installed; run `pip install open-clip-torch`")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.to(device)
    model.eval()

    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0]


def decide_occasion_mode(text: str, use_embedding: bool = True) -> OccasionDecision:
    if "no occasion" in text or "no specific occasion" in text:
        return OccasionDecision(mode="off", target=None, score=None)

    explicit = _explicit_occasion(text)
    if explicit:
        return OccasionDecision(mode="on", target=explicit, score=None)

    # Heuristic keyword rules for common ambiguity.
    if "park" in text or "everyday" in text or "casual" in text:
        return OccasionDecision(mode="on", target="casual_everyday", score=None)
    if "night" in text or "party" in text or "club" in text:
        if "date" in text or "romantic" in text or "dinner" in text:
            return OccasionDecision(mode="on", target="date_night", score=None)
        return OccasionDecision(mode="on", target="party_night_out", score=None)
    #TO-DO: make it scalable with reasoning done by LLMs (but this when everything else is done)

    # Token overlap scoring against occasion prompts.
    prompts = _load_occasions()
    if prompts:
        best_name = None
        best_score = 0
        tokens = set(re.findall(r"[a-z]+", text))
        for name, p_list in prompts.items():
            prompt_text = " ".join(p_list).lower()
            prompt_tokens = set(re.findall(r"[a-z]+", prompt_text))
            score = len(tokens & prompt_tokens)
            if score > best_score:
                best_score = score
                best_name = name
        if best_name and best_score > 0:
            return OccasionDecision(mode="on", target=best_name, score=float(best_score))

    if not use_embedding or not OCCASION_EMB.exists(): # if embedding inference is disabled or embeddings file is missing…
        return OccasionDecision(mode="off", target=None, score=None) # fallback to OFF

    q_emb = _embed_text(text)
    occ_data = np.load(OCCASION_EMB) 

    best_name = None # prepares to track the best‑matching occasion.
    best_score = -1.0
    for name in occ_data.files:
        occ_emb = occ_data[name]
        occ_mean = occ_emb.mean(axis=0)
        occ_mean = occ_mean / np.linalg.norm(occ_mean)
        score = float(np.dot(q_emb, occ_mean)) #cosine similarity between query and occasion (avg of all prompt emebeddigns for that occasion)
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= OCCASION_THRESHOLD: #if similarity is high enough…
        return OccasionDecision(mode="on", target=best_name, score=best_score) #urn occasion mode on

    return OccasionDecision(mode="off", target=None, score=best_score) #otherwise keep occasion mode off but return the best score for debugging


def parse_query(query: str, use_embedding: bool = True) -> dict:
    text = normalize_query(query)
    constraints = extract_constraints(text)
    occasion = decide_occasion_mode(text, use_embedding=use_embedding)

    return {
        "normalized": text,
        "constraints": constraints,
        "occasion": occasion,
    }


if __name__ == "__main__":
    test_queries = [
        "I want a pair of wine color jeans for a chill and casual day in the park, I like the cut to be straight",
        "I am looking for a night short dress red but that costs red that costs less than 50 euros",
    ]

    for q in test_queries:
        result = parse_query(q, use_embedding=False)
        print("\nQuery:", q)
        print("Normalized:", result["normalized"])
        print("Categories:", result["constraints"].categories)
        print("Colors:", result["constraints"].colors)
        print("Price min:", result["constraints"].price_min)
        print("Price max:", result["constraints"].price_max)
        print("Fit:", result["constraints"].fit)
        print("Occasion mode:", result["occasion"].mode)
        print("Occasion target:", result["occasion"].target)
        print("Occasion score:", result["occasion"].score)
