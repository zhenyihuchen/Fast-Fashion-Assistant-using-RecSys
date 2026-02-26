from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from streamlit_chat import message

from online.candidate_retrieval import (
    MODEL_PATHS,
    PARQUET_PATH,
    retrieve_candidates,
)
from online.explanation_generation_groq import generate_explanations
from online.final_ranking import rank_candidates
from online.occasion_suitability_scores import (
    MODEL_OCCASION_EMBEDDINGS_PATHS,
    compute_occasion_scores,
)
from online.query_processing_llm import parse_query_llm


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TIMEOUT_SECONDS = 60


@dataclass
class RecoResult:
    rows_by_model: dict[str, list[dict[str, Any]]]
    parsed: dict[str, Any]


def _call_groq(messages: list[dict[str, str]]) -> str:
    if not GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY is not set")
    client = Groq(api_key=GROQ_API_KEY)
    chat = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        timeout=TIMEOUT_SECONDS,
    )
    return chat.choices[0].message.content.strip()


def _assistant_intro() -> str:
    return (
        "Hey, I am Zara's virtual shopping assistant. I can help you look for your next "
        "purchase faster than traditional searching. What do you want to buy today and "
        "for what occasion?"
    )


def _ask_for_occasion() -> str:
    return "Thanks! What occasion is this for (e.g., date night, party night out, work/office, wedding guest)?"


def _build_response_message(rows: list[dict[str, Any]], parsed: dict[str, Any]) -> str:
    evidence = {
        "occasion": (parsed.get("occasion") or {}).get("target"),
        "items": [
            {
                "name": r.get("product_name", ""),
                "price": r.get("price", ""),
                "color": r.get("color", ""),
                "category": r.get("product_category", ""),
                "explanation": r.get("explanation", ""),
            }
            for r in rows[:5]
        ],
    }
    system = (
        "You are a shopping assistant. Write a short, friendly response (2-3 sentences max). "
        "Use only the provided evidence. Do not add new attributes or change item names. "
        "Do not invent availability or sizes."
    )
    user = {
        "evidence": evidence,
        "instructions": [
            "Mention the occasion if provided.",
            "Briefly introduce the top recommendations.",
            "Do not alter the item list or details.",
        ],
    }
    return _call_groq(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ]
    )


@st.cache_resource
def _load_catalog() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_PATH)
    df["row_id"] = df["row_id"].astype(str)
    df = df.set_index("row_id")
    return df


def _run_pipeline(query: str, topk: int = 30) -> RecoResult:
    parsed = parse_query_llm(query)
    candidates_by_model = retrieve_candidates(
        query,
        parsed=parsed,
        topk=topk,
        filter_first=True,
        use_faiss=True,
        embedding_model="both",
    )
    if not any(candidates_by_model.values()):
        return RecoResult(rows_by_model={}, parsed=parsed)

    df = _load_catalog()
    rows_by_model: dict[str, list[dict[str, Any]]] = {}

    for model_name, candidates in candidates_by_model.items():
        if not candidates:
            continue

        paths = MODEL_PATHS[model_name]
        product_ids = np.load(paths["image_ids"], allow_pickle=True).astype(str)
        embeddings = np.load(paths["image_embeddings"]).astype("float32", copy=False)

        occasion_scores = compute_occasion_scores(
            candidates,
            parsed=parsed,
            product_ids=product_ids,
            embeddings=embeddings,
            model_name=model_name,
            occasion_embeddings_path=MODEL_OCCASION_EMBEDDINGS_PATHS.get(model_name),
        )
        ranked = rank_candidates(candidates, occasion_scores)

        rows: list[dict[str, Any]] = []
        for row_id, relevance, occ_score, final_score in ranked:
            if row_id not in df.index:
                continue
            row = df.loc[row_id]
            rows.append(
                {
                    "model_name": model_name,
                    "row_id": row_id,
                    "relevance_score": relevance,
                    "occasion_score": occ_score,
                    "final_score": final_score,
                    "product_name": row.get("product_name", ""),
                    "product_description": row.get("product_description", ""),
                    "product_url": row.get("product_url", ""),
                    "price": row.get("price", ""),
                    "color": row.get("color", ""),
                    "product_category": row.get("product_category", ""),
                    "image_url": row.get("image_url", ""),
                }
            )

        explanations = generate_explanations(
            rows,
            candidates=candidates,
            parsed=parsed,
            product_ids=product_ids,
            embeddings=embeddings,
            occasion_scores=occasion_scores,
            model_name=model_name,
            occasion_embeddings_path=MODEL_OCCASION_EMBEDDINGS_PATHS.get(model_name),
        )
        for row in rows:
            row["explanation"] = explanations.get(str(row["row_id"]), "")

        rows_by_model[model_name] = rows

    return RecoResult(rows_by_model=rows_by_model, parsed=parsed)


def _run_pipeline_with_progress(
    query: str, topk: int = 30
) -> tuple[RecoResult, list[str]]:
    progress = st.progress(0)
    status = st.empty()
    steps_log: list[str] = []

    def update(step: int, total: int, text: str) -> None:
        pct = int((step / total) * 100)
        progress.progress(pct)
        status.write(text)
        steps_log.append(text)

    total_steps = 6
    update(1, total_steps, "1/6 Parsing query with LLM...")
    parsed = parse_query_llm(query)

    update(2, total_steps, "2/6 Retrieving candidates (CLIP + FashionCLIP)...")
    candidates_by_model = retrieve_candidates(
        query,
        parsed=parsed,
        topk=topk,
        filter_first=True,
        use_faiss=True,
        embedding_model="both",
    )

    if not any(candidates_by_model.values()):
        update(6, total_steps, "6/6 Done (no candidates found).")
        progress.empty()
        status.empty()
        return RecoResult(rows_by_model={}, parsed=parsed), steps_log

    update(3, total_steps, "3/6 Loading product catalog...")
    df = _load_catalog()

    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    available_models = [m for m, c in candidates_by_model.items() if c]
    for idx, model_name in enumerate(available_models, start=1):
        update(4, total_steps, f"4/6 Scoring and ranking {model_name} ({idx}/{len(available_models)})...")
        candidates = candidates_by_model[model_name]
        paths = MODEL_PATHS[model_name]
        product_ids = np.load(paths["image_ids"], allow_pickle=True).astype(str)
        embeddings = np.load(paths["image_embeddings"]).astype("float32", copy=False)

        occasion_scores = compute_occasion_scores(
            candidates,
            parsed=parsed,
            product_ids=product_ids,
            embeddings=embeddings,
            model_name=model_name,
            occasion_embeddings_path=MODEL_OCCASION_EMBEDDINGS_PATHS.get(model_name),
        )
        ranked = rank_candidates(candidates, occasion_scores)

        rows: list[dict[str, Any]] = []
        for row_id, relevance, occ_score, final_score in ranked:
            if row_id not in df.index:
                continue
            row = df.loc[row_id]
            rows.append(
                {
                    "model_name": model_name,
                    "row_id": row_id,
                    "relevance_score": relevance,
                    "occasion_score": occ_score,
                    "final_score": final_score,
                    "product_name": row.get("product_name", ""),
                    "product_description": row.get("product_description", ""),
                    "product_url": row.get("product_url", ""),
                    "price": row.get("price", ""),
                    "color": row.get("color", ""),
                    "product_category": row.get("product_category", ""),
                    "image_url": row.get("image_url", ""),
                }
            )

        update(5, total_steps, f"5/6 Generating explanations for {model_name}...")
        explanations = generate_explanations(
            rows,
            candidates=candidates,
            parsed=parsed,
            product_ids=product_ids,
            embeddings=embeddings,
            occasion_scores=occasion_scores,
            model_name=model_name,
            occasion_embeddings_path=MODEL_OCCASION_EMBEDDINGS_PATHS.get(model_name),
        )
        for row in rows:
            row["explanation"] = explanations.get(str(row["row_id"]), "")
        rows_by_model[model_name] = rows

    update(6, total_steps, "6/6 Done. Rendering recommendations...")
    progress.empty()
    status.empty()
    return RecoResult(rows_by_model=rows_by_model, parsed=parsed), steps_log


def _occasion_missing(parsed: dict[str, Any]) -> bool:
    occasion = parsed.get("occasion") or {}
    return occasion.get("mode") != "on" or not occasion.get("target")


def _render_recs_by_model(recs_by_model: dict[str, list[dict[str, Any]]]) -> None:
    for model_name, recs in recs_by_model.items():
        st.markdown(f"### {model_name.upper()} recommendations")
        for row in recs:
            st.markdown(f"**{row.get('product_name', '')}**")
            image_url = row.get("image_url", "")
            if image_url:
                st.image(image_url, width=260, use_container_width=False)
            final_score = row.get("final_score")
            if final_score is not None:
                st.write(f"Final score: {final_score:.4f}")
            st.write(row.get("explanation", ""))
            if row.get("product_url"):
                st.write(row["product_url"])
            st.write("---")


st.set_page_config(page_title="Zara's virtual shopping assistant", layout="wide")
st.title("Zara's virtual shopping assistant")
st.caption("This chat supports several interactions within the same session.")

if "sessions" not in st.session_state:
    st.session_state.sessions = []
if "current_session" not in st.session_state:
    st.session_state.current_session = 0
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": _assistant_intro()}]
if "awaiting_occasion" not in st.session_state:
    st.session_state.awaiting_occasion = False
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""
if not st.session_state.sessions:
    st.session_state.sessions.append(
        {
            "messages": list(st.session_state.messages),
            "awaiting_occasion": st.session_state.awaiting_occasion,
            "pending_query": st.session_state.pending_query,
        }
    )

with st.sidebar:
    st.header("Sessions")
    session_labels = [f"Session {i + 1}" for i in range(len(st.session_state.sessions))]
    selected = st.selectbox("Previous sessions", session_labels, index=st.session_state.current_session)
    selected_index = session_labels.index(selected)
    if selected_index != st.session_state.current_session:
        st.session_state.current_session = selected_index
        session = st.session_state.sessions[selected_index]
        st.session_state.messages = list(session["messages"])
        st.session_state.awaiting_occasion = session["awaiting_occasion"]
        st.session_state.pending_query = session["pending_query"]
        st.rerun()

    if st.button("Restart session"):
        st.session_state.current_session = len(st.session_state.sessions)
        st.session_state.messages = [{"role": "assistant", "content": _assistant_intro()}]
        st.session_state.awaiting_occasion = False
        st.session_state.pending_query = ""
        st.session_state.sessions.append(
            {
                "messages": list(st.session_state.messages),
                "awaiting_occasion": st.session_state.awaiting_occasion,
                "pending_query": st.session_state.pending_query,
            }
        )
        st.rerun()

    if st.button("Clear conversation"):
        st.session_state.messages = [{"role": "assistant", "content": _assistant_intro()}]
        st.session_state.awaiting_occasion = False
        st.session_state.pending_query = ""
        st.session_state.sessions[st.session_state.current_session] = {
            "messages": list(st.session_state.messages),
            "awaiting_occasion": st.session_state.awaiting_occasion,
            "pending_query": st.session_state.pending_query,
        }
        st.rerun()

for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=f"user-{idx}")
    else:
        message(msg["content"], is_user=False, key=f"assistant-{idx}")
        recs_by_model = msg.get("recs_by_model")
        if recs_by_model:
            _render_recs_by_model(recs_by_model)

user_input = st.chat_input("Tell me what you want to buy...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    message(user_input, is_user=True, key=f"user-input-{len(st.session_state.messages)}")

    combined_query = user_input
    if st.session_state.awaiting_occasion and st.session_state.pending_query:
        combined_query = f"{st.session_state.pending_query}. Occasion: {user_input}"

    result, _progress_log = _run_pipeline_with_progress(combined_query)

    if _occasion_missing(result.parsed):
        st.session_state.awaiting_occasion = True
        st.session_state.pending_query = combined_query
        assistant_text = _ask_for_occasion()
        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
        message(assistant_text, is_user=False, key=f"assistant-occasion-{len(st.session_state.messages)}")
    else:
        st.session_state.awaiting_occasion = False
        st.session_state.pending_query = ""

        all_rows = [r for rows in result.rows_by_model.values() for r in rows]
        if not all_rows:
            assistant_text = "I couldn't find matches for that request. Want to adjust the style, color, or budget?"
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})
            message(assistant_text, is_user=False, key=f"assistant-nomatch-{len(st.session_state.messages)}")
        else:
            assistant_text = _build_response_message(all_rows, result.parsed)
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})
            message(assistant_text, is_user=False, key=f"assistant-recs-{len(st.session_state.messages)}")

            recs_by_model = {
                model_name: rows[:5]
                for model_name, rows in result.rows_by_model.items()
                if rows
            }
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Here are your top picks by model:",
                    "recs_by_model": recs_by_model,
                }
            )
            _render_recs_by_model(recs_by_model)

    st.session_state.sessions[st.session_state.current_session] = {
        "messages": list(st.session_state.messages),
        "awaiting_occasion": st.session_state.awaiting_occasion,
        "pending_query": st.session_state.pending_query,
    }
