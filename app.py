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
from streamlit_extras.stylable_container import stylable_container

from online.candidate_retrieval import (
    EMBEDDINGS_PATH,
    IDS_PATH,
    PARQUET_PATH,
    retrieve_candidates,
)
from online.explanation_generation_langchain import generate_explanations
from online.final_ranking import rank_candidates
from online.occasion_suitability_scores import compute_occasion_scores
from online.query_processing_llm import parse_query_llm


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TIMEOUT_SECONDS = 60


@dataclass
class RecoResult:
    rows: list[dict[str, Any]]
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
def _load_assets() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_parquet(PARQUET_PATH)
    df["row_id"] = df["row_id"].astype(str)
    df = df.set_index("row_id")
    product_ids = np.load(IDS_PATH, allow_pickle=True).astype(str)
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32", copy=False)
    return df, product_ids, embeddings


def _run_pipeline(query: str, topk: int = 30) -> RecoResult:
    parsed = parse_query_llm(query)
    candidates = retrieve_candidates(
        query,
        parsed=parsed,
        topk=topk,
        filter_first=True,
        use_faiss=True,
    )
    if not candidates:
        return RecoResult(rows=[], parsed=parsed)

    df, product_ids, embeddings = _load_assets()
    occasion_scores = compute_occasion_scores(
        candidates,
        parsed=parsed,
        product_ids=product_ids,
        embeddings=embeddings,
    )
    ranked = rank_candidates(candidates, occasion_scores)

    rows: list[dict[str, Any]] = []
    for row_id, relevance, occ_score, final_score in ranked:
        if row_id not in df.index:
            continue
        row = df.loc[row_id]
        rows.append(
            {
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
    )
    for row in rows:
        row["explanation"] = explanations.get(str(row["row_id"]), "")

    return RecoResult(rows=rows, parsed=parsed)


def _occasion_missing(parsed: dict[str, Any]) -> bool:
    occasion = parsed.get("occasion") or {}
    return occasion.get("mode") != "on" or not occasion.get("target")


st.set_page_config(page_title="Zara's virtual shopping assistant", layout="wide")
st.title("Zara's virtual shopping assistant")
st.markdown(
    """
    <style>
    div[data-testid="stChatMessage"][data-message-author="user"],
    div[data-testid="stChatMessage"][data-testid*="user"],
    div[data-testid="stChatMessage"][data-testid="stChatMessageUser"],
    div[data-testid="stChatMessage"].st-chat-message-user,
    div[data-testid="stChatMessage"]:has(svg[aria-label="user"]) {
        justify-content: flex-end;
        flex-direction: row-reverse;
    }
    div[data-testid="stChatMessage"][data-message-author="user"] > div,
    div[data-testid="stChatMessage"][data-testid*="user"] > div,
    div[data-testid="stChatMessage"][data-testid="stChatMessageUser"] > div,
    div[data-testid="stChatMessage"].st-chat-message-user > div,
    div[data-testid="stChatMessage"]:has(svg[aria-label="user"]) > div {
        margin-left: auto;
    }
    div[data-testid="stChatMessage"][data-message-author="user"] .stMarkdown,
    div[data-testid="stChatMessage"][data-testid*="user"] .stMarkdown,
    div[data-testid="stChatMessage"][data-testid="stChatMessageUser"] .stMarkdown,
    div[data-testid="stChatMessage"].st-chat-message-user .stMarkdown,
    div[data-testid="stChatMessage"]:has(svg[aria-label="user"]) .stMarkdown,
    div[data-testid="stChatMessage"][data-message-author="user"] p,
    div[data-testid="stChatMessage"][data-testid*="user"] p,
    div[data-testid="stChatMessage"][data-testid="stChatMessageUser"] p,
    div[data-testid="stChatMessage"].st-chat-message-user p,
    div[data-testid="stChatMessage"]:has(svg[aria-label="user"]) p {
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

for msg in st.session_state.messages:
    if msg["role"] == "user":
        with stylable_container(
            key=f"user-msg-{hash(msg['content'])}",
            css_styles="""
            div[data-testid="stChatMessage"] {
                justify-content: flex-end;
                flex-direction: row-reverse;
            }
            div[data-testid="stChatMessageContent"] {
                text-align: right;
                margin-left: auto;
            }
            div[data-testid="stChatMessageContent"] p,
            div[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"],
            div[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] p {
                text-align: right;
            }
            """,
        ):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
    else:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

user_input = st.chat_input("Tell me what you want to buy...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    combined_query = user_input
    if st.session_state.awaiting_occasion and st.session_state.pending_query:
        combined_query = f"{st.session_state.pending_query}. Occasion: {user_input}"

    result = _run_pipeline(combined_query)

    if _occasion_missing(result.parsed):
        st.session_state.awaiting_occasion = True
        st.session_state.pending_query = combined_query
        assistant_text = _ask_for_occasion()
        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
        with st.chat_message("assistant"):
            st.write(assistant_text)
    else:
        st.session_state.awaiting_occasion = False
        st.session_state.pending_query = ""

        if not result.rows:
            assistant_text = "I couldn't find matches for that request. Want to adjust the style, color, or budget?"
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})
            with st.chat_message("assistant"):
                st.write(assistant_text)
        else:
            assistant_text = _build_response_message(result.rows, result.parsed)
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})
            with st.chat_message("assistant"):
                st.write(assistant_text)

            st.subheader("Top Picks")
            for row in result.rows[:5]:
                st.markdown(f"**{row.get('product_name', '')}**")
                image_url = row.get("image_url", "")
                if image_url:
                    st.image(image_url, width=260)
                final_score = row.get("final_score")
                if final_score is not None:
                    st.write(f"Final score: {final_score:.4f}")
                st.write(row.get("explanation", ""))
                if row.get("product_url"):
                    st.write(row["product_url"])
                st.write("---")

    st.session_state.sessions[st.session_state.current_session] = {
        "messages": list(st.session_state.messages),
        "awaiting_occasion": st.session_state.awaiting_occasion,
        "pending_query": st.session_state.pending_query,
    }
