"""FastAPI backend — wraps the existing recommendation pipeline and exposes
a single Server-Sent Events endpoint so the React frontend can stream
step-by-step progress and final results.

Start with:
    uvicorn backend.api:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, AsyncGenerator

# Must be set before any HuggingFace / transformers import.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Make the project root importable regardless of working directory.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import io

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from groq import Groq
from openai import OpenAI
from pydantic import BaseModel
from transformers import pipeline

from online.candidate_retrieval import MODEL_PATHS, PARQUET_PATH, retrieve_candidates
from online.explanation_generation_llm import generate_explanations
from online.final_ranking import rank_candidates
from online.occasion_suitability_scores import (
    MODEL_OCCASION_EMBEDDINGS_PATHS,
    compute_occasion_scores,
)
from online.query_processing_llm import parse_query_llm

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Fashion Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
_openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
SESSION_TITLE_MODEL = os.getenv("SESSION_TITLE_MODEL", "lidiya/bart-large-xsum-samsum")
_session_title_summarizer = None

# Cache the catalog in memory so it isn't re-read on every request.
_catalog_cache: pd.DataFrame | None = None


def _get_catalog() -> pd.DataFrame:
    global _catalog_cache
    if _catalog_cache is None:
        df = pd.read_parquet(PARQUET_PATH)
        df["row_id"] = df["row_id"].astype(str)
        _catalog_cache = df.set_index("row_id")
    return _catalog_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json(obj: Any) -> Any:
    """Recursively convert non-JSON-serialisable objects to plain dicts."""
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, (int, float, bool, str, type(None))):
        return obj
    # Pydantic / dataclass fallback
    if hasattr(obj, "__dict__"):
        return _safe_json(vars(obj))
    if hasattr(obj, "model_dump"):
        return _safe_json(obj.model_dump())
    return str(obj)


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(_safe_json(data))}\n\n"


def _has_occasion(parsed: dict) -> bool:
    occasion = parsed.get("occasion") or {}
    return occasion.get("mode") == "on" and bool(occasion.get("target"))


def _build_summary(rows: list[dict], parsed: dict) -> str:
    """Ask Groq for a short friendly summary of the top picks."""
    if not GROQ_API_KEY:
        return "Here are your top picks:"
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
    try:
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user)},
            ],
            temperature=0.2,
            timeout=60,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return "Here are your top picks:"


def _get_session_title_summarizer():
    global _session_title_summarizer
    if _session_title_summarizer is None:
        _session_title_summarizer = pipeline("summarization", model=SESSION_TITLE_MODEL)
    return _session_title_summarizer


def _normalize_title(text: str) -> str:
    title = " ".join(text.strip().split()).rstrip(".")
    return title[:80] or "New chat"


def _summarize_session_title(conversation: str) -> str:
    conversation = conversation.strip()
    if not conversation:
        return "New chat"

    summarizer = _get_session_title_summarizer()
    result = summarizer(conversation, truncation=True)
    summary_text = result[0].get("summary_text", "") if result else ""
    return _normalize_title(summary_text)


def _run_pipeline_sync(query: str) -> dict:
    """Synchronous pipeline execution (called from thread pool)."""
    parsed = parse_query_llm(query)
    parsed_safe = _safe_json(parsed)
    has_occ = _has_occasion(parsed_safe)

    candidates_by_model = retrieve_candidates(
        query,
        parsed=parsed,
        topk=30,
        filter_first=True,
        use_faiss=True,
        embedding_model="both",
        return_metadata=True,
    )

    if not any(result.get("candidates") for result in candidates_by_model.values()):
        return {"status": "no_results", "parsed": parsed_safe, "rows_by_model": {}}

    df = _get_catalog()
    rows_by_model: dict[str, list[dict]] = {}

    for model_name, result in candidates_by_model.items():
        candidates = result.get("candidates", [])
        if not candidates:
            continue
        match_stage = result.get("match_stage", "strict")
        match_message = result.get("match_message", "")

        paths = MODEL_PATHS[model_name]
        product_ids = np.load(paths["image_ids"], allow_pickle=True).astype(str)
        embeddings = np.load(paths["image_embeddings"]).astype("float32", copy=False)

        if has_occ:
            occasion_scores = compute_occasion_scores(
                candidates,
                parsed=parsed,
                product_ids=product_ids,
                embeddings=embeddings,
                model_name=model_name,
                occasion_embeddings_path=MODEL_OCCASION_EMBEDDINGS_PATHS.get(model_name),
            )
            ranked = rank_candidates(candidates, occasion_scores)
        else:
            occasion_scores = {}
            ranked = rank_candidates(candidates, occasion_scores, alpha=1.0, beta=0.0)

        rows: list[dict] = []
        for row_id, relevance, occ_score, final_score in ranked:
            if row_id not in df.index:
                continue
            row = df.loc[row_id]
            rows.append(
                {
                    "model_name": model_name,
                    "row_id": row_id,
                    "relevance_score": float(relevance),
                    "occasion_score": float(occ_score),
                    "final_score": float(final_score),
                    "product_name": str(row.get("product_name", "") or ""),
                    "product_description": str(row.get("product_description", "") or ""),
                    "product_url": str(row.get("product_url", "") or ""),
                    "price": str(row.get("price", "") or ""),
                    "color": str(row.get("color", "") or ""),
                    "product_category": str(row.get("product_category", "") or ""),
                    "image_url": str(row.get("image_url", "") or ""),
                    "match_stage": match_stage,
                    "match_message": match_message,
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

        rows_by_model[model_name] = rows[:5]

    all_rows = [r for rows in rows_by_model.values() for r in rows]
    summary = _build_summary(all_rows, parsed_safe) if all_rows else ""

    return {
        "status": "ok",
        "parsed": parsed_safe,
        "rows_by_model": rows_by_model,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Streaming pipeline with progress events
# ---------------------------------------------------------------------------

async def _stream_pipeline(query: str) -> AsyncGenerator[str, None]:
    loop = asyncio.get_event_loop()

    async def run(fn, *args, **kwargs):
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    try:
        # Step 1: Parse
        yield _sse("progress", {"step": 1, "total": 6, "message": "Parsing your query…"})
        parsed = await run(parse_query_llm, query)
        parsed_safe = _safe_json(parsed)

        has_occ = _has_occasion(parsed_safe)

        # Step 2: Retrieve candidates
        yield _sse("progress", {"step": 2, "total": 6, "message": "Retrieving candidates via CLIP & FashionCLIP…"})
        candidates_by_model = await run(
            retrieve_candidates,
            query,
            parsed,
            30,       # topk
            True,     # filter_first
            True,     # use_faiss
            False,    # use_text_embeddings
            0.7,      # image_weight
            0.3,      # text_weight
            "both",   # embedding_model
            True,     # return_metadata
        )

        if not any(result.get("candidates") for result in candidates_by_model.values()):
            yield _sse("results", {"parsed": parsed_safe, "rows_by_model": {}, "summary": ""})
            yield _sse("done", {})
            return

        # Step 3: Load catalog
        yield _sse("progress", {"step": 3, "total": 6, "message": "Loading product catalog…"})
        df = await run(_get_catalog)

        rows_by_model: dict[str, list[dict]] = {}
        available = [(m, r) for m, r in candidates_by_model.items() if r.get("candidates")]

        for idx, (model_name, result) in enumerate(available, 1):
            candidates = result.get("candidates", [])
            match_stage = result.get("match_stage", "strict")
            match_message = result.get("match_message", "")
            # Step 4: Score & rank
            yield _sse("progress", {
                "step": 4,
                "total": 6,
                "message": f"Scoring & ranking — {model_name.replace('_', ' ')} ({idx}/{len(available)})…",
            })

            paths = MODEL_PATHS[model_name]
            product_ids = np.load(paths["image_ids"], allow_pickle=True).astype(str)
            embeddings = np.load(paths["image_embeddings"]).astype("float32", copy=False)

            if has_occ:
                occasion_scores = await run(
                    compute_occasion_scores,
                    candidates, parsed, product_ids, embeddings, model_name,
                    MODEL_OCCASION_EMBEDDINGS_PATHS.get(model_name),
                )
                ranked = await run(rank_candidates, candidates, occasion_scores)
            else:
                occasion_scores = {}
                ranked = await run(rank_candidates, candidates, occasion_scores, alpha=1.0, beta=0.0)

            rows: list[dict] = []
            for row_id, relevance, occ_score, final_score in ranked:
                if row_id not in df.index:
                    continue
                row = df.loc[row_id]
                rows.append({
                    "model_name": model_name,
                    "row_id": row_id,
                    "relevance_score": float(relevance),
                    "occasion_score": float(occ_score),
                    "final_score": float(final_score),
                    "product_name": str(row.get("product_name", "") or ""),
                    "product_description": str(row.get("product_description", "") or ""),
                    "product_url": str(row.get("product_url", "") or ""),
                    "price": str(row.get("price", "") or ""),
                    "color": str(row.get("color", "") or ""),
                    "product_category": str(row.get("product_category", "") or ""),
                    "image_url": str(row.get("image_url", "") or ""),
                    "match_stage": match_stage,
                    "match_message": match_message,
                })

            # Step 5: Explain
            yield _sse("progress", {
                "step": 5,
                "total": 6,
                "message": f"Generating explanations — {model_name.replace('_', ' ')} ({idx}/{len(available)})…",
            })
            explanations = await run(
                generate_explanations,
                rows, candidates, parsed, product_ids, embeddings,
                occasion_scores, model_name,
                MODEL_OCCASION_EMBEDDINGS_PATHS.get(model_name),
            )
            for row in rows:
                row["explanation"] = explanations.get(str(row["row_id"]), "")

            rows_by_model[model_name] = rows[:5]

        # Step 6: Summarise
        yield _sse("progress", {"step": 6, "total": 6, "message": "Generating summary…"})
        all_rows = [r for rows in rows_by_model.values() for r in rows]
        summary = await run(_build_summary, all_rows, parsed_safe)

        yield _sse("results", {
            "parsed": parsed_safe,
            "rows_by_model": rows_by_model,
            "summary": summary,
        })
        yield _sse("done", {})

    except Exception as exc:
        yield _sse("error", {"message": str(exc)})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str


class SessionTitleRequest(BaseModel):
    conversation: str


@app.post("/api/search")
async def search(req: SearchRequest):
    return StreamingResponse(
        _stream_pipeline(req.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/session-title")
async def session_title(req: SessionTitleRequest):
    try:
        return {"title": _summarize_session_title(req.conversation)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Session title generation failed: {exc}")


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if _openai_client is None:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")
    content = await audio.read()
    filename = audio.filename or "recording.webm"
    result = _openai_client.audio.transcriptions.create(
        file=(filename, io.BytesIO(content), audio.content_type or "audio/webm"),
        model=OPENAI_TRANSCRIBE_MODEL,
    )
    return {"text": result.text}


@app.get("/health")
async def health():
    return {"status": "ok"}
