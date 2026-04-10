# Fast Fashion Assistant using RecSys

A conversational fashion recommendation system that takes a natural-language
query (e.g. *"I need a short black dress for a night out under 50 euros"*) and
returns ranked product recommendations with generated explanations. The
system combines LLM-based query parsing, CLIP / FashionCLIP embeddings,
metadata filtering, occasion-aware reranking, and an LLM-as-a-Judge (LLMaaJ)
evaluation framework.

---

## Folder structure

```
Fast-Fashion-Assistant-using-RecSys/
├── offline/          # Data preparation: scraping, embeddings, occasion library
├── online/           # Runtime pipeline components (parse → retrieve → rank → explain)
├── backend/          # FastAPI server that wraps the online pipeline
├── frontend/         # React + Vite + Tailwind chat UI
├── evaluation/       # LLM-as-a-Judge framework, prompts, results, tables
├── main.py           # Standalone CLI entry point (pipeline without UI)
├── results_test/     # Output of main.py runs (timestamped folders)
└── myenv/            # Python virtual environment
```

### `offline/` — data preparation
Everything that is run **once** to build the product catalog and embedding
indexes that the runtime pipeline reads from.

- **`web_scrapping/`** — web scraping of the product catalog
  (`web_scrapping.py`, `data_exploration.ipynb`, `refresh_missing_images.py`).
- **`data/`** — scraped catalog and cached assets:
  - `women_data.parquet` — the full product catalog
  - `images/` — cached product images
  - `image_embeddings/`, `text_embeddings/`, `prompt_embeddings/` — precomputed vectors
  - `web_scrapping_runs/` — raw scraping logs
- **`metadata_embeddings/`** — scripts that compute and index embeddings:
  - `clip_image_embeddings.py`, `clip_text_embeddings.py` — compute CLIP and
    FashionCLIP embeddings for product images and text
  - `cache_images_from_parquet.py` — download and cache images locally
  - `build_faiss_index.py` / `test_faiss_index.py` — build FAISS indexes for
    fast approximate nearest neighbour retrieval
  - `visualize_embeddings.py` / `.ipynb` — 2D UMAP visualizations of the
    embedding space
- **`occasion_library/`** — curated "occasion" embeddings used by the
  occasion-aware reranker:
  - `occasion_prompts_clavix.json` — the occasion prompt library (wedding,
    job interview, festival, etc.)
  - `clip_prompts_embeddings.py` — computes text embeddings for every
    occasion prompt
  - `top_images_for_prompts.py` — finds visually coherent images per occasion
  - `visualize_text_image_*.py` — diagnostic plots

### `online/` — runtime pipeline
Each file is one stage of the end-to-end recommendation pipeline. They are
orchestrated by `main.py` and `backend/api.py`.

- **`query_processing_llm.py`** — parses the natural-language query with an
  LLM, extracting categories, colors, fit, price range, and occasion (mode +
  target label).
- **`candidate_retrieval.py`** — retrieves candidates with two strategies:
  metadata pre-filtering (category/color/price) followed by CLIP /
  FashionCLIP similarity search via FAISS. Supports relaxed fallback stages
  if strict filters return nothing.
- **`occasion_suitability_scores.py`** — scores each candidate against the
  detected occasion using the precomputed occasion library embeddings.
- **`final_ranking.py`** — blends relevance and occasion scores
  (`final = α · relevance + β · occasion`) and sorts.
- **`explanation_generation_llm.py`** — generates a short, grounded
  explanation for each of the top recommendations, using product metadata
  and the user query.

### `backend/` — FastAPI server
- **`api.py`** — wraps the `online/` pipeline and exposes it via a
  Server-Sent Events endpoint so the frontend can stream step-by-step
  progress and final results. Also includes audio (Whisper) transcription
  for voice queries.
- **`requirements.txt`** — backend Python dependencies (FastAPI, OpenAI,
  Groq, etc.). The backend imports modules from `online/` and also reads
  the catalog / embeddings built in `offline/`.

### `frontend/` — React chat UI
- **`src/App.tsx`** — main chat interface
- **`src/components/`** — UI building blocks:
  `ChatInput.tsx`, `ChatMessage.tsx`, `ConstraintsViewer.tsx`,
  `ModelResults.tsx`, `ProductCard.tsx`, `ProgressIndicator.tsx`, `Sidebar.tsx`
- **`src/api/client.ts`** — talks to the FastAPI backend
- **`vite.config.ts`, `tailwind.config.js`, `tsconfig.json`** — build config

### `evaluation/` — LLM-as-a-Judge framework
Holds the **rubrics, prompts, runners, and results** of the evaluation
framework used to benchmark the pipeline.

- **Judges** (one file per evaluation scope):
  - `parser_judge.py` — grades the query parser on 3 rubrics
    (completeness, no-hallucination, occasion detection)
  - `item_judge.py` — grades each individual product on 3 rubrics
    (item relevance, occasion appropriateness, explanation quality)
  - `set_judge.py` — grades the full 5-item set on set-level answer relevance
  - `cross_model_judge.py` — pairwise CLIP vs FashionCLIP preference
  - `evaluators.py` — high-level wrappers + parallel `evaluate_query_result`
- **`prompts/`** — YAML rubric definitions for each judge. `_v2.yaml`
  variants use loosened scales to reduce LLM score compression:
  - `parser_judge.yaml` / `parser_judge_v2.yaml`
  - `item_judge.yaml` / `item_judge_v2.yaml`
  - `set_judge.yaml` / `set_judge_v2.yaml`
  - `cross_model_judge.yaml` / `cross_model_judge_v2.yaml`
- **`run_eval.py`** — batch evaluation runner; executes the full pipeline
  over `test_queries.json` and stores per-query + aggregate judge scores
- **`run_ablation.py`** — runs ablation conditions (no-occasion,
  no-pipeline baseline) over the tier-3 (complex) subset
- **`results_table.py`** — builds the main per-tier × per-model comparison
  table (CLIP / FashionCLIP / Random baseline) across all rubrics
- **`ablation_table.py`** — builds the Full Pipeline vs No Pipeline
  comparison tables for tier-3 queries
- **`dashboard.py`** — Streamlit dashboard for exploring eval results
- **`occasion_analysis.ipynb`** — notebook for analyzing occasion scores
- **`test_queries.json`** — 150 queries across 3 tiers
  (basic, medium, complex)
- **`results/`** — JSON + CSV outputs from all eval runs, including
  `ablation_no_pipeline/` and `ablation_no_occasion/` subfolders
- **`_client.py`** — shared OpenAI client with concurrency semaphores for
  multimodal calls

### Top level
- **`main.py`** — runs the full pipeline on a single query without any UI.
  Used for local testing and debugging. Saves intermediate results
  (top-30 relevance-only, top-30 reranked with occasion, top-5 with
  explanations) to `results_test/<timestamp>/`.
- **`results_test/`** — timestamped output folders from `main.py` runs.
- **`pipeline_choices.txt`** — notes on design decisions.

---

## Setup

### Prerequisites
- Python 3.12
- Node.js 18+ (for the frontend)
- An OpenAI API key (and optionally Groq for Whisper audio)

### 1. Python environment
```bash
python3.12 -m venv myenv
source myenv/bin/activate
pip install -r backend/requirements.txt
pip install -r evaluation/requirements.txt
```

### 2. Environment variables
Create a `.env` file at the project root:
```bash
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk-...         # optional, for voice transcription
```

### 3. Offline data
The pipeline expects the catalog and embeddings under `offline/data/`. If
you are rebuilding from scratch:
```bash
# 1. Scrape catalog
python offline/web_scrapping/web_scrapping.py

# 2. Cache images locally
python offline/metadata_embeddings/cache_images_from_parquet.py

# 3. Compute embeddings (CLIP and FashionCLIP)
python offline/metadata_embeddings/clip_image_embeddings.py
python offline/metadata_embeddings/clip_text_embeddings.py

# 4. Build FAISS index
python offline/metadata_embeddings/build_faiss_index.py

# 5. Build the occasion library
python offline/occasion_library/clip_prompts_embeddings.py
```

### 4. Frontend dependencies
```bash
cd frontend
npm install
```

---

## Running the application

### Backend (FastAPI)
From the project root, with `myenv` activated:
```bash
uvicorn backend.api:app --reload --port 8000
```
This exposes the pipeline at `http://localhost:8000`.

### Frontend (React + Vite)
In a second terminal:
```bash
cd frontend
npm run dev
```
Open `http://localhost:5173` in your browser to interact with the chat UI.

### Standalone pipeline (no UI)
For quick testing without running the servers:
```bash
# With a custom query
python main.py --query "I need a short black dress for a night out under 50 euros"

# With the built-in test query
python main.py --test-query
```
Results are saved to `results_test/<timestamp>/`:
- `pipeline_results.json` — parsed query + top-30 by relevance + top-30
  reranked with occasion + top-5 with explanations (per model)
- `top5_clip.csv`, `top5_fashion_clip.csv` — quick-view CSVs

---

## Running the evaluation

All evaluation commands run from the project root with `myenv` activated.

### Batch evaluation of the full pipeline
```bash
# Full test set (150 queries)
python -m evaluation.run_eval

# A slice (useful for day-by-day runs)
python -m evaluation.run_eval --start 0 --end 30
python -m evaluation.run_eval --start 30 --end 60
# ...
```
Outputs `eval_<timestamp>.json` and `aggregate_<timestamp>.csv` under
`evaluation/results/`.

### Ablation runs (tier-3 only)
```bash
# Relevance-only (α=1.0, β=0.0)
python -m evaluation.run_ablation --mode no-occasion

# No pipeline (raw embedding similarity, no filtering, no occasion)
python -m evaluation.run_ablation --mode no-pipeline
```
Outputs go to `evaluation/results/ablation_no_occasion/` and
`evaluation/results/ablation_no_pipeline/`.

### Result tables
```bash
# Main per-tier × per-model table
python -m evaluation.results_table --all

# Full pipeline vs no-pipeline ablation table
python -m evaluation.ablation_table
```
