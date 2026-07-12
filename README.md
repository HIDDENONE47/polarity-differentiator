# Family Office Micro-RAG

**PolarityIQ — Stage 1 Differentiator Assessment**

A validated, source-grounded intelligence pipeline for family office data: a LangGraph agent pipeline that discovers and extracts family office records, an adversarial auditor that independently re-verifies every high-value fact before it's allowed to be called "verified," and a FastAPI + FAISS Micro-RAG that answers questions over the dataset without ever stating something the retrieved data doesn't actually support.

**Live demo:** [Frontend (Vercel)](#) · [Backend API (Render)](#)

---

## Overview

Most scraped datasets treat "verified" as a label an LLM applies to itself. This project treats it as a claim that has to survive an adversarial check.

The system is built in three stages:

1. **Discovery & Extraction** — a LangGraph state machine searches the web (Tavily) for candidate family office entities and uses a Groq-hosted LLM to structure raw page text into a schema. Every field extracted here is stamped `UNVERIFIED` — extraction is never allowed to also grade its own homework.
2. **Adversarial Audit** — a separate agent re-fetches each claimed source *live* and asks a skeptical LLM pass whether the page actually, explicitly supports the claim. Only this layer is authorized to promote a field to `VERIFIED` or the honest fallback `COULD_NOT_VERIFY`. Work emails get an additional independent check against Hunter.io, and three "spotlight" records go through a second round of corroboration against an entirely separate source.
3. **Retrieval & Grounded Synthesis** — the cleaned, verified dataset is chunked one fact at a time into a FAISS index. At query time, the RAG backend retrieves relevant chunks and forces the LLM to answer *only* from what it retrieved — if nothing clears the similarity floor, it says so instead of guessing.

The result is a 50-record dataset where every "verified" cell has a live source URL, a stated verification method, and (for the 3 spotlight records) a second independent corroborating source — plus a queryable RAG layer that cites exactly which record and field it's answering from.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  DATA PIPELINE (data_pipeline/)                                 │
│                                                                   │
│  discover ──▶ extract ──▶ audit ──▶ filter ──▶ export           │
│  (Tavily)    (Groq LLM,   (live re-  (dedupe,   (CSV + JSONL    │
│               UNVERIFIED)  fetch +    action-     + rejection    │
│                             adversarial ability     log)         │
│                             LLM pass,  bar)                      │
│                             Hunter.io                            │
│                             for email)                           │
│                                                                   │
│  Orchestrated as a LangGraph StateGraph with conditional         │
│  routing, incremental crash-safe writes, and resumable state.    │
└────────────────────────────┬──────────────────────────────────--┘
                              │  family_offices_dataset.jsonl
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  BACKEND (backend/) — FastAPI                                    │
│                                                                    │
│  build_index.py        rag_retriever.py         main.py          │
│  chunks each verified   structured filter +      exposes          │
│  field into FAISS,      semantic FAISS search,   /query /health   │
│  keeps full provenance  grounding floor,         /stats over      │
│  in chunk_metadata.pkl  cited synthesis          HTTP (CORS-open) │
└────────────────────────────┬──────────────────────────────────--┘
                              │  REST / JSON
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  FRONTEND (frontend/) — React 19 + Vite                          │
│  Query box + entity filter → answer + a "source ledger":         │
│  every cited chunk shown with entity, field, similarity score,   │
│  and a live link back to its source.                             │
└─────────────────────────────────────────────────────────────────┘
```

The backend and frontend are fully decoupled: the frontend talks to the API purely over HTTP (`VITE_API_URL`), the API has no knowledge of any UI, and either can be deployed, scaled, or replaced independently. `main.py` is intentionally routing-only — all retrieval and grounding logic lives in `rag_retriever.py`, so the API layer stays a thin, swappable shell around the actual RAG logic.

---

## Core Features

### Epistemic Rigor & Cross-Validation
`agent_auditor.py` is the *only* code in the pipeline authorized to mark a field `VERIFIED`. It does this by re-fetching the claimed source live (never trusting the extraction agent's say-so) and running a deliberately skeptical adversarial LLM pass instructed to find reasons a claim is wrong, not reasons it might be right. For the three spotlight records (`run_spotlight.py` → `audit_record_deep`), high-value entity facts go through a second pass: an independent second source is searched for and checked for corroboration, with disagreement or absence stated honestly in `verification_notes` rather than silently resolved. Work emails get their own independent cross-check against Hunter.io's Email Finder, and a mismatch between the extraction agent's claim and Hunter's finding is flagged for manual review rather than auto-resolved in either direction. A field with no source URL or extraction method is structurally disallowed from claiming `VERIFIED` status — enforced at the schema level with a Pydantic validator, not just by convention.

### Decoupled Architecture
The system is split into three independently runnable layers — data pipeline, backend API, frontend — connected only by files (`.jsonl`/CSV) and HTTP/JSON, never by shared in-process state. The FastAPI backend exposes `/query`, `/health`, and `/stats`; the React/Vite frontend is a pure API consumer configured via a single `VITE_API_URL` environment variable. This means the RAG backend can be queried, tested, or redeployed with `curl` alone, with zero frontend involvement, and the frontend could be pointed at any backend implementing the same three endpoints.

### Hallucination Prevention
`rag_retriever.py` enforces a hard grounding discipline: the LLM is *never invoked at all* if no retrieved chunk clears the similarity floor (`MIN_SIMILARITY = 0.25`) — there's nothing legitimate for it to answer from, so the API returns an explicit "no verified information matches this query" response rather than letting the model fall back on its own training data. When it is invoked, the prompt requires every stated fact to cite the excerpt number it came from, and explicitly forbids inferring, estimating, or filling gaps with plausible-sounding information not present in the retrieved excerpts. Every answer returned by `/query` carries a `grounded` boolean and the full list of source chunks (entity, field, similarity score, live source URL) — the frontend renders this as a "source ledger" so provenance is visible, not just asserted.

### Rate Limit Resilience
Every LLM call in the extraction and audit agents goes through `invoke_llm_with_retry()` (`config.py`), a shared exponential-backoff wrapper around Groq's free-tier rate limit (429 `RateLimitError`). A rate limit hit is treated as an expected, routine condition under normal batch load — not a bug — so the pipeline waits and retries with doubling backoff (default: 6 attempts, starting at 10s) rather than crashing mid-run. The LangGraph orchestrator compounds this with its own resilience: `graph_orchestrator.py` writes each accepted record to a backup JSONL *incrementally*, so a run that dies partway through — whether from a rate limit exhausting all retries or anything else — resumes from where it left off instead of re-processing already-accepted entities.

### Honest-Blank Data Model
Every high-value field is wrapped in a `VerifiableField` (`config.py`) with three possible states: `VERIFIED`, `COULD_NOT_VERIFY`, or internal-only `UNVERIFIED` (which is structurally forbidden from reaching any exported file). `COULD_NOT_VERIFY` is treated as a legitimate, honest outcome — a blank the pipeline is candid about — rather than something the extraction agent should keep guessing at. Records also have to clear an actionability bar (`passes_actionability_bar`): at least one principal with a verified email, phone, or LinkedIn, or the record doesn't make the final dataset.

---

## Repository Structure

```
polarity-differentiator/
├── backend/                     # FastAPI Micro-RAG service
│   ├── main.py                  #   routing: /query /health /stats
│   ├── rag_retriever.py         #   retrieval + grounded synthesis
│   ├── requirements.txt
│   └── vector_store/
│       ├── build_index.py       #   builds the FAISS index from the dataset
│       ├── faiss_index.bin
│       └── chunk_metadata.pkl
│
├── data_pipeline/                # LangGraph discovery/extraction/audit pipeline
│   ├── config.py                 #   schema, VerifiableField, retry helper
│   ├── agent_extraction.py       #   Tavily discovery + LLM structuring
│   ├── agent_auditor.py          #   adversarial re-verification + Hunter.io
│   ├── graph_orchestrator.py     #   LangGraph StateGraph wiring
│   ├── clean_dataset.py          #   manual-review cleanup pass
│   └── regenerate_csv.py         #   rebuilds CSV from cleaned JSONL
│
├── docs/                          # Dataset outputs
│   ├── family_offices_dataset.csv
│   ├── family_offices_dataset.jsonl
│   ├── spotlight_records.json    #   the 3 deep-audited records
│   └── rejected_log.json         #   every rejected candidate + reason
│
├── frontend/                      # React 19 + Vite query UI
│   └── src/
│       ├── App.jsx                #   query box + source ledger UI
│       └── api.js                 #   thin fetch client for the backend
│
└── run_spotlight.py               # runs audit_record_deep on 3 records
```

---

## How to Run Locally

### Prerequisites
- Python 3.10+
- Node.js 18+
- API keys: [Groq](https://console.groq.com), [Tavily](https://tavily.com), [Hunter.io](https://hunter.io) (optional, used for email verification)

### 1. Environment variables
Create a `.env` file in `data_pipeline/` (the backend also reads from this path):

```env
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
HUNTER_API_KEY=your_hunter_key
```

### 2. Run the data pipeline (optional — a dataset is already included in `docs/`)

```bash
cd data_pipeline
pip install -r ../backend/requirements.txt
python graph_orchestrator.py        # discovers, extracts, audits, exports 50 records
python clean_dataset.py             # removes any manually-flagged false positives
python regenerate_csv.py            # keeps CSV in sync with cleaned JSONL

# Optional: run the deep cross-validation audit on 3 spotlight records
cd ..
python run_spotlight.py
```

### 3. Build the FAISS index

```bash
cd backend/vector_store
python build_index.py
```

This reads `docs/family_offices_dataset.jsonl`, chunks every verified field, embeds it with `all-MiniLM-L6-v2`, and writes `faiss_index.bin` + `chunk_metadata.pkl`.

### 4. Start the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Verify it's up: `curl http://localhost:8000/health`

### 5. Start the frontend

```bash
cd frontend
npm install
echo "VITE_API_URL=http://localhost:8000" > .env
npm run dev
```

The app will be available at `http://localhost:5173`.

---

## Tech Stack

| Layer | Technologies |
|---|---|
| Orchestration | LangGraph (`StateGraph`), LangChain |
| LLM inference | Groq (Llama 3.x family) via `langchain-groq` |
| Web discovery / re-fetch | Tavily Search & Extract API |
| Email verification | Hunter.io Email Finder |
| Vector search | FAISS (`faiss-cpu`), `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Backend API | FastAPI, Pydantic, Uvicorn |
| Frontend | React 19, Vite |
| Data | Pandas (CSV export), JSONL (nested/RAG-ready export) |

---

## Notes for Reviewers

- `docs/rejected_log.json` preserves every rejected candidate with a reason (duplicate, failed actionability bar, not a real family office, etc.) as a transparent methodology trail, not just the accepted 50.
- `docs/spotlight_records.json` is the output of the deep, double-sourced audit path (`audit_record_deep`) applied to 3 records, for closer methodology review.
- The `VerifiableField` schema enforces its own integrity rules at the Pydantic validation level (e.g. a field cannot be marked `VERIFIED` without a source URL) — so verification integrity is a structural property of the data, not just a convention the pipeline code happens to follow.