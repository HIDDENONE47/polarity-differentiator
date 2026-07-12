"""
main.py
FastAPI server entry point for the Micro-RAG.

Responsibility: expose the RAG (query -> grounded answer + sources) over
HTTP for the React frontend, plus a health check and a basic dataset stats
endpoint. All actual retrieval/grounding logic lives in rag_retriever.py —
this file is routing and request/response shaping only.
"""

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# .env lives in data_pipeline/, not backend/ — same reasoning as rag_retriever.py
_ENV_PATH = Path(__file__).resolve().parent.parent / "data_pipeline" / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

from rag_retriever import answer_query, _state, _ensure_loaded

# Single, clean initialization
app = FastAPI(title="PolarityIQ Family Office Micro-RAG", version="1.0.0")

# Single, clean CORS block
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all domains (safe enough for this assessment)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 8
    entity_filter: Optional[str] = None
    field_filter: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: list
    chunks_used: int
    grounded: bool

@app.on_event("startup")
def load_index_on_startup():
    """
    Loads the embedding model + FAISS index once, at server startup, rather
    than lazily on the first request — so the first real user query isn't
    the one eating a multi-second cold-start cost, and so a missing/corrupt
    index is caught immediately in the logs rather than surfacing as an
    opaque 500 error on someone's first query during a demo.
    """
    try:
        _ensure_loaded()
        print(f"Vector store loaded: {len(_state.metadata)} chunks indexed.")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("Server starting without a loaded index — /query will fail until build_index.py is run.")

@app.get("/")
def root():
    return {"service": "PolarityIQ Family Office Micro-RAG", "status": "running"}

@app.get("/health")
def health_check():
    """Reports whether the vector store actually loaded, not just whether the process is up."""
    index_loaded = _state.index is not None
    return {
        "status": "healthy" if index_loaded else "degraded",
        "index_loaded": index_loaded,
        "chunks_indexed": len(_state.metadata) if _state.metadata else 0,
    }

@app.post("/query", response_model=QueryResponse)
def query_dataset(request: QueryRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = answer_query(
            query=request.query,
            top_k=request.top_k,
            entity_filter=request.entity_filter,
            field_filter=request.field_filter,
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Vector store not available: {e}. Run build_index.py first.",
        )
    return result

@app.get("/stats")
def dataset_stats():
    """Basic dataset visibility for the frontend or a demo walkthrough — entity count, etc."""
    if _state.metadata is None:
        raise HTTPException(status_code=503, detail="Index not loaded yet.")

    entities = {c["entity_name"].split(" principal ")[0] for c in _state.metadata}
    return {
        "total_chunks": len(_state.metadata),
        "unique_entities": len(entities),
    }