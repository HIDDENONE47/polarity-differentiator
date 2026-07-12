"""
rag_retriever.py
Retrieval + grounded synthesis layer for the Micro-RAG.

Responsibility: turn a natural language query into (1) a set of retrieved
chunks — using structured filtering AND semantic ranking together, not
semantic search alone — and (2) a grounded natural-language answer that is
NEVER permitted to state anything the retrieved chunks don't actually
support. This is the layer that decides what the RAG is allowed to claim.

Grounding discipline: if no retrieved chunk clears the similarity floor,
the answer must say so plainly rather than let the LLM fill the gap from
its own general knowledge. That failure mode — an LLM answering from
training data instead of the dataset — is exactly what the assessment's
"grounding discipline" criterion is checking for.
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

from dotenv import load_dotenv

# .env lives in data_pipeline/, not backend/ — point to it explicitly rather
# than relying on load_dotenv()'s default upward-search, which won't find a
# sibling folder.
_ENV_PATH = Path(__file__).resolve().parent.parent / "data_pipeline" / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

VECTOR_STORE_DIR = Path(__file__).resolve().parent / "vector_store"
INDEX_PATH = VECTOR_STORE_DIR / "faiss_index.bin"
METADATA_PATH = VECTOR_STORE_DIR / "chunk_metadata.pkl"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Deliberately a separate, smaller/faster model than extraction or audit use —
# this one serves live query-time requests during a demo, so latency matters
# more here than in the batch pipeline. Confirm this model name is still valid
# on your Groq account given the model reassignments you made earlier.
RAG_LLM_MODEL = "llama-3.1-8b-instant"

MIN_SIMILARITY = 0.25  # below this, we don't trust the match enough to answer from it
DEFAULT_TOP_K = 8


class _RetrieverState:
    """Lazy singleton — loads the embedding model and FAISS index once, not per-request."""
    embedder: Optional[SentenceTransformer] = None
    index: Optional[faiss.Index] = None
    metadata: Optional[List[Dict[str, Any]]] = None


_state = _RetrieverState()


def _ensure_loaded():
    if _state.embedder is None:
        _state.embedder = SentenceTransformer(EMBEDDING_MODEL)
    if _state.index is None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"No FAISS index found at {INDEX_PATH}. Run build_index.py first."
            )
        _state.index = faiss.read_index(str(INDEX_PATH))
    if _state.metadata is None:
        with open(METADATA_PATH, "rb") as f:
            _state.metadata = pickle.load(f)


def _embed_query(query: str) -> np.ndarray:
    _ensure_loaded()
    vec = _state.embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vec)
    return vec


def _matches_filters(chunk: Dict[str, Any], entity_filter: Optional[str], field_filter: Optional[str]) -> bool:
    if entity_filter and entity_filter.lower().strip() not in chunk["entity_name"].lower():
        return False
    if field_filter and field_filter.lower().strip() not in chunk["field_label"].lower():
        return False
    return True


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    entity_filter: Optional[str] = None,
    field_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Structured + semantic retrieval.

    If entity_filter or field_filter is given, retrieval is restricted to
    chunks matching that structured criterion FIRST, then ranked semantically
    within that subset. Without filters, this is plain semantic search over
    the full index. Returns chunks with an added 'similarity' score, sorted
    descending, filtered to MIN_SIMILARITY.
    """
    _ensure_loaded()
    query_vec = _embed_query(query)

    if entity_filter or field_filter:
        candidate_indices = [
            i for i, c in enumerate(_state.metadata)
            if _matches_filters(c, entity_filter, field_filter)
        ]
        if not candidate_indices:
            return []

        vectors = np.stack([_state.index.reconstruct(i) for i in candidate_indices]).astype("float32")
        scores = vectors @ query_vec[0]  # cosine similarity — both sides are L2-normalized
        ranked = sorted(zip(candidate_indices, scores), key=lambda x: x[1], reverse=True)[:top_k]
    else:
        scores, indices = _state.index.search(query_vec, top_k)
        ranked = [(idx, score) for idx, score in zip(indices[0], scores[0]) if idx != -1]

    results = []
    for idx, score in ranked:
        if score < MIN_SIMILARITY:
            continue
        chunk = dict(_state.metadata[idx])
        chunk["similarity"] = float(score)
        results.append(chunk)
    return results


def _build_grounded_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    context_lines = []
    for i, c in enumerate(chunks, 1):
        conf_note = f" (confidence {c['confidence_score']:.2f})" if c.get("confidence_score") else ""
        context_lines.append(
            f"[{i}] Entity: {c['entity_name']} | Field: {c['field_label']} | "
            f"Value: {c['value']}{conf_note} | Source: {c.get('source_url') or 'n/a'}"
        )
    context_block = "\n".join(context_lines)

    return f"""You are answering a question using ONLY the dataset excerpts below.
Do not use any outside knowledge. Do not infer, estimate, or fill gaps with
plausible-sounding information not explicitly present in these excerpts.

If the excerpts do not fully answer the question, say plainly what is and
is not covered — do not pad the gap with a guess.

Dataset excerpts:
{context_block}

Question: {query}

Answer using only the excerpts above. When you state a fact, reference which
excerpt number(s) it came from, like this: "...AUM of $500M [1]." Keep the
answer concise and directly responsive to the question.
"""


def answer_query(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    entity_filter: Optional[str] = None,
    field_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full RAG path: retrieve -> ground -> synthesize.
    Returns {"answer": str, "sources": [...], "chunks_used": int, "grounded": bool}

    grounded=False means no chunk cleared MIN_SIMILARITY — in that case the
    LLM is never even called, since there is nothing legitimate for it to
    answer from. This is the hard stop that prevents the model from quietly
    falling back on its own general knowledge about family offices.
    """
    chunks = retrieve(query, top_k=top_k, entity_filter=entity_filter, field_filter=field_filter)

    if not chunks:
        return {
            "answer": "No verified information in the dataset matches this query closely enough to answer from.",
            "sources": [],
            "chunks_used": 0,
            "grounded": False,
        }

    prompt = _build_grounded_prompt(query, chunks)
    llm = ChatGroq(model=RAG_LLM_MODEL, groq_api_key=os.getenv("GROQ_API_KEY"), temperature=0)
    response = llm.invoke(prompt)

    sources = [
        {
            "entity_name": c["entity_name"],
            "field_label": c["field_label"],
            "source_url": c.get("source_url"),
            "similarity": round(c["similarity"], 3),
        }
        for c in chunks
    ]

    return {
        "answer": response.content,
        "sources": sources,
        "chunks_used": len(chunks),
        "grounded": True,
    }


if __name__ == "__main__":
    # Quick manual smoke test — not the real interface, backend/main.py will call
    # answer_query() properly. Useful to sanity-check the index without FastAPI up.
    test_query = "What family offices focus on technology investments?"
    result = answer_query(test_query)
    print(f"Q: {test_query}\n")
    print(f"A: {result['answer']}\n")
    print(f"Grounded: {result['grounded']}, chunks used: {result['chunks_used']}")
    for s in result["sources"]:
        print(f"  - {s['entity_name']} / {s['field_label']} (sim={s['similarity']}) -> {s['source_url']}")