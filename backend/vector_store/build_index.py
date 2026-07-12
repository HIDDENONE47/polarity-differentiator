"""
build_index.py
Builds a FAISS vector index from the validated Family Office dataset (JSONL).

Chunking strategy: one chunk per high-value field (not per record), so a
query about AUM retrieves a precise AUM chunk rather than a noisy blob
containing the whole entity's data. Every chunk carries metadata back to
its source entity/field/verification-status for citation and grounding.

Only VERIFIED and COULD_NOT_VERIFY fields are indexed — this file refuses
to embed anything still UNVERIFIED, as a defensive backstop even though the
auditor should never let that state leak into the exported JSONL.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "docs"
VECTOR_STORE_DIR = Path(__file__).resolve().parent
DATASET_PATH = DOCS_DIR / "family_offices_dataset.jsonl"
INDEX_PATH = VECTOR_STORE_DIR / "faiss_index.bin"
METADATA_PATH = VECTOR_STORE_DIR / "chunk_metadata.pkl"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _make_chunk(entity_name: str, field_label: str, field_dict: dict) -> Dict[str, Any] | None:
    """
    Turns one VerifiableField dict into a chunk + metadata pair.
    Returns None for fields with no usable value (nothing to embed).
    """
    status = field_dict.get("status")
    value = field_dict.get("value")

    if status not in ("verified", "could_not_verify"):
        return None  # defensive: never index an UNVERIFIED field
    if not value:
        return None  # COULD_NOT_VERIFY fields with no value have nothing to embed

    text = f"{entity_name} — {field_label.replace('_', ' ')}: {value}"
    return {
        "text": text,
        "entity_name": entity_name,
        "field_label": field_label,
        "value": value,
        "status": status,
        "source_url": field_dict.get("source_url"),
        "verification_notes": field_dict.get("verification_notes"),
        "confidence_score": field_dict.get("confidence_score"),
    }


def build_chunks_from_record(record: dict) -> List[Dict[str, Any]]:
    """Extracts every indexable chunk from one record dict (loaded from JSONL)."""
    chunks = []
    entity_name = record.get("entity_name", "Unknown Entity")

    entity_level_fields = [
        "investing_thesis", "investing_mandate", "background_info",
        "aum", "corporate_linkedin",
    ]
    for field_label in entity_level_fields:
        field_dict = record.get(field_label, {})
        chunk = _make_chunk(entity_name, field_label, field_dict)
        if chunk:
            chunks.append(chunk)

    # Structural fields (not VerifiableFields, but still useful to surface)
    for plain_field in ["entity_type", "location", "website"]:
        val = record.get(plain_field)
        if val:
            chunks.append({
                "text": f"{entity_name} — {plain_field.replace('_', ' ')}: {val}",
                "entity_name": entity_name,
                "field_label": plain_field,
                "value": val,
                "status": "structural",
                "source_url": record.get("discovery_source"),
                "verification_notes": None,
                "confidence_score": None,
            })

    for principal in record.get("principals", []):
        p_name = principal.get("name", "Unknown Principal")
        p_title = principal.get("title", "")
        label_prefix = f"principal_{p_name}"

        if p_title:
            chunks.append({
                "text": f"{entity_name} — principal {p_name} holds the title: {p_title}",
                "entity_name": entity_name,
                "field_label": f"{label_prefix}_title",
                "value": p_title,
                "status": "structural",
                "source_url": None,
                "verification_notes": None,
                "confidence_score": None,
            })

        for contact_field in ["linkedin_url", "work_email", "direct_phone"]:
            field_dict = principal.get(contact_field, {})
            chunk = _make_chunk(f"{entity_name} principal {p_name}", contact_field, field_dict)
            if chunk:
                chunks.append(chunk)

    for signal in record.get("signals", []):
        source_dict = signal.get("source", {})
        if source_dict.get("status") in ("verified", "could_not_verify") and source_dict.get("value"):
            desc = signal.get("description", "")
            signal_type = signal.get("signal_type", "signal")
            date = signal.get("date", "")
            text = f"{entity_name} — {signal_type}: {desc}" + (f" (dated {date})" if date else "")
            chunks.append({
                "text": text,
                "entity_name": entity_name,
                "field_label": f"signal_{signal_type}",
                "value": desc,
                "status": source_dict.get("status"),
                "source_url": source_dict.get("source_url"),
                "verification_notes": source_dict.get("verification_notes"),
                "confidence_score": source_dict.get("confidence_score"),
            })

    return chunks


def build_index():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Run graph_orchestrator.py first."
        )

    all_chunks: List[Dict[str, Any]] = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            all_chunks.extend(build_chunks_from_record(record))

    if not all_chunks:
        raise ValueError("No indexable chunks were produced — check the dataset file contents.")

    print(f"Embedding {len(all_chunks)} chunks from {DATASET_PATH.name}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # inner product on normalized vectors = cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"Index built: {index.ntotal} vectors, dimension {dimension}.")
    print(f"Saved to: {INDEX_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    build_index()