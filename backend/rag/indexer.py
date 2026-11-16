"""Chunk a judgment into overlapping windows, embed each, and index it in FAISS.

Shared by live ingestion (_process_row) and the one-shot backfill script so both
produce identical chunk rows + vectors.
"""
from sqlalchemy.orm import Session

from db.models import CaseLaw, CaseChunk
from rag.embedder import embed
from rag.vector_store import add_vectors_batch
from rag import keyword_search

# Fixed-size sliding window. Robust to PDF text with missing paragraph breaks.
# Revisit sizes / subject-prepend once the eval set shows it's worth it.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


def split_into_chunks(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping fixed-size character windows."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    step = size - overlap
    chunks = [text[i:i + size] for i in range(0, len(text), step)]
    return [c for c in chunks if c.strip()]


def index_case(case: CaseLaw, db: Session) -> list[int]:
    """Chunk a case's anonymized text, persist CaseChunk rows, embed and add to FAISS.
    Returns the chunk ids added (so a failed ingestion can remove them from FAISS)."""
    chunks = split_into_chunks(case.anonymized_text)
    if not chunks:
        return []
    rows = [CaseChunk(case_id=case.id, chunk_index=i, chunk_text=c) for i, c in enumerate(chunks)]
    db.add_all(rows)
    db.flush()  # assigns row ids
    vecs = embed(chunks)
    add_vectors_batch([r.id for r in rows], vecs)
    keyword_search.index_chunks(db, rows)
    return [r.id for r in rows]
