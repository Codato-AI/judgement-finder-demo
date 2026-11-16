"""BM25 keyword search over chunk text via SQLite FTS5. Complements vector search
so exact statute numbers / citations aren't missed by pure semantic matching.

A free-standing FTS5 table (not an FTS5 "external content" table) so writes are
simple explicit calls at the same points chunks are written elsewhere, no triggers.
"""
import re
from sqlalchemy import text
from sqlalchemy.orm import Session

from db.models import CaseChunk

_WORD = re.compile(r"\w+")


def ensure_fts_table(db: Session) -> None:
    db.execute(text("CREATE VIRTUAL TABLE IF NOT EXISTS case_chunks_fts USING fts5(chunk_text)"))
    db.commit()


def index_chunks(db: Session, chunks: list[CaseChunk]) -> None:
    """chunks must already have .id assigned (i.e. inserted + flushed)."""
    for c in chunks:
        db.execute(
            text("INSERT INTO case_chunks_fts(rowid, chunk_text) VALUES (:id, :text)"),
            {"id": c.id, "text": c.chunk_text},
        )


def remove_chunks(db: Session, chunk_ids: list[int]) -> None:
    for cid in chunk_ids:
        db.execute(text("DELETE FROM case_chunks_fts WHERE rowid = :id"), {"id": cid})


def clear(db: Session) -> None:
    db.execute(text("DELETE FROM case_chunks_fts"))


def _sanitize_query(query: str) -> str:
    """OR together quoted word tokens so free-text user input (punctuation,
    FTS5 query-syntax characters) can't break the MATCH query."""
    tokens = _WORD.findall(query)
    return " OR ".join(f'"{t}"' for t in tokens)


def search(db: Session, query: str, top_k: int) -> list[dict]:
    """Returns [{"id": chunk_id, "score": float}] sorted best-first.
    sqlite's bm25(): smaller (more negative) = better match; sign-flipped here
    so callers treat "higher = better" consistently with vector_store.search."""
    match = _sanitize_query(query)
    if not match:
        return []
    rows = db.execute(
        text(
            "SELECT rowid, bm25(case_chunks_fts) AS rank FROM case_chunks_fts "
            "WHERE case_chunks_fts MATCH :match ORDER BY rank LIMIT :k"
        ),
        {"match": match, "k": top_k},
    ).fetchall()
    return [{"id": row[0], "score": -row[1]} for row in rows]
