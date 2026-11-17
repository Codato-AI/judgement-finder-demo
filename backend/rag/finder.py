"""Judgement Finder: embed the query, retrieve matching chunks via both vector
(semantic) and keyword (BM25) search, fuse the two rankings, merge back to cases.
"""
import re

from sqlalchemy.orm import Session

from db.models import CaseChunk, CaseLaw
from rag.embedder import embed_one
from rag import vector_store, keyword_search

YEAR_RE = re.compile(r"\d{4}")


def eligible_case_ids(
    db: Session, year_from: int | None, year_to: int | None, acts_cited: str | None
) -> set[int] | None:
    """Cases matching the given filters, or None if no filters were given (meaning:
    don't restrict). judgment_date is free text in inconsistent formats, but every
    format seen has an unambiguous 4-digit year, so that's what's filtered on."""
    if year_from is None and year_to is None and not acts_cited:
        return None
    query = db.query(CaseLaw.id, CaseLaw.judgment_date)
    if acts_cited:
        query = query.filter(CaseLaw.acts_cited.ilike(f"%{acts_cited}%"))
    if year_from is None and year_to is None:
        return {row.id for row in query.all()}
    ids = set()
    for row in query.all():
        match = YEAR_RE.search(row.judgment_date or "")
        if not match:
            continue
        year = int(match.group())
        if year_from is not None and year < year_from:
            continue
        if year_to is not None and year > year_to:
            continue
        ids.add(row.id)
    return ids

# Chunks-per-case means we must over-fetch chunk hits to end up with top_k distinct cases.
FANOUT = 8

# Reciprocal Rank Fusion: combine two heterogeneous-scale rankings (cosine similarity,
# BM25) by rank position instead of raw score, so no score normalization is needed.
# Standard constant from the RRF literature, not tuned for this corpus.
RRF_K = 60

# Reject the whole query if the best raw vector cosine score is below this, i.e.
# nothing in the corpus is even semantically plausible. Must check the raw score, not
# the fused RRF score - RRF collapses score magnitude into rank position, so it can't
# tell "weak because nothing fits" from "weak but still the best option available".
# Raw vector top scores separate cleanly on this corpus/embedding model: no-match
# queries land around 0.29-0.454, genuine matches 0.499-0.74. Threshold is the midpoint.
MIN_VECTOR_SCORE = 0.46


def _rrf_fuse(*ranked_lists: list[dict]) -> dict[int, float]:
    """Each ranked_list is already sorted best-first: [{"id": ..., "score": ...}, ...].
    Returns {chunk_id: fused_score}, higher is better."""
    fused: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            fused[hit["id"]] = fused.get(hit["id"], 0.0) + 1.0 / (RRF_K + rank)
    return fused


def find_similar(
    query: str, db: Session, top_k: int = 5, allowed_case_ids: set[int] | None = None
) -> list[dict]:
    """Return up to top_k distinct cases, each with its best-matching chunk as the
    snippet. Shape: [{case_id, score, snippet}], ordered by fused rank; score is the
    raw cosine similarity for display, not the fused value used to pick the order.
    allowed_case_ids, if given, restricts results to that set (used by search filters)."""
    vec = embed_one(query)
    vector_hits = vector_store.search(vec, top_k=top_k * FANOUT)
    if not vector_hits or vector_hits[0]["score"] < MIN_VECTOR_SCORE:
        return []  # nothing in the corpus is even semantically plausible

    keyword_hits = keyword_search.search(db, query, top_k=top_k * FANOUT)
    fused = _rrf_fuse(vector_hits, keyword_hits)
    # RRF drives ranking order (rewards agreement between both signals), but its score
    # is a rank-based number with no meaning as a human-facing "similarity" percentage -
    # report the raw cosine score instead, falling back to the best known vector score
    # for the rare chunk that was found by keyword search alone (no cosine score of its own).
    vector_score_by_chunk = {h["id"]: h["score"] for h in vector_hits}
    best_vector_score = vector_hits[0]["score"]

    chunk_ids = [cid for cid, _ in sorted(fused.items(), key=lambda kv: -kv[1])]
    rows = {r.id: r for r in db.query(CaseChunk).filter(CaseChunk.id.in_(chunk_ids)).all()}

    best: dict[int, dict] = {}  # case_id -> hit; first (highest-ranked) chunk per case wins
    for cid in chunk_ids:
        row = rows.get(cid)
        if row is None or row.case_id in best:
            continue
        if allowed_case_ids is not None and row.case_id not in allowed_case_ids:
            continue
        display_score = vector_score_by_chunk.get(cid, best_vector_score)
        best[row.case_id] = {"case_id": row.case_id, "score": display_score, "snippet": row.chunk_text}

    return list(best.values())[:top_k]
