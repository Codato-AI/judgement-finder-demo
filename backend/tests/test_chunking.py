"""Tests for chunk splitting and the chunk-hits -> distinct-cases merge."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.database import Base
from db.models import CaseChunk
from rag import finder
from rag.indexer import split_into_chunks


def test_split_into_chunks():
    assert split_into_chunks("") == []
    assert split_into_chunks("   ") == []
    assert split_into_chunks("short text") == ["short text"]

    long = "".join(chr(65 + (i % 26)) for i in range(2000))
    chunks = split_into_chunks(long, size=900, overlap=150)
    assert len(chunks) == 3                     # windows start at 0, 750, 1500
    assert chunks[0] == long[0:900]
    assert chunks[1] == long[750:1650]
    assert chunks[2] == long[1500:2000]
    assert chunks[0][-150:] == chunks[1][:150]  # 150-char overlap between neighbours


def test_find_similar_merges_chunks_per_case(monkeypatch):
    db = sessionmaker(bind=create_engine("sqlite:///:memory:"))()
    Base.metadata.create_all(db.get_bind())
    # case 10 -> two chunks (1, 2); case 20 -> one chunk (3)
    db.add_all([
        CaseChunk(id=1, case_id=10, chunk_index=0, chunk_text="alpha"),
        CaseChunk(id=2, case_id=10, chunk_index=1, chunk_text="beta"),
        CaseChunk(id=3, case_id=20, chunk_index=0, chunk_text="gamma"),
    ])
    db.commit()

    # Canned vector hits, already sorted desc by score: chunk 2 best, then 3, then 1.
    # Keyword search finds nothing, so fusion is driven by vector rank alone.
    monkeypatch.setattr(finder, "embed_one", lambda q: None)
    monkeypatch.setattr(finder.vector_store, "search", lambda vec, top_k: [
        {"id": 2, "score": 0.9},
        {"id": 3, "score": 0.7},
        {"id": 1, "score": 0.5},
    ])
    monkeypatch.setattr(finder.keyword_search, "search", lambda db, query, top_k: [])

    results = finder.find_similar("q", db, top_k=5)
    assert [r["case_id"] for r in results] == [10, 20]   # distinct cases, best-first
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["snippet"] == "beta"               # higher-ranked chunk wins for case 10
    assert results[1]["snippet"] == "gamma"


def test_rrf_fuse_rewards_agreement_between_lists():
    vector_hits = [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.8}, {"id": 3, "score": 0.7}]
    keyword_hits = [{"id": 2, "score": 10.0}, {"id": 4, "score": 5.0}]

    fused = finder._rrf_fuse(vector_hits, keyword_hits)

    assert set(fused) == {1, 2, 3, 4}
    # chunk 2 is ranked in both lists, so it should beat chunk 1 (rank 1 in only one list)
    assert fused[2] > fused[1]


if __name__ == "__main__":
    test_split_into_chunks()
    test_rrf_fuse_rewards_agreement_between_lists()
    import pytest
    pytest.main([__file__, "-q"])
