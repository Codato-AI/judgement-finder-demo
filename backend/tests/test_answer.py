"""Tests for answer prompt assembly: per-case grouping, [#N] numbering aligned to rank
order, and source->case mapping. The LLM call (run_llm) isn't tested, no network here."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.database import Base
from db.models import CaseLaw
from rag.answer import build_prompt, MAX_CHUNKS_PER_CASE


def _db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _seed(db):
    db.add_all([
        CaseLaw(id=10, subject="Right to privacy", original_case_no="W.P. 494/2012",
                judgment_date="24 August, 2017"),
        CaseLaw(id=20, subject="Land acquisition", original_case_no="C.A. 111/1973",
                judgment_date="24/04/1973"),
    ])
    db.commit()


def _chunks():
    # Fused rank order: case 10 ranks first, then a second chunk of 10, then case 20.
    return [
        {"chunk_id": 1, "case_id": 10, "chunk_text": "privacy is intrinsic to liberty", "vector_score": 0.7},
        {"chunk_id": 2, "case_id": 10, "chunk_text": "the state must justify intrusion", "vector_score": 0.6},
        {"chunk_id": 3, "case_id": 20, "chunk_text": "fair compensation is mandatory", "vector_score": 0.55},
    ]


def test_sources_numbered_in_rank_order():
    db = _db(); _seed(db)
    out = build_prompt("is privacy a right?", _chunks(), db)
    assert [s["n"] for s in out["sources"]] == [1, 2]
    assert out["sources"][0]["case_id"] == 10  # top-ranked case is #1
    assert out["sources"][1]["case_id"] == 20


def test_source_passage_and_label_map_to_case():
    db = _db(); _seed(db)
    out = build_prompt("q", _chunks(), db)
    s1 = out["sources"][0]
    assert s1["passage"] == "privacy is intrinsic to liberty"  # best chunk of the case
    assert "Right to privacy" in s1["label"] and "2017" in s1["label"]


def test_prompt_numbers_passages_and_caps_per_case():
    db = _db(); _seed(db)
    out = build_prompt("q", _chunks(), db)
    assert "[#1]" in out["user"] and "[#2]" in out["user"]
    assert "privacy is intrinsic to liberty" in out["user"]
    # case 10 has two chunks in input; cap allows up to MAX_CHUNKS_PER_CASE
    assert MAX_CHUNKS_PER_CASE >= 2
    assert "the state must justify intrusion" in out["user"]


def test_empty_chunks_returns_none():
    db = _db(); _seed(db)
    assert build_prompt("q", [], db) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
