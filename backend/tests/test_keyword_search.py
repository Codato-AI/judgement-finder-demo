"""Tests for the FTS5-backed keyword search."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from rag import keyword_search


def _db():
    return sessionmaker(bind=create_engine("sqlite:///:memory:"))()


def test_search_ranks_exact_term_match_first():
    db = _db()
    keyword_search.ensure_fts_table(db)
    db.execute(text(
        "INSERT INTO case_chunks_fts(rowid, chunk_text) VALUES (1, 'the dispute involved Section 66A of the IT Act')"
    ))
    db.execute(text(
        "INSERT INTO case_chunks_fts(rowid, chunk_text) VALUES (2, 'an unrelated judgment about land reform')"
    ))
    db.commit()

    results = keyword_search.search(db, "Section 66A", top_k=5)
    assert results[0]["id"] == 1
    assert results[0]["score"] > 0  # sign-flipped from sqlite's bm25(), higher = better


def test_search_handles_punctuation_without_raising():
    db = _db()
    keyword_search.ensure_fts_table(db)
    # Should not raise even though quotes/parens would otherwise break raw FTS5 syntax.
    results = keyword_search.search(db, 'what about "Article 21(1)" and AND OR NOT?', top_k=5)
    assert results == []


def test_remove_and_clear():
    db = _db()
    keyword_search.ensure_fts_table(db)
    db.execute(text("INSERT INTO case_chunks_fts(rowid, chunk_text) VALUES (1, 'privacy and liberty')"))
    db.commit()

    keyword_search.remove_chunks(db, [1])
    db.commit()
    assert keyword_search.search(db, "privacy", top_k=5) == []

    db.execute(text("INSERT INTO case_chunks_fts(rowid, chunk_text) VALUES (2, 'privacy again')"))
    db.commit()
    keyword_search.clear(db)
    db.commit()
    assert keyword_search.search(db, "privacy", top_k=5) == []


if __name__ == "__main__":
    test_search_ranks_exact_term_match_first()
    test_search_handles_punctuation_without_raising()
    test_remove_and_clear()
    print("keyword_search self-check passed")
