"""Tests for search filter eligibility (year range + acts-cited substring)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.database import Base
from db.models import CaseLaw
from rag.finder import eligible_case_ids


def _db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _seed(db):
    db.add_all([
        CaseLaw(id=1, judgment_date="19 May, 1950", acts_cited="Preventive Detention Act"),
        CaseLaw(id=2, judgment_date="24/04/1973", acts_cited="Kerala Land Reforms Act"),
        CaseLaw(id=3, judgment_date="13 August, 1997", acts_cited="Constitution of India Articles 14, 21"),
        CaseLaw(id=4, judgment_date="10 January, 2020", acts_cited=None),
    ])
    db.commit()


def test_no_filters_returns_none():
    db = _db()
    _seed(db)
    assert eligible_case_ids(db, None, None, None) is None


def test_year_range_handles_both_date_formats():
    db = _db()
    _seed(db)
    assert eligible_case_ids(db, 1970, 2000, None) == {2, 3}


def test_acts_cited_substring_is_case_insensitive():
    db = _db()
    _seed(db)
    assert eligible_case_ids(db, None, None, "article") == {3}


def test_combined_filters_intersect():
    db = _db()
    _seed(db)
    assert eligible_case_ids(db, 1990, None, "constitution") == {3}


def test_no_match_returns_empty_set_not_none():
    db = _db()
    _seed(db)
    assert eligible_case_ids(db, 2100, None, None) == set()


if __name__ == "__main__":
    test_no_filters_returns_none()
    test_year_range_handles_both_date_formats()
    test_acts_cited_substring_is_case_insensitive()
    test_combined_filters_intersect()
    test_no_match_returns_empty_set_not_none()
    print("finder filters self-check passed")
