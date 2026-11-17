"""One-shot: populate CaseLaw.generated_summary for cases ingested before the column
existed, by parsing the summary back out of each case's already-written markdown
document (no LLM calls needed - the summary was already generated once at ingestion).

    python backfill_summaries.py
"""
import re

from sqlalchemy import inspect, text as sql_text

from db.database import SessionLocal, engine, Base
from db.models import CaseLaw

SUMMARY_RE = re.compile(r"## Generated Summary\n\n(.*?)\n\n---", re.DOTALL)


def main():
    Base.metadata.create_all(bind=engine)  # creates the table from scratch on a fresh DB
    # create_all doesn't add columns to a table that already exists, so for the live
    # DB (table already there, column new) we need an explicit ALTER TABLE.
    existing_columns = {c["name"] for c in inspect(engine).get_columns("case_laws")}
    if "generated_summary" not in existing_columns:
        with engine.begin() as conn:
            conn.execute(sql_text("ALTER TABLE case_laws ADD COLUMN generated_summary TEXT"))

    db = SessionLocal()
    try:
        cases = db.query(CaseLaw).filter(CaseLaw.generated_summary.is_(None)).all()
        updated = 0
        for case in cases:
            if not case.document_path:
                continue
            try:
                text = open(case.document_path, encoding="utf-8").read()
            except FileNotFoundError:
                continue
            match = SUMMARY_RE.search(text)
            if not match:
                continue
            case.generated_summary = match.group(1).strip()
            updated += 1
        db.commit()
        print(f"Backfilled {updated}/{len(cases)} cases.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
