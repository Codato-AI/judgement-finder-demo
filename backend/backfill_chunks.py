"""One-shot: rebuild the chunk table + FAISS index from existing cases' anonymized text.

Run once after switching to chunked search. The old case-level index (one vector per
case, keyed by case_id) is incompatible with the new chunk-keyed index, so this wipes
both the FAISS store and any existing chunk rows, then re-indexes every case from the
anonymized_text already stored in the DB (no PDF re-ingestion needed).

    python backfill_chunks.py
"""
from db.database import SessionLocal, engine, Base
from db.models import CaseLaw, CaseChunk
from rag import vector_store, keyword_search
from rag.indexer import index_case


def main():
    Base.metadata.create_all(bind=engine)  # ensure case_chunks table exists
    db = SessionLocal()
    try:
        keyword_search.ensure_fts_table(db)
        vector_store.reset()
        db.query(CaseChunk).delete()
        keyword_search.clear(db)
        db.commit()

        cases = db.query(CaseLaw).filter(CaseLaw.anonymized_text.isnot(None)).all()
        total_chunks = 0
        for case in cases:
            # persist=False: skip the per-case FAISS disk write (it rewrites the
            # whole index/cache every call), flush once at the end instead.
            ids = index_case(case, db, persist=False)
            case.has_embedding = bool(ids)
            total_chunks += len(ids)
            db.commit()
            print(f"case {case.id}: {len(ids)} chunks")
        vector_store.flush()
        print(f"Done: {len(cases)} cases, {total_chunks} chunks indexed.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
