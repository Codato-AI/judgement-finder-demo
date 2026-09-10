"""
KanoonSaathi — Demo Backend
FastAPI app exposing ingestion pipeline, judgement finder, and LLM model tester.
"""
from contextlib import asynccontextmanager
from pathlib import Path
import asyncio

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config import settings
from db.database import engine, get_db, Base
from db.models import CaseLaw, ModelTestResult
from pipeline.ingestion import load_dataset, row_to_dict
from pipeline.anonymizer import anonymize
from pipeline.document_creator import create_document
from llm.groq_client import chat
from llm.model_tester import run_model_test, MODELS
from llm.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from rag.embedder import embed_one, embed
from rag.vector_store import add_vector, add_vectors_batch, search, store_size, remove_vector
from rag.finder import find_similar


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    settings.documents_path()
    settings.vector_store_path()
    yield


app = FastAPI(
    title="KanoonSaathi API",
    description="Legal case law ingestion pipeline, judgement finder, and LLM comparison",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Stats ────────────────────────────────────────────────────────────────────

@app.get("/api/stats")
def get_stats(db: Session = Depends(get_db)):
    total_cases = db.query(CaseLaw).count()
    total_docs = db.query(CaseLaw).filter(CaseLaw.document_path.isnot(None)).count()
    total_tests = db.query(ModelTestResult).count()
    return {
        "total_cases": total_cases,
        "documents_generated": total_docs,
        "model_tests_run": total_tests,
        "vector_store_size": store_size(),
        "models_available": [m["label"] for m in MODELS],
    }


# ─── Bulk CSV Ingestion ───────────────────────────────────────────────────────

class BulkIngestResponse(BaseModel):
    ingested: int
    test_set_size: int
    skipped: int
    message: str


def _process_row(row_dict: dict, db: Session) -> CaseLaw | None:
    text = row_dict.get("text", "")
    if len(text) < 50:
        print(f"DEBUG: Skipping row - text too short ({len(text)} chars)")
        return None

    # Create savepoint for this row
    sp = db.begin_nested()
    try:
        known_names = [
            n for n in [row_dict.get("petitioner"), row_dict.get("respondent"), row_dict.get("judges")]
            if n and len(n) > 2
        ]

        print(f"DEBUG: Starting anonymization for text length {len(text)}")
        anon = anonymize(text, known_names=known_names)
        print(f"DEBUG: Anonymization complete")

        # Anonymize structured fields too
        anon_petitioner = "John Doe"
        anon_respondent = "Jane Doe"
        anon_judges = "Honourable Bench"

        case = CaseLaw(
            original_case_no=row_dict.get("case_no") or None,
            judgment_date=row_dict.get("date") or None,
            subject=row_dict.get("subject") or None,
            acts_cited=row_dict.get("acts") or None,
            anonymized_text=anon.anonymized_text[:50000],
        )
        db.add(case)
        db.flush()
        print(f"DEBUG: Case created with ID {case.id}")

        # Generate summary via best available model
        print(f"DEBUG: Starting summary generation")
        summary = _generate_summary(anon.anonymized_text)
        print(f"DEBUG: Summary generated: {len(summary)} chars")

        anon_case_data = {
            "date": row_dict.get("date"),
            "subject": row_dict.get("subject"),
            "acts": row_dict.get("acts"),
            "verdict": row_dict.get("verdict"),
            "petitioner": anon_petitioner,
            "respondent": anon_respondent,
            "judges": anon_judges,
        }
        doc_path = create_document(case.id, anon_case_data, summary)
        case.document_path = str(doc_path)
        print(f"DEBUG: Document created at {doc_path}")

        # Embed and store in vector store
        print(f"DEBUG: Starting embedding generation")
        embed_text = f"{row_dict.get('subject', '')} {anon.anonymized_text[:500]}"
        vec = embed_one(embed_text)
        faiss_pos = add_vector(case.id, vec)
        case.has_embedding = True
        case.faiss_index = faiss_pos
        print(f"DEBUG: Embedding added at position {faiss_pos}")

        # Commit savepoint on success
        sp.commit()
        return case
    except Exception as e:
        # Rollback savepoint on failure
        sp.rollback()
        print(f"ERROR: Failed to process row: {str(e)}")
        import traceback
        traceback.print_exc()

        # Cleanup: remove document and metadata files if case was created
        if 'case' in locals() and case.id:
            try:
                from pathlib import Path
                doc_path = Path(case.document_path) if case.document_path else None
                meta_path = settings.documents_path() / f"case_{case.id:06d}_meta.json"

                if doc_path and doc_path.exists():
                    doc_path.unlink()
                    print(f"DEBUG: Cleaned up document file for case {case.id}")
                if meta_path.exists():
                    meta_path.unlink()
                    print(f"DEBUG: Cleaned up metadata file for case {case.id}")

                # Remove FAISS entry
                remove_vector(case.id)
                print(f"DEBUG: Cleaned up FAISS entry for case {case.id}")
            except Exception as cleanup_err:
                print(f"ERROR: Cleanup failed for case {case.id}: {cleanup_err}")

        return None


def _generate_summary(anonymized_text: str, model_id: str = "qwen/qwen3.6-27b") -> str:
    truncated = anonymized_text[:6000]
    user_msg = USER_PROMPT_TEMPLATE.format(anonymized_text=truncated)
    resp = chat(model_id, SYSTEM_PROMPT, user_msg, max_tokens=1200)
    if resp["error"] or not resp["text"]:
        # Fallback to GPT OSS if Qwen fails
        resp = chat("openai/gpt-oss-20b", SYSTEM_PROMPT, user_msg, max_tokens=1200)
    return resp["text"] or "[Summary generation failed]"


@app.post("/api/ingest/bulk", response_model=BulkIngestResponse)
async def bulk_ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted")

    tmp_path = settings.data_path() / file.filename
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_bytes(await file.read())

    print(f"DEBUG: Loading CSV from {tmp_path}")
    pipeline_df, test_df = load_dataset(tmp_path)
    print(f"DEBUG: Pipeline rows: {len(pipeline_df)}, Test rows: {len(test_df)}")
    print(f"DEBUG: Pipeline columns: {list(pipeline_df.columns)}")

    ingested = 0
    skipped = 0
    for idx, row in pipeline_df.iterrows():
        row_dict = row_to_dict(row)
        print(f"DEBUG: Processing row {idx}, text length: {len(row_dict.get('text', ''))}")
        case = _process_row(row_dict, db)
        if case:
            ingested += 1
            print(f"DEBUG: Successfully ingested case {idx}")
        else:
            skipped += 1
            print(f"DEBUG: Skipped case {idx}")

    db.commit()
    print(f"DEBUG: Final count - ingested: {ingested}, skipped: {skipped}")

    return BulkIngestResponse(
        ingested=ingested,
        test_set_size=len(test_df),
        skipped=skipped,
        message=f"Ingested {ingested} cases into pipeline. {len(test_df)} cases reserved for LLM testing.",
    )


# ─── Cases ────────────────────────────────────────────────────────────────────

@app.get("/api/cases")
def list_cases(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    cases = db.query(CaseLaw).offset(skip).limit(limit).all()
    return [
        {
            "id": c.id,
            "ks_id": f"KS-{c.id:06d}",
            "case_no": c.original_case_no,
            "date": c.judgment_date,
            "subject": c.subject,
            "has_document": bool(c.document_path),
            "has_embedding": c.has_embedding,
            "ingested_at": c.ingested_at.isoformat(),
        }
        for c in cases
    ]


@app.get("/api/cases/{case_id}")
def get_case(case_id: int, db: Session = Depends(get_db)):
    case = db.query(CaseLaw).filter(CaseLaw.id == case_id).first()
    if not case:
        raise HTTPException(404, "Case not found")
    return {
        "id": case.id,
        "ks_id": f"KS-{case.id:06d}",
        "case_no": case.original_case_no,
        "date": case.judgment_date,
        "subject": case.subject,
        "acts_cited": case.acts_cited,
        "anonymized_text": case.anonymized_text,
        "document_path": case.document_path,
        "has_embedding": case.has_embedding,
        "ingested_at": case.ingested_at.isoformat(),
    }


@app.get("/api/cases/{case_id}/document")
def download_document(case_id: int, db: Session = Depends(get_db)):
    case = db.query(CaseLaw).filter(CaseLaw.id == case_id).first()
    if not case or not case.document_path:
        raise HTTPException(404, "Document not found")
    doc_path = Path(case.document_path)
    if not doc_path.exists():
        raise HTTPException(404, "Document file missing from disk")
    return FileResponse(
        path=str(doc_path),
        media_type="text/markdown",
        filename=f"KS-{case.id:06d}_summary.md",
    )


@app.get("/api/cases/{case_id}/document/view")
def view_document(case_id: int, db: Session = Depends(get_db)):
    case = db.query(CaseLaw).filter(CaseLaw.id == case_id).first()
    if not case or not case.document_path:
        raise HTTPException(404, "Document not found")
    doc_path = Path(case.document_path)
    if not doc_path.exists():
        raise HTTPException(404, "Document file missing from disk")
    return {"content": doc_path.read_text(encoding="utf-8")}


# ─── Judgement Finder ─────────────────────────────────────────────────────────

class FinderRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/api/finder/search")
def judgement_finder(req: FinderRequest, db: Session = Depends(get_db)):
    print(f"DEBUG: Search request received - query_id: {id(req)}, query_length: {len(req.query)}, top_k: {req.top_k}")

    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    print(f"DEBUG: Calling find_similar...")
    hits = find_similar(req.query, top_k=req.top_k)
    print(f"DEBUG: find_similar returned {len(hits)} hits: {hits}")

    if not hits:
        return {"results": [], "message": "No cases indexed yet. Run bulk ingestion first."}

    results = []
    for hit in hits:
        print(f"DEBUG: Looking up case ID {hit['case_id']}")
        case = db.query(CaseLaw).filter(CaseLaw.id == hit["case_id"]).first()
        if not case:
            print(f"DEBUG: Case {hit['case_id']} not found in database")
            continue
        print(f"DEBUG: Found case {case.id}")
        results.append({
            "case_id": case.id,
            "ks_id": f"KS-{case.id:06d}",
            "similarity_score": hit["score"],
            "date": case.judgment_date,
            "subject": case.subject,
            "acts_cited": case.acts_cited,
            "snippet": (case.anonymized_text or "")[:400] + "...",
            "has_document": bool(case.document_path),
        })

    print(f"DEBUG: Returning {len(results)} results")
    return {"results": results, "query": req.query}


# ─── Model Tester ─────────────────────────────────────────────────────────────

class ModelTestRequest(BaseModel):
    case_id: int
    model_ids: list[str] | None = None  # None = all models


@app.post("/api/models/test")
def test_models(req: ModelTestRequest, db: Session = Depends(get_db)):
    case = db.query(CaseLaw).filter(CaseLaw.id == req.case_id).first()
    if not case:
        raise HTTPException(404, "Case not found")
    if not case.anonymized_text:
        raise HTTPException(400, "Case has no anonymized text")

    models_to_run = MODELS
    if req.model_ids:
        models_to_run = [m for m in MODELS if m["id"] in req.model_ids]

    known_names = list((case.pii_map or {}).values())
    results = run_model_test(case.anonymized_text, known_names=known_names, models=models_to_run)

    saved = []
    for r in results:
        record = ModelTestResult(
            case_id=case.id,
            model_id=r["model_id"],
            model_label=r["model_label"],
            generated_summary=r["generated_summary"],
            score_completeness=r.get("score_completeness"),
            score_pii_safety=r.get("score_pii_safety"),
            score_readability=r.get("score_readability"),
            score_structure=r.get("score_structure"),
            score_legal_terms=r.get("score_legal_terms"),
            score_total=r.get("score_total"),
            latency_ms=r.get("latency_ms"),
            tokens_used=r.get("tokens_used"),
            error=r.get("error"),
        )
        db.add(record)
        saved.append(r)

    db.commit()
    return {"case_id": req.case_id, "results": saved}


@app.get("/api/models/results")
def list_model_results(case_id: int | None = None, db: Session = Depends(get_db)):
    q = db.query(ModelTestResult)
    if case_id:
        q = q.filter(ModelTestResult.case_id == case_id)
    records = q.order_by(ModelTestResult.created_at.desc()).limit(100).all()
    return [
        {
            "id": r.id,
            "case_id": r.case_id,
            "model_id": r.model_id,
            "model_label": r.model_label,
            "score_total": r.score_total,
            "score_completeness": r.score_completeness,
            "score_pii_safety": r.score_pii_safety,
            "score_readability": r.score_readability,
            "score_structure": r.score_structure,
            "score_legal_terms": r.score_legal_terms,
            "latency_ms": r.latency_ms,
            "tokens_used": r.tokens_used,
            "error": r.error,
            "generated_summary": r.generated_summary,
            "created_at": r.created_at.isoformat(),
        }
        for r in records
    ]


@app.get("/api/models/available")
def available_models():
    return {"models": MODELS}
