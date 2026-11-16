"""
KanoonSaathi — Demo Backend
FastAPI app exposing ingestion pipeline, judgement finder, and LLM model tester.
"""
from contextlib import asynccontextmanager
from pathlib import Path
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config import settings
from db.database import engine, get_db, Base
from db.models import CaseLaw, ModelTestResult, IngestionLog
from pipeline.ingestion import load_dataset, row_to_dict
from pipeline.anonymizer import anonymize
from pipeline.document_creator import create_document
from pipeline.pdf_extractor import extract_text_from_pdf, extract_metadata_via_llm, strip_thinking
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


def _safe_upload_path(filename: str, expected_ext: str) -> Path:
    """Store an upload under a random UUID name, keeping only the validated extension.
    Prevents path traversal / overwrite via attacker-controlled filenames (SEC-02)."""
    if not filename.lower().endswith(expected_ext):
        raise HTTPException(400, f"Only {expected_ext} files are accepted")
    data_dir = settings.data_path()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / f"{uuid.uuid4().hex}{expected_ext}"


def _process_row(row_dict: dict, db: Session) -> tuple[CaseLaw | None, str]:
    """Process one row atomically. Returns (case, error_message)."""
    text = row_dict.get("text", "")
    if len(text) < 50:
        return None, f"Text too short ({len(text)} chars)"

    # Create savepoint for this row so a mid-row failure rolls back cleanly
    sp = db.begin_nested()
    try:
        known_names = [
            n for n in [row_dict.get("petitioner"), row_dict.get("respondent"), row_dict.get("judges")]
            if n and len(n) > 2
        ]

        print(f"DEBUG: Starting anonymization for text length {len(text)}")
        anon = anonymize(text, known_names=known_names)
        print(f"DEBUG: Anonymization complete")

        # Placeholder names for the anonymized document (real names are never surfaced)
        anon_petitioner = "John Doe"
        anon_respondent = "Jane Doe"
        anon_judges = "Honourable Bench"

        case = CaseLaw(
            original_case_no=row_dict.get("case_no") or None,
            judgment_date=row_dict.get("date") or None,
            subject=row_dict.get("subject") or None,
            acts_cited=row_dict.get("acts") or None,
            raw_text=text[:50000],
            anonymized_text=anon.anonymized_text[:50000],
            pii_map=anon.pii_map,
            ingestion_status="success",
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
        return case, ""
    except Exception as e:
        # Rollback savepoint on failure
        sp.rollback()
        print(f"ERROR: Failed to process row: {str(e)}")
        import traceback
        traceback.print_exc()

        # Cleanup: remove document and metadata files if case was created
        if 'case' in locals() and case.id:
            try:
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

        return None, str(e)


def _generate_summary(anonymized_text: str, model_id: str = "openai/gpt-oss-120b") -> str:
    truncated = anonymized_text[:6000]
    user_msg = USER_PROMPT_TEMPLATE.format(anonymized_text=truncated)
    resp = chat(model_id, SYSTEM_PROMPT, user_msg, max_tokens=1500)
    if resp["error"] or not resp["text"]:
        resp = chat("openai/gpt-oss-20b", SYSTEM_PROMPT, user_msg, max_tokens=1500)
        if resp["error"] or not resp["text"]:
            # Qwen needs extra tokens to fit its <think> reasoning block
            resp = chat("qwen/qwen3.8-27b", SYSTEM_PROMPT, user_msg, max_tokens=3500)
    return strip_thinking(resp["text"] or "") or "[Summary generation failed]"


@app.post("/api/ingest/bulk", response_model=BulkIngestResponse)
async def bulk_ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    tmp_path = _safe_upload_path(file.filename, ".csv")
    tmp_path.write_bytes(await file.read())

    print(f"DEBUG: Loading CSV from {tmp_path}")
    pipeline_df, test_df = load_dataset(tmp_path)
    print(f"DEBUG: Pipeline rows: {len(pipeline_df)}, Test rows: {len(test_df)}")
    print(f"DEBUG: Pipeline columns: {list(pipeline_df.columns)}")

    ingested = 0
    failed = 0
    errors = []
    for idx, row in pipeline_df.iterrows():
        row_dict = row_to_dict(row)
        print(f"DEBUG: Processing row {idx}, text length: {len(row_dict.get('text', ''))}")
        case, error_msg = _process_row(row_dict, db)
        if case:
            ingested += 1
            print(f"DEBUG: Successfully ingested case {idx}")
        else:
            failed += 1
            errors.append(f"Row {idx}: {error_msg or 'Unknown error'}")
            print(f"DEBUG: Failed case {idx}: {error_msg}")

    status = "success" if failed == 0 else "partial" if ingested > 0 else "failed"
    db.add(IngestionLog(
        filename=file.filename,
        file_type="csv",
        status=status,
        cases_processed=ingested + failed,
        cases_failed=failed,
        error_message="\n".join(errors[:10]) if errors else None,
    ))
    db.commit()
    print(f"DEBUG: Final count - ingested: {ingested}, failed: {failed}")

    return BulkIngestResponse(
        ingested=ingested,
        test_set_size=len(test_df),
        skipped=failed,
        message=f"Ingested {ingested} cases into pipeline. {failed} failed. {len(test_df)} cases reserved for LLM testing.",
    )


class PdfIngestResponse(BaseModel):
    success: bool
    case_id: int | None
    message: str


@app.post("/api/ingest/pdf", response_model=PdfIngestResponse)
async def ingest_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    tmp_path = _safe_upload_path(file.filename, ".pdf")
    tmp_path.write_bytes(await file.read())

    print(f"DEBUG: Processing PDF from {tmp_path}")
    raw_text = extract_text_from_pdf(tmp_path)
    if not raw_text or len(raw_text) < 50:
        _log_ingestion(db, file.filename, "failed", "Could not extract sufficient text from PDF")
        db.commit()
        raise HTTPException(400, "Could not extract sufficient text from PDF")

    print(f"DEBUG: Extracted {len(raw_text)} chars; extracting metadata via LLM...")
    metadata = extract_metadata_via_llm(raw_text)
    if not metadata:
        _log_ingestion(db, file.filename, "failed", "Metadata extraction failed")
        db.commit()
        raise HTTPException(422, "Could not extract case metadata from PDF")

    row_dict = {"text": raw_text, **metadata}
    case, error_msg = _process_row(row_dict, db)
    if not case:
        _log_ingestion(db, file.filename, "failed", error_msg or "Processing failed")
        db.commit()
        return PdfIngestResponse(success=False, case_id=None, message=f"Failed to process case: {error_msg}")

    _log_ingestion(db, file.filename, "success", None)
    db.commit()
    return PdfIngestResponse(success=True, case_id=case.id, message="Successfully ingested PDF case.")


def _log_ingestion(db: Session, filename: str, status: str, error: str | None):
    db.add(IngestionLog(
        filename=filename, file_type="pdf", status=status,
        cases_processed=1, cases_failed=0 if status == "success" else 1,
        error_message=error,
    ))


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
            "ingestion_status": c.ingestion_status,
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
        "ingestion_status": case.ingestion_status,
        "ingestion_error": case.ingestion_error,
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


# ─── Ingestion Logs ───────────────────────────────────────────────────────────

@app.get("/api/ingestion/logs")
def list_ingestion_logs(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    logs = db.query(IngestionLog).order_by(IngestionLog.created_at.desc()).offset(skip).limit(limit).all()
    return [
        {
            "id": log.id,
            "filename": log.filename,
            "file_type": log.file_type,
            "status": log.status,
            "cases_processed": log.cases_processed,
            "cases_failed": log.cases_failed,
            "error_message": log.error_message,
            "created_at": log.created_at.isoformat(),
        }
        for log in logs
    ]
