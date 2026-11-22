"""Grounded answer generation: take the fused chunk hits from the finder, group them
per case in rank order, and have the LLM write a short answer that cites each claim back
to the exact retrieved passage ([#N]). Citations are passage-level, not paragraph/page -
the schema has no paragraph provenance, so we anchor to the verbatim chunk text instead.
Source #N here lines up with result card #N on the frontend because both consume the same
fused rank order from retrieve_chunks.
"""
from sqlalchemy.orm import Session

from db.models import CaseLaw
from rag.finder import YEAR_RE
from llm.groq_client import chat
from llm.prompts import ANSWER_SYSTEM_PROMPT, ANSWER_USER_TEMPLATE
from pipeline.pdf_extractor import strip_thinking

MAX_CASES = 5            # cap grounding to the top cases shown on the cards
MAX_CHUNKS_PER_CASE = 2  # a couple of passages per case is enough context
PRIMARY_MODEL = "openai/gpt-oss-120b"


def _label(case: CaseLaw | None) -> str:
    if case is None:
        return "Supreme Court judgment"
    subject = case.subject or "Supreme Court judgment"
    extra = []
    if case.original_case_no:
        extra.append(case.original_case_no)
    year = YEAR_RE.search(case.judgment_date or "")
    if year:
        extra.append(year.group())
    return f"{subject} ({', '.join(extra)})" if extra else subject


def build_prompt(query: str, chunks: list[dict], db: Session) -> dict | None:
    """Group fused chunks into per-case sources (case #1 = top-ranked), build the LLM
    user prompt and the sources payload. Returns None if there's nothing to ground on.
    Does all DB work up front so the caller can run the LLM call off the event loop."""
    order: list[int] = []               # case_ids in rank order
    passages: dict[int, list[str]] = {}
    for c in chunks:
        cid = c["case_id"]
        if cid not in passages:
            if len(order) >= MAX_CASES:
                continue
            order.append(cid)
            passages[cid] = []
        if len(passages[cid]) < MAX_CHUNKS_PER_CASE:
            passages[cid].append(c["chunk_text"].strip())
    if not order:
        return None

    cases = {r.id: r for r in db.query(CaseLaw).filter(CaseLaw.id.in_(order)).all()}
    blocks, sources = [], []
    for n, cid in enumerate(order, start=1):
        label = _label(cases.get(cid))
        quoted = "\n".join(f'"{p}"' for p in passages[cid])
        blocks.append(f"[#{n}] {label}:\n{quoted}")
        sources.append({
            "n": n,
            "case_id": cid,
            "label": label,
            "passage": passages[cid][0],  # representative passage shown in the UI
        })

    user = ANSWER_USER_TEMPLATE.format(query=query, passages="\n\n".join(blocks))
    return {"user": user, "sources": sources}


def run_llm(user: str, model_id: str = PRIMARY_MODEL) -> str:
    """Generate the answer text, same fallback cascade as summary generation."""
    resp = chat(model_id, ANSWER_SYSTEM_PROMPT, user, max_tokens=1200)
    if resp["error"] or not resp["text"]:
        resp = chat("openai/gpt-oss-20b", ANSWER_SYSTEM_PROMPT, user, max_tokens=1200)
        if resp["error"] or not resp["text"]:
            # Qwen needs extra tokens to fit its <think> reasoning block
            resp = chat("qwen/qwen3.8-27b", ANSWER_SYSTEM_PROMPT, user, max_tokens=3000)
    return strip_thinking(resp["text"] or "").strip()
