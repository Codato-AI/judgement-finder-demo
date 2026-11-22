"""
PDF ingestion helpers: extract raw text (local PyMuPDF, with a LlamaParse escalation
for scanned/complex-table documents) and structured metadata via an LLM.
"""
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import ftfy
import pymupdf

from config import settings
from llm.groq_client import chat

logger = logging.getLogger(__name__)

_META_KEYS = {
    "case_no": ["case_no", "case_number", "caseno"],
    "date": ["date", "date_of_judgment", "judgment_date", "decision_date"],
    "petitioner": ["petitioner", "appellant", "plaintiff"],
    "respondent": ["respondent", "defendant"],
    "judges": ["judges", "judge", "bench", "coram"],
    "subject": ["subject", "category", "topic", "headnote", "head_note"],
    "acts": ["acts", "act", "acts_cited", "statutes"],
    "verdict": ["verdict", "decision", "outcome", "holding"],
}

# Tuned by eyeballing the actual corpus, not formally validated — revisit
# if the dry-run sample shows misclassified documents.
_SCANNED_CHARS_PER_PAGE = 20        # below this, a page is "no usable text layer"
_SCANNED_IMAGE_AREA_RATIO = 0.6     # page mostly covered by one/few images
_SCANNED_PAGE_FRACTION = 0.5        # majority of pages must look scanned
_TABULAR_GAP_RE = re.compile(r"\S(?: {2,}|\t)\S")   # 2+ space / tab gap between tokens
_TABULAR_MIN_LINES = 3              # consecutive aligned lines to call it a table

# A run of the same unusual control char is a font/glyph-mapping failure baked into
# the source PDF (missing ToUnicode CMap entries show up as repeated "notdef" bytes) —
# re-extraction can't fix this, the glyphs were never there to read. Mark it instead of
# silently dropping it, so a lawyer sees that something was omitted, not a typo.
_GARBLED_RUN_RE = re.compile(r"([\x00-\x08\x0b\x0e-\x1f\x7f-\x9f])\1{3,}")


def clean_extracted_text(text: str) -> str:
    """Repair encoding glitches (mojibake, stray HTML entities, curly quotes) via ftfy,
    and mark — rather than silently delete — long runs of repeated garbage control
    characters left behind by a source PDF's broken font."""
    text = _GARBLED_RUN_RE.sub("[illegible in source]", text)
    return ftfy.fix_text(text)


@dataclass
class ExtractionResult:
    text: str
    method: str   # "local" | "llamaparse"
    reason: str


def _page_image_area_ratio(page: "pymupdf.Page") -> float:
    """Fraction of the page covered by embedded images (0 if none)."""
    rect_area = page.rect.width * page.rect.height
    if not rect_area:
        return 0.0
    covered = 0.0
    for img in page.get_images(full=True):
        try:
            covered += page.get_image_bbox(img).get_area()
        except ValueError:
            continue
    return covered / rect_area


def _is_scanned(doc: "pymupdf.Document", pages: list[str]) -> bool:
    """True if most pages have no usable text layer but do carry a near-full-page image —
    i.e. this is a scan, not a born-digital PDF."""
    if not pages:
        return True
    scanned_pages = 0
    for i, text in enumerate(pages):
        if len(text.strip()) >= _SCANNED_CHARS_PER_PAGE:
            continue
        if _page_image_area_ratio(doc[i]) >= _SCANNED_IMAGE_AREA_RATIO:
            scanned_pages += 1
    return scanned_pages / len(pages) >= _SCANNED_PAGE_FRACTION


def _table_is_ragged(rows: list[list]) -> bool:
    """A cleanly-parsed grid has a consistent cell count per row and few empty cells.
    Inconsistent row lengths or lots of Nones means the gridlines/merged cells confused
    the parser — treat it as too complex for local extraction to trust."""
    if len(rows) < 2:
        return False
    lengths = {len(r) for r in rows}
    if len(lengths) > 1:
        return True
    total_cells = sum(len(r) for r in rows)
    empty_cells = sum(1 for r in rows for c in r if not c or not str(c).strip())
    return total_cells > 0 and (empty_cells / total_cells) > 0.4


def _looks_tabular_untabled(raw_text: str) -> bool:
    """Several consecutive lines with multiple wide whitespace gaps between tokens look
    like column-aligned content that find_tables() should have caught but didn't —
    a sign the real structure is more complex (merged cells, rotated text, etc)."""
    run = 0
    for line in raw_text.split("\n"):
        if len(_TABULAR_GAP_RE.findall(line)) >= 2:
            run += 1
            if run >= _TABULAR_MIN_LINES:
                return True
        else:
            run = 0
    return False


def _complex_table_reason(doc: "pymupdf.Document", pages: list[str]) -> str:
    """Returns a truthy reason string if any page shows a table too complex to trust to
    local extraction, else "". Clean, well-formed tables are left alone (local is fine)."""
    prev_table_cols: int | None = None
    for i, raw_text in enumerate(pages):
        tables = doc[i].find_tables().tables
        if tables:
            for t in tables:
                rows = t.extract()
                if _table_is_ragged(rows):
                    return f"ragged table on page {i + 1}"
                if prev_table_cols is not None and rows and len(rows[0]) == prev_table_cols:
                    return f"table appears to span page break at page {i + 1}"
                prev_table_cols = len(rows[0]) if rows else None
        else:
            prev_table_cols = None
            if _looks_tabular_untabled(raw_text):
                return f"tabular-looking text undetected as a table on page {i + 1}"
    return ""


def _via_llamaparse(file_path: str | Path, pages: list[str], reason: str) -> ExtractionResult:
    """Escalate the whole document to LlamaParse. Falls back to the local text already
    extracted if LlamaParse isn't configured or fails — a bad PDF or an API outage must
    never abort ingestion."""
    if not settings.LLAMA_CLOUD_API_KEY:
        logger.warning("LlamaParse needed (%s) but no API key configured — using local text", reason)
        return ExtractionResult("\n".join(pages), "local", f"{reason} (llamaparse not configured)")

    try:
        from llama_cloud_services import LlamaParse
        from llama_cloud_services.parse.utils import ResultType

        parser = LlamaParse(
            api_key=settings.LLAMA_CLOUD_API_KEY,
            result_type=ResultType.MD,
            system_prompt_append="Extract all text and preserve any tables as markdown tables.",
            max_timeout=600,
        )
        # parser.parse() returns a JobResult backed by the API's actual per-page JSON
        # (job_result.pages[i].page) — unlike load_data(), whose "page" metadata is
        # reconstructed by re-splitting one joined string and is never populated.
        job_result = parser.parse(str(file_path))
        documents = job_result.get_markdown_documents(split_by_page=True)
        by_page: dict[int, list[str]] = {}
        for d in documents:
            try:
                page_num = int(d.metadata.get("page_number", 1))
            except (TypeError, ValueError):
                page_num = 1
            by_page.setdefault(page_num, []).append(d.text)
        ordered = ["\n\n".join(by_page[p]) for p in sorted(by_page)]
        text = "\n".join(ordered)
        if not text.strip():
            raise ValueError("LlamaParse returned no text")
        return ExtractionResult(text, "llamaparse", reason)
    except Exception:
        logger.exception("LlamaParse failed (%s) — falling back to local text", reason)
        return ExtractionResult("\n".join(pages), "local", f"{reason} (llamaparse failed, used local)")


def extract_pdf(file_path: str | Path) -> ExtractionResult:
    """Extract text from a PDF: PyMuPDF locally for the clean majority, escalating the
    whole document to LlamaParse for scans or tables too complex to trust locally."""
    doc = pymupdf.open(file_path)
    try:
        pages = [page.get_text() for page in doc]
        if _is_scanned(doc, pages):
            result = _via_llamaparse(file_path, pages, "scanned (no usable text layer)")
        else:
            reason = _complex_table_reason(doc, pages)
            if reason:
                result = _via_llamaparse(file_path, pages, reason)
            else:
                result = ExtractionResult("\n".join(pages), "local", "clean text layer")
    finally:
        doc.close()
    result.text = clean_extracted_text(result.text)
    return result


def extract_text_from_pdf(file_path: str | Path) -> str:
    """Extract all text from a PDF file."""
    return extract_pdf(file_path).text


def strip_thinking(text: str) -> str:
    """Strip reasoning/thought blocks like <think>...</think> from model output."""
    if "<think>" in text:
        parts = text.split("</think>")
        if len(parts) > 1:
            return parts[-1].strip()
        return text.split("<think>")[0].strip()
    return text


def _unfence_json(text: str) -> str:
    """Remove a leading ```json / ``` code fence if the model wrapped its output."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):]
    elif text.startswith("```"):
        text = text[len("```"):]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _normalize_metadata(data: dict) -> dict:
    """Map LLM-returned keys (which may use synonyms/casing) to our canonical keys."""
    normalized = {}
    for target, aliases in _META_KEYS.items():
        val = next((data[a] for a in aliases if a in data), None)
        if val is None:
            for k, v in data.items():
                if k.lower().replace("_", "").replace(" ", "") == target:
                    val = v
                    break
        if isinstance(val, list):
            val = ", ".join(str(item) for item in val)
        normalized[target] = val
    return normalized


def extract_metadata_via_llm(raw_text: str) -> dict:
    """Extract structured case metadata from judgment text using an LLM. {} on failure."""
    truncated = raw_text[:6000]  # keep within context window / avoid 413
    prompt = f"""You are a legal metadata extraction AI. Extract the following fields from the \
provided judgment text and return them as a valid JSON object. Do not return any other text, only the JSON.

Keys required:
- "case_no": (string or null) The official case number.
- "date": (string or null) The date of the judgment (e.g. '19 May, 1950').
- "petitioner": (string or null) The name of the petitioner / appellant.
- "respondent": (string or null) The name of the respondent / defendant.
- "judges": (string or null) The name(s) of the judges.
- "subject": (string or null) A brief 1-5 word category or subject of the case.
- "acts": (string or null) Any specific acts or statutes mentioned.
- "verdict": (string or null) A short summary of the outcome / verdict.

Judgment Text:
{truncated}
"""
    resp = chat("openai/gpt-oss-20b", "You are a helpful JSON extraction assistant.", prompt,
                max_tokens=1024, temperature=0.1)
    if resp["error"] or not resp["text"]:
        resp = chat("qwen/qwen3.8-27b", "You are a helpful JSON extraction assistant.", prompt,
                    max_tokens=1536, temperature=0.1)
        if resp["error"] or not resp["text"]:
            return {}

    try:
        data = json.loads(_unfence_json(strip_thinking(resp["text"])))
    except json.JSONDecodeError:
        logger.warning("Failed to parse metadata JSON from LLM. Raw output: %s", resp["text"])
        return {}
    return _normalize_metadata(data)


if __name__ == "__main__":
    # Self-check: parsing/normalization without hitting the LLM or network.
    assert strip_thinking("<think>reasoning</think>final") == "final"
    assert _unfence_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _normalize_metadata({"appellant": "A", "bench": ["J1", "J2"]}) == {
        "case_no": None, "date": None, "petitioner": "A", "respondent": None,
        "judges": "J1, J2", "subject": None, "acts": None, "verdict": None,
    }

    # Table-complexity detection
    assert _table_is_ragged([["a", "b"], ["c", "d", "e"]]) is True  # inconsistent row lengths
    assert _table_is_ragged([["", "", ""], [None, "", ""]]) is True  # mostly empty cells
    assert _table_is_ragged([["a", "b"], ["c", "d"]]) is False  # tidy grid

    assert _looks_tabular_untabled(
        "Name      Amount    Date\n"
        "Ravi       5000      01-01-2020\n"
        "Priya      7000      02-01-2020\n"
    ) is True
    assert _looks_tabular_untabled(
        "The appellant submitted that the impugned order was passed without\n"
        "jurisdiction and ought to be set aside in the interests of justice.\n"
    ) is False

    # Scanned-page detection, against real in-memory PDFs (no network, no sample files)
    scanned_doc = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 100), False)
    pix.set_rect(pix.irect, (255, 255, 255))
    page = scanned_doc.new_page()
    page.insert_image(page.rect, pixmap=pix)
    assert _is_scanned(scanned_doc, [p.get_text() for p in scanned_doc]) is True

    text_doc = pymupdf.open()
    page = text_doc.new_page()
    page.insert_text((50, 50), "This is a perfectly ordinary born-digital judgment page " * 10)
    assert _is_scanned(text_doc, [p.get_text() for p in text_doc]) is False

    # Garbled-text cleanup: mark broken-font control-char runs, don't touch real content
    assert clean_extracted_text(
        "[Provided \x05\x05\x05\x05\x05\x05\x05\x05.\nProvided further\x05\x05\x05\x05]"
    ) == "[Provided [illegible in source].\nProvided further[illegible in source]]"
    assert clean_extracted_text("ordinary text with no issues at all") == \
        "ordinary text with no issues at all"
    assert clean_extracted_text("E&#x26;D") == "E&D"  # stray HTML entity (ftfy)
    assert "\x05\x05\x05" not in clean_extracted_text("A" + "\x05" * 10 + "B")
    # a lone stray control char (not a run) isn't our [illegible] marker — ftfy
    # still quietly drops it as part of its own normal cleanup, which is fine
    assert "[illegible in source]" not in clean_extracted_text("normal\x05text")

    print("pdf_extractor self-check passed")
