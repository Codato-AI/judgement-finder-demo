"""
PDF ingestion helpers: extract raw text via pdfplumber and structured metadata via an LLM.
"""
import json
from pathlib import Path

import pdfplumber

from llm.groq_client import chat

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


def extract_text_from_pdf(file_path: str | Path) -> str:
    """Extract all text from a PDF file."""
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
    return "\n".join(pages)


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
    except json.JSONDecodeError as e:
        print(f"DEBUG: Failed to parse JSON from LLM: {e}\nRaw output: {resp['text']}")
        return {}
    return _normalize_metadata(data)


if __name__ == "__main__":
    # Self-check: parsing/normalization without hitting the LLM.
    assert strip_thinking("<think>reasoning</think>final") == "final"
    assert _unfence_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _normalize_metadata({"appellant": "A", "bench": ["J1", "J2"]}) == {
        "case_no": None, "date": None, "petitioner": "A", "respondent": None,
        "judges": "J1, J2", "subject": None, "acts": None, "verdict": None,
    }
    print("pdf_extractor self-check passed")
