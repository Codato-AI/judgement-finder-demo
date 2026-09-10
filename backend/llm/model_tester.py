"""
Runs anonymized case text through multiple Groq-hosted LLMs and scores each output.
Models tested: Kimi K2, Llama 3.3 70B, Gemma 2 9B, DeepSeek R1, Qwen QwQ 32B, Mixtral 8x7B.
"""
import re
from llm.groq_client import chat
from llm.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

MODELS = [
    {"id": "qwen/qwen3.6-27b", "label": "Qwen 3.6 27B (Alibaba)"},
    {"id": "openai/gpt-oss-20b", "label": "GPT OSS 20B (OpenAI)"},
    {"id": "openai/gpt-oss-120b", "label": "GPT OSS 120B (OpenAI)"},
    {"id": "groq/compound", "label": "Compound (Groq)"},
    {"id": "groq/compound-mini", "label": "Compound Mini (Groq)"},
]

_LEGAL_TERMS = [
    "petitioner", "respondent", "appellant", "bench", "ratio decidendi",
    "obiter dictum", "writ", "constitution", "article", "section",
    "judgment", "order", "appeal", "supreme court", "high court",
    "fundamental right", "natural justice", "prima facie", "locus standi",
    "ultra vires", "jurisdiction", "adjudication", "precedent",
]

_REQUIRED_HEADINGS = [
    "Summary of Facts",
    "Legal Issues",
    "Court's Analysis",
    "Ruling",
    "Key Legal Principles",
    "Statutes",
]

_PII_PATTERNS = [
    re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),           # Aadhaar
    re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),            # PAN
    re.compile(r"(\+91[\-\s]?)?(0?[6-9]\d{9})\b"),       # Phone
    re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"),    # Email
]


def _score_completeness(text: str) -> float:
    found = sum(1 for h in _REQUIRED_HEADINGS if h.lower() in text.lower())
    return round(found / len(_REQUIRED_HEADINGS) * 10, 2)


def _score_pii_safety(text: str, known_names: list[str]) -> float:
    score = 10.0
    for pattern in _PII_PATTERNS:
        if pattern.search(text):
            score -= 2.0
    for name in known_names:
        if name and len(name) > 3 and name.lower() in text.lower():
            score -= 1.5
    return max(round(score, 2), 0.0)


def _score_readability(text: str) -> float:
    sentences = re.split(r"[.!?]", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if not sentences:
        return 0.0
    avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
    # Ideal sentence length: 15–25 words
    if 15 <= avg_words <= 25:
        return 10.0
    elif 10 <= avg_words < 15 or 25 < avg_words <= 35:
        return 7.0
    elif avg_words < 10:
        return 5.0
    else:
        return 4.0


def _score_structure(text: str) -> float:
    # Checks for markdown headings and bullet lists
    headings = len(re.findall(r"^##?\s", text, re.MULTILINE))
    bullets = len(re.findall(r"^[-*]\s", text, re.MULTILINE))
    score = 0.0
    score += min(headings * 1.5, 6.0)
    score += min(bullets * 0.5, 4.0)
    return round(min(score, 10.0), 2)


def _score_legal_terms(text: str) -> float:
    text_lower = text.lower()
    found = sum(1 for t in _LEGAL_TERMS if t in text_lower)
    return round(min(found / 6 * 10, 10.0), 2)


def score_output(text: str, known_names: list[str] | None = None) -> dict:
    known_names = known_names or []
    completeness = _score_completeness(text)
    pii_safety = _score_pii_safety(text, known_names)
    readability = _score_readability(text)
    structure = _score_structure(text)
    legal_terms = _score_legal_terms(text)
    total = round(
        completeness * 0.30
        + pii_safety * 0.25
        + readability * 0.15
        + structure * 0.15
        + legal_terms * 0.15,
        2,
    )
    return {
        "completeness": completeness,
        "pii_safety": pii_safety,
        "readability": readability,
        "structure": structure,
        "legal_terms": legal_terms,
        "total": total,
    }


def run_model_test(
    anonymized_text: str,
    known_names: list[str] | None = None,
    models: list[dict] | None = None,
) -> list[dict]:
    """
    Run all models on the given anonymized text, score each, return results list.
    """
    models = models or MODELS
    known_names = known_names or []
    # Use first 6000 chars to stay within context limits for smaller models
    truncated = anonymized_text[:6000]
    user_msg = USER_PROMPT_TEMPLATE.format(anonymized_text=truncated)

    results = []
    for model in models:
        resp = chat(model["id"], SYSTEM_PROMPT, user_msg, max_tokens=1200)
        if resp["error"]:
            scores = {k: None for k in ["completeness", "pii_safety", "readability", "structure", "legal_terms", "total"]}
        else:
            scores = score_output(resp["text"], known_names)

        results.append({
            "model_id": model["id"],
            "model_label": model["label"],
            "generated_summary": resp["text"],
            "latency_ms": resp["latency_ms"],
            "tokens_used": resp["tokens_used"],
            "error": resp["error"],
            **{f"score_{k}": v for k, v in scores.items()},
        })

    return results
