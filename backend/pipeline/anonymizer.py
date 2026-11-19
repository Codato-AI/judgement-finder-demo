"""
PII anonymization layer.
Replaces names, phone numbers, Aadhaar, PAN, emails, and addresses
with neutral placeholders. Keeps a reversible map for internal use.
"""
import re
import spacy
from dataclasses import dataclass, field

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            # Only the NER component is used below — tagger/parser/lemmatizer
            # are dead weight for this pipeline, disabling them cuts memory
            # and time per document for free.
            _nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer", "attribute_ruler"])
        except OSError:
            # Fallback: blank model with no NER — regex still runs
            _nlp = spacy.blank("en")
    return _nlp


# Regex patterns for Indian-specific PII
_PATTERNS = {
    "AADHAAR": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "PAN": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    "PHONE": re.compile(r"(\+91[\-\s]?)?(0?[6-9]\d{9})\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"),
    "PINCODE": re.compile(r"\b[1-9][0-9]{5}\b"),
}

_PERSON_COUNTER = [0]
_ORG_COUNTER = [0]
_LOC_COUNTER = [0]


def _person_placeholder(idx: int) -> str:
    names = [
        "John Doe", "Jane Doe", "Person A", "Person B", "Person C",
        "Person D", "Person E", "Person F", "Person G", "Person H",
    ]
    return names[idx % len(names)]


@dataclass
class AnonymizationResult:
    anonymized_text: str
    pii_map: dict = field(default_factory=dict)  # placeholder -> original


def anonymize(
    text: str,
    known_names: list[str] | None = None,
) -> AnonymizationResult:
    """
    Anonymize PII in text. known_names are names extracted from structured
    columns (petitioner, respondent, judges) that get replaced first.
    """
    if not text:
        return AnonymizationResult(anonymized_text="", pii_map={})

    pii_map: dict[str, str] = {}
    person_idx = 0

    # 1. Replace known structured names first (most reliable)
    if known_names:
        for name in known_names:
            name = name.strip()
            if not name or len(name) < 3:
                continue
            placeholder = _person_placeholder(person_idx)
            person_idx += 1
            pii_map[placeholder] = name
            # Case-insensitive replacement
            text = re.sub(re.escape(name), placeholder, text, flags=re.IGNORECASE)

    # 2. spaCy NER for remaining names in text
    nlp = _get_nlp()
    # nlp.max_length (spaCy's own memory-safety ceiling, 1M chars by default)
    # bounds a single document's NER memory use — real judgments never get
    # close to it, this just stops a pathological upload from blowing up.
    doc = nlp(text[:nlp.max_length])

    ner_replacements: list[tuple[int, int, str]] = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            placeholder = _person_placeholder(person_idx)
            person_idx += 1
            pii_map[placeholder] = ent.text
            ner_replacements.append((ent.start_char, ent.end_char, placeholder))

    # Apply NER replacements in reverse order to preserve offsets
    for start, end, placeholder in sorted(ner_replacements, key=lambda x: x[0], reverse=True):
        text = text[:start] + placeholder + text[end:]

    # 3. Regex-based PII (Aadhaar, PAN, phone, email, pincode)
    text = _PATTERNS["AADHAAR"].sub("[AADHAAR]", text)
    text = _PATTERNS["PAN"].sub("[PAN]", text)
    text = _PATTERNS["PHONE"].sub("[PHONE]", text)
    text = _PATTERNS["EMAIL"].sub("[EMAIL]", text)
    text = _PATTERNS["PINCODE"].sub("[PINCODE]", text)

    return AnonymizationResult(anonymized_text=text, pii_map=pii_map)
