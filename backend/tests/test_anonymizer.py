"""Tests for PII anonymization, focused on the ALL-CAPS NER blind spot."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.anonymizer import anonymize, _truecase_shouting


def test_truecase_shouting_preserves_length_and_non_run_text():
    text = "PETITIONER:\nRAJ KALI KUER\n\nVs.\n\nRESPONDENT:\nRAM RATTAN PANDEY\n"
    view = _truecase_shouting(text)
    assert len(view) == len(text)
    assert "Raj Kali Kuer" in view
    assert "Ram Rattan Pandey" in view
    assert "Vs." in view  # single words / non-runs untouched


def test_all_caps_name_not_covered_by_known_names_gets_redacted():
    # Before the fix, spaCy's NER (casing-sensitive) never saw this name at all.
    text = "Counsel for the appellant, ANIL KUMAR VERMA, argued that the lower court erred."
    result = anonymize(text)
    assert "ANIL KUMAR VERMA" not in result.anonymized_text
    assert "Anil Kumar Verma" in result.pii_map.values()


def test_known_names_still_take_priority_over_ner():
    text = "PETITIONER:\nRAJ KALI KUER"
    result = anonymize(text, known_names=["RAJ KALI KUER"])
    assert "RAJ KALI KUER" not in result.anonymized_text
    assert result.pii_map["John Doe"] == "RAJ KALI KUER"


if __name__ == "__main__":
    test_truecase_shouting_preserves_length_and_non_run_text()
    test_all_caps_name_not_covered_by_known_names_gets_redacted()
    test_known_names_still_take_priority_over_ner()
    print("anonymizer self-check passed")
