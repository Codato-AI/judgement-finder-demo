"""
CSV ingestion layer. Handles flexible column naming from the SC judgments dataset.
Splits dataset: 80% for pipeline/RAG corpus, 20% for LLM model testing.
"""
import pandas as pd
from pathlib import Path

# Flexible column name mappings — tries each alias in order
_COL_MAP = {
    "case_no": ["case_no", "case_number", "case_id", "id", "caseno", "sl_no", "sno"],
    "date": ["date", "judgment_date", "date_of_judgment", "decision_date", "year"],
    "petitioner": ["petitioner", "appellant", "plaintiff", "party1", "party_1", "petitioner_name"],
    "respondent": ["respondent", "defendant", "party2", "party_2", "respondent_name"],
    "judges": ["judge", "judges", "bench", "coram", "judge_name", "justice"],
    "text": ["text", "judgment", "judgment_text", "full_text", "content", "body", "judgement", "judgement_text"],
    "subject": ["subject", "category", "topic", "area_of_law", "subject_category", "head_note", "headnote"],
    "acts": ["acts", "acts_cited", "statutes", "legislation", "act_cited", "laws"],
    "citation": ["citation", "cite", "neutral_citation", "citation_no"],
    "verdict": ["verdict", "decision", "outcome", "disposal", "result"],
}


def _resolve_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for alias in aliases:
        if alias in cols_lower:
            return cols_lower[alias]
    return None


def _safe_str(val) -> str:
    if pd.isna(val):
        return ""
    return str(val).strip()


def load_dataset(csv_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CSV, normalise columns, split 80/20.
    Returns (pipeline_df, test_df).
    """
    # Try different CSV reading strategies to handle malformed files
    try:
        df = pd.read_csv(csv_path, low_memory=False, on_bad_lines='skip')
    except TypeError:
        # Fallback for older pandas versions
        df = pd.read_csv(csv_path, low_memory=False, error_bad_lines=False, warn_bad_lines=True)
    except Exception as e:
        # If that fails, try with more permissive settings
        try:
            df = pd.read_csv(csv_path, low_memory=False, engine='python', on_bad_lines='skip')
        except TypeError:
            df = pd.read_csv(csv_path, low_memory=False, engine='python', error_bad_lines=False, warn_bad_lines=True)
    df = df.dropna(how="all").reset_index(drop=True)

    # Build normalised dataframe
    normed: dict[str, list] = {k: [] for k in _COL_MAP}

    resolved = {key: _resolve_col(df, aliases) for key, aliases in _COL_MAP.items()}

    for _, row in df.iterrows():
        for key, col in resolved.items():
            normed[key].append(_safe_str(row[col]) if col else "")

    normed_df = pd.DataFrame(normed)
    normed_df = normed_df[normed_df["text"].str.len() > 50].reset_index(drop=True)

    n = len(normed_df)
    split = max(1, int(n * 0.8))  # always keep at least 1 row in pipeline
    pipeline_df = normed_df.iloc[:split].reset_index(drop=True)
    test_df = normed_df.iloc[split:].reset_index(drop=True)

    return pipeline_df, test_df


def row_to_dict(row: pd.Series) -> dict:
    return {k: row[k] for k in _COL_MAP}
