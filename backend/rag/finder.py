"""Judgement Finder: embed user query, retrieve top-N similar cases."""
from rag.embedder import embed_one
from rag.vector_store import search


def find_similar(query: str, top_k: int = 5) -> list[dict]:
    """
    Embed the query and return top_k similar case IDs with similarity scores.
    The caller resolves case_ids to full records from the DB.
    """
    vec = embed_one(query)
    return search(vec, top_k=top_k)
