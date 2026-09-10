"""Lazy-loaded sentence-transformer embedder."""
from sentence_transformers import SentenceTransformer
import numpy as np
from config import settings

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBED_MODEL)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    model = get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]
