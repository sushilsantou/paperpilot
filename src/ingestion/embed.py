"""
Local embedding model wrapper — sentence-transformers, no API key or network
call at inference time beyond the one-time model download. Keeping this
swappable behind a function (not a hardcoded model) so the eval harness can
compare embedding models later without touching ingestion/agent code.
"""
from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 384))
    model = get_embedder()
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


def embedding_dim() -> int:
    return get_embedder().get_sentence_embedding_dimension()
