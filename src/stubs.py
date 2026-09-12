"""
Test doubles that let the REAL agent graph run end-to-end with no API key and
no network — the LLM and the embedding model are the only two things in this
project that need either, so stubbing exactly those two leaves everything else
(LangGraph routing, the critic's structural check, the retry loop, Qdrant
search, citation parsing) running as real production code.
"""
import hashlib
import json
import re
from types import SimpleNamespace
from typing import List

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config import settings

FAKE_DIM = 64


def fake_embed_texts(texts: List[str]) -> np.ndarray:
    """Deterministic hashed bag-of-words stand-in for a real embedding model.
    Not semantically meaningful, but stable and dimension-correct, which is all
    the retrieval plumbing needs to be exercised."""
    vectors = []
    for text in texts:
        vec = np.zeros(FAKE_DIM)
        for word in re.findall(r"[a-z]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % FAKE_DIM] += 1.0
        norm = np.linalg.norm(vec)
        vectors.append(vec / norm if norm > 0 else vec)
    return np.array(vectors)


class FakeLLM:
    """Routes on prompt content so one object can stand in for all three agents.
    Each agent's prompt has a distinctive opening line, so we dispatch on that
    rather than requiring the test to know call ordering."""

    def __init__(self, drafts: List[str], verdicts: List[dict], query_rewrites: List[str] | None = None):
        self.drafts = list(drafts)
        self.verdicts = list(verdicts)
        self.query_rewrites = list(query_rewrites or [])
        self.calls = {"rewrite": 0, "synthesis": 0, "critic": 0}
        self._last_draft = ""

    def invoke(self, prompt: str):
        if "Rewrite the following question" in prompt:
            self.calls["rewrite"] += 1
            text = self.query_rewrites.pop(0) if self.query_rewrites else "retrieval augmented generation"
        elif "research assistant" in prompt:
            self.calls["synthesis"] += 1
            if self.drafts:
                self._last_draft = self.drafts.pop(0)
            text = self._last_draft
        elif "fact-checker" in prompt:
            self.calls["critic"] += 1
            verdict = (
                self.verdicts.pop(0)
                if self.verdicts
                else {"faithful": True, "issues": "", "sufficient_context": True}
            )
            text = json.dumps(verdict)
        else:
            raise AssertionError(f"FakeLLM received an unrecognised prompt: {prompt[:120]!r}")
        return SimpleNamespace(content=text)


def build_populated_client(docs: dict[str, str]) -> QdrantClient:
    """docs: {source_id: text}. Returns an in-memory Qdrant with one chunk per doc."""
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=FAKE_DIM, distance=Distance.COSINE),
    )
    vectors = fake_embed_texts(list(docs.values()))
    client.upsert(
        collection_name=settings.qdrant_collection,
        points=[
            PointStruct(
                id=i + 1,
                vector=vectors[i].tolist(),
                payload={
                    "text": text,
                    "source_id": source_id,
                    "source_title": f"Paper {source_id}",
                    "chunk_index": 0,
                },
            )
            for i, (source_id, text) in enumerate(docs.items())
        ],
    )
    return client


CORPUS = {
    "2401.00001": (
        "Retrieval augmented generation combines a parametric language model with a "
        "retriever over an external knowledge source, grounding generated answers in "
        "retrieved evidence and reducing hallucination."
    ),
    "2402.00002": (
        "A vector database stores dense embeddings and supports approximate nearest "
        "neighbour search using indexes such as HNSW."
    ),
}
