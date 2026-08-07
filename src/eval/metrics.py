"""
Pure, dependency-free scoring functions for the eval harness. Kept separate
from run_eval.py so they're independently unit-testable without needing a
live Qdrant instance or an LLM call.
"""
import re
from typing import List

from src.agents.state import RetrievedChunk


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())


def keyword_recall(retrieved: List[RetrievedChunk], expected_keywords: List[str]) -> float:
    """
    Fraction of expected_keywords that appear somewhere across the retrieved
    chunks' text. A cheap, transparent proxy for retrieval relevance that
    doesn't require a labeled gold-passage set.
    """
    if not expected_keywords:
        return 1.0
    combined_text = _normalize(" ".join(c["text"] for c in retrieved))
    hits = sum(1 for kw in expected_keywords if _normalize(kw) in combined_text)
    return hits / len(expected_keywords)


def citation_validity_rate(cited_ids: List[str], retrieved: List[RetrievedChunk]) -> float:
    """1.0 if every cited source_id was actually retrieved, else the fraction that were."""
    if not cited_ids:
        return 0.0
    retrieved_ids = {c["source_id"] for c in retrieved}
    valid = sum(1 for cid in cited_ids if cid in retrieved_ids)
    return valid / len(cited_ids)


def has_citations(answer: str) -> bool:
    return bool(re.search(r"\[arxiv:[^\]]+\]", answer))
