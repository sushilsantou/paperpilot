"""
Retriever agent: rewrites the query for search, pulls a wide candidate set
from Qdrant (vector similarity), then re-ranks with a keyword-overlap score
against the rewritten query — a lightweight hybrid retrieval that doesn't
require a second search backend (e.g. Elasticsearch/BM25 service).

On a critic-triggered retry, this node receives the critic's feedback and
uses it to rewrite the query again (e.g. broaden terms, target a different
aspect of the question) rather than repeating the same search.
"""
import re
from collections import Counter
from typing import List

from qdrant_client import QdrantClient

from src.agents.llm import get_llm, invoke_with_retry, message_text
from src.agents.state import AgentState, RetrievedChunk
from src.config import settings
from src.ingestion.embed import embed_texts

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
    "for", "and", "or", "what", "how", "why", "does", "do", "did", "with",
    "that", "this", "it", "as", "by", "be", "have", "has", "can", "which",
}


def _keywords(text: str) -> Counter:
    words = re.findall(r"[a-z0-9\-]+", text.lower())
    return Counter(w for w in words if w not in _STOPWORDS and len(w) > 2)


def _keyword_overlap_score(query_kw: Counter, chunk_text: str) -> float:
    chunk_kw = _keywords(chunk_text)
    if not query_kw:
        return 0.0
    overlap = sum(min(count, chunk_kw.get(term, 0)) for term, count in query_kw.items())
    return overlap / sum(query_kw.values())


def rewrite_query(question: str, feedback: str = "") -> str:
    """Uses the LLM to turn a conversational question into a focused search query."""
    llm = get_llm(temperature=0.0)
    prompt = (
        "Rewrite the following question into a short, focused search query "
        "(keywords and key phrases, not a full sentence) suitable for retrieving "
        "relevant passages from a corpus of machine learning research papers.\n\n"
        f"Question: {question}\n"
    )
    if feedback:
        prompt += (
            f"\nA previous search using a similar query was judged insufficient because: "
            f"{feedback}\nAdjust the query to address this gap.\n"
        )
    prompt += "\nRespond with ONLY the search query, no explanation."
    response = invoke_with_retry(llm, prompt)
    return message_text(response).strip().strip('"')


def hybrid_search(
    client: QdrantClient,
    search_query: str,
    top_k: int,
    candidate_pool: int = 20,
    keyword_weight: float = 0.3,
) -> List[RetrievedChunk]:
    """
    keyword_weight=0.0 degrades this to pure vector search — used by the eval
    harness to A/B "vector only" vs "hybrid" retrieval on the same corpus.
    """
    query_vector = embed_texts([search_query])[0].tolist()
    # query_points, not the older search(): qdrant-client removed search() in
    # 1.13. query_points exists from 1.12 onward, so this works across both.
    hits = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=candidate_pool,
    ).points

    query_kw = _keywords(search_query)
    vector_weight = 1.0 - keyword_weight
    reranked = []
    for hit in hits:
        payload = hit.payload or {}
        chunk_text = payload.get("text", "")
        kw_score = _keyword_overlap_score(query_kw, chunk_text) if keyword_weight > 0 else 0.0
        blended = vector_weight * hit.score + keyword_weight * kw_score
        reranked.append((blended, hit, payload))

    reranked.sort(key=lambda x: x[0], reverse=True)
    top = reranked[:top_k]

    return [
        RetrievedChunk(
            text=payload.get("text", ""),
            source_id=payload.get("source_id", ""),
            source_title=payload.get("source_title", ""),
            chunk_index=payload.get("chunk_index", -1),
            score=round(float(score), 4),
        )
        for score, hit, payload in top
    ]


def retriever_node(state: AgentState, client: QdrantClient, keyword_weight: float = 0.3) -> AgentState:
    feedback = state.get("critic_feedback", "") if state.get("retry_count", 0) > 0 else ""
    search_query = rewrite_query(state["question"], feedback)
    retrieved = hybrid_search(client, search_query, top_k=settings.top_k, keyword_weight=keyword_weight)
    return {
        **state,
        "search_query": search_query,
        "retrieved": retrieved,
    }
