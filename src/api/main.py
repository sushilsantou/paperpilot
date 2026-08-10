"""
FastAPI wrapper around the multi-agent graph.

Run locally:    uvicorn src.api.main:app --reload
Run via Docker: docker compose up
"""
from fastapi import FastAPI, HTTPException
from qdrant_client import QdrantClient

from src.agents.graph import answer_question
from src.api.schemas import QueryRequest, QueryResponse, SourceRef
from src.config import settings
from src.vectorstore import get_qdrant_client

app = FastAPI(
    title="PaperPilot",
    description="Multi-agent RAG research assistant over arXiv ML/AI papers, with a critic-verified citation loop.",
    version="0.1.0",
)

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = get_qdrant_client()
    return _client


@app.get("/health")
def health():
    try:
        collections = [c.name for c in get_client().get_collections().collections]
        ingested = settings.qdrant_collection in collections
    except Exception as e:  # noqa: BLE001
        return {"status": "degraded", "qdrant_error": str(e)}
    return {"status": "ok", "collection_ready": ingested}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not settings.google_api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY is not configured on the server.")

    try:
        result = answer_question(req.question, client=get_client())
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Agent pipeline failed: {e}") from e

    return QueryResponse(
        question=result["question"],
        search_query=result.get("search_query"),
        answer=result.get("final_answer", "(no answer produced)"),
        sources_cited=result.get("source_ids_cited", []),
        retrieved_sources=[SourceRef(**c) for c in result.get("retrieved", [])],
        retries=result.get("retry_count", 0),
    )
