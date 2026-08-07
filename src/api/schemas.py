from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Question to ask over the ingested paper corpus")


class SourceRef(BaseModel):
    source_id: str
    source_title: str
    chunk_index: int
    score: float


class QueryResponse(BaseModel):
    question: str
    search_query: Optional[str] = None
    answer: str
    sources_cited: List[str] = []
    retrieved_sources: List[SourceRef] = []
    retries: int = 0
