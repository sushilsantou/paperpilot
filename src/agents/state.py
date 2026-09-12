"""
Shared state passed between nodes in the LangGraph agent graph.
Every agent reads/writes a subset of this — keeping it as one typed dict
(rather than passing bespoke args between functions) is what makes the
critic's re-retrieval loop possible: it can hand control back to the
retriever with an updated query and the graph just keeps going.
"""
from typing import List, TypedDict, Optional


class RetrievedChunk(TypedDict):
    text: str
    source_id: str
    source_title: str
    chunk_index: int
    score: float


class AgentState(TypedDict, total=False):
    question: str                      # original user question, never mutated
    search_query: str                  # possibly rewritten query used for retrieval
    retrieved: List[RetrievedChunk]     # chunks pulled back from Qdrant
    draft_answer: str                  # synthesis agent's answer with inline citations
    critic_verdict: str                # "approved" | "unverified" | "needs_retrieval" | "needs_rewrite"
    critic_feedback: str               # why the critic rejected it, fed back into retry
    final_answer: str                  # what actually gets returned to the caller
    retry_count: int                   # guards against infinite critic loops
    source_ids_cited: List[str]        # which source_ids the draft actually cited
