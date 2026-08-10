"""
Synthesis agent: drafts an answer strictly from the retrieved chunks, with
inline citations like [arxiv:2310.01234] after every claim. Forcing citation
formatting here (rather than hoping the critic catches missing ones) keeps
the critic's job to *verification*, not formatting cleanup.
"""
import re
from typing import List

from src.agents.llm import get_llm, invoke_with_retry, message_text
from src.agents.state import AgentState, RetrievedChunk


def _format_context(chunks: List[RetrievedChunk]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(
            f"[arxiv:{c['source_id']}] (\"{c['source_title']}\", chunk {c['chunk_index']})\n{c['text']}"
        )
    return "\n\n---\n\n".join(blocks)


SYNTHESIS_PROMPT = """You are a research assistant answering questions using ONLY the provided excerpts from arXiv papers. \
Do not use outside knowledge. If the excerpts don't contain enough information to answer, say so explicitly.

Rules:
- Every factual claim must end with a citation tag in the form [arxiv:ID], where ID matches one of the excerpts below.
- Do not invent citation IDs that aren't in the excerpts.
- Be concise: 3-6 sentences unless the question clearly needs more.

Question: {question}

Excerpts:
{context}

Answer (with inline [arxiv:ID] citations):"""


def extract_cited_ids(answer: str) -> List[str]:
    return sorted(set(re.findall(r"\[arxiv:([^\]]+)\]", answer)))


def synthesis_node(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.2)
    context = _format_context(state.get("retrieved", []))
    prompt = SYNTHESIS_PROMPT.format(question=state["question"], context=context)
    response = invoke_with_retry(llm, prompt)
    draft = message_text(response).strip()
    return {
        **state,
        "draft_answer": draft,
        "source_ids_cited": extract_cited_ids(draft),
    }
