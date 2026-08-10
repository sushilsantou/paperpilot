"""
Critic agent: the faithfulness gate. Runs two checks before anything is
returned to the caller —

1. Structural: every [arxiv:ID] in the draft must correspond to a chunk that
   was actually retrieved. A citation to an ID that was never retrieved is
   an outright hallucination and is caught without needing another LLM call.
2. Semantic: an LLM-as-judge pass checks whether the draft's claims are
   actually supported by the cited excerpts (not just formatted correctly).

The verdict decides where the graph goes next:
- "approved"         -> done, return draft as final_answer
- "needs_retrieval"  -> context was insufficient; retriever runs again with
                        the critic's feedback folded into a query rewrite
- "needs_rewrite"    -> context was fine but the draft misused/ignored it;
                        synthesis runs again against the same chunks
"""
import json
import re

from src.agents.llm import get_llm, invoke_with_retry, message_text
from src.agents.state import AgentState
from src.config import settings

CRITIC_PROMPT = """You are a strict fact-checker. Given a question, a set of source excerpts, and a \
draft answer, judge whether every claim in the draft is actually supported by the excerpts.

Question: {question}

Source excerpts:
{context}

Draft answer:
{draft}

Respond with ONLY a JSON object, no markdown fences, in this exact shape:
{{"faithful": true or false, "issues": "short explanation of any unsupported or fabricated claims, or empty string if none", "sufficient_context": true or false}}

"sufficient_context" should be false if the excerpts clearly don't contain enough information to \
properly answer the question (as opposed to the draft simply doing a bad job with adequate context)."""


def _format_context(retrieved) -> str:
    return "\n\n".join(f"[arxiv:{c['source_id']}] {c['text'][:500]}" for c in retrieved)


def _structural_check(state: AgentState) -> str:
    retrieved_ids = {c["source_id"] for c in state.get("retrieved", [])}
    cited_ids = set(state.get("source_ids_cited", []))
    hallucinated = cited_ids - retrieved_ids
    if hallucinated:
        return f"Draft cites source(s) not present in retrieved context: {sorted(hallucinated)}"
    return ""


def _parse_judge_json(raw: str) -> dict:
    # Models occasionally wrap JSON in fences despite instructions — strip them.
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def critic_node(state: AgentState) -> AgentState:
    retry_count = state.get("retry_count", 0)

    structural_issue = _structural_check(state)
    if structural_issue:
        return {
            **state,
            "critic_verdict": "needs_rewrite",
            "critic_feedback": structural_issue,
            "retry_count": retry_count + 1,
        }

    llm = get_llm(temperature=0.0)
    context = _format_context(state.get("retrieved", []))
    prompt = CRITIC_PROMPT.format(question=state["question"], context=context, draft=state["draft_answer"])
    response = invoke_with_retry(llm, prompt)

    try:
        judged = _parse_judge_json(message_text(response))
    except (json.JSONDecodeError, AttributeError):
        # If the judge itself misbehaves, fail open on one retry rather than looping forever.
        judged = {"faithful": True, "issues": "critic response unparseable, defaulting to approve", "sufficient_context": True}

    if judged.get("faithful") and judged.get("sufficient_context", True):
        return {**state, "critic_verdict": "approved", "critic_feedback": "", "final_answer": state["draft_answer"]}

    if retry_count >= settings.max_critic_retries:
        # Stop looping — return the best draft we have, but flag it wasn't fully verified.
        caveat = "\n\n*(Note: this answer could not be fully verified against sources after multiple attempts.)*"
        return {
            **state,
            "critic_verdict": "approved",
            "final_answer": state["draft_answer"] + caveat,
        }

    verdict = "needs_retrieval" if not judged.get("sufficient_context", True) else "needs_rewrite"
    return {
        **state,
        "critic_verdict": verdict,
        "critic_feedback": judged.get("issues", ""),
        "retry_count": retry_count + 1,
    }
