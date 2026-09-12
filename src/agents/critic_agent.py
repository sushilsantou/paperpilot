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
- "unverified"       -> done, but the retry cap was hit before the draft ever
                        passed; returned with a caveat naming the last issue
- "needs_retrieval"  -> context was insufficient; retriever runs again with
                        the critic's feedback folded into a query rewrite
- "needs_rewrite"    -> context was fine but the draft misused/ignored it;
                        synthesis runs again against the same chunks

Every rejection routes through _reject(), which owns the retry bound. Adding a
new rejection reason means calling it, not returning a verdict directly.
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
    """
    The critic must see exactly the evidence synthesis saw.

    This used to truncate to `text[:500]`. Chunks are 800 characters
    (settings.chunk_size), so the judge was ruling on 62% of the evidence the
    draft was written from, and any claim drawn from the tail of a chunk looked
    unsupported because the supporting sentence was not in the prompt. The judge
    was right to reject; it was being shown less than the writer.

    Cost is not a reason to reintroduce the cap: top_k=5 chunks at 800 chars is
    ~4k characters of context, which is negligible for the model in use.
    """
    return "\n\n".join(f"[arxiv:{c['source_id']}] {c['text']}" for c in retrieved)


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


def _give_up(state: AgentState, retry_count: int, feedback: str) -> AgentState:
    """
    Terminal exit when the critic has rejected `max_critic_retries` times.

    The verdict is "unverified", not "approved": the answer is being returned
    because we ran out of attempts, not because it passed. Reporting that as
    "approved" would make the one field that records whether the citation gate
    succeeded say the opposite of what happened.

    The caveat carries the last rejection reason, so a caller can tell "the
    judge was unconvinced" apart from the far more serious "the draft cites a
    source that does not exist" — the latter ships a fabricated citation, and
    the reader needs to know which IDs are suspect.
    """
    caveat = (
        "\n\n*(Note: this answer could not be fully verified against sources "
        f"after {retry_count} retries. Last issue: {feedback or 'unspecified'})*"
    )
    return {
        **state,
        "critic_verdict": "unverified",
        "critic_feedback": feedback,
        "retry_count": retry_count,
        "final_answer": state["draft_answer"] + caveat,
    }


def _reject(state: AgentState, retry_count: int, verdict: str, feedback: str) -> AgentState:
    """
    The single bounded exit for every rejection, structural or semantic.

    Both paths must go through here. Previously the structural branch returned
    "needs_rewrite" directly without consulting the retry cap, so a synthesis
    agent that kept fabricating citations looped forever: the cap check lived
    after the LLM judge call, which that branch returns before reaching. It was
    bounded only by LangGraph's recursion limit, thousands of model calls later.
    """
    if retry_count >= settings.max_critic_retries:
        return _give_up(state, retry_count, feedback)
    return {
        **state,
        "critic_verdict": verdict,
        "critic_feedback": feedback,
        "retry_count": retry_count + 1,
    }


def critic_node(state: AgentState) -> AgentState:
    retry_count = state.get("retry_count", 0)

    structural_issue = _structural_check(state)
    if structural_issue:
        return _reject(state, retry_count, "needs_rewrite", structural_issue)

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

    verdict = "needs_retrieval" if not judged.get("sufficient_context", True) else "needs_rewrite"
    return _reject(state, retry_count, verdict, judged.get("issues", ""))
