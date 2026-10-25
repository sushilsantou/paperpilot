"""
Critic agent: the faithfulness gate. Runs two checks before anything is
returned to the caller —

1. Structural: the draft must cite at least one source, and every [arxiv:ID]
   in it must correspond to a chunk that was actually retrieved. A citation to
   an ID that was never retrieved is an outright hallucination; a draft with
   no citations gives the gate nothing to check. Both are caught without
   needing another LLM call.
2. Semantic: an LLM-as-judge pass checks whether the draft's claims are
   actually supported by the cited excerpts (not just formatted correctly).
   A judge response that cannot be parsed is a failed check, not a pass.

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

    This used to truncate to `text[:500]`. Chunks are 800 *words*
    (settings.chunk_size is a word count; see src/ingestion/chunk.py), a median
    of 5,496 characters across the indexed corpus, so the judge saw about a
    tenth of each chunk. Any claim drawn from past the first few sentences
    looked unsupported because the supporting text was not in the prompt. The
    judge was right to reject; it was being shown less than the writer.

    Cost is not a reason to reintroduce the cap: top_k=5 chunks is roughly 27k
    characters (~7k tokens) of context, well inside the model's window, and a
    judge that sees less than the writer is not checking the draft.
    """
    return "\n\n".join(f"[arxiv:{c['source_id']}] {c['text']}" for c in retrieved)


def _structural_check(state: AgentState) -> tuple[str, str] | None:
    """
    Deterministic citation checks, run before the LLM judge. Returns
    (verdict, feedback) for a rejection, or None if the draft passes.
    """
    cited_ids = set(state.get("source_ids_cited", []))
    if not cited_ids:
        # Uncited drafts used to pass here: this check only compared cited IDs
        # against retrieved ones, and an empty set has nothing to compare. The
        # judge then asked whether claims were *supported*, never whether they
        # were *cited*, so a citation gate could approve an answer with no
        # citations at all.
        #
        # Routed to the retriever rather than synthesis: a writer that cited
        # nothing most likely found nothing it could attribute, and a fresh
        # search guided by this feedback addresses that. Re-drafting against
        # the same chunks only helps if it merely forgot the citation format.
        return "needs_retrieval", "Draft cited no sources, so none of its claims could be checked against the excerpts"

    retrieved_ids = {c["source_id"] for c in state.get("retrieved", [])}
    hallucinated = cited_ids - retrieved_ids
    if hallucinated:
        return "needs_rewrite", f"Draft cites source(s) not present in retrieved context: {sorted(hallucinated)}"
    return None


def _parse_judge_json(raw: str) -> dict:
    """
    Parse the judge's verdict, or raise ValueError if it isn't one.

    Tolerates the two common deviations from "ONLY a JSON object": markdown
    fences, and a sentence of prose around the object. Anything else (no
    object, or an object without a boolean "faithful") is a malformed verdict.
    """
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        judged = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        judged = json.loads(match.group(0))

    if not isinstance(judged, dict) or not isinstance(judged.get("faithful"), bool):
        raise ValueError(f"judge verdict has no boolean 'faithful' field: {raw[:200]!r}")
    if "sufficient_context" in judged and not isinstance(judged["sufficient_context"], bool):
        raise ValueError(f"judge verdict has a non-boolean 'sufficient_context': {raw[:200]!r}")
    return judged


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

    structural = _structural_check(state)
    if structural:
        verdict, issue = structural
        return _reject(state, retry_count, verdict, issue)

    llm = get_llm(temperature=0.0)
    context = _format_context(state.get("retrieved", []))
    prompt = CRITIC_PROMPT.format(question=state["question"], context=context, draft=state["draft_answer"])
    response = invoke_with_retry(llm, prompt)

    try:
        judged = _parse_judge_json(message_text(response))
    except (ValueError, AttributeError):
        # An unparseable judge is a failed check, not a passed one. This used
        # to fail open: prose instead of JSON set faithful=True, the draft came
        # back "approved", and nothing recorded that the judge never ruled. It
        # now rejects through _reject() like any other failure, so a judge that
        # never returns valid JSON ends "unverified" at the cap.
        #
        # It routes to synthesis because the graph has no judge-only retry;
        # that costs one extra synthesis call per malformed verdict.
        return _reject(state, retry_count, "needs_rewrite",
                       "Critic response was not a valid JSON verdict, so the draft was not verified")

    if judged.get("faithful") and judged.get("sufficient_context", True):
        return {**state, "critic_verdict": "approved", "critic_feedback": "", "final_answer": state["draft_answer"]}

    verdict = "needs_retrieval" if not judged.get("sufficient_context", True) else "needs_rewrite"
    return _reject(state, retry_count, verdict, judged.get("issues", ""))
