"""
Tests for the eval aggregation. These are the numbers RESULTS.md publishes, so
the arithmetic is worth pinning down — especially the separation between
answers the critic approved and answers it gave up on.
"""
import pytest

from src.eval.run_eval import _summarise, run_strategy
from src.stubs import CORPUS, FakeLLM, build_populated_client, fake_embed_texts


def _record(citation_validity=1.0, unverified=False, recall=1.0, retries=0):
    return {
        "recall": recall,
        "citation_validity": citation_validity,
        "latency": 1.0,
        "retries": retries,
        "has_citations": 1.0,
        "unverified": unverified,
    }


def test_empty_run_reports_zero_questions_not_a_crash():
    """A quota wall on the very first question must summarise cleanly."""
    assert _summarise([]) == {"questions_completed": 0}


def test_unverified_rate_is_the_fraction_that_hit_the_retry_cap():
    summary = _summarise([_record(), _record(), _record(unverified=True), _record(unverified=True)])
    assert summary["unverified_rate"] == 0.5
    assert summary["questions_completed"] == 4


def test_verified_subset_is_reported_separately_from_the_pool():
    """The distinction that motivated the metric: pooled citation validity
    conflates 'the gate passed' with 'we ran out of retries'."""
    records = [
        _record(citation_validity=1.0),
        _record(citation_validity=1.0),
        _record(citation_validity=0.0, unverified=True),
    ]
    summary = _summarise(records)

    assert summary["avg_citation_validity"] == pytest.approx(2 / 3)   # pooled
    assert summary["avg_citation_validity_verified"] == 1.0           # gate actually passed
    assert summary["unverified_rate"] == pytest.approx(1 / 3)


def test_verified_metric_is_omitted_rather_than_zeroed_when_nothing_passed():
    """0.0 would read as 'validity collapsed'; the honest report is 'no verified
    answers to measure'. MLflow takes any logged number at face value."""
    summary = _summarise([_record(citation_validity=0.0, unverified=True)] * 3)

    assert summary["unverified_rate"] == 1.0
    assert "avg_citation_validity_verified" not in summary


def test_run_strategy_flags_an_answer_the_critic_gave_up_on(monkeypatch):
    """Wiring test: a critic that never approves must surface as unverified in
    the eval output, not be silently averaged in as a normal result."""
    import src.agents.critic_agent as critic_mod
    import src.agents.retriever_agent as retriever_mod
    import src.agents.synthesis_agent as synthesis_mod

    monkeypatch.setattr(retriever_mod, "embed_texts", fake_embed_texts)
    fake = FakeLLM(
        drafts=["RAG grounds answers in retrieved evidence [arxiv:2401.00001]."] * 10,
        verdicts=[{"faithful": False, "issues": "unsupported claim", "sufficient_context": True}] * 10,
    )
    for mod in (retriever_mod, synthesis_mod, critic_mod):
        monkeypatch.setattr(mod, "get_llm", lambda temperature=0.0, _f=fake: _f)

    eval_set = [{"id": "q1", "question": "How does RAG reduce hallucination?",
                 "expected_keywords": ["retrieval", "hallucination"]}]

    summary = run_strategy("hybrid", {"keyword_weight": 0.3}, eval_set, build_populated_client(CORPUS))

    assert summary["questions_completed"] == 1
    assert summary["unverified_rate"] == 1.0
    assert "avg_citation_validity_verified" not in summary
