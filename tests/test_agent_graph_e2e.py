"""
End-to-end tests of the REAL LangGraph pipeline. Only the LLM and the embedding
model are stubbed; the graph construction, conditional routing, structural
citation check, retry accounting and Qdrant search are all the production code
paths. These are the tests that prove the agent loop actually behaves the way
the README claims it does.
"""
import pytest

from src.agents.graph import answer_question
from src.config import settings
from src.stubs import CORPUS, FakeLLM, build_populated_client, fake_embed_texts


@pytest.fixture
def client():
    return build_populated_client(CORPUS)


@pytest.fixture(autouse=True)
def stub_llm_and_embeddings(monkeypatch):
    """Returns a setter the tests call with their own FakeLLM script."""
    import src.agents.critic_agent as critic_mod
    import src.agents.retriever_agent as retriever_mod
    import src.agents.synthesis_agent as synthesis_mod

    monkeypatch.setattr(retriever_mod, "embed_texts", fake_embed_texts)

    def install(fake: FakeLLM):
        for mod in (retriever_mod, synthesis_mod, critic_mod):
            monkeypatch.setattr(mod, "get_llm", lambda temperature=0.0, _f=fake: _f)
        return fake

    return install


def test_happy_path_approved_first_try(client, stub_llm_and_embeddings):
    draft = "RAG grounds answers in retrieved evidence [arxiv:2401.00001]."
    fake = stub_llm_and_embeddings(
        FakeLLM(
            drafts=[draft],
            verdicts=[{"faithful": True, "issues": "", "sufficient_context": True}],
        )
    )

    result = answer_question("How does RAG reduce hallucination?", client=client)

    assert result["final_answer"] == draft
    assert result["critic_verdict"] == "approved"
    assert result["retry_count"] == 0
    assert result["source_ids_cited"] == ["2401.00001"]
    assert fake.calls == {"rewrite": 1, "synthesis": 1, "critic": 1}


def test_hallucinated_citation_is_caught_structurally_without_an_llm_call(client, stub_llm_and_embeddings):
    """A citation to a source that was never retrieved must be rejected by the
    deterministic check — the LLM judge should not even be consulted on that pass."""
    bad_draft = "RAG cuts hallucination by 40% [arxiv:9999.99999]."
    good_draft = "RAG grounds answers in retrieved evidence [arxiv:2401.00001]."

    fake = stub_llm_and_embeddings(
        FakeLLM(
            drafts=[bad_draft, good_draft],
            verdicts=[{"faithful": True, "issues": "", "sufficient_context": True}],
        )
    )

    result = answer_question("How does RAG reduce hallucination?", client=client)

    assert result["final_answer"] == good_draft
    assert result["retry_count"] == 1
    # synthesis ran twice, but the critic LLM only ran on the second (structurally valid) draft
    assert fake.calls["synthesis"] == 2
    assert fake.calls["critic"] == 1
    # the retriever was NOT re-run: bad context was never the problem
    assert fake.calls["rewrite"] == 1


def test_insufficient_context_routes_back_to_the_retriever(client, stub_llm_and_embeddings):
    draft = "RAG grounds answers in retrieved evidence [arxiv:2401.00001]."
    fake = stub_llm_and_embeddings(
        FakeLLM(
            drafts=[draft, draft],
            verdicts=[
                {"faithful": True, "issues": "excerpts do not cover multi-hop retrieval", "sufficient_context": False},
                {"faithful": True, "issues": "", "sufficient_context": True},
            ],
            query_rewrites=["multi-hop retrieval", "multi-hop retrieval iterative reasoning"],
        )
    )

    result = answer_question("How does multi-hop retrieval work?", client=client)

    assert result["critic_verdict"] == "approved"
    assert result["retry_count"] == 1
    # the distinguishing assertion: the RETRIEVER re-ran, not just synthesis
    assert fake.calls["rewrite"] == 2
    assert fake.calls["synthesis"] == 2


def test_retry_loop_is_bounded_and_flags_the_unverified_answer(client, stub_llm_and_embeddings):
    """A critic that never approves must not loop forever — it should give up
    after max_critic_retries and return the draft with an explicit caveat."""
    draft = "RAG grounds answers in retrieved evidence [arxiv:2401.00001]."
    fake = stub_llm_and_embeddings(
        FakeLLM(
            drafts=[draft] * 10,
            verdicts=[{"faithful": False, "issues": "unsupported claim", "sufficient_context": True}] * 10,
        )
    )

    result = answer_question("How does RAG reduce hallucination?", client=client)

    assert result["retry_count"] == settings.max_critic_retries
    assert "could not be fully verified" in result["final_answer"]
    assert fake.calls["critic"] == settings.max_critic_retries + 1


def test_critic_survives_an_unparseable_judge_response(client, stub_llm_and_embeddings):
    """If the judge returns malformed JSON the pipeline must fail open on the
    draft rather than crash — an LLM returning bad JSON is a normal event."""
    import src.agents.critic_agent as critic_mod

    draft = "RAG grounds answers in retrieved evidence [arxiv:2401.00001]."
    fake = stub_llm_and_embeddings(FakeLLM(drafts=[draft], verdicts=[]))

    class BadJudge(FakeLLM):
        def invoke(self, prompt):
            if "fact-checker" in prompt:
                self.calls["critic"] += 1
                from types import SimpleNamespace

                return SimpleNamespace(content="I think this looks fine, honestly")
            return super().invoke(prompt)

    bad = stub_llm_and_embeddings(BadJudge(drafts=[draft], verdicts=[]))

    result = answer_question("How does RAG reduce hallucination?", client=client)

    assert result["final_answer"] == draft
    assert result["critic_verdict"] == "approved"


def test_repeated_structural_failures_are_bounded(client, stub_llm_and_embeddings):
    """Regression: the structural check is a rejection path too, so it must obey
    max_critic_retries.

    It used to return "needs_rewrite" directly, without consulting the cap — the
    cap check sat after the LLM judge call, which this branch returns before
    reaching. A synthesis agent that kept fabricating citations therefore looped
    until LangGraph's recursion limit, thousands of model calls later. The
    existing bounded-retry test missed it because it only ever drove the
    *semantic* path, where the cap did apply.
    """
    fake = stub_llm_and_embeddings(
        FakeLLM(
            drafts=["RAG cuts hallucination by 40% [arxiv:9999.99999]."] * 40,
            verdicts=[{"faithful": True, "issues": "", "sufficient_context": True}] * 40,
        )
    )

    result = answer_question("How does RAG reduce hallucination?", client=client)

    assert result["retry_count"] == settings.max_critic_retries
    assert result["critic_verdict"] == "unverified"
    # synthesis runs once per attempt: the initial draft plus one per retry
    assert fake.calls["synthesis"] == settings.max_critic_retries + 1
    # the judge is never consulted — every draft dies at the structural gate
    assert fake.calls["critic"] == 0
    # the caveat must name the fabricated ID, not just say "unverified"
    assert "9999.99999" in result["final_answer"]


def test_giving_up_is_not_reported_as_approved(client, stub_llm_and_embeddings):
    """Regression: hitting the retry cap used to set critic_verdict="approved",
    making the field that records whether the citation gate passed say the
    opposite of what happened."""
    draft = "RAG grounds answers in retrieved evidence [arxiv:2401.00001]."
    stub_llm_and_embeddings(
        FakeLLM(
            drafts=[draft] * 10,
            verdicts=[{"faithful": False, "issues": "unsupported claim", "sufficient_context": True}] * 10,
        )
    )

    result = answer_question("How does RAG reduce hallucination?", client=client)

    assert result["critic_verdict"] == "unverified"
    assert "unsupported claim" in result["final_answer"]
