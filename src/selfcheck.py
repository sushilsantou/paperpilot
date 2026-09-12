"""
Runs the complete agent pipeline offline — no API key, no network, no Docker —
and prints what each stage did.

The point is to separate two failure modes that otherwise look identical when a
first run goes wrong: "my install/wiring is broken" versus "my API key or model
choice is wrong". If this passes, everything except the model calls themselves
is working, and any remaining failure is credentials or quota.

    python -m src.selfcheck
"""
import sys

from src.config import settings
from src.stubs import CORPUS, FakeLLM, build_populated_client, fake_embed_texts


def _install_stubs(fake: FakeLLM) -> None:
    import src.agents.critic_agent as critic_mod
    import src.agents.retriever_agent as retriever_mod
    import src.agents.synthesis_agent as synthesis_mod

    retriever_mod.embed_texts = fake_embed_texts
    for mod in (retriever_mod, synthesis_mod, critic_mod):
        mod.get_llm = lambda temperature=0.0, _f=fake: _f


def main() -> int:
    print("PaperPilot self-check — offline, no API key required\n")

    from src.agents.graph import answer_question

    checks_passed = 0
    checks_total = 0

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal checks_passed, checks_total
        checks_total += 1
        if condition:
            checks_passed += 1
            print(f"  [pass] {label}" + (f" — {detail}" if detail else ""))
        else:
            print(f"  [FAIL] {label}" + (f" — {detail}" if detail else ""))

    # --- 1. vector store + retrieval -------------------------------------
    print("1. Vector store and hybrid retrieval")
    client = build_populated_client(CORPUS)
    collections = [c.name for c in client.get_collections().collections]
    check("in-process Qdrant collection created", settings.qdrant_collection in collections,
          f"collection '{settings.qdrant_collection}', {len(CORPUS)} documents")

    # --- 2. happy path ----------------------------------------------------
    print("\n2. Full agent loop, answer approved first pass")
    draft = "RAG grounds answers in retrieved evidence [arxiv:2401.00001]."
    fake = FakeLLM(drafts=[draft], verdicts=[{"faithful": True, "issues": "", "sufficient_context": True}])
    _install_stubs(fake)
    result = answer_question("How does RAG reduce hallucination?", client=client)
    check("retriever rewrote the query and searched", fake.calls["rewrite"] == 1,
          f"search query: {result.get('search_query')!r}")
    check("synthesis produced a cited draft", result.get("source_ids_cited") == ["2401.00001"])
    check("critic approved without a retry", result.get("retry_count") == 0)

    # --- 3. hallucinated citation ----------------------------------------
    print("\n3. Hallucination gate, citation to a source never retrieved")
    fake = FakeLLM(
        drafts=["RAG cuts hallucination by 40% [arxiv:9999.99999].", draft],
        verdicts=[{"faithful": True, "issues": "", "sufficient_context": True}],
    )
    _install_stubs(fake)
    result = answer_question("How does RAG reduce hallucination?", client=client)
    check("fabricated citation rejected and redrafted", result["final_answer"] == draft,
          "caught by the structural check")
    check("no LLM judge call wasted on the bad draft", fake.calls["critic"] == 1)
    check("retriever not re-run (context was fine)", fake.calls["rewrite"] == 1)

    # --- 4. insufficient context -----------------------------------------
    print("\n4. Routing, insufficient context sends control back to the retriever")
    fake = FakeLLM(
        drafts=[draft, draft],
        verdicts=[
            {"faithful": True, "issues": "excerpts do not cover the question", "sufficient_context": False},
            {"faithful": True, "issues": "", "sufficient_context": True},
        ],
    )
    _install_stubs(fake)
    result = answer_question("How does multi-hop retrieval work?", client=client)
    check("retriever ran a second time", fake.calls["rewrite"] == 2)
    check("answer approved after re-retrieval", result["critic_verdict"] == "approved")

    # --- 5. bounded retries ----------------------------------------------
    print("\n5. Safety, a critic that never approves must not loop forever")
    fake = FakeLLM(
        drafts=[draft] * 10,
        verdicts=[{"faithful": False, "issues": "unsupported", "sufficient_context": True}] * 10,
    )
    _install_stubs(fake)
    result = answer_question("How does RAG reduce hallucination?", client=client)
    check("loop stopped at the configured cap", result["retry_count"] == settings.max_critic_retries,
          f"max_critic_retries={settings.max_critic_retries}")
    check("unverified answer flagged to the caller", "could not be fully verified" in result["final_answer"])

    # --- summary ----------------------------------------------------------
    print(f"\n{checks_passed}/{checks_total} checks passed")
    if checks_passed != checks_total:
        print("\nSomething in the pipeline itself is broken — fix that before adding an API key.")
        return 1

    print("\nThe pipeline is working end to end. The only untested part is the real model calls.")
    if settings.google_api_key:
        print("GOOGLE_API_KEY is set — you're ready to run the real thing:")
    else:
        print("GOOGLE_API_KEY is NOT set. Get a free key at https://aistudio.google.com/apikey,")
        print("put it in .env, then run:")
    print("\n  python -m src.ingestion.fetch_arxiv --query \"retrieval augmented generation\" --max-results 25")
    print("  python -m src.ingestion.ingest")
    print("  python -m src.eval.run_eval")
    return 0


if __name__ == "__main__":
    sys.exit(main())
