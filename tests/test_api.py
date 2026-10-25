"""
API contract test. The pipeline itself is covered in test_agent_graph_e2e.py;
this checks that what the pipeline decided survives the trip through /query.
"""
from fastapi.testclient import TestClient

import src.api.main as api
from src.config import settings


def test_query_reports_an_unverified_answer_as_unverified(monkeypatch):
    """The verdict used to be dropped at the API boundary, so a caller could
    only spot an answer the critic gave up on by parsing the caveat text."""
    monkeypatch.setattr(settings, "google_api_key", "test-key")
    monkeypatch.setattr(api, "get_client", lambda: None)
    monkeypatch.setattr(api, "answer_question", lambda question, client=None: {
        "question": question,
        "search_query": "rag hallucination",
        "final_answer": "Draft answer [arxiv:2401.00001].\n\n*(Note: this answer could not be fully verified ...)*",
        "source_ids_cited": ["2401.00001"],
        "retrieved": [{"text": "...", "source_id": "2401.00001", "source_title": "Paper",
                       "chunk_index": 0, "score": 0.9}],
        "retry_count": 2,
        "critic_verdict": "unverified",
    })

    response = TestClient(api.app).post("/query", json={"question": "How does RAG reduce hallucination?"})

    assert response.status_code == 200
    body = response.json()
    assert body["critic_verdict"] == "unverified"
    assert body["retries"] == 2
