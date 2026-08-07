from src.agents.retriever_agent import _keyword_overlap_score, _keywords


def test_keywords_strips_stopwords_and_short_tokens():
    kw = _keywords("What is the retrieval augmented generation approach for a language model?")
    assert "the" not in kw
    assert "is" not in kw
    assert "retrieval" in kw
    assert "augmented" in kw
    assert "language" in kw


def test_keyword_overlap_score_perfect_match():
    query_kw = _keywords("retrieval augmented generation")
    score = _keyword_overlap_score(query_kw, "This paper covers retrieval augmented generation in depth.")
    assert score == 1.0


def test_keyword_overlap_score_no_match():
    query_kw = _keywords("retrieval augmented generation")
    score = _keyword_overlap_score(query_kw, "This paper is about reinforcement learning for robotics.")
    assert score == 0.0


def test_keyword_overlap_score_empty_query():
    assert _keyword_overlap_score(_keywords(""), "any text") == 0.0
