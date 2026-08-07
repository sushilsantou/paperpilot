from src.eval.metrics import citation_validity_rate, has_citations, keyword_recall


def _chunk(text, source_id="p1"):
    return {"text": text, "source_id": source_id, "source_title": "t", "chunk_index": 0, "score": 1.0}


def test_keyword_recall_full_match():
    retrieved = [_chunk("Retrieval augmented generation improves grounding in language models.")]
    score = keyword_recall(retrieved, ["retrieval", "grounding", "language model"])
    assert score == 1.0


def test_keyword_recall_partial_match():
    retrieved = [_chunk("Retrieval is useful for many tasks.")]
    score = keyword_recall(retrieved, ["retrieval", "grounding"])
    assert score == 0.5


def test_keyword_recall_no_expected_keywords_defaults_to_full():
    assert keyword_recall([_chunk("anything")], []) == 1.0


def test_citation_validity_all_valid():
    retrieved = [_chunk("x", source_id="a"), _chunk("y", source_id="b")]
    assert citation_validity_rate(["a", "b"], retrieved) == 1.0


def test_citation_validity_partial():
    retrieved = [_chunk("x", source_id="a")]
    assert citation_validity_rate(["a", "hallucinated"], retrieved) == 0.5


def test_citation_validity_no_citations():
    assert citation_validity_rate([], [_chunk("x")]) == 0.0


def test_has_citations():
    assert has_citations("Some claim [arxiv:2310.01234].") is True
    assert has_citations("No citation here.") is False
