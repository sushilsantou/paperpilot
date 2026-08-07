from src.ingestion.chunk import chunk_text


def test_chunk_text_basic_windowing():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) > 1
    # first chunk should start at word0
    assert chunks[0].startswith("word0 word1")


def test_chunk_text_overlap():
    text = " ".join(f"word{i}" for i in range(50))
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    # last `overlap` words of chunk 0 should equal first `overlap` words of chunk 1
    assert first_words[-5:] == second_words[:5]


def test_chunk_text_empty_string():
    assert chunk_text("", chunk_size=20, overlap=5) == []


def test_chunk_text_shorter_than_chunk_size():
    text = "just a few words here"
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) == 1
    assert chunks[0] == text
