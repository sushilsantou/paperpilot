"""
PDF text extraction + fixed-size chunking with overlap.
Word-based (not char-based) chunking so chunk boundaries respect token-ish
units — good enough for a portfolio project without pulling in a tokenizer.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader


@dataclass
class Chunk:
    text: str
    chunk_index: int
    source_id: str
    source_title: str


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 — a single malformed page shouldn't kill extraction
            continue
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """
    Splits `text` into overlapping word-count windows.
    chunk_size / overlap are in words, not characters.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_paper(pdf_path: str, source_id: str, source_title: str, chunk_size: int, overlap: int) -> List[Chunk]:
    text = extract_text_from_pdf(pdf_path)
    raw_chunks = chunk_text(text, chunk_size, overlap)
    return [
        Chunk(text=c, chunk_index=i, source_id=source_id, source_title=source_title)
        for i, c in enumerate(raw_chunks)
    ]
