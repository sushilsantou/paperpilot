"""
Typed schema for what gets stored alongside every vector.

Before this, chunk payloads were an ad-hoc dict written in one place and read
by string key in three others — so adding a field meant re-ingesting and
hoping every reader agreed. This module makes the payload a declared contract:
one dataclass, one place that knows which fields exist, and one place that
declares which of them are indexed for filtering.

Indexed fields are the ones Qdrant can filter on server-side. That is the
difference between "we store metadata" and "metadata is queryable": a filtered
search restricts the candidate set *before* scoring, so `top_k` is spent on
eligible chunks instead of discarding most of them after the fact.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# Qdrant payload index types, keyed by field name. Only fields listed here can
# be filtered on efficiently; everything else is stored but scanned.
INDEXED_FIELDS: Dict[str, str] = {
    "source_id": "keyword",
    "primary_category": "keyword",
    "categories": "keyword",
    "authors": "keyword",
    "published_year": "integer",
    "chunk_index": "integer",
}

SCHEMA_VERSION = 2  # bumped when payload shape changes; written into every point


@dataclass
class ChunkPayload:
    """Everything stored with a chunk vector."""

    text: str
    source_id: str
    source_title: str
    chunk_index: int

    # Metadata carried from the paper record, enabling filtered search and
    # graph construction without a second lookup.
    authors: List[str] = field(default_factory=list)
    primary_category: str = ""
    categories: List[str] = field(default_factory=list)
    published: str = ""
    published_year: Optional[int] = None

    schema_version: int = SCHEMA_VERSION

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ChunkPayload":
        """
        Tolerant of older points: v1 payloads predate the metadata fields, so
        missing keys fall back to defaults rather than raising. That keeps a
        mixed-version collection readable instead of forcing a re-ingest.
        """
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


def year_from_published(published: str) -> Optional[int]:
    """'2025-06-08T01:33:05+00:00' -> 2025. Returns None rather than raising."""
    if not published or len(published) < 4 or not published[:4].isdigit():
        return None
    return int(published[:4])


def build_chunk_payload(chunk, paper: Dict[str, Any]) -> ChunkPayload:
    """Join a chunk with its paper's metadata record."""
    return ChunkPayload(
        text=chunk.text,
        source_id=chunk.source_id,
        source_title=chunk.source_title,
        chunk_index=chunk.chunk_index,
        authors=paper.get("authors", []),
        primary_category=paper.get("primary_category", ""),
        categories=paper.get("categories", []),
        published=paper.get("published", ""),
        published_year=year_from_published(paper.get("published", "")),
    )
