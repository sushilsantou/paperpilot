"""
End-to-end ingestion: PDFs on disk (from fetch_arxiv.py) -> chunks -> local
embeddings -> upserted into Qdrant.

Usage:
    python -m src.ingestion.ingest --meta data/raw/papers_meta.json
"""
import json
import uuid
from pathlib import Path
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

from src.config import settings
from src.ingestion.chunk import chunk_paper, Chunk
from src.ingestion.embed import embed_texts, embedding_dim
from src.metadata.schema import INDEXED_FIELDS, build_chunk_payload
from src.vectorstore import get_qdrant_client


def get_client() -> QdrantClient:
    return get_qdrant_client()


def ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=embedding_dim(), distance=Distance.COSINE),
        )
    ensure_payload_indexes(client)


def ensure_payload_indexes(client: QdrantClient) -> None:
    """
    Declare which payload fields are filterable (src/metadata/schema.py).

    Without an index Qdrant still *stores* the field but has to scan to filter
    on it. With one, a filtered search narrows candidates before scoring, so
    top_k is spent on chunks that already satisfy the filter. Creating an index
    that already exists is a no-op error, so it is swallowed deliberately.
    """
    for field_name, schema in INDEXED_FIELDS.items():
        try:
            client.create_payload_index(
                collection_name=settings.qdrant_collection,
                field_name=field_name,
                field_schema=schema,
            )
        except Exception:  # noqa: BLE001 — already-exists is the common case
            pass


def load_all_chunks(meta_path: str):
    """Returns (chunks, {arxiv_id: paper_record}) so payloads can carry metadata."""
    papers = json.loads(Path(meta_path).read_text())
    by_id = {p["arxiv_id"]: p for p in papers}
    all_chunks: List[Chunk] = []
    for paper in tqdm(papers, desc="Chunking papers"):
        chunks = chunk_paper(
            pdf_path=paper["pdf_path"],
            source_id=paper["arxiv_id"],
            source_title=paper["title"],
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        all_chunks.extend(chunks)
    return all_chunks, by_id


def upsert_chunks(client: QdrantClient, chunks: List[Chunk], papers_by_id: dict, batch_size: int = 64) -> int:
    total = 0
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding + upserting"):
        batch = chunks[i : i + batch_size]
        vectors = embed_texts([c.text for c in batch])
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[j].tolist(),
                payload=build_chunk_payload(c, papers_by_id.get(c.source_id, {})).to_payload(),
            )
            for j, c in enumerate(batch)
        ]
        client.upsert(collection_name=settings.qdrant_collection, points=points)
        total += len(points)
    return total


def run(meta_path: str = "data/raw/papers_meta.json") -> int:
    client = get_client()
    ensure_collection(client)
    chunks, papers_by_id = load_all_chunks(meta_path)
    print(f"{len(chunks)} chunks to embed and upsert")
    count = upsert_chunks(client, chunks, papers_by_id)
    print(f"Upserted {count} chunks into '{settings.qdrant_collection}'")
    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest chunked papers into Qdrant")
    parser.add_argument("--meta", default="data/raw/papers_meta.json")
    args = parser.parse_args()
    run(args.meta)
