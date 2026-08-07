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


def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection in existing:
        return
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=embedding_dim(), distance=Distance.COSINE),
    )


def load_all_chunks(meta_path: str) -> List[Chunk]:
    papers = json.loads(Path(meta_path).read_text())
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
    return all_chunks


def upsert_chunks(client: QdrantClient, chunks: List[Chunk], batch_size: int = 64) -> int:
    total = 0
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding + upserting"):
        batch = chunks[i : i + batch_size]
        vectors = embed_texts([c.text for c in batch])
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[j].tolist(),
                payload={
                    "text": c.text,
                    "source_id": c.source_id,
                    "source_title": c.source_title,
                    "chunk_index": c.chunk_index,
                },
            )
            for j, c in enumerate(batch)
        ]
        client.upsert(collection_name=settings.qdrant_collection, points=points)
        total += len(points)
    return total


def run(meta_path: str = "data/raw/papers_meta.json") -> int:
    client = get_client()
    ensure_collection(client)
    chunks = load_all_chunks(meta_path)
    print(f"{len(chunks)} chunks to embed and upsert")
    count = upsert_chunks(client, chunks)
    print(f"Upserted {count} chunks into '{settings.qdrant_collection}'")
    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest chunked papers into Qdrant")
    parser.add_argument("--meta", default="data/raw/papers_meta.json")
    args = parser.parse_args()
    run(args.meta)
