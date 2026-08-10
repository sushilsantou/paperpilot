"""
Single place that decides how to reach Qdrant.

Two modes, because the docker-compose server is not always available (no
Docker on a locked-down laptop, CI without a service container, a quick eval
run):

  * server mode  — QDRANT_URL points at a running instance (docker compose up
                   -d qdrant). What a real deployment uses.
  * embedded mode — QDRANT_PATH points at a directory and qdrant-client runs
                   the store in-process against local files. Same client API,
                   no server, no Docker.

Embedded mode holds an exclusive file lock on its directory, so exactly one
process may use it at a time — ingest, then eval, then serve; not two at
once. That is the tradeoff for dropping the server dependency, and it is why
server mode stays the default whenever QDRANT_URL is set.
"""
from qdrant_client import QdrantClient

from src.config import settings


def get_qdrant_client() -> QdrantClient:
    if settings.qdrant_path:
        return QdrantClient(path=settings.qdrant_path)
    return QdrantClient(url=settings.qdrant_url)
