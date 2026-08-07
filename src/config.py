"""
Central config, loaded from environment variables (.env in local dev).
Keeping all tunables here means the eval harness can vary them per-run
without touching agent/ingestion code.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LLM
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # Embeddings — local model, no API key required
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector store
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "arxiv_papers")

    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "5"))

    # Agent loop
    max_critic_retries: int = int(os.getenv("MAX_CRITIC_RETRIES", "2"))


settings = Settings()
