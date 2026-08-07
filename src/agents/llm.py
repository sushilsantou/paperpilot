"""
Single place that constructs the Gemini chat client, so every agent shares
config (model name, temperature per role) instead of instantiating its own.
"""
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    if not settings.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your Gemini API key."
        )
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=temperature,
    )
