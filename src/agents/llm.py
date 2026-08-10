"""
Single place that constructs the Gemini chat client, so every agent shares
config (model name, temperature per role) instead of instantiating its own.
"""
import random
import re
import time
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


class QuotaExhausted(RuntimeError):
    """Raised when the API is rate-limited and retrying will not help soon."""


def invoke_with_retry(llm, prompt, max_attempts: int = 4):
    """
    Call the model, backing off on 429 RESOURCE_EXHAUSTED.

    The Gemini free tier enforces a *per-day, per-model* request quota (20 on
    gemini-3.5-flash), not just a per-minute one. A per-minute limit is worth
    sleeping through; a daily one is not — retrying for hours would look like a
    hang. So this honours the server's `retryDelay` for short waits and gives
    up immediately once the delay implies a daily cap, raising QuotaExhausted
    so the caller can save partial results instead of losing the whole run.
    """
    for attempt in range(max_attempts):
        try:
            return llm.invoke(prompt)
        except Exception as e:  # noqa: BLE001 — provider raises a wrapped error type
            text = str(e)
            if "429" not in text and "RESOURCE_EXHAUSTED" not in text:
                raise

            per_day = "PerDay" in text or "requests_per_day" in text.lower()
            match = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+(?:\.\d+)?)s", text)
            delay = float(match.group(1)) if match else 2 ** attempt

            if per_day or delay > 120 or attempt == max_attempts - 1:
                raise QuotaExhausted(
                    f"Gemini quota exhausted (per-day cap reached). Original error: {text[:200]}"
                ) from e

            time.sleep(delay + random.uniform(0, 1))  # jitter so parallel callers desynchronise
    raise QuotaExhausted("exhausted retries")


def message_text(message) -> str:
    """
    Extract plain text from an LLM response across langchain versions.

    `response.content` used to be a `str`. Current langchain-google-genai
    returns a list of content blocks instead —
    `[{'type': 'text', 'text': 'OK', 'extras': {...}}]` — so calling
    `.content.strip()` raises AttributeError: 'list' object has no attribute
    'strip'. Every agent goes through this helper so the shape change is
    handled in exactly one place, and so older versions returning a bare
    string keep working.
    """
    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        if parts:
            return "".join(parts)

    # Fallback: `.text` is a property on current versions, a method on older ones.
    text = getattr(message, "text", None)
    if callable(text):
        text = text()
    return text or ""
