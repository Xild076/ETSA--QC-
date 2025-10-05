"""General utilities for loading environment variables and normalising text."""

from __future__ import annotations

import os
import re
from typing import Optional

try:  # pragma: no cover - optional dependency for development environments
    from dotenv import find_dotenv, load_dotenv
except Exception:  # pragma: no cover - tolerate missing dependency in production
    find_dotenv = None
    load_dotenv = None

SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^a-z0-9\s]")
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "is",
    "are",
    "be",
    "was",
    "were",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "my",
    "your",
    "our",
    "their",
    "very",
    "too",
    "so",
    "just",
    "but",
}

_ENV_LOADED = False


def ensure_env_loaded() -> None:
    """Load variables from .env (if present) and normalise Google API keys."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    if load_dotenv and find_dotenv:
        try:
            env_path = find_dotenv(usecwd=True)
            if env_path:
                load_dotenv(env_path, override=False)
        except Exception:
            # Missing .env files or dotenv errors should not crash the pipeline.
            pass

    key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GENAI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )
    if key and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = key

    _ENV_LOADED = True


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Return the requested environment variable after lazily loading .env."""
    ensure_env_loaded()
    return os.getenv(name, default)


def normalize_text(text: str) -> str:
    """Normalise text for comparisons by lowercasing and collapsing whitespace."""
    lowered = str(text or "").lower()
    cleaned = PUNCT_RE.sub(" ", lowered)
    collapsed = SPACE_RE.sub(" ", cleaned).strip()
    return collapsed