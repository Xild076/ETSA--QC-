import re

SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^a-z0-9\s]")
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for",
    "with", "is", "are", "be", "was", "were", "it", "its", "this", "that",
    "these", "those", "my", "your", "our", "their", "very", "too", "so",
    "just", "but",
}

def normalize_text(text: str) -> str:
    """A standardized function to clean and normalize text for comparisons."""
    lowered = str(text or "").lower()
    cleaned = PUNCT_RE.sub(" ", lowered)
    collapsed = SPACE_RE.sub(" ", cleaned).strip()
    return collapsed

def get_env(key: str, default: str = None) -> str:
    """Safely gets an environment variable."""
    import os
    return os.environ.get(key, default)