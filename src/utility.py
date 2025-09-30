import os
from typing import Optional

try:
	from dotenv import load_dotenv, find_dotenv
except Exception:
	load_dotenv = None
	find_dotenv = None

_ENV_LOADED = False

def ensure_env_loaded() -> None:
	global _ENV_LOADED
	if _ENV_LOADED:
		return
	if load_dotenv and find_dotenv:
		try:
			env_path = find_dotenv(usecwd=True)
			if env_path:
				load_dotenv(env_path, override=False)
		except Exception:
			pass
	key = os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
	if key and not os.getenv("GOOGLE_API_KEY"):
		os.environ["GOOGLE_API_KEY"] = key
	_ENV_LOADED = True

def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
	ensure_env_loaded()
	return os.getenv(name, default)

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