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

