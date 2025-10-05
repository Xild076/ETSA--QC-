"""Helpers for serialising pipeline cache structures to and from disk."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)

CacheKey = Tuple[Any, ...]


def serialize_cache_key(key: CacheKey) -> str:
    """Serialise tuple-like cache keys into JSON for file storage."""
    return json.dumps(key)


def _convert_nested_lists(value: Any) -> Any:
    """Convert nested list structures to tuples for immutability."""
    if isinstance(value, list):
        return tuple(_convert_nested_lists(item) for item in value)
    return value


def deserialize_cache_key(key_str: str) -> CacheKey:
    """Deserialise a JSON payload back into a tuple key (supports nesting)."""
    data = json.loads(key_str)
    if isinstance(data, list):
        return tuple(_convert_nested_lists(item) for item in data)
    raise TypeError(f"Could not deserialize cache key: {key_str}")

def load_cache_from_file(path: str | Path) -> Dict[CacheKey, Any]:
    """Load cache contents from ``path`` when it exists and is valid JSON."""
    if not path:
        return {}
    path_obj = Path(path)
    try:
        with path_obj.open("r", encoding="utf-8") as handle:
            data_from_file = json.load(handle)
            return {deserialize_cache_key(k): v for k, v in data_from_file.items()}
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        logger.info("Cache file not found or invalid at %s, starting fresh.", path_obj)
        return {}

def save_cache_to_file(path: str | Path, cache_data: Dict[CacheKey, Any]) -> None:
    """Persist ``cache_data`` to ``path`` in JSON form."""
    if not path:
        return
    path_obj = Path(path)
    try:
        if path_obj.parent:
            path_obj.parent.mkdir(parents=True, exist_ok=True)

        data_to_save = {serialize_cache_key(k): v for k, v in cache_data.items()}
        with path_obj.open("w", encoding="utf-8") as handle:
            json.dump(data_to_save, handle, indent=2)
    except Exception as e:
        logger.warning("Could not save cache to %s: %s", path_obj, e)
