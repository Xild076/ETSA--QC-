import json
import os
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

def serialize_cache_key(key: Tuple) -> str:
    """Serializes tuple-like keys into a JSON string for file storage."""
    return json.dumps(key)


def _convert_nested_lists(value):
    if isinstance(value, list):
        return tuple(_convert_nested_lists(item) for item in value)
    return value


def deserialize_cache_key(key_str: str) -> Tuple:
    """Deserializes a JSON string back into a tuple key (supports nested tuples)."""
    data = json.loads(key_str)
    if isinstance(data, list):
        return tuple(_convert_nested_lists(item) for item in data)
    raise TypeError(f"Could not deserialize cache key: {key_str}")

def load_cache_from_file(path: str) -> Dict[Tuple, Any]:
    """Loads a cache dictionary from a JSON file."""
    if not path:
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data_from_file = json.load(f)
            return {deserialize_cache_key(k): v for k, v in data_from_file.items()}
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        logger.info(f"Cache file not found or invalid at {path}, starting fresh.")
        return {}

def save_cache_to_file(path: str, cache_data: Dict[Tuple, Any]):
    """Saves a cache dictionary to a JSON file."""
    if not path:
        return
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        data_to_save = {serialize_cache_key(k): v for k, v in cache_data.items()}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save cache to {path}: {e}")
