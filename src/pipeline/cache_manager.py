"""File-backed cache for expensive intermediate pipeline computations."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CachePayload = Dict[str, Any]


def _make_json_serializable(obj: Any) -> Any:
    """Convert non-JSON-native containers into serialisable structures."""
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    else:
        return obj

class PipelineCache:
    """Persist intermediate pipeline artefacts keyed by text & settings."""

    def __init__(self, cache_dir: str = "cache"):
        """Create a cache rooted at ``cache_dir``."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.intermediate_cache_file = self.cache_dir / "intermediate_pipeline_cache.json"
        self._pickle_dir = self.cache_dir / "pickles"
        self._pickle_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, text: str, settings: Optional[Dict[str, Any]] = None) -> str:
        """Derive a stable hash key using the input text and settings."""
        key_data = {
            "text": text,
            "settings": settings or {},
        }
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()

    def get_intermediate_results(
        self,
        text: str,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Optional[CachePayload]:
        """Retrieve cached intermediate results for ``text`` when available."""
        if not self.intermediate_cache_file.exists():
            return None

        try:
            with self.intermediate_cache_file.open('r', encoding='utf-8') as handle:
                cache_data = json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read cache file %s: %s", self.intermediate_cache_file, exc)
            return None

        cache_key = self._get_cache_key(text, settings)
        entry = cache_data.get(cache_key)
        if not entry:
            return None

        entry = dict(entry)

        pickle_ref = entry.get('_graph_pickle')
        if pickle_ref:
            pickle_path = self._pickle_dir / pickle_ref
            try:
                with pickle_path.open('rb') as pf:
                    graph = pickle.load(pf)
                entry['_graph'] = graph
                entry['graph'] = graph
            except Exception as exc:
                logger.warning("Failed to load cached graph %s: %s", pickle_path, exc)
                entry.pop('_graph_pickle', None)

        return entry

    def store_intermediate_results(
        self,
        text: str,
        settings: Optional[Dict[str, Any]] = None,
        intermediate_results: Optional[CachePayload] = None,
        include_graph: bool = False,
    ) -> None:
        """Persist ``intermediate_results`` for the given ``text`` and ``settings``."""
        intermediate_results = intermediate_results or {}
        cache_key = self._get_cache_key(text, settings)

        cache_data = {}
        if self.intermediate_cache_file.exists():
            try:
                with self.intermediate_cache_file.open('r', encoding='utf-8') as handle:
                    cache_data = json.load(handle)
            except (json.JSONDecodeError, OSError):
                cache_data = {}

        entry = dict(intermediate_results)
        entry['_cache_signature'] = settings or {}
        entry = _make_json_serializable(entry)
        if include_graph and 'graph' in entry:
            try:
                pickle_name = f"graph_{cache_key}.pkl"
                pickle_path = self._pickle_dir / pickle_name
                with pickle_path.open('wb') as pf:
                    pickle.dump(entry['graph'], pf)
                entry.pop('graph', None)
                entry['_graph_pickle'] = pickle_name
            except Exception as exc:
                logger.warning("Failed to persist graph pickle %s: %s", pickle_path, exc)
                entry.pop('graph', None)

        cache_data[cache_key] = entry

        try:
            with self.intermediate_cache_file.open('w', encoding='utf-8') as handle:
                json.dump(cache_data, handle, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.warning("Failed to write cache file %s: %s", self.intermediate_cache_file, exc)
            return

    def clear_cache(self) -> None:
        """Remove cached intermediate artefacts and associated pickles."""
        try:
            self.intermediate_cache_file.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Failed to delete cache file %s: %s", self.intermediate_cache_file, exc)

        if self._pickle_dir.exists():
            for pickle_file in self._pickle_dir.glob("*.pkl"):
                try:
                    pickle_file.unlink()
                except OSError as exc:
                    logger.warning("Failed to delete pickle %s: %s", pickle_file, exc)
