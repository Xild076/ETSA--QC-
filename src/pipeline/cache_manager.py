import json
import hashlib
import os
from typing import Dict, Any, Optional
from pathlib import Path

def _make_json_serializable(obj):
    """Recursively convert sets to lists and other non-JSON types to strings."""
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    else:
        return obj

class PipelineCache:
    """Cache manager for intermediate pipeline results to speed up optimization runs."""

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.intermediate_cache_file = self.cache_dir / "intermediate_pipeline_cache.json"
        # We may store complex objects (like graphs) as pickles alongside the JSON index
        self._pickle_dir = self.cache_dir / "pickles"
        self._pickle_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, text: str, combiner_name: str = "", combiner_params: Dict[str, Any] = None) -> str:
        """Generate a unique cache key based on input text and optional combiner configuration.

        If combiner_name and combiner_params are omitted/empty, the key represents pre-combiner
        intermediate outputs (clauses, aspects, relation outputs, entity records) which are
        generally reusable across combiner parameter sweeps.
        """
        key_data = {
            "text": text,
            "combiner_name": combiner_name or "",
            "combiner_params": combiner_params or {}
        }
        # Use a deterministic JSON dump for stable hashing
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()

    def get_intermediate_results(self, text: str, combiner_name: str = "", combiner_params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Retrieve cached intermediate results if available.

        Returns a dictionary which may include a path to a pickled graph object under
        the key '_graph_pickle' or may include the graph object inline (if previously
        stored that way). Gracefully handles JSON decode errors and missing files.
        """
        if not self.intermediate_cache_file.exists():
            return None

        try:
            with open(self.intermediate_cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

        cache_key = self._get_cache_key(text, combiner_name, combiner_params)
        entry = cache_data.get(cache_key)
        if not entry:
            return None

        # Work on a shallow copy so callers do not accidentally mutate the cache snapshot.
        entry = dict(entry)

        # If the entry references a pickled graph, try to load it
        pickle_ref = entry.get('_graph_pickle')
        if pickle_ref:
            pickle_path = self._pickle_dir / pickle_ref
            try:
                import pickle
                with open(pickle_path, 'rb') as pf:
                    graph = pickle.load(pf)
                entry['_graph'] = graph
                entry['graph'] = graph
            except Exception:
                # If loading fails, continue without graph
                entry.pop('_graph_pickle', None)

        return entry

    def store_intermediate_results(self, text: str, combiner_name: str = "", combiner_params: Dict[str, Any] = None,
                                  intermediate_results: Dict[str, Any] = None, include_graph: bool = False) -> None:
        """Store intermediate pipeline results in cache.

        If include_graph is True and a 'graph' object is present in intermediate_results,
        the graph will be pickled separately and the JSON index will contain a reference.
        """
        intermediate_results = intermediate_results or {}
        cache_key = self._get_cache_key(text, combiner_name, combiner_params)

        # Load existing cache or create new
        cache_data = {}
        if self.intermediate_cache_file.exists():
            try:
                with open(self.intermediate_cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                cache_data = {}

        # Handle pickling of graph if requested
        entry = dict(intermediate_results)
        entry['_cached_combiner'] = combiner_name
        entry['_cached_combiner_params'] = combiner_params or {}
        entry = _make_json_serializable(entry)
        if include_graph and 'graph' in entry:
            try:
                import pickle
                pickle_name = f"graph_{cache_key}.pkl"
                pickle_path = self._pickle_dir / pickle_name
                with open(pickle_path, 'wb') as pf:
                    pickle.dump(entry['graph'], pf)
                # Remove graph object from JSON-able entry and leave a reference
                entry.pop('graph', None)
                entry['_graph_pickle'] = pickle_name
            except Exception:
                # If pickling fails, just drop the graph and continue
                entry.pop('graph', None)

        cache_data[cache_key] = entry

        # Save back to file
        try:
            with open(self.intermediate_cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
        except IOError:
            # Avoid crashing the pipeline if cache write fails
            return

    def clear_cache(self) -> None:
        """Clear all cached intermediate results."""
        if self.intermediate_cache_file.exists():
            try:
                os.remove(self.intermediate_cache_file)
            except OSError:
                pass
