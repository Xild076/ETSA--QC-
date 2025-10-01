import json
import hashlib
import os
from typing import Dict, Any, Optional
from pathlib import Path

def _make_json_serializable(obj):
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    else:
        return obj

class PipelineCache:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.intermediate_cache_file = self.cache_dir / "intermediate_pipeline_cache.json"
        self._pickle_dir = self.cache_dir / "pickles"
        self._pickle_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, text: str, combiner_name: str = "", combiner_params: Dict[str, Any] = None) -> str:
        key_data = {
            "text": text,
            "combiner_name": combiner_name or "",
            "combiner_params": combiner_params or {}
        }
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()

    def get_intermediate_results(self, text: str, combiner_name: str = "", combiner_params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
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

        entry = dict(entry)

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
                entry.pop('_graph_pickle', None)

        return entry

    def store_intermediate_results(self, text: str, combiner_name: str = "", combiner_params: Dict[str, Any] = None,
                                  intermediate_results: Dict[str, Any] = None, include_graph: bool = False) -> None:
        intermediate_results = intermediate_results or {}
        cache_key = self._get_cache_key(text, combiner_name, combiner_params)

        cache_data = {}
        if self.intermediate_cache_file.exists():
            try:
                with open(self.intermediate_cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                cache_data = {}

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
                entry.pop('graph', None)
                entry['_graph_pickle'] = pickle_name
            except Exception:
                entry.pop('graph', None)

        cache_data[cache_key] = entry

        try:
            with open(self.intermediate_cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
        except IOError:
            return

    def clear_cache(self) -> None:
        if self.intermediate_cache_file.exists():
            try:
                os.remove(self.intermediate_cache_file)
            except OSError:
                pass
