import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from src.sentiment.sentiment import (
        get_vader_sentiment,
        get_textblob_sentiment,
        get_flair_sentiment,
        get_pysentimiento_sentiment,
        get_swn_sentiment,
        get_nlptown_sentiment,
        get_finiteautomata_sentiment,
        get_ProsusAI_sentiment,
        get_distilbert_logit_sentiment,
    )
except Exception:
    from sentiment.sentiment import (
        get_vader_sentiment,
        get_textblob_sentiment,
        get_flair_sentiment,
        get_pysentimiento_sentiment,
        get_swn_sentiment,
        get_nlptown_sentiment,
        get_finiteautomata_sentiment,
        get_ProsusAI_sentiment,
        get_distilbert_logit_sentiment,
    )

class SentimentAnalysis:
    def analyze(self, text: str) -> dict:
        raise NotImplementedError
    
class DummySentimentAnalysis(SentimentAnalysis):
    def analyze(self, text: str) -> dict:
        return 0.0

class VADERSentimentAnalysis(SentimentAnalysis):
    def analyze(self, text: str) -> dict:
        return get_vader_sentiment(text)

class TextBlobSentimentAnalysis(SentimentAnalysis):
    def analyze(self, text: str) -> dict:
        return get_textblob_sentiment(text)

class FlairSentimentAnalysis(SentimentAnalysis):
    def analyze(self, text: str) -> dict:
        return get_flair_sentiment(text)

class PysentimientoSentimentAnalysis(SentimentAnalysis):
    def analyze(self, text: str) -> dict:
        return get_pysentimiento_sentiment(text)


class MultiSentimentAnalysis(SentimentAnalysis):
    NEGATION_PATTERNS = {
        'not', 'no', 'never', 'nothing', "n't", 'dont', "doesn't", "didn't",
        'cannot', "can't", 'wont', "won't", 'neither', 'nor', 'without', 'none',
        'nowhere', 'nobody', 'hardly', 'barely', 'scarcely'
    }
    
    def __init__(
        self,
        methods: List[Literal[
            'vader',
            'textblob',
            'flair',
            'pysentimiento',
            'swn',
            'nlptown',
            'finiteautomata',
            'prosusai',
            'distilbert_logit'
        ]],
        weights: Optional[List[float]] = None,
        use_adaptive_weighting: bool = True,
    ) -> None:
        self.methods = list(methods)
        resolved_weights = weights or [1.0] * len(methods)
        if len(resolved_weights) != len(self.methods):
            raise ValueError("Length of weights must equal length of methods")
        self.base_weights = [float(w) for w in resolved_weights]
        self.weights = list(self.base_weights)
        self.use_adaptive_weighting = use_adaptive_weighting
        self.method_funcs = {
            'vader': get_vader_sentiment,
            'textblob': get_textblob_sentiment,
            'flair': get_flair_sentiment,
            'pysentimiento': get_pysentimiento_sentiment,
            'swn': get_swn_sentiment,
            'nlptown': get_nlptown_sentiment,
            'finiteautomata': get_finiteautomata_sentiment,
            'prosusai': get_ProsusAI_sentiment,
            'distilbert_logit': get_distilbert_logit_sentiment,
        }
        self.method_strengths = {
            'vader': {'negation': 0.9, 'general': 0.8},
            'textblob': {'negation': 0.3, 'general': 0.7},
            'distilbert_logit': {'negation': 0.95, 'general': 0.9},
            'flair': {'negation': 0.85, 'general': 0.85},
            'pysentimiento': {'negation': 0.8, 'general': 0.8},
        }
        self._sentiment_cache: Dict[Tuple[str, Tuple[str, ...], Tuple[float, ...]], Dict[str, Any]] = {}
        self._cache_max_size = 5000
        self._batch_cache: Dict[str, Any] = {}
    
    def _detect_negation(self, text: str) -> bool:
        tokens = text.lower().split()
        text_lower = text.lower()
        return any(neg in tokens or neg in text_lower for neg in self.NEGATION_PATTERNS)
    
    def _get_adaptive_weights(self, text: str) -> List[float]:
        if not self.use_adaptive_weighting:
            return list(self.base_weights)
        
        has_negation = self._detect_negation(text)
        adaptive_weights = []
        
        for i, method in enumerate(self.methods):
            base_weight = self.base_weights[i]
            method_info = self.method_strengths.get(method, {'negation': 0.7, 'general': 0.7})
            
            if has_negation:
                strength_factor = method_info['negation']
            else:
                strength_factor = method_info['general']
            
            adaptive_weights.append(base_weight * strength_factor)
        
        total = sum(adaptive_weights)
        if total > 0:
            adaptive_weights = [w / total * sum(self.base_weights) for w in adaptive_weights]
        
        return adaptive_weights

    def analyze(self, text: str) -> dict:
        if not text or not text.strip():
            return {"aggregate": 0.0, "confidence": 0.0, "per_method": {}, "has_negation": False}
            
        cache_key = self._cache_key(text)
        if cache_key in self._sentiment_cache:
            return self._sentiment_cache[cache_key]

        has_negation = self._detect_negation(text)
        adaptive_weights = self._get_adaptive_weights(text)

        results: Dict[str, Any] = {}
        weighted_sum = 0.0
        total_weight = 0.0
        per_method_scores: Dict[str, float] = {}
        valid_scores = []

        for i, method in enumerate(self.methods):
            func = self.method_funcs.get(method)
            weight = adaptive_weights[i]
            if func is None:
                results[method] = {"skipped": "missing_function"}
                continue
            if weight == 0.0:
                results[method] = {"skipped": "zero_weight"}
                continue

            try:
                raw = func(text)
            except Exception as e:
                results[method] = {"error": str(e)}
                continue

            results[method] = raw

            try:
                score = self._score_from_result(raw)
                # Validate score is finite and reasonable
                if not isinstance(score, (int, float)) or not (-1.5 <= score <= 1.5):
                    continue
                score = max(min(float(score), 1.0), -1.0)  # Clamp to valid range
            except ValueError:
                continue

            per_method_scores[method] = score
            valid_scores.append(abs(score))  # For confidence calculation
            weighted_sum += score * weight
            total_weight += weight

        if total_weight > 0:
            aggregate = weighted_sum / total_weight
            aggregate = max(min(aggregate, 1.0), -1.0)
            
            if has_negation and aggregate > 0:
                aggregate *= 0.85
            elif has_negation and aggregate < 0:
                aggregate *= 1.15
            
            aggregate = max(min(aggregate, 1.0), -1.0)
            
            confidence = min(sum(valid_scores) / max(len(valid_scores), 1), 1.0) if valid_scores else 0.0
            final_result = {
                "aggregate": aggregate, 
                "confidence": confidence,
                "per_method_scores": per_method_scores, 
                "has_negation": has_negation,
                "adaptive_weights_used": adaptive_weights,
                "raw": results
            }
        else:
            final_result = {"aggregate": 0.0, "confidence": 0.0, "per_method": results, "has_negation": has_negation}

        # Improved cache management
        if len(self._sentiment_cache) >= self._cache_max_size:
            # Remove oldest 20% of cache entries for better performance
            keys_to_remove = list(self._sentiment_cache.keys())[:max(1, self._cache_max_size // 5)]
            for key in keys_to_remove:
                del self._sentiment_cache[key]
        self._sentiment_cache[cache_key] = final_result

        return final_result

    def _cache_key(self, text: str) -> Tuple[str, Tuple[str, ...], Tuple[float, ...]]:
        normalized = text.strip()
        return normalized, tuple(self.methods), tuple(self.weights)

    def _score_from_result(self, res: Any) -> float:
        if isinstance(res, (int, float)):
            return float(res)
        if isinstance(res, dict):
            for key in ("score", "compound", "polarity", "sentiment", "confidence", "confidence_score"):
                if key in res and isinstance(res[key], (int, float)):
                    return float(res[key])
            if "score" in res and isinstance(res.get("score"), (int, float)):
                return float(res.get("score"))
            for v in res.values():
                if isinstance(v, (int, float)):
                    return float(v)
        raise ValueError("No numeric score found in method result")
