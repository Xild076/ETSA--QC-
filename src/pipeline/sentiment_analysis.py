"""Wrappers for the various sentiment analysis back-ends used by the pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

try:  # pragma: no cover - support package and script execution
    from ..sentiment.sentiment import (
        get_ProsusAI_sentiment,
        get_distilbert_logit_sentiment,
        get_finiteautomata_sentiment,
        get_flair_sentiment,
        get_nlptown_sentiment,
        get_pysentimiento_sentiment,
        get_swn_sentiment,
        get_textblob_sentiment,
        get_vader_sentiment,
    )
except ImportError:  # pragma: no cover - fallback when package context unavailable
    from sentiment.sentiment import (  # type: ignore
        get_ProsusAI_sentiment,
        get_distilbert_logit_sentiment,
        get_finiteautomata_sentiment,
        get_flair_sentiment,
        get_nlptown_sentiment,
        get_pysentimiento_sentiment,
        get_swn_sentiment,
        get_textblob_sentiment,
        get_vader_sentiment,
    )

SentimentPayload = Dict[str, Any]

__all__ = [
    "SentimentAnalysis",
    "DummySentimentAnalysis",
    "VADERSentimentAnalysis",
    "TextBlobSentimentAnalysis",
    "FlairSentimentAnalysis",
    "PysentimientoSentimentAnalysis",
    "MultiSentimentAnalysis",
]


class SentimentAnalysis:
    """Base interface for sentiment back-ends used in the pipeline."""

    def analyze(self, text: str) -> SentimentPayload:
        """Return a structured sentiment payload for ``text``."""
        raise NotImplementedError


class DummySentimentAnalysis(SentimentAnalysis):
    """No-op sentiment analyzer that always returns a neutral score."""

    def analyze(self, text: str) -> SentimentPayload:
        return {"aggregate": 0.0, "raw": {}}


class VADERSentimentAnalysis(SentimentAnalysis):
    """Adapter around the VADER sentiment implementation."""

    def analyze(self, text: str) -> SentimentPayload:
        return get_vader_sentiment(text)


class TextBlobSentimentAnalysis(SentimentAnalysis):
    """Adapter around TextBlob sentiment output."""

    def analyze(self, text: str) -> SentimentPayload:
        return get_textblob_sentiment(text)


class FlairSentimentAnalysis(SentimentAnalysis):
    """Adapter around the Flair sentiment classifier."""

    def analyze(self, text: str) -> SentimentPayload:
        return get_flair_sentiment(text)


class PysentimientoSentimentAnalysis(SentimentAnalysis):
    """Adapter around the pysentimiento sentiment pipeline."""

    def analyze(self, text: str) -> SentimentPayload:
        return get_pysentimiento_sentiment(text)


class MultiSentimentAnalysis(SentimentAnalysis):
    """Blend multiple sentiment back-ends using weighted aggregation."""
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
    ) -> None:
        """Configure which sentiment engines to run and how to weight them."""
        self.methods = list(methods)
        resolved_weights = weights or [1.0] * len(methods)
        if len(resolved_weights) != len(self.methods):
            raise ValueError("Length of weights must equal length of methods")
        self.weights = [float(w) for w in resolved_weights]
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
        self._sentiment_cache: Dict[Tuple[str, Tuple[str, ...], Tuple[float, ...]], SentimentPayload] = {}
        self._cache_max_size = 2000
    
    def analyze(self, text: str) -> SentimentPayload:
        """Return a weighted aggregate sentiment across configured methods."""
        if not text or not text.strip():
            return {"aggregate": 0.0, "per_method_scores": {}, "raw": {}}
            
        cache_key = self._cache_key(text)
        if cache_key in self._sentiment_cache:
            return self._sentiment_cache[cache_key]

        results: Dict[str, Any] = {}
        weighted_sum = 0.0
        total_weight = 0.0
        per_method_scores: Dict[str, float] = {}

        for index, method in enumerate(self.methods):
            func = self.method_funcs.get(method)
            weight = self.weights[index]
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
                if not isinstance(score, (int, float)) or not (-1.5 <= score <= 1.5):
                    continue
                score = max(min(float(score), 1.0), -1.0)
            except ValueError:
                continue

            per_method_scores[method] = score
            weighted_sum += score * weight
            total_weight += weight

        if total_weight > 0:
            aggregate = weighted_sum / total_weight
            aggregate = max(min(aggregate, 1.0), -1.0)
        else:
            aggregate = 0.0

        final_result: SentimentPayload = {
            "aggregate": aggregate,
            "per_method_scores": per_method_scores,
            "raw": results,
        }

        if len(self._sentiment_cache) >= self._cache_max_size:
            oldest_key = next(iter(self._sentiment_cache))
            del self._sentiment_cache[oldest_key]
        self._sentiment_cache[cache_key] = final_result

        return final_result

    def _cache_key(self, text: str) -> Tuple[str, Tuple[str, ...], Tuple[float, ...]]:
        """Create a stable cache key for the given text and configuration."""
        normalized = text.strip()
        return normalized, tuple(self.methods), tuple(self.weights)

    def _score_from_result(self, res: Any) -> float:
        """Extract a numeric sentiment score from arbitrary model output."""
        if isinstance(res, (int, float)):
            return float(res)
        if isinstance(res, dict):
            for key in ("score", "compound", "polarity", "sentiment", "confidence", "confidence_score"):
                value = res.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
            for v in res.values():
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, dict):
                    try:
                        return self._score_from_result(v)
                    except ValueError:
                        continue
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, (int, float)):
                            return float(item)
                        if isinstance(item, dict):
                            try:
                                return self._score_from_result(item)
                            except ValueError:
                                continue
        raise ValueError("No numeric score found in method result")
