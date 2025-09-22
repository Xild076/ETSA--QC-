from sentiment.sentiment import get_vader_sentiment
from sentiment.sentiment import get_textblob_sentiment
from sentiment.sentiment import get_flair_sentiment
from sentiment.sentiment import get_pysentimiento_sentiment
from sentiment.sentiment import get_swn_sentiment
from sentiment.sentiment import get_nlptown_sentiment
from sentiment.sentiment import get_finiteautomata_sentiment
from sentiment.sentiment import get_ProsusAI_sentiment
from sentiment.sentiment import get_distilbert_logit_sentiment
from typing import List, Literal

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
    def __init__(self, methods: List[Literal['vader', 'textblob', 'flair', 'pysentimiento', 'swn', 'nlptown', 'finiteautomata', 'prosusai', 'distilbert_logit']], weights: List[float] = None):
        self.methods = methods
        self.weights = weights if weights else [1.0] * len(methods)
        self.method_funcs = {
            'vader': get_vader_sentiment,
            'textblob': get_textblob_sentiment,
            'flair': get_flair_sentiment,
            'pysentimiento': get_pysentimiento_sentiment,
            'swn': get_swn_sentiment,
            'nlptown': get_nlptown_sentiment,
            'finiteautomata': get_finiteautomata_sentiment,
            'prosusai': get_ProsusAI_sentiment,
            'distilbert_logit': get_distilbert_logit_sentiment
        }
    
    def analyze(self, text: str) -> dict:
        # Basic validation of weights
        if self.weights and len(self.weights) != len(self.methods):
            raise ValueError("Length of weights must equal length of methods")

        results = {}
        weighted_sum = 0.0
        total_weight = 0.0
        per_method_scores = {}

        for i, method in enumerate(self.methods):
            func = self.method_funcs.get(method)
            weight = self.weights[i] if i < len(self.weights) else 0.0
            if func is None or weight == 0.0:
                # Record missing function or zero weight but continue
                results[method] = None if func is None else func
                continue

            try:
                raw = func(text)
            except Exception as e:
                # Don't let a single method failure break the whole aggregation
                results[method] = {"error": str(e)}
                continue

            results[method] = raw

            # Try to extract a numeric score from the method result
            try:
                score = self._score_from_result(raw)
            except ValueError:
                # If a numeric score can't be extracted, skip it for aggregation
                continue

            per_method_scores[method] = score
            weighted_sum += score * weight
            total_weight += weight

        if total_weight > 0:
            aggregate = weighted_sum / total_weight
            return {"aggregate": aggregate, "per_method_scores": per_method_scores, "raw": results}
        # If no numeric scores were available, return raw results
        return {"per_method": results}

    def _score_from_result(self, res) -> float:
        """Extract a numeric score from various result shapes returned by sentiment methods.

        Supports:
        - numeric returns (int/float)
        - dict returns with common keys like 'score', 'compound', 'polarity', 'sentiment'
        - dict returns where one of the values is numeric

        Raises ValueError if no numeric score can be found.
        """
        if isinstance(res, (int, float)):
            return float(res)
        if isinstance(res, dict):
            # Common keys to check
            for key in ("score", "compound", "polarity", "sentiment", "confidence", "confidence_score"):
                if key in res and isinstance(res[key], (int, float)):
                    return float(res[key])
            # Some libraries return {'label': 'POS', 'score': 0.9}
            if "score" in res and isinstance(res.get("score"), (int, float)):
                return float(res.get("score"))
            # Fallback: scan values for a numeric
            for v in res.values():
                if isinstance(v, (int, float)):
                    return float(v)
        raise ValueError("No numeric score found in method result")