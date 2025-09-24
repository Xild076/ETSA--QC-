import sys
from pathlib import Path

# If the module is executed directly, ensure project root is on sys.path
_this_file = Path(__file__).resolve()
_project_root = _this_file.parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from src.sentiment.sentiment_analysis_complex import (
        get_vader_sentiment, get_textblob_sentiment, get_flair_sentiment,
        get_pysentimiento_sentiment, get_nlptown_sentiment,
        get_finiteautomata_sentiment, get_ProsusAI_sentiment, get_distilbert_logit_sentiment
    )
except ImportError:
    from sentiment.sentiment_analysis_complex import (
        get_vader_sentiment, get_textblob_sentiment, get_flair_sentiment,
        get_pysentimiento_sentiment, get_nlptown_sentiment,
        get_finiteautomata_sentiment, get_ProsusAI_sentiment, get_distilbert_logit_sentiment
    )
from typing import List, Literal, Dict

class SentimentAnalysis:
    def analyze(self, text: str) -> dict:
        raise NotImplementedError
    
class DummySentimentAnalysis(SentimentAnalysis):
    def analyze(self, text: str) -> dict:
        return {"aggregate": 0.0, "per_method_scores": {}, "raw": {}}

class VADERSentimentAnalysis(SentimentAnalysis):
    def analyze(self, text: str) -> dict:
        dist = get_vader_sentiment(text)
        return {"aggregate": dist['positive'] - dist['negative'], "distribution": dist}

class TextBlobSentimentAnalysis(SentimentAnalysis):
    def analyze(self, text: str) -> dict:
        dist = get_textblob_sentiment(text)
        return {"aggregate": dist['positive'] - dist['negative'], "distribution": dist}

class FlairSentimentAnalysis(SentimentAnalysis):
    def analyze(self, text: str) -> dict:
        dist = get_flair_sentiment(text)
        return {"aggregate": dist['positive'] - dist['negative'], "distribution": dist}

class PysentimientoSentimentAnalysis(SentimentAnalysis):
    def analyze(self, text: str) -> dict:
        dist = get_pysentimiento_sentiment(text)
        return {"aggregate": dist['positive'] - dist['negative'], "distribution": dist}

# --- START: IMPROVEMENT FOR ROOT CAUSE 2 ---
class MultiSentimentAnalysis(SentimentAnalysis):
    def __init__(self, methods: List[Literal['vader', 'textblob', 'flair', 'pysentimiento', 'nlptown', 'finiteautomata', 'prosusai', 'distilbert_logit']], weights: List[float] = None):
        self.methods = methods
        
        # New, more intelligent weighting scheme.
        # Prioritizes robust models (distilbert, flair) and de-prioritizes weaker ones (textblob, swn).
        if weights is None:
            default_weights = {
                'distilbert_logit': 0.35,
                'flair': 0.20,
                'pysentimiento': 0.15,
                'vader': 0.12,
                'nlptown': 0.10,
                'finiteautomata': 0.03,
                'prosusai': 0.03,
                'textblob': 0.02,
            }
            self.weights = [default_weights.get(m, 0.0) for m in self.methods]
        else:
            self.weights = weights

        self.method_funcs = {
            'vader': get_vader_sentiment, 'textblob': get_textblob_sentiment,
            'flair': get_flair_sentiment, 'pysentimiento': get_pysentimiento_sentiment,
            'nlptown': get_nlptown_sentiment,
            'finiteautomata': get_finiteautomata_sentiment, 'prosusai': get_ProsusAI_sentiment,
            'distilbert_logit': get_distilbert_logit_sentiment
        }
    
    def analyze(self, text: str) -> dict:
        if not text or not text.strip():
            return {"aggregate": 0.0, "per_method_scores": {}, "raw": {}}
            
        if len(self.weights) != len(self.methods):
            raise ValueError("Length of weights must equal length of methods")

        raw_results = {}
        weighted_pos, weighted_neg, weighted_neu = 0.0, 0.0, 0.0
        total_weight = 0.0
        per_method_scores = {}

        for i, method in enumerate(self.methods):
            func = self.method_funcs.get(method)
            weight = self.weights[i]
            if not func or weight == 0.0:
                continue

            try:
                # All `get_*` functions now return a standardized distribution dict.
                dist = func(text)
                raw_results[method] = dist
                
                pos, neg, neu = dist.get('positive', 0.0), dist.get('negative', 0.0), dist.get('neutral', 0.0)
                
                weighted_pos += pos * weight
                weighted_neg += neg * weight
                weighted_neu += neu * weight
                total_weight += weight
                
                # For logging/debugging, store the individual valence score
                per_method_scores[method] = pos - neg
                
            except Exception as e:
                raw_results[method] = {"error": str(e)}
                continue

        if total_weight > 0:
            # Normalize the aggregated probabilities
            final_pos = weighted_pos / total_weight
            final_neg = weighted_neg / total_weight
            
            # The final aggregate score is the difference between positive and negative evidence.
            aggregate = final_pos - final_neg
            return {"aggregate": aggregate, "per_method_scores": per_method_scores, "raw": raw_results}
        
        return {"aggregate": 0.0, "per_method_scores": {}, "raw": raw_results}

# --- END: IMPROVEMENT FOR ROOT CAUSE 2 ---