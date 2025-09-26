import sys
from pathlib import Path

# If the module is executed directly (python src/pipeline/sentiment_analysis.py),
# make sure the project root is on sys.path so package imports succeed.
_this_file = Path(__file__).resolve()
# parents[0] -> pipeline, parents[1] -> src, parents[2] -> project root
_project_root = _this_file.parents[2]
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
    # Fallback for alternate PYTHONPATH layouts
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


import pprint
def run_stress_tests():
    """
    Runs a series of stress tests against the MultiSentimentAnalysis class
    to evaluate the performance of different sentiment analysis models.
    """
    # Define the test cases, categorized by linguistic challenge.
    # Each tuple contains (sentence, expected_outcome).
    test_cases = {
        "Sarcasm and Irony": [
            ("Oh great, my flight was cancelled. Just what I needed.", "Negative"),
            ("I love spending my entire Saturday fixing a bug. It's so relaxing.", "Negative"),
            ("The movie was a real masterpiece of boredom.", "Negative")
        ],
        "Complex Negation and Qualifiers": [
            ("The service wasn't bad.", "Neutral or slightly Positive"),
            ("I wouldn't say it's the best pizza I've ever had.", "Negative"),
            ("It's hardly a five-star experience.", "Negative")
        ],
        "Ambiguity and Slang": [
            ("This new phone's camera is sick!", "Positive"),
            ("I felt sick after eating the fish.", "Negative"),
            ("The plot was completely predictable.", "Negative")
        ],
        "Mixed Sentiment": [
            ("The location was amazing, but the service was incredibly slow.", "Mixed/Neutral or Negative"),
            ("While the acting was superb, the story was a complete letdown.", "Mixed/Neutral or Negative")
        ],
        "Questions and Neutral Statements": [
            ("What time does the restaurant open?", "Neutral"),
            ("This report was generated on a Tuesday.", "Neutral"),
            ("Is this the new model?", "Neutral")
        ],
        "Minimal Pairs": [
            ("I quite like the food.", "Positive"),
            ("I quit liking the food.", "Negative")
        ]
    }

    # --- Initialize your sentiment analyzer ---
    # Using the original weights you provided
    original_weights = [0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.12]
    msa = MultiSentimentAnalysis(
        methods=['vader', 'textblob', 'flair', 'pysentimiento', 'swn', 'nlptown', 'finiteautomata', 'prosusai', 'distilbert_logit'],
        weights=original_weights
    )

    # --- (Optional) Initialize with recommended weights to compare ---
    # recommended_weights = [0.05, 0.05, 0.15, 0.10, 0.05, 0.10, 0.05, 0.05, 0.40]
    # msa_recommended = MultiSentimentAnalysis(
    #     methods=['vader', 'textblob', 'flair', 'pysentimiento', 'swn', 'nlptown', 'finiteautomata', 'prosusai', 'distilbert_logit'],
    #     weights=recommended_weights
    # )

    print("="*60)
    print("      SENTIMENT ANALYSIS MODEL STRESS TEST      ")
    print("="*60)

    # --- Run and Print Results ---
    for category, sentences in test_cases.items():
        print(f"\n\n===== Testing Category: {category.upper()} =====\n")
        for sentence, expected in sentences:
            print(f"--- Text: '{sentence}' ---")
            print(f"Expected Sentiment: {expected}")

            # Analyze the text
            results = msa.analyze(sentence)

            # Print formatted results
            print(f"Aggregate Score: {results['aggregate']:.4f}")
            print("Per-Method Scores:")
            pprint.pprint(results['per_method_scores'])
            print("-" * 50)

if __name__ == "__main__":
    run_stress_tests()
