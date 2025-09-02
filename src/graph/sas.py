import os, sys
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.sentiment.sentiment import get_distilbert_logit_sentiment, get_finiteautomata_sentiment, get_flair_sentiment
from src.sentiment.sentiment import get_nlptown_sentiment, get_pysentimiento_sentiment, get_swn_sentiment
from src.sentiment.sentiment import get_textblob_sentiment, get_vader_sentiment, get_ProsusAI_sentiment
from typing import List

class SentimentAnalyzerSystem:
    def __init__(self):
        pass

    def analyze_sentiment(self, text):
        raise NotImplementedError("This method should be overridden by subclasses")

class VADERSentimentAnalyzer(SentimentAnalyzerSystem):
    def analyze_sentiment(self, text):
        return get_vader_sentiment(text)

class SpacyENCOREWEBSMSentimentAnalyzer(SentimentAnalyzerSystem):
    def __init__(self):
        try:
            import spacy
            self.nlp = None
            for m in ("en_core_web_sm", "en_core_web_lg"):
                try:
                    self.nlp = spacy.load(m)
                    break
                except Exception:
                    continue
        except Exception:
            self.nlp = None
    def analyze_sentiment(self, text):
        if not getattr(self, "nlp", None):
            return 0.0
        doc = self.nlp(text)
        return sum(token.sentiment for token in doc) / len(doc) if doc else 0.0

class DISTILBERTLOGITSentimentAnalyzer(SentimentAnalyzerSystem):
    def analyze_sentiment(self, text):
        return get_distilbert_logit_sentiment(text)

class FiniteautomataSentimentAnalyzer(SentimentAnalyzerSystem):
    def analyze_sentiment(self, text):
        return get_finiteautomata_sentiment(text)

class FlairSentimentAnalyzer(SentimentAnalyzerSystem):
    def analyze_sentiment(self, text):
        return get_flair_sentiment(text)

class NLPTownSentimentAnalyzer(SentimentAnalyzerSystem):
    def analyze_sentiment(self, text):
        return get_nlptown_sentiment(text)

class PySentimientoSentimentAnalyzer(SentimentAnalyzerSystem):
    def analyze_sentiment(self, text):
        return get_pysentimiento_sentiment(text)

class SWNSentimentAnalyzer(SentimentAnalyzerSystem):
    def analyze_sentiment(self, text):
        return get_swn_sentiment(text)

class TextBlobSentimentAnalyzer(SentimentAnalyzerSystem):
    def analyze_sentiment(self, text):
        return get_textblob_sentiment(text)

class ProsusAISentimentAnalyzer(SentimentAnalyzerSystem):
    def analyze_sentiment(self, text):
        return get_ProsusAI_sentiment(text)

class WeightedEnsembleSentimentAnalyzer(SentimentAnalyzerSystem):
    def __init__(self, analyzers: List[SentimentAnalyzerSystem], weights: List[float]):
        self.analyzers = analyzers
        self.weights = weights

    def analyze_sentiment(self, text):
        scores = [
            analyzer.analyze_sentiment(text) * weight
            for analyzer, weight in zip(self.analyzers, self.weights)
        ]
        return sum(scores) / sum(self.weights) if scores else 0.0

class EnsembleSentimentAnalyzer(SentimentAnalyzerSystem):
    def __init__(self, analyzers: List[SentimentAnalyzerSystem]):
        self.analyzers = analyzers

    def analyze_sentiment(self, text):
        scores = [analyzer.analyze_sentiment(text) for analyzer in self.analyzers]
        return sum(scores) / len(scores) if scores else 0.0

class VASentimentAnalyzer(SentimentAnalyzerSystem):
    def __init__(self):
        self.analyzers = [
            VADERSentimentAnalyzer(),
            SWNSentimentAnalyzer(),
            TextBlobSentimentAnalyzer(),
            SpacyENCOREWEBSMSentimentAnalyzer()
        ]
    
    def analyze_sentiment(self, text):
        scores = []
        for analyzer in self.analyzers:
            try:
                score = analyzer.analyze_sentiment(text)
                if score is not None and not (isinstance(score, float) and (score != score)):
                    scores.append(score)
            except Exception:
                continue
        
        return max(scores, key=abs) if scores else 0.0

class PresetEnsembleSentimentAnalyzer(SentimentAnalyzerSystem):
    def __init__(self):
        self.analyzers = [
            VADERSentimentAnalyzer(),
            SpacyENCOREWEBSMSentimentAnalyzer(),
            DISTILBERTLOGITSentimentAnalyzer(),
            FiniteautomataSentimentAnalyzer(),
            FlairSentimentAnalyzer(),
            NLPTownSentimentAnalyzer(),
            PySentimientoSentimentAnalyzer(),
            SWNSentimentAnalyzer(),
            TextBlobSentimentAnalyzer(),
            ProsusAISentimentAnalyzer()
        ]

    def analyze_sentiment(self, text):
        scores = []
        for analyzer in self.analyzers:
            try:
                s = analyzer.analyze_sentiment(text)
            except Exception:
                continue
            if s is None:
                continue
            try:
                s = float(s)
            except Exception:
                continue
            if s != s:
                continue
            if s > 1:
                s = 1.0
            if s < -1:
                s = -1.0
            scores.append(s)
        if not scores:
            return 0.0
        pos = [s for s in scores if s > 0]
        neg = [s for s in scores if s < 0]
        zeros = [s for s in scores if s == 0]
        import numpy as np
        def median_abs(vals):
            return float(np.median(np.abs(vals))) if vals else 0.0
        # Neutral deadzone if overall signal is weak and votes are split
        overall_med = median_abs(scores)
        if abs(len(pos) - len(neg)) <= 1 and overall_med < 0.15:
            return 0.0
        # Majority sign, magnitude by median of that group
        if len(pos) > len(neg):
            mag = median_abs(pos)
            return mag
        elif len(neg) > len(pos):
            mag = median_abs(neg)
            return -mag
        # Tie: fall back to trimmed mean near zero
        trimmed = sorted(scores)
        k = max(1, int(0.15 * len(trimmed)))
        core = trimmed[k:-k] if len(trimmed) > 2 * k else trimmed
        mean = float(np.mean(core)) if core else float(np.mean(scores))
        # small tie goes neutral if magnitude tiny
        return 0.0 if abs(mean) < 0.1 else mean