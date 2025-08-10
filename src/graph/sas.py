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
        scores = [analyzer.analyze_sentiment(text) for analyzer in self.analyzers]
        return sum(scores) / len(scores) if scores else 0.0