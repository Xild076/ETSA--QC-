import os
import ssl
from functools import lru_cache
from typing import Literal, List, Dict, Any, Optional
from rich.console import Console
import math
import torch
import re
from textblob import TextBlob
from flair.data import Sentence
from flair.models import TextClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pysentimiento import create_analyzer
import numpy as np
from transformers import pipeline
import hashlib

import logging

_ENV_DEFAULTS = {
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "OMP_NUM_THREADS": "1",
    "KMP_AFFINITY": "disabled",
    "KMP_INIT_AT_FORK": "FALSE",
}
for key, value in _ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

ssl._create_default_https_context = ssl._create_unverified_context
console = Console(width=120)
try:
    import nltk
    from nltk.corpus import sentiwordnet as swn, wordnet
    import spacy
    from wordfreq import word_frequency
    nltk.download('sentiwordnet', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    NLP = spacy.load("en_core_web_lg")
except (ImportError, OSError):
    nltk, swn, wordnet, spacy, NLP, word_frequency = None, None, None, None, None, None
    console.print("[bold red]Warning: Key NLP libraries (nltk, spacy, wordfreq) or models not found.[/bold red]")
    console.print("         Functionality like WordNet expansion, validation, and cohesion scoring will be disabled.")
    console.print("         To enable all features, run: [cyan]pip install nltk spacy wordfreq torch textblob flair vaderSentiment pysentimiento transformers rich[/cyan]")
    console.print("         Then download the SpaCy model: [cyan]python -m spacy download en_core_web_lg[/cyan]")


_VADER_ANALYZER: Optional[SentimentIntensityAnalyzer] = None
_PYSENTIMIENTO_ANALYZER = None
_HF_PIPELINES: Dict[str, Any] = {}


def _get_hf_pipeline(model_name: str) -> Any:
    if model_name not in _HF_PIPELINES:
        _HF_PIPELINES[model_name] = pipeline("sentiment-analysis", model=model_name)
    return _HF_PIPELINES[model_name]


def _get_vader_analyzer() -> SentimentIntensityAnalyzer:
    global _VADER_ANALYZER
    if _VADER_ANALYZER is None:
        _VADER_ANALYZER = SentimentIntensityAnalyzer()
    return _VADER_ANALYZER


def _get_pysentimiento_analyzer():
    global _PYSENTIMIENTO_ANALYZER
    if _PYSENTIMIENTO_ANALYZER is None:
        _PYSENTIMIENTO_ANALYZER = create_analyzer(task="sentiment", lang="en")
    return _PYSENTIMIENTO_ANALYZER


_DEFAULT_POS_THRESHOLD = 0.1
_DEFAULT_NEG_THRESHOLD = -0.1


def _normalize_score(score: Any) -> float:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(min(value, 1.0), -1.0)


def _score_to_label(score: float, pos_threshold: float = _DEFAULT_POS_THRESHOLD, neg_threshold: float = _DEFAULT_NEG_THRESHOLD) -> str:
    if score >= pos_threshold:
        return "positive"
    if score <= neg_threshold:
        return "negative"
    return "neutral"


def _build_sentiment_result(
    score: Any,
    *,
    confidence: Optional[float] = None,
    raw: Optional[Any] = None,
    pos_threshold: float = _DEFAULT_POS_THRESHOLD,
    neg_threshold: float = _DEFAULT_NEG_THRESHOLD,
) -> Dict[str, Any]:
    normalized = _normalize_score(score)
    if confidence is None or not isinstance(confidence, (float, int)) or math.isnan(confidence):
        conf_val = abs(normalized)
    else:
        conf_val = float(confidence)
    conf_val = max(0.0, min(conf_val, 1.0))
    return {
        "score": normalized,
        "label": _score_to_label(normalized, pos_threshold, neg_threshold),
        "confidence": conf_val,
        "raw": raw,
    }

def _convert_confidence_to_valence(confidence_score: float, prediction_label: str, method:Literal['linear_transform', 'log_odds', 'tanh_scaling']) -> float:
    if prediction_label.upper() in ['NEGATIVE', 'NEG']:
        confidence_score = -confidence_score
    
    if method == "linear_transform":
        return confidence_score * 2 - 1 if confidence_score >= 0 else confidence_score * 2 + 1
    
    elif method == "log_odds":
        if abs(confidence_score) >= 0.999: 
            confidence_score = 0.999 * (1 if confidence_score > 0 else -1)
        
        log_odds = math.log(abs(confidence_score) / (1 - abs(confidence_score)))
        valence = math.tanh(log_odds / 3.0)
        return valence if confidence_score >= 0 else -valence
    
    elif method == "tanh_scaling":
        scaled = confidence_score * 3.0 
        return math.tanh(scaled)
    
    else:
        return confidence_score * 2 - 1 if confidence_score >= 0 else confidence_score * 2 + 1

def _get_hf_pipeline_score(result: List[Dict[str, Any]], is_confidence_only: bool, model_name:str) -> float:
    if is_confidence_only:
        if len(result) == 1:
            item = result[0]
            return _convert_confidence_to_valence(item['score'], item['label'], 'linear_transform')
        else:
            score_map = {item['label'].lower().replace('pos', 'positive').replace('neg', 'negative').replace('neu', 'neutral'): item['score'] for item in result}
            pos_conf = score_map.get('positive', 0.0)
            neg_conf = score_map.get('negative', 0.0)
            
            conf_diff = pos_conf - neg_conf
            return _convert_confidence_to_valence(abs(conf_diff), 'POSITIVE' if conf_diff >= 0 else 'NEGATIVE', 'linear_transform')

    else:
        score_map = {item['label'].lower().replace('pos', 'positive').replace('neg', 'negative').replace('neu', 'neutral'): item['score'] for item in result}
        if "nlptown" in model_name:
            score_val = int(re.search(r'\d+', result[0]['label']).group())
            return -1.0 + (2.0 * (score_val - 1) / 4.0)
        else:
            return score_map.get('positive', 0.0) - score_map.get('negative', 0.0)

def _get_hf_logit_score(text: str, model, tokenizer) -> float:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    id2label = model.config.id2label
    neg_idx, pos_idx = -1, -1
    for i, label in id2label.items():
        if label.lower() in ['negative', 'neg']: neg_idx = i
        if label.lower() in ['positive', 'pos']: pos_idx = i
    
    if pos_idx == -1 or neg_idx == -1: return 0.0

    pos_logit = logits[0, pos_idx].item()
    neg_logit = logits[0, neg_idx].item()
    
    log_odds = pos_logit - neg_logit
    
    valence = math.tanh(log_odds / 2.0)
    
    return valence

def get_vader_sentiment(text: str) -> Dict[str, Any]:
    logger.debug("Analyzing sentiment using VADER")
    if not text or not text.strip():
        return _build_sentiment_result(0.0, confidence=0.0, raw={"reason": "empty"}, pos_threshold=0.05, neg_threshold=-0.05)
    try:
        analyzer = _get_vader_analyzer()
        scores = analyzer.polarity_scores(text)
        compound = scores.get('compound', 0.0)
        confidence = max(scores.get('pos', 0.0), scores.get('neg', 0.0))
        return _build_sentiment_result(compound, confidence=confidence, raw=scores, pos_threshold=0.05, neg_threshold=-0.05)
    except Exception as exc:
        logger.debug("VADER sentiment failed: %s", exc)
        return _build_sentiment_result(0.0, confidence=0.0, raw={"error": str(exc)}, pos_threshold=0.05, neg_threshold=-0.05)

def get_textblob_sentiment(text: str) -> Dict[str, Any]:
    logger.debug("Analyzing sentiment using TextBlob")
    if not text or not text.strip():
        return _build_sentiment_result(0.0, confidence=0.0, raw={"reason": "empty"}, pos_threshold=0.05, neg_threshold=-0.05)
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        confidence = max(abs(polarity), subjectivity)
        raw = {"polarity": polarity, "subjectivity": subjectivity}
        return _build_sentiment_result(polarity, confidence=confidence, raw=raw, pos_threshold=0.05, neg_threshold=-0.05)
    except Exception as exc:
        logger.debug("TextBlob sentiment failed: %s", exc)
        return _build_sentiment_result(0.0, confidence=0.0, raw={"error": str(exc)}, pos_threshold=0.05, neg_threshold=-0.05)

@lru_cache(maxsize=1)
def _get_flair_classifier() -> TextClassifier:
    return TextClassifier.load('sentiment')


def get_flair_sentiment(text: str) -> Dict[str, Any]:
    logger.debug("Analyzing sentiment using Flair")
    if not text or not text.strip():
        return _build_sentiment_result(0.0, confidence=0.0, raw={"reason": "empty"})
    try:
        flair_model = _get_flair_classifier()
        flair_sentence = Sentence(text)
        flair_model.predict(flair_sentence)
        if not flair_sentence.labels:
            return _build_sentiment_result(0.0, confidence=0.0, raw={"reason": "no_labels"})
        flair_label = flair_sentence.labels[0]
        score = _convert_confidence_to_valence(flair_label.score, flair_label.value, 'linear_transform')
        raw = {"label": flair_label.value, "confidence": flair_label.score}
        return _build_sentiment_result(score, confidence=flair_label.score, raw=raw)
    except Exception as exc:
        logger.debug("Flair sentiment failed: %s", exc)
        return _build_sentiment_result(0.0, confidence=0.0, raw={"error": str(exc)})

def get_pysentimiento_sentiment(text: str) -> Dict[str, Any]:
    logger.debug("Analyzing sentiment using PySentimiento")
    if not text or not text.strip():
        return _build_sentiment_result(0.0, confidence=0.0, raw={"reason": "empty"})
    try:
        pysentimiento = _get_pysentimiento_analyzer()
        pysent_result = pysentimiento.predict(text)
        probas = getattr(pysent_result, 'probas', {}) or {}
        pos_prob = float(probas.get('POS', 0.0))
        neg_prob = float(probas.get('NEG', 0.0))
        score = pos_prob - neg_prob
        confidence = max(probas.values()) if probas else abs(score)
        raw = {"label": getattr(pysent_result, 'output', None), "probas": probas}
        return _build_sentiment_result(score, confidence=confidence, raw=raw)
    except Exception as exc:
        logger.debug("PySentimiento sentiment failed: %s", exc)
        return _build_sentiment_result(0.0, confidence=0.0, raw={"error": str(exc)})

def get_swn_sentiment(text: str) -> Dict[str, Any]:
    logger.debug("Analyzing sentiment using SentiWordNet")
    if not text or not text.strip():
        return _build_sentiment_result(0.0, confidence=0.0, raw={"reason": "empty"})
    if swn is None:
        return _build_sentiment_result(0.0, confidence=0.0, raw={"error": "sentiwordnet_unavailable"})
    try:
        tokens = re.findall(r'\b[a-zA-Z-]+\b', text.lower())
        pos_sum, neg_sum, count = 0.0, 0.0, 0
        for token in tokens:
            synsets = list(swn.senti_synsets(token))
            if synsets:
                pos_sum += np.mean([s.pos_score() for s in synsets])
                neg_sum += np.mean([s.neg_score() for s in synsets])
            count += 1
        score = (pos_sum / count) - (neg_sum / count) if count > 0 else 0.0
        confidence = min(1.0, abs(score))
        raw = {"tokens": len(tokens), "pos": pos_sum, "neg": neg_sum}
        return _build_sentiment_result(score, confidence=confidence, raw=raw)
    except Exception as exc:
        logger.debug("SentiWordNet sentiment failed: %s", exc)
        return _build_sentiment_result(0.0, confidence=0.0, raw={"error": str(exc)})

def get_nlptown_sentiment(text: str) -> Dict[str, Any]:
    logger.debug("Analyzing sentiment using NLP Town")
    if not text or not text.strip():
        return _build_sentiment_result(0.0, confidence=0.0, raw={"reason": "empty"})
    try:
        model = _get_hf_pipeline("nlptown/bert-base-multilingual-uncased-sentiment")
        results = model([text], batch_size=32, truncation=True)
        score = _get_hf_pipeline_score(results, is_confidence_only=False, model_name="nlptown")
        confidence = float(results[0].get('score', abs(score))) if results else abs(score)
        return _build_sentiment_result(score, confidence=confidence, raw=results)
    except Exception as exc:
        logger.debug("NLP Town sentiment failed: %s", exc)
        return _build_sentiment_result(0.0, confidence=0.0, raw={"error": str(exc)})

def get_finiteautomata_sentiment(text: str) -> Dict[str, Any]:
    logger.debug("Analyzing sentiment using Finite Automata")
    if not text or not text.strip():
        return _build_sentiment_result(0.0, confidence=0.0, raw={"reason": "empty"})
    try:
        model = _get_hf_pipeline("finiteautomata/bertweet-base-sentiment-analysis")
        results = model([text], batch_size=32, truncation=True)
        score = _get_hf_pipeline_score(results, is_confidence_only=False, model_name="finiteautomata")
        confidence = float(results[0].get('score', abs(score))) if results else abs(score)
        return _build_sentiment_result(score, confidence=confidence, raw=results)
    except Exception as exc:
        logger.debug("FiniteAutomata sentiment failed: %s", exc)
        return _build_sentiment_result(0.0, confidence=0.0, raw={"error": str(exc)})

def get_ProsusAI_sentiment(text: str) -> Dict[str, Any]:
    logger.debug("Analyzing sentiment using ProsusAI")
    if not text or not text.strip():
        return _build_sentiment_result(0.0, confidence=0.0, raw={"reason": "empty"})
    try:
        model = _get_hf_pipeline("ProsusAI/finbert")
        results = model([text], batch_size=32, truncation=True)
        score = _get_hf_pipeline_score(results, is_confidence_only=False, model_name="ProsusAI")
        confidence = float(results[0].get('score', abs(score))) if results else abs(score)
        return _build_sentiment_result(score, confidence=confidence, raw=results)
    except Exception as exc:
        logger.debug("ProsusAI sentiment failed: %s", exc)
        return _build_sentiment_result(0.0, confidence=0.0, raw={"error": str(exc)})

def get_distilbert_logit_sentiment(text: str) -> Dict[str, Any]:
    logger.debug("Analyzing sentiment using DistilBERT logits")
    if not text or not text.strip():
        return _build_sentiment_result(0.0, confidence=0.0, raw={"reason": "empty"})
    try:
        model = _get_hf_pipeline("distilbert-base-uncased-finetuned-sst-2-english")
        tokenizer = model.tokenizer
        score = _get_hf_logit_score(text, model.model, tokenizer)
        return _build_sentiment_result(score, confidence=abs(score), raw={"model": "distilbert-base-uncased-finetuned-sst-2-english"})
    except Exception as exc:
        logger.debug("DistilBERT sentiment failed: %s", exc)
        return _build_sentiment_result(0.0, confidence=0.0, raw={"error": str(exc)})
