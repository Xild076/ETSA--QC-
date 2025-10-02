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
import string

import logging

def get_optimal_device():
    """Get the optimal device for computation (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

_DEVICE = get_optimal_device()
print(f"Using device: {_DEVICE}")

_ENV_DEFAULTS = {
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "OMP_NUM_THREADS": "2" if _DEVICE == "cpu" else "1",
    "KMP_AFFINITY": "disabled",
    "KMP_INIT_AT_FORK": "FALSE",
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0" if _DEVICE == "mps" else None,
}
for key, value in _ENV_DEFAULTS.items():
    if value is not None:
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
    existing = _HF_PIPELINES.get(model_name)
    if existing is None or not getattr(existing, "return_all_scores", False):
        try:
            device_id = 0 if _DEVICE in ["cuda", "mps"] else -1
            _HF_PIPELINES[model_name] = pipeline(
                "sentiment-analysis", 
                model=model_name, 
                return_all_scores=True,
                device=device_id,
                torch_dtype=torch.float16 if _DEVICE != "cpu" else torch.float32
            )
            logger.info(f"Loaded {model_name} on device: {_DEVICE}")
        except Exception as e:
            logger.warning(f"Failed to load {model_name} on {_DEVICE}, falling back to CPU: {e}")
            _HF_PIPELINES[model_name] = pipeline(
                "sentiment-analysis", 
                model=model_name, 
                return_all_scores=True,
                device=-1
            )
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

def _label_valence_from_rating(label: str, max_rating: int = 5) -> float:
    digits = re.findall(r"\d+", label)
    if not digits:
        return 0.0
    rating = max(1, min(int(digits[0]), max_rating))
    if max_rating <= 1:
        return 0.0
    return -1.0 + 2.0 * (rating - 1) / (max_rating - 1)


def _label_valence_basic(label: str) -> float:
    lowered = label.lower()
    if any(token in lowered for token in ("positive", "pos", "bullish")):
        return 1.0
    if any(token in lowered for token in ("negative", "neg", "bearish")):
        return -1.0
    if any(token in lowered for token in ("neutral", "neu", "mixed", "none")):
        return 0.0
    return 0.0


def _get_hf_pipeline_score(result: List[Dict[str, Any]], is_confidence_only: bool, model_name:str) -> tuple[float, float]:
    if not result:
        return 0.0, 0.0
    records = result[0] if isinstance(result[0], list) else result
    if not isinstance(records, list):
        records = [records]
    model_lower = (model_name or "").lower()
    valence = 0.0
    confidence = 0.0
    total = 0.0
    use_ratings = "nlptown" in model_lower
    for item in records:
        score = float(item.get("score", 0.0))
        label = item.get("label", "")
        confidence = max(confidence, score)
        if use_ratings:
            polarity = _label_valence_from_rating(label, max_rating=5)
        else:
            polarity = _label_valence_basic(label)
        valence += polarity * score
        total += score
    if use_ratings and total > 0:
        valence = max(-1.0, min(1.0, valence))
    elif not use_ratings and total > 0:
        valence = max(-1.0, min(1.0, valence))
    return valence, confidence

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
    try:
        classifier = TextClassifier.load('sentiment')
        if _DEVICE in ["cuda", "mps"] and hasattr(classifier, 'to'):
            classifier.to(_DEVICE)
            logger.info(f"Loaded Flair classifier on device: {_DEVICE}")
        return classifier
    except Exception as e:
        logger.warning(f"Failed to load Flair on {_DEVICE}, using CPU: {e}")
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
        score, confidence = _get_hf_pipeline_score(results, is_confidence_only=False, model_name="nlptown")
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
        score, confidence = _get_hf_pipeline_score(results, is_confidence_only=False, model_name="finiteautomata")
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
        score, confidence = _get_hf_pipeline_score(results, is_confidence_only=False, model_name="ProsusAI")
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


                                                                            

def create_focused_context_window(text: str, aspect: str, modifiers: List[str], 
                                 window_size: int = 50) -> str:
    """
    Create a focused context window around an aspect while preserving sentence structure.
    Removes unnecessary words but keeps punctuation for proper idea separation.
    """
    if not text or not aspect:
        return text
    
                                  
    text_lower = text.lower()
    aspect_lower = aspect.lower()
    aspect_pos = text_lower.find(aspect_lower)
    
    if aspect_pos == -1:
                                                     
        aspect_words = aspect_lower.split()
        for i, word in enumerate(text_lower.split()):
            if any(aw in word for aw in aspect_words):
                aspect_pos = text_lower.find(word)
                break
    
    if aspect_pos == -1:
        return text                             
    
                                 
    start_pos = max(0, aspect_pos - window_size)
    end_pos = min(len(text), aspect_pos + len(aspect) + window_size)
    
                                   
    start_pos = _find_sentence_start(text, start_pos)
    end_pos = _find_sentence_end(text, end_pos)
    
    context_window = text[start_pos:end_pos].strip()
    
                                      
    focused_context = _clean_preserving_structure(context_window, aspect, modifiers)
    
    return focused_context

def _find_sentence_start(text: str, pos: int) -> int:
    """Find the start of the sentence containing the given position."""
    sentence_markers = {'.', '!', '?', '\n'}
    while pos > 0:
        if text[pos] in sentence_markers:
                                                
            pos += 1
            while pos < len(text) and text[pos].isspace():
                pos += 1
            break
        pos -= 1
    return pos

def _find_sentence_end(text: str, pos: int) -> int:
    """Find the end of the sentence containing the given position."""
    sentence_markers = {'.', '!', '?', '\n'}
    while pos < len(text):
        if text[pos] in sentence_markers:
            pos += 1
            break
        pos += 1
    return pos

def _clean_preserving_structure(text: str, aspect: str, modifiers: List[str]) -> str:
    """
    Remove unnecessary words while preserving sentence structure and punctuation.
    Keeps words that are relevant to the aspect or provide context for sentiment.
    """
    if not text:
        return text
    
                                               
    keep_words = {
                                   
        aspect.lower(),
                   
        *[mod.lower() for mod in modifiers],
                         
        'good', 'bad', 'great', 'terrible', 'amazing', 'awful', 'excellent', 'poor',
        'love', 'hate', 'like', 'dislike', 'enjoy', 'despise',
        'fast', 'slow', 'quick', 'sluggish', 'responsive', 'laggy',
        'beautiful', 'ugly', 'attractive', 'hideous', 'stunning',
        'reliable', 'unreliable', 'stable', 'unstable', 'buggy',
        'expensive', 'cheap', 'affordable', 'overpriced', 'costly',
        'worth', 'worthless', 'valuable', 'useless', 'useful',
                      
        'very', 'extremely', 'quite', 'really', 'super', 'incredibly', 
        'somewhat', 'rather', 'fairly', 'pretty', 'highly', 'totally',
                  
        'not', 'no', 'never', 'nothing', 'none', 'neither', 'nor',
        'dont', "don't", 'cant', "can't", 'wont', "won't", 'isnt', "isn't",
        'wasnt', "wasn't", 'arent', "aren't", 'werent', "weren't",
                                
        'can', 'could', 'should', 'would', 'will', 'shall', 'may', 'might',
        'must', 'ought', 'need', 'dare', 'used',
                      
        'better', 'worse', 'best', 'worst', 'more', 'less', 'most', 'least',
        'than', 'as', 'like', 'unlike', 'compared', 'versus',
                          
        'now', 'then', 'before', 'after', 'during', 'while', 'when',
        'initially', 'finally', 'eventually', 'suddenly', 'immediately',
                                             
        'and', 'or', 'but', 'however', 'although', 'though', 'despite',
        'because', 'since', 'so', 'therefore', 'thus', 'hence',
        'if', 'unless', 'when', 'while', 'whereas', 'until'
    }
    
                                                   
    tokens = re.findall(r'\w+|[^\w\s]', text)
    
    filtered_tokens = []
    for token in tokens:
        token_lower = token.lower()
        
                                 
        if token in string.punctuation:
            filtered_tokens.append(token)
                              
        elif token_lower in keep_words:
            filtered_tokens.append(token)
                                                
        elif aspect.lower() in token_lower or token_lower in aspect.lower():
            filtered_tokens.append(token)
                                             
        elif any(token_lower in mod.lower() or mod.lower() in token_lower for mod in modifiers):
            filtered_tokens.append(token)
                                                                
        elif any(token_lower.endswith(suffix) for suffix in ['ly', 'ing', 'ed', 'er', 'est', 'ful', 'less']):
            filtered_tokens.append(token)
                                                                            
        elif token_lower in {'the', 'a', 'an', 'this', 'that', 'these', 'those', 
                           'i', 'you', 'he', 'she', 'it', 'we', 'they',
                           'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                           'have', 'has', 'had', 'do', 'does', 'did'}:
                                                      
            continue
        else:
            filtered_tokens.append(token)
    
                                          
    result = ""
    for i, token in enumerate(filtered_tokens):
        if i == 0:
            result += token
        elif token in string.punctuation:
            result += token
        else:
            result += " " + token
    
    return result.strip()

def analyze_aspect_sentiment_focused(text: str, aspect: str, modifiers: List[str], 
                                   analysis_function) -> Dict[str, Any]:
    """
    Analyze sentiment for a specific aspect using focused context windows.
    Maps extracted information back to sentence structure while removing noise.
    """
    if not text or not aspect:
        return {"score": 0.0, "confidence": 0.0, "method": "focused_context", "error": "missing_input"}
    
                            
    focused_context = create_focused_context_window(text, aspect, modifiers)
    
                                                    
    if len(focused_context) < 10:
        focused_context = text
    
                                          
    try:
        result = analysis_function(focused_context)
        
                                                 
        if isinstance(result, dict):
            result["focused_context"] = focused_context
            result["original_length"] = len(text)
            result["focused_length"] = len(focused_context)
            result["context_reduction"] = 1.0 - (len(focused_context) / len(text)) if len(text) > 0 else 0.0
            result["method"] = "focused_context"
        
        return result
    except Exception as e:
        logger.warning(f"Focused sentiment analysis failed: {e}")
                                   
        return analysis_function(text)
