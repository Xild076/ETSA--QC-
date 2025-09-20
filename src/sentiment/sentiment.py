import ssl
from typing import Literal, List, Dict, Any
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

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    logit_diff = logits[0, pos_idx] - logits[0, neg_idx]
    return torch.tanh(logit_diff).item()

def get_vader_sentiment(text: str) -> float:
    logger.info(f"Analyzing sentiment using VADER for text: {text}")
    try:
        if not text or not text.strip():
            return 0.0
        
        # Handle implicit sentiment in short, suggestive phrases
        implicit_sentiment_map = {
            "recommend": 0.8,
            "stay away": -0.9,
            "worth the": 0.7,
            "must try": 0.85,
            "must-try": 0.85,
            "must have": 0.85,
            "must-have": 0.85,
            "go for": 0.6,
            "look for": 0.5,
            "avoid": -0.8,
            "skip": -0.7,
            "pass on": -0.6,
            "don't miss": 0.75,
            "do not miss": 0.75,
            "can't miss": 0.75,
            "cannot miss": 0.75,
            "a must": 0.85,
            "a plus": 0.7,
            "a bonus": 0.65,
            "a gem": 0.9,
            "a joke": -0.8,
            "a mess": -0.7,
            "a shame": -0.6,
            "a treat": 0.8,
            "a delight": 0.85,
            "a pleasure": 0.8,
            "a disappointment": -0.9,
            "a letdown": -0.85,
            "a failure": -0.9,
            "a success": 0.9,
            "a win": 0.8,
            "a loss": -0.8,
            "a find": 0.7,
            "a discovery": 0.6,
            "a revelation": 0.8,
            "a surprise": 0.5, # Can be neutral, but often positive in reviews
            "a pleasant surprise": 0.85,
            "an unpleasant surprise": -0.85,
            "a pleasant experience": 0.9,
            "an unpleasant experience": -0.9,
            "a good choice": 0.8,
            "a bad choice": -0.8,
            "a great choice": 0.9,
            "a terrible choice": -0.9,
            "a wise choice": 0.85,
            "a poor choice": -0.85,
            "a solid choice": 0.75,
            "a safe bet": 0.6,
            "a gamble": -0.4,
            "a risk": -0.5,
            "a safe choice": 0.6,
            "a safe option": 0.6,
            "a good option": 0.7,
            "a bad option": -0.7,
            "a great option": 0.8,
            "a terrible option": -0.8,
            "a decent option": 0.5,
            "a viable option": 0.4,
            "an option": 0.1, # Neutral
            "an alternative": 0.1, # Neutral
            "a backup": 0.0, # Neutral
            "a substitute": -0.1, # Slightly negative
            "a replacement": -0.1, # Slightly negative
            "a copy": -0.2,
            "an imitation": -0.4,
            "a fake": -0.8,
            "a fraud": -0.9,
            "a scam": -0.95,
            "a ripoff": -0.9,
            "a steal": 0.9, # Positive (good value)
            "a bargain": 0.85,
            "a deal": 0.8,
            "overpriced": -0.7,
            "underpriced": 0.5, # Can be good
            "pricey": -0.6,
            "costly": -0.5,
            "expensive": -0.4, # VADER handles this, but we can boost it
            "cheap": -0.3, # VADER handles this, but can be ambiguous
            "affordable": 0.6,
            "reasonable": 0.5,
            "fair price": 0.6,
            "good value": 0.8,
            "great value": 0.9,
            "bad value": -0.8,
            "poor value": -0.9,
            "not worth it": -0.85,
            "not worth the money": -0.9,
            "not worth the price": -0.9,
            "worth every penny": 0.95,
            "worth every cent": 0.95,
            "acceptable": 0.4,
            "decent": 0.5,
            "adequate": 0.3,
            "sufficient": 0.2,
            "satisfactory": 0.45,
            "passable": 0.25,
            "tolerable": 0.15,
            "unacceptable": -0.7,
            "inadequate": -0.6,
            "insufficient": -0.5,
            "unsatisfactory": -0.65
        }
        
        text_lower = text.lower()
        for phrase, score in implicit_sentiment_map.items():
            if phrase in text_lower:
                return score

        analyzer = SentimentIntensityAnalyzer()
        result = analyzer.polarity_scores(text)
    except:
        return 0.0
    return result['compound']

def get_textblob_sentiment(text: str) -> float:
    logger.info(f"Analyzing sentiment using TextBlob for text: {text}")
    try:
        if not text or not text.strip():
            return 0.0
        
        blob = TextBlob(text)
    except:
        return 0.0
    return blob.sentiment.polarity

def get_flair_sentiment(text: str) -> float:
    logger.info(f"Analyzing sentiment using Flair for text: {text}")
    try:
        flair_model = TextClassifier.load('sentiment')
        flair_sentence = Sentence(text)
        flair_model.predict(flair_sentence)
        flair_label = flair_sentence.labels[0]
        return _convert_confidence_to_valence(flair_label.score, flair_label.value, 'linear_transform')
    except:
        return 0.0

def get_pysentimiento_sentiment(text: str) -> float:
    logger.info(f"Analyzing sentiment using PySentimiento for text: {text}")
    try:
        pysentimiento = create_analyzer(task="sentiment", lang="en")
        pysent_result = pysentimiento.predict(text)
        return pysent_result.probas.get('POS', 0.0) - pysent_result.probas.get('NEG', 0.0)
    except:
        return 0.0

def get_swn_sentiment(text: str) -> float:
    logger.info(f"Analyzing sentiment using SentiWordNet for text: {text}")
    try:
        tokens = re.findall(r'\b[a-zA-Z-]+\b', text.lower())
        pos, neg, count = 0.0, 0.0, 0
        for token in tokens:
            synsets = list(swn.senti_synsets(token))
            if synsets:
                pos += np.mean([s.pos_score() for s in synsets])
                neg += np.mean([s.neg_score() for s in synsets])
            count += 1
        return (pos / count) - (neg / count) if count > 0 else 0.0
    except:
        return 0.0

def get_nlptown_sentiment(text: str) -> float:
    logger.info(f"Analyzing sentiment using NLP Town for text: {text}")
    try:
        if not text or not text.strip():
            return 0.0

        nlp_town_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
        results = nlp_town_model([text], batch_size=32, truncation=True)
        return _get_hf_pipeline_score(results, is_confidence_only=False, model_name="nlptown")
    except:
        return 0.0

def get_finiteautomata_sentiment(text: str) -> float:
    logger.info(f"Analyzing sentiment using Finite Automata for text: {text}")
    try:
        if not text or not text.strip():
            return 0.0
        finiteautomata_model = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
        results = finiteautomata_model([text], batch_size=32, truncation=True)
        return _get_hf_pipeline_score(results, is_confidence_only=False, model_name="finiteautomata")
    except:
        return 0.0

def context_aware_sentiment_adjustment(text: str, modifier_text: str, base_sentiment: float) -> float:
    """
    Adjust sentiment based on context to handle cases like 'I miss X' where X should be positive.
    """
    text_lower = text.lower()
    modifier_lower = modifier_text.lower()
    
    # Handle "miss" in positive contexts
    if "miss" in modifier_lower:
        # Patterns that indicate positive sentiment despite "miss"
        positive_miss_patterns = [
            r'\bi\s+miss\s+',  # "I miss"
            r'\bwe\s+miss\s+', # "we miss"
            r'\byou\s+miss\s+', # "you miss" 
            r'\bmiss\s+.+\s+(so\s+much|a\s+lot|terribly)',  # "miss X so much"
            r'\breally\s+miss\s+',  # "really miss"
            r'\bdefinitely\s+miss\s+',  # "definitely miss"
        ]
        
        for pattern in positive_miss_patterns:
            if re.search(pattern, text_lower):
                # If we detect positive "miss", invert the negative sentiment
                return abs(base_sentiment) if base_sentiment < 0 else base_sentiment
    
    # Handle "wait" in contexts like "worth the wait"
    if "wait" in modifier_lower and ("worth" in text_lower or "value" in text_lower):
        # "worth the wait" should be positive
        return abs(base_sentiment) if base_sentiment < 0 else base_sentiment
    
    # Handle other context adjustments here as needed
    
    return base_sentiment

def get_ProsusAI_sentiment(text: str) -> float:
    logger.info(f"Analyzing sentiment using ProsusAI for text: {text}")
    try:
        if not text or not text.strip():
            return 0.0

        prosusai_model = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        results = prosusai_model([text], batch_size=32, truncation=True)
        return _get_hf_pipeline_score(results, is_confidence_only=False, model_name="ProsusAI")
    except:
        return 0.0

def get_distilbert_logit_sentiment(text: str) -> float:
    logger.info(f"Analyzing sentiment using DistilBERT logits for text: {text}")
    try:
        if not text or not text.strip():
            return 0.0

        distilbert_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        tokenizer = distilbert_model.tokenizer
        return _get_hf_logit_score(text, distilbert_model.model, tokenizer)
    except:
        return 0.0