import ssl
from typing import Dict, Any
from textblob import TextBlob
from flair.data import Sentence
from flair.models import TextClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pysentimiento import create_analyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- MODEL CACHING ---
# Load models once to avoid repeated loading on each call
_models: Dict[str, Any] = {}

def _get_model(model_name: str, task: str = "sentiment-analysis"):
    if model_name not in _models:
        if "flair" in model_name:
            _models[model_name] = TextClassifier.load('sentiment')
        elif "pysentimiento" in model_name:
            _models[model_name] = create_analyzer(task="sentiment", lang="en")
        elif "vader" in model_name:
            _models[model_name] = SentimentIntensityAnalyzer()
        elif "distilbert" in model_name:
            # For logits, we need model and tokenizer separately
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            _models[model_name] = {"model": model, "tokenizer": tokenizer}
        else:
            _models[model_name] = pipeline(task, model=model_name)
    return _models[model_name]

# --- STANDARDIZED OUTPUT FUNCTIONS ---

def get_vader_sentiment(text: str) -> Dict[str, float]:
    if not text or not text.strip():
        return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
    analyzer = _get_model("vader")
    scores = analyzer.polarity_scores(text)
    return {'positive': scores['pos'], 'negative': scores['neg'], 'neutral': scores['neu']}

def get_textblob_sentiment(text: str) -> Dict[str, float]:
    if not text or not text.strip():
        return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return {'positive': polarity, 'negative': 0.0, 'neutral': 1.0 - polarity}
    else:
        return {'positive': 0.0, 'negative': -polarity, 'neutral': 1.0 + polarity}

def get_flair_sentiment(text: str) -> Dict[str, float]:
    if not text or not text.strip():
        return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
    flair_model = _get_model("flair/sentiment")
    sentence = Sentence(text)
    flair_model.predict(sentence)
    label = sentence.labels[0]
    confidence = label.score
    if label.value == 'POSITIVE':
        return {'positive': confidence, 'negative': 1.0 - confidence, 'neutral': 0.0}
    else:
        return {'positive': 1.0 - confidence, 'negative': confidence, 'neutral': 0.0}

def get_pysentimiento_sentiment(text: str) -> Dict[str, float]:
    if not text or not text.strip():
        return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
    analyzer = _get_model("pysentimiento")
    result = analyzer.predict(text)
    return {'positive': result.probas.get('POS', 0.0), 'negative': result.probas.get('NEG', 0.0), 'neutral': result.probas.get('NEU', 0.0)}

def _normalize_hf_results(results: list) -> Dict[str, float]:
    score_map = {}
    for item in results:
        label = item['label'].lower()
        if label in ('pos', 'positive'):
            score_map['positive'] = item['score']
        elif label in ('neg', 'negative'):
            score_map['negative'] = item['score']
        elif label in ('neu', 'neutral'):
            score_map['neutral'] = item['score']
    
    total = sum(score_map.values())
    if total > 0:
        return {k: v / total for k, v in score_map.items()}
    return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}


def get_nlptown_sentiment(text: str) -> Dict[str, float]:
    if not text or not text.strip():
        return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
    model = _get_model("nlptown/bert-base-multilingual-uncased-sentiment")
    result = model(text)[0]
    score_val = int(re.search(r'\d+', result['label']).group())
    # Simple mapping from 1-5 stars to a distribution
    if score_val == 1: return {'positive': 0.0, 'negative': 1.0, 'neutral': 0.0}
    if score_val == 2: return {'positive': 0.1, 'negative': 0.6, 'neutral': 0.3}
    if score_val == 3: return {'positive': 0.1, 'negative': 0.1, 'neutral': 0.8}
    if score_val == 4: return {'positive': 0.6, 'negative': 0.1, 'neutral': 0.3}
    if score_val == 5: return {'positive': 1.0, 'negative': 0.0, 'neutral': 0.0}
    return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}

def get_finiteautomata_sentiment(text: str) -> Dict[str, float]:
    if not text or not text.strip():
        return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
    model = _get_model("finiteautomata/bertweet-base-sentiment-analysis")
    results = model(text)
    return _normalize_hf_results(results)

def get_ProsusAI_sentiment(text: str) -> Dict[str, float]:
    if not text or not text.strip():
        return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
    model = _get_model("ProsusAI/finbert")
    results = model(text)
    return _normalize_hf_results(results)

def get_distilbert_logit_sentiment(text: str) -> Dict[str, float]:
    if not text or not text.strip():
        return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
    
    components = _get_model("distilbert-base-uncased-finetuned-sst-2-english")
    model, tokenizer = components["model"], components["tokenizer"]
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # This model only has POS/NEG classes, so we can use softmax
    probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
    
    id2label = model.config.id2label
    neg_idx, pos_idx = -1, -1
    for i, label in id2label.items():
        if label.lower() in ['negative', 'neg']: neg_idx = i
        if label.lower() in ['positive', 'pos']: pos_idx = i
        
    if pos_idx != -1 and neg_idx != -1:
        return {'positive': probabilities[pos_idx], 'negative': probabilities[neg_idx], 'neutral': 0.0}
        
    return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}