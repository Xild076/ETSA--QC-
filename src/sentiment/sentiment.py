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
    if not text or not text.strip():
        return 0.0
    
    analyzer = SentimentIntensityAnalyzer()
    result = analyzer.polarity_scores(text)
    
    return result['compound']

def get_textblob_sentiment(text: str) -> float:
    if not text or not text.strip():
        return 0.0
    
    blob = TextBlob(text)
    return blob.sentiment.polarity

def get_flair_sentiment(text: str) -> float:
    flair_model = TextClassifier.load('sentiment')
    flair_sentence = Sentence(text)
    flair_model.predict(flair_sentence)
    flair_label = flair_sentence.labels[0]
    return _convert_confidence_to_valence(flair_label.score, flair_label.value, 'linear_transform')

def get_pysentimiento_sentiment(text: str) -> float:
    pysentimiento = create_analyzer(task="sentiment", lang="en")
    pysent_result = pysentimiento.predict(text)
    return pysent_result.probas.get('POS', 0.0) - pysent_result.probas.get('NEG', 0.0)

def get_swn_sentiment(text: str) -> float:
    tokens = re.findall(r'\b[a-zA-Z-]+\b', text.lower())
    pos, neg, count = 0.0, 0.0, 0
    for token in tokens:
        synsets = list(swn.senti_synsets(token))
        if synsets:
            pos += np.mean([s.pos_score() for s in synsets])
            neg += np.mean([s.neg_score() for s in synsets])
            count += 1
    return (pos / count) - (neg / count) if count > 0 else 0.0

def get_nlptown_sentiment(text: str) -> float:
    if not text or not text.strip():
        return 0.0
    
    nlp_town_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    results = nlp_town_model([text], batch_size=32, truncation=True)
    return _get_hf_pipeline_score(results, is_confidence_only=False, model_name="nlptown")

def get_finiteautomata_sentiment(text: str) -> float:
    if not text or not text.strip():
        return 0.0

    finiteautomata_model = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
    results = finiteautomata_model([text], batch_size=32, truncation=True)
    return _get_hf_pipeline_score(results, is_confidence_only=False, model_name="finiteautomata")

def get_ProsusAI_sentiment(text: str) -> float:
    if not text or not text.strip():
        return 0.0

    prosusai_model = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    results = prosusai_model([text], batch_size=32, truncation=True)
    return _get_hf_pipeline_score(results, is_confidence_only=False, model_name="ProsusAI")

def get_distilbert_logit_sentiment(text: str) -> float:
    if not text or not text.strip():
        return 0.0

    distilbert_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    tokenizer = distilbert_model.tokenizer
    return _get_hf_logit_score(text, distilbert_model.model, tokenizer)