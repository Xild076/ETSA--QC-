from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from spicy import stats

def get_vader_valence(text: str) -> float:
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return scores['compound']

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def get_transformer_regression_valence(text: str) -> float:    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    probabilities = F.softmax(logits, dim=1).squeeze()
    
    weighted_average = torch.sum(probabilities * torch.arange(1, 6)).item()
    
    normalized_score = -1 + 2 * (weighted_average - 1) / 4
    return normalized_score

from transformers import pipeline

pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", top_k=None)

def get_transformer_3class_valence(text: str) -> float:
    pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", top_k=None)
    
    results = pipe(text)[0]
    score_map = {item['label'].lower(): item['score'] for item in results}
    
    valence = score_map.get('positive', 0.0) - score_map.get('negative', 0.0)
    return valence

from textblob import TextBlob

def get_textblob_valence(text: str) -> float:
    blob = TextBlob(text)
    return blob.sentiment.polarity

from flair.models import TextClassifier
from flair.data import Sentence

def get_flair_valence(text: str) -> float:
    classifier = TextClassifier.load('sentiment')
    sentence = Sentence(text)
    classifier.predict(sentence)
    
    label = sentence.labels[0].value
    score = sentence.labels[0].score
    
    valence = score if label == 'POSITIVE' else -score
    return valence

sentiment_methods = {
    "vader": get_vader_valence,
    "transformer_regression": get_transformer_regression_valence,
    "transformer_3class": get_transformer_3class_valence,
    "textblob": get_textblob_valence,
    "flair": get_flair_valence
}

def get_sentiment(text: str, method: str = "vader") -> float:
    sentiments = {}
    for method_name, method_func in sentiment_methods.items():
        sentiments[method_name] = method_func(text)
    output = {
        "text": text,
        "sentiments": sentiments,
        "selected_method": method,
        "selected_score": sentiments.get(method, 0.0),
        "mean_score": sum(sentiments.values()) / len(sentiments),
        "median_score": sorted(sentiments.values())[len(sentiments) // 2],
        "max_score": max(sentiments.values()),
        "min_score": min(sentiments.values()),
        "trim_mean_score": float(stats.trim_mean(list(sentiments.values()), proportiontocut=0.25)),
        "standard_deviation": float(stats.tstd(list(sentiments.values()))),
    }
    return output
