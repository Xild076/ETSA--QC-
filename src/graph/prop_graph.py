import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import mplcursors
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List
import spacy
import afinn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

nlp = spacy.load("en_core_web_sm")
_analyzer = SentimentIntensityAnalyzer()
_afinn = afinn.Afinn()

def _tokenize_with_word(text):
    return [token.text for token in nlp(text)]

def normalized_afinn_score(text):
    scores = [_afinn.score(w) for w in _tokenize_with_word(text)]
    relevant = [s for s in scores if s != 0]
    return sum(relevant) / len(relevant) / 5.0 if relevant else 0.0

def analyze_sentiment(text):
    return (_analyzer.polarity_scores(text)['compound'] + normalized_afinn_score(text)) / 2 if text else 0.0

def parabola(x, h0, h1, x_L, v_k):
    den = x_L * (x_L - 2 * v_k)
    if den == 0:
        if x_L == 0 or h1 == h0:
            a = 0
        else:
            raise ValueError("Cannot fit an upward parabola: x_L == 2*v_k but h1 != h0")
    else:
        a = abs((h1 - h0) / den)
    h_v = h0 - a * v_k**2
    return a * (x - v_k)**2 + h_v

def normalized_parabola_weights(n, h0, h1, xL, vk):
    x_values = np.linspace(0, xL, n)
    y_values = parabola(x_values, h0, h1, xL, vk)
    return y_values / np.sum(y_values)

def normalized_parabola_visualized(n, h0, h1, xL, vk):
    x_values = np.linspace(0, xL, n)
    y_values = parabola(x_values, h0, h1, xL, vk)
    weights = y_values / np.sum(y_values)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x_values, y_values, marker='o')
    plt.title("Parabola Values")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.subplot(1, 2, 2)
    plt.bar(x_values, weights, width=(xL / n) * 0.8)
    plt.title("Normalized Weights")
    plt.xlabel("x")
    plt.ylabel("Weight")

    plt.tight_layout()
    plt.show()

    return x_values, weights

class RelationshipType(Enum):
    ACTION = "action"
    ASSOCIATION = "association"
    BELONGING = "belonging"

"""
Data structure for a property graph node.

Nodes:
- id: Unique identifier for the node.
- label: Label of the node.
* sentence_n: index of sentences where the node has a reference:
    - references: list of references to the node in the sentences.
        - each reference is a list of tuples (sentence_index, start_index, end_index)
    - associated_tokens: list of tokens associated with the node.
        - each token is a tuple (sentence_index, token_index)

Relationships:
- relationships: Dictionary mapping relationship types to lists of tuples.
    - Each tuple contains the target node ID and the relationship details.
"""

class SentimentPropertyGraph:
    def __init__(self, text):
        self.G = nx.DiGraph()
        self.text = text
        self.sentences = [s.text for s in nlp(text).sents]
    
    def add_aspect():
        pass