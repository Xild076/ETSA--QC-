from functools import lru_cache
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch

class SentimentSystem:
    def analyze(self, text: str, aspect: str = None) -> float:
        raise NotImplementedError

class VADERSystem(SentimentSystem):
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str, aspect: str = None) -> float:
        return self.analyzer.polarity_scores(text)['compound']

class LSAeDeBERTaSystem(SentimentSystem):
    @lru_cache(maxsize=1)
    def _get_pipeline(self):
        model_name = "yangheng/deberta-v3-large-absa-v1.1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = 0 if torch.cuda.is_available() else -1
        return pipeline("text-classification", model=model, tokenizer=tokenizer, device=device, top_k=None)

    def analyze(self, text: str, aspect: str = None) -> float:
        if not aspect:
            return VADERSystem().analyze(text)
        pipe = self._get_pipeline()
        result = pipe(f"{text} [SEP] {aspect}")
        if isinstance(result, list):
            if result and isinstance(result[0], list):
                items = result[0]
            elif result and isinstance(result[0], dict):
                items = result
            else:
                items = []
        elif isinstance(result, dict):
            items = [result]
        else:
            items = []
        score_map = {item['label'].lower(): item['score'] for item in items if isinstance(item, dict) and 'label' in item and 'score' in item}
        pos_score = score_map.get('positive', 0.0)
        neg_score = score_map.get('negative', 0.0)
        return pos_score - neg_score

def get_sentiment_system(name: str) -> SentimentSystem:
    name = name.lower()
    if name == 'vader':
        return VADERSystem()
    elif name == 'lsae-deberta':
        return LSAeDeBERTaSystem()
    else:
        raise ValueError(f"Unknown sentiment system: '{name}'. Available: 'vader', 'lsae-deberta'.")