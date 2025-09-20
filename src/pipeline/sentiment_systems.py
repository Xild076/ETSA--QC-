import os
import sys
from functools import lru_cache
from typing import List
import numpy as np
import torch
from transformers import pipeline

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from sentiment.sentiment import (
    get_vader_sentiment, get_textblob_sentiment, get_flair_sentiment,
    get_nlptown_sentiment, get_finiteautomata_sentiment as get_bertweet_sentiment,
    get_ProsusAI_sentiment as get_finbert_sentiment,
)

class SentimentSystem:
    """Base class for sentiment analysis systems."""
    def analyze(self, text: str) -> float:
        raise NotImplementedError

class VADERSystem(SentimentSystem):
    """Simple VADER sentiment analyzer."""
    def analyze(self, text: str) -> float:
        return get_vader_sentiment(text)

class VADEREntityBaseline(SentimentSystem):
    """VADER with entity-aware context analysis."""
    def __init__(self):
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            self.nlp = None
    
    def analyze_with_entity(self, text: str, entity: str) -> float:
        """Analyze sentiment with entity-aware context."""
        if not self.nlp:
            return get_vader_sentiment(text)
        
        doc = self.nlp(text)
        entity_lower = entity.lower()
        
        # Find entity position and extract context
        for i, token in enumerate(doc):
            if entity_lower in token.text.lower():
                entity_start = max(0, i - 5)
                entity_end = min(len(doc), i + 6)
                context_tokens = doc[entity_start:entity_end]
                context_text = " ".join([token.text for token in context_tokens])
                return get_vader_sentiment(context_text)
        
        return get_vader_sentiment(text)
    
    def analyze(self, text: str) -> float:
        """Standard analyze method for compatibility."""
        return get_vader_sentiment(text)

class PresetEnsembleSystem(SentimentSystem):
    """Enhanced ensemble matching the models and methods used in ttw.py with GPU acceleration."""
    def __init__(self):
        # Device setup for GPU acceleration (Mac GPU or CUDA)
        if torch.cuda.is_available():
            self.device = 0
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'  # Apple Silicon GPU
        else:
            self.device = -1  # CPU
        
        # Models exactly matching ttw.py CONFIG
        self.analyzers = {
            get_vader_sentiment: 1.0,
            get_textblob_sentiment: 1.0,
            get_flair_sentiment: 1.5,
            get_nlptown_sentiment: 1.5,  # nlptown/bert-base-multilingual-uncased-sentiment
            get_bertweet_sentiment: 1.5,  # finiteautomata/bertweet-base-sentiment-analysis
            get_finbert_sentiment: 1.5,   # ProsusAI/finbert
        }
        
        # LRU cache for sentiment results to improve efficiency
        self._sentiment_cache = {}

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from sentiment.sentiment import (
    get_vader_sentiment, get_textblob_sentiment, get_flair_sentiment,
    get_nlptown_sentiment, get_finiteautomata_sentiment as get_bertweet_sentiment,
    get_ProsusAI_sentiment as get_finbert_sentiment,
)

class SentimentSystem:
    """Base class for sentiment analysis systems."""
    def analyze(self, text: str) -> float:
        raise NotImplementedError

class VADERSystem(SentimentSystem):
    """Simple VADER sentiment analyzer."""
    def analyze(self, text: str) -> float:
        return get_vader_sentiment(text)

class VADEREntityBaseline(SentimentSystem):
    """VADER with entity-aware context analysis."""
    def __init__(self):
        import spacy
        try:
            self.nlp = spacy.load("en_core_web_lg")
        except OSError:
            raise RuntimeError("SpaCy model not found. Run 'python -m spacy download en_core_web_lg'")
    
    def analyze_with_entity(self, text: str, entity: str) -> float:
        """Analyze sentiment with entity context."""
        if not entity:
            return get_vader_sentiment(text)
        
        doc = self.nlp(text)
        entity_lower = entity.lower()
        
        # Find entity position and extract context
        for i, token in enumerate(doc):
            if entity_lower in token.text.lower():
                entity_start = max(0, i - 5)
                entity_end = min(len(doc), i + 6)
                context_tokens = doc[entity_start:entity_end]
                context_text = " ".join([token.text for token in context_tokens])
                return get_vader_sentiment(context_text)
        
        return get_vader_sentiment(text)
    
    def analyze(self, text: str) -> float:
        """Standard analyze method for compatibility."""
        return get_vader_sentiment(text)

class PresetEnsembleSystem(SentimentSystem):
    """Optimized ensemble of multiple sentiment analyzers with caching and parallel processing."""
    def __init__(self):
        from concurrent.futures import ThreadPoolExecutor
        
        self.analyzers = {
            get_vader_sentiment: 1.0,
            get_textblob_sentiment: 1.0,
            get_flair_sentiment: 1.5,
            get_nlptown_sentiment: 1.5,
            get_bertweet_sentiment: 1.5,
            get_finbert_sentiment: 1.5
        }
        
        # Enhanced caching with size limit and cleanup
        self._sentiment_cache = {}
        self._cache_max_size = 1000
        
        # Thread pool for parallel sentiment analysis
        self._executor = ThreadPoolExecutor(max_workers=3)
        
        # Pre-compile enhancement patterns for efficiency
        self._strong_negative_patterns = {
            'overpriced', 'fell short', 'skip', 'avoid', 'terrible', 'awful', 
            'disgusting', 'dry', 'stale', 'burnt', 'cold', 'flavorless',
            'disappointing', 'bland', 'poor quality', 'bad taste'
        }
        
        self._strong_positive_patterns = {
            'fantastic', 'excellent', 'wonderful', 'amazing', 'delicious', 
            'perfect', 'outstanding', 'brilliant', 'superb', 'great', 'tasty',
            'fresh', 'flavorful', 'tender', 'juicy'
        }
        
        self._critical_patterns = {
            'should be more', 'should have been', 'needs to be', 'could be better',
            'would be better if', 'needs improvement', 'needs work', 'disappointing',
            'lacking in', 'falls short', 'not impressed', 'not satisfied'
        }
        
        self._quiet_patterns = {'not loud', 'not noisy', 'very quiet'}
        self._negated_noise_patterns = {
            "not audible", "aren't audible", "isn't audible", "not loud", "not noisy",
            "barely audible", "hardly audible", "inaudible", "silent", "quiet"
        }
        
    def _cleanup_cache(self):
        """Clean up cache when it gets too large."""
        if len(self._sentiment_cache) >= self._cache_max_size:
            # Remove oldest 20% of entries
            items_to_remove = list(self._sentiment_cache.keys())[:self._cache_max_size // 5]
            for key in items_to_remove:
                del self._sentiment_cache[key]
    
    def _apply_minimal_enhancement(self, text: str, base_score: float) -> float:
        """Simple, focused sentiment enhancement following OLD/graph architecture."""
        # Just return the base score - trust the ensemble
        return base_score

    def analyze_batch(self, texts: List[str]) -> List[float]:
        """Batch analysis with parallel processing."""
        results = []
        for text in texts:
            result = self.analyze(text)
            results.append(result)
        return results

    def analyze(self, text: str) -> float:
        """Optimized sentiment analysis with enhanced caching."""
        # Check cache first
        if text in self._sentiment_cache:
            return self._sentiment_cache[text]
        
        # Clean up cache if needed
        if len(self._sentiment_cache) >= self._cache_max_size:
            self._cleanup_cache()
            
        scores: List[float] = []
        weights: List[float] = []
        
        # Parallel execution for independent analyzers
        futures = []
        analyzer_list = list(self.analyzers.items())
        
        # Submit lightweight analyzers for parallel execution
        for analyzer_func, weight in analyzer_list[:3]:  # VADER, TextBlob, Flair
            future = self._executor.submit(self._safe_analyze, analyzer_func, text)
            futures.append((future, weight))
        
        # Execute heavier models sequentially to avoid memory issues
        for analyzer_func, weight in analyzer_list[3:]:  # Transformer models
            try:
                score = analyzer_func(text)
                if isinstance(score, (float, int)) and not np.isnan(score):
                    scores.append(np.clip(score, -1.0, 1.0))
                    weights.append(weight)
            except Exception:
                continue
        
        # Collect parallel results
        for future, weight in futures:
            try:
                score = future.result(timeout=10)  # 10-second timeout
                if isinstance(score, (float, int)) and not np.isnan(score):
                    scores.append(np.clip(score, -1.0, 1.0))
                    weights.append(weight)
            except Exception:
                continue
        
        if not scores:
            result = 0.0
        else:
            scores_arr = np.array(scores)
            weights_arr = np.array(weights)
            weighted_avg = np.average(scores_arr, weights=weights_arr)
            
            # Apply minimal enhancement
            enhanced_score = self._apply_minimal_enhancement(text, weighted_avg)
            
            # Temporarily disable neutral threshold for debugging
            # result = 0.0 if abs(enhanced_score) < 0.15 else enhanced_score
            result = enhanced_score
                
        # Cache result
        self._sentiment_cache[text] = result
        return result
    
    def analyze_with_context(self, text: str, entity_text: str = "", modifier_text: str = "") -> float:
        """Simple context-aware sentiment analysis following OLD/graph architecture."""
        # For now, just use the base sentiment - keep it simple and focused
        return self.analyze(text)
    
    def _safe_analyze(self, analyzer_func, text: str) -> float:
        """Safely execute analyzer with error handling."""
        try:
            return analyzer_func(text)
        except Exception:
            return 0.0
    
    def __del__(self):
        """Clean up thread pool."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)

class AdvancedTransformerSentiment(SentimentSystem):
    """
    Generalized transformer-based sentiment analysis that reduces manual rules.
    Uses pre-trained models that naturally handle negation, context, and domain adaptation.
    """
    def __init__(self):
        self.models = {}
        self.weights = {
            'roberta-base-sentiment': 0.35,      # General purpose, excellent negation handling
            'finbert': 0.25,                     # Financial domain (business reviews)
            'bertweet-sentiment': 0.20,          # Social media style text
            'cardiffnlp-sentiment': 0.20,        # Robust general sentiment
        }
        self._init_models()
    
    def _init_models(self):
        """Initialize transformer models with error handling."""
        model_configs = {
            'roberta-base-sentiment': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'finbert': 'ProsusAI/finbert', 
            'bertweet-sentiment': 'finiteautomata/bertweet-base-sentiment-analysis',
            'cardiffnlp-sentiment': 'j-hartmann/emotion-english-distilroberta-base'
        }
        
        for model_key, model_name in model_configs.items():
            try:
                self.models[model_key] = pipeline(
                    "sentiment-analysis", 
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1,
                    top_k=None
                )
            except Exception as e:
                print(f"Warning: Could not load {model_key}: {e}")
                # Remove from weights if model fails to load
                if model_key in self.weights:
                    del self.weights[model_key]
        
        # Normalize weights after potential model failures
        if self.weights:
            total_weight = sum(self.weights.values())
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
    
    def analyze(self, text: str) -> float:
        """Analyze sentiment using ensemble of transformer models."""
        if not self.models:
            # Fallback to VADER if no models loaded
            return get_vader_sentiment(text)
        
        scores = []
        weights = []
        
        for model_key, model in self.models.items():
            try:
                result = model(text)
                if isinstance(result, list) and len(result) > 0:
                    # Handle different output formats
                    if isinstance(result[0], list):
                        result = result[0]
                    
                    # Convert to standardized score (-1 to 1)
                    score = self._normalize_score(result)
                    scores.append(score)
                    weights.append(self.weights.get(model_key, 0.25))
                    
            except Exception as e:
                print(f"Warning: Error with {model_key}: {e}")
                continue
        
        if not scores:
            return get_vader_sentiment(text)  # Final fallback
        
        # Weighted average
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        return max(-1.0, min(1.0, weighted_score))  # Clamp to [-1, 1]
    
    def _normalize_score(self, result) -> float:
        """Normalize different model outputs to [-1, 1] scale."""
        if not result:
            return 0.0
        
        # Handle different output formats
        if isinstance(result, dict):
            label = result.get('label', '').lower()
            score = result.get('score', 0)
            
            # Common label mappings
            if 'pos' in label or 'positive' in label:
                return score
            elif 'neg' in label or 'negative' in label:
                return -score
            elif 'neutral' in label:
                return 0.0
        
        elif isinstance(result, list):
            # Multiple labels with scores
            positive_score = 0
            negative_score = 0
            
            for item in result:
                if isinstance(item, dict):
                    label = item.get('label', '').lower()
                    score = item.get('score', 0)
                    
                    if 'pos' in label or 'positive' in label:
                        positive_score += score
                    elif 'neg' in label or 'negative' in label:
                        negative_score += score
            
            return positive_score - negative_score
        
        return 0.0

class DeBERTaABSABaseline:
    """DeBERTa-based Aspect-Based Sentiment Analysis."""
    @lru_cache(maxsize=1)
    def _get_pipeline(self):
        model_name = "yangheng/deberta-v3-base-absa-v1.1"
        return pipeline("text-classification", model=model_name, device=0 if torch.cuda.is_available() else -1)

    def analyze(self, text: str, aspect: str = None) -> float:
        """Analyze sentiment for specific aspect using DeBERTa ABSA model."""
        if not aspect:
            return 0.0
            
        pipe = self._get_pipeline()
        result = pipe(text, text_pair=aspect)
        
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        
        label = result.get('label', '').lower()
        score = result.get('score', 0.0)
        
        if 'positive' in label:
            return score
        elif 'negative' in label:
            return -score
        else:
            return 0.0

class EfficiencyVADERSystem(SentimentSystem):
    def analyze(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0
        return get_vader_sentiment(text.strip())