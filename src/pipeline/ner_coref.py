import os
import warnings
import logging
import spacy
from collections import defaultdict
from dataclasses import dataclass, field
import json
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")

from transformers.utils import logging as tlog
tlog.set_verbosity_error()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model cache to avoid repeated loading
_MODEL_CACHE = {}

@lru_cache(maxsize=None)
def get_vader_lexicon():
    """Load VADER lexicon once."""
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.lexicon

@lru_cache(maxsize=2)
def get_cached_spacy_model(model_name: str):
    """Cache spaCy models globally to avoid repeated loading."""
    if model_name not in _MODEL_CACHE:
        logger.info(f"Loading spaCy model: {model_name}...")
        try:
            _MODEL_CACHE[model_name] = spacy.load(model_name)
        except OSError:
            raise
    return _MODEL_CACHE[model_name]

@lru_cache(maxsize=1)
def get_cached_maverick_model(device: str = "cpu"):
    """Cache Maverick model globally to avoid repeated loading."""
    cache_key = f"maverick_{device}"
    if cache_key not in _MODEL_CACHE:
        try:
            from maverick import Maverick
        except ImportError:
            raise RuntimeError("'maverick' package is required.")
        logger.info(f"Initializing Maverick with device: {device}")
        _MODEL_CACHE[cache_key] = Maverick(device=device)
    return _MODEL_CACHE[cache_key]

@lru_cache(maxsize=1)
def get_cached_ate_pipeline():
    """Cache ATE pipeline globally to avoid repeated loading."""
    if "ate_pipeline" not in _MODEL_CACHE:
        try:
            from transformers import pipeline
            logger.info("Loading ATE model...")
            # Use proper ATE model for aspect term extraction and sentiment
            _MODEL_CACHE["ate_pipeline"] = pipeline(
                "token-classification", 
                model="gauneg/roberta-base-absa-ate-sentiment",
                aggregation_strategy="simple"
            )
        except ImportError:
            logger.warning("transformers not available. Using spaCy NER fallback.")
            _MODEL_CACHE["ate_pipeline"] = None
        except Exception as e:
            logger.warning(f"Failed to load ATE model: {e}. Using spaCy NER fallback.")
            _MODEL_CACHE["ate_pipeline"] = None
    return _MODEL_CACHE["ate_pipeline"]

@dataclass(frozen=True)
class Mention:
    span: spacy.tokens.Span
    text: str = field(init=False)
    head_lemma: str = field(init=False)
    is_pronoun: bool = field(init=False)
    clause_id: int = field(init=False)  # Sentence number (0, 1, 2, ...)
    entity_id: int = field(default=None, init=False)  # Cluster ID (1, 2, 3, ...) set later

    def __post_init__(self):
        object.__setattr__(self, 'text', self.span.text.strip())
        object.__setattr__(self, 'head_lemma', self.span.root.lemma_.lower())
        object.__setattr__(self, 'is_pronoun', self.span.root.pos_ == 'PRON')
        
        # Determine clause_id from sentence boundaries (OLD graph pattern)
        doc_sentences = list(self.span.doc.sents)
        clause_id = 0
        for i, sent in enumerate(doc_sentences):
            if self.span.start >= sent.start and self.span.end <= sent.end:
                clause_id = i
                break
        object.__setattr__(self, 'clause_id', clause_id)
    
    def __hash__(self):
        # Use normalized text for better deduplication
        normalized_text = self.text.lower().strip()
        return hash((normalized_text, self.head_lemma, self.clause_id))
    
    def __eq__(self, other):
        if not isinstance(other, Mention):
            return False
        # Consider mentions equivalent if they have same head lemma and overlapping spans within same clause
        return (self.clause_id == other.clause_id and
                (self.head_lemma == other.head_lemma or 
                 self._spans_overlap(other) or
                 self._text_similarity(other) > 0.8))
    
    def _spans_overlap(self, other: 'Mention') -> bool:
        """Check if two mentions have overlapping character spans."""
        return not (self.span.end_char <= other.span.start_char or 
                   other.span.end_char <= self.span.start_char)
    
    def _text_similarity(self, other: 'Mention') -> float:
        """Calculate text similarity between mentions."""
        text1 = self.text.lower().strip()
        text2 = other.text.lower().strip()
        
        # Exact match
        if text1 == text2:
            return 1.0
        
        # One contains the other (e.g., "food" vs "the food")
        if text1 in text2 or text2 in text1:
            return 0.9
        
        # Same root word
        if self.head_lemma == other.head_lemma:
            return 0.85
        
        return 0.0

class NERCorefExtractor:
    """Base class for NER + Coreference extraction systems."""
    
    def analyze(self, text: str, similarity_threshold: float = 0.75) -> Dict[str, Any]:
        """Extract entities and resolve coreferences in text."""
        raise NotImplementedError("This method should be overridden by subclasses.")

class TransformerNERCorefExtractor(NERCorefExtractor):
    """ATE-first system with Maverick Coreference Resolution as addition."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.ate_pipeline = get_cached_ate_pipeline()
        self.mav = get_cached_maverick_model(device)
        self.nlp = get_cached_spacy_model("en_core_web_sm")
        
        # Category mapping for aspect consolidation
        self._category_mapping_cache = None
        
    def _get_category_mapping(self) -> Dict[str, str]:
        """Unified category mapping for aspect terms."""
        if self._category_mapping_cache is None:
            self._category_mapping_cache = {
                # Food-related terms
                'food': 'food', 'dish': 'food', 'meal': 'food', 'cuisine': 'food',
                'pizza': 'food', 'pasta': 'food', 'salad': 'food', 'soup': 'food', 
                'sandwich': 'food', 'burger': 'food', 'steak': 'food', 'chicken': 'food',
                'dessert': 'food', 'appetizer': 'food', 'entree': 'food', 'bread': 'food',
                
                # Service-related terms  
                'service': 'service', 'server': 'service', 'waiter': 'service', 
                'waitress': 'service', 'staff': 'service', 'waitstaff': 'service',
                'host': 'service', 'hostess': 'service', 'manager': 'service',
                
                # Place/ambience-related terms
                'place': 'place', 'restaurant': 'place', 'location': 'place',
                'spot': 'place', 'establishment': 'place', 'venue': 'place',
                'ambience': 'ambience', 'atmosphere': 'ambience', 'environment': 'ambience',
                'setting': 'ambience', 'decor': 'ambience', 'music': 'ambience',
                
                # Price/quality-related terms
                'price': 'price', 'cost': 'price', 'pricing': 'price', 'rates': 'price',
                'expensive': 'price', 'cheap': 'price', 'affordable': 'price', 'overpriced': 'price',
                'quality': 'quality', 'taste': 'quality', 'flavor': 'quality', 'texture': 'quality'
            }
        return self._category_mapping_cache
    
    def _extract_ate_aspects(self, text: str) -> List[Dict]:
        """Extract aspect terms using a hybrid approach: ATE model + filtered spaCy noun chunks."""
        aspect_spans = []
        processed_spans = set()
        vader_lexicon = get_vader_lexicon()

        # 1. Primary source: Enhanced NER/ATE model
        if self.ate_pipeline:
            try:
                ner_outputs = self.ate_pipeline(text)
                for output in ner_outputs:
                    word = output.get('word', '').strip()
                    if not word or word in ['.', ',', '!', '?', ';', ':']:
                        continue
                    
                    start, end = output.get('start', 0), output.get('end', 0)
                    aspect_spans.append({'text': word, 'start': start, 'end': end})
                    processed_spans.add((start, end))
            except Exception as e:
                logger.warning(f"ATE pipeline failed: {e}. Falling back to spaCy only.")

        # 2. Secondary source: spaCy noun chunks, filtered for opinion context
        doc = self.nlp(text)
        for chunk in doc.noun_chunks:
            start, end = chunk.start_char, chunk.end_char
            
            # Avoid adding duplicate or overlapping spans
            is_overlapping = any(max(start, p_start) < min(end, p_end) for p_start, p_end in processed_spans)
            if is_overlapping:
                continue

            # Filter noun chunks: must be associated with an opinion word
            is_opinion_related = False
            # Check words in the chunk itself
            for token in chunk:
                if token.lemma_.lower() in vader_lexicon:
                    is_opinion_related = True
                    break
            if is_opinion_related:
                aspect_spans.append({'text': chunk.text, 'start': start, 'end': end})
                processed_spans.add((start, end))
                continue

            # Check syntactic neighbors for opinion words
            for token in chunk:
                # Check head verb and its children
                if token.head.pos_ == 'VERB':
                    if token.head.lemma_.lower() in vader_lexicon:
                        is_opinion_related = True
                        break
                    for child in token.head.children:
                        if child.lemma_.lower() in vader_lexicon:
                            is_opinion_related = True
                            break
                # Check adjectives modifying the chunk's root
                if token.dep_ == 'ROOT':
                    for child in token.children:
                        if child.dep_ == 'amod' and child.lemma_.lower() in vader_lexicon:
                            is_opinion_related = True
                            break
            
            if is_opinion_related:
                aspect_spans.append({'text': chunk.text, 'start': start, 'end': end})
                processed_spans.add((start, end))
                
        return aspect_spans
    
    def _deduplicate_aspects(self, aspect_spans: List[Dict]) -> List[Dict]:
        """Remove duplicate aspects, preferring primary ATE detections."""
        seen_aspects = {}
        result = []
        
        # Sort by primary flag (True first) to prefer ATE model detections
        sorted_spans = sorted(aspect_spans, key=lambda x: not x['is_primary'])
        
        for span in sorted_spans:
            # Normalize text for comparison
            normalized_text = span['text'].lower().strip()
            
            # Skip if we've already seen this aspect
            if normalized_text not in seen_aspects:
                seen_aspects[normalized_text] = span
                result.append(span)
        
        return result
    
    def _extract_rule_based_aspects(self, text: str) -> List[Dict]:
        """Extract aspects using rule-based patterns as fallback."""
        import re
        
        # Hardware/technical terms patterns
        hardware_patterns = [
            r'\b(?:hard\s+drive|hdd|ssd)\b',
            r'\b(?:battery\s+life|battery)\b',
            r'\b(?:screen\s+quality|display|screen|monitor)\b',
            r'\b(?:keyboard|keys|trackpad|touchpad)\b',
            r'\b(?:processor|cpu|performance)\b',
            r'\b(?:memory|ram|storage)\b',
            r'\b(?:boot[s]?\s+up|startup|start[s]?\s+up|boots?\s*up)\b',
            r'\b(?:usb\d*|port|connectivity)\b',
            r'\b(?:cd\s+drive|dvd\s+drive|internal\s+cd\s+drive)\b',
            r'\b(?:operating\s+system|os|windows|macos|linux)\b',
            r'\b(?:size|weight|design|build\s+quality)\b',
            r'\b(?:price|cost|pricing|expensive|cheap|affordable)\b'
        ]
        
        # Restaurant/food terms patterns  
        food_patterns = [
            r'\b(?:food|dish|meal|cuisine|menu)\b',
            r'\b(?:service|staff|waiter|waitress|server)\b',
            r'\b(?:ambience|atmosphere|environment|decor)\b',
            r'\b(?:price|cost|pricing|bill|expensive|cheap)\b',
            r'\b(?:portion|portions|serving)\b',
            r'\b(?:taste|flavor|delicious|tasty)\b',
            r'\b(?:drinks|beverages|wine|beer)\b',
            r'\b(?:pizza|pasta|burger|sandwich|salad|soup)\b'
        ]
        
        all_patterns = hardware_patterns + food_patterns
        aspect_spans = []
        
        for pattern in all_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                aspect_text = match.group().strip()
                if len(aspect_text) > 1:  # Avoid single characters
                    aspect_spans.append({
                        'text': aspect_text,
                        'start': match.start(),
                        'end': match.end(),
                        'score': 0.7,  # Rule-based confidence
                        'entity_type': 'ASPECT',
                        'is_primary': False  # Mark as fallback detection
                    })
        
        return aspect_spans
    
    def _create_mention_from_ate(self, doc: spacy.tokens.Doc, ate_span: Dict) -> Optional[Mention]:
        """Create a Mention object from an ATE span."""
        try:
            # Find the spaCy token span that matches the ATE span
            char_start, char_end = ate_span['start'], ate_span['end']
            
            # Find tokens that overlap with the character span
            start_token = None
            end_token = None
            
            for token in doc:
                if token.idx <= char_start < token.idx + len(token.text):
                    start_token = token.i
                if token.idx < char_end <= token.idx + len(token.text):
                    end_token = token.i + 1
                    break
            
            if start_token is not None and end_token is not None:
                span = doc[start_token:end_token]
                return Mention(span)
                
        except Exception as e:
            logger.warning(f"Failed to create mention from ATE span {ate_span}: {e}")
        
        return None
    
    def _extract_additional_mentions(self, doc: spacy.tokens.Doc, text: str, ate_mentions: List[Mention]) -> List[Mention]:
        """Extract additional mentions from spaCy (supplement to ATE)."""
        additional_mentions = []
        seen_mentions = set(ate_mentions)
        
        # Supplement with spaCy entities and noun chunks (secondary)
        for span in list(doc.ents) + list(doc.noun_chunks):
            if span.root.pos_ in ('NOUN', 'PROPN', 'PRON'):
                candidate_mention = Mention(span)
                # Only add if not already covered by ATE
                is_duplicate = any(candidate_mention == existing for existing in seen_mentions)
                if not is_duplicate:
                    additional_mentions.append(candidate_mention)
                    seen_mentions.add(candidate_mention)
        
        # Add missed aspect keywords (tertiary)
        aspect_keywords = {
            'price': {'overpriced', 'expensive', 'cheap', 'costly', 'affordable', 'pricey', 'price', 'cost', 'pricing', 'rates'},
            'quality': {'excellent', 'terrible', 'amazing', 'awful', 'outstanding', 'poor', 'great', 'bad', 'good', 'wonderful', 'horrible', 'quality', 'taste', 'flavor', 'texture'},
            'service': {'server', 'waiter', 'waitress', 'staff', 'service', 'serving', 'served', 'host', 'hostess', 'manager', 'employee'},
            'food': {'dish', 'meal', 'cuisine', 'food', 'pizza', 'pasta', 'salad', 'soup', 'sandwich', 'burger', 'steak', 'chicken', 'dessert', 'appetizer', 'entree', 'bread', 'drinks', 'drink', 'beverage'},
            'ambience': {'atmosphere', 'ambience', 'environment', 'decor', 'music', 'setting', 'lighting', 'noise', 'mood'},
            'technology': {'windows', 'software', 'system', 'computer', 'laptop', 'screen', 'keyboard', 'mouse', 'drive', 'hardware', 'performance', 'speed', 'battery'},
            'product': {'product', 'item', 'device', 'machine', 'unit', 'model'}
        }
        
        for token in doc:
            if token.lemma_.lower() in {kw for kws in aspect_keywords.values() for kw in kws}:
                candidate_mention = Mention(doc[token.i:token.i+1])
                is_duplicate = any(candidate_mention == existing for existing in seen_mentions)
                if not is_duplicate:
                    additional_mentions.append(candidate_mention)
                    seen_mentions.add(candidate_mention)
        
        return additional_mentions
    
    def _extract_mentions(self, doc: spacy.tokens.Doc, text: str) -> List[Mention]:
        """Extract mentions combining ATE and spaCy entities."""
        mentions = []
        seen_mentions = set()
        
        # First, get ATE aspect terms (high precision)
        ate_aspects = self._extract_ate_aspects(text)
        for ate_span in ate_aspects:
            mention = self._create_mention_from_ate(doc, ate_span)
            if mention and mention not in seen_mentions:
                mentions.append(mention)
                seen_mentions.add(mention)
        
        # Supplement with spaCy entities and noun chunks
        for span in list(doc.ents) + list(doc.noun_chunks):
            if span.root.pos_ in ('NOUN', 'PROPN', 'PRON'):
                candidate_mention = Mention(span)
                # Check if this mention is similar to any existing mention
                is_duplicate = any(candidate_mention == existing for existing in seen_mentions)
                if not is_duplicate:
                    mentions.append(candidate_mention)
                    seen_mentions.add(candidate_mention)
        
        # Add aspect keywords that might be missed
        aspect_keywords = {
            'price': {'overpriced', 'expensive', 'cheap', 'costly', 'affordable', 'pricey', 'price', 'cost', 'pricing', 'rates'},
            'quality': {'excellent', 'terrible', 'amazing', 'awful', 'outstanding', 'poor', 'great', 'bad', 'good', 'wonderful', 'horrible', 'quality', 'taste', 'flavor', 'texture'},
            'service': {'server', 'waiter', 'waitress', 'staff', 'service', 'serving', 'served', 'host', 'hostess', 'manager', 'employee'},
            'food': {'dish', 'meal', 'cuisine', 'food', 'pizza', 'pasta', 'salad', 'soup', 'sandwich', 'burger', 'steak', 'chicken', 'dessert', 'appetizer', 'entree', 'bread', 'drinks', 'drink', 'beverage'},
            'ambience': {'atmosphere', 'ambience', 'environment', 'decor', 'music', 'setting', 'lighting', 'noise', 'mood'},
            'technology': {'windows', 'software', 'system', 'computer', 'laptop', 'screen', 'keyboard', 'mouse', 'drive', 'hardware', 'performance', 'speed', 'battery'},
            'product': {'product', 'item', 'device', 'machine', 'unit', 'model'}
        }
        
        for token in doc:
            if token.lemma_.lower() in {kw for kws in aspect_keywords.values() for kw in kws}:
                candidate_mention = Mention(doc[token.i:token.i+1])
                is_duplicate = any(candidate_mention == existing for existing in seen_mentions)
                if not is_duplicate:
                    mentions.append(candidate_mention)
                    seen_mentions.add(candidate_mention)
        
        return mentions
    
    def _link_ate_to_coref_clusters(self, ate_aspects: List[Dict], coref_clusters: List[List[Tuple[int, int]]]) -> List[int]:
        """Link ATE aspect spans to coreference clusters."""
        aspect_cluster_ids = []
        
        for ate_span in ate_aspects:
            ate_start, ate_end = ate_span['start'], ate_span['end']
            found_cluster = None
            
            # Find which coreference cluster contains this ATE span
            for cluster_id, cluster in enumerate(coref_clusters):
                for coref_start, coref_end in cluster:
                    # Check if ATE span overlaps with coreference span
                    if (ate_start >= coref_start and ate_end <= coref_end) or \
                       (coref_start >= ate_start and coref_start < ate_end) or \
                       (coref_end > ate_start and coref_end <= ate_end):
                        found_cluster = cluster_id
                        break
                if found_cluster is not None:
                    break
            
            aspect_cluster_ids.append(found_cluster)
        
        return aspect_cluster_ids
    
    def analyze(self, text: str, similarity_threshold: float = 0.75) -> Dict[str, Any]:
        """Analyze text using ATE-first approach with Maverick coreference resolution."""
        doc = self.nlp(text)
        
        # PRIMARY: Extract ATE aspects (highest priority)
        ate_aspects = self._extract_ate_aspects(text)
        
        # Convert ATE aspects to mentions
        ate_mentions = []
        for ate_span in ate_aspects:
            mention = self._create_mention_from_ate(doc, ate_span)
            if mention:
                ate_mentions.append(mention)
        
        # SECONDARY: Extract additional mentions not covered by ATE
        additional_mentions = self._extract_additional_mentions(doc, text, ate_mentions)
        
        # Combine all mentions (ATE first, then additional)
        all_mentions = ate_mentions + additional_mentions
        
        # ADDITION: Get Maverick coreference clusters for enhancement
        coref_output = self.mav.predict(text)
        coref_clusters = coref_output["clusters_char_offsets"]
        
        # Build mention-to-cluster mapping using coreference
        span_to_cluster_id = {}
        mention_spans = {(m.span.start_char, m.span.end_char): m for m in all_mentions}
        
        # Map mentions to Maverick clusters (enhancement)
        for i, chain in enumerate(coref_clusters):
            for s, e in chain:
                for (start, end), mention in mention_spans.items():
                    if start >= s and end <= e + 1:
                        span_to_cluster_id[(start, end)] = i
        
        # Group mentions into clusters (ATE-based with coreference enhancement)
        clusters = []
        head_lemma_to_cluster = {}
        
        for mention in all_mentions:
            mav_cluster_id = span_to_cluster_id.get((mention.span.start_char, mention.span.end_char))
            
            found_cluster = None
            if not mention.is_pronoun and mention.head_lemma in head_lemma_to_cluster:
                found_cluster = head_lemma_to_cluster[mention.head_lemma]
            
            if mav_cluster_id is not None:
                # Find or create cluster for this Maverick cluster ID
                related_cluster_idx = None
                for idx, cluster in enumerate(clusters):
                    for m in cluster:
                        if span_to_cluster_id.get((m.span.start_char, m.span.end_char)) == mav_cluster_id:
                            related_cluster_idx = idx
                            break
                    if related_cluster_idx is not None:
                        break
                
                if related_cluster_idx is not None:
                    clusters[related_cluster_idx].append(mention)
                    if not mention.is_pronoun:
                        head_lemma_to_cluster[mention.head_lemma] = related_cluster_idx
                else:
                    new_idx = len(clusters)
                    clusters.append([mention])
                    if not mention.is_pronoun:
                        head_lemma_to_cluster[mention.head_lemma] = new_idx
            elif found_cluster is not None:
                clusters[found_cluster].append(mention)
            else:
                new_idx = len(clusters)
                clusters.append([mention])
                if not mention.is_pronoun:
                    head_lemma_to_cluster[mention.head_lemma] = new_idx
        
        # Build final output with clause information
        final_output = {}
        category_mapping = self._get_category_mapping()
        
        # Assign entity IDs to clusters (OLD graph pattern)
        for cluster_idx, cluster in enumerate(clusters):
            entity_id = cluster_idx + 1  # Entity IDs start from 1
            
            # Assign entity_id to all mentions in this cluster
            for mention in cluster:
                object.__setattr__(mention, 'entity_id', entity_id)
        
        for cluster in clusters:
            non_pronouns = [m for m in cluster if not m.is_pronoun]
            if not non_pronouns:
                continue
            
            # Choose the longest, most descriptive mention as canonical (prefer ATE findings)
            canonical_name = max(non_pronouns, key=lambda m: (len(m.text), not m.text.lower().startswith('the'))).text
            
            # Collect all unique mention texts, avoiding duplicates, maintaining text order
            mentions_text = []
            seen_texts = set()
            
            # Sort mentions by their position in text to maintain order
            sorted_mentions = sorted(cluster, key=lambda m: m.span.start_char)
            
            for m in sorted_mentions:
                text_clean = m.text.strip()
                if (text_clean.lower() not in ["itself", "it", "they", "them"] and 
                    text_clean not in seen_texts):
                    mentions_text.append(text_clean)
                    seen_texts.add(text_clean)
            
            # Remove subset mentions while preserving order
            filtered_mentions = []
            for mention in mentions_text:
                is_subset = False
                for other_mention in mentions_text:
                    if (mention != other_mention and 
                        mention.lower() in other_mention.lower() and 
                        len(mention) < len(other_mention)):
                        is_subset = True
                        break
                if not is_subset:
                    filtered_mentions.append(mention)
            
            if filtered_mentions:
                # Get entity_id and clause info for OLD graph compatibility
                entity_id = cluster[0].entity_id  # All mentions in cluster have same entity_id
                clause_info = {}
                
                for m in cluster:
                    clause_id = m.clause_id  # Sentence number
                    if clause_id not in clause_info:
                        clause_info[clause_id] = []
                    clause_info[clause_id].append(m.text.strip())
                
                final_output[canonical_name] = {
                    "mentions": filtered_mentions,
                    "entity_id": entity_id,  # For OLD graph node creation
                    "clause_info": clause_info  # For temporal layer construction: {clause_id: [mentions]}
                }
        
        # Apply category mapping
        mapped_output = {}
        category_sentiments = defaultdict(list)
        
        for canonical_name, data in final_output.items():
            # Check if canonical name maps to a category
            category = canonical_name.lower()
            for mention in data["mentions"]:
                if mention.lower() in category_mapping:
                    category = category_mapping[mention.lower()]
                    break
            
            if category not in mapped_output:
                mapped_output[category] = {
                    "mentions": [], 
                    "entity_ids": [],  # Track entity IDs for OLD graph compatibility
                    "clause_info": {}
                }
            
            mapped_output[category]["mentions"].extend(data["mentions"])
            mapped_output[category]["entity_ids"].append(data["entity_id"])
            
            # Merge clause info
            for clause_id, mentions in data["clause_info"].items():
                if clause_id not in mapped_output[category]["clause_info"]:
                    mapped_output[category]["clause_info"][clause_id] = []
                mapped_output[category]["clause_info"][clause_id].extend(mentions)
            
            category_sentiments[category].append(canonical_name)
        
        # Remove duplicates but preserve entity_id structure
        for category in mapped_output:
            mapped_output[category]["mentions"] = list(set(mapped_output[category]["mentions"]))
            mapped_output[category]["entity_ids"] = list(set(mapped_output[category]["entity_ids"]))
        
        return mapped_output


class EntityConsolidator(NERCorefExtractor):
    """Legacy spaCy + Maverick coreference system (for ablation testing)."""
    def __init__(self, device: str = "-1"):
        self.device = self._cpu(device)
        self.mav = get_cached_maverick_model(self.device)
        self.nlp = get_cached_spacy_model("en_core_web_lg")
        
        # Add caching for expensive operations
        self._category_mapping_cache = None
        self._aspect_keywords_cache = None
        
    def _cpu(self, d):
        return "cpu" if d in (None, "-1", "CPU", "cpu") else d

    def _get_cached_category_mapping(self) -> Dict[str, str]:
        """Cached category mapping to avoid repeated dictionary creation."""
        if self._category_mapping_cache is None:
            self._category_mapping_cache = {
                # Food-related terms
                'dessert': 'food', 'pizza': 'food', 'pasta': 'food', 'salad': 'food', 'soup': 'food',
                'burger': 'food', 'sandwich': 'food', 'steak': 'food', 'chicken': 'food', 'fish': 'food',
                'bread': 'food', 'cheese': 'food', 'vegetables': 'food', 'fruit': 'food', 'wine': 'food',
                'beer': 'food', 'coffee': 'food', 'tea': 'food', 'drink': 'food', 'beverage': 'food',
                'appetizer': 'food', 'entree': 'food', 'main course': 'food', 'side dish': 'food',
                'the dessert': 'food', 'the pizza': 'food', 'the dish': 'food', 'the meal': 'food',
                
                # Price-related terms
                'overpriced': 'price', 'expensive': 'price', 'cheap': 'price', 'costly': 'price',
                'affordable': 'price', 'pricey': 'price', 'cost': 'price', 'pricing': 'price',
                
                # Service-related terms
                'server': 'service', 'waiter': 'service', 'waitress': 'service', 'staff': 'service',
                'host': 'service', 'hostess': 'service', 'manager': 'service', 'employee': 'service',
                
                # Quality-related terms
                'quality': 'quality', 'taste': 'quality', 'flavor': 'quality', 'texture': 'quality',
                
                # Ambience-related terms
                'atmosphere': 'ambience', 'environment': 'ambience', 'setting': 'ambience',
                'decor': 'ambience', 'music': 'ambience', 'lighting': 'ambience', 'noise': 'ambience'
            }
        return self._category_mapping_cache

    def _resolve_implicit_entities(self, doc: spacy.tokens.Doc, text: str) -> List[Mention]:
        """Resolve implicit food entities from context like 'They were dry' -> food."""
        implicit_mentions = []
        
        # Food context indicators
        food_context_words = ['dry', 'stale', 'flavorless', 'tasteless', 'crispy', 'juicy', 'tender', 'tough', 'bland', 'spicy', 'sweet', 'sour', 'salty', 'fresh', 'rotten']
        quality_adjectives = ['excellent', 'terrible', 'amazing', 'awful', 'outstanding', 'poor', 'great', 'bad', 'good', 'wonderful', 'horrible']
        
        for sent in doc.sents:
            # Check for pronouns with food-related adjectives
            pronouns = [token for token in sent if token.pos_ == 'PRON' and token.text.lower() in ['they', 'it', 'them']]
            
            for pronoun in pronouns:
                # Look for food context words in the same sentence
                food_context_found = any(token.text.lower() in food_context_words for token in sent)
                quality_context_found = any(token.text.lower() in quality_adjectives for token in sent)
                
                if food_context_found or quality_context_found:
                    # Create an implicit food mention at the pronoun position
                    implicit_span = doc[pronoun.i:pronoun.i+1]
                    implicit_mention = type('ImplicitMention', (), {
                        'span': implicit_span,
                        'text': 'food',  # Map to food category
                        'head_lemma': 'food',
                        'is_pronoun': False  # Treat as entity, not pronoun
                    })()
                    implicit_mentions.append(implicit_mention)
                    
        return implicit_mentions

    def _extract_mentions(self, doc: spacy.tokens.Doc):
        """Extract mentions with optimized aspect detection."""
        mentions = set()
        
        # Faster entity and noun chunk extraction
        for span in list(doc.ents) + list(doc.noun_chunks):
            if span.root.pos_ in ('NOUN', 'PROPN', 'PRON'):
                mentions.add(Mention(span))
        
        # Optimized aspect detection with compiled patterns
        aspect_keywords = {
            'price': {'overpriced', 'expensive', 'cheap', 'costly', 'affordable', 'pricey', 'price', 'cost', 'pricing', 'rates'},
            'quality': {'excellent', 'terrible', 'amazing', 'awful', 'outstanding', 'poor', 'great', 'bad'},
            'service': {'service', 'server', 'staff', 'waiter', 'waitress', 'serving', 'served'},
            'food': {'food', 'dish', 'meal', 'cuisine', 'recipe', 'cooking'},
            'ambience': {'atmosphere', 'ambience', 'ambiance', 'environment', 'setting', 'mood'}
        }
        
        # Build reverse lookup for faster matching
        keyword_to_aspect = {}
        for aspect_type, keywords in aspect_keywords.items():
            for keyword in keywords:
                keyword_to_aspect[keyword] = aspect_type
        
        # Vectorized token processing
        for token in doc:
            token_lower = token.text.lower()
            if token_lower in keyword_to_aspect:
                aspect_span = doc[token.i:token.i+1]
                mentions.add(Mention(aspect_span))
        
        # Optimized implicit entity resolution
        implicit_mentions = self._resolve_implicit_entities_fast(doc)
        mentions.update(implicit_mentions)
        
        return sorted(list(mentions), key=lambda m: m.span.start_char)

    def _resolve_implicit_entities_fast(self, doc: spacy.tokens.Doc) -> List[Mention]:
        """Optimized implicit food entity resolution."""
        implicit_mentions = []
        
        # Pre-compiled sets for faster lookup
        food_context_words = {'dry', 'stale', 'flavorless', 'tasteless', 'crispy', 'juicy', 'tender', 'tough', 'bland', 'spicy', 'sweet', 'sour', 'salty', 'fresh', 'rotten'}
        quality_adjectives = {'excellent', 'terrible', 'amazing', 'awful', 'outstanding', 'poor', 'great', 'bad', 'good', 'wonderful', 'horrible'}
        pronoun_targets = {'they', 'it', 'them'}
        
        for sent in doc.sents:
            # Collect pronouns and context words in one pass
            pronouns = []
            has_food_context = False
            has_quality_context = False
            
            for token in sent:
                token_lower = token.text.lower()
                if token.pos_ == 'PRON' and token_lower in pronoun_targets:
                    pronouns.append(token)
                elif token_lower in food_context_words:
                    has_food_context = True
                elif token_lower in quality_adjectives:
                    has_quality_context = True
            
            # Create implicit mentions if context is present
            if (has_food_context or has_quality_context) and pronouns:
                for pronoun in pronouns:
                    implicit_span = doc[pronoun.i:pronoun.i+1]
                    implicit_mention = type('ImplicitMention', (), {
                        'span': implicit_span,
                        'text': 'food',
                        'head_lemma': 'food',
                        'is_pronoun': False
                    })()
                    implicit_mentions.append(implicit_mention)
                    
        return implicit_mentions
                    
    def _get_category_mapping(self) -> Dict[str, str]:
        """Map specific entities to general aspect categories."""
        return {
            # Food-related terms
            'dessert': 'food', 'pizza': 'food', 'pasta': 'food', 'salad': 'food', 'soup': 'food',
            'burger': 'food', 'sandwich': 'food', 'steak': 'food', 'chicken': 'food', 'fish': 'food',
            'bread': 'food', 'cheese': 'food', 'vegetables': 'food', 'fruit': 'food', 'wine': 'food',
            'beer': 'food', 'coffee': 'food', 'tea': 'food', 'drink': 'food', 'beverage': 'food',
            'appetizer': 'food', 'entree': 'food', 'main course': 'food', 'side dish': 'food',
            'the dessert': 'food', 'the pizza': 'food', 'the dish': 'food', 'the meal': 'food',
            
            # Price-related terms
            'overpriced': 'price', 'expensive': 'price', 'cheap': 'price', 'costly': 'price',
            'affordable': 'price', 'pricey': 'price', 'cost': 'price', 'pricing': 'price',
            
            # Service-related terms
            'server': 'service', 'waiter': 'service', 'waitress': 'service', 'staff': 'service',
            'host': 'service', 'hostess': 'service', 'manager': 'service', 'employee': 'service',
            
            # Quality-related terms
            'quality': 'quality', 'taste': 'quality', 'flavor': 'quality', 'texture': 'quality',
            
            # Ambience-related terms
            'atmosphere': 'ambience', 'environment': 'ambience', 'setting': 'ambience',
            'decor': 'ambience', 'music': 'ambience', 'lighting': 'ambience', 'noise': 'ambience'
        }

    def _apply_category_mapping(self, final_output: Dict) -> Dict:
        """Apply category mapping to consolidate entities into standard categories."""
        category_mapping = self._get_cached_category_mapping()
        mapped_output = {}
        category_sentiments = defaultdict(list)
        
        # First pass: collect entities by category
        for canonical_name, data in final_output.items():
            category = category_mapping.get(canonical_name.lower(), canonical_name)
            if category not in mapped_output:
                mapped_output[category] = {"mentions": []}
            
            # Merge mentions and track for sentiment aggregation
            mapped_output[category]["mentions"].extend(data["mentions"])
            category_sentiments[category].append(canonical_name)
        
        # Remove duplicates from mentions more efficiently
        for category in mapped_output:
            mapped_output[category]["mentions"] = list(set(mapped_output[category]["mentions"]))
            
        return mapped_output

    def analyze_batch(self, texts: List[str], similarity_threshold=0.75) -> List[Dict]:
        """Analyze multiple texts in batch to amortize model loading costs."""
        results = []
        for text in texts:
            try:
                result = self.analyze(text, similarity_threshold)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing text '{text[:50]}...': {e}")
                results.append({})
        return results

    def _enhanced_coreference_resolution(self, doc: spacy.tokens.Doc, all_mentions: List['Mention']) -> Dict[Tuple, int]:
        """Enhanced coreference resolution with generalized patterns."""
        span_to_cluster_id = {}
        
        # Get maverick clusters
        mav_clusters = self.mav.predict(doc.text)["clusters_char_offsets"]
        mention_spans = {(m.span.start_char, m.span.end_char): m for m in all_mentions}
        
        # Map mentions to maverick clusters
        for i, chain in enumerate(mav_clusters):
            for s, e in chain:
                for (start, end), mention in mention_spans.items():
                    if start >= s and end <= e + 1:
                        span_to_cluster_id[(start, end)] = i
        
        # Enhanced pattern-based coreference beyond maverick
        pronoun_patterns = {
            'they': {'food', 'dishes', 'items', 'things'},
            'it': {'food', 'dish', 'item', 'thing', 'meal', 'experience'},
            'them': {'food', 'dishes', 'items'},
            'this': {'food', 'dish', 'place', 'restaurant', 'experience'},
            'that': {'food', 'dish', 'place', 'restaurant', 'experience'},
            'these': {'food', 'dishes', 'items'},
            'those': {'food', 'dishes', 'items'}
        }
        
        # Distance-based pronoun resolution
        for mention in all_mentions:
            if mention.is_pronoun and mention.text.lower() in pronoun_patterns:
                pronoun_pos = mention.span.start_char
                best_antecedent = None
                min_distance = float('inf')
                
                # Look for closest compatible antecedent within 3 sentences
                for candidate in all_mentions:
                    if (not candidate.is_pronoun and 
                        candidate.span.start_char < pronoun_pos and
                        pronoun_pos - candidate.span.start_char < 300):  # ~3 sentences
                        
                        distance = pronoun_pos - candidate.span.start_char
                        
                        # Check semantic compatibility
                        candidate_category = self._get_entity_category(candidate.text.lower())
                        compatible_categories = pronoun_patterns[mention.text.lower()]
                        
                        if candidate_category in compatible_categories and distance < min_distance:
                            min_distance = distance
                            best_antecedent = candidate
                
                # Link pronoun to best antecedent
                if best_antecedent:
                    mention_key = (mention.span.start_char, mention.span.end_char)
                    antecedent_key = (best_antecedent.span.start_char, best_antecedent.span.end_char)
                    
                    # Use same cluster ID for both
                    if antecedent_key in span_to_cluster_id:
                        span_to_cluster_id[mention_key] = span_to_cluster_id[antecedent_key]
                    else:
                        # Create new cluster
                        new_cluster_id = max(span_to_cluster_id.values(), default=-1) + 1
                        span_to_cluster_id[mention_key] = new_cluster_id
                        span_to_cluster_id[antecedent_key] = new_cluster_id
        
        return span_to_cluster_id
    
    def _get_entity_category(self, text: str) -> str:
        """Determine entity category for coreference compatibility."""
        category_mapping = self._get_cached_category_mapping()
        
        # Direct mapping
        if text in category_mapping:
            return category_mapping[text]
        
        # Pattern-based categorization
        food_patterns = {'dish', 'meal', 'food', 'dessert', 'pizza', 'pasta', 'salad', 'burger'}
        service_patterns = {'server', 'waiter', 'staff', 'service'}
        place_patterns = {'restaurant', 'place', 'location', 'establishment'}
        
        text_words = set(text.split())
        
        if food_patterns.intersection(text_words):
            return 'food'
        elif service_patterns.intersection(text_words):
            return 'service'
        elif place_patterns.intersection(text_words):
            return 'place'
        
    def analyze(self, text: str, similarity_threshold=0.75):
        doc = self.nlp(text)
        all_mentions = self._extract_mentions(doc)
        
        # Use enhanced coreference resolution
        span_to_cluster_id = self._enhanced_coreference_resolution(doc, all_mentions)
        
        # Build clusters with optimized lookups
        clusters = []
        head_lemma_to_cluster = {}  # Cache for faster head lemma lookups
        
        for mention in all_mentions:
            mav_cluster_id = span_to_cluster_id.get((mention.span.start_char, mention.span.end_char))
            
            found_cluster = None
            if not mention.is_pronoun and mention.head_lemma in head_lemma_to_cluster:
                found_cluster = head_lemma_to_cluster[mention.head_lemma]

            if mav_cluster_id is not None:
                # Find related clusters more efficiently
                related_clusters_indices = []
                for idx, cluster in enumerate(clusters):
                    for m in cluster:
                        if span_to_cluster_id.get((m.span.start_char, m.span.end_char)) == mav_cluster_id:
                            related_clusters_indices.append(idx)
                            break
                
                if related_clusters_indices:
                    master_idx = related_clusters_indices[0]
                    # Merge clusters in reverse order to maintain indices
                    for idx in sorted(related_clusters_indices[1:], reverse=True):
                        clusters[master_idx].extend(clusters.pop(idx))
                        # Update head lemma cache
                        for m in clusters[master_idx]:
                            if not m.is_pronoun:
                                head_lemma_to_cluster[m.head_lemma] = master_idx
                    
                    clusters[master_idx].append(mention)
                    if not mention.is_pronoun:
                        head_lemma_to_cluster[mention.head_lemma] = master_idx
                elif found_cluster is not None:
                    clusters[found_cluster].append(mention)
                    if not mention.is_pronoun:
                        head_lemma_to_cluster[mention.head_lemma] = found_cluster
                else:
                    new_idx = len(clusters)
                    clusters.append([mention])
                    if not mention.is_pronoun:
                        head_lemma_to_cluster[mention.head_lemma] = new_idx
            elif found_cluster is not None:
                clusters[found_cluster].append(mention)
            else:
                new_idx = len(clusters)
                clusters.append([mention])
                if not mention.is_pronoun:
                    head_lemma_to_cluster[mention.head_lemma] = new_idx

        # Optimized semantic merging with early termination
        merged = True
        while merged:
            merged = False
            # Cache representatives to avoid repeated calculations
            representatives = []
            for cluster in clusters:
                rep = next((m.span for m in cluster if not m.is_pronoun), cluster[0].span)
                representatives.append(rep)
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    if i >= len(clusters) or j >= len(clusters): 
                        continue
                    
                    rep1, rep2 = representatives[i], representatives[j]
                    
                    if (rep1.has_vector and rep2.has_vector and 
                        rep1.similarity(rep2) > similarity_threshold):
                        logger.info(f"Semantic Merge: Merging '{rep2.text}' into '{rep1.text}'")
                        clusters[i].extend(clusters.pop(j))
                        representatives.pop(j)  # Update representatives cache
                        merged = True
                        break
                if merged: 
                    break
                        
        final_output = {}
        for cluster in clusters:
            non_pronouns = [m for m in cluster if not m.is_pronoun]
            if not non_pronouns: continue
            
            canonical_name = max(non_pronouns, key=lambda m: len(m.text)).text
            
            mentions_text = {m.text for m in cluster if m.text.lower() not in ["itself"]}
            sorted_mentions = sorted(list(mentions_text), key=lambda m: text.find(m))
            
            if sorted_mentions:
                final_output[canonical_name] = {"mentions": sorted_mentions}
        
        # Apply category mapping to consolidate similar entities
        mapped_output = self._apply_category_mapping(final_output)
        return mapped_output
        doc = self.nlp(text)
        all_mentions = self._extract_mentions(doc)
        
        # Optimize maverick clustering with early termination
        span_to_cluster_id = {}
        mav_clusters = self.mav.predict(text)["clusters_char_offsets"]
        
        # Build efficient lookup for mention spans
        mention_spans = {(m.span.start_char, m.span.end_char): m for m in all_mentions}
        
        for i, chain in enumerate(mav_clusters):
            for s, e in chain:
                # Direct lookup instead of linear search
                for (start, end), mention in mention_spans.items():
                    if start >= s and end <= e + 1:
                        span_to_cluster_id[(start, end)] = i
                        
        # Build clusters with optimized lookups
        clusters = []
        head_lemma_to_cluster = {}  # Cache for faster head lemma lookups
        
        for mention in all_mentions:
            mav_cluster_id = span_to_cluster_id.get((mention.span.start_char, mention.span.end_char))
            
            found_cluster = None
            if not mention.is_pronoun and mention.head_lemma in head_lemma_to_cluster:
                found_cluster = head_lemma_to_cluster[mention.head_lemma]

            if mav_cluster_id is not None:
                # Find related clusters more efficiently
                related_clusters_indices = []
                for idx, cluster in enumerate(clusters):
                    for m in cluster:
                        if span_to_cluster_id.get((m.span.start_char, m.span.end_char)) == mav_cluster_id:
                            related_clusters_indices.append(idx)
                            break
                
                if related_clusters_indices:
                    master_idx = related_clusters_indices[0]
                    # Merge clusters in reverse order to maintain indices
                    for idx in sorted(related_clusters_indices[1:], reverse=True):
                        clusters[master_idx].extend(clusters.pop(idx))
                        # Update head lemma cache
                        for m in clusters[master_idx]:
                            if not m.is_pronoun:
                                head_lemma_to_cluster[m.head_lemma] = master_idx
                    
                    clusters[master_idx].append(mention)
                    if not mention.is_pronoun:
                        head_lemma_to_cluster[mention.head_lemma] = master_idx
                elif found_cluster is not None:
                    clusters[found_cluster].append(mention)
                    if not mention.is_pronoun:
                        head_lemma_to_cluster[mention.head_lemma] = found_cluster
                else:
                    new_idx = len(clusters)
                    clusters.append([mention])
                    if not mention.is_pronoun:
                        head_lemma_to_cluster[mention.head_lemma] = new_idx
            elif found_cluster is not None:
                clusters[found_cluster].append(mention)
            else:
                new_idx = len(clusters)
                clusters.append([mention])
                if not mention.is_pronoun:
                    head_lemma_to_cluster[mention.head_lemma] = new_idx

        # Optimized semantic merging with early termination
        merged = True
        while merged:
            merged = False
            # Cache representatives to avoid repeated calculations
            representatives = []
            for cluster in clusters:
                rep = next((m.span for m in cluster if not m.is_pronoun), cluster[0].span)
                representatives.append(rep)
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    if i >= len(clusters) or j >= len(clusters): 
                        continue
                    
                    rep1, rep2 = representatives[i], representatives[j]
                    
                    if (rep1.has_vector and rep2.has_vector and 
                        rep1.similarity(rep2) > similarity_threshold):
                        logger.info(f"Semantic Merge: Merging '{rep2.text}' into '{rep1.text}'")
                        clusters[i].extend(clusters.pop(j))
                        representatives.pop(j)  # Update representatives cache
                        merged = True
                        break
                if merged: 
                    break
                        
        final_output = {}
        for cluster in clusters:
            non_pronouns = [m for m in cluster if not m.is_pronoun]
            if not non_pronouns: continue
            
            canonical_name = max(non_pronouns, key=lambda m: len(m.text)).text
            
            mentions_text = {m.text for m in cluster if m.text.lower() not in ["itself"]}
            sorted_mentions = sorted(list(mentions_text), key=lambda m: text.find(m))
            
            if sorted_mentions:
                final_output[canonical_name] = {"mentions": sorted_mentions}
        
        # Apply category mapping to consolidate similar entities
        mapped_output = self._apply_category_mapping(final_output)
        return mapped_output

if __name__ == "__main__":
    # Test both systems
    restaurant_text = (
        "The food at Guido's was exceptional, but the service was a letdown. "
        "We ordered the pepperoni pizza, which was delicious. It had a perfectly crispy crust. "
        "However, our server seemed overwhelmed. The wait staff needs more training. "
        "The ambiance of the restaurant itself is quite nice, though."
    )

    print("\n--- Testing TransformerNERCorefExtractor (ATE + Maverick) ---")
    transformer_extractor = TransformerNERCorefExtractor()
    transformer_result = transformer_extractor.analyze(restaurant_text)
    print(json.dumps(transformer_result, indent=2))

    print("\n--- Testing EntityConsolidator (Legacy spaCy + Maverick) ---")
    legacy_extractor = EntityConsolidator()
    legacy_result = legacy_extractor.analyze(restaurant_text)
    print(json.dumps(legacy_result, indent=2))