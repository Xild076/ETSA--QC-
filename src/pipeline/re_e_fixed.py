import os
import json
import re
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from functools import lru_cache
import logging
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.WARNING)

try:
    import spacy
except Exception:
    spacy = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

_GLOBAL_GENAI_LIMITER = None

class _RateLimiter:
    def __init__(self, max_calls: int = 45, period_seconds: int = 60):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.timestamps: List[float] = []

    def acquire(self):
        now = time.time()
        self.timestamps = [t for t in self.timestamps if now - t < self.period_seconds]
        if len(self.timestamps) >= self.max_calls:
            sleep_time = self.period_seconds - (now - self.timestamps[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        self.timestamps.append(time.time())

def _ensure_global_rate_limiter(max_calls: int = 45, period_seconds: int = 60):
    global _GLOBAL_GENAI_LIMITER
    if _GLOBAL_GENAI_LIMITER is None:
        _GLOBAL_GENAI_LIMITER = _RateLimiter(max_calls=max_calls, period_seconds=period_seconds)
    return _GLOBAL_GENAI_LIMITER

def _default_rpm_for_model(model_name: str) -> int:
    name = (model_name or "").lower()
    if "gemma" in name:
        return 30
    if "2.5" in name and "pro" in name:
        return 5
    if "2.5" in name and "flash" in name:
        return 10
    if "2.0" in name and "flash" in name:
        return 15
    return 10

def _parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None

@dataclass
class GemmaConfig:
    model: str = "gemma-3-27b-it"
    temperature: float = 0.0
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int = 1024
    response_mime_type: str = "text/plain"

class RelationExtractor:
    def extract(self, text: str, entities: List[str]) -> Dict[str, Any]:
        raise NotImplementedError("This method should be overridden by subclasses.")

class DummyRelationExtractor(RelationExtractor):
    def extract(self, text: str, entities: List[str]) -> Dict[str, Any]:
        return {"approach_used": "dummy_extractor", "relations": [], "justification": ""}

class SpacyRelationExtractor(RelationExtractor):
    def __init__(self):
        self.nlp = self._load_spacy_model()
        
        self.sentiment_patterns = {
            'positive': ['good', 'great', 'excellent', 'amazing', 'fantastic', 'perfect', 'love', 'like', 'enjoy', 'delicious', 'fresh', 'fast', 'friendly', 'wonderful', 'outstanding', 'superb', 'terrific'],
            'negative': ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'slow', 'rude', 'cold', 'stale', 'expensive', 'disappointing', 'disgusting', 'poor', 'worst', 'mediocre'],
            'neutral': ['okay', 'fine', 'average', 'normal', 'typical', 'standard']
        }
        
        self.relation_indicators = {
            'direct_opinion': ['is', 'was', 'seems', 'looks', 'feels', 'tastes', 'sounds', 'appears', 'becomes'],
            'comparison': ['better', 'worse', 'than', 'compared', 'versus', 'more', 'less'],
            'causation': ['because', 'since', 'due to', 'leads to', 'results in', 'causes'],
            'temporal': ['after', 'before', 'during', 'while', 'when', 'then']
        }
    
    def _load_spacy_model(self):
        if spacy is None:
            raise RuntimeError("spaCy not available")
        try:
            return spacy.load("en_core_web_lg")
        except OSError:
            raise RuntimeError("spaCy model en_core_web_lg not found")
    
    def _get_dependency_path(self, token1, token2):
        path = []
        current = token1
        while current != token2 and current.head != current:
            path.append((current.text, current.dep_))
            current = current.head
            if len(path) > 10:
                break
        return path
    
    def _detect_sentiment_relation(self, entity_token, sentiment_token):
        path = self._get_dependency_path(entity_token, sentiment_token)
        
        if len(path) <= 2:
            return "direct_opinion"
        elif any(rel in [step[1] for step in path] for rel in ["nsubj", "dobj", "pobj"]):
            return "action"
        else:
            return "association"
    
    def _find_entity_tokens(self, doc, entities):
        entity_tokens = {}
        for entity in entities:
            tokens = []
            entity_lower = entity.lower()
            for token in doc:
                if entity_lower in token.text.lower() or token.text.lower() in entity_lower:
                    tokens.append(token)
            entity_tokens[entity] = tokens
        return entity_tokens
    
    def _extract_sentiment_relations(self, doc, entity_tokens):
        relations = []
        
        for entity, tokens in entity_tokens.items():
            for token in doc:
                if any(pattern in token.text.lower() for pattern_list in self.sentiment_patterns.values() for pattern in pattern_list):
                    
                    for entity_token in tokens:
                        rel_type = self._detect_sentiment_relation(entity_token, token)
                        
                        context_start = max(0, min(entity_token.i, token.i) - 2)
                        context_end = min(len(doc), max(entity_token.i, token.i) + 3)
                        context = " ".join([t.text for t in doc[context_start:context_end]])
                        
                        negated = any(child.dep_ == "neg" for child in token.children) or \
                                 any(t.text.lower() in ["not", "never", "no"] for t in doc[max(0, token.i-2):token.i])
                        
                        relations.append({
                            "entity": entity,
                            "sentiment_word": token.text,
                            "relation_type": rel_type,
                            "context": context,
                            "negated": negated,
                            "confidence": self._calculate_relation_confidence(entity_token, token, rel_type)
                        })
        
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                subj = None
                obj = None
                
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subj = child.text
                    elif child.dep_ in ["dobj", "attr", "acomp"]:
                        obj = child.text
                
                if subj and obj:
                    for entity in entity_tokens.keys():
                        if entity.lower() in subj.lower():
                            relations.append({
                                "entity": entity,
                                "sentiment_word": obj,
                                "relation_type": "direct_opinion",
                                "context": f"{subj} {token.text} {obj}",
                                "negated": any(child.dep_ == "neg" for child in token.children),
                                "confidence": 0.8
                            })
        
        return relations
    
    def _calculate_relation_confidence(self, entity_token, sentiment_token, rel_type):
        base_confidence = 0.7
        
        distance = abs(entity_token.i - sentiment_token.i)
        if distance <= 2:
            base_confidence += 0.2
        elif distance <= 5:
            base_confidence += 0.1
        
        if rel_type == "direct_opinion":
            base_confidence += 0.1
        
        if entity_token.dep_ in ["nsubj", "dobj"] and sentiment_token.pos_ in ["ADJ", "VERB"]:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)

    def extract(self, text: str, entities: List[str]) -> Dict[str, Any]:
        doc = self.nlp(text)
        entity_tokens = self._find_entity_tokens(doc, entities)
        sentiment_relations = self._extract_sentiment_relations(doc, entity_tokens)
        
        actions = []
        associations = []
        belongings = []
        
        for relation in sentiment_relations:
            if relation["confidence"] > 0.6:
                if relation["relation_type"] == "action":
                    actions.append({
                        "actor": relation["entity"],
                        "action": relation["sentiment_word"],
                        "target": "",
                        "phrase": relation["context"],
                        "negated": relation["negated"]
                    })
                elif relation["relation_type"] == "association":
                    associations.append({
                        "entity1": relation["entity"],
                        "entity2": relation["sentiment_word"],
                        "phrase": relation["context"],
                        "negated": relation["negated"]
                    })
                elif relation["relation_type"] == "direct_opinion":
                    actions.append({
                        "actor": relation["entity"],
                        "action": "has_quality",
                        "target": relation["sentiment_word"],
                        "phrase": relation["context"],
                        "negated": relation["negated"]
                    })
        
        relations = []
        for action in actions:
            relations.append({
                "subject": {"head": action["actor"]},
                "object": {"head": action["target"]},
                "relation": {"type": "action", "text": action["phrase"]}
            })
        
        for assoc in associations:
            relations.append({
                "subject": {"head": assoc["entity1"]},
                "object": {"head": assoc["entity2"]},
                "relation": {"type": "association", "text": assoc["phrase"]}
            })
        
        return {
            "entities": entities,
            "actions": actions,
            "associations": associations,
            "belongings": belongings,
            "relations": relations,
            "approach_used": "spacy_enhanced",
            "justification": f"Found {len(actions)} actions, {len(associations)} associations, {len(relations)} total relations via enhanced dependency analysis"
        }

class GemmaRelationExtractor(RelationExtractor):
    def __init__(self, api_key: str = None, config: GemmaConfig = None, rate_limiter: _RateLimiter = None):
        self.config = config or GemmaConfig()
        self.rate_limiter = rate_limiter or _ensure_global_rate_limiter(
            max_calls=_default_rpm_for_model(self.config.model)
        )
        self._init_api(api_key)

    def _init_api(self, api_key: str = None):
        if genai is None:
            raise RuntimeError("google-generativeai not available")
        
        if api_key:
            genai.configure(api_key=api_key)
        elif "GOOGLE_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        else:
            raise RuntimeError("No API key provided")

    def _create_prompt(self, sentence: str, entities: List[str]) -> str:
        entities_str = ", ".join(f'"{ent}"' for ent in entities)
        return f"""You are a precise relation extraction engine. Analyze the sentence to identify relationships between entities.

Sentence: "{sentence}"
Entities: {entities_str}

Extract relationships following these rules:
1. ACTION: When an entity performs an action on another entity or has a quality/state
2. ASSOCIATION: When entities are connected by proximity, coordination, or explicit mention
3. BELONGING: When one entity owns or contains another (possessive relationships)

For each relationship found:
- Identify the subject entity (from the entities list)
- Identify the object entity (can be from entities list or inferred)
- Classify the relationship type (action, association, belonging)
- Extract the connecting phrase

Output in JSON format:
{{
  "relations": [
    {{"subject": {{"head": "entity1"}}, "object": {{"head": "entity2"}}, "relation": {{"type": "action", "text": "connecting phrase"}}}},
  ],
  "actions": [
    {{"actor": "entity", "action": "verb", "target": "target_entity", "phrase": "full phrase", "negated": false}}
  ],
  "associations": [
    {{"entity1": "entity1", "entity2": "entity2", "phrase": "connecting phrase", "negated": false}}
  ],
  "belongings": [
    {{"parent": "entity1", "child": "entity2", "phrase": "possessive phrase", "negated": false}}
  ],
  "justification": "explanation of extracted relationships"
}}"""

    def _query_gemma(self, prompt: str) -> str:
        if self.rate_limiter:
            self.rate_limiter.acquire()
        
        model = genai.GenerativeModel(
            model_name=self.config.model,
            generation_config=genai.types.GenerationConfig(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                max_output_tokens=self.config.max_output_tokens,
                response_mime_type=self.config.response_mime_type,
            )
        )
        
        response = model.generate_content(prompt)
        return response.text

    def _parse_relation_response(self, text: str, entities: List[str]) -> Dict[str, Any]:
        parsed = _parse_json_from_text(text)
        if not parsed:
            return {
                "entities": entities,
                "actions": [],
                "associations": [],
                "belongings": [],
                "relations": [],
                "justification": "Failed to parse JSON response",
                "approach_used": "gemma-3-27b-it"
            }
        
        return {
            "entities": entities,
            "actions": parsed.get("actions", []),
            "associations": parsed.get("associations", []),
            "belongings": parsed.get("belongings", []),
            "relations": parsed.get("relations", []),
            "justification": parsed.get("justification", ""),
            "approach_used": "gemma-3-27b-it"
        }

    def extract(self, text: str, entities: List[str]) -> Dict[str, Any]:
        if not text or not entities:
            return {
                "entities": entities,
                "actions": [],
                "associations": [],
                "belongings": [],
                "relations": [],
                "justification": "No input provided",
                "approach_used": "gemma-3-27b-it"
            }
        
        prompt = self._create_prompt(text, entities)
        try:
            response = self._query_gemma(prompt)
            return self._parse_relation_response(response, entities)
        except Exception as e:
            return {
                "entities": entities,
                "actions": [],
                "associations": [],
                "belongings": [],
                "relations": [],
                "justification": f"Error during extraction: {str(e)}",
                "approach_used": "gemma-3-27b-it"
            }
