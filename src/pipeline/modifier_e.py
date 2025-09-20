import os
import json
import re
from typing import Dict, Any, List
from functools import lru_cache
from .re_e import GemmaRelationExtractor, _RateLimiter, _GLOBAL_GENAI_LIMITER
import logging
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.WARNING)

try:
    import spacy
except Exception:
    spacy = None

@lru_cache(maxsize=1)
def _get_spacy_nlp():
    if not spacy:
        return None
    for name in ("en_core_web_sm", "en_core_web_lg"):
        try:
            return spacy.load(name)
        except Exception:
            continue
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return None

class ModifierExtractor:
    def extract(self, text: str, entity: str) -> Dict[str, Any]:
        raise NotImplementedError("This method should be overridden by subclasses.")

class DummyModifierExtractor(ModifierExtractor):
    def extract(self, text: str, entity: str) -> Dict[str, Any]:
        return {
            "entity": entity,
            "modifiers": [],
            "justification": "Dummy extractor: no modifiers extracted.",
            "approach_used": "dummy_extractor"
        }

class GemmaModifierExtractor(ModifierExtractor):
    def __init__(self, api_key: str = None, rate_limiter: _RateLimiter = None):
        self._gemma_handler = GemmaRelationExtractor(api_key, rate_limiter)

    def _create_prompt(self, sentence: str, entity: str) -> str:
        return f"""You are a precise information extraction engine. First, analyze the sentence structure and identify the target entity.

Text: "{sentence}"
Entity: "{entity}"

Step 1: Locate the entity in the text and understand its role.
Step 2: Identify all adjectival and adverbial modifiers that directly describe this entity's QUALITIES.
Step 3: EXCLUDE factual context, parenthetical information, and location references.
Step 4: Include negation only when it directly negates a quality (not when it's factual info).
Step 5: Extract modifiers as exact text spans - do not paraphrase.

CRITICAL RULES:
- Parenthetical information like "(not on menu)" = EXCLUDE (factual context)
- "Try X" commands = EXCLUDE any negation modifiers (recommendation context)
- "not the best" = include as quality modifier for the actual entity being rated
- Location/availability info = EXCLUDE (e.g., "in New York", "on menu")
- Only include quality-describing words that affect sentiment

Examples:
1. "The staff should be more friendly" with entity "staff"
   → modifiers: ["should be more", "friendly"]

2. "BEST spicy tuna roll" with entity "roll"  
   → modifiers: ["BEST", "spicy", "tuna"]

3. "The ambiance was not welcoming" with entity "ambiance"
   → modifiers: ["not welcoming"]

4. "Try the rose roll (not on menu)" with entity "rose roll"
   → modifiers: [] (parenthetical is factual, not quality)

5. "Certainly not the best sushi in New York" with entity "sushi"
   → modifiers: ["Certainly not the best"] (quality assessment, exclude location)

6. "The service was disappointing and slow" with entity "service"
   → modifiers: ["disappointing", "slow"]

7. "Great asian salad" with entity "salad"
   → modifiers: ["Great", "asian"]

Rules:
1. Extract ONLY quality-describing modifiers that affect sentiment
2. EXCLUDE factual context, location info, parenthetical remarks
3. Include negation words only when they negate qualities, not facts
4. Be conservative - when in doubt, exclude rather than include
5. If the entity is not being qualitatively described, return empty modifiers list

Provide your output in STRICT JSON format:
{{"justification": "reasoning", "approach_used": "gemma-3-27b-it", "entity": "{entity}", "modifiers": ["modifier1", "modifier2"]}}
"""

    def _parse_modifier_response(self, text: str, entity: str) -> Dict[str, Any]:
        try:
            data = json.loads(re.search(r"\{[\s\S]*\}", text).group(0))
        except Exception:
            data = {}
        mods = []
        entity_clean = entity.lower().strip()
        
        # Define sentiment-bearing phrases that should be preserved even if they overlap
        sentiment_phrases = [
            'not the best', 'not good', 'not great', 'not bad', 'very good', 'very bad',
            'top notch', 'high quality', 'poor quality', 'excellent', 'terrible',
            'outstanding', 'disappointing', 'amazing', 'awful', 'fantastic', 'horrible',
            'best', 'worst', 'great', 'bad', 'good', 'poor', 'fresh', 'stale'
        ]
        
        for m in data.get("modifiers", []) or []:
            mm = (m or "").strip()
            if mm:
                modifier_clean = mm.lower().strip()
                
                # Check if this is a sentiment-bearing phrase that should be preserved
                is_sentiment_phrase = any(phrase in modifier_clean for phrase in sentiment_phrases)
                
                if is_sentiment_phrase:
                    # Always include sentiment-bearing phrases
                    mods.append(mm)
                    continue
                
                # For non-sentiment phrases, apply stricter overlap detection
                # 1. Check if modifier is identical to entity
                if modifier_clean == entity_clean:
                    continue
                    
                # 2. Check if modifier is just a substring of entity without adding value
                if modifier_clean in entity_clean and len(modifier_clean) < len(entity_clean) * 0.8:
                    continue
                    
                # 3. Check if entity is a substring of modifier (modifier is longer version)
                if entity_clean in modifier_clean and len(entity_clean) < len(modifier_clean) * 0.8:
                    continue
                
                mods.append(mm)
        return {
            "entity": entity,
            "modifiers": mods,
            "justification": data.get("justification", ""),
            "approach_used": data.get("approach_used", "gemma-3-27b-it")
        }

    def extract(self, text: str, entity: str) -> Dict[str, Any]:
        if not text or not entity:
            return {"entity": entity, "modifiers": [], "justification": "No input provided."}
        prompt = self._create_prompt(text, entity)
        try:
            response = self._gemma_handler._query_gemma(prompt)
            return self._parse_modifier_response(response, entity)
        except Exception as e:
            raise

class SpacyModifierExtractor(ModifierExtractor):
    def extract(self, text: str, entity: str) -> Dict[str, Any]:
        nlp = _get_spacy_nlp()
        if not nlp or not text or not entity:
            return {"entity": entity, "modifiers": [], "approach_used": "spacy_enhanced", "justification": "no spacy or empty input"}
        
        doc = nlp(text)
        ent_lower = entity.lower()
        mods = []
        negations = {"not", "n't", "no", "never", "hardly", "scarcely", "barely", "rarely"}

        def get_enhanced_modifier(token) -> str:
            """Get modifier with context, avoiding redundant entity words."""
            prefix_parts = []
            
            # Add meaningful negations and intensifiers
            for child in token.children:
                if child.dep_ == "neg":
                    prefix_parts.append(child.text)
                elif child.dep_ == "advmod" and child.text.lower() not in {"as", "the", "a", "an", "that", "this"}:
                    prefix_parts.append(child.text)
            
            # Look for nearby negations
            window_start = max(0, token.i - 3)
            window_end = min(len(doc), token.i)
            window_tokens = [t.text.lower() for t in doc[window_start:window_end]]
            
            for neg in negations:
                if neg in window_tokens and neg not in [p.lower() for p in prefix_parts]:
                    prefix_parts.insert(0, neg)
                    break
            
            # Clean result
            if prefix_parts:
                result = " ".join(prefix_parts) + " " + token.text
            else:
                result = token.text
            
            # Remove filler words
            result = re.sub(r'\b(as|the|a|an|that|this)\b', '', result, flags=re.IGNORECASE)
            result = re.sub(r'\s+', ' ', result).strip()
            
            return result if result else token.text

        def find_entity_token(doc, entity_text):
            entity_tokens = []
            ent_words = entity_text.lower().split()
            
            # First try to find exact multiword spans
            entity_text_clean = entity_text.lower().strip()
            doc_text_lower = doc.text.lower()
            
            if entity_text_clean in doc_text_lower:
                start_char = doc_text_lower.find(entity_text_clean)
                end_char = start_char + len(entity_text_clean)
                
                for token in doc:
                    if (token.idx >= start_char and 
                        token.idx + len(token.text) <= end_char):
                        entity_tokens.append(token)
            
            # Fallback to individual word matching, but only include substantive words
            if not entity_tokens:
                substantive_words = [word for word in ent_words if word not in {"the", "a", "an", "this", "that"}]
                for i, token in enumerate(doc):
                    if token.text.lower() in substantive_words:
                        entity_tokens.append(token)
            
            return entity_tokens

        entity_tokens = find_entity_token(doc, entity)
        
        for token in doc:
            if token.dep_ in ("amod", "advmod") and token.head in entity_tokens:
                mods.append(get_enhanced_modifier(token))
            
            elif token.dep_ in ("acomp", "attr") and token.head.pos_ in ("VERB", "AUX"):
                for child in token.head.children:
                    if child.dep_ in ("nsubj", "nsubjpass") and child in entity_tokens:
                        mods.append(get_enhanced_modifier(token))
                        break
            
            elif token.pos_ == "VERB":
                subj = next((c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")), None)
                obj = next((c for c in token.children if c.dep_ in ("dobj", "pobj")), None)
                
                # When entity is object of verb
                if obj and obj in entity_tokens:
                    mods.append(get_enhanced_modifier(token))
                
                # When entity is subject of verb - check for semantically meaningful verbs
                elif subj and subj in entity_tokens:
                    acomp = next((c for c in token.children if c.dep_ == "acomp"), None)
                    if acomp:
                        mods.append(get_enhanced_modifier(acomp))
                    
                    # Handle semantically meaningful verbs when entity is subject
                    verb_lemma = token.lemma_.lower()
                    negative_verbs = {
                        'slow', 'delay', 'hinder', 'impede', 'block', 'prevent', 'stop', 'break', 'fail',
                        'crash', 'freeze', 'lag', 'stutter', 'drop', 'decrease', 'reduce', 'worsen',
                        'deteriorate', 'degrade', 'corrupt', 'damage', 'harm', 'hurt', 'annoy', 'frustrate',
                        'disappoint', 'bore', 'tire', 'exhaust', 'overwhelm', 'confuse', 'complicate'
                    }
                    positive_verbs = {
                        'accelerate', 'speed', 'enhance', 'improve', 'boost', 'increase', 'strengthen',
                        'optimize', 'streamline', 'facilitate', 'help', 'assist', 'enable', 'empower',
                        'delight', 'please', 'satisfy', 'impress', 'amaze', 'excel', 'succeed', 'shine',
                        'perform', 'deliver', 'work', 'function', 'operate', 'run', 'execute'
                    }
                    
                    if verb_lemma in negative_verbs or verb_lemma in positive_verbs:
                        # Build full verb phrase including particles and adverbs
                        verb_parts = [token.text]
                        
                        # Add particles (like "down" in "slows down")
                        for child in token.children:
                            if child.dep_ == "prt":
                                verb_parts.append(child.text)
                            elif child.dep_ == "advmod" and child.text.lower() not in {"as", "the", "a", "an", "that", "this"}:
                                verb_parts.append(child.text)
                        
                        verb_phrase = " ".join(verb_parts)
                        mods.append(verb_phrase)
            
            elif token.dep_ == "compound" and token.head in entity_tokens:
                mods.append(get_enhanced_modifier(token))

        seen = set()
        unique_mods = []
        entity_clean = entity.lower().strip()
        
        # Define sentiment-bearing phrases that should be preserved even if they overlap
        sentiment_phrases = [
            'not the best', 'not good', 'not great', 'not bad', 'very good', 'very bad',
            'top notch', 'high quality', 'poor quality', 'excellent', 'terrible',
            'outstanding', 'disappointing', 'amazing', 'awful', 'fantastic', 'horrible',
            'best', 'worst', 'great', 'bad', 'good', 'poor', 'fresh', 'stale'
        ]
        
        for m in mods:
            if m.lower() not in seen:
                modifier_clean = m.lower().strip()
                
                # Check if this is a sentiment-bearing phrase that should be preserved
                is_sentiment_phrase = any(phrase in modifier_clean for phrase in sentiment_phrases)
                
                if is_sentiment_phrase:
                    # Always include sentiment-bearing phrases
                    seen.add(m.lower())
                    unique_mods.append(m)
                    continue
                
                # For non-sentiment phrases, apply stricter overlap detection
                # 1. Check if modifier is identical to entity
                if modifier_clean == entity_clean:
                    continue
                    
                # 2. Check if modifier is just a substring of entity without adding value
                if modifier_clean in entity_clean and len(modifier_clean) < len(entity_clean) * 0.8:
                    continue
                    
                # 3. Check if entity is a substring of modifier (modifier is longer version)
                if entity_clean in modifier_clean and len(entity_clean) < len(modifier_clean) * 0.8:
                    continue
                
                seen.add(m.lower())
                unique_mods.append(m)
        
        return {
            "entity": entity,
            "modifiers": unique_mods,
            "approach_used": "spacy_enhanced",
            "justification": f"Found {len(unique_mods)} modifiers via enhanced syntactic analysis"
        }

class RuleBasedModifierExtractor(ModifierExtractor):
    def extract(self, text: str, entity: str) -> Dict[str, Any]:
        nlp = _get_spacy_nlp()
        if not nlp or not text or not entity:
            return {"entity": entity, "modifiers": [], "approach_used": "rule_based", "justification": "no spacy"}
        doc = nlp(text)
        ent_lower = (entity or "").lower()
        mods: List[str] = []
        negations = {"not", "n't", "no", "never", "hardly", "scarcely"}

        def get_full_modifier(token) -> str:
            prefix = ""
            for child in token.children:
                if child.dep_ == "neg":
                    prefix += child.text + " "
                    break
            window = [t.text.lower() for t in doc[max(0, token.i - 2) : token.i]]
            if any(neg in window for neg in negations):
                 if "not" not in prefix:
                    prefix += "not "

            for child in token.children:
                if child.dep_ == "advmod":
                    prefix += child.text + " "
            
            return prefix + token.lemma_

        for token in doc:
            if token.dep_ in ("amod", "advmod") and token.head.text.lower() in ent_lower:
                mods.append(get_full_modifier(token))

            elif token.dep_ in ("acomp", "attr") and token.head.pos_ == "VERB":
                for child in token.head.children:
                    if child.dep_ in ("nsubj", "nsubjpass") and child.text.lower() in ent_lower:
                        mods.append(get_full_modifier(token))
                        break
            
            elif token.pos_ == "VERB":
                subj = next((c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")), None)
                obj = next((c for c in token.children if c.dep_ in ("dobj", "pobj")), None)
                
                if obj and obj.text.lower() in ent_lower:
                    mods.append(get_full_modifier(token))
                
                elif subj and subj.text.lower() in ent_lower:
                    acomp = next((c for c in token.children if c.dep_ == "acomp"), None)
                    if acomp:
                        mods.append(get_full_modifier(acomp))
        
        seen = set()
        uniq = []
        entity_clean = entity.lower().strip()
        
        # Define sentiment-bearing phrases that should be preserved even if they overlap
        sentiment_phrases = [
            'not the best', 'not good', 'not great', 'not bad', 'very good', 'very bad',
            'top notch', 'high quality', 'poor quality', 'excellent', 'terrible',
            'outstanding', 'disappointing', 'amazing', 'awful', 'fantastic', 'horrible',
            'best', 'worst', 'great', 'bad', 'good', 'poor', 'fresh', 'stale'
        ]
        
        for m in mods:
            if m not in seen:
                modifier_clean = m.lower().strip()
                
                # Check if this is a sentiment-bearing phrase that should be preserved
                is_sentiment_phrase = any(phrase in modifier_clean for phrase in sentiment_phrases)
                
                if is_sentiment_phrase:
                    # Always include sentiment-bearing phrases
                    seen.add(m)
                    uniq.append(m)
                    continue
                
                # For non-sentiment phrases, apply stricter overlap detection
                # 1. Check if modifier is identical to entity
                if modifier_clean == entity_clean:
                    continue
                    
                # 2. Check if modifier is just a substring of entity without adding value
                if modifier_clean in entity_clean and len(modifier_clean) < len(entity_clean) * 0.8:
                    continue
                    
                # 3. Check if entity is a substring of modifier (modifier is longer version)
                if entity_clean in modifier_clean and len(entity_clean) < len(modifier_clean) * 0.8:
                    continue
                
                seen.add(m)
                uniq.append(m)
        
        return {
            "entity": entity,
            "modifiers": uniq,
            "approach_used": "rule_based",
            "justification": f"Found {len(uniq)} modifiers via syntactic analysis."
        }

class CombinedModifierExtractor(ModifierExtractor):
    def __init__(self, api_key: str = None):
        self._gemma_extractor = GemmaModifierExtractor(api_key=api_key, rate_limiter=_GLOBAL_GENAI_LIMITER)
        self._rule_extractor = RuleBasedModifierExtractor()

    def extract(self, text: str, entity: str) -> Dict[str, Any]:
        try:
            gemma_result = self._gemma_extractor.extract(text, entity)
            if gemma_result and gemma_result.get("modifiers"):
                return gemma_result
        except Exception as e:
            logging.warning(f"Gemma extractor failed for entity '{entity}': {e}. Falling back to rule-based.")
        
        return self._rule_extractor.extract(text, entity)