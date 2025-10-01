import json
import re
import time
from typing import Dict, Any, List, Optional
from functools import lru_cache
from copy import deepcopy
import logging
import os

import config
from utility import get_env, ensure_env_loaded
from cache_utils import load_cache_from_file, save_cache_to_file

ensure_env_loaded()

try:
    import spacy
except Exception:
    spacy = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.WARNING)

class _RateLimiter:
    def __init__(self, max_calls: int = 20, period_seconds: int = 60):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.timestamps: List[float] = []

    def acquire(self) -> None:
        now = time.time()
        self.timestamps = [t for t in self.timestamps if now - t < self.period_seconds]
        if len(self.timestamps) >= self.max_calls:
            sleep_time = self.period_seconds - (now - self.timestamps[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        self.timestamps.append(time.time())


_GLOBAL_MODIFIER_LIMITER: Optional[_RateLimiter] = None


def _ensure_modifier_rate_limiter(max_calls: int = 20, period_seconds: int = 60) -> _RateLimiter:
    global _GLOBAL_MODIFIER_LIMITER
    if (
        _GLOBAL_MODIFIER_LIMITER is None
        or _GLOBAL_MODIFIER_LIMITER.max_calls != max_calls
        or _GLOBAL_MODIFIER_LIMITER.period_seconds != period_seconds
    ):
        _GLOBAL_MODIFIER_LIMITER = _RateLimiter(max_calls=max_calls, period_seconds=period_seconds)
    return _GLOBAL_MODIFIER_LIMITER

class ModifierExtractor:
    def extract(self, text: str, entity: str) -> Dict[str, Any]:
        raise NotImplementedError

class DummyModifierExtractor(ModifierExtractor):
    def extract(self, text: str, entity: str) -> Dict[str, Any]:
        return {"entity": entity, "modifiers": [], "approach_used": "dummy", "justification": "empty"}

@lru_cache(maxsize=1)
def _get_spacy_nlp():
    if not spacy:
        return None
    for name in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
        try:
            return spacy.load(name)
        except Exception:
            continue
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return None

def _parse_json_payload(txt: str) -> Optional[Dict[str, Any]]:
    if not txt:
        return None
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", txt)
    s = m.group(1) if m else None
    if not s:
        m2 = re.search(r"\{[\s\S]*\}", txt)
        s = m2.group(0) if m2 else None
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        try:
            s2 = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", s)
            return json.loads(s2)
        except Exception:
            return None

class GemmaModifierExtractor(ModifierExtractor):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "models/gemma-3-27b-it",
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 40,
        max_output_tokens: int = 768,
        backoff: float = 0.8,
        retries: int = 2,
        rate_limiter: Optional[_RateLimiter] = None,
        cache_file: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        self.backoff = backoff
        self.retries = retries
        self.rate_limiter = rate_limiter or _ensure_modifier_rate_limiter()
        self.cache_file = cache_file or config.LLM_MODIFIER_CACHE
        if self.cache_file:
            self._cache = load_cache_from_file(self.cache_file)
        else:
            self._cache: Dict[tuple[str, str], Dict[str, Any]] = {}
        if genai is None:
            raise RuntimeError("google-generativeai not available")
        ensure_env_loaded()
        key = api_key or get_env("GOOGLE_API_KEY") or get_env("GENAI_API_KEY") or get_env("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("No API key provided for GemmaModifierExtractor. Check .env file for GOOGLE_API_KEY")
        try:
            genai.configure(api_key=key)
            logger.info(f"GemmaModifierExtractor initialized with model: {self.model}")
        except Exception as e:
            raise RuntimeError(f"Failed to configure Gemini API: {e}")

    def _prompt(self, passage: str, entity: str) -> str:
        return f"""<<SYS>>
You are an expert linguist performing aspect-based sentiment analysis. Your task is to extract all **complete evaluative predicates** that describe a specific ENTITY within a PASSAGE. An evaluative predicate is the full verb phrase or adjectival phrase expressing an opinion, including negation, adverbs, and objects.

## Directives
1.  **Extract Full Predicates**: Do not extract single adjectives. Capture the entire evaluative phrase.
    -   **Correct**: "is very quiet", "lasts a long time", "is not worth the price", "works better than USB3"
    -   **Incorrect**: "quiet", "long", "not worth"
2.  **Verbatim Extraction**: All extracted modifiers must be exact, verbatim substrings from the PASSAGE.
3.  **Include Full Context**: Capture all words necessary to understand the sentiment, especially:
    -   **Negation**: "doesn't have", "is not", "can't", "won't", "never"
    -   **Modals**: "could be better", "should work", "might fail"
    -   **Comparisons**: "works better than", "worse than expected", "as good as"
    -   **Intensifiers**: "very quiet", "extremely fast", "quite good", "somewhat disappointing"
    -   **Qualifying phrases**: "worth the price", "good enough for", "better than expected"
4.  **Handle Multiple & Contrasting Opinions**: If an entity has multiple descriptions, extract all of them. For "The screen is bright but has glare", if ENTITY is "screen", extract both "is bright" and "has glare".
5.  **Capture Contextual Sentiment**: Include phrases that provide context for evaluation:
    -   **Value framing**: "expensive but worth it", "cheap and reliable"
    -   **Use case specific**: "good for gaming", "not suitable for work"
    -   **Temporal aspects**: "works well initially", "degrades over time"
6.  **Output Format**: Return a single, valid JSON object with no markdown fences.

## Schema
{{
  "entity": "{entity}",
  "justification": "Brief note on the extraction logic and patterns found.",
  "modifiers": ["string"],
  "approach_used": "{self.model}"
}}
<</SYS>>

## Examples

### Example 1
<<PASSAGE>>
The food was amazing, but the service was trash.
</PASSAGE>>
<<ENTITY>>
service
</ENTITY>>
<<RESPONSE>>
{{
  "entity": "service",
  "justification": "Extracted the full adjectival predicate 'was trash'.",
  "modifiers": ["was trash"],
  "approach_used": "{self.model}"
}}

### Example 2
<<PASSAGE>>
The battery life is good but the screen is not very bright.
</PASSAGE>>
<<ENTITY>>
screen
</ENTITY>>
<<RESPONSE>>
{{
  "entity": "screen",
  "justification": "Extracted the complete negated predicate 'is not very bright'.",
  "modifiers": ["is not very bright"],
  "approach_used": "{self.model}"
}}

### Example 3
<<PASSAGE>>
I love the keyboard, although I do wish it had a backlight.
</PASSAGE>>
<<ENTITY>>
keyboard
</ENTITY>>
<<RESPONSE>>
{{
  "entity": "keyboard",
  "justification": "Extracted the verb phrase 'love the keyboard' and the implicit negative desire 'wish it had a backlight'.",
  "modifiers": ["love the keyboard", "wish it had a backlight"],
  "approach_used": "{self.model}"
}}

## Task

<<PASSAGE>>
{passage}
</PASSAGE>>

<<ENTITY>>
{entity}
</ENTITY>>

<<RESPONSE>>
"""

    def _call(self, prompt: str) -> str:
        if self.rate_limiter:
            self.rate_limiter.acquire()
        config_kwargs: Dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
        }
        if "gemini" in (self.model or ""):
            config_kwargs["response_mime_type"] = "application/json"
        model = genai.GenerativeModel(
            model_name=self.model,
            generation_config=genai.types.GenerationConfig(**config_kwargs)
        )
        resp = model.generate_content(prompt)
        return resp.text

    def extract(self, text: str, entity: str) -> Dict[str, Any]:
        if not text or not entity:
            return {"entity": entity, "modifiers": [], "approach_used": self.model, "justification": "empty input"}
        cache_key = (text.strip(), entity.strip())
        cached = self._cache.get(cache_key)
        if cached:
            return cached
        p = self._prompt(text, entity)
        last_err = None
        for i in range(self.retries + 1):
            try:
                raw = self._call(p)
                data = _parse_json_payload(raw) or {}
                mods = data.get("modifiers", [])
                if not isinstance(mods, list):
                    mods = []
                norm = []
                seen = set()
                for m in mods:
                    if not isinstance(m, str):
                        continue
                    s = re.sub(r"\s+", " ", m.strip())
                    if not s:
                        continue
                    if s.lower() in seen:
                        continue
                    seen.add(s.lower())
                    norm.append(s)
                out = {
                    "entity": entity,
                    "modifiers": norm,
                    "approach_used": self.model,
                    "justification": data.get("justification", "")[:500]
                }
                self._cache[cache_key] = out
                if self.cache_file:
                    save_cache_to_file(self.cache_file, self._cache)
                return out
            except Exception as e:
                last_err = e
                if i < self.retries:
                    time.sleep(self.backoff * (i + 1))
        return {"entity": entity, "modifiers": [], "approach_used": self.model, "justification": f"failure: {last_err}"}

class SpacyModifierExtractor(ModifierExtractor):
    def _extract_helper(self, text: str, entity: str) -> Dict[str, Any]:
        nlp = _get_spacy_nlp()
        if not nlp or not text or not entity:
            return {"entity": entity, "modifiers": [], "approach_used": "spacy_enhanced", "justification": "unavailable"}
        doc = nlp(text)
        t = text.lower()
        e = entity.strip()
        if not e:
            return {"entity": entity, "modifiers": [], "approach_used": "spacy_enhanced", "justification": "empty"}
        
        # Find entity tokens more robustly
        entity_tokens = []
        entity_lower = e.lower()
        
        # Try exact phrase match first
        for i, token in enumerate(doc):
            if token.text.lower() == entity_lower:
                entity_tokens.append(token)
            elif entity_lower in token.text.lower():
                entity_tokens.append(token)
        
        # If no direct matches, try word-by-word matching
        if not entity_tokens:
            entity_words = {w for w in entity_lower.split() if w not in {"the","a","an","this","that"}}
            for token in doc:
                if token.text.lower() in entity_words:
                    entity_tokens.append(token)
        
        if not entity_tokens:
            return {"entity": entity, "modifiers": [], "approach_used": "spacy_enhanced", "justification": "entity not found"}
        
        # Find heads of entity tokens
        heads = set()
        for token in entity_tokens:
            head = token
            while head.head != head and head.head != head.head:
                head = head.head
            heads.add(head)
        
        mods = []
        
        # Extract comprehensive modifier patterns for each head
        for head in heads:
            # 1. Adjectival modifiers (enhanced)
            for child in head.children:
                if child.dep_ in ("amod", "acomp", "nmod", "compound"):
                    mod_phrase = self._extract_full_modifier_phrase(child)
                    if mod_phrase:
                        mods.append(mod_phrase)
            
            # 2. Adverbial modifiers  
            for child in head.children:
                if child.dep_ in ("advmod", "npadvmod"):
                    mod_phrase = self._extract_full_modifier_phrase(child)
                    if mod_phrase:
                        mods.append(mod_phrase)
            
            # 3. Verbal predicates (key for sentiment)
            if head.head != head:
                parent = head.head
                if parent.pos_ == "VERB":
                    # Extract full verb phrase with modifiers
                    verb_phrase = self._extract_verb_phrase(parent, head)
                    if verb_phrase:
                        mods.append(verb_phrase)
            
            # 4. Relative clauses and prepositional phrases
            for child in head.children:
                if child.dep_ in ("relcl", "acl", "prep", "pcomp"):
                    clause_phrase = self._extract_clause_phrase(child)
                    if clause_phrase:
                        mods.append(clause_phrase)
                        
            # 5. Negation and modal constructions  
            for child in head.children:
                if child.dep_ in ("neg", "aux", "auxpass"):
                    neg_phrase = self._extract_negation_phrase(child, head)
                    if neg_phrase:
                        mods.append(neg_phrase)
        
        # Filter and clean modifiers
        unique_mods = self._clean_and_filter_modifiers(mods, entity)
        return {"entity": entity, "modifiers": unique_mods, "approach_used": "spacy_enhanced", "justification": f"{len(unique_mods)} enhanced patterns"}
    
    def _extract_full_modifier_phrase(self, token) -> Optional[str]:
        """Extract full modifier phrase including intensifiers and qualifiers."""
        phrase_tokens = []
        
        # Include intensifiers (very, quite, extremely, etc.)
        for child in token.children:
            if child.dep_ == "advmod":
                phrase_tokens.append(child)
        
        # Add the main modifier
        phrase_tokens.append(token)
        
        # Include any dependent elements
        for child in token.children:
            if child.dep_ in ("prep", "pcomp", "dobj", "attr"):
                phrase_tokens.extend(list(child.subtree))
        
        phrase_tokens.sort(key=lambda t: t.i)
        phrase = " ".join([t.text for t in phrase_tokens]).strip()
        return phrase if len(phrase) > 1 else None
    
    def _extract_verb_phrase(self, verb_token, entity_token) -> Optional[str]:
        """Extract full verb phrase that relates to the entity."""
        phrase_tokens = []
        
        # Check if entity is subject or object of verb
        entity_is_subj = any(child.dep_ in ("nsubj", "nsubjpass") and child == entity_token for child in verb_token.children)
        entity_is_obj = any(child.dep_ in ("dobj", "pobj") and child == entity_token for child in verb_token.children)
        
        if not (entity_is_subj or entity_is_obj):
            return None
        
        # Include auxiliaries and negation
        for child in verb_token.children:
            if child.dep_ in ("aux", "auxpass", "neg"):
                phrase_tokens.append(child)
        
        # Add main verb
        phrase_tokens.append(verb_token)
        
        # Include adverbial modifiers
        for child in verb_token.children:
            if child.dep_ in ("advmod", "npadvmod"):
                phrase_tokens.extend(list(child.subtree))
        
        # Include complements and objects (but not the entity itself)
        for child in verb_token.children:
            if child.dep_ in ("dobj", "attr", "acomp") and child != entity_token:
                phrase_tokens.extend(list(child.subtree))
        
        phrase_tokens.sort(key=lambda t: t.i)
        phrase = " ".join([t.text for t in phrase_tokens]).strip()
        return phrase if len(phrase) > 1 else None
    
    def _extract_clause_phrase(self, token) -> Optional[str]:
        """Extract relative clauses and prepositional phrases."""
        if token.dep_ in ("relcl", "acl"):
            # Relative clause - extract full clause
            clause_tokens = list(token.subtree)
            clause_tokens.sort(key=lambda t: t.i)
            phrase = " ".join([t.text for t in clause_tokens]).strip()
            return phrase if len(phrase) > 3 else None
        elif token.dep_ in ("prep", "pcomp"):
            # Prepositional phrase
            prep_tokens = list(token.subtree)
            prep_tokens.sort(key=lambda t: t.i)
            phrase = " ".join([t.text for t in prep_tokens]).strip()
            return phrase if len(phrase) > 2 else None
        return None
    
    def _extract_negation_phrase(self, neg_token, head_token) -> Optional[str]:
        """Extract negation constructions."""
        phrase_tokens = [neg_token]
        
        # Find what's being negated
        if head_token.pos_ == "VERB":
            phrase_tokens.append(head_token)
            # Include verb complements
            for child in head_token.children:
                if child.dep_ in ("acomp", "attr", "dobj"):
                    phrase_tokens.extend(list(child.subtree))
        elif head_token.pos_ in ("ADJ", "NOUN"):
            phrase_tokens.append(head_token)
        
        phrase_tokens.sort(key=lambda t: t.i)
        phrase = " ".join([t.text for t in phrase_tokens]).strip()
        return phrase if len(phrase) > 2 else None
    
    def _clean_and_filter_modifiers(self, modifiers: List[str], entity: str) -> List[str]:
        """Clean and filter extracted modifiers."""
        cleaned = []
        entity_lower = entity.lower()
        
        for mod in modifiers:
            if not mod or len(mod.strip()) < 2:
                continue
            
            mod_clean = mod.strip()
            
            # Skip if modifier is just the entity itself
            if mod_clean.lower() == entity_lower:
                continue
            
            # Skip generic words
            if mod_clean.lower() in {"the", "a", "an", "this", "that", "it", "is", "was", "are", "were"}:
                continue
            
            # Skip if too long (likely extracted too much)
            if len(mod_clean) > 100:
                continue
                
            cleaned.append(mod_clean)
        
        return sorted(list(set(cleaned)))

    @lru_cache(maxsize=4096)
    def extract(self, text: str, entity: str) -> Dict[str, Any]:
        key_text = (text or "").strip()
        key_ent = (entity or "").strip()
        return self._extract_helper(key_text, key_ent)
