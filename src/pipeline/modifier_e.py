import json
import re
import time
from typing import Dict, Any, List, Optional, Tuple
from functools import lru_cache
from copy import deepcopy
import logging
import os

import config
from cache_utils import load_cache_from_file, save_cache_to_file

try:
    from utility import get_env as _root_get_env, ensure_env_loaded as _root_ensure_env_loaded
except Exception:  # pragma: no cover - defensive fallback for test environments
    try:
        from src.utility import get_env as _root_get_env, ensure_env_loaded as _root_ensure_env_loaded  # type: ignore
    except Exception:
        try:
            from pipeline.utility import get_env as _root_get_env  # type: ignore
        except Exception:
            def _root_get_env(key: str, default: Optional[str] = None) -> Optional[str]:  # type: ignore
                import os
                return os.environ.get(key, default)

        def _root_ensure_env_loaded() -> None:
            return None
else:
    if not callable(_root_ensure_env_loaded):
        try:
            from src.utility import ensure_env_loaded as _root_ensure_env_loaded  # type: ignore
        except Exception:
            def _root_ensure_env_loaded() -> None:  # pragma: no cover - fallback when ensure_env_loaded missing
                return None

get_env = _root_get_env
ensure_env_loaded = _root_ensure_env_loaded

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
        cache_only: bool = False,
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
        self.cache_only = cache_only
        if self.cache_file:
            self._cache = load_cache_from_file(self.cache_file)
        else:
            self._cache: Dict[tuple[str, str], Dict[str, Any]] = {}
        if self.cache_only:
            return
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
You are an expert linguist performing aspect-based sentiment analysis. Extract ONLY strongly evaluative predicates that express clear positive or negative sentiment about the ENTITY's INHERENT qualities, characteristics, or performance.

## CRITICAL DISTINCTIONS

**EVALUATIVE (sentiment-bearing) vs NON-EVALUATIVE (factual/descriptive):**

STRONGLY EVALUATIVE (extract these):
  ✓ "is excellent" - clear positive evaluation of entity's quality
  ✓ "works perfectly" - clear positive evaluation of entity's performance
  ✓ "is unmatched in quality" - clear positive evaluation of entity's superiority
  ✓ "is disappointing" - clear negative evaluation of entity's quality
  ✓ "has terrible performance" - clear negative evaluation of entity's attribute

WEAKLY EVALUATIVE or NEUTRAL (DO NOT extract):
  ✗ "had something for everyone" - too neutral, no clear sentiment direction
  ✗ "is scrumptious" - applies to food context but entity may not be food
  ✗ "is nice" - too weak/vague
  ✗ "is ok" - neutral/weak
  
NON-EVALUATIVE (DO NOT extract):
  ✗ "is my main computer" - states possession/relationship
  ✗ "is Windows 8" - identifies/equates
  ✗ "is either A or B" - presents options
  ✗ "is a design" - categorizes
  ✗ "since this is X" - provides context

## MANDATORY EXTRACTION RULES

**Rule 1: ENTITY MUST BE GRAMMATICAL SUBJECT WITH INHERENT QUALITY**
ONLY extract if the ENTITY is the subject performing/possessing the quality.
→ "Leonardo is clever" - YES (Leonardo possesses cleverness)
→ "The screen is bright" - YES (screen possesses brightness)
→ "Alistair was bullied" - NO (Alistair is object of action, not possessing quality)
→ "The food was eaten" - NO (food is object, not possessing quality)

**CRITICAL: DO NOT EXTRACT ACTIONS DONE TO THE ENTITY**
If someone/something else acts upon the entity, that's a RELATION, not a modifier:
  ✗ "was bullied" - action done TO entity (relation extraction's job)
  ✗ "was cheated" - action done TO entity (relation extraction's job)
  ✗ "was criticized" - action done TO entity (relation extraction's job)
  ✗ "was praised" - action done TO entity (relation extraction's job)
  ✗ "got stolen" - action done TO entity (relation extraction's job)

ONLY extract if entity is the AGENT (doer) or POSSESSOR of the quality:
  ✓ "is clever" - entity possesses cleverness
  ✓ "works well" - entity performs action
  ✓ "has excellent design" - entity possesses quality

**Rule 2: STRONG SENTIMENT REQUIRED**
Only extract modifiers that have CLEAR positive or negative sentiment.
→ "excellent", "terrible", "perfect", "awful", "amazing", "horrible" = YES
→ "something", "nice", "ok", "fine", "interesting" = NO (too weak/neutral)

**Rule 3: COMPLETE PREPOSITIONAL PHRASES**
If predicate has "in/for/with/about", include the ENTIRE prepositional phrase.
WRONG: "is unmatched" (incomplete)
RIGHT: "is unmatched in product quality" (complete)

**Rule 4: AVOID IDENTIFYING CONSTRUCTIONS**
Pattern: "is a/an/the [noun]"
→ This IDENTIFIES what something is, doesn't EVALUATE its quality
Examples to SKIP:
  - "is a defective design" (identifies as design)
  - "is the biggest complaint" (identifies role)
  - "is my main computer" (identifies possession)

**Rule 5: DISJUNCTIVE/LIST CONTEXTS**
Pattern: "either A or B", "A, B, or C"
→ Entities in lists are being presented as options, NOT evaluated
→ Extract modifiers ONLY if there's explicit strong evaluation

**Rule 6: PRONOUNS ARE NOT ASPECTS**
If ENTITY is: "this", "that", "it", "either", "my", "the", "their", "everything"
→ These are VAGUE references - return EMPTY modifiers unless there's VERY strong direct evaluation
→ The aspect extractor likely made an error; don't compound it

**Rule 7: POSSESSIVE/CONTEXTUAL PHRASES**
Patterns to AVOID:
  - "is my/your/their [X]"
  - "since [entity] is [X]"
  - "had [something]"
  - "has [something]"
→ These describe relationships or states, NOT the entity's inherent qualities

**Rule 8: CONTEXT MISMATCH**
If the modifier sentiment applies to a different context than the entity:
  - "is scrumptious" only for food/drink, NOT for "everything", "service", "atmosphere"
  - "is lush" only for physical spaces/plants, NOT for abstract concepts
→ Skip modifiers that don't semantically match the entity type

## OUTPUT REQUIREMENTS

Return **one** valid JSON object (no markdown fences):
{{
    "entity": "{entity}",
    "approach_used": "{self.model}",
    "justification": "Brief explanation of what was found/excluded and why",
    "ordered_modifiers": ["complete verbatim phrases in order"],
    "modifiers": [
        {{
            "text": "complete verbatim predicate with all necessary phrases",
            "order": 1,
            "context_type": "primary|contrast|value|usage|temporal|condition|desire|comparison|other",
            "contains_negation": false,
            "evidence_clause": "the source clause",
            "note": "why this is strongly evaluative of the ENTITY's inherent quality"
        }}
    ]
}}

## EXAMPLES WITH EXPLANATIONS

Example 1:
PASSAGE: "Apple is unmatched in product quality, aesthetics, and customer service."
ENTITY: "product quality"
CORRECT: [{{"text": "is unmatched", ...}}]
REASONING: "is unmatched" is strongly positive and directly evaluates superiority

Example 2:
PASSAGE: "Leonardo also bullied Alistair."
ENTITY: "Alistair"
CORRECT: []
REASONING: "bullied" is an action done TO Alistair, not a quality Alistair possesses. This is a relation, not a modifier.

Example 3:
PASSAGE: "The clever Leonardo used expired flour."
ENTITY: "Leonardo"
CORRECT: [{{"text": "is clever", ...}}]
REASONING: "is clever" is a quality Leonardo possesses (inherent attribute)

Example 4:
PASSAGE: "their brunch menu had something for everyone"
ENTITY: "their brunch menu"
CORRECT: []
REASONING: "had something for everyone" is neutral/descriptive, not evaluative

Example 5:
PASSAGE: "everything is scrumptious, from the excellent service to the extremely lush atmosphere"
ENTITY: "everything"
CORRECT: []
REASONING: "everything" is too vague, and "is scrumptious" applies to food, not abstract concepts

Example 6:
```

<</SYS>>

PASSAGE:
{passage}

ENTITY:
{entity}
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
            return deepcopy(cached)
        if self.cache_only:
            raise RuntimeError("Cache miss for GemmaModifierExtractor in cache-only mode")
        p = self._prompt(text, entity)
        last_err = None
        for i in range(self.retries + 1):
            try:
                raw = self._call(p)
                data = _parse_json_payload(raw) or {}

                allowed_context_types = {
                    "primary",
                    "contrast",
                    "value",
                    "usage",
                    "temporal",
                    "condition",
                    "desire",
                    "comparison",
                    "other",
                }

                def _clean_text(value: str) -> str:
                    return re.sub(r"\s+", " ", value.strip()) if isinstance(value, str) else ""

                raw_modifiers = data.get("modifiers", [])
                annotations: List[Dict[str, Any]] = []
                order_entries: List[Tuple[float, int]] = []

                if isinstance(raw_modifiers, list):
                    for idx, entry in enumerate(raw_modifiers):
                        if isinstance(entry, dict):
                            raw_text = entry.get("text")
                            cleaned_text = _clean_text(raw_text)
                            if not cleaned_text:
                                continue
                            order_val = entry.get("order")
                            if isinstance(order_val, (int, float)):
                                order_num = float(order_val)
                            else:
                                order_num = float(idx + 1)
                            context_type = _clean_text(entry.get("context_type") or "primary").lower() or "primary"
                            if context_type not in allowed_context_types:
                                context_type = "other"
                            contains_neg = bool(entry.get("contains_negation", False))
                            evidence_clause = _clean_text(entry.get("evidence_clause") or "")
                            note_raw = entry.get("note") or entry.get("justification") or entry.get("explanation") or ""
                            note = _clean_text(note_raw)
                            annotation = {
                                "text": cleaned_text,
                                "order": int(order_num) if order_num.is_integer() else order_num,
                                "context_type": context_type,
                                "contains_negation": contains_neg,
                                "evidence_clause": evidence_clause,
                                "note": note,
                            }
                            annotations.append(annotation)
                            order_entries.append((order_num, len(annotations) - 1))
                        elif isinstance(entry, str):
                            cleaned_text = _clean_text(entry)
                            if not cleaned_text:
                                continue
                            annotation = {
                                "text": cleaned_text,
                                "order": idx + 1,
                                "context_type": "primary",
                                "contains_negation": False,
                                "evidence_clause": "",
                                "note": "",
                            }
                            annotations.append(annotation)
                            order_entries.append((float(idx + 1), len(annotations) - 1))

                raw_ordered = data.get("ordered_modifiers", [])
                if isinstance(raw_ordered, list):
                    index_map = {ann["text"].lower(): idx for idx, ann in enumerate(annotations)}
                    for idx, entry in enumerate(raw_ordered):
                        if not isinstance(entry, str):
                            continue
                        cleaned_text = _clean_text(entry)
                        if not cleaned_text:
                            continue
                        key = cleaned_text.lower()
                        if key in index_map:
                            ann_idx = index_map[key]
                        else:
                            ann_idx = len(annotations)
                            annotations.append({
                                "text": cleaned_text,
                                "order": idx + 1,
                                "context_type": "primary",
                                "contains_negation": False,
                                "evidence_clause": "",
                                "note": "",
                            })
                            index_map[key] = ann_idx
                        order_entries.append((float(idx + 1), ann_idx))

                if not annotations and isinstance(raw_modifiers, list):
                    for idx, entry in enumerate(raw_modifiers):
                        if isinstance(entry, str):
                            cleaned_text = _clean_text(entry)
                            if cleaned_text:
                                annotations.append({
                                    "text": cleaned_text,
                                    "order": idx + 1,
                                    "context_type": "primary",
                                    "contains_negation": False,
                                    "evidence_clause": "",
                                    "note": "",
                                })
                                order_entries.append((float(idx + 1), len(annotations) - 1))

                ordered_annotations: List[Dict[str, Any]] = []
                ordered_texts: List[str] = []
                seen_texts: set[str] = set()
                for _, ann_idx in sorted(order_entries, key=lambda item: (item[0], item[1])):
                    if ann_idx < 0 or ann_idx >= len(annotations):
                        continue
                    record = annotations[ann_idx]
                    text_value = _clean_text(record.get("text", ""))
                    if not text_value:
                        continue
                    key = text_value.lower()
                    if key in seen_texts:
                        continue
                    seen_texts.add(key)
                    record_copy = dict(record)
                    record_copy["order"] = len(ordered_texts) + 1
                    ordered_annotations.append(record_copy)
                    ordered_texts.append(text_value)

                if not ordered_texts and annotations:
                    for record in annotations:
                        text_value = _clean_text(record.get("text", ""))
                        if not text_value:
                            continue
                        key = text_value.lower()
                        if key in seen_texts:
                            continue
                        seen_texts.add(key)
                        record_copy = dict(record)
                        record_copy["order"] = len(ordered_texts) + 1
                        ordered_annotations.append(record_copy)
                        ordered_texts.append(text_value)

                norm = ordered_texts
                out = {
                    "entity": entity,
                    "modifiers": norm,
                    "ordered_modifiers": ordered_texts,
                    "modifier_annotations": ordered_annotations,
                    "approach_used": self.model,
                    "justification": data.get("justification", "")[:500]
                }
                self._cache[cache_key] = deepcopy(out)
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
        
                                          
        entity_tokens = []
        entity_lower = e.lower()
        
                                      
        for i, token in enumerate(doc):
            if token.text.lower() == entity_lower:
                entity_tokens.append(token)
            elif entity_lower in token.text.lower():
                entity_tokens.append(token)
        
                                                         
        if not entity_tokens:
            entity_words = {w for w in entity_lower.split() if w not in {"the","a","an","this","that"}}
            for token in doc:
                if token.text.lower() in entity_words:
                    entity_tokens.append(token)
        
        if not entity_tokens:
            return {"entity": entity, "modifiers": [], "approach_used": "spacy_enhanced", "justification": "entity not found"}
        
                                     
        heads = set()
        for token in entity_tokens:
            head = token
            while head.head != head and head.head != head.head:
                head = head.head
            heads.add(head)
        
        mods = []
        
                                                               
        for head in heads:
                                                
            for child in head.children:
                if child.dep_ in ("amod", "acomp", "nmod", "compound"):
                    mod_phrase = self._extract_full_modifier_phrase(child)
                    if mod_phrase:
                        mods.append(mod_phrase)
            
                                      
            for child in head.children:
                if child.dep_ in ("advmod", "npadvmod"):
                    mod_phrase = self._extract_full_modifier_phrase(child)
                    if mod_phrase:
                        mods.append(mod_phrase)
            
                                                      
            if head.head != head:
                parent = head.head
                if parent.pos_ == "VERB":
                                                             
                    verb_phrase = self._extract_verb_phrase(parent, head)
                    if verb_phrase:
                        mods.append(verb_phrase)
            
                                                           
            for child in head.children:
                if child.dep_ in ("relcl", "acl", "prep", "pcomp"):
                    clause_phrase = self._extract_clause_phrase(child)
                    if clause_phrase:
                        mods.append(clause_phrase)
                        
                                                   
            for child in head.children:
                if child.dep_ in ("neg", "aux", "auxpass"):
                    neg_phrase = self._extract_negation_phrase(child, head)
                    if neg_phrase:
                        mods.append(neg_phrase)
        
                                    
        unique_mods = self._clean_and_filter_modifiers(mods, entity)
        return {"entity": entity, "modifiers": unique_mods, "approach_used": "spacy_enhanced", "justification": f"{len(unique_mods)} enhanced patterns"}
    
    def _extract_full_modifier_phrase(self, token) -> Optional[str]:
        """Extract full modifier phrase including intensifiers and qualifiers."""
        phrase_tokens = []
        
                                                             
        for child in token.children:
            if child.dep_ == "advmod":
                phrase_tokens.append(child)
        
                               
        phrase_tokens.append(token)
        
                                        
        for child in token.children:
            if child.dep_ in ("prep", "pcomp", "dobj", "attr"):
                phrase_tokens.extend(list(child.subtree))
        
        phrase_tokens.sort(key=lambda t: t.i)
        phrase = " ".join([t.text for t in phrase_tokens]).strip()
        return phrase if len(phrase) > 1 else None
    
    def _extract_verb_phrase(self, verb_token, entity_token) -> Optional[str]:
        """Extract full verb phrase that relates to the entity."""
        phrase_tokens = []
        
                                                      
        entity_is_subj = any(child.dep_ in ("nsubj", "nsubjpass") and child == entity_token for child in verb_token.children)
        entity_is_obj = any(child.dep_ in ("dobj", "pobj") and child == entity_token for child in verb_token.children)
        
        if not (entity_is_subj or entity_is_obj):
            return None
        
                                          
        for child in verb_token.children:
            if child.dep_ in ("aux", "auxpass", "neg"):
                phrase_tokens.append(child)
        
                       
        phrase_tokens.append(verb_token)
        
                                     
        for child in verb_token.children:
            if child.dep_ in ("advmod", "npadvmod"):
                phrase_tokens.extend(list(child.subtree))
        
                                                                     
        for child in verb_token.children:
            if child.dep_ in ("dobj", "attr", "acomp") and child != entity_token:
                phrase_tokens.extend(list(child.subtree))
        
        phrase_tokens.sort(key=lambda t: t.i)
        phrase = " ".join([t.text for t in phrase_tokens]).strip()
        return phrase if len(phrase) > 1 else None
    
    def _extract_clause_phrase(self, token) -> Optional[str]:
        """Extract relative clauses and prepositional phrases."""
        if token.dep_ in ("relcl", "acl"):
                                                   
            clause_tokens = list(token.subtree)
            clause_tokens.sort(key=lambda t: t.i)
            phrase = " ".join([t.text for t in clause_tokens]).strip()
            return phrase if len(phrase) > 3 else None
        elif token.dep_ in ("prep", "pcomp"):
                                  
            prep_tokens = list(token.subtree)
            prep_tokens.sort(key=lambda t: t.i)
            phrase = " ".join([t.text for t in prep_tokens]).strip()
            return phrase if len(phrase) > 2 else None
        return None
    
    def _extract_negation_phrase(self, neg_token, head_token) -> Optional[str]:
        """Extract negation constructions."""
        phrase_tokens = [neg_token]
        
                                   
        if head_token.pos_ == "VERB":
            phrase_tokens.append(head_token)
                                      
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
            
                                                        
            if mod_clean.lower() == entity_lower:
                continue
            
                                
            if mod_clean.lower() in {"the", "a", "an", "this", "that", "it", "is", "was", "are", "were"}:
                continue
            
                                                          
            if len(mod_clean) > 100:
                continue
                
            cleaned.append(mod_clean)
        
        return sorted(list(set(cleaned)))

    @lru_cache(maxsize=4096)
    def extract(self, text: str, entity: str) -> Dict[str, Any]:
        key_text = (text or "").strip()
        key_ent = (entity or "").strip()
        return self._extract_helper(key_text, key_ent)
