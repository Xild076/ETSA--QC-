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
    You are a Principal Linguistic Engineer. Your task is not just to extract information but to model a rigorous, step-by-step reasoning process for sentiment analysis. You MUST externalize this process in a "Mental Sandbox" before providing the final JSON output.

    ### The Expert Analyst's Reasoning Framework (You MUST follow these steps)

    1.  **Heuristic #1: Scan for Obvious Sentiment.** First, look for simple, powerful sentiment words ("love," "hate," "perfect," "nice," "awful," "great," "quickly," "safely," "dreadful") directly describing the ENTITY or its associated action. Prioritize this signal.

    2.  **Heuristic #2: Analyze the Predicate and Structure.** Identify the main verb phrase. Pay close attention to parallel structures. If a sentiment applies to the first item in a list (e.g., "easy to carry and handle"), it applies to all items.

    3.  **Heuristic #3: Evaluate the Consequence (User Experience).** What was the *result* of the entity's action? A neutral-sounding event can have a strong sentiment based on its outcome. A seemingly negative event (like an app restarting) can be positive if it's framed as a sign of stability that prevents a worse outcome.

    4.  **Heuristic #4: Check for Context-Dependent Polarity (CRITICAL).** The sentiment of a word is not fixed. You must use world knowledge.
        *   **Comparatives:** "**higher** price" is negative. "**less** expensive" is positive. "**bigger** power switch" may imply the old one was too small (negative).
        *   **Quantifiers & Absence:** "**ZERO** bloatware" is positive. "**only** 2 ports" is negative. "**removed** the jack" or "**except for** a program" indicates a missing feature (negative).
        *   **Idioms:** "can't beat" means "is excellent." "melts in your mouth" is positive.

    5.  **Heuristic #5: Identify the Speaker's True Stance.** Distinguish the author's opinion from opinions they are quoting or describing, especially in contrastive or counterfactual structures.
        *   **Refutation:** "Some complain about X, but I think it's great." -> The author's refutation determines the sentiment.
        *   **Counterfactual:** "I *would have* loved it *if not for* [the entity]." -> This is a strong negative evaluation of [the entity].

    6.  **Heuristic #6: Differentiate Evaluation vs. Neutral Statement.** Is the sentence evaluating the product, or just stating its *purpose*, a *user's past habit*, or a *factual list*?
        *   "I'm using this for **gaming**." -> Neutral statement of purpose.
        *   "I'm used to **Android**, so this is confusing." -> Neutral mention of "Android" as a reference for a user's habit.
        *   "Entrees include **lasagna**." -> Neutral factual listing.

    7.  **Heuristic #7: Final Attribution Check.** Confirm the evaluation is precisely about the TARGET ENTITY.
        *   "**Support** said the OS was corrupted." -> The negativity applies to the "OS," not "Support."

    ### Output Structure
    First, provide your step-by-step reasoning. Then, provide the final JSON.
<</SYS>>

<<PASSAGE>>
{passage}
</PASSAGE>>

<<ENTITY>>
{entity}
</ENTITY>>

<<EXAMPLES>>
P: The pizza was great, and the portion size was generous. E: portion size ->
Mental Sandbox (Internal Monologue):
1.  **Obvious Sentiment:** "great", "generous".
2.  **Predicate:** The entity "portion size" is the subject of the predicate "was generous".
3.  **Consequence:** N/A.
4.  **Context:** "generous" is a direct positive evaluation in this context.
5.  **Speaker's Stance:** Direct opinion.
6.  **Reality vs. Scope:** Describes reality.
7.  **Attribution:** "was generous" directly modifies "portion size".
Final Answer Formulation: The entity is the subject of a direct, positive predicate.
{{
"entity":"portion size",
"justification":"The entity is directly evaluated by the positive predicate 'was generous'.",
"modifiers":["was generous"],
"approach_used":"{self.model}"
}}

P: The brisket literally melts in your mouth! E: brisket ->
Mental Sandbox (Internal Monologue):
1.  **Obvious Sentiment:** "melts in your mouth" is a known positive phrase.
2.  **Predicate:** "literally melts in your mouth".
3.  **Consequence:** N/A.
4.  **Context:** This is a common positive idiom for tender food.
5.  **Speaker's Stance:** Direct opinion.
6.  **Reality vs. Scope:** Describes reality.
7.  **Attribution:** The idiom describes the "brisket".
Final Answer Formulation: Idiomatic evaluation.
{{
"entity":"brisket",
"justification":"Heuristic #4 (Context-Dependent Polarity) applies. The phrase 'melts in your mouth' is a well-known positive idiom for tender food.",
"modifiers":["literally melts in your mouth"],
"approach_used":"{self.model}"
}}

P: I would have loved this laptop, were it not for the terrible keyboard. E: keyboard ->
Mental Sandbox (Internal Monologue):
1.  **Obvious Sentiment:** "loved" (positive), "terrible" (negative).
2.  **Predicate:** The structure is a counterfactual conditional.
3.  **Consequence:** The keyboard's quality ruined the entire experience.
4.  **Context:** N/A.
5.  **Speaker's Stance:** Heuristic #5 is key. The structure 'would have [positive] if not for [entity]' establishes that the entity is the single blocking point for a positive experience. This is a very strong negative evaluation of the entity.
6.  **Reality vs. Scope:** Describes reality.
7.  **Attribution:** The negativity is squarely placed on "the terrible keyboard."
Final Answer Formulation: A counterfactual conditional identifies the entity as the source of failure.
{{
"entity":"keyboard",
"justification":"Heuristic #5 (Speaker's Stance) applies. The counterfactual 'would have loved... were it not for...' structure pinpoints the entity as the sole reason for a negative overall experience, which is a strong negative evaluation.",
"modifiers":["terrible"],
"approach_used":"{self.model}"
}}

P: I'm used to Android, so the iPhone's settings are confusing at first. E: Android ->
Mental Sandbox (Internal Monologue):
1.  **Obvious Sentiment:** "confusing" is negative.
2.  **Predicate:** N/A for Android.
3.  **Consequence:** N/A for Android.
4.  **Context:** N/A.
5.  **Speaker's Stance:** N/A.
6.  **Reality vs. Scope:** Heuristic #6 applies. The sentence uses "Android" as a neutral reference point to explain the user's personal learning curve with a different system. It is not evaluating Android.
7.  **Attribution:** The negative sentiment "confusing" applies to the "iPhone's settings," not to "Android."
Final Answer Formulation: The entity is a neutral reference for user habits.
{{
"entity":"Android",
"justification":"Heuristic #6 (Differentiate Evaluation vs. Neutral Statement) applies. The entity 'Android' is used as a neutral reference to explain the user's personal experience/learning curve with a different product. It is not being evaluated.",
"modifiers":[],
"approach_used":"{self.model}"
}}

P: The system is resilient; even when a program crashes, it simply restarts without bringing down the OS. E: program ->
Mental Sandbox (Internal Monologue):
1.  **Obvious Sentiment:** "crashes" is locally negative. "resilient" is globally positive.
2.  **Predicate:** "crashes".
3.  **Consequence:** Heuristic #3 is critical. A program crashing is usually negative. However, the sentence frames this as a positive. The consequence is that it "restarts without bringing down the OS," which is presented as a sign of the overall system's resilience. The crash is part of a positive story about stability.
4.  **Context:** The context "The system is resilient" sets a positive frame.
5.  **Speaker's Stance:** Direct observation.
6.  **Reality vs. Scope:** Describes reality.
7.  **Attribution:** The action is attributed to the program.
Final Answer Formulation: The consequence of the action is framed as a positive demonstration of system stability.
{{
"entity":"program",
"justification":"Heuristic #3 (Evaluate Consequence) applies. Although 'crashes' is a negative action, the sentence frames this as a positive outcome ('restarts without bringing down the OS'), demonstrating the system's resilience. The event contributes to an overall positive evaluation.",
"modifiers":["crashes", "simply restarts without bringing down the OS"],
"approach_used":"{self.model}"
}}
</EXAMPLES>>

<<RESPONSE>>
Mental Sandbox (Internal Monologue):
1.  **Obvious Sentiment:**
2.  **Predicate:**
3.  **Consequence:**
4.  **Idioms & Context:**
5.  **Speaker's Stance:**
6.  **Reality vs. Scope:**
7.  **Attribution:**
Final Answer Formulation:
{{
"entity": "{entity}",
"justification": "Explain how you identified the modifiers based on the heuristic analysis.",
"modifiers": [],
"approach_used":"{self.model}"
}}
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
        max_rate_limit_retries = 5  # More retries for rate limiting
        attempt = 0
        max_attempts = max(self.retries + 1, max_rate_limit_retries + 1)
        
        while attempt < max_attempts:
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
                error_str = str(e).lower()
                
                # Check if this is a rate limiting error
                is_rate_limit = (
                    "429" in str(e) or 
                    "quota" in error_str or 
                    "rate limit" in error_str or
                    "resource exhausted" in error_str or
                    "too many requests" in error_str
                )
                
                if is_rate_limit:
                    # Use exponential backoff for rate limiting with longer waits
                    retry_attempt = attempt + 1
                    if retry_attempt <= max_rate_limit_retries:
                        # Exponential backoff: 2, 4, 8, 16, 32 seconds
                        wait_time = min(2 ** retry_attempt, 60)  # Cap at 60 seconds
                        logger.warning(
                            f"⚠️  RATE LIMIT hit for entity '{entity[:50]}...' - "
                            f"Retry {retry_attempt}/{max_rate_limit_retries} after {wait_time}s wait"
                        )
                        print(
                            f"⚠️  RATE LIMIT: Waiting {wait_time}s before retry {retry_attempt}/{max_rate_limit_retries} "
                            f"for entity '{entity[:50]}{'...' if len(entity) > 50 else ''}'"
                        )
                        time.sleep(wait_time)
                        attempt += 1
                        continue
                    else:
                        logger.error(
                            f"❌ RATE LIMIT exceeded after {max_rate_limit_retries} retries for entity '{entity[:50]}...'"
                        )
                        print(
                            f"❌ RATE LIMIT: Failed after {max_rate_limit_retries} retries "
                            f"for entity '{entity[:50]}{'...' if len(entity) > 50 else ''}'"
                        )
                        # Cache the failure to avoid retrying the same entity repeatedly
                        failure_result = {
                            "entity": entity, 
                            "modifiers": [], 
                            "ordered_modifiers": [],
                            "modifier_annotations": [],
                            "approach_used": self.model, 
                            "justification": f"RATE_LIMIT_EXCEEDED: {str(e)[:200]}"
                        }
                        self._cache[cache_key] = deepcopy(failure_result)
                        if self.cache_file:
                            save_cache_to_file(self.cache_file, self._cache)
                        return failure_result
                else:
                    # Non-rate-limit error: use standard backoff
                    if attempt < self.retries:
                        wait_time = self.backoff * (attempt + 1)
                        logger.warning(f"API error for entity '{entity[:50]}...': {str(e)[:100]} - Retrying in {wait_time}s")
                        time.sleep(wait_time)
                        attempt += 1
                        continue
                    else:
                        logger.error(f"API error after {self.retries + 1} attempts for entity '{entity[:50]}...': {str(e)[:100]}")
                        break
                        
        # Final fallback after all retries
        failure_result = {
            "entity": entity, 
            "modifiers": [], 
            "ordered_modifiers": [],
            "modifier_annotations": [],
            "approach_used": self.model, 
            "justification": f"EXTRACTION_FAILED: {str(last_err)[:200]}"
        }
        # Cache failures to avoid repeated attempts
        self._cache[cache_key] = deepcopy(failure_result)
        if self.cache_file:
            save_cache_to_file(self.cache_file, self._cache)
        return failure_result

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
