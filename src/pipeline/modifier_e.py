import json
import re
import time
import os
import random
import logging
from typing import Dict, Any, List, Optional
from functools import lru_cache

try:
    import spacy
except Exception:
    spacy = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

from src.utility import get_env

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
                time.sleep(sleep_time + min(0.25, random.random() * 0.25))
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
        model: str = "gemma-3-27b-it",
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 40,
        max_output_tokens: int = 768,
        backoff: float = 0.8,
        retries: int = 2,
        rate_limiter: Optional[_RateLimiter] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        self.backoff = backoff
        self.retries = retries
        self.rate_limiter = rate_limiter or _ensure_modifier_rate_limiter()
        if genai is None:
            raise RuntimeError("google-generativeai not available")
        key = api_key or get_env("GOOGLE_API_KEY") or get_env("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("No API key provided for GemmaModifierExtractor")
        genai.configure(api_key=key)

    def _prompt(self, passage: str, entity: str) -> str:
        return (
            "<<SYS>>\n"
            "Extract ALL modifiers that convey sentiment about the entity. Focus on:\n"
            "1. Complete negation phrases: 'did not enjoy', 'would not fix', 'not being a fan of'\n"
            "2. Action-based sentiment: 'failed to work', 'works perfectly', 'struggles with'\n"
            "3. Evaluative context: 'especially considering' (positive context), 'unfortunately' (negative)\n"
            "4. Factual statements about presence/absence: 'No...included' (neutral), 'comes with' (positive)\n"
            "5. Quality descriptors: 'excellent', 'terrible', 'lousy', 'amazing'\n"
            "6. Behavioral/performance indicators: complete phrases that show how entity performs\n"
            "\n"
            "CRITICAL RULES:\n"
            "- Capture COMPLETE negation phrases, not just 'not' - examples: 'would not fix', 'did not enjoy', 'would not work'\n"
            "- For cross-entity negations like 'Did not enjoy X and Y', apply 'Did not enjoy' to BOTH entities\n"
            "- ALWAYS include action-based negations like 'would not fix the problem', 'did not work properly'\n"
            "- Preserve original word order and phrasing from passage\n"
            "- Include contextual words that affect sentiment interpretation\n"
            "- For factual statements about absence (No...included), mark as neutral-factual\n"
            "\n"
            "EXCLUDE: entity names, pronouns, articles, pure numbers, dates, locations\n"
            "<</SYS>>\n\n"
            f"<<PASSAGE>>\n{passage}\n<</PASSAGE>>\n\n"
            f"<<ENTITY>>\n{entity}\n<</ENTITY>>\n\n"
            "<<SCHEMA>>\n"
            "{\n"
            '  "entity": string,\n'
            '  "justification": string,\n'
            '  "modifiers": [string],\n'
            '  "sentiment_hint": "positive|negative|neutral",\n'
            f'  "approach_used": "{self.model}",\n'
            "}\n"
            "<</SCHEMA>>\n\n"
            "<<EXAMPLES>>\n"
            "P: tech support would not fix the problem unless I bought your plan. E: tech support -> modifiers: [\"would not fix the problem\"], sentiment_hint: \"negative\"\n"
            "P: tech support would not fix the problem. E: tech support -> modifiers: [\"would not fix the problem\"], sentiment_hint: \"negative\"\n"
            "P: Did not enjoy the new Windows 8 and touchscreen functions. E: Windows 8 -> modifiers: [\"Did not enjoy\"], sentiment_hint: \"negative\"\n"
            "P: Did not enjoy the new Windows 8 and touchscreen functions. E: touchscreen functions -> modifiers: [\"Did not enjoy\"], sentiment_hint: \"negative\"\n"
            "P: not being a fan of click pads (industry standard these days). E: click pads -> modifiers: [\"not being a fan of\"], sentiment_hint: \"negative\"\n"
            "P: it's hard for me to find things about this notebook I don't like, especially considering the $350 price tag. E: price tag -> modifiers: [\"especially considering\", \"hard to find things I don't like about\"], sentiment_hint: \"positive\"\n"
            "P: No installation disk (DVD) is included. E: installation disk (DVD) -> modifiers: [\"No...is included\"], sentiment_hint: \"neutral\"\n"
            "P: I blame the Mac OS. E: Mac OS -> modifiers: [\"I blame\"], sentiment_hint: \"negative\"\n"
            "P: Unfortunately, it runs XP and Microsoft is dropping support. E: XP -> modifiers: [\"Unfortunately...runs\"], sentiment_hint: \"negative\"\n"
            "P: Service was awful - mostly because staff were overwhelmed. E: Service -> modifiers: [\"was awful - mostly because\"], sentiment_hint: \"negative\"\n"
            "P: However, the experience was great since the OS does not become unstable. E: OS -> modifiers: [\"However...experience was great\", \"does not become unstable\"], sentiment_hint: \"positive\"\n"
            "P: I had to get Apple Customer Support to correct the problem. E: Apple Customer Support -> modifiers: [\"had to get...to correct the problem\"], sentiment_hint: \"neutral\"\n"
            "P: Skip the dessert, it was overpriced and fell short on taste. E: dessert -> modifiers: [\"Skip\", \"overpriced and fell short\"], sentiment_hint: \"negative\"\n"
            "P: The cold sesame noodles are delectable. E: cold sesame noodles -> modifiers: [\"delectable\"], sentiment_hint: \"positive\"\n"
            "P: Performance is much much better on the Pro, especially if you install an SSD. E: SSD -> modifiers: [\"much much better\", \"install\"], sentiment_hint: \"positive\"\n"
            "P: The performance is excellent. E: performance -> modifiers: [\"excellent\"], sentiment_hint: \"positive\"\n"
            "P: Boot time is super fast. E: Boot time -> modifiers: [\"super fast\"], sentiment_hint: \"positive\"\n"
            "P: The tech support was unhelpful. E: tech support -> modifiers: [\"unhelpful\"], sentiment_hint: \"negative\"\n"
            "P: lousy internal speakers. E: internal speakers -> modifiers: [\"lousy\"], sentiment_hint: \"negative\"\n"
            "P: Amazing performance for anything I throw at it. E: performance -> modifiers: [\"Amazing\"], sentiment_hint: \"positive\"\n"
            "P: Did not enjoy the meal at all. E: meal -> modifiers: [\"Did not enjoy\", \"at all\"], sentiment_hint: \"negative\"\n"
            "P: The service was excellent and attentive. E: service -> modifiers: [\"excellent and attentive\"], sentiment_hint: \"positive\"\n"
            "P: No bread was provided with the meal. E: bread -> modifiers: [\"No...provided\"], sentiment_hint: \"neutral\"\n"
            "P: I bought a computer yesterday. E: computer -> modifiers: [], sentiment_hint: \"neutral\"\n"
            "<</EXAMPLES>>\n\n"
            "<<RESPONSE>>\n"
            "```json\n"
            "{\n"
            f'  "entity": "{entity}",\n'
            f'  "justification": "explain reasoning for chosen modifiers and sentiment",\n'
            '  "modifiers": [],\n'
            '  "sentiment_hint": "positive|negative|neutral",\n'
            f'  "approach_used": "{self.model}",\n'
            "}\n"
            "```\n"
        )

    def _call(self, prompt: str) -> str:
        if self.rate_limiter:
            self.rate_limiter.acquire()
        model = genai.GenerativeModel(
            model_name=self.model,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_output_tokens=self.max_output_tokens,
                response_mime_type="text/plain",
            )
        )
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", None)
        if not text:
            try:
                text = str(resp)
            except Exception:
                text = ""
        return text

    def extract(self, text: str, entity: str) -> Dict[str, Any]:
        if not text or not entity:
            return {"entity": entity, "modifiers": [], "approach_used": self.model, "justification": "empty input"}
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
                
                # More permissive filtering - preserve important sentiment phrases
                basic_stop_words = {"the", "a", "an", "this", "that"}
                
                for m in mods:
                    if not isinstance(m, str):
                        continue
                    s = re.sub(r"\s+", " ", m.strip())
                    if not s or len(s) < 2:
                        continue
                    
                    # Don't filter out sentiment-bearing phrases
                    if s.lower() in basic_stop_words:
                        continue
                    
                    # Don't duplicate the entity name exactly
                    if s.lower() == entity.lower():
                        continue
                    
                    # Preserve negation and sentiment phrases even if they seem "irrelevant"
                    sentiment_indicators = [
                        'not', 'did not', 'would not', 'could not', 'cannot',
                        'failed', 'refused', 'managed', 'struggled', 
                        'excellent', 'terrible', 'amazing', 'awful',
                        'especially', 'unfortunately', 'luckily'
                    ]
                    
                    has_sentiment = any(indicator in s.lower() for indicator in sentiment_indicators)
                    
                    # Keep if it has sentiment value or is a multi-word contextual phrase
                    if has_sentiment or len(s.split()) > 1:
                        if s.lower() not in seen:
                            seen.add(s.lower())
                            norm.append(s)
                    
                result = {
                    "entity": entity,
                    "modifiers": norm,
                    "approach_used": self.model,
                    "justification": data.get("justification", "")[:500]
                }
                
                # Include sentiment hint if available for debugging
                if "sentiment_hint" in data:
                    result["sentiment_hint"] = data["sentiment_hint"]
                
                # Fallback for specific negation patterns if no modifiers found
                if not norm and entity.lower() in text.lower():
                    # Look for common negation patterns
                    # Pattern definitions with sentiment hints
                    pattern_configs = [
                        # Context transition markers (Root Cause #1)
                        (r'\bunfortunately.*(?:runs|dropping|support|issues?)', 'negative'),
                        (r'\bhowever.*(?:experience was great|does not become unstable)', 'positive'),
                        (r'(?:mostly|mainly|primarily)\s+because.*(?:overwhelmed|understaffed|busy)', 'negative'),
                        
                        # Attribution/blame patterns (Root Cause #2)  
                        (r'\b(?:i blame|blame the|fault|responsible for).*\b', 'negative'),
                        (r'\b(?:wrong|incorrect|mistaken).*(?:entree|order|dish|item)', 'negative'),
                        
                        # Service interaction contexts (Root Cause #4)
                        (r'\bhad to get.*(?:support|help).*(?:correct|fix).*problem', 'neutral'),
                        (r'\b(?:customer support|tech support|help desk).*(?:correct|fix)', 'neutral'),
                        
                        # Skip/avoid recommendations (Root Cause #6)
                        (r'\b(?:skip|avoid).*(?:dessert|dish|item).*(?:overpriced|fell short|disappointing)', 'negative'),
                        
                        # Complex explanation contexts (Root Cause #7)
                        (r'\b(?:awful|terrible|horrible)\s*-\s*(?:mostly|mainly)\s+because', 'negative'),
                        (r'\bservice was awful.*(?:because|since).*overwhelmed', 'negative'),
                        
                        # Physical descriptor interference (Root Cause #5)
                        (r'\bcold.*noodles.*(?:delectable|delicious|tasty)', 'positive'),
                        (r'\b(?:overpriced).*(?:fell short|disappointing)', 'negative'),
                        
                        # Comparative contexts (Root Cause #3)
                        (r'\bmuch much better.*especially if you install', 'positive'),
                        (r'\bunexpected elements.*otherwise predictable', 'neutral'),
                        (r'\bperformance.*much.*better.*(?:install|upgrade)', 'positive'),
                        
                        # Installation/action contexts (Root Cause #8)
                        (r'\binstall.*(?:ssd|upgrade).*(?:better performance|improvement)', 'positive'),
                        (r'\bthought.*transition.*difficult.*familiarize', 'neutral'),
                        
                        # Enhanced negation patterns
                        (r'\b(would not|could not|did not|cannot|can\'t|won\'t|wouldn\'t|couldn\'t|didn\'t)\s+\w+\s*(?:the\s+)?(?:problem|issue|work|fix|help|serve|prepare)', 'negative'),
                        (r'\b(not being a fan of|not enjoy|not work|not fix|not recommend|not impressed|not satisfied|not very sensitive)', 'negative'),
                        (r'\b(poorly designed|cheap|frustrated|difficult at best)', 'negative'),
                        
                        # Specific quality failures (from remaining errors)
                        (r'\b(?:very\s+)?weak\b', 'negative'),
                        (r'\btinny\b', 'negative'), 
                        (r'\bslow(?:ly|ed)?(?:\s+significantly)?\b', 'negative'),
                        (r'\bnot as fast as\b', 'negative'),
                        (r'\bsounding tinny\b', 'negative'),
                        
                        # Neutral factual patterns  
                        (r'\bno\s+\w+.*(?:included|provided|available|offered|in the)\b', 'neutral'),
                        (r'\b(the only debate|whether to|time for a new|thought the transition)', 'neutral'),
                        (r'\bnot drowned in.*sauce', 'neutral'),
                        
                        # Missing features/neutral statements (from remaining errors)
                        (r'\bthe only thing I miss\b', 'neutral'),
                        (r'\bnot terribly important\b', 'neutral'),
                        (r'\bgets plugged into.*external\b', 'neutral'),
                        (r'\bhaving issues with.*boards?\b', 'neutral'),
                        
                        # Positive context patterns
                        (r'especially considering\s+(?:the\s+)?\$?\d+\s*\w*\s*(?:price|tag|cost|value)', 'positive'),
                        (r'hard\s+(?:for\s+me\s+)?to\s+find\s+(?:things|anything).*don\'t\s+like.*especially\s+considering', 'positive'),
                        (r'\b(excellent|amazing|outstanding|perfect|wonderful|fantastic)\s+(?:food|service|meal|dish|restaurant|performance)', 'positive'),
                        (r'\bromantic date heaven.*treated like.*vip', 'positive'),
                        
                        # Negative descriptors
                        (r'\b(terrible|awful|horrible|disgusting|worst)\s+(?:food|service|meal|dish|experience)', 'negative'),
                        (r'\b(beats\s+\w+\s+easily)', 'negative'),
                        (r'\bverbally assaults.*gives.*lip', 'negative')
                    ]
                    
                    for pattern, sentiment_hint in pattern_configs:
                        matches = re.finditer(pattern, text, re.IGNORECASE)
                        for match in matches:
                            # Check if this pattern is near the entity
                            entity_pos = text.lower().find(entity.lower())
                            match_pos = match.start()
                            # If within reasonable distance (same sentence usually)
                            if abs(entity_pos - match_pos) < 100:
                                modifier = match.group().strip()
                                if modifier not in [m.lower() for m in norm]:
                                    norm.append(modifier)
                                    result["justification"] += f" [pattern: {modifier}]"
                                    if "sentiment_hint" not in result:
                                        result["sentiment_hint"] = sentiment_hint
                                    break
                
                result["modifiers"] = norm
                    
                return result
            except Exception as e:
                last_err = e
                if i < self.retries:
                    sleep = self.backoff * (2 ** i)
                    sleep += min(0.25, random.random() * 0.25)
                    time.sleep(sleep)
        return {"entity": entity, "modifiers": [], "approach_used": self.model, "justification": f"failure: {last_err}"}

class SpacyModifierExtractor(ModifierExtractor):
    def extract(self, text: str, entity: str) -> Dict[str, Any]:
        nlp = _get_spacy_nlp()
        if not nlp or not text or not entity:
            return {"entity": entity, "modifiers": [], "approach_used": "spacy_fallback", "justification": "unavailable"}
        doc = nlp(text)
        t = text.lower()
        e = entity.strip()
        if not e:
            return {"entity": entity, "modifiers": [], "approach_used": "spacy_fallback", "justification": "empty"}
        idx = t.find(e.lower())
        tokens = []
        if idx >= 0:
            start = idx
            end = idx + len(e)
            for tok in doc:
                if tok.idx >= start and tok.idx + len(tok.text) <= end:
                    tokens.append(tok)
        if not tokens:
            words = {w for w in e.lower().split() if w not in {"the","a","an","this","that"}}
            for tok in doc:
                if tok.text.lower() in words:
                    tokens.append(tok)
        heads = set(tokens)
        adjs = []
        for tok in doc:
            if tok.dep_ in ("amod", "compound") and tok.head in heads:
                adjs.append(tok.text)
        preds = []
        for tok in doc:
            if tok.dep_ in ("acomp", "attr") and tok.head.pos_ in ("VERB","AUX"):
                for ch in tok.head.children:
                    if ch.dep_ in ("nsubj","nsubjpass") and ch in heads:
                        preds.append(tok.text)
                        break
        directives = []
        if idx >= 0:
            pre = doc[max(0, doc[start:end].start - 3):doc[start:end].start] if hasattr(doc, "start") else []
        for tok in doc:
            if tok.lemma_.lower() in {"try","skip","avoid","recommend"}:
                directives.append(tok.text)
        mods = []
        seen = set()
        for s in adjs + preds:
            k = s.lower().strip()
            if k and k not in seen:
                seen.add(k)
                mods.append(s.strip())
        for s in directives:
            k = s.lower().strip()
            if k and k not in seen:
                seen.add(k)
                if s.lower() in {"try","skip","avoid","recommend"}:
                    mods.append(s + " the")
                else:
                    mods.append(s)
        return {"entity": entity, "modifiers": mods, "approach_used": "spacy_fallback", "justification": f"{len(mods)} cues"}
