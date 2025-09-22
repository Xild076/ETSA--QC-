import json
import re
import time
import os
import random
from typing import Dict, Any, List, Optional
from functools import lru_cache
import logging

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
            "Extract ALL modifiers that affect the sentiment or perception of the entity. Include:\n"
            "1. Quality-describing adjectives (good/bad/fast/slow)\n"
            "2. Action-based modifiers showing capability/performance (\"would not fix\", \"failed to\", \"successfully completed\")\n" 
            "3. Behavioral descriptors (\"was helpful\", \"refused to\", \"managed to\")\n"
            "4. Performance indicators through actions (\"broke down\", \"works perfectly\", \"struggles with\")\n"
            "5. Negation patterns that affect entity perception (\"did not work\", \"would not\", \"cannot\")\n"
            "The modifier must be grammatically connected to the entity and convey evaluative meaning.\n"
            "EXCLUDE: entity names themselves, pronouns, articles, pure facts without evaluative meaning, numbers, locations, temporal references.\n"
            "Use exact text from passage or reconstruct meaningful phrases. Return strict JSON.\n"
            "<</SYS>>\n\n"
            f"<<PASSAGE>>\n{passage}\n<</PASSAGE>>\n\n"
            f"<<ENTITY>>\n{entity}\n<</ENTITY>>\n\n"
            "<<SCHEMA>>\n"
            "{\n"
            '  "entity": string,\n'
            '  "justification": string,\n'
            '  "modifiers": [string],\n'
            f'  "approach_used": "{self.model}",\n'
            "}\n"
            "<</SCHEMA>>\n\n"
            "<<EXAMPLES>>\n"
            "P: The performance is excellent. E: performance -> modifiers: [\"excellent\"]\n"
            "P: Boot time is super fast, around 35 seconds. E: Boot time -> modifiers: [\"super fast\"]\n"
            "P: I did not enjoy the new Windows 8 functions. E: Windows 8 -> modifiers: [\"did not enjoy\"]\n"
            "P: The tech support was unhelpful. E: tech support -> modifiers: [\"unhelpful\"]\n"
            "P: tech support would not fix the problem. E: tech support -> modifiers: [\"would not fix the problem\"]\n"
            "P: The screen failed to turn on properly. E: screen -> modifiers: [\"failed to turn on properly\"]\n"
            "P: Battery life cannot last more than 2 hours. E: Battery life -> modifiers: [\"cannot last more than 2 hours\"]\n"
            "P: Amazing performance for anything I throw at it. E: performance -> modifiers: [\"Amazing\"]\n"
            "P: The receiver was full of superlatives for quality. E: quality -> modifiers: []\n"
            "P: I bought a computer yesterday. E: computer -> modifiers: []\n"
            "<</EXAMPLES>>\n\n"
            "<<RESPONSE>>\n"
            "```json\n"
            "{\n"
            f'  "entity": "{entity}",\n'
            f'  "justification": "explain how and why you chose the modifiers for {entity}",\n'
            '  "modifiers": [],\n'
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
                irrelevant_words = {"the", "a", "an", "this", "that", "of", "in", "on", "at", "to", "for", "with", 
                                  "is", "are", "was", "were", "been", "be", "have", "has", "had", "do", "does", "did",
                                  "will", "would", "could", "should", "may", "might", "can", "must", "shall"}
                
                for m in mods:
                    if not isinstance(m, str):
                        continue
                    s = re.sub(r"\s+", " ", m.strip())
                    if not s or len(s) < 2:
                        continue
                    
                    if s.lower() in irrelevant_words:
                        continue
                    
                    if s.lower() == entity.lower() or entity.lower() in s.lower():
                        if s.lower() == entity.lower():
                            continue
                    
                    words = s.lower().split()
                    if all(word in irrelevant_words for word in words):
                        continue
                    
                    if s.lower() in seen:
                        continue
                    seen.add(s.lower())
                    norm.append(s)
                return {
                    "entity": entity,
                    "modifiers": norm,
                    "approach_used": self.model,
                    "justification": data.get("justification", "")[:500]
                }
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
