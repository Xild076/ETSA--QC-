import json
import re
import time
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
        key = api_key or get_env("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("No API key provided for GemmaModifierExtractor")
        genai.configure(api_key=key)

    def _prompt(self, passage: str, entity: str) -> str:
        return f"""<<SYS>>
    You are a meticulous linguist specializing in aspect-based sentiment extraction. Your sole task is to extract ALL and ONLY modifier phrases that describe the single TARGET ENTITY.

    Core Principle:
    - A "modifier" describes the quality, state, evaluation, or context of ONE entity.
    - Do NOT extract relations between TWO OR MORE entities.

    Definitions (what to extract, as verbatim spans from the passage):
    - Direct Descriptions: adjectives/adverbs and their intensifiers (e.g., "very slow", "great").
    - Negated & Nuanced Descriptions: include negation and hedges (e.g., "not good", "anything but fresh", "not the best").
    - Clause-Level Predication (CRITICAL when no direct modifier exists): the main predicate or copular/adjectival/verb phrase that conveys sentiment about the entity (e.g., "was dreadful", "works perfectly", "fell apart").
    - Idioms & Fixed Expressions: complete idiomatic spans that evaluate the entity (e.g., "melts in your mouth", "a total ripoff").
    - Recommendations & Directives about the entity: evaluative guidance (e.g., "worth trying", "avoid at all costs", "highly recommend").

    Boundary Rules:
    - Include only spans that attribute evaluative content to the TARGET ENTITY.
    - Exclude purely factual/type labels (e.g., "asian salad" → exclude "asian" unless it carries evaluation).
    - Exclude actions describing another agent’s interaction with the entity if they do not evaluate the entity itself (e.g., "the waiter brought the food" is NOT a modifier of "food").
    - Do not paraphrase; extract minimal contiguous spans verbatim that carry the evaluative meaning. Preserve negation/modality exactly.
    - Use strict surface-form matching for the ENTITY: treat the provided ENTITY string as the reference mention.
    - Single-sentence focus: prefer the sentence containing the ENTITY; include cross-sentence spans only if directly anaphoric and clearly evaluative of the ENTITY.
    - Deduplicate identical spans. Order results by first occurrence in the passage.
    - Include modifiers that may already appear in the ENTITY string if they carry evaluative content (e.g., ENTITY="the great food" → modifier="great").

    Neutrality & Emptiness:
    - If the ENTITY is mentioned only in a neutral/factual way with no evaluative content, return an empty "modifiers" list.

    Output Requirements:
    - Return STRICT JSON only (no Markdown, no code fences), conforming exactly to the schema below.
    - Justification must briefly explain how you located the spans, especially when using clause-level predication.

    Schema:
    {{
    "entity": "string",
    "justification": "string",
    "modifiers": ["string"],
    "approach_used": "{self.model}"
    }}
    <</SYS>>

    <<PASSAGE>>
    {passage}
    <</PASSAGE>>

    <<ENTITY>>
    {entity}
    <</ENTITY>>

    <<EXAMPLES>>
    P: Did not enjoy the new Windows 8. E: Windows 8 ->
    {{
    "entity":"Windows 8",
    "justification":"Clause-level predicate 'Did not enjoy' evaluates the entity with negation.",
    "modifiers":["Did not enjoy"],
    "approach_used":"{self.model}"
    }}

    P: The service was dreadful! E: service ->
    {{
    "entity":"service",
    "justification":"Copular predicate 'was dreadful' directly evaluates the entity.",
    "modifiers":["was dreadful"],
    "approach_used":"{self.model}"
    }}

    P: tech support would not fix the problem. E: tech support ->
    {{
    "entity":"tech support",
    "justification":"Predicate with modal+negation 'would not fix' evaluates the subject's adequacy.",
    "modifiers":["would not fix"],
    "approach_used":"{self.model}"
    }}

    P: The gnocchi literally melts in your mouth! E: gnocchi ->
    {{
    "entity":"gnocchi",
    "justification":"Idiomatic positive evaluation; extract full verb phrase.",
    "modifiers":["literally melts in your mouth"],
    "approach_used":"{self.model}"
    }}

    P: Entrees include lasagna. E: lasagna ->
    {{
    "entity":"lasagna",
    "justification":"Factual inclusion without evaluation.",
    "modifiers":[],
    "approach_used":"{self.model}"
    }}

    P: The food was tasty and large in portion size. E: portion size ->
    {{
    "entity":"portion size",
    "justification":"Direct descriptor 'large in' evaluates the aspect.",
    "modifiers":["large in"],
    "approach_used":"{self.model}"
    }}
    <</EXAMPLES>>

    <<RESPONSE>>
    {{
    "entity": "{entity}",
    "justification": "Explain how you identified the modifiers, especially if you used clause-level predication.",
    "modifiers": [],
    "approach_used": "{self.model}"
    }}
    """

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
        return resp.text

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
                return {
                    "entity": entity,
                    "modifiers": norm,
                    "approach_used": self.model,
                    "justification": data.get("justification", "")[:500]
                }
            except Exception as e:
                last_err = e
                if i < self.retries:
                    time.sleep(self.backoff * (i + 1))
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
