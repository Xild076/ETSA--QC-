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
You are a meticulous linguist specializing in **aspect-based sentiment extraction**. Your sole task is to extract **ALL and ONLY modifier phrases** that evaluate (positively, negatively, or mixed) the single **TARGET ENTITY**.

## Objective
Given a short PASSAGE and a single ENTITY string, return **verbatim spans** from the passage that **evaluate the ENTITY**. Return **strict JSON** using the schema at the end—no prose, no Markdown.

## What Counts as a “Modifier” (extract these verbatim)
1.  **Direct descriptions** of the ENTITY: adjectives/adverbs **with their intensifiers/degree markers/negation**
    - e.g., “very slow”, “not good”, “anything but fresh”, “really well made”.
2.  **Clause-level predication about the ENTITY** (CRITICAL):
    Extract the **main predicate or copular/adjectival/verb phrase** that **as a whole** conveys evaluation, including **auxiliaries, modals, negation, hedges, quantifiers, adverbs**, and **light complements** required for meaning.
    - e.g., “was dreadful”, “works perfectly”, “should be a bit more friendly”, “wasn’t it”, “barely eatable”, “melts in your mouth”.
3.  **Comparative/expectational criticism** where the **entire clause expresses evaluation** of the ENTITY (even if the explicit adjective is neutral):
    - e.g., “is something I can make better at home”, “didn’t live up to the hype”, “(not) on par with other Japanese restaurants”, “could be better than most places” (keep negation/hedges).
4.  **Idioms & fixed expressions** that **evaluate** the ENTITY:
    - e.g., “a total ripoff”, “melts in your mouth”.
5.  **Recommendations & directives** **about the ENTITY**:
    - e.g., “worth trying”, “avoid at all costs”, “highly recommend”.

## Boundary & Span Rules
- **Verbatim**: Extract **exact surface spans** from the passage; no paraphrase.
- **Minimal-but-complete**: Include **all words that carry the evaluation** (negation, modals, hedges, intensifiers, necessary complements) but **exclude** trailing context not needed for evaluativity.
- **Clause-first principle**:
  If the opinion is expressed by a **clause/predicate**, extract the **entire predicate phrase**, not just its head adjective/verb.
  - Example: “should be a bit more friendly” (not just “friendly”).
- **Greedy-until-boundary for coordination**:
  When coordinated predicates or adjectives share the **same subject (the ENTITY)**, extract the **full coordinated span** (e.g., “was tasteless and burned”, “ALWAYS look angry and even ignore their high-tipping regulars”), **up to the first boundary** where the subject or scope clearly shifts.
- **Comparatives/expectations**:
  Include the **polarity carrier** (negation/unsatisfied expectation) and the **standard of comparison** if it is **syntactically necessary** to convey the evaluative meaning (e.g., “(not) on par with other Japanese restaurants”, “is something I can make better at home”).
- **Anaphora**:
  If the ENTITY is referenced by a **coreferent pronoun or description in the same or adjacent sentence**, and that phrase **clearly predicates about the ENTITY**, you may extract it (e.g., ENTITY=“food”; span from “it … wasn’t on par …”).
- **Deduplicate** exact repeats; order spans by **first occurrence**.

## Decision Procedure (apply in order)
1.  **Locate all mentions** of the ENTITY (exact string) and anaphoric references (“it”, “this”).
2.  **Collect candidate predicates/modifiers** that **predicate of** that mention (copular, verbal, coordinated, comparative).
3.  **Build the minimal-but-complete evaluative span**, including negation, modals, hedges, and coordinated parts.
4.  **Filter out** factual or non-evaluative candidates.
5.  **Sort, deduplicate, and output.**

## Edge-Case Clarifications
- **Identity/Authenticity criticism** (“was not a Nicoise salad” or “this wasn’t it”): extract the **negated identificational predicate** (e.g., “was not a”, “wasn’t it”).
- **Expectation statements** (“you’d expect it to be at least on par …”): If the context implies the expectation is unmet for the ENTITY, extract the evaluative predicate that captures the shortfall (e.g., “didn’t live up to the hype”, “(not) on par with other Japanese restaurants”).
- **Comparatives to self/home** (“I can make better at home”): Treat as negative evaluation; include the **full comparative clause**.

## Output Requirements
- **Strict JSON only** using the schema below (no Markdown, no code fences).
- The **“justification”** must briefly state how spans were located based on the rules.
- Use `"approach_used": "{self.model}"`.

### Schema
{{{{
  "entity": "string",
  "justification": "string",
  "modifiers": ["string"],
  "approach_used": "{self.model}"
}}}}
<</SYS>>

<<PASSAGE>>
{passage}
<</PASSAGE>>

<<ENTITY>>
{entity}
<</ENTITY>>

<<EXAMPLES>>
P: The staff should be a bit more friendly. E: staff ->
{{{{
  "entity": "staff",
  "justification": "Extracted the full clause-level predicate containing a modal ('should be') and a hedge ('a bit more') which evaluates the entity.",
  "modifiers": ["should be a bit more friendly"],
  "approach_used": "{self.model}"
}}}}

P: The fajita we tried was tasteless and burned. E: fajita ->
{{{{
  "entity": "fajita",
  "justification": "Extracted the full coordinated predicate ('was tasteless and burned') that shares the same subject, as per the greedy-until-boundary rule.",
  "modifiers": ["was tasteless and burned"],
  "approach_used": "{self.model}"
}}}}

P: I know real Indian food and this wasn't it. E: this ->
{{{{
  "entity": "this",
  "justification": "The negated copular predicate ('wasn't it') functions as a strong negative evaluation of the entity's authenticity and quality.",
  "modifiers": ["wasn't it"],
  "approach_used": "{self.model}"
}}}}

P: Frankly, the chinese food here is something I can make better at home. E: chinese food ->
{{{{
  "entity": "chinese food",
  "justification": "The entire comparative clause ('is something I can make better at home') serves as a complete negative evaluation.",
  "modifiers": ["is something I can make better at home"],
  "approach_used": "{self.model}"
}}}}

P: Entrees include lasagna, which is a baked pasta dish. E: lasagna ->
{{{{
  "entity": "lasagna",
  "justification": "The passage provides only a factual, non-evaluative mention of the entity. No evaluative modifiers found.",
  "modifiers": [],
  "approach_used": "{self.model}"
}}}}
<</EXAMPLES>>

<<RESPONSE>>
{{{{
  "entity": "{entity}",
  "justification": "Explain how you identified the modifiers based on the rules, especially clause-level predication, coordination, or comparatives.",
  "modifiers": [Insert any modifiers here. If none found, return an empty list.],
  "approach_used": "{self.model}"
}}}}
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
