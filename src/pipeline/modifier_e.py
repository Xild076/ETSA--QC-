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
You are a meticulous linguist specializing in **aspect-based sentiment extraction**. Recover **every evaluative span** (positive, negative, or mixed) that targets the **single ENTITY** while ignoring non-opinion text. Return **strict JSON** only.

## Objective
Given a PASSAGE and an ENTITY, output all **verbatim spans** whose semantics evaluate that entity. Preserve mixed polarity by returning each evaluative clause separately.

## Extraction Directives
1.  **Clause-level evaluations (CRITICAL)** — capture the full predicate (auxiliaries, modals, negation, intensifiers, complements) that judges the entity.
2.  **Descriptive phrases** — include adjectives/adverbs with their intensifiers, hedges, and negators that modify the entity mention.
3.  **Comparatives & expectations** — keep the entire construction that signals better/worse-than or failed expectations (e.g., “didn’t live up to the hype”, “on par with …”).
4.  **Idioms, appositives, recommendations** — extract opinionated idioms or directives about the entity (e.g., “a total ripoff”, “worth trying”).
5.  **Value & price framing** — retain the evaluative wording around costs or trade-offs (e.g., “especially considering the $350 price tag”, “worth every penny”); never isolate the number alone.
6.  **Mixed polarity** — when the passage alternates praise and criticism of the same entity, emit **each** evaluative span (e.g., a negative clause followed by a positive clause). Never drop a counter-balancing statement.

## Boundary & Span Rules
- **Verbatim & minimal-but-complete**: include negation, auxiliaries, degree markers, quantifiers, and required complements; exclude trailing factual context unrelated to sentiment.
- **Shared subjects / coordination**: when coordinated predicates share the entity, capture the full coordinated span until the subject/scope shifts.
- **Numbers & measurements**: only keep numeric expressions when they directly participate in an evaluative construction (“boots in under 30 seconds”). Do **not** output bare measurements alone; keep the praising or complaining cue (e.g., “only”, “at least”, “worth the”).
- **Coreference allowed**: pronouns or descriptive rementions (“it”, “the dessert”) qualify if they clearly refer to the ENTITY.
- **Deduplicate** exact repeats and keep spans in passage order.

## Decision Procedure
1. Map each surface or coreferent mention of the ENTITY.
2. Gather predicates/modifiers that ascribe sentiment to that mention.
3. Expand to the minimal evaluative span with all polarity cues.
4. Remove factual descriptions with no opinion content.
5. Sort by first appearance and output.

## Edge Cases
- **Identity/authenticity complaints**: capture the negated identificational predicate (“wasn’t it”, “was not a Nicoise salad”).
- **Expectation framing**: when expectations fail for the ENTITY, keep the clause signalling the shortfall (“didn’t live up …”, “should be …”).
- **Comparatives to self/home**: treat as negative; keep the full clause (“is something I can make better at home”).
- **Double-negative praise**: clauses like “hard to find something I don’t like”, “can’t complain”, or “nothing to hate” signal positivity—capture them as favorable sentiment on the ENTITY.
- **Value justifications**: when sentiment hinges on price/value trade-offs (“especially considering …”, “worth the price”), include the motivating phrase with its cue words.
- **Opposing clauses**: when praise and criticism occur back-to-back, output both spans.

## Output Requirements
- Produce **strict JSON only** with the schema below (no Markdown, no commentary).
- The "justification" must briefly cite how the rules located the spans (mention clause, coordination, negation, etc.).
- Do **not** leave placeholders—replace them with actual spans, or [] when none exist.
- Use "approach_used": "{self.model}".

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
  "justification": "Captured the full clause-level predicate with modal + hedge that evaluates the staff.",
  "modifiers": ["should be a bit more friendly"],
  "approach_used": "{self.model}"
}}}}

P: The fajita we tried was tasteless and burned. E: fajita ->
{{{{
  "entity": "fajita",
  "justification": "Included the coordinated predicate that jointly evaluates the fajita.",
  "modifiers": ["was tasteless and burned"],
  "approach_used": "{self.model}"
}}}}

P: Certainly not the best sushi in New York, however, it is always fresh. E: sushi ->
{{{{
  "entity": "sushi",
  "justification": "Negative and positive clauses both target the sushi, so both spans are returned in order of appearance.",
  "modifiers": ["Certainly not the best sushi", "it is always fresh"],
  "approach_used": "{self.model}"
}}}}

P: Boot time is super fast, around anywhere from 35 seconds to 1 minute. E: Boot time ->
{{{{
  "entity": "Boot time",
  "justification": "Only the evaluative predicate is emitted; the numeric range alone is factual so it is excluded.",
  "modifiers": ["is super fast"],
  "approach_used": "{self.model}"
}}}}

P: I know real Indian food and this wasn't it. E: this ->
{{{{
  "entity": "this",
  "justification": "The negated copular predicate conveys a strong negative authenticity judgement.",
  "modifiers": ["wasn't it"],
  "approach_used": "{self.model}"
}}}}
<</EXAMPLES>>

<<RESPONSE>>
{{{{
  "entity": "{entity}",
  "justification": "Explain (<=200 chars) how the clause/polarity rules selected the spans.",
  "modifiers": [List each extracted modifier span in order; use [] only if none apply.],
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
