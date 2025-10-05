"""Relation extraction strategies for linking entities within clauses."""

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from . import config
    from .cache_utils import load_cache_from_file, save_cache_to_file
except ImportError:
    import config
    from cache_utils import load_cache_from_file, save_cache_to_file

try:
    from src.utility import get_env
except ImportError:
    try:
        from ..utility import get_env
    except ImportError:
        from utility import get_env

try:
    import google.generativeai as genai
except Exception:
    genai = None

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class _RateLimiter:
    """Simple sliding-window rate limiter for outbound API calls."""

    def __init__(self, max_calls: int = 45, period_seconds: int = 60):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.timestamps: List[float] = []

    def acquire(self):
        """Block until another call can be made within the current window."""
        now = time.time()
        self.timestamps = [t for t in self.timestamps if now - t < self.period_seconds]
        if len(self.timestamps) >= self.max_calls:
            sleep_time = self.period_seconds - (now - self.timestamps[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        self.timestamps.append(time.time())

_GLOBAL_GENAI_LIMITER = None

def _ensure_global_rate_limiter(max_calls: int = 45, period_seconds: int = 60):
    """Return a shared limiter keyed by model throughput."""
    global _GLOBAL_GENAI_LIMITER
    if _GLOBAL_GENAI_LIMITER is None:
        _GLOBAL_GENAI_LIMITER = _RateLimiter(max_calls=max_calls, period_seconds=period_seconds)
    return _GLOBAL_GENAI_LIMITER

def _default_rpm_for_model(model_name: str) -> int:
    """Derive a safe default RPM setting based on model name."""
    name = (model_name or "").lower()
    if "gemma-3-27b-it" in name:
        return 240
    if "gemini-1.5-flash" in name:
        return 120
    if "gemini-1.5-pro" in name:
        return 10
    return 30

def _parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object embedded in free-form text."""
    if not text:
        return None
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        s = m.group(1)
    else:
        m2 = re.search(r"\{[\s\S]*\}", text)
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

class RelationExtractor:
    """Abstract base class for extracting structural relations."""

    def extract(self, text: str, entities: List[str]) -> Dict[str, Any]:
        """Return relation payload linking the supplied ``entities``."""
        raise NotImplementedError


class DummyRelationExtractor(RelationExtractor):
    """Fallback extractor that always returns empty relations."""

    def extract(self, text: str, entities: List[str]) -> Dict[str, Any]:
        """Return an empty relation payload for the given ``entities``."""
        return {
            "entities": entities,
            "actions": [],
            "associations": [],
            "belongings": [],
            "relations": [],
            "approach_used": "dummy_extractor",
            "justification": "",
        }

class SpacyRelationExtractor(RelationExtractor):
    """Placeholder for a spaCy-based rule extractor (currently disabled)."""

    def extract(self, text: str, entities: List[str]) -> Dict[str, Any]:
        """Return a sentinel response indicating the fallback is disabled."""
        return {"entities": entities, "actions": [], "associations": [], "belongings": [], "relations": [], "approach_used": "spacy_fallback", "justification": "disabled"}


class GemmaRelationExtractor(RelationExtractor):
    """Gemini-powered relation extractor with caching and rate limiting."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemma-3-27b-it",
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 40,
        max_output_tokens: int = 1024,
        backoff: float = 0.8,
        retries: int = 2,
        response_mime_type: Optional[str] = None,
        rate_limiter: Optional[_RateLimiter] = None,
        cache_file: Optional[str] = None,
    ):
        """Configure the extractor with model, sampling, retry, and cache knobs."""
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        self.backoff = backoff
        self.retries = retries
        if response_mime_type is not None:
            self.response_mime_type = response_mime_type
        elif "gemini" in (self.model or ""):
            self.response_mime_type = "application/json"
        else:
            self.response_mime_type = None
        rpm = _default_rpm_for_model(self.model)
        self.rate_limiter = rate_limiter or _ensure_global_rate_limiter(max_calls=rpm)
        self.cache_file = cache_file or config.LLM_RELATION_CACHE
        if self.cache_file:
            self._cache = load_cache_from_file(self.cache_file)
        else:
            self._cache: Dict[Tuple[str, str, Tuple[str, ...]], Dict[str, Any]] = {}
        self._init_api(api_key)

    def _init_api(self, api_key: Optional[str] = None):
        """Initialise the Google Generative AI client."""
        if genai is None:
            raise RuntimeError("google-generativeai not available")
        key = api_key or get_env("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("No API key provided for GemmaRelationExtractor")
        genai.configure(api_key=key)

    def _create_prompt(self, sentence: str, entities: List[str]) -> str:
        """Create a structured instruction payload for Gemini/Gemma."""
        ents = ", ".join(f'"{e}"' for e in entities)
        return (
            "<<SYS>>\n"
            "You are a meticulous linguistic analyst. Your task is to extract structural relations **between members of a closed set of entities** from a given passage. Follow the rules strictly.\n"
            "\n"
            "## Rules:\n"
            "1.  **Relation Types (mutually exclusive)**:\n"
            "    -   `ACTION`: An actor entity performs an action affecting a target entity. The `text` MUST be a verbatim verb phrase from the passage, including:\n"
            "        - Manner adverbs (e.g., 'secretly gave', 'deceptively used')\n"
            "        - Negation/modals (e.g., 'would not fix', 'never helped')\n"
            "        - Sentiment-bearing objects when relevant (e.g., 'used expired flour' not just 'used')\n"
            "        - Shared predicates applied to multiple listed entities (emit one ACTION per subject→object pair)\n"
            "    -   `ASSOCIATION`: Entities are linked by coordination (`and`, `with`, commas) without a directional predicate.\n"
            "    -   `BELONGING`: A child entity is an attribute/part of a parent entity (e.g., `'s`, `of`, `had`).\n"
            "2.  **Strict Entity Matching**: Use **only** the exact entity strings provided in the <<ENTITIES>> list. Do not invent, normalize, or alias entities.\n"
            "3.  **Directionality**:\n"
            "    -   `ACTION`: `subject`=actor, `object`=target.\n"
            "    -   `BELONGING`: `subject`=part/child, `object`=owner/whole.\n"
            "    -   `ASSOCIATION`: `subject`/`object` order is left-to-right as they appear in the passage.\n"
            "4.  **Enumerations & Shared Predicates**: When a clause lists entities (comma/`and`) that share a predicate such as 'came with', 'were required', or 'was all that was needed', create:\n"
            "    -   `ASSOCIATION` links between the coordinated entities, and\n"
            "    -   `ACTION` relations from the controlling verb to each relevant entity.\n"
            "5.  **Comparatives, Negations & Absence**: Capture verbs expressing contrast, failure, or lack (e.g., 'was higher than', 'would not fix', 'has no', 'lacks', 'there was no change', 'costs more than') as ACTION relations. Preserve the negation/value tokens and, when the verb encodes absence, include the critical object (e.g., 'has no HDMI port', 'lacks dedicated graphics').\n"
            "6.  **No Relation**: If no valid relations exist between the provided entities, return an empty `relations` array.\n"
            "7.  **Output Format**: Return a single, valid JSON object with no markdown fences.\n"
            "\n"
            "## Output Schema:\n"
            "{\n"
            '  "justification": "Brief rationale for each relation or explanation for none.",\n'
            '  "relations": [\n'
            '    {"subject": string, "object": string, "relation": {"type": "ACTION" | "ASSOCIATION" | "BELONGING", "text": string}}\n'
            "  ]\n"
            "}\n"
            "<</SYS>>\n\n"
            "## Examples:\n"
            'P: "The kind Alistair secretly gave a box to the family." E: ["Alistair", "box", "family"] -> {"justification":"ACTION with manner adverb included.","relations":[{"subject":"Alistair","object":"family","relation":{"type":"ACTION","text":"secretly gave"}},{"subject":"Alistair","object":"box","relation":{"type":"ACTION","text":"gave"}}]}\n'
            'P: "Leonardo deceptively used cheap, expired flour." E: ["Leonardo", "expired flour"] -> {"justification":"ACTION with manner adverb.","relations":[{"subject":"Leonardo","object":"expired flour","relation":{"type":"ACTION","text":"deceptively used"}}]}\n'
            'P: "tech support would not fix the problem." E: ["tech support", "problem"] -> {"justification":"Action relation found.","relations":[{"subject":"tech support","object":"problem","relation":{"type":"ACTION","text":"would not fix"}}]}\n'
            'P: "The laptop has no HDMI port." E: ["laptop", "HDMI port"] -> {"justification":"Negated possession captured as action.","relations":[{"subject":"laptop","object":"HDMI port","relation":{"type":"ACTION","text":"has no HDMI port"}}]}\n'
            'P: "Food and wine arrived." E: ["food","wine"] -> {"justification":"Coordination implies association.","relations":[{"subject":"food","object":"wine","relation":{"type":"ASSOCIATION","text":"and"}}]}\n'
            'P: "The laptop\'s battery is great." E: ["laptop","battery"] -> {"justification":"Possessive marks part-whole.","relations":[{"subject":"battery","object":"laptop","relation":{"type":"BELONGING","text":"\'s"}}]}\n'
            'P: "WiFi capability, disk drive and multiple USB ports were all that was required." E: ["WiFi capability","disk drive","multiple USB ports"] -> {"justification":"Shared predicate handled.","relations":[{"subject":"WiFi capability","object":"disk drive","relation":{"type":"ASSOCIATION","text":","}},{"subject":"disk drive","object":"multiple USB ports","relation":{"type":"ASSOCIATION","text":"and"}},{"subject":"WiFi capability","object":"disk drive","relation":{"type":"ACTION","text":"were all that was required"}},{"subject":"WiFi capability","object":"multiple USB ports","relation":{"type":"ACTION","text":"were all that was required"}},{"subject":"disk drive","object":"multiple USB ports","relation":{"type":"ACTION","text":"were all that was required"}}]}\n'
            'P: "The computer is fast." E: ["computer"] -> {"justification":"Only one entity; no pairwise relation possible.","relations":[]}\n'
            "\n"
            "## Task:\n"
            f"<<PASSAGE>>\n{sentence}\n<</PASSAGE>>\n\n"
            f"<<ENTITIES>>\n[{ents}]\n<</ENTITIES>>\n\n"
            "<<RESPONSE>>\n"
        )

    def _query_gemma(self, prompt: str) -> str:
        """Invoke the LLM, respecting rate limits and model config."""
        if self.rate_limiter:
            self.rate_limiter.acquire()
        config_kwargs: Dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
        }
        if self.response_mime_type:
            config_kwargs["response_mime_type"] = self.response_mime_type
        model = genai.GenerativeModel(
            model_name=self.model,
            generation_config=genai.types.GenerationConfig(**config_kwargs)
        )
        r = model.generate_content(prompt)
        return r.text

    def _clean_rel(self, r: Dict[str, Any], entities: List[str]) -> Optional[Dict[str, Any]]:
        """Validate and normalise a single relation emitted by the LLM."""
        if not isinstance(r, dict):
            return None
        sub = r.get("subject")
        obj = r.get("object")
        rel = r.get("relation") or {}
        
        sub_text = sub if isinstance(sub, str) else (sub.get("head") if isinstance(sub, dict) else None)
        obj_text = obj if isinstance(obj, str) else (obj.get("head") if isinstance(obj, dict) else None)

        t = rel.get("type")
        tx = rel.get("text", "")
        if not isinstance(sub_text, str) or not isinstance(obj_text, str) or not isinstance(t, str):
            return None
        
        sh_s = sub_text.strip()
        oh_s = obj_text.strip()
        t_s = t.strip().upper()
        
        if sh_s not in entities or oh_s not in entities:
            return None
        if t_s not in {"ACTION","ASSOCIATION","BELONGING"}:
            return None
        return {"subject":{"head":sh_s},"object":{"head":oh_s},"relation":{"type":t_s,"text":str(tx)}}

    def _parse_relation_response(self, text: str, entities: List[str]) -> Dict[str, Any]:
        """Convert the raw LLM response into the structured pipeline schema."""
        data = _parse_json_from_text(text) or {}
        rels = data.get("relations") or []
        out_rels = []
        for r in rels:
            c = self._clean_rel(r, entities)
            if c:
                out_rels.append(c)

        actions = [r for r in out_rels if r["relation"]["type"]=="ACTION"]
        associations = [r for r in out_rels if r["relation"]["type"]=="ASSOCIATION"]
        belongings = [r for r in out_rels if r["relation"]["type"]=="BELONGING"]
        return {
            "entities": entities,
            "actions": actions,
            "associations": associations,
            "belongings": belongings,
            "relations": out_rels,
            "justification": data.get("justification",""),
            "approach_used": self.model
        }

    def extract(self, text: str, entities: List[str]) -> Dict[str, Any]:
        """Return cached or freshly generated relations for ``text`` and ``entities``."""
        if not text or not entities:
            return {"entities": entities, "actions": [], "associations": [], "belongings": [], "relations": [], "justification": "empty input", "approach_used": self.model}

        cache_key = (
            self.model,
            text.strip(),
            tuple(sorted(e.strip() for e in entities)),
        )
        cached = self._cache.get(cache_key)
        if cached:
            return cached
        
        prompt = self._create_prompt(text, entities)
        last_err = None
        for i in range(self.retries + 1):
            try:
                resp = self._query_gemma(prompt)
                out = self._parse_relation_response(resp, entities)
                if len(self._cache) > 10000:
                    try:
                        self._cache.pop(next(iter(self._cache)))
                    except Exception:
                        self._cache.clear()
                self._cache[cache_key] = out
                if self.cache_file:
                    save_cache_to_file(self.cache_file, self._cache)
                return out
            except Exception as e:
                last_err = e
                if i < self.retries:
                    time.sleep(self.backoff * (i + 1))
        return {"entities": entities, "actions": [], "associations": [], "belongings": [], "relations": [], "justification": f"failure: {last_err}", "approach_used": self.model}
