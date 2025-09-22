import json
import re
import time
import os
import random
from typing import Dict, Any, Optional, List, Union
import logging

try:
    import google.generativeai as genai
except Exception:
    genai = None

from src.utility import get_env

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class _RateLimiter:
    def __init__(self, max_calls: int = 45, period_seconds: int = 60):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.timestamps: List[float] = []
    def acquire(self):
        now = time.time()
        self.timestamps = [t for t in self.timestamps if now - t < self.period_seconds]
        if len(self.timestamps) >= self.max_calls:
            sleep_time = self.period_seconds - (now - self.timestamps[0])
            if sleep_time > 0:
                time.sleep(sleep_time + min(0.25, random.random() * 0.25))
        self.timestamps.append(time.time())

_GLOBAL_GENAI_LIMITER = None

def _ensure_global_rate_limiter(max_calls: int = 45, period_seconds: int = 60):
    global _GLOBAL_GENAI_LIMITER
    if _GLOBAL_GENAI_LIMITER is None:
        _GLOBAL_GENAI_LIMITER = _RateLimiter(max_calls=max_calls, period_seconds=period_seconds)
    return _GLOBAL_GENAI_LIMITER

def _default_rpm_for_model(model_name: str) -> int:
    name = (model_name or "").lower()
    if "gemma" in name:
        return 30
    if "2.5" in name and "pro" in name:
        return 5
    if "2.5" in name and "flash" in name:
        return 10
    if "2.0" in name and "flash" in name:
        return 15
    return 10

def _parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
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
    def extract(self, text: str, entities: List[str]) -> Dict[str, Any]:
        raise NotImplementedError

class DummyRelationExtractor(RelationExtractor):
    def extract(self, text: str, entities: List[str]) -> Dict[str, Any]:
        return {"entities": entities, "actions": [], "associations": [], "belongings": [], "relations": [], "approach_used": "dummy_extractor", "justification": ""}

class SpacyRelationExtractor(RelationExtractor):
    def extract(self, text: str, entities: List[str]) -> Dict[str, Any]:
        return {"entities": entities, "actions": [], "associations": [], "belongings": [], "relations": [], "approach_used": "spacy_fallback", "justification": "disabled"}

class GemmaRelationExtractor(RelationExtractor):
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
        response_mime_type: str = "text/plain",
        rate_limiter: Optional[_RateLimiter] = None,
        fallback_extractor: Optional[RelationExtractor] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        self.backoff = backoff
        self.retries = retries
        self.response_mime_type = response_mime_type
        rpm = _default_rpm_for_model(self.model)
        self.rate_limiter = rate_limiter or _ensure_global_rate_limiter(max_calls=rpm)
        self.fallback_extractor: Optional[RelationExtractor] = fallback_extractor or SpacyRelationExtractor()
        self._enabled = True
        self._init_error: Optional[Exception] = None
        try:
            self._init_api(api_key)
        except Exception as exc:
            self._enabled = False
            self._init_error = exc
            logger.info(
                "GemmaRelationExtractor disabled during init (%s). Using %s fallback instead.",
                exc,
                type(self.fallback_extractor).__name__ if self.fallback_extractor else "no",
            )

    def _init_api(self, api_key: Optional[str] = None):
        if genai is None:
            raise RuntimeError("google-generativeai not available")
        key = api_key or get_env("GOOGLE_API_KEY") or get_env("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("No API key provided for GemmaRelationExtractor")
        genai.configure(api_key=key)

    def _create_prompt(self, sentence: str, entities: List[str]) -> str:
        ents = ", ".join(f'"{e}"' for e in entities)
        return (
            "<<SYS>>\n"
            "Extract ONLY clear, unambiguous structural relations among the provided entities. Be very conservative.\n"
            "ACTION: Only when one entity clearly performs a specific action ON another entity (not just mentions or proximity).\n"
            "ASSOCIATION: Only when entities are explicitly linked by clear connecting words (and, with, together, etc) or direct comparison.\n"
            "BELONGING: Only when one entity is clearly a part, attribute, or direct possession of another (child OF parent).\n"
            "CRITICAL: Return empty relations array if connections are unclear, ambiguous, or entities are just mentioned in same sentence.\n"
            "Use exact entity strings. Return strict JSON.\n"
            "<</SYS>>\n\n"
            f"<<PASSAGE>>\n{sentence}\n<</PASSAGE>>\n\n"
            f"<<ENTITIES>>\n[{ents}]\n<</ENTITIES>>\n\n"
            "<<OUTPUT_SCHEMA>>\n"
            "{\n"
            '  "justification": "",\n'
            '  "relations": [\n'
            '    {"subject": string, "object": string, "relation": {"type": "ACTION", "text": string}},\n'
            '    {"subject": string, "object": string, "relation": {"type": "ASSOCIATION", "text": string}},\n'
            '    {"subject": string, "object": string, "relation": {"type": "BELONGING", "text": string}}\n'
            "  ],\n"
            "}\n"
            "<</OUTPUT_SCHEMA>>\n\n"
            "<<EXAMPLES>>\n"
            'P: The waiter served the customer well. E: ["waiter","customer"] -> ACTION {"subject":"waiter","object":"customer","relation":{"type":"ACTION","text":"served"}}\n'
            'P: Food and wine complement each other. E: ["food","wine"] -> ASSOCIATION {"subject":"food","object":"wine","relation":{"type":"ASSOCIATION","text":"and complement"}}\n'
            'P: The laptop screen is bright. E: ["screen","laptop"] -> BELONGING {"subject":"screen","object":"laptop","relation":{"type":"BELONGING","text":"of"}}\n'
            'P: I like the performance and the design. E: ["performance","design"] -> relations: []\n'
            'P: The service was slow but the food was good. E: ["service","food"] -> relations: []\n'
            "<</EXAMPLES>>\n\n"
            "<<RESPONSE>>\n"
            "```json\n"
            "{\n"
            '  "justification": "",\n'
            '  "relations": [],\n'
            "}\n"
            "```\n"
        )

    def _query_gemma(self, prompt: str) -> str:
        if self.rate_limiter:
            self.rate_limiter.acquire()
        model = genai.GenerativeModel(
            model_name=self.model,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_output_tokens=self.max_output_tokens,
                response_mime_type=self.response_mime_type,
            )
        )
        r = model.generate_content(prompt)
        text = getattr(r, "text", None)
        if not text:
            try:
                text = str(r)
            except Exception:
                text = ""
        return text

    def _clean_rel(self, r: Dict[str, Any], entities: List[str]) -> Optional[Dict[str, Any]]:
        if not isinstance(r, dict):
            return None
        sub_raw = r.get("subject")
        obj_raw = r.get("object")
        rel_raw = r.get("relation")
        if isinstance(sub_raw, dict):
            sh = sub_raw.get("head")
        else:
            sh = sub_raw
        if isinstance(obj_raw, dict):
            oh = obj_raw.get("head")
        else:
            oh = obj_raw
        if isinstance(rel_raw, dict):
            t = rel_raw.get("type")
            tx = rel_raw.get("text", "")
        else:
            t = rel_raw
            tx = ""
        if not isinstance(sh, str) or not isinstance(oh, str) or not isinstance(t, str):
            return None
        sh_s = sh.strip()
        oh_s = oh.strip()
        t_s = t.strip().upper()
        if sh_s not in entities or oh_s not in entities:
            return None
        if t_s not in {"ACTION","ASSOCIATION","BELONGING"}:
            return None
        if sh_s == oh_s:
            return None
        
        spurious_pairs = [
            ("performance", "computer"), ("performance", "laptop"),
            ("quality", "receiver"), ("quality", "product"),
            ("camera", "performance"), ("anything", "performance"),
            ("receiver", "performance"), ("superlatives", "quality"),
            ("I", "performance"), ("you", "performance"), ("my", "anything"),
            ("that", "performance"), ("They", "performance"),
            ("what", "performance"), ("time", "seconds"), ("time", "minute")
        ]
        
        for sp1, sp2 in spurious_pairs:
            if (sh_s.lower() == sp1.lower() and oh_s.lower() == sp2.lower()) or \
               (sh_s.lower() == sp2.lower() and oh_s.lower() == sp1.lower()):
                return None
        
        return {"subject": sh_s, "object": oh_s, "relation": {"type": t_s, "text": str(tx)}}

    def _parse_relation_response(self, text: str, entities: List[str]) -> List[Dict[str, Any]]:
        data = _parse_json_from_text(text) or {}
        rels = data.get("relations") or []
        acts = data.get("actions") or []
        ascs = data.get("associations") or []
        bels = data.get("belongings") or []
        out_rels: List[Dict[str, Any]] = []
        for r in rels:
            c = self._clean_rel(r, entities)
            if c:
                out_rels.append(c)
        for r in acts:
            c = self._clean_rel(r, entities)
            if c:
                out_rels.append(c)
        for r in ascs:
            c = self._clean_rel(r, entities)
            if c:
                out_rels.append(c)
        for r in bels:
            c = self._clean_rel(r, entities)
            if c:
                out_rels.append(c)
        return out_rels

    def _run_fallback(
        self,
        text: str,
        raw_entities: Union[List[str], List[Dict[str, Any]]],
        ent_heads: List[str],
    ) -> List[Dict[str, Any]]:
        if not self.fallback_extractor:
            return []
        try:
            fallback_output = self.fallback_extractor.extract(text, raw_entities)
        except Exception as fallback_exc:
            logger.debug("Relation fallback extractor failed: %s", fallback_exc)
            return []

        normalized: List[Dict[str, Any]] = []
        candidates: List[Any]
        if isinstance(fallback_output, list):
            candidates = fallback_output
        elif isinstance(fallback_output, dict):
            candidates = []
            for key in ("relations", "actions", "associations", "belongings"):
                vals = fallback_output.get(key)
                if vals:
                    candidates.extend(vals)
        else:
            candidates = []

        for candidate in candidates:
            cleaned = self._clean_rel(candidate, ent_heads)
            if cleaned:
                normalized.append(cleaned)
        return normalized

    def extract(self, text: str, entities: Union[List[str], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        # normalize entities to list of heads (strings)
        raw_entities = entities or []
        ent_heads: List[str] = []
        for e in raw_entities:
            if isinstance(e, str):
                ent_heads.append(e)
            elif isinstance(e, dict):
                h = e.get("head") or e.get("text") or e.get("entity")
                if isinstance(h, str):
                    ent_heads.append(h)
        ent_heads = list(dict.fromkeys([h.strip() for h in ent_heads if h and h.strip()]))
        if not text or not ent_heads:
            return []
        if not self._enabled:
            return self._run_fallback(text, raw_entities, ent_heads)
        prompt = self._create_prompt(text, ent_heads)
        last_err = None
        for i in range(self.retries + 1):
            try:
                resp = self._query_gemma(prompt)
                return self._parse_relation_response(resp, ent_heads)
            except Exception as e:
                last_err = e
                if i < self.retries:
                    sleep = self.backoff * (2 ** i)
                    sleep += min(0.25, random.random() * 0.25)
                    time.sleep(sleep)
        self._enabled = False
        logger.info(
            "Gemma relation extraction falling back after failure: %s",
            last_err,
        )
        return self._run_fallback(text, raw_entities, ent_heads)
