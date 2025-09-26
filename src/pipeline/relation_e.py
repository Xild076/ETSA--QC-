import json
import re
import time
from typing import Dict, Any, Optional, List
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
                time.sleep(sleep_time)
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
        self._cache: Dict[tuple[str, tuple], Dict[str, Any]] = {}
        self._init_api(api_key)

    def _init_api(self, api_key: Optional[str] = None):
        if genai is None:
            raise RuntimeError("google-generativeai not available")
        key = api_key or get_env("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("No API key provided for GemmaRelationExtractor")
        genai.configure(api_key=key)

    def _create_prompt(self, sentence: str, entities: List[str]) -> str:
        ents = ", ".join(f'"{e}"' for e in entities)
        return (
            "<<SYS>>\n"
            "You are a meticulous linguistic analyst.\n"
            "Task: Given a passage and a closed set of entity surface forms, extract only structural relations that hold **between members of that set**.\n"
            "\n"
            "Rules:\n"
            "1) Relation Types (mutually exclusive):\n"
            "   • ACTION: An actor entity performs an action that affects a target entity. The action’s 'text' MUST be verbatim from the passage and include any negation or modal verbs (e.g., 'not fix', 'would not help').\n"
            "   • ASSOCIATION: Entities are linked by coordination/proximity (e.g., 'and', 'with', comma-delimited co-mentions) without directional predicate.\n"
            "   • BELONGING: A child entity is an attribute/part/possession of a parent entity (e.g., possessive/apostrophe, 'of').\n"
            "2) Strict Matching: Use **only** the exact entity strings provided (case-sensitive surface forms). Do not invent, normalize, split, merge, or alias entities.\n"
            "3) Directionality:\n"
            "   • ACTION: subject = initiator/actor; object = affected/target.\n"
            "   • BELONGING: subject = child/part; object = owner/whole.\n"
            "   • ASSOCIATION: subject/object order does not imply semantics; choose left-to-right as they appear.\n"
            "4) Evidence Text: The 'text' field must be a minimal, continuous span copied verbatim from the passage that signals the relation (e.g., the main predicate, coordinator, or possessive marker). No paraphrases.\n"
            "5) Pairing Scope: Consider only pairs where both items are in the entity list. Ignore entities not listed. Do not cross-sentence unless explicitly linked within the same sentence span.\n"
            "6) Deduplication: Emit at most one relation per (subject, object, type, text) combination.\n"
            "7) No Relation: If **no** valid relations exist between the provided entities, return an empty 'relations' array.\n"
            "8) Output Format: Return **strict JSON only** (no Markdown, no code fences). Conform exactly to the schema below. Do not include extra keys.\n"
            "\n"
            "Output Schema:\n"
            "{\n"
            '  "justification": "Brief evidence-based rationale for each included relation or an explanation for why none were found.",\n'
            '  "relations": [\n'
            '    {"subject": string, "object": string, "relation": {"type": "ACTION" | "ASSOCIATION" | "BELONGING", "text": string}}\n'
            "  ]\n"
            "}\n"
            "<</SYS>>\n\n"
            f"<<PASSAGE>>\n{sentence}\n<</PASSAGE>>\n\n"
            f"<<ENTITIES>>\n[{ents}]\n<</ENTITIES>>\n\n"
            "<<EXAMPLES>>\n"
            'P: tech support would not fix the problem. E: ["tech support", "problem"] -> {"justification":"Modal negation indicates failed action from actor to target.","relations":[{"subject":"tech support","object":"problem","relation":{"type":"ACTION","text":"would not fix"}}]}\n'
            'P: Food and wine arrived. E: ["food","wine"] -> {"justification":"Coordinated NP signals association.","relations":[{"subject":"food","object":"wine","relation":{"type":"ASSOCIATION","text":"and"}}]}\n'
            'P: The laptop\'s battery is great. E: ["laptop","battery"] -> {"justification":"Possessive marks part–whole.","relations":[{"subject":"battery","object":"laptop","relation":{"type":"BELONGING","text":"\'s"}}]}\n'
            'P: The computer is fast. E: ["computer"] -> {"justification":"Only one entity; no pairwise relation.","relations":[]}\n'
            "<</EXAMPLES>>\n\n"
            "<<RESPONSE>>\n"
            '{\n'
            '  "justification": "",\n'
            '  "relations": []\n'
            "}\n"
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
        return r.text

    def _clean_rel(self, r: Dict[str, Any], entities: List[str]) -> Optional[Dict[str, Any]]:
        if not isinstance(r, dict):
            return None
        sub = r.get("subject") or {}
        obj = r.get("object") or {}
        rel = r.get("relation") or {}
        sh = sub.get("head")
        oh = obj.get("head")
        t = rel.get("type")
        tx = rel.get("text", "")
        if not isinstance(sh, str) or not isinstance(oh, str) or not isinstance(t, str):
            return None
        sh_s = sh.strip()
        oh_s = oh.strip()
        t_s = t.strip().upper()
        if sh_s not in entities or oh_s not in entities:
            return None
        if t_s not in {"ACTION","ASSOCIATION","BELONGING"}:
            return None
        return {"subject":{"head":sh_s},"object":{"head":oh_s},"relation":{"type":t_s,"text":str(tx)}}

    def _parse_relation_response(self, text: str, entities: List[str]) -> Dict[str, Any]:
        data = _parse_json_from_text(text) or {}
        rels = data.get("relations") or []
        acts = data.get("actions") or []
        ascs = data.get("associations") or []
        bels = data.get("belongings") or []
        out_rels = []
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
        if not text or not entities:
            return {"entities": entities, "actions": [], "associations": [], "belongings": [], "relations": [], "justification": "empty input", "approach_used": self.model}
        # caching to avoid duplicate LLM queries
        cache_key = (text.strip(), tuple(e.strip() for e in entities))
        cached = self._cache.get(cache_key)
        if cached:
            return cached
        prompt = self._create_prompt(text, entities)
        last_err = None
        for i in range(self.retries + 1):
            try:
                resp = self._query_gemma(prompt)
                out = self._parse_relation_response(resp, entities)
                try:
                    # keep cache size bounded
                    if len(self._cache) > 10000:
                        # drop an arbitrary item (FIFO-like) to avoid unbounded growth
                        try:
                            self._cache.pop(next(iter(self._cache)))
                        except Exception:
                            self._cache.clear()
                    self._cache[cache_key] = out
                except Exception:
                    pass
                return out
            except Exception as e:
                last_err = e
                if i < self.retries:
                    time.sleep(self.backoff * (i + 1))
        return {"entities": entities, "actions": [], "associations": [], "belongings": [], "relations": [], "justification": f"failure: {last_err}", "approach_used": self.model}
