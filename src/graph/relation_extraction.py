import json
import os
from typing import List, Dict, Any, Tuple
from datetime import datetime
import time
import threading
import random
import google.generativeai as genai
try:
    from google.api_core import exceptions as gapi_exceptions
except Exception:
    gapi_exceptions = None
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass
API_KEY = os.getenv("GOOGLE_API_KEY")

GENAI_REQUESTS_PER_MIN = float(os.getenv("GENAI_REQUESTS_PER_MIN", "15"))
GENAI_MAX_RETRIES = int(os.getenv("GENAI_MAX_RETRIES", "6"))
GENAI_BASE_BACKOFF_SECONDS = float(os.getenv("GENAI_BASE_BACKOFF_SECONDS", "1.5"))
GENAI_MAX_BACKOFF_SECONDS = float(os.getenv("GENAI_MAX_BACKOFF_SECONDS", "30"))

import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class _RateLimiter:

    def __init__(self, rpm: float):
        self._interval = 60.0 / max(rpm, 0.001)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    @property
    def interval(self) -> float:
        return self._interval

    def acquire(self):
        now = time.monotonic()
        with self._lock:
            wait = max(0.0, self._next_allowed - now)
            if wait > 0:
                time.sleep(wait)
            now2 = time.monotonic()
            self._next_allowed = max(self._next_allowed, now2) + self._interval


_GLOBAL_GENAI_LIMITER = _RateLimiter(GENAI_REQUESTS_PER_MIN)

_EXTRACTOR_SINGLETON = None
_SPACY_SINGLETON = None

def _get_extractor(api_key: str | None = None):
    global _EXTRACTOR_SINGLETON
    if _EXTRACTOR_SINGLETON is None:
        _EXTRACTOR_SINGLETON = GemmaRelationExtractor(api_key=api_key)
    return _EXTRACTOR_SINGLETON

class GemmaRelationExtractor:
    def __init__(self, api_key: str = None, rate_limiter: _RateLimiter | None = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemma-3-27b-it")
        self._rate_limiter = rate_limiter or _GLOBAL_GENAI_LIMITER
        
    def extract_relations(self, sentence: str, entities: List[str]) -> Dict[str, Any]:
        if not sentence or not entities:
            return {
                "sentence": sentence,
                "entities": entities,
                "relations": [],
                "approach_used": "gemma_27b_empty_input",
                "error": "Empty sentence or entities provided",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": "Xild076"
            }
        
        prompt = self._create_prompt(sentence, entities)
        
        try:
            response = self._query_gemma(prompt)
            result = self._parse_response(response, sentence, entities)
            result["approach_used"] = "gemma-3-27b-it"
            result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result["user"] = "Xild076"
            
            result = self._post_process_relations(result)
            
            return result
            
        except Exception as e:
            print(f"Gemma extraction failed: {e}")
            return {
                "sentence": sentence,
                "entities": entities,
                "relations": [],
                "approach_used": "gemma_27b_failed",
                "error": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": "Xild076"
            }

    def _post_process_relations(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if "relations" not in result:
            return result
            
        cleaned_relations = []
        seen_relations = set()
        
        for relation in result["relations"]:
            if not self._is_valid_relation(relation):
                continue
                
            signature = (
                relation["subject"]["head"].lower(),
                relation["relation"]["type"].lower(),
                relation["relation"]["text"].lower(),
                relation["object"]["head"].lower()
            )
            
            if signature not in seen_relations:
                cleaned_relations.append(relation)
                seen_relations.add(signature)
        
        result["relations"] = cleaned_relations
        return result

    def _is_valid_relation(self, relation: Dict[str, Any]) -> bool:
        try:
            return (
                "subject" in relation and "head" in relation["subject"] and relation["subject"]["head"] and
                "object" in relation and "head" in relation["object"] and relation["object"]["head"] and
                "relation" in relation and "type" in relation["relation"] and relation["relation"]["type"] and
                relation["subject"]["head"] != relation["object"]["head"]
            )
        except (KeyError, TypeError):
            return False
    def _create_prompt(self, sentence: str, entities: List[str]) -> str:
        entities_json = json.dumps(entities)
        return (
            f"You are an expert relation extraction system for restaurant reviews. Extract ONLY relationships between the listed ENTITIES.\n\n"
            f"SENTENCE: \"{sentence}\"\n"
            f"ENTITIES: {entities_json}\n\n"
            "DOMAIN\n"
            "- Entities include food/drinks, staff, ambience/place, price/value.\n"
            "- Context: behaviors like chatting/giggling by staff while not serving imply negative service. Ignoring/forgetting/rushing by staff is negative. Serving/helping/bringing/taking orders is positive service.\n"
            "- Implications: waiting is negative for customers; slow service negative; fast/quick service positive; fresh food positive; cold-when-hot food negative.\n\n"
            "TYPES\n"
            "1) action: Directed verb phrase; include required particles/preps/adverbs; exclude objects and of/with-NP complements. Capture implied sentiment when staff behave socially instead of serving.\n"
            "2) association: Joint activity (work/collaborate/partner/play/share). Emit once per pair; not for location/path.\n"
            "3) belonging: Ownership/part-of (X has Y; Y of X; NP with NP as attributes). Not for mere spatial relations.\n\n"
            "ALGO\n"
            "- For all ordered pairs (A,B) in ENTITIES: add belonging if possessive/of/with-possession; add association if joint-activity verb with with/together/alongside; add action A→B if directed verb phrase.\n\n"
            "DECISIONS\n"
            "- Use only ENTITIES, exact heads. No subject==object. Reject preposition-only relation texts (of/with/from/to/...). Association needs agentive participants. Ignore pure location/path attachments.\n\n"
            "OUTPUT FORMAT (JSON only):\n"
            "{\n"
            f"    \"sentence\": \"{sentence}\",\n"
            f"    \"entities\": {entities_json},\n"
            "    \"relations\": [\n"
            "        {\n"
            "            \"subject\": { \"head\": \"entity_name\" },\n"
            "            \"relation\": { \"type\": \"action|association|belonging\", \"text\": \"relationship_description\" },\n"
            "            \"object\": { \"head\": \"entity_name\" }\n"
            "        }\n"
            "    ]\n"
            "}\n\n"
            "EXAMPLES\n"
            "- \"The biologist carefully cataloged rare specimens\" → biologist action \"carefully cataloged\" specimens.\n"
            "- \"The programmer collaborated with the designer on the interface\" → programmer association \"collaborated with\" designer.\n"
            "- \"A flock of birds crossed the valley\" → flock belonging \"of\" birds; flock action \"crossed\" valley.\n\n"
            "RESTAURANT\n"
            "- \"The waiter was chatting with other staff instead of serving us\" → waiter action \"was chatting with\" staff (negative); waiter action \"ignoring\" customers (implied).\n"
            "- \"Staff were giggling at the bar\" → staff action \"were giggling\" bar (negative).\n"
            "- \"The server quickly brought our food\" → server action \"quickly brought\" food (positive).\n\n"
            "EDGE\n"
            "- Do not create relations where the only connector is a path/location preposition unless covered by action verb phrases.\n"
            "- Do not include objects or of/with-NP complements inside action relation text.\n"
            "- If both association and action to a third object are present, include both; but emit association only once per pair.\n\n"
            "IMPORTANT\n"
            "- Use lowercase relation types. Use double quotes in JSON. Return valid JSON only."
        )

    def _query_gemma(self, prompt: str) -> str:
        generation_config = genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=800,
            top_p=0.6,
            top_k=20
        )

        def _is_rate_limited_error(err: Exception) -> bool:
            s = str(err).lower()
            if "429" in s or "rate" in s or "quota" in s or "exceed" in s or "resource exhausted" in s:
                return True
            if gapi_exceptions is not None:
                if isinstance(err, getattr(gapi_exceptions, "ResourceExhausted", tuple())):
                    return True
                if isinstance(err, getattr(gapi_exceptions, "TooManyRequests", tuple())):
                    return True
                if isinstance(err, getattr(gapi_exceptions, "DeadlineExceeded", tuple())):
                    return True
                if isinstance(err, getattr(gapi_exceptions, "ServiceUnavailable", tuple())):
                    return True
            return False

        attempts = 0
        last_err = None
        while attempts < GENAI_MAX_RETRIES:
            attempts += 1
            try:
                self._rate_limiter.acquire()
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                if not response.text or response.text.strip() == "":
                    raise ValueError("Empty response from Gemma model")
                return response.text
            except Exception as e:
                last_err = e
                is_rate = _is_rate_limited_error(e)
                if attempts >= GENAI_MAX_RETRIES or not is_rate:
                    logger.error(f"Gemma query failed (attempt {attempts}/{GENAI_MAX_RETRIES}): {e}")
                    raise
                base = max(GENAI_BASE_BACKOFF_SECONDS, self._rate_limiter.interval)
                delay = min(GENAI_MAX_BACKOFF_SECONDS, base * (2 ** (attempts - 1)))
                jitter = random.uniform(0.75, 1.25)
                sleep_s = delay * jitter
                logger.warning(f"Rate limit encountered; backing off for {sleep_s:.2f}s (attempt {attempts}/{GENAI_MAX_RETRIES})")
                time.sleep(sleep_s)
                continue

        if last_err:
            raise last_err
        raise RuntimeError("Gemma query failed without an explicit error")

    def _parse_response(self, response: str, sentence: str, entities: List[str]) -> Dict[str, Any]:
        try:
            response_clean = response.strip()
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            elif response_clean.startswith("```"):
                response_clean = response_clean[3:]
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]
            response_clean = response_clean.strip()
            json_start = response_clean.find('{')
            json_end = response_clean.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                response_clean = response_clean[json_start:json_end]
            response_clean = self._pre_sanitize_malformed_tokens(response_clean)
            response_clean = self._fix_json_quotes(response_clean)
            result = json.loads(response_clean)
            if "relations" not in result:
                result["relations"] = []
            result["sentence"] = sentence
            result["entities"] = entities
            for relation in result["relations"]:
                if "subject" not in relation:
                    relation["subject"] = {"head": ""}
                if "object" not in relation:
                    relation["object"] = {"head": ""}
                if "relation" not in relation:
                    relation["relation"] = {"type": "unknown", "text": ""}
                if "type" in relation["relation"]:
                    relation["relation"]["type"] = relation["relation"]["type"].lower()
            return result
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"Cleaned response: {response_clean[:200]}...")
            fallback_result = self._fallback_parse(response, sentence, entities)
            return fallback_result

    def _fix_json_quotes(self, json_str: str) -> str:
        import re
        protected = re.sub(r"(\w)'(\w)", r"\1__APOSTROPHE__\2", json_str)
        protected = re.sub(r"(\w)'s\b", r"\1__POSSESSIVE__", protected)
        fixed = protected.replace("'", '"')
        fixed = fixed.replace("__APOSTROPHE__", "'")
        fixed = fixed.replace("__POSSESSIVE__", "'s")
        return fixed

    def _pre_sanitize_malformed_tokens(self, text: str) -> str:
        import re
        s = text
        s = s.replace('[""s"]', '["\'s"]')
        s = s.replace(', ""s"', ', "\'s"')
        s = s.replace('""s"', "'s\"")
        s = re.sub(r"\[\s*\"\"\s*,", "[", s)
        s = re.sub(r",\s*\"\"\s*\]", "]", s)
        s = s.replace(", ,", ",")
        return s

    def _fallback_parse(self, response: str, sentence: str, entities: List[str]) -> Dict[str, Any]:
        relations = []
        response_lower = response.lower()
        sentence_lower = sentence.lower()
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i != j:
                    entity1_lower = entity1.lower()
                    entity2_lower = entity2.lower()
                    if (entity1_lower in response_lower or entity1_lower in sentence_lower) and \
                       (entity2_lower in response_lower or entity2_lower in sentence_lower):
                        relation = self._extract_relation_from_text(sentence, entity1, entity2, response)
                        if relation and not self._is_duplicate_relation(relations, relation):
                            relations.append(relation)
        return {
            "sentence": sentence,
            "entities": entities,
            "relations": relations,
            "fallback_used": True
        }

    def _extract_relation_from_text(self, sentence: str, entity1: str, entity2: str, response: str) -> Dict[str, Any]:
        sentence_lower = sentence.lower()
        response_lower = response.lower()
        
        entity1_pos = sentence_lower.find(entity1.lower())
        entity2_pos = sentence_lower.find(entity2.lower())
        
        if entity1_pos == -1 or entity2_pos == -1:
            return None
        
        rel_type = "action"
        rel_text = "related_to"
        
        if ("'s" in sentence_lower or " of " in sentence_lower or 
            any(word in response_lower for word in ['belonging', 'owns', 'has', 'possessive'])):
            rel_type = "belonging"
            if entity1_pos < entity2_pos:
                rel_text = "owns" if "'s" in sentence_lower else "has"
            else:
                rel_text = "belongs_to"
        
        elif (any(word in sentence_lower for word in [' and ', ' with ', 'together', 'collaborated']) or
              any(word in response_lower for word in ['association', 'together', 'collaborated'])):
            rel_type = "association"
            rel_text = "associated_with"
        
        elif entity1_pos < entity2_pos:
            between_text = sentence_lower[entity1_pos + len(entity1):entity2_pos]
            action_verbs = ['hit', 'helped', 'crashed', 'scolded', 'contains', 'displayed', 'played']
            for verb in action_verbs:
                if verb in between_text:
                    rel_text = verb
                    break
        
        return {
            "subject": {"head": entity1},
            "relation": {"type": rel_type, "text": rel_text},
            "object": {"head": entity2}
        }

    def _is_duplicate_relation(self, relations: List[Dict], new_relation: Dict) -> bool:
        for existing in relations:
            if (existing["subject"]["head"] == new_relation["subject"]["head"] and
                existing["object"]["head"] == new_relation["object"]["head"] and
                existing["relation"]["type"] == new_relation["relation"]["type"]):
                return True
        return False

def extract_entity_modifiers(sentence: str, entity: str) -> List[str]:
    try:
        extractor = _get_extractor(API_KEY)
        prompt = f"""You are an expert at extracting descriptive modifiers for entities. 

SENTENCE: "{sentence}"
TARGET ENTITY: "{entity}"

Extract ONLY meaningful descriptive modifiers (adjectives, colors, sizes, states, conditions, emotions) that describe the target entity. Ensure that all modifiers are listed separately, however, if there is a modifier to the modifier like an adverb, keep them together.

DO NOT include:
- Articles (a, an, the)
- Pronouns (his, her, its)
- Generic words
- Empty strings

Examples:
- "The angry red dog barked" → entity: dog → ["angry", "red"]
- "John's expensive car is fast" → entity: car → ["expensive", "fast"] 
- "The child was very sad" → entity: child → ["very sad"]
- "The man hit the child" → entity: man → []
- "A big house" → entity: house → ["big"]

Return ONLY a JSON array of meaningful modifier strings:
["modifier1", "modifier2", ...]

If no meaningful modifiers found, return: []"""

        response = extractor._query_gemma(prompt)
        
        response_clean = response.strip()
        if response_clean.startswith("```json"):
            response_clean = response_clean[7:]
        elif response_clean.startswith("```"):
            response_clean = response_clean[3:]
        if response_clean.endswith("```"):
            response_clean = response_clean[:-3]
        response_clean = response_clean.strip()
        
        start = response_clean.find('[')
        end = response_clean.rfind(']') + 1
        if start != -1 and end > start:
            json_str = response_clean[start:end]
            import json
            json_str = extractor._pre_sanitize_malformed_tokens(json_str)
            modifiers = json.loads(json_str)
            filtered_modifiers = []
            skip_words = {'', 'the', 'a', 'an', 'his', 'her', 'its', 'their', 'my', 'your', 'our', "'s"}
            for mod in modifiers:
                if not isinstance(mod, str):
                    continue
                mm = mod.strip().strip(',.;:!?').strip('"').strip()
                if mm.lower() in skip_words or not mm:
                    continue
                filtered_modifiers.append(mm)
            seen = set()
            uniq = []
            for m in filtered_modifiers:
                ml = m.lower()
                if ml not in seen:
                    seen.add(ml)
                    uniq.append(m)
            return uniq
        
        return []
    except Exception as e:
        logger.error(f"Error extracting modifiers with LLM for '{entity}': {e}")
        return []

def extract_entity_modifiers_spacy(sentence: str, entity: str) -> List[str]:
    try:
        global _SPACY_SINGLETON
        if _SPACY_SINGLETON is None:
            import spacy
            _SPACY_SINGLETON = spacy.blank("en") if not spacy.util.is_package("en_core_web_sm") else spacy.load("en_core_web_sm")
        doc = _SPACY_SINGLETON(sentence)
        mods: List[str] = []
        ent_l = entity.lower()

        def add_phrase(tok) -> None:
            phrase = tok.text
            advs = [c.text for c in tok.children if c.dep_ in ("advmod", "neg")]
            if advs:
                phrase = " ".join(advs + [phrase])
            mods.append(phrase)

        spans = list(doc.noun_chunks)
        for np in spans:
            if np.root.text.lower() == ent_l or np.text.lower() == ent_l:
                for tok in np.root.children:
                    if tok.dep_ == "amod":
                        add_phrase(tok)

        for tok in doc:
            if tok.dep_ in ("attr", "acomp") and tok.head.pos_ in ("AUX", "VERB"):
                nsubj = next((c for c in tok.head.children if c.dep_ in ("nsubj","nsubjpass") and c.text.lower() == ent_l), None)
                if nsubj is not None:
                    add_phrase(tok)

        skip = {"the","a","an","of","and","or","to","very","really"}
        out = []
        seen = set()
        for m in mods:
            mm = m.strip()
            if not mm or mm.lower() in skip:
                continue
            key = mm.lower()
            if key not in seen:
                seen.add(key)
                out.append(mm)
        return out
    except Exception:
        return []

def context_window(sentences: List[str], idx: int, width: int = 1) -> str:
    start = max(0, idx - width)
    end = min(len(sentences), idx + width + 1)
    return " ".join(s.strip() for s in sentences[start:end] if s)

def get_entity_modifiers(sentence: str, entity: str) -> List[str]:
    mods = extract_entity_modifiers_spacy(sentence, entity)
    if mods:
        return mods
    return extract_entity_modifiers(sentence, entity)

def re_api(sentence: str, entities: List[str], api_key: str = API_KEY) -> Dict[str, Any]:
    try:
        extractor = _get_extractor(api_key)
        res = extractor.extract_relations(sentence, entities)
        if res and res.get("relations"):
            return res
    except Exception as e:
        logger.error(f"Error extracting relations via LLM: {e}")

    try:
        global _SPACY_SINGLETON
        if _SPACY_SINGLETON is None:
            import spacy
            _SPACY_SINGLETON = spacy.blank("en") if not spacy.util.is_package("en_core_web_sm") else spacy.load("en_core_web_sm")
        doc = _SPACY_SINGLETON(sentence)
        ents_l = {e.lower() for e in entities}
        rels: List[Dict[str, Any]] = []
        for token in doc:
            if token.pos_ == "VERB" or (token.pos_ == "AUX" and any(c.dep_ in ("xcomp","ccomp","advcl") for c in token.children)):
                subj = next((c for c in token.children if c.dep_ in ("nsubj","nsubjpass") and c.text.lower() in ents_l), None)
                dobj = next((c for c in token.children if c.dep_ in ("dobj","obj") and c.text.lower() in ents_l), None)
                if subj is not None and dobj is not None and subj.text.lower() != dobj.text.lower():
                    rel_text = token.lemma_
                    advmods = [c.text for c in token.children if c.dep_ in ("advmod","neg")]
                    if advmods:
                        rel_text = " ".join(advmods + [rel_text])
                    rels.append({
                        "subject": {"head": subj.text},
                        "relation": {"type": "action", "text": rel_text},
                        "object": {"head": dobj.text}
                    })
        return {"sentence": sentence, "entities": entities, "relations": rels, "approach_used": "spacy_fallback"}
    except Exception as e2:
        logger.error(f"Error extracting relations via spaCy fallback: {e2}")
        return {"sentence": sentence, "entities": entities, "relations": [], "error": str(e2), "approach_used": "none"}

def test_gemma_relation_extraction():
    pass

def batch_extract_relations(sentences_and_entities: List[Tuple[str, List[str]]], api_key: str = None) -> List[Dict[str, Any]]:
    extractor = GemmaRelationExtractor(api_key)
    return [extractor.extract_relations(sentence, entities) for (sentence, entities) in sentences_and_entities]

