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

logging.basicConfig(level=logging.INFO)
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

class GemmaRelationExtractor:
    def __init__(self, api_key: str = None, rate_limiter: _RateLimiter | None = None):
        logger.info("Initializing GemmaRelationExtractor...")
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemma-3-27b-it")
        self._rate_limiter = rate_limiter or _GLOBAL_GENAI_LIMITER
        
    def extract_relations(self, sentence: str, entities: List[str]) -> Dict[str, Any]:
        logger.info("Extracting relations...")
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
        return f"""You are an expert relation extraction system. Extract ONLY relationships between the listed entities. Be exhaustive and precise.

SENTENCE: "{sentence}"
ENTITIES: {entities_json}

TYPES
1) action: Directed verb phrase where the subject acts on the object.
     - Include particles/prepositions/adverbs that are integral to the verb phrase (e.g., "tended to", "streaked across", "was moored in", "were covered in").
     - Do NOT include direct objects or attached "of/with" complements inside the relation text; stop the phrase before the object NP.
2) association: Cooperative/symmetric joint activity between two entities (worked/partnered/collaborated/played … together/with/alongside).
     - Use only for joint activity; not for location/path prepositions.
     - Emit association ONCE per unordered pair. Choose the earlier-mentioned entity as subject; do not output the reverse.
     - If the same verb also takes a separate object (e.g., "worked on X"), you may also output action relations to that object, but DO NOT also output a duplicate action between the two associates.
3) belonging: Ownership or part-of (X's Y, Y of X, owns, has/have/had, with [possession], belongs to).
     - Canonicalize as follows:
         a) Possessive (X's Y): subject=X, text="owns" or "has", object=Y.
         a2) Possessive pronoun referring to an entity (his/her/their/its Y): subject=referent entity, text="owns" or "has", object=Y.
         b) "Y of X":
                - If Y is a concrete part/property (wheel, door, screen, design), prefer owner direction: subject=X, text="has", object=Y.
                - If Y is a collection/measure/event noun (flock, herd, team, copy/copies, series, takeover), keep subject=Y, text="of", object=X.
         c) Noun-with-Noun ("room with ceilings"): subject=first noun, text="with", object=second noun as belonging.
     - Do NOT create belonging for mere spatial relations.

ALGORITHM (internal – do not output these steps)
Enumerate all ordered pairs of distinct entities (A, B) from ENTITIES. For each pair:
        - If possessive/of/with-possession pattern links them: add belonging per the rules above.
        - If a joint-activity verb connects them with with/together/alongside: add association for the pair once.
        - If a directed verb phrase goes from A to B: add action A→B; include verb + required particles/preps/adverbs, but no objects or "of/with NP" complements.
Do this for every pair; many sentences have multiple relations. Avoid duplicates.

DECISIONS CHECKLIST
- Use only ENTITIES; use exact surface form in heads.
- Output ALL relations present; don’t omit valid ones.
- Never output relations where subject == object.
- Reject preposition-only relation texts: of, with, from, to, into, onto, in, on, at, by, for, about, over, under, across, through, along, past, beyond. Use them only inside a verb phrase or as belonging/association per the rules.
- Disambiguate "with":
    • VERB with NP where VERB is symmetric (work, collaborate, partner, play, share): association between the two participants only.
    • NP with NP (possession/attributes): belonging.
- Collective NPs ("group/flock/team/herd of X"): the collective performs actions; do not also assign the same action to X unless explicitly stated.
- Event/collection nouns ("takeover of company", "copies of the book"): belonging with text "of" from the event/collection to its theme (subject=event/collection, object=theme).
 - Association requires agentive participants (people/animals/groups/organizations). Do not create association with inanimate objects, events, or moments.
 - Ignore path/location attachments like "from X", "in/at/on X" when they only indicate location/source and not a true action relation between two entities.

OUTPUT FORMAT (JSON only):
{{
    "sentence": "{sentence}",
    "entities": {entities_json},
    "relations": [
        {{
            "subject": {{ "head": "entity_name" }},
            "relation": {{ "type": "action|association|belonging", "text": "relationship_description" }},
            "object": {{ "head": "entity_name" }}
        }}
    ]
}}

POSITIVE EXAMPLES (generic; not from your dataset)
- "The biologist carefully cataloged rare specimens" → biologist action "carefully cataloged" specimens.
- "The programmer collaborated with the designer on the interface" → programmer association "collaborated with" designer; actions "worked on" may target "interface" but NOT between programmer↔designer again.
- "A flock of birds crossed the valley" → flock belonging "of" birds; flock action "crossed" valley; do not also say birds crossed, unless stated.
- "The wheel of the car was replaced" → car belonging "has" wheel.
- "The copies of the novel sold quickly" → copies belonging "of" novel.
- "The room with skylights felt bright" → room belonging "with" skylights.
- "They partnered together on the study" → association "partnered with" between participants only; action "worked on" may target study.

NEGATIVE/EDGE GUIDANCE
- Do not create relations where the only connector is a path/location preposition (from/to/into/onto/in/on/at/by/over/under/across/through/along/past/beyond) unless covered by action verb phrases.
- Do not include objects or "of/with NP" complements inside action relation text.
- If a relation phrase connects an entity to a non-entity concept, ignore it unless the concept is in ENTITIES.
- If both association and action to a third object are present, include both; but emit association only once per pair.

IMPORTANT
- Use lowercase relation types (action, association, belonging).
- Use only double quotes in JSON.
- Return valid JSON only, no extra text."""

    def _query_gemma(self, prompt: str) -> str:
        generation_config = genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=2000,
            top_p=0.9,
            top_k=40
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
    logger.info(f"Extracting modifiers for entity '{entity}' in sentence: {sentence}")
    try:
        extractor = GemmaRelationExtractor(api_key=API_KEY)
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
        print(f"Error extracting modifiers for {entity}: {e}")
        return []

def re_api(sentence: str, entities: List[str], api_key: str = API_KEY) -> Dict[str, Any]:
    logger.info(f"Extracting relations for sentence: {sentence} with entities: {entities}")
    try:
        extractor = GemmaRelationExtractor(api_key)
        return extractor.extract_relations(sentence, entities)
    except Exception as e:
        logger.error(f"Error extracting relations: {e}")
        return {"sentence": sentence, "entities": entities, "relations": [], "error": str(e), "approach_used": "none"}

def test_gemma_relation_extraction():
    logger.info("=== GEMMA 27B RELATION EXTRACTION TESTING ===")

    action_tests = [
        ("The man hit the child.", ["man", "child"]),
        ("The red car crashed into the blue truck.", ["car", "truck"]),
        ("Alice helped her sick friend Bob recover quickly.", ["Alice", "Bob"])
    ]
    
    association_tests = [
        ("John and Mary worked together on the difficult project.", ["John", "Mary"]),
        ("The big dog played with the small cat happily.", ["dog", "cat"]),
        ("Smart students and experienced teachers collaborated effectively on research.", ["students", "teachers"])
    ]
    
    belonging_tests = [
        ("John's expensive car is very fast.", ["John", "car"]),
        ("The rusty wheels of the old bicycle are completely broken.", ["wheels", "bicycle"]),
        ("The phone's wifi was terrible and slow.", ["phone", "wifi"])
    ]
    
    try:
        extractor = GemmaRelationExtractor(api_key=API_KEY)
        
        all_tests = [
            ("ACTION RELATIONS", action_tests),
            ("ASSOCIATION RELATIONS", association_tests), 
            ("BELONGING RELATIONS", belonging_tests)
        ]
        
        total_success = 0
        total_tests = 0
        
        for test_type, tests in all_tests:
            print(f"{test_type}:")
            print("-" * 70)
            
            for i, (sentence, entities) in enumerate(tests, 1):
                total_tests += 1
                print(f"Test {i}: {sentence}")
                print(f"Target entities: {entities}")
                
                result = extractor.extract_relations(sentence, entities)
                
                print(f"Model used: {result.get('approach_used', 'unknown')}")
                print(f"Timestamp: {result.get('timestamp', 'N/A')}")
                if result.get("relations"):
                    total_success += 1
                    for j, rel in enumerate(result["relations"], 1):
                        print(f"  ✅ Relation {j}:")
                        print(f"    Subject: '{rel['subject']['head']}'")
                        print(f"    Type: {rel['relation']['type'].upper()}")
                        print(f"    Text: '{rel['relation']['text']}'")
                        print(f"    Object: '{rel['object']['head']}'")
                else:
                    print("  ❌ No relations extracted")
                    if "error" in result:
                        print(f"  Error: {result['error']}")
                if result.get("fallback_used"):
                    print("  ⚠️ Fallback parsing was used")
                print()
            print()
        
        print("COMPLEX SENTENCE TESTING:")
        print("-" * 70)
        
        complex_tests = [
            ("The car's wheel was bad.", ["car", "wheel"]),
            ("The angry teacher scolded the lazy student harshly.", ["teacher", "student"]),
            ("Microsoft's new software contains serious security vulnerabilities.", ["Microsoft", "software", "vulnerabilities"]),
            ("The broken phone's cracked screen displayed fuzzy images.", ["phone", "screen"])
        ]
        
        for i, (sentence, entities) in enumerate(complex_tests, 1):
            total_tests += 1
            print(f"Complex Test {i}: {sentence}")
            print(f"Entities: {entities}")
            
            result = extractor.extract_relations(sentence, entities)
            
            if result.get("relations"):
                total_success += 1
                for rel in result["relations"]:
                    subj_str = f"'{rel['subject']['head']}'"
                    obj_str = f"'{rel['object']['head']}'"
                    print(f" FOUND! {subj_str} --{rel['relation']['type']}: {rel['relation']['text']}--> {obj_str}")
            else:
                print("No relations found")
            print()
        
        print("=" * 70)
        print(f"SUMMARY: {total_success}/{total_tests} tests extracted relations successfully")
        success_rate = (total_success / total_tests) * 100 if total_tests > 0 else 0
        print(f"Success Rate: {success_rate:.1f}%")
        
    except Exception as e:
        print(f"Failed to initialize Gemma extractor: {e}")

def batch_extract_relations(sentences_and_entities: List[Tuple[str, List[str]]], api_key: str = None) -> List[Dict[str, Any]]:
    extractor = GemmaRelationExtractor(api_key)
    results = []
    
    print(f"Processing {len(sentences_and_entities)} sentences with Gemma 27B...")
    
    for i, (sentence, entities) in enumerate(sentences_and_entities, 1):
        print(f"Processing {i}/{len(sentences_and_entities)}: {sentence[:50]}...")
        result = extractor.extract_relations(sentence, entities)
        results.append(result)
    
    return results

