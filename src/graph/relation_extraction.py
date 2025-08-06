from pathlib import Path
try:
    from dotenv import load_dotenv
    dotenv_path = Path(__file__).parent.parent.parent / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path)
except ImportError:
    pass

import json
import os
from typing import List, Dict, Any, Tuple
from datetime import datetime
import google.generativeai as genai

API_KEY = os.getenv("GOOGLE_API_KEY")

class GemmaRelationExtractor:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemma-3-27b-it")
        
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
            # Skip malformed relations
            if not self._is_valid_relation(relation):
                continue
                
            # Create a signature to avoid duplicates
            signature = (
                relation["subject"]["head"].lower(),
                relation["relation"]["type"].lower(),
                relation["relation"]["text"].lower(),
                relation["object"]["head"].lower()
            )
            
            if signature not in seen_relations:
                # Clean up modifiers (remove empty strings and duplicates)
                if "modifiers" in relation["subject"]:
                    relation["subject"]["modifiers"] = list(filter(None, set(relation["subject"]["modifiers"])))
                if "modifiers" in relation["object"]:
                    relation["object"]["modifiers"] = list(filter(None, set(relation["object"]["modifiers"])))
                    
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
                relation["subject"]["head"] != relation["object"]["head"]  # No self-relations
            )
        except (KeyError, TypeError):
            return False

    def _create_prompt(self, sentence: str, entities: List[str]) -> str:
        entities_json = json.dumps(entities)
        
        return f"""You are an expert relation extraction system. Analyze the given sentence and extract relationships between the specified entities.

SENTENCE: "{sentence}"
ENTITIES: {entities_json}

Extract relationships of these types:
1. ACTION: One entity performs an action on another (hit, help, attack, save, etc.)
2. ASSOCIATION: Entities are connected or work together (with, together, alongside, etc.)  
3. BELONGING: One entity belongs to or is owned by another (possessive, part-of relationships)

For each entity mention, also extract descriptive MODIFIERS (adjectives, colors, sizes, descriptive phrases like "was/is/are [insert something]", etc.).

CRITICAL: Use lowercase relation types (action, association, belonging) and ensure ALL strings use double quotes, not single quotes.

Return ONLY a valid JSON object in this exact format:
{{
  "sentence": "{sentence}",
  "entities": {entities_json},
  "relations": [
    {{
      "subject": {{
        "head": "entity_name",
        "modifiers": ["modifier1", "modifier2"]
      }},
      "relation": {{
        "type": "action",
        "text": "relationship_description"
      }},
      "object": {{
        "head": "entity_name", 
        "modifiers": ["modifier1", "modifier2"]
      }}
    }}
  ]
}}

Examples:
- "The red car crashed into the blue truck" → car (modifiers: ["red"]) ACTION "crashed into" truck (modifiers: ["blue"])
- "John and Mary worked together" → John ASSOCIATION "worked together" Mary
- "Alice's phone is broken" → Alice BELONGING "owns" phone (modifiers: ["broken"])
- "The phone's wifi was terrible" → phone BELONGING "owns" wifi (modifiers: ["was terrible"])

IMPORTANT: Extract meaningful directional relations. Avoid creating symmetric relations unless they truly are bidirectional. Return valid JSON only with double quotes."""

    def _query_gemma(self, prompt: str) -> str:
        generation_config = genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=2000,
            top_p=0.9,
            top_k=40
        )
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            if not response.text or response.text.strip() == "":
                raise ValueError("Empty response from Gemma model")
            return response.text
        except Exception as e:
            print(f"Error querying Gemma model: {e}")
            raise

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
            response_clean = self._fix_json_quotes(response_clean)
            result = json.loads(response_clean)
            if "relations" not in result:
                result["relations"] = []
            result["sentence"] = sentence
            result["entities"] = entities
            for relation in result["relations"]:
                if "subject" not in relation:
                    relation["subject"] = {"head": "", "modifiers": []}
                if "object" not in relation:
                    relation["object"] = {"head": "", "modifiers": []}
                if "relation" not in relation:
                    relation["relation"] = {"type": "unknown", "text": ""}
                if "modifiers" not in relation["subject"]:
                    relation["subject"]["modifiers"] = []
                if "modifiers" not in relation["object"]:
                    relation["object"]["modifiers"] = []
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
            "subject": {"head": entity1, "modifiers": []},
            "relation": {"type": rel_type, "text": rel_text},
            "object": {"head": entity2, "modifiers": []}
        }

    def _is_duplicate_relation(self, relations: List[Dict], new_relation: Dict) -> bool:
        """Check if a relation already exists to avoid duplicates."""
        for existing in relations:
            if (existing["subject"]["head"] == new_relation["subject"]["head"] and
                existing["object"]["head"] == new_relation["object"]["head"] and
                existing["relation"]["type"] == new_relation["relation"]["type"]):
                return True
        return False

def re_api(sentence: str, entities: List[str], api_key: str = API_KEY) -> Dict[str, Any]:
    extractor = GemmaRelationExtractor(api_key)
    return extractor.extract_relations(sentence, entities)

def test_gemma_relation_extraction():
    print("=== GEMMA 27B RELATION EXTRACTION TESTING ===")
    print(f"Current Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"User: Xild076")
    print(f"Model: Gemma 27B via Google AI Studio\n")
    
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
                        print(f"    Subject: '{rel['subject']['head']}'", end="")
                        if rel['subject'].get('modifiers'):
                            print(f" (modifiers: {rel['subject']['modifiers']})", end="")
                        print()
                        print(f"    Type: {rel['relation']['type'].upper()}")
                        print(f"    Text: '{rel['relation']['text']}'")
                        print(f"    Object: '{rel['object']['head']}'", end="")
                        if rel['object'].get('modifiers'):
                            print(f" (modifiers: {rel['object']['modifiers']})", end="")
                        print()
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
                    if rel['subject'].get('modifiers'):
                        subj_str += f" (modifiers: {rel['subject']['modifiers']})"
                    
                    obj_str = f"'{rel['object']['head']}'"
                    if rel['object'].get('modifiers'):
                        obj_str += f" (modifiers: {rel['object']['modifiers']})"
                    
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

"""
if __name__ == "__main__":
    test_gemma_relation_extraction()"""