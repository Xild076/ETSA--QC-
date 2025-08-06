from typing import Callable, List, Dict, Any
from graph import RelationGraph,  EnsembleSentimentAnalyzer, VADERSentimentAnalyzer, TextBlobSentimentAnalyzer, SWNSentimentAnalyzer, NLPTownSentimentAnalyzer, FiniteAutomataSentimentAnalyzer, DistilBERTLogitSentimentAnalyzer, ProsusAISentimentAnalyzer, PysentimientoSentimentAnalyzer

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from relation_extraction import re_api, GemmaRelationExtractor, API_KEY
from e_c.coref import resolve
from survey.formulas import SentimentFormula
import json
import numpy as np
from survey.survey_question_optimizer import load_optimal_models

def extract_entity_modifiers(sentence: str, entity: str) -> List[str]:
    try:
        extractor = GemmaRelationExtractor(api_key=API_KEY)
        prompt = f"""You are an expert at extracting descriptive modifiers for entities. 

SENTENCE: "{sentence}"
TARGET ENTITY: "{entity}"

Extract ONLY meaningful descriptive modifiers (adjectives, colors, sizes, states, conditions, emotions) that describe the target entity. 

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
            modifiers = json.loads(json_str)
            filtered_modifiers = []
            skip_words = {'', 'the', 'a', 'an', 'his', 'her', 'its', 'their', 'my', 'your', 'our'}
            for mod in modifiers:
                if isinstance(mod, str) and mod.strip().lower() not in skip_words and len(mod.strip()) > 0:
                    filtered_modifiers.append(mod.strip())
            return filtered_modifiers
        
        return []
        
    except Exception as e:
        print(f"Error extracting modifiers for {entity}: {e}")
        return []

def find_cluster_id_for_mention(clusters: Dict[int, Dict[str, Any]], mention_to_find: tuple) -> int:
    for cid, data in clusters.items():
        if mention_to_find in data.get("entity_references", []):
            return cid
    return -1 

def find_cluster_id_for_headword(clusters: Dict, sent_map_entities: List[tuple], headword: str) -> int:
    for mention_tuple in sent_map_entities:
        if mention_tuple[0].lower() == headword.lower():
            return find_cluster_id_for_mention(clusters, mention_tuple)
    return -1

def build_graph_with_optimal_functions(text: str) -> tuple[RelationGraph, dict]:
    actor_func, target_func, association_func, aggregate_func, belonging_func = load_optimal_models()
    
    def compound_action_func(actor_sentiment_init, action_sentiment_init, target_sentiment_init):
        actor_compound = actor_func(actor_sentiment_init, action_sentiment_init)
        target_compound = target_func(target_sentiment_init, action_sentiment_init)
        return actor_compound, target_compound
    
    return build_graph(text, compound_action_func, association_func, belonging_func, aggregate_func)

def build_graph(text: str, action_function: Callable, association_function: Callable, belonging_function: Callable, aggregate_function: Callable) -> tuple[RelationGraph, dict]:
    vader_analyzer = EnsembleSentimentAnalyzer([VADERSentimentAnalyzer(),
                                                TextBlobSentimentAnalyzer(),
                                                SWNSentimentAnalyzer(),
                                                NLPTownSentimentAnalyzer(),
                                                FiniteAutomataSentimentAnalyzer(),
                                                DistilBERTLogitSentimentAnalyzer(),
                                                ProsusAISentimentAnalyzer(),
                                                PysentimientoSentimentAnalyzer()])
    graph = RelationGraph(text, sentiment_analyzer_system=vader_analyzer)
    clusters, sent_map = resolve(text)
    print("--- Coref Output ---")
    print("Clusters:", clusters)
    print("Sentence Map:", sent_map)
    print("\n--- Building Graph ---")
    sentences = text.split('. ')
    for i in range(len(sent_map)):
        clause_key = f"clause_{i}"
        if i < len(sentences):
            clause_text = sentences[i].strip()
            if not clause_text.endswith('.'):
                clause_text += '.'
        else:
            clause_text = ""
        
        current_clause_entities = sent_map[clause_key].get("entities", [])
        if not current_clause_entities or not clause_text:
            continue
        for entity_mention in current_clause_entities:
            entity_text, _ = entity_mention
            cluster_id = find_cluster_id_for_mention(clusters, entity_mention)
            if cluster_id != -1 and not graph.graph.has_node(f"{cluster_id}_{i}"):
                print(f"Adding node: id={cluster_id}, head='{entity_text}', layer={i}")
                graph.add_entity_node(id=cluster_id,
                                      head=entity_text,
                                      modifier=[],
                                      entity_role="associate",
                                      clause_layer=i)
        entity_headwords = [ent[0] for ent in current_clause_entities]
        
        relation_output = re_api(clause_text, entities=entity_headwords)
        relations_found = []
        if relation_output and relation_output.get("relations"):
            relations_found = relation_output.get("relations", [])
            print(f"Found {len(relations_found)} relations in clause")
        
        entity_modifiers = {}
        for entity_name in entity_headwords:
            modifier_output = extract_entity_modifiers(clause_text, entity_name)
            if modifier_output:
                entity_modifiers[entity_name] = modifier_output
                print(f"Modifiers for '{entity_name}': {modifier_output}")
        
        for entity_mention in current_clause_entities:
            entity_text, _ = entity_mention
            cluster_id = find_cluster_id_for_mention(clusters, entity_mention)
            if cluster_id != -1:
                modifiers = entity_modifiers.get(entity_text, [])
                if modifiers:
                    print(f"Adding modifiers {modifiers} to entity {entity_text} (ID: {cluster_id})")
                    graph.add_entity_modifier(entity_id=cluster_id, modifier=modifiers, clause_layer=i)
        
        for relation in relations_found:
            print("Relation found:", relation)
            relation_type = relation.get("relation", {}).get("type", "")
            subject = relation.get("subject", {})
            object_ = relation.get("object", {})
            subject_id = find_cluster_id_for_headword(clusters, current_clause_entities, subject.get("head", ""))
            object_id = find_cluster_id_for_headword(clusters, current_clause_entities, object_.get("head", ""))
            if subject_id == -1 or object_id == -1:
                print(f"Warning: Could not find cluster ID for subject '{subject.get('head')}' or object '{object_.get('head')}'")
                continue
            
            relation_text = relation.get("relation", {}).get("text", "")
            print(f"Adding edge: {relation_type} between {subject_id} and {object_id} in layer {i} ('{relation_text}')")
            if relation_type == "action":
                graph.set_entity_role(subject_id, "actor", i)
                graph.set_entity_role(object_id, "target", i)
                graph.add_action_edge(actor_id=subject_id,
                                      target_id=object_id,
                                      clause_layer=i,
                                      head=relation_text,
                                      modifier=[])
            elif relation_type == "association":
                graph.add_association_edge(entity1_id=subject_id,
                                           entity2_id=object_id,
                                           clause_layer=i)
            elif relation_type == "belonging":
                graph.add_belonging_edge(parent_id=subject_id,
                                         child_id=object_id,
                                         clause_layer=i)
    for entity_id in graph.entity_ids:
        graph.add_temporal_edge(entity_id=entity_id)
    print("\n--- Sentiment Calculations ---")
    graph.run_compound_action_sentiment_calculations(function=action_function)
    graph.run_compound_association_sentiment_calculations(function=association_function)
    graph.run_compound_belonging_sentiment_calculations(function=belonging_function)
    results = {}
    for entity_id in graph.entity_ids:
        entity_name = clusters[entity_id]['entity_references'][0][0]
        agg_sentiment = graph.run_aggregate_sentiment_calculations(entity_id, function=aggregate_function)
        results[f"{entity_name} (ID {entity_id})"] = agg_sentiment
    return graph, results

if __name__ == '__main__':
    # Original working text
    text = "The angry man hit the innocent child. The frightened child was very sad."
    final_graph, aggregate_results = build_graph_with_optimal_functions(text)
    print("\n--- Aggregate Sentiments Over the Entire Story ---")
    for name, sentiment in aggregate_results.items():
        print(f"{name}: {sentiment:.4f}")
    
    # Demonstrate that belonging function is now properly implemented
    print("\n--- Testing Belonging Function ---")
    _, _, _, _, belonging_func = load_optimal_models()
    
    # Test the belonging function with sample parent-child sentiment pairs
    test_cases = [
        (0.5, -0.3, "positive parent, negative child"),
        (-0.2, 0.4, "negative parent, positive child"),  
        (0.7, 0.8, "positive parent, positive child"),
        (-0.6, -0.4, "negative parent, negative child")
    ]
    
    for parent_sent, child_sent, description in test_cases:
        parent_result, child_result = belonging_func(parent_sent, child_sent)
        print(f"{description}: parent {parent_sent:.3f} → {parent_result:.3f}, child {child_sent:.3f} → {child_result:.3f}")
    
    from graph import GraphVisualizer
    visualizer = GraphVisualizer(final_graph)
    visualizer.draw_graph(save_path="integrated_graph.html")
