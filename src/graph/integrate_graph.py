from typing import Callable, List, Dict, Any
from graph import RelationGraph, VADERSentimentAnalyzer, EnsembleSentimentAnalyzer, WeightedEnsembleSentimentAnalyzer

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from relation_extraction import re_api, GemmaRelationExtractor, API_KEY
from e_c.coref import resolve
from survey.formulas import SentimentFormula
import json
import numpy as np

def load_optimal_functions():
    try:
        current_dir = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        file_path = os.path.join(project_root, 'src', 'survey', 'optimal_formulas', 'all_optimal_parameters.json')
        with open(file_path, 'r') as f:
            params = json.load(f)
        return params
    except Exception as e:
        print(f"Could not load optimal functions: {e}")
        return {}

def create_optimal_sentiment_functions():
    optimal_funcs = load_optimal_functions()
    
    def action_func(actor_s, action_s, target_s):
        if 'Actor' in optimal_funcs and 'Target' in optimal_funcs:
            actor_data = optimal_funcs['Actor']
            target_data = optimal_funcs['Target']
            
            actor_result = actor_s
            target_result = target_s
            
            if actor_data.get('split_type') == 'action_target':
                if action_s > 0 and target_s > 0 and 'pos_pos_params' in actor_data.get('sub_models', {}):
                    params = np.fromstring(actor_data['sub_models']['pos_pos_params']['params'].strip('[]'), sep=' ')
                    driver = action_s * target_s
                    if actor_data.get('function') == 'actor_formula_v2':
                        actor_result = np.tanh(params[0] * actor_s + params[1] * driver + params[2])
                elif action_s > 0 and target_s <= 0 and 'pos_neg_params' in actor_data.get('sub_models', {}):
                    params = np.fromstring(actor_data['sub_models']['pos_neg_params']['params'].strip('[]'), sep=' ')
                    driver = action_s * target_s
                    if actor_data.get('function') == 'actor_formula_v2':
                        actor_result = np.tanh(params[0] * actor_s + params[1] * driver + params[2])
                elif action_s <= 0 and target_s > 0 and 'neg_pos_params' in actor_data.get('sub_models', {}):
                    params = np.fromstring(actor_data['sub_models']['neg_pos_params']['params'].strip('[]'), sep=' ')
                    driver = action_s * target_s
                    if actor_data.get('function') == 'actor_formula_v2':
                        actor_result = np.tanh(params[0] * actor_s + params[1] * driver + params[2])
            
            if target_data.get('split_type') == 'driver':
                driver = action_s * target_s
                if driver > 0 and 'pos_driver_params' in target_data.get('sub_models', {}):
                    params = np.fromstring(target_data['sub_models']['pos_driver_params']['params'].strip('[]'), sep=' ')
                    if target_data.get('function') == 'target_formula_v2':
                        target_result = np.tanh(params[0] * target_s + params[1] * action_s + params[2])
                elif driver <= 0 and 'neg_driver_params' in target_data.get('sub_models', {}):
                    params = np.fromstring(target_data['sub_models']['neg_driver_params']['params'].strip('[]'), sep=' ')
                    if target_data.get('function') == 'target_formula_v2':
                        target_result = np.tanh(params[0] * target_s + params[1] * action_s + params[2])
            
            return (actor_result, target_result)
        return (actor_s + action_s, target_s + action_s)
    
    def association_func(e1_s, e2_s):
        if 'Association' in optimal_funcs:
            assoc_data = optimal_funcs['Association']
            if assoc_data.get('split_type') == 'entity_other':
                if e1_s > 0 and 'pos_params' in assoc_data.get('sub_models', {}):
                    params = np.fromstring(assoc_data['sub_models']['pos_params']['params'].strip('[]'), sep=' ')
                    if assoc_data.get('function') == 'assoc_formula_v2':
                        e1_result = np.tanh(params[0] * e1_s + params[1] * e2_s + params[2])
                        e2_result = np.tanh(params[0] * e2_s + params[1] * e1_s + params[2])
                        return (e1_result, e2_result)
                elif e1_s <= 0 and e2_s <= 0 and 'neg_neg_params' in assoc_data.get('sub_models', {}):
                    params = np.fromstring(assoc_data['sub_models']['neg_neg_params']['params'].strip('[]'), sep=' ')
                    if assoc_data.get('function') == 'assoc_formula_v2':
                        e1_result = np.tanh(params[0] * e1_s + params[1] * e2_s + params[2])
                        e2_result = np.tanh(params[0] * e2_s + params[1] * e1_s + params[2])
                        return (e1_result, e2_result)
        return ((e1_s + e2_s) / 2, (e1_s + e2_s) / 2)
    
    def belonging_func(parent_s, child_s):
        if 'Belonging Parent' in optimal_funcs:
            belong_data = optimal_funcs['Belonging Parent']
            if belong_data.get('split_type') == 'parent_child':
                if parent_s > 0 and child_s <= 0 and 'pos_neg_params' in belong_data.get('sub_models', {}):
                    params = np.fromstring(belong_data['sub_models']['pos_neg_params']['params'].strip('[]'), sep=' ')
                    if belong_data.get('function') == 'belong_formula_v2':
                        parent_result = np.tanh(params[0] * parent_s + params[1] * child_s + params[2])
                        child_result = np.tanh(params[0] * child_s + params[1] * parent_s + params[2])
                        return (parent_result, child_result)
        return ((parent_s + child_s) / 2, (parent_s + child_s) / 2)
    
    def aggregate_func(sentiments: List[float]):
        if not sentiments:
            return 0.0
        for key in ['normal', 'dynamic', 'logistic']:
            agg_key = f'Aggregate_{key}'
            if agg_key in optimal_funcs:
                agg_data = optimal_funcs[agg_key]
                if 'function' in agg_data:
                    func_name = agg_data['function']
                    if 'params' in agg_data:
                        params = np.fromstring(agg_data['params'].strip('[]'), sep=' ') if isinstance(agg_data['params'], str) else agg_data['params']
                        if func_name == 'aggregate_normal':
                            return np.mean(sentiments) * params[0] + params[1]
                        elif func_name == 'aggregate_dynamic':
                            mean_val = np.mean(sentiments)
                            return params[0] * mean_val + params[1] * np.exp(params[2] * mean_val) + params[3]
                        elif func_name == 'aggregate_logistic':
                            mean_val = np.mean(sentiments)
                            return params[0] / (1 + np.exp(-params[1] * (mean_val - params[2]))) + params[3]
                break
        return sum(sentiments) / len(sentiments)
    
    return action_func, association_func, belonging_func, aggregate_func

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
    action_func, association_func, belonging_func, aggregate_func = create_optimal_sentiment_functions()
    return build_graph(text, action_func, association_func, belonging_func, aggregate_func)

def build_graph(text: str, action_function: Callable, association_function: Callable, belonging_function: Callable, aggregate_function: Callable) -> tuple[RelationGraph, dict]:
    vader_analyzer = WeightedEnsembleSentimentAnalyzer()
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
    text = "The angry man hit the innocent child. The frightened child was very sad."
    final_graph, aggregate_results = build_graph_with_optimal_functions(text)
    print("\n--- Aggregate Sentiments Over the Entire Story ---")
    for name, sentiment in aggregate_results.items():
        print(f"{name}: {sentiment:.4f}")
    from graph import GraphVisualizer
    visualizer = GraphVisualizer(final_graph)
    visualizer.draw_graph(save_path="integrated_graph.html")
