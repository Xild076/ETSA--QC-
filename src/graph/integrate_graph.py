
from typing import Callable, List, Dict, Any
from graph import RelationGraph, VADERSentimentAnalyzer

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from e_c.coref import resolve

def rebel_api(sentence: str, entities: List[str]) -> Dict[str, Any]:
    print(f"-> Mock REBEL processing clause: '{sentence}' with entities: {entities}")
    if "hit" in sentence and "man" in entities and "child" in entities:
        return {
            "sentence": sentence,
            "entities": ["man", "child"],
            "relations": [{
                "subject": {"head": "man", "modifiers": []},
                "relation": {"type": "action", "text": "hit"},
                "object": {"head": "child", "modifiers": []}
            }]
        }
    if "sad" in sentence and "child" in entities:
        return {"sentence": sentence, "entities": ["child"], "relations": []}
    return None

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

def build_graph(text: str, action_function: Callable, association_function: Callable, belonging_function: Callable, aggregate_function: Callable) -> tuple[RelationGraph, dict]:
    vader_analyzer = VADERSentimentAnalyzer()
    graph = RelationGraph(text, sentiment_analyzer_system=vader_analyzer)
    clusters, sent_map = resolve(text)
    print("--- Coref Output ---")
    print("Clusters:", clusters)
    print("Sentence Map:", sent_map)
    print("\n--- Building Graph ---")
    for i in range(len(sent_map)):
        clause_key = f"clause_{i}"
        clause_text = text[sent_map[clause_key]['entities'][0][1][0]:sent_map[clause_key]['entities'][-1][1][1]]
        current_clause_entities = sent_map[clause_key].get("entities", [])
        if not current_clause_entities:
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
        rebel_output = rebel_api(clause_text, entities=entity_headwords)
        if not rebel_output:
            continue
        for relation in rebel_output.get("relations", []):
            relation_type = relation.get("relation", {}).get("type", "")
            subject = relation.get("subject", {})
            object_ = relation.get("object", {})
            subject_id = find_cluster_id_for_headword(clusters, current_clause_entities, subject.get("head", ""))
            object_id = find_cluster_id_for_headword(clusters, current_clause_entities, object_.get("head", ""))
            if subject_id == -1 or object_id == -1:
                print(f"Warning: Could not find cluster ID for subject '{subject.get('head')}' or object '{object_.get('head')}'")
                continue
            graph.add_entity_modifier(id=subject_id, modifier=relation.get("modifier", []), clause_layer=i)
            graph.add_entity_modifier(id=object_id, modifier=relation.get("modifier", []), clause_layer=i)
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
    text = "The man hit the child. The child was sad."
    def simple_action_logic(actor_s, action_s, target_s):
        return (actor_s + action_s, target_s + action_s)
    def simple_association_logic(e1_s, e2_s):
        return ((e1_s + e2_s) / 2, (e1_s + e2_s) / 2)
    def simple_belonging_logic(parent_s, child_s):
        return ((parent_s + child_s) / 2, (parent_s + child_s) / 2)
    def average_sentiment(sentiments: List[float]):
        return sum(sentiments) / len(sentiments) if sentiments else 0.0
    final_graph, aggregate_results = build_graph(
        text,
        action_function=simple_action_logic,
        association_function=simple_association_logic,
        belonging_function=simple_belonging_logic,
        aggregate_function=average_sentiment
    )
    print("\n--- Aggregate Sentiments Over the Entire Story ---")
    for name, sentiment in aggregate_results.items():
        print(f"{name}: {sentiment:.4f}")
    from graph import GraphVisualizer
    visualizer = GraphVisualizer(final_graph)
    visualizer.draw_graph(save_path="integrated_graph.html")