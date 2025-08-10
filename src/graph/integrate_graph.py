from typing import Callable, List, Dict, Any
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass
from src.graph.graph import RelationGraph
from src.graph.sas import PresetEnsembleSentimentAnalyzer
from src.graph.relation_extraction import re_api, GemmaRelationExtractor, API_KEY, extract_entity_modifiers
from src.e_c.coref import resolve
from src.survey.survey_question_optimizer import (
    get_actor_function,
    get_target_function,
    get_parent_function,
    get_child_function,
    get_association_function,
    get_aggregate_function,
)
import json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_cluster_id_for_mention(clusters: Dict[int, Dict[str, Any]], mention_to_find: tuple) -> int:
    logger.info(f"Finding cluster ID for mention: {mention_to_find}")
    for cid, data in clusters.items():
        if mention_to_find in data.get("entity_references", []):
            return cid
    return -1 

def find_cluster_id_for_headword(clusters: Dict, sent_map_entities: List[tuple], headword: str) -> int:
    logger.info(f"Finding cluster ID for headword: {headword}")
    for mention_tuple in sent_map_entities:
        if mention_tuple[0].lower() == headword.lower():
            return find_cluster_id_for_mention(clusters, mention_tuple)
    return -1

def build_graph_with_optimal_functions(text: str) -> tuple[RelationGraph, dict]:
    logger.info("Building graph with optimal functions...")
    actor_function = get_actor_function()
    target_function = get_target_function()
    def action_function(actor_sentiment_init, action_sentiment_init, target_sentiment_init):
        return actor_function(actor_sentiment_init, action_sentiment_init, target_sentiment_init), target_function(target_sentiment_init, action_sentiment_init)
    assoc_function = get_association_function()
    def association_function(entity1_sentiment_init, entity2_sentiment_init):
        return assoc_function(entity1_sentiment_init, entity2_sentiment_init), assoc_function(entity2_sentiment_init, entity1_sentiment_init)
    parent_function = get_parent_function()
    child_function = get_child_function()
    def belonging_function(parent_sentiment_init, child_sentiment_init):
        return parent_function(parent_sentiment_init, child_sentiment_init), child_function(child_sentiment_init, parent_sentiment_init)
    aggregate_function = get_aggregate_function()
    return build_graph(text, action_function, association_function, belonging_function, aggregate_function)

def build_graph(text: str, action_function: Callable, association_function: Callable, belonging_function: Callable, aggregate_function: Callable) -> tuple[RelationGraph, dict]:
    logger.info("Starting graph building process...")
    vader_analyzer = PresetEnsembleSentimentAnalyzer()
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
    from src.graph.graph import GraphVisualizer
    visualizer = GraphVisualizer(final_graph)
    visualizer.draw_graph(save_path="integrated_graph.html")
