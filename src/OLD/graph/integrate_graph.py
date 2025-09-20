from typing import Callable, List, Dict, Any
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass
import os, sys
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.graph.graph import RelationGraph
from src.graph.sas import PresetEnsembleSentimentAnalyzer
from src.graph.relation_extraction import (
    re_api,
    context_window,
    get_entity_modifiers,
)
from src.e_c.coref import resolve
from src.survey.survey_question_optimizer import (
    get_actor_function,
    get_target_function,
    get_parent_function,
    get_child_function,
    get_association_function,
    get_aggregate_function,
)
import json  # kept if future debug/serialization is needed
import logging
from pprint import pformat
logger = logging.getLogger(__name__)

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

def build_graph_with_optimal_functions(text: str, debug: bool = False) -> tuple[RelationGraph, dict]:
    
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
    return build_graph(text, action_function, association_function, belonging_function, aggregate_function, debug=debug)

def build_graph(text: str, action_function: Callable, association_function: Callable, belonging_function: Callable, aggregate_function: Callable, debug: bool = False) -> tuple[RelationGraph, dict]:
    env_debug = str(os.getenv("ETSA_DEBUG", "0")).lower() in {"1", "true", "yes", "on"}
    debug = bool(debug or env_debug)
    _print = (lambda *a, **k: print(*a, **k)) if debug else (lambda *a, **k: None)
    _pp = lambda o: pformat(o, width=100, compact=False)

    _print("== Build Graph: start ==")
    _print("Input text:\n" + text.strip())
    analyzer = PresetEnsembleSentimentAnalyzer()
    graph = RelationGraph(text, sentiment_analyzer_system=analyzer)
    clusters, sent_map = resolve(text)
    _print("Coref clusters:")
    _print(_pp(clusters))
    _print("Sentence map (keys only):", list(sent_map.keys()))
    
    import re
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    _print(f"Split into {len(sentences)} sentence(s)")
    
    for i in range(len(sent_map)):
        _print("\n-- Clause", i, "--")
        clause_key = f"clause_{i}"
        if i < len(sentences):
            clause_text = sentences[i].strip()
            if not clause_text.endswith(('.', '!', '?')):
                clause_text += '.'
        else:
            clause_text = ""
        _print("Clause text:", clause_text)
        
        current_clause_entities = sent_map[clause_key].get("entities", [])
        if not current_clause_entities or not clause_text:
            _print("No entities or empty clause; skipping.")
            continue
        _print("Entities (mentions) in clause:", _pp(current_clause_entities))
            
        for entity_mention in current_clause_entities:
            entity_text, _ = entity_mention
            cluster_id = find_cluster_id_for_mention(clusters, entity_mention)
            if cluster_id != -1 and not graph.graph.has_node(f"{cluster_id}_{i}"):
                graph.add_entity_node(id=cluster_id,
                                      head=entity_text,
                                      modifier=[],
                                      entity_role="associate",
                                      clause_layer=i)
                _print(f"Added entity node: id={cluster_id} head='{entity_text}' layer={i}")
        
        entity_headwords = [ent[0] for ent in current_clause_entities]
        try:
            import spacy
            nlp = spacy.blank("en") if not spacy.util.is_package("en_core_web_sm") else spacy.load("en_core_web_sm")
            doc = nlp(clause_text)
            for np in doc.noun_chunks:
                cand = np.text.strip()
                if cand and cand.lower() not in {e.lower() for e in entity_headwords}:
                    if any(tok.dep_ == "compound" for tok in np.root.children) or len(cand.split()) >= 2:
                        entity_headwords.append(cand)
            _print("Augmented entity headwords:", _pp(entity_headwords))
        except Exception:
            _print("spaCy noun chunking unavailable; proceeding with existing headwords.")
            pass

        ctx = context_window(sentences, i, width=1)
        _print("Context window:", _pp(ctx if ctx else clause_text))
        entity_modifiers: dict[str, list[str]] = {}
        for entity_name in entity_headwords:
            try:
                modifier_output = get_entity_modifiers(clause_text, entity_name)
                if modifier_output:
                    entity_modifiers[entity_name] = modifier_output
                    _print(f"Modifiers for '{entity_name}':", _pp(modifier_output))
            except Exception:
                _print(f"Modifier extraction failed for '{entity_name}'.")
                continue

        for entity_mention in current_clause_entities:
            entity_text, _ = entity_mention
            cluster_id = find_cluster_id_for_mention(clusters, entity_mention)
            if cluster_id != -1:
                modifiers = entity_modifiers.get(entity_text, [])
                if modifiers:
                    graph.add_entity_modifier(entity_id=cluster_id, modifier=modifiers, clause_layer=i)
                    _print(f"Applied modifiers to node {cluster_id} @ layer {i}:", _pp(modifiers))

        relation_output = re_api(ctx if ctx else clause_text, entities=entity_headwords)
        relations_found = relation_output.get("relations", []) if relation_output else []
        _print("Relation API input entities:", _pp(entity_headwords))
        _print("Relation API output relations:", _pp(relations_found))
        if not relations_found:
            try:
                import spacy
                nlp = spacy.blank("en") if not spacy.util.is_package("en_core_web_sm") else spacy.load("en_core_web_sm")
                doc = nlp(clause_text)
                ents_l = {e.lower() for e in entity_headwords}
                for token in doc:
                    if token.dep_ in ("ROOT",) and token.pos_ in ("VERB","AUX"):
                        subj = next((c for c in token.children if c.dep_ in ("nsubj","nsubjpass") and c.text.lower() in ents_l), None)
                        obj = next((c for c in token.children if c.dep_ in ("dobj","obj","attr") and c.text.lower() in ents_l), None)
                        if subj is not None and obj is not None and subj.text.lower()!=obj.text.lower():
                            advs = [c.text for c in token.children if c.dep_ in ("advmod","neg")]
                            rel_text = (" ".join(advs+[token.lemma_])).strip()
                            relations_found.append({
                                "subject": {"head": subj.text},
                                "relation": {"type": "action", "text": rel_text},
                                "object": {"head": obj.text}
                            })
                if relations_found:
                    _print("Heuristic relations (spaCy) found:", _pp(relations_found))
                else:
                    _print("No relations found (API and heuristic).")
            except Exception:
                _print("spaCy heuristic relation extraction unavailable.")
                pass
        if relations_found:
            for relation in relations_found:
                relation_type = relation.get("relation", {}).get("type", "")
                subject = relation.get("subject", {})
                object_ = relation.get("object", {})
                subject_id = find_cluster_id_for_headword(clusters, current_clause_entities, subject.get("head", ""))
                object_id = find_cluster_id_for_headword(clusters, current_clause_entities, object_.get("head", ""))
                if subject_id == -1 or object_id == -1:
                    _print("Skipping relation due to unresolved subject/object cluster:", _pp(relation))
                    continue

                relation_text = relation.get("relation", {}).get("text", "")
                if relation_type == "action":
                    graph.set_entity_role(subject_id, "actor", i)
                    graph.set_entity_role(object_id, "target", i)
                    graph.add_action_edge(
                        actor_id=subject_id,
                        target_id=object_id,
                        clause_layer=i,
                        head=relation_text,
                        modifier=[],
                    )
                    _print(f"Add action edge: {subject_id} -> {object_id} @ layer {i} head='{relation_text}'")
                elif relation_type == "association":
                    graph.add_association_edge(
                        entity1_id=subject_id,
                        entity2_id=object_id,
                        clause_layer=i,
                    )
                    _print(f"Add association edge: {subject_id} -- {object_id} @ layer {i}")
                elif relation_type == "belonging":
                    graph.add_belonging_edge(
                        parent_id=subject_id,
                        child_id=object_id,
                        clause_layer=i,
                    )
                    _print(f"Add belonging edge: parent {subject_id} -> child {object_id} @ layer {i}")
    
    for entity_id in graph.entity_ids:
        graph.add_temporal_edge(entity_id=entity_id)
    _print(f"Added temporal edges for {len(graph.entity_ids)} entity/entities")
    
    graph.run_compound_action_sentiment_calculations(function=action_function)
    graph.run_compound_association_sentiment_calculations(function=association_function)
    graph.run_compound_belonging_sentiment_calculations(function=belonging_function)
    _print("Ran compound sentiment calculations (action, association, belonging)")
    
    results = {}
    for entity_id in graph.entity_ids:
        entity_name = clusters[entity_id]['entity_references'][0][0]
        agg_sentiment = graph.run_aggregate_sentiment_calculations(entity_id, function=aggregate_function)
        results[f"{entity_name} (ID {entity_id})"] = agg_sentiment
        _print(f"Aggregate sentiment for {entity_name} (ID {entity_id}):", _pp(agg_sentiment))
    
    _print("== Build Graph: done ==")
    return graph, results

 
if __name__ == "__main__":
    text = """Alex was the most loyal friend anyone could ask for, a constant source of encouragement and unwavering support for Ben's ambitions. Convinced he was only protecting their shared future, Alex quietly submitted his own name instead of Ben's for the single fellowship opening overseas. From his apartment, Ben read the acceptance email addressed to his best friend and felt a deep, hollow sadness settle in his chest.
    """
    graph, results = build_graph_with_optimal_functions(text, debug=True)
    print("Results:")
    from pprint import pprint as _pprint
    _pprint(results)