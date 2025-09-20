import os
import sys
import json
import logging
import re
from typing import Dict, Any, List, Callable
from pprint import pformat

logger = logging.getLogger(__name__)

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from .graph import RelationGraph, GraphVisualizer
from .ner_coref import EntityConsolidator
from .clause_splitter import SpacyClauseSplitter as ClauseSplitter
from .re_e import RelationExtractor
from .modifier_e import ModifierExtractor
from .sentiment_systems import SentimentSystem
from ..survey.formula_loader import (
    get_actor_function, get_target_function, get_association_function,
    get_parent_function, get_child_function, get_aggregate_function
)

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

class Pipeline:
    def __init__(self,
                 entity_consolidator: EntityConsolidator,
                 clause_splitter: ClauseSplitter,
                 relation_extractor: RelationExtractor,
                 modifier_extractor: ModifierExtractor,
                 sentiment_system: SentimentSystem,
                 mode: str = "full_stack"):
        self.entity_consolidator = entity_consolidator
        self.clause_splitter = clause_splitter
        self.relation_extractor = relation_extractor
        self.modifier_extractor = modifier_extractor
        self.sentiment_system = sentiment_system
        self.mode = mode

        if mode == "no_formulas":
            self.actor_func = lambda x: sum(x) / len(x) if x else 0.0
            self.target_func = lambda x: sum(x) / len(x) if x else 0.0
            self.assoc_func = lambda x: sum(x) / len(x) if x else 0.0
            self.parent_func = lambda x: sum(x) / len(x) if x else 0.0
            self.child_func = lambda x: sum(x) / len(x) if x else 0.0
            self.agg_func = lambda scores: sum(scores) / len(scores) if scores else 0.0
        else:
            self.actor_func = get_actor_function()
            self.target_func = get_target_function()
            self.assoc_func = get_association_function()
            self.parent_func = get_parent_function()
            self.child_func = get_child_function()
            self.agg_func = get_aggregate_function()

    def run(self, text: str, debug_run_name: str = None) -> Dict[str, Any]:
        env_debug = str(os.getenv("ETSA_DEBUG", "0")).lower() in {"1", "true", "yes", "on"}
        debug = env_debug
        _print = (lambda *a, **k: print(*a, **k)) if debug else (lambda *a, **k: None)
        _pp = lambda o: pformat(o, width=100, compact=False)

        _print("== Build Graph: start ==")
        _print("Input text:\n" + text.strip())
        
        graph = RelationGraph(text, sentiment_analyzer_system=self.sentiment_system)
        
        try:
            clusters_dict = self.entity_consolidator.analyze(text)
            clusters = {}
            for idx, (canonical_name, cluster_data) in enumerate(clusters_dict.items()):
                clusters[idx] = {
                    "canonical_name": canonical_name,
                    "entity_references": cluster_data.get("entity_references", [])
                }
        except Exception as e:
            logger.error(f"Entity consolidation failed: {e}")
            clusters = {}

        _print("Coref clusters:")
        _print(_pp(clusters))

        try:
            clauses = self.clause_splitter.split(text)
            if not clauses:
                clauses = [text]
        except Exception as e:
            logger.error(f"Clause splitting failed: {e}")
            clauses = [text]

        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        _print(f"Split into {len(sentences)} sentence(s)")
        
        sent_map = {}
        clause_positions = []
        current_pos = 0
        
        for i, clause in enumerate(clauses):
            clause_stripped = clause.strip().rstrip(',').rstrip()
            try:
                start_pos = text.index(clause_stripped, current_pos)
                end_pos = start_pos + len(clause_stripped)
                clause_positions.append((start_pos, end_pos))
                current_pos = end_pos
            except ValueError:
                start_pos = current_pos
                end_pos = current_pos + len(clause)
                clause_positions.append((start_pos, end_pos))
                current_pos = end_pos + 1
                
            clause_entities = []
            for cid, cdata in clusters.items():
                name = cdata.get("canonical_name")
                if not name:
                    continue
                entity_references = cdata.get("entity_references", [])
                clause_start, clause_end = clause_positions[i]
                
                mention_in_clause = any(
                    clause_start <= span[0] and span[1] <= clause_end
                    for _, span in entity_references
                )
                
                if mention_in_clause:
                    clause_entities.append((name, (clause_start, clause_end)))
            
            sent_map[f"clause_{i}"] = {"entities": clause_entities}
        
        for i in range(len(sent_map)):
            _print(f"\n-- Clause {i} --")
            clause_key = f"clause_{i}"
            clause_text = clauses[i] if i < len(clauses) else ""
            if clause_text and not clause_text.endswith(('.', '!', '?')):
                clause_text += '.'
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

            entity_modifiers = {}
            for entity_name in entity_headwords:
                try:
                    modifier_result = self.modifier_extractor.extract(clause_text, entity_name)
                    modifier_output = modifier_result.get("modifiers", []) if modifier_result else []
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

            try:
                relation_result = self.relation_extractor.extract(clause_text, entity_headwords)
                relations_found = relation_result.get("relations", []) if relation_result else []
            except Exception as e:
                _print(f"Relation extraction failed: {e}")
                relations_found = []
                
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
        
        if self.mode != "no_formulas":
            graph.run_compound_sentiment_calculations(
                self.actor_func, self.target_func, self.assoc_func, 
                self.parent_func, self.child_func
            )
            _print("Ran compound sentiment calculations (action, association, belonging)")
        
        results = graph.run_aggregate_sentiment_calculations(self.agg_func)
        
        _print("== Build Graph: done ==")

        pipeline_trace = {
            "text": text,
            "clauses": clauses,
            "clusters": {str(cid): data for cid, data in clusters.items()},
            "graph_nodes": [
                (node, data) for node, data in graph.graph.nodes(data=True)
            ],
            "graph_edges": [
                (u, v, data) for u, v, data in graph.graph.edges(data=True)
            ],
            "final_sentiments": results,
            "execution_trace": {
                "modules": {
                    "ner_coref": {"status": "success", "clusters_found": len(clusters)},
                    "clause_splitter": {"status": "success", "clauses_generated": len(clauses)},
                    "graph_creation": {"status": "success", "nodes_created": len(graph.graph.nodes), "edges_created": len(graph.graph.edges)},
                    "sentiment_calculation": {"status": "success", "entities_analyzed": len(results)}
                }
            }
        }

        if debug_run_name:
            os.makedirs(os.path.dirname(debug_run_name), exist_ok=True)
            html_path = f"{debug_run_name}_graph.html"
            visualizer = GraphVisualizer(graph, clusters)
            visualizer.draw_graph(save_path=html_path)
            pipeline_trace["graph_html_path"] = html_path

        return pipeline_trace

def build_graph_with_optimal_functions(text: str, debug: bool = False) -> tuple[RelationGraph, dict]:
    from .sentiment_systems import PresetEnsembleSystem
    from .ner_coref import EntityConsolidator
    from .clause_splitter import SpacyClauseSplitter
    from .re_e import SpacyRelationExtractor
    from .modifier_e import SpacyModifierExtractor
    
    pipeline = Pipeline(
        entity_consolidator=EntityConsolidator(),
        clause_splitter=SpacyClauseSplitter(),
        relation_extractor=SpacyRelationExtractor(),
        modifier_extractor=SpacyModifierExtractor(),
        sentiment_system=PresetEnsembleSystem(),
        mode="full_stack"
    )
    
    result = pipeline.run(text)
    return result
