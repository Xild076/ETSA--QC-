import os
import sys
import json
import logging
import numpy as np
from typing import Dict, Any, List

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

def _names_match(name1: str, name2: str) -> bool:
    if not name1 or not name2:
        return False
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()
    return n1 == n2 or n1 in n2 or n2 in n1

def _find_cluster_id_for_name(clusters: Dict[int, Dict[str, Any]], name: str) -> int | None:
    target = (name or "").lower().strip()
    for cid, data in clusters.items():
        if data.get("canonical_name", "").lower().strip() == target:
            return cid
        for mention, _ in data.get("entity_references", []):
            if (mention or "").lower().strip() == target:
                return cid
    return None

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
            self.actor_func = lambda x: np.mean(x)
            self.target_func = lambda x: np.mean(x)
            self.assoc_func = lambda x: np.mean(x)
            self.parent_func = lambda x: np.mean(x)
            self.child_func = lambda x: np.mean(x)
            self.agg_func = lambda scores: np.mean(scores) if scores else 0.0
        else:
            self.actor_func = get_actor_function()
            self.target_func = get_target_function()
            self.assoc_func = get_association_function()
            self.parent_func = get_parent_function()
            self.child_func = get_child_function()
            self.agg_func = get_aggregate_function()

    def run(self, text: str, debug_run_name: str = None) -> Dict[str, Any]:
        execution_trace = {
            "modules": {},
            "stages": []
        }
        
        try:
            clusters_dict = self.entity_consolidator.analyze(text)
            clusters = {}
            for idx, (canonical_name, cluster_data) in enumerate(clusters_dict.items()):
                clusters[str(idx)] = {
                    "canonical_name": canonical_name,
                    "entity_references": cluster_data.get("entity_references", [])
                }
            
            execution_trace["modules"]["ner_coref"] = {
                "status": "success",
                "method": "entity_consolidator",
                "clusters_found": len(clusters)
            }
        except Exception as e:
            logger.error(f"Entity consolidation failed: {e}")
            clusters = {}
            execution_trace["modules"]["ner_coref"] = {
                "status": "failed",
                "method": "entity_consolidator",
                "error": str(e),
                "clusters_found": 0
            }

        execution_trace["stages"].append({
            "stage": "ner_coref",
            "output": {"clusters": clusters, "clause_mapping": {}}
        })

        try:
            clauses = self.clause_splitter.split(text)
            if not clauses:
                clauses = [text]
            execution_trace["modules"]["clause_splitter"] = {
                "status": "success",
                "clauses_generated": len(clauses)
            }
        except Exception as e:
            clauses = [text]
            execution_trace["modules"]["clause_splitter"] = {
                "status": "fallback",
                "error": str(e),
                "clauses_generated": len(clauses)
            }

        execution_trace["stages"].append({
            "stage": "clause_splitting",
            "output": {"clauses": clauses}
        })

        graph = RelationGraph(text, self.sentiment_system)
        execution_trace["stages"].append({
            "stage": "graph_initialization",
            "output": {"graph_created": True}
        })

        nodes_created = 0
        node_creation_details = []
        
        clause_positions = []
        current_pos = 0
        for clause in clauses:
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
        
        for i, clause in enumerate(clauses):
            clause_start, clause_end = clause_positions[i]

            for cid, cdata in clusters.items():
                name = cdata.get("canonical_name")
                if not name: 
                    continue
                
                entity_references = cdata.get("entity_references", [])
                
                mention_in_clause = any(
                    clause_start <= span[0] and span[1] <= clause_end
                    for _, span in entity_references
                )
                
                if mention_in_clause:
                    try:
                        modifier_data = self.modifier_extractor.extract(clause, name)
                        mods = modifier_data.get("modifiers", [])
                        
                        if mods:
                            entity_words = set(name.lower().split())
                            cleaned_mods = []
                            for mod in mods:
                                mod_words = set(mod.lower().split())
                                if not mod_words.issubset(entity_words) and mod.strip():
                                    cleaned_mods.append(mod.strip())
                            
                            if cleaned_mods:
                                entity_text = name + ", " + ", ".join(cleaned_mods)
                            else:
                                entity_text = name
                        else:
                            entity_text = name
                        
                        init_sentiment = self.sentiment_system.analyze(entity_text)
                        entity_role = "associate"
                        
                        graph.add_entity_node(
                            id=int(cid), 
                            head=name, 
                            modifier=mods,
                            entity_role=entity_role,
                            clause_layer=i
                        )
                        
                        graph.graph.nodes[f"{cid}_{i}"]["init_sentiment"] = init_sentiment
                        
                        nodes_created += 1
                        node_creation_details.append({
                            "cluster_id": cid,
                            "name": name,
                            "clause_index": i,
                            "modifiers": mods,
                            "initial_sentiment": init_sentiment,
                            "entity_text": entity_text
                        })
                        
                    except Exception as e:
                        logger.error(f"Node creation failed for {name}: {e}")
        
        execution_trace["modules"]["node_creation"] = {
            "status": "success",
            "nodes_created": nodes_created,
            "details": node_creation_details
        }
        execution_trace["stages"].append({
            "stage": "node_creation",
            "output": {"nodes_created": nodes_created, "details": node_creation_details}
        })
        
        edges_created = 0
        edge_creation_details = []
        
        for i, clause in enumerate(clauses):
            clause_entities = []
            entity_name_to_id = {}
            
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
                    clause_entities.append(name)
                    entity_name_to_id[name] = int(cid)
            
            if len(clause_entities) > 0:
                try:
                    relation_result = self.relation_extractor.extract(clause, clause_entities)
                    relations = relation_result.get("relations", [])
                    
                    for relation in relations:
                        subj_name = relation.get("subject", {}).get("head", "")
                        obj_name = relation.get("object", {}).get("head", "")
                        rel_type = relation.get("relation", {}).get("type", "")
                        rel_text = relation.get("relation", {}).get("text", "")
                        
                        subj_id = None
                        obj_id = None
                        
                        for entity_name, entity_id in entity_name_to_id.items():
                            if _names_match(entity_name, subj_name):
                                subj_id = entity_id
                            if _names_match(entity_name, obj_name):
                                obj_id = entity_id
                        
                        if subj_id is not None:
                            if rel_type == "action":
                                if obj_id is not None:
                                    graph.add_action_edge(subj_id, obj_id, i, "action", [rel_text])
                                else:
                                    graph.add_relation_edge(subj_id, subj_id, i, "self_action", rel_text)
                            elif rel_type == "association" and obj_id is not None:
                                graph.add_association_edge(subj_id, obj_id, i, rel_text)
                            elif rel_type == "belonging" and obj_id is not None:
                                graph.add_belonging_edge(subj_id, obj_id, i)
                        
                        edge_creation_details.append({
                            "subject": subj_name,
                            "object": obj_name,
                            "relation_type": rel_type,
                            "subject_id": subj_id,
                            "object_id": obj_id,
                            "clause_index": i
                        })
                        edges_created += 1
                        
                except Exception as e:
                    logger.error(f"Relation extraction failed for clause {i}: {e}")
        
        for cid in clusters.keys():
            entity_id = int(cid)
            graph.add_temporal_edge(entity_id)
        
        execution_trace["modules"]["edge_creation"] = {
            "status": "success",
            "edges_created": edges_created,
            "details": edge_creation_details
        }
        execution_trace["stages"].append({
            "stage": "edge_creation",
            "output": {"edges_created": edges_created, "details": edge_creation_details}
        })
        
        if self.mode != "no_formulas":
            graph.run_compound_sentiment_calculations(
                self.actor_func, 
                self.target_func, 
                self.assoc_func, 
                self.parent_func, 
                self.child_func
            )
        
        final_sentiments = graph.run_aggregate_sentiment_calculations(self.agg_func)
        
        execution_trace["modules"]["sentiment_calculation"] = {
            "status": "success",
            "entities_analyzed": len(final_sentiments)
        }
        execution_trace["stages"].append({
            "stage": "sentiment_calculation",
            "output": {"final_sentiments": final_sentiments}
        })

        pipeline_trace = {
            "text": text,
            "clauses": clauses,
            "clusters": {cid: data for cid, data in clusters.items()},
            "graph_nodes": [
                (node, data) for node, data in graph.graph.nodes(data=True)
            ],
            "graph_edges": [
                (u, v, data) for u, v, data in graph.graph.edges(data=True)
            ],
            "final_sentiments": final_sentiments,
            "execution_trace": execution_trace
        }

        if debug_run_name:
            os.makedirs(os.path.dirname(debug_run_name), exist_ok=True)
            html_path = f"{debug_run_name}_graph.html"
            visualizer = GraphVisualizer(graph, clusters)
            visualizer.draw_graph(save_path=html_path)
            pipeline_trace["graph_html_path"] = html_path

        return pipeline_trace
