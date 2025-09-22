from collections import defaultdict
import os
import re
from pprint import pformat
from typing import Dict, List, Tuple, Any, Optional
import sys
from pathlib import Path
_base = Path(__file__).resolve()
_src = _base.parents[1]
_root = _src.parent
for p in (str(_src), str(_root)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.pipeline.graph import RelationGraph
    from src.pipeline.relation_e import RelationExtractor, GemmaRelationExtractor, SpacyRelationExtractor
    from src.pipeline.modifier_e import ModifierExtractor, GemmaModifierExtractor, SpacyModifierExtractor
    from src.pipeline.ner_coref_s import ATE, HybridAspectExtractor
    from src.pipeline.clause_s import ClauseSplitter, BeneparClauseSplitter, NLTKSentenceSplitter
    from src.pipeline.sentiment_model import (
        SentimentModel,
        ActionSentimentModel,
        AssociationSentimentModel,
        BelongingSentimentModel,
        AggregateSentimentModel,
        DummySentimentModel,
        DuoDummySentimentModel
    )
    from src.pipeline.sentiment_analysis import (
        SentimentAnalysis,
        VADERSentimentAnalysis,
        TextBlobSentimentAnalysis,
        FlairSentimentAnalysis,
        PysentimientoSentimentAnalysis,
        DummySentimentAnalysis,
        MultiSentimentAnalysis,
    )
except ImportError:
    from graph import RelationGraph
    from relation_e import RelationExtractor, GemmaRelationExtractor, SpacyRelationExtractor
    from modifier_e import ModifierExtractor, GemmaModifierExtractor, SpacyModifierExtractor
    from ner_coref_s import ATE, HybridAspectExtractor
    from clause_s import ClauseSplitter, BeneparClauseSplitter, NLTKSentenceSplitter
    from sentiment_model import (
        SentimentModel,
        ActionSentimentModel,
        AssociationSentimentModel,
        BelongingSentimentModel,
        AggregateSentimentModel,
        DummySentimentModel,
        DuoDummySentimentModel
    )
    from sentiment_analysis import (
        SentimentAnalysis,
        VADERSentimentAnalysis,
        TextBlobSentimentAnalysis,
        FlairSentimentAnalysis,
        PysentimientoSentimentAnalysis,
        DummySentimentAnalysis,
        MultiSentimentAnalysis
    )

class SentimentPipeline:
    def __init__(self,
                 clause_splitter:ClauseSplitter,
                 aspect_extractor:ATE,
                 modifier_extractor:ModifierExtractor,
                 relation_extractor:RelationExtractor,
                 sentiment_analysis:SentimentAnalysis,
                 action_sentiment_model:SentimentModel,
                 association_sentiment_model:SentimentModel,
                 belonging_sentiment_model:SentimentModel,
                 aggregate_sentiment_model:SentimentModel):
        self.clause_splitter = clause_splitter
        self.aspect_extractor = aspect_extractor
        self.modifier_extractor = modifier_extractor
        self.relation_extractor = relation_extractor
        self.sentiment_analysis = sentiment_analysis
        self.action_sentiment_model = action_sentiment_model
        self.association_sentiment_model = association_sentiment_model
        self.belonging_sentiment_model = belonging_sentiment_model
        self.aggregate_sentiment_model = aggregate_sentiment_model

    def process(self, text: str, debug: bool = False) -> Dict[str, Any]:
        clauses = self.clause_splitter.split(text)
        aspects = self.aspect_extractor.analyze(clauses)
        graph = RelationGraph(text, clauses, self.sentiment_analysis)
        for aspect in aspects:
            aspect = aspects[aspect]
            aspect_id = graph.get_new_unique_entity_id()
            for mention in aspect.get("mentions", []):
                mention_name = mention[0]
                mention_clause_index = mention[1]
                # Correct argument order: text first, then entity
                modifier_result = self.modifier_extractor.extract(clauses[mention_clause_index], mention_name)
                modifiers = modifier_result["modifiers"]
                sentiment_hint = modifier_result.get("sentiment_hint")
                graph.add_entity_node(aspect_id, mention_name, modifiers, "none", mention_clause_index, sentiment_hint)

        
        for clause_index, clause in enumerate(clauses):
            entities_in_clause = graph.get_entities_at_layer(clause_index)
            relations = self.relation_extractor.extract(clause, entities_in_clause)

            def match_head_entity(head: str) -> Optional[int]:
                for ent in entities_in_clause:
                    if ent.get("head") == head:
                        return ent.get("entity_id")
                return None

            for relation in relations:
                subject_head = relation.get("subject")
                object_head = relation.get("object")
                if not isinstance(subject_head, str) or not isinstance(object_head, str):
                    continue
                subject_id = match_head_entity(subject_head)
                object_id = match_head_entity(object_head)
                if subject_id is None or object_id is None:
                    continue
                relation_type = relation.get("relation", {}).get("type", "")

                if relation_type == "ACTION":
                    action_text = relation.get("relation", {}).get("text", "")
                    graph.add_action_edge(subject_id, object_id, clause_index, action_text, [])
                elif relation_type == "ASSOCIATION":
                    graph.add_association_edge(subject_id, object_id, clause_index)
                elif relation_type == "BELONGING":
                    graph.add_belonging_edge(subject_id, object_id, clause_index)

        graph.run_compound_action_sentiment_calculations(self.action_sentiment_model.calculate)
        graph.run_compound_association_sentiment_calculations(self.association_sentiment_model.calculate)
        graph.run_compound_belonging_sentiment_calculations(self.belonging_sentiment_model.calculate)

        entity_sentiments: Dict[int, Dict[str, Any]] = {}
        for entity_id in sorted(graph.entity_ids):
            aggregate_sentiment = graph.run_aggregate_sentiment_calculations(
                entity_id, self.aggregate_sentiment_model.calculate
            )

            mentions = graph.get_all_entity_mentions(entity_id)

            roles = []
            modifiers: List[str] = []
            for (eid, layer), node_data in graph.graph.nodes(data=True):
                if eid != entity_id:
                    continue
                role = node_data.get("entity_role") or "associate"
                if role not in roles:
                    roles.append(role)
                modifiers.extend(node_data.get("modifier", []) or [])

            unique_modifiers: List[str] = []
            seen_modifiers = set()
            for modifier in modifiers:
                normalized = modifier.strip()
                if not normalized:
                    continue
                key = normalized.lower()
                if key in seen_modifiers:
                    continue
                seen_modifiers.add(key)
                unique_modifiers.append(normalized)

            relation_counts: Dict[str, int] = defaultdict(int)
            relation_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            def _node_attr(node_key: Any, attr: str, default: Any = "") -> Any:
                if node_key in graph.graph.nodes:
                    return graph.graph.nodes[node_key].get(attr, default)
                return default
            for _, _, edge_data in graph.graph.edges(data=True):
                relation_type = edge_data.get("relation")
                if relation_type == "action":
                    actor = edge_data.get("actor")
                    target = edge_data.get("target")
                    head = edge_data.get("head", "")
                    actor_clause = _node_attr(actor, "clause_layer", None)
                    if isinstance(actor, tuple) and actor[0] == entity_id:
                        relation_counts["action_out"] += 1
                        relation_examples["action_out"].append({
                            "target": _node_attr(target, "head", ""),
                            "head": head,
                            "clause_index": actor_clause,
                        })
                    if isinstance(target, tuple) and target[0] == entity_id:
                        relation_counts["action_in"] += 1
                        relation_examples["action_in"].append({
                            "actor": _node_attr(actor, "head", ""),
                            "head": head,
                            "clause_index": _node_attr(target, "clause_layer", None),
                        })
                elif relation_type == "association":
                    entity1 = edge_data.get("entity1")
                    entity2 = edge_data.get("entity2")
                    if isinstance(entity1, tuple) and entity1[0] == entity_id:
                        relation_counts["association"] += 1
                        relation_examples["association"].append({
                            "other": _node_attr(entity2, "head", ""),
                            "clause_index": _node_attr(entity1, "clause_layer", None),
                        })
                    if isinstance(entity2, tuple) and entity2[0] == entity_id:
                        relation_counts["association"] += 1
                        relation_examples["association"].append({
                            "other": _node_attr(entity1, "head", ""),
                            "clause_index": _node_attr(entity2, "clause_layer", None),
                        })
                elif relation_type == "belonging":
                    parent = edge_data.get("parent")
                    child = edge_data.get("child")
                    if isinstance(parent, tuple) and parent[0] == entity_id:
                        relation_counts["belonging_parent"] += 1
                        relation_examples["belonging_parent"].append({
                            "child": _node_attr(child, "head", ""),
                            "clause_index": _node_attr(parent, "clause_layer", None),
                        })
                    if isinstance(child, tuple) and child[0] == entity_id:
                        relation_counts["belonging_child"] += 1
                        relation_examples["belonging_child"].append({
                            "parent": _node_attr(parent, "head", ""),
                            "clause_index": _node_attr(child, "clause_layer", None),
                        })

            entity_sentiments[entity_id] = {
                "mentions": mentions,
                "aggregate_sentiment": aggregate_sentiment,
                "roles": roles or ["associate"],
                "modifiers": unique_modifiers,
                "relation_counts": dict(relation_counts),
                "relation_examples": {k: v for k, v in relation_examples.items() if v},
            }
        # Return a structured dict expected by build_graph/print utilities
        return {
            "graph": graph,
            "clauses": clauses,
            "aggregate_results": entity_sentiments,
            "relations": [],
            "debug_messages": [],
        }


def build_graph(text: str, pipeline: Optional[SentimentPipeline] = None, debug: bool = False) -> Tuple[RelationGraph, Dict[str, Any]]:
    active_pipeline = pipeline or build_default_pipeline()
    result = active_pipeline.process(text, debug=debug)
    return result["graph"], result


def build_graph_with_optimal_functions(text: str, debug: bool = False) -> Tuple[RelationGraph, Dict[str, Any]]:
    pipeline = build_default_pipeline()
    return build_graph(text, pipeline=pipeline, debug=debug)


def interpret_pipeline_output(result: Dict[str, Any]) -> str:
    clauses = result.get("clauses") or []
    aggregate = result.get("aggregate_results") or {}
    relations = result.get("relations") or []
    lines: List[str] = []

    if not aggregate:
        lines.append("No entities detected; nothing to report.")
    else:
        lines.append("=== Entity Sentiment Summary ===")
        sorted_entities = sorted(
            aggregate.items(),
            key=lambda item: item[1].get("aggregate_sentiment", 0.0),
            reverse=True,
        )
        for label, data in sorted_entities:
            aggregate_value = data.get("aggregate_sentiment", 0.0)
            roles = data.get("roles") or ["associate"]
            lines.append(f"- {label}: {aggregate_value:+.3f} [{', '.join(roles)}]")

            modifiers = data.get("modifiers") or []
            if modifiers:
                preview = ", ".join(modifiers[:3])
                if len(modifiers) > 3:
                    preview += ", ..."
                lines.append(f"    modifiers: {preview}")

            mentions = data.get("mentions") or []
            if mentions:
                mention_preview = ", ".join(
                    f"{m.get('text', '')}@{m.get('clause_index')}" for m in mentions[:3]
                )
                if len(mentions) > 3:
                    mention_preview += ", ..."
                lines.append(f"    mentions: {mention_preview}")

            rel_counts = data.get("relation_counts") or {}
            counts_summary = ", ".join(
                f"{key.replace('_', ' ')}={value}" for key, value in rel_counts.items() if value
            )
            if counts_summary:
                lines.append(f"    relations: {counts_summary}")

            examples = data.get("relation_examples") or {}
            sample_action = examples.get("action_out") or examples.get("action_in") or []
            if sample_action:
                sample = sample_action[0]
                if "target" in sample:
                    lines.append(
                        f"    sample action: {label.split(' (ID')[0]} -> {sample.get('target')} via '{sample.get('head', '')}'"
                    )
                elif "actor" in sample:
                    lines.append(
                        f"    sample action: {sample.get('actor')} -> {label.split(' (ID')[0]} via '{sample.get('head', '')}'"
                    )

            association_example = examples.get("association") or []
            if association_example:
                lines.append(
                    f"    sample association: linked with {association_example[0].get('other')}"
                )

            belonging_parent_example = examples.get("belonging_parent") or []
            if belonging_parent_example:
                lines.append(
                    f"    sample parent link: {belonging_parent_example[0].get('child')}"
                )

            belonging_child_example = examples.get("belonging_child") or []
            if belonging_child_example:
                lines.append(
                    f"    sample child link: parent {belonging_child_example[0].get('parent')}"
                )

    if clauses:
        lines.append("")
        lines.append("Clauses:")
        for idx, clause in enumerate(clauses):
            lines.append(f"  [{idx}] {clause}")

    if relations:
        total_actions = sum(len(entry.get("output", {}).get("actions", []) or []) for entry in relations)
        total_associations = sum(len(entry.get("output", {}).get("associations", []) or []) for entry in relations)
        total_belongings = sum(len(entry.get("output", {}).get("belongings", []) or []) for entry in relations)
        lines.append("")
        lines.append("Relation Extraction Summary:")
        lines.append(
            f"  actions={total_actions}, associations={total_associations}, belongings={total_belongings}"
        )

    debug_messages = result.get("debug_messages") or []
    if debug_messages:
        lines.append("")
        lines.append(f"Debug messages captured: {len(debug_messages)}")

    return "\n".join(lines)


def print_pipeline_output(result: Dict[str, Any]) -> None:
    print(interpret_pipeline_output(result))


def build_default_pipeline():
    clause_splitter = BeneparClauseSplitter()
    aspect_extractor = HybridAspectExtractor()
    modifier_extractor = GemmaModifierExtractor()
    relation_extractor = GemmaRelationExtractor()
    sentiment_analysis = MultiSentimentAnalysis(["vader", "flair", "nlptown"], [0.33, 0.33, 0.34])
    action_sentiment_model = ActionSentimentModel()
    association_sentiment_model = AssociationSentimentModel()
    belonging_sentiment_model = BelongingSentimentModel()
    aggregate_sentiment_model = AggregateSentimentModel()
    return SentimentPipeline(
        clause_splitter=clause_splitter,
        aspect_extractor=aspect_extractor,
        modifier_extractor=modifier_extractor,
        relation_extractor=relation_extractor,
        sentiment_analysis=sentiment_analysis,
        action_sentiment_model=action_sentiment_model,
        association_sentiment_model=association_sentiment_model,
        belonging_sentiment_model=belonging_sentiment_model,
        aggregate_sentiment_model=aggregate_sentiment_model,
    )


def main():
    sample_text = "The food was amazing, but the service was trash."
    graph, result = build_graph_with_optimal_functions(sample_text)
    print_pipeline_output(result)
    return graph


if __name__ == "__main__":  # pragma: no cover
    main()
