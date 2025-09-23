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
        clauses = self.clause_splitter.split(text) or [text]
        raw_aspects = self.aspect_extractor.analyze(clauses) or {}
        graph = RelationGraph(text, clauses, self.sentiment_analysis)

        if isinstance(raw_aspects, dict):
            aspects: Dict[int, Dict[str, Any]] = {}
            for key, value in raw_aspects.items():
                if not isinstance(value, dict):
                    continue
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    idx = len(aspects) + 1
                aspects[idx] = value
        elif isinstance(raw_aspects, list):
            aspects = {idx: value for idx, value in enumerate(raw_aspects, start=1) if isinstance(value, dict)}
        else:
            aspects = {}

        debug_messages: List[str] = []
        entity_records: Dict[int, Dict[str, Any]] = {}

        def clamp_clause_index(index_value: Any) -> int:
            if not clauses:
                return 0
            try:
                idx = int(index_value)
            except (TypeError, ValueError):
                return 0
            return max(0, min(idx, len(clauses) - 1))

        for aspect_data in aspects.values():
            mentions = aspect_data.get('mentions', []) if isinstance(aspect_data, dict) else []
            cleaned_mentions: List[Dict[str, Any]] = []
            for mention in mentions:
                if not isinstance(mention, (list, tuple)) or len(mention) < 2:
                    continue
                mention_text, mention_clause = mention[0], mention[1]
                if not isinstance(mention_text, str):
                    continue
                clause_index = clamp_clause_index(mention_clause)
                cleaned_mentions.append({
                    'text': mention_text.strip(),
                    'clause_index': clause_index,
                })
            if not cleaned_mentions:
                continue

            aspect_id = graph.get_new_unique_entity_id()
            canonical_name = aspect_data.get('first_mention') if isinstance(aspect_data, dict) else None
            canonical_name = (canonical_name or cleaned_mentions[0]['text']).strip()

            entity_records[aspect_id] = {
                'label': canonical_name or f'Entity {aspect_id}',
                'mentions': [],
                'modifiers': set(),
                'roles': set(),
                'relation_counts': defaultdict(int),
                'relation_examples': defaultdict(list),
            }

            for mention_entry in cleaned_mentions:
                clause_index = mention_entry['clause_index']
                mention_text = mention_entry['text']
                clause_text = clauses[clause_index] if 0 <= clause_index < len(clauses) else text
                try:
                    modifier_payload = self.modifier_extractor.extract(clause_text, mention_text)
                except Exception as exc:  # pragma: no cover - defensive
                    modifier_payload = {}
                    debug_messages.append(
                        f"modifier_extractor failed for '{mention_text}' in clause {clause_index}: {exc}"
                    )
                modifiers: List[str] = []
                if isinstance(modifier_payload, dict):
                    mods = modifier_payload.get('modifiers', [])
                    if isinstance(mods, list):
                        modifiers = [m for m in mods if isinstance(m, str)]

                node_key = (aspect_id, clause_index)
                if graph.graph.has_node(node_key):
                    if modifiers:
                        try:
                            graph.add_entity_modifier(aspect_id, modifiers, clause_index)
                        except Exception as exc:  # pragma: no cover - defensive
                            debug_messages.append(
                                f"add_entity_modifier failed for entity {aspect_id} at clause {clause_index}: {exc}"
                            )
                else:
                    try:
                        graph.add_entity_node(aspect_id, mention_text, modifiers, 'associate', clause_index)
                    except Exception as exc:  # pragma: no cover - defensive
                        debug_messages.append(
                            f"add_entity_node failed for '{mention_text}' in clause {clause_index}: {exc}"
                        )
                        continue

                mention_record = {
                    'text': mention_text,
                    'clause_index': clause_index,
                }
                if modifiers:
                    mention_record['modifiers'] = modifiers
                entity_records[aspect_id]['mentions'].append(mention_record)
                entity_records[aspect_id]['modifiers'].update(modifiers)
                entity_records[aspect_id]['roles'].add('associate')

        relation_outputs: List[Dict[str, Any]] = []

        for clause_index, clause in enumerate(clauses):
            entities_in_clause = graph.get_entities_at_layer(clause_index)
            entity_heads = [entity.get('head') for entity in entities_in_clause if isinstance(entity, dict) and entity.get('head')]
            head_map: Dict[str, List[int]] = defaultdict(list)
            for entity in entities_in_clause:
                if not isinstance(entity, dict):
                    continue
                head = entity.get('head')
                entity_id = entity.get('entity_id')
                if not head or entity_id is None:
                    continue
                head_map[head].append(entity_id)

            try:
                relations_payload = self.relation_extractor.extract(clause, entity_heads)
            except Exception as exc:  # pragma: no cover - defensive
                relations_payload = {'relations': []}
                debug_messages.append(f'relation_extractor failed for clause {clause_index}: {exc}')

            relation_outputs.append({
                'clause_index': clause_index,
                'clause': clause,
                'entities': entity_heads,
                'output': relations_payload,
            })

            def match_head_entity(entry: Any) -> Optional[int]:
                if isinstance(entry, dict):
                    candidate = entry.get('head') or entry.get('text')
                elif isinstance(entry, str):
                    candidate = entry
                else:
                    candidate = None
                if not isinstance(candidate, str):
                    return None
                if candidate in head_map and head_map[candidate]:
                    return head_map[candidate][0]
                lowered = candidate.lower()
                for head_value, ids in head_map.items():
                    if isinstance(head_value, str) and head_value.lower() == lowered and ids:
                        return ids[0]
                return None

            relation_candidates: List[Dict[str, Any]] = []
            if isinstance(relations_payload, dict):
                for key in ('relations', 'actions', 'associations', 'belongings'):
                    value = relations_payload.get(key)
                    if isinstance(value, list):
                        relation_candidates.extend(rel for rel in value if isinstance(rel, dict))
            elif isinstance(relations_payload, list):
                relation_candidates = [rel for rel in relations_payload if isinstance(rel, dict)]

            for relation in relation_candidates:
                relation_info = relation.get('relation') if isinstance(relation.get('relation'), dict) else {}
                relation_type = (relation_info.get('type') or '').upper()
                relation_text = relation_info.get('text') if isinstance(relation_info.get('text'), str) else ''
                subject_id = match_head_entity(relation.get('subject'))
                object_id = match_head_entity(relation.get('object'))
                if subject_id is None or object_id is None:
                    continue

                if relation_type == 'ACTION':
                    try:
                        graph.add_action_edge(subject_id, object_id, clause_index, relation_text, [])
                    except Exception as exc:  # pragma: no cover - defensive
                        debug_messages.append(
                            f"add_action_edge failed for clause {clause_index} ({subject_id}->{object_id}): {exc}"
                        )
                    else:
                        for entity_id, role in ((subject_id, 'actor'), (object_id, 'target')):
                            try:
                                graph.set_entity_role(entity_id, role, clause_index)
                            except Exception:  # pragma: no cover - defensive
                                pass
                            record = entity_records.get(entity_id)
                            if record:
                                record['roles'].add(role)
                        if subject_id in entity_records:
                            entity_records[subject_id]['relation_counts']['action_out'] += 1
                            if len(entity_records[subject_id]['relation_examples']['action_out']) < 3:
                                entity_records[subject_id]['relation_examples']['action_out'].append({
                                    'target': entity_records.get(object_id, {}).get('label', str(object_id)),
                                    'head': relation_text,
                                    'clause_index': clause_index,
                                })
                        if object_id in entity_records:
                            entity_records[object_id]['relation_counts']['action_in'] += 1
                            if len(entity_records[object_id]['relation_examples']['action_in']) < 3:
                                entity_records[object_id]['relation_examples']['action_in'].append({
                                    'actor': entity_records.get(subject_id, {}).get('label', str(subject_id)),
                                    'head': relation_text,
                                    'clause_index': clause_index,
                                })

                if relation_type == 'ASSOCIATION':
                    try:
                        graph.add_association_edge(subject_id, object_id, clause_index)
                    except Exception as exc:  # pragma: no cover - defensive
                        debug_messages.append(
                            f"add_association_edge failed for clause {clause_index} ({subject_id}<->{object_id}): {exc}"
                        )
                    else:
                        for entity_id, other_id in ((subject_id, object_id), (object_id, subject_id)):
                            record = entity_records.get(entity_id)
                            other = entity_records.get(other_id)
                            if not record:
                                continue
                            record['relation_counts']['association'] += 1
                            if len(record['relation_examples']['association']) < 3:
                                record['relation_examples']['association'].append({
                                    'other': other.get('label', str(other_id)) if other else str(other_id),
                                    'clause_index': clause_index,
                                })

                if relation_type == 'BELONGING':
                    try:
                        graph.add_belonging_edge(subject_id, object_id, clause_index)
                    except Exception as exc:  # pragma: no cover - defensive
                        debug_messages.append(
                            f"add_belonging_edge failed for clause {clause_index} ({subject_id}->{object_id}): {exc}"
                        )
                    else:
                        try:
                            graph.set_entity_role(subject_id, 'parent', clause_index)
                        except Exception:  # pragma: no cover - defensive
                            pass
                        try:
                            graph.set_entity_role(object_id, 'child', clause_index)
                        except Exception:  # pragma: no cover - defensive
                            pass
                        parent_record = entity_records.get(subject_id)
                        child_record = entity_records.get(object_id)
                        if parent_record:
                            parent_record['roles'].add('parent')
                            parent_record['relation_counts']['belonging_parent'] += 1
                            if len(parent_record['relation_examples']['belonging_parent']) < 3:
                                parent_record['relation_examples']['belonging_parent'].append({
                                    'child': child_record.get('label', str(object_id)) if child_record else str(object_id),
                                    'clause_index': clause_index,
                                })
                        if child_record:
                            child_record['roles'].add('child')
                            child_record['relation_counts']['belonging_child'] += 1
                            if len(child_record['relation_examples']['belonging_child']) < 3:
                                child_record['relation_examples']['belonging_child'].append({
                                    'parent': parent_record.get('label', str(subject_id)) if parent_record else str(subject_id),
                                    'clause_index': clause_index,
                                })

        try:
            graph.run_compound_action_sentiment_calculations(self.action_sentiment_model.calculate)
        except Exception as exc:  # pragma: no cover - defensive
            debug_messages.append(f'compound action sentiment calculation failed: {exc}')

        try:
            graph.run_compound_association_sentiment_calculations(self.association_sentiment_model.calculate)
        except Exception as exc:  # pragma: no cover - defensive
            debug_messages.append(f'compound association sentiment calculation failed: {exc}')

        try:
            graph.run_compound_belonging_sentiment_calculations(self.belonging_sentiment_model.calculate)
        except Exception as exc:  # pragma: no cover - defensive
            debug_messages.append(f'compound belonging sentiment calculation failed: {exc}')

        for (entity_id, _), node_data in graph.graph.nodes(data=True):
            record = entity_records.get(entity_id)
            if not record:
                continue
            role = node_data.get('entity_role')
            if isinstance(role, str):
                record['roles'].add(role)
            modifiers = node_data.get('modifier', [])
            if isinstance(modifiers, list):
                record['modifiers'].update(m for m in modifiers if isinstance(m, str))

        aggregate_results: Dict[int, Dict[str, Any]] = {}
        entity_sentiments: Dict[int, Dict[str, Any]] = {}

        for entity_id, record in entity_records.items():
            try:
                aggregate_sentiment = graph.run_aggregate_sentiment_calculations(
                    entity_id, self.aggregate_sentiment_model.calculate
                )
            except Exception as exc:  # pragma: no cover - defensive
                aggregate_sentiment = 0.0
                debug_messages.append(f'aggregate sentiment calculation failed for entity {entity_id}: {exc}')

            mentions = record['mentions']
            modifiers = sorted(record['modifiers']) if record['modifiers'] else []
            roles = sorted(record['roles']) if record['roles'] else ['associate']
            relation_counts = {k: v for k, v in record['relation_counts'].items() if v}
            relation_examples = {k: v for k, v in record['relation_examples'].items() if v}

            aggregate_results[entity_id] = {
                'label': record['label'],
                'aggregate_sentiment': aggregate_sentiment,
                'mentions': mentions,
                'modifiers': modifiers,
                'roles': roles,
                'relation_counts': relation_counts,
                'relation_examples': relation_examples,
            }

            entity_sentiments[entity_id] = {
                'label': record['label'],
                'mentions': mentions,
                'sentiment': aggregate_sentiment,
                'roles': roles,
                'modifiers': modifiers,
                'relation_counts': relation_counts,
                'relation_examples': relation_examples,
            }

        return {
            'text': text,
            'clauses': clauses,
            'aspects': aspects,
            'graph': graph,
            'entity_sentiments': entity_sentiments,
            'aggregate_results': aggregate_results,
            'relations': relation_outputs,
            'debug_messages': debug_messages,
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
        for entity_id, data in sorted_entities:
            label = data.get("label") or f"Entity {entity_id}"
            display_name = f"{label} (ID {entity_id})"
            aggregate_value = data.get("aggregate_sentiment", 0.0)
            roles = data.get("roles") or ["associate"]
            lines.append(f"- {display_name}: {aggregate_value:+.3f} [{', '.join(roles)}]")

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
                        f"    sample action: {label} -> {sample.get('target')} via '{sample.get('head', '')}'"
                    )
                elif "actor" in sample:
                    lines.append(
                        f"    sample action: {sample.get('actor')} -> {label} via '{sample.get('head', '')}'"
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
    sentiment_analysis = MultiSentimentAnalysis(["flair", "pysentimiento", "vader"], [0.33, 0.33, 0.34])
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
