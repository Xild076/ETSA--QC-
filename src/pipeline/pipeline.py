from collections import defaultdict
import json
import os
import re
from pprint import pformat
from typing import Dict, List, Tuple, Any, Optional
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


_base = Path(__file__).resolve()
_src = _base.parents[1]
_root = _src.parent
for p in (str(_base.parent), str(_src), str(_root)):
    if p not in sys.path:
        sys.path.insert(0, p)

from graph import RelationGraph
from relation_e import RelationExtractor, GemmaRelationExtractor
from modifier_e import ModifierExtractor, GemmaModifierExtractor, DummyModifierExtractor
from ner_coref_s import ATE, HybridAspectExtractor
from clause_s import ClauseSplitter, BeneparClauseSplitter
from sentiment_model import (
    SentimentModel,
    ActionSentimentModel,
    AssociationSentimentModel,
    BelongingSentimentModel,
    AggregateSentimentModel
)
from sentiment_analysis import MultiSentimentAnalysis
from cache_manager import PipelineCache


class SentimentPipeline:
    def __init__(self,
                 clause_splitter:ClauseSplitter,
                 aspect_extractor:ATE,
                 modifier_extractor:ModifierExtractor,
                 relation_extractor:RelationExtractor,
                 sentiment_analysis:Any,
                 action_sentiment_model:SentimentModel,
                 association_sentiment_model:SentimentModel,
                 belonging_sentiment_model:SentimentModel,
                 aggregate_sentiment_model:SentimentModel,
                 combiner: str = "contextual_v3",
                 combiner_params: Optional[Dict[str, Any]] = None,
                 use_cache: bool = True):
        self.clause_splitter = clause_splitter
        self.aspect_extractor = aspect_extractor
        self.modifier_extractor = modifier_extractor
        self.relation_extractor = relation_extractor
        self.sentiment_analysis = sentiment_analysis
        self.action_sentiment_model = action_sentiment_model
        self.association_sentiment_model = association_sentiment_model
        self.belonging_sentiment_model = belonging_sentiment_model
        self.aggregate_sentiment_model = aggregate_sentiment_model
        self.combiner = combiner
        self.combiner_params = combiner_params or {}
        self.use_cache = use_cache
        self.cache = PipelineCache() if use_cache else None

    def _run_full_processing(self, text: str) -> Dict[str, Any]:
        clauses = self.clause_splitter.split(text) or [text]
        raw_aspects = self.aspect_extractor.analyze(clauses) or {}
        graph = RelationGraph(text, clauses, self.sentiment_analysis, self.combiner, self.combiner_params)

        aspects = {}
        if isinstance(raw_aspects, dict):
            for key, value in raw_aspects.items():
                if not isinstance(value, dict): continue
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    idx = len(aspects) + 1
                aspects[idx] = value
        elif isinstance(raw_aspects, list):
            aspects = {idx: val for idx, val in enumerate(raw_aspects, 1) if isinstance(val, dict)}

        debug_messages = []
        entity_records = {}

        def clamp_clause_index(index_value: Any) -> int:
            if not clauses: return 0
            try:
                idx = int(index_value)
            except (TypeError, ValueError):
                return 0
            return max(0, min(idx, len(clauses) - 1))

        for aspect_data in aspects.values():
            mentions = aspect_data.get('mentions', [])
            cleaned_mentions = []
            for mention in mentions:
                if not isinstance(mention, (list, tuple)) or len(mention) < 2: continue
                mention_text, mention_clause = mention[0], mention[1]
                if not isinstance(mention_text, str): continue
                cleaned_mentions.append({'text': mention_text.strip(), 'clause_index': clamp_clause_index(mention_clause)})
            
            if not cleaned_mentions: continue
            
            aspect_id = graph.get_new_unique_entity_id()
            canonical_name = (aspect_data.get('first_mention') or cleaned_mentions[0]['text']).strip()
            
            entity_records[aspect_id] = {
                'label': canonical_name or f'Entity {aspect_id}', 'mentions': [], 'modifiers': set(),
                'roles': set(), 'relation_counts': defaultdict(int), 'relation_examples': defaultdict(list),
            }

            for mention_entry in cleaned_mentions:
                clause_index, mention_text = mention_entry['clause_index'], mention_entry['text']
                clause_text = clauses[clause_index] if 0 <= clause_index < len(clauses) else text
                
                try:
                    mods_payload = self.modifier_extractor.extract(clause_text, mention_text)
                    modifiers = mods_payload.get('modifiers', [])
                except Exception as exc:
                    modifiers = []
                    debug_messages.append(f"modifier_extractor failed for '{mention_text}': {exc}")
                
                node_key = (aspect_id, clause_index)
                if not graph.graph.has_node(node_key):
                    graph.add_entity_node(aspect_id, mention_text, modifiers, 'associate', clause_index)
                else:
                    if modifiers:
                        graph.add_entity_modifier(aspect_id, modifiers, clause_index)

                node_data = graph.graph.nodes.get(node_key, {})
                mention_record = {
                    'text': mention_text,
                    'clause_index': clause_index,
                    'modifiers': modifiers,
                    'head_sentiment': node_data.get('head_sentiment'),
                    'modifier_sentiment': node_data.get('modifier_sentiment'),
                    'modifier_context_sentiment': node_data.get('modifier_context_sentiment'),
                    'modifier_sentiment_components': node_data.get('modifier_sentiment_components', {}),
                }
                entity_records[aspect_id]['mentions'].append(mention_record)
                entity_records[aspect_id]['modifiers'].update(modifiers)
                entity_records[aspect_id]['roles'].add('associate')

        relation_outputs = []
        for clause_index, clause in enumerate(clauses):
            entities_in_clause = graph.get_entities_at_layer(clause_index)
            entity_heads = [e['head'] for e in entities_in_clause if e.get('head')]
            head_map = defaultdict(list)
            for entity in entities_in_clause:
                if entity.get('head') and entity.get('entity_id') is not None:
                    head_map[entity['head']].append(entity['entity_id'])

            try:
                relations_payload = self.relation_extractor.extract(clause, entity_heads)
            except Exception as exc:
                relations_payload = {'relations': []}
                debug_messages.append(f'relation_extractor failed for clause {clause_index}: {exc}')
            
            relation_outputs.append({'clause_index': clause_index, 'clause': clause, 'entities': entity_heads, 'output': relations_payload})
            
            def match_head_entity(entry: Any) -> Optional[int]:
                candidate = entry.get('head') if isinstance(entry, dict) else entry
                if isinstance(candidate, str) and candidate in head_map and head_map[candidate]:
                    return head_map[candidate][0]
                return None

            rels = relations_payload.get('relations', [])
            for rel in rels:
                rel_info = rel.get('relation', {})
                rel_type = (rel_info.get('type') or '').upper()
                rel_text = rel_info.get('text', '')
                sub_id, obj_id = match_head_entity(rel.get('subject')), match_head_entity(rel.get('object'))
                if sub_id is None or obj_id is None: continue

                if rel_type == 'ACTION':
                    graph.add_action_edge(sub_id, obj_id, clause_index, rel_text, [])
                elif rel_type == 'ASSOCIATION':
                    graph.add_association_edge(sub_id, obj_id, clause_index)
                elif rel_type == 'BELONGING':
                    graph.add_belonging_edge(sub_id, obj_id, clause_index)

        graph.run_compound_action_sentiment_calculations(self.action_sentiment_model.calculate)
        graph.run_compound_association_sentiment_calculations(self.association_sentiment_model.calculate)
        graph.run_compound_belonging_sentiment_calculations(self.belonging_sentiment_model.calculate)

        for (entity_id, _), node_data in graph.graph.nodes(data=True):
            if entity_id in entity_records:
                if 'entity_role' in node_data: entity_records[entity_id]['roles'].add(node_data['entity_role'])
                if 'modifier' in node_data: entity_records[entity_id]['modifiers'].update(node_data['modifier'])
        
        return {
            'graph': graph, 'clauses': clauses, 'aspects': aspects,
            'entity_records': entity_records, 'relation_outputs': relation_outputs,
            'debug_messages': debug_messages,
        }

    def process(self, text: str, debug: bool = False) -> Dict[str, Any]:
        if self.cache:
            cached_intermediate = self.cache.get_intermediate_results(text)
            if cached_intermediate:
                intermediate_results = cached_intermediate
            else:
                intermediate_results = self._run_full_processing(text)
                self.cache.store_intermediate_results(
                    text,
                    intermediate_results=intermediate_results,
                    include_graph=True,
                )
        else:
            intermediate_results = self._run_full_processing(text)

        graph = intermediate_results.get('graph') or intermediate_results.get('_graph')
        if graph is None:
            raise RuntimeError("Cached intermediate results are missing the relation graph")

        # Normalize entity record containers (sets survived serialization as lists).
        entity_records = {}
        for entity_id, record in intermediate_results['entity_records'].items():
            normalized = dict(record)
            normalized['modifiers'] = set(record.get('modifiers', []))
            normalized['roles'] = set(record.get('roles', []))
            try:
                entity_key = int(entity_id)
            except (TypeError, ValueError):
                entity_key = entity_id
            entity_records[entity_key] = normalized

        for entity_id, record in entity_records.items():
            for mention in record.get('mentions', []):
                clause_index = mention.get('clause_index')
                if clause_index is None:
                    continue
                node_key = (entity_id, clause_index)
                if node_key not in graph.graph.nodes:
                    continue
                node_data = graph.graph.nodes[node_key]
                if 'head_sentiment' in mention and mention['head_sentiment'] is not None:
                    node_data['head_sentiment'] = mention['head_sentiment']
                if 'modifier_sentiment' in mention and mention['modifier_sentiment'] is not None:
                    node_data['modifier_sentiment'] = mention['modifier_sentiment']
                if 'modifier_context_sentiment' in mention and mention['modifier_context_sentiment'] is not None:
                    node_data['modifier_context_sentiment'] = mention['modifier_context_sentiment']
                if 'modifier_sentiment_components' in mention and mention['modifier_sentiment_components']:
                    node_data['modifier_sentiment_components'] = dict(mention['modifier_sentiment_components'])

        # Ensure the cached graph reflects the requested combiner configuration before aggregation.
        graph.refresh_with_combiner(
            self.combiner,
            self.combiner_params,
            self.action_sentiment_model.calculate,
            self.association_sentiment_model.calculate,
            self.belonging_sentiment_model.calculate,
        )

        intermediate_results['graph'] = graph
        clauses = intermediate_results['clauses']
        aspects = intermediate_results['aspects']
        relation_outputs = intermediate_results['relation_outputs']
        debug_messages = intermediate_results['debug_messages']

        aggregate_results = {}
        for entity_id, record in entity_records.items():
            try:
                aggregate_sentiment = graph.run_aggregate_sentiment_calculations(entity_id, self.aggregate_sentiment_model.calculate)
                record['aggregate_sentiment'] = aggregate_sentiment
                aggregate_results[entity_id] = record
            except Exception as exc:
                debug_messages.append(f'aggregate sentiment failed for entity {entity_id}: {exc}')

        return {
            'text': text, 'clauses': clauses, 'aspects': aspects, 'graph': graph,
            'entity_sentiments': aggregate_results, 'aggregate_results': aggregate_results,
            'relations': relation_outputs, 'debug_messages': debug_messages,
        }


def load_manual_combiner_config() -> Optional[Tuple[str, Dict[str, Any]]]:
    """Load manual combiner configuration from combiner_config.json in project root."""
    # Try project root first
    config_paths = [
        Path(__file__).resolve().parents[2] / "combiner_config.json",  # Project root
        Path("combiner_config.json"),  # Current directory
        Path(__file__).resolve().parent / "combiner_config.json",  # Pipeline directory
    ]
    
    for config_file in config_paths:
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('combiner_name'), data.get('combiner_params', {})
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load manual combiner config from {config_file}: {e}")
                continue
    return None

def build_default_pipeline(combiner: Optional[str] = None, combiner_params: Optional[Dict[str, Any]] = None):
    # If no specific combiner/params are requested, try to load manual configuration
    if combiner is None and combiner_params is None:
        manual_config = load_manual_combiner_config()
        if manual_config:
            logger.info("Loading manual combiner configuration from combiner_config.json")
            combiner, combiner_params = manual_config
    
    # Fallback to default if no manual config is found
    if combiner is None:
        combiner = "adaptive_v6"  # Changed from contextual_v3
    
    # Get optimal device for GPU acceleration
    try:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    except:
        device = "cpu"
        
    clause_splitter = BeneparClauseSplitter()
    aspect_extractor = HybridAspectExtractor(device=device)
    modifier_extractor = DummyModifierExtractor()
    relation_extractor = GemmaRelationExtractor()
    sentiment_analysis = MultiSentimentAnalysis(
        methods=['distilbert_logit', 'flair', 'pysentimiento', 'vader'],
        weights=[0.55, 0.2, 0.15, 0.1],
        use_adaptive_weighting=True
    )
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
        combiner=combiner,
        combiner_params=combiner_params or {} # Pass the loaded or specified params
    )

def interpret_pipeline_output(result: Dict[str, Any]) -> str:
    lines = []
    aggregate = result.get("aggregate_results") or {}
    if not aggregate:
        lines.append("No entities detected.")
        return "\n".join(lines)
    
    lines.append("=== Entity Sentiment Summary ===")
    sorted_entities = sorted(aggregate.items(), key=lambda item: item[1].get("aggregate_sentiment", 0.0), reverse=True)
    for entity_id, data in sorted_entities:
        label = data.get("label") or f"Entity {entity_id}"
        sentiment = data.get("aggregate_sentiment", 0.0)
        lines.append(f"- {label} (ID {entity_id}): {sentiment:+.3f}")
        modifiers = data.get("modifiers") or []
        if modifiers:
            lines.append(f"    Modifiers: {', '.join(sorted(list(modifiers)))}")
    return "\n".join(lines)


def print_pipeline_output(result: Dict[str, Any]) -> None:
    print(interpret_pipeline_output(result))

if __name__ == "__main__":
    sample_text = "The food was amazing, but the service was trash."
    pipeline = build_default_pipeline()
    result = pipeline.process(sample_text)
    print_pipeline_output(result)
