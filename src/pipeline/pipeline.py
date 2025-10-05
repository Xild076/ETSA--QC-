"""High-level orchestration for the sentiment and relation extraction pipeline."""

from collections import defaultdict
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


try:  # pragma: no cover - allows execution as package or script
    from .graph import RelationGraph
    from .relation_e import (
        RelationExtractor,
        GemmaRelationExtractor,
        SpacyRelationExtractor,
        DummyRelationExtractor,
    )
    from .modifier_e import (
        ModifierExtractor,
        GemmaModifierExtractor,
        DummyModifierExtractor,
        SpacyModifierExtractor,
    )
    from .ner_coref_s import ATE, HybridAspectExtractor
    from .clause_s import ClauseSplitter, BeneparClauseSplitter
    from .sentiment_model import (
        SentimentModel,
        ActionSentimentModel,
        AssociationSentimentModel,
        BelongingSentimentModel,
        AggregateSentimentModel,
    )
    from .sentiment_analysis import MultiSentimentAnalysis
    from .cache_manager import PipelineCache
except ImportError:  # pragma: no cover - fallback for direct execution
    from graph import RelationGraph
    from relation_e import (
        RelationExtractor,
        GemmaRelationExtractor,
        SpacyRelationExtractor,
        DummyRelationExtractor,
    )
    from modifier_e import (
        ModifierExtractor,
        GemmaModifierExtractor,
        DummyModifierExtractor,
        SpacyModifierExtractor,
    )
    from ner_coref_s import ATE, HybridAspectExtractor
    from clause_s import ClauseSplitter, BeneparClauseSplitter
    from sentiment_model import (
        SentimentModel,
        ActionSentimentModel,
        AssociationSentimentModel,
        BelongingSentimentModel,
        AggregateSentimentModel,
    )
    from sentiment_analysis import MultiSentimentAnalysis
    from cache_manager import PipelineCache


__all__ = [
    "SentimentPipeline",
    "build_default_pipeline",
    "interpret_pipeline_output",
    "print_pipeline_output",
    "visualize_graph",
]


class SentimentPipeline:
    """Run aspect, relation, and sentiment extraction end-to-end.

    The pipeline stitches together clause splitting, aspect extraction,
    modifier/relation detection, and sentiment models to produce a populated
    :class:`RelationGraph` plus aggregate sentiment summaries.
    """
    def __init__(
        self,
        clause_splitter: ClauseSplitter,
        aspect_extractor: ATE,
        modifier_extractor: ModifierExtractor,
        relation_extractor: RelationExtractor,
        sentiment_analysis: Any,
        action_sentiment_model: SentimentModel,
        association_sentiment_model: SentimentModel,
        belonging_sentiment_model: SentimentModel,
        aggregate_sentiment_model: SentimentModel,
        use_cache: bool = True,
    ):
        """Construct a reusable sentiment pipeline instance.

        Args:
            clause_splitter: Component responsible for breaking input text into
                manageable clauses.
            aspect_extractor: Aspect term extractor implementation.
            modifier_extractor: Modifier extractor associated with aspects.
            relation_extractor: Relation extractor linking aspects together.
            sentiment_analysis: Base sentiment analysis strategy used by the
                :class:`RelationGraph`.
            action_sentiment_model: Model that evaluates propagated action
                sentiments.
            association_sentiment_model: Model that evaluates association
                sentiments.
            belonging_sentiment_model: Model that evaluates belonging
                sentiments.
            aggregate_sentiment_model: Final aggregation model combining
                sentiments per entity.
            use_cache: Toggle for caching intermediate results between runs.
        """
        self.clause_splitter = clause_splitter
        self.aspect_extractor = aspect_extractor
        self.modifier_extractor = modifier_extractor
        self.relation_extractor = relation_extractor
        self.sentiment_analysis = sentiment_analysis
        self.action_sentiment_model = action_sentiment_model
        self.association_sentiment_model = association_sentiment_model
        self.belonging_sentiment_model = belonging_sentiment_model
        self.aggregate_sentiment_model = aggregate_sentiment_model
        self.use_cache = use_cache
        self.cache = PipelineCache() if use_cache else None
        self.cache_signature = {
            "sentiment_strategy": "context_snippet_v1",
            "modifier_prompt_version": "contextual_ordering_v2",
        }

    def _run_full_processing(self, text: str) -> Dict[str, Any]:
        """Execute the full pipeline without consulting the cache.

        Args:
            text: Raw text to process.

        Returns:
            A dictionary containing the populated graph, per-clause artifacts,
            extracted entity metadata, relation outputs, and debug messages.
        """
        clauses = self.clause_splitter.split(text) or [text]
        raw_aspects = self.aspect_extractor.analyze(clauses) or {}
        graph = RelationGraph(text, clauses, self.sentiment_analysis)

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
        """Process text through the pipeline, optionally using cached results.

        Args:
            text: Raw input sequence to analyse.
            debug: When ``True`` retain intermediate debug details. Currently
                used for parity with older interfaces.

        Returns:
            Structured pipeline output containing clauses, entities, the graph,
            aggregate sentiments, relations, and debug information.
        """
        if self.cache:
            cached_intermediate = self.cache.get_intermediate_results(text, self.cache_signature)
            if cached_intermediate:
                intermediate_results = cached_intermediate
            else:
                intermediate_results = self._run_full_processing(text)
                self.cache.store_intermediate_results(
                    text,
                    settings=self.cache_signature,
                    intermediate_results=intermediate_results,
                    include_graph=True,
                )
        else:
            intermediate_results = self._run_full_processing(text)

        graph = intermediate_results.get('graph') or intermediate_results.get('_graph')
        if graph is None:
            raise RuntimeError("Cached intermediate results are missing the relation graph")

                                                                                    
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
        graph.aggregate_sentiments.clear()
        graph.run_compound_action_sentiment_calculations(self.action_sentiment_model.calculate)
        graph.run_compound_association_sentiment_calculations(self.association_sentiment_model.calculate)
        graph.run_compound_belonging_sentiment_calculations(self.belonging_sentiment_model.calculate)

        for _, _, edge_data in graph.graph.edges(data=True):
            if edge_data.get("relation") == "action":
                action_node = edge_data.get("action")
                if action_node in graph.graph.nodes:
                    edge_data["init_sentiment"] = float(
                        graph.graph.nodes[action_node].get("init_sentiment", 0.0)
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
def build_default_pipeline(
    *, 
    use_cache: bool = True,
    ablate_modifiers: bool = False,
    ablate_relations: bool = False
) -> SentimentPipeline:
    """Build the default sentiment analysis pipeline.
    
    Args:
        use_cache: Whether to use caching for intermediate results
        ablate_modifiers: If True, use DummyModifierExtractor instead of GemmaModifierExtractor
        ablate_relations: If True, use DummyRelationExtractor instead of GemmaRelationExtractor
    
    Returns:
        Configured SentimentPipeline instance
    """
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
    
    if ablate_modifiers:
        logger.info("ABLATION: Using DummyModifierExtractor (modifiers disabled)")
        modifier_extractor = DummyModifierExtractor()
    else:
        try:
            modifier_extractor = GemmaModifierExtractor()
        except Exception as exc:
            logger.warning(f"Failed to initialize GemmaModifierExtractor: {exc}. Falling back to spaCy extractor.")
            try:
                modifier_extractor = SpacyModifierExtractor()
            except Exception as spa_exc:
                logger.warning(f"Failed to initialize SpacyModifierExtractor: {spa_exc}. Using dummy extractor.")
                modifier_extractor = DummyModifierExtractor()
    
    if ablate_relations:
        logger.info("🔬 ABLATION: Using DummyRelationExtractor (relations disabled)")
        relation_extractor = DummyRelationExtractor()
    else:
        relation_extractor = GemmaRelationExtractor()
    sentiment_analysis = MultiSentimentAnalysis(
        methods=['distilbert_logit', 'vader'],
        weights=[.7, .3] 
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
        use_cache=use_cache,
    )

def interpret_pipeline_output(result: Dict[str, Any]) -> str:
    """Return a human-readable summary from a pipeline processing result."""
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
    """Convenience wrapper that prints :func:`interpret_pipeline_output`."""
    print(interpret_pipeline_output(result))


def visualize_graph(
    graph,
    output_dir: str = "output/visualizations",
    prefix: str = "graph",
    layout: str = "compact",
    scale: float = 1.0,
    max_modifier_chars: int = 18,
    dpi: int = 300,
    combined: bool = True,
) -> Dict[str, Path]:
    """Generate modular visualization panels for the sentiment pipeline.

    This refactored version focuses on:
    - Compact, shrink-friendly typography and spacing
    - Consistent vertical alignment across panels
    - Minimal wasted whitespace and balanced margins
    - Optional scaling and layout future extension
    - Deterministic ordering (entity id ascending, clause ascending)
    - Accessibility: high-contrast edge strokes and clear font hierarchy

        Panels produced (independent PNGs):
            1. Entities & Clusters (entity summary)
            2. Initial Sentiments & Modifiers (per clause)
            3. Compound Sentiments (after relational propagation)
            4. Aggregate Sentiments (final collapses)
            5. Legend (reusable across figures)
            6. (Optional) Combined full pipeline figure (entities → initial → compound → aggregate + legend)

    Args:
        graph: RelationGraph instance.
        output_dir: Destination directory.
        prefix: Filename prefix.
        layout: Currently only 'compact'. Placeholder for future variants.
        scale: Global scale multiplier for figure sizing (1.0 = default).
        max_modifier_chars: Truncate modifier string for readability when shrinking.
        dpi: Output DPI (publication quality = 300+).
        combined: If ``True``, also produce a single integrated pipeline explainer figure.

    Returns:
        Mapping of panel key -> Path object.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
    import numpy as np
    import json
    from datetime import datetime
    from collections import defaultdict
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Publication settings
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['font.family'] = 'serif'
    base_font = 9 * scale
    plt.rcParams['font.size'] = base_font
    
    paths = {}
    
    # Color scheme
    sentiment_colors = {
        'positive': '#2ecc71',
        'negative': '#e74c3c',
        'neutral': '#95a5a6'
    }
    
    relation_colors = {
        'action': '#3498db',
        'association': '#9b59b6',
        'belonging': '#f39c12'
    }
    
    def get_sentiment_color(sentiment_value: float) -> str:
        if sentiment_value > 0.1:
            return sentiment_colors['positive']
        elif sentiment_value < -0.1:
            return sentiment_colors['negative']
        return sentiment_colors['neutral']

    def build_edge_label(data: Dict[str, Any]) -> str:
        rel = data.get('relation', '')
        if rel == 'action':
            # Prefer explicit head + modifiers
            head = data.get('action_head')
            modifiers = data.get('action_modifier') or []
            if not head and 'action' in data:
                # Attempt to look up action node head
                try:
                    action_node = data['action']
                    head = graph.graph.nodes[action_node].get('head')
                except Exception:
                    head = None
            parts = []
            if head:
                parts.append(head)
            if modifiers:
                parts.extend(modifiers)
            if parts:
                return " ".join(parts)[:30]
            return 'action'
        if rel == 'association':
            return 'association'
        if rel == 'belonging':
            return 'belongs'
        return rel or ''

    def draw_offset_arrow(ax, x1, y1, x2, y2, color, label=None):
        """Draw an arrow between node centers but trim so arrowheads meet circle edge, not overlap.
        """
        radius = 0.23 * scale  # node radius
        dx, dy = x2 - x1, y2 - y1
        dist = (dx**2 + dy**2) ** 0.5
        if dist == 0:
            return
        shrink = radius / dist
        start = (x1 + dx * shrink, y1 + dy * shrink)
        end = (x2 - dx * shrink, y2 - dy * shrink)
        arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=14 * scale,
                                 color=color, linewidth=1.4 * scale, alpha=0.8, zorder=3)
        ax.add_patch(arrow)
        if label:
            mid = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
            # Offset perpendicular for readability
            if dist != 0:
                nx, ny = -dy / dist, dx / dist
            else:
                nx, ny = 0, 0
            ax.text(mid[0] + nx * 0.08, mid[1] + ny * 0.08, label,
                    ha='center', va='center', fontsize=base_font * 0.85,
                    fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor=color, linewidth=0.8 * scale, alpha=0.9), zorder=4)
    
    # Collect data
    entity_data = defaultdict(lambda: {
        'heads': set(),
        'clause_indices': set(),
        'nodes': []
    })
    
    max_clause = 0
    for (entity_id, clause_idx), node_data in graph.graph.nodes(data=True):
        max_clause = max(max_clause, clause_idx)
        entity_data[entity_id]['heads'].add(node_data.get('head', f'E{entity_id}'))
        entity_data[entity_id]['clause_indices'].add(clause_idx)
        entity_data[entity_id]['nodes'].append((clause_idx, node_data))
    
    num_clauses = max_clause + 1
    num_entities = len(entity_data)
    
    if num_entities == 0:
        logger.warning("No entities found in graph")
        return paths
    
    sorted_entities = sorted(entity_data.items(), key=lambda x: x[0])
    
    # Build node positions for graphs - PREVENT OVERLAP
    # Each entity gets its own row, nodes are spaced horizontally by clause
    node_positions = {}
    entity_to_row = {}  # Map entity_id to its row position
    
    for row, (entity_id, data) in enumerate(sorted_entities):
        entity_to_row[entity_id] = row
        for clause_idx, node_data in data['nodes']:
            # x = clause, y = entity row (ensures vertical separation)
            node_positions[(entity_id, clause_idx)] = (clause_idx, row)
    
    # === PART 1: ENTITIES AND CLUSTERS ===
    entity_box_height = 1.05 * scale
    fig1, ax1 = plt.subplots(figsize=(3.2 * scale, (num_entities * entity_box_height) + 1.4 * scale))
    ax1.set_xlim(0, 1)
    # Provide a little more breathing room so top title & top box aren't clipped
    ax1.set_ylim(-0.3, num_entities - 0.7 + entity_box_height)
    ax1.axis('off')
    ax1.set_title('Entities & Clusters', fontsize=base_font * 1.8, fontweight='bold', pad=10 * scale)
    
    for row, (entity_id, data) in enumerate(sorted_entities):
        entity_label = list(data['heads'])[0] if data['heads'] else f'E{entity_id}'

        box = FancyBboxPatch(
            (0.05, row - 0.35), 0.90, 0.7,
            boxstyle="round,pad=0.06",
            facecolor='lavender', edgecolor='#222', linewidth=1.4 * scale, alpha=0.85
        )
        ax1.add_patch(box)

        ax1.text(0.5, row + 0.22, f'E{entity_id}', ha='center', va='center',
                 fontsize=base_font * 1.4, fontweight='bold', color='#102c6b')
        ax1.text(0.5, row + 0.04, entity_label, ha='center', va='center',
                 fontsize=base_font * 1.2, fontweight='bold')
        mentions_text = "Mentions: " + ", ".join(f"C{c}" for c in sorted(data['clause_indices']))
        ax1.text(0.5, row - 0.14, mentions_text, ha='center', va='center',
                 fontsize=base_font, style='italic')
    
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    path1 = output_path / f"{prefix}_1_entities_clusters.png"
    plt.savefig(path1, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    paths['entities_clusters'] = path1
    
    # === PART 2: INIT SENTIMENTS GRAPH ===
    lane_height = 1.0
    fig2, ax2 = plt.subplots(figsize=((num_clauses * 1.6 + 0.6) * scale, (num_entities * lane_height) * 1.4 * scale))
    ax2.set_xlim(-0.6, num_clauses - 0.4)
    ax2.set_ylim(-0.4, num_entities - 0.6 + lane_height)
    ax2.axis('off')
    ax2.set_title('Initial Sentiments & Modifiers', fontsize=base_font * 1.7, fontweight='bold', pad=8 * scale)
    
    # Clause dividers and labels
    for c in range(1, num_clauses):
        ax2.axvline(x=c - 0.5, color='#999', linestyle=':', linewidth=1.2 * scale, alpha=0.55)

    for c in range(num_clauses):
        ax2.text(c, num_entities - 0.05, f'C{c}', ha='center', va='bottom',
                 fontsize=base_font * 1.3, fontweight='bold', style='italic', color='#102c6b')
    
    # Draw nodes with init sentiments
    for row, (entity_id, data) in enumerate(sorted_entities):
        for clause_idx, node_data in data['nodes']:
            x, y = node_positions[(entity_id, clause_idx)]
            
            init_sent = node_data.get('init_sentiment', 0.0)
            if init_sent is None:
                init_sent = 0.0
            
            color = get_sentiment_color(init_sent)
            circle = Circle((x, y), 0.23 * scale, facecolor=color, edgecolor='#111',
                            linewidth=1.4 * scale, alpha=0.9, zorder=4)
            ax2.add_patch(circle)

            ax2.text(x, y, f'E{entity_id}', ha='center', va='center',
                     fontsize=base_font * 1.15, fontweight='bold', zorder=5)

            ax2.text(x + 0.26 * scale, y, f'{init_sent:+.2f}', ha='left', va='center',
                     fontsize=base_font, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                               edgecolor=color, linewidth=1.0 * scale, alpha=0.95))

            modifiers = node_data.get('modifier', [])
            if modifiers:
                mod_text = ', '.join(list(modifiers))
                if len(mod_text) > max_modifier_chars:
                    mod_text = mod_text[: max_modifier_chars - 1] + '…'
                ax2.text(x, y + 0.30 * scale, mod_text,
                         ha='center', va='bottom', fontsize=base_font * 0.9, style='italic',
                         bbox=dict(boxstyle='round,pad=0.12', facecolor='#fff8c6',
                                   edgecolor='#c8b100', linewidth=0.8 * scale, alpha=0.85))
    
    # Temporal connections
    for row, (entity_id, data) in enumerate(sorted_entities):
        sorted_clauses = sorted(data['clause_indices'])
        for i in range(len(sorted_clauses) - 1):
            c1, c2 = sorted_clauses[i], sorted_clauses[i + 1]
            if (entity_id, c1) in node_positions and (entity_id, c2) in node_positions:
                x1, y1 = node_positions[(entity_id, c1)]
                x2, y2 = node_positions[(entity_id, c2)]
                ax2.plot([x1, x2], [y1, y2], color='#555', linewidth=0.9 * scale, alpha=0.35, zorder=1)
    
    # Relations (with better arrow placement & labels)
    for (u_id, u_clause), (v_id, v_clause), edge_data in graph.graph.edges(data=True):
        if (u_id, u_clause) not in node_positions or (v_id, v_clause) not in node_positions:
            continue
        x1, y1 = node_positions[(u_id, u_clause)]
        x2, y2 = node_positions[(v_id, v_clause)]
        rel_type = edge_data.get('relation', 'unknown')
        if rel_type == 'temporal':
            continue
        rel_color = relation_colors.get(rel_type, 'gray')
        label = build_edge_label(edge_data)
        draw_offset_arrow(ax2, x1, y1, x2, y2, rel_color, label=label)
    
    plt.tight_layout()
    path2 = output_path / f"{prefix}_2_init_sentiments.png"
    plt.savefig(path2, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    paths['init_sentiments'] = path2
    
    # === PART 3: COMPOUND SENTIMENTS GRAPH ===
    fig3, ax3 = plt.subplots(figsize=((num_clauses * 1.6 + 0.6) * scale, (num_entities * lane_height) * 1.4 * scale))
    ax3.set_xlim(-0.6, num_clauses - 0.4)
    ax3.set_ylim(-0.4, num_entities - 0.6 + lane_height)
    ax3.axis('off')
    ax3.set_title('Compound Sentiments', fontsize=base_font * 1.7, fontweight='bold', pad=8 * scale)
    
    # Clause dividers and labels
    for c in range(1, num_clauses):
        ax3.axvline(x=c - 0.5, color='#999', linestyle=':', linewidth=1.2 * scale, alpha=0.55)

    for c in range(num_clauses):
        ax3.text(c, num_entities - 0.05, f'C{c}', ha='center', va='bottom',
                 fontsize=base_font * 1.3, fontweight='bold', style='italic', color='#102c6b')
    
    # Draw nodes with compound sentiments
    for row, (entity_id, data) in enumerate(sorted_entities):
        for clause_idx, node_data in data['nodes']:
            x, y = node_positions[(entity_id, clause_idx)]
            
            comp_sent = node_data.get('compound_sentiment', node_data.get('init_sentiment', 0.0))
            if comp_sent is None:
                comp_sent = node_data.get('init_sentiment', 0.0)
            
            color = get_sentiment_color(comp_sent)
            circle = Circle((x, y), 0.23 * scale, facecolor=color, edgecolor='#111',
                            linewidth=1.4 * scale, alpha=0.9, zorder=4)
            ax3.add_patch(circle)

            ax3.text(x, y, f'E{entity_id}', ha='center', va='center',
                     fontsize=base_font * 1.15, fontweight='bold', zorder=5)

            ax3.text(x + 0.26 * scale, y, f'{comp_sent:+.2f}', ha='left', va='center',
                     fontsize=base_font, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                               edgecolor=color, linewidth=1.0 * scale, alpha=0.95))
    
    # Temporal connections
    for row, (entity_id, data) in enumerate(sorted_entities):
        sorted_clauses = sorted(data['clause_indices'])
        for i in range(len(sorted_clauses) - 1):
            c1, c2 = sorted_clauses[i], sorted_clauses[i + 1]
            if (entity_id, c1) in node_positions and (entity_id, c2) in node_positions:
                x1, y1 = node_positions[(entity_id, c1)]
                x2, y2 = node_positions[(entity_id, c2)]
                ax3.plot([x1, x2], [y1, y2], color='#555', linewidth=0.9 * scale, alpha=0.35, zorder=1)
    
    # Relations (with better arrow placement & labels)
    for (u_id, u_clause), (v_id, v_clause), edge_data in graph.graph.edges(data=True):
        if (u_id, u_clause) not in node_positions or (v_id, v_clause) not in node_positions:
            continue
        x1, y1 = node_positions[(u_id, u_clause)]
        x2, y2 = node_positions[(v_id, v_clause)]
        rel_type = edge_data.get('relation', 'unknown')
        if rel_type == 'temporal':
            continue
        rel_color = relation_colors.get(rel_type, 'gray')
        label = build_edge_label(edge_data)
        draw_offset_arrow(ax3, x1, y1, x2, y2, rel_color, label=label)
    
    plt.tight_layout()
    path3 = output_path / f"{prefix}_3_compound_sentiments.png"
    plt.savefig(path3, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    paths['compound_sentiments'] = path3
    
    # === PART 4: AGGREGATE SENTIMENTS ===
    fig4, ax4 = plt.subplots(figsize=(3.2 * scale, (num_entities * entity_box_height) + 1.4 * scale))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(-0.3, num_entities - 0.7 + entity_box_height)
    ax4.axis('off')
    ax4.set_title('Aggregate Sentiments', fontsize=base_font * 1.8, fontweight='bold', pad=10 * scale)
    
    for row, (entity_id, data) in enumerate(sorted_entities):
        entity_label = list(data['heads'])[0] if data['heads'] else f'E{entity_id}'

        agg_sent = graph.aggregate_sentiments.get(entity_id, 0.0)
        color = get_sentiment_color(agg_sent)

        box = FancyBboxPatch(
            (0.05, row - 0.35), 0.90, 0.7,
            boxstyle="round,pad=0.06",
            facecolor=color, edgecolor='#222', linewidth=1.4 * scale, alpha=0.9
        )
        ax4.add_patch(box)

        ax4.text(0.5, row + 0.22, f'E{entity_id}', ha='center', va='center',
                 fontsize=base_font * 1.4, fontweight='bold', color='#102c6b')
        ax4.text(0.5, row + 0.04, entity_label, ha='center', va='center',
                 fontsize=base_font * 1.2, fontweight='bold')
        ax4.text(0.5, row - 0.14, f'{agg_sent:+.3f}', ha='center', va='center',
                 fontsize=base_font * 1.3, fontweight='bold')
    
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    path4 = output_path / f"{prefix}_4_aggregate_sentiments.png"
    plt.savefig(path4, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    paths['aggregate_sentiments'] = path4
    
    # === CREATE LEGEND ===
    fig_legend, ax_legend = plt.subplots(figsize=(8 * scale, 1.2 * scale))
    ax_legend.axis('off')
    
    legend_elements = [
        mpatches.Patch(color=relation_colors['action'], label='Action Relation'),
        mpatches.Patch(color=relation_colors['association'], label='Association'),
        mpatches.Patch(color=relation_colors['belonging'], label='Belonging'),
        mpatches.Patch(color=sentiment_colors['positive'], label='Positive Sentiment'),
        mpatches.Patch(color=sentiment_colors['negative'], label='Negative Sentiment'),
        mpatches.Patch(color=sentiment_colors['neutral'], label='Neutral')
    ]
    
    legend = ax_legend.legend(
        handles=legend_elements,
        loc='center', ncol=3,
        frameon=True, fontsize=base_font * 1.15,
        edgecolor='#111', fancybox=True, title='Legend', title_fontsize=base_font * 1.3
    )
    ax_legend.text(
        0.5, -0.35,
        'Dotted = clause | Solid = temporal | Arrows = relations (action direction actor → target)',
        ha='center', va='center', transform=ax_legend.transAxes,
        fontsize=base_font * 0.9, style='italic', color='#444'
    )
    
    plt.tight_layout()
    path_legend = output_path / f"{prefix}_5_legend.png"
    plt.savefig(path_legend, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    paths['legend'] = path_legend

    # === COMBINED PIPELINE FIGURE ===
    if combined:
        import matplotlib.gridspec as gridspec
        # Heights for sections (relative)
        ent_h = max(1.2, num_entities * 0.55)
        init_h = max(1.2, num_entities * 0.65)
        comp_h = max(1.2, num_entities * 0.65)
        agg_h = max(1.0, num_entities * 0.55)
        leg_h = 1.4
        total_h = ent_h + init_h + comp_h + agg_h + leg_h + 1.0
        fig_comb = plt.figure(figsize=((max(num_clauses * 1.6, 6.5) + 1.0) * scale, total_h * scale))
        gs = gridspec.GridSpec(5, 1, height_ratios=[ent_h, init_h, comp_h, agg_h, leg_h], hspace=0.35)

        # ENTITIES
        ax_ent = fig_comb.add_subplot(gs[0])
        ax_ent.set_xlim(0, 1)
        ax_ent.set_ylim(-0.1, num_entities - 0.1)
        ax_ent.axis('off')
        ax_ent.set_title('Pipeline: Entities → Initial → Compound → Aggregate', fontsize=base_font * 2.0, fontweight='bold', pad=8 * scale)
        for row, (entity_id, data) in enumerate(sorted_entities):
            entity_label = list(data['heads'])[0] if data['heads'] else f'E{entity_id}'
            box = FancyBboxPatch((0.05, row - 0.35), 0.90, 0.7, boxstyle="round,pad=0.05",
                                 facecolor='lavender', edgecolor='#222', linewidth=1.2 * scale, alpha=0.9)
            ax_ent.add_patch(box)
            ax_ent.text(0.5, row + 0.13, f'E{entity_id}', ha='center', va='center', fontsize=base_font * 1.25,
                        fontweight='bold', color='#102c6b')
            ax_ent.text(0.5, row - 0.05, entity_label, ha='center', va='center', fontsize=base_font * 1.05, fontweight='bold')

        # INITIAL
        ax_init = fig_comb.add_subplot(gs[1])
        ax_init.set_xlim(-0.6, num_clauses - 0.4)
        ax_init.set_ylim(-0.4, num_entities - 0.6 + lane_height)
        ax_init.axis('off')
        ax_init.set_title('Initial Sentiments', fontsize=base_font * 1.6, fontweight='bold')
        for c in range(1, num_clauses):
            ax_init.axvline(x=c - 0.5, color='#bbb', linestyle=':', linewidth=1.0 * scale, alpha=0.55)
        for c in range(num_clauses):
            ax_init.text(c, num_entities - 0.05, f'C{c}', ha='center', va='bottom', fontsize=base_font * 1.1,
                         fontweight='bold', style='italic', color='#102c6b')
        for row, (entity_id, data) in enumerate(sorted_entities):
            for clause_idx, node_data in data['nodes']:
                x, y = node_positions[(entity_id, clause_idx)]
                init_sent = node_data.get('init_sentiment', 0.0) or 0.0
                color = get_sentiment_color(init_sent)
                circle = Circle((x, y), 0.23 * scale, facecolor=color, edgecolor='#111', linewidth=1.2 * scale, alpha=0.9, zorder=4)
                ax_init.add_patch(circle)
                ax_init.text(x, y, f'E{entity_id}', ha='center', va='center', fontsize=base_font, fontweight='bold')
                ax_init.text(x + 0.25 * scale, y, f'{init_sent:+.2f}', ha='left', va='center', fontsize=base_font * 0.9,
                             bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor=color, linewidth=0.9 * scale, alpha=0.95))
        # INITIAL RELATIONS
        for (u_id, u_clause), (v_id, v_clause), edge_data in graph.graph.edges(data=True):
            if (u_id, u_clause) not in node_positions or (v_id, v_clause) not in node_positions:
                continue
            rel_type = edge_data.get('relation')
            if rel_type == 'temporal':
                continue
            x1, y1 = node_positions[(u_id, u_clause)]
            x2, y2 = node_positions[(v_id, v_clause)]
            draw_offset_arrow(ax_init, x1, y1, x2, y2, relation_colors.get(rel_type, 'gray'), label=build_edge_label(edge_data))

        # COMPOUND
        ax_comp = fig_comb.add_subplot(gs[2])
        ax_comp.set_xlim(-0.6, num_clauses - 0.4)
        ax_comp.set_ylim(-0.4, num_entities - 0.6 + lane_height)
        ax_comp.axis('off')
        ax_comp.set_title('Compound Sentiments', fontsize=base_font * 1.6, fontweight='bold')
        for c in range(1, num_clauses):
            ax_comp.axvline(x=c - 0.5, color='#bbb', linestyle=':', linewidth=1.0 * scale, alpha=0.55)
        for c in range(num_clauses):
            ax_comp.text(c, num_entities - 0.05, f'C{c}', ha='center', va='bottom', fontsize=base_font * 1.1,
                         fontweight='bold', style='italic', color='#102c6b')
        for row, (entity_id, data) in enumerate(sorted_entities):
            for clause_idx, node_data in data['nodes']:
                x, y = node_positions[(entity_id, clause_idx)]
                comp_sent = node_data.get('compound_sentiment', node_data.get('init_sentiment', 0.0)) or 0.0
                color = get_sentiment_color(comp_sent)
                circle = Circle((x, y), 0.23 * scale, facecolor=color, edgecolor='#111', linewidth=1.2 * scale, alpha=0.9, zorder=4)
                ax_comp.add_patch(circle)
                ax_comp.text(x, y, f'E{entity_id}', ha='center', va='center', fontsize=base_font, fontweight='bold')
                ax_comp.text(x + 0.25 * scale, y, f'{comp_sent:+.2f}', ha='left', va='center', fontsize=base_font * 0.9,
                             bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor=color, linewidth=0.9 * scale, alpha=0.95))
        # COMPOUND RELATIONS
        for (u_id, u_clause), (v_id, v_clause), edge_data in graph.graph.edges(data=True):
            if (u_id, u_clause) not in node_positions or (v_id, v_clause) not in node_positions:
                continue
            rel_type = edge_data.get('relation')
            if rel_type == 'temporal':
                continue
            x1, y1 = node_positions[(u_id, u_clause)]
            x2, y2 = node_positions[(v_id, v_clause)]
            draw_offset_arrow(ax_comp, x1, y1, x2, y2, relation_colors.get(rel_type, 'gray'), label=build_edge_label(edge_data))

        # AGGREGATE
        ax_agg = fig_comb.add_subplot(gs[3])
        ax_agg.set_xlim(0, 1)
        ax_agg.set_ylim(-0.25, num_entities - 0.05)
        ax_agg.axis('off')
        ax_agg.set_title('Aggregate Sentiments', fontsize=base_font * 1.6, fontweight='bold')
        for row, (entity_id, data) in enumerate(sorted_entities):
            agg_sent = graph.aggregate_sentiments.get(entity_id, 0.0)
            color = get_sentiment_color(agg_sent)
            box = FancyBboxPatch((0.05, row - 0.35), 0.90, 0.7, boxstyle="round,pad=0.05",
                                 facecolor=color, edgecolor='#222', linewidth=1.2 * scale, alpha=0.9)
            ax_agg.add_patch(box)
            ax_agg.text(0.5, row + 0.12, f'E{entity_id}', ha='center', va='center', fontsize=base_font * 1.15,
                        fontweight='bold', color='#102c6b')
            ax_agg.text(0.5, row - 0.04, f'{agg_sent:+.3f}', ha='center', va='center', fontsize=base_font * 1.05, fontweight='bold')

        # LEGEND / KEY
        ax_key = fig_comb.add_subplot(gs[4])
        ax_key.axis('off')
        rel_patches = [mpatches.Patch(color=relation_colors['action'], label='Action'),
                       mpatches.Patch(color=relation_colors['association'], label='Association'),
                       mpatches.Patch(color=relation_colors['belonging'], label='Belonging')]
        sent_patches = [mpatches.Patch(color=sentiment_colors['positive'], label='Positive'),
                        mpatches.Patch(color=sentiment_colors['negative'], label='Negative'),
                        mpatches.Patch(color=sentiment_colors['neutral'], label='Neutral')]
        leg1 = ax_key.legend(handles=rel_patches, loc='center left', bbox_to_anchor=(0.05, 0.55), frameon=True,
                             fontsize=base_font * 1.05, title='Relations', title_fontsize=base_font * 1.2, fancybox=True)
        leg2 = ax_key.legend(handles=sent_patches, loc='center left', bbox_to_anchor=(0.32, 0.55), frameon=True,
                             fontsize=base_font * 1.05, title='Sentiment', title_fontsize=base_font * 1.2, fancybox=True)
        ax_key.add_artist(leg1)
        ax_key.add_artist(leg2)
        ax_key.text(0.05, 0.25, 'Arrow labels show action words/modifiers linking entities.\nArrows stop at node boundary for clarity.',
                    ha='left', va='top', fontsize=base_font * 0.95)

        # BIG PIPELINE ARROWS (between sections)
        # Use figure-level annotation arrows pointing downward
        fig_comb.text(0.97, 1 - (ent_h / total_h) + 0.01, '↓', ha='center', va='center', fontsize=base_font * 3.0, fontweight='bold')
        fig_comb.text(0.97, 1 - ((ent_h + init_h) / total_h) + 0.01, '↓', ha='center', va='center', fontsize=base_font * 3.0, fontweight='bold')
        fig_comb.text(0.97, 1 - ((ent_h + init_h + comp_h) / total_h) + 0.01, '↓', ha='center', va='center', fontsize=base_font * 3.0, fontweight='bold')

        path_comb = output_path / f"{prefix}_6_combined_pipeline.png"
        plt.tight_layout()
        fig_comb.savefig(path_comb, dpi=dpi, facecolor='white', bbox_inches='tight')
        plt.close(fig_comb)
        paths['combined_pipeline'] = path_comb
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'prefix': prefix,
        'num_entities': len(entity_data),
        'num_clauses': num_clauses,
        'num_nodes': graph.graph.number_of_nodes(),
        'num_edges': graph.graph.number_of_edges(),
        'entities': {
            entity_id: {
                'label': list(data['heads'])[0] if data['heads'] else f'Entity {entity_id}',
                'clauses': sorted(data['clause_indices']),
                'num_mentions': len(data['clause_indices']),
                'aggregate_sentiment': float(graph.aggregate_sentiments.get(entity_id, 0.0))
            }
            for entity_id, data in entity_data.items()
        },
        'visualizations': {k: str(v) for k, v in paths.items()}
    }
    
    metadata_path = output_path / f"{prefix}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Created {len(paths)} separate visualization panels:")
    for viz_type, path in paths.items():
        logger.info(f"  • {viz_type}: {path.name}")
    
    return paths

if __name__ == "__main__":
    sample_text = """The bully hurt a child. However, the child had many friends. The friends ostracized the bully."""
    pipeline = build_default_pipeline(use_cache=False)
    result = pipeline.process(sample_text)
    print(result)
    print_pipeline_output(result)
    
    # Create visualizations
    visualize_graph(result['graph'])
