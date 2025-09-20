import networkx as nx
from typing import Dict, Any, List, Callable, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .sentiment_systems import SentimentSystem

logger = logging.getLogger(__name__)

ENTITY_ROLES = {
    "actor": {"description": "Initiates an action"},
    "target": {"description": "Receives an action"},
    "parent": {"description": "Owns a child entity"},
    "child": {"description": "Belongs to a parent"},
    "associate": {"description": "Is connected to another entity"}
}

RELATION_TYPES = {
    "temporal": {"description": "Indicates a temporal relationship"},
    "action": {"description": "Indicates an action between entities"},
    "belonging": {"description": "Indicates a belonging relationship"},
    "association": {"description": "Indicates an association between entities"}
}

class RelationGraph:
    def __init__(self, text="", sentiment_analyzer_system=None):
        self.graph = nx.Graph()
        self.text = text
        self.sentiment_analyzer_system = sentiment_analyzer_system
        self.entity_ids = set()
        self.aggregate_sentiments = {}
        
        # Optimization: Batch operations
        self._pending_edges = []
        self._pending_modifiers = {}
        self._sentiment_cache = {}
        
        # Pre-split text for clause context optimization
        self._clause_parts = None

    def _validate_role(self, role):
        if role not in ENTITY_ROLES:
            raise ValueError(f"Invalid entity role: {role}. Valid roles are: {list(ENTITY_ROLES.keys())}")
    
    def _validate_relation(self, relation):
        if relation not in RELATION_TYPES:
            raise ValueError(f"Invalid relation type: {relation}. Valid relations are: {list(RELATION_TYPES.keys())}")

    def add_entity_node(self, id: int, head: str, modifier: List[str], entity_role: str, clause_layer: int):
        logger.info("Adding entity node...")
        self._validate_role(entity_role)
        self.entity_ids.add(id)
        
        # Batch node creation for efficiency
        node_key = f"{id}_{clause_layer}"
        self.graph.add_node(
            node_key,
            head=head,
            modifier=modifier[:],  # Copy to avoid reference issues
            text=self.text,
            entity_role=entity_role,
            clause_layer=clause_layer,
            init_sentiment=None
        )
        
        # Store for batch processing
        if modifier:
            self._pending_modifiers[node_key] = modifier[:]

    def _get_clause_context(self, clause_layer: int) -> str:
        """Optimized clause context extraction with caching."""
        if self._clause_parts is None:
            import re
            self._clause_parts = [p.strip() for p in re.split(r'[.!?]+', self.text) if p.strip()]
        
        if 0 <= clause_layer < len(self._clause_parts):
            return self._clause_parts[clause_layer]
        return ""
    
    def _analyze_sentiment_cached(self, text: str) -> float:
        """Cached sentiment analysis to avoid duplicate calculations."""
        if not text or not text.strip():
            return 0.0
            
        if text in self._sentiment_cache:
            return self._sentiment_cache[text]
        
        try:
            if self.sentiment_analyzer_system:
                result = self.sentiment_analyzer_system.analyze(text)
                self._sentiment_cache[text] = result
                return result
        except Exception:
            pass
        
        self._sentiment_cache[text] = 0.0
        return 0.0
    
    def add_entity_modifier(self, entity_id: int, modifier: List[str], clause_layer: int):
        logger.info("Adding entity modifiers...")
        if entity_id not in self.entity_ids:
            raise ValueError(f"Entity ID {entity_id} not found in the graph.")
        node_key = f"{entity_id}_{clause_layer}"
        if node_key not in self.graph.nodes:
            raise ValueError(f"Node {node_key} not found in the graph.")
        
        current_modifiers = self.graph.nodes[node_key].get('modifier', [])
        
        # Optimize: Use set for fast duplicate checking
        current_set = set(current_modifiers)
        new_modifiers = [mod for mod in modifier if mod not in current_set]
        
        if new_modifiers:
            all_modifiers = current_modifiers + new_modifiers
            self.graph.nodes[node_key]['modifier'] = all_modifiers
            
            # CORRECTED: Entity-specific approach - only analyze sentiment of modifiers
            unique_modifiers = list(dict.fromkeys(all_modifiers))
            modifier_text = ' '.join(unique_modifiers).strip()
            if modifier_text:
                self.graph.nodes[node_key]['init_sentiment'] = self._analyze_sentiment_cached(modifier_text)
            else:
                self.graph.nodes[node_key]['init_sentiment'] = 0.0
    
    def finalize_sentiment_analysis(self):
        """Batch process sentiment analysis for nodes without modifiers."""
        logger.info("Finalizing sentiment analysis for nodes without modifiers...")
        
        # Collect all texts for batch processing
        batch_texts = []
        batch_nodes = []
        
        for node_key in self.graph.nodes:
            node_data = self.graph.nodes[node_key]
            if node_data.get('init_sentiment') is None:
                modifiers = node_data.get('modifier', [])
                modifier_text = ' '.join(modifiers).strip()
                
                if modifier_text:
                    batch_texts.append(modifier_text)
                    batch_nodes.append(node_key)
                else:
                    # No modifiers = neutral entity
                    self.graph.nodes[node_key]['init_sentiment'] = 0.0
        
        # Batch sentiment analysis if available
        if batch_texts and self.sentiment_analyzer_system:
            if hasattr(self.sentiment_analyzer_system, 'analyze_batch'):
                results = self.sentiment_analyzer_system.analyze_batch(batch_texts)
                for node_key, result in zip(batch_nodes, results):
                    self.graph.nodes[node_key]['init_sentiment'] = result
            else:
                # Fallback to individual analysis
                for node_key, text in zip(batch_nodes, batch_texts):
                    self.graph.nodes[node_key]['init_sentiment'] = self._analyze_sentiment_cached(text)
    
    def set_entity_role(self, entity_id: int, entity_role: str, clause_layer: int):
        logger.info("Setting entity role...")
        self._validate_role(entity_role)
        if entity_id not in self.entity_ids:
            raise ValueError(f"Entity ID {entity_id} not found in the graph.")
        node_key = f"{entity_id}_{clause_layer}"
        if node_key not in self.graph.nodes:
            raise ValueError(f"Node {node_key} not found in the graph.")
        self.graph.nodes[node_key]['entity_role'] = entity_role

    def add_temporal_edge(self, entity_id: int):
        logger.info("Adding temporal edge...")
        if entity_id not in self.entity_ids:
            raise ValueError(f"Entity ID {entity_id} not found in the graph.")
        layers = []
        for node in self.graph.nodes:
            id_int = int(str(node).split('_')[0])
            clause_layer = int(str(node).split('_')[1])
            if id_int == entity_id:
                layers.append(clause_layer)
        layers = sorted(layers)
        for i in range(len(layers) - 1):
            self.graph.add_edge(f"{entity_id}_{layers[i]}", f"{entity_id}_{layers[i + 1]}", relation="temporal")
    
    def add_action_edge(self, actor_id: int, target_id: int, clause_layer: int, head: str, modifier: List[str]):
        logger.info(f"Adding action edge between {actor_id} and {target_id}...")
        if actor_id not in self.entity_ids or target_id not in self.entity_ids:
            raise ValueError(f"Actor ID {actor_id} or Target ID {target_id} not found in the graph.")
        self.graph.add_edge(
            f"{actor_id}_{clause_layer}",
            f"{target_id}_{clause_layer}",
            actor=f"{actor_id}_{clause_layer}",
            target=f"{target_id}_{clause_layer}",
            relation="action",
            head=head,
            modifier=modifier,
        )
        try:
            action_text = (head + ' ' + ' '.join(modifier)).strip()
            if self.sentiment_analyzer_system and action_text:
                action_sent = self.sentiment_analyzer_system.analyze(action_text)
            else:
                action_sent = 0.0
        except Exception:
            action_sent = 0.0
        edge_key = (f"{actor_id}_{clause_layer}", f"{target_id}_{clause_layer}")
        self.graph.edges[edge_key]['init_sentiment'] = action_sent

    def add_belonging_edge(self, parent_id: int, child_id: int, clause_layer: int):
        logger.info(f"Adding belonging edge between {parent_id} and {child_id}...")
        if parent_id not in self.entity_ids or child_id not in self.entity_ids:
            raise ValueError(f"Parent ID {parent_id} or Child ID {child_id} not found in the graph.")
        self.graph.add_edge(f"{parent_id}_{clause_layer}", f"{child_id}_{clause_layer}", 
                            parent=f"{parent_id}_{clause_layer}", child=f"{child_id}_{clause_layer}",
                            relation="belonging")

    def add_association_edge(self, entity1_id: int, entity2_id: int, clause_layer: int, rel_text: str = ""):
        logger.info(f"Adding association edge between {entity1_id} and {entity2_id}...")
        if entity1_id not in self.entity_ids or entity2_id not in self.entity_ids:
            raise ValueError(f"Entity1 ID {entity1_id} or Entity2 ID {entity2_id} not found in the graph.")
        self.graph.add_edge(f"{entity1_id}_{clause_layer}", f"{entity2_id}_{clause_layer}", 
                            entity1=f"{entity1_id}_{clause_layer}", entity2=f"{entity2_id}_{clause_layer}",
                            relation="association", text=rel_text)

    def add_relation_edge(self, entity1_id: int, entity2_id: int, clause_layer: int, relation_type: str, relation_text: str = ""):
        logger.info(f"Adding {relation_type} edge between {entity1_id} and {entity2_id}...")
        if entity1_id not in self.entity_ids or entity2_id not in self.entity_ids:
            raise ValueError(f"Entity ID {entity1_id} or {entity2_id} not found in the graph.")
        self.graph.add_edge(f"{entity1_id}_{clause_layer}", f"{entity2_id}_{clause_layer}", 
                            relation=relation_type, text=relation_text)

    def run_compound_sentiment_calculations(self, actor_func: Callable, target_func: Callable, 
                                           assoc_func: Callable, parent_func: Callable, child_func: Callable):
        logger.info("Running compound sentiment calculations...")
        
        # First, handle temporal edges (coreference propagation) - CRITICAL for entity-specific architecture
        for edge in self.graph.edges:
            edge_data = self.graph.edges[edge]
            if 'relation' in edge_data and edge_data['relation'] == "temporal":
                node1, node2 = edge
                node1_data = self.graph.nodes[node1]
                node2_data = self.graph.nodes[node2]
                
                # Determine chronological order (lower clause layer = earlier)
                layer1 = node1_data.get('clause_layer', 0)
                layer2 = node2_data.get('clause_layer', 0)
                
                if layer1 < layer2:
                    # node1 is earlier, node2 is later - propagate sentiment forward
                    earlier_node, later_node = node1, node2
                else:
                    # node2 is earlier, node1 is later - propagate sentiment forward  
                    earlier_node, later_node = node2, node1
                
                earlier_sentiment = self.graph.nodes[earlier_node].get('init_sentiment', 0.0)
                later_sentiment = self.graph.nodes[later_node].get('init_sentiment', 0.0)
                
                # Propagate sentiment across temporal edge (coreference resolution)
                # If later mention has strong sentiment and earlier is neutral, propagate backwards
                if abs(later_sentiment) > 0.1 and abs(earlier_sentiment) < 0.1:
                    self.graph.nodes[earlier_node]['compound_sentiment'] = later_sentiment
                    logger.info(f"Temporal propagation: {earlier_node} inherits sentiment {later_sentiment} from {later_node}")
                # If earlier mention has sentiment, ensure later mention inherits it if neutral
                elif abs(earlier_sentiment) > 0.1 and abs(later_sentiment) < 0.1:
                    self.graph.nodes[later_node]['compound_sentiment'] = earlier_sentiment
                    logger.info(f"Temporal propagation: {later_node} inherits sentiment {earlier_sentiment} from {earlier_node}")
                # If both have sentiment, use weighted average favoring the stronger sentiment
                elif abs(earlier_sentiment) > 0.1 and abs(later_sentiment) > 0.1:
                    combined_sentiment = (earlier_sentiment + later_sentiment) / 2.0
                    self.graph.nodes[earlier_node]['compound_sentiment'] = combined_sentiment
                    self.graph.nodes[later_node]['compound_sentiment'] = combined_sentiment
                    logger.info(f"Temporal propagation: {earlier_node} and {later_node} combined sentiment {combined_sentiment}")
        
        # Then handle action edges
        for edge in self.graph.edges:
            edge_data = self.graph.edges[edge]
            if 'relation' in edge_data and edge_data['relation'] == "action":
                actor = edge_data['actor']
                target = edge_data['target']
                actor_sentiment_init = self.graph.nodes[actor].get('compound_sentiment', self.graph.nodes[actor]['init_sentiment'])
                action_sentiment_init = edge_data.get('init_sentiment', 0.0)
                target_sentiment_init = self.graph.nodes[target].get('compound_sentiment', self.graph.nodes[target]['init_sentiment'])

                actor_sentiment_compound = actor_func([actor_sentiment_init, action_sentiment_init])
                target_sentiment_compound = target_func([target_sentiment_init, action_sentiment_init])

                self.graph.nodes[actor]['compound_sentiment'] = actor_sentiment_compound
                self.graph.nodes[target]['compound_sentiment'] = target_sentiment_compound
            
            elif 'relation' in edge_data and edge_data['relation'] == "belonging":
                parent = edge_data['parent']
                child = edge_data['child']
                parent_sentiment_init = self.graph.nodes[parent].get('compound_sentiment', self.graph.nodes[parent]['init_sentiment'])
                child_sentiment_init = self.graph.nodes[child].get('compound_sentiment', self.graph.nodes[child]['init_sentiment'])

                parent_sentiment_compound = parent_func([parent_sentiment_init, child_sentiment_init])
                child_sentiment_compound = child_func([child_sentiment_init, parent_sentiment_init])

                self.graph.nodes[parent]['compound_sentiment'] = parent_sentiment_compound
                self.graph.nodes[child]['compound_sentiment'] = child_sentiment_compound
            
            elif 'relation' in edge_data and edge_data['relation'] == "association":
                entity1 = edge_data['entity1']
                entity2 = edge_data['entity2']
                entity1_sentiment_init = self.graph.nodes[entity1].get('compound_sentiment', self.graph.nodes[entity1]['init_sentiment'])
                entity2_sentiment_init = self.graph.nodes[entity2].get('compound_sentiment', self.graph.nodes[entity2]['init_sentiment'])

                entity1_sentiment_compound = assoc_func([entity1_sentiment_init, entity2_sentiment_init])
                entity2_sentiment_compound = assoc_func([entity2_sentiment_init, entity1_sentiment_init])

                self.graph.nodes[entity1]['compound_sentiment'] = entity1_sentiment_compound
                self.graph.nodes[entity2]['compound_sentiment'] = entity2_sentiment_compound

    def run_compound_action_sentiment_calculations(self, function: Callable = None):
        """
        Run compound action sentiment calculations following OLD architecture pattern
        """
        logger.info("Running compound action sentiment calculations...")
        for edge in self.graph.edges:
            edge_data = self.graph.edges[edge]
            if 'relation' in edge_data and edge_data['relation'] == "action":
                actor = edge_data['actor']
                target = edge_data['target']
                actor_sentiment_init = self.graph.nodes[actor]['init_sentiment']
                action_sentiment_init = edge_data.get('init_sentiment', 0.0)
                target_sentiment_init = self.graph.nodes[target]['init_sentiment']

                if function is None:
                    raise ValueError("No function provided for compound sentiment calculation")
                actor_sentiment_compound, target_sentiment_compound = function(actor_sentiment_init, action_sentiment_init, target_sentiment_init)

                self.graph.nodes[actor]['compound_sentiment'] = actor_sentiment_compound
                self.graph.nodes[target]['compound_sentiment'] = target_sentiment_compound

    def run_compound_belonging_sentiment_calculations(self, function: Callable = None):
        """
        Run compound belonging sentiment calculations following OLD architecture pattern
        """
        logger.info("Running compound belonging sentiment calculations...")
        for edge in self.graph.edges:
            edge_data = self.graph.edges[edge]
            if 'relation' in edge_data and edge_data['relation'] == "belonging":
                parent = edge_data['parent']
                child = edge_data['child']
                parent_sentiment_init = self.graph.nodes[parent]['init_sentiment']
                child_sentiment_init = self.graph.nodes[child]['init_sentiment']

                if function is None:
                    raise ValueError("No function provided for compound sentiment calculation")
                parent_sentiment_compound, child_sentiment_compound = function(parent_sentiment_init, child_sentiment_init)

                self.graph.nodes[parent]['compound_sentiment'] = parent_sentiment_compound
                self.graph.nodes[child]['compound_sentiment'] = child_sentiment_compound

    def run_compound_association_sentiment_calculations(self, function: Callable = None):
        """
        Run compound association sentiment calculations following OLD architecture pattern
        """
        logger.info("Running compound association sentiment calculations...")
        for edge in self.graph.edges:
            edge_data = self.graph.edges[edge]
            if 'relation' in edge_data and edge_data['relation'] == "association":
                entity1 = edge_data['entity1']
                entity2 = edge_data['entity2']
                entity1_sentiment_init = self.graph.nodes[entity1]['init_sentiment']
                entity2_sentiment_init = self.graph.nodes[entity2]['init_sentiment']

                if function is None:
                    raise ValueError("No function provided for compound sentiment calculation")
                entity1_sentiment_compound, entity2_sentiment_compound = function(entity1_sentiment_init, entity2_sentiment_init)

                self.graph.nodes[entity1]['compound_sentiment'] = entity1_sentiment_compound
                self.graph.nodes[entity2]['compound_sentiment'] = entity2_sentiment_compound

    def run_aggregate_sentiment_calculations(self, aggregate_func: Callable):
        """
        CRITICAL FIX: Aggregate sentiment across ALL temporal instances of same entity
        Following OLD architecture pattern for proper entity-specific aggregation
        """
        logger.info("Running aggregate sentiment calculations...")
        results = {}
        
        # Group nodes by entity_id following OLD architecture approach
        entity_groups = {}
        for node in self.graph.nodes:
            node_data = self.graph.nodes[node]
            entity_id = int(str(node).split('_')[0])
            clause_layer = int(str(node).split('_')[1])
            entity_name = node_data.get('head', f'Entity{entity_id}')
            
            if entity_id not in entity_groups:
                entity_groups[entity_id] = {
                    'entity_name': entity_name,
                    'sentiments': [],
                    'nodes': []
                }
            
            # Prefer compound_sentiment, fall back to init_sentiment
            sentiment = node_data.get('compound_sentiment', node_data.get('init_sentiment', 0.0))
            entity_groups[entity_id]['sentiments'].append(sentiment)
            entity_groups[entity_id]['nodes'].append(node)
            
            # Update entity name to the most descriptive one (longest text)
            if len(entity_name) > len(entity_groups[entity_id]['entity_name']):
                entity_groups[entity_id]['entity_name'] = entity_name
        
        # Calculate aggregate sentiment for each entity following OLD architecture
        for entity_id, group_data in entity_groups.items():
            sentiments = group_data['sentiments']
            entity_name = group_data['entity_name']
            
            if sentiments:
                # Use aggregate function from OLD architecture (survey optimization)
                aggregated_sentiment = aggregate_func(sentiments)
                results[entity_name] = aggregated_sentiment
                self.aggregate_sentiments[entity_id] = aggregated_sentiment
                
                logger.info(f"Entity {entity_name} (ID {entity_id}): sentiments {sentiments} -> aggregated {aggregated_sentiment}")
            else:
                results[entity_name] = 0.0
                self.aggregate_sentiments[entity_id] = 0.0
        
        return results


class GraphVisualizer:
    def __init__(self, relation_graph: RelationGraph, clusters: Dict[str, Any] = None):
        self.relation_graph = relation_graph
        self.graph = relation_graph.graph
        self.clusters = clusters or {}
        self.edge_color_map = {
            "temporal": "grey",
            "action": "blue",
            "belonging": "purple",
            "association": "orange"
        }

    def _get_node_colors(self):
        colors = {}
        for node, data in self.graph.nodes(data=True):
            sentiment = data.get('compound_sentiment', data.get('init_sentiment', 0.0))
            if sentiment > 0.1:
                colors[node] = 'green'
            elif sentiment < -0.1:
                colors[node] = 'red'
            else:
                colors[node] = 'lightgray'
        return colors

    def draw_graph(self, save_path=None):
        logger.info("Drawing graph...")
        if not self.graph.nodes:
            logger.info("Graph is empty. Nothing to draw.")
            return
        try:
            import plotly.graph_objects as go
        except Exception as exc:
            logger.warning("Plotly not available; cannot draw graph: %s", exc)
            return

        pos_2d = nx.spring_layout(self.graph, k=2.0, iterations=100, seed=42)
        pos_3d = {
            node: (pos_2d[node][0], pos_2d[node][1], data['clause_layer'])
            for node, data in self.graph.nodes(data=True)
        }

        edge_traces = []
        edges_by_relation = {rel_type: [] for rel_type in self.edge_color_map.keys()}
        for u, v, data in self.graph.edges(data=True):
            relation = data.get('relation')
            if relation in edges_by_relation:
                edges_by_relation[relation].append((u, v))

        for relation, edges in edges_by_relation.items():
            if not edges:
                continue
            edge_x, edge_y, edge_z = [], [], []
            for u, v in edges:
                x0, y0, z0 = pos_3d[u]
                x1, y1, z1 = pos_3d[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_z.extend([z0, z1, None])
            
            trace = go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                line=dict(width=3, color=self.edge_color_map[relation]),
                hoverinfo='none',
                mode='lines',
                name=f'{relation.capitalize()} Relation'
            )
            edge_traces.append(trace)

        node_x, node_y, node_z = [], [], []
        node_text = []
        node_colors = []
        node_colors_dict = self._get_node_colors()

        for node, data in sorted(self.graph.nodes(data=True)):
            x, y, z = pos_3d[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_colors.append(node_colors_dict[node])
            
            sentiment = data.get('compound_sentiment', data.get('init_sentiment', 0.0))
            sentiment_str = f"{sentiment:.2f}"
            if 'compound_sentiment' in data:
                 sentiment_str += f" (compounded from {data.get('init_sentiment', 0.0):.2f})"

            hover_info = (f"<b>{data['head']} ({node})</b><br>"
                          f"Role: {data['entity_role']}<br>"
                          f"Layer: {data['clause_layer']}<br>"
                          f"Sentiment: {sentiment_str}")
            node_text.append(hover_info)

        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                color=node_colors,
                size=15,
                line=dict(width=1, color='black')
            ),
            name='Entities'
        )

        fig = go.Figure(data=edge_traces + [node_trace])

        fig.update_layout(
            title='3D Relation Graph',
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.6)'),
            margin=dict(l=0, r=0, b=0, t=40),
            scene=dict(
                xaxis=dict(showticklabels=False, title=''),
                yaxis=dict(showticklabels=False, title=''),
                zaxis=dict(title='Clause Layer', nticks=max(pos[2] for pos in pos_3d.values()) + 1)
            )
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Interactive graph saved to {save_path}")
        else:
            fig.show()
