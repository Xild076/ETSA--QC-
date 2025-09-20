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
        self.graph.add_node(
            f"{id}_{clause_layer}",
            head=head,
            modifier=modifier,
            text=self.text,
            entity_role=entity_role,
            clause_layer=clause_layer,
        )
        text_for_sent = (head + ' ' + ' '.join(modifier)).strip()
        if self.sentiment_analyzer_system:
            self.graph.nodes[f"{id}_{clause_layer}"]['init_sentiment'] = self.sentiment_analyzer_system.analyze(text_for_sent)
        else:
            self.graph.nodes[f"{id}_{clause_layer}"]['init_sentiment'] = 0.0

    def _get_clause_context(self, clause_layer: int) -> str:
        import re
        parts = [p.strip() for p in re.split(r'[.!?]+', self.text) if p.strip()]
        if 0 <= clause_layer < len(parts):
            return parts[clause_layer]
        return ""
    
    def add_entity_modifier(self, entity_id: int, modifier: List[str], clause_layer: int):
        logger.info("Adding entity modifiers...")
        if entity_id not in self.entity_ids:
            raise ValueError(f"Entity ID {entity_id} not found in the graph.")
        node_key = f"{entity_id}_{clause_layer}"
        if node_key not in self.graph.nodes:
            raise ValueError(f"Node {node_key} not found in the graph.")
        current_modifiers = self.graph.nodes[node_key].get('modifier', [])
        new_mods = current_modifiers + modifier
        self.graph.nodes[node_key]['modifier'] = new_mods
        try:
            head = self.graph.nodes[node_key].get('head', '')
            text_for_sent = (head + ' ' + ' '.join(new_mods)).strip()
            if self.sentiment_analyzer_system:
                self.graph.nodes[node_key]['init_sentiment'] = self.sentiment_analyzer_system.analyze(text_for_sent)
        except Exception:
            pass
    
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

    def add_association_edge(self, entity1_id: int, entity2_id: int, clause_layer: int):
        logger.info(f"Adding association edge between {entity1_id} and {entity2_id}...")
        if entity1_id not in self.entity_ids or entity2_id not in self.entity_ids:
            raise ValueError(f"Entity1 ID {entity1_id} or Entity2 ID {entity2_id} not found in the graph.")
        self.graph.add_edge(f"{entity1_id}_{clause_layer}", f"{entity2_id}_{clause_layer}", 
                            entity1=f"{entity1_id}_{clause_layer}", entity2=f"{entity2_id}_{clause_layer}",
                            relation="association")

    def add_relation_edge(self, entity1_id: int, entity2_id: int, clause_layer: int, relation_type: str, relation_text: str = ""):
        logger.info(f"Adding {relation_type} edge between {entity1_id} and {entity2_id}...")
        if entity1_id not in self.entity_ids or entity2_id not in self.entity_ids:
            raise ValueError(f"Entity ID {entity1_id} or {entity2_id} not found in the graph.")
        self.graph.add_edge(f"{entity1_id}_{clause_layer}", f"{entity2_id}_{clause_layer}", 
                            relation=relation_type, text=relation_text)

    def run_compound_sentiment_calculations(self, actor_func: Callable, target_func: Callable, 
                                           assoc_func: Callable, parent_func: Callable, child_func: Callable):
        logger.info("Running compound sentiment calculations...")
        for edge in self.graph.edges:
            edge_data = self.graph.edges[edge]
            if 'relation' in edge_data and edge_data['relation'] == "action":
                actor = edge_data['actor']
                target = edge_data['target']
                actor_sentiment_init = self.graph.nodes[actor]['init_sentiment']
                action_sentiment_init = edge_data.get('init_sentiment', 0.0)
                target_sentiment_init = self.graph.nodes[target]['init_sentiment']

                actor_sentiment_compound = actor_func([actor_sentiment_init, action_sentiment_init])
                target_sentiment_compound = target_func([target_sentiment_init, action_sentiment_init])

                self.graph.nodes[actor]['compound_sentiment'] = actor_sentiment_compound
                self.graph.nodes[target]['compound_sentiment'] = target_sentiment_compound
            
            elif 'relation' in edge_data and edge_data['relation'] == "belonging":
                parent = edge_data['parent']
                child = edge_data['child']
                parent_sentiment_init = self.graph.nodes[parent]['init_sentiment']
                child_sentiment_init = self.graph.nodes[child]['init_sentiment']

                parent_sentiment_compound = parent_func([parent_sentiment_init, child_sentiment_init])
                child_sentiment_compound = child_func([child_sentiment_init, parent_sentiment_init])

                self.graph.nodes[parent]['compound_sentiment'] = parent_sentiment_compound
                self.graph.nodes[child]['compound_sentiment'] = child_sentiment_compound
            
            elif 'relation' in edge_data and edge_data['relation'] == "association":
                entity1 = edge_data['entity1']
                entity2 = edge_data['entity2']
                entity1_sentiment_init = self.graph.nodes[entity1]['init_sentiment']
                entity2_sentiment_init = self.graph.nodes[entity2]['init_sentiment']

                entity1_sentiment_compound = assoc_func([entity1_sentiment_init, entity2_sentiment_init])
                entity2_sentiment_compound = assoc_func([entity2_sentiment_init, entity1_sentiment_init])

                self.graph.nodes[entity1]['compound_sentiment'] = entity1_sentiment_compound
                self.graph.nodes[entity2]['compound_sentiment'] = entity2_sentiment_compound

    def run_aggregate_sentiment_calculations(self, aggregate_func: Callable):
        logger.info("Running aggregate sentiment calculations...")
        results = {}
        for entity_id in self.entity_ids:
            layers = []
            for node in self.graph.nodes:
                id_int = int(str(node).split('_')[0])
                clause_layer = int(str(node).split('_')[1])
                if id_int == entity_id:
                    layers.append(clause_layer)
            layers = sorted(layers)
            sentiments = []
            for layer in layers:
                node = f"{entity_id}_{layer}"
                if 'compound_sentiment' in self.graph.nodes[node]:
                    sentiments.append(self.graph.nodes[node].get('compound_sentiment', 0.0))
                else:
                    sentiments.append(self.graph.nodes[node].get('init_sentiment', 0.0))
            
            result = aggregate_func(sentiments) if sentiments else 0.0
            self.aggregate_sentiments[entity_id] = result
            
            entity_name = None
            for node in self.graph.nodes:
                if node.startswith(f"{entity_id}_"):
                    entity_name = self.graph.nodes[node].get('head', f'Entity{entity_id}')
                    break
            if entity_name:
                results[entity_name] = result
        
        return results
            else:
                entity_text = head
            
            # Use the provided init_sentiment (calculated with full context) instead of recalculating
            entity_sentiment = init_sentiment
            
            self.graph.add_node(
                node_name,
                entity_id=id,
                head=head,
                modifiers=modifier,
                clause_layer=clause_layer,
                entity_role=entity_role,
                init_sentiment=float(entity_sentiment),
                sentiment=float(entity_sentiment),
                compound_sentiment=float(entity_sentiment)
            )
    
    def add_temporal_edges(self):
        nodes_by_entity = defaultdict(list)
        for node, data in self.graph.nodes(data=True):
            nodes_by_entity[data['entity_id']].append(data)

        for entity_id, nodes in nodes_by_entity.items():
            sorted_nodes = sorted(nodes, key=lambda x: x['clause_layer'])
            for i in range(len(sorted_nodes) - 1):
                from_node = f"{entity_id}_{sorted_nodes[i]['clause_layer']}"
                to_node = f"{entity_id}_{sorted_nodes[i+1]['clause_layer']}"
                self.graph.add_edge(from_node, to_node, type='temporal')

    def add_action_edge(self, actor_id: int, target_id: int, clause_layer: int, head: str, modifier: List[str]):
        """Add action edge with actor-target relationship"""
        logger.info(f"Adding action edge between {actor_id} (actor) and {target_id} (target)...")
        actor_node = f"{actor_id}_{clause_layer}"
        target_node = f"{target_id}_{clause_layer}"

        if not self.graph.has_node(actor_node) or not self.graph.has_node(target_node):
            return

        # Set entity roles
        self.graph.nodes[actor_node]['entity_role'] = 'actor'
        self.graph.nodes[target_node]['entity_role'] = 'target'

        # Create action text and analyze sentiment
        action_text = (head + ' ' + ' '.join(modifier)).strip() if modifier else head
        action_sentiment = self.sentiment_system.analyze(action_text) if action_text else 0.0

        self.graph.add_edge(
            actor_node,
            target_node,
            type='action',
            text=action_text,
            clause_layer=clause_layer,
            actor=actor_node,
            target=target_node,
            init_sentiment=action_sentiment
        )

    def add_belonging_edge(self, parent_id: int, child_id: int, clause_layer: int):
        """Add belonging edge with parent-child relationship"""
        logger.info(f"Adding belonging edge between {parent_id} (parent) and {child_id} (child)...")
        parent_node = f"{parent_id}_{clause_layer}"
        child_node = f"{child_id}_{clause_layer}"

        if not self.graph.has_node(parent_node) or not self.graph.has_node(child_node):
            return

        # Set entity roles
        self.graph.nodes[parent_node]['entity_role'] = 'parent'
        self.graph.nodes[child_node]['entity_role'] = 'child'

        self.graph.add_edge(
            parent_node,
            child_node,
            type='belonging',
            clause_layer=clause_layer,
            parent=parent_node,
            child=child_node
        )

    def add_association_edge(self, entity1_id: int, entity2_id: int, clause_layer: int, rel_text: str = ""):
        """Add association edge between entities"""
        logger.info(f"Adding association edge between {entity1_id} and {entity2_id}...")
        entity1_node = f"{entity1_id}_{clause_layer}"
        entity2_node = f"{entity2_id}_{clause_layer}"

        if not self.graph.has_node(entity1_node) or not self.graph.has_node(entity2_node):
            return

        # Set entity roles
        self.graph.nodes[entity1_node]['entity_role'] = 'associate'
        self.graph.nodes[entity2_node]['entity_role'] = 'associate'

        self.graph.add_edge(
            entity1_node,
            entity2_node,
            type='association',
            text=rel_text,
            clause_layer=clause_layer,
            entity1=entity1_node,
            entity2=entity2_node
        )

        # Add reverse edge for association
        self.graph.add_edge(
            entity2_node,
            entity1_node,
            type='association',
            text=rel_text,
            clause_layer=clause_layer,
            entity1=entity2_node,
            entity2=entity1_node
        )

    def add_relation_edge(self, subj_id: int, obj_id: int, clause_layer: int, rel_type: str, rel_text: str):
        """Generic relation edge method for backward compatibility"""
        if rel_type == "action":
            action_parts = rel_text.split()
            head = action_parts[0] if action_parts else ""
            modifiers = action_parts[1:] if len(action_parts) > 1 else []
            self.add_action_edge(subj_id, obj_id, clause_layer, head, modifiers)
        elif rel_type == "belonging":
            self.add_belonging_edge(subj_id, obj_id, clause_layer)
        elif rel_type == "association":
            self.add_association_edge(subj_id, obj_id, clause_layer, rel_text)
        else:
            # Fallback for unknown relation types
            subj_node = f"{subj_id}_{clause_layer}"
            obj_node = f"{obj_id}_{clause_layer}"
            if self.graph.has_node(subj_node) and self.graph.has_node(obj_node):
                self.graph.add_edge(
                    subj_node, obj_node,
                    type=rel_type, text=rel_text, clause_layer=clause_layer
                )

    def add_relation_edge(self, subj_id: int, obj_id: int, clause_layer: int, rel_type: str, rel_text: str):
        subj_node = f"{subj_id}_{clause_layer}"
        obj_node = f"{obj_id}_{clause_layer}"

        if not self.graph.has_node(subj_node) or not self.graph.has_node(obj_node):
            return

        self.graph.add_edge(
            subj_node,
            obj_node,
            type=rel_type,
            text=rel_text,
            clause_layer=clause_layer
        )

        if rel_type == "association":
            self.graph.add_edge(
                obj_node,
                subj_node,
                type=rel_type,
                text=rel_text,
                clause_layer=clause_layer
            )

    def run_compound_sentiment_calculations(
        self,
        actor_func: Callable,
        target_func: Callable,
        assoc_func: Callable,
        parent_func: Callable,
        child_func: Callable
    ):
        """Run role-aware compound sentiment calculations"""
        # Process action relationships
        for u, v, data in self.graph.edges(data=True):
            if data.get('type') == 'action':
                actor_node = data.get('actor')
                target_node = data.get('target')
                action_sentiment = data.get('init_sentiment', 0.0)
                
                if actor_node and target_node:
                    actor_init = self.graph.nodes[actor_node]['init_sentiment']
                    target_init = self.graph.nodes[target_node]['init_sentiment']
                    
                    try:
                        # Apply actor formula: (actor_init, action_sentiment)
                        new_actor_sentiment = actor_func([actor_init, action_sentiment])
                        # Apply target formula: (target_init, action_sentiment)
                        new_target_sentiment = target_func([target_init, action_sentiment])
                        
                        if not np.isnan(new_actor_sentiment):
                            self.graph.nodes[actor_node]['compound_sentiment'] = new_actor_sentiment
                        if not np.isnan(new_target_sentiment):
                            self.graph.nodes[target_node]['compound_sentiment'] = new_target_sentiment
                            
                    except Exception as e:
                        logger.error(f"Error in action sentiment calculation: {e}")

        # Process belonging relationships
        for u, v, data in self.graph.edges(data=True):
            if data.get('type') == 'belonging':
                parent_node = data.get('parent')
                child_node = data.get('child')
                
                if parent_node and child_node:
                    parent_init = self.graph.nodes[parent_node]['compound_sentiment']
                    child_init = self.graph.nodes[child_node]['compound_sentiment']
                    
                    try:
                        # Apply belonging formulas
                        new_parent_sentiment = parent_func([parent_init, child_init])
                        new_child_sentiment = child_func([child_init, parent_init])
                        
                        if not np.isnan(new_parent_sentiment):
                            self.graph.nodes[parent_node]['compound_sentiment'] = new_parent_sentiment
                        if not np.isnan(new_child_sentiment):
                            self.graph.nodes[child_node]['compound_sentiment'] = new_child_sentiment
                            
                    except Exception as e:
                        logger.error(f"Error in belonging sentiment calculation: {e}")

        # Process association relationships
        processed_associations = set()
        for u, v, data in self.graph.edges(data=True):
            if data.get('type') == 'association':
                entity1_node = data.get('entity1')
                entity2_node = data.get('entity2')
                
                # Avoid processing the same association twice (bidirectional)
                edge_key = tuple(sorted([entity1_node, entity2_node]))
                if edge_key in processed_associations:
                    continue
                processed_associations.add(edge_key)
                
                if entity1_node and entity2_node:
                    entity1_init = self.graph.nodes[entity1_node]['compound_sentiment']
                    entity2_init = self.graph.nodes[entity2_node]['compound_sentiment']
                    
                    try:
                        # Apply association formulas
                        new_entity1_sentiment = assoc_func([entity1_init, entity2_init])
                        new_entity2_sentiment = assoc_func([entity2_init, entity1_init])
                        
                        if not np.isnan(new_entity1_sentiment):
                            self.graph.nodes[entity1_node]['compound_sentiment'] = new_entity1_sentiment
                        if not np.isnan(new_entity2_sentiment):
                            self.graph.nodes[entity2_node]['compound_sentiment'] = new_entity2_sentiment
                            
                    except Exception as e:
                        logger.error(f"Error in association sentiment calculation: {e}")

    def run_aggregate_sentiment_calculations(
        self,
        agg_func: Callable
    ) -> Dict[str, float]:
        final_sentiments = {}
        nodes_by_entity = defaultdict(list)
        for node, data in self.graph.nodes(data=True):
            nodes_by_entity[data['entity_id']].append(data)

        for entity_id, nodes in nodes_by_entity.items():
            canonical_name = nodes[0].get("head") if nodes else None
            if not canonical_name:
                continue

            all_scores = [node['compound_sentiment'] for node in nodes]

            if not all_scores:
                final_sentiments[canonical_name] = 0.0
            elif len(all_scores) == 1:
                final_sentiments[canonical_name] = all_scores[0]
            else:
                try:
                    final_sentiments[canonical_name] = agg_func(all_scores)
                except Exception as e:
                    logger.warning(f"Aggregation failed for {canonical_name}: {e}, using mean")
                    final_sentiments[canonical_name] = np.mean(all_scores)
        
        return final_sentiments

class GraphVisualizer:
    def __init__(self, graph: RelationGraph, clusters: Dict[int, Dict[str, Any]]):
        self.graph = graph.graph
        self.clusters = clusters

    def draw_graph(self, save_path: str = "relation_graph.html"):
        try:
            from pyvis.network import Network
        except ImportError:
            logger.warning("pyvis not installed. Cannot visualize graph. Run 'pip install pyvis'")
            return

        net = Network(height="800px", width="100%", directed=True, notebook=False)

        for node, attrs in self.graph.nodes(data=True):
            entity_id = attrs.get('entity_id')
            name = self.clusters.get(entity_id, {}).get("canonical_name", "Unknown")
            layer = attrs.get('clause_layer', -1)
            init_sent = attrs.get('init_sentiment', 0.0)
            comp_sent = attrs.get('compound_sentiment', 0.0)
            
            title = (
                f"ID: {entity_id}\n"
                f"Name: {name}\n"
                f"Layer: {layer}\n"
                f"Init Sentiment: {init_sent:.2f}\n"
                f"Final Sentiment: {comp_sent:.2f}\n"
                f"Modifiers: {attrs.get('modifiers')}"
            )
            
            color = self._get_color_for_sentiment(comp_sent)
            
            net.add_node(node, label=f"{name}_{layer}", title=title, color=color)

        for u, v, attrs in self.graph.edges(data=True):
            edge_type = attrs.get('type', 'unknown')
            title = f"Type: {edge_type}"
            color, style, arrows = self._get_edge_style(edge_type)
            
            if edge_type != 'temporal':
                title += f"\nText: {attrs.get('text', '')}"

            net.add_edge(u, v, title=title, color=color, dashes=(style=='dashed'), arrows=arrows)

        net.set_options("""
        var options = {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -30000,
              "centralGravity": 0.1,
              "springLength": 150
            },
            "minVelocity": 0.75
          }
        }
        """)
        net.save_graph(save_path)
        logger.info(f"Graph visualization saved to {save_path}")

    @staticmethod
    def _get_color_for_sentiment(sentiment: float) -> str:
        if sentiment > 0.1:
            return "#a1d99b"  # green
        elif sentiment < -0.1:
            return "#fb6a4a"  # red
        else:
            return "#bdbdbd"  # grey

    @staticmethod
    def _get_edge_style(edge_type: str) -> Tuple[str, str, bool]:
        if edge_type == 'temporal':
            return "#d9d9d9", 'dashed', False
        elif edge_type == 'action':
            return "#3182bd", 'solid', True
        elif edge_type == 'association':
            return "#fec44f", 'solid', False
        elif edge_type == 'belonging':
            return "#756bb1", 'solid', True
        else:
            return "#000000", 'solid', True