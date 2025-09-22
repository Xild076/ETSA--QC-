from typing import Callable, List, Dict, Tuple, Optional
import logging
import re
import networkx as nx

logger = logging.getLogger(__name__)

ENTITY_ROLES: Dict[str, Dict[str, str]] = {
    "none": {"description": "No specific role"},
    "actor": {"description": "Initiates an action"},
    "target": {"description": "Receives an action"},
    "parent": {"description": "Owns a child entity"},
    "child": {"description": "Belongs to a parent"},
    "associate": {"description": "Is connected to another entity"},
}

RELATION_TYPES: Dict[str, Dict[str, str]] = {
    "temporal": {"description": "Indicates a temporal relationship"},
    "action": {"description": "Indicates an action between entities"},
    "belonging": {"description": "Indicates a belonging relationship"},
    "association": {"description": "Indicates an association between entities"},
}


class RelationGraph:
    def __init__(self, text: str = "", clauses: List[str] = [], sentiment_analyzer_system=None):
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.text: str = text
        self.clauses: List[str] = clauses
        self.sentiment_analyzer_system = sentiment_analyzer_system
        self.entity_ids: set[int] = set()
        self.aggregate_sentiments: Dict[int, float] = {}

    def _validate_role(self, role: str) -> None:
        if role not in ENTITY_ROLES:
            raise ValueError(f"Invalid entity role: {role}. Valid roles are: {list(ENTITY_ROLES.keys())}")

    def _validate_relation(self, relation: str) -> None:
        if relation not in RELATION_TYPES:
            raise ValueError(f"Invalid relation type: {relation}. Valid relations are: {list(RELATION_TYPES.keys())}")

    def _node_key(self, entity_id: int, clause_layer: int) -> Tuple[int, int]:
        return (entity_id, clause_layer)

    def _assert_node_exists(self, entity_id: int, clause_layer: int) -> None:
        if entity_id not in self.entity_ids:
            raise ValueError(f"Entity ID {entity_id} not found in the graph.")
        key = self._node_key(entity_id, clause_layer)
        if key not in self.graph.nodes:
            raise ValueError(f"Node {key} not found in the graph.")

    def _sent(self, text: str) -> float:
        try:
            res = self.sentiment_analyzer_system.analyze(text or "")
            if isinstance(res, (int, float)):
                return float(res)
            if isinstance(res, dict):
                for key in ("aggregate", "score", "compound", "polarity"):
                    if key in res and isinstance(res[key], (int, float)):
                        return float(res[key])
            return 0.0
        except Exception:
            return 0.0

    def get_new_unique_entity_id(self) -> int:
        new_id = 1
        while new_id in self.entity_ids:
            new_id += 1
        return new_id

    def add_entity_node(self, id: int, head: str, modifier: List[str], entity_role: str, clause_layer: int) -> None:
        logger.info("Adding entity node...")
        self._validate_role(entity_role)
        self.entity_ids.add(id)
        key = self._node_key(id, clause_layer)
        if modifier:
            text_for_sent = (", ".join(modifier) + " " + head).strip()
        else:
            try:
                text_for_sent = self.clauses[clause_layer]
            except Exception:
                text_for_sent = head
        sentiment = self._sent(text_for_sent)
        logger.debug("Entity %s sentiment %.4f context %s", id, sentiment, text_for_sent)
        self.graph.add_node(
            key,
            head=head,
            modifier=list(modifier),
            text=self.text,
            entity_role=entity_role,
            clause_layer=clause_layer,
            init_sentiment=sentiment,
        )
    
    def get_entities_at_layer(self, clause_layer: int) -> List[Dict]:
        entities = []
        for (entity_id, layer), data in self.graph.nodes(data=True):
            if layer == clause_layer:
                entity_info = {
                    "entity_id": entity_id,
                    "head": data.get("head", ""),
                    "modifier": data.get("modifier", []),
                    "entity_role": data.get("entity_role", ""),
                    "clause_layer": layer,
                    "init_sentiment": data.get("init_sentiment", 0.0),
                    "compound_sentiment": data.get("compound_sentiment", None),
                }
                entities.append(entity_info)
        return entities
    
    def get_all_entity_mentions(self, entity_id: int) -> List[Dict]:
        if entity_id not in self.entity_ids:
            raise ValueError(f"Entity ID {entity_id} not found in the graph.")
        mentions = []
        for (eid, layer), data in self.graph.nodes(data=True):
            if eid == entity_id:
                mentions.append({
                    "text": data.get("head", ""),
                    "clause_index": data.get("clause_layer", 0)
                })
        return mentions

    def add_entity_modifier(self, entity_id: int, modifier: List[str], clause_layer: int) -> None:
        logger.info("Adding entity modifiers...")
        self._assert_node_exists(entity_id, clause_layer)
        key = self._node_key(entity_id, clause_layer)
        current_modifiers = list(self.graph.nodes[key].get("modifier", []))
        new_mods = current_modifiers + list(modifier)
        self.graph.nodes[key]["modifier"] = new_mods
        head = self.graph.nodes[key].get("head", "")
        text_for_sent = (" " + ", ".join(new_mods) + head).strip()
        self.graph.nodes[key]["init_sentiment"] = self._sent(text_for_sent)

    def set_entity_role(self, entity_id: int, entity_role: str, clause_layer: int) -> None:
        logger.info("Setting entity role...")
        self._validate_role(entity_role)
        self._assert_node_exists(entity_id, clause_layer)
        key = self._node_key(entity_id, clause_layer)
        self.graph.nodes[key]["entity_role"] = entity_role

    def add_temporal_edge(self, entity_id: int) -> None:
        logger.info("Adding temporal edge...")
        if entity_id not in self.entity_ids:
            raise ValueError(f"Entity ID {entity_id} not found in the graph.")
        layers = sorted(layer for (eid, layer) in self.graph.nodes if eid == entity_id)
        for i in range(len(layers) - 1):
            u = self._node_key(entity_id, layers[i])
            v = self._node_key(entity_id, layers[i + 1])
            self.graph.add_edge(u, v, relation="temporal")

    def add_action_edge(self, actor_id: int, target_id: int, clause_layer: int, head: str, modifier: List[str]) -> None:
        logger.info(f"Adding action edge between {actor_id} and {target_id}...")
        if actor_id not in self.entity_ids or target_id not in self.entity_ids:
            raise ValueError(f"Actor ID {actor_id} or Target ID {target_id} not found in the graph.")
        actor = self._node_key(actor_id, clause_layer)
        target = self._node_key(target_id, clause_layer)
        if actor not in self.graph.nodes or target not in self.graph.nodes:
            raise ValueError(f"Layer {clause_layer} missing actor or target node.")
        action_text = (head + " " + " ".join(modifier)).strip()
        action_sent = self._sent(action_text) if action_text else 0.0
        self.graph.add_edge(
            actor,
            target,
            relation="action",
            actor=actor,
            target=target,
            head=head,
            modifier=list(modifier),
            init_sentiment=action_sent,
        )

    def add_belonging_edge(self, parent_id: int, child_id: int, clause_layer: int) -> None:
        logger.info(f"Adding belonging edge between {parent_id} and {child_id}...")
        if parent_id not in self.entity_ids or child_id not in self.entity_ids:
            raise ValueError(f"Parent ID {parent_id} or Child ID {child_id} not found in the graph.")
        parent = self._node_key(parent_id, clause_layer)
        child = self._node_key(child_id, clause_layer)
        if parent not in self.graph.nodes or child not in self.graph.nodes:
            raise ValueError(f"Layer {clause_layer} missing parent or child node.")
        self.graph.add_edge(parent, child, relation="belonging", parent=parent, child=child)

    def add_association_edge(self, entity1_id: int, entity2_id: int, clause_layer: int) -> None:
        logger.info(f"Adding association edge between {entity1_id} and {entity2_id}...")
        if entity1_id not in self.entity_ids or entity2_id not in self.entity_ids:
            raise ValueError(f"Entity1 ID {entity1_id} or Entity2 ID {entity2_id} not found in the graph.")
        e1 = self._node_key(entity1_id, clause_layer)
        e2 = self._node_key(entity2_id, clause_layer)
        if e1 not in self.graph.nodes or e2 not in self.graph.nodes:
            raise ValueError(f"Layer {clause_layer} missing association endpoints.")
        self.graph.add_edge(e1, e2, relation="association", entity1=e1, entity2=e2)

    def run_compound_action_sentiment_calculations(self, function: Optional[Callable] = None) -> None:
        logger.info("Running compound action sentiment calculations...")
        if function is None:
            raise ValueError("No function provided for compound sentiment calculation")
        for u, v, data in self.graph.edges(data=True):
            if data.get("relation") == "action":
                actor = data["actor"]
                target = data["target"]
                actor_init = float(self.graph.nodes[actor].get("init_sentiment", 0.0))
                action_init = float(data.get("init_sentiment", 0.0))
                target_init = float(self.graph.nodes[target].get("init_sentiment", 0.0))
                a_s, t_s = function(actor_init, action_init, target_init)
                self.graph.nodes[actor]["compound_sentiment"] = float(a_s)
                self.graph.nodes[target]["compound_sentiment"] = float(t_s)

    def run_compound_belonging_sentiment_calculations(self, function: Optional[Callable] = None) -> None:
        logger.info("Running compound belonging sentiment calculations...")
        if function is None:
            raise ValueError("No function provided for compound sentiment calculation")
        for u, v, data in self.graph.edges(data=True):
            if data.get("relation") == "belonging":
                parent = data["parent"]
                child = data["child"]
                p_init = float(self.graph.nodes[parent].get("init_sentiment", 0.0))
                c_init = float(self.graph.nodes[child].get("init_sentiment", 0.0))
                p_s, c_s = function(p_init, c_init)
                self.graph.nodes[parent]["compound_sentiment"] = float(p_s)
                self.graph.nodes[child]["compound_sentiment"] = float(c_s)

    def run_compound_association_sentiment_calculations(self, function: Optional[Callable] = None) -> None:
        logger.info("Running compound association sentiment calculations...")
        if function is None:
            raise ValueError("No function provided for compound sentiment calculation")
        for u, v, data in self.graph.edges(data=True):
            if data.get("relation") == "association":
                e1 = data["entity1"]
                e2 = data["entity2"]
                s1 = float(self.graph.nodes[e1].get("init_sentiment", 0.0))
                s2 = float(self.graph.nodes[e2].get("init_sentiment", 0.0))
                r1, r2 = function(s1, s2)
                self.graph.nodes[e1]["compound_sentiment"] = float(r1)
                self.graph.nodes[e2]["compound_sentiment"] = float(r2)

    def run_aggregate_sentiment_calculations(self, entity_id: int, function: Optional[Callable] = None) -> float:
        logger.info(f"Running aggregate sentiment calculations for entity {entity_id}...")
        if function is None:
            raise ValueError("No function provided for aggregate sentiment calculation")
        layers = sorted(layer for (eid, layer) in self.graph.nodes if eid == entity_id)
        sentiments: List[float] = []
        for layer in layers:
            key = self._node_key(entity_id, layer)
            if "compound_sentiment" in self.graph.nodes[key]:
                sentiments.append(float(self.graph.nodes[key].get("compound_sentiment", 0.0)))
            else:
                sentiments.append(float(self.graph.nodes[key].get("init_sentiment", 0.0)))
        result = float(function(sentiments)) if sentiments else 0.0
        self.aggregate_sentiments[entity_id] = result
        return result


class GraphVisualizer:
    def __init__(self, relation_graph: RelationGraph):
        self.relation_graph = relation_graph
        self.graph: nx.MultiDiGraph = relation_graph.graph
        self.edge_color_map: Dict[str, str] = {
            "temporal": "grey",
            "action": "blue",
            "belonging": "purple",
            "association": "orange",
        }

    def _get_node_colors(self) -> Dict[Tuple[int, int], str]:
        colors: Dict[Tuple[int, int], str] = {}
        for node, data in self.graph.nodes(data=True):
            sentiment = float(data.get("compound_sentiment", data.get("init_sentiment", 0.0)))
            if sentiment > 0.1:
                colors[node] = "green"
            elif sentiment < -0.1:
                colors[node] = "red"
            else:
                colors[node] = "lightgray"
        return colors

    def draw_graph(self, save_path: Optional[str] = None) -> None:
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
        pos_3d: Dict[Tuple[int, int], Tuple[float, float, int]] = {}
        for node, data in self.graph.nodes(data=True):
            x, y = pos_2d[node]
            z = int(data.get("clause_layer", 0))
            pos_3d[node] = (x, y, z)
        edge_traces = []
        edges_by_relation: Dict[str, List[Tuple[Tuple[int, int], Tuple[int, int]]]] = {rel: [] for rel in self.edge_color_map}
        for u, v, data in self.graph.edges(data=True):
            rel = data.get("relation")
            if rel in edges_by_relation:
                edges_by_relation[rel].append((u, v))
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
                x=edge_x,
                y=edge_y,
                z=edge_z,
                line=dict(width=3, color=self.edge_color_map[relation]),
                hoverinfo="none",
                mode="lines",
                name=f"{relation.capitalize()} Relation",
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
            sentiment = float(data.get("compound_sentiment", data.get("init_sentiment", 0.0)))
            sentiment_str = f"{sentiment:.2f}"
            if "compound_sentiment" in data:
                sentiment_str += f" (compounded from {float(data.get('init_sentiment', 0.0)):.2f})"
            label = f"{data.get('head','')} ({node[0]}_{node[1]})"
            hover_info = (
                f"<b>{label}</b><br>"
                f"Role: {data.get('entity_role','')}<br>"
                f"Layer: {data.get('clause_layer',0)}<br>"
                f"Sentiment: {sentiment_str}"
            )
            node_text.append(hover_info)
        node_trace = go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers",
            hoverinfo="text",
            text=node_text,
            marker=dict(color=node_colors, size=15, line=dict(width=1, color="black")),
            name="Entities",
        )
        fig = go.Figure(data=edge_traces + [node_trace])
        max_layer = max(z for _, (_, _, z) in pos_3d.items())
        fig.update_layout(
            title="3D Relation Graph (Warning: Not using the real sentiment formulas)",
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.6)"),
            margin=dict(l=0, r=0, b=0, t=40),
            scene=dict(
                xaxis=dict(showticklabels=False, title=""),
                yaxis=dict(showticklabels=False, title=""),
                zaxis=dict(title="Clause Layer", nticks=int(max_layer) + 1),
            ),
        )
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive graph saved to {save_path}")
        else:
            fig.show()
