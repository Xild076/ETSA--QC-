import networkx as nx

class RelationGraph:
    def __init__(self, text=None, sentiment_analyzer_system=None):
        self.text = text
        self.graph = nx.MultiDiGraph()
        self.sentiment_analyzer_system = sentiment_analyzer_system

    def _node_key(self, entity_id, clause_layer):
        return f"{entity_id}_{clause_layer}"

    def add_entity_node(self, id, head, modifier, entity_role, clause_layer):
        key = self._node_key(id, clause_layer)
        self.graph.add_node(key, id=id, head=head, modifier=modifier, entity_role=entity_role, clause_layer=clause_layer)

    def add_action_edge(self, actor_id, target_id, clause_layer, head, modifier=None):
        src = self._node_key(actor_id, clause_layer)
        dst = self._node_key(target_id, clause_layer)
        self.graph.add_edge(src, dst, type="action", head=head, modifier=modifier or [], clause_layer=clause_layer)

    def add_belonging_edge(self, parent_id, child_id, clause_layer):
        src = self._node_key(parent_id, clause_layer)
        dst = self._node_key(child_id, clause_layer)
        self.graph.add_edge(src, dst, type="belonging", clause_layer=clause_layer)

    def add_association_edge(self, entity1_id, entity2_id, clause_layer):
        a = self._node_key(entity1_id, clause_layer)
        b = self._node_key(entity2_id, clause_layer)
        self.graph.add_edge(a, b, type="association", clause_layer=clause_layer)
        self.graph.add_edge(b, a, type="association", clause_layer=clause_layer)

    def run_compound_action_sentiment_calculations(self, function):
        for u, v, data in self.graph.edges(data=True):
            if data.get("type") != "action":
                continue
            a_node = self.graph.nodes[u]
            t_node = self.graph.nodes[v]
            a = self.sentiment_analyzer_system.analyze_sentiment(" ".join(a_node.get("modifier", []))) if self.sentiment_analyzer_system else 0.0
            t = self.sentiment_analyzer_system.analyze_sentiment(" ".join(t_node.get("modifier", []))) if self.sentiment_analyzer_system else 0.0
            a_new, t_new = function(a, 0.0, t)
            a_node.setdefault("compound_sentiment", []).append(a_new)
            t_node.setdefault("compound_sentiment", []).append(t_new)

    def run_compound_belonging_sentiment_calculations(self, function):
        for u, v, data in self.graph.edges(data=True):
            if data.get("type") != "belonging":
                continue
            p_node = self.graph.nodes[u]
            c_node = self.graph.nodes[v]
            p = self.sentiment_analyzer_system.analyze_sentiment(" ".join(p_node.get("modifier", []))) if self.sentiment_analyzer_system else 0.0
            c = self.sentiment_analyzer_system.analyze_sentiment(" ".join(c_node.get("modifier", []))) if self.sentiment_analyzer_system else 0.0
            p_new, c_new = function(p, c)
            p_node.setdefault("compound_sentiment", []).append(p_new)
            c_node.setdefault("compound_sentiment", []).append(c_new)

    def run_compound_association_sentiment_calculations(self, function):
        for u, v, data in self.graph.edges(data=True):
            if data.get("type") != "association":
                continue
            n1 = self.graph.nodes[u]
            n2 = self.graph.nodes[v]
            s1 = self.sentiment_analyzer_system.analyze_sentiment(" ".join(n1.get("modifier", []))) if self.sentiment_analyzer_system else 0.0
            s2 = self.sentiment_analyzer_system.analyze_sentiment(" ".join(n2.get("modifier", []))) if self.sentiment_analyzer_system else 0.0
            r1, r2 = function(s1, s2)
            n1.setdefault("compound_sentiment", []).append(r1)
            n2.setdefault("compound_sentiment", []).append(r2)

    def run_aggregate_sentiment_calculations(self, entity_id, function):
        total = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("id") == entity_id:
                total.extend(data.get("compound_sentiment", []))
        return function(total)

    def add_entity_modifier(self, entity_id, modifier, clause_layer):
        """Add modifiers to an entity node"""
        key = self._node_key(entity_id, clause_layer)
        if key in self.graph.nodes:
            current_modifiers = self.graph.nodes[key].get("modifier", [])
            # Extend with new modifiers, avoiding duplicates
            for mod in modifier:
                if mod not in current_modifiers:
                    current_modifiers.append(mod)
            self.graph.nodes[key]["modifier"] = current_modifiers

    def set_entity_role(self, entity_id, role, clause_layer):
        """Set the role of an entity (actor, target, associate)"""
        key = self._node_key(entity_id, clause_layer)
        if key in self.graph.nodes:
            self.graph.nodes[key]["entity_role"] = role

    def add_temporal_edge(self, entity_id):
        """Add temporal edges connecting same entity across clause layers"""
        # Find all nodes for this entity across different layers
        entity_nodes = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("id") == entity_id:
                entity_nodes.append(node_id)
        
        # Connect consecutive layers
        entity_nodes.sort(key=lambda x: int(x.split('_')[1]))  # Sort by layer
        for i in range(len(entity_nodes) - 1):
            self.graph.add_edge(entity_nodes[i], entity_nodes[i+1], type="temporal")

    @property
    def entity_ids(self):
        """Get all unique entity IDs in the graph"""
        ids = set()
        for node_id, data in self.graph.nodes(data=True):
            if "id" in data:
                ids.add(data["id"])
        return list(ids)
