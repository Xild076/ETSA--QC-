import unittest
from src.graph.graph import RelationGraph

class DummyAnalyzer:
    def analyze_sentiment(self, text):
        return 0.0

class TestRelationGraph(unittest.TestCase):
    def test_basic_nodes_edges_and_compound(self):
        g = RelationGraph(text="t", sentiment_analyzer_system=DummyAnalyzer())
        g.add_entity_node(id=1, head="Alice", modifier=["brilliant"], entity_role="actor", clause_layer=0)
        g.add_entity_node(id=2, head="Project", modifier=["valuable"], entity_role="target", clause_layer=0)
        g.add_entity_node(id=3, head="Company", modifier=["big"], entity_role="parent", clause_layer=0)
        g.add_action_edge(actor_id=1, target_id=2, clause_layer=0, head="approves", modifier=["quickly"])
        g.add_belonging_edge(parent_id=3, child_id=2, clause_layer=0)
        g.add_association_edge(entity1_id=1, entity2_id=3, clause_layer=0)
        def action_fn(a, act, t):
            return a + 0.1, t + 0.2
        def belong_fn(p, c):
            return p + 0.3, c + 0.4
        def assoc_fn(e1, e2):
            m = (e1 + e2) / 2
            return m, m
        g.run_compound_action_sentiment_calculations(function=action_fn)
        g.run_compound_belonging_sentiment_calculations(function=belong_fn)
        g.run_compound_association_sentiment_calculations(function=assoc_fn)
        self.assertIn("1_0", g.graph.nodes)
        self.assertIn("2_0", g.graph.nodes)
        self.assertIn("3_0", g.graph.nodes)
        self.assertIn("compound_sentiment", g.graph.nodes["1_0"])  
        self.assertIn("compound_sentiment", g.graph.nodes["2_0"])  
        self.assertIn("compound_sentiment", g.graph.nodes["3_0"])  
        def agg_fn(vals):
            return sum(vals) / len(vals) if vals else 0.0
        v1 = g.run_aggregate_sentiment_calculations(entity_id=1, function=agg_fn)
        v2 = g.run_aggregate_sentiment_calculations(entity_id=2, function=agg_fn)
        v3 = g.run_aggregate_sentiment_calculations(entity_id=3, function=agg_fn)
        self.assertIsInstance(v1, float)
        self.assertIsInstance(v2, float)
        self.assertIsInstance(v3, float)

if __name__ == "__main__":
    unittest.main()
