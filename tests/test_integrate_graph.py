import unittest
from unittest.mock import patch
from src.graph.integrate_graph import build_graph, build_graph_with_optimal_functions
from src.graph.graph import RelationGraph

class DummyAnalyzer:
    def analyze_sentiment(self, text):
        return 0.0

def dummy_action(a, act, t):
    return a, t

def dummy_assoc(e1, e2):
    return e1, e2

def dummy_belong(p, c):
    return p, c

def dummy_agg(vals):
    return sum(vals) / len(vals) if vals else 0.0

class TestIntegrateGraph(unittest.TestCase):
    @patch("src.graph.integrate_graph.resolve", autospec=True)
    @patch("src.graph.integrate_graph.re_api", autospec=True)
    @patch("src.graph.integrate_graph.PresetEnsembleSentimentAnalyzer", autospec=True)
    def test_build_graph_happy_path(self, mock_analyzer_cls, mock_re_api, mock_resolve):
        mock_analyzer = mock_analyzer_cls.return_value
        mock_analyzer.analyze_sentiment.return_value = 0.0
        clusters = {
            1: {"entity_references": [("Alice", (0, 0))]},
            2: {"entity_references": [("Project", (0, 10))]},
        }
        sent_map = {
            "clause_0": {
                "entities": [("Alice", (0, 0)), ("Project", (0, 10))]
            }
        }
        mock_resolve.return_value = (clusters, sent_map)
        mock_re_api.return_value = {
            "relations": [
                {
                    "relation": {"type": "action", "text": "approved"},
                    "subject": {"head": "Alice"},
                    "object": {"head": "Project"}
                }
            ]
        }
        g, results = build_graph(
            text="Alice approved the project.",
            action_function=dummy_action,
            association_function=dummy_assoc,
            belonging_function=dummy_belong,
            aggregate_function=dummy_agg,
        )
        self.assertIsInstance(g, RelationGraph)
        self.assertTrue(results)
        self.assertIn("Alice", next(iter(results.keys())))

    @patch("src.graph.integrate_graph.get_actor_function", autospec=True)
    @patch("src.graph.integrate_graph.get_target_function", autospec=True)
    @patch("src.graph.integrate_graph.get_association_function", autospec=True)
    @patch("src.graph.integrate_graph.get_parent_function", autospec=True)
    @patch("src.graph.integrate_graph.get_child_function", autospec=True)
    @patch("src.graph.integrate_graph.get_aggregate_function", autospec=True)
    @patch("src.graph.integrate_graph.build_graph", autospec=True)
    def test_build_graph_with_optimal_functions(self, mock_build, mock_get_agg, mock_get_child, mock_get_parent, mock_get_assoc, mock_get_target, mock_get_actor):
        mock_get_actor.return_value = lambda a, act, t: a
        mock_get_target.return_value = lambda t, act: t
        mock_get_assoc.return_value = lambda e1, e2: (e1 + e2) / 2
        mock_get_parent.return_value = lambda p, c: p
        mock_get_child.return_value = lambda c, p: c
        mock_get_agg.return_value = lambda vals: sum(vals) / len(vals) if vals else 0.0
        mock_build.return_value = (RelationGraph(text="t", sentiment_analyzer_system=DummyAnalyzer()), {"x": 0.0})
        g, results = build_graph_with_optimal_functions("t")
        self.assertIsNotNone(g)
        self.assertTrue(results)
        self.assertTrue(mock_build.called)

if __name__ == "__main__":
    unittest.main()
