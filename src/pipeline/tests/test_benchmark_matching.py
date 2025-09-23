import sys
import types
import unittest
from types import SimpleNamespace


def _install_benchmark_stubs():
    if "matplotlib" not in sys.modules:
        matplotlib_stub = types.ModuleType("matplotlib")
        matplotlib_stub.use = lambda *_args, **_kwargs: None
        sys.modules["matplotlib"] = matplotlib_stub
    if "matplotlib.pyplot" not in sys.modules:
        sys.modules["matplotlib.pyplot"] = types.ModuleType("matplotlib.pyplot")
    if "seaborn" not in sys.modules:
        sys.modules["seaborn"] = types.ModuleType("seaborn")
    if "sklearn" not in sys.modules:
        sys.modules["sklearn"] = types.ModuleType("sklearn")
    if "sklearn.metrics" not in sys.modules:
        metrics_stub = types.ModuleType("sklearn.metrics")

        def _stub_metric(*_args, **_kwargs):
            return 0.0

        metrics_stub.accuracy_score = _stub_metric
        metrics_stub.balanced_accuracy_score = _stub_metric
        metrics_stub.classification_report = _stub_metric
        metrics_stub.confusion_matrix = _stub_metric
        sys.modules["sklearn.metrics"] = metrics_stub


_install_benchmark_stubs()

from src.pipeline.benchmark import (
    PredictedAspect,
    GoldAspect,
    _lenient_similarity,
    _prepare_predicted,
    _tokens,
    _normalize_text,
    _head,
)


def _build_predicted(entity_id: int, text: str, score: float) -> PredictedAspect:
    tokens = _tokens(text)
    return PredictedAspect(
        entity_id=entity_id,
        canonical=text,
        polarity="positive" if score > 0 else "negative" if score < 0 else "neutral",
        score=score,
        mentions=[{"text": text, "clause_index": 0}],
        norm=_normalize_text(text),
        tokens=tokens,
        head=_head(tokens, text),
    )


def _build_gold(text: str, polarity: str = "positive") -> GoldAspect:
    annotation = SimpleNamespace(term=text, polarity=polarity, start=0, end=len(text))
    tokens = _tokens(text)
    return GoldAspect(
        annotation=annotation,
        norm=_normalize_text(text),
        tokens=tokens,
        head=_head(tokens, text),
    )


class BenchmarkMatchingTests(unittest.TestCase):
    def test_lenient_similarity_rewards_token_overlap(self):
        predicted = _build_predicted(1, "speedy WiFi connection", 0.6)
        gold = _build_gold("WiFi connection")
        score = _lenient_similarity(predicted, gold)
        self.assertGreater(score, 0.9)

    def test_lenient_similarity_penalizes_unrelated_words(self):
        predicted = _build_predicted(1, "I", 0.1)
        gold = _build_gold("WiFi connection")
        score = _lenient_similarity(predicted, gold)
        self.assertLess(score, 0.5)

    def test_prepare_predicted_fallback_creates_sentence_level_aspect(self):
        fallback_text = "Set up was easy."

        class DummyGraph:
            def compute_text_sentiment(self, text):
                if text != fallback_text:
                    raise AssertionError("unexpected text")
                return 0.75

        predicted = _prepare_predicted(
            {},
            pos_thresh=0.1,
            neg_thresh=-0.1,
            fallback_text=fallback_text,
            graph=DummyGraph(),
        )

        self.assertEqual(len(predicted), 1)
        aspect = predicted[0]
        self.assertEqual(aspect.canonical, fallback_text)
        self.assertEqual(aspect.polarity, "positive")


if __name__ == "__main__":
    unittest.main()
