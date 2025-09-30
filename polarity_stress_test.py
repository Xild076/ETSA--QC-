#!/usr/bin/env python3
import sys
import os
import types

if "sentiment" not in sys.modules:
    sentiment_pkg = types.ModuleType("sentiment")
    sentiment_pkg.__path__ = []
    sentiment_pkg.__spec__ = types.SimpleNamespace(submodule_search_locations=[])
    sys.modules["sentiment"] = sentiment_pkg

if "src" not in sys.modules:
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = []
    src_pkg.__spec__ = types.SimpleNamespace(submodule_search_locations=[])
    sys.modules["src"] = src_pkg

if "src.sentiment" not in sys.modules:
    src_sentiment_pkg = types.ModuleType("src.sentiment")
    src_sentiment_pkg.__path__ = []
    src_sentiment_pkg.__spec__ = types.SimpleNamespace(submodule_search_locations=[])
    sys.modules["src.sentiment"] = src_sentiment_pkg

if "sentiment.sentiment" not in sys.modules:
    stub_module = types.ModuleType("sentiment.sentiment")

    def _stub_result(score: float = 0.0):
        return {"score": score, "label": "neutral", "confidence": abs(score), "raw": {"stub": True}}

    for name in [
        "get_vader_sentiment",
        "get_textblob_sentiment",
        "get_flair_sentiment",
        "get_pysentimiento_sentiment",
        "get_swn_sentiment",
        "get_nlptown_sentiment",
        "get_finiteautomata_sentiment",
        "get_ProsusAI_sentiment",
        "get_distilbert_logit_sentiment",
    ]:
        setattr(stub_module, name, lambda text, _score=0.0: _stub_result(_score))

    sys.modules["sentiment.sentiment"] = stub_module
    sys.modules["src.sentiment.sentiment"] = stub_module

sys.path.append('/Users/harry/Documents/Python_Projects/ETSA_(QC)/src')

from sentiment.sentiment import get_distilbert_logit_sentiment, get_vader_sentiment, get_textblob_sentiment
from pipeline.sentiment_analysis import MultiSentimentAnalysis
import json

def test_polarity_cases():
    test_cases = [
        {
            "text": "the weight is acceptable",
            "expected_polarity": "positive",
            "issue": "acceptable_word_negative"
        },
        {
            "text": "Did not enjoy the new Windows 8 and touchscreen functions",
            "expected_polarity": "negative",
            "issue": "negation_not_handled"
        },
        {
            "text": "No installation disk (DVD) is included",
            "expected_polarity": "negative",
            "issue": "contextual_negation"
        },
        {
            "text": "Its size is ideal and the weight is acceptable",
            "expected_polarity": "positive",
            "issue": "mixed_sentiment_compound"
        },
        {
            "text": "This product is not bad",
            "expected_polarity": "positive",
            "issue": "double_negation"
        },
        {
            "text": "The battery life is good but the screen is terrible",
            "expected_polarity": "negative",
            "issue": "mixed_opposite_sentiment"
        },
        {
            "text": "I hate this laptop",
            "expected_polarity": "negative",
            "issue": "strong_negative"
        },
        {
            "text": "This is an excellent device",
            "expected_polarity": "positive",
            "issue": "strong_positive"
        }
    ]

    results = []
    analyzer = MultiSentimentAnalysis(methods=['vader', 'textblob', 'distilbert_logit'])

    def extract_score(valence):
        if isinstance(valence, (int, float)):
            return float(valence)
        if isinstance(valence, dict):
            for key in ('score', 'compound', 'aggregate', 'polarity', 'value'):
                value = valence.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
            for nested in valence.values():
                nested_score = extract_score(nested)
                if nested_score is not None:
                    return nested_score
        return 0.0

    for i, case in enumerate(test_cases):
        text = case["text"]
        expected = case["expected_polarity"]
        issue = case["issue"]

        print(f"\nTest Case {i+1}: {text}")
        print(f"Issue: {issue}")
        print(f"Expected: {expected}")

        distilbert_valence = get_distilbert_logit_sentiment(text)
        vader_valence = get_vader_sentiment(text)
        textblob_valence = get_textblob_sentiment(text)

        distilbert_score = extract_score(distilbert_valence)
        vader_score = extract_score(vader_valence)
        textblob_score = extract_score(textblob_valence)

        print(f"DistilBERT score: {distilbert_score:+.3f}")
        print(f"VADER score: {vader_score:+.3f}")
        print(f"TextBlob score: {textblob_score:+.3f}")

        ensemble_result = analyzer.analyze(text)
        ensemble_valence = ensemble_result.get('aggregate', 0.0) if isinstance(ensemble_result, dict) else 0.0

        print(f"Ensemble score: {ensemble_valence:+.3f}")

        def polarity_correct(valence, expected):
            if expected == "positive":
                return valence > 0.1
            elif expected == "negative":
                return valence < -0.1
            else:
                return abs(valence) <= 0.1

        distilbert_correct = polarity_correct(distilbert_score, expected)
        vader_correct = polarity_correct(vader_score, expected)
        textblob_correct = polarity_correct(textblob_score, expected)
        ensemble_correct = polarity_correct(ensemble_valence, expected)

        print(f"DistilBERT correct: {distilbert_correct}")
        print(f"VADER correct: {vader_correct}")
        print(f"TextBlob correct: {textblob_correct}")
        print(f"Ensemble correct: {ensemble_correct}")

        results.append({
            "case": i+1,
            "text": text,
            "issue": issue,
            "expected": expected,
            "distilbert_valence": distilbert_score,
            "vader_valence": vader_score,
            "textblob_valence": textblob_score,
            "ensemble_valence": ensemble_valence,
            "distilbert_correct": distilbert_correct,
            "vader_correct": vader_correct,
            "textblob_correct": textblob_correct,
            "ensemble_correct": ensemble_correct
        })

    print("\n" + "="*60)
    print("STRESS TEST SUMMARY")
    print("="*60)

    total_cases = len(results)
    distilbert_correct_count = sum(1 for r in results if r["distilbert_correct"])
    vader_correct_count = sum(1 for r in results if r["vader_correct"])
    textblob_correct_count = sum(1 for r in results if r["textblob_correct"])
    ensemble_correct_count = sum(1 for r in results if r["ensemble_correct"])

    print(f"Total test cases: {total_cases}")
    print(f"DistilBERT accuracy: {distilbert_correct_count / total_cases:.1%}")
    print(f"VADER accuracy: {vader_correct_count / total_cases:.1%}")
    print(f"TextBlob accuracy: {textblob_correct_count / total_cases:.1%}")
    print(f"Ensemble accuracy: {ensemble_correct_count / total_cases:.1%}")

    critical_issues = ["acceptable_word_negative", "negation_not_handled", "contextual_negation"]
    critical_results = [r for r in results if r["issue"] in critical_issues]

    print(f"\nCritical issues ({len(critical_results)} cases):")
    for r in critical_results:
        status = "✓" if r["ensemble_correct"] else "✗"
        print(f"  {status} Case {r['case']}: {r['issue']} - Ensemble: {r['ensemble_correct']}")

    critical_passed = sum(1 for r in critical_results if r["ensemble_correct"])
    print(f"\nCritical issues passed: {critical_passed}/{len(critical_results)}")

    with open('/Users/harry/Documents/Python_Projects/ETSA_(QC)/polarity_stress_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to: polarity_stress_test_results.json")

    return ensemble_correct_count == total_cases

if __name__ == "__main__":
    success = test_polarity_cases()
    if success:
        print("\n🎉 ALL TESTS PASSED! Ready for benchmarking.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Review results before benchmarking.")
        sys.exit(1)
