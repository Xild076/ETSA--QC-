"""
Analyze different sentiment analyzer configurations to find the best performer.
Tests various sentiment analysis methods and weights on the test harness.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.sentiment_analysis import MultiSentimentAnalysis
from pipeline.graph import RelationGraph
from tools.test_modifier_optimization import TEST_CASES, ModifierTestCase


@dataclass
class SentimentConfig:
    """Configuration for sentiment analyzer"""
    name: str
    methods: List[str]
    weights: List[float]
    description: str


# Test configurations
SENTIMENT_CONFIGS = [
    SentimentConfig(
        name="current",
        methods=['distilbert_logit', 'flair', 'pysentimiento', 'vader'],
        weights=[0.55, 0.2, 0.15, 0.1],
        description="Current weighted ensemble"
    ),
    SentimentConfig(
        name="distilbert_only",
        methods=['distilbert_logit'],
        weights=[1.0],
        description="DistilBERT only (fastest, transformer-based)"
    ),
    SentimentConfig(
        name="flair_only",
        methods=['flair'],
        weights=[1.0],
        description="Flair only (context-aware)"
    ),
    SentimentConfig(
        name="pysentimiento_only",
        methods=['pysentimiento'],
        weights=[1.0],
        description="PySentimiento only (Spanish/English)"
    ),
    SentimentConfig(
        name="vader_only",
        methods=['vader'],
        weights=[1.0],
        description="VADER only (lexicon-based)"
    ),
    SentimentConfig(
        name="transformers_heavy",
        methods=['distilbert_logit', 'flair', 'pysentimiento'],
        weights=[0.5, 0.3, 0.2],
        description="Transformers-only ensemble (no VADER)"
    ),
    SentimentConfig(
        name="balanced_ensemble",
        methods=['distilbert_logit', 'flair', 'pysentimiento', 'vader'],
        weights=[0.4, 0.3, 0.2, 0.1],
        description="More balanced weights"
    ),
    SentimentConfig(
        name="flair_heavy",
        methods=['distilbert_logit', 'flair', 'pysentimiento', 'vader'],
        weights=[0.3, 0.5, 0.1, 0.1],
        description="Flair-weighted for context sensitivity"
    ),
]


def test_sentiment_config(config: SentimentConfig, test_cases: List[ModifierTestCase]) -> Dict[str, Any]:
    """Test a sentiment configuration on test cases"""
    print(f"\nTesting: {config.name}")
    print(f"  {config.description}")
    print(f"  Methods: {config.methods}")
    print(f"  Weights: {config.weights}")
    
    # Initialize sentiment analyzer
    sentiment_system = MultiSentimentAnalysis(
        methods=config.methods,
        weights=config.weights
    )
    
    results = []
    correct = 0
    
    for test_case in test_cases:
        # Create graph and add entity with empty modifiers to test pure entity sentiment
        graph = RelationGraph(test_case.sentence, [test_case.sentence], sentiment_system)
        graph.add_entity_node(
            id=1,
            head=test_case.entity,
            modifier=[],  # Test with no modifiers to see entity sentiment
            entity_role='associate',
            clause_layer=0
        )
        
        node_key = (1, 0)
        node_data = graph.graph.nodes.get(node_key, {})
        init_sentiment = node_data.get("init_sentiment", 0.0)
        
        # Determine polarity
        if init_sentiment > 0.15:
            predicted = "positive"
        elif init_sentiment < -0.15:
            predicted = "negative"
        else:
            predicted = "neutral"
        
        is_correct = predicted == test_case.expected_polarity
        if is_correct:
            correct += 1
        
        results.append({
            "test_name": test_case.name,
            "entity": test_case.entity,
            "expected": test_case.expected_polarity,
            "predicted": predicted,
            "score": init_sentiment,
            "correct": is_correct
        })
    
    accuracy = correct / len(test_cases) if test_cases else 0
    
    # Category breakdown
    category_stats = {}
    for test_case, result in zip(test_cases, results):
        cat = test_case.category
        if cat not in category_stats:
            category_stats[cat] = {"correct": 0, "total": 0}
        category_stats[cat]["total"] += 1
        if result["correct"]:
            category_stats[cat]["correct"] += 1
    
    print(f"  Accuracy: {accuracy:.1%} ({correct}/{len(test_cases)})")
    
    return {
        "config_name": config.name,
        "description": config.description,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(test_cases),
        "results": results,
        "category_stats": category_stats
    }


def analyze_failure_patterns(all_results: Dict[str, Any]):
    """Analyze which test cases are consistently failing across configs"""
    print("\n" + "="*80)
    print("FAILURE PATTERN ANALYSIS")
    print("="*80)
    
    # Collect all test names
    test_names = set()
    for config_result in all_results.values():
        for result in config_result["results"]:
            test_names.add(result["test_name"])
    
    # Count failures per test
    failure_counts = {}
    for test_name in test_names:
        failures = 0
        for config_result in all_results.values():
            for result in config_result["results"]:
                if result["test_name"] == test_name and not result["correct"]:
                    failures += 1
        failure_counts[test_name] = failures
    
    # Sort by failure count
    sorted_failures = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("\nMost Problematic Test Cases:")
    print(f"{'Test Name':<35} {'Failures':<10} {'Success Rate':<15}")
    print("-" * 80)
    
    for test_name, fail_count in sorted_failures[:10]:
        success_rate = 1 - (fail_count / len(all_results))
        print(f"{test_name:<35} {fail_count}/{len(all_results):<9} {success_rate:>13.1%}")


def run_sentiment_analysis():
    """Run sentiment analyzer comparison"""
    print("="*80)
    print("SENTIMENT ANALYZER COMPARISON")
    print("="*80)
    print(f"\nTesting {len(SENTIMENT_CONFIGS)} configurations on {len(TEST_CASES)} test cases")
    
    all_results = {}
    
    for config in SENTIMENT_CONFIGS:
        result = test_sentiment_config(config, TEST_CASES)
        all_results[config.name] = result
    
    # Comparison table
    print("\n" + "="*80)
    print("CONFIGURATION COMPARISON")
    print("="*80)
    print(f"\n{'Configuration':<25} {'Accuracy':<12} {'Description':<40}")
    print("-" * 80)
    
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    
    for config_name, result in sorted_results:
        print(f"{config_name:<25} {result['accuracy']:>10.1%}  {result['description']}")
    
    # Best configuration
    best_config_name, best_result = sorted_results[0]
    print(f"\n🏆 BEST CONFIGURATION: {best_config_name}")
    print(f"   Accuracy: {best_result['accuracy']:.1%}")
    print(f"   Description: {best_result['description']}")
    
    # Category breakdown for best
    print(f"\n   Performance by Category:")
    for cat, stats in sorted(best_result["category_stats"].items(), 
                             key=lambda x: x[1]["correct"]/x[1]["total"] if x[1]["total"] > 0 else 0):
        cat_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"     {cat:<30} {cat_acc:>6.1%}  ({stats['correct']}/{stats['total']})")
    
    # Failure analysis
    analyze_failure_patterns(all_results)
    
    # Specific problem cases for best config
    print(f"\n   Failed Cases for Best Config ({best_config_name}):")
    failed = [r for r in best_result["results"] if not r["correct"]]
    if failed:
        for fail in failed:
            print(f"     - {fail['test_name']}: {fail['entity']}")
            print(f"       Expected: {fail['expected']}, Predicted: {fail['predicted']}, Score: {fail['score']:.3f}")
    else:
        print("     None! Perfect accuracy!")
    
    print("\n" + "="*80)
    
    return all_results


if __name__ == "__main__":
    results = run_sentiment_analysis()
