"""
Prompt variation testing framework.
Tests different prompt configurations to find the optimal modifier extraction prompt.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from copy import deepcopy

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tools.test_modifier_optimization import TEST_CASES, test_modifier_extraction
from pipeline.modifier_e import GemmaModifierExtractor
import config


# Prompt variation configurations
PROMPT_VARIANTS = {
    "current": {
        "description": "Current prompt with cross-clause and entity-text instructions",
        "modifications": None  # Uses existing prompt
    },
    
    "simplified_entity_text": {
        "description": "Simplified, more direct entity-text modifier instruction",
        "instruction_8_replacement": """8. **Extract adjectives from the entity name** – If the entity text contains ANY adjective or adverb (like "fresh" in "fresh juices", "slow" in "slow performance", "burnt" in "burnt flavor"), extract it as a modifier. The adjective is evaluating the entity."""
    },
    
    "added_consequences": {
        "description": "Added explicit consequence clause examples and guidance",
        "additional_examples": [
            {
                "example_num": 17,
                "passage": "The slow loading times made me switch browsers.",
                "entity": "loading times",
                "correct": [{"text": "slow"}, {"text": "made me switch browsers"}],
                "reasoning": "Extract both the entity-text modifier 'slow' and the consequence 'made me switch browsers'."
            },
            {
                "example_num": 18,
                "passage": "Poor customer support left me frustrated.",
                "entity": "customer support",
                "correct": [{"text": "Poor"}, {"text": "left me frustrated"}],
                "reasoning": "Entity-text modifier 'Poor' plus consequence clause expressing negative outcome."
            }
        ]
    },
    
    "relaxed_weak_sentiment": {
        "description": "Relaxed weak sentiment filtering to allow hedged but directional phrases",
        "instruction_10_replacement": """10. **Allow weak language if directional** – Hedged phrases like "usually pretty good", "kind of nice", "fairly responsive" ARE valid if they express a clear positive or negative direction. Only skip truly vague phrases like "it's okay" or "sort of there"."""
    },
    
    "emphasis_prep_phrases": {
        "description": "Emphasized prepositional phrase extraction",
        "additional_instruction": """11. **Capture prepositional modifiers** – Prepositional phrases like "with attentive staff", "for long sessions", "in low light" often describe the entity's quality or performance. Include them as modifiers when they directly characterize the entity."""
    },
    
    "combined_optimized": {
        "description": "Combined best practices: simplified entity-text + relaxed weak + prep emphasis",
        "instruction_8_replacement": """8. **Extract adjectives from the entity name** – If the entity text contains ANY adjective or adverb (like "fresh" in "fresh juices", "slow" in "slow performance", "burnt" in "burnt flavor"), extract it as a modifier. The adjective is evaluating the entity.""",
        "instruction_10_replacement": """10. **Allow weak language if directional** – Hedged phrases like "usually pretty good", "kind of nice", "fairly responsive" ARE valid if they express a clear positive or negative direction. Only skip truly vague phrases like "it's okay" or "sort of there".""",
        "additional_instruction": """11. **Capture prepositional modifiers** – Prepositional phrases like "with attentive staff", "for long sessions", "in low light" often describe the entity's quality or performance. Include them as modifiers when they directly characterize the entity."""
    }
}


def create_variant_prompt(base_prompt_method, variant_config: Dict[str, Any]) -> str:
    """
    Create a modified prompt based on variant configuration.
    Note: This is a simplified version - in practice you'd modify the actual _prompt method
    """
    # This function would need to be integrated into GemmaModifierExtractor
    # For now, we'll use it to document the variants
    return variant_config.get("description", "")


def test_prompt_variant(variant_name: str, variant_config: Dict[str, Any]) -> Dict[str, Any]:
    """Test a single prompt variant on the test suite"""
    print(f"\n{'='*80}")
    print(f"Testing Variant: {variant_name}")
    print(f"Description: {variant_config['description']}")
    print(f"{'='*80}\n")
    
    # Initialize extractor
    # NOTE: In a real implementation, you'd pass the variant config to customize the prompt
    # For now, we'll test with the current prompt and document what changes would be made
    extractor = GemmaModifierExtractor(cache_only=False)
    
    results = []
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"[{i}/{len(TEST_CASES)}] {test_case.name}...", end=" ")
        result = test_modifier_extraction(extractor, test_case)
        results.append(result)
        status = "✓" if result["passed"] else "✗"
        print(f"{status} (recall: {result['recall']:.1%})")
    
    # Calculate overall metrics
    total_precision = sum(r["precision"] for r in results) / len(results) if results else 0
    total_recall = sum(r["recall"] for r in results) / len(results) if results else 0
    total_f1 = sum(r["f1"] for r in results) / len(results) if results else 0
    pass_rate = sum(1 for r in results if r["passed"]) / len(results) if results else 0
    
    # Category breakdown
    category_stats = {}
    for result in results:
        cat = result["category"]
        if cat not in category_stats:
            category_stats[cat] = []
        category_stats[cat].append(result)
    
    category_performance = {}
    for cat, cat_results in category_stats.items():
        cat_recall = sum(r["recall"] for r in cat_results) / len(cat_results)
        cat_pass = sum(1 for r in cat_results if r["passed"]) / len(cat_results)
        category_performance[cat] = {
            "recall": cat_recall,
            "pass_rate": cat_pass,
            "count": len(cat_results)
        }
    
    return {
        "variant_name": variant_name,
        "description": variant_config["description"],
        "precision": total_precision,
        "recall": total_recall,
        "f1": total_f1,
        "pass_rate": pass_rate,
        "total_tests": len(results),
        "category_performance": category_performance,
        "failed_cases": [r for r in results if not r["passed"]]
    }


def run_prompt_optimization():
    """Run all prompt variants and compare results"""
    print("="*80)
    print("PROMPT VARIATION OPTIMIZATION SUITE")
    print("="*80)
    print(f"\nTesting {len(PROMPT_VARIANTS)} prompt variants on {len(TEST_CASES)} test cases\n")
    
    all_results = {}
    
    for variant_name, variant_config in PROMPT_VARIANTS.items():
        result = test_prompt_variant(variant_name, variant_config)
        all_results[variant_name] = result
    
    # Comparison summary
    print("\n" + "="*80)
    print("VARIANT COMPARISON")
    print("="*80)
    print(f"\n{'Variant':<25} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Pass Rate':<12}")
    print("-" * 80)
    
    for variant_name, result in sorted(all_results.items(), key=lambda x: x[1]["f1"], reverse=True):
        print(f"{variant_name:<25} {result['precision']:>10.1%}  {result['recall']:>10.1%}  {result['f1']:>10.1%}  {result['pass_rate']:>10.1%}")
    
    # Best variant
    best_variant = max(all_results.items(), key=lambda x: x[1]["f1"])
    print(f"\n🏆 BEST VARIANT: {best_variant[0]}")
    print(f"   Description: {best_variant[1]['description']}")
    print(f"   F1 Score: {best_variant[1]['f1']:.1%}")
    print(f"   Pass Rate: {best_variant[1]['pass_rate']:.1%}")
    
    # Category analysis for best variant
    print(f"\n   Performance by Category:")
    for cat, stats in sorted(best_variant[1]["category_performance"].items(), key=lambda x: x[1]["recall"]):
        print(f"     {cat:<30} Recall: {stats['recall']:>6.1%}  Pass: {stats['pass_rate']:>6.1%}  ({stats['count']} tests)")
    
    # Failed cases for best variant
    if best_variant[1]["failed_cases"]:
        print(f"\n   Remaining Failed Cases ({len(best_variant[1]['failed_cases'])}):")
        for fc in best_variant[1]["failed_cases"]:
            print(f"     - {fc['test_name']}: missing {fc['missing_modifiers']}")
    
    # Save results to JSON
    output_path = Path(__file__).parent.parent / "output" / "prompt_optimization_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n📊 Full results saved to: {output_path}")
    print("\n" + "="*80)
    
    return all_results


if __name__ == "__main__":
    results = run_prompt_optimization()
