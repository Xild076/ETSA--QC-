#!/usr/bin/env python3
"""
Comprehensive Analysis of Wrong_Polarity Errors
Goal: Identify root causes and reach 85% accuracy from current 75%
Current State: 11 wrong_polarity errors out of 44 aspects
"""

import sys
sys.path.insert(0, 'src')

import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any

print("="*80)
print("COMPREHENSIVE WRONG_POLARITY ANALYSIS")
print("="*80)

benchmark_dir = Path("output/benchmarks/improved_system_test_test_laptop_2014_20250930_235334")
metrics_file = benchmark_dir / "metrics.json"
wrong_polarity_dir = benchmark_dir / "graph_reports/wrong_polarity"

with open(metrics_file, 'r') as f:
    metrics = json.load(f)

print(f"\n### CURRENT PERFORMANCE ###")
print(f"Accuracy: {metrics['accuracy']:.1%}")
print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.1%}")
print(f"Precision (macro): {metrics['classification_report']['macro avg']['precision']:.1%}")
print(f"Recall (macro): {metrics['classification_report']['macro avg']['recall']:.1%}")
print(f"F1 (macro): {metrics['classification_report']['macro avg']['f1-score']:.1%}")
print(f"\nError Breakdown:")
print(f"  Wrong Polarity: {metrics['error_summary']['wrong_polarity']}")
print(f"  Missing Aspect: {metrics['error_summary']['missing_aspect']}")
print(f"  Spurious Aspect: {metrics['error_summary']['spurious_aspect']}")

total_aspects = sum(metrics['classification_report'][k]['support'] for k in ['positive', 'negative', 'neutral'])
current_correct = int(metrics['accuracy'] * total_aspects)
target_85 = int(0.85 * total_aspects)
need_to_fix = target_85 - current_correct

print(f"\n### GAP TO 85% ACCURACY ###")
print(f"Total aspects: {total_aspects}")
print(f"Current correct: {current_correct}")
print(f"Need for 85%: {target_85}")
print(f"Must improve: {need_to_fix} predictions")
print(f"Wrong_polarity count: {metrics['error_summary']['wrong_polarity']}")
print(f"→ If we fix ALL wrong_polarity: {(current_correct + metrics['error_summary']['wrong_polarity'])/total_aspects:.1%} accuracy")

print(f"\n### ANALYZING WRONG_POLARITY CASES ###")

wrong_polarity_files = list(wrong_polarity_dir.glob("*.json"))
print(f"Found {len(wrong_polarity_files)} wrong_polarity error files")

analysis = {
    "predicted_vs_expected": defaultdict(int),
    "combiner_types": Counter(),
    "sentiment_scores": [],
    "reliability_scores": [],
    "text_characteristics": {
        "has_negation": 0,
        "has_modifier": 0,
        "multiple_clauses": 0,
        "entity_roles": Counter(),
    },
    "cases": []
}

for wp_file in sorted(wrong_polarity_files):
    with open(wp_file, 'r') as f:
        data = json.load(f)
    
    text = data.get('text', '')
    clauses = data.get('clauses', [])
    nodes = data.get('nodes', [])
    agg_sentiments = data.get('aggregate_sentiments', {})
    
    case_info = {
        "file": wp_file.name,
        "text": text,
        "num_clauses": len(clauses),
        "num_nodes": len(nodes),
        "aggregate_sentiments": agg_sentiments
    }
    
    analysis["text_characteristics"]["multiple_clauses"] += (len(clauses) > 1)
    
    if any(word in text.lower() for word in ['not', 'no', 'never', 'nothing', "n't", "dont", "doesn't", "didn't"]):
        analysis["text_characteristics"]["has_negation"] += 1
        case_info["has_negation"] = True
    
    for node in nodes:
        combiner_type = node.get('combiner_debug', {}).get('combiner_type', 'Unknown')
        analysis["combiner_types"][combiner_type] += 1
        
        reliability = node.get('combiner_debug', {}).get('reliability', 0)
        analysis["reliability_scores"].append(reliability)
        
        init_sentiment = node.get('init_sentiment', 0)
        analysis["sentiment_scores"].append(init_sentiment)
        
        entity_role = node.get('entity_role', 'unknown')
        analysis["text_characteristics"]["entity_roles"][entity_role] += 1
        
        modifiers = node.get('modifier', [])
        if modifiers:
            analysis["text_characteristics"]["has_modifier"] += 1
            case_info.setdefault("node_modifiers", []).append(modifiers)
    
    analysis["cases"].append(case_info)

print(f"\n### WRONG_POLARITY PATTERNS ###")
print(f"Combiner Type Distribution:")
for combiner, count in analysis["combiner_types"].most_common():
    print(f"  {combiner}: {count}")

print(f"\nText Characteristics:")
print(f"  Contains negation: {analysis['text_characteristics']['has_negation']}/{len(wrong_polarity_files)}")
print(f"  Has modifiers: {analysis['text_characteristics']['has_modifier']}/{len(wrong_polarity_files)}")
print(f"  Multiple clauses: {analysis['text_characteristics']['multiple_clauses']}/{len(wrong_polarity_files)}")

print(f"\nEntity Role Distribution:")
for role, count in analysis['text_characteristics']['entity_roles'].most_common():
    print(f"  {role}: {count}")

if analysis["sentiment_scores"]:
    avg_sentiment = sum(analysis["sentiment_scores"]) / len(analysis["sentiment_scores"])
    avg_reliability = sum(analysis["reliability_scores"]) / len(analysis["reliability_scores"])
    print(f"\nAverage Sentiment Score: {avg_sentiment:+.3f}")
    print(f"Average Reliability: {avg_reliability:.3f}")

print(f"\n### SPECIFIC WRONG_POLARITY CASES ###")
for case in analysis["cases"][:5]:
    print(f"\nFile: {case['file']}")
    print(f"Text: {case['text'][:100]}...")
    print(f"Clauses: {case['num_clauses']}, Nodes: {case['num_nodes']}")
    if case.get('has_negation'):
        print(f"  ⚠ Contains NEGATION")
    print(f"Aggregate Sentiments: {case['aggregate_sentiments']}")

print(f"\n### HYPOTHESIS: ROOT CAUSES ###")
print("Based on the analysis, potential root causes for wrong_polarity errors:")
print()
print("1. COMBINER WEIGHTING ISSUES")
print("   - Combiner may be over/under-weighting modifiers vs heads")
print("   - Reliability scores may not accurately reflect sentiment quality")
print()
print("2. NEGATION HANDLING")
print(f"   - {analysis['text_characteristics']['has_negation']} of {len(wrong_polarity_files)} cases contain negation")
print("   - Base sentiment methods (TextBlob) fail on negation (from earlier diagnostic)")
print("   - Combiner may not be properly amplifying negation signals")
print()
print("3. NEUTRAL MISCLASSIFICATION")
confusion = metrics['confusion_matrix']
labels = confusion['labels']
matrix = confusion['matrix']
neutral_idx = labels.index('neutral')
neutral_row = matrix[neutral_idx]
print(f"   - Neutral class recall: {metrics['classification_report']['neutral']['recall']:.1%}")
print(f"   - Neutral confused as: {dict(zip(labels, neutral_row))}")
print("   - Threshold boundaries (±0.1) may be causing near-neutral scores to misclassify")
print()
print("4. MODIFIER EXTRACTION QUALITY")
print("   - Using DummyModifierExtractor (no real modifiers)")
print("   - Real modifiers would provide crucial sentiment context")
print()
print("5. AGGREGATE MODEL")
print("   - May not be properly balancing multiple entity sentiments")
print("   - Could be averaging away strong signals")

print(f"\n### COMPONENT-BY-COMPONENT RECOMMENDATIONS ###")
print()
print("IMMEDIATE FIXES (High Impact):")
print("1. **Fix Combiner Thresholds**")
print("   - Current: pos_thresh=0.1, neg_thresh=-0.1")
print("   - Consider: pos_thresh=0.05, neg_thresh=-0.05 (tighter neutral band)")
print("   - OR: Dynamic thresholds based on reliability")
print()
print("2. **Improve Negation Handling**")
print("   - Add negation detection in combiner logic")
print("   - Boost negative sentiment weight when negation detected")
print("   - Consider TextBlob alternatives for negation-heavy text")
print()
print("3. **Tune Combiner Weights**")
print("   - Current adaptive_v6: mod 0.7, head 0.2")
print("   - Test: Higher head weight for short phrases")
print("   - Test: Context-aware weight adjustment")
print()
print("MEDIUM PRIORITY:")
print("4. **Enable Real Modifier Extraction**")
print("   - Switch from DummyModifierExtractor to GemmaModifierExtractor")
print("   - Ensure .env loads with GOOGLE_API_KEY")
print()
print("5. **Optimize Aggregate Model**")
print("   - Consider weighted average based on entity relevance")
print("   - Apply confidence-based filtering")
print()
print("LOW PRIORITY:")
print("6. **Fine-tune Sentiment Methods**")
print("   - Base methods (DistilBERT, VADER) perform well (100% in simple test)")
print("   - Issue is in the PIPELINE, not base methods")

print(f"\n### ACTION PLAN TO REACH 85% ###")
print()
print("Phase 1: Fix Critical Issues (Expected gain: +5-7%)")
print("  → Adjust classification thresholds")
print("  → Improve combiner negation handling")
print("  → Fine-tune combiner weights via optimization")
print()
print("Phase 2: Enable Better Features (Expected gain: +2-3%)")
print("  → Enable GemmaModifierExtractor with proper API key loading")
print("  → Optimize aggregate model parameters")
print()
print("Phase 3: Validation")
print("  → Run full benchmark with changes")
print("  → Verify 85%+ accuracy achieved")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}\n")
