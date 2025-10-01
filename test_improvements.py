#!/usr/bin/env python3
"""Test script to verify improvements are working"""
import sys
sys.path.insert(0, 'src')

from pipeline.sentiment_analysis import MultiSentimentAnalysis
from pipeline.combiners import AdaptiveV6Combiner

print("=" * 80)
print("TESTING SENTIMENT ANALYSIS IMPROVEMENTS")
print("=" * 80)

# Test 1: Adaptive weighting with negation
print("\n### Test 1: Adaptive Weighting with Negation ###")
sa = MultiSentimentAnalysis(
    methods=['distilbert_logit', 'vader'],
    weights=[0.6, 0.4],
    use_adaptive_weighting=True
)

text_with_negation = "not good at all"
result = sa.analyze(text_with_negation)
print(f"Text: '{text_with_negation}'")
print(f"Sentiment: {result['aggregate']:.3f}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Has negation detected: {result.get('has_negation', False)}")
if 'adaptive_weights_used' in result:
    print(f"Adaptive weights: {result['adaptive_weights_used']}")

# Test 2: Combiner with negation boost
print("\n### Test 2: Combiner Negation Boost ###")
combiner = AdaptiveV6Combiner(
    negation_boost=1.25,
    modifier_quality_weight=0.15
)

# Test case with negation
head_text = "screen"
head_sentiment = 0.3  # Slightly positive
modifiers = ["not bright"]
modifier_sentiment = -0.4  # Negative
clause_text = "the screen is not bright"

score, justification, extras = combiner.combine(
    head_text=head_text,
    head_sentiment=head_sentiment,
    modifiers=modifiers,
    modifier_sentiment=modifier_sentiment,
    threshold=(0.1, -0.1),
    context_score=None,
    per_scores={},
    clause_text=clause_text
)

print(f"Clause: '{clause_text}'")
print(f"Head: '{head_text}' ({head_sentiment:+.2f})")
print(f"Modifiers: {modifiers} ({modifier_sentiment:+.2f})")
print(f"Combined score: {score:+.3f}")
print(f"Justification: {justification}")
print(f"Negation in heuristics: {'negation_detected' in extras.get('heuristic_notes', [])}")

print("\n" + "=" * 80)
print("TESTS COMPLETE")
print("=" * 80)
