#!/usr/bin/env python3
"""
Detailed analysis of wrong_polarity errors comparing Laptop vs Restaurant datasets
"""
import json
from pathlib import Path
from collections import Counter, defaultdict
import re

def analyze_wrong_polarity(benchmark_dir, dataset_name):
    print(f"\n{'='*80}")
    print(f"WRONG_POLARITY ANALYSIS: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    error_details_path = benchmark_dir / "error_details.json"
    with open(error_details_path, 'r') as f:
        errors = json.load(f)
    
    wrong_polarity = [e for e in errors if e['issue_type'] == 'wrong_polarity']
    
    print(f"\nTotal wrong_polarity errors: {len(wrong_polarity)}")
    
    # Pattern analysis: gold → predicted
    patterns = Counter()
    for e in wrong_polarity:
        gold = e['gold']['polarity']
        pred = e['predicted']['polarity']
        patterns[f"{gold} → {pred}"] += 1
    
    print("\n### Misclassification Patterns ###")
    for pattern, count in patterns.most_common():
        print(f"  {pattern:20s}: {count:4d} ({count/len(wrong_polarity)*100:5.1f}%)")
    
    # Score distribution analysis
    neutral_misclass = [e for e in wrong_polarity if e['gold']['polarity'] == 'neutral']
    positive_misclass = [e for e in wrong_polarity if e['gold']['polarity'] == 'positive']
    negative_misclass = [e for e in wrong_polarity if e['gold']['polarity'] == 'negative']
    
    print(f"\n### Score Distribution ###")
    print(f"Neutral misclassified: {len(neutral_misclass)} / {len(wrong_polarity)} ({len(neutral_misclass)/len(wrong_polarity)*100:.1f}%)")
    if neutral_misclass:
        scores = [e['predicted']['score'] for e in neutral_misclass]
        print(f"  Score range: [{min(scores):.3f}, {max(scores):.3f}]")
        print(f"  Mean: {sum(scores)/len(scores):.3f}")
        
        # Count by predicted polarity
        pred_counts = Counter(e['predicted']['polarity'] for e in neutral_misclass)
        for pol, cnt in pred_counts.most_common():
            print(f"  Predicted as {pol}: {cnt} ({cnt/len(neutral_misclass)*100:.1f}%)")
    
    print(f"\nPositive misclassified: {len(positive_misclass)} / {len(wrong_polarity)} ({len(positive_misclass)/len(wrong_polarity)*100:.1f}%)")
    if positive_misclass:
        scores = [e['predicted']['score'] for e in positive_misclass]
        print(f"  Score range: [{min(scores):.3f}, {max(scores):.3f}]")
        print(f"  Mean: {sum(scores)/len(scores):.3f}")
    
    print(f"\nNegative misclassified: {len(negative_misclass)} / {len(wrong_polarity)} ({len(negative_misclass)/len(wrong_polarity)*100:.1f}%)")
    if negative_misclass:
        scores = [e['predicted']['score'] for e in negative_misclass]
        print(f"  Score range: [{min(scores):.3f}, {max(scores):.3f}]")
        print(f"  Mean: {sum(scores)/len(scores):.3f}")
    
    # Text characteristics
    print("\n### Text Characteristics ###")
    negation_words = ['not', 'no', 'never', "n't", 'nothing', 'neither', 'nor', 'nobody', 'none', 'nowhere', "don't", "didn't", "doesn't", "won't", "wasn't", "weren't", "haven't", "hasn't", "hadn't"]
    
    has_negation = 0
    for e in wrong_polarity:
        text = e['text'].lower()
        if any(word in text for word in negation_words):
            has_negation += 1
    
    print(f"Contains negation: {has_negation} / {len(wrong_polarity)} ({has_negation/len(wrong_polarity)*100:.1f}%)")
    
    # Sample worst errors
    print("\n### Sample Worst Errors (largest score deviation) ###")
    threshold_errors = []
    for e in wrong_polarity:
        gold = e['gold']['polarity']
        score = e['predicted']['score']
        
        # Calculate how far from correct classification
        if gold == 'positive':
            deviation = 0.1 - score  # Should be > 0.1
        elif gold == 'negative':
            deviation = score - (-0.1)  # Should be < -0.1
        elif gold == 'neutral':
            if score > 0.1:
                deviation = score - 0.1
            elif score < -0.1:
                deviation = -0.1 - score
            else:
                deviation = 0
        
        threshold_errors.append((deviation, e))
    
    threshold_errors.sort(reverse=True, key=lambda x: x[0])
    
    for i, (dev, e) in enumerate(threshold_errors[:5]):
        print(f"\n{i+1}. Gold: {e['gold']['polarity']}, Pred: {e['predicted']['polarity']} (score: {e['predicted']['score']:+.3f}, deviation: {dev:.3f})")
        text = e['text']
        if len(text) > 100:
            text = text[:97] + "..."
        print(f"   Text: {text}")
        print(f"   Aspect: {e['predicted']['canonical']}")

# Analyze both datasets
laptop_dir = Path("output/benchmarks/newest_test_test_laptop_2014_test_laptop_2014_20251001_124408")
restaurant_dir = Path("output/benchmarks/newest_test_test_restaurant_2014_test_restaurant_2014_20251001_124459")

analyze_wrong_polarity(laptop_dir, "Laptop")
analyze_wrong_polarity(restaurant_dir, "Restaurant")

# Comparison
print(f"\n{'='*80}")
print("COMPARISON SUMMARY")
print(f"{'='*80}")

with open(laptop_dir / "metrics.json") as f:
    laptop_metrics = json.load(f)
with open(restaurant_dir / "metrics.json") as f:
    restaurant_metrics = json.load(f)

print(f"\nAccuracy:")
print(f"  Laptop:     {laptop_metrics['accuracy']:.1%} ({laptop_metrics['error_summary']['wrong_polarity']} wrong_polarity)")
print(f"  Restaurant: {restaurant_metrics['accuracy']:.1%} ({restaurant_metrics['error_summary']['wrong_polarity']} wrong_polarity)")

print(f"\nNeutral Recall (key difference):")
print(f"  Laptop:     {laptop_metrics['classification_report']['neutral']['recall']:.1%}")
print(f"  Restaurant: {restaurant_metrics['classification_report']['neutral']['recall']:.1%}")

print(f"\nPositive Recall:")
print(f"  Laptop:     {laptop_metrics['classification_report']['positive']['recall']:.1%}")
print(f"  Restaurant: {restaurant_metrics['classification_report']['positive']['recall']:.1%}")

print(f"\nNegative Recall:")
print(f"  Laptop:     {laptop_metrics['classification_report']['negative']['recall']:.1%}")
print(f"  Restaurant: {restaurant_metrics['classification_report']['negative']['recall']:.1%}")
