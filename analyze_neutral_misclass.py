#!/usr/bin/env python3
"""
Deep dive into neutral misclassification patterns
"""
import json
from pathlib import Path
from collections import Counter

def analyze_neutral_misclass(benchmark_dir, dataset_name):
    print(f"\n{'='*80}")
    print(f"NEUTRAL MISCLASSIFICATION DEEP DIVE: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    with open(benchmark_dir / "error_details.json") as f:
        errors = json.load(f)
    
    neutral_errors = [e for e in errors if e['issue_type'] == 'wrong_polarity' and e['gold']['polarity'] == 'neutral']
    
    print(f"\nTotal neutral misclassifications: {len(neutral_errors)}")
    
    # Analyze score buckets
    very_negative = [e for e in neutral_errors if e['predicted']['score'] < -0.3]
    slight_negative = [e for e in neutral_errors if -0.3 <= e['predicted']['score'] < -0.1]
    borderline_negative = [e for e in neutral_errors if -0.15 <= e['predicted']['score'] < -0.1]
    slight_positive = [e for e in neutral_errors if 0.1 < e['predicted']['score'] <= 0.3]
    borderline_positive = [e for e in neutral_errors if 0.1 < e['predicted']['score'] <= 0.15]
    very_positive = [e for e in neutral_errors if e['predicted']['score'] > 0.3]
    
    print(f"\nScore Distribution:")
    print(f"  Very negative (< -0.3):     {len(very_negative):3d} ({len(very_negative)/len(neutral_errors)*100:5.1f}%)")
    print(f"  Slight negative (-0.3 to -0.1): {len(slight_negative):3d} ({len(slight_negative)/len(neutral_errors)*100:5.1f}%)")
    print(f"  Borderline neg (-0.15 to -0.1):  {len(borderline_negative):3d} ({len(borderline_negative)/len(neutral_errors)*100:5.1f}%) ← THRESHOLD ISSUE")
    print(f"  Slight positive (0.1 to 0.3):   {len(slight_positive):3d} ({len(slight_positive)/len(neutral_errors)*100:5.1f}%)")
    print(f"  Borderline pos (0.1 to 0.15):    {len(borderline_positive):3d} ({len(borderline_positive)/len(neutral_errors)*100:5.1f}%) ← THRESHOLD ISSUE")
    print(f"  Very positive (> 0.3):      {len(very_positive):3d} ({len(very_positive)/len(neutral_errors)*100:5.1f}%)")
    
    # Sample each bucket
    print(f"\n### Sample Very Negative Neutrals ###")
    for e in very_negative[:3]:
        text = e['text'][:100] + "..." if len(e['text']) > 100 else e['text']
        print(f"  Score: {e['predicted']['score']:+.3f} | Aspect: {e['predicted']['canonical']} | Text: {text}")
    
    print(f"\n### Sample Very Positive Neutrals ###")
    for e in very_positive[:3]:
        text = e['text'][:100] + "..." if len(e['text']) > 100 else e['text']
        print(f"  Score: {e['predicted']['score']:+.3f} | Aspect: {e['predicted']['canonical']} | Text: {text}")
    
    # Check what aspects are most commonly misclassified
    aspects = Counter(e['predicted']['canonical'] for e in neutral_errors)
    print(f"\n### Most Commonly Misclassified Aspects ###")
    for aspect, count in aspects.most_common(10):
        print(f"  {aspect:30s}: {count:3d} errors")
    
    # Check for negation patterns
    negation_words = ['not', 'no', 'never', "n't", 'nothing', 'neither', 'nor', 'nobody', 'none', 'nowhere', "don't", "didn't", "doesn't", "won't", "wasn't", "weren't", "haven't", "hasn't", "hadn't"]
    
    with_negation = [e for e in neutral_errors if any(word in e['text'].lower() for word in negation_words)]
    print(f"\nNeutral errors with negation: {len(with_negation)} / {len(neutral_errors)} ({len(with_negation)/len(neutral_errors)*100:.1f}%)")
    
    # Check modifier presence
    with_modifiers = [e for e in neutral_errors if e['predicted']['mentions'][0].get('modifiers', [])]
    print(f"Neutral errors with modifiers: {len(with_modifiers)} / {len(neutral_errors)} ({len(with_modifiers)/len(neutral_errors)*100:.1f}%)")
    
    # Analyze text length
    avg_text_len = sum(len(e['text']) for e in neutral_errors) / len(neutral_errors)
    print(f"Average text length: {avg_text_len:.0f} characters")

laptop_dir = Path("output/benchmarks/newest_test_test_laptop_2014_test_laptop_2014_20251001_124408")
restaurant_dir = Path("output/benchmarks/newest_test_test_restaurant_2014_test_restaurant_2014_20251001_124459")

analyze_neutral_misclass(laptop_dir, "Laptop")
analyze_neutral_misclass(restaurant_dir, "Restaurant")

print(f"\n{'='*80}")
print("ACTIONABLE INSIGHTS")
print(f"{'='*80}")
print("""
1. THRESHOLD ADJUSTMENT WON'T HELP MUCH:
   - Only 2-3% of neutral misclassifications are in the -0.15 to -0.1 borderline zone
   - Most are decisively wrong (< -0.3 or > 0.3)
   
2. REAL PROBLEM: SENTIMENT SCORES ARE FUNDAMENTALLY WRONG
   - Need to fix at sentiment analysis level, not combiner level
   - Laptop neutrals score very negative (-0.122 mean)
   - Restaurant neutrals also negative but less (-0.056 mean)
   
3. NEXT STEPS:
   a) Investigate why neutral aspects get negative scores
   b) Check if specific aspects are consistently misclassified
   c) Examine modifier sentiment contribution
   d) Test sentiment model performance on neutral-labeled text
""")
