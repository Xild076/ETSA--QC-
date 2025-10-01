#!/usr/bin/env python3
import sys
import os
import json
from typing import Dict, List, Any

sys.path.insert(0, '/Users/harry/Documents/Python_Projects/ETSA_(QC)/src')

from sentiment.sentiment import get_distilbert_logit_sentiment, get_vader_sentiment, get_textblob_sentiment

TEST_CASES = [
    {
        "text": "the weight is acceptable",
        "expected_polarity": "positive",
        "expected_score_range": (0.1, 1.0),
        "issue": "acceptable_word_negative",
        "priority": "critical"
    },
    {
        "text": "Did not enjoy the new Windows 8 and touchscreen functions",
        "expected_polarity": "negative",
        "expected_score_range": (-1.0, -0.1),
        "issue": "negation_not_handled",
        "priority": "critical"
    },
    {
        "text": "No installation disk (DVD) is included",
        "expected_polarity": "negative",
        "expected_score_range": (-1.0, -0.1),
        "issue": "contextual_negation",
        "priority": "critical"
    },
    {
        "text": "Its size is ideal and the weight is acceptable",
        "expected_polarity": "positive",
        "expected_score_range": (0.1, 1.0),
        "issue": "mixed_sentiment_compound",
        "priority": "high"
    },
    {
        "text": "This product is not bad",
        "expected_polarity": "positive",
        "expected_score_range": (0.0, 1.0),
        "issue": "double_negation",
        "priority": "medium"
    },
    {
        "text": "I hate this laptop",
        "expected_polarity": "negative",
        "expected_score_range": (-1.0, -0.1),
        "issue": "strong_negative",
        "priority": "high"
    },
    {
        "text": "This is an excellent device",
        "expected_polarity": "positive",
        "expected_score_range": (0.1, 1.0),
        "issue": "strong_positive",
        "priority": "high"
    }
]

def extract_score(valence) -> float:
    if isinstance(valence, (int, float)):
        return float(valence)
    if isinstance(valence, dict):
        for key in ('score', 'compound', 'aggregate', 'polarity', 'value'):
            value = valence.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        for nested in valence.values():
            if isinstance(nested, (dict, int, float)):
                nested_score = extract_score(nested)
                if nested_score is not None:
                    return nested_score
    return 0.0

def check_polarity(score: float, expected: str, score_range: tuple) -> Dict[str, Any]:
    min_score, max_score = score_range
    in_range = min_score <= score <= max_score
    
    if expected == "positive":
        correct = score > 0.1
    elif expected == "negative":
        correct = score < -0.1
    else:
        correct = abs(score) <= 0.1
    
    return {
        "correct": correct,
        "in_range": in_range,
        "score": score
    }

def test_base_sentiment_methods():
    print("="*80)
    print("TESTING BASE SENTIMENT METHODS")
    print("="*80)
    
    results = []
    
    for case in TEST_CASES:
        text = case["text"]
        expected = case["expected_polarity"]
        score_range = case["expected_score_range"]
        issue = case["issue"]
        priority = case["priority"]
        
        print(f"\n[{priority.upper()}] {issue}")
        print(f"Text: '{text}'")
        print(f"Expected: {expected} (range: {score_range})")
        
        distilbert_raw = get_distilbert_logit_sentiment(text)
        vader_raw = get_vader_sentiment(text)
        textblob_raw = get_textblob_sentiment(text)
        
        distilbert_score = extract_score(distilbert_raw)
        vader_score = extract_score(vader_raw)
        textblob_score = extract_score(textblob_raw)
        
        distilbert_check = check_polarity(distilbert_score, expected, score_range)
        vader_check = check_polarity(vader_score, expected, score_range)
        textblob_check = check_polarity(textblob_score, expected, score_range)
        
        print(f"  DistilBERT: {distilbert_score:+.3f} {'✓' if distilbert_check['correct'] else '✗'}")
        print(f"  VADER:      {vader_score:+.3f} {'✓' if vader_check['correct'] else '✗'}")
        print(f"  TextBlob:   {textblob_score:+.3f} {'✓' if textblob_check['correct'] else '✗'}")
        
        results.append({
            "text": text,
            "issue": issue,
            "priority": priority,
            "expected": expected,
            "methods": {
                "distilbert": distilbert_check,
                "vader": vader_check,
                "textblob": textblob_check
            }
        })
    
    return results

def generate_summary(base_results):
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY - BASE SENTIMENT METHODS")
    print("="*80)
    
    print("\n--- RESULTS BY PRIORITY ---")
    for priority in ["critical", "high", "medium"]:
        priority_cases = [r for r in base_results if r["priority"] == priority]
        if not priority_cases:
            continue
        
        print(f"\n{priority.upper()} issues ({len(priority_cases)}):")
        for r in priority_cases:
            methods = r["methods"]
            d_pass = "✓" if methods["distilbert"]["correct"] else "✗"
            v_pass = "✓" if methods["vader"]["correct"] else "✗"
            t_pass = "✓" if methods["textblob"]["correct"] else "✗"
            
            d_score = methods["distilbert"]["score"]
            v_score = methods["vader"]["score"]
            t_score = methods["textblob"]["score"]
            
            print(f"  {r['issue']}:")
            print(f"    Expected: {r['expected']}")
            print(f"    DistilBERT: {d_pass} ({d_score:+.3f})")
            print(f"    VADER:      {v_pass} ({v_score:+.3f})")
            print(f"    TextBlob:   {t_pass} ({t_score:+.3f})")
    
    print("\n--- FAILURE ANALYSIS ---")
    
    critical_failures_by_method = {"distilbert": [], "vader": [], "textblob": []}
    all_methods_fail = []
    
    for r in base_results:
        if r["priority"] != "critical":
            continue
        
        methods = r["methods"]
        
        if not methods["distilbert"]["correct"]:
            critical_failures_by_method["distilbert"].append(r["issue"])
        if not methods["vader"]["correct"]:
            critical_failures_by_method["vader"].append(r["issue"])
        if not methods["textblob"]["correct"]:
            critical_failures_by_method["textblob"].append(r["issue"])
        
        if not any(m["correct"] for m in methods.values()):
            all_methods_fail.append(r["issue"])
    
    print(f"\nCritical issues where ALL methods fail ({len(all_methods_fail)}):")
    for issue in all_methods_fail:
        print(f"  ✗ {issue}")
    
    print(f"\nCritical failures by method:")
    for method, failures in critical_failures_by_method.items():
        print(f"  {method}: {len(failures)}/{len([r for r in base_results if r['priority'] == 'critical'])}")
        for issue in failures:
            print(f"    - {issue}")
    
    print("\n--- ACCURACY BY METHOD ---")
    total_cases = len(base_results)
    for method in ["distilbert", "vader", "textblob"]:
        correct_count = sum(1 for r in base_results if r["methods"][method]["correct"])
        accuracy = correct_count / total_cases * 100
        print(f"  {method.capitalize()}: {correct_count}/{total_cases} ({accuracy:.1f}%)")
    
    return {
        "all_methods_fail": all_methods_fail,
        "critical_failures_by_method": critical_failures_by_method
    }

def main():
    print("Starting Simplified Polarity Diagnostic")
    print("="*80 + "\n")
    
    base_results = test_base_sentiment_methods()
    failure_summary = generate_summary(base_results)
    
    output_file = '/Users/harry/Documents/Python_Projects/ETSA_(QC)/polarity_diagnostic_simple_results.json'
    
    output_data = {
        "base_sentiment_results": base_results,
        "failure_summary": failure_summary
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n\nFull results saved to: {output_file}")
    
    total_critical_failures = len(failure_summary["all_methods_fail"])
    
    if total_critical_failures == 0:
        print("\n✓ No critical failures where all methods fail!")
        return 0
    else:
        print(f"\n✗ {total_critical_failures} critical failure(s) where ALL methods fail")
        return 1

if __name__ == "__main__":
    sys.exit(main())
