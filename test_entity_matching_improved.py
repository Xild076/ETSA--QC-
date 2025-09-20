#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pipeline.benchmark import find_best_from_names

def test_entity_matching():
    print("=== Testing Entity Matching with Category Mapping ===")
    
    # Test cases from our benchmark errors
    test_cases = [
        {
            "predicted": ["The bread"],
            "gold_terms": ["bread", "food"],
            "description": "Bread example from benchmark"
        },
        {
            "predicted": ["the coffee"],
            "gold_terms": ["coffee", "food"],
            "description": "Coffee example from benchmark"
        },
        {
            "predicted": ["the fastest delivery times"],
            "gold_terms": ["service"],
            "description": "Service example from benchmark"
        },
        {
            "predicted": ["Our waiter"],
            "gold_terms": ["service"],
            "description": "Waiter service example"
        },
        {
            "predicted": ["the place"],
            "gold_terms": ["ambience", "anecdotes/miscellaneous"],
            "description": "Place/ambience example"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['description']}")
        print(f"   Predicted entities: {test_case['predicted']}")
        
        for term in test_case['gold_terms']:
            match = find_best_from_names(test_case['predicted'], term)
            print(f"   Gold term '{term}' -> Match: '{match}' {'✓' if match else '✗'}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    test_entity_matching()
