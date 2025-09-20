#!/usr/bin/env python3

import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pipeline.benchmark import find_best_from_names

# Example from our trace
pred_entities = ["The bread"]
gold_terms = ["bread", "food"]

print("Testing entity matching:")
print(f"Predicted entities: {pred_entities}")
print(f"Gold terms: {gold_terms}")
print()

for term in gold_terms:
    match = find_best_from_names(pred_entities, term)
    print(f"Gold term '{term}' -> Best match: '{match}'")

print("\n" + "="*50)

# Let's test the reverse as well
pred_entities2 = ["bread", "coffee", "service", "the fastest delivery times", "the city"]
for term in ["food", "service", "delivery", "ambience"]:
    match = find_best_from_names(pred_entities2, term)
    print(f"Gold term '{term}' -> Best match: '{match}'")
