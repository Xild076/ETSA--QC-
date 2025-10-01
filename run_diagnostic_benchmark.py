#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

from pipeline.benchmark import run_benchmark

print("Running benchmark with improved sentiment analysis (adaptive weighting + stronger negation handling)...")
results = run_benchmark(
    dataset_name='test_laptop_2014',
    run_name='accuracy_diagnostic_stronger_negation',
    limit=50,
    run_mode='full_stack',
    combiner='adaptive_v6',
    combiner_params={
        'negation_boost': 1.5,
        'modifier_quality_weight': 0.25
    }
)

print(f'\n\n===== FINAL METRICS =====')
print(f'Accuracy: {results["accuracy"]:.2%}')
print(f'Balanced Accuracy: {results["balanced_accuracy"]:.2%}')
print(f'Precision (macro): {results["classification_report"]["macro avg"]["precision"]:.2%}')
print(f'Recall (macro): {results["classification_report"]["macro avg"]["recall"]:.2%}')
print(f'F1 (macro): {results["classification_report"]["macro avg"]["f1-score"]:.2%}')
print(f'\nError Summary:')
print(f'  Wrong Polarity: {results["error_summary"]["wrong_polarity"]}')
print(f'  Missing Aspect: {results["error_summary"]["missing_aspect"]}')
print(f'  Spurious Aspect: {results["error_summary"]["spurious_aspect"]}')
print(f'\nTotal Aspects Evaluated: {sum(results["classification_report"][k]["support"] for k in ["positive", "negative", "neutral"] if k in results["classification_report"])}')
print(f'\n===== TO REACH 85% ACCURACY =====')
current_correct = int(results["accuracy"] * sum(results["classification_report"][k]["support"] for k in ["positive", "negative", "neutral"] if k in results["classification_report"]))
total = sum(results["classification_report"][k]["support"] for k in ["positive", "negative", "neutral"] if k in results["classification_report"])
target_correct = int(0.85 * total)
need_to_fix = target_correct - current_correct
print(f'Current correct: {current_correct}/{total}')
print(f'Need for 85%: {target_correct}/{total}')
print(f'Must fix: {need_to_fix} wrong predictions')
print(f'Current wrong_polarity errors: {results["error_summary"]["wrong_polarity"]}')
print(f'If we fix all wrong_polarity, accuracy would be: {(current_correct + results["error_summary"]["wrong_polarity"])/total:.2%}')
