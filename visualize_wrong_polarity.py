#!/usr/bin/env python3
"""
Visualize score distributions for wrong_polarity errors
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_wrong_polarity(benchmark_dir):
    error_details_path = benchmark_dir / "error_details.json"
    with open(error_details_path, 'r') as f:
        errors = json.load(f)
    return [e for e in errors if e['issue_type'] == 'wrong_polarity']

laptop_dir = Path("output/benchmarks/newest_test_test_laptop_2014_test_laptop_2014_20251001_124408")
restaurant_dir = Path("output/benchmarks/newest_test_test_restaurant_2014_test_restaurant_2014_20251001_124459")

laptop_errors = load_wrong_polarity(laptop_dir)
restaurant_errors = load_wrong_polarity(restaurant_dir)

# Separate by gold polarity
def separate_by_gold(errors):
    neutral = [e['predicted']['score'] for e in errors if e['gold']['polarity'] == 'neutral']
    positive = [e['predicted']['score'] for e in errors if e['gold']['polarity'] == 'positive']
    negative = [e['predicted']['score'] for e in errors if e['gold']['polarity'] == 'negative']
    return neutral, positive, negative

laptop_neutral, laptop_positive, laptop_negative = separate_by_gold(laptop_errors)
restaurant_neutral, restaurant_positive, restaurant_negative = separate_by_gold(restaurant_errors)

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Wrong Polarity Score Distributions: Laptop vs Restaurant', fontsize=16, fontweight='bold')

threshold_pos = 0.1
threshold_neg = -0.1

# Laptop - Neutral
ax = axes[0, 0]
ax.hist(laptop_neutral, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(threshold_pos, color='green', linestyle='--', linewidth=2, label=f'Threshold +{threshold_pos}')
ax.axvline(threshold_neg, color='red', linestyle='--', linewidth=2, label=f'Threshold {threshold_neg}')
ax.axvline(np.mean(laptop_neutral), color='orange', linestyle='-', linewidth=2, label=f'Mean: {np.mean(laptop_neutral):.3f}')
ax.set_title(f'Laptop: Gold=Neutral (n={len(laptop_neutral)})', fontweight='bold')
ax.set_xlabel('Predicted Score')
ax.set_ylabel('Count')
ax.legend()
ax.grid(alpha=0.3)

# Laptop - Positive
ax = axes[0, 1]
ax.hist(laptop_positive, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
ax.axvline(threshold_pos, color='green', linestyle='--', linewidth=2, label=f'Threshold +{threshold_pos}')
ax.axvline(np.mean(laptop_positive), color='orange', linestyle='-', linewidth=2, label=f'Mean: {np.mean(laptop_positive):.3f}')
ax.set_title(f'Laptop: Gold=Positive (n={len(laptop_positive)})', fontweight='bold')
ax.set_xlabel('Predicted Score')
ax.set_ylabel('Count')
ax.legend()
ax.grid(alpha=0.3)

# Laptop - Negative
ax = axes[0, 2]
ax.hist(laptop_negative, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
ax.axvline(threshold_neg, color='red', linestyle='--', linewidth=2, label=f'Threshold {threshold_neg}')
ax.axvline(np.mean(laptop_negative), color='orange', linestyle='-', linewidth=2, label=f'Mean: {np.mean(laptop_negative):.3f}')
ax.set_title(f'Laptop: Gold=Negative (n={len(laptop_negative)})', fontweight='bold')
ax.set_xlabel('Predicted Score')
ax.set_ylabel('Count')
ax.legend()
ax.grid(alpha=0.3)

# Restaurant - Neutral
ax = axes[1, 0]
ax.hist(restaurant_neutral, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(threshold_pos, color='green', linestyle='--', linewidth=2, label=f'Threshold +{threshold_pos}')
ax.axvline(threshold_neg, color='red', linestyle='--', linewidth=2, label=f'Threshold {threshold_neg}')
ax.axvline(np.mean(restaurant_neutral), color='orange', linestyle='-', linewidth=2, label=f'Mean: {np.mean(restaurant_neutral):.3f}')
ax.set_title(f'Restaurant: Gold=Neutral (n={len(restaurant_neutral)})', fontweight='bold')
ax.set_xlabel('Predicted Score')
ax.set_ylabel('Count')
ax.legend()
ax.grid(alpha=0.3)

# Restaurant - Positive
ax = axes[1, 1]
ax.hist(restaurant_positive, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
ax.axvline(threshold_pos, color='green', linestyle='--', linewidth=2, label=f'Threshold +{threshold_pos}')
ax.axvline(np.mean(restaurant_positive), color='orange', linestyle='-', linewidth=2, label=f'Mean: {np.mean(restaurant_positive):.3f}')
ax.set_title(f'Restaurant: Gold=Positive (n={len(restaurant_positive)})', fontweight='bold')
ax.set_xlabel('Predicted Score')
ax.set_ylabel('Count')
ax.legend()
ax.grid(alpha=0.3)

# Restaurant - Negative
ax = axes[1, 2]
ax.hist(restaurant_negative, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
ax.axvline(threshold_neg, color='red', linestyle='--', linewidth=2, label=f'Threshold {threshold_neg}')
ax.axvline(np.mean(restaurant_negative), color='orange', linestyle='-', linewidth=2, label=f'Mean: {np.mean(restaurant_negative):.3f}')
ax.set_title(f'Restaurant: Gold=Negative (n={len(restaurant_negative)})', fontweight='bold')
ax.set_xlabel('Predicted Score')
ax.set_ylabel('Count')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('wrong_polarity_score_distribution.png', dpi=150, bbox_inches='tight')
print("Saved visualization to: wrong_polarity_score_distribution.png")

# Print key statistics
print("\n" + "="*80)
print("KEY STATISTICS")
print("="*80)
print("\nLAPTOP:")
print(f"  Neutral misclass mean: {np.mean(laptop_neutral):.3f} (std: {np.std(laptop_neutral):.3f})")
print(f"  Positive misclass mean: {np.mean(laptop_positive):.3f} (std: {np.std(laptop_positive):.3f})")
print(f"  Negative misclass mean: {np.mean(laptop_negative):.3f} (std: {np.std(laptop_negative):.3f})")

print("\nRESTAURANT:")
print(f"  Neutral misclass mean: {np.mean(restaurant_neutral):.3f} (std: {np.std(restaurant_neutral):.3f})")
print(f"  Positive misclass mean: {np.mean(restaurant_positive):.3f} (std: {np.std(restaurant_positive):.3f})")
print(f"  Negative misclass mean: {np.mean(restaurant_negative):.3f} (std: {np.std(restaurant_negative):.3f})")

print("\n" + "="*80)
print("THRESHOLD ANALYSIS")
print("="*80)

# Count how many would be fixed with wider threshold
laptop_neutral_in_threshold = sum(1 for s in laptop_neutral if -0.15 <= s <= 0.15)
restaurant_neutral_in_threshold = sum(1 for s in restaurant_neutral if -0.15 <= s <= 0.15)

print(f"\nWith threshold ±0.15 instead of ±0.10:")
print(f"  Laptop neutral: {laptop_neutral_in_threshold}/{len(laptop_neutral)} would be corrected ({laptop_neutral_in_threshold/len(laptop_neutral)*100:.1f}%)")
print(f"  Restaurant neutral: {restaurant_neutral_in_threshold}/{len(restaurant_neutral)} would be corrected ({restaurant_neutral_in_threshold/len(restaurant_neutral)*100:.1f}%)")
