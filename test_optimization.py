#!/usr/bin/env python3
"""Quick harness to run the joint combiner optimizer across the SemEval datasets."""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline.benchmark import get_dataset
from src.pipeline.optimization import run_optimization, get_best_combiner_config

DATASETS = ["test_laptop_2014", "test_restaurant_2014"]


def build_loaders(selected: List[str]) -> List[Tuple[str, callable]]:
    return [(name, (lambda dataset=name: get_dataset(dataset))) for name in selected]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the combiner optimizer for quick verification.")
    parser.add_argument("dataset", choices=DATASETS + ["both"], default="both", nargs="?", help="Dataset to optimize. Use 'both' for the joint run.")
    parser.add_argument("-t", "--trials", type=int, default=50, help="Number of Optuna trials to execute.")
    parser.add_argument("-c", "--combiner", type=str, default=None, help="Optional combiner key to optimize.")
    parser.add_argument("--no-autofill", action="store_true", help="Disable automatic cache population.")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "both" else [args.dataset]
    loaders = build_loaders(datasets)

    print(f"=== Optimizing combiners for datasets: {', '.join(datasets)} ===")
    results = run_optimization(
        loaders,
        combiner_name=args.combiner,
        n_trials=args.trials,
        auto_fill_cache=not args.no_autofill,
    )

    best_combiner, best_params = get_best_combiner_config(results)
    for name, result in results.items():
        print(f"\n[{name}] best macro accuracy: {result.best_score:.4f}")
        for attr, value in sorted(result.best_trial_user_attrs.items()):
            if attr.startswith("accuracy/"):
                print(f"  {attr.split('/', 1)[1]} = {value:.4f}")
        print(f"  params: {result.best_params}")

    if best_combiner:
        print("\n=== Best Overall ===")
        print(f"Combiner: {best_combiner}")
        print(f"Macro accuracy: {results[best_combiner].best_score:.4f}")
        print(f"Params: {best_params}\n")


if __name__ == "__main__":
    main()
