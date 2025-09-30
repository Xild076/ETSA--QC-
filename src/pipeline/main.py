import argparse
import json
import logging
from typing import Any, Dict, List, Tuple
from tqdm import tqdm
from dotenv import load_dotenv

import config
from pipeline import build_default_pipeline
from benchmark import run_benchmark, get_dataset
from optimization import run_optimization, get_best_combiner_config
from cache_manager import PipelineCache

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATASET_CHOICES = ["test_laptop_2014", "test_restaurant_2014"]


def _resolve_datasets(selection: str) -> List[str]:
    if selection == "both":
        return DATASET_CHOICES
    return [selection]

def pre_cache_dataset(dataset_name: str):
    """
    Runs the expensive Stage 1 of the pipeline on a dataset to populate the intermediate cache.
    This is a prerequisite for running fast optimization.
    """
    logger.info(f"Starting pre-caching process for dataset: '{dataset_name}'...")
    
    try:
        dataset_items = get_dataset(dataset_name)
    except FileNotFoundError as e:
        logger.error(e)
        return

    # Use a default pipeline configuration for caching, as this stage is pre-combiner.
    pipeline = build_default_pipeline()
    
    # Disable internal caching within the pipeline process call for this specific task,
    # as we are only interested in generating the cache, not reading from it.
    pipeline.use_cache = False
    
    cache_hits = 0
    cache_misses = 0
    
    with tqdm(dataset_items, desc=f"Caching {dataset_name}", unit="sentence") as progress:
        for item in progress:
            # Check if cache already exists to avoid re-processing
            if pipeline.cache.get_intermediate_results(item.text):
                cache_hits += 1
            else:
                cache_misses += 1
                try:
                    # Run the full pipeline to generate and store the intermediate result
                    pipeline.process(item.text)
                except Exception as e:
                    logger.error(f"Failed to process and cache sentence ID {item.sentence_id}: {e}")
            progress.set_postfix({"hits": cache_hits, "misses": cache_misses})
            
    logger.info(f"Pre-caching for '{dataset_name}' complete.")
    logger.info(f"Cache Status: {cache_hits} items already cached, {cache_misses} new items processed and cached.")


def do_benchmark(args):
    """Handler for the 'benchmark' command."""
    datasets = _resolve_datasets(args.dataset)
    aggregated_metrics = {}

    for dataset in datasets:
        run_name = args.name if len(datasets) == 1 else f"{args.name}_{dataset}"
        logger.info("Starting benchmark run '%s' on dataset '%s'...", run_name, dataset)

        metrics = run_benchmark(
            run_name=run_name,
            dataset_name=dataset,
            limit=args.limit,
            combiner=args.combiner,
        )

        aggregated_metrics[dataset] = metrics

        logger.info("Benchmark complete for '%s'. Metrics:", dataset)
        print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}, indent=2))

        report_path = metrics.get('metrics_path', 'output/benchmarks')
        logger.info("Full report saved in the directory of: %s", report_path)

    return aggregated_metrics


def do_optimize(args):
    """Handler for the 'optimize' command."""
    datasets = _resolve_datasets(args.dataset)
    dataset_loaders: List[Tuple[str, callable]] = [
        (dataset, (lambda dataset_name=dataset: get_dataset(dataset_name)))
        for dataset in datasets
    ]

    if len(datasets) > 1 and args.separate:
        results_by_dataset = {}
        for dataset, loader in dataset_loaders:
            logger.info("Starting combiner optimization (separate) for dataset '%s'...", dataset)
            results = run_optimization(
                [(dataset, loader)],
                combiner_name=args.combiner,
                n_trials=args.trials,
                auto_fill_cache=not args.no_autofill,
            )
            results_by_dataset[dataset] = results
            _log_optimization_summary(dataset, results)
        return results_by_dataset

    logger.info("Starting joint combiner optimization for datasets: %s", "+".join(datasets))
    results = run_optimization(
        dataset_loaders,
        combiner_name=args.combiner,
        n_trials=args.trials,
        auto_fill_cache=not args.no_autofill,
    )
    _log_optimization_summary("+".join(datasets), results)
    return results


def _log_optimization_summary(label: str, results: Dict[str, Any]) -> None:
    if not results:
        logger.warning("No optimization results available for '%s'.", label)
        return

    best_combiner, best_params = get_best_combiner_config(results)
    logger.info("\n--- Optimization Summary (%s) ---", label)
    for combiner, result in results.items():
        logger.info("  - %s: Best macro accuracy = %.4f", combiner, result.best_score)
        for attr_key, attr_value in sorted(result.best_trial_user_attrs.items()):
            if attr_key.startswith("accuracy/"):
                logger.info("      %s = %.4f", attr_key.split("/", 1)[1], attr_value)
        logger.info("      params = %s", result.best_params)

    if best_combiner:
        logger.info("\nBest overall combiner for '%s' is '%s'", label, best_combiner)
        logger.info("Parameters: %s", best_params)
        safe_label = label.replace('/', '_')
        logger.info(
            "Optimized configuration stored in '%s'",
            config.CACHE_DIR / f"best_combiner_config_{safe_label}.json",
        )


def do_cache(args):
    """Handler for the 'cache' command."""
    for dataset in _resolve_datasets(args.dataset):
        pre_cache_dataset(dataset)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Control Center for the Sentiment Analysis Pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Cache Command ---
    parser_cache = subparsers.add_parser("cache", help="Pre-cache a dataset for fast optimization and analysis.")
    parser_cache.add_argument("dataset", choices=DATASET_CHOICES + ["both"], help="The dataset to process and cache. Use 'both' to process all datasets.")
    parser_cache.set_defaults(func=do_cache)

    # --- Benchmark Command ---
    parser_benchmark = subparsers.add_parser("benchmark", help="Run a full benchmark on a dataset.")
    parser_benchmark.add_argument("dataset", choices=DATASET_CHOICES + ["both"], help="The dataset to evaluate against. Use 'both' to evaluate all datasets.")
    parser_benchmark.add_argument("-n", "--name", type=str, default="benchmark_run", help="A custom name for the benchmark output folder.")
    parser_benchmark.add_argument("-l", "--limit", type=int, default=None, help="Limit the benchmark to a random subset of N sentences.")
    parser_benchmark.add_argument("-c", "--combiner", type=str, default=None, help="Specify a combiner to use, overriding the optimized default (e.g., 'contextual_v3').")
    parser_benchmark.set_defaults(func=do_benchmark)

    # --- Optimize Command ---
    parser_optimize = subparsers.add_parser("optimize", help="Run hyperparameter optimization for sentiment combiners.")
    parser_optimize.add_argument("dataset", choices=DATASET_CHOICES + ["both"], help="Dataset to use for optimization (must be cached first). Use 'both' to optimize across all datasets.")
    parser_optimize.add_argument("-c", "--combiner", type=str, default=None, help="Optimize only a specific combiner (e.g., 'contextual_v3'). By default, optimizes all.")
    parser_optimize.add_argument("-t", "--trials", type=int, default=200, help="Number of Optuna trials to run.")
    parser_optimize.add_argument("--no-autofill", action="store_true", help="Disable automatic cache population during optimization.")
    parser_optimize.add_argument("--separate", action="store_true", help="Optimize each dataset independently even when 'both' is selected.")
    parser_optimize.set_defaults(func=do_optimize)
    
    # --- Clear Cache Command ---
    parser_clear = subparsers.add_parser("clear-cache", help="Clear all intermediate pipeline results.")
    def do_clear_cache(args):
        logger.info("Clearing intermediate pipeline cache...")
        cache = PipelineCache()
        cache.clear_cache()
        logger.info("Cache cleared.")
    parser_clear.set_defaults(func=do_clear_cache)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
