"""Command-line entry points for running pipeline benchmarks and caching."""

import argparse
import json
import logging
from typing import Any, Dict, List

from dotenv import load_dotenv
from tqdm import tqdm

try:  # pragma: no cover - support package and script execution
    from .pipeline import build_default_pipeline
    from .benchmark import run_benchmark, get_dataset
    from .cache_manager import PipelineCache
except ImportError:  # pragma: no cover - fallback when executed directly
    from pipeline import build_default_pipeline
    from benchmark import run_benchmark, get_dataset
    from cache_manager import PipelineCache

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATASET_CHOICES = ["test_laptop_2014", "test_restaurant_2014"]


def _resolve_datasets(selection: str) -> List[str]:
    """Normalise dataset selection flags into concrete dataset names."""
    if selection == "both":
        return DATASET_CHOICES
    return [selection]

def pre_cache_dataset(dataset_name: str) -> None:
    """Populate the intermediate cache for a given dataset."""
    logger.info(f"Starting pre-caching process for dataset: '{dataset_name}'...")
    
    try:
        dataset_items = get_dataset(dataset_name)
    except FileNotFoundError as e:
        logger.error(e)
        return

    pipeline = build_default_pipeline()
    pipeline.use_cache = False
    
    cache_hits = 0
    cache_misses = 0
    
    with tqdm(dataset_items, desc=f"Caching {dataset_name}", unit="sentence") as progress:
        for item in progress:
            if pipeline.cache.get_intermediate_results(item.text, pipeline.cache_signature):
                cache_hits += 1
            else:
                cache_misses += 1
                try:
                    pipeline.process(item.text)
                except Exception as e:
                    logger.error(f"Failed to process and cache sentence ID {item.sentence_id}: {e}")
            progress.set_postfix({"hits": cache_hits, "misses": cache_misses})
            
    logger.info(f"Pre-caching for '{dataset_name}' complete.")
    logger.info(f"Cache Status: {cache_hits} items already cached, {cache_misses} new items processed and cached.")


def do_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    """Handler for the ``benchmark`` command."""
    datasets = _resolve_datasets(args.dataset)
    aggregated_metrics = {}

    for dataset in datasets:
        run_name = args.name if len(datasets) == 1 else f"{args.name}_{dataset}"
        logger.info("Starting benchmark run '%s' on dataset '%s'...", run_name, dataset)

        metrics = run_benchmark(
            run_name=run_name,
            dataset_name=dataset,
            limit=args.limit,
            ablate_modifiers=getattr(args, 'ablate_modifiers', False),
            ablate_relations=getattr(args, 'ablate_relations', False),
        )

        aggregated_metrics[dataset] = metrics

        logger.info("Benchmark complete for '%s'. Metrics:", dataset)
        print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}, indent=2))

        report_path = metrics.get('metrics_path', 'output/benchmarks')
        logger.info("Full report saved in the directory of: %s", report_path)

    return aggregated_metrics

def do_cache(args: argparse.Namespace) -> None:
    """Handler for the ``cache`` command."""
    for dataset in _resolve_datasets(args.dataset):
        pre_cache_dataset(dataset)


def do_ablation(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    """Handler for the ``ablation`` command."""
    datasets = _resolve_datasets(args.dataset)
    all_results = {}
    
    # Determine which ablation tests to run
    run_no_modifiers = args.mode in ["no-modifiers", "both"]
    run_no_relations = args.mode in ["no-relations", "both"]
    
    for dataset in datasets:
        dataset_results = {}
        
        # Run ablation: no modifiers
        if run_no_modifiers:
            run_name = f"{args.name}_no_modifiers" if len(datasets) == 1 else f"{args.name}_no_modifiers_{dataset}"
            logger.info("="*70)
            logger.info("🔬 ABLATION TEST: NO MODIFIERS")
            logger.info("   Testing system performance WITHOUT modifier extraction")
            logger.info("   Dataset: %s | Run name: %s", dataset, run_name)
            logger.info("="*70)
            
            metrics_no_mods = run_benchmark(
                run_name=run_name,
                dataset_name=dataset,
                limit=args.limit,
                ablate_modifiers=True,
                ablate_relations=False,
            )
            dataset_results['no_modifiers'] = metrics_no_mods
            
            logger.info("\n📊 NO MODIFIERS Results:")
            logger.info("   Accuracy: %.3f", metrics_no_mods.get('accuracy', 0))
            logger.info("   F1 Score: %.3f", metrics_no_mods.get('aspect_f1', 0))
            logger.info("   Balanced Accuracy: %.3f", metrics_no_mods.get('balanced_accuracy', 0))
        
        # Run ablation: no relations
        if run_no_relations:
            run_name = f"{args.name}_no_relations" if len(datasets) == 1 else f"{args.name}_no_relations_{dataset}"
            logger.info("\n" + "="*70)
            logger.info("🔬 ABLATION TEST: NO RELATIONS")
            logger.info("   Testing system performance WITHOUT relation extraction")
            logger.info("   Dataset: %s | Run name: %s", dataset, run_name)
            logger.info("="*70)
            
            metrics_no_rels = run_benchmark(
                run_name=run_name,
                dataset_name=dataset,
                limit=args.limit,
                ablate_modifiers=False,
                ablate_relations=True,
            )
            dataset_results['no_relations'] = metrics_no_rels
            
            logger.info("\n📊 NO RELATIONS Results:")
            logger.info("   Accuracy: %.3f", metrics_no_rels.get('accuracy', 0))
            logger.info("   F1 Score: %.3f", metrics_no_rels.get('aspect_f1', 0))
            logger.info("   Balanced Accuracy: %.3f", metrics_no_rels.get('balanced_accuracy', 0))
        
        all_results[dataset] = dataset_results
    
    # Print comparison summary
    logger.info("\n" + "="*70)
    logger.info("🎯 ABLATION STUDY SUMMARY")
    logger.info("="*70)
    
    for dataset, results in all_results.items():
        logger.info(f"\nDataset: {dataset}")
        
        if 'no_modifiers' in results:
            logger.info("  Without Modifiers:")
            logger.info("    - Accuracy: %.3f", results['no_modifiers'].get('accuracy', 0))
            logger.info("    - F1: %.3f", results['no_modifiers'].get('aspect_f1', 0))
            logger.info("    - Balanced Acc: %.3f", results['no_modifiers'].get('balanced_accuracy', 0))
        
        if 'no_relations' in results:
            logger.info("  Without Relations:")
            logger.info("    - Accuracy: %.3f", results['no_relations'].get('accuracy', 0))
            logger.info("    - F1: %.3f", results['no_relations'].get('aspect_f1', 0))
            logger.info("    - Balanced Acc: %.3f", results['no_relations'].get('balanced_accuracy', 0))
    
    logger.info("\n💡 To compare with full system, run a regular benchmark and compare metrics.")
    logger.info("="*70)
    
    return all_results


def main():
    """Parse CLI arguments and dispatch to the chosen sub-command."""
    parser = argparse.ArgumentParser(description="Comprehensive control center for the sentiment analysis pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

                           
    parser_cache = subparsers.add_parser("cache", help="Pre-cache a dataset for fast optimization and analysis.")
    parser_cache.add_argument("dataset", choices=DATASET_CHOICES + ["both"], help="The dataset to process and cache. Use 'both' to process all datasets.")
    parser_cache.set_defaults(func=do_cache)

                               
    parser_benchmark = subparsers.add_parser("benchmark", help="Run a full benchmark on a dataset.")
    parser_benchmark.add_argument("dataset", choices=DATASET_CHOICES + ["both"], help="The dataset to evaluate against. Use 'both' to evaluate all datasets.")
    parser_benchmark.add_argument("-n", "--name", type=str, default="benchmark_run", help="A custom name for the benchmark output folder.")
    parser_benchmark.add_argument("-l", "--limit", type=int, default=None, help="Limit the benchmark to a random subset of N sentences.")
    parser_benchmark.set_defaults(func=do_benchmark)
    
    # ABLATION
    parser_ablation = subparsers.add_parser("ablation", help="Run ablation studies to measure component contributions.")
    parser_ablation.add_argument("dataset", choices=DATASET_CHOICES + ["both"], help="The dataset to evaluate against. Use 'both' to evaluate all datasets.")
    parser_ablation.add_argument("mode", choices=["no-modifiers", "no-relations", "both"], 
                                  help="Ablation mode: 'no-modifiers' (use dummy modifier extraction), 'no-relations' (use dummy relation extraction), or 'both' (run both ablations).")
    parser_ablation.add_argument("-n", "--name", type=str, default="ablation", help="A custom name prefix for the ablation output folders.")
    parser_ablation.add_argument("-l", "--limit", type=int, default=None, help="Limit the ablation test to a random subset of N sentences.")
    parser_ablation.set_defaults(func=do_ablation)
                                     
    parser_clear = subparsers.add_parser("clear-cache", help="Clear all intermediate pipeline results.")
    def do_clear_cache(args: argparse.Namespace) -> None:
        """Clear all cached intermediate pipeline artefacts."""
        logger.info("Clearing intermediate pipeline cache...")
        cache = PipelineCache()
        cache.clear_cache()
        logger.info("Cache cleared.")
    parser_clear.set_defaults(func=do_clear_cache)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
