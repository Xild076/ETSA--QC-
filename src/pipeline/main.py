import argparse
import json
import logging
from typing import Any, Dict, List
from tqdm import tqdm
from dotenv import load_dotenv

from pipeline import build_default_pipeline
from benchmark import run_benchmark, get_dataset
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
        )

        aggregated_metrics[dataset] = metrics

        logger.info("Benchmark complete for '%s'. Metrics:", dataset)
        print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}, indent=2))

        report_path = metrics.get('metrics_path', 'output/benchmarks')
        logger.info("Full report saved in the directory of: %s", report_path)

    return aggregated_metrics

def do_cache(args):
    """Handler for the 'cache' command."""
    for dataset in _resolve_datasets(args.dataset):
        pre_cache_dataset(dataset)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Control Center for the Sentiment Analysis Pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

                           
    parser_cache = subparsers.add_parser("cache", help="Pre-cache a dataset for fast optimization and analysis.")
    parser_cache.add_argument("dataset", choices=DATASET_CHOICES + ["both"], help="The dataset to process and cache. Use 'both' to process all datasets.")
    parser_cache.set_defaults(func=do_cache)

                               
    parser_benchmark = subparsers.add_parser("benchmark", help="Run a full benchmark on a dataset.")
    parser_benchmark.add_argument("dataset", choices=DATASET_CHOICES + ["both"], help="The dataset to evaluate against. Use 'both' to evaluate all datasets.")
    parser_benchmark.add_argument("-n", "--name", type=str, default="benchmark_run", help="A custom name for the benchmark output folder.")
    parser_benchmark.add_argument("-l", "--limit", type=int, default=None, help="Limit the benchmark to a random subset of N sentences.")
    parser_benchmark.set_defaults(func=do_benchmark)
    
                                 
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
