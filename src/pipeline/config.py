"""Centralised filesystem configuration for the sentiment pipeline."""

from pathlib import Path

                    
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "cache"

                                      
SEMEVAL_2014_LAPTOP_TEST = DATA_DIR / "dataset" / "test_laptop_2014.xml"
SEMEVAL_2014_RESTAURANT_TEST = DATA_DIR / "dataset" / "test_restaurant_2014.xml"

                            
ASPECT_EXTRACTOR_RULES = MODELS_DIR / "ner_coref" / "rules" / "best" / "best_rules.pkl"

                     
PIPELINE_INTERMEDIATE_CACHE_DIR = CACHE_DIR / "pipeline_intermediate"
OPTIMIZATION_RESULTS_DIR = CACHE_DIR / "optimization_results"
LLM_RELATION_CACHE = CACHE_DIR / "relation_cache.json"
LLM_MODIFIER_CACHE = CACHE_DIR / "modifier_cache.json"

                      
BENCHMARK_OUTPUT_DIR = OUTPUT_DIR / "benchmarks"