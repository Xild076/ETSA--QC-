import logging
import sys

# Configure basic logging at the very top
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Benchmark script started.")

import os
import json
import logging
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Callable, Optional
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from dotenv import load_dotenv
import random

import sys
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Load environment variables
load_dotenv(os.path.join(SRC_DIR, '.env'))

from src.run_pipeline import run_pipeline_for_text

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

def load_semeval_2016_task5(file_path: str) -> List[Dict[str, Any]]:
    tree = ET.parse(file_path)
    root = tree.getroot()
    data = []
    for review in root.findall('Review'):
        sentences = review.findall('sentences/sentence')
        sentences_with_aspects = []
        all_aspects = []
        
        for sentence in sentences:
            text_elem = sentence.find('text')
            if text_elem is None or not text_elem.text:
                continue
                
            sentence_text = text_elem.text.strip()
            sentence_aspects = []
            
            for opinion in sentence.findall('.//Opinion'):
                category = opinion.get('category')
                polarity = opinion.get('polarity')
                target = opinion.get('target', 'NULL')
                
                if not category or not polarity or polarity == 'conflict':
                    continue
                
                parts = category.split('#')
                if len(parts) != 2:
                    continue
                
                entity, aspect_term = parts
                
                if target and target != 'NULL':
                    term = target
                else:
                    term = aspect_term
                
                aspect_data = {
                    "term": term, 
                    "polarity": polarity, 
                    "entity": entity,
                    "sentence_id": sentence.get('id')
                }
                sentence_aspects.append(aspect_data)
                all_aspects.append(aspect_data)
            
            if sentence_aspects:
                sentences_with_aspects.append({
                    "text": sentence_text,
                    "aspects": sentence_aspects,
                    "sentence_id": sentence.get('id')
                })
        
        if sentences_with_aspects:
            full_text = " ".join(s["text"] for s in sentences_with_aspects)
            
            data.append({
                "text": full_text, 
                "aspects": all_aspects, 
                "id": review.get('rid'),
                "sentences": sentences_with_aspects,
                "is_long_document": len(sentences_with_aspects) > 1
            })
    return data

def load_semeval_2014_task4(file_path: str) -> List[Dict[str, Any]]:
    """Loads SemEval 2014 Task 4 datasets with sentiment polarity labels."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    data = []
    
    for sentence in root.findall('sentence'):
        text_elem = sentence.find('text')
        if text_elem is None or not text_elem.text:
            continue
            
        sentence_text = text_elem.text.strip()
        sentence_aspects = []
        
        # Extract aspect terms with polarity
        aspect_terms = sentence.find('aspectTerms')
        if aspect_terms is not None:
            for aspect_term in aspect_terms.findall('aspectTerm'):
                term = aspect_term.get('term')
                polarity = aspect_term.get('polarity')
                
                if term and polarity and polarity != 'conflict':
                    aspect_data = {
                        "term": term,
                        "polarity": polarity,
                        "entity": "NULL",  # 2014 format doesn't have entity categories
                        "sentence_id": sentence.get('id')
                    }
                    sentence_aspects.append(aspect_data)
        
        # Also extract aspect categories if available
        aspect_categories = sentence.find('aspectCategories')
        if aspect_categories is not None:
            for aspect_category in aspect_categories.findall('aspectCategory'):
                category = aspect_category.get('category')
                polarity = aspect_category.get('polarity')
                
                if category and polarity and polarity != 'conflict':
                    aspect_data = {
                        "term": category,
                        "polarity": polarity,
                        "entity": category,
                        "sentence_id": sentence.get('id')
                    }
                    sentence_aspects.append(aspect_data)
        
        if sentence_aspects:
            data.append({
                "text": sentence_text,
                "aspects": sentence_aspects,
                "id": sentence.get('id'),
                "sentences": [{
                    "text": sentence_text,
                    "aspects": sentence_aspects,
                    "sentence_id": sentence.get('id')
                }],
                "is_long_document": False
            })
    
    return data

def get_dataset_loader(dataset_name: str) -> Callable:
    """Returns the correct dataset loader based on the name."""
    if "2016" in dataset_name:
        return load_semeval_2016_task5
    return load_semeval_2014_task4

def get_dataset_path(dataset_name: str) -> str:
    """Maps a dataset name to its file path."""
    mapping = {
        "test_laptop_2014": "data/dataset/test_laptop_2014.xml",
        "test_restaurant_2014": "data/dataset/test_restaurant_2014.xml",
        "test_laptop_2016": "data/dataset/test_laptop_2016.xml", 
        "test_restaurant_2016": "data/dataset/test_restaurant_2016.xml",
        # Unified datasets
        "unified_2014": ["data/dataset/test_restaurant_2014.xml", "data/dataset/test_laptop_2014.xml"],
        "unified_2016": ["data/dataset/test_restaurant_2016.xml", "data/dataset/test_laptop_2016.xml"],
    }
    path = mapping.get(dataset_name)
    if isinstance(path, list):
        # Multi-file dataset
        for p in path:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Dataset file not found: {p}")
        return path
    elif not path or not os.path.exists(path):
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found at path: {path}")
    return path

def score_to_polarity(score: float, pos_thresh: float, neg_thresh: float) -> str:
    if score > pos_thresh: return "positive"
    if score < neg_thresh: return "negative"
    return "neutral"

def find_best_from_names(entity_names: List[str], aspect_term: str) -> str:
    a_raw = aspect_term or ""
    a = a_raw.lower().strip()
    if not a: return a_raw

    # Handle spelling variations and synonyms
    spelling_variants = {
        'ambience': ['ambiance', 'atmosphere'],
        'ambiance': ['ambience', 'atmosphere'], 
        'anecdotes/miscellaneous': ['place', 'restaurant', 'location'],
        'place': ['anecdotes/miscellaneous', 'restaurant', 'location'],
        'restaurant': ['place', 'anecdotes/miscellaneous', 'location']
    }
    
    # Category mapping for specific items to general categories
    category_mappings = {
        'food': ['bread', 'coffee', 'tea', 'wine', 'beer', 'drink', 'drinks', 'meal', 'dish', 'dishes', 
                'pasta', 'pizza', 'burger', 'sandwich', 'salad', 'soup', 'dessert', 'appetizer', 
                'entree', 'main course', 'seafood', 'chicken', 'beef', 'pork', 'fish', 'sushi',
                'rice', 'noodles', 'sauce', 'cheese', 'vegetable', 'fruit', 'meat', 'cuisine',
                'noodles', 'portions'],
        'service': ['waiter', 'waitress', 'server', 'staff', 'waitstaff', 'manager', 'host', 'hostess',
                   'delivery', 'takeout', 'reservation', 'wait time', 'waiting', 'service', 'attendant'],
        'ambience': ['atmosphere', 'ambiance', 'environment', 'setting', 'decor', 'decoration',
                    'music', 'noise', 'lighting', 'seating', 'tables', 'chairs', 'interior'],
        'price': ['cost', 'price', 'expensive', 'cheap', 'affordable', 'value', 'money', 'bill', 'tab'],
        'anecdotes/miscellaneous': ['place', 'restaurant', 'location', 'establishment', 'experience', 'service', 'food', 'ambience', 'staff', 'it', 'everything'],
        # 2016 dataset abstract categories - map to broader contexts
        'GENERAL': ['place', 'restaurant', 'location', 'establishment', 'this', 'it', 'here', 'overall', 'experience'],
        'QUALITY': ['quality', 'food', 'dish', 'meal', 'service', 'staff', 'ambience', 'decor', 'taste', 'flavor', 'performance', 'speed', 'battery'],
        'PRICES': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'value', 'money', 'bill', 'tab', 'tip'],
        'MISCELLANEOUS': ['everything', 'nothing', 'anything', 'something', 'all', 'none', 'other']
    }
    
    # Check for exact matches including variants
    for name in entity_names:
        n_raw = name or ""
        n = n_raw.lower().strip()
        if not n: continue
        
        if n == a: return n_raw
        
        # Check spelling variants
        if a in spelling_variants:
            if n in spelling_variants[a]:
                return n_raw
        if n in spelling_variants:
            if a in spelling_variants[n]:
                return n_raw

    a_tokens = set(a.split())
    best_name, best_score = a_raw, 0.0

    for name in entity_names:
        n_raw = name or ""
        n = n_raw.lower().strip()
        if not n: continue
        
        if n == a: return n_raw
        
        n_tokens = set(n.split())
        
        overlap = len(a_tokens & n_tokens)
        union = len(a_tokens | n_tokens)
        jaccard_score = overlap / union if union > 0 else 0
        
        aspect_tokens_covered = len(a_tokens & n_tokens)
        aspect_coverage = aspect_tokens_covered / len(a_tokens) if a_tokens else 0
        
        pred_tokens_relevant = len(a_tokens & n_tokens)
        pred_precision = pred_tokens_relevant / len(n_tokens) if n_tokens else 0
        
        exact_substring_score = 0.0
        if a in n:
            exact_substring_score = 0.8
        elif n in a:
            exact_substring_score = 0.6
        
        affix_score = 0.0
        if a.startswith(n[:3]) and len(n) >= 3:
            affix_score += 0.3
        if a.endswith(n[-3:]) and len(n) >= 3:
            affix_score += 0.3
        if n.startswith(a[:3]) and len(a) >= 3:
            affix_score += 0.2
        if n.endswith(a[-3:]) and len(a) >= 3:
            affix_score += 0.2
        
        word_order_score = 0.0
        if len(a_tokens & n_tokens) >= 2:
            common_tokens = sorted(list(a_tokens & n_tokens))
            if len(common_tokens) >= 2:
                a_positions = {token: i for i, token in enumerate(a.split()) if token in common_tokens}
                n_positions = {token: i for i, token in enumerate(n.split()) if token in common_tokens}
                
                order_preserved = 0
                for i in range(len(common_tokens) - 1):
                    token1, token2 = common_tokens[i], common_tokens[i + 1]
                    if (a_positions.get(token1, 0) < a_positions.get(token2, 0) and 
                        n_positions.get(token1, 0) < n_positions.get(token2, 0)):
                        order_preserved += 1
                word_order_score = order_preserved / max(1, len(common_tokens) - 1)
        
        combined_score = (
            0.3 * jaccard_score +
            0.25 * aspect_coverage +
            0.15 * pred_precision +
            0.2 * exact_substring_score +
            0.1 * affix_score +
            0.0 * word_order_score
        )
        
        if combined_score > best_score and combined_score >= 0.2:
            best_name, best_score = n_raw, combined_score

    # If no good match found, try category-based mapping
    if best_score < 0.2:
        # Handle abstract 2016 categories specially
        if a.upper() in ['GENERAL', 'QUALITY', 'PRICES', 'MISCELLANEOUS']:
            category_key = a.upper()
            if category_key in category_mappings:
                keywords = category_mappings[category_key]
                
                # For abstract categories, use fuzzy matching
                for name in entity_names:
                    n_lower = name.lower()
                    # Check if any keyword appears in entity name
                    for keyword in keywords:
                        if keyword in n_lower or n_lower in keyword:
                            return name
                    
                    # For GENERAL, also match restaurant/place-related entities
                    if category_key == 'GENERAL':
                        general_terms = ['place', 'restaurant', 'this', 'it', 'here', 'there']
                        if any(term in n_lower for term in general_terms):
                            return name
                            
                    # For QUALITY, match evaluative terms
                    elif category_key == 'QUALITY':
                        quality_indicators = ['food', 'dish', 'meal', 'cuisine', 'cooking', 'works', 'install']
                        if any(term in n_lower for term in quality_indicators):
                            return name
        
        # Original category mapping logic
        for category, keywords in category_mappings.items():
            if a == category:  # Looking for category directly
                # Find entities that contain keywords from this category
                for name in entity_names:
                    n_lower = name.lower()
                    for keyword in keywords:
                        if keyword in n_lower or any(word in keyword for word in n_lower.split()):
                            return name
                            
        # Also try reverse - if we have a specific item, map to category
        for name in entity_names:
            n_lower = name.lower()
            for category, keywords in category_mappings.items():
                if a == category:  # We want this category
                    # Check if this entity contains any category keywords
                    for keyword in keywords:
                        if keyword in n_lower or n_lower in keyword:
                            return name

    return best_name

def match_aspects(gold_aspects: list, predicted_sentiments: dict, text: str) -> list:
    """Matches gold standard aspects with predicted sentiments."""
    matched_results = []
    
    # If there are no predicted sentiments, we can't match anything.
    if not predicted_sentiments:
        return [{
            'term': gold['term'],
            'gold': gold['polarity'],
            'matched_entity': None,
            'score': 0.0,
            'pred': 'neutral'
        } for gold in gold_aspects]

    predicted_entity_names = list(predicted_sentiments.keys())
    
    for gold in gold_aspects:
        aspect_term = gold['term']
        
        # Find the best matching predicted entity for the current gold standard aspect
        matched_entity_name = find_best_from_names(predicted_entity_names, aspect_term)
        
        # If a match is found, get its score and determine polarity
        if matched_entity_name and matched_entity_name in predicted_sentiments:
            score = predicted_sentiments[matched_entity_name]
            
            # Simple polarity classification
            if score > 0.1:
                pred = 'positive'
            elif score < -0.1:
                pred = 'negative'
            else:
                pred = 'neutral'
        
        # If no match is found, default to neutral
        else:
            matched_entity_name = None
            score = 0.0
            pred = 'neutral'
            
        matched_results.append({
            'term': aspect_term,
            'gold': gold['polarity'],
            'matched_entity': matched_entity_name,
            'score': score,
            'pred': pred
        })
        
    return matched_results


def save_artifacts(run_dir: str, y_true: List[str], y_pred: List[str], error_analysis_data: List[Dict]):
    """Saves all benchmark artifacts (metrics, plots, errors) to the run directory."""
    if not y_true:
        logger.warning("No ground truth labels found. Skipping artifact generation.")
        return

    labels = sorted(list(set(y_true) | set(y_pred)))
    
    # --- Start of Change: Save metrics first ---
    # Calculate and save core metrics immediately to prevent data loss if plotting fails.
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, labels=labels, digits=4, output_dict=True, zero_division=0)
    }
    metrics_path = os.path.join(run_dir, "metrics.json")
    try:
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Core metrics saved to {metrics_path}")
    except Exception as e:
        logger.error(f"Failed to save core metrics: {e}")
    # --- End of Change ---

    report_str = classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0)
    logger.info(f"\n--- Benchmark Results ---\n{report_str}")
    
    cm_path = os.path.join(run_dir, "confusion_matrix.png")
    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(cm_path)
        plt.close()
        logger.info(f"Confusion Matrix saved to {cm_path}")
    except Exception as e:
        logger.error(f"Failed to generate confusion matrix: {e}")

    if error_analysis_data:
        df_errors = pd.DataFrame(error_analysis_data)
        errors_path = os.path.join(run_dir, "error_analysis.csv")
        df_errors.to_csv(errors_path, index=False)
        logger.info(f"Error analysis saved to {errors_path}")

def run_benchmark(
    run_name: str,
    dataset_name: str,
    run_mode: str,
    limit: int,
    pos_thresh: float,
    neg_thresh: float,
    progress_callback: Optional[Callable] = None
):
    # Standardize run_name format: {run_name}_{dataset_name}_{mode}_{timestamp}
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    standardized_run_name = f"{run_name}_{dataset_name}_{run_mode}_{timestamp}"
    
    run_dir = os.path.join("outputs", "benchmarks", standardized_run_name)
    traces_dir = os.path.join(run_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)
    run_dir = os.path.join("outputs", "benchmarks", standardized_run_name)
    traces_dir = os.path.join(run_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)
    log_file_path = os.path.join(run_dir, "benchmark.log")
    
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file_path for h in logger.handlers):
        logger.addHandler(file_handler)
    
    logger.propagate = False

    logger.info(f"Starting benchmark: {standardized_run_name}")
    logger.info(f"Parameters: dataset={dataset_name}, mode={run_mode}, limit={limit}, pos_thresh={pos_thresh}, neg_thresh={neg_thresh}")

    # Initialize the pipeline once to avoid reloading models for every item
    logger.info(f"Initializing pipeline in '{run_mode}' mode...")
    pipeline = run_pipeline_for_text(text="", mode=run_mode, return_pipeline=True)
    logger.info("Pipeline initialized.")

    try:
        dataset_path = get_dataset_path(dataset_name)
        loader = get_dataset_loader(dataset_name)
        
        # Handle unified datasets (multiple files)
        if isinstance(dataset_path, list):
            data = []
            for path in dataset_path:
                file_data = loader(path)
                # Add source info to each item
                for item in file_data:
                    item['source_file'] = os.path.basename(path)
                data.extend(file_data)
            logger.info(f"Loaded {len(data)} items from unified dataset {dataset_name} ({len(dataset_path)} files).")
        else:
            data = loader(dataset_path)
            logger.info(f"Loaded {len(data)} items from {dataset_name}.")
    except FileNotFoundError as e:
        logger.error(str(e))
        raise

    # Randomize the data order to get better coverage in limited runs
    if limit > 0 and limit < len(data):
        # random.seed(42)  # Fixed seed for reproducibility
        random.shuffle(data)
        logger.info(f"Shuffled dataset for better coverage in limited runs.")

    if limit > 0:
        data = data[:limit]
        logger.info(f"Limiting run to {len(data)} items.")

    y_true, y_pred = [], []
    error_analysis_data = []
    processed_items = set()

    for i, item in enumerate(tqdm(data, desc="Running Benchmark")):
        text = item['text']
        gold_aspects = item['aspects']
        item_id = item.get('id', str(i))
        
        try:
            # Use the pre-initialized pipeline to execute the analysis
            pipeline_trace = pipeline.run(text)
            pred_sentiments = pipeline_trace.get("final_sentiments", {})
            matched_results = match_aspects(gold_aspects, pred_sentiments, text)
            
            for res in matched_results:
                y_true.append(res['gold'])
                y_pred.append(res['pred'])
                
                if res['gold'] != res['pred']:
                    trace_path = None
                    if item_id not in processed_items:
                        trace_filename = f"trace_{item_id}.json"
                        trace_path = os.path.join(traces_dir, trace_filename)
                        
                        detailed_trace = {
                            "item_id": item_id,
                            "input_text": text,
                            "gold_aspects": gold_aspects,
                            "pipeline_mode": run_mode,
                            "execution_trace": pipeline_trace,
                            "matched_results": matched_results,
                            "thresholds": {"positive": pos_thresh, "negative": neg_thresh}
                        }
                        
                        with open(trace_path, 'w') as f:
                            json.dump(detailed_trace, f, indent=2, default=str)
                        processed_items.add(item_id)

                    error_analysis_data.append({
                        "id": item_id,
                        "text": text,
                        "term": res['term'],
                        "gold": res['gold'],
                        "pred": res['pred'],
                        "score": res['score'],
                        "matched_entity": res['matched_entity'],
                        "trace_path": trace_path
                    })
                    
        except Exception as e:
            logger.error(f"Failed to process item {item_id}: {e}", exc_info=True)

        if progress_callback:
            progress_callback(i, len(data))

    logger.info("Benchmark processing complete. Generating artifacts...")
    save_artifacts(run_dir, y_true, y_pred, error_analysis_data)
    logger.info(f"Benchmark run '{run_name}' finished. Saved {len(processed_items)} detailed error traces.")
    
    logger.removeHandler(file_handler)
    file_handler.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        prog="benchmark", 
        description="Run ETSA benchmark evaluation on sentiment analysis datasets"
    )
    parser.add_argument(
        "--dataset", 
        required=True, 
        choices=["test_laptop_2014", "test_restaurant_2014", "test_laptop_2016", "test_restaurant_2016", "unified_2014", "unified_2016"],
        help="Dataset to benchmark against"
    )
    parser.add_argument(
        "--mode", 
        default="full_stack", 
        choices=["full_stack", "efficiency", "no_formulas", "vader_baseline", "transformer_absa", "ner_basic", "no_modifiers", "no_relations"],
        help="Pipeline mode to run. full_stack (complete pipeline), efficiency (fast rule-based), no_formulas (ablation test - null averages), vader_baseline (VADER sentiment on all entities), transformer_absa (DeBERTa-v3 end-to-end ABSA), ner_basic (basic NER no coreference), no_modifiers (no modifier extraction), no_relations (no relation extraction)"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=0,
        help="Limit number of items to process (0 = all)"
    )
    parser.add_argument(
        "--pos-threshold", 
        type=float, 
        default=0.1,
        help="Positive sentiment threshold"
    )
    parser.add_argument(
        "--neg-threshold", 
        type=float, 
        default=-0.1,
        help="Negative sentiment threshold"
    )
    parser.add_argument(
        "--run-name", 
        default=None,
        help="Custom name for this benchmark run (default: {dataset}_{mode})"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Use unified naming scheme: dataset_mode (same as app.py)
    if args.run_name is None:
        run_name = f"{args.dataset}_{args.mode}"
    else:
        run_name = args.run_name
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Validate dataset exists
        dataset_path = get_dataset_path(args.dataset)
        logger.info(f"Running benchmark on dataset: {args.dataset}")
        logger.info(f"Dataset path(s): {dataset_path}")
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Run name: {run_name}")
        logger.info(f"Thresholds: pos={args.pos_threshold}, neg={args.neg_threshold}")
        
        if args.limit > 0:
            logger.info(f"Processing limit: {args.limit} items")
        else:
            logger.info("Processing all items")
        
        # Run the benchmark
        run_benchmark(
            run_name=run_name,
            dataset_name=args.dataset,
            run_mode=args.mode,
            limit=args.limit,
            pos_thresh=args.pos_threshold,
            neg_thresh=args.neg_threshold
        )
        
        logger.info("Benchmark completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"Dataset error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.warning("Benchmark interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
