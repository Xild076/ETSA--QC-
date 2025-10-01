import json
import logging
import math
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import time
from dataclasses import dataclass
from copy import deepcopy

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    TPESampler = None
    MedianPruner = None
    OPTUNA_AVAILABLE = False
    
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from combiners import COMBINERS
from cache_manager import PipelineCache
import config
from utility import normalize_text
from sentiment_model import (
    AggregateSentimentModel,
    ActionSentimentModel,
    AssociationSentimentModel,
    BelongingSentimentModel,
)

# Suppress Optuna's INFO logging to keep the output clean
if OPTUNA_AVAILABLE:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    best_params: Dict[str, Any]
    best_score: float
    study_trials_dataframe: Dict[str, Any]
    evaluation_time: float
    best_trial_user_attrs: Dict[str, Any]

def _score_to_label(score: float, pos_thresh: float, neg_thresh: float) -> str:
    score = max(-1.0, min(1.0, score))
    
    if score >= pos_thresh:
        return "positive"
    if score <= neg_thresh:
        return "negative"
    return "neutral"


def _normalize_polarity(polarity: str) -> Optional[str]:
    pol = polarity.lower().strip()
    if pol in {"positive", "pos", "favorable", "good"}:
        return "positive"
    if pol in {"negative", "neg", "unfavorable", "bad"}:
        return "negative"
    if pol in {"neutral", "neu", "objective", "mixed"}:
        return "neutral"
    if pol in {"conflict", "both", "contradictory"}:
        return "neutral"
    return None

class CombinerOptimizer:

    def __init__(self, combiner_name: str, test_data_with_cache: List[Dict[str, Any]]):
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna is required but not installed. Please run 'pip install optuna'.")
        if combiner_name not in COMBINERS:
            raise ValueError(f"Unknown combiner: {combiner_name}")

        self.combiner_name = combiner_name
        self.combiner_class = COMBINERS[combiner_name].__class__
        self.param_ranges = getattr(self.combiner_class, 'PARAM_RANGES', {})
        
        # CRITICAL: Limit test data size for speed
        max_items = min(200, len(test_data_with_cache))  # Cap at 200 items for speed
        self.test_data_with_cache = test_data_with_cache[:max_items]
        
        self.pos_thresh = 0.1
        self.neg_thresh = -0.1
        self.aggregate_model = AggregateSentimentModel()
        self.action_model = ActionSentimentModel()
        self.association_model = AssociationSentimentModel()
        self.belonging_model = BelongingSentimentModel()

        self._action_function = (
            self.action_model.calculate
            if getattr(self.action_model, 'actor_func', None) and getattr(self.action_model, 'target_func', None)
            else None
        )
        
        # Pre-cache base graphs to avoid repeated deepcopy
        self._cache_base_graphs()
        
        # Pre-cache base graphs to avoid repeated deepcopy
        self._cache_base_graphs()
        
    def _cache_base_graphs(self):
        """Pre-process and cache base graphs to avoid repeated deep copying"""
        for item in self.test_data_with_cache:
            cached_data = item['cached_data']
            base_graph = cached_data.get('graph') or cached_data.get('_graph')
            if base_graph is not None:
                # Store reference to avoid repeated deepcopy
                item['_base_graph_ref'] = base_graph

    def _suggest_param(self, trial, name: str, spec: Any) -> Any:
        if isinstance(spec, dict):
            param_type = spec.get('type', 'categorical')
            if param_type == 'float':
                return trial.suggest_float(
                    name,
                    spec.get('low', 0.0),
                    spec.get('high', 1.0),
                    step=spec.get('step'),
                    log=spec.get('log', False),
                )
            if param_type == 'int':
                return trial.suggest_int(
                    name,
                    spec.get('low', 0),
                    spec.get('high', 10),
                    step=spec.get('step', 1),
                    log=spec.get('log', False),
                )
            if param_type == 'bool':
                return trial.suggest_categorical(name, [True, False])
            if param_type == 'categorical':
                return trial.suggest_categorical(name, spec.get('choices', []))
            if param_type == 'discrete_float':
                low = spec.get('low', 0.0)
                high = spec.get('high', 1.0)
                q = spec.get('step', 0.05)
                return trial.suggest_float(name, low, high, step=q)
            if 'default' in spec:
                return spec['default']
            raise ValueError(f"Unsupported parameter spec for '{name}': {spec}")

        if isinstance(spec, (list, tuple)):
            return trial.suggest_categorical(name, list(spec))

        return spec

    def _objective(self, trial) -> float:
        params = {}
        for param_name, spec in self.param_ranges.items():
            params[param_name] = self._suggest_param(trial, param_name, spec)

        all_predictions = []
        all_ground_truth = []
        per_dataset_predictions: Dict[str, List[str]] = defaultdict(list)
        per_dataset_truth: Dict[str, List[str]] = defaultdict(list)

        # Process items in batches to reduce memory usage
        for item in self.test_data_with_cache:
            cached_data = item['cached_data']
            true_sentiments = item['gold_sentiments']
            dataset_name = item.get('dataset') or 'dataset'
            
            # Use cached base graph reference
            base_graph = item.get('_base_graph_ref')
            if base_graph is None:
                base_graph = cached_data.get('graph') or cached_data.get('_graph')
                if base_graph is None:
                    continue

            # Optimize: Use shallow copy when possible, deep copy only when necessary
            try:
                graph = deepcopy(base_graph)  # Still needed for combiner refresh
                graph.refresh_with_combiner(
                    self.combiner_name,
                    params,
                    self._action_function,
                    self.association_model.calculate,
                    self.belonging_model.calculate,
                )
            except (ValueError, Exception):
                continue

            entity_label_map = item['entity_label_map']

            # Process aspects in batch
            aspect_results = []
            for label, true_polarity in true_sentiments.items():
                entity_id = entity_label_map.get(label)
                if entity_id is None:
                    continue

                try:
                    final_score = graph.run_aggregate_sentiment_calculations(entity_id, self.aggregate_model.calculate)
                    if math.isnan(final_score) or math.isinf(final_score):
                        final_score = 0.0
                    
                    final_score = max(-1.0, min(1.0, final_score))
                    pred_label = _score_to_label(final_score, self.pos_thresh, self.neg_thresh)
                    
                    aspect_results.append((pred_label, true_polarity.lower()))
                except Exception:
                    continue
            
            # Batch append results
            for pred_label, true_polarity in aspect_results:
                all_predictions.append(pred_label)
                all_ground_truth.append(true_polarity)
                per_dataset_predictions[dataset_name].append(pred_label)
                per_dataset_truth[dataset_name].append(true_polarity)

        if not all_predictions:
            return 0.0

        overall_accuracy = accuracy_score(all_ground_truth, all_predictions)
        
        labels = sorted(list(set(all_ground_truth + all_predictions)))
        precision = precision_score(all_ground_truth, all_predictions, labels=labels, average='macro', zero_division=0)
        recall = recall_score(all_ground_truth, all_predictions, labels=labels, average='macro', zero_division=0)
        f1 = f1_score(all_ground_truth, all_predictions, labels=labels, average='macro', zero_division=0)
        
        dataset_accuracies: Dict[str, float] = {}
        dataset_precisions: Dict[str, float] = {}
        dataset_recalls: Dict[str, float] = {}
        dataset_f1s: Dict[str, float] = {}
        
        for name, preds in per_dataset_predictions.items():
            truth = per_dataset_truth[name]
            if truth:
                dataset_accuracies[name] = accuracy_score(truth, preds)
                dataset_labels = sorted(list(set(truth + preds)))
                dataset_precisions[name] = precision_score(truth, preds, labels=dataset_labels, average='macro', zero_division=0)
                dataset_recalls[name] = recall_score(truth, preds, labels=dataset_labels, average='macro', zero_division=0)
                dataset_f1s[name] = f1_score(truth, preds, labels=dataset_labels, average='macro', zero_division=0)
                
                trial.set_user_attr(f"accuracy/{name}", dataset_accuracies[name])
                trial.set_user_attr(f"precision/{name}", dataset_precisions[name])
                trial.set_user_attr(f"recall/{name}", dataset_recalls[name])
                trial.set_user_attr(f"f1/{name}", dataset_f1s[name])

        if dataset_accuracies:
            macro_accuracy = sum(dataset_accuracies.values()) / len(dataset_accuracies)
            macro_precision = sum(dataset_precisions.values()) / len(dataset_precisions)
            macro_recall = sum(dataset_recalls.values()) / len(dataset_recalls)
            macro_f1 = sum(dataset_f1s.values()) / len(dataset_f1s)
        else:
            macro_accuracy = overall_accuracy
            macro_precision = precision
            macro_recall = recall
            macro_f1 = f1

        trial.set_user_attr("accuracy/overall_weighted", overall_accuracy)
        trial.set_user_attr("accuracy/macro", macro_accuracy)
        trial.set_user_attr("precision/overall", precision)
        trial.set_user_attr("precision/macro", macro_precision)
        trial.set_user_attr("recall/overall", recall)
        trial.set_user_attr("recall/macro", macro_recall)
        trial.set_user_attr("f1/overall", f1)
        trial.set_user_attr("f1/macro", macro_f1)

        base_precision_weight = 0.5
        adaptive_precision_boost = max(0, (0.3 - macro_precision) * 0.8)
        final_precision_weight = min(0.7, base_precision_weight + adaptive_precision_boost)
        
        remaining_weight = 1.0 - final_precision_weight
        accuracy_weight = remaining_weight * 0.3
        f1_weight = remaining_weight * 0.7
        
        precision_penalty = max(0, (0.1 - macro_precision) * 3)
        
        objective_score = (
            accuracy_weight * macro_accuracy +
            final_precision_weight * macro_precision +
            f1_weight * macro_f1 -
            precision_penalty
        )
        
        if macro_precision > 0.4 and macro_recall > 0.75 and macro_accuracy > 0.65:
            objective_score += 0.08
        
        trial.set_user_attr("objective_score", objective_score)
        return objective_score

    def optimize(self, n_trials: int = 100) -> OptimizationResult:  # Reduced from 500 to 100
        start_time = time.time()
        
        sampler = None
        if TPESampler is not None:
            sampler = TPESampler(
                multivariate=True, 
                group=True, 
                seed=None,
                n_startup_trials=min(20, n_trials // 5),  # Reduced startup trials
                n_ei_candidates=24,  # Reduced from 64 to 24
                prior_weight=1.0,
                consider_prior=False,
                consider_magic_clip=False,
                consider_endpoints=False,
                warn_independent_sampling=False
            )
        
        study = optuna.create_study(
            direction="maximize", 
            sampler=sampler,
            pruner=None
        )
        
        study.optimize(
            self._objective, 
            n_trials=n_trials, 
            show_progress_bar=True,
            timeout=1800  # 30 minutes max
        )
        
        evaluation_time = time.time() - start_time
        df = study.trials_dataframe()

        return OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            study_trials_dataframe=df.to_dict(),
            evaluation_time=evaluation_time,
            best_trial_user_attrs=dict(study.best_trial.user_attrs),
        )

# ... (The rest of the file: run_optimization, get_best_combiner_config, etc. remains the same)
def run_optimization(
    dataset_loaders: List[Tuple[str, callable]],
    combiner_name: Optional[str] = None,
    n_trials: int = 100,  # Reduced from 500 to 100
    auto_fill_cache: bool = True,
):
    if not dataset_loaders:
        raise ValueError("No dataset loaders provided to run_optimization")

    dataset_names = [name for name, _ in dataset_loaders]
    dataset_label = "+".join(dataset_names)
    logger.info("Loading cached data for datasets: %s", dataset_label)

    cache = PipelineCache()
    test_data_with_cache = []
    coverage_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {'gold': 0, 'matched': 0})

    for dataset_name, loader in dataset_loaders:
        try:
            test_items = loader()
        except Exception as exc:
            logger.error("Failed to load dataset '%s': %s", dataset_name, exc)
            continue

        logger.info("Processing %d items from dataset '%s'", len(test_items), dataset_name)

        for item in test_items:
            text = getattr(item, 'text', None)
            if text is None and isinstance(item, dict):
                text = item.get('text')
            if not text:
                continue

            try:
                cached_data = cache.get_intermediate_results(text)
            except Exception:
                continue
                
            if not cached_data or '_graph' not in cached_data:
                continue

            gold_sentiments: Dict[str, str] = {}
            aspects = getattr(item, 'aspects', None)
            if aspects:
                for aspect in aspects:
                    term = getattr(aspect, 'term', None)
                    polarity = getattr(aspect, 'polarity', None)
                    if term is None and isinstance(aspect, dict):
                        term = aspect.get('term')
                        polarity = aspect.get('polarity')
                    if term and polarity:
                        norm_term = normalize_text(str(term))
                        if not norm_term:
                            continue
                        mapped_polarity = _normalize_polarity(str(polarity))
                        if mapped_polarity is None:
                            continue
                        gold_sentiments[norm_term] = mapped_polarity
            else:
                sentiments_dict = getattr(item, 'sentiments', None)
                if sentiments_dict is None and isinstance(item, dict):
                    sentiments_dict = item.get('sentiments')
                if isinstance(sentiments_dict, dict):
                    for term, value in sentiments_dict.items():
                        norm_term = normalize_text(str(term))
                        if not norm_term:
                            continue
                        if value > 0:
                            gold_sentiments[norm_term] = 'positive'
                        elif value < 0:
                            gold_sentiments[norm_term] = 'negative'
                        else:
                            gold_sentiments[norm_term] = 'neutral'

            if not gold_sentiments:
                continue

            entity_label_map: Dict[str, int] = {}
            entity_records = cached_data.get('entity_records') or {}
            for raw_entity_id, record in entity_records.items():
                label = record.get('label') if isinstance(record, dict) else None
                if not label:
                    continue
                try:
                    entity_id_int = int(raw_entity_id)
                except (TypeError, ValueError):
                    continue
                norm_label = normalize_text(str(label))
                if not norm_label:
                    continue
                if norm_label not in entity_label_map:
                    entity_label_map[norm_label] = entity_id_int

            coverage_stats[dataset_name]['gold'] += len(gold_sentiments)
            matched = sum(1 for label in gold_sentiments if label in entity_label_map)
            coverage_stats[dataset_name]['matched'] += matched

            test_data_with_cache.append({
                'gold_sentiments': gold_sentiments,
                'cached_data': cached_data,
                'entity_label_map': entity_label_map,
                'dataset': dataset_name,
            })

    if not test_data_with_cache:
        raise RuntimeError("No valid cached data found")

    combiners_to_optimize = [
        combiner_name
    ] if combiner_name else [
        name
        for name, inst in COMBINERS.items()
        if getattr(inst, 'PARAM_RANGES', None)
    ]

    for dataset_name, stats in coverage_stats.items():
        gold = stats['gold'] or 1
        logger.info(
            "Dataset '%s': matched %.1f%% of gold aspects (%d/%d).",
            dataset_name,
            100.0 * stats['matched'] / gold,
            stats['matched'],
            gold,
        )

    results: Dict[str, OptimizationResult] = {}
    for name in combiners_to_optimize:
        logger.info("\n--- Optimizing %s (datasets=%s) ---", name, dataset_label)
        optimizer = CombinerOptimizer(name, test_data_with_cache)
        result = optimizer.optimize(n_trials=n_trials)
        results[name] = result

    best_combiner, best_params = get_best_combiner_config(results)
    if best_combiner:
        best_result = results[best_combiner]
        save_optimized_combiner_config(
            best_combiner,
            best_params,
            dataset_name=dataset_label,
            score=best_result.best_score,
        )
        logger.info("\n--- Optimization Complete (datasets=%s) ---", dataset_label)
        logger.info(
            "Best overall combiner is '%s' with macro accuracy %.4f",
            best_combiner,
            best_result.best_score,
        )
        logger.info(
            "Optimal parameters saved to: %s",
            config.CACHE_DIR / "best_combiner_config.json",
        )

    return results

def get_best_combiner_config(optimization_results: Dict[str, OptimizationResult]) -> Tuple[Optional[str], Dict[str, Any]]:
    best_combiner, best_score, best_params = None, -1.0, {}
    for combiner_name, result in optimization_results.items():
        if result.best_score > best_score:
            best_score = result.best_score
            best_combiner = combiner_name
            best_params = result.best_params
    return best_combiner, best_params

def save_optimized_combiner_config(
    combiner_name: str,
    params: Dict[str, Any],
    dataset_name: Optional[str] = None,
    score: Optional[float] = None,
) -> None:
    config_file = config.CACHE_DIR / "best_combiner_config.json"
    config_file.parent.mkdir(exist_ok=True)
    try:
        payload = {'combiner_name': combiner_name, 'params': params}
        if score is not None:
            payload['score'] = score
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

        if dataset_name:
            dataset_safe = dataset_name.replace('/', '_').replace('+', '_')
            dataset_file = config.CACHE_DIR / f"best_combiner_config_{dataset_safe}.json"
            dataset_payload = dict(payload)
            dataset_payload['dataset'] = dataset_name
            with open(dataset_file, 'w', encoding='utf-8') as df:
                json.dump(dataset_payload, df, indent=2)
    except IOError as e:
        logger.error(f"Failed to save optimized combiner config: {e}")
