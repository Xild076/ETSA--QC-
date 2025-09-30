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
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    TPESampler = None
    OPTUNA_AVAILABLE = False
    
from sklearn.metrics import accuracy_score

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
    """Converts a raw sentiment score to a polarity label based on thresholds."""
    if score >= pos_thresh:
        return "positive"
    if score <= neg_thresh:
        return "negative"
    return "neutral"


def _normalize_polarity(polarity: str) -> Optional[str]:
    pol = polarity.lower()
    if pol in {"positive", "pos", "favorable"}:
        return "positive"
    if pol in {"negative", "neg", "unfavorable"}:
        return "negative"
    if pol in {"neutral", "neu", "objective"}:
        return "neutral"
    if pol in {"conflict", "both"}:
        return "neutral"
    return None

class CombinerOptimizer:
    """Optimizes sentiment combiner parameters using Optuna for intelligent search."""

    def __init__(self, combiner_name: str, test_data_with_cache: List[Dict[str, Any]]):
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna is required but not installed. Please run 'pip install optuna'.")
        if combiner_name not in COMBINERS:
            raise ValueError(f"Unknown combiner: {combiner_name}")

        self.combiner_name = combiner_name
        self.combiner_class = COMBINERS[combiner_name].__class__
        self.param_ranges = getattr(self.combiner_class, 'PARAM_RANGES', {})
        self.test_data_with_cache = test_data_with_cache
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

    def _suggest_param(self, trial, name: str, spec: Any) -> Any:
        """Suggest a parameter value based on the provided spec."""
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
            # Unknown spec; fall back to default value if provided
            if 'default' in spec:
                return spec['default']
            raise ValueError(f"Unsupported parameter spec for '{name}': {spec}")

        if isinstance(spec, (list, tuple)):
            return trial.suggest_categorical(name, list(spec))

        # Constant parameter value
        return spec

    def _objective(self, trial) -> float:
        """The objective function that Optuna will try to maximize."""
        params = {}
        for param_name, spec in self.param_ranges.items():
            params[param_name] = self._suggest_param(trial, param_name, spec)

        all_predictions = []
        all_ground_truth = []
        per_dataset_predictions: Dict[str, List[str]] = defaultdict(list)
        per_dataset_truth: Dict[str, List[str]] = defaultdict(list)

        for item in self.test_data_with_cache:
            cached_data = item['cached_data']
            true_sentiments = item['gold_sentiments']
            dataset_name = item.get('dataset') or 'dataset'
            base_graph = cached_data.get('graph') or cached_data.get('_graph')
            if base_graph is None:
                continue

            graph = deepcopy(base_graph)
            try:
                graph.refresh_with_combiner(
                    self.combiner_name,
                    params,
                    self._action_function,
                    self.association_model.calculate,
                    self.belonging_model.calculate,
                )
            except ValueError:
                continue

            entity_label_map = item['entity_label_map']

            for label, true_polarity in true_sentiments.items():
                entity_id = entity_label_map.get(label)
                if entity_id is None:
                    continue

                final_score = graph.run_aggregate_sentiment_calculations(entity_id, self.aggregate_model.calculate)
                if math.isnan(final_score):
                    final_score = 0.0
                pred_label = _score_to_label(final_score, self.pos_thresh, self.neg_thresh)

                all_predictions.append(pred_label)
                all_ground_truth.append(true_polarity.lower())
                per_dataset_predictions[dataset_name].append(pred_label)
                per_dataset_truth[dataset_name].append(true_polarity.lower())

        if not all_predictions:
            return 0.0

        overall_accuracy = accuracy_score(all_ground_truth, all_predictions)
        dataset_accuracies: Dict[str, float] = {}
        for name, preds in per_dataset_predictions.items():
            truth = per_dataset_truth[name]
            if truth:
                dataset_accuracies[name] = accuracy_score(truth, preds)
                trial.set_user_attr(f"accuracy/{name}", dataset_accuracies[name])

        if dataset_accuracies:
            macro_accuracy = sum(dataset_accuracies.values()) / len(dataset_accuracies)
        else:
            macro_accuracy = overall_accuracy

        trial.set_user_attr("accuracy/overall_weighted", overall_accuracy)
        trial.set_user_attr("accuracy/macro", macro_accuracy)

        return macro_accuracy

    def optimize(self, n_trials: int = 200) -> OptimizationResult:
        """Run the Optuna optimization study."""
        start_time = time.time()
        
        sampler = None
        if TPESampler is not None:
            sampler = TPESampler(multivariate=True, group=True, seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(self._objective, n_trials=n_trials, show_progress_bar=True)
        
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
    n_trials: int = 200,
    auto_fill_cache: bool = True,
):
    """Run the optimization process across one or more datasets."""
    if not dataset_loaders:
        raise ValueError("No dataset loaders provided to run_optimization")

    dataset_names = [name for name, _ in dataset_loaders]
    dataset_label = "+".join(dataset_names)
    logger.info("Loading and preparing cached data for datasets: %s", dataset_label)

    cache = PipelineCache()
    pipeline = None
    test_data_with_cache = []
    coverage_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {'gold': 0, 'matched': 0})

    for dataset_name, loader in dataset_loaders:
        try:
            test_items = loader()
        except Exception as exc:
            logger.error("Failed to load dataset '%s': %s", dataset_name, exc)
            continue

        logger.info("Preparing %d items from dataset '%s'", len(test_items), dataset_name)

        for item in test_items:
            text = getattr(item, 'text', None)
            if text is None and isinstance(item, dict):
                text = item.get('text')
            if not text:
                continue

            cached_data = cache.get_intermediate_results(text)
            if (not cached_data or '_graph' not in cached_data) and auto_fill_cache:
                if pipeline is None:
                    from pipeline import build_default_pipeline
                    pipeline = build_default_pipeline()
                try:
                    pipeline.process(text)
                    cached_data = cache.get_intermediate_results(text)
                except Exception as exc:
                    logger.debug(
                        "Auto cache fill failed for dataset '%s' text '%s': %s",
                        dataset_name,
                        text[:50],
                        exc,
                    )
                    cached_data = None

            if not cached_data or '_graph' not in cached_data:
                logger.debug(
                    "Skipping text '%s' (dataset '%s') due to missing cache.",
                    text[:50],
                    dataset_name,
                )
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
            for raw_entity_id, record in (cached_data.get('entity_records') or {}).items():
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

            test_data_with_cache.append(
                {
                    'gold_sentiments': gold_sentiments,
                    'cached_data': cached_data,
                    'entity_label_map': entity_label_map,
                    'dataset': dataset_name,
                }
            )

    if not test_data_with_cache:
        raise RuntimeError("No valid cached data found. Run 'python main.py cache <dataset>' first.")

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
