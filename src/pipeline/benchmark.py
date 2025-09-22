from __future__ import annotations

import csv
import json
import logging
import math
import re
import xml.etree.ElementTree as ET
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from networkx.readwrite import json_graph
from tqdm import tqdm

try:
    from .pipeline import SentimentPipeline, build_default_pipeline
except Exception:
    from pipeline import SentimentPipeline, build_default_pipeline


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "dataset"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "benchmarks"

if TYPE_CHECKING:
    from .graph import RelationGraph

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "is",
    "are",
    "be",
    "was",
    "were",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "my",
    "your",
    "our",
    "their",
    "very",
    "too",
    "so",
    "just",
    "but",
}

SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^a-z0-9\s]")


@dataclass
class AspectAnnotation:
    term: str
    polarity: str
    start: int
    end: int


@dataclass
class DatasetItem:
    sentence_id: str
    text: str
    aspects: List[AspectAnnotation]


@dataclass
class PredictedAspect:
    entity_id: int
    canonical: str
    polarity: str
    score: float
    mentions: List[Dict[str, Any]]
    norm: str
    tokens: Tuple[str, ...]
    head: str


@dataclass
class GoldAspect:
    annotation: AspectAnnotation
    norm: str
    tokens: Tuple[str, ...]
    head: str


def _safe_filename(*segments: str) -> str:
    names: List[str] = []
    for segment in segments:
        cleaned = re.sub(r"[^0-9a-zA-Z_-]+", "_", segment or "").strip("_")
        if cleaned:
            names.append(cleaned)
    return "_".join(names) if names else "report"


def _graph_snapshot(graph: Optional["RelationGraph"]) -> Dict[str, Any]:
    if graph is None:
        return {}
    try:
        data = json_graph.node_link_data(graph.graph)
    except Exception:
        return {}
    data["clauses"] = list(getattr(graph, "clauses", []))
    data["text"] = getattr(graph, "text", "")
    data["aggregate_sentiments"] = dict(getattr(graph, "aggregate_sentiments", {}))
    return data


def _persist_graph_snapshot(snapshot: Dict[str, Any], directory: Path, sequence: int, sentence_id: str, issue_type: str) -> Optional[Path]:
    if not snapshot:
        return None
    filename = f"{sequence:05d}_{_safe_filename(sentence_id, issue_type)}.json"
    path = directory / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2)
    return path


def _module_hypothesis(issue_type: str) -> List[str]:
    mapping = {
        "missing_aspect": ["aspect_extractor", "relation_extractor"],
        "spurious_aspect": ["relation_extractor", "modifier_extractor"],
        "wrong_polarity": ["sentiment_analysis", "aggregate_sentiment_model"],
        "pipeline_exception": ["pipeline"],
    }
    return mapping.get(issue_type, ["unknown"])


def _predicted_to_dict(pred: PredictedAspect) -> Dict[str, Any]:
    return {
        "entity_id": pred.entity_id,
        "canonical": pred.canonical,
        "polarity": pred.polarity,
        "score": pred.score,
        "mentions": pred.mentions,
        "norm": pred.norm,
        "tokens": list(pred.tokens),
        "head": pred.head,
    }


def _gold_to_dict(gold: GoldAspect) -> Dict[str, Any]:
    return {
        "term": gold.annotation.term,
        "polarity": gold.annotation.polarity,
        "start": gold.annotation.start,
        "end": gold.annotation.end,
        "norm": gold.norm,
        "tokens": list(gold.tokens),
        "head": gold.head,
    }


def _closest_pred_for_gold(gold: GoldAspect, predicted: Sequence[PredictedAspect]) -> Tuple[Optional[PredictedAspect], float]:
    best: Optional[PredictedAspect] = None
    best_score = 0.0
    for pred in predicted:
        score = _lenient_similarity(pred, gold)
        if score > best_score:
            best_score = score
            best = pred
    return best, best_score


def _closest_gold_for_pred(pred: PredictedAspect, golds: Sequence[GoldAspect]) -> Tuple[Optional[GoldAspect], float]:
    best: Optional[GoldAspect] = None
    best_score = 0.0
    for gold in golds:
        score = _lenient_similarity(pred, gold)
        if score > best_score:
            best_score = score
            best = gold
    return best, best_score


_PIPELINE_CACHE: Dict[str, SentimentPipeline] = {}


def get_dataset_path(dataset_name: str) -> str:
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    filename, _ = DATASET_REGISTRY[dataset_name]
    dataset_path = DATA_DIR / filename
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    return str(dataset_path)


def get_dataset_loader(dataset_name: str) -> Callable[[Path], List[DatasetItem]]:
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    _, loader = DATASET_REGISTRY[dataset_name]
    return loader


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    cleaned = PUNCT_RE.sub(" ", lowered)
    collapsed = SPACE_RE.sub(" ", cleaned).strip()
    return collapsed


def _tokens(text: str) -> Tuple[str, ...]:
    norm = _normalize_text(text)
    tokens = [tok for tok in norm.split(" ") if tok and tok not in STOPWORDS]
    return tuple(tokens)


def _head(tokens: Sequence[str], fallback: str) -> str:
    if tokens:
        return tokens[-1]
    norm = _normalize_text(fallback)
    return norm.split(" ")[-1] if norm else ""


def _lenient_similarity(pred: PredictedAspect, gold: GoldAspect) -> float:
    if not pred.norm or not gold.norm:
        return 0.0
    if pred.norm == gold.norm:
        return 1.0
    score = 0.0
    if pred.norm in gold.norm or gold.norm in pred.norm:
        score = max(score, 0.92)
    if pred.head and pred.head == gold.head:
        score = max(score, 0.88)
    common = set(pred.tokens) & set(gold.tokens)
    if common:
        coverage = len(common) / max(len(pred.tokens), len(gold.tokens), 1)
        score = max(score, 0.6 + 0.4 * coverage)
    seq_ratio = SequenceMatcher(None, pred.norm, gold.norm).ratio()
    score = max(score, seq_ratio)
    return score


def _match_aspects(
    predicted: List[PredictedAspect],
    golds: List[GoldAspect],
    min_score: float = 0.58,
) -> Tuple[List[Tuple[GoldAspect, PredictedAspect, float]], List[GoldAspect], List[PredictedAspect]]:
    if not predicted or not golds:
        return [], list(golds), list(predicted)
    scores: List[Tuple[int, int, float]] = []
    for gi, g in enumerate(golds):
        for pi, p in enumerate(predicted):
            sim = _lenient_similarity(p, g)
            if sim >= min_score:
                scores.append((gi, pi, sim))
    used_gold = set()
    used_pred = set()
    matches: List[Tuple[GoldAspect, PredictedAspect, float]] = []
    for gi, pi, sim in sorted(scores, key=lambda x: x[2], reverse=True):
        if gi in used_gold or pi in used_pred:
            continue
        used_gold.add(gi)
        used_pred.add(pi)
        matches.append((golds[gi], predicted[pi], sim))
    unmatched_gold = [g for idx, g in enumerate(golds) if idx not in used_gold]
    unmatched_pred = [p for idx, p in enumerate(predicted) if idx not in used_pred]
    return matches, unmatched_gold, unmatched_pred


def _load_semeval_xml(path: Path) -> List[DatasetItem]:
    tree = ET.parse(path)
    root = tree.getroot()
    dataset: List[DatasetItem] = []
    for sentence_el in root.iterfind("sentence"):
        sid = sentence_el.get("id", "")
        text_el = sentence_el.find("text")
        text = text_el.text.strip() if text_el is not None and text_el.text else ""
        aspects: List[AspectAnnotation] = []
        aspects_el = sentence_el.find("aspectTerms")
        if aspects_el is not None:
            for term_el in aspects_el.findall("aspectTerm"):
                term = term_el.get("term", "").strip()
                if not term or term.lower() == "null":
                    continue
                polarity = term_el.get("polarity", "neutral").lower()
                start = int(term_el.get("from", 0))
                end = int(term_el.get("to", 0))
                aspects.append(AspectAnnotation(term=term, polarity=polarity, start=start, end=end))
        dataset.append(DatasetItem(sentence_id=sid, text=text, aspects=aspects))
    return dataset

DATASET_REGISTRY: Dict[str, Tuple[str, Callable[[Path], List[DatasetItem]]]] = {
    "test_laptop_2014": ("test_laptop_2014.xml", _load_semeval_xml),
    "test_restaurant_2014": ("test_restaurant_2014.xml", _load_semeval_xml),
}


def _prepare_gold(aspects: Iterable[AspectAnnotation]) -> List[GoldAspect]:
    prepared: List[GoldAspect] = []
    for asp in aspects:
        tokens = _tokens(asp.term)
        prepared.append(
            GoldAspect(
                annotation=asp,
                norm=_normalize_text(asp.term),
                tokens=tokens,
                head=_head(tokens, asp.term),
            )
        )
    return prepared


def _prepare_predicted(
    aggregate_results: Dict[Any, Dict[str, Any]],
    pos_thresh: float,
    neg_thresh: float,
) -> List[PredictedAspect]:
    predicted: List[PredictedAspect] = []
    for entity_id, data in aggregate_results.items():
        mentions = data.get("mentions") or []
        canonical = ""
        if mentions:
            canonical = mentions[0].get("text", "")
        if not canonical:
            canonical = f"entity_{entity_id}"
        score = float(data.get("aggregate_sentiment", 0.0) or 0.0)
        if math.isnan(score):
            score = 0.0
        if score >= pos_thresh:
            polarity = "positive"
        elif score <= neg_thresh:
            polarity = "negative"
        else:
            polarity = "neutral"
        tokens = _tokens(canonical)
        predicted.append(
            PredictedAspect(
                entity_id=int(entity_id),
                canonical=canonical,
                polarity=polarity,
                score=score,
                mentions=mentions,
                norm=_normalize_text(canonical),
                tokens=tokens,
                head=_head(tokens, canonical),
            )
        )
    return predicted


def _normalize_gold_polarity(polarity: str) -> str:
    if not polarity:
        return "neutral"
    value = polarity.lower().strip()
    if value not in {"positive", "negative", "neutral", "conflict"}:
        return "neutral"
    return value


def _classification_labels(gold_labels: List[str], pred_labels: List[str]) -> List[str]:
    label_set = sorted(set(gold_labels) | set(pred_labels))
    return label_set or ["neutral"]


def _serialize_report(report: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    serializable: Dict[str, Dict[str, Any]] = {}
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            entry: Dict[str, Any] = {}
            for key, value in metrics.items():
                if key == "support":
                    entry[key] = int(value)
                else:
                    entry[key] = float(value)
            serializable[label] = entry
        else:
            serializable[label] = float(metrics)
    return serializable


def _ensure_output_dir(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)


def _plot_confusion_matrix(cm: List[List[int]], labels: List[str], path: Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.ylabel("Gold")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _build_pipeline_for_mode(run_mode: str) -> SentimentPipeline:
    if run_mode == "full_stack":
        return build_default_pipeline()
    logger.warning("Run mode '%s' not specialized; falling back to default pipeline.", run_mode)
    return build_default_pipeline()


def _get_pipeline(run_mode: str) -> SentimentPipeline:
    pipeline = _PIPELINE_CACHE.get(run_mode)
    if pipeline is None:
        pipeline = _build_pipeline_for_mode(run_mode)
        _PIPELINE_CACHE[run_mode] = pipeline
    return pipeline


def run_benchmark(
    run_name: str,
    dataset_name: str,
    run_mode: str = "full_stack",
    limit: Optional[int] = None,
    pos_thresh: float = 0.1,
    neg_thresh: float = -0.1,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    dataset_path = Path(get_dataset_path(dataset_name))
    loader = get_dataset_loader(dataset_name)
    items = loader(dataset_path)
    if limit and limit > 0:
        items = items[:limit]
    total_items = len(items)
    logger.info("Starting benchmark: dataset=%s, run_mode=%s, samples=%d", dataset_name, run_mode, total_items)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_run_name = f"{run_name}_{timestamp}"
    run_dir = OUTPUT_ROOT / final_run_name
    _ensure_output_dir(run_dir)
    graph_reports_dir = run_dir / "graph_reports"
    graph_reports_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "benchmark.log"
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    pipeline = _get_pipeline(run_mode)

    gold_labels: List[str] = []
    pred_labels: List[str] = []
    error_details: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []
    match_rows: List[Dict[str, Any]] = []
    sentence_records: List[Dict[str, Any]] = []
    error_summary: Counter[str] = Counter()
    sentence_precision_values: List[float] = []
    sentence_recall_values: List[float] = []
    sentence_f1_values: List[float] = []

    stats = defaultdict(int)
    stats["requested_sentences"] = total_items
    stats["sentence_errors"] = 0
    graph_sequence = 1

    def _register_error(
        issue_type: str,
        sentence_item: DatasetItem,
        snapshot: Dict[str, Any],
        gold_entry: Optional[GoldAspect],
        pred_entry: Optional[PredictedAspect],
        match_score: Optional[float],
        analysis: Dict[str, Any],
    ) -> None:
        nonlocal graph_sequence
        path = _persist_graph_snapshot(snapshot, graph_reports_dir, graph_sequence, sentence_item.sentence_id, issue_type)
        if path is not None:
            graph_sequence += 1
            graph_reference = str(path.relative_to(PROJECT_ROOT))
        else:
            graph_reference = None
        modules = _module_hypothesis(issue_type)
        detail = {
            "sentence_id": sentence_item.sentence_id,
            "text": sentence_item.text,
            "issue_type": issue_type,
            "probable_modules": modules,
            "graph_report": graph_reference,
            "gold": _gold_to_dict(gold_entry) if gold_entry else None,
            "predicted": _predicted_to_dict(pred_entry) if pred_entry else None,
            "match_score": match_score,
            "analysis": analysis,
        }
        error_details.append(detail)
        error_summary[issue_type] += 1
        csv_row = {
            "issue_type": issue_type,
            "probable_modules": ";".join(modules),
            "graph_report": graph_reference or "",
            "id": sentence_item.sentence_id,
            "text": sentence_item.text,
            "gold_aspect": gold_entry.annotation.term if gold_entry else analysis.get("closest_gold_term", ""),
            "gold_polarity": _normalize_gold_polarity(gold_entry.annotation.polarity) if gold_entry else analysis.get("closest_gold_polarity", ""),
            "pred_aspect": pred_entry.canonical if pred_entry else analysis.get("closest_pred_aspect", ""),
            "pred_polarity": pred_entry.polarity if pred_entry else analysis.get("closest_pred_polarity", ""),
            "pred_score": pred_entry.score if pred_entry else analysis.get("closest_pred_score", ""),
            "match_score": match_score if match_score is not None else analysis.get("similarity_score", ""),
            "analysis": json.dumps(analysis, sort_keys=True),
        }
        error_rows.append(csv_row)
        if issue_type == "pipeline_exception":
            logger.error("Pipeline exception on sentence %s", sentence_item.sentence_id)
        else:
            logger.warning("Detected %s on sentence %s", issue_type, sentence_item.sentence_id)

    try:
        with tqdm(items, total=total_items, desc="Benchmark", unit="sentence", disable=total_items == 0) as progress:
            for enumerated_index, item in enumerate(progress, start=1):
                zero_based_index = enumerated_index - 1
                has_error = False
                sentence_issue_types: List[str] = []
                graph_snapshot: Dict[str, Any] = {}
                try:
                    result = pipeline.process(item.text, debug=False)
                except Exception as exc:
                    stats["processing_errors"] += 1
                    stats["sentence_errors"] += 1
                    logger.exception("Failed to process sentence %s", item.sentence_id)
                    analysis = {
                        "exception": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                    _register_error("pipeline_exception", item, {}, None, None, None, analysis)
                    sentence_records.append(
                        {
                            "sentence_id": item.sentence_id,
                            "text": item.text,
                            "gold_count": 0,
                            "pred_count": 0,
                            "matched_count": 0,
                            "issue_types": ["pipeline_exception"],
                            "precision": 0.0,
                            "recall": 0.0,
                            "f1": 0.0,
                        }
                    )
                    sentence_precision_values.append(0.0)
                    sentence_recall_values.append(0.0)
                    sentence_f1_values.append(0.0)
                    if progress_callback:
                        try:
                            progress_callback(zero_based_index, total_items)
                        except Exception:
                            logger.debug("Progress callback raised an exception", exc_info=True)
                    progress.set_postfix(errors=stats["sentence_errors"], matched=stats["matched_aspects"])
                    continue

                aggregate_results = result.get("aggregate_results") or {}
                predicted_aspects = _prepare_predicted(aggregate_results, pos_thresh, neg_thresh)
                gold_aspects = _prepare_gold(item.aspects)
                graph_snapshot = _graph_snapshot(result.get("graph"))

                matches, unmatched_gold, unmatched_pred = _match_aspects(predicted_aspects, gold_aspects)
                stats["total_sentences"] += 1
                stats["gold_aspects"] += len(gold_aspects)
                stats["pred_aspects"] += len(predicted_aspects)
                stats["matched_aspects"] += len(matches)
                stats["unmatched_gold"] += len(unmatched_gold)
                stats["unmatched_pred"] += len(unmatched_pred)

                gold_count = len(gold_aspects)
                pred_count = len(predicted_aspects)
                match_count = len(matches)
                if gold_count == 0 and pred_count == 0:
                    sentence_precision = 1.0
                    sentence_recall = 1.0
                    sentence_f1 = 1.0
                else:
                    sentence_precision = match_count / pred_count if pred_count else 0.0
                    sentence_recall = match_count / gold_count if gold_count else 0.0
                    sentence_f1 = (2 * sentence_precision * sentence_recall / (sentence_precision + sentence_recall)) if (sentence_precision + sentence_recall) else 0.0
                sentence_precision_values.append(sentence_precision)
                sentence_recall_values.append(sentence_recall)
                sentence_f1_values.append(sentence_f1)

                for gold_entry, pred_entry, sim in matches:
                    gold_pol = _normalize_gold_polarity(gold_entry.annotation.polarity)
                    pred_pol = pred_entry.polarity
                    gold_labels.append(gold_pol)
                    pred_labels.append(pred_pol)
                    record = {
                        "issue_type": "correct",
                        "id": item.sentence_id,
                        "text": item.text,
                        "gold_aspect": gold_entry.annotation.term,
                        "gold_polarity": gold_pol,
                        "pred_aspect": pred_entry.canonical,
                        "pred_polarity": pred_pol,
                        "pred_score": pred_entry.score,
                        "match_score": sim,
                    }
                    if gold_pol != pred_pol:
                        has_error = True
                        sentence_issue_types.append("wrong_polarity")
                        record["issue_type"] = "wrong_polarity"
                        analysis = {
                            "predicted_score": pred_entry.score,
                            "positive_threshold": pos_thresh,
                            "negative_threshold": neg_thresh,
                            "gold_polarity": gold_pol,
                            "predicted_polarity": pred_pol,
                        }
                        _register_error("wrong_polarity", item, graph_snapshot, gold_entry, pred_entry, sim, analysis)
                    match_rows.append(record)

                for gold_entry in unmatched_gold:
                    has_error = True
                    sentence_issue_types.append("missing_aspect")
                    closest_pred, closest_score = _closest_pred_for_gold(gold_entry, predicted_aspects)
                    analysis = {
                        "detail": "gold aspect not matched by predictions",
                        "similarity_score": closest_score,
                        "closest_pred": _predicted_to_dict(closest_pred) if closest_pred else None,
                        "closest_pred_aspect": closest_pred.canonical if closest_pred else "",
                        "closest_pred_polarity": closest_pred.polarity if closest_pred else "",
                        "closest_pred_score": closest_pred.score if closest_pred else "",
                        "closest_gold_term": gold_entry.annotation.term,
                        "closest_gold_polarity": gold_entry.annotation.polarity,
                    }
                    _register_error("missing_aspect", item, graph_snapshot, gold_entry, closest_pred, None, analysis)

                for pred_entry in unmatched_pred:
                    has_error = True
                    sentence_issue_types.append("spurious_aspect")
                    closest_gold, closest_score = _closest_gold_for_pred(pred_entry, gold_aspects)
                    analysis = {
                        "detail": "predicted aspect without gold match",
                        "similarity_score": closest_score,
                        "closest_gold": _gold_to_dict(closest_gold) if closest_gold else None,
                        "closest_gold_term": closest_gold.annotation.term if closest_gold else "",
                        "closest_gold_polarity": closest_gold.annotation.polarity if closest_gold else "",
                        "closest_pred_aspect": pred_entry.canonical,
                        "closest_pred_polarity": pred_entry.polarity,
                        "closest_pred_score": pred_entry.score,
                    }
                    _register_error("spurious_aspect", item, graph_snapshot, closest_gold, pred_entry, None, analysis)

                sentence_records.append(
                    {
                        "sentence_id": item.sentence_id,
                        "text": item.text,
                        "gold_count": gold_count,
                        "pred_count": pred_count,
                        "matched_count": match_count,
                        "issue_types": sorted(set(sentence_issue_types)),
                        "precision": sentence_precision,
                        "recall": sentence_recall,
                        "f1": sentence_f1,
                    }
                )

                if has_error:
                    stats["sentence_errors"] += 1

                if progress_callback:
                    try:
                        progress_callback(zero_based_index, total_items)
                    except Exception:
                        logger.debug("Progress callback raised an exception", exc_info=True)

                progress.set_postfix(errors=stats["sentence_errors"], matched=stats["matched_aspects"])
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()

    labels = _classification_labels(gold_labels, pred_labels)
    cm = confusion_matrix(gold_labels, pred_labels, labels=labels) if gold_labels else []
    cm_list = cm.tolist() if hasattr(cm, "tolist") else []

    accuracy = accuracy_score(gold_labels, pred_labels) if gold_labels else 0.0
    try:
        balanced_acc = balanced_accuracy_score(gold_labels, pred_labels) if gold_labels else 0.0
    except ValueError:
        balanced_acc = 0.0

    if gold_labels:
        report_dict = classification_report(
            gold_labels,
            pred_labels,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        report = _serialize_report(report_dict)
    else:
        report = {}

    precision = stats["matched_aspects"] / stats["pred_aspects"] if stats["pred_aspects"] else 0.0
    recall = stats["matched_aspects"] / stats["gold_aspects"] if stats["gold_aspects"] else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    avg_sentence_precision = sum(sentence_precision_values) / len(sentence_precision_values) if sentence_precision_values else 0.0
    avg_sentence_recall = sum(sentence_recall_values) / len(sentence_recall_values) if sentence_recall_values else 0.0
    avg_sentence_f1 = sum(sentence_f1_values) / len(sentence_f1_values) if sentence_f1_values else 0.0
    sentence_error_rate = stats["sentence_errors"] / total_items if total_items else 0.0
    total_errors = sum(error_summary.values())

    metrics = {
        "run_name": final_run_name,
        "dataset": dataset_name,
        "mode": run_mode,
        "pos_threshold": pos_thresh,
        "neg_threshold": neg_thresh,
        "total_sentences": total_items,
        "processed_sentences": stats["total_sentences"],
        "processing_errors": stats["processing_errors"],
        "sentence_errors": stats["sentence_errors"],
        "sentence_error_rate": sentence_error_rate,
        "gold_aspects": stats["gold_aspects"],
        "pred_aspects": stats["pred_aspects"],
        "matched_aspects": stats["matched_aspects"],
        "unmatched_gold": stats["unmatched_gold"],
        "unmatched_pred": stats["unmatched_pred"],
        "aspect_precision": precision,
        "aspect_recall": recall,
        "aspect_f1": f1,
        "avg_sentence_precision": avg_sentence_precision,
        "avg_sentence_recall": avg_sentence_recall,
        "avg_sentence_f1": avg_sentence_f1,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "classification_report": report,
        "error_summary": dict(error_summary),
        "error_count": total_errors,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm_list,
        },
        "graph_reports_dir": str(graph_reports_dir.relative_to(PROJECT_ROOT)),
    }

    metrics_path = run_dir / "metrics.json"
    error_details_path = run_dir / "error_details.json"
    sentence_summary_path = run_dir / "sentence_summary.json"
    errors_path = run_dir / "error_analysis.csv"
    matches_path = run_dir / "matches.json"

    metrics.update(
        {
            "metrics_path": str(metrics_path.relative_to(PROJECT_ROOT)),
            "error_details_path": str(error_details_path.relative_to(PROJECT_ROOT)),
            "sentence_summary_path": str(sentence_summary_path.relative_to(PROJECT_ROOT)),
            "error_analysis_csv_path": str(errors_path.relative_to(PROJECT_ROOT)),
            "matches_path": str(matches_path.relative_to(PROJECT_ROOT)),
        }
    )

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    if cm_list:
        cm_path = run_dir / "confusion_matrix.png"
        _plot_confusion_matrix(cm_list, labels, cm_path)

    with error_details_path.open("w", encoding="utf-8") as handle:
        json.dump(error_details, handle, indent=2)

    with sentence_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(sentence_records, handle, indent=2)

    fieldnames = [
        "issue_type",
        "probable_modules",
        "graph_report",
        "id",
        "text",
        "gold_aspect",
        "gold_polarity",
        "pred_aspect",
        "pred_polarity",
        "pred_score",
        "match_score",
        "analysis",
    ]
    with errors_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        if error_rows:
            writer.writerows(error_rows)

    with matches_path.open("w", encoding="utf-8") as handle:
        json.dump(match_rows, handle, indent=2)

    logger.info(
        "Benchmark complete: matched=%d precision=%.3f recall=%.3f accuracy=%.3f errors=%d sentences_with_errors=%d",
        stats["matched_aspects"],
        precision,
        recall,
        accuracy,
        total_errors,
        stats["sentence_errors"],
    )

    return metrics

metrics = run_benchmark(
    "test_laptop_2014",
    "test_laptop_2014",
    "full_stack",
    None,
    0.1,
    -0.1
)

print(metrics)
