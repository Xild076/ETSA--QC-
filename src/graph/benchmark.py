import os
import sys
import json
import csv
import argparse
import hashlib
from typing import Dict, Any, List, Tuple, Optional
import xml.etree.ElementTree as ET
from sklearn.metrics import confusion_matrix, matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
from datetime import datetime, timezone

def _hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _normalize_label(lbl: str) -> str:
    if lbl is None:
        return "unknown"
    s = str(lbl).strip().lower()
    mapping = {
        "pos": "positive", "positive": "positive", "+1": "positive", "1": "positive",
        "neg": "negative", "negative": "negative", "-1": "negative",
        "0": "neutral", "neu": "neutral", "neutral": "neutral", "conflict": "neutral",
    }
    return mapping.get(s, s)


def _sanitize_for_path(val: Any) -> str:
    s = str(val)
    return s.replace(' ', '_').replace('/', '_').replace(':', '_').replace('..', '.').replace('-', 'm').replace('.', 'p')


def _load_aspectterms_xml(path: str) -> List[Dict[str, Any]]:
    """Load both MAMS and SemEval restaurant format XML files."""
    data = []
    tree = ET.parse(path)
    root = tree.getroot()
    
    for sent in root.findall("sentence"):
        text_node = sent.find("text")
        text = text_node.text if text_node is not None else ""
        aspects = []
        
        at_node = sent.find("aspectTerms")
        if at_node is not None:
            for at in at_node.findall("aspectTerm"):
                term = at.attrib.get("term", "").strip()
                pol = at.attrib.get("polarity", "").strip()
                if term:
                    aspects.append({"term": term, "polarity": pol})
        
        if text and aspects:
            data.append({"text": text, "aspects": aspects})
    
    return data

def _extract_samples(dataset: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    samples = []
    for item in dataset:
        text = item.get("text", "")
        if not text:
            continue
        for aspect_info in item.get("aspects", []):
            term = aspect_info.get("term", "")
            pol = _normalize_label(aspect_info.get("polarity", ""))
            if term:
                samples.append((text, term, pol))
    return samples

def _jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def _find_best_entity_match(clusters: Dict, sent_map: Dict, aspect_term: str, text: str) -> Tuple[Optional[int], str]:
    aspect_lower = aspect_term.lower().strip()
    aspect_words = set(aspect_lower.split())
    
    best_cluster_id = None
    best_entity_text = ""
    best_score = 0.0
    
    for cluster_id, cluster_data in clusters.items():
        for entity_text, span in cluster_data.get("entity_references", []):
            entity_lower = entity_text.lower().strip()
            entity_words = set(entity_lower.split())
            
            if entity_lower == aspect_lower:
                return cluster_id, entity_text
            
            jaccard_score = _jaccard_similarity(aspect_words, entity_words)
            if jaccard_score > best_score:
                best_score = jaccard_score
                best_cluster_id = cluster_id
                best_entity_text = entity_text
            
            if aspect_lower in entity_lower or entity_lower in aspect_lower:
                overlap_score = len(aspect_words & entity_words) / len(aspect_words | entity_words)
                if overlap_score > best_score:
                    best_score = overlap_score
                    best_cluster_id = cluster_id
                    best_entity_text = entity_text
    
    if best_cluster_id is not None and best_score > 0.2:
        return best_cluster_id, best_entity_text

    return None, aspect_term

def _get_entity_sentiment_score(cluster_id: Optional[int], aggregate_results: Dict[str, float]) -> float:
    if cluster_id is None:
        return 0.0
    
    for key, score in aggregate_results.items():
        if f"(ID {cluster_id})" in key:
            return score
    
    return 0.0

def _label_from_score(score: float, pos_thresh: float = 0.015, neg_thresh: float = -0.015) -> str:
    if score > pos_thresh:
        return "positive"
    elif score < neg_thresh:
        return "negative"
    else:
        return "neutral"

def main():
    parser = argparse.ArgumentParser(prog="benchmark_v2")
    parser.add_argument("--input", required=True, help="Path to XML dataset file")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--pos-thresh", type=float, default=0.02)
    parser.add_argument("--neg-thresh", type=float, default=-0.02)
    parser.add_argument("--cache-dir", default="src/graph/cache")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Dataset not found: {args.input}")
        sys.exit(1)
    
    data = _load_aspectterms_xml(args.input)
    samples = _extract_samples(data)
    
    if args.limit and args.limit > 0:
        samples = samples[:args.limit]
    
    os.makedirs(args.cache_dir, exist_ok=True)

    # If outputs not specified, save under outputs/benchmark with timestamp and specs
    outputs_dir = os.path.join("outputs", "benchmark")
    os.makedirs(outputs_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(args.input))[0] or "dataset"
    base = _sanitize_for_path(base)
    pos_s = _sanitize_for_path(args.pos_thresh)
    neg_s = _sanitize_for_path(args.neg_thresh)
    limit_s = _sanitize_for_path(args.limit or 'all')
    default_stub = f"bench_{base}_pos{pos_s}_neg{neg_s}_lim{limit_s}_{ts}"
    if not args.out_json:
        args.out_json = os.path.join(outputs_dir, default_stub + ".json")
    if not args.out_csv:
        args.out_csv = os.path.join(outputs_dir, default_stub + ".csv")
    cache_path = os.path.join(args.cache_dir, "benchmark_cache.jsonl")
    cache = {}
    
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    cache[obj["hash"]] = obj["result"]
                except:
                    pass
    
    from src.graph.integrate_graph import build_graph_with_optimal_functions
    from src.e_c.coref import resolve
    
    results_rows = []
    labels = ["positive", "neutral", "negative"]
    per_class = {lbl: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for lbl in labels}
    total = 0
    correct_total = 0
    y_true = []
    y_pred = []
    
    for idx, (text, aspect_term, gold_label) in enumerate(samples, start=1):
        print(f"Processing {idx}/{len(samples)}: {aspect_term}")
        
        text_hash = _hash_text(text)
        
        if text_hash in cache:
            cached_result = cache[text_hash]
            clusters = cached_result["clusters"]
            sent_map = cached_result["sent_map"]
            aggregate_results = cached_result["aggregate_results"]
        else:
            try:
                clusters, sent_map = resolve(text)
                graph, aggregate_results = build_graph_with_optimal_functions(text)
                
                result_to_cache = {
                    "clusters": clusters,
                    "sent_map": sent_map,
                    "aggregate_results": aggregate_results
                }
                
                with open(cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"hash": text_hash, "result": result_to_cache}, ensure_ascii=False) + "\n")
                
                cache[text_hash] = result_to_cache
                
            except Exception as e:
                print(f"Error processing text: {e}")
                clusters = {}
                sent_map = {}
                aggregate_results = {}

        cluster_id, matched_entity = _find_best_entity_match(clusters, sent_map, aspect_term, text)
        score = _get_entity_sentiment_score(cluster_id, aggregate_results)
        pred_label = _label_from_score(score, args.pos_thresh, args.neg_thresh)

        match_type = "entity_match" if cluster_id is not None else "no_match"

        if gold_label in labels:
            total += 1
            per_class[gold_label]["support"] += 1
            if pred_label == gold_label:
                correct_total += 1
                per_class[gold_label]["tp"] += 1
            else:
                per_class[gold_label]["fn"] += 1
                if pred_label in labels:
                    per_class[pred_label]["fp"] += 1
            y_true.append(gold_label)
            y_pred.append(pred_label if pred_label in labels else "neutral")

        results_rows.append({
            "id": idx,
            "text": text,
            "aspect": aspect_term,
            "gold": gold_label,
            "pred": pred_label,
            "score": score,
            "entity_matched": matched_entity,
            "cluster_id": cluster_id,
            "matched_by": match_type
        })
    
    def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1
    
    per_class_metrics = {}
    micro_tp = micro_fp = micro_fn = 0
    macro_p = macro_r = macro_f1 = 0.0
    total_support = sum(pc["support"] for pc in per_class.values())
    weighted_p = weighted_r = weighted_f1 = 0.0
    
    for lbl in labels:
        tp_c = per_class[lbl]["tp"]
        fp_c = per_class[lbl]["fp"]
        fn_c = per_class[lbl]["fn"]
        sup = per_class[lbl]["support"]
        p, r, f1 = _prf(tp_c, fp_c, fn_c)
        per_class_metrics[lbl] = {
            "tp": tp_c, "fp": fp_c, "fn": fn_c, "support": sup,
            "precision": p, "recall": r, "f1": f1,
        }
        micro_tp += tp_c
        micro_fp += fp_c
        micro_fn += fn_c
        macro_p += p
        macro_r += r
        macro_f1 += f1
        if total_support > 0:
            weight = sup / total_support
            weighted_p += p * weight
            weighted_r += r * weight
            weighted_f1 += f1 * weight
    
    macro_count = len(labels)
    micro_p, micro_r, micro_f1 = _prf(micro_tp, micro_fp, micro_fn)
    
    accuracy = (correct_total / total) if total else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist() if total else [[0]*len(labels) for _ in labels]
    mcc = matthews_corrcoef(y_true, y_pred) if total else 0.0
    kappa = cohen_kappa_score(y_true, y_pred) if total else 0.0
    bal_acc = balanced_accuracy_score(y_true, y_pred) if total else 0.0
    
    summary = {
        "total": total,
        "correct": correct_total,
        "accuracy": accuracy,
        "per_class": per_class_metrics,
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro": {
            "precision": (macro_p / macro_count) if macro_count else 0.0,
            "recall": (macro_r / macro_count) if macro_count else 0.0,
            "f1": (macro_f1 / macro_count) if macro_count else 0.0,
        },
        "weighted": {"precision": weighted_p, "recall": weighted_r, "f1": weighted_f1},
        "confusion_matrix": {"labels": labels, "matrix": cm},
        "mcc": mcc,
        "cohen_kappa": kappa,
        "balanced_accuracy": bal_acc,
        "pos_thresh": args.pos_thresh,
        "neg_thresh": args.neg_thresh,
    }
    
    if args.out_json:
        out = {
            "spec": {
                "input": args.input,
                "limit": args.limit,
                "pos_thresh": args.pos_thresh,
                "neg_thresh": args.neg_thresh,
                "cache_dir": args.cache_dir,
                "timestamp_utc": ts
            },
            "summary": summary,
            "items": results_rows
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        # also write a lightweight pointer to latest
        try:
            with open(os.path.join(outputs_dir, "latest_run.txt"), "w", encoding="utf-8") as lf:
                lf.write(args.out_json)
        except Exception:
            pass
    
    if args.out_csv:
        if results_rows:
            with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(results_rows[0].keys()))
                w.writeheader()
                for r in results_rows:
                    w.writerow(r)
    
    print(json.dumps({"summary": summary}, indent=2))

if __name__ == "__main__":
    main()
