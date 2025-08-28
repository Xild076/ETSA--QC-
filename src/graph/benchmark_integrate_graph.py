import os
import sys
import json
import csv
import argparse
import hashlib
from typing import Dict, Any, List, Tuple
import xml.etree.ElementTree as ET
from sklearn.metrics import confusion_matrix, matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score


def _hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _load_json_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError("Unsupported JSON dataset format: expected a list of examples")
    return data


def _load_csv_dataset(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _load_mams_xml(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    tree = ET.parse(path)
    root = tree.getroot()
    for sent in root.findall("sentence"):
        text_node = sent.find("text")
        text = text_node.text if text_node is not None else ""
        ac_node = sent.find("aspectCategories")
        aspects: List[Dict[str, Any]] = []
        if ac_node is not None:
            for ac in ac_node.findall("aspectCategory"):
                cat = ac.attrib.get("category", "").strip()
                pol = ac.attrib.get("polarity", "").strip()
                if cat:
                    aspects.append({"category": cat, "polarity": pol})
        if text and aspects:
            data.append({"text": text, "aspects": aspects})
    return data


def _normalize_label(lbl: str) -> str:
    if lbl is None:
        return "unknown"
    s = str(lbl).strip().lower()
    mapping = {
        "pos": "positive",
        "positive": "positive",
        "+1": "positive",
        "1": "positive",
        "neg": "negative",
        "negative": "negative",
        "-1": "negative",
        "0": "neutral",
        "neu": "neutral",
        "neutral": "neutral",
        "conflict": "neutral",
    }
    return mapping.get(s, s)


def _extract_samples(dataset: List[Dict[str, Any]], task: str) -> List[Tuple[str, str, str]]:
    samples: List[Tuple[str, str, str]] = []
    for item in dataset:
        text = item.get("text") or item.get("sentence") or item.get("content") or ""
        if not text:
            continue
        aspects = []
        if isinstance(item.get("aspects"), list):
            for a in item["aspects"]:
                term = a.get("term") or a.get("aspect") or a.get("target") or a.get("category") or ""
                pol = _normalize_label(a.get("polarity") or a.get("label") or a.get("sentiment"))
                if term:
                    aspects.append((term, pol))
        elif isinstance(item.get("targets"), list):
            for a in item["targets"]:
                term = a.get("term") or a.get("target") or a.get("aspect") or a.get("category") or ""
                pol = _normalize_label(a.get("polarity") or a.get("label") or a.get("sentiment"))
                if term:
                    aspects.append((term, pol))
        else:
            term = item.get("aspect") or item.get("term") or item.get("target") or item.get("category")
            pol = _normalize_label(item.get("polarity") or item.get("label") or item.get("sentiment"))
            if term:
                aspects.append((term, pol))
        for term, pol in aspects:
            samples.append((text, term, pol))
    return samples


def _label_from_score(score: float, pos_thresh: float = 0.05, neg_thresh: float = -0.05) -> str:
    if score > pos_thresh:
        return "positive"
    if score < neg_thresh:
        return "negative"
    return "neutral"


def _best_entity_match(results: Dict[str, float], aspect: str) -> Tuple[str, float]:
    if not results:
        return "", 0.0
    aspect_l = aspect.lower().strip()
    exact = [k for k in results.keys() if k.lower() == aspect_l]
    if exact:
        k = exact[0]
        return k, results[k]
    subs = [k for k in results.keys() if aspect_l in k.lower() or k.lower() in aspect_l]
    if subs:
        k = sorted(subs, key=lambda x: len(x))[0]
        return k, results[k]
    atoks = set(aspect_l.split())
    best_k = ""
    best_overlap = -1
    for k in results.keys():
        ktoks = set(k.lower().split())
        ov = len(atoks & ktoks)
        if ov > best_overlap:
            best_overlap = ov
            best_k = k
    if best_k:
        return best_k, results[best_k]
    return "", 0.0


def _ensure_cache_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(prog="benchmark_integrate_graph")
    parser.add_argument("--dataset", choices=["mams", "semeval14", "semeval16", "custom"], required=True)
    parser.add_argument("--input", required=True, help="Path to dataset file (json or csv)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--pos-thresh", type=float, default=0.05)
    parser.add_argument("--neg-thresh", type=float, default=-0.05)
    parser.add_argument("--rpm", type=float, default=None, help="Requests per minute cap for LLM calls")
    parser.add_argument("--cache-dir", default="src/graph/cache")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--ablate", action="append", default=[], help="Comma-separated or repeatable flags: no-coref,no-mods,no-assoc,no-belong,only-action")
    parser.add_argument("--sent-model", default="preset", choices=[
        "preset","vader","textblob","swn","flair","prosus","distilbertlogit","finiteautomata","nlptown","pysentimiento",
        "ensemble-lex","ensemble-transf","ensemble-lextrans"
    ])
    parser.add_argument("--formula", default="optimized", choices=["optimized","avg","linear"])
    parser.add_argument("--score-key", default="user_sentiment_score_mapped", choices=[
        "user_sentiment_score_mapped","user_normalized_sentiment_scores","user_sentiment_score"
    ])
    parser.add_argument("--with-pyabsa", action="store_true")
    parser.add_argument("--pyabsa-checkpoint", default="english")
    args = parser.parse_args()

    if args.rpm is not None:
        os.environ["GENAI_REQUESTS_PER_MIN"] = str(args.rpm)

    path = args.input
    if not os.path.exists(path):
        print(f"Dataset not found: {path}")
        sys.exit(1)
    if path.lower().endswith(".json"):
        data = _load_json_dataset(path)
    elif path.lower().endswith(".csv"):
        data = _load_csv_dataset(path)
    elif path.lower().endswith(".xml") and args.dataset == "mams":
        data = _load_mams_xml(path)
    else:
        print("Unsupported dataset file type; use .json, .csv or .xml (for MAMS)")
        sys.exit(1)

    samples = _extract_samples(data, args.dataset)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    _ensure_cache_dir(args.cache_dir)
    cache_path = os.path.join(args.cache_dir, "integrate_graph_cache.jsonl")
    cache: Dict[str, Dict[str, float]] = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    cache[obj["hash"]] = obj["results"]
                except Exception:
                    pass

    # Prepare ablations by monkeypatching integrate_graph as needed
    from src.graph import integrate_graph as ig
    from src.graph.integrate_graph import build_graph_with_optimal_functions
    from src.graph import relation_extraction as re_mod
    from src.graph import sas as sas
    from src.survey import survey_question_optimizer as sqo

    ablate_flags: List[str] = []
    for item in args.ablate:
        if not item:
            continue
        ablate_flags.extend([x.strip() for x in str(item).split(',') if x.strip()])
    ablate_flags = list(dict.fromkeys(ablate_flags))

    if "no-coref" in ablate_flags:
        _orig_resolve = ig.resolve
        def _resolve_no_coref(text: str, device: str = "-1"):
            clusters, sent_map = _orig_resolve(text, device)
            new_clusters = {}
            next_id = 0
            seen = set()
            for clause_key, info in sent_map.items():
                for (txt, span) in info.get("entities", []):
                    s, e = span
                    if (s, e) in seen:
                        continue
                    new_clusters[next_id] = {"entity_references": [(txt, [s, e])]} 
                    seen.add((s, e))
                    next_id += 1
            return new_clusters, sent_map
        ig.resolve = _resolve_no_coref

    if "no-mods" in ablate_flags:
        ig.extract_entity_modifiers = lambda clause_text, entity_name: []

    if any(flag in ablate_flags for flag in ("only-action", "no-assoc", "no-belong")):
        _orig_re_api = re_mod.re_api
        def _re_api_filtered(sentence: str, entities: List[str], api_key: str = re_mod.API_KEY):
            out = _orig_re_api(sentence, entities, api_key)
            rels = out.get("relations", [])
            keep = rels
            if "only-action" in ablate_flags:
                keep = [r for r in rels if r.get("relation", {}).get("type") == "action"]
            else:
                if "no-assoc" in ablate_flags:
                    rels = [r for r in rels if r.get("relation", {}).get("type") != "association"]
                if "no-belong" in ablate_flags:
                    rels = [r for r in rels if r.get("relation", {}).get("type") != "belonging"]
                keep = rels
            out["relations"] = keep
            return out
        ig.re_api = _re_api_filtered

    # Sentiment model ablation: swap analyzer factory used in integrate_graph
    def _make_analyzer(name: str):
        if name == "preset":
            return sas.PresetEnsembleSentimentAnalyzer()
        if name == "vader":
            return sas.VADERSentimentAnalyzer()
        if name == "textblob":
            return sas.TextBlobSentimentAnalyzer()
        if name == "swn":
            return sas.SWNSentimentAnalyzer()
        if name == "flair":
            return sas.FlairSentimentAnalyzer()
        if name == "prosus":
            return sas.ProsusAISentimentAnalyzer()
        if name == "distilbertlogit":
            return sas.DISTILBERTLOGITSentimentAnalyzer()
        if name == "finiteautomata":
            return sas.FiniteautomataSentimentAnalyzer()
        if name == "nlptown":
            return sas.NLPTownSentimentAnalyzer()
        if name == "pysentimiento":
            return sas.PySentimientoSentimentAnalyzer()
        if name == "ensemble-lex":
            return sas.EnsembleSentimentAnalyzer([
                sas.VADERSentimentAnalyzer(), sas.TextBlobSentimentAnalyzer(), sas.SWNSentimentAnalyzer()
            ])
        if name == "ensemble-transf":
            return sas.EnsembleSentimentAnalyzer([
                sas.DISTILBERTLOGITSentimentAnalyzer(), sas.ProsusAISentimentAnalyzer(), sas.FlairSentimentAnalyzer(), sas.FiniteautomataSentimentAnalyzer()
            ])
        if name == "ensemble-lextrans":
            return sas.EnsembleSentimentAnalyzer([
                sas.VADERSentimentAnalyzer(), sas.TextBlobSentimentAnalyzer(), sas.SWNSentimentAnalyzer(),
                sas.DISTILBERTLOGITSentimentAnalyzer(), sas.ProsusAISentimentAnalyzer()
            ])
        return sas.PresetEnsembleSentimentAnalyzer()

    analyzer_instance = _make_analyzer(args.sent_model)
    ig.PresetEnsembleSentimentAnalyzer = lambda: analyzer_instance

    # Formula ablation: override formula providers in integrate_graph
    def _clamp(x: float) -> float:
        return max(-1.0, min(1.0, float(x)))

    if args.formula == "optimized":
        ig.get_actor_function = lambda: sqo.get_actor_function(score_key=args.score_key)
        ig.get_target_function = lambda: sqo.get_target_function(score_key=args.score_key)
        ig.get_association_function = lambda: sqo.get_association_function(score_key=args.score_key)
        ig.get_parent_function = lambda: sqo.get_parent_function(score_key=args.score_key)
        ig.get_child_function = lambda: sqo.get_child_function(score_key=args.score_key)
        ig.get_aggregate_function = lambda: sqo.get_aggregate_function(score_key=args.score_key)
    elif args.formula == "avg":
        ig.get_actor_function = lambda: (lambda s_actor, s_action, s_target: _clamp((s_actor + s_action) / 2.0))
        ig.get_target_function = lambda: (lambda s_target, s_action: _clamp((s_target + s_action) / 2.0))
        ig.get_association_function = lambda: (lambda s1, s2: _clamp((s1 + s2) / 2.0))
        ig.get_parent_function = lambda: (lambda s_parent, s_child: _clamp(0.7 * s_parent + 0.3 * s_child))
        ig.get_child_function = lambda: (lambda s_child, s_parent: _clamp(0.7 * s_child + 0.3 * s_parent))
        ig.get_aggregate_function = lambda: (lambda s_inits: _clamp(sum(s_inits) / len(s_inits) if s_inits else 0.0))
    elif args.formula == "linear":
        ig.get_actor_function = lambda: (lambda s_actor, s_action, s_target: _clamp(0.5 * s_actor + 0.4 * s_action - 0.1 * s_target))
        ig.get_target_function = lambda: (lambda s_target, s_action: _clamp(0.6 * s_target + 0.4 * s_action))
        ig.get_association_function = lambda: (lambda s1, s2: _clamp(0.5 * s1 + 0.5 * s2))
        ig.get_parent_function = lambda: (lambda s_parent, s_child: _clamp(0.8 * s_parent + 0.2 * s_child))
        ig.get_child_function = lambda: (lambda s_child, s_parent: _clamp(0.8 * s_child + 0.2 * s_parent))
        ig.get_aggregate_function = lambda: (lambda s_inits: _clamp(sum(s_inits) / len(s_inits) if s_inits else 0.0))

    results_rows: List[Dict[str, Any]] = []
    labels = ["positive", "neutral", "negative"]
    per_class = {lbl: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for lbl in labels}
    total = 0
    correct_total = 0
    y_true: List[str] = []
    y_pred: List[str] = []

    cfg_str = f"sent={args.sent_model}|formula={args.formula}|score_key={args.score_key}|ablate={','.join(ablate_flags)}|pos={args.pos_thresh}|neg={args.neg_thresh}"
    for idx, (text, aspect, gold) in enumerate(samples, start=1):
        h = _hash_text(text + '|' + cfg_str)
        if h in cache:
            agg = cache[h]
        else:
            try:
                _, agg_map = build_graph_with_optimal_functions(text)
            except Exception as e:
                agg_map = {}
            agg = {}
            for k, v in agg_map.items():
                try:
                    agg[k] = float(v)
                except Exception:
                    pass
            with open(cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"hash": h, "results": agg}, ensure_ascii=False) + "\n")
            cache[h] = agg

        ent, score = _best_entity_match(agg, aspect)
        pred_label = _label_from_score(score, args.pos_thresh, args.neg_thresh) if ent else "unknown"
        gold_label = _normalize_label(gold)
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
            "aspect": aspect,
            "gold": gold_label,
            "pred": pred_label,
            "score": score,
            "entity_matched": ent
        })

    def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    per_class_metrics: Dict[str, Any] = {}
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
            "tp": tp_c,
            "fp": fp_c,
            "fn": fn_c,
            "support": sup,
            "precision": p,
            "recall": r,
            "f1": f1,
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
    macro_metrics = {
        "precision": (macro_p / macro_count) if macro_count else 0.0,
        "recall": (macro_r / macro_count) if macro_count else 0.0,
        "f1": (macro_f1 / macro_count) if macro_count else 0.0,
    }
    weighted_metrics = {
        "precision": weighted_p,
        "recall": weighted_r,
        "f1": weighted_f1,
    }
    accuracy = (correct_total / total) if total else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist() if total else [[0]*len(labels) for _ in labels]
    mcc = matthews_corrcoef(y_true, y_pred) if total else 0.0
    kappa = cohen_kappa_score(y_true, y_pred) if total else 0.0
    bal_acc = balanced_accuracy_score(y_true, y_pred) if total else 0.0

    summary = {
        "total": total,
        "correct": correct_total,
        "accuracy": accuracy,
        "tp": micro_tp,
        "fp": micro_fp,
        "fn": micro_fn,
        "per_class": per_class_metrics,
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro": macro_metrics,
        "weighted": weighted_metrics,
        "confusion_matrix": {"labels": labels, "matrix": cm},
        "mcc": mcc,
        "cohen_kappa": kappa,
        "balanced_accuracy": bal_acc,
        "pos_thresh": args.pos_thresh,
        "neg_thresh": args.neg_thresh,
        "rpm": args.rpm,
        "ablation": ablate_flags,
        "sent_model": args.sent_model,
        "formula": args.formula,
        "score_key": args.score_key,
    }

    baseline = None
    if args.with_pyabsa:
        try:
            try:
                from pyabsa import APCCheckpointManager
                clf = APCCheckpointManager.get_sentiment_classifier(checkpoint=args.pyabsa_checkpoint, auto_device=True)
            except Exception:
                from pyabsa import AspectPolarityClassification as APC
                clf = APC.APCCheckpointManager.get_sentiment_classifier(checkpoint=args.pyabsa_checkpoint, auto_device=True)
            payload = [{"text": t, "aspect": a} for (t, a, g) in samples]
            try:
                outputs = clf.predict(payload, print_result=False, ignore_error=True)
            except Exception:
                outputs = clf.batch_predict(payload, print_result=False, ignore_error=True)
            by_idx = []
            for o in outputs:
                if isinstance(o, dict):
                    pred = str(o.get("sentiment") or o.get("polarity") or o.get("label") or "neutral").lower()
                else:
                    pred = str(o).lower()
                if pred not in labels:
                    if "pos" in pred:
                        pred = "positive"
                    elif "neg" in pred:
                        pred = "negative"
                    else:
                        pred = "neutral"
                by_idx.append(pred)
            base_true = []
            base_pred = []
            for i, (t, a, g) in enumerate(samples):
                gl = _normalize_label(g)
                if gl in labels:
                    base_true.append(gl)
                    base_pred.append(by_idx[i] if i < len(by_idx) else "neutral")
            b_total = len(base_true)
            b_correct = sum(1 for i in range(b_total) if base_true[i] == base_pred[i])
            b_cm = confusion_matrix(base_true, base_pred, labels=labels).tolist() if b_total else [[0]*len(labels) for _ in labels]
            b_mcc = matthews_corrcoef(base_true, base_pred) if b_total else 0.0
            b_kappa = cohen_kappa_score(base_true, base_pred) if b_total else 0.0
            b_bal_acc = balanced_accuracy_score(base_true, base_pred) if b_total else 0.0
            pc = {lbl: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for lbl in labels}
            for i in range(b_total):
                gl = base_true[i]
                pl = base_pred[i]
                pc[gl]["support"] += 1
                if pl == gl:
                    pc[gl]["tp"] += 1
                else:
                    pc[gl]["fn"] += 1
                    if pl in labels:
                        pc[pl]["fp"] += 1
            def _prf(tp: int, fp: int, fn: int):
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                return p, r, f1
            micro_tp = micro_fp = micro_fn = 0
            macro_p = macro_r = macro_f1 = 0.0
            total_support = sum(pc[l]["support"] for l in labels)
            weighted_p = weighted_r = weighted_f1 = 0.0
            per_class_metrics_b = {}
            for lbl in labels:
                tp_c = pc[lbl]["tp"]
                fp_c = pc[lbl]["fp"]
                fn_c = pc[lbl]["fn"]
                sup = pc[lbl]["support"]
                p, r, f1 = _prf(tp_c, fp_c, fn_c)
                per_class_metrics_b[lbl] = {"tp": tp_c, "fp": fp_c, "fn": fn_c, "support": sup, "precision": p, "recall": r, "f1": f1}
                micro_tp += tp_c
                micro_fp += fp_c
                micro_fn += fn_c
                macro_p += p
                macro_r += r
                macro_f1 += f1
                if total_support > 0:
                    w = sup / total_support
                    weighted_p += p * w
                    weighted_r += r * w
                    weighted_f1 += f1 * w
            micro_p_b, micro_r_b, micro_f1_b = _prf(micro_tp, micro_fp, micro_fn)
            macro_count = len(labels)
            macro_metrics_b = {"precision": (macro_p / macro_count) if macro_count else 0.0, "recall": (macro_r / macro_count) if macro_count else 0.0, "f1": (macro_f1 / macro_count) if macro_count else 0.0}
            weighted_metrics_b = {"precision": weighted_p, "recall": weighted_r, "f1": weighted_f1}
            baseline = {
                "total": b_total,
                "correct": b_correct,
                "accuracy": (b_correct / b_total) if b_total else 0.0,
                "per_class": per_class_metrics_b,
                "micro": {"precision": micro_p_b, "recall": micro_r_b, "f1": micro_f1_b},
                "macro": macro_metrics_b,
                "weighted": weighted_metrics_b,
                "confusion_matrix": {"labels": labels, "matrix": b_cm},
                "mcc": b_mcc,
                "cohen_kappa": b_kappa,
                "balanced_accuracy": b_bal_acc,
                "checkpoint": args.pyabsa_checkpoint,
            }
        except Exception as e:
            baseline = {"error": str(e)}

    if args.out_json:
        out = {
            "summary": summary,
            "items": results_rows,
        }
        if baseline is not None:
            out["pyabsa_baseline"] = baseline
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    if args.out_csv:
        with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results_rows[0].keys()) if results_rows else ["id","text","aspect","gold","pred","score","entity_matched"])
            w.writeheader()
            for r in results_rows:
                w.writerow(r)

    final_out = {"summary": summary}
    if baseline is not None:
        final_out["pyabsa_baseline"] = baseline
    print(json.dumps(final_out, indent=2))


if __name__ == "__main__":
    main()
