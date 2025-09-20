import json
import csv
from typing import List, Dict, Any, Tuple
from datetime import datetime
import os
from relation_extraction import GemmaRelationExtractor, API_KEY, extract_entity_modifiers

def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}

def evaluate_on_dataset(extractor: GemmaRelationExtractor, dataset: List[Dict[str, Any]], save_dir: str = "src/graph/data") -> Dict[str, Any]:
    print("="*80)
    print("Starting Evaluation on Micro-Dataset...")
    print(f"Total datapoints: {len(dataset)}")
    print("="*80)

    total_tp, total_fp, total_fn = 0, 0, 0
    total_gt_modifiers = 0
    total_correct_modifiers = 0
    total_pred_modifiers = 0
    per_item_rows: List[Dict[str, Any]] = []
    failed_cases = []

    for i, item in enumerate(dataset):
        sentence = item["sentence"]
        entities = item["entities"]
        ground_truth = item["ground_truth"]
        
        print(f"[{i+1}/{len(dataset)}] Processing: \"{sentence}\"")
        
        prediction_result = extractor.extract_relations(sentence, entities)
        predicted_relations = prediction_result.get("relations", [])
        
        gt_signatures = {
            (r["subject"]["head"].lower(), r["relation"]["type"].lower(), r["object"]["head"].lower())
            for r in ground_truth
        }
        
        pred_signatures = {
            (r["subject"]["head"].lower(), r["relation"]["type"].lower(), r["object"]["head"].lower())
            for r in predicted_relations
        }
        
        tp = len(gt_signatures.intersection(pred_signatures))
        fp = len(pred_signatures - gt_signatures)
        fn = len(gt_signatures - pred_signatures)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        if fp > 0 or fn > 0:
            failed_cases.append({
                "sentence": sentence,
                "ground_truth": ground_truth,
                "predicted": predicted_relations,
                "reason": f"FP: {fp}, FN: {fn}"
            })
            print(f"  -> Result: ❌ (TP: {tp}, FP: {fp}, FN: {fn})")
        else:
            print(f"  -> Result: ✅ (TP: {tp}, FP: {fp}, FN: {fn})")

        correct_signatures = gt_signatures.intersection(pred_signatures)
        for sig in correct_signatures:
            gt_rel = next((r for r in ground_truth if (r["subject"]["head"].lower(), r["relation"]["type"].lower(), r["object"]["head"].lower()) == sig), None)
            if gt_rel:
                subj = gt_rel["subject"]["head"]
                obj = gt_rel["object"]["head"]
                gt_subj_mods = set(m.lower() for m in gt_rel["subject"].get("modifiers", []))
                gt_obj_mods = set(m.lower() for m in gt_rel["object"].get("modifiers", []))
                pred_subj_mods = set(m.lower() for m in extract_entity_modifiers(sentence, subj))
                pred_obj_mods = set(m.lower() for m in extract_entity_modifiers(sentence, obj))
                total_gt_modifiers += len(gt_subj_mods) + len(gt_obj_mods)
                total_pred_modifiers += len(pred_subj_mods) + len(pred_obj_mods)
                total_correct_modifiers += len(gt_subj_mods.intersection(pred_subj_mods))
                total_correct_modifiers += len(gt_obj_mods.intersection(pred_obj_mods))

        per_item_rows.append({
            "sentence": sentence,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "num_predicted_relations": len(predicted_relations),
            "num_gt_relations": len(ground_truth),
        })

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    print("\n--- Core Relation Metrics ---")
    print(f"True Positives:  {total_tp}")
    print(f"False Positives: {total_fp}")
    print(f"False Negatives: {total_fn}")
    
    metrics = calculate_metrics(total_tp, total_fp, total_fn)
    print("\n--- Performance ---")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1-Score:  {metrics['f1']:.2%}")

    print("\n--- Modifier Metrics (on correctly identified relations) ---")
    modifier_accuracy = total_correct_modifiers / total_gt_modifiers if total_gt_modifiers > 0 else 0
    modifier_precision = total_correct_modifiers / total_pred_modifiers if total_pred_modifiers > 0 else 0
    modifier_recall = total_correct_modifiers / total_gt_modifiers if total_gt_modifiers > 0 else 0
    modifier_f1 = 2 * modifier_precision * modifier_recall / (modifier_precision + modifier_recall) if (modifier_precision + modifier_recall) > 0 else 0
    print(f"Total Ground Truth Modifiers: {total_gt_modifiers}")
    print(f"Total Predicted Modifiers: {total_pred_modifiers}")
    print(f"Correctly Identified Modifiers: {total_correct_modifiers}")
    print(f"Modifier Accuracy: {modifier_accuracy:.2%}")
    print(f"Modifier Precision: {modifier_precision:.2%}")
    print(f"Modifier Recall: {modifier_recall:.2%}")
    print(f"Modifier F1: {modifier_f1:.2%}")

    if failed_cases:
        print("\n" + "-"*80)
        print(f"Analysis of {len(failed_cases)} Failed Cases:")
        print("-"*80)
        for i, case in enumerate(failed_cases, 1):
            print(f"\nCase #{i}: {case['sentence']}")
            print(f"  Reason: {case['reason']}")
            print(f"  Ground Truth: {json.dumps(case['ground_truth'], indent=2)}")
            print(f"  Model Predicted: {json.dumps(case['predicted'], indent=2)}")
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "core": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            **metrics,
        },
        "modifiers": {
            "gt": total_gt_modifiers,
            "pred": total_pred_modifiers,
            "correct": total_correct_modifiers,
            "accuracy": modifier_accuracy,
            "precision": modifier_precision,
            "recall": modifier_recall,
            "f1": modifier_f1,
        },
        "failed_cases": failed_cases,
    }
    json_path = os.path.join(save_dir, f"re_eval_{timestamp}.json")
    with open(json_path, "w") as jf:
        json.dump(summary, jf, indent=2)

    csv_path = os.path.join(save_dir, f"re_eval_items_{timestamp}.csv")
    with open(csv_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=["sentence", "tp", "fp", "fn", "num_predicted_relations", "num_gt_relations"])
        writer.writeheader()
        writer.writerows(per_item_rows)

    print(f"\nSaved summary to {json_path}\nSaved per-item to {csv_path}")
    print("\n" + "="*80)
    return summary

if __name__ == "__main__":
    try:
        if not API_KEY:
            raise ValueError("GOOGLE_API_KEY not found. Please set it.")
            
        gemma_extractor = GemmaRelationExtractor(api_key=API_KEY)
        
        dataset_file = "src/graph/data/micro_dataset.json"
        with open(dataset_file, 'r') as f:
            test_dataset = json.load(f)
            
        evaluate_on_dataset(gemma_extractor, test_dataset)
        
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{dataset_file}'.")
        print("Please run 'python create_dataset.py' first to generate it.")
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")