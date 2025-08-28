import os
import sys
import json
import itertools
import argparse
import subprocess
from datetime import datetime


def run_benchmark(input_path, out_json, ablate, sent_model, formula, score_key, rpm, pos_thresh, neg_thresh, with_pyabsa, pyabsa_checkpoint, limit):
    cmd = [sys.executable, "-m", "src.graph.benchmark_integrate_graph", "--dataset", "mams", "--input", input_path, "--out-json", out_json, "--sent-model", sent_model, "--formula", formula, "--score-key", score_key, "--pos-thresh", str(pos_thresh), "--neg-thresh", str(neg_thresh)]
    if rpm is not None:
        cmd.extend(["--rpm", str(rpm)])
    if with_pyabsa:
        cmd.append("--with-pyabsa")
        cmd.extend(["--pyabsa-checkpoint", pyabsa_checkpoint])
    for a in ablate:
        cmd.extend(["--ablate", a])
    if limit is not None and limit > 0:
        cmd.extend(["--limit", str(limit)])
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip())
    with open(out_json, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(prog="run_mams_experiments")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rpm", type=float, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--pos-thresh", type=float, default=0.05)
    parser.add_argument("--neg-thresh", type=float, default=-0.05)
    parser.add_argument("--score-key", default="user_sentiment_score_mapped")
    parser.add_argument("--pyabsa-checkpoint", default="english")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sent_models = ["preset", "ensemble-lex", "ensemble-transf"]
    formulas = ["optimized", "avg", "linear"]
    ablation_sets = [[], ["no-coref"], ["no-mods"], ["no-assoc"], ["no-belong"], ["only-action"], ["no-coref", "no-mods"], ["no-assoc", "no-belong"]]

    all_rows = []
    combined = []

    baseline_done = False
    for sent_model, formula in itertools.product(sent_models, formulas):
        for ablate in ablation_sets:
            out_json = os.path.join(args.output_dir, f"mams_{sent_model}_{formula}_{'-'.join(ablate) if ablate else 'none'}_{timestamp}.json")
            res = run_benchmark(
                input_path=args.input,
                out_json=out_json,
                ablate=ablate,
                sent_model=sent_model,
                formula=formula,
                score_key=args.score_key,
                rpm=args.rpm,
                pos_thresh=args.pos_thresh,
                neg_thresh=args.neg_thresh,
                with_pyabsa=(not baseline_done and len(ablate) == 0 and sent_model == "preset" and formula == "optimized"),
                pyabsa_checkpoint=args.pyabsa_checkpoint,
                limit=args.limit if args.limit and args.limit > 0 else None,
            )
            baseline_done = baseline_done or ("pyabsa_baseline" in res)
            combined.append(res)
            s = res.get("summary", {})
            micro = s.get("micro", {})
            macro = s.get("macro", {})
            weighted = s.get("weighted", {})
            row = {
                "sent_model": sent_model,
                "formula": formula,
                "ablation": ",".join(ablate) if ablate else "none",
                "total": s.get("total", 0),
                "accuracy": s.get("accuracy", 0.0),
                "micro_precision": micro.get("precision", 0.0),
                "micro_recall": micro.get("recall", 0.0),
                "micro_f1": micro.get("f1", 0.0),
                "macro_precision": macro.get("precision", 0.0),
                "macro_recall": macro.get("recall", 0.0),
                "macro_f1": macro.get("f1", 0.0),
                "weighted_precision": weighted.get("precision", 0.0),
                "weighted_recall": weighted.get("recall", 0.0),
                "weighted_f1": weighted.get("f1", 0.0),
                "mcc": s.get("mcc", 0.0),
                "cohen_kappa": s.get("cohen_kappa", 0.0),
                "balanced_accuracy": s.get("balanced_accuracy", 0.0),
            }
            all_rows.append(row)

    agg_json = os.path.join(args.output_dir, f"mams_experiments_{timestamp}.json")
    with open(agg_json, "w", encoding="utf-8") as f:
        json.dump({"runs": combined}, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(args.output_dir, f"mams_experiments_{timestamp}.csv")
    cols = [
        "sent_model","formula","ablation","total","accuracy","micro_precision","micro_recall","micro_f1","macro_precision","macro_recall","macro_f1","weighted_precision","weighted_recall","weighted_f1","mcc","cohen_kappa","balanced_accuracy"
    ]
    import csv
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(json.dumps({"json": agg_json, "csv": csv_path}, indent=2))


if __name__ == "__main__":
    main()
