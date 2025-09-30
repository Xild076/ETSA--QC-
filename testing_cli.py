import argparse
import sys
import os
import subprocess
from datetime import datetime
from colorama import Fore, Style

def run_app():
    print(f"{Fore.CYAN}Launching Streamlit app...{Style.RESET_ALL}")
    subprocess.run(["streamlit", "run", "src/survey/survey.py"], cwd=os.path.dirname(__file__))

def run_tests():
    print(f"{Fore.GREEN}Running tests...{Style.RESET_ALL}")
    from src.survey.survey_question_optimizer import test_all_parameterizations
    out = test_all_parameterizations()
    print("\nSummary:")
    for sk, summary in out.items():
        print(f"score_key={sk}")
        for k, v in summary.items():
            if v and "best" in v:
                print(f"- {k}: MSE={v['best']['mse']:.4f} | model={v['best']['model']} | split={v['best']['split']} | outliers={v['best'].get('remove_outliers','none')}")

def run_benchmark(
    dataset: str,
    input_path: str,
    limit: int,
    pos_thresh: float,
    neg_thresh: float,
    rpm: float | None,
    output_dir: str,
    ablate: list[str],
    sent_model: str,
    formula: str,
    score_key: str,
    with_pyabsa: bool,
    pyabsa_checkpoint: str,
):
    repo_dir = os.path.dirname(__file__)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(repo_dir, output_dir)
    os.makedirs(out_dir, exist_ok=True)
    base = f"{dataset}_{ts}"
    out_json = os.path.join(out_dir, f"{base}.json")
    out_csv = os.path.join(out_dir, f"{base}.csv")

    print(f"{Fore.CYAN}Running benchmark on {dataset} ({limit or 'all'} items)…{Style.RESET_ALL}")

    cmd = [
        sys.executable, "-m", "src.pipeline.benchmark",
        "--dataset", dataset,
        "--input", input_path,
        "--pos-thresh", str(pos_thresh),
        "--neg-thresh", str(neg_thresh),
        "--formula", formula,
        "--score-key", score_key,
        "--sent-model", sent_model,
        "--out-json", out_json,
        "--out-csv", out_csv,
    ]
    if limit and limit > 0:
        cmd.extend(["--limit", str(limit)])
    if rpm is not None:
        cmd.extend(["--rpm", str(rpm)])
    for flag in ablate or []:
        cmd.extend(["--ablate", flag])
    if with_pyabsa:
        cmd.append("--with-pyabsa")
        if pyabsa_checkpoint:
            cmd.extend(["--pyabsa-checkpoint", pyabsa_checkpoint])

    # Display a short, colorized preview of the command
    printable = " ".join(cmd)
    print(f"{Fore.MAGENTA}{printable}{Style.RESET_ALL}")

    res = subprocess.run(cmd, cwd=repo_dir)
    if res.returncode != 0:
        print(f"{Fore.RED}Benchmark failed with exit code {res.returncode}.{Style.RESET_ALL}")
        sys.exit(res.returncode)

    print(f"{Fore.GREEN}Benchmark complete.{Style.RESET_ALL}")
    print(f"- JSON: {out_json}")
    print(f"- CSV:  {out_csv}")

def main():
    parser = argparse.ArgumentParser(description="ETSA CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sp_app = subparsers.add_parser("app", help="Launch Streamlit app")

    sp_tests = subparsers.add_parser("tests", help="Run optimizer tests")

    sp_ttw = subparsers.add_parser("ttw", help="Run ttw demo")

    sp_bench = subparsers.add_parser("benchmark", help="Run ABSA benchmark on a dataset (e.g., MAMS)")
    sp_bench.add_argument("--dataset", choices=["mams", "semeval14", "semeval16", "custom"], default="mams")
    sp_bench.add_argument("--input", default=os.path.join("data", "dataset", "test_mams.xml"), help="Path to dataset file (.xml for MAMS, .json/.csv for others)")
    sp_bench.add_argument("--limit", type=int, default=10)
    sp_bench.add_argument("--pos-thresh", type=float, default=0.05)
    sp_bench.add_argument("--neg-thresh", type=float, default=-0.05)
    sp_bench.add_argument("--rpm", type=float, default=None, help="Requests per minute cap for LLM calls")
    sp_bench.add_argument("--output-dir", default=os.path.join("outputs", "benchmarks"))
    sp_bench.add_argument("--ablate", action="append", default=[], help="Comma-separated or repeatable flags: no-coref,no-mods,no-assoc,no-belong,only-action")
    sp_bench.add_argument("--sent-model", default="preset", choices=[
        "preset","vader","textblob","swn","flair","prosus","distilbertlogit","finiteautomata","nlptown","pysentimiento",
        "ensemble-lex","ensemble-transf","ensemble-lextrans"
    ])
    sp_bench.add_argument("--formula", default="optimized", choices=["optimized","avg","linear"])
    sp_bench.add_argument("--score-key", default="user_sentiment_score_mapped", choices=[
        "user_sentiment_score_mapped","user_normalized_sentiment_scores","user_sentiment_score"
    ])
    sp_bench.add_argument("--with-pyabsa", action="store_true")
    sp_bench.add_argument("--pyabsa-checkpoint", default="english")

    args = parser.parse_args()

    if args.command == "app":
        run_app()
    elif args.command == "tests":
        run_tests()
    elif args.command == "ttw":
        print(f"{Fore.YELLOW}Running ttw.py...{Style.RESET_ALL}")
        subprocess.run([sys.executable, "src/sentiment/ttw.py"], cwd=os.path.dirname(__file__))
    elif args.command == "benchmark":
        run_benchmark(
            dataset=args.dataset,
            input_path=args.input,
            limit=args.limit,
            pos_thresh=args.pos_thresh,
            neg_thresh=args.neg_thresh,
            rpm=args.rpm,
            output_dir=args.output_dir,
            ablate=args.ablate,
            sent_model=args.sent_model,
            formula=args.formula,
            score_key=args.score_key,
            with_pyabsa=args.with_pyabsa,
            pyabsa_checkpoint=args.pyabsa_checkpoint,
        )

if __name__ == "__main__":
    main()