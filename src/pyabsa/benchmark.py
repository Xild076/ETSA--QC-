import os
import warnings
import logging
from typing import Literal
from datetime import datetime
import csv

import pandas as pd
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from afinn import Afinn
from tqdm import tqdm
from colorama import Fore, Style, init as colorama_init

from pyabsa import AspectTermExtraction as ATEPC, available_checkpoints

nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()
afn = Afinn()

def tokenize_with_word(text):
    return [token.text for token in nlp(text)]

def split_into_independent_clauses(text):
    clauses, current = [], []
    for token in nlp(text):
        if token.text in ("'", "’") and token.i > 0:
            if current: current[-1] += token.text
            continue
        if token.text in ("n't", "'re", "'s", "'m", "'ll", "'d", "'ve"):
            if current: current[-1] += token.text
            continue
        current.append(token.text)
        if token.dep_ == "cc" and token.head.dep_ == "ROOT":
            clauses.append(" ".join(current).strip())
            current = []
    if current:
        clauses.append(" ".join(current).strip())
    independent = []
    for c in clauses:
        span = nlp(c)
        roots = [t for t in span if t.dep_ == "ROOT"]
        if roots and any(t.dep_ == "nsubj" for t in span):
            independent.append(c)
    return independent

def normalized_afinn_score(text):
    scores = [afn.score(w) for w in tokenize_with_word(text)]
    relevant = [s for s in scores if s != 0]
    return sum(relevant) / len(relevant) / 5.0 if relevant else 0.0

def get_va_compound_sentiment(text):
    return (normalized_afinn_score(text) + sia.polarity_scores(text)["compound"]) / 2.0

def get_sentiment(text, accurate: Literal["high", "medium", "low"] = "high"):
    if not text or text.isspace():
        return 0.0
    if accurate == "high":
        try:
            r = transformer_pipeline_global(text, truncation=True, max_length=512)[0]
            return r["score"] if r["label"] == "POSITIVE" else -r["score"]
        except Exception:
            return get_va_compound_sentiment(text)
    if accurate == "medium":
        return get_va_compound_sentiment(text)
    return sia.polarity_scores(text)["compound"]

def check_sent_switch(text, accurate: Literal["high", "medium", "low"] = "high"):
    cnt, prev = 0, None
    for clause in split_into_independent_clauses(text):
        cur = get_sentiment(clause, accurate=accurate)
        if prev is not None and cur * prev < 0:
            cnt += 1
        prev = cur
    return cnt

def determine_sentence_validity(text, accurate: Literal["high", "medium", "low"] = "high"):
    c = check_sent_switch(text, accurate=accurate)
    return c > 0 and c % 2 == 1

transformer_pipeline_global = None

def mode_verify_dataset(cutoff):
    start = datetime.now()
    path = "data/dataset/amazon_reviews.csv"
    if not os.path.exists(path):
        print(Fore.RED + f"Error: {path} not found." + Style.RESET_ALL)
        return
    df = pd.read_csv(path, header=None, names=["sentiment", "title", "review"], quoting=csv.QUOTE_MINIMAL)
    valid_reviews, valid_sentiments = [], []
    for text, sent in tqdm(zip(df.review.astype(str), df.sentiment), desc="Verifying", total=len(df)):
        if determine_sentence_validity(text):
            valid_reviews.append(text)
            valid_sentiments.append(sent)
            if len(valid_reviews) >= cutoff:
                break
    pd.DataFrame({"review": valid_reviews, "sentiment": valid_sentiments}) \
      .to_csv("data/dataset/amazon_valid_reviews.csv", index=False)
    print(Fore.GREEN + f"Verified {len(valid_reviews)} in {datetime.now() - start}." + Style.RESET_ALL)

def mode_extract_aspects():
    start = datetime.now()
    if not os.path.exists("data/dataset/amazon_valid_reviews.csv"):
        print(Fore.RED + "Run vd first." + Style.RESET_ALL)
        return
    df = pd.read_csv("data/dataset/amazon_valid_reviews.csv")
    vr, vs = df.review.astype(str).tolist(), df.sentiment.tolist()
    aspects_file = "data/dataset/amazon_valid_aspects.csv"
    if os.path.exists(aspects_file):
        existing = pd.read_csv(aspects_file)
        processed = len(existing)
        kept = existing
    else:
        processed = 0
        kept = pd.DataFrame(columns=["review", "sentiment", "aspect"])
    to_process = list(range(processed, len(vr)))
    if not to_process:
        print(Fore.YELLOW + "All reviews processed." + Style.RESET_ALL)
        return
    texts = [vr[i] for i in to_process]
    sents = [vs[i] for i in to_process]
    results = extractor.predict(texts, save_result=False, print_result=False, ignore_error=True)
    rows, dropped = [], 0
    for text, sent, r in tqdm(zip(texts, sents, results), desc="Extracting", total=len(texts)):
        a_list = r.get("aspect", [])
        if a_list:
            rows.append({"review": text, "sentiment": sent, "aspect": a_list[0]})
        else:
            dropped += 1
    new_df = pd.concat([kept, pd.DataFrame(rows)], ignore_index=True)
    new_df.to_csv(aspects_file, index=False)
    print(Fore.YELLOW + f"Dropped {dropped}, kept {len(new_df)} in {datetime.now() - start}." + Style.RESET_ALL)

def mode_benchmark():
    start = datetime.now()
    if not os.path.exists("data/dataset/amazon_valid_aspects.csv"):
        print(Fore.RED + "Run ea first." + Style.RESET_ALL)
        return
    df = pd.read_csv("data/dataset/amazon_valid_aspects.csv")
    reviews, aspects, sentiments = df.review.astype(str), df.aspect.astype(str), df.sentiment
    tagged = [r.replace(a, f"[ASP]{a}[ASP]", 1) for r, a in zip(reviews, aspects)]
    true_labels = ["Positive" if s == 2 else "Negative" for s in sentiments]
    results_file = "data/dataset/amazon_benchmark_results.csv"
    if os.path.exists(results_file):
        existing = pd.read_csv(results_file)
        processed = len(existing)
    else:
        existing = pd.DataFrame(columns=["review", "aspect", "tagged", "true_label", "predicted_label"])
        processed = 0
    to_process = list(range(processed, len(tagged)))
    if to_process:
        texts = [tagged[i] for i in to_process]
        trues = [true_labels[i] for i in to_process]
        asps = [aspects[i] for i in to_process]
        results = extractor.predict(texts, pred_sentiment=True, save_result=False, print_result=False, ignore_error=True)
        rows = []
        for rev, asp, txt, true, r in zip(reviews, asps, texts, trues, results):
            s_list = r.get("sentiment", [])
            pred = s_list[0] if s_list else ""
            rows.append({"review": rev, "aspect": asp, "tagged": txt, "true_label": true, "predicted_label": pred})
        df_all = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        df_all.to_csv(results_file, index=False)
    else:
        df_all = existing
    total = len(df_all)
    correct = (df_all.true_label == df_all.predicted_label).sum()
    acc = correct / total if total else 0.0
    print(Fore.GREEN + f"Accuracy: {acc:.4f} over {total} in {datetime.now() - start}." + Style.RESET_ALL)

def mode_all():
    try:
        cutoff = int(input(Fore.CYAN + "Cutoff for valid reviews: " + Style.RESET_ALL))
    except:
        cutoff = 1000
    mode_verify_dataset(cutoff)
    mode_extract_aspects()
    mode_benchmark()

def main():
    colorama_init(autoreset=True)
    mode = input(Fore.CYAN + "Enter mode ('vd','ea','b','all'): " + Style.RESET_ALL).strip().lower()
    global transformer_pipeline_global, extractor
    transformer_pipeline_global = pipeline("sentiment-analysis")
    available_checkpoints()
    extractor = ATEPC.AspectExtractor("multilingual", auto_device=True, cal_perplexity=True)
    if mode == "vd":
        try:
            cutoff = int(input(Fore.CYAN + "Cutoff: " + Style.RESET_ALL))
        except:
            cutoff = 1000
        mode_verify_dataset(cutoff)
    elif mode == "ea":
        mode_extract_aspects()
    elif mode == "b":
        mode_benchmark()
    elif mode == "all":
        mode_all()
    else:
        print(Fore.RED + "Invalid mode." + Style.RESET_ALL)

if __name__ == "__main__":
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONWARNINGS"] = "ignore"
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.CRITICAL)
    for lib in ["transformers", "huggingface_hub", "pyabsa", "coreferee", "pkg_resources", "PIL"]:
        logging.getLogger(lib).setLevel(logging.CRITICAL)
    main()
