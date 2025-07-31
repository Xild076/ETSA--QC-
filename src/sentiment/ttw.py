import torch
import numpy as np
from collections import defaultdict
import re
from rich.console import Console
from tqdm import tqdm
import json
import ssl
import sys
import ast
import os
import argparse
from typing import Dict, List, Any, Set, Tuple

ssl._create_default_https_context = ssl._create_unverified_context
console = Console(width=120)
try:
    import nltk
    from nltk.corpus import sentiwordnet as swn, wordnet
    import spacy
    from wordfreq import word_frequency
    nltk.download('sentiwordnet', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    NLP = spacy.load("en_core_web_lg")
except (ImportError, OSError):
    nltk, swn, wordnet, spacy, NLP, word_frequency = None, None, None, None, None, None
    console.print("[bold red]Warning: Key NLP libraries (nltk, spacy, wordfreq) or models not found.[/bold red]")
    console.print("         Functionality like WordNet expansion, validation, and cohesion scoring will be disabled.")
    console.print("         To enable all features, run: [cyan]pip install nltk spacy wordfreq torch textblob flair vaderSentiment pysentimiento transformers rich[/cyan]")
    console.print("         Then download the SpaCy model: [cyan]python -m spacy download en_core_web_lg[/cyan]")

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from flair.models import TextClassifier
from flair.data import Sentence
from transformers import pipeline
from pysentimiento import create_analyzer

CONFIG = {
    "num_to_select": 5,
    "word_options_dir": "src/sentiment/word_options",
    "output_dir": "src/sentiment/lexicons",
    "output_filename": "optimized_lexicon.json",
    "huggingface_models_to_load": [
        "distilbert-base-uncased-finetuned-sst-2-english",
        "nlptown/bert-base-multilingual-uncased-sentiment",
        "finiteautomata/bertweet-base-sentiment-analysis",
        "ProsusAI/finbert",
    ],
    "pysentimiento_model": "pysentimiento/robertuito-sentiment-analysis",
    "enable_wordnet_expansion": True,
    "MIN_WORD_FREQUENCY": 1e-6, # Corresponds to words like 'hesitant', 'witty', 'abhorrent'
    "valence_map": {
        "very": {"range": [0.80, 1.0], "midpoint": 0.90},
        "strong": {"range": [0.60, 0.80], "midpoint": 0.70},
        "moderate": {"range": [0.35, 0.60], "midpoint": 0.45},
        "slight": {"range": [0.1, 0.35], "midpoint": 0.20},
    },
    "LAXNESS_CONFIG": {
        "max_passes": 4,
        "score_leniency_step": 0.05,
        "sd_leniency_step": 0.05,
        "initial_sd_threshold": 0.35
    },
    "stopwords": {'a', 'an', 'the', 'person', 'and', 'to', 'use', 'very', 'truly', 'really', 'quite', 'so', 'yours', 'soul', 'mortal', 'individual'},
    "cohesion_anchor_words": {
        "pos_nouns": ["wonderful", "kind", "honorable", "virtuous", "good"],
        "neg_nouns": ["terrible", "cruel", "selfish", "toxic", "bad"],
        "pos_verbs": ["helped", "supported", "praised", "rescued", "inspired", "loved"],
        "neg_verbs": ["attacked", "betrayed", "harmed", "deceived", "manipulated"],
        "pos_desc": ["excellent", "perfect", "reliable", "effective", "wonderful", "beautiful"],
        "neg_desc": ["terrible", "awful", "broken", "useless", "faulty", "dangerous"]
    }
}

class SentimentAnalyzer:
    def __init__(self, console: Console, hf_models: List[str]):
        self.console = console
        self.device = 0 if torch.cuda.is_available() else -1
        console.print("[bold cyan]Initializing sentiment models...[/bold cyan]")
        self.vader = SentimentIntensityAnalyzer()
        self.flair = TextClassifier.load('sentiment')
        self.pysentimiento = create_analyzer(task="sentiment", lang="en")
        self.hf_pipelines = {}
        for model_name in hf_models:
            console.print(f"  > Loading Hugging Face model: [bold magenta]{model_name}[/bold magenta]...")
            try:
                self.hf_pipelines[model_name] = pipeline("sentiment-analysis", model=model_name, device=self.device, top_k=None)
            except Exception as e:
                console.print(f"[bold red]Failed to load model {model_name}: {e}[/bold red]")
        console.print("[bold green]All sentiment models initialized.[/bold green]")

    def _get_hf_score(self, result: List[Dict[str, Any]], model_name: str) -> float:
        score_map = {item['label'].lower().replace('pos', 'positive').replace('neg', 'negative').replace('neu', 'neutral'): item['score'] for item in result}
        if "nlptown" in model_name:
            score_val = int(re.search(r'\d+', result[0]['label']).group())
            return -1.0 + (2.0 * (score_val - 1) / 4.0)
        elif "distilbert" in model_name:
            sign = 1.0 if 'positive' in result[0]['label'].lower() else -1.0
            return result[0]['score'] * sign
        else:
            return score_map.get('positive', 0.0) - score_map.get('negative', 0.0)

    def analyze(self, texts: List[str]) -> Dict[str, Dict[str, float]]:
        all_results = defaultdict(dict)
        for text in tqdm(texts, desc="Analyzing (lightweight models)", leave=False, ncols=120):
            all_results[text]['vader'] = self.vader.polarity_scores(text)['compound']
            all_results[text]['textblob'] = TextBlob(text).sentiment.polarity
            flair_sentence = Sentence(text)
            self.flair.predict(flair_sentence)
            flair_label = flair_sentence.labels[0]
            all_results[text]['flair'] = flair_label.score if flair_label.value == 'POSITIVE' else -flair_label.score
            pysent_result = self.pysentimiento.predict(text)
            all_results[text]['pysentimiento'] = pysent_result.probas.get('POS', 0.0) - pysent_result.probas.get('NEG', 0.0)
            if swn:
                tokens = re.findall(r'\b[a-zA-Z-]+\b', text.lower())
                pos, neg, count = 0.0, 0.0, 0
                for token in tokens:
                    synsets = list(swn.senti_synsets(token))
                    if synsets:
                        pos += np.mean([s.pos_score() for s in synsets])
                        neg += np.mean([s.neg_score() for s in synsets])
                        count += 1
                all_results[text]['sentiwordnet'] = (pos - neg) / count if count > 0 else 0.0
        
        for model_name, pipe in self.hf_pipelines.items():
            model_short_name = model_name.split('/')[-1]
            try:
                hf_outputs = pipe(texts, batch_size=32, truncation=True)
                for text, result in zip(texts, hf_outputs):
                    all_results[text][model_short_name] = self._get_hf_score(result, model_name)
            except Exception as e:
                self.console.print(f"[bold red]Error with {model_name}: {e}[/bold red]")
                for text in texts: all_results[text][model_short_name] = 0.0
        return all_results

class LexiconOptimizer:
    def __init__(self, candidate_pools: Dict[str, List[str]], analyzer: SentimentAnalyzer, config: Dict):
        self.candidate_pools = candidate_pools
        self.analyzer = analyzer
        self.config = config
        self.anchor_docs = {cat: NLP(" ".join(words)) for cat, words in config["cohesion_anchor_words"].items()} if NLP else {}

    def _get_key_lemmas(self, text: str) -> Set[str]:
        if not NLP: return {w for w in re.findall(r'\b[a-zA-Z-]+\b', text.lower()) if w not in self.config["stopwords"]}
        doc = NLP(text.lower())
        return {token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.text not in self.config["stopwords"]}

    def _is_common_enough(self, text: str) -> bool:
        if not word_frequency: return True
        words = re.findall(r'\b[a-zA-Z-]+\b', text.lower())
        if not words: return False
        return all(word_frequency(w, 'en') >= self.config["MIN_WORD_FREQUENCY"] for w in words if w not in self.config["stopwords"])

    def _is_grammatically_valid(self, text: str, category_type: str) -> Tuple[bool, str]:
        if not NLP: return True, text
        
        if category_type == "verbs":
            test_sentence = f"They {text} them."
            doc = NLP(test_sentence)
            verbs = [tok for tok in doc if tok.pos_ == "VERB"]
            if not verbs: return False, text
            root_verb = next((tok for tok in doc if tok.dep_ == "ROOT"), None)
            if not root_verb or root_verb.pos_ != "VERB": return False, text
            
            # Simple past tense normalization
            if root_verb.tag_ != "VBD":
                try: # Try TextBlob first
                    past_tense_verb = TextBlob(root_verb.lemma_).words[0].lemmatize("v")
                except: # Fallback to SpaCy lemma
                    past_tense_verb = root_verb.lemma_
                
                # Reconstruct original phrase with past tense verb
                original_doc = NLP(text)
                original_verb = next((tok for tok in original_doc if tok.pos_ == "VERB"), None)
                if original_verb:
                    text = text.replace(original_verb.text, past_tense_verb)
            return True, text
        
        doc = NLP(text)
        if category_type == "desc":
            last_word = next((token for token in reversed(doc) if not token.is_punct), None)
            return last_word is not None and last_word.pos_ in {"ADJ", "ADV"}, text
        if category_type == "nouns":
            return any(token.pos_ == "NOUN" for token in doc) and not any(token.pos_ == "ADV" for token in doc), text
        return True, text

    def _get_cohesion_score(self, text: str, category: str) -> float:
        if not NLP or category not in self.anchor_docs: return 0.5
        doc = NLP(text)
        anchor_doc = self.anchor_docs[category]
        if not doc.has_vector or not anchor_doc.has_vector or doc.vector_norm == 0 or anchor_doc.vector_norm == 0: return 0.5
        return doc.similarity(anchor_doc)

    def _normalize_values(self, values: List[float]) -> np.ndarray:
        arr = np.array(values, dtype=float)
        min_val, max_val = arr.min(), arr.max()
        return np.zeros_like(arr) if max_val == min_val else (arr - min_val) / (max_val - min_val)

    def run(self) -> Dict[str, Any]:
        final_lexicons = defaultdict(lambda: defaultdict(list))
        global_used_lemmas = defaultdict(set)
        
        all_unique_candidates = set()
        processed_pools = {}
        for category, items in self.candidate_pools.items():
            category_type = category.split('_')[1]
            candidate_set = {item.strip() for item in items if item}
            
            valid_candidates = set()
            for cand in candidate_set:
                is_valid, normalized_cand = self._is_grammatically_valid(cand, category_type)
                if is_valid and self._is_common_enough(normalized_cand):
                    valid_candidates.add(normalized_cand)
            processed_pools[category] = valid_candidates
            all_unique_candidates.update(valid_candidates)
        
        unique_candidate_list = list(all_unique_candidates)
        if not unique_candidate_list:
            self.analyzer.console.print("[bold yellow]No valid candidates found to analyze.[/bold yellow]")
            return {}
            
        self.analyzer.console.print(f"[yellow]Analyzing sentiment for [bold]{len(unique_candidate_list)}[/bold] unique, valid candidate phrases...[/yellow]")
        master_scores_map = self.analyzer.analyze(unique_candidate_list)
        
        all_scored_candidates_by_cat = defaultdict(list)
        for category, candidate_set in processed_pools.items():
            for text in candidate_set:
                scores_dict = master_scores_map.get(text)
                if not scores_dict or not scores_dict.values(): continue
                scores = list(scores_dict.values())
                all_scored_candidates_by_cat[category].append({
                    "text": text,
                    "combined_score": np.mean([np.median(scores), np.mean(scores)]),
                    "standard_deviation": np.std(scores),
                })
        
        for category, scored_candidates in all_scored_candidates_by_cat.items():
            category_type = category.split('_')[1]
            remaining_candidates = list(scored_candidates)
            target_polarity = 1.0 if 'pos' in category else -1.0

            for level in self.config["valence_map"].keys():
                lax_cfg = self.config["LAXNESS_CONFIG"]
                
                for i in range(lax_cfg["max_passes"]):
                    if len(final_lexicons[category][level]) >= self.config["num_to_select"]: break
                    
                    score_leniency = i * lax_cfg["score_leniency_step"]
                    sd_threshold = lax_cfg["initial_sd_threshold"] + (i * lax_cfg["sd_leniency_step"])
                    min_score_abs, max_score_abs = self.config["valence_map"][level]["range"]
                    
                    min_target = target_polarity * (max_score_abs + score_leniency)
                    max_target = target_polarity * (min_score_abs - score_leniency)
                    if target_polarity > 0: min_target, max_target = max_target, min_target

                    potential_pool = [c for c in remaining_candidates if min_target <= c['combined_score'] <= max_target and c['standard_deviation'] <= sd_threshold]
                    if not potential_pool: continue

                    target_midpoint = target_polarity * self.config["valence_map"][level]["midpoint"]
                    distances = [abs(c['combined_score'] - target_midpoint) for c in potential_pool]
                    stds = [c['standard_deviation'] for c in potential_pool]
                    cohesions = [self._get_cohesion_score(c['text'], category) for c in potential_pool]
                    
                    norm_distances, norm_stds, norm_cohesions = self._normalize_values(distances), self._normalize_values(stds), self._normalize_values(cohesions)
                    
                    for j, cand in enumerate(potential_pool):
                        cand['goodness'] = (norm_distances[j] * 0.4) + (norm_stds[j] * 0.4) + ((1 - norm_cohesions[j]) * 0.2)
                    
                    for cand in sorted(potential_pool, key=lambda x: x['goodness']):
                        if len(final_lexicons[category][level]) >= self.config["num_to_select"]: break
                        key_lemmas = self._get_key_lemmas(cand['text'])
                        if not key_lemmas: continue
                        
                        if not any(k.intersection(global_used_lemmas[category_type]) for k in [key_lemmas]):
                            final_lexicons[category][level].append(cand)
                            global_used_lemmas[category_type].update(key_lemmas)
                            remaining_candidates = [c for c in remaining_candidates if c['text'] != cand['text']]
        
        return final_lexicons

    def format_output(self, optimized_lexicons: Dict, source_name: str) -> str:
        num_select = self.config['num_to_select']
        output_str = f"# DEFINITIVE SENTIMENT LEXICONS (Top {num_select} - Complete)\n# Source: {source_name}\n"
        for category, levels in sorted(optimized_lexicons.items()):
            output_str += f"\n{category} = {{\n"
            for level in self.config["valence_map"].keys():
                output_str += f'    "{level}": [\n'
                items = levels.get(level, [])
                if items:
                    is_negative = "neg" in category
                    sorted_items = sorted(items, key=lambda x: x['combined_score'], reverse=not is_negative)
                    for item in sorted_items:
                        score, std = item['combined_score'], item['standard_deviation']
                        output_str += f'        # Score: {score:.3f}, SD: {std:.3f}\n'
                        output_str += f'        "{item["text"]}",\n'
                else:
                    output_str += '        # (No suitable candidates found)\n'
                output_str += '    ],\n'
            output_str += "}}\n"
        return output_str

def load_candidate_pools(path: str, console: Console) -> Dict[str, List[str]]:
    candidate_pools = {}
    if not os.path.exists(path):
        console.print(f"[bold red]Error: Input path not found -> '{path}'[/bold red]"); sys.exit(1)
    files_to_load = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')] if os.path.isdir(path) else ([path] if path.endswith('.txt') else [])
    if not files_to_load: console.print(f"[bold red]Error: No .txt files found in '{path}'.[/bold red]"); sys.exit(1)
    
    for filepath in files_to_load:
        with open(filepath, 'r', encoding='utf-8') as f: content = f.read().strip()
        try:
            word_list = ast.literal_eval(content)
            if not isinstance(word_list, list): raise ValueError
        except (ValueError, SyntaxError):
            word_list = [line.strip().strip("'\",") for line in content.splitlines() if line.strip() and not line.strip().startswith('#')]
        key = os.path.basename(filepath).replace(".txt", "")
        seen = set()
        candidate_pools[key] = [w.lower() for w in word_list if w and not (w.lower() in seen or seen.add(w.lower()))]
    return candidate_pools

def save_results(lexicons: Dict, output_path: str):
    lexicons_for_json = {cat: {level: [{k: v for k, v in item.items() if k in ['text', 'combined_score', 'standard_deviation']} for item in items] for level, items in levels.items()} for cat, levels in lexicons.items()}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f: json.dump(lexicons_for_json, f, indent=4)
    console.print(f"[bold green]Optimized lexicons saved to {output_path}[/bold green]")

def main():
    parser = argparse.ArgumentParser(description="Generate definitive sentiment lexicons from candidate word lists.")
    parser.add_argument("--input", "-i", type=str, default=CONFIG["word_options_dir"], help=f"Path to candidate .txt files/directory. Default: {CONFIG['word_options_dir']}")
    parser.add_argument("--output", "-o", type=str, default=os.path.join(CONFIG["output_dir"], CONFIG["output_filename"]), help=f"Path to save output JSON. Default: {os.path.join(CONFIG['output_dir'], CONFIG['output_filename'])}")
    args = parser.parse_args()

    if any(x is None for x in [nltk, spacy, NLP, word_frequency]):
        CONFIG["enable_wordnet_expansion"] = False
        
    candidate_pools = load_candidate_pools(args.input, console)
    if not candidate_pools: return
        
    analyzer = SentimentAnalyzer(console, CONFIG["huggingface_models_to_load"])
    optimizer = LexiconOptimizer(candidate_pools, analyzer, CONFIG)
    optimized_lexicons = optimizer.run()
    
    if not optimized_lexicons: console.print("[bold red]Lexicon optimization failed to produce any results.[/bold red]"); return
    
    source_name = f"Input: {os.path.basename(args.input)}"
    save_results(optimized_lexicons, args.output)
    console.print(optimizer.format_output(optimized_lexicons, source_name))

    num_to_select, missing_count, total_slots, filled_slots = CONFIG['num_to_select'], 0, 0, 0
    for cat, levels in optimized_lexicons.items():
        for level in CONFIG["valence_map"].keys():
            total_slots += num_to_select
            num_found = len(levels.get(level, []))
            filled_slots += num_found
            if num_found < num_to_select:
                missing_count += 1
                console.print(f"  [yellow]Partially filled for [bold]{cat} -> {level}[/bold]: found {num_found} of {num_to_select}[/yellow]")

    if missing_count == 0:
        console.print("\n[bold green]✓ All categories and levels have been successfully filled![/bold green]")
    else:
        fill_percentage = (filled_slots / total_slots) * 100 if total_slots > 0 else 0
        console.print(f"\n[bold yellow]! Lexicon generation complete with some missing entries. Overall fill rate: {fill_percentage:.1f}%[/bold yellow]")

if __name__ == "__main__":
    main()