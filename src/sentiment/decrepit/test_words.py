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
from typing import Dict, List, Any, Set

ssl._create_default_https_context = ssl._create_unverified_context
console = Console()
try:
    import nltk
    from nltk.corpus import sentiwordnet as swn, wordnet
    from nltk.stem import WordNetLemmatizer
    import spacy
    nltk.download('sentiwordnet', quiet=True)
    nltk.download('wordnet', quiet=True)
    NLP = spacy.load("en_core_web_lg")
except (ImportError, OSError):
    nltk, swn, wordnet, spacy, NLP = None, None, None, None, None
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from flair.models import TextClassifier
from flair.data import Sentence
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from pysentimiento import create_analyzer

CONFIG = {
    "num_to_select": 5,
    "word_options_dir": "src/sentiment/word_options",
    "output_dir": "src/sentiment/lexicons",
    "output_filename": "definitive_lexicons.json",
    "huggingface_models_to_load": [
        "distilbert-base-uncased-finetuned-sst-2-english",
        "nlptown/bert-base-multilingual-uncased-sentiment",
        "finiteautomata/bertweet-base-sentiment-analysis",
        "ProsusAI/finbert",
    ],
    "pysentimiento_model": "pysentimiento/robertuito-sentiment-analysis",
    "enable_wordnet_expansion": True,
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
    "stopwords": {'a', 'an', 'the', 'person', 'and', 'to', 'use', 'very', 'truly', 'really', 'quite', 'so', 'yours'},
    "cohesion_anchor_words": {
        "pos_nouns": ["wonderful person", "kind person", "honorable person", "virtuous person"],
        "neg_nouns": ["terrible person", "cruel person", "selfish person", "toxic person"],
        "pos_verbs": ["helped", "supported", "praised", "rescued", "inspired", "loved"],
        "neg_verbs": ["attacked", "betrayed", "harmed", "deceived", "manipulated"],
        "pos_desc": ["excellent", "perfect", "flawless", "reliable", "effective"],
        "neg_desc": ["terrible", "awful", "broken", "useless", "faulty", "dangerous"]
    }
}

class SentimentAnalyzer:
    def __init__(self, console: Console):
        console.print("[bold cyan]Initializing lightweight, non-transformer models...[/bold cyan]")
        self.console = console
        self.vader = SentimentIntensityAnalyzer()
        self.flair = TextClassifier.load('sentiment')
        self.pysentimiento = create_analyzer(task="sentiment", lang="en")
        
        self.device = -1
        if torch.cuda.is_available():
            self.device = 0
            console.print("[green]CUDA detected. Using GPU for Hugging Face models.[/green]")
        else:
            console.print("[yellow]No CUDA detected. Using CPU for Hugging Face models.[/yellow]")
        console.print("[bold green]Lightweight models loaded.[/bold green]")

    def analyze_with_lightweight_models(self, text: str) -> Dict[str, float]:
        sentiments = {}
        sentiments['vader'] = self.vader.polarity_scores(text)['compound']
        sentiments['textblob'] = TextBlob(text).sentiment.polarity
        flair_sentence = Sentence(text)
        self.flair.predict(flair_sentence)
        flair_score = flair_sentence.labels[0].score
        sentiments['flair'] = flair_score if flair_sentence.labels[0].value == 'POSITIVE' else -flair_score
        pysent_result = self.pysentimiento.predict(text)
        sentiments['pysentimiento'] = pysent_result.probas.get('POS', 0.0) - pysent_result.probas.get('NEG', 0.0)
        if swn:
            tokens = re.findall(r'\b[a-zA-Z-]+\b', text.lower())
            pos_score_total, neg_score_total, count = 0.0, 0.0, 0
            for token in tokens:
                synsets = list(swn.senti_synsets(token))
                if synsets:
                    pos_score_total += np.mean([s.pos_score() for s in synsets])
                    neg_score_total += np.mean([s.neg_score() for s in synsets])
                    count += 1
            sentiments['sentiwordnet'] = (pos_score_total - neg_score_total) / count if count > 0 else 0.0
        return sentiments

    def analyze_with_hf_model(self, texts: List[str], model_name: str) -> Dict[str, float]:
        self.console.print(f"  > Loading model: [bold magenta]{model_name}[/bold magenta]...")
        pipe = pipeline("sentiment-analysis", model=model_name, device=self.device, top_k=None)
        results_map = {}
        for text in tqdm(texts, desc=f"    Analyzing with {model_name.split('/')[-1]}", leave=False, ncols=100):
            result = pipe(text)[0]
            score = 0.0
            if "nlptown" in model_name:
                label = result[0]['label']
                score_val = int(re.search(r'\d+', label).group())
                score = -1 + 2 * (score_val - 1) / 4
            else:
                score_map = {item['label'].lower().replace('pos', 'positive').replace('neg', 'negative').replace('neu', 'neutral'): item['score'] for item in result}
                if "distilbert" in model_name:
                    score = score_map.get(result[0]['label'].lower(), 0.0) * (1 if 'positive' in result[0]['label'].lower() else -1)
                else:
                    score = score_map.get('positive', 0.0) - score_map.get('negative', 0.0)
            results_map[text] = score
        del pipe
        if self.device == 0:
            torch.cuda.empty_cache()
        return results_map

class LexiconOptimizer:
    def __init__(self, candidate_pools: Dict[str, List[str]], analyzer: SentimentAnalyzer, config: Dict):
        self.candidate_pools = candidate_pools
        self.analyzer = analyzer
        self.config = config
        self.lemmatizer = WordNetLemmatizer() if nltk else None
        self.anchor_docs = {cat: NLP(" ".join(words)) for cat, words in config["cohesion_anchor_words"].items()} if NLP else {}

    def _get_key_words(self, text: str) -> Set[str]:
        words = re.findall(r'\b[a-zA-Z-]+\b', text.lower())
        lemmatized = [self.lemmatizer.lemmatize(w, pos='v') for w in words]
        return {w for w in lemmatized if w not in self.config["stopwords"]}

    def _is_grammatically_valid(self, text: str, category_type: str) -> bool:
        if not NLP: return True
        doc = NLP(text)
        if category_type == "verbs":
            return any(token.pos_.startswith("VERB") for token in doc)
        if category_type == "desc":
            last_word = next((token for token in reversed(doc) if not token.is_punct), None)
            return last_word is not None and last_word.pos_ == "ADJ"
        if category_type == "nouns":
            return not any(token.pos_ == "ADV" for token in doc)
        return True

    def _get_cohesion_score(self, text: str, category: str) -> float:
        if not NLP or category not in self.anchor_docs: return 0.5
        doc = NLP(text)
        anchor_doc = self.anchor_docs[category]
        if not doc.has_vector or not anchor_doc.has_vector or doc.vector_norm == 0 or anchor_doc.vector_norm == 0:
            return 0.5
        return 1.0 - doc.similarity(anchor_doc)

    def _normalize_values(self, values: List[float]) -> np.ndarray:
        arr = np.array(values)
        min_val, max_val = arr.min(), arr.max()
        return np.zeros_like(arr) if max_val == min_val else (arr - min_val) / (max_val - min_val)

    def _expand_candidates_with_wordnet(self, initial_candidates: Set[str]) -> Set[str]:
        if not wordnet: return set()
        new_candidates = set()
        for candidate_phrase in list(initial_candidates)[:20]:
            key_words = list(self._get_key_words(candidate_phrase))
            if not key_words: continue
            base_word = key_words[-1]
            try:
                template = re.sub(r'\b' + re.escape(base_word) + r'\b', '{}', candidate_phrase, 1)
            except re.error:
                continue
            for syn in wordnet.synsets(base_word):
                for lemma in syn.lemmas():
                    new_word = lemma.name().replace('_', ' ').lower()
                    if new_word != base_word and len(new_word.split()) <= 2:
                        new_phrase = template.format(new_word)
                        new_candidates.add(new_phrase)
        return {c for c in new_candidates if c not in initial_candidates and len(c) > 2}

    def run(self) -> Dict[str, Any]:
        final_lexicons = {}
        global_used_lemmas = defaultdict(set)
        
        all_scored_candidates_by_cat = {}

        for category, items in self.candidate_pools.items():
            category_type = category.split('_')[1]
            candidate_set = {item.strip() for item in items if item}
            if self.config["enable_wordnet_expansion"] and nltk:
                new_words = self._expand_candidates_with_wordnet(candidate_set)
                candidate_set.update(new_words)
            valid_candidate_list = [c for c in list(candidate_set) if self._is_grammatically_valid(c, category_type)]
            
            self.analyzer.console.print(f"[yellow]Analyzing sentiment for category: [bold]{category}[/bold] ({len(valid_candidate_list)} items)...[/yellow]")
            if not valid_candidate_list:
                all_scored_candidates_by_cat[category] = []
                continue

            all_scores_by_text = defaultdict(dict)
            for text in tqdm(valid_candidate_list, desc="    Analyzing with lightweight models", leave=False, ncols=100):
                all_scores_by_text[text].update(self.analyzer.analyze_with_lightweight_models(text))

            for model_name in self.config["huggingface_models_to_load"]:
                hf_results = self.analyzer.analyze_with_hf_model(valid_candidate_list, model_name)
                for text, score in hf_results.items():
                    all_scores_by_text[text][model_name.split('/')[-1]] = score
            
            scored_candidates = []
            for text, scores_dict in all_scores_by_text.items():
                if not scores_dict: continue
                scores = list(scores_dict.values())
                scored_candidates.append({
                    "text": text,
                    "combined_score": np.mean([np.median(scores), np.mean(scores)]),
                    "standard_deviation": np.std(scores)
                })
            all_scored_candidates_by_cat[category] = scored_candidates

        for category, scored_candidates in all_scored_candidates_by_cat.items():
            category_type = category.split('_')[1]
            final_results = defaultdict(list)
            
            # Use a copy of the candidates that we can safely remove items from
            remaining_candidates = list(scored_candidates)

            # Process from most intense to least intense to give them priority
            for level in self.config["valence_map"].keys():
                lax_cfg = self.config["LAXNESS_CONFIG"]
                
                for i in range(lax_cfg["max_passes"]):
                    if len(final_results[level]) >= self.config["num_to_select"]:
                        break
                    
                    score_leniency = i * lax_cfg["score_leniency_step"]
                    sd_threshold = lax_cfg["initial_sd_threshold"] + (i * lax_cfg["sd_leniency_step"])
                    
                    level_range = self.config["valence_map"][level]["range"]
                    min_score = max(0, level_range[0] - score_leniency)
                    max_score = min(1.0, level_range[1] + score_leniency)

                    # Find candidates within the current (potentially lax) range
                    potential_pool = [
                        cand for cand in remaining_candidates
                        if min_score <= abs(cand['combined_score']) < max_score and cand['standard_deviation'] <= sd_threshold
                    ]
                    
                    if not potential_pool:
                        continue

                    # Calculate goodness for this specific pool
                    distances = [abs(abs(c['combined_score']) - self.config["valence_map"][level]["midpoint"]) for c in potential_pool]
                    stds = [c['standard_deviation'] for c in potential_pool]
                    cohesions = [self._get_cohesion_score(c['text'], category) for c in potential_pool]
                    
                    norm_distances = self._normalize_values(distances)
                    norm_stds = self._normalize_values(stds)
                    norm_cohesions = self._normalize_values(cohesions)
                    
                    for j, cand in enumerate(potential_pool):
                        cand['goodness'] = (norm_cohesions[j] * 0.5) + (norm_stds[j] * 0.3) + (norm_distances[j] * 0.2)
                    
                    sorted_pool = sorted(potential_pool, key=lambda x: x.get('goodness', 999))
                    
                    for cand in sorted_pool:
                        if len(final_results[level]) >= self.config["num_to_select"]:
                            break
                        key_lemmas = self._get_key_words(cand['text'])
                        if not key_lemmas: continue
                        
                        is_new_lemma = not any(k.intersection(global_used_lemmas[category_type]) for k in [key_lemmas])
                        if is_new_lemma:
                            final_results[level].append(cand)
                            global_used_lemmas[category_type].update(key_lemmas)
                            # Remove the selected candidate from the main pool so it can't be chosen again
                            remaining_candidates.remove(cand)

            final_lexicons[category] = final_results
        
        return final_lexicons

    def format_output(self, optimized_lexicons: Dict, source_name: str) -> str:
        num_select = self.config['num_to_select']
        output_str = f"\n# DEFINITIVE SENTIMENT LEXICONS (Top {num_select} - Complete)\n# Source: {source_name}\n"
        for category, levels in optimized_lexicons.items():
            output_str += f"\n{category} = {{\n"
            for level in self.config["valence_map"].keys():
                output_str += f'    "{level}": [\n'
                items = levels.get(level, [])
                if items:
                    is_negative = "neg" in category
                    sorted_items = sorted(items, key=lambda x: x['combined_score'], reverse=not is_negative)
                    for item in sorted_items:
                        score, std = item['combined_score'], item['standard_deviation']
                        output_str += f'        # Score: {score:.2f}, SD: {std:.2f}\n'
                        output_str += f'        "{item["text"]}",\n'
                else:
                    output_str += '        # (No suitable candidates found)\n'
                output_str += '    ],\n'
            output_str += "}}"
        return output_str

def load_candidate_pools(directory: str, console: Console) -> Dict[str, List[str]]:
    candidate_pools = {}
    try:
        text_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    except FileNotFoundError:
        console.print(f"[bold red]Error: Directory not found -> '{directory}'[/bold red]")
        sys.exit(1)
    for text_file in text_files:
        path = os.path.join(directory, text_file)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            try:
                word_list = ast.literal_eval(content)
                if not isinstance(word_list, list):
                    raise ValueError
            except (ValueError, SyntaxError):
                word_list = [line.strip().strip("'\",") for line in content.splitlines() if line.strip() and not line.strip().startswith('#')]
            # Remove empty strings and deduplicate
            word_list = [w for w in word_list if w]
            candidate_pools[text_file.replace(".txt", "")] = word_list
    return candidate_pools

def load_one_file_pool(filepath: str, console: Console) -> List[str]:
    if not os.path.exists(filepath):
        console.print(f"[bold red]Error: File not found -> '{filepath}'[/bold red]")
        sys.exit(1)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        try:
            word_list = ast.literal_eval(content)
            if not isinstance(word_list, list):
                raise ValueError
        except (ValueError, SyntaxError):
            word_list = [line.strip().strip("'\",") for line in content.splitlines() if line.strip() and not line.strip().startswith('#')]
        word_list = [w for w in word_list if w]
    # Always return a dict with the same key format as load_candidate_pools
    key = os.path.basename(filepath).replace(".txt", "")
    return {key: word_list}

def save_results(lexicons: Dict, config: Dict, source_name: str):
    lexicons_for_json = {
        cat: {
            level: [
                {k: v for k, v in item.items() if k in ['text', 'combined_score', 'standard_deviation']}
                for item in items
            ]
            for level, items in levels.items()
        }
        for cat, levels in lexicons.items()
    }
    output_dir = config["output_dir"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, config["output_filename"])
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(lexicons_for_json, f, indent=4)
    console.print(f"[bold green]Optimized lexicons saved to {filepath}[/bold green]")

def main():
    if not all(x is not None for x in [nltk, spacy, NLP]):
        console.print("[bold red]Required NLP libraries not found or model not available.[/bold red]")
        console.print("WordNet/SpaCy features will be disabled. Functionality will be limited.")
        console.print("Please run: pip install nltk spacy torch")
        console.print("Then download the SpaCy model: python -m spacy download en_core_web_lg")
        CONFIG["enable_wordnet_expansion"] = False
        
    candidate_pools = load_candidate_pools(CONFIG["word_options_dir"], console)
    candidate_pools = load_one_file_pool('src/sentiment/word_options/pos_verbs.txt', console)
    analyzer = SentimentAnalyzer(console)
    optimizer = LexiconOptimizer(candidate_pools, analyzer, CONFIG)
    optimized_lexicons = optimizer.run()
    
    save_results(optimized_lexicons, CONFIG, "Enriched Source Files + Laxness Algorithm")
    console.print(optimizer.format_output(optimized_lexicons, "Enriched Source Files + Laxness Algorithm"))

    num_to_select = CONFIG['num_to_select']
    missing_count = 0
    for cat, levels in optimized_lexicons.items():
        for level, items in levels.items():
            if len(items) < num_to_select:
                missing_count +=1
                console.print(f"  [red]Still missing for {cat} -> {level}[/red]: only found {len(items)} of {num_to_select}")
    if missing_count == 0:
        console.print("\n[bold green]All categories and levels have been successfully filled![/bold green]")


if __name__ == "__main__":
    main()