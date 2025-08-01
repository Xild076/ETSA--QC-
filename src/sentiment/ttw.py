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
import math

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
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from pysentimiento import create_analyzer

CONFIG = {
    "num_to_select": 5,
    "word_options_dir": "src/sentiment/word_options",
    "output_dir": "src/sentiment/lexicons",
    "output_filename": "optimized_lexicon.json",
    "huggingface_models_to_load": [
        "nlptown/bert-base-multilingual-uncased-sentiment",
        "finiteautomata/bertweet-base-sentiment-analysis",
        "ProsusAI/finbert",
    ],
    "logit_based_models": [
        "distilbert-base-uncased-finetuned-sst-2-english"
    ],
    "pysentimiento_model": "pysentimiento/robertuito-sentiment-analysis",
    "perplexity_model": "gpt2",
    "MIN_WORD_FREQUENCY": 1e-6,
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
    "GOODNESS_WEIGHTS": {
        "distance": 0.30,
        "std_dev": 0.25,
        "cohesion": 0.15,
        "perplexity": 0.20,
        "brevity": 0.10
    },
    "VALIDATION_CONFIG": {
        "verb_ppl_threshold_multiplier": 5.0,
        "verb_forbidden_pos": {"NOUN", "ADP", "SCONJ"}
    },
    "DYNAMIC_INTENSIFIER_CONFIG": {
        "amplification_factor": 1.5,
        "dampening_factor": 0.7,
    },
    "stopwords": {'a', 'an', 'the', 'person', 'and', 'to', 'use', 'very', 'truly', 'really', 'quite', 'so', 'yours', 'soul', 'mortal', 'individual'},
    "intensifiers": {'repeatedly', 'constantly', 'deeply', 'harshly', 'openly', 'brutally', 'utterly', 'truly', 'really', 'very', 'incredibly', 'extremely', 'slightly', 'somewhat'},
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
    def __init__(self, console: Console, hf_models: List[str], logit_models_list: List[str], perplexity_model_name: str):
        self.console = console
        self.device = 0 if torch.cuda.is_available() else -1
        console.print("[bold cyan]Initializing sentiment models...[/bold cyan]")
        self.vader = SentimentIntensityAnalyzer()
        self.flair = TextClassifier.load('sentiment')
        self.pysentimiento = create_analyzer(task="sentiment", lang="en")
        
        self.hf_pipelines = {}
        for model_name in hf_models:
            console.print(f"  > Loading Hugging Face pipeline: [bold magenta]{model_name}[/bold magenta]...")
            try:
                self.hf_pipelines[model_name] = pipeline("sentiment-analysis", model=model_name, device=self.device, top_k=None)
            except Exception as e:
                console.print(f"[bold red]Failed to load model {model_name}: {e}[/bold red]")

        self.logit_models = {}
        for model_name in logit_models_list:
            console.print(f"  > Loading Hugging Face model for logit extraction: [bold magenta]{model_name}[/bold magenta]...")
            try:
                model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device if self.device != -1 else 'cpu')
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.logit_models[model_name] = {'model': model, 'tokenizer': tokenizer}
            except Exception as e:
                console.print(f"[bold red]Failed to load model {model_name}: {e}[/bold red]")

        console.print(f"  > Loading Perplexity model: [bold magenta]{perplexity_model_name}[/bold magenta]...")
        try:
            self.perplexity_model = AutoModelForCausalLM.from_pretrained(perplexity_model_name).to(self.device if self.device != -1 else 'cpu')
            self.perplexity_tokenizer = AutoTokenizer.from_pretrained(perplexity_model_name)
        except Exception as e:
            console.print(f"[bold red]Failed to load perplexity model {perplexity_model_name}: {e}[/bold red]")
            self.perplexity_model = None
        console.print("[bold green]All sentiment models initialized.[/bold green]")

    def _get_hf_pipeline_score(self, result: List[Dict[str, Any]], model_name: str) -> float:
        score_map = {item['label'].lower().replace('pos', 'positive').replace('neg', 'negative').replace('neu', 'neutral'): item['score'] for item in result}
        if "nlptown" in model_name:
            score_val = int(re.search(r'\d+', result[0]['label']).group())
            return -1.0 + (2.0 * (score_val - 1) / 4.0)
        else:
            return score_map.get('positive', 0.0) - score_map.get('negative', 0.0)
    
    def _get_hf_logit_score(self, text: str, model_info: Dict) -> float:
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits
        
        id2label = model.config.id2label
        neg_idx, pos_idx = -1, -1
        for i, label in id2label.items():
            if label.lower() in ['negative', 'neg']: neg_idx = i
            if label.lower() in ['positive', 'pos']: pos_idx = i
        
        if pos_idx == -1 or neg_idx == -1: return 0.0

        logit_diff = logits[0, pos_idx] - logits[0, neg_idx]
        return torch.tanh(logit_diff).item()

    def _get_perplexity(self, text: str) -> float:
        if not self.perplexity_model or not text:
            return 100000.0

        encodings = self.perplexity_tokenizer(text, return_tensors='pt').to(self.perplexity_model.device)
        max_length = self.perplexity_model.config.n_positions
        stride = 512
        seq_len = encodings.input_ids.size(1)

        nlls = []
        with torch.no_grad():
            for i in range(0, seq_len, stride):
                begin_loc = max(i + stride - max_length, 0)
                end_loc = min(i + stride, seq_len)
                trg_len = end_loc - i
                input_ids = encodings.input_ids[:, begin_loc:end_loc]
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                outputs = self.perplexity_model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len
                nlls.append(neg_log_likelihood)
        
        ppl = torch.exp(torch.stack(nlls).sum() / end_loc).item()
        return ppl if not (math.isinf(ppl) or math.isnan(ppl)) else 100000.0

    def analyze(self, texts: List[str]) -> Dict[str, Dict[str, float]]:
        all_results = defaultdict(dict)
        for text in tqdm(texts, desc="Analyzing (lightweight models & perplexity)", leave=False, ncols=120):
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
            
            all_results[text]['perplexity'] = self._get_perplexity(text)

        for model_name, pipe in self.hf_pipelines.items():
            model_short_name = model_name.split('/')[-1]
            try:
                hf_outputs = pipe(texts, batch_size=32, truncation=True)
                for text, result in zip(texts, hf_outputs):
                    all_results[text][model_short_name] = self._get_hf_pipeline_score(result, model_name)
            except Exception as e:
                self.console.print(f"[bold red]Error with {model_name}: {e}[/bold red]")
                for text in texts: all_results[text][model_short_name] = 0.0

        for model_name, model_info in self.logit_models.items():
            model_short_name = model_name.split('/')[-1]
            for text in tqdm(texts, desc=f"Analyzing ({model_short_name} logits)", leave=False, ncols=120):
                try:
                    all_results[text][model_short_name] = self._get_hf_logit_score(text, model_info)
                except Exception as e:
                    self.console.print(f"[bold red]Error with {model_name} logit scoring: {e}[/bold red]")
                    all_results[text][model_short_name] = 0.0

        return all_results

class LexiconOptimizer:
    def __init__(self, candidate_pools: Dict[str, List[str]], analyzer: SentimentAnalyzer, config: Dict):
        self.candidate_pools = candidate_pools
        self.analyzer = analyzer
        self.config = config
        self.anchor_docs = {cat: NLP(" ".join(words)) for cat, words in config["cohesion_anchor_words"].items()} if NLP else {}
        self._baseline_ppl = None

    def _get_key_lemmas(self, text: str) -> Set[str]:
        if not NLP: return {w for w in re.findall(r'\b[a-zA-Z-]+\b', text.lower()) if w not in self.config["stopwords"]}
        doc = NLP(text.lower())
        return {token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.text not in self.config["stopwords"]}

    def _is_common_enough(self, text: str) -> bool:
        if not word_frequency: return True
        words = re.findall(r'\b[a-zA-Z-]+\b', text.lower())
        if not words: return False
        return all(word_frequency(w, 'en') >= self.config["MIN_WORD_FREQUENCY"] for w in words if w not in self.config["stopwords"])

    def _is_grammatically_valid(self, text: str, category_type: str, analyzer: SentimentAnalyzer) -> Tuple[bool, str]:
        if not NLP: return True, text
        
        if category_type == "verbs":
            doc_phrase = NLP(text)
            forbidden_pos = self.config["VALIDATION_CONFIG"]["verb_forbidden_pos"]
            if any(token.pos_ in forbidden_pos for token in doc_phrase):
                return False, text

            test_sentence = f"Someone {text} someone."
            if self._baseline_ppl is None:
                self._baseline_ppl = analyzer._get_perplexity("Someone helped someone.")
            
            ppl = analyzer._get_perplexity(test_sentence)
            threshold = self._baseline_ppl * self.config["VALIDATION_CONFIG"]["verb_ppl_threshold_multiplier"]
            if ppl > threshold:
                return False, text

            doc = NLP(test_sentence)
            root_verb = next((tok for tok in doc if tok.dep_ == "ROOT"), None)
            if not root_verb or root_verb.pos_ != "VERB":
                return False, text
            
            if root_verb.tag_ != "VBD":
                try:
                    past_tense_verb = TextBlob(root_verb.lemma_).words[0].lemmatize("v")
                except:
                    past_tense_verb = root_verb.lemma_
                
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

    def _calculate_dynamic_intensified_score(self, base_score, intensifier_score):
        cfg = self.config["DYNAMIC_INTENSIFIER_CONFIG"]
        if base_score * intensifier_score >= 0:
            final_score = base_score * (1 + abs(intensifier_score) * cfg["amplification_factor"])
        else:
            final_score = base_score * (1 - abs(intensifier_score) * cfg["dampening_factor"])
        return np.clip(final_score, -1.0, 1.0)

    def run(self) -> Dict[str, Any]:
        final_lexicons = defaultdict(lambda: defaultdict(list))
        global_used_lemmas = defaultdict(set)
        
        phrases_to_analyze = set()
        processed_pools = {}
        intensifiers = self.config.get("intensifiers", set())
        phrases_to_analyze.update(intensifiers)

        for category, items in self.candidate_pools.items():
            category_type = category.split('_')[1]
            candidate_set = {item.strip() for item in items if item}
            
            valid_candidates = set()
            for cand in candidate_set:
                is_valid, normalized_cand = self._is_grammatically_valid(cand, category_type, self.analyzer)
                if is_valid and self._is_common_enough(normalized_cand):
                    valid_candidates.add(normalized_cand)
            processed_pools[category] = valid_candidates
            phrases_to_analyze.update(valid_candidates)

        templates = {
            "nouns": "This is about {}.",
            "verbs": "They {} them.",
            "desc": "The item was {}."
        }
        for category, cands in processed_pools.items():
            cat_type = category.split('_')[1]
            template = templates.get(cat_type)
            if template:
                for cand in cands:
                    phrases_to_analyze.add(template.format(cand))

        unique_candidate_list = list(phrases_to_analyze)
        if not unique_candidate_list:
            self.analyzer.console.print("[bold yellow]No valid candidates found to analyze.[/bold yellow]")
            return {}
            
        self.analyzer.console.print(f"[yellow]Analyzing sentiment for [bold]{len(unique_candidate_list)}[/bold] total phrases...[/yellow]")
        master_scores_map = self.analyzer.analyze(unique_candidate_list)
        
        all_scored_candidates_by_cat = defaultdict(list)
        for category, candidate_set in processed_pools.items():
            category_type = category.split('_')[1]
            template = templates.get(category_type)

            for text in candidate_set:
                templated_text = template.format(text) if template else text
                templated_scores_dict = master_scores_map.get(templated_text)
                if not templated_scores_dict: continue

                words = text.split()
                intensifier_words = [word for word in words if word in intensifiers]
                base_words = [word for word in words if word not in intensifiers]
                
                final_score = 0.0
                if not base_words:
                    base_text = text
                else:
                    base_text = " ".join(base_words)
                
                base_templated_text = template.format(base_text) if template else base_text
                base_templated_scores_dict = master_scores_map.get(base_templated_text)
                if not base_templated_scores_dict: continue
                
                base_scores = [v for k, v in base_templated_scores_dict.items() if k != 'perplexity']
                if not base_scores: continue
                base_score = np.mean([np.median(base_scores), np.mean(base_scores)])

                if intensifier_words:
                    intensifier_scores_list = []
                    for intensifier in intensifier_words:
                        intensifier_score_dict = master_scores_map.get(intensifier)
                        if intensifier_score_dict:
                            intensifier_all_scores = [v for k, v in intensifier_score_dict.items() if k != 'perplexity']
                            if intensifier_all_scores:
                                intensifier_scores_list.append(np.mean([np.median(intensifier_all_scores), np.mean(intensifier_all_scores)]))
                    
                    if intensifier_scores_list:
                        avg_intensifier_score = np.mean(intensifier_scores_list)
                        final_score = self._calculate_dynamic_intensified_score(base_score, avg_intensifier_score)
                    else:
                        final_score = base_score
                else:
                    final_score = base_score
                
                templated_scores = [v for k,v in templated_scores_dict.items() if k != 'perplexity']
                all_scored_candidates_by_cat[category].append({
                    "text": text,
                    "combined_score": final_score,
                    "standard_deviation": np.std(templated_scores) if templated_scores else 0.0,
                    "perplexity": master_scores_map.get(text, {}).get('perplexity', 100000.0),
                    "word_count": len(words)
                })
        
        for category, scored_candidates in all_scored_candidates_by_cat.items():
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
                    perplexities = [c.get('perplexity', 100000.0) for c in potential_pool]
                    word_counts = [c.get('word_count', 10) for c in potential_pool]
                    
                    norm_distances = self._normalize_values(distances)
                    norm_stds = self._normalize_values(stds)
                    norm_cohesions = self._normalize_values(cohesions)
                    norm_perplexities = self._normalize_values(perplexities)
                    norm_brevity = self._normalize_values(word_counts)
                    
                    weights = self.config["GOODNESS_WEIGHTS"]
                    for j, cand in enumerate(potential_pool):
                        cand['goodness'] = (norm_distances[j] * weights['distance']) + \
                                           (norm_stds[j] * weights['std_dev']) + \
                                           ((1 - norm_cohesions[j]) * weights['cohesion']) + \
                                           (norm_perplexities[j] * weights['perplexity']) + \
                                           (norm_brevity[j] * weights['brevity'])
                    
                    for cand in sorted(potential_pool, key=lambda x: x['goodness']):
                        if len(final_lexicons[category][level]) >= self.config["num_to_select"]: break
                        key_lemmas = self._get_key_lemmas(cand['text'])
                        if not key_lemmas: continue
                        
                        category_type = category.split('_')[1]
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
                        ppl = item.get('perplexity', 'N/A')
                        wc = item.get('word_count', '?')
                        ppl_str = f"{ppl:.2f}" if isinstance(ppl, float) else str(ppl)
                        output_str += f'        # Score: {score:.3f}, SD: {std:.3f}, PPL: {ppl_str}, WC: {wc}\n'
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
    lexicons_for_json = {cat: {level: [{k: v for k, v in item.items() if k in ['text', 'combined_score', 'standard_deviation', 'perplexity', 'word_count']} for item in items] for level, items in levels.items()} for cat, levels in lexicons.items()}
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
    
    analyzer = SentimentAnalyzer(console, CONFIG["huggingface_models_to_load"], CONFIG["logit_based_models"], CONFIG["perplexity_model"])
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