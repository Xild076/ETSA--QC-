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
from textblob import TextBlob
from sentiment import (
    get_vader_sentiment, get_textblob_sentiment, get_flair_sentiment,
    get_pysentimiento_sentiment, get_swn_sentiment, get_nlptown_sentiment,
    get_finiteautomata_sentiment, get_ProsusAI_sentiment, get_distilbert_logit_sentiment
)

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
from transformers import AutoModelForCausalLM, AutoTokenizer

CONFIG = {
    "num_to_select": 5,
    "word_options_dir": "src/sentiment/word_options",
    "output_dir": "src/sentiment/lexicons",
    "output_filename": "optimized_lexicon.json",
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
    "PROBLEMATIC_WORDS_FILTER": {
        "words": ["unthinkable"],
        "max_word_count": 4,  # Filter phrases longer than 4 words
        "min_sentiment_confidence": 0.15,  # Filter words with very low sentiment confidence
        "max_perplexity_threshold": 50000.0  # Filter words that are too unusual/nonsensical
    },
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
    def __init__(self, console: Console, config: Dict = None):
        self.console = console
        self.config = config or CONFIG
        console.print("[bold cyan]Initializing sentiment models...[/bold cyan]")
        
        try:
            self.perplexity_model = AutoModelForCausalLM.from_pretrained("gpt2")
            self.perplexity_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            console.print("[bold green]Perplexity model loaded.[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Failed to load perplexity model: {e}[/bold red]")
            self.perplexity_model = None
        
        self._models_initialized = False
        self._sentiment_models = {}

    def _initialize_models(self):
        if self._models_initialized:
            return
        
        self.console.print("[bold cyan]Loading batch sentiment models...[/bold cyan]")
        
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._sentiment_models['vader'] = SentimentIntensityAnalyzer()
            self.console.print("[green]✓ VADER loaded[/green]")
        except Exception as e:
            self.console.print(f"[red]✗ VADER failed: {e}[/red]")
        
        try:
            from flair.models import TextClassifier
            self._sentiment_models['flair'] = TextClassifier.load('sentiment')
            self.console.print("[green]✓ Flair loaded[/green]")
        except Exception as e:
            self.console.print(f"[red]✗ Flair failed: {e}[/red]")
        
        try:
            from pysentimiento import create_analyzer
            self._sentiment_models['pysentimiento'] = create_analyzer(task="sentiment", lang="en")
            self.console.print("[green]✓ PySentimiento loaded[/green]")
        except Exception as e:
            self.console.print(f"[red]✗ PySentimiento failed: {e}[/red]")
        
        try:
            from transformers import pipeline
            self._sentiment_models['nlptown'] = pipeline("sentiment-analysis", 
                                                        model="nlptown/bert-base-multilingual-uncased-sentiment")
            self.console.print("[green]✓ NLPTown loaded[/green]")
        except Exception as e:
            self.console.print(f"[red]✗ NLPTown failed: {e}[/red]")
        
        try:
            self._sentiment_models['finiteautomata'] = pipeline("sentiment-analysis", 
                                                              model="finiteautomata/bertweet-base-sentiment-analysis")
            self.console.print("[green]✓ FiniteAutomata loaded[/green]")
        except Exception as e:
            self.console.print(f"[red]✗ FiniteAutomata failed: {e}[/red]")
        
        try:
            self._sentiment_models['prosusai'] = pipeline("sentiment-analysis", 
                                                         model="ProsusAI/finbert")
            self.console.print("[green]✓ ProsusAI loaded[/green]")
        except Exception as e:
            self.console.print(f"[red]✗ ProsusAI failed: {e}[/red]")
        
        try:
            self._sentiment_models['distilbert'] = pipeline("sentiment-analysis", 
                                                           model="distilbert-base-uncased-finetuned-sst-2-english")
            self.console.print("[green]✓ DistilBERT loaded[/green]")
        except Exception as e:
            self.console.print(f"[red]✗ DistilBERT failed: {e}[/red]")
        
        self._models_initialized = True
        self.console.print("[bold green]All models initialized![/bold green]")

    def _batch_vader_sentiment(self, texts: List[str]) -> List[float]:
        if 'vader' not in self._sentiment_models:
            return [get_vader_sentiment(text) for text in texts]
        
        analyzer = self._sentiment_models['vader']
        results = []
        for text in texts:
            if not text or not text.strip():
                results.append(0.0)
            else:
                result = analyzer.polarity_scores(text)
                results.append(result['compound'])
        return results

    def _batch_textblob_sentiment(self, texts: List[str]) -> List[float]:
        results = []
        for text in texts:
            if not text or not text.strip():
                results.append(0.0)
            else:
                blob = TextBlob(text)
                results.append(blob.sentiment.polarity)
        return results

    def _batch_flair_sentiment(self, texts: List[str]) -> List[float]:
        if 'flair' not in self._sentiment_models:
            return [get_flair_sentiment(text) for text in texts]
        
        from flair.data import Sentence
        model = self._sentiment_models['flair']
        results = []
        
        for text in texts:
            if not text or not text.strip():
                results.append(0.0)
            else:
                sentence = Sentence(text)
                model.predict(sentence)
                label = sentence.labels[0]
                confidence_score = label.score
                if label.value.upper() in ['NEGATIVE', 'NEG']:
                    confidence_score = -confidence_score
                results.append(confidence_score * 2 - 1 if confidence_score >= 0 else confidence_score * 2 + 1)
        return results

    def _batch_pysentimiento_sentiment(self, texts: List[str]) -> List[float]:
        if 'pysentimiento' not in self._sentiment_models:
            return [get_pysentimiento_sentiment(text) for text in texts]
        
        analyzer = self._sentiment_models['pysentimiento']
        results = []
        for text in texts:
            if not text or not text.strip():
                results.append(0.0)
            else:
                pysent_result = analyzer.predict(text)
                results.append(pysent_result.probas.get('POS', 0.0) - pysent_result.probas.get('NEG', 0.0))
        return results

    def _batch_swn_sentiment(self, texts: List[str]) -> List[float]:
        if not swn:
            return [0.0] * len(texts)
        
        results = []
        for text in texts:
            tokens = re.findall(r'\b[a-zA-Z-]+\b', text.lower())
            pos, neg, count = 0.0, 0.0, 0
            for token in tokens:
                synsets = list(swn.senti_synsets(token))
                if synsets:
                    pos += np.mean([s.pos_score() for s in synsets])
                    neg += np.mean([s.neg_score() for s in synsets])
                    count += 1
            results.append((pos / count) - (neg / count) if count > 0 else 0.0)
        return results

    def _batch_pipeline_sentiment(self, texts: List[str], model_key: str, model_name: str) -> List[float]:
        if model_key not in self._sentiment_models:
            fallback_func = {
                'nlptown': get_nlptown_sentiment,
                'finiteautomata': get_finiteautomata_sentiment,
                'prosusai': get_ProsusAI_sentiment,
                'distilbert': get_distilbert_logit_sentiment
            }.get(model_key, lambda x: 0.0)
            return [fallback_func(text) for text in texts]
        
        model = self._sentiment_models[model_key]
        valid_texts = [text if text and text.strip() else "neutral" for text in texts]
        
        try:
            batch_results = model(valid_texts, batch_size=32, truncation=True)
            results = []
            
            for i, result in enumerate(batch_results):
                if not texts[i] or not texts[i].strip():
                    results.append(0.0)
                    continue
                
                if model_name == "nlptown":
                    star_rating = int(result['label'].split()[0])
                    normalized_score = (star_rating - 3) / 2.0
                    results.append(normalized_score)
                elif model_name in ["finiteautomata", "prosusai"]:
                    confidence_score = result['score']
                    if result['label'].upper() in ['NEGATIVE', 'NEG']:
                        confidence_score = -confidence_score
                    results.append(confidence_score * 2 - 1 if confidence_score >= 0 else confidence_score * 2 + 1)
                else:  # distilbert
                    confidence_score = result['score']
                    if result['label'].upper() in ['NEGATIVE', 'NEG']:
                        confidence_score = -confidence_score
                    results.append(confidence_score * 2 - 1 if confidence_score >= 0 else confidence_score * 2 + 1)
            
            return results
        except Exception as e:
            self.console.print(f"[red]Batch processing failed for {model_name}, falling back to individual processing: {e}[/red]")
            fallback_func = {
                'nlptown': get_nlptown_sentiment,
                'finiteautomata': get_finiteautomata_sentiment,
                'prosusai': get_ProsusAI_sentiment,
                'distilbert': get_distilbert_logit_sentiment
            }.get(model_key, lambda x: 0.0)
            return [fallback_func(text) for text in texts]

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
        self._initialize_models()
        all_results = defaultdict(dict)
        
        self.console.print(f"[yellow]Processing {len(texts)} texts with batch analysis...[/yellow]")
        
        # Batch process all sentiment models
        vader_scores = self._batch_vader_sentiment(texts)
        textblob_scores = self._batch_textblob_sentiment(texts)
        flair_scores = self._batch_flair_sentiment(texts)
        pysentimiento_scores = self._batch_pysentimiento_sentiment(texts)
        swn_scores = self._batch_swn_sentiment(texts) if swn else [0.0] * len(texts)
        nlptown_scores = self._batch_pipeline_sentiment(texts, 'nlptown', 'nlptown')
        finiteautomata_scores = self._batch_pipeline_sentiment(texts, 'finiteautomata', 'finiteautomata')
        prosusai_scores = self._batch_pipeline_sentiment(texts, 'prosusai', 'prosusai')
        distilbert_scores = self._batch_pipeline_sentiment(texts, 'distilbert', 'distilbert')
        
        # Process perplexity with progress bar (this is still slow)
        self.console.print("[yellow]Computing perplexity scores...[/yellow]")
        perplexity_scores = []
        for text in tqdm(texts, desc="Perplexity", leave=False, ncols=120):
            perplexity_scores.append(self._get_perplexity(text))
        
        # Combine all results
        for i, text in enumerate(texts):
            all_results[text]['vader'] = vader_scores[i]
            all_results[text]['textblob'] = textblob_scores[i]
            all_results[text]['flair'] = flair_scores[i]
            all_results[text]['pysentimiento'] = pysentimiento_scores[i]
            if swn:
                all_results[text]['sentiwordnet'] = swn_scores[i]
            all_results[text]['nlptown'] = nlptown_scores[i]
            all_results[text]['finiteautomata'] = finiteautomata_scores[i]
            all_results[text]['ProsusAI'] = prosusai_scores[i]
            all_results[text]['distilbert'] = distilbert_scores[i]
            all_results[text]['perplexity'] = perplexity_scores[i]

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

    def _is_content_appropriate(self, text: str) -> bool:
        text_lower = text.lower()
        words = set(re.findall(r'\b[a-zA-Z-]+\b', text_lower))
        
        if any(word in self.config["PROBLEMATIC_WORDS_FILTER"]['words'] for word in words):
            self.analyzer.console.print(f"[red]Filtered out bad content: {text}[/red]")
            return False
        
        return True

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
        
        self.analyzer.console.print("[yellow]Applied Content Filters:[/yellow]")
        filter_config = self.config["PROBLEMATIC_WORDS_FILTER"]
        self.analyzer.console.print(f"  • Max word count: {filter_config['max_word_count']}")
        self.analyzer.console.print(f"  • Min sentiment confidence: {filter_config['min_sentiment_confidence']}")
        self.analyzer.console.print(f"  • Max perplexity threshold: {filter_config['max_perplexity_threshold']}")
        
        phrases_to_analyze = set()
        processed_pools = {}
        intensifiers = self.config.get("intensifiers", set())
        phrases_to_analyze.update(intensifiers)

        for category, items in self.candidate_pools.items():
            category_type = category.split('_')[1]
            candidate_set = {item.strip() for item in items if item}
            original_count = len(candidate_set)
            
            valid_candidates = set()
            for cand in candidate_set:
                is_valid, normalized_cand = self._is_grammatically_valid(cand, category_type, self.analyzer)
                if (is_valid and 
                    self._is_common_enough(normalized_cand) and 
                    self._is_content_appropriate(normalized_cand)):
                    valid_candidates.add(normalized_cand)
                    
            filtered_count = len(valid_candidates)
            if original_count > filtered_count:
                self.analyzer.console.print(f"  [dim]{category}: {original_count} → {filtered_count} candidates (filtered {original_count - filtered_count})[/dim]")
                
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
                
                min_confidence = self.config["PROBLEMATIC_WORDS_FILTER"]["min_sentiment_confidence"]
                if abs(base_score) < min_confidence:
                    continue

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
                perplexity_score = master_scores_map.get(text, {}).get('perplexity', 100000.0)
                
                max_perplexity = self.config["PROBLEMATIC_WORDS_FILTER"]["max_perplexity_threshold"]
                if perplexity_score > max_perplexity:
                    continue
                    
                all_scored_candidates_by_cat[category].append({
                    "text": text,
                    "combined_score": final_score,
                    "standard_deviation": np.std(templated_scores) if templated_scores else 0.0,
                    "perplexity": perplexity_score,
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
        output_str = f"# SENTIMENT LEXICONS (Top {num_select} - Complete)\n# Source: {source_name}\n"
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
    
    analyzer = SentimentAnalyzer(console, CONFIG)
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