import os
import warnings
import logging
import spacy
from collections import defaultdict
from dataclasses import dataclass, field
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")

from transformers.utils import logging as tlog
tlog.set_verbosity_error()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Mention:
    span: spacy.tokens.Span
    text: str = field(init=False)
    head_lemma: str = field(init=False)
    is_pronoun: bool = field(init=False)

    def __post_init__(self):
        self.text = self.span.text
        self.head_lemma = self.span.root.lemma_.lower()
        self.is_pronoun = self.span.root.pos_ == 'PRON'

    def __hash__(self):
        return hash((self.span.start_char, self.span.end_char))

    def __eq__(self, other):
        return isinstance(other, Mention) and self.span.start_char == other.span.start_char and self.span.end_char == other.span.end_char

class EntityConsolidator:
    def __init__(self, device: str = "-1"):
        self.device = self._cpu(device)
        self.mav = self._get_mav_model()
        self.nlp = self._get_spacy_model("en_core_web_lg")
        
    def _cpu(self, d):
        return "cpu" if d in (None, "-1", "CPU", "cpu") else d

    def _get_mav_model(self):
        try:
            from maverick import Maverick
        except ImportError:
            raise RuntimeError("'maverick' package is required.")
        logger.info(f"Initializing Maverick with device: {self.device}")
        return Maverick(device=self.device)

    def _get_spacy_model(self, model_name):
        logger.info(f"Loading spaCy model: {model_name}...")
        try:
            return spacy.load(model_name)
        except OSError:
            raise
    
    def _extract_mentions(self, doc: spacy.tokens.Doc):
        mentions = set()
        for span in list(doc.ents) + list(doc.noun_chunks):
            if span.root.pos_ in ('NOUN', 'PROPN', 'PRON'):
                mentions.add(Mention(span))
        return sorted(list(mentions), key=lambda m: m.span.start_char)

    def analyze(self, text: str, similarity_threshold=0.75):
        doc = self.nlp(text)
        all_mentions = self._extract_mentions(doc)
        
        span_to_cluster_id = {}
        mav_clusters = self.mav.predict(text)["clusters_char_offsets"]
        for i, chain in enumerate(mav_clusters):
            for s, e in chain:
                for mention in all_mentions:
                    if mention.span.start_char >= s and mention.span.end_char <= e + 1:
                        span_to_cluster_id[(mention.span.start_char, mention.span.end_char)] = i
                        break
        
        clusters = []
        for mention in all_mentions:
            mav_cluster_id = span_to_cluster_id.get((mention.span.start_char, mention.span.end_char))
            
            found_cluster = None
            if not mention.is_pronoun:
                for i, cluster in enumerate(clusters):
                    if any(m.head_lemma == mention.head_lemma for m in cluster):
                        found_cluster = i
                        break

            if mav_cluster_id is not None:
                related_clusters_indices = [
                    idx for idx, cluster in enumerate(clusters)
                    if any(span_to_cluster_id.get((m.span.start_char, m.span.end_char)) == mav_cluster_id for m in cluster)
                ]
                if related_clusters_indices:
                    master_idx = related_clusters_indices[0]
                    for idx in sorted(related_clusters_indices[1:], reverse=True):
                        clusters[master_idx].extend(clusters.pop(idx))
                    clusters[master_idx].append(mention)
                elif found_cluster is not None:
                    clusters[found_cluster].append(mention)
                else:
                    clusters.append([mention])
            elif found_cluster is not None:
                clusters[found_cluster].append(mention)
            else:
                clusters.append([mention])

        merged = True
        while merged:
            merged = False
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    if i >= len(clusters) or j >= len(clusters): continue
                    
                    rep1 = next((m.span for m in clusters[i] if not m.is_pronoun), clusters[i][0].span)
                    rep2 = next((m.span for m in clusters[j] if not m.is_pronoun), clusters[j][0].span)
                    
                    if rep1.has_vector and rep2.has_vector and rep1.similarity(rep2) > similarity_threshold:
                        logger.info(f"Semantic Merge: Merging '{rep2.text}' into '{rep1.text}'")
                        clusters[i].extend(clusters.pop(j))
                        merged = True
                        break
                if merged: break
                        
        final_output = {}
        for cluster in clusters:
            non_pronouns = [m for m in cluster if not m.is_pronoun]
            if not non_pronouns: continue
            
            canonical_name = max(non_pronouns, key=lambda m: len(m.text)).text
            
            mentions_text = {m.text for m in cluster if m.text.lower() not in ["itself"]}
            sorted_mentions = sorted(list(mentions_text), key=lambda m: text.find(m))
            
            if sorted_mentions:
                final_output[canonical_name] = {"mentions": sorted_mentions}
            
        return final_output

if __name__ == "__main__":
    consolidator = EntityConsolidator()

    restaurant_text = (
        "The food at Guido's was exceptional, but the service was a letdown. "
        "We ordered the pepperoni pizza, which was delicious. It had a perfectly crispy crust. "
        "However, our server seemed overwhelmed. The wait staff needs more training. "
        "The ambiance of the restaurant itself is quite nice, though."
    )

    print("\n--- Analyzing Restaurant Review ---")
    analysis_result = consolidator.analyze(restaurant_text)

    print("\n--- Final Analysis Result ---")
    print(json.dumps(analysis_result, indent=2))