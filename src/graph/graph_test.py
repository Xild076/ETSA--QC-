import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum
import spacy

nlp = spacy.load("en_core_web_sm")

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
_analyzer = SentimentIntensityAnalyzer()
def analyze_sentiment(text): return _analyzer.polarity_scores(text)['compound'] if text else 0.0

class RelationshipType(Enum):
    ASSOCITION = "association"
    BELONGING = "belonging"
    ACTION = "action"

class SentimentPropertyGraph:
    def __init__(self, text, intra_sentence_eq=None, inter_sentence_eq=None):
        self.G = nx.DiGraph()
        self.text = text
        self.sentences = [nlp(sent).text for sent in nlp(text).sents]
        self.intra_sentence_eq = intra_sentence_eq
        self.inter_sentence_eq = inter_sentence_eq
        self.aspects = {}
        self.relationships = {}
        self.aspect_id_counter = 0
        self.relationship_id_counter = 0
        
    def add_aspect(self, references, label=None):
        self.aspect_id_counter += 1
        aspect_id = self.aspect_id_counter
        self.aspects[aspect_id] = {}
        for i in range(len(self.sentences)):
            self.aspects[aspect_id][i] = []
            for ref in references[i]:
                self.aspects[aspect_id][i].append({
                    "text": ref['text'],
                    "token_idx_is": ref['token_idx_is'],
                    "token_idx_ft": ref['token_idx_ft'],
                    "terms": ref["terms"] if "terms" in ref else []
                })
        self.aspects[aspect_id]["label"] = label if label else (references[0][0]["text"] if references else f"Aspect_{aspect_id}")
        self.G.add_node(aspect_id, label=self.aspects[aspect_id]["label"], aspect_id=aspect_id, aspects=self.aspects[aspect_id])
    
    def add_references(self, aspect_id, references):
        if aspect_id not in self.aspects:
            raise ValueError(f"Aspect ID {aspect_id} does not exist.")
        entry = self.aspects[aspect_id]
        for i, refs in enumerate(references):
            for ref in refs:
                bucket = entry[i]
                if bucket["text"] is None:
                    bucket["text"] = ref["text"]
                    bucket["token_idx_is"] = ref["token_idx_is"]
                    bucket["token_idx_ft"] = ref["token_idx_ft"]
                bucket["terms"].append({
                    "text": ref["text"],
                    "token_idx_is": ref["token_idx_is"],
                    "token_idx_ft": ref["token_idx_ft"]
                })
                self.G.nodes[aspect_id]["aspects"][i]["terms"].append({
                    "text": ref["text"],
                    "token_idx_is": ref["token_idx_is"],
                    "token_idx_ft": ref["token_idx_ft"]
                })
        self.G.nodes[aspect_id]["aspects"] = entry
    
    def add_aspect_term(self, aspect_id, sentence_idx, term_text, idx_ia, idx_ft):
        if aspect_id not in self.aspects:
            raise ValueError(f"Aspect ID {aspect_id} does not exist.")
        bucket = self.aspects[aspect_id][sentence_idx]
        if bucket["text"] is None:
            bucket["text"] = term_text
            bucket["token_idx_is"] = idx_ia
            bucket["token_idx_ft"] = idx_ft
        term = {
            "text": term_text,
            "token_idx_is": idx_ia,
            "token_idx_ft": idx_ft
        }
        bucket["terms"].append(term)
        self.G.nodes[aspect_id]["aspects"][sentence_idx]["terms"].append(term)
    
    def add_relationship(self, source_aspect_id, target_aspect_id, relationship_type, sentence_idx, term_text, idx_ia, idx_ft):
        if source_aspect_id not in self.aspects or target_aspect_id not in self.aspects:
            raise ValueError("Both source and target aspect IDs must exist.")
        
        self.relationship_id_counter += 1
        relationship_id = self.relationship_id_counter
        
        relationship = {
            "source": source_aspect_id,
            "target": target_aspect_id,
            "type": relationship_type,
            "sentence_idx": sentence_idx,
            "term_text": term_text,
            "idx_ia": idx_ia,
            "idx_ft": idx_ft,
        }
        
        if relationship_id not in self.relationships:
            self.relationships[relationship_id] = []
        
        self.relationships[relationship_id].append(relationship)
        
        self.G.add_edge(source_aspect_id, target_aspect_id, relationship=relationship, type=relationship_type)

    def _merge_adjacent_terms(self, terms):
        sorted_terms = sorted(terms, key=lambda t: t["token_idx_is"])
        merged = []
        if not sorted_terms:
            return merged
        start = sorted_terms[0]["token_idx_is"]
        end = sorted_terms[0]["token_idx_ft"]
        texts = [sorted_terms[0]["text"]]
        for t in sorted_terms[1:]:
            if t["token_idx_is"] <= end + 1:
                end = max(end, t["token_idx_ft"])
                texts.append(t["text"])
            else:
                merged.append({"text": " ".join(texts), "token_idx_is": start, "token_idx_ft": end})
                start, end, texts = t["token_idx_is"], t["token_idx_ft"], [t["text"]]
        merged.append({"text": " ".join(texts), "token_idx_is": start, "token_idx_ft": end})
        return merged
    
    def calculate_intra_sent_entity_sentiment(self):
        for sentence in range(len(self.sentences)):
            for aspect_id, aspect_data in self.aspects.items():
                if sentence in aspect_data:
                    terms = aspect_data[sentence]["terms"]
                    if terms:
                        sentiments = [analyze_sentiment(term["text"]) for term in terms]
                        intra_sentiment = self.intra_sentence_eq(sentiments) if self.intra_sentence_eq else sum(sentiments) / len(sentiments)
                        self.G.nodes[aspect_id]["sentiment"] = intra_sentiment
        for relationship_id, relationships in self.relationships.items():
            for relationship in relationships:
                source_sentiment = self.G.nodes[relationship["source"]].get("sentiment", 0.0)
                target_sentiment = self.G.nodes[relationship["target"]].get("sentiment", 0.0)
                relationship_text_sentiment = analyze_sentiment(relationship["term_text"])
                relationship_sentiment = self.inter_sentence_eq(source_s=source_sentiment, target_s=target_sentiment, relationship_s=relationship_text_sentiment, relation_type=relationship["type"])
                self.G.edges[relationship["source"], relationship["target"]]["sentiment"] = relationship_sentiment
    
    def calculate_inter_sent_entity_sentiment(self):
        for entity in self.G.nodes:
            if "sentiment" not in self.G.nodes[entity]:
                self.G.nodes[entity]["sentiment"] = 0.0
            history = self.G.nodes[entity].get("history", [])
            if history:
                sentiments = [s for _, s in history]
                if sentiments:
                    intra_sentiment = self.inter_sentence_eq(sentiments)
                    self.aspects[entity]["sentiment"] = intra_sentiment
                    self.G.nodes[entity]["sentiment"] = intra_sentiment
    
    def visualize(self):
        pos = nx.spring_layout(self.G, seed=42)
        node_sizes = [500 + 100 * abs(self.G.nodes[n].get("sentiment", 0.0)) for n in self.G.nodes]
        node_colors = ["green" if self.G.nodes[n].get("sentiment", 0.0) > 0 else "red" for n in self.G.nodes]
        edge_colors = ["blue" if d["type"] == RelationshipType.ASSOCITION else "orange" for _, _, d in self.G.edges(data=True)]
        
        plt.figure(figsize=(12, 8))
        nx.draw(self.G, pos, with_labels=True, node_size=node_sizes, node_color=node_colors, edge_color=edge_colors, font_size=10, font_color="black")
        plt.title("Sentiment Property Graph")
        plt.show()