import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import mplcursors
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List
import spacy
import afinn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nlp = spacy.load("en_core_web_sm")
_analyzer = SentimentIntensityAnalyzer()
_afinn = afinn.Afinn()

def _tokenize_with_word(text):
    return [token.text for token in nlp(text)]

def normalized_afinn_score(text):
    scores = [_afinn.score(w) for w in _tokenize_with_word(text)]
    relevant = [s for s in scores if s != 0]
    return sum(relevant) / len(relevant) / 5.0 if relevant else 0.0

def analyze_sentiment(text):
    return (_analyzer.polarity_scores(text)['compound'] + normalized_afinn_score(text)) / 2 if text else 0.0

class RelationshipType(Enum):
    ASSOCIATION = "association"
    BELONGING   = "belonging"
    ACTION      = "action"

@dataclass
class Term:
    text: str
    token_idx_is: int
    token_idx_ft: int

@dataclass
class Aspect:
    aspect_id: int
    label: str
    references: Dict[int, List[Term]]

@dataclass
class Relationship:
    relationship_id: int
    source: int
    target: int
    type: RelationshipType
    sentence_idx: int
    term_text: str
    idx_ia: int
    idx_ft: int

class SentimentPropertyGraph:
    def __init__(self, text):
        self.G = nx.DiGraph()
        self.text = text
        self.sentences = [s.text for s in nlp(text).sents]
        self.aspects: Dict[int, Aspect] = {}
        self.relationships: Dict[int, Relationship] = {}
        self.sentences_data = {i: {"aspects": [], "terms": [], "relationships": []}
                               for i in range(len(self.sentences))}
        self.aspect_id_counter = 0
        self.relationship_id_counter = 0

    def add_aspect(self, refs_dict: Dict[int, List[Dict]], label=None):
        self.aspect_id_counter += 1
        aid = self.aspect_id_counter
        refs = {}
        for i in range(len(self.sentences)):
            terms = [Term(r["text"], r["token_idx_is"], r["token_idx_ft"])
                     for r in refs_dict.get(i, [])]
            refs[i] = terms
            if terms:
                self.sentences_data[i]["aspects"].append(aid)
                self.sentences_data[i]["terms"].extend(terms)
        
        default_label_text = ""
        first_sentence_refs = refs_dict.get(0)
        if first_sentence_refs and len(first_sentence_refs) > 0 and isinstance(first_sentence_refs[0], dict):
            default_label_text = first_sentence_refs[0].get("text", "")
        
        lbl = label or default_label_text or f"Aspect_{aid}"
        self.aspects[aid] = Aspect(aid, lbl, refs)
        self.G.add_node(aid, label=lbl, history=[], sentiment=0.0)

    def add_aspect_term(self, aspect_id, sentence_idx, term_text, idx_ia, idx_ft):
        t = Term(term_text, idx_ia, idx_ft)
        self.aspects[aspect_id].references[sentence_idx].append(t)
        self.sentences_data[sentence_idx]["aspects"].append(aspect_id)
        self.sentences_data[sentence_idx]["terms"].append(t)

    def add_relationship(self, source_id, target_id, rel_type,
                         sentence_idx, term_text, idx_ia, idx_ft):
        self.relationship_id_counter += 1
        rid = self.relationship_id_counter
        r = Relationship(rid, source_id, target_id, rel_type,
                         sentence_idx, term_text, idx_ia, idx_ft)
        self.relationships[rid] = r
        self.sentences_data[sentence_idx]["relationships"].append(r)
        self.G.add_edge(source_id, target_id, relationship=r, type=rel_type)

    def _merge_adjacent_terms(self, terms: List[Term]):
        sorted_terms = sorted(terms, key=lambda t: t.token_idx_is)
        merged = []
        if not sorted_terms:
            return merged
        start = sorted_terms[0].token_idx_is
        end   = sorted_terms[0].token_idx_ft
        texts = [sorted_terms[0].text]
        for t in sorted_terms[1:]:
            if t.token_idx_is <= end + 1:
                end = max(end, t.token_idx_ft)
                texts.append(t.text)
            else:
                merged.append({"text":" ".join(texts),"token_idx_is":start,"token_idx_ft":end})
                start, end, texts = t.token_idx_is, t.token_idx_ft, [t.text]
        merged.append({"text":" ".join(texts),"token_idx_is":start,"token_idx_ft":end})
        return merged

    def animate_processing(self):
        phases = []
        for i in range(len(self.sentences)):
            phases += [("Merge & Raw", i),
                       ("Relationships", i),
                       ("Intra Adjust", i),
                       ("Inter Adjust", i)]
        fig, ax = plt.subplots(figsize=(10,6))
        btn_ax = fig.add_axes([0.82, 0.01, 0.15, 0.06])
        btn = Button(btn_ax, "Next")
        pos = nx.spring_layout(self.G, seed=42)
        state = {"idx": 0}

        def show_phase(event=None):
            idx = state["idx"]
            if idx >= len(phases):
                btn.label.set_text("Done")
                btn.on_clicked(lambda e: None)
                return
            phase, sent_i = phases[idx]
            ax.clear()
            info = []
            if phase == "Merge & Raw":
                for aid in self.sentences_data[sent_i]["aspects"]:
                    merged = self._merge_adjacent_terms(self.aspects[aid].references[sent_i])
                    val = sum(analyze_sentiment(m["text"]) for m in merged) / len(merged) if merged else 0.0
                    self.G.nodes[aid]["history"].append(val)
                    self.G.nodes[aid]["sentiment"] = val
                    info.append(f"{self.aspects[aid].label}: merged={ [m['text'] for m in merged] } -> raw={val:.2f}")
            elif phase == "Relationships":
                for r in self.sentences_data[sent_i]["relationships"]:
                    info.append(f"Rel {r.relationship_id}: {r.source}->{r.target} [{r.type.value}] “{r.term_text}”")
            elif phase == "Intra Adjust":
                for r in self.sentences_data[sent_i]["relationships"]:
                    for aid in (r.source, r.target):
                        base = self.G.nodes[aid]["sentiment"]
                        opp = r.target if aid==r.source else r.source
                        opp_val = self.G.nodes[opp]["sentiment"]
                        if r.type == RelationshipType.ACTION:
                            rs = analyze_sentiment(r.term_text)
                            new = (base*3 + rs*opp_val) / 4
                        else:
                            new = (base*3 + opp_val) / 4
                        self.G.nodes[aid]["sentiment"] = new
                        self.G.nodes[aid]["history"].append(new)
                        info.append(f"{self.aspects[aid].label}: adjusted={new:.2f}")
            else:
                for aid,data in self.G.nodes(data=True):
                    hist = data["history"]
                    n = len(hist)
                    if n < 2:
                        val = hist[-1] if hist else 0.0
                    else:
                        weights = [((j/(n-1)-0.5)**2) for j in range(n)]
                        total = sum(weights)
                        val = sum(h*w for h,w in zip(hist,weights)) / total if total else 0.0
                    self.G.nodes[aid]["sentiment"] = val
                    info.append(f"{self.aspects[aid].label}: inter={val:.2f} hist={['{:.2f}'.format(h) for h in hist]}")
            node_colors = ["green" if self.G.nodes[n]["sentiment"]>0 else "red" if self.G.nodes[n]["sentiment"]<0 else "gray"
                           for n in self.G.nodes]
            sizes = [600 + 150*abs(self.G.nodes[n]["sentiment"]) for n in self.G]
            edge_colors = ["blue" if phase=="Relationships" and d["relationship"].sentence_idx==sent_i else "lightgray"
                           for _,_,d in self.G.edges(data=True)]
            widths = [3 if phase=="Relationships" and d["relationship"].sentence_idx==sent_i else 1
                      for _,_,d in self.G.edges(data=True)]
            nodes = nx.draw_networkx_nodes(self.G, pos, node_color=node_colors, node_size=sizes, ax=ax)
            nx.draw_networkx_edges(self.G, pos, edge_color=edge_colors, width=widths, ax=ax)
            nx.draw_networkx_labels(self.G, pos, ax=ax)
            ax.set_title(f"{phase} (Sentence {sent_i+1}/{len(self.sentences)})")
            ax.text(1.02, 0.5, "\n".join(info), transform=ax.transAxes, va="center", fontsize=9,
                    bbox=dict(facecolor="white",edgecolor="black",boxstyle="round,pad=0.5"))
            mplcursors.cursor(nodes, hover=True).connect(
                "add", lambda sel: sel.annotation.set_text(
                    f"{self.G.nodes[list(self.G.nodes())[sel.index]]}"
                ))
            state["idx"] += 1
            plt.draw()

        btn.on_clicked(show_phase)
        show_phase()
        plt.show()

    def visualize(self):
        pos = nx.spring_layout(self.G, seed=42)
        node_colors = ["green" if d["sentiment"]>0 else "red" if d["sentiment"]<0 else "gray"
                       for _,d in self.G.nodes(data=True)]
        sizes = [600 + 150*abs(d["sentiment"]) for _,d in self.G.nodes(data=True)]
        plt.figure(figsize=(10,6))
        nx.draw_networkx(self.G, pos, node_color=node_colors, node_size=sizes, arrowsize=12, font_size=10)
        plt.show()

if __name__=='__main__':
    text = "The battery life is fantastic. The camera quality is disappointing. The phone is decent."
    spg = SentimentPropertyGraph(text)
    spg.add_aspect({0:[{"text":"battery","token_idx_is":1,"token_idx_ft":1},
                       {"text":"life","token_idx_is":2,"token_idx_ft":2}]}, "battery_life")
    spg.add_aspect({1:[{"text":"camera","token_idx_is":1,"token_idx_ft":1},
                       {"text":"quality","token_idx_is":2,"token_idx_ft":2}]}, "camera_quality")
    spg.add_aspect({2:[{"text":"phone","token_idx_is":1,"token_idx_ft":1}]}, "phone")
    spg.add_aspect_term(1,0,"fantastic",4,4)
    spg.add_aspect_term(2,1,"disappointing",4,4)
    spg.add_aspect_term(3,2,"decent",3,3)
    spg.add_relationship(1,3,RelationshipType.BELONGING,2,"belongs",2,2)
    spg.add_relationship(2,3,RelationshipType.BELONGING,2,"belongs",3,3)
    spg.animate_processing()
    spg.visualize()