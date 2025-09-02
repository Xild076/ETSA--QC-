import os, warnings, logging, nltk, spacy
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
logging.disable(logging.CRITICAL)
nltk.download("punkt", quiet=True)

from functools import lru_cache
from typing import Dict, Any, List, Tuple
from maverick import Maverick
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.clause import clause_extraction as ce
from transformers.utils import logging as tlog
tlog.set_verbosity_error()

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _cpu(d): return "cpu" if d in (None, "-1", "CPU", "cpu") else d

@lru_cache(maxsize=None)
def _mav(device: str):
    logger.info(f"Initializing Maverick with device: {device}")
    return Maverick(device=_cpu(device))

@lru_cache(maxsize=None)
def _spacy():
    logger.info("Loading SpaCy model...")
    for m in ("en_core_web_trf", "en_core_web_lg", "en_core_web_sm"):
        try: return spacy.load(m)
        except: pass
    raise RuntimeError("install a spaCy English model")

def _sent_spans(text: str) -> List[Tuple[int, int]]:
    nlp = _spacy()
    return [(s.start_char, s.end_char) for s in nlp(text).sents]

_det = ("the ", "The ", "a ", "A ", "an ", "An ")

_poss = ("my ", "My ", "our ", "Our ", "your ", "Your ", "his ", "His ", "her ", "Her ", "their ", "Their ")
_quant = ("plenty of ", "Plenty of ", "lots of ", "Lots of ", "lot of ", "Lot of ", "bunch of ", "Bunch of ", "kind of ", "Kind of ", "sort of ", "Sort of ")

_stop_entities = {"i", "we", "they", "he", "she", "it", "this", "that", "these", "those", "here", "there", "rest", "us", "you", "someone", "something", "anything", "everything", "nothing", "plenty", "favorite", "wannabe", "wannabes"}

_food_lex = {"food","onion","ring","rings","chili","soup","soups","pizza","burger","pasta","steak","fish","beer","wine","coffee","drink","drinks","guinness","menu"}
_service_lex = {"service","staff","server","servers","waiter","waitress","host","hostess","bartender","maitre","maitre d'","buy-back"}
_amb_lex = {"ambience","ambiance","atmosphere","music","noise","decor","lighting","environment","place","bar","restaurant","kitchen"}

def _strip(t: str, s: int) -> Tuple[str, int]:
    for d in _det:
        if t.startswith(d): return t[len(d):], s + len(d)
    for p in _poss:
        if t.startswith(p): return t[len(p):], s + len(p)
    for q in _quant:
        if t.startswith(q): return t[len(q):], s + len(q)
    return t, s

def _normalize_to_head(doc: "spacy.tokens.Doc", s: int, e: int, t: str) -> Tuple[str, int, int]:
    span = doc.char_span(s, e, alignment_mode="expand")
    if span is None or not span:
        return t, s, e
    head = span.root
    if head is None:
        return t, s, e
    try:
        if any(tok.text.lower() == "of" for tok in span):
            for tok in span:
                if tok.dep_ == "pobj" and (tok.lemma_.lower() in _food_lex):
                    ys = tok.left_edge.idx
                    ye = tok.right_edge.idx + len(tok.right_edge.text)
                    text_y = doc.text[ys:ye]
                    text_y2, ys2 = _strip(text_y, ys)
                    return text_y2, ys2, ys2 + len(text_y2)
    except Exception:
        pass
    compound_parts = [head]
    for tok in head.children:
        if tok.dep_ in ("compound", "amod") and tok.i < head.i:
            compound_parts.append(tok)
    compound_parts = sorted(compound_parts, key=lambda x: x.i)
    hs = compound_parts[0].idx
    he = head.idx + len(head.text)
    text_span = doc.text[hs:he]
    text_span2, hs2 = _strip(text_span, hs)
    if text_span2.strip().lower() in _stop_entities:
        return text_span2, hs2, hs2 + len(text_span2)
    return text_span2, hs2, hs2 + len(text_span2)

def resolve(text: str, device: str = "-1") -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    logger.info(f"Resolving entities in text with device: {device}")
    mav, nlp = _mav(device), _spacy()
    clusters, seen = {}, set()
    for cid, chain in enumerate(mav.predict(text)["clusters_char_offsets"]):
        refs = []
        for s0, e0 in chain:
            raw = text[s0:e0 + 1]
            t, s = _strip(raw, s0)
            e = s + len(t)
            t2, s2, e2 = _normalize_to_head(nlp(text), s, e, t)
            if t2.strip().lower() in _stop_entities:
                continue
            refs.append((t2, [s2, e2]))
            seen.add((s2, e2))
        clusters[cid] = {"entity_references": refs}
    next_id = len(clusters)
    doc = nlp(text)
    for span in list(doc.ents) + list(doc.noun_chunks):
        has_ent = any(getattr(tok, "ent_type_", "") for tok in span)
        if span.root.pos_ not in ("NOUN", "PROPN") and not has_ent:
            continue
        t, s = _strip(span.text, span.start_char)
        e = s + len(t)
        t, s, e = _normalize_to_head(doc, s, e, t)
        if t.strip().lower() in _stop_entities:
            continue
        if (s, e) not in seen:
            clusters[next_id] = {"entity_references": [(t, [s, e])]}
            seen.add((s, e)); next_id += 1
    spans = _sent_spans(text)
    sent_map = {f"clause_{i}": {"entities": []} for i in range(len(spans))}
    for info in clusters.values():
        for txt, (s, e) in info["entity_references"]:
            for i, (ss, se) in enumerate(spans):
                if ss <= s < se:
                    sent_map[f"clause_{i}"]["entities"].append((txt, [s, e])); break
    return clusters, sent_map

"""text = "The man hit the child. The child was sad."
c, s = resolve(text)
print(c)
print(s)"""