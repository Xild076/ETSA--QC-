import os, warnings, logging, nltk, spacy
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
nltk.download("punkt", quiet=True)

from functools import lru_cache
from typing import Dict, Any, List, Tuple
from maverick import Maverick
from transformers.utils import logging as tlog
tlog.set_verbosity_error()

def _cpu(d): return "cpu" if d in (None, "-1", "CPU", "cpu") else d

@lru_cache(maxsize=None)
def _mav(device: str):
    return Maverick(device=_cpu(device))

@lru_cache(maxsize=None)
def _spacy():
    for m in ("en_core_web_trf", "en_core_web_lg", "en_core_web_sm"):
        try: return spacy.load(m)
        except: pass
    raise RuntimeError("install a spaCy English model")

def _sent_spans(text: str) -> List[Tuple[int, int]]:
    spans, i = [], 0
    for s in nltk.sent_tokenize(text):
        j = text.index(s, i); spans.append((j, j + len(s))); i = j + len(s)
    return spans

_det = ("the ", "The ", "a ", "A ", "an ", "An ")

def _strip(t: str, s: int) -> Tuple[str, int]:
    for d in _det:
        if t.startswith(d): return t[len(d):], s + len(d)
    return t, s

def resolve(text: str, device: str = "-1") -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    mav, nlp = _mav(device), _spacy()
    clusters, seen = {}, set()
    for cid, chain in enumerate(mav.predict(text)["clusters_char_offsets"]):
        refs = []
        for s0, e0 in chain:
            t, s = _strip(text[s0:e0 + 1], s0)
            e = s + len(t)
            refs.append((t, [s, e]))
            seen.add((s, e))
        clusters[cid] = {"entity_references": refs}
    next_id = len(clusters)
    doc = nlp(text)
    for span in list(doc.ents) + list(doc.noun_chunks):
        if span.root.pos_ not in ("NOUN", "PROPN") and not span.ent_type_: continue
        t, s = _strip(span.text, span.start_char)
        e = s + len(t)
        if (s, e) not in seen:
            clusters[next_id] = {"entity_references": [(t, [s, e])]}
            seen.add((s, e)); next_id += 1
    spans = _sent_spans(text)
    sent_map = {f"sentence_{i}": {"entities": []} for i in range(len(spans))}
    for info in clusters.values():
        for txt, (s, e) in info["entity_references"]:
            for i, (ss, se) in enumerate(spans):
                if ss <= s < se:
                    sent_map[f"sentence_{i}"]["entities"].append((txt, [s, e])); break
    return clusters, sent_map

text = "The man hit the child. The child was sad."
c, s = resolve(text)
print(c)
print(s)