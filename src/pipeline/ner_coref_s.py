"""Hybrid aspect extraction and coreference helpers for the sentiment pipeline."""

import ast
import json
import logging
import os
import pickle
import random
import re
import time
import warnings
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from string import punctuation
from typing import Any, Dict, List, Set, Tuple

import spacy
import torch

try:
    from . import config
    from ..utility import normalize_text
except ImportError:
    import config
    from utility import normalize_text

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

_MODEL_CACHE = {}

def get_optimal_device():
    """Get the optimal device for computation (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

@lru_cache(maxsize=4)
def get_cached_spacy_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        logger.info(f"Loading spaCy model: {model_name}...")
        try:
            _MODEL_CACHE[model_name] = spacy.load(model_name)
        except OSError:
            logger.info(f"Model '{model_name}' not found. Downloading...")
            spacy.cli.download(model_name)
            _MODEL_CACHE[model_name] = spacy.load(model_name)
    return _MODEL_CACHE[model_name]

@lru_cache(maxsize=1)
def get_cached_maverick_model(device: str = None):
    if device is None:
        device = get_optimal_device()
    
                                                                       
    if device == "mps":
        device = "cpu"
        
    cache_key = f"maverick_{device}"
    if cache_key not in _MODEL_CACHE:
        try:
            from maverick import Maverick
            logger.info(f"Initializing Maverick with device: {device}")
            _MODEL_CACHE[cache_key] = Maverick(device=device)
        except ImportError:
            logger.warning("Maverick package not available, coreference resolution will be skipped")
            _MODEL_CACHE[cache_key] = None
        except Exception as e:
            logger.warning(f"Failed to initialize Maverick on {device}, falling back to CPU: {e}")
            try:
                from maverick import Maverick
                _MODEL_CACHE[f"maverick_cpu"] = Maverick(device="cpu")
                return _MODEL_CACHE[f"maverick_cpu"]
            except:
                _MODEL_CACHE[f"maverick_cpu"] = None
                return None
    return _MODEL_CACHE[cache_key]

class ATE:
    def analyze(self, sentences: List[str]) -> Dict[int, Dict[str, Any]]:
        raise NotImplementedError("This method should be overridden by subclasses.")

@dataclass(eq=True, frozen=True)
class TrainConfig:
    model:str="auto"
    model_candidates:tuple=("en_core_web_trf","en_core_web_lg","en_core_web_md")
    restrict_pos:tuple=("NOUN","PROPN","ADJ","VERB")
    min_pos_support:int=3
    min_rule_precision:float=0.55
    max_stages:int=20
    max_iterations_per_stage:int=18
    new_rules_per_iteration:int=90
    n_process:int=1
    batch_size:int=4000
    target_precision:float=0.95
    target_recall:float=0.95
    target_f1:float=0.95
    pr_margin:float=0.02
    f1_eval_sample:int=2000
    f1_eval_topk:int=1000
    prune:bool=True
    prune_strategy:str="keep_topk"
    stage_keep_topk:int=220
    min_stage_keep:int=50
    no_prune_if_stage_gain_ge:float=0.015
    max_prune_drop:float=0.0
    early_stop_residual:int=3000
    min_gain_per_stage:float=0.0005
    debug:bool=False
    debug_every_n_stages:int=3
    debug_top_k:int=15
    dump_dir:str|None=None
    use_grandparent:bool=True
    use_child_profile:bool=True
    child_profile_k:int=3
    head_focus_k:int=8
    adapt_head_min_support:int=4
    use_entities:bool=True
    use_chunks:bool=True
    use_neighbor_pos:bool=True
    neighbor_window:int=2
    coord_backoff:bool=True
    soft_restrict:bool=True
    trace_level:str="INFO"
    max_block_phrases:int=45
    block_singletons:bool=True
    protect_heads_min_freq:int=12
    use_tags:bool=True
    dep_path_k:int=4
    use_head_morph:bool=True
    use_head_children_pos:bool=True
    use_det_child:bool=True
    use_prep_child:bool=True
    use_is_stop:bool=True
    use_entity_iob:bool=True
    use_in_chunk:bool=True
    use_dep_distance:bool=True
    use_neighbor_lemmas:bool=True
    neighbor_lemmas_k:int=2
    use_coord_siblings:bool=True
    coord_siblings_k:int=2
    use_sent_len_bucket:bool=True
    use_surface_whitelist:bool=True
    surface_min_support_single:int=6
    surface_min_support_multi:int=3
    surface_max_len:int=5
    surface_precision_threshold_single:float=0.55
    surface_precision_threshold_multi:float=0.4
    generic_block_heads: frozenset = field(default_factory=lambda: frozenset({"place","restaurant","time","thing","table","dinner","drinks","price","prices","laptop","computer","bar","dining","product"}))
    hard_block_singletons: frozenset = field(default_factory=lambda: frozenset({"area","room"}))
    risky_hl_heads: frozenset = field(default_factory=lambda: frozenset({"food","service","restaurant","price","prices","pizza","staff","sushi","wine","meal","wait","chicken","dinner","quality","menu","drink","drinks","table","bar","place"}))

class MultiStageRuleExtractor(ATE):
    def __init__(self, cfg:TrainConfig):
        self.cfg = cfg
        self.nlp = self._load_spacy_model(cfg)
        self.rule_stages = []
        self.rule_allowed_heads = {}
        self.stopset = set(spacy.lang.en.stop_words.STOP_WORDS) | set(punctuation)
        self.aspect_lex_head_lemmas = set()
        self.frequent_single_gold_heads = set()
        self.block_phrases = set()
        self.last_selection_cache = {}
        self.chronic_heads = set()
        self.gold_head_freq = Counter()
        self.surface_positive_counts = Counter()
        self.surface_whitelist_single = set()
        self.surface_whitelist_multi_by_len = defaultdict(set)
        self.surface_precision = {}
        logging.basicConfig(level=getattr(logging, cfg.trace_level.upper(), logging.INFO), format="%(message)s")
        self.log = logging.getLogger("RuleExtractor")

    def _load_spacy_model(self, cfg):
        ordered = []
        if cfg.model and cfg.model != "auto": ordered.append(cfg.model)
        for name in cfg.model_candidates:
            if name not in ordered: ordered.append(name)
        if not ordered: ordered.append("en_core_web_md")
        base_dir = Path(os.path.abspath(__file__)).resolve().parents[3] if "__file__" in locals() else Path.cwd()
        models_dir = base_dir / "models"
        for name in ordered:
            targets = [name]
            if models_dir.is_dir():
                direct = models_dir / name
                if direct.exists(): targets.insert(0, str(direct))
                else:
                    for candidate in models_dir.glob(f"{name}*"):
                        if candidate.is_dir(): targets.insert(0, str(candidate))
            tried = set()
            for target in targets:
                if target in tried: continue
                tried.add(target)
                try:
                    return spacy.load(target, disable=[])
                except (OSError, Exception):
                    continue
            try:
                if cfg.model and cfg.model != "auto" and name == cfg.model and not (Path(name).exists() or os.path.sep in name):
                    spacy.cli.download(name)
                    return spacy.load(name, disable=[])
            except Exception:
                continue
        raise RuntimeError(f"No spaCy model available from: {', '.join(ordered)}")

    def _filter_allowed_heads(self, heads):
        if not heads: return set()
        return {h for h in heads if not (h in self.cfg.generic_block_heads and self.gold_head_freq[h] < self.cfg.protect_heads_min_freq) and h not in self.cfg.hard_block_singletons}

    def _build_surface_whitelist(self):
        self.surface_whitelist_single = set()
        self.surface_whitelist_multi_by_len = defaultdict(set)
        if not self.cfg.use_surface_whitelist: return
        for surf, count in self.surface_positive_counts.items():
            toks = tuple(surf.split())
            if not toks: continue
            if len(toks) > self.cfg.surface_max_len: continue
            if len(toks) == 1: continue
            if count >= self.cfg.surface_min_support_multi:
                self.surface_whitelist_multi_by_len[len(toks)].add(toks)

    def _augment_with_surface_whitelist(self, doc, res):
        if not self.cfg.use_surface_whitelist: return res
        if self.surface_whitelist_multi_by_len:
            lowered = [tok.text.lower() for tok in doc]
            n = len(lowered)
            for length, patterns in self.surface_whitelist_multi_by_len.items():
                if length > n: continue
                for i in range(n - length + 1):
                    span = tuple(lowered[i:i+length])
                    if span not in patterns: continue
                    surf = " ".join(span)
                    if surf in res: continue
                    if not self._valid_surface(surf): continue
                    res.add(surf)
        return res

    def _should_keep_surface(self, surf):
        if not self.surface_precision: return True
        key = surf.lower()
        stats = self.surface_precision.get(key)
        if not stats: return True
        prec, _, _ = stats
        thresh = self.cfg.surface_precision_threshold_multi if len(key.split()) > 1 else self.cfg.surface_precision_threshold_single
        return prec >= thresh

    def _refresh_surface_precision(self, prepared):
        self.surface_precision = {}
        if not prepared: return
        combined = set().union(*self.rule_stages) if self.rule_stages else set()
        total = Counter()
        tp = Counter()
        for ex in prepared:
            pred = {t.lower() for t in self.predict_doc(ex["doc"], ex["sigs"], combined, skip_surface_filter=True)}
            gt = ex["gold_terms"]
            for surf in pred:
                total[surf] += 1
                if surf in gt:
                    tp[surf] += 1
        precision_map = {}
        for surf, count in total.items():
            precision_map[surf] = (tp[surf] / count if count else 0.0, tp[surf], count)
        self.surface_precision = precision_map

    def _valid_surface(self, s):
        if not s: return False
        t = s.lower().strip()
        if len(t) < 2: return False
        if t in self.stopset: return False
        return True

    def _np_text(self, token):
        left_keep = {"compound","amod","nummod","flat","poss"}
        right_keep = {"compound","amod","nummod","flat","appos"}
        left = [t for t in token.lefts if t.dep_ in left_keep]
        right = [t for t in token.rights if t.dep_ in right_keep]
        parts = left + [token] + right
        parts.sort(key=lambda t: t.i)
        return " ".join(t.text for t in parts)

    def _compound_len(self, token):
        c = 1
        for t in token.lefts:
            if t.dep_ in {"compound","amod","nummod","flat","poss"}: c += 1
        for t in token.rights:
            if t.dep_ in {"compound","amod","nummod","flat","appos"}: c += 1
        return c

    def _child_profile(self, token):
        if not self.cfg.use_child_profile: return ()
        deps = sorted([c.dep_ for c in token.children])
        if not deps: return ()
        k = self.cfg.child_profile_k
        if k <= 0: return tuple(deps)
        step = max(1, len(deps)//k)
        return tuple(deps[::step][:k])

    def _neighbors(self, doc, i):
        if not self.cfg.use_neighbor_pos or self.cfg.neighbor_window <= 0: return ()
        L = []
        for off in range(1, self.cfg.neighbor_window+1):
            if i-off >= 0: L.append((off, doc[i-off].pos_, doc[i-off].dep_))
            if i+off < len(doc): L.append((off, doc[i+off].pos_, doc[i+off].dep_))
        return tuple(L) if L else ()

    def _neighbor_lemmas(self, doc, i):
        if not self.cfg.use_neighbor_lemmas or self.cfg.neighbor_lemmas_k <= 0: return ()
        k = self.cfg.neighbor_lemmas_k
        L = []
        for off in range(1, k+1):
            if i-off >= 0: L.append(("L", off, doc[i-off].lemma_.lower(), doc[i-off].pos_))
            if i+off < len(doc): L.append(("R", off, doc[i+off].lemma_.lower(), doc[i+off].pos_))
        return tuple(L) if L else ()

    def _chunk_info(self, token, chunk_map):
        if not self.cfg.use_chunks or token.i not in chunk_map: return (False,0,"",0)
        c = chunk_map[token.i]
        return (token.i==c.root.i, len(c), c.root.dep_, sum(1 for w in c if w.dep_=="amod"))

    def _entity_pair(self, token):
        if not self.cfg.use_entities: return ("","")
        ent = token.ent_type_ or ""
        hl = token.head.ent_type_ if token.head is not None and token.head.ent_type_ else ""
        return (ent, hl)

    def _sent_pos(self, token):
        s = token.sent
        if not s: return 0
        length = len(s)
        idx = token.i - s.start
        if length == 0: return 0
        if idx < length*0.33: return 0
        if idx < length*0.66: return 1
        return 2

    def _sent_len_bucket(self, token):
        if not self.cfg.use_sent_len_bucket: return 0
        s = token.sent
        if not s: return 0
        n = len(s)
        if n <= 8: return 0
        if n <= 16: return 1
        if n <= 30: return 2
        return 3

    def _morph_bits(self, token):
        m = token.morph
        return tuple(sorted(str(m).split("|"))) if str(m) else ()

    def _affixes(self, token):
        t = token.text
        return (t[:1].lower(), t[:2].lower(), t[:3].lower(), t[-1:].lower(), t[-2:].lower(), t[-3:].lower())

    def _subtree_size(self, token):
        return sum(1 for _ in token.subtree)

    def _head_path(self, token, k=2):
        p = []
        cur = token
        for _ in range(k):
            if cur.head is None or cur.head == cur: break
            cur = cur.head
            p.append((cur.lemma_, cur.pos_, cur.dep_))
        return tuple(p)

    def _dep_path_to_root(self, token, k):
        p, cur = [], token
        for _ in range(k):
            if cur.head is None or cur.head == cur: break
            p.append(cur.dep_)
            cur = cur.head
        return tuple(p)

    def _dep_distance(self, token):
        if not self.cfg.use_dep_distance: return 0
        cur, d = token, 0
        while cur.head is not None and cur.head != cur and d < 10:
            cur = cur.head
            d += 1
        return d

    def _head_children_pos(self, token):
        if token.head is None or not self.cfg.use_head_children_pos: return ()
        return tuple(sorted(c.pos_ for c in token.head.children))

    def _has_det_child(self, token):
        if not self.cfg.use_det_child: return False
        return any(c.dep_ == "det" for c in token.children)

    def _has_prep_child(self, token):
        if not self.cfg.use_prep_child: return False
        return any(c.dep_ == "prep" for c in token.children)

    def _coord_siblings(self, token):
        if not self.cfg.use_coord_siblings or token.dep_ != "conj": return ()
        root = token.head
        if not root: return ()
        k = self.cfg.coord_siblings_k
        sibs = [c.lemma_.lower() for c in root.children if c.dep_ == "conj" and c.i != token.i]
        if root.dep_ == "conj":
            sibs.append(root.lemma_.lower())
        return tuple(sorted(sibs[:k])) if sibs else ()

    def _sig_variants(self, token, doc, chunk_map):
        if self.cfg.restrict_pos and token.pos_ not in self.cfg.restrict_pos: return []
        if token.head is None: return []
        out = []
        ch = tuple(sorted([c.dep_ for c in token.children]))
        out.extend([
            (0, token.pos_, token.dep_, token.head.lemma_, token.head.pos_, ch),
            (1, token.pos_, token.dep_, token.head.lemma_, token.head.pos_, ()),
            (2, token.pos_, token.dep_, token.head.lemma_, None, ()),
            (3, token.pos_, token.dep_, None, None, ()),
        ])
        if self.cfg.use_grandparent and token.head.head is not None:
            gp = token.head.head
            out.append((4, token.pos_, token.dep_, token.head.lemma_, token.head.pos_, (gp.lemma_, gp.pos_)))
        if self.cfg.use_child_profile and ch:
            out.append((5, token.pos_, token.dep_, token.head.lemma_, token.head.pos_, self._child_profile(token)))
        if token.lemma_.lower() not in self.cfg.risky_hl_heads:
            out.extend([
                (6, token.pos_, token.dep_, None, None, ("HL", token.lemma_)),
                (7, None, None, None, None, ("HL", token.lemma_)),
            ])
        if self.cfg.coord_backoff and token.dep_=="conj" and token.head.pos_ in ("NOUN","PROPN"):
            out.append((8, 'conj', token.head.pos_, tuple(sorted(c.dep_ for c in token.head.children))))
        if self.cfg.use_entities:
            ent_tok, ent_head = self._entity_pair(token)
            if ent_tok or ent_head: out.append((9, token.pos_, token.dep_, ent_tok, ent_head))
        if self.cfg.use_chunks:
            is_root, clen, crootdep, amods = self._chunk_info(token, chunk_map)
            if clen > 1: out.append((10, is_root, clen, crootdep, amods))
        if self.cfg.use_neighbor_pos:
            nctx = self._neighbors(doc, token.i)
            if nctx: out.append((11, token.pos_, token.dep_, nctx))
        out.append((12, token.pos_, token.dep_, token.shape_, token.like_num, token.is_title, token.is_upper))
        out.append((13, token.pos_, token.dep_, self._sent_pos(token)))
        if token.lemma_.lower() not in self.cfg.risky_hl_heads:
            out.append((14, 'HL_ONLY', token.lemma_, token.pos_))
        out.append((15, token.pos_, token.dep_, self._morph_bits(token)))
        out.append((16, token.pos_, token.dep_, self._affixes(token)))
        out.append((17, token.pos_, token.dep_, self._subtree_size(token)))
        out.append((18, token.pos_, token.dep_, self._head_path(token, k=2)))
        out.append((19, token.pos_, token.dep_, token.head.pos_, token.head.dep_))
        if self.cfg.use_tags:
            out.append((20, token.tag_, token.dep_, token.head.tag_ if token.head is not None else None))
        if self.cfg.use_entity_iob:
            out.append((21, token.pos_, token.dep_, token.ent_iob_))
        if self.cfg.use_is_stop:
            out.append((22, token.pos_, token.dep_, token.is_stop))
        if self.cfg.dep_path_k > 0:
            out.append((23, token.pos_, tuple(self._dep_path_to_root(token, self.cfg.dep_path_k))))
        if self.cfg.use_head_morph and token.head is not None:
            out.append((24, token.pos_, token.dep_, tuple(sorted(str(token.head.morph).split("|"))) if str(token.head.morph) else (), token.head.shape_))
        if self.cfg.use_head_children_pos:
            out.append((25, token.pos_, token.dep_, self._head_children_pos(token)))
        if self.cfg.use_dep_distance:
            out.append((26, token.pos_, token.dep_, self._dep_distance(token)))
        if self.cfg.use_neighbor_lemmas:
            nl = self._neighbor_lemmas(doc, token.i)
            if nl: out.append((27, token.pos_, token.dep_, nl))
        if self.cfg.use_coord_siblings:
            cs = self._coord_siblings(token)
            if cs: out.append((28, token.pos_, token.dep_, cs))
        if self.cfg.use_det_child or self.cfg.use_prep_child:
            out.append((29, token.pos_, token.dep_, self._has_det_child(token), self._has_prep_child(token)))
        if self.cfg.use_in_chunk:
            in_chunk = token.i in chunk_map
            out.append((30, token.pos_, token.dep_, in_chunk))
        if self.cfg.use_sent_len_bucket:
            out.append((31, token.pos_, token.dep_, self._sent_len_bucket(token)))
        child_focus = sorted({c.lemma_.lower() for c in token.children if c.dep_ in {"amod","compound","conj","pobj","dobj","attr","acomp","xcomp","advmod"}})
        if child_focus:
            limit = self.cfg.child_profile_k if self.cfg.child_profile_k > 0 else len(child_focus)
            out.append((32, token.pos_, token.dep_, tuple(child_focus[:limit])))
        if token.head is not None:
            head_focus = sorted({c.lemma_.lower() for c in token.head.children if c.dep_ in {"attr","dobj","pobj","acomp","xcomp","conj"} and c.i != token.i})
            if head_focus:
                limit = self.cfg.child_profile_k if self.cfg.child_profile_k > 0 else len(head_focus)
                out.append((33, token.pos_, token.dep_, token.head.lemma_.lower(), tuple(head_focus[:limit])))
        return out

    def prepare(self, raw_items, update_stats=True):
        self.log.info("\n--- Pre-processing ---")
        t0 = time.time()
        texts = [it["text"] for it in raw_items]
        docs = list(self.nlp.pipe(texts, n_process=self.cfg.n_process, batch_size=self.cfg.batch_size))
        prepped = []
        gold_heads = []
        single_head_freq = Counter()
        if update_stats:
            self.surface_positive_counts.clear()
        for it, doc in zip(raw_items, docs):
            chunk_map = {w.i: chunk for chunk in doc.noun_chunks for w in chunk}
            gold_terms, gold_roots = set(), []
            for asp in it.get("aspects_with_offsets", []):
                span = doc.char_span(asp["from"], asp["to"])
                if span:
                    root = span.root
                    surf = self._np_text(root).lower()
                    if self._valid_surface(surf):
                        gold_terms.add(surf); gold_roots.append(root)
                        hl = root.lemma_.lower()
                        gold_heads.append(hl)
                        if surf == hl: single_head_freq.update([hl])
                        if update_stats:
                            self.surface_positive_counts.update([surf])
            sigs = [self._sig_variants(t, doc, chunk_map) for t in doc]
            sid = {tok.i: si for si, s in enumerate(doc.sents) for tok in s}
            prepped.append({"doc": doc, "gold_terms": gold_terms, "gold_roots": gold_roots, "sigs": sigs, "sent_id": sid})
        if update_stats:
            self.aspect_lex_head_lemmas = set(gold_heads)
            self.gold_head_freq = Counter(gold_heads)
            self.frequent_single_gold_heads = {w for w,c in single_head_freq.items() if c >= 10}
            self._build_surface_whitelist()
        self.log.info(f"Pre-processing done in {time.time()-t0:.2f}s")
        return prepped

    def _allowed_by_context(self, tok, gate_union):
        hl = tok.lemma_.lower()
        entity_ok = self.cfg.use_entities and tok.ent_type_ in {"PRODUCT", "ORG", "WORK_OF_ART", "MONEY", "FAC"}
        compound_ok = self._compound_len(tok) >= 2
        frequent_head = hl in self.frequent_single_gold_heads or self.gold_head_freq[hl] >= self.cfg.adapt_head_min_support
        if tok.pos_ == "VERB" and hl not in gate_union and hl not in self.aspect_lex_head_lemmas and not frequent_head:
            return False
        if gate_union:
            if hl in gate_union: return True
            if hl in self.cfg.generic_block_heads and hl not in gate_union: return False
            if entity_ok or compound_ok or (self.cfg.soft_restrict and hl in self.aspect_lex_head_lemmas) or frequent_head:
                return True
        if hl in self.cfg.hard_block_singletons and hl not in gate_union:
            return False
        if hl in self.cfg.generic_block_heads: return False
        if self.cfg.soft_restrict and hl in self.aspect_lex_head_lemmas: return True
        if frequent_head: return True
        if compound_ok: return True
        if entity_ok: return True
        return False

    def predict_doc(self, doc, sigs, rules, skip_surface_filter=False):
        res = set()
        for tok, sv in zip(doc, sigs):
            surf = self._np_text(tok)
            if surf.lower() in self.block_phrases: continue
            fire, gate_union = False, set()
            for s in sv:
                if s in rules:
                    fire = True
                    gate_union.update(self._filter_allowed_heads(self.rule_allowed_heads.get(s, set())))
                    if s[0] in (6,7,14):
                        lemma_lower = tok.lemma_.lower()
                        if lemma_lower not in self.cfg.generic_block_heads or self.gold_head_freq[lemma_lower] >= self.cfg.protect_heads_min_freq:
                            gate_union.add(lemma_lower)
            if not fire: continue
            if not self._allowed_by_context(tok, gate_union): continue
            if self._valid_surface(surf): res.add(surf)
        res = self._augment_with_surface_whitelist(doc, res)
        final = set()
        for surf in res:
            parts = surf.split()
            if len(parts) == 1 and parts[0] in self.cfg.hard_block_singletons:
                continue
            if skip_surface_filter or self._should_keep_surface(surf):
                final.add(surf)
        return sorted(final)

    def _evaluate_prepared(self, prepared, rules):
        tp = pp = ap = 0
        for ex in prepared:
            pred = {t.lower() for t in self.predict_doc(ex["doc"], ex["sigs"], rules)}
            gt = ex["gold_terms"]
            tp += len(pred & gt); pp += len(pred); ap += len(gt)
        p = tp / pp if pp else 0.0
        r = tp / ap if ap else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return {"precision": p, "recall": r, "f1": f1, "tp": tp, "pp": pp, "ap": ap}

    def _score_candidates(self, residual, existing_rules, chronic_focus=None):
        pos_counts, tot_counts, head_hits = Counter(), Counter(), defaultdict(set)
        combined = set().union(*existing_rules) if existing_rules else set()
        for ex in residual:
            pred = {t.lower() for t in self.predict_doc(ex["doc"], ex["sigs"], combined)}
            missed_roots = [r for r in ex["gold_roots"] if self._np_text(r).lower() not in pred]
            for tok in missed_roots:
                for sv in ex["sigs"][tok.i]:
                    pos_counts[sv] += 1
                    head_hits[sv].add(tok.lemma_.lower())
            for sv_list in ex["sigs"]:
                for s in sv_list: tot_counts[s] += 1
        scored = []
        min_sup = self.cfg.min_pos_support
        for sig, pos in pos_counts.items():
            needed = self.cfg.adapt_head_min_support if chronic_focus and any(h in chronic_focus for h in head_hits[sig]) else min_sup
            if pos < needed: continue
            prec = pos / max(tot_counts[sig], pos)
            if prec >= self.cfg.min_rule_precision and sig not in combined:
                scored.append((sig, pos, tot_counts[sig], prec))
        scored.sort(key=lambda x: (-x[3], -x[1], x[2]))
        return scored, head_hits

    def _build_sample_index(self, sample):
        index = defaultdict(list)
        for i, ex in enumerate(sample):
            for j, sv_list in enumerate(ex["sigs"]):
                for s in sv_list: index[s].append((i, j))
        return index

    def _baseline_sets(self, sample, combined):
        pred_sets = [set() for _ in sample]
        tp = pp = ap = 0
        for i, ex in enumerate(sample):
            pred = {t.lower() for t in self.predict_doc(ex["doc"], ex["sigs"], combined)}
            pred_sets[i] = pred
            gt = ex["gold_terms"]
            tp += len(pred & gt); pp += len(pred); ap += len(gt)
        return pred_sets, tp, pp, ap

    def _sig_delta_on_sample(self, sig, allowed_heads, sample, sample_index, pred_sets):
        add_tp, add_fp = 0, 0
        seen = [set() for _ in sample]
        for i, j in sample_index.get(sig, []):
            ex = sample[i]
            tok = ex["doc"][j]
            surf = self._np_text(tok).lower()
            if surf in pred_sets[i] or surf in seen[i] or surf in self.block_phrases or not self._valid_surface(surf): continue
            gate_union = self._filter_allowed_heads(allowed_heads) | self._filter_allowed_heads(self.rule_allowed_heads.get(sig, set()))
            if sig[0] in (6,7,14): gate_union.add(tok.lemma_.lower())
            if not self._allowed_by_context(tok, gate_union): continue
            seen[i].add(surf)
            if surf in ex["gold_terms"]: add_tp += 1
            else: add_fp += 1
        return add_tp, add_fp

    def _select_by_f1_gain_fast(self, sample, combined, scored, head_hits, budget):
        pred_sets, tp, pp, ap = self._baseline_sets(sample, combined)
        base_p = tp / pp if pp else 0.0
        base_r = tp / ap if ap else 0.0
        base_f1 = 2*base_p*base_r/(base_p+base_r) if (base_p+base_r) else 0.0
        index = self._build_sample_index(sample)
        gains = []
        cache = {}
        for sig, pos, tot, prec in scored[:self.cfg.f1_eval_topk]:
            atp, afp = self._sig_delta_on_sample(sig, head_hits.get(sig, set()), sample, index, pred_sets)
            if atp == 0 and afp == 0: continue
            m_pp, m_tp = pp + atp + afp, tp + atp
            m_p = m_tp / m_pp if m_pp else 0.0
            m_r = m_tp / ap if ap else 0.0
            m_f1 = 2*m_p*m_r/(m_p+m_r) if (m_p+m_r) else 0.0
            p_ok = (m_p + self.cfg.pr_margin >= min(self.cfg.target_precision, base_p)) or (m_p >= self.cfg.target_precision)
            if m_f1 - base_f1 > 0 and p_ok:
                gains.append((m_f1 - base_f1, sig, pos, tot, prec, atp, afp, m_p, m_r, m_f1))
                cache[sig] = {"Δtp":atp, "Δfp":afp, "pos":pos, "tot":tot, "rule_prec":prec, "m_p":m_p, "m_r":m_r, "m_f1":m_f1}
        gains.sort(key=lambda x: (-x[0], -x[4], -x[2]))
        out = [g[1] for g in gains[:budget]]
        allowed = {sig: head_hits[sig] for sig in out}
        self.last_selection_cache = cache
        return set(out), allowed, base_f1, gains

    def _learn_rules_iteratively(self, residual, existing_rule_sets):
        stage_rules = set()
        rng = random.Random(13)
        sample = residual if len(residual) <= self.cfg.f1_eval_sample else rng.sample(residual, self.cfg.f1_eval_sample)
        last_base_f1 = -1.0
        miss_heads = self._estimate_chronic_heads(residual)
        self.chronic_heads = {h for h, _ in miss_heads[:self.cfg.head_focus_k]}
        if self.chronic_heads:
            self.log.info(f"  Chronic heads focus: {list(self.chronic_heads)[:10]}")
        for it in range(1, self.cfg.max_iterations_per_stage + 1):
            current_rules = existing_rule_sets + [stage_rules]
            scored, head_hits = self._score_candidates(residual, current_rules, chronic_focus=self.chronic_heads)
            if not scored:
                self.log.info("  No viable candidates.")
                break
            dyn_budget = min(self.cfg.new_rules_per_iteration, max(16, len(scored)//8))
            combined = set().union(*current_rules)
            selected, allowed, base_f1, gains = self._select_by_f1_gain_fast(sample, combined, scored, head_hits, dyn_budget)
            self.log.info(f"  Iter {it}: evaluated={min(len(scored), self.cfg.f1_eval_topk)} selected={len(selected)} baseF1={base_f1:.4f}")
            if not selected:
                self.log.info("  No candidates improved under constraints.")
                break
            stage_rules |= selected
            for k,v in allowed.items():
                filtered_heads = self._filter_allowed_heads(v)
                if filtered_heads:
                    self.rule_allowed_heads.setdefault(k, set()).update(filtered_heads)
            gain = base_f1 - last_base_f1 if last_base_f1 >= 0 else base_f1
            last_base_f1 = base_f1
            if it > 2 and gain < self.cfg.min_gain_per_stage and len(residual) < self.cfg.early_stop_residual:
                self.log.info("  Early stop: diminishing returns.")
                break
        return stage_rules

    def _estimate_chronic_heads(self, residual):
        c = Counter()
        combined = set().union(*self.rule_stages) if self.rule_stages else set()
        for ex in residual:
            pred = {t.lower() for t in self.predict_doc(ex["doc"], ex["sigs"], combined)}
            for r in ex["gold_roots"]:
                if self._np_text(r).lower() not in pred: c.update([r.lemma_.lower()])
        return c.most_common()

    def _harvest_block_phrases(self, prepared, rules, top_k=80):
        fp_counter = Counter()
        for ex in prepared:
            pred = self.predict_doc(ex["doc"], ex["sigs"], rules)
            fps = {p.lower() for p in pred if p.lower() not in ex["gold_terms"]}
            fp_counter.update(fps)
        candidates = []
        for surf, c in fp_counter.most_common(top_k):
            toks = surf.split()
            if len(toks) == 1 and not self.cfg.block_singletons: continue
            if toks and toks[-1] in self.cfg.generic_block_heads and self.gold_head_freq[toks[-1]] < self.cfg.protect_heads_min_freq:
                candidates.append(surf)
            if len(candidates) >= self.cfg.max_block_phrases: break
        return set(candidates)

    def _score_rules_on_slice(self, rules, sample, combined):
        rng = random.Random(23)
        val = sample if len(sample) <= 1000 else rng.sample(sample, 1000)
        index = self._build_sample_index(val)
        pred_sets, tp, pp, ap = self._baseline_sets(val, combined)
        per_rule = {r: self._sig_delta_on_sample(r, self.rule_allowed_heads.get(r,set()), val, index, pred_sets) for r in rules}
        return per_rule, (tp, pp, ap)

    def _auto_lambda(self, tp, pp, ap, target_p):
        p = tp/pp if pp else 1.0
        return max(1.0, (target_p - p) * 50.0) if p < target_p else 1.0

    def _prune_rules(self, prepared, existing_rule_sets, new_stage_rules):
        if not new_stage_rules: return new_stage_rules
        before = self._evaluate_prepared(prepared, set().union(*existing_rule_sets))
        after_all = self._evaluate_prepared(prepared, set().union(*existing_rule_sets, new_stage_rules))
        if self.cfg.prune_strategy != "keep_topk":
            all_rules = set().union(*existing_rule_sets, new_stage_rules)
            base_f1 = after_all["f1"]
            keep = set(new_stage_rules)
            dropped = 0
            for r in sorted(list(new_stage_rules), key=str):
                if len(keep) <= self.cfg.min_stage_keep: break
                trial = all_rules - {r}
                m = self._evaluate_prepared(prepared, trial)
                if m["f1"] >= base_f1 - self.cfg.max_prune_drop:
                    keep.remove(r); all_rules.remove(r); base_f1 = m["f1"]; dropped += 1
            self.log.info(f"  Pruned {dropped}/{len(new_stage_rules)}; kept {len(keep)}.")
            return keep
        if after_all["f1"] - before["f1"] >= self.cfg.no_prune_if_stage_gain_ge:
            self.log.info("  Skipping prune: stage delivered strong net F1 gain.")
            return new_stage_rules
        combined = set().union(*existing_rule_sets)
        per_rule, base = self._score_rules_on_slice(new_stage_rules, prepared, combined)
        tp, pp, ap = base
        lam = self._auto_lambda(tp, pp, ap, self.cfg.target_precision)
        scored = sorted([(per_rule[r][0] - lam*per_rule[r][1], per_rule[r][0], per_rule[r][1], r) for r in new_stage_rules], reverse=True)
        keep = [r for _,_,_,r in scored[:max(self.cfg.min_stage_keep, min(self.cfg.stage_keep_topk, len(scored)))]]
        self.log.info(f"  Prune keep_topk: kept {len(keep)}/{len(new_stage_rules)} (λ={lam:.2f})")
        return set(keep)

    def train(self, training_data):
        prepared = self.prepare(training_data)
        self.rule_stages = []
        self.surface_precision = {}
        t0 = time.time()
        for stage in range(1, self.cfg.max_stages + 1):
            combined = set().union(*self.rule_stages) if self.rule_stages else set()
            m = self._evaluate_prepared(prepared, combined)
            self.log.info(f"\n--- Stage {stage}/{self.cfg.max_stages} --- P:{m['precision']:.4f} R:{m['recall']:.4f} F1:{m['f1']:.4f} | TP:{m['tp']} PP:{m['pp']} AP:{m['ap']}")
            if m["f1"] >= self.cfg.target_f1:
                self.log.info("Targets reached; stopping.")
                break
            residual = [ex for ex in prepared if not ex["gold_terms"].issubset({t.lower() for t in self.predict_doc(ex["doc"], ex["sigs"], combined)})]
            if not residual:
                self.log.info("No residuals; stopping.")
                break
            stage_rules = self._learn_rules_iteratively(residual, self.rule_stages)
            if not stage_rules:
                self.log.info("No rules learned; stopping.")
                break
            if self.cfg.prune:
                stage_rules = self._prune_rules(prepared, self.rule_stages, stage_rules)
            if not stage_rules:
                recov = sorted(list(self.last_selection_cache.items()), key=lambda kv: (-(kv[1].get("Δtp",0) - kv[1].get("Δfp",0)), -kv[1].get("Δtp",0)))[:self.cfg.min_stage_keep]
                stage_rules = set([kv[0] for kv in recov])
                if not stage_rules: break
            self.rule_stages.append(stage_rules)
            self.block_phrases.update(self._harvest_block_phrases(prepared, set().union(*self.rule_stages), top_k=80))
        self._refresh_surface_precision(prepared)
        self.training_time_sec = time.time() - t0
        self.log.info(f"\n--- Training Done in {self.training_time_sec:.2f}s ---")

    def predict(self, text):
        doc = self.nlp(text)
        chunk_map = {w.i: chunk for chunk in doc.noun_chunks for w in chunk}
        sigs = [self._sig_variants(t, doc, chunk_map) for t in doc]
        combined = set().union(*self.rule_stages)
        return self.predict_doc(doc, sigs, combined)

    def analyze(self, sentences: List[str]) -> Dict[int, Dict[str, Any]]:
        return self.predict(" ".join(sentences))

    def save(self, file_path: str):
        state = {
            "cfg": self.cfg.__dict__,
            "rule_stages": [list(stage) for stage in self.rule_stages],
            "rule_allowed_heads": {str(k): list(v) for k, v in self.rule_allowed_heads.items()},
            "block_phrases": list(self.block_phrases),
            "surface_precision": self.surface_precision,
            "gold_head_freq": self.gold_head_freq
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        logger.info(f"Rule extractor state saved to {file_path}")

    @classmethod
    def load(cls, file_path: str):
        file_path = str(file_path)
        if file_path.endswith('.pkl'):
            import __main__
            __main__.TrainConfig = TrainConfig
            with open(file_path, 'rb') as f:
                state = pickle.load(f)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                state = json.load(f)

        cfg_data = state["cfg"]
        if hasattr(cfg_data, '__dict__'):
            cfg_data = cfg_data.__dict__
        cfg_data["generic_block_heads"] = frozenset(cfg_data.get("generic_block_heads", []))
        cfg_data["hard_block_singletons"] = frozenset(cfg_data.get("hard_block_singletons", []))
        cfg_data["risky_hl_heads"] = frozenset(cfg_data.get("risky_hl_heads", []))
        
        config_fields = {f.name for f in fields(TrainConfig) if f.init}
        filtered_cfg_data = {k: v for k, v in cfg_data.items() if k in config_fields}
        
        cfg = TrainConfig(**filtered_cfg_data)
        extractor = cls(cfg)

        extractor.rule_stages = [set(map(tuple, stage)) for stage in state["rule_stages"]]
        
        extractor.rule_allowed_heads = {
            k: set(v) for k, v in state.get("rule_allowed_heads", {}).items()
        }
        extractor.block_phrases = set(state.get("block_phrases", []))
        extractor.surface_precision = state.get("surface_precision", {})
        extractor.gold_head_freq = Counter(state.get("gold_head_freq", {}))
        
        logger.info(f"Rule extractor loaded from {file_path}.")
        return extractor

@dataclass(frozen=True)
class Mention:
    span: spacy.tokens.Span
    origin: str = "other"
    clause_boundaries: List[Tuple[int, int]] = field(default_factory=list)
    text: str = field(init=False)
    head_lemma: str = field(init=False)
    is_pronoun: bool = field(init=False)
    clause_id: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'text', self.span.text.strip())
        object.__setattr__(self, 'head_lemma', self.span.root.lemma_.lower())
        object.__setattr__(self, 'is_pronoun', self.span.root.pos_ == 'PRON')
        
        clause_id = 0
        if self.clause_boundaries:
            span_start_char = self.span.start_char
            for i, (start, end) in enumerate(self.clause_boundaries):
                if start <= span_start_char < end:
                    clause_id = i
                    break
        else:
            for i, sent in enumerate(self.span.doc.sents):
                if self.span.start >= sent.start and self.span.end <= sent.end:
                    clause_id = i
                    break
        object.__setattr__(self, 'clause_id', clause_id)

    def __hash__(self):
        return hash((self.span.start_char, self.span.end_char))

    def __eq__(self, other):
        if not isinstance(other, Mention):
            return False
        return self.span.start_char == other.span.start_char and self.span.end_char == other.span.end_char

class HybridAspectExtractor(ATE):
    def __init__(self, rule_extractor_path: str = None, spacy_model: str = "en_core_web_sm", device: str = None):
        if device is None:
            device = get_optimal_device()
        self.device = device
        path = rule_extractor_path or config.ASPECT_EXTRACTOR_RULES
        self.rule_extractor = MultiStageRuleExtractor.load(path)
        self.nlp = get_cached_spacy_model(spacy_model)
        self.mav = get_cached_maverick_model(device)
        logger.info(f"HybridAspectExtractor initialized with device: {device}")
        self._preferred_heads = {
            "restaurant", "service", "ambiance", "ambience", "staff",
            "server", "food", "pizza", "crust", "drink", "drinks",
            "training", "price", "menu", "place", "battery", "screen",
            "keyboard", "performance", "laptop", "computer", "display"
        }

    def _normalize_for_output(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        lowered = cleaned.lower()
        for prefix in ("the ", "this ", "that ", "these ", "those ", "a ", "an ", "my ", "our "):
            if lowered.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break
        return cleaned.strip() or text.strip()

    def _mention_bucket(self, mention: Mention) -> str:
        ent = mention.span.root.ent_type_ or getattr(mention.span, "label_", None)
        if mention.is_pronoun:
            return "PRON"
        if ent in {"PERSON"}:
            return "PERSON"
        if ent in {"ORG", "FAC", "GPE", "LOC", "PRODUCT"}:
            return "ENTITY"
        if mention.span.root.pos_ == "PROPN":
            return "ENTITY"
        return "GENERIC"

    def _compatible_bucket(self, bucket: str, existing: set[str]) -> bool:
        if bucket == "PRON":
            return True
        core_existing = {b for b in existing if b not in {"PRON", "GENERIC"}}
        if not core_existing:
            return True
        if bucket == "GENERIC":
            return True
        return bucket in core_existing

    def _is_semantically_compatible(self, mention: Mention, cluster: List[Mention]) -> bool:
        bucket = self._mention_bucket(mention)
        existing = {self._mention_bucket(m) for m in cluster}
        return self._compatible_bucket(bucket, existing)

    def _span_overlap_ratio(self, mention: Mention, start: int, end: int) -> float:
        m_start, m_end = mention.span.start_char, mention.span.end_char
        overlap = max(0, min(m_end, end) - max(m_start, start))
        if overlap == 0:
            return 0.0
        target = min(m_end - m_start, end - start)
        return overlap / target if target else 0.0

    def _dedupe_cluster(self, cluster: List[Mention]) -> List[Mention]:
        seen = set()
        ordered = sorted(cluster, key=lambda m: (m.clause_id, m.span.start, m.span.end))
        result = []
        for mention in ordered:
            if mention not in seen:
                seen.add(mention)
                result.append(mention)
        return result

    def _mention_source_priority(self, mention: Mention) -> int:
        return {
            "rule": 4,
            "entity": 3,
            "chunk": 2,
            "noun": 1,
            "pron": 0,
            "other": 0,
        }.get(mention.origin, 0)

    def _mention_priority(self, mention: Mention) -> tuple:
        bucket = self._mention_bucket(mention)
        bucket_score = {
            "GENERIC": 4,
            "ENTITY": 3,
            "PERSON": 1,
            "PRON": 0,
        }.get(bucket, 2)
        source_bonus = self._mention_source_priority(mention) * 2
        tokens = list(mention.span)
        token_len = len(tokens)
        char_len = len(self._normalize_for_output(mention.text))
        proper_tokens = sum(1 for t in tokens if t.pos_ == "PROPN")
        is_proper_phrase = proper_tokens == token_len and token_len > 0
        
        # Strong boost for proper names (people, places, etc.)
        proper_name_bonus = 10 if is_proper_phrase else 0
        
        head = mention.span.root.lemma_.lower()
        head_bonus = 3 if head in self._preferred_heads else 0
        coverage_bonus = 1 if token_len > 1 else 0
        
        return (
            bucket_score + head_bonus + coverage_bonus + source_bonus + proper_name_bonus,
            -is_proper_phrase,
            token_len,
            char_len,
            -mention.clause_id,
        )

    def _merge_clusters(self, clusters: List[List[Mention]], dst_idx: int, src_idx: int, assignment_map: Dict[Mention, int]) -> None:
        if dst_idx == src_idx or src_idx < 0 or src_idx >= len(clusters):
            return
        for mention in clusters[src_idx]:
            if mention not in clusters[dst_idx]:
                clusters[dst_idx].append(mention)
                assignment_map[mention] = dst_idx
        clusters[src_idx] = []

    def _build_clause_candidates(self, mentions: List[Mention]) -> Dict[int, List[Mention]]:
        clause_map: Dict[int, List[Mention]] = defaultdict(list)
        for mention in mentions:
            if mention.is_pronoun:
                continue
            clause_map[mention.clause_id].append(mention)
        for clause_mentions in clause_map.values():
            clause_mentions.sort(key=self._mention_priority, reverse=True)
        return clause_map

    def _generate_clause_fallback_mentions(
        self,
        doc: spacy.tokens.Doc,
        clause_id: int,
        clause_boundaries: List[Tuple[int, int]],
    ) -> List[Mention]:
        if clause_id < 0 or clause_id >= len(clause_boundaries):
            return []
        start_char, end_char = clause_boundaries[clause_id]
        span = doc.char_span(start_char, end_char, alignment_mode="expand")
        if span is None:
            return []

        seen_offsets = set()
        fallback_mentions: List[Mention] = []
        for token in span:
            if token.pos_ not in {"NOUN", "PROPN"}:
                continue
            if token.is_stop:
                continue
            token_span = doc[token.i : token.i + 1]
            offsets = (token_span.start_char, token_span.end_char)
            if offsets in seen_offsets:
                continue
            seen_offsets.add(offsets)
            fallback_mentions.append(Mention(token_span, origin="fallback", clause_boundaries=clause_boundaries))
        fallback_mentions.sort(key=self._mention_priority, reverse=True)
        return fallback_mentions

    def _apply_pronoun_fallback(
        self,
        clusters: List[List[Mention]],
        mentions: List[Mention],
        clause_boundaries: List[Tuple[int, int]],
        doc: spacy.tokens.Doc,
    ) -> None:
        assignment_map: Dict[Mention, int] = {}
        for idx, cluster in enumerate(clusters):
            for mention in cluster:
                assignment_map[mention] = idx

        clause_candidates = self._build_clause_candidates(mentions)

        for idx, cluster in enumerate(clusters):
            if not cluster or any(not mention.is_pronoun for mention in cluster):
                continue

            clause_id = min(mention.clause_id for mention in cluster)
            candidates = clause_candidates.get(clause_id, [])
            if not candidates:
                if clause_candidates:
                    nearest_clause = min(
                        clause_candidates.keys(),
                        key=lambda cid: (abs(cid - clause_id), cid),
                    )
                    candidates = clause_candidates.get(nearest_clause, [])
            if not candidates:
                candidates = self._generate_clause_fallback_mentions(doc, clause_id, clause_boundaries)
            if not candidates:
                continue

            fallback = max(candidates, key=self._mention_priority)
            existing_idx = assignment_map.get(fallback)
            if existing_idx is not None and existing_idx != idx:
                self._merge_clusters(clusters, idx, existing_idx, assignment_map)
            else:
                if fallback not in cluster:
                    cluster.append(fallback)
                    assignment_map[fallback] = idx

    def _select_canonical(self, cluster: List[Mention]) -> Mention:
        non_pronouns = [m for m in cluster if not m.is_pronoun]
        if not non_pronouns:
            return cluster[0]
        return max(non_pronouns, key=self._mention_priority)

    def _ordered_mentions(self, cluster: List[Mention], canonical: Mention) -> List[Tuple[str, int]]:
        canonical_key = (self._normalize_for_output(canonical.text), canonical.clause_id)
        ordered = [canonical_key]
        seen = {canonical_key}
        sorted_mentions = sorted(
            cluster,
            key=lambda m: (
                m is canonical,
                -self._mention_source_priority(m),
                -(m.span.end_char - m.span.start_char),
                -len(list(m.span)),
                m.clause_id,
                m.span.start,
            ),
            reverse=True,
        )
        for mention in sorted_mentions:
            key = (self._normalize_for_output(mention.text), mention.clause_id)
            if key in seen:
                continue
            ordered.append(key)
            seen.add(key)
        return ordered

    def _assign_coref_clusters(self, mentions: List[Mention], coref_offsets: List[List[Tuple[int, int]]]):
        cluster_mentions: Dict[int, List[Mention]] = defaultdict(list)
        cluster_buckets: Dict[int, set[str]] = defaultdict(set)
        mention_to_cluster: Dict[Mention, int] = {}

        for cluster_id, chain in enumerate(coref_offsets):
            for mention in mentions:
                if mention in mention_to_cluster:
                    continue
                for start_char, end_char in chain:
                    if self._span_overlap_ratio(mention, start_char, end_char) >= 0.6:
                        bucket = self._mention_bucket(mention)
                        if not self._compatible_bucket(bucket, cluster_buckets[cluster_id]):
                            continue
                        cluster_mentions[cluster_id].append(mention)
                        cluster_buckets[cluster_id].add(bucket)
                        mention_to_cluster[mention] = cluster_id
                        break

        clusters: List[List[Mention]] = []
        cluster_id_mapping: Dict[int, int] = {}
        for coref_id in sorted(cluster_mentions.keys()):
            deduped = self._dedupe_cluster(cluster_mentions[coref_id])
            if not deduped:
                continue
            cluster_id_mapping[coref_id] = len(clusters)
            clusters.append(deduped)

        return clusters, mention_to_cluster, cluster_id_mapping

    def _find_spans_for_text(self, doc: spacy.tokens.Doc, term: str) -> List[spacy.tokens.Span]:
        spans = []
        for match in re.finditer(re.escape(term), doc.text, re.IGNORECASE):
            span = doc.char_span(match.start(), match.end())
            if span:
                spans.append(span)
        return spans

    def _extract_candidates(self, doc: spacy.tokens.Doc, clause_boundaries: List[Tuple[int, int]]) -> List[Mention]:
        mentions = set()
        text = doc.text

        # Priority 1: Rule-based extraction (highest priority)
        rule_aspects = self.rule_extractor.predict(text)
        for aspect_text in rule_aspects:
            for span in self._find_spans_for_text(doc, aspect_text):
                mentions.add(Mention(span, origin="rule", clause_boundaries=clause_boundaries))
        
        # Priority 2: Named entities from spaCy NER
        for ent in doc.ents:
            mentions.add(Mention(ent, origin="entity", clause_boundaries=clause_boundaries))
        
        # Priority 3: Noun chunks (all noun phrases)
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ in ("NOUN", "PROPN"):
                mentions.add(Mention(chunk, origin="chunk", clause_boundaries=clause_boundaries))
        
        # Priority 4: Individual nouns and proper nouns not captured above
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN"):
                span = doc[token.i:token.i+1]
                mentions.add(Mention(span, origin="noun", clause_boundaries=clause_boundaries))
        
        # Priority 5: Pronouns for coreference
        for token in doc:
            if token.pos_ == "PRON":
                span = doc[token.i:token.i+1]
                mentions.add(Mention(span, origin="pron", clause_boundaries=clause_boundaries))
        
        return sorted(list(mentions), key=lambda m: m.span.start)

    def analyze(self, sentences: List[str]) -> Dict[int, Dict[str, Any]]:
        if not sentences:
            return {}
        full_text = " ".join(sentences)
        doc = self.nlp(full_text)
        
        clause_boundaries = []
        current_pos = 0
        for sentence in sentences:
            clause_boundaries.append((current_pos, current_pos + len(sentence)))
            current_pos += len(sentence) + 1 
        
        all_mentions = self._extract_candidates(doc, clause_boundaries)
        if self.mav is not None:
            coref_output = self.mav.predict(full_text)
            coref_clusters = coref_output.get("clusters_char_offsets", [])
        else:
            coref_clusters = []
        clusters, mention_to_cluster, _ = self._assign_coref_clusters(all_mentions, coref_clusters)

        lemma_index: Dict[str, set] = defaultdict(set)
        for idx, cluster in enumerate(clusters):
            for mention in cluster:
                if not mention.is_pronoun:
                    lemma_index[mention.head_lemma].add(idx)

        unassigned = [m for m in all_mentions if m not in mention_to_cluster]
        for mention in unassigned:
            assigned_idx = None
            normalized_mention_text = normalize_text(mention.text)
            for idx, cluster in enumerate(clusters):
                if any(normalize_text(existing.text) == normalized_mention_text for existing in cluster):
                    if self._is_semantically_compatible(mention, cluster):
                        assigned_idx = idx
                        break
            if assigned_idx is None and not mention.is_pronoun:
                for idx in lemma_index.get(mention.head_lemma, set()):
                    if self._is_semantically_compatible(mention, clusters[idx]):
                        assigned_idx = idx
                        break
            if assigned_idx is None:
                assigned_idx = len(clusters)
                clusters.append([])
            clusters[assigned_idx].append(mention)
            if not mention.is_pronoun:
                lemma_index.setdefault(mention.head_lemma, set()).add(assigned_idx)

        self._apply_pronoun_fallback(clusters, all_mentions, clause_boundaries, doc)

        normalized_clusters = []
        for cluster in clusters:
            deduped = self._dedupe_cluster(cluster)
            if deduped:
                normalized_clusters.append(deduped)

        if not normalized_clusters:
            return {}

        normalized_clusters.sort(key=lambda c: (min(m.clause_id for m in c), min(m.span.start for m in c)))

        final_output = {}
        entity_id_counter = 1
        for cluster in normalized_clusters:
            canonical_mention = self._select_canonical(cluster)
            canonical_name = self._normalize_for_output(canonical_mention.text)
            mention_list = self._ordered_mentions(cluster, canonical_mention)
            if not mention_list:
                continue
            final_output[entity_id_counter] = {
                "first_mention": canonical_name,
                "mentions": mention_list
            }
            entity_id_counter += 1
        return final_output

def load_data(file_paths):
    all_data = []
    for file_path in file_paths:
        if not os.path.exists(file_path): continue
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            tag, term_key = ("aspectTerm", "term") if root.tag == "sentences" else ("Opinion", "target")
            for sent in root.findall(".//sentence"):
                if sent.get("OutOfScope") == "TRUE": continue
                text_el = sent.find("text")
                if text_el is None or text_el.text is None: continue
                aspects = []
                for term_el in sent.findall(f".//{tag}"):
                    term, fr, to = term_el.get(term_key), term_el.get("from"), term_el.get("to")
                    if term and term != "NULL" and fr and to:
                        aspects.append({"term": term, "from": int(fr), "to": int(to)})
                if aspects: all_data.append({"text": text_el.text, "aspects_with_offsets": aspects})
        except ET.ParseError: continue
    return all_data

if __name__ == "__main__":
    hybrid_system = HybridAspectExtractor()
    
    text_to_analyze = [
        "The food at Guido's was exceptional, but the service was a letdown.",
        "We ordered the pepperoni pizza, and it was delicious with a perfectly crispy crust.",
        "However, our server seemed overwhelmed and the staff needs more training.",
        "The ambiance of the restaurant itself is quite nice, though."
    ]

    print("\n--- Analyzing Text with Hybrid System ---")
    print("Input Sentences:")
    for i, sent in enumerate(text_to_analyze):
        print(f"  [{i}] {sent}")
    extraction_result = hybrid_system.analyze(text_to_analyze)
    
    print("\n--- Extraction Result ---")
    print(json.dumps(extraction_result, indent=2))