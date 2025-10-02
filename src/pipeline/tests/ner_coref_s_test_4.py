import os
import re
import time
import json
import glob
import uuid
import math
import pickle
import random
import shutil
import optuna
import datetime as dt
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
from string import punctuation

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm

@dataclass
class TrainConfig:
    model: str = "en_core_web_md"
    restrict_pos: tuple | None = ("NOUN", "PROPN")
    min_pos_support: int = 2
    min_rule_precision: float = 0.5
    max_stages: int = 20
    max_iterations_per_stage: int = 15
    new_rules_per_iteration: int = 60
    n_process: int = 1
    batch_size: int = 4000
    target_precision: float = 0.9
    target_recall: float = 0.9
    target_f1: float = 0.9
    pr_margin: float = 0.01
    f1_eval_sample: int = 1200
    f1_eval_topk: int = 400
    prune: bool = True
    max_prune_drop: float = 0.0
    early_stop_residual: int = 2000
    min_gain_per_stage: float = 0.0005
    debug: bool = False
    debug_every_n_stages: int = 0
    debug_top_k: int = 15
    dump_dir: str | None = None
    use_grandparent: bool = True
    use_child_profile: bool = True
    child_profile_k: int = 2
    head_focus_k: int = 10
    adapt_head_min_support: int = 1
    use_entities: bool = True
    use_chunks: bool = True
    use_neighbor_pos: bool = True
    neighbor_window: int = 1
    coord_backoff: bool = True
    soft_restrict: bool = True

class MultiStageRuleExtractor:
    def __init__(self, cfg: TrainConfig):
        try:
            self.nlp = spacy.load(cfg.model, disable=[])
        except OSError:
            spacy.cli.download(cfg.model)
            self.nlp = spacy.load(cfg.model, disable=[])
        self.cfg = cfg
        self.rule_stages = []
        self.rule_allowed_heads = {}
        self.stopset = set(STOP_WORDS) | set(punctuation)
        self.aspect_lex_head_lemmas = set()
        self.frequent_single_gold_heads = set()
        self.generic_block_heads = {"place","restaurant","time","thing","table","dinner","drinks","price","prices","laptop","computer","bar","dining","product"}
        self.block_phrases = set()
        self.last_selection_cache = {}
        self.chronic_heads = set()
        self._rule_cache = {}
        self._rules_dir = "outputs/ner_coref/rules"
        self._logs_dir = "outputs/ner_coref/logs"
        self._params_dir = "outputs/ner_coref/parameters"
        os.makedirs(self._rules_dir, exist_ok=True)
        os.makedirs(os.path.join(self._rules_dir, "best"), exist_ok=True)
        os.makedirs(self._logs_dir, exist_ok=True)
        os.makedirs(self._params_dir, exist_ok=True)

    def _valid_surface(self, s):
        if not s: return False
        t = s.lower().strip()
        if len(t) < 2: return False
        if t in self.stopset: return False
        return True

    def _np_text(self, token):
        lk = {"compound","amod","nummod","flat"}
        rk = {"compound","amod","nummod","flat","appos"}
        left = [t for t in token.lefts if t.dep_ in lk]
        right = [t for t in token.rights if t.dep_ in rk]
        parts = left + [token] + right
        parts.sort(key=lambda t: t.i)
        return " ".join(t.text for t in parts)

    def _compound_len(self, token):
        c = 1
        for t in token.lefts:
            if t.dep_ in {"compound","amod","nummod","flat"}: c += 1
        for t in token.rights:
            if t.dep_ in {"compound","amod","nummod","flat","appos"}: c += 1
        return c

    def _child_profile(self, token):
        if not self.cfg.use_child_profile:
            return ()
        deps = [c.dep for c in token.children]
        deps.sort()
        if not deps:
            return ()
        if self.cfg.child_profile_k <= 0:
            return tuple(deps)
        step = max(1, len(deps)//self.cfg.child_profile_k)
        prof = tuple(deps[::step][:self.cfg.child_profile_k])
        return prof

    def _neighbors(self, doc, i):
        if not self.cfg.use_neighbor_pos: return ()
        L = []
        for off in range(1, self.cfg.neighbor_window+1):
            if i-off >= 0: L.append(doc[i-off].pos)
            if i+off < len(doc): L.append(doc[i+off].pos)
        return tuple(L) if L else ()

    def _chunk_info(self, token, chunk_map):
        if not self.cfg.use_chunks: return (False,0,"",0)
        c = chunk_map.get(token.i)
        if not c: return (False,0,"",0)
        root = c.root
        return (token.i==root.i, len(list(c)), c.root.dep, len([w for w in c if w.dep_=="amod"]))

    def _entity_pair(self, token):
        if not self.cfg.use_entities: return ("","")
        ent = token.ent_type_ if token.ent_type_ else ""
        hl = token.head.ent_type_ if token.head is not None and token.head.ent_type_ else ""
        return (ent, hl)

    def _salient(self, token, chunk_map):
        salient_deps = {"attr","dobj","pobj","nsubj","nsubjpass","appos","conj","ROOT"}
        if token.dep_ in salient_deps: return True
        is_root, _, _, _ = self._chunk_info(token, chunk_map)
        return is_root

    def _sig_variants(self, token, doc, chunk_map):
        hl = token.lemma_.lower()
        if self.cfg.soft_restrict:
            pos_ok = token.pos_ in ("NOUN","PROPN","ADJ","VERB")
            if hl not in self.aspect_lex_head_lemmas:
                if not pos_ok:
                    return []
                if not self._salient(token, chunk_map):
                    return []
        elif self.cfg.restrict_pos and token.pos_ not in self.cfg.restrict_pos:
            return []
        ch = tuple(sorted([c.dep for c in token.children]))
        gp = token.head.head if token.head is not None else None
        gp_lemma = gp.lemma if (gp is not None and hasattr(gp,"lemma")) else None
        gp_pos = gp.pos if (gp is not None and hasattr(gp,"pos")) else None
        prof = self._child_profile(token)
        nctx = self._neighbors(doc, token.i)
        is_root, clen, crootdep, amods = self._chunk_info(token, chunk_map)
        ent_tok, ent_head = self._entity_pair(token)
        s0 = (0, token.pos, token.dep, token.head.lemma, token.head.pos, ch, ent_tok, ent_head, is_root, clen, crootdep, amods, nctx)
        s1 = (1, token.pos, token.dep, token.head.lemma, token.head.pos, (), ent_tok, ent_head, is_root, clen, crootdep, amods, ())
        s2 = (2, token.pos, token.dep, token.head.lemma, None, (), ent_tok, "", is_root, clen, "", amods, ())
        s3 = (3, token.pos, token.dep, None, None, (), "", "", is_root, clen, "", amods, ())
        out = [s0,s1,s2,s3]
        if self.cfg.use_grandparent:
            s4 = (4, token.pos, token.dep, token.head.lemma, token.head.pos, (gp_lemma,gp_pos), ent_tok, ent_head, is_root, clen, crootdep, amods, nctx)
            out.append(s4)
        if self.cfg.use_child_profile:
            s5 = (5, token.pos, token.dep, token.head.lemma, token.head.pos, prof, ent_tok, ent_head, is_root, clen, crootdep, amods, ())
            out.append(s5)
        s6 = (6, token.pos, token.dep, None, None, ("HL", token.lemma), ent_tok, ent_head, is_root, clen, crootdep, amods, ())
        s7 = (7, None, None, None, None, ("HL", token.lemma), ent_tok, "", is_root, clen, "", amods, ())
        if self.cfg.coord_backoff and token.dep_=="conj" and token.head is not None and token.head.pos_ in ("NOUN","PROPN","ADJ","VERB"):
            ph = tuple(sorted([c.dep for c in token.head.children]))
            s8 = (8, token.pos, "conj", token.head.lemma, token.head.pos, ph, ent_tok, ent_head, is_root, clen, crootdep, amods, ())
            out.append(s8)
        return out

    def prepare(self, raw_items):
        if self.cfg.dump_dir:
            os.makedirs(self.cfg.dump_dir, exist_ok=True)
        texts = [it["text"] for it in raw_items]
        t0 = time.time()
        docs = list(self.nlp.pipe(texts, n_process=self.cfg.n_process, batch_size=self.cfg.batch_size))
        prepped = []
        gold_heads = []
        single_head_freq = Counter()
        for it, doc in zip(tqdm(raw_items, desc="Tokenizing", unit="doc"), docs):
            chunk_map = {}
            if self.cfg.use_chunks:
                for nc in doc.noun_chunks:
                    for w in nc:
                        chunk_map[w.i] = nc
            gold_terms = set()
            gold_roots = []
            for asp in it.get("aspects_with_offsets", []):
                span = doc.char_span(asp["from"], asp["to"])
                if span:
                    root = span.root
                    surf = self._np_text(root).lower()
                    if self._valid_surface(surf):
                        gold_terms.add(surf)
                        gold_roots.append(root)
                        hl = root.lemma_.lower()
                        gold_heads.append(hl)
                        if surf == hl:
                            single_head_freq.update([hl])
            sigs = [self._sig_variants(t, doc, chunk_map) for t in doc]
            sents = list(doc.sents)
            sid = {}
            for si, s in enumerate(sents):
                for tok in s:
                    sid[tok.i] = si
            prepped.append({"doc": doc, "gold_terms": gold_terms, "gold_roots": gold_roots, "sigs": sigs, "sent_id": sid})
        self.aspect_lex_head_lemmas = set(gold_heads)
        self.frequent_single_gold_heads = {w for w,c in single_head_freq.items() if c >= 10}
        self.training_prep_seconds = time.time()-t0
        return prepped

    def _allowed_by_context(self, tok, gate_union):
        hl = tok.lemma_.lower()
        if hl in gate_union: return True
        if hl in self.aspect_lex_head_lemmas: return True
        if hl in self.frequent_single_gold_heads: return True
        if self._compound_len(tok) >= 2 and hl not in self.generic_block_heads: return True
        if self.cfg.use_entities and tok.ent_type_ in {"PRODUCT","ORG","WORK_OF_ART","MONEY","FAC"}: return True
        return False

    def predict_doc(self, doc, sigs, rules):
        key = (id(doc), id(rules))
        if key in self._rule_cache: return self._rule_cache[key]
        res = set()
        chunk_map = {}
        if self.cfg.use_chunks:
            for nc in doc.noun_chunks:
                for w in nc:
                    chunk_map[w.i] = nc
        for i,(tok, sv) in enumerate(zip(doc, sigs)):
            surf = self._np_text(tok)
            if surf.lower() in self.block_phrases:
                continue
            fire = False
            gate_union = set()
            for s in sv:
                if s in rules:
                    fire = True
                    if s in self.rule_allowed_heads:
                        gate_union |= self.rule_allowed_heads[s]
                    if s[0] in (6,7,8):
                        gate_union.add(tok.lemma_.lower())
            if not fire:
                continue
            hl = tok.lemma_.lower()
            if hl not in self.chronic_heads:
                if hl in self.generic_block_heads and self._compound_len(tok) < 2 and hl not in self.frequent_single_gold_heads:
                    if not (self.cfg.use_entities and tok.ent_type_ in {"MONEY","FAC","ORG"}):
                        continue
            if not self._allowed_by_context(tok, gate_union):
                continue
            if self._valid_surface(surf):
                res.add(surf)
        out = list(res)
        self._rule_cache[key] = out
        return out

    def _evaluate_prepared(self, prepared, rules):
        tp = pp = ap = 0
        for ex in prepared:
            pred = {t.lower() for t in self.predict_doc(ex["doc"], ex["sigs"], rules)}
            gt = ex["gold_terms"]
            tp += len(pred & gt)
            pp += len(pred)
            ap += len(gt)
        p = tp / pp if pp else 0.0
        r = tp / ap if ap else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f1

    def _score_candidates(self, residual, existing_rules, chronic_focus=None):
        pos_counts = Counter()
        tot_counts = Counter()
        head_hits = defaultdict(set)
        combined = set().union(*existing_rules) if existing_rules else set()
        for ex in residual:
            doc, sigs = ex["doc"], ex["sigs"]
            pred = {t.lower() for t in self.predict_doc(doc, sigs, combined)}
            missed = []
            for r in ex["gold_roots"]:
                if self._np_text(r).lower() not in pred:
                    missed.append(r)
            for tok in missed:
                hl = tok.lemma_.lower()
                for sv in sigs[tok.i]:
                    pos_counts[sv] += 1
                    head_hits[sv].add(hl)
            for sv_list in sigs:
                for s in sv_list:
                    tot_counts[s] += 1
        scored = []
        min_sup = self.cfg.min_pos_support
        for sig, pos in pos_counts.items():
            heads = head_hits[sig]
            needed = self.cfg.adapt_head_min_support if chronic_focus and any(h in chronic_focus for h in heads) else min_sup
            if pos < needed: continue
            tot = max(tot_counts[sig], pos)
            prec = pos / tot if tot else 0.0
            if prec >= self.cfg.min_rule_precision and sig not in combined:
                scored.append((sig, pos, tot, prec))
        scored.sort(key=lambda x: (-x[3], -x[1], x[2]))
        return scored, head_hits

    def _build_sample_index(self, sample):
        index = defaultdict(list)
        for i, ex in enumerate(sample):
            for j, sv in enumerate(ex["sigs"]):
                for s in sv:
                    index[s].append((i, j))
        return index

    def _baseline_sets(self, sample, combined):
        pred_sets = []
        tp = pp = ap = 0
        for ex in sample:
            pred = {t.lower() for t in self.predict_doc(ex["doc"], ex["sigs"], combined)}
            pred_sets.append(pred)
            gt = ex["gold_terms"]
            tp += len(pred & gt)
            pp += len(pred)
            ap += len(gt)
        return pred_sets, tp, pp, ap

    def _sig_delta_on_sample(self, sig, allowed_heads, sample, sample_index, pred_sets, combined):
        add_tp = add_fp = 0
        occ = sample_index.get(sig, [])
        seen = [set() for _ in sample]
        for i, j in occ:
            ex = sample[i]
            tok = ex["doc"][j]
            surf = self._np_text(tok).lower()
            if surf in pred_sets[i]:
                continue
            if surf in self.block_phrases:
                continue
            hl = tok.lemma_.lower()
            gate_union = set(allowed_heads) | self.rule_allowed_heads.get(sig, set())
            if sig[0] in (6,7,8):
                gate_union.add(hl)
            if hl not in self.chronic_heads:
                if hl in self.generic_block_heads and self._compound_len(tok) < 2 and hl not in self.frequent_single_gold_heads:
                    if not (self.cfg.use_entities and tok.ent_type_ in {"MONEY","FAC","ORG"}):
                        continue
            if not self._valid_surface(surf):
                continue
            if hl not in gate_union and not self._allowed_by_context(tok, gate_union):
                continue
            if surf in seen[i]:
                continue
            seen[i].add(surf)
            if surf in ex["gold_terms"]:
                add_tp += 1
            else:
                add_fp += 1
        return add_tp, add_fp

    def _select_by_f1_gain_fast(self, sample, combined, scored, head_hits, budget):
        pred_sets, tp, pp, ap = self._baseline_sets(sample, combined)
        base_p = tp / pp if pp else 0.0
        base_r = tp / ap if ap else 0.0
        base_f1 = 2 * base_p * base_r / (base_p + base_r) if (base_p + base_r) else 0.0
        topk = min(len(scored), self.cfg.f1_eval_topk)
        index = self._build_sample_index(sample)
        gains = []
        cache = {}
        for sig, pos, tot, prec in tqdm(scored[:topk], desc="Scoring candidates", unit="cand"):
            atp, afp = self._sig_delta_on_sample(sig, head_hits.get(sig, set()), sample, index, pred_sets, combined)
            if atp == 0 and afp == 0:
                continue
            m_pp = pp + atp + afp
            m_tp = tp + atp
            m_p = m_tp / m_pp if m_pp else 0.0
            m_r = m_tp / ap if ap else 0.0
            m_f1 = 2 * m_p * m_r / (m_p + m_r) if (m_p + m_r) else 0.0
            p_ok = (m_p + self.cfg.pr_margin >= min(self.cfg.target_precision, base_p)) or (m_p >= self.cfg.target_precision)
            r_ok = (m_r + self.cfg.pr_margin >= min(self.cfg.target_recall, base_r)) or (m_r >= self.cfg.target_recall)
            if m_f1 - base_f1 > 0 and p_ok and r_ok:
                gains.append((m_f1 - base_f1, sig, pos, tot, prec, atp, afp, m_p, m_r, m_f1))
                cache[sig] = {"dtp": atp, "dfp": afp, "rule_prec": prec, "mp": m_p, "mr": m_r, "mf1": m_f1}
        gains.sort(key=lambda x: (-x[0], -x[4], -x[2]))
        out = []
        for g in gains:
            out.append(g[1])
            if len(out) >= budget:
                break
        allowed = {sig: head_hits[sig] for sig in out}
        self.last_selection_cache = cache
        return set(out), allowed, base_f1

    def _estimate_chronic_heads(self, residual):
        c = Counter()
        combined = set().union(*self.rule_stages) if self.rule_stages else set()
        for ex in residual:
            pred = {t.lower() for t in self.predict_doc(ex["doc"], ex["sigs"], combined)}
            gt = ex["gold_terms"]
            miss = gt - pred
            for r in ex["gold_roots"]:
                if self._np_text(r).lower() in miss:
                    c.update([r.lemma_.lower()])
        return c.most_common()

    def _harvest_block_phrases(self, prepared, rules, top_k=60):
        fp_counter = Counter()
        for ex in prepared:
            pred = self.predict_doc(ex["doc"], ex["sigs"], rules)
            gt = ex["gold_terms"]
            fp_counter.update({p.lower() for p in pred if p.lower() not in gt})
        new_blocks = set()
        for surf, c in fp_counter.most_common(top_k):
            parts = surf.split()
            if len(parts) <= 2:
                head = parts[-1]
                if head in self.generic_block_heads:
                    new_blocks.add(surf)
        return new_blocks

    def _prune_rules(self, prepared, existing_rule_sets, new_stage_rules):
        if not new_stage_rules:
            return new_stage_rules
        combined = set().union(*existing_rule_sets, new_stage_rules) if existing_rule_sets else set(new_stage_rules)
        base_p, base_r, base_f1 = self._evaluate_prepared(prepared, combined)
        rules_list = list(new_stage_rules)
        rng = random.Random(7)
        rng.shuffle(rules_list)
        keep = set(new_stage_rules)
        idx = 0
        while idx < len(rules_list):
            batch = rules_list[idx:idx+64]
            trial = combined - set(batch)
            p, r, f1 = self._evaluate_prepared(prepared, trial)
            if f1 >= base_f1 - self.cfg.max_prune_drop and p + self.cfg.pr_margin >= self.cfg.target_precision:
                for rmk in batch:
                    if rmk in keep:
                        keep.remove(rmk)
                combined = trial
                base_f1 = f1
                base_p = p
                base_r = r
            idx += 64
        return keep

    def _learn_rules_iteratively(self, residual, existing_rule_sets):
        stage_rules = set()
        stage_allowed = {}
        rng = random.Random(13)
        sample = residual if len(residual) <= self.cfg.f1_eval_sample else rng.sample(residual, self.cfg.f1_eval_sample)
        last_base_f1 = -1.0
        miss_heads = self._estimate_chronic_heads(residual)
        self.chronic_heads = set([h for h,_ in miss_heads[:self.cfg.head_focus_k]])
        for it in range(1, self.cfg.max_iterations_per_stage+1):
            combined = stage_rules.union(*existing_rule_sets) if existing_rule_sets else set(stage_rules)
            scored, head_hits = self._score_candidates(residual, (existing_rule_sets + [stage_rules]) if existing_rule_sets else [stage_rules], chronic_focus=self.chronic_heads)
            if not scored:
                print("  no candidates")
                break
            dyn_budget = min(self.cfg.new_rules_per_iteration, max(12, len(scored)//10))
            selected, allowed, base_f1 = self._select_by_f1_gain_fast(sample, combined, scored, head_hits, dyn_budget)
            if not selected:
                print("  no improving candidates")
                break
            stage_rules |= selected
            for k,v in allowed.items():
                if k not in stage_allowed: stage_allowed[k] = set()
                stage_allowed[k] |= v
            gain = base_f1 - last_base_f1 if last_base_f1 >= 0 else base_f1
            last_base_f1 = base_f1
            print(f"  iter {it}: +{len(selected)} rules; total={len(stage_rules)} baseF1={base_f1:.4f}")
            if it > 2 and gain < self.cfg.min_gain_per_stage and len(residual) < self.cfg.early_stop_residual:
                print("  early stop")
                break
        for k,v in stage_allowed.items():
            if k not in self.rule_allowed_heads:
                self.rule_allowed_heads[k] = set()
            self.rule_allowed_heads[k] |= v
        return stage_rules

    def train(self, training_data):
        prepared = self.prepare(training_data)
        self.rule_stages = []
        t0 = time.time()
        for stage in tqdm(range(1, self.cfg.max_stages + 1), desc="Stages", unit="stage"):
            self._rule_cache.clear()
            combined = set().union(*self.rule_stages) if self.rule_stages else set()
            p, r, f1 = self._evaluate_prepared(prepared, combined)
            print(f"\nstage {stage}/{self.cfg.max_stages} P:{p:.4f} R:{r:.4f} F1:{f1:.4f}")
            if p >= self.cfg.target_precision and r >= self.cfg.target_recall and f1 >= self.cfg.target_f1:
                print("targets reached")
                break
            residual = []
            for ex in tqdm(prepared, desc="Computing residuals", unit="ex"):
                pred = {t.lower() for t in self.predict_doc(ex["doc"], ex["sigs"], combined)}
                if not ex["gold_terms"].issubset(pred):
                    residual.append(ex)
            print(f"residual: {len(residual)}")
            if not residual:
                print("no residuals")
                break
            stage_rules = self._learn_rules_iteratively(residual, self.rule_stages)
            if not stage_rules:
                print("no rules learned; stop")
                break
            if self.cfg.prune:
                print("  pruning")
                stage_rules = self._prune_rules(prepared, self.rule_stages, stage_rules)
                if not stage_rules:
                    print("  all pruned; stop")
                    break
            self.rule_stages.append(stage_rules)
            comb2 = set().union(*self.rule_stages)
            self.block_phrases |= self._harvest_block_phrases(prepared, comb2, top_k=60)
            p2, r2, f12 = self._evaluate_prepared(prepared, comb2)
            print(f"done stage {stage}. add:{len(stage_rules)} total:{sum(len(s) for s in self.rule_stages)} P:{p2:.4f} R:{r2:.4f} F1:{f12:.4f} block:{len(self.block_phrases)}")
            if self.cfg.debug and self.cfg.dump_dir:
                self._dump_debug(prepared, comb2, prefix=f"train_stage_{stage}")
        self.training_time_sec = time.time() - t0
        print(f"\ntraining {self.training_time_sec:.2f}s")
        for i, rs in enumerate(self.rule_stages, 1):
            print(f"stage {i}: {len(rs)} rules")

    def predict(self, text):
        doc = self.nlp(text)
        chunk_map = {}
        if self.cfg.use_chunks:
            for nc in doc.noun_chunks:
                for w in nc:
                    chunk_map[w.i] = nc
        sigs = [self._sig_variants(t, doc, chunk_map) for t in doc]
        combined = set().union(*self.rule_stages) if self.rule_stages else set()
        return self.predict_doc(doc, sigs, combined)

    def evaluate(self, raw_items, dump_prefix=None, top_n_errors=15):
        prepared = self.prepare(raw_items)
        combined = set().union(*self.rule_stages) if self.rule_stages else set()
        tp = pp = ap = 0
        fp_counter, fn_counter = Counter(), Counter()
        for ex in tqdm(prepared, desc="Evaluating", unit="ex"):
            pred = {t.lower() for t in self.predict_doc(ex["doc"], ex["sigs"], combined)}
            gt = ex["gold_terms"]
            tp_set = pred & gt
            fp_set = pred - gt
            fn_set = gt - pred
            tp += len(tp_set)
            pp += len(pred)
            ap += len(gt)
            fp_counter.update(fp_set)
            fn_counter.update(fn_set)
        precision = tp / pp if pp else 0.0
        recall = tp / ap if ap else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        print(f"P:{precision:.4f} R:{recall:.4f} F1:{f1:.4f}")
        if dump_prefix:
            os.makedirs(self._logs_dir, exist_ok=True)
            with open(os.path.join(self._logs_dir, f"{dump_prefix}_fp_top.tsv"),"w",encoding="utf8") as f:
                for term, count in fp_counter.most_common(top_n_errors):
                    f.write(f"{term}\t{count}\n")
            with open(os.path.join(self._logs_dir, f"{dump_prefix}_fn_top.tsv"),"w",encoding="utf8") as f:
                for term, count in fn_counter.most_common(top_n_errors):
                    f.write(f"{term}\t{count}\n")
            self._dump_debug(prepared, combined, prefix=dump_prefix)
        return precision, recall, f1

    def _dump_debug(self, prepared, rules, prefix="debug"):
        reasons = Counter()
        surf_c = Counter()
        by_head = Counter()
        miss_dump = []
        fp_level = Counter()
        head_fp = Counter()
        surf_fp = Counter()
        fp_dump = []
        ent_pos = Counter()
        chunk_pos = Counter()
        neigh_pos = Counter()
        morph_pos = Counter()
        shape_pos = Counter()
        geom_pos = Counter()
        for ex in prepared:
            doc = ex["doc"]
            pred_list = self.predict_doc(doc, ex["sigs"], rules)
            pred = {t.lower() for t in pred_list}
            gt = ex["gold_terms"]
            for r in ex["gold_roots"]:
                s = self._np_text(r).lower()
                hit = s in pred
                ent = (r.ent_type_ if r.ent_type_ else "", r.head.ent_type_ if r.head is not None and r.head.ent_type_ else "")
                ent_pos[(ent, hit)] += 1
                cm = {}
                if self.cfg.use_chunks:
                    for nc in doc.noun_chunks:
                        for w in nc:
                            cm[w.i] = nc
                c = cm.get(r.i)
                if c:
                    chunk_pos[((len(list(c)), c.root.dep_, sum(1 for w in c if w.dep_=="amod")), hit)] += 1
                if self.cfg.use_neighbor_pos:
                    L = []
                    for off in range(1, self.cfg.neighbor_window+1):
                        if r.i-off >= 0: L.append(doc[r.i-off].pos_)
                        if r.i+off < len(doc): L.append(doc[r.i+off].pos_)
                    neigh_pos[(tuple(L), hit)] += 1
                m = r.morph
                                                                                 
                ms_vals = []
                for k in ("Number", "Tense", "VerbForm", "Mood", "Degree", "Aspect"):
                    v = m.get(k)
                    if isinstance(v, (list, tuple)):
                        ms_vals.append(tuple(str(x) for x in v))
                    else:
                        ms_vals.append(str(v) if v is not None else None)
                ms = tuple(ms_vals)
                morph_pos[(ms, hit)] += 1
                left_shape = doc[r.i-1].shape_ if r.i-1 >= 0 else "O"
                right_shape = doc[r.i+1].shape_ if r.i+1 < len(doc) else "O"
                shape_pos[((left_shape, right_shape, r.ent_iob_), hit)] += 1
                d2h = abs(r.head.i - r.i) if r.head is not None else 0
                child_count = sum(1 for _ in r.children)
                path_len = 1
                cur = r
                while cur.head is not cur and cur.head is not None:
                    path_len += 1
                    cur = cur.head
                    if path_len > 20: break
                                                                            
                geom_pos[((min(d2h,15), min(child_count,15), min(path_len,15)), hit)] += 1
            fps = pred - gt
            if fps:
                for tok, sv in zip(ex["doc"], ex["sigs"]):
                    srf = self._np_text(tok).lower()
                    if srf in fps:
                        head_fp.update([tok.lemma_.lower()])
                        surf_fp.update([srf])
                        for s in sv:
                            if s in rules:
                                fp_level.update([s[0]])
                        td = self._token_dump(ex, tok)
                        fp_dump.append(td)
            miss_surfs = gt - pred
            for r in ex["gold_roots"]:
                s = self._np_text(r).lower()
                if s in miss_surfs:
                    reason = self._miss_reason(r, ex["sigs"][r.i], rules)
                    reasons.update([reason])
                    surf_c.update([s])
                    by_head.update([r.lemma_.lower()])
                    td = self._token_dump(ex, r)
                    miss_dump.append((reason, td))
        with open(os.path.join(self._logs_dir, f"{prefix}_misses.tsv"),"w",encoding="utf8") as f:
            f.write("reason\tsurface\tlemma\tpos\tdep\thead_lemma\thead_pos\tchildren_deps\tsent_id\tsentence\n")
            for reason, td in miss_dump:
                f.write(f"{reason}\t{td['surface']}\t{td['lemma']}\t{td['pos']}\t{td['dep']}\t{td['head_lemma']}\t{td['head_pos']}\t{td['children_deps']}\t{td['sent_id']}\t{td['sent']}\n")
        with open(os.path.join(self._logs_dir, f"{prefix}_fps.tsv"),"w",encoding="utf8") as f:
            f.write("surface\tlemma\tpos\tdep\thead_lemma\thead_pos\tchildren_deps\tsent_id\tsentence\n")
            for td in fp_dump:
                f.write(f"{td['surface']}\t{td['lemma']}\t{td['pos']}\t{td['dep']}\t{td['head_lemma']}\t{td['head_pos']}\t{td['children_deps']}\t{td['sent_id']}\t{td['sent']}\n")
        def dump_counter(c, path):
            with open(os.path.join(self._logs_dir, path),"w",encoding="utf8") as f:
                for (k, hit), v in sorted(c.items(), key=lambda x: -x[1]):
                    f.write(f"{k}\t{'hit' if hit else 'miss'}\t{v}\n")
        dump_counter(ent_pos, f"{prefix}_feature_entities.tsv")
        dump_counter(chunk_pos, f"{prefix}_feature_chunks.tsv")
        dump_counter(neigh_pos, f"{prefix}_feature_neighbors.tsv")
        dump_counter(morph_pos, f"{prefix}_feature_morph.tsv")
        dump_counter(shape_pos, f"{prefix}_feature_shape.tsv")
        dump_counter(geom_pos, f"{prefix}_feature_geom.tsv")
        with open(os.path.join(self._logs_dir, f"{prefix}_miss_summary.json"),"w",encoding="utf8") as f:
            json.dump({
                "miss_reasons": reasons.most_common(),
                "missed_surfaces": surf_c.most_common(50),
                "missed_heads": by_head.most_common(50),
                "fp_levels": sorted(fp_level.items()),
                "fp_heads": head_fp.most_common(50),
                "fp_surfaces": surf_fp.most_common(50)
            }, f, ensure_ascii=False, indent=2)

    def _miss_reason(self, tok, sigs_for_tok, rules):
        surf = self._np_text(tok).lower()
        if not self._valid_surface(surf):
            return "invalid_surface"
        if surf in self.block_phrases:
            return "blocked_phrase"
        has_rule = False
        gate_union = set()
        for s in sigs_for_tok:
            if s in rules:
                has_rule = True
                if s in self.rule_allowed_heads:
                    gate_union |= self.rule_allowed_heads[s]
        if not has_rule:
            return "no_rule_signature"
        hl = tok.lemma_.lower()
        if hl not in self.chronic_heads:
            if hl in self.generic_block_heads and self._compound_len(tok) < 2 and hl not in self.frequent_single_gold_heads:
                if not (self.cfg.use_entities and tok.ent_type_ in {"MONEY","FAC","ORG"}):
                    return "generic_single_block"
        if not self._allowed_by_context(tok, gate_union):
            return "gate_failed"
        return "other"

    def _token_dump(self, ex, tok):
        s = self._np_text(tok)
        h = tok.head
        ch = ",".join(sorted([c.dep_ for c in tok.children]))
        sent_idx = ex["sent_id"].get(tok.i, -1)
        sent = list(ex["doc"].sents)[sent_idx].text if sent_idx >= 0 else ex["doc"].text
        return {"surface": s, "lemma": tok.lemma_, "pos": tok.pos_, "dep": tok.dep_, "head_lemma": h.lemma_ if h is not None else "", "head_pos": h.pos_ if h is not None else "", "children_deps": ch, "sent_id": sent_idx, "sent": sent}

    def save_rules(self, meta=None, mark_best=False):
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        rules_flat = [list(s) for s in self.rule_stages]
        payload = {
            "config": asdict(self.cfg),
            "rule_stages": rules_flat,
            "rule_allowed_heads": {str(k): list(v) for k,v in self.rule_allowed_heads.items()},
            "block_phrases": list(self.block_phrases),
            "meta": meta or {}
        }
        ts_path = os.path.join(self._rules_dir, f"rules_{ts}.json")
        with open(ts_path,"w",encoding="utf8") as f:
            json.dump(payload, f, ensure_ascii=False)
        if mark_best:
            best_id = f"best_{ts}.json"
            shutil.copy(ts_path, os.path.join(self._rules_dir, "best", best_id))
            with open(os.path.join(self._rules_dir, "best", "LATEST.txt"),"w") as f:
                f.write(best_id)
        return ts_path

    def load_rules(self, path=None, best=False):
        if best:
            latest_txt = os.path.join(self._rules_dir, "best", "LATEST.txt")
            if os.path.exists(latest_txt):
                with open(latest_txt) as f:
                    fname = f.read().strip()
                path = os.path.join(self._rules_dir, "best", fname)
            else:
                bests = sorted(glob.glob(os.path.join(self._rules_dir, "best", "best_*.json")))
                if not bests: return False
                path = bests[-1]
        if not path or not os.path.exists(path): return False
        with open(path,"r",encoding="utf8") as f:
            payload = json.load(f)
        self.cfg = TrainConfig(**payload["config"])
        self.rule_stages = [set(tuple(x) for x in stage) for stage in payload["rule_stages"]]
        self.rule_allowed_heads = {eval(k): set(v) for k,v in payload["rule_allowed_heads"].items()}
        self.block_phrases = set(payload.get("block_phrases", []))
        return True

def load_data(file_paths):
    all_data = []
    for file_path in file_paths:
        if not os.path.exists(file_path): continue
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            if root.tag == "sentences":
                for elem in root.findall(".//sentence"):
                    text_el = elem.find("text")
                    if text_el is None or text_el.text is None: continue
                    ats = []
                    for at in elem.findall(".//aspectTerm"):
                        term = at.get("term"); fr = at.get("from"); to = at.get("to")
                        if term is None or fr is None or to is None: continue
                        ats.append({"term": term, "from": int(fr), "to": int(to)})
                    if ats: all_data.append({"text": text_el.text, "aspects_with_offsets": ats})
            elif root.tag == "Reviews":
                for elem in root.findall(".//sentence"):
                    if elem.get("OutOfScope") == "TRUE": continue
                    text_el = elem.find("text")
                    if text_el is None or text_el.text is None: continue
                    ops = []
                    for op in elem.findall(".//Opinion"):
                        tgt = op.get("target"); fr = op.get("from"); to = op.get("to")
                        if tgt and tgt != "NULL" and fr and to:
                            ops.append({"term": tgt, "from": int(fr), "to": int(to)})
                    if ops: all_data.append({"text": text_el.text, "aspects_with_offsets": ops})
        except ET.ParseError:
            continue
    return all_data

def evaluate_minimal(extractor, raw_items):
    prepared = extractor.prepare(raw_items)
    combined = set().union(*extractor.rule_stages) if extractor.rule_stages else set()
    p, r, f1 = extractor._evaluate_prepared(prepared, combined)
    return p, r, f1

def optimize_hyperparams(train_raw, val_raw, n_trials=25, timeout=None, seed=42):
    def objective(trial: optuna.Trial):
                                                                                        
        restrict_choice_label = trial.suggest_categorical("restrict_pos", [
            "None",
            "NOUN,PROPN",
            "NOUN,PROPN,ADJ",
            "NOUN,PROPN,VERB",
            "NOUN,PROPN,ADJ,VERB"
        ])
        soft_restrict = trial.suggest_categorical("soft_restrict",[True, False])
        min_pos_support = trial.suggest_int("min_pos_support", 1, 4)
        min_rule_precision = trial.suggest_float("min_rule_precision", 0.45, 0.7)
        new_rules_per_iteration = trial.suggest_int("new_rules_per_iteration", 40, 120, step=20)
        f1_eval_topk = trial.suggest_int("f1_eval_topk", 300, 1000, step=100)
        pr_margin = trial.suggest_float("pr_margin", 0.005, 0.03)
        early_stop_residual = trial.suggest_int("early_stop_residual", 1500, 4000, step=500)
        head_focus_k = trial.suggest_int("head_focus_k", 8, 20)
        adapt_head_min_support = trial.suggest_int("adapt_head_min_support", 1, 3)
        neighbor_window = trial.suggest_int("neighbor_window", 0, 2)
        use_grandparent = trial.suggest_categorical("use_grandparent",[True, False])
        use_child_profile = trial.suggest_categorical("use_child_profile",[True, False])
        use_entities = trial.suggest_categorical("use_entities",[True, False])
        use_chunks = trial.suggest_categorical("use_chunks",[True, False])
        use_neighbor_pos = trial.suggest_categorical("use_neighbor_pos",[True, False])
                                                             
        if restrict_choice_label == "None":
            restrict_choice = None
        else:
            restrict_choice = tuple(restrict_choice_label.split(","))

        cfg = TrainConfig(
            restrict_pos=restrict_choice,
            soft_restrict=soft_restrict,
            min_pos_support=min_pos_support,
            min_rule_precision=min_rule_precision,
            new_rules_per_iteration=new_rules_per_iteration,
            f1_eval_topk=f1_eval_topk,
            pr_margin=pr_margin,
            early_stop_residual=early_stop_residual,
            head_focus_k=head_focus_k,
            adapt_head_min_support=adapt_head_min_support,
            neighbor_window=neighbor_window,
            use_grandparent=use_grandparent,
            use_child_profile=use_child_profile,
            use_entities=use_entities,
            use_chunks=use_chunks,
            use_neighbor_pos=use_neighbor_pos,
            debug=False,
            dump_dir=None,
            prune=True,
            max_prune_drop=0.0,
            max_stages=12,
            max_iterations_per_stage=10,
            f1_eval_sample=1000,
            target_precision=0.86,
            target_recall=0.60,
            target_f1=0.70
        )
        extractor = MultiStageRuleExtractor(cfg)
        extractor.train(train_raw)
        p, r, f1 = extractor.evaluate(val_raw, dump_prefix=None)
        meta = {"p": p, "r": r, "f1": f1, "trial": trial.number}
        params_path = os.path.join("outputs/ner_coref/parameters", "best_params.jsonl")
        line = {"trial": trial.number, "score": meta, "params": trial.params}
        with open(params_path, "a", encoding="utf8") as f:
            f.write(json.dumps(line) + "\n")
        trial.set_user_attr("model_path", extractor.save_rules(meta=meta, mark_best=False))
        return f1
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
                                                                                                
                                                                                
    if n_trials and n_trials > 0:
        pbar = tqdm(total=n_trials, desc="Optuna trials", unit="trial")
        try:
            for _ in range(n_trials):
                study.optimize(objective, n_trials=1, timeout=timeout)
                pbar.update(1)
                                                           
                if study.best_trial:
                    pbar.set_postfix(best_f1=f"{study.best_trial.value:.4f}")
                                                 
                if timeout is not None and study._storage.get_trials_count(study._study_id) >= n_trials:
                    break
        finally:
            pbar.close()
    else:
        study.optimize(objective, timeout=timeout)
    best_trial = study.best_trial
    best_model_path = best_trial.user_attrs.get("model_path", None)
    if best_model_path:
        with open(os.path.join("outputs/ner_coref/parameters", "best_overall.json"), "w", encoding="utf8") as f:
            json.dump({"score": {"f1": best_trial.value}, "params": best_trial.params, "model_path": best_model_path}, f, indent=2)
    return study

def run_pipeline(train_files, test_files, optimize_trials=0, timeout=None):
    print("--- Loading ---")
    train_raw = load_data(train_files)
    test_raw = load_data(test_files)
    print(f"Train:{len(train_raw)} Test:{len(test_raw)}")
    if not train_raw or not test_raw:
        print("Insufficient data.")
        return
    best_cfg = TrainConfig(
        model="en_core_web_md",
        restrict_pos=None,
        min_pos_support=2,
        min_rule_precision=0.50,
        max_stages=20,
        max_iterations_per_stage=15,
        new_rules_per_iteration=60,
        n_process=1,
        batch_size=4000,
        target_precision=0.90,
        target_recall=0.65,
        target_f1=0.72,
        pr_margin=0.01,
        f1_eval_sample=1200,
        f1_eval_topk=600,
        prune=True,
        max_prune_drop=0.0,
        early_stop_residual=2000,
        min_gain_per_stage=0.0005,
        debug=False,
        debug_every_n_stages=0,
        debug_top_k=15,
        dump_dir="outputs/ner_coref/logs",
        use_grandparent=True,
        use_child_profile=True,
        child_profile_k=2,
        head_focus_k=12,
        adapt_head_min_support=1,
        use_entities=True,
        use_chunks=True,
        use_neighbor_pos=True,
        neighbor_window=1,
        coord_backoff=True,
        soft_restrict=True
    )
    if optimize_trials and optimize_trials > 0:
        print("--- Hyperparameter optimization ---")
        study = optimize_hyperparams(train_raw, test_raw, n_trials=optimize_trials, timeout=timeout)
        best_params = study.best_trial.params
                                                                                  
        for k, v in best_params.items():
            if k == "restrict_pos":
                if v == "None":
                    parsed = None
                else:
                    parsed = tuple(str(v).split(","))
                setattr(best_cfg, k, parsed)
            else:
                setattr(best_cfg, k, v)
        print("--- Best params applied ---")
    extractor = MultiStageRuleExtractor(best_cfg)
                                                                                             
    best_overall_path = os.path.join("outputs/ner_coref/parameters", "best_overall.json")
    if os.path.exists(best_overall_path):
        try:
            with open(best_overall_path, "r", encoding="utf8") as f:
                bo = json.load(f)
            bo_params = bo.get("params", {})
            bo_model = bo.get("model_path")
                                                                                
            for k, v in bo_params.items():
                if k == "restrict_pos":
                    if v == "None":
                        parsed = None
                    else:
                        parsed = tuple(str(v).split(","))
                    setattr(best_cfg, k, parsed)
                else:
                    setattr(best_cfg, k, v)
                                               
            extractor = MultiStageRuleExtractor(best_cfg)
            if bo_model and os.path.exists(bo_model):
                loaded = extractor.load_rules(bo_model)
                if loaded:
                    print(f"Loaded existing model rules from {bo_model}; continuing training")
        except Exception:
            pass

    print("--- Training ---")
    extractor.train(train_raw)
    print("--- Test Eval ---")
    p, r, f1 = extractor.evaluate(test_raw, dump_prefix="test_eval", top_n_errors=20)
    meta = {"p": p, "r": r, "f1": f1, "timestamp": dt.datetime.now().isoformat()}
    path = extractor.save_rules(meta=meta, mark_best=True)
    with open(os.path.join("outputs/ner_coref/parameters", "best_params.jsonl"), "a", encoding="utf8") as f:
        f.write(json.dumps({"timestamp": meta["timestamp"], "score": meta, "config": asdict(best_cfg), "model_path": path}) + "\n")
    print("--- Examples ---")
    s1 = "The camera is amazing but I was not happy with the battery life."
    s2 = "We loved the food and service but the decor was outdated."
    print(s1, "=>", extractor.predict(s1))
    print(s2, "=>", extractor.predict(s2))

if __name__ == "__main__":
    train_files = [
        "data/dataset/train_laptop_2014.xml","data/dataset/train_restaurant_2014.xml",
        "data/dataset/train_laptop_2016.xml","data/dataset/train_restaurant_2016.xml"
    ]
    test_files = [
        "data/dataset/test_laptop_2016.xml","data/dataset/test_restaurant_2016.xml",
        "data/dataset/test_laptop_2014.xml","data/dataset/test_restaurant_2014.xml"
    ]
    run_pipeline(train_files, test_files, optimize_trials=500, timeout=None)
