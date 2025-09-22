import spacy
import xml.etree.ElementTree as ET
import os
import time
import random
import logging
from dataclasses import dataclass
from collections import Counter, defaultdict
from string import punctuation

@dataclass
class TrainConfig:
    model:str="en_core_web_md"
    restrict_pos:tuple=None # ("NOUN","PROPN")
    min_pos_support:int=2
    min_rule_precision:float=0.48
    max_stages:int=20
    max_iterations_per_stage:int=15
    new_rules_per_iteration:int=120
    n_process:int=1
    batch_size:int=4000
    target_precision:float=0.98
    target_recall:float=0.98
    target_f1:float=0.98
    pr_margin:float=0.012
    f1_eval_sample:int=1200
    f1_eval_topk:int=800
    prune:bool=True
    prune_strategy:str="keep_topk"
    stage_keep_topk:int=250
    min_stage_keep:int=40
    no_prune_if_stage_gain_ge:float=0.01
    max_prune_drop:float=0.0
    early_stop_residual:int=3000
    min_gain_per_stage:float=0.0005
    debug:bool=False
    debug_every_n_stages:int=3
    debug_top_k:int=15
    dump_dir:str|None=None
    use_grandparent:bool=False
    use_child_profile:bool=True
    child_profile_k:int=2
    head_focus_k:int=9
    adapt_head_min_support:int=3
    use_entities:bool=True
    use_chunks:bool=False
    use_neighbor_pos:bool=True
    neighbor_window:int=0
    coord_backoff:bool=True
    soft_restrict:bool=True
    trace_level:str="INFO"
    max_block_phrases:int=20
    block_singletons:bool=False
    protect_heads_min_freq:int=8

class MultiStageRuleExtractor:
    def __init__(self, cfg:TrainConfig):
        try:
            self.nlp = spacy.load(cfg.model, disable=[])
        except OSError:
            spacy.cli.download(cfg.model)
            self.nlp = spacy.load(cfg.model, disable=[])
        self.cfg = cfg
        self.rule_stages = []
        self.rule_allowed_heads = {}
        self.stopset = set(spacy.lang.en.stop_words.STOP_WORDS) | set(punctuation)
        self.aspect_lex_head_lemmas = set()
        self.frequent_single_gold_heads = set()
        self.generic_block_heads = {"place","restaurant","time","thing","table","dinner","drinks","price","prices","laptop","computer","bar","dining","product"}
        self.block_phrases = set()
        self.last_selection_cache = {}
        self.chronic_heads = set()
        self.gold_head_freq = Counter()
        logging.basicConfig(level=getattr(logging, cfg.trace_level.upper(), logging.INFO), format="%(message)s")
        self.log = logging.getLogger("RuleExtractor")

    def _valid_surface(self, s):
        if not s: return False
        t = s.lower().strip()
        if len(t) < 2: return False
        if t in self.stopset: return False
        return True

    def _np_text(self, token):
        left_keep = {"compound","amod","nummod","flat"}
        right_keep = {"compound","amod","nummod","flat","appos"}
        left = [t for t in token.lefts if t.dep_ in left_keep]
        right = [t for t in token.rights if t.dep_ in right_keep]
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
        if not self.cfg.use_child_profile: return ()
        deps = sorted([c.dep_ for c in token.children])
        if not deps: return ()
        if self.cfg.child_profile_k <= 0: return tuple(deps)
        step = max(1, len(deps)//self.cfg.child_profile_k)
        return tuple(deps[::step][:self.cfg.child_profile_k])

    def _neighbors(self, doc, i):
        if not self.cfg.use_neighbor_pos: return ()
        L = [doc[i-off].pos_ for off in range(1, self.cfg.neighbor_window+1) if i-off >= 0]
        L += [doc[i+off].pos_ for off in range(1, self.cfg.neighbor_window+1) if i+off < len(doc)]
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
            prof = self._child_profile(token)
            out.append((5, token.pos_, token.dep_, token.head.lemma_, token.head.pos_, prof))
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
            is_root, clen, crootdep, _ = self._chunk_info(token, chunk_map)
            if clen > 1: out.append((10, is_root, clen, crootdep))
        if self.cfg.use_neighbor_pos:
            nctx = self._neighbors(doc, token.i)
            if nctx: out.append((11, token.pos_, token.dep_, nctx))
        out.append((12, token.pos_, token.dep_, token.shape_, token.like_num))
        if token.dep_ == 'appos': out.append((13, 'appos', token.head.pos_))
        out.append((14, 'HL_ONLY', token.lemma_, token.pos_))
        return out

    def prepare(self, raw_items):
        self.log.info("\n--- Pre-processing ---")
        t0 = time.time()
        texts = [it["text"] for it in raw_items]
        docs = list(self.nlp.pipe(texts, n_process=self.cfg.n_process, batch_size=self.cfg.batch_size))
        prepped = []
        gold_heads = []
        single_head_freq = Counter()
        for it, doc in zip(raw_items, docs):
            chunk_map = {}
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
            sigs = [self._sig_variants(t, doc, chunk_map) for t in doc]
            sid = {tok.i: si for si, s in enumerate(doc.sents) for tok in s}
            prepped.append({"doc": doc, "gold_terms": gold_terms, "gold_roots": gold_roots, "sigs": sigs, "sent_id": sid})
        self.aspect_lex_head_lemmas = set(gold_heads)
        self.gold_head_freq = Counter(gold_heads)
        self.frequent_single_gold_heads = {w for w,c in single_head_freq.items() if c >= 10}
        self.log.info(f"Pre-processing done in {time.time()-t0:.2f}s")
        return prepped

    def _allowed_by_context(self, tok, gate_union):
        hl = tok.lemma_.lower()
        if hl in gate_union: return True
        if self.cfg.soft_restrict and hl in self.aspect_lex_head_lemmas: return True
        if hl in self.frequent_single_gold_heads: return True
        if self._compound_len(tok) >= 2: return True
        if self.cfg.use_entities and tok.ent_type_ in {"PRODUCT", "ORG", "WORK_OF_ART", "MONEY", "FAC"}: return True
        return True

    def predict_doc(self, doc, sigs, rules):
        res = set()
        for tok, sv in zip(doc, sigs):
            surf = self._np_text(tok)
            if surf.lower() in self.block_phrases: continue
            fire, gate_union = False, set()
            for s in sv:
                if s in rules:
                    fire = True
                    gate_union.update(self.rule_allowed_heads.get(s, set()))
                    if s[0] in (6,7,14): gate_union.add(tok.lemma_.lower())
            if not fire: continue
            if not self._allowed_by_context(tok, gate_union): continue
            if self._valid_surface(surf): res.add(surf)
        return list(res)

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
            gate_union = set(allowed_heads) | self.rule_allowed_heads.get(sig, set())
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
        stage_rules, stage_allowed = set(), {}
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
            dyn_budget = min(self.cfg.new_rules_per_iteration, max(12, len(scored)//10))
            combined = set().union(*current_rules)
            selected, allowed, base_f1, gains = self._select_by_f1_gain_fast(sample, combined, scored, head_hits, dyn_budget)
            self.log.info(f"  Iter {it}: evaluated={min(len(scored), self.cfg.f1_eval_topk)} selected={len(selected)} baseF1={base_f1:.4f}")
            if not selected:
                self.log.info("  No candidates improved under constraints.")
                break
            stage_rules |= selected
            for k,v in allowed.items(): self.rule_allowed_heads.setdefault(k, set()).update(v)
            gain = base_f1 - last_base_f1 if last_base_f1 >= 0 else base_f1
            last_base_f1 = base_f1
            top_show = []
            for g in gains[:10]:
                sig = g[1]
                c = self.last_selection_cache.get(sig, {})
                top_show.append((sig, c.get("Δtp",0), c.get("Δfp",0), f"{c.get('m_p',0):.3f}", f"{c.get('m_r',0):.3f}", f"{c.get('m_f1',0):.3f}"))
            if top_show:
                self.log.info("    Top selected preview: " + ", ".join([f"{ts[0]} Δ+{ts[1]}/+{ts[2]} P{ts[3]} R{ts[4]} F{ts[5]}" for ts in top_show]))
            self.log.info(f"  Iter {it}: +{len(selected)} rules; stage_total={len(stage_rules)}")
            if it > 2 and gain < self.cfg.min_gain_per_stage and len(residual) < self.cfg.early_stop_residual:
                self.log.info("  Early stop: diminishing returns.")
                break
        for k,v in stage_allowed.items(): self.rule_allowed_heads.setdefault(k, set()).update(v)
        return stage_rules

    def _estimate_chronic_heads(self, residual):
        c = Counter()
        combined = set().union(*self.rule_stages) if self.rule_stages else set()
        for ex in residual:
            pred = {t.lower() for t in self.predict_doc(ex["doc"], ex["sigs"], combined)}
            for r in ex["gold_roots"]:
                if self._np_text(r).lower() not in pred: c.update([r.lemma_.lower()])
        return c.most_common()

    def _harvest_block_phrases(self, prepared, rules, top_k=60):
        fp_counter = Counter()
        for ex in prepared:
            pred = self.predict_doc(ex["doc"], ex["sigs"], rules)
            fps = {p.lower() for p in pred if p.lower() not in ex["gold_terms"]}
            fp_counter.update(fps)
        candidates = []
        for surf, c in fp_counter.most_common(top_k):
            toks = surf.split()
            if len(toks) == 1 and not self.cfg.block_singletons: continue
            if toks and toks[-1] in self.generic_block_heads and self.gold_head_freq[toks[-1]] < self.cfg.protect_heads_min_freq:
                candidates.append(surf)
            if len(candidates) >= self.cfg.max_block_phrases: break
        return set(candidates)

    def _score_rules_on_slice(self, rules, sample, combined):
        rng = random.Random(23)
        val = sample if len(sample) <= 800 else rng.sample(sample, 800)
        index = self._build_sample_index(val)
        pred_sets, tp, pp, ap = self._baseline_sets(val, combined)
        per_rule = {}
        for r in rules:
            atp, afp = self._sig_delta_on_sample(r, self.rule_allowed_heads.get(r,set()), val, index, pred_sets)
            per_rule[r] = (atp, afp)
        return per_rule, (tp, pp, ap)

    def _auto_lambda(self, tp, pp, ap, target_p):
        p = tp/pp if pp else 1.0
        if p >= target_p: return 1.0
        return max(1.0, (target_p - p) * 50.0)

    def _prune_rules(self, prepared, existing_rule_sets, new_stage_rules, selection_sample):
        if not new_stage_rules: return new_stage_rules
        before = self._evaluate_prepared(prepared, set().union(*existing_rule_sets))
        after_all = self._evaluate_prepared(prepared, set().union(*existing_rule_sets, new_stage_rules))
        self.log.info(f"  Stage gain before prune: ΔP={after_all['precision']-before['precision']:.4f} ΔR={after_all['recall']-before['recall']:.4f} ΔF1={after_all['f1']-before['f1']:.4f}")
        if self.cfg.prune_strategy!="keep_topk":
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
        t0 = time.time()
        rng = random.Random(99)
        for stage in range(1, self.cfg.max_stages + 1):
            combined = set().union(*self.rule_stages) if self.rule_stages else set()
            m = self._evaluate_prepared(prepared, combined)
            self.log.info(f"\n--- Stage {stage}/{self.cfg.max_stages} --- P:{m['precision']:.4f} R:{m['recall']:.4f} F1:{m['f1']:.4f} | TP:{m['tp']} PP:{m['pp']} AP:{m['ap']}")
            if m["f1"] >= self.cfg.target_f1:
                self.log.info("Targets reached; stopping.")
                break
            residual = [ex for ex in prepared if not ex["gold_terms"].issubset({t.lower() for t in self.predict_doc(ex["doc"], ex["sigs"], combined)})]
            self.log.info(f"Residual: {len(residual)}")
            if not residual:
                self.log.info("No residuals; stopping.")
                break
            heads = self._estimate_chronic_heads(residual)
            if heads:
                self.log.info("Top missed heads: " + ", ".join([f"{h}:{c}" for h,c in heads[:15]]))
            stage_rules = self._learn_rules_iteratively(residual, self.rule_stages)
            if not stage_rules:
                self.log.info("No rules learned; stopping.")
                break
            if self.cfg.prune:
                self.log.info("  Pruning...")
                selection_sample = residual if len(residual) <= self.cfg.f1_eval_sample else rng.sample(residual, self.cfg.f1_eval_sample)
                kept = self._prune_rules(prepared, self.rule_stages, stage_rules, selection_sample)
                if not kept:
                    self.log.info("  Prune returned empty; keeping minimal nucleus from selection cache.")
                    recov = sorted(list(self.last_selection_cache.items()), key=lambda kv: (-(kv[1].get("Δtp",0) - kv[1].get("Δfp",0)), -kv[1].get("Δtp",0)))[:self.cfg.min_stage_keep]
                    kept = set([kv[0] for kv in recov]) if recov else set()
                    if not kept:
                        self.log.info("  Recovery failed; stopping.")
                        break
                stage_rules = kept
            self.rule_stages.append(stage_rules)
            comb2 = set().union(*self.rule_stages)
            new_blocks = self._harvest_block_phrases(prepared, comb2, top_k=60)
            self.block_phrases.update(new_blocks)
            post = self._evaluate_prepared(prepared, comb2)
            self.log.info(f"Completed Stage {stage}. New:{len(stage_rules)} TotalRules:{len(comb2)} | P:{post['precision']:.4f} R:{post['recall']:.4f} F1:{post['f1']:.4f} | Blocklist:{len(self.block_phrases)}")
        self.training_time_sec = time.time() - t0
        self.log.info(f"\n--- Training Done in {self.training_time_sec:.2f}s ---")
        for i, rs in enumerate(self.rule_stages, 1): self.log.info(f"Stage {i}: {len(rs)} rules")

    def predict(self, text):
        doc = self.nlp(text)
        chunk_map = {}
        sigs = [self._sig_variants(t, doc, chunk_map) for t in doc]
        combined = set().union(*self.rule_stages)
        return self.predict_doc(doc, sigs, combined)

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

def evaluate_and_show_failures_prepared(extractor, raw_items, top_n_errors=10):
    print(f"\n--- Evaluating on {len(raw_items)} sentences ---")
    prepared = extractor.prepare(raw_items)
    combined = set().union(*extractor.rule_stages)
    tp, pp, ap = 0, 0, 0
    fp_counter, fn_counter = Counter(), Counter()
    fp_examples, fn_examples = defaultdict(list), defaultdict(list)
    for ex in prepared:
        pred = {t.lower() for t in extractor.predict_doc(ex["doc"], ex["sigs"], combined)}
        gt = ex["gold_terms"]
        fp_set, fn_set = pred - gt, gt - pred
        tp += len(pred & gt); pp += len(pred); ap += len(gt)
        fp_counter.update(fp_set)
        fn_counter.update(fn_set)
        for term in fp_set:
            if len(fp_examples[term]) < 2: fp_examples[term].append(ex['doc'].text)
        for term in fn_set:
            if len(fn_examples[term]) < 2: fn_examples[term].append(ex['doc'].text)
    precision = tp / pp if pp else 0.0
    recall = tp / ap if ap else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print("\n--- Test Results ---")
    print(f"Precision: {precision:.4f}\nRecall:    {recall:.4f}\nF1-Score:  {f1_score:.4f}")
    print("--------------------")
    print(f"\nTop {top_n_errors} False Positives")
    for term, count in fp_counter.most_common(top_n_errors):
        print(f"'{term}': {count}")
        for sent in fp_examples[term]: print(f"  - Ex: \"{sent[:85].strip()}...\"")
    print(f"\nTop {top_n_errors} False Negatives")
    for term, count in fn_counter.most_common(top_n_errors):
        print(f"'{term}': {count}")
        for sent in fn_examples[term]: print(f"  - Ex: \"{sent[:85].strip()}...\"")

if __name__ == "__main__":
    base_path = "."
    train_files = [os.path.join(base_path, f) for f in ["data/dataset/train_laptop_2014.xml", "data/dataset/train_restaurant_2014.xml", "data/dataset/train_laptop_2016.xml", "data/dataset/train_restaurant_2016.xml"]]
    test_files = [os.path.join(base_path, f) for f in ["data/dataset/test_laptop_2014.xml", "data/dataset/test_restaurant_2014.xml", "data/dataset/test_laptop_2016.xml", "data/dataset/test_restaurant_2016.xml"]]

    print("--- Loading ---")
    train_raw = load_data(train_files)
    test_raw  = load_data(test_files)
    print(f"Train:{len(train_raw)} Test:{len(test_raw)}")
    if not train_raw:
        print("Training data not found or empty. Please check paths.")
    else:
        cfg = TrainConfig()
        extractor = MultiStageRuleExtractor(cfg)
        extractor.train(train_raw)
        if test_raw:
            evaluate_and_show_failures_prepared(extractor, test_raw, top_n_errors=15)

        print("\n--- Examples ---")
        s1 = "The camera is amazing but I was not happy with the battery life."
        print(s1, "=>", extractor.predict(s1))
        s2 = "We loved the food and service but the decor was outdated."
        print(s2, "=>", extractor.predict(s2))
        s3 = "While the processor is fast, the screen resolution is lacking."
        print(s3, "=>", extractor.predict(s3))
