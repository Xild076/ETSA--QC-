import spacy
import xml.etree.ElementTree as ET
import os
import time
import random
from dataclasses import dataclass
from collections import Counter, defaultdict
from string import punctuation

@dataclass
class TrainConfig:
    model:str="en_core_web_sm"
    restrict_pos:tuple=("NOUN","PROPN")
    min_pos_support:int=2
    min_rule_precision:float=0.50
    max_rules:int|None=None
    max_stages:int=20
    max_iterations_per_stage:int=15
    new_rules_per_iteration:int=60
    n_process:int=1
    batch_size:int=4000
    target_precision:float=0.90
    target_recall:float=0.90
    target_f1:float=0.90
    pr_margin:float=0.01
    f1_eval_sample:int=1200
    f1_eval_topk:int=400
    prune:bool=True
    max_prune_drop:float=0.0
    early_stop_residual:int=2000
    min_gain_per_stage:float=0.0005
    debug:bool=False
    debug_every_n_stages:int=0
    debug_top_k:int=15
    dump_dir:str|None=None
    use_grandparent:bool=True
    use_child_profile:bool=True
    child_profile_k:int=2
    head_focus_k:int=8
    adapt_head_min_support:int=1

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

    def _sig_variants(self, token):
        if self.cfg.restrict_pos and token.pos_ not in self.cfg.restrict_pos:
            return []
        ch = tuple(sorted([c.dep for c in token.children]))
        gp = token.head.head if token.head is not None else None
        gp_lemma = gp.lemma if (gp is not None and hasattr(gp,"lemma")) else None
        gp_pos = gp.pos if (gp is not None and hasattr(gp,"pos")) else None
        prof = self._child_profile(token)
        s0 = (0, token.pos, token.dep, token.head.lemma, token.head.pos, ch)
        s1 = (1, token.pos, token.dep, token.head.lemma, token.head.pos, ())
        s2 = (2, token.pos, token.dep, token.head.lemma, None, ())
        s3 = (3, token.pos, token.dep, None, None, ())
        out = [s0,s1,s2,s3]
        if self.cfg.use_grandparent:
            s4 = (4, token.pos, token.dep, token.head.lemma, token.head.pos, (gp_lemma,gp_pos))
            out.append(s4)
        if self.cfg.use_child_profile:
            s5 = (5, token.pos, token.dep, token.head.lemma, token.head.pos, prof)
            out.append(s5)
        s6 = (6, token.pos, token.dep, None, None, ("HL", token.lemma))
        s7 = (7, None, None, None, None, ("HL", token.lemma))
        out.extend([s6, s7])
        return out

    def prepare(self, raw_items):
        print("\n--- Pre-processing ---")
        t0 = time.time()
        texts = [it["text"] for it in raw_items]
        docs = list(self.nlp.pipe(texts, n_process=self.cfg.n_process, batch_size=self.cfg.batch_size))
        prepped = []
        gold_heads = []
        single_head_freq = Counter()
        for it, doc in zip(raw_items, docs):
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
            sigs = [self._sig_variants(t) for t in doc]
            sents = list(doc.sents)
            sid = {}
            for si, s in enumerate(sents):
                for tok in s:
                    sid[tok.i] = si
            prepped.append({"doc": doc, "gold_terms": gold_terms, "gold_roots": gold_roots, "sigs": sigs, "sent_id": sid})
        self.aspect_lex_head_lemmas = set(gold_heads)
        self.frequent_single_gold_heads = {w for w,c in single_head_freq.items() if c >= 10}
        print(f"Pre-processing done in {time.time()-t0:.2f}s")
        return prepped

    def _allowed_by_context(self, tok, gate_union):
        hl = tok.lemma_.lower()
        if hl in gate_union: return True
        if hl in self.aspect_lex_head_lemmas: return True
        if hl in self.frequent_single_gold_heads: return True
        if self._compound_len(tok) >= 2 and hl not in self.generic_block_heads: return True
        return False

    def predict_doc(self, doc, sigs, rules):
        res = set()
        for tok, sv in zip(doc, sigs):
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
                    if s[0] in (6,7):
                        gate_union.add(tok.lemma_.lower())
            if not fire:
                continue
            hl = tok.lemma_.lower()
            if hl in self.generic_block_heads and self._compound_len(tok) < 2 and hl not in self.frequent_single_gold_heads:
                continue
            if not self._allowed_by_context(tok, gate_union):
                continue
            if self._valid_surface(surf):
                res.add(surf)
        return list(res)

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
        return {"precision": p, "recall": r, "f1": f1}

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
            if chronic_focus and any(h in chronic_focus for h in head_hits[sig]):
                needed = self.cfg.adapt_head_min_support
            else:
                needed = min_sup
            if pos < needed:
                continue
            tot = max(tot_counts[sig], pos)
            prec = pos / tot if tot else 0.0
            if prec >= self.cfg.min_rule_precision and sig not in combined:
                scored.append((sig, pos, tot, prec))
        scored.sort(key=lambda x: (-x[3], -x[1], x[2]))
        return scored, head_hits

    def _build_sample_index(self, sample):
        index = defaultdict(list)
        for i, ex in enumerate(sample):
            doc, sigs = ex["doc"], ex["sigs"]
            for j, (tok, sv) in enumerate(zip(doc, sigs)):
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
            if sig[0] in (6,7):
                gate_union.add(hl)
            if hl in self.generic_block_heads and self._compound_len(tok) < 2 and hl not in self.frequent_single_gold_heads:
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
        head_outcomes = defaultdict(lambda: {"proposed":0,"accepted":0,"Δtp":0,"Δfp":0,"blocked_pr":0,"blocked_r":0,"no_gain":0})
        level_outcomes = defaultdict(lambda: {"accepted":0,"Δtp":0,"Δfp":0})
        for sig, pos, tot, prec in scored[:topk]:
            atp, afp = self._sig_delta_on_sample(sig, head_hits.get(sig, set()), sample, index, pred_sets, combined)
            if atp == 0 and afp == 0:
                head_outcomes[tuple(sorted(list(head_hits.get(sig,set()))))[0] if head_hits.get(sig) else "_"]["no_gain"] += 1
                continue
            m_pp = pp + atp + afp
            m_tp = tp + atp
            m_p = m_tp / m_pp if m_pp else 0.0
            m_r = m_tp / ap if ap else 0.0
            m_f1 = 2 * m_p * m_r / (m_p + m_r) if (m_p + m_r) else 0.0
            p_ok = (m_p + self.cfg.pr_margin >= min(self.cfg.target_precision, base_p)) or (m_p >= self.cfg.target_precision)
            r_ok = (m_r + self.cfg.pr_margin >= min(self.cfg.target_recall, base_r)) or (m_r >= self.cfg.target_recall)
            head_key = tuple(sorted(list(head_hits.get(sig,set()))))[0] if head_hits.get(sig) else "_"
            if m_f1 - base_f1 > 0 and p_ok and r_ok:
                gains.append((m_f1 - base_f1, sig, pos, tot, prec, atp, afp, m_p, m_r, m_f1))
                cache[sig] = {"Δtp":atp,"Δfp":afp,"pos":pos,"tot":tot,"rule_prec":prec,"m_p":m_p,"m_r":m_r,"m_f1":m_f1}
                head_outcomes[head_key]["proposed"] += 1
                head_outcomes[head_key]["accepted"] += 1
                head_outcomes[head_key]["Δtp"] += atp
                head_outcomes[head_key]["Δfp"] += afp
                level_outcomes[sig[0]]["accepted"] += 1
                level_outcomes[sig[0]]["Δtp"] += atp
                level_outcomes[sig[0]]["Δfp"] += afp
            else:
                head_outcomes[head_key]["proposed"] += 1
                if not p_ok: head_outcomes[head_key]["blocked_pr"] += 1
                if not r_ok: head_outcomes[head_key]["blocked_r"] += 1
        gains.sort(key=lambda x: (-x[0], -x[4], -x[2]))
        out = []
        for g in gains:
            out.append(g[1])
            if len(out) >= budget:
                break
        allowed = {sig: head_hits[sig] for sig in out}
        self.last_selection_cache = cache
        self._last_head_outcomes = head_outcomes
        self._last_level_outcomes = level_outcomes
        return set(out), allowed, base_f1

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
                print("  No viable candidates.")
                break
            dyn_budget = min(self.cfg.new_rules_per_iteration, max(12, len(scored)//10))
            selected, allowed, base_f1 = self._select_by_f1_gain_fast(sample, combined, scored, head_hits, dyn_budget)
            if not selected:
                print("  No candidates improved under constraints.")
                break
            stage_rules |= selected
            for k,v in allowed.items():
                if k not in stage_allowed: stage_allowed[k] = set()
                stage_allowed[k] |= v
            gain = base_f1 - last_base_f1 if last_base_f1 >= 0 else base_f1
            last_base_f1 = base_f1
            print(f"  Iter {it}: +{len(selected)} rules; stage_total={len(stage_rules)} baseF1={base_f1:.4f}")
            if self.cfg.debug and it == 1 and self.last_selection_cache:
                self._print_selection_snapshot(scored, head_hits)
            if self.cfg.debug and self._last_head_outcomes:
                self._print_head_level_outcomes()
            if it > 2 and gain < self.cfg.min_gain_per_stage and len(residual) < self.cfg.early_stop_residual:
                print("  Early stop: diminishing returns.")
                break
        for k,v in stage_allowed.items():
            if k not in self.rule_allowed_heads:
                self.rule_allowed_heads[k] = set()
            self.rule_allowed_heads[k] |= v
        return stage_rules

    def _estimate_chronic_heads(self, residual):
        c = Counter()
        for ex in residual:
            doc = ex["doc"]
            pred = {t.lower() for t in self.predict_doc(doc, ex["sigs"], set().union(*self.rule_stages) if self.rule_stages else set())}
            gt = ex["gold_terms"]
            missed = gt - pred
            for r in ex["gold_roots"]:
                if self._np_text(r).lower() in missed:
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
        base = self._evaluate_prepared(prepared, combined)
        base_f1 = base["f1"]
        keep = set(new_stage_rules)
        removed = 0
        for r in list(new_stage_rules):
            trial = combined - {r}
            m = self._evaluate_prepared(prepared, trial)
            if m["f1"] >= base_f1 - self.cfg.max_prune_drop and m["precision"] + self.cfg.pr_margin >= self.cfg.target_precision:
                keep.remove(r)
                combined = trial
                base_f1 = m["f1"]
                removed += 1
        print(f"  Pruned {removed}/{len(new_stage_rules)}; kept {len(keep)}.")
        return keep

    def _token_dump(self, ex, tok):
        s = self._np_text(tok)
        h = tok.head
        ch = ",".join(sorted([c.dep_ for c in tok.children]))
        sent_idx = ex["sent_id"].get(tok.i, -1)
        sent = list(ex["doc"].sents)[sent_idx].text if sent_idx >= 0 else ex["doc"].text
        return {"surface": s, "lemma": tok.lemma_, "pos": tok.pos_, "dep": tok.dep_, "head_lemma": h.lemma_ if h is not None else "", "head_pos": h.pos_ if h is not None else "", "children_deps": ch, "sent_id": sent_idx, "sent": sent}

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
        if hl in self.generic_block_heads and self._compound_len(tok) < 2 and hl not in self.frequent_single_gold_heads:
            return "generic_single_block"
        if not self._allowed_by_context(tok, gate_union):
            return "gate_failed"
        return "other"

    def _head_recall_report(self, prepared, rules, top_k):
        gold_c = Counter()
        hit_c = Counter()
        for ex in prepared:
            pred = {t.lower() for t in self.predict_doc(ex["doc"], ex["sigs"], rules)}
            gt = ex["gold_terms"]
            for r in ex["gold_roots"]:
                h = r.lemma_.lower()
                gold_c[h] += 1
                if self._np_text(r).lower() in pred:
                    hit_c[h] += 1
        print("Head recall (top):")
        for h,c in gold_c.most_common(top_k):
            rec = hit_c[h]/c if c else 0.0
            print(f"{h}: {hit_c[h]}/{c} ({rec:.3f})")

    def _chronic_miss_report(self, prepared, rules, top_k):
        reasons = Counter()
        surf_c = Counter()
        by_head = Counter()
        dump = []
        for ex in prepared:
            pred = {t.lower() for t in self.predict_doc(ex["doc"], ex["sigs"], rules)}
            gt = ex["gold_terms"]
            miss_surfs = gt - pred
            for r in ex["gold_roots"]:
                s = self._np_text(r).lower()
                if s in miss_surfs:
                    reason = self._miss_reason(r, ex["sigs"][r.i], rules)
                    reasons.update([reason])
                    surf_c.update([s])
                    by_head.update([r.lemma_.lower()])
                    dump.append((reason, self._token_dump(ex, r)))
        print("Miss reasons:")
        for k,v in reasons.most_common():
            print(f"{k}: {v}")
        print("Missed surfaces (top):")
        for s,c in surf_c.most_common(top_k):
            print(f"'{s}': {c}")
        print("Missed heads (top):")
        for h,c in by_head.most_common(top_k):
            print(f"{h}: {c}")
        if self.cfg.dump_dir:
            try:
                os.makedirs(self.cfg.dump_dir, exist_ok=True)
                path = os.path.join(self.cfg.dump_dir, "misses.tsv")
                with open(path,"w",encoding="utf8") as f:
                    f.write("reason\tsurface\tlemma\tpos\tdep\thead_lemma\thead_pos\tchildren_deps\tsent_id\tsentence\n")
                    for reason, td in dump:
                        f.write(f"{reason}\t{td['surface']}\t{td['lemma']}\t{td['pos']}\t{td['dep']}\t{td['head_lemma']}\t{td['head_pos']}\t{td['children_deps']}\t{td['sent_id']}\t{td['sent']}\n")
                print(f"Miss dump: {path}")
            except Exception:
                pass

    def _fp_attribution_report(self, prepared, rules, top_k):
        sig_level = Counter()
        head_fp = Counter()
        surf_fp = Counter()
        dump = []
        for ex in prepared:
            pred_list = self.predict_doc(ex["doc"], ex["sigs"], rules)
            pred = {t.lower() for t in pred_list}
            gt = ex["gold_terms"]
            fps = pred - gt
            if not fps:
                continue
            for tok, sv in zip(ex["doc"], ex["sigs"]):
                srf = self._np_text(tok).lower()
                if srf in fps:
                    head_fp.update([tok.lemma_.lower()])
                    surf_fp.update([srf])
                    for s in sv:
                        if s in rules:
                            sig_level.update([s[0]])
                    dump.append(self._token_dump(ex, tok))
        print("FP signature-level usage:")
        for lev,c in sorted(sig_level.items()):
            print(f"level{lev}: {c}")
        print("FP surfaces (top):")
        for s,c in surf_fp.most_common(top_k):
            print(f"'{s}': {c}")
        print("FP heads (top):")
        for h,c in head_fp.most_common(top_k):
            print(f"{h}: {c}")
        if self.cfg.dump_dir:
            try:
                os.makedirs(self.cfg.dump_dir, exist_ok=True)
                path = os.path.join(self.cfg.dump_dir, "fps.tsv")
                with open(path,"w",encoding="utf8") as f:
                    f.write("surface\tlemma\tpos\tdep\thead_lemma\thead_pos\tchildren_deps\tsent_id\tsentence\n")
                    for td in dump:
                        f.write(f"{td['surface']}\t{td['lemma']}\t{td['pos']}\t{td['dep']}\t{td['head_lemma']}\t{td['head_pos']}\t{td['children_deps']}\t{td['sent_id']}\t{td['sent']}\n")
                print(f"FP dump: {path}")
            except Exception:
                pass

    def _selection_rule_report(self):
        if not self.last_selection_cache:
            print("No selection cache.")
            return
        items = []
        for sig, meta in self.last_selection_cache.items():
            lev = sig[0]
            items.append((meta["m_f1"], meta["m_p"], meta["m_r"], meta["Δtp"], meta["Δfp"], meta["rule_prec"], meta["pos"], meta["tot"], lev, sig))
        items.sort(key=lambda x: (-x[0], -x[5], -x[3]))
        print("Top selected candidate stats:")
        for m_f1, m_p, m_r, dtp, dfp, rprec, pos, tot, lev, sig in items[:self.cfg.debug_top_k]:
            print(f"lev{lev} ΔTP:{dtp} ΔFP:{dfp} ruleP:{rprec:.2f} pos:{pos} tot:{tot} -> P:{m_p:.3f} R:{m_r:.3f} F1:{m_f1:.3f} sig:{sig}")

    def _print_selection_snapshot(self, scored, head_hits):
        print("Selection snapshot (first iteration):")
        shown = 0
        for sig, pos, tot, prec in scored[:self.cfg.debug_top_k]:
            lev = sig[0]
            hh = ",".join(sorted(list(head_hits.get(sig,set())))[:6])
            meta = self.last_selection_cache.get(sig, {})
            d = f"lev{lev} pos:{pos} tot:{tot} ruleP:{prec:.2f} heads:[{hh}]"
            if meta:
                d += f" ΔTP:{meta.get('Δtp',0)} ΔFP:{meta.get('Δfp',0)} -> P:{meta.get('m_p',0):.3f} R:{meta.get('m_r',0):.3f} F1:{meta.get('m_f1',0):.3f}"
            print(d)
            shown += 1
            if shown >= self.cfg.debug_top_k:
                break

    def _print_head_level_outcomes(self):
        heads = sorted(self._last_head_outcomes.items(), key=lambda kv: -(kv[1]["Δtp"]))
        print("Head outcomes:")
        for h, m in heads[:self.cfg.debug_top_k]:
            print(f"{h} proposed:{m['proposed']} accepted:{m['accepted']} ΔTP:{m['Δtp']} ΔFP:{m['Δfp']} blockedPR:{m['blocked_pr']} blockedR:{m['blocked_r']}")
        print("Level outcomes:")
        for lev, m in sorted(self._last_level_outcomes.items()):
            print(f"lev{lev} accepted:{m['accepted']} ΔTP:{m['Δtp']} ΔFP:{m['Δfp']}")

    def _maybe_debug(self, stage_idx, prepared, rules):
        if not self.cfg.debug:
            return
        if self.cfg.debug_every_n_stages and (stage_idx % self.cfg.debug_every_n_stages != 1):
            return
        print("\n--- DEBUG REPORT ---")
        self._selection_rule_report()
        self._head_recall_report(prepared, rules, self.cfg.debug_top_k)
        self._chronic_miss_report(prepared, rules, self.cfg.debug_top_k)
        self._fp_attribution_report(prepared, rules, self.cfg.debug_top_k)
        print("--- END DEBUG ---\n")

    def train(self, training_data):
        prepared = self.prepare(training_data)
        self.rule_stages = []
        t0 = time.time()
        for stage in range(1, self.cfg.max_stages + 1):
            combined = set().union(*self.rule_stages) if self.rule_stages else set()
            m = self._evaluate_prepared(prepared, combined)
            print(f"\n--- Stage {stage}/{self.cfg.max_stages} --- P:{m['precision']:.4f} R:{m['recall']:.4f} F1:{m['f1']:.4f}")
            if m["precision"] >= self.cfg.target_precision and m["recall"] >= self.cfg.target_recall and m["f1"] >= self.cfg.target_f1:
                print("Targets reached; stopping.")
                break
            residual = []
            for ex in prepared:
                pred = {t.lower() for t in self.predict_doc(ex["doc"], ex["sigs"], combined)}
                if not ex["gold_terms"].issubset(pred):
                    residual.append(ex)
            print(f"Residual: {len(residual)}")
            if not residual:
                print("No residuals; stopping.")
                break
            stage_rules = self._learn_rules_iteratively(residual, self.rule_stages)
            if not stage_rules:
                print("No rules learned; stopping.")
                break
            if self.cfg.prune:
                print("  Pruning...")
                stage_rules = self._prune_rules(prepared, self.rule_stages, stage_rules)
                if not stage_rules:
                    print("  All pruned; stopping.")
                    break
            self.rule_stages.append(stage_rules)
            comb2 = set().union(*self.rule_stages)
            self.block_phrases |= self._harvest_block_phrases(prepared, comb2, top_k=60)
            post = self._evaluate_prepared(prepared, comb2)
            print(f"Completed Stage {stage}. New:{len(stage_rules)} TotalRules:{sum(len(s) for s in self.rule_stages)} | P:{post['precision']:.4f} R:{post['recall']:.4f} F1:{post['f1']:.4f} | Blocklist:{len(self.block_phrases)}")
            self._maybe_debug(stage, prepared, comb2)
        self.training_time_sec = time.time() - t0
        print(f"\n--- Training Done in {self.training_time_sec:.2f}s ---")
        for i, rs in enumerate(self.rule_stages, 1):
            print(f"Stage {i}: {len(rs)} rules")

    def predict(self, text):
        doc = self.nlp(text)
        sigs = [self._sig_variants(t) for t in doc]
        combined = set().union(*self.rule_stages) if self.rule_stages else set()
        return self.predict_doc(doc, sigs, combined)

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

def evaluate_and_show_failures_prepared(extractor, raw_items, top_n_errors=10, debug=False, dump_dir=None):
    print(f"\n--- Evaluating on {len(raw_items)} sentences ---")
    prepared = extractor.prepare(raw_items)
    combined = set().union(*extractor.rule_stages) if extractor.rule_stages else set()
    tp = pp = ap = 0
    fp_counter, fn_counter = Counter(), Counter()
    for ex in prepared:
        pred = {t.lower() for t in extractor.predict_doc(ex["doc"], ex["sigs"], combined)}
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
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print("\n--- Test Results ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1_score:.4f}")
    print("--------------------")
    print(f"\nTop {top_n_errors} False Positives")
    for term, count in fp_counter.most_common(top_n_errors): print(f"'{term}': {count}")
    print(f"\nTop {top_n_errors} False Negatives")
    for term, count in fn_counter.most_common(top_n_errors): print(f"'{term}': {count}")
    if debug:
        print("\n--- DEBUG (TEST) ---")
        extractor._head_recall_report(prepared, combined, top_n_errors)
        extractor._chronic_miss_report(prepared, combined, top_n_errors)
        extractor._fp_attribution_report(prepared, combined, top_n_errors)
        print("--- END DEBUG ---")
    if dump_dir:
        try:
            os.makedirs(dump_dir, exist_ok=True)
            with open(os.path.join(dump_dir,"fn_top.tsv"),"w",encoding="utf8") as f:
                for term, count in fn_counter.most_common(top_n_errors):
                    f.write(f"{term}\t{count}\n")
            with open(os.path.join(dump_dir,"fp_top.tsv"),"w",encoding="utf8") as f:
                for term, count in fp_counter.most_common(top_n_errors):
                    f.write(f"{term}\t{count}\n")
            print(f"Aggregate dumps in {dump_dir}")
        except Exception:
            pass

if __name__ == "__main__":
    train_files = [
        "data/dataset/train_laptop_2014.xml","data/dataset/train_restaurant_2014.xml",
        "data/dataset/train_laptop_2016.xml","data/dataset/train_restaurant_2016.xml"
    ]
    test_files = [
        "data/dataset/test_laptop_2016.xml","data/dataset/test_restaurant_2016.xml",
        "data/dataset/test_laptop_2014.xml","data/dataset/test_restaurant_2014.xml"
    ]
    print("--- Loading ---")
    train_raw = load_data(train_files)
    test_raw  = load_data(test_files)
    print(f"Train:{len(train_raw)} Test:{len(test_raw)}")
    if not train_raw or not test_raw:
        print("Insufficient data.")
    else:
        cfg = TrainConfig(
            model="en_core_web_sm",
            restrict_pos=("NOUN","PROPN"),
            min_pos_support=2,
            min_rule_precision=0.50,
            max_rules=None,
            max_stages=20,
            max_iterations_per_stage=15,
            new_rules_per_iteration=60,
            n_process=1,
            batch_size=4000,
            target_precision=0.90,
            target_recall=0.90,
            target_f1=0.90,
            pr_margin=0.01,
            f1_eval_sample=1200,
            f1_eval_topk=400,
            prune=True,
            max_prune_drop=0.0,
            early_stop_residual=2000,
            min_gain_per_stage=0.0005,
            debug=True,
            debug_every_n_stages=3,
            debug_top_k=15,
            dump_dir="debug_dumps",
            use_grandparent=True,
            use_child_profile=True,
            child_profile_k=2,
            head_focus_k=10,
            adapt_head_min_support=1
        )
        extractor = MultiStageRuleExtractor(cfg)
        extractor.train(train_raw)
        evaluate_and_show_failures_prepared(extractor, test_raw, top_n_errors=15, debug=True, dump_dir="debug_dumps_eval")
        s1 = "The camera is amazing but I was not happy with the battery life."
        s2 = "We loved the food and service but the decor was outdated."
        print("\n--- Examples ---")
        print(s1, "=>", extractor.predict(s1))
        print(s2, "=>", extractor.predict(s2))
