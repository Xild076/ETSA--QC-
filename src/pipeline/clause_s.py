import re
import logging
from typing import List, Optional, Iterable
import nltk

from utility import normalize_text

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.WARNING)

try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        pass

try:
    import spacy
except Exception:
    spacy = None

try:
    import benepar
except Exception:
    benepar = None

from nltk.tree import ParentedTree


class ClauseSplitter:
    def split(self, text: str) -> List[str]:
        raise NotImplementedError


def _score_clause(s: str) -> float:
    s = s or ""
    if not s.strip():
        return 0.0
    score = 0.0
    if len(s) >= 1:
        score += 0.2
    if len(s.split()) >= 3:
        score += 0.3
    if any(c in s for c in ".!?;:"):
        score += 0.2
    if s[:1].isupper():
        score += 0.2
    if len(s) > 60:
        score += 0.1
    return min(score, 1.0)


def _post_filter(clauses: Iterable[str], min_score: float = 0.5) -> List[str]:
    out = []
    for c in clauses:
        # Use the original non-normalized text for scoring and output
        t_normalized_for_check = " ".join(c.split())
        if t_normalized_for_check and _score_clause(t_normalized_for_check) >= min_score:
            out.append(t_normalized_for_check)
    seen = set()
    dedup = []
    for c in out:
        # Use a normalized key for deduplication
        k = normalize_text(c)
        if k not in seen:
            seen.add(k)
            dedup.append(c)
    return dedup


class NLTKSentenceSplitter(ClauseSplitter):
    def __init__(self):
        self._tokenize = nltk.sent_tokenize

    def _secondary_splits(self, s: str) -> List[str]:
        parts = []
        queue = [s]
        tmp = []
        for piece in queue:
            tmp.extend(re.split(r"(?<=[!?;:])\s+", piece))
        parts = []
        for piece in tmp:
            parts.extend(re.split(r",\s+(?=[A-Z])", piece))
        return [p for p in parts if " ".join(p.split())]

    def split(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        sents = self._tokenize(text)
        chunks = []
        for s in sents:
            chunks.append(s)
            for sub in self._secondary_splits(s):
                chunks.append(sub)
        return _post_filter(chunks, min_score=0.5)


from typing import List

try:
    import spacy
except Exception:
    spacy = None

try:
    import benepar
except Exception:
    benepar = None


def _benepar_normalize_text(s: str) -> str:
    s = " ".join(s.split())
    s = s.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?").replace(" ;", ";").replace(" :", ":")
    s = s.replace(" ’", "’").replace(" ’ s", "’s").replace(" n't", "n't")
    return s.strip()

def _join_tokens(tokens) -> str:
    if not tokens:
        return ""
    toks = sorted(tokens, key=lambda t: t.i)
    out = []
    for t in toks:
        out.append(t.text)
        out.append(t.whitespace_)
    return _benepar_normalize_text("".join(out))

def _is_verblike(t) -> bool:
    return t.pos_ in {"VERB", "AUX"} or t.tag_.startswith("VB")

class BeneparClauseSplitter:
    def __init__(self, spacy_model: str = "en_core_web_sm", benepar_model: str = "benepar_en3"):
        self.has_spacy = spacy is not None
        self.has_benepar = False
        self._nlp = None
        if self.has_spacy:
            try:
                self._nlp = spacy.load(spacy_model)
            except Exception:
                self._nlp = None
        if self._nlp is not None and benepar is not None:
            try:
                if hasattr(spacy, "__version__") and spacy.__version__.startswith("2"):
                    self._nlp.add_pipe(benepar.BeneparComponent(benepar_model))
                else:
                    self._nlp.add_pipe("benepar", config={"model": benepar_model})
                self.has_benepar = True
            except Exception:
                self.has_benepar = False

    def _segment_on_semicolons(self, sent):
        segs = []
        cur = []
        for tok in sent:
            if tok.text == ";":
                if cur:
                    segs.append(cur)
                    cur = []
            else:
                cur.append(tok)
        if cur:
            segs.append(cur)
        if not segs:
            segs = [[t for t in sent]]
        return segs

    def _heads_in_segment(self, seg_tokens):
        heads = []
        for t in seg_tokens:
            if t.dep_ == "ROOT" and (_is_verblike(t) or t.pos_ in {"NOUN", "PROPN", "ADJ"}):
                heads.append(t)
        for t in seg_tokens:
            if t.dep_ == "conj" and _is_verblike(t):
                h = t
                while h.head.dep_ == "conj":
                    h = h.head
                if h in seg_tokens:
                    heads.append(t)
        if not heads:
            for t in seg_tokens:
                if t.dep_ == "ROOT":
                    heads.append(t)
                    break
        uniq = []
        seen = set()
        for h in sorted(heads, key=lambda x: x.i):
            if h.i not in seen:
                uniq.append(h)
                seen.add(h.i)
        return uniq

    def _assign_tokens(self, seg_tokens, heads):
        if not heads:
            seg_list = list(seg_tokens)
            if not seg_list:
                return {}
            heads = [seg_list[0]]
        head_set = set(heads)
        groups = {h: set() for h in heads}
        idx_heads = [h.i for h in heads]
        for t in seg_tokens:
            anc = t
            assigned = None
            visited = set()
            while anc is not None and anc not in visited:
                visited.add(anc)
                if anc in head_set:
                    assigned = anc
                    break
                if anc.head is anc:
                    break
                anc = anc.head
            if assigned is None:
                dists = [abs(t.i - hi) for hi in idx_heads]
                k = dists.index(min(dists))
                assigned = heads[k]
            groups[assigned].add(t)
        return groups

    def _groups_to_strings(self, groups):
        parts = []
        for h in sorted(groups.keys(), key=lambda x: x.i):
            txt = _join_tokens(groups[h])
            if txt:
                parts.append(txt)
        return parts

    def _dep_split(self, text: str) -> List[str]:
        if self._nlp is None:
            return []
        doc = self._nlp(text)
        out = []
        for sent in doc.sents:
            segs = self._segment_on_semicolons(sent)
            for seg in segs:
                heads = self._heads_in_segment(seg)
                groups = self._assign_tokens(seg, heads)
                parts = self._groups_to_strings(groups)
                out.extend(parts)
        return out

    def _benepar_split(self, text: str) -> List[str]:
        if self._nlp is None:
            return []
        return self._dep_split(text)

    def split(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        if self._nlp is None:
            try:
                return NLTKSentenceSplitter().split(text)
            except Exception:
                return [x.strip() for x in text.replace("\n", " ").split(".") if x.strip()]
        res = self._benepar_split(text) if self.has_benepar else self._dep_split(text)
        try:
            return _post_filter(res, min_score=0.5)
        except Exception:
            return res

def make_clause_splitter(mode: str = "advanced") -> ClauseSplitter:
    m = (mode or "").lower().strip()
    if m in {"nltk", "simple", "fast"}:
        return NLTKSentenceSplitter()
    return BeneparClauseSplitter()