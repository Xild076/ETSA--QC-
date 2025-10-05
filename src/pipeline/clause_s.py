"""Clause splitting strategies for preparing text snippets for analysis."""

import logging
import re
from typing import Any, Dict, Iterable, List, Sequence, Set, TYPE_CHECKING

import nltk

try:
    from src.utility import normalize_text
except ImportError:
    try:
        from ..utility import normalize_text
    except ImportError:
        from utility import normalize_text

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.WARNING)

# Ensure the Punkt sentence tokenizer is available at runtime.
try:  # pragma: no cover - download only when missing
    nltk.data.find("tokenizers/punkt")
except Exception:  # pragma: no cover - best-effort download
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        pass

try:  # pragma: no cover - optional dependency
    import spacy
except Exception:  # pragma: no cover - spaCy absent in lightweight installs
    spacy = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import benepar
except Exception:  # pragma: no cover - benepar absent in lightweight installs
    benepar = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spacy.tokens import Doc, Token
else:  # pragma: no cover - fallback aliases when spaCy not installed
    Doc = Any
    Token = Any

__all__ = [
    "ClauseSplitter",
    "NLTKSentenceSplitter",
    "BeneparClauseSplitter",
    "make_clause_splitter",
]


class ClauseSplitter:
    """Interface for turning free-form text into clause-like segments."""

    def split(self, text: str) -> List[str]:
        """Return a list of clauses extracted from ``text``."""
        raise NotImplementedError


def _score_clause(sentence: str) -> float:
    """Assign a heuristic quality score in ``[0, 1]`` to a candidate clause."""
    sentence = sentence or ""
    if not sentence.strip():
        return 0.0
    score = 0.0
    if len(sentence) >= 1:
        score += 0.2
    if len(sentence.split()) >= 3:
        score += 0.3
    if any(c in sentence for c in ".!?;:"):
        score += 0.2
    if sentence[:1].isupper():
        score += 0.2
    if len(sentence) > 60:
        score += 0.1
    return min(score, 1.0)


def _post_filter(clauses: Iterable[str], min_score: float = 0.5) -> List[str]:
    """Normalise, score, and deduplicate clause candidates."""
    filtered: List[str] = []
    for clause in clauses:
        clause_norm = " ".join(clause.split())
        if clause_norm and _score_clause(clause_norm) >= min_score:
            filtered.append(clause_norm)

    seen = set()
    unique_clauses: List[str] = []
    for clause in filtered:
        key = normalize_text(clause)
        if key not in seen:
            seen.add(key)
            unique_clauses.append(clause)
    return unique_clauses


class NLTKSentenceSplitter(ClauseSplitter):
    """Quick sentence splitter leveraging NLTK's Punkt tokenizer."""

    def __init__(self) -> None:
        self._tokenize = nltk.sent_tokenize

    def _secondary_splits(self, sentence: str) -> List[str]:
        """Perform lightweight clause segmentation within a sentence."""
        queue = [sentence]
        intermediate: List[str] = []
        for piece in queue:
            intermediate.extend(re.split(r"(?<=[!?;:])\s+", piece))
        clauses: List[str] = []
        for item in intermediate:
            clauses.extend(re.split(r",\s+(?=[A-Z])", item))
        return [p for p in clauses if " ".join(p.split())]

    def split(self, text: str) -> List[str]:  # noqa: D401 - brief override
        """Split ``text`` into clauses using Punkt + punctuation heuristics."""
        if not text or not text.strip():
            return []
        sentences = self._tokenize(text)
        candidates: List[str] = []
        for sentence in sentences:
            candidates.append(sentence)
            for sub_clause in self._secondary_splits(sentence):
                candidates.append(sub_clause)
        return _post_filter(candidates, min_score=0.5)


def _benepar_normalize_text(text: str) -> str:
    """Normalise whitespace and punctuation spacing emitted by spaCy tokens."""
    text = " ".join(text.split())
    text = text.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    text = text.replace(" ;", ";").replace(" :", ":")
    text = text.replace(" ’", "’").replace(" ’ s", "’s").replace(" n't", "n't")
    return text.strip()


def _join_tokens(tokens: Sequence["Token"]) -> str:
    """Reconstruct a textual span from spaCy tokens while keeping spacing."""
    if not tokens:
        return ""
    ordered = sorted(tokens, key=lambda tok: tok.i)
    pieces: List[str] = []
    for tok in ordered:
        pieces.append(tok.text)
        pieces.append(tok.whitespace_)
    return _benepar_normalize_text("".join(pieces))


def _is_verblike(token: "Token") -> bool:
    """Return ``True`` when *token* behaves like a verbal head."""
    return token.pos_ in {"VERB", "AUX"} or token.tag_.startswith("VB")


class BeneparClauseSplitter(ClauseSplitter):
    """Dependency-aware clause splitter powered by spaCy and Benepar."""

    def __init__(self, spacy_model: str = "en_core_web_sm", benepar_model: str = "benepar_en3") -> None:
        """Initialise spaCy/Benepar pipelines when the dependencies are present."""
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

    def _segment_on_semicolons(self, sentence: Sequence["Token"]) -> List[List["Token"]]:
        """Split a sentence on semicolons while preserving token objects."""
        segments = []
        current = []
        for token in sentence:
            if token.text == ";":
                if current:
                    segments.append(current)
                    current = []
            else:
                current.append(token)
        if current:
            segments.append(current)
        if not segments:
            segments = [[token for token in sentence]]
        return segments

    def _heads_in_segment(self, segment_tokens: Sequence["Token"]) -> List["Token"]:
        """Identify candidate head tokens driving each clause segment."""
        heads = []
        for token in segment_tokens:
            if token.dep_ == "ROOT" and (_is_verblike(token) or token.pos_ in {"NOUN", "PROPN", "ADJ"}):
                heads.append(token)
        for token in segment_tokens:
            if token.dep_ == "conj" and _is_verblike(token):
                head = token
                while head.head.dep_ == "conj":
                    head = head.head
                if head in segment_tokens:
                    heads.append(token)
        if not heads:
            for token in segment_tokens:
                if token.dep_ == "ROOT":
                    heads.append(token)
                    break
        unique_heads = []
        seen = set()
        for head in sorted(heads, key=lambda tok: tok.i):
            if head.i not in seen:
                unique_heads.append(head)
                seen.add(head.i)
        return unique_heads

    def _assign_tokens(
        self,
        segment_tokens: Sequence["Token"],
        heads: Sequence["Token"],
    ) -> Dict["Token", Set["Token"]]:
        """Group tokens by their nearest governing head."""
        if not heads:
            segment_list = list(segment_tokens)
            if not segment_list:
                return {}
            heads = [segment_list[0]]
        head_set = set(heads)
        groups = {head: set() for head in heads}
        head_indices = [head.i for head in heads]
        for token in segment_tokens:
            ancestor = token
            assigned = None
            visited = set()
            while ancestor is not None and ancestor not in visited:
                visited.add(ancestor)
                if ancestor in head_set:
                    assigned = ancestor
                    break
                if ancestor.head is ancestor:
                    break
                ancestor = ancestor.head
            if assigned is None:
                distances = [abs(token.i - idx) for idx in head_indices]
                closest = distances.index(min(distances))
                assigned = heads[closest]
            groups[assigned].add(token)
        return groups

    def _groups_to_strings(self, groups: Dict["Token", Set["Token"]]) -> List[str]:
        """Convert token clusters to readable clause strings."""
        parts: List[str] = []
        for head in sorted(groups.keys(), key=lambda tok: tok.i):
            text = _join_tokens(groups[head])
            if text:
                parts.append(text)
        return parts

    def _dep_split(self, text: str) -> List[str]:
        """Split ``text`` using dependency-based heuristics."""
        if self._nlp is None:
            return []
        try:
            doc = self._nlp(text)
        except (AssertionError, StopIteration, Exception) as exc:  # pragma: no cover - spaCy edge cases
            logger.warning("Spacy/benepar processing failed: %s. Using fallback.", exc)
            try:
                return NLTKSentenceSplitter().split(text)
            except Exception:
                return [chunk.strip() for chunk in text.replace("\n", " ").split(".") if chunk.strip()]

        clauses: List[str] = []
        for sentence in doc.sents:
            segments = self._segment_on_semicolons(sentence)
            for segment in segments:
                heads = self._heads_in_segment(segment)
                groups = self._assign_tokens(segment, heads)
                parts = self._groups_to_strings(groups)
                clauses.extend(parts)
        return clauses

    def _benepar_split(self, text: str) -> List[str]:
        """Use the Benepar-enhanced parser when available."""
        if self._nlp is None:
            return []
        return self._dep_split(text)

    def split(self, text: str) -> List[str]:  # noqa: D401 - brief override
        """Split ``text`` into clauses via dependency or sentence heuristics."""
        if not text or not text.strip():
            return []
        if self._nlp is None:
            try:
                return NLTKSentenceSplitter().split(text)
            except Exception:
                return [chunk.strip() for chunk in text.replace("\n", " ").split(".") if chunk.strip()]
        raw_clauses = self._benepar_split(text) if self.has_benepar else self._dep_split(text)
        try:
            return _post_filter(raw_clauses, min_score=0.5)
        except Exception:
            return raw_clauses


def make_clause_splitter(mode: str = "advanced") -> ClauseSplitter:
    """Factory returning the configured clause splitter implementation."""
    normalized = (mode or "").lower().strip()
    if normalized in {"nltk", "simple", "fast"}:
        return NLTKSentenceSplitter()
    return BeneparClauseSplitter()