"""Wrapper around the yangheng/deberta-v3-base-end2end-absa baseline model."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from transformers import pipeline
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "transformers is required to use the transformer baseline. Install it via requirements.txt."
    ) from exc


logger = logging.getLogger(__name__)


@dataclass
class AspectResult:
    text: str
    polarity: str
    confidence: float
    start: int
    end: int
    raw: Dict[str, Any] = field(default_factory=dict)


class TransformerABSABaseline:
    """Light wrapper that exposes predictions from the SOTA ABSA model."""

    def __init__(
        self,
        model_name: str = "yangheng/deberta-v3-base-end2end-absa",
        *,
        device: Optional[int] = None,
        local_files_only: bool = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.local_files_only = local_files_only
        self.cache_dir = cache_dir
        self._token_pipeline = None
        self._sequence_pipeline = None
        self._initialize()

    def predict(self, text: str) -> List[AspectResult]:
        """Return aspect predictions with polarity labels for a sentence."""
        text = text or ""
        if not text.strip():
            return []

        predictions: List[AspectResult] = []
        token_outputs = self._run_token_pipeline(text)
        if token_outputs:
            for entry in token_outputs:
                label = str(entry.get("entity_group") or entry.get("label") or "").strip()
                polarity = self._polarity_from_label(label)
                if polarity == "unknown":
                    continue
                raw_word = entry.get("word") or entry.get("text") or ""
                cleaned_word = self._clean_word(raw_word)
                if not cleaned_word:
                    continue
                start = int(entry.get("start") or 0)
                end = int(entry.get("end") or start + len(cleaned_word))
                confidence = float(entry.get("score") or 0.0)
                predictions.append(
                    AspectResult(
                        text=cleaned_word,
                        polarity=polarity,
                        confidence=confidence,
                        start=start,
                        end=end,
                        raw=dict(entry),
                    )
                )

        if not predictions:
            sequence_outputs = self._run_sequence_pipeline(text)
            for entry in sequence_outputs:
                label = str(entry.get("label") or "").strip()
                polarity = self._polarity_from_label(label)
                if polarity == "unknown":
                    continue
                confidence = float(entry.get("score") or 0.0)
                predictions.append(
                    AspectResult(
                        text=text.strip(),
                        polarity=polarity,
                        confidence=confidence,
                        start=0,
                        end=len(text),
                        raw=dict(entry),
                    )
                )

        deduped: Dict[Tuple[str, str], AspectResult] = {}
        ordered: List[AspectResult] = []
        for result in predictions:
            key = (result.text.lower(), result.polarity)
            existing = deduped.get(key)
            if existing is None or result.confidence > existing.confidence:
                deduped[key] = result
                if existing is None:
                    ordered.append(result)
        return ordered

    def polarity_to_valence(self, polarity: str, confidence: float) -> float:
        norm_conf = max(0.0, min(confidence, 1.0))
        if polarity == "positive":
            return norm_conf
        if polarity == "negative":
            return -norm_conf
        return 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        common_kwargs = {
            "model": self.model_name,
            "tokenizer": self.model_name,
            "device": self.device,
            "cache_dir": self.cache_dir,
            "model_kwargs": {"local_files_only": self.local_files_only},
            "tokenizer_kwargs": {"local_files_only": self.local_files_only, "use_fast": True},
        }
        try:
            self._token_pipeline = pipeline(
                "token-classification",
                aggregation_strategy="average",
                **common_kwargs,
            )
        except Exception as exc:  # pragma: no cover - dependent on environment
            logger.warning("Falling back to sequence classification for ABSA token pipeline: %s", exc)
            self._token_pipeline = None

        try:
            self._sequence_pipeline = pipeline(
                "text-classification",
                top_k=None,
                **common_kwargs,
            )
        except Exception:
            self._sequence_pipeline = None

        if self._token_pipeline is None and self._sequence_pipeline is None:
            raise RuntimeError(
                "Unable to initialize transformer baseline. Ensure the model is available locally or network access is enabled."
            )

    def _run_token_pipeline(self, text: str) -> List[Dict[str, Any]]:
        if self._token_pipeline is None:
            return []
        try:
            outputs = self._token_pipeline(text)
        except Exception as exc:  # pragma: no cover
            logger.warning("Token-level ABSA inference failed: %s", exc)
            return []
        if isinstance(outputs, dict):
            outputs = [outputs]
        return list(outputs or [])

    def _run_sequence_pipeline(self, text: str) -> List[Dict[str, Any]]:
        if self._sequence_pipeline is None:
            return []
        try:
            outputs = self._sequence_pipeline(text)
        except Exception as exc:  # pragma: no cover
            logger.warning("Sequence-level ABSA inference failed: %s", exc)
            return []
        if isinstance(outputs, dict):
            outputs = [outputs]
        return list(outputs or [])

    @staticmethod
    def _polarity_from_label(label: str) -> str:
        if not label:
            return "unknown"
        normalized = label.upper()
        if "POS" in normalized or "FAVOR" in normalized:
            return "positive"
        if "NEG" in normalized or "AGAINST" in normalized:
            return "negative"
        if "NEU" in normalized or "NONE" in normalized or "NEUTRAL" in normalized:
            return "neutral"
        if "ASP" in normalized or "OPINION" in normalized:
            # Assume neutral when only aspect tag is present
            return "neutral"
        return "unknown"

    @staticmethod
    def _clean_word(word: str) -> str:
        cleaned = word.replace("▁", " ").replace("##", "").replace("Ġ", " ")
        cleaned = cleaned.replace("<s>", "").replace("</s>", "")
        cleaned = " ".join(cleaned.split())
        return cleaned.strip()


class TransformerBaselinePipeline:
    """Adapter exposing the transformer baseline behind the pipeline interface."""

    def __init__(self, *, model: Optional[TransformerABSABaseline] = None) -> None:
        self.model = model or TransformerABSABaseline()

    def process(self, text: str, debug: bool = False) -> Dict[str, Any]:
        aspects = self.model.predict(text)
        aggregate_results: Dict[int, Dict[str, Any]] = {}
        for idx, aspect in enumerate(aspects, start=1):
            aggregate_results[idx] = {
                "mentions": [
                    {
                        "text": aspect.text,
                        "clause_index": 0,
                        "span": [aspect.start, aspect.end],
                    }
                ],
                "aggregate_sentiment": self.model.polarity_to_valence(aspect.polarity, aspect.confidence),
                "roles": ["associate"],
                "modifiers": [],
                "relation_counts": {},
                "relation_examples": {},
                "transformer": {
                    "raw_polarity": aspect.polarity,
                    "confidence": aspect.confidence,
                    "raw": aspect.raw,
                },
            }

        return {
            "graph": None,
            "clauses": [text],
            "aggregate_results": aggregate_results,
            "relations": [],
            "debug_messages": [{"transformer_debug": debug}] if debug else [],
        }

