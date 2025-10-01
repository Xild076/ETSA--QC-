from typing import List, Dict, Tuple, Optional, Any
import math
import re

_DOUBLE_POSITIVE_PATTERNS = [
    re.compile(r"hard to (?:find|see)[^.!?]*?(?:don't|do not|can't|cannot)\s+(?:like|hate|complain)", re.IGNORECASE),
    re.compile(r"can't\s+complain", re.IGNORECASE),
    re.compile(r"nothing\s+to\s+(?:complain|hate|dislike)", re.IGNORECASE),
    re.compile(r"no\s+(?:real\s+)?complaints?", re.IGNORECASE),
    re.compile(r"nothing\s+like\s+(?:a|an|the)?\s*\w+", re.IGNORECASE),
]

_EXPECTATION_SHORTFALL_PATTERNS = [
    re.compile(r"make better at home", re.IGNORECASE),
    re.compile(r"better at home", re.IGNORECASE),
    re.compile(r"shouldn'?t\s+take", re.IGNORECASE),
    re.compile(r"only\s+\d+\s+usb", re.IGNORECASE),
    re.compile(r"lacks?\s+", re.IGNORECASE),
    re.compile(r"\b(increase|larger)\s+in\s+size\b", re.IGNORECASE),
]

_VALUE_POSITIVE_PATTERNS = [
    re.compile(r"especially considering", re.IGNORECASE),
    re.compile(r"worth (?:every|the)", re.IGNORECASE),
    re.compile(r"hard\s+for\s+me\s+to\s+find\s+things\s+i\s+don't\s+like", re.IGNORECASE),
    re.compile(r"justifies\s+the\s+lack", re.IGNORECASE),
    re.compile(r"\b(low|quiet|cool)\s+(heat|noise|temps|operation)\b", re.IGNORECASE),
    re.compile(r"\b(ultra|very)\s+quiet\b", re.IGNORECASE),
]

_NEUTRAL_LISTING_PATTERNS = [
    re.compile(r"include[s]?\s+classics", re.IGNORECASE),
    re.compile(r"features? include", re.IGNORECASE),
    re.compile(r"comes with", re.IGNORECASE),
    re.compile(r"has (?:a|an|the)?\s*(?:built-in|integrated)", re.IGNORECASE),
    re.compile(r"specs?\s+(?:are|include)", re.IGNORECASE),
]

# New patterns for aspect-sentiment alignment
_ASPECT_FOCUS_PATTERNS = [
    # Patterns that indicate sentiment is specifically about the aspect
    re.compile(r"\b(?:this|that|the)\s+(\w+)\s+(?:is|was|are|were)\s+(very\s+)?(\w+)", re.IGNORECASE),
    re.compile(r"(\w+)\s+(?:is|was|are|were)\s+(?:really|quite|very|extremely)?\s*(\w+)", re.IGNORECASE),
    re.compile(r"(?:love|hate|like|dislike)\s+(?:the|this|that)?\s*(\w+)", re.IGNORECASE),
]

_CONTEXT_ISOLATION_PATTERNS = [
    # Patterns to isolate sentiment from general context
    re.compile(r"(?:although|though|however|but|while|whereas)", re.IGNORECASE),
    re.compile(r"(?:on the other hand|in contrast|conversely)", re.IGNORECASE),
    re.compile(r"(?:except|apart from|aside from)", re.IGNORECASE),
]

_ASPECT_RELEVANCE_KEYWORDS = {
    # Technology/Computer aspects
    "performance", "speed", "battery", "screen", "display", "keyboard", "trackpad", 
    "build", "quality", "design", "price", "value", "size", "weight", "portability",
    "software", "hardware", "memory", "storage", "processor", "graphics", "camera",
    "speakers", "audio", "connectivity", "ports", "wifi", "bluetooth",
    # General product aspects
    "reliability", "durability", "usability", "functionality", "appearance", 
    "comfort", "convenience", "efficiency", "effectiveness", "versatility"
}

_HEAD_OPINION_KEYWORDS = {
    "good",
    "great",
    "bad",
    "poor",
    "awful",
    "amazing",
    "terrible",
    "lousy",
    "fast",
    "slow",
    "worth",
    "balanced",
    "quiet",
    "loud",
    "cheap",
    "expensive",
    "overpriced",
    "reliable",
    "unreliable",
    "love",
    "hate",
    "fantastic",
}

def _clamp_score(score: float) -> float:
    # Handle NaN and infinity cases
    if not math.isfinite(score):
        return 0.0
    return max(min(score, 1.0), -1.0)

def _modifier_polarity_summary(per_scores: Dict[str, float], threshold: tuple) -> Dict[str, int]:
    pos_threshold, neg_threshold = threshold
    return {
        "positive": sum(1 for score in per_scores.values() if score > pos_threshold),
        "negative": sum(1 for score in per_scores.values() if score < neg_threshold),
        "neutral": sum(1 for score in per_scores.values() if neg_threshold <= score <= pos_threshold),
    }

def _adjust_head_sentiment(head_text: str, head_sentiment: float, modifiers: List[str], head_dampening_factor: float = 0.35) -> Tuple[float, Optional[str]]:
    if modifiers:
        return head_sentiment, None
    cleaned = (head_text or "").strip()
    if not cleaned:
        return 0.0, "empty head -> neutral"
    tokens = cleaned.lower().split()
    if any(token in _HEAD_OPINION_KEYWORDS for token in tokens):
        return head_sentiment, None
    dampened = head_sentiment * head_dampening_factor
    return dampened, f"head-only noun dampened from {head_sentiment:+.2f} to {dampened:+.2f}"

def _adjust_modifier_sentiment(
    modifiers: List[str],
    base_score: float,
    clause_text: str,
) -> Tuple[float, Optional[str]]:
    if not modifiers:
        return base_score, None
    joined = " ".join(modifiers)
    composite = f"{joined} {clause_text or ''}".lower()
    adjusted = base_score
    notes: List[str] = []

    for pattern in _DOUBLE_POSITIVE_PATTERNS:
        if pattern.search(composite):
            if adjusted < 0:
                adjusted = abs(adjusted)
                notes.append("double-negative praise flipped positive")
            break

    for pattern in _EXPECTATION_SHORTFALL_PATTERNS:
        if pattern.search(composite):
            if adjusted > 0:
                adjusted = -abs(adjusted)
                notes.append("expectation shortfall forced negative")
            break

    for pattern in _VALUE_POSITIVE_PATTERNS:
        if pattern.search(composite):
            if adjusted <= 0:
                adjusted = max(abs(adjusted), 0.4)
            notes.append("value framing boosted positive sentiment")
            break

    for pattern in _NEUTRAL_LISTING_PATTERNS:
        if pattern.search(composite):
            adjusted *= 0.3
            notes.append("listing detected -> magnitude dampened")
            break

    return _clamp_score(adjusted), "; ".join(notes) if notes else None

def _score_polarity(score: float, threshold: tuple) -> int:
    pos_threshold, neg_threshold = threshold
    if score > pos_threshold:
        return 1
    if score < neg_threshold:
        return -1
    return 0

def _legacy_blend(
    head_sentiment: float,
    modifier_sentiment: float,
    has_modifiers: bool,
    threshold: tuple,
) -> Tuple[float, str]:
    pos_threshold, neg_threshold = threshold
    head_polarity = _score_polarity(head_sentiment, threshold)
    modifier_polarity = _score_polarity(modifier_sentiment, threshold)

    if not has_modifiers:
        if abs(head_sentiment) >= 0.5:
            return head_sentiment, f"No modifiers; using strong head sentiment ({head_sentiment:+.2f})."
        return 0.0, f"No modifiers and weak head sentiment ({head_sentiment:+.2f}); defaulting to neutral."

    if modifier_polarity != 0 and head_polarity != 0 and modifier_polarity != head_polarity:
        return modifier_sentiment, (
            f"Polarity conflict: Head ({head_sentiment:+.2f}) vs. Modifier ({modifier_sentiment:+.2f}). "
            "Modifier sentiment overrides."
        )

    if head_polarity == modifier_polarity and head_polarity != 0:
        final = (head_sentiment + modifier_sentiment) / 2
        return final, (
            f"Polarities agree: Head ({head_sentiment:+.2f}) and Modifier ({modifier_sentiment:+.2f}). "
            "Sentiments averaged."
        )

    if modifier_polarity == 0:
        return head_sentiment, f"Neutral modifier; using head sentiment ({head_sentiment:+.2f})."

    final = (head_sentiment + modifier_sentiment) / 2
    return final, (
        f"Default case (concordant polarity): Averaging Head ({head_sentiment:+.2f}) and Modifier ({modifier_sentiment:+.2f})."
    )

class SentimentCombiner:
    def __init__(self, **kwargs):
        self.head_dampening_factor = kwargs.get('head_dampening_factor', 0.35)
        self.modifier_weight = kwargs.get('modifier_weight', 0.5)
        self.head_weight = kwargs.get('head_weight', 0.5)
        self.context_weight = kwargs.get('context_weight', 0.3)
        self.conflict_modifier_priority = kwargs.get('conflict_modifier_priority', True)
        self.mixed_polarity_context_boost = kwargs.get('mixed_polarity_context_boost', 0.4)
        self.adaptive_strength_threshold = kwargs.get('adaptive_strength_threshold', 0.0)
        self.no_modifier_dampening = kwargs.get('no_modifier_dampening', 0.4)

    PARAM_RANGES = {
        'head_dampening_factor': {'type': 'float', 'low': 0.1, 'high': 0.7, 'step': 0.005},
        'no_modifier_dampening': {'type': 'float', 'low': 0.2, 'high': 0.6, 'step': 0.005},
    }

    def combine(
        self,
        head_text: str,
        head_sentiment: float,
        modifiers: List[str],
        modifier_sentiment: float,
        threshold: tuple,
        context_score: Optional[float],
        per_scores: Dict[str, float],
        clause_text: str,
    ) -> Tuple[float, str, Dict[str, Any]]:
        raise NotImplementedError

    def get_debug_info(self) -> Dict[str, Any]:
        return {
            'combiner_type': self.__class__.__name__,
            'parameters': {k: v for k, v in self.__dict__.items() if not k.startswith('_')},
            'param_ranges': getattr(self, 'PARAM_RANGES', {})
        }
    
    def _assess_aspect_relevance(self, head_text: str, clause_text: str) -> float:
        """Assess how relevant the head aspect is to the context for sentiment focus."""
        if not head_text or not clause_text:
            return 1.0
        
        head_lower = head_text.lower()
        clause_lower = clause_text.lower()
        
        # Check if aspect is explicitly mentioned with sentiment indicators
        for pattern in _ASPECT_FOCUS_PATTERNS:
            if pattern.search(clause_lower):
                match = pattern.search(clause_lower)
                if match and head_lower in match.group().lower():
                    return 1.2  # Higher relevance for explicit aspect-sentiment patterns
        
        # Check if aspect is a recognizable product aspect
        aspect_words = head_lower.split()
        relevant_count = sum(1 for word in aspect_words if word in _ASPECT_RELEVANCE_KEYWORDS)
        if relevant_count > 0:
            return 1.0 + (relevant_count * 0.1)  # Boost for product-relevant aspects
        
        # Check if aspect appears multiple times (indicates importance)
        aspect_frequency = clause_lower.count(head_lower)
        if aspect_frequency > 1:
            return 1.0 + min(aspect_frequency * 0.05, 0.2)
        
        return 1.0
    
    def _detect_context_isolation(self, clause_text: str) -> bool:
        """Detect if the clause contains patterns that separate contexts."""
        if not clause_text:
            return False
        
        for pattern in _CONTEXT_ISOLATION_PATTERNS:
            if pattern.search(clause_text):
                return True
        return False
    
    def _apply_aspect_focus_adjustment(self, sentiment: float, head_text: str, 
                                     clause_text: str, modifiers: List[str]) -> Tuple[float, List[str]]:
        """Apply adjustments to focus sentiment on the specific aspect."""
        notes = []
        
        # Assess aspect relevance
        relevance = self._assess_aspect_relevance(head_text, clause_text)
        if relevance > 1.0:
            sentiment *= relevance
            notes.append(f"aspect_relevance_boost: {relevance:.2f}")
        
        # Check for context isolation (conflicting sentiments in same sentence)
        if self._detect_context_isolation(clause_text):
            # Reduce influence of general context when there are contrasting elements
            sentiment *= 0.9  # Slight dampening to focus on aspect-specific sentiment
            notes.append("context_isolation_detected")
        
        # If no modifiers but aspect appears in a sentiment-rich context, be conservative
        if not modifiers and clause_text:
            clause_lower = clause_text.lower()
            sentiment_word_count = sum(1 for word in clause_lower.split() 
                                     if word in {'good', 'bad', 'great', 'terrible', 'amazing', 
                                               'awful', 'excellent', 'poor', 'love', 'hate'})
            if sentiment_word_count > 2:  # Rich sentiment context but no direct modifiers
                sentiment *= 0.8  # Be more conservative
                notes.append("rich_context_conservative")
        
        return sentiment, notes

class BalancedV1Combiner(SentimentCombiner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.context_blend_weight = kwargs.get('context_blend_weight', 0.7)
        self.context_blend_factor = kwargs.get('context_blend_factor', 0.3)
        self.modifier_confidence_weight = kwargs.get('modifier_confidence_weight', 0.55)
        self.neutral_guard_band = kwargs.get('neutral_guard_band', 0.1)
        self.confidence_gamma = kwargs.get('confidence_gamma', 1.2)
        self.context_recovery_weight = kwargs.get('context_recovery_weight', 0.4)
    PARAM_RANGES = {
        'context_blend_weight': {'type': 'float', 'low': 0.3, 'high': 0.85, 'step': 0.005},
        'context_blend_factor': {'type': 'float', 'low': 0.0, 'high': 0.6, 'step': 0.005},
        'modifier_confidence_weight': {'type': 'float', 'low': 0.2, 'high': 0.8, 'step': 0.005},
        'neutral_guard_band': {'type': 'float', 'low': 0.0, 'high': 0.25, 'step': 0.005},
        'confidence_gamma': {'type': 'float', 'low': 0.5, 'high': 2.5, 'step': 0.01},
        'context_recovery_weight': {'type': 'float', 'low': 0.0, 'high': 0.8, 'step': 0.005},
        'head_dampening_factor': {'type': 'float', 'low': 0.1, 'high': 0.7, 'step': 0.005},
    }

    def combine(
        self,
        head_text: str,
        head_sentiment: float,
        modifiers: List[str],
        modifier_sentiment: float,
        threshold: tuple,
        context_score: Optional[float],
        per_scores: Dict[str, float],
        clause_text: str,
    ) -> Tuple[float, str, Dict[str, Any]]:
        head_adjusted, head_note = _adjust_head_sentiment(head_text, head_sentiment, modifiers, self.head_dampening_factor)
        modifier_adjusted, modifier_note = _adjust_modifier_sentiment(modifiers, modifier_sentiment, clause_text)
        base_score, justification = _legacy_blend(head_adjusted, modifier_adjusted, bool(modifiers), threshold)

        # Confidence modelling based on consistency and modifier strength
        modifier_strength = abs(modifier_adjusted)
        head_strength = abs(head_adjusted)
        reliability = (modifier_strength * self.modifier_confidence_weight) + (head_strength * (1 - self.modifier_confidence_weight))
        reliability += min(0.3, 0.05 * len(modifiers))
        reliability = max(0.0, min(1.0, reliability))

        consensus = (head_adjusted + modifier_adjusted) / 2 if modifiers else head_adjusted
        smoothed = base_score
        if modifiers:
            smoothed = (1 - self.neutral_guard_band) * base_score + self.neutral_guard_band * consensus

        confidence_scaled = smoothed * max(0.0, reliability ** self.confidence_gamma)

        context_applied = confidence_scaled
        context_delta = 0.0
        if context_score is not None and modifiers:
            blended = self.context_blend_weight * confidence_scaled + self.context_blend_factor * context_score
            context_applied = (1 - self.context_recovery_weight) * confidence_scaled + self.context_recovery_weight * blended
            context_delta = context_applied - confidence_scaled
            justification += f" Context adjusted ({context_score:+.2f}) with recovery weight {self.context_recovery_weight:.2f}."

        heuristic_notes = [note for note in (head_note, modifier_note) if note]
        heuristic_notes.append(f"reliability={reliability:.2f}")
        if context_delta:
            heuristic_notes.append(f"context_delta={context_delta:+.2f}")
        justification += " Heuristics: " + "; ".join(heuristic_notes) + "."

        final_score = _clamp_score(context_applied)

        return final_score, justification, {
            "head_adjusted": head_adjusted,
            "modifier_adjusted": modifier_adjusted,
            "heuristic_notes": heuristic_notes,
            "combiner_debug": {
                "combiner_type": "BalancedV1Combiner",
                "context_blend_weight": self.context_blend_weight,
                "context_blend_factor": self.context_blend_factor,
                "context_recovery_weight": self.context_recovery_weight,
                "confidence_gamma": self.confidence_gamma,
                "reliability": reliability,
                "smoothed_score": smoothed,
                "context_applied": context_applied,
                "final_score": final_score,
            }
        }

class ModifierDominantV2Combiner(SentimentCombiner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.modifier_weight_conflict = kwargs.get('modifier_weight_conflict', 0.8)
        self.head_weight_conflict = kwargs.get('head_weight_conflict', 0.2)
        self.context_stabilizer_weight = kwargs.get('context_stabilizer_weight', 0.85)
        self.context_stabilizer_factor = kwargs.get('context_stabilizer_factor', 0.15)
        self.reliability_floor = kwargs.get('reliability_floor', 0.3)
        self.reliability_gamma = kwargs.get('reliability_gamma', 1.1)

    def combine(
        self,
        head_text: str,
        head_sentiment: float,
        modifiers: List[str],
        modifier_sentiment: float,
        threshold: tuple,
        context_score: Optional[float],
        per_scores: Dict[str, float],
        clause_text: str,
    ) -> Tuple[float, str, Dict[str, Any]]:
        head_adjusted, head_note = _adjust_head_sentiment(head_text, head_sentiment, modifiers, self.head_dampening_factor)
        modifier_adjusted, modifier_note = _adjust_modifier_sentiment(modifiers, modifier_sentiment, clause_text)
        head_pol = _score_polarity(head_adjusted, threshold)
        mod_pol = _score_polarity(modifier_adjusted, threshold)

        reliability = self.reliability_floor
        if modifiers:
            if head_pol != 0 and mod_pol != 0 and head_pol != mod_pol:
                score = modifier_adjusted
                justification = (
                    f"Modifier priority resolved polarity conflict: modifier {modifier_adjusted:+.2f} overrides head {head_adjusted:+.2f}."
                )
            else:
                score = self.modifier_weight_conflict * modifier_adjusted + self.head_weight_conflict * head_adjusted
                justification = (
                    f"Modifier-forward blend ({self.modifier_weight_conflict:.1f}/{self.head_weight_conflict:.1f}) of modifier ({modifier_adjusted:+.2f}) and head ({head_adjusted:+.2f})."
                )
            if context_score is not None:
                score = self.context_stabilizer_weight * score + self.context_stabilizer_factor * context_score
                justification += f" Context stabiliser ({context_score:+.2f}) applied."
            reliability = max(self.reliability_floor, min(1.0, abs(modifier_adjusted) + 0.5 * abs(head_adjusted)))
        else:
            score = head_adjusted * self.no_modifier_dampening
            justification = f"No modifiers; head sentiment ({head_adjusted:+.2f}) softened to avoid noun bias."
            reliability = max(self.reliability_floor, min(1.0, abs(head_adjusted)))

        score *= max(self.reliability_floor, reliability ** self.reliability_gamma)
        justification += f" Reliability scaled ({reliability:.2f})."

        heuristic_notes = [note for note in (head_note, modifier_note) if note]
        heuristic_notes.append(f"reliability={reliability:.2f}")
        if heuristic_notes:
            justification += " Heuristics: " + "; ".join(heuristic_notes) + "."

        final_score = _clamp_score(score)

        return final_score, justification, {
            "head_adjusted": head_adjusted,
            "modifier_adjusted": modifier_adjusted,
            "heuristic_notes": heuristic_notes,
            "combiner_debug": {
                "combiner_type": "ModifierDominantV2Combiner",
                "modifier_weight_conflict": self.modifier_weight_conflict,
                "head_weight_conflict": self.head_weight_conflict,
                "context_stabilizer_weight": self.context_stabilizer_weight,
                "context_stabilizer_factor": self.context_stabilizer_factor,
                "no_modifier_dampening": self.no_modifier_dampening,
                "head_dampening_factor": self.head_dampening_factor,
                "conflict_resolution": "modifier_override" if (modifiers and head_pol != 0 and mod_pol != 0 and head_pol != mod_pol) else "weighted_blend",
                "context_applied": context_score is not None and modifiers,
                "final_blend": f"{self.context_stabilizer_weight:.2f} * {score:.3f} + {self.context_stabilizer_factor:.2f} * {context_score:.3f}" if context_score is not None and modifiers else "No context blending",
                "reliability": reliability,
                "final_score": final_score,
            }
        }

    PARAM_RANGES = {
        'modifier_weight_conflict': {'type': 'float', 'low': 0.6, 'high': 0.95, 'step': 0.005},
        'head_weight_conflict': {'type': 'float', 'low': 0.05, 'high': 0.4, 'step': 0.005},
        'context_stabilizer_weight': {'type': 'float', 'low': 0.6, 'high': 0.95, 'step': 0.005},
        'context_stabilizer_factor': {'type': 'float', 'low': 0.05, 'high': 0.4, 'step': 0.005},
        'no_modifier_dampening': {'type': 'float', 'low': 0.2, 'high': 0.6, 'step': 0.005},
        'head_dampening_factor': {'type': 'float', 'low': 0.1, 'high': 0.7, 'step': 0.005},
        'reliability_floor': {'type': 'float', 'low': 0.1, 'high': 0.5, 'step': 0.005},
        'reliability_gamma': {'type': 'float', 'low': 0.8, 'high': 1.6, 'step': 0.005},
    }

class ContextualV3Combiner(SentimentCombiner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mixed_polarity_avg_weight = kwargs.get('mixed_polarity_avg_weight', 0.6)
        self.mixed_polarity_context_weight = kwargs.get('mixed_polarity_context_weight', 0.4)
        self.modifier_weight_strong = kwargs.get('modifier_weight_strong', 0.55)
        self.modifier_weight_weak = kwargs.get('modifier_weight_weak', 0.45)
        self.head_weight_strong = kwargs.get('head_weight_strong', 0.3)
        self.head_weight_weak = kwargs.get('head_weight_weak', 0.25)
        self.no_modifier_head_weight = kwargs.get('no_modifier_head_weight', 0.5)
        self.no_modifier_context_weight = kwargs.get('no_modifier_context_weight', 0.5)
        self.no_modifier_dampening_v3 = kwargs.get('no_modifier_dampening_v3', 0.25)
        self.context_confidence_weight = kwargs.get('context_confidence_weight', 0.5)
        self.reliability_gamma = kwargs.get('reliability_gamma', 1.15)
        self.reliability_floor = kwargs.get('reliability_floor', 0.25)
        self.mixed_context_boost = kwargs.get('mixed_context_boost', 0.35)
    PARAM_RANGES = {
        'mixed_polarity_avg_weight': {'type': 'float', 'low': 0.4, 'high': 0.8, 'step': 0.005},
        'mixed_polarity_context_weight': {'type': 'float', 'low': 0.2, 'high': 0.6, 'step': 0.005},
        'modifier_weight_strong': {'type': 'float', 'low': 0.45, 'high': 0.65, 'step': 0.005},
        'modifier_weight_weak': {'type': 'float', 'low': 0.35, 'high': 0.55, 'step': 0.005},
        'head_weight_strong': {'type': 'float', 'low': 0.2, 'high': 0.4, 'step': 0.005},
        'head_weight_weak': {'type': 'float', 'low': 0.15, 'high': 0.35, 'step': 0.005},
        'no_modifier_head_weight': {'type': 'float', 'low': 0.3, 'high': 0.7, 'step': 0.005},
        'no_modifier_context_weight': {'type': 'float', 'low': 0.3, 'high': 0.7, 'step': 0.005},
        'no_modifier_dampening_v3': {'type': 'float', 'low': 0.15, 'high': 0.35, 'step': 0.005},
        'context_confidence_weight': {'type': 'float', 'low': 0.2, 'high': 0.8, 'step': 0.005},
        'reliability_gamma': {'type': 'float', 'low': 0.8, 'high': 1.8, 'step': 0.005},
        'reliability_floor': {'type': 'float', 'low': 0.1, 'high': 0.5, 'step': 0.01},
        'mixed_context_boost': {'type': 'float', 'low': 0.1, 'high': 0.6, 'step': 0.01},
        'head_dampening_factor': {'type': 'float', 'low': 0.1, 'high': 0.7, 'step': 0.01},
    }

    def combine(
        self,
        head_text: str,
        head_sentiment: float,
        modifiers: List[str],
        modifier_sentiment: float,
        threshold: tuple,
        context_score: Optional[float],
        per_scores: Dict[str, float],
        clause_text: str,
    ) -> Tuple[float, str, Dict[str, Any]]:
        head_adjusted, head_note = _adjust_head_sentiment(head_text, head_sentiment, modifiers, self.head_dampening_factor)
        modifier_adjusted, modifier_note = _adjust_modifier_sentiment(modifiers, modifier_sentiment, clause_text)

        summary = _modifier_polarity_summary(per_scores, threshold)
        heuristic_notes = [note for note in (head_note, modifier_note) if note]

        reliability = self.reliability_floor
        if modifiers:
            if summary["positive"] and summary["negative"]:
                ctx_component = context_score if context_score is not None else 0.0
                score = self.mixed_polarity_avg_weight * ((modifier_adjusted + head_adjusted) / 2) + self.mixed_polarity_context_weight * ctx_component
                ctx_phrase = f"{ctx_component:+.2f}" if context_score is not None else "0.00"
                justification = (
                    f"Mixed modifier polarities ({summary['positive']}+/ {summary['negative']}-); averaged with context ({ctx_phrase})."
                )
                reliability = min(1.0, self.reliability_floor + self.mixed_context_boost * (summary['positive'] + summary['negative']))
            else:
                weight_mod = self.modifier_weight_strong if abs(modifier_adjusted) >= abs(head_adjusted) else self.modifier_weight_weak
                weight_head = self.head_weight_strong if abs(head_adjusted) > abs(modifier_adjusted) else self.head_weight_weak
                weight_ctx = 1.0 - weight_mod - weight_head
                if weight_ctx < 0:
                    weight_ctx = 0.0
                total_weight = weight_mod + weight_head + weight_ctx
                if total_weight > 0:
                    weight_mod /= total_weight
                    weight_head /= total_weight
                    weight_ctx /= total_weight
                ctx_component = context_score if context_score is not None else modifier_adjusted
                score = (
                    weight_mod * modifier_adjusted
                    + weight_head * head_adjusted
                    + weight_ctx * ctx_component
                )
                justification = (
                    "Context-aware blend "
                    f"(mod {weight_mod:.2f}, head {weight_head:.2f}, ctx {weight_ctx:.2f}) using context {ctx_component:+.2f}."
                )
                reliability = max(
                    self.reliability_floor,
                    min(1.0, abs(modifier_adjusted) * self.context_confidence_weight + abs(head_adjusted) * (1 - self.context_confidence_weight)),
                )
        else:
            if context_score is not None:
                score = self.no_modifier_head_weight * head_adjusted + self.no_modifier_context_weight * context_score
                justification = (
                    f"No modifiers; balanced head ({head_adjusted:+.2f}) with clause context ({context_score:+.2f})."
                )
                reliability = max(self.reliability_floor, min(1.0, abs(head_adjusted) + abs(context_score) * 0.5))
            else:
                score = head_adjusted * self.no_modifier_dampening_v3
                justification = (
                    f"No modifiers or context; heavily dampened head sentiment ({head_adjusted:+.2f})."
                )
                reliability = max(self.reliability_floor, min(1.0, abs(head_adjusted)))

        reliability = max(self.reliability_floor, min(1.0, reliability))
        score *= max(self.reliability_floor, reliability ** self.reliability_gamma)
        heuristic_notes.append(f"reliability={reliability:.2f}")

        # Apply aspect-focused adjustments for improved precision
        score, aspect_notes = self._apply_aspect_focus_adjustment(score, head_text, clause_text, modifiers)
        heuristic_notes.extend(aspect_notes)

        if heuristic_notes:
            justification += " Heuristics: " + "; ".join(heuristic_notes) + "."

        final_score = _clamp_score(score)

        return final_score, justification, {
            "head_adjusted": head_adjusted,
            "modifier_adjusted": modifier_adjusted,
            "heuristic_notes": heuristic_notes,
            "combiner_debug": {
                "combiner_type": "ContextualV3Combiner",
                "reliability": reliability,
                "final_score": final_score,
                "context_confidence_weight": self.context_confidence_weight,
                "reliability_gamma": self.reliability_gamma,
                "reliability_floor": self.reliability_floor,
                "mixed_context_boost": self.mixed_context_boost,
            }
        }

class HeadDominantV4Combiner(SentimentCombiner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.head_weight_v4 = kwargs.get('head_weight_v4', 0.8)
        self.modifier_weight_v4 = kwargs.get('modifier_weight_v4', 0.2)
        self.context_stabilizer_weight_v4 = kwargs.get('context_stabilizer_weight_v4', 0.9)
        self.context_stabilizer_factor_v4 = kwargs.get('context_stabilizer_factor_v4', 0.1)
        self.reliability_gamma_v4 = kwargs.get('reliability_gamma_v4', 1.0)
        self.reliability_floor_v4 = kwargs.get('reliability_floor_v4', 0.25)

    def combine(
        self,
        head_text: str,
        head_sentiment: float,
        modifiers: List[str],
        modifier_sentiment: float,
        threshold: tuple,
        context_score: Optional[float],
        per_scores: Dict[str, float],
        clause_text: str,
    ) -> Tuple[float, str, Dict[str, Any]]:
        head_adjusted, head_note = _adjust_head_sentiment(head_text, head_sentiment, modifiers, self.head_dampening_factor)
        modifier_adjusted, modifier_note = _adjust_modifier_sentiment(modifiers, modifier_sentiment, clause_text)

        reliability = self.reliability_floor_v4
        if modifiers:
            score = self.head_weight_v4 * head_adjusted + self.modifier_weight_v4 * modifier_adjusted
            justification = f"Head-dominant blend ({self.head_weight_v4:.1f}/{self.modifier_weight_v4:.1f}) of head ({head_adjusted:+.2f}) and modifier ({modifier_adjusted:+.2f})."
            if context_score is not None:
                score = self.context_stabilizer_weight_v4 * score + self.context_stabilizer_factor_v4 * context_score
                justification += f" Context stabilizer ({context_score:+.2f}) applied."
            reliability = max(self.reliability_floor_v4, min(1.0, abs(head_adjusted)))
        else:
            score = head_adjusted
            justification = f"No modifiers; using head sentiment ({head_adjusted:+.2f})."
            reliability = max(self.reliability_floor_v4, min(1.0, abs(head_adjusted)))

        score *= max(self.reliability_floor_v4, reliability ** self.reliability_gamma_v4)
        justification += f" Reliability scaled ({reliability:.2f})."

        heuristic_notes = [note for note in (head_note, modifier_note) if note]
        heuristic_notes.append(f"reliability={reliability:.2f}")
        if heuristic_notes:
            justification += " Heuristics: " + "; ".join(heuristic_notes) + "."

        final_score = _clamp_score(score)

        return final_score, justification, {
            "head_adjusted": head_adjusted,
            "modifier_adjusted": modifier_adjusted,
            "heuristic_notes": heuristic_notes,
            "combiner_debug": {
                "combiner_type": "HeadDominantV4Combiner",
                "reliability": reliability,
                "final_score": final_score,
            }
        }
    PARAM_RANGES = {
        'head_weight_v4': {'type': 'float', 'low': 0.6, 'high': 0.95, 'step': 0.005},
        'modifier_weight_v4': {'type': 'float', 'low': 0.05, 'high': 0.4, 'step': 0.005},
        'context_stabilizer_weight_v4': {'type': 'float', 'low': 0.7, 'high': 0.95, 'step': 0.005},
        'context_stabilizer_factor_v4': {'type': 'float', 'low': 0.05, 'high': 0.3, 'step': 0.005},
        'head_dampening_factor': {'type': 'float', 'low': 0.1, 'high': 0.7, 'step': 0.005},
        'reliability_gamma_v4': {'type': 'float', 'low': 0.8, 'high': 1.6, 'step': 0.005},
        'reliability_floor_v4': {'type': 'float', 'low': 0.15, 'high': 0.5, 'step': 0.005},
    }

class EqualWeightV5Combiner(SentimentCombiner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.equal_weight_blend = kwargs.get('equal_weight_blend', 0.5)
        self.context_influence_weight = kwargs.get('context_influence_weight', 0.8)
        self.context_influence_factor = kwargs.get('context_influence_factor', 0.2)
        self.no_modifier_softening = kwargs.get('no_modifier_softening', 0.5)
        self.reliability_gamma_v5 = kwargs.get('reliability_gamma_v5', 1.05)
        self.reliability_floor_v5 = kwargs.get('reliability_floor_v5', 0.25)

    def combine(
        self,
        head_text: str,
        head_sentiment: float,
        modifiers: List[str],
        modifier_sentiment: float,
        threshold: tuple,
        context_score: Optional[float],
        per_scores: Dict[str, float],
        clause_text: str,
    ) -> Tuple[float, str, Dict[str, Any]]:
        head_adjusted, head_note = _adjust_head_sentiment(head_text, head_sentiment, modifiers, self.head_dampening_factor)
        modifier_adjusted, modifier_note = _adjust_modifier_sentiment(modifiers, modifier_sentiment, clause_text)

        reliability = self.reliability_floor_v5
        if modifiers:
            blend = max(0.0, min(1.0, self.equal_weight_blend))
            score = blend * head_adjusted + (1 - blend) * modifier_adjusted
            justification = f"Equal-weight blend ({blend:.2f}/{1 - blend:.2f}) of head ({head_adjusted:+.2f}) and modifier ({modifier_adjusted:+.2f})."
            if context_score is not None:
                score = self.context_influence_weight * score + self.context_influence_factor * context_score
                justification += f" Context influence ({context_score:+.2f}) added."
            reliability = max(self.reliability_floor_v5, min(1.0, (abs(head_adjusted) + abs(modifier_adjusted)) / 2))
        else:
            score = head_adjusted * self.no_modifier_softening
            justification = f"No modifiers; softened head sentiment ({head_adjusted:+.2f})."
            reliability = max(self.reliability_floor_v5, min(1.0, abs(head_adjusted)))

        score *= max(self.reliability_floor_v5, reliability ** self.reliability_gamma_v5)
        justification += f" Reliability scaled ({reliability:.2f})."

        heuristic_notes = [note for note in (head_note, modifier_note) if note]
        heuristic_notes.append(f"reliability={reliability:.2f}")
        if heuristic_notes:
            justification += " Heuristics: " + "; ".join(heuristic_notes) + "."

        final_score = _clamp_score(score)

        return final_score, justification, {
            "head_adjusted": head_adjusted,
            "modifier_adjusted": modifier_adjusted,
            "heuristic_notes": heuristic_notes,
            "combiner_debug": {
                "combiner_type": "EqualWeightV5Combiner",
                "reliability": reliability,
                "final_score": final_score,
            }
        }
    PARAM_RANGES = {
        'equal_weight_blend': {'type': 'float', 'low': 0.3, 'high': 0.7, 'step': 0.005},
        'context_influence_weight': {'type': 'float', 'low': 0.6, 'high': 0.95, 'step': 0.005},
        'context_influence_factor': {'type': 'float', 'low': 0.05, 'high': 0.35, 'step': 0.005},
        'no_modifier_softening': {'type': 'float', 'low': 0.3, 'high': 0.7, 'step': 0.005},
        'head_dampening_factor': {'type': 'float', 'low': 0.1, 'high': 0.7, 'step': 0.005},
        'reliability_gamma_v5': {'type': 'float', 'low': 0.8, 'high': 1.5, 'step': 0.005},
        'reliability_floor_v5': {'type': 'float', 'low': 0.15, 'high': 0.5, 'step': 0.005},
    }

class AdaptiveV6Combiner(SentimentCombiner):
    NEGATION_PATTERNS = {
        'not', 'no', 'never', 'nothing', "n't", 'dont', "doesn't", "didn't",
        'cannot', "can't", 'wont', "won't", 'neither', 'nor', 'without', 'none'
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.adaptive_mod_strong_weight = kwargs.get('adaptive_mod_strong_weight', 0.7)
        self.adaptive_head_strong_weight = kwargs.get('adaptive_head_strong_weight', 0.3)
        self.adaptive_mod_weak_weight = kwargs.get('adaptive_mod_weak_weight', 0.4)
        self.adaptive_head_weak_weight = kwargs.get('adaptive_head_weak_weight', 0.6)
        self.adaptive_context_weight = kwargs.get('adaptive_context_weight', 0.85)
        self.adaptive_context_factor = kwargs.get('adaptive_context_factor', 0.15)
        self.adaptive_no_modifier_dampening = kwargs.get('adaptive_no_modifier_dampening', 0.6)
        self.reliability_gamma_v6 = kwargs.get('reliability_gamma_v6', 1.1)
        self.reliability_floor_v6 = kwargs.get('reliability_floor_v6', 0.25)
        self.negation_boost = kwargs.get('negation_boost', 1.25)
        self.modifier_quality_weight = kwargs.get('modifier_quality_weight', 0.15)
    
    def _detect_negation(self, text: str) -> bool:
        tokens = text.lower().split()
        text_lower = text.lower()
        return any(neg in tokens or neg in text_lower for neg in self.NEGATION_PATTERNS)
    
    def _assess_modifier_quality(self, modifiers: List[str]) -> float:
        if not modifiers:
            return 0.0
        quality_score = 0.0
        for mod in modifiers:
            mod_lower = mod.lower()
            if any(neg in mod_lower.split() for neg in self.NEGATION_PATTERNS):
                quality_score += 0.3
            if len(mod.split()) >= 3:
                quality_score += 0.2
            if any(word in mod_lower for word in ['very', 'extremely', 'quite', 'really', 'highly']):
                quality_score += 0.15
            quality_score += 0.35
        return min(quality_score / len(modifiers), 1.0)

    def combine(
        self,
        head_text: str,
        head_sentiment: float,
        modifiers: List[str],
        modifier_sentiment: float,
        threshold: tuple,
        context_score: Optional[float],
        per_scores: Dict[str, float],
        clause_text: str,
    ) -> Tuple[float, str, Dict[str, Any]]:
        head_adjusted, head_note = _adjust_head_sentiment(head_text, head_sentiment, modifiers, self.head_dampening_factor)
        modifier_adjusted, modifier_note = _adjust_modifier_sentiment(modifiers, modifier_sentiment, clause_text)

        has_negation = self._detect_negation(clause_text) or any(self._detect_negation(m) for m in modifiers)
        modifier_quality = self._assess_modifier_quality(modifiers)

        reliability = self.reliability_floor_v6
        if modifiers:
            mod_strength = abs(modifier_adjusted)
            head_strength = abs(head_adjusted)
            
            quality_factor = 1.0 + (modifier_quality * self.modifier_quality_weight)
            mod_strength_adjusted = mod_strength * quality_factor
            
            if mod_strength_adjusted > head_strength:
                weight_mod = self.adaptive_mod_strong_weight
                weight_head = self.adaptive_head_strong_weight
            else:
                weight_mod = self.adaptive_mod_weak_weight
                weight_head = self.adaptive_head_weak_weight
            
            score = weight_mod * modifier_adjusted + weight_head * head_adjusted
            justification = f"Adaptive blend (mod {weight_mod:.1f}, head {weight_head:.1f}) based on strengths ({mod_strength:.2f} vs {head_strength:.2f})."
            
            if has_negation:
                if score < 0:
                    score *= self.negation_boost
                    justification += f" Negation boost applied ({self.negation_boost:.2f}x)."
                elif score > 0:
                    score *= (2.0 - self.negation_boost)
                    justification += f" Negation damping applied."
            
            if context_score is not None:
                score = self.adaptive_context_weight * score + self.adaptive_context_factor * context_score
                justification += f" Context adjustment ({context_score:+.2f})."
            
            reliability = max(self.reliability_floor_v6, min(1.0, (mod_strength_adjusted + head_strength) / 2))
            if modifier_quality > 0.5:
                reliability *= (1.0 + modifier_quality * 0.2)
                reliability = min(1.0, reliability)
        else:
            score = head_adjusted * self.adaptive_no_modifier_dampening
            justification = f"No modifiers; moderately dampened head ({head_adjusted:+.2f})."
            reliability = max(self.reliability_floor_v6, min(1.0, abs(head_adjusted)))

        score *= max(self.reliability_floor_v6, reliability ** self.reliability_gamma_v6)
        justification += f" Reliability scaled ({reliability:.2f})."

        heuristic_notes = [note for note in (head_note, modifier_note) if note]
        heuristic_notes.append(f"reliability={reliability:.2f}")
        if has_negation:
            heuristic_notes.append("negation_detected")
        if modifier_quality > 0.5:
            heuristic_notes.append(f"quality={modifier_quality:.2f}")
        if heuristic_notes:
            justification += " Heuristics: " + "; ".join(heuristic_notes) + "."

        final_score = _clamp_score(score)

        return final_score, justification, {
            "head_adjusted": head_adjusted,
            "modifier_adjusted": modifier_adjusted,
            "heuristic_notes": heuristic_notes,
            "combiner_debug": {
                "combiner_type": "AdaptiveV6Combiner",
                "reliability": reliability,
                "final_score": final_score,
            }
        }
    PARAM_RANGES = {
        'adaptive_mod_strong_weight': {'type': 'float', 'low': 0.5, 'high': 0.85, 'step': 0.005},
        'adaptive_head_strong_weight': {'type': 'float', 'low': 0.15, 'high': 0.45, 'step': 0.005},
        'adaptive_mod_weak_weight': {'type': 'float', 'low': 0.25, 'high': 0.55, 'step': 0.005},
        'adaptive_head_weak_weight': {'type': 'float', 'low': 0.45, 'high': 0.75, 'step': 0.005},
        'adaptive_context_weight': {'type': 'float', 'low': 0.7, 'high': 0.95, 'step': 0.005},
        'adaptive_context_factor': {'type': 'float', 'low': 0.05, 'high': 0.3, 'step': 0.005},
        'adaptive_no_modifier_dampening': {'type': 'float', 'low': 0.4, 'high': 0.75, 'step': 0.005},
        'head_dampening_factor': {'type': 'float', 'low': 0.1, 'high': 0.7, 'step': 0.005},
        'reliability_gamma_v6': {'type': 'float', 'low': 0.8, 'high': 1.6, 'step': 0.005},
        'reliability_floor_v6': {'type': 'float', 'low': 0.15, 'high': 0.5, 'step': 0.005},
    }


class LogisticReliabilityV7Combiner(SentimentCombiner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logistic_scale = kwargs.get('logistic_scale', 3.2)
        self.logistic_shift = kwargs.get('logistic_shift', 0.1)
        self.modifier_bias = kwargs.get('modifier_bias', 0.6)
        self.context_gate = kwargs.get('context_gate', 0.45)
        self.residual_weight = kwargs.get('residual_weight', 0.12)
        self.context_floor = kwargs.get('context_floor', 0.05)

    PARAM_RANGES = {
        'logistic_scale': {'type': 'float', 'low': 1.0, 'high': 6.0, 'step': 0.05},
        'logistic_shift': {'type': 'float', 'low': -0.3, 'high': 0.3, 'step': 0.01},
        'modifier_bias': {'type': 'float', 'low': 0.3, 'high': 0.8, 'step': 0.01},
        'context_gate': {'type': 'float', 'low': 0.1, 'high': 0.8, 'step': 0.01},
        'residual_weight': {'type': 'float', 'low': 0.05, 'high': 0.3, 'step': 0.005},
        'context_floor': {'type': 'float', 'low': 0.0, 'high': 0.2, 'step': 0.005},
        'head_dampening_factor': {'type': 'float', 'low': 0.1, 'high': 0.7, 'step': 0.005},
    }

    def combine(
        self,
        head_text: str,
        head_sentiment: float,
        modifiers: List[str],
        modifier_sentiment: float,
        threshold: tuple,
        context_score: Optional[float],
        per_scores: Dict[str, float],
        clause_text: str,
    ) -> Tuple[float, str, Dict[str, Any]]:
        head_adjusted, head_note = _adjust_head_sentiment(head_text, head_sentiment, modifiers, self.head_dampening_factor)
        modifier_adjusted, modifier_note = _adjust_modifier_sentiment(modifiers, modifier_sentiment, clause_text)

        modifier_strength = abs(modifier_adjusted)
        head_strength = abs(head_adjusted)
        reliability = modifier_strength * self.modifier_bias + head_strength * (1 - self.modifier_bias)
        reliability = max(0.0, min(1.0, reliability))

        logistic_input = self.logistic_scale * (reliability - self.logistic_shift)
        logistic_weight = 1.0 / (1.0 + math.exp(-logistic_input))

        base = logistic_weight * modifier_adjusted + (1.0 - logistic_weight) * head_adjusted
        base += self.residual_weight * (modifier_adjusted - head_adjusted)

        context_gate = self.context_floor
        if context_score is not None:
            context_gate = max(self.context_floor, min(1.0, self.context_gate * abs(context_score) + reliability))
            base = (1.0 - context_gate) * base + context_gate * context_score

        heuristic_notes = [note for note in (head_note, modifier_note) if note]
        heuristic_notes.append(f"reliability={reliability:.2f}")
        heuristic_notes.append(f"logistic_weight={logistic_weight:.2f}")
        if context_score is not None:
            heuristic_notes.append(f"context_gate={context_gate:.2f}")

        justification = (
            "Logistic reliability fusion balances head and modifier sentiment based on confidence."
        )
        justification += " Heuristics: " + "; ".join(heuristic_notes) + "."

        final_score = _clamp_score(base)

        return final_score, justification, {
            "head_adjusted": head_adjusted,
            "modifier_adjusted": modifier_adjusted,
            "heuristic_notes": heuristic_notes,
            "combiner_debug": {
                "combiner_type": "LogisticReliabilityV7Combiner",
                "logistic_weight": logistic_weight,
                "reliability": reliability,
                "context_gate": context_gate,
                "final_score": final_score,
            }
        }


class ContextFusionV8Combiner(SentimentCombiner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.context_primary_weight = kwargs.get('context_primary_weight', 0.45)
        self.context_secondary_weight = kwargs.get('context_secondary_weight', 0.15)
        self.head_gate_threshold = kwargs.get('head_gate_threshold', 0.35)
        self.modifier_gate_threshold = kwargs.get('modifier_gate_threshold', 0.25)
        self.context_decay = kwargs.get('context_decay', 0.4)
        self.context_amplifier = kwargs.get('context_amplifier', 0.65)
        self.context_minimum = kwargs.get('context_minimum', 0.05)

    PARAM_RANGES = {
        'context_primary_weight': {'type': 'float', 'low': 0.2, 'high': 0.8, 'step': 0.01},
        'context_secondary_weight': {'type': 'float', 'low': 0.0, 'high': 0.4, 'step': 0.01},
        'head_gate_threshold': {'type': 'float', 'low': 0.2, 'high': 0.6, 'step': 0.01},
        'modifier_gate_threshold': {'type': 'float', 'low': 0.15, 'high': 0.5, 'step': 0.01},
        'context_decay': {'type': 'float', 'low': 0.1, 'high': 0.7, 'step': 0.01},
        'context_amplifier': {'type': 'float', 'low': 0.3, 'high': 1.0, 'step': 0.01},
        'context_minimum': {'type': 'float', 'low': 0.0, 'high': 0.2, 'step': 0.005},
        'head_dampening_factor': {'type': 'float', 'low': 0.1, 'high': 0.7, 'step': 0.005},
    }

    def combine(
        self,
        head_text: str,
        head_sentiment: float,
        modifiers: List[str],
        modifier_sentiment: float,
        threshold: tuple,
        context_score: Optional[float],
        per_scores: Dict[str, float],
        clause_text: str,
    ) -> Tuple[float, str, Dict[str, Any]]:
        head_adjusted, head_note = _adjust_head_sentiment(head_text, head_sentiment, modifiers, self.head_dampening_factor)
        modifier_adjusted, modifier_note = _adjust_modifier_sentiment(modifiers, modifier_sentiment, clause_text)

        context_component = context_score if context_score is not None else 0.0
        head_strength = abs(head_adjusted)
        modifier_strength = abs(modifier_adjusted)

        head_gate = min(1.0, head_strength / max(self.head_gate_threshold, 1e-6))
        modifier_gate = min(1.0, modifier_strength / max(self.modifier_gate_threshold, 1e-6))
        context_gate = max(
            self.context_minimum,
            min(1.0, self.context_amplifier * modifier_gate + (1 - self.context_decay) * head_gate),
        )

        if modifiers:
            blend_base = 0.6 * modifier_adjusted + 0.4 * head_adjusted
        else:
            blend_base = head_adjusted

        fused = (1 - context_gate) * blend_base + context_gate * context_component * self.context_primary_weight
        fused += self.context_secondary_weight * context_component

        heuristic_notes = [note for note in (head_note, modifier_note) if note]
        heuristic_notes.append(f"context_gate={context_gate:.2f}")
        heuristic_notes.append(f"head_gate={head_gate:.2f}")
        heuristic_notes.append(f"modifier_gate={modifier_gate:.2f}")

        justification = (
            "Context fusion adjusts sentiment using gated head/modifier confidence."
        )
        justification += " Heuristics: " + "; ".join(heuristic_notes) + "."

        final_score = _clamp_score(fused)

        return final_score, justification, {
            "head_adjusted": head_adjusted,
            "modifier_adjusted": modifier_adjusted,
            "heuristic_notes": heuristic_notes,
            "combiner_debug": {
                "combiner_type": "ContextFusionV8Combiner",
                "context_gate": context_gate,
                "head_gate": head_gate,
                "modifier_gate": modifier_gate,
                "final_score": final_score,
            }
        }

COMBINERS = {
    "balanced_v1": BalancedV1Combiner(),
    "modifier_dominant_v2": ModifierDominantV2Combiner(),
    "contextual_v3": ContextualV3Combiner(),
    "head_dominant_v4": HeadDominantV4Combiner(),
    "equal_weight_v5": EqualWeightV5Combiner(),
    "adaptive_v6": AdaptiveV6Combiner(),
    "logistic_v7": LogisticReliabilityV7Combiner(),
    "context_fusion_v8": ContextFusionV8Combiner(),
}
