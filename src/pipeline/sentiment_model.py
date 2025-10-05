"""High-level wrappers around survey-derived sentiment aggregation formulas."""

from __future__ import annotations

from typing import Callable, Iterable, Sequence, Tuple

try:  # pragma: no cover - support package execution
    from ..survey.survey_question_optimizer import (
        get_actor_function,
        get_aggregate_function,
        get_association_function,
        get_child_function,
        get_parent_function,
        get_target_function,
    )
except ImportError:  # pragma: no cover - fallback when executed directly
    from survey.survey_question_optimizer import (  # type: ignore
        get_actor_function,
        get_aggregate_function,
        get_association_function,
        get_child_function,
        get_parent_function,
        get_target_function,
    )

Numeric = float
NumericPair = Tuple[Numeric, Numeric]

__all__ = [
    "SentimentModel",
    "DummySentimentModel",
    "DuoDummySentimentModel",
    "ActionSentimentModel",
    "AssociationSentimentModel",
    "BelongingSentimentModel",
    "AggregateSentimentModel",
]


class SentimentModel:
    """Abstract interface used by the pipeline's compound sentiment stages."""

    def calculate(self, **kwargs) -> Numeric | NumericPair:
        """Compute a sentiment score (or pair of scores) from keyword inputs."""
        raise NotImplementedError


class DummySentimentModel(SentimentModel):
    """Return a simple arithmetic mean of supplied keyword arguments."""

    def calculate(self, **kwargs) -> Numeric:
        return sum(kwargs.values()) / len(kwargs) if kwargs else 0.0


class DuoDummySentimentModel(SentimentModel):
    """Return the same dummy score twice, simulating paired outputs."""

    def calculate(self, **kwargs) -> NumericPair:
        score = sum(kwargs.values()) / len(kwargs) if kwargs else 0.0
        return score, score


class ActionSentimentModel(SentimentModel):
    """Model actor/target sentiment propagation for action relations."""

    def __init__(self, score_key: str = 'user_normalized_sentiment_scores'):
        self.actor_func: Callable[[Numeric, Numeric, Numeric], Numeric] = get_actor_function(score_key)
        self.target_func: Callable[[Numeric, Numeric], Numeric] = get_target_function(score_key)

    def calculate(self, s_actor: Numeric, s_action: Numeric, s_target: Numeric) -> NumericPair:
        return (
            self.actor_func(s_actor, s_action, s_target),
            self.target_func(s_target, s_action),
        )


class AssociationSentimentModel(SentimentModel):
    """Symmetric sentiment update for association relations."""

    def __init__(self, score_key: str = 'user_normalized_sentiment_scores'):
        self.func: Callable[[Numeric, Numeric], Numeric] | None = get_association_function(score_key)

    def calculate(self, s_entity: Numeric, s_other: Numeric, split: bool = False) -> NumericPair:
        if not self.func:
            return 0.0, 0.0
        forward = self.func(s_entity, s_other)
        reverse = self.func(s_other, s_entity)
        if split:
            return forward, reverse
        return forward, reverse


class BelongingSentimentModel(SentimentModel):
    """Directional sentiment propagation for parent/child relations."""

    def __init__(self, score_key: str = 'user_normalized_sentiment_scores'):
        self.parent_func: Callable[[Numeric, Numeric], Numeric] | None = get_parent_function(score_key)
        self.child_func: Callable[[Numeric, Numeric], Numeric] | None = get_child_function(score_key)

    def calculate(self, s_parent: Numeric, s_child: Numeric, split: bool = False) -> NumericPair:
        if not self.parent_func or not self.child_func:
            return 0.0, 0.0
        return (
            self.parent_func(s_parent, s_child),
            self.child_func(s_child, s_parent),
        )


class AggregateSentimentModel(SentimentModel):
    """Collapse per-clause scores into a single aggregate entity sentiment."""

    def __init__(self, score_key: str = 'user_normalized_sentiment_scores'):
        self.func: Callable[[Sequence[Numeric] | Iterable[Numeric]], Numeric] | None = get_aggregate_function(score_key)

    def calculate(self, sent_list: Iterable[Numeric]) -> Numeric:
        if not self.func:
            return 0.0
        return self.func(sent_list)
