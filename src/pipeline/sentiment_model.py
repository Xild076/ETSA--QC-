import sys
from pathlib import Path
_base = Path(__file__).resolve()
_src = _base.parents[1]
_root = _src.parent
for p in (str(_src), str(_root)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.survey.survey_question_optimizer import (
        get_actor_function,
        get_target_function,
        get_association_function,
        get_parent_function,
        get_child_function,
        get_aggregate_function,
    )
except ImportError: 
    from survey.survey_question_optimizer import (
        get_actor_function,
        get_target_function,
        get_association_function,
        get_parent_function,
        get_child_function,
        get_aggregate_function,
    )

class SentimentModel:
    def calculate(self, **kwargs) -> float:
        raise NotImplementedError

class DummySentimentModel(SentimentModel):
    def calculate(self, **kwargs) -> float:
        return sum(kwargs.values()) / len(kwargs) if kwargs else 0.0

class DuoDummySentimentModel(SentimentModel):
    def calculate(self, **kwargs) -> float:
        return (sum(kwargs.values()) / len(kwargs) if kwargs else 0.0, 
                sum(kwargs.values()) / len(kwargs) if kwargs else 0.0)

class ActionSentimentModel(SentimentModel):
    def __init__(self, score_key: str = 'user_normalized_sentiment_scores'):
        self.actor_func = get_actor_function(score_key)
        self.target_func = get_target_function(score_key)

    def calculate(self, s_actor: float, s_action: float, s_target: float) -> float:
        return self.actor_func(s_actor, s_action, s_target), self.target_func(s_target, s_action)

class AssociationSentimentModel(SentimentModel):
    def __init__(self, score_key: str = 'user_normalized_sentiment_scores'):
        self.func = get_association_function(score_key)

    def calculate(self, s_entity: float, s_other: float, split: bool = False) -> float:
        if not self.func:
            return 0.0
        return self.func(s_entity, s_other), self.func(s_other, s_entity) if split else self.func(s_entity, s_other)

class BelongingSentimentModel(SentimentModel):
    def __init__(self, score_key: str = 'user_normalized_sentiment_scores'):
        self.parent_func = get_parent_function(score_key)
        self.child_func = get_child_function(score_key)

    def calculate(self, s_parent: float, s_child: float, split: bool = False) -> float:
        if not self.parent_func or not self.child_func:
            return 0.0
        if split:
            return self.parent_func(s_parent, s_child), self.child_func(s_child, s_parent)
        # By default return a symmetric aggregation (parent perspective)
        return self.parent_func(s_parent, s_child)

class AggregateSentimentModel(SentimentModel):
    def __init__(self, score_key: str = 'user_normalized_sentiment_scores'):
        self.func = get_aggregate_function(score_key)

    def calculate(self, sent_list) -> float:
        if not self.func:
            return 0.0
        return self.func(sent_list)
