"""Sentiment and relation extraction pipeline package."""

from .pipeline import SentimentPipeline, build_default_pipeline
from .graph import RelationGraph, GraphVisualizer
from .ner_coref_s import HybridAspectExtractor
from .modifier_e import (
    ModifierExtractor,
    GemmaModifierExtractor,
    SpacyModifierExtractor,
)
from .relation_e import (
    RelationExtractor,
    GemmaRelationExtractor,
    SpacyRelationExtractor,
)
from .clause_s import (
    ClauseSplitter,
    BeneparClauseSplitter,
    NLTKSentenceSplitter,
)
from .sentiment_model import (
    SentimentModel,
    ActionSentimentModel,
    AssociationSentimentModel,
    BelongingSentimentModel,
    AggregateSentimentModel,
    DummySentimentModel,
    DuoDummySentimentModel,
)
from .sentiment_analysis_save import (
    SentimentAnalysis,
    VADERSentimentAnalysis,
    TextBlobSentimentAnalysis,
    FlairSentimentAnalysis,
    PysentimientoSentimentAnalysis,
    DummySentimentAnalysis,
    MultiSentimentAnalysis,
)
from .presets import available_presets, build_pipeline, preset_details
# from .task_system import TaskSystem, initialize_default_task_system

__all__ = [
    "SentimentPipeline",
    "build_default_pipeline",
    "RelationGraph",
    "GraphVisualizer",
    "HybridAspectExtractor",
    "ModifierExtractor",
    "GemmaModifierExtractor",
    "SpacyModifierExtractor",
    "RelationExtractor",
    "GemmaRelationExtractor",
    "SpacyRelationExtractor",
    "ClauseSplitter",
    "BeneparClauseSplitter",
    "NLTKSentenceSplitter",
    "SentimentModel",
    "ActionSentimentModel",
    "AssociationSentimentModel",
    "BelongingSentimentModel",
    "AggregateSentimentModel",
    "DummySentimentModel",
    "DuoDummySentimentModel",
    "SentimentAnalysis",
    "VADERSentimentAnalysis",
    "TextBlobSentimentAnalysis",
    "FlairSentimentAnalysis",
    "PysentimientoSentimentAnalysis",
    "DummySentimentAnalysis",
    "MultiSentimentAnalysis",
    "available_presets",
    "build_pipeline",
    "preset_details",
    # "TaskSystem",
    # "initialize_default_task_system",
]
