"""Sentiment and relation extraction pipeline package."""

try:
    from .cache_manager import PipelineCache
    from .clause_s import (
        ClauseSplitter,
        BeneparClauseSplitter,
        NLTKSentenceSplitter,
        make_clause_splitter,
    )
    from .graph import GraphVisualizer, RelationGraph
    from .modifier_e import GemmaModifierExtractor, ModifierExtractor, SpacyModifierExtractor
    from .ner_coref_s import HybridAspectExtractor
    from .pipeline import SentimentPipeline, build_default_pipeline
    from .relation_e import GemmaRelationExtractor, RelationExtractor, SpacyRelationExtractor
    from .sentiment_analysis import (
        DummySentimentAnalysis,
        FlairSentimentAnalysis,
        MultiSentimentAnalysis,
        PysentimientoSentimentAnalysis,
        SentimentAnalysis,
        TextBlobSentimentAnalysis,
        VADERSentimentAnalysis,
    )
    from .sentiment_model import (
        ActionSentimentModel,
        AggregateSentimentModel,
        AssociationSentimentModel,
        BelongingSentimentModel,
        DummySentimentModel,
        DuoDummySentimentModel,
        SentimentModel,
    )
except ImportError:
    from cache_manager import PipelineCache
    from clause_s import (
        ClauseSplitter,
        BeneparClauseSplitter,
        NLTKSentenceSplitter,
        make_clause_splitter,
    )
    from graph import GraphVisualizer, RelationGraph
    from modifier_e import GemmaModifierExtractor, ModifierExtractor, SpacyModifierExtractor
    from ner_coref_s import HybridAspectExtractor
    from pipeline import SentimentPipeline, build_default_pipeline
    from relation_e import GemmaRelationExtractor, RelationExtractor, SpacyRelationExtractor
    from sentiment_analysis import (
        DummySentimentAnalysis,
        FlairSentimentAnalysis,
        MultiSentimentAnalysis,
        PysentimientoSentimentAnalysis,
        SentimentAnalysis,
        TextBlobSentimentAnalysis,
        VADERSentimentAnalysis,
    )
    from sentiment_model import (
        ActionSentimentModel,
        AggregateSentimentModel,
        AssociationSentimentModel,
        BelongingSentimentModel,
        DummySentimentModel,
        DuoDummySentimentModel,
        SentimentModel,
    )

__all__ = [
    "SentimentPipeline",
    "build_default_pipeline",
    "PipelineCache",
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
    "make_clause_splitter",
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
]
