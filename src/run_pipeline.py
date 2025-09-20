import os
import sys
import json
import logging
import re
from typing import Dict, Any

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.pipeline.integrate_graph import Pipeline
from src.pipeline.ner_coref import EntityConsolidator
from src.pipeline.clause_splitter import SpacyClauseSplitter
from src.pipeline.re_e import GemmaRelationExtractor, DummyRelationExtractor, SpacyRelationExtractor
from src.pipeline.modifier_e import GemmaModifierExtractor, DummyModifierExtractor, SpacyModifierExtractor
from src.pipeline.sentiment_systems import PresetEnsembleSystem, VADEREntityBaseline, EfficiencyVADERSystem

logger = logging.getLogger(__name__)

def get_pipeline_components(mode: str):
    # Set the API key from the known working key
    api_key = "AIzaSyD3e-8l2yX23DcsB3BaCx6Zy3xBZgqPoQU"
    
    if mode == 'full_stack':
        return GemmaRelationExtractor(api_key=api_key), GemmaModifierExtractor(api_key=api_key), PresetEnsembleSystem()
    elif mode == 'efficiency':
        return SpacyRelationExtractor(), SpacyModifierExtractor(), EfficiencyVADERSystem()
    elif mode == 'no_formulas':
        return GemmaRelationExtractor(api_key=api_key), GemmaModifierExtractor(api_key=api_key), PresetEnsembleSystem()
    elif mode == 'vader_baseline':
        return DummyRelationExtractor(), DummyModifierExtractor(), VADEREntityBaseline()
    elif mode == 'transformer_absa':
        from src.pipeline.sentiment_systems import DeBERTaABSABaseline
        return DummyRelationExtractor(), DummyModifierExtractor(), DeBERTaABSABaseline()
    elif mode == 'ner_basic':
        return DummyRelationExtractor(), DummyModifierExtractor(), PresetEnsembleSystem()
    elif mode == 'no_modifiers':
        return GemmaRelationExtractor(), DummyModifierExtractor(), PresetEnsembleSystem()
    elif mode == 'no_relations':
        return DummyRelationExtractor(), GemmaModifierExtractor(), PresetEnsembleSystem()
    else:
        return GemmaRelationExtractor(), GemmaModifierExtractor(), PresetEnsembleSystem()

def run_pipeline_for_text(text: str, mode: str, debug_run_name: str = None, return_pipeline: bool = False) -> Dict[str, Any]:
    """Main pipeline runner that routes to appropriate baseline or full pipeline."""
    logging.info(f"Initializing pipeline in '{mode}' mode.")
    
    # Route to baseline implementations
    if mode == 'ner_basic':
        return run_ner_basic_baseline(text, PresetEnsembleSystem())
    elif mode == 'vader_baseline':
        return run_vader_baseline(text, VADEREntityBaseline())
    elif mode == 'transformer_absa':
        from src.pipeline.sentiment_systems import DeBERTaABSABaseline
        return run_transformer_baseline(text, DeBERTaABSABaseline())
    else:
        rel_extractor, mod_extractor, sentiment_system = get_pipeline_components(mode)
        
        # Use TransformerNERCorefExtractor for full_stack mode
        if mode == 'full_stack':
            from src.pipeline.ner_coref import TransformerNERCorefExtractor
            entity_consolidator = TransformerNERCorefExtractor()
        else:
            from src.pipeline.ner_coref import EntityConsolidator
            entity_consolidator = EntityConsolidator()
        
        pipeline = Pipeline(
            entity_consolidator=entity_consolidator,
            clause_splitter=SpacyClauseSplitter(),
            relation_extractor=rel_extractor,
            modifier_extractor=mod_extractor,
            sentiment_system=sentiment_system,
            mode=mode
        )
        
        if return_pipeline:
            return pipeline
            
        return pipeline.run(text, debug_run_name=debug_run_name)

def run_ner_basic_baseline(text: str, sentiment_system) -> Dict[str, Any]:
    """Basic NER using simple keyword matching for common aspect categories."""
    # Basic aspect keywords for restaurant domain
    aspect_keywords = {
        'food': ['food', 'meal', 'dish', 'pizza', 'pasta', 'chicken', 'beef', 'salad', 'soup', 'dessert', 'appetizer', 'entree', 'cuisine'],
        'service': ['service', 'staff', 'waiter', 'waitress', 'server', 'waitstaff', 'manager'],
        'ambiance': ['atmosphere', 'ambiance', 'ambience', 'environment', 'decor', 'music', 'lighting'],
        'price': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'value', 'money']
    }
    
    clusters = {}
    final_sentiments = {}
    entity_id = 0
    text_lower = text.lower()
    
    # Apply overall sentiment to detected aspects
    overall_sentiment = sentiment_system.analyze(text)
    
    for aspect_name, keywords in aspect_keywords.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                clusters[entity_id] = {
                    'canonical_name': aspect_name,
                    'entity_references': [[aspect_name, [0, len(text)]]]
                }
                final_sentiments[aspect_name] = overall_sentiment
                entity_id += 1
                break
    
    return {
        "text": text,
        "clauses": [text],
        "clusters": clusters,
        "graph_nodes": [],
        "graph_edges": [],
        "final_sentiments": final_sentiments,
        "execution_trace": {
            "pipeline_mode": "ner_basic",
            "modules": {"ner_basic": {"status": "success", "entities_found": len(clusters)}}
        }
    }

def run_transformer_baseline(text: str, sentiment_system) -> Dict[str, Any]:
    """Transformer ABSA baseline using DeBERTa model."""
    entity_consolidator = EntityConsolidator()
    clusters_dict = entity_consolidator.analyze(text)
    final_sentiments = {}
    
    for canonical_name, data in clusters_dict.items():
        sentiment_score = sentiment_system.analyze(text, canonical_name)
        final_sentiments[canonical_name] = sentiment_score
    
    return {
        "text": text,
        "clauses": [text],
        "clusters": clusters_dict,
        "graph_nodes": [],
        "graph_edges": [],
        "final_sentiments": final_sentiments,
        "execution_trace": {
            "pipeline_mode": "transformer_absa",
            "modules": {"transformer_absa": {"status": "success"}}
        }
    }

def run_vader_baseline(text: str, sentiment_system) -> Dict[str, Any]:
    """VADER baseline that applies overall text sentiment to all detected entities."""
    entity_consolidator = EntityConsolidator()
    clusters_dict = entity_consolidator.analyze(text)
    final_sentiments = {}
    
    # Get overall sentiment and apply to all entities
    overall_sentiment = sentiment_system.analyze(text)
    
    for canonical_name, data in clusters_dict.items():
        final_sentiments[canonical_name] = overall_sentiment
    
    return {
        "text": text,
        "clauses": [text],
        "clusters": clusters_dict,
        "graph_nodes": [],
        "graph_edges": [],
        "final_sentiments": final_sentiments,
        "execution_trace": {
            "pipeline_mode": "vader_baseline",
            "modules": {
                "vader_baseline": {
                    "status": "success", 
                    "entities_analyzed": len(final_sentiments), 
                    "overall_sentiment": overall_sentiment
                }
            }
        }
    }

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    test_text = "The food was delicious, but the service was terribly slow."
    print("Running pipeline for test text")
    
    results = run_pipeline_for_text(test_text, mode='full_stack')
    print("\nResults:")
    print(json.dumps(results, indent=2))
