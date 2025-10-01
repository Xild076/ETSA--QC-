#!/usr/bin/env python3
import sys
import os
import types
import json
from typing import Dict, List, Any

if "sentiment" not in sys.modules:
    sentiment_pkg = types.ModuleType("sentiment")
    sentiment_pkg.__path__ = []
    sentiment_pkg.__spec__ = types.SimpleNamespace(submodule_search_locations=[])
    sys.modules["sentiment"] = sentiment_pkg

if "src" not in sys.modules:
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = []
    src_pkg.__spec__ = types.SimpleNamespace(submodule_search_locations=[])
    sys.modules["src"] = src_pkg

if "src.sentiment" not in sys.modules:
    src_sentiment_pkg = types.ModuleType("src.sentiment")
    src_sentiment_pkg.__path__ = []
    src_sentiment_pkg.__spec__ = types.SimpleNamespace(submodule_search_locations=[])
    sys.modules["src.sentiment"] = src_sentiment_pkg

if "sentiment.sentiment" not in sys.modules:
    stub_module = types.ModuleType("sentiment.sentiment")

    def _stub_result(score: float = 0.0):
        return {"score": score, "label": "neutral", "confidence": abs(score), "raw": {"stub": True}}

    for name in [
        "get_vader_sentiment",
        "get_textblob_sentiment",
        "get_flair_sentiment",
        "get_pysentimiento_sentiment",
        "get_swn_sentiment",
        "get_nlptown_sentiment",
        "get_finiteautomata_sentiment",
        "get_ProsusAI_sentiment",
        "get_distilbert_logit_sentiment",
    ]:
        setattr(stub_module, name, lambda text, _score=0.0: _stub_result(_score))

    sys.modules["sentiment.sentiment"] = stub_module
    sys.modules["src.sentiment.sentiment"] = stub_module

sys.path.append('/Users/harry/Documents/Python_Projects/ETSA_(QC)/src')

from sentiment.sentiment import get_distilbert_logit_sentiment, get_vader_sentiment, get_textblob_sentiment
from pipeline.sentiment_analysis import MultiSentimentAnalysis
from pipeline.clause_s import ClauseSplitter
from pipeline.modifier_e import ModifierExtractor
from pipeline.relation_e import RelationExtractor
from pipeline.graph import RelationGraph
from pipeline.combiners import get_combiner
from pipeline.pipeline import SentimentPipeline

TEST_CASES = [
    {
        "text": "the weight is acceptable",
        "expected_polarity": "positive",
        "expected_score_range": (0.1, 1.0),
        "issue": "acceptable_word_negative",
        "priority": "critical"
    },
    {
        "text": "Did not enjoy the new Windows 8 and touchscreen functions",
        "expected_polarity": "negative",
        "expected_score_range": (-1.0, -0.1),
        "issue": "negation_not_handled",
        "priority": "critical"
    },
    {
        "text": "No installation disk (DVD) is included",
        "expected_polarity": "negative",
        "expected_score_range": (-1.0, -0.1),
        "issue": "contextual_negation",
        "priority": "critical"
    },
    {
        "text": "Its size is ideal and the weight is acceptable",
        "expected_polarity": "positive",
        "expected_score_range": (0.1, 1.0),
        "issue": "mixed_sentiment_compound",
        "priority": "high"
    },
    {
        "text": "This product is not bad",
        "expected_polarity": "positive",
        "expected_score_range": (0.0, 1.0),
        "issue": "double_negation",
        "priority": "medium"
    },
    {
        "text": "I hate this laptop",
        "expected_polarity": "negative",
        "expected_score_range": (-1.0, -0.1),
        "issue": "strong_negative",
        "priority": "high"
    },
    {
        "text": "This is an excellent device",
        "expected_polarity": "positive",
        "expected_score_range": (0.1, 1.0),
        "issue": "strong_positive",
        "priority": "high"
    }
]

def extract_score(valence) -> float:
    if isinstance(valence, (int, float)):
        return float(valence)
    if isinstance(valence, dict):
        for key in ('score', 'compound', 'aggregate', 'polarity', 'value'):
            value = valence.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        for nested in valence.values():
            if isinstance(nested, (dict, int, float)):
                nested_score = extract_score(nested)
                if nested_score is not None:
                    return nested_score
    return 0.0

def check_polarity(score: float, expected: str, score_range: tuple) -> Dict[str, Any]:
    min_score, max_score = score_range
    in_range = min_score <= score <= max_score
    
    if expected == "positive":
        correct = score > 0.1
    elif expected == "negative":
        correct = score < -0.1
    else:
        correct = abs(score) <= 0.1
    
    return {
        "correct": correct,
        "in_range": in_range,
        "score": score
    }

def test_base_sentiment_methods():
    print("="*80)
    print("TESTING BASE SENTIMENT METHODS")
    print("="*80)
    
    results = []
    
    for case in TEST_CASES:
        text = case["text"]
        expected = case["expected_polarity"]
        score_range = case["expected_score_range"]
        issue = case["issue"]
        priority = case["priority"]
        
        print(f"\n[{priority.upper()}] {issue}")
        print(f"Text: '{text}'")
        print(f"Expected: {expected} (range: {score_range})")
        
        distilbert_raw = get_distilbert_logit_sentiment(text)
        vader_raw = get_vader_sentiment(text)
        textblob_raw = get_textblob_sentiment(text)
        
        distilbert_score = extract_score(distilbert_raw)
        vader_score = extract_score(vader_raw)
        textblob_score = extract_score(textblob_raw)
        
        distilbert_check = check_polarity(distilbert_score, expected, score_range)
        vader_check = check_polarity(vader_score, expected, score_range)
        textblob_check = check_polarity(textblob_score, expected, score_range)
        
        print(f"  DistilBERT: {distilbert_score:+.3f} {'✓' if distilbert_check['correct'] else '✗'}")
        print(f"  VADER:      {vader_score:+.3f} {'✓' if vader_check['correct'] else '✗'}")
        print(f"  TextBlob:   {textblob_score:+.3f} {'✓' if textblob_check['correct'] else '✗'}")
        
        results.append({
            "text": text,
            "issue": issue,
            "priority": priority,
            "expected": expected,
            "methods": {
                "distilbert": distilbert_check,
                "vader": vader_check,
                "textblob": textblob_check
            }
        })
    
    return results

def test_multi_sentiment_aggregation(base_results):
    print("\n" + "="*80)
    print("TESTING MULTI-SENTIMENT AGGREGATION")
    print("="*80)
    
    analyzer = MultiSentimentAnalysis(methods=['vader', 'textblob', 'distilbert_logit'])
    results = []
    
    for i, case in enumerate(TEST_CASES):
        text = case["text"]
        expected = case["expected_polarity"]
        score_range = case["expected_score_range"]
        issue = case["issue"]
        
        print(f"\n[{case['priority'].upper()}] {issue}")
        
        ensemble_result = analyzer.analyze(text)
        ensemble_score = extract_score(ensemble_result)
        ensemble_check = check_polarity(ensemble_score, expected, score_range)
        
        base_method_results = base_results[i]["methods"]
        
        print(f"  Base methods correct: D={base_method_results['distilbert']['correct']}, V={base_method_results['vader']['correct']}, T={base_method_results['textblob']['correct']}")
        print(f"  Ensemble: {ensemble_score:+.3f} {'✓' if ensemble_check['correct'] else '✗'}")
        
        results.append({
            "text": text,
            "issue": issue,
            "ensemble": ensemble_check,
            "base_correct_count": sum(1 for m in base_method_results.values() if m['correct'])
        })
    
    return results

def test_clause_splitting():
    print("\n" + "="*80)
    print("TESTING CLAUSE SPLITTING")
    print("="*80)
    
    splitter = ClauseSplitter()
    results = []
    
    for case in TEST_CASES:
        text = case["text"]
        issue = case["issue"]
        
        clauses = splitter.split(text)
        print(f"\n{issue}: {len(clauses)} clause(s)")
        for i, clause in enumerate(clauses):
            print(f"  [{i}] '{clause}'")
        
        results.append({
            "text": text,
            "issue": issue,
            "clause_count": len(clauses),
            "clauses": clauses
        })
    
    return results

def test_modifier_extraction():
    print("\n" + "="*80)
    print("TESTING MODIFIER EXTRACTION")
    print("="*80)
    
    try:
        modifier_extractor = ModifierExtractor()
        results = []
        
        for case in TEST_CASES:
            text = case["text"]
            issue = case["issue"]
            
            modifiers = modifier_extractor.extract_modifiers(text)
            print(f"\n{issue}:")
            print(f"  Modifiers found: {len(modifiers)}")
            if modifiers:
                for mod in modifiers[:3]:
                    print(f"    - {mod}")
            
            results.append({
                "text": text,
                "issue": issue,
                "modifier_count": len(modifiers),
                "modifiers": modifiers[:5]
            })
        
        return results
    except Exception as e:
        print(f"  ERROR: {e}")
        return []

def test_relation_extraction():
    print("\n" + "="*80)
    print("TESTING RELATION EXTRACTION")
    print("="*80)
    
    try:
        relation_extractor = RelationExtractor()
        results = []
        
        for case in TEST_CASES:
            text = case["text"]
            issue = case["issue"]
            
            relations = relation_extractor.extract_relations(text)
            print(f"\n{issue}:")
            print(f"  Relations found: {len(relations) if relations else 0}")
            
            results.append({
                "text": text,
                "issue": issue,
                "relation_count": len(relations) if relations else 0
            })
        
        return results
    except Exception as e:
        print(f"  ERROR: {e}")
        return []

def test_graph_building():
    print("\n" + "="*80)
    print("TESTING GRAPH BUILDING")
    print("="*80)
    
    try:
        from pipeline.sentiment_analysis import DummySentimentAnalysis
        results = []
        
        for case in TEST_CASES:
            text = case["text"]
            issue = case["issue"]
            
            graph = RelationGraph(sentiment_analyzer_system=DummySentimentAnalysis())
            graph.add_entity_node(1, "entity", [], "entity", 0)
            
            node_count = len(graph.graph.nodes)
            edge_count = len(graph.graph.edges)
            
            print(f"\n{issue}: {node_count} nodes, {edge_count} edges")
            
            results.append({
                "text": text,
                "issue": issue,
                "node_count": node_count,
                "edge_count": edge_count
            })
        
        return results
    except Exception as e:
        print(f"  ERROR: {e}")
        return []

def test_combiners():
    print("\n" + "="*80)
    print("TESTING COMBINER FUNCTIONS")
    print("="*80)
    
    combiner_names = ["contextual_v3", "aspect_focused", "global_aggregate"]
    results = []
    
    for combiner_name in combiner_names:
        try:
            combiner = get_combiner(combiner_name)
            print(f"\n{combiner_name}: {combiner.__class__.__name__} {'✓' if combiner else '✗'}")
            results.append({
                "combiner": combiner_name,
                "loaded": combiner is not None
            })
        except Exception as e:
            print(f"\n{combiner_name}: ERROR - {e}")
            results.append({
                "combiner": combiner_name,
                "loaded": False,
                "error": str(e)
            })
    
    return results

def test_full_pipeline():
    print("\n" + "="*80)
    print("TESTING FULL PIPELINE")
    print("="*80)
    
    try:
        pipeline = SentimentPipeline()
        results = []
        
        for case in TEST_CASES:
            text = case["text"]
            expected = case["expected_polarity"]
            score_range = case["expected_score_range"]
            issue = case["issue"]
            priority = case["priority"]
            
            print(f"\n[{priority.upper()}] {issue}")
            
            try:
                result = pipeline.process(text)
                
                final_score = result.get("final_sentiment", 0.0)
                final_check = check_polarity(final_score, expected, score_range)
                
                entity_sentiments = result.get("entity_sentiments", {})
                
                print(f"  Final sentiment: {final_score:+.3f} {'✓' if final_check['correct'] else '✗'}")
                print(f"  Entity sentiments: {len(entity_sentiments)}")
                
                results.append({
                    "text": text,
                    "issue": issue,
                    "priority": priority,
                    "expected": expected,
                    "final": final_check,
                    "entity_count": len(entity_sentiments)
                })
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "text": text,
                    "issue": issue,
                    "error": str(e)
                })
        
        return results
    except Exception as e:
        print(f"  PIPELINE INIT ERROR: {e}")
        return []

def generate_summary(all_results):
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    base_results = all_results["base_sentiment"]
    ensemble_results = all_results["ensemble"]
    pipeline_results = all_results["pipeline"]
    
    print("\n--- BASE SENTIMENT METHODS ---")
    for priority in ["critical", "high", "medium"]:
        priority_cases = [r for r in base_results if r["priority"] == priority]
        if not priority_cases:
            continue
        
        print(f"\n{priority.upper()} issues ({len(priority_cases)}):")
        for r in priority_cases:
            methods = r["methods"]
            d_pass = "✓" if methods["distilbert"]["correct"] else "✗"
            v_pass = "✓" if methods["vader"]["correct"] else "✗"
            t_pass = "✓" if methods["textblob"]["correct"] else "✗"
            print(f"  {r['issue']}: D={d_pass} V={v_pass} T={t_pass}")
    
    print("\n--- ENSEMBLE AGGREGATION ---")
    for priority in ["critical", "high", "medium"]:
        priority_cases = [r for r in ensemble_results if TEST_CASES[[c["issue"] for c in TEST_CASES].index(r["issue"])]["priority"] == priority]
        if not priority_cases:
            continue
        
        print(f"\n{priority.upper()} issues ({len(priority_cases)}):")
        for r in priority_cases:
            ensemble_pass = "✓" if r["ensemble"]["correct"] else "✗"
            print(f"  {r['issue']}: {ensemble_pass} (base correct: {r['base_correct_count']}/3)")
    
    print("\n--- FULL PIPELINE ---")
    for priority in ["critical", "high", "medium"]:
        priority_cases = [r for r in pipeline_results if r.get("priority") == priority]
        if not priority_cases:
            continue
        
        print(f"\n{priority.upper()} issues ({len(priority_cases)}):")
        for r in priority_cases:
            if "error" in r:
                print(f"  {r['issue']}: ERROR - {r['error']}")
            else:
                pipeline_pass = "✓" if r["final"]["correct"] else "✗"
                print(f"  {r['issue']}: {pipeline_pass} (score: {r['final']['score']:+.3f})")
    
    print("\n--- FAILURE ANALYSIS ---")
    critical_base_failures = []
    critical_ensemble_failures = []
    critical_pipeline_failures = []
    
    for i, case in enumerate(TEST_CASES):
        if case["priority"] != "critical":
            continue
        
        base_r = base_results[i]
        ensemble_r = ensemble_results[i]
        pipeline_r = pipeline_results[i] if i < len(pipeline_results) else None
        
        base_all_fail = not any(m["correct"] for m in base_r["methods"].values())
        ensemble_fail = not ensemble_r["ensemble"]["correct"]
        pipeline_fail = pipeline_r and not pipeline_r.get("final", {}).get("correct", False)
        
        if base_all_fail:
            critical_base_failures.append(case["issue"])
        if ensemble_fail:
            critical_ensemble_failures.append(case["issue"])
        if pipeline_fail:
            critical_pipeline_failures.append(case["issue"])
    
    print(f"\nCritical failures at BASE level: {len(critical_base_failures)}")
    for issue in critical_base_failures:
        print(f"  - {issue}")
    
    print(f"\nCritical failures at ENSEMBLE level: {len(critical_ensemble_failures)}")
    for issue in critical_ensemble_failures:
        print(f"  - {issue}")
    
    print(f"\nCritical failures at PIPELINE level: {len(critical_pipeline_failures)}")
    for issue in critical_pipeline_failures:
        print(f"  - {issue}")
    
    return {
        "critical_base_failures": critical_base_failures,
        "critical_ensemble_failures": critical_ensemble_failures,
        "critical_pipeline_failures": critical_pipeline_failures
    }

def main():
    print("Starting Polarity Diagnostic Benchmark")
    print("="*80 + "\n")
    
    all_results = {}
    
    all_results["base_sentiment"] = test_base_sentiment_methods()
    all_results["ensemble"] = test_multi_sentiment_aggregation(all_results["base_sentiment"])
    all_results["clause_splitting"] = test_clause_splitting()
    all_results["modifier_extraction"] = test_modifier_extraction()
    all_results["relation_extraction"] = test_relation_extraction()
    all_results["graph_building"] = test_graph_building()
    all_results["combiners"] = test_combiners()
    all_results["pipeline"] = test_full_pipeline()
    
    failure_summary = generate_summary(all_results)
    
    output_file = '/Users/harry/Documents/Python_Projects/ETSA_(QC)/polarity_diagnostic_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n\nFull results saved to: {output_file}")
    
    total_critical_failures = (
        len(failure_summary["critical_base_failures"]) +
        len(failure_summary["critical_ensemble_failures"]) +
        len(failure_summary["critical_pipeline_failures"])
    )
    
    if total_critical_failures == 0:
        print("\n✓ No critical failures detected!")
        return 0
    else:
        print(f"\n✗ {total_critical_failures} critical failure(s) detected across pipeline stages")
        return 1

if __name__ == "__main__":
    sys.exit(main())
