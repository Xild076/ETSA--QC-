#!/usr/bin/env python3
"""
In-depth graph-level analysis of wrong_polarity errors
Analyzes actual graph structure, node sentiments, modifiers, and sentiment flow
"""
import json
from pathlib import Path
from collections import Counter, defaultdict
import re

def analyze_graph_report(graph_file):
    """Deep analysis of a single graph report"""
    with open(graph_file, 'r') as f:
        data = json.load(f)
    
    analysis = {
        'file': graph_file.name,
        'text': data.get('text', ''),
        'clauses': data.get('clauses', []),
        'gold_polarity': data.get('gold_aspects', [{}])[0].get('polarity') if data.get('gold_aspects') else None,
        'pred_polarity': data.get('predicted_aspects', [{}])[0].get('polarity') if data.get('predicted_aspects') else None,
        'pred_score': data.get('predicted_aspects', [{}])[0].get('score') if data.get('predicted_aspects') else None,
        'nodes': [],
        'issues': []
    }
    
    # Analyze each node
    for node in data.get('nodes', []):
        node_analysis = {
            'head': node.get('head'),
            'head_sentiment': node.get('head_sentiment'),
            'modifiers': node.get('modifier', []),
            'modifier_sentiment': node.get('modifier_sentiment'),
            'init_sentiment': node.get('init_sentiment'),
            'combiner_strategy': node.get('sentiment_strategy'),
            'justification': node.get('sentiment_justification', ''),
            'heuristics': node.get('sentiment_heuristics', []),
        }
        
        # Check for issues
        if node_analysis['head_sentiment'] is not None and node_analysis['modifier_sentiment'] is not None:
            head_pol = 'pos' if node_analysis['head_sentiment'] > 0.1 else 'neg' if node_analysis['head_sentiment'] < -0.1 else 'neu'
            mod_pol = 'pos' if node_analysis['modifier_sentiment'] > 0.1 else 'neg' if node_analysis['modifier_sentiment'] < -0.1 else 'neu'
            init_pol = 'pos' if node_analysis['init_sentiment'] > 0.1 else 'neg' if node_analysis['init_sentiment'] < -0.1 else 'neu'
            
            # Issue 1: Head and modifier disagree
            if head_pol != mod_pol and head_pol != 'neu' and mod_pol != 'neu':
                node_analysis['issue'] = f'HEAD_MOD_CONFLICT: head={head_pol}, mod={mod_pol}, result={init_pol}'
            
            # Issue 2: Combiner flips polarity
            if (node_analysis['head_sentiment'] > 0.2 and node_analysis['init_sentiment'] < -0.1) or \
               (node_analysis['head_sentiment'] < -0.2 and node_analysis['init_sentiment'] > 0.1):
                node_analysis['issue'] = f'COMBINER_FLIP: head={node_analysis["head_sentiment"]:.3f} -> init={node_analysis["init_sentiment"]:.3f}'
            
            # Issue 3: Weak modifiers dominating
            if node_analysis['modifiers'] and abs(node_analysis['modifier_sentiment']) < 0.2 and \
               abs(node_analysis['head_sentiment']) > 0.3 and \
               abs(node_analysis['init_sentiment'] - node_analysis['modifier_sentiment']) < 0.1:
                node_analysis['issue'] = f'WEAK_MOD_DOMINATES: mod={node_analysis["modifier_sentiment"]:.3f} overrides head={node_analysis["head_sentiment"]:.3f}'
        
        analysis['nodes'].append(node_analysis)
    
    # Detect negation
    negation_words = ['not', 'no', 'never', "n't", 'nothing', 'neither', 'nor', 'nobody', 'none', 'nowhere', "don't", "didn't", "doesn't", "won't", "wasn't", "weren't", "haven't", "hasn't", "hadn't"]
    has_negation = any(word in analysis['text'].lower() for word in negation_words)
    analysis['has_negation'] = has_negation
    
    # Check if negation was detected in heuristics
    negation_detected_in_pipeline = any('negation' in ' '.join(node.get('heuristics', [])).lower() for node in analysis['nodes'])
    if has_negation and not negation_detected_in_pipeline:
        analysis['issues'].append('NEGATION_NOT_DETECTED')
    
    # Check for sarcasm/irony indicators
    sarcasm_patterns = [
        r'which i think is ridiculous',
        r'which is ridiculous',
        r'which is silly',
        r'which is absurd',
        r'but actually',
        r'but in reality',
        r'although.*complain.*ridiculous'
    ]
    for pattern in sarcasm_patterns:
        if re.search(pattern, analysis['text'].lower()):
            analysis['issues'].append(f'SARCASM_PATTERN: {pattern}')
            break
    
    # Check for contrastive markers
    contrastive = ['although', 'however', 'but', 'yet', 'despite', 'even though', 'whereas', 'while']
    if any(word in analysis['text'].lower() for word in contrastive):
        analysis['has_contrastive'] = True
    
    return analysis

def print_detailed_analysis(analysis):
    """Pretty print a graph analysis"""
    print(f"\n{'='*100}")
    print(f"File: {analysis['file']}")
    print(f"{'='*100}")
    print(f"Text: {analysis['text']}")
    print(f"\nGold: {analysis['gold_polarity']}")
    if analysis['pred_score'] is not None:
        print(f"Pred: {analysis['pred_polarity']} (score: {analysis['pred_score']:+.3f})")
    else:
        print(f"Pred: {analysis['pred_polarity']} (score: None)")
    print(f"\nNegation present: {analysis.get('has_negation', False)}")
    print(f"Contrastive: {analysis.get('has_contrastive', False)}")
    
    if analysis['issues']:
        print(f"\n⚠️  ISSUES DETECTED:")
        for issue in analysis['issues']:
            print(f"    - {issue}")
    
    print(f"\nNodes: {len(analysis['nodes'])}")
    for i, node in enumerate(analysis['nodes'], 1):
        print(f"\n  Node {i}: {node['head']}")
        print(f"    Head sentiment: {node['head_sentiment']:+.3f}" if node['head_sentiment'] is not None else "    Head sentiment: None")
        if node['modifiers']:
            print(f"    Modifiers: {node['modifiers']}")
            print(f"    Modifier sentiment: {node['modifier_sentiment']:+.3f}" if node['modifier_sentiment'] is not None else "    Modifier sentiment: None")
        print(f"    Init sentiment: {node['init_sentiment']:+.3f}" if node['init_sentiment'] is not None else "    Init sentiment: None")
        print(f"    Strategy: {node['combiner_strategy']}")
        if node.get('heuristics'):
            print(f"    Heuristics: {'; '.join(node['heuristics'])}")
        if 'issue' in node:
            print(f"    ⚠️  NODE ISSUE: {node['issue']}")

# Analyze laptop worst errors
laptop_dir = Path("output/benchmarks/newest_test_test_laptop_2014_test_laptop_2014_20251001_124408/graph_reports/wrong_polarity")

print("="*100)
print("IN-DEPTH GRAPH ANALYSIS: LAPTOP DATASET WORST WRONG_POLARITY ERRORS")
print("="*100)

# Get worst errors from our previous analysis
worst_cases = [
    "00045_1063_373_wrong_polarity.json",  # "Screen - although some people might complain about low res..."
    "00152_1074_1_wrong_polarity.json",     # Likely another bad one
    "00159_29_186_wrong_polarity.json",     # "Came with iPhoto..."
    "00161_768_1_wrong_polarity.json",      # TBD
    "00198_258_1_wrong_polarity.json",      # TBD
]

for case_file in worst_cases:
    graph_file = laptop_dir / case_file
    if graph_file.exists():
        analysis = analyze_graph_report(graph_file)
        print_detailed_analysis(analysis)
    else:
        print(f"\n❌ File not found: {case_file}")

# Summary statistics
print(f"\n\n{'='*100}")
print("SUMMARY ANALYSIS ACROSS ALL LAPTOP WRONG_POLARITY ERRORS")
print(f"{'='*100}")

all_analyses = []
for graph_file in sorted(laptop_dir.glob("*.json"))[:50]:  # Analyze first 50
    analysis = analyze_graph_report(graph_file)
    all_analyses.append(analysis)

# Count issues
issue_counts = Counter()
for a in all_analyses:
    for issue in a['issues']:
        issue_type = issue.split(':')[0]
        issue_counts[issue_type] += 1
    for node in a['nodes']:
        if 'issue' in node:
            issue_type = node['issue'].split(':')[0]
            issue_counts[issue_type] += 1

print(f"\nIssue frequency (n={len(all_analyses)} errors analyzed):")
for issue, count in issue_counts.most_common():
    print(f"  {issue:30s}: {count:3d} ({count/len(all_analyses)*100:5.1f}%)")

# Combiner strategy usage
combiner_counts = Counter()
for a in all_analyses:
    for node in a['nodes']:
        if node['combiner_strategy']:
            combiner_counts[node['combiner_strategy']] += 1

print(f"\nCombiner strategy distribution:")
for strategy, count in combiner_counts.most_common():
    print(f"  {strategy:30s}: {count:4d}")

# Negation detection rate
negation_present = sum(1 for a in all_analyses if a.get('has_negation', False))
negation_detected = sum(1 for a in all_analyses if a.get('has_negation', False) and 'NEGATION_NOT_DETECTED' not in a['issues'])

print(f"\nNegation handling:")
print(f"  Cases with negation: {negation_present} / {len(all_analyses)} ({negation_present/len(all_analyses)*100:.1f}%)")
print(f"  Negation detected by pipeline: {negation_detected} / {negation_present} ({negation_detected/negation_present*100:.1f}%)" if negation_present > 0 else "  No negation cases")

# Contrastive markers
contrastive_present = sum(1 for a in all_analyses if a.get('has_contrastive', False))
print(f"  Cases with contrastive markers: {contrastive_present} / {len(all_analyses)} ({contrastive_present/len(all_analyses)*100:.1f}%)")

print("\n" + "="*100)
print("RECOMMENDATIONS BASED ON GRAPH ANALYSIS")
print("="*100)
print("""
1. Fix combiner polarity flips: Head sentiment being overridden by weak modifiers
2. Improve negation detection: System missing negation in text
3. Handle sarcasm/irony: "which I think is ridiculous" patterns
4. Better contrastive handling: "although", "however", "but" patterns
5. Modifier quality assessment: Weak modifiers shouldn't dominate strong heads
""")
