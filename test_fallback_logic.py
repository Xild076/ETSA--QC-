"""
Test to verify fallback sentiment logic
"""
import sys
sys.path.append('src')
from pipeline import build_default_pipeline

# Test case where aspect might not be detected
test_sentences = [
    "This place has great ambiance.",  # Gold: "ambiance"
    "The location is convenient.", # Gold: "location"
    "Prices are reasonable.", # Gold: "Prices"
]

print("Testing fallback logic...")
pipeline = build_default_pipeline()

for text in test_sentences:
    print(f"\n{'='*60}")
    print(f"Text: {text}")
    result = pipeline.process(text)
    
    predicted = list(result['aggregate_results'].items())
    print(f"\nPredicted aspects ({len(predicted)}):")
    for entity_id, data in predicted:
        label = data.get('label', f'entity_{entity_id}')
        sentiment = data.get('aggregate_sentiment', 0.0)
        print(f"  - {label}: {sentiment:+.3f}")
    
    # Compute overall sentiment
    graph = result.get("graph")
    if graph and hasattr(graph, "compute_text_sentiment"):
        overall = graph.compute_text_sentiment(text)
        print(f"\nOverall sentence sentiment: {overall:+.3f}")
