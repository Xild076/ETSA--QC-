from src.run_pipeline import run_pipeline_for_text

text = 'The coffee is OUTSTANDING and the service was slow'
result = run_pipeline_for_text(text, mode='efficiency')
print('Pipeline executed successfully!')
print(f'Final sentiments: {result.get("final_sentiments", {})}')
print(f'Graph nodes: {len(result.get("graph_nodes", []))}')
print(f'Graph edges: {len(result.get("graph_edges", []))}')
