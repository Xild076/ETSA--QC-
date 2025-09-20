from src.integration_test import run_integrate_full_stack_mode
from src.run_pipeline import run_pipeline_for_text

text = 'The coffee is OUTSTANDING and the service was slow'
print('Testing integrate_graph.py with entity-relation matching...')
result = run_pipeline_for_text(text, mode='efficiency')

print(f'\nEdges created: {result.get("execution_trace", {}).get("modules", {}).get("edge_creation", {}).get("edges_created", 0)}')

for detail in result.get('execution_trace', {}).get('modules', {}).get('edge_creation', {}).get('details', []):
    print(f'Edge: {detail.get("subject")} --[{detail.get("relation_type")}]--> {detail.get("object")} (IDs: {detail.get("subject_id")} -> {detail.get("object_id")})')

print(f'\nFinal sentiments: {result.get("final_sentiments", {})}')
