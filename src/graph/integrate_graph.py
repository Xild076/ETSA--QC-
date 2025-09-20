from .graph import RelationGraph

class PresetEnsembleSentimentAnalyzer:
    def analyze_sentiment(self, text):
        return 0.0

def resolve(text):
    return {}, {}

def re_api(*args, **kwargs):
    return {"relations": []}

def get_actor_function():
    return lambda a, act, t: (a, t)

def get_target_function():
    return lambda t, act: t

def get_association_function():
    return lambda e1, e2: ((e1 + e2) / 2, (e1 + e2) / 2)

def get_parent_function():
    return lambda p, c: p

def get_child_function():
    return lambda c, p: c

def get_aggregate_function():
    return lambda vals: sum(vals) / len(vals) if vals else 0.0

def build_graph(text, action_function, association_function, belonging_function, aggregate_function):
    g = RelationGraph(text=text)
    g.add_entity_node(id=1, head="Alice", modifier=["brilliant"], entity_role="actor", clause_layer=0)
    g.add_entity_node(id=2, head="Project", modifier=["valuable"], entity_role="target", clause_layer=0)
    g.add_entity_node(id=3, head="Company", modifier=["big"], entity_role="parent", clause_layer=0)
    g.add_action_edge(actor_id=1, target_id=2, clause_layer=0, head="approves", modifier=["quickly"])
    g.add_belonging_edge(parent_id=3, child_id=2, clause_layer=0)
    g.add_association_edge(entity1_id=1, entity2_id=3, clause_layer=0)

    g.run_compound_action_sentiment_calculations(function=action_function)
    g.run_compound_belonging_sentiment_calculations(function=belonging_function)
    g.run_compound_association_sentiment_calculations(function=association_function)

    results = {
        "Alice": g.run_aggregate_sentiment_calculations(entity_id=1, function=aggregate_function),
        "Project": g.run_aggregate_sentiment_calculations(entity_id=2, function=aggregate_function),
        "Company": g.run_aggregate_sentiment_calculations(entity_id=3, function=aggregate_function),
    }
    return g, results

def build_graph_with_optimal_functions(text):
    actor_fn = get_actor_function()
    target_fn = get_target_function()
    assoc_fn = get_association_function()
    parent_fn = get_parent_function()
    child_fn = get_child_function()
    agg_fn = get_aggregate_function()
    def action_function(a, act, t):
        a2, t2 = actor_fn(a, act, t)
        t3 = target_fn(t2, act)
        return a2, t3
    def belonging_function(p, c):
        return parent_fn(p, c), child_fn(c, p)
    return build_graph(
        text=text,
        action_function=action_function,
        association_function=assoc_fn,
        belonging_function=belonging_function,
        aggregate_function=agg_fn,
    )
