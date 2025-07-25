from typing import Callable, List
from enum import Enum
import networkx as nx
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import plotly.graph_objects as go

class SentimentAnalyzerSystem:
    def __init__(self):
        pass

    def analyze_sentiment(self, text):
        raise NotImplementedError("This method should be overridden by subclasses")

class VADERSentimentAnalyzer(SentimentAnalyzerSystem):
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text):
        return self.analyzer.polarity_scores(text)['compound'] if text else 0.0

class AFINNSentimentAnalyzer(SentimentAnalyzerSystem):
    def __init__(self):
        import afinn
        self.afinn = afinn.Afinn()

    def analyze_sentiment(self, text):
        return self.afinn.score(text) / 5.0 if text else 0.0

class SpacyENCOREWEBSMSentimentAnalyzer(SentimentAnalyzerSystem):
    def __init__(self):
        import spacy
        self.nlp = spacy.load("en_core_web_sm")

    def analyze_sentiment(self, text):
        doc = self.nlp(text)
        return sum(token.sentiment for token in doc) / len(doc) if doc else 0.0

class HFPipelineSentimentAnalyzer(SentimentAnalyzerSystem):
    def __init__(self):
        self.pipeline = pipeline("sentiment-analysis")

    def analyze_sentiment(self, text):
        if not text:
            return 0.0
        result = self.pipeline(text)[0]
        return result['score'] if result['label'] == 'POSITIVE' else -result['score']

ENTITY_ROLES = {
    "actor": {"description": "Initiates an action"},
    "target": {"description": "Receives an action"},
    "parent": {"description": "Owns a child entity"},
    "child": {"description": "Belongs to a parent"},
    "associate": {"description": "Is connected to another entity"}
}

RELATION_TYPES = {
    "temporal": {"description": "Indicates a temporal relationship"},
    "action": {"description": "Indicates an action between entities"},
    "belonging": {"description": "Indicates a belonging relationship"},
    "association": {"description": "Indicates an association between entities"}
}


class RelationGraph:
    def __init__(self, text="", sentiment_analyzer_system=None):
        self.graph = nx.Graph()
        self.text = text
        self.sentiment_analyzer_system = sentiment_analyzer_system
        self.entity_ids = set()
        self.aggregate_sentiments = {}
        if self.sentiment_analyzer_system is None:
            self.sentiment_analyzer_system = VADERSentimentAnalyzer()
    
    def _validate_role(self, role):
        if role not in ENTITY_ROLES:
            raise ValueError(f"Invalid entity role: {role}. Valid roles are: {list(ENTITY_ROLES.keys())}")
    
    def _validate_relation(self, relation):
        if relation not in RELATION_TYPES:
            raise ValueError(f"Invalid relation type: {relation}. Valid relations are: {list(RELATION_TYPES.keys())}")

    def add_entity_node(self, id:int, head:str, modifier:List[str], entity_role:str, sentence_layer:int):
        self._validate_role(entity_role)
        self.entity_ids.add(id)
        self.graph.add_node(f"{id}_{sentence_layer}", head=head, modifier=modifier, text=self.text, entity_role=entity_role, sentence_layer=sentence_layer)
        self.graph.nodes[f"{id}_{sentence_layer}"]['init_sentiment'] = self.sentiment_analyzer_system.analyze_sentiment(head + ' ' + ' '.join(modifier))

    def add_temporal_edge(self, entity_id:int):
        if entity_id not in self.entity_ids:
            raise ValueError(f"Entity ID {entity_id} not found in the graph.")
        layers = []
        for node in self.graph.nodes:
            id_int = int(str(node).split('_')[0])
            sentence_layer = int(str(node).split('_')[1])
            if id_int == entity_id:
                layers.append(sentence_layer)
        layers = sorted(layers)
        for i in range(len(layers) - 1):
            self.graph.add_edge(f"{entity_id}_{layers[i]}", f"{entity_id}_{layers[i + 1]}", relation="temporal")
    
    def add_action_edge(self, actor_id:int, target_id:int, sentence_layer:int, head:str, modifier:List[str]):
        if actor_id not in self.entity_ids or target_id not in self.entity_ids:
            raise ValueError(f"Actor ID {actor_id} or Target ID {target_id} not found in the graph.")
        self.graph.add_edge(f"{actor_id}_{sentence_layer}", f"{target_id}_{sentence_layer}", 
                            actor=f"{actor_id}_{sentence_layer}", target=f"{target_id}_{sentence_layer}",
                            relation="action", head=head, modifier=modifier)
        self.graph.nodes[f"{actor_id}_{sentence_layer}"]['init_sentiment'] = self.sentiment_analyzer_system.analyze_sentiment(head + ' ' + ' '.join(modifier))

    def add_belonging_edge(self, parent_id:int, child_id:int, sentence_layer:int):
        if parent_id not in self.entity_ids or child_id not in self.entity_ids:
            raise ValueError(f"Parent ID {parent_id} or Child ID {child_id} not found in the graph.")
        self.graph.add_edge(f"{parent_id}_{sentence_layer}", f"{child_id}_{sentence_layer}", 
                            parent=f"{parent_id}_{sentence_layer}", child=f"{child_id}_{sentence_layer}",
                            relation="belonging")

    def add_association_edge(self, entity1_id:int, entity2_id:int, sentence_layer:int):
        if entity1_id not in self.entity_ids or entity2_id not in self.entity_ids:
            raise ValueError(f"Entity1 ID {entity1_id} or Entity2 ID {entity2_id} not found in the graph.")
        self.graph.add_edge(f"{entity1_id}_{sentence_layer}", f"{entity2_id}_{sentence_layer}", 
                            entity1=f"{entity1_id}_{sentence_layer}", entity2=f"{entity2_id}_{sentence_layer}",
                            relation="association")

    def run_compound_action_sentiment_calculations(self, function:Callable=None):
        for edge in self.graph.edges:
            edge_data = self.graph.edges[edge]
            if 'relation' in edge_data and edge_data['relation'] == "action":
                actor = edge_data['actor']
                target = edge_data['target']
                actor_sentiment_init = self.graph.nodes[actor]['init_sentiment']
                action_sentiment_init = edge_data.get('init_sentiment', 0.0)
                target_sentiment_init = self.graph.nodes[target]['init_sentiment']

                if function is None:
                    raise ValueError("No function provided for compound sentiment calculation")
                actor_sentiment_compound, target_sentiment_compound = function(actor_sentiment_init, action_sentiment_init, target_sentiment_init)

                self.graph.nodes[actor]['compound_sentiment'] = actor_sentiment_compound
                self.graph.nodes[target]['compound_sentiment'] = target_sentiment_compound

    def run_compound_belonging_sentiment_calculations(self, function:Callable=None):
        for edge in self.graph.edges:
            edge_data = self.graph.edges[edge]
            if 'relation' in edge_data and edge_data['relation'] == "belonging":
                parent = edge_data['parent']
                child = edge_data['child']
                parent_sentiment_init = self.graph.nodes[parent]['init_sentiment']
                child_sentiment_init = self.graph.nodes[child]['init_sentiment']

                if function is None:
                    raise ValueError("No function provided for compound sentiment calculation")
                parent_sentiment_compound, child_sentiment_compound = function(parent_sentiment_init, child_sentiment_init)

                self.graph.nodes[parent]['compound_sentiment'] = parent_sentiment_compound
                self.graph.nodes[child]['compound_sentiment'] = child_sentiment_compound

    def run_compound_association_sentiment_calculations(self, function:Callable=None):
        for edge in self.graph.edges:
            edge_data = self.graph.edges[edge]
            if 'relation' in edge_data and edge_data['relation'] == "association":
                entity1 = edge_data['entity1']
                entity2 = edge_data['entity2']
                entity1_sentiment_init = self.graph.nodes[entity1]['init_sentiment']
                entity2_sentiment_init = self.graph.nodes[entity2]['init_sentiment']

                if function is None:
                    raise ValueError("No function provided for compound sentiment calculation")
                entity1_sentiment_compound, entity2_sentiment_compound = function(entity1_sentiment_init, entity2_sentiment_init)

                self.graph.nodes[entity1]['compound_sentiment'] = entity1_sentiment_compound
                self.graph.nodes[entity2]['compound_sentiment'] = entity2_sentiment_compound

    def run_aggregate_sentiment_calculations(self, entity_id, function:Callable=None):
        layers = []
        for node in self.graph.nodes:
            id_int = int(str(node).split('_')[0])
            sentence_layer = int(str(node).split('_')[1])
            if id_int == entity_id:
                layers.append(sentence_layer)
        layers = sorted(layers)
        sentiments = []
        for layer in layers:
            node = f"{entity_id}_{layer}"
            sentiments.append(self.graph.nodes[node].get('compound_sentiment', 0.0)) if 'compound_sentiment' in self.graph.nodes[node] else self.graph.nodes[node].get('init_sentiment', 0.0)
        if function is None:
            raise ValueError("No function provided for aggregate sentiment calculation")
        result = function(sentiments) if sentiments else 0.0
        self.aggregate_sentiments[entity_id] = result
        return result


class GraphVisualizer:
    def __init__(self, relation_graph: RelationGraph):
        self.relation_graph = relation_graph
        self.graph = relation_graph.graph
        self.edge_color_map = {
            "temporal": "grey",
            "action": "blue",
            "belonging": "purple",
            "association": "orange"
        }

    def _get_node_colors(self):
        colors = {}
        for node, data in self.graph.nodes(data=True):
            sentiment = data.get('compound_sentiment', data.get('init_sentiment', 0.0))
            if sentiment > 0.1:
                colors[node] = 'green'
            elif sentiment < -0.1:
                colors[node] = 'red'
            else:
                colors[node] = 'lightgray'
        return colors

    def draw_graph(self, save_path=None):
        if not self.graph.nodes:
            print("Graph is empty. Nothing to draw.")
            return

        pos_2d = nx.spring_layout(self.graph, k=2.0, iterations=100, seed=42)
        pos_3d = {
            node: (pos_2d[node][0], pos_2d[node][1], data['sentence_layer'])
            for node, data in self.graph.nodes(data=True)
        }

        edge_traces = []
        edges_by_relation = {rel_type: [] for rel_type in self.edge_color_map.keys()}
        for u, v, data in self.graph.edges(data=True):
            relation = data.get('relation')
            if relation in edges_by_relation:
                edges_by_relation[relation].append((u, v))

        for relation, edges in edges_by_relation.items():
            if not edges:
                continue
            edge_x, edge_y, edge_z = [], [], []
            for u, v in edges:
                x0, y0, z0 = pos_3d[u]
                x1, y1, z1 = pos_3d[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_z.extend([z0, z1, None])
            
            trace = go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                line=dict(width=3, color=self.edge_color_map[relation]),
                hoverinfo='none',
                mode='lines',
                name=f'{relation.capitalize()} Relation'
            )
            edge_traces.append(trace)

        node_x, node_y, node_z = [], [], []
        node_text = []
        node_colors = []
        node_colors_dict = self._get_node_colors()

        for node, data in sorted(self.graph.nodes(data=True)):
            x, y, z = pos_3d[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_colors.append(node_colors_dict[node])
            
            sentiment = data.get('compound_sentiment', data.get('init_sentiment', 0.0))
            sentiment_str = f"{sentiment:.2f}"
            if 'compound_sentiment' in data:
                 sentiment_str += f" (compounded from {data.get('init_sentiment', 0.0):.2f})"

            hover_info = (f"<b>{data['head']} ({node})</b><br>"
                          f"Role: {data['entity_role']}<br>"
                          f"Layer: {data['sentence_layer']}<br>"
                          f"Sentiment: {sentiment_str}")
            node_text.append(hover_info)

        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                color=node_colors,
                size=15,
                line=dict(width=1, color='black')
            ),
            name='Entities'
        )

        fig = go.Figure(data=edge_traces + [node_trace])

        fig.update_layout(
            title='3D Relation Graph (Warning: Not using the real sentiment formulas)',
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.6)'),
            margin=dict(l=0, r=0, b=0, t=40),
            scene=dict(
                xaxis=dict(showticklabels=False, title=''),
                yaxis=dict(showticklabels=False, title=''),
                zaxis=dict(title='Sentence Layer', nticks=max(pos[2] for pos in pos_3d.values()) + 1)
            )
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Interactive graph saved to {save_path}")
        else:
            fig.show()


if __name__ == '__main__':
    story_text = """
    Alice, a brilliant engineer, proposed an innovative project.
    The ambitious rival, Bob, watched Alice with envy.
    The company's CEO, a wise leader, approved the valuable project.
    However, Bob treacherously sabotaged the project's delicate code.
    Alice, initially devastated, felt a surge of determination.
    She masterfully fixed the corrupted code, saving the entire project.
    """
    
    vader_analyzer = VADERSentimentAnalyzer()
    relation_graph = RelationGraph(text=story_text, sentiment_analyzer_system=vader_analyzer)

    relation_graph.add_entity_node(id=1, head="Alice", modifier=["brilliant engineer"], entity_role="actor", sentence_layer=0)
    relation_graph.add_entity_node(id=2, head="project", modifier=["innovative"], entity_role="target", sentence_layer=0)
    relation_graph.add_action_edge(actor_id=1, target_id=2, sentence_layer=0, head="proposed", modifier=[])

    relation_graph.add_entity_node(id=3, head="Bob", modifier=["ambitious", "rival"], entity_role="actor", sentence_layer=1)
    relation_graph.add_entity_node(id=1, head="Alice", modifier=[], entity_role="target", sentence_layer=1)
    relation_graph.add_action_edge(actor_id=3, target_id=1, sentence_layer=1, head="watched", modifier=["with envy"])

    relation_graph.add_entity_node(id=4, head="CEO", modifier=["wise leader"], entity_role="actor", sentence_layer=2)
    relation_graph.add_entity_node(id=5, head="company", modifier=[], entity_role="parent", sentence_layer=2)
    relation_graph.add_entity_node(id=2, head="project", modifier=["valuable"], entity_role="target", sentence_layer=2)
    relation_graph.add_action_edge(actor_id=4, target_id=2, sentence_layer=2, head="approved", modifier=[])
    relation_graph.add_belonging_edge(parent_id=5, child_id=4, sentence_layer=2)

    relation_graph.add_entity_node(id=3, head="Bob", modifier=[], entity_role="actor", sentence_layer=3)
    relation_graph.add_entity_node(id=6, head="code", modifier=["delicate"], entity_role="target", sentence_layer=3)
    relation_graph.add_entity_node(id=2, head="project", modifier=[], entity_role="parent", sentence_layer=3)
    relation_graph.add_action_edge(actor_id=3, target_id=6, sentence_layer=3, head="sabotaged", modifier=["treacherously"])
    relation_graph.add_belonging_edge(parent_id=2, child_id=6, sentence_layer=3)
    
    relation_graph.add_entity_node(id=1, head="Alice", modifier=["initially devastated", "determined"], entity_role="actor", sentence_layer=4)
    
    relation_graph.add_entity_node(id=1, head="She (Alice)", modifier=[], entity_role="actor", sentence_layer=5)
    relation_graph.add_entity_node(id=6, head="code", modifier=["corrupted"], entity_role="target", sentence_layer=5)
    relation_graph.add_entity_node(id=2, head="project", modifier=[], entity_role="associate", sentence_layer=5)
    relation_graph.add_action_edge(actor_id=1, target_id=6, sentence_layer=5, head="fixed", modifier=["masterfully"])
    relation_graph.add_association_edge(entity1_id=1, entity2_id=2, sentence_layer=5)

    relation_graph.add_temporal_edge(entity_id=1)
    relation_graph.add_temporal_edge(entity_id=2)
    relation_graph.add_temporal_edge(entity_id=3)
    relation_graph.add_temporal_edge(entity_id=6)

    def simple_action_logic(actor_s, action_s, target_s):
        return (actor_s + action_s, target_s + action_s)

    def simple_belonging_logic(parent_s, child_s):
        avg_s = (parent_s + child_s) / 2
        return (avg_s, avg_s)
    
    def simple_association_logic(e1_s, e2_s):
        avg_s = (e1_s + e2_s) / 2
        return (avg_s, avg_s)

    def average_sentiment(sentiments: List[float]):
        return sum(sentiments) / len(sentiments) if sentiments else 0.0

    relation_graph.run_compound_action_sentiment_calculations(function=simple_action_logic)
    relation_graph.run_compound_belonging_sentiment_calculations(function=simple_belonging_logic)
    relation_graph.run_compound_association_sentiment_calculations(function=simple_association_logic)

    print("\n--- Aggregate Sentiments Over the Entire Story ---")
    entity_names = {1: "Alice", 2: "Project", 3: "Bob", 4: "CEO", 5: "Company", 6: "Code"}
    for entity_id, name in entity_names.items():
        agg_sentiment = relation_graph.run_aggregate_sentiment_calculations(entity_id=entity_id, function=average_sentiment)
        print(f"{name} (ID {entity_id}): {agg_sentiment:.4f}")

    visualizer = GraphVisualizer(relation_graph)
    visualizer.draw_graph(save_path="graph_sentiment_analysis.html")
