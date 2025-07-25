from typing import Callable, List
from enum import Enum
import networkx as nx
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


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

class EntityRole(Enum):
    ACTOR = "actor"
    TARGET = "target"
    PARENT = "parent"
    CHILD = "child"
    ASSOCIATE = "associate"

class RelationType(Enum):
    TEMPORAL = "temporal"
    ACTION = "action"
    BELONGING = "belonging"
    ASSOCIATION = "association"

class RelationGraph:
    def __init__(self, text="", sentiment_analyzer_system=None):
        self.graph = nx.Graph()
        self.text = text
        self.sentiment_analyzer_system = sentiment_analyzer_system
        self.entity_ids = set()
        self.aggregate_sentiments = {}
        if self.sentiment_analyzer_system is None:
            self.sentiment_analyzer_system = VADERSentimentAnalyzer()

    def add_entity_node(self, id:int, head:str, modifier:List[str], entity_role:EntityRole, sentence_layer:int):
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
            self.graph.add_edge(f"{entity_id}_{layers[i]}", f"{entity_id}_{layers[i + 1]}", relation=RelationType.TEMPORAL)
    
    def add_action_edge(self, actor_id:int, target_id:int, sentence_layer:int, head:str, modifier:List[str]):
        if actor_id not in self.entity_ids or target_id not in self.entity_ids:
            raise ValueError(f"Actor ID {actor_id} or Target ID {target_id} not found in the graph.")
        self.graph.add_edge(f"{actor_id}_{sentence_layer}", f"{target_id}_{sentence_layer}", 
                            actor=f"{actor_id}_{sentence_layer}", target=f"{target_id}_{sentence_layer}",
                            relation=RelationType.ACTION, head=head, modifier=modifier)
        self.graph.nodes[f"{actor_id}_{sentence_layer}"]['init_sentiment'] = self.sentiment_analyzer_system.analyze_sentiment(head + ' ' + ' '.join(modifier))

    def add_belonging_edge(self, parent_id:int, child_id:int, sentence_layer:int):
        if parent_id not in self.entity_ids or child_id not in self.entity_ids:
            raise ValueError(f"Parent ID {parent_id} or Child ID {child_id} not found in the graph.")
        self.graph.add_edge(f"{parent_id}_{sentence_layer}", f"{child_id}_{sentence_layer}", 
                            parent=f"{parent_id}_{sentence_layer}", child=f"{child_id}_{sentence_layer}",
                            relation=RelationType.BELONGING)

    def add_association_edge(self, entity1_id:int, entity2_id:int, sentence_layer:int):
        if entity1_id not in self.entity_ids or entity2_id not in self.entity_ids:
            raise ValueError(f"Entity1 ID {entity1_id} or Entity2 ID {entity2_id} not found in the graph.")
        self.graph.add_edge(f"{entity1_id}_{sentence_layer}", f"{entity2_id}_{sentence_layer}", 
                            entity1=f"{entity1_id}_{sentence_layer}", entity2=f"{entity2_id}_{sentence_layer}",
                            relation=RelationType.ASSOCIATION)

    def run_compound_action_sentiment_calculations(self, function:Callable=None):
        for edge in self.graph.edges:
            if 'relation' in self.graph[edge] and self.graph[edge]['relation'] == RelationType.ACTION:
                actor = self.graph[edge]['actor']
                target = self.graph[edge]['target']
                actor_sentiment_init = self.graph.nodes[actor]['init_sentiment']
                action_sentiment_init = self.graph.edges[edge]['init_sentiment']
                target_sentiment_init = self.graph.nodes[target]['init_sentiment']

                if function is None:
                    raise ValueError("No function provided for compound sentiment calculation")
                actor_sentiment_compound, target_sentiment_compound = function(actor_sentiment_init, action_sentiment_init, target_sentiment_init)

                self.graph.nodes[actor]['compound_sentiment'] = actor_sentiment_compound
                self.graph.nodes[target]['compound_sentiment'] = target_sentiment_compound

    def run_compound_belonging_sentiment_calculations(self, function:Callable=None):
        for edge in self.graph.edges:
            if 'relation' in self.graph[edge] and self.graph[edge]['relation'] == RelationType.BELONGING:
                parent = self.graph[edge]['parent']
                child = self.graph[edge]['child']
                parent_sentiment_init = self.graph.nodes[parent]['init_sentiment']
                child_sentiment_init = self.graph.nodes[child]['init_sentiment']

                if function is None:
                    raise ValueError("No function provided for compound sentiment calculation")
                parent_sentiment_compound, child_sentiment_compound = function(parent_sentiment_init, child_sentiment_init)

                self.graph.nodes[parent]['compound_sentiment'] = parent_sentiment_compound
                self.graph.nodes[child]['compound_sentiment'] = child_sentiment_compound

    def run_compound_association_sentiment_calculations(self, function:Callable=None):
        for edge in self.graph.edges:
            if 'relation' in self.graph[edge] and self.graph[edge]['relation'] == RelationType.ASSOCIATION:
                entity1 = self.graph[edge]['entity1']
                entity2 = self.graph[edge]['entity2']
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