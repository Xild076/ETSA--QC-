from utility import create_model, prepare_data, train_model, evaluate_model, predict, load_dataset, extract_features_from_doc, extract_features_from_token, nlp
from utility import DocFeatureType as DFT
from utility import TokenFeatureType as TFT
import pandas as pd

TARGET_DEPS = {"nsubj", "dobj", "ROOT" "prep", "pobj", "nsubjpass","auxpass", "agent"}

def extract_doc_data(data):
    """
    Extracts document-level features from the dataset.
    Args:
        data (list): A list of dictionaries containing the dataset.
    Outputs:
        pd.DataFrame: A DataFrame containing the extracted features.
    """
    feats = [extract_features_from_doc(d["text"], [DFT.NUM_TOKENS, DFT.ROOT_IS_NOUN, DFT.ROOT_IS_VERB, DFT.NUM_ALPHA, DFT.NUM_NSUBJ, DFT.NUM_DOBJ, DFT.NUM_POBJ, DFT.NUM_NSUBJPASS]) for d in data]
    return feats

def prepare_data_for_asvc(data):
    """
    Prepare the data for training by extracting features and labels.
    Args:
        data (list): A list of dictionaries containing the dataset.
    Outputs:
        pd.DataFrame: A DataFrame containing the extracted features.
        list: A list of labels.
    """
    feats = []
    labels = []
    for doc in data:
        doc_feats = extract_features_from_doc(doc["text"], [DFT.NUM_TOKENS, DFT.ROOT_IS_NOUN, DFT.ROOT_IS_VERB, DFT.NUM_ALPHA, DFT.NUM_NSUBJ, DFT.NUM_DOBJ, DFT.NUM_POBJ, DFT.NUM_NSUBJPASS])
        for idx, token in enumerate(nlp(doc["text"])):
            feats.append(extract_features_from_token(token, [TFT.INDEX, TFT.DEP, TFT.POS, TFT.NUM_ANCESTORS, TFT.IS_ALPHA, TFT.HEAD_DEP, TFT.ANCESTOR_CONTAINS_DEP], TARGET_DEPS, [], [], []) | doc_feats)
            if idx == doc['actor_subject']:
                labels.append(0)
            elif idx in doc["actor"]:
                labels.append(1)
            elif idx in doc["action"]:
                labels.append(2)
            elif idx in doc["victim"]:
                labels.append(3)
            else:
                labels.append(4)
    return feats, labels

if __name__ == "__main__":
    data = load_dataset("data/dataset.csv")
    
    x, y = prepare_data_for_asvc(data)
    x, y = prepare_data(x, y)
    
    model = train_model(x, y)
    
    print(evaluate_model(model, x, y))
    for i, clf in enumerate(model.estimators_):
        print(f"Class {i} rules:")
        for rule in clf.rules_:
            print(rule)
        print()

