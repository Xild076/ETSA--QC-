from utility import create_model, prepare_data, train_model, evaluate_model, predict, load_dataset, extract_features_from_doc, extract_features_from_token
from utility import DocFeatureType as DFT
import pandas as pd

def prepare_data_for_type(data):
    """
    Prepare the data for training by extracting features and labels.
    Args:
        data (list): A list of dictionaries containing the dataset.
    Outputs:
        pd.DataFrame: A DataFrame containing the extracted features.
        list: A list of labels.
    """
    feats = [extract_features_from_doc(d["text"], [DFT.NUM_TOKENS, DFT.ROOT_IS_NOUN, DFT.ROOT_IS_VERB, DFT.NUM_ALPHA, DFT.NUM_NSUBJ, DFT.NUM_DOBJ, DFT.NUM_POBJ, DFT.NUM_NSUBJPASS]) for d in data]
    labels = [d["text_type"] for d in data]
    return feats, labels

if __name__ == "__main__":
    data = load_dataset("data/dataset.csv")
    
    x, y = prepare_data_for_type(data)
    x, y = prepare_data(x, y)
    
    model = train_model(x, y)
    
    print(evaluate_model(model, x, y))
    for i, clf in enumerate(model.estimators_):
        print(f"Class {i} rules:")
        for rule in clf.rules_:
            print(rule)
        print()
