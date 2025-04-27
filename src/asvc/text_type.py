from skrules import SkopeRules
import spacy
import pandas as pd
import sklearn
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from packaging import version
import numpy as np
from utility import _binary_train, load_dataset


nlp = spacy.load("en_core_web_sm")

def extract_features_for_type(text):
    """
    Extract linguistic features from text using spaCy.
    Args:
        text (str): The input text.
    Outputs:
        dict: A dictionary of extracted features.
    """
    doc = nlp(text)
    subj = [t for t in doc if t.dep_ in ("nsubj", "nsubjpass")]
    obj  = [t for t in doc if t.dep_ == "dobj"]
    verb = [t for t in doc if t.pos_ == "VERB"]
    root = list(doc.sents)[0].root if doc.sents else doc[0]
    def _mindist(a, b):
        return min(abs(x.i - y.i) for x in a for y in b) if a and b else -1
    return {
        "num_tokens"        : len(doc),
        "avg_tok_len"       : np.mean([len(t) for t in doc]) if doc else 0.0,
        "num_subjects"      : len(subj),
        "num_objects"       : len(obj),
        "num_verbs"         : len(verb),
        "num_adj"           : len([t for t in doc if t.pos_ == "ADJ"]),
        "num_adv"           : len([t for t in doc if t.pos_ == "ADV"]),
        "num_propn"         : len([t for t in doc if t.pos_ == "PROPN"]),
        "num_punct"         : len([t for t in doc if t.is_punct]),
        "has_passive"       : int(any(t.dep_ == "auxpass" for t in doc)),
        "has_verb"          : int(bool(verb)),
        "has_dobj"          : int(bool(obj)),
        "has_subject"       : int(bool(subj)),
        "root_is_verb"      : int(root.pos_ == "VERB"),
        "root_is_noun"      : int(root.pos_ == "NOUN"),
        "verb_dobj_dist"    : _mindist(verb, obj),
        "subj_first_idx"    : subj[0].i if subj else -1
    }

def prepare_data_for_type(data):
    """
    Prepare the data for training by extracting features and labels.
    Args:
        data (list): A list of dictionaries containing the dataset.
    Outputs:
        x_df (pd.DataFrame): A DataFrame containing the extracted features.
        y (list): A list of labels.
    """
    feats = [extract_features_for_type(d["text"]) for d in data]
    labels = [d["text_type"] for d in data]
    return pd.DataFrame(feats), labels

def train_skopes_rules_for_type(x_df, y, path_out):
    """
    Train SkopeRules models for each class in a multiclass classification problem.
    Args:
        x_df (pd.DataFrame): A DataFrame containing the extracted features.
        y (list): A list of labels.
        path_out (str): The path to save the trained models.
    Outputs:
        None
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    models = {}
    feature_names = list(x_df.columns)
    
    print("--- Training SkopeRules Models ---")
    for idx, lbl in enumerate(le.classes_):
        y_bin = (y_enc == idx).astype(int)
        
        print(f"\nProcessing class: '{lbl}'")
        if y_bin.sum() < 2:
            print(f"  Skipping '{lbl}': requires at least 2 positive samples, found {y_bin.sum()}.")
            continue
        if (len(y_bin) - y_bin.sum()) < 1:
             print(f"  Skipping '{lbl}': requires at least 1 negative sample, found {len(y_bin) - y_bin.sum()}.")
             continue

        print(f"  Training model for '{lbl}'...")
        try:
            m = _binary_train(x_df, y_bin, feature_names)
            rules = m.rules_
            if not rules:
                 print(f"  No rules found for '{lbl}' with current settings.")
                 models[lbl] = None
                 continue

            print(f"\n  Rules found for '{lbl}':")
            for rule_tuple in rules:
                 print(f"    {rule_tuple[0]} (precision={rule_tuple[1][0]:.2f}, recall={rule_tuple[1][1]:.2f}, support={rule_tuple[1][2]})")
            models[lbl] = m
        except Exception as e:
            print(f"  Error training model for '{lbl}': {e}")
            models[lbl] = None

    print("\n--- Saving Models ---")
    with open(path_out, "wb") as f:
        pickle.dump({"models": models, "le": le, "feature_names": feature_names}, f)
    print(f"Models saved to {path_out}")

def classify(text, model_path="models/skopes_models/skopes_text_type.pkl"):
    """
    Classify text using pre-trained SkopeRules models.
    Args:
        text (str): The input text to classify.
        model_path (str): Path to the saved models file.
    Outputs:
        tuple: (best_label, scores) where best_label is the predicted class
               or 'unknown', and scores is a dictionary of scores per class.
    """
    try:
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        models = saved["models"]
        feature_names = saved["feature_names"]
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return "error", {}
    except Exception as e:
        print(f"Error loading models: {e}")
        return "error", {}

    feats_dict = extract_features_for_type(text)
    feats_df = pd.DataFrame([feats_dict], columns=feature_names)
    for col in feature_names:
        if col not in feats_df.columns:
            feats_df[col] = 0
    feats_df = feats_df[feature_names]

    scores = {}
    for lbl, mdl in models.items():
        if mdl is not None:
            try:
                 score = mdl.score_top_rules(feats_df.values)[0]
                 scores[lbl] = score if not np.isnan(score) else 0.0
            except Exception as e:
                 print(f"Error scoring with model for '{lbl}': {e}")
                 scores[lbl] = 0.0
        else:
            scores[lbl] = 0.0

    positive_scores = {lbl: score for lbl, score in scores.items() if score > 0}

    if not positive_scores:
        best_label = 'unknown'
    else:
        best_label = max(positive_scores, key=positive_scores.get)

    return best_label, scores


if __name__ == "__main__":
    try:
        data = load_dataset()
        if not data:
             print("Error: No data loaded. Check 'data/dataset.csv'.")
        else:
             X_df, Y = prepare_data_for_type(data)
             model_save_path = "models/skopes_models/skopes_text_type.pkl"
             import os
             os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

             train_skopes_rules_for_type(X_df, Y, model_save_path)

             print("\n--- Classifying Example Text ---")
             example_text = "The cat sat on the mat."
             predicted_type, all_scores = classify(example_text, model_save_path)
             print(f"\nText: '{example_text}'")
             print(f"Predicted Type: {predicted_type}")
             print(f"Scores: {all_scores}")

             example_text_2 = "Release version 1.2.3"
             predicted_type_2, all_scores_2 = classify(example_text_2, model_save_path)
             print(f"\nText: '{example_text_2}'")
             print(f"Predicted Type: {predicted_type_2}")
             print(f"Scores: {all_scores_2}")

    except FileNotFoundError:
         print("Error: 'data/dataset.csv' not found. Please ensure the dataset exists.")
    except Exception as e:
         print(f"An unexpected error occurred: {e}")
