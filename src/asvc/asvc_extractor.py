import spacy
import pandas as pd
import numpy as np
import pickle
import os
from skrules import SkopeRules
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.model_selection import train_test_split
from utility import _binary_train, load_dataset


nlp = spacy.load("en_core_web_sm")

def _doc_stats(doc):
    """Calculates various statistics for a spaCy Doc object."""
    subj = [t for t in doc if t.dep_ in ("nsubj", "nsubjpass")]
    obj = [t for t in doc if t.dep_ == "dobj"]
    verb = [t for t in doc if t.pos_ == "VERB"]
    num_tokens = len(doc)
    return {
        "num_tokens": num_tokens,
        "avg_tok_len": sum(len(t) for t in doc) / num_tokens if num_tokens > 0 else 0.0,
        "num_subjects": len(subj),
        "num_objects": len(obj),
        "num_verbs": len(verb),
        "num_adj": len([t for t in doc if t.pos_ == "ADJ"]),
        "num_adv": len([t for t in doc if t.pos_ == "ADV"]),
        "num_propn": len([t for t in doc if t.pos_ == "PROPN"]),
        "num_punct": len([t for t in doc if t.is_punct]),
        "has_passive": int(any(t.dep_ == "auxpass" for t in doc)),
        "has_verb": int(bool(verb)),
        "has_dobj": int(bool(obj)),
        "has_subject": int(bool(subj)),
        "root_is_verb": int(doc[0].pos_ == "VERB") if num_tokens > 0 else 0,
        "root_is_noun": int(doc[0].pos_ == "NOUN") if num_tokens > 0 else 0,
        "verb_dobj_dist": min((abs(x.i - y.i) for x in verb for y in obj), default=-1) if verb and obj else -1,
        "subj_first_idx": subj[0].i if subj else -1,
    }

TARGET_DEPS = {"nsubj", "dobj", "amod", "advmod", "ROOT", "conj", "prep", "pobj", "nsubjpass", "ccomp", "acl", "relcl", "auxpass", "agent"}
TARGET_POS = {"VERB", "NOUN", "ADJ", "ADV", "PROPN", "ADP"}

def token_features(doc, idx, stats):
    """Generates features for a single token within a spaCy Doc (Reduced Feature Set)."""
    tok = doc[idx]
    feats = stats.copy()
    feats.update(
        {
            "dep": tok.dep_,
            "pos": tok.pos_,
            "index": idx,
            "num_ancestors": len(list(tok.ancestors)),
            "num_children": len(list(tok.children)),
            "head_dep": tok.head.dep_,
            "head_pos": tok.head.pos_,
            "head_index": tok.head.i,
            "is_first": int(idx == 0),
            "is_last": int(idx == len(doc) - 1),
        }
    )

    for d in TARGET_DEPS:
        feats[f"has_child_{d}"] = 0
        feats[f"has_ancestor_{d}"] = 0
        feats[f"has_subtree_{d}"] = 0
    for p in TARGET_POS:
        feats[f"has_child_pos_{p}"] = 0
        feats[f"has_ancestor_pos_{p}"] = 0
        feats[f"has_subtree_pos_{p}"] = 0

    for child in tok.children:
        if child.dep_ in TARGET_DEPS: feats[f"has_child_{child.dep_}"] = 1
        if child.pos_ in TARGET_POS: feats[f"has_child_pos_{child.pos_}"] = 1
    for ancestor in tok.ancestors:
        if ancestor.dep_ in TARGET_DEPS: feats[f"has_ancestor_{ancestor.dep_}"] = 1
        if ancestor.pos_ in TARGET_POS: feats[f"has_ancestor_pos_{ancestor.pos_}"] = 1
    for node in tok.subtree:
        if node.dep_ in TARGET_DEPS: feats[f"has_subtree_{node.dep_}"] = 1
        if node.pos_ in TARGET_POS: feats[f"has_subtree_pos_{node.pos_}"] = 1

    return feats

def _encode_dataframe(df, mapping=None):
    """Encodes categorical string columns in a DataFrame using provided mapping."""
    is_new_mapping = mapping is None
    if is_new_mapping:
        mapping = {}
    for c in df.columns:
        if df[c].dtype == "object":
            if c not in mapping:
                if is_new_mapping:

                    mapping[c] = {v: i for i, v in enumerate(sorted(df[c].unique()))}
                else:

                    pass
            df[c] = df[c].map(mapping.get(c, {})).fillna(-1).astype(int)

    if is_new_mapping:
        return df, mapping
    else:
        return df

def sort_data_for_tvc(data):
    """Filters dataset for items with text_type 'tvc'."""
    return [item for item in data if item.get("text_type") == "tvc"]

def build_dataset_for_ascv(data):
    """Builds features and labels for ASVC task from processed data."""
    features, y = [], []
    texts = [item["text"] for item in data]
    docs = list(nlp.pipe(texts))

    for i, doc in enumerate(docs):
        item = data[i]
        stats = _doc_stats(doc)
        actors, actions, victims = item["actor"], item["action"], item["victim"]
        actor_subject = item["actor_subject"]

        for idx in range(len(doc)):
            clause = 0
            if idx in actors:
                clause = 1
            elif idx in actions:
                clause = 2
            elif idx in victims:
                clause = 3
            a_c = int(idx == actor_subject)

            features.append(token_features(doc, idx, stats))
            y.append([clause, a_c])

    return features, y

def train_skopes_rules_for_asvc(features, y, path_out="models/skopes_models/skopes_asvc.pkl"):
    """
    Trains SkopeRules models for clause labels and actor–subject coreference.

    Args:
        features (list[dict]): List of feature dictionaries for each token.
        y (list[list]): List of [clause_label, actor_coref_label] for each token.
        path_out (str): Path to save the trained models and associated data.
    """
    if not features:
        print("No features provided for training. Exiting.")
        return

    feature_names = list(features[0].keys())
    x_df = pd.DataFrame(features, columns=feature_names)
    x_df, enc_map = _encode_dataframe(x_df)

    y_arr = np.asarray(y)
    clause_labels = y_arr[:, 0]
    ac_labels = y_arr[:, 1]

    models_clause = {}
    print("\n=== Training Clause Models ===")
    unique_clause_labels = sorted(set(clause_labels))
    for lbl in unique_clause_labels:
        print(f"\nProcessing clause label: '{lbl}'")
        y_bin = (clause_labels == lbl).astype(int)

        if y_bin.sum() < 2 or (len(y_bin) - y_bin.sum()) < 2:
            print(f"  Skipping '{lbl}': insufficient positive or negative samples.")
            models_clause[lbl] = None
            continue

        print(f"  Training model for '{lbl}'...")

        m = _binary_train(x_df, y_bin, feature_names)
        if not hasattr(m, 'rules_') or not m.rules_:
            print(f"  No rules found for '{lbl}'.")
            models_clause[lbl] = None
        else:
            print(f"\n  Rules found for '{lbl}':")

            try:
                 for rule_details in m.rules_:

                    rule = rule_details[0]
                    prec, rec = rule_details[1][0], rule_details[1][1]

                    print(f"    {rule} (precision={prec:.2f}, recall={rec:.2f})")
            except (IndexError, TypeError, AttributeError) as e:
                print(f"    Could not display rules for {lbl} due to format issue: {e}")
            models_clause[lbl] = m

    print("\n=== Training Actor-Subject Model ===")
    model_ac = None
    if ac_labels.sum() < 2 or (len(ac_labels) - ac_labels.sum()) < 2:
        print("  Skipping actor-coref model: insufficient positive or negative samples.")
    else:
        print("  Training actor-subject model…")
        model_ac = _binary_train(x_df, ac_labels, feature_names)
        if not hasattr(model_ac, 'rules_') or not model_ac.rules_:
            print("  No rules found for actor-subject.")
            model_ac = None
        else:
             print("  Rules found for actor-subject.")


    print("\n=== Saving Models ===")
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    try:
        with open(path_out, "wb") as f:
            pickle.dump(
                {
                    "models_clause": models_clause,
                    "model_ac": model_ac,
                    "feature_names": feature_names,
                    "enc_map": enc_map,
                },
                f,
            )
        print(f"Models saved to {path_out}")
    except Exception as e:
        print(f"Error saving models to {path_out}: {e}")

def classify_document_asvc(text, model_path="models/skopes_models/skopes_asvc.pkl"):
    """
    Predicts actor/action/victim tokens and the coreferent actor-subject token.

    Args:
        text (str): The input text document.
        model_path (str): Path to the saved models file.

    Returns:
        tuple: (text, actor_indices, actor_coref_index, action_indices,
                victim_indices, irrelevant_indices)
    """
    try:
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return text, [], None, [], [], []
    except Exception as e:
        print(f"Error loading model file {model_path}: {e}")
        return text, [], None, [], [], []

    models_clause = saved["models_clause"]
    model_ac = saved["model_ac"]
    feature_names = saved["feature_names"]
    enc_map = saved["enc_map"]

    doc = nlp(text)
    if not doc:
        return text, [], None, [], [], []

    stats = _doc_stats(doc)


    all_token_features = [token_features(doc, idx, stats) for idx in range(len(doc))]


    df = pd.DataFrame(all_token_features, columns=feature_names).fillna(0)

    df_encoded = _encode_dataframe(df.copy(), mapping=enc_map)


    df_encoded = df_encoded[feature_names]
    token_data = df_encoded.values

    clause_preds = np.zeros(len(df_encoded), dtype=int)
    ac_preds = np.zeros(len(df_encoded), dtype=int)


    for idx in range(len(token_data)):
        token_feat_vector = token_data[idx:idx+1, :]
        best_clause, best_score = 0, -np.inf


        for lbl, mdl in models_clause.items():
            if mdl is None: continue
            try:

                sc = mdl.score_top_rules(token_feat_vector)[0]
                if not np.isnan(sc) and sc > best_score:
                    best_score, best_clause = sc, lbl
            except Exception as e:

                 pass
        clause_preds[idx] = best_clause


        if model_ac is not None:
            try:
                s = model_ac.score_top_rules(token_feat_vector)[0]
                ac_preds[idx] = int(not np.isnan(s) and s > 0)
            except Exception as e:

                ac_preds[idx] = 0


    actor_idx = [i for i, p in enumerate(clause_preds) if p == 1]
    action_idx = [i for i, p in enumerate(clause_preds) if p == 2]
    victim_idx = [i for i, p in enumerate(clause_preds) if p == 3]
    irrelevant_idx = [i for i, p in enumerate(clause_preds) if p == 0]


    actor_coref_idx_list = [i for i, p in enumerate(ac_preds) if p == 1]
    actor_coref_idx = actor_coref_idx_list[0] if actor_coref_idx_list else None

    return (
        text,
        actor_idx,
        actor_coref_idx,
        action_idx,
        victim_idx,
        irrelevant_idx,
    )

"""if __name__ == "__main__":
    try:
        data = load_dataset()
        tvc_data = sort_data_for_tvc(data)
        if not tvc_data:
             print("No data with text_type 'tvc' found.")
        else:
            print(f"Building dataset from {len(tvc_data)} items...")
            feats, labels = build_dataset_for_ascv(tvc_data)
            print("Training models...")
            train_skopes_rules_for_asvc(feats, labels)

            print("\nTesting classification on a sample sentence...")
            test_sentence = "While complaining, John beat up Adam."

            model_file = "models/skopes_models/skopes_asvc.pkl"
            result = classify_document_asvc(test_sentence, model_path=model_file)

            print("\nClassification Result:")
            print(f"Text: {result[0]}")
            doc_test = nlp(result[0])
            tokens = [t.text for t in doc_test]
            print(f"Actor Tokens (Indices: {result[1]}): {[tokens[i] for i in result[1]]}")
            actor_coref_text = tokens[result[2]] if result[2] is not None else 'None'
            print(f"Actor Coref Token (Index: {result[2]}): {actor_coref_text}")
            print(f"Action Tokens (Indices: {result[3]}): {[tokens[i] for i in result[3]]}")
            print(f"Victim Tokens (Indices: {result[4]}): {[tokens[i] for i in result[4]]}")
            print(f"Irrelevant Tokens (Indices: {result[5]}): {[tokens[i] for i in result[5]]}")

    except NameError as e:
        print(f"Error: Missing definition or import for: {e}")
        print("Please ensure 'load_dataset' and '_binary_train' are defined or imported correctly.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")"""