import pandas as pd
import ssl
import ast
import numpy as np
from scipy.optimize import curve_fit, minimize
from sklearn.metrics import mean_squared_error
import re
from rich import print
import matplotlib.pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context

url = 'https://docs.google.com/spreadsheets/d/1xAvDLhU0w-p2hAZ49QYM7-XBMQCek0zVYJWpiN1Mvn0/export?format=csv&gid=0'
df = pd.read_csv(url)

intensity_map = {
    'very': 0.85,  # Midpoint of VADER's [0.7, 1.0] range
    'medium': 0.6,   # Midpoint of [0.5, 0.7]
    'somewhat': 0.4, # Midpoint of [0.3, 0.5]
    'slightly': 0.2, # Midpoint of [0.1, 0.3]
    'neutral': 0.0 # Midpoint of [-0.1, 0.1]
}
sentiment_sign_map = {'positive': 1, 'negative': -1}

def get_initial_score(intensity_str, descriptor_sign_str):
    if not isinstance(intensity_str, str) or not isinstance(descriptor_sign_str, str):
        return 0
    sign = sentiment_sign_map.get(descriptor_sign_str, 0)
    magnitude = intensity_map.get(intensity_str, 0)
    return sign * magnitude

def show_formulas(*formula_dicts):
    merged = {}
    for d in formula_dicts:
        merged.update(d)
    count = len(merged)
    height = max(4, count * 0.8)
    fig = plt.figure(figsize=(8, height))
    y = 0.95
    dy = 0.9 / count if count else 1
    for key, latex in merged.items():
        title = key.replace('_', ' ').title()
        text = rf"$\bf{{{title}}}$: {latex}"
        fig.text(0.05, y, text, fontsize=14)
        y -= dy
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def create_action_df():
    action_df = df[df['item_type'] == 'compound_action'].copy()
    action_data = []
    for _, row in action_df.iterrows():
        intensity = ast.literal_eval(row['intensity'])
        descriptor_sign = ast.literal_eval(row['descriptor'])
        item_id = row['item_id']
        seed = row['seed']

        target_entities = row['all_entities']
        target_entities = ast.literal_eval(target_entities)

        item_rows = df[(df['item_id'] == item_id) & (df['seed'] == seed)]

        score_actor = item_rows[item_rows['entity'] == target_entities[0]]['user_sentiment_label'].iloc[0]
        score_target = item_rows[item_rows['entity'] == target_entities[1]]['user_sentiment_label'].iloc[0]

        if score_actor == "Positive":
            score_actor = "medium positive"
        elif score_actor == "Negative":
            score_actor = "medium negative"
        if score_target == "Positive":
            score_target = "medium positive"
        elif score_target == "Negative":
            score_target = "medium negative"

        score_actor_split = score_actor.split(' ') if score_actor != "Neutral" else ['neutral', '0']
        if score_actor_split[0] == 'extremely':
            score_actor_split[0] = 'very'
        score_target_split = score_target.split(' ') if score_target != "Neutral" else ['neutral', '0']
        if score_target_split[0] == 'extremely':
            score_target_split[0] = 'very'
        score_actor_split = [sas.lower() for sas in score_actor_split]
        score_target_split = [sts.lower() for sts in score_target_split]

        action_data.append({
            's_init_actor': get_initial_score(intensity[0], descriptor_sign[0]),
            's_action': get_initial_score(intensity[1], descriptor_sign[1]),
            's_init_target': get_initial_score(intensity[2], descriptor_sign[2]),
            's_user_actor': get_initial_score(score_actor_split[0], score_actor_split[1]),
            's_user_target': get_initial_score(score_target_split[0], score_target_split[1]),
        })
    action_model_df = pd.DataFrame(action_data)
    return action_model_df

def actor_formula(X, lambda_actor, w, b):
    s_init_actor, driver = X
    s_new = lambda_actor * s_init_actor + (1 - lambda_actor) * w * driver + b
    return np.tanh(s_new)

def determine_actor_formula_parameters():
    action_model_df = create_action_df()

    action_model_df['driver'] = action_model_df['s_action'] * action_model_df['s_init_target']

    pos_driver_df = action_model_df[action_model_df['driver'] > 0]
    neg_driver_df = action_model_df[action_model_df['driver'] < 0]

    print("\n--- Actor Formula Validation ---")

    if not pos_driver_df.empty:
        X_pos = pos_driver_df[['s_init_actor', 'driver']].to_numpy().T
        y_pos = pos_driver_df['s_user_actor'].to_numpy()
        try:
            params_pos, _ = curve_fit(actor_formula, X_pos, y_pos, bounds=([0, -5, -1], [1, 5, 1]))
            y_pred_pos = actor_formula(X_pos, *params_pos)
            mse_pos = mean_squared_error(y_pos, y_pred_pos)
            print(f"Positive Driver - Optimal Params (lambda, w, b): {np.round(params_pos, 4)}")
            print(f"Positive Driver - MSE: {mse_pos:.4f}")
        except Exception as e:
            print(f"Could not fit positive driver model: {e}")
    else:
        print("No data available for positive driver.")

    if not neg_driver_df.empty:
        X_neg = neg_driver_df[['s_init_actor', 'driver']].to_numpy().T
        y_neg = neg_driver_df['s_user_actor'].to_numpy()
        try:
            params_neg, _ = curve_fit(actor_formula, X_neg, y_neg, bounds=([0, -5, -1], [1, 5, 1]))
            y_pred_neg = actor_formula(X_neg, *params_neg)
            mse_neg = mean_squared_error(y_neg, y_pred_neg)
            print(f"Negative Driver - Optimal Params (lambda, w, b): {np.round(params_neg, 4)}")
            print(f"Negative Driver - MSE: {mse_neg:.4f}")
        except Exception as e:
            print(f"Could not fit negative driver model: {e}")
    else:
        print("No data available for negative driver.")

    return (params_pos[0], params_pos[1], params_pos[2]), (params_neg[0], params_neg[1], params_neg[2])

def target_formula(X, lambda_target, w, b):
    s_init_target, s_action = X
    s_new = lambda_target * s_init_target + (1 - lambda_target) * w * s_action + b
    return np.tanh(s_new)

def determine_target_formula_parameters():
    action_model_df = create_action_df()

    pos_action_df = action_model_df[action_model_df['s_action'] > 0]
    neg_action_df = action_model_df[action_model_df['s_action'] < 0]

    print("\n--- Target Formula Validation ---")

    if not pos_action_df.empty:
        X_pos = pos_action_df[['s_init_target', 's_action']].to_numpy().T
        y_pos = pos_action_df['s_user_target'].to_numpy()
        try:
            params_pos, _ = curve_fit(target_formula, X_pos, y_pos, bounds=([0, -5, -1], [1, 5, 1]))
            y_pred_pos = target_formula(X_pos, *params_pos)
            mse_pos = mean_squared_error(y_pos, y_pred_pos)
            print(f"Positive Action - Optimal Params (lambda, w, b): {np.round(params_pos, 4)}")
            print(f"Positive Action - MSE: {mse_pos:.4f}")
        except Exception as e:
            print(f"Could not fit positive action model: {e}")
    else:
        print("No data available for positive action.")

    if not neg_action_df.empty:
        X_neg = neg_action_df[['s_init_target', 's_action']].to_numpy().T
        y_neg = neg_action_df['s_user_target'].to_numpy()
        try:
            params_neg, _ = curve_fit(target_formula, X_neg, y_neg, bounds=([0, -5, -1], [1, 5, 1]))
            y_pred_neg = target_formula(X_neg, *params_neg)
            mse_neg = mean_squared_error(y_neg, y_pred_neg)
            print(f"Negative Action - Optimal Params (lambda, w, b): {np.round(params_neg, 4)}")
            print(f"Negative Action - MSE: {mse_neg:.4f}")
        except Exception as e:
            print(f"Could not fit negative action model: {e}")
    else:
        print("No data available for negative action.")

    return (params_pos[0], params_pos[1], params_pos[2]), (params_neg[0], params_neg[1], params_neg[2])

def print_compound_action_formula(lambda_actor_pos, w_actor_pos, b_actor_pos, lambda_target_pos, w_target_pos, b_target_pos, lambda_actor_neg, w_actor_neg, b_actor_neg, lambda_target_neg, w_target_neg, b_target_neg):
    print("\n--- Action Formula Text ---")
    lambda_actor_pos = round(lambda_actor_pos, 6)
    w_actor_pos = round(w_actor_pos, 6)
    b_actor_pos = round(b_actor_pos, 6)
    lambda_target_pos = round(lambda_target_pos, 6)
    w_target_pos = round(w_target_pos, 6)
    b_target_pos = round(b_target_pos, 6)
    lambda_actor_neg = round(lambda_actor_neg, 6)
    w_actor_neg = round(w_actor_neg, 6)
    b_actor_neg = round(b_actor_neg, 6)
    lambda_target_neg = round(lambda_target_neg, 6)
    w_target_neg = round(w_target_neg, 6)
    b_target_neg = round(b_target_neg, 6)

    print(f"Positive Actor Formula:")
    print(f"    s_final_actor = {lambda_actor_pos} * s_init_actor + (1 - {lambda_actor_pos}) * {w_actor_pos} * s_action * s_init_target  + {b_actor_pos}")

    print(f"Negative Actor Formula:")
    print(f"    s_final_actor = {lambda_actor_neg} * s_init_actor + (1 - {lambda_actor_neg}) * {w_actor_neg} * s_action * s_init_target  + {b_actor_neg}")

    print(f"Positive Target Formula:")
    print(f"    s_final_target = {lambda_target_pos} * s_init_target + (1 - {lambda_target_pos}) * {w_target_pos} * s_action + {b_target_pos}")

    print(f"Negative Target Formula:")
    print(f"    s_final_target = {lambda_target_neg} * s_init_target + (1 - {lambda_target_neg}) * {w_target_neg} * s_action + {b_target_neg}")

    return {
        "Positive actor formula": f"$s_{{final\\_actor}} = \\tanh[{lambda_actor_pos} \\cdot s_{{init\\_actor}} + (1 - {lambda_actor_pos}) \\cdot {w_actor_pos} \\cdot s_{{action}} \\cdot s_{{init\\_target}} + {b_actor_pos}]$",
        "Negative actor formula": f"$s_{{final\\_actor}} = \\tanh[{lambda_actor_neg} \\cdot s_{{init\\_actor}} + (1 - {lambda_actor_neg}) \\cdot {w_actor_neg} \\cdot s_{{action}} \\cdot s_{{init\\_target}} + {b_actor_neg}]$",
        "Positive target formula": f"$s_{{final\\_target}} = \\tanh[{lambda_target_pos} \\cdot s_{{init\\_target}} + (1 - {lambda_target_pos}) \\cdot {w_target_pos} \\cdot s_{{action}} + {b_target_pos}]$",
        "Negative target formula": f"$s_{{final\\_target}} = \\tanh[{lambda_target_neg} \\cdot s_{{init\\_target}} + (1 - {lambda_target_neg}) \\cdot {w_target_neg} \\cdot s_{{action}} + {b_target_neg}]$"
    }


def create_association_df():
    assoc_df = df[df['item_type'] == 'compound_association'].copy()
    assoc_data = []
    for _, row in assoc_df.iterrows():
        intensity = ast.literal_eval(row['intensity'])
        descriptor_sign = ast.literal_eval(row['descriptor'])
        code_key = row['code_key']
        item_id = row['item_id']
        seed = row['seed']

        names = re.findall(r'actor\[\[([^\]_]+)_', code_key)

        target_entities = row['all_entities']
        target_entities = ast.literal_eval(target_entities)
        
        item_rows = df[(df['item_id'] == item_id) & (df['seed'] == seed)]

        score_entity = item_rows[item_rows['entity'] == names[0]]['user_sentiment_label'].iloc[0]
        score_other = item_rows[item_rows['entity'] == names[1]]['user_sentiment_label'].iloc[0]

        if score_entity == "Positive":
            score_entity = "medium positive"
        elif score_entity == "Negative":
            score_entity = "medium negative"
        if score_other == "Positive":
            score_other = "medium positive"
        elif score_other == "Negative":
            score_other = "medium negative"

        score_entity_split = score_entity.split(' ') if score_entity != "Neutral" else ['neutral', '0']
        if score_entity_split[0] == 'extremely':
            score_entity_split[0] = 'very'
        score_other_split = score_other.split(' ') if score_other != "Neutral" else ['neutral', '0']
        if score_other_split[0] == 'extremely':
            score_other_split[0] = 'very'
        score_entity_split = [sas.lower() for sas in score_entity_split]
        score_other_split = [sts.lower() for sts in score_other_split]

        entity_index = target_entities.index(names[0])
        other_index = target_entities.index(names[1])

        assoc_data.append({
            's_init_entity': get_initial_score(intensity[entity_index], descriptor_sign[entity_index]),
            's_init_other': get_initial_score(intensity[other_index], descriptor_sign[other_index]),
            's_final_user': get_initial_score(score_entity_split[0], score_entity_split[1]),
        })
        assoc_data.append({
            's_init_entity': get_initial_score(intensity[other_index], descriptor_sign[other_index]),
            's_init_other': get_initial_score(intensity[entity_index], descriptor_sign[entity_index]),
            's_final_user': get_initial_score(score_other_split[0], score_other_split[1]),
        })
    assoc_model_df = pd.DataFrame(assoc_data)
    return assoc_model_df

def assoc_formula(X, lambda_val, w, b):
    s_init, s_other = X
    s_new = lambda_val * s_init + (1 - lambda_val) * w * s_other + b
    return np.tanh(s_new)

def determine_association_formula_parameters():
    assoc_model_df = create_association_df()

    pos_other_df = assoc_model_df[assoc_model_df['s_init_other'] > 0]
    neg_other_df = assoc_model_df[assoc_model_df['s_init_other'] < 0]

    print("\n--- Association Formula Validation ---")
    if not pos_other_df.empty:
        X_pos = pos_other_df[['s_init_entity', 's_init_other']].to_numpy().T
        y_pos = pos_other_df['s_final_user'].to_numpy()
        try:
            params_pos, _ = curve_fit(assoc_formula, X_pos, y_pos, bounds=([0, -5, -1], [1, 5, 1]))
            mse = mean_squared_error(y_pos, assoc_formula(X_pos, *params_pos))
            print(f"Positive 'Other' - Optimal Params (lambda, w, b): {np.round(params_pos, 4)}")
            print(f"Positive 'Other' - MSE: {mse:.4f}")
        except Exception:
            print("Could not fit model for Positive 'Other'.")

    if not neg_other_df.empty:
        X_neg = neg_other_df[['s_init_entity', 's_init_other']].to_numpy().T
        y_neg = neg_other_df['s_final_user'].to_numpy()
        try:
            params_neg, _ = curve_fit(assoc_formula, X_neg, y_neg, bounds=([0, -5, -1], [1, 5, 1]))
            mse = mean_squared_error(y_neg, assoc_formula(X_neg, *params_neg))
            print(f"Negative 'Other' - Optimal Params (lambda, w, b): {np.round(params_neg, 4)}")
            print(f"Negative 'Other' - MSE: {mse:.4f}")
        except Exception:
            print("Could not fit model for Negative 'Other'.")

    return (params_pos[0], params_pos[1], params_pos[2]), (params_neg[0], params_neg[1], params_neg[2])

def print_compound_association_formula(lambda_pos, w_pos, b_pos, lambda_neg, w_neg, b_neg):
    print("\n--- Association Formula Text ---")
    lambda_pos = round(lambda_pos, 6)
    w_pos = round(w_pos, 6)
    b_pos = round(b_pos, 6)
    lambda_neg = round(lambda_neg, 6)
    w_neg = round(w_neg, 6)
    b_neg = round(b_neg, 6)

    print(f"Positive Formula:")
    print(f"    s_final = {lambda_pos} * s_init_entity + (1 - {lambda_pos}) * {w_pos} * s_init_other + {b_pos}")

    print(f"Negative Formula:")
    print(f"    s_final = {lambda_neg} * s_init_entity + (1 - {lambda_neg}) * {w_neg} * s_init_other + {b_neg}")

    return {
        "Positive Association Formula": f"$s_{{final}} = \\tanh[{lambda_pos} \\cdot s_{{init\\_entity}} + (1 - {lambda_pos}) \\cdot {w_pos} \\cdot s_{{init\\_other}} + {b_pos}]$",
        "Negative Association Formula": f"$s_{{final}} = \\tanh[{lambda_neg} \\cdot s_{{init\\_entity}} + (1 - {lambda_neg}) \\cdot {w_neg} \\cdot s_{{init\\_other}} + {b_neg}]$"
    }


def create_belonging_df():
    action_df = df[df['item_type'] == 'compound_belonging'].copy()
    action_data = []
    for _, row in action_df.iterrows():
        intensity = ast.literal_eval(row['intensity'])
        descriptor_sign = ast.literal_eval(row['descriptor'])
        item_id = row['item_id']
        seed = row['seed']

        target_entities = row['all_entities']
        target_entities = ast.literal_eval(target_entities)

        item_rows = df[(df['item_id'] == item_id) & (df['seed'] == seed)]

        score_parent = item_rows[item_rows['entity'] == target_entities[0]]['user_sentiment_label'].iloc[0]
        score_child = item_rows[item_rows['entity'] == target_entities[1]]['user_sentiment_label'].iloc[0]

        if score_parent == "Positive":
            score_parent = "medium positive"
        elif score_parent == "Negative":
            score_parent = "medium negative"
        if score_child == "Positive":
            score_child = "medium positive"
        elif score_child == "Negative":
            score_child = "medium negative"

        score_parent_split = score_parent.split(' ') if score_parent != "Neutral" else ['neutral', '0']
        if score_parent_split[0] == 'extremely':
            score_parent_split[0] = 'very'
        score_child_split = score_child.split(' ') if score_child != "Neutral" else ['neutral', '0']
        if score_child_split[0] == 'extremely':
            score_child_split[0] = 'very'
        score_parent_split = [sas.lower() for sas in score_parent_split]
        score_child_split = [sts.lower() for sts in score_child_split]

        action_data.append({
            's_init_parent': get_initial_score(intensity[0], descriptor_sign[0]),
            's_init_child': get_initial_score(intensity[1], descriptor_sign[1]),
            's_user_parent': get_initial_score(score_parent_split[0], score_parent_split[1]),
            's_user_child': get_initial_score(score_child_split[0], score_child_split[1]),
        })
    action_model_df = pd.DataFrame(action_data)
    return action_model_df

def belong_formula(X, lambda_parent, w, b):
    s_init, s_other = X
    s_new = lambda_parent * s_init + (1 - lambda_parent) * w * s_other + b
    return np.tanh(s_new)

def determine_parent_formula_parameters():
    assoc_model_df = create_belonging_df()

    pos_other_df = assoc_model_df[assoc_model_df['s_init_child'] > 0]
    neg_other_df = assoc_model_df[assoc_model_df['s_init_child'] < 0]

    print("\n--- Parent Belonging Formula Validation ---")
    if not pos_other_df.empty:
        X_pos = pos_other_df[['s_init_parent', 's_init_child']].to_numpy().T
        y_pos = pos_other_df['s_init_parent'].to_numpy()
        try:
            params_pos, _ = curve_fit(belong_formula, X_pos, y_pos, bounds=([0, -5, -1], [1, 5, 1]))
            mse = mean_squared_error(y_pos, belong_formula(X_pos, *params_pos))
            print(f"Positive 'Other' - Optimal Params (lambda, w, b): {np.round(params_pos, 4)}")
            print(f"Positive 'Other' - MSE: {mse:.4f}")
        except Exception:
            print("Could not fit model for Positive 'Other'.")

    if not neg_other_df.empty:
        X_neg = neg_other_df[['s_init_parent', 's_init_child']].to_numpy().T
        y_neg = neg_other_df['s_init_parent'].to_numpy()
        try:
            params_neg, _ = curve_fit(belong_formula, X_neg, y_neg, bounds=([0, -5, -1], [1, 5, 1]))
            mse = mean_squared_error(y_neg, belong_formula(X_neg, *params_neg))
            print(f"Negative 'Other' - Optimal Params (lambda, w, b): {np.round(params_neg, 4)}")
            print(f"Negative 'Other' - MSE: {mse:.4f}")
        except Exception:
            print("Could not fit model for Negative 'Other'.")
    return (params_pos[0], params_pos[1], params_pos[2]), (params_neg[0], params_neg[1], params_neg[2])

def determine_child_formula_parameters():
    assoc_model_df = create_belonging_df()

    pos_other_df = assoc_model_df[assoc_model_df['s_init_parent'] > 0]
    neg_other_df = assoc_model_df[assoc_model_df['s_init_parent'] < 0]

    print("\n--- Child Belonging Formula Validation ---")
    if not pos_other_df.empty:
        X_pos = pos_other_df[['s_init_child', 's_init_parent']].to_numpy().T
        y_pos = pos_other_df['s_init_child'].to_numpy()
        try:
            params_pos, _ = curve_fit(belong_formula, X_pos, y_pos, bounds=([0, -5, -1], [1, 5, 1]))
            mse = mean_squared_error(y_pos, belong_formula(X_pos, *params_pos))
            print(f"Positive 'Other' - Optimal Params (lambda, w, b): {np.round(params_pos, 4)}")
            print(f"Positive 'Other' - MSE: {mse:.4f}")
        except Exception:
            print("Could not fit model for Positive 'Other'.")

    if not neg_other_df.empty:
        X_neg = neg_other_df[['s_init_child', 's_init_parent']].to_numpy().T
        y_neg = neg_other_df['s_init_child'].to_numpy()
        try:
            params_neg, _ = curve_fit(belong_formula, X_neg, y_neg, bounds=([0, -5, -1], [1, 5, 1]))
            mse = mean_squared_error(y_neg, belong_formula(X_neg, *params_neg))
            print(f"Negative 'Other' - Optimal Params (lambda, w, b): {np.round(params_neg, 4)}")
            print(f"Negative 'Other' - MSE: {mse:.4f}")
        except Exception:
            print("Could not fit model for Negative 'Other'.")

    return (params_pos[0], params_pos[1], params_pos[2]), (params_neg[0], params_neg[1], params_neg[2])

def print_compound_belonging_formula(lambda_parent_pos, w_parent_pos, b_parent_pos, lambda_parent_neg, w_parent_neg, b_parent_neg, lambda_child_pos, w_child_pos, b_child_pos, lambda_child_neg, w_child_neg, b_child_neg):
    print("\n--- Belonging Formula Text ---")
    lambda_parent_pos = round(lambda_parent_pos, 6)
    w_parent_pos = round(w_parent_pos, 6)
    b_parent_pos = round(b_parent_pos, 6)
    lambda_parent_neg = round(lambda_parent_neg, 6)
    w_parent_neg = round(w_parent_neg, 6)
    b_parent_neg = round(b_parent_neg, 6)
    lambda_child_pos = round(lambda_child_pos, 6)
    w_child_pos = round(w_child_pos, 6)
    b_child_pos = round(b_child_pos, 6)
    lambda_child_neg = round(lambda_child_neg, 6)
    w_child_neg = round(w_child_neg, 6)
    b_child_neg = round(b_child_neg, 6)

    print(f"Positive Parent Formula:")
    print(f"    s_final_parent = {lambda_parent_pos} * s_init_parent + (1 - {lambda_parent_pos}) * {w_parent_pos} * s_init_child + {b_parent_pos}")

    print(f"Negative Parent Formula:")
    print(f"    s_final_parent = {lambda_parent_neg} * s_init_parent + (1 - {lambda_parent_neg}) * {w_parent_neg} * s_init_child + {b_parent_neg}")

    print(f"Positive Child Formula:")
    print(f"    s_final_child = {lambda_child_pos} * s_init_child + (1 - {lambda_child_pos}) * {w_child_pos} * s_init_parent + {b_child_pos}")

    print(f"Negative Child Formula:")
    print(f"    s_final_child = {lambda_child_neg} * s_init_child + (1 - {lambda_child_neg}) * {w_child_neg} * s_init_parent + {b_child_neg}")

    return {
        "Positive Parent Formula": f"$s_{{final\\_parent}} = \\tanh[{lambda_parent_pos} \\cdot s_{{init\\_parent}} + (1 - {lambda_parent_pos}) \\cdot {w_parent_pos} \\cdot s_{{init\\_child}} + {b_parent_pos}]$",
        "Negative Parent Formula": f"$s_{{final\\_parent}} = \\tanh[{lambda_parent_neg} \\cdot s_{{init\\_parent}} + (1 - {lambda_parent_neg}) \\cdot {w_parent_neg} \\cdot s_{{init\\_child}} + {b_parent_neg}]$",
        "Positive Child Formula": f"$s_{{final\\_child}} = \\tanh[{lambda_child_pos} \\cdot s_{{init\\_child}} + (1 - {lambda_child_pos}) \\cdot {w_child_pos} \\cdot s_{{init\\_parent}} + {b_child_pos}]$",
        "Negative Child Formula": f"$s_{{final\\_child}} = \\tanh[{lambda_child_neg} \\cdot s_{{init\\_child}} + (1 - {lambda_child_neg}) \\cdot {w_child_neg} \\cdot s_{{init\\_parent}} + {b_child_neg}]$"
    }


def create_aggregate_df():
    agg_df = df[df['item_type'].str.contains('aggregate', na=False)].copy()
    agg_data = []
    for _, row in agg_df.iterrows():
        full_descriptors = ast.literal_eval(row['descriptor'])
        full_intensities = ast.literal_eval(row['intensity'])

        item_id = row['item_id']
        seed = row['seed']

        item_rows = df[(df['item_id'] == item_id) & (df['seed'] == seed)]

        initial_scores = [get_initial_score(i, d) for d, i in zip(full_descriptors, full_intensities)]

        for n in range(2, len(initial_scores) + 1):
            score_entity = item_rows[item_rows['packet_step'] == n]['user_sentiment_label'].iloc[0]

            if score_entity == "Positive":
                score_entity = "medium positive"
            elif score_entity == "Negative":
                score_entity = "medium negative"

            score_entity_split = score_entity.split(' ') if score_entity != "Neutral" else ['neutral', '0']
            if score_entity_split[0] == 'extremely':
                score_entity_split[0] = 'very'

            score_entity_split = [sas.lower() for sas in score_entity_split]

            agg_data.append({
                'N': n,
                'initial_scores': initial_scores[:n],
                'final_user_score': get_initial_score(score_entity_split[0], score_entity_split[1]),
            })
    agg_model_df = pd.DataFrame(agg_data)
    return agg_model_df

def logistic_function(N, L, k, N0, b):
    return b + L / (1 + np.exp(-k * (N - N0)))

def get_weights(N, alpha, beta):
    k_vals = np.arange(1, N + 1)
    alpha, beta = max(alpha, 0.01), max(beta, 0.01)
    numerator = (k_vals**(alpha - 1)) * ((N - k_vals + 1)**(beta - 1))
    denominator = np.sum(numerator)
    return numerator / denominator if denominator != 0 else np.zeros(N)

def objective_function(params, data):
    L_a, k_a, N0_a, b_a, L_b, k_b, N0_b, b_b = params
    total_error = 0
    
    for _, row in data.iterrows():
        N = row['N']
        initial_scores = np.array(row['initial_scores'])
        true_score = row['final_user_score']
        
        alpha = logistic_function(N, L_a, k_a, N0_a, b_a)
        beta = logistic_function(N, L_b, k_b, N0_b, b_b)
        
        weights = get_weights(N, alpha, beta)
        predicted_score = np.sum(weights * initial_scores)
        
        total_error += (predicted_score - true_score)**2
        
    return total_error

def determine_aggregate_formula_parameters():
    agg_df = create_aggregate_df()

    initial_guess = [-10, 1, 3, 10, 10, 1, 3, 1] 

    bounds = [(-20, 20), (0.1, 5), (1, 6), (0.01, 20),
            (-20, 20), (0.1, 5), (1, 6), (0.01, 20)]

    result = minimize(
        objective_function,
        initial_guess,
        args=(agg_df,),
        method='L-BFGS-B',
        bounds=bounds
    )

    optimized_params = result.x
    final_mse = result.fun / len(agg_df)

    print("\n--- Aggregate Model (Non-Linear) ---")
    L_a, k_a, N0_a, b_a, L_b, k_b, N0_b, b_b = optimized_params
    print("Optimized Logistic Equations for α(N) and β(N):")
    print(f"α(N) = {b_a:.2f} + ({L_a:.2f}) / (1 + exp(-{k_a:.2f} * (N - {N0_a:.2f})))")
    print(f"β(N) = {b_b:.2f} + ({L_b:.2f}) / (1 + exp(-{k_b:.2f} * (N - {N0_b:.2f})))")

    print(f"MSE: {final_mse:.4f}\n")

    return L_a, k_a, N0_a, b_a, L_b, k_b, N0_b, b_b

def print_aggregate_formula(L_a, k_a, N0_a, b_a, L_b, k_b, N0_b, b_b):
    print("\n--- Aggregate Formula Text ---")

    print(f"General Formula:")
    print(f"    s_final = Σ (w_i * s_i) for i = 1 to N")

    print(f"Where:")
    print(f"    w_i = (i^[α-1]*[N-i+1]^[β-1]) / Σ (k^[α-1]*[N-k+1]^[β-1]) for k = 1 to N")

    print(f"Where:")
    print(f"    α(N) [Primacy] = {b_a:.2f} + ({L_a:.2f}) / (1 + exp(-{k_a:.2f} * (N - {N0_a:.2f})))")
    print(f"    β(N) [Recency] = {b_b:.2f} + ({L_b:.2f}) / (1 + exp(-{k_b:.2f} * (N - {N0_b:.2f})))")
    
    return {
        "General Aggregate Formula": "$s_{final} = \\sum_{i=1}^{N} (w_i \\cdot s_i)$",
        "Aggregate Weight Formula": "$w_i = \\frac{i^{\\alpha-1} \\cdot (N-i+1)^{\\beta-1}}{\\sum_{k=1}^{N} (k^{\\alpha-1} \\cdot (N-k+1)^{\\beta-1})}$",
        "Alpha Formula": f"$\\alpha(N) = {b_a:.2f} + \\frac{{{L_a:.2f}}}{{1 + e^{{-{k_a:.2f} \\cdot (N - {N0_a:.2f})}}}}$",
        "Beta Formula": f"$\\beta(N) = {b_b:.2f} + \\frac{{{L_b:.2f}}}{{1 + e^{{-{k_b:.2f} \\cdot (N - {N0_b:.2f})}}}}$"
    }



print("Using mapped scores")
actor_params = determine_actor_formula_parameters()
target_params = determine_target_formula_parameters()
association_params = determine_association_formula_parameters()
parent_params = determine_parent_formula_parameters()
child_params = determine_child_formula_parameters()
aggregate_params = determine_aggregate_formula_parameters()

action_formulas_dict = print_compound_action_formula(*actor_params[0], *target_params[0], *actor_params[1], *target_params[1])
association_formulas_dict = print_compound_association_formula(*association_params[0], *association_params[1])
belonging_formulas_dict = print_compound_belonging_formula(*parent_params[0], *parent_params[1], *child_params[0], *child_params[1])
aggregate_formulas_dict = print_aggregate_formula(*aggregate_params)

show_formulas(
    action_formulas_dict,
    association_formulas_dict,
    belonging_formulas_dict,
    aggregate_formulas_dict
)