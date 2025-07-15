import pandas as pd
import ssl
import ast
import numpy as np
from scipy.optimize import curve_fit, minimize
from sklearn.metrics import mean_squared_error
from rich import print
import matplotlib.pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context

url = 'https://docs.google.com/spreadsheets/d/1xAvDLhU0w-p2hAZ49QYM7-XBMQCek0zVYJWpiN1Mvn0/export?format=csv&gid=0'
df = pd.read_csv(url)

intensity_map = {
    'very': 0.85, 'medium': 0.6, 'somewhat': 0.4, 'slightly': 0.2, 'neutral': 0.0
}
sentiment_sign_map = {'positive': 1, 'negative': -1}

def get_initial_score(intensity_str, descriptor_sign_str):
    intensity_str = intensity_str.lower().replace("extremely", "very")
    descriptor_sign_str = descriptor_sign_str.lower()
    sign = sentiment_sign_map.get(descriptor_sign_str, 0)
    magnitude = intensity_map.get(intensity_str, 0)
    return sign * magnitude

def process_user_label(label_str):
    if label_str == "Neutral": return 0.0
    parts = label_str.split()
    if len(parts) == 1: intensity, sign_word = "medium", parts[0]
    elif len(parts) == 2: intensity, sign_word = parts[0], parts[1]
    else: return 0.0
    return get_initial_score(intensity, sign_word)

def show_formulas(*formula_dicts):
    merged = {}
    for d in formula_dicts: merged.update(d)
    count = len(merged)
    height = max(4, count * 0.8)
    fig = plt.figure(figsize=(12, height))
    y = 0.95
    dy = 0.9 / count if count else 1
    for key, latex in merged.items():
        title = key.replace('_', ' ').title()
        text = f"$\\bf{{{title}}}$: {latex}"
        fig.text(0.02, y, text, fontsize=14)
        y -= dy
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def safe_get_score(df_slice, entity_name):
    filtered_rows = df_slice[df_slice['entity'] == entity_name]
    return filtered_rows['user_sentiment_label'].iloc[0] if len(filtered_rows) == 1 else "Neutral"

# --- Data Creation Functions (No changes needed here) ---
def create_action_df():
    action_df = df[df['item_type'] == 'compound_action'].copy()
    action_data = []
    processed_items = set()
    for _, row in action_df.iterrows():
        item_id, seed = row['item_id'], row['seed']
        if (item_id, seed) in processed_items: continue
        processed_items.add((item_id, seed))
        
        intensity = ast.literal_eval(row['intensity'])
        descriptor_sign = ast.literal_eval(row['descriptor'])
        target_entities = ast.literal_eval(row['all_entities'])
        item_rows = df[(df['item_id'] == item_id) & (df['seed'] == seed)]
        
        actor_name, target_name = target_entities[0], target_entities[1]
        score_actor_label = safe_get_score(item_rows, actor_name)
        score_target_label = safe_get_score(item_rows, target_name)

        action_data.append({
            's_init_actor': get_initial_score(intensity[0], descriptor_sign[0]),
            's_action': get_initial_score(intensity[1], descriptor_sign[1]),
            's_init_target': get_initial_score(intensity[2], descriptor_sign[2]),
            's_user_actor': process_user_label(score_actor_label),
            's_user_target': process_user_label(score_target_label),
        })
    return pd.DataFrame(action_data)

def create_association_df():
    assoc_df = df[df['item_type'] == 'compound_association'].copy()
    assoc_data = []
    processed_items = set()
    for _, row in assoc_df.iterrows():
        item_id, seed = row['item_id'], row['seed']
        if (item_id, seed) in processed_items: continue
        processed_items.add((item_id, seed))
        intensity = ast.literal_eval(row['intensity']); descriptor_sign = ast.literal_eval(row['descriptor'])
        entities = ast.literal_eval(row['all_entities'])
        if len(entities) < 2: continue
        entity1_name, entity2_name = entities[0], entities[1]
        item_rows = df[(df['item_id'] == item_id) & (df['seed'] == seed)]
        score1_label = safe_get_score(item_rows, entity1_name); score2_label = safe_get_score(item_rows, entity2_name)
        s_init1 = get_initial_score(intensity[0], descriptor_sign[0]); s_init2 = get_initial_score(intensity[1], descriptor_sign[1])
        s_final1 = process_user_label(score1_label); s_final2 = process_user_label(score2_label)
        assoc_data.append({'s_init_entity': s_init1, 's_init_other': s_init2, 's_final_user': s_final1})
        assoc_data.append({'s_init_entity': s_init2, 's_init_other': s_init1, 's_final_user': s_final2})
    return pd.DataFrame(assoc_data)

def create_belonging_df():
    belong_df = df[df['item_type'] == 'compound_belonging'].copy()
    belong_data = []
    processed_items = set()
    for _, row in belong_df.iterrows():
        item_id, seed = row['item_id'], row['seed']
        if (item_id, seed) in processed_items: continue
        processed_items.add((item_id, seed))
        intensity = ast.literal_eval(row['intensity']); descriptor_sign = ast.literal_eval(row['descriptor'])
        entities = ast.literal_eval(row['all_entities'])
        parent_name, child_name = entities[0], entities[1]
        item_rows = df[(df['item_id'] == item_id) & (df['seed'] == seed)]
        score_parent_label = safe_get_score(item_rows, parent_name); score_child_label = safe_get_score(item_rows, child_name)
        belong_data.append({
            's_init_parent': get_initial_score(intensity[0], descriptor_sign[0]),
            's_init_child': get_initial_score(intensity[1], descriptor_sign[1]),
            's_user_parent': process_user_label(score_parent_label),
            's_user_child': process_user_label(score_child_label),
        })
    return pd.DataFrame(belong_data)

# UPGRADE 1: All compound formulas now use the new regression-style structure.
def compound_formula(X, w_init, w_update, b):
    s_init, driver = X
    s_new = w_init * s_init + w_update * driver + b
    return np.tanh(s_new)

def determine_actor_formula_parameters():
    action_model_df = create_action_df()
    action_model_df['driver'] = action_model_df['s_action'] * action_model_df['s_init_target']
    pos_driver_df = action_model_df[action_model_df['driver'] > 0]
    neg_driver_df = action_model_df[action_model_df['driver'] < 0]
    params_pos = [0,0,0]; params_neg = [0,0,0]
    print("\n--- Actor Formula Validation ---")
    if not pos_driver_df.empty:
        X_pos, y_pos = pos_driver_df[['s_init_actor', 'driver']].to_numpy().T, pos_driver_df['s_user_actor'].to_numpy()
        try:
            # UPGRADE 2: Adjust bounds for w_init, w_update, and b.
            params_pos, _ = curve_fit(compound_formula, X_pos, y_pos, bounds=([0, -5, -1], [5, 5, 1]))
            print(f"Positive Driver - Optimal Params (w_init, w_upd, b): {np.round(params_pos, 4)}, MSE: {mean_squared_error(y_pos, compound_formula(X_pos, *params_pos)):.4f}")
        except Exception as e: print(f"Could not fit positive driver model: {e}")
    if not neg_driver_df.empty:
        X_neg, y_neg = neg_driver_df[['s_init_actor', 'driver']].to_numpy().T, neg_driver_df['s_user_actor'].to_numpy()
        try:
            params_neg, _ = curve_fit(compound_formula, X_neg, y_neg, bounds=([0, -5, -1], [5, 5, 1]))
            print(f"Negative Driver - Optimal Params (w_init, w_upd, b): {np.round(params_neg, 4)}, MSE: {mean_squared_error(y_neg, compound_formula(X_neg, *params_neg)):.4f}")
        except Exception as e: print(f"Could not fit negative driver model: {e}")
    return tuple(params_pos), tuple(params_neg)

def determine_target_formula_parameters():
    action_model_df = create_action_df()
    pos_action_df = action_model_df[action_model_df['s_action'] > 0]
    neg_action_df = action_model_df[action_model_df['s_action'] < 0]
    params_pos = [0,0,0]; params_neg = [0,0,0]
    print("\n--- Target Formula Validation ---")
    if not pos_action_df.empty:
        X_pos, y_pos = pos_action_df[['s_init_target', 's_action']].to_numpy().T, pos_action_df['s_user_target'].to_numpy()
        try:
            params_pos, _ = curve_fit(compound_formula, X_pos, y_pos, bounds=([0, -5, -1], [5, 5, 1]))
            print(f"Positive Action - Optimal Params (w_init, w_upd, b): {np.round(params_pos, 4)}, MSE: {mean_squared_error(y_pos, compound_formula(X_pos, *params_pos)):.4f}")
        except Exception as e: print(f"Could not fit positive action model: {e}")
    if not neg_action_df.empty:
        X_neg, y_neg = neg_action_df[['s_init_target', 's_action']].to_numpy().T, neg_action_df['s_user_target'].to_numpy()
        try:
            params_neg, _ = curve_fit(compound_formula, X_neg, y_neg, bounds=([0, -5, -1], [5, 5, 1]))
            print(f"Negative Action - Optimal Params (w_init, w_upd, b): {np.round(params_neg, 4)}, MSE: {mean_squared_error(y_neg, compound_formula(X_neg, *params_neg)):.4f}")
        except Exception as e: print(f"Could not fit negative action model: {e}")
    return tuple(params_pos), tuple(params_neg)

def determine_association_formula_parameters():
    assoc_model_df = create_association_df()
    pos_other_df = assoc_model_df[assoc_model_df['s_init_other'] > 0]
    neg_other_df = assoc_model_df[assoc_model_df['s_init_other'] < 0]
    params_pos = [0,0,0]; params_neg = [0,0,0]
    print("\n--- Association Formula Validation ---")
    if not pos_other_df.empty:
        X_pos, y_pos = pos_other_df[['s_init_entity', 's_init_other']].to_numpy().T, pos_other_df['s_final_user'].to_numpy()
        try:
            params_pos, _ = curve_fit(compound_formula, X_pos, y_pos, bounds=([0, -5, -1], [5, 5, 1]))
            print(f"Positive 'Other' - Optimal Params (w_init, w_upd, b): {np.round(params_pos, 4)}, MSE: {mean_squared_error(y_pos, compound_formula(X_pos, *params_pos)):.4f}")
        except Exception as e: print(f"Could not fit model for Positive 'Other': {e}")
    if not neg_other_df.empty:
        X_neg, y_neg = neg_other_df[['s_init_entity', 's_init_other']].to_numpy().T, neg_other_df['s_final_user'].to_numpy()
        try:
            params_neg, _ = curve_fit(compound_formula, X_neg, y_neg, bounds=([0, -5, -1], [5, 5, 1]))
            print(f"Negative 'Other' - Optimal Params (w_init, w_upd, b): {np.round(params_neg, 4)}, MSE: {mean_squared_error(y_neg, compound_formula(X_neg, *params_neg)):.4f}")
        except Exception as e: print(f"Could not fit model for Negative 'Other': {e}")
    return tuple(params_pos), tuple(params_neg)

def determine_parent_formula_parameters():
    belong_model_df = create_belonging_df()
    pos_child_df = belong_model_df[belong_model_df['s_init_child'] > 0]
    neg_child_df = belong_model_df[belong_model_df['s_init_child'] < 0]
    params_pos=[0,0,0]; params_neg=[0,0,0]
    print("\n--- Parent Belonging Formula Validation ---")
    if not pos_child_df.empty:
        X_pos, y_pos = pos_child_df[['s_init_parent', 's_init_child']].to_numpy().T, pos_child_df['s_user_parent'].to_numpy()
        try:
            params_pos, _ = curve_fit(compound_formula, X_pos, y_pos, bounds=([0, -5, -1], [5, 5, 1]))
            print(f"Positive Child - Optimal Params (w_init, w_upd, b): {np.round(params_pos, 4)}, MSE: {mean_squared_error(y_pos, compound_formula(X_pos, *params_pos)):.4f}")
        except Exception as e: print(f"Could not fit for Positive Child: {e}")
    if not neg_child_df.empty:
        X_neg, y_neg = neg_child_df[['s_init_parent', 's_init_child']].to_numpy().T, neg_child_df['s_user_parent'].to_numpy()
        try:
            params_neg, _ = curve_fit(compound_formula, X_neg, y_neg, bounds=([0, -5, -1], [5, 5, 1]))
            print(f"Negative Child - Optimal Params (w_init, w_upd, b): {np.round(params_neg, 4)}, MSE: {mean_squared_error(y_neg, compound_formula(X_neg, *params_neg)):.4f}")
        except Exception as e: print(f"Could not fit for Negative Child: {e}")
    return tuple(params_pos), tuple(params_neg)

def determine_child_formula_parameters():
    belong_model_df = create_belonging_df()
    pos_parent_df = belong_model_df[belong_model_df['s_init_parent'] > 0]
    neg_parent_df = belong_model_df[belong_model_df['s_init_parent'] < 0]
    params_pos=[0,0,0]; params_neg=[0,0,0]
    print("\n--- Child Belonging Formula Validation ---")
    if not pos_parent_df.empty:
        X_pos, y_pos = pos_parent_df[['s_init_child', 's_init_parent']].to_numpy().T, pos_parent_df['s_user_child'].to_numpy()
        try:
            params_pos, _ = curve_fit(compound_formula, X_pos, y_pos, bounds=([0, -5, -1], [5, 5, 1]))
            print(f"Positive Parent - Optimal Params (w_init, w_upd, b): {np.round(params_pos, 4)}, MSE: {mean_squared_error(y_pos, compound_formula(X_pos, *params_pos)):.4f}")
        except Exception as e: print(f"Could not fit for Positive Parent: {e}")
    if not neg_parent_df.empty:
        X_neg, y_neg = neg_parent_df[['s_init_child', 's_init_parent']].to_numpy().T, neg_parent_df['s_user_child'].to_numpy()
        try:
            params_neg, _ = curve_fit(compound_formula, X_neg, y_neg, bounds=([0, -5, -1], [5, 5, 1]))
            print(f"Negative Parent - Optimal Params (w_init, w_upd, b): {np.round(params_neg, 4)}, MSE: {mean_squared_error(y_neg, compound_formula(X_neg, *params_neg)):.4f}")
        except Exception as e: print(f"Could not fit for Negative Parent: {e}")
    return tuple(params_pos), tuple(params_neg)

# --- Aggregate Model Functions (Unchanged) ---
def create_aggregate_df():
    agg_df = df[df['item_type'].str.contains('aggregate', na=False)].copy()
    agg_data = []
    for group_keys, item_group in agg_df.groupby(['seed', 'item_id']):
        first_row = item_group.iloc[0]
        full_descriptors = ast.literal_eval(first_row['descriptor'])
        full_intensities = ast.literal_eval(first_row['intensity'])
        initial_scores = [get_initial_score(i, d) for d, i in zip(full_intensities, full_descriptors)]
        for step_n_float in item_group['packet_step'].unique():
            step_n = int(step_n_float)
            step_row = item_group[item_group['packet_step'] == step_n_float]
            if step_row.empty: continue
            score_label = step_row['user_sentiment_label'].iloc[0]
            agg_data.append({'N': step_n, 'initial_scores': initial_scores[:step_n], 'final_user_score': process_user_label(score_label)})
    return pd.DataFrame(agg_data)

def logistic_function(N, L, k, N0, b): return b + L / (1 + np.exp(-k * (N - N0)))

def get_weights(N, alpha, beta):
    k_vals = np.arange(1, N + 1)
    alpha, beta = max(alpha, 0.01), max(beta, 0.01)
    with np.errstate(over='ignore', invalid='ignore'): numerator = np.power(k_vals, alpha - 1) * np.power(N - k_vals + 1, beta - 1)
    denominator = np.sum(numerator)
    return numerator / denominator if denominator != 0 and np.isfinite(denominator) else np.full_like(k_vals, 1/N, dtype=float)

def objective_function(params, data):
    L_a, k_a, N0_a, b_a, L_b, k_b, N0_b, b_b = params
    total_error = 0
    for _, row in data.iterrows():
        N, initial_scores, true_score = row['N'], np.array(row['initial_scores']), row['final_user_score']
        alpha, beta = logistic_function(N, L_a, k_a, N0_a, b_a), logistic_function(N, L_b, k_b, N0_b, b_b)
        weights = get_weights(N, alpha, beta)
        predicted_score = np.sum(weights * initial_scores)
        total_error += (predicted_score - true_score)**2
    return total_error

def determine_aggregate_formula_parameters():
    agg_df = create_aggregate_df()
    initial_guess = [-10, 1, 3, 10, 10, 1, 3, 1] 
    bounds = [(-20, 20), (0.1, 5), (1, 10), (0.01, 20), (-20, 20), (0.1, 5), (1, 10), (0.01, 20)]
    result = minimize(objective_function, initial_guess, args=(agg_df,), method='L-BFGS-B', bounds=bounds)
    print("\n--- Aggregate Model (Non-Linear) ---")
    L_a, k_a, N0_a, b_a, L_b, k_b, N0_b, b_b = result.x
    print(f"α(N) = {b_a:.2f} + ({L_a:.2f}) / (1 + exp(-{k_a:.2f} * (N - {N0_a:.2f})))")
    print(f"β(N) = {b_b:.2f} + ({L_b:.2f}) / (1 + exp(-{k_b:.2f} * (N - {N0_b:.2f})))")
    print(f"MSE: {result.fun / len(agg_df):.4f}\n")
    return result.x

def print_formulas(actor_params, target_params, association_params, parent_params, child_params, aggregate_params):
    # UPGRADE 3: Update the formula strings for the new regression-style model.
    formulas = {
        "Positive Actor Formula": f"s_{{final}} = \\tanh[{actor_params[0][0]:.2f}s_{{init}} + {actor_params[0][1]:.2f}s_{{driver}} + {actor_params[0][2]:.2f}]",
        "Negative Actor Formula": f"s_{{final}} = \\tanh[{actor_params[1][0]:.2f}s_{{init}} + {actor_params[1][1]:.2f}s_{{driver}} + {actor_params[1][2]:.2f}]",
        "Positive Target Formula": f"s_{{final}} = \\tanh[{target_params[0][0]:.2f}s_{{init}} + {target_params[0][1]:.2f}s_{{action}} + {target_params[0][2]:.2f}]",
        "Negative Target Formula": f"s_{{final}} = \\tanh[{target_params[1][0]:.2f}s_{{init}} + {target_params[1][1]:.2f}s_{{action}} + {target_params[1][2]:.2f}]",
        "Positive Association Formula": f"s_{{final}} = \\tanh[{association_params[0][0]:.2f}s_{{init}} + {association_params[0][1]:.2f}s_{{other}} + {association_params[0][2]:.2f}]",
        "Negative Association Formula": f"s_{{final}} = \\tanh[{association_params[1][0]:.2f}s_{{init}} + {association_params[1][1]:.2f}s_{{other}} + {association_params[1][2]:.2f}]",
        "Positive Parent Formula": f"s_{{final}} = \\tanh[{parent_params[0][0]:.2f}s_{{init}} + {parent_params[0][1]:.2f}s_{{child}} + {parent_params[0][2]:.2f}]",
        "Negative Parent Formula": f"s_{{final}} = \\tanh[{parent_params[1][0]:.2f}s_{{init}} + {parent_params[1][1]:.2f}s_{{child}} + {parent_params[1][2]:.2f}]",
        "Positive Child Formula": f"s_{{final}} = \\tanh[{child_params[0][0]:.2f}s_{{init}} + {child_params[0][1]:.2f}s_{{parent}} + {child_params[0][2]:.2f}]",
        "Negative Child Formula": f"s_{{final}} = \\tanh[{child_params[1][0]:.2f}s_{{init}} + {child_params[1][1]:.2f}s_{{parent}} + {child_params[1][2]:.2f}]",
    }
    L_a, k_a, N0_a, b_a, L_b, k_b, N0_b, b_b = aggregate_params
    formulas["Alpha Formula"] = f"\\alpha(N) = {b_a:.2f} + \\frac{{{L_a:.2f}}}{{1 + e^{{-{k_a:.2f}(N - {N0_a:.2f})}}}}"
    formulas["Beta Formula"] = f"\\beta(N) = {b_b:.2f} + \\frac{{{L_b:.2f}}}{{1 + e^{{-{k_b:.2f}(N - {N0_b:.2f})}}}}"
    show_formulas(formulas)

print("Using mapped scores")
actor_params = determine_actor_formula_parameters()
target_params = determine_target_formula_parameters()
association_params = determine_association_formula_parameters()
parent_params = determine_parent_formula_parameters()
child_params = determine_child_formula_parameters()
aggregate_params = determine_aggregate_formula_parameters()

print_formulas(actor_params, target_params, association_params, parent_params, child_params, aggregate_params)