from typing import Literal
import pandas as pd
import ssl
import ast
import numpy as np
from scipy.optimize import curve_fit, minimize, least_squares
from sklearn.metrics import mean_squared_error
import re
from rich import print
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression
import inspect

ssl._create_default_https_context = ssl._create_unverified_context

url = 'https://docs.google.com/spreadsheets/d/1xAvDLhU0w-p2hAZ49QYM7-XBMQCek0zVYJWpiN1Mvn0/export?format=csv&gid=0'
df = pd.read_csv(url)

intensity_map_string = {
    'very': 0.85,  # Midpoint of VADER's [0.7, 1.0] range
    'medium': 0.6,   # Midpoint of [0.5, 0.7]
    'somewhat': 0.4, # Midpoint of [0.3, 0.5]
    'slightly': 0.2, # Midpoint of [0.1, 0.3]
    'neutral': 0.0 # Midpoint of [-0.1, 0.1]
}

intensity_map_integer = {
    4: 0.85,  # Midpoint of VADER's [0.7, 1.0] range
    3: 0.6,   # Midpoint of [0.5, 0.7]
    2: 0.4,   # Midpoint of [0.3, 0.5]
    1: 0.2,   # Midpoint of [0.1, 0.3]
    0: 0.0 # Midpoint of [-0.1, 0.1]
}

sentiment_sign_map = {'positive': 1, 'negative': -1}

def associate_sentiment_integer(integer):
    if not isinstance(integer, int):
        integer = int(integer)
    return intensity_map_integer.get(abs(integer), 0) * ((integer > 0) - (integer < 0))

def fit(formula, X, y, bounds, remove_outliers_method):
    if remove_outliers_method == "lsquares":
        def residuals(params, X_res, y_res):
            pred = formula(X_res, *params)
            return pred.flatten() - y_res.flatten()
        try:
            result = least_squares(
                residuals,
                x0=np.average(bounds),
                args=(X, y),
                bounds=bounds,
                loss='soft_l1',
                f_scale=0.3
            )
            params = result.x
        except Exception as e:
            print(f"Could not fit model: {e}")
            return None
    elif remove_outliers_method == "droptop":
        try:
            if len(y) < 10: 
                p = 80
            else:
                p = 90
            mask = y < np.percentile(y, p)
            y_filtered = y[mask]
            X_filtered = X[:, mask]
            
            if X_filtered.shape[1] < len(bounds[0]):
                print(f"Could not fit model: Not enough data points after outlier removal.")
                return None

            params, _ = curve_fit(formula, X_filtered, y_filtered, p0=np.average(bounds), bounds=bounds)
        except Exception as e:
            print(f"Could not fit model: {e}")
            return None
    elif remove_outliers_method == "none":
        try:
            params, _ = curve_fit(formula, X, y, p0=np.average(bounds), bounds=bounds)
        except Exception as e:
            print(f"Could not fit model: {e}")
            return None
    y_pred = formula(X, *params)
    mse = mean_squared_error(y, y_pred)
    soft_l1_loss = np.sum(np.square(y - y_pred)) / len(y)
    r2_loss = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
    return {
        'params': params,
        'mse': mse,
        'soft_l1_loss': soft_l1_loss,
        'r2_loss': r2_loss
    }

def add_score_interpretations(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    theoretical_intensity_map = {'very': 0.85, 'medium': 0.6, 'somewhat': 0.4, 'slightly': 0.2, 'neutral': 0.0}
    polarity_map = {'positive': 1, 'negative': -1, 'neutral': 0}

    def get_stimulus_sentiment(row):
        try:
            descriptor = ast.literal_eval(row['descriptor'])[0]
            intensity = ast.literal_eval(row['intensity'])[0]
            return polarity_map.get(descriptor, 0) * theoretical_intensity_map.get(intensity, 0)
        except (ValueError, SyntaxError, IndexError):
            return 0.0

    reference_df = df[df['packet_step'] == 1].copy()
    
    reference_df['ground_truth_stimulus_sentiment'] = reference_df.apply(get_stimulus_sentiment, axis=1)

    user_final_normalizers = {}

    for seed in df['seed'].unique():
        user_ref_data = reference_df[reference_df['seed'] == seed]
        
        if len(user_ref_data) >= 2:
            X = user_ref_data[['user_sentiment_score']]
            y = user_ref_data['ground_truth_stimulus_sentiment']
            model = LinearRegression().fit(X, y)
            m, c = model.coef_[0], model.intercept_
            user_final_normalizers[seed] = (m, c)
        else:
            user_final_normalizers[seed] = (1/4, 0)
            
    def apply_final_normalization(row):
        m, c = user_final_normalizers.get(row['seed'])
        return m * row['user_sentiment_score'] + c

    df['user_normalized_sentiment_scores'] = df.apply(apply_final_normalization, axis=1)

    df['user_sentiment_score_mapped'] = df['user_sentiment_score'].apply(associate_sentiment_integer)
    
    return df, user_final_normalizers

def fit_compound(formula, X, y, remove_outliers_method):
    sig = inspect.signature(formula)
    params = sig.parameters
    num_params = len(params)

    bounds = ([-5] * num_params, [5] * num_params)

    return fit(formula, X, y, bounds, remove_outliers_method)


df, _ = add_score_interpretations(df)


def create_action_df(score_key: Literal['user_sentiment_score', 'user_normalized_sentiment_scores', 'user_sentiment_score_mapped'] = 'user_sentiment_score_mapped') -> pd.DataFrame:
    action_df = df[df['item_type'] == 'compound_action'].copy()
    
    cols_to_ignore = ['submission_timestamp_utc']
    subset_cols = [col for col in action_df.columns if col not in cols_to_ignore]
    action_df.drop_duplicates(subset=subset_cols, inplace=True)
    
    action_data = []

    for _, row in action_df.iterrows():
        entities = ast.literal_eval(row['all_entities'])
        seed = row['seed']
        s_user_actor = action_df[(action_df['seed'] == seed) & (action_df['entity'] == entities[0])][score_key]
        s_user_target = action_df[(action_df['seed'] == seed) & (action_df['entity'] == entities[1])][score_key]

        if isinstance(s_user_actor, pd.Series):
            s_user_actor = s_user_actor.iloc[0] if not s_user_actor.empty else None
        if isinstance(s_user_target, pd.Series):
            s_user_target = s_user_target.iloc[0] if not s_user_target.empty else None

        descriptor = ast.literal_eval(row['descriptor'])
        intensity = ast.literal_eval(row['intensity'])

        base_sentiments = []
        for desc, intens in zip(descriptor, intensity):
            base_sentiments.append(intensity_map_string.get(intens, 0) * sentiment_sign_map.get(desc, 0))

        action_data.append({
            's_init_actor': base_sentiments[0],
            's_init_action': base_sentiments[1],
            's_init_target': base_sentiments[2],
            's_user_actor': s_user_actor,
            's_user_target': s_user_target
        })
    
    action_df = pd.DataFrame(action_data)
    action_df.drop_duplicates(inplace=True)

    return action_df

def create_association_df(score_key: Literal['user_sentiment_score', 'user_normalized_sentiment_scores', 'user_sentiment_score_mapped'] = 'user_sentiment_score_mapped') -> pd.DataFrame:
    association_df = df[df['item_type'] == 'compound_association'].copy()
    
    cols_to_ignore = ['submission_timestamp_utc']
    subset_cols = [col for col in association_df.columns if col not in cols_to_ignore]
    association_df.drop_duplicates(subset=subset_cols, inplace=True)
    
    association_data = []

    for _, row in association_df.iterrows():
        entities = ast.literal_eval(row['all_entities'])
        seed = row['seed']
        s_user_entity = association_df[(association_df['seed'] == seed) & (association_df['entity'] == entities[0])][score_key]
        s_user_other = association_df[(association_df['seed'] == seed) & (association_df['entity'] == entities[1])][score_key]

        if isinstance(s_user_entity, pd.Series):
            s_user_entity = s_user_entity.iloc[0] if not s_user_entity.empty else None
        if isinstance(s_user_other, pd.Series):
            s_user_other = s_user_other.iloc[0] if not s_user_other.empty else None


        descriptor = ast.literal_eval(row['descriptor'])
        intensity = ast.literal_eval(row['intensity'])

        base_sentiments = []
        for desc, intens in zip(descriptor, intensity):
            base_sentiments.append(intensity_map_string.get(intens, 0) * sentiment_sign_map.get(desc, 0))
        
        association_data.append({
            's_init_entity': base_sentiments[0],
            's_init_other': base_sentiments[1],
            's_user_entity': s_user_entity,
            's_user_other': s_user_other
        })
        association_data.append({
            's_init_entity': base_sentiments[1],
            's_init_other': base_sentiments[0],
            's_user_entity': s_user_other,
            's_user_other': s_user_entity
        })
    
    association_df = pd.DataFrame(association_data)
    association_df.drop_duplicates(inplace=True)

    return association_df

def create_belonging_df(score_key: Literal['user_sentiment_score', 'user_normalized_sentiment_scores', 'user_sentiment_score_mapped'] = 'user_sentiment_score_mapped') -> pd.DataFrame:
    belonging_df = df[df['item_type'] == 'compound_belonging'].copy()

    cols_to_ignore = ['submission_timestamp_utc']
    subset_cols = [col for col in belonging_df.columns if col not in cols_to_ignore]
    belonging_df.drop_duplicates(subset=subset_cols, inplace=True)

    belonging_data = []

    for _, row in belonging_df.iterrows():
        entities = ast.literal_eval(row['all_entities'])

        seed = row['seed']
        s_user_entity = belonging_df[(belonging_df['seed'] == seed) & (belonging_df['entity'] == entities[0])][score_key]
        s_user_other = belonging_df[(belonging_df['seed'] == seed) & (belonging_df['entity'] == entities[1])][score_key]

        if isinstance(s_user_entity, pd.Series):
            s_user_entity = s_user_entity.iloc[0] if not s_user_entity.empty else None
        if isinstance(s_user_other, pd.Series):
            s_user_other = s_user_other.iloc[0] if not s_user_other.empty else None

        descriptor = ast.literal_eval(row['descriptor'])
        intensity = ast.literal_eval(row['intensity'])

        base_sentiments = []
        for desc, intens in zip(descriptor, intensity):
            base_sentiments.append(intensity_map_string.get(intens, 0) * sentiment_sign_map.get(desc, 0))
        
        belonging_data.append({
            's_init_parent': base_sentiments[0],
            's_init_child': base_sentiments[1],
            's_user_parent': s_user_entity,
            's_user_child': s_user_other
        })

    belonging_df = pd.DataFrame(belonging_data)
    belonging_df.drop_duplicates(inplace=True)

    return belonging_df

def create_aggregate_df(score_key: Literal['user_sentiment_score', 'user_normalized_sentiment_scores', 'user_sentiment_score_mapped'] = 'user_sentiment_score_mapped') -> pd.DataFrame:
    aggregate_df = df[df['item_type'].str.contains('aggregate')].copy()

    cols_to_ignore = ['submission_timestamp_utc']
    subset_cols = [col for col in aggregate_df.columns if col not in cols_to_ignore]
    aggregate_df.drop_duplicates(subset=subset_cols, inplace=True)

    aggregate_data = []

    for _, group in aggregate_df.groupby("item_id"):

        descriptor = ast.literal_eval(group.iloc[0]['descriptor'])
        intensity = ast.literal_eval(group.iloc[0]['intensity'])
        base_sentiments = []
        for desc, intens in zip(descriptor, intensity):
            base_sentiments.append(intensity_map_string.get(intens, 0) * sentiment_sign_map.get(desc, 0))
        for i, row in enumerate(group.sort_values("packet_step").iterrows()):
            row = row[1]
            if i == 0:
                continue
            aggregate_data.append({
                's_inits': base_sentiments[:i+1],
                's_user': row[score_key],
            })
    
    aggregate_df = pd.DataFrame(aggregate_data)

    return aggregate_df


print(create_action_df('user_normalized_sentiment_scores'))
print(create_association_df('user_normalized_sentiment_scores'))
print(create_belonging_df('user_normalized_sentiment_scores'))
print(create_aggregate_df('user_normalized_sentiment_scores'))

