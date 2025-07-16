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
                x0=[2.5, 2.5, 2.5],
                args=(X, y),
                bounds=bounds,
                loss='soft_l1',
                f_scale=0.3
            )
            params = result.x
            y_pred = formula(X, *params)
            mse = mean_squared_error(y, y_pred)
            soft_l1_loss = np.sum(np.square(y - y_pred)) / len(y)
            print(f"Optimal Params (lambda, w, b): {np.round(params, 4)}")
            print(f"MSE: {mse:.4f}")
            print(f"Soft L1 Loss: {soft_l1_loss:.4f}")
            return params
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

            params, _ = curve_fit(formula, X_filtered, y_filtered, bounds=bounds)
            y_pred = formula(X_filtered, *params)
            mse = mean_squared_error(y_filtered, y_pred)
            print(f"Optimal Params (lambda, w, b): {np.round(params, 4)}")
            print(f"MSE: {mse:.4f}")
            return params
        except Exception as e:
            print(f"Could not fit model: {e}")
            return None
    elif remove_outliers_method == "none":
        try:
            params, _ = curve_fit(formula, X, y, bounds=bounds)
            y_pred = formula(X, *params)
            mse = mean_squared_error(y, y_pred)
            print(f"Optimal Params (lambda, w, b): {np.round(params, 4)}")
            print(f"MSE: {mse:.4f}")
            return params
        except Exception as e:
            print(f"Could not fit model: {e}")
            return None
    return None


def create_action_df():
    action_df = df[df['item_type'] == 'compound_action'].copy()
    
    cols_to_ignore = ['submission_timestamp_utc']
    subset_cols = [col for col in action_df.columns if col not in cols_to_ignore]
    action_df.drop_duplicates(subset=subset_cols, inplace=True)
        
    action_df.drop_duplicates(inplace=True)

    action_data = []

    for _, row in action_df.iterrows():
        entities = ast.literal_eval(row['all_entities'])
        seed = row['seed']
        s_user_actor = associate_sentiment_integer(action_df[(action_df['seed'] == seed) & (action_df['entity'] == entities[0])]['user_sentiment_score'])
        s_user_target = associate_sentiment_integer(action_df[(action_df['seed'] == seed) & (action_df['entity'] == entities[1])]['user_sentiment_score'])

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

def actor_formula_v1(X, lambda_actor, w, b):
    s_init_actor, driver = X
    s_new = lambda_actor * s_init_actor + (1 - lambda_actor) * w * driver + b
    return np.tanh(s_new)

def actor_formula_v2(X, w_actor, w_driver, b):
    s_init_actor, driver = X
    s_new = w_actor * s_init_actor + w_driver * driver + b
    return np.tanh(s_new)


def create_association_df():
    association_df = df[df['item_type'] == 'compound_association'].copy()
    
    cols_to_ignore = ['submission_timestamp_utc']
    subset_cols = [col for col in association_df.columns if col not in cols_to_ignore]
    association_df.drop_duplicates(subset=subset_cols, inplace=True)
    
    association_df.drop_duplicates(inplace=True)

    association_data = []

    for _, row in association_df.iterrows():
        entities = ast.literal_eval(row['all_entities'])
        seed = row['seed']
        s_user_entity1 = associate_sentiment_integer(association_df[(association_df['seed'] == seed) & (association_df['entity'] == entities[0])]['user_sentiment_score'])
        s_user_entity2 = associate_sentiment_integer(association_df[(association_df['seed'] == seed) & (association_df['entity'] == entities[1])]['user_sentiment_score'])

        descriptor = ast.literal_eval(row['descriptor'])
        intensity = ast.literal_eval(row['intensity'])

        base_sentiments = []
        for desc, intens in zip(descriptor, intensity):
            base_sentiments.append(intensity_map_string.get(intens, 0) * sentiment_sign_map.get(desc, 0))
        
        association_data.append({
            's_init_entity1': base_sentiments[0],
            's_init_association': base_sentiments[1],
            's_init_entity2': base_sentiments[2],
            's_user_entity1': s_user_entity1,
            's_user_entity2': s_user_entity2
        })
    
    association_df = pd.DataFrame(association_data)
    association_df.drop_duplicates(inplace=True)

    return association_df