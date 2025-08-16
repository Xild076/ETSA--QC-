from typing import Literal, Callable
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import pandas as pd
import ssl
import ast
import numpy as np
from scipy.optimize import curve_fit, minimize, least_squares
from sklearn.metrics import mean_squared_error
import re
from rich import print
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression
import inspect
import warnings
from scipy.optimize import OptimizeWarning
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
import json
import os
import sys
from datetime import datetime
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

ssl._create_default_https_context = ssl._create_unverified_context

url = 'https://docs.google.com/spreadsheets/d/1xAvDLhU0w-p2hAZ49QYM7-XBMQCek0zVYJWpiN1Mvn0/export?format=csv&gid=0'
try:
    df = pd.read_csv(url)
except Exception as e:
    print(f"Error reading CSV from {url}: {e}")

intensity_map_string = {
    'very': 0.90,
    'strong': 0.70,
    'moderate': 0.45,
    'slight': 0.20,
    'neutral': 0.0
}

intensity_map_integer = {
    4: 0.90,
    3: 0.70,
    2: 0.45,
    1: 0.20,
    0: 0.00
}

sentiment_sign_map = {'positive': 1, 'negative': -1}

def associate_sentiment_integer(integer):
    logger.info("associate_sentiment_integer")
    if not isinstance(integer, int):
        integer = int(integer)
    return intensity_map_integer.get(abs(integer), 0) * ((integer > 0) - (integer < 0))

def fit(formula, X, y, bounds, remove_outliers_method:Literal['lsquares', 'droptop', 'none']='none'):
    logger.info("fit")
    x0 = [(l + u) / 2 for l, u in zip(bounds[0], bounds[1])]
    if remove_outliers_method == "lsquares":
        def residuals(params, X_res, y_res):
            pred = formula(X_res, *params)
            return pred.flatten() - y_res.flatten()
        try:
            result = least_squares(
                residuals,
                x0=x0,
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

            params, _ = curve_fit(formula, X_filtered, y_filtered, p0=x0, bounds=bounds)
        except Exception as e:
            print(f"Could not fit model: {e}")
            return None
    elif remove_outliers_method == "none":
        try:
            params, _ = curve_fit(formula, X, y, p0=x0, bounds=bounds)
        except Exception as e:
            print(f"Could not fit model: {e}")
            return None
    y_pred = formula(X, *params)
    mse = mean_squared_error(y, y_pred)
    soft_l1_loss = np.sum(np.square(y - y_pred)) / len(y)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot > 0:
        r2_loss = 1 - (ss_res / ss_tot)
    else:
        r2_loss = 1.0 if ss_res == 0 else 0.0

    return {
        'params': params,
        'mse': mse,
        'soft_l1_loss': soft_l1_loss,
        'r2_loss': r2_loss
    }

def add_score_interpretations(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Normalizes user sentiment scores based on their individual responses to baseline questions.
    This creates a more robust mapping from a user's personal scale (-4 to 4) to the
    theoretical sentiment scale (-1 to 1).
    """
    intensity_map = {'very': 0.85, 'strong': 0.6, 'moderate': 0.4, 'slight': 0.2}
    polarity_map = {'positive': 1, 'negative': -1}

    def get_ground_truth_sentiment(row):
        try:
            desc_list = ast.literal_eval(row.get('descriptor', '[]'))
            intens_list = ast.literal_eval(row.get('intensity', '[]'))
            if desc_list and intens_list:
                desc = desc_list[0] if isinstance(desc_list, list) else desc_list
                intens = intens_list[0] if isinstance(intens_list, list) else intens_list
                return polarity_map.get(desc, 0) * intensity_map.get(intens, 0.0)
        except (ValueError, SyntaxError, TypeError):
            pass
        return 0.0

    user_final_normalizers = {}
    unique_seeds = df['seed'].dropna().unique()

    for seed in unique_seeds:
        user_df = df[df['seed'] == seed]
        baseline_points = []

        calib_df = user_df[user_df['item_type'] == 'calibration']
        for _, row in calib_df.iterrows():
            x = row['user_sentiment_score']
            y = get_ground_truth_sentiment(row)
            baseline_points.append((x, y))

        packet_step1_df = user_df[user_df['packet_step'] == 1]
        for _, row in packet_step1_df.iterrows():
            x = row['user_sentiment_score']
            y = get_ground_truth_sentiment(row)
            baseline_points.append((x, y))
        unique_baseline_points = sorted(list(set(baseline_points)))
        
        m, c = 0.25, 0.0
        
        if len(unique_baseline_points) >= 2:
            X_points = np.array([p[0] for p in unique_baseline_points]).reshape(-1, 1)
            y_points = np.array([p[1] for p in unique_baseline_points])
            
            if np.std(X_points) > 0:
                try:
                    model = LinearRegression().fit(X_points, y_points)
                    m, c = model.coef_[0], model.intercept_
                except Exception:
                    pass
        user_final_normalizers[seed] = (m, c)

    def apply_norm(row):
        if pd.isna(row['seed']) or pd.isna(row['user_sentiment_score']):
            return np.nan
        m, c = user_final_normalizers.get(row['seed'], (0.25, 0.0))
        return m * float(row['user_sentiment_score']) + c

    df['user_normalized_sentiment_scores'] = df.apply(apply_norm, axis=1)
    
    if 'user_sentiment_score_mapped' not in df.columns:
        df['user_sentiment_score_mapped'] = df['user_sentiment_score'].apply(
            lambda x: associate_sentiment_integer(x) if pd.notna(x) else np.nan
        )
        
    return df, user_final_normalizers

def fit_compound(formula, X, y, remove_outliers_method:Literal['lsquares', 'droptop', 'none']='none'):
    sig = inspect.signature(formula)
    params = sig.parameters
    num_params = len(params) - 1
    if num_params <= 0:
        y_pred = np.array([formula((X[0, i], X[1, i])) for i in range(X.shape[1])], dtype=float)
        mse = mean_squared_error(y.astype(float), y_pred)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_loss = 1 - (ss_res / ss_tot) if ss_tot > 0 else (1.0 if ss_res == 0 else 0.0)
        return {'params': np.array([]), 'mse': mse, 'soft_l1_loss': np.sum(np.square(y - y_pred)) / len(y), 'r2_loss': r2_loss}
    bounds = ([0] + [-5] * (num_params - 1), [1] + [5] * (num_params - 1))
    return fit(formula, X, y, bounds, remove_outliers_method)

try:
    df, _ = add_score_interpretations(df)
except:
    print("Error adding score interpretations")

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

    for _, group in aggregate_df.groupby(['seed', 'item_type']):
        descriptor = ast.literal_eval(group.iloc[0]['descriptor'])
        intensity = ast.literal_eval(group.iloc[0]['intensity'])
        base_sentiments = []
        for desc, intens in zip(descriptor, intensity):
            base_sentiments.append(intensity_map_string.get(intens, 0) * sentiment_sign_map.get(desc, 0))

        for n in range(2, len(base_sentiments) + 1):
            score = group[group['packet_step'] == n][score_key].iloc[0]
            aggregate_data.append({
                'N': n,
                's_inits': base_sentiments[:n],
                's_user': score,
            })

    aggregate_df = pd.DataFrame(aggregate_data)

    return aggregate_df


def determine_actor_parameters(action_model_df: pd.DataFrame,
                                function: Callable,
                                remove_outlier_method: Literal['lsquares', 'droptop', 'none'] = 'none',
                                splits: Literal['none', 'driver', 'action_target'] = 'none',
                                print_process=False) -> pd.DataFrame:
    action_model_df['driver'] = action_model_df['s_init_action'] * action_model_df['s_init_target']

    pos_driver_df = action_model_df[action_model_df['driver'] > 0]
    neg_driver_df = action_model_df[action_model_df['driver'] <= 0]
    pos_action_pos_action_model = action_model_df[(action_model_df['s_init_action'] > 0) & (action_model_df['s_init_target'] > 0)]
    pos_action_neg_action_model = action_model_df[(action_model_df['s_init_action'] > 0) & (action_model_df['s_init_target'] <= 0)]
    neg_action_pos_action_model = action_model_df[(action_model_df['s_init_action'] <= 0) & (action_model_df['s_init_target'] > 0)]
    neg_action_neg_action_model = action_model_df[(action_model_df['s_init_action'] <= 0) & (action_model_df['s_init_target'] <= 0)]

    output = {}
    if splits == 'driver':
        if not pos_driver_df.empty:
            X = pos_driver_df[['s_init_actor', 'driver']].to_numpy().T
            y = pos_driver_df['s_user_actor'].to_numpy()
            pos_driver_params = fit_compound(function, X, y, remove_outlier_method)
            output['pos_driver_params'] = pos_driver_params
        if not neg_driver_df.empty:
            X = neg_driver_df[['s_init_actor', 'driver']].to_numpy().T
            y = neg_driver_df['s_user_actor'].to_numpy()
            neg_driver_params = fit_compound(function, X, y, remove_outlier_method)
            output['neg_driver_params'] = neg_driver_params
    elif splits == 'action_target':
        if not pos_action_pos_action_model.empty:
            X = pos_action_pos_action_model[['s_init_actor', 'driver']].to_numpy().T
            y = pos_action_pos_action_model['s_user_actor'].to_numpy()
            pos_pos_params = fit_compound(function, X, y, remove_outlier_method)
            output['pos_pos_params'] = pos_pos_params
        if not pos_action_neg_action_model.empty:
            X = pos_action_neg_action_model[['s_init_actor', 'driver']].to_numpy().T
            y = pos_action_neg_action_model['s_user_actor'].to_numpy()
            pos_neg_params = fit_compound(function, X, y, remove_outlier_method)
            output['pos_neg_params'] = pos_neg_params
        if not neg_action_pos_action_model.empty:
            X = neg_action_pos_action_model[['s_init_actor', 'driver']].to_numpy().T
            y = neg_action_pos_action_model['s_user_actor'].to_numpy()
            neg_pos_params = fit_compound(function, X, y, remove_outlier_method)
            output['neg_pos_params'] = neg_pos_params
        if not neg_action_neg_action_model.empty:
            X = neg_action_neg_action_model[['s_init_actor', 'driver']].to_numpy().T
            y = neg_action_neg_action_model['s_user_actor'].to_numpy()
            neg_neg_params = fit_compound(function, X, y, remove_outlier_method)
            output['neg_neg_params'] = neg_neg_params
    else:
        X = action_model_df[['s_init_actor', 'driver']].to_numpy().T
        y = action_model_df['s_user_actor'].to_numpy()
        output['params'] = fit_compound(function, X, y, remove_outlier_method)
    if print_process:
        for key, value in output.items():
            if value is not None:
                print(f"--- {key}: ---")
                print(f"Parameters {value['params'] if isinstance(value, dict) else value}")
                print(f"MSE: {value['mse']}")
                print(f"Soft L1 Loss: {value['soft_l1_loss']}")
                print(f"R2 Loss: {value['r2_loss']}")
            else:
                print(f"{key}: No data available")
    return output

def actor_skeleton_formula(s_actor, s_action, s_target, split: Literal['driver', 'action_target', 'none'], function: Callable, fitted: dict):
    driver = s_action * s_target
    key = 'params'
    if split == 'driver':
        key = 'pos_driver_params' if driver > 0 else 'neg_driver_params'
    elif split == 'action_target':
        if s_action > 0 and s_target > 0:
            key = 'pos_pos_params'
        elif s_action > 0 and s_target <= 0:
            key = 'pos_neg_params'
        elif s_action <= 0 and s_target > 0:
            key = 'neg_pos_params'
        else:
            key = 'neg_neg_params'
    selected = fitted.get(key) or fitted.get('params')
    if selected is None:
        return np.nan
    params_arr = selected['params'] if isinstance(selected, dict) and 'params' in selected else selected
    return function((s_actor, driver), *params_arr)


def determine_target_parameters(action_model_df: pd.DataFrame,
                                function: Callable,
                                remove_outlier_method: Literal['lsquares', 'droptop', 'none'] = 'none',
                                splits: Literal['none', 'action'] = 'none',
                                print_process=False) -> pd.DataFrame:
    pos_action_df = action_model_df[action_model_df['s_init_action'] > 0]
    neg_action_df = action_model_df[action_model_df['s_init_action'] <= 0]

    output = {}
    if splits == 'action':
        if not pos_action_df.empty:
            X = pos_action_df[['s_init_target', 's_init_action']].to_numpy().T
            y = pos_action_df['s_user_target'].to_numpy()
            pos_action_params = fit_compound(function, X, y, remove_outlier_method)
            output['pos_action_params'] = pos_action_params
        if not neg_action_df.empty:
            X = neg_action_df[['s_init_target', 's_init_action']].to_numpy().T
            y = neg_action_df['s_user_target'].to_numpy()
            neg_action_params = fit_compound(function, X, y, remove_outlier_method)
            output['neg_action_params'] = neg_action_params
    else:
        X = action_model_df[['s_init_target', 's_init_action']].to_numpy().T
        y = action_model_df['s_user_target'].to_numpy()
        output['params'] = fit_compound(function, X, y, remove_outlier_method)
    if print_process:
        for key, value in output.items():
            if value is not None:
                print(f"--- {key}: ---")
                print(f"Parameters {value['params'] if isinstance(value, dict) else value}")
                print(f"MSE: {value['mse']}")
                print(f"Soft L1 Loss: {value['soft_l1_loss']}")
                print(f"R2 Loss: {value['r2_loss']}")
            else:
                print(f"{key}: No data available")
    return output

def target_skeleton_formula(s_target, s_action, split: Literal['action', 'none'], function: Callable, fitted: dict):
    key = 'params'
    if split == 'action':
        key = 'pos_action_params' if s_action > 0 else 'neg_action_params'
    selected = fitted.get(key) or fitted.get('params')
    if selected is None:
        return np.nan
    params_arr = selected['params'] if isinstance(selected, dict) and 'params' in selected else selected
    return function((s_target, s_action), *params_arr)


def determine_association_parameters(association_model_df: pd.DataFrame,
                                        function: Callable,
                                        remove_outlier_method: Literal['lsquares', 'droptop', 'none'] = 'none',
                                        splits: Literal['none', 'other', 'entity_other'] = 'none',
                                        print_process=False) -> pd.DataFrame:
    pos_entity_df = association_model_df[association_model_df['s_init_other'] > 0]
    neg_entity_df = association_model_df[association_model_df['s_init_other'] <= 0]

    pos_entity_pos_other_df = association_model_df[(association_model_df['s_init_entity'] > 0) & (association_model_df['s_init_other'] > 0)]
    pos_entity_neg_other_df = association_model_df[(association_model_df['s_init_entity'] > 0) & (association_model_df['s_init_other'] <= 0)]
    neg_entity_neg_other_df = association_model_df[(association_model_df['s_init_entity'] <= 0) & (association_model_df['s_init_other'] <= 0)]
    neg_entity_pos_other_df = association_model_df[(association_model_df['s_init_entity'] <= 0) & (association_model_df['s_init_other'] > 0)]


    output = {}
    if splits == 'other':
        if not pos_entity_df.empty:
            X = pos_entity_df[['s_init_entity', 's_init_other']].to_numpy().T
            y = pos_entity_df['s_user_entity'].to_numpy()
            pos_params = fit_compound(function, X, y, remove_outlier_method)
            output['pos_params'] = pos_params
        if not neg_entity_df.empty:
            X = neg_entity_df[['s_init_entity', 's_init_other']].to_numpy().T
            y = neg_entity_df['s_user_entity'].to_numpy()
            neg_params = fit_compound(function, X, y, remove_outlier_method)
            output['neg_params'] = neg_params
    elif splits == 'entity_other':
        if not pos_entity_pos_other_df.empty:
            X = pos_entity_pos_other_df[['s_init_entity', 's_init_other']].to_numpy().T
            y = pos_entity_pos_other_df['s_user_entity'].to_numpy()
            pos_params = fit_compound(function, X, y, remove_outlier_method)
            output['pos_pos_params'] = pos_params
        if not neg_entity_pos_other_df.empty:
            X = neg_entity_pos_other_df[['s_init_entity', 's_init_other']].to_numpy().T
            y = neg_entity_pos_other_df['s_user_entity'].to_numpy()
            neg_params = fit_compound(function, X, y, remove_outlier_method)
            output['neg_pos_params'] = neg_params
        if not pos_entity_neg_other_df.empty:
            X = pos_entity_neg_other_df[['s_init_entity', 's_init_other']].to_numpy().T
            y = pos_entity_neg_other_df['s_user_entity'].to_numpy()
            pos_neg_params = fit_compound(function, X, y, remove_outlier_method)
            output['pos_neg_params'] = pos_neg_params
        if not neg_entity_neg_other_df.empty:
            X = neg_entity_neg_other_df[['s_init_entity', 's_init_other']].to_numpy().T
            y = neg_entity_neg_other_df['s_user_entity'].to_numpy()
            neg_neg_params = fit_compound(function, X, y, remove_outlier_method)
            output['neg_neg_params'] = neg_neg_params
    else:
        X = association_model_df[['s_init_entity', 's_init_other']].to_numpy().T
        y = association_model_df['s_user_entity'].to_numpy()
        output['params'] = fit_compound(function, X, y, remove_outlier_method)
    if print_process:
        for key, value in output.items():
            if value is not None:
                print(f"--- {key}: ---")
                print(f"Parameters {value['params'] if isinstance(value, dict) else value}")
                print(f"MSE: {value['mse']}")
                print(f"Soft L1 Loss: {value['soft_l1_loss']}")
                print(f"R2 Loss: {value['r2_loss']}")
            else:
                print(f"{key}: No data available")
    return output

def association_skeleton_formula(s_entity, s_other, split: Literal['other', 'entity_other', 'none'], function: Callable, fitted: dict):
    key = 'params'
    if split == 'other':
        key = 'pos_params' if s_other > 0 else 'neg_params'
    elif split == 'entity_other':
        if s_entity > 0 and s_other > 0:
            key = 'pos_pos_params'
        elif s_entity > 0 and s_other <= 0:
            key = 'pos_neg_params'
        elif s_entity <= 0 and s_other > 0:
            key = 'neg_pos_params'
        else:
            key = 'neg_neg_params'
    selected = fitted.get(key) or fitted.get('params')
    if selected is None:
        return np.nan
    params_arr = selected['params'] if isinstance(selected, dict) and 'params' in selected else selected
    return function((s_entity, s_other), *params_arr)


def determine_parent_parameters(belonging_model_df: pd.DataFrame,
                                    function: Callable,
                                    remove_outlier_method: Literal['lsquares', 'droptop', 'none'] = 'none',
                                    splits: Literal['none', 'parent_child', 'child'] = 'none',
                                    print_process=False) -> pd.DataFrame:
    pos_parent_df = belonging_model_df[belonging_model_df['s_init_child'] > 0]
    neg_parent_df = belonging_model_df[belonging_model_df['s_init_child'] <= 0]

    pos_parent_neg_child_df = belonging_model_df[(belonging_model_df['s_init_parent'] > 0) & (belonging_model_df['s_init_child'] <= 0)]
    pos_parent_pos_child_df = belonging_model_df[(belonging_model_df['s_init_parent'] > 0) & (belonging_model_df['s_init_child'] > 0)]
    neg_parent_neg_child_df = belonging_model_df[(belonging_model_df['s_init_parent'] <= 0) & (belonging_model_df['s_init_child'] <= 0)]
    neg_parent_pos_child_df = belonging_model_df[(belonging_model_df['s_init_parent'] <= 0) & (belonging_model_df['s_init_child'] > 0)]

    output = {}
    if splits == 'child':
        if not pos_parent_df.empty:
            X = pos_parent_df[['s_init_parent', 's_init_child']].to_numpy().T
            y = pos_parent_df['s_user_parent'].to_numpy()
            pos_params = fit_compound(function, X, y, remove_outlier_method)
            output['pos_params'] = pos_params
        if not neg_parent_df.empty:
            X = neg_parent_df[['s_init_parent', 's_init_child']].to_numpy().T
            y = neg_parent_df['s_user_parent'].to_numpy()
            neg_params = fit_compound(function, X, y, remove_outlier_method)
            output['neg_params'] = neg_params
    elif splits == 'parent_child':
        if not pos_parent_neg_child_df.empty:
            X = pos_parent_neg_child_df[['s_init_parent', 's_init_child']].to_numpy().T
            y = pos_parent_neg_child_df['s_user_parent'].to_numpy()
            pos_neg_params = fit_compound(function, X, y, remove_outlier_method)
            output['pos_neg_params'] = pos_neg_params
        if not neg_parent_neg_child_df.empty:
            X = neg_parent_neg_child_df[['s_init_parent', 's_init_child']].to_numpy().T
            y = neg_parent_neg_child_df['s_user_parent'].to_numpy()
            neg_neg_params = fit_compound(function, X, y, remove_outlier_method)
            output['neg_neg_params'] = neg_neg_params
        if not pos_parent_pos_child_df.empty:
            X = pos_parent_pos_child_df[['s_init_parent', 's_init_child']].to_numpy().T
            y = pos_parent_pos_child_df['s_user_parent'].to_numpy()
            pos_pos_params = fit_compound(function, X, y, remove_outlier_method)
            output['pos_pos_params'] = pos_pos_params
        if not neg_parent_pos_child_df.empty:
            X = neg_parent_pos_child_df[['s_init_parent', 's_init_child']].to_numpy().T
            y = neg_parent_pos_child_df['s_user_parent'].to_numpy()
            neg_pos_params = fit_compound(function, X, y, remove_outlier_method)
            output['neg_pos_params'] = neg_pos_params
    else:
        X = belonging_model_df[['s_init_parent', 's_init_child']].to_numpy().T
        y = belonging_model_df['s_user_parent'].to_numpy()
        output['params'] = fit_compound(function, X, y, remove_outlier_method)
    if print_process:
        for key, value in output.items():
            if value is not None:
                print(f"--- {key}: ---")
                print(f"Parameters {value['params'] if isinstance(value, dict) else value}")
                print(f"MSE: {value['mse']}")
                print(f"Soft L1 Loss: {value['soft_l1_loss']}")
                print(f"R2 Loss: {value['r2_loss']}")
            else:
                print(f"{key}: No data available")
    return output

def determine_child_parameters(belonging_model_df: pd.DataFrame,
                                    function: Callable,
                                    remove_outlier_method: Literal['lsquares', 'droptop', 'none'] = 'none',
                                    splits: Literal['none', 'parent_child', 'parent'] = 'none',
                                    print_process=False) -> pd.DataFrame:
    pos_child_df = belonging_model_df[belonging_model_df['s_init_parent'] > 0]
    neg_child_df = belonging_model_df[belonging_model_df['s_init_parent'] <= 0]

    pos_child_neg_parent_df = belonging_model_df[(belonging_model_df['s_init_child'] > 0) & (belonging_model_df['s_init_parent'] <= 0)]
    pos_child_pos_parent_df = belonging_model_df[(belonging_model_df['s_init_child'] > 0) & (belonging_model_df['s_init_parent'] > 0)]
    neg_child_neg_parent_df = belonging_model_df[(belonging_model_df['s_init_child'] <= 0) & (belonging_model_df['s_init_parent'] <= 0)]
    neg_child_pos_parent_df = belonging_model_df[(belonging_model_df['s_init_child'] <= 0) & (belonging_model_df['s_init_parent'] > 0)]

    output = {}
    if splits == 'parent':
        if not pos_child_df.empty:
            X = pos_child_df[['s_init_child', 's_init_parent']].to_numpy().T
            y = pos_child_df['s_user_child'].to_numpy()
            pos_params = fit_compound(function, X, y, remove_outlier_method)
            output['pos_params'] = pos_params
        if not neg_child_df.empty:
            X = neg_child_df[['s_init_child', 's_init_parent']].to_numpy().T
            y = neg_child_df['s_user_child'].to_numpy()
            neg_params = fit_compound(function, X, y, remove_outlier_method)
            output['neg_params'] = neg_params
    elif splits == 'parent_child':
        if not pos_child_neg_parent_df.empty:
            X = pos_child_neg_parent_df[['s_init_child', 's_init_parent']].to_numpy().T
            y = pos_child_neg_parent_df['s_user_child'].to_numpy()
            pos_neg_params = fit_compound(function, X, y, remove_outlier_method)
            output['pos_neg_params'] = pos_neg_params
        if not neg_child_neg_parent_df.empty:
            X = neg_child_neg_parent_df[['s_init_child', 's_init_parent']].to_numpy().T
            y = neg_child_neg_parent_df['s_user_child'].to_numpy()
            neg_neg_params = fit_compound(function, X, y, remove_outlier_method)
            output['neg_neg_params'] = neg_neg_params
        if not pos_child_pos_parent_df.empty:
            X = pos_child_pos_parent_df[['s_init_child', 's_init_parent']].to_numpy().T
            y = pos_child_pos_parent_df['s_user_child'].to_numpy()
            pos_pos_params = fit_compound(function, X, y, remove_outlier_method)
            output['pos_pos_params'] = pos_pos_params
        if not neg_child_pos_parent_df.empty:
            X = neg_child_pos_parent_df[['s_init_child', 's_init_parent']].to_numpy().T
            y = neg_child_pos_parent_df['s_user_child'].to_numpy()
            neg_pos_params = fit_compound(function, X, y, remove_outlier_method)
            output['neg_pos_params'] = neg_pos_params
    else:
        X = belonging_model_df[['s_init_child', 's_init_parent']].to_numpy().T
        y = belonging_model_df['s_user_child'].to_numpy()
        output['params'] = fit_compound(function, X, y, remove_outlier_method)
    if print_process:
        for key, value in output.items():
            if value is not None:
                print(f"--- {key}: ---")
                print(f"Parameters {value['params'] if isinstance(value, dict) else value}")
                print(f"MSE: {value['mse']}")
                print(f"Soft L1 Loss: {value['soft_l1_loss']}")
                print(f"R2 Loss: {value['r2_loss']}")
            else:
                print(f"{key}: No data available")
    return output

def parent_skeleton_formula(s_parent, s_child, split: Literal['parent_child', 'child', 'none'], function: Callable, fitted: dict):
    key = 'params'
    if split == 'parent_child':
        if s_parent > 0 and s_child > 0:
            key = 'pos_pos_params'
        elif s_parent > 0 and s_child <= 0:
            key = 'pos_neg_params'
        elif s_parent <= 0 and s_child > 0:
            key = 'neg_pos_params'
        else:
            key = 'neg_neg_params'
    elif split == 'child':
        key = 'pos_params' if s_child > 0 else 'neg_params'
    selected = fitted.get(key) or fitted.get('params')
    if selected is None:
        return np.nan
    params_arr = selected['params'] if isinstance(selected, dict) and 'params' in selected else selected
    return function((s_parent, s_child), *params_arr)

def child_skeleton_formula(s_child, s_parent, split: Literal['parent_child', 'parent', 'none'], function: Callable, fitted: dict):
    key = 'params'
    if split == 'parent_child':
        if s_child > 0 and s_parent > 0:
            key = 'pos_pos_params'
        elif s_child > 0 and s_parent <= 0:
            key = 'pos_neg_params'
        elif s_child <= 0 and s_parent > 0:
            key = 'neg_pos_params'
        else:
            key = 'neg_neg_params'
    elif split == 'parent':
        key = 'pos_params' if s_parent > 0 else 'neg_params'
    selected = fitted.get(key) or fitted.get('params')
    if selected is None:
        return np.nan
    params_arr = selected['params'] if isinstance(selected, dict) and 'params' in selected else selected
    return function((s_child, s_parent), *params_arr)


def calculate_weights(n, alpha, beta):
    k = np.arange(1, n + 1)
    alpha, beta = max(alpha, 0.01), max(beta, 0.01)
    numerator = (k**(alpha - 1)) * ((n - k + 1)**(beta - 1))
    denominator = np.sum(numerator)
    if denominator == 0:
        return np.full(n, 1/n)
    return numerator / denominator

def logistic_function(N, L, k, N0, b):
    return b + L / (1 + np.exp(-k * (N - N0)))

def aggregate_formula(s_inits, params):
    alpha, beta = params
    n = len(s_inits)
    weights = calculate_weights(n, alpha, beta)
    predicted_sentiment = np.sum(weights * np.array(s_inits))
    return predicted_sentiment

def aggregate_formula_dynamic(s_inits, params):
    m_a, c_a, m_b, c_b = params
    n = len(s_inits)
    alpha_n = m_a * n + c_a
    beta_n = m_b * n + c_b
    weights = calculate_weights(n, alpha_n, beta_n)
    predicted_sentiment = np.sum(weights * np.array(s_inits))
    return predicted_sentiment

def aggregate_formula_logistic(s_inits, params):
    L_a, k_a, N0_a, b_a, L_b, k_b, N0_b, b_b = params
    N = len(s_inits)
    alpha = logistic_function(N, L_a, k_a, N0_a, b_a)
    beta = logistic_function(N, L_b, k_b, N0_b, b_b)
    weights = calculate_weights(N, alpha, beta)
    predicted_sentiment = np.sum(weights * np.array(s_inits))
    return predicted_sentiment

def _calculate_error(s_user_predicted, s_user_actual, loss_type='mse'):
    if loss_type == 'softl1':
        delta = 1.0
        return 2 * (np.sqrt(1 + ((s_user_predicted - s_user_actual)**2) / delta**2) - 1)
    return (s_user_predicted - s_user_actual) ** 2

def aggregate_error_mse(params, data):
    total_error = 0
    for _, row in data.iterrows():
        s_user_predicted = aggregate_formula(row['s_inits'], params)
        total_error += _calculate_error(s_user_predicted, row['s_user'], 'mse')
    return total_error

def aggregate_error_softl1(params, data):
    total_error = 0
    for _, row in data.iterrows():
        s_user_predicted = aggregate_formula(row['s_inits'], params)
        total_error += _calculate_error(s_user_predicted, row['s_user'], 'softl1')
    return total_error

def aggregate_error_dynamic_mse(params, data):
    total_error = 0
    for _, row in data.iterrows():
        s_user_predicted = aggregate_formula_dynamic(row['s_inits'], params)
        total_error += _calculate_error(s_user_predicted, row['s_user'], 'mse')
    return total_error

def aggregate_error_dynamic_softl1(params, data):
    total_error = 0
    for _, row in data.iterrows():
        s_user_predicted = aggregate_formula_dynamic(row['s_inits'], params)
        total_error += _calculate_error(s_user_predicted, row['s_user'], 'softl1')
    return total_error

def aggregate_error_logistic_mse(params, data):
    total_error = 0
    for _, row in data.iterrows():
        s_user_predicted = aggregate_formula_logistic(row['s_inits'], params)
        total_error += _calculate_error(s_user_predicted, row['s_user'], 'mse')
    return total_error

def aggregate_error_logistic_softl1(params, data):
    total_error = 0
    for _, row in data.iterrows():
        s_user_predicted = aggregate_formula_logistic(row['s_inits'], params)
        total_error += _calculate_error(s_user_predicted, row['s_user'], 'softl1')
    return total_error

all_loss_functions = {
    'aggregate_error_mse': aggregate_error_mse,
    'aggregate_error_softl1': aggregate_error_softl1,
    'aggregate_error_dynamic_mse': aggregate_error_dynamic_mse,
    'aggregate_error_dynamic_softl1': aggregate_error_dynamic_softl1,
    'aggregate_error_logistic_mse': aggregate_error_logistic_mse,
    'aggregate_error_logistic_softl1': aggregate_error_logistic_softl1,
}

def determine_aggregate_parameters(aggregate_df: pd.DataFrame,
        loss_function: Callable = aggregate_error_dynamic_softl1,
        print_process: bool = False
    ) -> dict:

    loss_name = loss_function.__name__

    if "logistic" in loss_name:
        initial_guess = [-10, 1, 3, 10, 10, 1, 3, 1]
        param_bounds = [(-20, 20), (0.1, 5), (1, 10), (0.01, 20),
                        (-20, 20), (0.1, 5), (1, 10), (0.01, 20)]
    elif "dynamic" in loss_name:
        initial_guess = [0.0, 1.0, 0.0, 1.0]
        param_bounds = [(-5, 5), (-5, 5), (-5, 5), (-5, 5)]
    else:
        initial_guess = [1.0, 1.0]
        param_bounds = [(-5, 5), (-5, 5)]

    result = minimize(
        fun=loss_function,
        x0=initial_guess,
        args=(aggregate_df,),
        bounds=param_bounds,
        method='L-BFGS-B'
    )

    if print_process:
        if result.success:
            print("Aggregate parameters found successfully.")
            print(f"Model Type: {loss_name}")
            print(f"Parameters: {result.x}")
            if 'mse' in loss_name:
                mse_loss_name = loss_name
                softl1_loss_name = loss_name.replace('mse', 'softl1')
            else:
                mse_loss_name = loss_name.replace('softl1', 'mse')
                softl1_loss_name = loss_name

            mse_func = all_loss_functions[mse_loss_name]
            softl1_func = all_loss_functions[softl1_loss_name]

            total_mse = mse_func(result.x, aggregate_df)
            total_softl1 = softl1_func(result.x, aggregate_df)

            print(f"Final Mean Squared Error (MSE): {total_mse / len(aggregate_df):.4f}")
            print(f"Final Total SoftL1 Loss: {total_softl1:.4f}")
        else:
            print("Aggregate parameter optimization failed.")
            print(result.message)

    if "logistic" in loss_name:
        func = aggregate_formula_logistic
        model_type = "logistic"
    elif "dynamic" in loss_name:
        func = aggregate_formula_dynamic
        model_type = "dynamic"
    else:
        func = aggregate_formula
        model_type = "normal"

    return {
        'params': result.x,
        'loss': result.fun / len(aggregate_df),
        'success': result.success,
        'function': func,
        'model_type': model_type
    }

def aggregate_skeleton_formula(s_inits, formula, params):
    return formula(s_inits, params)

CONFIG = {
    "test_train_split": 0.8
}

def _safe_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    logger.info("_safe_mse")
    mask = ~np.isnan(y_pred)
    if mask.sum() == 0:
        return float('inf')
    return mean_squared_error(y_true[mask], y_pred[mask])


def test_actor_parameters(score_key: str = 'user_sentiment_score_mapped') -> dict:
    logger.info("test_actor_parameters")
    from formulas import actor_formula_v1, actor_formula_v2, null_identity, null_avg, null_linear

    action_df = create_action_df(score_key)
    if action_df.empty:
        print("No action data available.")
        return {}

    action_df = action_df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_train = int(len(action_df) * CONFIG["test_train_split"])
    train_df = action_df.iloc[:n_train].copy()
    test_df = action_df.iloc[n_train:].copy()

    candidates = [
        {"function": actor_formula_v1},
        {"function": actor_formula_v2},
        {"function": null_identity},
        {"function": null_avg},
        {"function": null_linear},
    ]
    remove_outliers = ["none", "lsquares", "droptop"]
    splits = ["none", "driver", "action_target"]

    results = []
    for cand in candidates:
        func = cand["function"]
        for ro in remove_outliers:
            for sp in splits:
                fitted = determine_actor_parameters(train_df, func, ro, sp, print_process=False)
                preds = []
                for _, r in test_df.iterrows():
                    preds.append(
                        actor_skeleton_formula(r['s_init_actor'], r['s_init_action'], r['s_init_target'], sp, func, fitted)
                    )
                preds = np.array(preds, dtype=float)
                mse = _safe_mse(test_df['s_user_actor'].to_numpy(dtype=float), preds)
                results.append({
                    "model": func.__name__,
                    "split": sp,
                    "remove_outliers": ro,
                    "mse": mse,
                    "score_key": score_key,
                    "fitted": fitted,
                })
                print(f"actor | {func.__name__} | split={sp} | outliers={ro} | MSE={mse:.4f}")
    if not results:
        return {}
    best = min(results, key=lambda x: x["mse"])
    print(f"Best actor config: {best}")
    return {"best": best, "all": results}

def test_target_parameters(score_key: str = 'user_sentiment_score_mapped') -> dict:
    logger.info("test_target_parameters")
    from formulas import target_formula_v1, target_formula_v2, null_identity, null_avg, null_linear

    action_df = create_action_df(score_key)
    if action_df.empty:
        print("No action data available for target model.")
        return {}

    action_df = action_df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_train = int(len(action_df) * CONFIG["test_train_split"])
    train_df = action_df.iloc[:n_train].copy()
    test_df = action_df.iloc[n_train:].copy()

    candidates = [
        {"function": target_formula_v1},
        {"function": target_formula_v2},
        {"function": null_identity},
        {"function": null_avg},
        {"function": null_linear},
    ]
    remove_outliers = ["none", "lsquares", "droptop"]
    splits = ["none", "action"]

    results = []
    for cand in candidates:
        func = cand["function"]
        for ro in remove_outliers:
            for sp in splits:
                fitted = determine_target_parameters(train_df, func, ro, sp, print_process=False)
                preds = []
                for _, r in test_df.iterrows():
                    preds.append(
                        target_skeleton_formula(r['s_init_target'], r['s_init_action'], sp, func, fitted)
                    )
                preds = np.array(preds, dtype=float)
                mse = _safe_mse(test_df['s_user_target'].to_numpy(dtype=float), preds)
                results.append({
                    "model": func.__name__,
                    "split": sp,
                    "remove_outliers": ro,
                    "mse": mse,
                    "score_key": score_key,
                    "fitted": fitted,
                })
                print(f"target | {func.__name__} | split={sp} | outliers={ro} | MSE={mse:.4f}")
    if not results:
        return {}
    best = min(results, key=lambda x: x["mse"])
    print(f"Best target config: {best}")
    return {"best": best, "all": results}

def test_association_parameters(score_key: str = 'user_sentiment_score_mapped') -> dict:
    logger.info("test_association_parameters")
    from formulas import assoc_formula_v1, assoc_formula_v2, null_identity, null_avg, null_linear

    assoc_df = create_association_df(score_key)
    if assoc_df.empty:
        print("No association data available.")
        return {}

    assoc_df = assoc_df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_train = int(len(assoc_df) * CONFIG["test_train_split"])
    train_df = assoc_df.iloc[:n_train].copy()
    test_df = assoc_df.iloc[n_train:].copy()

    candidates = [
        {"function": assoc_formula_v1},
        {"function": assoc_formula_v2},
        {"function": null_identity},
        {"function": null_avg},
        {"function": null_linear},
    ]
    remove_outliers = ["none", "lsquares", "droptop"]
    splits = ["none", "other", "entity_other"]

    results = []
    for cand in candidates:
        func = cand["function"]
        for ro in remove_outliers:
            for sp in splits:
                fitted = determine_association_parameters(train_df, func, ro, sp, print_process=False)
                preds = []
                for _, r in test_df.iterrows():
                    preds.append(
                        association_skeleton_formula(r['s_init_entity'], r['s_init_other'], sp, func, fitted)
                    )
                preds = np.array(preds, dtype=float)
                mse = _safe_mse(test_df['s_user_entity'].to_numpy(dtype=float), preds)
                results.append({
                    "model": func.__name__,
                    "split": sp,
                    "remove_outliers": ro,
                    "mse": mse,
                    "score_key": score_key,
                    "fitted": fitted,
                })
                print(f"assoc | {func.__name__} | split={sp} | outliers={ro} | MSE={mse:.4f}")
    if not results:
        return {}
    best = min(results, key=lambda x: x["mse"])
    print(f"Best association config: {best}")
    return {"best": best, "all": results}

def test_parent_parameters(score_key: str = 'user_sentiment_score_mapped') -> dict:
    logger.info("test_parent_parameters")
    from formulas import belong_formula_v1, belong_formula_v2, null_identity, null_avg, null_linear

    bel_df = create_belonging_df(score_key)
    if bel_df.empty:
        print("No belonging data available for parent model.")
        return {}

    bel_df = bel_df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_train = int(len(bel_df) * CONFIG["test_train_split"])
    train_df = bel_df.iloc[:n_train].copy()
    test_df = bel_df.iloc[n_train:].copy()

    candidates = [
        {"function": belong_formula_v1},
        {"function": belong_formula_v2},
        {"function": null_identity},
        {"function": null_avg},
        {"function": null_linear},
    ]
    remove_outliers = ["none", "lsquares", "droptop"]
    splits = ["none", "child", "parent_child"]

    results = []
    for cand in candidates:
        func = cand["function"]
        for ro in remove_outliers:
            for sp in splits:
                fitted = determine_parent_parameters(train_df, func, ro, sp, print_process=False)
                preds = []
                for _, r in test_df.iterrows():
                    preds.append(
                        parent_skeleton_formula(r['s_init_parent'], r['s_init_child'], sp, func, fitted)
                    )
                preds = np.array(preds, dtype=float)
                mse = _safe_mse(test_df['s_user_parent'].to_numpy(dtype=float), preds)
                results.append({
                    "model": func.__name__,
                    "split": sp,
                    "remove_outliers": ro,
                    "mse": mse,
                    "score_key": score_key,
                    "fitted": fitted,
                })
                print(f"parent | {func.__name__} | split={sp} | outliers={ro} | MSE={mse:.4f}")
    if not results:
        return {}
    best = min(results, key=lambda x: x["mse"])
    print(f"Best parent config: {best}")
    return {"best": best, "all": results}

def test_child_parameters(score_key: str = 'user_sentiment_score_mapped') -> dict:
    logger.info("test_child_parameters")
    from formulas import belong_formula_v1, belong_formula_v2, null_identity, null_avg, null_linear

    bel_df = create_belonging_df(score_key)
    if bel_df.empty:
        print("No belonging data available for child model.")
        return {}

    bel_df = bel_df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_train = int(len(bel_df) * CONFIG["test_train_split"])
    train_df = bel_df.iloc[:n_train].copy()
    test_df = bel_df.iloc[n_train:].copy()

    candidates = [
        {"function": belong_formula_v1},
        {"function": belong_formula_v2},
        {"function": null_identity},
        {"function": null_avg},
        {"function": null_linear},
    ]
    remove_outliers = ["none", "lsquares", "droptop"]
    splits = ["none", "parent", "parent_child"]

    results = []
    for cand in candidates:
        func = cand["function"]
        for ro in remove_outliers:
            for sp in splits:
                fitted = determine_child_parameters(train_df, func, ro, sp, print_process=False)
                preds = []
                for _, r in test_df.iterrows():
                    preds.append(
                        child_skeleton_formula(r['s_init_child'], r['s_init_parent'], sp, func, fitted)
                    )
                preds = np.array(preds, dtype=float)
                mse = _safe_mse(test_df['s_user_child'].to_numpy(dtype=float), preds)
                results.append({
                    "model": func.__name__,
                    "split": sp,
                    "remove_outliers": ro,
                    "mse": mse,
                    "score_key": score_key,
                    "fitted": fitted,
                })
                print(f"child | {func.__name__} | split={sp} | outliers={ro} | MSE={mse:.4f}")
    if not results:
        return {}
    best = min(results, key=lambda x: x["mse"])
    print(f"Best child config: {best}")
    return {"best": best, "all": results}

def test_aggregate_parameters(score_key: str = 'user_sentiment_score_mapped') -> dict:
    logger.info("test_aggregate_parameters")
    agg_df = create_aggregate_df(score_key)
    if agg_df.empty:
        print("No aggregate data available.")
        return {}
    agg_df = agg_df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_train = int(len(agg_df) * CONFIG["test_train_split"])
    train_df = agg_df.iloc[:n_train].copy()
    test_df = agg_df.iloc[n_train:].copy()
    losses = [aggregate_error_mse, aggregate_error_softl1, aggregate_error_dynamic_mse, aggregate_error_dynamic_softl1, aggregate_error_logistic_mse, aggregate_error_logistic_softl1]
    results = []
    null_avg_name = 'aggregate_null_average'
    if not agg_df.empty:
        preds = []
        for _, r in test_df.iterrows():
            preds.append(np.mean(r['s_inits']))
        preds = np.array(preds, dtype=float)
        mse = _safe_mse(test_df['s_user'].to_numpy(dtype=float), preds)
        results.append({
            "model": null_avg_name,
            "split": "none",
            "remove_outliers": "none",
            "mse": mse,
            "score_key": score_key,
            "params": [],
        })
        print(f"aggregate | {null_avg_name} | MSE={mse:.4f}")
    for loss_fn in losses:
        fitted = determine_aggregate_parameters(train_df, loss_fn, print_process=False)
        name = loss_fn.__name__
        if 'dynamic' in loss_fn.__name__:
            func = aggregate_formula_dynamic
        elif 'logistic' in loss_fn.__name__:
            func = aggregate_formula_logistic
        else:
            func = aggregate_formula
        preds = []
        for _, r in test_df.iterrows():
            preds.append(func(r['s_inits'], fitted['params']))
        preds = np.array(preds, dtype=float)
        mse = _safe_mse(test_df['s_user'].to_numpy(dtype=float), preds)
        results.append({
            "model": name,
            "split": "none",
            "remove_outliers": "none",
            "mse": mse,
            "score_key": score_key,
            "params": fitted['params'],
        })
        print(f"aggregate | {name} | MSE={mse:.4f}")
    best = min(results, key=lambda x: x["mse"]) if results else {}
    print(f"Best aggregate config: {best}")
    return {"best": best, "all": results}


def _serialize_fitted(fitted: dict) -> dict:
    out = {}
    for k, v in fitted.items():
        if v is None:
            continue
        if isinstance(v, dict) and 'params' in v:
            arr = v['params']
        else:
            arr = v
        if isinstance(arr, np.ndarray):
            arr = arr.tolist()
        elif isinstance(arr, (float, int)):
            arr = [float(arr)]
        out[k] = arr
    return out


def save_optimal_parameters(all_results: dict, out_path: str = "src/survey/optimal_formulas/all_optimal_parameters.json") -> None:
    entries = []
    for score_key, summary in all_results.items():
        for category, data in summary.items():
            if not data or 'best' not in data:
                continue
            best = data['best']
            entry = {
                "category": category,
                "score_key": score_key,
                "model": best.get("model"),
                "split": best.get("split", "none"),
                "remove_outliers": best.get("remove_outliers", "none"),
                "mse": float(best.get("mse", float('inf'))),
            }
            if category == 'aggregate':
                params = best.get('params')
                if isinstance(params, np.ndarray):
                    params = params.tolist()
                entry["params_by_key"] = {"params": params}
            else:
                fitted = best.get('fitted', {})
                entry["params_by_key"] = _serialize_fitted(fitted)
            entries.append(entry)
    payload = {"version": 1, "saved_at": datetime.utcnow().isoformat() + 'Z', "entries": entries}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=4)

def load_optimal_parameters(in_path: str = "src/survey/optimal_formulas/all_optimal_parameters.json") -> dict:
    if not os.path.exists(in_path):
        return {}
    with open(in_path, 'r') as f:
        data = json.load(f)
    out = {}
    for e in data.get('entries', []):
        out[(e['category'], e['score_key'])] = e
    return out

def _formula_by_name(name: str) -> Callable:
    from formulas import actor_formula_v1, actor_formula_v2, target_formula_v1, target_formula_v2, assoc_formula_v1, assoc_formula_v2, belong_formula_v1, belong_formula_v2, null_identity, null_avg, null_linear
    m = {
        'actor_formula_v1': actor_formula_v1,
        'actor_formula_v2': actor_formula_v2,
        'target_formula_v1': target_formula_v1,
        'target_formula_v2': target_formula_v2,
        'assoc_formula_v1': assoc_formula_v1,
        'assoc_formula_v2': assoc_formula_v2,
        'belong_formula_v1': belong_formula_v1,
        'belong_formula_v2': belong_formula_v2,
        'aggregate_normal': aggregate_formula,
        'aggregate_dynamic': aggregate_formula_dynamic,
        'aggregate_logistic': aggregate_formula_logistic,
        'null_identity': null_identity,
        'null_avg': null_avg,
        'null_linear': null_linear,
        'aggregate_null_average': lambda s_inits, _: float(np.mean(s_inits)) if len(s_inits) else 0.0,
    }
    return m.get(name)

def get_actor_function(score_key: str = 'user_sentiment_score_mapped') -> Callable:
    confs = load_optimal_parameters()
    conf = confs.get(('actor', score_key)) or confs.get(('actor', 'user_sentiment_score_mapped'))
    if not conf:
        return None
    func = _formula_by_name(conf['model'])
    split = conf.get('split', 'none')
    fitted = conf.get('params_by_key', {})
    def f(s_actor, s_action, s_target):
        return actor_skeleton_formula(s_actor, s_action, s_target, split, func, fitted)
    return f

def get_target_function(score_key: str = 'user_sentiment_score_mapped') -> Callable:
    confs = load_optimal_parameters()
    conf = confs.get(('target', score_key)) or confs.get(('target', 'user_sentiment_score_mapped'))
    if not conf:
        return None
    func = _formula_by_name(conf['model'])
    split = conf.get('split', 'none')
    fitted = conf.get('params_by_key', {})
    def f(s_target, s_action):
        return target_skeleton_formula(s_target, s_action, split, func, fitted)
    return f

def get_association_function(score_key: str = 'user_sentiment_score_mapped') -> Callable:
    confs = load_optimal_parameters()
    conf = confs.get(('association', score_key)) or confs.get(('association', 'user_sentiment_score_mapped'))
    if not conf:
        return None
    func = _formula_by_name(conf['model'])
    split = conf.get('split', 'none')
    fitted = conf.get('params_by_key', {})
    def f(s_entity, s_other):
        return association_skeleton_formula(s_entity, s_other, split, func, fitted)
    return f

def get_parent_function(score_key: str = 'user_sentiment_score_mapped') -> Callable:
    confs = load_optimal_parameters()
    conf = confs.get(('parent', score_key)) or confs.get(('parent', 'user_sentiment_score_mapped'))
    if not conf:
        return None
    func = _formula_by_name(conf['model'])
    split = conf.get('split', 'none')
    fitted = conf.get('params_by_key', {})
    def f(s_parent, s_child):
        return parent_skeleton_formula(s_parent, s_child, split, func, fitted)
    return f

def get_child_function(score_key: str = 'user_sentiment_score_mapped') -> Callable:
    confs = load_optimal_parameters()
    conf = confs.get(('child', score_key)) or confs.get(('child', 'user_sentiment_score_mapped'))
    if not conf:
        return None
    func = _formula_by_name(conf['model'])
    split = conf.get('split', 'none')
    fitted = conf.get('params_by_key', {})
    def f(s_child, s_parent):
        return child_skeleton_formula(s_child, s_parent, split, func, fitted)
    return f

def get_aggregate_function(score_key: str = 'user_sentiment_score_mapped') -> Callable:
    confs = load_optimal_parameters()
    conf = confs.get(('aggregate', score_key)) or confs.get(('aggregate', 'user_sentiment_score_mapped'))
    if not conf:
        return None
    model = conf['model']
    if 'dynamic' in model:
        func = aggregate_formula_dynamic
    elif 'logistic' in model:
        func = aggregate_formula_logistic
    else:
        func = aggregate_formula
    params = conf.get('params_by_key', {}).get('params', [])
    def f(s_inits):
        return func(s_inits, params)
    return f


def test_all_parameterizations() -> dict:
    logger.info("test_all_parameterizations")
    score_keys = ['user_sentiment_score', 'user_normalized_sentiment_scores', 'user_sentiment_score_mapped']
    all_results = {}
    for sk in score_keys:
        all_results[sk] = {
            "actor": test_actor_parameters(sk),
            "target": test_target_parameters(sk),
            "association": test_association_parameters(sk),
            "parent": test_parent_parameters(sk),
            "child": test_child_parameters(sk),
            "aggregate": test_aggregate_parameters(sk)
        }
    save_optimal_parameters(all_results)
    return all_results

"""if __name__ == '__main__':
    out = test_all_parameterizations()
    print("\nSummary:")
    for sk, summary in out.items():
        print(f"score_key={sk}")
        for k, v in summary.items():
            if v and "best" in v:
                print(f"- {k}: MSE={v['best']['mse']:.4f} | model={v['best']['model']} | split={v['best']['split']} | outliers={v['best'].get('remove_outliers','none')}")"""
