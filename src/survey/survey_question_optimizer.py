"""Tools for calibrating and analysing survey sentiment models."""

import ast
import inspect
import json
import logging
import math
import os
import re
import ssl
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy.optimize import OptimizeWarning, curve_fit, least_squares, minimize
from scipy.stats import spearmanr, t as t_dist, ttest_rel, wilcoxon
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=OptimizeWarning)
VERBOSE = True
RANDOM_STATE = 41
CV_FOLDS = 5

def _vprint(msg: str):
    print(msg)

ssl._create_default_https_context = ssl._create_unverified_context

def remove_zero_dominant_seeds(df: pd.DataFrame, threshold: float = 0.5) -> tuple[pd.DataFrame, list]:
    """Filter out seeds that submit mostly zero-valued responses."""
    scores = pd.to_numeric(df.get('user_sentiment_score'), errors='coerce')
    seeds = df.get('seed')
    tmp = pd.DataFrame({'seed': seeds, 'score': scores}).dropna(subset=['seed'])
    if tmp.empty:
        return df, []
    counts = tmp.groupby('seed')['score'].apply(lambda x: x.notna().sum())
    zeros = tmp.groupby('seed')['score'].apply(lambda x: (x.fillna(np.inf) == 0).sum())
    ratio = zeros / counts.replace(0, np.nan)
    bad_seeds = ratio[ratio > threshold].index.tolist()
    if bad_seeds:
        df = df[~df['seed'].isin(bad_seeds)].copy()
    return df, bad_seeds

url = 'data/survey_responses.csv'
try:
    df = pd.read_csv(url)
    df = df[df['attention_check_passed'] == True]
    df, _removed_zero_seeds = remove_zero_dominant_seeds(df)
    if _removed_zero_seeds:
        print(f"Removed {len(_removed_zero_seeds)} seeds with >50% zero answers")
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
    """Map Likert-style integer responses to a signed sentiment score."""
    if not isinstance(integer, int):
        integer = int(integer)
    return intensity_map_integer.get(abs(integer), 0) * ((integer > 0) - (integer < 0))

def fit(formula, X, y, bounds, remove_outliers_method:Literal['lsquares', 'none']='none'):
    """Fit a parametric curve to the provided data with optional robust loss."""
    x0 = [(l + u) / 2 for l, u in zip(bounds[0], bounds[1])]
    params = None
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
            logger.debug(f"Could not fit model (lsquares): {e}")
            return None
    elif remove_outliers_method == "none":
        try:
            params, _ = curve_fit(formula, X, y, p0=x0, bounds=bounds)
        except Exception as e:
            logger.debug(f"Could not fit model (none): {e}")
            return None
    else:
        return None
    if params is None:
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
    """Augment survey data with interpretable sentiment scores per respondent."""
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

def fit_compound(formula, X, y, remove_outliers_method:Literal['lsquares', 'none']='none'):
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
    bounds = ([-50] * (num_params), [50] * (num_params))
    return fit(formula, X, y, bounds, remove_outliers_method)

try:
    df, _ = add_score_interpretations(df)
except:
    print("Error adding score interpretations")

def create_action_df(score_key: Literal['user_sentiment_score', 'user_normalized_sentiment_scores', 'user_sentiment_score_mapped'] = 'user_sentiment_score_mapped') -> pd.DataFrame:
    """Create the action sentiment modelling dataframe for the given score."""
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
            's_user_target': s_user_target,
            'seed': seed
        })

    action_df = pd.DataFrame(action_data)
    action_df.drop_duplicates(inplace=True)

    return action_df

def create_association_df(score_key: Literal['user_sentiment_score', 'user_normalized_sentiment_scores', 'user_sentiment_score_mapped'] = 'user_sentiment_score_mapped') -> pd.DataFrame:
    """Create the association sentiment modelling dataframe for the given score."""
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
            's_user_other': s_user_other,
            'seed': seed
        })
        association_data.append({
            's_init_entity': base_sentiments[1],
            's_init_other': base_sentiments[0],
            's_user_entity': s_user_other,
            's_user_other': s_user_entity,
            'seed': seed
        })

    association_df = pd.DataFrame(association_data)
    association_df.drop_duplicates(inplace=True)

    return association_df

def create_belonging_df(score_key: Literal['user_sentiment_score', 'user_normalized_sentiment_scores', 'user_sentiment_score_mapped'] = 'user_sentiment_score_mapped') -> pd.DataFrame:
    """Create the belonging sentiment modelling dataframe for the given score."""
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
            's_user_child': s_user_other,
            'seed': seed
        })

    belonging_df = pd.DataFrame(belonging_data)
    belonging_df.drop_duplicates(inplace=True)

    return belonging_df

def create_aggregate_df(score_key: Literal['user_sentiment_score', 'user_normalized_sentiment_scores', 'user_sentiment_score_mapped'] = 'user_sentiment_score_mapped') -> pd.DataFrame:
    """Create the aggregate sentiment modelling dataframe for the given score."""
    aggregate_df = df[df['item_type'].str.contains('aggregate')].copy()

    cols_to_ignore = ['submission_timestamp_utc']
    subset_cols = [col for col in aggregate_df.columns if col not in cols_to_ignore]
    aggregate_df.drop_duplicates(subset=subset_cols, inplace=True)

    aggregate_data = []

    for _, group in aggregate_df.groupby(['seed', 'item_type']):
        seed = group.iloc[0]['seed']
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
                'seed': seed,
            })

    aggregate_df = pd.DataFrame(aggregate_data)

    return aggregate_df


def determine_actor_parameters(action_model_df: pd.DataFrame,
                                function: Callable,
                                remove_outlier_method: Literal['lsquares', 'none'] = 'none',
                                splits: Literal['none', 'driver', 'action_target'] = 'none',
                                print_process=False) -> pd.DataFrame:
    """Fit actor parameters across multiple sentiment driver splits."""
    action_model_df = action_model_df.copy()
    action_model_df.loc[:, 'driver'] = action_model_df['s_init_action'] * action_model_df['s_init_target']

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
                                remove_outlier_method: Literal['lsquares', 'none'] = 'none',
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
                                        remove_outlier_method: Literal['lsquares', 'none'] = 'none',
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
                                    remove_outlier_method: Literal['lsquares', 'none'] = 'none',
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
                                    remove_outlier_method: Literal['lsquares', 'none'] = 'none',
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
        param_bounds = [(-20, 20), (-20, 20), (-20, 20), (-20, 20)]
    else:
        initial_guess = [1.0, 1.0]
        param_bounds = [(-20, 20), (-20, 20)]

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
    mask = ~np.isnan(y_pred)
    if mask.sum() == 0:
        return float('inf')
    return mean_squared_error(y_true[mask], y_pred[mask])

def _safe_mae(y_true: np.ndarray, y_pred: np.ndarray):
    mask = ~np.isnan(y_pred)
    if mask.sum() == 0:
        return None
    return float(mean_absolute_error(y_true[mask], y_pred[mask]))

def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray):
    mask = ~np.isnan(y_pred)
    if mask.sum() < 2:
        return None
    yt = y_true[mask]
    yp = y_pred[mask]
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot > 0:
        return float(1.0 - (ss_res / ss_tot))
    else:
        return float(1.0 if ss_res == 0 else 0.0)

def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray):
    mask = ~np.isnan(y_pred)
    if mask.sum() < 2:
        return None
    rho, _ = spearmanr(y_true[mask], y_pred[mask])
    if np.isnan(rho):
        return None
    return float(rho)

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        'mse': float(_safe_mse(y_true, y_pred)),
        'mae': _safe_mae(y_true, y_pred),
        'r2': _safe_r2(y_true, y_pred),
        'spearman': _safe_spearman(y_true, y_pred),
    }

def _cv_evaluate(df_in: pd.DataFrame, fold_eval: Callable[[pd.DataFrame, pd.DataFrame], tuple[np.ndarray, np.ndarray]]) -> dict:
    if len(df_in) < 2:
        return {
            'cv_mean_mse': float('inf'), 'cv_std_mse': 0.0,
            'cv_mean_r2': None, 'cv_std_r2': 0.0,
            'cv_mean_spearman': None, 'cv_std_spearman': 0.0,
            'fold_mse': [], 'fold_r2': [], 'fold_spearman': [],
        }
    k = min(CV_FOLDS, len(df_in))
    if k < 2:
        k = 2
    kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    mses, r2s, sps = [], [], []
    X = df_in.reset_index(drop=True)
    for tr_idx, te_idx in kf.split(X):
        tr = X.iloc[tr_idx]
        te = X.iloc[te_idx]
        y_true, y_pred = fold_eval(tr, te)
        if y_true is None or y_pred is None or len(y_true) == 0 or len(y_pred) == 0:
            continue
        m = _compute_metrics(y_true, y_pred)
        mses.append(m['mse'])
        r2s.append(m['r2'])
        sps.append(m['spearman'])
    if not mses:
        return {
            'cv_mean_mse': float('inf'), 'cv_std_mse': 0.0,
            'cv_mean_r2': None, 'cv_std_r2': 0.0,
            'cv_mean_spearman': None, 'cv_std_spearman': 0.0,
            'fold_mse': [], 'fold_r2': [], 'fold_spearman': [],
        }
    valid_mses = [x for x in mses if np.isfinite(x)]
    mean_mse = float(np.mean(valid_mses)) if valid_mses else float('inf')
    std_mse = float(np.std(valid_mses, ddof=1)) if len(valid_mses) > 1 else 0.0
    valid_r2 = [x for x in r2s if (x is not None and np.isfinite(x))]
    mean_r2 = float(np.mean(valid_r2)) if valid_r2 else None
    std_r2 = float(np.std(valid_r2, ddof=1)) if len(valid_r2) > 1 else 0.0
    valid_sps = [x for x in sps if (x is not None and np.isfinite(x))]
    mean_sp = float(np.mean(valid_sps)) if valid_sps else None
    std_sp = float(np.std(valid_sps, ddof=1)) if len(valid_sps) > 1 else 0.0
    return {
        'cv_mean_mse': mean_mse, 'cv_std_mse': std_mse,
        'cv_mean_r2': mean_r2, 'cv_std_r2': std_r2,
        'cv_mean_spearman': mean_sp, 'cv_std_spearman': std_sp,
        'fold_mse': [float(x) for x in mses],
        'fold_r2': [float(x) if x is not None else None for x in r2s],
        'fold_spearman': [float(x) if x is not None else None for x in sps],
    }

def _fisher_z(r: float) -> float | None:
    if r is None or not isinstance(r, (int, float)) or np.isnan(r) or r <= -1.0 or r >= 1.0:
        if isinstance(r, (int, float)) and not np.isnan(r):
            r = float(np.clip(r, -0.999999, 0.999999))
        else:
            return None
    return float(np.arctanh(r))

def _mean_std_ci(values: list[float]) -> dict:
    arr = np.array([v for v in values if isinstance(v, (int, float)) and np.isfinite(v)], dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": None, "std": 0.0, "ci_low": None, "ci_high": None, "n": 0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if n > 1:
        tcrit = float(t_dist.ppf(0.975, df=n-1))
        half = tcrit * (std / np.sqrt(n)) if std > 0 else 0.0
        return {"mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half, "n": n}
    return {"mean": mean, "std": std, "ci_low": mean, "ci_high": mean, "n": n}

def _paired_stats(x: list[float], y: list[float]) -> dict:
    a = np.array([v for v in x if isinstance(v, (int, float))])
    b = np.array([v for v in y if isinstance(v, (int, float))])
    n = min(len(a), len(b))
    if n == 0:
        return {"n": 0, "ttest_p": None, "wilcoxon_p": None, "cohens_d": None, "mean_diff": None}
    a = a[:n]
    b = b[:n]
    diff = a - b
    d_mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    d = (d_mean / sd) if sd > 0 else None
    try:
        t_p = float(ttest_rel(a, b, nan_policy='omit').pvalue)
    except Exception:
        t_p = None
    try:
        if np.allclose(diff, 0):
            w_p = None
        else:
            w_p = float(wilcoxon(a, b, zero_method='wilcox', correction=False, alternative='two-sided').pvalue)
    except Exception:
        w_p = None
    return {"n": n, "ttest_p": t_p, "wilcoxon_p": w_p, "cohens_d": d, "mean_diff": d_mean}

def _to_01(x: np.ndarray) -> np.ndarray:
    return np.clip((x + 1.0) / 2.0, 0.0, 1.0)

def _ece_and_bins(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> dict:
    yt = _to_01(y_true.astype(float))
    yp = _to_01(y_pred.astype(float))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(yp, bins) - 1
    ece = 0.0
    bin_summary = []
    m = len(yp)
    for b in range(n_bins):
        sel = inds == b
        if not np.any(sel):
            bin_summary.append({"bin": b, "count": 0, "mean_pred": None, "mean_true": None})
            continue
        mean_pred = float(np.mean(yp[sel]))
        mean_true = float(np.mean(yt[sel]))
        weight = float(np.sum(sel)) / m
        ece += weight * abs(mean_pred - mean_true)
        bin_summary.append({"bin": b, "count": int(np.sum(sel)), "mean_pred": mean_pred, "mean_true": mean_true})
    brier = float(np.mean((yt - yp) ** 2))
    return {"ece": float(ece), "brier": brier, "bins": bin_summary}

def _plot_calibration(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: str, n_bins: int = 10) -> str | None:
    try:
        yt = _to_01(y_true.astype(float))
        yp = _to_01(y_pred.astype(float))
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        inds = np.digitize(yp, bins) - 1
        xs, ys = [], []
        for b in range(n_bins):
            sel = inds == b
            if not np.any(sel):
                continue
            xs.append(float(np.mean(yp[sel])))
            ys.append(float(np.mean(yt[sel])))
        plt.figure(figsize=(4, 4))
        if xs:
            plt.plot(xs, ys, marker='o', label='binned')
        plt.plot([0, 1], [0, 1], '--', color='gray', label='ideal')
        plt.xlabel('Predicted (mapped to [0,1])')
        plt.ylabel('Observed (mapped to [0,1])')
        plt.title(title)
        plt.legend()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return out_path
    except Exception:
        return None

def _ensure_optimal_params_saved() -> None:
    opt_path = "src/survey/optimal_formulas/all_optimal_parameters.json"
    if not os.path.exists(opt_path):
        all_results = test_all_parameterizations()
        save_optimal_parameters(all_results)

def _baseline_predict_row(category: str, row: pd.Series) -> float:
    if category == 'actor':
        return float(np.mean([row['s_init_actor'], row['s_init_action'] * row['s_init_target']]))
    if category == 'target':
        return float(np.mean([row['s_init_target'], row['s_init_action']]))
    if category == 'association':
        return float(np.mean([row['s_init_entity'], row['s_init_other']]))
    if category == 'parent':
        return float(np.mean([row['s_init_parent'], row['s_init_child']]))
    if category == 'child':
        return float(np.mean([row['s_init_child'], row['s_init_parent']]))
    if category == 'aggregate':
        s_inits = row['s_inits'] if isinstance(row['s_inits'], (list, tuple, np.ndarray)) else []
        return float(np.mean(s_inits)) if len(s_inits) else 0.0
    return np.nan

def _candidate_predict_fn(category: str, score_key: str) -> Callable | None:
    if category == 'actor':
        return get_actor_function(score_key)
    if category == 'target':
        return get_target_function(score_key)
    if category == 'association':
        return get_association_function(score_key)
    if category == 'parent':
        return get_parent_function(score_key)
    if category == 'child':
        return get_child_function(score_key)
    if category == 'aggregate':
        return get_aggregate_function(score_key)
    return None

def _category_dataframe(category: str, score_key: str) -> tuple[pd.DataFrame, str, Callable[[pd.Series], float]]:
    if category == 'actor' or category == 'target':
        df_cat = create_action_df(score_key)
    elif category == 'association':
        df_cat = create_association_df(score_key)
    elif category == 'parent' or category == 'child':
        df_cat = create_belonging_df(score_key)
    elif category == 'aggregate':
        df_cat = create_aggregate_df(score_key)
    else:
        return pd.DataFrame(), '', lambda r: np.nan
    if category == 'actor':
        y_col = 's_user_actor'
        def pred_row_fn_cand(r, f):
            return f(r['s_init_actor'], r['s_init_action'], r['s_init_target']) if f else np.nan
    elif category == 'target':
        y_col = 's_user_target'
        def pred_row_fn_cand(r, f):
            return f(r['s_init_target'], r['s_init_action']) if f else np.nan
    elif category == 'association':
        y_col = 's_user_entity'
        def pred_row_fn_cand(r, f):
            return f(r['s_init_entity'], r['s_init_other']) if f else np.nan
    elif category == 'parent':
        y_col = 's_user_parent'
        def pred_row_fn_cand(r, f):
            return f(r['s_init_parent'], r['s_init_child']) if f else np.nan
    elif category == 'child':
        y_col = 's_user_child'
        def pred_row_fn_cand(r, f):
            return f(r['s_init_child'], r['s_init_parent']) if f else np.nan
    else:             
        y_col = 's_user'
        def pred_row_fn_cand(r, f):
            return f(r['s_inits']) if f else np.nan
    return df_cat, y_col, pred_row_fn_cand

def evaluate_category_per_participant(category: str, score_key: str, n_bins_calib: int = 10, out_dir: str = "outputs/benchmark") -> dict:
    _ensure_optimal_params_saved()
    df_cat, y_col, pred_row_fn_cand = _category_dataframe(category, score_key)
    if df_cat.empty:
        return {"category": category, "score_key": score_key, "per_participant": [], "aggregate": {}, "calibration": {}}
    cand_fn = _candidate_predict_fn(category, score_key)
    preds_c, preds_b, y = [], [], []
    seeds = []
    per_seed = {}
    for seed, g in df_cat.groupby('seed'):
        y_true = pd.to_numeric(g[y_col], errors='coerce').to_numpy(dtype=float)
        y_pred_c = np.array([pred_row_fn_cand(r, cand_fn) for _, r in g.iterrows()], dtype=float)
        y_pred_b = np.array([_baseline_predict_row(category, r) for _, r in g.iterrows()], dtype=float)
        mask = ~np.isnan(y_true)
        y_true = y_true[mask]
        y_pred_c = y_pred_c[mask]
        y_pred_b = y_pred_b[mask]
        if len(y_true) == 0:
            continue
        m_c = _compute_metrics(y_true, y_pred_c)
        m_b = _compute_metrics(y_true, y_pred_b)
        rho_c = m_c['spearman']
        rho_b = m_b['spearman']
        z_c = _fisher_z(rho_c)
        z_b = _fisher_z(rho_b)
        rec = {
            "seed": seed,
            "n": int(len(y_true)),
            "mse_c": m_c['mse'], "mae_c": m_c['mae'], "r2_c": m_c['r2'], "rho_c": rho_c, "z_c": z_c,
            "mse_b": m_b['mse'], "mae_b": m_b['mae'], "r2_b": m_b['r2'], "rho_b": rho_b, "z_b": z_b,
        }
        per_seed[seed] = rec
        preds_c.append(y_pred_c)
        preds_b.append(y_pred_b)
        y.append(y_true)
        seeds.append(seed)
    per_list = list(per_seed.values())
    mse_c_list = [r['mse_c'] for r in per_list if isinstance(r['mse_c'], (int, float)) and np.isfinite(r['mse_c'])]
    mse_b_list = [r['mse_b'] for r in per_list if isinstance(r['mse_b'], (int, float)) and np.isfinite(r['mse_b'])]
    mae_c_list = [r['mae_c'] for r in per_list if isinstance(r['mae_c'], (int, float)) and np.isfinite(r['mae_c'])]
    mae_b_list = [r['mae_b'] for r in per_list if isinstance(r['mae_b'], (int, float)) and np.isfinite(r['mae_b'])]
    rho_c_list = [r['rho_c'] for r in per_list if r['rho_c'] is not None]
    rho_b_list = [r['rho_b'] for r in per_list if r['rho_b'] is not None]
    z_c_list = [r['z_c'] for r in per_list if r['z_c'] is not None]
    z_b_list = [r['z_b'] for r in per_list if r['z_b'] is not None]

    agg = {
        "mse_c": _mean_std_ci(mse_c_list),
        "mse_b": _mean_std_ci(mse_b_list),
        "mae_c": _mean_std_ci(mae_c_list),
        "mae_b": _mean_std_ci(mae_b_list),
        "rho_c": _mean_std_ci(rho_c_list),
        "rho_b": _mean_std_ci(rho_b_list),
        "z_c": _mean_std_ci(z_c_list),
        "z_b": _mean_std_ci(z_b_list),
        "paired_mse": _paired_stats(mse_c_list, mse_b_list),
        "paired_z": _paired_stats(z_c_list, z_b_list),
        "n_participants": len(per_list),
    }

    if len(y) > 0:
        y_all = np.concatenate(y)
        yp_all_c = np.concatenate(preds_c)
        yp_all_b = np.concatenate(preds_b)
        calib_c = _ece_and_bins(y_all, yp_all_c)
        calib_b = _ece_and_bins(y_all, yp_all_b)
        ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        out_base = os.path.join(out_dir, "calibration")
        os.makedirs(out_base, exist_ok=True)
        p1 = _plot_calibration(y_all, yp_all_c, f"Calibration {category} cand ({score_key})", os.path.join(out_base, f"calib_{category}_{score_key}_cand_{ts}.png"))
        p2 = _plot_calibration(y_all, yp_all_b, f"Calibration {category} base ({score_key})", os.path.join(out_base, f"calib_{category}_{score_key}_base_{ts}.png"))
        calibration = {"candidate": calib_c, "baseline": calib_b, "plots": {"candidate": p1, "baseline": p2}}
    else:
        calibration = {}

    return {"category": category, "score_key": score_key, "per_participant": per_list, "aggregate": agg, "calibration": calibration}

def evaluate_all_per_participant(score_keys: list[str] | None = None, categories: list[str] | None = None, out_dir: str = "outputs/benchmark") -> dict:
    if score_keys is None:
        score_keys = ['user_sentiment_score', 'user_sentiment_score_mapped', 'user_normalized_sentiment_scores']
    if categories is None:
        categories = ['actor', 'target', 'association', 'parent', 'child', 'aggregate']
    out = {}
    for sk in score_keys:
        out[sk] = {}
        for cat in categories:
            out[sk][cat] = evaluate_category_per_participant(cat, sk, out_dir=out_dir)
    return out

def robustness_summary() -> dict:
    _ensure_optimal_params_saved()
    confs = load_optimal_parameters()
    by_cat: dict[str, dict[str, str]] = {}
    for (cat, sk), entry in confs.items():
        by_cat.setdefault(cat, {})[sk] = entry.get('model')
    stability = {}
    for cat, m in by_cat.items():
        unique = set(m.values())
        stability[cat] = {"unique_models": len(unique), "models": m}
    return {"by_category": by_cat, "stability": stability}



def _fmt_num(x) -> str:
    try:
        xf = float(x)
        s = f"{xf:.4g}"
        if s.startswith('.'): s = '0' + s
        if s.startswith('-.'): s = s.replace('-.', '-0.')
        return s
    except Exception:
        return str(x)

def _format_actor(model: str, params: list[float]) -> tuple[str, str]:
    if model == 'actor_formula_v1':
        la, w, b = params
        eq = "y = tanh(λ_a·s_actor + (1−λ_a)·w·(s_action·s_target) + b)"
        vals = f"λ_a={_fmt_num(la)}, w={_fmt_num(w)}, b={_fmt_num(b)}"
    elif model == 'actor_formula_v2':
        w_actor, w_driver, b = params
        eq = "y = tanh(w_actor·s_actor + w_driver·(s_action·s_target) + b)"
        vals = f"w_actor={_fmt_num(w_actor)}, w_driver={_fmt_num(w_driver)}, b={_fmt_num(b)}"
    elif model == 'null_identity':
        eq, vals = "y = s_actor", "(null identity)"
    elif model == 'null_avg':
        eq, vals = "y = (s_actor + s_action + s_target)/3", "(null avg)"
    elif model == 'null_linear':
        w1, w2, b = params
        eq = "y = w1·s_actor + w2·(s_action·s_target) + b"
        vals = f"w1={_fmt_num(w1)}, w2={_fmt_num(w2)}, b={_fmt_num(b)}"
    else:
        eq, vals = model, ''
    return eq, vals

def _format_target(model: str, params: list[float]) -> tuple[str, str]:
    if model == 'target_formula_v1':
        lt, w, b = params
        eq = "y = tanh(λ_t·s_target + (1−λ_t)·w·s_action + b)"
        vals = f"λ_t={_fmt_num(lt)}, w={_fmt_num(w)}, b={_fmt_num(b)}"
    elif model == 'target_formula_v2':
        wt, wa, b = params
        eq = "y = tanh(w_target·s_target + w_action·s_action + b)"
        vals = f"w_target={_fmt_num(wt)}, w_action={_fmt_num(wa)}, b={_fmt_num(b)}"
    elif model == 'null_identity':
        eq, vals = "y = s_target", "(null identity)"
    elif model == 'null_avg':
        eq, vals = "y = (s_target + s_action)/2", "(null avg)"
    elif model == 'null_linear':
        w1, w2, b = params
        eq = "y = w1·s_target + w2·s_action + b"
        vals = f"w1={_fmt_num(w1)}, w2={_fmt_num(w2)}, b={_fmt_num(b)}"
    else:
        eq, vals = model, ''
    return eq, vals

def _format_assoc(model: str, params: list[float]) -> tuple[str, str]:
    if model == 'assoc_formula_v1':
        lmb, w, b = params
        eq = "y = tanh(λ·s_entity + (1−λ)·w·s_other + b)"
        vals = f"λ={_fmt_num(lmb)}, w={_fmt_num(w)}, b={_fmt_num(b)}"
    elif model == 'assoc_formula_v2':
        we, wo, b = params
        eq = "y = tanh(w_entity·s_entity + w_other·s_other + b)"
        vals = f"w_entity={_fmt_num(we)}, w_other={_fmt_num(wo)}, b={_fmt_num(b)}"
    elif model == 'null_identity':
        eq, vals = "y = s_entity", "(null identity)"
    elif model == 'null_avg':
        eq, vals = "y = (s_entity + s_other)/2", "(null avg)"
    elif model == 'null_linear':
        w1, w2, b = params
        eq = "y = w1·s_entity + w2·s_other + b"
        vals = f"w1={_fmt_num(w1)}, w2={_fmt_num(w2)}, b={_fmt_num(b)}"
    else:
        eq, vals = model, ''
    return eq, vals

def _format_belong(model: str, params: list[float], parent_side: bool) -> tuple[str, str]:
    a, btxt = ("s_parent", "s_child") if parent_side else ("s_child", "s_parent")
    if model == 'belong_formula_v1':
        lmb, w, b = params
        eq = f"y = tanh(λ·{a} + (1−λ)·w·{btxt} + b)"
        vals = f"λ={_fmt_num(lmb)}, w={_fmt_num(w)}, b={_fmt_num(b)}"
    elif model == 'belong_formula_v2':
        w1, w2, b = params
        eq = f"y = tanh(w1·{a} + w2·{btxt} + b)"
        vals = f"w1={_fmt_num(w1)}, w2={_fmt_num(w2)}, b={_fmt_num(b)}"
    elif model == 'null_identity':
        eq, vals = f"y = {a}", "(null identity)"
    elif model == 'null_avg':
        eq, vals = f"y = ({a} + {btxt})/2", "(null avg)"
    elif model == 'null_linear':
        w1, w2, b = params
        eq = f"y = w1·{a} + w2·{btxt} + b"
        vals = f"w1={_fmt_num(w1)}, w2={_fmt_num(w2)}, b={_fmt_num(b)}"
    else:
        eq, vals = model, ''
    return eq, vals

def _format_aggregate(model: str, params: list[float]) -> list[str]:
    lines = []
    if model == 'aggregate_error_mse' or model == 'aggregate_error_softl1' or model == 'aggregate_normal':
        if len(params) >= 2:
            alpha, beta = params[:2]
            lines.append("y = Σ_i w_i·s_i, where w_i ∝ i^(α−1)·(N−i+1)^(β−1) normalized")
            lines.append(f"α={_fmt_num(alpha)}, β={_fmt_num(beta)}")
        else:
            lines.append("y = Σ_i w_i·s_i (Beta weights)")
    elif 'dynamic' in model:
        if len(params) >= 4:
            m_a, c_a, m_b, c_b = params[:4]
            lines.append("y = Σ_i w_i·s_i, w_i from Beta(α,β) with α=m_a·N+c_a, β=m_b·N+c_b")
            lines.append(f"m_a={_fmt_num(m_a)}, c_a={_fmt_num(c_a)}, m_b={_fmt_num(m_b)}, c_b={_fmt_num(c_b)}")
        else:
            lines.append("y = Σ_i w_i·s_i (dynamic α,β)")
    elif 'logistic' in model:
        if len(params) >= 8:
            L_a, k_a, N0_a, b_a, L_b, k_b, N0_b, b_b = params[:8]
            lines.append("y = Σ_i w_i·s_i, with α=b_a+L_a/(1+e^{−k_a(N−N0_a)}), β=b_b+L_b/(1+e^{−k_b(N−N0_b)})")
            lines.append(f"L_a={_fmt_num(L_a)}, k_a={_fmt_num(k_a)}, N0_a={_fmt_num(N0_a)}, b_a={_fmt_num(b_a)}")
            lines.append(f"L_b={_fmt_num(L_b)}, k_b={_fmt_num(k_b)}, N0_b={_fmt_num(N0_b)}, b_b={_fmt_num(b_b)}")
        else:
            lines.append("y = Σ_i w_i·s_i (logistic α,β)")
    elif model == 'aggregate_null_average':
        lines.append("y = mean(s_i)")
    else:
        lines.append(model)
    return lines

def _simplify_linear(terms: list[tuple[float, str]], bias: float | None, drop_thresh: float = 1e-6) -> str:
    parts = []
    for c, name in terms:
        if not isinstance(c, (int, float)) or np.isnan(c) or abs(c) < drop_thresh:
            continue
        sign = '+' if c >= 0 else '-'
        coef = abs(c)
        parts.append(f" {sign} {_fmt_num(coef)}·{name}")
    s = ''.join(parts).lstrip()
    if not s:
        s = '0'
    if isinstance(bias, (int, float)) and not np.isnan(bias) and abs(bias) >= drop_thresh:
        sign = '+' if bias >= 0 else '-'
        s += f" {sign} {_fmt_num(abs(bias))}"
    return s

def _numeric_eq(cat: str, model: str, params: list[float]) -> list[str]:
    lines: list[str] = []
    try:
        if cat == 'actor':
            if model == 'actor_formula_v1' and len(params) >= 3:
                la, w, b = params[:3]
                c_actor = la
                c_driver = (1 - la) * w
                inner = _simplify_linear([(c_actor, 's_actor'), (c_driver, '(s_action·s_target)')], b)
                lines.append(f"y = tanh({inner})")
            elif model == 'actor_formula_v2' and len(params) >= 3:
                w_actor, w_driver, b = params[:3]
                inner = _simplify_linear([(w_actor, 's_actor'), (w_driver, '(s_action·s_target)')], b)
                lines.append(f"y = tanh({inner})")
            elif model == 'null_identity':
                lines.append("y = s_actor")
            elif model == 'null_avg':
                lines.append("y = (s_actor + s_action + s_target)/3")
            elif model == 'null_linear' and len(params) >= 3:
                w1, w2, b = params[:3]
                inner = _simplify_linear([(w1, 's_actor'), (w2, '(s_action·s_target)')], b)
                lines.append(f"y = {inner}")
        elif cat == 'target':
            if model == 'target_formula_v1' and len(params) >= 3:
                lt, w, b = params[:3]
                inner = _simplify_linear([(lt, 's_target'), ((1 - lt) * w, 's_action')], b)
                lines.append(f"y = tanh({inner})")
            elif model == 'target_formula_v2' and len(params) >= 3:
                wt, wa, b = params[:3]
                inner = _simplify_linear([(wt, 's_target'), (wa, 's_action')], b)
                lines.append(f"y = tanh({inner})")
            elif model == 'null_identity':
                lines.append("y = s_target")
            elif model == 'null_avg':
                lines.append("y = (s_target + s_action)/2")
            elif model == 'null_linear' and len(params) >= 3:
                w1, w2, b = params[:3]
                inner = _simplify_linear([(w1, 's_target'), (w2, 's_action')], b)
                lines.append(f"y = {inner}")
        elif cat == 'association':
            if model == 'assoc_formula_v1' and len(params) >= 3:
                lmb, w, b = params[:3]
                inner = _simplify_linear([(lmb, 's_entity'), ((1 - lmb) * w, 's_other')], b)
                lines.append(f"y = tanh({inner})")
            elif model == 'assoc_formula_v2' and len(params) >= 3:
                we, wo, b = params[:3]
                inner = _simplify_linear([(we, 's_entity'), (wo, 's_other')], b)
                lines.append(f"y = tanh({inner})")
            elif model == 'null_identity':
                lines.append("y = s_entity")
            elif model == 'null_avg':
                lines.append("y = (s_entity + s_other)/2")
            elif model == 'null_linear' and len(params) >= 3:
                w1, w2, b = params[:3]
                inner = _simplify_linear([(w1, 's_entity'), (w2, 's_other')], b)
                lines.append(f"y = {inner}")
        elif cat in ('parent','child'):
            a, bname = ("s_parent","s_child") if cat == 'parent' else ("s_child","s_parent")
            if model == 'belong_formula_v1' and len(params) >= 3:
                lmb, w, b = params[:3]
                inner = _simplify_linear([(lmb, a), ((1 - lmb) * w, bname)], b)
                lines.append(f"y = tanh({inner})")
            elif model == 'belong_formula_v2' and len(params) >= 3:
                w1, w2, b = params[:3]
                inner = _simplify_linear([(w1, a), (w2, bname)], b)
                lines.append(f"y = tanh({inner})")
            elif model == 'null_identity':
                lines.append(f"y = {a}")
            elif model == 'null_avg':
                lines.append(f"y = ({a} + {bname})/2")
            elif model == 'null_linear' and len(params) >= 3:
                w1, w2, b = params[:3]
                inner = _simplify_linear([(w1, a), (w2, bname)], b)
                lines.append(f"y = {inner}")
        elif cat == 'aggregate':
            pass
    except Exception:
        return []
    return lines

def _render_piecewise(split: str, parts: dict[str, list[float]], build_eq: Callable[[list[float]], tuple[str, str]], conditions: dict[str, str]) -> list[str]:
    lines = [f"Split: {split}", "Piecewise:"]
    order = ['pos_driver_params','neg_driver_params','pos_pos_params','pos_neg_params','neg_pos_params','neg_neg_params','pos_action_params','neg_action_params','pos_params','neg_params']
    for key in order:
        if key not in parts:
            continue
        params = parts[key]
        eq, vals = build_eq(params)
        cond = conditions.get(key, key)
        lines.append(f"  if {cond}: {eq}")
        if vals:
            lines.append(f"    with {vals}")
    return lines

def print_optimal_formulas_math(score_keys: list[str] | None = None) -> dict:
    _ensure_optimal_params_saved()
    opt = load_optimal_parameters()
    if not opt:
        print("[red]No optimal parameters found.[/red]")
        return {}
    console = Console()
    categories = ['actor','target','association','parent','child','aggregate']
    if score_keys is None:
        score_keys = ['user_sentiment_score','user_sentiment_score_mapped','user_normalized_sentiment_scores']
    rendered: dict = {}
    for cat in categories:
        for sk in score_keys:
            entry = opt.get((cat, sk))
            if not entry:
                continue
            model = entry.get('model')
            split = entry.get('split', 'none')
            console.rule(f"{cat} — {sk} — {model} ({split})")
            params_by_key = entry.get('params_by_key', {}) or {}
            lines: list[str] = []
            if cat == 'aggregate':
                params = params_by_key.get('params') or []
                lines.extend(_format_aggregate(model, params))
            else:
                def _as_list(v):
                    if isinstance(v, dict) and 'params' in v: v = v['params']
                    return list(v) if isinstance(v, (list, tuple, np.ndarray)) else []
                flat = {k: _as_list(v) for k, v in params_by_key.items()}
                if cat == 'actor':
                    def build(p): return _format_actor(model, p)
                    conds = {
                        'pos_driver_params': 'driver > 0',
                        'neg_driver_params': 'driver ≤ 0',
                        'pos_pos_params': 's_action > 0 ∧ s_target > 0',
                        'pos_neg_params': 's_action > 0 ∧ s_target ≤ 0',
                        'neg_pos_params': 's_action ≤ 0 ∧ s_target > 0',
                        'neg_neg_params': 's_action ≤ 0 ∧ s_target ≤ 0',
                        'params': 'no split',
                    }
                elif cat == 'target':
                    def build(p): return _format_target(model, p)
                    conds = {'pos_action_params': 's_action > 0', 'neg_action_params': 's_action ≤ 0', 'params': 'no split'}
                elif cat == 'association':
                    def build(p): return _format_assoc(model, p)
                    conds = {
                        'pos_params': 's_other > 0', 'neg_params': 's_other ≤ 0',
                        'pos_pos_params': 's_entity > 0 ∧ s_other > 0',
                        'pos_neg_params': 's_entity > 0 ∧ s_other ≤ 0',
                        'neg_pos_params': 's_entity ≤ 0 ∧ s_other > 0',
                        'neg_neg_params': 's_entity ≤ 0 ∧ s_other ≤ 0',
                        'params': 'no split',
                    }
                elif cat == 'parent':
                    def build(p): return _format_belong(model, p, parent_side=True)
                    conds = {
                        'pos_params': 's_child > 0', 'neg_params': 's_child ≤ 0',
                        'pos_pos_params': 's_parent > 0 ∧ s_child > 0',
                        'pos_neg_params': 's_parent > 0 ∧ s_child ≤ 0',
                        'neg_pos_params': 's_parent ≤ 0 ∧ s_child > 0',
                        'neg_neg_params': 's_parent ≤ 0 ∧ s_child ≤ 0',
                        'params': 'no split',
                    }
                else:
                    def build(p): return _format_belong(model, p, parent_side=False)
                    conds = {
                        'pos_params': 's_parent > 0', 'neg_params': 's_parent ≤ 0',
                        'pos_pos_params': 's_child > 0 ∧ s_parent > 0',
                        'pos_neg_params': 's_child > 0 ∧ s_parent ≤ 0',
                        'neg_pos_params': 's_child ≤ 0 ∧ s_parent > 0',
                        'neg_neg_params': 's_child ≤ 0 ∧ s_parent ≤ 0',
                        'params': 'no split',
                    }
                if split == 'none' or ('params' in flat and len(flat) == 1):
                    plist = flat.get('params') or []
                    eq, vals = build(plist)
                    lines.append(eq)
                    if vals: lines.append(f"with {vals}")
                    num = _numeric_eq(cat, model, plist)
                    lines.extend(num)
                else:
                    lines.extend(_render_piecewise(split, flat, build, conds))
                    order = ['pos_driver_params','neg_driver_params','pos_pos_params','pos_neg_params','neg_pos_params','neg_neg_params','pos_action_params','neg_action_params','pos_params','neg_params']
                    for key in order:
                        if key in flat:
                            num = _numeric_eq(cat, model, flat[key])
                            for ln in num:
                                lines.append(f"  {ln}")
            for ln in lines:
                console.print(ln)
            rendered[(cat, sk)] = {"model": model, "split": split, "lines": lines}
    return rendered

def _aggregate_weights(model: str, params: list[float], N: int) -> np.ndarray:
    N = int(max(1, N))
    if model == 'aggregate_null_average':
        w = np.ones(N)
    elif 'dynamic' in model and len(params) >= 4:
        m_a, c_a, m_b, c_b = params[:4]
        alpha = m_a * N + c_a
        beta = m_b * N + c_b
        w = calculate_weights(N, alpha, beta)
        return w
    elif 'logistic' in model and len(params) >= 8:
        L_a, k_a, N0_a, b_a, L_b, k_b, N0_b, b_b = params[:8]
        alpha = logistic_function(N, L_a, k_a, N0_a, b_a)
        beta = logistic_function(N, L_b, k_b, N0_b, b_b)
        w = calculate_weights(N, alpha, beta)
        return w
    else:
        if len(params) >= 2:
            alpha, beta = params[:2]
            w = calculate_weights(N, alpha, beta)
        else:
            w = np.ones(N)
    return w / np.sum(w)

def show_aggregate_weight_slider(score_keys: list[str] | None = None, maxN: int = 12) -> None:
    _ensure_optimal_params_saved()
    if score_keys is None:
        score_keys = ['user_sentiment_score','user_sentiment_score_mapped','user_normalized_sentiment_scores']
    opt = load_optimal_parameters()
    for sk in score_keys:
        entry = opt.get(('aggregate', sk))
        if not entry:
            continue
        model = entry.get('model')
        params = (entry.get('params_by_key', {}) or {}).get('params') or []
        N0 = 6
        w = _aggregate_weights(model, params, N0)
        fig, ax = plt.subplots(figsize=(6, 4))
        plt.subplots_adjust(bottom=0.25)
        bars = ax.bar(np.arange(1, len(w)+1), w)
        ax.set_ylim(0, max(0.01, max(w)*1.2))
        ax.set_xlabel('i (position)')
        ax.set_ylabel('weight w_i')
        ax.set_title(f'Aggregate weights — {sk} — {model}')
        axN = plt.axes([0.15, 0.1, 0.7, 0.03])
        sN = Slider(axN, 'N', 2, maxN, valinit=N0, valstep=1)
        def update(val):
            Nv = int(sN.val)
            wv = _aggregate_weights(model, params, Nv)
            ax.clear()
            ax.bar(np.arange(1, Nv+1), wv)
            ax.set_ylim(0, max(0.01, max(wv)*1.2))
            ax.set_xlabel('i (position)')
            ax.set_ylabel('weight w_i')
            ax.set_title(f'Aggregate weights — {sk} — {model} — N={Nv}')
            fig.canvas.draw_idle()
        sN.on_changed(update)
        ax.set_title(f'Aggregate weights — {sk} — {model} — N={N0}')
        plt.show()


def test_actor_parameters(score_key: str = 'user_sentiment_score_mapped') -> dict:
    logger.info("test_actor_parameters")
    from formulas import actor_formula_v1, actor_formula_v2, null_identity, null_avg, null_linear
    action_df = create_action_df(score_key)
    if action_df.empty:
        print("No action data available.")
        return {}
    n_seeds = int(action_df['seed'].nunique()) if 'seed' in action_df.columns else 0
    action_df = action_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    candidates = [{"function": actor_formula_v1}, {"function": actor_formula_v2}, {"function": null_identity}, {"function": null_avg}, {"function": null_linear}]
    remove_outliers = ["none", "lsquares"]
    splits = ["none", "driver", "action_target"]
    results = []
    for cand in candidates:
        func = cand["function"]
        for ro in remove_outliers:
            for sp in splits:
                def fold_eval(tr, te):
                    fitted = determine_actor_parameters(tr, func, ro, sp, print_process=False)
                    y_true = te['s_user_actor'].to_numpy(dtype=float)
                    y_pred = np.array([actor_skeleton_formula(r['s_init_actor'], r['s_init_action'], r['s_init_target'], sp, func, fitted) for _, r in te.iterrows()], dtype=float)
                    return y_true, y_pred
                cv = _cv_evaluate(action_df, fold_eval)
                fitted_full = determine_actor_parameters(action_df, func, ro, sp, print_process=False)
                rec = {
                    "model": func.__name__,
                    "split": sp,
                    "remove_outliers": ro,
                    "score_key": score_key,
                    "fitted": fitted_full,
                    "cv_mean_mse": cv['cv_mean_mse'],
                    "cv_std_mse": cv['cv_std_mse'],
                    "cv_mean_r2": cv['cv_mean_r2'],
                    "cv_std_r2": cv['cv_std_r2'],
                    "cv_mean_spearman": cv['cv_mean_spearman'],
                    "cv_std_spearman": cv['cv_std_spearman'],
                    "fold_mse": cv['fold_mse'],
                    "fold_r2": cv['fold_r2'],
                    "fold_spearman": cv['fold_spearman'],
                }
                results.append(rec)
                _vprint(f"actor | {func.__name__} | split={sp} | outliers={ro} | cv_mse={rec['cv_mean_mse']:.4f}±{rec['cv_std_mse']:.4f} | cv_r2={(rec['cv_mean_r2'] if rec['cv_mean_r2'] is not None else float('nan')):.4f} | cv_spearman={(rec['cv_mean_spearman'] if rec['cv_mean_spearman'] is not None else float('nan')):.4f}")
    if not results:
        return {}
    best_avg = min(results, key=lambda x: x["cv_mean_mse"]) 
    def _sval(rec):
        s = rec.get('cv_mean_spearman')
        return s if s is not None else -1e9
    best_spearman = max(results, key=_sval)
    return {"best_avg": best_avg, "best_spearman": best_spearman, "all": results, "n_seeds": n_seeds}

def test_target_parameters(score_key: str = 'user_sentiment_score_mapped') -> dict:
    logger.info("test_target_parameters")
    from formulas import target_formula_v1, target_formula_v2, null_identity, null_avg, null_linear
    action_df = create_action_df(score_key)
    if action_df.empty:
        print("No action data available for target model.")
        return {}
    n_seeds = int(action_df['seed'].nunique()) if 'seed' in action_df.columns else 0
    action_df = action_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    candidates = [{"function": target_formula_v1}, {"function": target_formula_v2}, {"function": null_identity}, {"function": null_avg}, {"function": null_linear}]
    remove_outliers = ["none", "lsquares"]
    splits = ["none", "action"]
    results = []
    for cand in candidates:
        func = cand["function"]
        for ro in remove_outliers:
            for sp in splits:
                def fold_eval(tr, te):
                    fitted = determine_target_parameters(tr, func, ro, sp, print_process=False)
                    y_true = te['s_user_target'].to_numpy(dtype=float)
                    y_pred = np.array([target_skeleton_formula(r['s_init_target'], r['s_init_action'], sp, func, fitted) for _, r in te.iterrows()], dtype=float)
                    return y_true, y_pred
                cv = _cv_evaluate(action_df, fold_eval)
                fitted_full = determine_target_parameters(action_df, func, ro, sp, print_process=False)
                rec = {
                    "model": func.__name__,
                    "split": sp,
                    "remove_outliers": ro,
                    "score_key": score_key,
                    "fitted": fitted_full,
                    "cv_mean_mse": cv['cv_mean_mse'],
                    "cv_std_mse": cv['cv_std_mse'],
                    "cv_mean_r2": cv['cv_mean_r2'],
                    "cv_std_r2": cv['cv_std_r2'],
                    "cv_mean_spearman": cv['cv_mean_spearman'],
                    "cv_std_spearman": cv['cv_std_spearman'],
                    "fold_mse": cv['fold_mse'],
                    "fold_r2": cv['fold_r2'],
                    "fold_spearman": cv['fold_spearman'],
                }
                results.append(rec)
                _vprint(f"target | {func.__name__} | split={sp} | outliers={ro} | cv_mse={rec['cv_mean_mse']:.4f}±{rec['cv_std_mse']:.4f} | cv_r2={(rec['cv_mean_r2'] if rec['cv_mean_r2'] is not None else float('nan')):.4f} | cv_spearman={(rec['cv_mean_spearman'] if rec['cv_mean_spearman'] is not None else float('nan')):.4f}")
    if not results:
        return {}
    best_avg = min(results, key=lambda x: x["cv_mean_mse"]) 
    def _sval(rec):
        s = rec.get('cv_mean_spearman')
        return s if s is not None else -1e9
    best_spearman = max(results, key=_sval)
    return {"best_avg": best_avg, "best_spearman": best_spearman, "all": results, "n_seeds": n_seeds}

def test_association_parameters(score_key: str = 'user_sentiment_score_mapped') -> dict:
    logger.info("test_association_parameters")
    from formulas import assoc_formula_v1, assoc_formula_v2, null_identity, null_avg, null_linear
    assoc_df = create_association_df(score_key)
    if assoc_df.empty:
        print("No association data available.")
        return {}
    n_seeds = int(assoc_df['seed'].nunique()) if 'seed' in assoc_df.columns else 0
    assoc_df = assoc_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    candidates = [{"function": assoc_formula_v1}, {"function": assoc_formula_v2}, {"function": null_identity}, {"function": null_avg}, {"function": null_linear}]
    remove_outliers = ["none", "lsquares"]
    splits = ["none", "other", "entity_other"]
    results = []
    for cand in candidates:
        func = cand["function"]
        for ro in remove_outliers:
            for sp in splits:
                def fold_eval(tr, te):
                    fitted = determine_association_parameters(tr, func, ro, sp, print_process=False)
                    y_true = te['s_user_entity'].to_numpy(dtype=float)
                    y_pred = np.array([association_skeleton_formula(r['s_init_entity'], r['s_init_other'], sp, func, fitted) for _, r in te.iterrows()], dtype=float)
                    return y_true, y_pred
                cv = _cv_evaluate(assoc_df, fold_eval)
                fitted_full = determine_association_parameters(assoc_df, func, ro, sp, print_process=False)
                rec = {
                    "model": func.__name__,
                    "split": sp,
                    "remove_outliers": ro,
                    "score_key": score_key,
                    "fitted": fitted_full,
                    "cv_mean_mse": cv['cv_mean_mse'],
                    "cv_std_mse": cv['cv_std_mse'],
                    "cv_mean_r2": cv['cv_mean_r2'],
                    "cv_std_r2": cv['cv_std_r2'],
                    "cv_mean_spearman": cv['cv_mean_spearman'],
                    "cv_std_spearman": cv['cv_std_spearman'],
                    "fold_mse": cv['fold_mse'],
                    "fold_r2": cv['fold_r2'],
                    "fold_spearman": cv['fold_spearman'],
                }
                results.append(rec)
                _vprint(f"assoc | {func.__name__} | split={sp} | outliers={ro} | cv_mse={rec['cv_mean_mse']:.4f}±{rec['cv_std_mse']:.4f} | cv_r2={(rec['cv_mean_r2'] if rec['cv_mean_r2'] is not None else float('nan')):.4f} | cv_spearman={(rec['cv_mean_spearman'] if rec['cv_mean_spearman'] is not None else float('nan')):.4f}")
    if not results:
        return {}
    best_avg = min(results, key=lambda x: x["cv_mean_mse"]) 
    def _sval(rec):
        s = rec.get('cv_mean_spearman')
        return s if s is not None else -1e9
    best_spearman = max(results, key=_sval)
    return {"best_avg": best_avg, "best_spearman": best_spearman, "all": results, "n_seeds": n_seeds}

def test_parent_parameters(score_key: str = 'user_sentiment_score_mapped') -> dict:
    logger.info("test_parent_parameters")
    from formulas import belong_formula_v1, belong_formula_v2, null_identity, null_avg, null_linear
    bel_df = create_belonging_df(score_key)
    if bel_df.empty:
        print("No belonging data available for parent model.")
        return {}
    n_seeds = int(bel_df['seed'].nunique()) if 'seed' in bel_df.columns else 0
    bel_df = bel_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    candidates = [{"function": belong_formula_v1}, {"function": belong_formula_v2}, {"function": null_identity}, {"function": null_avg}, {"function": null_linear}]
    remove_outliers = ["none", "lsquares"]
    splits = ["none", "child", "parent_child"]
    results = []
    for cand in candidates:
        func = cand["function"]
        for ro in remove_outliers:
            for sp in splits:
                def fold_eval(tr, te):
                    fitted = determine_parent_parameters(tr, func, ro, sp, print_process=False)
                    y_true = te['s_user_parent'].to_numpy(dtype=float)
                    y_pred = np.array([parent_skeleton_formula(r['s_init_parent'], r['s_init_child'], sp, func, fitted) for _, r in te.iterrows()], dtype=float)
                    return y_true, y_pred
                cv = _cv_evaluate(bel_df, fold_eval)
                fitted_full = determine_parent_parameters(bel_df, func, ro, sp, print_process=False)
                rec = {
                    "model": func.__name__,
                    "split": sp,
                    "remove_outliers": ro,
                    "score_key": score_key,
                    "fitted": fitted_full,
                    "cv_mean_mse": cv['cv_mean_mse'],
                    "cv_std_mse": cv['cv_std_mse'],
                    "cv_mean_r2": cv['cv_mean_r2'],
                    "cv_std_r2": cv['cv_std_r2'],
                    "cv_mean_spearman": cv['cv_mean_spearman'],
                    "cv_std_spearman": cv['cv_std_spearman'],
                    "fold_mse": cv['fold_mse'],
                    "fold_r2": cv['fold_r2'],
                    "fold_spearman": cv['fold_spearman'],
                }
                results.append(rec)
                _vprint(f"parent | {func.__name__} | split={sp} | outliers={ro} | cv_mse={rec['cv_mean_mse']:.4f}±{rec['cv_std_mse']:.4f} | cv_r2={(rec['cv_mean_r2'] if rec['cv_mean_r2'] is not None else float('nan')):.4f} | cv_spearman={(rec['cv_mean_spearman'] if rec['cv_mean_spearman'] is not None else float('nan')):.4f}")
    if not results:
        return {}
    best_avg = min(results, key=lambda x: x["cv_mean_mse"]) 
    def _sval(rec):
        s = rec.get('cv_mean_spearman')
        return s if s is not None else -1e9
    best_spearman = max(results, key=_sval)
    return {"best_avg": best_avg, "best_spearman": best_spearman, "all": results, "n_seeds": n_seeds}

def test_child_parameters(score_key: str = 'user_sentiment_score_mapped') -> dict:
    logger.info("test_child_parameters")
    from formulas import belong_formula_v1, belong_formula_v2, null_identity, null_avg, null_linear
    bel_df = create_belonging_df(score_key)
    if bel_df.empty:
        print("No belonging data available for child model.")
        return {}
    n_seeds = int(bel_df['seed'].nunique()) if 'seed' in bel_df.columns else 0
    bel_df = bel_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    candidates = [{"function": belong_formula_v1}, {"function": belong_formula_v2}, {"function": null_identity}, {"function": null_avg}, {"function": null_linear}]
    remove_outliers = ["none", "lsquares"]
    splits = ["none", "parent", "parent_child"]
    results = []
    for cand in candidates:
        func = cand["function"]
        for ro in remove_outliers:
            for sp in splits:
                def fold_eval(tr, te):
                    fitted = determine_child_parameters(tr, func, ro, sp, print_process=False)
                    y_true = te['s_user_child'].to_numpy(dtype=float)
                    y_pred = np.array([child_skeleton_formula(r['s_init_child'], r['s_init_parent'], sp, func, fitted) for _, r in te.iterrows()], dtype=float)
                    return y_true, y_pred
                cv = _cv_evaluate(bel_df, fold_eval)
                fitted_full = determine_child_parameters(bel_df, func, ro, sp, print_process=False)
                rec = {
                    "model": func.__name__,
                    "split": sp,
                    "remove_outliers": ro,
                    "score_key": score_key,
                    "fitted": fitted_full,
                    "cv_mean_mse": cv['cv_mean_mse'],
                    "cv_std_mse": cv['cv_std_mse'],
                    "cv_mean_r2": cv['cv_mean_r2'],
                    "cv_std_r2": cv['cv_std_r2'],
                    "cv_mean_spearman": cv['cv_mean_spearman'],
                    "cv_std_spearman": cv['cv_std_spearman'],
                    "fold_mse": cv['fold_mse'],
                    "fold_r2": cv['fold_r2'],
                    "fold_spearman": cv['fold_spearman'],
                }
                results.append(rec)
                _vprint(f"child | {func.__name__} | split={sp} | outliers={ro} | cv_mse={rec['cv_mean_mse']:.4f}±{rec['cv_std_mse']:.4f} | cv_r2={(rec['cv_mean_r2'] if rec['cv_mean_r2'] is not None else float('nan')):.4f} | cv_spearman={(rec['cv_mean_spearman'] if rec['cv_mean_spearman'] is not None else float('nan')):.4f}")
    if not results:
        return {}
    best_avg = min(results, key=lambda x: x["cv_mean_mse"]) 
    def _sval(rec):
        s = rec.get('cv_mean_spearman')
        return s if s is not None else -1e9
    best_spearman = max(results, key=_sval)
    return {"best_avg": best_avg, "best_spearman": best_spearman, "all": results, "n_seeds": n_seeds}

def test_aggregate_parameters(score_key: str = 'user_sentiment_score_mapped') -> dict:
    logger.info("test_aggregate_parameters")
    agg_df = create_aggregate_df(score_key)
    if agg_df.empty:
        print("No aggregate data available.")
        return {}
    n_seeds = int(agg_df['seed'].nunique()) if 'seed' in agg_df.columns else 0
    agg_df = agg_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    losses = [aggregate_error_mse, aggregate_error_softl1, aggregate_error_dynamic_mse, aggregate_error_dynamic_softl1, aggregate_error_logistic_mse, aggregate_error_logistic_softl1]
    results = []
    null_avg_name = 'aggregate_null_average'
    def fold_eval_null(tr, te):
        yt = te['s_user'].to_numpy(dtype=float)
        yp = np.array([float(np.mean(r['s_inits'])) for _, r in te.iterrows()], dtype=float)
        return yt, yp
    cv_null = _cv_evaluate(agg_df, fold_eval_null)
    results.append({
        "model": null_avg_name,
        "split": "none",
        "remove_outliers": "none",
        "score_key": score_key,
        "params": [],
        "cv_mean_mse": cv_null['cv_mean_mse'],
        "cv_std_mse": cv_null['cv_std_mse'],
        "cv_mean_r2": cv_null['cv_mean_r2'],
        "cv_std_r2": cv_null['cv_std_r2'],
        "cv_mean_spearman": cv_null['cv_mean_spearman'],
        "cv_std_spearman": cv_null['cv_std_spearman'],
        "fold_mse": cv_null['fold_mse'],
        "fold_r2": cv_null['fold_r2'],
        "fold_spearman": cv_null['fold_spearman'],
    })
    _vprint(f"aggregate | {null_avg_name} | cv_mse={cv_null['cv_mean_mse']:.4f}±{cv_null['cv_std_mse']:.4f} | cv_r2={(cv_null['cv_mean_r2'] if cv_null['cv_mean_r2'] is not None else float('nan')):.4f} | cv_spearman={(cv_null['cv_mean_spearman'] if cv_null['cv_mean_spearman'] is not None else float('nan')):.4f}")
    for loss_fn in losses:
        name = loss_fn.__name__
        def fold_eval(tr, te):
            fitted = determine_aggregate_parameters(tr, loss_fn, print_process=False)
            if 'dynamic' in loss_fn.__name__:
                func = aggregate_formula_dynamic
            elif 'logistic' in loss_fn.__name__:
                func = aggregate_formula_logistic
            else:
                func = aggregate_formula
            yt = te['s_user'].to_numpy(dtype=float)
            yp = np.array([func(r['s_inits'], fitted['params']) for _, r in te.iterrows()], dtype=float)
            return yt, yp
        cv = _cv_evaluate(agg_df, fold_eval)
        fitted_full = determine_aggregate_parameters(agg_df, loss_fn, print_process=False)
        if 'dynamic' in loss_fn.__name__:
            func = aggregate_formula_dynamic
        elif 'logistic' in loss_fn.__name__:
            func = aggregate_formula_logistic
        else:
            func = aggregate_formula
        results.append({
            "model": name,
            "split": "none",
            "remove_outliers": "none",
            "score_key": score_key,
            "params": fitted_full['params'],
            "cv_mean_mse": cv['cv_mean_mse'],
            "cv_std_mse": cv['cv_std_mse'],
            "cv_mean_r2": cv['cv_mean_r2'],
            "cv_std_r2": cv['cv_std_r2'],
            "cv_mean_spearman": cv['cv_mean_spearman'],
            "cv_std_spearman": cv['cv_std_spearman'],
            "fold_mse": cv['fold_mse'],
            "fold_r2": cv['fold_r2'],
            "fold_spearman": cv['fold_spearman'],
        })
        _vprint(f"aggregate | {name} | cv_mse={cv['cv_mean_mse']:.4f}±{cv['cv_std_mse']:.4f} | cv_r2={(cv['cv_mean_r2'] if cv['cv_mean_r2'] is not None else float('nan')):.4f} | cv_spearman={(cv['cv_mean_spearman'] if cv['cv_mean_spearman'] is not None else float('nan')):.4f}")
    if not results:
        return {}
    best_avg = min(results, key=lambda x: x["cv_mean_mse"]) if results else {}
    def _sval(rec):
        s = rec.get('cv_mean_spearman')
        return s if s is not None else -1e9
    best_spearman = max(results, key=_sval) if results else {}
    return {"best_avg": best_avg, "best_spearman": best_spearman, "all": results, "n_seeds": n_seeds}


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
            if not data:
                continue
            all_runs = data.get('all') or []
            ranked = _rank_models_by_spearman(all_runs)
            if ranked:
                best = ranked[0].get('best_run')
                selected_by = 'pareto_topsis_cv'
            else:
                best = None
                selected_by = 'none'
            if not best:
                continue
            n_seeds = int(data.get('n_seeds') or 0)
            entry = {
                "category": category,
                "score_key": score_key,
                "model": best.get("model"),
                "split": best.get("split", "none"),
                "remove_outliers": best.get("remove_outliers", "none"),
                "cv_mean_mse": float(best.get("cv_mean_mse", float('inf'))),
                "cv_std_mse": float(best.get("cv_std_mse", 0.0)),
                "cv_mean_r2": best.get("cv_mean_r2"),
                "cv_std_r2": float(best.get("cv_std_r2", 0.0)),
                "cv_mean_spearman": best.get("cv_mean_spearman"),
                "cv_std_spearman": float(best.get("cv_std_spearman", 0.0)),
                "n_seeds": n_seeds,
                "selected_by": selected_by,
                "fold_mse": best.get('fold_mse'),
                "fold_r2": best.get('fold_r2'),
                "fold_spearman": best.get('fold_spearman'),
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
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.dirname(out_path)
    os.makedirs(base_dir, exist_ok=True)
    payload = {"version": 2, "saved_at": datetime.now(timezone.utc).isoformat(), "entries": entries}
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=4)
    snapshot_path = os.path.join(base_dir, f"all_optimal_parameters_{ts}.json")
    with open(snapshot_path, 'w') as f:
        json.dump(payload, f, indent=4)
    full_results_path = os.path.join(base_dir, f"all_optimal_full_results_{ts}.json")
    with open(full_results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)

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

def get_actor_function(score_key: str = 'user_normalized_sentiment_scores') -> Callable:
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

def get_target_function(score_key: str = 'user_normalized_sentiment_scores') -> Callable:
    confs = load_optimal_parameters()
    conf = confs.get(('target', score_key)) or confs.get(('target', 'user_normalized_sentiment_scores'))
    if not conf:
        return None
    func = _formula_by_name(conf['model'])
    split = conf.get('split', 'none')
    fitted = conf.get('params_by_key', {})
    def f(s_target, s_action):
        return target_skeleton_formula(s_target, s_action, split, func, fitted)
    return f

def get_association_function(score_key: str = 'user_normalized_sentiment_scores') -> Callable:
    confs = load_optimal_parameters()
    conf = confs.get(('association', score_key)) or confs.get(('association', 'user_normalized_sentiment_scores'))
    if not conf:
        return None
    func = _formula_by_name(conf['model'])
    split = conf.get('split', 'none')
    fitted = conf.get('params_by_key', {})
    def f(s_entity, s_other):
        return association_skeleton_formula(s_entity, s_other, split, func, fitted)
    return f

def get_parent_function(score_key: str = 'user_normalized_sentiment_scores') -> Callable:
    confs = load_optimal_parameters()
    conf = confs.get(('parent', score_key)) or confs.get(('parent', 'user_normalized_sentiment_scores'))
    if not conf:
        return None
    func = _formula_by_name(conf['model'])
    split = conf.get('split', 'none')
    fitted = conf.get('params_by_key', {})
    def f(s_parent, s_child):
        return parent_skeleton_formula(s_parent, s_child, split, func, fitted)
    return f

def get_child_function(score_key: str = 'user_normalized_sentiment_scores') -> Callable:
    confs = load_optimal_parameters()
    conf = confs.get(('child', score_key)) or confs.get(('child', 'user_normalized_sentiment_scores'))
    if not conf:
        return None
    func = _formula_by_name(conf['model'])
    split = conf.get('split', 'none')
    fitted = conf.get('params_by_key', {})
    def f(s_child, s_parent):
        return child_skeleton_formula(s_child, s_parent, split, func, fitted)
    return f

def get_aggregate_function(score_key: str = 'user_normalized_sentiment_scores') -> Callable:
    confs = load_optimal_parameters()
    conf = confs.get(('aggregate', score_key)) or confs.get(('aggregate', 'user_normalized_sentiment_scores'))
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

def _rank_models_by_average_mse(results_list: list[dict], mse_key: str = "avg_mse") -> list[dict]:
    if not results_list:
        return []
    buckets: dict[str, list[dict]] = {}
    for r in results_list:
        m = r.get('model')
        if not m:
            continue
        buckets.setdefault(m, []).append(r)
    ranked = []
    for model, runs in buckets.items():
        mses = [x[mse_key] for x in runs if np.isfinite(x.get(mse_key, np.inf))]
        if not mses:
            continue
        avg_mse = float(np.mean(mses))
        best_run = min(runs, key=lambda x: x.get(mse_key, np.inf))
        ranked.append({
            'model': model,
            'avg_mse': avg_mse,
            'n_configs': len(mses),
            'best_mse': float(best_run.get(mse_key, np.inf)),
            'best_split': best_run.get('split', 'none'),
            'best_outliers': best_run.get('remove_outliers', 'none'),
            'best_run': best_run,
        })
    ranked.sort(key=lambda x: x['avg_mse'])
    return ranked

def _rank_models_by_spearman(results_list: list[dict]) -> list[dict]:
    if not results_list:
        return []
    def _valid(r):
        s = r.get('cv_mean_spearman')
        r2 = r.get('cv_mean_r2')
        mse = r.get('cv_mean_mse', float('inf'))
        return (s is not None) and np.isfinite(mse) and (r2 is not None)
    runs = [r for r in results_list if r.get('model') and _valid(r)]
    if not runs:
        return []
    objs = []
    for r in runs:
        objs.append((r, float(r.get('cv_mean_spearman')), float(r.get('cv_mean_mse')), float(r.get('cv_mean_r2'))))
    pareto = []
    for i, (ri, si, mi, r2i) in enumerate(objs):
        dominated = False
        for j, (rj, sj, mj, r2j) in enumerate(objs):
            if i == j:
                continue
            if (mj <= mi) and (sj >= si) and (r2j >= r2i) and ((mj < mi) or (sj > si) or (r2j > r2i)):
                dominated = True
                break
        if not dominated:
            pareto.append((ri, si, mi, r2i))
    if not pareto:
        pareto = objs
    X = np.array([[s, m, r2] for (_, s, m, r2) in pareto], dtype=float)
    if len(X) == 0:
        return []
    norm = np.linalg.norm(X, axis=0)
    norm[norm == 0] = 1.0
    R = X / norm
    w = np.array([1.0, 1.0, 1.0], dtype=float)
    V = R * w
    ideal_best = np.array([np.max(V[:, 0]), np.min(V[:, 1]), np.max(V[:, 2])], dtype=float)
    ideal_worst = np.array([np.min(V[:, 0]), np.max(V[:, 1]), np.min(V[:, 2])], dtype=float)
    S_plus = np.linalg.norm(V - ideal_best, axis=1)
    S_minus = np.linalg.norm(V - ideal_worst, axis=1)
    CC = S_minus / (S_plus + S_minus + 1e-12)
    ranked = []
    for idx, (r, s, m, r2) in enumerate(pareto):
        ranked.append({
            'model': r.get('model'),
            's': s,
            'cv_mean_mse': m,
            'cv_mean_r2': r2,
            'n_configs': int(sum(1 for q in runs if q.get('model') == r.get('model'))),
            'best_split': r.get('split', 'none'),
            'best_outliers': r.get('remove_outliers', 'none'),
            'best_run': r,
            'topsis': float(CC[idx]),
        })
    best_by_model = {}
    for row in ranked:
        key = row['model']
        prev = best_by_model.get(key)
        if (prev is None) or (row['topsis'] > prev['topsis']):
            best_by_model[key] = row
    ranked_unique = list(best_by_model.values())
    ranked_unique.sort(key=lambda x: x['topsis'], reverse=True)
    return ranked_unique

def _rank_models_by_topsis_all(results_list: list[dict]) -> list[dict]:
    if not results_list:
        return []
    runs = []
    for r in results_list:
        if not r.get('model'):
            continue
        s = r.get('cv_mean_spearman')
        r2 = r.get('cv_mean_r2')
        mse = r.get('cv_mean_mse', float('inf'))
        if not np.isfinite(mse) or s is None:
            continue
        r2v = float(r2) if (isinstance(r2, (int, float)) and not isinstance(r2, bool)) else -1e9
        runs.append((r, float(s), float(mse), r2v))
    if not runs:
        return []
    X = np.array([[s, m, r2] for (_, s, m, r2) in runs], dtype=float)
    norm = np.linalg.norm(X, axis=0)
    norm[norm == 0] = 1.0
    R = X / norm
    w = np.array([1.0, 1.0, 1.0], dtype=float)
    V = R * w
    ideal_best = np.array([np.max(V[:, 0]), np.min(V[:, 1]), np.max(V[:, 2])], dtype=float)
    ideal_worst = np.array([np.min(V[:, 0]), np.max(V[:, 1]), np.min(V[:, 2])], dtype=float)
    S_plus = np.linalg.norm(V - ideal_best, axis=1)
    S_minus = np.linalg.norm(V - ideal_worst, axis=1)
    CC = S_minus / (S_plus + S_minus + 1e-12)
    ranked = []
    for idx, (r, s, m, r2) in enumerate(runs):
        ranked.append({
            'model': r.get('model'),
            's': s,
            'cv_mean_mse': m,
            'cv_mean_r2': r2 if r2 > -1e9/2 else None,
            'n_configs': int(sum(1 for q in results_list if q.get('model') == r.get('model'))),
            'best_split': r.get('split', 'none'),
            'best_outliers': r.get('remove_outliers', 'none'),
            'best_run': r,
            'topsis': float(CC[idx]),
        })
    best_by_model = {}
    for row in ranked:
        key = row['model']
        prev = best_by_model.get(key)
        if (prev is None) or (row['topsis'] > prev['topsis']):
            best_by_model[key] = row
    ranked_unique = list(best_by_model.values())
    ranked_unique.sort(key=lambda x: x['topsis'], reverse=True)
    return ranked_unique

def print_ranked_summary(all_results: dict) -> None:
    if not all_results:
        print("[red]No results to summarize.[/red]")
        return
    console = Console()
    for score_key, categories in all_results.items():
        console.rule(f"Results — score_key={score_key}")
        for category, payload in categories.items():
            if not payload or not payload.get('all'):
                continue
            
            ranked_by_s = _rank_models_by_spearman(payload['all'])
            if not ranked_by_s:
                continue
            n_seeds = int(payload.get('n_seeds') or 0)
            table = Table(title=f"{category} — Pareto + TOPSIS Rankings (N={n_seeds})")
            table.add_column("Rank", justify="right")
            table.add_column("Model", style="bold")
            table.add_column("S", justify="right")
            table.add_column("CV MSE", justify="right")
            table.add_column("CV R2", justify="right")
            table.add_column("Configs", justify="right")
            table.add_column("Best Config", justify="left")
            table.add_column("Params/IO", justify="left", overflow="fold")

            for i, row in enumerate(ranked_by_s, start=1):
                mark = "⭐" if i == 1 else ""
                best_detail = row['best_run']
                test_mse = row.get('cv_mean_mse', float('inf'))
                test_mse_std = row['best_run'].get('cv_std_mse') if row.get('best_run') else None
                test_r2 = row.get('cv_mean_r2')
                test_r2_std = row['best_run'].get('cv_std_r2') if row.get('best_run') else None

                pview, io = '', ''
                if best_detail:
                    split = best_detail.get('split', 'none')
                    outliers = best_detail.get('remove_outliers', 'none')
                    config_str = f"{split}/{outliers}"
                    if category != 'aggregate':
                        fitted = best_detail.get('fitted', {})
                        parts = []
                        for k, v in fitted.items():
                            if v is None:
                                continue
                            arr = v.get('params') if isinstance(v, dict) else v
                            if isinstance(arr, np.ndarray):
                                arr = arr.tolist()
                            if isinstance(arr, (list, tuple)):
                                arr_str = '[' + ','.join(f"{float(x):.3f}" for x in arr) + ']'
                            elif arr is None:
                                arr_str = 'None'
                            else:
                                arr_str = str(arr)
                            parts.append(f"{k}={arr_str}")
                        pview = '; '.join(parts) if parts else 'params'
                        if category == 'actor':
                            io = "(s_actor, s_action*s_target)->s_user_actor"
                        elif category == 'target':
                            io = "(s_target, s_action)->s_user_target"
                        elif category == 'association':
                            io = "(s_entity, s_other)->s_user_entity"
                        elif category == 'parent':
                            io = "(s_parent, s_child)->s_user_parent"
                        elif category == 'child':
                            io = "(s_child, s_parent)->s_user_child"
                    else:
                        params = best_detail.get('params')
                        if isinstance(params, np.ndarray):
                            params = params.tolist()
                        if isinstance(params, (list, tuple)):
                            pview = '[' + ','.join(f"{float(x):.3f}" for x in params) + ']'
                        else:
                            pview = str(params)
                        io = "(s_inits)->s_user"
                        config_str = "none/none"
                    
                    if len(pview) > 80:
                        pview = pview[:80] + '…'
                else:
                    config_str = "none/none"

                test_mse_str = f"{test_mse:.4f}" if np.isfinite(test_mse) else "inf"
                if isinstance(test_mse_std, (int, float)) and np.isfinite(test_mse_std) and test_mse_std > 0:
                    test_mse_str += f" ±{test_mse_std:.3f}"
                test_r2_str = f"{test_r2:.4f}" if isinstance(test_r2, (int, float)) and not isinstance(test_r2, bool) and np.isfinite(test_r2) else "-"
                if isinstance(test_r2_std, (int, float)) and np.isfinite(test_r2_std) and test_r2_std > 0:
                    test_r2_str += f" ±{test_r2_std:.3f}"
                s_str = f"{row.get('s'):.4f}" if row.get('s') is not None else "-"

                table.add_row(
                    str(i),
                    f"{row['model']} {mark}",
                    s_str,
                    test_mse_str,
                    test_r2_str,
                    str(row['n_configs']),
                    config_str,
                    f"{pview} {io}",
                )
            console.print(table)

            all_ranked = _rank_models_by_topsis_all(payload['all'])
            if all_ranked and len(all_ranked) > len(ranked_by_s):
                table2 = Table(title=f"{category} — All models (including dominated) — TOPSIS (N={n_seeds})")
                table2.add_column("Rank", justify="right")
                table2.add_column("Model", style="bold")
                table2.add_column("S", justify="right")
                table2.add_column("CV MSE", justify="right")
                table2.add_column("CV R2", justify="right")
                table2.add_column("Configs", justify="right")
                table2.add_column("Best Config", justify="left")
                table2.add_column("Params/IO", justify="left", overflow="fold")

                for i, row in enumerate(all_ranked, start=1):
                    mark = "⭐" if i == 1 else ""
                    best_detail = row['best_run']
                    test_mse = row.get('cv_mean_mse', float('inf'))
                    test_mse_std = row['best_run'].get('cv_std_mse') if row.get('best_run') else None
                    test_r2 = row.get('cv_mean_r2')
                    test_r2_std = row['best_run'].get('cv_std_r2') if row.get('best_run') else None

                    pview, io = '', ''
                    if best_detail:
                        split = best_detail.get('split', 'none')
                        outliers = best_detail.get('remove_outliers', 'none')
                        config_str = f"{split}/{outliers}"
                        if category != 'aggregate':
                            fitted = best_detail.get('fitted', {})
                            parts = []
                            for k, v in fitted.items():
                                if v is None:
                                    continue
                                arr = v.get('params') if isinstance(v, dict) else v
                                if isinstance(arr, np.ndarray):
                                    arr = arr.tolist()
                                if isinstance(arr, (list, tuple)):
                                    arr_str = '[' + ','.join(f"{float(x):.3f}" for x in arr) + ']'
                                elif arr is None:
                                    arr_str = 'None'
                                else:
                                    arr_str = str(arr)
                                parts.append(f"{k}={arr_str}")
                            pview = '; '.join(parts) if parts else 'params'
                            if category == 'actor':
                                io = "(s_actor, s_action*s_target)->s_user_actor"
                            elif category == 'target':
                                io = "(s_target, s_action)->s_user_target"
                            elif category == 'association':
                                io = "(s_entity, s_other)->s_user_entity"
                            elif category == 'parent':
                                io = "(s_parent, s_child)->s_user_parent"
                            elif category == 'child':
                                io = "(s_child, s_parent)->s_user_child"
                        else:
                            params = best_detail.get('params')
                            if isinstance(params, np.ndarray):
                                params = params.tolist()
                            if isinstance(params, (list, tuple)):
                                pview = '[' + ','.join(f"{float(x):.3f}" for x in params) + ']'
                            else:
                                pview = str(params)
                            io = "(s_inits)->s_user"
                            config_str = "none/none"
                        if len(pview) > 80:
                            pview = pview[:80] + '…'
                    else:
                        config_str = "none/none"

                    test_mse_str = f"{test_mse:.4f}" if np.isfinite(test_mse) else "inf"
                    if isinstance(test_mse_std, (int, float)) and np.isfinite(test_mse_std) and test_mse_std > 0:
                        test_mse_str += f" ±{test_mse_std:.3f}"
                    test_r2_str = f"{test_r2:.4f}" if isinstance(test_r2, (int, float)) and not isinstance(test_r2, bool) and np.isfinite(test_r2) else "-"
                    if isinstance(test_r2_std, (int, float)) and np.isfinite(test_r2_std) and test_r2_std > 0:
                        test_r2_str += f" ±{test_r2_std:.3f}"
                    s_str = f"{row.get('s'):.4f}" if row.get('s') is not None else "-"

                    table2.add_row(
                        str(i),
                        f"{row['model']} {mark}",
                        s_str,
                        test_mse_str,
                        test_r2_str,
                        str(row['n_configs']),
                        config_str,
                        f"{pview} {io}",
                    )
                console.print(table2)


if __name__ == '__main__':

    print_optimal_formulas_math()
    try:
        show_aggregate_weight_slider()
    except Exception as e:
        print(f"[yellow]Could not show aggregate weight slider: {e}[/yellow]")