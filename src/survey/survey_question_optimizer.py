from typing import Literal, Callable, List
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
import inspect
import sys
from pathlib import Path
import pickle

CONFIG = {
    'test_train_split': 0.2,
    'random_seed': 42,
    'test_data_save_location': 'src/survey/test_data/',
    'enable_test_split': True
}

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from formulas import SentimentFormula, actor_formula_v2, target_formula_v2, assoc_formula_v2, belong_formula_v2
except ImportError:
    from .formulas import SentimentFormula, actor_formula_v2, target_formula_v2, assoc_formula_v2, belong_formula_v2

ssl._create_default_https_context = ssl._create_unverified_context

url = 'https://docs.google.com/spreadsheets/d/1xAvDLhU0w-p2hAZ49QYM7-XBMQCek0zVYJWpiN1Mvn0/export?format=csv&gid=0'
df = pd.read_csv(url)

def create_test_train_split(dataframe):
    if not CONFIG['enable_test_split']:
        return dataframe, pd.DataFrame()
    
    np.random.seed(CONFIG['random_seed'])
    
    test_size = int(len(dataframe) * CONFIG['test_train_split'])
    indices = np.random.permutation(len(dataframe))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    test_df = dataframe.iloc[test_indices].copy()
    train_df = dataframe.iloc[train_indices].copy()
    
    os.makedirs(CONFIG['test_data_save_location'], exist_ok=True)
    test_df.to_csv(os.path.join(CONFIG['test_data_save_location'], 'test_data.csv'), index=False)
    
    print(f"Data split: {len(train_df)} training samples, {len(test_df)} test samples")
    print(f"Test data saved to: {CONFIG['test_data_save_location']}")
    
    return train_df, test_df

df_train, df_test = create_test_train_split(df)
df = df_train

intensity_map_string = {
    'very': 0.85,
    'strong': 0.6,
    'moderate': 0.4,
    'slight': 0.2,
    'neutral': 0.0
}

intensity_map_integer = {
    4: 0.85,
    3: 0.6,
    2: 0.4,
    1: 0.2,
    0: 0.0
}

sentiment_sign_map = {'positive': 1, 'negative': -1}

def associate_sentiment_integer(integer):
    if not isinstance(integer, int):
        integer = int(integer)
    return intensity_map_integer.get(abs(integer), 0) * ((integer > 0) - (integer < 0))

def fit(formula, X, y, bounds, remove_outliers_method:Literal['lsquares', 'droptop', 'none']='none'):
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
    calibration_df = df[df['item_type'].str.contains('calibration', na=False)].copy()
    user_final_normalizers = {}
    for seed in df['seed'].unique():
        cal_m, cal_c = None, None
        user_cal_data = calibration_df[calibration_df['seed'] == seed]
        pos_response = user_cal_data[user_cal_data['code_key'] == 'calibration_positive']
        neg_response = user_cal_data[user_cal_data['code_key'] == 'calibration_negative']
        if not pos_response.empty and not neg_response.empty:
            try:
                desc_pos = json.loads(pos_response['description'].iloc[0])
                desc_neg = json.loads(neg_response['description'].iloc[0])
                y1 = desc_pos.get('ground_truth')
                y2 = desc_neg.get('ground_truth')
                x1 = pos_response['user_sentiment_score'].iloc[0]
                x2 = neg_response['user_sentiment_score'].iloc[0]
                if x1 is not None and x2 is not None and y1 is not None and y2 is not None and x1 != x2:
                    cal_m = (y2 - y1) / (x2 - x1)
                    cal_c = y1 - cal_m * x1
            except (json.JSONDecodeError, KeyError, IndexError):
                pass
        ref_m, ref_c = None, None
        user_ref_data = reference_df[reference_df['seed'] == seed]
        if len(user_ref_data) >= 2:
            X = user_ref_data[['user_sentiment_score']]
            y = user_ref_data['ground_truth_stimulus_sentiment']
            model = LinearRegression().fit(X, y)
            ref_m, ref_c = model.coef_[0], model.intercept_
        if cal_m is not None and cal_c is not None:
            user_final_normalizers[seed] = (cal_m, cal_c)
        elif ref_m is not None and ref_c is not None:
            user_final_normalizers[seed] = (ref_m, ref_c)
        else:
            user_final_normalizers[seed] = (1/4, 0)
    def apply_final_normalization(row):
        m, c = user_final_normalizers.get(row['seed'], (1/4, 0))
        return m * row['user_sentiment_score'] + c
    df['user_normalized_sentiment_scores'] = df.apply(apply_final_normalization, axis=1)
    df['user_sentiment_score_mapped'] = df['user_sentiment_score'].apply(associate_sentiment_integer)
    return df, user_final_normalizers

def fit_compound(formula, X, y, remove_outliers_method:Literal['lsquares', 'droptop', 'none']='none'):
    sig = inspect.signature(formula)
    params = sig.parameters
    num_params = len(params) - 1
    bounds = ([0] + [-5] * (num_params - 1), [1] + [5] * (num_params - 1))
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


def actor_formula_v1(X, lambda_actor, w, b):
    s_init_actor, driver = X
    s_new = lambda_actor * s_init_actor + (1 - lambda_actor) * w * driver + b
    return np.tanh(s_new)

def actor_formula_v2(X, w_actor, w_driver, b):
    s_init_actor, driver = X
    s_new = w_actor * s_init_actor + w_driver * driver + b
    return np.tanh(s_new)

def determine_actor_parameters(action_model_df: pd.DataFrame,
                                function: SentimentFormula,
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
            pos_driver_params = fit_compound(function.function, X, y, remove_outlier_method)
            output['pos_driver_params'] = pos_driver_params
        if not neg_driver_df.empty:
            X = neg_driver_df[['s_init_actor', 'driver']].to_numpy().T
            y = neg_driver_df['s_user_actor'].to_numpy()
            neg_driver_params = fit_compound(function.function, X, y, remove_outlier_method)
            output['neg_driver_params'] = neg_driver_params
    elif splits == 'action_target':
        if not pos_action_pos_action_model.empty:
            X = pos_action_pos_action_model[['s_init_actor', 'driver']].to_numpy().T
            y = pos_action_pos_action_model['s_user_actor'].to_numpy()
            pos_pos_params = fit_compound(function.function, X, y, remove_outlier_method)
            output['pos_pos_params'] = pos_pos_params
        if not pos_action_neg_action_model.empty:
            X = pos_action_neg_action_model[['s_init_actor', 'driver']].to_numpy().T
            y = pos_action_neg_action_model['s_user_actor'].to_numpy()
            pos_neg_params = fit_compound(function.function, X, y, remove_outlier_method)
            output['pos_neg_params'] = pos_neg_params
        if not neg_action_pos_action_model.empty:
            X = neg_action_pos_action_model[['s_init_actor', 'driver']].to_numpy().T
            y = neg_action_pos_action_model['s_user_actor'].to_numpy()
            neg_pos_params = fit_compound(function.function, X, y, remove_outlier_method)
            output['neg_pos_params'] = neg_pos_params
        if not neg_action_neg_action_model.empty:
            X = neg_action_neg_action_model[['s_init_actor', 'driver']].to_numpy().T
            y = neg_action_neg_action_model['s_user_actor'].to_numpy()
            neg_neg_params = fit_compound(function.function, X, y, remove_outlier_method)
            output['neg_neg_params'] = neg_neg_params
    else:
        X = action_model_df[['s_init_actor', 'driver']].to_numpy().T
        y = action_model_df['s_user_actor'].to_numpy()
        output['params'] = fit_compound(function.function, X, y, remove_outlier_method)
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
    function.set_params(output.get('params', {}).get('params', None) if output.get('params') else None)
    return output, function

def target_formula_v1(X, lambda_target, w, b):
    s_init_target, s_action = X
    s_new = lambda_target * s_init_target + (1 - lambda_target) * w * s_action + b
    return np.tanh(s_new)

def target_formula_v2(X, w_target, w_action, b):
    s_init_target, s_action = X
    s_new = w_target * s_init_target + w_action * s_action + b
    return np.tanh(s_new)

def determine_target_parameters(action_model_df: pd.DataFrame,
                                function: SentimentFormula,
                                remove_outlier_method: Literal['lsquares', 'droptop', 'none'] = 'none',
                                splits: Literal['none', 'driver', 'action_target'] = 'none',
                                print_process=False) -> pd.DataFrame:
    pos_action_df = action_model_df[action_model_df['s_init_action'] > 0]
    neg_action_df = action_model_df[action_model_df['s_init_action'] <= 0]

    output = {}
    if splits == 'driver':
        if not pos_action_df.empty:
            X = pos_action_df[['s_init_target', 's_init_action']].to_numpy().T
            y = pos_action_df['s_user_target'].to_numpy()
            pos_driver_params = fit_compound(function.function, X, y, remove_outlier_method)
            output['pos_driver_params'] = pos_driver_params
        if not neg_action_df.empty:
            X = neg_action_df[['s_init_target', 's_init_action']].to_numpy().T
            y = neg_action_df['s_user_target'].to_numpy()
            neg_driver_params = fit_compound(function.function, X, y, remove_outlier_method)
            output['neg_driver_params'] = neg_driver_params
    else:
        X = action_model_df[['s_init_target', 's_init_action']].to_numpy().T
        y = action_model_df['s_user_target'].to_numpy()
        output['params'] = fit_compound(function.function, X, y, remove_outlier_method)
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
    function.set_params(output.get('params', {}).get('params', None) if output.get('params') else None)
    return output, function


def assoc_formula_v1(X, lambda_val, w, b):
    s_init, s_other = X
    s_new = lambda_val * s_init + (1 - lambda_val) * w * s_other + b
    return np.tanh(s_new)

def assoc_formula_v2(X, w_entity, w_other, b):
    s_init, s_other = X
    s_new = w_entity * s_init + w_other * s_other + b
    return np.tanh(s_new)

def determine_association_parameters(association_model_df: pd.DataFrame,
                                        function: SentimentFormula,
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
            pos_params = fit_compound(function.function, X, y, remove_outlier_method)
            output['pos_params'] = pos_params
        if not neg_entity_df.empty:
            X = neg_entity_df[['s_init_entity', 's_init_other']].to_numpy().T
            y = neg_entity_df['s_user_entity'].to_numpy()
            neg_params = fit_compound(function.function, X, y, remove_outlier_method)
            output['neg_params'] = neg_params
    elif splits == 'entity_other':
        if not pos_entity_pos_other_df.empty:
            X = pos_entity_pos_other_df[['s_init_entity', 's_init_other']].to_numpy().T
            y = pos_entity_pos_other_df['s_user_entity'].to_numpy()
            pos_params = fit_compound(function.function, X, y, remove_outlier_method)
            output['pos_params'] = pos_params
        if not neg_entity_pos_other_df.empty:
            X = neg_entity_pos_other_df[['s_init_entity', 's_init_other']].to_numpy().T
            y = neg_entity_pos_other_df['s_user_entity'].to_numpy()
            neg_params = fit_compound(function.function, X, y, remove_outlier_method)
            output['neg_params'] = neg_params
        if not pos_entity_neg_other_df.empty:
            X = pos_entity_neg_other_df[['s_init_entity', 's_init_other']].to_numpy().T
            y = pos_entity_neg_other_df['s_user_entity'].to_numpy()
            pos_neg_params = fit_compound(function.function, X, y, remove_outlier_method)
            output['pos_neg_params'] = pos_neg_params
        if not neg_entity_neg_other_df.empty:
            X = neg_entity_neg_other_df[['s_init_entity', 's_init_other']].to_numpy().T
            y = neg_entity_neg_other_df['s_user_entity'].to_numpy()
            neg_neg_params = fit_compound(function.function, X, y, remove_outlier_method)
            output['neg_neg_params'] = neg_neg_params
    else:
        X = association_model_df[['s_init_entity', 's_init_other']].to_numpy().T
        y = association_model_df['s_user_entity'].to_numpy()
        output['params'] = fit_compound(function.function, X, y, remove_outlier_method)
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
    function.set_params(output.get('params', {}).get('params', None) if output.get('params') else None)
    return output, function


def belong_formula_v1(X, lambda_parent, w, b):
    s_entity, s_other = X
    s_new = lambda_parent * s_entity + (1 - lambda_parent) * w * s_other + b
    return np.tanh(s_new)

def belong_formula_v2(X, w_parent, w_child, b):
    s_entity, s_child = X
    s_new = w_parent * s_entity + w_child * s_child + b
    return np.tanh(s_new)

def determine_parent_parameters(belonging_model_df: pd.DataFrame,
                                    function: SentimentFormula,
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
            pos_params = fit_compound(function.function, X, y, remove_outlier_method)
            output['pos_params'] = pos_params
        if not neg_parent_df.empty:
            X = neg_parent_df[['s_init_parent', 's_init_child']].to_numpy().T
            y = neg_parent_df['s_user_parent'].to_numpy()
            neg_params = fit_compound(function.function, X, y, remove_outlier_method)
            output['neg_params'] = neg_params
    elif splits == 'parent_child':
        if not pos_parent_neg_child_df.empty:
            X = pos_parent_neg_child_df[['s_init_parent', 's_init_child']].to_numpy().T
            y = pos_parent_neg_child_df['s_user_parent'].to_numpy()
            pos_neg_params = fit_compound(function.function, X, y, remove_outlier_method)
            output['pos_neg_params'] = pos_neg_params
        if not neg_parent_neg_child_df.empty:
            X = neg_parent_neg_child_df[['s_init_parent', 's_init_child']].to_numpy().T
            y = neg_parent_neg_child_df['s_user_parent'].to_numpy()
            neg_neg_params = fit_compound(function.function, X, y, remove_outlier_method)
            output['neg_neg_params'] = neg_neg_params
        if not pos_parent_pos_child_df.empty:
            X = pos_parent_pos_child_df[['s_init_parent', 's_init_child']].to_numpy().T
            y = pos_parent_pos_child_df['s_user_parent'].to_numpy()
            pos_pos_params = fit_compound(function.function, X, y, remove_outlier_method)
            output['pos_pos_params'] = pos_pos_params
        if not neg_parent_pos_child_df.empty:
            X = neg_parent_pos_child_df[['s_init_parent', 's_init_child']].to_numpy().T
            y = neg_parent_pos_child_df['s_user_parent'].to_numpy()
            neg_pos_params = fit_compound(function.function, X, y, remove_outlier_method)
            output['neg_pos_params'] = neg_pos_params
    else:
        X = belonging_model_df[['s_init_parent', 's_init_child']].to_numpy().T
        y = belonging_model_df['s_user_parent'].to_numpy()
        output['params'] = fit_compound(function.function, X, y, remove_outlier_method)
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
    function.set_params(output.get('params', {}).get('params', None) if output.get('params') else None)
    return output, function

def determine_child_parameters(belonging_model_df: pd.DataFrame,
                                    function: SentimentFormula,
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
            pos_params = fit_compound(function.function, X, y, remove_outlier_method)
            output['pos_params'] = pos_params
        if not neg_child_df.empty:
            X = neg_child_df[['s_init_child', 's_init_parent']].to_numpy().T
            y = neg_child_df['s_user_child'].to_numpy()
            neg_params = fit_compound(function.function, X, y, remove_outlier_method)
            output['neg_params'] = neg_params
    elif splits == 'parent_child':
        if not pos_child_neg_parent_df.empty:
            X = pos_child_neg_parent_df[['s_init_child', 's_init_parent']].to_numpy().T
            y = pos_child_neg_parent_df['s_user_child'].to_numpy()
            pos_neg_params = fit_compound(function.function, X, y, remove_outlier_method)
            output['pos_neg_params'] = pos_neg_params
        if not neg_child_neg_parent_df.empty:
            X = neg_child_neg_parent_df[['s_init_child', 's_init_parent']].to_numpy().T
            y = neg_child_neg_parent_df['s_user_child'].to_numpy()
            neg_neg_params = fit_compound(function.function, X, y, remove_outlier_method)
            output['neg_neg_params'] = neg_neg_params
        if not pos_child_pos_parent_df.empty:
            X = pos_child_pos_parent_df[['s_init_child', 's_init_parent']].to_numpy().T
            y = pos_child_pos_parent_df['s_user_child'].to_numpy()
            pos_pos_params = fit_compound(function.function, X, y, remove_outlier_method)
            output['pos_pos_params'] = pos_pos_params
        if not neg_child_pos_parent_df.empty:
            X = neg_child_pos_parent_df[['s_init_child', 's_init_parent']].to_numpy().T
            y = neg_child_pos_parent_df['s_user_child'].to_numpy()
            neg_pos_params = fit_compound(function.function, X, y, remove_outlier_method)
            output['neg_pos_params'] = neg_pos_params
    else:
        X = belonging_model_df[['s_init_child', 's_init_parent']].to_numpy().T
        y = belonging_model_df['s_user_child'].to_numpy()
        output['params'] = fit_compound(function.function, X, y, remove_outlier_method)
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
    function.set_params(output.get('params', {}).get('params', None) if output.get('params') else None)
    return output, function


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
        'function': SentimentFormula(f"aggregate_{model_type}", "aggregate", func, result.x.tolist())
    }


def test_all_parameterizations():
    console = Console()

    def format_formula_string(func, params):
        source_lines = inspect.getsource(func).split('\n')
        param_names = list(inspect.signature(func).parameters.keys())[1:]
        formula_line = ""
        for line in source_lines:
            if 's_new =' in line:
                formula_line = line.strip()
                break
        if not formula_line: return "Could not parse formula."
        substitutions = {}
        if 'lambda' in param_names[0]:
            lambda_val = params[0]
            substitutions[param_names[0]] = f"{lambda_val:.4f}"
            substitutions[f"(1 - {param_names[0]})"] = f"{(1 - lambda_val):.4f}"
            for i, p_name in enumerate(param_names[1:], 1):
                substitutions[p_name] = f"{params[i]:.4f}"
        else:
            for p_name, p_val in zip(param_names, params):
                substitutions[p_name] = f"{p_val:.4f}"
        for old, new in substitutions.items():
            formula_line = re.sub(r'\b' + re.escape(old) + r'\b', new, formula_line)
        return f"{formula_line}\ns_final = np.tanh(s_new)"

    all_results, fitting_errors = [], []
    score_keys = ["user_sentiment_score", "user_normalized_sentiment_scores", "user_sentiment_score_mapped"]
    outlier_methods = ['none', 'lsquares', 'droptop']
    model_functions = {"Actor": [("actor_formula_v1", actor_formula_v1), ("actor_formula_v2", actor_formula_v2)], "Target": [("target_formula_v1", target_formula_v1), ("target_formula_v2", target_formula_v2)], "Association": [("assoc_formula_v1", assoc_formula_v1), ("assoc_formula_v2", assoc_formula_v2)], "Belonging Parent": [("belong_formula_v1", belong_formula_v1), ("belong_formula_v2", belong_formula_v2)], "Belonging Child": [("belong_formula_v1", belong_formula_v1), ("belong_formula_v2", belong_formula_v2)]}
    model_param_determiners = {"Actor": (determine_actor_parameters, ['none', 'driver', 'action_target']), "Target": (determine_target_parameters, ['none', 'driver', 'action_target']), "Association": (determine_association_parameters, ['none', 'other', 'entity_other']), "Belonging Parent": (determine_parent_parameters, ['none', 'parent_child', 'child']), "Belonging Child": (determine_child_parameters, ['none', 'parent_child', 'parent'])}
    aggregate_loss_functions = [("aggregate_error_mse", aggregate_error_mse), ("aggregate_error_softl1", aggregate_error_softl1), ("aggregate_error_dynamic_mse", aggregate_error_dynamic_mse), ("aggregate_error_dynamic_softl1", aggregate_error_dynamic_softl1), ("aggregate_error_logistic_mse", aggregate_error_logistic_mse), ("aggregate_error_logistic_softl1", aggregate_error_logistic_softl1)]

    with console.status("[bold yellow]Running comprehensive analysis...") as status:
        for score_key in score_keys:
            action_df, association_df, belonging_df, aggregate_df = create_action_df(score_key), create_association_df(score_key), create_belonging_df(score_key), create_aggregate_df(score_key)
            data_dfs = {"Actor": action_df, "Target": action_df, "Association": association_df, "Belonging Parent": belonging_df, "Belonging Child": belonging_df}
            for model_type, (determiner, splits) in model_param_determiners.items():
                status.update(f"Processing: {model_type} on {score_key}")
                for func_name, func in model_functions[model_type]:
                    for method in outlier_methods:
                        for split in splits:
                            with warnings.catch_warnings():
                                warnings.simplefilter("error", OptimizeWarning)
                                try:
                                    results_group, formula = determiner(data_dfs[model_type], SentimentFormula(func_name, model_type, func), method, split)
                                    valid_fits = {k: v for k, v in results_group.items() if v is not None}
                                    if not valid_fits:
                                        fitting_errors.append({'model_type': model_type, 'score_key': score_key, 'function': func_name, 'outlier_method': method, 'split': split, 'error': 'All sub-models failed to fit.'})
                                        continue
                                    avg_r2, avg_mse = np.mean([r['r2_loss'] for r in valid_fits.values()]), np.mean([r['mse'] for r in valid_fits.values()])
                                    all_results.append({'model_type': model_type, 'score_key': score_key, 'function': func_name, 'outlier_method': method, 'split_type': split, 'avg_r2': avg_r2, 'avg_mse': avg_mse, 'sub_models': valid_fits, 'formulas': {k: SentimentFormula(f"{func_name}_{k}", model_type, func, v['params']) for k, v in valid_fits.items()}})
                                except (ValueError, OptimizeWarning, RuntimeError) as e:
                                    fitting_errors.append({'model_type': model_type, 'score_key': score_key, 'function': func_name, 'outlier_method': method, 'split': split, 'error': str(e)})
            status.update(f"Processing: Aggregate on {score_key}")
            for loss_name, loss_func in aggregate_loss_functions:
                res = determine_aggregate_parameters(aggregate_df, loss_func)
                if res['success']: all_results.append({'model_type': 'Aggregate', 'score_key': score_key, 'loss_function': loss_name, **res})

    console.print("\n[bold green]✅ Comprehensive Analysis Complete.[/bold green]\n")

    for model_type in model_param_determiners.keys():
        results = sorted([r for r in all_results if r['model_type'] == model_type], key=lambda x: x.get('avg_mse', float('inf')))
        if not results: continue
        best_model = results[0]
        console.print(Panel(f"[bold white on blue] 🏆 Analysis for: {model_type} [/bold white on blue]", expand=False))
        best_model_table = Table(title="[bold]Best Performing Model (Lowest MSE)[/bold]", show_header=False, box=None, padding=(0, 2))
        best_model_table.add_column(style="cyan bold"), best_model_table.add_column(style="white")
        best_model_table.add_row("Score Key:", best_model['score_key']), best_model_table.add_row("Function:", best_model['function']), best_model_table.add_row("Outlier Method:", best_model['outlier_method']), best_model_table.add_row("Split Strategy:", best_model['split_type']), best_model_table.add_row("Avg. MSE:", f"[bold green]{best_model['avg_mse']:.4f}[/bold green]"), best_model_table.add_row("Avg. R² Score:", f"{best_model['avg_r2']:.4f}")
        param_table = Table(title="[bold]Parameters of Best Model[/bold]", show_header=True, header_style="bold magenta", box=None, padding=(0, 2))
        param_table.add_column("Context"), param_table.add_column("Parameters"), param_table.add_column("MSE")
        for context, details in best_model['sub_models'].items(): param_table.add_row(context.replace("_params", "").replace("_", " ").title(), str(np.round(details['params'], 4)), f"{details['mse']:.4f}")
        first_context, first_details = list(best_model['sub_models'].items())[0]
        func_obj = dict(model_functions[model_type])[best_model['function']]
        formula_str = format_formula_string(func_obj, first_details['params'])
        console.print(best_model_table), console.print(param_table), console.print(Panel(formula_str, title=f"[yellow]Formula Interpretation ({first_context.replace('_params', '').replace('_', ' ').title()})[/yellow]", border_style="yellow", padding=(1,2)))
        full_log_table = Table(title=f"[bold]Full Test Log for {model_type}[/bold]", show_header=True, header_style="bold white")
        full_log_table.add_column("Score Key", style="dim"), full_log_table.add_column("Function"), full_log_table.add_column("Method"), full_log_table.add_column("Split"), full_log_table.add_column("Avg MSE", style="green", justify="right")
        for res in results: full_log_table.add_row(res['score_key'], res['function'], res['outlier_method'], res['split_type'], f"{res['avg_mse']:.6f}")
        console.print(full_log_table)
        console.print("\n" + "="*80 + "\n")

    agg_results = [r for r in all_results if r['model_type'] == 'Aggregate']
    if agg_results:
        console.print(Panel("[bold white on blue] 🏆 Analysis for: Aggregate Models [/bold white on blue]", expand=False))
        best_agg_models = []
        for key in ['normal', 'dynamic', 'logistic']:
            models = [r for r in agg_results if (('dynamic' not in r['loss_function'] and 'logistic' not in r['loss_function']) if key == 'normal' else key in r['loss_function'])]
            if models: best_agg_models.append(sorted(models, key=lambda x: x['loss'])[0])
        table = Table(title="[bold]Top Aggregate Models by Formula Type (Lowest Loss)[/bold]", show_header=True, header_style="bold magenta")
        table.add_column("Formula Type"), table.add_column("Score Key"), table.add_column("Loss Function"), table.add_column("Final Loss", justify="right"), table.add_column("Parameters")
        for res in best_agg_models:
            formula_type = "Normal" if 'dynamic' not in res['loss_function'] and 'logistic' not in res['loss_function'] else ("Dynamic" if 'dynamic' in res['loss_function'] else "Logistic")
            table.add_row(formula_type, res['score_key'], res['loss_function'], f"{res['loss']:.4f}", str(np.round(res['params'], 4)))
        console.print(table)
        console.print("\n[bold cyan]📊 Generating interactive plots for best aggregate models...[/bold cyan]")
        for result in best_agg_models:
            score_key, loss_name, params = result['score_key'], result['loss_function'], result['params']
            df_agg = create_aggregate_df(score_key)
            formula_type = "Normal" if 'dynamic' not in loss_name and 'logistic' not in loss_name else ("Dynamic" if 'dynamic' in loss_name else "Logistic")
            df_agg['s_user_pred'] = df_agg.apply(lambda row: (aggregate_formula_dynamic(row['s_inits'], params) if formula_type == "Dynamic" else (aggregate_formula_logistic(row['s_inits'], params) if formula_type == "Logistic" else aggregate_formula(row['s_inits'], params))), axis=1)

            if formula_type == 'Normal':
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                fig.suptitle(f"Aggregate Model: {formula_type} | {score_key} | {loss_name}", fontsize=16)
                ax1.scatter(df_agg['s_user'], df_agg['s_user_pred'], alpha=0.6, edgecolors='k', linewidth=0.5, label="Data points")
                ax1.plot([-1, 1], [-1, 1], 'r--', label='Ideal y=x line'), ax1.set_title('Actual vs. Predicted Sentiment'), ax1.set_xlabel('Actual User Sentiment'), ax1.set_ylabel('Predicted Sentiment')
                ax1.grid(True, linestyle='--', alpha=0.6), ax1.legend(), ax1.set_aspect('equal', adjustable='box')
                ax2.set_title(r'Weight Distribution ($w_i$) vs. Item Position (i)'), ax2.set_xlabel('Item Position (i)'), ax2.set_ylabel('Weight')
                for n_val in [3, 5, 10]:
                    if n_val <= df_agg['N'].max(): ax2.plot(range(1, n_val + 1), calculate_weights(n_val, params[0], params[1]), '-o', label=f'N={n_val}')
                ax2.legend(), ax2.grid(True, linestyle='--', alpha=0.6)
            else:
                fig = plt.figure(figsize=(14, 10))
                gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
                ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
                ax3 = fig.add_subplot(gs[1, :])
                fig.suptitle(f"Aggregate Model: {formula_type} | {score_key} | {loss_name}", fontsize=16)
                plt.subplots_adjust(bottom=0.2, hspace=0.3)
                ax1.scatter(df_agg['s_user'], df_agg['s_user_pred'], alpha=0.6, edgecolors='k', linewidth=0.5, label="Data points")
                ax1.plot([-1, 1], [-1, 1], 'r--', label='Ideal y=x line'), ax1.set_title('Actual vs. Predicted Sentiment'), ax1.set_xlabel('Actual User Sentiment'), ax1.set_ylabel('Predicted Sentiment')
                ax1.grid(True, linestyle='--', alpha=0.6), ax1.legend(), ax1.set_aspect('equal', adjustable='box')
                max_n = df_agg['N'].max() if not df_agg.empty else 10
                ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor='lightgoldenrodyellow')
                n_slider = Slider(ax=ax_slider, label='N', valmin=2, valmax=max_n, valinit=int(max_n/2), valstep=1)

                def update(val):
                    n = int(n_slider.val)
                    ax2.clear(), ax3.clear()
                    n_range = np.arange(2, n + 1)
                    if formula_type == 'Dynamic':
                        alphas, betas = (params[0] * n_range + params[1]), (params[2] * n_range + params[3])
                    else:
                        alphas = logistic_function(n_range, params[0], params[1], params[2], params[3])
                        betas = logistic_function(n_range, params[4], params[5], params[6], params[7])
                    ax2.plot(n_range, alphas, 'b-o', label=r'$\alpha_N$ (Primacy)'), ax2.plot(n_range, betas, 'g-s', label=r'$\beta_N$ (Recency)')
                    ax2.set_title(f'Parameters vs. N (up to N={n})'), ax2.set_xlabel('Number of Items (N)'), ax2.set_ylabel('Parameter Value'), ax2.legend(), ax2.grid(True, linestyle='--', alpha=0.6)

                    current_alpha, current_beta = alphas[-1], betas[-1]
                    weights = calculate_weights(n, current_alpha, current_beta)
                    markerline, stemlines, baseline = ax3.stem(range(1, n + 1), weights, label=f'N={n}', basefmt=" ")
                    plt.setp(markerline, 'color', 'red'), plt.setp(stemlines, 'color', 'grey', 'linestyle', ':')
                    ax3.plot(range(1, n + 1), weights, 'r-')
                    ax3.set_title(fr'Weight Distribution for N={n} ($\alpha={current_alpha:.2f}, \beta={current_beta:.2f}$)'), ax3.set_xlabel('Item Position (i)'), ax3.set_ylabel('Weight'), ax3.set_ylim(bottom=0), ax3.grid(True, linestyle='--', alpha=0.6)
                    fig.canvas.draw_idle()

                update(int(max_n/2))
                n_slider.on_changed(update)
            plt.show()

    if fitting_errors:
        console.print("\n\n")
        error_table = Table(title="[bold red]⚠️ Fitting Errors and Warnings Log[/bold red]", show_header=True, header_style="bold yellow", expand=True)
        error_table.add_column("Model Type"), error_table.add_column("Score Key"), error_table.add_column("Function"), error_table.add_column("Method"), error_table.add_column("Split"), error_table.add_column("Error Message", width=50)
        for err in fitting_errors:
            error_table.add_row(err['model_type'], err['score_key'], err['function'], err['outlier_method'], err['split'], err['error'])
        console.print(error_table)
    
    print("\n[yellow]Saving results to 'survey_question_optimizer_results.txt'...[/yellow]")
    with open('survey_question_optimizer_results.txt', 'w') as f:
        for res in all_results:
            f.write(f"Model Type: {res['model_type']}, Score Key: {res['score_key']}, Function: {res.get('function', 'N/A')}, Outlier Method: {res.get('outlier_method', 'N/A')}, Split Type: {res.get('split_type', 'N/A')}, Avg MSE: {res.get('avg_mse', 'N/A')}, Avg R²: {res.get('avg_r2', 'N/A')}\n")
            if 'sub_models' in res:
                for context, details in res['sub_models'].items():
                    f.write(f"  Context: {context}, Params: {details['params']}, MSE: {details['mse']}\n")
            if 'params' in res:
                f.write(f"  Parameters: {res['params']}\n")
        f.write("\nFitting Errors:\n")
        for err in fitting_errors:
            f.write(f"  Model Type: {err['model_type']}, Score Key: {err['score_key']}, Function: {err['function']}, Outlier Method: {err['outlier_method']}, Split: {err['split']}, Error Message: {err['error']}\n")

    os.makedirs('src/survey/optimal_formulas', exist_ok=True)
    
    all_params = {}
    for model_type in model_param_determiners.keys():
        results = sorted([r for r in all_results if r['model_type'] == model_type], key=lambda x: x.get('avg_mse', float('inf')))
        if results:
            best_model = results[0]
            all_params[model_type] = best_model
            if 'formulas' in best_model:
                os.makedirs(f'src/survey/optimal_formulas/{model_type}', exist_ok=True)
                for context, formula in best_model['formulas'].items():
                    formula.save(f"src/survey/optimal_formulas/{model_type}/{context}_{best_model['score_key']}_{best_model['function']}.json")

    agg_results = [r for r in all_results if r['model_type'] == 'Aggregate']
    if agg_results:
        best_agg_models = []
        for key in ['normal', 'dynamic', 'logistic']:
            models = [r for r in agg_results if (('dynamic' not in r['loss_function'] and 'logistic' not in r['loss_function']) if key == 'normal' else key in r['loss_function'])]
            if models: 
                best = sorted(models, key=lambda x: x['loss'])[0]
                best_agg_models.append(best)
        best_agg_models = sorted(best_agg_models, key=lambda x: x['loss'])
        all_params['Aggregate'] = best_agg_models[0]
        if 'formulas' in best_agg_models[0]:
            os.makedirs('src/survey/optimal_formulas/Aggregate', exist_ok=True)
            for context, formula in best_agg_models[0]['formulas'].items():
                formula.save(f"src/survey/optimal_formulas/Aggregate/{context}_{best_agg_models[0]['score_key']}_{best_agg_models[0]['loss_function']}.json")
        all_params['Aggregate']['function_save_location'] = f"src/survey/optimal_formulas/Aggregate/{best_agg_models[0]['score_key']}_{best_agg_models[0]['loss_function']}.json"
        table = Table(title="[bold]Top Aggregate Models by Formula Type (Lowest Loss)[/bold]", show_header=True, header_style="bold magenta")
        table.add_column("Formula Type"), table.add_column("Score Key"), table.add_column("Loss Function"), table.add_column("Final Loss", justify="right"), table.add_column("Parameters")
        for res in best_agg_models:
            formula_type = "Normal" if 'dynamic' not in res['loss_function'] and 'logistic' not in res['loss_function'] else ("Dynamic" if 'dynamic' in res['loss_function'] else "Logistic")
            table.add_row(formula_type, res['score_key'], res['loss_function'], f"{res['loss']:.4f}", str(np.round(res['params'], 4)))
        console.print(table)
    
    with open('src/survey/optimal_formulas/all_optimal_parameters.json', 'w') as f:
        json.dump(all_params, f, indent=2, default=str)

def load_optimal_models():
    try:
        with open('src/survey/optimal_formulas/all_optimal_parameters.json', 'r') as f:
            all_params = json.load(f)
    except FileNotFoundError:
        print("No optimal parameters found. Please run the analysis first.")
        return None, None, None, None

    def load_formula_from_path(path):
        if os.path.exists(path):
            try:
                formula = SentimentFormula.load_from_file(path)
                return formula
            except:
                pass
            try:
                with open(path, 'r') as f:
                    data = json.load(f)

                def get_function_by_name(name, namespace=None):
                    if namespace is None:
                        frame = inspect.currentframe().f_back
                        namespace = {}
                        namespace.update(frame.f_globals)
                        namespace.update(frame.f_locals)
                    return namespace.get(name)
                
                name = data.get('name', '')
                function = get_function_by_name(name)
                if function is None:
                    if "actor" in name:
                        function = actor_formula_v1 if "v1" in name else actor_formula_v2
                    elif "target" in name:
                        function = target_formula_v1 if "v1" in name else target_formula_v2
                    elif "assoc" in name:
                        function = assoc_formula_v1 if "v1" in name else assoc_formula_v2
                    elif "belong" in name:
                        function = belong_formula_v1 if "v1" in name else belong_formula_v2
                    else:
                        function = None
                
                params = data.get('params', [])
                if isinstance(params, list) and len(params) > 0:
                    params = np.array(params)
                
                return SentimentFormula(
                    name=data.get('name', 'fallback_formula'),
                    model_type=data.get('model_type', 'unknown'),
                    function=function,
                    params=params
                )
            except:
                pass
        return None

    actor_params = all_params.get('Actor', {})
    actor_split = actor_params.get('split_type', 'none')
    actor_formulas = actor_params.get('formulas', {})
    def actor_func(s_init_actor, driver, split_context=None):
        from rich import print as print
        ctx = split_context
        if actor_split == 'driver':
            ctx = 'pos_driver_params' if driver > 0 else 'neg_driver_params'
        elif actor_split == 'action_target':
            if driver > 0:
                ctx = 'pos_pos_params' if s_init_actor > 0 else 'neg_pos_params'
            else:
                ctx = 'pos_neg_params' if s_init_actor > 0 else 'neg_neg_params'
        if actor_split != 'none' and ctx in actor_formulas:
            try:
                formula = load_formula_from_path(f"src/survey/optimal_formulas/Actor/{ctx}_{actor_params['score_key']}_{actor_params['function']}.json")
                if formula is None:
                    if isinstance(actor_formulas.get(ctx), str):
                        formula = None
                    else:
                        formula = actor_formulas[ctx]
            except (AttributeError, KeyError):
                if isinstance(actor_formulas.get(ctx), str):
                    formula = None
                elif ctx in actor_formulas:
                    formula = actor_formulas[ctx]
                else:
                    formula = None
            print(f"[bold cyan]Actor split:[/] {actor_split} | context: {ctx} | s_init_actor: {s_init_actor} | driver: {driver}")
            if formula:
                return formula([s_init_actor, driver])
        else:
            try:
                formula = load_formula_from_path(f"src/survey/optimal_formulas/Actor/default_{actor_params['score_key']}_{actor_params['function']}.json")
                if formula is None and 'default' in actor_formulas:
                    if isinstance(actor_formulas.get('default'), str):
                        formula = None
                    else:
                        formula = actor_formulas['default']
                elif not actor_formulas:
                    raise RuntimeError("No actor formula found.")
                else:
                    for key, val in actor_formulas.items():
                        if not isinstance(val, str):
                            formula = val
                            break
                    else:
                        raise RuntimeError("No valid actor formula found.")
            except (AttributeError, KeyError):
                if actor_formulas:
                    for key, val in actor_formulas.items():
                        if not isinstance(val, str):
                            formula = val
                            break
                    else:
                        raise RuntimeError("No valid actor formula found.")
                else:
                    raise RuntimeError("No actor formula found.")
            print(f"[bold cyan]Actor default used | s_init_actor: {s_init_actor} | driver: {driver}")
            if formula:
                return formula([s_init_actor, driver])

    target_params = all_params.get('Target', {})
    target_split = target_params.get('split_type', 'none')
    target_formulas = target_params.get('formulas', {})
    def target_func(s_init_target, s_action, split_context=None):
        from rich import print as print
        ctx = split_context
        if target_split == 'driver':
            ctx = 'pos_driver_params' if s_action > 0 else 'neg_driver_params'
        elif target_split == 'action_target':
            if s_action > 0:
                ctx = 'pos_pos_params' if s_init_target > 0 else 'neg_pos_params'
            else:
                ctx = 'pos_neg_params' if s_init_target > 0 else 'neg_neg_params'
        if target_split != 'none' and ctx in target_formulas:
            try:
                formula = load_formula_from_path(f"src/survey/optimal_formulas/Target/{ctx}_{target_params['score_key']}_{target_params['function']}.json")
                if formula is None:
                    if isinstance(target_formulas.get(ctx), str):
                        formula = None
                    else:
                        formula = target_formulas[ctx]
            except (AttributeError, KeyError):
                if isinstance(target_formulas.get(ctx), str):
                    formula = None
                elif ctx in target_formulas:
                    formula = target_formulas[ctx]
                else:
                    formula = None
            print(f"[bold cyan]Target split:[/] {target_split} | context: {ctx} | s_init_target: {s_init_target} | s_action: {s_action}")
            if formula:
                return formula([s_init_target, s_action])
        else:
            try:
                formula = load_formula_from_path(f"src/survey/optimal_formulas/Target/default_{target_params['score_key']}_{target_params['function']}.json")
                if formula is None and 'default' in target_formulas:
                    if isinstance(target_formulas.get('default'), str):
                        formula = None
                    else:
                        formula = target_formulas['default']
                elif not target_formulas:
                    raise RuntimeError("No target formula found.")
                else:
                    for key, val in target_formulas.items():
                        if not isinstance(val, str):
                            formula = val
                            break
                    else:
                        raise RuntimeError("No valid target formula found.")
            except (AttributeError, KeyError):
                if target_formulas:
                    for key, val in target_formulas.items():
                        if not isinstance(val, str):
                            formula = val
                            break
                    else:
                        raise RuntimeError("No valid target formula found.")
                else:
                    raise RuntimeError("No target formula found.")
            print(f"[bold cyan]Target default used | s_init_target: {s_init_target} | s_action: {s_action}")
            if formula:
                return formula([s_init_target, s_action])

    assoc_params = all_params.get('Association', {})
    assoc_split = assoc_params.get('split_type', 'none')
    assoc_formulas = assoc_params.get('formulas', {})
    def association_func(s_init_entity, s_init_other, split_context=None):
        ctx = split_context
        if assoc_split == 'other':
            ctx = 'pos_params' if s_init_other > 0 else 'neg_params'
        elif assoc_split == 'entity_other':
            if s_init_entity > 0 and s_init_other > 0:
                ctx = 'pos_params'
            elif s_init_entity > 0 and s_init_other <= 0:
                ctx = 'pos_neg_params'
            elif s_init_entity <= 0 and s_init_other > 0:
                ctx = 'neg_pos_params'
            else:
                ctx = 'neg_neg_params'
        if assoc_split != 'none' and ctx in assoc_formulas:
            try:
                formula = load_formula_from_path(f"src/survey/optimal_formulas/Association/{ctx}_{assoc_params['score_key']}_{assoc_params['function']}.json")
                if formula is None:
                    if isinstance(assoc_formulas.get(ctx), str):
                        formula = None
                    else:
                        formula = assoc_formulas[ctx]
            except (AttributeError, pickle.PickleError):
                if isinstance(assoc_formulas.get(ctx), str):
                    formula = None
                elif ctx in assoc_formulas:
                    formula = assoc_formulas[ctx]
                else:
                    formula = None
            print(f"[bold cyan]Association split:[/] {assoc_split} | context: {ctx} | s_init_entity: {s_init_entity} | s_init_other: {s_init_other}")
            if formula:
                return formula([s_init_entity, s_init_other])
        else:
            try:
                formula = load_formula_from_path(f"src/survey/optimal_formulas/Association/default_{assoc_params['score_key']}_{assoc_params['function']}.json")
                if formula is None and 'default' in assoc_formulas:
                    if isinstance(assoc_formulas.get('default'), str):
                        formula = None
                    else:
                        formula = assoc_formulas['default']
                elif not assoc_formulas:
                    raise RuntimeError("No association formula found.")
                else:
                    for key, val in assoc_formulas.items():
                        if not isinstance(val, str):
                            formula = val
                            break
                    else:
                        raise RuntimeError("No valid association formula found.")
            except (AttributeError, pickle.PickleError):
                if assoc_formulas:
                    for key, val in assoc_formulas.items():
                        if not isinstance(val, str):
                            formula = val
                            break
                    else:
                        raise RuntimeError("No valid association formula found.")
                else:
                    raise RuntimeError("No association formula found.")
            print(f"[bold cyan]Association default used | s_init_entity: {s_init_entity} | s_init_other: {s_init_other}")
            if formula:
                return formula([s_init_entity, s_init_other])

    agg_params = all_params.get('Aggregate', {})
    agg_formula_path = agg_params.get('function_save_location', None)
    agg_formula = None
    if agg_formula_path:
        agg_formula = load_formula_from_path(agg_formula_path)
    def aggregate_func(s_inits, split_context=None):
        if agg_formula is not None:
            return agg_formula(s_inits)
        import numpy as np
        return np.mean(s_inits) if s_inits else 0.0

    parent_params = all_params.get('Belonging Parent', {})
    parent_split = parent_params.get('split_type', 'none')
    parent_formulas = parent_params.get('formulas', {})
    
    child_params = all_params.get('Belonging Child', {})
    child_split = child_params.get('split_type', 'none')
    child_formulas = child_params.get('formulas', {})
    
    def belonging_func(parent_sentiment_init, child_sentiment_init, split_context=None):
        ctx = split_context
        
        parent_ctx = None
        if parent_split == 'parent_child':
            if parent_sentiment_init > 0 and child_sentiment_init > 0:
                parent_ctx = 'pos_pos_params'
            elif parent_sentiment_init > 0 and child_sentiment_init <= 0:
                parent_ctx = 'pos_neg_params'
            elif parent_sentiment_init <= 0 and child_sentiment_init > 0:
                parent_ctx = 'neg_pos_params'
            else:
                parent_ctx = 'neg_neg_params'
        elif parent_split == 'child':
            parent_ctx = 'pos_child_params' if child_sentiment_init > 0 else 'neg_child_params'
        
        child_ctx = None
        if child_split == 'parent_child':
            if parent_sentiment_init > 0 and child_sentiment_init > 0:
                child_ctx = 'pos_pos_params'
            elif parent_sentiment_init > 0 and child_sentiment_init <= 0:
                child_ctx = 'pos_neg_params'
            elif parent_sentiment_init <= 0 and child_sentiment_init > 0:
                child_ctx = 'neg_pos_params'
            else:
                child_ctx = 'neg_neg_params'
        elif child_split == 'parent':
            child_ctx = 'pos_parent_params' if parent_sentiment_init > 0 else 'neg_parent_params'
        
        parent_formula = None
        if parent_split != 'none' and parent_ctx in parent_formulas:
            try:
                parent_formula = load_formula_from_path(f"src/survey/optimal_formulas/Belonging Parent/{parent_ctx}_{parent_params['score_key']}_{parent_params['function']}.json")
                if parent_formula is None:
                    if not isinstance(parent_formulas.get(parent_ctx), str):
                        parent_formula = parent_formulas[parent_ctx]
            except (AttributeError, pickle.PickleError):
                if not isinstance(parent_formulas.get(parent_ctx), str):
                    parent_formula = parent_formulas[parent_ctx]
        else:
            try:
                parent_formula = load_formula_from_path(f"src/survey/optimal_formulas/Belonging Parent/default_{parent_params['score_key']}_{parent_params['function']}.json")
                if parent_formula is None:
                    for key, val in parent_formulas.items():
                        if not isinstance(val, str):
                            parent_formula = val
                            break
            except (AttributeError, pickle.PickleError):
                for key, val in parent_formulas.items():
                    if not isinstance(val, str):
                        parent_formula = val
                        break
        
        child_formula = None
        if child_split != 'none' and child_ctx in child_formulas:
            try:
                child_formula = load_formula_from_path(f"src/survey/optimal_formulas/Belonging Child/{child_ctx}_{child_params['score_key']}_{child_params['function']}.json")
                if child_formula is None:
                    if not isinstance(child_formulas.get(child_ctx), str):
                        child_formula = child_formulas[child_ctx]
            except (AttributeError, pickle.PickleError):
                if not isinstance(child_formulas.get(child_ctx), str):
                    child_formula = child_formulas[child_ctx]
        else:
            try:
                child_formula = load_formula_from_path(f"src/survey/optimal_formulas/Belonging Child/default_{child_params['score_key']}_{child_params['function']}.json")
                if child_formula is None:
                    for key, val in child_formulas.items():
                        if not isinstance(val, str):
                            child_formula = val
                            break
            except (AttributeError, pickle.PickleError):
                for key, val in child_formulas.items():
                    if not isinstance(val, str):
                        child_formula = val
                        break
        
        parent_compound = parent_formula([parent_sentiment_init, child_sentiment_init]) if parent_formula else parent_sentiment_init
        child_compound = child_formula([parent_sentiment_init, child_sentiment_init]) if child_formula else child_sentiment_init
        
        print(f"[bold magenta]Belonging[/] parent: {parent_sentiment_init:.4f} -> {parent_compound:.4f} | child: {child_sentiment_init:.4f} -> {child_compound:.4f}")
        return parent_compound, child_compound

    return actor_func, target_func, association_func, aggregate_func, belonging_func

