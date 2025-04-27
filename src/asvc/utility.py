from sklearn.model_selection import train_test_split
import numpy as np
from skrules import SkopeRules
import ast
import pandas as pd
from packaging import version
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
import sklearn

if version.parse(sklearn.__version__) >= version.parse("1.4.0"):
    # Patch BaggingClassifier and BaggingRegressor to accept base_estimator as a parameter
    class _PatchedBaggingClassifier(BaggingClassifier):
        def __init__(self, base_estimator=None, **kw):
            super().__init__(estimator=base_estimator, **kw)
            self.base_estimator = base_estimator
    class _PatchedBaggingRegressor(BaggingRegressor):
        def __init__(self, base_estimator=None, **kw):
            super().__init__(estimator=base_estimator, **kw)
            self.base_estimator = base_estimator
    import skrules.skope_rules as _sr_mod
    _sr_mod.BaggingClassifier = _PatchedBaggingClassifier
    _sr_mod.BaggingRegressor = _PatchedBaggingRegressor

def _binary_train(x_df, y_bin, feature_names):
    """
    Train a SkopeRules model for binary classification.
    Args:
        x_df (pd.DataFrame): A DataFrame containing the extracted features.
        y_bin (np.array): A numpy array of binary labels.
        feature_names (list): List of feature names.
    Outputs:
        model (SkopeRules): The trained SkopeRules model.
    """
    tr_x, tr_y = x_df.values, y_bin
    
    if y_bin.sum() >= 2 and (len(y_bin) - y_bin.sum()) >= 2:
         min_samples_per_class = np.min(np.bincount(y_bin))
         if min_samples_per_class >= 2:
             tr_x, _, tr_y, _ = train_test_split(
                 x_df.values, y_bin, stratify=y_bin, test_size=0.2, random_state=42
             )

    model = SkopeRules(
        feature_names=feature_names,
        n_estimators=30,
        max_samples=0.7,
        precision_min=0.9,
        recall_min=0.05,
        max_depth=6,
        max_depth_duplication=4,
        random_state=42,
    )

    model.fit(tr_x, tr_y)
    return model

def load_dataset(file_path='data/dataset.csv'):
    """
    Load the dataset from a CSV file and convert it to a dictionary.
    Args:
        file_path (str): The path to the CSV file.
    Outputs:
        dataset (dict): A dictionary containing the dataset.
    """
    df = pd.read_csv(file_path)
    df["actor"] = df["actor"].apply(ast.literal_eval)
    df["action"] = df["action"].apply(ast.literal_eval)
    df["victim"] = df["victim"].apply(ast.literal_eval)
    df["actor_subject"] = df["actor_subject"].fillna(-1).astype(int)
    return list(df.to_dict(orient="index").values())
