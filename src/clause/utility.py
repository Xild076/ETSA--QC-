from imodels import RuleFitClassifier
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Literal, Any, Union

def create_model(n_estimators:int=25, tree_size:int=3) -> OneVsRestClassifier:
    """Creates a multiclass RuleFitClassifier model with basic specified parameters.
    Args:
        n_estimators (int): Number of estimators in the ensemble.
        tree_size (int): Maximum size of the trees.
    Returns:
        model (OneVsRestClassifier): A OneVsRestClassifier model with RuleFitClassifier as the base estimator.
    """

    model = OneVsRestClassifier(RuleFitClassifier(
        n_estimators=n_estimators,
        tree_size=tree_size
    ))
    return model

def prepare_data(X: dict, y: list) -> tuple:
    """Preprocesses input data for training.
    Args:
        X (dict): A dictionary containing the input features.
        y (list): A list of labels.
    Returns:
        tuple: A tuple containing the preprocessed features (X) and labels (y).
    """

    X = pd.DataFrame(X)

    l_e = LabelEncoder()
    y_enc = l_e.fit_transform(y)

    return X, y_enc

def train_model(X, y):
    """Basic training function for the model.
    Args:
        X (pd.DataFrame): A DataFrame containing the input features.
        y (list): A list of labels.
    Returns:
        model (OneVsRestClassifier): A trained OneVsRestClassifier model.
    """

    model = create_model()
    model.fit(X, y)
    return model

def predict(model:OneVsRestClassifier, X:Union[pd.DataFrame, dict], single_input=True) -> list:
    """Predicts the labels for the input data using the trained model.
    Args:
        model (OneVsRestClassifier): The trained model.
        X (pd.DataFrame, dict): A DataFrame containing the input features.
    Returns:
        list: A list of predicted labels.
    """
    
    if isinstance(X, dict):
        X = pd.DataFrame(X)
    elif not isinstance(X, pd.DataFrame):
        raise ValueError("Input data must be a DataFrame or a dictionary.")

    y_pred = model.predict(X)
    if single_input:
        return y_pred[0]
    return y_pred