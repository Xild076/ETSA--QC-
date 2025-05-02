from imodels import RuleFitClassifier
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, Union
from enum import Enum
import spacy
from spacy.tokens.doc import Doc
from spacy.tokens import Token
from sklearn.metrics import classification_report
import ast
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

def create_model(n_estimators:int=25, tree_size:int=3) -> OneVsRestClassifier:
    """Creates a multiclass RuleFitClassifier model with basic specified parameters.
    Args:
        n_estimators (int): Number of estimators in the ensemble.
        tree_size (int): Maximum size of the trees.
    Returns:
        OneVsRestClassifier: A OneVsRestClassifier model with RuleFitClassifier as the base estimator.
    """
    logger.info(f"Creating OneVsRestClassifier model with RuleFitClassifier architecture (n_estimators={n_estimators}, tree_size={tree_size})")

    model = OneVsRestClassifier(RuleFitClassifier(
        n_estimators=n_estimators,
        tree_size=tree_size
    ))
    return model

def prepare_data(X:list, y:list) -> tuple:
    """Preprocesses input data for training.
    Args:
        X (list): A list containing dicts of input features.
        y (list): A list of labels.
    Returns:
        tuple: A tuple containing the preprocessed features (X) and labels (y).
    """
    logger.info("Preparing data for training...")

    X = pd.DataFrame(X)

    l_e = LabelEncoder()
    y_enc = l_e.fit_transform(y)

    return X, y_enc

def train_model(X:pd.DataFrame, y:list) -> OneVsRestClassifier:
    """Basic training function for the model.
    Args:
        X (pd.DataFrame): A DataFrame containing the input features.
        y (list): A list of labels.
    Returns:
        OneVsRestClassifier: A trained OneVsRestClassifier model.
    """
    logger.info("Training model...")

    model = create_model()
    model.fit(X, y)
    return model

def evaluate_model(model:OneVsRestClassifier, X:pd.DataFrame, y:list):
    """Evaluate models with sklearn.metrics's classification report.
    Args:
        model (OneVsRestClassifier): A trained OneVsRestClassifier model.
        X (pd.DataFrame): A DataFrame containing input features.
        y (list): A list of real labels.
    Returns:
        str | dict: A classification report.
    """
    logger.info("Evaluating model...")

    y_pred = model.predict(X)
    return classification_report(y, y_pred)

def predict(model:OneVsRestClassifier, X:Union[pd.DataFrame, dict], single_input=True) -> list:
    """Predicts the labels for the input data using the trained model.
    Args:
        model (OneVsRestClassifier): The trained model.
        X (pd.DataFrame, dict): A DataFrame containing the input features.
    Returns:
        list: A list of predicted labels.
    """
    logger.info(f"Predicting labels (single_input={single_input})...")
    
    if isinstance(X, dict):
        X = pd.DataFrame(X)
    elif not isinstance(X, pd.DataFrame):
        raise ValueError("Input data must be a DataFrame or a dictionary.")

    y_pred = model.predict(X)
    if single_input:
        if len(y_pred) != 1:
            raise ValueError("Expected single input row but got multiple.")
        return y_pred[0]
    return y_pred

def load_dataset(file_path='data/dataset.csv'):
    """
    Load the dataset from a CSV file and convert it to a dictionary.
    Args:
        file_path (str): The path to the CSV file.
    Returns:
        dict: A dictionary containing the dataset.
    """
    logger.info(f"Loading dataset from {file_path}...") 

    df = pd.read_csv(file_path)
    df["actor"] = df["actor"].apply(ast.literal_eval)
    df["action"] = df["action"].apply(ast.literal_eval)
    df["victim"] = df["victim"].apply(ast.literal_eval)
    df["actor_subject"] = df["actor_subject"].fillna(-1).astype(int)
    return list(df.to_dict(orient="index").values())

class DocFeatureType(Enum):
    """Enum for document-wide feature types."""
    NUM_TOKENS = "num_tokens"
    NUM_WORDS = "num_words"
    NUM_SENTENCES = "num_sentences"
    AVERAGE_TOKEN_LENGTH = "average_token_length"
    AVERAGE_WORD_LENGTH = "average_word_length"
    AVERAGE_SENTENCE_LENGTH = "average_sentence_length"
    NUM_ALPHA = "num_alpha"
    NUM_NSUBJ = "num_nsubj"
    NUM_NSUBJPASS = "num_nsubjpass"
    NUM_DOBJ = "num_dobj"
    NUM_POBJ = "num_pobj"
    NUM_VERBS = "num_verbs"
    NUM_ADJ = "num_adj"
    NUM_ADV = "num_adv"
    HAS_PASSIVE = "has_passive"
    ROOT_IS_VERB = "root_is_verb"
    ROOT_IS_NOUN = "root_is_noun"

class TokenFeatureType(Enum):
    """Enum for token-specific feature types."""
    DEP = "dep"
    POS = "pos"
    INDEX = "index"
    NUM_ANCESTORS = "num_ancestors"
    NUM_DESCENDANTS = "num_descendants"
    HEAD_DEP = "head_dep"
    HEAD_POS = "head_pos"
    HEAD_INDEX = "head_index"
    IS_FIRST = "is_first"
    IS_LAST = "is_last"
    IS_ALPHA = "is_alpha"
    ANCESTOR_CONTAINS_DEP = "ancestor_contains_dep"
    ANCESTOR_CONTAINS_POS = "ancestor_contains_pos"
    CHILDREN_CONTAINS_DEP = "children_contains_dep"
    CHILDREN_CONTAINS_POS = "children_contains_pos"

def extract_features_from_doc(text:Union[str, Doc], features:list) -> Dict[str, Any]:
    """Extracts document-wide features from the input text.
    Args:
        text (str): The input text.
        features (list): A list of feature types to extract.
    Returns:
        dict: A dictionary containing the extracted features.
    """
    logger.info(f"Extracting features: {features} from document...")

    if isinstance(text, str):
        doc = nlp(text)
    extracted_features = {}
    for feature in features:
        if feature == DocFeatureType.NUM_TOKENS:
            extracted_features[feature.value] = len(doc)
        elif feature == DocFeatureType.NUM_WORDS:
            extracted_features[feature.value] = len([token for token in doc if not token.is_punct])
        elif feature == DocFeatureType.NUM_SENTENCES:
            extracted_features[feature.value] = len(list(doc.sents))
        elif feature == DocFeatureType.AVERAGE_TOKEN_LENGTH:
            extracted_features[feature.value] = sum(len(token) for token in doc) / len(doc)
        elif feature == DocFeatureType.AVERAGE_WORD_LENGTH:
            extracted_features[feature.value] = sum(len(token) for token in doc if not token.is_punct) / len([token for token in doc if not token.is_punct])
        elif feature == DocFeatureType.AVERAGE_SENTENCE_LENGTH:
            extracted_features[feature.value] = sum(len(sentence) for sentence in doc.sents) / len(list(doc.sents))
        elif feature == DocFeatureType.NUM_ALPHA:
            extracted_features[feature.value] = len([token for token in doc if token.is_alpha])
        elif feature == DocFeatureType.NUM_NSUBJ:
            extracted_features[feature.value] = len([token for token in doc if token.dep_ == "nsubj"])
        elif feature == DocFeatureType.NUM_NSUBJPASS:
            extracted_features[feature.value] = len([token for token in doc if token.dep_ == "nsubjpass"])
        elif feature == DocFeatureType.NUM_DOBJ:
            extracted_features[feature.value] = len([token for token in doc if token.dep_ == "dobj"])
        elif feature == DocFeatureType.NUM_POBJ:
            extracted_features[feature.value] = len([token for token in doc if token.dep_ == "pobj"])
        elif feature == DocFeatureType.NUM_VERBS:
            extracted_features[feature.value] = len([token for token in doc if token.pos_ == "VERB"])
        elif feature == DocFeatureType.NUM_ADJ:
            extracted_features[feature.value] = len([token for token in doc if token.pos_ == "ADJ"])
        elif feature == DocFeatureType.NUM_ADV:
            extracted_features[feature.value] = len([token for token in doc if token.pos_ == "ADV"])
        elif feature == DocFeatureType.HAS_PASSIVE:
            extracted_features[feature.value] = int(any(token.dep_ == "nsubjpass" for token in doc))
        elif feature == DocFeatureType.ROOT_IS_VERB:
            extracted_features[feature.value] = int(any(token.dep_ == "ROOT" and token.pos_ == "VERB" for token in doc))
        elif feature == DocFeatureType.ROOT_IS_NOUN:
            extracted_features[feature.value] = int(any(token.dep_ == "ROOT" and token.pos_ == "NOUN" for token in doc))
        else:
            print(f"Feature {feature} not recognized.")
    return extracted_features

def extract_features_from_token(token:Token, features:list, acd=[], acp=[], ccd=[], ccp=[]) -> Dict[str, Any]:
    """Extracts token-specific features from the input token.
    Args:
        token: The input token.
        features (list): A list of feature types to extract.
        acd (list): Ancestor contains dep.
        acp (list): Ancestor contains pos.
        ccd (list): Children contains dep.
        ccp (list): Children contains pos.
    Returns:
        dict: A dictionary containing the extracted features.
    """
    logger.info(f"Extracting features: {features} from sentence...")

    extracted_features = {}
    for feature in features:
        if feature == TokenFeatureType.DEP:
            extracted_features[feature.value] = token.dep_
        elif feature == TokenFeatureType.POS:
            extracted_features[feature.value] = token.pos_
        elif feature == TokenFeatureType.INDEX:
            extracted_features[feature.value] = token.i
        elif feature == TokenFeatureType.NUM_ANCESTORS:
            extracted_features[feature.value] = len(list(token.ancestors))
        elif feature == TokenFeatureType.NUM_DESCENDANTS:
            extracted_features[feature.value] = len(list(token.children))
        elif feature == TokenFeatureType.HEAD_DEP:
            extracted_features[feature.value] = token.head.dep_
        elif feature == TokenFeatureType.HEAD_POS:
            extracted_features[feature.value] = token.head.pos_
        elif feature == TokenFeatureType.HEAD_INDEX:
            extracted_features[feature.value] = token.head.i
        elif feature == TokenFeatureType.IS_FIRST:
            extracted_features[feature.value] = token.i == 0
        elif feature == TokenFeatureType.IS_LAST:
            extracted_features[feature.value] = token.i == len(token.doc) - 1
        elif feature == TokenFeatureType.IS_ALPHA:
            extracted_features[feature.value] = token.is_alpha
        elif feature == TokenFeatureType.ANCESTOR_CONTAINS_DEP:
            extracted_features[feature.value] = 0
            for ancestor in token.ancestors:
                if ancestor.dep_ in acd:
                    extracted_features[feature.value] = 1
                    break
        elif feature == TokenFeatureType.ANCESTOR_CONTAINS_POS:
            extracted_features[feature.value] = 0
            for ancestor in token.ancestors:
                if ancestor.pos_ in acp:
                    extracted_features[feature.value] = 1
                    break
        elif feature == TokenFeatureType.CHILDREN_CONTAINS_DEP:
            extracted_features[feature.value] = 0
            for child in token.children:
                if child.dep_ in ccd:
                    extracted_features[feature.value] = 1
                    break
        elif feature == TokenFeatureType.CHILDREN_CONTAINS_POS:
            extracted_features[feature.value] = 0
            for child in token.children:
                if child.pos_ in ccp:
                    extracted_features[feature.value] = 1
                    break
        else:
            print(f"Token Feature {feature} not recognized.")
    return