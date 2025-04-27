import spacy
import pandas as pd
import os
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(filename="logs/QCLog.log", level=logging.INFO)

nlp = spacy.load("en_core_web_sm")

def create_dataset_file(file_name='data/dataset.csv'):
    """
    Create a dataset file if it does not exist.
    Args:
        file_name (str): The name of the dataset file to create.
    Outputs:
        None
    """
    if os.path.exists(file_name):
        logging.info("Dataset file already exists. Skipping dataset creation.")
        return
    else:
        with open(file_name, "w") as f:
            f.write("text,text_type,actor,actor_subject,action,victim,extra\n")
            logging.info("Dataset file created.")

def split_text(text:str):
    """
    Split the text into sentences using spaCy.
    Args:
        text (str): The text to split.
    Outputs:
        sentences (list): A list of sentences.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

def load_text_file_names(directory:str):
    """
    Load text file names from a directory.
    Args:
        directory (str): The directory to load text files from.
    Outputs:
        text_files (list): A list of text file names.
    """
    text_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            text_files.append(os.path.join(directory, filename))
    
    logger.info(f"Text files loaded: {text_files}")
    return text_files

def load_text_file(file_path:str):
    """
    Load a text file and return its content.
    Args:
        file_path (str): The path to the text file.
    Outputs:
        text (str): The content of the text file.
    """
    with open(file_path, "r") as f:
        text = f.read()
    
    logger.info(f"Text file read: {file_path}")
    return text

def write_data(text:str, text_type:str, actor:list, actor_subject:int, action:list, victim:list, extra:list):
    """
    Write data to a CSV file.
    Args:
        text (str): The text to write.
        text_type (str): The type of the text (tvc: transitive verb construct, ntvc: non-transitive verb construct, ic: irrelevant construct).
        actor (list): The index of tokens associated with the actor in the text.
        actor_subject (int): The index of the subject of the actor.
        action (list): The index of tokens associated with the action in the text.
        victim (list): The index of tokens associated with the victim in the text.
        extra (list): Any extra information to be added to the dataset.
    Outputs:
        None
    """
    data = {
        'text': text,
        'text_type': text_type,
        'actor': actor,
        'actor_subject': actor_subject,
        'action': action,
        'victim': victim
    }

    df = pd.DataFrame([data])
    if os.path.exists('data/dataset.csv'):
        df.to_csv('data/dataset.csv', mode='a', header=False, index=False)
    else:
        df.to_csv('data/dataset.csv', index=False)

    logger.info(f"Data written to dataset: {data}")
