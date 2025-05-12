import streamlit as st
import spacy
from spacy import displacy
import pandas as pd
import os
import logging
from collections import defaultdict
import gspread
from google.oauth2.service_account import Credentials
import time
import json

st.set_page_config(layout="wide", page_title="Annotation Tool")

LOG_DIR = "logs"
DATA_DIR = "data"
TEXTS_DIR = os.path.join(DATA_DIR, "texts")
DATASET_FILE = os.path.join(DATA_DIR, 'dataset.csv')
DEFAULT_SPACY_MODEL = "en_core_web_sm"
DEFAULT_DISPLACY_DISTANCE = 100
ROLES = ['-', 'Actor', 'Action', 'Victim']
ROLE_COLORS = {'Actor': '#007bff', 'Action': '#28a745', 'Victim': '#dc3545', '-': 'inherit'}
ROLE_SHORT = {'Actor': 'A', 'Action': 'Act', 'Victim': 'V', '-': '-'}

GOOGLE_SHEET_ID = "1PG9W38Ygn1l1hW23vAYJauzSfT7jk7YtrGo_m7Ar4Qs"
SERVICE_ACCOUNT_FILE_PATH = "google_credentials.json" 

LOCK_SHEET_NAME = "AnnotationLocks"
LOCK_TIMEOUT_SECONDS = 300
LOCK_COL_ID = "LockID"
LOCK_COL_ANNOTATOR = "Annotator"
LOCK_COL_TIMESTAMP = "TimestampEpoch"
LOCK_COL_FILEPATH = "FilePath"
LOCK_COL_SENTENCE_IDX = "SentenceIndex"
LOCK_HEADER = [LOCK_COL_ID, LOCK_COL_ANNOTATOR, LOCK_COL_TIMESTAMP, LOCK_COL_FILEPATH, LOCK_COL_SENTENCE_IDX]

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
logging.basicConfig(filename=os.path.join(LOG_DIR, "AnnotationLog.log"), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

if not os.path.isdir(TEXTS_DIR):
    st.error(
        f"Error: The required directory for text files ('{TEXTS_DIR}') was not found. "
        f"If you are hosting this application (e.g., on Streamlit Cloud from GitHub), "
        f"please ensure that the '{TEXTS_DIR}' directory exists in your repository, "
        f"is correctly spelled, and contains your .txt files. "
        f"The application expects these files to be present in that location within the deployed environment."
    )

if 'annotator_name' not in st.session_state or not st.session_state.annotator_name:
    st.session_state.annotator_name = ""
    placeholder = st.empty()
    with placeholder.container():
        st.title("Welcome to the Collaborative Annotation Tool")
        st.markdown("Please enter a unique annotator name/ID to begin.")
        name_input = st.text_input("Your Annotator Name/ID:", key="annotator_name_input_field", placeholder="E.g., user_gamma")
        if st.button("Confirm Name and Start Annotation", key="confirm_annotator_name", type="primary"):
            if name_input.strip():
                st.session_state.annotator_name = name_input.strip()
                logger.info(f"Annotator '{st.session_state.annotator_name}' started session.")
                placeholder.empty()
                st.rerun()
            else:
                st.warning("Annotator name cannot be empty. Please enter a valid name.")
    st.stop()

@st.cache_resource
def load_spacy_model(model_name=DEFAULT_SPACY_MODEL):
    try:
        return spacy.load(model_name)
    except OSError:
        st.error(f"SpaCy model '{model_name}' not found. Please download it by running: `python -m spacy download {model_name}` in your terminal.")
        st.stop()
nlp = load_spacy_model()

def create_dataset_file_if_needed(file_name):
    if not os.path.exists(file_name) or os.path.getsize(file_name) == 0:
        try:
            with open(file_name, "w", encoding='utf-8') as f:
                f.write("text,text_type,actor,actor_subject,action,victim,extra,annotator\n")
            logger.info(f"Created local dataset file: {file_name}")
        except IOError as e:
            st.error(f"Failed to create or access local dataset file: {e}")
            logger.error(f"IOError creating/accessing local dataset file {file_name}: {e}")
            st.stop()
create_dataset_file_if_needed(DATASET_FILE)

@st.cache_resource
def get_gspread_client():
    logger.info("Attempting to get gspread client.")
    st.session_state.gspread_client_available = False # Default to False
    try:
        scopes = ["https.www.googleapis.com/auth/spreadsheets"]
        creds_json_str_or_dict = st.secrets.get("google_service_account_credentials")
        creds = None
        if creds_json_str_or_dict:
            logger.info("Found Google credentials in Streamlit secrets.")
            if isinstance(creds_json_str_or_dict, str):
                try:
                    creds_dict = json.loads(creds_json_str_or_dict) # Removed strict=False
                    logger.info("Successfully parsed JSON string from secrets.")
                except json.JSONDecodeError as e:
                    st.error(f"Failed to parse Google service account credentials from Streamlit secrets (JSON error): {e}.")
                    logger.error(f"JSONDecodeError parsing Google credentials from st.secrets: {e}")
                    return None # gspread_client_available remains False
            elif isinstance(creds_json_str_or_dict, dict):
                creds_dict = creds_json_str_or_dict
                logger.info("Credentials in secrets are already a dict.")
            else:
                st.error("Google service account credentials in Streamlit secrets are not in a valid format (should be JSON string or dict).")
                logger.error("Invalid format for Google credentials in st.secrets.")
                return None # gspread_client_available remains False
            creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
            logger.info("Credentials loaded from service account info (secrets).")
        elif os.path.exists(SERVICE_ACCOUNT_FILE_PATH):
            logger.info(f"Found Google credentials at local file path: {SERVICE_ACCOUNT_FILE_PATH}.")
            creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE_PATH, scopes=scopes)
            logger.warning("Loaded Google credentials from local file. For security, consider using Streamlit secrets for production deployments.")
        else:
            st.warning("Google Sheets credentials not found in Streamlit secrets or local file. Collaborative features disabled.", icon="⚠️")
            logger.warning("Critical: Neither Streamlit secrets nor local credentials file found.")
            return None # gspread_client_available remains False

        client = gspread.authorize(creds)
        logger.info("gspread.authorize successful.")
        st.session_state.gspread_client_available = True # Set to True only on full success
        logger.info("Successfully connected to Google Sheets API and set gspread_client_available to True.")
        return client
    except Exception as e:
        # gspread_client_available remains False (or is set again)
        st.session_state.gspread_client_available = False 
        st.warning(f"Could not connect to Google Sheets: {e}. Collaborative features may be limited or unavailable.", icon="📡")
        logger.error(f"Google Sheets connection failed in get_gspread_client: {e}", exc_info=True)
        return None
gs_client = get_gspread_client()

@st.cache_resource
def get_locks_worksheet():
    logger.info("Attempting to get locks worksheet.")
    if not st.session_state.get('gspread_client_available', False) or gs_client is None:
        logger.warning("Cannot get locks worksheet: gspread client not available or gs_client is None.")
        return None
    if not GOOGLE_SHEET_ID or GOOGLE_SHEET_ID == "YOUR_GOOGLE_SHEET_ID_HERE":
        st.warning("Google Sheet ID for locking is not configured. Real-time lock management will be disabled.", icon="⚙️")
        logger.warning("Google Sheet ID for locking not configured.")
        return None
    try:
        logger.info(f"Opening spreadsheet by key: {GOOGLE_SHEET_ID}")
        spreadsheet = gs_client.open_by_key(GOOGLE_SHEET_ID)
        logger.info(f"Successfully opened spreadsheet: {spreadsheet.title}")
        try:
            worksheet = spreadsheet.worksheet(LOCK_SHEET_NAME)
            logger.info(f"Found existing locks worksheet: {LOCK_SHEET_NAME}")
        except gspread.exceptions.WorksheetNotFound:
            logger.info(f"Locks worksheet '{LOCK_SHEET_NAME}' not found, creating it.")
            worksheet = spreadsheet.add_worksheet(title=LOCK_SHEET_NAME, rows="1", cols=str(len(LOCK_HEADER)))
            worksheet.append_row(LOCK_HEADER, value_input_option='USER_ENTERED')
            logger.info(f"Created missing '{LOCK_SHEET_NAME}' sheet in Google Sheet ID: {GOOGLE_SHEET_ID}.")
        return worksheet
    except Exception as e: 
        st.error(f"Error accessing or creating the locks sheet ('{LOCK_SHEET_NAME}') in Google Sheets: {e}")
        logger.error(f"Error with locks sheet '{LOCK_SHEET_NAME}': {e}", exc_info=True)
        return None
locks_ws = get_locks_worksheet()

def make_lock_id(file_path, sentence_index):
    return f"{os.path.basename(file_path)}_{sentence_index}"

def acquire_lock(lock_id, annotator_name, file_path, sentence_index):
    if locks_ws is None:
        logger.warning(f"Acquire lock for {lock_id} skipped: locks_ws is None.")
        return True, "Lock system unavailable (Google Sheets not connected or sheet misconfigured), proceeding without lock."
    logger.info(f"Attempting to acquire lock: {lock_id} for {annotator_name}")
    try:
        current_time_epoch = int(time.time())
        cells = locks_ws.findall(lock_id, in_column=LOCK_HEADER.index(LOCK_COL_ID) + 1)
        
        if cells:
            cell = cells[0] 
            if len(cells) > 1:
                 logger.warning(f"Multiple lock entries found for {lock_id}. Using the first one at row {cell.row}.")

            row_values = locks_ws.row_values(cell.row)
            locked_by_idx = LOCK_HEADER.index(LOCK_COL_ANNOTATOR)
            timestamp_idx = LOCK_HEADER.index(LOCK_COL_TIMESTAMP)

            if len(row_values) <= max(locked_by_idx, timestamp_idx):
                logger.error(f"Lock row {cell.row} for {lock_id} is malformed: {row_values}. Releasing potentially corrupt lock.")
                locks_ws.delete_rows(cell.row) 
                new_row = [lock_id, annotator_name, current_time_epoch, os.path.basename(file_path), sentence_index]
                locks_ws.append_row(new_row, value_input_option='USER_ENTERED')
                return True, "Acquired lock after removing malformed entry."

            locked_by = row_values[locked_by_idx]
            lock_time = int(row_values[timestamp_idx])

            if locked_by == annotator_name:
                locks_ws.update_cell(cell.row, timestamp_idx + 1, current_time_epoch)
                logger.info(f"Lock refreshed for {lock_id} by {annotator_name}.")
                return True, "Lock refreshed."
            elif (current_time_epoch - lock_time) < LOCK_TIMEOUT_SECONDS:
                time_remaining = LOCK_TIMEOUT_SECONDS - (current_time_epoch - lock_time)
                locked_until_str = time.strftime('%H:%M:%S', time.localtime(lock_time + LOCK_TIMEOUT_SECONDS))
                logger.info(f"Lock attempt failed for {lock_id}. Locked by {locked_by} for another {time_remaining}s.")
                return False, f"Sentence locked by **{locked_by}**. Try again in {time_remaining//60}m {time_remaining%60}s (until ~{locked_until_str})."
            else: 
                locks_ws.update_cell(cell.row, locked_by_idx + 1, annotator_name)
                locks_ws.update_cell(cell.row, timestamp_idx + 1, current_time_epoch)
                logger.info(f"Acquired stale lock for {lock_id} from {locked_by} by {annotator_name}.")
                return True, f"Acquired stale lock from {locked_by}."
        else: 
            new_row = [lock_id, annotator_name, current_time_epoch, os.path.basename(file_path), sentence_index]
            locks_ws.append_row(new_row, value_input_option='USER_ENTERED')
            logger.info(f"Lock acquired for {lock_id} by {annotator_name}.")
            return True, "Lock acquired."
    except gspread.exceptions.APIError as e:
        err_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                err_details = e.response.json().get('error', {}).get('message', e.response.text)
                err_msg = f"{err_details} (Raw response: {e.response.text})"
            except json.JSONDecodeError:
                err_msg = e.response.text
        logger.error(f"Google Sheets APIError during acquire_lock for {lock_id} by {annotator_name}: {err_msg}", exc_info=True)
        st.error(f"Google Sheets API Error (acquiring lock): {err_msg}")
        return False, "Google Sheets API error prevented lock acquisition. Please try again."
    except Exception as e:
        logger.error(f"Unexpected error in acquire_lock for {lock_id} by {annotator_name}: {e}", exc_info=True)
        st.error(f"An unexpected error occurred with the locking system: {e}")
        return False, "Locking system error."

def release_lock(lock_id, annotator_name):
    if locks_ws is None:
        logger.warning(f"Release lock for {lock_id} skipped: locks_ws is None.")
        return True
    logger.info(f"Attempting to release lock: {lock_id} for {annotator_name}")
    try:
        cells = locks_ws.findall(lock_id, in_column=LOCK_HEADER.index(LOCK_COL_ID) + 1)
        if cells:
            for cell in cells: 
                try:
                    annotator_col_val = locks_ws.cell(cell.row, LOCK_HEADER.index(LOCK_COL_ANNOTATOR) + 1).value
                    if annotator_col_val == annotator_name:
                        locks_ws.delete_rows(cell.row)
                        logger.info(f"Lock {lock_id} released by {annotator_name} from row {cell.row}.")
                    else:
                        logger.warning(f"Attempt to release lock {lock_id} by {annotator_name}, but it's held by {annotator_col_val} or cell is empty.")
                        return False 
                except gspread.exceptions.CellNotFound: 
                    logger.warning(f"Cell not found when trying to release lock {lock_id} at row {cell.row}. Already released or deleted.")
                except Exception as e_inner: 
                    logger.error(f"Error processing cell {cell.row} for lock {lock_id} release by {annotator_name}: {e_inner}", exc_info=True)
            return True 
        logger.info(f"No lock found for {lock_id} to release.")
        return True 
    except gspread.exceptions.APIError as e:
        err_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                err_details = e.response.json().get('error', {}).get('message', e.response.text)
                err_msg = f"{err_details} (Raw response: {e.response.text})"
            except json.JSONDecodeError:
                err_msg = e.response.text
        logger.error(f"Google Sheets APIError during release_lock for {lock_id} by {annotator_name}: {err_msg}", exc_info=True)
        st.error(f"Google Sheets API Error (releasing lock): {err_msg}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in release_lock for {lock_id} by {annotator_name}: {e}", exc_info=True)
        st.error(f"An unexpected error occurred while releasing the lock: {e}")
        return False

@st.cache_data
def load_text_file_names_cached(directory:str):
    if not os.path.isdir(directory):
        logger.warning(f"Texts directory '{directory}' not found when trying to load file names.")
        return []
    try:
        return sorted([os.path.join(directory, fn) for fn in os.listdir(directory) if fn.lower().endswith(".txt")])
    except Exception as e:
        logger.error(f"Error loading file list from {directory}: {e}", exc_info=True)
        st.error(f"Could not load file list from '{directory}'. Check permissions and path.")
        return []

@st.cache_data
def load_text_file_cached(file_path:str):
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        st.error(f"File not found: {os.path.basename(file_path)}")
        return None
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
        st.error(f"Could not read file: {os.path.basename(file_path)}. Error: {e}")
        return None

@st.cache_data
def split_text_cached(_text_hash: int, text:str): 
    if not text:
        return []
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def write_to_google_sheet(df_to_append):
    logger.info("Attempting to write to Google Sheet.")
    if not st.session_state.get('gspread_client_available', False) or gs_client is None:
        logger.warning("Cannot write to Google Sheet: gspread client not available or gs_client is None.")
        st.toast("Google Sheets client not available. Data not saved to cloud.", icon="⚠️")
        return False
    if not GOOGLE_SHEET_ID or GOOGLE_SHEET_ID == "YOUR_GOOGLE_SHEET_ID_HERE":
        logger.warning("Google Sheet ID for data not configured. Skipping Google Sheet write.")
        st.toast("Google Sheet ID not configured. Data not saved to cloud.", icon="⚙️")
        return False
    
    logger.info(f"Target Google Sheet ID for data: {GOOGLE_SHEET_ID}")
    try:
        logger.info(f"Opening spreadsheet by key: {GOOGLE_SHEET_ID}")
        spreadsheet = gs_client.open_by_key(GOOGLE_SHEET_ID)
        logger.info(f"Successfully opened spreadsheet for data: {spreadsheet.title}")
        worksheet = spreadsheet.sheet1 
        logger.info(f"Successfully accessed data worksheet: {worksheet.title}")
        
        header = worksheet.row_values(1) if worksheet.row_count > 0 else []
        if not header or set(header) != set(df_to_append.columns):
            logger.info("Data sheet header mismatch or sheet empty. Clearing and writing new header.")
            worksheet.clear() 
            worksheet.append_row(list(df_to_append.columns), value_input_option='USER_ENTERED')
            logger.info("Initialized Google Sheet header for annotations (data sheet).")

        logger.info(f"Attempting to append {len(df_to_append)} rows to data worksheet '{worksheet.title}'.")
        worksheet.append_rows(df_to_append.values.tolist(), value_input_option='USER_ENTERED')
        logger.info(f"Successfully appended {len(df_to_append)} rows to Google Sheet (data sheet).")
        return True
    except gspread.exceptions.APIError as e:
        err_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                err_details = e.response.json().get('error', {}).get('message', e.response.text)
                err_msg = f"{err_details} (Raw response: {e.response.text})"
            except json.JSONDecodeError:
                 err_msg = e.response.text
        st.error(f"Google Sheets API Error (writing data): {err_msg}")
        logger.error(f"Google Sheets APIError during write_to_google_sheet (data): {err_msg}", exc_info=True)
        return False
    except Exception as e:
        st.error(f"Failed to write annotation to Google Sheet (data): {e}")
        logger.error(f"Failed to write to Google Sheet (data): {e}", exc_info=True)
        return False

def write_data_to_local_csv(df_to_append):
    try:
        file_exists_and_not_empty = os.path.exists(DATASET_FILE) and os.path.getsize(DATASET_FILE) > 0
        df_to_append.to_csv(DATASET_FILE, mode='a', header=not file_exists_and_not_empty, index=False, encoding='utf-8')
        logger.info(f"Appended {len(df_to_append)} rows to local CSV: {DATASET_FILE}")
    except IOError as e:
        st.error(f"Failed to write annotation to local CSV file: {e}")
        logging.error(f"IOError writing to local CSV {DATASET_FILE}: {e}", exc_info=True)

def save_annotation_data(text:str, text_type:str, actor:list, actor_subject:int, action:list, victim:list, extra:list, annotator:str):
    data = {
        'text': [text],
        'text_type': [text_type],
        'actor': [str(sorted(list(set(actor))))],
        'actor_subject': [actor_subject],
        'action': [str(sorted(list(set(action))))],
        'victim': [str(sorted(list(set(victim))))],
        'extra': [str(sorted(list(set(extra))))],
        'annotator': [annotator]
    }
    df = pd.DataFrame(data)
    write_data_to_local_csv(df.copy()) 

    if write_to_google_sheet(df.copy()):
        st.toast("Annotation also saved to Google Sheet.", icon="☁️")
    else:
        st.toast("Annotation saved locally. Failed to save to Google Sheet.", icon="💾") 
    
    logging.info(f"Annotation by {annotator} for text starting with: '{text[:50]}...' processed. Check toasts for GSheet status.")

def initialize_state():
    defaults = {
        'file_index': 0,
        'sentence_index': 0,
        'sentences': [],
        'current_doc': None,
        'text_type': 'ic',
        'token_roles': {},
        'actor_subject_index': -1,
        'all_files': load_text_file_names_cached(TEXTS_DIR),
        'current_file_path': None,
        'displacy_distance': DEFAULT_DISPLACY_DISTANCE,
        'gspread_client_available': st.session_state.get('gspread_client_available', False), # Persist this
        'current_sentence_lock_details': None,
        'sentence_load_status': "NONE",
        'sentence_lock_info_message': "",
        'last_known_good_sentence_index': -1,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    
    # Ensure gs_client is initialized if not already, this might re-trigger logs if it failed silently before
    global gs_client 
    if 'gs_client_initialized_run' not in st.session_state: # Run gs_client init once per session effectively
        gs_client = get_gspread_client() # This will log if it fails
        st.session_state.gs_client_initialized_run = True


    if not st.session_state.all_files and os.path.isdir(TEXTS_DIR): 
        st.warning(f"No .txt files found in the '{TEXTS_DIR}' directory. Please add text files and refresh the page.")
    elif st.session_state.current_file_path is None and st.session_state.all_files:
        st.session_state.current_file_path = st.session_state.all_files[st.session_state.file_index]
        load_file(st.session_state.current_file_path)

def load_file(file_path, target_sentence_idx=0):
    if st.session_state.get('current_sentence_lock_details'):
        release_lock(st.session_state.current_sentence_lock_details['id'], st.session_state.annotator_name)
        st.session_state.current_sentence_lock_details = None
        logger.info(f"Released previous sentence lock while loading new file: {file_path}")

    try:
        st.session_state.file_index = st.session_state.all_files.index(file_path)
    except ValueError:
        st.error(f"File '{os.path.basename(file_path)}' is not in the current list of files. Refreshing file list.")
        logger.error(f"File {file_path} not found in all_files list. Forcing reload of file list.")
        st.session_state.all_files = load_text_file_names_cached(TEXTS_DIR)
        if file_path in st.session_state.all_files:
            st.session_state.file_index = st.session_state.all_files.index(file_path)
        else:
            st.session_state.current_file_path = None
            st.session_state.sentences = []
            st.session_state.current_doc = None
            st.session_state.sentence_load_status = "FILE_NOT_FOUND_POST_REFRESH"
            st.session_state.sentence_lock_info_message = f"File '{os.path.basename(file_path)}' could not be found even after refreshing the list."
            return

    st.session_state.current_file_path = file_path
    st.session_state.last_known_good_sentence_index = -1
    text_content = load_text_file_cached(file_path)

    if text_content is not None:
        st.session_state.sentences = split_text_cached(hash(text_content), text_content)
        if not st.session_state.sentences:
            st.warning(f"The file '{os.path.basename(file_path)}' is empty or contains no parsable sentences.")
            logger.warning(f"File {file_path} resulted in no sentences.")
            st.session_state.current_doc = None
            st.session_state.token_roles = {}
            st.session_state.sentence_load_status = "EMPTY_FILE"
            st.session_state.sentence_lock_info_message = f"File '{os.path.basename(file_path)}' is empty or has no sentences."
        else:
            load_sentence(target_sentence_idx) 
    else:
        st.session_state.sentences = []
        st.session_state.current_doc = None
        st.session_state.token_roles = {}
        st.session_state.sentence_load_status = "FILE_LOAD_ERROR"
        st.session_state.sentence_lock_info_message = f"Error loading content from '{os.path.basename(file_path)}'."
        logger.error(f"Failed to load text content for {file_path}")

def load_sentence(target_idx):
    if not st.session_state.annotator_name:
        st.warning("Annotator name is not set. Please confirm your annotator name first.")
        logger.warning("load_sentence called without annotator_name set.")
        st.session_state.sentence_load_status = "ERROR_NO_ANNOTATOR"
        return

    if not st.session_state.sentences:
        st.session_state.current_doc = None
        st.session_state.token_roles = {}
        st.session_state.actor_subject_index = -1
        st.session_state.sentence_load_status = "NO_SENTENCES_IN_FILE"
        st.session_state.sentence_lock_info_message = "No sentences available in the current file."
        logger.info("load_sentence called but no sentences are loaded for the current file.")
        return

    clamped_idx = max(0, min(target_idx, len(st.session_state.sentences) - 1))
    current_fpath = st.session_state.current_file_path
    lock_id = make_lock_id(current_fpath, clamped_idx)

    old_lock_details = st.session_state.get('current_sentence_lock_details')
    if old_lock_details and old_lock_details['id'] != lock_id:
        release_lock(old_lock_details['id'], st.session_state.annotator_name)
        st.session_state.current_sentence_lock_details = None
        logger.info(f"Released lock for {old_lock_details['id']} before acquiring new lock for {lock_id}.")

    with st.spinner(f"Attempting to load and lock sentence {clamped_idx + 1}..."):
        can_lock, lock_msg = acquire_lock(lock_id, st.session_state.annotator_name, current_fpath, clamped_idx)

    if can_lock:
        st.session_state.current_sentence_lock_details = {'id': lock_id, 'file': current_fpath, 'idx': clamped_idx}
        st.session_state.sentence_index = clamped_idx
        st.session_state.last_known_good_sentence_index = clamped_idx
        sentence_text = st.session_state.sentences[clamped_idx]

        st.session_state.actor_subject_index = -1
        st.session_state.token_roles = {}
        st.session_state.text_type = 'ic' 

        try:
            st.session_state.current_doc = nlp(sentence_text)
            st.session_state.token_roles = {token.i: '-' for token in st.session_state.current_doc}
            st.session_state.sentence_load_status = "SUCCESS"
            st.session_state.sentence_lock_info_message = f"Sentence {clamped_idx + 1} of '{os.path.basename(current_fpath)}' loaded and locked by you. {lock_msg}" # lock_msg might indicate GSheet issue
            st.toast(st.session_state.sentence_lock_info_message, icon="✅" if "unavailable" not in lock_msg else "⚠️" )
            logger.info(f"Successfully loaded and locked sentence {clamped_idx} of {current_fpath} by {st.session_state.annotator_name}. Lock message: {lock_msg}")
        except Exception as e:
            st.error(f"Error processing sentence '{sentence_text[:50]}...': {e}")
            logger.error(f"SpaCy or processing error for sentence '{sentence_text[:50]}...' in {current_fpath}: {e}", exc_info=True)
            st.session_state.current_doc = None
            st.session_state.token_roles = {}
            if "unavailable" not in lock_msg : # Only release if we think we got a real lock
                release_lock(lock_id, st.session_state.annotator_name) 
            st.session_state.current_sentence_lock_details = None
            st.session_state.sentence_load_status = "PROCESSING_ERROR"
            st.session_state.sentence_lock_info_message = f"Error processing sentence {clamped_idx+1}. Lock released."
    else: 
        st.session_state.sentence_lock_info_message = lock_msg 
        st.error(lock_msg)
        logger.warning(f"Failed to acquire lock for sentence {clamped_idx} of {current_fpath}. Message: {lock_msg}")

        if st.session_state.last_known_good_sentence_index != -1 and st.session_state.last_known_good_sentence_index != clamped_idx:
            st.toast(f"Failed to lock sentence {clamped_idx+1}. Reverting to previously loaded sentence {st.session_state.last_known_good_sentence_index+1}.", icon="↩️")
            logger.info(f"Reverting to sentence {st.session_state.last_known_good_sentence_index} due to lock failure on {clamped_idx}.")
            load_sentence(st.session_state.last_known_good_sentence_index) 
            return 

        st.session_state.current_doc = None 
        st.session_state.sentence_load_status = "LOCKED_BY_OTHER"

def handle_token_role_update(token_index, new_role):
    st.session_state.token_roles[token_index] = new_role
    if new_role != 'Actor' and st.session_state.actor_subject_index == token_index:
        st.session_state.actor_subject_index = -1
    logger.debug(f"Token {token_index} role updated to {new_role} by {st.session_state.annotator_name}. Actor subject index is now {st.session_state.actor_subject_index}.")

initialize_state() # gs_client gets initialized here if not already

with st.sidebar:
    st.header(f"Annotator: {st.session_state.annotator_name}")
    st.divider()
    st.header("📄 File Navigation")
    if st.session_state.all_files:
        current_file_idx_in_options = 0
        if st.session_state.current_file_path and st.session_state.current_file_path in st.session_state.all_files:
            try:
                current_file_idx_in_options = st.session_state.all_files.index(st.session_state.current_file_path)
            except ValueError: 
                 logger.warning(f"current_file_path {st.session_state.current_file_path} not in all_files. Resetting index.")
                 current_file_idx_in_options = 0

        selected_fpath = st.selectbox("Select Text File", options=st.session_state.all_files,
            index=current_file_idx_in_options,
            format_func=lambda x: os.path.basename(x) if x else "None", key="file_selector")

        if selected_fpath and selected_fpath != st.session_state.current_file_path:
            load_file(selected_fpath, target_sentence_idx=0)
            st.rerun()

        st.divider()
        st.subheader("🧭 Sentence Navigation")
        nav_cols = st.columns(2)
        prev_disabled = st.session_state.sentence_index <= 0 or not st.session_state.sentences or st.session_state.sentence_load_status != "SUCCESS"
        next_disabled = not st.session_state.sentences or st.session_state.sentence_index >= len(st.session_state.sentences) - 1 or st.session_state.sentence_load_status != "SUCCESS"

        if nav_cols[0].button("⬅️ Prev", use_container_width=True, key="prev_sent", disabled=prev_disabled):
            if st.session_state.sentence_index > 0:
                load_sentence(st.session_state.sentence_index - 1)
                st.rerun()

        if nav_cols[1].button("Next ➡️", use_container_width=True, key="next_sent", disabled=next_disabled):
            if st.session_state.sentences and st.session_state.sentence_index < len(st.session_state.sentences) - 1:
                load_sentence(st.session_state.sentence_index + 1)
                st.rerun()

        if st.session_state.sentences and st.session_state.sentence_load_status == "SUCCESS":
            current_slider_val = st.session_state.sentence_index + 1
            slider_key = f"sentence_slider_{st.session_state.current_file_path}_{len(st.session_state.sentences)}"
            new_slider_val = st.slider("Go to Sentence", 1, max(1, len(st.session_state.sentences)),
                                        current_slider_val, key=slider_key,
                                        disabled=(st.session_state.sentence_load_status != "SUCCESS"))
            if new_slider_val != current_slider_val:
                 load_sentence(new_slider_val - 1)
                 st.rerun()

            progress_val = (st.session_state.sentence_index + 1) / len(st.session_state.sentences) if len(st.session_state.sentences) > 0 else 0
            st.progress(progress_val, text=f"File Progress: Sentence {st.session_state.sentence_index + 1} of {len(st.session_state.sentences)}")
        elif st.session_state.sentences: 
             st.slider("Go to Sentence", 1, max(1, len(st.session_state.sentences)),
                                        st.session_state.sentence_index + 1 if st.session_state.sentence_index >=0 else 1,
                                        key=f"sentence_slider_disabled_{st.session_state.current_file_path}",
                                        disabled=True)
             st.progress(0, text="Sentence not loaded")

        st.divider()
        num_files = len(st.session_state.all_files)
        st.metric("Current File", f"{st.session_state.file_index + 1}/{num_files}" if num_files > 0 else "0/0",
                  delta=os.path.basename(st.session_state.current_file_path) if st.session_state.current_file_path else "N/A")

        num_sents_in_file = len(st.session_state.sentences)
        current_sent_display = f"{st.session_state.sentence_index + 1}/{num_sents_in_file}" if num_sents_in_file > 0 and st.session_state.current_doc else "0/0"
        st.metric("Current Sentence in File", current_sent_display)

        if st.session_state.current_sentence_lock_details and st.session_state.sentence_load_status == "SUCCESS":
            lock = st.session_state.current_sentence_lock_details
            lock_msg_sidebar = st.session_state.sentence_lock_info_message
            if "unavailable" in lock_msg_sidebar:
                 st.warning(f"⚠️ Sentence {lock['idx']+1} loaded. {lock_msg_sidebar}")
            else:
                 st.success(f"🔒 Sentence {lock['idx']+1} ('{os.path.basename(lock['file'])}') locked by you.")
        elif st.session_state.sentence_load_status == "LOCKED_BY_OTHER":
            st.error(f"🔒 {st.session_state.sentence_lock_info_message}")

        st.divider()
        st.subheader("⚙️ Display Options")
        st.session_state.displacy_distance = st.slider("Dependency Spacing", 60, 200, st.session_state.displacy_distance, 10, key="dist_slider",
                                                       help="Adjust vertical spacing for dependency parse. For overall size, use your browser's zoom (Ctrl/Cmd +/-).")
    else:
        st.info(f"No .txt files found in '{TEXTS_DIR}'. Please add text files to this directory and refresh the application.")
        if os.path.isdir(TEXTS_DIR):
            if st.button("Refresh File List", key="refresh_files_empty"):
                st.session_state.all_files = load_text_file_names_cached(TEXTS_DIR)
                if st.session_state.all_files:
                    st.session_state.current_file_path = st.session_state.all_files[0]
                    load_file(st.session_state.current_file_path)
                st.rerun()

st.title("📝 Annotation Interface")

if st.session_state.sentence_load_status == "SUCCESS" and st.session_state.current_doc:
    current_sentence_text = st.session_state.sentences[st.session_state.sentence_index]
    st.markdown(f"**Current Sentence ({st.session_state.sentence_index + 1}/{len(st.session_state.sentences)} from '{os.path.basename(st.session_state.current_file_path)}'):**")
    st.markdown(f"> `{current_sentence_text}`")
    st.divider()

    st.subheader("🔍 Dependency Parse")
    if len(st.session_state.current_doc) > 0:
        dep_html = displacy.render(st.session_state.current_doc, style="dep", jupyter=False,
            options={'compact': False, 'bg': '#fafafa', 'color': '#1E1E1E', 'distance': st.session_state.displacy_distance, 'word_spacing': 30, 'arrow_spacing': 10})
        st.components.v1.html(dep_html, height=60 + len(st.session_state.current_doc) * 15, scrolling=True)
    else:
        st.info("Sentence is empty or has no tokens that can be parsed.")
    
    st.divider() 
    st.subheader("🏷️ Annotation Controls")
    radio_key_base = f"type_{st.session_state.current_file_path}_{st.session_state.sentence_index}"

    current_type_idx = ['tvc', 'ntvc', 'ic'].index(st.session_state.text_type) if st.session_state.text_type in ['tvc', 'ntvc', 'ic'] else 2

    new_text_type = st.radio("**Sentence Type:**", ('tvc', 'ntvc', 'ic'),
                                index=current_type_idx, horizontal=True, key=radio_key_base,
                                help="TVC: Token-level Violence Corpus (requires role assignment). NTVC: Not TVC. IC: Irrelevant/Ignore.")
    if new_text_type != st.session_state.text_type:
        st.session_state.text_type = new_text_type
        logger.info(f"Sentence type changed to {new_text_type} for sentence {st.session_state.sentence_index} in {st.session_state.current_file_path} by {st.session_state.annotator_name}.")
        if new_text_type != 'tvc': 
                st.session_state.token_roles = {token.i: '-' for token in st.session_state.current_doc} if st.session_state.current_doc else {}
                st.session_state.actor_subject_index = -1
        st.rerun()

    if st.session_state.text_type == 'tvc':
        st.markdown("**Assign Token Roles:** (Actor, Action, Victim)")
        num_tokens = len(st.session_state.current_doc)
        grid_h = min(700, 100 + ((num_tokens + 2) // 3 * 70))

        with st.container(height=grid_h, border=False):
            token_cols_layout = st.columns(3) 
            for token_idx_in_doc, token_obj in enumerate(st.session_state.current_doc):
                current_role_for_token = st.session_state.token_roles.get(token_obj.i, '-')
                text_color_for_token = ROLE_COLORS.get(current_role_for_token, 'inherit')
                
                col_for_token = token_cols_layout[token_idx_in_doc % 3]
                with col_for_token:
                    with st.container(border=(current_role_for_token != '-')):
                        st.markdown(f"<div style='margin-bottom: 3px; text-align: center;'><strong style='color:{text_color_for_token}; font-size: 1.05em;'>{token_obj.text}</strong><br><code style='font-size:0.8em;'>id:{token_obj.i}</code></div>", unsafe_allow_html=True)
                        
                        current_role_idx = ROLES.index(current_role_for_token) if current_role_for_token in ROLES else 0
                        radio_button_key = f"role_radio_tok{token_obj.i}_{radio_key_base}"

                        selected_role = st.radio(
                            label=f"Role for token {token_obj.i}", 
                            options=ROLES,
                            index=current_role_idx,
                            key=radio_button_key,
                            format_func=lambda x: ROLE_SHORT[x],
                            horizontal=True,
                            label_visibility="collapsed"
                        )
                        if selected_role != current_role_for_token:
                            handle_token_role_update(token_obj.i, selected_role)

        st.markdown("**Assign Actor Subject (if applicable):**")
        actor_idxs = [idx for idx, r_val in st.session_state.token_roles.items() if r_val == 'Actor']
        actor_opts = { f"{st.session_state.current_doc[idx].text} (id:{idx})": idx for idx in sorted(actor_idxs) if idx < len(st.session_state.current_doc) }

        if actor_opts:
            opts_list = ["- (None Selected)"] + list(actor_opts.keys())
            curr_sel_idx_for_selectbox = 0 
            if st.session_state.actor_subject_index != -1:
                subj_disp_str = next((txt_disp for txt_disp, tok_idx_val in actor_opts.items() if tok_idx_val == st.session_state.actor_subject_index), None)
                if subj_disp_str and subj_disp_str in opts_list:
                    curr_sel_idx_for_selectbox = opts_list.index(subj_disp_str)
                else: 
                    st.session_state.actor_subject_index = -1
                    logger.info(f"Actor subject index {st.session_state.actor_subject_index} was invalid, reset to -1.")

            sel_subj_disp = st.selectbox("Select one token (from 'Actor' list) as the primary subject:",
                                            options=opts_list, index=curr_sel_idx_for_selectbox,
                                            key=f"subject_selector_{radio_key_base}",
                                            help="Choose the main actor performing the action.")
            new_subj_idx = -1
            if sel_subj_disp != "- (None Selected)" and sel_subj_disp in actor_opts:
                new_subj_idx = actor_opts[sel_subj_disp]

            if st.session_state.actor_subject_index != new_subj_idx:
                st.session_state.actor_subject_index = new_subj_idx
                logger.info(f"Actor subject index updated to {new_subj_idx} by {st.session_state.annotator_name}.")
                st.rerun() 
        else:
            st.info("To select an Actor Subject, first assign the 'Actor' role to one or more tokens.")
            if st.session_state.actor_subject_index != -1: 
                st.session_state.actor_subject_index = -1
                logger.info("Actor subject index reset to -1 as no tokens are marked 'Actor'.")
                st.rerun() 

    st.divider()
    save_cols = st.columns([1,3,1]) 
    with save_cols[0]: 
        if st.button("💾 Save & Next ➡️", type="primary", use_container_width=True, key="save_next",
                        help="Save current annotation and move to the next sentence. Releases lock on this sentence."):
            actor_to_save, action_to_save, victim_to_save, extra_to_save = [], [], [], []
            subj_to_save = -1
            all_token_indices = set(range(len(st.session_state.current_doc)))
            assigned_indices_in_tvc = set()

            if st.session_state.text_type == 'tvc':
                roles = st.session_state.token_roles
                actor_to_save = sorted([i for i, r in roles.items() if r == 'Actor'])
                action_to_save = sorted([i for i, r in roles.items() if r == 'Action'])
                victim_to_save = sorted([i for i, r in roles.items() if r == 'Victim'])
                subj_to_save = st.session_state.actor_subject_index

                assigned_indices_in_tvc.update(actor_to_save)
                assigned_indices_in_tvc.update(action_to_save)
                assigned_indices_in_tvc.update(victim_to_save)

                if not actor_to_save and (action_to_save or victim_to_save):
                    st.toast("Warning: Saving 'TVC' with Action/Victim roles but no 'Actor' token.", icon="⚠️")
                    logger.warning(f"TVC saved by {st.session_state.annotator_name} with Action/Victim but no Actor for: {current_sentence_text[:30]}")
                if actor_to_save and subj_to_save == -1 :
                    st.toast("Warning: Saving 'TVC' with 'Actor' token(s) but no Actor Subject selected.", icon="⚠️")
                    logger.warning(f"TVC saved by {st.session_state.annotator_name} with Actors but no subject for: {current_sentence_text[:30]}")
                elif subj_to_save != -1 and subj_to_save not in actor_to_save:
                        st.toast("Error: The selected Actor Subject is no longer marked as 'Actor'. Subject will not be saved.", icon="🚨")
                        logger.error(f"Actor Subject {subj_to_save} not in Actor list {actor_to_save} for {st.session_state.annotator_name} on: {current_sentence_text[:30]}. Subject reset.")
                        subj_to_save = -1 

            extra_to_save = sorted(list(all_token_indices - assigned_indices_in_tvc))

            save_annotation_data(current_sentence_text, st.session_state.text_type,
                                    actor_to_save, subj_to_save, action_to_save, victim_to_save,
                                    extra_to_save, st.session_state.annotator_name)


            if st.session_state.current_sentence_lock_details:
                # Only release if it wasn't a "proceeding without lock" situation
                if "unavailable" not in st.session_state.current_sentence_lock_details.get('lock_message', ''):
                    release_lock(st.session_state.current_sentence_lock_details['id'], st.session_state.annotator_name)
                st.session_state.current_sentence_lock_details = None # Clear it regardless
                # logger.info(f"Lock released for sentence {st.session_state.sentence_index} of {st.session_state.current_file_path} after saving.") # Redundant due to release_lock logging

            if st.session_state.sentences and st.session_state.sentence_index < len(st.session_state.sentences) - 1:
                load_sentence(st.session_state.sentence_index + 1)
            else: 
                st.toast("Last sentence of this file annotated and saved! Select a new file or sentence.", icon="🏁")
                logger.info(f"End of file {st.session_state.current_file_path} reached by {st.session_state.annotator_name}.")
                st.session_state.current_doc = None
                st.session_state.sentence_load_status = "END_OF_FILE"
                st.session_state.sentence_lock_info_message = "End of file reached. Please select another file."
            st.rerun()

elif st.session_state.sentence_load_status == "LOCKED_BY_OTHER":
    st.warning(f"**This sentence cannot be loaded at the moment:**")
    st.markdown(f"> {st.session_state.sentence_lock_info_message}")
    st.info("Please select a different sentence from the sidebar, or try loading this one again in a few minutes. The lock might have expired or been released by then.")
    if st.button("Try to Reload This Sentence", key="reload_locked_sentence"):
        load_sentence(st.session_state.sentence_index if st.session_state.sentence_index != -1 else 0) 
        st.rerun()

elif st.session_state.sentence_load_status != "NONE": 
    display_message = st.session_state.sentence_lock_info_message or "Please select a file and sentence to start annotating."
    if st.session_state.sentence_load_status in ["EMPTY_FILE", "FILE_LOAD_ERROR", "NO_SENTENCES_IN_FILE", "PROCESSING_ERROR", "END_OF_FILE", "ERROR_NO_ANNOTATOR", "FILE_NOT_FOUND_POST_REFRESH"]:
        if st.session_state.sentence_load_status.startswith("ERROR"):
             st.error(f"Error: {display_message}")
        else:
             st.warning(f"Info: {display_message}")
    else: 
        st.info(display_message)
    st.markdown("---")
    st.info("Use the sidebar navigation to select a file and sentence. If issues persist, check the application logs or contact support.")

elif not st.session_state.all_files :
    if not os.path.isdir(TEXTS_DIR): 
        pass 
    elif not load_text_file_names_cached(TEXTS_DIR): 
        st.info(f"Welcome, {st.session_state.annotator_name}! No .txt files were found in the '{TEXTS_DIR}' directory. Please add your text files there and click 'Refresh File List' or reload the page.")
    
    if os.path.isdir(TEXTS_DIR): 
        if st.button("Refresh File List", key="refresh_files_initial"):
            st.session_state.all_files = load_text_file_names_cached(TEXTS_DIR)
            if st.session_state.all_files: 
                st.session_state.current_file_path = st.session_state.all_files[0]
                load_file(st.session_state.current_file_path) 
            st.rerun()
else: 
    st.info("Please select a file from the sidebar to begin the annotation process.")
