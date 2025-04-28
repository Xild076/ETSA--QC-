import streamlit as st
import spacy
from spacy import displacy
import pandas as pd
import os
import logging
from collections import defaultdict

st.set_page_config(layout="wide", page_title="Annotation Tool v2")

LOG_DIR = "logs"
DATA_DIR = "data"
TEXTS_DIR = os.path.join(DATA_DIR, "texts")
DATASET_FILE = os.path.join(DATA_DIR, 'dataset.csv')
DEFAULT_SPACY_MODEL = "en_core_web_sm"
DEFAULT_DISPLACY_DISTANCE = 100
ROLES = ['-', 'Actor', 'Action', 'Victim']
ROLE_COLORS = {'Actor': 'blue', 'Action': 'green', 'Victim': 'red', '-': 'grey'}
ROLE_SHORT = {'Actor': 'A', 'Action': 'Act', 'Victim': 'V', '-': '-'}

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
logging.basicConfig(filename=os.path.join(LOG_DIR, "QCLog.log"), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(TEXTS_DIR):
    os.makedirs(TEXTS_DIR)
    st.warning(f"Created texts directory at '{TEXTS_DIR}'. Please add your .txt files there.")

@st.cache_resource
def load_spacy_model(model_name=DEFAULT_SPACY_MODEL):
    try:
        return spacy.load(model_name)
    except OSError:
        st.error(f"SpaCy model '{model_name}' not found. Download: python -m spacy download {model_name}")
        st.stop()

nlp = load_spacy_model()

def create_dataset_file_if_needed(file_name):
    if not os.path.exists(file_name) or os.path.getsize(file_name) == 0:
        try:
            with open(file_name, "w", encoding='utf-8') as f:
                f.write("text,text_type,actor,actor_subject,action,victim,extra\n")
            logging.info(f"Dataset file created/header written: {file_name}")
        except IOError as e:
            st.error(f"Failed to create/write header for dataset file: {e}")
            logging.error(f"Failed to create/write header for dataset file: {e}")
            st.stop()

create_dataset_file_if_needed(DATASET_FILE)

@st.cache_data
def load_text_file_names_cached(directory:str):
    text_files = []
    if not os.path.isdir(directory):
        return []
    try:
        for filename in os.listdir(directory):
            if filename.lower().endswith(".txt"):
                text_files.append(os.path.join(directory, filename))
    except Exception as e:
        logging.error(f"Error loading file names from {directory}: {e}")
        return []
    return sorted(text_files)

@st.cache_data
def load_text_file_cached(file_path:str):
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            text = f.read()
        return text
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return None

@st.cache_data
def split_text_cached(_text_hash: int, text:str):
    if not text: return []
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def write_data_to_csv(text:str, text_type:str, actor:list, actor_subject:int, action:list, victim:list, extra:list):
    data = {'text': [text], 'text_type': [text_type], 'actor': [str(sorted(actor))], 'actor_subject': [actor_subject], 'action': [str(sorted(action))], 'victim': [str(sorted(victim))], 'extra': [str(sorted(extra))]}
    df = pd.DataFrame(data)
    try:
        header = not os.path.exists(DATASET_FILE) or os.path.getsize(DATASET_FILE) == 0
        df.to_csv(DATASET_FILE, mode='a', header=header, index=False, encoding='utf-8')
        logging.info(f"Data written: {text[:50]}...")
    except IOError as e:
        st.error(f"Error writing to dataset file: {e}")
        logging.error(f"Error writing to dataset file: {e}")

def initialize_state():
    defaults = {
        'file_index': 0, 'sentence_index': 0, 'sentences': [], 'current_doc': None,
        'text_type': 'ic', 'token_roles': {}, 'actor_subject_index': -1,
        'all_files': load_text_file_names_cached(TEXTS_DIR), 'current_file_path': None,
        'displacy_distance': DEFAULT_DISPLACY_DISTANCE
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

    if not st.session_state.all_files: st.sidebar.warning(f"No .txt files found in '{TEXTS_DIR}'.")
    elif st.session_state.current_file_path is None:
         st.session_state.current_file_path = st.session_state.all_files[st.session_state.file_index]
         load_file(st.session_state.current_file_path)

def load_file(file_path):
    try: index = st.session_state.all_files.index(file_path)
    except ValueError: st.error(f"File '{os.path.basename(file_path)}' not found."); return

    text = load_text_file_cached(file_path)
    if text is not None:
        text_hash = hash(text)
        st.session_state.sentences = split_text_cached(text_hash, text)
        st.session_state.sentence_index = 0
        st.session_state.file_index = index
        st.session_state.current_file_path = file_path
        load_sentence(0)
    else:
        st.session_state.sentences = []; st.session_state.current_doc = None
        st.session_state.token_roles = {}; st.session_state.current_file_path = file_path

def load_sentence(index):
    st.session_state.actor_subject_index = -1
    st.session_state.token_roles = {}

    if st.session_state.sentences and 0 <= index < len(st.session_state.sentences):
        st.session_state.sentence_index = index
        sentence_text = st.session_state.sentences[index]
        st.session_state.current_doc = nlp(sentence_text)
        st.session_state.token_roles = {token.i: '-' for token in st.session_state.current_doc}
        st.session_state.text_type = 'ic'
    elif st.session_state.sentences:
        current_index = st.session_state.sentence_index
        if index >= len(st.session_state.sentences):
            st.toast("End of sentences."); st.session_state.sentence_index = len(st.session_state.sentences) - 1
        else:
            st.toast("First sentence."); st.session_state.sentence_index = 0
        if st.session_state.sentence_index != current_index:
             sentence_text = st.session_state.sentences[st.session_state.sentence_index]
             st.session_state.current_doc = nlp(sentence_text)
             st.session_state.token_roles = {token.i: '-' for token in st.session_state.current_doc}
    else:
        st.session_state.current_doc = None; st.session_state.token_roles = {}

def update_token_role(token_index, new_role):
    st.session_state.token_roles[token_index] = new_role
    if new_role != 'Actor' and st.session_state.actor_subject_index == token_index:
        st.session_state.actor_subject_index = -1

initialize_state()

with st.sidebar:
    st.header("📄 File Navigation")
    if st.session_state.all_files:
        selected_file = st.selectbox(
            "Select Text File", options=st.session_state.all_files, index=st.session_state.file_index,
            format_func=lambda x: os.path.basename(x) if x else "None", key="file_selector", help="Select file"
        )
        if selected_file != st.session_state.current_file_path:
            load_file(selected_file); st.rerun()

        st.divider()
        st.subheader("🧭 Sentence Navigation")
        nav_cols = st.columns(2)
        if nav_cols[0].button("⬅️ Prev", use_container_width=True, key="prev_sent"):
            if st.session_state.sentence_index > 0: load_sentence(st.session_state.sentence_index - 1); st.rerun()
            else: st.toast("First sentence.")
        if nav_cols[1].button("Next ➡️", use_container_width=True, key="next_sent"):
            if st.session_state.sentences and st.session_state.sentence_index < len(st.session_state.sentences) - 1:
                load_sentence(st.session_state.sentence_index + 1); st.rerun()
            elif st.session_state.sentences: st.toast("Last sentence.")

        if st.session_state.sentences:
             slider_key = f"slider_{st.session_state.current_file_path}"
             current_slider_val = st.session_state.sentence_index + 1
             new_slider_val = st.slider(
                  "Go to Sentence", 1, len(st.session_state.sentences), current_slider_val, key=slider_key
             )
             if new_slider_val != current_slider_val: load_sentence(new_slider_val - 1); st.rerun()

        st.divider()
        st.metric("Current File", f"{st.session_state.file_index + 1}/{len(st.session_state.all_files)}")
        st.metric("Current Sentence", f"{st.session_state.sentence_index + 1}/{len(st.session_state.sentences) if st.session_state.sentences else '0'}")
        st.divider()

        st.subheader("⚙️ Display Options")
        st.session_state.displacy_distance = st.slider(
            "Dependency Spacing", 60, 200, st.session_state.displacy_distance, 10, key="dist_slider",
             help="Adjust vertical spacing in the dependency parse. Use browser zoom (Ctrl +/-) for overall scaling."
        )
    else: st.info(f"Add .txt files to '{TEXTS_DIR}' to begin.")

st.title("📝 Annotation Interface")

if st.session_state.current_doc:
    current_sentence_text = st.session_state.sentences[st.session_state.sentence_index]

    st.markdown(f"**Sentence:**")
    st.markdown(f"> `{current_sentence_text}`")
    st.divider()

    main_cols = st.columns([6, 4])

    with main_cols[0]:
        st.subheader("🔍 Dependency Parse")
        dep_html = displacy.render(
            st.session_state.current_doc, style="dep", jupyter=False,
            options={'compact': True, 'bg': '#fafafa', 'color': '#1E1E1E', 'distance': st.session_state.displacy_distance}
        )
        st.components.v1.html(dep_html, height=400, scrolling=True)


    with main_cols[1]:
        st.subheader("🏷️ Annotation Controls")
        type_key = f"type_{st.session_state.current_file_path}_{st.session_state.sentence_index}"
        st.session_state.text_type = st.radio(
            "**Sentence Type:**", ('tvc', 'ntvc', 'ic'), index=['tvc', 'ntvc', 'ic'].index(st.session_state.text_type),
            horizontal=True, key=type_key, help="Transitive Verb (tvc), Non-Transitive (ntvc), Irrelevant (ic)."
        )

        if st.session_state.text_type == 'tvc':
            st.markdown("**Assign Token Roles (Click Button):**")
            annotation_grid = st.container(height=350)
            with annotation_grid:
                cols_per_row = 7
                token_cols = st.columns(cols_per_row)
                col_idx = 0
                for token in st.session_state.current_doc:
                    current_role = st.session_state.token_roles.get(token.i, '-')
                    with token_cols[col_idx % cols_per_row].container(border=True):
                         st.markdown(f"**{token.text}** `({token.i})`")
                         button_cols = st.columns(len(ROLES))
                         for i, role in enumerate(ROLES):
                             button_key = f"role_{token.i}_{role}_{st.session_state.sentence_index}"
                             is_selected = (current_role == role)
                             button_type = "primary" if is_selected else "secondary"
                             if button_cols[i].button(ROLE_SHORT[role], key=button_key, type=button_type, use_container_width=True, help=f"Set role to {role}"):
                                 update_token_role(token.i, role)
                                 st.rerun()
                    col_idx += 1


            st.markdown("**Assign Actor Subject:**")
            actor_indices = [idx for idx, role in st.session_state.token_roles.items() if role == 'Actor']
            actor_token_options = {
                f"{st.session_state.current_doc[idx].text} ({idx})": idx for idx in sorted(actor_indices)
            }

            if actor_token_options:
                actor_options_list = ["- (None)"] + list(actor_token_options.keys())
                current_subject_display = "- (None)"
                current_select_index = 0
                for i, display_str in enumerate(actor_options_list):
                    if display_str != "- (None)" and actor_token_options[display_str] == st.session_state.actor_subject_index:
                        current_subject_display = display_str; current_select_index = i; break

                subject_key = f"subject_{st.session_state.current_file_path}_{st.session_state.sentence_index}"
                selected_subject_display = st.selectbox(
                    "Select the *single* subject token from 'Actor' tokens:", options=actor_options_list,
                    index=current_select_index, key=subject_key, help="Main subject word among Actors."
                )
                st.session_state.actor_subject_index = actor_token_options.get(selected_subject_display, -1)
            else:
                st.info("Assign 'Actor' roles above to select a subject.")
                st.session_state.actor_subject_index = -1
        else:
            st.info("Assign roles and subject only for 'tvc' sentences.")
            st.session_state.actor_subject_index = -1

    st.divider()
    save_cols = st.columns([1, 3])
    with save_cols[0]:
        if st.button("💾 Save & Next ➡️", type="primary", use_container_width=True, key="save_next"):
            actor_indices_to_save, action_indices_to_save, victim_indices_to_save, extra_indices_to_save = [], [], [], []
            actor_subject_index_to_save = -1
            all_indices = list(range(len(st.session_state.current_doc)))
            assigned_indices = set()

            if st.session_state.text_type == 'tvc':
                roles_dict = st.session_state.token_roles
                actor_indices_to_save = sorted([idx for idx, role in roles_dict.items() if role == 'Actor'])
                action_indices_to_save = sorted([idx for idx, role in roles_dict.items() if role == 'Action'])
                victim_indices_to_save = sorted([idx for idx, role in roles_dict.items() if role == 'Victim'])
                actor_subject_index_to_save = st.session_state.actor_subject_index
                assigned_indices = set(roles_dict.keys()) - set(idx for idx, role in roles_dict.items() if role == '-')

                if actor_subject_index_to_save == -1 and actor_indices_to_save:
                     st.toast("Warning: Saving TVC without Actor Subject.", icon="⚠️")
                elif actor_subject_index_to_save != -1 and actor_subject_index_to_save not in actor_indices_to_save:
                     st.toast("Warning: Actor Subject is not marked as 'Actor'.", icon="⚠️")

            extra_indices_to_save = sorted([idx for idx in all_indices if idx not in assigned_indices])

            write_data_to_csv(
                text=current_sentence_text, text_type=st.session_state.text_type,
                actor=actor_indices_to_save, actor_subject=actor_subject_index_to_save,
                action=action_indices_to_save, victim=victim_indices_to_save, extra=extra_indices_to_save
            )
            st.toast(f"Annotation saved!", icon="✅")

            if st.session_state.sentences and st.session_state.sentence_index < len(st.session_state.sentences) - 1:
                load_sentence(st.session_state.sentence_index + 1); st.rerun()
            elif st.session_state.sentences: st.toast("Last sentence annotated for this file.")

elif not st.session_state.all_files:
    st.info(f"Please add text files (.txt) to the '{TEXTS_DIR}' directory and refresh.")
else:
    st.info("Select a file from the sidebar to start.")