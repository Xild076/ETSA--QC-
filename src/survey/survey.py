import random
import numpy as np
import streamlit as st
import pandas as pd
import json
import gspread
from google.oauth2.service_account import Credentials
import datetime
import re
from survey_question_gen import survey_gen

GOOGLE_SHEET_ID = "1xAvDLhU0w-p2hAZ49QYM7-XBMQCek0zVYJWpiN1Mvn0" 
GOOGLE_CREDENTIALS_SECRET_KEY = "google_service_account_credentials"

ENTITY_COLORS = ["#2C3E50", "#8B4513", "#556B2F", "#8B0000", "#483D8B", "#696969"] 
HIGHLIGHT_BACKGROUND_COLOR = "#F5F5F5" 

def initialize_session_state():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+Pro:ital,wght@0,400;0,600;1,400&display=swap');
        
        * {
            font-family: 'Source Serif Pro', 'Georgia', serif !important;
        }
        
        .main {
            padding-top: 2rem;
            background-color: #FAFAFA;
        }
        
        .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span {
            font-family: 'Source Serif Pro', 'Georgia', serif !important;
            font-size: 16px !important;
            line-height: 1.7 !important;
            color: #2C3E50 !important;
        }
        
        .stMarkdown h1 {
            color: #1A252F !important;
            font-family: 'Source Serif Pro', serif !important;
            border-bottom: 2px solid #34495E !important;
            padding-bottom: 8px !important;
            font-size: 28px !important;
        }
        
        .stMarkdown h2 {
            color: #2C3E50 !important;
            font-family: 'Source Serif Pro', serif !important;
            font-size: 22px !important;
        }
        
        .stMarkdown h3 {
            color: #34495E !important;
            font-family: 'Source Serif Pro', serif !important;
            font-size: 18px !important;
        }
        
        .stProgress .st-bo {
            background-color: #34495E !important;
        }
        
        .stProgress .st-bn {
            background-color: #E8E8E8 !important;
        }
        
        .stProgress > div > div > div {
            background-color: #34495E !important;
        }
        
        .stProgress > div {
            background-color: #E8E8E8 !important;
            border-radius: 2px !important;
        }
        
        div[data-testid="stProgress"] > div {
            background-color: #E8E8E8 !important;
            border-radius: 2px !important;
        }
        
        div[data-testid="stProgress"] > div > div {
            background-color: #34495E !important;
            border-radius: 2px !important;
        }
        
        .css-1d391kg {
            background-color: #2C3E50;
        }
        
        .academic-paper {
            background: white;
            padding: 25px;
            border-radius: 3px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin: 15px 0;
            font-family: 'Source Serif Pro', 'Georgia', serif !important;
            line-height: 1.7;
            border: 1px solid #E8E8E8;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #34495E 0%, #2C3E50 100%);
            padding: 15px;
            border-radius: 3px;
            color: white;
            text-align: center;
            margin: 8px 0;
        }
        
        .stButton > button {
            background-color: #34495E !important;
            color: white !important;
            border: 1px solid #2C3E50 !important;
            border-radius: 3px !important;
            font-family: 'Source Serif Pro', serif !important;
            font-size: 16px !important;
        }
        
        .stButton > button:hover {
            background-color: #2C3E50 !important;
            border-color: #1A252F !important;
        }
        
        .stSlider {
            font-family: 'Source Serif Pro', serif !important;
        }
        
        .stSlider label {
            font-family: 'Source Serif Pro', serif !important;
            color: #2C3E50 !important;
        }
        
        .stCheckbox {
            font-family: 'Source Serif Pro', serif !important;
        }
        
        .stCheckbox label {
            font-family: 'Source Serif Pro', serif !important;
            color: #2C3E50 !important;
        }
        
        .stSelectbox label {
            font-family: 'Source Serif Pro', serif !important;
            color: #2C3E50 !important;
        }
        
        .stTextInput label {
            font-family: 'Source Serif Pro', serif !important;
            color: #2C3E50 !important;
        }
        
        .stTextArea label {
            font-family: 'Source Serif Pro', serif !important;
            color: #2C3E50 !important;
        }
        
        .stNumberInput label {
            font-family: 'Source Serif Pro', serif !important;
            color: #2C3E50 !important;
        }
        
        .stDataFrame {
            font-family: 'Source Serif Pro', serif !important;
        }
        
        .stSuccess {
            font-family: 'Source Serif Pro', serif !important;
        }
        
        .stError {
            font-family: 'Source Serif Pro', serif !important;
        }
        
        .stWarning {
            font-family: 'Source Serif Pro', serif !important;
        }
        
        .stInfo {
            font-family: 'Source Serif Pro', serif !important;
        }
        
        div[data-testid="stMarkdownContainer"] {
            font-family: 'Source Serif Pro', serif !important;
        }
        
        div[data-testid="column"] {
            font-family: 'Source Serif Pro', serif !important;
        }
        
        div[data-testid="metric-container"] {
            font-family: 'Source Serif Pro', serif !important;
        }
        
        .element-container {
            font-family: 'Source Serif Pro', serif !important;
        }
        
        .block-container {
            font-family: 'Source Serif Pro', serif !important;
        }
    </style>
    """, unsafe_allow_html=True)

    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
    st.markdown(hide_menu_style, unsafe_allow_html=True)
    
    if 'consent_given' not in st.session_state: st.session_state.consent_given = False
    if 'consent_read_understood_all' not in st.session_state: st.session_state.consent_read_understood_all = False
    if 'consent_age_confirmed' not in st.session_state: st.session_state.consent_age_confirmed = False
    if 'consent_voluntary_participation' not in st.session_state: st.session_state.consent_voluntary_participation = False
    if 'consent_data_anonymity_use' not in st.session_state: st.session_state.consent_data_anonymity_use = False
    if 'sentences_data' not in st.session_state: st.session_state.sentences_data = []
    if 'packet_data' not in st.session_state: st.session_state.packet_data = []
    if 'shuffled_indices' not in st.session_state: st.session_state.shuffled_indices = []
    if 'current_question_index' not in st.session_state: st.session_state.current_question_index = 0
    if 'survey_complete' not in st.session_state: st.session_state.survey_complete = False
    if 'user_responses' not in st.session_state: st.session_state.user_responses = []
    if 'current_scores' not in st.session_state: st.session_state.current_scores = {}
    if 'seed' not in st.session_state: st.session_state.seed = None
    if 'packet_sentence_index' not in st.session_state: st.session_state.packet_sentence_index = 0
    if 'packet_sentiment_history' not in st.session_state: st.session_state.packet_sentiment_history = []

def display_consent_form():
    st.title("Research Survey Consent Form")
    st.markdown("""
<div class="academic-paper">

**Welcome to Our Research Study on Sentence Sentiment!**

Thank you for considering participating in our study. Your input is valuable. Please read the information below carefully.

**1. What is this study about?**
   - **Purpose:** We want to understand how people perceive sentiment (positive, negative, neutral feelings) in complex sentences.
   - **Goal:** We're looking at how sentence structure and word choices affect the sentiment you feel towards **highlighted words or phrases** (entities) in those sentences.
   - **Impact:** Your responses will help us build better computer models that can understand human language more accurately.

**2. What will I be asked to do?**
   - **Task:** You'll read 20 unique sentences. These sentences are generated by a computer program for this study.
   - **Action:** For each sentence, you will see some words or phrases highlighted. You'll then use a slider to rate the sentiment you feel is directed towards *each* highlighted entity.
   - **Time:** The survey should take about 10-15 minutes to complete.

**3. Is my participation voluntary?**
   - **Voluntary:** Yes, completely. You can choose to participate or not.
   - **Withdrawal:** You can stop at any time by simply closing your browser window. There's no penalty for withdrawing. If you withdraw, any data collected from your session up to that point will not be used.

**4. How will my data be handled?**
   - **Anonymity:** Your responses are anonymous. We do **not** collect any personally identifiable information (like your name, email, or IP address).
   - **Usage:** The anonymized data will be used for academic research. This might include publications in scientific journals, presentations at conferences, or sharing with other researchers to advance science.
   - **Data Sharing:** Anonymized data may be made publicly available in research datasets.

**5. Are there any risks or benefits?**
   - **Risks:** There are no significant risks expected. Some sentences might describe negative situations or use negative language – this is a necessary part of studying a full range of sentiments.
   - **Benefits:** There are no direct personal benefits for participating. However, your contribution is very important for advancing our understanding of language and improving technology.
   
**6. Who can I contact for more information?**
   - If you have any questions about this study, please contact:
     - **Harry Yin or USFCA MAGICS Lab** 
     - **harry.d.yin.gpc@gmail.com** 

---

**Please confirm your understanding and consent by checking all boxes below:**

</div>
    """, unsafe_allow_html=True)

    st.session_state.consent_read_understood_all = st.checkbox("I confirm that I have read and understood all the information provided above.", value=st.session_state.get('consent_read_understood_all', False))
    st.session_state.consent_age_confirmed = st.checkbox("I confirm that I am 18 years of age or older.", value=st.session_state.get('consent_age_confirmed', False))
    st.session_state.consent_voluntary_participation = st.checkbox("I understand that my participation is voluntary and I can withdraw at any time without penalty.", value=st.session_state.get('consent_voluntary_participation', False))
    st.session_state.consent_data_anonymity_use = st.checkbox("I consent to my anonymized data being used for research purposes, including publications, presentations, and potential public sharing for scientific advancement.", value=st.session_state.get('consent_data_anonymity_use', False))

    all_consents_given_now = (
        st.session_state.consent_read_understood_all and
        st.session_state.consent_age_confirmed and
        st.session_state.consent_voluntary_participation and
        st.session_state.consent_data_anonymity_use
    )

    if st.button("Start Survey", type="primary", use_container_width=True, disabled=not all_consents_given_now):
        st.session_state.consent_given = True
        question_data = survey_gen()
        st.session_state.seed = question_data.get('seed', None)
        st.session_state.sentences_data = question_data.get('items', [])
        st.session_state.packet_data = question_data.get('packets', [])
        
        all_data = st.session_state.sentences_data + st.session_state.packet_data
        
        if not all_data: 
            st.error("Failed to generate sentences. Please try again.")
            return
            
        st.session_state.shuffled_indices = list(range(len(all_data)))
        random.shuffle(st.session_state.shuffled_indices)
        st.session_state.current_question_index = 0
        st.session_state.survey_complete = False
        st.session_state.user_responses = []
        st.session_state.current_scores = {}
        st.session_state.packet_sentence_index = 0
        st.session_state.packet_sentiment_history = []
        st.rerun()

sentiment_scale = {1: "Extremely Negative", 2: "Negative", 3: "Slightly Negative", 4: "Neutral", 5: "Slightly Positive", 6: "Positive", 7: "Extremely Positive"}

def get_scorable_entities(item):
    if item.get('type') in ['compound_action', 'compound_association', 'compound_belonging']:
        return item.get('entities', [])
    elif item.get('type') in ['aggregate_short', 'aggregate_long']:
        return [item.get('entity', '')]
    return []

def get_entities_to_highlight(main_entity, text):
    entities_to_highlight = [main_entity]
    
    pronouns = ['it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves']
    
    for pronoun in pronouns:
        if re.search(r'\b' + re.escape(pronoun) + r'\b', text, re.IGNORECASE):
            entities_to_highlight.append(pronoun)
    
    return entities_to_highlight

def highlight_entities_in_sentence(original_text, entities):
    entity_color_map = {} 
    highlight_details = [] 

    sorted_entities = sorted(list(set(entities)), key=len, reverse=True)

    patterns_for_entities = []
    for i, entity in enumerate(sorted_entities):
        color = ENTITY_COLORS[i % len(ENTITY_COLORS)]
        entity_color_map[entity] = color
        patterns_for_entities.append(
            (re.compile(r'\b(' + re.escape(entity) + r')\b', re.IGNORECASE), entity, color)
        )
    
    all_potential_matches = []
    for pattern, entity, color in patterns_for_entities:
        for match in pattern.finditer(original_text):
            all_potential_matches.append({
                "start": match.start(1), "end": match.end(1), 
                "text": match.group(1), 
                "color": color, "entity": entity
            })

    all_potential_matches.sort(key=lambda m: (m["start"], -(m["end"] - m["start"])))

    last_highlight_end = -1
    for match_info in all_potential_matches:
        if match_info["start"] >= last_highlight_end:
            highlight_details.append(match_info)
            last_highlight_end = match_info["end"]
            if match_info["text"] != match_info["entity"] and match_info["text"] not in entity_color_map:
                entity_color_map[match_info["text"]] = match_info["color"]
    
    highlighted_sentence_parts = []
    current_pos = 0
    highlight_details.sort(key=lambda m: m["start"]) 

    for detail in highlight_details:
        if detail["start"] > current_pos:
            highlighted_sentence_parts.append(original_text[current_pos:detail["start"]])
        
        highlight_span = f"<span style='color:{detail['color']}; background-color:{HIGHLIGHT_BACKGROUND_COLOR}; padding:0.1em 0.2em; border-radius:2px; font-weight:600; font-family: \"Source Serif Pro\", serif;'>{detail['text']}</span>"
        highlighted_sentence_parts.append(highlight_span)
        current_pos = detail["end"]

    if current_pos < len(original_text):
        highlighted_sentence_parts.append(original_text[current_pos:])
    
    return "".join(highlighted_sentence_parts), entity_color_map

def display_packet_question(sentence_item, actual_sentence_index):
    item_id = f"{sentence_item.get('type', 'unknown')}_{actual_sentence_index}"
    sentences = sentence_item.get('sentences', [])
    entity = sentence_item.get('entity', '')
    
    current_sentence_idx = st.session_state.packet_sentence_index
    total_sentences = len(sentences)
    
    if current_sentence_idx >= total_sentences:
        st.session_state.packet_sentence_index = 0
        st.session_state.packet_sentiment_history = []
        return True
    
    cumulative_text = " ".join(sentences[:current_sentence_idx + 1])
    
    st.markdown(f"<h3 style='font-family: \"Source Serif Pro\", serif; color: #2C3E50;'>Step {current_sentence_idx + 1} of {total_sentences}:</h3>", unsafe_allow_html=True)
    
    entities_to_highlight = get_entities_to_highlight(entity, cumulative_text)
    highlighted_sentence_html, entity_color_map = highlight_entities_in_sentence(cumulative_text, entities_to_highlight)
    st.markdown(f"<div style='font-size: 17px; border: 1px solid #D5D5D5; padding: 20px; border-radius: 3px; margin-bottom:20px; line-height:1.7; background-color: white; font-family: \"Source Serif Pro\", serif; color: #2C3E50;'>{highlighted_sentence_html}</div>", unsafe_allow_html=True)
    
    if st.session_state.packet_sentiment_history:
        st.markdown("<h3 style='font-family: \"Source Serif Pro\", serif; color: #2C3E50;'>Your Previous Ratings:</h3>", unsafe_allow_html=True)
        for i, score in enumerate(st.session_state.packet_sentiment_history):
            st.markdown(f"<p style='font-family: \"Source Serif Pro\", serif; color: #2C3E50;'><strong>Step {i + 1}:</strong> {score} ({sentiment_scale[score]})</p>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='font-family: \"Source Serif Pro\", serif; color: #2C3E50;'>Current Rating:</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-family: \"Source Serif Pro\", serif; color: #2C3E50;'>Rate the sentiment you feel is <strong>directed towards</strong> the highlighted entity.</p>", unsafe_allow_html=True)
    
    display_color = entity_color_map.get(entity, ENTITY_COLORS[0])
    st.markdown(f"<p style='font-family: \"Source Serif Pro\", serif; color: #2C3E50;'>Sentiment towards: <span style='color:{display_color}; background-color:{HIGHLIGHT_BACKGROUND_COLOR}; padding:0.1em 0.2em; border-radius:2px; font-weight:600; font-family: \"Source Serif Pro\", serif;'>\"{entity}\"</span></p>", unsafe_allow_html=True)
    
    slider_key = f"packet_slider_{item_id}_{current_sentence_idx}"
    default_value = 4
    if st.session_state.packet_sentiment_history:
        default_value = st.session_state.packet_sentiment_history[-1]
    
    score = st.slider(
        label=f"Rate for \"{entity}\"", 
        min_value=1, 
        max_value=7, 
        value=default_value, 
        format="%d", 
        key=slider_key, 
        label_visibility="collapsed", 
        help=f"Rating scale: 1 ({sentiment_scale[1]}) to 7 ({sentiment_scale[7]})"
    )
    
    st.markdown(f"<p style='font-family: \"Source Serif Pro\", serif; color: #2C3E50;'>Your current rating: <strong>{score} ({sentiment_scale[score]})</strong></p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col2:
        button_text = "Next Step" if current_sentence_idx < total_sentences - 1 else "Complete Question"
        if st.button(button_text, key=f"next_btn_{item_id}_{current_sentence_idx}", type="primary"):
            st.session_state.packet_sentiment_history.append(score)
            
            if current_sentence_idx < total_sentences - 1:
                st.session_state.packet_sentence_index += 1
                st.rerun()
            else:
                response_data = {
                    'item_id': item_id,
                    'item_type': sentence_item.get('type', ''),
                    'description': sentence_item.get('description', ''),
                    'sentences': json.dumps(sentences),
                    'code_key': sentence_item.get('code_key', ''),
                    'entity': entity,
                    'sentiment_scores_list': json.dumps(st.session_state.packet_sentiment_history),
                    'final_sentiment_score': st.session_state.packet_sentiment_history[-1],
                    'final_sentiment_label': sentiment_scale[st.session_state.packet_sentiment_history[-1]],
                    'seed': st.session_state.seed
                }
                
                if 'descriptor' in sentence_item:
                    response_data['descriptor'] = json.dumps(sentence_item['descriptor'])
                if 'intensity' in sentence_item:
                    response_data['intensity'] = json.dumps(sentence_item['intensity'])
                
                st.session_state.user_responses.append(response_data)
                st.session_state.packet_sentence_index = 0
                st.session_state.packet_sentiment_history = []
                return True
    
    return False

def display_regular_question(sentence_item, actual_sentence_index):
    item_id = f"{sentence_item.get('type', 'unknown')}_{actual_sentence_index}"
    sentences = sentence_item.get('sentences', [])
    combined_text = " ".join(sentences)
    
    st.markdown("<h3 style='font-family: \"Source Serif Pro\", serif; color: #2C3E50;'>Sentence:</h3>", unsafe_allow_html=True)
    scorable_entities = get_scorable_entities(sentence_item)
    
    highlighted_sentence_html, entity_color_map = highlight_entities_in_sentence(combined_text, scorable_entities)
    st.markdown(f"<div style='font-size: 17px; border: 1px solid #D5D5D5; padding: 20px; border-radius: 3px; margin-bottom:20px; line-height:1.7; background-color: white; font-family: \"Source Serif Pro\", serif; color: #2C3E50;'>{highlighted_sentence_html}</div>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='font-family: \"Source Serif Pro\", serif; color: #2C3E50;'>Sentiment Scoring:</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-family: \"Source Serif Pro\", serif; color: #2C3E50;'>For each <strong>highlighted entity</strong>, rate the sentiment you feel is <strong>directed towards them</strong> in the sentence.</p>", unsafe_allow_html=True)

    if item_id not in st.session_state.current_scores: 
        st.session_state.current_scores[item_id] = {}

    for entity in scorable_entities:
        display_color = entity_color_map.get(entity, "#000000")

        slider_key = f"slider_{item_id}_{entity.replace(' ', '_').replace('\"','_')}"
        current_value = st.session_state.current_scores[item_id].get(entity, 4)

        st.markdown(f"<p style='font-family: \"Source Serif Pro\", serif; color: #2C3E50;'>Sentiment towards: <span style='color:{display_color}; background-color:{HIGHLIGHT_BACKGROUND_COLOR}; padding:0.1em 0.2em; border-radius:2px; font-weight:600; font-family: \"Source Serif Pro\", serif;'>\"{entity}\"</span></p>", unsafe_allow_html=True)
        
        score = st.slider(label=f"Rate for \"{entity}\"", min_value=1, max_value=7, value=current_value, format="%d", key=slider_key, label_visibility="collapsed", help=f"Rating scale: 1 ({sentiment_scale[1]}) to 7 ({sentiment_scale[7]})")
        
        st.session_state.current_scores[item_id][entity] = score
        
        st.markdown(f"<p style='font-family: \"Source Serif Pro\", serif; color: #2C3E50;'>Your rating: <strong>{score} ({sentiment_scale[score]})</strong></p>", unsafe_allow_html=True)
        st.markdown("---")
        
    if st.button("Next Question", key=f"next_btn_{item_id}", type="primary", use_container_width=True):
        for entity in scorable_entities:
            score_value = st.session_state.current_scores[item_id].get(entity)
            if score_value is not None:
                response_data = {
                    'item_id': item_id,
                    'item_type': sentence_item.get('type', ''),
                    'description': sentence_item.get('description', ''),
                    'sentences': json.dumps(sentences),
                    'combined_text': combined_text,
                    'code_key': sentence_item.get('code_key', ''),
                    'entity': entity,
                    'user_sentiment_score': score_value,
                    'user_sentiment_label': sentiment_scale[score_value],
                    'seed': st.session_state.seed
                }
                
                if 'descriptor' in sentence_item:
                    response_data['descriptor'] = json.dumps(sentence_item['descriptor'])
                if 'intensity' in sentence_item:
                    response_data['intensity'] = json.dumps(sentence_item['intensity'])
                if 'entities' in sentence_item:
                    response_data['all_entities'] = json.dumps(sentence_item['entities'])
                
                st.session_state.user_responses.append(response_data)
        
        return True
    
    return False

def display_question():
    current_q_idx = st.session_state.current_question_index
    all_data = st.session_state.sentences_data + st.session_state.packet_data
    total_questions = len(st.session_state.shuffled_indices) if st.session_state.shuffled_indices else 0

    if total_questions == 0: 
        st.error("No questions loaded. Please restart.")
        st.session_state.survey_complete = True
        st.rerun()
        return
    
    progress_value = (current_q_idx + 1) / total_questions if total_questions > 0 else 0
    st.markdown(f"<p style='font-family: \"Source Serif Pro\", serif; color: #2C3E50; text-align: center; margin-bottom: 10px;'><strong>Question {current_q_idx + 1} of {total_questions}</strong></p>", unsafe_allow_html=True)
    st.progress(progress_value)
    
    if not (0 <= current_q_idx < total_questions): 
        st.error("Question index out of bounds.")
        st.session_state.survey_complete = True
        st.rerun()
        return
        
    actual_sentence_index = st.session_state.shuffled_indices[current_q_idx]
    if not (0 <= actual_sentence_index < len(all_data)): 
        st.error("Sentence index out of bounds.")
        st.session_state.survey_complete = True
        st.rerun()
        return
        
    sentence_item = all_data[actual_sentence_index]
    
    is_packet_question = sentence_item.get('type') in ['aggregate_short', 'aggregate_long']
    
    if is_packet_question:
        question_complete = display_packet_question(sentence_item, actual_sentence_index)
    else:
        question_complete = display_regular_question(sentence_item, actual_sentence_index)
    
    if question_complete:
        next_q_idx = current_q_idx + 1
        if next_q_idx < total_questions: 
            st.session_state.current_question_index = next_q_idx
        else: 
            st.session_state.survey_complete = True
        st.session_state.current_scores = {} 
        st.rerun()

def export_to_google_sheets(data_df):
    creds_str = st.secrets.get(GOOGLE_CREDENTIALS_SECRET_KEY)
    if not creds_str: 
        st.error(f"Secret '{GOOGLE_CREDENTIALS_SECRET_KEY}' not found for Google Sheets export.")
        return
        
    try: 
        creds_dict = json.loads(creds_str)
    except json.JSONDecodeError: 
        st.error("Failed to parse Google credentials JSON for Sheets export.")
        return
    
    try:
        data_df = data_df.fillna('')
        data_df = data_df.replace([np.inf, -np.inf], '')
        
        for col in data_df.select_dtypes(include=[np.number]).columns:
            data_df[col] = data_df[col].astype(str)
        
        creds = Credentials.from_service_account_info(creds_dict, scopes=["https://www.googleapis.com/auth/spreadsheets"])
        client = gspread.authorize(creds)
        if GOOGLE_SHEET_ID == "YOUR_SPREADSHEET_ID_HERE" or not GOOGLE_SHEET_ID: 
            st.error("GOOGLE_SHEET_ID is not set or is a placeholder; cannot export to Google Sheets.")
            return
        
        spreadsheet = client.open_by_key(GOOGLE_SHEET_ID)
        ws_name = "Sheet1"
        ws = None
        try: 
            ws = spreadsheet.worksheet(ws_name)
        except gspread.WorksheetNotFound: 
            ws = spreadsheet.add_worksheet(title=ws_name, rows="1", cols=len(data_df.columns))
        
        headers = ws.row_values(1) 
        if not headers or all(h == "" for h in headers): 
            ws.update([data_df.columns.values.tolist()] + data_df.values.tolist())
        else: 
            ws.append_rows(data_df.values.tolist(), value_input_option='USER_ENTERED') 
        st.success(f"Data successfully exported to Google Sheet (ID: {GOOGLE_SHEET_ID})")
    except gspread.exceptions.SpreadsheetNotFound: 
        st.error(f"Google Sheet (ID: '{GOOGLE_SHEET_ID}') not found or not shared with the service account: {creds_dict.get('client_email', 'your service account email')}.")
    except Exception as e: 
        st.error(f"An error occurred during Google Sheets export: {e}")
        st.info(f"Please check your Google Sheet ID, sharing settings (share with {creds_dict.get('client_email', 'your service account email')}), and API permissions.")

def display_finish_screen():
    st.title("Survey Completed!")
    st.success("Thank you for your participation! Your responses are greatly appreciated.")
    
    st.markdown("""
    <div class="academic-paper">
    
    **Contact Information:**
    
    If you have any questions or concerns, please feel free to reach out to us at: 
    - harry.d.yin.gpc@gmail.com
    - USFCA MAGICS Lab
    - dgbrizan@usfca.edu

    ---
    
    **Looking at the data:**
    
    You can take a look at all the data we've collected so far at this Google Sheet link: [Google Sheet Link](https://docs.google.com/spreadsheets/d/1xAvDLhU0w-p2hAZ49QYM7-XBMQCek0zVYJWpiN1Mvn0/edit?usp=sharing)
    
    ---
    
    **Project:**
    
    You can take a look at the project page here: [Project Page](https://github.com/Xild076/ETSA--QC-)
    
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.user_responses:
        df = pd.DataFrame(st.session_state.user_responses)
        df['submission_timestamp_utc'] = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        
        current_user_login = "anonymous" 
        df['submitted_by_user_login'] = current_user_login

        cols = ['submitted_by_user_login', 'submission_timestamp_utc'] + [c for c in df.columns if c not in ['submitted_by_user_login', 'submission_timestamp_utc']]
        df = df[cols]

        if GOOGLE_SHEET_ID != "YOUR_SPREADSHEET_ID_HERE" and GOOGLE_SHEET_ID and st.secrets.get(GOOGLE_CREDENTIALS_SECRET_KEY):
            export_to_google_sheets(df)
        elif GOOGLE_SHEET_ID == "YOUR_SPREADSHEET_ID_HERE" or not GOOGLE_SHEET_ID:
            st.warning("Automatic Google Sheet Export is not configured by the admin (GOOGLE_SHEET_ID is missing or is a placeholder). Your data has not been automatically uploaded.")
        elif not st.secrets.get(GOOGLE_CREDENTIALS_SECRET_KEY):
            st.warning(f"Automatic Google Sheet Export is not configured by the admin (the '{GOOGLE_CREDENTIALS_SECRET_KEY}' secret is missing). Your data has not been automatically uploaded.")
        
        st.markdown("---")
        st.subheader("Your Data")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Survey Data (CSV)", data=csv, use_container_width=True,
            file_name=f"survey_data_{current_user_login}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
    else: 
        st.warning("No responses were recorded in this session.")
    st.markdown("---")

def main():
    st.set_page_config(page_title="Sentence Sentiment Survey", layout="centered", initial_sidebar_state="collapsed")
    initialize_session_state()
    if not st.session_state.consent_given: 
        display_consent_form()
    elif not st.session_state.survey_complete: 
        display_question()
    else: 
        display_finish_screen()

if __name__ == "__main__": 
    main()