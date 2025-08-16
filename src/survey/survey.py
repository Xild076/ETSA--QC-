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
from survey_question_gen import calibration_gen

def display_calibration_questions():
    st.markdown("""
    <div class="academic-paper">
    <h3>Practice Questions</h3>
    <p>Before you begin the main survey, please answer two quick practice questions. This will help you get familiar with the rating scale. Each practice question shows a single phrase. Rate the sentiment the phrase conveys, from -4 (Extremely Negative) to 4 (Extremely Positive).</p>
    </div>
    """, unsafe_allow_html=True)
    pos_calibration = st.session_state.get('calibration_questions', {}).get('positive', {})
    neg_calibration = st.session_state.get('calibration_questions', {}).get('negative', {})
    pos_text = pos_calibration.get('word') or pos_calibration.get('text') or ''
    st.markdown(f"<div style='margin-top:1.5em; margin-bottom:0.5em;'><b>Example 1:</b> <i>{pos_text}</i></div>", unsafe_allow_html=True)
    pos_score = st.slider(
        label="How would you rate the sentiment of this phrase?",
        min_value=-4, max_value=4, value=0, format="%d", key="calib_pos_slider",
        help="Rating scale: -4 (Extremely Negative) to 4 (Extremely Positive)",
        step=1
    )
    slider_view()
    st.markdown(f"You rated the sentiment as: <b>{pos_score} ({sentiment_scale[pos_score]})</b>", unsafe_allow_html=True)

    neg_text = neg_calibration.get('word') or neg_calibration.get('text') or ''
    st.markdown(f"<div style='margin-top:2em; margin-bottom:0.5em;'><b>Example 2:</b> <i>{neg_text}</i></div>", unsafe_allow_html=True)
    neg_score = st.slider(
        label="How would you rate the sentiment of this phrase?",
        min_value=-4, max_value=4, value=0, format="%d", key="calib_neg_slider",
        help="Rating scale: -4 (Extremely Negative) to 4 (Extremely Positive)",
        step=1
    )
    slider_view()
    st.markdown(f"You rated the sentiment as: <b>{neg_score} ({sentiment_scale[neg_score]})</b>", unsafe_allow_html=True)
    if st.button("Continue", key="calib_continue_btn", type="primary", use_container_width=True):
        st.session_state['calibration_pos_score'] = pos_score
        st.session_state['calibration_neg_score'] = neg_score
        st.session_state['calibration_complete'] = True
        st.rerun()
    st.stop()

def display_calibration_confirmation():
    pos_score = st.session_state.get('calibration_pos_score', None)
    neg_score = st.session_state.get('calibration_neg_score', None)
    pos_text = st.session_state.get('calibration_questions', {}).get('positive', {}).get('word') or st.session_state.get('calibration_questions', {}).get('positive', {}).get('text', '')
    neg_text = st.session_state.get('calibration_questions', {}).get('negative', {}).get('word') or st.session_state.get('calibration_questions', {}).get('negative', {}).get('text', '')
    if isinstance(pos_text, (list, tuple)):
        pos_text = random.choice(pos_text)
    if isinstance(neg_text, (list, tuple)):
        neg_text = random.choice(neg_text)
    st.markdown(f"""
    <div class="academic-paper">
    <h3>Practice Confirmation</h3>
    <p>You said the sentiment conveyed about <b>'{pos_text}'</b> is <b>{pos_score} ({sentiment_scale.get(pos_score, pos_score)})</b>.</p>
    <p>You said the sentiment conveyed about <b>'{neg_text}'</b> is <b>{neg_score} ({sentiment_scale.get(neg_score, neg_score)})</b>.</p>
    <p>Does this seem right?</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Yes, continue to survey", key="calib_confirm_final_btn", type="primary", use_container_width=True):
        st.session_state['calibration_confirmed'] = True
        calibration = st.session_state.get('calibration_questions', {})
        pos_meta = calibration.get('positive', {})
        neg_meta = calibration.get('negative', {})
        pos_row = {
            'item_id': 'calibration_positive',
            'item_type': 'calibration',
            'description': 'practice positive',
            'sentences': json.dumps([pos_text]),
            'combined_text': pos_text,
            'code_key': pos_meta.get('code_key', ''),
            'entity': pos_text,
            'seed': st.session_state.get('seed', ''),
            'descriptor': json.dumps(['positive']),
            'intensity': json.dumps([pos_meta.get('intensity', '')]),
            'all_entities': json.dumps([pos_text]),
            'packet_step': '',
            'user_sentiment_score': pos_score,
            'user_sentiment_label': sentiment_scale.get(pos_score, pos_score),
            'sentence_at_step': '',
            'new_sentence_for_step': '',
            'descriptor_for_step': '',
            'intensity_for_step': '',
            'mark': '',
            'marks': ''
        }
        neg_row = {
            'item_id': 'calibration_negative',
            'item_type': 'calibration',
            'description': 'practice negative',
            'sentences': json.dumps([neg_text]),
            'combined_text': neg_text,
            'code_key': neg_meta.get('code_key', ''),
            'entity': neg_text,
            'seed': st.session_state.get('seed', ''),
            'descriptor': json.dumps(['negative']),
            'intensity': json.dumps([neg_meta.get('intensity', '')]),
            'all_entities': json.dumps([neg_text]),
            'packet_step': '',
            'user_sentiment_score': neg_score,
            'user_sentiment_label': sentiment_scale.get(neg_score, neg_score),
            'sentence_at_step': '',
            'new_sentence_for_step': '',
            'descriptor_for_step': '',
            'intensity_for_step': '',
            'mark': '',
            'marks': ''
        }
        st.session_state.user_responses.append(pos_row)
        st.session_state.user_responses.append(neg_row)
        st.rerun()
    if st.button("No, redo practice questions", key="calib_redo_btn", use_container_width=True):
        st.session_state['calibration_complete'] = False
        st.session_state.pop('calib_pos_idx', None)
        st.session_state.pop('calib_neg_idx', None)
        st.rerun()
    st.stop()

def slider_view():
    st.markdown("""
        <style>
        .slider-scale {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            font-family: "Source Serif Pro", serif;
            font-size: 13px;
            margin: 0.2em 0 1.2em 0;
            border-top: 1px solid #DDD;
            padding-top: 0.5em;
        }
        .scale-item {
            text-align: center;
            flex: 1;
            min-width: 0;
        }
        .scale-number {
            font-weight: 600;
            font-size: 13px;
        }
        .scale-label {
            font-size: 10px;
            line-height: 1.1;
            margin-top: 2px;
        }
        .scale-separator {
            border-left: 1px solid #AAA;
            height: 32px;
            margin: 0 0.1em;
        }
        @media (max-width: 768px) {
            .scale-item {
                min-width: 20px;
            }
            .scale-label {
                font-size: 8px;
            }
            .scale-separator {
                margin: 0 0.05em;
            }
        }
        @media (max-width: 480px) {
            .scale-item {
                min-width: 15px;
            }
            .scale-label {
                font-size: 7px;
            }
            .scale-separator {
                margin: 0 0.02em;
                height: 24px;
            }
        }
        </style>
        <div class='slider-scale'>
            <div class='scale-item'>
                <div class='scale-number'>-4</div>
                <div class='scale-label'>
                    <span class='desktop-label'>Extremely<br>Negative</span>
                    <span class='mobile-label'>Ext.<br>Neg.</span>
                </div>
            </div>
            <div class='scale-separator'></div>
            <div class='scale-item'>
                <div class='scale-number'>-3</div>
                <div class='scale-label'>
                    <span class='desktop-label'>Negative</span>
                    <span class='mobile-label'>Neg.</span>
                </div>
            </div>
            <div class='scale-separator'></div>
            <div class='scale-item'>
                <div class='scale-number'>-2</div>
                <div class='scale-label'>
                    <span class='desktop-label'>Somewhat<br>Negative</span>
                    <span class='mobile-label'>Som.<br>Neg.</span>
                </div>
            </div>
            <div class='scale-separator'></div>
            <div class='scale-item'>
                <div class='scale-number'>-1</div>
                <div class='scale-label'>
                    <span class='desktop-label'>Slightly<br>Negative</span>
                    <span class='mobile-label'>Sl.<br>Neg.</span>
                </div>
            </div>
            <div class='scale-separator'></div>
            <div class='scale-item'>
                <div class='scale-number'>0</div>
                <div class='scale-label'>
                    <span class='desktop-label'>Neutral</span>
                    <span class='mobile-label'>Neu.</span>
                </div>
            </div>
            <div class='scale-separator'></div>
            <div class='scale-item'>
                <div class='scale-number'>1</div>
                <div class='scale-label'>
                    <span class='desktop-label'>Slightly<br>Positive</span>
                    <span class='mobile-label'>Sl.<br>Pos.</span>
                </div>
            </div>
            <div class='scale-separator'></div>
            <div class='scale-item'>
                <div class='scale-number'>2</div>
                <div class='scale-label'>
                    <span class='desktop-label'>Somewhat<br>Positive</span>
                    <span class='mobile-label'>Som.<br>Pos.</span>
                </div>
            </div>
            <div class='scale-separator'></div>
            <div class='scale-item'>
                <div class='scale-number'>3</div>
                <div class='scale-label'>
                    <span class='desktop-label'>Positive</span>
                    <span class='mobile-label'>Pos.</span>
                </div>
            </div>
            <div class='scale-separator'></div>
            <div class='scale-item'>
                <div class='scale-number'>4</div>
                <div class='scale-label'>
                    <span class='desktop-label'>Extremely<br>Positive</span>
                    <span class='mobile-label'>Ext.<br>Pos.</span>
                </div>
            </div>
        </div>
        <style>
        @media (min-width: 481px) {
            .mobile-label { display: none; }
        }
        @media (max-width: 480px) {
            .desktop-label { display: none; }
        }
        </style>
        """, unsafe_allow_html=True)

GOOGLE_SHEET_ID = "1xAvDLhU0w-p2hAZ49QYM7-XBMQCek0zVYJWpiN1Mvn0"
GOOGLE_CREDENTIALS_SECRET_KEY = "google_service_account_credentials"

ENTITY_COLORS = ["#2C3E50", "#8B4513", "#556B2F", "#8B0000", "#483D8B", "#696969"]
HIGHLIGHT_BACKGROUND_COLOR = "#F5F5F5"

def initialize_session_state():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+Pro:ital,wght@0,400;0,600;1,400&display=swap');
        * { font-family: 'Source Serif Pro', 'Georgia', serif !important; }
        .main { padding-top: 2rem; }
        .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span { font-family: 'Source Serif Pro', 'Georgia', serif !important; font-size: 16px !important; line-height: 1.7 !important; }
        .stMarkdown h1 { font-family: 'Source Serif Pro', serif !important; padding-bottom: 8px !important; font-size: 28px !important; }
        .stMarkdown h2 { font-family: 'Source Serif Pro', serif !important; font-size: 22px !important; }
        .stMarkdown h3 { font-family: 'Source Serif Pro', serif !important; font-size: 18px !important; }
        .stProgress .st-bo, .stProgress > div > div > div, div[data-testid="stProgress"] > div > div { background-color: #34495E !important; }
        .stProgress > div, div[data-testid="stProgress"] > div { border-radius: 2px !important; }
        div[data-testid="stProgress"] > div > div { border-radius: 2px !important; }
        .css-1d391kg { background-color: #2C3E50; }
        .academic-paper { padding: 25px; border-radius: 3px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin: 15px 0; font-family: 'Source Serif Pro', 'Georgia', serif !important; line-height: 1.7; }
        .metric-card { background: linear-gradient(135deg, #34495E 0%, #2C3E50 100%); padding: 15px; border-radius: 3px; color: white; text-align: center; margin: 8px 0; }
        .stButton > button { background-color: #34495E !important; color: white !important; border: 1px solid #2C3E50 !important; border-radius: 3px !important; font-family: 'Source Serif Pro', serif !important; font-size: 16px !important; }
        .stButton > button:hover { background-color: #2C3E50 !important; border-color: #1A252F !important; }
        .stButton > button:disabled { background-color: #CCCCCC !important; color: #666666 !important; border: 1px solid #AAAAAA !important; cursor: not-allowed !important; }
        .stButton > button:disabled:hover { background-color: #CCCCCC !important; border-color: #AAAAAA !important; }
        .stSlider, .stCheckbox, .stSelectbox, .stTextInput, .stTextArea, .stNumberInput { font-family: 'Source Serif Pro', serif !important; }
        .stSlider label, .stCheckbox label, .stSelectbox label, .stTextInput label, .stTextArea label, .stNumberInput label { font-family: 'Source Serif Pro', serif !important; }
        .stDataFrame, .stSuccess, .stError, .stWarning, .stInfo { font-family: 'Source Serif Pro', serif !important; }
        div[data-testid="stMarkdownContainer"], div[data-testid="column"], div[data-testid="metric-container"], .element-container, .block-container { font-family: 'Source Serif Pro', serif !important; }
        body[data-theme="light"] .main { background-color: #FAFAFA; }
        body[data-theme="light"] .stMarkdown, body[data-theme="light"] .stMarkdown p, body[data-theme="light"] .stMarkdown div, body[data-theme="light"] .stMarkdown span { color: #2C3E50 !important; }
        body[data-theme="light"] .stMarkdown h1 { color: #1A252F !important; border-bottom: 2px solid #34495E !important; }
        body[data-theme="light"] .stMarkdown h2 { color: #2C3E50 !important; }
        body[data-theme="light"] .stMarkdown h3 { color: #34495E !important; }
        body[data-theme="light"] .stProgress .st-bn, body[data-theme="light"] .stProgress > div, body[data-theme="light"] div[data-testid="stProgress"] > div { background-color: #E8E8E8 !important; }
        body[data-theme="light"] .academic-paper { background: white; border: 1px solid #E8E8E8; color: #2C3E50; }
        body[data-theme="light"] .stSlider label, body[data-theme="light"] .stCheckbox label, body[data-theme="light"] .stSelectbox label, body[data-theme="light"] .stTextInput label, body[data-theme="light"] .stTextArea label, body[data-theme="light"] .stNumberInput label { color: #2C3E50 !important; }
        body[data-theme="light"] .text-content { background-color: white !important; color: #2C3E50 !important; }
        body[data-theme="light"] .stButton > button:disabled { background-color: #CCCCCC !important; color: #666666 !important; border: 1px solid #AAAAAA !important; }
        body[data-theme="light"] .stButton > button:disabled:hover { background-color: #CCCCCC !important; border-color: #AAAAAA !important; }
        body[data-theme="dark"] .main { background-color: #0E1117; }
        body[data-theme="dark"] .stMarkdown, body[data-theme="dark"] .stMarkdown p, body[data-theme="dark"] .stMarkdown div, body[data-theme="dark"] .stMarkdown span { color: #FAFAFA !important; }
        body[data-theme="dark"] .stMarkdown h1 { color: #FFFFFF !important; border-bottom: 2px solid #FAFAFA !important; }
        body[data-theme="dark"] .stMarkdown h2 { color: #FAFAFA !important; }
        body[data-theme="dark"] .stMarkdown h3 { color: #FAFAFA !important; }
        body[data-theme="dark"] .stProgress .st-bn, body[data-theme="dark"] .stProgress > div, body[data-theme="dark"] div[data-testid="stProgress"] > div { background-color: #262730 !important; }
        body[data-theme="dark"] .academic-paper { background: #262730; border: 1px solid #404040; color: #FAFAFA; }
        body[data-theme="dark"] .stSlider label, body[data-theme="dark"] .stCheckbox label, body[data-theme="dark"] .stSelectbox label, body[data-theme="dark"] .stTextInput label, body[data-theme="dark"] .stTextArea label, body[data-theme="dark"] .stNumberInput label { color: #FAFAFA !important; }
        body[data-theme="dark"] .text-content { background-color: #262730 !important; color: #FAFAFA !important; }
        body[data-theme="dark"] .stButton > button:disabled { background-color: #404040 !important; color: #888888 !important; border: 1px solid #555555 !important; }
        body[data-theme="dark"] .stButton > button:disabled:hover { background-color: #404040 !important; border-color: #555555 !important; }
        body[data-theme="dark"] * { color: #FAFAFA !important; }
        body[data-theme="dark"] .stForm, body[data-theme="dark"] .stFormSubmitButton, body[data-theme="dark"] .stCheckbox div, body[data-theme="dark"] .stCheckbox span { color: #FAFAFA !important; }
        body[data-theme="dark"] .stWarning, body[data-theme="dark"] .stSuccess, body[data-theme="dark"] .stError, body[data-theme="dark"] .stInfo { color: #FAFAFA !important; }
        body[data-theme="dark"] .stAlert { color: #FAFAFA !important; background-color: #262730 !important; }
        body[data-theme="dark"] p, body[data-theme="dark"] div, body[data-theme="dark"] span, body[data-theme="dark"] label, body[data-theme="dark"] strong, body[data-theme="dark"] b, body[data-theme="dark"] i, body[data-theme="dark"] em { color: #FAFAFA !important; }
    </style>
    <script>
        function detectTheme() {
            const isDark = window.getComputedStyle(document.body).backgroundColor === 'rgb(14, 17, 23)' ||
                          document.querySelector('[data-testid="stAppViewContainer"]')?.style.backgroundColor === 'rgb(14, 17, 23)' ||
                          document.querySelector('.main')?.style.backgroundColor === 'rgb(14, 17, 23)';
            document.body.setAttribute('data-theme', isDark ? 'dark' : 'light');
        }
        detectTheme();
        const observer = new MutationObserver(detectTheme);
        observer.observe(document.body, { attributes: true, attributeFilter: ['style', 'class'] });
        setTimeout(detectTheme, 100); setTimeout(detectTheme, 500); setTimeout(detectTheme, 1000);
    </script>
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
    if 'seed' not in st.session_state: st.session_state.seed = None
    if 'packet_sentence_index' not in st.session_state: st.session_state.packet_sentence_index = 0
    if 'packet_sentiment_history' not in st.session_state: st.session_state.packet_sentiment_history = []
    if 'attention_check_shown' not in st.session_state: st.session_state.attention_check_shown = False
    if 'attention_check_passed' not in st.session_state: st.session_state.attention_check_passed = True

def display_attention_check():
    st.markdown("""
    <div class=\"academic-paper\">
    <h3 style='font-family: \"Source Serif Pro\", serif;'>Attention Check</h3>
    <p style='font-size: 17px;'>
    To ensure data quality, please select <b>Somewhat Positive (2)</b> as your answer for this question.
    </p>
    </div>
    """, unsafe_allow_html=True)
    score = st.slider(label="Please select Somewhat Positive (2)", min_value=-4, max_value=4, value=0, format="%d", key="attention_check_slider", label_visibility="collapsed")
    slider_view()
    st.markdown(f"<p style='font-family: \"Source Serif Pro\", serif;'>Your answer: <strong>{score} ({sentiment_scale[score]})</strong></p>", unsafe_allow_html=True)
    if st.button("Next", key="attention_check_next_btn", type="primary", use_container_width=True):
        st.session_state.attention_check_shown = True
        st.session_state.attention_check_passed = (score == 2)
        st.rerun()
    return False

def display_consent_form():
    st.title("Research Survey Consent Form")
    st.markdown("""
<div class="academic-paper">

Estimated time to complete: 10-15 minutes

We thank you for participating in a research study titled "Quantum Criticism—Entity-Targeted Sentiment Analysis". We will describe this study to you and answer any of your questions. This study is being led by Harry Yin, a research student currently associated with MAGICS Lab at USFCA. The Faculty Advisor for this study is Associate Professor David Guy Brizan, Department of CS at USFCA.

---

**Purpose:**

The purpose of this research is to collect ground truth data on peoples' emotional response towards entities or characters within a text, depending on factors such as how the text is formatted, the intensity of the language, and how the entities/characters interact with each other. The ultimate goal is to use this data in order to develop tools to quantify this phenomenon.

**What we will ask you to do:**

We will ask you to, given a text, to read the text and give each highlighted entity/character within the text a rating on a 9-point scale from -4 (Extremely Negative) to 4 (Extremely Positive) depending on how you feel about the entity/character. There will be 30 questions in the survey and should take at most 10 minutes to complete.

    """, unsafe_allow_html=True)

    with st.expander("More Information", expanded=False):
        st.markdown("""
<div class="academic-paper">

**Risks and discomforts:**

We do not anticipate any significant risks from participating in this research. The given sample text may describe mildly intense situations, but they are brief and fictional. We believe that the risk of discomfort is minimal.

**Benefits:**

We do not anticipate any direct benefits to the participant.

However, we hope to learn more clearly about the way text and textual formatting shapes sentiment. This information may benefit other people now or in the future by mitigating the effects of manipulative language online.

**Incentives for participation:**

There are no incentives for participation.

**Privacy/Confidentiality/Data Security:**

The responses are anonymous. We do not collect any personally identifiable information (ex: name, email, or IP address). There is an optional field at the end of the survey to provide your email address if you wish to receive a summary of the results, but this is not required. If you choose to provide your email, it will be stored separately from your responses and will not be linked to your survey data.

Please note that the survey is being hosted by Streamlit, a company not affiliated with USFCA and with its own privacy and security policies that you can find at its website, https://streamlit.io/. We anticipate that your participation in this survey presents no greater risk than everyday use of the Internet.

**Sharing Data Collected in this Research:**

Data from this study may be shared with the research community at large in the form of a paper, presentation, or dataset to advance science and health. We will remove or code any personal information that could identify you before files are shared with other researchers to ensure that, by current scientific standards and known methods, no one will be able to identify you from the information we share. Despite these measures, we cannot guarantee anonymity of your personal data.

**Taking part is voluntary:**

Participant involvement is completely voluntary. If at any moment the participant may feel uncomfortable with this survey, the participant may withdraw by closing the brower window. If the participant withdraws, any data collected up to that point will not be used.

**If you have questions:**

The main researcher conducting this study is Harry Yin, a student currently associated with the MAGICS Lab at USFCA. If you have questions, you may contact Harry Yin at harry.d.yin.gpc@gmail.com. You may also contact the MAGICS Lab at USFCA or the Faculty Advisor, Associate Professor David Guy Brizan, at dgbrizan@usfca.edu.

        </div>
        """, unsafe_allow_html=True)
    st.markdown("""
<div class="academic-paper">

Please check the following boxes to consent to this survey.

</div>
    """, unsafe_allow_html=True)

    st.session_state.consent_read_understood_all = st.checkbox("I confirm that I have read and understood all the information provided above.", value=st.session_state.get('consent_read_understood_all', False))
    st.session_state.consent_age_confirmed = st.checkbox("I confirm that I am 18 years of age or older.", value=st.session_state.get('consent_age_confirmed', False))
    st.session_state.consent_voluntary_participation = st.checkbox("I understand that my participation is voluntary and I can withdraw at any time without penalty.", value=st.session_state.get('consent_voluntary_participation', False))
    st.session_state.consent_data_anonymity_use = st.checkbox("I consent to my anonymized data being used for research purposes, including publications, presentations, and potential public sharing for scientific advancement.", value=st.session_state.get('consent_data_anonymity_use', False))

    all_consents_given_now = (st.session_state.consent_read_understood_all and st.session_state.consent_age_confirmed and st.session_state.consent_voluntary_participation and st.session_state.consent_data_anonymity_use)

    if st.button("Start Survey", type="primary", use_container_width=True, disabled=not all_consents_given_now):
        st.session_state.consent_given = True
        question_data = survey_gen()
        st.session_state.seed = question_data.get('seed', None)
        st.session_state.calibration_questions = question_data.get('calibration', {})
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
        st.session_state.packet_sentence_index = 0
        st.session_state.packet_sentiment_history = []
        st.rerun()

sentiment_scale = {-4: "Extremely Negative", -3: "Negative", -2: "Somewhat Negative", -1: "Slightly Negative", 0: "Neutral", 1: "Slightly Positive", 2: "Somewhat Positive", 3: "Positive", 4: "Extremely Positive"}

def get_scorable_entities(item):
    if item.get('type') in ['compound_action', 'compound_association', 'compound_belonging']:
        return item.get('entities', [])
    elif item.get('type') in ['aggregate_short', 'aggregate_medium', 'aggregate_long']:
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
        patterns_for_entities.append((re.compile(r'\b(' + re.escape(entity) + r')\b', re.IGNORECASE), entity, color))
    all_potential_matches = []
    for pattern, entity, color in patterns_for_entities:
        for match in pattern.finditer(original_text):
            all_potential_matches.append({"start": match.start(1), "end": match.end(1), "text": match.group(1), "color": color, "entity": entity})
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
        highlight_span = f"<span style='color:{detail['color']}; background-color:rgba(245,245,245,0.8); padding:0.1em 0.2em; border-radius:2px; font-weight:600; font-family: \"Source Serif Pro\", serif;'>{detail['text']}</span>"
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
    st.markdown(f"<h3 style='font-family: \"Source Serif Pro\", serif;'>Step {current_sentence_idx + 1} of {total_sentences}:</h3>", unsafe_allow_html=True)
    entities_to_highlight = get_entities_to_highlight(entity, cumulative_text)
    highlighted_sentence_html, entity_color_map = highlight_entities_in_sentence(cumulative_text, entities_to_highlight)
    st.markdown(f"<div class='text-content' style='font-size: 17px; border: 1px solid #D5D5D5; padding: 20px; border-radius: 3px; margin-bottom:20px; line-height:1.7; font-family: \"Source Serif Pro\", serif;'>{highlighted_sentence_html}</div>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-family: \"Source Serif Pro\", serif;'>Current Rating:</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-family: \"Source Serif Pro\", serif;'>Evaluate the sentiment conveyed about the entity in the sentence.</p>", unsafe_allow_html=True)
    display_color = entity_color_map.get(entity, ENTITY_COLORS[0])
    st.markdown(f"<p style='font-family: \"Source Serif Pro\", serif;'>Sentiment towards: <span style='color:{display_color}; background-color:rgba(245,245,245,0.8); padding:0.1em 0.2em; border-radius:2px; font-weight:600; font-family: \"Source Serif Pro\", serif;'>\"{entity}\"</span></p>", unsafe_allow_html=True)
    slider_key = f"packet_slider_{item_id}_{current_sentence_idx}"
    st.markdown(f"<p style='font-family: \"Source Serif Pro\", serif; color: {display_color}; background-color:rgba(245,245,245,0.8); padding:0.1em 0.2em; border-radius:2px; font-weight:600; margin-bottom: 0.5em;'>Entity: \"{entity}\"</p>", unsafe_allow_html=True)
    score = st.slider(
        label=f"Rate for \"{entity}\"",
        min_value=-4,
        max_value=4,
        value=0,
        format="%d",
        key=slider_key,
        label_visibility="collapsed",
        help=f"Rating scale: -4 ({sentiment_scale[-4]}) to 4 ({sentiment_scale[4]})",
        step=1,
    )
    slider_view()
    st.markdown(f"<p style='font-family: \"Source Serif Pro\", serif;'>Your current rating: <strong>{score} ({sentiment_scale[score]})</strong></p>", unsafe_allow_html=True)
    col_left, col_center, col_right = st.columns([1,2,1])
    confirm_key = f"confirm_packet_{item_id}_{current_sentence_idx}"
    if confirm_key not in st.session_state:
        st.session_state[confirm_key] = False
    with col_center:
        button_text = "Next Step" if current_sentence_idx < total_sentences - 1 else "Complete Question"
        if st.button(button_text, key=f"next_btn_{item_id}_{current_sentence_idx}", type="primary", use_container_width=True):
            if score == 0 and not st.session_state[confirm_key]:
                st.session_state[confirm_key] = True
                st.warning("Are you sure you want to continue without changing the default value? Click again to confirm.")
                st.stop()
            st.session_state[confirm_key] = False
            st.session_state.packet_sentiment_history.append(score)
            if current_sentence_idx < total_sentences - 1:
                st.session_state.packet_sentence_index += 1
                st.rerun()
            else:
                for i, recorded_score in enumerate(st.session_state.packet_sentiment_history):
                    response_data = {'item_id': item_id, 'item_type': sentence_item.get('type', ''), 'description': sentence_item.get('description', ''), 'sentences': json.dumps(sentences), 'combined_text': " ".join(sentences), 'code_key': sentence_item.get('code_key', ''), 'entity': entity, 'seed': st.session_state.seed, 'descriptor': json.dumps(sentence_item.get('descriptor', [])), 'intensity': json.dumps(sentence_item.get('intensity', [])), 'all_entities': json.dumps([entity]), 'packet_step': i + 1, 'user_sentiment_score': recorded_score, 'user_sentiment_label': sentiment_scale[recorded_score], 'sentence_at_step': " ".join(sentences[:i + 1]), 'new_sentence_for_step': sentences[i], 'descriptor_for_step': sentence_item.get('descriptor', [])[i] if i < len(sentence_item.get('descriptor', [])) else '', 'intensity_for_step': sentence_item.get('intensity', [])[i] if i < len(sentence_item.get('intensity', [])) else '', 'mark': sentence_item.get('marks', [])[i] if i < len(sentence_item.get('marks', [])) else '', 'marks': json.dumps(sentence_item.get('marks', []))}
                    st.session_state.user_responses.append(response_data)
                st.session_state.packet_sentence_index = 0
                st.session_state.packet_sentiment_history = []
                return True
    return False

def display_regular_question(sentence_item, actual_sentence_index):
    item_id = f"{sentence_item.get('type', 'unknown')}_{actual_sentence_index}"
    sentences = sentence_item.get('sentences', [])
    combined_text = " ".join(sentences)
    st.markdown("<h3 style='font-family: \"Source Serif Pro\", serif;'>Sentence:</h3>", unsafe_allow_html=True)
    scorable_entities = get_scorable_entities(sentence_item)
    highlighted_sentence_html, entity_color_map = highlight_entities_in_sentence(combined_text, scorable_entities)
    st.markdown(f"<div class='text-content' style='font-size: 17px; border: 1px solid #D5D5D5; padding: 20px; border-radius: 3px; margin-bottom:20px; line-height:1.7; font-family: \"Source Serif Pro\", serif;'>{highlighted_sentence_html}</div>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-family: \"Source Serif Pro\", serif;'>Sentiment Scoring:</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-family: \"Source Serif Pro\", serif;'>For each highlighted entity, evaluate the sentiment conveyed about it in the sentence.</p>", unsafe_allow_html=True)

    confirm_key = f"confirm_regular_{item_id}"
    if confirm_key not in st.session_state:
        st.session_state[confirm_key] = False
    slider_scores = {}
    for entity in scorable_entities:
        display_color = entity_color_map.get(entity, "#000000")
        slider_key = f"slider_{item_id}_{entity.replace(' ', '_').replace('"','_')}"
        st.markdown(f"<p style='font-family: \"Source Serif Pro\", serif; color: {display_color}; background-color:rgba(245,245,245,0.8); padding:0.1em 0.2em; border-radius:2px; font-weight:600; margin-bottom: 0.5em;'>Entity: \"{entity}\"</p>", unsafe_allow_html=True)
        score = st.slider(label=f"Rate for \"{entity}\"", min_value=-4, max_value=4, value=0, format="%d", key=slider_key, label_visibility="collapsed", help=f"Rating scale: -4 ({sentiment_scale[-4]}) to 4 ({sentiment_scale[4]})")
        slider_view()
        slider_scores[entity] = score
        st.markdown(f"<p style='font-family: \"Source Serif Pro\", serif;'>Your rating: <strong>{score} ({sentiment_scale[score]})</strong></p>", unsafe_allow_html=True)
        st.markdown("---")

    if st.button("Next Question", key=f"next_btn_{item_id}", type="primary", use_container_width=True):
        all_default = all(s == 0 for s in slider_scores.values())
        if all_default and not st.session_state[confirm_key]:
            st.session_state[confirm_key] = True
            st.warning("Are you sure you want to continue without changing the default value(s)? Click again to confirm.")
            st.stop()
        st.session_state[confirm_key] = False
        for entity in scorable_entities:
            slider_key = f"slider_{item_id}_{entity.replace(' ', '_').replace('\"','_')}"
            score_value = st.session_state[slider_key]
            if score_value is not None:
                response_data = {'item_id': item_id, 'item_type': sentence_item.get('type', ''), 'description': sentence_item.get('description', ''), 'sentences': json.dumps(sentences), 'combined_text': combined_text, 'code_key': sentence_item.get('code_key', ''), 'entity': entity, 'user_sentiment_score': score_value, 'user_sentiment_label': sentiment_scale[score_value], 'seed': st.session_state.seed, 'descriptor': json.dumps(sentence_item.get('descriptor', [])), 'intensity': json.dumps(sentence_item.get('intensity', [])), 'all_entities': json.dumps(sentence_item.get('entities', [])), 'packet_step': '', 'sentence_at_step': '', 'new_sentence_for_step': '', 'descriptor_for_step': '', 'intensity_for_step': '', 'marks': '', 'mark': ''}
                st.session_state.user_responses.append(response_data)
        return True
    return False

def display_question():
    current_q_idx = st.session_state.current_question_index
    all_data = st.session_state.sentences_data + st.session_state.packet_data
    total_questions = len(st.session_state.shuffled_indices) if st.session_state.shuffled_indices else 0

    midpoint = total_questions // 2 if total_questions > 0 else 0
    is_attention_check = (
        not st.session_state.attention_check_shown and
        current_q_idx == midpoint and
        total_questions > 3
    )

    progress_total = total_questions + (1 if not st.session_state.attention_check_shown else 0)
    st.markdown(f"<p style='font-family: \"Source Serif Pro\", serif; text-align: center; margin-bottom: 10px;'><strong>Question {current_q_idx + 1} of {progress_total}</strong></p>", unsafe_allow_html=True)
    st.progress((current_q_idx + 1) / progress_total if progress_total > 0 else 0)

    if is_attention_check:
        display_attention_check()
        return

    if total_questions == 0:
        st.error("No questions loaded. Please restart.")
        st.session_state.survey_complete = True
        st.rerun()
        return

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
    is_packet_question = sentence_item.get('type') in ['aggregate_short', 'aggregate_medium', 'aggregate_long']

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
        creds = Credentials.from_service_account_info(creds_dict, scopes=["https://www.googleapis.com/auth/spreadsheets"])
        client = gspread.authorize(creds)
        if GOOGLE_SHEET_ID == "YOUR_SPREADSHEET_ID_HERE" or not GOOGLE_SHEET_ID:
            st.error("GOOGLE_SHEET_ID is not set or is a placeholder; cannot export to Google Sheets.")
            return
        spreadsheet = client.open_by_key(GOOGLE_SHEET_ID)
        ws_name = "Sheet1"
        try:
            ws = spreadsheet.worksheet(ws_name)
        except gspread.WorksheetNotFound:
            ws = spreadsheet.add_worksheet(title=ws_name, rows=1000, cols=len(data_df.columns))
        expected_columns = ['submitted_by_user_login', 'submission_timestamp_utc', 'demographic_age', 'demographic_country', 'item_id', 'item_type', 'description', 'sentences', 'combined_text', 'code_key', 'entity', 'seed', 'descriptor', 'intensity', 'all_entities', 'packet_step', 'user_sentiment_score', 'user_sentiment_label', 'sentence_at_step', 'new_sentence_for_step', 'descriptor_for_step', 'intensity_for_step', 'mark', 'marks']
        df_ordered = data_df.reindex(columns=expected_columns, fill_value='')
        for col in df_ordered.columns:
            df_ordered[col] = df_ordered[col].astype(str).replace(['nan', 'None'], '')
        existing_data = ws.get_all_values()
        if not existing_data or not any(existing_data[0]):
            header_row = expected_columns
            data_rows = df_ordered.values.tolist()
            all_rows = [header_row] + data_rows
            ws.update('A1', all_rows, value_input_option='USER_ENTERED')
        else:
            data_rows = df_ordered.values.tolist()
            next_row = len(existing_data) + 1
            ws.update(f'A{next_row}', data_rows, value_input_option='USER_ENTERED')
        st.success(f"Data successfully exported to Google Sheet")
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"Google Sheet not found or not shared with the service account: {creds_dict.get('client_email', 'your service account email')}.")
    except Exception as e:
        st.error(f"An error occurred during Google Sheets export: {e}")

def display_demographic_section():
    st.markdown("""
    <div class="academic-paper" style="background:#fffbe6; border:1px solid #ffe58f;">
    <h3>Optional Demographic Questions</h3>
    <p><b>Warning:</b> The following information is personal and will be recorded and linked to your survey responses if you choose to provide it. However, your responses will remain anonymous and will not be linked to your identity. You may skip this section if you prefer by <i>leaving the fields blank</i>.</p>
    </div>
    """, unsafe_allow_html=True)
    with st.form("demographic_form", clear_on_submit=False):
        age = st.text_input("What is your age? (Optional)", value=st.session_state.get('demographic_age', ""))
        country = st.text_input("What country do you currently reside in? (If US, you may specify state)", value=st.session_state.get('demographic_country', ""))
        submitted = st.form_submit_button("Submit Demographics (Optional)")
        if submitted:
            st.session_state['demographic_age'] = age
            st.session_state['demographic_country'] = country
            st.session_state['demographic_complete'] = True
            st.success("Demographic information saved.")
            st.rerun()

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

**Project:**

You can take a look at the project page here: [Project Page](https://github.com/Xild076/ETSA--QC-)

</div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Your Data")
    if st.session_state.user_responses:
        df = pd.DataFrame(st.session_state.user_responses)
        df['submission_timestamp_utc'] = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        df['submitted_by_user_login'] = "anonymous"
        demographic_age = st.session_state.get('demographic_age', "")
        demographic_country = st.session_state.get('demographic_country', "")
        df['demographic_age'] = demographic_age
        df['demographic_country'] = demographic_country
        if (
            st.session_state.get('attention_check_passed', True)
            and GOOGLE_SHEET_ID != "YOUR_SPREADSHEET_ID_HERE"
            and GOOGLE_SHEET_ID
            and st.secrets.get(GOOGLE_CREDENTIALS_SECRET_KEY)
        ):
            export_to_google_sheets(df)
        elif GOOGLE_SHEET_ID == "YOUR_SPREADSHEET_ID_HERE" or not GOOGLE_SHEET_ID:
            st.warning("Automatic Google Sheet Export is not configured by the admin (GOOGLE_SHEET_ID is missing or is a placeholder). Your data has not been automatically uploaded.")
        elif not st.secrets.get(GOOGLE_CREDENTIALS_SECRET_KEY):
            st.warning(f"Automatic Google Sheet Export is not configured by the admin (the '{GOOGLE_CREDENTIALS_SECRET_KEY}' secret is missing). Your data has not been automatically uploaded.")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Survey Data (CSV)", data=csv, use_container_width=True, file_name=f"survey_data_anonymous_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
    else:
        st.warning("No responses were recorded in this session.")
    st.markdown("---")
    return

def main():
    st.set_page_config(page_title="Sentence Sentiment Survey", layout="centered", initial_sidebar_state="collapsed")
    initialize_session_state()
    if not st.session_state.consent_given:
        display_consent_form()
    elif not st.session_state.get('calibration_complete', False):
        display_calibration_questions()
    elif not st.session_state.get('calibration_confirmed', False):
        display_calibration_confirmation()
    elif not st.session_state.survey_complete:
        display_question()
    elif not st.session_state.get('demographic_complete', False):
        display_demographic_section()
    else:
        display_finish_screen()

if __name__ == "__main__":
    main()
