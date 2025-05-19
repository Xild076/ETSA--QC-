import random
import numpy as np
import streamlit as st
import pandas as pd
import json
import gspread
from google.oauth2.service_account import Credentials
import datetime
import re

GOOGLE_SHEET_ID = "1xAvDLhU0w-p2hAZ49QYM7-XBMQCek0zVYJWpiN1Mvn0" 
GOOGLE_CREDENTIALS_SECRET_KEY = "google_service_account_credentials"

ENTITY_COLORS = ["#3E7CB1", "#D5A021", "#4A875F", "#A35D6A", "#5F4B8B", "#8E8E8E"] 
HIGHLIGHT_BACKGROUND_COLOR = "#f0f0f0" 

noun_phrases={
    "Very Negative":["the vile mass murderer","the sadistic serial killer","the abhorrent war criminal"],
    "Negative":["the hateful bully","the cruel landlord","the cruel manager"],
    "Somewhat Negative":["the annoying coworker","the irritable neighbor","the petty supervisor"],
    "Leaning Negative":["the occasionally untidy roommate","the occasionally slow assistant","the sometimes forgetful intern"],
    "Neutral":["the ordinary neighbor","the average student","the plain clerk"],
    "Leaning Positive":["the slightly pleasant colleague","the generally polite teammate","the somewhat cordial neighbor"],
    "Somewhat Positive":["the friendly mentor","the pleasant teammate","the supportive trainer"],
    "Positive":["the inspirational leader","the loving guardian","the joyful musician"],
    "Very Positive":["the incredibly awesome champion","the magnificent visionary","the absolutely miraculous savior"]
}

verbs={
    "Very Negative":["brutally assaulted","sadistically tortured"],
    "Negative":["terrorized","hated"],
    "Somewhat Negative":["criticized","undermined"],
    "Leaning Negative":["briefly ignored","mildly snubbed"],
    "Neutral":["observed","handled"],
    "Leaning Positive":["invited","warmly included"],
    "Somewhat Positive":["supported","helped"],
    "Positive":["celebrated","inspired"],
    "Very Positive":["enthusiastically championed","absolutely idolized"]
}

sentiment_categories_map = {
    "Positive": ["Leaning Positive", "Somewhat Positive", "Positive", "Very Positive"],
    "Negative": ["Very Negative", "Negative", "Somewhat Negative", "Leaning Negative"],
    "Neutral": ["Neutral"]
}

all_sentiment_groups_list = ["Positive", "Negative", "Neutral"]

length_3_combinations = [
    [False, False, False], [False, False, True], [False, True, False], [False, True, True],
    [True, False, False], [True, False, True], [True, True, False], [True, True, True]
]

length_4_combinations = [
    [False, False, False, False],[False, False, False, True],[False, False, True, False],[False, False, True, True],
    [False, True, False, False],[False, True, False, True],[False, True, True, False],[False, True, True, True],
    [True, False, False, False],[True, False, False, True],[True, False, True, False],[True, False, True, True],
    [True, True, False, False],[True, True, False, True],[True, True, True, False],[True, True, True, True]
]

length_5_combinations = [
    [False,False,False,False,False],[False,False,False,False,True],[False,False,False,True,False],[False,False,False,True,True],
    [False,False,True,False,False],[False,False,True,False,True],[False,False,True,True,False],[False,False,True,True,True],
    [False,True,False,False,False],[False,True,False,False,True],[False,True,False,True,False],[False,True,False,True,True],
    [False,True,True,False,False],[False,True,True,False,True],[False,True,True,True,False],[False,True,True,True,True],
    [True,False,False,False,False],[True,False,False,False,True],[True,False,False,True,False],[True,False,False,True,True],
    [True,False,True,False,False],[True,False,True,False,True],[True,False,True,True,False],[True,False,True,True,True],
    [True,True,False,False,False],[True,True,False,False,True],[True,True,False,True,False],[True,True,False,True,True],
    [True,True,True,False,False],[True,True,True,False,True],[True,True,True,True,False],[True,True,True,True,True]
]

def active_1a_1v(actor, verb, victim): return f"{actor} {verb} {victim}.".capitalize()
def active_1a_2v(actor, verb1, victim_1, victim_2): return f"{actor} {verb1} {victim_1} and {victim_2}.".capitalize()
def active_2a_1v(actor_1, actor_2, verb, victim): return f"{actor_1} and {actor_2} {verb} {victim}.".capitalize()
def active_2a_2v(actor_1, actor_2, verb, victim_1, victim_2): return f"{actor_1} and {actor_2} {verb} {victim_1} and {victim_2}.".capitalize()
def passive_1a_1v(actor, verb, victim): return f"{victim} was {verb} by {actor}.".capitalize()
def passive_1a_2v(actor, verb1, victim_1, victim_2): return f"{victim_1} and {victim_2} were {verb1} by {actor}.".capitalize()
def passive_2a_1v(actor_1, actor_2, verb, victim): return f"{victim} was {verb} by {actor_1} and {actor_2}.".capitalize()
def passive_2a_2v(actor_1, actor_2, verb, victim_1, victim_2): return f"{victim_1} and {victim_2} were {verb} by {actor_1} and {actor_2}.".capitalize()
def subj_acts_and_is_acted_upon_by_two(subj_phrase, active_verb_phrase, passive_verb_phrase, agent1_phrase, agent2_phrase): return f"{subj_phrase} {active_verb_phrase} and was also {passive_verb_phrase} by {agent1_phrase} and {agent2_phrase}.".capitalize()

def get_component_details(is_target_base_group, base_sentiment_group, component_source_dict, excluded_phrases=None):
    excluded_phrases = excluded_phrases or []
    if is_target_base_group: final_sentiment_group_for_component = base_sentiment_group
    else:
        if base_sentiment_group == "Positive": final_sentiment_group_for_component = "Negative" if np.random.choice(["Neg", "Neut"], p=[0.8,0.2]) == "Neg" else random.choice(["Positive","Negative"])
        elif base_sentiment_group == "Negative": final_sentiment_group_for_component = "Positive" if np.random.choice(["Pos", "Neut"], p=[0.8,0.2]) == "Pos" else random.choice(["Positive","Negative"])
        else: final_sentiment_group_for_component = random.choice(["Positive", "Negative"])
    granular_sentiment_key = random.choice(sentiment_categories_map[final_sentiment_group_for_component])
    available_phrases = [p for p in component_source_dict[granular_sentiment_key] if p not in excluded_phrases]
    if not available_phrases: available_phrases = component_source_dict[granular_sentiment_key]
    phrase = random.choice(available_phrases) if available_phrases else "default_phrase"
    return {'phrase': phrase, 'sentiment_category': granular_sentiment_key, 'sentiment_group': final_sentiment_group_for_component, 'is_aligned_with_base_group': is_target_base_group}

def generate_sentences_data():
    sentences_metadata = []
    num_1a1v = len(length_3_combinations) // 2
    for i in range(num_1a1v):
        base_sent = random.choice(all_sentiment_groups_list)
        actor1 = get_component_details(length_3_combinations[i][0], base_sent, noun_phrases)
        verb = get_component_details(length_3_combinations[i][1], base_sent, verbs)
        victim = get_component_details(length_3_combinations[i][2], base_sent, noun_phrases, [actor1['phrase']])
        s_id = f"s_{len(sentences_metadata)}"
        if random.choice([True, False]):
            txt = active_1a_1v(actor1['phrase'], verb['phrase'], victim['phrase']); t_type = 'active_1a_1v'; comps = [{'r': 'actor', **actor1}, {'r': 'v', **verb}, {'r': 'victim', **victim}]
        else:
            txt = passive_1a_1v(actor1['phrase'], verb['phrase'], victim['phrase']); t_type = 'passive_1a_1v'; comps = [{'r': 'victim', **victim}, {'r': 'v', **verb}, {'r': 'agent', **actor1}]
        sentences_metadata.append({'id': s_id, 'txt': txt, 'type': t_type, 'base': base_sent, 'comps': comps})

    num_complex = len(length_5_combinations) // 2; num_subj_acts = num_complex // 4; num_2a2v = num_complex - num_subj_acts
    complex_order = random.choice([True, False])
    if complex_order:
        for i in range(num_subj_acts):
            base_sent = random.choice(all_sentiment_groups_list); combo = length_5_combinations[i]
            subj = get_component_details(combo[0], base_sent, noun_phrases)
            act_v = get_component_details(combo[1], base_sent, verbs); pass_v = get_component_details(combo[2], base_sent, verbs)
            ag1 = get_component_details(combo[3], base_sent, noun_phrases, [subj['phrase']])
            ag2 = get_component_details(combo[4], base_sent, noun_phrases, [subj['phrase'], ag1['phrase']])
            s_id = f"s_{len(sentences_metadata)}"
            txt = subj_acts_and_is_acted_upon_by_two(subj['phrase'], act_v['phrase'], pass_v['phrase'], ag1['phrase'], ag2['phrase'])
            comps = [{'r': 'subject_actor_victim', **subj}, {'r': 'act_v', **act_v}, {'r': 'pass_v', **pass_v}, {'r': 'agent1', **ag1}, {'r': 'agent2', **ag2}]
            sentences_metadata.append({'id': s_id, 'txt': txt, 'type': 'subj_acts_and_is_acted_upon_by_two', 'base': base_sent, 'comps': comps})
        for i in range(num_2a2v):
            base_sent = random.choice(all_sentiment_groups_list); combo = length_5_combinations[i + num_subj_acts]
            a1 = get_component_details(combo[0], base_sent, noun_phrases); a2 = get_component_details(combo[1], base_sent, noun_phrases, [a1['phrase']])
            vrb = get_component_details(combo[2], base_sent, verbs)
            v1 = get_component_details(combo[3], base_sent, noun_phrases, [a1['phrase'], a2['phrase']])
            v2 = get_component_details(combo[4], base_sent, noun_phrases, [a1['phrase'], a2['phrase'], v1['phrase']])
            s_id = f"s_{len(sentences_metadata)}"
            if random.choice([True, False]):
                txt = active_2a_2v(a1['phrase'], a2['phrase'], vrb['phrase'], v1['phrase'], v2['phrase']); t_type = 'active_2a_2v'; comps = [{'r': 'actor1', **a1}, {'r': 'actor2', **a2}, {'r': 'v', **vrb}, {'r': 'victim1', **v1}, {'r': 'victim2', **v2}]
            else:
                txt = passive_2a_2v(a1['phrase'], a2['phrase'], vrb['phrase'], v1['phrase'], v2['phrase']); t_type = 'passive_2a_2v'; comps = [{'r': 'victim1', **v1}, {'r': 'victim2', **v2}, {'r': 'v', **vrb}, {'r': 'agent1', **a1}, {'r': 'agent2', **a2}]
            sentences_metadata.append({'id': s_id, 'txt': txt, 'type': t_type, 'base': base_sent, 'comps': comps})
    else:
        num_4_half = len(length_4_combinations) // 2
        for i in range(num_4_half):
            base_sent = random.choice(all_sentiment_groups_list); combo = length_4_combinations[i]
            a1 = get_component_details(combo[0], base_sent, noun_phrases); a2 = get_component_details(combo[1], base_sent, noun_phrases, [a1['phrase']])
            vrb = get_component_details(combo[2], base_sent, verbs); vict = get_component_details(combo[3], base_sent, noun_phrases, [a1['phrase'],a2['phrase']])
            s_id = f"s_{len(sentences_metadata)}"
            if random.choice([True, False]):
                txt = active_2a_1v(a1['phrase'], a2['phrase'], vrb['phrase'], vict['phrase']); t_type = 'active_2a_1v'; comps = [{'r': 'actor1', **a1}, {'r': 'actor2', **a2}, {'r': 'v', **vrb}, {'r': 'victim', **vict}]
            else:
                txt = passive_2a_1v(a1['phrase'], a2['phrase'], vrb['phrase'], vict['phrase']); t_type = 'passive_2a_1v'; comps = [{'r': 'victim', **vict}, {'r': 'v', **vrb}, {'r': 'agent1', **a1}, {'r': 'agent2', **a2}]
            sentences_metadata.append({'id': s_id, 'txt': txt, 'type': t_type, 'base': base_sent, 'comps': comps})
        for i in range(num_4_half):
            base_sent = random.choice(all_sentiment_groups_list); combo = length_4_combinations[i + num_4_half]
            act = get_component_details(combo[0], base_sent, noun_phrases); vrb = get_component_details(combo[1], base_sent, verbs)
            v1 = get_component_details(combo[2], base_sent, noun_phrases, [act['phrase']]); v2 = get_component_details(combo[3], base_sent, noun_phrases, [act['phrase'],v1['phrase']])
            s_id = f"s_{len(sentences_metadata)}"
            if random.choice([True, False]):
                txt = active_1a_2v(act['phrase'], vrb['phrase'], v1['phrase'], v2['phrase']); t_type = 'active_1a_2v'; comps = [{'r': 'actor', **act}, {'r': 'v', **vrb}, {'r': 'victim1', **v1}, {'r': 'victim2', **v2}]
            else:
                txt = passive_1a_2v(act['phrase'], vrb['phrase'], v1['phrase'], v2['phrase']); t_type = 'passive_1a_2v'; comps = [{'r': 'victim1', **v1}, {'r': 'victim2', **v2}, {'r': 'v', **vrb}, {'r': 'agent', **act}]
            sentences_metadata.append({'id': s_id, 'txt': txt, 'type': t_type, 'base': base_sent, 'comps': comps})
    return sentences_metadata[:20]

def initialize_session_state():
    if 'consent_given' not in st.session_state: st.session_state.consent_given = False
    if 'consent_read_understood_all' not in st.session_state: st.session_state.consent_read_understood_all = False
    if 'consent_age_confirmed' not in st.session_state: st.session_state.consent_age_confirmed = False
    if 'consent_voluntary_participation' not in st.session_state: st.session_state.consent_voluntary_participation = False
    if 'consent_data_anonymity_use' not in st.session_state: st.session_state.consent_data_anonymity_use = False
    if 'sentences_data' not in st.session_state: st.session_state.sentences_data = []
    if 'shuffled_indices' not in st.session_state: st.session_state.shuffled_indices = []
    if 'current_question_index' not in st.session_state: st.session_state.current_question_index = 0
    if 'survey_complete' not in st.session_state: st.session_state.survey_complete = False
    if 'user_responses' not in st.session_state: st.session_state.user_responses = []
    if 'current_scores' not in st.session_state: st.session_state.current_scores = {}

def display_consent_form():
    st.title("Research Survey Consent Form")
    st.markdown("""
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
     - **Harry Yin or USF MAGICS Lab** 
     - **harry.d.yin.gpc@gmail.com** 

---
**Please confirm your understanding and consent by checking all boxes below:**
    """)

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
        st.session_state.sentences_data = generate_sentences_data()
        if not st.session_state.sentences_data: 
            st.error("Failed to generate sentences. Please try again.")
            return
        st.session_state.shuffled_indices = list(range(len(st.session_state.sentences_data)))
        random.shuffle(st.session_state.shuffled_indices)
        st.session_state.current_question_index = 0
        st.session_state.survey_complete = False
        st.session_state.user_responses = []
        st.session_state.current_scores = {}
        st.rerun()

sentiment_scale = {1: "Extremely Negative", 2: "Negative", 3: "Slightly Negative", 4: "Neutral", 5: "Slightly Positive", 6: "Positive", 7: "Extremely Positive"}

def get_scorable_components_and_phrases(components_list):
    scorable_entities = []
    canonical_phrases_seen = set()
    for comp in components_list:
        role = comp.get('r', '')
        canonical_phrase = comp.get('phrase', '')
        if role not in ['v', 'act_v', 'pass_v'] and canonical_phrase and canonical_phrase not in canonical_phrases_seen:
            scorable_entities.append(comp) 
            canonical_phrases_seen.add(canonical_phrase)
    return scorable_entities

def highlight_entities_in_sentence(original_text, scorable_canonical_phrases):
    entity_color_map = {} 
    highlight_details = [] 

    sorted_canonical_phrases = sorted(list(set(scorable_canonical_phrases)), key=len, reverse=True)

    patterns_for_canonical = []
    for i, canonical_phrase in enumerate(sorted_canonical_phrases):
        color = ENTITY_COLORS[i % len(ENTITY_COLORS)]
        entity_color_map[canonical_phrase] = color
        patterns_for_canonical.append(
            (re.compile(r'(?<!\w)(' + re.escape(canonical_phrase) + r')(?!\w)', re.IGNORECASE), canonical_phrase, color)
        )
    
    all_potential_matches = []
    for pattern, canonical_phrase, color in patterns_for_canonical:
        for match in pattern.finditer(original_text):
            all_potential_matches.append({
                "start": match.start(1), "end": match.end(1), 
                "text": match.group(1), 
                "color": color, "canonical_phrase": canonical_phrase
            })

    all_potential_matches.sort(key=lambda m: (m["start"], -(m["end"] - m["start"])))

    last_highlight_end = -1
    for match_info in all_potential_matches:
        if match_info["start"] >= last_highlight_end:
            highlight_details.append(match_info)
            last_highlight_end = match_info["end"]
            if match_info["text"] != match_info["canonical_phrase"] and match_info["text"] not in entity_color_map:
                entity_color_map[match_info["text"]] = match_info["color"]
    
    highlighted_sentence_parts = []
    current_pos = 0
    highlight_details.sort(key=lambda m: m["start"]) 

    for detail in highlight_details:
        if detail["start"] > current_pos:
            highlighted_sentence_parts.append(original_text[current_pos:detail["start"]])
        
        highlight_span = f"<span style='color:{detail['color']}; background-color:{HIGHLIGHT_BACKGROUND_COLOR}; padding:0.1em 0.3em; border-radius:0.3em; font-weight:bold;'>{detail['text']}</span>"
        highlighted_sentence_parts.append(highlight_span)
        current_pos = detail["end"]

    if current_pos < len(original_text):
        highlighted_sentence_parts.append(original_text[current_pos:])
    
    return "".join(highlighted_sentence_parts), entity_color_map


def display_question():
    current_q_idx = st.session_state.current_question_index
    total_questions = len(st.session_state.shuffled_indices) if st.session_state.shuffled_indices else 0

    if total_questions == 0: st.error("No questions loaded. Please restart."); st.session_state.survey_complete = True; st.rerun(); return
    
    progress_value = (current_q_idx + 1) / total_questions if total_questions > 0 else 0
    st.progress(progress_value, text=f"Question {current_q_idx + 1} of {total_questions}")
    
    if not (0 <= current_q_idx < total_questions): st.error("Question index out of bounds."); st.session_state.survey_complete = True; st.rerun(); return
    actual_sentence_index = st.session_state.shuffled_indices[current_q_idx]
    if not (0 <= actual_sentence_index < len(st.session_state.sentences_data)): st.error("Sentence index out of bounds."); st.session_state.survey_complete = True; st.rerun(); return
    sentence_item = st.session_state.sentences_data[actual_sentence_index]
    if not (isinstance(sentence_item, dict) and 'id' in sentence_item and 'txt' in sentence_item and 'comps' in sentence_item): 
        st.error("Invalid sentence data structure."); st.session_state.survey_complete = True; st.rerun(); return
        
    sentence_id = sentence_item['id']
    original_sentence_text = sentence_item['txt']
    components = sentence_item['comps']
    
    st.subheader("Sentence:")
    scorable_components = get_scorable_components_and_phrases(components)
    scorable_canonical_phrases = [comp['phrase'] for comp in scorable_components]
    
    highlighted_sentence_html, entity_color_map = highlight_entities_in_sentence(original_sentence_text, scorable_canonical_phrases)
    st.markdown(f"<div style='font-size: 1.25rem; border: 1px solid #ddd; padding: 1rem; border-radius: 0.3rem; margin-bottom:1.5rem; line-height:1.8;'>{highlighted_sentence_html}</div>", unsafe_allow_html=True)
    
    st.subheader("Sentiment Scoring:")
    st.markdown("For each **highlighted entity**, rate the sentiment you feel is **directed towards them** in the sentence.")

    if sentence_id not in st.session_state.current_scores: st.session_state.current_scores[sentence_id] = {}

    for comp_to_score in scorable_components:
        canonical_phrase = comp_to_score['phrase']
        display_color = entity_color_map.get(canonical_phrase, entity_color_map.get(canonical_phrase.lower(), entity_color_map.get(canonical_phrase.capitalize(), "#000000")))

        slider_key = f"slider_{sentence_id}_{canonical_phrase.replace(' ', '_').replace('\"','_')}"
        current_value = st.session_state.current_scores[sentence_id].get(canonical_phrase, 4)

        st.markdown(f"Sentiment towards: <span style='color:{display_color}; background-color:{HIGHLIGHT_BACKGROUND_COLOR}; padding:0.1em 0.3em; border-radius:0.3em; font-weight:bold;'>\"{canonical_phrase}\"</span>", unsafe_allow_html=True)
        
        score = st.slider(label=f"Rate for \"{canonical_phrase}\"", min_value=1, max_value=7, value=current_value, format="%d", key=slider_key, label_visibility="collapsed", help=f"Rating scale: 1 ({sentiment_scale[1]}) to 7 ({sentiment_scale[7]})")
        
        st.session_state.current_scores[sentence_id][canonical_phrase] = score
        
        st.markdown(f"Your rating: **{score} ({sentiment_scale[score]})**")
        st.markdown("---")
        
    if st.button("Next Sentence", key=f"next_btn_{sentence_id}", type="primary", use_container_width=True):
        for comp_data in scorable_components:
            phrase_to_log = comp_data['phrase']
            score_value = st.session_state.current_scores[sentence_id].get(phrase_to_log)
            if score_value is not None:
                 st.session_state.user_responses.append({
                    'sentence_id': sentence_id, 'sentence_text': original_sentence_text,
                    'sentence_template_type': sentence_item.get('type'), 'base_sentiment_group': sentence_item.get('base'),
                    'component_role_internal': comp_data.get('r'), 'component_phrase': phrase_to_log,
                    'component_sentiment_cat': comp_data.get('sentiment_category'), 'component_sentiment_grp': comp_data.get('sentiment_group'),
                    'component_aligned_base': comp_data.get('is_aligned_with_base_group'),
                    'user_sentiment_score': score_value, 'user_sentiment_label': sentiment_scale[score_value]
                })
        
        next_q_idx = current_q_idx + 1
        if next_q_idx < total_questions: st.session_state.current_question_index = next_q_idx
        else: st.session_state.survey_complete = True
        st.session_state.current_scores = {} 
        st.rerun()

def export_to_google_sheets(data_df):
    creds_str = st.secrets.get(GOOGLE_CREDENTIALS_SECRET_KEY)
    if not creds_str: st.error(f"Secret '{GOOGLE_CREDENTIALS_SECRET_KEY}' not found for Google Sheets export."); return
    try: creds_dict = json.loads(creds_str)
    except json.JSONDecodeError: st.error("Failed to parse Google credentials JSON for Sheets export."); return
    
    try:
        creds = Credentials.from_service_account_info(creds_dict, scopes=["https://www.googleapis.com/auth/spreadsheets"])
        client = gspread.authorize(creds)
        if GOOGLE_SHEET_ID == "YOUR_SPREADSHEET_ID_HERE" or not GOOGLE_SHEET_ID: 
            st.error("GOOGLE_SHEET_ID is not set or is a placeholder; cannot export to Google Sheets.")
            return
        
        spreadsheet = client.open_by_key(GOOGLE_SHEET_ID)
        ws_name = "Sheet1"; ws = None
        try: ws = spreadsheet.worksheet(ws_name)
        except gspread.WorksheetNotFound: ws = spreadsheet.add_worksheet(title=ws_name, rows="1", cols=len(data_df.columns))
        
        headers = ws.row_values(1) 
        if not headers or all(h == "" for h in headers) : ws.update([data_df.columns.values.tolist()] + data_df.values.tolist())
        else: ws.append_rows(data_df.values.tolist(), value_input_option='USER_ENTERED') 
        st.success(f"Data successfully exported to Google Sheet (ID: {GOOGLE_SHEET_ID})")
    except gspread.exceptions.SpreadsheetNotFound: st.error(f"Google Sheet (ID: '{GOOGLE_SHEET_ID}') not found or not shared with the service account: {creds_dict.get('client_email', 'your service account email')}.")
    except Exception as e: st.error(f"An error occurred during Google Sheets export: {e}"); st.info(f"Please check your Google Sheet ID, sharing settings (share with {creds_dict.get('client_email', 'your service account email')}), and API permissions.")

def display_finish_screen():
    st.title("🎉 Survey Completed! 🎉")
    st.balloons(); st.success("Thank you for your participation! Your responses are greatly appreciated.")

    if st.session_state.user_responses:
        df = pd.DataFrame(st.session_state.user_responses)
        df['submission_timestamp_utc'] = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        
        current_user_login = "Xild076" 
        df['submitted_by_user_login'] = current_user_login

        cols = ['submitted_by_user_login', 'submission_timestamp_utc'] + [c for c in df.columns if c not in ['submitted_by_user_login', 'submission_timestamp_utc']]
        df = df[cols]

        if GOOGLE_SHEET_ID != "YOUR_SPREADSHEET_ID_HERE" and GOOGLE_SHEET_ID and st.secrets.get(GOOGLE_CREDENTIALS_SECRET_KEY):
            export_to_google_sheets(df)
        elif GOOGLE_SHEET_ID == "YOUR_SPREADSHEET_ID_HERE" or not GOOGLE_SHEET_ID:
            st.warning("Automatic Google Sheet Export is not configured by the admin (GOOGLE_SHEET_ID is missing or is a placeholder). Your data has not been automatically uploaded.")
        elif not st.secrets.get(GOOGLE_CREDENTIALS_SECRET_KEY):
            st.warning(f"Automatic Google Sheet Export is not configured by the admin (the '{GOOGLE_CREDENTIALS_SECRET_KEY}' secret is missing). Your data has not been automatically uploaded.")
        
        st.markdown("---"); st.subheader("Your Data")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Survey Data (CSV)", data=csv, use_container_width=True,
            file_name=f"survey_data_{current_user_login}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
    else: st.warning("No responses were recorded in this session.")
    st.markdown("---")

def main():
    st.set_page_config(page_title="Sentence Sentiment Survey", layout="centered", initial_sidebar_state="collapsed")
    initialize_session_state()
    if not st.session_state.consent_given: display_consent_form()
    elif not st.session_state.survey_complete: display_question()
    else: display_finish_screen()

if __name__ == "__main__": 
    main()