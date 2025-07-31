import spacy
from transformers import pipeline

def find_entity_and_get_spacy_tokens(sentence, entity_name):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)

    ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    entities = ner_pipeline(sentence)

    target_entity_info = None
    for entity in entities:
        if entity['word'].strip().lower() == entity_name.strip().lower():
            target_entity_info = entity
            break

    if target_entity_info:
        start_char = target_entity_info['start']
        end_char = target_entity_info['end']

        span = doc.char_span(start_char, end_char, label=target_entity_info['entity_group'])
        return span

    return None

sentence = "John beat up the child."
entity_to_find = "John"

entity_span = find_entity_and_get_spacy_tokens(sentence, entity_to_find)

if entity_span:
    print(f"Found entity: '{entity_span.text}'")
    print(f"Associated spaCy tokens (Text, ID): {[(token.text, token.i) for token in entity_span]}")
    print(f"Entity label: {entity_span.label_}")
    print(f"Root token of the span: {entity_span.root.text}")
