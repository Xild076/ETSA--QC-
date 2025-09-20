import spacy
from src.pipeline.modifier_e import SpacyModifierExtractor

nlp = spacy.load("en_core_web_lg")
text = 'The coffee is OUTSTANDING and the service was slow'
doc = nlp(text)

print("Doc tokens:")
for i, token in enumerate(doc):
    print(f"{i}: '{token.text}' (pos: {token.pos_}, dep: {token.dep_}, head: {token.head.text})")

print("\nTesting entity token finding...")

# Test the entity finding logic
def find_entity_token_debug(doc, entity_text):
    entity_tokens = []
    ent_words = entity_text.lower().split()
    
    # First try to find exact multiword spans
    entity_text_clean = entity_text.lower().strip()
    doc_text_lower = doc.text.lower()
    print(f"Looking for '{entity_text_clean}' in '{doc_text_lower}'")
    
    if entity_text_clean in doc_text_lower:
        start_char = doc_text_lower.find(entity_text_clean)
        end_char = start_char + len(entity_text_clean)
        print(f"Found at char positions {start_char}-{end_char}")
        
        for token in doc:
            if (token.idx >= start_char and 
                token.idx + len(token.text) <= end_char):
                entity_tokens.append(token)
                print(f"  Added token: '{token.text}' at {token.idx}")
    
    # Fallback to individual word matching, but only include substantive words
    if not entity_tokens:
        print("Falling back to individual word matching")
        substantive_words = [word for word in ent_words if word not in {"the", "a", "an", "this", "that"}]
        print(f"Substantive words: {substantive_words}")
        for i, token in enumerate(doc):
            if token.text.lower() in substantive_words:
                entity_tokens.append(token)
                print(f"  Added token: '{token.text}'")
    
    return entity_tokens

print("\nFor 'The coffee':")
tokens1 = find_entity_token_debug(doc, "The coffee")
print(f"Entity tokens: {[t.text for t in tokens1]}")

print("\nFor 'the service':")
tokens2 = find_entity_token_debug(doc, "the service")
print(f"Entity tokens: {[t.text for t in tokens2]}")
