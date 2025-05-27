import spacy

def analyze_text_for_ner_and_coreference(text_to_analyze):
    """
    Performs Named Entity Recognition (NER) and Coreference Resolution
    on the given text using SpaCy and Coreferee.
    """
    # Load the SpaCy language model
    # Using en_core_web_lg for better accuracy. Make sure you've downloaded it:
    # python -m spacy download en_core_web_lg
    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError:
        print(
            "Missing SpaCy model. Please run: python -m spacy download en_core_web_lg"
        )
        return

    # Add Coreferee to the SpaCy pipeline
    # This step assumes coreferee has been installed (pip install coreferee)
    # and its models are available for the SpaCy model in use.
    if "coreferee" not in nlp.pipe_names:
        try:
            # coreferee.add_to_pipe(nlp) # An alternative way to add
            nlp.add_pipe("coreferee")
            print("Added Coreferee to the pipeline.")
        except Exception as e:
            print(f"Could not add Coreferee to the pipeline: {e}")
            print("Coreference resolution will be skipped.")
            print("Ensure 'coreferee' is installed: pip install coreferee")
    
    # Process the text
    doc = nlp(text_to_analyze)

    print("\n--- Named Entities (NER) ---")
    if doc.ents:
        for ent in doc.ents:
            print(f"- Entity: '{ent.text}', Label: {ent.label_}")
    else:
        print("No named entities found.")

    print("\n--- Coreference Resolution (with Coreferee) ---")
    if "coreferee" in nlp.pipe_names and hasattr(doc._, 'coref_chains') and doc._.coref_chains:
        for chain_idx, chain in enumerate(doc._.coref_chains): # chain is a coreferee.data_model.Chain
            print(f"\nChain {chain_idx + 1}:")
            mention_texts = []
            for mention in chain.mentions: # mention is a coreferee.data_model.Mention
                # Get the span of text for this mention
                start_token = min(mention.token_indexes)
                end_token = max(mention.token_indexes) + 1 # +1 for exclusive end in slicing
                mention_span = doc[start_token:end_token]
                mention_texts.append(f"'{mention_span.text}' (tokens {start_token}-{end_token-1})")
            print("Mentions referring to the same entity: " + " | ".join(mention_texts))
    elif "coreferee" not in nlp.pipe_names:
        print("Coreferee component was not added to the pipeline. Skipping coreference.")
    else:
        print("No coreference chains found by Coreferee, or Coreferee is not properly initialized.")

# Example Usage
if __name__ == "__main__":
    example_text = (
        "Apple Inc. is looking at buying a U.K. startup for $1 billion. "
        "Tim Cook, Apple's CEO, said he is excited about the potential acquisition. "
        "The company hasn't confirmed the deal yet. It hopes to finalize it by next week."
    )
    
    print(f"Analyzing text: \"{example_text}\"")
    analyze_text_for_ner_and_coreference(example_text)

    print("\n\n--- Another Example ---")
    another_example_text = (
        "Dr. Emily Carter presented her research on sustainable energy. "
        "She argued that new policies are needed. Many attendees agreed with her."
    )
    print(f"Analyzing text: \"{another_example_text}\"")
    analyze_text_for_ner_and_coreference(another_example_text)