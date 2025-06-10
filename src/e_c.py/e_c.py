import spacy
import json

nlp = spacy.load("en_core_web_lg")
try:
    if "coreferee" not in nlp.pipe_names:
        nlp.add_pipe("coreferee")
except Exception as e:
    print(f"Could not add coreferee pipe: {e}")
    print("Please ensure 'coreferee' is installed and the model supports it.")
    print("Install with: pip install coreferee")
    print("Download compatible model if needed, e.g.: python -m spacy download en_core_web_lg")
    exit()

def extract_entities_with_coref(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    token_to_sent_idx = {token.i: idx + 1 for idx, sent in enumerate(sentences) for token in sent}
    
    entity_dict = {} 
    token_to_chain_key = {} 
    chain_key_to_member_tokens = {}

    coref_chains = getattr(doc._, "coref_chains", [])

    for chain in coref_chains:
        if not chain.mentions:
            continue

        rep_mention = chain.mentions[0]
        rep_start, rep_end = min(rep_mention.token_indexes), max(rep_mention.token_indexes)
        chain_initial_key = doc[rep_start : rep_end + 1].text
        
        current_chain_data = entity_dict.setdefault(chain_initial_key, {"references": [], "reference_names": set()})
        chain_key_to_member_tokens.setdefault(chain_initial_key, set())

        for mention in chain.mentions:
            uni_m_start, uni_m_end = min(mention.token_indexes), max(mention.token_indexes)
            m_text = doc[uni_m_start : uni_m_end + 1].text
            
            sent_idx = token_to_sent_idx.get(uni_m_start, -1)
            inter_sent_m_start, inter_sent_m_end = -1, -1

            if sent_idx != -1 and (sent_idx - 1) < len(sentences):
                sentence_span = sentences[sent_idx - 1]
                inter_sent_m_start = uni_m_start - sentence_span.start
                inter_sent_m_end = uni_m_end - sentence_span.start
            
            ref_tuple = (sent_idx, (uni_m_start, uni_m_end), (inter_sent_m_start, inter_sent_m_end))
            
            if ref_tuple not in current_chain_data["references"]:
                 current_chain_data["references"].append(ref_tuple)
            current_chain_data["reference_names"].add(m_text)
            
            for token_idx in mention.token_indexes:
                token_to_chain_key[token_idx] = chain_initial_key
                chain_key_to_member_tokens[chain_initial_key].add(token_idx)

    for ent in doc.ents:
        ent_was_merged_or_added_to_chain = False
        ent_tokens = set(range(ent.start, ent.end))
        
        overlapping_chain_key = None
        for token_idx in ent_tokens:
            if token_idx in token_to_chain_key:
                overlapping_chain_key = token_to_chain_key[token_idx]
                break
        
        uni_ent_start = ent.start
        uni_ent_end_inclusive = ent.end - 1 

        sent_idx_for_ent = token_to_sent_idx.get(uni_ent_start, -1)
        inter_sent_ent_start, inter_sent_ent_end_inclusive = -1, -1
        if sent_idx_for_ent != -1 and (sent_idx_for_ent - 1) < len(sentences):
            sentence_span = sentences[sent_idx_for_ent - 1]
            inter_sent_ent_start = uni_ent_start - sentence_span.start
            inter_sent_ent_end_inclusive = uni_ent_end_inclusive - sentence_span.start
        
        current_ent_ref_tuple = (
            sent_idx_for_ent, 
            (uni_ent_start, uni_ent_end_inclusive), 
            (inter_sent_ent_start, inter_sent_ent_end_inclusive)
        )

        if overlapping_chain_key:
            ent_was_merged_or_added_to_chain = True
            
            target_key_for_chain = overlapping_chain_key
            if len(ent.text) > len(overlapping_chain_key) and ent.text != overlapping_chain_key:
                promoted_key = ent.text
                old_key = overlapping_chain_key
                target_key_for_chain = promoted_key

                if old_key in entity_dict:
                    data_to_move = entity_dict.pop(old_key)
                    promoted_key_data = entity_dict.setdefault(promoted_key, {"references": [], "reference_names": set()})
                    for ref in data_to_move["references"]:
                        if ref not in promoted_key_data["references"]:
                            promoted_key_data["references"].append(ref)
                    promoted_key_data["reference_names"].update(data_to_move["reference_names"])

                if old_key in chain_key_to_member_tokens:
                    tokens_of_old_chain = chain_key_to_member_tokens.pop(old_key)
                    chain_key_to_member_tokens.setdefault(promoted_key, set()).update(tokens_of_old_chain)
                    for t_idx in tokens_of_old_chain:
                        if t_idx in token_to_chain_key and token_to_chain_key[t_idx] == old_key:
                            token_to_chain_key[t_idx] = promoted_key
            
            final_chain_data = entity_dict.setdefault(target_key_for_chain, {"references": [], "reference_names": set()})
            final_chain_data["reference_names"].add(ent.text)
            if current_ent_ref_tuple not in final_chain_data["references"]:
                final_chain_data["references"].append(current_ent_ref_tuple)
            
            chain_key_to_member_tokens.setdefault(target_key_for_chain, set())
            for t_idx in ent_tokens:
                token_to_chain_key[t_idx] = target_key_for_chain
                chain_key_to_member_tokens[target_key_for_chain].add(t_idx)

        if not ent_was_merged_or_added_to_chain:
            key = ent.text
            new_entity_data = entity_dict.setdefault(key, {"references": [], "reference_names": set()})
            if current_ent_ref_tuple not in new_entity_data["references"]:
                new_entity_data["references"].append(current_ent_ref_tuple)
            new_entity_data["reference_names"].add(ent.text)

    output_dict = {}
    for key, data in entity_dict.items():
        temp_unique_references = sorted(list(set(data["references"])), key=lambda x: (x[0], x[1][0], -x[1][1]))

        final_filtered_references = []
        for i in range(len(temp_unique_references)):
            current_ref_is_subsumed = False
            for j in range(len(temp_unique_references)):
                if i == j:
                    continue
                
                current_ref = temp_unique_references[i]
                potential_subsumer_ref = temp_unique_references[j]
                
                if current_ref[0] == potential_subsumer_ref[0] and \
                   potential_subsumer_ref[1][0] <= current_ref[1][0] and \
                   potential_subsumer_ref[1][1] >= current_ref[1][1] and \
                   (potential_subsumer_ref[1][0] != current_ref[1][0] or \
                    potential_subsumer_ref[1][1] != current_ref[1][1]):
                    current_ref_is_subsumed = True
                    break
            if not current_ref_is_subsumed:
                final_filtered_references.append(temp_unique_references[i])
        
        output_dict[key] = {
            "references": final_filtered_references, 
            "reference_names": sorted(list(data["reference_names"]))
        }
        
    result = {"name_entity": output_dict}
    return result

def print_entity_summary(result, doc_text):
    doc = nlp(doc_text)
    print("\n=== Entity Summary ===")
    if not result["name_entity"]:
        print("No entities found.")
        return
        
    for key, data in result["name_entity"].items():
        print(f"\nEntity: {key}")
        print(f"Aliases: {', '.join(data['reference_names'])}")
        print("Mentions:")
        if not data["references"]:
            print("  No mentions recorded.")
            continue
        for ref_data in data["references"]:
            sent_idx, universal_token_span, inter_sentence_token_span = ref_data
            uni_start, uni_end = universal_token_span
            inter_start, inter_end = inter_sentence_token_span
            
            span_text = doc[uni_start : uni_end + 1].text
            
            print(f"  - Sentence {sent_idx}: \"{span_text}\" "
                  f"(Universal Tokens {uni_start}-{uni_end}, "
                  f"Sentence Tokens {inter_start}-{inter_end})")

text = "Barack Obama was the 44th President of the United States. He was born in Hawaii. Obama served two terms."
result = extract_entities_with_coref(text)
print_entity_summary(result, text)

text2 = "The European Union announced new regulations. These regulations will affect tech companies. The EU hopes this fosters innovation."
result2 = extract_entities_with_coref(text2)
print_entity_summary(result2, text2)