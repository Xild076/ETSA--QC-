#!/usr/bin/env python3

import sys
sys.path.append('src')

from pipeline.modifier_e import GemmaModifierExtractor, SpacyModifierExtractor
import os

# Set API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyD3e-8l2yX23DcsB3BaCx6Zy3xBZgqPoQU"

def test_modifier_filtering():
    """Test that modifiers don't include words already in the entity"""
    
    gemma_extractor = GemmaModifierExtractor()
    spacy_extractor = SpacyModifierExtractor()
    
    test_cases = [
        {
            "text": "They have the fastest delivery times in the city.",
            "entity": "the fastest delivery times",
            "expected_behavior": "Should not extract 'the fastest' as modifier since it's already in entity"
        },
        {
            "text": "The bread was top notch quality.",
            "entity": "The bread", 
            "expected_behavior": "Should extract 'top notch' as modifier"
        },
        {
            "text": "The service was disappointingly slow.",
            "entity": "service",
            "expected_behavior": "Should extract 'disappointingly slow' as modifier"
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Text: {case['text']}")
        print(f"Entity: {case['entity']}")
        print(f"Expected: {case['expected_behavior']}")
        
        try:
            # Test SpaCy extractor
            spacy_result = spacy_extractor.extract(case['text'], case['entity'])
            print(f"SpaCy modifiers: {spacy_result.get('modifiers', [])}")
            
            # Test Gemma extractor
            gemma_result = gemma_extractor.extract(case['text'], case['entity'])
            print(f"Gemma modifiers: {gemma_result.get('modifiers', [])}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_modifier_filtering()
