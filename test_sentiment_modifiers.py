#!/usr/bin/env python3

import sys
sys.path.append('src')

from pipeline.modifier_e import GemmaModifierExtractor, SpacyModifierExtractor
import os

# Set API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyD3e-8l2yX23DcsB3BaCx6Zy3xBZgqPoQU"

def test_sentiment_modifier_extraction():
    """Test that sentiment modifiers like 'not the best' are extracted properly"""
    
    gemma_extractor = GemmaModifierExtractor()
    spacy_extractor = SpacyModifierExtractor()
    
    test_cases = [
        {
            "text": "Certainly not the best sushi in New York.",
            "entity": "sushi",
            "expected": "Should extract 'not the best' as sentiment modifier"
        },
        {
            "text": "The bread was top notch quality.",
            "entity": "The bread", 
            "expected": "Should extract 'top notch' as sentiment modifier"
        },
        {
            "text": "Very good service but poor food quality.",
            "entity": "service",
            "expected": "Should extract 'very good' as sentiment modifier"
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Text: {case['text']}")
        print(f"Entity: {case['entity']}")
        print(f"Expected: {case['expected']}")
        
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
    test_sentiment_modifier_extraction()
