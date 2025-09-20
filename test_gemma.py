#!/usr/bin/env python3

import sys
sys.path.append('src')

from pipeline.re_e import GemmaRelationExtractor
from pipeline.modifier_e import GemmaModifierExtractor
import os

# Set API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyD3e-8l2yX23DcsB3BaCx6Zy3xBZgqPoQU"

def test_gemma_functionality():
    """Test Gemma extractors to ensure they're working with API"""
    
    try:
        # Test relation extractor
        print("=== Testing GemmaRelationExtractor ===")
        rel_extractor = GemmaRelationExtractor()
        
        test_text = "The bread was top notch quality."
        test_entities = ["The bread", "quality"]
        
        relations = rel_extractor.extract(test_text, test_entities)
        print(f"Text: {test_text}")
        print(f"Entities: {test_entities}")
        print(f"Relations: {relations}")
        print()
        
        # Test modifier extractor  
        print("=== Testing GemmaModifierExtractor ===")
        mod_extractor = GemmaModifierExtractor()
        
        modifier_result = mod_extractor.extract(test_text, "The bread")
        print(f"Text: {test_text}")
        print(f"Entity: The bread")
        print(f"Modifiers: {modifier_result}")
        
        # Test another case
        test_text2 = "Certainly not the best sushi in New York."
        modifier_result2 = mod_extractor.extract(test_text2, "Certainly not the best sushi")
        print(f"\nText: {test_text2}")
        print(f"Entity: Certainly not the best sushi")
        print(f"Modifiers: {modifier_result2}")
        
        print("\n✅ Gemma API tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing Gemma API: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gemma_functionality()
