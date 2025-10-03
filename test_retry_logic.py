#!/usr/bin/env python3
"""
Quick test to verify API retry logic with exponential backoff.
"""

import sys
import time
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, 'src')

from pipeline.modifier_e import GemmaModifierExtractor

def test_retry_logic():
    """Test that 429 errors trigger exponential backoff"""
    
    print("Testing retry logic with 429 quota errors...")
    
    # Mock the genai module
    with patch('pipeline.modifier_e.genai') as mock_genai:
        # Create a mock that simulates quota exceeded error
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model
        
        # First call raises 429, second call succeeds
        call_count = [0]
        start_times = []
        
        def mock_generate(*args, **kwargs):
            start_times.append(time.time())
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: simulate quota exceeded
                raise Exception("429 Resource exhausted: Quota exceeded")
            else:
                # Second call: succeed
                response = Mock()
                response.text = '```json\n{"modifiers": [], "justification": "test"}\n```'
                return response
        
        mock_model.generate_content.side_effect = mock_generate
        
        # Create extractor
        extractor = GemmaModifierExtractor(
            api_key="fake_key_for_testing",
            retries=2,
            cache_only=False
        )
        
        # Disable cache for this test
        extractor._cache = {}
        
        # Try extraction
        print("Calling extract() - expecting 429 on first attempt...")
        result = extractor.extract("Test sentence", "test entity")
        
        print(f"\n✓ Extract succeeded after {call_count[0]} attempts")
        print(f"✓ Result: {result}")
        
        # Verify retry happened
        assert call_count[0] == 2, f"Expected 2 API calls, got {call_count[0]}"
        
        # Verify exponential backoff (should wait ~1 second between attempts)
        if len(start_times) >= 2:
            wait_time = start_times[1] - start_times[0]
            print(f"✓ Wait time between attempts: {wait_time:.2f}s (expected ~1s for retry_attempt=0)")
            assert 0.8 <= wait_time <= 1.5, f"Wait time {wait_time}s not in expected range"
        
        print("\n✅ All retry logic tests passed!")
        return True

if __name__ == "__main__":
    try:
        test_retry_logic()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
