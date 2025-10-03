#!/usr/bin/env python3
"""
Quick test to verify rate limiting improvements are working correctly.
This simulates rate limit errors without actually hitting the API.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'pipeline'))

from modifier_e import GemmaModifierExtractor
from unittest.mock import MagicMock, patch
import time


def test_rate_limit_detection():
    """Test that rate limit errors are properly detected and handled"""
    print("Testing rate limit error detection...")
    
    # Create extractor with minimal retries for quick testing
    extractor = GemmaModifierExtractor(retries=1, backoff=0.1)
    
    # Mock the _call method to simulate rate limit errors
    original_call = extractor._call
    call_count = [0]
    
    def mock_call_rate_limit(prompt):
        call_count[0] += 1
        if call_count[0] <= 2:
            # Simulate rate limit error for first 2 calls
            raise Exception("429 Resource has been exhausted (e.g. check quota).")
        else:
            # Return valid response on 3rd call
            return '''```json
{
    "entity": "test",
    "justification": "Success after retry",
    "modifiers": ["good"],
    "approach_used": "models/gemma-3-27b-it"
}
```'''
    
    extractor._call = mock_call_rate_limit
    
    print("  Simulating rate limit (will retry with exponential backoff)...")
    start_time = time.time()
    result = extractor.extract("This is a good test", "test")
    elapsed = time.time() - start_time
    
    print(f"  Result: {result}")
    print(f"  API calls made: {call_count[0]}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    
    assert call_count[0] == 3, f"Expected 3 calls (2 failures + 1 success), got {call_count[0]}"
    assert len(result['modifiers']) > 0, "Expected modifiers to be extracted after retry"
    assert elapsed >= 2, f"Expected at least 2s wait (2^1 + 2^2), got {elapsed:.1f}s"
    
    print("  ✅ Rate limit retry with exponential backoff works!\n")


def test_rate_limit_exhaustion():
    """Test that excessive rate limit errors are cached as failures"""
    print("Testing rate limit exhaustion (max retries exceeded)...")
    
    extractor = GemmaModifierExtractor(retries=1, backoff=0.1)
    
    # Mock the _call method to always fail with rate limit
    call_count = [0]
    
    def mock_call_always_fail(prompt):
        call_count[0] += 1
        raise Exception("429 Too many requests")
    
    extractor._call = mock_call_always_fail
    
    print("  Simulating continuous rate limiting...")
    start_time = time.time()
    result = extractor.extract("This is a test", "test")
    elapsed = time.time() - start_time
    
    print(f"  Result: {result}")
    print(f"  API calls made: {call_count[0]}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    
    # Should make max_rate_limit_retries + 1 calls (default is 5+1=6)
    assert call_count[0] >= 5, f"Expected at least 5 retry attempts, got {call_count[0]}"
    assert len(result['modifiers']) == 0, "Expected empty modifiers on failure"
    assert "RATE_LIMIT_EXCEEDED" in result.get('justification', ''), \
        f"Expected RATE_LIMIT_EXCEEDED in justification, got: {result.get('justification', '')}"
    
    # Check if it was cached
    cache_key = ("This is a test", "test")
    assert cache_key in extractor._cache, "Failed result should be cached"
    cached = extractor._cache[cache_key]
    assert "RATE_LIMIT_EXCEEDED" in cached.get('justification', ''), \
        "Cached result should contain RATE_LIMIT_EXCEEDED"
    
    print("  ✅ Rate limit exhaustion is properly cached!\n")


def test_normal_error_handling():
    """Test that non-rate-limit errors use standard retry logic"""
    print("Testing normal error handling (non-rate-limit)...")
    
    extractor = GemmaModifierExtractor(retries=2, backoff=0.1)
    
    call_count = [0]
    
    def mock_call_normal_error(prompt):
        call_count[0] += 1
        if call_count[0] <= 1:
            raise Exception("Connection timeout")
        else:
            return '''```json
{
    "entity": "test",
    "justification": "Success after normal error retry",
    "modifiers": ["great"],
    "approach_used": "models/gemma-3-27b-it"
}
```'''
    
    extractor._call = mock_call_normal_error
    
    print("  Simulating normal API error...")
    start_time = time.time()
    result = extractor.extract("This is a great test", "test")
    elapsed = time.time() - start_time
    
    print(f"  Result: {result}")
    print(f"  API calls made: {call_count[0]}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    
    assert call_count[0] == 2, f"Expected 2 calls (1 failure + 1 success), got {call_count[0]}"
    assert len(result['modifiers']) > 0, "Expected modifiers after retry"
    # Normal retry uses linear backoff: 0.1 * 1 = 0.1s
    assert 0.05 <= elapsed <= 0.5, f"Expected ~0.1s wait for normal retry, got {elapsed:.1f}s"
    
    print("  ✅ Normal error retry with linear backoff works!\n")


def test_cache_hit():
    """Test that cached results are returned without API calls"""
    print("Testing cache hit (no API call should be made)...")
    
    extractor = GemmaModifierExtractor(retries=1)
    
    # Pre-populate cache
    cache_key = ("Cached test", "test")
    extractor._cache[cache_key] = {
        "entity": "test",
        "modifiers": ["cached"],
        "ordered_modifiers": ["cached"],
        "modifier_annotations": [],
        "approach_used": "models/gemma-3-27b-it",
        "justification": "From cache"
    }
    
    call_count = [0]
    
    def mock_call_should_not_be_called(prompt):
        call_count[0] += 1
        raise Exception("Should not be called!")
    
    extractor._call = mock_call_should_not_be_called
    
    print("  Retrieving from cache...")
    result = extractor.extract("Cached test", "test")
    
    print(f"  Result: {result}")
    print(f"  API calls made: {call_count[0]}")
    
    assert call_count[0] == 0, f"Expected 0 API calls (cache hit), got {call_count[0]}"
    assert result['modifiers'] == ['cached'], "Expected cached modifier"
    assert result['justification'] == "From cache", "Expected cached justification"
    
    print("  ✅ Cache hit works correctly!\n")


if __name__ == "__main__":
    print("=" * 70)
    print("RATE LIMITING IMPROVEMENTS - UNIT TESTS")
    print("=" * 70)
    print()
    
    try:
        test_cache_hit()
        test_normal_error_handling()
        test_rate_limit_detection()
        test_rate_limit_exhaustion()
        
        print("=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print()
        print("Summary:")
        print("  • Cache hits avoid API calls ✓")
        print("  • Normal errors use linear backoff ✓")
        print("  • Rate limit errors use exponential backoff ✓")
        print("  • Rate limit exhaustion is properly cached ✓")
        print()
        print("The modifier extraction system now properly handles:")
        print("  ✓ Rate limiting with visible warnings")
        print("  ✓ Exponential backoff for rate limits")
        print("  ✓ Failure caching to prevent repeated attempts")
        print("  ✓ Distinction between rate limit and other errors")
        
    except AssertionError as e:
        print()
        print("=" * 70)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 70)
        sys.exit(1)
    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ UNEXPECTED ERROR: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)
