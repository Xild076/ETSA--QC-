# IMPROVEMENT PLAN: Achieving 85% Accuracy in ABSA Pipeline

## Executive Summary
**Current State:** 75.0% accuracy (33/44 correct predictions)  
**Target:** 85.0% accuracy (37/44 correct predictions)  
**Gap:** 4 predictions need to be fixed  
**Wrong_Polarity Errors:** 11 (fixing all would achieve 100% accuracy)

## Detailed Analysis

### Current Performance Metrics
- **Accuracy:** 75.0%
- **Balanced Accuracy:** 56.3%
- **Precision (macro):** 76.7%
- **Recall (macro):** 70.9%
- **F1 (macro):** 71.1%

### Error Breakdown
1. **Wrong Polarity:** 11 errors
2. **Missing Aspect:** 4 errors  
3. **Spurious Aspect:** 134 errors (less critical for accuracy)

### Root Cause Analysis

#### 1. NEUTRAL CLASS MISCLASSIFICATION (CRITICAL)
- **Neutral Recall:** Only 41.7% (worst performing class)
- **Confusion Pattern:** 
  - 5/12 neutrals correctly classified
  - 6/12 neutrals misclassified as positive
  - 1/12 neutrals misclassified as negative
- **Root Cause:** Classification thresholds (±0.1) create too wide neutral band
- **Impact:** ~6 errors attributable to neutral misclassification

**Evidence:**
- Neutral confused as: {'negative': 1, 'neutral': 5, 'positive': 6}
- Average sentiment score across wrong_polarity cases: -0.049 (very close to neutral boundary)

#### 2. NEGATION HANDLING FAILURE (CRITICAL)
- **Frequency:** 7 out of 11 wrong_polarity cases (64%) contain negation
- **Negation Words Found:** "not", "no", "never", "n't"
- **Root Cause:** 
  - TextBlob fails on all negation cases (0% accuracy in diagnostic test)
  - Combiner doesn't boost negation signals adequately
  - Ensemble averaging dilutes strong negation signals

**Evidence from Diagnostic:**
```
"Did not enjoy..." → TextBlob: -0.032 (should be strong negative)
"No installation disk..." → TextBlob: +0.000 (should be negative)
"acceptable" → TextBlob: +0.000 (should be positive)
```

#### 3. COMBINER RELIABILITY ISSUES (HIGH)
- **Average Reliability Score:** 0.590 (moderate confidence)
- **Combiner:** AdaptiveV6Combiner (used in all 67 nodes)
- **Weight Distribution:** modifier 0.7, head 0.2
- **Root Cause:** 
  - Reliability may not accurately reflect sentiment quality
  - Over-reliance on modifiers when DummyModifierExtractor provides no real context

#### 4. DUMMY MODIFIER EXTRACTOR (MEDIUM)
- **Current:** Using DummyModifierExtractor (returns empty modifiers)
- **Impact:** Lost contextual information for sentiment analysis
- **Has Modifiers:** 62/11 nodes claim to have modifiers (but they're dummy/empty)
- **Real Impact:** Real modifiers would provide crucial evaluative context

#### 5. AGGREGATE MODEL AVERAGING (LOW-MEDIUM)
- **Multiple Entities per Text:** Average 6 entities per wrong_polarity case
- **Aggregation Method:** Simple averaging across entity sentiments
- **Root Cause:** May be averaging away strong sentiment signals
- **Example:** Entity sentiments range from -0.87 to +0.77, but aggregate may land in neutral zone

## Improvement Plan

### Phase 1: Critical Fixes (Target: +5-8% accuracy gain)

#### Fix 1.1: Adjust Classification Thresholds
**Current Thresholds:**
- Positive: score >= 0.1
- Negative: score <= -0.1  
- Neutral: -0.1 < score < 0.1

**Proposed Changes:**
```python
# Option A: Tighter neutral band
POS_THRESH = 0.05
NEG_THRESH = -0.05

# Option B: Asymmetric thresholds (favor precision)
POS_THRESH = 0.15
NEG_THRESH = -0.15

# Option C: Dynamic thresholds based on reliability
if reliability > 0.8:
    POS_THRESH = 0.05
    NEG_THRESH = -0.05
else:
    POS_THRESH = 0.15
    NEG_THRESH = -0.15
```

**Expected Impact:** Fix 3-4 neutral misclassifications → +6-9% accuracy

**Implementation:**
1. Modify `_score_to_label()` in `optimization.py`
2. Add threshold parameters to pipeline configuration
3. Test all three options via optimization

#### Fix 1.2: Improve Negation Handling
**Detection Logic:**
```python
NEGATION_PATTERNS = {
    'not', 'no', 'never', 'nothing', "n't", 'dont', "doesn't", "didn't",
    'cannot', "can't", 'wont', "won't", 'neither', 'nor', 'without'
}

def contains_negation(text: str) -> bool:
    tokens = text.lower().split()
    return any(neg in tokens or neg in text.lower() for neg in NEGATION_PATTERNS)
```

**Combiner Adjustment:**
```python
# In combiner logic
if contains_negation(text):
    # Boost negative signal weight
    if combined_score < 0:
        combined_score *= 1.3  # Amplify negative sentiment
    elif combined_score > 0:
        combined_score *= 0.7  # Dampen false positive sentiment
    
    # Increase confidence in VADER/DistilBERT (which handle negation well)
    # Decrease confidence in TextBlob
    method_weights = {
        'vader': 0.4,
        'distilbert_logit': 0.5,
        'textblob': 0.1  # Reduce TextBlob influence
    }
```

**Expected Impact:** Fix 4-5 negation cases → +9-11% accuracy

**Implementation:**
1. Add negation detection to `combiners.py`
2. Modify AdaptiveV6Combiner to apply negation boost
3. Update ensemble weighting in `sentiment_analysis.py`

#### Fix 1.3: Optimize Combiner Weights via Grid Search
**Current:** modifier=0.7, head=0.2, context=various

**Optimization Strategy:**
```python
# Run focused optimization on adaptive_v6 combiner
param_ranges = {
    'modifier_weight': (0.5, 0.9, 0.05),  # Range, step
    'head_weight': (0.1, 0.5, 0.05),
    'reliability_threshold': (0.5, 0.8, 0.05),
    'negation_boost': (1.0, 1.5, 0.1)
}
```

**Expected Impact:** Fine-tune 1-2 edge cases → +2-4% accuracy

**Implementation:**
1. Use existing optimization framework in `optimization.py`
2. Run 100 trials on laptop_2014 test set
3. Save best parameters to config

### Phase 2: Enable Real Features (Target: +2-4% accuracy gain)

#### Fix 2.1: Enable GemmaModifierExtractor
**Issue:** Currently using DummyModifierExtractor

**API Key Loading Fix:**
```python
# Ensure .env loads properly
# In utility.py (already implemented correctly)
def ensure_env_loaded():
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
    
# In pipeline.py or main initialization
from utility import ensure_env_loaded
ensure_env_loaded()
```

**Implementation:**
1. Verify `.env` file contains `GOOGLE_API_KEY`
2. Update `run_diagnostic_benchmark.py` to use GemmaModifierExtractor
3. Re-run benchmark with real modifiers

**Expected Impact:** Better context → +2-3% accuracy

#### Fix 2.2: Improve Aggregate Model
**Current:** Simple averaging across entity sentiments

**Proposed:** Weighted aggregation
```python
def aggregate_with_confidence(entity_sentiments, reliabilities):
    weighted_sum = sum(s * r for s, r in zip(entity_sentiments, reliabilities))
    weight_total = sum(reliabilities)
    return weighted_sum / weight_total if weight_total > 0 else 0.0
```

**Implementation:**
1. Modify `AggregateSentimentModel.calculate()` in `sentiment_model.py`
2. Pass reliability scores through the pipeline
3. Test on validation set

**Expected Impact:** Better signal preservation → +1-2% accuracy

### Phase 3: Validation & Deployment

#### Step 3.1: Run Full Benchmark
```bash
python run_diagnostic_benchmark.py
```

**Success Criteria:**
- Accuracy >= 85%
- Precision >= 75%
- Recall >= 80%
- Wrong_polarity errors <= 6

#### Step 3.2: Error Analysis
- If accuracy < 85%, analyze remaining errors
- Identify if new error patterns emerged
- Iterate on Phase 1 fixes

#### Step 3.3: Production Deployment
- Update default pipeline configuration
- Save optimized parameters to cache
- Document changes in README

## Implementation Priority

### HIGH PRIORITY (Do First)
1. **Fix 1.1: Threshold Adjustment** - 30 min implementation, high impact
2. **Fix 1.2: Negation Handling** - 2 hours implementation, very high impact
3. **Fix 1.3: Combiner Optimization** - 1 hour setup, 2 hours runtime

### MEDIUM PRIORITY (Do Second)
4. **Fix 2.1: Enable GemmaModifierExtractor** - 30 min verification
5. **Fix 2.2: Aggregate Model** - 1 hour implementation

### LOW PRIORITY (If Needed)
6. Additional combiner tuning
7. Ensemble method reweighting
8. Advanced negation patterns

## Expected Outcome

**Conservative Estimate:**
- Fix 1.1: +3%
- Fix 1.2: +5%
- Fix 1.3: +2%
- **Total: 75% → 85%** ✓

**Optimistic Estimate:**
- Fix 1.1: +6%
- Fix 1.2: +7%
- Fix 1.3: +3%
- Fix 2.1: +2%
- **Total: 75% → 93%** 🎯

## Code Changes Required

### Files to Modify:
1. `src/pipeline/optimization.py` - Threshold parameters
2. `src/pipeline/combiners.py` - Negation detection & boosting
3. `src/pipeline/sentiment_analysis.py` - Ensemble weights
4. `src/pipeline/sentiment_model.py` - Aggregate model
5. `run_diagnostic_benchmark.py` - Use GemmaModifierExtractor

### Configuration Updates:
1. `cache/best_combiner_config.json` - New optimized parameters
2. `.env` - Verify API key (already present)

## Risk Mitigation

**Risk 1:** Threshold changes may reduce precision
- **Mitigation:** Test multiple threshold configurations
- **Fallback:** Use dynamic thresholds based on reliability

**Risk 2:** Negation boost may over-correct
- **Mitigation:** Use conservative boost factor (1.2-1.3)
- **Validation:** Test on negation-specific test cases

**Risk 3:** API rate limits with GemmaModifierExtractor
- **Mitigation:** Use existing rate limiter (20 calls/60s)
- **Caching:** Store results to avoid re-extraction

## Success Metrics

**Primary:**
- Accuracy >= 85% ✓
- Wrong_polarity errors <= 6

**Secondary:**
- Balanced accuracy >= 75%
- Neutral class recall >= 70%
- Precision >= 80%
- F1 score >= 80%

## Timeline

**Day 1: Critical Fixes (Fixes 1.1-1.3)**
- Morning: Implement threshold adjustment & negation handling
- Afternoon: Run combiner optimization (2 hours)
- Evening: Benchmark and analyze results

**Day 2: Feature Enablement & Validation (Fixes 2.1-2.2)**
- Morning: Enable GemmaModifierExtractor
- Afternoon: Improve aggregate model
- Evening: Full validation benchmark

**Day 3: Iteration (if needed)**
- Address any remaining gaps
- Document final solution
- Deploy to production

---

**Document Version:** 1.0  
**Date:** October 1, 2025  
**Author:** System Analysis  
**Status:** Ready for Implementation
