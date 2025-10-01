# Wrong Polarity Analysis: Laptop vs Restaurant Performance Gap

**Date:** October 1, 2025  
**Laptop Accuracy:** 62.7% (206 wrong_polarity errors)  
**Restaurant Accuracy:** 74.2% (270 wrong_polarity errors)  
**Performance Gap:** 11.5 percentage points

---

## Executive Summary

The laptop dataset performs **significantly worse** (62.7%) compared to restaurant (74.2%) despite having **fewer total wrong_polarity errors** (206 vs 270). The key issue is **neutral recall**:

- **Laptop neutral recall: 22.5%** (catastrophic)
- **Restaurant neutral recall: 27.6%** (still poor, but better)

This means the laptop dataset has **135 total neutral aspects**, but the system only correctly identifies **36 of them** (77.5% failure rate).

---

## Key Findings

### 1. "Conflict" Polarity Explanation

**Q: Why is "conflict" a polarity?**  
**A:** "Conflict" represents aspects with **mixed sentiment** (both positive and negative in the same mention).

**Examples:**
- "The track pad is a bit squirrely at times- sometimes too sensitive, and sometimes a bit unresp onsive, but it's usable." (Gold: conflict, Pred: negative)
- "So noise is reduced at least 50% and the heat is much better, now it doesn't feel hot but warm." (Gold: conflict)

**System Issue:** The pipeline doesn't predict "conflict" because it's designed as a 3-class classifier (positive/negative/neutral). It treats conflict as an error class:
- Laptop: 16 conflict aspects, 0% recall (all misclassified)
- Restaurant: 14 conflict aspects, 0% recall (all misclassified)

**Recommendation:** Either:
1. Map conflict → neutral in gold standard preprocessing (treat mixed as neutral)
2. Implement multi-label classification to detect mixed sentiments

---

## 2. Misclassification Pattern Analysis

### Laptop Dataset (206 errors)
```
neutral → negative   77 (37.4%)  ← BIGGEST PROBLEM
neutral → positive   47 (22.8%)
positive → negative  46 (22.3%)  ← SECOND BIGGEST
positive → neutral   21 (10.2%)
negative → positive  11 (5.3%)
negative → neutral    4 (1.9%)
```

**Critical Issue:** 60.2% of wrong_polarity errors are **neutral misclassifications**
- Mean predicted score: -0.122 (slight negative bias)
- 62.1% misclassified as negative
- 37.9% misclassified as positive

### Restaurant Dataset (270 errors)
```
neutral → negative   76 (28.1%)
neutral → positive   63 (23.3%)
positive → negative  63 (23.3%)
positive → neutral   28 (10.4%)
negative → positive  25 (9.3%)
negative → neutral   15 (5.6%)
```

**Key Difference:** Restaurant has **more balanced** error distribution
- Neutral misclassifications: 51.5% (vs 60.2% laptop)
- Mean predicted score for neutrals: -0.056 (vs -0.122 laptop)
- More even split: 54.7% → negative, 45.3% → positive

---

## 3. Root Cause Analysis

### Why Laptop Performs Worse

#### A. Neutral Classification Bias
- **Laptop neutral mean score: -0.122** (negative-leaning)
- **Restaurant neutral mean score: -0.056** (closer to true neutral)
- **Threshold: ±0.1** means any score < -0.1 is classified negative

**Problem:** Laptop neutrals have a systematic negative bias that pushes them below the -0.1 threshold.

#### B. Domain-Specific Language Differences

**Laptop reviews contain more:**
1. **Technical jargon** that sentiment models misinterpret:
   - "low res" → incorrectly negative (Gold: positive context: "ridiculous to complain")
   - "Windows 7" → incorrectly negative (Gold: positive)
   - "Firewire 800" → neutral but scored negative

2. **Comparative/contrastive statements:**
   - "although some people might complain... which I think is ridiculous" → Gold: positive, Pred: negative (-0.764)
   - "The price is 200 dollars down" → Gold: positive (price drop), Pred: negative (-0.693)

3. **Double negation and hedging:**
   - "does not become unstable" → Gold: positive, Pred: negative (-0.693)

**Restaurant reviews are more straightforward:**
- "delicious", "great", "terrible", "awful" → clear sentiment words
- Less technical terminology
- More direct expressions

#### C. Score Distribution Comparison

| Metric | Laptop | Restaurant |
|--------|--------|------------|
| Neutral mean score | -0.122 | -0.056 |
| Neutral score range | [-0.698, 0.783] | [-0.696, 0.770] |
| Positive mean score (when misclass) | -0.295 | -0.260 |
| Negative mean score (when misclass) | +0.310 | +0.265 |

**Insight:** Laptop has **stronger polarity shift** when misclassifying, suggesting sentiment models are more confused by laptop language.

---

## 4. Negation Handling Analysis

- **Laptop:** 30.6% of wrong_polarity contain negation words
- **Restaurant:** 28.1% of wrong_polarity contain negation words

**Similar rates suggest negation is NOT the primary driver of the performance gap.**

---

## 5. Worst Error Examples

### Laptop Top 5 Worst Errors

1. **"Screen - although some people might complain about low res which I think is ridiculous."**
   - Gold: positive, Pred: negative (-0.764), Deviation: 0.864
   - **Issue:** Sarcasm/disagreement pattern ("which I think is ridiculous")

2. **"Screen - although some people might complain about low res which I think is ridiculous."**
   - Gold: positive, Pred: negative (-0.750), Deviation: 0.850
   - Aspect: "low res"
   - **Issue:** "low res" has inherent negative connotation, context says it's not a problem

3. **"A very important feature is Firewire 800 which in my experience works better then USB3..."**
   - Gold: positive, Pred: negative (-0.729), Deviation: 0.829
   - Aspect: "Windows 7"
   - **Issue:** Aspect extraction error + technical term misunderstood

4. **"However, the experience was great since the OS does not become unstable and the application will..."**
   - Gold: positive, Pred: negative (-0.693), Deviation: 0.793
   - Aspect: "application"
   - **Issue:** Double negation ("does not become unstable" = stable = positive)

5. **"The price is 200 dollars down."**
   - Gold: positive, Pred: negative (-0.693), Deviation: 0.793
   - Aspect: "price"
   - **Issue:** "down" without context sounds negative, but "200 dollars down" = price reduction = positive

---

## 6. Recommendations

### Immediate Actions (High Impact)

1. **Adjust Neutral Threshold to [-0.15, +0.15]**
   - Current: ±0.1 is too narrow
   - Laptop neutrals average -0.122, barely outside threshold
   - Wider threshold would capture borderline neutrals

2. **Domain-Specific Sentiment Lexicon for Laptops**
   - Add technical terms with context:
     - "low res" + "ridiculous/silly to complain" → positive
     - "price down" → positive
     - "does not [negative_word]" → positive

3. **Improve Double Negation Detection**
   - Pattern: "not/does not + [negative_word]" → positive
   - Examples: "not unstable", "no complaints", "nothing to hate"

4. **Context Window Expansion for Contrastive Patterns**
   - Detect "although/however/but + [contrasting_view]"
   - Look for meta-commentary: "I think [opinion]", "which is ridiculous"

### Medium-Term Improvements

5. **Handle Conflict Polarity**
   - Option A: Preprocess gold standard to map conflict → neutral
   - Option B: Implement multi-label classifier for mixed sentiments

6. **Comparative Statement Handling**
   - "better than", "worse than", "compared to"
   - "X is down" (price/weight decrease usually positive for those aspects)

7. **Technical Domain Fine-Tuning**
   - Fine-tune DistilBERT on laptop-specific reviews
   - Add laptop domain embeddings

---

## 7. Expected Impact

### If Neutral Threshold Adjusted to ±0.15:

**Laptop Dataset:**
- Current neutral predictions getting -0.122 average would shift to correct
- Estimated 40-50 errors fixed (from 77 neutral→negative)
- **Projected accuracy: ~70%** (from 62.7%)

**Restaurant Dataset:**
- Smaller impact since neutrals already at -0.056
- Estimated 15-20 errors fixed
- **Projected accuracy: ~76%** (from 74.2%)

### If Double Negation + Context Fixed:

- Laptop: Additional ~20-30 errors fixed
- **Projected accuracy: ~75%**

### Combined Improvements:

- **Laptop: 75-78% accuracy** (from 62.7%)
- **Restaurant: 78-80% accuracy** (from 74.2%)

---

## 8. Technical Debt Items

1. **Conflict polarity not handled** → Need decision on mapping strategy
2. **Neutral threshold hardcoded** → Should be domain-adaptive or learnable
3. **No domain-specific preprocessing** → Laptop vs restaurant need different handling
4. **Aspect extraction errors contaminate sentiment** → "Windows 7" aspect in Firewire sentence
5. **Sarcasm/irony not detected** → "which I think is ridiculous" reverses sentiment

---

## Conclusion

The 11.5% performance gap between laptop (62.7%) and restaurant (74.2%) is primarily driven by:

1. **Neutral classification failure** (22.5% recall vs 27.6%)
2. **Laptop domain complexity** (technical jargon, comparative statements, double negation)
3. **Systematic negative bias** in laptop neutral predictions (-0.122 vs -0.056)

**Quick Win:** Adjust neutral threshold from ±0.1 to ±0.15 → **~7% accuracy gain**

**Full Fix:** Domain-specific handling + improved negation → **~15% accuracy gain** (reaching 75-78%)
