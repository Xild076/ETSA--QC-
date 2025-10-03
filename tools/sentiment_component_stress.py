import sys
from pathlib import Path
from typing import List, Dict, Any, Literal

# Add src and root to path, as requested
try:
    # This structure assumes the script is in a subdirectory like 'tests/' or 'scripts/'
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pipeline.sentiment_analysis import MultiSentimentAnalysis
except ImportError:
    print("Could not import MultiSentimentAnalysis. Please ensure the script is run from a location with access to the 'pipeline' module.")
    sys.exit(1)

# --- Test Case Definition ---
# Expanded and more complex cases based on deep analysis of error logs and datasets.

SentimentPolarity = Literal['positive', 'negative', 'neutral']

stress_test_cases: List[Dict[str, Any]] = [
    # --- Category 1: Context-Dependent Polarity (Advanced) ---
    {"category": "Context-Dependent Polarity", "text": "The price was higher than I wanted.", "expected": "negative", "notes": "Polarity of 'higher' depends on the noun 'price'."},
    {"category": "Context-Dependent Polarity", "text": "The performance was higher than I wanted.", "expected": "positive", "notes": "Polarity of 'higher' flips for 'performance'."},
    {"category": "Context-Dependent Polarity", "text": "The price is down by $200.", "expected": "positive", "notes": "'down' is positive for price."},
    {"category": "Context-Dependent Polarity", "text": "The system performance is down since the update.", "expected": "negative", "notes": "'down' is negative for performance."},
    {"category": "Context-Dependent Polarity", "text": "The heat output is impressively low.", "expected": "positive", "notes": "'low' is positive for 'heat output'."},
    {"category": "Context-Dependent Polarity", "text": "The speaker volume is disappointingly low.", "expected": "negative", "notes": "'low' is negative for 'volume'."},
    {"category": "Context-Dependent Polarity", "text": "This laptop needs a bigger power switch.", "expected": "negative", "notes": "'bigger' implies the current one is inadequate (too small)."},
    {"category": "Context-Dependent Polarity", "text": "USB3 Peripherals are noticeably less expensive.", "expected": "positive", "notes": "'less' is positive for 'expensive'."},

    # --- Category 2: Quantifiers, Absence, and User Need (Advanced) ---
    {"category": "Quantifiers & Absence", "text": "It came with ZERO battery cycle count.", "expected": "positive", "notes": "'ZERO' is positive for a wear-and-tear feature."},
    {"category": "Quantifiers & Absence", "text": "This laptop has only 2 USB ports.", "expected": "negative", "notes": "'only' implies insufficiency."},
    {"category": "Quantifiers & Absence", "text": "The new model has no HDMI receptacle.", "expected": "negative", "notes": "Absence of a standard feature."},
    {"category": "Quantifiers & Absence", "text": "It has everything I need except for a word program.", "expected": "negative", "notes": "'except for' indicates a missing, desirable feature."},
    {"category": "Quantifiers & Absence", "text": "Apple removed the DVD drive.", "expected": "negative", "notes": "Removal of a feature is a negative evaluation."},
    {"category": "Quantifiers & Absence", "text": "It has more than enough memory for my tasks.", "expected": "positive", "notes": "Positive quantifier indicating sufficiency."},

    # --- Category 3: Complex Stances (Sarcasm, Counterfactuals, Reported Speech) ---
    {"category": "Complex Stances", "text": "A three-hour wait for a salad? Oh, that was just wonderful.", "expected": "negative", "notes": "Sarcasm. The positive word 'wonderful' is used to mean the opposite."},
    {"category": "Complex Stances", "text": "The service is great, if you can ever manage to get their attention.", "expected": "negative", "notes": "Conditional praise that is negated by the condition."},
    {"category": "Complex Stances", "text": "I would have given it 5 stars was it not for Windows 8.", "expected": "negative", "notes": "Counterfactual structure strongly blames 'Windows 8'."},
    {"category": "Complex Stances", "text": "Some people complain about the low resolution, but I think that's ridiculous.", "expected": "positive", "notes": "Author refutes a negative opinion, making the stance positive."},
    {"category": "Complex Stances", "text": "I thought the transition would be difficult, but it was easy.", "expected": "neutral", "notes": "Evaluates a past *expectation* ('thought'), not the entity itself."},
    {"category": "Complex Stances", "text": "Toshiba support informed me the mother board was faulty.", "expected": "neutral", "notes": "Neutral act of reporting. Negativity is on the 'mother board', not 'support'."},
    {"category": "Complex Stances", "text": "The only solution is to turn the brightness down, which is a hassle.", "expected": "negative", "notes": "The *need* for a solution implies a problem. The entity is the subject of a negative situation."},
    {"category": "Complex Stances", "text": "I miss my old Windows computer.", "expected": "positive", "notes": "'miss' is a positive sentiment towards the object being missed."},

    # --- Category 4: Factual Statements & Scope ---
    {"category": "Factual & Scope", "text": "I'm using this computer for word processing and gaming.", "expected": "neutral", "notes": "Statement of purpose, not evaluation."},
    {"category": "Factual & Scope", "text": "I've installed an additional SSD and 16Gb RAM.", "expected": "neutral", "notes": "Factual statement of an action taken."},
    {"category": "Factual & Scope", "text": "Other ports include FireWire 800 and Gigabit Ethernet.", "expected": "neutral", "notes": "Simple factual listing of features."},
    {"category": "Factual & Scope", "text": "I am used to Mac, so Windows feels confusing.", "expected": "neutral", "notes": "Entity 'Mac' is a neutral reference for user habit, not being evaluated."},

    # --- Category 5: Implicit Sentiment & World Knowledge ---
    {"category": "Implicit & World Knowledge", "text": "The laptop's battery lasts for a full 90 minutes.", "expected": "negative", "notes": "Requires world knowledge that 90 mins is very poor for a laptop battery."},
    {"category": "Implicit & World Knowledge", "text": "The pizza arrived cold.", "expected": "negative", "notes": "Requires world knowledge that pizza is expected to be hot."},
    {"category": "Implicit & World Knowledge", "text": "I asked for medium-rare and the steak came out well-done.", "expected": "negative", "notes": "Implicitly negative as it describes a failure to meet a user's request."},
    {"category": "Implicit & World Knowledge", "text": "The experience was great since the application will simply shutdown and reopen without crashing the OS.", "expected": "positive", "notes": "Locally negative 'shutdown' is framed as globally positive (resilience)."},
    {"category": "Implicit & World Knowledge", "text": "The battery dies if you just look at it the wrong way.", "expected": "negative", "notes": "Figurative language (hyperbole) implying extreme fragility and poor quality."},

    # --- Category 6: Sanity Checks (Simple, Direct Sentiment) ---
    {"category": "Sanity Check", "text": "The service was dreadful and the food was awful.", "expected": "negative", "notes": "Basic strong negative words."},
    {"category": "Sanity Check", "text": "A truly wonderful experience.", "expected": "positive", "notes": "Basic strong positive words."},
    {"category": "Sanity Check", "text": "Mac tutorials do help.", "expected": "positive", "notes": "Simple positive verb phrase."},
    {"category": "Sanity Check", "text": "tried windows 8 and hated it.", "expected": "negative", "notes": "Very strong and direct negative emotion."},
]

# --- Model & Configuration Definition ---

models_to_test = {
    # Individual Models
    "Vader (standalone)": MultiSentimentAnalysis(methods=['vader'], weights=[1.0]),
    "Flair (standalone)": MultiSentimentAnalysis(methods=['flair'], weights=[1.0]),
    "Pysentimiento (standalone)": MultiSentimentAnalysis(methods=['pysentimiento'], weights=[1.0]),
    "Distilbert (standalone)": MultiSentimentAnalysis(methods=['distilbert_logit'], weights=[1.0]),
    # Combined Configurations
    "Pysentimiento-Heavy (Default)": MultiSentimentAnalysis(
        methods=['distilbert_logit', 'flair', 'pysentimiento', 'vader'],
        weights=[0.3, 0.1, 0.5, 0.1]
    ),
    "Equally-Weighted": MultiSentimentAnalysis(
        methods=['distilbert_logit', 'flair', 'pysentimiento', 'vader'],
        weights=[0.25, 0.25, 0.25, 0.25]
    ),
    "Lexical+Transformer": MultiSentimentAnalysis(
        methods=['vader', 'distilbert_logit'],
        weights=[0.5, 0.5]
    ),
    "Transformer-Only": MultiSentimentAnalysis(
        methods=['distilbert_logit', 'pysentimiento'],
        weights=[0.5, 0.5]
    )
}

# --- Test Execution and Reporting ---

# ANSI color codes for reporting
C_GREEN = '\033[92m'
C_RED = '\033[91m'
C_YELLOW = '\033[93m'
C_BLUE = '\033[94m'
C_BOLD = '\033[1m'
C_END = '\033[0m'

def get_polarity_from_score(score: float, pos_thresh: float = 0.1, neg_thresh: float = -0.1) -> SentimentPolarity:
    """Maps a float score to a polarity string."""
    if score >= pos_thresh:
        return 'positive'
    elif score <= neg_thresh:
        return 'negative'
    return 'neutral'

def run_stress_test():
    """Executes the stress test and prints a detailed report."""
    print(f"{C_BOLD}--- Starting Advanced Sentiment Analysis Stress Test ---{C_END}")

    results: Dict[str, Dict[str, str]] = {}
    model_stats: Dict[str, Dict[str, int]] = {name: {"correct": 0, "total": 0} for name in models_to_test}

    # Execute all tests
    for test in stress_test_cases:
        text = test["text"]
        results[text] = {}
        for model_name, model in models_to_test.items():
            try:
                result = model.analyze(text)
                score = result.get('aggregate', 0.0)
                polarity = get_polarity_from_score(score)
                results[text][model_name] = polarity
                
                model_stats[model_name]["total"] += 1
                if polarity == test["expected"]:
                    model_stats[model_name]["correct"] += 1
            except Exception as e:
                results[text][model_name] = f"ERROR: {e}"

    # Print detailed results
    current_category = ""
    for test in stress_test_cases:
        if test["category"] != current_category:
            current_category = test["category"]
            print(f"\n{C_BLUE}--- {current_category} ---{C_END}")
        
        print(f"\n{C_BOLD}Text:{C_END}      '{test['text']}'")
        print(f"{C_BOLD}Notes:{C_END}     {test['notes']}")
        print(f"{C_BOLD}Expected:{C_END}  {C_YELLOW}{test['expected'].upper()}{C_END}")
        
        for model_name in models_to_test.keys():
            actual = results[test["text"]][model_name]
            if actual == test["expected"]:
                status = f"{C_GREEN}PASS{C_END}"
            else:
                status = f"{C_RED}FAIL{C_END}"
            print(f"  - {model_name:<35}: {actual.upper():<10} [{status}]")

    # Print summary table
    print(f"\n\n{C_BOLD}--- Stress Test Summary ---{C_END}")
    print(f"{'Model Configuration':<40} | {'Correct':>10} | {'Total':>10} | {'Accuracy':>10}")
    print("-" * 75)
    for model_name, stats in model_stats.items():
        correct = stats['correct']
        total = stats['total']
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"{model_name:<40} | {correct:>10} | {total:>10} | {f'{accuracy:.2f}%':>10}")
    
    print("\n--- Analysis & Recommendations ---")
    print("1. Sarcasm, world knowledge, and complex stances are the clear failure points for most models.")
    print("2. `Vader` (lexical) fails on almost all context-heavy cases but is fast and reliable for simple sentiment.")
    print("3. Transformer models (`Distilbert`, `Pysentimiento`) show the best performance on complex context but are not perfect.")
    print("4. Compare the `Transformer-Only` configuration to your default. If it performs significantly better, the lexical models might be adding more noise than value for these complex cases.")
    print("5. If all configurations fail on a specific category (e.g., Sarcasm), it indicates a 'linguistic ceiling' that cannot be solved by re-weighting alone. This requires either fine-tuning a model on such examples or using rule-based pre-processing.")


if __name__ == "__main__":
    run_stress_test()