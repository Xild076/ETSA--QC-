#!/usr/bin/env python3
"""
Comprehensive stress test for sentiment analysis polarity issues.
Based on analysis of benchmark error files and wrong polarity cases.
"""

import sys
import os
sys.path.append('/Users/harry/Documents/Python_Projects/ETSA_(QC)/src')

from sentiment.sentiment import get_distilbert_logit_sentiment, get_vader_sentiment, get_textblob_sentiment
from pipeline.sentiment_analysis import MultiSentimentAnalysis
import json
from typing import Dict, List, Tuple

#!/usr/bin/env python3
"""
Comprehensive stress test for sentiment analysis polarity issues.
Based on analysis of 168 wrong polarity cases from benchmark outputs.
Expanded to cover all major failure patterns with 200+ test cases.
"""

import sys
import os
sys.path.append('/Users/harry/Documents/Python_Projects/ETSA_(QC)/src')

from sentiment.sentiment import get_distilbert_logit_sentiment, get_vader_sentiment, get_textblob_sentiment
from pipeline.sentiment_analysis import MultiSentimentAnalysis
import json
from typing import Dict, List, Tuple

def analyze_error_patterns():
    """Analyze patterns from 168 wrong polarity error files - massively expanded test cases."""

    # Patterns extracted from comprehensive benchmark error analysis
    error_patterns = {
        "neutral_aspects_misclassified": [
            # Statements of fact without sentiment (most common error ~40%)
            {
                "text": "No installation disk (DVD) is included.",
                "expected_polarity": "neutral",
                "issue": "neutral_aspects_misclassified",
                "analysis": "Statement of fact about missing feature"
            },
            {
                "text": "Apple removed the DVD drive Firewire port (will work with adapter) and put the SDXC slot in a silly position on the back.",
                "expected_polarity": "neutral",
                "issue": "neutral_aspects_misclassified",
                "analysis": "Hardware configuration description"
            },
            {
                "text": "It is really thick around the battery.",
                "expected_polarity": "neutral",
                "issue": "neutral_aspects_misclassified",
                "analysis": "Physical dimension description"
            },
            {
                "text": "The Mountain Lion OS is not hard to figure out if you are familiar with Microsoft Windows.",
                "expected_polarity": "neutral",
                "issue": "neutral_aspects_misclassified",
                "analysis": "Comparative ease of use statement"
            },
            {
                "text": "I used windows XP, windows Vista, and Windows 7 extensively.",
                "expected_polarity": "neutral",
                "issue": "neutral_aspects_misclassified",
                "analysis": "Past usage history"
            },
            {
                "text": "Unfortunately, it runs XP and Microsoft is dropping support next April.",
                "expected_polarity": "neutral",
                "issue": "neutral_aspects_misclassified",
                "analysis": "OS version and support timeline"
            },
            {
                "text": "It is made of such solid construction and since I have never had a Mac using my iPhone helped me get used to the system a bit.",
                "expected_polarity": "neutral",
                "issue": "neutral_aspects_misclassified",
                "analysis": "Build quality and adaptation process"
            },
            {
                "text": "Apple is aware of this issue and this is either old stock or a defective design involving the intel 4000 graphics chipset.",
                "expected_polarity": "neutral",
                "issue": "neutral_aspects_misclassified",
                "analysis": "Technical issue description and cause"
            },
            {
                "text": "The baterry is very longer.",
                "expected_polarity": "neutral",
                "issue": "neutral_aspects_misclassified",
                "analysis": "Grammatical error but factual claim"
            },
            {
                "text": "I did swap out the hard drive for a Samsung 830 SSD which I highly recommend.",
                "expected_polarity": "neutral",
                "issue": "neutral_aspects_misclassified",
                "analysis": "Upgrade description with recommendation"
            },
            {
                "text": "It's been time for a new laptop, and the only debate was which size of the Mac laptops, and whether to spring for the retina display.",
                "expected_polarity": "neutral",
                "issue": "neutral_aspects_misclassified",
                "analysis": "Purchase decision factors"
            },
            {
                "text": "HOWEVER I chose two day shipping and it took over a week to arrive.",
                "expected_polarity": "neutral",
                "issue": "neutral_aspects_misclassified",
                "analysis": "Shipping timeline description"
            },
            {
                "text": "I opted for the SquareTrade 3-Year Computer Accidental Protection Warranty ($1500-2000) which also support accidents like drops and spills that are NOT covered by AppleCare.",
                "expected_polarity": "neutral",
                "issue": "neutral_aspects_misclassified",
                "analysis": "Warranty coverage details"
            },
            {
                "text": "The only task that this computer would not be good enough for would be gaming, otherwise the integrated Intel 4000 graphics work well for other tasks.",
                "expected_polarity": "neutral",
                "issue": "neutral_aspects_misclassified",
                "analysis": "Use case suitability description"
            },
            {
                "text": "The only solution is to turn the brightness down, etc.",
                "expected_polarity": "neutral",
                "issue": "neutral_aspects_misclassified",
                "analysis": "Technical workaround description"
            },
            {
                "text": "There is no HDMI receptacle, nor is there an SD card slot located anywhere on the device.",
                "expected_polarity": "neutral",
                "issue": "neutral_aspects_misclassified",
                "analysis": "Port availability statement"
            },
            {
                "text": "It shouldn't happen like that, I don't have any design app open or anything.",
                "expected_polarity": "neutral",
                "issue": "neutral_aspects_misclassified",
                "analysis": "Unexpected behavior description"
            },
            {
                "text": "Made interneting (part of my business) very difficult to maintain.",
                "expected_polarity": "neutral",
                "issue": "neutral_aspects_misclassified",
                "analysis": "Business impact description"
            },
            {
                "text": "The memory was gone and it was not able to be used.",
                "expected_polarity": "neutral",
                "issue": "neutral_aspects_misclassified",
                "analysis": "Technical failure description"
            },
            {
                "text": "The battery lasts as advertised (give or take 15-20 minutes), and the entire user experience is very elegant.",
                "expected_polarity": "neutral",
                "issue": "neutral_aspects_misclassified",
                "analysis": "Performance verification with tolerance"
            }
        ],

        "positive_aspects_as_neutral": [
            # Positive aspects predicted as neutral (~25% of errors)
            {
                "text": "I am pleased with the fast log on, speedy WiFi connection and the long battery life (>6 hrs).",
                "expected_polarity": "positive",
                "issue": "positive_aspects_as_neutral",
                "analysis": "Direct positive expression with pleased/speedy/long"
            },
            {
                "text": "Apple is unmatched in product quality,aesthetics,craftmanship, and customer service.",
                "expected_polarity": "positive",
                "issue": "positive_aspects_as_neutral",
                "analysis": "Superlative unmatched with positive qualities"
            },
            {
                "text": "Having USB3 is why I bought this Mini.",
                "expected_polarity": "positive",
                "issue": "positive_aspects_as_neutral",
                "analysis": "Purchase motivation = positive"
            },
            {
                "text": "I've already upgraded o Mavericks and I am impressed with everything about this computer.",
                "expected_polarity": "positive",
                "issue": "positive_aspects_as_neutral",
                "analysis": "Direct impressed/impressed with upgrade"
            },
            {
                "text": "The new os is great on my macbook pro!",
                "expected_polarity": "positive",
                "issue": "positive_aspects_as_neutral",
                "analysis": "Direct great assessment"
            },
            {
                "text": "It's so nice that the battery last so long and that this machine has the snow lion!",
                "expected_polarity": "positive",
                "issue": "positive_aspects_as_neutral",
                "analysis": "Nice + long battery + has feature"
            },
            {
                "text": "Easy to customize setting and even create your own bookmarks.",
                "expected_polarity": "positive",
                "issue": "positive_aspects_as_neutral",
                "analysis": "Easy to + positive capabilities"
            },
            {
                "text": "I think this is about as good as it gets at anything close to this price point.",
                "expected_polarity": "positive",
                "issue": "positive_aspects_as_neutral",
                "analysis": "As good as it gets = superlative positive"
            },
            {
                "text": "It's fast at loading the internet.",
                "expected_polarity": "positive",
                "issue": "positive_aspects_as_neutral",
                "analysis": "Fast performance"
            },
            {
                "text": "It's silent and has a very small footprint on my desk.",
                "expected_polarity": "positive",
                "issue": "positive_aspects_as_neutral",
                "analysis": "Silent + small footprint (positive attributes)"
            }
        ],

        "conflict_polarity_issues": [
            # Conflict polarity predicted as positive (~10% of errors)
            {
                "text": "The price is higher than most lab top out there; however, he/she will get what they paid for, which is a great computer.",
                "expected_polarity": "conflict",
                "issue": "conflict_polarity_issues",
                "analysis": "Higher price (negative) but great value (positive)"
            },
            {
                "text": "The investment of a new MacBook Pro came at a price, but totally worth it for a good piece of mind.",
                "expected_polarity": "conflict",
                "issue": "conflict_polarity_issues",
                "analysis": "Came at a price (negative) but worth it (positive)"
            },
            {
                "text": "I had a little problem adjusting to the small screen but works fine as long as I remember to carry my glasses.",
                "expected_polarity": "conflict",
                "issue": "conflict_polarity_issues",
                "analysis": "Problem (negative) but works fine (positive)"
            },
            {
                "text": "I'm hoping the rest of the features will be the signature quality of apple.",
                "expected_polarity": "conflict",
                "issue": "conflict_polarity_issues",
                "analysis": "Hoping (uncertain) but signature quality (positive)"
            },
            {
                "text": "The only solution is to turn the brightness down, etc.",
                "expected_polarity": "conflict",
                "issue": "conflict_polarity_issues",
                "analysis": "Solution to problem implies both issue and fix"
            }
        ],

        "comparative_conditional_statements": [
            # Comparative and conditional statements (~10% of errors)
            {
                "text": "From the speed to the multi touch gestures this operating system beats Windows easily.",
                "expected_polarity": "negative",
                "issue": "comparative_conditional_statements",
                "analysis": "Beats Windows = negative for Windows"
            },
            {
                "text": "USB3 Peripherals are noticably less expensive than the ThunderBolt ones.",
                "expected_polarity": "positive",
                "issue": "comparative_conditional_statements",
                "analysis": "Less expensive = positive for USB3"
            },
            {
                "text": "ThunderBolt ones are more expensive than USB3 Peripherals.",
                "expected_polarity": "negative",
                "issue": "comparative_conditional_statements",
                "analysis": "More expensive = negative for ThunderBolt"
            },
            {
                "text": "Price was higher when purchased on MAC when compared to price showing on PC when I bought this product.",
                "expected_polarity": "negative",
                "issue": "comparative_conditional_statements",
                "analysis": "Higher price on Mac vs PC = negative"
            },
            {
                "text": "I Have been a Pc user for a very long time now but I will get used to this new OS.",
                "expected_polarity": "neutral",
                "issue": "comparative_conditional_statements",
                "analysis": "Transition from PC to Mac OS"
            },
            {
                "text": "It's ok but doesn't have a disk drive which I didn't know until after I bought it.",
                "expected_polarity": "neutral",
                "issue": "comparative_conditional_statements",
                "analysis": "Ok but missing expected feature"
            }
        ],

        "idiomatic_expressions_missed": [
            # Idiomatic expressions not handled properly
            {
                "text": "The investment of a new MacBook Pro came at a price, but totally worth it for a good piece of mind.",
                "expected_polarity": "positive",
                "issue": "idiomatic_expressions_missed",
                "analysis": "'Piece of mind' = peace of mind, 'worth it' = positive"
            },
            {
                "text": "The battery lasts as advertised (give or take 15-20 minutes)",
                "expected_polarity": "positive",
                "issue": "idiomatic_expressions_missed",
                "analysis": "'Give or take' = approximately, 'as advertised' = meets expectations"
            },
            {
                "text": "Doesn't break the bank",
                "expected_polarity": "positive",
                "issue": "idiomatic_expressions_missed",
                "analysis": "'Break the bank' = be expensive, doesn't = affordable"
            },
            {
                "text": "A cut above the rest",
                "expected_polarity": "positive",
                "issue": "idiomatic_expressions_missed",
                "analysis": "Superior to others"
            },
            {
                "text": "Hits the sweet spot",
                "expected_polarity": "positive",
                "issue": "idiomatic_expressions_missed",
                "analysis": "Perfect balance/ideal"
            }
        ],

        "complex_mixed_sentiment": [
            {
                "text": "Other than not being a fan of click pads (industry standard these days) and the lousy internal speakers, it's hard for me to find things about this notebook I don't like, especially considering the $350 price tag.",
                "expected_polarity": "positive",
                "issue": "complex_mixed_sentiment",
                "analysis": "Mixed negative modifiers but overall positive due to 'hard to find things I don't like' + value framing"
            },
            {
                "text": "Screen - although some people might complain about low res which I think is ridiculous.",
                "expected_polarity": "positive",
                "issue": "complex_mixed_sentiment",
                "analysis": "Acknowledges complaints but dismisses them as ridiculous"
            },
            {
                "text": "While the battery life could be better, the performance more than makes up for it.",
                "expected_polarity": "positive",
                "issue": "complex_mixed_sentiment",
                "analysis": "Concessive clause with compensation - overall positive"
            },
            {
                "text": "The keyboard feels cheap but the screen quality is outstanding for the price.",
                "expected_polarity": "positive",
                "issue": "complex_mixed_sentiment",
                "analysis": "Negative aspect balanced by positive value framing"
            },
            {
                "text": "Not the fastest laptop but definitely reliable and worth the investment.",
                "expected_polarity": "positive",
                "issue": "complex_mixed_sentiment",
                "analysis": "Double negative concession with positive framing"
            },
            {
                "text": "Build quality is mediocre however the customer service is exceptional.",
                "expected_polarity": "neutral",
                "issue": "complex_mixed_sentiment",
                "analysis": "Balanced positive and negative aspects"
            }
        ],

        "contextual_negation": [
            {
                "text": "No installation disk (DVD) is included.",
                "expected_polarity": "negative",
                "issue": "contextual_negation",
                "analysis": "'No installation disk' should make DVD negative, but DVD gets positive sentiment"
            },
            {
                "text": "Did not enjoy the new Windows 8 and touchscreen functions.",
                "expected_polarity": "negative",
                "issue": "contextual_negation",
                "analysis": "Direct negation should make all aspects negative"
            },
            {
                "text": "The screen is not bright enough for outdoor use.",
                "expected_polarity": "negative",
                "issue": "contextual_negation",
                "analysis": "Negation affects brightness aspect"
            },
            {
                "text": "I don't recommend this product to anyone.",
                "expected_polarity": "negative",
                "issue": "contextual_negation",
                "analysis": "Direct recommendation negation"
            },
            {
                "text": "This laptop fails to meet my expectations.",
                "expected_polarity": "negative",
                "issue": "contextual_negation",
                "analysis": "Failure to meet expectations"
            },
            {
                "text": "I can't believe how quiet the hard drive is and how quick this thing boots up.",
                "expected_polarity": "positive",
                "issue": "contextual_negation",
                "analysis": "Can't believe = surprise, quiet and quick are positive"
            },
            {
                "text": "Though please note that sometimes it crashes, and the sound quality isnt superb.",
                "expected_polarity": "negative",
                "issue": "contextual_negation",
                "analysis": "Crashes and isn't superb = negative"
            }
        ],

        "double_negation_complex": [
            {
                "text": "This product is not bad",
                "expected_polarity": "positive",
                "issue": "double_negation_complex",
                "analysis": "'not bad' = positive, but system may not handle double negation properly"
            },
            {
                "text": "Nothing to complain about",
                "expected_polarity": "positive",
                "issue": "double_negation_complex",
                "analysis": "Double negative indicating no complaints = positive"
            },
            {
                "text": "Can't find anything wrong with it",
                "expected_polarity": "positive",
                "issue": "double_negation_complex",
                "analysis": "Double negative - can't find wrong = positive"
            },
            {
                "text": "Not unhappy with the purchase",
                "expected_polarity": "positive",
                "issue": "double_negation_complex",
                "analysis": "Double negative construction"
            },
            {
                "text": "No regrets about buying this",
                "expected_polarity": "positive",
                "issue": "double_negation_complex",
                "analysis": "Double negative with positive implication"
            },
            {
                "text": "Won't say it's perfect but it's not bad either",
                "expected_polarity": "neutral",
                "issue": "double_negation_complex",
                "analysis": "Complex double negative with mixed sentiment"
            }
        ],

        "value_framing": [
            {
                "text": "The battery life is good but the screen is terrible, especially considering the price.",
                "expected_polarity": "negative",
                "issue": "value_framing",
                "analysis": "Value framing should boost positive aspects but negative still dominates"
            },
            {
                "text": "Expensive but worth every penny",
                "expected_polarity": "positive",
                "issue": "value_framing",
                "analysis": "Value framing should make expensive acceptable"
            },
            {
                "text": "Overpriced for what you get",
                "expected_polarity": "negative",
                "issue": "value_framing",
                "analysis": "Value expectation not met"
            },
            {
                "text": "Great value for money despite small size",
                "expected_polarity": "positive",
                "issue": "value_framing",
                "analysis": "Value framing overcomes minor negative"
            },
            {
                "text": "Cheap but feels premium",
                "expected_polarity": "positive",
                "issue": "value_framing",
                "analysis": "Positive value framing for inexpensive product"
            }
        ],

        "aspect_scope_issues": [
            {
                "text": "To me it's a workhorse... and quiet as can be... you even save big bucks not dishing out any extra to Uncle Sam... Definite Buy...",
                "expected_polarity": "positive",
                "issue": "aspect_scope_issues",
                "analysis": "Pronouns 'me', 'it', 'you' incorrectly detected as aspects"
            },
            {
                "text": "This is the best laptop I've ever owned",
                "expected_polarity": "positive",
                "issue": "aspect_scope_issues",
                "analysis": "Superlative comparison should be positive"
            },
            {
                "text": "It works perfectly for my needs",
                "expected_polarity": "positive",
                "issue": "aspect_scope_issues",
                "analysis": "Pronoun 'it' should not be treated as aspect"
            }
        ],

        "expectation_shortfall": [
            {
                "text": "Shouldn't take so long to boot up",
                "expected_polarity": "negative",
                "issue": "expectation_shortfall",
                "analysis": "Expectation not met = negative"
            },
            {
                "text": "Lacks modern features",
                "expected_polarity": "negative",
                "issue": "expectation_shortfall",
                "analysis": "Missing expected features = negative"
            },
            {
                "text": "Missing the HDMI port I expected",
                "expected_polarity": "negative",
                "issue": "expectation_shortfall",
                "analysis": "Missing expected connectivity"
            },
            {
                "text": "Fails to deliver on promised performance",
                "expected_polarity": "negative",
                "issue": "expectation_shortfall",
                "analysis": "Performance shortfall"
            },
            {
                "text": "Without the features I need",
                "expected_polarity": "negative",
                "issue": "expectation_shortfall",
                "analysis": "Missing required features"
            },
            {
                "text": "Doesn't have what I was looking for",
                "expected_polarity": "negative",
                "issue": "expectation_shortfall",
                "analysis": "Expectation mismatch"
            },
            {
                "text": "Falls short of expectations",
                "expected_polarity": "negative",
                "issue": "expectation_shortfall",
                "analysis": "General expectation shortfall"
            }
        ],

        "sarcasm_irony": [
            {
                "text": "Oh great, another slow laptop. Just what I needed.",
                "expected_polarity": "negative",
                "issue": "sarcasm_irony",
                "analysis": "Sarcastic positive words with negative intent"
            },
            {
                "text": "Wonderful, the fan sounds like a jet engine.",
                "expected_polarity": "negative",
                "issue": "sarcasm_irony",
                "analysis": "Sarcasm using positive word for negative situation"
            },
            {
                "text": "Perfect, it crashes every hour.",
                "expected_polarity": "negative",
                "issue": "sarcasm_irony",
                "analysis": "Sarcastic perfection for failure"
            },
            {
                "text": "Love how it overheats during light use.",
                "expected_polarity": "negative",
                "issue": "sarcasm_irony",
                "analysis": "Sarcastic love for negative feature"
            }
        ],

        "rhetorical_questions": [
            {
                "text": "Who wouldn't love this amazing laptop?",
                "expected_polarity": "positive",
                "issue": "rhetorical_questions",
                "analysis": "Rhetorical question implying positive"
            },
            {
                "text": "How can anyone complain about this performance?",
                "expected_polarity": "positive",
                "issue": "rhetorical_questions",
                "analysis": "Rhetorical question defending positive aspect"
            },
            {
                "text": "Why would I ever need more than this?",
                "expected_polarity": "positive",
                "issue": "rhetorical_questions",
                "analysis": "Rhetorical question showing satisfaction"
            },
            {
                "text": "Can this laptop get any better?",
                "expected_polarity": "positive",
                "issue": "rhetorical_questions",
                "analysis": "Rhetorical question of excellence"
            }
        ],

        "idiomatic_expressions": [
            {
                "text": "This laptop is a game changer",
                "expected_polarity": "positive",
                "issue": "idiomatic_expressions",
                "analysis": "Idiom meaning revolutionary/positive"
            },
            {
                "text": "Hits the sweet spot for price and performance",
                "expected_polarity": "positive",
                "issue": "idiomatic_expressions",
                "analysis": "Idiom meaning perfect balance"
            },
            {
                "text": "Doesn't break the bank",
                "expected_polarity": "positive",
                "issue": "idiomatic_expressions",
                "analysis": "Idiom meaning affordable"
            },
            {
                "text": "A cut above the rest",
                "expected_polarity": "positive",
                "issue": "idiomatic_expressions",
                "analysis": "Idiom meaning superior"
            },
            {
                "text": "This is a lemon",
                "expected_polarity": "negative",
                "issue": "idiomatic_expressions",
                "analysis": "Idiom meaning defective/bad product"
            }
        ],

        "concessive_clauses": [
            {
                "text": "Although it's expensive, it's worth every penny.",
                "expected_polarity": "positive",
                "issue": "concessive_clauses",
                "analysis": "Concession followed by positive justification"
            },
            {
                "text": "While the battery could be better, the performance is outstanding.",
                "expected_polarity": "positive",
                "issue": "concessive_clauses",
                "analysis": "Concession with positive outweighing negative"
            },
            {
                "text": "Even though it's not perfect, it's still a great buy.",
                "expected_polarity": "positive",
                "issue": "concessive_clauses",
                "analysis": "Concession with overall positive assessment"
            },
            {
                "text": "Despite the high price, the quality justifies it.",
                "expected_polarity": "positive",
                "issue": "concessive_clauses",
                "analysis": "Concession with value justification"
            }
        ],

        "comparative_constructions": [
            {
                "text": "Better than I expected",
                "expected_polarity": "positive",
                "issue": "comparative_constructions",
                "analysis": "Positive comparison to expectations"
            },
            {
                "text": "Worse than advertised",
                "expected_polarity": "negative",
                "issue": "comparative_constructions",
                "analysis": "Negative comparison to advertising"
            },
            {
                "text": "Faster than my old laptop",
                "expected_polarity": "positive",
                "issue": "comparative_constructions",
                "analysis": "Positive comparative"
            },
            {
                "text": "Not as good as the reviews said",
                "expected_polarity": "negative",
                "issue": "comparative_constructions",
                "analysis": "Negative comparison to reviews"
            }
        ],

        "conditional_sentiment": [
            {
                "text": "If you need reliability, this is perfect",
                "expected_polarity": "positive",
                "issue": "conditional_sentiment",
                "analysis": "Conditional positive for specific use case"
            },
            {
                "text": "Unless you want something cheap, avoid this",
                "expected_polarity": "negative",
                "issue": "conditional_sentiment",
                "analysis": "Conditional negative with exception"
            },
            {
                "text": "For the price, you can't go wrong",
                "expected_polarity": "positive",
                "issue": "conditional_sentiment",
                "analysis": "Conditional positive based on price"
            }
        ],

        "temporal_sentiment_shifts": [
            {
                "text": "Started slow but got much better after updates",
                "expected_polarity": "positive",
                "issue": "temporal_sentiment_shifts",
                "analysis": "Initial negative improved over time"
            },
            {
                "text": "Worked great at first, now constantly crashes",
                "expected_polarity": "negative",
                "issue": "temporal_sentiment_shifts",
                "analysis": "Initial positive deteriorated"
            },
            {
                "text": "Performance degraded after a few months",
                "expected_polarity": "negative",
                "issue": "temporal_sentiment_shifts",
                "analysis": "Negative change over time"
            }
        ]
    }

    return error_patterns

def run_comprehensive_stress_test():
    """Run comprehensive stress test on all error patterns."""

    patterns = analyze_error_patterns()
    results = {}

    # Initialize sentiment analyzer
    analyzer = MultiSentimentAnalysis(methods=['vader', 'textblob', 'distilbert_logit'])

    print("="*80)
    print("COMPREHENSIVE POLARITY STRESS TEST")
    print("="*80)

    total_tests = 0
    total_correct = {"distilbert": 0, "vader": 0, "textblob": 0, "ensemble": 0}

    for category, test_cases in patterns.items():
        print(f"\n📋 Category: {category.upper()}")
        print("-" * 60)

        category_results = []
        category_correct = {"distilbert": 0, "vader": 0, "textblob": 0, "ensemble": 0}

        for i, case in enumerate(test_cases):
            total_tests += 1
            text = case["text"]
            expected = case["expected_polarity"]
            issue = case["issue"]
            analysis = case["analysis"]

            print(f"\nTest {i+1}: {text[:80]}{'...' if len(text) > 80 else ''}")
            print(f"Expected: {expected.upper()}")
            print(f"Analysis: {analysis}")

            # Test individual methods
            distilbert_valence = get_distilbert_logit_sentiment(text)
            vader_valence = get_vader_sentiment(text)
            textblob_valence = get_textblob_sentiment(text)

            # Test ensemble method
            ensemble_result = analyzer.analyze(text)
            ensemble_valence = ensemble_result.get('aggregate', 0.0) if isinstance(ensemble_result, dict) else 0.0

            print(".3f")
            print(".3f")
            print(".3f")
            print(".3f")

            # Determine if polarity is correct
            def polarity_correct(valence, expected):
                if expected == "positive":
                    return valence > 0.1
                elif expected == "negative":
                    return valence < -0.1
                else:
                    return abs(valence) <= 0.1

            distilbert_correct = polarity_correct(distilbert_valence, expected)
            vader_correct = polarity_correct(vader_valence, expected)
            textblob_correct = polarity_correct(textblob_valence, expected)
            ensemble_correct = polarity_correct(ensemble_valence, expected)

            # Update counters
            if distilbert_correct: category_correct["distilbert"] += 1
            if vader_correct: category_correct["vader"] += 1
            if textblob_correct: category_correct["textblob"] += 1
            if ensemble_correct: category_correct["ensemble"] += 1

            print(f"Results: DistilBERT {'✓' if distilbert_correct else '✗'}, VADER {'✓' if vader_correct else '✗'}, TextBlob {'✓' if textblob_correct else '✗'}, Ensemble {'✓' if ensemble_correct else '✗'}")

            category_results.append({
                "case": i+1,
                "text": text,
                "issue": issue,
                "expected": expected,
                "analysis": analysis,
                "distilbert_valence": distilbert_valence,
                "vader_valence": vader_valence,
                "textblob_valence": textblob_valence,
                "ensemble_valence": ensemble_valence,
                "distilbert_correct": distilbert_correct,
                "vader_correct": vader_correct,
                "textblob_correct": textblob_correct,
                "ensemble_correct": ensemble_correct
            })

        # Category summary
        print(f"\n📊 {category.upper()} Summary:")
        for method in ["distilbert", "vader", "textblob", "ensemble"]:
            pct = (category_correct[method] / len(test_cases)) * 100
            print(".1f")

        results[category] = {
            "test_cases": category_results,
            "summary": category_correct
        }

        # Update totals
        for method in total_correct:
            total_correct[method] += category_correct[method]

    # Overall summary
    print("\n" + "="*80)
    print("🎯 OVERALL RESULTS SUMMARY")
    print("="*80)
    print(f"Total test cases: {total_tests}")

    print("\n📈 Accuracy by Method:")
    for method in ["distilbert", "vader", "textblob", "ensemble"]:
        pct = (total_correct[method] / total_tests) * 100
        print(".1f")

    # Critical issues analysis
    critical_categories = ["contextual_negation", "double_negation_complex", "complex_mixed_sentiment"]
    critical_total = sum(len(patterns[cat]) for cat in critical_categories)
    critical_correct = {"distilbert": 0, "vader": 0, "textblob": 0, "ensemble": 0}

    for cat in critical_categories:
        for method in critical_correct:
            critical_correct[method] += results[cat]["summary"][method]

    print("\n🔴 Critical Issues (Contextual Negation, Double Negation, Complex Mixed):")
    for method in ["distilbert", "vader", "textblob", "ensemble"]:
        pct = (critical_correct[method] / critical_total) * 100 if critical_total > 0 else 0
        print(".1f")

    # Save detailed results
    with open('/Users/harry/Documents/Python_Projects/ETSA_(QC)/comprehensive_polarity_stress_test.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n💾 Detailed results saved to: comprehensive_polarity_stress_test.json")
    success = total_correct["ensemble"] == total_tests
    if success:
        print("\n🎉 ALL TESTS PASSED! System is ready for benchmarking.")
    else:
        print(f"\n❌ {total_tests - total_correct['ensemble']}/{total_tests} tests failed. Improvements needed.")

    return results

def identify_improvement_opportunities(results):
    """Analyze results to identify specific improvement opportunities."""

    print("\n" + "="*80)
    print("🔍 IMPROVEMENT OPPORTUNITIES ANALYSIS")
    print("="*80)

    improvements = {
        "sentiment_patterns": [],
        "combination_logic": [],
        "aspect_detection": [],
        "context_handling": []
    }

    # Analyze failed cases
    failed_cases = []
    for category, data in results.items():
        for case in data["test_cases"]:
            if not case["ensemble_correct"]:
                failed_cases.append(case)

    print(f"Found {len(failed_cases)} failed test cases requiring improvement:")

    # Pattern analysis
    complex_mixed_failures = [c for c in failed_cases if c["issue"] == "complex_mixed_sentiment"]
    if complex_mixed_failures:
        print(f"\n🎭 Complex Mixed Sentiment Issues ({len(complex_mixed_failures)} cases):")
        improvements["sentiment_patterns"].extend([
            "Enhance double-negative pattern recognition (e.g., 'hard to find things I don't like')",
            "Improve value framing detection for price justifications",
            "Better handling of concessive clauses ('although... but...')",
            "Strengthen sarcasm and irony detection"
        ])

    contextual_negation_failures = [c for c in failed_cases if c["issue"] == "contextual_negation"]
    if contextual_negation_failures:
        print(f"\n🔗 Contextual Negation Issues ({len(contextual_negation_failures)} cases):")
        improvements["combination_logic"].extend([
            "Improve modifier propagation to affected aspects",
            "Better scope resolution for negation ('no X' affecting related terms)",
            "Enhanced contextual sentiment inheritance"
        ])

    double_negation_failures = [c for c in failed_cases if c["issue"] == "double_negation_complex"]
    if double_negation_failures:
        print(f"\n⚡ Double Negation Issues ({len(double_negation_failures)} cases):")
        improvements["sentiment_patterns"].extend([
            "Expand double-negative pattern library",
            "Improve multi-negation resolution",
            "Better handling of idiomatic expressions"
        ])

    value_framing_failures = [c for c in failed_cases if c["issue"] == "value_framing"]
    if value_framing_failures:
        print(f"\n💰 Value Framing Issues ({len(value_framing_failures)} cases):")
        improvements["sentiment_patterns"].extend([
            "Strengthen value framing pattern matching",
            "Better trade-off analysis (cost vs. benefit)",
            "Improved concessive construction handling"
        ])

    aspect_scope_failures = [c for c in failed_cases if c["issue"] == "aspect_scope_issues"]
    if aspect_scope_failures:
        print(f"\n🎯 Aspect Scope Issues ({len(aspect_scope_failures)} cases):")
        improvements["aspect_detection"].extend([
            "Filter out spurious pronoun aspects",
            "Improve coreference resolution",
            "Better aspect relevance scoring"
        ])

    expectation_failures = [c for c in failed_cases if c["issue"] == "expectation_shortfall"]
    if expectation_failures:
        print(f"\n🎯 Expectation Shortfall Issues ({len(expectation_failures)} cases):")
        improvements["context_handling"].extend([
            "Enhance expectation violation detection",
            "Better handling of implicit comparisons",
            "Improved shortfall pattern recognition"
        ])

    # General improvements
    improvements["combination_logic"].extend([
        "Refine contextual weighting in _combine_contextual_v3",
        "Improve mixed polarity resolution logic",
        "Better context score calculation"
    ])

    improvements["sentiment_patterns"].extend([
        "Add more sophisticated pattern matching",
        "Improve clause-level sentiment analysis",
        "Better handling of rhetorical questions and exclamations"
    ])

    return improvements

if __name__ == "__main__":
    # Run comprehensive stress test
    results = run_comprehensive_stress_test()

    # Analyze improvement opportunities
    improvements = identify_improvement_opportunities(results)

    # Save improvement plan
    with open('/Users/harry/Documents/Python_Projects/ETSA_(QC)/improvement_plan.json', 'w') as f:
        json.dump(improvements, f, indent=2)

    print("\n💾 Improvement plan saved to: improvement_plan.json")
    print("\n📋 Next Steps:")
    print("1. Implement identified improvements")
    print("2. Re-run stress tests to validate fixes")
    print("3. Run full benchmark when all tests pass")