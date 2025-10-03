"""
Comprehensive modifier extraction test harness for optimization.
Tests modifier extraction and sentiment calculation on hand-crafted failure cases.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.modifier_e import GemmaModifierExtractor
from pipeline.graph import RelationGraph
from pipeline.sentiment_analysis import MultiSentimentAnalysis
import pipeline.config


@dataclass
class ModifierTestCase:
    """Test case for modifier extraction"""
    name: str
    sentence: str
    entity: str
    expected_modifiers: List[str]  # Modifiers that MUST be extracted
    expected_polarity: str  # positive, negative, or neutral
    category: str  # Category of test (entity_text, cross_clause, weak_sentiment, etc.)


# Comprehensive test suite based on actual failure cases
TEST_CASES = [
    # CATEGORY 1: Entity-text modifiers (adjectives IN the entity text itself)
    ModifierTestCase(
        name="fresh_juices",
        sentence="They serve fresh juices that taste incredible.",
        entity="fresh juices",
        expected_modifiers=["fresh", "taste incredible"],
        expected_polarity="positive",
        category="entity_text_modifier"
    ),
    ModifierTestCase(
        name="burnt_flavor",
        sentence="The burnt flavor ruined the entire dish.",
        entity="burnt flavor",
        expected_modifiers=["burnt", "ruined the entire dish"],
        expected_polarity="negative",
        category="entity_text_modifier"
    ),
    ModifierTestCase(
        name="quiet_hard_drive",
        sentence="The quiet hard drive never makes any noise.",
        entity="quiet hard drive",
        expected_modifiers=["quiet", "never makes any noise"],
        expected_polarity="positive",
        category="entity_text_modifier"
    ),
    ModifierTestCase(
        name="slow_performance",
        sentence="It was also suffering from hardware issues, relatively slow performance and shortening battery lifetime.",
        entity="slow performance",
        expected_modifiers=["slow", "relatively"],
        expected_polarity="negative",
        category="entity_text_modifier"
    ),
    ModifierTestCase(
        name="cool_features",
        sentence="working with Mac is so much easier, so many cool features.",
        entity="many cool features",
        expected_modifiers=["cool", "many"],
        expected_polarity="positive",
        category="entity_text_modifier"
    ),
    ModifierTestCase(
        name="other_features",
        sentence="Has all the other features I wanted including a VGA port, HDMI, ethernet and 3 USB ports.",
        entity="other features",
        expected_modifiers=["I wanted"],
        expected_polarity="positive",
        category="entity_text_modifier"
    ),
    
    # CATEGORY 2: Cross-clause sentiment propagation
    ModifierTestCase(
        name="cross_clause_pleased",
        sentence="I am pleased with the appearance and functionality.",
        entity="appearance and functionality",
        expected_modifiers=["I am pleased"],
        expected_polarity="positive",
        category="cross_clause"
    ),
    ModifierTestCase(
        name="cross_clause_disappointed",
        sentence="I was disappointed in the battery life and screen quality.",
        entity="battery life",
        expected_modifiers=["I was disappointed"],
        expected_polarity="negative",
        category="cross_clause"
    ),
    ModifierTestCase(
        name="cross_clause_love",
        sentence="I love the form factor.",
        entity="form factor",
        expected_modifiers=["I love"],
        expected_polarity="positive",
        category="cross_clause"
    ),
    
    # CATEGORY 3: Weak/hedged sentiment (should still extract if directional)
    ModifierTestCase(
        name="weak_sentiment_usually_good",
        sentence="Service usually pretty good but can be slow.",
        entity="Service",
        expected_modifiers=["usually pretty good"],
        expected_polarity="positive",
        category="weak_sentiment"
    ),
    ModifierTestCase(
        name="weak_sentiment_kind_of_nice",
        sentence="The display is kind of nice and vibrant.",
        entity="display",
        expected_modifiers=["kind of nice", "vibrant"],
        expected_polarity="positive",
        category="weak_sentiment"
    ),
    
    # CATEGORY 4: Prepositional complements
    ModifierTestCase(
        name="prep_complement_with_staff",
        sentence="Their service with incredibly attentive staff made checkout effortless.",
        entity="service",
        expected_modifiers=["with incredibly attentive staff", "made checkout effortless"],
        expected_polarity="positive",
        category="prepositional_complement"
    ),
    
    # CATEGORY 5: Coordinated entities sharing sentiment
    ModifierTestCase(
        name="coordinated_clientele",
        sentence="The bar was full of rowdy, loud-mouthed commuters and aggressive clientele.",
        entity="clientele",
        expected_modifiers=["aggressive"],
        expected_polarity="negative",
        category="coordinated_entities"
    ),
    
    # CATEGORY 6: Consequence clauses
    ModifierTestCase(
        name="consequence_left_waiting",
        sentence="Support left me waiting forty minutes on hold.",
        entity="Support",
        expected_modifiers=["left me waiting forty minutes on hold"],
        expected_polarity="negative",
        category="consequence_clause"
    ),
    ModifierTestCase(
        name="consequence_made_effortless",
        sentence="The interface made navigation effortless.",
        entity="interface",
        expected_modifiers=["made navigation effortless"],
        expected_polarity="positive",
        category="consequence_clause"
    ),
    
    # CATEGORY 7: Negation and absence
    ModifierTestCase(
        name="negation_no_graphics",
        sentence="There is no dedicated graphics card.",
        entity="dedicated graphics card",
        expected_modifiers=["no dedicated graphics card"],
        expected_polarity="negative",
        category="negation"
    ),
    ModifierTestCase(
        name="negation_never_boots",
        sentence="Software updates never installed correctly.",
        entity="Software updates",
        expected_modifiers=["never installed correctly"],
        expected_polarity="negative",
        category="negation"
    ),
    
    # CATEGORY 8: Value/price language
    ModifierTestCase(
        name="value_worth_penny",
        sentence="The battery lasts as advertised and is worth every penny.",
        entity="battery",
        expected_modifiers=["lasts as advertised", "is worth every penny"],
        expected_polarity="positive",
        category="value_language"
    ),
    
    # CATEGORY 9: Comparison and relative language
    ModifierTestCase(
        name="comparison_beats_windows",
        sentence="From the speed to the multi touch gestures this operating system beats Windows easily.",
        entity="operating system",
        expected_modifiers=["beats Windows easily"],
        expected_polarity="positive",
        category="comparison"
    ),
    ModifierTestCase(
        name="comparison_better_than",
        sentence="The new model is significantly better than the old one.",
        entity="new model",
        expected_modifiers=["significantly better than the old one"],
        expected_polarity="positive",
        category="comparison"
    ),
    
    # CATEGORY 10: Edge cases from actual failures
    ModifierTestCase(
        name="edge_safari_browser",
        sentence="Web browsing is very quick with Safari browser.",
        entity="Safari browser",
        expected_modifiers=["Web browsing is very quick"],
        expected_polarity="positive",
        category="edge_case"
    ),
    ModifierTestCase(
        name="edge_practicality",
        sentence="if yo like practicality this is the laptop for you.",
        entity="practicality",
        expected_modifiers=["this is the laptop for you"],
        expected_polarity="positive",
        category="edge_case"
    ),
    ModifierTestCase(
        name="edge_capabilities",
        sentence="The MBP is beautiful has many wonderful capabilities.",
        entity="capabilities",
        expected_modifiers=["many", "wonderful"],
        expected_polarity="positive",
        category="entity_text_modifier"
    ),
    
    # CATEGORY 11: Verbal predicates with negation (NEW - from benchmark failures)
    ModifierTestCase(
        name="verbal_not_helpful",
        sentence="The technical support was not helpful as well.",
        entity="technical support",
        expected_modifiers=["was not helpful"],
        expected_polarity="negative",
        category="verbal_negation"
    ),
    ModifierTestCase(
        name="verbal_was_crappy",
        sentence="The sound was crappy even when you turn up the volume.",
        entity="sound",
        expected_modifiers=["was crappy"],
        expected_polarity="negative",
        category="verbal_predicate"
    ),
    ModifierTestCase(
        name="verbal_dropping_support",
        sentence="Unfortunately, Microsoft is dropping support next April.",
        entity="support",
        expected_modifiers=["is dropping", "Unfortunately"],
        expected_polarity="negative",
        category="verbal_predicate"
    ),
    
    # CATEGORY 12: Copula constructions (NEW - from benchmark failures)
    ModifierTestCase(
        name="copula_looks_nice",
        sentence="The bar looks nice and the atmosphere is relaxing.",
        entity="bar",
        expected_modifiers=["looks nice"],
        expected_polarity="positive",
        category="copula"
    ),
    ModifierTestCase(
        name="copula_seems_good",
        sentence="The restaurant seems good based on the reviews.",
        entity="restaurant",
        expected_modifiers=["seems good"],
        expected_polarity="positive",
        category="copula"
    ),
    ModifierTestCase(
        name="copula_appears_broken",
        sentence="The screen appears broken or defective.",
        entity="screen",
        expected_modifiers=["appears broken"],
        expected_polarity="negative",
        category="copula"
    ),
    
    # CATEGORY 13: Sarcasm and rhetorical questions (NEW)
    ModifierTestCase(
        name="sarcasm_like_this",
        sentence="How can they stay in business with service like this?",
        entity="service",
        expected_modifiers=["like this"],
        expected_polarity="negative",
        category="sarcasm"
    ),
    ModifierTestCase(
        name="sarcasm_call_this",
        sentence="You call this customer support?",
        entity="customer support",
        expected_modifiers=["You call this"],
        expected_polarity="negative",
        category="sarcasm"
    ),
    
    # CATEGORY 14: Idioms - Negative (NEW)
    ModifierTestCase(
        name="idiom_gave_lip",
        sentence="The waitress gave her lip about sending it back.",
        entity="waitress",
        expected_modifiers=["gave her lip about"],
        expected_polarity="negative",
        category="idiom_negative"
    ),
    ModifierTestCase(
        name="idiom_pain_in_neck",
        sentence="The installation process was a pain in the neck.",
        entity="installation process",
        expected_modifiers=["was a pain in the neck"],
        expected_polarity="negative",
        category="idiom_negative"
    ),
    
    # CATEGORY 15: Idioms - Positive (NEW)
    ModifierTestCase(
        name="idiom_melts_in_mouth",
        sentence="The gnocchi literally melts in your mouth!",
        entity="gnocchi",
        expected_modifiers=["literally melts in your mouth"],
        expected_polarity="positive",
        category="idiom_positive"
    ),
    ModifierTestCase(
        name="idiom_piece_of_cake",
        sentence="The setup was a piece of cake.",
        entity="setup",
        expected_modifiers=["was a piece of cake"],
        expected_polarity="positive",
        category="idiom_positive"
    ),
    
    # CATEGORY 16: Double negation (NEW)
    ModifierTestCase(
        name="double_neg_not_hard",
        sentence="The interface is not hard to figure out.",
        entity="interface",
        expected_modifiers=["is not hard to figure out"],
        expected_polarity="positive",
        category="double_negation"
    ),
    ModifierTestCase(
        name="double_neg_not_bad",
        sentence="The performance is not bad for the price.",
        entity="performance",
        expected_modifiers=["is not bad"],
        expected_polarity="positive",
        category="double_negation"
    ),
    
    # CATEGORY 17: Absence construction (NEW)
    ModifierTestCase(
        name="absence_no_graphics",
        sentence="There is no dedicated graphics card in this laptop.",
        entity="graphics card",
        expected_modifiers=["There is no dedicated graphics card", "no"],
        expected_polarity="negative",
        category="absence"
    ),
    ModifierTestCase(
        name="absence_no_support",
        sentence="There is no customer support available.",
        entity="customer support",
        expected_modifiers=["There is no customer support available", "no"],
        expected_polarity="negative",
        category="absence"
    ),
    
    # CATEGORY 18: Price/value context (NEW)
    ModifierTestCase(
        name="price_should_be_included",
        sentence="For this price it should be included.",
        entity="price",
        expected_modifiers=["for this price it should be included"],
        expected_polarity="negative",
        category="price_context"
    ),
    ModifierTestCase(
        name="price_too_expensive",
        sentence="The laptop costs too much for what it offers.",
        entity="laptop",
        expected_modifiers=["costs too much for what it offers"],
        expected_polarity="negative",
        category="price_context"
    ),
    
    # CATEGORY 19: Context-dependent modifiers (NEW - challenging cases)
    ModifierTestCase(
        name="context_half_chicken",
        sentence="Half a chicken with a mountain of rice for $6.25.",
        entity="Half a chicken",
        expected_modifiers=["with a mountain of rice", "for $6.25"],
        expected_polarity="positive",
        category="context_dependent"
    ),
    ModifierTestCase(
        name="context_late_lunch",
        sentence="We went last Tuesday for a late lunch with friends.",
        entity="late lunch",
        expected_modifiers=["with friends"],
        expected_polarity="positive",
        category="context_dependent"
    ),
    
    # CATEGORY 21: Basic adjective predicates (from benchmark failures)
    ModifierTestCase(
        name="adjective_was_sweet",
        sentence="The foie gras was sweet and luscious.",
        entity="foie gras",
        expected_modifiers=["was sweet and luscious"],
        expected_polarity="positive",
        category="basic_adjective"
    ),
    ModifierTestCase(
        name="adjective_looks_nice",
        sentence="The bar looks nice and welcoming.",
        entity="bar",
        expected_modifiers=["looks nice and welcoming"],
        expected_polarity="positive",
        category="basic_adjective"
    ),
    ModifierTestCase(
        name="adjective_is_beautiful",
        sentence="The MBP is beautiful has many wonderful capabilities.",
        entity="MBP",
        expected_modifiers=["is beautiful"],
        expected_polarity="positive",
        category="basic_adjective"
    ),
    
    # CATEGORY 22: Arrives + state descriptions
    ModifierTestCase(
        name="arrives_nice_hot",
        sentence="The soup always arrives nice and hot.",
        entity="soup",
        expected_modifiers=["arrives nice and hot"],
        expected_polarity="positive",
        category="verbal_state"
    ),
    ModifierTestCase(
        name="arrives_fresh",
        sentence="My coffee arrives fresh every morning.",
        entity="coffee",
        expected_modifiers=["arrives fresh"],
        expected_polarity="positive",
        category="verbal_state"
    ),
    
    # CATEGORY 23: Imperative recommendations (high value!)
    ModifierTestCase(
        name="imperative_must_try",
        sentence="You must try the Odessa stew!",
        entity="Odessa stew",
        expected_modifiers=["must try"],
        expected_polarity="positive",
        category="imperative"
    ),
    ModifierTestCase(
        name="imperative_try_the",
        sentence="Try the Peanut Butter Sorbet!",
        entity="Peanut Butter Sorbet",
        expected_modifiers=["Try"],
        expected_polarity="positive",
        category="imperative"
    ),
    ModifierTestCase(
        name="imperative_must_try_rabbit",
        sentence="You must try Odessa stew or Rabbit stew.",
        entity="Rabbit stew",
        expected_modifiers=["must try"],
        expected_polarity="positive",
        category="imperative"
    ),
    
    # CATEGORY 24: Superlatives "does best"
    ModifierTestCase(
        name="superlative_does_best",
        sentence="Stick to the items the place does best, brisket, ribs, wings.",
        entity="brisket",
        expected_modifiers=["does best"],
        expected_polarity="positive",
        category="superlative"
    ),
    ModifierTestCase(
        name="superlative_does_best_ribs",
        sentence="Stick to the items the place does best, brisket, ribs, wings.",
        entity="ribs",
        expected_modifiers=["does best"],
        expected_polarity="positive",
        category="superlative"
    ),
    
    # CATEGORY 25: Comparative value
    ModifierTestCase(
        name="comparative_better_price",
        sentence="Could have had better for 1/3 the price in Chinatown.",
        entity="price",
        expected_modifiers=["Could have had better for 1/3"],
        expected_polarity="negative",
        category="comparative"
    ),
    ModifierTestCase(
        name="comparative_3x_high",
        sentence="Anywhere else, the prices would be 3x as high!",
        entity="prices",
        expected_modifiers=["would be 3x as high"],
        expected_polarity="positive",
        category="comparative"
    ),
    
    # CATEGORY 26: Slang verbs ("dug")
    ModifierTestCase(
        name="slang_dug",
        sentence="Dug the blue bar area too.",
        entity="blue bar area",
        expected_modifiers=["Dug"],
        expected_polarity="positive",
        category="slang"
    ),
    ModifierTestCase(
        name="slang_really_dug",
        sentence="Really dug the outdoor seating area.",
        entity="outdoor seating area",
        expected_modifiers=["Really dug"],
        expected_polarity="positive",
        category="slang"
    ),
    
    # CATEGORY 27: "Hot spot" idiom
    ModifierTestCase(
        name="idiom_hot_spot",
        sentence="This is literally a hot spot when it comes to the food.",
        entity="food",
        expected_modifiers=["is literally a hot spot"],
        expected_polarity="positive",
        category="idiom_positive"
    ),
    
    # CATEGORY 28: Indirect modifiers "issues with"
    ModifierTestCase(
        name="indirect_major_issues",
        sentence="There are MAJOR issues with the touchpad which render the device nearly useless.",
        entity="touchpad",
        expected_modifiers=["MAJOR issues with", "render the device nearly useless"],
        expected_polarity="negative",
        category="indirect_negative"
    ),
    ModifierTestCase(
        name="indirect_problems_with",
        sentence="Serious problems with the keyboard make it frustrating.",
        entity="keyboard",
        expected_modifiers=["Serious problems with", "make it frustrating"],
        expected_polarity="negative",
        category="indirect_negative"
    ),
    
    # CATEGORY 29: Purchase justifications
    ModifierTestCase(
        name="purchase_why_bought",
        sentence="Having USB3 is why I bought this Mini.",
        entity="USB3",
        expected_modifiers=["is why I bought this Mini"],
        expected_polarity="positive",
        category="purchase_justification"
    ),
    ModifierTestCase(
        name="purchase_reason_chose",
        sentence="The warranty is the reason I chose this brand.",
        entity="warranty",
        expected_modifiers=["is the reason I chose this brand"],
        expected_polarity="positive",
        category="purchase_justification"
    ),
    
    # CATEGORY 30: Coordinated subjects (ensure both get modifiers)
    ModifierTestCase(
        name="coordinated_servers",
        sentence="The host (owner) and servers are personable and caring.",
        entity="servers",
        expected_modifiers=["are personable and caring"],
        expected_polarity="positive",
        category="coordinated_subjects"
    ),
    ModifierTestCase(
        name="coordinated_trackpad",
        sentence="The keyboard, trackpad, and speakers were all surprisingly responsive.",
        entity="trackpad",
        expected_modifiers=["was surprisingly responsive"],
        expected_polarity="positive",
        category="coordinated_subjects"
    ),
    
    # CATEGORY 31: Only/Limited quantifiers
    ModifierTestCase(
        name="quantifier_only_2_ports",
        sentence="This laptop has only 2 USB ports.",
        entity="USB ports",
        expected_modifiers=["only 2"],
        expected_polarity="negative",
        category="quantifier"
    ),
    ModifierTestCase(
        name="quantifier_only_2_usb",
        sentence="Only 2 usb ports...seems kind of...limited.",
        entity="usb ports",
        expected_modifiers=["Only 2", "seems kind of...limited"],
        expected_polarity="negative",
        category="quantifier"
    ),
    
    # CATEGORY 32: Meta-opinions (defending via criticism)
    ModifierTestCase(
        name="meta_ridiculous_screen",
        sentence="Some people might complain about low res which I think is ridiculous.",
        entity="Screen",
        expected_modifiers=["I think is ridiculous"],
        expected_polarity="positive",
        category="meta_opinion"
    ),
    
    # CATEGORY 33: Sarcasm
    ModifierTestCase(
        name="sarcasm_pretentious_lunch",
        sentence="How pretentious and inappropriate for MJ Grill to claim that it provides power lunch!",
        entity="lunch",
        expected_modifiers=["pretentious and inappropriate...to claim"],
        expected_polarity="negative",
        category="sarcasm"
    ),
    ModifierTestCase(
        name="sarcasm_service_like_this",
        sentence="How can they stay in business with service like this?",
        entity="service",
        expected_modifiers=["like this"],
        expected_polarity="negative",
        category="sarcasm"
    ),
    
    # CATEGORY 34: Not-ADJECTIVE constructions
    ModifierTestCase(
        name="negation_not_crowded",
        sentence="The place was not-crowded, perfect for a quiet dinner.",
        entity="place",
        expected_modifiers=["not-crowded", "perfect for a quiet dinner"],
        expected_polarity="positive",
        category="negation_positive"
    ),
    
    # CATEGORY 35: Relative slow/fast
    ModifierTestCase(
        name="adjective_slow_performance",
        sentence="It was also suffering from hardware issues, relatively slow performance and shortening battery lifetime.",
        entity="performance",
        expected_modifiers=["relatively slow"],
        expected_polarity="negative",
        category="basic_adjective"
    ),
    
    # CATEGORY 36: Conditional behavior (neutral cases)
    ModifierTestCase(
        name="neutral_fan_behavior",
        sentence="Fan only comes on when you are playing a game.",
        entity="Fan",
        expected_modifiers=[],
        expected_polarity="neutral",
        category="non_evaluative"
    ),
    
    # CATEGORY 37: Thick/physical descriptions
    ModifierTestCase(
        name="physical_thick_battery",
        sentence="It is really thick around the battery.",
        entity="battery",
        expected_modifiers=["is really thick around"],
        expected_polarity="negative",
        category="physical_description"
    ),
]


def test_modifier_extraction(extractor: GemmaModifierExtractor, test_case: ModifierTestCase, max_retries: int = 10, max_wait: int = 60) -> Dict[str, Any]:
    """Test a single modifier extraction case with DYNAMIC retry logic for API errors
    
    Args:
        max_retries: Maximum number of retry attempts (default 10)
        max_wait: Maximum wait time between retries in seconds (default 60s)
    """
    
    extracted_modifiers = []
    api_error = None
    total_wait = 0
    
    for attempt in range(max_retries):
        try:
            result = extractor.extract(test_case.sentence, test_case.entity)
            extracted_modifiers = result.get("modifiers", [])
            
            # Check if this was a quota error that returned empty
            justification = result.get("justification", "")
            if "429" in justification and not extracted_modifiers:
                # This is a quota error, retry with dynamic backoff
                if attempt < max_retries - 1:
                    # Exponential backoff capped at max_wait
                    wait_time = min(2 ** attempt, max_wait)
                    total_wait += wait_time
                    print(f"       API quota hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries} (total wait: {total_wait}s)...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Max retries reached
                    api_error = "API quota exceeded after retries"
                    print(f"       ⚠ API quota exceeded after {max_retries} retries, marking as partial failure")
                    break
            else:
                # Success!
                if attempt > 0:
                    print(f"       ✓ Succeeded after {attempt} retries ({total_wait}s total wait)")
                break
                
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, max_wait)
                    total_wait += wait_time
                    print(f"       API quota hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries} (total wait: {total_wait}s)...")
                    time.sleep(wait_time)
                    continue
                else:
                    api_error = f"API error: {str(e)[:50]}"
                    print(f"       ⚠ API error after {max_retries} retries: {str(e)[:50]}")
                    break
            else:
                # Non-quota error, don't retry
                api_error = f"Extraction error: {str(e)[:50]}"
                print(f"       ✗ Extraction error: {str(e)[:50]}")
                break
    
    # Check which expected modifiers were found
    found_modifiers = []
    missing_modifiers = []
    
    for expected in test_case.expected_modifiers:
        # Flexible matching: check if expected modifier appears in any extracted modifier
        expected_lower = expected.lower()
        found = any(expected_lower in mod.lower() or mod.lower() in expected_lower for mod in extracted_modifiers)
        if found:
            found_modifiers.append(expected)
        else:
            missing_modifiers.append(expected)
    
    # Calculate scores
    precision = len(found_modifiers) / len(extracted_modifiers) if extracted_modifiers else 0
    recall = len(found_modifiers) / len(test_case.expected_modifiers) if test_case.expected_modifiers else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "test_name": test_case.name,
        "category": test_case.category,
        "sentence": test_case.sentence,
        "entity": test_case.entity,
        "expected_modifiers": test_case.expected_modifiers,
        "extracted_modifiers": extracted_modifiers,
        "found_modifiers": found_modifiers,
        "missing_modifiers": missing_modifiers,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "passed": recall >= 0.8,  # Pass if we found at least 80% of expected modifiers
        "api_error": api_error  # Track if this was an API error
    }


def test_sentiment_calculation(graph: RelationGraph, test_case: ModifierTestCase, modifiers: List[str]) -> Dict[str, Any]:
    """Test sentiment calculation on entity node"""
    # Add entity node with extracted modifiers
    graph.add_entity_node(
        id=1,
        head=test_case.entity,
        modifier=modifiers,
        entity_role='associate',
        clause_layer=0
    )
    
    # Get calculated sentiment
    node_key = (1, 0)
    node_data = graph.graph.nodes.get(node_key, {})
    init_sentiment = node_data.get("init_sentiment", 0.0)
    
    # Determine polarity from sentiment score (MATCH BENCHMARK THRESHOLDS: ±0.1)
    if init_sentiment >= 0.1:
        predicted_polarity = "positive"
    elif init_sentiment <= -0.1:
        predicted_polarity = "negative"
    else:
        predicted_polarity = "neutral"
    
    polarity_correct = predicted_polarity == test_case.expected_polarity
    
    return {
        "test_name": test_case.name,
        "entity": test_case.entity,
        "modifiers": modifiers,
        "init_sentiment": init_sentiment,
        "expected_polarity": test_case.expected_polarity,
        "predicted_polarity": predicted_polarity,
        "polarity_correct": polarity_correct,
        "sentiment_score": init_sentiment
    }


def run_test_suite(rate_limit_seconds: float = 4.5):
    """Run the complete test suite with rate limiting to avoid API quota issues
    
    Args:
        rate_limit_seconds: Seconds to wait between API calls (default 4.5s = ~13 RPM, under 15 RPM limit)
    """
    print("=" * 80)
    print("MODIFIER EXTRACTION & SENTIMENT TEST SUITE")
    print("=" * 80)
    
    # Initialize extractor
    print("\nInitializing GemmaModifierExtractor...")
    extractor = GemmaModifierExtractor(cache_only=False)
    
    # Initialize sentiment analyzer
    print("Initializing SentimentAnalyzerSystem...")
    sentiment_system = MultiSentimentAnalysis(
        methods=['distilbert_logit', 'flair', 'pysentimiento', 'vader'],
        weights=[0.3, 0.1, 0.5, 0.1]  # Pysentimiento-heavy: fixes "unmatched" misclassification
    )
    
    # Run tests
    modifier_results = []
    sentiment_results = []
    api_errors = 0
    
    print(f"\nRunning {len(TEST_CASES)} test cases...")
    print(f"Rate limit: {rate_limit_seconds}s between calls (~{60/rate_limit_seconds:.0f} RPM)")
    print(f"Estimated time: {len(TEST_CASES) * rate_limit_seconds / 60:.1f} minutes\n")
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"[{i}/{len(TEST_CASES)}] Testing: {test_case.name} ({test_case.category})")
        
        # Test modifier extraction (with retry logic)
        mod_result = test_modifier_extraction(extractor, test_case)
        modifier_results.append(mod_result)
        
        # Track API errors
        if mod_result.get("api_error"):
            api_errors += 1
        
        # Test sentiment calculation
        graph = RelationGraph(test_case.sentence, [test_case.sentence], sentiment_system)
        sent_result = test_sentiment_calculation(graph, test_case, mod_result["extracted_modifiers"])
        sentiment_results.append(sent_result)
        
        # Print result
        status = "✓ PASS" if mod_result["passed"] and sent_result["polarity_correct"] else "✗ FAIL"
        error_marker = " [API ERROR]" if mod_result.get("api_error") else ""
        print(f"  {status}{error_marker} | Modifiers: {mod_result['recall']:.1%} recall | Sentiment: {sent_result['predicted_polarity']} (expected: {sent_result['expected_polarity']})")
        if mod_result["missing_modifiers"]:
            print(f"       Missing: {', '.join(mod_result['missing_modifiers'])}")
        
        # Rate limiting: wait between tests to avoid quota (except on last test)
        if i < len(TEST_CASES):
            time.sleep(rate_limit_seconds)
        
        print()
    
    # Calculate overall statistics
    print("=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)
    
    # API error statistics
    if api_errors > 0:
        print(f"\n⚠ API Errors: {api_errors}/{len(TEST_CASES)} tests hit quota limits")
        print(f"  Success rate: {((len(TEST_CASES) - api_errors) / len(TEST_CASES)):.1%}")
    
    # Modifier extraction stats (excluding API errors for accurate metrics)
    valid_results = [r for r in modifier_results if not r.get("api_error")]
    if valid_results:
        total_precision = sum(r["precision"] for r in valid_results) / len(valid_results)
        total_recall = sum(r["recall"] for r in valid_results) / len(valid_results)
        total_f1 = sum(r["f1"] for r in valid_results) / len(valid_results)
        modifier_pass_rate = sum(1 for r in valid_results if r["passed"]) / len(valid_results)
        
        print(f"\nModifier Extraction (excluding {len(modifier_results) - len(valid_results)} API errors):")
        print(f"  Precision: {total_precision:.1%}")
        print(f"  Recall:    {total_recall:.1%}")
        print(f"  F1 Score:  {total_f1:.1%}")
        print(f"  Pass Rate: {modifier_pass_rate:.1%}")
    else:
        print("\n⚠ All tests hit API errors - cannot calculate metrics")
        print("  Tip: Wait a few minutes and try again, or use smaller batches")
    
    # Sentiment calculation stats
    sentiment_accuracy = sum(1 for r in sentiment_results if r["polarity_correct"]) / len(sentiment_results)
    print(f"\nSentiment Calculation:")
    print(f"  Polarity Accuracy: {sentiment_accuracy:.1%}")
    
    # Category breakdown
    print(f"\nResults by Category:")
    categories = set(r["category"] for r in modifier_results)
    for cat in sorted(categories):
        cat_results = [r for r in modifier_results if r["category"] == cat]
        cat_valid = [r for r in cat_results if not r.get("api_error")]
        
        if cat_valid:
            cat_recall = sum(r["recall"] for r in cat_valid) / len(cat_valid)
            cat_passed = sum(1 for r in cat_valid if r["passed"]) / len(cat_valid)
            api_err_count = len(cat_results) - len(cat_valid)
            err_marker = f" ({api_err_count} API errors)" if api_err_count > 0 else ""
            print(f"  {cat:25} Recall: {cat_recall:.1%}  Pass: {cat_passed:.1%}{err_marker}")
        else:
            print(f"  {cat:25} ⚠ All {len(cat_results)} tests hit API errors")
    
    # Failed cases (excluding API errors)
    failed_cases = [r for r in modifier_results if not r["passed"] and not r.get("api_error")]
    if failed_cases:
        print(f"\nFailed Cases ({len(failed_cases)}):")
        for result in failed_cases:
            print(f"  - {result['test_name']}: missing {result['missing_modifiers']}")
    
    # API error cases
    error_cases = [r for r in modifier_results if r.get("api_error")]
    if error_cases:
        print(f"\nAPI Error Cases ({len(error_cases)}):")
        for result in error_cases[:10]:  # Limit to first 10
            print(f"  - {result['test_name']}: {result['api_error']}")
        if len(error_cases) > 10:
            print(f"  ... and {len(error_cases) - 10} more")
    
    print("\n" + "=" * 80)
    
    return {
        "modifier_precision": total_precision,
        "modifier_recall": total_recall,
        "modifier_f1": total_f1,
        "modifier_pass_rate": modifier_pass_rate,
        "sentiment_accuracy": sentiment_accuracy,
        "total_tests": len(TEST_CASES),
        "modifier_results": modifier_results,
        "sentiment_results": sentiment_results
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test modifier extraction and sentiment calculation")
    parser.add_argument(
        "--rate-limit", 
        type=float, 
        default=4.5,
        help="Seconds to wait between API calls (default: 4.5s = ~13 RPM)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run without rate limiting (may hit API quota limits)"
    )
    
    args = parser.parse_args()
    
    rate_limit = 0.5 if args.fast else args.rate_limit
    
    results = run_test_suite(rate_limit_seconds=rate_limit)
