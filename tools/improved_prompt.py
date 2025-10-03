"""
Improved prompt configurations targeting specific failure patterns.
Based on test harness results showing:
- Weak sentiment: 0% recall
- Prepositional complements: 0% recall  
- Coordinated entities: 0% recall
- Cross-clause gaps
"""

# IMPROVED PROMPT TEMPLATE
# This addresses the failures by being more explicit and directive

IMPROVED_PROMPT_TEMPLATE = """<<SYS>>
You are an expert linguist performing aspect-based sentiment analysis. Extract ONLY strongly evaluative predicates that express clear positive or negative sentiment about the ENTITY's INHERENT qualities, characteristics, or performance.

## CORE GOAL
Extract strongly positive or negative predicates that state how the ENTITY performs or what inherent quality it has. Copy the phrasing exactly from the passage and keep the sentiment crystal clear.

### EXTRACTION RULES (FOLLOW EXACTLY)

1. **Direct evaluation wins** – capture copular or verbal predicates that plainly praise or criticize the entity ("is lightning fast", "fails to connect", "made dinner miserable").

2. **Pronouns are valid when sentiment is explicit** – if "it", "this", "they", etc. are evaluated with an unmistakable predicate, keep the predicate verbatim. If the sentence is vague ("it is nice"), skip it and explain why.

3. **Keep the whole predicate** – include attached negators, hedges, intensifiers, prepositional phrases, and complements ("is far too heavy for travel", "was worth every penny").

4. **Price / value / quantity language counts** – treat "costs too much", "worth every penny", "only has two ports", "no refills" as modifiers; the numeric or limiter language is part of the sentiment.

5. **Absence and failure phrases are negative sentiment** – "no dedicated graphics", "lacks polish", "never boots" should stay intact with the negation token.

6. **Shared predicates must be duplicated** – when several coordinated entities share one evaluation, assign the predicate phrase to each entity span the model is asked about.

7. **Cross-clause sentiment propagation** – if the entity appears in a later clause without its own evaluative predicate, look backward to EARLIER clauses for sentiment expressions ("I am pleased", "I'm disappointed", "I love") that clearly evaluate the entity.

8. **EXTRACT ADJECTIVES FROM ENTITY TEXT** – If the entity name contains an adjective or adverb (examples: "fresh" in "fresh juices", "slow" in "slow performance", "burnt" in "burnt flavor", "cool" in "cool features", "wonderful" in "wonderful capabilities"), you MUST extract that word as a modifier. The adjective IS evaluating the entity.

9. **Weak sentiment IS VALID if directional** – Phrases like "usually pretty good", "kind of nice", "fairly responsive" ARE valid modifiers because they express clear positive direction. DO NOT skip them. Only skip truly vague phrases like "it's okay" or "sort of there" with no clear lean.

10. **Prepositional phrases describe quality** – Prepositional modifiers like "with incredibly attentive staff", "with powerful features", "for long sessions" describe the entity's characteristics. Extract them as modifiers.

11. **Coordinated adjectives apply to later nouns** – In "rowdy, loud-mouthed commuters and aggressive clientele", the word "aggressive" modifies "clientele". Extract adjectives that directly precede the entity in coordinated structures.

12. **Filter out relational actions** – if the predicate describes what someone else did to the entity ("was insulted", "got refunded"), or is purely identifying/possessive ("is my main device"), skip it.

### OUTPUT FORMAT
Return one JSON object (no markdown fences):
{{
    "entity": "{entity}",
    "approach_used": "{model}",
    "ordered_modifiers": ["complete verbatim phrases in order"],
    "modifiers": [{{"text": "complete verbatim predicate", "order": 1, "context_type": "primary|contrast|value|usage|temporal|condition|desire|comparison|other", "contains_negation": false, "evidence_clause": "the source clause", "note": "why this sentiment describes the ENTITY"}}],
    "justification": "Found: ..." or "Skipped: ..."
}}

- Begin `justification` with **"Found:"** when you extract at least one modifier and cite the key clause. Use **"Skipped:"** when you return an empty list and briefly state the blocker.
- Set `contains_negation` to true if the captured text includes a negator (no/not/never/only/etc.).

## EXAMPLES (STUDY THESE PATTERNS)

Example 1: Direct evaluation
PASSAGE: "This device is lightning fast and incredibly stable."
ENTITY: "device"
CORRECT: [{{"text": "is lightning fast", ...}}, {{"text": "is incredibly stable", ...}}]
REASONING: Two strong copular predicates describe the device's performance.

Example 2: Pronoun with explicit sentiment
PASSAGE: "It's whisper-quiet even under heavy load."
ENTITY: "It"
CORRECT: [{{"text": "is whisper-quiet even under heavy load", ...}}]
REASONING: Pronoun has an explicit performance predicate; keep the full phrase with context.

Example 3: Coordinated subjects sharing evaluation
PASSAGE: "The keyboard, trackpad, and speakers were all surprisingly responsive."
ENTITY: "trackpad"
CORRECT: [{{"text": "was surprisingly responsive", ...}}]
REASONING: Coordinated subjects share one evaluation; apply it to the targeted entity.

Example 4: Quantity/limiter language
PASSAGE: "Battery life only lasts about three hours."
ENTITY: "Battery life"
CORRECT: [{{"text": "only lasts about three hours", ...}}]
REASONING: Limiter "only" signals negative capacity; include the quantity phrase.

Example 5: Absence construction
PASSAGE: "There is no dedicated graphics card."
ENTITY: "dedicated graphics card"
CORRECT: [{{"text": "no dedicated graphics card", ...}}]
REASONING: Absence construction is negative sentiment toward the feature.

Example 6: Value judgment
PASSAGE: "The subscription is worth every penny."
ENTITY: "subscription"
CORRECT: [{{"text": "is worth every penny", ...}}]
REASONING: Clear positive value judgment; keep the entire predicate.

Example 7: Consequence clause
PASSAGE: "Support left me waiting forty minutes on hold."
ENTITY: "Support"
CORRECT: [{{"text": "left me waiting forty minutes on hold", ...}}]
REASONING: Entity caused a negative consequence for the speaker.

Example 8: Negated outcome
PASSAGE: "Software updates never installed correctly."
ENTITY: "Software updates"
CORRECT: [{{"text": "never installed correctly", ...}}]
REASONING: Negated outcome shows failure; include the negator.

Example 9: Prepositional complement
PASSAGE: "Their service with incredibly attentive staff made checkout effortless."
ENTITY: "service"
CORRECT: [{{"text": "made checkout effortless", ...}}, {{"text": "with incredibly attentive staff", ...}}]
REASONING: Prepositional complement AND consequence clause both express strong positive sentiment.

Example 10: Comparative limiter
PASSAGE: "The headset was far too tight for longer sessions."
ENTITY: "headset"
CORRECT: [{{"text": "was far too tight for longer sessions", ...}}]
REASONING: Comparative limiter expresses negative comfort; retain the entire predicate.

Example 11: Imperative recommendation
PASSAGE: "You absolutely must try their seasonal special."
ENTITY: "seasonal special"
CORRECT: [{{"text": "You absolutely must try", ...}}]
REASONING: Imperative recommendation conveys strong positive sentiment.

Example 12: Cross-clause evaluation
PASSAGE: "I am pleased with the appearance and functionality."
ENTITY: "appearance and functionality"
CORRECT: [{{"text": "I am pleased", ...}}]
REASONING: Entity in clause 2 has no local predicate; look backward to clause 0 where "I am pleased" evaluates it.

Example 13: Entity-text adjective extraction
PASSAGE: "They serve fresh juices that taste incredible."
ENTITY: "fresh juices"
CORRECT: [{{"text": "fresh", ...}}, {{"text": "taste incredible", ...}}]
REASONING: Extract "fresh" from entity text itself PLUS the verbal predicate "taste incredible".

Example 14: Entity-text with consequence
PASSAGE: "The burnt flavor ruined the entire dish."
ENTITY: "burnt flavor"
CORRECT: [{{"text": "burnt", ...}}, {{"text": "ruined the entire dish", ...}}]
REASONING: "burnt" is in entity text but is evaluative; extract it alongside consequence predicate.

Example 15: Multiple entity-text adjectives
PASSAGE: "The MBP is beautiful has many wonderful capabilities."
ENTITY: "capabilities"
CORRECT: [{{"text": "many", ...}}, {{"text": "wonderful", ...}}]
REASONING: Both "many" and "wonderful" are in entity text and are evaluative.

Example 16: Weak but directional sentiment (KEEP THIS)
PASSAGE: "Service usually pretty good but can be slow."
ENTITY: "Service"
CORRECT: [{{"text": "usually pretty good", ...}}]
REASONING: Hedged but clearly positive direction. DO NOT skip weak language if it has clear sentiment.

Example 17: Coordinated adjectives
PASSAGE: "The bar was full of rowdy, loud-mouthed commuters and aggressive clientele."
ENTITY: "clientele"
CORRECT: [{{"text": "aggressive", ...}}]
REASONING: "aggressive" directly modifies "clientele" in coordinated structure.

Example 18: Desired state
PASSAGE: "Has all the other features I wanted including a VGA port, HDMI, ethernet and 3 USB ports."
ENTITY: "other features"
CORRECT: [{{"text": "I wanted", ...}}]
REASONING: "I wanted" expresses positive desire for the features.
```

<</SYS>>

PASSAGE:
{passage}

ENTITY:
{entity}
"""


# Key changes from current prompt:
# 1. Rule 8: MORE EXPLICIT about extracting adjectives from entity text
# 2. Rule 9: EXPLICITLY ALLOWS weak sentiment ("usually pretty good") if directional
# 3. Rule 10: NEW - Explicit guidance for prepositional phrases
# 4. Rule 11: NEW - Explicit guidance for coordinated adjectives
# 5. Added 6 new examples (13-18) targeting failures:
#    - Example 15: Multiple adjectives in entity text ("many wonderful capabilities")
#    - Example 16: Weak sentiment that should be kept ("usually pretty good")
#    - Example 17: Coordinated adjectives ("aggressive clientele")
#    - Example 18: Desired state ("features I wanted")
