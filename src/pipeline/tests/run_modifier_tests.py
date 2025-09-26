from src.pipeline.modifier_e import SpacyModifierExtractor, GemmaModifierExtractor

cases = [
    ("Other than not being a fan of click pads (industry standard these days) and the lousy internal speakers, it's hard for me to find things about this notebook I don't like, especially considering the $350 price tag.", "notebook"),
    ("The nicest part is the low heat output and ultra quiet operation.", "operation"),
    ("...I still think there's nothing like a Mac.", "Mac"),
    ("The lack of an external power supply justifies the small increase in size.", "external power supply"),
]

spacy_ex = SpacyModifierExtractor()

for text, ent in cases:
    out = spacy_ex.extract(text, ent)
    print("TEXT:\n", text)
    print("ENTITY:", ent)
    print("SPACY MODIFIERS:", out)
    print('-'*80)

# Note: GemmaModifierExtractor requires API key; we only print the generated prompt for inspection if available
try:
    gm = GemmaModifierExtractor(api_key=None)
    for text, ent in cases:
        print('GEMMA PROMPT for:', ent)
        print(gm._prompt(text, ent)[:500])
        print('-'*80)
except Exception as e:
    print('GemmaModifierExtractor unavailable:', e)
