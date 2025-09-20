from src.pipeline.modifier_e import SpacyModifierExtractor
extractor = SpacyModifierExtractor()
result = extractor.extract('The coffee is OUTSTANDING and the service was slow', 'The coffee')
print('Coffee modifiers:', result)
result2 = extractor.extract('The coffee is OUTSTANDING and the service was slow', 'the service') 
print('Service modifiers:', result2)
