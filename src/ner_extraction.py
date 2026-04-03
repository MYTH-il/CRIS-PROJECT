import nltk
from collections import defaultdict

_nltk_loaded = False

def load_nlp():
    global _nltk_loaded
    if not _nltk_loaded:
        try:
            nltk.data.find('chunkers/maxent_ne_chunker_tab')
        except LookupError:
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('maxent_ne_chunker_tab', quiet=True)
            nltk.download('words', quiet=True)
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            nltk.download('punkt', quiet=True)
        _nltk_loaded = True
    return True

def extract_entities(text):
    """
    Extracts named entities from the incident text using pure-Python NLTK Statistical NER.
    This bypasses Windows Defender Application Control DLL blocking.
    Returns a dictionary grouped by entity type.
    """
    if not isinstance(text, str) or not text.strip():
        return {}
        
    load_nlp()
    
    # Tokenize, tag, and chunk
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    tree = nltk.ne_chunk(tags)
    
    entities = defaultdict(list)
    
    for chunk in tree:
        if hasattr(chunk, 'label'):
            label = chunk.label()
            entity_text = ' '.join(c[0] for c in chunk)
            
            if label in ['PERSON', 'ORGANIZATION', 'GPE', 'LOCATION', 'FACILITY']:
                # Normalize tags to match expectations
                if label == 'GPE': label = 'LOCATION'
                if label == 'ORGANIZATION': label = 'ORG'
                
                # Avoid duplicates
                if entity_text not in entities[label]:
                    entities[label].append(entity_text)
            
    return dict(entities)
