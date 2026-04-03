import nltk
try:
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('maxent_ne_chunker_tab')
    nltk.download('averaged_perceptron_tagger_eng')

    sentence = "John Doe stole $500 from the Bank of America in Texas."
    tokens = nltk.word_tokenize(sentence)
    tags = nltk.pos_tag(tokens)
    tree = nltk.ne_chunk(tags)

    entities = []
    for chunk in tree:
        if hasattr(chunk, 'label'):
            entities.append((chunk.label(), ' '.join(c[0] for c in chunk)))
            
    print("entities:", entities)
except Exception as e:
    print(f"error: {repr(e)}")
