import os
import re
import requests
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

def _regex_entities(text):
    entities = defaultdict(list)

    ipc_matches = re.findall(r"\b(?:IPC\s*Sections?\s*)?(\d{3}[A-Z]?)\b", text, flags=re.IGNORECASE)
    for m in ipc_matches:
        if m not in entities["IPC_SECTION"]:
            entities["IPC_SECTION"].append(m)

    fir_matches = re.findall(r"\bFIR/\d{4}/[A-Z]{2,4}/\d{4,6}\b", text, flags=re.IGNORECASE)
    for m in fir_matches:
        if m not in entities["FIR_NUMBER"]:
            entities["FIR_NUMBER"].append(m)

    phone_matches = re.findall(r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b", text)
    for m in phone_matches:
        if m not in entities["PHONE_NUMBER"]:
            entities["PHONE_NUMBER"].append(m)

    vehicle_matches = re.findall(r"\b[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,2}\s?\d{4}\b", text)
    for m in vehicle_matches:
        if m not in entities["VEHICLE_NUMBER"]:
            entities["VEHICLE_NUMBER"].append(m)

    weapon_keywords = ["knife", "gun", "pistol", "rifle", "rod", "bat", "firearm", "explosive", "bomb"]
    for w in weapon_keywords:
        if re.search(rf"\b{re.escape(w)}\b", text, flags=re.IGNORECASE):
            entities["WEAPON"].append(w.title())

    station_matches = re.findall(r"\bPS\s?\d{3,4}\b", text, flags=re.IGNORECASE)
    for m in station_matches:
        if m not in entities["POLICE_STATION"]:
            entities["POLICE_STATION"].append(m.upper().replace(" ", ""))

    return dict(entities)


def _hf_ner(text):
    token = os.getenv("HF_API_TOKEN")
    model = os.getenv("HF_NER_MODEL", "dslim/bert-base-NER")
    if not token:
        return {}
    try:
        resp = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers={"Authorization": f"Bearer {token}"},
            json={"inputs": text},
            timeout=20,
        )
        if resp.status_code != 200:
            return {}
        data = resp.json()
    except Exception:
        return {}

    entities = defaultdict(list)
    if isinstance(data, list):
        for item in data:
            label = item.get("entity_group") or item.get("entity")
            word = item.get("word")
            if not label or not word:
                continue
            label = label.replace("I-", "").replace("B-", "")
            if word not in entities[label]:
                entities[label].append(word)
    return dict(entities)


def extract_entities(text, use_hf=False):
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

    # Merge regex entities
    regex_entities = _regex_entities(text)
    for label, items in regex_entities.items():
        for item in items:
            if item not in entities[label]:
                entities[label].append(item)

    # Optionally merge HF NER (WDAC-safe via API)
    if use_hf:
        hf_entities = _hf_ner(text)
        for label, items in hf_entities.items():
            for item in items:
                if item not in entities[label]:
                    entities[label].append(item)

    return dict(entities)
