import os
import requests
from langdetect import detect


def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return "unknown"


def translate_to_english(text):
    token = os.getenv("HF_API_TOKEN")
    model = os.getenv("HF_TRANSLATION_MODEL", "Helsinki-NLP/opus-mt-mul-en")
    if not token:
        return None, "HF_API_TOKEN not set."
    try:
        resp = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers={"Authorization": f"Bearer {token}"},
            json={"inputs": text},
            timeout=25,
        )
        if resp.status_code != 200:
            return None, f"HF API error: {resp.status_code}"
        data = resp.json()
        if isinstance(data, list) and data and "translation_text" in data[0]:
            return data[0]["translation_text"], None
        return None, "Unexpected HF response."
    except Exception as e:
        return None, f"HF request failed: {str(e)}"

