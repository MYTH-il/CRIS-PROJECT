import os
import json
from datetime import datetime


REVIEW_PATH = os.path.join("data", "reviews", "review_queue.jsonl")


def enqueue_review(text, prediction, confidence, corrected_label=None, reviewer=None):
    os.makedirs(os.path.dirname(REVIEW_PATH), exist_ok=True)
    record = {
        "text": text,
        "prediction": prediction,
        "confidence": float(confidence) if confidence is not None else None,
        "corrected_label": corrected_label,
        "reviewer": reviewer,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(REVIEW_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return True


def load_reviews(limit=200):
    if not os.path.exists(REVIEW_PATH):
        return []
    records = []
    with open(REVIEW_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records[-limit:]

