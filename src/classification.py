import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import json
from datetime import datetime
import requests

MODEL_DIR = "models"
MODEL_PATHS = {
    "mlp": os.path.join(MODEL_DIR, "mlp_classifier.pkl"),
    "logreg": os.path.join(MODEL_DIR, "logreg_classifier.pkl"),
}
VECTORIZER_PATH = os.path.join(MODEL_DIR, "baseline_vectorizer.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")

def _split_data(X, y, seed=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_baseline_model(data_path, text_col='incident_description', target_col='crime_type', model_type="mlp", sample_size=1500):
    """
    Trains a Scikit-Learn model on the dataset.
    model_type: "mlp" or "logreg"
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # 1. Load Data
    df = pd.read_csv(data_path)
    
    # Drop rows where text or target is missing
    df = df.dropna(subset=[text_col, target_col])
    
    # DOWNSAMPLE for snappy demo performance. Set sample_size=None to use full data.
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
        
    X = df[text_col]
    y = df[target_col]

    # 2. Split Data
    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(X, y)

    # 3. Text Vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 4. Train Model
    if model_type == "logreg":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    else:
        clf = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', max_iter=20, random_state=42)
    clf.fit(X_train_vec, y_train)

    # 5. Evaluate Baseline
    y_pred = clf.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    report = classification_report(y_test, y_pred, output_dict=True)

    # 7. Save Model & Vectorizer
    model_path = MODEL_PATHS.get(model_type, MODEL_PATHS["mlp"])
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"model_type": model_type, "trained_at": datetime.utcnow().isoformat() + "Z"}, f)
    
    return accuracy, report

def predict_crime_type(text, model_type=None):
    """
    Predicts the crime type for a given incident description.
    Loads the model gracefully if available.
    """
    if model_type is None and os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                model_type = json.load(f).get("model_type")
        except Exception:
            model_type = None

    model_path = MODEL_PATHS.get(model_type or "mlp", MODEL_PATHS["mlp"])

    if not os.path.exists(model_path) or not os.path.exists(VECTORIZER_PATH):
        return None, "Model not trained yet."
        
    clf = joblib.load(model_path)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    # Vectorize the raw text
    text_vec = vectorizer.transform([text])
    
    # Predict
    prediction = clf.predict(text_vec)[0]
    
    # Get probability from Deep Learning Softmax layer
    probabilities = clf.predict_proba(text_vec)[0]
    max_prob = max(probabilities) * 100
    
    return prediction, max_prob


def predict_zero_shot(text, labels):
    """
    Zero-shot classification via Hugging Face Inference API.
    Requires HF_API_TOKEN in environment.
    """
    token = os.getenv("HF_API_TOKEN")
    model = os.getenv("HF_ZERO_SHOT_MODEL", "joeddav/xlm-roberta-large-xnli")
    if not token:
        return None, "HF_API_TOKEN not set."
    if not labels:
        return None, "No labels provided."
    try:
        resp = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers={"Authorization": f"Bearer {token}"},
            json={"inputs": text, "parameters": {"candidate_labels": labels}},
            timeout=25,
        )
        if resp.status_code != 200:
            return None, f"HF API error: {resp.status_code}"
        data = resp.json()
        if isinstance(data, dict) and "labels" in data and "scores" in data:
            top_label = data["labels"][0]
            top_score = data["scores"][0] * 100
            return top_label, top_score
        return None, "Unexpected HF response."
    except Exception as e:
        return None, f"HF request failed: {str(e)}"
