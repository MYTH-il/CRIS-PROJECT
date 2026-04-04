import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def evaluate_model(data_path, text_col="incident_description", target_col="crime_type",
                   model_type="logreg", sample_size=None, output_path="reports/evaluation.json"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    df = df.dropna(subset=[text_col, target_col])
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    X = df[text_col]
    y = df[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    if model_type == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(100, 50), activation="relu", max_iter=20, random_state=42)
    else:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    labels = list(clf.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "model_type": model_type,
        "sample_size": sample_size,
        "accuracy": accuracy,
        "classification_report": report,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload

