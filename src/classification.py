import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "baseline_classifier.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "baseline_vectorizer.pkl")

def train_baseline_model(data_path, text_col='incident_description', target_col='crime_type'):
    """
    Trains a baseline TF-IDF + Logistic Regression model on the dataset.
    This serves as the scientific baseline for our research paper before we use Transformers.
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # 1. Load Data
    df = pd.read_csv(data_path)
    
    # Drop rows where text or target is missing
    df = df.dropna(subset=[text_col, target_col])
    
    X = df[text_col]
    y = df[target_col]

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Text Vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 4. Train Model
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train_vec, y_train)

    # 5. Evaluate Baseline
    y_pred = clf.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # 6. Save Model & Vectorizer
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    return accuracy, report

def predict_crime_type(text):
    """
    Predicts the crime type for a given incident description.
    Loads the model gracefully if available.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, "Model not trained yet."
        
    clf = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    # Vectorize the raw text
    text_vec = vectorizer.transform([text])
    
    # Predict
    prediction = clf.predict(text_vec)[0]
    
    # Get probability
    probabilities = clf.predict_proba(text_vec)[0]
    max_prob = max(probabilities) * 100
    
    return prediction, max_prob
