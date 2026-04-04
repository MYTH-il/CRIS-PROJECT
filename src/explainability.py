import os
import joblib
import numpy as np

try:
    from lime.lime_text import LimeTextExplainer
    _LIME_AVAILABLE = True
except Exception:
    _LIME_AVAILABLE = False


def explain_prediction(text, model_type="mlp", top_k=8):
    """
    Returns a list of (feature, weight) for model explanation.
    Uses LIME if available; falls back to linear coefficients for LogReg.
    """
    model_path = os.path.join("models", f"{model_type}_classifier.pkl")
    vectorizer_path = os.path.join("models", "baseline_vectorizer.pkl")
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return []

    clf = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    if model_type == "logreg":
        try:
            vec = vectorizer.transform([text])
            coef = clf.coef_
            if coef.shape[0] > 1:
                # Multi-class: pick predicted class coefficients
                pred = clf.predict(vec)[0]
                class_idx = list(clf.classes_).index(pred)
                weights = coef[class_idx]
            else:
                weights = coef[0]
            top_idx = np.argsort(weights)[-top_k:][::-1]
            features = np.array(vectorizer.get_feature_names_out())[top_idx]
            return [(f, float(weights[i])) for f, i in zip(features, top_idx)]
        except Exception:
            return []

    if _LIME_AVAILABLE:
        try:
            explainer = LimeTextExplainer(class_names=list(getattr(clf, "classes_", [])))
            exp = explainer.explain_instance(
                text, lambda texts: clf.predict_proba(vectorizer.transform(texts)), num_features=top_k
            )
            return exp.as_list()
        except Exception:
            return []

    return []

