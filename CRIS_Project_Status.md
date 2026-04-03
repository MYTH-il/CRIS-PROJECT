# CRIS Project Status, Goals, and Next Steps

## Desired Goal
Build a Crime Report Intelligence System (CRIS) that turns unstructured crime reports into actionable intelligence. The target system should:
- Classify crime types accurately.
- Extract entities and evidence from narratives.
- Link similar cases and detect patterns.
- Provide temporal and spatial analytics.
- Forecast trends responsibly.
- Support investigators and policymakers with transparent, explainable outputs.

## Current State (Prototype)
The current codebase is a working Streamlit demo that:
- Loads a synthetic CSV dataset for EDA and visualization.
- Runs TF-IDF + MLP classification.
- Extracts entities with NLTK NER.
- Computes severity using VADER sentiment.
- Finds similar cases via cosine similarity.
- Predicts hotspot density using a RandomForest grid model.
- Exports a PDF dossier.

This is a functional prototype, but it is not yet research-grade or production-ready.

## What It Lacks
- Verified, reproducible evaluation (train/val/test splits with real metrics).
- A clear separation between demo behavior and research claims.
- A data schema/validation layer to enforce column types and ranges.
- A robust similarity/search layer with embeddings and vector storage.
- Multilingual handling (English + regional languages).
- Ethical and governance controls in both documentation and UI.
- A scalable backend architecture (API + DB) for real deployments.

## Improvements We Propose (and Why)
1. **Evaluation Pipeline (real metrics, fixed splits)**
   - **Why:** Research credibility depends on trustworthy metrics. The current demo inflates accuracy/confidence, which invalidates conclusions.
2. **Data Schema + Validation**
   - **Why:** Prevent silent failures and enforce consistent inputs, especially when moving to real datasets.
3. **Model Baselines + Real Comparisons**
   - **Why:** A proper baseline (LogReg/SVM) plus a stronger model (Transformer) is needed to justify improvements.
4. **Embedding-Based Similarity + Vector Store**
   - **Why:** Cosine similarity on TF-IDF is weak for semantic linkage. Embeddings enable meaningful case linking.
5. **Ethics and Governance Layer**
   - **Why:** Law enforcement applications require bias awareness, anonymization, and strict non-predictive-arrest safeguards.
6. **Backend API + Database**
   - **Why:** Streamlit is fine for prototyping, but a real system needs an API and persistent storage for scaling.

## Alternatives We Propose (and Why)
- **H3/Geohash spatial binning** instead of simple coordinate rounding.
  - **Why:** Provides consistent spatial resolution and better grid stability across latitudes.
- **Prophet/ARIMA baselines** alongside RandomForest for forecasting.
  - **Why:** Time-series baselines provide interpretability and sanity checks.
- **spaCy/Transformer NER** as the primary model, with NLTK as fallback.
  - **Why:** NLTK is lightweight but weaker. Research-grade NER needs stronger models.

## Synthetic Data: Early Need and Current Role
We began with synthetic data to enable rapid prototyping without legal or privacy risks. It allowed us to build UI flows, test pipelines, and iterate quickly.

However, synthetic data cannot support valid research claims. It may not match real-world distributions, language patterns, or reporting bias.

## Real Data for Research Basis (Next Step)
We are actively exploring public and legally permissible datasets for real evaluation. The plan is:
- Keep synthetic data for UI demos and rapid iteration.
- Add at least one real dataset for training and evaluation.
- Separate “demo” results from “research” results in reporting.

This will make the project suitable for publication, defensible metrics, and real-world relevance.

