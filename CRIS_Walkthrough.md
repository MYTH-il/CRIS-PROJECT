# CRIS Development Walkthrough

## Phase 1: Project Setup & EDA Dashboard (Completed: 2026-04-03)

We have successfully set up the development environment, project structure, and the core Streamlit application.

### Changes Made
- **Folder Structure**: Created `data/`, `notebooks/`, and `src/` directories.
- **Dataset Handled**: Moved your `crime_reports_synthetic.csv` safely into the `data/` folder.
- **Dependencies**: Appended the necessary tools for text analysis (`streamlit`, `scikit-learn`, `transformers`) to your `requirements.txt`.
- **Application**: Built `app.py`, which is an interactive Streamlit dashboard set up with tabs to explore the raw synthetic data, view missing values, and see basic categorical distributions using Plotly.
- **Data Pipeline**: Initialized `src/preprocess.py` to begin organizing our future NLP cleanup tasks.

### Validation / Testing

To see your new dashboard in action:

1. Open your terminal in the project directory.
2. Activate your environment:
   ```cmd
   .\.venv\Scripts\activate
   ```
3. Install the updated dependencies using uv (or standard pip):
   ```cmd
   uv pip install -r requirements.txt
   ```
4. Run the newly created application:
   ```cmd
   streamlit run app.py
   ```

A browser window will open automatically displaying your 60k rows being dynamically loaded into an interactive dataframe, alongside your missing value stats and pie charts for categorical variables.

> [!TIP]
> The app caches your massive dataset (`@st.cache_data`) when it first runs, so switching tabs and reloading components will be practically instant!

## Phase 2: NLP Baseline Pipeline (Completed: 2026-04-03)

We have officially built the scientific baseline intelligence engine for the project, directly integrated into the Streamlit dashboard.

### Changes Made
- **Classification Engine**: Created `src/classification.py`. It uses a `TfidfVectorizer` paired with a `LogisticRegression` classifier to map your `incident_description` to your `crime_type` mathematically. It automatically balances classes and caches the resulting ML models to the `models/` directory for instant inference later.
- **Entity Extractor (NER)**: Created `src/ner_extraction.py`. Due to strict Windows Defender Application Control (WDAC) policies blocking C-compiled libraries, we opted to use a pure-Python NLTK Statistical NER chunker (`maxent_ne_chunker`). This instantly dissects crime narratives and categorizes any Persons, Locations, Organizations, and Dates/Times it discovers without tripping security locks!
- **Streamlit Integration**: We modified `app.py` to add a brand new **Intelligence Demo** tab. This acts as a unified hub where you can 1) Train your baseline model, and 2) Paste fake or real incident narratives to see the NLP pipeline extract entities and classify crimes in real-time.

### Validation / Testing
1. In your local browser window that is running Streamlit, refresh the page.
2. Navigate to the new 4th tab: **Intelligence Demo**.
3. Open the `Train Baseline Model` expander and click **Start Training**. Watch as your 60k rows are converted into a mathematical matrix.
4. Once training completes, paste any fake incident report into the text box and hit **Analyze with CRIS** to see your ML model spit out predictions and colored entities!
