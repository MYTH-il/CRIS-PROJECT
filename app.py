import streamlit as st
import pandas as pd
import plotly.express as px
import os
from src.classification import train_baseline_model, predict_crime_type
from src.ner_extraction import extract_entities
from src.analytics import generate_hotspot_map, generate_temporal_forecast
from src.advanced_intel import AdvancedIntelligenceEngine

st.set_page_config(page_title="CRIS Dashboard", page_icon="🕵️‍♂️", layout="wide")

st.title("CRIS: Crime Report Intelligence System")
st.markdown("### Exploratory Data Analysis & System Prototyping")

# Constants
DATA_PATH = os.path.join("data", "crime_reports_synthetic.csv")

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        return df
    else:
        st.error(f"Dataset not found at {DATA_PATH}. Please ensure the file is in the data/ directory.")
        return None

with st.spinner("Loading 60k Synthetic Dataset..."):
    df = load_data()

if df is not None:
    st.success(f"Successfully loaded dataset with {df.shape[0]:,} rows and {df.shape[1]} columns!")
    
    # Overview Tab
    tab1, tab2, tab3, tab4 = st.tabs(["Data Viewer", "Missing Values", "Advanced Analytics & Hotspots", "Intelligence Demo"])
    
    with tab1:
        st.subheader("Raw Data Sample")
        view_df = df.head(100)
        # Remove the 'data source' column from the viewer if it exists
        cols_to_drop = [col for col in view_df.columns if 'data_source' in col.lower() or 'data source' in col.lower()]
        if cols_to_drop:
            view_df = view_df.drop(columns=cols_to_drop)
        st.dataframe(view_df, use_container_width=True)
        
    with tab2:
        st.subheader("Missing Values Count")
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ["Column", "Missing Count"]
        missing_df = missing_df[missing_df["Missing Count"] > 0]
        
        if len(missing_df) > 0:
            fig = px.bar(missing_df, x="Column", y="Missing Count", title="Missing Values per Column")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No missing values found in the dataset!")
            
    with tab3:
        st.subheader("Advanced Analytics & Regional Forecasting")
        
        # UI Toggles
        if 'crime_type' in df.columns:
            crime_cats = ["All Crimes"] + list(df['crime_type'].dropna().unique())
            selected_crime = st.selectbox("🎯 Filter Analytics By Crime Type:", crime_cats)
        else:
            selected_crime = None
            st.warning("No 'crime_type' column found. Displaying all crimes.")
            
        st.markdown("---")
        
        # Dual Column Layout for Maps and Charts
        chart_col, map_col = st.columns([1, 1])
        
        with chart_col:
            st.markdown("#### 📈 Temporal Trend Analysis")
            with st.spinner("Calculating moving averages..."):
                forecast_fig = generate_temporal_forecast(df, selected_crime)
                if forecast_fig:
                    st.plotly_chart(forecast_fig, use_container_width=True)
                else:
                    st.error("Missing valid 'report_date' column. Cannot generate temporal trends.")
                    
        with map_col:
            st.markdown("#### 🗺️ Geospatial Hotspots")
            with st.spinner("Mapping incident coordinates..."):
                map_fig = generate_hotspot_map(df, selected_crime)
                if map_fig:
                    st.plotly_chart(map_fig, use_container_width=True)
                else:
                    st.error("Missing 'latitude' / 'longitude' columns. Cannot render map.")

    with tab4:
        st.subheader("Intelligence Engine (Baseline)")
        
        # Training Expander
        with st.expander("⚙️ Train Baseline Model (Run Once)"):
            st.markdown("This will train a `TF-IDF + Logistic Regression` baseline on your 60k dataset.")
            if st.button("Start Training"):
                with st.spinner("Training baseline classifier... This may take a minute."):
                    try:
                        accuracy, report = train_baseline_model(DATA_PATH)
                        st.success(f"Model trained successfully! Accuracy: {accuracy*100:.2f}%")
                        st.json(report)
                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")

        st.markdown("---")
        st.markdown("### Paste an Incident Report")
        
        sample_text = st.text_area("Incident Narrative", height=200, 
                                   placeholder="e.g. On Tuesday evening, John Doe broke into the warehouse on 5th Ave using a crowbar...")
                                   
        if st.button("Analyze with CRIS"):
            if not sample_text.strip():
                st.warning("Please enter some text to analyze.")
            else:
                # Initialize Advanced Engine
                VECTORIZER_PATH = os.path.join("models", "baseline_vectorizer.pkl")
                intel_engine = AdvancedIntelligenceEngine(DATA_PATH, VECTORIZER_PATH)
                
                colA, colB, colC = st.columns(3)
                
                with colA:
                    st.markdown("#### 🔍 Expected Crime")
                    with st.spinner("Classifying..."):
                        prediction, prob = predict_crime_type(sample_text)
                    if prediction:
                        st.success(f"**{prediction}**")
                        st.metric("Confidence", f"{prob:.1f}%")
                    else:
                        st.error(prob)
                        
                with colB:
                    st.markdown("#### ⚠️ Severity Triage")
                    with st.spinner("Analyzing distress..."):
                        severity = intel_engine.calculate_severity_score(sample_text)
                        
                    if severity >= 8:
                        st.error(f"CODE RED: {severity}/10")
                        st.markdown("🚨 *High distress or weapons detected!*")
                    elif severity >= 5:
                        st.warning(f"ELEVATED: {severity}/10")
                    else:
                        st.info(f"STANDARD: {severity}/10")
                        
                with colC:
                    st.markdown("#### 🏷️ Entity Network")
                    with st.spinner("Extracting..."):
                        entities = extract_entities(sample_text)
                    
                    if entities:
                        for label, items in entities.items():
                            st.markdown(f"**{label}**: {', '.join(items)}")
                    else:
                        st.info("No notable entities found.")
                        
                st.markdown("---")
                
                # Bottom Row: Palantir-style advanced features
                bot_col1, bot_col2 = st.columns([1.5, 1])
                
                with bot_col1:
                    st.markdown("#### 🕸️ Syndicate Knowledge Graph")
                    if entities:
                        graph_fig = intel_engine.build_syndicate_graph(entities)
                        st.plotly_chart(graph_fig, use_container_width=True)
                    else:
                        st.info("Not enough entities to build a network.")
                        
                with bot_col2:
                    st.markdown("#### 📂 Modus Operandi Matches")
                    with st.spinner("Scanning 60k database for similarity..."):
                        matches = intel_engine.find_mo_similarity(sample_text)
                    
                    if matches:
                        for m in matches:
                            st.info(f"**{m['similarity']:.1f}% Match** | {m['crime']}\n\n*\"{m['snippet']}\"*")
                    else:
                        st.warning("No similar historical cases found.")
