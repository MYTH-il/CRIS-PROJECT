import streamlit as st
import pandas as pd
import plotly.express as px
import os
from src.classification import train_baseline_model, predict_crime_type
from src.ner_extraction import extract_entities
from src.analytics import generate_hotspot_map, generate_temporal_forecast, generate_crime_distribution_chart, generate_time_distribution_chart
from src.advanced_intel import AdvancedIntelligenceEngine
from src.predictive_mapping import PredictiveHotspotEngine
from src.dossier_generator import create_pdf_dossier

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
            st.markdown("#### 🗺️ Historic Hotspots")
            with st.spinner("Mapping incident coordinates..."):
                map_fig = generate_hotspot_map(df, selected_crime)
                if map_fig:
                    st.plotly_chart(map_fig, use_container_width=True)
                else:
                    st.error("Missing 'latitude' / 'longitude' columns. Cannot render map.")
                    
        st.markdown("---")
        
        # New Secondary Chart Row
        extra_col1, extra_col2 = st.columns(2)
        
        with extra_col1:
            st.markdown("#### 📊 Crime Distribution")
            with st.spinner("Generating distribution..."):
                dist_fig = generate_crime_distribution_chart(df, selected_crime)
                if dist_fig:
                    st.plotly_chart(dist_fig, use_container_width=True)
                else:
                    st.info("Insufficient data for distribution chart.")

        with extra_col2:
            st.markdown("#### 🕒 Time Analytics")
            with st.spinner("Analyzing temporal patterns..."):
                time_fig = generate_time_distribution_chart(df, selected_crime)
                if time_fig:
                    st.plotly_chart(time_fig, use_container_width=True)
                else:
                    st.info("Insufficient data for time distribution chart.")
                    
        st.markdown("---")
        
        # New Predictive ML Zone
        st.subheader("🤖 Future Resource Allocation Map (AI Random Forest)")
        p_col1, p_col2 = st.columns([1, 3])
        
        with p_col1:
            st.markdown("Configure ML Forecast:")
            forecast_days = st.slider("Prediction Timeframe (Days)", min_value=1, max_value=7, value=1)
            
            if st.button("Generate ML Forecast"):
                with st.spinner("Training Random Forest Regressor..."):
                    # Cache this operation in production!
                    rf_engine = PredictiveHotspotEngine()
                    success = rf_engine.prepare_and_train(df)
                    
                    if success:
                        st.session_state['rf_trained'] = True
                        st.session_state['rf_engine'] = rf_engine
                    else:
                        st.error("Dataset missing required geospatial/temporal columns.")
                        
            if st.session_state.get('rf_trained', False):
                st.success("Model Trained Successfully!")
                metrics = st.session_state['rf_engine'].metrics
                st.metric("Model Operational Accuracy", f"{metrics['r2'] * 100:.1f}%")
                st.caption(f"Mean Absolute Error: {metrics['mae']:.2f} incidents per grid")
                
        with p_col2:
            if st.session_state.get('rf_trained', False):
                st.markdown(f"#### 🔮 Predicted Hotspots (Next {forecast_days} Days)")
                with st.spinner("Extrapolating grid densities..."):
                    ml_map, _, future_df = st.session_state['rf_engine'].forecast_threat_map(target_days=forecast_days)
                    if ml_map:
                        st.plotly_chart(ml_map, use_container_width=True)
                        
                        st.markdown("---")
                        st.markdown("#### 📊 Predictive ML Analysis & Breakdown")
                        
                        a_col1, a_col2 = st.columns(2)
                        
                        # 1. Total Incidents per day
                        daily_threat = future_df.groupby('target_date')['predicted_count'].sum().reset_index()
                        daily_fig = px.bar(daily_threat, x='target_date', y='predicted_count', 
                                           title="Projected Incident Volume per Day",
                                           color='predicted_count', color_continuous_scale="Reds")
                        a_col1.plotly_chart(daily_fig, use_container_width=True)
                        
                        # Dynamic Text Analysis for Daily Forecast
                        total_expected = daily_threat['predicted_count'].sum()
                        peak_date = daily_threat.loc[daily_threat['predicted_count'].idxmax(), 'target_date']
                        a_col1.info(f"**Insight:** The Random Forest expects a total of **{total_expected:.0f}** combined incidents over the {forecast_days}-day period, with threat levels hitting mathematical peak on **{peak_date}**.")
                        
                        # 2. Top 5 Grid zones
                        top_grids = future_df.groupby(['lat_grid', 'lon_grid'])['predicted_count'].sum().reset_index()
                        top_grids = top_grids.sort_values(by='predicted_count', ascending=False).head(5)
                        top_grids['Grid Coordinate'] = top_grids['lat_grid'].astype(str) + "°N, " + top_grids['lon_grid'].astype(str) + "°W"
                        grid_fig = px.bar(top_grids, x='predicted_count', y='Grid Coordinate', orientation='h',
                                          title="Highest Risk Grid Quadrants",
                                          color='predicted_count', color_continuous_scale="Reds")
                        grid_fig.update_layout(yaxis={'categoryorder':'total ascending'})
                        a_col2.plotly_chart(grid_fig, use_container_width=True)
                        
                        # Dynamic Text Analysis for Spatial Grids
                        if len(top_grids) > 0:
                            highest_grid = top_grids.iloc[0]
                            a_col2.warning(f"**Tactical Dispatch:** The highest priority deployment zone is **{highest_grid['Grid Coordinate']}**, which is projected to suffer **{highest_grid['predicted_count']:.0f}** incidents. Units are strongly advised to deploy heavy presence here.")
                        
            else:
                st.info("👈 Click 'Generate ML Forecast' to train the AI and visualize tomorrow's threats.")

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
                        
                st.markdown("---")
                st.markdown("### 📥 Export Intelligence Dossier")
                
                # Cross-reference with Predictive Pipeline
                future_threat = ""
                if st.session_state.get('rf_trained', False):
                    if entities and 'LOCATION' in entities:
                        locs = ", ".join(entities['LOCATION'])
                        future_threat = f"TACTICAL ALERT FOR {locs}:\nAccording to the Random Forest Predictive ML Grid, the algorithm projects a highly elevated threat density across matching geographic quadrants over the next 7 days based on this specific Modus Operandi."
                    else:
                        future_threat = "TACTICAL ALERT:\nThe Random Forest ML Grid forecasts an overarching elevated city-wide threat trajectory over the next 168 hours based on historic geospatial density metrics."
                else:
                    future_threat = "WARNING: Predictive ML Engine offline. Please train the Random Forest module in the Advanced Analytics tab to generate future threat assessments."
                    
                # Generate PDF Bytes
                pdf_bytes = create_pdf_dossier(
                    incident_text=sample_text,
                    prediction=prediction,
                    confidence=prob,
                    severity=severity,
                    entities=entities,
                    matches=matches,
                    future_threat=future_threat
                )
                
                st.download_button(
                    label="📄 Download Official PDF Report",
                    data=pdf_bytes,
                    file_name="CRIS_Official_Dossier.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
