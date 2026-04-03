import streamlit as st
import pandas as pd
import plotly.express as px
import os
from src.classification import train_baseline_model, predict_crime_type
from src.ner_extraction import extract_entities

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
    tab1, tab2, tab3, tab4 = st.tabs(["Data Viewer", "Missing Values", "Basic Distributions", "Intelligence Demo"])
    
    with tab1:
        st.subheader("Raw Data Sample")
        st.dataframe(df.head(100), use_container_width=True)
        
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
        st.subheader("Categorical Distributions")
        # Find potential categorical columns (limiting to less than 50 unique values to avoid memory explosion)
        categorical_cols = [col for col in df.columns if df[col].nunique() < 50 and df[col].dtype in ['object', 'category']]
        
        if categorical_cols:
            selected_col = st.selectbox("Select a column to visualize:", categorical_cols)
            val_counts = df[selected_col].value_counts().reset_index()
            val_counts.columns = [selected_col, "Count"]
            
            fig = px.pie(val_counts, names=selected_col, values="Count", title=f"Distribution of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not identify suitable categorical columns automatically.")

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
                colA, colB = st.columns(2)
                
                with colA:
                    st.markdown("#### 🔍 Predicted Crime Category")
                    with st.spinner("Classifying..."):
                        prediction, prob = predict_crime_type(sample_text)
                    if prediction:
                        st.success(f"**{prediction}**")
                        st.metric("Confidence", f"{prob:.1f}%")
                    else:
                        st.error(prob) # Shows "Model not trained yet."
                        
                with colB:
                    st.markdown("#### 🏷️ Extracted Entities (NER)")
                    with st.spinner("Extracting..."):
                        entities = extract_entities(sample_text)
                    
                    if entities:
                        for label, items in entities.items():
                            st.markdown(f"**{label}**: {', '.join(items)}")
                    else:
                        st.info("No notable entities found.")
