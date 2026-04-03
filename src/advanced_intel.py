import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import networkx as nx
import plotly.graph_objects as go

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

class AdvancedIntelligenceEngine:
    def __init__(self, data_path, vectorizer_path):
        self.df = None
        self.vectorizer = None
        if os.path.exists(data_path):
            self.df = pd.read_csv(data_path)
        if os.path.exists(vectorizer_path):
            self.vectorizer = joblib.load(vectorizer_path)
            
        self.sia = SentimentIntensityAnalyzer()

    def calculate_severity_score(self, text):
        """
        Calculates a Distress & Severity Threat Triage Score (1-10) using VADER.
        Highly negative statements (distress) + long narratives increase the score.
        """
        if not text:
            return 1
            
        scores = self.sia.polarity_scores(text)
        # Compound is between -1 (most negative) and +1 (most positive)
        # We invert it because negative emotion = higher distress
        distress = -scores['compound']  # Now between -1 and 1 where 1 is highest distress
        
        # Scale distress (0 to 1 range approx)
        distress_scaled = (distress + 1) / 2
        
        # Base severity out of 10
        severity = distress_scaled * 8
        
        # Bump up for heavy distress words inherently missed by basic sentiment
        keywords = ['gun', 'weapon', 'dead', 'shooting', 'knife', 'blood', 'hostage', 'bomb', 'fire']
        for word in keywords:
            if word in text.lower():
                severity += 2
                
        # Cap at 10 and Floor at 1
        return max(1, min(10, round(severity)))

    def find_mo_similarity(self, input_text, text_col='incident_description', top_n=3):
        """
        Modus Operandi Matching: Uses Cosine Similarity against historical TF-IDF vectors 
        to find the most mathematically similar unsolved or historic cases.
        """
        if self.df is None or self.vectorizer is None or text_col not in self.df.columns:
            return None
            
        # We need a clean dataframe to compare against
        valid_df = self.df.dropna(subset=[text_col]).copy()
        
        # Re-vectorize historical data (in production, cache this matrix)
        # For prototype speed, we sample 5000 random cases to compare 
        if len(valid_df) > 5000:
            valid_df = valid_df.sample(5000, random_state=42)
            
        historic_vectors = self.vectorizer.transform(valid_df[text_col])
        input_vector = self.vectorizer.transform([input_text])
        
        # Calculate similarity
        similarities = cosine_similarity(input_vector, historic_vectors)[0]
        
        # Get top N indices
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        results = []
        valid_df_reset = valid_df.reset_index(drop=True)
        for idx in top_indices:
            score = similarities[idx] * 100
            if score > 0: # Only return if there's actually some similarity
                row = valid_df_reset.iloc[idx]
                results.append({
                    'similarity': score,
                    'crime': row.get('crime_type', 'Unknown'),
                    'location': row.get('incident_location', 'Unknown'),
                    'date': row.get('report_date', 'Unknown'),
                    'snippet': row[text_col][:150] + "..."
                })
        return results

    def build_syndicate_graph(self, entities_dict):
        """
        Builds a Polished, Premium Plotly Knowledge Graph networking the extracted entities.
        """
        if not entities_dict:
            return None
            
        G = nx.Graph()
        
        # Create central incident node
        G.add_node("Current Incident", type="incident", size=35)
        
        connection_count = 0
        # Add entity nodes and edges
        for category, items in entities_dict.items():
            for item in items:
                node_id = f"{item} ({category})"
                G.add_node(node_id, type=category, size=20)
                G.add_edge("Current Incident", node_id)
                connection_count += 1
                
        # Generate layout positions (increase 'k' for better spacing)
        pos = nx.spring_layout(G, k=0.8, seed=42)
        
        # Plotly Traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='rgba(200, 200, 200, 0.4)'),
            hoverinfo='none',
            mode='lines'
        )

        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        # Premium Apple-Like Color Palette
        color_map = {
            'incident': '#ff3b30', # Vibrant Red
            'PERSON': '#0a84ff',   # Neon Blue
            'LOCATION': '#30d158', # Neon Green
            'ORG': '#ff9f0a'       # Vibrant Orange
        }
        
        for node in G.nodes():
            x, y = pos[node]
            node_type = G.nodes[node]['type']
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_color.append(color_map.get(node_type, '#8e8e93'))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            textfont=dict(family="Inter, sans-serif", size=13, color="#e5e5ea"),
            marker=dict(
                color=node_color,
                size=[G.nodes[n]['size'] for n in G.nodes()],
                line=dict(width=2, color='#ffffff')
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title=dict(
                    text=f"<b>Intelligence Network Link Analysis</b><br><sup>Total Active Connections: {connection_count}</sup>",
                    font=dict(size=18, family="Inter, sans-serif", color="#ffffff")
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=60),
                paper_bgcolor="rgba(0,0,0,0)", # Transparent to match Streamlit theme
                plot_bgcolor="rgba(28,28,30, 0.5)", # Subtle dark container
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
        return fig
