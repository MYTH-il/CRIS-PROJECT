import pandas as pd
import plotly.express as px
import numpy as np

def generate_hotspot_map(df, crime_filter=None):
    """
    Creates a spatial density map of incidents using Latitude and Longitude.
    Automatically drops rows missing geospatial data to prevent errors.
    """
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        return None

    # Filter data
    # Copy to avoid SettingWithCopyWarning
    map_df = df.dropna(subset=['latitude', 'longitude']).copy()
    
    if crime_filter and crime_filter != "All Crimes":
        if 'crime_type' in map_df.columns:
            map_df = map_df[map_df['crime_type'] == crime_filter]

    if len(map_df) == 0:
        return None

    # We sample down to max 5000 points to keep Streamlit browser fast
    if len(map_df) > 5000:
        map_df = map_df.sample(5000, random_state=42)

    # Convert coordinates to numeric just in case
    map_df['latitude'] = pd.to_numeric(map_df['latitude'], errors='coerce')
    map_df['longitude'] = pd.to_numeric(map_df['longitude'], errors='coerce')
    map_df = map_df.dropna(subset=['latitude', 'longitude'])

    # Build Mapbox Scatter
    fig = px.scatter_mapbox(
        map_df, 
        lat="latitude", 
        lon="longitude", 
        color="crime_type" if 'crime_type' in map_df.columns and map_df['crime_type'].nunique() < 20 else None,
        hover_name="incident_location" if 'incident_location' in map_df.columns else None,
        hover_data=["report_date", "time_of_day"],
        zoom=10, 
        height=600,
        title=f"Crime Hotspots ({crime_filter if crime_filter else 'All Crimes'})"
    )
    
    # Use open street map style so no API key is required
    fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":40,"l":0,"b":0})
    return fig

def generate_temporal_forecast(df, crime_filter=None):
    """
    Aggregates reports by date, calculates moving averages, 
    and returns a time-series plot.
    """
    if 'report_date' not in df.columns:
        return None

    time_df = df.copy()
    
    # Attempt to convert to datetime
    time_df['report_date'] = pd.to_datetime(time_df['report_date'], errors='coerce')
    time_df = time_df.dropna(subset=['report_date'])

    if crime_filter and crime_filter != "All Crimes":
        if 'crime_type' in time_df.columns:
            time_df = time_df[time_df['crime_type'] == crime_filter]

    if len(time_df) == 0:
        return None

    # Group by month (Period) and convert back to timestamp for plotly
    time_df['YearMonth'] = time_df['report_date'].dt.to_period('M').dt.to_timestamp()
    monthly_counts = time_df.groupby('YearMonth').size().reset_index(name='Incident Count')
    
    # Sort chronologically
    monthly_counts = monthly_counts.sort_values('YearMonth')
    
    # Calculate a simple 3-month moving average as our 'Baseline Forecast Trend'
    monthly_counts['Trend (3-Mo Average)'] = monthly_counts['Incident Count'].rolling(window=3, min_periods=1).mean()

    fig = px.line(
        monthly_counts, 
        x='YearMonth', 
        y=['Incident Count', 'Trend (3-Mo Average)'],
        title=f"Temporal Trend Analysis ({crime_filter if crime_filter else 'All Crimes'})",
        labels={'value': 'Number of Incidents', 'YearMonth': 'Date', 'variable': 'Metric'}
    )
    
    
    fig.update_layout(hovermode="x unified")
    return fig

def generate_crime_distribution_chart(df, crime_filter=None):
    """
    Shows a Pie chart of Crime Types OR a Bar chart of Top Locations if a crime is selected.
    """
    if 'crime_type' not in df.columns:
        return None
        
    plot_df = df.copy()
    if crime_filter and crime_filter != "All Crimes":
        # If they filtered a specific crime, show top 10 locations for that crime
        if 'incident_location' not in plot_df.columns:
            return None
        plot_df = plot_df[plot_df['crime_type'] == crime_filter]
        counts = plot_df['incident_location'].value_counts().reset_index().head(10)
        counts.columns = ['Location', 'Incident Count']
        fig = px.bar(counts, x='Incident Count', y='Location', orientation='h', 
                     title=f"Top 10 Locations for {crime_filter}", 
                     color='Incident Count', color_continuous_scale='Blues')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        return fig
    else:
        # Show all crime distributions
        counts = plot_df['crime_type'].value_counts().reset_index().head(10)
        counts.columns = ['Crime Type', 'Incident Count']
        fig = px.pie(counts, values='Incident Count', names='Crime Type', 
                     title="Distribution of Top 10 Crimes", hole=0.4, 
                     color_discrete_sequence=px.colors.sequential.RdBu)
        return fig

def generate_time_distribution_chart(df, crime_filter=None):
    """
    Shows a Bar Chart indicating what Time of Day crimes happen most.
    """
    if 'time_of_day' not in df.columns:
        return None
        
    plot_df = df.copy()
    if crime_filter and crime_filter != "All Crimes":
        plot_df = plot_df[plot_df['crime_type'] == crime_filter]
        
    counts = plot_df['time_of_day'].value_counts().reset_index()
    counts.columns = ['Time of Day', 'Incident Count']
    fig = px.bar(counts, x='Time of Day', y='Incident Count', 
                 title=f"Incident Frequencies by Time ({crime_filter if crime_filter else 'All Crimes'})", 
                 color='Incident Count', color_continuous_scale='Purples')
    
    # Optional: order categories logically if they are standard shifts (Morning, Afternoon, etc.)
    return fig
