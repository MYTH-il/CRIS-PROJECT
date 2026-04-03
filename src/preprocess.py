def clean_missing_values(df):
    """
    Handles missing values in the CRIS dataset.
    """
    # For now, just a placeholder structure
    return df.dropna(subset=['Category']) if 'Category' in df.columns else df

def normalize_text(text):
    """
    Basic NLP text normalization.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    return text
