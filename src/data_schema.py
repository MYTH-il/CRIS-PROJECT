import pandas as pd


def validate_dataset(df):
    """
    Lightweight, non-destructive validation for CRIS datasets.
    Returns a list of human-readable issues for UI display.
    """
    issues = []

    required = ["incident_description", "crime_type"]
    for col in required:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")
        else:
            missing = df[col].isna().sum()
            if missing > 0:
                issues.append(f"{col}: {missing} missing values")

    if "report_date" in df.columns:
        parsed = pd.to_datetime(df["report_date"], errors="coerce")
        invalid = parsed.isna().sum()
        if invalid > 0:
            issues.append(f"report_date: {invalid} invalid/unparseable values")
    else:
        issues.append("Missing column: report_date (temporal analytics limited)")

    if "latitude" in df.columns and "longitude" in df.columns:
        lat = pd.to_numeric(df["latitude"], errors="coerce")
        lon = pd.to_numeric(df["longitude"], errors="coerce")
        invalid_lat = lat.isna().sum()
        invalid_lon = lon.isna().sum()
        if invalid_lat > 0:
            issues.append(f"latitude: {invalid_lat} invalid/unparseable values")
        if invalid_lon > 0:
            issues.append(f"longitude: {invalid_lon} invalid/unparseable values")
    else:
        issues.append("Missing latitude/longitude (geospatial analytics limited)")

    return issues

