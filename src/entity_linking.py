import re
import pandas as pd


def _safe_contains(series, pattern):
    try:
        return series.str.contains(pattern, case=False, regex=True, na=False)
    except Exception:
        return series.astype(str).str.contains(pattern, case=False, regex=True, na=False)


def find_entity_links(df, entities, text_col="incident_description", max_results=5, sample_size=5000):
    """
    Find related cases by matching extracted entities in historical narratives.
    Returns a list of dicts with match counts and snippets.
    """
    if entities is None or not entities:
        return []
    if text_col not in df.columns:
        return []

    work_df = df.dropna(subset=[text_col]).copy()
    if sample_size and len(work_df) > sample_size:
        work_df = work_df.sample(sample_size, random_state=42)

    entity_terms = []
    for label, items in entities.items():
        for item in items:
            if not item:
                continue
            entity_terms.append(re.escape(str(item)))

    if not entity_terms:
        return []

    pattern = "|".join(entity_terms)
    mask = _safe_contains(work_df[text_col], pattern)
    matches = work_df[mask].copy()

    if len(matches) == 0:
        return []

    def count_hits(text):
        count = 0
        for term in entity_terms:
            if re.search(term, str(text), flags=re.IGNORECASE):
                count += 1
        return count

    matches["hit_count"] = matches[text_col].apply(count_hits)
    matches = matches.sort_values(by="hit_count", ascending=False).head(max_results)

    results = []
    for _, row in matches.iterrows():
        results.append({
            "hit_count": int(row["hit_count"]),
            "crime": row.get("crime_type", "Unknown"),
            "location": row.get("incident_location", "Unknown"),
            "date": row.get("report_date", "Unknown"),
            "snippet": str(row.get(text_col, ""))[:150] + "...",
        })
    return results

