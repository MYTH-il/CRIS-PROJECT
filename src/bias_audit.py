import pandas as pd


def run_bias_audit(df, sensitive_cols, target_col="crime_type"):
    """
    Scaffolding for bias audits: returns per-group counts and target distribution.
    """
    results = {}
    if target_col not in df.columns:
        return results

    for col in sensitive_cols:
        if col not in df.columns:
            continue
        group_counts = df[col].value_counts(dropna=False).to_dict()
        group_target = (
            df.groupby(col)[target_col]
            .value_counts(normalize=True)
            .rename("share")
            .reset_index()
            .to_dict(orient="records")
        )
        results[col] = {
            "counts": group_counts,
            "target_distribution": group_target,
        }
    return results

