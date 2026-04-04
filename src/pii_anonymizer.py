import re


def mask_pii(text):
    if not isinstance(text, str):
        return ""

    patterns = [
        (r"\b\d{4}\s?\d{4}\s?\d{4}\b", "[AADHAAR]"),
        (r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b", "[PHONE]"),
        (r"\b[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,2}\s?\d{4}\b", "[VEHICLE]"),
        (r"\bFIR/\d{4}/[A-Z]{2,4}/\d{4,6}\b", "[FIR]"),
        (r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[EMAIL]"),
    ]

    masked = text
    for pattern, repl in patterns:
        masked = re.sub(pattern, repl, masked, flags=re.IGNORECASE)
    return masked


def mask_dataframe(df, text_columns):
    if df is None or not text_columns:
        return df
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(mask_pii)
    return df
