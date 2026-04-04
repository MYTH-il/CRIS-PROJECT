import os
import glob
import pandas as pd
from src.pii_anonymizer import mask_dataframe


def load_ncrb_tables(folder_path, anonymize=False, text_columns=None):
    """
    Load all NCRB CSV tables from a folder and return a concatenated DataFrame.
    Adds a 'source_file' column to preserve provenance.
    """
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        return pd.DataFrame()

    frames = []
    for path in csv_files:
        df = pd.read_csv(path)
        df["source_file"] = os.path.basename(path)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if anonymize:
        df = mask_dataframe(df, text_columns or [])
    return df


def load_rti_fir_summaries(folder_path, anonymize=False, text_columns=None):
    """
    Load RTI-obtained FIR summaries from CSV or JSONL files in a folder.
    Returns a single DataFrame with a 'source_file' column.
    """
    frames = []
    for path in glob.glob(os.path.join(folder_path, "*.csv")):
        df = pd.read_csv(path)
        df["source_file"] = os.path.basename(path)
        frames.append(df)

    for path in glob.glob(os.path.join(folder_path, "*.jsonl")):
        df = pd.read_json(path, lines=True)
        df["source_file"] = os.path.basename(path)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if anonymize:
        df = mask_dataframe(df, text_columns or [])
    return df


def load_judgment_corpus(folder_path, anonymize=False, text_columns=None):
    """
    Load plain-text judgments (Indian Kanoon / JUDIS) from .txt files.
    Returns a DataFrame with 'document_text' and 'source_file'.
    """
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    if not txt_files:
        return pd.DataFrame(columns=["document_text", "source_file"])

    records = []
    for path in txt_files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        records.append({"document_text": text, "source_file": os.path.basename(path)})

    df = pd.DataFrame(records)
    if anonymize:
        df = mask_dataframe(df, text_columns or ["document_text"])
    return df
