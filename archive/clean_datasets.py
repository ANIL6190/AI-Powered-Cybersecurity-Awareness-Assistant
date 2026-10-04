"""
Data Cleaning and Preprocessing Pipeline
AI-Powered Cybersecurity Awareness Assistant
---------------------------------------------
This script performs end-to-end data cleaning, auditing, deduplication, schema normalization,
text sanitization, datetime parsing, and quality validation on all cybersecurity datasets.
"""

import os
import re
import html
import json
import pandas as pd
import numpy as np

OUTPUT_DIR = "cleaned_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

audit_log = {}

def clean_sms_spam():
    print("Processing SMSSpamCollection...")
    raw_file = "SMSSpamCollection"
    
    rows = []
    with open(raw_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split('\t', 1)
            if len(parts) == 2:
                rows.append({"raw_label": parts[0].strip(), "text": parts[1].strip()})
    
    df = pd.DataFrame(rows)
    initial_count = len(df)
    
    # 1. Clean text: HTML decode, strip leading/trailing whitespace, replace multi-spaces
    df['clean_text'] = df['text'].apply(lambda x: html.unescape(x))
    df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    
    # 2. Text metrics
    df['char_length'] = df['clean_text'].apply(len)
    df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
    
    # 3. Label encoding: ham -> 0, spam -> 1
    df['label'] = df['raw_label'].map({'ham': 0, 'spam': 1})
    
    # 4. Deduplication
    dups_raw = df.duplicated(subset=['text']).sum()
    dups_clean = df.duplicated(subset=['clean_text', 'label']).sum()
    df_clean = df.drop_duplicates(subset=['clean_text', 'label']).reset_index(drop=True)
    final_count = len(df_clean)
    
    # Reorder columns
    df_clean = df_clean[['raw_label', 'label', 'clean_text', 'char_length', 'word_count']]
    
    out_csv = os.path.join(OUTPUT_DIR, "sms_spam_cleaned.csv")
    out_json = os.path.join(OUTPUT_DIR, "sms_spam_cleaned.json")
    df_clean.to_csv(out_csv, index=False)
    df_clean.to_json(out_json, orient='records', indent=2)
    
    stats = {
        "initial_rows": initial_count,
        "final_rows": final_count,
        "duplicates_removed": dups_clean,
        "label_distribution": df_clean['raw_label'].value_counts().to_dict(),
        "numeric_label_distribution": df_clean['label'].value_counts().to_dict(),
        "avg_char_length": round(df_clean['char_length'].mean(), 2),
        "avg_word_count": round(df_clean['word_count'].mean(), 2)
    }
    audit_log["SMSSpamCollection"] = stats
    print(f"SMSSpamCollection cleaned: {initial_count} -> {final_count} rows ({dups_clean} duplicates removed).")
    return stats

def clean_phiusiil():
    print("Processing PhiUSIIL_Phishing_URL_Dataset.csv...")
    raw_file = "PhiUSIIL_Phishing_URL_Dataset.csv"
    
    df = pd.read_csv(raw_file)
    initial_count = len(df)
    
    # 1. Strip UTF-8 BOM from column names
    df.columns = [col.replace('\ufeff', '').strip() for col in df.columns]
    
    # 2. Fix typos in column names
    rename_dict = {
        'DegitRatioInURL': 'DigitRatioInURL',
        'NoOfDegitsInURL': 'NoOfDigitsInURL',
        'SpacialCharRatioInURL': 'SpecialCharRatioInURL'
    }
    df.rename(columns=rename_dict, inplace=True)
    
    # 3. Clean string columns (URL, Domain, Title, TLD)
    str_cols = ['URL', 'Domain', 'Title', 'TLD']
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(lambda x: html.unescape(x) if pd.notnull(x) else x)
            df[col] = df[col].apply(lambda x: re.sub(r'\s+', ' ', x).strip() if pd.notnull(x) else x)
    
    # 4. Deduplicate based on URL & label
    dups_count = df.duplicated(subset=['URL']).sum()
    df_clean = df.drop_duplicates(subset=['URL']).reset_index(drop=True)
    final_count = len(df_clean)
    
    # 5. Check nulls
    nulls_count = df_clean.isnull().sum().sum()
    
    out_csv = os.path.join(OUTPUT_DIR, "phiusiil_phishing_cleaned.csv")
    df_clean.to_csv(out_csv, index=False)
    
    stats = {
        "initial_rows": initial_count,
        "final_rows": final_count,
        "total_columns": len(df_clean.columns),
        "duplicates_removed": dups_count,
        "null_values_remaining": int(nulls_count),
        "label_distribution": df_clean['label'].value_counts().to_dict(),
        "renamed_columns": rename_dict
    }
    audit_log["PhiUSIIL_Phishing_URL"] = stats
    print(f"PhiUSIIL dataset cleaned: {initial_count} -> {final_count} rows ({dups_count} duplicate URLs removed).")
    return stats

def clean_verified_online():
    print("Processing verified_online.csv...")
    raw_file = "verified_online.csv"
    
    df = pd.read_csv(raw_file)
    initial_count = len(df)
    
    # 1. Clean HTML entities in target and url
    df['url'] = df['url'].astype(str).str.strip()
    df['target'] = df['target'].astype(str).apply(lambda x: html.unescape(x).strip())
    
    # 2. Convert timestamps to datetime ISO strings
    df['submission_time_parsed'] = pd.to_datetime(df['submission_time'], errors='coerce')
    df['verification_time_parsed'] = pd.to_datetime(df['verification_time'], errors='coerce')
    
    # 3. Handle duplicate URLs
    dups_count = df.duplicated(subset=['url']).sum()
    df_clean = df.drop_duplicates(subset=['url']).reset_index(drop=True)
    final_count = len(df_clean)
    
    # 4. Standardize target column (lowercase normalized target for grouping)
    df_clean['target_normalized'] = df_clean['target'].str.lower().str.replace(r'[^a-z0-9]', '_', regex=True)
    
    out_csv = os.path.join(OUTPUT_DIR, "verified_online_cleaned.csv")
    df_clean.to_csv(out_csv, index=False)
    
    stats = {
        "initial_rows": initial_count,
        "final_rows": final_count,
        "duplicates_removed": dups_count,
        "top_targets": df_clean['target'].value_counts().head(10).to_dict(),
        "parsed_datetimes_valid": int(df_clean['submission_time_parsed'].notnull().sum())
    }
    audit_log["verified_online"] = stats
    print(f"verified_online cleaned: {initial_count} -> {final_count} rows ({dups_count} duplicate URLs removed).")
    return stats

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    else:
        return obj

def run_pipeline():
    sms_stats = clean_sms_spam()
    phi_stats = clean_phiusiil()
    ver_stats = clean_verified_online()
    
    clean_audit_log = sanitize_for_json(audit_log)
    
    summary_path = os.path.join(OUTPUT_DIR, "cleaning_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(clean_audit_log, f, indent=2)
    print("\nData Cleaning Pipeline completed successfully!")
    print(f"Summary written to {summary_path}")

if __name__ == "__main__":
    run_pipeline()
