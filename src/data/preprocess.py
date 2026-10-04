"""
Data Cleaning and Preprocessing Pipeline
AI-Powered Cybersecurity Awareness Assistant
------------------------------------------------
Stage 1: Cleaning and Preprocessing
Implements modular functions for all 11 stages required by DVC pipeline:
1. load_and_standardize_schema
2. profile_raw_data
3. handle_missing_and_bad_rows
4. remove_duplicates
5. clean_text
6. nlp_preprocess
7. url_specific_cleaning
8. handle_class_imbalance
9. split_data
10. prepare_features
11. validate_and_version
"""

import os
import re
import html
import json
import pickle
import logging
import argparse
import urllib.parse
from typing import Dict, List, Tuple, Any, Optional

import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("preprocess")

# Signal words to preserve during stopword filtering
DEFAULT_KEEP_SIGNALS = {
    "not", "no", "nor", "now", "urgent", "urgently", "free", "win", "won",
    "winner", "prize", "claim", "cash", "money", "alert", "warn", "warning",
    "security", "verify", "verification", "update", "action", "immediately",
    "stop", "limited", "offer", "call", "txt", "text", "code", "password",
    "login", "account", "bank", "banking", "blocked", "suspended", "safe",
    "danger", "risk", "fail", "important", "confirm", "secure", "threat",
    "expire", "expired", "pay", "payment", "billing", "unauthorized", "locked"
}

SUSPICIOUS_URL_KEYWORDS = [
    "login", "signin", "verify", "verification", "account", "update", "secure",
    "security", "banking", "authenticate", "webscr", "ebayisapi", "password",
    "wallet", "admin", "confirm", "billing", "support", "service", "appleid",
    "recovery", "credential", "suspend", "unlock"
]


# ==============================================================================
# 1. Load and Standardize the Schema
# ==============================================================================
def load_and_standardize_schema(
    sms_path: str = "SMSSpamCollection",
    phiusiil_path: str = "PhiUSIIL_Phishing_URL_Dataset.csv",
    verified_online_path: str = "verified_online.csv",
    email_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Step 1: Read each dataset (SMS, email, URL) and rename columns to:
    text | type | label | source.
    Maps labels to unified set: 'phishing', 'spam', 'legit'.
    Corrects PhiUSIIL label direction (1 = legitimate -> 'legit', 0 = phishing -> 'phishing').
    Fixes encodings (UTF-8) and casts to string types.
    """
    logger.info("Step 1: Loading and standardizing schemas across all datasets...")
    dfs = []

    # 1.1 SMS Spam Collection
    if os.path.exists(sms_path):
        logger.info(f"Loading SMS dataset from '{sms_path}'...")
        sms_rows = []
        with open(sms_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                parts = line_str.split("\t", 1)
                if len(parts) == 2:
                    raw_lbl, txt = parts[0].strip(), parts[1].strip()
                    # Label mapping: ham -> legit, spam -> spam
                    mapped_lbl = "legit" if raw_lbl.lower() == "ham" else ("spam" if raw_lbl.lower() == "spam" else "spam")
                    sms_rows.append({
                        "text": txt,
                        "type": "sms",
                        "label": mapped_lbl,
                        "source": "SMSSpamCollection"
                    })
        sms_df = pd.DataFrame(sms_rows)
        logger.info(f"Loaded {len(sms_df)} rows from SMSSpamCollection.")
        dfs.append(sms_df)
    else:
        logger.warning(f"SMS path '{sms_path}' not found, skipping.")

    # 1.2 PhiUSIIL Phishing URL Dataset
    if os.path.exists(phiusiil_path):
        logger.info(f"Loading PhiUSIIL dataset from '{phiusiil_path}'...")
        phi_df = pd.read_csv(phiusiil_path, encoding="utf-8", low_memory=False)
        # Strip UTF-8 BOM
        phi_df.columns = [col.replace("\ufeff", "").strip() for col in phi_df.columns]
        
        # Verify URL column
        url_col = "URL" if "URL" in phi_df.columns else [c for c in phi_df.columns if "url" in c.lower()][0]
        lbl_col = "label" if "label" in phi_df.columns else [c for c in phi_df.columns if "label" in c.lower()][0]
        
        # In PhiUSIIL: 1 = legitimate, 0 = phishing
        phi_clean = pd.DataFrame({
            "text": phi_df[url_col].astype(str),
            "type": "url",
            "label": phi_df[lbl_col].apply(lambda x: "legit" if int(x) == 1 else "phishing"),
            "source": "PhiUSIIL"
        })
        logger.info(f"Loaded {len(phi_clean)} rows from PhiUSIIL.")
        dfs.append(phi_clean)
    else:
        logger.warning(f"PhiUSIIL path '{phiusiil_path}' not found, skipping.")

    # 1.3 Verified Online (PhishTank)
    if os.path.exists(verified_online_path):
        logger.info(f"Loading PhishTank dataset from '{verified_online_path}'...")
        tank_df = pd.read_csv(verified_online_path, encoding="utf-8", low_memory=False)
        tank_df.columns = [col.replace("\ufeff", "").strip() for col in tank_df.columns]
        url_col = "url" if "url" in tank_df.columns else [c for c in tank_df.columns if "url" in c.lower()][0]
        
        # All rows in verified_online are confirmed phishing attacks
        tank_clean = pd.DataFrame({
            "text": tank_df[url_col].astype(str),
            "type": "url",
            "label": "phishing",
            "source": "PhishTank"
        })
        logger.info(f"Loaded {len(tank_clean)} rows from PhishTank verified_online.")
        dfs.append(tank_clean)
    else:
        logger.warning(f"PhishTank path '{verified_online_path}' not found, skipping.")

    # 1.4 Optional Email Dataset
    if email_path and os.path.exists(email_path):
        logger.info(f"Loading Email dataset from '{email_path}'...")
        email_df = pd.read_csv(email_path, encoding="utf-8", low_memory=False)
        text_col = "text" if "text" in email_df.columns else "email"
        label_col = "label" if "label" in email_df.columns else "class"
        
        def map_email_label(val):
            s = str(val).lower().strip()
            if s in ["1", "phishing", "phish"]:
                return "phishing"
            elif s in ["spam"]:
                return "spam"
            else:
                return "legit"

        email_clean = pd.DataFrame({
            "text": email_df[text_col].astype(str),
            "type": "email",
            "label": email_df[label_col].apply(map_email_label),
            "source": "EmailDataset"
        })
        logger.info(f"Loaded {len(email_clean)} rows from Email dataset.")
        dfs.append(email_clean)

    if not dfs:
        raise ValueError("No datasets could be loaded! Check paths in params.yaml.")

    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Standardize column data types
    combined_df["text"] = combined_df["text"].astype(str)
    combined_df["type"] = combined_df["type"].astype(str)
    combined_df["label"] = combined_df["label"].astype(str)
    combined_df["source"] = combined_df["source"].astype(str)

    logger.info(f"Step 1 Complete: Total combined raw rows: {len(combined_df):,}. Schema: {list(combined_df.columns)}")
    logger.info(f"Label distribution:\n{combined_df['label'].value_counts().to_dict()}")
    return combined_df


# ==============================================================================
# 2. Profile the Raw Data
# ==============================================================================
def profile_raw_data(
    df: pd.DataFrame,
    output_dir: str = "reports",
    max_profile_sample: int = 50000
) -> dict:
    """
    Step 2: Profile raw data before modifications.
    Computes missing values, duplicates, class balance, and text length distributions.
    Saves HTML profile report and JSON summary report.
    """
    logger.info("Step 2: Profiling raw data...")
    os.makedirs(output_dir, exist_ok=True)

    text_lens = df["text"].apply(len)
    word_counts = df["text"].apply(lambda x: len(x.split()))

    stats = {
        "total_rows": int(len(df)),
        "columns": list(df.columns),
        "missing_values": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
        "empty_text_rows": int((df["text"].str.strip() == "").sum()),
        "exact_duplicates": int(df.duplicated(subset=["text"]).sum()),
        "duplicates_by_text_and_label": int(df.duplicated(subset=["text", "label"]).sum()),
        "class_balance": {k: int(v) for k, v in df["label"].value_counts().to_dict().items()},
        "class_balance_pct": {k: round(v / len(df) * 100, 2) for k, v in df["label"].value_counts().to_dict().items()},
        "type_distribution": {k: int(v) for k, v in df["type"].value_counts().to_dict().items()},
        "source_distribution": {k: int(v) for k, v in df["source"].value_counts().to_dict().items()},
        "text_length": {
            "min": int(text_lens.min()),
            "max": int(text_lens.max()),
            "mean": round(float(text_lens.mean()), 2),
            "median": float(text_lens.median()),
            "q25": float(text_lens.quantile(0.25)),
            "q75": float(text_lens.quantile(0.75)),
            "q95": float(text_lens.quantile(0.95))
        },
        "word_count": {
            "min": int(word_counts.min()),
            "max": int(word_counts.max()),
            "mean": round(float(word_counts.mean()), 2),
            "median": float(word_counts.median())
        }
    }

    # Save JSON report
    raw_summary_path = os.path.join(output_dir, "raw_data_profile.json")
    with open(raw_summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved raw data profile JSON to '{raw_summary_path}'.")

    # Generate ydata-profiling HTML report
    html_path = os.path.join(output_dir, "raw_data_profile.html")
    try:
        from ydata_profiling import ProfileReport
        sample_df = df if len(df) <= max_profile_sample else df.sample(n=max_profile_sample, random_state=42)
        # Use minimal=True to generate the report rapidly without heavy pairwise interactions
        profile = ProfileReport(
            sample_df,
            title="Raw Data Quality Profile - Cybersecurity Awareness Assistant",
            minimal=True,
            explorative=False
        )
        profile.to_file(html_path)
        logger.info(f"Saved ydata-profiling raw report to '{html_path}'.")
    except Exception as e:
        logger.warning(f"Could not generate ydata-profiling HTML: {e}. Generating fallback HTML.")
        # Self-contained fallback HTML report
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Raw Data Quality Profile</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 30px; background: #0f172a; color: #f8fafc; }}
        .card {{ background: #1e293b; padding: 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #334155; }}
        h1, h2 {{ color: #38bdf8; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ color: #94a3b8; }}
        .badge {{ background: #0284c7; padding: 4px 8px; border-radius: 4px; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Raw Data Quality Profile Report</h1>
    <div class="card">
        <h2>Dataset Overview</h2>
        <p><strong>Total Rows:</strong> {stats['total_rows']:,}</p>
        <p><strong>Exact Duplicates:</strong> {stats['exact_duplicates']:,}</p>
        <p><strong>Missing / Null Values:</strong> {sum(stats['missing_values'].values())}</p>
    </div>
    <div class="card">
        <h2>Class Balance</h2>
        <table>
            <tr><th>Label</th><th>Count</th><th>Percentage</th></tr>
            {''.join(f"<tr><td><span class='badge'>{k}</span></td><td>{v:,}</td><td>{stats['class_balance_pct'][k]}%</td></tr>" for k, v in stats['class_balance'].items())}
        </table>
    </div>
    <div class="card">
        <h2>Text Length Distribution</h2>
        <p>Min: {stats['text_length']['min']} | Median: {stats['text_length']['median']} | Mean: {stats['text_length']['mean']} | Max: {stats['text_length']['max']} | 95th Percentile: {stats['text_length']['q95']}</p>
    </div>
</body>
</html>"""
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Saved fallback raw HTML report to '{html_path}'.")

    return stats


# ==============================================================================
# 3. Handle Missing and Bad Rows
# ==============================================================================
def handle_missing_and_bad_rows(
    df: pd.DataFrame,
    min_text_length: int = 3,
    remove_corrupted: bool = True
) -> pd.DataFrame:
    """
    Step 3: Handle missing and corrupted rows:
    - Drops rows with null/empty text or missing labels.
    - Removes rows with text shorter than min_text_length (< 3 chars).
    - Removes corrupted rows containing null bytes, replacement characters, or binary garbage.
    """
    logger.info("Step 3: Handling missing and bad rows...")
    initial_count = len(df)

    # 1. Drop nulls
    df_clean = df.dropna(subset=["text", "label", "type"]).copy()
    null_dropped = initial_count - len(df_clean)

    # 2. Trim whitespace and check length
    df_clean["text"] = df_clean["text"].astype(str).str.strip()
    df_clean = df_clean[df_clean["text"].str.len() >= min_text_length].copy()
    short_dropped = initial_count - null_dropped - len(df_clean)

    # 3. Filter corrupted / binary junk
    if remove_corrupted:
        def is_clean_string(s: str) -> bool:
            # Check for null bytes, unicode replacement character, or non-printable binary junk
            if "\x00" in s or "\ufffd" in s:
                return False
            # Check ratio of non-ASCII / non-printable characters
            non_printable = sum(1 for c in s if ord(c) < 32 and c not in "\n\r\t")
            if non_printable > 0:
                return False
            return True

        df_clean = df_clean[df_clean["text"].apply(is_clean_string)].copy()

    corrupt_dropped = initial_count - null_dropped - short_dropped - len(df_clean)
    final_count = len(df_clean)

    logger.info(
        f"Step 3 Complete: {initial_count:,} -> {final_count:,} rows. "
        f"(Dropped: {null_dropped} nulls, {short_dropped} too short (< {min_text_length}), "
        f"{corrupt_dropped} corrupted/binary junk)."
    )
    return df_clean.reset_index(drop=True)


# ==============================================================================
# 4. Remove Duplicates
# ==============================================================================
def remove_duplicates(
    df: pd.DataFrame,
    deduplicate_cross_dataset: bool = True
) -> pd.DataFrame:
    """
    Step 4: Remove exact duplicates and near-duplicates (after lowercasing and trimming).
    Performs cross-dataset deduplication to prevent data leakage across sources / train / test.
    """
    logger.info("Step 4: Removing exact and near duplicates...")
    initial_count = len(df)

    # Normalized text for near-duplicate detection
    norm_text = df["text"].astype(str).str.lower().str.strip().str.replace(r"\s+", " ", regex=True)

    if deduplicate_cross_dataset:
        # Deduplicate strictly across all datasets so identical text never appears twice
        # Keeping first occurrence
        mask = ~norm_text.duplicated(keep="first")
        df_clean = df[mask].copy()
    else:
        # Deduplicate per (normalized text, label)
        df_temp = df.copy()
        df_temp["_norm"] = norm_text
        df_clean = df_temp.drop_duplicates(subset=["_norm", "label"]).drop(columns=["_norm"]).copy()

    final_count = len(df_clean)
    dups_removed = initial_count - final_count

    logger.info(f"Step 4 Complete: Removed {dups_removed:,} duplicates ({initial_count:,} -> {final_count:,} rows).")
    return df_clean.reset_index(drop=True)


# ==============================================================================
# 5. Clean the Text
# ==============================================================================
def clean_text_single(
    text: str,
    url_token: str = "<URL>",
    email_token: str = "<EMAIL>",
    phone_token: str = "<PHONE>",
    amount_token: str = "<AMOUNT>",
    otp_token: str = "<OTP>"
) -> str:
    """
    Cleans a single text string:
    - Strips HTML tags and email headers
    - Converts to lowercase
    - Normalizes whitespace
    - Replaces URLs with <URL> token
    - Replaces emails with <EMAIL>, phone numbers with <PHONE>, amounts with <AMOUNT>, OTPs with <OTP>
    - Collapses repeated characters and normalizes emojis
    - Preserves security signals and tokens
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Unescape HTML entities
    text = html.unescape(text)

    # 2. Strip HTML tags: <a href=...>Click</a> -> Click
    text = re.sub(r"<[^>]+>", " ", text)

    # 3. Strip email headers (e.g. "Subject: ...", "From: ...")
    text = re.sub(r"^(From|To|Subject|Date|Cc|Bcc|Reply-To|Received):[^\n]*\n?", " ", text, flags=re.MULTILINE | re.IGNORECASE)

    # 4. Replace Email addresses with token (MUST run before URL regex so domain in email isn't consumed)
    email_regex = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    text = re.sub(email_regex, f" {email_token} ", text)

    # 5. Replace URLs with token (regex handles http, https, ftp, www, domain-like patterns)
    url_regex = r"(https?://\S+|www\.\S+|[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/\S*)?)"
    text = re.sub(url_regex, f" {url_token} ", text)

    # 6. Replace Phone numbers with token (+91 98765..., (123) 456-7890, etc.)
    phone_regex = r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"
    text = re.sub(phone_regex, f" {phone_token} ", text)

    # 7. Replace Monetary amounts ($500, £2000, 100 USD, 50 EUR, Rs. 500)
    amount_regex = r"(?:[$€£₹]|rs\.?|inr|usd)\s*\d+(?:[.,]\d+)?|\b\d+(?:[.,]\d+)?\s*(?:dollars|euros|pounds|rupees|gbp|usd)\b"
    text = re.sub(amount_regex, f" {amount_token} ", text, flags=re.IGNORECASE)

    # 8. Replace OTPs and security codes (e.g., "code is 123456", "OTP: 9876")
    text = re.sub(r"\b(code|otp|pin|password|token)\s*(?:is|:)?\s*(\d{4,8})\b", rf"\1 {otp_token}", text, flags=re.IGNORECASE)

    # 9. Lowercase
    text = text.lower()

    # 10. Handle repeated characters (e.g. 'freeee!!!' -> 'free!!')
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # 11. Normalize whitespace and special characters
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_text_corpus(
    df: pd.DataFrame,
    url_token: str = "<URL>",
    email_token: str = "<EMAIL>",
    phone_token: str = "<PHONE>",
    amount_token: str = "<AMOUNT>",
    otp_token: str = "<OTP>"
) -> pd.DataFrame:
    """
    Step 5: Clean the text corpus across all rows.
    Creates a 'cleaned_text' column while preserving original 'text' for reference.
    """
    logger.info("Step 5: Cleaning text (HTML stripping, tokenization of URLs/phones/emails, normalization)...")
    
    # Apply text cleaning
    df_clean = df.copy()
    df_clean["cleaned_text"] = df_clean["text"].apply(
        lambda t: clean_text_single(
            t, url_token=url_token, email_token=email_token,
            phone_token=phone_token, amount_token=amount_token, otp_token=otp_token
        )
    )
    
    logger.info("Step 5 Complete: Text cleaned successfully.")
    return df_clean


# ==============================================================================
# 6. NLP Preprocessing (NLTK)
# ==============================================================================
def nlp_preprocess(
    df: pd.DataFrame,
    keep_signals: Optional[set] = None
) -> pd.DataFrame:
    """
    Step 6: NLP preprocessing using NLTK:
    - Tokenization
    - Stopword removal (retaining critical signal and urgency words)
    - WordNet Lemmatization with in-memory memoization
    - Preservation of punctuation signals ('!', '$') and special tokens
    Produces 'processed_text' column.
    """
    logger.info("Step 6: Running NLP preprocessing (tokenization, stopword filtering, lemmatization)...")
    
    signals = keep_signals if keep_signals is not None else DEFAULT_KEEP_SIGNALS
    stop_words = set(stopwords.words("english")) - set(signals)
    lemmatizer = WordNetLemmatizer()
    lemma_cache: Dict[str, str] = {}

    def memoized_lemmatize(word: str) -> str:
        if word not in lemma_cache:
            # Lemmatize as verb then noun
            lemma = lemmatizer.lemmatize(word, pos="v")
            if lemma == word:
                lemma = lemmatizer.lemmatize(word, pos="n")
            lemma_cache[word] = lemma
        return lemma_cache[word]

    special_tokens = {"<url>", "<email>", "<phone>", "<amount>", "<otp>"}

    def merge_special_tokens(tokens: List[str]) -> List[str]:
        # word_tokenize splits "<url>" into "<", "url", ">"; re-join placeholders so they survive.
        merged, i = [], 0
        while i < len(tokens):
            if i + 2 < len(tokens) and tokens[i] == "<" and tokens[i + 2] == ">" \
                    and f"<{tokens[i + 1].lower()}>" in special_tokens:
                merged.append(f"<{tokens[i + 1].lower()}>")
                i += 3
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def preprocess_sentence(text: str) -> str:
        # Tokenize words
        tokens = merge_special_tokens(word_tokenize(text))
        cleaned_tokens = []
        for t in tokens:
            t_lower = t.lower()
            if t_lower in special_tokens:
                cleaned_tokens.append(t_lower)
            elif t in ["!", "$"]:
                cleaned_tokens.append(t)
            elif t_lower in stop_words:
                continue
            elif t_lower.isalnum():
                lemma = memoized_lemmatize(t_lower)
                cleaned_tokens.append(lemma)
        return " ".join(cleaned_tokens) if cleaned_tokens else text

    df_nlp = df.copy()
    df_nlp["processed_text"] = df_nlp["cleaned_text"].apply(preprocess_sentence)
    
    logger.info(f"Step 6 Complete: NLP preprocessing done. Cached {len(lemma_cache):,} unique lemmas.")
    return df_nlp


# ==============================================================================
# 7. URL-Specific Cleaning & Feature Extraction
# ==============================================================================
def extract_single_url_features(url_str: str) -> dict:
    """Extracts security and lexical features from a URL."""
    try:
        parsed = urllib.parse.urlparse(url_str)
        domain = parsed.netloc.lower() if parsed.netloc else parsed.path.split("/")[0].lower()
    except Exception:
        domain = ""

    # Percent-decoding
    decoded_url = urllib.parse.unquote(url_str).lower().rstrip("/")

    # Check for IP address in domain
    ip_pattern = r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?::\d+)?$"
    has_ip = 1 if re.match(ip_pattern, domain) else 0

    # Count subdomains
    domain_parts = domain.split(".")
    num_subdomains = max(0, len(domain_parts) - 2) if len(domain_parts) > 1 else 0

    # Suspicious keywords in URL
    kw_count = sum(1 for kw in SUSPICIOUS_URL_KEYWORDS if kw in decoded_url)

    return {
        "url_length": len(decoded_url),
        "num_dots": decoded_url.count("."),
        "has_at": 1 if "@" in decoded_url else 0,
        "has_ip": has_ip,
        "num_subdomains": num_subdomains,
        "is_https": 1 if decoded_url.startswith("https://") else 0,
        "suspicious_keywords_count": kw_count
    }


def url_specific_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 7: URL-specific cleaning & feature extraction:
    - Normalizes URLs (domain lowercasing, trailing slash removal, percent-decoding)
    - Extracts tabular features: length, dots, '@', IP address, subdomains, HTTPS flag, suspicious keywords
    - Drops identifier columns that leak the answer (FILENAME, Domain, Title)
    """
    logger.info("Step 7: Performing URL-specific cleaning and feature extraction...")
    
    features_list = []
    for idx, row in df.iterrows():
        if row["type"] == "url":
            feats = extract_single_url_features(row["text"])
        else:
            # For non-URL rows, fill neutral default features
            feats = {
                "url_length": 0,
                "num_dots": 0,
                "has_at": 0,
                "has_ip": 0,
                "num_subdomains": 0,
                "is_https": 0,
                "suspicious_keywords_count": 0
            }
        features_list.append(feats)

    features_df = pd.DataFrame(features_list)
    df_combined = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

    # Drop identifier columns that leak the answer if present
    leak_cols = ["FILENAME", "Domain", "Title", "phish_detail_url", "submission_time", "verification_time"]
    existing_leaks = [c for c in leak_cols if c in df_combined.columns]
    if existing_leaks:
        df_combined = df_combined.drop(columns=existing_leaks)
        logger.info(f"Dropped answer-leaking identifier columns: {existing_leaks}")

    logger.info("Step 7 Complete: Extracted 7 tabular URL features.")
    return df_combined


# ==============================================================================
# 8. Handle Class Imbalance
# ==============================================================================
def handle_class_imbalance(
    train_df: pd.DataFrame,
    strategy: str = "class_weights"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Step 8: Handle class imbalance:
    - Checks label counts.
    - Computes balanced class weights.
    - Supports 'class_weights', 'oversample', or 'undersample'.
    - CRITICAL: Applied ONLY to the training set!
    """
    logger.info(f"Step 8: Handling class imbalance using strategy='{strategy}' (Applied to TRAIN set ONLY)...")
    
    label_counts = train_df["label"].value_counts().to_dict()
    classes = np.array(sorted(list(label_counts.keys())))
    
    # Calculate balanced class weights
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_df["label"])
    class_weights_dict = {cls: round(float(w), 4) for cls, w in zip(classes, weights)}
    logger.info(f"Computed training class weights: {class_weights_dict}")

    resampled_df = train_df.copy()
    if strategy == "oversample":
        max_size = max(label_counts.values())
        resampled_dfs = []
        for cls in classes:
            sub = train_df[train_df["label"] == cls]
            resampled = sub.sample(n=max_size, replace=True, random_state=42)
            resampled_dfs.append(resampled)
        resampled_df = pd.concat(resampled_dfs, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
        logger.info(f"Oversampled training set to balanced size: {len(resampled_df):,} rows.")
    elif strategy == "undersample":
        min_size = min(label_counts.values())
        resampled_dfs = []
        for cls in classes:
            sub = train_df[train_df["label"] == cls]
            resampled = sub.sample(n=min_size, replace=False, random_state=42)
            resampled_dfs.append(resampled)
        resampled_df = pd.concat(resampled_dfs, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
        logger.info(f"Undersampled training set to balanced size: {len(resampled_df):,} rows.")

    metadata = {
        "strategy": strategy,
        "original_train_distribution": label_counts,
        "final_train_distribution": resampled_df["label"].value_counts().to_dict(),
        "class_weights": class_weights_dict
    }

    return resampled_df, metadata


# ==============================================================================
# 9. Split the Data
# ==============================================================================
def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Step 9: Split data into Train, Validation, and Test sets (e.g. 70/15/15),
    stratified by label.
    Splitting occurs BEFORE fitting any vectorizer or tokenizer to prevent data leakage.
    Random seed fixed in params.yaml.
    """
    logger.info(f"Step 9: Performing stratified split ({train_ratio*100:.0f}/{val_ratio*100:.0f}/{test_ratio*100:.0f}) with seed={random_seed}...")
    
    # Validation of ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    assert np.isclose(total_ratio, 1.0), f"Ratios must sum to 1.0, got {total_ratio}"

    # Step 1: Split into train and temporary (val + test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df["label"],
        random_state=random_seed
    )

    # Step 2: Split temp into validation and test sets
    val_share_of_temp = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_share_of_temp),
        stratify=temp_df["label"],
        random_state=random_seed
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(f"Split results:")
    logger.info(f"  Train set: {len(train_df):,} rows ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"  Validation set: {len(val_df):,} rows ({len(val_df)/len(df)*100:.1f}%)")
    logger.info(f"  Test set: {len(test_df):,} rows ({len(test_df)/len(df)*100:.1f}%)")

    # Verify zero data leakage
    train_texts = set(train_df["text"].str.lower().str.strip())
    val_texts = set(val_df["text"].str.lower().str.strip())
    test_texts = set(test_df["text"].str.lower().str.strip())

    train_val_overlap = len(train_texts.intersection(val_texts))
    train_test_overlap = len(train_texts.intersection(test_texts))
    assert train_val_overlap == 0, f"Data leakage detected! {train_val_overlap} train/val overlap."
    assert train_test_overlap == 0, f"Data leakage detected! {train_test_overlap} train/test overlap."
    logger.info("Zero data leakage verified across Train, Validation, and Test splits.")

    return train_df, val_df, test_df


# ==============================================================================
# 10. Feature Preparation (TF-IDF & Deep Learning Tokenizer)
# ==============================================================================
class SimpleTokenizer:
    """
    Lightweight, Keras-compatible sequence tokenizer.
    Fits vocabulary only on training text, converts texts to padded sequences,
    and serializes directly to tokenizer.json.
    """
    def __init__(self, num_words: int = 10000, oov_token: str = "<OOV>", max_len: int = 100):
        self.num_words = num_words
        self.oov_token = oov_token
        self.max_len = max_len
        self.word_index: Dict[str, int] = {}
        self.index_word: Dict[int, str] = {}
        self.word_counts: Dict[str, int] = {}

    def fit_on_texts(self, texts: List[str]):
        counts: Dict[str, int] = {}
        for t in texts:
            words = str(t).split()
            for w in words:
                counts[w] = counts.get(w, 0) + 1
        self.word_counts = counts

        # Sort by frequency
        sorted_words = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        self.word_index = {self.oov_token: 1}
        idx = 2
        for w, _ in sorted_words:
            if idx > self.num_words:
                break
            self.word_index[w] = idx
            idx += 1
        self.index_word = {v: k for k, v in self.word_index.items()}

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        oov_idx = self.word_index.get(self.oov_token, 1)
        sequences = []
        for t in texts:
            seq = [self.word_index.get(w, oov_idx) for w in str(t).split()]
            sequences.append(seq)
        return sequences

    def pad_sequences(self, sequences: List[List[int]], padding: str = "post", truncating: str = "post") -> np.ndarray:
        padded = np.zeros((len(sequences), self.max_len), dtype=np.int32)
        for i, seq in enumerate(sequences):
            if not seq:
                continue
            if len(seq) > self.max_len:
                if truncating == "post":
                    s = seq[:self.max_len]
                else:
                    s = seq[-self.max_len:]
            else:
                s = seq
            if padding == "post":
                padded[i, :len(s)] = s
            else:
                padded[i, -len(s):] = s
        return padded

    def to_json(self) -> str:
        data = {
            "num_words": self.num_words,
            "oov_token": self.oov_token,
            "max_len": self.max_len,
            "vocab_size": len(self.word_index),
            "word_index": self.word_index
        }
        return json.dumps(data, indent=2)

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())


def prepare_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tfidf_max_features: int = 5000,
    tfidf_ngram_range: Tuple[int, int] = (1, 2),
    seq_max_vocab: int = 10000,
    seq_max_len: int = 100,
    output_dir: str = "data/processed"
) -> dict:
    """
    Step 10: Feature preparation:
    - Baseline: Fit TF-IDF vectorizer ONLY on training text, transform val and test.
    - Deep Learning: Fit Keras-compatible sequence tokenizer on training text, pad to fixed length.
    - Serializes tokenizer.json and tfidf_vectorizer.pkl.
    - Saves processed datasets (train.csv, val.csv, test.csv) as DVC stage outputs.
    """
    logger.info("Step 10: Preparing features (TF-IDF & Sequence Tokenizer)...")
    os.makedirs(output_dir, exist_ok=True)

    # 10.1 TF-IDF Vectorizer (Fit strictly on TRAIN text only)
    logger.info("Fitting TF-IDF Vectorizer on training text...")
    tfidf = TfidfVectorizer(
        max_features=tfidf_max_features,
        ngram_range=tfidf_ngram_range,
        min_df=2
    )
    tfidf.fit(train_df["processed_text"])
    
    tfidf_path = os.path.join(output_dir, "tfidf_vectorizer.pkl")
    with open(tfidf_path, "wb") as f:
        pickle.dump(tfidf, f)
    logger.info(f"Saved TF-IDF vectorizer to '{tfidf_path}'. Vocabulary size: {len(tfidf.vocabulary_)}.")

    # 10.2 Sequence Tokenizer for BiLSTM / GRU
    logger.info("Fitting Sequence Tokenizer on training text...")
    tokenizer = SimpleTokenizer(num_words=seq_max_vocab, max_len=seq_max_len)
    tokenizer.fit_on_texts(train_df["processed_text"].tolist())

    tokenizer_json_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_json_path)
    logger.info(f"Saved Tokenizer to '{tokenizer_json_path}'. Vocab size: {len(tokenizer.word_index)}.")

    # 10.3 Convert to padded sequences and append sequence string representation
    train_seqs = tokenizer.pad_sequences(tokenizer.texts_to_sequences(train_df["processed_text"].tolist()))
    val_seqs = tokenizer.pad_sequences(tokenizer.texts_to_sequences(val_df["processed_text"].tolist()))
    test_seqs = tokenizer.pad_sequences(tokenizer.texts_to_sequences(test_df["processed_text"].tolist()))

    train_out = train_df.copy()
    val_out = val_df.copy()
    test_out = test_df.copy()

    # Save sequence arrays as compressed numpy files for fast DL loading
    np.save(os.path.join(output_dir, "train_sequences.npy"), train_seqs)
    np.save(os.path.join(output_dir, "val_sequences.npy"), val_seqs)
    np.save(os.path.join(output_dir, "test_sequences.npy"), test_seqs)

    # Save CSV outputs
    train_csv_path = os.path.join(output_dir, "train.csv")
    val_csv_path = os.path.join(output_dir, "val.csv")
    test_csv_path = os.path.join(output_dir, "test.csv")

    train_out.to_csv(train_csv_path, index=False)
    val_out.to_csv(val_csv_path, index=False)
    test_out.to_csv(test_csv_path, index=False)

    logger.info(f"Step 10 Complete: Processed data saved to '{output_dir}'.")
    return {
        "tfidf_vocab_size": len(tfidf.vocabulary_),
        "tokenizer_vocab_size": len(tokenizer.word_index),
        "train_rows": len(train_out),
        "val_rows": len(val_out),
        "test_rows": len(test_out),
        "sequence_shape": list(train_seqs.shape)
    }


# ==============================================================================
# 11. Validate and Version
# ==============================================================================
def validate_and_version(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    raw_profile_path: str = "reports/raw_data_profile.json",
    output_dir: str = "reports"
) -> dict:
    """
    Step 11: Validate and version:
    - Re-profiles cleaned datasets and generates comparison report against raw data.
    - Validates data integrity: no nulls, valid labels, zero train/test overlap.
    - Generates validation metrics and updates documentation.
    """
    logger.info("Step 11: Validating processed datasets and generating profile comparison...")
    os.makedirs(output_dir, exist_ok=True)

    combined_cleaned = pd.concat([train_df, val_df, test_df], ignore_index=True)
    clean_lens = combined_cleaned["processed_text"].apply(len)

    # Check 1: No nulls in key columns
    null_counts = combined_cleaned[["text", "processed_text", "label", "type"]].isnull().sum().to_dict()
    assert sum(null_counts.values()) == 0, f"Integrity check failed! Nulls found: {null_counts}"

    # Check 2: Valid labels
    valid_labels = {"phishing", "spam", "legit"}
    actual_labels = set(combined_cleaned["label"].unique())
    assert actual_labels.issubset(valid_labels), f"Integrity check failed! Invalid labels: {actual_labels - valid_labels}"

    # Check 3: Zero train/test overlap
    train_texts = set(train_df["text"].str.lower().str.strip())
    test_texts = set(test_df["text"].str.lower().str.strip())
    overlap = len(train_texts.intersection(test_texts))
    assert overlap == 0, f"Integrity check failed! {overlap} overlapping samples between train and test."

    # Load raw profile for before vs after comparison
    raw_stats = {}
    if os.path.exists(raw_profile_path):
        with open(raw_profile_path, "r", encoding="utf-8") as f:
            raw_stats = json.load(f)

    comparison = {
        "raw_data": {
            "total_rows": raw_stats.get("total_rows", 0),
            "duplicates": raw_stats.get("exact_duplicates", 0),
            "label_distribution": raw_stats.get("class_balance", {}),
            "mean_text_length": raw_stats.get("text_length", {}).get("mean", 0)
        },
        "cleaned_data": {
            "total_rows": len(combined_cleaned),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "label_distribution": combined_cleaned["label"].value_counts().to_dict(),
            "mean_processed_text_length": round(float(clean_lens.mean()), 2),
            "median_processed_text_length": float(clean_lens.median())
        },
        "data_quality_checks": {
            "null_values_in_key_fields": 0,
            "valid_labels_verified": True,
            "train_test_data_leakage": 0,
            "status": "PASSED"
        }
    }

    comparison_path = os.path.join(output_dir, "cleaning_comparison.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"Saved cleaning comparison metrics to '{comparison_path}'.")

    # Generate cleaned HTML profile report
    html_path = os.path.join(output_dir, "cleaned_data_profile.html")
    try:
        from ydata_profiling import ProfileReport
        sample_size = min(50000, len(combined_cleaned))
        sample_clean = combined_cleaned.sample(n=sample_size, random_state=42)
        report = ProfileReport(
            sample_clean[["type", "label", "source", "url_length", "suspicious_keywords_count"]],
            title="Cleaned Data Quality Profile - Cybersecurity Awareness Assistant",
            minimal=True,
            explorative=False
        )
        report.to_file(html_path)
        logger.info(f"Saved ydata-profiling cleaned report to '{html_path}'.")
    except Exception as e:
        logger.warning(f"Could not generate ydata-profiling HTML for cleaned data: {e}.")

    logger.info("Step 11 Complete: All validation checks passed!")
    return comparison


# ==============================================================================
# Pipeline Orchestrator (DVC Entry Point)
# ==============================================================================
def run_pipeline(config_path: str = "params.yaml") -> dict:
    """
    Executes the complete end-to-end data cleaning and preprocessing pipeline:
    Steps 1 to 11 orchestrated in strict sequential order.
    """
    logger.info(f"Starting Stage 1 Data Cleaning Pipeline using config '{config_path}'...")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Load & standardize
    raw_cfg = config.get("raw_data", {})
    df_raw = load_and_standardize_schema(
        sms_path=raw_cfg.get("sms_path", "SMSSpamCollection"),
        phiusiil_path=raw_cfg.get("phiusiil_path", "PhiUSIIL_Phishing_URL_Dataset.csv"),
        verified_online_path=raw_cfg.get("verified_online_path", "verified_online.csv"),
        email_path=raw_cfg.get("email_path")
    )

    # 2. Profile raw data
    out_cfg = config.get("output", {})
    reports_dir = out_cfg.get("reports_dir", "reports")
    profile_raw_data(df_raw, output_dir=reports_dir)

    # 3. Handle missing and bad rows
    clean_cfg = config.get("cleaning", {})
    df_no_bad = handle_missing_and_bad_rows(
        df_raw,
        min_text_length=clean_cfg.get("min_text_length", 3),
        remove_corrupted=clean_cfg.get("remove_corrupted", True)
    )

    # 4. Remove duplicates
    df_dedup = remove_duplicates(
        df_no_bad,
        deduplicate_cross_dataset=clean_cfg.get("deduplicate_cross_dataset", True)
    )

    # 5. Clean text
    df_clean_text = clean_text_corpus(
        df_dedup,
        url_token=clean_cfg.get("url_token", "<URL>"),
        email_token=clean_cfg.get("email_token", "<EMAIL>"),
        phone_token=clean_cfg.get("phone_token", "<PHONE>"),
        amount_token=clean_cfg.get("amount_token", "<AMOUNT>"),
        otp_token=clean_cfg.get("otp_token", "<OTP>")
    )

    # 6. NLP preprocessing (NLTK)
    keep_signals = set(clean_cfg.get("keep_signals", [])) if clean_cfg.get("keep_signals") else None
    df_nlp = nlp_preprocess(df_clean_text, keep_signals=keep_signals)

    # 7. URL-specific cleaning and feature extraction
    df_features = url_specific_cleaning(df_nlp)

    # 8. Split the data (BEFORE fitting any feature extraction or handling imbalance)
    split_cfg = config.get("split", {})
    base_cfg = config.get("base", {})
    seed = base_cfg.get("random_seed", 42)
    train_df, val_df, test_df = split_data(
        df_features,
        train_ratio=split_cfg.get("train_ratio", 0.70),
        val_ratio=split_cfg.get("val_ratio", 0.15),
        test_ratio=split_cfg.get("test_ratio", 0.15),
        random_seed=seed
    )

    # 9. Handle class imbalance (TRAINING SET ONLY)
    imb_cfg = config.get("imbalance", {})
    train_balanced, imb_meta = handle_class_imbalance(
        train_df,
        strategy=imb_cfg.get("strategy", "class_weights")
    )

    # 10. Feature preparation (TF-IDF & Tokenizer)
    feat_cfg = config.get("features", {})
    tfidf_cfg = feat_cfg.get("tfidf", {})
    seq_cfg = feat_cfg.get("sequence", {})
    processed_dir = out_cfg.get("processed_dir", "data/processed")

    prepare_features(
        train_balanced, val_df, test_df,
        tfidf_max_features=tfidf_cfg.get("max_features", 5000),
        tfidf_ngram_range=tuple(tfidf_cfg.get("ngram_range", [1, 2])),
        seq_max_vocab=seq_cfg.get("max_vocab_size", 10000),
        seq_max_len=seq_cfg.get("max_sequence_length", 100),
        output_dir=processed_dir
    )

    # 11. Validate and version
    raw_profile_path = os.path.join(reports_dir, "raw_data_profile.json")
    validation_summary = validate_and_version(
        train_balanced, val_df, test_df,
        raw_profile_path=raw_profile_path,
        output_dir=reports_dir
    )

    logger.info("Pipeline execution finished successfully!")
    return validation_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Cleaning and Preprocessing Pipeline")
    parser.add_argument("--config", type=str, default="params.yaml", help="Path to params.yaml configuration file")
    args = parser.parse_args()
    run_pipeline(config_path=args.config)
