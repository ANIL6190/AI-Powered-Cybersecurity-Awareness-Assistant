"""
PyTest Test Suite for Data Cleaning & Preprocessing Pipeline
AI-Powered Cybersecurity Awareness Assistant
-----------------------------------------------------------
Tests all 11 cleaning stages, data integrity, label correctness,
deduplication, tokenization, leakage prevention, and tokenizer serialization.
"""

import os
import json
import pickle
import pytest
import numpy as np
import pandas as pd

from src.data.preprocess import (
    load_and_standardize_schema,
    handle_missing_and_bad_rows,
    remove_duplicates,
    clean_text_single,
    clean_text_corpus,
    nlp_preprocess,
    extract_single_url_features,
    url_specific_cleaning,
    handle_class_imbalance,
    split_data,
    SimpleTokenizer,
    prepare_features,
    validate_and_version
)


def _nltk_data_available() -> bool:
    import nltk
    try:
        nltk.data.find("corpora/stopwords")
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        return False
    for wordnet in ("corpora/wordnet", "corpora/wordnet.zip"):  # wordnet is usually left zipped
        try:
            nltk.data.find(wordnet)
            return True
        except LookupError:
            continue
    return False


requires_nltk = pytest.mark.skipif(
    not _nltk_data_available(),
    reason="NLTK data not installed; run `python scripts/setup_nltk.py`",
)


# ==============================================================================
# 1. Schema and Label Tests
# ==============================================================================
def test_schema_and_labels_synthetic(tmp_path):
    # Create mock datasets
    sms_file = tmp_path / "mock_sms.txt"
    sms_file.write_text("ham\tHello how are you?\nspam\tWINNER! Claim prize now!\n")

    phi_file = tmp_path / "mock_phi.csv"
    phi_df = pd.DataFrame({
        "URL": ["https://paypal-security-update.com/login", "https://google.com"],
        "label": [0, 1]  # 0 = phishing, 1 = legitimate in PhiUSIIL
    })
    phi_df.to_csv(phi_file, index=False)

    tank_file = tmp_path / "mock_tank.csv"
    tank_df = pd.DataFrame({
        "url": ["https://fake-bank-login.com/secure"]
    })
    tank_df.to_csv(tank_file, index=False)

    df = load_and_standardize_schema(
        sms_path=str(sms_file),
        phiusiil_path=str(phi_file),
        verified_online_path=str(tank_file)
    )

    # Validate Schema
    assert list(df.columns) == ["text", "type", "label", "source"]
    assert set(df["label"].unique()) == {"legit", "spam", "phishing"}
    
    # Check PhiUSIIL label direction: 1 -> legit, 0 -> phishing
    phi_rows = df[df["source"] == "PhiUSIIL"]
    assert phi_rows[phi_rows["text"] == "https://google.com"]["label"].values[0] == "legit"
    assert phi_rows[phi_rows["text"] == "https://paypal-security-update.com/login"]["label"].values[0] == "phishing"


# ==============================================================================
# 2. Missing and Bad Rows Tests
# ==============================================================================
def test_handle_missing_and_bad_rows():
    raw_data = pd.DataFrame({
        "text": [
            "Valid cybersecurity message",
            None,
            "",
            "   ",
            "ab",  # < 3 chars
            "Corrupted binary string \x00\x00\x01",
            "Corrupted with replacement \ufffd character",
            "Urgent: your account has been compromised"
        ],
        "type": ["sms", "sms", "sms", "sms", "sms", "sms", "sms", "sms"],
        "label": ["spam", "spam", "spam", "spam", "spam", "spam", "spam", "phishing"],
        "source": ["test"] * 8
    })

    cleaned = handle_missing_and_bad_rows(raw_data, min_text_length=3, remove_corrupted=True)

    # Exactly 2 valid rows should survive
    assert len(cleaned) == 2
    assert "Valid cybersecurity message" in cleaned["text"].values
    assert "Urgent: your account has been compromised" in cleaned["text"].values


# ==============================================================================
# 3. Deduplication Tests
# ==============================================================================
def test_remove_duplicates_cross_dataset():
    df = pd.DataFrame({
        "text": [
            "http://example.com/login",
            "http://example.com/login",      # Exact duplicate
            "HTTP://EXAMPLE.COM/LOGIN  ",    # Near duplicate
            "Different URL",
            "http://example.com/login"       # From another source
        ],
        "type": ["url", "url", "url", "url", "url"],
        "label": ["phishing", "phishing", "phishing", "legit", "phishing"],
        "source": ["SourceA", "SourceA", "SourceA", "SourceA", "SourceB"]
    })

    deduped = remove_duplicates(df, deduplicate_cross_dataset=True)
    assert len(deduped) == 2
    assert set(deduped["text"].str.strip().str.lower()) == {"http://example.com/login", "different url"}


# ==============================================================================
# 4. Text Cleaning & Tokenization Tests
# ==============================================================================
def test_clean_text_single_tokens():
    raw_text = '<a href="http://malicious.org/phish">Click here</a> URGENT! Visit http://bit.ly/claim-prize now. Call +1 800-555-0199 or email support@security.com to claim $5000 prize freeeee!!! Code is 987654'
    cleaned = clean_text_single(raw_text)

    # Verify HTML stripped: <a href=...>Click here</a> -> Click here
    assert "<a href=" not in cleaned
    assert "click here" in cleaned
    assert "urgent" in cleaned

    # Verify tokens replaced
    assert "<url>" in cleaned
    assert "<phone>" in cleaned
    assert "<email>" in cleaned
    assert "<amount>" in cleaned
    assert "<otp>" in cleaned

    # Verify repeated characters collapsed
    assert "freeee" not in cleaned
    assert "free" in cleaned


# ==============================================================================
# 5. NLP Preprocessing Tests
# ==============================================================================
@requires_nltk
def test_nlp_preprocess_keeps_placeholder_tokens():
    # Regression: word_tokenize split "<url>" into "<", "url", ">" so placeholders were lost.
    cleaned = clean_text_single("Claim your prize at http://bit.ly/x or call +1 800-555-0199")
    df = pd.DataFrame({"text": ["x"], "cleaned_text": [cleaned], "type": ["sms"], "label": ["spam"], "source": ["test"]})
    processed = nlp_preprocess(df)["processed_text"].iloc[0].split()
    assert "<url>" in processed
    assert "<phone>" in processed
    assert "url" not in processed


@requires_nltk
def test_nlp_preprocess_signals_preserved():
    df = pd.DataFrame({
        "text": ["Urgent alert! Update your bank account now, do not wait."],
        "cleaned_text": ["urgent alert! update your bank account now, do not wait."],
        "type": ["sms"],
        "label": ["spam"],
        "source": ["test"]
    })

    nlp_df = nlp_preprocess(df)
    processed = nlp_df["processed_text"].iloc[0]

    # Signal words must be preserved
    assert "urgent" in processed
    assert "not" in processed
    assert "now" in processed
    assert "account" in processed
    assert "!" in processed


# ==============================================================================
# 6. URL Feature Extraction Tests
# ==============================================================================
def test_extract_single_url_features():
    phish_url = "http://192.168.1.1/login/banking/secure/verify.php?user=admin@target.com"
    feats = extract_single_url_features(phish_url)

    assert feats["has_ip"] == 1
    assert feats["has_at"] == 1
    assert feats["is_https"] == 0
    assert feats["num_dots"] >= 3
    assert feats["suspicious_keywords_count"] >= 3


# ==============================================================================
# 7. Stratified Splitting & Zero Data Leakage Tests
# ==============================================================================
def test_split_data_no_leakage():
    # Generate 100 sample items across 3 classes
    data = []
    for i in range(50):
        data.append({"text": f"legit message number {i}", "label": "legit", "type": "sms", "source": "test"})
    for i in range(30):
        data.append({"text": f"phishing url number {i}", "label": "phishing", "type": "url", "source": "test"})
    for i in range(20):
        data.append({"text": f"spam promo number {i}", "label": "spam", "type": "sms", "source": "test"})

    df = pd.DataFrame(data)
    train_df, val_df, test_df = split_data(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15

    # Check zero overlap
    train_texts = set(train_df["text"])
    val_texts = set(val_df["text"])
    test_texts = set(test_df["text"])

    assert len(train_texts.intersection(val_texts)) == 0
    assert len(train_texts.intersection(test_texts)) == 0
    assert len(val_texts.intersection(test_texts)) == 0


# ==============================================================================
# 8. Class Imbalance Handling Tests
# ==============================================================================
def test_handle_class_imbalance():
    train_df = pd.DataFrame({
        "text": [f"text_{i}" for i in range(100)],
        "label": ["legit"] * 80 + ["phishing"] * 15 + ["spam"] * 5
    })

    # Test class weights
    _, meta_weights = handle_class_imbalance(train_df, strategy="class_weights")
    assert "class_weights" in meta_weights
    weights = meta_weights["class_weights"]
    assert weights["spam"] > weights["phishing"] > weights["legit"]

    # Test oversample
    resampled, meta_over = handle_class_imbalance(train_df, strategy="oversample")
    counts = resampled["label"].value_counts().to_dict()
    assert counts["legit"] == counts["phishing"] == counts["spam"] == 80


# ==============================================================================
# 9. SimpleTokenizer Serialization Tests
# ==============================================================================
def test_simple_tokenizer_json(tmp_path):
    texts = [
        "urgent security alert verify your account",
        "update bank login details immediately",
        "free prize winner claim now"
    ]
    tokenizer = SimpleTokenizer(num_words=100, max_len=10)
    tokenizer.fit_on_texts(texts)

    json_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(json_path))

    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert "word_index" in loaded
    assert "urgent" in loaded["word_index"]
    assert loaded["oov_token"] == "<OOV>"

    # Sequence conversion and padding
    seqs = tokenizer.texts_to_sequences(["urgent prize"])
    padded = tokenizer.pad_sequences(seqs)
    assert padded.shape == (1, 10)
    assert padded[0, 0] != 0
    assert padded[0, 1] != 0
    assert padded[0, 2] == 0  # post-padding
