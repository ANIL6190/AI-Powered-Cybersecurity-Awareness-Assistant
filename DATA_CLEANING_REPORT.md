# Stage 1: Data Cleaning & Preprocessing Report
**Project:** AI-Powered Cybersecurity Awareness Assistant  
**Pipeline File:** [`src/data/preprocess.py`](file:///home/anil_a/AI-Powered-Cybersecurity-Awareness-Assistant/src/data/preprocess.py)  
**Pipeline Configuration:** [`params.yaml`](file:///home/anil_a/AI-Powered-Cybersecurity-Awareness-Assistant/params.yaml)  
**DVC Pipeline:** [`dvc.yaml`](file:///home/anil_a/AI-Powered-Cybersecurity-Awareness-Assistant/dvc.yaml) & [`dvc.lock`](file:///home/anil_a/AI-Powered-Cybersecurity-Awareness-Assistant/dvc.lock)  
**Test Suite:** [`tests/test_preprocess.py`](file:///home/anil_a/AI-Powered-Cybersecurity-Awareness-Assistant/tests/test_preprocess.py)  
**Status:** ✅ Fully Implemented, Tested, DVC-Versioned, and Reproducible  

---

## Quick Checklist & Verification

| Checklist Item | Requirement | Status | Verification Detail |
|:---|:---|:---:|:---|
| **Unified Schema & Labels** | `text \| type \| label \| source`; labels: `phishing`, `spam`, `legit` | ✅ Passed | PhiUSIIL (1=legit, 0=phish), SMS (ham=legit, spam=spam), PhishTank (phishing) |
| **Raw Profiling Report** | ydata-profiling HTML + JSON summary before changes | ✅ Passed | Saved to [`reports/raw_data_profile.html`](file:///home/anil_a/AI-Powered-Cybersecurity-Awareness-Assistant/reports/raw_data_profile.html) & [`reports/raw_data_profile.json`](file:///home/anil_a/AI-Powered-Cybersecurity-Awareness-Assistant/reports/raw_data_profile.json) |
| **Nulls & Bad Rows Removed** | Drop null text/label, text < 3 chars, corrupted/binary junk | ✅ Passed | 5 invalid short rows removed, 0 nulls remain |
| **Duplicates Removed** | Exact & near-duplicates, cross-dataset deduplication | ✅ Passed | 3,292 duplicate instances removed across entire unified corpus |
| **Text Cleaned with Tokens** | Replace `<URL>`, `<PHONE>`, `<EMAIL>`, `<AMOUNT>`, `<OTP>` | ✅ Passed | Privacy-compliant tokens, repeated chars collapsed (`freeee!!!` → `free!`), signals kept |
| **NLP Preprocessing** | NLTK word tokenize, signal-preserving stopwords, lemmatization | ✅ Passed | 7,873 unique lemmas cached; urgency signals (`urgent`, `verify`, `not`, `!`) preserved |
| **URL-Specific Features** | Lexical feature extraction, drop answer-leaking identifiers | ✅ Passed | 7 tabular features extracted; `FILENAME`, `Domain`, `Title` dropped |
| **Class Imbalance Handled** | Class weights or resampling applied to **TRAIN ONLY** | ✅ Passed | Computed balanced weights on training set: `legit: 0.7511`, `phishing: 0.6015`, `spam: 163.2049` |
| **Stratified Split** | 70/15/15 Train/Val/Test stratified split before vectorization | ✅ Passed | Train (219,837), Val (47,108), Test (47,108); 0 data leakage |
| **Feature Preparation** | TF-IDF (baseline) & Sequence Tokenizer (BiLSTM/GRU) | ✅ Passed | `tfidf_vectorizer.pkl` (4,920 vocab), `tokenizer.json` (5,465 vocab), padded `.npy` sequences |
| **Validation & DVC Versioning** | Cleaned profile report, automated PyTest suite, DVC tracked | ✅ Passed | 9/9 PyTest tests passed, `dvc repro` verified, locked in `dvc.lock` |

---

## 1. Executive Summary & Metrics Comparison

The end-to-end preprocessing pipeline orchestrates all 11 stages as modular, deterministic functions in [`src/data/preprocess.py`](file:///home/anil_a/AI-Powered-Cybersecurity-Awareness-Assistant/src/data/preprocess.py). The pipeline processes 317,350 raw samples into 314,053 pristine, normalized records with zero train/test leakage.

### Before vs. After Cleaning Statistics

| Metric | Raw Input Data | Cleaned & Processed Output | Impact / Delta |
|:---|:---:|:---:|:---|
| **Total Rows** | 317,350 | **314,053** | 3,297 rows filtered (bad rows + duplicates) |
| **SMSSpamCollection** | 5,574 | **5,160** | 414 duplicate SMS messages removed |
| **PhiUSIIL Phishing URLs** | 235,795 | **232,918** | 2,877 duplicate & short URLs removed |
| **PhishTank verified_online** | 75,981 | **75,975** | Cross-dataset duplicates with PhiUSIIL resolved |
| **Duplicate Rows** | 2,954 | **0** | Zero duplicate samples remaining |
| **Null Values in Key Fields** | 0 | **0** | Clean, complete fields |
| **Train Set Rows (70%)** | — | **219,837** | Stratified by class label |
| **Validation Set Rows (15%)** | — | **47,108** | Natural evaluation distribution |
| **Test Set Rows (15%)** | — | **47,108** | Natural evaluation distribution |
| **Train/Test Data Overlap** | — | **0** (0.00%) | **Zero data leakage guaranteed** |
| **TF-IDF Vocabulary Size** | — | **4,920** features | Fitted strictly on training set |
| **Tokenizer Vocabulary Size** | — | **5,465** words | Serialized to `tokenizer.json` |

---

## 2. Detailed Breakdown of the 11 Pipeline Stages

### Step 1: Schema Standardization (`load_and_standardize_schema`)
- Standardized all input datasets to a unified four-column schema: `text | type | label | source`.
- Fixed character encodings (UTF-8) and stripped Byte Order Marks (`\ufeff`) from CSV headers.
- Unified labels into `phishing`, `spam`, and `legit`:
  - `SMSSpamCollection`: `ham` → `legit`, `spam` → `spam` (`type: sms`, `source: SMSSpamCollection`).
  - `PhiUSIIL_Phishing_URL_Dataset.csv`: Corrected label direction where `1 = legitimate` → `legit`, `0 = phishing` → `phishing` (`type: url`, `source: PhiUSIIL`).
  - `verified_online.csv`: PhishTank verified phishing attacks → `phishing` (`type: url`, `source: PhishTank`).
  - Optional extensible support for Email datasets (`type: email`).

### Step 2: Raw Data Quality Profiling (`profile_raw_data`)
- Generated pre-cleaning profiling using `ydata-profiling` to evaluate missing values, duplicate rates, class distributions, and token lengths.
- Saved interactive visualization to [`reports/raw_data_profile.html`](file:///home/anil_a/AI-Powered-Cybersecurity-Awareness-Assistant/reports/raw_data_profile.html).
- Exported machine-readable summary to [`reports/raw_data_profile.json`](file:///home/anil_a/AI-Powered-Cybersecurity-Awareness-Assistant/reports/raw_data_profile.json) for presentation and audit logging.

### Step 3: Missing and Bad Row Handling (`handle_missing_and_bad_rows`)
- Eliminated empty strings and null entries.
- Dropped uninformative rows with text length `< 3` characters (5 invalid entries).
- Removed corrupted entries containing null bytes (`\x00`), Unicode replacement characters (`\ufffd`), or binary non-printable control characters.

### Step 4: Deduplication & Leakage Prevention (`remove_duplicates`)
- Removed exact duplicates on `text`.
- Removed near-duplicates after case-folding, whitespace trimming, and punctuation normalization.
- Enforced cross-dataset deduplication: duplicate URLs appearing in both PhiUSIIL and PhishTank were consolidated to avoid identical samples ending up in different training/testing folds.

### Step 5: Text Sanitization & Privacy Tokenization (`clean_text_corpus`)
- Stripped raw HTML tags (`<a href=...>Click</a>` → `Click`) and email header metadata (`Subject:`, `From:`, `To:`).
- Preserved privacy while maintaining classification signals through domain-specific tokens:
  - URLs: `http://bit.ly/x` → `<URL>`
  - Emails: `user@domain.com` → `<EMAIL>`
  - Phone Numbers: `+1 800-555-0199` → `<PHONE>`
  - Monetary Amounts: `$5,000`, `100 USD`, `£2000` → `<AMOUNT>`
  - OTP / Security Verification Codes: `code is 987654` → `code <OTP>`
- Collapsed character repetitions (`freeee!!!` → `free!`).
- Preserved punctuation signals (`!`, `$`) and urgency indicators.

### Step 6: NLP Preprocessing with NLTK (`nlp_preprocess`)
- Applied word tokenization (`nltk.word_tokenize`).
- Stopword filtering using NLTK English stopwords while explicitly preserving critical cybersecurity and urgency signals:
  `{'urgent', 'urgently', 'verify', 'update', 'not', 'no', 'now', 'free', 'win', 'prize', 'claim', 'alert', 'warn', 'security', 'login', 'account', 'bank', 'blocked', 'suspended', 'immediately', 'stop', 'action', 'safe', 'danger', 'risk'}`.
- WordNet lemmatization (`WordNetLemmatizer`) with in-memory memoization cache (7,873 unique lemmas cached) achieving high execution speed.

### Step 7: URL Lexical & Structural Feature Extraction (`url_specific_cleaning`)
- Normalized URLs (percent-decoding via `urllib.parse.unquote`, lowercasing domain, removing trailing slashes).
- Extracted 7 tabular features for hybrid and tree-based models:
  1. `url_length`: Total character count of normalized URL.
  2. `num_dots`: Number of dot (`.`) separators.
  3. `has_at`: Indicator for `@` in URL (phishing credential confusion).
  4. `has_ip`: Indicator for direct IPv4 addresses in the domain.
  5. `num_subdomains`: Count of subdomain hierarchy levels.
  6. `is_https`: Protocol security flag (1 for HTTPS, 0 for HTTP).
  7. `suspicious_keywords_count`: Frequency of credential harvesting keywords (`login`, `signin`, `verify`, `account`, `banking`, `secure`, `webscr`, `ebayisapi`, `password`, `wallet`, `admin`).
- Dropped identifier columns that cause target leakage: `FILENAME`, `Domain`, `Title`, `phish_detail_url`.

### Step 8: Class Imbalance Strategy (`handle_class_imbalance`)
- Post-cleaning label distribution across 314,053 samples:
  - `phishing`: 174,046 (55.4%)
  - `legit`: 139,365 (44.4%)
  - `spam`: 642 (0.2%)
- Balanced class weights computed on training set:
  - `legit`: **0.7511**
  - `phishing`: **0.6015**
  - `spam`: **163.2049**
- **Strict Isolation Rule:** Resampling and class weighting are applied **strictly to the training set only**. Validation and test folds maintain natural real-world distributions.

### Step 9: Stratified Train / Validation / Test Splitting (`split_data`)
- Split configured via [`params.yaml`](file:///home/anil_a/AI-Powered-Cybersecurity-Awareness-Assistant/params.yaml) with fixed random seed `42`:
  - **Train Set (70%):** 219,837 rows
  - **Validation Set (15%):** 47,108 rows
  - **Test Set (15%):** 47,108 rows
- Stratified by target label to maintain uniform class ratios across folds.
- Splitting was performed **before** fitting vectorizers or tokenizers. Zero text overlap between train and test/val was verified programmatically.

### Step 10: Feature Preparation (`prepare_features`)
- **Baseline Models:** Fitted `TfidfVectorizer` (max features: 5,000, n-gram range: (1, 2), min_df: 2) **only** on the training set. Serialized vectorizer to [`data/processed/tfidf_vectorizer.pkl`](file:///home/anil_a/AI-Powered-Cybersecurity-Awareness-Assistant/data/processed/tfidf_vectorizer.pkl).
- **Deep Learning Models (BiLSTM / GRU):** Fitted sequence tokenizer on training text, converted text to padded integer sequences (`max_sequence_length: 100`), and exported:
  - [`data/processed/tokenizer.json`](file:///home/anil_a/AI-Powered-Cybersecurity-Awareness-Assistant/data/processed/tokenizer.json) (5,465 vocabulary items, Keras-compatible).
  - Sequence arrays: `train_sequences.npy`, `val_sequences.npy`, `test_sequences.npy`.
  - Processed CSV splits: `train.csv`, `val.csv`, `test.csv`.

### Step 11: Validation, Re-Profiling, and DVC Versioning (`validate_and_version`)
- Automated validation checks:
  - 0 null values in `text`, `processed_text`, `label`, or `type`.
  - All labels belong to `{'phishing', 'spam', 'legit'}`.
  - Exactly 0 samples overlapping between training and test sets.
- Generated comparison report: [`reports/cleaning_comparison.json`](file:///home/anil_a/AI-Powered-Cybersecurity-Awareness-Assistant/reports/cleaning_comparison.json).
- Generated cleaned data profiling HTML: [`reports/cleaned_data_profile.html`](file:///home/anil_a/AI-Powered-Cybersecurity-Awareness-Assistant/reports/cleaned_data_profile.html).
- Registered DVC stage `preprocess` in [`dvc.yaml`](file:///home/anil_a/AI-Powered-Cybersecurity-Awareness-Assistant/dvc.yaml) with parameters tracked in [`params.yaml`](file:///home/anil_a/AI-Powered-Cybersecurity-Awareness-Assistant/params.yaml) and locked in [`dvc.lock`](file:///home/anil_a/AI-Powered-Cybersecurity-Awareness-Assistant/dvc.lock).

---

## 3. Automated Test Suite Results

The automated PyTest suite in [`tests/test_preprocess.py`](file:///home/anil_a/AI-Powered-Cybersecurity-Awareness-Assistant/tests/test_preprocess.py) tests all cleaning rules and boundary conditions:

```bash
$ python3 -m pytest tests/ -v
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /home/anil_a/AI-Powered-Cybersecurity-Awareness-Assistant
collected 9 items

tests/test_preprocess.py::test_schema_and_labels_synthetic PASSED        [ 11%]
tests/test_preprocess.py::test_handle_missing_and_bad_rows PASSED        [ 22%]
tests/test_preprocess.py::test_remove_duplicates_cross_dataset PASSED    [ 33%]
tests/test_preprocess.py::test_clean_text_single_tokens PASSED           [ 44%]
tests/test_preprocess.py::test_nlp_preprocess_signals_preserved PASSED   [ 55%]
tests/test_preprocess.py::test_extract_single_url_features PASSED        [ 66%]
tests/test_preprocess.py::test_split_data_no_leakage PASSED              [ 77%]
tests/test_preprocess.py::test_handle_class_imbalance PASSED             [ 88%]
tests/test_preprocess.py::test_simple_tokenizer_json PASSED              [100%]

============================== 9 passed in 5.76s ===============================
```

---

## 4. DVC Pipeline Execution & Reproducibility

The preprocessing stage can be reproduced at any time with:

```bash
# Run DVC pipeline reproduction
dvc repro

# Verify DVC status
dvc status
# Output: "Data and pipelines are up to date."
```

### Generated Artifacts Summary

```
├── params.yaml                             # Central configuration
├── dvc.yaml                                # DVC pipeline stage definition
├── dvc.lock                                # Checksums of dependencies & outputs
├── src/
│   └── data/
│       ├── __init__.py
│       └── preprocess.py                   # Stage 1 11-step pipeline module
├── tests/
│   └── test_preprocess.py                  # PyTest verification suite
├── reports/
│   ├── raw_data_profile.html               # Raw data visual profiling
│   ├── raw_data_profile.json               # Raw data metrics
│   ├── cleaned_data_profile.html           # Cleaned data visual profiling
│   └── cleaning_comparison.json            # Before/after comparative metrics
└── data/
    └── processed/
        ├── train.csv                       # Training fold (219,837 rows)
        ├── val.csv                         # Validation fold (47,108 rows)
        ├── test.csv                        # Test fold (47,108 rows)
        ├── train_sequences.npy             # Padded sequences for BiLSTM/GRU
        ├── val_sequences.npy               # Validation sequences
        ├── test_sequences.npy              # Test sequences
        ├── tokenizer.json                  # Keras-compatible sequence tokenizer
        └── tfidf_vectorizer.pkl            # TF-IDF baseline vectorizer
```
