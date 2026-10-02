# AI-Powered Cybersecurity Awareness Assistant

An end-to-end machine learning pipeline for detecting **phishing URLs**, **spam SMS**, and **email-based threats** using NLP and deep learning.

## Project Structure

```
├── src/data/preprocess.py       # Stage 1: Full data cleaning pipeline (11 steps)
├── tests/test_preprocess.py     # PyTest suite (9 tests, all passing)
├── params.yaml                  # Pipeline hyperparameters
├── dvc.yaml                     # DVC pipeline stage definition
├── dvc.lock                     # Reproducibility lock file
└── DATA_CLEANING_REPORT.md      # Full Stage 1 documentation
```

## Datasets

| Dataset | Source | Type | Rows |
|---|---|---|---|
| SMSSpamCollection | UCI ML Repository | SMS | 5,574 |
| PhiUSIIL Phishing URL Dataset | Kaggle | URL | 235,795 |
| PhishTank verified_online | PhishTank | URL | 75,981 |

## Stage 1: Data Cleaning & Preprocessing

Pipeline implemented in `src/data/preprocess.py` with 11 modular steps:

1. **Schema standardization** — unified `text | type | label | source` schema
2. **Raw data profiling** — ydata-profiling HTML + JSON evidence reports  
3. **Missing & bad row handling** — nulls, short text, corrupted binary junk
4. **Cross-dataset deduplication** — 3,292 duplicates removed, zero leakage
5. **Text sanitization** — `<URL>` `<EMAIL>` `<PHONE>` `<AMOUNT>` `<OTP>` tokens
6. **NLP preprocessing** — NLTK tokenization, signal-preserving stopwords, lemmatization
7. **URL feature extraction** — 7 tabular lexical features, leaking columns dropped
8. **Class imbalance handling** — balanced class weights on train set only
9. **Stratified split** — 70/15/15 Train/Val/Test, zero data leakage verified
10. **Feature preparation** — TF-IDF vectorizer + Keras sequence tokenizer
11. **Validation & versioning** — PyTest checks, DVC lock file

### Results

| Metric | Value |
|---|---|
| Raw rows | 317,350 |
| Clean rows | 314,053 |
| Train set | 219,837 (70%) |
| Validation set | 47,108 (15%) |
| Test set | 47,108 (15%) |
| Train/Test overlap | 0 (zero leakage) |
| TF-IDF vocab | 4,920 |
| Tokenizer vocab | 5,465 |

## Quick Start

```bash
# Install dependencies
pip install scikit-learn nltk ydata-profiling dvc pyyaml

# Download NLTK data
python3 -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

# Run the full preprocessing pipeline
dvc repro

# Or run directly
python3 -m src.data.preprocess --config params.yaml

# Run tests
python3 -m pytest tests/ -v
```

## Pipeline (DVC)

```bash
dvc repro          # Re-run pipeline (only changed stages)
dvc status         # Check if outputs are up to date
dvc dag            # Visualize pipeline DAG
```

## Labels

| Label | Meaning |
|---|---|
| `phishing` | Phishing URL or email |
| `spam` | Spam SMS or unsolicited message |
| `legit` | Legitimate / benign content |
