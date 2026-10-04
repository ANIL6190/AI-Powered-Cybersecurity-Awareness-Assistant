"""Tests for TF-IDF features, model training and evaluation metrics (synthetic data)."""

import os

import numpy as np
import pandas as pd
import pytest

from src.features import build_features
from src.models import evaluate, train


# ------------------------------------------------------------------ metrics
def test_compute_metrics_known_values():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    proba = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.2, 0.3])
    m = evaluate.compute_metrics(y, proba, threshold=0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 4)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["false_positive_rate"] == pytest.approx(1 / 5)
    assert 0 <= m["pr_auc"] <= 1


def test_recall_only_rejects_mixed_labels():
    assert evaluate.compute_recall_only(np.ones(4), np.array([0.9, 0.1, 0.6, 0.4]))["recall"] == 0.5
    with pytest.raises(ValueError):
        evaluate.compute_recall_only(np.array([1, 0]), np.array([0.9, 0.1]))


def test_recall_by_period_skips_undated_and_legitimate():
    labels = np.array([1, 1, 1, 0, 1])
    proba = np.array([0.9, 0.2, 0.8, 0.9, 0.7])
    periods = pd.Series([2022, 2022, 2024, 2024, None], dtype="Int64")
    out = evaluate.recall_by_period(labels, proba, periods, 0.5)
    assert out == {
        "2022": {"n_phishing": 2, "tp": 1, "fn": 1, "recall": 0.5, "false_negative_rate": 0.5},
        "2024": {"n_phishing": 1, "tp": 1, "fn": 0, "recall": 1.0, "false_negative_rate": 0.0},
    }


# ------------------------------------------------------------------ TF-IDF
def test_tfidf_is_fitted_on_train_only(prepared_project):
    config_path, cfg, _ = prepared_project
    build_features.run(config_path)
    import joblib

    vec = joblib.load(os.path.join(cfg["output"]["models_dir"], "tfidf_vectorizer.joblib"))
    train_text = " ".join(pd.read_csv(os.path.join(cfg["output"]["processed_dir"], "train.csv"))["model_text"])
    # Every vocabulary n-gram must occur in the training text (char_wb pads words with spaces).
    padded = " " + " ".join(f" {w} " for w in train_text.split()) + " "
    assert all(term in padded for term in vec.vocabulary_)
    # A string seen only in the temporal set contributes no new vocabulary
    assert "vn-new-scam" not in " ".join(vec.vocabulary_)
    assert vec.analyzer == "char_wb" and vec.ngram_range == (3, 5)


def test_to_text_array_accepts_tabular_input():
    df = pd.DataFrame({"url": ["a.com", "b.com/x"]})
    assert list(build_features.to_text_array(df)) == ["a.com", "b.com/x"]
    assert list(build_features.to_text_array(["a.com"])) == ["a.com"]


# ------------------------------------------------------------------ training + evaluation
@pytest.mark.parametrize("model_name", ["logreg", "xgboost"])
def test_training_smoke(prepared_project, model_name):
    config_path, cfg, _ = prepared_project
    build_features.run(config_path)
    report = train.run(config_path, model_name)
    assert os.path.exists(os.path.join(cfg["output"]["models_dir"], f"{model_name}.joblib"))
    assert 0.0 <= report["validation"]["recall"] <= 1.0
    assert report["mlflow_run_id"]


def test_evaluate_end_to_end(prepared_project):
    config_path, cfg, _ = prepared_project
    build_features.run(config_path)
    for name in ("logreg", "xgboost"):
        train.run(config_path, name)
    metrics = evaluate.run(config_path)
    for name in ("logreg", "xgboost"):
        m = metrics["models"][name]
        for key in ("test", "test_no_path_subset", "shift_test", "temporal_test", "temporal_phishing_only",
                    "phishvn_phishing_pre_cutoff"):
            assert key in m
        assert m["temporal_phishing_only"]["n_phishing"] == 10
        assert set(m["candidate_quality_gate_metrics"]) == {"phishvn_post_split_phishing_recall", "heldout_phishing_f1"}
        # Synthetic NCSC phishing: 12 dated 2022, 10 dated 2024 (the invalid-date row stays undated)
        by_year = m["phishvn_phishing_recall_by_year"]
        assert {y: v["n_phishing"] for y, v in by_year.items()} == {"2022": 12, "2024": 10}
    reports = cfg["output"]["reports_dir"]
    assert os.path.exists(os.path.join(reports, "metrics.json"))
    assert os.path.exists(os.path.join(reports, "model_comparison.md"))
    assert os.path.exists(os.path.join(reports, "plots", "pr_curve_temporal_test.png"))
