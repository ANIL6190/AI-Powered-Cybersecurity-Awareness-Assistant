"""Tests for the Phase 1 validation stage: clean data passes, injected problems fail."""

import os

import pandas as pd

from src.data import validate


def _checks(splits, cfg):
    return {r["check"]: r["passed"] for r in validate.run_checks(splits, cfg).results}


def test_clean_prepared_data_passes(prepared_project):
    config_path, cfg, _ = prepared_project
    assert validate.run(config_path) is True
    assert os.path.exists(os.path.join(cfg["output"]["reports_dir"], "validation.json"))


def test_text_leakage_is_detected(prepared_project):
    _, cfg, _ = prepared_project
    splits = validate.load_splits(cfg["output"]["processed_dir"])
    splits["test"] = pd.concat([splits["test"], splits["train"].head(1)], ignore_index=True)
    results = _checks(splits, cfg)
    assert results["leakage.text.train_vs_test"] is False
    assert results["leakage.domain.train_vs_test"] is False


def test_domain_leakage_is_detected(prepared_project):
    _, cfg, _ = prepared_project
    splits = validate.load_splits(cfg["output"]["processed_dir"])
    leaked = splits["train"].head(1).copy()
    leaked["model_text"] = leaked["model_text"] + "/other-path"
    splits["shift_test"] = pd.concat([splits["shift_test"], leaked.assign(dataset="phishvn")], ignore_index=True)
    assert _checks(splits, cfg)["leakage.domain.train_vs_shift_test"] is False


def test_invalid_label_is_detected(prepared_project):
    _, cfg, _ = prepared_project
    splits = validate.load_splits(cfg["output"]["processed_dir"])
    splits["val"].loc[0, "label"] = 2
    assert _checks(splits, cfg)["val.labels_binary"] is False


def test_pre_cutoff_temporal_row_is_detected(prepared_project):
    _, cfg, _ = prepared_project
    splits = validate.load_splits(cfg["output"]["processed_dir"])
    splits["temporal_test"].loc[0, "event_date"] = "2020-01-01"
    assert _checks(splits, cfg)["temporal.all_after_cutoff"] is False


def test_dated_phiusiil_row_is_detected(prepared_project):
    _, cfg, _ = prepared_project
    splits = validate.load_splits(cfg["output"]["processed_dir"])
    splits["train"].loc[0, "event_date"] = "2023-01-01"
    assert _checks(splits, cfg)["temporal.phiusiil_has_no_dates"] is False


def test_validation_failure_returns_false(prepared_project):
    config_path, cfg, _ = prepared_project
    path = os.path.join(cfg["output"]["processed_dir"], "temporal_test.csv")
    df = pd.read_csv(path, keep_default_na=False, dtype={"event_date": str})
    df.loc[0, "event_date"] = "2019-01-01"
    df.to_csv(path, index=False)
    assert validate.run(config_path) is False
